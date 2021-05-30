import gc
import time
import logging
import dataclasses
from typing import Dict

import numpy as np
import pandas as pd
import torch
from torch.autograd import detect_anomaly
from fastprogress import master_bar, progress_bar
import matplotlib.pyplot as plt

from .factory import (
    get_dataloader,
    get_nn_model,
    get_loss,
    get_optim,
    get_scheduler,
    get_metrics,
)
from src import const


@dataclasses.dataclass
class NNTrainer:
    cfg: Dict
    run_name: str
    multi_gpu: bool = False
    debug: bool = False

    def __post_init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def train(
        self,
        train_df: pd.DataFrame,
        valid_df: pd.DataFrame,
    ) -> float:
        cv = 0
        self.oof = np.zeros((len(valid_df), self.cfg.model.n_classes))

        use_cv_idx = valid_df[
            ~valid_df["audio_id"].isin([7019, 7954, 14473])
        ].index.values
        valid_df = valid_df.iloc[use_cv_idx].reset_index(drop=True)

        print(
            "\n========================== FOLD 1 / 1 ... ==========================\n"
        )
        logging.debug(
            "\n========================== FOLD 1 / 1 ... ==========================\n"
        )
        if self.debug:
            print("Debug Mode. Only train on first 100 batches.")
            logging.debug("Debug Mode. Only train on first 100 batches.")

        train_loader = get_dataloader(train_df, self.cfg.data.train)
        valid_loader = get_dataloader(valid_df, self.cfg.data.valid)

        model = get_nn_model(self.cfg.model, multi_gpu=self.multi_gpu).to(self.device)
        criterion = get_loss(self.cfg)
        optimizer = get_optim(self.cfg, model.parameters())
        scheduler = get_scheduler(self.cfg, optimizer)

        best_epoch = -1
        best_val_score = -np.inf
        scorer = get_metrics(self.cfg.metrics.name)
        mb = master_bar(range(self.cfg.epochs))

        train_loss_list = []
        val_loss_list = []
        val_score_list = []

        for epoch in mb:
            start_time = time.time()

            with detect_anomaly():
                model, avg_loss = self._train_epoch(
                    model, train_loader, criterion, optimizer, mb
                )

            valid_preds, avg_val_loss = self._val_epoch(model, valid_loader, criterion)

            val_y = valid_df["birds"].apply(lambda x: 1 if x != "nocall" else 0).values
            val_score = scorer(val_y, valid_preds)

            train_loss_list.append(avg_loss)
            val_loss_list.append(avg_val_loss)
            val_score_list.append(val_score)

            if self.cfg.scheduler.name != "ReduceLROnPlateau":
                scheduler.step()
            elif self.cfg.scheduler.name == "ReduceLROnPlateau":
                scheduler.step(avg_val_loss)

            elapsed = time.time() - start_time
            mb.write(
                f"Epoch {epoch+1} - avg_train_loss: {avg_loss:.6f}  avg_val_loss: {avg_val_loss:.6f} val_score: {val_score:.6f} time: {elapsed:.0f}s"
            )
            logging.debug(
                f"Epoch {epoch+1} - avg_train_loss: {avg_loss:.6f}  avg_val_loss: {avg_val_loss:.6f} val_score: {val_score:.6f} time: {elapsed:.0f}s"
            )

            if val_score > best_val_score:
                best_epoch = epoch + 1
                best_val_score = val_score
                best_valid_preds = valid_preds
                if self.multi_gpu:
                    best_model = model.module.state_dict()
                else:
                    best_model = model.state_dict()

                torch.save(
                    best_model, const.LOG_DIR / f"{self.run_name}/weight_best.pt"
                )

        self.oof = best_valid_preds
        cv += best_val_score

        print(f"\nEpoch {best_epoch} - val_score: {best_val_score:.6f}")
        logging.debug(f"\nEpoch {best_epoch} - val_score: {best_val_score:.6f}")

        print("\n\n===================================\n")
        print(f"CV: {cv:.6f}")
        logging.debug(f"\n\nCV: {cv:.6f}")
        print("\n===================================\n\n")

        return cv

    def save(self) -> None:
        log_dir = const.LOG_DIR / self.run_name
        np.save(log_dir / "oof.npy", self.oof)

    def _train_epoch(self, model, train_loader, criterion, optimizer, mb):
        model.train()
        avg_loss = 0.0

        for i, (images, targets) in enumerate(progress_bar(train_loader, parent=mb)):
            images = images.to(self.device)
            targets = targets.to(self.device)

            r = np.random.rand()
            if self.cfg.data.train.mixup and r < 0.5:
                feats, targets = mixup(images, targets, 1.0, normal=False)

            if self.debug and i + 1 == 100:
                break

            preds = model(images)
            loss = criterion(preds, targets)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            avg_loss += loss.item() / len(train_loader)

        del images, targets
        gc.collect()

        return model, avg_loss

    def _val_epoch(self, model, valid_loader, criterion):
        model.eval()
        valid_preds = np.zeros(
            (
                len(valid_loader.dataset),
                self.cfg.model.n_classes * self.cfg.data.valid.tta.iter_num,
            )
        )
        valid_preds_tta = np.zeros(
            (len(valid_loader.dataset), self.cfg.model.n_classes)
        )

        avg_val_loss = 0.0

        with torch.no_grad():
            for t in range(self.cfg.data.valid.tta.iter_num):
                for i, (images, targets) in enumerate(valid_loader):
                    images = images.to(self.device)
                    targets = targets.to(self.device)

                    preds = model(images, is_train=False)
                    loss = criterion(preds, targets)

                    start_batch_idx = i * valid_loader.batch_size
                    end_batch_idx = (i + 1) * valid_loader.batch_size

                    valid_preds[start_batch_idx:end_batch_idx, :] = (
                        preds.sigmoid().cpu().detach().numpy()
                    )

                    avg_val_loss += loss.item() / len(valid_loader)

        for i in range(self.cfg.model.n_classes):
            preds_col_idx = [
                i + self.cfg.model.n_classes * j
                for j in range(self.cfg.data.valid.tta.iter_num)
            ]
            valid_preds_tta[:, i] = np.mean(
                valid_preds[:, preds_col_idx], axis=1
            ).reshape(-1)

        return valid_preds_tta, avg_val_loss

    def _predict_fold(self, fold_num: int, test_df: pd.DataFrame):
        test_loader = get_dataloader(test_df, self.cfg.data.test)

        model = get_nn_model(self.cfg, is_train=False).to(self.device)
        model.load_state_dict(
            torch.load(const.LOG_DIR / self.run_name / f"weight_best_{fold_num}.pt")
        )

        test_fold_preds = np.zeros((len(test_df), self.cfg.model.n_classes))

        model.eval()
        with torch.no_grad():
            for i, feats in enumerate(test_loader):
                if type(feats) == dict:
                    for k, v in feats.items():
                        feats[k] = v.to(self.device)
                else:
                    feats = feats.to(self.device)

                preds = model(feats)
                preds = preds.cpu().detach().numpy()

                start_batch_idx = i * test_loader.batch_size
                end_batch_idx = (i + 1) * test_loader.batch_size
                test_fold_preds[start_batch_idx:end_batch_idx, :] = preds

        return test_fold_preds

    def _save_loss_png(self, train_loss_list, val_loss_list, val_score_list, fold_num):
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 5))

        ax1.plot(
            range(len(train_loss_list)),
            train_loss_list,
            color="blue",
            linestyle="-",
            label="train_loss",
        )
        ax1.plot(
            range(len(val_loss_list)),
            val_loss_list,
            color="green",
            linestyle="-",
            label="val_loss",
        )
        ax1.legend()
        ax1.set_xlabel("epochs")
        ax1.set_ylabel("loss")
        ax1.set_title(f"Training and validation {self.cfg.loss.name}")
        ax1.grid()

        ax2.plot(
            range(len(val_score_list)),
            val_score_list,
            color="blue",
            linestyle="-",
            label="val_score",
        )
        ax2.legend()
        ax2.set_xlabel("epochs")
        ax2.set_ylabel("score")
        ax2.set_title("Training and validation score")
        ax2.grid()

        plt.savefig(const.LOG_DIR / self.run_name / f"learning_curve_{fold_num}.png")


def mixup(images, targets, alpha, normal=True):
    indices = torch.randperm(images.size(0))
    shuffled_images = images[indices]
    shuffled_targets = targets[indices]

    lam = np.random.beta(alpha, alpha)
    images = images * lam + shuffled_images * (1 - lam)

    if normal:
        targets = targets * lam + shuffled_targets * (1 - lam)
    else:
        targets = torch.clamp(targets + shuffled_targets, 0, 1)

    return images, targets
