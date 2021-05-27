import os
import re
import argparse
import datetime
import logging
import warnings

import pandas as pd
from pathlib import Path

from .trainer import NNTrainer
from .factory import get_fold, get_drop_idx
from src import const
from src.utils import DataHandler, Timer, seed_everything, Notificator  # , Kaggle

warnings.filterwarnings("ignore")

# ===============
# Settings
# ===============
parser = argparse.ArgumentParser()
parser.add_argument("--notify", default=const.CONFIG_DIR / "notify.yml")
parser.add_argument("--debug", action="store_true")
parser.add_argument("-m", "--multigpu", action="store_true")
parser.add_argument("-c", "--comment")
options = parser.parse_args()

exp_dir = Path(__file__).resolve().parents[0]
exp_id = str(exp_dir).split("/")[-1]

dh = DataHandler()
cfg = dh.load(exp_dir / "config.yml")

notify_params = dh.load(options.notify)

comment = options.comment
model_name = cfg.model.backbone
now = datetime.datetime.now()
run_name = f"{exp_id}_{now:%Y%m%d%H%M%S}"

logger_path = Path(const.LOG_DIR / run_name)


# ===============
# Main
# ===============
def main():
    t = Timer()
    seed_everything(cfg.seed)

    logger_path.mkdir(exist_ok=True)
    logging.basicConfig(filename=logger_path / "train.log", level=logging.DEBUG)

    dh.save(logger_path / "config.yml", cfg)

    with t.timer("load data"):
        train_df = dh.load(const.INPUT_DATA_DIR / "train_metadata.csv")
        valid_df = dh.load(const.INPUT_DATA_DIR / "train_soundscape_labels.csv")

    with t.timer("preprocess"):
        train_df["target"] = train_df["primary_label"].map(const.BIRD_CODE)

        valid_df["file_id"] = valid_df["audio_id"].astype(str) + "_" + valid_df["site"]
        valid_df["target"] = valid_df["birds"].map(const.BIRD_CODE)
        valid_df["file_name"] = valid_df["file_id"].map(const.TRAIN_SOUNDSCAPES_CODE)

        whole_label_df = pd.concat(
            [
                train_df[["primary_label"]],
                valid_df[["birds"]].rename(columns={"birds": "primary_label"}),
            ],
            axis=0,
            ignore_index=True,
        )
        whole_label_df["target"] = whole_label_df["primary_label"].map(const.BIRD_CODE)
        whole_target_df = pd.get_dummies(whole_label_df["target"])
        target_df = whole_target_df.iloc[: len(train_df)]
        valid_target_df = whole_target_df.iloc[len(train_df) :].reset_index(drop=True)

    with t.timer("drop rows"):
        if cfg.drop is not None:
            drop_idx = get_drop_idx(cfg.drop)
            train_df = train_df.drop(drop_idx, axis=0).reset_index(drop=True)
            target_df = target_df.drop(drop_idx, axis=0).reset_index(drop=True)

    with t.timer("train model"):
        trainer = NNTrainer(cfg, run_name, options.multigpu, options.debug)
        cv = trainer.train(
            train_df=train_df,
            target_df=target_df,
            valid_df=valid_df,
            valid_target_df=valid_target_df,
        )
        trainer.save()

        run_name_cv = f"{run_name}_{cv:.3f}"
        logger_path.rename(const.LOG_DIR / run_name_cv)
        logging.disable(logging.FATAL)

    with t.timer("notify"):
        process_minutes = t.get_processing_time()
        notificator = Notificator(
            run_name=run_name_cv,
            model_name=cfg.model.backbone,
            cv=round(cv, 4),
            process_time=round(process_minutes, 2),
            comment=comment,
            params=notify_params,
        )
        notificator.send_line()
        notificator.send_notion()


if __name__ == "__main__":
    main()
