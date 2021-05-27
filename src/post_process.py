import sys

import numpy as np

sys.path.append("./src")
import const


# ref: https://www.kaggle.com/theoviel/inference-theo?scriptVersionId=42527667
def post_process_site_12(preds, threshold=0.5, maxpreds=3):
    preds = preds * (preds >= threshold)  # remove preds < threshold

    #     next_preds = np.concatenate([preds[1:], preds[-1:]])  # pred corresponding to next window
    #     prev_preds = np.concatenate([preds[:1], preds[:-1]])  # pred corresponding to previous window

    next_preds = np.concatenate(
        [preds[1:], np.zeros((1, preds.shape[-1]))]
    )  # pred corresponding to next window
    prev_preds = np.concatenate(
        [np.zeros((1, preds.shape[-1])), preds[:-1]]
    )  # pred corresponding to previous window

    score = preds + next_preds + prev_preds  # Aggregating

    n_birds = (score >= threshold - 1e-5).sum(-1)  # threshold ?
    n_birds = np.clip(n_birds, 0, maxpreds)  # keep at most maxpreds birds

    labels = [np.argsort(-score[i])[: n_birds[i]].tolist() for i in range(len(preds))]
    #     class_labels = [" ".join([CLASSES[l] for l in label]) for label in labels]
    class_labels = [
        " ".join([const.INV_BIRD_CODE[l] for l in label])
        if len(label) > 0
        else "nocall"
        for label in labels
    ]

    return class_labels


def post_process_arranged(preds, threshold=0.5, maxpreds=3):
    preds = preds * (preds >= threshold)  # remove preds < threshold

    #     next_preds = np.concatenate([preds[1:], preds[-1:]])  # pred corresponding to next window
    #     prev_preds = np.concatenate([preds[:1], preds[:-1]])  # pred corresponding to previous window

    next1_preds = np.concatenate(
        [preds[1:], np.zeros((1, preds.shape[-1]))]
    )  # pred corresponding to next window
    prev1_preds = np.concatenate(
        [np.zeros((1, preds.shape[-1])), preds[:-1]]
    )  # pred corresponding to previous window

    next2_preds = np.concatenate(
        [preds[2:], np.zeros((2, preds.shape[-1]))]
    )  # pred corresponding to next window
    prev2_preds = np.concatenate(
        [np.zeros((2, preds.shape[-1])), preds[:-2]]
    )  # pred corresponding to previous window

    score = (
        (next2_preds * 0.25)
        + (next1_preds * 0.75)
        + preds
        + (prev1_preds * 0.75)
        + (prev2_preds * 0.25)
    )  # Aggregating

    n_birds = (score >= threshold - 1e-5).sum(-1)  # threshold ?
    n_birds = np.clip(n_birds, 0, maxpreds)  # keep at most maxpreds birds

    labels = [np.argsort(-score[i])[: n_birds[i]].tolist() for i in range(len(preds))]
    #     class_labels = [" ".join([CLASSES[l] for l in label]) for label in labels]
    class_labels = [
        " ".join([const.INV_BIRD_CODE[l] for l in label])
        if len(label) > 0
        else "nocall"
        for label in labels
    ]

    return class_labels
