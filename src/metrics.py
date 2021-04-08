from typing import List

import numpy as np
from sklearn import metrics


def f1_score(y_true: np.array, y_pred: np.array, average: str = "micro") -> float:
    return metrics.f1_score(y_true, y_pred, average="micro")


def row_wise_micro_averaged_f1_score(y_true: List[str], y_pred: List[str]) -> float:
    n_rows = len(y_true)
    f1_score = 0.0
    for true_row, predicted_row in zip(y_true, y_pred):
        f1_score += micro_f1_similarity(true_row, predicted_row) / n_rows
    return f1_score


def micro_f1_similarity(y_true: str, y_pred: str) -> float:
    true_labels = y_true.split()
    pred_labels = y_pred.split()

    true_pos, false_pos, false_neg = 0, 0, 0

    for true_elem in true_labels:
        if true_elem in pred_labels:
            true_pos += 1
        else:
            false_neg += 1

    for pred_el in pred_labels:
        if pred_el not in true_labels:
            false_pos += 1

    f1_similarity = 2 * true_pos / (2 * true_pos + false_neg + false_pos)

    return f1_similarity
