"""Metrics for failure class 1; no test evaluation is performed here."""
from sklearn.metrics import (
    precision_score, recall_score, f1_score, roc_auc_score,
    average_precision_score, accuracy_score, confusion_matrix,
)


def score(actual, predicted, probability):
    tn, fp, fn, tp = confusion_matrix(actual, predicted, labels=[0, 1]).ravel()
    return {
        "precision": precision_score(actual, predicted, zero_division=0),
        "recall": recall_score(actual, predicted, zero_division=0),
        "f1": f1_score(actual, predicted, zero_division=0),
        "roc_auc": roc_auc_score(actual, probability),
        "average_precision": average_precision_score(actual, probability),
        "accuracy": accuracy_score(actual, predicted),
        "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
    }
