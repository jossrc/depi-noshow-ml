"""Métricas binarias. PR-AUC se identifica como average precision."""

import numpy as np
from sklearn.metrics import (average_precision_score, brier_score_loss, confusion_matrix,
                             f1_score, precision_score, recall_score, roc_auc_score)


def binary_metrics(y, probability, threshold=.5):
    y, probability = np.asarray(y), np.asarray(probability)
    predicted = probability >= threshold
    two_classes = len(np.unique(y)) == 2
    return {
        "rows": len(y), "no_show": int(y.sum()), "prevalence": float(y.mean()),
        "roc_auc": float(roc_auc_score(y, probability)) if two_classes else None,
        "pr_auc_average_precision": float(average_precision_score(y, probability)) if two_classes else None,
        "precision": float(precision_score(y, predicted, zero_division=0)),
        "recall": float(recall_score(y, predicted, zero_division=0)),
        "f1": float(f1_score(y, predicted, zero_division=0)),
        "brier_score": float(brier_score_loss(y, probability)),
        "confusion_matrix": confusion_matrix(y, predicted, labels=[0, 1]).tolist(),
        "confusion_matrix_order": [["TN", "FP"], ["FN", "TP"]],
        "threshold": threshold, "threshold_status": "diagnostic_only; no operational risk bands validated",
    }
