"""Detection metrics. Accuracy is deliberately absent: with ~97% normal sessions,
always predicting "normal" already scores ~97%."""
import numpy as np
from sklearn.metrics import average_precision_score, precision_recall_curve


def best_f1_threshold(y_true: np.ndarray, scores: np.ndarray) -> float:
    """Threshold that maximises F1, chosen on validation data and then frozen for test."""
    if y_true.sum() == 0:
        return float(np.max(scores)) + 1e-9  # no positives to tune on: flag nothing
    precision, recall, thresholds = precision_recall_curve(y_true, scores)
    f1 = np.divide(2 * precision * recall, precision + recall,
                   out=np.zeros_like(precision), where=(precision + recall) > 0)
    return float(thresholds[np.argmax(f1[:-1])])


def evaluate(y_true: np.ndarray, scores: np.ndarray, threshold: float, period_days: float) -> dict:
    y_true = np.asarray(y_true).astype(bool)
    pred = np.asarray(scores) >= threshold
    tp = int(np.sum(pred & y_true))
    fp = int(np.sum(pred & ~y_true))
    fn = int(np.sum(~pred & y_true))
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "pr_auc": float(average_precision_score(y_true, scores)) if y_true.any() else float("nan"),
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "false_alarms_per_day": fp / period_days if period_days > 0 else float("nan"),
    }
