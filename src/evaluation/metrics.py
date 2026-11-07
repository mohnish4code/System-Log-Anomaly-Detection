"""Detection metrics. Accuracy is deliberately absent: with ~97% normal sessions,
always predicting "normal" already scores ~97%."""

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, precision_recall_curve


def alert_counts(y_true, alerted) -> dict:
    """TP / FP / FN with precision, recall and F1 for a set of binary alerts.

    precision is None only when nothing was flagged, recall is None only when there are
    no positives, and F1 is None when either is undefined. A defined but zero precision
    or recall (tp == 0) gives 0.0, never None.
    """
    y_true = np.asarray(y_true).astype(bool)
    alerted = np.asarray(alerted).astype(bool)
    tp = int(np.sum(alerted & y_true))
    fp = int(np.sum(alerted & ~y_true))
    fn = int(np.sum(~alerted & y_true))
    precision = tp / (tp + fp) if tp + fp else None
    recall = tp / (tp + fn) if tp + fn else None
    if precision is None or recall is None:
        f1 = None
    elif tp == 0:
        f1 = 0.0
    else:
        f1 = 2 * precision * recall / (precision + recall)
    return {"tp": tp, "fp": fp, "fn": fn, "precision": precision, "recall": recall, "f1": f1}


def detection_delay_minutes(scores: pd.DataFrame, dataset: str) -> pd.Series:
    """Minutes from when an anomaly began to its alert, for correctly alerted groups.

    Groups already running when scoring began (started_live False) are excluded: their
    start is unknown to the detector, so the delay would not be meaningful. For BGL the
    anomaly begins at the first labelled alert line; HDFS labels are per session, so a
    session's first event is used.
    """
    hits = scores[scores["alerted"].astype(bool) & scores["label"].astype(bool)]
    if "started_live" in hits:
        hits = hits[hits["started_live"].astype(bool)]
    if dataset == "bgl" and "first_anomaly_time" in hits:
        began = hits["first_anomaly_time"].fillna(hits["start_time"])
    else:
        began = hits["start_time"]
    return (hits["alert_time"] - began) / 60


def best_f1_threshold(y_true: np.ndarray, scores: np.ndarray) -> float:
    """Threshold that maximises F1, chosen on validation data and then frozen for test."""
    if y_true.sum() == 0:
        return float(np.max(scores)) + 1e-9  # no positives to tune on: flag nothing
    precision, recall, thresholds = precision_recall_curve(y_true, scores)
    f1 = np.divide(
        2 * precision * recall, precision + recall, out=np.zeros_like(precision), where=(precision + recall) > 0
    )
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
