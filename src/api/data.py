"""Read-side data for the web app: experiment results, replay alerts and templates.

Every loader is cached on the source files' modification times, so re-running
training or the replay is picked up without restarting the server.
"""

import json
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd

from src.config import ARTIFACTS_DIR, DATASETS, PARSED_DIR, RESULTS_DIR, SEQUENCES_DIR, TRAIN_FRAC
from src.evaluation.metrics import alert_counts, detection_delay_minutes
from src.evaluation.report import load_results

STREAM_DIR = RESULTS_DIR / "stream"


def _mtimes(*paths: Path) -> tuple:
    return tuple(p.stat().st_mtime if p.exists() else None for p in paths)


def _records(df: pd.DataFrame) -> list[dict]:
    """DataFrame -> JSON-safe records (NaN becomes null)."""
    return json.loads(df.to_json(orient="records"))


def trained_datasets() -> list[str]:
    return [d for d in DATASETS if (ARTIFACTS_DIR / d / "classical.joblib").exists()]


# --- Overview --------------------------------------------------------------------------


def overview(dataset: str) -> dict:
    sessions = SEQUENCES_DIR / f"{dataset}_sessions.parquet"
    templates = PARSED_DIR / f"{dataset}_templates.csv"
    return _overview(dataset, _mtimes(sessions, templates))


@lru_cache(maxsize=4)
def _overview(dataset: str, _key: tuple) -> dict:
    s = pd.read_parquet(
        SEQUENCES_DIR / f"{dataset}_sessions.parquet", columns=["start_time", "end_time", "n_events", "label"]
    )
    n_templates = len(pd.read_csv(PARSED_DIR / f"{dataset}_templates.csv"))
    return {
        "dataset": dataset,
        "grouping": "block-ID sessions" if dataset == "hdfs" else "hourly time windows",
        "groups": len(s),
        "anomalous": int(s["label"].sum()),
        "anomaly_rate": float(s["label"].mean()),
        "events": int(s["n_events"].sum()),
        "templates": n_templates,
        "start_time": int(s["start_time"].min()),
        "end_time": int(s["end_time"].max()),
    }


def results(dataset: str) -> list[dict]:
    paths = [RESULTS_DIR / f"{dataset}_{f}.csv" for f in ("classical", "sequence")]
    return _results(dataset, _mtimes(*paths))


@lru_cache(maxsize=4)
def _results(dataset: str, _key: tuple) -> list[dict]:
    df = load_results(dataset)
    if df.empty:
        return []
    cols = [
        "model",
        "precision",
        "recall",
        "f1",
        "pr_auc",
        "fp",
        "tp",
        "fn",
        "false_alarms_per_day",
        "val_f1",
        "threshold",
        "fit_seconds",
    ]
    df = df[[c for c in cols if c in df]].copy()
    df["supervised"] = df["model"].str.contains("supervised")
    df["family"] = np.select(
        [df["model"].str.startswith("Keyword"), df["model"].isin(["LSTM (DeepLog-style)", "Transformer"])],
        ["rule", "sequence"],
        "count-vector",
    )
    return _records(df)


# --- Streaming replay ------------------------------------------------------------------


def _stream_paths(dataset: str) -> tuple[Path, Path]:
    return STREAM_DIR / f"{dataset}_scores.csv", STREAM_DIR / f"{dataset}_alerts.jsonl"


@lru_cache(maxsize=4)
def _stream(dataset: str, _key: tuple) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    scores_path, alerts_path = _stream_paths(dataset)
    scores = pd.read_csv(scores_path)
    with open(alerts_path, encoding="utf-8") as f:
        details = {a["session_id"]: a for a in map(json.loads, f)}
    alerts = scores[scores["alerted"].astype(bool)].copy()
    alerts["outcome"] = np.where(alerts["label"] == 1, "true_alarm", "false_alarm")
    return scores, alerts.sort_values("score", ascending=False), details


def has_stream(dataset: str) -> bool:
    return all(p.exists() for p in _stream_paths(dataset))


def _load_stream(dataset: str):
    paths = _stream_paths(dataset)
    if not all(p.exists() for p in paths):
        return None
    return _stream(dataset, _mtimes(*paths))


def stream_summary(dataset: str) -> dict | None:
    loaded = _load_stream(dataset)
    if loaded is None:
        return None
    scores, _, _ = loaded
    y, alerted = scores["label"].astype(bool), scores["alerted"].astype(bool)
    counts = alert_counts(y, alerted)
    delay = detection_delay_minutes(scores, dataset)

    bucket = 3600 if dataset == "hdfs" else 86400  # BGL spans months: bin by day
    t = (scores["start_time"] // bucket) * bucket
    timeline = (
        pd.DataFrame(
            {
                "t": t,
                "alerts": alerted.astype(int),
                "anomalous": y.astype(int),
                "groups": 1,
                "false_alarms": (alerted & ~y).astype(int),
            }
        )
        .groupby("t")
        .sum()
        .reset_index()
    )
    return {
        "groups": len(scores),
        "anomalous": int(y.sum()),
        "alerts": int(alerted.sum()),
        **counts,
        "delay_minutes": {"median": float(delay.median()), "p90": float(delay.quantile(0.9)), "max": float(delay.max())}
        if len(delay)
        else None,
        "bucket_seconds": bucket,
        "timeline": _records(timeline),
    }


def alerts(dataset: str, outcome: str | None, limit: int, offset: int) -> dict | None:
    loaded = _load_stream(dataset)
    if loaded is None:
        return None
    _, df, _ = loaded
    if outcome:
        df = df[df["outcome"] == outcome]
    page = df.iloc[offset : offset + limit][
        ["session_id", "score", "outcome", "n_events", "start_time", "end_time", "alert_time"]
    ]
    return {"total": len(df), "items": _records(page)}


def alert_detail(dataset: str, session_id: str) -> dict | None:
    loaded = _load_stream(dataset)
    if loaded is None:
        return None
    detail = loaded[2].get(session_id)
    if detail is None:
        return None
    return {**detail, "outcome": "true_alarm" if detail["label"] == 1 else "false_alarm"}


# --- Templates -------------------------------------------------------------------------


def templates(dataset: str) -> dict:
    paths = (
        PARSED_DIR / f"{dataset}_templates.csv",
        PARSED_DIR / f"{dataset}_structured.parquet",
        SEQUENCES_DIR / f"{dataset}_sessions.parquet",
    )
    return _templates(dataset, _mtimes(*paths))


@lru_cache(maxsize=4)
def _templates(dataset: str, _key: tuple) -> dict:
    inv = pd.read_csv(PARSED_DIR / f"{dataset}_templates.csv")
    parsed = pd.read_parquet(PARSED_DIR / f"{dataset}_structured.parquet", columns=["event_id", "timestamp"])
    first_seen = parsed.groupby("event_id")["timestamp"].min().rename("first_seen")
    starts = pd.read_parquet(SEQUENCES_DIR / f"{dataset}_sessions.parquet", columns=["start_time"])["start_time"]
    train_end = int(starts.sort_values().iloc[int(len(starts) * TRAIN_FRAC)])
    inv = inv.join(first_seen, on="event_id").sort_values("count", ascending=False)
    inv["share"] = inv["count"] / inv["count"].sum()
    new = inv[inv["first_seen"] >= train_end].sort_values("first_seen")
    return {
        "total": len(inv),
        "train_end": train_end,
        "new_count": len(new),
        "frequent": _records(inv.head(25)),
        "new": _records(new.head(200)),
    }
