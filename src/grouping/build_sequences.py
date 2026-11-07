"""Group parsed events into sessions (HDFS block IDs) or time windows (BGL).

    python -m src.grouping.build_sequences --dataset hdfs
    python -m src.grouping.build_sequences --dataset bgl --window-minutes 60 --step-minutes 60

Output: data/sequences/<dataset>_sessions.parquet with one row per group:
    session_id, start_time, end_time, n_events, keyword_hits, label, events (ordered event IDs)
"""

import argparse

import numpy as np
import pandas as pd

from src.config import DATASETS, PARSED_DIR, RAW_DIR, SEQUENCES_DIR


def _session_frame(session_id, start, end, events, keyword_hits, label) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "session_id": session_id,
            "start_time": start,
            "end_time": end,
            "n_events": np.fromiter(map(len, events), dtype=np.int64, count=len(events)),
            "keyword_hits": keyword_hits,
            "label": label,
            "events": events,
        }
    )


def build_hdfs_sessions(parsed: pd.DataFrame, labels: pd.Series) -> pd.DataFrame:
    """parsed: line_id, timestamp, event_id, keyword, block_id. labels: 0/1 indexed by block_id."""
    df = parsed.sort_values(["block_id", "line_id"], kind="stable")
    keys = df["block_id"].to_numpy()
    starts = np.flatnonzero(np.r_[True, keys[1:] != keys[:-1]])
    ends = np.r_[starts[1:], len(keys)]

    ev = df["event_id"].to_numpy()
    ts = df["timestamp"].to_numpy()
    session_labels = labels.reindex(keys[starts])
    missing = session_labels.isna().to_numpy()
    if missing.any():
        print(f"Dropping {missing.sum()} sessions with no label")

    out = _session_frame(
        session_id=[f"blk_{k}" for k in keys[starts]],
        start=np.minimum.reduceat(ts, starts),
        end=np.maximum.reduceat(ts, starts),
        events=[ev[s:e].tolist() for s, e in zip(starts, ends, strict=True)],
        keyword_hits=np.add.reduceat(df["keyword"].to_numpy(dtype=np.int64), starts),
        label=session_labels.fillna(-1).to_numpy(dtype=np.int8),
    )
    return out[~missing].sort_values("start_time", kind="stable").reset_index(drop=True)


def build_time_windows(parsed: pd.DataFrame, window_seconds: int, step_seconds: int) -> pd.DataFrame:
    """parsed: line_id, timestamp, event_id, keyword, label. A window is anomalous if any line is."""
    df = parsed.sort_values(["timestamp", "line_id"], kind="stable")
    ts = df["timestamp"].to_numpy()
    ev = df["event_id"].to_numpy()

    win_start = np.arange(ts[0], ts[-1] + 1, step_seconds)
    lo = np.searchsorted(ts, win_start, side="left")
    hi = np.searchsorted(ts, win_start + window_seconds, side="left")
    keep = hi > lo
    win_start, lo, hi = win_start[keep], lo[keep], hi[keep]

    def window_sum(values: np.ndarray) -> np.ndarray:
        cs = np.r_[0, np.cumsum(values, dtype=np.int64)]
        return cs[hi] - cs[lo]

    return _session_frame(
        session_id=[f"w{int(t)}" for t in win_start],
        start=win_start,
        end=win_start + window_seconds,
        events=[ev[a:b].tolist() for a, b in zip(lo, hi, strict=True)],
        keyword_hits=window_sum(df["keyword"].to_numpy(dtype=np.int64)),
        label=(window_sum(df["label"].to_numpy(dtype=np.int64)) > 0).astype(np.int8),
    )


def load_hdfs_labels() -> pd.Series:
    matches = sorted((RAW_DIR / "hdfs").rglob("anomaly_label.csv"))
    if not matches:
        raise FileNotFoundError(f"anomaly_label.csv not found under {RAW_DIR / 'hdfs'}")
    raw = pd.read_csv(matches[0])
    block_ids = raw["BlockId"].str.slice(4).astype(np.int64)
    return pd.Series((raw["Label"] == "Anomaly").astype(np.int8).to_numpy(), index=block_ids)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", choices=DATASETS, required=True)
    ap.add_argument("--window-minutes", type=int, default=60, help="BGL window length")
    ap.add_argument("--step-minutes", type=int, default=60, help="BGL window step (< length = sliding)")
    args = ap.parse_args()

    parsed = pd.read_parquet(PARSED_DIR / f"{args.dataset}_structured.parquet")
    if args.dataset == "hdfs":
        sessions = build_hdfs_sessions(parsed, load_hdfs_labels())
    else:
        sessions = build_time_windows(parsed.drop(columns="node"), args.window_minutes * 60, args.step_minutes * 60)

    SEQUENCES_DIR.mkdir(parents=True, exist_ok=True)
    out_path = SEQUENCES_DIR / f"{args.dataset}_sessions.parquet"
    sessions.to_parquet(out_path, index=False)
    print(
        f"{len(sessions):,} groups, {sessions['label'].mean():.2%} anomalous, "
        f"median length {sessions['n_events'].median():.0f} -> {out_path}"
    )


if __name__ == "__main__":
    main()
