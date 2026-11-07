"""Replay a historical log line by line, as a live system would emit it, and raise alerts.

    python -m src.streaming.replay --dataset hdfs --start-fraction 0.8 --max-lines 300000
    python -m src.streaming.replay --dataset bgl --start-fraction 0.8

Scoring starts --start-fraction of the way into the file (0.8 ≈ the test period).
Lines before that, from --warmup-fraction on, are read silently first, as if the
detector had been running all along: without warm-up, an HDFS block created earlier
is seen only from its tail (e.g. just deletions), looks abnormal, and floods alerts.

Lines are parsed online with Drain and grouped as they arrive.
  HDFS: a block session is scored each time it goes quiet for --idle-minutes of log
        time, and again at the end of the stream. It alerts the first time its score
        crosses the threshold, so a long-lived session can be caught before it finishes.
  BGL:  a time window is scored when the log clock passes its end.
Detection delay = alert time (log clock) - when the anomaly began: the first alert
line for BGL, the session's first event for HDFS (HDFS labels are per session).

Outputs results/stream/<dataset>_scores.csv (every group) and <dataset>_alerts.jsonl.
"""

import argparse
import json
import time

import numpy as np
import pandas as pd
from tqdm import tqdm

from src.config import DATASETS, RESULTS_DIR
from src.evaluation.metrics import alert_counts, detection_delay_minutes
from src.grouping.build_sequences import load_hdfs_labels
from src.parsing.log_formats import LINE_PARSERS
from src.parsing.parse_logs import find_raw_log
from src.scoring import Scorer, Session, session_keys

STREAM_DIR = RESULTS_DIR / "stream"


class Replay:
    def __init__(
        self,
        scorer: Scorer,
        window_seconds: int,
        idle_seconds: int,
        labels: pd.Series | None,
        evict_seconds: int | None = None,
    ):
        """evict_seconds (HDFS): drop a session from memory once it has no unscored events and
        has been quiet this long (default 10 x idle_seconds). Its scored record is kept."""
        self.scorer, self.window_seconds, self.idle_seconds = scorer, window_seconds, idle_seconds
        self.evict_seconds = 10 * idle_seconds if evict_seconds is None else evict_seconds
        self.labels = labels
        self.open: dict[str, Session] = {}
        self.dirty: set[str] = set()  # sessions with events not yet scored
        self.records: dict[str, dict] = {}
        self.alerts: list[dict] = []

    def _score(self, keys: list[str], now: int) -> None:
        sessions = [self.open[k] for k in keys]
        for s, score in zip(sessions, self.scorer.score([s.events for s in sessions]), strict=True):
            rec = self.records.setdefault(s.session_id, {"session_id": s.session_id, "alert_time": None})
            label = s.label if self.labels is None else int(self.labels.get(int(s.session_id[4:]), 0))
            rec.update(
                start_time=s.start_time,
                end_time=s.end_time,
                n_events=len(s.events),
                score=float(score),
                label=label,
                first_anomaly_time=s.first_anomaly_time,
            )
            if score >= self.scorer.threshold and rec["alert_time"] is None:
                rec["alert_time"] = now
                self.alerts.append({**rec, "top_events": self.scorer.explain(s.events), "raw_lines": list(s.raw_lines)})
        self.dirty.difference_update(keys)

    def add(self, key: str, event_id: int, rec, raw: str, live: bool = True) -> None:
        """live=False is warm-up: the event builds session history but nothing is scored."""
        self.open.setdefault(key, Session(key)).add(event_id, rec, raw, max_raw=50 if live else 0)
        if live:
            self.dirty.add(key)

    def checkpoint(self, now: int) -> None:
        if self.scorer.dataset == "bgl":
            due = [k for k in self.dirty if int(k[1:]) + self.window_seconds <= now]
            self._score(due, now)
            for k in [k for k in self.open if int(k[1:]) + self.window_seconds <= now]:
                del self.open[k]  # a closed window never receives more lines
        else:
            due = [k for k in self.dirty if self.open[k].end_time + self.idle_seconds <= now]
            self._score(due, now)
            # Without eviction every block ever seen stays in memory for the whole replay.
            # A session quiet for evict_seconds is treated as finished (warm-up-only ones too).
            stale = [k for k, s in self.open.items() if k not in self.dirty and s.end_time + self.evict_seconds <= now]
            for k in stale:
                del self.open[k]

    def finish(self, now: int) -> None:
        self._score(sorted(self.dirty), now)


def _first_timestamp_after(log_path, byte_offset: int, parse_line) -> int | None:
    """Timestamp of the first parseable line after byte_offset, or None if there is none."""
    with open(log_path, encoding="utf-8", errors="replace") as f:
        f.seek(byte_offset)
        if byte_offset:
            f.readline()  # skip the partial line we landed in
        for raw in f:
            if (rec := parse_line(raw)) is not None:
                return rec.timestamp
    return None


def run(
    dataset: str,
    model: str | None,
    start_fraction: float,
    warmup_fraction: float,
    max_lines: int | None,
    window_minutes: int,
    idle_minutes: int,
    speed: float,
) -> pd.DataFrame:
    if not 0 <= warmup_fraction <= start_fraction < 1:
        raise ValueError("need 0 <= warmup_fraction <= start_fraction < 1")
    scorer = Scorer(dataset, model)
    print(f"Model: {scorer.model_name} (threshold {scorer.threshold:.4g})")
    labels = load_hdfs_labels() if dataset == "hdfs" else None
    replay = Replay(scorer, window_minutes * 60, idle_minutes * 60, labels)
    parse_line = LINE_PARSERS[dataset]

    log_path = find_raw_log(dataset)
    size = log_path.stat().st_size
    live_from = _first_timestamp_after(log_path, int(size * start_fraction), parse_line)
    if live_from is None:
        print(f"Nothing to replay: {log_path} has no parseable lines after --start-fraction {start_fraction}.")
        return pd.DataFrame()
    warmup_byte = int(size * warmup_fraction)
    n_live = 0
    last_check, last_ts, wall_start, log_start = None, None, time.perf_counter(), None
    with open(log_path, encoding="utf-8", errors="replace") as f:
        if warmup_byte:
            f.seek(warmup_byte)
            f.readline()
        for raw in tqdm(f, unit=" lines", mininterval=2):
            rec = parse_line(raw)
            if rec is None:
                continue
            live = rec.timestamp >= live_from
            if live:
                n_live += 1
                if max_lines is not None and n_live > max_lines:
                    break
            if speed > 0 and live:
                log_start = log_start or rec.timestamp
                ahead = (rec.timestamp - log_start) / speed - (time.perf_counter() - wall_start)
                if ahead > 0:
                    time.sleep(ahead)
            event_id = scorer.parser.parse(rec.content)  # online: new templates are learned as they appear
            for key in session_keys(dataset, rec, replay.window_seconds):
                replay.add(key, event_id, rec, raw, live)
            last_ts = rec.timestamp if last_ts is None else max(last_ts, rec.timestamp)
            if last_check is None or last_ts - last_check >= 60:
                replay.checkpoint(last_ts)
                last_check = last_ts
    if last_ts is not None:
        replay.finish(last_ts)
    if not replay.records:
        print("Nothing was scored: no parseable log lines in the replay range (empty input or --max-lines 0).")
        return pd.DataFrame()

    STREAM_DIR.mkdir(parents=True, exist_ok=True)
    scores = pd.DataFrame(replay.records.values()).sort_values("start_time")
    scores["alerted"] = scores["alert_time"].notna()
    # Sessions already running when scoring began: their detection delay is not meaningful.
    scores["started_live"] = scores["start_time"] >= live_from
    scores.to_csv(STREAM_DIR / f"{dataset}_scores.csv", index=False)
    with open(STREAM_DIR / f"{dataset}_alerts.jsonl", "w", encoding="utf-8") as out:
        for alert in replay.alerts:
            out.write(json.dumps(alert, default=int) + "\n")
    summarize(scores, dataset)
    return scores


def _fmt(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.3f}"


def summarize(scores: pd.DataFrame, dataset: str) -> dict:
    y, alerted = scores["label"].astype(bool), scores["alerted"].astype(bool)
    m = alert_counts(y, alerted)
    print(f"\n{len(scores):,} groups, {int(y.sum()):,} anomalous, {int(alerted.sum()):,} alerts")
    print(
        f"precision {_fmt(m['precision'])}  recall {_fmt(m['recall'])}  F1 {_fmt(m['f1'])}  "
        f"(TP {m['tp']}, FP {m['fp']}, FN {m['fn']})"
    )

    delay_min = detection_delay_minutes(scores, dataset)
    if len(delay_min):
        print(
            f"detection delay (minutes): median {np.median(delay_min):.1f}, "
            f"p90 {np.percentile(delay_min, 90):.1f}, max {delay_min.max():.1f}"
        )
    return m


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", choices=DATASETS, required=True)
    ap.add_argument("--model", help="model name; defaults to the best on validation F1")
    ap.add_argument(
        "--start-fraction", type=float, default=0.8, help="start scoring this far into the file (0.8 ≈ the test period)"
    )
    ap.add_argument(
        "--warmup-fraction",
        type=float,
        default=0.0,
        help="read silently from here up to --start-fraction to build session history",
    )
    ap.add_argument("--max-lines", type=int, help="stop after this many scored lines")
    ap.add_argument("--window-minutes", type=int, default=60, help="BGL window length")
    ap.add_argument("--idle-minutes", type=int, default=30, help="HDFS: score a session after this much quiet")
    ap.add_argument("--speed", type=float, default=0, help="log-seconds per wall-second; 0 = as fast as possible")
    args = ap.parse_args()
    run(
        args.dataset,
        args.model,
        args.start_fraction,
        args.warmup_fraction,
        args.max_lines,
        args.window_minutes,
        args.idle_minutes,
        args.speed,
    )


if __name__ == "__main__":
    main()
