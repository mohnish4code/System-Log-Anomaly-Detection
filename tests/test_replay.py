import numpy as np

from src.parsing.log_formats import LogRecord
from src.streaming.replay import Replay


class StubScorer:
    """Flags any session that has fewer than 3 events: an incomplete-looking session."""
    dataset = "hdfs"
    threshold = 0.5

    def score(self, sequences):
        return np.array([float(len(s) < 3) for s in sequences])

    def explain(self, events):
        return []


def rec(ts):
    return LogRecord(timestamp=ts, level="INFO", component="c", content="x")


def test_warmup_builds_history_without_scoring():
    replay = Replay(StubScorer(), window_seconds=3600, idle_seconds=60, labels=None)
    replay.add("blk_1", 1, rec(0), "line a", live=False)
    replay.add("blk_1", 2, rec(10), "line b", live=False)
    replay.add("blk_2", 1, rec(20), "line c", live=False)  # finishes during warm-up
    replay.checkpoint(1000)
    assert replay.records == {}

    replay.add("blk_1", 3, rec(1000), "line d", live=True)
    replay.checkpoint(2000)
    # Scored with its full 3-event history, so it is not mistaken for a truncated session.
    assert replay.records["blk_1"]["n_events"] == 3
    assert replay.records["blk_1"]["alert_time"] is None
    assert "blk_2" not in replay.records
    assert replay.open["blk_1"].raw_lines == ["line d"]  # raw lines only kept once live


def test_session_without_warmup_history_alerts():
    replay = Replay(StubScorer(), window_seconds=3600, idle_seconds=60, labels=None)
    replay.add("blk_9", 3, rec(1000), "tail only", live=True)
    replay.finish(1100)
    assert replay.records["blk_9"]["alert_time"] == 1100
    assert len(replay.alerts) == 1
