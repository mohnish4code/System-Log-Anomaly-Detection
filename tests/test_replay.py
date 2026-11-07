import numpy as np
import pandas as pd
import pytest

import src.streaming.replay as replay_mod
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
    replay = Replay(StubScorer(), window_seconds=3600, idle_seconds=60, labels=None, evict_seconds=10_000)
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


def test_idle_scored_sessions_are_evicted_but_records_kept():
    replay = Replay(StubScorer(), window_seconds=3600, idle_seconds=60, labels=None, evict_seconds=600)
    replay.add("blk_1", 1, rec(0), "a", live=True)
    replay.add("blk_2", 1, rec(500), "b", live=True)
    replay.checkpoint(100)  # blk_1 scored, still recent enough to keep
    assert set(replay.open) == {"blk_1", "blk_2"} and "blk_1" in replay.records
    replay.checkpoint(700)  # blk_1 quiet for 700s > 600s: evicted; blk_2 is scored and kept
    assert set(replay.open) == {"blk_2"}
    assert set(replay.records) == {"blk_1", "blk_2"}

    replay.add("blk_3", 1, rec(4990), "c", live=True)  # not idle long enough to score yet
    replay.checkpoint(5000)
    assert set(replay.open) == {"blk_3"} and "blk_3" not in replay.records


def test_session_without_warmup_history_alerts():
    replay = Replay(StubScorer(), window_seconds=3600, idle_seconds=60, labels=None)
    replay.add("blk_9", 3, rec(1000), "tail only", live=True)
    replay.finish(1100)
    assert replay.records["blk_9"]["alert_time"] == 1100
    assert len(replay.alerts) == 1


# --- run(): the whole replay over a small log file ---------------------------------------

HDFS_LINE = "081109 {hms} 148 INFO dfs.DataNode$DataXceiver: event for block {blk}"


class StubParser:
    def parse(self, content):
        return 1


class RunScorer(StubScorer):
    model_name = "stub"
    parser = StubParser()

    def __init__(self, dataset, model=None):
        pass


@pytest.fixture
def run_env(tmp_path, monkeypatch):
    log = tmp_path / "HDFS.log"
    monkeypatch.setattr(replay_mod, "Scorer", RunScorer)
    monkeypatch.setattr(replay_mod, "find_raw_log", lambda dataset: log)
    monkeypatch.setattr(replay_mod, "load_hdfs_labels", lambda: pd.Series(dtype=int))
    monkeypatch.setattr(replay_mod, "STREAM_DIR", tmp_path / "stream")
    return log


def _run(warmup_fraction=0.0, start_fraction=0.0, max_lines=None):
    return replay_mod.run("hdfs", None, start_fraction, warmup_fraction, max_lines, 60, 30, 0)


def test_run_empty_file_reports_and_returns_empty(run_env, capsys):
    run_env.write_text("")
    assert _run().empty
    assert "Nothing to replay" in capsys.readouterr().out
    assert not (run_env.parent / "stream").exists()


def test_run_max_lines_zero_scores_nothing(run_env, capsys):
    run_env.write_text("".join(HDFS_LINE.format(hms=f"2036{i:02d}", blk="blk_1") + "\n" for i in range(5)))
    assert _run(max_lines=0).empty
    assert "Nothing was scored" in capsys.readouterr().out


def test_run_warmup_gives_sessions_their_history(run_env):
    early = [HDFS_LINE.format(hms=f"2036{i:02d}", blk="blk_1") for i in range(3)]
    late = [HDFS_LINE.format(hms="204000", blk="blk_1"), HDFS_LINE.format(hms="204001", blk="blk_2")]
    head = "".join(line + "\n" for line in early)
    run_env.write_text(head + "".join(line + "\n" for line in late))
    # Land on the newline that ends the early part, so scoring starts at the first late line.
    start = (len(head) - 0.5) / run_env.stat().st_size

    with_warmup = _run(warmup_fraction=0.0, start_fraction=start).set_index("session_id")
    assert with_warmup.loc["blk_1", "n_events"] == 4  # 3 warm-up events + 1 live
    assert not with_warmup.loc["blk_1", "alerted"] and not with_warmup.loc["blk_1", "started_live"]
    assert with_warmup.loc["blk_2", "started_live"]

    cold = _run(warmup_fraction=start, start_fraction=start).set_index("session_id")
    assert cold.loc["blk_1", "n_events"] == 1  # seen only from its tail: looks truncated
    assert cold.loc["blk_1", "alerted"]
    assert (run_env.parent / "stream" / "hdfs_scores.csv").exists()
