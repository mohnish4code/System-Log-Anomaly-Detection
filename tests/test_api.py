"""End-to-end check of the scoring API on tiny artifacts built from synthetic HDFS and BGL lines."""

import joblib
import numpy as np
import pytest
from fastapi.testclient import TestClient

import src.api.main as api
import src.scoring as scoring
from src.features.count_vectors import EventCountVectorizer
from src.models.classical import PCADetector
from src.parsing.drain import LogParser
from src.parsing.log_formats import parse_bgl_line, parse_hdfs_line

NORMAL_SESSION = [
    "081109 203615 148 INFO dfs.DataNode$DataXceiver: Receiving block {blk} "
    "src: /10.250.19.102:54106 dest: /10.250.19.102:50010",
    "081109 203615 149 INFO dfs.FSNamesystem: BLOCK* NameSystem.allocateBlock: /mnt/hadoop/job.jar. {blk}",
    "081109 203616 150 INFO dfs.DataNode$PacketResponder: Received block {blk} of size 67108864 from /10.250.10.6",
    "081109 203617 151 INFO dfs.DataNode$PacketResponder: PacketResponder 1 for block {blk} terminating",
]
FAILURE = (
    "081109 203618 152 WARN dfs.DataNode$DataXceiver: writeBlock {blk} "
    "received exception java.io.IOException: Connection reset by peer"
)


# BGL: one normal hourly window is a few routine kernel messages; failures are FATAL lines.
BGL_T0 = 1117839600  # a multiple of 3600, so 10- and 60-minute windows both start here
BGL_LINE = "{alert} {ts} 2005.06.03 R02-M1-N0-C:J12-U11 2005-06-03-15.42.50.675872 R02-M1-N0-C:J12-U11 RAS KERNEL {msg}"
BGL_NORMAL = [
    ("-", "INFO instruction cache parity error corrected"),
    ("-", "INFO generating core.2275"),
    ("-", "INFO CE sym 2, at 0x0b85eee0, mask 0x05"),
]
BGL_FAILURE = ("KERNDTLB", "FATAL data TLB error interrupt")


def bgl_line(offset: int, alert: str, msg: str) -> str:
    return BGL_LINE.format(alert=alert, ts=BGL_T0 + offset, msg=msg)


def save_artifacts(out, parser, sessions) -> None:
    vec = EventCountVectorizer().fit(sessions)
    X = vec.transform(sessions)
    pca = PCADetector().fit(X)
    out.mkdir()
    joblib.dump(parser, out / "drain.joblib")
    joblib.dump(
        {
            "vectorizer": vec,
            "normal_mean": np.asarray(X.mean(axis=0)).ravel(),
            "normal_std": np.ones(vec.n_features),
            "models": {"PCA": {"model": pca, "threshold": 0.5, "val_f1": 1.0}},
        },
        out / "classical.joblib",
    )


def build_hdfs(out) -> None:
    parser = LogParser()
    sessions = []
    for i in range(50):
        lines = [line.format(blk=f"blk_{i}") for line in NORMAL_SESSION]
        sessions.append([parser.parse(parse_hdfs_line(line).content) for line in lines])
    parser.parse(parse_hdfs_line(FAILURE.format(blk="blk_0")).content)  # template known, never in training
    save_artifacts(out, parser, sessions)


def build_bgl(out) -> None:
    parser = LogParser()
    sessions = [[parser.parse(parse_bgl_line(bgl_line(0, a, m)).content) for a, m in BGL_NORMAL] for _ in range(50)]
    parser.parse(parse_bgl_line(bgl_line(0, *BGL_FAILURE)).content)
    save_artifacts(out, parser, sessions)


def make_client(tmp_path, monkeypatch, datasets):
    for name, build in datasets.items():
        build(tmp_path / name)
    monkeypatch.setattr(scoring, "ARTIFACTS_DIR", tmp_path)
    monkeypatch.setattr(api, "ARTIFACTS_DIR", tmp_path)
    monkeypatch.setattr(api.data, "ARTIFACTS_DIR", tmp_path)
    monkeypatch.setattr(api.data, "STREAM_DIR", tmp_path / "stream")
    api.get_scorer.cache_clear()
    return TestClient(api.app)


@pytest.fixture
def client(tmp_path, monkeypatch):
    yield make_client(tmp_path, monkeypatch, {"hdfs": build_hdfs})
    api.get_scorer.cache_clear()


@pytest.fixture
def bgl_client(tmp_path, monkeypatch):
    yield make_client(tmp_path, monkeypatch, {"bgl": build_bgl})
    api.get_scorer.cache_clear()


def test_health_and_models(client):
    assert client.get("/api/health").json() == {"status": "ok", "trained": ["hdfs"]}
    assert client.get("/api/hdfs/models").json()["default"] == "PCA"
    assert client.get("/api/hdfs/stream").status_code == 404  # no replay in the temp dir
    assert client.get("/api/hdfs/alerts").status_code == 404
    assert client.get("/api/bgl/results").status_code == 503


def test_score_flags_session_with_failure_and_explains_it(client):
    lines = [line.format(blk="blk_100") for line in NORMAL_SESSION]
    lines += [line.format(blk="blk_200") for line in NORMAL_SESSION[:2]] + [FAILURE.format(blk="blk_200")] * 3
    lines.append("not a log line")
    body = client.post("/api/score", json={"dataset": "hdfs", "lines": lines}).json()

    assert body["n_unparsed"] == 1
    by_id = {s["session_id"]: s for s in body["sessions"]}
    assert not by_id["blk_100"]["is_anomaly"]
    flagged = by_id["blk_200"]
    assert flagged["is_anomaly"]
    assert any("exception" in e["template"] for e in flagged["top_events"])


def test_score_rejects_bad_requests(client):
    assert client.post("/api/score", json={"dataset": "hdfs", "lines": []}).status_code == 422
    assert client.post("/api/score", json={"dataset": "bgl", "lines": ["x"]}).status_code == 503
    assert client.post("/api/score", json={"dataset": "hdfs", "lines": ["x"], "model": "nope"}).status_code == 404


def test_score_rejects_oversized_input(client, monkeypatch):
    too_long = "x" * (api.MAX_LINE_CHARS + 1)
    assert client.post("/api/score", json={"dataset": "hdfs", "lines": [too_long]}).status_code == 422

    monkeypatch.setattr(api, "MAX_TOTAL_CHARS", 100)
    lines = [line.format(blk="blk_1") for line in NORMAL_SESSION]
    assert client.post("/api/score", json={"dataset": "hdfs", "lines": lines}).status_code == 422

    monkeypatch.setattr(api, "MAX_BODY_BYTES", 50)
    assert client.post("/api/score", json={"dataset": "hdfs", "lines": lines}).status_code == 413


def test_score_bgl_groups_lines_into_time_windows(bgl_client):
    normal = [bgl_line(i * 10, a, m) for i, (a, m) in enumerate(BGL_NORMAL)]
    failures = [bgl_line(700 + i, *BGL_FAILURE) for i in range(3)]  # second 10-minute window
    lines = normal + failures

    body = bgl_client.post("/api/score", json={"dataset": "bgl", "lines": lines, "window_minutes": 10}).json()
    by_id = {s["session_id"]: s for s in body["sessions"]}
    assert set(by_id) == {f"w{BGL_T0}", f"w{BGL_T0 + 600}"}
    assert not by_id[f"w{BGL_T0}"]["is_anomaly"]
    flagged = by_id[f"w{BGL_T0 + 600}"]
    assert flagged["is_anomaly"] and flagged["n_events"] == 3
    assert (flagged["start_time"], flagged["end_time"]) == (BGL_T0 + 700, BGL_T0 + 702)
    assert any("TLB" in e["template"] for e in flagged["top_events"])

    # One hour-long window holds every line.
    body = bgl_client.post("/api/score", json={"dataset": "bgl", "lines": lines, "window_minutes": 60}).json()
    assert [s["session_id"] for s in body["sessions"]] == [f"w{BGL_T0}"]
    assert body["sessions"][0]["n_events"] == len(lines)

    assert (
        bgl_client.post("/api/score", json={"dataset": "bgl", "lines": lines, "window_minutes": 0}).status_code == 422
    )
