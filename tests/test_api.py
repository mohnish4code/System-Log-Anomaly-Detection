"""End-to-end check of the scoring API on tiny artifacts built from synthetic HDFS lines."""
import joblib
import numpy as np
import pytest
from fastapi.testclient import TestClient

import src.api.main as api
import src.scoring as scoring
from src.features.count_vectors import EventCountVectorizer
from src.models.classical import PCADetector
from src.parsing.drain import LogParser
from src.parsing.log_formats import parse_hdfs_line

NORMAL_SESSION = [
    "081109 203615 148 INFO dfs.DataNode$DataXceiver: Receiving block {blk} src: /10.250.19.102:54106 dest: /10.250.19.102:50010",
    "081109 203615 149 INFO dfs.FSNamesystem: BLOCK* NameSystem.allocateBlock: /mnt/hadoop/job.jar. {blk}",
    "081109 203616 150 INFO dfs.DataNode$PacketResponder: Received block {blk} of size 67108864 from /10.250.10.6",
    "081109 203617 151 INFO dfs.DataNode$PacketResponder: PacketResponder 1 for block {blk} terminating",
]
FAILURE = "081109 203618 152 WARN dfs.DataNode$DataXceiver: writeBlock {blk} received exception java.io.IOException: Connection reset by peer"


@pytest.fixture
def client(tmp_path, monkeypatch):
    parser = LogParser()
    sessions = []
    for i in range(50):
        lines = [l.format(blk=f"blk_{i}") for l in NORMAL_SESSION]
        sessions.append([parser.parse(parse_hdfs_line(l).content) for l in lines])
    parser.parse(parse_hdfs_line(FAILURE.format(blk="blk_0")).content)  # template known, never in training

    vec = EventCountVectorizer().fit(sessions)
    X = vec.transform(sessions)
    pca = PCADetector().fit(X)
    out = tmp_path / "hdfs"
    out.mkdir()
    joblib.dump(parser, out / "drain.joblib")
    joblib.dump({"vectorizer": vec, "normal_mean": np.asarray(X.mean(axis=0)).ravel(),
                 "normal_std": np.ones(vec.n_features),
                 "models": {"PCA": {"model": pca, "threshold": 0.5, "val_f1": 1.0}}}, out / "classical.joblib")

    monkeypatch.setattr(scoring, "ARTIFACTS_DIR", tmp_path)
    monkeypatch.setattr(api, "ARTIFACTS_DIR", tmp_path)
    monkeypatch.setattr(api.data, "ARTIFACTS_DIR", tmp_path)
    monkeypatch.setattr(api.data, "STREAM_DIR", tmp_path / "stream")
    api.get_scorer.cache_clear()
    yield TestClient(api.app)
    api.get_scorer.cache_clear()


def test_health_and_models(client):
    assert client.get("/api/health").json() == {"status": "ok", "trained": ["hdfs"]}
    assert client.get("/api/hdfs/models").json()["default"] == "PCA"
    assert client.get("/api/hdfs/stream").status_code == 404  # no replay in the temp dir
    assert client.get("/api/hdfs/alerts").status_code == 404
    assert client.get("/api/bgl/results").status_code == 503


def test_score_flags_session_with_failure_and_explains_it(client):
    lines = [l.format(blk="blk_100") for l in NORMAL_SESSION]
    lines += [l.format(blk="blk_200") for l in NORMAL_SESSION[:2]] + [FAILURE.format(blk="blk_200")] * 3
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
