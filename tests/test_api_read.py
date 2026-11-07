"""Read endpoints (/datasets, /results, /stream, /alerts, /templates) on tiny fixture files."""

import json

import pandas as pd
import pytest
from fastapi.testclient import TestClient

import src.api.main as api
from src.api import data
from src.evaluation import report

T0 = 1_226_000_000

SCORES = pd.DataFrame(
    {
        "session_id": ["blk_1", "blk_2", "blk_3", "blk_4", "blk_5"],
        "alert_time": [T0 + 600, T0 + 900, None, None, T0 + 7300],
        "start_time": [T0, T0 + 300, T0 + 400, T0 + 500, T0 + 7200],
        "end_time": [T0 + 100, T0 + 350, T0 + 450, T0 + 550, T0 + 7250],
        "n_events": [3, 5, 14, 2, 4],
        "score": [0.9, 0.7, 0.1, 0.2, 0.95],
        "label": [1, 0, 0, 1, 1],
        "first_anomaly_time": [None] * 5,
        "alerted": [True, True, False, False, True],
        "started_live": [True, True, True, True, False],
    }
)


def _alert_detail(row: pd.Series) -> dict:
    rec = {k: (None if pd.isna(v) else v) for k, v in row.drop(["alerted", "started_live"]).items()}
    return {**rec, "top_events": [{"event_id": 1, "template": "t", "count": 1, "reason": "r"}], "raw_lines": ["x"]}


def write_stream(stream_dir, scores: pd.DataFrame, dataset: str = "hdfs") -> None:
    stream_dir.mkdir(exist_ok=True)
    scores.to_csv(stream_dir / f"{dataset}_scores.csv", index=False)
    with open(stream_dir / f"{dataset}_alerts.jsonl", "w", encoding="utf-8") as f:
        for _, row in scores[scores["alerted"]].iterrows():
            f.write(json.dumps(_alert_detail(row), default=int) + "\n")


@pytest.fixture
def paths(tmp_path, monkeypatch):
    for name in ("artifacts", "parsed", "sequences", "results"):
        (tmp_path / name).mkdir()
    monkeypatch.setattr(data, "ARTIFACTS_DIR", tmp_path / "artifacts")
    monkeypatch.setattr(data, "PARSED_DIR", tmp_path / "parsed")
    monkeypatch.setattr(data, "SEQUENCES_DIR", tmp_path / "sequences")
    monkeypatch.setattr(data, "STREAM_DIR", tmp_path / "results" / "stream")
    monkeypatch.setattr(report, "RESULTS_DIR", tmp_path / "results")
    # Loaders are cached on file mtimes; start each test from an empty cache.
    for fn in (data._overview, data._results, data._stream, data._templates):
        fn.cache_clear()
    return tmp_path


@pytest.fixture
def client(paths):
    """A 'trained' HDFS dataset: marker artifact, sessions, templates, results and a replay."""
    (paths / "artifacts" / "hdfs").mkdir()
    (paths / "artifacts" / "hdfs" / "classical.joblib").touch()

    pd.DataFrame(
        {
            "start_time": [T0 + i * 100 for i in range(10)],
            "end_time": [T0 + i * 100 + 50 for i in range(10)],
            "n_events": [4] * 10,
            "label": [0] * 8 + [1] * 2,
        }
    ).to_parquet(paths / "sequences" / "hdfs_sessions.parquet")
    pd.DataFrame({"event_id": [1, 2, 3], "template": ["a <*>", "b", "c"], "count": [70, 20, 10]}).to_csv(
        paths / "parsed" / "hdfs_templates.csv", index=False
    )
    # Event 3 first appears after the training period (60% of sessions -> starts at T0 + 600).
    pd.DataFrame({"event_id": [1, 2, 3, 1], "timestamp": [T0, T0 + 10, T0 + 800, T0 + 900]}).to_parquet(
        paths / "parsed" / "hdfs_structured.parquet"
    )
    cols = ["model", "threshold", "val_f1", "fit_seconds", "precision", "recall", "f1", "pr_auc", "tp", "fp", "fn"]
    pd.DataFrame(
        [
            ["PCA", 0.5, 0.8, 1.0, 0.9, 0.8, 0.85, 0.9, 8, 1, 2],
            ["Keyword rule", 1, 0.5, 0.0, 0.5, 0.5, 0.5, None, 5, 5, 5],
        ],
        columns=cols,
    ).to_csv(paths / "results" / "hdfs_classical.csv", index=False)
    pd.DataFrame([["LSTM (DeepLog-style)", 2.0, 0.9, 30.0, 0.95, 0.9, 0.92, 0.93, 9, 0, 1]], columns=cols).to_csv(
        paths / "results" / "hdfs_sequence.csv", index=False
    )
    write_stream(paths / "results" / "stream", SCORES)
    return TestClient(api.app)


def test_health_and_datasets(client):
    assert client.get("/api/health").json() == {"status": "ok", "trained": ["hdfs"]}
    (ds,) = client.get("/api/datasets").json()
    assert ds["dataset"] == "hdfs" and ds["grouping"] == "block-ID sessions"
    assert (ds["groups"], ds["anomalous"], ds["events"], ds["templates"]) == (10, 2, 40, 3)
    assert ds["anomaly_rate"] == pytest.approx(0.2)
    assert (ds["start_time"], ds["end_time"]) == (T0, T0 + 950)


def test_results_tag_model_families(client):
    rows = {r["model"]: r for r in client.get("/api/hdfs/results").json()}
    assert rows["PCA"]["family"] == "count-vector"
    assert rows["Keyword rule"]["family"] == "rule" and rows["Keyword rule"]["pr_auc"] is None
    assert rows["LSTM (DeepLog-style)"]["family"] == "sequence"
    assert not any(r["supervised"] for r in rows.values())


def test_stream_summary(client):
    s = client.get("/api/hdfs/stream").json()
    assert (s["groups"], s["anomalous"], s["alerts"]) == (5, 3, 3)
    assert (s["tp"], s["fp"], s["fn"]) == (2, 1, 1)
    assert s["precision"] == pytest.approx(2 / 3) and s["recall"] == pytest.approx(2 / 3)
    assert s["f1"] == pytest.approx(2 / 3)
    # blk_5 started before scoring began, so only blk_1 (alert 10 min after start) counts.
    assert s["delay_minutes"] == {"median": 10.0, "p90": 10.0, "max": 10.0}
    assert s["bucket_seconds"] == 3600
    assert sum(p["alerts"] for p in s["timeline"]) == 3


def test_alerts_filter_and_page(client):
    page = client.get("/api/hdfs/alerts").json()
    assert page["total"] == 3
    assert [a["session_id"] for a in page["items"]] == ["blk_5", "blk_1", "blk_2"]  # by score, descending

    false = client.get("/api/hdfs/alerts", params={"outcome": "false_alarm"}).json()
    assert false["total"] == 1 and false["items"][0]["session_id"] == "blk_2"

    second = client.get("/api/hdfs/alerts", params={"limit": 1, "offset": 1}).json()
    assert second["total"] == 3 and [a["session_id"] for a in second["items"]] == ["blk_1"]

    assert client.get("/api/hdfs/alerts", params={"outcome": "maybe"}).status_code == 422
    assert client.get("/api/hdfs/alerts", params={"limit": 0}).status_code == 422


def test_alert_detail_found_and_missing(client):
    detail = client.get("/api/hdfs/alerts/blk_2").json()
    assert detail["outcome"] == "false_alarm" and detail["raw_lines"] == ["x"]
    assert client.get("/api/hdfs/alerts/blk_1").json()["outcome"] == "true_alarm"
    missing = client.get("/api/hdfs/alerts/blk_3")  # scored but never alerted
    assert missing.status_code == 404 and "blk_3" in missing.json()["detail"]


def test_templates_lists_frequent_and_new(client):
    t = client.get("/api/hdfs/templates").json()
    assert (t["total"], t["train_end"], t["new_count"]) == (3, T0 + 600, 1)
    assert [r["event_id"] for r in t["frequent"]] == [1, 2, 3]
    assert t["frequent"][0]["share"] == pytest.approx(0.7)
    assert [r["event_id"] for r in t["new"]] == [3]


def test_missing_artifacts_and_replay(client, paths):
    for url in ("/api/bgl/results", "/api/bgl/stream", "/api/bgl/alerts", "/api/bgl/alerts/w1", "/api/bgl/templates"):
        assert client.get(url).status_code == 503, url

    for f in (paths / "results" / "stream").iterdir():
        f.unlink()
    for url in ("/api/hdfs/stream", "/api/hdfs/alerts", "/api/hdfs/alerts/blk_1"):
        r = client.get(url)
        assert r.status_code == 404 and "replay" in r.json()["detail"], url


def test_no_trained_datasets(paths):
    c = TestClient(api.app)
    assert c.get("/api/health").json()["trained"] == []
    assert c.get("/api/datasets").json() == []
    assert c.get("/api/hdfs/results").status_code == 503


# --- stream_summary edge cases ---------------------------------------------------------


def test_stream_summary_without_alerts(paths):
    scores = SCORES.assign(alerted=False, alert_time=None)
    write_stream(paths / "results" / "stream", scores)
    s = data.stream_summary("hdfs")
    assert (s["tp"], s["fp"], s["fn"]) == (0, 0, 3)
    assert s["precision"] is None  # nothing flagged: undefined, not zero
    assert s["recall"] == 0.0
    assert s["f1"] is None
    assert s["delay_minutes"] is None


def test_stream_summary_zero_precision_is_not_null(paths):
    scores = SCORES.assign(label=[0, 0, 1, 1, 0])  # every alert is a false alarm
    write_stream(paths / "results" / "stream", scores)
    s = data.stream_summary("hdfs")
    assert (s["tp"], s["fp"], s["fn"]) == (0, 3, 2)
    assert s["precision"] == 0.0 and s["recall"] == 0.0 and s["f1"] == 0.0
    assert s["delay_minutes"] is None


def test_stream_summary_no_replay(paths):
    assert data.stream_summary("hdfs") is None
