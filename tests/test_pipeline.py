import numpy as np
import pandas as pd
import pytest

from src.evaluation.metrics import alert_counts, best_f1_threshold, detection_delay_minutes, evaluate
from src.features.count_vectors import EventCountVectorizer
from src.grouping.build_sequences import build_hdfs_sessions, build_time_windows
from src.grouping.split import chronological_split
from src.models.classical import default_models


def test_hdfs_sessions_keep_line_order_and_labels():
    parsed = pd.DataFrame(
        {
            "line_id": [0, 1, 2, 3, 4],
            "timestamp": [10, 11, 12, 13, 14],
            "event_id": [1, 1, 2, 3, 2],
            "keyword": [False, False, False, True, False],
            "block_id": [7, -5, 7, -5, 7],
        }
    )
    labels = pd.Series([0, 1], index=[7, -5])
    s = build_hdfs_sessions(parsed, labels).set_index("session_id")
    assert s.loc["blk_7", "events"] == [1, 2, 2]
    assert s.loc["blk_-5", "events"] == [1, 3]
    assert s.loc["blk_-5", "label"] == 1 and s.loc["blk_-5", "keyword_hits"] == 1
    assert (s.loc["blk_7", "start_time"], s.loc["blk_7", "end_time"]) == (10, 14)


def test_time_windows_label_if_any_line_anomalous():
    parsed = pd.DataFrame(
        {
            "line_id": range(6),
            "timestamp": [0, 30, 70, 130, 400, 410],
            "event_id": [1, 2, 1, 3, 1, 1],
            "keyword": [False] * 6,
            "label": [0, 0, 0, 1, 0, 0],
        }
    )
    w = build_time_windows(parsed, window_seconds=60, step_seconds=60)
    assert w["events"].tolist() == [[1, 2], [1], [3], [1, 1]]  # empty windows dropped
    assert w["label"].tolist() == [0, 0, 1, 0]


def test_chronological_split_has_no_time_overlap():
    rng = np.random.default_rng(0)
    start = rng.permutation(1000)
    sessions = pd.DataFrame({"start_time": start, "end_time": start + 5, "label": 0})
    train, val, test = chronological_split(sessions, 0.6, 0.2)
    assert (len(train), len(val), len(test)) == (600, 200, 200)
    assert train["start_time"].max() < val["start_time"].min()
    assert val["start_time"].max() < test["start_time"].min()

    _, val_p, test_p = chronological_split(sessions, 0.6, 0.2, purge_overlap=True)
    assert val_p["start_time"].min() >= train["end_time"].max()
    assert len(val_p) < 200 and len(test_p) < 200

    with pytest.raises(ValueError):
        chronological_split(sessions, 0.8, 0.3)


def test_chronological_split_purges_overlapping_windows():
    # Sliding 60 s windows every 30 s: each window shares half its span with the next one.
    start = np.arange(10) * 30
    windows = pd.DataFrame({"start_time": start, "end_time": start + 60, "label": 0})
    train, val, test = chronological_split(windows, 0.6, 0.2, purge_overlap=True)
    assert train["start_time"].tolist() == [0, 30, 60, 90, 120, 150]
    assert val["start_time"].tolist() == [210]  # 180 overlaps the last train window (ends at 210)
    assert test["start_time"].tolist() == [270]  # 240 overlaps the kept val window (ends at 270)
    assert val["start_time"].min() >= train["end_time"].max()
    assert test["start_time"].min() >= val["end_time"].max()

    # Back-to-back windows that only touch are not overlapping, so nothing is dropped.
    touching = pd.DataFrame({"start_time": start, "end_time": start + 30, "label": 0})
    parts = chronological_split(touching, 0.6, 0.2, purge_overlap=True)
    assert [len(p) for p in parts] == [6, 2, 2]


def test_alert_counts_distinguish_zero_from_undefined():
    m = alert_counts([1, 0, 1, 0], [1, 1, 0, 0])
    assert (m["tp"], m["fp"], m["fn"]) == (1, 1, 1) and m["f1"] == 0.5
    m = alert_counts([1, 0], [0, 1])  # one false alarm, one miss
    assert (m["precision"], m["recall"], m["f1"]) == (0.0, 0.0, 0.0)
    m = alert_counts([1, 0], [0, 0])  # nothing flagged
    assert (m["precision"], m["recall"], m["f1"]) == (None, 0.0, None)
    m = alert_counts([0, 0], [1, 0])  # no positives
    assert (m["precision"], m["recall"], m["f1"]) == (0.0, None, None)


def test_detection_delay_uses_first_anomalous_line_for_bgl():
    scores = pd.DataFrame(
        {
            "alerted": [True, True, True, False],
            "label": [1, 1, 0, 1],
            "start_time": [0, 0, 0, 0],
            "first_anomaly_time": [600, None, None, None],
            "alert_time": [1200, 1800, 60, None],
            "started_live": [True, True, True, True],
        }
    )
    assert detection_delay_minutes(scores, "bgl").tolist() == [10.0, 30.0]
    assert detection_delay_minutes(scores, "hdfs").tolist() == [20.0, 30.0]
    assert detection_delay_minutes(scores.assign(started_live=False), "hdfs").empty


def test_vectorizer_maps_unseen_events_to_unk():
    vec = EventCountVectorizer().fit([[1, 2, 2], [5]])
    X = vec.transform([[2, 2, 9, -1], []]).toarray()
    assert vec.n_features == 4
    assert X.tolist() == [[0, 2, 0, 2], [0, 0, 0, 0]]


def test_best_threshold_and_metrics():
    y = np.array([0, 0, 0, 1, 1])
    scores = np.array([0.1, 0.2, 0.3, 0.8, 0.9])
    threshold = best_f1_threshold(y, scores)
    m = evaluate(y, scores, threshold, period_days=2.0)
    assert m["f1"] == 1.0 and m["pr_auc"] == 1.0 and m["false_alarms_per_day"] == 0.0

    m = evaluate(y, scores, 0.25, period_days=2.0)
    assert (m["tp"], m["fp"], m["fn"]) == (2, 1, 0)
    assert m["false_alarms_per_day"] == 0.5


@pytest.mark.parametrize("model", default_models(), ids=lambda m: m.name)
def test_models_rank_injected_anomalies_highest(model):
    rng = np.random.default_rng(0)
    normal = [list(rng.choice([1, 2, 3], size=10, p=[0.5, 0.3, 0.2])) for _ in range(300)]
    anomalous = [[1, 4, 4, 4, 5, 5] for _ in range(10)]
    sequences = normal + anomalous
    y = np.r_[np.zeros(len(normal)), np.ones(len(anomalous))].astype(int)

    vec = EventCountVectorizer().fit(sequences)
    X = vec.transform(sequences)
    # A tight cluster of identical anomalies in the training data becomes its own
    # principal component, so the unsupervised check fits on normal sessions only.
    if model.supervised:
        model.fit(X, y)
    else:
        model.fit(X[y == 0])
    scores = model.score(X)
    assert scores[y == 1].min() > np.median(scores[y == 0])
