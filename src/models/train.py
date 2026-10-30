"""Train and evaluate the rule baseline and count-vector models.

    python -m src.models.train --dataset hdfs

Thresholds are chosen to maximise F1 on the validation split, then frozen and
applied to the later test split. PR-AUC is threshold-free and reported alongside.
Unsupervised models train on all training sessions without labels, as they would
in production where nobody has labelled the logs.
"""
import argparse
import time

import joblib
import pandas as pd

from src.config import ARTIFACTS_DIR, DATASETS, RESULTS_DIR, SEQUENCES_DIR
from src.evaluation.metrics import best_f1_threshold, evaluate
from src.features.count_vectors import EventCountVectorizer
from src.grouping.split import chronological_split
from src.models.classical import default_models


def _period_days(df: pd.DataFrame) -> float:
    return (df["end_time"].max() - df["start_time"].min()) / 86400


def run(dataset: str, purge_overlap: bool = False) -> pd.DataFrame:
    sessions = pd.read_parquet(SEQUENCES_DIR / f"{dataset}_sessions.parquet")
    train, val, test = chronological_split(sessions, purge_overlap=purge_overlap)
    for name, part in (("train", train), ("val", val), ("test", test)):
        print(f"{name:5s} {len(part):>8,} groups  {part['label'].mean():6.2%} anomalous  "
              f"{_period_days(part):5.1f} days")

    vectorizer = EventCountVectorizer().fit(train["events"])
    X_train, X_val, X_test = (vectorizer.transform(p["events"]) for p in (train, val, test))
    y_train, y_val, y_test = (p["label"].to_numpy() for p in (train, val, test))
    test_days = _period_days(test)
    unseen = int(X_test[:, vectorizer.unk_index_].sum())
    print(f"{vectorizer.n_features - 1} event types in train; {unseen:,} test events are unseen types\n")

    rows = [{"model": "Keyword rule (ERROR/FATAL/fail/exception)", "threshold": 1.0,
             "fit_seconds": 0.0, **evaluate(y_test, test["keyword_hits"].to_numpy(), 1.0, test_days)}]
    fitted = {}
    for model in default_models():
        start = time.perf_counter()
        model.fit(X_train, y_train if model.supervised else None)
        fit_seconds = time.perf_counter() - start
        threshold = best_f1_threshold(y_val, model.score(X_val))
        metrics = evaluate(y_test, model.score(X_test), threshold, test_days)
        rows.append({"model": model.name, "threshold": threshold, "fit_seconds": fit_seconds, **metrics})
        fitted[model.name] = (model, threshold)
        print(f"{model.name:35s} F1={metrics['f1']:.3f}  ({fit_seconds:.1f}s)")

    artifact_dir = ARTIFACTS_DIR / dataset
    artifact_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump({"vectorizer": vectorizer, "models": fitted}, artifact_dir / "classical.joblib")
    return pd.DataFrame(rows)


def to_markdown(results: pd.DataFrame) -> str:
    lines = ["| Model | Precision | Recall | F1 | PR-AUC | False alarms / day |",
             "|---|---|---|---|---|---|"]
    for r in results.itertuples():
        lines.append(f"| {r.model} | {r.precision:.3f} | {r.recall:.3f} | {r.f1:.3f} | "
                     f"{r.pr_auc:.3f} | {r.false_alarms_per_day:.1f} |")
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", choices=DATASETS, required=True)
    ap.add_argument("--purge-overlap", action="store_true",
                    help="drop groups overlapping the previous split (needed for sliding windows)")
    args = ap.parse_args()

    results = run(args.dataset, args.purge_overlap)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    results.to_csv(RESULTS_DIR / f"{args.dataset}_classical.csv", index=False)
    table = to_markdown(results)
    (RESULTS_DIR / f"{args.dataset}_classical.md").write_text(table + "\n", encoding="utf-8")
    print("\n" + table)


if __name__ == "__main__":
    main()
