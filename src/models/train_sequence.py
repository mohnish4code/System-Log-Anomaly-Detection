"""Train and evaluate next-event sequence models.

    python -m src.models.train_sequence --dataset hdfs --model lstm
    python -m src.models.train_sequence --dataset hdfs --model transformer

Models learn from normal training sessions only (semi-supervised, as in DeepLog).
The alert threshold on the miss count is tuned on validation, as for every other model.
"""

import argparse
import time

import pandas as pd
import torch

from src.config import ARTIFACTS_DIR, DATASETS, RESULTS_DIR
from src.evaluation.metrics import best_f1_threshold, evaluate
from src.evaluation.report import write_report
from src.models.sequence import SequenceDetector
from src.models.train import load_split, period_days


def run(
    dataset: str,
    arch: str,
    epochs: int,
    top_g: int,
    max_len: int,
    purge_overlap: bool = False,
    use_mlflow: bool = False,
) -> dict:
    train, val, test = load_split(dataset, purge_overlap)
    normal_train = train.loc[train["label"] == 0, "events"]

    det = SequenceDetector(arch=arch, top_g=top_g, max_len=max_len, epochs=epochs)
    start = time.perf_counter()
    det.fit(normal_train)
    fit_seconds = time.perf_counter() - start

    val_scores = det.score(val["events"])
    threshold = best_f1_threshold(val["label"].to_numpy(), val_scores)
    val_f1 = evaluate(val["label"], val_scores, threshold, period_days(val))["f1"]
    metrics = evaluate(test["label"], det.score(test["events"]), threshold, period_days(test))
    print(f"{det.name}: val F1={val_f1:.3f}  test F1={metrics['f1']:.3f}  ({fit_seconds:.0f}s)")

    artifact_dir = ARTIFACTS_DIR / dataset
    artifact_dir.mkdir(parents=True, exist_ok=True)
    # Plain Python floats so the file loads with torch.load(weights_only=True).
    torch.save({**det.state(), "threshold": float(threshold), "val_f1": float(val_f1)}, artifact_dir / f"{arch}.pt")
    if use_mlflow:
        from src.evaluation.tracking import log_run

        log_run(
            f"{dataset}-log-anomaly",
            det.name,
            {"family": "sequence", "epochs": epochs, "top_g": top_g, "max_len": max_len},
            {"val_f1": val_f1, "fit_seconds": fit_seconds, **metrics},
        )
    return {"model": det.name, "threshold": threshold, "val_f1": val_f1, "fit_seconds": fit_seconds, **metrics}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", choices=DATASETS, required=True)
    ap.add_argument("--model", choices=["lstm", "transformer"], default="lstm")
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--top-g", type=int, default=9, help="an event is normal if among the top-g predictions")
    ap.add_argument("--max-len", type=int, default=128, help="longest context the model sees at once")
    ap.add_argument("--purge-overlap", action="store_true")
    ap.add_argument("--mlflow", action="store_true", help="log runs to ./mlruns")
    args = ap.parse_args()

    row = run(args.dataset, args.model, args.epochs, args.top_g, args.max_len, args.purge_overlap, args.mlflow)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    path = RESULTS_DIR / f"{args.dataset}_sequence.csv"
    existing = pd.read_csv(path) if path.exists() else pd.DataFrame()
    if not existing.empty:
        existing = existing[existing["model"] != row["model"]]
    pd.concat([existing, pd.DataFrame([row])], ignore_index=True).to_csv(path, index=False)
    print("\n" + write_report(args.dataset))


if __name__ == "__main__":
    main()
