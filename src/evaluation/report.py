"""Merge per-family result CSVs into one table per dataset.

    python -m src.evaluation.report --dataset hdfs
"""
import argparse

import pandas as pd

from src.config import DATASETS, RESULTS_DIR

FAMILIES = ("classical", "sequence")


def load_results(dataset: str) -> pd.DataFrame:
    frames = [pd.read_csv(p) for f in FAMILIES if (p := RESULTS_DIR / f"{dataset}_{f}.csv").exists()]
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def to_markdown(results: pd.DataFrame) -> str:
    lines = ["| Model | Precision | Recall | F1 | PR-AUC | False alarms (test) | False alarms / day |",
             "|---|---|---|---|---|---|---|"]
    for r in results.itertuples():
        lines.append(f"| {r.model} | {r.precision:.3f} | {r.recall:.3f} | {r.f1:.3f} | "
                     f"{r.pr_auc:.3f} | {r.fp:,} | {r.false_alarms_per_day:,.1f} |")
    return "\n".join(lines)


def write_report(dataset: str) -> str:
    table = to_markdown(load_results(dataset))
    (RESULTS_DIR / f"{dataset}_results.md").write_text(table + "\n", encoding="utf-8")
    return table


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", choices=DATASETS, required=True)
    print(write_report(ap.parse_args().dataset))


if __name__ == "__main__":
    main()
