"""Parsing (grouping) accuracy against Loghub's hand-labelled 2k-line samples.

    python -m src.data.download --dataset hdfs --sample
    python -m src.evaluation.parsing_accuracy --dataset hdfs

A line counts as correctly parsed only if the set of lines sharing its predicted
template is exactly the set sharing its ground-truth template (Zhu et al., ICSE-SEIP'19).
"""
import argparse

import numpy as np
import pandas as pd

from src.config import DATASETS, RAW_DIR
from src.parsing.drain import LogParser


def grouping_accuracy(truth: np.ndarray, pred: np.ndarray) -> float:
    df = pd.DataFrame({"truth": truth, "pred": pred})
    truth_sizes = df["truth"].value_counts()
    correct = 0
    for _, group in df.groupby("pred"):
        truth_ids = group["truth"].unique()
        if len(truth_ids) == 1 and truth_sizes[truth_ids[0]] == len(group):
            correct += len(group)
    return correct / len(df)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", choices=DATASETS, required=True)
    ap.add_argument("--sim-th", type=float, nargs="+", default=[0.3, 0.4, 0.5, 0.6, 0.7])
    args = ap.parse_args()

    path = RAW_DIR / args.dataset / f"{args.dataset.upper()}_2k.log_structured.csv"
    if not path.exists():
        raise FileNotFoundError(f"{path} missing. Run: python -m src.data.download --dataset {args.dataset} --sample")
    gt = pd.read_csv(path)

    for sim_th in args.sim_th:
        parser = LogParser(sim_th=sim_th)
        pred = np.array([parser.parse(str(c)) for c in gt["Content"]])
        acc = grouping_accuracy(gt["EventId"].to_numpy(), pred)
        print(f"sim_th={sim_th:.2f}  templates={len(np.unique(pred)):4d} "
              f"(truth {gt['EventId'].nunique()})  grouping accuracy={acc:.3f}")


if __name__ == "__main__":
    main()
