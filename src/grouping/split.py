"""Chronological train / validation / test split.

Random shuffling lets a model train on sessions from after the test period,
which inflates scores. Here everything in train starts before everything in
validation, which starts before everything in test.
"""

import pandas as pd

from src.config import TRAIN_FRAC, VAL_FRAC


def chronological_split(
    sessions: pd.DataFrame, train_frac: float = TRAIN_FRAC, val_frac: float = VAL_FRAC, purge_overlap: bool = False
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split by start_time. With purge_overlap, groups that overlap the previous split in time
    are dropped; use it for sliding windows, where neighbouring windows share log lines."""
    if not (train_frac > 0 and val_frac > 0 and train_frac + val_frac < 1):
        raise ValueError("train_frac and val_frac must be positive and sum to less than 1")

    df = sessions.sort_values("start_time", kind="stable").reset_index(drop=True)
    n_train = int(len(df) * train_frac)
    n_val = int(len(df) * val_frac)
    train = df.iloc[:n_train]
    val = df.iloc[n_train : n_train + n_val]
    test = df.iloc[n_train + n_val :]

    if purge_overlap:
        val = val[val["start_time"] >= train["end_time"].max()]
        test = test[test["start_time"] >= val["end_time"].max()]
    return train.reset_index(drop=True), val.reset_index(drop=True), test.reset_index(drop=True)
