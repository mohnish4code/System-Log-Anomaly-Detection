from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PARSED_DIR = DATA_DIR / "parsed"
SEQUENCES_DIR = DATA_DIR / "sequences"
RESULTS_DIR = ROOT / "results"
ARTIFACTS_DIR = ROOT / "artifacts"

DATASETS = ("hdfs", "bgl")

# Chronological split fractions (the remainder is the test set).
TRAIN_FRAC = 0.6
VAL_FRAC = 0.2

RANDOM_STATE = 42
