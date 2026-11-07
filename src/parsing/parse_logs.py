"""Parse a raw Loghub file into structured events with Drain.

    python -m src.parsing.parse_logs --dataset hdfs
    python -m src.parsing.parse_logs --dataset bgl --max-lines 200000

Outputs:
    data/parsed/<dataset>_structured.parquet  one row per (line, group key)
    data/parsed/<dataset>_templates.csv       event_id, template, count
    artifacts/<dataset>/drain.joblib          parser state, reused for streaming

The file is parsed once in its original (chronological) order, exactly as an
online parser would see it, so templates for the test period are never used
before they appear.
"""

import argparse
import time
from pathlib import Path

import joblib
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from tqdm import tqdm

from src.config import ARTIFACTS_DIR, DATASETS, PARSED_DIR, RAW_DIR
from src.parsing.drain import LogParser
from src.parsing.log_formats import BLOCK_RE, LINE_PARSERS, is_keyword_alert

RAW_FILES = {"hdfs": "HDFS.log", "bgl": "BGL.log"}

SCHEMAS = {
    "hdfs": pa.schema(
        [
            ("line_id", pa.int64()),
            ("timestamp", pa.int64()),
            ("event_id", pa.int32()),
            ("keyword", pa.bool_()),
            ("block_id", pa.int64()),
        ]
    ),
    "bgl": pa.schema(
        [
            ("line_id", pa.int64()),
            ("timestamp", pa.int64()),
            ("event_id", pa.int32()),
            ("keyword", pa.bool_()),
            ("label", pa.int8()),
            ("node", pa.string()),
        ]
    ),
}


def find_raw_log(dataset: str) -> Path:
    matches = sorted((RAW_DIR / dataset).rglob(RAW_FILES[dataset]))
    if not matches:
        raise FileNotFoundError(
            f"{RAW_FILES[dataset]} not found under {RAW_DIR / dataset}. "
            f"Run: python -m src.data.download --dataset {dataset}"
        )
    return matches[0]


def parse_file(
    dataset: str,
    log_path: Path,
    out_dir: Path,
    parser: LogParser | None = None,
    max_lines: int | None = None,
    chunk_size: int = 1_000_000,
) -> LogParser:
    parse_line = LINE_PARSERS[dataset]
    schema = SCHEMAS[dataset]
    parser = parser or LogParser()
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{dataset}_structured.parquet"

    cols: dict[str, list] = {name: [] for name in schema.names}
    skipped = 0

    def flush(writer: pq.ParquetWriter) -> None:
        writer.write_table(pa.Table.from_pydict(cols, schema=schema))
        for v in cols.values():
            v.clear()

    with open(log_path, encoding="utf-8", errors="replace") as f, pq.ParquetWriter(out_path, schema) as writer:
        for line_id, line in enumerate(tqdm(f, unit=" lines", mininterval=2, total=max_lines)):
            if max_lines is not None and line_id >= max_lines:
                break
            rec = parse_line(line)
            if rec is None:
                skipped += 1
                continue
            event_id = parser.parse(rec.content)
            keyword = is_keyword_alert(rec.level, rec.content)

            if dataset == "hdfs":
                # A line can mention several blocks (e.g. batch deletes); it belongs to each.
                for blk in set(BLOCK_RE.findall(rec.content)):
                    cols["line_id"].append(line_id)
                    cols["timestamp"].append(rec.timestamp)
                    cols["event_id"].append(event_id)
                    cols["keyword"].append(keyword)
                    cols["block_id"].append(int(blk[4:]))
            else:
                cols["line_id"].append(line_id)
                cols["timestamp"].append(rec.timestamp)
                cols["event_id"].append(event_id)
                cols["keyword"].append(keyword)
                cols["label"].append(rec.label)
                cols["node"].append(rec.node)

            if len(cols["line_id"]) >= chunk_size:
                flush(writer)
        flush(writer)

    templates = pd.DataFrame(
        [(eid, tpl, n) for eid, (tpl, n) in parser.templates().items()],
        columns=["event_id", "template", "count"],
    ).sort_values("event_id")
    templates.to_csv(out_dir / f"{dataset}_templates.csv", index=False)
    print(f"{len(templates)} templates, {skipped} unparseable lines skipped -> {out_path}")
    return parser


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", choices=DATASETS, required=True)
    ap.add_argument("--log-path", type=Path, help="override the raw log location")
    ap.add_argument("--max-lines", type=int, help="parse only the first N lines (for quick runs)")
    ap.add_argument("--sim-th", type=float, default=0.4, help="Drain similarity threshold")
    args = ap.parse_args()

    log_path = args.log_path or find_raw_log(args.dataset)
    start = time.perf_counter()
    parser = parse_file(args.dataset, log_path, PARSED_DIR, LogParser(sim_th=args.sim_th), args.max_lines)
    print(f"Parsed in {time.perf_counter() - start:.0f}s")

    artifact_dir = ARTIFACTS_DIR / args.dataset
    artifact_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(parser, artifact_dir / "drain.joblib")


if __name__ == "__main__":
    main()
