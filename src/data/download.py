"""Download Loghub datasets into data/raw/<dataset>/.

python -m src.data.download --dataset hdfs
python -m src.data.download --dataset bgl
python -m src.data.download --dataset hdfs --sample   # 2k labelled-template sample only
"""

import argparse
import shutil
import urllib.request
import zipfile
from pathlib import Path

from tqdm import tqdm

from src.config import DATASETS, RAW_DIR

ZENODO = "https://zenodo.org/records/8196385/files"
GITHUB = "https://raw.githubusercontent.com/logpai/loghub/master"

FULL_URLS = {
    "hdfs": f"{ZENODO}/HDFS_v1.zip?download=1",
    "bgl": f"{ZENODO}/BGL.zip?download=1",
}
# 2k-line samples with ground-truth templates, used to measure parsing accuracy.
SAMPLE_URLS = {
    "hdfs": f"{GITHUB}/HDFS/HDFS_2k.log_structured.csv",
    "bgl": f"{GITHUB}/BGL/BGL_2k.log_structured.csv",
}


def _fetch(url: str, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    with urllib.request.urlopen(url) as resp, open(tmp, "wb") as out:
        total = int(resp.headers.get("Content-Length") or 0) or None
        with tqdm(total=total, unit="B", unit_scale=True, desc=dest.name) as bar:
            while chunk := resp.read(1 << 20):
                out.write(chunk)
                bar.update(len(chunk))
    tmp.replace(dest)


def download(dataset: str, sample: bool = False) -> Path:
    target = RAW_DIR / dataset
    if sample:
        dest = target / Path(SAMPLE_URLS[dataset]).name
        if not dest.exists():
            _fetch(SAMPLE_URLS[dataset], dest)
        return dest

    if any(target.rglob(f"{dataset.upper()}.log")):
        print(f"{dataset.upper()}.log already extracted; skipping download")
        return target

    archive = target / f"{dataset}.zip"
    if not archive.exists():
        _fetch(FULL_URLS[dataset], archive)
    with zipfile.ZipFile(archive) as zf:
        for member in zf.infolist():
            # Guard against path traversal in archive member names.
            out = (target / member.filename).resolve()
            if not out.is_relative_to(target.resolve()):
                raise ValueError(f"Unsafe path in archive: {member.filename}")
            if member.is_dir():
                out.mkdir(parents=True, exist_ok=True)
            elif not out.exists():
                out.parent.mkdir(parents=True, exist_ok=True)
                with zf.open(member) as src, open(out, "wb") as dst:
                    shutil.copyfileobj(src, dst)
    return target


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", choices=DATASETS, required=True)
    ap.add_argument("--sample", action="store_true", help="download only the 2k structured sample")
    args = ap.parse_args()
    print(f"Saved to {download(args.dataset, args.sample)}")


if __name__ == "__main__":
    main()
