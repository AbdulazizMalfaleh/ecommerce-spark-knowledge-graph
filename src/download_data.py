"""Download the REES46 e-commerce events dataset from Kaggle into data/raw/.

Needs a Kaggle API token (one-time setup):
    1. kaggle.com -> Settings -> API -> "Create New Token" -> downloads kaggle.json
    2. put it at ~/.kaggle/kaggle.json  (Windows: C:\\Users\\<you>\\.kaggle\\kaggle.json)
    3. pip install kaggle

Usage:
    python -m src.download_data                          # 2019-Oct.csv (~42M events, ~5 GB)
    python -m src.download_data --file 2019-Nov.csv
    python -m src.download_data --sample-rows 2000000    # keep only the first 2M rows (faster dev loop)
"""
import argparse
import subprocess
import zipfile
from pathlib import Path

from src.config import KAGGLE_DATASET, KAGGLE_FILES, RAW_DIR


def download(file_name: str, sample_rows: int | None) -> None:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    target = RAW_DIR / file_name

    if not target.exists():
        print(f"Downloading {file_name} from Kaggle ...")
        subprocess.run(
            ["kaggle", "datasets", "download", "-d", KAGGLE_DATASET,
             "-f", file_name, "-p", str(RAW_DIR)],
            check=True,
        )
        zip_path = RAW_DIR / f"{file_name}.zip"
        if zip_path.exists():
            with zipfile.ZipFile(zip_path) as zf:
                zf.extractall(RAW_DIR)
            zip_path.unlink()
    print(f"Raw file: {target}")

    if sample_rows:
        sample = RAW_DIR / f"sample-{sample_rows}-{file_name}"
        with open(target, encoding="utf-8") as src, open(sample, "w", encoding="utf-8") as dst:
            for i, line in enumerate(src):
                if i > sample_rows:
                    break
                dst.write(line)
        print(f"Sample with {sample_rows:,} rows: {sample}")
        print("Tip: move the full file out of data/raw/ so the pipeline reads only the sample.")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--file", choices=KAGGLE_FILES, default=KAGGLE_FILES[0])
    p.add_argument("--sample-rows", type=int)
    a = p.parse_args()
    download(a.file, a.sample_rows)
