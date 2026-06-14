# /// script
# requires-python = ">=3.11"
# dependencies = ["kagglehub"]
# ///
"""Download the NYC taxi dataset from Kaggle and load it into HDFS.

Run with:  uv run download_data.py
Needs:     the cluster running (docker compose up -d).
"""

import shutil
import subprocess
from pathlib import Path

import kagglehub

DATASET = "elemento/nyc-yellow-taxi-trip-data"
DATA_DIR = Path("./data")


def download() -> None:
    DATA_DIR.mkdir(exist_ok=True)

    # kagglehub downloads + unzips into a cache dir and returns that path
    cache_path = Path(kagglehub.dataset_download(DATASET))
    print(f"Downloaded to cache: {cache_path}")

    # Copy every CSV out of the cache into ./data
    csv_files = list(cache_path.rglob("*.csv"))
    if not csv_files:
        raise SystemExit(f"No CSV files found under {cache_path}")

    for csv in csv_files:
        dest = DATA_DIR / csv.name
        shutil.copy(csv, dest)
        print(f"Copied {csv.name} -> {dest}")


def load_into_hdfs() -> None:
    subprocess.run(
        ["docker", "compose", "exec", "-T", "namenode",
         "hdfs", "dfs", "-mkdir", "-p", "/data"],
        check=True,
    )
    # The glob expands *inside* the container, so run it through a shell
    subprocess.run(
        ["docker", "compose", "exec", "-T", "namenode",
         "bash", "-c", "hdfs dfs -put -f /data-local/*.csv /data/"],
        check=True,
    )
    print("Loaded into HDFS — verify at http://localhost:9870")


if __name__ == "__main__":
    download()
    load_into_hdfs()