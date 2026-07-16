"""
00b_load_test_shards.py

Loads the 21 official CIC IoMT 2024 TEST files (separate capture sessions
from the 51 TRAIN files already loaded), labels from filename, writes
memory-safe parquet shards -- mirrors 00_load_real_data.py.

These files become the TEST / keep-out zone directly (no further splitting
of them) -- they are the dataset's own held-out set, which is stronger
than the previous run's self-carved split out of the train files alone.
"""

import os
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import numpy as np
import pandas as pd
from csra.config import DATA_DIR

# Use environment variable if set, otherwise fall back to project-root/data/raw
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = Path(os.environ.get("CSRA_DATA_DIR", str(_PROJECT_ROOT / "data" / "raw")))
SHARD_DIR = DATA_DIR / "test_shards"


def label_from_filename(path: Path) -> int:
    return 0 if path.stem.lower().startswith("benign") else 1


def attack_type_from_filename(path: Path) -> str:
    name = path.stem
    if name.endswith("_test_pcap"):
        name = name[: -len("_test_pcap")]
    return name


def main() -> None:
    files = sorted(RAW_DIR.glob("*_test_pcap.csv"))
    if not files:
        raise FileNotFoundError(f"No *_test_pcap.csv files found in {RAW_DIR}")
    print(f"Found {len(files)} official test files.")

    SHARD_DIR.mkdir(parents=True, exist_ok=True)
    running_offset = 0

    for file_order, path in enumerate(files):
        df = pd.read_csv(path)
        float_cols = df.select_dtypes(include="float64").columns
        df[float_cols] = df[float_cols].astype(np.float32)

        df["label"] = np.int8(label_from_filename(path))
        df["attack_type"] = attack_type_from_filename(path)
        df["source_file"] = path.name
        df["file_order"] = np.int16(file_order)
        df["row_order_in_file"] = np.arange(len(df), dtype=np.int32)
        df["pseudo_sequence_index"] = np.arange(running_offset, running_offset + len(df), dtype=np.int64)
        running_offset += len(df)

        shard_path = SHARD_DIR / f"{file_order:02d}_{path.stem}.parquet"
        df.to_parquet(shard_path, index=False)
        print(f"  [{file_order:2d}] {path.name:45s} rows={len(df):7,d} label={df['label'].iloc[0]}  -> {shard_path.name}")
        del df

    print(f"\nTotal official test rows: {running_offset:,} across {len(files)} shards written to {SHARD_DIR}")


if __name__ == "__main__":
    main()