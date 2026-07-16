
"""
00_load_real_data.py

Loads all 51 real CIC IoMT 2024 attack-scenario CSV files, derives the
label from filename (Benign=0, everything else=1), and assigns a
pseudo-chronological order.

HONEST LIMITATION, stated explicitly rather than hidden: this dataset
format has NO per-row timestamp column. Each file is a separate capture
of one attack scenario; there's no ground-truth ordering that says
"ARP_Spoofing capture happened before TCP_IP-DDoS-SYN1 capture." The
pseudo-order used here is: files sorted alphabetically, rows within each
file kept in their original (capture) order. This is a reproducible
convention, not a true timeline. Anywhere this matters (the chronological
split), that limitation is real and should be reported as such, not
smoothed over.
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
SHARD_DIR = DATA_DIR / "shards"


def label_from_filename(path: Path) -> int:
    return 0 if path.stem.lower().startswith("benign") else 1


def attack_type_from_filename(path: Path) -> str:
    name = path.stem
    for suffix in ("_train_pcap",):
        if name.endswith(suffix):
            name = name[: -len(suffix)]
    return name


def main() -> None:
    # CRITICAL FIX: Only load training files, NOT test files
    files = sorted(RAW_DIR.glob("*_train_pcap.csv"))
    if not files:
        raise FileNotFoundError(f"No *_train_pcap.csv files found in {RAW_DIR}")
    print(f"Found {len(files)} training files. Processing file-by-file to keep memory bounded.")

    SHARD_DIR.mkdir(parents=True, exist_ok=True)
    row_counts = []
    running_offset = 0

    for file_order, path in enumerate(files):
        df = pd.read_csv(path)
        # downcast float64 -> float32 to roughly halve memory footprint
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
        row_counts.append((path.name, len(df), int(df["label"].iloc[0])))
        print(f"  [{file_order:2d}] {path.name:45s} rows={len(df):7,d} label={df['label'].iloc[0]}  -> {shard_path.name}")
        del df

    total = sum(r for _, r, _ in row_counts)
    print(f"\nTotal rows: {total:,} across {len(files)} shards written to {SHARD_DIR}")
    print("Rows are in pseudo-chronological order: file order (alphabetical) then in-file row order.")
    print("This is a documented convention, not a true timestamp-based ordering (see module docstring).")


if __name__ == "__main__":
    main()