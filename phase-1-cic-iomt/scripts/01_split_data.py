"""
01_split_data.py (v3 -- using the dataset's own official train/test split)

CIC IoMT 2024 ships separate TRAIN and TEST capture files. This version
uses that real split rather than self-carving a test set out of the
training files (which is what the previous run did, for lack of test
files at the time):

  - The 51 official TRAIN files are split per-session at 75%/25%
    (preserving in-session order, no shuffling) into TRAIN and TUNE.
  - The 21 official TEST files are used WHOLESALE as TEST -- the dataset's
    own held-out set, never touched during training or calibration.

Why 75/25 rather than another ratio: test files are ~18.4% of the combined
train+test corpus. Splitting train files 75/25 gives overall proportions
of roughly 61% train / 20% tune / 18% test -- close to the spec's 60/20/20
target while respecting the dataset's real train/test boundary rather than
forcing an exact 60/20/20 split that would require ignoring that boundary.

Same per-session rationale as before still applies: there's no real
cross-session timestamp within the 51 train files, so splitting each
session at its own 75% mark (not shuffling) is the honest way to respect
"no shuffling" where a real within-session timeline actually exists.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import pyarrow as pa
import pyarrow.parquet as pq
from csra.config import DATA_DIR

TRAIN_SHARD_DIR = DATA_DIR / "shards"
TEST_SHARD_DIR = DATA_DIR / "test_shards"

TRAIN_FRAC_OF_TRAINFILES = 0.75  # -> 25% of train files becomes TUNE


def main() -> None:
    train_shards = sorted(TRAIN_SHARD_DIR.glob("*.parquet"))
    test_shards = sorted(TEST_SHARD_DIR.glob("*.parquet"))
    if not train_shards:
        raise FileNotFoundError(f"No shards in {TRAIN_SHARD_DIR}. Run 00_load_real_data.py first.")
    if not test_shards:
        raise FileNotFoundError(f"No shards in {TEST_SHARD_DIR}. Run 00b_load_test_shards.py first.")

    out_paths = {k: DATA_DIR / f"{k}.parquet" for k in ("train", "tune", "test")}
    writers = {k: None for k in out_paths}
    counts = {k: 0 for k in out_paths}
    label_counts = {k: {0: 0, 1: 0} for k in out_paths}

    def write_chunk(split_name, chunk):
        if chunk.num_rows == 0:
            return
        if writers[split_name] is None:
            writers[split_name] = pq.ParquetWriter(out_paths[split_name], chunk.schema)
        writers[split_name].write_table(chunk)
        counts[split_name] += chunk.num_rows
        labels = chunk.column("label").to_pylist()
        label_counts[split_name][0] += labels.count(0)
        label_counts[split_name][1] += labels.count(1)

    # --- TRAIN files -> TRAIN (first 75% of each session) + TUNE (last 25%) ---
    for shard_path in train_shards:
        table = pq.read_table(shard_path)
        n = table.num_rows
        cut = int(n * TRAIN_FRAC_OF_TRAINFILES)
        write_chunk("train", table.slice(0, cut))
        write_chunk("tune", table.slice(cut, n - cut))

    # --- TEST files -> TEST, wholesale, untouched ordering ---
    for shard_path in test_shards:
        table = pq.read_table(shard_path)
        write_chunk("test", table)

    for w in writers.values():
        if w is not None:
            w.close()

    total = sum(counts.values())
    print(f"Total rows across all splits: {total:,}\n")
    for name in ("train", "tune", "test"):
        c, lc = counts[name], label_counts[name]
        pct = 100 * c / total
        print(f"{name.capitalize():6s}: {c:,} rows ({pct:.1f}%)  label=0(benign): {lc[0]:,}  "
              f"label=1(attack): {lc[1]:,}  -> {out_paths[name]}")

    for name in ("train", "tune", "test"):
        assert label_counts[name][0] > 0, f"{name} split has zero benign samples"
        assert label_counts[name][1] > 0, f"{name} split has zero attack samples"

    print("\nAll three splits contain both classes. TEST is the dataset's OWN official held-out")
    print("set (never touched during train/tune) -- do not open test.parquet again until Step 1.4.")


if __name__ == "__main__":
    main()
