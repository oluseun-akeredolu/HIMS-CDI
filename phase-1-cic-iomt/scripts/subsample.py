"""
subsample.py

This sandbox has 1 CPU and ~3.9GB RAM total -- not enough to train on the
full 7.16M-row real dataset in memory (confirmed empirically: a full-data
HistGradientBoostingClassifier fit was killed by the OOM killer). This is
a SANDBOX RESOURCE CONSTRAINT, not a property of the method: the same code
runs on the full corpus given a machine with adequate RAM (a "lab server"
per the original blueprint, not a 4GB container).

This module takes a SYSTEMATIC (fixed-stride, order-preserving) subsample
of each split -- not a random sample -- so the pseudo-chronological
ordering within each split is preserved rather than destroyed.
"""

from pathlib import Path
import numpy as np
import pyarrow.parquet as pq

FEATURE_COLS = [
    "Header_Length", "Protocol Type", "Duration", "Rate", "Srate", "Drate",
    "fin_flag_number", "syn_flag_number", "rst_flag_number", "psh_flag_number",
    "ack_flag_number", "ece_flag_number", "cwr_flag_number", "ack_count",
    "syn_count", "fin_count", "rst_count", "HTTP", "HTTPS", "DNS", "Telnet",
    "SMTP", "SSH", "IRC", "TCP", "UDP", "DHCP", "ARP", "ICMP", "IGMP", "IPv",
    "LLC", "Tot sum", "Min", "Max", "AVG", "Std", "Tot size", "IAT", "Number",
    "Magnitue", "Radius", "Covariance", "Variance", "Weight",
]


def systematic_subsample(parquet_path: Path, target_rows: int, columns=None):
    """Reads a parquet file in row-group batches and keeps every Nth row,
    where N is chosen so the total kept is close to target_rows. Streams
    batches rather than loading the whole file, so peak memory stays low
    regardless of the source file's total size."""
    columns = columns or (FEATURE_COLS + ["label"])
    pf = pq.ParquetFile(parquet_path)
    total_rows = pf.metadata.num_rows
    stride = max(1, total_rows // target_rows)

    kept_chunks = []
    seen = 0
    for batch in pf.iter_batches(batch_size=50_000, columns=columns):
        arr = batch.to_pandas()
        # keep every `stride`-th row, offset so we don't always land on
        # batch boundaries the same way
        local_idx = np.arange(len(arr))
        global_idx = local_idx + seen
        mask = (global_idx % stride) == 0
        if mask.any():
            kept_chunks.append(arr[mask])
        seen += len(arr)

    import pandas as pd
    result = pd.concat(kept_chunks, ignore_index=True) if kept_chunks else pd.DataFrame(columns=columns)
    return result
