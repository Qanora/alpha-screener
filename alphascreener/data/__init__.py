"""Data I/O layer: Parquet read/write, Hive-partitioned storage, zstd archiving.

Issue #86: Data I/O layer.
Reference: PRD 7.6.1 / 7.6.3.
"""

from alphascreener.data.io import (
    archive_old_data,
    read_parquet,
    scan_parquet,
    write_parquet,
)
from alphascreener.data.paths import (
    DATA_CATEGORIES,
    get_archive_dir,
    get_data_dir,
    get_data_home,
    get_partition_path,
)

__all__ = [
    "DATA_CATEGORIES",
    "archive_old_data",
    "get_archive_dir",
    "get_data_dir",
    "get_data_home",
    "get_partition_path",
    "read_parquet",
    "scan_parquet",
    "write_parquet",
]
