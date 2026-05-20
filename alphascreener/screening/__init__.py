"""Phase 1 hard filtering and dynamic threshold adjustment (Issue #94).

Must-satisfy conditions (4):
  - MOM_5D > 0
  - VOL_ANOMALY = 1 OR MFI_14 > 40
  - ATR_RATIO < 0.8
  - RSI_14 in [25, 75]

Optional bonus conditions (5):
  - BB_SQUEEZE = 1, PTH > 0.90, CMF_21 > 0, PEAD_FLAG = 1, INSIDER_BUY = 1

Reference: PRD 3.2.1 / 3.2.2.
"""

from alphascreener.screening.phase1 import (
    compute_filter_rate,
    hard_filter,
)
from alphascreener.screening.threshold import (
    COOLDOWN_DAYS,
    DEFAULT_THRESHOLDS,
    FILTER_RATE_BANDS,
    MAX_RELAXATION_PCT,
    STEP_PCT,
    DynamicThreshold,
)

__all__ = [
    "COOLDOWN_DAYS",
    "DEFAULT_THRESHOLDS",
    "DynamicThreshold",
    "FILTER_RATE_BANDS",
    "MAX_RELAXATION_PCT",
    "STEP_PCT",
    "compute_filter_rate",
    "hard_filter",
]
