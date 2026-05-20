"""Screening pipeline: Phase 1 hard filtering + Phase 2 weighted scoring (Issues #94, #95).

Phase 1 (PRD 3.2.1 / 3.2.2):
  - 4 must-satisfy hard conditions with dynamic threshold adjustment
  - 5 optional bonus conditions

Phase 2 (PRD 3.2.3 / 3.3):
  - Weighted Breakout_Score composite (MVP factor weights from PRD 3.1.1)
  - GICS Sector / Industry dedup (Sector cap <= 3, Industry cap <= 2)
  - Top 30 -> dedup -> Top 20 output
"""

from alphascreener.screening.phase1 import (
    compute_filter_rate,
    hard_filter,
)
from alphascreener.screening.phase2 import (
    DEFAULT_FINAL_N,
    DEFAULT_INDUSTRY_CAP,
    DEFAULT_SECTOR_CAP,
    DEFAULT_TOP_N,
    MVP_WEIGHTS,
    apply_industry_dedup,
    compute_breakout_score,
    phase2_pipeline,
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
    # Phase 1
    "COOLDOWN_DAYS",
    "DEFAULT_THRESHOLDS",
    "DynamicThreshold",
    "FILTER_RATE_BANDS",
    "MAX_RELAXATION_PCT",
    "STEP_PCT",
    "compute_filter_rate",
    "hard_filter",
    # Phase 2
    "DEFAULT_FINAL_N",
    "DEFAULT_TOP_N",
    "DEFAULT_SECTOR_CAP",
    "DEFAULT_INDUSTRY_CAP",
    "MVP_WEIGHTS",
    "apply_industry_dedup",
    "compute_breakout_score",
    "phase2_pipeline",
]
