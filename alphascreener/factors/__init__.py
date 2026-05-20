"""Factor computation engine (Issue #93).

Provides 14 factor formulas, z-score normalisation, missing-data handling,
and chunked-streaming computation via polars.

Reference: PRD 3.1.1 / 3.1.2 / 3.1.3 / 3.1.4.
"""

from alphascreener.factors.engine import (
    FactorEngine,
    compute_factors,
    normalize_factors,
    process_chunk,
)
from alphascreener.factors.formulas import (
    FACTOR_NAMES,
    MOMENTUM_FACTORS,
    VOLATILITY_FACTORS,
    MONEY_FLOW_FACTORS,
    TECHNICAL_FACTORS,
    FUNDAMENTAL_FACTORS,
    compute_all_technical_factors,
    compute_atr_ratio,
    compute_bb_squeeze,
    compute_cmf_21,
    compute_golden_cross,
    compute_insider_buy,
    compute_macd_cross,
    compute_mfi_14,
    compute_mom_5d,
    compute_mom_slope,
    compute_pead_flag,
    compute_pth,
    compute_rev_accel,
    compute_rsi_oversold,
    compute_vol_anomaly,
)

__all__ = [
    # Engine
    "FactorEngine",
    "compute_factors",
    "normalize_factors",
    "process_chunk",
    # Formulas
    "FACTOR_NAMES",
    "MOMENTUM_FACTORS",
    "VOLATILITY_FACTORS",
    "MONEY_FLOW_FACTORS",
    "TECHNICAL_FACTORS",
    "FUNDAMENTAL_FACTORS",
    "compute_all_technical_factors",
    "compute_mom_5d",
    "compute_pth",
    "compute_mom_slope",
    "compute_bb_squeeze",
    "compute_atr_ratio",
    "compute_mfi_14",
    "compute_cmf_21",
    "compute_vol_anomaly",
    "compute_rsi_oversold",
    "compute_macd_cross",
    "compute_golden_cross",
    "compute_pead_flag",
    "compute_insider_buy",
    "compute_rev_accel",
]
