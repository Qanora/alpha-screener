"""End-to-end: download SP500 data, screen, push results to Feishu.

Usage:
    uv run python scripts/run_and_push.py [--tickers N] [--skip-download]
"""

import argparse
import sys
from datetime import date, timedelta
from pathlib import Path

import polars as pl

# Add project root to path so imports work
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def fetch_sp500_tickers() -> list[str]:
    """Return a representative list of SP500 tickers across all sectors.

    Top 150 tickers by market cap / representativeness across GICS sectors.
    This avoids external network calls when Wikipedia is unavailable.
    """
    return [
        # Technology
        "AAPL",
        "MSFT",
        "NVDA",
        "GOOGL",
        "META",
        "AVGO",
        "ORCL",
        "CRM",
        "ADBE",
        "AMD",
        "INTC",
        "QCOM",
        "TXN",
        "AMAT",
        "INTU",
        "IBM",
        "CSCO",
        "NOW",
        "LRCX",
        "ADI",
        "KLAC",
        "APH",
        "SNPS",
        "CDNS",
        "MSI",
        "ROP",
        "ADSK",
        "FTNT",
        "PANW",
        "CRWD",
        "PLTR",
        "MU",
        "ANET",
        "DELL",
        "HPQ",
        # Communication Services
        "GOOG",
        "NFLX",
        "DIS",
        "CMCSA",
        "VZ",
        "T",
        "TMUS",
        "CHTR",
        # Consumer Cyclical
        "AMZN",
        "TSLA",
        "HD",
        "MCD",
        "NKE",
        "LOW",
        "SBUX",
        "BKNG",
        "TJX",
        "CMG",
        "ORLY",
        "AZO",
        "MAR",
        "ABNB",
        "ROST",
        "TGT",
        "YUM",
        # Consumer Defensive
        "WMT",
        "PG",
        "KO",
        "PEP",
        "COST",
        "PM",
        "MDLZ",
        "MO",
        "CL",
        "KMB",
        "EL",
        "SYY",
        "GIS",
        "KHC",
        # Healthcare
        "JNJ",
        "UNH",
        "PFE",
        "ABBV",
        "MRK",
        "TMO",
        "ABT",
        "DHR",
        "BMY",
        "LLY",
        "AMGN",
        "GILD",
        "REGN",
        "ISRG",
        "VRTX",
        "CI",
        "CVS",
        "ZTS",
        # Financials
        "BRK-B",
        "JPM",
        "V",
        "MA",
        "BAC",
        "WFC",
        "MS",
        "GS",
        "BLK",
        "C",
        "SCHW",
        "AXP",
        "SPGI",
        "MMC",
        "CB",
        "PYPL",
        "USB",
        "PNC",
        "TFC",
        # Industrials
        "GE",
        "CAT",
        "UNP",
        "RTX",
        "HON",
        "LMT",
        "UPS",
        "BA",
        "DE",
        "ETN",
        "WM",
        "GD",
        "ITW",
        "FDX",
        "NSC",
        "PH",
        "NOC",
        "CSX",
        "EMR",
        # Energy
        "XOM",
        "CVX",
        "COP",
        "EOG",
        "SLB",
        "MPC",
        "PSX",
        "OXY",
        "WMB",
        "KMI",
        # Utilities
        "NEE",
        "SO",
        "DUK",
        "D",
        "AEP",
        "EXC",
        "SRE",
        "ED",
        # Real Estate
        "PLD",
        "AMT",
        "CCI",
        "SPG",
        "EQIX",
        "O",
        "PSA",
        # Materials
        "LIN",
        "FCX",
        "APD",
        "NEM",
        "SHW",
        "ECL",
        "CTVA",
    ]


def download_data(tickers: list[str], end_date: date) -> pl.DataFrame:
    """Download OHLCV for tickers and write Parquet partitions."""
    import pandas as pd
    import yfinance as yf

    from alphascreener.data import write_parquet

    lookback = 30  # download last 30 calendar days of data
    start_date = (end_date - timedelta(days=lookback)).isoformat()

    print(f"\n[1/5] Downloading OHLCV for {len(tickers)} tickers from yfinance ...")
    print(f"      Date range: {start_date} → {end_date.isoformat()}")

    all_frames = []
    batch_size = 50
    for i in range(0, len(tickers), batch_size):
        batch = tickers[i : i + batch_size]
        ticker_str = " ".join(batch)
        print(f"      Batch {i // batch_size + 1}: {len(batch)} tickers ...", end=" ", flush=True)
        try:
            pd_df = yf.download(
                ticker_str,
                start=start_date,
                end=(end_date + timedelta(days=1)).isoformat(),
                threads=False,
                progress=False,
                auto_adjust=True,
            )
            if pd_df is not None and not pd_df.empty:
                # Convert to polars
                records = []
                if isinstance(pd_df.columns, pd.core.indexes.multi.MultiIndex):
                    for t in batch:
                        try:
                            sub = pd_df.xs(t, axis=1, level=1).dropna(how="all")
                        except KeyError:
                            continue
                        for idx, row in sub.iterrows():
                            dt_val = (
                                idx.date()
                                if hasattr(idx, "date")
                                else date.fromisoformat(str(idx)[:10])
                            )
                            records.append(
                                {
                                    "ticker": t,
                                    "dt": dt_val,
                                    "open": float(row.get("Open", 0) or 0),
                                    "high": float(row.get("High", 0) or 0),
                                    "low": float(row.get("Low", 0) or 0),
                                    "close": float(row.get("Close", 0) or 0),
                                    "volume": int(row.get("Volume", 0) or 0),
                                }
                            )
                else:
                    # Single ticker
                    t = batch[0]
                    for idx, row in pd_df.dropna(how="all").iterrows():
                        dt_val = (
                            idx.date()
                            if hasattr(idx, "date")
                            else date.fromisoformat(str(idx)[:10])
                        )
                        records.append(
                            {
                                "ticker": t,
                                "dt": dt_val,
                                "open": float(row.get("Open", 0) or 0),
                                "high": float(row.get("High", 0) or 0),
                                "low": float(row.get("Low", 0) or 0),
                                "close": float(row.get("Close", 0) or 0),
                                "volume": int(row.get("Volume", 0) or 0),
                            }
                        )
                if records:
                    all_frames.append(
                        pl.DataFrame(records).with_columns(pl.col("dt").cast(pl.Date))
                    )
                print(f"{len(records)} rows")
            else:
                print("empty")
        except Exception as e:
            print(f"failed: {e}")

    if not all_frames:
        raise RuntimeError("No data downloaded from yfinance")

    combined = pl.concat(all_frames)
    print(f"      Total: {combined.height} rows, {combined['ticker'].n_unique()} tickers")

    # Write Parquet partitions
    print("\n[2/5] Writing OHLCV Parquet partitions ...")
    write_parquet(combined, "ohlcv")
    print(f"      Written to ~/.alphascreener/data/ohlcv/")

    return combined


def fetch_meta(tickers: list[str]) -> pl.DataFrame:
    """Fetch sector/industry metadata for tickers."""
    import yfinance as yf
    from datetime import datetime, UTC

    records = []
    for i, ticker in enumerate(tickers):
        try:
            t = yf.Ticker(ticker)
            info = t.info
            if info:
                records.append(
                    {
                        "ticker": ticker,
                        "sector": info.get("sector", ""),
                        "industry": info.get("industry", ""),
                        "market_cap": info.get("marketCap") or 0.0,
                        "index_source": "SP500",
                        "refreshed_at": datetime.now(UTC).isoformat(),
                    }
                )
        except Exception:
            pass
        if (i + 1) % 20 == 0:
            print(f"      Meta: {i + 1}/{len(tickers)} ...")

    return pl.DataFrame(records)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--tickers", type=int, default=100, help="Number of top SP500 tickers to download"
    )
    parser.add_argument(
        "--skip-download", action="store_true", help="Skip download, use cached data"
    )
    parser.add_argument("--top", type=int, default=20, help="Number of top candidates to output")
    args = parser.parse_args()

    today = date.today()
    print(f"Alpha Screener — End-to-End Run")
    print(f"  Date    : {today.isoformat()}")
    print(f"  Tickers : {args.tickers}")
    print(f"  Top N   : {args.top}")

    tickers = fetch_sp500_tickers()
    print(f"  SP500   : {len(tickers)} tickers found")

    # Take the first N tickers
    tickers = tickers[: args.tickers]

    if not args.skip_download:
        ohlcv_df = download_data(tickers, today)
        print(f"      Latest trading day: {ohlcv_df['dt'].max()}")

        # Fetch and cache universe metadata
        print("\n[3/5] Fetching sector/industry metadata ...")
        meta_df = fetch_meta(tickers)
        if meta_df.height > 0:
            from alphascreener.universe.meta import write_meta_cache

            write_meta_cache(meta_df)
            print(f"      Written {meta_df.height} metadata records")
    else:
        print("\n[skip] Using cached OHLCV data")

    # Run screening
    print("\n[4/5] Running screening pipeline ...")
    from alphascreener.data.io import scan_parquet
    from alphascreener.factors.engine import compute_factors
    from alphascreener.screening.phase1 import hard_filter_with_fallback
    from alphascreener.screening.phase2 import phase2_pipeline

    ohlcv_lf = scan_parquet("ohlcv")
    ohlcv = ohlcv_lf.collect()
    latest_date = ohlcv["dt"].max()
    df = ohlcv.filter(pl.col("dt") == latest_date)

    # Dedup: keep the row with the most complete OHLCV for each ticker
    df = (
        df.with_columns(
            (
                pl.col("open").is_not_null().cast(int) + pl.col("close").is_not_null().cast(int)
            ).alias("_completeness")
        )
        .sort("_completeness", descending=True)
        .unique(subset=["ticker"], keep="first")
        .drop("_completeness")
    )

    print(f"      Data date : {latest_date}")
    print(f"      Tickers   : {df.height}")

    # Load universe meta for sector/industry
    meta = None
    try:
        from alphascreener.universe.meta import read_meta_cache

        meta = read_meta_cache().select(["ticker", "sector", "industry"]).collect()
    except Exception:
        pass

    factors = compute_factors(df, dt=latest_date)

    if meta is not None and meta.height > 0:
        factors = factors.join(meta, on="ticker", how="left")

    filtered, relaxed_used = hard_filter_with_fallback(factors)
    passed = filtered.filter(pl.col("pass_phase1"))
    relax_note = " (relaxed)" if relaxed_used else ""
    print(f"      Phase 1 pass : {passed.height} / {filtered.height}{relax_note}")

    if passed.height == 0:
        print("      No tickers passed Phase 1 (even after relaxation) — aborting.")
        return
    else:
        results = phase2_pipeline(passed, n_final=args.top)
        print(f"      Phase 2 output: {results.height} candidates")

    # Print results
    print(f"\n      Top {min(args.top, results.height)} candidates:")
    print(f"      {'#':<4} {'Ticker':<8} {'Score':<10}")
    print(f"      {'-' * 22}")
    for i, row in enumerate(results.select(["ticker", "breakout_score"]).iter_rows(named=True)):
        print(f"      {i + 1:<4} {row['ticker']:<8} {row['breakout_score']:.4f}")

    # Push to Feishu
    print("\n[5/5] Pushing results to Feishu ...")
    from alphascreener.config import Settings
    from alphascreener.feishu.card import CardData
    from alphascreener.feishu.push import push_daily_report

    settings = Settings()

    if (
        not settings.feishu_app_id
        or not settings.feishu_app_secret
        or not settings.feishu_target_openid
    ):
        print("      WARNING: Feishu credentials not configured, skipping push")
        return

    # Build top_five list for the card
    top_five = []
    for i, row in enumerate(
        results.select(["ticker", "breakout_score"]).head(5).iter_rows(named=True)
    ):
        top_five.append(
            {
                "rank": i + 1,
                "ticker": row["ticker"],
                "score": round(row["breakout_score"], 4),
            }
        )

    # Determine alerts summary
    n_total = df["ticker"].n_unique()
    n_passed = passed.height if passed.height > 0 else 0

    if n_passed == 0:
        alerts = "Phase 1: 0 tickers passed hard filters (thresholds may be too strict for current market)"
    else:
        alerts = "ok"

    data = CardData(
        report_date=today.isoformat(),
        total_symbols=n_total,
        coarse_pass=n_passed,
        refine_count=n_passed,
        top_five=top_five,
        p20_pure=None,
        p20_llm=None,
        lift_pure=None,
        lift_llm=None,
        base_rate=None,
        win_rate=None,
        sharpe=None,
        avg_return=None,
        daily_cost=None,
        monthly_cost=None,
        alerts_summary=alerts,
    )

    result = push_daily_report(data)
    print(f"      Push result: {result.value}")


if __name__ == "__main__":
    main()
