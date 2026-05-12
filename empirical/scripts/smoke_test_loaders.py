"""Smoke test: verify load_returns and load_news_headlines on a small universe.

Run from empirical/:  python3 scripts/smoke_test_loaders.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data_loader import (
    load_returns,
    load_news_headlines,
    load_microstructure,
)


SMOKE_TICKERS = ["AAPL", "MSFT", "NVDA", "JPM", "XOM"]
START = "2023-01-01"
END = "2024-01-01"


def main():
    print("=" * 60)
    print("SMOKE TEST — data_loader.py against yfinance + EDGAR")
    print("=" * 60)

    print(f"\n[1/3] load_returns({SMOKE_TICKERS}, {START}, {END})")
    rets = load_returns(START, END, tickers=SMOKE_TICKERS, cache_key="smoke")
    print(f"      rows = {len(rets):,}")
    print(f"      firms covered = {rets['firm_id'].nunique()}")
    print(f"      date range = {rets['date'].min().date()} .. {rets['date'].max().date()}")
    print(f"      mean |ret| = {rets['ret'].abs().mean():.4f}")
    print(f"      sample row:\n{rets.head(2).to_string(index=False)}")

    print(f"\n[2/3] load_microstructure({SMOKE_TICKERS}, {START}, {END})")
    micro = load_microstructure(START, END, tickers=SMOKE_TICKERS)
    print(f"      rows = {len(micro):,}")
    print(f"      median spread_cs = {micro['spread_cs'].median():.5f}")
    print(f"      median dollar_volume = ${micro['dollar_volume'].median():,.0f}")

    print(f"\n[3/3] load_news_headlines({SMOKE_TICKERS}, {START}, {END}, fetch_bodies=True)")
    print("      (this hits SEC EDGAR — slow first run, fast on rerun via cache)")
    news = load_news_headlines(START, END, tickers=SMOKE_TICKERS, fetch_bodies=True)
    print(f"      rows = {len(news):,}")
    print(f"      firms with filings = {news['firm_id'].nunique()}")
    if len(news):
        print(f"      sample headline: {news.iloc[0]['headline']!r}")
        print(f"      sample lead    : {news.iloc[0]['lead_paragraph'][:200]!r}...")

    print("\nAll three loaders returned non-empty frames — schemas wired.\n")


if __name__ == "__main__":
    main()
