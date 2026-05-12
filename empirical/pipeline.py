"""End-to-end pipeline orchestration.

Modes:
  --synthetic   generate panel data consistent with Eqs (1)-(2), then estimate
                them, recovering the planted parameters as a sanity check.
  --live        run on real data: yfinance returns + EDGAR 8-Ks + dictionary
                or LLM sentiment + Stage 1/2/3 estimators + LLM-era Chow break.

The synthetic mode is the integration test for the formal model + estimators.
The live mode tests Hypothesis H1 and the central compression claim on real data.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# allow `python3 pipeline.py` from any cwd
sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.halflife import H_MAX, estimate_panel_halflives  # noqa: E402
from src.regressions import (                              # noqa: E402
    test_h1_cross_sectional,
    test_llm_era_compression,
    simex_correction,
)


# -----------------------------------------------------------------------------
# Synthetic data generator: simulate a world that obeys Equations (1) and (2).
# -----------------------------------------------------------------------------

def simulate_panel(
    n_firms: int = 300,
    n_months: int = 36,
    headlines_per_firm_month: int = 10,
    seed: int = 42,
    # Planted parameters (we should recover these):
    beta_0_mean: float = 0.020,    # 200bps per unit sentiment at h=1 (clean SNR)
    alpha_true: float = 0.5,        # attention elasticity of hazard
    phi_true: float = 0.8,          # friction elasticity of hazard
    gamma_true: float = 0.3,        # competition elasticity of hazard
    lambda_baseline: float = 0.30,  # baseline hazard => tau ~ 2.3 days
    sigma_noise: float = 0.005,     # 50bps daily idiosync. noise (sanity-check SNR)
) -> pd.DataFrame:
    """Generate a headline panel obeying the formal model."""
    rng = np.random.default_rng(seed)
    months = pd.date_range("2023-01-31", periods=n_months, freq="ME")

    rows = []
    for firm in range(n_firms):
        log_attention = rng.normal(0, 1)
        log_friction = rng.normal(0, 1)
        size = rng.normal(0, 1)
        bm = rng.normal(0, 1)
        beta_0_firm = beta_0_mean * (1 + 0.3 * rng.normal())

        for m_idx, month_end in enumerate(months):
            log_competition = np.log1p(m_idx)

            log_lambda = (np.log(lambda_baseline)
                          + alpha_true * log_attention
                          + gamma_true * log_competition
                          - phi_true * log_friction)
            lam = np.exp(log_lambda)

            for k in range(headlines_per_firm_month):
                date = month_end - pd.Timedelta(days=rng.integers(1, 28))
                S = rng.uniform(-1, 1)
                noise = rng.normal(0, sigma_noise, size=H_MAX)
                drift = np.array([
                    beta_0_firm * S * np.exp(-(h + 1) * lam)
                    for h in range(H_MAX)
                ])
                rets = drift + noise
                row = {
                    "firm_id": f"F{firm:04d}",
                    "date": pd.Timestamp(date),
                    "sentiment": S,
                    "log_attention": log_attention,
                    "log_friction": log_friction,
                    "log_competition": log_competition,
                    "size": size,
                    "bm": bm,
                    "industry": f"IND{firm % 12}",
                }
                for h in range(H_MAX):
                    row[f"ret_h{h+1}"] = rets[h]
                rows.append(row)

    return pd.DataFrame(rows)


def synthetic_mode():
    print("=" * 60)
    print("ASL pipeline — synthetic mode (planted-parameter recovery)")
    print("=" * 60)

    print("\n[1/3] Generating synthetic headline panel...")
    headlines = simulate_panel()
    print(f"      n_headlines = {len(headlines):,}, n_firms = {headlines['firm_id'].nunique()}")

    print("\n[2/3] Estimating firm-month half-lives (Stage 1 + Stage 2)...")
    panel_for_hl = headlines[["firm_id", "date", "sentiment",
                              *[f"ret_h{h+1}" for h in range(H_MAX)]]]
    tau_panel = estimate_panel_halflives(panel_for_hl)
    print(f"      n_firm_months estimated = {len(tau_panel):,}")
    print(f"      median tau_hat = {tau_panel['tau'].median():.2f} days")

    print("\n[3/3] Stage 3: testing H1 (alpha > 0, phi > 0)...")
    firm_month_covars = (
        headlines.assign(month_end=lambda d: d["date"] + pd.offsets.MonthEnd(0))
        .groupby(["firm_id", "month_end"])
        .agg(attention=("log_attention", lambda s: np.exp(s.iloc[0])),
             competition=("log_competition", lambda s: np.exp(s.iloc[0])),
             friction=("log_friction", lambda s: np.exp(s.iloc[0])),
             size=("size", "first"),
             bm=("bm", "first"),
             industry=("industry", "first"))
        .reset_index()
    )
    firm_month_covars["mom"] = 0.0
    firm_month_covars["rev"] = 0.0

    panel = tau_panel.merge(firm_month_covars, on=["firm_id", "month_end"], how="inner")
    print(f"      n_obs in Stage 3 panel = {len(panel):,}")

    result = test_h1_cross_sectional(panel, time_effects=False, industry_effects=True)
    print()
    print("Recovered parameters:")
    print(f"  alpha_hat = {result.coef_attention:+.3f}  (true = +0.500, SE = {result.se_attention:.3f})")
    print(f"  phi_hat   = {result.coef_friction:+.3f}  (true = +0.800, SE = {result.se_friction:.3f})")
    print(f"  gamma_hat = {result.coef_competition:+.3f}  (true = +0.300, SE = {result.se_competition:.3f})")
    print(f"  R^2 within = {result.r2_within:.3f}, n = {result.n_obs}")

    print("\nIf alpha_hat and phi_hat are positive and within 2 SE of the true ")
    print("values, the estimators are consistent — the empirical pipeline is wired.\n")


# -----------------------------------------------------------------------------
# Live mode (real data)
# -----------------------------------------------------------------------------

def build_headline_returns_panel(
    headlines: pd.DataFrame,
    returns: pd.DataFrame,
    h_max: int = H_MAX,
) -> pd.DataFrame:
    """Attach forward 1-day returns r_{t+1}, ..., r_{t+H} to each headline.

    For each headline at time t (filing timestamp), find the firm's next
    trading day in `returns` and read ret on day t+1, t+2, ..., t+H_MAX.
    Single-period (not cumulative) returns — matches the synthetic generator
    and Stage 1's per-horizon OLS.

    Returns: firm_id, date (next trading day after filing), sentiment, ret_h1..ret_hH.
    """
    rets = returns.copy()
    rets["date"] = pd.to_datetime(rets["date"])
    rets = rets.sort_values(["firm_id", "date"]).reset_index(drop=True)

    forward = []
    for fid, fr in rets.groupby("firm_id"):
        fr = fr.set_index("date")["ret"].sort_index()
        ftbl = pd.DataFrame(index=fr.index)
        for h in range(1, h_max + 1):
            ftbl[f"ret_h{h}"] = fr.shift(-h)
        ftbl["firm_id"] = fid
        forward.append(ftbl.reset_index())
    forward = pd.concat(forward, ignore_index=True)

    headlines = headlines.copy()
    headlines["timestamp"] = pd.to_datetime(headlines["timestamp"])

    out_rows = []
    for fid, hgrp in headlines.groupby("firm_id"):
        firm_fwd = forward[forward["firm_id"] == fid].sort_values("date").reset_index(drop=True)
        if firm_fwd.empty:
            continue
        firm_dates = firm_fwd["date"].values
        for _, h in hgrp.iterrows():
            ts = pd.Timestamp(h["timestamp"]).normalize().to_datetime64()
            idx = np.searchsorted(firm_dates, ts)
            if idx >= len(firm_dates):
                continue
            row_fwd = firm_fwd.iloc[idx]
            new_row = {
                "firm_id": fid,
                "date": row_fwd["date"],
                "sentiment": h.get("sentiment", 0.0),
            }
            for hh in range(1, h_max + 1):
                new_row[f"ret_h{hh}"] = row_fwd[f"ret_h{hh}"]
            out_rows.append(new_row)
    out = pd.DataFrame(out_rows)
    return out.dropna(subset=[f"ret_h{h}" for h in range(1, h_max + 1)])


def live_mode(
    start: str = "2020-01-01",
    end: str = "2025-04-01",
    n_firms: int = 50,
    sentiment_backend: str = "dictionary",
    break_date: str = "2023-03-15",
):
    """Run the H1 + compression pipeline on real data.

    Stages:
      1. Universe: top-N S&P 500 by market cap (proxied by Wiki order).
      2. Returns: yfinance daily, Jan 2020 - Apr 2025.
      3. News: SEC EDGAR 8-Ks within the window.
      4. Sentiment: LM dictionary by default; pass `--llm` for Anthropic.
      5. Half-life panel: Stage 1 (per-horizon OLS) + Stage 2 (NLS).
      6. Stage 3 covariates: SVI for attention, Corwin-Schultz for friction.
      7. H1 cross-section + SIMEX correction.
      8. LLM-era Chow break test (the central compression claim).
    """
    from src.data_loader import (
        load_sp500_universe, load_returns, load_news_headlines,
        load_attention_wikipedia, load_attention_svi, load_microstructure,
    )
    from src.sentiment import score_headlines_batch

    print("=" * 60)
    print(f"ASL pipeline — live mode  (n_firms={n_firms}, {start} .. {end})")
    print(f"sentiment backend = {sentiment_backend}, break_date = {break_date}")
    print("=" * 60)

    out_dir = Path(__file__).resolve().parent / "output"
    out_dir.mkdir(exist_ok=True)

    print("\n[1/8] Universe...")
    universe = load_sp500_universe()
    tickers = universe["ticker"].head(n_firms).tolist()
    print(f"      using {len(tickers)} tickers: {tickers[:6]}{' ...' if len(tickers) > 6 else ''}")

    print(f"\n[2/8] Returns (yfinance, {start} .. {end})...")
    rets = load_returns(start, end, tickers=tickers, cache_key=f"live_n{n_firms}")
    print(f"      rows = {len(rets):,}, firms covered = {rets['firm_id'].nunique()}")

    print(f"\n[3/8] EDGAR 8-Ks ({start} .. {end})...")
    news = load_news_headlines(start, end, tickers=tickers, fetch_bodies=True)
    print(f"      filings = {len(news):,}, firms with filings = {news['firm_id'].nunique()}")

    print(f"\n[4/8] Sentiment scoring (backend = {sentiment_backend})...")
    scored = score_headlines_batch(news, backend=sentiment_backend)
    nonzero = (scored["sentiment"].abs() > 0).sum()
    print(f"      scored = {len(scored):,}, nonzero sentiment = {nonzero:,}")

    print("\n[5/8] Building headline-level forward-returns panel...")
    panel_for_hl = build_headline_returns_panel(scored, rets, h_max=H_MAX)
    print(f"      panel rows = {len(panel_for_hl):,}")
    if panel_for_hl.empty:
        print("      [abort] no headlines aligned to forward-returns. Check date ranges.")
        return

    print("\n[6/8] Estimating firm-month half-lives...")
    tau_panel = estimate_panel_halflives(
        panel_for_hl[["firm_id", "date", "sentiment",
                      *[f"ret_h{h+1}" for h in range(H_MAX)]]],
        window_days=504,  # 2-year rolling window (8-K corpus is thin)
    )
    print(f"      n_firm_months estimated = {len(tau_panel):,}")
    if not tau_panel.empty:
        print(f"      median tau_hat = {tau_panel['tau'].replace(np.inf, np.nan).median():.2f} days")
    tau_panel.to_parquet(out_dir / "tau_panel.parquet")

    print("\n[7/8] Stage 3 covariates: Wikipedia attention + Corwin-Schultz friction...")
    micro = load_microstructure(start, end, tickers=tickers)
    micro["month_end"] = pd.to_datetime(micro["date"]) + pd.offsets.MonthEnd(0)
    friction_fm = (micro.groupby(["firm_id", "month_end"])["spread_cs"]
                   .mean().reset_index().rename(columns={"spread_cs": "friction"}))

    attention_fm = pd.DataFrame()
    try:
        wiki = load_attention_wikipedia(tickers, start, end,
                                        name_lookup=dict(zip(universe["ticker"],
                                                             universe["name"])))
        if not wiki.empty:
            wiki["month_end"] = pd.to_datetime(wiki["date"]) + pd.offsets.MonthEnd(0)
            attention_fm = (wiki.groupby(["ticker", "month_end"])["views"].mean()
                            .reset_index()
                            .rename(columns={"ticker": "firm_id", "views": "attention"}))
            print(f"      attention source: Wikipedia page views ({wiki['ticker'].nunique()} firms covered)")
    except Exception as e:
        print(f"      [warn] Wikipedia attention failed: {e}")

    if attention_fm.empty:
        try:
            svi = load_attention_svi(tickers, start, end)
            svi["month_end"] = pd.to_datetime(svi["week_end"]) + pd.offsets.MonthEnd(0)
            attention_fm = (svi.groupby(["ticker", "month_end"])["svi"].mean()
                            .reset_index()
                            .rename(columns={"ticker": "firm_id", "svi": "attention"}))
            print("      attention source: Google SVI (fallback)")
        except Exception as e:
            print(f"      [warn] SVI also failed ({e}); using filing-rate proxy.")
            attention_fm = (scored.assign(month_end=lambda d: pd.to_datetime(d["timestamp"])
                                          + pd.offsets.MonthEnd(0))
                            .groupby(["firm_id", "month_end"])
                            .size().reset_index(name="attention"))

    # Competition proxy: trailing 60-day count of 8-K filings across the universe
    # in the same GICS sector. For first pass: use a simple time index that steps
    # at foundation-model release dates.
    release_dates = [pd.Timestamp("2022-11-30"),  # ChatGPT
                     pd.Timestamp("2023-03-14"),  # GPT-4
                     pd.Timestamp("2024-05-13")]  # GPT-4o
    me_series = tau_panel["month_end"]
    competition_fm = pd.DataFrame({
        "month_end": me_series.unique(),
    })
    competition_fm["competition"] = competition_fm["month_end"].apply(
        lambda d: 1 + sum(1 for r in release_dates if d >= r)
    )

    panel = (tau_panel
             .merge(attention_fm, on=["firm_id", "month_end"], how="inner")
             .merge(friction_fm, on=["firm_id", "month_end"], how="inner")
             .merge(competition_fm, on=["month_end"], how="left"))

    # Industry from universe; controls.
    panel = panel.merge(universe[["ticker", "sector"]]
                        .rename(columns={"ticker": "firm_id", "sector": "industry"}),
                        on="firm_id", how="left")
    panel["size"] = 0.0  # placeholder; wire to market cap if you have it
    panel["bm"] = 0.0
    panel["mom"] = 0.0
    panel["rev"] = 0.0

    panel = panel[panel["tau"].replace(np.inf, np.nan).notna()]
    panel = panel[(panel["tau"] > 0) & (panel["tau"] < 365)]
    print(f"      Stage 3 panel rows = {len(panel):,}")
    panel.to_parquet(out_dir / "stage3_panel.parquet")

    if len(panel) < 50:
        print("      [abort] too few firm-months for Stage 3. Increase n_firms or window.")
        return

    print("\n[8/8] Running H1 cross-section + SIMEX + Chow compression test...")
    h1 = test_h1_cross_sectional(panel, time_effects=True, industry_effects=True)
    print()
    print("H1 cross-sectional (clustered SE on firm + month):")
    print(f"  alpha_hat = {h1.coef_attention:+.3f}  (SE {h1.se_attention:.3f})")
    print(f"  phi_hat   = {h1.coef_friction:+.3f}  (SE {h1.se_friction:.3f})")
    print(f"  gamma_hat = {h1.coef_competition:+.3f}  (SE {h1.se_competition:.3f})  (note: time-FE absorbs most)")
    print(f"  R^2 within = {h1.r2_within:.3f}, n = {h1.n_obs}")

    sx = simex_correction(panel, sigma_log_tau=0.30, n_replicates=30,
                          time_effects=True, industry_effects=True)
    print()
    print("SIMEX attenuation correction (sigma_log_tau = 0.30):")
    print(f"  alpha:  naive {sx['naive_alpha']:+.3f}  ->  corrected {sx['corrected_alpha']:+.3f}")
    print(f"  phi  :  naive {sx['naive_phi']:+.3f}  ->  corrected {sx['corrected_phi']:+.3f}")
    print(f"  gamma:  naive {sx['naive_gamma']:+.3f}  ->  corrected {sx['corrected_gamma']:+.3f}")

    comp = test_llm_era_compression(panel, break_date=break_date)
    print()
    print(f"LLM-era compression test (break = {break_date}):")
    print(f"  median tau pre  = {comp['median_tau_pre']:.2f}d  (n = {comp['n_pre']})")
    print(f"  median tau post = {comp['median_tau_post']:.2f}d  (n = {comp['n_post']})")
    print(f"  delta log(tau)  = {comp['delta_log_tau']:+.3f}  "
          f"(compression {comp['compression_pct']:+.1f}%)")
    print(f"  Welch t = {comp['t_stat']:+.2f}, p = {comp['p_value']:.4f}")
    print(f"  Chow F  = {comp['chow_f']:.2f}, p = {comp['chow_p']:.4f}")
    if comp.get("by_attention_quartile"):
        print("  By attention quartile (low -> high):")
        for q in sorted(comp["by_attention_quartile"]):
            r = comp["by_attention_quartile"][q]
            print(f"    Q{q+1}: {r['median_tau_pre']:.2f}d -> {r['median_tau_post']:.2f}d "
                  f"(delta log = {r['delta_log_tau']:+.3f}, n={r['n_pre']}/{r['n_post']})")

    results = {
        "h1": {
            "alpha_hat": h1.coef_attention, "se_alpha": h1.se_attention,
            "phi_hat": h1.coef_friction, "se_phi": h1.se_friction,
            "gamma_hat": h1.coef_competition, "se_gamma": h1.se_competition,
            "r2_within": h1.r2_within, "n_obs": h1.n_obs,
        },
        "simex": {k: v for k, v in sx.items() if k != "trace"},
        "compression": comp,
        "config": {
            "start": start, "end": end, "n_firms": n_firms,
            "sentiment_backend": sentiment_backend, "break_date": break_date,
        },
    }
    (out_dir / "results.json").write_text(json.dumps(results, indent=2, default=str))
    print(f"\nResults written to {out_dir / 'results.json'}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--synthetic", action="store_true",
                    help="Run end-to-end on simulated data (sanity check).")
    ap.add_argument("--live", action="store_true",
                    help="Run on real data.")
    ap.add_argument("--start", default="2020-01-01")
    ap.add_argument("--end", default="2025-04-01")
    ap.add_argument("--n-firms", type=int, default=50)
    ap.add_argument("--llm", action="store_true",
                    help="Use Anthropic LLM for sentiment instead of LM dictionary.")
    ap.add_argument("--break-date", default="2023-03-15",
                    help="LLM-era break for compression test (default: GPT-4 release).")
    args = ap.parse_args()

    if args.synthetic:
        synthetic_mode()
    elif args.live:
        live_mode(
            start=args.start, end=args.end, n_firms=args.n_firms,
            sentiment_backend="anthropic" if args.llm else "dictionary",
            break_date=args.break_date,
        )
    else:
        ap.print_help()


if __name__ == "__main__":
    main()
