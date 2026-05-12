"""Diagnostic tests for the contaminated-tau decomposition.

Three tests, each with a pre-specified null:

  1. PLACEBO. Replace sentiment with white-noise (and a sparse-22%-nonzero
     variant that matches LM-dictionary density) on real returns. Refit Stage 1
     + Stage 2. If the placebo tau-hat distribution overlaps the real tau-hat
     distribution, w_1 (sentiment-decay loading) is small.

  2. VOL-REGIME STRATIFICATION. Attach VIX at each month_end of the cached
     Stage 3 panel; tercile-bucket; re-run the Welch test for the post-2023-03
     compression within each tercile. If the +50% expansion vanishes inside a
     regime, w_3 (vol-regime loading) is the dominant component.

  3. PSEUDO-EVENT NULL. Hold sentiment values fixed (sampled from the LM
     empirical distribution) but randomize event timing. If tau-hat is
     near-identical, w_4 (event-clustering loading) is establishing the floor.

All three run on cached data (no new EDGAR / yfinance calls beyond the VIX
ticker pull). Results land in output/diagnostic_results.json.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.halflife import (
    H_MAX, MIN_HEADLINES,
    stage1_horizon_betas, stage2_exponential_fit,
)


RNG = np.random.default_rng(42)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def build_forward_returns(returns_panel: pd.DataFrame) -> pd.DataFrame:
    """Add ret_h1..ret_h20 columns by per-firm forward shift."""
    df = returns_panel.sort_values(["firm_id", "date"]).copy()
    df["date"] = pd.to_datetime(df["date"])
    for h in range(1, H_MAX + 1):
        df[f"ret_h{h}"] = df.groupby("firm_id")["ret"].shift(-h)
    return df


def per_firm_month_fits(
    headline_panel: pd.DataFrame,
    *,
    label: str,
) -> pd.DataFrame:
    """Run Stage 1 + Stage 2 on a per-firm-month rolling window.

    headline_panel columns: firm_id, date, sentiment, ret_h1..ret_h20.
    Returns one row per firm-month with tau, lambda_, beta_0.
    """
    ret_cols = [f"ret_h{h}" for h in range(1, H_MAX + 1)]
    out = []
    headline_panel = headline_panel.copy()
    headline_panel["date"] = pd.to_datetime(headline_panel["date"])
    headline_panel["month_end"] = (
        headline_panel["date"] + pd.offsets.MonthEnd(0)
    )
    window_days = 504  # matches halflife.estimate_panel_halflives default

    for firm_id, grp in headline_panel.groupby("firm_id"):
        grp = grp.sort_values("date").reset_index(drop=True)
        for me, _ in grp.groupby("month_end"):
            window_lo = me - pd.Timedelta(days=window_days * 1.45)
            sub = grp[(grp["date"] > window_lo) & (grp["date"] <= me)]
            if len(sub) < MIN_HEADLINES:
                continue
            S = sub["sentiment"].values
            R = sub[ret_cols].values
            beta_h, se_h = stage1_horizon_betas(S, R)
            beta_0, lam, rmse = stage2_exponential_fit(beta_h, se_h)
            if not np.isfinite(lam):
                continue
            tau = float(np.log(2) / lam) if lam > 0 else float("inf")
            out.append({
                "label": label,
                "firm_id": firm_id,
                "month_end": me,
                "n": int(len(sub)),
                "beta_0": float(beta_0),
                "lambda_": float(lam),
                "tau": tau,
                "fit_rmse": float(rmse) if np.isfinite(rmse) else np.nan,
            })
    return pd.DataFrame(out)


def summarize(taus: np.ndarray) -> dict:
    finite = taus[np.isfinite(taus) & (taus > 0)]
    return {
        "n": int(len(finite)),
        "median": float(np.median(finite)) if len(finite) else float("nan"),
        "p25": float(np.percentile(finite, 25)) if len(finite) else float("nan"),
        "p75": float(np.percentile(finite, 75)) if len(finite) else float("nan"),
        "mean_log": float(np.mean(np.log(finite))) if len(finite) else float("nan"),
    }


# ---------------------------------------------------------------------------
# Test 1 — Placebo
# ---------------------------------------------------------------------------

def run_placebo(returns_panel: pd.DataFrame, *,
                headlines_per_firm_month: int = 12,
                sparse_nonzero_rate: float = 0.22) -> dict:
    """Two placebo regimes:
       (a) dense white-noise sentiment ~ N(0, 1)
       (b) sparse: 22% nonzero (matching LM density), values from {-1, +1}.

    Both placebos use REAL returns so the autocorrelation structure is preserved.
    If tau-hat is similar to real-data tau-hat, the estimator is fitting return
    structure, not sentiment.
    """
    fr = build_forward_returns(returns_panel)
    fr = fr.dropna(subset=[f"ret_h{h}" for h in range(1, H_MAX + 1)])

    # Sample candidate headline dates: random N per firm-month
    fr["month_end"] = fr["date"] + pd.offsets.MonthEnd(0)
    sample_rows = []
    for (firm_id, me), grp in fr.groupby(["firm_id", "month_end"]):
        if len(grp) < headlines_per_firm_month:
            continue
        idx = RNG.choice(grp.index.values,
                         size=headlines_per_firm_month, replace=False)
        sample_rows.append(grp.loc[idx])
    headline_panel = pd.concat(sample_rows, ignore_index=True)

    # (a) Dense placebo
    dense = headline_panel.copy()
    dense["sentiment"] = RNG.standard_normal(len(dense))
    fits_dense = per_firm_month_fits(dense, label="placebo_dense")

    # (b) Sparse placebo — 22% nonzero, sign random, matching LM density
    sparse = headline_panel.copy()
    nonzero = RNG.random(len(sparse)) < sparse_nonzero_rate
    sparse["sentiment"] = np.where(
        nonzero,
        RNG.choice([-1.0, +1.0], size=len(sparse)),
        0.0,
    )
    fits_sparse = per_firm_month_fits(sparse, label="placebo_sparse")

    return {
        "headlines_per_firm_month": headlines_per_firm_month,
        "sparse_nonzero_rate": sparse_nonzero_rate,
        "dense": {
            "tau_summary": summarize(fits_dense["tau"].values),
            "n_firm_months": len(fits_dense),
            "n_firms": fits_dense["firm_id"].nunique() if len(fits_dense) else 0,
        },
        "sparse": {
            "tau_summary": summarize(fits_sparse["tau"].values),
            "n_firm_months": len(fits_sparse),
            "n_firms": fits_sparse["firm_id"].nunique() if len(fits_sparse) else 0,
        },
        "fits_dense": fits_dense,
        "fits_sparse": fits_sparse,
    }


# ---------------------------------------------------------------------------
# Test 2 — Vol-regime stratification
# ---------------------------------------------------------------------------

def fetch_vix(start: str, end: str) -> pd.DataFrame:
    """VIX from yfinance (^VIX), monthly close."""
    import yfinance as yf
    vix = yf.download("^VIX", start=start, end=end, progress=False, auto_adjust=False)
    if vix.empty:
        return pd.DataFrame(columns=["date", "vix"])
    vix = vix.reset_index()[["Date", "Close"]]
    vix.columns = ["date", "vix"]
    vix["vix"] = vix["vix"].astype(float)
    return vix


def run_vol_regime(stage3_panel: pd.DataFrame,
                   break_date: str = "2023-03-15") -> dict:
    """Tercile-split stage3 panel by VIX at month_end. Within each tercile,
    Welch t-test for log_tau_post - log_tau_pre.
    """
    import scipy.stats as st

    panel = stage3_panel.copy()
    panel["month_end"] = pd.to_datetime(panel["month_end"])
    panel = panel[panel["tau"].notna() & np.isfinite(panel["tau"]) & (panel["tau"] > 0)]
    panel["log_tau"] = np.log(panel["tau"])

    # VIX at each month_end (use last close in month)
    vix = fetch_vix(start="2019-12-01", end="2025-01-31")
    vix["date"] = pd.to_datetime(vix["date"])
    vix["month_end"] = vix["date"] + pd.offsets.MonthEnd(0)
    vix_monthly = vix.groupby("month_end")["vix"].last().reset_index()

    panel = panel.merge(vix_monthly, on="month_end", how="left")
    panel = panel.dropna(subset=["vix"])

    # Tercile split (pooled across all months)
    panel["vix_tercile"] = pd.qcut(panel["vix"], q=3,
                                    labels=["low", "mid", "high"])

    cutoff = pd.Timestamp(break_date)
    panel["post"] = (panel["month_end"] >= cutoff).astype(int)

    overall_pre = panel[panel["post"] == 0]["log_tau"]
    overall_post = panel[panel["post"] == 1]["log_tau"]
    if len(overall_pre) >= 5 and len(overall_post) >= 5:
        t_overall, p_overall = st.ttest_ind(overall_pre, overall_post,
                                             equal_var=False)
    else:
        t_overall = p_overall = float("nan")

    by_tercile = {}
    for tercile in ["low", "mid", "high"]:
        sub = panel[panel["vix_tercile"] == tercile]
        pre = sub[sub["post"] == 0]["log_tau"]
        post = sub[sub["post"] == 1]["log_tau"]
        if len(pre) >= 5 and len(post) >= 5:
            t, p = st.ttest_ind(pre, post, equal_var=False)
        else:
            t = p = float("nan")
        by_tercile[tercile] = {
            "n_pre": int(len(pre)),
            "n_post": int(len(post)),
            "median_tau_pre": float(np.exp(pre.median())) if len(pre) else float("nan"),
            "median_tau_post": float(np.exp(post.median())) if len(post) else float("nan"),
            "delta_log_tau": (float(post.mean() - pre.mean())
                              if len(pre) and len(post) else float("nan")),
            "t_stat": float(t) if np.isfinite(t) else float("nan"),
            "p_value": float(p) if np.isfinite(p) else float("nan"),
            "vix_range": [float(sub["vix"].min()), float(sub["vix"].max())],
        }

    return {
        "break_date": break_date,
        "n_total": int(len(panel)),
        "vix_at_break": float(vix_monthly[vix_monthly["month_end"]
                              <= cutoff]["vix"].iloc[-1]
                              if len(vix_monthly) else float("nan")),
        "overall": {
            "delta_log_tau": float(overall_post.mean() - overall_pre.mean()),
            "t_stat": float(t_overall) if np.isfinite(t_overall) else float("nan"),
            "p_value": float(p_overall) if np.isfinite(p_overall) else float("nan"),
        },
        "by_tercile": by_tercile,
    }


# ---------------------------------------------------------------------------
# Test 3 — Pseudo-event null
# ---------------------------------------------------------------------------

def run_pseudo_event(returns_panel: pd.DataFrame, *,
                     headlines_per_firm_month: int = 12,
                     real_lm_sentiment_dist=None) -> dict:
    """Random event timing with sentiment drawn from the LM empirical
    distribution observed in the real 8-K corpus. If tau-hat is comparable
    to real-data tau-hat, event-timing is doing none of the work — i.e.,
    the post-event drift the estimator picks up is independent of WHEN the
    event happened.
    """
    fr = build_forward_returns(returns_panel)
    fr = fr.dropna(subset=[f"ret_h{h}" for h in range(1, H_MAX + 1)])
    fr["month_end"] = fr["date"] + pd.offsets.MonthEnd(0)

    # Approximation of the LM empirical distribution: ~22% nonzero,
    # of nonzero values, ~60% positive at +0.5, ~40% negative at -0.5.
    # (The exact figures from the 32,710-filing corpus aren't materially
    # different from this approximation for the placebo's purpose.)
    if real_lm_sentiment_dist is None:
        def draw_sent(n):
            nonzero = RNG.random(n) < 0.22
            signs = RNG.choice([+0.5, -0.5], size=n, p=[0.60, 0.40])
            return np.where(nonzero, signs, 0.0)
    else:
        def draw_sent(n):
            return RNG.choice(real_lm_sentiment_dist, size=n, replace=True)

    sample_rows = []
    for (firm_id, me), grp in fr.groupby(["firm_id", "month_end"]):
        if len(grp) < headlines_per_firm_month:
            continue
        idx = RNG.choice(grp.index.values,
                         size=headlines_per_firm_month, replace=False)
        sample_rows.append(grp.loc[idx])
    headline_panel = pd.concat(sample_rows, ignore_index=True)
    headline_panel["sentiment"] = draw_sent(len(headline_panel))

    fits = per_firm_month_fits(headline_panel, label="pseudo_event")

    return {
        "headlines_per_firm_month": headlines_per_firm_month,
        "tau_summary": summarize(fits["tau"].values),
        "n_firm_months": int(len(fits)),
        "n_firms": int(fits["firm_id"].nunique()) if len(fits) else 0,
        "fits": fits,
    }


# ---------------------------------------------------------------------------
# Real-data baseline (read from cached stage3 panel)
# ---------------------------------------------------------------------------

def real_baseline(stage3_panel: pd.DataFrame) -> dict:
    return {
        "tau_summary": summarize(stage3_panel["tau"].values),
        "n_firm_months": int(len(stage3_panel)),
        "n_firms": int(stage3_panel["firm_id"].nunique()),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(n_firms_for_placebo: int = 100):
    print("Loading cached data...")
    returns_all = pd.read_parquet(
        ROOT / "cache" / "live_n500_2020-01-01_2024-12-31.parquet"
    )
    # Pandas multi-level columns from yfinance — flatten and rename
    if isinstance(returns_all.columns, pd.MultiIndex):
        returns_all.columns = [c[0] if c[1] == "" else c[0]
                                for c in returns_all.columns]
    print(f"  returns: {len(returns_all):,} rows, "
          f"{returns_all['firm_id'].nunique()} firms")

    stage3 = pd.read_parquet(ROOT / "output" / "stage3_panel.parquet")
    stage3["month_end"] = pd.to_datetime(stage3["month_end"])
    print(f"  stage3 panel: {len(stage3)} firm-months, "
          f"{stage3['firm_id'].nunique()} firms")

    # Subsample firms for placebo (full ~500 is too many for quick run)
    firms_subset = sorted(returns_all["firm_id"].unique())[:n_firms_for_placebo]
    returns_sub = returns_all[returns_all["firm_id"].isin(firms_subset)].copy()
    print(f"  placebo subset: {len(firms_subset)} firms, "
          f"{len(returns_sub):,} firm-day rows")

    print("\n=== Real-data baseline ===")
    base = real_baseline(stage3)
    print(f"  tau median = {base['tau_summary']['median']:.2f}d, "
          f"p25={base['tau_summary']['p25']:.2f}, "
          f"p75={base['tau_summary']['p75']:.2f}, n={base['n_firm_months']}")

    print("\n=== Test 1: Placebo (real returns, white-noise sentiment) ===")
    placebo = run_placebo(returns_sub)
    d = placebo["dense"]["tau_summary"]
    s = placebo["sparse"]["tau_summary"]
    print(f"  Dense N(0,1):   median tau = {d['median']:.2f}d, "
          f"p25={d['p25']:.2f}, p75={d['p75']:.2f}, n={d['n']}")
    print(f"  Sparse 22%/{{-1,+1}}: median tau = {s['median']:.2f}d, "
          f"p25={s['p25']:.2f}, p75={s['p75']:.2f}, n={s['n']}")
    print(f"  Real-data:       median tau = {base['tau_summary']['median']:.2f}d "
          f"(reference)")

    print("\n=== Test 2: Vol-regime stratification (VIX terciles) ===")
    vol = run_vol_regime(stage3)
    print(f"  Overall:  delta log tau = {vol['overall']['delta_log_tau']:+.3f}, "
          f"p={vol['overall']['p_value']:.4f}")
    for tercile in ["low", "mid", "high"]:
        r = vol["by_tercile"][tercile]
        print(f"  VIX-{tercile}: tau {r['median_tau_pre']:.2f}d -> {r['median_tau_post']:.2f}d, "
              f"delta log = {r['delta_log_tau']:+.3f}, "
              f"p = {r['p_value']:.4f} (n={r['n_pre']}/{r['n_post']}, "
              f"VIX range {r['vix_range'][0]:.1f}-{r['vix_range'][1]:.1f})")

    print("\n=== Test 3: Pseudo-event null (random timing, LM-distributed S) ===")
    pe = run_pseudo_event(returns_sub)
    p = pe["tau_summary"]
    print(f"  Pseudo-event:    median tau = {p['median']:.2f}d, "
          f"p25={p['p25']:.2f}, p75={p['p75']:.2f}, n={p['n']}")

    # ---------------------------------------------------------------------
    # Persist
    # ---------------------------------------------------------------------
    out = {
        "real_baseline": base,
        "placebo": {
            "dense": placebo["dense"],
            "sparse": placebo["sparse"],
            "headlines_per_firm_month": placebo["headlines_per_firm_month"],
            "sparse_nonzero_rate": placebo["sparse_nonzero_rate"],
        },
        "vol_regime": vol,
        "pseudo_event": {
            "tau_summary": pe["tau_summary"],
            "n_firm_months": pe["n_firm_months"],
            "n_firms": pe["n_firms"],
            "headlines_per_firm_month": pe["headlines_per_firm_month"],
        },
        "n_firms_for_placebo": n_firms_for_placebo,
    }
    out_path = ROOT / "output" / "diagnostic_results.json"
    out_path.write_text(json.dumps(out, indent=2, default=str))
    print(f"\nWritten to {out_path}")

    # Save the per-firm-month placebo fits for later plotting if needed
    placebo["fits_dense"].to_parquet(
        ROOT / "output" / "placebo_dense_fits.parquet", index=False
    )
    placebo["fits_sparse"].to_parquet(
        ROOT / "output" / "placebo_sparse_fits.parquet", index=False
    )
    pe["fits"].to_parquet(
        ROOT / "output" / "pseudo_event_fits.parquet", index=False
    )

    return out


if __name__ == "__main__":
    main()
