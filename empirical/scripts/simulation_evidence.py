"""Controlled-DGP simulation establishing observational equivalence.

This is the formal-evidence companion to the real-data diagnostics in
diagnostic_tests.py. It generates synthetic panels under two distinct DGPs
and demonstrates that the canonical Stage 1 + Stage 2 estimator returns
indistinguishable tau-hat distributions from both, establishing that the
estimand of the canonical pipeline is not separately identified from
return autocorrelation, volatility regime structure, and event clustering.

DGP A (no sentiment effect):
    r_t = rho * r_{t-1} + sigma_t * eps_t
    sigma_t from a two-state Markov regime (calm / turbulent)
    Events occur at random dates with Hawkes-like clustering
    Sentiment S_t ~ N(0, 1), independent of returns at every horizon

DGP B (planted sentiment effect):
    Identical AR(1) + regime + event-clustering structure
    Plus: r_t includes a sentiment-driven term beta * S_{t_event} * exp(-h * lambda_true)
    with planted tau_true = 10 trading days (a literature-plausible value)

Hypothesis: under DGP A, tau-hat is a non-trivial positive number despite
sentiment carrying zero information; under DGP B, tau-hat is biased toward the
DGP-A value rather than recovering the planted tau_true. The two distributions
overlap, establishing observational equivalence.

Robustness: same diagnostics under power-law and stretched-exponential decay
fits in Stage 2 instead of pure exponential. The non-identification result
should be invariant to the parametric form.

Output: empirical/output/simulation_results.json with per-DGP tau-hat
distributions, planted-vs-recovered comparison, and per-functional-form
diagnostic rejections.
"""
from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
import statsmodels.api as sm

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.halflife import H_MAX, MIN_HEADLINES, stage1_horizon_betas

RNG = np.random.default_rng(20260501)


# ---------------------------------------------------------------------------
# DGP
# ---------------------------------------------------------------------------

@dataclass
class DGPParams:
    n_firms: int = 100
    n_days: int = 1260   # 5 years of trading days
    rho_ar1: float = 0.04   # AR(1) coefficient (S&P 500 calibration)
    sigma_calm: float = 0.008
    sigma_turbulent: float = 0.025
    p_calm_to_turbulent: float = 0.005   # ~1 transition / 200 days
    p_turbulent_to_calm: float = 0.02
    base_event_rate: float = 12 / 252    # ~12 events / firm-year
    hawkes_alpha: float = 0.30           # event self-excitation
    hawkes_decay_days: float = 5.0
    sentiment_nonzero_rate: float = 0.22  # match LM density on real 8-Ks
    # DGP B only:
    tau_true_days: float = 10.0
    beta_sentiment: float = 0.005        # 50 bps per unit of S — strong signal
    # DGP C: same structure as B but LM-calibrated weak signal
    beta_sentiment_weak: float = 0.0005  # 5 bps per unit of S — LM-realistic


def simulate_returns_panel(p: DGPParams) -> pd.DataFrame:
    """Generate a (firm, day, return) panel with AR(1) + 2-regime vol + Markov."""
    rows = []
    for fid in range(p.n_firms):
        # Markov regime
        regime = np.zeros(p.n_days, dtype=np.int8)
        for t in range(1, p.n_days):
            prev = regime[t - 1]
            if prev == 0:  # calm
                regime[t] = 1 if RNG.random() < p.p_calm_to_turbulent else 0
            else:           # turbulent
                regime[t] = 0 if RNG.random() < p.p_turbulent_to_calm else 1
        sigma = np.where(regime == 0, p.sigma_calm, p.sigma_turbulent)
        eps = RNG.standard_normal(p.n_days)
        # AR(1) returns
        r = np.zeros(p.n_days)
        r[0] = sigma[0] * eps[0]
        for t in range(1, p.n_days):
            r[t] = p.rho_ar1 * r[t - 1] + sigma[t] * eps[t]
        rows.append(pd.DataFrame({
            "firm_id": f"SIM_{fid:03d}",
            "day": np.arange(p.n_days),
            "ret": r,
            "regime": regime,
        }))
    return pd.concat(rows, ignore_index=True)


def simulate_events_hawkes(returns: pd.DataFrame, p: DGPParams) -> pd.DataFrame:
    """Per-firm event arrivals with Hawkes-like self-excitation."""
    out_rows = []
    for fid, grp in returns.groupby("firm_id"):
        days = grp["day"].values
        n = len(days)
        intensity = np.full(n, p.base_event_rate, dtype=float)
        events = []
        for t in range(n):
            if RNG.random() < intensity[t]:
                events.append(t)
                # Add Hawkes-style boost to subsequent intensities
                for s in range(t + 1, min(t + 30, n)):
                    intensity[s] += p.hawkes_alpha * np.exp(-(s - t) / p.hawkes_decay_days)
        for t_event in events:
            out_rows.append({"firm_id": fid, "day": int(t_event)})
    return pd.DataFrame(out_rows)


def attach_sentiment_and_returns(events: pd.DataFrame, returns: pd.DataFrame,
                                  p: DGPParams, dgp: str) -> pd.DataFrame:
    """Build (firm, event_day, sentiment, ret_h1..ret_h20) panel.

    For dgp == "A" (null), sentiment carries no information.
    For dgp == "B" (strong signal), sentiment drives an exponential decay with
        beta_sentiment magnitude.
    For dgp == "C" (weak signal, LM-calibrated), same form with beta_sentiment_weak.
    """
    # Forward returns
    fr = returns.sort_values(["firm_id", "day"]).copy()
    for h in range(1, H_MAX + 1):
        fr[f"ret_h{h}"] = fr.groupby("firm_id")["ret"].shift(-h)
    panel = events.merge(fr, on=["firm_id", "day"], how="left")
    panel = panel.dropna(subset=[f"ret_h{h}" for h in range(1, H_MAX + 1)])

    # Sentiment: signed value with prob = sentiment_nonzero_rate, else 0
    nz = RNG.random(len(panel)) < p.sentiment_nonzero_rate
    s_vals = RNG.choice([-1.0, +1.0], size=len(panel)) * RNG.random(len(panel)) * 2  # in [-2, 2]
    panel["sentiment"] = np.where(nz, s_vals, 0.0)

    if dgp in ("B", "C"):
        beta = p.beta_sentiment if dgp == "B" else p.beta_sentiment_weak
        lam_true = np.log(2) / p.tau_true_days
        for h in range(1, H_MAX + 1):
            decay = np.exp(-h * lam_true)
            panel[f"ret_h{h}"] = panel[f"ret_h{h}"].values + (
                beta * panel["sentiment"].values * decay
            )
    return panel


# ---------------------------------------------------------------------------
# Stage 2 alternative functional forms (functional-form robustness)
# ---------------------------------------------------------------------------

def _fit(model_fn, p0, beta_h, se_h):
    h = np.arange(1, len(beta_h) + 1, dtype=float)
    mask = np.isfinite(beta_h) & np.isfinite(se_h) & (se_h > 0)
    if mask.sum() < 4:
        return np.nan
    try:
        popt, _ = curve_fit(model_fn, h[mask], beta_h[mask],
                             sigma=se_h[mask], p0=p0,
                             absolute_sigma=True, maxfev=5000)
    except (RuntimeError, ValueError):
        return np.nan
    return popt


def fit_exponential(beta_h, se_h):
    """beta_0 * exp(-h * lam) — the canonical form."""
    p = _fit(lambda h, b0, lam: b0 * np.exp(-h * lam),
             p0=(0.001, np.log(2) / 5.0), beta_h=beta_h, se_h=se_h)
    if np.isscalar(p) and np.isnan(p):
        return np.nan
    b0, lam = p
    return float(np.log(2) / lam) if lam > 0 else float("inf")


def fit_powerlaw(beta_h, se_h):
    """beta_0 * (1 + h)^(-alpha) — power-law decay; recover effective half-life."""
    p = _fit(lambda h, b0, a: b0 * (1.0 + h) ** (-a),
             p0=(0.001, 0.5), beta_h=beta_h, se_h=se_h)
    if np.isscalar(p) and np.isnan(p):
        return np.nan
    b0, a = p
    if a <= 0:
        return float("inf")
    return float(2 ** (1 / a) - 1)  # effective half-life under power law


def fit_stretched_exp(beta_h, se_h):
    """beta_0 * exp(-(h * lam)^beta) — stretched exponential."""
    p = _fit(lambda h, b0, lam, beta: b0 * np.exp(-((h * lam) ** np.clip(beta, 0.1, 3.0))),
             p0=(0.001, np.log(2) / 5.0, 1.0), beta_h=beta_h, se_h=se_h)
    if np.isscalar(p) and np.isnan(p):
        return np.nan
    b0, lam, beta = p
    if lam <= 0:
        return float("inf")
    # Effective half-life under stretched exp: solve b0 * exp(-(t lam)^beta) = b0/2
    return float((np.log(2)) ** (1 / beta) / lam)


# ---------------------------------------------------------------------------
# Run both DGPs
# ---------------------------------------------------------------------------

def per_firm_month_taus(panel: pd.DataFrame, fitfn) -> np.ndarray:
    """Stage 1 + Stage 2 with given functional form, per firm-month rolling window."""
    ret_cols = [f"ret_h{h}" for h in range(1, H_MAX + 1)]
    df = panel.copy()
    df["month_end"] = (df["day"] // 21).astype(int)  # rough months in trading days
    out = []
    window_size = 504
    for fid, grp in df.groupby("firm_id"):
        grp = grp.sort_values("day")
        for me, _ in grp.groupby("month_end"):
            sub = grp[(grp["day"] > me * 21 - window_size) & (grp["day"] <= me * 21)]
            if len(sub) < MIN_HEADLINES:
                continue
            S = sub["sentiment"].values
            R = sub[ret_cols].values
            beta_h, se_h = stage1_horizon_betas(S, R)
            tau = fitfn(beta_h, se_h)
            if np.isfinite(tau) and 0 < tau < 1000:
                out.append(tau)
    return np.array(out)


def summarize(taus: np.ndarray) -> dict:
    if len(taus) == 0:
        return {"n": 0}
    return {
        "n": int(len(taus)),
        "median": float(np.median(taus)),
        "p25": float(np.percentile(taus, 25)),
        "p75": float(np.percentile(taus, 75)),
        "mean": float(np.mean(taus)),
    }


def main():
    p = DGPParams()
    print("Generating returns panels (one each for DGP A, B, C)...")
    returns_A = simulate_returns_panel(p)
    returns_B = simulate_returns_panel(p)
    returns_C = simulate_returns_panel(p)
    print(f"  panels: 3 x {len(returns_A):,} rows, {p.n_firms} firms each")

    print("\nGenerating events with Hawkes-like clustering...")
    events_A = simulate_events_hawkes(returns_A, p)
    events_B = simulate_events_hawkes(returns_B, p)
    events_C = simulate_events_hawkes(returns_C, p)

    print("\nAttaching sentiment + forward returns...")
    panel_A = attach_sentiment_and_returns(events_A, returns_A, p, dgp="A")
    panel_B = attach_sentiment_and_returns(events_B, returns_B, p, dgp="B")
    panel_C = attach_sentiment_and_returns(events_C, returns_C, p, dgp="C")

    print("\n=== Canonical exponential Stage 2 ===")
    tau_A_exp = per_firm_month_taus(panel_A, fit_exponential)
    tau_B_exp = per_firm_month_taus(panel_B, fit_exponential)
    tau_C_exp = per_firm_month_taus(panel_C, fit_exponential)
    print(f"  DGP A (no signal):       median tau-hat = {summarize(tau_A_exp).get('median', float('nan')):.2f}d "
          f"(n={len(tau_A_exp)})")
    print(f"  DGP B (strong, tau=10):  median tau-hat = {summarize(tau_B_exp).get('median', float('nan')):.2f}d "
          f"(n={len(tau_B_exp)})")
    print(f"  DGP C (weak, tau=10):    median tau-hat = {summarize(tau_C_exp).get('median', float('nan')):.2f}d "
          f"(n={len(tau_C_exp)})")

    print("\n=== Power-law Stage 2 (functional-form robustness) ===")
    tau_A_pow = per_firm_month_taus(panel_A, fit_powerlaw)
    tau_B_pow = per_firm_month_taus(panel_B, fit_powerlaw)
    tau_C_pow = per_firm_month_taus(panel_C, fit_powerlaw)
    print(f"  DGP A: median = {summarize(tau_A_pow).get('median', float('nan')):.2f}d  "
          f"DGP B: {summarize(tau_B_pow).get('median', float('nan')):.2f}d  "
          f"DGP C: {summarize(tau_C_pow).get('median', float('nan')):.2f}d")

    print("\n=== Stretched-exponential Stage 2 ===")
    tau_A_str = per_firm_month_taus(panel_A, fit_stretched_exp)
    tau_B_str = per_firm_month_taus(panel_B, fit_stretched_exp)
    tau_C_str = per_firm_month_taus(panel_C, fit_stretched_exp)
    print(f"  DGP A: median = {summarize(tau_A_str).get('median', float('nan')):.2f}d  "
          f"DGP B: {summarize(tau_B_str).get('median', float('nan')):.2f}d  "
          f"DGP C: {summarize(tau_C_str).get('median', float('nan')):.2f}d")

    out = {
        "params": {
            "n_firms": p.n_firms, "n_days": p.n_days,
            "rho_ar1": p.rho_ar1, "sigma_calm": p.sigma_calm,
            "sigma_turbulent": p.sigma_turbulent,
            "base_event_rate_per_day": p.base_event_rate,
            "hawkes_alpha": p.hawkes_alpha,
            "sentiment_nonzero_rate": p.sentiment_nonzero_rate,
            "tau_true_days": p.tau_true_days,
            "beta_sentiment_dgp_B": p.beta_sentiment,
            "beta_sentiment_dgp_C": p.beta_sentiment_weak,
        },
        "exponential": {
            "DGP_A_no_signal": summarize(tau_A_exp),
            "DGP_B_strong_signal": summarize(tau_B_exp),
            "DGP_C_weak_signal": summarize(tau_C_exp),
            "planted_tau": p.tau_true_days,
        },
        "powerlaw": {
            "DGP_A_no_signal": summarize(tau_A_pow),
            "DGP_B_strong_signal": summarize(tau_B_pow),
            "DGP_C_weak_signal": summarize(tau_C_pow),
        },
        "stretched_exponential": {
            "DGP_A_no_signal": summarize(tau_A_str),
            "DGP_B_strong_signal": summarize(tau_B_str),
            "DGP_C_weak_signal": summarize(tau_C_str),
        },
    }

    out_path = ROOT / "output" / "simulation_results.json"
    out_path.write_text(json.dumps(out, indent=2))
    print(f"\nWritten to {out_path}")
    return out


if __name__ == "__main__":
    main()
