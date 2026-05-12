"""Stage 1 + Stage 2 estimators for the formal model in `formal_model.md`.

Stage 1: estimate horizon coefficients beta_hat(h) for h = 1..H_MAX via OLS.
Stage 2: fit beta_hat(h) = beta_0 * exp(-h * lambda) by weighted NLS,
         recovering tau_hat = ln(2) / lambda_hat.

Implements Equation (1) of the formal model.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
import statsmodels.api as sm

H_MAX = 20  # forecast horizon, in trading days
MIN_HEADLINES = 12  # below this we don't trust tau_hat (thin-corpus default;
                    # tighten to 25+ when you wire press-wire / news feeds)


@dataclass
class HalfLifeFit:
    firm_id: str
    window_end: pd.Timestamp
    n_headlines: int
    beta_0: float
    lambda_: float
    tau: float            # in trading days; np.inf if lambda_ <= 0
    beta_h: np.ndarray    # shape (H_MAX,), the stage-1 coefficients
    se_beta_h: np.ndarray
    fit_rmse: float       # NLS residual RMSE on stage-1 coefs


def stage1_horizon_betas(
    sentiment: np.ndarray,
    returns: np.ndarray,
    controls: Optional[np.ndarray] = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Estimate beta_hat(h) and SE for h = 1..H_MAX.

    Args:
        sentiment: (N,) sentiment scores S_{i,t_k} for each headline k = 1..N.
        returns:   (N, H_MAX) realized returns r_{i,t_k+h} aligned with sentiment.
        controls:  optional (N, K) controls X_{i,t_k} added to each horizon regression.

    Returns:
        beta_h:    (H_MAX,) horizon coefficients on sentiment.
        se_beta_h: (H_MAX,) heteroskedasticity-consistent SE.
    """
    n, h_max = returns.shape
    assert sentiment.shape == (n,), "sentiment must align with returns rows"

    beta_h = np.full(h_max, np.nan)
    se_h = np.full(h_max, np.nan)

    X_base = sentiment.reshape(-1, 1)
    if controls is not None:
        X_base = np.hstack([X_base, controls])
    X = sm.add_constant(X_base, has_constant="add")

    for h in range(h_max):
        y = returns[:, h]
        mask = np.isfinite(y) & np.all(np.isfinite(X), axis=1)
        if mask.sum() < 10:
            continue
        result = sm.OLS(y[mask], X[mask]).fit(cov_type="HC3")
        beta_h[h] = result.params[1]   # index 0 is the constant; 1 is sentiment
        se_h[h] = result.bse[1]

    return beta_h, se_h


def stage2_exponential_fit(
    beta_h: np.ndarray,
    se_h: np.ndarray,
) -> tuple[float, float, float]:
    """Fit beta_hat(h) = beta_0 * exp(-h * lambda) by weighted NLS.

    Returns:
        beta_0:  intercept of the decay curve at h = 0.
        lambda_: decay rate (per trading day). Negative or zero => no decay (tau = inf).
        rmse:    weighted residual RMSE.
    """
    h = np.arange(1, len(beta_h) + 1, dtype=float)
    mask = np.isfinite(beta_h) & np.isfinite(se_h) & (se_h > 0)
    if mask.sum() < 4:
        return np.nan, np.nan, np.nan

    h_fit = h[mask]
    y_fit = beta_h[mask]
    sigma_fit = se_h[mask]

    def model(hh, b0, lam):
        return b0 * np.exp(-hh * lam)

    # Initial guess: use h=1 estimate for beta_0 and a reasonable half-life of 5 days.
    p0 = (y_fit[0], np.log(2) / 5.0)
    try:
        popt, _ = curve_fit(
            model, h_fit, y_fit, sigma=sigma_fit, p0=p0,
            absolute_sigma=True, maxfev=2000,
        )
    except (RuntimeError, ValueError):
        return np.nan, np.nan, np.nan

    beta_0, lambda_ = popt
    residuals = (y_fit - model(h_fit, *popt)) / sigma_fit
    rmse = float(np.sqrt(np.mean(residuals**2)))
    return float(beta_0), float(lambda_), rmse


def estimate_halflife(
    firm_id: str,
    window_end: pd.Timestamp,
    sentiment: np.ndarray,
    returns: np.ndarray,
    controls: Optional[np.ndarray] = None,
) -> Optional[HalfLifeFit]:
    """End-to-end half-life estimation for one firm-window."""
    n = len(sentiment)
    if n < MIN_HEADLINES:
        return None

    beta_h, se_h = stage1_horizon_betas(sentiment, returns, controls)
    beta_0, lambda_, rmse = stage2_exponential_fit(beta_h, se_h)

    if not np.isfinite(lambda_):
        return None

    tau = np.log(2) / lambda_ if lambda_ > 0 else np.inf
    return HalfLifeFit(
        firm_id=firm_id,
        window_end=window_end,
        n_headlines=n,
        beta_0=beta_0,
        lambda_=lambda_,
        tau=float(tau),
        beta_h=beta_h,
        se_beta_h=se_h,
        fit_rmse=rmse,
    )


def estimate_panel_halflives(
    panel: pd.DataFrame,
    window_days: int = 504,
) -> pd.DataFrame:
    """Rolling firm-month half-life panel.

    `panel` schema (one row per headline):
        firm_id, date (timestamp), sentiment (float),
        ret_h1, ret_h2, ..., ret_h20 (forward returns), and any control columns.

    Returns a firm-month panel of half-life fits.
    """
    ret_cols = [f"ret_h{h}" for h in range(1, H_MAX + 1)]
    control_cols = [c for c in panel.columns
                    if c not in {"firm_id", "date", "sentiment", *ret_cols}]

    out_rows = []
    for firm_id, grp in panel.groupby("firm_id"):
        grp = grp.sort_values("date").reset_index(drop=True)
        # Roll forward at month-ends
        grp["month_end"] = grp["date"] + pd.offsets.MonthEnd(0)
        for me, _ in grp.groupby("month_end"):
            window_lo = me - pd.Timedelta(days=window_days * 1.45)  # calendar buffer
            sub = grp[(grp["date"] > window_lo) & (grp["date"] <= me)]
            if len(sub) < MIN_HEADLINES:
                continue
            X = sub[control_cols].values if control_cols else None
            fit = estimate_halflife(
                firm_id=firm_id,
                window_end=me,
                sentiment=sub["sentiment"].values,
                returns=sub[ret_cols].values,
                controls=X,
            )
            if fit is None:
                continue
            out_rows.append({
                "firm_id": fit.firm_id,
                "month_end": fit.window_end,
                "n_headlines": fit.n_headlines,
                "beta_0": fit.beta_0,
                "lambda_": fit.lambda_,
                "tau": fit.tau,
                "fit_rmse": fit.fit_rmse,
            })

    return pd.DataFrame(out_rows)
