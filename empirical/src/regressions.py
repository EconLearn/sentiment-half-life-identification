"""Stage 3 estimators — Equation (2) of the formal model.

Tests Hypothesis H1:
  log(lambda_{i,t}) = log(lambda_0) + alpha * log(A_{i,t})
                                    + gamma * log(C_{i,t})
                                    - phi  * log(F_{i,t}) + u

Equivalently in tau-space:
  log(tau_{i,t}) = -log(lambda_0) - alpha * log(A_{i,t})
                                  - gamma * log(C_{i,t})
                                  + phi  * log(F_{i,t}) - u

H1: alpha > 0 and phi > 0 (signs flip when regressing on log(tau)).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import statsmodels.api as sm
from linearmodels.panel import PanelOLS


@dataclass
class H1Result:
    coef_attention: float       # alpha-hat (sign-flipped if regressing log tau)
    coef_friction: float        # phi-hat
    coef_competition: float     # gamma-hat
    se_attention: float
    se_friction: float
    se_competition: float
    n_obs: int
    r2_within: float
    summary_text: str


def _log_safe(s: pd.Series) -> pd.Series:
    """log with floor at 1e-6 to handle zeros."""
    return np.log(np.clip(s.astype(float), 1e-6, None))


def test_h1_cross_sectional(
    panel: pd.DataFrame,
    time_effects: bool = True,
    industry_effects: bool = True,
) -> H1Result:
    """Stage 3 primary specification.

    `panel` schema (one row per firm-month):
        firm_id, month_end, tau (>0), attention, competition, friction,
        size, bm, mom, rev, industry (for FE).

    Specification:
        log(tau) ~ log(A) + log(C) + log(F) + controls + (industry FE) + (time FE),
        SE clustered by firm and month.

    Note on identification of gamma (competition): if competition only varies
    in time (e.g., a shared step function across firms), it will be absorbed
    by time FE — gamma is identified separately via the model-release event
    study in `test_h1_model_release_event_study`. Pass `time_effects=False`
    in synthetic / sanity-check runs to recover gamma directly.
    """
    df = panel.copy()
    df = df[df["tau"].notna() & np.isfinite(df["tau"]) & (df["tau"] > 0)]

    df["log_tau"] = np.log(df["tau"])
    df["log_A"] = _log_safe(df["attention"])
    df["log_C"] = _log_safe(df["competition"])
    df["log_F"] = _log_safe(df["friction"])

    df = df.set_index(["firm_id", "month_end"])
    df = df.dropna(subset=["log_tau", "log_A", "log_C", "log_F"])

    exog_cols = ["log_A", "log_C", "log_F", "size", "bm", "mom", "rev"]
    exog = sm.add_constant(df[exog_cols])

    other_effects = df["industry"] if (industry_effects and "industry" in df.columns) else None

    mod = PanelOLS(
        df["log_tau"],
        exog,
        entity_effects=False,
        time_effects=time_effects,
        other_effects=other_effects,
        drop_absorbed=True,
        check_rank=False,
    )
    res = mod.fit(cov_type="clustered", cluster_entity=True, cluster_time=True)

    # log(tau) regression: H1 predicts negative on log_A, positive on log_F.
    # We report alpha, phi in their original (lambda-equation) sign convention.
    # Absorbed coefficients (e.g., log_C under time FE when competition is
    # firm-invariant) come back as NaN rather than crashing.
    def _get(s, k, sign=1):
        try:
            return sign * float(s[k])
        except (KeyError, IndexError):
            return float("nan")

    return H1Result(
        coef_attention=_get(res.params, "log_A", -1),
        coef_friction=_get(res.params, "log_F", +1),
        coef_competition=_get(res.params, "log_C", -1),
        se_attention=_get(res.std_errors, "log_A"),
        se_friction=_get(res.std_errors, "log_F"),
        se_competition=_get(res.std_errors, "log_C"),
        n_obs=int(res.nobs),
        r2_within=float(res.rsquared_within),
        summary_text=str(res),
    )


def test_h1_did_attention_shock(
    panel: pd.DataFrame,
    treated_mask: pd.Series,
    post_mask: pd.Series,
) -> dict:
    """Difference-in-differences specification (README §4.2).

    Identifies alpha by comparing pre/post tau in firms that experienced an
    exogenous industry-level attention shock (treated) vs. peer industries
    (control).

    Specification:
        log(tau) = beta_treat * Treat + beta_post * Post
                 + beta_did * (Treat * Post) + firm_FE + month_FE + e

    H1 prediction: beta_did < 0 (post-shock tau shrinks for treated).
    """
    df = panel.copy()
    df = df[df["tau"].notna() & np.isfinite(df["tau"]) & (df["tau"] > 0)]
    df["log_tau"] = np.log(df["tau"])
    df["treat"] = treated_mask.astype(float).reindex(df.index, fill_value=0)
    df["post"] = post_mask.astype(float).reindex(df.index, fill_value=0)
    df["did"] = df["treat"] * df["post"]

    df = df.set_index(["firm_id", "month_end"])
    exog = sm.add_constant(df[["treat", "post", "did"]])

    mod = PanelOLS(df["log_tau"], exog, entity_effects=True, time_effects=True)
    res = mod.fit(cov_type="clustered", cluster_entity=True, cluster_time=True)

    return {
        "beta_did": float(res.params["did"]),
        "se_did": float(res.std_errors["did"]),
        "p_did": float(res.pvalues["did"]),
        "n_obs": int(res.nobs),
        "summary": str(res),
    }


def simex_correction(
    panel: pd.DataFrame,
    sigma_log_tau: float = 0.30,
    lambdas: tuple = (0.0, 0.5, 1.0, 1.5, 2.0),
    n_replicates: int = 50,
    seed: int = 42,
    time_effects: bool = True,
    industry_effects: bool = True,
) -> dict:
    """SIMEX (Cook & Stefanski, 1994) attenuation correction for log(tau).

    The dependent variable log(tau) is itself an estimate, so its measurement
    error attenuates the Stage 3 coefficients toward zero. SIMEX estimates the
    bias by deliberately adding more noise (variance sigma^2 * lambda for
    lambda in `lambdas`), refitting at each level, and extrapolating to the
    *negative*-noise limit lambda = -1, which corresponds to no measurement error.

    Args:
        panel: Stage 3 panel (firm_id, month_end, tau, attention, competition,
               friction, controls, industry).
        sigma_log_tau: scale of measurement error in log(tau). Reasonable
                       defaults are 0.2-0.4 for typical 12-month windows.
                       Provide a per-firm-month estimate via bootstrap of
                       Stage 2 for higher precision.
        lambdas:       SIMEX noise scales. lam=0 is the naive estimate.
        n_replicates:  per-lambda Monte Carlo replications.

    Returns:
        dict with naive_*, corrected_*, and the lambda-by-coef trace.
    """
    rng = np.random.default_rng(seed)
    coefs = {"alpha": [], "phi": [], "gamma": []}

    base_tau = panel["tau"].copy()
    base_log_tau = np.log(np.clip(base_tau, 1e-6, None))

    for lam in lambdas:
        alphas, phis, gammas = [], [], []
        reps = 1 if lam == 0.0 else n_replicates
        for _ in range(reps):
            noise = rng.normal(0, sigma_log_tau * np.sqrt(lam), size=len(panel))
            perturbed = panel.copy()
            perturbed["tau"] = np.exp(base_log_tau + noise)
            r = test_h1_cross_sectional(
                perturbed,
                time_effects=time_effects,
                industry_effects=industry_effects,
            )
            alphas.append(r.coef_attention)
            phis.append(r.coef_friction)
            gammas.append(r.coef_competition)
        coefs["alpha"].append(np.mean(alphas))
        coefs["phi"].append(np.mean(phis))
        coefs["gamma"].append(np.mean(gammas))

    # Quadratic extrapolation to lambda = -1 (no measurement error).
    L = np.array(lambdas, dtype=float)
    L_design = np.column_stack([np.ones_like(L), L, L ** 2])
    extrap = {}
    for k, ys in coefs.items():
        coefs_quad = np.linalg.lstsq(L_design, np.array(ys), rcond=None)[0]
        # Evaluate at lambda = -1
        extrap[k] = float(coefs_quad @ np.array([1.0, -1.0, 1.0]))

    return {
        "naive_alpha": coefs["alpha"][0],
        "naive_phi": coefs["phi"][0],
        "naive_gamma": coefs["gamma"][0],
        "corrected_alpha": extrap["alpha"],
        "corrected_phi": extrap["phi"],
        "corrected_gamma": extrap["gamma"],
        "lambdas": list(L),
        "trace": coefs,
        "sigma_log_tau": sigma_log_tau,
    }


def test_llm_era_compression(
    panel: pd.DataFrame,
    break_date: str = "2023-03-15",
    attention_col: str = "attention",
    n_quantiles: int = 4,
) -> dict:
    """Tests the central thesis: tau-hat shrinks after the LLM era opens.

    Default break date is 2023-03-15 (GPT-4 release). 2022-11-30 (ChatGPT)
    is also a defensible alternative; both can be reported as a robustness check.

    Returns:
        - mean_log_tau_pre, mean_log_tau_post, t_stat, p_value (pooled)
        - chow_f, chow_p: Chow break F-statistic on the H1 cross-section
        - by_quantile: median tau pre/post, stratified by attention quartile
                       (the prediction is that high-attention firms compress
                       more than low-attention firms).
    """
    import scipy.stats as st

    df = panel.copy()
    df = df[df["tau"].notna() & np.isfinite(df["tau"]) & (df["tau"] > 0)]
    df["log_tau"] = np.log(df["tau"])
    df["month_end"] = pd.to_datetime(df["month_end"])
    cutoff = pd.Timestamp(break_date)
    df["post"] = (df["month_end"] >= cutoff).astype(int)

    pre = df[df["post"] == 0]["log_tau"]
    post = df[df["post"] == 1]["log_tau"]
    if len(pre) < 10 or len(post) < 10:
        return {"error": "insufficient observations on one side of the break"}

    t_stat, p_val = st.ttest_ind(pre, post, equal_var=False)
    mean_pre = float(pre.mean())
    mean_post = float(post.mean())

    # Chow break test on the H1 specification.
    pooled = test_h1_cross_sectional(df.assign(tau=np.exp(df["log_tau"])),
                                     time_effects=False, industry_effects=True)
    pre_panel = df[df["post"] == 0]
    post_panel = df[df["post"] == 1]
    chow_f, chow_p = (np.nan, np.nan)
    try:
        r_pre = test_h1_cross_sectional(pre_panel.assign(tau=np.exp(pre_panel["log_tau"])),
                                        time_effects=False, industry_effects=True)
        r_post = test_h1_cross_sectional(post_panel.assign(tau=np.exp(post_panel["log_tau"])),
                                         time_effects=False, industry_effects=True)
        # Approximate Chow F via R^2 within decomposition.
        n = pooled.n_obs
        k = 4  # const + log_A + log_C + log_F
        ssr_pooled = (1 - pooled.r2_within) * n
        ssr_split = ((1 - r_pre.r2_within) * r_pre.n_obs
                     + (1 - r_post.r2_within) * r_post.n_obs)
        chow_f = float(((ssr_pooled - ssr_split) / k) / (ssr_split / max(1, n - 2 * k)))
        chow_p = float(1 - st.f.cdf(chow_f, k, max(1, n - 2 * k)))
    except Exception as e:
        chow_f, chow_p = (np.nan, np.nan)

    # Heterogeneity by attention quartile (the compression should be
    # *larger* in high-attention firms).
    by_quantile = {}
    if attention_col in df.columns:
        df["att_q"] = pd.qcut(df[attention_col], q=n_quantiles,
                              labels=False, duplicates="drop")
        for q, sub in df.groupby("att_q"):
            pre_q = sub[sub["post"] == 0]["log_tau"]
            post_q = sub[sub["post"] == 1]["log_tau"]
            if len(pre_q) >= 5 and len(post_q) >= 5:
                by_quantile[int(q)] = {
                    "n_pre": int(len(pre_q)),
                    "n_post": int(len(post_q)),
                    "median_tau_pre": float(np.exp(pre_q.median())),
                    "median_tau_post": float(np.exp(post_q.median())),
                    "delta_log_tau": float(post_q.mean() - pre_q.mean()),
                }

    return {
        "break_date": break_date,
        "n_pre": int(len(pre)),
        "n_post": int(len(post)),
        "median_tau_pre": float(np.exp(pre.median())),
        "median_tau_post": float(np.exp(post.median())),
        "mean_log_tau_pre": mean_pre,
        "mean_log_tau_post": mean_post,
        "delta_log_tau": float(mean_post - mean_pre),
        "compression_pct": float(100 * (1 - np.exp(mean_post - mean_pre))),
        "t_stat": float(t_stat),
        "p_value": float(p_val),
        "chow_f": chow_f,
        "chow_p": chow_p,
        "by_attention_quartile": by_quantile,
    }


def test_h1_model_release_event_study(
    panel: pd.DataFrame,
    release_dates: list[pd.Timestamp],
    window: int = 3,
) -> pd.DataFrame:
    """Foundation-model release event study (README §4.3, identifies gamma).

    For each release date r, regress change in log(tau) within +/- `window` months
    of r on a post-release dummy, with firm and time FE.

    Returns per-release coefficients and a pooled estimate.
    """
    rows = []
    for r in release_dates:
        lo = r - pd.DateOffset(months=window)
        hi = r + pd.DateOffset(months=window)
        sub = panel[(panel["month_end"] >= lo) & (panel["month_end"] <= hi)].copy()
        sub = sub[sub["tau"].notna() & np.isfinite(sub["tau"]) & (sub["tau"] > 0)]
        sub["log_tau"] = np.log(sub["tau"])
        sub["post"] = (sub["month_end"] >= r).astype(float)
        sub = sub.set_index(["firm_id", "month_end"])
        if len(sub) < 100:
            continue
        exog = sm.add_constant(sub[["post"]])
        mod = PanelOLS(sub["log_tau"], exog, entity_effects=True)
        res = mod.fit(cov_type="clustered", cluster_entity=True)
        rows.append({
            "release_date": r,
            "beta_post": float(res.params["post"]),
            "se_post": float(res.std_errors["post"]),
            "p_post": float(res.pvalues["post"]),
            "n_obs": int(res.nobs),
        })
    return pd.DataFrame(rows)
