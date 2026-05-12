"""Re-run Stage 3 on the cached panel from a previous live_mode invocation.

Avoids re-fetching EDGAR / yfinance — useful when the regression code changes.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.regressions import (
    test_h1_cross_sectional, simex_correction, test_llm_era_compression,
)


def main():
    panel = pd.read_parquet(ROOT / "output" / "stage3_panel.parquet")
    panel["month_end"] = pd.to_datetime(panel["month_end"])
    print(f"Stage 3 panel rows = {len(panel):,}, "
          f"firms = {panel['firm_id'].nunique()}, "
          f"months = {panel['month_end'].nunique()}")
    print(f"tau:    median={panel['tau'].median():.2f}d, "
          f"p25={panel['tau'].quantile(0.25):.2f}, "
          f"p75={panel['tau'].quantile(0.75):.2f}")
    print(f"attention: median={panel['attention'].median():.3f}")
    print(f"friction:  median={panel['friction'].median():.5f}")
    print(f"competition: unique values = {panel['competition'].unique()}")

    print("\nH1 cross-section (time + industry FE):")
    h1 = test_h1_cross_sectional(panel, time_effects=True, industry_effects=True)
    print(f"  alpha_hat = {h1.coef_attention:+.3f}  (SE {h1.se_attention:.3f})")
    print(f"  phi_hat   = {h1.coef_friction:+.3f}  (SE {h1.se_friction:.3f})")
    print(f"  gamma_hat = {h1.coef_competition:+.3f}  (SE {h1.se_competition:.3f})")
    print(f"  R2_within = {h1.r2_within:.4f}, n = {h1.n_obs}")

    print("\nH1 cross-section (industry FE only — recovers gamma):")
    h1b = test_h1_cross_sectional(panel, time_effects=False, industry_effects=True)
    print(f"  alpha_hat = {h1b.coef_attention:+.3f}  (SE {h1b.se_attention:.3f})")
    print(f"  phi_hat   = {h1b.coef_friction:+.3f}  (SE {h1b.se_friction:.3f})")
    print(f"  gamma_hat = {h1b.coef_competition:+.3f}  (SE {h1b.se_competition:.3f})")
    print(f"  R2_within = {h1b.r2_within:.4f}, n = {h1b.n_obs}")

    print("\nSIMEX attenuation correction (sigma_log_tau = 0.30):")
    sx = simex_correction(panel, sigma_log_tau=0.30, n_replicates=20,
                          time_effects=False, industry_effects=True)
    print(f"  alpha:  naive {sx['naive_alpha']:+.3f}  ->  corrected {sx['corrected_alpha']:+.3f}")
    print(f"  phi  :  naive {sx['naive_phi']:+.3f}  ->  corrected {sx['corrected_phi']:+.3f}")
    print(f"  gamma:  naive {sx['naive_gamma']:+.3f}  ->  corrected {sx['corrected_gamma']:+.3f}")

    print("\nLLM-era compression test (break = 2023-03-15):")
    comp = test_llm_era_compression(panel, break_date="2023-03-15")
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

    out = {
        "h1_pooled": {"alpha": h1.coef_attention, "phi": h1.coef_friction,
                      "gamma": h1.coef_competition, "r2": h1.r2_within, "n": h1.n_obs,
                      "se_alpha": h1.se_attention, "se_phi": h1.se_friction},
        "h1_no_time_fe": {"alpha": h1b.coef_attention, "phi": h1b.coef_friction,
                           "gamma": h1b.coef_competition, "r2": h1b.r2_within, "n": h1b.n_obs},
        "simex": {k: v for k, v in sx.items() if k != "trace"},
        "compression": comp,
    }
    (ROOT / "output" / "results.json").write_text(json.dumps(out, indent=2, default=str))
    print(f"\nResults written to {ROOT / 'output' / 'results.json'}")


if __name__ == "__main__":
    main()
