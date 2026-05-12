# Empirical Pipeline — Adaptive Sentiment Loop

This directory contains the empirical research design and code skeleton that estimate the formal model in `../formal_model.md`. The pipeline tests Hypothesis H1 from the main paper:

> *The predictive half-life of AI-derived sentiment signals is inversely proportional to institutional attention and directly proportional to information-processing frictions.*

---

## 1. Research design overview

The pipeline runs in three estimation stages, each a separately testable component of the formal model.

| Stage | Estimand | Source equation | Module |
|---|---|---|---|
| 1 | $\widehat\beta(h)$ for $h = 1, \dots, 20$ | Eq. (1) — horizon regressions | `src/halflife.py` |
| 2 | $\widehat\tau_{i,t}, \widehat\beta_{0,i,t}$ | Eq. (1) — NLS fit of exponential decay | `src/halflife.py` |
| 3 | $\widehat\alpha, \widehat\gamma, \widehat\phi$ | Eq. (2) — log-linear regression | `src/regressions.py` |

Stage 3 is the hypothesis test: the signs and significance of $\widehat\alpha$ and $\widehat\phi$ are H1.

---

## 2. Data sources

| Data | Vendor (academic) | Open-data substitute | Notes |
|---|---|---|---|
| Daily returns, volumes | CRSP via WRDS | Yahoo Finance via `yfinance` | CRSP preferred for delisting handling |
| Fundamentals | Compustat via WRDS | EDGAR / SEC API | for size, B/M, leverage |
| Microstructure (spread, retail share) | TAQ via WRDS | IEX / Polygon (API) | required for friction composite |
| News headlines | RavenPack, Refinitiv | EDGAR 8-K filings, GDELT | filings give clean firm-event linkage |
| Institutional attention | Bloomberg AIA | Google Trends SVI (Da, Engelberg, & Gao, 2011) | SVI is open and tractable |
| LLM sentiment | OpenAI / Anthropic API | open-weight model (Llama, Qwen) via local inference | open-weight preferred for reproducibility & contamination control |

Sample period proposal: **2018-01-01 to 2025-12-31**, with the analysis window restricted to headlines published *after* the chosen LLM's training cutoff (Section 3.1 of main paper) — typically 2024–2025 for frontier models.

---

## 3. Variable construction

See `data_spec.md` for full definitions. Summary:

- **$S_{i,t}$**: LLM zero-shot sentiment on each headline mentioning firm $i$ at time $t$, score in $[-1, 1]$, with confidence weight.
- **$r_{i,t+h}$**: log return, day $t \to t+h$, where $t$ is the trading day immediately following the news timestamp.
- **$A_{i,t}$**: Bloomberg AIA dummy (high/low) if available; otherwise 30-day SVI z-score.
- **$F_{i,t}$**: composite z-score of: (a) Corwin–Schultz daily bid-ask spread, (b) idiosyncratic vol (Fama–French 5 residual), (c) retail order share (Boehmer et al., 2021 sub-penny method or alternative).
- **$C_{i,t}$**: foundation-model release calendar dummies (GPT-3.5, GPT-4, Claude 3, Llama 3, Claude 4, etc.). $C$ enters as a step function at each release date.

---

## 4. Identification strategy

### 4.1 Cross-sectional (Stage 3, primary)

$$
\log \widehat\tau_{i,t} \;=\; \alpha_0 \;-\; \alpha \log A_{i,t} \;+\; \phi \log F_{i,t} \;+\; \mathbf{X}_{i,t}'\boldsymbol{\delta} \;+\; \mu_j \;+\; \nu_t \;+\; u_{i,t}
$$

with industry FE $\mu_j$ and year–month FE $\nu_t$. Standard errors clustered by firm and date.

H1 prediction: $\alpha > 0$ and $\phi > 0$.

### 4.2 Difference-in-differences around attention shocks

Following Engelberg & Parsons (2011), use *peer-firm* news events that draw exogenous industry-level institutional attention without changing the focal firm's fundamentals.

Treatment: firms in same SIC-3 industry as a peer experiencing a top-1% news-volume day.
Outcome: $\widehat\tau_{i,t}$ in the 5-day pre-event vs. 5-day post-event window.
Prediction: post-event $\widehat\tau$ shrinks for treated firms (consistent with $\alpha > 0$), with no parallel effect on controls.

### 4.3 Foundation-model release event study (identifies $\gamma$)

Around each major LLM release (e.g., GPT-4 on 2023-03-14, Claude 3 on 2024-03-04, Claude 4 family in 2025), regress changes in $\widehat\tau$ on a release dummy. A negative coefficient identifies $\gamma > 0$ via the discrete jump in $C$.

This test is the cleanest direct evidence for the endogeneity / reflexivity argument in Section 3.3 of the main paper.

---

## 5. Robustness & falsification

- **Random sentiment falsification.** Replace $S_{i,t}$ with permuted noise; coefficients should be statistically indistinguishable from zero.
- **Alternative LLM vintages.** Repeat with Llama-3, Claude 3, GPT-4o; consistent signs across models guard against single-model artifacts.
- **Subsample stability.** Pre-2023 (limited deployed AI) vs. post-2023 (heavy AI deployment) — H1 magnitudes should be larger post-2023.
- **Liquidity tier splits.** Expect H1 effects strongest among high-friction (small-cap) names, weakest among mega-caps where attention is saturated.
- **Excluding earnings windows.** Earnings news is structurally different from other news; H1 should hold both with and without.

---

## 6. Pipeline structure

```
empirical/
├── README.md             ← this file
├── data_spec.md          ← variable definitions
├── requirements.txt      ← Python dependencies
├── pipeline.py           ← main orchestration
├── src/
│   ├── __init__.py
│   ├── data_loader.py    ← CRSP / Compustat / news loaders (stub)
│   ├── sentiment.py      ← LLM scoring of headlines (stub)
│   ├── halflife.py       ← Eq. (1) estimation — IMPLEMENTED
│   ├── attention.py      ← AIA / SVI construction (stub)
│   ├── frictions.py      ← spread / IVOL / retail-share composite (stub)
│   └── regressions.py    ← Eq. (2) hypothesis tests — IMPLEMENTED
└── notebooks/
    └── exploration.ipynb ← exploratory plots (stub)
```

The two load-bearing modules — `halflife.py` and `regressions.py` — are implemented and runnable on synthetic data (see `pipeline.py`). The data acquisition modules are deliberately stubs because they require institutional credentials (WRDS, Bloomberg) or paid APIs.

---

## 7. Quick start

```bash
cd /Users/judewallis/Research/empirical
python3 -m pip install -r requirements.txt
python3 pipeline.py --synthetic   # runs end-to-end on simulated data
```

The synthetic mode generates panel data consistent with Equations (1)–(2), then estimates them, recovering the planted parameters as a sanity check.

---

## 8. Compute & cost notes

- **LLM scoring** is the dominant cost. At 100,000 headlines × $0.0001/headline ≈ $10 per pass with a frontier model; ≈ free with local open-weight inference.
- **Half-life estimation** is embarrassingly parallel across firms: $O(N \cdot H)$ regressions, each with $\sim 10^3$ observations; ~5 min on a laptop for the full panel.
- **Storage**: full pipeline outputs ~2 GB intermediate parquet, dominated by horizon-stacked sentiment-return panels.

---

## 9. Outstanding work before submission

A real working paper using this pipeline still needs:
1. Production LLM scoring run on a real news corpus (vendor or EDGAR-derived).
2. Hand-validated sentiment labels on ~500 headlines for LLM calibration.
3. Decision on Bloomberg AIA vs. SVI as the primary attention proxy (drives main-table results).
4. Pre-registration of $\widehat\alpha, \widehat\phi$ signs prior to running on the full sample, to bind the analysis against p-hacking concerns Harvey et al. (2016) raise.
