# Formalizing the Adaptive Sentiment Loop

A three-equation formalization of the framework introduced in Section 4 of the main paper. The goal is to give the ASL a precise mathematical statement so that the empirical pipeline (see `empirical/`) has well-defined estimation targets.

---

## Notation

| Symbol | Meaning | Domain |
|---|---|---|
| $i$, $t$, $h$ | firm, time (trading days), forecast horizon | indices |
| $r_{i,t+h}$ | log return of firm $i$, day $t \to t+h$ | $\mathbb{R}$ |
| $S_{i,t}$ | AI-derived sentiment score | $[-1, 1]$ |
| $\tau_{i,t}$ | signal half-life | $\mathbb{R}_{>0}$ |
| $\lambda_{i,t} = \ln 2 / \tau_{i,t}$ | decay hazard rate | $\mathbb{R}_{>0}$ |
| $\beta_{i,t}$ | sentiment-return coefficient at $h=0$ | $\mathbb{R}$ |
| $A_{i,t}$ | attention (e.g., Bloomberg AIA, SVI) | $\mathbb{R}_{>0}$ |
| $C_{i,t}$ | competition / crowding | $\mathbb{R}_{\geq 1}$ |
| $F_{i,t}$ | friction composite (spread, IVOL, retail share) | $\mathbb{R}_{>0}$ |
| $\sigma_{i,t}$ | return volatility | $\mathbb{R}_{>0}$ |
| $\eta_i$ | Bouchaud-style impact coefficient | $\mathbb{R}_{>0}$ |
| $\kappa$ | risk aversion | $\mathbb{R}_{>0}$ |
| $\mathbf{X}_{i,t}$ | controls (size, B/M, momentum, etc.) | $\mathbb{R}^k$ |

---

## Equation 1 — Signal-return relationship with state-dependent decay

The conditional expected return at horizon $h$ given a sentiment signal $S_{i,t}$ is:

$$
\mathbb{E}\!\left[r_{i,t+h} \mid S_{i,t}, \mathbf{X}_{i,t}\right] \;=\; \beta_{i,t}\, S_{i,t}\, e^{-h\,\lambda_{i,t}} \;+\; \mathbf{X}_{i,t}'\,\boldsymbol{\gamma}
\tag{1}
$$

**Interpretation.** The predictive coefficient on sentiment decays exponentially in horizon at rate $\lambda_{i,t}$. Setting $\lambda_{i,t} \equiv \bar{\lambda}$ recovers the standard Tetlock (2007) / Loughran–McDonald (2011) horizon regressions; letting $\lambda$ vary across $(i, t)$ turns the signal into a state-dependent predictor.

**Identification.** $\beta_{i,t}$ and $\lambda_{i,t}$ are jointly identified by observing returns at multiple horizons after a sentiment event. The standard estimation moment is:

$$
\widehat{\beta}(h) \;=\; \frac{\mathrm{Cov}(r_{i,t+h},\, S_{i,t})}{\mathrm{Var}(S_{i,t})} \;\approx\; \beta_{i,t}\, e^{-h\,\lambda_{i,t}}
$$

Estimated $\widehat{\beta}(h)$ across $h = 1, \dots, H$ is then fit to $\beta_0 e^{-h\lambda}$ by nonlinear least squares, recovering $(\widehat\beta_0, \widehat\lambda)$ and hence $\widehat\tau = \ln 2 / \widehat\lambda$.

---

## Equation 1′ — The contaminated observation equation (added after the diagnostic finding)

Equation 1 is the *target* model; the empirical pipeline does not estimate it directly. What the canonical two-stage procedure ($\widehat\beta(h)$ → exponential fit) actually recovers, on event-driven equity data with realistic corpus density, is

$$
\widehat\tau\!_{\,\text{obs}} \;=\; w_1\, \tau\!_{\,\text{sentiment}} \;+\; w_2\, \tau\!_{\,\text{autocorr}} \;+\; w_3\, \tau\!_{\,\text{volclust}} \;+\; w_4\, \tau\!_{\,\text{eventclust}} \;+\; \varepsilon.
\tag{1$'$}
$$

The four components are:

| Component | Source | Identifying diagnostic (see `paper_diagnostic.md`, §3) |
|---|---|---|
| $\tau\!_{\,\text{sentiment}}$ | The quantity of interest. The decay rate of $\mathbb{E}[r \mid S]$ given a real, signed sentiment shock. | Whatever survives the three nulls below. |
| $\tau\!_{\,\text{autocorr}}$ | Apparent decay produced by short-horizon return autocorrelation in the underlying price process — sampled at event dates. | **Placebo test.** Replace $S$ with white noise on the same returns; refit. If $\widehat\tau\!_{\,\text{obs}}$ is unchanged, $w_1 \approx 0$. |
| $\tau\!_{\,\text{volclust}}$ | Apparent decay produced by volatility-regime shifts that change the level and persistence of return autocorrelation across the rolling Stage 1 window. | **Vol-regime stratification.** Within VIX terciles, re-run the pre/post test. Effect that vanishes inside a regime is regime contamination. |
| $\tau\!_{\,\text{eventclust}}$ | Apparent decay produced by post-event drift unrelated to the sentiment signal — i.e., the part of $\mathbb{E}[r \mid t]$ that depends on event timing alone. | **Pseudo-event null.** Randomize event timing while preserving the marginal $S$ distribution; refit. If $\widehat\tau\!_{\,\text{obs}}$ is unchanged, $w_4$ is the floor. |

The weights $w_1, \dots, w_4$ depend on three observable corpus properties:

- **Sentiment density** $\rho_S = \Pr(S \neq 0)$. As $\rho_S \to 0$, Stage 1 OLS regresses returns on a near-constant; the variance share of $S$ in $r$ collapses; $w_1 \to 0$.
- **Macro-window stationarity.** When the Stage 1 rolling window spans regime changes, the homogeneity assumption on $\widehat\beta(h)$ within the window fails; $w_3$ grows.
- **Event-information correlation.** When events are common but their information content is weakly correlated with their timing (8-K filings clustered at quarter-ends, scheduled announcements), $w_4$ grows.

**Pilot evidence (S&P 500, 2020–2024, 8-K + LM dictionary).** All three nulls are rejected (`empirical/output/diagnostic_results.json`):

- Real-data $\widehat\tau\!_{\,\text{obs}}$ median = 1.31 days. Placebo (white-noise $S$) $\widehat\tau$ median = 1.56–1.85 days. Indistinguishable. → $w_1 \approx 0$.
- Pooled post-2023 $\Delta \log \widehat\tau = +0.41$ ($p = 0.006$). Within mid-VIX tercile $\Delta \log \widehat\tau = +0.06$ ($p = 0.84$). → effect is regime-contaminated.
- Pseudo-event $\widehat\tau$ median = 1.91 days. Indistinguishable from real-event $\widehat\tau$. → event timing carries no information beyond the marginal distributions.

**Implication for Equation 2.** The Stage 3 cross-sectional regression $\log \widehat\tau = \log\lambda_0^{-1} - \alpha\log A - \gamma\log C + \phi\log F + u$ is identified as a regression on $\tau\!_{\,\text{sentiment}}$ only when $w_1$ dominates. Under sparse-corpus, regime-spanning conditions, the regression coefficients $\widehat\alpha, \widehat\phi, \widehat\gamma$ are jointly identified by the AI / attention / friction mechanism *and* by uncontrolled covariation of $A_{i,t}, F_{i,t}, C_{i,t}$ with the contamination components. The corrected estimator (`paper_diagnostic.md`, §7) — corpus densification + vol-regime stratification + pseudo-event subtraction — is the precondition for a clean test of Equation 2.

The estimator-validation logic is now: a Stage 3 result on real data is informative about Equation 2 only if Equations 1 and 1′ are jointly satisfied with $w_1 \gg w_2 + w_3 + w_4$.

---

## Equation 2 — Attention-driven decay hazard

The decay hazard rate is multiplicative in attention, competition, and (inversely) friction:

$$
\lambda_{i,t} \;=\; \lambda_0 \cdot A_{i,t}^{\alpha} \cdot C_{i,t}^{\gamma} \cdot F_{i,t}^{-\phi},
\qquad \alpha, \gamma, \phi \geq 0.
\tag{2}
$$

In log-linear form (the estimation specification):

$$
\log \lambda_{i,t} \;=\; \log \lambda_0 \;+\; \alpha \log A_{i,t} \;+\; \gamma \log C_{i,t} \;-\; \phi \log F_{i,t} \;+\; u_{i,t}
$$

**Interpretation.**

- $\alpha > 0$: attention compresses the predictive horizon (the result Ben-Rephael, Da, & Israelsen (2017) document non-parametrically).
- $\gamma > 0$: crowding accelerates decay, as in McLean & Pontiff (2016) post-publication evidence and Khandani & Lo (2007) on quant unwinds.
- $\phi > 0$: friction extends the predictive horizon by impeding fast price adjustment (consistent with Bouchaud et al., 2018 on impact and Lopez-Lira & Tang, 2023 on the small-cap concentration of LLM effects).

**Hypothesis H1 (formal).** $\alpha > 0$ and $\phi > 0$.

The competition term $C_{i,t}$ is hardest to measure ex ante; the empirical pipeline treats $\gamma$ as a secondary parameter, identified from cross-vintage variation in foundation-model deployment (see `empirical/README.md`, §4.3).

---

## Equation 3 — Optimal position size (Kyle–Bouchaud)

Given $S_{i,t}$, the trader chooses position $q$ to maximize expected payoff net of variance and Bouchaud-style market impact:

$$
\max_{q}\; q \cdot \underbrace{\int_0^\infty \mathbb{E}\!\left[r_{i,t+h} \mid S_{i,t}\right]\,dh}_{\text{cumulative signal value}} \;-\; \tfrac{\kappa}{2}\, q^2 \sigma_{i,t}^2 \;-\; \eta_i\, |q|^{1+\delta}\, \sigma_{i,t}.
$$

The cumulative signal value, integrating Equation 1 over horizon, is:

$$
\int_0^\infty \beta_{i,t}\,S_{i,t}\, e^{-h\lambda_{i,t}}\, dh \;=\; \frac{\beta_{i,t}\, S_{i,t}}{\lambda_{i,t}} \;=\; \frac{\beta_{i,t}\, S_{i,t}\, \tau_{i,t}}{\ln 2}.
$$

For linear impact ($\delta = 0$) the FOC has a closed form:

$$
q^*_{i,t} \;=\; \frac{1}{\kappa\, \sigma_{i,t}^2}\!\left[\frac{\beta_{i,t}\, S_{i,t}\, \tau_{i,t}}{\ln 2} \;-\; \eta_i\, \sigma_{i,t}\, \mathrm{sgn}(q^*_{i,t})\right]_{+}.
\tag{3}
$$

For Bouchaud's empirically-supported $\delta \approx 0.5$ (square-root impact), $q^*$ has no closed form but the comparative statics are unchanged:

$$
\frac{\partial q^*_{i,t}}{\partial S_{i,t}} > 0,\quad
\frac{\partial q^*_{i,t}}{\partial \tau_{i,t}} > 0,\quad
\frac{\partial q^*_{i,t}}{\partial \sigma_{i,t}} < 0,\quad
\frac{\partial q^*_{i,t}}{\partial \eta_i} < 0.
$$

**Interpretation.** Position size is *linear in $\tau_{i,t}$* for a fixed sentiment score. This is the operational consequence of treating $\tau$ as a state variable: the same nominal $S = +0.8$ generates a position in a small-cap retail-attended stock that is roughly an order of magnitude larger than the position in a mega-cap, institutionally-attended name (since $\tau$ differs by an order of magnitude per the worked example in §4.5 of the main paper).

---

## Closing the loop: how the three equations interact

The three equations form a closed system that maps directly onto the four ASL layers:

| Layer | Object | Equation |
|---|---|---|
| L1: Ingestion | $S_{i,t}$ measured at latency $\ell_{i,t}$ | input to (1) |
| L2: Interpretation | $\beta_{i,t}, S_{i,t}$ | (1) |
| L3: Sizing | $q^*_{i,t}$ | (3) |
| L4: Decay monitoring | $\widehat\tau_{i,t}$ updated continuously, $\alpha, \gamma, \phi$ | (1) → (2) feedback |

The feedback loop (Figure 1 in the main paper) is concretely the dependence of L3 on $\widehat\tau$ from L4, and the dependence of L4 on the realized residuals of (1).

---

## Testable implications beyond H1

The formalization sharpens four predictions, each falsifiable with the pipeline in `empirical/`:

1. **Half-life heterogeneity (H1).** $\widehat\tau_{i,t}$ varies systematically with $A_{i,t}$ and $F_{i,t}$ in the directions specified by Equation 2.
2. **Position-size monotonicity.** Out-of-sample, portfolios sized by $q^*$ from (3) outperform portfolios sized by $S$ alone, with the gap concentrated in the cross-sectional tails of $\widehat\tau$.
3. **Decay shock around model releases.** $\widehat\tau$ for AI-derived signals exhibits step decreases around major foundation-model releases, identifying $\gamma > 0$ via difference-in-differences.
4. **Falsification.** Replacing $S_{i,t}$ with random noise produces $\widehat\beta_{i,t} \approx 0$ and $\widehat\tau$ uncorrelated with $A, F$ — a clean null.

---

## Limitations of the formalization

This is a partial-equilibrium model with three deliberate simplifications:

- **No general equilibrium.** $A, C, F$ are taken as exogenous to the trader. A full equilibrium model would close the loop between $q^*$ and $C$ (more deployed AI → more competition → faster decay → lower $q^*$).
- **No regime switching.** $\beta$ and $\lambda$ are assumed locally constant in $t$ for estimation. Lo's (2004) Adaptive Markets logic suggests they should be regime-dependent; this is left to extensions.
- **Risk-neutral integration.** Equation 3 integrates expected returns to $\infty$. In practice the position is closed when $\widehat\beta_{i,t}\, e^{-h\lambda} < $ trading cost; a finite-horizon version is straightforward but loses the closed-form clarity.

These simplifications are the cost of an estimable model. The empirical pipeline below tests Equations 1 and 2 directly.
