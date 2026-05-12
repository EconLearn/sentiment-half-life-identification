# Data Specification

Precise definitions for every variable used in the pipeline. Aligns with the formal model in `../formal_model.md`.

---

## Sample

- **Universe:** common stocks (CRSP share codes 10, 11) listed on NYSE, NASDAQ, AMEX.
- **Period:** 2018-01-01 to 2025-12-31. Estimation window for headlines restricted to 2024-01-01 onward to mitigate LLM train–test contamination (see main paper §3.1).
- **Filters:**
  - Price ≥ \$1 at $t-1$.
  - Market cap ≥ \$50M at $t-1$.
  - Has at least 60 trading days of returns in the prior year.
  - Has at least 1 news headline in the sample (otherwise unidentified).

---

## Outcome variable

| Symbol | Definition | Frequency | Source |
|---|---|---|---|
| $r_{i,t+h}$ | $\log P_{i,t+h} - \log P_{i,t}$, where $t$ is the first trading day after headline timestamp | daily, $h = 1, \dots, 20$ | CRSP `dlret`-adjusted |

Returns are cum-dividend, delisting-return-adjusted. For overnight news (timestamped after 16:00 ET), $t$ is set to the next trading day's close.

---

## Sentiment input

| Symbol | Definition | Source |
|---|---|---|
| $S_{i,t}$ | LLM zero-shot sentiment score on headline + first paragraph | OpenAI / Anthropic API or open-weight inference |
| $w_{i,t}$ | LLM-reported confidence, $w \in [0, 1]$ | same |
| $\text{topic}_{i,t}$ | one of {earnings, M&A, regulation, product, macro, other} | same |

**Prompt template:**
```
You are a financial analyst. Read the following news item about ${TICKER}.
Return JSON with three fields:
  sentiment: a float in [-1, 1]
  confidence: a float in [0, 1]
  topic: one of [earnings, M&A, regulation, product, macro, other]

News (published ${TIMESTAMP}):
${HEADLINE}
${LEAD_PARAGRAPH}
```

A held-out hand-labeled set of ~500 headlines (per topic, balanced) calibrates the LLM output. Calibration target: Spearman $\rho \geq 0.65$ vs. human labels.

---

## Attention proxies

| Symbol | Definition | Source |
|---|---|---|
| $A_{i,t}^{\text{AIA}}$ | dummy = 1 if Bloomberg AIA score is 3 or 4 (intense reading day) | Bloomberg via WRDS |
| $A_{i,t}^{\text{SVI}}$ | log Google Trends weekly SVI for ticker, demeaned by firm and time | `pytrends` API |
| $A_{i,t}^{\text{coverage}}$ | count of analysts covering firm $i$ in I/B/E/S month $t$ | I/B/E/S |

The pipeline uses $A^{\text{AIA}}$ if available, otherwise the standardized $A^{\text{SVI}}$. $A^{\text{coverage}}$ is a robustness alternative that captures *latent* rather than *active* attention.

---

## Friction composite

$F_{i,t}$ is the equally-weighted average of three z-scored components:

| Component | Definition | Notes |
|---|---|---|
| Bid–ask spread | Corwin–Schultz (2012) high–low estimator, 21-day rolling mean | requires daily H, L only — works on Yahoo data |
| Idiosyncratic volatility | std. dev. of residuals from FF5 regression on prior 60 trading days | Fama–French five-factor data from Ken French website |
| Retail order share | Boehmer, Jones, Zhang & Zhang (2021) sub-penny price method, 21-day average | requires TAQ; substitute Robintrack-style holdings flow if TAQ unavailable |

All three are normalized to N(0,1) cross-sectionally each month, then averaged. $F_{i,t}$ is the resulting composite (lower = more liquid; higher = more friction).

---

## Competition / crowding proxy

$C_{i,t}$ is implemented as a step function across the sample:

| Date | Event | $\Delta C$ |
|---|---|---|
| 2022-11-30 | ChatGPT public release | +1 |
| 2023-03-14 | GPT-4 release | +1 |
| 2023-09-27 | Llama 2 open release | +1 |
| 2024-03-04 | Claude 3 family | +1 |
| 2024-05-13 | GPT-4o | +1 |
| 2024-07-23 | Llama 3.1-405B | +1 |
| 2025-…(populate per release calendar) | … | +1 |

Cumulative $C_{i,t}$ is firm-invariant by construction; the step variation identifies $\gamma$ via difference-in-differences in the release event study (`README.md` §4.3).

---

## Controls $\mathbf{X}_{i,t}$

| Variable | Definition | Source |
|---|---|---|
| Size | log market cap at $t-1$ | CRSP × shares × price |
| B/M | book equity / market equity, prior fiscal year | Compustat (CEQ + TXDB − PSTKL) / market cap |
| MOM | cumulative return from $t-252$ to $t-21$ | CRSP |
| REV | return from $t-21$ to $t-1$ | CRSP |
| Analyst dispersion | std. dev. of EPS forecasts, prior month | I/B/E/S |
| Industry | SIC-3 group | Compustat |

---

## Half-life estimate $\widehat\tau_{i,t}$

Output of Stage 2 estimation (`src/halflife.py`). Computed firm-by-firm on a rolling 252-trading-day window of headlines:

1. For each headline $k$ in the window, observe returns $r_{i,t_k+h}$ for $h = 1, \dots, 20$.
2. Run cross-headline regression at each $h$:
   $$r_{i,t_k+h} = \alpha_h + \beta(h) S_{i,t_k} + \mathbf{X}'_{i,t_k}\gamma_h + \varepsilon$$
3. Fit $\widehat\beta(h) = \beta_0 \exp(-h\lambda)$ by NLS, weighted by $1/\mathrm{Var}(\widehat\beta(h))$.
4. Recover $\widehat\tau = \ln 2 / \widehat\lambda$.

A firm enters the panel only with $\geq 25$ headlines in the rolling window, otherwise $\widehat\tau$ is too noisy to use.

---

## Panel structure

The final estimation panel is at the **firm-month** level: $\widehat\tau_{i,t}$ is one observation per firm per month, regressed on month-end values of $A_{i,t}, F_{i,t}, C_{i,t}, \mathbf{X}_{i,t}$.

Expected panel size:
- ~3,000 firms with sufficient news
- ~96 months (2018–2025)
- ~150,000–200,000 firm-month observations after filters
