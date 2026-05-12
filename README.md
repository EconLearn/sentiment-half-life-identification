# Sentiment Half-Life Is Not Separately Identified from Return Autocorrelation in Event-Driven Equity Panels

Working paper and replication code.

## What this is

A methodological paper arguing that the canonical two-stage estimator of sentiment-driven price-impact half-life — horizon-by-horizon OLS coefficients fit to an exponential decay — does *not* separately identify the structural sentiment-decay parameter from coexisting return autocorrelation, volatility-regime clustering, and event-clustering structure, under the corpus-density conditions typical of regulatory-filing studies.

Three pre-specified diagnostics (placebo, volatility-regime stratification, pseudo-event null) are run on a 500-firm S&P 500 pilot covering 2020–2024 with sentiment scored from SEC 8-K filings via the Loughran–McDonald dictionary. All three reject. A controlled Monte Carlo simulation identifies a critical signal-strength threshold above which the canonical estimator recovers the planted structural value and below which it returns a fixed noise floor. A corrected identification strategy that jointly boosts signal density and signal magnitude recovers 93% of the planted value in simulation.

## Paper

- [`paper_diagnostic.pdf`](paper_diagnostic.pdf) — full working paper (~9,400 words)
- [`paper_diagnostic.md`](paper_diagnostic.md) — source markdown
- [`formal_model.md`](formal_model.md) — the formal-model companion (Eq 1, 1′, 2, 3)

## Replication

The empirical pipeline runs end-to-end on free-tier compute. Returns come from `yfinance` as a CRSP substitute, events from the SEC EDGAR 8-K corpus, sentiment from the Loughran-McDonald dictionary, attention from Wikipedia page views, and the friction proxy from Corwin–Schultz spreads.

```bash
cd empirical
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# Generate the 500-firm panel (requires network; first run ~60 min)
python3 pipeline.py --live --start 2020-01-01 --end 2024-12-31 --n-firms 500

# Run the three diagnostics on the cached panel (~3 min)
python3 scripts/diagnostic_tests.py

# Controlled-DGP simulation evidence (~5 min)
python3 scripts/simulation_evidence.py

# Identification sweep + M1 validation (~5 min)
python3 scripts/identification_sweep.py
```

Outputs land in `empirical/output/`.

## Repository structure

```
.
├── paper_diagnostic.pdf
├── paper_diagnostic.md
├── formal_model.md
├── README.md
└── empirical/
    ├── README.md              # detailed pipeline overview
    ├── data_spec.md           # data dictionary
    ├── requirements.txt
    ├── pipeline.py            # end-to-end runner (synthetic + live modes)
    ├── src/                   # core modules
    │   ├── data_loader.py     # yfinance + EDGAR + Wikipedia loaders
    │   ├── sentiment.py       # LM dictionary + LLM scoring
    │   ├── halflife.py        # Stage 1 + Stage 2 estimators
    │   └── regressions.py     # Stage 3 H1 + diagnostics
    └── scripts/
        ├── diagnostic_tests.py    # placebo + vol-regime + pseudo-event
        ├── simulation_evidence.py # DGP A/B/C controlled simulation
        ├── identification_sweep.py# 1D β sweep + M1 validation
        ├── rerun_stage3.py        # H1 regressions on cached panel
        └── smoke_test_loaders.py  # loader sanity check
```

## Citation

If you find any of this useful, please cite as:

> Wallis, J. *Sentiment Half-Life Is Not Separately Identified from Return Autocorrelation in Event-Driven Equity Panels.* Working paper, 2026. https://github.com/EconLearn/sentiment-half-life-identification

## Contact

Comments welcome at <judehudsonwallis@icloud.com>.

## Acknowledgments

The author used Anthropic Claude for coding assistance with the empirical pipeline and for drafting and copy-editing portions of the manuscript. All claims, empirical results, and analytical conclusions were independently validated by the author, who takes full responsibility for the content.
