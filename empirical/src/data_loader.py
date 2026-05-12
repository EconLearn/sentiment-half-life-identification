"""Data acquisition — free-tier implementations.

Live sources:
  - load_returns          -> yfinance (CRSP substitute)
  - load_news_headlines   -> SEC EDGAR 8-K filings (clean structured corpus)
  - load_attention_svi    -> Google Trends via pytrends
  - load_microstructure   -> Corwin-Schultz spread derived from yfinance OHLC

Vendor-only stubs (raise NotImplementedError):
  - load_fundamentals     -> Compustat (WRDS license)
  - load_attention_aia    -> Bloomberg Abnormal Institutional Attention

All live loaders cache to parquet under empirical/cache/ to avoid re-fetch.
The SEC requires a User-Agent header per their fair-access policy; set
SEC_USER_AGENT env var to your "Name email@domain" string before calling
EDGAR endpoints.
"""

from __future__ import annotations

import os
import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Iterable, Optional

import pandas as pd

CACHE_DIR = Path(__file__).resolve().parents[1] / "cache"
CACHE_DIR.mkdir(exist_ok=True)

SEC_USER_AGENT = os.environ.get(
    "SEC_USER_AGENT",
    "ASL Research Pipeline contact@example.com",
)
SEC_HEADERS = {"User-Agent": SEC_USER_AGENT, "Accept-Encoding": "gzip, deflate"}
SEC_RATE_LIMIT_SEC = 0.11  # ~9 req/sec, under SEC's 10 req/sec cap
_SEC_LOCK = threading.Lock()
_SEC_LAST = [0.0]


# ---------------------------------------------------------------------------
# Universe
# ---------------------------------------------------------------------------

def load_sp500_universe() -> pd.DataFrame:
    """Pull current S&P 500 constituents from Wikipedia.

    Schema: ticker (str), name (str), sector (str), cik (str, 10-digit zero-padded).

    Note: this is the *current* membership, not point-in-time. For published work,
    use the CRSP S&P 500 historical constituents file. For a first-pass empirical,
    survivorship bias attenuates magnitudes but does not flip signs in the H1
    cross-section.
    """
    cache = CACHE_DIR / "sp500_universe.parquet"
    if cache.exists():
        return pd.read_parquet(cache)

    import requests
    from io import StringIO
    url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
    r = requests.get(url, headers={"User-Agent": SEC_USER_AGENT}, timeout=30)
    r.raise_for_status()
    tables = pd.read_html(StringIO(r.text))
    df = tables[0].rename(columns={
        "Symbol": "ticker",
        "Security": "name",
        "GICS Sector": "sector",
        "CIK": "cik",
    })[["ticker", "name", "sector", "cik"]]
    df["ticker"] = df["ticker"].str.replace(".", "-", regex=False)  # yfinance uses BRK-B
    df["cik"] = df["cik"].astype(str).str.zfill(10)
    df.to_parquet(cache)
    return df


def _ticker_to_cik() -> dict[str, str]:
    """SEC's authoritative ticker -> CIK mapping (refreshed daily)."""
    cache = CACHE_DIR / "company_tickers.parquet"
    if cache.exists() and (time.time() - cache.stat().st_mtime) < 86400:
        return dict(zip(*pd.read_parquet(cache).T.values))

    import requests
    r = requests.get(
        "https://www.sec.gov/files/company_tickers.json",
        headers=SEC_HEADERS, timeout=30,
    )
    r.raise_for_status()
    data = r.json()
    mapping = {v["ticker"]: str(v["cik_str"]).zfill(10) for v in data.values()}
    pd.DataFrame(list(mapping.items()), columns=["ticker", "cik"]).to_parquet(cache)
    return mapping


# ---------------------------------------------------------------------------
# Returns (yfinance)
# ---------------------------------------------------------------------------

def load_returns(
    start: str,
    end: str,
    tickers: Optional[Iterable[str]] = None,
    cache_key: str = "returns_default",
) -> pd.DataFrame:
    """Daily log-returns for a ticker universe.

    Schema:
        firm_id (str, ticker), date (datetime), ret (log return),
        prc (close), open, high, low, volume.

    `firm_id` is the ticker for the yfinance backend. Persist a stable mapping
    if you later need PERMNO-level joins.
    """
    cache = CACHE_DIR / f"{cache_key}_{start}_{end}.parquet"
    if cache.exists():
        return pd.read_parquet(cache)

    if tickers is None:
        tickers = load_sp500_universe()["ticker"].tolist()
    tickers = list(tickers)

    import numpy as np
    import yfinance as yf

    raw = yf.download(
        tickers, start=start, end=end,
        auto_adjust=True, progress=False, threads=True,
        group_by="ticker",
    )
    rows = []
    for t in tickers:
        if t not in raw.columns.get_level_values(0):
            continue
        sub = raw[t].dropna(how="all")
        if sub.empty:
            continue
        sub = sub.reset_index().rename(columns=str.lower)
        sub["firm_id"] = t
        sub["ret"] = np.log(sub["close"]).diff()
        rows.append(sub[["firm_id", "date", "ret", "close", "open", "high", "low", "volume"]]
                   .rename(columns={"close": "prc"}))
    if not rows:
        return pd.DataFrame(columns=["firm_id", "date", "ret", "prc",
                                     "open", "high", "low", "volume"])
    out = pd.concat(rows, ignore_index=True).dropna(subset=["ret"])
    out.to_parquet(cache)
    return out


# ---------------------------------------------------------------------------
# News (EDGAR 8-Ks)
# ---------------------------------------------------------------------------

def _sec_get(url: str, **kwargs):
    """Rate-limited SEC fetch. Global lock enforces SEC's 10 req/sec cap even
    when called from a thread pool (used by the bulk-filings fetcher)."""
    import requests
    with _SEC_LOCK:
        wait = SEC_RATE_LIMIT_SEC - (time.time() - _SEC_LAST[0])
        if wait > 0:
            time.sleep(wait)
        _SEC_LAST[0] = time.time()
    r = requests.get(url, headers=SEC_HEADERS, timeout=30, **kwargs)
    r.raise_for_status()
    return r


def _list_filings(cik: str, form: str = "8-K") -> pd.DataFrame:
    """All filings of a given form for a CIK, via SEC submissions JSON.

    Returns columns: cik, form, accession_no, filing_date, primary_doc, primary_doc_url.
    """
    cache = CACHE_DIR / "filings_index" / f"{cik}_{form}.parquet"
    cache.parent.mkdir(exist_ok=True)
    if cache.exists() and (time.time() - cache.stat().st_mtime) < 86400 * 7:
        return pd.read_parquet(cache)

    r = _sec_get(f"https://data.sec.gov/submissions/CIK{cik}.json")
    j = r.json()

    frames = []
    recent = j.get("filings", {}).get("recent", {})
    if recent and any(recent.values()):
        frames.append(pd.DataFrame(recent))
    for f in j.get("filings", {}).get("files", []):
        rr = _sec_get(f"https://data.sec.gov/submissions/{f['name']}")
        rj = rr.json()
        if rj and any(v for v in rj.values() if isinstance(v, list)):
            frames.append(pd.DataFrame(rj))

    if not frames:
        return pd.DataFrame(columns=["cik", "form", "accession_no", "filing_date",
                                     "primary_doc", "primary_doc_url"])

    frames = [f for f in frames if not f.empty]
    if not frames:
        return pd.DataFrame(columns=["cik", "form", "accession_no", "filing_date",
                                     "primary_doc", "primary_doc_url"])
    df = pd.concat(frames, ignore_index=True)
    df = df[df["form"] == form].copy()
    if df.empty:
        out = pd.DataFrame(columns=["cik", "form", "accession_no", "filing_date",
                                    "primary_doc", "primary_doc_url"])
        out.to_parquet(cache)
        return out

    df["cik"] = cik
    df["accession_no"] = df["accessionNumber"]
    df["filing_date"] = pd.to_datetime(df["filingDate"])
    df["primary_doc"] = df["primaryDocument"]
    cik_int = int(cik)
    df["primary_doc_url"] = df.apply(
        lambda r: (f"https://www.sec.gov/Archives/edgar/data/{cik_int}/"
                   f"{r['accession_no'].replace('-', '')}/{r['primary_doc']}"),
        axis=1,
    )
    out = df[["cik", "form", "accession_no", "filing_date",
              "primary_doc", "primary_doc_url"]].copy()
    out.to_parquet(cache)
    return out


_8K_ITEM_RE = re.compile(
    r"Item\s+(\d+\.\d+)[\s\.\-:]+([^\.\n]{5,200})",
    flags=re.IGNORECASE,
)


def _parse_8k(html: str) -> tuple[str, str]:
    """Extract a representative headline + lead paragraph from an 8-K HTML body.

    The "headline" is the first matched Item heading (e.g.,
    "Item 2.02 Results of Operations and Financial Condition"). The "lead
    paragraph" is the first ~600 chars of plain text after that heading.
    Falls back to the first non-empty paragraph if no Item heading is found.
    """
    from bs4 import BeautifulSoup
    parser = "lxml-xml" if html.lstrip().startswith("<?xml") else "lxml"
    soup = BeautifulSoup(html, parser)
    text = soup.get_text(separator="\n")
    text = re.sub(r"\n{2,}", "\n\n", text)
    text = re.sub(r"[ \t]+", " ", text)

    m = _8K_ITEM_RE.search(text)
    if m:
        item_no, item_title = m.group(1), m.group(2).strip().rstrip(".")
        headline = f"Item {item_no} {item_title}"
        body_start = m.end()
        lead = text[body_start:body_start + 600].strip()
    else:
        paras = [p.strip() for p in text.split("\n\n") if p.strip()]
        headline = paras[0][:200] if paras else ""
        lead = (paras[1] if len(paras) > 1 else "")[:600]
    return headline, lead


def _fetch_body(accession_no: str, url: str) -> tuple[str, str]:
    cache = CACHE_DIR / "filings_text" / f"{accession_no}.txt"
    cache.parent.mkdir(exist_ok=True)
    if cache.exists():
        return _parse_8k(cache.read_text(errors="ignore"))
    try:
        body = _sec_get(url).text
        cache.write_text(body, errors="ignore")
        return _parse_8k(body)
    except Exception as e:
        return "", f"[fetch_error: {e}]"


def load_news_headlines(
    start: str,
    end: str,
    tickers: Optional[Iterable[str]] = None,
    fetch_bodies: bool = True,
    max_workers: int = 6,
) -> pd.DataFrame:
    """8-K filings for a ticker universe over [start, end].

    Schema:
        firm_id, timestamp, headline, lead_paragraph, source ("EDGAR 8-K"), url.

    Bodies are fetched with a `max_workers`-thread pool, sharing a global
    rate-limit lock so the SEC 10 req/sec cap is respected. Caches per
    accession number for re-runs.
    """
    if tickers is None:
        universe = load_sp500_universe()
        ticker_cik = dict(zip(universe["ticker"], universe["cik"]))
    else:
        cik_map = _ticker_to_cik()
        ticker_cik = {t: cik_map[t] for t in tickers if t in cik_map}

    start_ts = pd.Timestamp(start)
    end_ts = pd.Timestamp(end)
    rows = []
    from tqdm import tqdm

    # Phase 1: per-firm filings index (sequential — usually fast).
    pending_bodies = []
    for ticker, cik in tqdm(ticker_cik.items(), desc="EDGAR index"):
        try:
            filings = _list_filings(cik, form="8-K")
        except Exception as e:
            print(f"[warn] {ticker} ({cik}): {e}")
            continue
        filings = filings[(filings["filing_date"] >= start_ts) &
                          (filings["filing_date"] <= end_ts)]
        for _, f in filings.iterrows():
            rows.append({
                "firm_id": ticker,
                "timestamp": f["filing_date"],
                "source": "EDGAR 8-K",
                "url": f["primary_doc_url"],
                "accession_no": f["accession_no"],
                "headline": "",
                "lead_paragraph": "",
            })
            if fetch_bodies:
                pending_bodies.append((len(rows) - 1, f["accession_no"], f["primary_doc_url"]))

    # Phase 2: parallel body fetch (rate-limit lock keeps global rate < 10/sec).
    if fetch_bodies and pending_bodies:
        with ThreadPoolExecutor(max_workers=max_workers) as ex:
            futures = {
                ex.submit(_fetch_body, acc, url): idx
                for idx, acc, url in pending_bodies
            }
            for fut in tqdm(as_completed(futures), total=len(futures),
                            desc="EDGAR bodies"):
                idx = futures[fut]
                try:
                    h, l = fut.result()
                except Exception as e:
                    h, l = "", f"[body_error: {e}]"
                rows[idx]["headline"] = h
                rows[idx]["lead_paragraph"] = l

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Attention (Wikipedia page views — primary; Google SVI — fallback)
# ---------------------------------------------------------------------------

def _wiki_article_candidates(name: str, ticker: str) -> list[str]:
    """Try several article-slug forms. Most S&P 500 names match directly."""
    name = (name or "").strip()
    base = re.sub(r"\s*\([^)]*\)\s*", "", name)  # strip "(Class A)" etc.
    stripped = re.sub(r"\s*(Inc\.?|Corp\.?|Co\.?|Ltd\.?|LLC|plc|N\.?V\.?)$",
                      "", base, flags=re.IGNORECASE).strip()
    cands = [
        base.replace(" ", "_"),
        stripped.replace(" ", "_"),
        f"{stripped.replace(' ', '_')}_(company)",
        ticker,
    ]
    return list(dict.fromkeys([c for c in cands if c]))


def load_attention_wikipedia(
    tickers: list[str],
    start: str,
    end: str,
    name_lookup: Optional[dict[str, str]] = None,
) -> pd.DataFrame:
    """Wikipedia page-views per company per day (Wikimedia REST API).

    Schema: ticker, date, views (int).

    Free, no auth, no DNS games. Tries several article-slug candidates per
    ticker (full name, stripped suffix, "(company)" disambiguator, ticker).
    Caches per ticker.
    """
    import requests
    cache_dir = CACHE_DIR / "wiki"
    cache_dir.mkdir(exist_ok=True)

    if name_lookup is None:
        try:
            universe = load_sp500_universe()
            name_lookup = dict(zip(universe["ticker"], universe["name"]))
        except Exception:
            name_lookup = {}

    start_fmt = pd.Timestamp(start).strftime("%Y%m%d")
    end_fmt = pd.Timestamp(end).strftime("%Y%m%d")
    headers = {"User-Agent": SEC_USER_AGENT, "Accept": "application/json"}

    rows = []
    from tqdm import tqdm
    for t in tqdm(tickers, desc="Wikipedia views"):
        cache = cache_dir / f"{t}_{start_fmt}_{end_fmt}.parquet"
        if cache.exists():
            rows.append(pd.read_parquet(cache))
            continue
        df = pd.DataFrame()
        for slug in _wiki_article_candidates(name_lookup.get(t, t), t):
            url = ("https://wikimedia.org/api/rest_v1/metrics/pageviews/"
                   f"per-article/en.wikipedia/all-access/all-agents/"
                   f"{slug}/daily/{start_fmt}/{end_fmt}")
            try:
                r = requests.get(url, headers=headers, timeout=15)
                if r.status_code != 200:
                    continue
                items = r.json().get("items", [])
                if not items:
                    continue
                df = pd.DataFrame(items)[["timestamp", "views"]]
                df["date"] = pd.to_datetime(df["timestamp"].str[:8], format="%Y%m%d")
                df = df[["date", "views"]].copy()
                df["ticker"] = t
                df["wiki_article"] = slug
                break
            except Exception:
                continue
        if df.empty:
            continue
        df.to_parquet(cache)
        rows.append(df)

    if not rows:
        return pd.DataFrame(columns=["ticker", "date", "views", "wiki_article"])
    return pd.concat(rows, ignore_index=True)


# ---------------------------------------------------------------------------
# Attention (Google SVI fallback)
# ---------------------------------------------------------------------------

def load_attention_svi(
    tickers: list[str],
    start: str,
    end: str,
    geo: str = "US",
) -> pd.DataFrame:
    """Google Search Volume Index per ticker.

    Schema: ticker, week_end (Sunday), svi (0-100, normalized within ticker).

    pytrends rate limits aggressively; this function batches to 5 tickers/req
    and sleeps between calls. Cache is per-ticker. For 500 firms expect
    ~30-60 minutes wall time.
    """
    cache_dir = CACHE_DIR / "svi"
    cache_dir.mkdir(exist_ok=True)

    from pytrends.request import TrendReq
    pt = TrendReq(hl="en-US", tz=300)
    timeframe = f"{start} {end}"

    rows = []
    for t in tickers:
        cache = cache_dir / f"{t}.parquet"
        if cache.exists():
            rows.append(pd.read_parquet(cache))
            continue
        try:
            pt.build_payload([t], timeframe=timeframe, geo=geo)
            df = pt.interest_over_time()
            if df.empty:
                continue
            df = df.reset_index().rename(columns={"date": "week_end", t: "svi"})
            df["ticker"] = t
            df = df[["ticker", "week_end", "svi"]]
            df.to_parquet(cache)
            rows.append(df)
            time.sleep(2.0)  # be polite
        except Exception as e:
            print(f"[warn] SVI {t}: {e}")
    if not rows:
        return pd.DataFrame(columns=["ticker", "week_end", "svi"])
    return pd.concat(rows, ignore_index=True)


# ---------------------------------------------------------------------------
# Microstructure (Corwin-Schultz spread from OHLC)
# ---------------------------------------------------------------------------

def load_microstructure(
    start: str,
    end: str,
    tickers: Optional[Iterable[str]] = None,
) -> pd.DataFrame:
    """Corwin-Schultz (2012) high-low spread estimator.

    Schema: firm_id, date, spread_cs, dollar_volume.

    Implements:
        beta_t = (ln(H_t/L_t))^2 + (ln(H_{t+1}/L_{t+1}))^2
        gamma_t = (ln(max(H_t,H_{t+1}) / min(L_t,L_{t+1})))^2
        alpha_t = (sqrt(2*beta) - sqrt(beta)) / (3 - 2*sqrt(2)) - sqrt(gamma/(3-2*sqrt(2)))
        S_t = 2*(exp(alpha) - 1) / (1 + exp(alpha))
    Negative S clipped to 0 per CS.
    """
    import numpy as np

    rets = load_returns(start, end, tickers=tickers, cache_key="microstructure")
    if rets.empty:
        return pd.DataFrame(columns=["firm_id", "date", "spread_cs", "dollar_volume"])

    out_rows = []
    sqrt2 = np.sqrt(2)
    denom = 3 - 2 * sqrt2
    for fid, grp in rets.groupby("firm_id"):
        g = grp.sort_values("date").reset_index(drop=True).copy()
        H, L = g["high"].values, g["low"].values
        H1, L1 = np.roll(H, -1), np.roll(L, -1)
        with np.errstate(invalid="ignore", divide="ignore"):
            beta = np.log(H / L) ** 2 + np.log(H1 / L1) ** 2
            H_max = np.maximum(H, H1)
            L_min = np.minimum(L, L1)
            gamma = np.log(H_max / L_min) ** 2
            alpha = (sqrt2 * np.sqrt(beta) - np.sqrt(beta)) / denom \
                    - np.sqrt(gamma / denom)
            spread = 2 * (np.exp(alpha) - 1) / (1 + np.exp(alpha))
        spread = np.clip(spread, 0, None)
        spread[-1] = np.nan  # last obs has no t+1
        g["spread_cs"] = spread
        g["dollar_volume"] = g["prc"] * g["volume"]
        out_rows.append(g[["firm_id", "date", "spread_cs", "dollar_volume"]])
    return pd.concat(out_rows, ignore_index=True)


# ---------------------------------------------------------------------------
# Vendor-only stubs (kept to preserve the original schema contract)
# ---------------------------------------------------------------------------

def load_fundamentals(year_range: tuple[int, int]) -> pd.DataFrame:
    """Compustat fundamentals (annual). Schema: firm_id, fyear, ceq, txdb, pstkl,
    at, sale, ni, sic.
    """
    raise NotImplementedError("Compustat requires WRDS. Skip for first-pass empirical.")


def load_attention_aia(start: str, end: str) -> pd.DataFrame:
    """Bloomberg AIA. Schema: firm_id, date, aia_score (0-4)."""
    raise NotImplementedError("Bloomberg AIA — license-gated. Use load_attention_svi.")
