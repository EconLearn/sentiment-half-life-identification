"""Sentiment scoring for headlines / 8-K disclosures.

Two backends:
  - LM dictionary (default, free): Loughran-McDonald-style word counting,
    bounded to a tight financial-tone subset. Use for prototyping and as a
    falsification check against the LLM scores.
  - Anthropic LLM (optional, paid): structured JSON output, persisted to
    parquet by content hash so re-runs do not re-bill.

Both return rows with: firm_id, timestamp, sentiment in [-1,1], confidence
in [0,1], topic.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import pandas as pd

CACHE_DIR = Path(__file__).resolve().parents[1] / "cache" / "sentiment"
CACHE_DIR.mkdir(parents=True, exist_ok=True)

PROMPT_TEMPLATE = """You are a financial analyst. Read the following SEC 8-K disclosure for {ticker}.
Return JSON with three fields:
  sentiment: a float in [-1, 1] (negative = bearish for equity, positive = bullish)
  confidence: a float in [0, 1] (how strongly the text signals direction)
  topic: one of [earnings, mna, regulation, product, officer_change, other]

Filing item: {headline}
Lead paragraph:
{lead_paragraph}

Return only the JSON object, no surrounding prose.
"""


@dataclass
class SentimentScore:
    sentiment: float
    confidence: float
    topic: str


# ---------------------------------------------------------------------------
# LM-style dictionary baseline (no API cost)
# ---------------------------------------------------------------------------
# Hand-curated financial-tone vocabulary drawn from Loughran-McDonald (2011).
# This is intentionally a tight subset (~60 words) to keep precision high on
# 8-K disclosures, where the dominant Items are 2.02 (earnings) and 1.01
# (material agreements). For full LM coverage, swap in the canonical CSV.

_LM_POS = {
    "exceeded", "exceed", "beat", "beats", "strong", "strongly", "record",
    "growth", "grew", "growing", "increase", "increased", "increases",
    "improved", "improvement", "outperformed", "outperform", "raised",
    "raises", "raise", "above", "ahead", "robust", "favorable", "gains",
    "gained", "rose", "highest", "expanded", "accelerated", "achieved",
    "successful", "successfully", "positive", "profitable", "milestone",
    "exceeded expectations", "approved", "approval",
}

_LM_NEG = {
    "missed", "miss", "weak", "weaker", "decline", "declined", "declines",
    "decrease", "decreased", "decreases", "below", "lowered", "lowers",
    "reduce", "reduced", "lawsuit", "litigation", "investigation",
    "subpoena", "restatement", "restate", "impairment", "loss", "losses",
    "deficit", "downturn", "shortfall", "underperformed", "underperform",
    "warning", "warned", "negative", "adverse", "unfavorable", "fell",
    "lowest", "delayed", "delay", "concern", "concerns", "fraud",
    "deficiencies", "material weakness", "going concern", "default",
    "bankruptcy", "terminated", "termination", "resigned", "resignation",
}

_TOPIC_KEYWORDS = {
    "earnings": ["item 2.02", "results of operations", "earnings", "quarterly results"],
    "mna": ["item 1.01", "material definitive agreement", "merger", "acquisition"],
    "regulation": ["item 8.01", "investigation", "subpoena", "regulatory"],
    "officer_change": ["item 5.02", "officer", "director", "resignation", "appointment"],
}

_WORD = re.compile(r"\b[a-z][a-z\-]+\b")


def _classify_topic(text_lower: str) -> str:
    for topic, keywords in _TOPIC_KEYWORDS.items():
        if any(k in text_lower for k in keywords):
            return topic
    return "other"


def score_dictionary(headline: str, lead_paragraph: str) -> SentimentScore:
    """Loughran-McDonald-style positive/negative word counting.

    Sentiment = (n_pos - n_neg) / max(n_pos + n_neg, 1).
    Confidence scales with total tone words / sqrt(doc length).
    """
    text = f"{headline}\n{lead_paragraph}".lower()
    tokens = _WORD.findall(text)
    if not tokens:
        return SentimentScore(0.0, 0.0, "other")
    pos = sum(1 for t in tokens if t in _LM_POS)
    neg = sum(1 for t in tokens if t in _LM_NEG)
    tone_words = pos + neg
    if tone_words == 0:
        return SentimentScore(0.0, 0.0, _classify_topic(text))
    sentiment = (pos - neg) / tone_words
    confidence = min(1.0, tone_words / max(1.0, len(tokens) ** 0.5))
    return SentimentScore(
        sentiment=float(sentiment),
        confidence=float(confidence),
        topic=_classify_topic(text),
    )


# ---------------------------------------------------------------------------
# Anthropic LLM backend (paid)
# ---------------------------------------------------------------------------

def _content_hash(headline: str, lead_paragraph: str, model: str) -> str:
    h = hashlib.sha256()
    h.update(model.encode())
    h.update(b"\x00")
    h.update(headline.encode())
    h.update(b"\x00")
    h.update(lead_paragraph.encode())
    return h.hexdigest()[:16]


def parse_llm_response(raw: str) -> SentimentScore:
    """Robust JSON extraction from an LLM response."""
    lo = raw.find("{")
    hi = raw.rfind("}")
    if lo == -1 or hi == -1 or hi < lo:
        return SentimentScore(0.0, 0.0, "other")
    try:
        data = json.loads(raw[lo:hi + 1])
        return SentimentScore(
            sentiment=float(data.get("sentiment", 0.0)),
            confidence=float(data.get("confidence", 0.0)),
            topic=str(data.get("topic", "other")),
        )
    except (json.JSONDecodeError, ValueError, TypeError):
        return SentimentScore(0.0, 0.0, "other")


def score_anthropic(
    *,
    ticker: str,
    timestamp: str,
    headline: str,
    lead_paragraph: str,
    model: str = "claude-haiku-4-5-20251001",
    client=None,
) -> SentimentScore:
    """Single-headline LLM scoring with on-disk caching by content hash."""
    key = _content_hash(headline, lead_paragraph, model)
    cache_file = CACHE_DIR / f"{key}.json"
    if cache_file.exists():
        d = json.loads(cache_file.read_text())
        return SentimentScore(d["sentiment"], d["confidence"], d["topic"])

    if client is None:
        try:
            import anthropic
        except ImportError:
            raise RuntimeError(
                "Anthropic SDK not installed. `pip install anthropic`, or use "
                "score_dictionary() for a no-API fallback."
            )
        client = anthropic.Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))

    prompt = PROMPT_TEMPLATE.format(
        ticker=ticker, headline=headline, lead_paragraph=lead_paragraph,
    )
    msg = client.messages.create(
        model=model,
        max_tokens=200,
        messages=[{"role": "user", "content": prompt}],
    )
    text = msg.content[0].text if msg.content else ""
    score = parse_llm_response(text)
    cache_file.write_text(json.dumps({
        "sentiment": score.sentiment,
        "confidence": score.confidence,
        "topic": score.topic,
        "model": model,
        "ticker": ticker,
        "timestamp": str(timestamp),
    }))
    return score


# ---------------------------------------------------------------------------
# Batch dispatch
# ---------------------------------------------------------------------------

def score_headlines_batch(
    headlines: pd.DataFrame,
    backend: str = "dictionary",
    model: str = "claude-haiku-4-5-20251001",
    max_rows: Optional[int] = None,
    sleep_sec: float = 0.0,
) -> pd.DataFrame:
    """Score a DataFrame of headlines.

    Input schema: firm_id, timestamp, headline, lead_paragraph.
    Output schema: input + sentiment, confidence, topic.

    backend:
        "dictionary" — Loughran-McDonald word counting (free, fast).
        "anthropic"  — Claude API; requires ANTHROPIC_API_KEY env var.
                       Cached by content hash so re-runs do not re-bill.
    """
    df = headlines.copy()
    if max_rows is not None:
        df = df.head(max_rows)

    sentiments, confidences, topics = [], [], []
    if backend == "dictionary":
        for _, r in df.iterrows():
            s = score_dictionary(r.get("headline", ""), r.get("lead_paragraph", ""))
            sentiments.append(s.sentiment)
            confidences.append(s.confidence)
            topics.append(s.topic)
    elif backend == "anthropic":
        from tqdm import tqdm
        for _, r in tqdm(df.iterrows(), total=len(df), desc="Anthropic scoring"):
            s = score_anthropic(
                ticker=r.get("firm_id", ""),
                timestamp=str(r.get("timestamp", "")),
                headline=r.get("headline", ""),
                lead_paragraph=r.get("lead_paragraph", ""),
                model=model,
            )
            sentiments.append(s.sentiment)
            confidences.append(s.confidence)
            topics.append(s.topic)
            if sleep_sec:
                time.sleep(sleep_sec)
    else:
        raise ValueError(f"Unknown backend: {backend}")

    df = df.copy()
    df["sentiment"] = sentiments
    df["confidence"] = confidences
    df["topic"] = topics
    return df
