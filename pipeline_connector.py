"""
Scanner → Research Pipeline Connector
--------------------------------------
Runs all your scanners, collects top-tier tickers,
deduplicates, prioritizes, then feeds into Research Agent.

Drop this file alongside your existing scanner scripts.
"""

import os
import sys
import importlib
import numpy as np
import yfinance as yf
import pandas as pd
from datetime import datetime

# ── Import your existing scanners ─────────────────────────────────────────────
# Add your scanner scripts' directory to path if needed
# sys.path.append("/path/to/your/scanners")

# We'll call each scanner as a function that returns a dict:
# { "T1": [...tickers], "T2": [...tickers], "T3": [...tickers] }
# If your scanners print results instead, see ADAPTER section below.


# ── SCANNER REGISTRY ──────────────────────────────────────────────────────────
# Register each scanner with a name and weight.
# Higher weight = scanner's picks get priority in final ranking.

SCANNER_REGISTRY = [
    {
        "name":    "SMA50 Pullback",
        "module":  "scanner_sma50",        # your Python filename (no .py)
        "fn":      "run_scan",             # function to call inside the module
        "weight":  1.0,
    },
    {
        "name":    "EMA20 + MACD Zero Cross",
        "module":  "nearema20",
        "fn":      "run_scan",
        "weight":  1.2,                    # slightly higher — momentum signal
    },
    {
        "name":    "SMA150 Breakout",
        "module":  "scanner_sma150",
        "fn":      "run_scan",
        "weight":  1.0,
    },
    {
        "name":    "MA Compression Breakout",
        "module":  "scanner_ma_compression",
        "fn":      "run_scan",
        "weight":  1.1,
    },
    {
        "name":    "SMA50/150 Retest + EMA20",
        "module":  "scanner_sma_retest",
        "fn":      "run_scan",
        "weight":  1.0,
    },
]

# ── TIER WEIGHTS ──────────────────────────────────────────────────────────────
TIER_WEIGHT = {"T1": 3, "T2": 2, "T3": 1}

# Max tickers to pass to Research Agent (cost control)
MAX_RESEARCH_TICKERS = 3                         # cost-optimized: top 3 only


# ── ADAPTER: if your scanner prints instead of returning ─────────────────────
# Wrap it like this:
#
# import io, contextlib
# from scanner_sma50 import main as sma50_main
#
# def run_scan():
#     buf = io.StringIO()
#     with contextlib.redirect_stdout(buf):
#         sma50_main()
#     output = buf.getvalue()
#     return parse_printed_output(output)
#
# def parse_printed_output(text):
#     # parse T1/T2/T3 sections from printed text
#     tiers = {"T1": [], "T2": [], "T3": []}
#     current = None
#     for line in text.splitlines():
#         if "Tier 1" in line or "T1" in line: current = "T1"
#         elif "Tier 2" in line or "T2" in line: current = "T2"
#         elif "Tier 3" in line or "T3" in line: current = "T3"
#         elif current and line.strip().isupper() and len(line.strip()) <= 6:
#             tiers[current].append(line.strip())
#     return tiers


# ── CORE: RUN ALL SCANNERS ────────────────────────────────────────────────────
def run_all_scanners() -> dict:
    """
    Calls each registered scanner, collects tiered results.
    Returns: { ticker: score } where score = weighted sum across scanners/tiers
    """
    scores = {}   # ticker → float score
    sources = {}  # ticker → list of scanner names (for logging)

    for scanner in SCANNER_REGISTRY:
        print(f"  ▶ Running {scanner['name']}...")
        try:
            mod = importlib.import_module(scanner["module"])
            fn  = getattr(mod, scanner["fn"])
            result = fn()   # expects {"T1": [], "T2": [], "T3": []}

            for tier, tickers in result.items():
                tier_w = TIER_WEIGHT.get(tier, 1)
                for ticker in tickers:
                    ticker = ticker.upper().strip()
                    points = tier_w * scanner["weight"]
                    scores[ticker]  = scores.get(ticker, 0) + points
                    sources[ticker] = sources.get(ticker, []) + [f"{scanner['name']}({tier})"]

            print(f"    ✅ {sum(len(v) for v in result.values())} tickers found")

        except Exception as e:
            print(f"    ⚠️  {scanner['name']} failed: {e}")

    return scores, sources


# ── FILTER: BASIC QUALITY GATE ────────────────────────────────────────────────
def apply_quality_gate(tickers: list[str]) -> list[str]:
    """
    Fast yfinance check — remove illiquid / bad data tickers.
    Keeps tickers with: price > $2, avg volume > 200k, market cap > $100M
    """
    passed = []
    print(f"\n  🔎 Quality gate for {len(tickers)} tickers...")
    for ticker in tickers:
        try:
            info = yf.Ticker(ticker).info
            price  = info.get("regularMarketPrice", 0) or 0
            vol    = info.get("averageVolume", 0) or 0
            mktcap = info.get("marketCap", 0) or 0
            if price > 2 and vol > 200_000 and mktcap > 100_000_000:
                passed.append(ticker)
        except:
            pass
    print(f"  ✅ {len(passed)} passed quality gate")
    return passed


# ── RANK AND SELECT ───────────────────────────────────────────────────────────
def select_top_tickers(scores: dict, sources: dict, n: int = MAX_RESEARCH_TICKERS) -> list[dict]:
    """
    Rank tickers by score, apply quality gate, return top N with metadata.
    """
    # Sort by score
    ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    top_tickers = [t for t, _ in ranked[:n * 2]]   # fetch 2x for quality gate buffer

    # Quality gate
    passed = apply_quality_gate(top_tickers)

    # Final selection
    final = []
    for ticker in [t for t, _ in ranked if t in passed][:n]:
        final.append({
            "ticker":  ticker,
            "score":   round(scores[ticker], 2),
            "sources": sources.get(ticker, []),
        })

    return final


# ── MAIN PIPELINE FUNCTION ────────────────────────────────────────────────────
def run_pipeline() -> list[str]:
    """
    Full pipeline: scan → rank → gate → return ticker list for Research Agent.
    Also prints a summary table.
    """
    print("\n" + "=" * 55)
    print("  SCANNER → RESEARCH PIPELINE")
    print(f"  {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print("=" * 55)

    print("\n📡 Running scanners...")
    scores, sources = run_all_scanners()
    print(f"\n  Total unique tickers from all scanners: {len(scores)}")

    print("\n🏆 Selecting top tickers...")
    top = select_top_tickers(scores, sources)

    # Print summary table
    print("\n┌─────────┬────────┬───────────────────────────────────┐")
    print("│ Ticker  │ Score  │ Sources                           │")
    print("├─────────┼────────┼───────────────────────────────────┤")
    for item in top:
        src_str = ", ".join(item["sources"])[:35]
        print(f"│ {item['ticker']:<7} │ {item['score']:<6} │ {src_str:<35} │")
    print("└─────────┴────────┴───────────────────────────────────┘")

    ticker_list = [item["ticker"] for item in top]
    print(f"\n✅ Passing to Research Agent: {ticker_list}\n")
    return ticker_list


# ── ENTRY POINT ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    # Standalone test — runs pipeline and prints tickers
    tickers = run_pipeline()

    # Then kick off Research Agent
    from research_agent import run_research_agent
    run_research_agent(tickers)
