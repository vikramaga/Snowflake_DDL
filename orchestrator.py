"""
Master Orchestrator
--------------------
Runs all three agents in the correct order:
  1. DevOps Agent     — fix any broken workflows first
  2. Scanner Pipeline — scan + rank tickers
  3. Research Agent   — deep-dive top picks → email report

Single entry point for GitHub Actions.
"""

import os
import sys
from datetime import datetime

def main():
    print("\n" + "=" * 55)
    print("  MASTER ORCHESTRATOR (cost-optimized)")
    print(f"  {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print("=" * 55)

    # ── 1. Scanner Pipeline (collect top tickers) ─────────────────────────────
    print("\n[1/2] Running Scanner Pipeline...")
    try:
        from pipeline_connector import run_pipeline
        tickers = run_pipeline()
    except Exception as e:
        print(f"  ⚠️  Scanner Pipeline error: {e}")
        tickers = os.environ.get("FALLBACK_TICKERS", "HOOD,CRDO,ALAB").split(",")
        print(f"  Using fallback tickers: {tickers}")

    # ── 2. Research Agent (deep-dive + email) ─────────────────────────────────
    if tickers:
        print(f"\n[2/2] Running Research Agent on {len(tickers)} tickers...")
        try:
            from research_agent import run_research_agent
            run_research_agent(tickers)
        except Exception as e:
            print(f"  ⚠️  Research Agent error: {e}")
    else:
        print("\n[2/2] Skipping Research Agent — no tickers from scanner.")

    print("\n" + "=" * 55)
    print("  ORCHESTRATOR COMPLETE")
    print("=" * 55 + "\n")


if __name__ == "__main__":
    main()
