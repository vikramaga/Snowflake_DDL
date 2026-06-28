"""
Research Agent - Autonomous Multi-Stock Deep Dive
Uses Claude API with web_search tool to analyze stocks autonomously.
Outputs a ranked research report via email.
"""

import os
import json
import smtplib
import yfinance as yf
from datetime import datetime
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import anthropic

# ── CONFIG ────────────────────────────────────────────────────────────────────
ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY")
EMAIL_FROM        = os.environ.get("EMAIL_FROM")       # your Gmail
EMAIL_PASSWORD    = os.environ.get("EMAIL_PASSWORD")   # app password
EMAIL_TO          = os.environ.get("EMAIL_TO")         # destination

client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)

# ── STEP 1: FETCH FUNDAMENTALS FROM YFINANCE ─────────────────────────────────
def get_fundamentals(ticker: str) -> dict:
    """Pull key fundamentals locally — no API cost."""
    try:
        t = yf.Ticker(ticker)
        info = t.info
        hist = t.history(period="3mo")
        price_now  = hist["Close"].iloc[-1] if not hist.empty else None
        price_3m   = hist["Close"].iloc[0]  if not hist.empty else None
        perf_3m    = round(((price_now - price_3m) / price_3m) * 100, 1) if price_now and price_3m else None

        return {
            "ticker":           ticker,
            "name":             info.get("longName", ticker),
            "sector":           info.get("sector", "N/A"),
            "market_cap_B":     round(info.get("marketCap", 0) / 1e9, 2),
            "price":            round(price_now, 2) if price_now else None,
            "perf_3m_pct":      perf_3m,
            "pe_fwd":           info.get("forwardPE"),
            "revenue_growth":   info.get("revenueGrowth"),
            "gross_margin":     info.get("grossMargins"),
            "debt_to_equity":   info.get("debtToEquity"),
            "short_float_pct":  info.get("shortPercentOfFloat"),
            "analyst_target":   info.get("targetMeanPrice"),
            "analyst_rating":   info.get("recommendationKey", "N/A"),
        }
    except Exception as e:
        return {"ticker": ticker, "error": str(e)}


# ── HAIKU PRICING (per million tokens) ───────────────────────────────────────
HAIKU_INPUT_COST_PER_M  = 1.00   # $1.00 per 1M input tokens
HAIKU_OUTPUT_COST_PER_M = 5.00   # $5.00 per 1M output tokens
USD_TO_INR              = 84.0   # update if needed

# Global token counters for the run
_total_input_tokens  = 0
_total_output_tokens = 0


# ── STEP 2: CLAUDE RESEARCH AGENT (agentic loop) ─────────────────────────────
def research_ticker(ticker: str, fundamentals: dict) -> str:
    """
    Runs Claude as an autonomous research agent.
    Claude decides what to search, iterates until satisfied, returns analysis.
    Tracks token usage globally.
    """
    global _total_input_tokens, _total_output_tokens

    system_prompt = """Equity research analyst. Use web_search once for recent news + catalysts.
Output exactly:
TICKER: X
VERDICT: BUY/HOLD/AVOID
CONVICTION: High/Medium/Low
CATALYSTS: 2 bullets
RISKS: 2 bullets
SUMMARY: 2 sentences
SCORE: 1-10"""

    user_msg = f"""Research this stock thoroughly:

Ticker: {ticker}
Fundamentals:
{json.dumps(fundamentals, indent=2)}

Use web search to find recent news and catalysts. Then give your verdict."""

    messages = [{"role": "user", "content": user_msg}]
    tools = [{"type": "web_search_20250305", "name": "web_search"}]

    ticker_input  = 0
    ticker_output = 0

    # Agentic loop — Claude searches autonomously until done
    max_iterations = 2                      # cost-optimized: 2 searches max
    for _ in range(max_iterations):
        response = client.messages.create(
            model="claude-haiku-4-5-20251001",  # cost-optimized: 3x cheaper than Sonnet
            max_tokens=800,                     # cost-optimized: shorter output
            system=system_prompt,
            tools=tools,
            messages=messages,
        )

        # Track tokens
        ticker_input  += response.usage.input_tokens
        ticker_output += response.usage.output_tokens

        # Collect assistant turn
        messages.append({"role": "assistant", "content": response.content})

        if response.stop_reason == "end_turn":
            _total_input_tokens  += ticker_input
            _total_output_tokens += ticker_output
            print(f"    📊 Tokens — in: {ticker_input:,}  out: {ticker_output:,}")
            for block in response.content:
                if hasattr(block, "text"):
                    return block.text
            break

        if response.stop_reason == "tool_use":
            tool_results = []
            for block in response.content:
                if block.type == "tool_use":
                    tool_results.append({
                        "type": "tool_result",
                        "tool_use_id": block.id,
                        "content": "Search executed."
                    })
            if tool_results:
                messages.append({"role": "user", "content": tool_results})

    _total_input_tokens  += ticker_input
    _total_output_tokens += ticker_output
    return f"TICKER: {ticker}\nVERDICT: HOLD\nSUMMARY: Research incomplete.\nSCORE: 5"


# ── STEP 2b: COST SUMMARY ─────────────────────────────────────────────────────
def build_cost_summary() -> str:
    """Calculate and format cost summary for the email."""
    input_cost  = (_total_input_tokens  / 1_000_000) * HAIKU_INPUT_COST_PER_M
    output_cost = (_total_output_tokens / 1_000_000) * HAIKU_OUTPUT_COST_PER_M
    total_usd   = input_cost + output_cost
    total_inr   = total_usd * USD_TO_INR

    return f"""
╔══════════════════════════════════════╗
  COST TRACKER (claude-haiku-4-5)
  Input  tokens : {_total_input_tokens:>10,}   ${input_cost:.4f}
  Output tokens : {_total_output_tokens:>10,}   ${output_cost:.4f}
  ─────────────────────────────────────
  Total cost    :              ${total_usd:.4f}
  In INR        :              ₹{total_inr:.2f}
╚══════════════════════════════════════╝
"""


# ── STEP 3: SCORE PARSER ──────────────────────────────────────────────────────
def parse_score(research_text: str) -> int:
    """Extract numeric score from research output."""
    for line in research_text.splitlines():
        if line.startswith("SCORE:"):
            try:
                return int(line.split(":")[1].strip())
            except:
                pass
    return 5


# ── STEP 4: EMAIL REPORT ──────────────────────────────────────────────────────
def send_email(subject: str, body: str):
    msg = MIMEMultipart("alternative")
    msg["Subject"] = subject
    msg["From"]    = EMAIL_FROM
    msg["To"]      = EMAIL_TO
    msg.attach(MIMEText(body, "plain"))

    with smtplib.SMTP_SSL("smtp.gmail.com", 465) as server:
        server.login(EMAIL_FROM, EMAIL_PASSWORD)
        server.sendmail(EMAIL_FROM, EMAIL_TO, msg.as_string())
    print("✅ Email sent.")


# ── MAIN PIPELINE ─────────────────────────────────────────────────────────────
def run_research_agent(tickers: list[str]):
    """
    Main entry point.
    Pass a list of tickers → get ranked research report by email.
    """
    print(f"\n🔍 Research Agent starting for {len(tickers)} tickers...\n")
    results = []
    global _total_input_tokens, _total_output_tokens
    _total_input_tokens = _total_output_tokens = 0  # reset for this run

    for ticker in tickers:
        print(f"  Analyzing {ticker}...")
        fundamentals = get_fundamentals(ticker)
        if "error" in fundamentals:
            print(f"    ⚠️  Skipped {ticker}: {fundamentals['error']}")
            continue
        research = research_ticker(ticker, fundamentals)
        score    = parse_score(research)
        results.append((score, ticker, research))
        print(f"    ✅ Done — Score: {score}/10")

    # Sort by score descending
    results.sort(key=lambda x: x[0], reverse=True)

    # Build report
    date_str = datetime.now().strftime("%Y-%m-%d")
    report_lines = [
        f"RESEARCH AGENT REPORT — {date_str}",
        f"Tickers analyzed: {len(results)}",
        "=" * 60,
        "",
    ]
    for score, ticker, research in results:
        report_lines.append(research)
        report_lines.append("-" * 60)

    # Append cost tracker
    report_lines.append(build_cost_summary())
    report = "\n".join(report_lines)

    # Email it
    if EMAIL_FROM and EMAIL_TO:
        send_email(f"Research Report {date_str} — Top pick: {results[0][1]}", report)
    else:
        print("\n📋 EMAIL NOT CONFIGURED — printing report:\n")
        print(report)

    return results


# ── ENTRY POINT ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    # Replace with your scanner output or hardcode a watchlist
    TICKERS = ["HOOD", "CRDO", "ALAB", "FLYW", "OSCR"]
    run_research_agent(TICKERS)
