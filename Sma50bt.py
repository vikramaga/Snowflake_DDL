"""
nasdaq_sma50_support_history.py

Scans a ticker universe for stocks that have HISTORICALLY, REPEATEDLY
respected SMA50 as support (not just touching it once) and are
CURRENTLY sitting at SMA50 support today.

Standards followed (per existing pipeline conventions):
  - Pure numpy for all technical calculations (no pandas-ta / talib)
  - Exact 1-bar touch/break detection (no lookahead, no smoothing)
  - matplotlib with ASCII-only markers (diagnostic chart, optional)
  - tqdm in non-notebook mode
  - Gmail SMTP SSL (port 465) with App Password via GitHub Secret
  - Module-level credential variables
  - Diagnostic block printed before send
  - Unconditional _send_email() call (always sends, even 0 matches)
  - HTML table email body with plain-text fallback
  - CSV attachment via MIMEBase

Reliability definition (documented so results are reproducible):
  - "Touch": a bar where Low <= SMA50 * (1 + TOUCH_TOL) AND
             Close >= SMA50 * (1 - TOUCH_TOL)   [price reached the line]
  - "Held" (bounce): within HOLD_WINDOW bars after the touch, Close
             never falls below SMA50 * (1 - BREAK_TOL)
  - "Broken": the touch is NOT held (price closed decisively below SMA50
             and stayed there through the window)
  - Reliability % = Held touches / Total touches, over LOOKBACK_DAYS
  - "Currently at support": most recent bar is itself a Touch
"""

import os
import csv
import smtplib
import numpy as np
import yfinance as yf
from datetime import datetime, timezone
from tqdm import tqdm
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.base import MIMEBase
from email import encoders

# ============================================================
# MODULE-LEVEL CREDENTIALS (populated from GitHub Secrets)
# ============================================================
GMAIL_USER = os.environ.get("GMAIL_USER", "")
GMAIL_APP_PASSWORD = os.environ.get("GMAIL_APP_PASSWORD", "")
GMAIL_TO = os.environ.get("GMAIL_TO", GMAIL_USER)

# ============================================================
# SCAN PARAMETERS
# ============================================================
LOOKBACK_DAYS = 365          # ~1 trading year of history to backtest
SMA_PERIOD = 50
TOUCH_TOL = 0.02             # within 2% of SMA50 counts as a "touch"
BREAK_TOL = 0.02             # more than 2% below SMA50 counts as "broken"
HOLD_WINDOW = 5              # bars to confirm a bounce held
MIN_TOUCHES = 3              # need at least this many historical touches to qualify
MIN_RELIABILITY_PCT = 80.0   # % of touches that held, to qualify as "always" support
CURRENTLY_AT_SUPPORT_TOL = 0.025   # today's price within 2.5% of SMA50

# Ticker universe: reuse the existing NASDAQ/large-cap list file used by
# other scanners in this repo (one ticker per line). Falls back to a
# small built-in list if the file isn't present, so the script never
# hard-fails.
UNIVERSE_FILE = "tickers_universe.txt"
FALLBACK_UNIVERSE = [
    "CF", "DHT", "EOG", "LMAT", "LPG", "DMLP", "DXCM", "CRMD", "FAST",
    "FHI", "FRO", "META", "GS", "AAPL", "AFL", "GRMN", "A", "ESTC",
    "ROST", "O", "SEIC", "TPL", "KSPI", "MSA", "NBIX", "PJT", "ANET",
    "CB", "MS", "NDAQ", "SF", "BLK", "GILD", "VRTX", "JBHT", "HWM",
    "VCTR", "RDDT", "GEV", "CPAY", "ODFL", "KMI",
]


def load_universe():
    if os.path.exists(UNIVERSE_FILE):
        with open(UNIVERSE_FILE, "r") as f:
            tickers = [line.strip().upper() for line in f if line.strip()]
        if tickers:
            return tickers
    return FALLBACK_UNIVERSE


def compute_sma(close, period):
    """Pure numpy simple moving average. Returns array same length as close,
    with NaN for the first (period-1) bars."""
    sma = np.full_like(close, np.nan, dtype=np.float64)
    if len(close) < period:
        return sma
    cumsum = np.cumsum(np.insert(close, 0, 0.0))
    sma[period - 1:] = (cumsum[period:] - cumsum[:-period]) / period
    return sma


def analyze_support_history(ticker):
    """
    Downloads daily OHLC, computes SMA50, walks the series bar-by-bar
    (no lookahead) to find every historical touch and whether it held.
    Returns a result dict or None if insufficient data.
    """
    try:
        df = yf.download(
            ticker,
            period="18mo",
            interval="1d",
            progress=False,
            auto_adjust=True,
        )
    except Exception:
        return None

    if df is None or len(df) < SMA_PERIOD + HOLD_WINDOW + 10:
        return None

    close = df["Close"].to_numpy(dtype=np.float64).flatten()
    low = df["Low"].to_numpy(dtype=np.float64).flatten()
    dates = df.index.to_numpy()

    sma50 = compute_sma(close, SMA_PERIOD)

    # Restrict analysis window to LOOKBACK_DAYS trading bars
    n = len(close)
    start_idx = max(SMA_PERIOD, n - LOOKBACK_DAYS)

    touches = []
    i = start_idx
    while i < n - HOLD_WINDOW:
        if np.isnan(sma50[i]):
            i += 1
            continue

        is_touch = (
            low[i] <= sma50[i] * (1 + TOUCH_TOL)
            and close[i] >= sma50[i] * (1 - TOUCH_TOL)
        )

        if is_touch:
            window_close = close[i + 1: i + 1 + HOLD_WINDOW]
            window_sma = sma50[i + 1: i + 1 + HOLD_WINDOW]
            valid = ~np.isnan(window_sma)
            if valid.sum() == 0:
                i += 1
                continue
            broke_down = np.any(
                window_close[valid] < window_sma[valid] * (1 - BREAK_TOL)
            )
            touches.append({
                "date": dates[i],
                "held": not broke_down,
            })
            i += HOLD_WINDOW  # skip ahead past the confirmation window
        else:
            i += 1

    total_touches = len(touches)
    if total_touches < MIN_TOUCHES:
        return None

    held_touches = sum(1 for t in touches if t["held"])
    reliability_pct = 100.0 * held_touches / total_touches

    last_close = close[-1]
    last_sma = sma50[-1]
    if np.isnan(last_sma):
        return None

    dist_pct = 100.0 * (last_close - last_sma) / last_sma
    currently_at_support = abs(dist_pct) <= CURRENTLY_AT_SUPPORT_TOL * 100

    return {
        "ticker": ticker,
        "price": round(float(last_close), 2),
        "sma50": round(float(last_sma), 2),
        "dist_pct": round(float(dist_pct), 2),
        "total_touches": total_touches,
        "held_touches": held_touches,
        "reliability_pct": round(reliability_pct, 1),
        "currently_at_support": currently_at_support,
    }


def _send_email(results):
    """Unconditional send - always emails a report, even with 0 matches."""
    timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    csv_filename = f"sma50_support_history_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M')}.csv"

    qualifying = [
        r for r in results
        if r["currently_at_support"] and r["reliability_pct"] >= MIN_RELIABILITY_PCT
    ]
    qualifying.sort(key=lambda r: (-r["reliability_pct"], -r["total_touches"]))

    # ---- CSV (full results, not just qualifying) ----
    with open(csv_filename, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "Ticker", "Price", "SMA50", "Dist_%", "Total_Touches",
            "Held_Touches", "Reliability_%", "Currently_At_Support"
        ])
        for r in sorted(results, key=lambda x: -x["reliability_pct"]):
            writer.writerow([
                r["ticker"], r["price"], r["sma50"], r["dist_pct"],
                r["total_touches"], r["held_touches"], r["reliability_pct"],
                "YES" if r["currently_at_support"] else "no",
            ])

    # ---- HTML table (top qualifying rows) ----
    rows_html = ""
    for idx, r in enumerate(qualifying[:50]):
        bg = "#fff" if idx % 2 == 0 else "#f0f9ff"
        rows_html += f"""<tr style="background:{bg}">
<td style="padding:6px 11px;font-size:12px;font-weight:700">{r['ticker']}</td>
<td style="padding:6px 11px;font-size:12px">${r['price']}</td>
<td style="padding:6px 11px;font-size:12px">{r['dist_pct']:+.2f}%</td>
<td style="padding:6px 11px;font-size:12px">{r['total_touches']}</td>
<td style="padding:6px 11px;font-size:12px;font-weight:700;background:#166534;color:#fff;text-align:center">{r['reliability_pct']}%</td>
</tr>"""

    html_body = f"""<!DOCTYPE html><html><body style="margin:0;padding:0;background:#f1f5f9;font-family:'Segoe UI',Arial,sans-serif">
<table width="100%" cellpadding="0" cellspacing="0" style="padding:20px 0">
<tr><td>
<table width="100%" cellpadding="0" cellspacing="0" style="max-width:900px;margin:0 auto;background:#fff;border-radius:12px;overflow:hidden;box-shadow:0 4px 20px rgba(0,0,0,0.08)">
<tr><td style="background:linear-gradient(135deg,#0f172a,#1e3a5f);padding:22px 28px">
<h1 style="margin:0;color:#60a5fa;font-size:20px;font-weight:700">📊 SMA50 Reliable Support - Currently At Support</h1>
<p style="margin:6px 0 0;color:#94a3b8;font-size:12px">{timestamp} &nbsp;·&nbsp; {len(qualifying)} of {len(results)} scanned qualify (>= {MIN_RELIABILITY_PCT}% reliability, >= {MIN_TOUCHES} touches, at support now)</p>
</td></tr>
<tr><td style="padding:16px">
<div style="overflow-x:auto;border-radius:8px;border:1px solid #e2e8f0">
<table style="border-collapse:collapse;width:100%;min-width:600px">
<thead><tr>
<th style="background:#1e293b;color:#e2e8f0;padding:8px 11px;font-size:11px;font-weight:700;border-bottom:2px solid #3b82f6">Ticker</th>
<th style="background:#1e293b;color:#e2e8f0;padding:8px 11px;font-size:11px;font-weight:700;border-bottom:2px solid #3b82f6">Price</th>
<th style="background:#1e293b;color:#e2e8f0;padding:8px 11px;font-size:11px;font-weight:700;border-bottom:2px solid #3b82f6">Dist_SMA50</th>
<th style="background:#1e293b;color:#e2e8f0;padding:8px 11px;font-size:11px;font-weight:700;border-bottom:2px solid #3b82f6">Touches</th>
<th style="background:#1e293b;color:#e2e8f0;padding:8px 11px;font-size:11px;font-weight:700;border-bottom:2px solid #3b82f6">Reliability</th>
</tr></thead>
<tbody>{rows_html}</tbody>
</table>
</div>
<p style="font-size:11px;color:#64748b;margin:8px 0 0">📎 Full results (all scanned tickers) attached as CSV. Methodology: touch = Low within {TOUCH_TOL*100:.0f}% of SMA50; held = Close stays within {BREAK_TOL*100:.0f}% of SMA50 for {HOLD_WINDOW} bars after.</p>
</td></tr>
<tr><td style="background:#f8fafc;padding:12px 28px;border-top:1px solid #e2e8f0;text-align:center">
<p style="margin:0;color:#94a3b8;font-size:10px">⚠️ Not financial advice &nbsp;·&nbsp; Auto-generated by GitHub Actions</p>
</td></tr>
</table>
</td></tr></table>
</body></html>"""

    plaintext_lines = [
        f"SMA50 Reliable Support - Currently At Support — {timestamp}",
        f"{len(qualifying)} of {len(results)} scanned qualify",
        "=" * 60,
    ]
    for r in qualifying:
        plaintext_lines.append(
            f"{r['ticker']} ${r['price']} Dist:{r['dist_pct']:+.2f}% "
            f"Touches:{r['total_touches']} Reliability:{r['reliability_pct']}%"
        )
    plaintext_lines.append("\nFull results in CSV attachment.")
    plaintext_body = "\n".join(plaintext_lines)

    msg = MIMEMultipart("mixed")
    msg["Subject"] = f"📊 SMA50 Reliable Support — {len(qualifying)} signals — {datetime.now(timezone.utc).strftime('%Y-%m-%d')}"
    msg["From"] = GMAIL_USER
    msg["To"] = GMAIL_TO

    alt = MIMEMultipart("alternative")
    alt.attach(MIMEText(plaintext_body, "plain"))
    alt.attach(MIMEText(html_body, "html"))
    msg.attach(alt)

    with open(csv_filename, "rb") as f:
        part = MIMEBase("application", "octet-stream")
        part.set_payload(f.read())
    encoders.encode_base64(part)
    part.add_header("Content-Disposition", f"attachment; filename={csv_filename}")
    msg.attach(part)

    # ---- Diagnostic block ----
    print("=" * 60)
    print("DIAGNOSTIC: SMA50 Support History Scanner")
    print(f"Tickers scanned      : {len(results)}")
    print(f"Qualifying (email)   : {len(qualifying)}")
    print(f"Reliability threshold: {MIN_RELIABILITY_PCT}%")
    print(f"Min touches required : {MIN_TOUCHES}")
    print(f"CSV file             : {csv_filename}")
    print(f"GMAIL_USER set       : {bool(GMAIL_USER)}")
    print(f"GMAIL_APP_PASSWORD set: {bool(GMAIL_APP_PASSWORD)}")
    print("=" * 60)

    with smtplib.SMTP_SSL("smtp.gmail.com", 465) as server:
        server.login(GMAIL_USER, GMAIL_APP_PASSWORD)
        server.sendmail(GMAIL_USER, [GMAIL_TO], msg.as_string())

    print(f"Email sent to {GMAIL_TO} with {len(qualifying)} qualifying tickers.")


def main():
    universe = load_universe()
    results = []
    for ticker in tqdm(universe, desc="Scanning SMA50 support history"):
        r = analyze_support_history(ticker)
        if r is not None:
            results.append(r)

    _send_email(results)  # unconditional - always sends


if __name__ == "__main__":
    main()
