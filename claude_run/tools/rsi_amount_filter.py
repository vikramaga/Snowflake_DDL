# Daily RSI / cost digest over the ClaudeRunner screener results.
#
# Reads every sized CSV (with Quantity/Total_Amount) in a results run folder,
# computes Wilder RSI(14) on daily closes (same formula as the scanners), and
# keeps rows with  RSI_MIN <= RSI <= RSI_MAX  and  0 < Total_Amount < MAX_AMOUNT.
# Writes rsi_<min>_<max>_under_<amount>.csv and, with --email, emails it with
# a summary table (one row per ticker, best-scoring screener first).
#
# Env: RESULTS_RUN=<run_id> (default: newest run with daily_* CSVs)
#      RSI_MIN=58 RSI_MAX=64 MAX_AMOUNT=550
#      GMAIL_USER / GMAIL_PASS / EMAIL_TO (or EMAIL_FROM / EMAIL_PASSWORD) for --email
# Run daily by .github/workflows/rsi_digest.yml after the 06:00 IST ClaudeRunner run.
# NSE tickers (.NS/.BO) are priced in rupees: their limit is MAX_AMOUNT converted
# at the live USD/INR rate, and Amount_USD shows the dollar equivalent.
import glob, os, re, sys
from datetime import datetime, timezone
import pandas as pd

RSI_MIN = float(os.environ.get("RSI_MIN", "58"))
RSI_MAX = float(os.environ.get("RSI_MAX", "64"))
MAX_AMOUNT = float(os.environ.get("MAX_AMOUNT", "550"))
ROOT = "claude_run/results"
LOW_PRICE = 10.0


def _fmt(x):
    return f"{x:g}"


def pick_run():
    want = os.environ.get("RESULTS_RUN", "").strip()
    if want and glob.glob(f"{ROOT}/{want}/*/*.csv"):
        return want
    if want:
        print(f"Results for run {want} not found - using the newest run instead")
    runs = [d for d in os.listdir(ROOT) if d.isdigit() and glob.glob(f"{ROOT}/{d}/daily_*/*.csv")]
    return max(runs, key=int) if runs else None


def calc_rsi(close, period=14):
    delta = close.diff()
    gain, loss = delta.clip(lower=0), -delta.clip(upper=0)
    ag = gain.ewm(alpha=1/period, min_periods=period, adjust=False).mean()
    al = loss.ewm(alpha=1/period, min_periods=period, adjust=False).mean()
    return 100 - 100 / (1 + ag / al)


def load_results(run):
    frames, files = [], []
    for c in sorted(glob.glob(f"{ROOT}/{run}/*/*.csv")):
        b = os.path.basename(c)
        if b.startswith("tv_"):
            continue
        df = pd.read_csv(c)
        if not {"Ticker", "Quantity", "Total_Amount"} <= set(df.columns):
            continue
        entry = "Saved_High" if "armed" in b else ("Entry_Price" if "Entry_Price" in df else "Price")
        keep = ["Ticker"] + (["Score"] if "Score" in df else []) + [entry, "Stop_Loss", "Quantity", "Total_Amount"]
        if "RSI" in df:
            keep.append("RSI")
        d = df[keep].rename(columns={entry: "Entry", "RSI": "Scanner_RSI"})
        d.insert(0, "Scanner", re.sub(r"_\d{8}_\d{4}\.csv$", "", b))
        frames.append(d)
        files.append((b, len(df)))
        print(f"  {b}: {len(df)} rows")
    return (pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()), files


def latest_rsi(tickers):
    import yfinance as yf
    raw = yf.download(tickers, period="2y", interval="1d", auto_adjust=True,
                      actions=False, progress=False, group_by="ticker", threads=True)
    out, last = {}, None
    for t in tickers:
        try:
            cl = (raw[t]["Close"] if isinstance(raw.columns, pd.MultiIndex) else raw["Close"]).dropna()
            if len(cl) > 30:
                out[t] = float(calc_rsi(cl).iloc[-1])
                last = max(last, cl.index[-1]) if last is not None else cl.index[-1]
        except Exception:
            pass
    return out, last


def fund_flags(tickers):
    """{ticker: 'ETF' / 'ETN' / ...} for non-common-stock tickers (best effort)."""
    import yfinance as yf
    out = {}
    for t in tickers:
        try:
            qt = (yf.Ticker(t).get_info() or {}).get("quoteType", "")
            if qt and qt.upper() not in ("EQUITY", ""):
                out[t] = qt.upper()
        except Exception:
            pass
    return out


def usd_inr():
    try:
        sys.path.insert(0, os.getcwd())
        from position_sizing import _usd_to
        return _usd_to("INR")[0]
    except Exception:
        return 88.0


def _money(r):
    if r.Currency == "INR":
        return f"₹{r.Total_Amount:,.0f} (≈${r.Amount_USD:,.0f})"
    return f"${r.Total_Amount:,.0f}"


def build_email(hit, by_tkr, run, files, asof):
    date = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    n = len(by_tkr)
    subj = f"📊 RSI {_fmt(RSI_MIN)}-{_fmt(RSI_MAX)} · under ${MAX_AMOUNT:,.0f} — {n} stocks — {date}"
    crit = (f"RSI(14) {_fmt(RSI_MIN)}–{_fmt(RSI_MAX)} (daily, as of {asof}), Total_Amount under "
            f"${MAX_AMOUNT:,.0f} (India: ₹ equivalent), Quantity &gt; 0 (max loss $40 per trade)")
    src = ", ".join(f"{b.rsplit('_2026', 1)[0]} ({k})" for b, k in files)
    th = "padding:6px 10px;background:#1f2937;color:#fff;text-align:left;font-size:12px"
    td = "padding:5px 10px;border-bottom:1px solid #e5e7eb;font-size:12px"
    rows = []
    for _, r in by_tkr.iterrows():
        flag = f' <span style="color:#b45309">⚠ {r.Flags}</span>' if r.Flags else ""
        score = "" if pd.isna(r.get("Score")) else f"{r.Score:.0f}"
        rows.append(
            f"<tr><td style='{td}'><b>{r.Ticker}</b>{flag}</td><td style='{td}'>{score}</td>"
            f"<td style='{td}'>{r.RSI:.1f}</td><td style='{td}'>{r.Entry:,.2f}</td>"
            f"<td style='{td}'>{r.Stop_Loss:,.2f}</td><td style='{td}'>{int(r.Quantity)}</td>"
            f"<td style='{td}'>{_money(r)}</td><td style='{td}'>{r.Screeners}</td></tr>")
    table = ("<table style='border-collapse:collapse;font-family:Arial'>"
             f"<tr><th style='{th}'>Ticker</th><th style='{th}'>Score</th><th style='{th}'>RSI</th>"
             f"<th style='{th}'>Entry</th><th style='{th}'>Stop</th><th style='{th}'>Qty</th>"
             f"<th style='{th}'>Amount</th><th style='{th}'>Screeners</th></tr>" + "".join(rows) + "</table>")
    tv = ",".join(by_tkr.Ticker)
    html = (f"<div style='font-family:Arial;font-size:13px'>"
            f"<h3 style='margin:0 0 6px'>RSI {_fmt(RSI_MIN)}–{_fmt(RSI_MAX)} · under ${MAX_AMOUNT:,.0f} — {n} stocks</h3>"
            f"<p style='margin:0 0 4px'>{crit}</p>"
            f"<p style='margin:0 0 10px;color:#6b7280'>From ClaudeRunner run {run}: {src}. "
            f"One row per stock; entry/stop/qty from its highest-scoring screener. "
            f"⚠ marks ETFs/ETNs and stocks under ${LOW_PRICE:.0f}.</p>"
            + (table if n else "<p><b>No stocks matched today.</b></p>") +
            f"<p style='margin-top:12px'>TradingView list:<br><code>{tv}</code></p>"
            f"<p style='color:#6b7280'>📎 All {len(hit)} matching rows (every screener) are in the attached CSV.</p></div>")
    plain = [f"RSI {_fmt(RSI_MIN)}-{_fmt(RSI_MAX)} under ${MAX_AMOUNT:,.0f} - {n} stocks ({date})", ""]
    for _, r in by_tkr.iterrows():
        plain.append(f"{r.Ticker:6s} RSI {r.RSI:4.1f}  {r.Entry:,.2f} -> {r.Stop_Loss:,.2f}  "
                     f"qty {int(r.Quantity)}  {_money(r)}  [{r.Screeners}]"
                     + (f"  ! {r.Flags}" if r.Flags else ""))
    plain += ["", "TradingView: " + tv]
    return subj, html, "\n".join(plain)


def send_email(subj, html, plain, csv_path):
    import smtplib
    from email.mime.multipart import MIMEMultipart
    from email.mime.text import MIMEText
    from email.mime.base import MIMEBase
    from email import encoders
    gu = os.environ.get("GMAIL_USER") or os.environ.get("EMAIL_FROM", "")
    gp = os.environ.get("GMAIL_PASS") or os.environ.get("EMAIL_PASSWORD", "")
    et = os.environ.get("EMAIL_TO", "")
    if not (gu and gp and et):
        print("[Email] ❌  GMAIL_USER / GMAIL_PASS / EMAIL_TO not set — email skipped")
        return False
    eto = [e.strip() for e in et.split(",") if e.strip()]
    msg = MIMEMultipart("mixed")
    msg["Subject"], msg["From"], msg["To"] = subj, gu, ", ".join(eto)
    alt = MIMEMultipart("alternative")
    alt.attach(MIMEText(plain, "plain"))
    alt.attach(MIMEText(html, "html"))
    msg.attach(alt)
    with open(csv_path, "rb") as f:
        part = MIMEBase("application", "octet-stream")
        part.set_payload(f.read())
    encoders.encode_base64(part)
    part.add_header("Content-Disposition", f"attachment; filename={os.path.basename(csv_path)}")
    msg.attach(part)
    try:
        with smtplib.SMTP_SSL("smtp.gmail.com", 465) as srv:
            srv.login(gu, gp.replace(" ", ""))
            srv.sendmail(gu, eto, msg.as_string())
        print(f"[Email] ✅  Sent to {', '.join(eto)} — {subj}")
        return True
    except smtplib.SMTPException as e:
        print(f"[Email] ❌  SMTP error: {e}")
        return False


def main():
    email = "--email" in sys.argv
    run = pick_run()
    if not run:
        print("No screener results found"); return
    print(f"Results run: {run}   filter: RSI {_fmt(RSI_MIN)}-{_fmt(RSI_MAX)}, Total_Amount < ${MAX_AMOUNT:,.0f}")
    allr, files = load_results(run)
    cols = ["Scanner", "Ticker", "Score", "Entry", "Stop_Loss", "Quantity", "Total_Amount",
            "Currency", "Amount_USD", "RSI", "Flags"]
    if allr.empty:
        hit, asof = pd.DataFrame(columns=cols), "n/a"
    else:
        tickers = sorted(allr.Ticker.dropna().unique())
        print(f"Downloading daily closes for {len(tickers)} tickers ...")
        rsi, last = latest_rsi(tickers)
        asof = last.strftime("%Y-%m-%d") if last is not None else "n/a"
        allr["RSI"] = allr.Ticker.map(rsi).round(1)
        if "Scanner_RSI" in allr:
            chk = allr.dropna(subset=["Scanner_RSI", "RSI"])
            if len(chk):
                print(f"RSI check vs scanner's own RSI: max diff {(chk.RSI - chk.Scanner_RSI).abs().max():.2f} over {len(chk)} rows")
            allr = allr.drop(columns="Scanner_RSI")
        if "Score" not in allr:
            allr["Score"] = float("nan")
        inr = allr.Ticker.astype(str).str.upper().str.endswith((".NS", ".BO"))
        rate = usd_inr() if inr.any() else 1.0
        if inr.any():
            print(f"NSE rows: {int(inr.sum())} — limit ₹{MAX_AMOUNT * rate:,.0f} at USD/INR {rate:.2f}")
        allr["Currency"] = ["INR" if x else "USD" for x in inr]
        allr["Amount_USD"] = (allr.Total_Amount / [rate if x else 1.0 for x in inr]).round(2)
        hit = allr[(allr.RSI >= RSI_MIN) & (allr.RSI <= RSI_MAX)
                   & (allr.Quantity > 0) & (allr.Amount_USD < MAX_AMOUNT)].copy()
        kinds = fund_flags(sorted(hit.Ticker.unique())) if len(hit) else {}
        hit["Flags"] = [", ".join(f for f in (kinds.get(t, ""), f"under ${LOW_PRICE:.0f}" if (e < LOW_PRICE and c == "USD") else "") if f)
                        for t, e, c in zip(hit.Ticker, hit.Entry, hit.Currency)]
        hit = hit[cols].sort_values(["Ticker", "Scanner"])

    out_csv = f"rsi_{int(RSI_MIN)}_{int(RSI_MAX)}_under_{int(MAX_AMOUNT)}.csv"
    hit.to_csv(out_csv, index=False)

    if len(hit):
        best = hit.sort_values("Score", ascending=False, na_position="last").drop_duplicates("Ticker")
        scr = hit.groupby("Ticker").Scanner.apply(
            lambda s: ", ".join(sorted({x.replace("daily_", "").replace("_v1", "") for x in s})))
        by_tkr = best.assign(Screeners=best.Ticker.map(scr), N=best.Ticker.map(hit.groupby("Ticker").Scanner.nunique()))
        by_tkr["_fund"] = by_tkr.Flags.str.contains("ETF|ETN|FUND", regex=True)
        by_tkr = by_tkr.sort_values(["_fund", "N", "Score"], ascending=[True, False, False]).drop(columns="_fund")
    else:
        by_tkr = pd.DataFrame(columns=cols + ["Screeners", "N"])

    pd.set_option("display.width", 250)
    print(f"\n{len(hit)} rows, {len(by_tkr)} unique tickers  (RSI as of {asof})\n")
    print(by_tkr.drop(columns=["Scanner"]).to_string(index=False) if len(by_tkr) else "No matches")

    if email:
        send_email(*build_email(hit, by_tkr, run, files, asof), out_csv)


if __name__ == "__main__":
    main()
