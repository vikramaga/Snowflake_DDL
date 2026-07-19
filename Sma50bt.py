# ============================================================
# NASDAQ — SMA50 Reliable Support History Scanner
# ============================================================
#
# Finds stocks that have HISTORICALLY, REPEATEDLY respected SMA50
# as support (not just touched it once) AND are sitting at SMA50
# support RIGHT NOW.
#
# Uses the same ticker universe, batched download, and email
# pipeline conventions as nasdaq_fundamental_technical_support.py.
#
# RELIABILITY METHODOLOGY (documented for reproducibility):
#   "Touch"  = a bar where Low <= SMA50*(1+TOUCH_TOL) AND
#              Close >= SMA50*(1-TOUCH_TOL)
#   "Held"   = over the next HOLD_WINDOW bars, Close never falls
#              below SMA50*(1-BREAK_TOL)  -> counted as a bounce
#   "Broken" = touch is not held -> price closed decisively below
#   Reliability % = Held touches / Total touches, over LOOKBACK_DAYS
#   "Currently at support" = most recent bar is itself a Touch
#
# ============================================================

import subprocess, sys, os

def pip_install(*packages):
    subprocess.check_call(
        [sys.executable, "-m", "pip", "install", "--upgrade", "-q", *packages],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )

pip_install("yfinance", "pandas", "numpy", "requests", "tqdm")
print("Dependencies installed")

def _detect_notebook():
    try:
        if "google.colab" in sys.modules: return True
        if os.environ.get("COLAB_BACKEND_VERSION"): return True
        if os.environ.get("JPY_PARENT_PID"):
            import importlib
            if importlib.util.find_spec("ipywidgets") is not None: return True
    except Exception: pass
    return False

_IN_NOTEBOOK = _detect_notebook()
from tqdm import tqdm

import yfinance as yf
import pandas as pd
import numpy as np
import requests, time, warnings, io, csv as _csv
from datetime import datetime, timedelta
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.base import MIMEBase
from email import encoders

warnings.filterwarnings("ignore")
pd.set_option("display.max_rows", 200)
env = "Colab/Jupyter" if _IN_NOTEBOOK else "Script/CI"
print(f"yfinance {yf.__version__}  |  numpy {np.__version__}  |  [{env}]")

# -- Email secret diagnostic (same secret names as other scanners) --
_GMAIL_USER = os.environ.get("GMAIL_USER", "")
_GMAIL_PASS = os.environ.get("GMAIL_PASS", "")
_EMAIL_TO   = os.environ.get("EMAIL_TO",   "")

print()
print("-"*65)
print("  EMAIL CONFIGURATION")
print("-"*65)
if _GMAIL_USER and _GMAIL_PASS and _EMAIL_TO:
    print(f"  OK GMAIL_USER  : {_GMAIL_USER[:4]}***{_GMAIL_USER[-4:]}")
    print(f"  OK GMAIL_PASS  : {'*'*16}  ({len(_GMAIL_PASS.replace(' ',''))} chars)")
    print(f"  OK EMAIL_TO    : {_EMAIL_TO}")
    print(f"  Email will be sent after scan")
else:
    missing = [k for k,v in [("GMAIL_USER",_GMAIL_USER),
                               ("GMAIL_PASS",_GMAIL_PASS),
                               ("EMAIL_TO",_EMAIL_TO)] if not v]
    print(f"  MISSING secrets: {', '.join(missing)}")
    print(f"  Go to: GitHub repo -> Settings -> Secrets -> Actions")
    print(f"       Add: GMAIL_USER, GMAIL_PASS (App Password), EMAIL_TO")
    print(f"  Email will be SKIPPED this run")
print("-"*65)
print()

# -- CONFIG --------------------------------------------------------
CFG = {
    "history_days"       : 400,   # ~1 trading year + buffer for SMA50 warmup

    # -- Support-touch definition --
    "touch_tol_pct"       : 2.0,   # Low within 2% of SMA50 = touch
    "break_tol_pct"       : 2.0,   # Close >2% below SMA50 = broken
    "hold_window_bars"    : 5,     # bars to confirm bounce held
    "currently_at_pct"    : 2.5,   # today's price within 2.5% of SMA50

    # -- Qualification gates --
    "min_touches"         : 3,     # need >= this many historical touches
    "min_reliability_pct" : 80.0,  # % of touches that held

    # -- Filters --
    "min_avg_volume"      : 80_000,
    "min_price"           : 2.0,

    "batch_size"          : 50,
    "batch_sleep"         : 1.5,
}

# -- Pure numpy SMA (exact, no lookahead) ---------------------------
def compute_sma_np(close_arr, period):
    sma = np.full_like(close_arr, np.nan, dtype=np.float64)
    if len(close_arr) < period:
        return sma
    cumsum = np.cumsum(np.insert(close_arr, 0, 0.0))
    sma[period - 1:] = (cumsum[period:] - cumsum[:-period]) / period
    return sma

# -- Support-history analysis ----------------------------------------
def analyze_support_history(sym, df):
    n = len(df)
    price = float(df["Close"].iloc[-1])
    avg_vol = float(df["Volume"].tail(20).mean())

    if price   < CFG["min_price"]:      return None
    if avg_vol < CFG["min_avg_volume"]: return None
    if n < 50 + CFG["hold_window_bars"] + 10: return None

    close = df["Close"].to_numpy(dtype=np.float64).flatten()
    low   = df["Low"].to_numpy(dtype=np.float64).flatten()
    dates = df.index

    sma50 = compute_sma_np(close, 50)
    if np.isnan(sma50[-1]): return None

    touch_tol = CFG["touch_tol_pct"] / 100
    break_tol = CFG["break_tol_pct"] / 100
    hold_win  = CFG["hold_window_bars"]

    touches = []
    i = 50
    while i < n - hold_win:
        if np.isnan(sma50[i]):
            i += 1; continue
        is_touch = (
            low[i] <= sma50[i] * (1 + touch_tol)
            and close[i] >= sma50[i] * (1 - touch_tol)
        )
        if is_touch:
            wc = close[i+1:i+1+hold_win]
            ws = sma50[i+1:i+1+hold_win]
            valid = ~np.isnan(ws)
            broke = bool(np.any(wc[valid] < ws[valid] * (1 - break_tol))) if valid.sum() > 0 else False
            touches.append({"idx": i, "date": dates[i], "held": not broke})
            i += hold_win
        else:
            i += 1

    total_touches = len(touches)
    if total_touches < CFG["min_touches"]:
        return None

    held_touches = sum(1 for t in touches if t["held"])
    reliability_pct = round(100.0 * held_touches / total_touches, 1)
    if reliability_pct < CFG["min_reliability_pct"]:
        return None

    last_sma = float(sma50[-1])
    dist_pct = round((price - last_sma) / last_sma * 100, 2)
    currently_at_support = abs(dist_pct) <= CFG["currently_at_pct"]
    if not currently_at_support:
        return None

    last_touch_date = touches[-1]["date"]
    try:
        last_touch_str = pd.Timestamp(last_touch_date).strftime("%Y-%m-%d")
    except Exception:
        last_touch_str = str(last_touch_date)

    return {
        "Ticker"           : sym,
        "Price"            : round(price, 2),
        "SMA50"            : round(last_sma, 2),
        "Dist_%"           : dist_pct,
        "Total_Touches"    : total_touches,
        "Held_Touches"     : held_touches,
        "Reliability_%"    : reliability_pct,
        "Last_Touch_Date"  : last_touch_str,
    }

# -- Download (identical batching pattern to sibling scanner) --------
def _clean(df, min_bars=60):
    if df is None or df.empty: return None
    need = [c for c in ["Open","High","Low","Close","Volume"] if c in df.columns]
    if not all(c in need for c in ["High","Low","Close","Volume"]): return None
    df = df[need].copy()
    df.index = pd.to_datetime(df.index)
    if hasattr(df.index,"tz") and df.index.tz:
        df.index = df.index.tz_localize(None)
    df.dropna(subset=["Close","Volume"], inplace=True)
    return df if (len(df)>=min_bars and float(df["Close"].iloc[-1])>0) else None

def download(symbols, days):
    end = datetime.today(); start = end - timedelta(days=days)
    out = {}
    try:
        raw = yf.download(symbols, start=start.strftime("%Y-%m-%d"),
                          end=end.strftime("%Y-%m-%d"), group_by="ticker",
                          auto_adjust=True, actions=False,
                          threads=True, progress=False)
        if raw is not None and not raw.empty:
            pf = {"Open","High","Low","Close","Volume","Adj Close"}
            if isinstance(raw.columns, pd.MultiIndex):
                l0 = set(raw.columns.get_level_values(0))
                for sym in symbols:
                    try:
                        df = raw.xs(sym,axis=1,level=1) if l0&pf else raw[sym]
                        df = _clean(df)
                        if df is not None: out[sym] = df
                    except Exception: pass
            elif len(symbols) == 1:
                df = _clean(raw)
                if df is not None: out[symbols[0]] = df
    except Exception: pass
    for sym in [s for s in symbols if s not in out]:
        for _ in range(2):
            try:
                df = yf.Ticker(sym).history(
                    start=start.strftime("%Y-%m-%d"),
                    end=end.strftime("%Y-%m-%d"),
                    auto_adjust=True, actions=False)
                df = _clean(df)
                if df is not None: out[sym] = df; break
            except Exception: time.sleep(0.2)
        time.sleep(0.04)
    return out

# -- Ticker universe (identical to nasdaq_fundamental_technical_support.py) --
print("-"*65)
print("  STEP 1  FETCH TICKERS")
print("-"*65)

def get_tickers():
    pool = set()
    hdrs = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}
    for url, label in [
        ("https://www.nasdaqtrader.com/dynamic/SymDir/nasdaqlisted.txt","nasdaqlisted"),
        ("https://www.nasdaqtrader.com/dynamic/SymDir/otherlisted.txt", "otherlisted"),
    ]:
        try:
            r  = requests.get(url,headers=hdrs,timeout=20); r.raise_for_status()
            df = pd.read_csv(io.StringIO(r.text),sep="|")
            df.columns = [c.strip() for c in df.columns]
            sc = next((c for c in ["Symbol","ACT Symbol","Nasdaq Symbol"] if c in df.columns),None)
            ec = next((c for c in ["ETF","Is ETF"] if c in df.columns),None)
            if not sc: continue
            b = len(pool)
            for _,row in df.iterrows():
                s = str(row[sc]).strip()
                if not s or s=="nan": continue
                if any(x in s for x in ["^","/","."," ","-"]): continue
                if s.endswith(tuple("WRUPQ")): continue
                if not(s.isalpha() and 1<=len(s)<=5): continue
                if ec and str(row.get(ec,"")).strip().upper()=="Y": continue
                pool.add(s.upper())
            print(f"  OK {label:<18}: +{len(pool)-b:>4} -> {len(pool)}")
        except Exception as e: print(f"  WARN {label}: {e}")
    try:
        r = requests.get("https://api.nasdaq.com/api/screener/stocks"
                       "?tableonly=true&limit=10000&exchange=nasdaq&download=true",
                       headers={**hdrs,"Referer":"https://www.nasdaq.com/"},timeout=25)
        r.raise_for_status()
        rows = r.json()["data"]["rows"]
        t = {row["symbol"].strip() for row in rows
             if row.get("symbol","").strip().isalpha()
             and 1<=len(row["symbol"].strip())<=5}
        b = len(pool); pool |= t
        print(f"  OK {'NASDAQ API':<18}: +{len(pool)-b:>4} -> {len(pool)}")
    except Exception as e: print(f"  WARN NASDAQ API: {e}")
    static = {
        "AAPL","MSFT","NVDA","AMZN","META","GOOGL","GOOG","TSLA","AVGO","COST",
        "NFLX","AMD","INTC","CSCO","ADBE","QCOM","TXN","AMAT","MU","KLAC",
        "LRCX","MRVL","MELI","PANW","CRWD","SNPS","CDNS","TEAM","WDAY","PLTR",
        "ALAB","SMCI","HOOD","COIN","SOFI","UPST","BILL","ZS","OKTA","DDOG",
        "SNOW","MDB","REGN","VRTX","ALNY","PODD","MPWR","ONTO","ENTG","SWKS",
        "ADSK","ANSS","BIIB","CPRT","ENPH","FAST","FTNT","IDXX","ISRG","LULU",
        "ODFL","ORLY","PAYX","PCAR","SBUX","TMUS","VRSK","ZM","ZBRA","RBRK",
        "SMAR","PSTG","NET","GTLB","CFLT","MNDY","HUBS","VEEV","PCTY","PAYC",
        "BRZE","IONQ","ABNB","DASH","RBLX","KVYO","SOUN","CRWV","MSTR","MARA",
        "QUBT","RGTI","ASTS","RKLB","LUNR","FSLR","PYPL","ROKU","ROST","POOL",
        "ALGN","AMGN","CTAS","DOCU","EA","FISV","GILD","INTU","MCHP","MNST",
        "NXPI","PDD","SIRI","ULTA","XEL","ESTC","QRVO","ACLS","EXAS","IRTC",
    }
    b = len(pool); pool |= static
    print(f"  OK {'Static fallback':<18}: +{len(pool)-b:>4} -> {len(pool)}")
    clean = sorted({s.upper() for s in pool if isinstance(s,str)
                    and s.isalpha() and 1<=len(s)<=5})
    print(f"\n  Total: {len(clean)} tickers")
    return clean

TICKERS = get_tickers()
print()

# -- Main scan (single pass -- pure technical, no .info calls needed) --
print("-"*65)
print(f"  STEP 2  SCANNING {len(TICKERS)} TICKERS FOR SMA50 SUPPORT RELIABILITY")
print("-"*65 + "\n")

results  = []
no_data  = 0
batches  = [TICKERS[i:i+CFG["batch_size"]] for i in range(0, len(TICKERS), CFG["batch_size"])]

with tqdm(total=len(TICKERS), desc="Scanning", unit="stk",
          bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]") as pbar:
    for batch in batches:
        data_map = download(batch, CFG["history_days"])
        no_data += len(batch) - len(data_map)
        for sym in batch:
            pbar.update(1)
            if sym not in data_map: continue
            try:
                r = analyze_support_history(sym, data_map[sym])
                if r is not None:
                    results.append(r)
            except Exception: pass
        time.sleep(CFG["batch_sleep"])

got = len(TICKERS) - no_data
pct = got / max(len(TICKERS), 1) * 100
print(f"\n  Data retrieved: {got}/{len(TICKERS)} ({pct:.0f}%)")
print(f"  Qualifying (>= {CFG['min_reliability_pct']}% reliability, "
      f">= {CFG['min_touches']} touches, at support now): {len(results)}\n")

results.sort(key=lambda r: (-r["Reliability_%"], -r["Total_Touches"]))

# -- Save CSV --------------------------------------------------------
ts = datetime.today().strftime("%Y%m%d_%H%M")
out_dir = "." if not _IN_NOTEBOOK else "/content"
fpath = os.path.join(out_dir, f"sma50_support_history_{ts}.csv")

cols = ["Ticker","Price","SMA50","Dist_%","Total_Touches",
        "Held_Touches","Reliability_%","Last_Touch_Date"]
with open(fpath, "w", newline="") as f:
    w = _csv.writer(f)
    w.writerow(cols)
    for r in results:
        w.writerow([r[c] for c in cols])
print(f"  CSV saved: {fpath} ({len(results)} rows)")

# -- Live print top rows ----------------------------------------------
print(f"\n{'Ticker':<8}{'Price':<10}{'Dist_%':<9}{'Touches':<9}{'Reliability':<12}")
for r in results[:20]:
    print(f"{r['Ticker']:<8}${r['Price']:<9}{r['Dist_%']:+.2f}%{'':<3}"
          f"{r['Total_Touches']:<9}{r['Reliability_%']}%")

# -- Email (same pipeline/secret names as sibling scanner) -----------
def _send_email(rl, csv_path):
    gu, gp, et = _GMAIL_USER, _GMAIL_PASS, _EMAIL_TO
    if not gu:
        print("[Email] GMAIL_USER secret is empty"); return
    if not gp:
        print("[Email] GMAIL_PASS secret is empty"); return
    if not et:
        print("[Email] EMAIL_TO secret is empty"); return

    eto = [e.strip() for e in et.split(",") if e.strip()]
    cnt = len(rl)

    try:
        rows_e = ""
        for idx, r in enumerate(rl[:50]):
            bg = "#fff" if idx % 2 == 0 else "#f0f9ff"
            rows_e += f"""<tr style="background:{bg}">
<td style="padding:6px 11px;font-size:12px;font-weight:700">{r['Ticker']}</td>
<td style="padding:6px 11px;font-size:12px">${r['Price']}</td>
<td style="padding:6px 11px;font-size:12px">{r['Dist_%']:+.2f}%</td>
<td style="padding:6px 11px;font-size:12px">{r['Total_Touches']}</td>
<td style="padding:6px 11px;font-size:12px;font-weight:700;background:#166534;color:#fff;text-align:center">{r['Reliability_%']}%</td>
<td style="padding:6px 11px;font-size:12px">{r['Last_Touch_Date']}</td>
</tr>"""

        no_results_msg = ""
        if cnt == 0:
            no_results_msg = (
                '<tr><td colspan="6" style="padding:20px;text-align:center;'
                'color:#64748b;font-size:13px">No matches found today - '
                'no tickers currently at a historically reliable SMA50 support</td></tr>'
            )

        html_e = f"""<!DOCTYPE html><html><body style="margin:0;padding:0;
background:#f1f5f9;font-family:'Segoe UI',Arial,sans-serif">
<table width="100%" cellpadding="0" cellspacing="0" style="padding:20px 0">
<tr><td>
<table width="100%" cellpadding="0" cellspacing="0"
   style="max-width:900px;margin:0 auto;background:#fff;
          border-radius:12px;overflow:hidden;
          box-shadow:0 4px 20px rgba(0,0,0,0.08)">
  <tr><td style="background:linear-gradient(135deg,#0f172a,#1e3a5f);padding:22px 28px">
<h1 style="margin:0;color:#60a5fa;font-size:20px;font-weight:700">
  SMA50 Reliable Support - Currently At Support
</h1>
<p style="margin:6px 0 0;color:#94a3b8;font-size:12px">
  {datetime.today().strftime('%Y-%m-%d %H:%M UTC')} &nbsp;.&nbsp;
  {cnt} match{'es' if cnt!=1 else ''} found
</p>
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
<th style="background:#1e293b;color:#e2e8f0;padding:8px 11px;font-size:11px;font-weight:700;border-bottom:2px solid #3b82f6">Last Touch</th>
</tr></thead>
    <tbody>{rows_e or no_results_msg}</tbody>
  </table>
</div>
<p style="font-size:11px;color:#64748b;margin:8px 0 0">
  Full results attached as CSV. Methodology: touch = Low within {CFG['touch_tol_pct']:.0f}% of SMA50;
  held = Close stays within {CFG['break_tol_pct']:.0f}% of SMA50 for {CFG['hold_window_bars']} bars after.
</p>
  </td></tr>
  <tr><td style="background:#f8fafc;padding:12px 28px;
             border-top:1px solid #e2e8f0;text-align:center">
<p style="margin:0;color:#94a3b8;font-size:10px">
  Not financial advice &nbsp;.&nbsp; Auto-generated by GitHub Actions
</p>
  </td></tr>
</table>
</td></tr></table>
</body></html>"""

        plain_lines = [
            f"SMA50 Reliable Support - Currently At Support - {datetime.today().strftime('%Y-%m-%d')}",
            f"{cnt} matches",
            "="*60,
        ]
        if rl:
            for r in rl[:50]:
                plain_lines.append(
                    f"{r['Ticker']:<7} ${r['Price']:.2f}  Dist:{r['Dist_%']:+.2f}%  "
                    f"Touches:{r['Total_Touches']}  Reliability:{r['Reliability_%']}%  "
                    f"LastTouch:{r['Last_Touch_Date']}"
                )
        else:
            plain_lines.append("No matches today")
        plain_lines.append("\nFull results in CSV attachment.")
        plain_e = "\n".join(plain_lines)

        subj = (f"SMA50 Reliable Support - {cnt} signal{'s' if cnt!=1 else ''}"
                f" - {datetime.today().strftime('%Y-%m-%d')}")

        msg = MIMEMultipart("mixed")
        msg["Subject"] = subj
        msg["From"]    = gu
        msg["To"]      = ", ".join(eto)

        alt = MIMEMultipart("alternative")
        alt.attach(MIMEText(plain_e, "plain"))
        alt.attach(MIMEText(html_e,  "html"))
        msg.attach(alt)

    except Exception as e:
        print(f"[Email] Failed to build email body: {type(e).__name__}: {e}")
        return

    if csv_path and os.path.exists(csv_path):
        try:
            with open(csv_path, "rb") as f:
                part = MIMEBase("application", "octet-stream")
                part.set_payload(f.read())
            encoders.encode_base64(part)
            part.add_header("Content-Disposition",
                f"attachment; filename={os.path.basename(csv_path)}")
            msg.attach(part)
            sz = os.path.getsize(csv_path)
            print(f"[Email] Attached: {os.path.basename(csv_path)} ({sz:,} bytes)")
        except Exception as e:
            print(f"[Email] CSV attach failed: {e}")

    try:
        print(f"[Email] Connecting to smtp.gmail.com:465 ...")
        with smtplib.SMTP_SSL("smtp.gmail.com", 465) as srv:
            srv.login(gu, gp.replace(" ", ""))
            srv.sendmail(gu, eto, msg.as_string())
        print(f"[Email] Sent successfully to: {', '.join(eto)}")
        print(f"[Email]    Subject: {subj}")
    except smtplib.SMTPAuthenticationError:
        print("[Email] AUTHENTICATION FAILED")
        print("         GMAIL_PASS must be a Gmail App Password, NOT your login password")
        print("         Generate one: myaccount.google.com/apppasswords")
    except smtplib.SMTPException as e:
        print(f"[Email] SMTP error: {e}")
    except Exception as e:
        print(f"[Email] Unexpected error: {type(e).__name__}: {e}")

try:
    _send_email(results, fpath)  # unconditional -- always sends, even 0 matches
except Exception as e:
    print(f"[Email] Unexpected top-level error: {type(e).__name__}: {e}")
    print("[Email]    Continuing -- CSV is still saved.")

if _IN_NOTEBOOK:
    try:
        from google.colab import files
        files.download(fpath)
    except Exception: pass
else:
    print("  (CI: file in workspace, email sent)")

print(f"""
------------------------------------------------------
  METHODOLOGY
  Touch     = Low within {CFG['touch_tol_pct']:.0f}% of SMA50
  Held      = Close stays within {CFG['break_tol_pct']:.0f}% of SMA50
              for {CFG['hold_window_bars']} bars after the touch
  Reliability % = Held touches / Total touches (lookback ~{CFG['history_days']}d)
  Qualifies = Reliability >= {CFG['min_reliability_pct']}%,
              >= {CFG['min_touches']} touches, AND at support NOW

  TUNE IF 0 RESULTS
  min_reliability_pct   80 -> 65
  min_touches            3 -> 2
  currently_at_pct     2.5 -> 4
  touch_tol_pct          2 -> 3
------------------------------------------------------
""")
