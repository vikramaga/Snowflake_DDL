# ============================================================
# NASDAQ — Daily RSI Pop + Weekly/Monthly RSI Confirm Scanner (v1)
# ============================================================
#
# SIGNAL (all required, scanned over the last signal_lookback_days
# trading days, not just today):
#
#   1. DAILY RSI TRANSITION: yesterday's RSI(14) was BELOW 55, AND
#      today's RSI(14) is now in the 59–64 band (inclusive) — a
#      fresh pop up into that specific zone, not a level that's
#      been sitting there for a while.
#
#   2. WEEKLY: RSI(14) is currently ABOVE 55 (level only, no
#      freshness requirement).
#
#   3. MONTHLY: RSI(14) is currently ABOVE 55 (level only, no
#      freshness requirement).
#
# DATA — only 1 download per ticker: daily bars (~1500 days, for
# stable monthly RSI). Weekly and Monthly RSI are both computed from
# the SAME daily data via resampling (Weekly = 'W', Monthly = 'ME')
# — no separate network calls needed.
#
# OUTPUT: Entry_Price and Stop_Loss are reasonable defaults (Entry =
# the signal day's close; Stop = a recent swing-low proxy), not
# explicitly requested.
#
# SINGLE PASS — purely technical, no fundamentals fetch.
#
# ============================================================

import subprocess, sys, os

def pip_install(*packages):
    subprocess.check_call(
        [sys.executable, "-m", "pip", "install", "--upgrade", "-q", *packages],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )

pip_install("yfinance", "pandas", "numpy", "requests", "tqdm", "matplotlib")
print("✅  Dependencies installed")

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

def display_html(h):
    if _IN_NOTEBOOK and "IPython" in sys.modules:
        try:
            sys.modules["IPython"].display.display(
                sys.modules["IPython"].display.HTML(h))
            return True
        except Exception: pass
    return False

def display(obj):
    if _IN_NOTEBOOK and "IPython" in sys.modules:
        try: sys.modules["IPython"].display.display(obj); return
        except Exception: pass
    try: print(obj.to_string())
    except Exception: print(obj)

import yfinance as yf
import pandas as pd
import numpy as np
import requests, time, warnings, io
from datetime import datetime, timedelta
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

warnings.filterwarnings("ignore")
pd.set_option("display.max_rows", 200)
env = "Colab/Jupyter" if _IN_NOTEBOOK else "Script/CI"
print(f"✅  yfinance {yf.__version__}  |  numpy {np.__version__}  |  [{env}]")

# ── Email secret diagnostic ────────────────────────────────────
_GMAIL_USER = os.environ.get("GMAIL_USER", "")
_GMAIL_PASS = os.environ.get("GMAIL_PASS", "")
_EMAIL_TO   = os.environ.get("EMAIL_TO",   "")

print()
print("━"*65)
print("  EMAIL CONFIGURATION")
print("━"*65)
if _GMAIL_USER and _GMAIL_PASS and _EMAIL_TO:
    print(f"  ✅ GMAIL_USER  : {_GMAIL_USER[:4]}***{_GMAIL_USER[-4:]}")
    print(f"  ✅ GMAIL_PASS  : {'*'*16}  ({len(_GMAIL_PASS.replace(' ',''))} chars)")
    print(f"  ✅ EMAIL_TO    : {_EMAIL_TO}")
    print(f"  ✅ Email will be sent after scan")
else:
    missing = [k for k,v in [("GMAIL_USER",_GMAIL_USER),
                               ("GMAIL_PASS",_GMAIL_PASS),
                               ("EMAIL_TO",_EMAIL_TO)] if not v]
    print(f"  ⚠️  Missing secrets: {', '.join(missing)}")
    print(f"  ℹ️  Go to: GitHub repo → Settings → Secrets → Actions")
    print(f"       Add: GMAIL_USER, GMAIL_PASS (App Password), EMAIL_TO")
    print(f"  ℹ️  Email will be SKIPPED this run")
print("━"*65)
print()

# ── CONFIG ────────────────────────────────────────────────────
CFG = {
    "history_days"          : 1500,  # ~6 years — gives ~70 monthly bars, ~215 weekly
                                      # bars, plenty for stable RSI(14) on all 3
                                      # timeframes

    # ── RSI levels (same 14-period RSI on every timeframe) ──────
    "rsi_period"             : 14,
    "daily_rsi_prev_max"     : 55.0, # yesterday's RSI must be BELOW this
    "daily_rsi_now_min"      : 59.0, # today's RSI must be within this range
    "daily_rsi_now_max"      : 64.0, # ... (inclusive)
    "weekly_rsi_min"         : 55.0, # weekly RSI currently above this (level only)
    "monthly_rsi_min"        : 55.0, # monthly RSI currently above this (level only)

    "signal_lookback_days"   : 15,   # ~3 trading weeks — how far back to look
                                      # for the daily RSI transition, not just today

    # ── Filters ─────────────────────────────────────────────────
    "min_avg_volume"         : 80_000,
    "min_price"              : 2.0,

    "batch_size"             : 50,
    "batch_sleep"            : 1.5,
}

def calc_rsi(close, period=14):
    """Standard Wilder-smoothed RSI."""
    delta = close.diff()
    gain  = delta.clip(lower=0)
    loss  = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1/period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1/period, min_periods=period, adjust=False).mean()
    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))

def resample_ohlcv(df, rule):
    """Resample an OHLCV dataframe to a coarser bar size."""
    agg = {"Open": "first", "High": "max", "Low": "min",
           "Close": "last", "Volume": "sum"}
    out = df.resample(rule).agg(agg)
    out.dropna(subset=["Close"], inplace=True)
    return out

def check_rsi_transition_at(daily_rsi, weekly_rsi_ff, monthly_rsi_ff, i, cfg):
    """
    Checks the full pattern with the candidate day anchored at bar
    `i`. This is the SINGLE SOURCE OF TRUTH for the pattern logic.

    Returns (passed: bool, details: dict).
    """
    if i < 1 or i >= len(daily_rsi):
        return False, {}
    r_prev, r_now = daily_rsi.iloc[i-1], daily_rsi.iloc[i]
    w, m = weekly_rsi_ff.iloc[i], monthly_rsi_ff.iloc[i]
    if any(np.isnan(v) for v in [r_prev, r_now, w, m]):
        return False, {}

    daily_ok = (r_prev < cfg["daily_rsi_prev_max"] and
                cfg["daily_rsi_now_min"] <= r_now <= cfg["daily_rsi_now_max"])
    weekly_ok = w > cfg["weekly_rsi_min"]
    monthly_ok = m > cfg["monthly_rsi_min"]
    passed = daily_ok and weekly_ok and monthly_ok

    return passed, {
        "idx": i, "rsi_prev": float(r_prev), "rsi_now": float(r_now),
        "weekly_rsi": float(w), "monthly_rsi": float(m),
    }

def find_rsi_transition_signals(daily_rsi, weekly_rsi_ff, monthly_rsi_ff, cfg):
    """
    Scans the last `signal_lookback_days` trading days for the full
    pattern. Returns a list of hit dicts, most recent first.
    """
    n = len(daily_rsi)
    lb = cfg["signal_lookback_days"]
    hits = []
    for back in range(0, lb):
        i = (n - 1) - back
        passed, details = check_rsi_transition_at(daily_rsi, weekly_rsi_ff, monthly_rsi_ff, i, cfg)
        if passed:
            hits.append(details)
    return hits

# ── Technical signal: daily RSI pop + weekly/monthly confirm ─────
def analyze_rsi_transition(sym, df_daily):
    """
    Returns dict with score and setup details, or None if no
    required condition is met anywhere in the lookback window.
    """
    if df_daily is None:
        return None

    price   = float(df_daily["Close"].iloc[-1])
    avg_vol = float(df_daily["Volume"].tail(20).mean())
    if price   < CFG["min_price"]:      return None
    if avg_vol < CFG["min_avg_volume"]: return None

    n = len(df_daily)
    period = CFG["rsi_period"]
    if n < 400:
        return None   # not enough history for a stable monthly RSI

    df_weekly  = resample_ohlcv(df_daily, "W")
    df_monthly = resample_ohlcv(df_daily, "ME")

    rsi_daily   = calc_rsi(df_daily["Close"], period)
    rsi_weekly  = calc_rsi(df_weekly["Close"], period)
    rsi_monthly = calc_rsi(df_monthly["Close"], period)

    # forward-fill weekly/monthly RSI onto the daily index, so each
    # daily bar sees the most recently COMPLETED weekly/monthly RSI
    # as of that date
    weekly_rsi_ff  = rsi_weekly.reindex(df_daily.index, method="ffill")
    monthly_rsi_ff = rsi_monthly.reindex(df_daily.index, method="ffill")

    hits = find_rsi_transition_signals(rsi_daily, weekly_rsi_ff, monthly_rsi_ff, CFG)
    if not hits:
        return None

    sig = hits[0]   # most recent
    sig_idx = sig["idx"]
    days_since_signal = (n - 1) - sig_idx
    recent_signals = [
        {"date": df_daily.index[h["idx"]], "bars_ago": (n-1)-h["idx"]}
        for h in hits
    ]

    # ── Entry / Stop (reasonable defaults — see header note) ─────────
    entry_price = float(df_daily["Close"].iloc[sig_idx])
    stop_loss   = float(df_daily["Low"].tail(10).min()) if sig_idx == n-1 else                   float(df_daily["Low"].iloc[max(0,sig_idx-9):sig_idx+1].min())
    if entry_price <= stop_loss:
        return None
    risk_pct = (entry_price - stop_loss) / entry_price * 100 if entry_price > 0 else 0

    # ── Score (0-100) ────────────────────────────────────────────
    score = 0
    reasons = []
    mid = (CFG["daily_rsi_now_min"] + CFG["daily_rsi_now_max"]) / 2
    band_half = (CFG["daily_rsi_now_max"] - CFG["daily_rsi_now_min"]) / 2
    score += max(0, min(25, 25 - abs(sig["rsi_now"] - mid) / band_half * 25))
    reasons.append(f"DailyRSI{sig['rsi_now']:.0f}")
    score += max(0, min(20, (CFG["daily_rsi_prev_max"] - sig["rsi_prev"])))
    reasons.append(f"PrevRSI{sig['rsi_prev']:.0f}")
    score += min(20, (sig["weekly_rsi"] - CFG["weekly_rsi_min"]) * 0.5)
    reasons.append(f"WeeklyRSI{sig['weekly_rsi']:.0f}")
    score += min(20, (sig["monthly_rsi"] - CFG["monthly_rsi_min"]) * 0.5)
    reasons.append(f"MonthlyRSI{sig['monthly_rsi']:.0f}")
    freshness_pts = max(0, 15 - days_since_signal)
    score += freshness_pts
    reasons.append(f"{days_since_signal}dAgo")
    score = round(min(100, max(0, score)))

    return {
        "Score"          : score,
        "Price"          : round(price, 2),
        "Entry_Price"    : round(entry_price, 2),
        "Stop_Loss"      : round(stop_loss, 2),
        "Risk_%"         : round(risk_pct, 1),
        "Daily_RSI"      : round(sig["rsi_now"], 1),
        "Daily_RSI_Prev" : round(sig["rsi_prev"], 1),
        "Weekly_RSI"     : round(sig["weekly_rsi"], 1),
        "Monthly_RSI"    : round(sig["monthly_rsi"], 1),
        "Signal_Date"    : df_daily.index[sig_idx].strftime("%Y-%m-%d"),
        "Days_Since_Signal"  : days_since_signal,
        "Recent_Signal_Count": len(recent_signals),
        "Recent_Signals" : " | ".join(
            f"{s['date'].strftime('%Y-%m-%d')}" for s in recent_signals),
        "Flags"          : " | ".join(reasons),
        "_df_daily"      : df_daily,
        "_rsi_daily"     : rsi_daily,
        "_weekly_rsi_ff" : weekly_rsi_ff, "_monthly_rsi_ff": monthly_rsi_ff,
    }

# ── Download: daily (→ also Weekly + Monthly via resample) ──────
def _clean(df, min_bars=200):
    if df is None or df.empty: return None
    need = [c for c in ["Open","High","Low","Close","Volume"] if c in df.columns]
    if not all(c in need for c in ["High","Low","Close","Volume"]): return None
    df = df[need].copy()
    df.index = pd.to_datetime(df.index)
    if hasattr(df.index,"tz") and df.index.tz:
        df.index = df.index.tz_localize(None)
    df.dropna(subset=["Close","Volume"], inplace=True)
    return df if (len(df)>=min_bars and float(df["Close"].iloc[-1])>0) else None

def download_daily(symbols, days):
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
                        df = _clean(df, min_bars=300)
                        if df is not None: out[sym] = df
                    except Exception: pass
            elif len(symbols) == 1:
                df = _clean(raw, min_bars=300)
                if df is not None: out[symbols[0]] = df
    except Exception: pass
    for sym in [s for s in symbols if s not in out]:
        for _ in range(2):
            try:
                df = yf.Ticker(sym).history(
                    start=start.strftime("%Y-%m-%d"),
                    end=end.strftime("%Y-%m-%d"),
                    auto_adjust=True, actions=False)
                df = _clean(df, min_bars=300)
                if df is not None: out[sym] = df; break
            except Exception: time.sleep(0.2)
        time.sleep(0.04)
    return out

# ── Live print ────────────────────────────────────────────────
LIVE_COLS = ["Ticker","Price","Score","Entry_Price","Stop_Loss",
             "Daily_RSI","Weekly_RSI","Monthly_RSI"]
_CW = {"Ticker":8,"Price":10,"Score":7,"Entry_Price":12,"Stop_Loss":11,
       "Daily_RSI":10,"Weekly_RSI":11,"Monthly_RSI":12}
_CF = {"Price":"${:.2f}","Score":"{:.0f}","Entry_Price":"${:.2f}",
       "Stop_Loss":"${:.2f}"}
_hdr_done = False

def _live_header():
    global _hdr_done
    if _hdr_done: return
    print("\n" + "━"*95)
    print("  📊  LIVE MATCHES  —  each stock printed the moment it passes all 3 timeframes")
    print("━"*95)
    h = "".join(f"  {c:<{_CW.get(c,12)}}" for c in LIVE_COLS)
    print(h)
    print("  " + "─"*93)
    _hdr_done = True

def live_print(r):
    _live_header()
    row = ""
    for c in LIVE_COLS:
        val = r.get(c,"—")
        w   = _CW.get(c,12)
        fmt = _CF.get(c)
        try:   s = fmt.format(val) if (fmt and val not in("—",None)) else str(val)
        except Exception: s = str(val)
        row += f"  {s:<{w}}"
    print(row)

# ── Health check ──────────────────────────────────────────────
print("━"*65)
print("  STEP 1  DATA CHECK")
print("━"*65)
chk_d = download_daily(["AAPL","MSFT","NVDA"], CFG["history_days"])
if not chk_d:
    print("❌  No data.")
else:
    for s, dd in chk_d.items():
        print(f"  ✅ {s}: daily {len(dd)} bars (${float(dd['Close'].iloc[-1]):.2f}, "
              f"{dd.index[-1].date()})")
print()

# ── Ticker list ───────────────────────────────────────────────
print("━"*65)
print("  STEP 2  FETCH TICKERS")
print("━"*65)

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
            print(f"  ✅ {label:<18}: +{len(pool)-b:>4} → {len(pool)}")
        except Exception as e: print(f"  ⚠️  {label}: {e}")
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
        print(f"  ✅ {'NASDAQ API':<18}: +{len(pool)-b:>4} → {len(pool)}")
    except Exception as e: print(f"  ⚠️  NASDAQ API: {e}")
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
    print(f"  ✅ {'Static fallback':<18}: +{len(pool)-b:>4} → {len(pool)}")
    clean = sorted({s.upper() for s in pool if isinstance(s,str)
                    and s.isalpha() and 1<=len(s)<=5})
    print(f"\n  🎯 Total: {len(clean)} tickers")
    return clean

TICKERS = get_tickers()
print()

# ── Main scan — single pass (daily download only, then check) ────
print("━"*65)
print(f"  STEP 3  SCANNING {len(TICKERS)} TICKERS")
print("━"*65)
print("  Fetching daily bars (→ also gives Weekly and Monthly via resample)")
print("  A stock only matches if Monthly + Weekly + Daily RSI conditions all fire\n")

_hdr_done = False
results = []
no_daily_data = 0

daily_batches = [TICKERS[i:i+CFG["batch_size"]]
                 for i in range(0, len(TICKERS), CFG["batch_size"])]

daily_map = {}

with tqdm(total=len(TICKERS), desc="Daily fetch", unit="stk",
          bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]") as pbar:
    for batch in daily_batches:
        got = download_daily(batch, CFG["history_days"])
        daily_map.update(got)
        no_daily_data += len(batch) - len(got)
        pbar.update(len(batch))
        time.sleep(CFG["batch_sleep"])

got_daily = len(TICKERS) - no_daily_data
print(f"\n  Daily data : {got_daily}/{len(TICKERS)} tickers")
print()

for sym in tqdm(list(daily_map.keys()), desc="Checking RSI reversal", unit="stk"):
    try:
        r = analyze_rsi_transition(sym, daily_map[sym])
        if r is None: continue
        r["Ticker"] = sym
        results.append(r)
        live_print(r)
    except Exception: pass

print(f"\n{'━'*65}")
print(f"  SCAN COMPLETE")
print(f"  Tickers scanned : {len(TICKERS)}")
print(f"  Daily data      : {got_daily}")
print(f"  ✅ Matches       : {len(results)}")
print(f"{'━'*65}")

if not results:
    print("\n  No matches. Try relaxing:")
    print("   daily_rsi_prev_max         55.0 → 58.0")
    print("   daily_rsi_now_min/max      59-64 → 57-66  (widen the band)")
    print("   weekly_rsi_min             55.0 → 50.0")
    print("   monthly_rsi_min            55.0 → 50.0")
    print("   min_price                  2 → 1")
    print("   min_avg_volume         80000 → 50000")

results.sort(key=lambda x: x["Score"], reverse=True)

# ── Always build df_out and save/email (even if 0 results) ────
ts      = datetime.today().strftime("%Y%m%d_%H%M")
out_dir = os.environ.get("GITHUB_WORKSPACE", os.getcwd())

COLS = [
    "Ticker","Price","Score",
    "Entry_Price","Stop_Loss","Risk_%",
    "Monthly_RSI","Weekly_RSI","Daily_RSI","Daily_RSI_Prev",
    "Days_Since_Signal","Recent_Signal_Count","Recent_Signals",
    "Flags",
]
df_out = pd.DataFrame([{k:v for k,v in r.items() if not k.startswith("_")}
                        for r in results]) if results else pd.DataFrame(columns=COLS)
if not df_out.empty:
    df_out = df_out[[c for c in COLS if c in df_out.columns]]
    df_out.reset_index(drop=True, inplace=True)

FMT = {
    "Price"       : lambda v: f"${v:.2f}",
    "Score"       : lambda v: f"{v:.0f}",
    "Entry_Price" : lambda v: f"${v:.2f}",
    "Stop_Loss"   : lambda v: f"${v:.2f}",

    "Risk_%"      : lambda v: f"{v:.1f}%",

    "Monthly_RSI" : lambda v: f"{v:.1f}",
    "Weekly_RSI"  : lambda v: f"{v:.1f}",
    "Daily_RSI"   : lambda v: f"{v:.1f}",
    "Daily_RSI_Prev": lambda v: f"{v:.1f}",
    "Days_Since_Signal": lambda v: f"{int(v)}d ago",

}

def fmt_v(col, val):
    if val is None or (isinstance(val, float) and np.isnan(val)): return "—"
    try:
        if col in FMT: return FMT[col](val)
    except Exception: pass
    return str(val) if str(val) not in ("nan","None","") else "—"

if _IN_NOTEBOOK and results:
    DISP = ["Ticker","Price","Score","Entry_Price","Stop_Loss",
            "Daily_RSI","Daily_RSI_Prev","Weekly_RSI","Monthly_RSI","Signal_Date"]
    DISP = [c for c in DISP if c in df_out.columns]

    gc = "#22c55e"

    th = "".join(
        f'<th style="background:#0f172a;color:#e2e8f0;padding:9px 12px;'
        f'font-size:11px;font-weight:700;text-align:center;'
        f'border-bottom:2px solid {gc};white-space:nowrap">{c}</th>'
        for c in DISP
    )
    rows_html = ""
    for i, r in enumerate(results):
        bg  = "#ffffff" if i%2==0 else "#f0f9ff"
        tds = ""
        for col in DISP:
            raw  = r.get(col)
            disp = fmt_v(col, raw)
            sty  = ""
            if col == "Score":
                try:
                    v = float(raw)
                    g = int(min(220, 80 + v*1.4))
                    sty = f"background:rgb(20,{g},60);color:#fff;font-weight:700;text-align:center"
                except Exception: pass
            elif col == "Daily_RSI":
                try:
                    v = float(raw)
                    clr = "#22c55e" if CFG["daily_rsi_now_min"] <= v <= CFG["daily_rsi_now_max"] else "#f59e0b"
                    sty = f"color:{clr};font-weight:600"
                except Exception: pass
            tds += (f'<td style="padding:7px 12px;font-size:12px;'
                    f'border-bottom:1px solid #e2e8f0;white-space:nowrap;{sty}">'
                    f'{disp}</td>')
        rows_html += f'<tr style="background:{bg}">{tds}</tr>\n'

    table_html = f"""
<div style="margin:14px 0">
  <div style="background:linear-gradient(90deg,{gc}22,#0f172a);
          border-left:4px solid {gc};border-radius:6px 6px 0 0;
          padding:10px 18px;display:flex;align-items:center;gap:10px">
    <span style="font-size:18px">🎯</span>
    <span style="color:#f1f5f9;font-size:15px;font-weight:700">Daily RSI Pop + Weekly/Monthly Confirm</span>
    <span style="color:{gc};font-size:12px;margin-left:8px">{len(results)} stock{'s' if len(results)!=1 else ''}</span>
  </div>
  <div style="overflow-x:auto;border:1px solid #e2e8f0;border-top:none;
          border-radius:0 0 8px 8px;box-shadow:0 2px 8px rgba(0,0,0,0.05)">
    <table style="border-collapse:collapse;width:100%;min-width:700px">
      <thead><tr>{th}</tr></thead>
      <tbody>{rows_html}</tbody>
    </table>
  </div>
</div>"""

    header_html = f"""
<div style="background:linear-gradient(135deg,#0f172a,#1e3a5f);
        border-radius:10px;padding:18px 24px;margin-bottom:8px;
        font-family:'Segoe UI',Arial,sans-serif">
  <h2 style="margin:0;color:#60a5fa;font-size:20px;font-weight:700">
    📈 Daily RSI Pop + Weekly/Monthly RSI Confirm
  </h2>
  <p style="margin:6px 0 0;color:#94a3b8;font-size:12px">
    {datetime.today().strftime('%Y-%m-%d %H:%M')} &nbsp;·&nbsp;
    <b style="color:#22c55e">{len(results)} matches</b> from {len(TICKERS)} tickers
    &nbsp;·&nbsp; pattern checked over the last {CFG['signal_lookback_days']} trading days
  </p>
</div>"""

    legend_html = f"""
<div style="background:#f8fafc;border:1px solid #e2e8f0;border-radius:8px;
        padding:12px 18px;margin-top:6px;font-size:11px;color:#64748b;
        font-family:'Segoe UI',Arial,sans-serif">
  <b style="color:#475569">GUIDE</b> &nbsp;·&nbsp;
  Monthly RSI currently at/above 60 (current month) &nbsp;·&nbsp;
  Weekly RSI currently above 60 (current week) &nbsp;·&nbsp;
  Daily RSI crossed above 40 with a GREEN candle that day, any day in
  the last {CFG['signal_lookback_days']} trading days &nbsp;·&nbsp;
  Entry_Price/Stop_Loss = that green candle's high/low (may be a few
  days old — see Days_Since_Signal) &nbsp;·&nbsp;
  Entry_Price/Stop_Loss are reasonable defaults, not explicitly requested
</div>"""

    display_html(header_html + table_html + legend_html)

elif results:
    # ASCII table (CLI/GitHub Actions mode)
    CLI_COLS = ["Ticker","Price","Score","Entry_Price","Stop_Loss",
                "Daily_RSI","Daily_RSI_Prev","Weekly_RSI","Monthly_RSI","Signal_Date"]
    CLI_COLS = [c for c in CLI_COLS if c in df_out.columns]
    col_w = {c: max(len(c), max(
        len(fmt_v(c, df_out[c].iloc[i])) for i in range(len(df_out))
    ))+2 for c in CLI_COLS}
    top  = "┬".join("─"*col_w[c] for c in CLI_COLS)
    sep  = "┼".join("─"*col_w[c] for c in CLI_COLS)
    bot  = "┴".join("─"*col_w[c] for c in CLI_COLS)
    hdr  = "│".join(c.center(col_w[c]) for c in CLI_COLS)
    inner= sum(col_w.values()) + len(CLI_COLS) - 1
    print()
    print(f"  ╔{'═'*inner}╗")
    tit = f"  RSI Multi-TF Reversal (Monthly+Weekly+Daily)   {datetime.today().strftime('%Y-%m-%d')}   {len(df_out)} matches"
    print(f"  ║{tit.center(inner)}║")
    print(f"  ╚{'═'*inner}╝\n")
    print(f"  ┌{top}┐")
    print(f"  │{hdr}│")
    print(f"  ├{sep}┤")
    for i,(_, row_) in enumerate(df_out.iterrows()):
        cells=[fmt_v(c,row_.get(c)).center(col_w[c]) for c in CLI_COLS]
        print(f"  │{'│'.join(cells)}│")
        if i<len(df_out)-1: print(f"  ├{sep}┤")
    print(f"  └{bot}┘")
    print(f"""
  COLUMN KEY
  ──────────────────────────────────────────────────────
  Score           0-100 (RSI strength on all 3 timeframes + risk:reward)
  Entry_Price     high of the signal day's green candle (breakout trigger)
  Stop_Loss       low of that same green candle
  Daily_RSI_Prev  yesterday's daily RSI (must have been < daily_rsi_prev_max)
  Daily_RSI       today's daily RSI (must land in the 59-64 band)
  Days_Since_Signal  how many trading days ago the daily RSI cross fired
                     (0 = today; checked over the last daily_cross_
                     lookback_days trading days, not just today)
  ──────────────────────────────────────────────────────""")

# Save
fpath = os.path.join(out_dir, f"rsi_daily_pop_weekly_monthly_confirm_{ts}.csv")
df_out.to_csv(fpath, index=False)
print(f"\n  💾 CSV → {fpath}")
tv = os.path.join(out_dir, f"tv_rsi_daily_pop_weekly_monthly_confirm_{ts}.txt")
with open(tv,"w") as f:
    f.write(f"###Daily RSI Pop + Weekly/Monthly Confirm {datetime.today().strftime('%Y-%m-%d')}\n")
    for r in results: f.write(f"NASDAQ:{r['Ticker']}\n")
print(f"  📋 TradingView → {tv}")
if results:
    print(f"\n  📋 Tickers (comma-separated):")
    print(f"  {', '.join(r['Ticker'] for r in results)}")

# ── Email with CSV attached ───────────────────────────────
def _send_email(rl, csv_path):
    import smtplib
    from email.mime.multipart import MIMEMultipart
    from email.mime.text      import MIMEText
    from email.mime.base      import MIMEBase
    from email                import encoders

    gu = _GMAIL_USER; gp = _GMAIL_PASS; et = _EMAIL_TO

    if not gu:
        print("[Email] ❌  GMAIL_USER secret is empty")
        return
    if not gp:
        print("[Email] ❌  GMAIL_PASS secret is empty")
        print("         → Must be a Gmail App Password (16 chars, no spaces)")
        return
    if not et:
        print("[Email] ❌  EMAIL_TO secret is empty")
        return

    eto = [e.strip() for e in et.split(",") if e.strip()]
    cnt = len(rl)

    try:
        print(f"[Email] Sending to {et}  ({cnt} results)...")

        th_e = "".join(
            f'<th style="background:#1e293b;color:#e2e8f0;padding:8px 11px;'
            f'font-size:11px;font-weight:700;border-bottom:2px solid #3b82f6;'
            f'white-space:nowrap">{c}</th>'
            for c in ["Ticker","Price","Score","Entry_Price","Stop_Loss",
                      "Daily_RSI","Weekly_RSI","Monthly_RSI"]
        )
        rows_e = ""
        for i, r in enumerate(rl[:50]):
            bg  = "#fff" if i % 2 == 0 else "#f0f9ff"
            ticker = r.get("Ticker","—")
            price  = r.get("Price",0) or 0
            score  = r.get("Score",0) or 0
            entry  = r.get("Entry_Price",0) or 0
            stop   = r.get("Stop_Loss",0) or 0
            drsi   = r.get("Daily_RSI",0) or 0
            wrsi   = r.get("Weekly_RSI",0) or 0
            mrsi   = r.get("Monthly_RSI",0) or 0
            dsig   = r.get("Days_Since_Signal")
            dsig_disp = f"{int(dsig)}d ago" if dsig is not None else "—"
            rows_e += (
                f'<tr style="background:{bg}">'
                f'<td style="padding:6px 11px;font-size:12px;font-weight:700">{ticker}</td>'
                f'<td style="padding:6px 11px;font-size:12px">${float(price):.2f}</td>'
                f'<td style="padding:6px 11px;font-size:12px;font-weight:700;'
                f'background:#166534;color:#fff;text-align:center">{float(score):.0f}</td>'
                f'<td style="padding:6px 11px;font-size:12px;color:#22c55e">${float(entry):.2f}</td>'
                f'<td style="padding:6px 11px;font-size:12px;color:#ef4444">${float(stop):.2f}</td>'
                f'<td style="padding:6px 11px;font-size:12px;color:#3b82f6">${float(target):.2f}</td>'
                f'<td style="padding:6px 11px;font-size:12px;font-weight:600">{float(rr):.2f}</td>'
                f'<td style="padding:6px 11px;font-size:12px;text-align:center;'
                f'color:#a78bfa;font-weight:600">{dsig_disp}</td>'
                f'</tr>'
            )
        no_results_msg = ('<tr><td colspan="8" style="padding:20px;text-align:center;'
                           'color:#94a3b8;font-size:13px">No matches today</td></tr>')

        html_e = f"""<!DOCTYPE html><html><body style="margin:0;padding:0;
background:#f1f5f9;font-family:'Segoe UI',Arial,sans-serif">
<table width="100%" cellpadding="0" cellspacing="0"><tr><td align="center" style="padding:20px 10px">
<table width="100%" style="max-width:800px;background:#fff;border-radius:12px;
       overflow:hidden;box-shadow:0 4px 16px rgba(0,0,0,0.08)">
  <tr><td style="background:linear-gradient(135deg,#0f172a,#1e3a5f);padding:22px 28px">
<h1 style="margin:0;color:#60a5fa;font-size:20px;font-weight:700">
  📊 Daily RSI Pop + Weekly/Monthly Confirm
</h1>
<p style="margin:6px 0 0;color:#94a3b8;font-size:12px">
  {datetime.today().strftime('%Y-%m-%d %H:%M UTC')} &nbsp;·&nbsp;
  {cnt} match{'es' if cnt!=1 else ''} found — Monthly RSI cross, Weekly RSI strength,
  Daily RSI reversal, all at once
</p>
  </td></tr>
  <tr><td style="padding:16px">
<div style="overflow-x:auto;border-radius:8px;border:1px solid #e2e8f0">
  <table style="border-collapse:collapse;width:100%;min-width:600px">
    <thead><tr>{th_e}</tr></thead>
    <tbody>{rows_e or no_results_msg}</tbody>
  </table>
</div>
<p style="font-size:11px;color:#64748b;margin:8px 0 0">
  📎 Full results attached as CSV
</p>
{f'''<div style="margin-top:10px;background:#f8fafc;border:1px solid #e2e8f0;
        border-radius:6px;padding:10px 14px">
  <p style="margin:0 0 4px;font-size:10px;color:#94a3b8;font-weight:700">
    TICKERS (comma-separated, copy/paste)
  </p>
  <p style="margin:0;font-size:12px;color:#1e293b;font-family:monospace;
      word-break:break-all">{", ".join(r.get("Ticker","") for r in rl)}</p>
</div>''' if rl else ''}
  </td></tr>
  <tr><td style="background:#f8fafc;padding:12px 28px;
             border-top:1px solid #e2e8f0;text-align:center">
<p style="margin:0;color:#94a3b8;font-size:10px">
  ⚠️ Not financial advice &nbsp;·&nbsp; Auto-generated by GitHub Actions
</p>
  </td></tr>
</table>
</td></tr></table>
</body></html>"""

        plain_lines = [
            f"Daily RSI Pop + Weekly/Monthly Confirm (checked over last {CFG['signal_lookback_days']} trading days) — {datetime.today().strftime('%Y-%m-%d')}",
            f"{cnt} matches (Monthly RSI cross + Weekly RSI strength + Daily RSI reversal)",
            "="*60,
        ]
        if rl:
            for r in rl[:50]:
                ticker = r.get("Ticker","—")
                price  = r.get("Price",0) or 0
                score  = r.get("Score",0) or 0
                entry  = r.get("Entry_Price",0) or 0
                stop   = r.get("Stop_Loss",0) or 0
                drsi   = r.get("Daily_RSI",0) or 0
                wrsi   = r.get("Weekly_RSI",0) or 0
                mrsi   = r.get("Monthly_RSI",0) or 0
                dsig   = r.get("Days_Since_Signal")
                dsig_disp = f"{int(dsig)}d ago" if dsig is not None else "—"
                plain_lines.append(
                    f"{ticker:<7} ${float(price):.2f}  Score:{float(score):.0f}  "
                    f"Entry:${float(entry):.2f}  SL:${float(stop):.2f}  "
                    f"DailyRSI:{float(drsi):.1f}  WeeklyRSI:{float(wrsi):.1f}  MonthlyRSI:{float(mrsi):.1f}"
                )
            plain_lines.append("")
            plain_lines.append("Tickers (comma-separated):")
            plain_lines.append(", ".join(r.get("Ticker","") for r in rl))
        else:
            plain_lines.append("No matches today")
        plain_lines.append("\nFull results in CSV attachment.")
        plain_e = "\n".join(plain_lines)

        subj = (f"📊 RSI Multi-TF Reversal (M+W+D) — {cnt} signal{'s' if cnt!=1 else ''}"
                f" — {datetime.today().strftime('%Y-%m-%d')}")

        msg = MIMEMultipart("mixed")
        msg["Subject"] = subj
        msg["From"]    = gu
        msg["To"]      = ", ".join(eto)

        alt = MIMEMultipart("alternative")
        alt.attach(MIMEText(plain_e, "plain"))
        alt.attach(MIMEText(html_e,  "html"))
        msg.attach(alt)

    except Exception as e:
        print(f"[Email] ❌  Failed to build email body: {type(e).__name__}: {e}")
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
            print(f"[Email] 📎 Attached: {os.path.basename(csv_path)} ({sz:,} bytes)")
        except Exception as e:
            print(f"[Email] ⚠️  CSV attach failed: {e}")

    try:
        print(f"[Email] Connecting to smtp.gmail.com:465 ...")
        with smtplib.SMTP_SSL("smtp.gmail.com", 465) as srv:
            srv.login(gu, gp.replace(" ", ""))
            srv.sendmail(gu, eto, msg.as_string())
        print(f"[Email] ✅  Sent successfully to: {', '.join(eto)}")
        print(f"[Email]    Subject: {subj}")
    except smtplib.SMTPAuthenticationError:
        print("[Email] ❌  AUTHENTICATION FAILED")
        print("         GMAIL_PASS must be a Gmail App Password, NOT your login password")
    except smtplib.SMTPException as e:
        print(f"[Email] ❌  SMTP error: {e}")
    except Exception as e:
        print(f"[Email] ❌  Unexpected error: {type(e).__name__}: {e}")

try:
    _send_email(results, fpath)
except Exception as e:
    print(f"[Email] ❌  Unexpected top-level error: {type(e).__name__}: {e}")
    print("[Email]    Continuing — CSV is still saved.")

if _IN_NOTEBOOK:
    try:
        from google.colab import files
        files.download(fpath); files.download(tv)
    except Exception: pass
else:
    print("  (CI: files in workspace, email sent)")

# ── Charts for top 5 (price + entry/stop/target + RSI panel) ────
if results:
    top = results[:min(5,len(results))]
    fig, axes = plt.subplots(len(top),2,figsize=(16,4.2*len(top)),facecolor="#0f172a",
                              gridspec_kw={"height_ratios":[1]*len(top), "width_ratios":[3,1]})
    if len(top)==1: axes = axes.reshape(1,2)
    for row, r in zip(axes, top):
        ax, ax_rsi = row[0], row[1]
        df_p = r["_df_daily"].tail(150).copy()
        ax.set_facecolor("#0f172a")
        ax.plot(df_p.index, df_p["Close"], color="#60a5fa", lw=1.6, label="Daily Close", zorder=5)
        ax.scatter([df_p.index[-1]], [r["Entry_Price"]], color="#22c55e", s=60, zorder=6,
                   marker="^", label="Entry")
        ax.axhline(r["Stop_Loss"], color="#ef4444", lw=1.0, ls="--", alpha=0.85,
                  label=f"Stop ${r['Stop_Loss']:.2f}")

        ax.set_title(
            f"{r['Ticker']}  |  ${r['Price']:.2f}  |  Score {r['Score']}  |  "
            f"Entry ${r['Entry_Price']:.2f}  RSI {r['Daily_RSI']:.0f}",
            color="#e2e8f0", fontsize=9, fontweight="bold", pad=7)
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %y"))
        ax.tick_params(colors="#94a3b8", labelsize=8)
        for sp in ax.spines.values(): sp.set_edgecolor("#1e3a5f")
        ax.legend(loc="upper left", facecolor="#1e293b", labelcolor="#e2e8f0",
                  fontsize=7, framealpha=0.9)
        ax.grid(color="#1e3a5f", ls="--", lw=0.5, alpha=0.6)

        rsi_p = r["_rsi_daily"].reindex(df_p.index)
        ax_rsi.set_facecolor("#0f172a")
        ax_rsi.plot(df_p.index, rsi_p, color="#f472b6", lw=1.3)
        ax_rsi.axhline(40, color="#94a3b8", lw=0.8, ls=":")
        ax_rsi.axhline(60, color="#94a3b8", lw=0.8, ls=":")
        ax_rsi.set_ylim(0,100)
        ax_rsi.set_title(f"Daily RSI  M:{r['Monthly_RSI']:.0f} W:{r['Weekly_RSI']:.0f}",
                          color="#e2e8f0", fontsize=8, pad=5)
        ax_rsi.xaxis.set_major_formatter(mdates.DateFormatter("%b"))
        ax_rsi.tick_params(colors="#94a3b8", labelsize=7)
        for sp in ax_rsi.spines.values(): sp.set_edgecolor("#1e3a5f")
        ax_rsi.grid(color="#1e3a5f", ls="--", lw=0.5, alpha=0.6)
    plt.suptitle(
        f"Daily RSI Pop + Weekly/Monthly Confirm  ·  {datetime.today().strftime('%Y-%m-%d')}",
        color="#60a5fa", fontsize=12, fontweight="bold", y=1.001)
    plt.tight_layout()
    cp = os.path.join(out_dir, f"rsi_daily_pop_weekly_monthly_confirm_chart_{ts}.png")
    plt.savefig(cp, dpi=150, bbox_inches="tight", facecolor="#0f172a")
    if _IN_NOTEBOOK: plt.show()
    else: plt.close()
    print(f"  📊 Chart → {cp}")
    if _IN_NOTEBOOK:
        try:
            from google.colab import files; files.download(cp)
        except Exception: pass

print(f"""
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  📋 SIGNAL (all required — scanned over the last
  signal_lookback_days trading days, {CFG['signal_lookback_days']}d ≈ 3 weeks,
  not just today)

  1) DAILY RSI TRANSITION: yesterday's RSI(14) was BELOW
     daily_rsi_prev_max ({CFG['daily_rsi_prev_max']:.0f}), AND today's RSI(14) is
     now between daily_rsi_now_min and daily_rsi_now_max
     ({CFG['daily_rsi_now_min']:.0f}–{CFG['daily_rsi_now_max']:.0f}, inclusive) — a fresh pop into
     that specific zone.
  2) WEEKLY: RSI(14) currently ABOVE weekly_rsi_min
     ({CFG['weekly_rsi_min']:.0f}) — level only, no freshness requirement.
  3) MONTHLY: RSI(14) currently ABOVE monthly_rsi_min
     ({CFG['monthly_rsi_min']:.0f}) — level only, no freshness requirement.

  If the transition fired more than once in the window, the MOST
  RECENT occurrence is used for Entry/Stop/scoring.

  📋 DATA SOURCING
  Only 1 download per ticker: daily bars (~6 years, for stable
  monthly RSI). Weekly and Monthly RSI are both computed from the
  SAME daily data via resampling ('W' and 'ME'), then forward-filled
  onto the daily index — no separate network calls needed.

  📋 OUTPUT (reasonable defaults — not explicitly requested)
  Entry_Price = the signal day's CLOSE
  Stop_Loss   = the lowest LOW in the 10 trading days up to and
                including the signal day
  Signal_Date = the exact calendar date the transition fired
  Days_Since_Signal = how many trading days ago (0 = today)
  Recent_Signals    = every date the transition fired within the window

  📋 SCORE (0-100)
  Daily RSI centered in the 59-64 band (0-25) + how far below 55
  yesterday's RSI was (0-20) + weekly RSI strength above 55 (0-20) +
  monthly RSI strength above 55 (0-20) + freshness (0-15)

  💡 BEST SETUPS
  Score > 70                strong alignment across all 3 timeframes
  Daily_RSI near 61.5           dead center of the 59-64 band
  Weekly_RSI / Monthly_RSI high    well above 55, confirms real strength
  Days_Since_Signal = 0-3              freshest transition

  ⚙️  TUNE IF 0 RESULTS
  daily_rsi_prev_max          55.0 → 58.0
  daily_rsi_now_min/max        59-64 → 57-66  (widen the target band)
  weekly_rsi_min               55.0 → 50.0
  monthly_rsi_min              55.0 → 50.0
  signal_lookback_days           15 → 25    (search further back)
  min_price                        2 → 1
  min_avg_volume               80000 → 50000
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
""")
