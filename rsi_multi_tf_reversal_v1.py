# ============================================================
# NASDAQ — RSI Multi-Timeframe Reversal Scanner (v1)
# ============================================================
#
# SIGNAL (all required):
#   1. MONTHLY: RSI(14) just crossed ABOVE 60 (was below 60 last
#      month, is at/above 60 this month) — a fresh long-term
#      momentum shift, not a level it's been sitting above for a
#      while.
#   2. WEEKLY: RSI(14) is currently ABOVE 60 — confirms the
#      medium-term trend is already participating (no freshness
#      requirement here, just the level).
#   3. DAILY: RSI(14) just crossed ABOVE 40 from below 40 (a fresh
#      short-term reversal out of a pullback), AND that exact day's
#      candle is GREEN (close > open).
#
# DATA — only 1 download per ticker: daily bars. Weekly and Monthly
# RSI are both computed from the SAME daily data via resampling
# (Weekly = 'W', Monthly = 'ME') — no separate network calls needed.
#
# OUTPUT:
#   Entry_Price = HIGH of the signal day's green candle (the
#                 breakout trigger — enter once price clears it)
#   Stop_Loss   = LOW of that same green candle
#   Target      = the last CONFIRMED swing high before the signal
#                 (a fractal high with `swing_arm` bars of lower
#                 highs on both sides) — the trade is only valid if
#                 this target sits above the entry price
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
                                      # timeframes plus a full swing-high search window

    # ── RSI levels (same 14-period RSI on every timeframe) ──────
    "rsi_period"             : 14,
    "monthly_rsi_cross_level": 60,   # fresh cross above this, on the monthly bar
    "weekly_rsi_level"       : 60,   # just needs to be above this currently
    "daily_rsi_cross_level"  : 40,   # fresh cross above this, on today's bar
    "daily_cross_lookback_days": 21, # ~1 trading month — how far back to look
                                      # for the daily RSI cross, not just today

    # ── Swing-high target ────────────────────────────────────────
    "swing_arm"              : 5,    # bars of lower highs required on each side
                                      # to confirm a fractal swing high
    "swing_search_window"    : 252,  # how far back (trading days) to search for
                                      # the last confirmed swing high

    # ── Filters ─────────────────────────────────────────────────
    "min_avg_volume"         : 80_000,
    "min_price"              : 2.0,

    "batch_size"             : 50,
    "batch_sleep"            : 1.5,
}

# ── Uptrend test (reused across all 4 timeframes) ──────────────
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

def find_last_swing_high(df, before_idx, arm=5, window=252, min_value=None):
    """
    Searches backward from just before `before_idx` for the most
    recent CONFIRMED fractal swing high — High[i] greater than the
    High of `arm` bars on both sides — within `window` bars.

    If `min_value` is given, swing highs at or below it are skipped
    (they aren't valid upside targets — price has already exceeded
    them) and the search continues further back for one that is.

    Returns (idx, high_value) or (None, None) if none found.
    """
    start_search = before_idx - arm - 1
    earliest = max(arm, before_idx - window)
    for i in range(start_search, earliest - 1, -1):
        if i - arm < 0: continue
        left  = df["High"].iloc[i-arm:i]
        right = df["High"].iloc[i+1:i+1+arm]
        if len(right) < arm: continue
        h = df["High"].iloc[i]
        if h > left.max() and h >= right.max():
            if min_value is not None and h <= min_value:
                continue   # not a valid target — keep looking further back
            return i, float(h)
    return None, None

# ── Technical signal: RSI multi-timeframe reversal ───────────────
def analyze_rsi_multi_tf_reversal(sym, df_daily):
    """
    Returns dict with score and setup details, or None if any
    required condition fails.
    """
    if df_daily is None:
        return None

    price   = float(df_daily["Close"].iloc[-1])
    avg_vol = float(df_daily["Volume"].tail(20).mean())
    if price   < CFG["min_price"]:      return None
    if avg_vol < CFG["min_avg_volume"]: return None

    n = len(df_daily)
    period = CFG["rsi_period"]
    if n < CFG["swing_search_window"] + CFG["swing_arm"] + period + 5:
        return None   # not enough history for a stable monthly RSI + swing search

    df_weekly  = resample_ohlcv(df_daily, "W")
    df_monthly = resample_ohlcv(df_daily, "ME")

    rsi_daily   = calc_rsi(df_daily["Close"], period)
    rsi_weekly  = calc_rsi(df_weekly["Close"], period)
    rsi_monthly = calc_rsi(df_monthly["Close"], period)

    if len(rsi_monthly) < 2 or len(rsi_weekly) < 1 or len(rsi_daily) < 2:
        return None

    cur_rsi_m, prev_rsi_m = rsi_monthly.iloc[-1], rsi_monthly.iloc[-2]
    cur_rsi_w             = rsi_weekly.iloc[-1]
    cur_rsi_d, prev_rsi_d = rsi_daily.iloc[-1], rsi_daily.iloc[-2]
    if any(np.isnan(v) for v in [cur_rsi_m, prev_rsi_m, cur_rsi_w, cur_rsi_d, prev_rsi_d]):
        return None

    # ── Condition 1: Monthly RSI just crossed above 60 ─────────────
    m_level = CFG["monthly_rsi_cross_level"]
    monthly_cross = (prev_rsi_m < m_level) and (cur_rsi_m >= m_level)
    if not monthly_cross:
        return None

    # ── Condition 2: Weekly RSI currently above 60 ──────────────────
    w_level = CFG["weekly_rsi_level"]
    weekly_above = cur_rsi_w > w_level
    if not weekly_above:
        return None

    # ── Condition 3: Daily RSI just crossed above 40, green candle —
    #    scans the last daily_cross_lookback_days trading days (not
    #    just today), so a setup that formed within the last ~month
    #    still surfaces even if today itself has no fresh cross ──────
    d_level = CFG["daily_rsi_cross_level"]
    lb = CFG["daily_cross_lookback_days"]
    hits = []
    for back in range(0, lb):
        i = (n - 1) - back
        if i < 1: break
        c_i, c_p = rsi_daily.iloc[i], rsi_daily.iloc[i-1]
        if np.isnan(c_i) or np.isnan(c_p): continue
        if not (c_p < d_level and c_i >= d_level): continue
        o_i = float(df_daily["Open"].iloc[i])
        cl_i = float(df_daily["Close"].iloc[i])
        if cl_i > o_i:   # green candle that same day
            hits.append({"idx": i, "rsi_prev": float(c_p), "rsi_cur": float(c_i)})
    if not hits:
        return None

    sig = hits[0]   # most recent hit (scanned newest-first)
    sig_idx = sig["idx"]
    cur_rsi_d, prev_rsi_d = sig["rsi_cur"], sig["rsi_prev"]
    days_since_signal = (n - 1) - sig_idx
    recent_signals = [
        {"date": df_daily.index[h["idx"]], "bars_ago": (n-1)-h["idx"]}
        for h in hits
    ]

    # ── Entry / Stop / Target (as of the signal day, which may not
    #    be today if the freshest cross was earlier in the window) ──
    entry_price = float(df_daily["High"].iloc[sig_idx])   # breakout above the green candle
    stop_loss   = float(df_daily["Low"].iloc[sig_idx])
    if entry_price <= stop_loss:
        return None

    sw_idx, swing_high = find_last_swing_high(
        df_daily, before_idx=sig_idx, arm=CFG["swing_arm"],
        window=CFG["swing_search_window"], min_value=entry_price)
    if swing_high is None:
        return None   # no valid upside target above the entry within the window

    risk   = entry_price - stop_loss
    reward = swing_high - entry_price
    risk_reward = reward / risk if risk > 0 else 0
    risk_pct = risk / entry_price * 100 if entry_price > 0 else 0

    # ── Score (0-100) ────────────────────────────────────────────
    score = 0
    reasons = []
    score += min(25, 10 + (cur_rsi_m - m_level))
    reasons.append(f"MonthlyRSI{cur_rsi_m:.0f}")
    score += min(25, 10 + (cur_rsi_w - w_level) * 0.5)
    reasons.append(f"WeeklyRSI{cur_rsi_w:.0f}")
    score += min(20, 10 + (cur_rsi_d - d_level))
    reasons.append(f"DailyRSI{cur_rsi_d:.0f}Cross")
    score += min(30, risk_reward * 10)
    reasons.append(f"RR{risk_reward:.1f}")
    score = round(min(100, max(0, score)))

    return {
        "Score"          : score,
        "Price"          : round(price, 2),
        "Entry_Price"    : round(entry_price, 2),
        "Stop_Loss"      : round(stop_loss, 2),
        "Target"         : round(swing_high, 2),
        "Risk_%"         : round(risk_pct, 1),
        "Risk_Reward"    : round(risk_reward, 2),
        "Monthly_RSI"    : round(cur_rsi_m, 1),
        "Weekly_RSI"     : round(cur_rsi_w, 1),
        "Daily_RSI"      : round(cur_rsi_d, 1),
        "Daily_RSI_Prev" : round(prev_rsi_d, 1),
        "Swing_High_Bars_Ago": (n - 1) - sw_idx,
        "Days_Since_Signal"  : days_since_signal,
        "Recent_Signal_Count": len(recent_signals),
        "Recent_Signals" : " | ".join(
            f"{s['date'].strftime('%Y-%m-%d')}({s['bars_ago']}d ago)" for s in recent_signals),
        "Flags"          : " | ".join(reasons),
        "_df_daily"      : df_daily,
        "_rsi_daily"     : rsi_daily,
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
LIVE_COLS = ["Ticker","Price","Score","Entry_Price","Stop_Loss","Target",
             "Risk_Reward","Days_Since_Signal"]
_CW = {"Ticker":8,"Price":10,"Score":7,"Entry_Price":12,"Stop_Loss":11,
       "Target":10,"Risk_Reward":12,"Days_Since_Signal":16}
_CF = {"Price":"${:.2f}","Score":"{:.0f}","Entry_Price":"${:.2f}",
       "Stop_Loss":"${:.2f}","Target":"${:.2f}","Risk_Reward":"{:.2f}"}
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
        r = analyze_rsi_multi_tf_reversal(sym, daily_map[sym])
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
    print("   monthly_rsi_cross_level   60 → 55")
    print("   weekly_rsi_level          60 → 55")
    print("   daily_rsi_cross_level     40 → 45")
    print("   swing_arm                  5 → 3")
    print("   swing_search_window      252 → 400")
    print("   min_price                  2 → 1")
    print("   min_avg_volume         80000 → 50000")

results.sort(key=lambda x: x["Score"], reverse=True)

# ── Always build df_out and save/email (even if 0 results) ────
ts      = datetime.today().strftime("%Y%m%d_%H%M")
out_dir = os.environ.get("GITHUB_WORKSPACE", os.getcwd())

COLS = [
    "Ticker","Price","Score",
    "Entry_Price","Stop_Loss","Target","Risk_%","Risk_Reward",
    "Monthly_RSI","Weekly_RSI","Daily_RSI","Daily_RSI_Prev",
    "Days_Since_Signal","Recent_Signal_Count","Recent_Signals",
    "Swing_High_Bars_Ago","Flags",
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
    "Target"      : lambda v: f"${v:.2f}",
    "Risk_%"      : lambda v: f"{v:.1f}%",
    "Risk_Reward" : lambda v: f"{v:.2f}",
    "Monthly_RSI" : lambda v: f"{v:.1f}",
    "Weekly_RSI"  : lambda v: f"{v:.1f}",
    "Daily_RSI"   : lambda v: f"{v:.1f}",
    "Daily_RSI_Prev": lambda v: f"{v:.1f}",
    "Days_Since_Signal": lambda v: f"{int(v)}d ago",
    "Swing_High_Bars_Ago": lambda v: f"{int(v)}d ago",
}

def fmt_v(col, val):
    if val is None or (isinstance(val, float) and np.isnan(val)): return "—"
    try:
        if col in FMT: return FMT[col](val)
    except Exception: pass
    return str(val) if str(val) not in ("nan","None","") else "—"

if _IN_NOTEBOOK and results:
    DISP = ["Ticker","Price","Score","Entry_Price","Stop_Loss","Target",
            "Risk_Reward","Days_Since_Signal","Monthly_RSI","Weekly_RSI","Daily_RSI"]
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
            elif col == "Risk_Reward":
                try:
                    v = float(raw)
                    clr = "#22c55e" if v >= 2 else "#f59e0b"
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
    <span style="color:#f1f5f9;font-size:15px;font-weight:700">RSI Multi-Timeframe Reversal</span>
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
    📈 RSI Multi-Timeframe Reversal (Monthly + Weekly + Daily)
  </h2>
  <p style="margin:6px 0 0;color:#94a3b8;font-size:12px">
    {datetime.today().strftime('%Y-%m-%d %H:%M')} &nbsp;·&nbsp;
    <b style="color:#22c55e">{len(results)} matches</b> from {len(TICKERS)} tickers
    &nbsp;·&nbsp; daily cross checked over the last {CFG['daily_cross_lookback_days']} trading days
  </p>
</div>"""

    legend_html = f"""
<div style="background:#f8fafc;border:1px solid #e2e8f0;border-radius:8px;
        padding:12px 18px;margin-top:6px;font-size:11px;color:#64748b;
        font-family:'Segoe UI',Arial,sans-serif">
  <b style="color:#475569">GUIDE</b> &nbsp;·&nbsp;
  Monthly RSI just crossed above 60 (current month) &nbsp;·&nbsp;
  Weekly RSI currently above 60 (current week) &nbsp;·&nbsp;
  Daily RSI crossed above 40 with a GREEN candle that day, any day in
  the last {CFG['daily_cross_lookback_days']} trading days &nbsp;·&nbsp;
  Entry_Price/Stop_Loss = that green candle's high/low (may be a few
  days old — see Days_Since_Signal) &nbsp;·&nbsp;
  Target = last confirmed swing high above entry
</div>"""

    display_html(header_html + table_html + legend_html)

elif results:
    # ASCII table (CLI/GitHub Actions mode)
    CLI_COLS = ["Ticker","Price","Score","Entry_Price","Stop_Loss","Target",
                "Risk_Reward","Days_Since_Signal","Monthly_RSI","Weekly_RSI","Daily_RSI"]
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
  Target          last confirmed swing high above the entry
  Risk_Reward     (Target - Entry) / (Entry - Stop_Loss)
  Days_Since_Signal  how many trading days ago the daily RSI cross fired
                     (0 = today; checked over the last daily_cross_
                     lookback_days trading days, not just today)
  ──────────────────────────────────────────────────────""")

# Save
fpath = os.path.join(out_dir, f"rsi_multi_tf_reversal_{ts}.csv")
df_out.to_csv(fpath, index=False)
print(f"\n  💾 CSV → {fpath}")
tv = os.path.join(out_dir, f"tv_rsi_multi_tf_reversal_{ts}.txt")
with open(tv,"w") as f:
    f.write(f"###RSI Multi-Timeframe Reversal {datetime.today().strftime('%Y-%m-%d')}\n")
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
                      "Target","Risk_Reward","Days_Since_Signal"]
        )
        rows_e = ""
        for i, r in enumerate(rl[:50]):
            bg  = "#fff" if i % 2 == 0 else "#f0f9ff"
            ticker = r.get("Ticker","—")
            price  = r.get("Price",0) or 0
            score  = r.get("Score",0) or 0
            entry  = r.get("Entry_Price",0) or 0
            stop   = r.get("Stop_Loss",0) or 0
            target = r.get("Target",0) or 0
            rr     = r.get("Risk_Reward",0) or 0
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
  📊 RSI Multi-Timeframe Reversal
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
            f"RSI Multi-Timeframe Reversal (daily cross checked over last {CFG['daily_cross_lookback_days']} trading days) — {datetime.today().strftime('%Y-%m-%d')}",
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
                target = r.get("Target",0) or 0
                rr     = r.get("Risk_Reward",0) or 0
                dsig   = r.get("Days_Since_Signal")
                dsig_disp = f"{int(dsig)}d ago" if dsig is not None else "—"
                plain_lines.append(
                    f"{ticker:<7} ${float(price):.2f}  Score:{float(score):.0f}  "
                    f"Entry:${float(entry):.2f}  SL:${float(stop):.2f}  "
                    f"Target:${float(target):.2f}  R:R:{float(rr):.2f}  Signal:{dsig_disp}"
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
        ax.axhline(r["Target"], color="#3b82f6", lw=1.0, ls="--", alpha=0.85,
                  label=f"Target ${r['Target']:.2f}")
        ax.set_title(
            f"{r['Ticker']}  |  ${r['Price']:.2f}  |  Score {r['Score']}  |  "
            f"Entry ${r['Entry_Price']:.2f}  R:R {r['Risk_Reward']:.2f}",
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
        f"RSI Multi-Timeframe Reversal  ·  {datetime.today().strftime('%Y-%m-%d')}",
        color="#60a5fa", fontsize=12, fontweight="bold", y=1.001)
    plt.tight_layout()
    cp = os.path.join(out_dir, f"rsi_multi_tf_reversal_chart_{ts}.png")
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
  📋 SIGNAL (all required)
  1) MONTHLY: RSI(14) just crossed ABOVE 60 (below 60 last month,
     at/above 60 this month) — a fresh long-term momentum shift.
     Evaluated at the CURRENT month only (not a rolling window).
  2) WEEKLY: RSI(14) currently ABOVE 60 (just the level, no
     freshness requirement). Evaluated at the CURRENT week only.
  3) DAILY: RSI(14) crossed ABOVE 40 from below, with that day's
     candle GREEN — checked over the last
     daily_cross_lookback_days trading days ({CFG['daily_cross_lookback_days']}d,
     ~1 month), not just today. If it fired more than once in that
     window, the MOST RECENT occurrence is used for Entry/Stop/Target.

  📋 DATA SOURCING
  Only 1 download per ticker: daily bars (~6 years). Weekly and
  Monthly RSI are both computed from the SAME daily data via
  resampling ('W' and 'ME') — no separate network calls needed.

  📋 OUTPUT
  Entry_Price = HIGH of the signal day's green candle (the
                breakout trigger — enter once price clears it).
                NOTE: if Days_Since_Signal > 0, this is the price
                as of that earlier day, not today's live price.
  Stop_Loss   = LOW of that same green candle
  Target      = the last CONFIRMED swing high before the signal
                (a fractal high with swing_arm bars of lower highs
                on both sides) — only valid if it sits above entry
  Risk_Reward = (Target - Entry) / (Entry - Stop_Loss)
  Days_Since_Signal = how many trading days ago the daily cross fired
  Recent_Signals    = every date the daily cross fired within the
                      lookback window (there may be more than one)

  📋 SCORE (0-100)
  Monthly RSI strength (0-25) + Weekly RSI strength (0-25) +
  Daily RSI cross strength (0-20) + Risk:Reward bonus (0-30)

  💡 BEST SETUPS
  Score > 70            strong RSI alignment + good risk:reward
  Risk_Reward > 2         target well above entry relative to stop
  Days_Since_Signal = 0-3   freshest daily reversal
  Swing_High_Bars_Ago high  target is a well-established prior
                            high, not just recent noise

  ⚙️  TUNE IF 0 RESULTS
  monthly_rsi_cross_level    60 → 55   (looser monthly trigger)
  weekly_rsi_level           60 → 55
  daily_rsi_cross_level      40 → 45
  daily_cross_lookback_days  21 → 42   (search further back, ~2 months)
  swing_arm                   5 → 3    (less strict swing confirmation)
  swing_search_window       252 → 400  (search further back for a target)
  min_price                    2 → 1
  min_avg_volume           80000 → 50000
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
""")
