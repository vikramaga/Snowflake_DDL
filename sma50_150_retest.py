# ============================================================
# NASDAQ — SMA50/SMA150 Retest + EMA20 Cross Above
# ============================================================
#
# PATTERN:
#
#  PHASE 1 — BULL STRUCTURE
#    Price above SMA150 (long-term uptrend intact)
#    SMA50 > SMA150 (medium-term above long-term)
#
#  PHASE 2 — RETEST OF SMA50 OR SMA150
#    In last retest_lookback bars, price came down and
#    TOUCHED or came within retest_zone_pct% of:
#      - SMA50 (50-day simple moving average)  OR
#      - SMA150 (150-day simple moving average)
#    = Price pulled back to a key support level
#    = Institutions buying at these levels
#
#  PHASE 3 — EMA20 EXACT 1-BAR CROSS (entry trigger)
#    Bar[-2] close < EMA20[-2]  ← previous bar BELOW EMA20
#    Bar[-1] close >= EMA20[-1] ← today CLOSED ABOVE EMA20
#    = Price just broke back above the short-term trend line
#    = Earliest momentum signal the retest is over
#
#  LOGIC:
#    SMA50/SMA150 retest = higher-timeframe support confirmed
#    EMA20 cross = short-term trigger to enter
#    Together = low-risk entry at a key support with momentum
#
# SCORING (0-100):
#    EMA20 cross freshness    0-30  (today = 30)
#    Retest tightness         0-25  (closer to SMA = better)
#    Retest type              0-15  (SMA150 > SMA50 = stronger)
#    MA stack strength        0-15  (SMA50 vs SMA150 gap)
#    Volume on cross          0-10  (conviction)
#    RSI momentum             0-5   (health check)
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

import yfinance as yf
import pandas as pd
import numpy as np
import requests, time, warnings, io
from datetime import datetime, timedelta
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

warnings.filterwarnings("ignore")
pd.set_option("display.max_rows", 200)
env = "Colab/Jupyter" if _IN_NOTEBOOK else "Script/CI"
print(f"✅  yfinance {yf.__version__}  |  numpy {np.__version__}  |  [{env}]")

# ── CONFIG ────────────────────────────────────────────────────
CFG = {
    "history_days"              : 400,

    # ── MA periods ────────────────────────────────────────────
    "ema20_period"              : 20,
    "sma50_period"              : 50,
    "sma150_period"             : 150,

    # ── Phase 1: Bull structure ───────────────────────────────
    # Price must be above SMA150
    "require_above_sma150"      : True,
    # SMA50 must be above SMA150
    "require_sma50_above_sma150": True,

    # ── Phase 2: Retest of SMA50 or SMA150 ───────────────────
    # Look back this many bars for the retest
    "retest_lookback"           : 30,
    # Candle LOW must have come within this % of the SMA level
    # (catches cases where the wick touched but close didn't)
    "retest_zone_pct"           : 3.0,
    # Price must NOT have closed below SMA150 during the retest
    # (that would be a breakdown, not a retest)
    "max_below_sma150_pct"      : 1.0,

    # ── Phase 3: EMA20 exact 1-bar cross ─────────────────────
    # Bar[-2] close < EMA20[-2]   (was below EMA20 yesterday)
    # Bar[-1] close >= EMA20[-1]  (closed above EMA20 today)
    # Cross must be within last cross_lookback bars
    "cross_lookback"            : 5,

    # ── Volume on cross bar ───────────────────────────────────
    "cross_vol_mult"            : 0.8,  # volume >= 0.8× avg on cross bar
    "vol_avg_bars"              : 20,

    # ── RSI ───────────────────────────────────────────────────
    "rsi_min"                   : 35,

    # ── Filters ───────────────────────────────────────────────
    "min_avg_volume"            : 80_000,
    "min_price"                 : 1.0,

    "batch_size"                : 50,
    "batch_sleep"               : 1.5,
}

# ── Indicators ───────────────────────────────────────────────
def calc_ema(series, period):
    return series.ewm(span=period, adjust=False).mean()

def calc_rsi(close, period=14):
    d  = close.diff()
    g  = d.clip(lower=0); l = -d.clip(upper=0)
    ag = g.ewm(alpha=1/period, adjust=False).mean()
    al = l.ewm(alpha=1/period, adjust=False).mean()
    rs = ag / al.replace(0, np.nan)
    return 100 - (100 / (1 + rs))

# ── Download ──────────────────────────────────────────────────
def _clean(df, min_bars=160):
    if df is None or df.empty: return None
    need = [c for c in ["Open","High","Low","Close","Volume"] if c in df.columns]
    if not all(c in need for c in ["Open","High","Low","Close","Volume"]): return None
    df = df[need].copy()
    df.index = pd.to_datetime(df.index)
    if hasattr(df.index, "tz") and df.index.tz:
        df.index = df.index.tz_localize(None)
    df.dropna(subset=["Open","High","Low","Close","Volume"], inplace=True)
    return df if (len(df) >= min_bars and float(df["Close"].iloc[-1]) > 0) else None

def download(symbols, days):
    end = datetime.today(); start = end - timedelta(days=days)
    out = {}
    try:
        raw = yf.download(symbols,
                          start=start.strftime("%Y-%m-%d"),
                          end=end.strftime("%Y-%m-%d"),
                          group_by="ticker", auto_adjust=True,
                          actions=False, threads=True, progress=False)
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

# ── Core detection ────────────────────────────────────────────
def detect_pattern(sym, df):
    """
    Phase 1: Bull structure — price > SMA150, SMA50 > SMA150
    Phase 2: Retest — price touched SMA50 or SMA150 within retest_lookback
    Phase 3: EMA20 exact 1-bar cross — prev below, today above
    """
    df       = df.copy(); df.index = pd.to_datetime(df.index)
    n        = len(df)
    price    = float(df["Close"].iloc[-1])
    avg_vol  = float(df["Volume"].tail(CFG["vol_avg_bars"]).mean())

    if price   < CFG["min_price"]:      return None
    if avg_vol < CFG["min_avg_volume"]: return None

    # ── Compute MAs ───────────────────────────────────────────
    ema20_s  = calc_ema(df["Close"], CFG["ema20_period"])
    sma50_s  = df["Close"].rolling(CFG["sma50_period"]).mean()
    sma150_s = df["Close"].rolling(CFG["sma150_period"]).mean()
    rsi_s    = calc_rsi(df["Close"])

    cur_ema20  = float(ema20_s.iloc[-1])
    cur_sma50  = float(sma50_s.iloc[-1])
    cur_sma150 = float(sma150_s.iloc[-1])
    cur_rsi    = float(rsi_s.iloc[-1]) if not np.isnan(rsi_s.iloc[-1]) else 50

    if any(np.isnan([cur_ema20, cur_sma50, cur_sma150])): return None
    if cur_rsi < CFG["rsi_min"]: return None

    # ─────────────────────────────────────────────────────────
    # PHASE 1: Bull structure
    # ─────────────────────────────────────────────────────────
    if CFG["require_above_sma150"] and price < cur_sma150: return None
    if CFG["require_sma50_above_sma150"] and cur_sma50 <= cur_sma150: return None

    sma50_vs_150_pct  = (cur_sma50  - cur_sma150) / cur_sma150 * 100
    price_vs_150_pct  = (price      - cur_sma150) / cur_sma150 * 100
    price_vs_50_pct   = (price      - cur_sma50)  / cur_sma50  * 100

    # ─────────────────────────────────────────────────────────
    # PHASE 3: EMA20 exact 1-bar cross (check FIRST — fast gate)
    # Bar[-2] close < EMA20[-2]   (was below EMA20)
    # Bar[-1] close >= EMA20[-1]  (crossed above today)
    # ─────────────────────────────────────────────────────────
    cl = CFG["cross_lookback"]
    ema_cross_bar  = None
    ema_cross_date = None

    for i in range(max(1, n - cl), n):
        pc  = float(df["Close"].iloc[i-1])
        cc  = float(df["Close"].iloc[i])
        pe  = float(ema20_s.iloc[i-1]) if not np.isnan(ema20_s.iloc[i-1]) else np.nan
        ce  = float(ema20_s.iloc[i])   if not np.isnan(ema20_s.iloc[i])   else np.nan
        if np.isnan(pe) or np.isnan(ce): continue
        # Exact 1-bar: was below, now above
        if pc < pe and cc >= ce:
            ema_cross_bar  = i
            ema_cross_date = df.index[i]

    if ema_cross_bar is None: return None

    bars_since_cross = n - 1 - ema_cross_bar

    # Volume on cross bar
    cross_vol = float(df["Volume"].iloc[ema_cross_bar])
    cross_vm  = cross_vol / avg_vol if avg_vol > 0 else 0
    if cross_vm < CFG["cross_vol_mult"]: return None

    # Cross bar close (for reference)
    cross_close = float(df["Close"].iloc[ema_cross_bar])
    cross_ema20 = float(ema20_s.iloc[ema_cross_bar])
    cross_dist_pct = (cross_close - cross_ema20) / cross_ema20 * 100

    # Previous close (bar before the cross) was below EMA20
    prev_close = float(df["Close"].iloc[ema_cross_bar - 1])
    prev_ema20 = float(ema20_s.iloc[ema_cross_bar - 1])

    # ─────────────────────────────────────────────────────────
    # PHASE 2: Retest of SMA50 or SMA150
    # Search for a retest in the window BEFORE the EMA20 cross
    # Price LOW must have touched within retest_zone_pct% of
    # SMA50 or SMA150
    # ─────────────────────────────────────────────────────────
    lb           = CFG["retest_lookback"]
    zone         = CFG["retest_zone_pct"] / 100
    max_below    = CFG["max_below_sma150_pct"] / 100

    # Search window: from (cross_bar - lookback) to cross_bar
    search_start = max(0, ema_cross_bar - lb)
    search_end   = ema_cross_bar   # exclusive — only bars BEFORE cross

    retest_sma50  = False
    retest_sma150 = False
    retest_low_val   = float("inf")
    retest_low_bar   = None
    retest_dist_50   = float("inf")
    retest_dist_150  = float("inf")

    for i in range(search_start, search_end):
        lo_i   = float(df["Low"].iloc[i])
        cl_i   = float(df["Close"].iloc[i])
        s50_i  = float(sma50_s.iloc[i])  if not np.isnan(sma50_s.iloc[i])  else cur_sma50
        s150_i = float(sma150_s.iloc[i]) if not np.isnan(sma150_s.iloc[i]) else cur_sma150

        # Check SMA50 retest (low came within zone% of SMA50)
        dist50_lo = abs(lo_i - s50_i) / s50_i
        if dist50_lo <= zone:
            retest_sma50 = True
            if dist50_lo < retest_dist_50:
                retest_dist_50 = dist50_lo
                retest_low_val = lo_i
                retest_low_bar = i

        # Check SMA150 retest (low came within zone% of SMA150)
        dist150_lo = abs(lo_i - s150_i) / s150_i
        if dist150_lo <= zone:
            retest_sma150 = True
            if dist150_lo < retest_dist_150:
                retest_dist_150 = dist150_lo
                if retest_low_bar is None:
                    retest_low_val = lo_i
                    retest_low_bar = i

        # Reject if close went too far BELOW SMA150 (breakdown)
        if cl_i < s150_i * (1 - max_below):
            return None

    # Must have retested at least one of the SMA levels
    if not retest_sma50 and not retest_sma150:
        return None

    # Retest must have happened before the EMA20 cross (confirms sequence)
    if retest_low_bar is None: return None

    # Determine retest type
    if retest_sma150 and retest_sma50:
        retest_type  = "SMA150 + SMA50"
        retest_score = 15
    elif retest_sma150:
        retest_type  = "SMA150"
        retest_score = 15
    else:
        retest_type  = "SMA50"
        retest_score = 10

    # Best retest distance
    best_dist_pct = min(
        retest_dist_50  * 100 if retest_sma50  else float("inf"),
        retest_dist_150 * 100 if retest_sma150 else float("inf"),
    )
    retest_low_date  = df.index[retest_low_bar].strftime("%Y-%m-%d")
    bars_since_retest= ema_cross_bar - retest_low_bar  # bars from low to cross

    # ── Score (0-100) ─────────────────────────────────────────
    score = 0

    # EMA20 cross freshness (0-30): today = 30, yesterday = 25
    score += max(0, 30 - bars_since_cross * 5)

    # Retest tightness (0-25): closer to SMA = better setup
    score += max(0, 25 - int(best_dist_pct * 5))

    # Retest type (0-15)
    score += retest_score

    # MA stack strength (0-15)
    score += min(15, int(sma50_vs_150_pct * 1.5))

    # Volume on cross bar (0-10)
    score += min(10, int(cross_vm * 4))

    # RSI health (0-5)
    score += min(5, max(0, int((cur_rsi - 40) / 4)))

    score = min(100, max(0, score))

    return {
        "Ticker"              : sym,
        "Price"               : round(price, 2),
        "Score"               : score,

        # Phase 3 — EMA20 cross
        "EMA20_Cross_Date"    : ema_cross_date.strftime("%Y-%m-%d"),
        "Bars_Since_Cross"    : bars_since_cross,
        "Cross_Close"         : round(cross_close, 2),
        "Cross_Dist_%"        : round(cross_dist_pct, 2),
        "Prev_Close"          : round(prev_close, 2),
        "Prev_EMA20"          : round(prev_ema20, 2),
        "Cross_Vol_x"         : round(cross_vm, 2),

        # Phase 2 — Retest
        "Retest_Type"         : retest_type,
        "Retest_Low_Date"     : retest_low_date,
        "Retest_Dist_%"       : round(best_dist_pct, 2),
        "Bars_Retest_to_Cross": bars_since_retest,

        # Phase 1 — MAs
        "EMA20"               : round(cur_ema20,  2),
        "SMA50"               : round(cur_sma50,  2),
        "SMA150"              : round(cur_sma150, 2),
        "SMA50_vs_150_%"      : round(sma50_vs_150_pct, 2),
        "Price_vs_SMA50_%"    : round(price_vs_50_pct,  2),
        "Price_vs_SMA150_%"   : round(price_vs_150_pct, 2),

        # Indicators
        "RSI"                 : round(cur_rsi, 1),
        "Avg_Vol_20d"         : int(avg_vol),

        # Internals
        "_df"                 : df,
        "_ema20"              : ema20_s,
        "_sma50"              : sma50_s,
        "_sma150"             : sma150_s,
        "_cross_bar"          : ema_cross_bar,
        "_retest_bar"         : retest_low_bar,
    }

# ── Live print ────────────────────────────────────────────────
LIVE_COLS = [
    "Ticker","Price","Score",
    "Retest_Type","Retest_Low_Date","Retest_Dist_%","Bars_Retest_to_Cross",
    "EMA20_Cross_Date","Bars_Since_Cross","Cross_Dist_%",
    "EMA20","SMA50","SMA150","SMA50_vs_150_%",
    "Cross_Vol_x","RSI",
]
_CW = {
    "Ticker":8,"Price":10,"Score":7,
    "Retest_Type":16,"Retest_Low_Date":15,"Retest_Dist_%":13,
    "Bars_Retest_to_Cross":20,
    "EMA20_Cross_Date":16,"Bars_Since_Cross":16,"Cross_Dist_%":12,
    "EMA20":9,"SMA50":9,"SMA150":9,"SMA50_vs_150_%":14,
    "Cross_Vol_x":12,"RSI":6,
}
_CF = {
    "Price"               : "${:.2f}",
    "Score"               : "{:.0f}",
    "Retest_Dist_%"       : "{:.2f}%",
    "Bars_Retest_to_Cross": "{:.0f}d",
    "Bars_Since_Cross"    : "{:.0f}d",
    "Cross_Dist_%"        : "{:+.2f}%",
    "EMA20"               : "${:.2f}",
    "SMA50"               : "${:.2f}",
    "SMA150"              : "${:.2f}",
    "SMA50_vs_150_%"      : "{:+.2f}%",
    "Cross_Vol_x"         : "{:.2f}×",
    "RSI"                 : "{:.1f}",
}
_hdr_done = False

def _live_header():
    global _hdr_done
    if _hdr_done: return
    sep = "━" * 200
    print(f"\n{sep}")
    print("  📊  LIVE MATCHES  —  SMA50/SMA150 Retest + EMA20 Cross")
    print(sep)
    print("".join(f"  {c:<{_CW.get(c,10)}}" for c in LIVE_COLS))
    print("  " + "─" * 198)
    _hdr_done = True

def live_print(r):
    _live_header()
    row = ""
    for c in LIVE_COLS:
        val=r.get(c,"—"); w=_CW.get(c,10); fmt=_CF.get(c)
        try:   s=fmt.format(val) if (fmt and val not in("—",None)) else str(val)
        except: s=str(val)
        row += f"  {s:<{w}}"
    print(row)

# ── Health check ──────────────────────────────────────────────
print("━"*65)
print("  STEP 1  DATA CHECK")
print("━"*65)
chk = download(["AAPL","NVDA","MSFT"], 300)
if not chk: print("❌  No data.")
else:
    for s, d in chk.items():
        p    = float(d["Close"].iloc[-1])
        e20  = float(calc_ema(d["Close"], 20).iloc[-1])
        s50  = float(d["Close"].rolling(50).mean().iloc[-1])
        s150 = float(d["Close"].rolling(150).mean().iloc[-1])
        stack= "✅" if s50 > s150 and p > s150 else "❌"
        print(f"  ✅ {s}: ${p:.2f}  EMA20=${e20:.2f}  "
              f"SMA50=${s50:.2f}  SMA150=${s150:.2f}  "
              f"Stack:{stack}  {d.index[-1].date()}")
print()

# ── Diagnostic ────────────────────────────────────────────────
print("━"*65)
print("  STEP 2  DIAGNOSTIC (10 sample stocks)")
print("━"*65+"\n")

DIAG = ["AAPL","NVDA","AMD","PLTR","META","CRWD","AVGO","DDOG","MU","MSFT"]
diag_data = download(DIAG, CFG["history_days"])
print(f"  Downloaded {len(diag_data)}/{len(DIAG)}\n")

t = lambda b: "✅" if b else "❌"
print(f"  {'SYM':<7} {'PRICE':>8}  {'STACK':>6}  {'RETEST':>12}  "
      f"{'EMA20_X':>8}  {'SCORE':>6}  RESULT")
print("  " + "─" * 65)

for sym in DIAG:
    if sym not in diag_data:
        print(f"  {sym:<7} — no data"); continue
    try:
        df_d = diag_data[sym]
        p    = float(df_d["Close"].iloc[-1])
        s50  = float(df_d["Close"].rolling(50).mean().iloc[-1])
        s150 = float(df_d["Close"].rolling(150).mean().iloc[-1])
        stack= p > s150 and s50 > s150
        r    = detect_pattern(sym, df_d)
        if r:
            print(f"  {sym:<7} ${p:>7.2f}  {t(stack):>6}  "
                  f"{r['Retest_Type']:>12}  "
                  f"{r['EMA20_Cross_Date']:>8}  "
                  f"{r['Score']:>6}  ✅")
        else:
            print(f"  {sym:<7} ${p:>7.2f}  {t(stack):>6}  "
                  f"{'—':>12}  {'—':>8}  {'—':>6}  ❌")
    except Exception as e:
        print(f"  {sym:<7} error: {e}")

print(f"""
  Pattern:
    Phase 1  Bull structure: price > SMA150, SMA50 > SMA150
    Phase 2  Retest: low touched within {CFG['retest_zone_pct']}% of SMA50 or SMA150
             in last {CFG['retest_lookback']} bars before the EMA20 cross
    Phase 3  EMA20 exact cross: prev bar below EMA20, today above EMA20

  Tune if mostly ❌:
    retest_zone_pct   3.0 → 5.0   (wider retest zone)
    retest_lookback    30 → 45     (look further back)
    cross_lookback      5 → 10
    rsi_min            35 → 25
    max_below_sma150   1.0 → 2.0
""")
print("━"*65+"\n")

# ── Ticker list ───────────────────────────────────────────────
print("━"*65)
print("  STEP 3  FETCH TICKERS")
print("━"*65)

def get_tickers():
    pool = set()
    hdrs = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}
    for url, label in [
        ("https://www.nasdaqtrader.com/dynamic/SymDir/nasdaqlisted.txt","nasdaqlisted"),
        ("https://www.nasdaqtrader.com/dynamic/SymDir/otherlisted.txt", "otherlisted"),
    ]:
        try:
            r  = requests.get(url, headers=hdrs, timeout=20); r.raise_for_status()
            df = pd.read_csv(io.StringIO(r.text), sep="|")
            df.columns = [c.strip() for c in df.columns]
            sc = next((c for c in ["Symbol","ACT Symbol","Nasdaq Symbol"] if c in df.columns),None)
            ec = next((c for c in ["ETF","Is ETF"] if c in df.columns),None)
            if not sc: continue
            b  = len(pool)
            for _, row in df.iterrows():
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
        r = requests.get(
            "https://api.nasdaq.com/api/screener/stocks"
            "?tableonly=true&limit=10000&exchange=nasdaq&download=true",
            headers={**hdrs,"Referer":"https://www.nasdaq.com/"}, timeout=25)
        r.raise_for_status()
        rows = r.json()["data"]["rows"]
        t_   = {row["symbol"].strip() for row in rows
                if row.get("symbol","").strip().isalpha()
                and 1<=len(row["symbol"].strip())<=5}
        b = len(pool); pool |= t_
        print(f"  ✅ {'NASDAQ API':<18}: +{len(pool)-b:>4} → {len(pool)}")
    except Exception as e: print(f"  ⚠️  NASDAQ API: {e}")
    static = {
        "AAPL","MSFT","NVDA","AMZN","META","GOOGL","GOOG","TSLA","AVGO","COST",
        "NFLX","AMD","INTC","CSCO","ADBE","QCOM","TXN","AMAT","MU","KLAC",
        "LRCX","MRVL","MELI","PANW","CRWD","SNPS","CDNS","TEAM","WDAY","PLTR",
        "ALAB","SMCI","HOOD","COIN","SOFI","UPST","DDOG","SNOW","MDB","REGN",
        "VRTX","ISRG","LULU","FTNT","IDXX","SBUX","TMUS","RBRK","NET","MARA",
        "QUBT","RGTI","ASTS","RKLB","IONQ","FSLR","PYPL","ROKU","ROST","POOL",
        "AMGN","GILD","INTU","MCHP","MNST","NXPI","XEL","ACLS","IRTC","IREN",
    }
    b = len(pool); pool |= static
    print(f"  ✅ {'Static fallback':<18}: +{len(pool)-b:>4} → {len(pool)}")
    clean = sorted({s.upper() for s in pool if isinstance(s,str)
                    and s.isalpha() and 1<=len(s)<=5})
    print(f"\n  🎯 Total: {len(clean)} tickers")
    return clean

TICKERS = get_tickers(); print()

# ── Main scan ─────────────────────────────────────────────────
print("━"*65)
print(f"  STEP 4  SCANNING {len(TICKERS)} TICKERS")
print("━"*65+"\n")

_hdr_done = False; results = []; no_data = 0
batches = [TICKERS[i:i+CFG["batch_size"]] for i in range(0,len(TICKERS),CFG["batch_size"])]

with tqdm(total=len(TICKERS), desc="Scanning", unit="stk",
          bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]") as pbar:
    for batch in batches:
        data_map = download(batch, CFG["history_days"])
        no_data += len(batch) - len(data_map)
        for sym in batch:
            pbar.update(1)
            if sym not in data_map: continue
            try:
                r = detect_pattern(sym, data_map[sym])
                if r: results.append(r); live_print(r)
            except Exception: pass
        time.sleep(CFG["batch_sleep"])

got = len(TICKERS) - no_data; pct = got/max(len(TICKERS),1)*100
print(f"\n{'━'*65}")
print(f"  SCAN COMPLETE | {len(TICKERS)} tickers | {got} ({pct:.0f}%) | ✅ {len(results)} matches")
print(f"{'━'*65}")

# ── Results ───────────────────────────────────────────────────
if not results:
    print("\n  No matches. Try relaxing:")
    print("   retest_zone_pct   3.0 → 5.0")
    print("   retest_lookback    30 → 45")
    print("   cross_lookback      5 → 10")
    print("   rsi_min            35 → 25")
    print("   max_below_sma150  1.0 → 2.0")
else:
    results.sort(key=lambda x: x["Score"], reverse=True)

    COLS = [
        "Ticker","Price","Score",
        "Retest_Type","Retest_Low_Date","Retest_Dist_%","Bars_Retest_to_Cross",
        "EMA20_Cross_Date","Bars_Since_Cross","Cross_Close","Cross_Dist_%",
        "Prev_Close","Prev_EMA20","Cross_Vol_x",
        "EMA20","SMA50","SMA150",
        "SMA50_vs_150_%","Price_vs_SMA50_%","Price_vs_SMA150_%",
        "RSI","Avg_Vol_20d",
    ]
    df_out = pd.DataFrame([{k:v for k,v in r.items() if not k.startswith("_")}
                            for r in results])
    df_out = df_out[[c for c in COLS if c in df_out.columns]]
    df_out.reset_index(drop=True, inplace=True)

    ts = datetime.today().strftime("%Y%m%d_%H%M")

    # ── Format helpers ────────────────────────────────────────
    FMT = {
        "Price"               : lambda v: f"${v:.2f}",
        "Score"               : lambda v: f"{v:.0f}",
        "Retest_Dist_%"       : lambda v: f"{v:.2f}%",
        "Bars_Retest_to_Cross": lambda v: f"{int(v)}d",
        "EMA20_Cross_Date"    : lambda v: str(v),
        "Bars_Since_Cross"    : lambda v: f"{int(v)}d",
        "Cross_Close"         : lambda v: f"${v:.2f}",
        "Cross_Dist_%"        : lambda v: f"{v:+.2f}%",
        "Prev_Close"          : lambda v: f"${v:.2f}",
        "Prev_EMA20"          : lambda v: f"${v:.2f}",
        "Cross_Vol_x"         : lambda v: f"{v:.2f}×",
        "EMA20"               : lambda v: f"${v:.2f}",
        "SMA50"               : lambda v: f"${v:.2f}",
        "SMA150"              : lambda v: f"${v:.2f}",
        "SMA50_vs_150_%"      : lambda v: f"{v:+.2f}%",
        "Price_vs_SMA50_%"    : lambda v: f"{v:+.2f}%",
        "Price_vs_SMA150_%"   : lambda v: f"{v:+.2f}%",
        "RSI"                 : lambda v: f"{v:.1f}",
        "Avg_Vol_20d"         : lambda v: f"{v:,.0f}",
    }

    def fmt_v(col, val):
        if val is None or (isinstance(val, float) and np.isnan(val)): return "—"
        try:
            if col in FMT: return FMT[col](val)
        except Exception: pass
        return str(val) if str(val) not in ("nan","None","") else "—"

    # ── Tier grouping by Retest_Type ──────────────────────────
    TIER_ORDER = ["SMA150 + SMA50", "SMA150", "SMA50"]
    TIER_COLORS= {"SMA150 + SMA50": "#22c55e",
                  "SMA150"        : "#3b82f6",
                  "SMA50"         : "#f59e0b"}
    TIER_ICONS = {"SMA150 + SMA50": "🏆",
                  "SMA150"        : "🥈",
                  "SMA50"         : "🥉"}

    if _IN_NOTEBOOK:
        # ── Rich HTML table grouped by retest type ────────────
        DISP_COLS = [
            "Ticker","Price","Score",
            "Retest_Type","Retest_Low_Date","Retest_Dist_%","Bars_Retest_to_Cross",
            "EMA20_Cross_Date","Bars_Since_Cross","Cross_Dist_%",
            "SMA50","SMA150","SMA50_vs_150_%","Cross_Vol_x","RSI",
        ]
        DISP_COLS = [c for c in DISP_COLS if c in df_out.columns]

        def make_tier_block(tier_rows, tier_name, tc, icon):
            if not tier_rows: return ""
            th = "".join(
                f'<th style="background:#0f172a;color:#e2e8f0;padding:9px 12px;'
                f'font-size:11px;font-weight:700;text-align:center;'
                f'border-bottom:2px solid {tc};white-space:nowrap">{c}</th>'
                for c in DISP_COLS
            )
            rows_html = ""
            for i, r in enumerate(tier_rows):
                bg = "#ffffff" if i % 2 == 0 else "#f0f9ff"
                tds = ""
                for col in DISP_COLS:
                    raw  = r.get(col)
                    disp = fmt_v(col, raw)
                    sty  = ""
                    if col == "Score":
                        try:
                            v = float(raw)
                            g = int(min(220, 80 + v * 1.4))
                            sty = (f"background:rgb(20,{g},60);color:#fff;"
                                   f"font-weight:700;text-align:center")
                        except Exception: pass
                    elif col == "Retest_Type":
                        sty = f"color:{tc};font-weight:700"
                    elif col in ("Retest_Dist_%",):
                        try:
                            v = float(raw)
                            if v < 1.0: sty = "color:#22c55e;font-weight:700"
                            elif v < 2.0: sty = "color:#86efac"
                        except Exception: pass
                    elif col == "Bars_Since_Cross":
                        try:
                            v = int(float(raw))
                            if v == 0: sty = "color:#22c55e;font-weight:700;text-align:center"
                            elif v <= 1: sty = "color:#86efac;text-align:center"
                        except Exception: pass
                    elif col == "Cross_Dist_%":
                        try:
                            v = float(raw)
                            sty = ("color:#22c55e;font-weight:600" if v >= 0
                                   else "color:#ef4444;font-weight:600")
                        except Exception: pass
                    elif col == "Cross_Vol_x":
                        try:
                            v = float(str(raw).replace("×",""))
                            if v >= 2: sty = "color:#f59e0b;font-weight:700"
                            elif v >= 1.5: sty = "color:#fbbf24"
                        except Exception: pass
                    tds += (f'<td style="padding:7px 12px;font-size:12px;'
                            f'border-bottom:1px solid #e2e8f0;white-space:nowrap;{sty}">'
                            f'{disp}</td>')
                rows_html += f'<tr style="background:{bg}">{tds}</tr>\n'

            return f"""
<div style="margin:12px 0">
  <div style="background:linear-gradient(90deg,{tc}22,#0f172a);
              border-left:4px solid {tc};border-radius:6px 6px 0 0;
              padding:10px 18px;display:flex;align-items:center;gap:10px">
    <span style="font-size:20px">{icon}</span>
    <div>
      <span style="color:#f1f5f9;font-size:15px;font-weight:700">{tier_name} Retest</span>
      <span style="color:{tc};font-size:12px;margin-left:12px;font-weight:600">
        {len(tier_rows)} stock{'s' if len(tier_rows)!=1 else ''}
      </span>
    </div>
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
    📊 SMA50 / SMA150 Retest + EMA20 Cross Above
  </h2>
  <p style="margin:6px 0 0;color:#94a3b8;font-size:12px">
    {datetime.today().strftime('%Y-%m-%d %H:%M')} &nbsp;·&nbsp;
    <span style="color:#22c55e;font-weight:700">{len(results)} matches</span>
    from {len(TICKERS)} tickers
    &nbsp;·&nbsp;
    🏆 Both: {sum(1 for r in results if r['Retest_Type']=='SMA150 + SMA50')}
    &nbsp;·&nbsp;
    🥈 SMA150: {sum(1 for r in results if r['Retest_Type']=='SMA150')}
    &nbsp;·&nbsp;
    🥉 SMA50: {sum(1 for r in results if r['Retest_Type']=='SMA50')}
  </p>
</div>"""

        tier_blocks = ""
        for t_name in TIER_ORDER:
            t_rows = [r for r in results if r["Retest_Type"] == t_name]
            tier_blocks += make_tier_block(
                t_rows, t_name,
                TIER_COLORS[t_name], TIER_ICONS[t_name])

        legend = """
<div style="background:#f8fafc;border:1px solid #e2e8f0;border-radius:8px;
            padding:12px 18px;margin-top:8px;font-size:11px;color:#64748b">
  <b style="color:#475569">GUIDE</b> &nbsp;·&nbsp;
  Retest_Dist_% = how close the low came to the SMA
  (<span style="color:#22c55e">green &lt;1%</span> = very tight retest) &nbsp;·&nbsp;
  Bars_Since_Cross = 0 = today's cross &nbsp;·&nbsp;
  Bars_Retest_to_Cross = how long from retest low to EMA20 cross &nbsp;·&nbsp;
  🏆 SMA150+SMA50 = strongest (both levels tested)
</div>"""

        display_html(header_html + tier_blocks + legend)

    else:
        # ── ASCII box table for CLI/GitHub ────────────────────
        CLI_COLS = ["Ticker","Price","Score","Retest_Type",
                    "Retest_Low_Date","Retest_Dist_%","Bars_Retest_to_Cross",
                    "EMA20_Cross_Date","Bars_Since_Cross","Cross_Dist_%",
                    "SMA50_vs_150_%","Cross_Vol_x","RSI"]
        CLI_COLS = [c for c in CLI_COLS if c in df_out.columns]

        for t_name in TIER_ORDER:
            t_rows = [r for r in results if r["Retest_Type"] == t_name]
            if not t_rows: continue
            icon = TIER_ICONS[t_name]
            print(f"\n  {icon}  {t_name} Retest  ({len(t_rows)} stocks)\n")

            col_w = {c: max(len(c), max(
                len(fmt_v(c, r.get(c))) for r in t_rows
            )) + 2 for c in CLI_COLS}
            top  = "┬".join("─"*col_w[c] for c in CLI_COLS)
            sep  = "┼".join("─"*col_w[c] for c in CLI_COLS)
            bot  = "┴".join("─"*col_w[c] for c in CLI_COLS)
            hdr  = "│".join(c.center(col_w[c]) for c in CLI_COLS)
            print(f"  ┌{top}┐")
            print(f"  │{hdr}│")
            print(f"  ├{sep}┤")
            for i, r in enumerate(t_rows):
                cells = [fmt_v(c, r.get(c)).center(col_w[c]) for c in CLI_COLS]
                print(f"  │{'│'.join(cells)}│")
                if i < len(t_rows) - 1:
                    print(f"  ├{sep}┤")
            print(f"  └{bot}┘")

        print(f"""
  COLUMN KEY
  ─────────────────────────────────────────────────────────
  Retest_Type        SMA150+SMA50 / SMA150 / SMA50
  Retest_Low_Date    date price touched the SMA support
  Retest_Dist_%      how close the low came to the SMA (lower = tighter)
  Bars_Retest_to_Cross  days from retest low to EMA20 cross
  EMA20_Cross_Date   date of the EMA20 cross
  Bars_Since_Cross   0 = today, 1 = yesterday
  Cross_Dist_%       how far above EMA20 on the cross bar
  SMA50_vs_150_%     how far SMA50 is above SMA150
  ─────────────────────────────────────────────────────────""")

    # ── Save outputs ──────────────────────────────────────────
    out_dir = os.environ.get("GITHUB_WORKSPACE", os.getcwd())
    fpath   = os.path.join(out_dir, f"sma_retest_ema20_cross_{ts}.csv")
    df_out.to_csv(fpath, index=False)
    print(f"\n  💾 CSV → {fpath}")

    tv = os.path.join(out_dir, f"tv_sma_retest_{ts}.txt")
    with open(tv,"w") as f:
        f.write(f"###SMA Retest EMA20 Cross {datetime.today().strftime('%Y-%m-%d')}\n")
        for r in results: f.write(f"NASDAQ:{r['Ticker']}\n")
    print(f"  📋 TradingView → {tv}")

    # ── Email with CSV ────────────────────────────────────────
    def _send_email(rl, csv_path):
        import smtplib
        from email.mime.base import MIMEBase; from email import encoders
        gu = os.environ.get("GMAIL_USER",""); gp = os.environ.get("GMAIL_PASS","")
        et = os.environ.get("EMAIL_TO","")
        if not gu or not gp or not et:
            print("[Email] Skipped — set GMAIL_USER, GMAIL_PASS, EMAIL_TO"); return
        eto  = [e.strip() for e in et.split(",") if e.strip()]
        cnt  = len(rl)
        t1   = sum(1 for r in rl if r["Retest_Type"]=="SMA150 + SMA50")
        t2   = sum(1 for r in rl if r["Retest_Type"]=="SMA150")
        t3   = sum(1 for r in rl if r["Retest_Type"]=="SMA50")

        SHOW = ["Ticker","Price","Score","Retest_Type","Retest_Dist_%",
                "EMA20_Cross_Date","Bars_Since_Cross","Cross_Dist_%",
                "SMA50_vs_150_%","RSI"]
        th_e = "".join(
            f'<th style="background:#1e293b;color:#e2e8f0;padding:8px 11px;'
            f'font-size:11px;font-weight:700;border-bottom:2px solid #3b82f6;'
            f'white-space:nowrap">{c}</th>' for c in SHOW)
        rows_e = ""
        for i, r in enumerate(rl[:50]):
            bg = "#fff" if i%2==0 else "#f0f9ff"
            tds = "".join(
                f'<td style="padding:6px 11px;font-size:11px;'
                f'border-bottom:1px solid #e2e8f0;white-space:nowrap">'
                f'{fmt_v(c, r.get(c))}</td>'
                for c in SHOW)
            rows_e += f'<tr style="background:{bg}">{tds}</tr>\n'
        html_e = (
            f'<html><body style="font-family:Arial;background:#f1f5f9">'
            f'<div style="max-width:900px;margin:20px auto;background:#0f172a;'
            f'border-radius:10px;padding:20px 24px">'
            f'<h2 style="color:#60a5fa;margin:0">📊 SMA Retest + EMA20 Cross</h2>'
            f'<p style="color:#94a3b8;font-size:12px">'
            f'{datetime.today().strftime("%Y-%m-%d")} &nbsp;·&nbsp; {cnt} matches &nbsp;·&nbsp; '
            f'🏆 Both:{t1} &nbsp; 🥈 SMA150:{t2} &nbsp; 🥉 SMA50:{t3}</p></div>'
            f'<div style="max-width:900px;margin:8px auto;overflow-x:auto;'
            f'border-radius:8px;border:1px solid #e2e8f0">'
            f'<table style="border-collapse:collapse;width:100%">'
            f'<thead><tr>{th_e}</tr></thead><tbody>{rows_e}</tbody></table></div>'
            f'<p style="color:#94a3b8;font-size:10px;text-align:center;margin:8px auto;max-width:900px">'
            f'⚠️ Not financial advice · Full results in CSV attachment</p>'
            f'</body></html>')
        plain_e = (f"SMA Retest + EMA20 Cross — {cnt} matches\n"
                   f"Both:{t1}  SMA150:{t2}  SMA50:{t3}\n\n" +
                   "\n".join(f"{r['Ticker']:<7} ${r.get('Price',0):.2f}  "
                              f"Score:{r.get('Score',0):.0f}  "
                              f"{r.get('Retest_Type','—')}  "
                              f"Cross:{r.get('EMA20_Cross_Date','—')}"
                              for r in rl[:50]))
        subj = (f"📊 SMA Retest+EMA20 — {cnt} signals "
                f"(🏆{t1} 🥈{t2} 🥉{t3}) — "
                f"{datetime.today().strftime('%Y-%m-%d')}")
        msg = MIMEMultipart("mixed")
        msg["Subject"]=subj; msg["From"]=gu; msg["To"]=", ".join(eto)
        alt = MIMEMultipart("alternative")
        alt.attach(MIMEText(plain_e,"plain")); alt.attach(MIMEText(html_e,"html"))
        msg.attach(alt)
        if csv_path and os.path.exists(csv_path):
            try:
                with open(csv_path,"rb") as f:
                    p = MIMEBase("application","octet-stream"); p.set_payload(f.read())
                encoders.encode_base64(p)
                p.add_header("Content-Disposition",
                    f"attachment; filename={os.path.basename(csv_path)}")
                msg.attach(p); print("[Email] 📎 CSV attached")
            except Exception as e: print(f"[Email] ⚠️  {e}")
        try:
            with smtplib.SMTP_SSL("smtp.gmail.com",465) as srv:
                srv.login(gu,gp.replace(" ","")); srv.sendmail(gu,eto,msg.as_string())
            print(f"[Email] ✅  Sent to {et}  |  {cnt} matches")
        except smtplib.SMTPAuthenticationError: print("[Email] ❌  Auth failed")
        except Exception as e: print(f"[Email] ❌  {e}")

    from email.mime.text import MIMEText
    from email.mime.multipart import MIMEMultipart
    _send_email(results, fpath)

    if _IN_NOTEBOOK:
        try:
            from google.colab import files
            files.download(fpath); files.download(tv)
        except Exception: pass
    else:
        print("  (CI: files saved to workspace, email sent)")

# ── Charts for top 5 ──────────────────────────────────────────
if results:
    top  = results[:min(5, len(results))]
    rows = len(top)
    fig, axes = plt.subplots(rows, 1, figsize=(15, 5.5*rows), facecolor="#0f172a")
    if rows == 1: axes = [axes]

    for idx, r in enumerate(top):
        ax = axes[idx]
        ax.set_facecolor("#0f172a")

        df_p   = r["_df"].tail(60).copy()
        ema20  = r["_ema20"].reindex(df_p.index)
        sma50  = r["_sma50"].reindex(df_p.index)
        sma150 = r["_sma150"].reindex(df_p.index)
        n_p    = len(df_p)
        fn     = len(r["_df"]); off = fn - n_p

        # Candlestick
        for i, (_, row_) in enumerate(df_p.iterrows()):
            o=float(row_["Open"]); h=float(row_["High"])
            l=float(row_["Low"]);  c=float(row_["Close"])
            clr = "#34d399" if c >= o else "#ef4444"
            ax.plot([i,i],[l,h], color=clr, lw=0.7, zorder=2)
            blo=min(o,c); bhi=max(o,c); bh=max(bhi-blo,(h-l)*0.005)
            rect=mpatches.FancyBboxPatch((i-0.3,blo),0.6,bh,
                 boxstyle="square,pad=0",facecolor=clr,edgecolor=clr,lw=0.4,zorder=3)
            ax.add_patch(rect)

        # MA lines
        ax.plot(range(n_p), ema20.values,
                color="#34d399", lw=1.8, ls="-",  label="EMA20 🟢", zorder=5)
        ax.plot(range(n_p), sma50.values,
                color="#3b82f6", lw=1.5, ls="-",  label="SMA50 🔵", zorder=4)
        ax.plot(range(n_p), sma150.values,
                color="#f472b6", lw=1.5, ls="-.", label="SMA150 🔴", zorder=4)

        # Shade retest zone around the relevant SMA
        rt = r["Retest_Type"]
        if "SMA150" in rt:
            for i in range(n_p):
                sv = float(sma150.iloc[i]) if not np.isnan(sma150.iloc[i]) else None
                if sv:
                    ax.axhspan(sv*0.97, sv*1.03, alpha=0.06, color="#f472b6", zorder=0)
                    break
        if "SMA50" in rt:
            for i in range(n_p):
                sv = float(sma50.iloc[i]) if not np.isnan(sma50.iloc[i]) else None
                if sv:
                    ax.axhspan(sv*0.97, sv*1.03, alpha=0.06, color="#3b82f6", zorder=0)
                    break

        # Mark retest low
        rb = r["_retest_bar"] - off if r["_retest_bar"] is not None else None
        if rb is not None and 0 <= rb < n_p:
            ax.scatter([rb], [float(df_p["Low"].iloc[rb])],
                       color="#fbbf24", s=150, zorder=8, marker="v",
                       label=f"Retest Low ({r['Retest_Type']})")
            ax.axvline(rb, color="#fbbf24", lw=1.0, ls=":", alpha=0.5)

        # Mark EMA20 cross
        cb = r["_cross_bar"] - off
        if 0 <= cb < n_p:
            ax.scatter([cb], [float(df_p["Close"].iloc[cb])],
                       color="#34d399", s=180, zorder=9, marker="★",
                       label=f"EMA20 Cross ({r['EMA20_Cross_Date']})")
            ax.axvline(cb, color="#34d399", lw=1.5, ls="--", alpha=0.8)

        # x-axis
        tick_step = max(1, n_p//8)
        ax.set_xticks(range(0, n_p, tick_step))
        ax.set_xticklabels(
            [df_p.index[i].strftime("%m/%d") for i in range(0, n_p, tick_step)],
            color="#94a3b8", fontsize=7)
        ax.set_xlim(-0.5, n_p - 0.5)

        ax.set_title(
            f"{r['Ticker']}  ${r['Price']:.2f}  |  Score {r['Score']}/100  |  "
            f"{r['Retest_Type']} Retest ({r['Retest_Dist_%']:.2f}% from SMA)  |  "
            f"EMA20 Cross {r['EMA20_Cross_Date']} ({r['Bars_Since_Cross']}d ago)  |  "
            f"Bars Retest→Cross: {r['Bars_Retest_to_Cross']}  |  "
            f"Vol {r['Cross_Vol_x']:.1f}×  RSI {r['RSI']:.0f}",
            color="#e2e8f0", fontsize=8, fontweight="bold", pad=7)

        ax.tick_params(colors="#94a3b8", labelsize=7)
        for sp_ in ax.spines.values(): sp_.set_edgecolor("#1e3a5f")
        ax.legend(loc="upper left", facecolor="#1e293b",
                  labelcolor="#e2e8f0", fontsize=7, framealpha=0.9, ncol=2)
        ax.grid(color="#1e3a5f", ls="--", lw=0.4, alpha=0.4, axis="y")

    plt.suptitle(
        f"SMA50/SMA150 Retest + EMA20 Cross Above  ·  "
        f"{datetime.today().strftime('%Y-%m-%d')}\n"
        f"🔵 SMA50 zone  🔴 SMA150 zone  ▼ Retest Low  ★ EMA20 Cross",
        color="#60a5fa", fontsize=10, fontweight="bold", y=1.001)
    plt.tight_layout()
    cp = os.path.join(
        os.environ.get("GITHUB_WORKSPACE", os.getcwd()),
        f"sma_retest_ema20_chart_{ts}.png")
    plt.savefig(cp, dpi=150, bbox_inches="tight", facecolor="#0f172a")
    if _IN_NOTEBOOK: plt.show()
    else: plt.close()
    print(f"  📊 Chart → {cp}")
    if _IN_NOTEBOOK:
        try:
            from google.colab import files; files.download(cp)
        except Exception: pass

print("""
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  📋 PATTERN EXPLAINED

  PHASE 1 — BULL STRUCTURE
    Price > SMA150  (long-term uptrend intact)
    SMA50 > SMA150  (medium-term above long-term)

  PHASE 2 — SMA RETEST  (key support tested)
    Price pulled back and candle LOW touched within 3%
    of SMA50 (blue) or SMA150 (pink) in last 30 bars
    Price did NOT close below SMA150 (no breakdown)
    = Institutional support confirmed at key levels

  PHASE 3 — EMA20 CROSS  (entry trigger)
    Bar[-2] close was BELOW EMA20 (yesterday below green)
    Bar[-1] close is  ABOVE EMA20 (today crossed above green)
    = Exact 1-bar cross — the earliest momentum signal
    = Pullback ended, price reclaiming short-term trend

  RESULT TIERS
    🏆 SMA150 + SMA50  both levels tested  (strongest)
    🥈 SMA150 only     major support tested (strong)
    🥉 SMA50 only      key support tested   (moderate)

  💡 BEST SETUPS
  Retest_Dist_%  < 1%    very tight retest — strong support
  Bars_Since_Cross = 0   fresh cross today = earliest entry
  Bars_Retest_to_Cross small = fast bounce off support
  Cross_Dist_%  near 0%  barely crossed — tight entry
  SMA50_vs_150_% > 3%    healthy bull structure

  ⚙️  TUNE IF 0 RESULTS
  retest_zone_pct   3.0 → 5.0
  retest_lookback    30 → 45
  cross_lookback      5 → 10
  rsi_min            35 → 25
  max_below_sma150  1.0 → 2.0
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
""")
