# ============================================================
# NASDAQ — SMA50/SMA150/EMA20 Retest + EMA20 Cross
# ============================================================
#
# EXACT 3-PHASE PATTERN:
#
#  PHASE 1 — BULL STRUCTURE  (prerequisite)
#      Price is above BOTH SMA50 and SMA150
#      SMA50 > SMA150  (uptrend confirmed)
#      EMA20 > SMA50 > SMA150  (full bull stack preferred)
#
#  PHASE 2 — RETEST OF SMA50, SMA150, OR EMA20  (the pullback)
#      In the last retest_lookback bars, price came down
#      and TOUCHED one or more of: SMA50, SMA150, EMA20
#        Touch = candle LOW came within retest_touch_pct% of the level
#      Price did NOT close below SMA150 by more than
#        max_close_below_pct% (support held)
#      This is the "retest" — buyers defended the key level
#
#  PHASE 3 — CROSSED ABOVE EMA20  (the entry signal)
#      EXACT 1-bar cross (most recent bar):
#        Bar[-2] close < EMA20[-2]   ← was below EMA20
#        Bar[-1] close >= EMA20[-1]  ← crossed above today
#      The cross happened AFTER the retest low
#      = price bounced off the retest level and is now
#        reclaiming EMA20 — the entry trigger
#
# LOGIC FLOW:
#   Strong uptrend → Pullback to SMA50/SMA150/EMA20 → Hold support
#   → Bounce begins → Crosses above EMA20 → BUY
#
# WHICH LEVEL(S) WERE RETESTED:
#   EMA20         → shallowest pullback, fastest continuation setups
#   SMA50         → tighter pullback, usually higher quality setups
#   SMA150        → deeper pullback, bigger potential recovery move
#   Multiple      → very strong support confluence (highest priority)
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
import matplotlib.patches as mpatches

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
    "history_days"              : 300,

    # ── MA periods ────────────────────────────────────────────
    "ema20_period"              : 20,
    "sma50_period"              : 50,
    "sma150_period"             : 150,

    # ── Phase 1: Bull structure ───────────────────────────────
    # SMA50 must be above SMA150
    "require_sma50_above_sma150": True,
    # Price must be above SMA150 currently
    "require_price_above_sma150": True,

    # ── Phase 2: Retest ───────────────────────────────────────
    # Look for retest in last N bars
    "retest_lookback"           : 30,
    # Candle LOW must have come within X% of SMA50 or SMA150
    "retest_touch_pct"          : 3.0,
    # Price must NOT have closed below SMA150 by more than X%
    "max_close_below_sma150_pct": 1.5,
    # Minimum bars from retest low to the EMA20 cross
    # (ensures we waited for confirmation)
    "min_bars_after_retest"     : 0,

    # ── Phase 3: EMA20 cross ──────────────────────────────────
    # Exact 1-bar cross within last cross_lookback bars
    "cross_lookback"            : 5,
    # Cross must have happened AFTER the retest
    # (enforced by timeline check)

    # ── Volume ────────────────────────────────────────────────
    "vol_avg_bars"              : 20,
    # Volume on cross bar >= mult x avg
    "cross_vol_mult"            : 0.8,

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
    delta = close.diff()
    gain  = delta.clip(lower=0)
    loss  = -delta.clip(upper=0)
    avg_g = gain.ewm(alpha=1/period, adjust=False).mean()
    avg_l = loss.ewm(alpha=1/period, adjust=False).mean()
    rs    = avg_g / avg_l.replace(0, np.nan)
    return 100 - (100 / (1 + rs))

def calc_macd(close, fast=12, slow=26, signal=9):
    ema_f = calc_ema(close, fast)
    ema_s = calc_ema(close, slow)
    macd  = ema_f - ema_s
    sig   = calc_ema(macd, signal)
    return macd, sig, macd - sig

# ── Download ──────────────────────────────────────────────────
def _clean(df, min_bars=60):
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
    P1: Bull structure  — SMA50 > SMA150, price above SMA150
    P2: Retest          — price touched SMA50 or SMA150 in last N bars
    P3: EMA20 cross     — exact 1-bar: prev below, today above EMA20
                          cross happened AFTER the retest low
    """
    df      = df.copy(); df.index = pd.to_datetime(df.index)
    n       = len(df)
    price   = float(df["Close"].iloc[-1])
    avg_vol = float(df["Volume"].tail(CFG["vol_avg_bars"]).mean())

    if price   < CFG["min_price"]:      return None
    if avg_vol < CFG["min_avg_volume"]: return None
    if n < 3: return None

    # ── Compute MAs ───────────────────────────────────────────
    ema20_s  = calc_ema(df["Close"], CFG["ema20_period"])
    sma50_s  = df["Close"].rolling(CFG["sma50_period"]).mean()
    sma150_s = df["Close"].rolling(CFG["sma150_period"]).mean()
    rsi_s    = calc_rsi(df["Close"])
    macd_s, _, hist_s = calc_macd(df["Close"])

    cur_ema20  = float(ema20_s.iloc[-1])
    cur_sma50  = float(sma50_s.iloc[-1])
    cur_sma150 = float(sma150_s.iloc[-1])
    cur_rsi    = float(rsi_s.iloc[-1]) if not np.isnan(rsi_s.iloc[-1]) else 50
    cur_macd   = float(macd_s.iloc[-1]) if not np.isnan(macd_s.iloc[-1]) else 0
    cur_hist   = float(hist_s.iloc[-1]) if not np.isnan(hist_s.iloc[-1]) else 0

    if any(np.isnan([cur_ema20, cur_sma50, cur_sma150])): return None
    if cur_rsi < CFG["rsi_min"]: return None

    # ─────────────────────────────────────────────────────────
    # PHASE 1: BULL STRUCTURE
    # ─────────────────────────────────────────────────────────
    if CFG["require_sma50_above_sma150"] and cur_sma50 <= cur_sma150:
        return None

    if CFG["require_price_above_sma150"] and price < cur_sma150:
        return None

    # ─────────────────────────────────────────────────────────
    # PHASE 3: EMA20 EXACT 1-BAR CROSS  (check this first —
    # it's a hard gate before we do the slower retest search)
    # ─────────────────────────────────────────────────────────
    cl = CFG["cross_lookback"]
    ema20_cross_bar  = None
    ema20_cross_date = None

    for i in range(max(1, n - cl), n):
        pc  = float(df["Close"].iloc[i-1])
        cc  = float(df["Close"].iloc[i])
        pe  = float(ema20_s.iloc[i-1]) if not np.isnan(ema20_s.iloc[i-1]) else np.nan
        ce  = float(ema20_s.iloc[i])   if not np.isnan(ema20_s.iloc[i])   else np.nan
        if np.isnan(pe) or np.isnan(ce): continue
        # Exact 1-bar cross
        if pc < pe and cc >= ce:
            ema20_cross_bar  = i
            ema20_cross_date = df.index[i]

    if ema20_cross_bar is None: return None   # no EMA20 cross found

    # Volume on cross bar
    cross_vol  = float(df["Volume"].iloc[ema20_cross_bar])
    cross_vm   = cross_vol / avg_vol if avg_vol > 0 else 0
    if cross_vm < CFG["cross_vol_mult"]: return None

    # ─────────────────────────────────────────────────────────
    # PHASE 2: RETEST OF SMA50 OR SMA150
    # Search in the retest_lookback bars BEFORE the EMA20 cross
    # The retest low must have occurred before the cross
    # ─────────────────────────────────────────────────────────
    rt_lb = CFG["retest_lookback"]
    touch_pct = CFG["retest_touch_pct"] / 100

    # Search window: from rt_lb bars before cross, up to (cross - min_bars_after)
    rt_search_start = max(0, ema20_cross_bar - rt_lb)
    rt_search_end   = max(0, ema20_cross_bar - CFG["min_bars_after_retest"])

    retest_sma50  = False
    retest_sma150 = False
    retest_ema20  = False
    retest_bar    = None          # bar where the deepest retest occurred
    retest_price  = float("inf") # lowest close during retest window
    retest_which  = []

    for i in range(rt_search_start, rt_search_end + 1):
        lo   = float(df["Low"].iloc[i])
        cl_i = float(df["Close"].iloc[i])
        s50  = float(sma50_s.iloc[i])  if not np.isnan(sma50_s.iloc[i])  else np.nan
        s150 = float(sma150_s.iloc[i]) if not np.isnan(sma150_s.iloc[i]) else np.nan
        e20  = float(ema20_s.iloc[i])  if not np.isnan(ema20_s.iloc[i])  else np.nan

        if np.isnan(s50) or np.isnan(s150): continue

        # Check if candle LOW touched SMA50 (within touch_pct)
        if s50 > 0 and abs(lo - s50) / s50 <= touch_pct:
            retest_sma50 = True

        # Check if candle LOW touched SMA150 (within touch_pct)
        if s150 > 0 and abs(lo - s150) / s150 <= touch_pct:
            retest_sma150 = True

        # Check if candle LOW touched EMA20 (within touch_pct)
        if not np.isnan(e20) and e20 > 0 and abs(lo - e20) / e20 <= touch_pct:
            retest_ema20 = True

        # Track the deepest close for retest bar
        if cl_i < retest_price:
            retest_price = cl_i
            retest_bar   = i

    # Must have retested at least one of SMA50, SMA150, or EMA20
    if not retest_sma50 and not retest_sma150 and not retest_ema20: return None

    # Determine which level(s) were retested — build label from all matches
    if retest_sma50:  retest_which.append("SMA50")
    if retest_sma150: retest_which.append("SMA150")
    if retest_ema20:  retest_which.append("EMA20")
    retest_label = " + ".join(retest_which)

    # ── Validate: price did not close significantly below SMA150 ─
    max_close_below_s150 = 0.0
    for i in range(rt_search_start, ema20_cross_bar + 1):
        cl_i = float(df["Close"].iloc[i])
        s150 = float(sma150_s.iloc[i]) if not np.isnan(sma150_s.iloc[i]) else cur_sma150
        pct_below = (s150 - cl_i) / s150 * 100 if s150 > 0 else 0
        if pct_below > max_close_below_s150:
            max_close_below_s150 = pct_below

    if max_close_below_s150 > CFG["max_close_below_sma150_pct"]: return None

    # ── Retest must be BEFORE the EMA20 cross ─────────────────
    if retest_bar is not None and retest_bar >= ema20_cross_bar:
        return None   # retest happened after cross — wrong order

    # ── Metrics ───────────────────────────────────────────────
    bars_since_cross  = n - 1 - ema20_cross_bar
    bars_from_retest  = ema20_cross_bar - retest_bar if retest_bar is not None else 0
    dist_ema20_pct    = (price - cur_ema20) / cur_ema20 * 100
    dist_sma50_pct    = (price - cur_sma50) / cur_sma50 * 100
    dist_sma150_pct   = (price - cur_sma150) / cur_sma150 * 100
    sma50_vs_sma150   = (cur_sma50 - cur_sma150) / cur_sma150 * 100

    # Retest depth: how close the low got to the retested level(s)
    # Use the TIGHTEST (smallest) depth among all levels touched —
    # that represents the most precise support/resistance test
    if retest_bar is not None:
        rt_lo  = float(df["Low"].iloc[retest_bar])
        s50_rt = float(sma50_s.iloc[retest_bar])  if not np.isnan(sma50_s.iloc[retest_bar])  else cur_sma50
        s150_rt= float(sma150_s.iloc[retest_bar]) if not np.isnan(sma150_s.iloc[retest_bar]) else cur_sma150
        e20_rt = float(ema20_s.iloc[retest_bar])  if not np.isnan(ema20_s.iloc[retest_bar])  else cur_ema20

        depths = []
        if "SMA50"  in retest_which: depths.append(abs(rt_lo - s50_rt)  / s50_rt  * 100)
        if "SMA150" in retest_which: depths.append(abs(rt_lo - s150_rt) / s150_rt * 100)
        if "EMA20"  in retest_which: depths.append(abs(rt_lo - e20_rt)  / e20_rt  * 100)
        retest_depth_pct = min(depths) if depths else 0.0
    else:
        retest_depth_pct = 0.0

    # ── Score (0-100) ─────────────────────────────────────────
    score = 0

    # Retest quality (0-30): tighter retest = better
    score += max(0, 30 - int(retest_depth_pct * 5))

    # Multiple levels retested = bonus (scales with count)
    levels_touched = len(retest_which)
    if levels_touched >= 3:
        score += 20   # SMA50 + SMA150 + EMA20 — maximum confluence
    elif levels_touched == 2:
        score += 12
    else:
        score += 0

    # EMA20 cross freshness (0-20): today = 20, yesterday = 15...
    score += max(0, 20 - bars_since_cross * 5)

    # Bars from retest to cross (0-15): faster = better
    if bars_from_retest <= 3:
        score += 15
    elif bars_from_retest <= 7:
        score += 10
    elif bars_from_retest <= 15:
        score += 5

    # Volume on cross (0-10)
    score += min(10, int(cross_vm * 4))

    # MACD turning up (0-5)
    score += 5 if cur_hist > 0 else 0

    # RSI health (0-5)
    score += min(5, max(0, int((cur_rsi - 40) / 4)))

    score = min(100, max(0, score))

    return {
        "Ticker"              : sym,
        "Price"               : round(price, 2),
        "Score"               : score,
        # Retest info
        "Retest_Level"        : retest_label,
        "Retest_Count"        : levels_touched,
        "Retest_Both"         : "✅" if levels_touched >= 2 else "—",
        "Retest_SMA50"        : "✅" if retest_sma50  else "—",
        "Retest_SMA150"       : "✅" if retest_sma150 else "—",
        "Retest_EMA20"        : "✅" if retest_ema20  else "—",
        "Retest_Bar_Date"     : df.index[retest_bar].strftime("%Y-%m-%d") if retest_bar is not None else "—",
        "Retest_Depth_%"      : round(retest_depth_pct, 2),
        "Bars_Retest_to_Cross": bars_from_retest,
        "Max_Below_SMA150_%"  : round(max_close_below_s150, 2),
        # EMA20 cross
        "EMA20_Cross_Date"    : ema20_cross_date.strftime("%Y-%m-%d"),
        "Bars_Since_Cross"    : bars_since_cross,
        "Cross_Vol_x"         : round(cross_vm, 2),
        # MA levels
        "EMA20"               : round(cur_ema20, 2),
        "SMA50"               : round(cur_sma50, 2),
        "SMA150"              : round(cur_sma150, 2),
        "Dist_EMA20_%"        : round(dist_ema20_pct, 2),
        "Dist_SMA50_%"        : round(dist_sma50_pct, 2),
        "Dist_SMA150_%"       : round(dist_sma150_pct, 2),
        "SMA50_vs_SMA150_%"   : round(sma50_vs_sma150, 2),
        # Indicators
        "RSI"                 : round(cur_rsi, 1),
        "MACD_Hist"           : round(cur_hist, 4),
        "Avg_Vol_20d"         : int(avg_vol),
        # internals
        "_df"                 : df,
        "_ema20"              : ema20_s,
        "_sma50"              : sma50_s,
        "_sma150"             : sma150_s,
        "_cross_bar"          : ema20_cross_bar,
        "_retest_bar"         : retest_bar,
    }

# ── Live print ────────────────────────────────────────────────
LIVE_COLS = [
    "Ticker","Price","Score",
    "Retest_Level","Retest_Count","Retest_Bar_Date","Retest_Depth_%",
    "Bars_Retest_to_Cross",
    "EMA20_Cross_Date","Bars_Since_Cross","Cross_Vol_x",
    "EMA20","SMA50","SMA150","Dist_EMA20_%","RSI",
]
_CW = {
    "Ticker":8,"Price":10,"Score":7,
    "Retest_Level":22,"Retest_Count":13,"Retest_Bar_Date":15,"Retest_Depth_%":13,
    "Bars_Retest_to_Cross":20,
    "EMA20_Cross_Date":16,"Bars_Since_Cross":16,"Cross_Vol_x":12,
    "EMA20":9,"SMA50":9,"SMA150":9,"Dist_EMA20_%":12,"RSI":6,
}
_CF = {
    "Price":"${:.2f}","Score":"{:.0f}",
    "Retest_Count":"{:.0f}",
  