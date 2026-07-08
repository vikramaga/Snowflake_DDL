# ============================================================
# NASDAQ — SMA50 Retest + EMA20 + Camarilla S3 Cross
# ============================================================
#
# EXACT 4-CONDITION PATTERN:
#
#  C1 — BULL STRUCTURE  (prerequisite)
#      Price is currently ABOVE SMA150
#      (long-term uptrend intact)
#
#  C2 — RETEST OF SMA50 ONLY  (the pullback — SMA50 specific)
#      In the last retest_lookback bars, price pulled back and
#      candle LOW touched SMA50 (within retest_touch_pct%)
#      Price must NOT have also retested SMA150 in that window —
#      this isolates the shallower SMA50-only pullback pattern
#      Price did NOT close below SMA150 by more than
#        max_close_below_pct% (support held)
#
#  C3 — CURRENTLY ABOVE EMA20
#      Current price > EMA20  (short-term trend reclaimed)
#
#  C4 — JUST CROSSED ABOVE CAMARILLA MONTHLY S3
#      EXACT 1-bar cross (most recent bar):
#        Bar[-2] close < Camarilla S3   ← was below S3 yesterday
#        Bar[-1] close >= Camarilla S3  ← crossed above today
#      Camarilla S3 = Close - (High - Low) × 1.1 / 4
#      (calculated from the prior completed month's H/L/C)
#      = A key institutional support/pivot level just reclaimed
#
# LOGIC FLOW:
#   Uptrend (price > SMA150) → Shallow pullback to SMA50 only
#   → Price reclaims EMA20 → Price reclaims Camarilla S3
#   = Confluence of short-term trend + monthly pivot support
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

    # ── C1: Bull structure ────────────────────────────────────
    # Price must be above SMA150 currently
    "require_price_above_sma150": True,

    # ── C2: Retest — SMA50 ONLY (not SMA150, not EMA20) ───────
    # Look for retest in last N bars
    "retest_lookback"           : 30,
    # Candle LOW must have come within X% of SMA50
    "retest_touch_pct"          : 3.0,
    # Price must NOT have closed below SMA150 by more than X%
    "max_close_below_sma150_pct": 1.5,
    # If True, reject if price ALSO touched SMA150 in the same
    # window — enforces "SMA50 only" (not a deeper SMA150 dip)
    "require_sma50_only"        : True,

    # ── C3: Currently above EMA20 ──────────────────────────────
    # (checked directly on current bar — no lookback needed)

    # ── C4: Camarilla S3 exact 1-bar cross ────────────────────
    # Bar[-2] close < Cam_S3   AND   Bar[-1] close >= Cam_S3
    "cam_cross_lookback"        : 5,

    # ── Volume ────────────────────────────────────────────────
    "vol_avg_bars"              : 20,
    # Volume on the S3 cross bar >= mult x avg
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

def cam_s3(high, low, close):
    """Camarilla S3 support level formula."""
    return close - (high - low) * 1.1 / 4.0

def get_monthly_cam_s3(df):
    """
    Returns the Camarilla S3 level for the CURRENT (in-progress)
    month, computed from last month's completed High/Low/Close.
    This is the standard approach — pivot levels for the current
    month are derived from the PRIOR completed month's range.
    Returns None if not enough data.
    """
    df = df.copy()
    df.index = pd.to_datetime(df.index)
    today = pd.Timestamp.today().normalize()
    prev_month = today.to_period("M") - 1

    sub = df[df.index.to_period("M") == prev_month]
    if len(sub) < 5:
        return None

    hi = float(sub["High"].max())
    lo = float(sub["Low"].min())
    cl = float(sub["Close"].iloc[-1])
    return round(cam_s3(hi, lo, cl), 4)

def build_cam_s3_series(df):
    """
    Builds a per-bar Camarilla S3 series: for each bar, the S3
    level is derived from the PRIOR completed month's H/L/C.
    This lets us detect the exact bar where price crossed above
    its month's S3 level.
    """
    df = df.copy()
    df.index = pd.to_datetime(df.index)
    periods = df.index.to_period("M")
    unique_months = sorted(periods.unique())

    # Precompute S3 for every month using the PRIOR month's H/L/C
    month_s3 = {}
    for i, mp in enumerate(unique_months):
        if i == 0:
            continue   # no prior month available
        prev_mp = unique_months[i-1]
        sub = df[periods == prev_mp]
        if len(sub) < 5:
            continue
        hi = float(sub["High"].max())
        lo = float(sub["Low"].min())
        cl = float(sub["Close"].iloc[-1])
        month_s3[mp] = cam_s3(hi, lo, cl)

    # Map each bar to its month's S3 level
    s3_values = [month_s3.get(mp, np.nan) for mp in periods]
    return pd.Series(s3_values, index=df.index)

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
    C1: Bull structure    — price currently above SMA150
    C2: SMA50-only retest — price touched SMA50 (not SMA150) in
                             last N bars, support held
    C3: Above EMA20       — current price > EMA20
    C4: Cam S3 cross      — exact 1-bar: prev below, today above
                             the monthly Camarilla S3 level
    """
    df      = df.copy(); df.index = pd.to_datetime(df.index)
    n       = len(df)
    price   = float(df["Close"].iloc[-1])
    avg_vol = float(df["Volume"].tail(CFG["vol_avg_bars"]).mean())

    if price   < CFG["min_price"]:      return None
    if avg_vol < CFG["min_avg_volume"]: return None
    if n < 30: return None

    # ── Compute MAs ───────────────────────────────────────────
    ema20_s  = calc_ema(df["Close"], CFG["ema20_period"])
    sma50_s  = df["Close"].rolling(CFG["sma50_period"]).mean()
    sma150_s = df["Close"].rolling(CFG["sma150_period"]).mean()
    rsi_s    = calc_rsi(df["Close"])
    macd_s, _, hist_s = calc_macd(df["Close"])
    cam_s3_s = build_cam_s3_series(df)

    cur_ema20  = float(ema20_s.iloc[-1])
    cur_sma50  = float(sma50_s.iloc[-1])
    cur_sma150 = float(sma150_s.iloc[-1])
    cur_rsi    = float(rsi_s.iloc[-1]) if not np.isnan(rsi_s.iloc[-1]) else 50
    cur_hist   = float(hist_s.iloc[-1]) if not np.isnan(hist_s.iloc[-1]) else 0
    cur_cam_s3 = float(cam_s3_s.iloc[-1]) if not np.isnan(cam_s3_s.iloc[-1]) else None

    if any(np.isnan([cur_ema20, cur_sma50, cur_sma150])): return None
    if cur_cam_s3 is None: return None   # not enough history for S3
    if cur_rsi < CFG["rsi_min"]: return None

    # ─────────────────────────────────────────────────────────
    # C1: BULL STRUCTURE — price currently above SMA150
    # ─────────────────────────────────────────────────────────
    if CFG["require_price_above_sma150"] and price < cur_sma150:
        return None

    # ─────────────────────────────────────────────────────────
    # C3: CURRENTLY ABOVE EMA20  (check early — fast gate)
    # ─────────────────────────────────────────────────────────
    if price < cur_ema20:
        return None

    # ─────────────────────────────────────────────────────────
    # C4: CAMARILLA S3 EXACT 1-BAR CROSS
    # Bar[-2] close < S3   AND   Bar[-1] close >= S3
    # ─────────────────────────────────────────────────────────
    ccl = CFG["cam_cross_lookback"]
    cam_cross_bar  = None
    cam_cross_date = None

    for i in range(max(1, n - ccl), n):
        pc  = float(df["Close"].iloc[i-1])
        cc  = float(df["Close"].iloc[i])
        ps3 = float(cam_s3_s.iloc[i-1]) if not np.isnan(cam_s3_s.iloc[i-1]) else np.nan
        cs3 = float(cam_s3_s.iloc[i])   if not np.isnan(cam_s3_s.iloc[i])   else np.nan
        if np.isnan(ps3) or np.isnan(cs3): continue
        # Exact 1-bar cross above Camarilla S3
        if pc < ps3 and cc >= cs3:
            cam_cross_bar  = i
            cam_cross_date = df.index[i]

    if cam_cross_bar is None: return None   # no S3 cross found

    # Volume on the S3 cross bar
    cross_vol = float(df["Volume"].iloc[cam_cross_bar])
    cross_vm  = cross_vol / avg_vol if avg_vol > 0 else 0
    if cross_vm < CFG["cross_vol_mult"]: return None

    # ─────────────────────────────────────────────────────────
    # C2: SMA50-ONLY RETEST
    # Search in the retest_lookback bars BEFORE the S3 cross
    # Must have touched SMA50, must NOT have also touched SMA150
    # ─────────────────────────────────────────────────────────
    rt_lb     = CFG["retest_lookback"]
    touch_pct = CFG["retest_touch_pct"] / 100

    rt_search_start = max(0, cam_cross_bar - rt_lb)
    rt_search_end   = cam_cross_bar   # up to (not including) the cross bar

    retest_sma50  = False
    retest_sma150 = False
    retest_bar    = None
    retest_price  = float("inf")

    for i in range(rt_search_start, rt_search_end):
        lo   = float(df["Low"].iloc[i])
        cl_i = float(df["Close"].iloc[i])
        s50  = float(sma50_s.iloc[i])  if not np.isnan(sma50_s.iloc[i])  else np.nan
        s150 = float(sma150_s.iloc[i]) if not np.isnan(sma150_s.iloc[i]) else np.nan

        if np.isnan(s50) or np.isnan(s150): continue

        # Check if candle LOW touched SMA50 (within touch_pct)
        if s50 > 0 and abs(lo - s50) / s50 <= touch_pct:
            retest_sma50 = True
            if cl_i < retest_price:
                retest_price = cl_i
                retest_bar   = i

        # Track if SMA150 was ALSO touched (to enforce "SMA50 only")
        if s150 > 0 and abs(lo - s150) / s150 <= touch_pct:
            retest_sma150 = True

    # Must have retested SMA50
    if not retest_sma50: return None

    # Enforce "SMA50 only" — reject if SMA150 was also touched
    if CFG["require_sma50_only"] and retest_sma150:
        return None

    # If SMA50 touch found but no distinct low bar tracked, use
    # the deepest close in the window as a fallback reference
    if retest_bar is None:
        for i in range(rt_search_start, rt_search_end):
            cl_i = float(df["Close"].iloc[i])
            if cl_i < retest_price:
                retest_price = cl_i
                retest_bar   = i

    # ── Validate: price did not close significantly below SMA150 ─
    max_close_below_s150 = 0.0
    for i in range(rt_search_start, cam_cross_bar + 1):
        cl_i = float(df["Close"].iloc[i])
        s150 = float(sma150_s.iloc[i]) if not np.isnan(sma150_s.iloc[i]) else cur_sma150
        pct_below = (s150 - cl_i) / s150 * 100 if s150 > 0 else 0
        if pct_below > max_close_below_s150:
            max_close_below_s150 = pct_below

    if max_close_below_s150 > CFG["max_close_below_sma150_pct"]: return None

    # ── Retest must be BEFORE the Camarilla S3 cross ──────────
    if retest_bar is not None and retest_bar >= cam_cross_bar:
        return None   # wrong order

    # ── Metrics ───────────────────────────────────────────────
    bars_since_cross  = n - 1 - cam_cross_bar
    bars_from_retest  = cam_cross_bar - retest_bar if retest_bar is not None else 0
    dist_ema20_pct    = (price - cur_ema20) / cur_ema20 * 100
    dist_sma50_pct    = (price - cur_sma50) / cur_sma50 * 100
    dist_sma150_pct   = (price - cur_sma150) / cur_sma150 * 100
    dist_cam_s3_pct   = (price - cur_cam_s3) / cur_cam_s3 * 100 if cur_cam_s3 else 0
    sma50_vs_sma150   = (cur_sma50 - cur_sma150) / cur_sma150 * 100

    # Retest depth: how close the low got to SMA50
    if retest_bar is not None:
        rt_lo  = float(df["Low"].iloc[retest_bar])
        s50_rt = float(sma50_s.iloc[retest_bar]) if not np.isnan(sma50_s.iloc[retest_bar]) else cur_sma50
        retest_depth_pct = abs(rt_lo - s50_rt) / s50_rt * 100 if s50_rt > 0 else 0.0
    else:
        retest_depth_pct = 0.0

    # Camarilla S3 value at the cross bar (for reference)
    cross_cam_s3 = float(cam_s3_s.iloc[cam_cross_bar])
    cross_close  = float(df["Close"].iloc[cam_cross_bar])

    # ── Score (0-100) ─────────────────────────────────────────
    score = 0

    # Retest quality (0-25): tighter SMA50 retest = better
    score += max(0, 25 - int(retest_depth_pct * 5))

    # Camarilla S3 cross freshness (0-25): today = 25
    score += max(0, 25 - bars_since_cross * 5)

    # Bars from retest to S3 cross (0-20): faster = better
    if bars_from_retest <= 3:
        score += 20
    elif bars_from_retest <= 7:
        score += 14
    elif bars_from_retest <= 15:
        score += 8

    # Distance above EMA20 (0-10): closer to EMA20 = fresher
    score += max(0, 10 - int(abs(dist_ema20_pct) * 2))

    # Volume on S3 cross (0-10)
    score += min(10, int(cross_vm * 4))

    # MACD momentum (0-5)
    score += 5 if cur_hist > 0 else 0

    # RSI health (0-5)
    score += min(5, max(0, int((cur_rsi - 40) / 4)))

    score = min(100, max(0, score))

    return {
        "Ticker"              : sym,
        "Price"               : round(price, 2),
        "Score"               : score,
        # Retest info (SMA50 only)
        "Retest_Bar_Date"     : df.index[retest_bar].strftime("%Y-%m-%d") if retest_bar is not None else "\u2014",
        "Retest_Depth_%"      : round(retest_depth_pct, 2),
        "Bars_Retest_to_Cross": bars_from_retest,
        "Max_Below_SMA150_%"  : round(max_close_below_s150, 2),
        # Camarilla S3 cross
        "Cam_S3_Cross_Date"   : cam_cross_date.strftime("%Y-%m-%d"),
        "Bars_Since_Cross"    : bars_since_cross,
        "Cam_S3_Level"        : round(cross_cam_s3, 2),
        "Cam_S3_Cross_Close"  : round(cross_close, 2),
        "Cross_Vol_x"         : round(cross_vm, 2),
        # MA levels
        "EMA20"               : round(cur_ema20, 2),
        "SMA50"               : round(cur_sma50, 2),
        "SMA150"              : round(cur_sma150, 2),
        "Cam_S3"              : round(cur_cam_s3, 2),
        "Dist_EMA20_%"        : round(dist_ema20_pct, 2),
        "Dist_SMA50_%"        : round(dist_sma50_pct, 2),
        "Dist_SMA150_%"       : round(dist_sma150_pct, 2),
        "Dist_Cam_S3_%"       : round(dist_cam_s3_pct, 2),
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
        "_cam_s3_series"      : cam_s3_s,
        "_cross_bar"          : cam_cross_bar,
        "_retest_bar"         : retest_bar,
    }


# ── Live print ────────────────────────────────────────────────
LIVE_COLS = [
    "Ticker","Price","Score",
    "Retest_Bar_Date","Retest_Depth_%","Bars_Retest_to_Cross",
    "Cam_S3_Cross_Date","Bars_Since_Cross","Cam_S3","Cross_Vol_x",
    "EMA20","SMA50","SMA150","Dist_EMA20_%","Dist_Cam_S3_%","RSI",
]
_CW = {
    "Ticker":8,"Price":10,"Score":7,
    "Retest_Bar_Date":15,"Retest_Depth_%":13,"Bars_Retest_to_Cross":20,
    "Cam_S3_Cross_Date":18,"Bars_Since_Cross":16,"Cam_S3":10,"Cross_Vol_x":12,
    "EMA20":9,"SMA50":9,"SMA150":9,"Dist_EMA20_%":13,"Dist_Cam_S3_%":14,"RSI":6,
}
_CF = {
    "Price":"${:.2f}","Score":"{:.0f}",
    "Retest_Depth_%":"{:.2f}%","Bars_Retest_to_Cross":"{:.0f}",
    "Bars_Since_Cross":"{:.0f}","Cam_S3":"${:.2f}","Cross_Vol_x":"{:.2f}×",
    "EMA20":"${:.2f}","SMA50":"${:.2f}","SMA150":"${:.2f}",
    "Dist_EMA20_%":"{:+.2f}%","Dist_Cam_S3_%":"{:+.2f}%","RSI":"{:.1f}",
}
_hdr_done = False

def _live_header():
    global _hdr_done
    if _hdr_done: return
    sep = "━" * 200
    print(f"\n{sep}")
    print("  📊  LIVE MATCHES  —  SMA50 Retest + EMA20 + Camarilla S3 Cross")
    print(sep)
    print("".join(f"  {c:<{_CW.get(c,10)}}" for c in LIVE_COLS))
    print("  " + "─"*198)
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
chk = download(["AAPL","MSFT","NVDA"], 200)
if not chk: print("❌  No data.")
else:
    for s, d in chk.items():
        p    = float(d["Close"].iloc[-1])
        e20  = float(calc_ema(d["Close"], 20).iloc[-1])
        s50  = float(d["Close"].rolling(50).mean().iloc[-1])
        s150 = float(d["Close"].rolling(150).mean().iloc[-1])
        print(f"  ✅ {s}: ${p:.2f}  EMA20=${e20:.2f}  SMA50=${s50:.2f}  "
              f"SMA150=${s150:.2f}  "
              f"Stack:{'✅' if e20>s50>s150 else '❌'}  {d.index[-1].date()}")
print()

# ── Diagnostic ────────────────────────────────────────────────
print("━"*65)
print("  STEP 2  DIAGNOSTIC (10 sample stocks)")
print("━"*65+"\n")

DIAG = ["AAPL","MSFT","NVDA","AMD","PLTR","META","CRWD","AVGO","DDOG","MU"]
diag_data = download(DIAG, CFG["history_days"])
print(f"  Downloaded {len(diag_data)}/{len(DIAG)}\n")
print(f"  {'SYM':<7} {'P':>8}  {'P1':>4}  {'RETEST':>12}  "
      f"{'EMA20X':>8}  {'SCORE':>6}  RESULT")
print("  "+"─"*60)

for sym in DIAG:
    if sym not in diag_data:
        print(f"  {sym:<7} — no data"); continue
    try:
        df_d = diag_data[sym]
        p    = float(df_d["Close"].iloc[-1])
        s150 = float(df_d["Close"].rolling(150).mean().iloc[-1])
        t    = lambda b: "✅" if b else "❌"
        p1   = p > s150
        r    = detect_pattern(sym, df_d)
        if r:
            print(f"  {sym:<7} ${p:>7.2f}  {t(p1):>4}  "
                  f"{r['Retest_Bar_Date']:>22}  "
                  f"{r['Cam_S3_Cross_Date']:>8}  "
                  f"{r['Score']:>6}  ✅")
        else:
            print(f"  {sym:<7} ${p:>7.2f}  {t(p1):>4}  "
                  f"{'no match':>22}  {'—':>8}  {'—':>6}  ❌")
    except Exception as e:
        print(f"  {sym:<7} error: {e}")

print(f"""
  Pattern:
    C1  Bull structure  : price currently above SMA150
    C2  SMA50-only retest: candle LOW touched SMA50 (NOT SMA150)
                          within ±{CFG['retest_touch_pct']}% in last {CFG['retest_lookback']} bars
    C3  Above EMA20     : current price > EMA20
    C4  Cam S3 cross    : Bar[-2] close < Camarilla S3
                          Bar[-1] close >= Camarilla S3 (exact 1-bar)
                          cross must happen AFTER the SMA50 retest

  Tune if mostly ❌:
    retest_touch_pct    {CFG['retest_touch_pct']} → 5    (wider retest zone)
    retest_lookback     {CFG['retest_lookback']} → 40   (look further back)
    cam_cross_lookback  {CFG['cam_cross_lookback']} → 10   (wider cross window)
    rsi_min             {CFG['rsi_min']} → 25
    max_close_below_sma150_pct {CFG['max_close_below_sma150_pct']} → 3
    require_sma50_only  True → False (allow SMA150 also touched)
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
            b = len(pool)
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
        t    = {row["symbol"].strip() for row in rows
                if row.get("symbol","").strip().isalpha()
                and 1<=len(row["symbol"].strip())<=5}
        b = len(pool); pool |= t
        print(f"  ✅ {'NASDAQ API':<18}: +{len(pool)-b:>4} → {len(pool)}")
    except Exception as e: print(f"  ⚠️  NASDAQ API: {e}")
    static = {
        "AAPL","MSFT","NVDA","AMZN","META","GOOGL","GOOG","TSLA","AVGO","COST",
        "NFLX","AMD","INTC","CSCO","ADBE","QCOM","TXN","AMAT","MU","KLAC",
        "LRCX","MRVL","MELI","PANW","CRWD","SNPS","CDNS","TEAM","WDAY","PLTR",
        "ALAB","SMCI","HOOD","COIN","SOFI","UPST","DDOG","SNOW","MDB","REGN",
        "VRTX","ISRG","LULU","FTNT","IDXX","SBUX","TMUS","RBRK","NET","MARA",
        "QUBT","RGTI","ASTS","RKLB","IONQ","FSLR","PYPL","ROKU","ROST","POOL",
        "AMGN","GILD","INTU","MCHP","MNST","NXPI","XEL","ACLS","IRTC","UPST",
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

got = len(TICKERS)-no_data; pct = got/max(len(TICKERS),1)*100
print(f"\n{'━'*65}")
print(f"  SCAN COMPLETE | {len(TICKERS)} tickers | {got} ({pct:.0f}%) | ✅ {len(results)} matches")
print(f"{'━'*65}")

# ── Results ───────────────────────────────────────────────────
if not results:
    print("\n  No matches. Try:")
    print("   retest_touch_pct           3 → 5")
    print("   retest_lookback           30 → 40")
    print("   cam_cross_lookback         5 → 10")
    print("   max_close_below_sma150_pct 1.5 → 3")
    print("   rsi_min                   35 → 25")
    print("   require_sma50_only      True → False")

# Sort by score (always runs, even on empty list)
results.sort(key=lambda x: x["Score"], reverse=True)

# ── Always build df_out and save/email (even if 0 results) ────
ts      = datetime.today().strftime("%Y%m%d_%H%M")
out_dir = os.environ.get("GITHUB_WORKSPACE", os.getcwd())

COLS = [
    "Ticker","Price","Score",
    "Retest_Bar_Date","Retest_Depth_%","Bars_Retest_to_Cross","Max_Below_SMA150_%",
    "Cam_S3_Cross_Date","Bars_Since_Cross","Cam_S3","Cam_S3_Cross_Close","Cross_Vol_x",
    "EMA20","SMA50","SMA150",
    "Dist_EMA20_%","Dist_SMA50_%","Dist_SMA150_%","Dist_Cam_S3_%","SMA50_vs_SMA150_%",
    "RSI","MACD_Hist","Avg_Vol_20d",
]
df_out = pd.DataFrame([{k:v for k,v in r.items() if not k.startswith("_")}
                        for r in results]) if results else pd.DataFrame(columns=COLS)
if not df_out.empty:
    df_out = df_out[[c for c in COLS if c in df_out.columns]]
    df_out.reset_index(drop=True, inplace=True)

FMT = {
    "Price"               : lambda v: f"${v:.2f}",
    "Score"               : lambda v: f"{v:.0f}",
    "EMA20"               : lambda v: f"${v:.2f}",
    "SMA50"               : lambda v: f"${v:.2f}",
    "SMA150"              : lambda v: f"${v:.2f}",
    "Cam_S3"              : lambda v: f"${v:.2f}",
    "Cam_S3_Cross_Close"  : lambda v: f"${v:.2f}",
    "Retest_Depth_%"      : lambda v: f"{v:.2f}%",
    "Max_Below_SMA150_%"  : lambda v: f"{v:.2f}%",
    "Bars_Retest_to_Cross": lambda v: f"{int(v)}",
    "Bars_Since_Cross"    : lambda v: f"{int(v)}",
    "Cross_Vol_x"         : lambda v: f"{v:.2f}×",
    "Dist_EMA20_%"        : lambda v: f"{v:+.2f}%",
    "Dist_SMA50_%"        : lambda v: f"{v:+.2f}%",
    "Dist_SMA150_%"       : lambda v: f"{v:+.2f}%",
    "Dist_Cam_S3_%"       : lambda v: f"{v:+.2f}%",
    "SMA50_vs_SMA150_%"   : lambda v: f"{v:+.2f}%",
    "RSI"                 : lambda v: f"{v:.1f}",
    "MACD_Hist"           : lambda v: f"{v:.4f}",
    "Avg_Vol_20d"         : lambda v: f"{v:,.0f}",
}

def fmt_v(col, val):
    if val is None or (isinstance(val, float) and np.isnan(val)): return "—"
    try:
        if col in FMT: return FMT[col](val)
    except Exception: pass
    return str(val) if str(val) not in ("nan","None","") else "—"

if _IN_NOTEBOOK and results:
    DISP = ["Ticker","Price","Score",
            "Retest_Bar_Date","Retest_Depth_%","Bars_Retest_to_Cross",
            "Cam_S3_Cross_Date","Bars_Since_Cross","Cam_S3","Cross_Vol_x",
            "EMA20","SMA50","SMA150","Dist_EMA20_%","Dist_Cam_S3_%","RSI","MACD_Hist"]
    DISP = [c for c in DISP if c in df_out.columns]

    gc = "#22c55e"   # single accent colour for this pattern

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
            elif col in ("Dist_EMA20_%","Dist_SMA50_%","Dist_Cam_S3_%"):
                try:
                    v = float(str(raw).replace("%","").replace("+",""))
                    clr = "#22c55e" if v >= 0 else "#ef4444"
                    sty = f"color:{clr};font-weight:600"
                except Exception: pass
            elif col == "Bars_Since_Cross":
                try:
                    v = int(float(raw))
                    if v == 0: sty = "color:#22c55e;font-weight:700;text-align:center"
                    elif v <= 1: sty = "color:#86efac;text-align:center"
                except Exception: pass
            elif col == "Retest_Depth_%":
                try:
                    v = float(str(raw).replace("%",""))
                    if v <= 1.0: sty = "color:#22c55e;font-weight:700"
                    elif v <= 2.0: sty = "color:#86efac"
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
    <span style="color:#f1f5f9;font-size:15px;font-weight:700">SMA50 Retest → EMA20 → Camarilla S3 Cross</span>
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
    📈 SMA50 Retest + EMA20 + Camarilla S3 Cross
  </h2>
  <p style="margin:6px 0 0;color:#94a3b8;font-size:12px">
    {datetime.today().strftime('%Y-%m-%d %H:%M')} &nbsp;·&nbsp;
    <b style="color:#22c55e">{len(results)} matches</b> from {len(TICKERS)} tickers
  </p>
</div>"""

    legend_html = """
<div style="background:#f8fafc;border:1px solid #e2e8f0;border-radius:8px;
        padding:12px 18px;margin-top:6px;font-size:11px;color:#64748b;
        font-family:'Segoe UI',Arial,sans-serif">
  <b style="color:#475569">GUIDE</b> &nbsp;·&nbsp;
  Price above SMA150 &nbsp;·&nbsp;
  Retested SMA50 only (not SMA150) &nbsp;·&nbsp;
  Currently above EMA20 &nbsp;·&nbsp;
  Just crossed above monthly Camarilla S3 &nbsp;·&nbsp;
  Retest_Depth_% = how close the low came to SMA50 (0% = exact touch) &nbsp;·&nbsp;
  Bars_Since_Cross 0 = S3 crossed today
</div>"""

    display_html(header_html + table_html + legend_html)

elif results:
    # ASCII table (CLI/GitHub Actions mode)
    CLI_COLS = ["Ticker","Price","Score",
                "Retest_Depth_%","Bars_Retest_to_Cross",
                "Cam_S3_Cross_Date","Bars_Since_Cross","Dist_EMA20_%","Dist_Cam_S3_%","RSI"]
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
    tit  = f"  SMA50 Retest + EMA20 + Cam S3 Cross   {datetime.today().strftime('%Y-%m-%d')}   {len(df_out)} matches"
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
  Retest_Depth_%     0% = exact touch on SMA50, lower = better
  Bars_Retest_Cross  bars between SMA50 retest and Cam S3 cross
  Bars_Since_Cross   0 = Camarilla S3 crossed today
  Dist_EMA20_%       how far price is above EMA20 now
  Dist_Cam_S3_%      how far price is above Camarilla S3 now
  ──────────────────────────────────────────────────────""")

# Save
out_dir = os.environ.get("GITHUB_WORKSPACE", os.getcwd())
fpath   = os.path.join(out_dir, f"sma_retest_ema20_cross_{ts}.csv")
df_out.to_csv(fpath, index=False)
print(f"\n  💾 CSV → {fpath}")
tv = os.path.join(out_dir, f"tv_sma_retest_{ts}.txt")
with open(tv,"w") as f:
    f.write(f"###SMA Retest EMA20 Cross {datetime.today().strftime('%Y-%m-%d')}\n")
    for r in results: f.write(f"NASDAQ:{r['Ticker']}\n")
print(f"  📋 TradingView → {tv}")

# ── Email with CSV attached ───────────────────────────────
def _send_email(rl, csv_path):
    import smtplib
    from email.mime.multipart import MIMEMultipart
    from email.mime.text      import MIMEText
    from email.mime.base      import MIMEBase
    from email                import encoders

    # Use module-level vars (already read at startup)
    gu = _GMAIL_USER
    gp = _GMAIL_PASS
    et = _EMAIL_TO

    if not gu:
        print("[Email] ❌  GMAIL_USER secret is empty")
        print("         → Repo → Settings → Secrets → Actions → GMAIL_USER")
        return
    if not gp:
        print("[Email] ❌  GMAIL_PASS secret is empty")
        print("         → Must be a Gmail App Password (16 chars, no spaces)")
        print("         → Get one at: myaccount.google.com/apppasswords")
        return
    if not et:
        print("[Email] ❌  EMAIL_TO secret is empty")
        print("         → Repo → Settings → Secrets → Actions → EMAIL_TO")
        return

    eto = [e.strip() for e in et.split(",") if e.strip()]
    cnt = len(rl)

    try:
        print(f"[Email] Sending to {et}  ({cnt} results)...")

        th_e = "".join(
            f'<th style="background:#1e293b;color:#e2e8f0;padding:8px 11px;'
            f'font-size:11px;font-weight:700;border-bottom:2px solid #3b82f6;'
            f'white-space:nowrap">{c}</th>'
            for c in ["Ticker","Price","Score","Retest_Depth_%",
                      "Bars_Since_Cross","Dist_EMA20_%","Dist_Cam_S3_%","RSI"]
        )
        rows_e = ""
        for i, r in enumerate(rl[:50]):
            bg  = "#fff" if i % 2 == 0 else "#f0f9ff"
            ticker = r.get("Ticker","—")
            price  = r.get("Price",0) or 0
            score  = r.get("Score",0) or 0
            depth  = r.get("Retest_Depth_%",0) or 0
            bsc    = r.get("Bars_Since_Cross",99)
            edist  = r.get("Dist_EMA20_%",0) or 0
            s3dist = r.get("Dist_Cam_S3_%",0) or 0
            rsi    = r.get("RSI",0) or 0
            rows_e += (
                f'<tr style="background:{bg}">'
                f'<td style="padding:6px 11px;font-size:12px;font-weight:700">{ticker}</td>'
                f'<td style="padding:6px 11px;font-size:12px">${float(price):.2f}</td>'
                f'<td style="padding:6px 11px;font-size:12px;font-weight:700;'
                f'background:#166534;color:#fff;text-align:center">{float(score):.0f}</td>'
                f'<td style="padding:6px 11px;font-size:12px">{float(depth):.2f}%</td>'
                f'<td style="padding:6px 11px;font-size:12px;color:'
                f'{"#22c55e" if bsc==0 else "#94a3b8"};font-weight:700">'
                f'{bsc}d</td>'
                f'<td style="padding:6px 11px;font-size:12px">{float(edist):+.2f}%</td>'
                f'<td style="padding:6px 11px;font-size:12px">{float(s3dist):+.2f}%</td>'
                f'<td style="padding:6px 11px;font-size:12px">{float(rsi):.1f}</td>'
                f'</tr>'
            )

        no_results_msg = ""
        if cnt == 0:
            no_results_msg = (
                '<tr><td colspan="8" style="padding:20px;text-align:center;'
                'color:#64748b;font-size:13px">No matches found today — '
                'market conditions did not trigger the pattern</td></tr>'
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
  📊 SMA50 Retest + EMA20 + Camarilla S3 Cross
</h1>
<p style="margin:6px 0 0;color:#94a3b8;font-size:12px">
  {datetime.today().strftime('%Y-%m-%d %H:%M UTC')} &nbsp;·&nbsp;
  {cnt} match{'es' if cnt!=1 else ''} found
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
            f"SMA50 Retest + EMA20 + Camarilla S3 Cross — {datetime.today().strftime('%Y-%m-%d')}",
            f"{cnt} matches",
            "="*60,
        ]
        if rl:
            for r in rl[:50]:
                ticker = r.get("Ticker","—")
                price  = r.get("Price",0) or 0
                score  = r.get("Score",0) or 0
                depth  = r.get("Retest_Depth_%",0) or 0
                bsc    = r.get("Bars_Since_Cross",0) or 0
                plain_lines.append(
                    f"{ticker:<7} ${float(price):.2f}  Score:{float(score):.0f}  "
                    f"SMA50 Depth:{float(depth):.1f}%  S3 Cross:{bsc}d ago"
                )
        else:
            plain_lines.append("No matches today")
        plain_lines.append("\nFull results in CSV attachment.")
        plain_e = "\n".join(plain_lines)

        subj = (f"📊 SMA50 Retest + Cam S3 Cross — {cnt} signal{'s' if cnt!=1 else ''}"
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

    # Attach CSV
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

    # Send
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
        print("         Generate one: myaccount.google.com/apppasswords")
        print("         → Google Account → Security → 2-Step Verification → App Passwords")
    except smtplib.SMTPException as e:
        print(f"[Email] ❌  SMTP error: {e}")
    except Exception as e:
        print(f"[Email] ❌  Unexpected error: {type(e).__name__}: {e}")

try:
    _send_email(results, fpath)
except Exception as e:
    print(f"[Email] ❌  Unexpected top-level error: {type(e).__name__}: {e}")
    print("[Email]    Continuing — CSV and charts are still saved.")

if _IN_NOTEBOOK:
    try:
        from google.colab import files
        files.download(fpath); files.download(tv)
    except Exception: pass
else:
    print("  (CI: files in workspace, email sent)")

# ── Charts for top 5 ──────────────────────────────────────────
if results:
    top = results[:min(5, len(results))]
    fig, axes = plt.subplots(len(top), 1, figsize=(15, 5*len(top)), facecolor="#0f172a")
    if len(top)==1: axes=[axes]

    for idx, r in enumerate(top):
        ax    = axes[idx]
        df_p  = r["_df"].tail(60).copy()
        ema20 = r["_ema20"].reindex(df_p.index)
        sma50 = r["_sma50"].reindex(df_p.index)
        sma150= r["_sma150"].reindex(df_p.index)
        cams3 = r["_cam_s3_series"].reindex(df_p.index)
        n_p   = len(df_p)
        fn    = len(r["_df"]); off = fn - n_p

        ax.set_facecolor("#0f172a")

        # Candlestick
        for i, (_, row_) in enumerate(df_p.iterrows()):
            o=float(row_["Open"]); h=float(row_["High"])
            l=float(row_["Low"]);  c=float(row_["Close"])
            clr="#34d399" if c>=o else "#ef4444"
            ax.plot([i,i],[l,h],color=clr,lw=0.7,zorder=2)
            blo=min(o,c); bhi=max(o,c); bh=max(bhi-blo,(h-l)*0.005)
            rect=mpatches.FancyBboxPatch((i-0.3,blo),0.6,bh,
                 boxstyle="square,pad=0",facecolor=clr,edgecolor=clr,lw=0.4,zorder=3)
            ax.add_patch(rect)

        # MAs + Camarilla S3 (matching chart colours)
        ax.plot(range(n_p), ema20.values,  color="#34d399", lw=1.6, label="EMA20 🟢", zorder=5)
        ax.plot(range(n_p), sma50.values,  color="#3b82f6", lw=1.5, label="SMA50 🔵", zorder=4)
        ax.plot(range(n_p), sma150.values, color="#f472b6", lw=1.5, ls="-.", label="SMA150 🩷", zorder=4)
        ax.plot(range(n_p), cams3.values,  color="#f59e0b", lw=1.4, ls=":",  label="Cam S3 🟠", zorder=4)

        # SMA50 retest zone shading
        rt_pos = r["_retest_bar"] - off if r["_retest_bar"] is not None else None
        if rt_pos is not None and 0 <= rt_pos < n_p:
            ax.scatter([rt_pos],[float(df_p["Low"].iloc[rt_pos])],
                       color="#fbbf24", s=120, zorder=7, marker="v",
                       label=f"SMA50 Retest {r['Retest_Bar_Date']}")
            ax.axvline(rt_pos, color="#fbbf24", lw=1.0, ls=":", alpha=0.6)

        ax.axhspan(float(r["SMA50"])*0.97, float(r["SMA50"])*1.03,
                   alpha=0.05, color="#3b82f6", zorder=1)

        # Camarilla S3 cross bar
        xp = r["_cross_bar"] - off
        if 0 <= xp < n_p:
            ax.axvline(xp, color="#f59e0b", lw=1.8, ls="--", alpha=0.9)
            ax.scatter([xp],[float(df_p["Close"].iloc[xp])],
                       color="#f59e0b", s=180, zorder=8, marker="^",
                       label=f"Cam S3 Cross {r['Cam_S3_Cross_Date']}")

        tick_step = max(1, n_p//8)
        ax.set_xticks(range(0, n_p, tick_step))
        ax.set_xticklabels(
            [df_p.index[i].strftime("%m/%d") for i in range(0,n_p,tick_step)],
            color="#94a3b8", fontsize=7)
        ax.set_xlim(-0.5, n_p-0.5)
        ax.set_title(
            f"{r['Ticker']}  ${r['Price']:.2f}  |  Score {r['Score']}/100  |  "
            f"SMA50 Retest {r['Retest_Bar_Date']} (depth {r['Retest_Depth_%']:.1f}%)  |  "
            f"→  Cam S3 cross {r['Cam_S3_Cross_Date']} "
            f"({r['Bars_Since_Cross']}d ago)  ({r['Bars_Retest_to_Cross']}bars later)  |  "
            f"RSI {r['RSI']:.0f}",
            color="#e2e8f0", fontsize=8, fontweight="bold", pad=6)
        ax.tick_params(colors="#94a3b8", labelsize=7)
        for sp_ in ax.spines.values(): sp_.set_edgecolor("#1e3a5f")
        ax.legend(loc="upper left", facecolor="#1e293b",
                  labelcolor="#e2e8f0", fontsize=7, framealpha=0.9)
        ax.grid(color="#1e3a5f", ls="--", lw=0.4, alpha=0.4, axis="y")

    plt.suptitle(
        f"SMA50 Retest → EMA20 → Camarilla S3 Cross  ·  "
        f"{datetime.today().strftime('%Y-%m-%d')}\n"
        f"🟢 EMA20  🔵 SMA50  🩷 SMA150  🟠 Cam S3  ▼ = SMA50 Retest  ▲ = S3 Cross",
        color="#60a5fa", fontsize=10, fontweight="bold", y=1.001)
    plt.tight_layout()
    cp = os.path.join(os.environ.get("GITHUB_WORKSPACE", os.getcwd()),
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

  C1  BULL STRUCTURE  (prerequisite)
      Price > SMA150  (above long-term support)
      = Stock is in a healthy long-term uptrend

  C2  SMA50-ONLY RETEST  (the pullback — SMA50 specific)
      Price pulled back and candle LOW touched SMA50 (🔵 blue)
      Price did NOT also touch SMA150 in the same window
        (that would be a deeper SMA150 retest, not this pattern)
      Support held — no significant close below SMA150
      = A shallow, controlled pullback to medium-term support

  C3  CURRENTLY ABOVE EMA20
      Price is now trading above EMA20 (🟢 green)
      = Short-term trend has been reclaimed

  C4  CAMARILLA S3 EXACT 1-BAR CROSS  (the entry trigger)
      Bar[-2] close < Camarilla S3   ← was below S3 yesterday
      Bar[-1] close >= Camarilla S3  ← crossed above S3 today
      Camarilla S3 (🟠 orange) = Close - (High-Low) × 1.1 / 4
      calculated from the PRIOR completed month's range
      = Price just reclaimed a key monthly pivot support level

  LOGIC FLOW:
    Uptrend (price > SMA150)
      → Shallow pullback to SMA50 only (not SMA150)
      → Price reclaims EMA20
      → Price reclaims Camarilla S3
      = Confluence entry: short-term trend + monthly pivot support

  💡 BEST SETUPS
  Retest_Depth_% < 1%       almost exact SMA50 touch = clean retest
  Bars_Since_Cross = 0      Cam S3 crossed today = freshest entry
  Bars_Retest_to_Cross ≤ 5  quick recovery = strong momentum
  Dist_EMA20_% small         price just above EMA20, not extended
  RSI 45–65                  healthy, not overbought

  ⚙️  TUNE IF 0 RESULTS
  retest_touch_pct           3 → 5
  retest_lookback           30 → 40
  cam_cross_lookback         5 → 10
  max_close_below_sma150_pct 1.5 → 3
  rsi_min                   35 → 25
  require_sma50_only      True → False  (allow SMA150 also touched)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
""")
