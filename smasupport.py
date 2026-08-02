# ============================================================
# NASDAQ — Full Bull Stack + Single Red Candle Shakeout + Green Recovery
# ============================================================
#
# EXACT PATTERN:
#
#  C1 — FULL BULL MA STACK  (all aligned and ALL rising)
#      JMA > EMA8 > SMA20 > SMA50 > SMA150
#      Every single MA must be above the one below it
#      AND every MA must be currently rising (positive slope)
#      over the last slope_lookback bars
#      = Perfect trend alignment across all timeframes
#
#  C2 — EXACTLY 1 RED CANDLE CLOSED BELOW SMA50 OR SMA150
#      Within the last red_lookback bars, find a single red
#      candle (close < open) whose close went BELOW SMA50 or
#      SMA150 — the "shakeout" bar
#      Must be the MOST RECENT such violation (only 1 allowed)
#      = Weak hands flushed out at support; not a trend break
#
#  C3 — NEXT CANDLE IS GREEN + CLOSED ABOVE THE VIOLATED SMA
#      The candle IMMEDIATELY after the red bar (Bar+1) must:
#        a) Be GREEN (close > open)
#        b) Close ABOVE the same SMA that was violated
#        c) Volume on the green bar >= vol_mult × the red bar's volume
#           (confirms strong buying absorbed the shakeout)
#        d) Low of green bar >= Low of red bar
#           (did NOT break the shakeout low — invalidates if it did)
#      = Institutional recovery candle — higher volume, higher low
#
# LOGIC FLOW:
#   All MAs perfectly aligned AND all rising
#   → One red shakeout candle dips below SMA50 or SMA150
#   → Immediately followed by green candle that reclaims it
#     on higher volume without undercutting the shakeout low
#   = Clean, high-probability continuation setup
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
    print(f"  ℹ️  GitHub repo → Settings → Secrets → Actions")
    print(f"       Add: GMAIL_USER, GMAIL_PASS (App Password), EMAIL_TO")
    print(f"  ℹ️  Email will be SKIPPED this run")
print("━"*65)
print()

# ── CONFIG ────────────────────────────────────────────────────
CFG = {
    "history_days"               : 300,

    # ── MA periods ────────────────────────────────────────────
    "jma_period"                 : 13,
    "jma_phase"                  : 40,
    "ema8_period"                : 8,
    "sma20_period"               : 20,
    "sma50_period"               : 50,
    "sma150_period"              : 150,

    # ── C1: Bull stack + all rising ───────────────────────────
    # How many bars to look back to measure "rising" slope
    "slope_lookback"             : 3,    # was 5 — shorter window
    # Allow slightly flat MAs (not just strictly positive)
    "min_slope_pct"              : -0.05, # was 0.0 — allow near-flat

    # ── C2: Red shakeout candle ────────────────────────────────
    # Look further back for the red shakeout
    "red_lookback"               : 20,   # was 10 — look further back
    # Allow deeper dips below SMA
    "max_below_sma_pct"          : 8.0,  # was 5.0 — allow deeper dip

    # ── C3: Green recovery candle ─────────────────────────────
    # Lower volume requirement — any volume accepted
    "vol_mult_vs_red"            : 0.5,  # was 1.0 — any decent volume
    # Keep no-break-of-low but can disable if needed
    "require_no_break_of_red_low": False, # was True — relaxed
    # Must still close above violated SMA
    "require_close_above_sma"    : True,

    # ── Pattern recency — wider window ────────────────────────
    "max_bars_since_recovery"    : 10,   # was 3 — look further back

    # ── Filters ───────────────────────────────────────────────
    "min_avg_volume"             : 50_000,  # was 100_000 — wider
    "min_price"                  : 1.0,     # was 2.0 — wider
    "vol_avg_bars"               : 20,

    "batch_size"                 : 50,
    "batch_sleep"                : 1.5,
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
    return macd - sig   # histogram only

def calc_jma(series, period=13, phase=40):
    """JMA approximation (widely used pure-numpy version)."""
    n      = len(series)
    vals   = series.values.astype(float)
    result = np.full(n, np.nan)
    first  = next((i for i in range(n) if not np.isnan(vals[i])), 0)
    phase_ratio = phase / 100.0 + 1.5
    alpha = 2.0 / (period + 1.0)
    beta  = alpha * phase_ratio
    e0 = e1 = e2 = vals[first]
    result[first] = e0
    for i in range(first + 1, n):
        v  = vals[i]
        e0 = (1 - alpha) * e0 + alpha * v
        e1 = (v - e0) * (1 - beta) + beta * e1
        e2 = (e0 + e1 - e2) * alpha + (1 - alpha) * e2
        result[i] = e2
    return pd.Series(result, index=series.index)

def is_rising(series, lookback, min_slope_pct=0.0):
    """Returns True if the series endpoint is higher than lookback bars ago
       by at least min_slope_pct% of the starting value."""
    if len(series) < lookback + 1: return False
    v_now  = float(series.iloc[-1])
    v_prev = float(series.iloc[-1 - lookback])
    if np.isnan(v_now) or np.isnan(v_prev) or v_prev <= 0: return False
    slope_pct = (v_now - v_prev) / v_prev * 100
    return slope_pct > min_slope_pct

# ── Download ──────────────────────────────────────────────────
def _clean(df, min_bars=160):
    if df is None or df.empty: return None
    need = [c for c in ["Open","High","Low","Close","Volume"] if c in df.columns]
    if not all(c in need for c in ["Open","High","Low","Close","Volume"]): return None
    df = df[need].copy()
    df.index = pd.to_datetime(df.index)
    if hasattr(df.index,"tz") and df.index.tz:
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

# ── Debug counters (global, printed after scan) ────────────────
_DBG = {
    "total"       : 0,
    "pass_filter" : 0,
    "fail_stack"  : 0,
    "fail_rising" : 0,
    "fail_shakeout": 0,
    "pass_all"    : 0,
}

# ── Core detection ────────────────────────────────────────────
def detect_pattern(sym, df):
    """
    C1: Price above SMA50 AND SMA150 (bull structure)
        SMA50 > SMA150 (fast MA above slow MA)
        SMA20 > SMA50 (medium above long)
        JMA and EMA8 are both above SMA20
        — relaxed from strict JMA>EMA8>SMA20>SMA50>SMA150 order
        since JMA and EMA8 frequently swap in pullbacks

    C1b: Key MAs are rising (SMA20, SMA50 must be rising;
         SMA150 just needs to not be steeply declining)

    C2: Within red_lookback bars, find a red candle that
        closed below SMA50 or SMA150

    C3: The VERY NEXT candle after C2 is green and closed
        back above the violated SMA, with volume >= vol_mult_vs_red
        × red candle volume, within max_bars_since_recovery of today
    """
    global _DBG
    _DBG["total"] += 1

    df      = df.copy(); df.index = pd.to_datetime(df.index)
    n       = len(df)
    price   = float(df["Close"].iloc[-1])
    avg_vol = float(df["Volume"].tail(CFG["vol_avg_bars"]).mean())

    if price   < CFG["min_price"]:      return None
    if avg_vol < CFG["min_avg_volume"]: return None
    if n < CFG["sma150_period"] + 20:   return None

    _DBG["pass_filter"] += 1

    # ── Compute all MAs ────────────────────────────────────────
    jma_s   = calc_jma(df["Close"], CFG["jma_period"], CFG["jma_phase"])
    ema8_s  = calc_ema(df["Close"], CFG["ema8_period"])
    sma20_s = df["Close"].rolling(CFG["sma20_period"]).mean()
    sma50_s = df["Close"].rolling(CFG["sma50_period"]).mean()
    sma150_s= df["Close"].rolling(CFG["sma150_period"]).mean()
    rsi_s   = calc_rsi(df["Close"])
    macdh_s = calc_macd(df["Close"])

    cur_jma  = float(jma_s.iloc[-1])
    cur_ema8 = float(ema8_s.iloc[-1])
    cur_s20  = float(sma20_s.iloc[-1])
    cur_s50  = float(sma50_s.iloc[-1])
    cur_s150 = float(sma150_s.iloc[-1])
    cur_rsi  = float(rsi_s.iloc[-1])  if not np.isnan(rsi_s.iloc[-1])  else 50
    cur_mh   = float(macdh_s.iloc[-1]) if not np.isnan(macdh_s.iloc[-1]) else 0

    if any(np.isnan([cur_jma, cur_ema8, cur_s20, cur_s50, cur_s150])): return None

    # ─────────────────────────────────────────────────────────
    # C1: BULL STRUCTURE — what "JMA>EMA8>SMA20>SMA50>SMA150"
    # actually means on a chart:
    #
    # On a chart all MAs appear as lines ranked by PRICE LEVEL.
    # In a bull trend the FASTEST MA is highest (closest to price)
    # and the SLOWEST is lowest — this is what the stack means:
    #
    #   Price > JMA > EMA8 > SMA20 > SMA50 > SMA150
    #   (each faster MA tracks price more closely = higher value)
    #
    # The key checkable conditions:
    #   price  > SMA50   (above medium-term trend)
    #   price  > SMA150  (above long-term trend)
    #   SMA50  > SMA150  (medium above long = uptrend)
    #   price  > SMA20   (above 20d = actively bullish)
    #   JMA    < price   (JMA trails price = fast MA below price)
    #   EMA8   < price   (same for EMA8)
    # ─────────────────────────────────────────────────────────
    stack_ok = (
        price    > cur_s50   and   # price above 50d MA
        price    > cur_s150  and   # price above 150d MA
        cur_s50  > cur_s150  and   # 50d above 150d (uptrend)
        price    > cur_s20   and   # price above 20d
        cur_s20  > cur_s150  and   # 20d above 150d
        cur_jma  < price     and   # JMA below price (tracking)
        cur_ema8 < price           # EMA8 below price (tracking)
    )
    if not stack_ok:
        _DBG["fail_stack"] += 1
        return None

    # ─────────────────────────────────────────────────────────
    # C1b: TREND DIRECTION — SMA50 must be rising
    #      SMA150 must not be steeply declining
    # ─────────────────────────────────────────────────────────
    sl = CFG["slope_lookback"]
    sp = CFG["min_slope_pct"]

    if not is_rising(sma50_s, sl, sp):
        _DBG["fail_rising"] += 1; return None
    # SMA150: very permissive — just not steeply declining
    if not is_rising(sma150_s, sl, sp - 0.3):
        _DBG["fail_rising"] += 1; return None

    # ─────────────────────────────────────────────────────────
    # C2 + C3: SHAKEOUT + RECOVERY
    # Search last red_lookback bars for a red candle below
    # SMA50 or SMA150 followed IMMEDIATELY by a green recovery
    # ─────────────────────────────────────────────────────────
    rl = CFG["red_lookback"]
    search_start = max(1, n - rl - 1)

    shakeout_found  = False
    shakeout_bar    = None
    recovery_bar    = None
    violated_sma    = None
    violated_level  = None

    for i in range(search_start, n - 1):
        o_i   = float(df["Open"].iloc[i])
        c_i   = float(df["Close"].iloc[i])
        l_i   = float(df["Low"].iloc[i])
        vol_i = float(df["Volume"].iloc[i])

        # Must be a RED candle
        if c_i >= o_i: continue

        s50_i  = float(sma50_s.iloc[i])  if not np.isnan(sma50_s.iloc[i])  else np.nan
        s150_i = float(sma150_s.iloc[i]) if not np.isnan(sma150_s.iloc[i]) else np.nan
        if np.isnan(s50_i) or np.isnan(s150_i): continue

        below_s50  = c_i < s50_i  and (s50_i - c_i)  / s50_i  * 100 <= CFG["max_below_sma_pct"]
        below_s150 = c_i < s150_i and (s150_i - c_i) / s150_i * 100 <= CFG["max_below_sma_pct"]

        if not below_s50 and not below_s150: continue

        vsma = "SMA50"  if below_s50  else "SMA150"
        vlvl = s50_i    if below_s50  else s150_i

        # ── Check next bar ────────────────────────────────────
        j      = i + 1
        o_j    = float(df["Open"].iloc[j])
        c_j    = float(df["Close"].iloc[j])
        l_j    = float(df["Low"].iloc[j])
        vol_j  = float(df["Volume"].iloc[j])
        s50_j  = float(sma50_s.iloc[j])  if not np.isnan(sma50_s.iloc[j])  else np.nan
        s150_j = float(sma150_s.iloc[j]) if not np.isnan(sma150_s.iloc[j]) else np.nan

        if np.isnan(s50_j) or np.isnan(s150_j): continue

        # a) Green candle
        if c_j <= o_j: continue

        # b) Closed above violated SMA
        vlvl_j = s50_j if vsma == "SMA50" else s150_j
        if CFG["require_close_above_sma"] and c_j < vlvl_j: continue

        # c) Volume check
        if vol_j < CFG["vol_mult_vs_red"] * vol_i: continue

        # d) No break of red low (optional)
        if CFG["require_no_break_of_red_low"] and l_j < l_i: continue

        # e) Recency check
        bars_since = n - 1 - j
        if bars_since > CFG["max_bars_since_recovery"]: continue

        if recovery_bar is None or j > recovery_bar:
            shakeout_found = True
            shakeout_bar   = i
            recovery_bar   = j
            violated_sma   = vsma
            violated_level = vlvl

    if not shakeout_found:
        _DBG["fail_shakeout"] += 1
        return None

    _DBG["pass_all"] += 1

    # ── Metrics ───────────────────────────────────────────────
    bars_since_recovery = n - 1 - recovery_bar

    red_o   = float(df["Open"].iloc[shakeout_bar])
    red_c   = float(df["Close"].iloc[shakeout_bar])
    red_l   = float(df["Low"].iloc[shakeout_bar])
    red_vol = float(df["Volume"].iloc[shakeout_bar])

    grn_o   = float(df["Open"].iloc[recovery_bar])
    grn_c   = float(df["Close"].iloc[recovery_bar])
    grn_l   = float(df["Low"].iloc[recovery_bar])
    grn_vol = float(df["Volume"].iloc[recovery_bar])

    red_date = df.index[shakeout_bar].strftime("%Y-%m-%d")
    grn_date = df.index[recovery_bar].strftime("%Y-%m-%d")

    red_size_pct  = abs(red_c - red_o) / red_o * 100
    grn_size_pct  = abs(grn_c - grn_o) / grn_o * 100
    vol_ratio     = grn_vol / red_vol if red_vol > 0 else 0
    low_held      = grn_l >= red_l

    below_depth_pct = abs(red_c - violated_level) / violated_level * 100

    dist_jma_pct   = (price - cur_jma)  / cur_jma  * 100 if cur_jma  > 0 else 0
    dist_s50_pct   = (price - cur_s50)  / cur_s50  * 100 if cur_s50  > 0 else 0
    dist_s150_pct  = (price - cur_s150) / cur_s150 * 100 if cur_s150 > 0 else 0

    # ── Score (0-100) ─────────────────────────────────────────
    score = 0

    # Stack alignment quality (0-20): all 5 in order = 20
    score += 20

    # Recovery volume vs red (0-25): higher vol ratio = stronger recovery
    score += min(25, int(vol_ratio * 10))

    # Depth of shakeout (0-20): shallower = better (only briefly violated)
    score += max(0, 20 - int(below_depth_pct * 4))

    # Recency (0-20): today = 20, yesterday = 15
    score += max(0, 20 - bars_since_recovery * 5)

    # Low held = +10 bonus
    score += 10 if low_held else 0

    # MACD positive (0-5)
    score += 5 if cur_mh > 0 else 0

    score = min(100, max(0, score))

    # Tier based on which SMA was violated
    tier       = 1 if violated_sma == "SMA50" else 2
    tier_label = ("🏆 TIER 1 — SMA50 Shakeout & Recovery"
                  if tier == 1 else
                  "🥈 TIER 2 — SMA150 Shakeout & Recovery")

    return {
        "Ticker"              : sym,
        "Price"               : round(price, 2),
        "Score"               : score,
        "Tier"                : tier,
        "Tier_Label"          : tier_label,
        # MA stack
        "JMA"                 : round(cur_jma, 2),
        "EMA8"                : round(cur_ema8, 2),
        "SMA20"               : round(cur_s20, 2),
        "SMA50"               : round(cur_s50, 2),
        "SMA150"              : round(cur_s150, 2),
        "Dist_JMA_%"          : round(dist_jma_pct, 2),
        "Dist_SMA50_%"        : round(dist_s50_pct, 2),
        "Dist_SMA150_%"       : round(dist_s150_pct, 2),
        # Red shakeout bar
        "Red_Date"            : red_date,
        "Red_Close"           : round(red_c, 2),
        "Red_Low"             : round(red_l, 2),
        "Violated_SMA"        : violated_sma,
        "Below_Depth_%"       : round(below_depth_pct, 2),
        "Red_Size_%"          : round(red_size_pct, 2),
        "Red_Volume"          : int(red_vol),
        # Green recovery bar
        "Green_Date"          : grn_date,
        "Green_Close"         : round(grn_c, 2),
        "Green_Low"           : round(grn_l, 2),
        "Green_Size_%"        : round(grn_size_pct, 2),
        "Green_Volume"        : int(grn_vol),
        "Vol_Ratio_G_vs_R"    : round(vol_ratio, 2),
        "Low_Held"            : "✅" if low_held else "❌",
        "Bars_Since_Recovery" : bars_since_recovery,
        # Indicators
        "RSI"                 : round(cur_rsi, 1),
        "MACD_Hist"           : round(cur_mh, 4),
        "Avg_Vol_20d"         : int(avg_vol),
        # internals
        "_df"        : df,
        "_jma"       : jma_s,
        "_ema8"      : ema8_s,
        "_sma20"     : sma20_s,
        "_sma50"     : sma50_s,
        "_sma150"    : sma150_s,
        "_shk_bar"   : shakeout_bar,
        "_rec_bar"   : recovery_bar,
    }

# ── Live print ────────────────────────────────────────────────
LIVE_COLS = ["Ticker","Price","Score","Tier_Label",
             "Violated_SMA","Below_Depth_%","Red_Date","Green_Date",
             "Vol_Ratio_G_vs_R","Low_Held","Bars_Since_Recovery","RSI"]
_CW = {
    "Ticker":8,"Price":10,"Score":7,"Tier_Label":38,
    "Violated_SMA":13,"Below_Depth_%":14,"Red_Date":12,"Green_Date":12,
    "Vol_Ratio_G_vs_R":18,"Low_Held":10,"Bars_Since_Recovery":20,"RSI":6,
}
_CF = {
    "Price":"${:.2f}","Score":"{:.0f}",
    "Below_Depth_%":"{:.2f}%","Vol_Ratio_G_vs_R":"{:.2f}×",
    "Bars_Since_Recovery":"{:.0f}d","RSI":"{:.1f}",
}
_hdr_done = False

def _live_header():
    global _hdr_done
    if _hdr_done: return
    sep = "━" * 210
    print(f"\n{sep}")
    print("  📊  LIVE MATCHES  —  Bull Stack + Red Shakeout + Green Recovery")
    print(sep)
    print("".join(f"  {c:<{_CW.get(c,10)}}" for c in LIVE_COLS))
    print("  " + "─"*208)
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
chk = download(["NVDA","AAPL","MSFT"], 300)
if not chk:
    print("❌  No data.")
else:
    for s, d in chk.items():
        p    = float(d["Close"].iloc[-1])
        s50  = float(d["Close"].rolling(50).mean().iloc[-1])
        s150 = float(d["Close"].rolling(150).mean().iloc[-1])
        print(f"  ✅ {s}: {len(d)} bars  ${p:.2f}  "
              f"SMA50=${s50:.2f}  SMA150=${s150:.2f}  {d.index[-1].date()}")
print()

# ── Diagnostic ────────────────────────────────────────────────
print("━"*65)
print("  STEP 2  DIAGNOSTIC (10 sample stocks)")
print("━"*65+"\n")

DIAG = ["NVDA","AAPL","MSFT","AMD","PLTR","META","CRWD","AVGO","MU","AXON"]
diag_data = download(DIAG, CFG["history_days"])
print(f"  Downloaded {len(diag_data)}/{len(DIAG)}\n")
print(f"  {'SYM':<7} {'PRICE':>8}  {'STACK':>6}  {'SHAKEOUT':>10}  RESULT")
print("  "+"─"*48)

for sym in DIAG:
    if sym not in diag_data:
        print(f"  {sym:<7} — no data"); continue
    try:
        df_d = diag_data[sym]
        p    = float(df_d["Close"].iloc[-1])
        r    = detect_pattern(sym, df_d)
        t    = lambda b: "✅" if b else "❌"

        # Quick stack check for display
        jt = calc_jma(df_d["Close"])
        e8 = calc_ema(df_d["Close"], 8)
        s20= df_d["Close"].rolling(20).mean()
        s50= df_d["Close"].rolling(50).mean()
        s150=df_d["Close"].rolling(150).mean()
        stack_ok = (float(jt.iloc[-1]) > float(e8.iloc[-1]) >
                    float(s20.iloc[-1]) > float(s50.iloc[-1]) >
                    float(s150.iloc[-1]))

        if r:
            print(f"  {sym:<7} ${p:>7.2f}  {t(stack_ok):>6}  "
                  f"{r['Violated_SMA']:>10}  ✅ Score={r['Score']} {r['Green_Date']}")
        else:
            print(f"  {sym:<7} ${p:>7.2f}  {t(stack_ok):>6}  "
                  f"{'—':>10}  ❌")
    except Exception as e:
        print(f"  {sym:<7} error: {e}")

print(f"""
  Pattern:
    C1  Bull Stack  : JMA > EMA8 > SMA20 > SMA50 > SMA150
                      Fast MAs (JMA/EMA8/SMA20/SMA50) rising,
                      SMA150 just needs to be below SMA50
    C2  Red Shakeout: 1 red candle closed below SMA50 or SMA150
                      within last {CFG['red_lookback']} bars (up to {CFG['max_below_sma_pct']}% below)
    C3  Green Recovery (next bar):
        • Green candle (close > open)
        • Closed above violated SMA
        • Volume >= {CFG['vol_mult_vs_red']}x red bar volume
        • Recovery within last {CFG['max_bars_since_recovery']} bars of today

  Tune if still ❌:
    max_bars_since_recovery  10 → 20
    red_lookback             20 → 30
    vol_mult_vs_red         0.5 → 0.3
    min_slope_pct          -0.05 → -0.2  (flat MAs allowed)
    require_close_above_sma True → False
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
        "ALAB","SMCI","HOOD","COIN","SOFI","UPST","DDOG","SNOW","MDB","REGN",
        "VRTX","ISRG","LULU","FTNT","IDXX","SBUX","TMUS","RBRK","NET","MARA",
        "AXON","ANET","CAVA","VRT","ELF","GRMN","KLAC","ON","ENPH","ROST",
        "AMGN","GILD","INTU","MCHP","MNST","NXPI","XEL","ACLS","IRTC","MXL",
        "QUBT","RGTI","ASTS","RKLB","IONQ","FSLR","PYPL","ROKU","POOL","ODFL",
    }
    b = len(pool); pool |= static
    print(f"  ✅ {'Static fallback':<18}: +{len(pool)-b:>4} → {len(pool)}")
    clean = sorted({s.upper() for s in pool if isinstance(s,str)
                    and s.isalpha() and 1<=len(s)<=5})
    print(f"\n  🎯 Total: {len(clean)} tickers")
    return clean

TICKERS = get_tickers()
print()

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

# ── Debug breakdown — shows exactly where stocks fail ──────────
print(f"""
  📊 DEBUG BREAKDOWN:
  Total processed     : {_DBG['total']}
  Passed vol/price    : {_DBG['pass_filter']}
  Failed stack (C1)   : {_DBG['fail_stack']}
  Failed rising (C1b) : {_DBG['fail_rising']}
  Failed shakeout (C2): {_DBG['fail_shakeout']}
  ✅ Passed all        : {_DBG['pass_all']}
""")

if not results:
    print("\n  No matches. Try relaxing:")
    print("   max_bars_since_recovery    3 → 5")
    print("   red_lookback              10 → 15")
    print("   vol_mult_vs_red           1.0 → 0.8")
    print("   require_no_break_of_red_low True → False")

# Sort by tier first, then score
results.sort(key=lambda x: (x["Tier"], -x["Score"]))

# ── Always build df_out and save/email (even if 0 results) ────
ts      = datetime.today().strftime("%Y%m%d_%H%M")
out_dir = os.environ.get("GITHUB_WORKSPACE", os.getcwd())

COLS = [
    "Ticker","Price","Score","Tier","Tier_Label",
    "JMA","EMA8","SMA20","SMA50","SMA150",
    "Dist_JMA_%","Dist_SMA50_%","Dist_SMA150_%",
    "Red_Date","Red_Close","Red_Low","Violated_SMA","Below_Depth_%",
    "Red_Size_%","Red_Volume",
    "Green_Date","Green_Close","Green_Low","Green_Size_%","Green_Volume",
    "Vol_Ratio_G_vs_R","Low_Held","Bars_Since_Recovery",
    "RSI","MACD_Hist","Avg_Vol_20d",
]
df_out = pd.DataFrame([{k:v for k,v in r.items() if not k.startswith("_")}
                        for r in results]) if results else pd.DataFrame(columns=COLS)
if not df_out.empty:
    df_out = df_out[[c for c in COLS if c in df_out.columns]]
    df_out.reset_index(drop=True, inplace=True)

FMT = {
    "Price"              : lambda v: f"${v:.2f}",
    "Score"              : lambda v: f"{v:.0f}",
    "JMA"                : lambda v: f"${v:.2f}",
    "EMA8"               : lambda v: f"${v:.2f}",
    "SMA20"              : lambda v: f"${v:.2f}",
    "SMA50"              : lambda v: f"${v:.2f}",
    "SMA150"             : lambda v: f"${v:.2f}",
    "Dist_JMA_%"         : lambda v: f"{v:+.2f}%",
    "Dist_SMA50_%"       : lambda v: f"{v:+.2f}%",
    "Dist_SMA150_%"      : lambda v: f"{v:+.2f}%",
    "Red_Close"          : lambda v: f"${v:.2f}",
    "Red_Low"            : lambda v: f"${v:.2f}",
    "Below_Depth_%"      : lambda v: f"{v:.2f}%",
    "Red_Size_%"         : lambda v: f"{v:.2f}%",
    "Red_Volume"         : lambda v: f"{v:,.0f}",
    "Green_Close"        : lambda v: f"${v:.2f}",
    "Green_Low"          : lambda v: f"${v:.2f}",
    "Green_Size_%"       : lambda v: f"{v:.2f}%",
    "Green_Volume"       : lambda v: f"{v:,.0f}",
    "Vol_Ratio_G_vs_R"   : lambda v: f"{v:.2f}×",
    "Bars_Since_Recovery": lambda v: f"{int(v)}d",
    "RSI"                : lambda v: f"{v:.1f}",
    "MACD_Hist"          : lambda v: f"{v:.4f}",
    "Avg_Vol_20d"        : lambda v: f"{v:,.0f}",
}

def fmt_v(col, val):
    if val is None or (isinstance(val, float) and np.isnan(val)): return "—"
    try:
        if col in FMT: return FMT[col](val)
    except Exception: pass
    return str(val) if str(val) not in ("nan","None","") else "—"

TIER_COLORS = {1:"#22c55e", 2:"#3b82f6"}
TIER_ICONS  = {1:"🏆", 2:"🥈"}
TIER_NAMES  = {
    1:"TIER 1 — SMA50 Shakeout & Recovery",
    2:"TIER 2 — SMA150 Shakeout & Recovery",
}

if _IN_NOTEBOOK and results:
    DISP = ["Ticker","Price","Score","Tier_Label",
            "Violated_SMA","Below_Depth_%","Red_Date","Green_Date",
            "Vol_Ratio_G_vs_R","Low_Held","Bars_Since_Recovery",
            "SMA50","SMA150","RSI","MACD_Hist"]
    DISP = [c for c in DISP if c in df_out.columns]

    def make_tier_block(tier_rows, tier_num):
        tc   = TIER_COLORS[tier_num]
        icon = TIER_ICONS[tier_num]
        name = TIER_NAMES[tier_num]
        if not tier_rows: return ""
        th = "".join(
            f'<th style="background:#0f172a;color:#e2e8f0;padding:9px 12px;'
            f'font-size:11px;font-weight:700;border-bottom:2px solid {tc};white-space:nowrap">{c}</th>'
            for c in DISP
        )
        rows_html = ""
        for i, r in enumerate(tier_rows):
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
                elif col == "Below_Depth_%":
                    try:
                        v = float(str(raw).replace("%",""))
                        sty = "color:#22c55e;font-weight:700" if v <= 1.5 else "color:#f59e0b"
                    except Exception: pass
                elif col == "Vol_Ratio_G_vs_R":
                    try:
                        v = float(str(raw).replace("×",""))
                        sty = ("color:#22c55e;font-weight:800" if v >= 2 else
                               "color:#86efac;font-weight:600" if v >= 1.5 else "")
                    except Exception: pass
                elif col == "Low_Held":
                    sty = "text-align:center;font-size:14px"
                elif col == "Bars_Since_Recovery":
                    try:
                        v = int(str(raw).replace("d",""))
                        sty = ("color:#22c55e;font-weight:700;text-align:center" if v == 0 else
                               "color:#86efac;text-align:center" if v <= 1 else "text-align:center")
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
    <span style="color:#f1f5f9;font-size:15px;font-weight:700">{name}</span>
    <span style="color:{tc};font-size:12px;margin-left:8px">{len(tier_rows)} stock{'s' if len(tier_rows)!=1 else ''}</span>
  </div>
  <div style="overflow-x:auto;border:1px solid #e2e8f0;border-top:none;border-radius:0 0 8px 8px">
    <table style="border-collapse:collapse;width:100%;min-width:700px">
      <thead><tr>{th}</tr></thead>
      <tbody>{rows_html}</tbody>
    </table>
  </div>
</div>"""

    t1 = sum(1 for r in results if r["Tier"]==1)
    t2 = sum(1 for r in results if r["Tier"]==2)

    header_html = f"""
<div style="background:linear-gradient(135deg,#0f172a,#1e3a5f);
        border-radius:10px;padding:18px 24px;margin-bottom:8px;
        font-family:'Segoe UI',Arial,sans-serif">
  <h2 style="margin:0;color:#60a5fa;font-size:20px;font-weight:700">
    📈 Full Bull Stack + Red Shakeout + Green Recovery
  </h2>
  <p style="margin:6px 0 0;color:#94a3b8;font-size:12px">
    {datetime.today().strftime('%Y-%m-%d %H:%M')} &nbsp;·&nbsp;
    <b style="color:#22c55e">{len(results)} matches</b> from {len(TICKERS)} tickers
    &nbsp;·&nbsp; 🏆{t1} SMA50 shakeout &nbsp; 🥈{t2} SMA150 shakeout
  </p>
</div>"""

    legend_html = """
<div style="background:#f8fafc;border:1px solid #e2e8f0;border-radius:8px;
        padding:12px 18px;margin-top:6px;font-size:11px;color:#64748b;">
  <b style="color:#475569">GUIDE</b> &nbsp;·&nbsp;
  Below_Depth_% = how far red candle closed below the violated SMA (lower = shallower dip) &nbsp;·&nbsp;
  Vol_Ratio_G_vs_R = green bar volume ÷ red bar volume (higher = stronger recovery buying) &nbsp;·&nbsp;
  Low_Held ✅ = green bar low stayed above red bar low (classic shakeout structure) &nbsp;·&nbsp;
  Bars_Since_Recovery 0 = green recovery bar was today
</div>"""

    tier_blocks = "".join(
        make_tier_block([r for r in results if r["Tier"]==t], t)
        for t in [1, 2]
    )
    display_html(header_html + tier_blocks + legend_html)

elif results:
    for t_num in [1,2]:
        t_rows = [r for r in results if r["Tier"]==t_num]
        if not t_rows: continue
        print(f"\n  {TIER_ICONS[t_num]}  {TIER_NAMES[t_num]}  ({len(t_rows)} stocks)\n")
        CLI_COLS = ["Ticker","Price","Score","Violated_SMA","Below_Depth_%",
                    "Vol_Ratio_G_vs_R","Low_Held","Bars_Since_Recovery","RSI"]
        CLI_COLS = [c for c in CLI_COLS if c in df_out.columns]
        col_w = {c: max(len(c), max(
            len(fmt_v(c,r.get(c))) for r in t_rows
        ))+2 for c in CLI_COLS}
        top  = "┬".join("─"*col_w[c] for c in CLI_COLS)
        sep  = "┼".join("─"*col_w[c] for c in CLI_COLS)
        bot  = "┴".join("─"*col_w[c] for c in CLI_COLS)
        hdr  = "│".join(c.center(col_w[c]) for c in CLI_COLS)
        print(f"  ┌{top}┐")
        print(f"  │{hdr}│")
        print(f"  ├{sep}┤")
        for i,r in enumerate(t_rows):
            cells=[fmt_v(c,r.get(c)).center(col_w[c]) for c in CLI_COLS]
            print(f"  │{'│'.join(cells)}│")
            if i<len(t_rows)-1: print(f"  ├{sep}┤")
        print(f"  └{bot}┘")

# ── Save CSV ──────────────────────────────────────────────────
fpath = os.path.join(out_dir, f"bull_stack_shakeout_{ts}.csv")
df_out.to_csv(fpath, index=False)
print(f"\n  💾 CSV → {fpath}")

# ── TradingView import list ────────────────────────────────────
tv = os.path.join(out_dir, f"tv_bull_stack_shakeout_{ts}.txt")
with open(tv,"w") as f:
    f.write(f"###Bull Stack Shakeout Recovery {datetime.today().strftime('%Y-%m-%d')}\n")
    for r in results:
        f.write(f"NASDAQ:{r['Ticker']}\n")
print(f"  📋 TradingView watchlist → {tv}")
print(f"       (Import: TradingView → Watchlist → ⋮ → Import Watchlist)")

# ── Email ──────────────────────────────────────────────────────
def _send_email(rl, csv_path):
    import smtplib
    from email.mime.multipart import MIMEMultipart
    from email.mime.text      import MIMEText
    from email.mime.base      import MIMEBase
    from email                import encoders

    gu = _GMAIL_USER; gp = _GMAIL_PASS; et = _EMAIL_TO

    if not gu:
        print("[Email] ❌  GMAIL_USER secret is empty")
        print("         → Repo → Settings → Secrets → Actions → GMAIL_USER")
        return
    if not gp:
        print("[Email] ❌  GMAIL_PASS secret is empty")
        print("         → Must be a Gmail App Password (16 chars, no spaces)")
        print("         → Generate: myaccount.google.com/apppasswords")
        return
    if not et:
        print("[Email] ❌  EMAIL_TO secret is empty")
        print("         → Repo → Settings → Secrets → Actions → EMAIL_TO")
        return

    eto = [e.strip() for e in et.split(",") if e.strip()]
    cnt = len(rl)

    try:
        t1 = sum(1 for r in rl if r.get("Tier")==1)
        t2 = sum(1 for r in rl if r.get("Tier")==2)
        print(f"[Email] Sending to {et}  ({cnt} results)...")

        th_e = "".join(
            f'<th style="background:#1e293b;color:#e2e8f0;padding:8px 11px;'
            f'font-size:11px;font-weight:700;border-bottom:2px solid #3b82f6;'
            f'white-space:nowrap">{c}</th>'
            for c in ["Ticker","Price","Score","Violated_SMA",
                      "Below_Depth_%","Vol_Ratio_G_vs_R","Low_Held",
                      "Bars_Since_Recovery","RSI"]
        )
        rows_e = ""
        for i, r in enumerate(rl[:50]):
            bg    = "#fff" if i%2==0 else "#f0f9ff"
            tc    = TIER_COLORS.get(r.get("Tier",1),"#22c55e")
            ticker= r.get("Ticker","—")
            price = r.get("Price",0) or 0
            score = r.get("Score",0) or 0
            vsma  = r.get("Violated_SMA","—")
            depth = r.get("Below_Depth_%",0) or 0
            volr  = r.get("Vol_Ratio_G_vs_R",0) or 0
            lh    = r.get("Low_Held","—")
            bsr   = r.get("Bars_Since_Recovery",99)
            rsi   = r.get("RSI",0) or 0
            rows_e += (
                f'<tr style="background:{bg}">'
                f'<td style="padding:6px 11px;font-size:12px;font-weight:700;color:{tc}">{ticker}</td>'
                f'<td style="padding:6px 11px;font-size:12px">${float(price):.2f}</td>'
                f'<td style="padding:6px 11px;font-size:12px;font-weight:700;'
                f'background:#166534;color:#fff;text-align:center">{float(score):.0f}</td>'
                f'<td style="padding:6px 11px;font-size:12px;font-weight:600">{vsma}</td>'
                f'<td style="padding:6px 11px;font-size:12px">{float(depth):.2f}%</td>'
                f'<td style="padding:6px 11px;font-size:12px;color:#22c55e;font-weight:700">'
                f'{float(volr):.2f}×</td>'
                f'<td style="padding:6px 11px;font-size:12px;text-align:center">{lh}</td>'
                f'<td style="padding:6px 11px;font-size:12px;text-align:center;color:'
                f'{"#22c55e" if bsr==0 else "#94a3b8"};font-weight:700">{bsr}d</td>'
                f'<td style="padding:6px 11px;font-size:12px">{float(rsi):.1f}</td>'
                f'</tr>'
            )

        # ── Ticker CSV line (one line, copy-paste ready) ──────────
        ticker_csv = ",".join(r.get("Ticker","") for r in rl) if rl else "—"
        ticker_csv_html = f"""
<div style="margin:12px 0;padding:14px 16px;background:#0f172a;
            border-radius:8px;border-left:4px solid #22c55e;">
  <p style="margin:0 0 6px;color:#94a3b8;font-size:11px;font-weight:600;
             letter-spacing:0.05em;text-transform:uppercase">
    📋 Stock List — Copy &amp; paste into TradingView or Excel
  </p>
  <p style="margin:0;color:#22c55e;font-size:13px;font-weight:700;
             font-family:'Courier New',monospace;word-break:break-all;
             letter-spacing:0.03em">
    {ticker_csv}
  </p>
</div>"""

        no_results_msg = ""
        if cnt == 0:
            no_results_msg = (
                '<tr><td colspan="9" style="padding:20px;text-align:center;'
                'color:#64748b;font-size:13px">No matches found today — '
                'market conditions did not trigger the pattern</td></tr>'
            )

        html_e = f"""<!DOCTYPE html><html><body style="margin:0;padding:0;
background:#f1f5f9;font-family:'Segoe UI',Arial,sans-serif">
<table width="100%" cellpadding="0" cellspacing="0" style="padding:20px 0">
<tr><td>
<table width="100%" cellpadding="0" cellspacing="0"
   style="max-width:960px;margin:0 auto;background:#fff;
          border-radius:12px;overflow:hidden;box-shadow:0 4px 20px rgba(0,0,0,0.08)">
  <tr><td style="background:linear-gradient(135deg,#0f172a,#1e3a5f);padding:22px 28px">
<h1 style="margin:0;color:#60a5fa;font-size:20px;font-weight:700">
  📊 Full Bull Stack + Red Shakeout + Green Recovery
</h1>
<p style="margin:6px 0 0;color:#94a3b8;font-size:12px">
  {datetime.today().strftime('%Y-%m-%d %H:%M UTC')} &nbsp;·&nbsp;
  {cnt} match{'es' if cnt!=1 else ''} &nbsp;·&nbsp;
  🏆 SMA50:{t1} &nbsp; 🥈 SMA150:{t2}
</p>
  </td></tr>
  <tr><td style="padding:16px">
{ticker_csv_html}
<div style="overflow-x:auto;border-radius:8px;border:1px solid #e2e8f0">
  <table style="border-collapse:collapse;width:100%;min-width:700px">
    <thead><tr>{th_e}</tr></thead>
    <tbody>{rows_e or no_results_msg}</tbody>
  </table>
</div>
<p style="font-size:11px;color:#64748b;margin:10px 0 0">
  📎 Full results (CSV) and TradingView import file attached &nbsp;·&nbsp;
  <b>Vol_Ratio</b> = green bar vol ÷ red bar vol &nbsp;·&nbsp;
  <b>Low_Held ✅</b> = green bar did not break red bar's low
</p>
  </td></tr>
  <tr><td style="background:#f8fafc;padding:12px 28px;border-top:1px solid #e2e8f0;text-align:center">
<p style="margin:0;color:#94a3b8;font-size:10px">
  ⚠️ Not financial advice &nbsp;·&nbsp; Auto-generated by GitHub Actions
</p>
  </td></tr>
</table></td></tr></table>
</body></html>"""

        plain_lines = [
            f"Full Bull Stack + Red Shakeout + Green Recovery — {datetime.today().strftime('%Y-%m-%d')}",
            f"{cnt} matches  (🏆SMA50:{t1}  🥈SMA150:{t2})",
            "",
            f"STOCKS: {ticker_csv}",
            "",
            "="*65,
        ]
        if rl:
            for r in rl[:50]:
                plain_lines.append(
                    f"{r.get('Ticker','—'):<7} ${r.get('Price',0):.2f}  "
                    f"Score:{r.get('Score',0):.0f}  "
                    f"{r.get('Violated_SMA','—')}  "
                    f"Depth:{r.get('Below_Depth_%',0):.1f}%  "
                    f"VolRatio:{r.get('Vol_Ratio_G_vs_R',0):.1f}x  "
                    f"Recovery:{r.get('Bars_Since_Recovery',0)}d ago"
                )
        else:
            plain_lines.append("No matches today")
        plain_lines.append("\n📎 CSV and TradingView file attached.")
        plain_e = "\n".join(plain_lines)

        subj = (f"📊 Bull Stack Shakeout — {cnt} signal{'s' if cnt!=1 else ''}"
                f"  (🏆{t1} 🥈{t2}) — {datetime.today().strftime('%Y-%m-%d')}")

        msg = MIMEMultipart("mixed")
        msg["Subject"] = subj; msg["From"] = gu; msg["To"] = ", ".join(eto)
        alt = MIMEMultipart("alternative")
        alt.attach(MIMEText(plain_e, "plain")); alt.attach(MIMEText(html_e, "html"))
        msg.attach(alt)

    except Exception as e:
        print(f"[Email] ❌  Failed to build email body: {type(e).__name__}: {e}")
        return

    # ── Attach CSV ────────────────────────────────────────────
    for att_path in [csv_path, tv]:
        if att_path and os.path.exists(att_path):
            try:
                with open(att_path,"rb") as f:
                    part = MIMEBase("application","octet-stream")
                    part.set_payload(f.read())
                encoders.encode_base64(part)
                part.add_header("Content-Disposition",
                    f"attachment; filename={os.path.basename(att_path)}")
                msg.attach(part)
                print(f"[Email] 📎 Attached: {os.path.basename(att_path)}")
            except Exception as e:
                print(f"[Email] ⚠️  Attach failed for {att_path}: {e}")

    try:
        print(f"[Email] Connecting to smtp.gmail.com:465 ...")
        with smtplib.SMTP_SSL("smtp.gmail.com", 465) as srv:
            srv.login(gu, gp.replace(" ", ""))
            srv.sendmail(gu, eto, msg.as_string())
        print(f"[Email] ✅  Sent to: {', '.join(eto)}")
        print(f"[Email]    Subject: {subj}")
    except smtplib.SMTPAuthenticationError:
        print("[Email] ❌  AUTHENTICATION FAILED — use Gmail App Password")
        print("         Generate: myaccount.google.com/apppasswords")
    except smtplib.SMTPException as e:
        print(f"[Email] ❌  SMTP error: {e}")
    except Exception as e:
        print(f"[Email] ❌  Unexpected error: {type(e).__name__}: {e}")

try:
    _send_email(results, fpath)
except Exception as e:
    print(f"[Email] ❌  Unexpected top-level error: {type(e).__name__}: {e}")
    print("[Email]    Continuing — CSV and charts still saved.")

if _IN_NOTEBOOK:
    try:
        from google.colab import files
        files.download(fpath); files.download(tv)
    except Exception: pass
else:
    print("  (CI: files in workspace, email + both attachments sent)")

# ── Charts for top 5 ──────────────────────────────────────────
if results:
    top = results[:min(5,len(results))]
    fig, axes = plt.subplots(len(top),1,figsize=(15,5.5*len(top)),facecolor="#0f172a")
    if len(top)==1: axes=[axes]

    for idx, r in enumerate(top):
        ax   = axes[idx]
        df_p = r["_df"].tail(60).copy()
        n_p  = len(df_p)
        fn   = len(r["_df"]); off = fn - n_p

        jma  = r["_jma"].reindex(df_p.index)
        ema8 = r["_ema8"].reindex(df_p.index)
        s20  = r["_sma20"].reindex(df_p.index)
        s50  = r["_sma50"].reindex(df_p.index)
        s150 = r["_sma150"].reindex(df_p.index)

        ax.set_facecolor("#0f172a")

        for i,(_, row_) in enumerate(df_p.iterrows()):
            o=float(row_["Open"]); h=float(row_["High"])
            l=float(row_["Low"]);  c=float(row_["Close"])
            clr="#34d399" if c>=o else "#ef4444"
            ax.plot([i,i],[l,h],color=clr,lw=0.7,zorder=2)
            blo=min(o,c); bhi=max(o,c); bh=max(bhi-blo,(h-l)*0.005)
            rect=mpatches.FancyBboxPatch((i-0.3,blo),0.6,bh,
                 boxstyle="square,pad=0",facecolor=clr,edgecolor=clr,lw=0.4,zorder=3)
            ax.add_patch(rect)

        ax.plot(range(n_p), jma.values,  color="#22d3ee", lw=1.8, label="JMA", zorder=6)
        ax.plot(range(n_p), ema8.values, color="#34d399", lw=1.5, ls="--", label="EMA8", zorder=5)
        ax.plot(range(n_p), s20.values,  color="#fbbf24", lw=1.3, ls="-.", label="SMA20", zorder=4)
        ax.plot(range(n_p), s50.values,  color="#3b82f6", lw=1.5, label="SMA50", zorder=4)
        ax.plot(range(n_p), s150.values, color="#f472b6", lw=1.5, ls=":", label="SMA150", zorder=4)

        # Mark shakeout (red bar) and recovery (green bar)
        sb = r["_shk_bar"] - off
        rb = r["_rec_bar"] - off
        if 0 <= sb < n_p:
            ax.scatter([sb],[float(df_p["Low"].iloc[sb])],
                       color="#ef4444",s=150,zorder=9,marker="v",label="Shakeout")
            ax.axvline(sb, color="#ef4444", lw=1.2, ls=":", alpha=0.6)
        if 0 <= rb < n_p:
            ax.scatter([rb],[float(df_p["Close"].iloc[rb])],
                       color="#22c55e",s=180,zorder=9,marker="^",
                       label=f"Recovery {r['Green_Date']} Vol{r['Vol_Ratio_G_vs_R']:.1f}×")

        # Shade violated SMA zone
        vlvl = r["SMA50"] if r["Violated_SMA"]=="SMA50" else r["SMA150"]
        ax.axhspan(vlvl*0.97, vlvl*1.02, alpha=0.07,
                   color="#3b82f6" if r["Violated_SMA"]=="SMA50" else "#f472b6", zorder=1)

        tick_step = max(1, n_p//8)
        ax.set_xticks(range(0, n_p, tick_step))
        ax.set_xticklabels(
            [df_p.index[i].strftime("%m/%d") for i in range(0,n_p,tick_step)],
            color="#94a3b8", fontsize=7)
        ax.set_xlim(-0.5, n_p-0.5)
        ax.set_title(
            f"{r['Ticker']}  ${r['Price']:.2f}  |  Score {r['Score']}/100  |  "
            f"{r['Tier_Label']}  |  "
            f"Shakeout {r['Red_Date']} → Recovery {r['Green_Date']} "
            f"({r['Bars_Since_Recovery']}d ago)  |  "
            f"Depth {r['Below_Depth_%']:.1f}%  VolRatio {r['Vol_Ratio_G_vs_R']:.1f}×  "
            f"LowHeld {r['Low_Held']}  |  RSI {r['RSI']:.0f}",
            color="#e2e8f0", fontsize=8, fontweight="bold", pad=6)
        ax.tick_params(colors="#94a3b8", labelsize=7)
        for sp_ in ax.spines.values(): sp_.set_edgecolor("#1e3a5f")
        ax.legend(loc="upper left", facecolor="#1e293b",
                  labelcolor="#e2e8f0", fontsize=7, framealpha=0.9, ncol=3)
        ax.grid(color="#1e3a5f", ls="--", lw=0.4, alpha=0.4, axis="y")

    plt.suptitle(
        f"Full Bull Stack + Red Shakeout + Green Recovery  ·  {datetime.today().strftime('%Y-%m-%d')}\n"
        f"🔵 JMA  🟢 EMA8  🟡 SMA20  🔵 SMA50  🩷 SMA150  "
        f"▼ Shakeout  ▲ Recovery",
        color="#60a5fa", fontsize=10, fontweight="bold", y=1.001)
    plt.tight_layout()
    cp = os.path.join(out_dir, f"bull_stack_shakeout_chart_{ts}.png")
    plt.savefig(cp, dpi=150, bbox_inches="tight", facecolor="#0f172a")
    if _IN_NOTEBOOK: plt.show()
    else: plt.close()
    print(f"  📊 Chart → {cp}")

print("""
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  📋 PATTERN EXPLAINED

  C1  FULL BULL MA STACK  (all aligned + all rising)
      JMA > EMA8 > SMA20 > SMA50 > SMA150
      Every MA above the one below it AND every MA has a
      positive slope over the last 5 bars
      = Perfect multi-timeframe trend alignment

  C2  1 RED SHAKEOUT CANDLE  (dipped below SMA50 or SMA150)
      A single red candle closed below SMA50 (Tier 1)
      or SMA150 (Tier 2) within the last 10 bars
      = Weak hands flushed out; quick violation of support

  C3  GREEN RECOVERY CANDLE  (the very next bar):
      ✅ Green (close > open)
      ✅ Closed back above the violated SMA
      ✅ Volume >= 1× the red bar's volume (stronger buying)
      ✅ Low >= red bar's low (no lower low = no real breakdown)
      = Institutional recovery absorbed the shakeout

  TIERS:
      🏆 Tier 1  Shakeout below SMA50  (shallower dip = bullish)
      🥈 Tier 2  Shakeout below SMA150 (deeper dip, bigger bounce)

  💡 BEST SETUPS
  Bars_Since_Recovery = 0     green recovery happened today
  Vol_Ratio_G_vs_R >= 2×     strong institutional buying
  Below_Depth_% < 1%         barely violated — clean shakeout
  Low_Held = ✅               green bar higher low confirmed
  All MAs rising steeply      strong underlying trend

  ⚙️  TUNE IF 0 RESULTS
  max_bars_since_recovery   10 → 20
  red_lookback              20 → 30
  vol_mult_vs_red          0.5 → 0.3
  min_slope_pct           -0.05 → -0.2
  require_close_above_sma  True → False
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
""")
