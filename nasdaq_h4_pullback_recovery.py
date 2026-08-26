# ============================================================
# NASDAQ — Declining H4 + SMA Near H4 + Gentle Pullback + Green Recovery
# ============================================================
#
# EXACT PATTERN (4 sequential steps from chart):
#
#  STEP 1 — DECLINING MONTHLY CAMARILLA H4
#      This month's H4 < Previous month's H4
#      H4 = Close + (High-Low) × 1.1/2  (from prior month)
#      = The pivot resistance level is contracting downward,
#        building energy for a bigger breakout when it fires
#
#  STEP 2 — SMA50 AND SMA150 BOTH NEAR MONTHLY H4
#      Both SMA50 and SMA150 are within sma_h4_zone_pct%
#      of the current month's H4 level
#      = The MAs and the Camarilla H4 are all clustered
#        together — a major confluence support/resistance zone
#
#  STEP 3 — PRICE CROSSED ABOVE BOTH SMAs AND H4
#      Within cross_lookback bars, price broke above
#      SMA50, SMA150, and H4 simultaneously
#      THEN pulled back gently toward EMA8/JMA:
#        - Pullback bars were RED candles
#        - Each red candle declined <= max_red_candle_pct%
#          (not more than 4% per day)
#        - Price came close to EMA8/JMA (within ema_zone_pct%)
#        - Volume on red candles was DECLINING (drying up)
#
#  STEP 4 — GREEN CANDLE CLOSED ABOVE JMA AND EMA8
#      The most recent (or very recent) candle is GREEN
#      and closes above BOTH JMA and EMA8
#      Volume on this green candle > volume on the
#      immediately preceding red candle
#      = Buyers stepped back in at the EMA8/JMA support
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
import requests, time, warnings, io, smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text      import MIMEText
from email.mime.base      import MIMEBase
from email                import encoders
from datetime             import datetime, timedelta
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
    print(f"  ⚠️  Missing: {', '.join(missing)}")
    print(f"  ℹ️  GitHub → Settings → Secrets → Actions")
    print(f"  ℹ️  Email will be SKIPPED this run")
print("━"*65)
print()

# ── CONFIG ────────────────────────────────────────────────────
CFG = {
    "history_days"              : 400,

    # ── MA periods ────────────────────────────────────────────
    "jma_period"                : 13,
    "jma_phase"                 : 40,
    "ema8_period"               : 8,
    "sma50_period"              : 50,
    "sma150_period"             : 150,

    # ── STEP 1: Declining H4 ──────────────────────────────────
    # This month's H4 must be < last month's H4
    "require_declining_h4"      : True,

    # ── STEP 2: SMA50 + SMA150 near H4 ───────────────────────
    # Both SMAs must be within this % of H4
    "sma_h4_zone_pct"           : 8.0,

    # ── STEP 3: Cross above SMAs + H4, then gentle pullback ───
    # How many bars back to find the original cross
    "cross_lookback"            : 20,
    # Max single-candle RED decline (body % of open)
    "max_red_candle_pct"        : 4.0,
    # How many consecutive red pullback bars to look for
    "min_pullback_bars"         : 1,
    "max_pullback_bars"         : 10,
    # Price must come within this % of EMA8 or JMA during pullback
    "ema_zone_pct"              : 3.0,
    # Volume during pullback must be declining (each bar <= prev)
    "require_declining_pb_vol"  : True,

    # ── STEP 4: Green candle above JMA and EMA8 ──────────────
    # Green candle volume > last red candle volume
    "require_vol_gt_last_red"   : True,
    # How recent the green candle must be (bars from today)
    "max_bars_since_green"      : 3,

    # ── Filters ───────────────────────────────────────────────
    "vol_avg_bars"              : 20,
    "min_avg_volume"            : 50_000,
    "min_price"                 : 1.0,

    "batch_size"                : 50,
    "batch_sleep"               : 1.5,
}

# ── Indicators ───────────────────────────────────────────────
def calc_jma(series, period=13, phase=40):
    n = len(series); vals = series.values.astype(float)
    result = np.full(n, np.nan)
    first = next((i for i in range(n) if not np.isnan(vals[i])), 0)
    phase_ratio = phase / 100.0 + 1.5
    alpha = 2.0 / (period + 1.0); beta = alpha * phase_ratio
    e0 = e1 = e2 = vals[first]; result[first] = e0
    for i in range(first + 1, n):
        v = vals[i]
        e0 = (1 - alpha) * e0 + alpha * v
        e1 = (v - e0) * (1 - beta) + beta * e1
        e2 = (e0 + e1 - e2) * alpha + (1 - alpha) * e2
        result[i] = e2
    return pd.Series(result, index=series.index)

def calc_ema(series, period):
    return series.ewm(span=period, adjust=False).mean()

def calc_rsi(close, period=14):
    delta = close.diff()
    gain  = delta.clip(lower=0); loss = -delta.clip(upper=0)
    ag    = gain.ewm(alpha=1/period, adjust=False).mean()
    al    = loss.ewm(alpha=1/period, adjust=False).mean()
    rs    = ag / al.replace(0, np.nan)
    return 100 - (100 / (1 + rs))

def calc_macd_hist(close, fast=12, slow=26, signal=9):
    ef = calc_ema(close, fast); es = calc_ema(close, slow)
    ml = ef - es; sig = calc_ema(ml, signal)
    return ml - sig

# ── Camarilla H4 ─────────────────────────────────────────────
def cam_h4(high, low, close):
    """H4 = Close + (High - Low) × 1.1 / 2"""
    return float(close) + (float(high) - float(low)) * 1.1 / 2.0

def get_monthly_h4_levels(df):
    """
    Returns dict {period: H4_value} for recent months.
    H4 for month M = built from prior month M-1 H/L/C.
    """
    try:
        df = df.copy(); df.index = pd.to_datetime(df.index)
        periods = df.index.to_period("M")
        unique_months = sorted(periods.unique())
        month_h4 = {}
        for i, mp in enumerate(unique_months):
            if i == 0: continue
            sub = df[periods == unique_months[i-1]]
            if len(sub) < 5: continue
            hi = float(sub["High"].max())
            lo = float(sub["Low"].min())
            cl = float(sub["Close"].iloc[-1])
            month_h4[mp] = round(cam_h4(hi, lo, cl), 4)
        return month_h4
    except Exception:
        return {}

def build_h4_series(df):
    """Per-bar H4 series."""
    try:
        df = df.copy(); df.index = pd.to_datetime(df.index)
        periods = df.index.to_period("M")
        month_h4 = get_monthly_h4_levels(df)
        vals = [month_h4.get(mp, np.nan) for mp in periods]
        return pd.Series(vals, index=df.index)
    except Exception:
        return pd.Series(np.full(len(df), np.nan), index=df.index)

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

# ── Debug counters ────────────────────────────────────────────
_DBG = {
    "total"         : 0,
    "pass_filter"   : 0,
    "fail_s1_h4"    : 0,   # Step 1: H4 not declining
    "fail_s2_sma"   : 0,   # Step 2: SMAs not near H4
    "fail_s3_cross" : 0,   # Step 3: no cross found
    "fail_s3_pb"    : 0,   # Step 3: pullback conditions not met
    "fail_s4_green" : 0,   # Step 4: no qualifying green candle
    "pass_all"      : 0,
}

# ── Core detection ────────────────────────────────────────────
def detect_pattern(sym, df):
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

    # ── Compute indicators ────────────────────────────────────
    jma_s  = calc_jma(df["Close"], CFG["jma_period"], CFG["jma_phase"])
    ema8_s = calc_ema(df["Close"], CFG["ema8_period"])
    s50_s  = df["Close"].rolling(CFG["sma50_period"]).mean()
    s150_s = df["Close"].rolling(CFG["sma150_period"]).mean()
    h4_s   = build_h4_series(df)
    rsi_s  = calc_rsi(df["Close"])
    mh_s   = calc_macd_hist(df["Close"])

    cur_jma  = float(jma_s.iloc[-1])  if not np.isnan(jma_s.iloc[-1])  else np.nan
    cur_ema8 = float(ema8_s.iloc[-1]) if not np.isnan(ema8_s.iloc[-1]) else np.nan
    cur_s50  = float(s50_s.iloc[-1])  if not np.isnan(s50_s.iloc[-1])  else np.nan
    cur_s150 = float(s150_s.iloc[-1]) if not np.isnan(s150_s.iloc[-1]) else np.nan
    cur_h4   = float(h4_s.iloc[-1])   if not np.isnan(h4_s.iloc[-1])   else np.nan
    cur_rsi  = float(rsi_s.iloc[-1])  if not np.isnan(rsi_s.iloc[-1])  else 50
    cur_mh   = float(mh_s.iloc[-1])   if not np.isnan(mh_s.iloc[-1])   else 0

    if any(np.isnan([cur_jma, cur_ema8, cur_s50, cur_s150])): return None
    if np.isnan(cur_h4): return None

    # ─────────────────────────────────────────────────────────
    # STEP 1: THIS MONTH'S H4 < PREVIOUS MONTH'S H4
    # ─────────────────────────────────────────────────────────
    month_h4_dict = get_monthly_h4_levels(df)
    today = pd.Timestamp.today().normalize()
    cur_period  = today.to_period("M")
    prev_period = cur_period - 1
    prev2_period= cur_period - 2

    h4_cur  = month_h4_dict.get(cur_period,  None)
    h4_prev = month_h4_dict.get(prev_period, None)
    h4_prev2= month_h4_dict.get(prev2_period,None)

    if h4_cur is None or h4_prev is None:
        _DBG["fail_s1_h4"] += 1; return None

    if CFG["require_declining_h4"] and h4_cur >= h4_prev:
        _DBG["fail_s1_h4"] += 1; return None

    h4_decline_pct = round((h4_prev - h4_cur) / h4_prev * 100, 2)

    # ─────────────────────────────────────────────────────────
    # STEP 2: SMA50 AND SMA150 BOTH NEAR H4
    # Both must be within sma_h4_zone_pct% of current H4
    # ─────────────────────────────────────────────────────────
    zone = CFG["sma_h4_zone_pct"] / 100
    dist_s50_h4  = abs(cur_s50  - cur_h4) / cur_h4 if cur_h4 > 0 else np.inf
    dist_s150_h4 = abs(cur_s150 - cur_h4) / cur_h4 if cur_h4 > 0 else np.inf

    if dist_s50_h4 > zone or dist_s150_h4 > zone:
        _DBG["fail_s2_sma"] += 1; return None

    # ─────────────────────────────────────────────────────────
    # STEP 3: FIND THE CROSS + GENTLE PULLBACK
    #
    # Search cross_lookback bars back for a point where price
    # crossed above SMA50, SMA150, AND H4.
    # After the cross, scan for a pullback phase:
    #   - Red candles (close < open)
    #   - Each red candle body <= max_red_candle_pct% of open
    #   - Price came within ema_zone_pct% of EMA8 or JMA
    #   - Volume declining during pullback
    # ─────────────────────────────────────────────────────────
    cl = CFG["cross_lookback"]
    cross_bar = None

    # Find the most recent bar where price crossed above
    # SMA50 + SMA150 + H4 (was below at least one, now above all)
    for i in range(max(1, n - cl), n):
        pc   = float(df["Close"].iloc[i])
        s50i = float(s50_s.iloc[i])  if not np.isnan(s50_s.iloc[i])  else np.nan
        s15i = float(s150_s.iloc[i]) if not np.isnan(s150_s.iloc[i]) else np.nan
        h4i  = float(h4_s.iloc[i])   if not np.isnan(h4_s.iloc[i])   else np.nan
        if any(np.isnan([s50i, s15i, h4i])): continue
        # current bar: above all three
        if not (pc > s50i and pc > s15i and pc > h4i): continue
        # previous bar: was below at least one
        pp   = float(df["Close"].iloc[i-1])
        s50p = float(s50_s.iloc[i-1]) if not np.isnan(s50_s.iloc[i-1]) else np.nan
        s15p = float(s150_s.iloc[i-1]) if not np.isnan(s150_s.iloc[i-1]) else np.nan
        h4p  = float(h4_s.iloc[i-1])  if not np.isnan(h4_s.iloc[i-1])  else np.nan
        if any(np.isnan([s50p, s15p, h4p])): continue
        if pp < s50p or pp < s15p or pp < h4p:
            # Valid cross found — keep most recent
            if cross_bar is None or i > cross_bar:
                cross_bar = i

    if cross_bar is None:
        _DBG["fail_s3_cross"] += 1; return None

    # ── Analyse pullback after the cross ──────────────────────
    # Scan bars AFTER the cross bar up to today
    pb_start = cross_bar + 1
    pb_end   = n  # inclusive of today

    # Find consecutive red candle sequence that brought price
    # close to EMA8/JMA, with declining volume
    mr = CFG["max_red_candle_pct"] / 100
    ez = CFG["ema_zone_pct"] / 100
    min_pb = CFG["min_pullback_bars"]
    max_pb = CFG["max_pullback_bars"]

    pullback_found  = False
    pullback_start  = None
    pullback_end    = None
    touched_ema_jma = False
    vol_declining   = True

    # Look for a sequence of red candles after the cross
    i = pb_start
    while i < pb_end:
        o_i = float(df["Open"].iloc[i])
        c_i = float(df["Close"].iloc[i])
        v_i = float(df["Volume"].iloc[i])
        j_i = float(jma_s.iloc[i])  if not np.isnan(jma_s.iloc[i])  else np.nan
        e_i = float(ema8_s.iloc[i]) if not np.isnan(ema8_s.iloc[i]) else np.nan

        # Must start with a red candle
        if c_i >= o_i:
            i += 1; continue

        # Check this red candle body size
        body_pct = (o_i - c_i) / o_i if o_i > 0 else 0
        if body_pct > mr:
            i += 1; continue   # too big a drop — not gentle pullback

        # Start of a potential pullback sequence
        seq_start = i
        seq_end   = i
        seq_vols  = [v_i]
        touched   = False

        if not np.isnan(j_i) and not np.isnan(e_i):
            d_jma  = abs(c_i - j_i) / j_i if j_i > 0 else np.inf
            d_ema8 = abs(c_i - e_i) / e_i if e_i > 0 else np.inf
            if d_jma <= ez or d_ema8 <= ez:
                touched = True

        # Extend the sequence
        j2 = i + 1
        while j2 < pb_end and j2 - seq_start < max_pb:
            o2 = float(df["Open"].iloc[j2])
            c2 = float(df["Close"].iloc[j2])
            v2 = float(df["Volume"].iloc[j2])
            j2_jma = float(jma_s.iloc[j2])  if not np.isnan(jma_s.iloc[j2])  else np.nan
            j2_ema = float(ema8_s.iloc[j2]) if not np.isnan(ema8_s.iloc[j2]) else np.nan

            if c2 >= o2: break   # green candle — end of pullback

            body2 = (o2 - c2) / o2 if o2 > 0 else 0
            if body2 > mr: break   # too large a drop

            seq_end = j2
            seq_vols.append(v2)

            if not np.isnan(j2_jma) and not np.isnan(j2_ema):
                d_jma2 = abs(c2 - j2_jma) / j2_jma if j2_jma > 0 else np.inf
                d_e2   = abs(c2 - j2_ema) / j2_ema  if j2_ema  > 0 else np.inf
                if d_jma2 <= ez or d_e2 <= ez:
                    touched = True

            j2 += 1

        seq_len = seq_end - seq_start + 1
        if seq_len < min_pb:
            i = j2; continue

        # Check volume declining in sequence
        vol_ok = True
        if CFG["require_declining_pb_vol"] and len(seq_vols) > 1:
            for vi in range(1, len(seq_vols)):
                if seq_vols[vi] > seq_vols[vi-1] * 1.1:   # allow 10% tolerance
                    vol_ok = False; break

        if touched:
            pullback_found = True
            pullback_start = seq_start
            pullback_end   = seq_end
            touched_ema_jma= True
            vol_declining  = vol_ok

        i = j2

    if not pullback_found:
        _DBG["fail_s3_pb"] += 1; return None

    # ─────────────────────────────────────────────────────────
    # STEP 4: GREEN CANDLE ABOVE JMA AND EMA8
    # After the pullback, look for a green candle that:
    #   a) Is GREEN (close > open)
    #   b) Close > JMA and close > EMA8
    #   c) Volume > previous red candle volume
    # Must be within max_bars_since_green of today
    # ─────────────────────────────────────────────────────────
    mg = CFG["max_bars_since_green"]
    green_bar  = None
    green_date = None

    # Search from pullback_end+1 onward (and also pullback_end
    # itself if it flipped green)
    search_from = pullback_end  # could be the recovery started on last bar

    for i in range(max(search_from, n - mg - 1), n):
        o_i = float(df["Open"].iloc[i])
        c_i = float(df["Close"].iloc[i])
        v_i = float(df["Volume"].iloc[i])
        j_i = float(jma_s.iloc[i])  if not np.isnan(jma_s.iloc[i])  else np.nan
        e_i = float(ema8_s.iloc[i]) if not np.isnan(ema8_s.iloc[i]) else np.nan

        # a) Must be green
        if c_i <= o_i: continue
        # b) Close above JMA and EMA8
        if np.isnan(j_i) or np.isnan(e_i): continue
        if not (c_i > j_i and c_i > e_i): continue
        # c) Volume > previous bar's volume (the last red candle)
        if i > 0:
            prev_vol = float(df["Volume"].iloc[i-1])
            if CFG["require_vol_gt_last_red"] and v_i <= prev_vol: continue
        # Recency
        if n - 1 - i > mg: continue

        if green_bar is None or i > green_bar:
            green_bar  = i
            green_date = df.index[i]

    if green_bar is None:
        _DBG["fail_s4_green"] += 1; return None

    _DBG["pass_all"] += 1

    # ── Metrics ───────────────────────────────────────────────
    bars_since_cross = n - 1 - cross_bar
    bars_since_green = n - 1 - green_bar
    pb_len = pullback_end - pullback_start + 1

    # Average red candle size during pullback
    red_bodies = []
    for i in range(pullback_start, pullback_end + 1):
        o = float(df["Open"].iloc[i]); c = float(df["Close"].iloc[i])
        if c < o: red_bodies.append((o-c)/o*100)
    avg_red_pct = round(np.mean(red_bodies), 2) if red_bodies else 0

    # Volume ratio: green candle vs last red candle
    grn_vol = float(df["Volume"].iloc[green_bar])
    red_vol = float(df["Volume"].iloc[green_bar-1]) if green_bar > 0 else grn_vol
    vol_ratio = grn_vol / red_vol if red_vol > 0 else 0

    dist_jma_pct   = (price - cur_jma)  / cur_jma  * 100 if cur_jma  > 0 else 0
    dist_ema8_pct  = (price - cur_ema8) / cur_ema8 * 100 if cur_ema8 > 0 else 0
    dist_h4_pct    = (price - cur_h4)   / cur_h4   * 100 if cur_h4   > 0 else 0
    dist_s50_h4    = round(dist_s50_h4  * 100, 2)
    dist_s150_h4   = round(dist_s150_h4 * 100, 2)

    # Score (0-100)
    score = 0
    # H4 decline magnitude (0-20)
    score += min(20, int(h4_decline_pct * 4))
    # Pullback quality: gentle + touched EMA8/JMA (0-20)
    score += 15 if touched_ema_jma else 5
    score += 5  if vol_declining else 0
    # Green candle freshness (0-20)
    score += max(0, 20 - bars_since_green * 5)
    # Vol ratio: green vs red (0-20)
    score += min(20, int(vol_ratio * 8))
    # SMA/H4 proximity (0-15): tighter = stronger cluster
    avg_sma_h4_dist = (abs(dist_s50_h4) + abs(dist_s150_h4)) / 2
    score += max(0, 15 - int(avg_sma_h4_dist * 2))
    # RSI above 50 (0-5)
    score += 5 if cur_rsi > 50 else 0
    score = min(100, max(0, score))

    return {
        "Ticker"           : sym,
        "Price"            : round(price, 2),
        "Score"            : score,
        # Step 1
        "H4_This"          : round(h4_cur, 2),
        "H4_Prev"          : round(h4_prev, 2),
        "H4_Decline_%"     : h4_decline_pct,
        # Step 2
        "SMA50"            : round(cur_s50, 2),
        "SMA150"           : round(cur_s150, 2),
        "Cam_H4"           : round(cur_h4, 2),
        "Dist_S50_H4_%"    : dist_s50_h4,
        "Dist_S150_H4_%"   : dist_s150_h4,
        # Step 3
        "Cross_Ago"        : bars_since_cross,
        "PB_Bars"          : pb_len,
        "Avg_Red_%"        : avg_red_pct,
        "Touched_EMA_JMA"  : "✅" if touched_ema_jma else "—",
        "Vol_Declining"    : "✅" if vol_declining else "—",
        # Step 4
        "Green_Date"       : green_date.strftime("%Y-%m-%d"),
        "Green_Ago"        : bars_since_green,
        "Vol_Green_vs_Red" : round(vol_ratio, 2),
        # Current levels
        "JMA"              : round(cur_jma, 2),
        "EMA8"             : round(cur_ema8, 2),
        "Dist_JMA_%"       : round(dist_jma_pct, 2),
        "Dist_EMA8_%"      : round(dist_ema8_pct, 2),
        "Dist_H4_%"        : round(dist_h4_pct, 2),
        "RSI"              : round(cur_rsi, 1),
        "MACD_Hist"        : round(cur_mh, 4),
        "Avg_Vol_20d"      : int(avg_vol),
        # internals
        "_df"          : df,
        "_jma"         : jma_s,
        "_ema8"        : ema8_s,
        "_s50"         : s50_s,
        "_s150"        : s150_s,
        "_h4"          : h4_s,
        "_cross_bar"   : cross_bar,
        "_pb_start"    : pullback_start,
        "_pb_end"      : pullback_end,
        "_green_bar"   : green_bar,
    }

# ── Live print ────────────────────────────────────────────────
LIVE_COLS = ["Ticker","Price","Score",
             "H4_Decline_%","Dist_S50_H4_%","Dist_S150_H4_%",
             "Cross_Ago","PB_Bars","Avg_Red_%",
             "Touched_EMA_JMA","Vol_Declining",
             "Green_Date","Green_Ago","Vol_Green_vs_Red","RSI"]
_CW = {"Ticker":8,"Price":10,"Score":7,
       "H4_Decline_%":13,"Dist_S50_H4_%":14,"Dist_S150_H4_%":15,
       "Cross_Ago":10,"PB_Bars":9,"Avg_Red_%":10,
       "Touched_EMA_JMA":15,"Vol_Declining":13,
       "Green_Date":12,"Green_Ago":10,"Vol_Green_vs_Red":17,"RSI":6}
_CF = {"Price":"${:.2f}","Score":"{:.0f}",
       "H4_Decline_%":"{:.2f}%","Dist_S50_H4_%":"{:+.2f}%","Dist_S150_H4_%":"{:+.2f}%",
       "Cross_Ago":"{:.0f}d","PB_Bars":"{:.0f}","Avg_Red_%":"{:.2f}%",
       "Green_Ago":"{:.0f}d","Vol_Green_vs_Red":"{:.2f}×","RSI":"{:.1f}"}
_hdr_done = False

def _live_header():
    global _hdr_done
    if _hdr_done: return
    sep="━"*230
    print(f"\n{sep}")
    print("  📊  LIVE — Declining H4 + SMA Near H4 + Gentle Pullback + Green Recovery")
    print(sep)
    print("".join(f"  {c:<{_CW.get(c,10)}}" for c in LIVE_COLS))
    print("  "+"─"*228)
    _hdr_done = True

def live_print(r):
    _live_header()
    row=""
    for c in LIVE_COLS:
        val=r.get(c,"—"); w=_CW.get(c,10); fmt=_CF.get(c)
        try:   s=fmt.format(val) if (fmt and val not in("—",None)) else str(val)
        except: s=str(val)
        row+=f"  {s:<{w}}"
    print(row)

# ── Health check ──────────────────────────────────────────────
print("━"*65)
print("  STEP 1  DATA CHECK")
print("━"*65)
chk = download(["AAPL","NVDA","AMD"], CFG["history_days"])
if not chk: print("❌  No data.")
else:
    for s, d in chk.items():
        p = float(d["Close"].iloc[-1])
        mh = get_monthly_h4_levels(d)
        today = pd.Timestamp.today().normalize()
        hc = mh.get(today.to_period("M"), None)
        hp = mh.get((today.to_period("M")-1), None)
        print(f"  ✅ {s}: ${p:.2f}  H4_this={hc}  H4_prev={hp}  declining={'✅' if (hc and hp and hc<hp) else '❌'}  {d.index[-1].date()}")
print()

# ── Diagnostic ────────────────────────────────────────────────
print("━"*65)
print("  STEP 2  DIAGNOSTIC")
print("━"*65+"\n")
DIAG = ["AAPL","NVDA","AMD","PLTR","MU","SMCI","META","CRWD","MXL","AXON"]
diag_data = download(DIAG, CFG["history_days"])
print(f"  Downloaded {len(diag_data)}/{len(DIAG)}\n")
print(f"  {'SYM':<7} {'PRICE':>8}  {'H4↓':>5}  {'SMA_H4':>7}  {'CROSS':>6}  {'PB':>4}  RESULT")
print("  "+"─"*52)
for sym in DIAG:
    if sym not in diag_data:
        print(f"  {sym:<7} — no data"); continue
    try:
        df_d = diag_data[sym]
        p    = float(df_d["Close"].iloc[-1])
        r    = detect_pattern(sym, df_d)
        if r:
            print(f"  {sym:<7} ${p:>7.2f}  {'✅':>5}  {'✅':>7}  "
                  f"{r['Cross_Ago']:>5}d  {r['PB_Bars']:>3}d  "
                  f"✅ Score={r['Score']} Green={r['Green_Date']}")
        else:
            print(f"  {sym:<7} ${p:>7.2f}  {'—':>5}  {'—':>7}  {'—':>6}  {'—':>4}  ❌")
    except Exception as e:
        print(f"  {sym:<7} error: {e}")

print(f"""
  Steps:
    Step 1  H4 declining : this month H4 < prev month H4
    Step 2  SMA near H4  : SMA50 + SMA150 within {CFG['sma_h4_zone_pct']}% of H4
    Step 3  Cross + PB   : price crossed above SMA50+SMA150+H4
                           then pulled back gently (red candles <= {CFG['max_red_candle_pct']}%/day)
                           touching EMA8 or JMA within {CFG['ema_zone_pct']}%
    Step 4  Green recover: green candle > JMA + EMA8, vol > prev red candle

  Tune if mostly ❌:
    sma_h4_zone_pct       8 → 12
    ema_zone_pct          3 → 5
    max_red_candle_pct    4 → 6
    cross_lookback       20 → 30
    max_bars_since_green  3 → 5
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
        ("https://www.nasdaqtrader.com/dynamic/SymDir/otherlisted.txt","otherlisted"),
    ]:
        try:
            r = requests.get(url,headers=hdrs,timeout=20); r.raise_for_status()
            df = pd.read_csv(io.StringIO(r.text),sep="|")
            df.columns=[c.strip() for c in df.columns]
            sc=next((c for c in ["Symbol","ACT Symbol","Nasdaq Symbol"] if c in df.columns),None)
            ec=next((c for c in ["ETF","Is ETF"] if c in df.columns),None)
            if not sc: continue
            b=len(pool)
            for _,row in df.iterrows():
                s=str(row[sc]).strip()
                if not s or s=="nan": continue
                if any(x in s for x in ["^","/","."," ","-"]): continue
                if s.endswith(tuple("WRUPQ")): continue
                if not(s.isalpha() and 1<=len(s)<=5): continue
                if ec and str(row.get(ec,"")).strip().upper()=="Y": continue
                pool.add(s.upper())
            print(f"  ✅ {label:<18}: +{len(pool)-b:>4} → {len(pool)}")
        except Exception as e: print(f"  ⚠️  {label}: {e}")
    try:
        r=requests.get(
            "https://api.nasdaq.com/api/screener/stocks"
            "?tableonly=true&limit=10000&exchange=nasdaq&download=true",
            headers={**hdrs,"Referer":"https://www.nasdaq.com/"},timeout=25)
        r.raise_for_status()
        rows=r.json()["data"]["rows"]
        t={row["symbol"].strip() for row in rows
           if row.get("symbol","").strip().isalpha()
           and 1<=len(row["symbol"].strip())<=5}
        b=len(pool); pool|=t
        print(f"  ✅ {'NASDAQ API':<18}: +{len(pool)-b:>4} → {len(pool)}")
    except Exception as e: print(f"  ⚠️  NASDAQ API: {e}")
    static={
        "AAPL","MSFT","NVDA","AMZN","META","GOOGL","TSLA","AVGO","COST","NFLX",
        "AMD","CSCO","ADBE","QCOM","TXN","AMAT","MU","KLAC","LRCX","MRVL",
        "PANW","CRWD","SNPS","CDNS","TEAM","WDAY","PLTR","DDOG","SNOW","MDB",
        "VRTX","ISRG","LULU","FTNT","SBUX","TMUS","RBRK","NET","AXON","ANET",
        "CAVA","VRT","ELF","GRMN","ON","ENPH","ROST","HOOD","COIN","UPST",
        "SMCI","MXL","ACLS","IRTC","MNDY","HUBS","GTLB","GLBE","CELH","DKNG",
        "SMAR","BILL","APPN","ALRM","FIVE","BOOT","INMD","LGIH","UFPT","PRCT",
        "IONQ","RGTI","QUBT","ASTS","RKLB","FSLR","PYPL","ROKU","SNAP","PINS",
    }
    b=len(pool); pool|=static
    print(f"  ✅ {'Static fallback':<18}: +{len(pool)-b:>4} → {len(pool)}")
    clean=sorted({s.upper() for s in pool if isinstance(s,str)
                  and s.isalpha() and 1<=len(s)<=5})
    print(f"\n  🎯 Total: {len(clean)} tickers")
    return clean

TICKERS = get_tickers()
print()

# ── Main scan ─────────────────────────────────────────────────
print("━"*65)
print(f"  STEP 4  SCANNING {len(TICKERS)} TICKERS")
print("━"*65+"\n")

_hdr_done=False; results=[]; no_data=0
batches=[TICKERS[i:i+CFG["batch_size"]] for i in range(0,len(TICKERS),CFG["batch_size"])]

with tqdm(total=len(TICKERS),desc="Scanning",unit="stk",
          bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]") as pbar:
    for batch in batches:
        data_map=download(batch,CFG["history_days"])
        no_data+=len(batch)-len(data_map)
        for sym in batch:
            pbar.update(1)
            if sym not in data_map: continue
            try:
                r=detect_pattern(sym,data_map[sym])
                if r: results.append(r); live_print(r)
            except Exception: pass
        time.sleep(CFG["batch_sleep"])

got=len(TICKERS)-no_data; pct=got/max(len(TICKERS),1)*100
print(f"\n{'━'*65}")
print(f"  SCAN COMPLETE | {len(TICKERS)} | {got} ({pct:.0f}%) | ✅ {len(results)}")
print(f"{'━'*65}")
print(f"""
  📊 DEBUG BREAKDOWN:
  Total processed           : {_DBG['total']}
  Passed vol/price filter   : {_DBG['pass_filter']}
  ❌ Failed Step 1 (H4↓)    : {_DBG['fail_s1_h4']}
  ❌ Failed Step 2 (SMA/H4) : {_DBG['fail_s2_sma']}
  ❌ Failed Step 3 (cross)   : {_DBG['fail_s3_cross']}
  ❌ Failed Step 3 (pullback): {_DBG['fail_s3_pb']}
  ❌ Failed Step 4 (green)   : {_DBG['fail_s4_green']}
  ✅ Passed all              : {_DBG['pass_all']}
""")

if not results:
    print("  Relax the failing condition:")
    print("   S1: require_declining_h4  True → False")
    print("   S2: sma_h4_zone_pct          8 → 12")
    print("   S3: cross_lookback          20 → 30")
    print("   S3: ema_zone_pct             3 → 5")
    print("   S3: max_red_candle_pct       4 → 6")
    print("   S4: max_bars_since_green     3 → 5")

results.sort(key=lambda x: -x["Score"])

# ── Build df_out ──────────────────────────────────────────────
ts      = datetime.today().strftime("%Y%m%d_%H%M")
out_dir = os.environ.get("GITHUB_WORKSPACE", os.getcwd())

COLS=[
    "Ticker","Price","Score",
    "H4_This","H4_Prev","H4_Decline_%",
    "SMA50","SMA150","Cam_H4","Dist_S50_H4_%","Dist_S150_H4_%",
    "Cross_Ago","PB_Bars","Avg_Red_%","Touched_EMA_JMA","Vol_Declining",
    "Green_Date","Green_Ago","Vol_Green_vs_Red",
    "JMA","EMA8","Dist_JMA_%","Dist_EMA8_%","Dist_H4_%",
    "RSI","MACD_Hist","Avg_Vol_20d",
]
df_out=pd.DataFrame([{k:v for k,v in r.items() if not k.startswith("_")}
                      for r in results]) if results else pd.DataFrame(columns=COLS)
if not df_out.empty:
    df_out=df_out[[c for c in COLS if c in df_out.columns]]
    df_out.reset_index(drop=True,inplace=True)

FMT={
    "Price"          : lambda v: f"${v:.2f}",
    "Score"          : lambda v: f"{v:.0f}",
    "H4_This"        : lambda v: f"${v:.2f}",
    "H4_Prev"        : lambda v: f"${v:.2f}",
    "H4_Decline_%"   : lambda v: f"{v:.2f}%",
    "SMA50"          : lambda v: f"${v:.2f}",
    "SMA150"         : lambda v: f"${v:.2f}",
    "Cam_H4"         : lambda v: f"${v:.2f}",
    "Dist_S50_H4_%"  : lambda v: f"{v:+.2f}%",
    "Dist_S150_H4_%" : lambda v: f"{v:+.2f}%",
    "Cross_Ago"      : lambda v: f"{int(v)}d",
    "PB_Bars"        : lambda v: f"{int(v)}",
    "Avg_Red_%"      : lambda v: f"{v:.2f}%",
    "Green_Ago"      : lambda v: f"{int(v)}d",
    "Vol_Green_vs_Red": lambda v: f"{v:.2f}×",
    "JMA"            : lambda v: f"${v:.2f}",
    "EMA8"           : lambda v: f"${v:.2f}",
    "Dist_JMA_%"     : lambda v: f"{v:+.2f}%",
    "Dist_EMA8_%"    : lambda v: f"{v:+.2f}%",
    "Dist_H4_%"      : lambda v: f"{v:+.2f}%",
    "RSI"            : lambda v: f"{v:.1f}",
    "MACD_Hist"      : lambda v: f"{v:.4f}",
    "Avg_Vol_20d"    : lambda v: f"{v:,.0f}",
}

def fmt_v(col,val):
    if val is None or (isinstance(val,float) and np.isnan(val)): return "—"
    try:
        if col in FMT: return FMT[col](val)
    except Exception: pass
    return str(val) if str(val) not in ("nan","None","") else "—"

# ── Notebook display ──────────────────────────────────────────
if _IN_NOTEBOOK and results:
    DISP=["Ticker","Price","Score","H4_Decline_%",
          "Dist_S50_H4_%","Cross_Ago","PB_Bars","Avg_Red_%",
          "Touched_EMA_JMA","Vol_Declining",
          "Green_Date","Green_Ago","Vol_Green_vs_Red","RSI"]
    DISP=[c for c in DISP if c in df_out.columns]
    gc="#22c55e"
    th="".join(
        f'<th style="background:#0f172a;color:#e2e8f0;padding:9px 12px;'
        f'font-size:11px;font-weight:700;border-bottom:2px solid {gc};white-space:nowrap">'
        f'{c}</th>' for c in DISP)
    rows_html=""
    for i,r in enumerate(results):
        bg="#fff" if i%2==0 else "#f0f9ff"
        tds=""
        for col in DISP:
            raw=r.get(col); disp=fmt_v(col,raw); sty=""
            if col=="Score":
                try:
                    v=float(raw); g=int(min(220,80+v*1.4))
                    sty=f"background:rgb(20,{g},60);color:#fff;font-weight:700;text-align:center"
                except: pass
            elif col=="Vol_Green_vs_Red":
                try:
                    v=float(str(raw).replace("×",""))
                    sty="color:#22c55e;font-weight:800" if v>=2 else "color:#86efac;font-weight:600"
                except: pass
            elif col in("Touched_EMA_JMA","Vol_Declining"):
                sty="color:#22c55e;font-weight:700;text-align:center" if raw=="✅" else "text-align:center;color:#94a3b8"
            elif col=="Avg_Red_%":
                try:
                    v=float(str(raw).replace("%",""))
                    sty="color:#22c55e;font-weight:700" if v<=2 else "color:#f59e0b" if v<=3 else "color:#ef4444"
                except: pass
            tds+=f'<td style="padding:7px 12px;font-size:12px;border-bottom:1px solid #e2e8f0;white-space:nowrap;{sty}">{disp}</td>'
        rows_html+=f'<tr style="background:{bg}">{tds}</tr>\n'

    ticker_csv_str=",".join(r["Ticker"] for r in results)
    display_html(f"""
<div style="background:linear-gradient(135deg,#0f172a,#1e3a5f);border-radius:10px;
            padding:18px 24px;margin-bottom:8px">
  <h2 style="margin:0;color:#60a5fa;font-size:20px;font-weight:700">
    📈 Declining H4 + SMA Near H4 + Gentle Pullback → Green Recovery
  </h2>
  <p style="margin:6px 0 0;color:#94a3b8;font-size:12px">
    {datetime.today().strftime('%Y-%m-%d %H:%M')} &nbsp;·&nbsp;
    <b style="color:{gc}">{len(results)} matches</b>
  </p>
</div>
<div style="background:#0f172a;border-radius:8px;padding:14px 16px;margin:8px 0;
            border-left:4px solid {gc};">
  <p style="margin:0 0 4px;color:#94a3b8;font-size:11px;font-weight:600;
             text-transform:uppercase;letter-spacing:.05em">
    📋 Stock List (CSV) — copy &amp; paste
  </p>
  <p style="margin:0;color:{gc};font-size:13px;font-weight:700;
             font-family:'Courier New',monospace;word-break:break-all">
    {ticker_csv_str}
  </p>
</div>
<div style="overflow-x:auto;border:1px solid #e2e8f0;border-radius:8px">
  <table style="border-collapse:collapse;width:100%;min-width:700px">
    <thead><tr>{th}</tr></thead><tbody>{rows_html}</tbody>
  </table>
</div>
<div style="background:#f8fafc;border:1px solid #e2e8f0;border-radius:8px;
            padding:12px 18px;margin-top:6px;font-size:11px;color:#64748b">
  <b>GUIDE</b> &nbsp;·&nbsp;
  H4_Decline_% = how much H4 fell from prev month (the energy coil) &nbsp;·&nbsp;
  Avg_Red_% = average daily red candle size during pullback (lower = gentler) &nbsp;·&nbsp;
  Vol_Green_vs_Red = green candle volume ÷ previous red candle volume &nbsp;·&nbsp;
  Touched_EMA_JMA ✅ = pullback came close to EMA8/JMA (tightest retest)
</div>""")

elif results:
    CLI=["Ticker","Price","Score","H4_Decline_%","PB_Bars",
         "Avg_Red_%","Touched_EMA_JMA","Green_Ago","Vol_Green_vs_Red","RSI"]
    col_w={c:max(len(c),max(len(fmt_v(c,r.get(c))) for r in results))+2 for c in CLI}
    top="┬".join("─"*col_w[c] for c in CLI)
    sep="┼".join("─"*col_w[c] for c in CLI)
    bot="┴".join("─"*col_w[c] for c in CLI)
    hdr="│".join(c.center(col_w[c]) for c in CLI)
    inner=sum(col_w.values())+len(CLI)-1
    print(f"\n  ╔{'═'*inner}╗")
    print(f"  ║{'  Declining H4 + Gentle Pullback + Green Recovery   '+datetime.today().strftime('%Y-%m-%d')+'   '+str(len(df_out))+' matches'.center(inner)}║")
    print(f"  ╚{'═'*inner}╝\n")
    print(f"  ┌{top}┐\n  │{hdr}│\n  ├{sep}┤")
    for i,r in enumerate(results):
        cells=[fmt_v(c,r.get(c)).center(col_w[c]) for c in CLI]
        print(f"  │{'│'.join(cells)}│")
        if i<len(results)-1: print(f"  ├{sep}┤")
    print(f"  └{bot}┘")

# ── Save CSV + TV ──────────────────────────────────────────────
fpath=os.path.join(out_dir,f"h4_pullback_recovery_{ts}.csv")
df_out.to_csv(fpath,index=False)
print(f"\n  💾 CSV → {fpath}")
tv=os.path.join(out_dir,f"tv_h4_pullback_{ts}.txt")
with open(tv,"w") as f:
    f.write(f"###H4 Pullback Recovery {datetime.today().strftime('%Y-%m-%d')}\n")
    for r in results: f.write(f"NASDAQ:{r['Ticker']}\n")
print(f"  📋 TradingView → {tv}")

# ── Email ──────────────────────────────────────────────────────
def _send_email(rl, csv_path):
    gu=_GMAIL_USER; gp=_GMAIL_PASS; et=_EMAIL_TO
    if not gu: print("[Email] ❌  GMAIL_USER secret is empty"); return
    if not gp: print("[Email] ❌  GMAIL_PASS secret is empty\n         → myaccount.google.com/apppasswords"); return
    if not et: print("[Email] ❌  EMAIL_TO secret is empty"); return
    eto=[e.strip() for e in et.split(",") if e.strip()]; cnt=len(rl)

    try:
        ticker_csv=",".join(r.get("Ticker","") for r in rl) if rl else "—"
        print(f"[Email] Sending to {et}  ({cnt} results)...")

        ticker_html=f"""
<div style="margin:14px 0;padding:14px 16px;background:#0f172a;
            border-radius:8px;border-left:4px solid #22c55e;">
  <p style="margin:0 0 6px;color:#94a3b8;font-size:11px;font-weight:600;
             letter-spacing:.05em;text-transform:uppercase">
    📋 Stock List — Copy &amp; paste
  </p>
  <p style="margin:0;color:#22c55e;font-size:13px;font-weight:700;
             font-family:'Courier New',monospace;word-break:break-all">
    {ticker_csv}
  </p>
</div>"""

        th_e="".join(
            f'<th style="background:#1e293b;color:#e2e8f0;padding:8px 11px;'
            f'font-size:11px;font-weight:700;border-bottom:2px solid #3b82f6;white-space:nowrap">{c}</th>'
            for c in ["Ticker","Price","Score","H4↓%","PB_Bars","Avg_Red%",
                      "EMA_Touch","Green_Ago","Vol_G/R","RSI"])
        rows_e=""
        for i,r in enumerate(rl[:50]):
            bg="#fff" if i%2==0 else "#f0f9ff"
            vgr=float(r.get("Vol_Green_vs_Red",0))
            rows_e+=(
                f'<tr style="background:{bg}">'
                f'<td style="padding:6px 11px;font-size:12px;font-weight:700;color:#22c55e">{r.get("Ticker","—")}</td>'
                f'<td style="padding:6px 11px;font-size:12px">${float(r.get("Price",0)):.2f}</td>'
                f'<td style="padding:6px 11px;font-size:12px;font-weight:700;background:#166534;color:#fff;text-align:center">{float(r.get("Score",0)):.0f}</td>'
                f'<td style="padding:6px 11px;font-size:12px">{float(r.get("H4_Decline_%",0)):.2f}%</td>'
                f'<td style="padding:6px 11px;font-size:12px;text-align:center">{int(r.get("PB_Bars",0))}</td>'
                f'<td style="padding:6px 11px;font-size:12px">{float(r.get("Avg_Red_%",0)):.2f}%</td>'
                f'<td style="padding:6px 11px;font-size:12px;text-align:center">{r.get("Touched_EMA_JMA","—")}</td>'
                f'<td style="padding:6px 11px;font-size:12px;text-align:center">{int(r.get("Green_Ago",0))}d</td>'
                f'<td style="padding:6px 11px;font-size:12px;color:{"#22c55e" if vgr>=1.5 else "#94a3b8"};font-weight:700">{vgr:.2f}×</td>'
                f'<td style="padding:6px 11px;font-size:12px">{float(r.get("RSI",0)):.1f}</td>'
                f'</tr>'
            )

        no_res="" if cnt else (
            '<tr><td colspan="10" style="padding:20px;text-align:center;color:#64748b;font-size:13px">No matches found today</td></tr>'
        )

        html_e=f"""<!DOCTYPE html><html><body style="margin:0;padding:0;
background:#f1f5f9;font-family:'Segoe UI',Arial,sans-serif">
<table width="100%" cellpadding="0" cellspacing="0" style="padding:20px 0"><tr><td>
<table width="100%" cellpadding="0" cellspacing="0"
   style="max-width:960px;margin:0 auto;background:#fff;border-radius:12px;
          overflow:hidden;box-shadow:0 4px 20px rgba(0,0,0,0.08)">
  <tr><td style="background:linear-gradient(135deg,#0f172a,#1e3a5f);padding:22px 28px">
    <h1 style="margin:0;color:#60a5fa;font-size:20px;font-weight:700">
      📊 Declining H4 + SMA Near H4 + Gentle Pullback → Green Recovery
    </h1>
    <p style="margin:6px 0 0;color:#94a3b8;font-size:12px">
      {datetime.today().strftime('%Y-%m-%d %H:%M UTC')} &nbsp;·&nbsp;
      {cnt} match{'es' if cnt!=1 else ''} found
    </p>
  </td></tr>
  <tr><td style="padding:16px">
    {ticker_html}
    <div style="overflow-x:auto;border-radius:8px;border:1px solid #e2e8f0">
      <table style="border-collapse:collapse;width:100%;min-width:700px">
        <thead><tr>{th_e}</tr></thead>
        <tbody>{rows_e or no_res}</tbody>
      </table>
    </div>
    <p style="font-size:11px;color:#64748b;margin:10px 0 0">
      📎 CSV + TradingView watchlist attached &nbsp;·&nbsp;
      <b>H4↓%</b> = how much Camarilla H4 declined this month vs last &nbsp;·&nbsp;
      <b>EMA_Touch ✅</b> = pullback touched EMA8/JMA support &nbsp;·&nbsp;
      <b>Vol_G/R</b> = green candle volume ÷ previous red candle volume
    </p>
  </td></tr>
  <tr><td style="background:#f8fafc;padding:12px 28px;border-top:1px solid #e2e8f0;text-align:center">
    <p style="margin:0;color:#94a3b8;font-size:10px">⚠️ Not financial advice &nbsp;·&nbsp; Auto-generated by GitHub Actions</p>
  </td></tr>
</table></td></tr></table></body></html>"""

        plain_e="\n".join([
            f"Declining H4 + Gentle Pullback Recovery — {datetime.today().strftime('%Y-%m-%d')}",
            f"{cnt} matches",
            "",
            f"STOCKS: {ticker_csv}",
            "",
            "="*65,
        ]+([
            f"{r.get('Ticker','—'):<7} ${float(r.get('Price',0)):.2f}  "
            f"Score:{float(r.get('Score',0)):.0f}  "
            f"H4↓:{float(r.get('H4_Decline_%',0)):.2f}%  "
            f"PB:{int(r.get('PB_Bars',0))}bars({float(r.get('Avg_Red_%',0)):.1f}%/bar)  "
            f"Green:{int(r.get('Green_Ago',0))}d ago  "
            f"VolRatio:{float(r.get('Vol_Green_vs_Red',0)):.1f}×"
            for r in rl[:50]
        ] if rl else ["No matches today"])+["\n📎 CSV + TradingView attached."])

        subj=(f"📊 H4 Pullback Recovery — {cnt} setup{'s' if cnt!=1 else ''}"
              f" — {datetime.today().strftime('%Y-%m-%d')}")

        msg=MIMEMultipart("mixed")
        msg["Subject"]=subj; msg["From"]=gu; msg["To"]=", ".join(eto)
        alt=MIMEMultipart("alternative")
        alt.attach(MIMEText(plain_e,"plain")); alt.attach(MIMEText(html_e,"html"))
        msg.attach(alt)

    except Exception as e:
        print(f"[Email] ❌  Body failed: {type(e).__name__}: {e}"); return

    for att in [csv_path,tv]:
        if att and os.path.exists(att):
            try:
                with open(att,"rb") as f:
                    part=MIMEBase("application","octet-stream"); part.set_payload(f.read())
                encoders.encode_base64(part)
                part.add_header("Content-Disposition",f"attachment; filename={os.path.basename(att)}")
                msg.attach(part); print(f"[Email] 📎 {os.path.basename(att)}")
            except Exception as e: print(f"[Email] ⚠️  Attach: {e}")

    try:
        with smtplib.SMTP_SSL("smtp.gmail.com",465) as srv:
            srv.login(gu,gp.replace(" ",""))
            srv.sendmail(gu,eto,msg.as_string())
        print(f"[Email] ✅  Sent to: {', '.join(eto)}")
    except smtplib.SMTPAuthenticationError:
        print("[Email] ❌  AUTH FAILED — use Gmail App Password")
        print("         Generate: myaccount.google.com/apppasswords")
    except smtplib.SMTPException as e:
        print(f"[Email] ❌  SMTP: {e}")
    except Exception as e:
        print(f"[Email] ❌  {type(e).__name__}: {e}")

try:
    _send_email(results, fpath)
except Exception as e:
    print(f"[Email] ❌  Top-level: {type(e).__name__}: {e}")
    print("[Email]    CSV and charts still saved.")

if _IN_NOTEBOOK:
    try:
        from google.colab import files
        files.download(fpath); files.download(tv)
    except Exception: pass

# ── Charts for top 5 ──────────────────────────────────────────
if results:
    top=results[:min(5,len(results))]
    fig,axes=plt.subplots(len(top),1,figsize=(15,5.5*len(top)),facecolor="#0f172a")
    if len(top)==1: axes=[axes]

    for idx,r in enumerate(top):
        ax=axes[idx]
        df_p=r["_df"].tail(60).copy(); n_p=len(df_p)
        fn=len(r["_df"]); off=fn-n_p

        jma  = r["_jma"].reindex(df_p.index)
        ema8 = r["_ema8"].reindex(df_p.index)
        s50  = r["_s50"].reindex(df_p.index)
        s150 = r["_s150"].reindex(df_p.index)
        h4   = r["_h4"].reindex(df_p.index)

        ax.set_facecolor("#0f172a")
        for i,(_,row_) in enumerate(df_p.iterrows()):
            o=float(row_["Open"]); h=float(row_["High"])
            l=float(row_["Low"]); c=float(row_["Close"])
            clr="#34d399" if c>=o else "#ef4444"
            ax.plot([i,i],[l,h],color=clr,lw=0.7,zorder=2)
            blo=min(o,c); bhi=max(o,c); bh=max(bhi-blo,(h-l)*0.005)
            rect=mpatches.FancyBboxPatch((i-0.3,blo),0.6,bh,
                 boxstyle="square,pad=0",facecolor=clr,edgecolor=clr,lw=0.3,zorder=3)
            ax.add_patch(rect)

        ax.plot(range(n_p),jma.values,  color="#22d3ee",lw=1.8,label="JMA",zorder=6)
        ax.plot(range(n_p),ema8.values, color="#34d399",lw=1.5,ls="--",label="EMA8",zorder=5)
        ax.plot(range(n_p),s50.values,  color="#3b82f6",lw=1.5,label="SMA50",zorder=5)
        ax.plot(range(n_p),s150.values, color="#f472b6",lw=1.3,ls="-.",label="SMA150",zorder=4)
        ax.plot(range(n_p),h4.values,   color="#f59e0b",lw=2.0,ls=":",label="Cam H4",zorder=6)

        # Shade pullback zone
        ps=r["_pb_start"]-off; pe=r["_pb_end"]-off
        if 0<=ps<n_p and 0<=pe<n_p:
            ax.axvspan(ps,pe+1,alpha=0.12,color="#ef4444",label="Pullback",zorder=1)

        # Mark cross bar
        cb=r["_cross_bar"]-off
        if 0<=cb<n_p:
            ax.axvline(cb,color="#fbbf24",lw=1.5,ls=":",alpha=0.8)
            ax.scatter([cb],[float(df_p["Close"].iloc[cb])],
                       color="#fbbf24",s=120,zorder=8,marker="^",label="H4 Cross")

        # Mark green recovery bar
        gb=r["_green_bar"]-off
        if 0<=gb<n_p:
            ax.axvline(gb,color="#22c55e",lw=2.0,ls="--",alpha=0.9)
            ax.scatter([gb],[float(df_p["Close"].iloc[gb])],
                       color="#22c55e",s=200,zorder=9,marker="^",label=f"Recovery {r['Green_Date']}")

        tick_step=max(1,n_p//8)
        ax.set_xticks(range(0,n_p,tick_step))
        ax.set_xticklabels(
            [df_p.index[i].strftime("%m/%d") for i in range(0,n_p,tick_step)],
            color="#94a3b8",fontsize=7)
        ax.set_xlim(-0.5,n_p-0.5)
        ax.set_title(
            f"{r['Ticker']}  ${r['Price']:.2f}  |  Score {r['Score']}/100  |  "
            f"H4↓{r['H4_Decline_%']:.2f}%  |  "
            f"Pullback: {r['PB_Bars']}bars avg {r['Avg_Red_%']:.1f}%/bar  "
            f"EMA_Touch:{r['Touched_EMA_JMA']}  |  "
            f"Green {r['Green_Date']} ({r['Green_Ago']}d) Vol{r['Vol_Green_vs_Red']:.1f}×  |  "
            f"RSI {r['RSI']:.0f}",
            color="#e2e8f0",fontsize=8,fontweight="bold",pad=5)
        ax.tick_params(colors="#94a3b8",labelsize=7)
        for sp_ in ax.spines.values(): sp_.set_edgecolor("#1e3a5f")
        ax.legend(loc="upper left",facecolor="#1e293b",
                  labelcolor="#e2e8f0",fontsize=7,framealpha=0.9,ncol=4)
        ax.grid(color="#1e3a5f",ls="--",lw=0.4,alpha=0.4,axis="y")

    plt.suptitle(
        f"Declining H4 + SMA Near H4 + Gentle Pullback → Green Recovery  ·  {datetime.today().strftime('%Y-%m-%d')}\n"
        f"🔵 JMA  🟢 EMA8  🔵 SMA50  🩷 SMA150  🟠 Cam H4  🔴=pullback  🟡=H4 cross  ▲=recovery",
        color="#60a5fa",fontsize=9,fontweight="bold",y=1.001)
    plt.tight_layout()
    cp=os.path.join(out_dir,f"h4_pullback_chart_{ts}.png")
    plt.savefig(cp,dpi=150,bbox_inches="tight",facecolor="#0f172a")
    if _IN_NOTEBOOK: plt.show()
    else: plt.close()
    print(f"  📊 Chart → {cp}")

print("""
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  📋 PATTERN EXPLAINED (from chart)

  STEP 1 — DECLINING CAMARILLA H4
    This month's H4 < Previous month's H4
    = The pivot resistance is contracting — energy coiling
    Stocks that break above a declining H4 often run hard

  STEP 2 — SMA50 + SMA150 CLUSTERED NEAR H4
    Both MAs within 8% of the Cam H4 level
    = Triple confluence zone (2 MAs + monthly pivot)
    When all 3 are bunched, the breakout is more significant

  STEP 3 — GENTLE PULLBACK TO EMA8/JMA AFTER CROSS
    Price crosses above SMA50 + SMA150 + H4
    Then GENTLY retraces back toward EMA8/JMA:
    • Each red candle <= 4% body (not a violent selloff)
    • Volume declining during pullback (sellers dry up)
    • Price comes within 3% of EMA8 or JMA
    This is the "high and tight" pullback — controlled

  STEP 4 — GREEN CANDLE CLOSES ABOVE JMA + EMA8
    First green candle after the pullback closes above
    BOTH JMA and EMA8 on HIGHER volume than last red bar
    = Buyers came back exactly at the fast MA support

  💡 BEST SETUPS
  H4_Decline_% > 2%     meaningful contraction = more energy
  Avg_Red_% < 2%        very gentle pullback = controlled
  Touched_EMA_JMA ✅     precise retest of fast MAs
  Vol_Green_vs_Red > 2× strong buyer conviction
  Green_Ago = 0-1d      fresh recovery — earliest entry

  ⚙️  TUNE IF 0 RESULTS
  sma_h4_zone_pct       8 → 12
  ema_zone_pct          3 → 5
  max_red_candle_pct    4 → 6
  cross_lookback       20 → 30
  max_bars_since_green  3 → 5
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
""")
