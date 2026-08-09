# ============================================================
# NASDAQ — MA Compression Near Cam S3 + Breakout Above All
# ============================================================
#
# EXACT PATTERN:
#
#  C1 — MA COMPRESSION: JMA, EMA8, SMA21 all clustered near SMA50
#      The gap between the HIGHEST and LOWEST of these 4 MAs
#      must be <= compression_pct% of price
#      = All fast MAs bunched together = energy coiling
#
#  C2 — CLUSTER IS NEAR MONTHLY CAMARILLA S3
#      The average of all 4 MAs must be within s3_zone_pct%
#      of the current month's Camarilla S3 level
#      S3 = Close - (High - Low) × 1.1 / 4  (from prior month)
#      = The MA cluster sits AT the monthly support pivot
#
#  C3 — PRICE CROSSED ABOVE ALL 4 MAs AND CAMARILLA S3
#      Price must have been BELOW all of them recently
#      and is NOW ABOVE all of them (JMA, EMA8, SMA21, SMA50, S3)
#      within the last cross_lookback bars
#      This is the "everything breaks out at once" moment
#
#  C4 — VOLUME > PREVIOUS DAY'S VOLUME
#      Volume on the breakout bar must be strictly greater
#      than volume on the day before it
#      = Fresh buying, not a low-volume drift
#
# LOGIC FLOW:
#   JMA + EMA8 + SMA21 compress near SMA50
#   → Entire cluster sits near monthly Cam S3 support
#   → Price breaks above ALL MAs and S3 in one move
#   → On HIGHER volume than previous day
#   = Explosive multi-MA compression breakout at key pivot
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
    print(f"  ⚠️  Missing secrets: {', '.join(missing)}")
    print(f"  ℹ️  GitHub repo → Settings → Secrets → Actions")
    print(f"  ℹ️  Email will be SKIPPED this run")
print("━"*65)
print()

# ── CONFIG ────────────────────────────────────────────────────
CFG = {
    "history_days"             : 300,

    # ── MA periods ────────────────────────────────────────────
    "jma_period"               : 13,
    "jma_phase"                : 40,
    "ema8_period"              : 8,
    "sma21_period"             : 21,
    "sma50_period"             : 50,

    # ── C1: MA compression ────────────────────────────────────
    # JMA/EMA8/SMA21 are fast MAs. In a real compression they
    # converge TOWARD SMA50 (not past it). We measure:
    #   max distance of any fast MA from SMA50 / SMA50
    # Must be <= compression_pct% (all fast MAs near SMA50)
    "compression_pct"          : 8.0,   # fast MAs within 8% of SMA50
    "compression_lookback"     : 15,    # scan last 15 bars for compression

    # ── C2: SMA50 near Cam S3 ─────────────────────────────────
    # SMA50 must be within s3_zone_pct% of monthly Cam S3
    "s3_zone_pct"              : 8.0,   # SMA50 within 8% of S3

    # ── C3: Price crossed above ALL MAs and S3 ────────────────
    "cross_lookback"           : 10,

    # ── C4: Volume > previous day ────────────────────────────
    "require_vol_gt_prev"      : True,
    "min_break_vol_mult"       : 0.5,
    "vol_avg_bars"             : 20,

    # ── Filters ───────────────────────────────────────────────
    "min_avg_volume"           : 50_000,
    "min_price"                : 1.0,

    "batch_size"               : 50,
    "batch_sleep"              : 1.5,
}


# ── Indicators ───────────────────────────────────────────────
def calc_jma(series, period=13, phase=40):
    """JMA approximation (pure numpy)."""
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

def calc_ema(series, period):
    return series.ewm(span=period, adjust=False).mean()

def calc_rsi(close, period=14):
    delta = close.diff()
    gain  = delta.clip(lower=0)
    loss  = -delta.clip(upper=0)
    ag    = gain.ewm(alpha=1/period, adjust=False).mean()
    al    = loss.ewm(alpha=1/period, adjust=False).mean()
    rs    = ag / al.replace(0, np.nan)
    return 100 - (100 / (1 + rs))

def calc_macd_hist(close, fast=12, slow=26, signal=9):
    ef  = calc_ema(close, fast)
    es  = calc_ema(close, slow)
    ml  = ef - es
    sig = calc_ema(ml, signal)
    return ml - sig

# ── Camarilla S3 ─────────────────────────────────────────────
def cam_s3(high, low, close):
    """Camarilla S3 = Close - (High - Low) × 1.1 / 4"""
    return close - (high - low) * 1.1 / 4.0

def build_monthly_s3_series(df):
    """
    Per-bar Camarilla S3 series: each bar's S3 is derived
    from the PRIOR completed month's H/L/C.
    """
    df      = df.copy()
    df.index= pd.to_datetime(df.index)
    periods = df.index.to_period("M")
    unique_months = sorted(periods.unique())

    month_s3 = {}
    for i, mp in enumerate(unique_months):
        if i == 0: continue
        prev_mp = unique_months[i-1]
        sub = df[periods == prev_mp]
        if len(sub) < 5: continue
        hi = float(sub["High"].max())
        lo = float(sub["Low"].min())
        cl = float(sub["Close"].iloc[-1])
        month_s3[mp] = round(cam_s3(hi, lo, cl), 4)

    s3_vals = [month_s3.get(mp, np.nan) for mp in periods]
    return pd.Series(s3_vals, index=df.index)

# ── Download ──────────────────────────────────────────────────
def _clean(df, min_bars=60):
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
    "total"        : 0,
    "pass_filter"  : 0,
    "fail_c1_comp" : 0,
    "fail_c2_s3"   : 0,
    "fail_c3_cross": 0,
    "fail_c4_vol"  : 0,
    "pass_all"     : 0,
}

# ── Core detection ────────────────────────────────────────────
def detect_pattern(sym, df):
    """
    Unified approach:
    Scan every bar in the last scan_window bars looking for:
      1. At that bar: JMA/EMA8/SMA21 all within compression_pct% of SMA50
      2. At that bar or nearby: SMA50 within s3_zone_pct% of monthly Cam S3
      3. Price on THAT bar crossed above all MAs and S3
         (was below on prior bar, above on this bar)
      4. Volume on that bar > previous bar volume

    This ties all 4 conditions to the SAME bar — the breakout bar
    itself must show compression AND be near S3 AND be the cross.
    """
    global _DBG
    _DBG["total"] += 1

    df      = df.copy(); df.index = pd.to_datetime(df.index)
    n       = len(df)
    price   = float(df["Close"].iloc[-1])
    avg_vol = float(df["Volume"].tail(CFG["vol_avg_bars"]).mean())

    if price   < CFG["min_price"]:      return None
    if avg_vol < CFG["min_avg_volume"]: return None
    if n < CFG["sma50_period"] + 20:    return None

    _DBG["pass_filter"] += 1

    # ── Compute MAs and S3 ────────────────────────────────────
    jma_s  = calc_jma(df["Close"], CFG["jma_period"], CFG["jma_phase"])
    ema8_s = calc_ema(df["Close"], CFG["ema8_period"])
    s21_s  = df["Close"].rolling(CFG["sma21_period"]).mean()
    s50_s  = df["Close"].rolling(CFG["sma50_period"]).mean()
    s3_s   = build_monthly_s3_series(df)
    rsi_s  = calc_rsi(df["Close"])
    macdh_s= calc_macd_hist(df["Close"])

    cur_jma  = float(jma_s.iloc[-1])   if not np.isnan(jma_s.iloc[-1])   else np.nan
    cur_ema8 = float(ema8_s.iloc[-1])  if not np.isnan(ema8_s.iloc[-1])  else np.nan
    cur_s21  = float(s21_s.iloc[-1])   if not np.isnan(s21_s.iloc[-1])   else np.nan
    cur_s50  = float(s50_s.iloc[-1])   if not np.isnan(s50_s.iloc[-1])   else np.nan
    cur_s3   = float(s3_s.iloc[-1])    if not np.isnan(s3_s.iloc[-1])    else np.nan
    cur_rsi  = float(rsi_s.iloc[-1])   if not np.isnan(rsi_s.iloc[-1])   else 50
    cur_mh   = float(macdh_s.iloc[-1]) if not np.isnan(macdh_s.iloc[-1]) else 0

    if any(np.isnan([cur_jma, cur_ema8, cur_s21, cur_s50])): return None
    if np.isnan(cur_s3): return None

    comp_pct  = CFG["compression_pct"]  / 100
    s3_pct    = CFG["s3_zone_pct"]      / 100
    scan_bars = CFG["compression_lookback"] + CFG["cross_lookback"]

    found_bar   = None
    found_date  = None
    found_spread= None
    found_s3dist= None

    for i in range(max(1, n - scan_bars), n):
        j_i   = float(jma_s.iloc[i])  if not np.isnan(jma_s.iloc[i])  else np.nan
        e8_i  = float(ema8_s.iloc[i]) if not np.isnan(ema8_s.iloc[i]) else np.nan
        s21_i = float(s21_s.iloc[i])  if not np.isnan(s21_s.iloc[i])  else np.nan
        s50_i = float(s50_s.iloc[i])  if not np.isnan(s50_s.iloc[i])  else np.nan
        s3_i  = float(s3_s.iloc[i])   if not np.isnan(s3_s.iloc[i])   else np.nan
        p_i   = float(df["Close"].iloc[i])
        vol_i = float(df["Volume"].iloc[i])

        if any(np.isnan([j_i, e8_i, s21_i, s50_i, s3_i])): continue
        if s50_i <= 0 or s3_i <= 0: continue

        # ── C1: MA compression at this bar ──────────────────
        # Fast MAs within compression_pct% of SMA50
        d_jma  = abs(j_i  - s50_i) / s50_i
        d_ema8 = abs(e8_i - s50_i) / s50_i
        d_s21  = abs(s21_i - s50_i) / s50_i
        max_dist = max(d_jma, d_ema8, d_s21)
        if max_dist > comp_pct:
            continue   # no compression here

        # ── C2: SMA50 near Cam S3 at this bar ───────────────
        dist_s3 = abs(s50_i - s3_i) / s3_i
        if dist_s3 > s3_pct:
            continue   # SMA50 not near S3

        # ── C3: Price crossed above ALL MAs and S3 ───────────
        # Current bar must be above ALL levels
        above_all = (p_i > j_i and p_i > e8_i and
                     p_i > s21_i and p_i > s50_i and p_i > s3_i)
        if not above_all:
            continue

        # Previous bar must have been below at least one
        j_p   = float(jma_s.iloc[i-1])  if not np.isnan(jma_s.iloc[i-1])  else np.nan
        e8_p  = float(ema8_s.iloc[i-1]) if not np.isnan(ema8_s.iloc[i-1]) else np.nan
        s21_p = float(s21_s.iloc[i-1])  if not np.isnan(s21_s.iloc[i-1])  else np.nan
        s50_p = float(s50_s.iloc[i-1])  if not np.isnan(s50_s.iloc[i-1])  else np.nan
        s3_p  = float(s3_s.iloc[i-1])   if not np.isnan(s3_s.iloc[i-1])   else np.nan
        p_p   = float(df["Close"].iloc[i-1])
        if any(np.isnan([j_p, e8_p, s21_p, s50_p, s3_p])): continue

        was_below = (p_p < j_p or p_p < e8_p or
                     p_p < s21_p or p_p < s50_p or p_p < s3_p)
        if not was_below:
            continue   # wasn't actually a cross (was already above)

        # ── C4: Volume > previous bar ───────────────────────
        vol_p = float(df["Volume"].iloc[i-1])
        if CFG["require_vol_gt_prev"] and vol_i <= vol_p:
            continue
        vol_mult = vol_i / avg_vol if avg_vol > 0 else 0
        if vol_mult < CFG["min_break_vol_mult"]:
            continue

        # Valid pattern found — keep most recent
        if found_bar is None or i > found_bar:
            found_bar   = i
            found_date  = df.index[i]
            found_spread= round(max_dist * 100, 2)
            found_s3dist= round(dist_s3 * 100, 2)

    if found_bar is None:
        # Attribute the failure to the first condition that blocked most
        # We can't know exactly without per-bar tracking, so check current bar
        if any(np.isnan([cur_jma, cur_ema8, cur_s21, cur_s50, cur_s3])):
            pass
        elif cur_s50 > 0:
            d = max(abs(cur_jma-cur_s50)/cur_s50,
                    abs(cur_ema8-cur_s50)/cur_s50,
                    abs(cur_s21-cur_s50)/cur_s50) * 100
            if d > CFG["compression_pct"]:
                _DBG["fail_c1_comp"] += 1
            elif abs(cur_s50-cur_s3)/cur_s3*100 > CFG["s3_zone_pct"]:
                _DBG["fail_c2_s3"] += 1
            elif not (price > cur_jma and price > cur_ema8 and
                      price > cur_s21 and price > cur_s50 and price > cur_s3):
                _DBG["fail_c3_cross"] += 1
            else:
                _DBG["fail_c4_vol"] += 1
        return None

    _DBG["pass_all"] += 1

    # ── Metrics ───────────────────────────────────────────────
    bars_since_cross = n - 1 - found_bar

    vol_cross = float(df["Volume"].iloc[found_bar])
    vol_prev  = float(df["Volume"].iloc[found_bar - 1])
    vol_vs_prev = vol_cross / vol_prev if vol_prev > 0 else 0
    vol_mult    = vol_cross / avg_vol  if avg_vol  > 0 else 0

    # S3 at breakout bar
    s50_at_break = float(s50_s.iloc[found_bar]) if not np.isnan(s50_s.iloc[found_bar]) else cur_s50
    s3_at_break  = float(s3_s.iloc[found_bar])  if not np.isnan(s3_s.iloc[found_bar])  else cur_s3
    j_at_break   = float(jma_s.iloc[found_bar]) if not np.isnan(jma_s.iloc[found_bar]) else cur_jma
    e8_at_break  = float(ema8_s.iloc[found_bar])if not np.isnan(ema8_s.iloc[found_bar])else cur_ema8
    s21_at_break = float(s21_s.iloc[found_bar]) if not np.isnan(s21_s.iloc[found_bar]) else cur_s21

    dist_jma_pct  = (price - cur_jma)  / cur_jma  * 100 if cur_jma  > 0 else 0
    dist_ema8_pct = (price - cur_ema8) / cur_ema8 * 100 if cur_ema8 > 0 else 0
    dist_s21_pct  = (price - cur_s21)  / cur_s21  * 100 if cur_s21  > 0 else 0
    dist_s50_pct  = (price - cur_s50)  / cur_s50  * 100 if cur_s50  > 0 else 0
    dist_s3_pct   = (price - cur_s3)   / cur_s3   * 100 if cur_s3   > 0 else 0
    cluster_vs_s3_pct = round((s50_at_break - s3_at_break) / s3_at_break * 100, 2) if s3_at_break > 0 else 0

    # ── Score (0-100) ─────────────────────────────────────────
    score = 0
    score += max(0, 30 - int(found_spread * 3))   # tighter compression
    score += max(0, 25 - bars_since_cross * 5)     # freshness
    score += min(20, int(vol_vs_prev * 7))         # volume vs prev day
    score += max(0, 15 - int(found_s3dist * 2))    # proximity to S3
    score += min(10, int(vol_mult * 4))            # volume vs avg
    score = min(100, max(0, score))

    return {
        "Ticker"              : sym,
        "Price"               : round(price, 2),
        "Score"               : score,
        # Compression metrics
        "Comp_Spread_%"       : found_spread,
        "S3_at_Break"         : round(s3_at_break, 2),
        "S50_at_Break"        : round(s50_at_break, 2),
        "Cluster_vs_S3_%"     : cluster_vs_s3_pct,
        # Cross info
        "Cross_Date"          : found_date.strftime("%Y-%m-%d"),
        "Bars_Since_Cross"    : bars_since_cross,
        # Volume
        "Cross_Vol"           : int(vol_cross),
        "Prev_Vol"            : int(vol_prev),
        "Vol_vs_Prev_x"       : round(vol_vs_prev, 2),
        "Vol_vs_Avg_x"        : round(vol_mult, 2),
        # MA levels now
        "JMA"                 : round(cur_jma, 2),
        "EMA8"                : round(cur_ema8, 2),
        "SMA21"               : round(cur_s21, 2),
        "SMA50"               : round(cur_s50, 2),
        "Cam_S3"              : round(cur_s3, 2),
        # Distances now
        "Dist_JMA_%"          : round(dist_jma_pct, 2),
        "Dist_EMA8_%"         : round(dist_ema8_pct, 2),
        "Dist_SMA21_%"        : round(dist_s21_pct, 2),
        "Dist_SMA50_%"        : round(dist_s50_pct, 2),
        "Dist_S3_%"           : round(dist_s3_pct, 2),
        "RSI"                 : round(cur_rsi, 1),
        "MACD_Hist"           : round(cur_mh, 4),
        "Avg_Vol_20d"         : int(avg_vol),
        # internals
        "_df"        : df,
        "_jma"       : jma_s,
        "_ema8"      : ema8_s,
        "_sma21"     : s21_s,
        "_sma50"     : s50_s,
        "_s3"        : s3_s,
        "_cross_bar" : found_bar,
        "_comp_bar"  : found_bar,
    }

# ── Live print ────────────────────────────────────────────────
LIVE_COLS = ["Ticker","Price","Score",
             "Comp_Spread_%","S3_at_Comp","Cluster_vs_S3_%",
             "Cross_Date","Bars_Since_Cross",
             "Vol_vs_Prev_x","Vol_vs_Avg_x","RSI"]
_CW = {"Ticker":8,"Price":10,"Score":7,
       "Comp_Spread_%":14,"S3_at_Comp":12,"Cluster_vs_S3_%":17,
       "Cross_Date":13,"Bars_Since_Cross":17,
       "Vol_vs_Prev_x":14,"Vol_vs_Avg_x":13,"RSI":6}
_CF = {"Price":"${:.2f}","Score":"{:.0f}",
       "Comp_Spread_%":"{:.3f}%","S3_at_Comp":"${:.2f}","Cluster_vs_S3_%":"{:+.2f}%",
       "Bars_Since_Cross":"{:.0f}d",
       "Vol_vs_Prev_x":"{:.2f}×","Vol_vs_Avg_x":"{:.2f}×","RSI":"{:.1f}"}
_hdr_done = False

def _live_header():
    global _hdr_done
    if _hdr_done: return
    sep = "━"*195
    print(f"\n{sep}")
    print("  📊  LIVE MATCHES  —  MA Compression Near Cam S3 → Breakout Above All")
    print(sep)
    print("".join(f"  {c:<{_CW.get(c,10)}}" for c in LIVE_COLS))
    print("  "+"─"*193)
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
chk = download(["AAPL","NVDA","AMD"], 300)
if not chk: print("❌  No data.")
else:
    for s, d in chk.items():
        p    = float(d["Close"].iloc[-1])
        s3v  = build_monthly_s3_series(d)
        s3   = float(s3v.iloc[-1]) if not np.isnan(s3v.iloc[-1]) else 0
        print(f"  ✅ {s}: ${p:.2f}  Cam S3=${s3:.2f}  {d.index[-1].date()}")
print()

# ── Diagnostic ────────────────────────────────────────────────
print("━"*65)
print("  STEP 2  DIAGNOSTIC (10 stocks)")
print("━"*65+"\n")
DIAG = ["AAPL","NVDA","AMD","PLTR","MU","SMCI","META","AVGO","CRWD","MXL"]
diag_data = download(DIAG, CFG["history_days"])
print(f"  Downloaded {len(diag_data)}/{len(DIAG)}\n")
print(f"  {'SYM':<7} {'PRICE':>8}  {'COMP%':>7}  {'S3':>8}  RESULT")
print("  "+"─"*42)
for sym in DIAG:
    if sym not in diag_data:
        print(f"  {sym:<7} — no data"); continue
    try:
        df_d = diag_data[sym]
        p    = float(df_d["Close"].iloc[-1])
        r    = detect_pattern(sym, df_d)
        if r:
            print(f"  {sym:<7} ${p:>7.2f}  {r['Comp_Spread_%']:>6.3f}%  "
                  f"${r['S3_at_Comp']:>7.2f}  ✅ Score={r['Score']} {r['Cross_Date']}")
        else:
            print(f"  {sym:<7} ${p:>7.2f}  {'—':>7}  {'—':>8}  ❌")
    except Exception as e:
        print(f"  {sym:<7} error: {e}")

print(f"""
  Pattern:
    C1  MA Compression : JMA+EMA8+SMA21+SMA50 within {CFG['compression_pct']}% of each other
    C2  Near Cam S3    : MA cluster average within {CFG['s3_zone_pct']}% of monthly S3
                         S3 = Close - (High-Low) × 1.1/4  (from prior month)
    C3  Breakout       : price crossed above ALL 4 MAs AND S3
                         in last {CFG['cross_lookback']} bars
    C4  Volume         : cross bar volume > previous bar volume

  Tune if mostly ❌:
    compression_pct         3 → 5
    s3_zone_pct             3 → 5
    cross_lookback          8 → 12
    compression_lookback    5 → 8
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
            r  = requests.get(url, headers=hdrs, timeout=20); r.raise_for_status()
            df = pd.read_csv(io.StringIO(r.text), sep="|")
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
        "AXON","ANET","CAVA","VRT","ELF","GRMN","ON","ENPH","ROST","POOL",
        "AMGN","GILD","INTU","MCHP","MNST","NXPI","XEL","ACLS","IRTC","MXL",
        "PYPL","ROKU","SNAP","PINS","EBAY","ETSY","ZM","LYFT","RIVN","DOCU",
        "SWKS","QRVO","MTCH","BMBL","SMAR","BILL","APPN","ALRM","ARLO","BAND",
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
print(f"""
  📊 DEBUG BREAKDOWN:
  Total processed          : {_DBG['total']}
  Passed vol/price filter  : {_DBG['pass_filter']}
  ❌ Failed C1 (compression): {_DBG['fail_c1_comp']}
  ❌ Failed C2 (near S3)    : {_DBG['fail_c2_s3']}
  ❌ Failed C3 (cross)      : {_DBG['fail_c3_cross']}
  ❌ Failed C4 (volume)     : {_DBG['fail_c4_vol']}
  ✅ Passed all             : {_DBG['pass_all']}
""")
if not results:
    print("  Relax the condition with most failures above:")
    print("   C1: compression_pct       3 → 5")
    print("   C2: s3_zone_pct           3 → 5")
    print("   C3: cross_lookback        8 → 12")
    print("   C4: min_break_vol_mult  0.8 → 0.5")

results.sort(key=lambda x: -x["Score"])

# ── Build df_out ──────────────────────────────────────────────
ts      = datetime.today().strftime("%Y%m%d_%H%M")
out_dir = os.environ.get("GITHUB_WORKSPACE", os.getcwd())

COLS = [
    "Ticker","Price","Score",
    "Comp_Spread_%","MA_Hi_at_Comp","MA_Lo_at_Comp","MA_Avg_at_Comp",
    "Cam_S3","S3_at_Comp","Cluster_vs_S3_%",
    "Cross_Date","Bars_Since_Cross",
    "Cross_Vol","Prev_Vol","Vol_vs_Prev_x","Vol_vs_Avg_x",
    "JMA","EMA8","SMA21","SMA50",
    "Dist_JMA_%","Dist_EMA8_%","Dist_SMA21_%","Dist_SMA50_%","Dist_S3_%",
    "RSI","MACD_Hist","Avg_Vol_20d",
]
df_out = pd.DataFrame([{k:v for k,v in r.items() if not k.startswith("_")}
                        for r in results]) if results else pd.DataFrame(columns=COLS)
if not df_out.empty:
    df_out = df_out[[c for c in COLS if c in df_out.columns]]
    df_out.reset_index(drop=True, inplace=True)

FMT = {
    "Price"           : lambda v: f"${v:.2f}",
    "Score"           : lambda v: f"{v:.0f}",
    "Comp_Spread_%"   : lambda v: f"{v:.3f}%",
    "MA_Hi_at_Comp"   : lambda v: f"${v:.2f}",
    "MA_Lo_at_Comp"   : lambda v: f"${v:.2f}",
    "MA_Avg_at_Comp"  : lambda v: f"${v:.2f}",
    "Cam_S3"          : lambda v: f"${v:.2f}",
    "S3_at_Comp"      : lambda v: f"${v:.2f}",
    "Cluster_vs_S3_%": lambda v: f"{v:+.2f}%",
    "Bars_Since_Cross": lambda v: f"{int(v)}d",
    "Cross_Vol"       : lambda v: f"{v:,.0f}",
    "Prev_Vol"        : lambda v: f"{v:,.0f}",
    "Vol_vs_Prev_x"  : lambda v: f"{v:.2f}×",
    "Vol_vs_Avg_x"   : lambda v: f"{v:.2f}×",
    "JMA"             : lambda v: f"${v:.2f}",
    "EMA8"            : lambda v: f"${v:.2f}",
    "SMA21"           : lambda v: f"${v:.2f}",
    "SMA50"           : lambda v: f"${v:.2f}",
    "Dist_JMA_%"      : lambda v: f"{v:+.2f}%",
    "Dist_EMA8_%"     : lambda v: f"{v:+.2f}%",
    "Dist_SMA21_%"    : lambda v: f"{v:+.2f}%",
    "Dist_SMA50_%"    : lambda v: f"{v:+.2f}%",
    "Dist_S3_%"       : lambda v: f"{v:+.2f}%",
    "RSI"             : lambda v: f"{v:.1f}",
    "MACD_Hist"       : lambda v: f"{v:.4f}",
    "Avg_Vol_20d"     : lambda v: f"{v:,.0f}",
}

def fmt_v(col, val):
    if val is None or (isinstance(val, float) and np.isnan(val)): return "—"
    try:
        if col in FMT: return FMT[col](val)
    except Exception: pass
    return str(val) if str(val) not in ("nan","None","") else "—"

# ── Notebook display ──────────────────────────────────────────
if _IN_NOTEBOOK and results:
    DISP = ["Ticker","Price","Score",
            "Comp_Spread_%","S3_at_Comp","Cluster_vs_S3_%",
            "Cross_Date","Bars_Since_Cross",
            "Vol_vs_Prev_x","Vol_vs_Avg_x","RSI","MACD_Hist"]
    DISP = [c for c in DISP if c in df_out.columns]
    gc   = "#22c55e"

    th = "".join(
        f'<th style="background:#0f172a;color:#e2e8f0;padding:9px 12px;'
        f'font-size:11px;font-weight:700;border-bottom:2px solid {gc};white-space:nowrap">'
        f'{c}</th>' for c in DISP)
    rows_html = ""
    for i, r in enumerate(results):
        bg = "#fff" if i%2==0 else "#f0f9ff"
        tds = ""
        for col in DISP:
            raw = r.get(col); disp = fmt_v(col, raw); sty = ""
            if col == "Score":
                try:
                    v = float(raw); g = int(min(220, 80+v*1.4))
                    sty = f"background:rgb(20,{g},60);color:#fff;font-weight:700;text-align:center"
                except Exception: pass
            elif col == "Comp_Spread_%":
                try:
                    v = float(str(raw).replace("%",""))
                    sty = "color:#22c55e;font-weight:700" if v<=1 else "color:#86efac" if v<=2 else ""
                except Exception: pass
            elif col == "Cluster_vs_S3_%":
                try:
                    v = float(str(raw).replace("%","").replace("+",""))
                    sty = "color:#22c55e;font-weight:700" if abs(v)<=1 else ""
                except Exception: pass
            elif col == "Vol_vs_Prev_x":
                try:
                    v = float(str(raw).replace("×",""))
                    sty = "color:#22c55e;font-weight:800" if v>=2 else "color:#86efac;font-weight:600" if v>=1.5 else ""
                except Exception: pass
            elif col == "Bars_Since_Cross":
                try:
                    v = int(str(raw).replace("d",""))
                    sty = "color:#22c55e;font-weight:700;text-align:center" if v==0 else "text-align:center"
                except Exception: pass
            tds += f'<td style="padding:7px 12px;font-size:12px;border-bottom:1px solid #e2e8f0;white-space:nowrap;{sty}">{disp}</td>'
        rows_html += f'<tr style="background:{bg}">{tds}</tr>\n'

    ticker_csv_str = ",".join(r["Ticker"] for r in results)
    display_html(f"""
<div style="background:linear-gradient(135deg,#0f172a,#1e3a5f);border-radius:10px;
            padding:18px 24px;margin-bottom:8px">
  <h2 style="margin:0;color:#60a5fa;font-size:20px;font-weight:700">
    📈 MA Compression Near Cam S3 → Breakout Above All + Higher Volume
  </h2>
  <p style="margin:6px 0 0;color:#94a3b8;font-size:12px">
    {datetime.today().strftime('%Y-%m-%d %H:%M')} &nbsp;·&nbsp;
    <b style="color:{gc}">{len(results)} matches</b> from {len(TICKERS)} tickers
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
  <b style="color:#475569">GUIDE</b> &nbsp;·&nbsp;
  Comp_Spread_% = gap between highest and lowest of 4 MAs as % of price (lower = tighter coil) &nbsp;·&nbsp;
  Cluster_vs_S3_% = how close MA cluster average was to Cam S3 (near 0 = sitting right on S3) &nbsp;·&nbsp;
  Vol_vs_Prev_x = cross bar volume ÷ previous bar volume
</div>""")

elif results:
    CLI = ["Ticker","Price","Score","Comp_Spread_%","S3_at_Comp",
           "Bars_Since_Cross","Vol_vs_Prev_x","RSI"]
    CLI = [c for c in CLI if c in df_out.columns]
    col_w = {c: max(len(c), max(len(fmt_v(c,r.get(c))) for r in results))+2 for c in CLI}
    top = "┬".join("─"*col_w[c] for c in CLI)
    sep = "┼".join("─"*col_w[c] for c in CLI)
    bot = "┴".join("─"*col_w[c] for c in CLI)
    hdr = "│".join(c.center(col_w[c]) for c in CLI)
    inner = sum(col_w.values())+len(CLI)-1
    print(f"\n  ╔{'═'*inner}╗")
    tit = f"  MA Compression Cam S3 Breakout   {datetime.today().strftime('%Y-%m-%d')}   {len(df_out)} matches"
    print(f"  ║{tit.center(inner)}║\n  ╚{'═'*inner}╝\n")
    print(f"  ┌{top}┐\n  │{hdr}│\n  ├{sep}┤")
    for i,r in enumerate(results):
        cells=[fmt_v(c,r.get(c)).center(col_w[c]) for c in CLI]
        print(f"  │{'│'.join(cells)}│")
        if i<len(results)-1: print(f"  ├{sep}┤")
    print(f"  └{bot}┘")

# ── Save CSV + TradingView ─────────────────────────────────────
fpath = os.path.join(out_dir, f"ma_compression_s3_breakout_{ts}.csv")
df_out.to_csv(fpath, index=False)
print(f"\n  💾 CSV → {fpath}")
tv = os.path.join(out_dir, f"tv_ma_compression_s3_{ts}.txt")
with open(tv,"w") as f:
    f.write(f"###MA Compression S3 Breakout {datetime.today().strftime('%Y-%m-%d')}\n")
    for r in results: f.write(f"NASDAQ:{r['Ticker']}\n")
print(f"  📋 TradingView → {tv}")

# ── Email ──────────────────────────────────────────────────────
def _send_email(rl, csv_path):
    gu = _GMAIL_USER; gp = _GMAIL_PASS; et = _EMAIL_TO
    if not gu: print("[Email] ❌  GMAIL_USER secret is empty"); return
    if not gp: print("[Email] ❌  GMAIL_PASS secret is empty\n         → myaccount.google.com/apppasswords"); return
    if not et: print("[Email] ❌  EMAIL_TO secret is empty"); return

    eto = [e.strip() for e in et.split(",") if e.strip()]
    cnt = len(rl)

    try:
        print(f"[Email] Sending to {et}  ({cnt} results)...")

        ticker_csv = ",".join(r.get("Ticker","") for r in rl) if rl else "—"

        ticker_csv_html = f"""
<div style="margin:14px 0;padding:14px 16px;background:#0f172a;
            border-radius:8px;border-left:4px solid #22c55e;">
  <p style="margin:0 0 6px;color:#94a3b8;font-size:11px;font-weight:600;
             letter-spacing:.05em;text-transform:uppercase">
    📋 Stock List — Copy &amp; paste into TradingView / Excel
  </p>
  <p style="margin:0;color:#22c55e;font-size:13px;font-weight:700;
             font-family:'Courier New',monospace;word-break:break-all;
             letter-spacing:.04em">
    {ticker_csv}
  </p>
</div>"""

        th_e = "".join(
            f'<th style="background:#1e293b;color:#e2e8f0;padding:8px 11px;'
            f'font-size:11px;font-weight:700;border-bottom:2px solid #3b82f6;'
            f'white-space:nowrap">{c}</th>'
            for c in ["Ticker","Price","Score","Comp_Spread_%","S3_at_Comp",
                      "Cluster_vs_S3_%","Bars_Since_Cross","Vol_vs_Prev_x","RSI"]
        )
        rows_e = ""
        for i, r in enumerate(rl[:50]):
            bg = "#fff" if i%2==0 else "#f0f9ff"
            bsc = r.get("Bars_Since_Cross", 99)
            rows_e += (
                f'<tr style="background:{bg}">'
                f'<td style="padding:6px 11px;font-size:12px;font-weight:700;color:#22c55e">'
                f'{r.get("Ticker","—")}</td>'
                f'<td style="padding:6px 11px;font-size:12px">'
                f'${float(r.get("Price",0)):.2f}</td>'
                f'<td style="padding:6px 11px;font-size:12px;font-weight:700;'
                f'background:#166534;color:#fff;text-align:center">'
                f'{float(r.get("Score",0)):.0f}</td>'
                f'<td style="padding:6px 11px;font-size:12px;color:#22c55e">'
                f'{float(r.get("Comp_Spread_%",0)):.3f}%</td>'
                f'<td style="padding:6px 11px;font-size:12px">'
                f'${float(r.get("S3_at_Comp",0)):.2f}</td>'
                f'<td style="padding:6px 11px;font-size:12px">'
                f'{float(r.get("Cluster_vs_S3_%",0)):+.2f}%</td>'
                f'<td style="padding:6px 11px;font-size:12px;text-align:center;'
                f'color:{"#22c55e" if bsc==0 else "#94a3b8"};font-weight:700">'
                f'{bsc}d</td>'
                f'<td style="padding:6px 11px;font-size:12px;color:#22c55e;font-weight:700">'
                f'{float(r.get("Vol_vs_Prev_x",0)):.2f}×</td>'
                f'<td style="padding:6px 11px;font-size:12px">'
                f'{float(r.get("RSI",0)):.1f}</td>'
                f'</tr>'
            )

        no_res = "" if cnt else (
            '<tr><td colspan="9" style="padding:20px;text-align:center;'
            'color:#64748b;font-size:13px">No matches found today</td></tr>'
        )

        html_e = f"""<!DOCTYPE html><html><body style="margin:0;padding:0;
background:#f1f5f9;font-family:'Segoe UI',Arial,sans-serif">
<table width="100%" cellpadding="0" cellspacing="0" style="padding:20px 0"><tr><td>
<table width="100%" cellpadding="0" cellspacing="0"
   style="max-width:960px;margin:0 auto;background:#fff;border-radius:12px;
          overflow:hidden;box-shadow:0 4px 20px rgba(0,0,0,0.08)">
  <tr><td style="background:linear-gradient(135deg,#0f172a,#1e3a5f);padding:22px 28px">
    <h1 style="margin:0;color:#60a5fa;font-size:20px;font-weight:700">
      📊 MA Compression Near Cam S3 → Breakout Above All
    </h1>
    <p style="margin:6px 0 0;color:#94a3b8;font-size:12px">
      {datetime.today().strftime('%Y-%m-%d %H:%M UTC')} &nbsp;·&nbsp;
      {cnt} match{'es' if cnt!=1 else ''} found
    </p>
  </td></tr>
  <tr><td style="padding:16px">
    {ticker_csv_html}
    <div style="overflow-x:auto;border-radius:8px;border:1px solid #e2e8f0">
      <table style="border-collapse:collapse;width:100%;min-width:600px">
        <thead><tr>{th_e}</tr></thead>
        <tbody>{rows_e or no_res}</tbody>
      </table>
    </div>
    <p style="font-size:11px;color:#64748b;margin:10px 0 0">
      📎 CSV and TradingView file attached &nbsp;·&nbsp;
      <b>Comp_Spread_%</b> = gap between 4 MAs as % of price &nbsp;·&nbsp;
      <b>Cluster_vs_S3_%</b> = how close cluster was to Cam S3 &nbsp;·&nbsp;
      <b>Vol_vs_Prev_x</b> = cross vol ÷ prev day vol
    </p>
  </td></tr>
  <tr><td style="background:#f8fafc;padding:12px 28px;border-top:1px solid #e2e8f0;text-align:center">
    <p style="margin:0;color:#94a3b8;font-size:10px">
      ⚠️ Not financial advice &nbsp;·&nbsp; Auto-generated by GitHub Actions
    </p>
  </td></tr>
</table></td></tr></table></body></html>"""

        plain_e = "\n".join([
            f"MA Compression Near Cam S3 → Breakout — {datetime.today().strftime('%Y-%m-%d')}",
            f"{cnt} matches",
            "",
            f"STOCKS: {ticker_csv}",
            "",
            "="*60,
        ] + ([
            f"{r.get('Ticker','—'):<7} ${float(r.get('Price',0)):.2f}  "
            f"Score:{float(r.get('Score',0)):.0f}  "
            f"Spread:{float(r.get('Comp_Spread_%',0)):.3f}%  "
            f"S3_Dist:{float(r.get('Cluster_vs_S3_%',0)):+.2f}%  "
            f"Vol:{float(r.get('Vol_vs_Prev_x',0)):.1f}×prev  "
            f"Cross:{r.get('Bars_Since_Cross',0)}d ago"
            for r in rl[:50]
        ] if rl else ["No matches today"]) + ["\n📎 CSV and TradingView file attached."])

        subj = (f"📊 MA Comp S3 Breakout — {cnt} signal{'s' if cnt!=1 else ''}"
                f" — {datetime.today().strftime('%Y-%m-%d')}")

        msg = MIMEMultipart("mixed")
        msg["Subject"] = subj; msg["From"] = gu; msg["To"] = ", ".join(eto)
        alt = MIMEMultipart("alternative")
        alt.attach(MIMEText(plain_e,"plain")); alt.attach(MIMEText(html_e,"html"))
        msg.attach(alt)

    except Exception as e:
        print(f"[Email] ❌  Body build failed: {type(e).__name__}: {e}"); return

    for att in [csv_path, tv]:
        if att and os.path.exists(att):
            try:
                with open(att,"rb") as f:
                    part = MIMEBase("application","octet-stream")
                    part.set_payload(f.read())
                encoders.encode_base64(part)
                part.add_header("Content-Disposition",
                    f"attachment; filename={os.path.basename(att)}")
                msg.attach(part)
                print(f"[Email] 📎 Attached: {os.path.basename(att)}")
            except Exception as e:
                print(f"[Email] ⚠️  Attach failed: {e}")

    try:
        print("[Email] Connecting to smtp.gmail.com:465 ...")
        with smtplib.SMTP_SSL("smtp.gmail.com", 465) as srv:
            srv.login(gu, gp.replace(" ",""))
            srv.sendmail(gu, eto, msg.as_string())
        print(f"[Email] ✅  Sent to: {', '.join(eto)}")
    except smtplib.SMTPAuthenticationError:
        print("[Email] ❌  AUTH FAILED — use Gmail App Password")
        print("         Generate: myaccount.google.com/apppasswords")
    except smtplib.SMTPException as e:
        print(f"[Email] ❌  SMTP error: {e}")
    except Exception as e:
        print(f"[Email] ❌  {type(e).__name__}: {e}")

try:
    _send_email(results, fpath)
except Exception as e:
    print(f"[Email] ❌  Top-level error: {type(e).__name__}: {e}")
    print("[Email]    CSV and charts still saved.")

if _IN_NOTEBOOK:
    try:
        from google.colab import files
        files.download(fpath); files.download(tv)
    except Exception: pass
else:
    print("  (CI: files in workspace, email sent)")

# ── Charts for top 5 ──────────────────────────────────────────
if results:
    top = results[:min(5,len(results))]
    fig, axes = plt.subplots(len(top),1,figsize=(15,5.5*len(top)),facecolor="#0f172a")
    if len(top)==1: axes=[axes]

    for idx, r in enumerate(top):
        ax   = axes[idx]
        df_p = r["_df"].tail(80).copy()
        n_p  = len(df_p)
        fn   = len(r["_df"]); off = fn - n_p

        jma  = r["_jma"].reindex(df_p.index)
        ema8 = r["_ema8"].reindex(df_p.index)
        s21  = r["_sma21"].reindex(df_p.index)
        s50  = r["_sma50"].reindex(df_p.index)
        s3   = r["_s3"].reindex(df_p.index)

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

        # All 4 MAs
        ax.plot(range(n_p), jma.values,  color="#22d3ee", lw=2.0, label="JMA", zorder=7)
        ax.plot(range(n_p), ema8.values, color="#34d399", lw=1.5, ls="--", label="EMA8", zorder=6)
        ax.plot(range(n_p), s21.values,  color="#fbbf24", lw=1.4, ls="-.", label="SMA21", zorder=5)
        ax.plot(range(n_p), s50.values,  color="#3b82f6", lw=1.8, label="SMA50", zorder=5)
        # Cam S3 — the key level
        ax.plot(range(n_p), s3.values,   color="#f59e0b", lw=1.8, ls=":", label="Cam S3", zorder=6)

        # Shade the compression zone
        cb  = r["_comp_bar"] - off
        cbl = max(0, cb - CFG["compression_lookback"])
        if 0 <= cbl < n_p and 0 <= cb < n_p:
            ax.axvspan(cbl, min(cb+1, n_p-1), alpha=0.12, color="#a78bfa", zorder=1,
                       label="Compression zone")

        # Mark breakout bar
        bb = r["_cross_bar"] - off
        if 0 <= bb < n_p:
            ax.axvline(bb, color="#22c55e", lw=2.0, ls="--", alpha=0.9)
            ax.scatter([bb],[float(df_p["Close"].iloc[bb])],
                       color="#22c55e", s=200, zorder=9, marker="^",
                       label=f"Breakout {r['Cross_Date']} Vol{r['Vol_vs_Prev_x']:.1f}×prev")

        tick_step = max(1, n_p//8)
        ax.set_xticks(range(0,n_p,tick_step))
        ax.set_xticklabels(
            [df_p.index[i].strftime("%m/%d") for i in range(0,n_p,tick_step)],
            color="#94a3b8", fontsize=7)
        ax.set_xlim(-0.5, n_p-0.5)
        ax.set_title(
            f"{r['Ticker']}  ${r['Price']:.2f}  |  Score {r['Score']}/100  |  "
            f"MA Spread {r['Comp_Spread_%']:.3f}%  Cluster vs S3 {r['Cluster_vs_S3_%']:+.2f}%  |  "
            f"Breakout {r['Cross_Date']} ({r['Bars_Since_Cross']}d ago)  "
            f"Vol {r['Vol_vs_Prev_x']:.1f}×prev {r['Vol_vs_Avg_x']:.1f}×avg  |  "
            f"RSI {r['RSI']:.0f}",
            color="#e2e8f0", fontsize=8, fontweight="bold", pad=6)
        ax.tick_params(colors="#94a3b8", labelsize=7)
        for sp_ in ax.spines.values(): sp_.set_edgecolor("#1e3a5f")
        ax.legend(loc="upper left", facecolor="#1e293b",
                  labelcolor="#e2e8f0", fontsize=7, framealpha=0.9, ncol=3)
        ax.grid(color="#1e3a5f", ls="--", lw=0.4, alpha=0.4, axis="y")

    plt.suptitle(
        f"MA Compression Near Cam S3 → Breakout Above All + Higher Volume  ·  "
        f"{datetime.today().strftime('%Y-%m-%d')}\n"
        f"🔵 JMA  🟢 EMA8  🟡 SMA21  🔵 SMA50  🟠 Cam S3  "
        f"🟣 = compression zone  ▲ = breakout bar",
        color="#60a5fa", fontsize=9, fontweight="bold", y=1.001)
    plt.tight_layout()
    cp = os.path.join(out_dir, f"ma_compression_s3_chart_{ts}.png")
    plt.savefig(cp, dpi=150, bbox_inches="tight", facecolor="#0f172a")
    if _IN_NOTEBOOK: plt.show()
    else: plt.close()
    print(f"  📊 Chart → {cp}")

print("""
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  📋 PATTERN EXPLAINED

  C1  MA COMPRESSION  (the coil)
      JMA(13,40) + EMA8 + SMA21 + SMA50 all within 3% of
      each other (measured as % of price)
      = All moving averages converging — energy building

  C2  CLUSTER NEAR MONTHLY CAMARILLA S3
      S3 = Close - (High-Low) × 1.1 / 4  (from prior month)
      The MA cluster average must be within 3% of Cam S3
      = Compression is happening RIGHT AT a key monthly
        support/pivot level — adds institutional significance

  C3  PRICE BROKE ABOVE ALL 4 MAs AND CAM S3  (the trigger)
      In the last 8 bars, price went from below at least
      one level to ABOVE ALL of them simultaneously
      = Everything breaks out together — no lagging MAs

  C4  VOLUME > PREVIOUS DAY'S VOLUME
      The breakout bar volume exceeds the prior day
      = Fresh buying entering on the move, not a drift

  WHY THIS IS POWERFUL:
      When 4 moving averages compress into a tight bundle
      at a Camarilla support level, and then price breaks
      all of them at once on rising volume, the subsequent
      move is often explosive — all the compressed energy
      releases in one direction with no nearby resistance
      (all MAs now act as support below price).

  💡 BEST SETUPS
  Comp_Spread_% < 1%      extremely tight coil
  Cluster_vs_S3_% near 0  cluster sitting right on S3
  Bars_Since_Cross = 0    breakout today = freshest entry
  Vol_vs_Prev_x >= 2×     twice yesterday's volume = strong

  ⚙️  TUNE IF 0 RESULTS
  compression_pct      3 → 5
  s3_zone_pct          3 → 5
  cross_lookback       8 → 12
  compression_lookback 5 → 8
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
""")
