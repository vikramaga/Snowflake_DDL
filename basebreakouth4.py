# ============================================================
# NASDAQ — Extended Base Breakout: Cam H4 + JMA + MACD + RSI
# ============================================================
#
# PATTERN LOGIC:
#
#  C1 — BEARISH STRUCTURE (the base condition)
#      SMA50 < SMA150  (medium-term below long-term = bearish)
#      Both SMA50 AND SMA150 are below the monthly Camarilla H4
#      = Stock has been in a downtrend / extended weakness,
#        with the monthly H4 acting as overhead resistance
#
#  C2 — 5-MONTH CONSOLIDATION (the coil)
#      Over the last consolidation_bars (~5 months = ~105 bars),
#      price was trading in a tight range:
#        Range = (highest High - lowest Low) / lowest Low
#        Must be <= max_consolidation_range_pct%
#      Price stayed mostly BELOW or near the Camarilla H4 level
#      during this consolidation window
#      = Sellers exhausted, range-bound accumulation underway
#
#  C3 — PRICE CLOSED ABOVE MONTHLY CAMARILLA H4 (the trigger)
#      H4 = Close + (High - Low) × 1.1 / 2  (from prior month)
#      EXACT 1-bar break:
#        Bar[-2] close < H4   ← was below H4 yesterday
#        Bar[-1] close >= H4  ← closed above H4 today
#      OR price closed above H4 within the last cross_lookback bars
#
#  C4 — ABOVE JMA (Jurik Moving Average)
#      Current price > JMA(13,40)
#      = Short-term trend reclaimed from below
#
#  C5 — MACD LINE ABOVE ZERO
#      MACD line (EMA12 - EMA26) > 0
#      (Not just the histogram — the actual MACD line)
#      = Medium-term momentum has turned positive
#
#  C6 — RSI ABOVE 50
#      RSI(14) > 50
#      = Momentum is above neutral, buyers in control
#
# LOGIC FLOW:
#   Extended bearish structure (SMA50 < SMA150, both below H4)
#   → 5+ months of tight consolidation below H4
#   → Price breaks above monthly Camarilla H4
#   → JMA confirms (price above it)
#   → MACD line crosses zero (medium-term momentum flips)
#   → RSI > 50 (momentum confirming)
#   = High-conviction base breakout from extended weakness
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
    print(f"  ℹ️  Email will be SKIPPED this run")
print("━"*65)
print()

# ── CONFIG ────────────────────────────────────────────────────
CFG = {
    "history_days"                 : 550,   # need ~18 months for context

    # ── MA periods ────────────────────────────────────────────
    "jma_period"                   : 13,
    "jma_phase"                    : 40,
    "sma50_period"                 : 50,
    "sma150_period"                : 150,
    "macd_fast"                    : 12,
    "macd_slow"                    : 26,
    "macd_signal"                  : 9,
    "rsi_period"                   : 14,

    # ── C1: Bearish structure ──────────────────────────────────
    # SMA50 must be below SMA150
    "require_sma50_below_sma150"   : True,
    # Both MAs must be below Camarilla H4 AT THE START of
    # the consolidation window (not necessarily today)
    "require_mas_below_h4"         : True,

    # ── C2: Consolidation window ───────────────────────────────
    # Number of bars to define "5 months" of consolidation
    # (~21 trading days/month × 5 = 105)
    "consolidation_bars"           : 105,
    # Max price range during consolidation as % of low
    # (how tight the coil must be)
    "max_consolidation_range_pct"  : 35.0,

    # ── C3: Camarilla H4 break ─────────────────────────────────
    # Look back this many bars for the H4 cross
    "cross_lookback"               : 10,

    # ── C4: JMA ────────────────────────────────────────────────
    # Current price must be above JMA

    # ── C5: MACD line above zero ───────────────────────────────
    # MACD LINE (not histogram) must be > 0

    # ── C6: RSI above 50 ───────────────────────────────────────
    "rsi_min"                      : 50,

    # ── Volume ────────────────────────────────────────────────
    "vol_avg_bars"                 : 20,
    "min_break_vol_mult"           : 1.0,   # break bar vol >= 1x avg

    # ── Filters ───────────────────────────────────────────────
    "min_avg_volume"               : 80_000,
    "min_price"                    : 1.0,

    "batch_size"                   : 50,
    "batch_sleep"                  : 1.5,
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

def calc_macd(close, fast=12, slow=26, signal=9):
    ef  = calc_ema(close, fast)
    es  = calc_ema(close, slow)
    ml  = ef - es          # MACD LINE
    sig = calc_ema(ml, signal)
    return ml, sig, ml - sig   # line, signal, histogram

# ── Camarilla H4 ─────────────────────────────────────────────
def cam_h4(high, low, close):
    return close + (high - low) * 1.1 / 2.0

def get_monthly_h4(df):
    """
    Returns Camarilla H4 for the CURRENT month
    (built from previous completed month's H/L/C)
    and the PREVIOUS completed month's H4.
    Returns (h4_this, h4_prev) — either can be None.
    """
    df = df.copy()
    df.index = pd.to_datetime(df.index)
    today  = pd.Timestamp.today().normalize()
    levels = []
    for offset in [1, 2]:
        p   = today.to_period("M") - offset
        sub = df[df.index.to_period("M") == p]
        if len(sub) >= 5:
            hi = float(sub["High"].max())
            lo = float(sub["Low"].min())
            cl = float(sub["Close"].iloc[-1])
            levels.append(round(cam_h4(hi, lo, cl), 4))
        else:
            levels.append(None)
    return levels   # [this_month_H4, prev_month_H4]

def build_monthly_h4_series(df):
    """
    Builds a per-bar series of the H4 level applicable
    to each bar (H4 for that bar's month = from prior month).
    """
    df = df.copy()
    df.index = pd.to_datetime(df.index)
    periods = df.index.to_period("M")
    unique_months = sorted(periods.unique())

    month_h4 = {}
    for i, mp in enumerate(unique_months):
        if i == 0: continue
        prev_mp = unique_months[i - 1]
        sub = df[periods == prev_mp]
        if len(sub) < 5: continue
        hi = float(sub["High"].max())
        lo = float(sub["Low"].min())
        cl = float(sub["Close"].iloc[-1])
        month_h4[mp] = cam_h4(hi, lo, cl)

    h4_vals = [month_h4.get(mp, np.nan) for mp in periods]
    return pd.Series(h4_vals, index=df.index)

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

# ── Debug counters ────────────────────────────────────────────
_DBG = {
    "total"        : 0,
    "pass_filter"  : 0,
    "fail_c1_struct": 0,
    "fail_c2_consol": 0,
    "fail_c3_h4"   : 0,
    "fail_c4_jma"  : 0,
    "fail_c5_macd" : 0,
    "fail_c6_rsi"  : 0,
    "pass_all"     : 0,
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
    if n < CFG["sma150_period"] + CFG["consolidation_bars"] + 20: return None

    _DBG["pass_filter"] += 1

    # ── Compute indicators ────────────────────────────────────
    jma_s    = calc_jma(df["Close"], CFG["jma_period"], CFG["jma_phase"])
    sma50_s  = df["Close"].rolling(CFG["sma50_period"]).mean()
    sma150_s = df["Close"].rolling(CFG["sma150_period"]).mean()
    h4_s     = build_monthly_h4_series(df)
    rsi_s    = calc_rsi(df["Close"], CFG["rsi_period"])
    macd_l, macd_sig, macd_hist = calc_macd(
        df["Close"], CFG["macd_fast"], CFG["macd_slow"], CFG["macd_signal"])

    cur_jma   = float(jma_s.iloc[-1])   if not np.isnan(jma_s.iloc[-1])   else np.nan
    cur_s50   = float(sma50_s.iloc[-1]) if not np.isnan(sma50_s.iloc[-1]) else np.nan
    cur_s150  = float(sma150_s.iloc[-1])if not np.isnan(sma150_s.iloc[-1])else np.nan
    cur_h4    = float(h4_s.iloc[-1])    if not np.isnan(h4_s.iloc[-1])    else np.nan
    cur_rsi   = float(rsi_s.iloc[-1])   if not np.isnan(rsi_s.iloc[-1])   else 50
    cur_macdl = float(macd_l.iloc[-1])  if not np.isnan(macd_l.iloc[-1])  else 0

    if any(np.isnan([cur_jma, cur_s50, cur_s150])): return None
    if np.isnan(cur_h4): return None

    # ─────────────────────────────────────────────────────────
    # C1: BEARISH STRUCTURE
    # SMA50 < SMA150  AND  both below monthly Camarilla H4
    # We check this at the START of the consolidation window
    # (not today — because today price may have broken out)
    # ─────────────────────────────────────────────────────────
    cb = CFG["consolidation_bars"]
    ref_idx = max(0, n - cb - 1)   # start of consolidation window

    s50_ref  = float(sma50_s.iloc[ref_idx])  if not np.isnan(sma50_s.iloc[ref_idx])  else np.nan
    s150_ref = float(sma150_s.iloc[ref_idx]) if not np.isnan(sma150_s.iloc[ref_idx]) else np.nan
    h4_ref   = float(h4_s.iloc[ref_idx])     if not np.isnan(h4_s.iloc[ref_idx])     else np.nan

    if np.isnan(s50_ref) or np.isnan(s150_ref) or np.isnan(h4_ref):
        _DBG["fail_c1_struct"] += 1; return None

    # SMA50 must be below SMA150 (bearish structure)
    if CFG["require_sma50_below_sma150"] and s50_ref >= s150_ref:
        _DBG["fail_c1_struct"] += 1; return None

    # Both MAs below monthly H4 at start of consolidation
    if CFG["require_mas_below_h4"]:
        if s50_ref >= h4_ref or s150_ref >= h4_ref:
            _DBG["fail_c1_struct"] += 1; return None

    # ─────────────────────────────────────────────────────────
    # C2: 5-MONTH CONSOLIDATION
    # Price range during the consolidation window must be tight
    # ─────────────────────────────────────────────────────────
    consol_window = df.iloc[ref_idx: n - 1]   # exclude today's breakout bar
    if len(consol_window) < 20:
        _DBG["fail_c2_consol"] += 1; return None

    consol_hi  = float(consol_window["High"].max())
    consol_lo  = float(consol_window["Low"].min())
    if consol_lo <= 0:
        _DBG["fail_c2_consol"] += 1; return None

    consol_range_pct = (consol_hi - consol_lo) / consol_lo * 100
    if consol_range_pct > CFG["max_consolidation_range_pct"]:
        _DBG["fail_c2_consol"] += 1; return None

    # ─────────────────────────────────────────────────────────
    # C3: PRICE CLOSED ABOVE MONTHLY CAMARILLA H4
    # Look back cross_lookback bars for the exact close above H4
    # ─────────────────────────────────────────────────────────
    cl = CFG["cross_lookback"]
    h4_break_bar  = None
    h4_break_date = None

    for i in range(max(1, n - cl), n):
        pc   = float(df["Close"].iloc[i-1])
        cc   = float(df["Close"].iloc[i])
        ph4  = float(h4_s.iloc[i-1]) if not np.isnan(h4_s.iloc[i-1]) else np.nan
        ch4  = float(h4_s.iloc[i])   if not np.isnan(h4_s.iloc[i])   else np.nan
        if np.isnan(ph4) or np.isnan(ch4): continue
        if pc < ph4 and cc >= ch4:
            h4_break_bar  = i
            h4_break_date = df.index[i]

    if h4_break_bar is None:
        _DBG["fail_c3_h4"] += 1; return None

    # ─────────────────────────────────────────────────────────
    # C4: PRICE ABOVE JMA
    # ─────────────────────────────────────────────────────────
    if price <= cur_jma:
        _DBG["fail_c4_jma"] += 1; return None

    # ─────────────────────────────────────────────────────────
    # C5: MACD LINE ABOVE ZERO  (the line, not histogram)
    # ─────────────────────────────────────────────────────────
    if cur_macdl <= 0:
        _DBG["fail_c5_macd"] += 1; return None

    # ─────────────────────────────────────────────────────────
    # C6: RSI ABOVE 50
    # ─────────────────────────────────────────────────────────
    if cur_rsi < CFG["rsi_min"]:
        _DBG["fail_c6_rsi"] += 1; return None

    # Volume on H4 break bar
    break_vol    = float(df["Volume"].iloc[h4_break_bar])
    break_vol_mx = break_vol / avg_vol if avg_vol > 0 else 0

    _DBG["pass_all"] += 1

    # ── Metrics ───────────────────────────────────────────────
    bars_since_break  = n - 1 - h4_break_bar
    dist_h4_pct       = (price - cur_h4) / cur_h4 * 100 if cur_h4 > 0 else 0
    dist_jma_pct      = (price - cur_jma)  / cur_jma  * 100 if cur_jma > 0 else 0
    sma50_vs_sma150   = (cur_s50 - cur_s150) / cur_s150 * 100 if cur_s150 > 0 else 0
    ma_vs_h4_ref      = (s50_ref - h4_ref) / h4_ref * 100 if h4_ref > 0 else 0
    cur_macdh = float(macd_hist.iloc[-1]) if not np.isnan(macd_hist.iloc[-1]) else 0

    # ── Score (0-100) ─────────────────────────────────────────
    score = 0
    # H4 break freshness (0-25)
    score += max(0, 25 - bars_since_break * 5)
    # Consolidation tightness (0-20): tighter = more compressed
    score += max(0, 20 - int(consol_range_pct * 0.5))
    # Volume on break (0-20)
    score += min(20, int(break_vol_mx * 8))
    # RSI strength above 50 (0-15)
    score += min(15, max(0, int((cur_rsi - 50) * 1.5)))
    # MACD line magnitude (0-10)
    score += min(10, max(0, int(cur_macdl * 5)))
    # JMA proximity (0-5): just above JMA = fresher
    score += max(0, 5 - int(abs(dist_jma_pct)))
    # Depth of bearish structure vs H4 (0-5): deeper = bigger breakout
    score += min(5, max(0, int(abs(ma_vs_h4_ref) * 0.5)))
    score = min(100, max(0, score))

    return {
        "Ticker"             : sym,
        "Price"              : round(price, 2),
        "Score"              : score,
        # C1: bearish structure at start of consolidation
        "SMA50_at_Start"     : round(s50_ref, 2),
        "SMA150_at_Start"    : round(s150_ref, 2),
        "H4_at_Start"        : round(h4_ref, 2),
        "MA_vs_H4_Ref_%"     : round(ma_vs_h4_ref, 2),
        # C2: consolidation
        "Consol_Bars"        : cb,
        "Consol_Hi"          : round(consol_hi, 2),
        "Consol_Lo"          : round(consol_lo, 2),
        "Consol_Range_%"     : round(consol_range_pct, 2),
        # C3: H4 break
        "H4_Level"           : round(cur_h4, 2),
        "H4_Break_Date"      : h4_break_date.strftime("%Y-%m-%d"),
        "Bars_Since_Break"   : bars_since_break,
        "Break_Vol_x"        : round(break_vol_mx, 2),
        "Dist_H4_%"          : round(dist_h4_pct, 2),
        # C4: JMA
        "JMA"                : round(cur_jma, 2),
        "Dist_JMA_%"         : round(dist_jma_pct, 2),
        # C5: MACD
        "MACD_Line"          : round(cur_macdl, 4),
        "MACD_Signal"        : round(float(macd_sig.iloc[-1]), 4),
        "MACD_Hist"          : round(cur_macdh, 4),
        # C6: RSI
        "RSI"                : round(cur_rsi, 1),
        # Current MAs
        "SMA50"              : round(cur_s50, 2),
        "SMA150"             : round(cur_s150, 2),
        "SMA50_vs_SMA150_%"  : round(sma50_vs_sma150, 2),
        "Avg_Vol_20d"        : int(avg_vol),
        # internals
        "_df"      : df,
        "_jma"     : jma_s,
        "_sma50"   : sma50_s,
        "_sma150"  : sma150_s,
        "_h4"      : h4_s,
        "_macd_l"  : macd_l,
        "_rsi"     : rsi_s,
        "_break_bar": h4_break_bar,
        "_ref_idx" : ref_idx,
    }

# ── Live print ────────────────────────────────────────────────
LIVE_COLS = ["Ticker","Price","Score",
             "Consol_Range_%","H4_Level","H4_Break_Date",
             "Bars_Since_Break","Break_Vol_x",
             "JMA","MACD_Line","RSI"]
_CW = {"Ticker":8,"Price":10,"Score":7,
       "Consol_Range_%":15,"H4_Level":11,"H4_Break_Date":15,
       "Bars_Since_Break":17,"Break_Vol_x":13,
       "JMA":9,"MACD_Line":12,"RSI":7}
_CF = {"Price":"${:.2f}","Score":"{:.0f}",
       "Consol_Range_%":"{:.1f}%","H4_Level":"${:.2f}",
       "Bars_Since_Break":"{:.0f}d","Break_Vol_x":"{:.2f}×",
       "JMA":"${:.2f}","MACD_Line":"{:.4f}","RSI":"{:.1f}"}
_hdr_done = False

def _live_header():
    global _hdr_done
    if _hdr_done: return
    sep = "━"*185
    print(f"\n{sep}")
    print("  📊  LIVE MATCHES  —  Base Breakout: Cam H4 + JMA + MACD Zero + RSI>50")
    print(sep)
    print("".join(f"  {c:<{_CW.get(c,10)}}" for c in LIVE_COLS))
    print("  "+"─"*183)
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
chk = download(["AAPL","NVDA","AMD"], 550)
if not chk: print("❌  No data.")
else:
    for s, d in chk.items():
        p    = float(d["Close"].iloc[-1])
        h4l  = get_monthly_h4(d)
        h4v  = h4l[0] if h4l[0] else "—"
        print(f"  ✅ {s}: {len(d)} bars  ${p:.2f}  H4=${h4v}  {d.index[-1].date()}")
print()

# ── Diagnostic ────────────────────────────────────────────────
print("━"*65)
print("  STEP 2  DIAGNOSTIC (10 stocks)")
print("━"*65+"\n")
DIAG = ["AAPL","NVDA","AMD","PLTR","MU","SMCI","INTC","PYPL","ROKU","SNAP"]
diag_data = download(DIAG, CFG["history_days"])
print(f"  Downloaded {len(diag_data)}/{len(DIAG)}\n")
print(f"  {'SYM':<7} {'PRICE':>8}  {'S50<S150':>9}  {'RANGE%':>7}  RESULT")
print("  "+"─"*45)
for sym in DIAG:
    if sym not in diag_data:
        print(f"  {sym:<7} — no data"); continue
    try:
        df_d = diag_data[sym]
        p    = float(df_d["Close"].iloc[-1])
        s50  = float(df_d["Close"].rolling(50).mean().iloc[-1])
        s150 = float(df_d["Close"].rolling(150).mean().iloc[-1])
        t    = lambda b: "✅" if b else "❌"
        r    = detect_pattern(sym, df_d)
        if r:
            print(f"  {sym:<7} ${p:>7.2f}  {t(s50<s150):>9}  "
                  f"{r['Consol_Range_%']:>6.1f}%  ✅ Score={r['Score']}")
        else:
            print(f"  {sym:<7} ${p:>7.2f}  {t(s50<s150):>9}  "
                  f"{'—':>7}  ❌")
    except Exception as e:
        print(f"  {sym:<7} error: {e}")

print(f"""
  Pattern:
    C1  Bearish structure    : SMA50 < SMA150  AND  both below Cam H4
                               (measured at start of consolidation window)
    C2  Consolidation        : price range <= {CFG['max_consolidation_range_pct']}% over {CFG['consolidation_bars']} bars (~5 months)
    C3  H4 break             : price closed above monthly Camarilla H4
                               within last {CFG['cross_lookback']} bars
    C4  Above JMA            : price > JMA({CFG['jma_period']}, phase={CFG['jma_phase']})
    C5  MACD line > 0        : EMA12 - EMA26 > 0 (not just histogram)
    C6  RSI > {CFG['rsi_min']}           : momentum above neutral

  Tune if mostly ❌:
    max_consolidation_range_pct  35 → 50
    consolidation_bars          105 → 84   (~4 months)
    cross_lookback               10 → 15
    rsi_min                      50 → 45
    require_mas_below_h4       True → False
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
        "AXON","ANET","CAVA","VRT","ELF","GRMN","ON","ENPH","ROST","POOL",
        "AMGN","GILD","INTU","MCHP","MNST","NXPI","XEL","ACLS","IRTC","MXL",
        "PYPL","ROKU","SNAP","PINS","INTC","EBAY","ETSY","PTON","ZM","LYFT",
        "RIVN","LCID","NKLA","QS","PLUG","FCEL","BLNK","CHPT","WKHS","FSR",
        "SWKS","QRVO","MCHP","MTCH","BMBL","SMAR","BILL","DOCU","APPN","ALRM",
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
  ❌ Failed C1 (structure) : {_DBG['fail_c1_struct']}
  ❌ Failed C2 (consol)    : {_DBG['fail_c2_consol']}
  ❌ Failed C3 (H4 break)  : {_DBG['fail_c3_h4']}
  ❌ Failed C4 (JMA)       : {_DBG['fail_c4_jma']}
  ❌ Failed C5 (MACD)      : {_DBG['fail_c5_macd']}
  ❌ Failed C6 (RSI)       : {_DBG['fail_c6_rsi']}
  ✅ Passed all            : {_DBG['pass_all']}
""")

if not results:
    print("  No matches. Relax the condition with most failures above:")
    print("   C2: max_consolidation_range_pct  35 → 50")
    print("   C2: consolidation_bars          105 → 84")
    print("   C3: cross_lookback               10 → 15")
    print("   C1: require_mas_below_h4       True → False")
    print("   C6: rsi_min                      50 → 45")

results.sort(key=lambda x: -x["Score"])

# ── Build df_out ──────────────────────────────────────────────
ts      = datetime.today().strftime("%Y%m%d_%H%M")
out_dir = os.environ.get("GITHUB_WORKSPACE", os.getcwd())

COLS = [
    "Ticker","Price","Score",
    "SMA50_at_Start","SMA150_at_Start","H4_at_Start","MA_vs_H4_Ref_%",
    "Consol_Bars","Consol_Hi","Consol_Lo","Consol_Range_%",
    "H4_Level","H4_Break_Date","Bars_Since_Break","Break_Vol_x","Dist_H4_%",
    "JMA","Dist_JMA_%","MACD_Line","MACD_Signal","MACD_Hist",
    "RSI","SMA50","SMA150","SMA50_vs_SMA150_%","Avg_Vol_20d",
]
df_out = pd.DataFrame([{k:v for k,v in r.items() if not k.startswith("_")}
                        for r in results]) if results else pd.DataFrame(columns=COLS)
if not df_out.empty:
    df_out = df_out[[c for c in COLS if c in df_out.columns]]
    df_out.reset_index(drop=True, inplace=True)

FMT = {
    "Price"              : lambda v: f"${v:.2f}",
    "Score"              : lambda v: f"{v:.0f}",
    "SMA50_at_Start"     : lambda v: f"${v:.2f}",
    "SMA150_at_Start"    : lambda v: f"${v:.2f}",
    "H4_at_Start"        : lambda v: f"${v:.2f}",
    "MA_vs_H4_Ref_%"     : lambda v: f"{v:+.2f}%",
    "Consol_Hi"          : lambda v: f"${v:.2f}",
    "Consol_Lo"          : lambda v: f"${v:.2f}",
    "Consol_Range_%"     : lambda v: f"{v:.1f}%",
    "H4_Level"           : lambda v: f"${v:.2f}",
    "Bars_Since_Break"   : lambda v: f"{int(v)}d",
    "Break_Vol_x"        : lambda v: f"{v:.2f}×",
    "Dist_H4_%"          : lambda v: f"{v:+.2f}%",
    "JMA"                : lambda v: f"${v:.2f}",
    "Dist_JMA_%"         : lambda v: f"{v:+.2f}%",
    "MACD_Line"          : lambda v: f"{v:.4f}",
    "MACD_Signal"        : lambda v: f"{v:.4f}",
    "MACD_Hist"          : lambda v: f"{v:.4f}",
    "RSI"                : lambda v: f"{v:.1f}",
    "SMA50"              : lambda v: f"${v:.2f}",
    "SMA150"             : lambda v: f"${v:.2f}",
    "SMA50_vs_SMA150_%"  : lambda v: f"{v:+.2f}%",
    "Avg_Vol_20d"        : lambda v: f"{v:,.0f}",
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
            "Consol_Range_%","H4_Level","H4_Break_Date",
            "Bars_Since_Break","Break_Vol_x","Dist_H4_%",
            "JMA","Dist_JMA_%","MACD_Line","RSI"]
    DISP = [c for c in DISP if c in df_out.columns]
    gc   = "#22c55e"
    th   = "".join(
        f'<th style="background:#0f172a;color:#e2e8f0;padding:9px 12px;'
        f'font-size:11px;font-weight:700;border-bottom:2px solid {gc};'
        f'white-space:nowrap">{c}</th>' for c in DISP)
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
            elif col == "RSI":
                try:
                    v = float(str(raw).replace("%",""))
                    sty = "color:#22c55e;font-weight:700" if v>=60 else "color:#86efac"
                except Exception: pass
            elif col == "MACD_Line":
                try:
                    v = float(raw)
                    sty = "color:#22c55e;font-weight:700" if v>0 else "color:#ef4444"
                except Exception: pass
            elif col == "Bars_Since_Break":
                try:
                    v = int(str(raw).replace("d",""))
                    sty = "color:#22c55e;font-weight:700;text-align:center" if v==0 else "text-align:center"
                except Exception: pass
            tds += f'<td style="padding:7px 12px;font-size:12px;border-bottom:1px solid #e2e8f0;white-space:nowrap;{sty}">{disp}</td>'
        rows_html += f'<tr style="background:{bg}">{tds}</tr>\n'

    ticker_csv_str = ",".join(r["Ticker"] for r in results)
    header = f"""
<div style="background:linear-gradient(135deg,#0f172a,#1e3a5f);border-radius:10px;
            padding:18px 24px;margin-bottom:8px">
  <h2 style="margin:0;color:#60a5fa;font-size:20px;font-weight:700">
    📈 Base Breakout: Cam H4 + JMA + MACD Zero + RSI&gt;50
  </h2>
  <p style="margin:6px 0 0;color:#94a3b8;font-size:12px">
    {datetime.today().strftime('%Y-%m-%d %H:%M')} &nbsp;·&nbsp;
    <b style="color:{gc}">{len(results)} matches</b> from {len(TICKERS)} tickers
  </p>
</div>
<div style="background:#0f172a;border-radius:8px;padding:14px 16px;
            margin:8px 0;border-left:4px solid {gc};">
  <p style="margin:0 0 4px;color:#94a3b8;font-size:11px;font-weight:600;
             text-transform:uppercase;letter-spacing:.05em">
    📋 Stock List (CSV) — copy &amp; paste
  </p>
  <p style="margin:0;color:{gc};font-size:13px;font-weight:700;
             font-family:'Courier New',monospace;word-break:break-all">
    {ticker_csv_str}
  </p>
</div>"""
    table = f"""
<div style="overflow-x:auto;border:1px solid #e2e8f0;border-radius:8px">
  <table style="border-collapse:collapse;width:100%;min-width:700px">
    <thead><tr>{th}</tr></thead><tbody>{rows_html}</tbody>
  </table>
</div>"""
    legend = """
<div style="background:#f8fafc;border:1px solid #e2e8f0;border-radius:8px;
            padding:12px 18px;margin-top:6px;font-size:11px;color:#64748b">
  <b style="color:#475569">GUIDE</b> &nbsp;·&nbsp;
  Consol_Range_% = price range during 5-month consolidation (lower = tighter coil) &nbsp;·&nbsp;
  H4_Level = monthly Camarilla H4 that was broken &nbsp;·&nbsp;
  Bars_Since_Break 0 = H4 broken today &nbsp;·&nbsp;
  MACD_Line must be > 0 (the actual MACD line, not histogram)
</div>"""
    display_html(header + table + legend)

elif results:
    CLI = ["Ticker","Price","Score","Consol_Range_%","H4_Level",
           "Bars_Since_Break","Break_Vol_x","MACD_Line","RSI"]
    CLI = [c for c in CLI if c in df_out.columns]
    col_w = {c: max(len(c), max(len(fmt_v(c,r.get(c))) for r in results))+2 for c in CLI}
    top = "┬".join("─"*col_w[c] for c in CLI)
    sep = "┼".join("─"*col_w[c] for c in CLI)
    bot = "┴".join("─"*col_w[c] for c in CLI)
    hdr = "│".join(c.center(col_w[c]) for c in CLI)
    inner = sum(col_w.values())+len(CLI)-1
    print(f"\n  ╔{'═'*inner}╗")
    tit = f"  Base Breakout Cam H4 + JMA + MACD + RSI   {datetime.today().strftime('%Y-%m-%d')}   {len(df_out)} matches"
    print(f"  ║{tit.center(inner)}║\n  ╚{'═'*inner}╝\n")
    print(f"  ┌{top}┐\n  │{hdr}│\n  ├{sep}┤")
    for i,r in enumerate(results):
        cells=[fmt_v(c,r.get(c)).center(col_w[c]) for c in CLI]
        print(f"  │{'│'.join(cells)}│")
        if i<len(results)-1: print(f"  ├{sep}┤")
    print(f"  └{bot}┘")

# ── Save CSV + TradingView ─────────────────────────────────────
fpath = os.path.join(out_dir, f"base_breakout_h4_jma_{ts}.csv")
df_out.to_csv(fpath, index=False)
print(f"\n  💾 CSV → {fpath}")
tv = os.path.join(out_dir, f"tv_base_breakout_h4_{ts}.txt")
with open(tv,"w") as f:
    f.write(f"###Base Breakout H4 JMA MACD RSI {datetime.today().strftime('%Y-%m-%d')}\n")
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
            for c in ["Ticker","Price","Score","Consol_Range_%","H4_Level",
                      "Bars_Since_Break","Break_Vol_x","MACD_Line","RSI"]
        )
        rows_e = ""
        for i, r in enumerate(rl[:50]):
            bg = "#fff" if i%2==0 else "#f0f9ff"
            rows_e += (
                f'<tr style="background:{bg}">'
                f'<td style="padding:6px 11px;font-size:12px;font-weight:700;color:#22c55e">'
                f'{r.get("Ticker","—")}</td>'
                f'<td style="padding:6px 11px;font-size:12px">'
                f'${float(r.get("Price",0)):.2f}</td>'
                f'<td style="padding:6px 11px;font-size:12px;font-weight:700;'
                f'background:#166534;color:#fff;text-align:center">'
                f'{float(r.get("Score",0)):.0f}</td>'
                f'<td style="padding:6px 11px;font-size:12px">'
                f'{float(r.get("Consol_Range_%",0)):.1f}%</td>'
                f'<td style="padding:6px 11px;font-size:12px">'
                f'${float(r.get("H4_Level",0)):.2f}</td>'
                f'<td style="padding:6px 11px;font-size:12px;text-align:center;color:'
                f'{"#22c55e" if r.get("Bars_Since_Break",99)==0 else "#94a3b8"};font-weight:700">'
                f'{r.get("Bars_Since_Break",0)}d</td>'
                f'<td style="padding:6px 11px;font-size:12px">'
                f'{float(r.get("Break_Vol_x",0)):.2f}×</td>'
                f'<td style="padding:6px 11px;font-size:12px;color:#22c55e;font-weight:700">'
                f'{float(r.get("MACD_Line",0)):.4f}</td>'
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
      📊 Base Breakout: Cam H4 + JMA + MACD Zero + RSI&gt;50
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
      SMA50 &lt; SMA150 &amp; both below H4 at consolidation start &nbsp;·&nbsp;
      MACD_Line = EMA12−EMA26 (must be &gt;0)
    </p>
  </td></tr>
  <tr><td style="background:#f8fafc;padding:12px 28px;border-top:1px solid #e2e8f0;text-align:center">
    <p style="margin:0;color:#94a3b8;font-size:10px">
      ⚠️ Not financial advice &nbsp;·&nbsp; Auto-generated by GitHub Actions
    </p>
  </td></tr>
</table></td></tr></table></body></html>"""

        plain_e = "\n".join([
            f"Base Breakout: Cam H4 + JMA + MACD + RSI>50 — {datetime.today().strftime('%Y-%m-%d')}",
            f"{cnt} matches",
            "",
            f"STOCKS: {ticker_csv}",
            "",
            "="*60,
        ] + ([
            f"{r.get('Ticker','—'):<7} ${float(r.get('Price',0)):.2f}  "
            f"Score:{float(r.get('Score',0)):.0f}  "
            f"Range:{float(r.get('Consol_Range_%',0)):.1f}%  "
            f"H4Break:{r.get('Bars_Since_Break',0)}d ago  "
            f"RSI:{float(r.get('RSI',0)):.1f}"
            for r in rl[:50]
        ] if rl else ["No matches today"]) + ["\n📎 CSV and TradingView file attached."])

        subj = (f"📊 Base Breakout H4+JMA+MACD — {cnt} signal{'s' if cnt!=1 else ''}"
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
    fig, axes = plt.subplots(len(top)*2, 1,
                              figsize=(15, 7*len(top)),
                              facecolor="#0f172a",
                              gridspec_kw={"height_ratios":[3,1]*len(top)})
    if len(top)==1: axes = list(axes)

    for idx, r in enumerate(top):
        ax_p  = axes[idx*2]
        ax_m  = axes[idx*2+1]
        df_p  = r["_df"].tail(180).copy()
        n_p   = len(df_p)
        fn    = len(r["_df"]); off = fn - n_p

        jma   = r["_jma"].reindex(df_p.index)
        s50   = r["_sma50"].reindex(df_p.index)
        s150  = r["_sma150"].reindex(df_p.index)
        h4    = r["_h4"].reindex(df_p.index)
        macd_line = r["_macd_l"].reindex(df_p.index)
        rsi_l = r["_rsi"].reindex(df_p.index)

        ax_p.set_facecolor("#0f172a")
        ax_m.set_facecolor("#0f172a")

        for i,(_, row_) in enumerate(df_p.iterrows()):
            o=float(row_["Open"]); hh=float(row_["High"])
            l=float(row_["Low"]);  c=float(row_["Close"])
            clr="#34d399" if c>=o else "#ef4444"
            ax_p.plot([i,i],[l,hh],color=clr,lw=0.6,zorder=2)
            blo=min(o,c); bhi=max(o,c); bh=max(bhi-blo,(hh-l)*0.005)
            rect=mpatches.FancyBboxPatch((i-0.3,blo),0.6,bh,
                 boxstyle="square,pad=0",facecolor=clr,edgecolor=clr,lw=0.3,zorder=3)
            ax_p.add_patch(rect)

        ax_p.plot(range(n_p), jma.values,  color="#22d3ee", lw=1.8, label="JMA", zorder=6)
        ax_p.plot(range(n_p), s50.values,  color="#3b82f6", lw=1.4, ls="--", label="SMA50", zorder=5)
        ax_p.plot(range(n_p), s150.values, color="#f472b6", lw=1.4, ls="-.", label="SMA150", zorder=5)
        ax_p.plot(range(n_p), h4.values,   color="#f59e0b", lw=1.6, ls=":", label="Cam H4", zorder=5)
        ax_p.axhline(r["H4_Level"], color="#f59e0b", lw=0.8, alpha=0.4)

        # Shade consolidation window
        ref_plot = r["_ref_idx"] - off
        if ref_plot >= 0:
            ax_p.axvspan(max(0, ref_plot), n_p-2, alpha=0.04,
                         color="#22c55e", zorder=1)

        # Mark H4 break bar
        bb = r["_break_bar"] - off
        if 0 <= bb < n_p:
            ax_p.axvline(bb, color="#22c55e", lw=2.0, ls="--", alpha=0.9)
            ax_p.scatter([bb],[float(df_p["Close"].iloc[bb])],
                         color="#22c55e", s=200, zorder=9, marker="^",
                         label=f"H4 Break {r['H4_Break_Date']}")

        # MACD line subplot
        ax_m.plot(range(n_p), macd_line.values, color="#3b82f6", lw=1.4, label="MACD Line")
        ax_m.axhline(0, color="#94a3b8", lw=1.0, ls="--")
        ax_m.fill_between(range(n_p), macd_line.values, 0,
                          where=(macd_line.values > 0), alpha=0.3, color="#22c55e")
        ax_m.fill_between(range(n_p), macd_line.values, 0,
                          where=(macd_line.values <= 0), alpha=0.3, color="#ef4444")

        tick_step = max(1, n_p//8)
        for ax in [ax_p, ax_m]:
            ax.set_xticks(range(0,n_p,tick_step))
            ax.set_xticklabels(
                [df_p.index[i].strftime("%b%y") for i in range(0,n_p,tick_step)],
                color="#94a3b8", fontsize=7)
            ax.set_xlim(-0.5, n_p-0.5)
            ax.tick_params(colors="#94a3b8", labelsize=7)
            for sp_ in ax.spines.values(): sp_.set_edgecolor("#1e3a5f")
            ax.grid(color="#1e3a5f", ls="--", lw=0.4, alpha=0.4, axis="y")

        ax_p.set_title(
            f"{r['Ticker']}  ${r['Price']:.2f}  |  Score {r['Score']}/100  |  "
            f"Consol {r['Consol_Range_%']:.1f}% range ({r['Consol_Bars']}bars)  |  "
            f"H4=${r['H4_Level']:.2f} broken {r['H4_Break_Date']} ({r['Bars_Since_Break']}d ago) "
            f"Vol{r['Break_Vol_x']:.1f}×  |  MACD={r['MACD_Line']:.4f}  RSI={r['RSI']:.1f}",
            color="#e2e8f0", fontsize=8, fontweight="bold", pad=6)
        ax_p.legend(loc="upper left", facecolor="#1e293b",
                    labelcolor="#e2e8f0", fontsize=7, framealpha=0.9, ncol=2)
        ax_m.set_ylabel("MACD Line", color="#94a3b8", fontsize=7)
        ax_m.legend(loc="upper left", facecolor="#1e293b",
                    labelcolor="#e2e8f0", fontsize=7, framealpha=0.9)

    plt.suptitle(
        f"Base Breakout: SMA50<SMA150 Below Cam H4 → Breakout + JMA + MACD>0 + RSI>50  ·  "
        f"{datetime.today().strftime('%Y-%m-%d')}\n"
        f"🔵 JMA  🔵 SMA50  🩷 SMA150  🟠 Cam H4  ▲ H4 Break  🟢 shaded = consolidation zone",
        color="#60a5fa", fontsize=9, fontweight="bold", y=1.001)
    plt.tight_layout()
    cp = os.path.join(out_dir, f"base_breakout_chart_{ts}.png")
    plt.savefig(cp, dpi=150, bbox_inches="tight", facecolor="#0f172a")
    if _IN_NOTEBOOK: plt.show()
    else: plt.close()
    print(f"  📊 Chart → {cp}")

print("""
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  📋 PATTERN EXPLAINED

  C1  BEARISH STRUCTURE  (at start of consolidation window)
      SMA50 < SMA150 — medium-term trend below long-term
      Both SMA50 AND SMA150 are below Camarilla H4
      = Stock has been in extended weakness/downtrend,
        with H4 acting as overhead resistance

  C2  5-MONTH CONSOLIDATION  (~105 trading bars)
      Price range during this period <= 35%
      = Volatility contraction, potential accumulation
        (sellers exhausted, range-bound base forming)

  C3  PRICE CLOSED ABOVE MONTHLY CAMARILLA H4
      H4 = Close + (High-Low) × 1.1/2  from prior month
      Exact 1-bar cross within last 10 bars
      = Key monthly pivot resistance decisively broken

  C4  ABOVE JMA (13, phase 40)
      Price above the Jurik Moving Average
      = Short-term trend confirmed bullish

  C5  MACD LINE ABOVE ZERO
      EMA12 - EMA26 > 0  (the MACD line itself, not histogram)
      = Medium-term momentum has turned positive

  C6  RSI > 50
      RSI(14) above the neutral 50 level
      = Buyers are now in control of momentum

  💡 BEST SETUPS
  Bars_Since_Break = 0     H4 broken today = earliest entry
  Consol_Range_% < 20%     very tight base = more compressed
  Break_Vol_x >= 2×         strong volume conviction
  MACD_Line > 0 strongly   momentum well above zero
  RSI 55-65                healthy range, not yet overbought

  ⚙️  TUNE IF 0 RESULTS
  max_consolidation_range_pct  35 → 50
  consolidation_bars          105 → 84   (~4 months)
  cross_lookback               10 → 15
  rsi_min                      50 → 45
  require_mas_below_h4       True → False
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
""")
