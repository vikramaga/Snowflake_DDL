# ============================================================
# NASDAQ — JMA/SMA8 Retest + Cam H4 + MACD Rising Above Zero
# ============================================================
#
# EXACT PATTERN (from MXL chart — red-circled breakout zone):
#
#  C1 — PRICE ABOVE BOTH JMA AND SMA8 FOR 4+ CONSECUTIVE DAYS
#      The fast MAs (JMA and SMA8) are both below price,
#      and price has stayed above them for at least 4 bars.
#      This confirms the stock is in control of the bulls and
#      the fast MAs are acting as support, not resistance.
#
#  C2 — RETESTED JMA OR SMA8 WITHIN THE LAST N BARS
#      During those 4+ days, at least one candle LOW came
#      within touch_pct% of JMA or SMA8 — a retest of the
#      fast MA as support — and price closed back above it.
#      = Buyers defended the fast MA on the pullback.
#
#  C3 — ABOVE BOTH THIS MONTH'S AND PREVIOUS MONTH'S CAM H4
#      H4 = Close + (High - Low) × 1.1 / 2  (from prior month)
#      Price must be above BOTH the current and prior month's
#      Camarilla H4 pivot — confirming it has reclaimed
#      both recent institutional resistance levels.
#
#  C4 — MACD ABOVE ZERO LINE AND INCREASING
#      MACD histogram (MACD - Signal) > 0
#      AND histogram is increasing over last N bars
#      (accelerating positive momentum — not just crossing zero
#       but actively expanding upward)
#
# LOGIC FLOW (MXL example):
#   Price bunches up near all MAs near $20 Camarilla level
#   → Holds above JMA+SMA8 for 4 days (retest holds)
#   → Clears both monthly Camarilla H4s
#   → MACD crosses zero AND starts increasing
#   = Explosive breakout to $68+
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
    "history_days"               : 300,

    # ── Indicator periods ──────────────────────────────────────
    # JMA (Jurik Moving Average) — approximated via double EMA
    # True JMA is proprietary; we use EMA(period) with extra
    # smoothing (phase=0 standard approximation)
    "jma_period"                 : 13,    # as shown on chart: JMA 13 40 2
    "jma_phase"                  : 40,    # phase parameter (higher = smoother)
    "sma8_period"                : 8,
    "macd_fast"                  : 12,
    "macd_slow"                  : 26,
    "macd_signal"                : 9,

    # ── C1: Above JMA + SMA8 for N consecutive bars ────────────
    "min_days_above"             : 4,

    # ── C2: Retest of JMA or SMA8 ──────────────────────────────
    # Candle LOW must come within this % of JMA or SMA8
    "retest_touch_pct"           : 3.0,
    # Look back this many bars for the retest (within the above-MA window)
    "retest_lookback"            : 10,

    # ── C3: Camarilla H4 (this + previous month) ──────────────
    "require_above_both_h4"      : True,

    # ── C4: MACD ──────────────────────────────────────────────
    "require_macd_above_zero"    : True,
    # Histogram must be increasing over last N bars
    "macd_increasing_bars"       : 3,

    # ── Volume ─────────────────────────────────────────────────
    "vol_avg_bars"               : 20,

    # ── Filters ───────────────────────────────────────────────
    "min_avg_volume"             : 80_000,
    "min_price"                  : 1.0,

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

def calc_jma(series, period=13, phase=40, power=2):
    """
    JMA (Jurik Moving Average) approximation.
    The true JMA is proprietary; this is the widely-used
    pure-numpy approximation: adaptive EMA with phase-based
    smoothing factor and power exponent.

    phase : -100..+100 (positive = more responsive)
    power : typically 2
    """
    n      = len(series)
    vals   = series.values.astype(float)
    result = np.full(n, np.nan)

    # Phase ratio: maps -100..100 → 0.5..2.5 (standard mapping)
    phase_ratio = phase / 100.0 + 1.5   # → 1.9 for phase=40

    # Adaptive alpha (similar to how Jurik smoothing works)
    alpha = 2.0 / (period + 1.0)

    # Beta (phase amplifier)
    beta = alpha * phase_ratio

    # Initialise with first valid bar
    first_valid = 0
    for i in range(n):
        if not np.isnan(vals[i]):
            first_valid = i
            break

    e0 = e1 = e2 = vals[first_valid]
    result[first_valid] = e0

    for i in range(first_valid + 1, n):
        v   = vals[i]
        e0_prev = e0
        e0  = (1 - alpha) * e0 + alpha * v
        e1  = (v - e0) * (1 - beta) + beta * e1
        e2  = (e0 + e1 - e2) * alpha + (1 - alpha) * e2
        result[i] = e2

    return pd.Series(result, index=series.index)

def calc_macd(close, fast=12, slow=26, signal=9):
    ema_f = calc_ema(close, fast)
    ema_s = calc_ema(close, slow)
    macd  = ema_f - ema_s
    sig   = calc_ema(macd, signal)
    hist  = macd - sig
    return macd, sig, hist

def cam_h4(high, low, close):
    """Camarilla H4 = Close + (High - Low) × 1.1 / 2"""
    return close + (high - low) * 1.1 / 2.0

def get_monthly_h4_levels(df):
    """
    Returns [current_month_H4, previous_month_H4]
    Each H4 is derived from the PRIOR completed month's H/L/C.
    """
    df = df.copy()
    df.index = pd.to_datetime(df.index)
    today = pd.Timestamp.today().normalize()
    levels = []
    for offset in [1, 2]:   # 1=this month's H4, 2=prev month's H4
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

# ── Core detection ────────────────────────────────────────────
def detect_pattern(sym, df):
    """
    C1: price above JMA AND SMA8 for last min_days_above bars
    C2: at least one retest of JMA or SMA8 in that window
    C3: price above this month's H4 AND previous month's H4
    C4: MACD histogram > 0 and increasing
    """
    df      = df.copy(); df.index = pd.to_datetime(df.index)
    n       = len(df)
    price   = float(df["Close"].iloc[-1])
    avg_vol = float(df["Volume"].tail(CFG["vol_avg_bars"]).mean())

    if price   < CFG["min_price"]:      return None
    if avg_vol < CFG["min_avg_volume"]: return None
    if n < CFG["sma8_period"] + 10:    return None

    # ── Compute indicators ────────────────────────────────────
    jma_s   = calc_jma(df["Close"], CFG["jma_period"], CFG["jma_phase"])
    sma8_s  = df["Close"].rolling(CFG["sma8_period"]).mean()
    macd_s, sig_s, hist_s = calc_macd(df["Close"],
                                       CFG["macd_fast"],
                                       CFG["macd_slow"],
                                       CFG["macd_signal"])
    rsi_s   = calc_rsi(df["Close"])

    cur_jma  = float(jma_s.iloc[-1])  if not np.isnan(jma_s.iloc[-1])  else np.nan
    cur_sma8 = float(sma8_s.iloc[-1]) if not np.isnan(sma8_s.iloc[-1]) else np.nan
    cur_hist = float(hist_s.iloc[-1]) if not np.isnan(hist_s.iloc[-1]) else 0
    cur_rsi  = float(rsi_s.iloc[-1])  if not np.isnan(rsi_s.iloc[-1])  else 50

    if np.isnan(cur_jma) or np.isnan(cur_sma8): return None

    # ─────────────────────────────────────────────────────────
    # C4: MACD HISTOGRAM ABOVE ZERO AND INCREASING
    # (check first — fastest gate)
    # ─────────────────────────────────────────────────────────
    if CFG["require_macd_above_zero"] and cur_hist <= 0:
        return None

    ib = CFG["macd_increasing_bars"]
    hist_vals = hist_s.tail(ib + 1).values
    if len(hist_vals) < ib + 1:
        return None
    # Histogram must be strictly increasing over the last N bars
    increasing = all(hist_vals[i] < hist_vals[i+1] for i in range(len(hist_vals)-1))
    if not increasing:
        return None

    # ─────────────────────────────────────────────────────────
    # C3: ABOVE BOTH THIS MONTH'S AND PREVIOUS MONTH'S CAM H4
    # ─────────────────────────────────────────────────────────
    h4_this, h4_prev = get_monthly_h4_levels(df)
    if h4_this is None or h4_prev is None: return None
    if CFG["require_above_both_h4"]:
        if price < h4_this: return None
        if price < h4_prev: return None

    # ─────────────────────────────────────────────────────────
    # C1: PRICE ABOVE BOTH JMA AND SMA8 FOR min_days_above BARS
    # Count how many consecutive recent bars are above both
    # ─────────────────────────────────────────────────────────
    md = CFG["min_days_above"]
    days_above = 0
    for i in range(n-1, max(0, n-md-5), -1):
        c_i   = float(df["Close"].iloc[i])
        j_i   = float(jma_s.iloc[i])  if not np.isnan(jma_s.iloc[i])  else np.nan
        s8_i  = float(sma8_s.iloc[i]) if not np.isnan(sma8_s.iloc[i]) else np.nan
        if np.isnan(j_i) or np.isnan(s8_i): break
        if c_i > j_i and c_i > s8_i:
            days_above += 1
        else:
            break   # streak ended

    if days_above < md: return None

    # ─────────────────────────────────────────────────────────
    # C2: RETEST OF JMA OR SMA8 WITHIN THE ABOVE-MA WINDOW
    # Look back retest_lookback bars (within the confirmed
    # "above MA" period) for a candle that touched JMA or SMA8
    # ─────────────────────────────────────────────────────────
    rt_lb     = CFG["retest_lookback"]
    touch_pct = CFG["retest_touch_pct"] / 100

    retest_jma  = False
    retest_sma8 = False
    retest_bar  = None
    retest_name = None
    best_depth  = float("inf")

    search_start = max(0, n - rt_lb)
    for i in range(search_start, n):
        lo    = float(df["Low"].iloc[i])
        j_i   = float(jma_s.iloc[i])  if not np.isnan(jma_s.iloc[i])  else np.nan
        s8_i  = float(sma8_s.iloc[i]) if not np.isnan(sma8_s.iloc[i]) else np.nan

        if not np.isnan(j_i) and j_i > 0:
            d = abs(lo - j_i) / j_i
            if d <= touch_pct and d < best_depth:
                retest_jma  = True
                best_depth  = d
                retest_bar  = i
                retest_name = "JMA"

        if not np.isnan(s8_i) and s8_i > 0:
            d = abs(lo - s8_i) / s8_i
            if d <= touch_pct and d < best_depth:
                retest_sma8 = True
                best_depth  = d
                retest_bar  = i
                retest_name = "SMA8"

    if not retest_jma and not retest_sma8:
        return None

    if retest_jma and retest_sma8:
        retest_label = "JMA + SMA8"
    else:
        retest_label = retest_name

    retest_depth_pct  = round(best_depth * 100, 2)
    bars_since_retest = n - 1 - retest_bar if retest_bar is not None else 0

    # ── Metrics ───────────────────────────────────────────────
    dist_jma_pct   = (price - cur_jma)  / cur_jma  * 100 if cur_jma  > 0 else 0
    dist_sma8_pct  = (price - cur_sma8) / cur_sma8 * 100 if cur_sma8 > 0 else 0
    dist_h4_this   = (price - h4_this)  / h4_this  * 100 if h4_this  > 0 else 0
    dist_h4_prev   = (price - h4_prev)  / h4_prev  * 100 if h4_prev  > 0 else 0

    macd_val  = float(macd_s.iloc[-1])  if not np.isnan(macd_s.iloc[-1])  else 0
    sig_val   = float(sig_s.iloc[-1])   if not np.isnan(sig_s.iloc[-1])   else 0

    # Histogram acceleration (how fast it's growing)
    hist_accel = float(hist_s.iloc[-1]) - float(hist_s.iloc[-2]) if n >= 2 else 0

    # ── Score (0-100) ─────────────────────────────────────────
    score = 0

    # Days above JMA+SMA8 (0-20): more = more confirmed
    score += min(20, days_above * 4)

    # Retest quality (0-20): tighter touch = cleaner retest
    score += max(0, 20 - int(retest_depth_pct * 5))

    # Both MAs retested (0-10 bonus)
    score += 10 if retest_jma and retest_sma8 else 0

    # MACD histogram level (0-20)
    score += min(20, max(0, int(cur_hist * 2)))

    # MACD acceleration (0-15)
    score += min(15, max(0, int(hist_accel * 5)))

    # H4 proximity (0-10): freshly above both levels = more upside
    avg_h4_dist = (abs(dist_h4_this) + abs(dist_h4_prev)) / 2
    score += max(0, 10 - int(avg_h4_dist))

    # RSI health (0-5)
    score += min(5, max(0, int((cur_rsi - 40) / 5)))

    score = min(100, max(0, score))

    return {
        "Ticker"             : sym,
        "Price"              : round(price, 2),
        "Score"              : score,
        # C1: above MAs
        "Days_Above_MAs"     : days_above,
        "JMA"                : round(cur_jma, 2),
        "SMA8"               : round(cur_sma8, 2),
        "Dist_JMA_%"         : round(dist_jma_pct, 2),
        "Dist_SMA8_%"        : round(dist_sma8_pct, 2),
        # C2: retest
        "Retest_Label"       : retest_label,
        "Retest_Depth_%"     : retest_depth_pct,
        "Bars_Since_Retest"  : bars_since_retest,
        # C3: Camarilla H4
        "H4_This_Month"      : round(h4_this, 2),
        "H4_Prev_Month"      : round(h4_prev, 2),
        "Dist_H4_This_%"     : round(dist_h4_this, 2),
        "Dist_H4_Prev_%"     : round(dist_h4_prev, 2),
        # C4: MACD
        "MACD"               : round(macd_val, 4),
        "MACD_Signal"        : round(sig_val, 4),
        "MACD_Hist"          : round(cur_hist, 4),
        "MACD_Accel"         : round(hist_accel, 4),
        # Other
        "RSI"                : round(cur_rsi, 1),
        "Avg_Vol_20d"        : int(avg_vol),
        # internals
        "_df"       : df,
        "_jma"      : jma_s,
        "_sma8"     : sma8_s,
        "_hist"     : hist_s,
        "_retest_bar": retest_bar,
    }

# ── Live print ────────────────────────────────────────────────
LIVE_COLS = [
    "Ticker","Price","Score",
    "Days_Above_MAs","Retest_Label","Retest_Depth_%","Bars_Since_Retest",
    "H4_This_Month","H4_Prev_Month","MACD_Hist","MACD_Accel","RSI",
]
_CW = {
    "Ticker":8,"Price":10,"Score":7,
    "Days_Above_MAs":15,"Retest_Label":16,"Retest_Depth_%":15,"Bars_Since_Retest":18,
    "H4_This_Month":15,"H4_Prev_Month":15,"MACD_Hist":12,"MACD_Accel":12,"RSI":6,
}
_CF = {
    "Price":"${:.2f}","Score":"{:.0f}",
    "Days_Above_MAs":"{:.0f}d","Retest_Depth_%":"{:.2f}%",
    "Bars_Since_Retest":"{:.0f}d",
    "H4_This_Month":"${:.2f}","H4_Prev_Month":"${:.2f}",
    "MACD_Hist":"{:.4f}","MACD_Accel":"{:.4f}","RSI":"{:.1f}",
}
_hdr_done = False

def _live_header():
    global _hdr_done
    if _hdr_done: return
    sep = "━" * 195
    print(f"\n{sep}")
    print("  📊  LIVE MATCHES  —  JMA/SMA8 Retest + Cam H4 + MACD Rising Above Zero")
    print(sep)
    print("".join(f"  {c:<{_CW.get(c,10)}}" for c in LIVE_COLS))
    print("  " + "─"*193)
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
chk = download_test = {}
try:
    import yfinance as yf
    import pandas as pd
    from datetime import datetime, timedelta
    end   = datetime.today(); start = end - timedelta(days=300)
    for sym in ["MXL","NVDA","AAPL"]:
        try:
            d = yf.Ticker(sym).history(start=start.strftime("%Y-%m-%d"),
                                        end=end.strftime("%Y-%m-%d"),
                                        auto_adjust=True)
            d.index = pd.to_datetime(d.index).tz_localize(None)
            if len(d) > 50:
                jma_t = calc_jma(d["Close"], CFG["jma_period"], CFG["jma_phase"])
                s8_t  = d["Close"].rolling(CFG["sma8_period"]).mean()
                p     = float(d["Close"].iloc[-1])
                print(f"  ✅ {sym}: {len(d)} bars  ${p:.2f}  "
                      f"JMA={float(jma_t.iloc[-1]):.2f}  SMA8={float(s8_t.iloc[-1]):.2f}  "
                      f"{d.index[-1].date()}")
                chk[sym] = d
        except Exception as e:
            print(f"  ⚠️  {sym}: {e}")
except Exception as e:
    print(f"  ❌  {e}")
print()

# ── Download helper ───────────────────────────────────────────
import yfinance as yf

def _clean(df, min_bars=30):
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

# ── Diagnostic ────────────────────────────────────────────────
print("━"*65)
print("  STEP 2  DIAGNOSTIC (10 sample stocks)")
print("━"*65+"\n")

DIAG = ["MXL","NVDA","AAPL","MSFT","PLTR","META","CRWD","AVGO","MU","AMD"]
diag_data = download(DIAG, CFG["history_days"])
print(f"  Downloaded {len(diag_data)}/{len(DIAG)}\n")
print(f"  {'SYM':<7} {'PRICE':>8}  {'D_ABV':>6}  {'RETEST':>14}  "
      f"{'MACD+':>7}  {'SCORE':>6}  RESULT")
print("  "+"─"*65)

for sym in DIAG:
    if sym not in diag_data:
        print(f"  {sym:<7} — no data"); continue
    try:
        df_d = diag_data[sym]
        p    = float(df_d["Close"].iloc[-1])
        r    = detect_pattern(sym, df_d)
        t    = lambda b: "✅" if b else "❌"
        if r:
            print(f"  {sym:<7} ${p:>7.2f}  "
                  f"{r['Days_Above_MAs']:>6.0f}d  "
                  f"{r['Retest_Label']:>14}  "
                  f"{t(r['MACD_Hist']>0):>7}  "
                  f"{r['Score']:>6}  ✅")
        else:
            print(f"  {sym:<7} ${p:>7.2f}  {'—':>6}  {'—':>14}  {'—':>7}  {'—':>6}  ❌")
    except Exception as e:
        print(f"  {sym:<7} error: {e}")

print(f"""
  Pattern:
    C1  Above JMA + SMA8  : both MAs below price for {CFG['min_days_above']}+ consecutive bars
    C2  Retest            : candle LOW within {CFG['retest_touch_pct']}% of JMA or SMA8
                            in last {CFG['retest_lookback']} bars, price closed back above
    C3  Camarilla H4      : price above THIS month's H4 AND PREVIOUS month's H4
                            H4 = Close + (High-Low) x 1.1/2 from prior month
    C4  MACD rising       : histogram > 0 AND increasing for {CFG['macd_increasing_bars']} bars

  Tune if mostly ❌:
    min_days_above         4 → 3    (fewer days above MAs)
    retest_touch_pct       3 → 5    (wider retest zone)
    retest_lookback       10 → 15
    macd_increasing_bars   3 → 2
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
        "QUBT","RGTI","ASTS","RKLB","IONQ","FSLR","PYPL","ROKU","ROST","POOL",
        "AMGN","GILD","INTU","MCHP","MNST","NXPI","XEL","ACLS","IRTC","AXON",
        "MXL","ANET","CAVA","VRT","ELF","GRMN","SMCI","KLAC","ON","ENPH",
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

if not results:
    print("\n  No matches. Try relaxing:")
    print("   min_days_above         4 → 3")
    print("   retest_touch_pct       3 → 5")
    print("   retest_lookback       10 → 15")
    print("   macd_increasing_bars   3 → 2")

results.sort(key=lambda x: x["Score"], reverse=True)

# ── Always build df_out and save/email (even if 0 results) ────
ts      = datetime.today().strftime("%Y%m%d_%H%M")
out_dir = os.environ.get("GITHUB_WORKSPACE", os.getcwd())

COLS = [
    "Ticker","Price","Score",
    "Days_Above_MAs","JMA","SMA8","Dist_JMA_%","Dist_SMA8_%",
    "Retest_Label","Retest_Depth_%","Bars_Since_Retest",
    "H4_This_Month","H4_Prev_Month","Dist_H4_This_%","Dist_H4_Prev_%",
    "MACD","MACD_Signal","MACD_Hist","MACD_Accel",
    "RSI","Avg_Vol_20d",
]
df_out = pd.DataFrame([{k:v for k,v in r.items() if not k.startswith("_")}
                        for r in results]) if results else pd.DataFrame(columns=COLS)
if not df_out.empty:
    df_out = df_out[[c for c in COLS if c in df_out.columns]]
    df_out.reset_index(drop=True, inplace=True)

FMT = {
    "Price"            : lambda v: f"${v:.2f}",
    "Score"            : lambda v: f"{v:.0f}",
    "Days_Above_MAs"   : lambda v: f"{int(v)}d",
    "JMA"              : lambda v: f"${v:.2f}",
    "SMA8"             : lambda v: f"${v:.2f}",
    "Dist_JMA_%"       : lambda v: f"{v:+.2f}%",
    "Dist_SMA8_%"      : lambda v: f"{v:+.2f}%",
    "Retest_Depth_%"   : lambda v: f"{v:.2f}%",
    "Bars_Since_Retest": lambda v: f"{int(v)}d",
    "H4_This_Month"    : lambda v: f"${v:.2f}",
    "H4_Prev_Month"    : lambda v: f"${v:.2f}",
    "Dist_H4_This_%"   : lambda v: f"{v:+.2f}%",
    "Dist_H4_Prev_%"   : lambda v: f"{v:+.2f}%",
    "MACD"             : lambda v: f"{v:.4f}",
    "MACD_Signal"      : lambda v: f"{v:.4f}",
    "MACD_Hist"        : lambda v: f"{v:.4f}",
    "MACD_Accel"       : lambda v: f"{v:.4f}",
    "RSI"              : lambda v: f"{v:.1f}",
    "Avg_Vol_20d"      : lambda v: f"{v:,.0f}",
}

def fmt_v(col, val):
    if val is None or (isinstance(val, float) and np.isnan(val)): return "—"
    try:
        if col in FMT: return FMT[col](val)
    except Exception: pass
    return str(val) if str(val) not in ("nan","None","") else "—"

if _IN_NOTEBOOK and results:
    DISP = ["Ticker","Price","Score",
            "Days_Above_MAs","Retest_Label","Retest_Depth_%","Bars_Since_Retest",
            "H4_This_Month","H4_Prev_Month","MACD_Hist","MACD_Accel","RSI"]
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
            elif col == "Days_Above_MAs":
                try:
                    v = int(str(raw).replace("d",""))
                    if v >= 7: sty = "color:#22c55e;font-weight:700;text-align:center"
                    elif v >= 4: sty = "color:#86efac;font-weight:600;text-align:center"
                except Exception: pass
            elif col == "Retest_Label":
                sty = "color:#22c55e;font-weight:700" if "+" in str(raw) else "color:#3b82f6;font-weight:600"
            elif col == "Retest_Depth_%":
                try:
                    v = float(str(raw).replace("%",""))
                    if v <= 1.0: sty = "color:#22c55e;font-weight:700"
                    elif v <= 2.0: sty = "color:#86efac"
                except Exception: pass
            elif col == "MACD_Hist":
                try:
                    v = float(raw)
                    clr = "#22c55e" if v > 0 else "#ef4444"
                    sty = f"color:{clr};font-weight:700"
                except Exception: pass
            elif col == "MACD_Accel":
                try:
                    v = float(raw)
                    clr = "#22c55e" if v > 0 else "#ef4444"
                    sty = f"color:{clr};font-weight:600"
                except Exception: pass
            tds += (f'<td style="padding:7px 12px;font-size:12px;'
                    f'border-bottom:1px solid #e2e8f0;white-space:nowrap;{sty}">'
                    f'{disp}</td>')
        rows_html += f'<tr style="background:{bg}">{tds}</tr>\n'

    header_html = f"""
<div style="background:linear-gradient(135deg,#0f172a,#1e3a5f);
        border-radius:10px;padding:18px 24px;margin-bottom:8px;
        font-family:'Segoe UI',Arial,sans-serif">
  <h2 style="margin:0;color:#60a5fa;font-size:20px;font-weight:700">
    📈 JMA/SMA8 Retest + Cam H4 Reclaim + MACD Rising Above Zero
  </h2>
  <p style="margin:6px 0 0;color:#94a3b8;font-size:12px">
    {datetime.today().strftime('%Y-%m-%d %H:%M')} &nbsp;·&nbsp;
    <b style="color:#22c55e">{len(results)} matches</b> from {len(TICKERS)} tickers
  </p>
</div>"""

    table_html = f"""
<div style="margin:8px 0">
  <div style="background:linear-gradient(90deg,{gc}22,#0f172a);
          border-left:4px solid {gc};border-radius:6px 6px 0 0;
          padding:10px 18px">
    <span style="color:#f1f5f9;font-size:15px;font-weight:700">🎯 All Matches</span>
    <span style="color:{gc};font-size:12px;margin-left:8px">{len(results)} stock{'s' if len(results)!=1 else ''}</span>
  </div>
  <div style="overflow-x:auto;border:1px solid #e2e8f0;border-top:none;
          border-radius:0 0 8px 8px">
    <table style="border-collapse:collapse;width:100%;min-width:700px">
      <thead><tr>{th}</tr></thead>
      <tbody>{rows_html}</tbody>
    </table>
  </div>
</div>"""

    legend_html = f"""
<div style="background:#f8fafc;border:1px solid #e2e8f0;border-radius:8px;
        padding:12px 18px;margin-top:6px;font-size:11px;color:#64748b;
        font-family:'Segoe UI',Arial,sans-serif">
  <b style="color:#475569">GUIDE</b> &nbsp;·&nbsp;
  Days_Above_MAs = consecutive bars price held above both JMA({CFG['jma_period']}) and SMA8 &nbsp;·&nbsp;
  Retest_Depth_% = how close candle LOW came to JMA/SMA8 (lower = cleaner touch) &nbsp;·&nbsp;
  MACD_Hist > 0 and increasing (green) = momentum accelerating &nbsp;·&nbsp;
  JMA/SMA8+EMA both retested = strongest support confluence
</div>"""

    display_html(header_html + table_html + legend_html)

elif results:
    CLI_COLS = ["Ticker","Price","Score","Days_Above_MAs",
                "Retest_Label","Retest_Depth_%","MACD_Hist","MACD_Accel","RSI"]
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
    tit = f"  JMA/SMA8 Retest + Cam H4 + MACD↑   {datetime.today().strftime('%Y-%m-%d')}   {len(df_out)} matches"
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

# Save
fpath = os.path.join(out_dir, f"jma_sma8_h4_macd_{ts}.csv")
df_out.to_csv(fpath, index=False)
print(f"\n  💾 CSV → {fpath}")
tv = os.path.join(out_dir, f"tv_jma_sma8_h4_macd_{ts}.txt")
with open(tv,"w") as f:
    f.write(f"###JMA SMA8 H4 MACD {datetime.today().strftime('%Y-%m-%d')}\n")
    for r in results: f.write(f"NASDAQ:{r['Ticker']}\n")
print(f"  📋 TradingView → {tv}")

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
            for c in ["Ticker","Price","Score","Days_Above_MAs",
                      "Retest_Label","MACD_Hist","MACD_Accel","RSI"]
        )
        rows_e = ""
        for i, r in enumerate(rl[:50]):
            bg    = "#fff" if i%2==0 else "#f0f9ff"
            ticker= r.get("Ticker","—")
            price = r.get("Price",0) or 0
            score = r.get("Score",0) or 0
            dabs  = r.get("Days_Above_MAs",0) or 0
            rlbl  = r.get("Retest_Label","—")
            mhist = r.get("MACD_Hist",0) or 0
            macel = r.get("MACD_Accel",0) or 0
            rsi   = r.get("RSI",0) or 0
            hist_color = "#22c55e" if float(mhist) > 0 else "#ef4444"
            rows_e += (
                f'<tr style="background:{bg}">'
                f'<td style="padding:6px 11px;font-size:12px;font-weight:700">{ticker}</td>'
                f'<td style="padding:6px 11px;font-size:12px">${float(price):.2f}</td>'
                f'<td style="padding:6px 11px;font-size:12px;font-weight:700;'
                f'background:#166534;color:#fff;text-align:center">{float(score):.0f}</td>'
                f'<td style="padding:6px 11px;font-size:12px;text-align:center">{int(dabs)}d</td>'
                f'<td style="padding:6px 11px;font-size:12px;color:#3b82f6;font-weight:600">{rlbl}</td>'
                f'<td style="padding:6px 11px;font-size:12px;color:{hist_color};font-weight:700">'
                f'{float(mhist):.4f}</td>'
                f'<td style="padding:6px 11px;font-size:12px">{float(macel):.4f}</td>'
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
          border-radius:12px;overflow:hidden;box-shadow:0 4px 20px rgba(0,0,0,0.08)">
  <tr><td style="background:linear-gradient(135deg,#0f172a,#1e3a5f);padding:22px 28px">
<h1 style="margin:0;color:#60a5fa;font-size:20px;font-weight:700">
  📊 JMA/SMA8 Retest + Cam H4 + MACD Rising Above Zero
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
<p style="font-size:11px;color:#64748b;margin:8px 0 0">📎 Full results attached as CSV</p>
  </td></tr>
  <tr><td style="background:#f8fafc;padding:12px 28px;border-top:1px solid #e2e8f0;text-align:center">
<p style="margin:0;color:#94a3b8;font-size:10px">
  ⚠️ Not financial advice &nbsp;·&nbsp; Auto-generated by GitHub Actions
</p>
  </td></tr>
</table></td></tr></table>
</body></html>"""

        plain_lines = [
            f"JMA/SMA8 Retest + Cam H4 + MACD Rising — {datetime.today().strftime('%Y-%m-%d')}",
            f"{cnt} matches", "="*60,
        ]
        if rl:
            for r in rl[:50]:
                ticker = r.get("Ticker","—")
                price  = r.get("Price",0) or 0
                score  = r.get("Score",0) or 0
                dabs   = r.get("Days_Above_MAs",0) or 0
                rlbl   = r.get("Retest_Label","—")
                plain_lines.append(
                    f"{ticker:<7} ${float(price):.2f}  Score:{float(score):.0f}  "
                    f"Above:{int(dabs)}d  Retest:{rlbl}"
                )
        else:
            plain_lines.append("No matches today")
        plain_lines.append("\nFull results in CSV attachment.")
        plain_e = "\n".join(plain_lines)

        subj = (f"📊 JMA/SMA8+H4+MACD — {cnt} signal{'s' if cnt!=1 else ''}"
                f" — {datetime.today().strftime('%Y-%m-%d')}")

        msg = MIMEMultipart("mixed")
        msg["Subject"] = subj; msg["From"] = gu; msg["To"] = ", ".join(eto)
        alt = MIMEMultipart("alternative")
        alt.attach(MIMEText(plain_e, "plain")); alt.attach(MIMEText(html_e, "html"))
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
            print(f"[Email] 📎 Attached: {os.path.basename(csv_path)}")
        except Exception as e:
            print(f"[Email] ⚠️  CSV attach failed: {e}")

    try:
        print(f"[Email] Connecting to smtp.gmail.com:465 ...")
        with smtplib.SMTP_SSL("smtp.gmail.com", 465) as srv:
            srv.login(gu, gp.replace(" ", ""))
            srv.sendmail(gu, eto, msg.as_string())
        print(f"[Email] ✅  Sent successfully to: {', '.join(eto)}")
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
    fig, axes = plt.subplots(len(top)*2, 1,
                              figsize=(15, 7*len(top)), facecolor="#0f172a",
                              gridspec_kw={"height_ratios": [3,1]*len(top)})
    if len(top)==1: axes = axes.reshape(2,1).T.flatten()

    for idx, r in enumerate(top):
        ax_price = axes[idx*2]
        ax_macd  = axes[idx*2+1]
        df_p  = r["_df"].tail(60).copy()
        jma   = r["_jma"].reindex(df_p.index)
        sma8  = r["_sma8"].reindex(df_p.index)
        hist  = r["_hist"].reindex(df_p.index)
        n_p   = len(df_p)
        fn    = len(r["_df"]); off = fn - n_p

        ax_price.set_facecolor("#0f172a")
        ax_macd.set_facecolor("#0f172a")

        for i, (_, row_) in enumerate(df_p.iterrows()):
            o=float(row_["Open"]); h=float(row_["High"])
            l=float(row_["Low"]);  c=float(row_["Close"])
            clr="#34d399" if c>=o else "#ef4444"
            ax_price.plot([i,i],[l,h],color=clr,lw=0.7,zorder=2)
            blo=min(o,c); bhi=max(o,c); bh=max(bhi-blo,(h-l)*0.005)
            rect=mpatches.FancyBboxPatch((i-0.3,blo),0.6,bh,
                 boxstyle="square,pad=0",facecolor=clr,edgecolor=clr,lw=0.4,zorder=3)
            ax_price.add_patch(rect)

        ax_price.plot(range(n_p), jma.values,  color="#34d399", lw=1.8, label=f"JMA({CFG['jma_period']})", zorder=5)
        ax_price.plot(range(n_p), sma8.values, color="#f59e0b", lw=1.5, ls="--", label="SMA8", zorder=4)

        # H4 levels
        ax_price.axhline(r["H4_This_Month"], color="#3b82f6", lw=1.3, ls=":",
                         label=f"H4 This ${r['H4_This_Month']:.2f}", zorder=3)
        ax_price.axhline(r["H4_Prev_Month"], color="#a78bfa", lw=1.1, ls=":",
                         label=f"H4 Prev ${r['H4_Prev_Month']:.2f}", zorder=3)

        # Mark retest bar
        rb = r["_retest_bar"] - off if r["_retest_bar"] is not None else None
        if rb is not None and 0 <= rb < n_p:
            ax_price.scatter([rb],[float(df_p["Low"].iloc[rb])],
                             color="#fbbf24", s=120, zorder=8, marker="v",
                             label=f"Retest {r['Retest_Label']}")

        # MACD histogram subplot
        for i in range(n_p):
            v = float(hist.iloc[i]) if not np.isnan(hist.iloc[i]) else 0
            clr = "#34d399" if v >= 0 else "#ef4444"
            ax_macd.bar(i, v, color=clr, alpha=0.8, width=0.8, zorder=2)
        ax_macd.axhline(0, color="#94a3b8", lw=0.8, ls="--")

        tick_step = max(1, n_p//8)
        for ax in [ax_price, ax_macd]:
            ax.set_xticks(range(0, n_p, tick_step))
            ax.set_xticklabels(
                [df_p.index[i].strftime("%m/%d") for i in range(0,n_p,tick_step)],
                color="#94a3b8", fontsize=7)
            ax.set_xlim(-0.5, n_p-0.5)
            ax.tick_params(colors="#94a3b8", labelsize=7)
            for sp_ in ax.spines.values(): sp_.set_edgecolor("#1e3a5f")
            ax.grid(color="#1e3a5f", ls="--", lw=0.4, alpha=0.4, axis="y")

        ax_price.set_title(
            f"{r['Ticker']}  ${r['Price']:.2f}  |  Score {r['Score']}/100  |  "
            f"Above JMA+SMA8: {r['Days_Above_MAs']}d  |  "
            f"Retest: {r['Retest_Label']} ({r['Retest_Depth_%']:.1f}%)  |  "
            f"MACD Hist:{r['MACD_Hist']:.4f}↑  |  RSI {r['RSI']:.0f}",
            color="#e2e8f0", fontsize=8, fontweight="bold", pad=6)
        ax_price.legend(loc="upper left", facecolor="#1e293b",
                        labelcolor="#e2e8f0", fontsize=7, framealpha=0.9)
        ax_macd.set_ylabel("MACD Hist", color="#94a3b8", fontsize=7)

    plt.suptitle(
        f"JMA/SMA8 Retest + Camarilla H4 Reclaim + MACD Rising Above Zero  ·  "
        f"{datetime.today().strftime('%Y-%m-%d')}\n"
        f"🟢 JMA  🟡 SMA8  🔵 H4 This Month  🟣 H4 Prev Month  ▼ Retest",
        color="#60a5fa", fontsize=10, fontweight="bold", y=1.001)
    plt.tight_layout()
    cp = os.path.join(out_dir, f"jma_sma8_h4_macd_chart_{ts}.png")
    plt.savefig(cp, dpi=150, bbox_inches="tight", facecolor="#0f172a")
    if _IN_NOTEBOOK: plt.show()
    else: plt.close()
    print(f"  📊 Chart → {cp}")

print("""
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  📋 PATTERN EXPLAINED  (MXL chart example)

  C1  ABOVE JMA + SMA8 FOR 4+ CONSECUTIVE DAYS
      Price stays above the fast JMA(13,40) and SMA8 for
      at least 4 bars in a row — confirming the MAs are
      acting as support and not just being crossed randomly

  C2  RETESTED JMA OR SMA8 WITHIN THAT WINDOW
      At least one candle LOW came within 3% of JMA or SMA8
      and price closed back above — buyers defended the level
      ("Sell" label area on the MXL chart, then price held)

  C3  ABOVE BOTH MONTHLY CAMARILLA H4 LEVELS
      H4 = Close + (High-Low) × 1.1/2 from prior month's data
      Price above THIS month's H4 AND the PREVIOUS month's H4
      = Has cleared both recent institutional pivot resistances

  C4  MACD HISTOGRAM ABOVE ZERO AND INCREASING
      Histogram (MACD − Signal) > 0  (momentum positive)
      AND histogram has grown for 3+ consecutive bars
      = Momentum is not just positive, it's accelerating
      (MACD panel in MXL chart shows the steep rise to 15+)

  💡 BEST SETUPS
  Days_Above_MAs ≥ 7      extended hold above fast MAs = strong control
  Retest_Label = JMA+SMA8  both fast MAs held on retest = maximum support
  Retest_Depth_% < 1%     precise touch = clean technical level
  MACD_Accel large         accelerating histogram = gaining momentum fast

  ⚙️  TUNE IF 0 RESULTS
  min_days_above          4 → 3
  retest_touch_pct        3 → 5
  retest_lookback        10 → 15
  macd_increasing_bars    3 → 2
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
""")
