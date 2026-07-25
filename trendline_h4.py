# ============================================================
# NASDAQ — Downtrend Trendline Break + Camarilla H4 Reclaim
# ============================================================
#
# EXACT PATTERN (from AMD chart — the red-circled base):
#
#  C1 — DOWNTREND TRENDLINE  (last ~6 months)
#      Find swing highs over the lookback window and fit a
#      descending trendline connecting them (line slopes down)
#      This is the resistance line price has been rejected from
#      repeatedly ("Sell" labels on the chart at each touch)
#
#  C2 — PRICE JUST CLOSED ABOVE THE TRENDLINE
#      EXACT 1-bar break (most recent bar):
#        Bar[-2] close < trendline value at bar[-2]
#        Bar[-1] close >= trendline value at bar[-1]
#      = Price broke the multi-month descending resistance
#
#  C3 — ABOVE BOTH THIS MONTH'S AND PREVIOUS MONTH'S CAMARILLA H4
#      H4 = Close + (High - Low) x 1.1 / 2
#      Current price must be above:
#        - Current (in-progress) month's H4  (derived from
#          prior month's H/L/C, per standard Camarilla convention)
#        - Previous completed month's H4
#      = Price has reclaimed both recent monthly resistance pivots
#
# LOGIC FLOW:
#   6-month downtrend (repeated rejection at descending trendline)
#   → Trendline break (first close above it)
#   → Confirmed by reclaiming both monthly Camarilla H4 levels
#   = High-conviction base breakout signal
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
    "history_days"               : 400,

    # ── C1: Downtrend trendline ────────────────────────────────
    # How far back to look for the downtrend (~6 months of bars)
    "trendline_lookback_bars"    : 126,   # ~6 months of trading days
    # Swing high detection: a bar is a swing high if it's the
    # highest High within this many bars on each side
    "swing_window"               : 5,
    # Minimum number of swing highs needed to fit a valid trendline
    "min_swing_points"           : 3,
    # The fitted trendline slope must be negative (downtrend) by
    # at least this % per bar (filters out near-flat/noisy fits)
    "min_downtrend_slope_pct"    : -0.02,
    # R² threshold for how well the swing highs fit a straight line
    "min_fit_quality"            : 0.5,

    # ── C2: Trendline break — exact 1-bar ─────────────────────
    "cross_lookback"             : 5,

    # ── C3: Camarilla H4 (this + previous month) ──────────────
    # Price must be above BOTH the current month's H4 (built from
    # the completed prior month) and the previous completed
    # month's H4 (built from the month before that)
    "require_above_both_h4"      : True,

    # ── Volume / quality ────────────────────────────────────────
    "vol_avg_bars"               : 20,
    "min_vol_mult"               : 0.8,
    "rsi_min"                    : 35,

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

def cam_h4(high, low, close):
    """Standard Camarilla H4 = Close + (High - Low) * 1.1 / 2"""
    return close + (high - low) * 1.1 / 2.0

def get_monthly_h4_levels(df):
    """
    Returns [current_month_H4, previous_month_H4] using the
    standard Camarilla convention: each month's H4 is derived
    from the PRIOR completed month's High/Low/Close.
    """
    df = df.copy()
    df.index = pd.to_datetime(df.index)
    today = pd.Timestamp.today().normalize()
    levels = []
    # offset=1 -> H4 for the CURRENT month (built from last month's H/L/C)
    # offset=2 -> H4 for the PREVIOUS completed month (built from month before that)
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

# ── Trendline detection (swing highs + linear fit) ──────────────
def find_swing_highs(high_series, window=5):
    """
    Returns list of (index_position, high_value) for each bar
    that is a local swing high within `window` bars on each side.
    """
    vals = high_series.values.astype(float)
    n    = len(vals)
    swings = []
    for i in range(window, n - window):
        seg = vals[i-window:i+window+1]
        if vals[i] == np.max(seg):
            swings.append((i, vals[i]))
    return swings

def fit_downtrend_line(swings, min_points=3, min_slope_pct=-0.02, min_r2=0.5):
    """
    Fits a straight line through the swing-high points using
    least squares. Returns (slope, intercept, r2) or None if the
    fit isn't a valid, sufficiently-clean downtrend.
    """
    if len(swings) < min_points:
        return None

    xs = np.array([s[0] for s in swings], dtype=float)
    ys = np.array([s[1] for s in swings], dtype=float)

    # Least squares fit: y = slope*x + intercept
    A = np.vstack([xs, np.ones(len(xs))]).T
    result = np.linalg.lstsq(A, ys, rcond=None)
    slope, intercept = result[0]

    # R² goodness of fit
    y_pred    = slope * xs + intercept
    ss_res    = np.sum((ys - y_pred) ** 2)
    ss_tot    = np.sum((ys - np.mean(ys)) ** 2)
    r2        = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0

    # Slope as % of average price level (normalize across price scales)
    avg_price = np.mean(ys)
    slope_pct = (slope / avg_price) * 100 if avg_price > 0 else 0

    if slope_pct > min_slope_pct:   # slope must be negative enough
        return None
    if r2 < min_r2:                 # points must reasonably fit a line
        return None

    return {"slope": slope, "intercept": intercept, "r2": r2, "slope_pct": slope_pct}

def trendline_value_at(line, x):
    """Evaluate the fitted trendline at position x."""
    return line["slope"] * x + line["intercept"]

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
    C1: Fit a descending trendline through swing highs over the
        last ~6 months
    C2: Exact 1-bar close above the trendline
    C3: Price above BOTH this month's and previous month's
        Camarilla H4
    """
    df      = df.copy(); df.index = pd.to_datetime(df.index)
    n       = len(df)
    price   = float(df["Close"].iloc[-1])
    avg_vol = float(df["Volume"].tail(CFG["vol_avg_bars"]).mean())

    if price   < CFG["min_price"]:      return None
    if avg_vol < CFG["min_avg_volume"]: return None

    lb = CFG["trendline_lookback_bars"]
    if n < lb + 10: return None

    rsi_s   = calc_rsi(df["Close"])
    cur_rsi = float(rsi_s.iloc[-1]) if not np.isnan(rsi_s.iloc[-1]) else 50
    if cur_rsi < CFG["rsi_min"]: return None

    # ─────────────────────────────────────────────────────────
    # C1: FIND DESCENDING TRENDLINE over the lookback window
    # (excluding the most recent few bars so the breakout itself
    #  doesn't distort the trendline fit)
    # ─────────────────────────────────────────────────────────
    window_end   = n - 1
    window_start = max(0, window_end - lb)
    sub_high     = df["High"].iloc[window_start:window_end]

    swings = find_swing_highs(sub_high, window=CFG["swing_window"])
    # Convert swing positions back to full-dataframe indices
    swings = [(pos + window_start, val) for pos, val in swings]

    line = fit_downtrend_line(
        swings,
        min_points=CFG["min_swing_points"],
        min_slope_pct=CFG["min_downtrend_slope_pct"],
        min_r2=CFG["min_fit_quality"],
    )
    if line is None: return None   # no valid downtrend found

    # ─────────────────────────────────────────────────────────
    # C2: EXACT 1-BAR TRENDLINE BREAK
    # Bar[-2] close < trendline value there
    # Bar[-1] close >= trendline value there
    # ─────────────────────────────────────────────────────────
    cl = CFG["cross_lookback"]
    break_bar  = None
    break_date = None

    for i in range(max(1, n - cl), n):
        tv_prev = trendline_value_at(line, i - 1)
        tv_cur  = trendline_value_at(line, i)
        pc      = float(df["Close"].iloc[i-1])
        cc      = float(df["Close"].iloc[i])
        if pc < tv_prev and cc >= tv_cur:
            break_bar  = i
            break_date = df.index[i]

    if break_bar is None: return None   # no trendline break found

    # Volume on the break bar
    break_vol = float(df["Volume"].iloc[break_bar])
    vol_mult  = break_vol / avg_vol if avg_vol > 0 else 0
    if vol_mult < CFG["min_vol_mult"]: return None

    # ─────────────────────────────────────────────────────────
    # C3: ABOVE BOTH THIS MONTH'S AND PREVIOUS MONTH'S CAM H4
    # ─────────────────────────────────────────────────────────
    h4_this, h4_prev = get_monthly_h4_levels(df)
    if h4_this is None or h4_prev is None: return None

    if CFG["require_above_both_h4"]:
        if price < h4_this: return None
        if price < h4_prev: return None

    # ── Metrics ───────────────────────────────────────────────
    bars_since_break = n - 1 - break_bar
    trendline_at_break = trendline_value_at(line, break_bar)
    break_close         = float(df["Close"].iloc[break_bar])
    break_dist_pct       = (break_close - trendline_at_break) / trendline_at_break * 100 if trendline_at_break > 0 else 0

    dist_h4_this_pct = (price - h4_this) / h4_this * 100 if h4_this else 0
    dist_h4_prev_pct = (price - h4_prev) / h4_prev * 100 if h4_prev else 0

    trendline_now_val = trendline_value_at(line, n - 1)
    dist_trendline_pct = (price - trendline_now_val) / trendline_now_val * 100 if trendline_now_val > 0 else 0

    swing_count = len(swings)
    downtrend_months = round(lb / 21, 1)

    # ── Score (0-100) ─────────────────────────────────────────
    score = 0

    # Trendline fit quality (0-25): tighter fit = more reliable line
    score += min(25, int(line["r2"] * 25))

    # Number of swing-high touches (0-20): more touches = stronger resistance broken
    score += min(20, swing_count * 4)

    # Break freshness (0-20): today = 20
    score += max(0, 20 - bars_since_break * 4)

    # Distance above both H4 levels (0-15): closer = fresher reclaim
    avg_h4_dist = (abs(dist_h4_this_pct) + abs(dist_h4_prev_pct)) / 2
    score += max(0, 15 - int(avg_h4_dist * 2))

    # Volume on break (0-15)
    score += min(15, int(vol_mult * 6))

    # RSI health (0-5)
    score += min(5, max(0, int((cur_rsi - 40) / 4)))

    score = min(100, max(0, score))

    return {
        "Ticker"              : sym,
        "Price"               : round(price, 2),
        "Score"               : score,
        # Trendline info
        "Downtrend_Months"    : downtrend_months,
        "Swing_Touches"       : swing_count,
        "Trendline_R2"        : round(line["r2"], 3),
        "Trendline_Slope_%"   : round(line["slope_pct"], 3),
        "Dist_Trendline_%"    : round(dist_trendline_pct, 2),
        # Break info
        "Break_Date"          : break_date.strftime("%Y-%m-%d"),
        "Bars_Since_Break"    : bars_since_break,
        "Break_Dist_%"        : round(break_dist_pct, 2),
        "Break_Vol_x"         : round(vol_mult, 2),
        # Camarilla H4
        "H4_This_Month"       : round(h4_this, 2),
        "H4_Prev_Month"       : round(h4_prev, 2),
        "Dist_H4_This_%"      : round(dist_h4_this_pct, 2),
        "Dist_H4_Prev_%"      : round(dist_h4_prev_pct, 2),
        # Indicators
        "RSI"                 : round(cur_rsi, 1),
        "Avg_Vol_20d"         : int(avg_vol),
        # internals
        "_df"       : df,
        "_line"     : line,
        "_swings"   : swings,
        "_break_bar": break_bar,
        "_window_start": window_start,
    }

# ── Live print ────────────────────────────────────────────────
LIVE_COLS = [
    "Ticker","Price","Score",
    "Downtrend_Months","Swing_Touches","Trendline_R2",
    "Break_Date","Bars_Since_Break","Break_Vol_x",
    "H4_This_Month","H4_Prev_Month","RSI",
]
_CW = {
    "Ticker":8,"Price":10,"Score":7,
    "Downtrend_Months":18,"Swing_Touches":15,"Trendline_R2":14,
    "Break_Date":13,"Bars_Since_Break":17,"Break_Vol_x":12,
    "H4_This_Month":15,"H4_Prev_Month":15,"RSI":6,
}
_CF = {
    "Price":"${:.2f}","Score":"{:.0f}",
    "Downtrend_Months":"{:.1f}mo","Swing_Touches":"{:.0f}","Trendline_R2":"{:.3f}",
    "Bars_Since_Break":"{:.0f}d","Break_Vol_x":"{:.2f}×",
    "H4_This_Month":"${:.2f}","H4_Prev_Month":"${:.2f}","RSI":"{:.1f}",
}
_hdr_done = False

def _live_header():
    global _hdr_done
    if _hdr_done: return
    sep = "━" * 190
    print(f"\n{sep}")
    print("  📊  LIVE MATCHES  —  Downtrend Trendline Break + Camarilla H4 Reclaim")
    print(sep)
    print("".join(f"  {c:<{_CW.get(c,10)}}" for c in LIVE_COLS))
    print("  " + "─"*188)
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
chk = download(["AMD","NVDA","AAPL"], 400)
if not chk: print("❌  No data.")
else:
    for s, d in chk.items():
        p = float(d["Close"].iloc[-1])
        swings = find_swing_highs(d["High"].tail(126), window=5)
        print(f"  ✅ {s}: {len(d)} bars  ${p:.2f}  swing highs found={len(swings)}  {d.index[-1].date()}")
print()

# ── Diagnostic ────────────────────────────────────────────────
print("━"*65)
print("  STEP 2  DIAGNOSTIC (10 sample stocks)")
print("━"*65+"\n")

DIAG = ["AMD","NVDA","AAPL","MSFT","PLTR","META","CRWD","AVGO","MU","SMCI"]
diag_data = download(DIAG, CFG["history_days"])
print(f"  Downloaded {len(diag_data)}/{len(DIAG)}\n")
print(f"  {'SYM':<7} {'PRICE':>8}  {'SWINGS':>7}  {'TREND':>6}  {'SCORE':>6}  RESULT")
print("  "+"─"*55)

for sym in DIAG:
    if sym not in diag_data:
        print(f"  {sym:<7} — no data"); continue
    try:
        df_d = diag_data[sym]
        p    = float(df_d["Close"].iloc[-1])
        r    = detect_pattern(sym, df_d)
        if r:
            print(f"  {sym:<7} ${p:>7.2f}  {r['Swing_Touches']:>7}  "
                  f"{'✅':>6}  {r['Score']:>6}  ✅ break {r['Break_Date']}")
        else:
            print(f"  {sym:<7} ${p:>7.2f}  {'—':>7}  {'—':>6}  {'—':>6}  ❌")
    except Exception as e:
        print(f"  {sym:<7} error: {e}")

print(f"""
  Pattern:
    C1  Downtrend trendline : fit descending line through swing highs
                              over last {CFG['trendline_lookback_bars']} bars (~6 months)
                              min {CFG['min_swing_points']} touches, R² >= {CFG['min_fit_quality']}
    C2  Trendline break     : Bar[-2] close < line  AND  Bar[-1] close >= line
    C3  Camarilla H4        : price above THIS month's H4 AND PREVIOUS month's H4
                              H4 = Close + (High-Low) x 1.1/2

  Tune if mostly ❌:
    min_fit_quality          0.5 → 0.3   (allow looser trendline fit)
    min_swing_points           3 → 2
    min_downtrend_slope_pct -0.02 → -0.01 (allow shallower downtrends)
    trendline_lookback_bars  126 → 90    (shorter lookback, ~4 months)
    min_vol_mult              0.8 → 0.5
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
    print("   min_fit_quality          0.5 → 0.3")
    print("   min_swing_points           3 → 2")
    print("   min_downtrend_slope_pct -0.02 → -0.01")
    print("   trendline_lookback_bars  126 → 90")
    print("   min_vol_mult              0.8 → 0.5")

# Sort by score (always runs, even on empty list)
results.sort(key=lambda x: x["Score"], reverse=True)

# ── Always build df_out and save/email (even if 0 results) ────
ts      = datetime.today().strftime("%Y%m%d_%H%M")
out_dir = os.environ.get("GITHUB_WORKSPACE", os.getcwd())

COLS = [
    "Ticker","Price","Score",
    "Downtrend_Months","Swing_Touches","Trendline_R2","Trendline_Slope_%",
    "Dist_Trendline_%","Break_Date","Bars_Since_Break","Break_Dist_%","Break_Vol_x",
    "H4_This_Month","H4_Prev_Month","Dist_H4_This_%","Dist_H4_Prev_%",
    "RSI","Avg_Vol_20d",
]
df_out = pd.DataFrame([{k:v for k,v in r.items() if not k.startswith("_")}
                        for r in results]) if results else pd.DataFrame(columns=COLS)
if not df_out.empty:
    df_out = df_out[[c for c in COLS if c in df_out.columns]]
    df_out.reset_index(drop=True, inplace=True)

FMT = {
    "Price"             : lambda v: f"${v:.2f}",
    "Score"             : lambda v: f"{v:.0f}",
    "Downtrend_Months"  : lambda v: f"{v:.1f}mo",
    "Swing_Touches"     : lambda v: f"{int(v)}",
    "Trendline_R2"      : lambda v: f"{v:.3f}",
    "Trendline_Slope_%" : lambda v: f"{v:.3f}%",
    "Dist_Trendline_%"  : lambda v: f"{v:+.2f}%",
    "Bars_Since_Break"  : lambda v: f"{int(v)}d",
    "Break_Dist_%"      : lambda v: f"{v:+.2f}%",
    "Break_Vol_x"       : lambda v: f"{v:.2f}×",
    "H4_This_Month"     : lambda v: f"${v:.2f}",
    "H4_Prev_Month"     : lambda v: f"${v:.2f}",
    "Dist_H4_This_%"    : lambda v: f"{v:+.2f}%",
    "Dist_H4_Prev_%"    : lambda v: f"{v:+.2f}%",
    "RSI"               : lambda v: f"{v:.1f}",
    "Avg_Vol_20d"       : lambda v: f"{v:,.0f}",
}

def fmt_v(col, val):
    if val is None or (isinstance(val, float) and np.isnan(val)): return "—"
    try:
        if col in FMT: return FMT[col](val)
    except Exception: pass
    return str(val) if str(val) not in ("nan","None","") else "—"

if _IN_NOTEBOOK and results:
    DISP = ["Ticker","Price","Score",
            "Downtrend_Months","Swing_Touches","Trendline_R2",
            "Break_Date","Bars_Since_Break","Break_Vol_x",
            "H4_This_Month","H4_Prev_Month","RSI"]
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
            elif col == "Swing_Touches":
                try:
                    v = int(float(raw))
                    if v >= 5: sty = "color:#22c55e;font-weight:700;text-align:center"
                    elif v >= 3: sty = "color:#86efac;text-align:center"
                except Exception: pass
            elif col == "Trendline_R2":
                try:
                    v = float(raw)
                    if v >= 0.8: sty = "color:#22c55e;font-weight:700"
                    elif v >= 0.6: sty = "color:#86efac"
                except Exception: pass
            elif col == "Bars_Since_Break":
                try:
                    v = int(str(raw).replace("d",""))
                    if v == 0: sty = "color:#22c55e;font-weight:700;text-align:center"
                    elif v <= 1: sty = "color:#86efac;text-align:center"
                except Exception: pass
            elif col == "Break_Vol_x":
                try:
                    v = float(str(raw).replace("×",""))
                    if v >= 2: sty = "color:#f59e0b;font-weight:700"
                    elif v >= 1.5: sty = "color:#fbbf24;font-weight:600"
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
    <span style="color:#f1f5f9;font-size:15px;font-weight:700">Downtrend Trendline Break + Camarilla H4 Reclaim</span>
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
    📈 Downtrend Trendline Break + Camarilla H4 Reclaim
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
  Swing_Touches = number of swing highs the descending trendline was fitted through &nbsp;·&nbsp;
  Trendline_R2 = fit quality (closer to 1.0 = cleaner line) &nbsp;·&nbsp;
  H4_This_Month / H4_Prev_Month = monthly Camarilla H4 pivot levels &nbsp;·&nbsp;
  Bars_Since_Break 0 = broke trendline today
</div>"""

    display_html(header_html + table_html + legend_html)

elif results:
    CLI_COLS = ["Ticker","Price","Score",
                "Downtrend_Months","Swing_Touches","Trendline_R2",
                "Break_Date","Bars_Since_Break","Break_Vol_x","RSI"]
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
    tit  = f"  Trendline Break + Cam H4 Reclaim   {datetime.today().strftime('%Y-%m-%d')}   {len(df_out)} matches"
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
  Downtrend_Months   how far back the trendline lookback spans
  Swing_Touches       number of swing highs fitted on the line
  Trendline_R2        fit quality (closer to 1.0 = cleaner)
  Break_Date          date price broke above the trendline
  Bars_Since_Break     0 = broke today
  Break_Vol_x          volume on the break bar vs 20d average
  ──────────────────────────────────────────────────────""")

# Save
fpath = os.path.join(out_dir, f"trendline_break_h4_{ts}.csv")
df_out.to_csv(fpath, index=False)
print(f"\n  💾 CSV → {fpath}")
tv = os.path.join(out_dir, f"tv_trendline_break_h4_{ts}.txt")
with open(tv,"w") as f:
    f.write(f"###Trendline Break + Cam H4 {datetime.today().strftime('%Y-%m-%d')}\n")
    for r in results: f.write(f"NASDAQ:{r['Ticker']}\n")
print(f"  📋 TradingView → {tv}")

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
            for c in ["Ticker","Price","Score","Swing_Touches","Trendline_R2",
                      "Bars_Since_Break","Break_Vol_x","RSI"]
        )
        rows_e = ""
        for i, r in enumerate(rl[:50]):
            bg  = "#fff" if i % 2 == 0 else "#f0f9ff"
            ticker = r.get("Ticker","—")
            price  = r.get("Price",0) or 0
            score  = r.get("Score",0) or 0
            swings = r.get("Swing_Touches",0) or 0
            r2     = r.get("Trendline_R2",0) or 0
            bsb    = r.get("Bars_Since_Break",99)
            volx   = r.get("Break_Vol_x",0) or 0
            rsi    = r.get("RSI",0) or 0
            rows_e += (
                f'<tr style="background:{bg}">'
                f'<td style="padding:6px 11px;font-size:12px;font-weight:700">{ticker}</td>'
                f'<td style="padding:6px 11px;font-size:12px">${float(price):.2f}</td>'
                f'<td style="padding:6px 11px;font-size:12px;font-weight:700;'
                f'background:#166534;color:#fff;text-align:center">{float(score):.0f}</td>'
                f'<td style="padding:6px 11px;font-size:12px;text-align:center">{int(swings)}</td>'
                f'<td style="padding:6px 11px;font-size:12px">{float(r2):.3f}</td>'
                f'<td style="padding:6px 11px;font-size:12px;color:'
                f'{"#22c55e" if bsb==0 else "#94a3b8"};font-weight:700">'
                f'{bsb}d</td>'
                f'<td style="padding:6px 11px;font-size:12px">{float(volx):.2f}×</td>'
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
  📊 Downtrend Trendline Break + Camarilla H4 Reclaim
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
            f"Downtrend Trendline Break + Camarilla H4 Reclaim — {datetime.today().strftime('%Y-%m-%d')}",
            f"{cnt} matches",
            "="*60,
        ]
        if rl:
            for r in rl[:50]:
                ticker = r.get("Ticker","—")
                price  = r.get("Price",0) or 0
                score  = r.get("Score",0) or 0
                swings = r.get("Swing_Touches",0) or 0
                bsb    = r.get("Bars_Since_Break",0) or 0
                plain_lines.append(
                    f"{ticker:<7} ${float(price):.2f}  Score:{float(score):.0f}  "
                    f"Swings:{int(swings)}  Break:{bsb}d ago"
                )
        else:
            plain_lines.append("No matches today")
        plain_lines.append("\nFull results in CSV attachment.")
        plain_e = "\n".join(plain_lines)

        subj = (f"📊 Trendline Break + H4 — {cnt} signal{'s' if cnt!=1 else ''}"
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
    fig, axes = plt.subplots(len(top), 1, figsize=(15, 5.5*len(top)), facecolor="#0f172a")
    if len(top)==1: axes=[axes]

    for idx, r in enumerate(top):
        ax    = axes[idx]
        df_p  = r["_df"].tail(180).copy()   # show enough history for the trendline
        n_p   = len(df_p)
        fn    = len(r["_df"]); off = fn - n_p
        line  = r["_line"]

        ax.set_facecolor("#0f172a")

        for i, (_, row_) in enumerate(df_p.iterrows()):
            o=float(row_["Open"]); h=float(row_["High"])
            l=float(row_["Low"]);  c=float(row_["Close"])
            clr="#34d399" if c>=o else "#ef4444"
            ax.plot([i,i],[l,h],color=clr,lw=0.6,zorder=2)
            blo=min(o,c); bhi=max(o,c); bh=max(bhi-blo,(h-l)*0.005)
            rect=mpatches.FancyBboxPatch((i-0.3,blo),0.6,bh,
                 boxstyle="square,pad=0",facecolor=clr,edgecolor=clr,lw=0.3,zorder=3)
            ax.add_patch(rect)

        # Plot the fitted downtrend trendline across the chart
        xs_full = np.arange(n_p)
        xs_orig = xs_full + off   # positions in the ORIGINAL full dataframe
        line_vals = np.array([trendline_value_at(line, x) for x in xs_orig])
        ax.plot(xs_full, line_vals, color="#ef4444", lw=2.0, ls="--",
                label="Downtrend Trendline", zorder=5)

        # Mark swing-high touch points
        for pos, val in r["_swings"]:
            plot_pos = pos - off
            if 0 <= plot_pos < n_p:
                ax.scatter([plot_pos], [val], color="#f87171", s=60,
                           zorder=6, marker="v")

        # Mark the H4 levels as horizontal lines
        ax.axhline(r["H4_This_Month"], color="#f59e0b", lw=1.4, ls=":",
                   alpha=0.9, label=f"H4 This Month ${r['H4_This_Month']:.2f}", zorder=4)
        ax.axhline(r["H4_Prev_Month"], color="#fbbf24", lw=1.2, ls=":",
                   alpha=0.7, label=f"H4 Prev Month ${r['H4_Prev_Month']:.2f}", zorder=4)

        # Mark the break bar
        bb = r["_break_bar"] - off
        if 0 <= bb < n_p:
            ax.axvline(bb, color="#22c55e", lw=1.8, ls="--", alpha=0.9)
            ax.scatter([bb], [float(df_p["Close"].iloc[bb])],
                       color="#22c55e", s=180, zorder=8, marker="^",
                       label=f"Trendline Break {r['Break_Date']}")

        tick_step = max(1, n_p//8)
        ax.set_xticks(range(0, n_p, tick_step))
        ax.set_xticklabels(
            [df_p.index[i].strftime("%m/%d") for i in range(0,n_p,tick_step)],
            color="#94a3b8", fontsize=7)
        ax.set_xlim(-0.5, n_p-0.5)
        ax.set_title(
            f"{r['Ticker']}  ${r['Price']:.2f}  |  Score {r['Score']}/100  |  "
            f"{r['Downtrend_Months']:.1f}mo downtrend, {r['Swing_Touches']} touches (R²={r['Trendline_R2']:.2f})  |  "
            f"Break {r['Break_Date']} ({r['Bars_Since_Break']}d ago) Vol {r['Break_Vol_x']:.1f}×  |  "
            f"RSI {r['RSI']:.0f}",
            color="#e2e8f0", fontsize=8, fontweight="bold", pad=6)
        ax.tick_params(colors="#94a3b8", labelsize=7)
        for sp_ in ax.spines.values(): sp_.set_edgecolor("#1e3a5f")
        ax.legend(loc="upper left", facecolor="#1e293b",
                  labelcolor="#e2e8f0", fontsize=7, framealpha=0.9)
        ax.grid(color="#1e3a5f", ls="--", lw=0.4, alpha=0.4, axis="y")

    plt.suptitle(
        f"Downtrend Trendline Break + Camarilla H4 Reclaim  ·  "
        f"{datetime.today().strftime('%Y-%m-%d')}\n"
        f"🔴 Descending Trendline  🟠 H4 This Month  🟡 H4 Prev Month  ▼ Swing High  ▲ Break Bar",
        color="#60a5fa", fontsize=10, fontweight="bold", y=1.001)
    plt.tight_layout()
    cp = os.path.join(out_dir, f"trendline_break_h4_chart_{ts}.png")
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
  📋 PATTERN EXPLAINED  (AMD chart example)

  C1  DOWNTREND TRENDLINE  (the red-circled base)
      A descending line is fitted through swing highs over the
      last ~6 months — the line the chart marks with repeated
      "Sell" labels each time price touched and got rejected
      Requires at least 3 touch points and a clean R² fit

  C2  TRENDLINE BREAK  (the trigger)
      Bar[-2] close was BELOW the trendline
      Bar[-1] close is ABOVE the trendline
      = Price broke the multi-month descending resistance

  C3  ABOVE BOTH MONTHLY CAMARILLA H4 LEVELS
      H4 = Close + (High - Low) x 1.1 / 2
      Price must be above THIS month's H4 AND the PREVIOUS
      month's H4 — confirming the breakout through recent
      institutional resistance pivots, not just the trendline

  WHY THIS COMBINATION:
      A lone trendline break can be a false signal.
      Requiring it to ALSO clear two consecutive months of
      Camarilla H4 resistance adds a second, independent
      confirmation — exactly the kind of base-breakout setup
      visible in the AMD chart before its parabolic move.

  💡 BEST SETUPS
  Swing_Touches >= 5        many rejections = stronger line broken
  Trendline_R2 >= 0.8       clean, reliable trendline fit
  Bars_Since_Break = 0      fresh break today = earliest entry
  Break_Vol_x > 2×          strong volume confirming the break
  Dist_H4_This/Prev small   just cleared, not yet extended

  ⚙️  TUNE IF 0 RESULTS
  min_fit_quality          0.5 → 0.3
  min_swing_points           3 → 2
  min_downtrend_slope_pct -0.02 → -0.01
  trendline_lookback_bars  126 → 90
  min_vol_mult              0.8 → 0.5
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
""")
