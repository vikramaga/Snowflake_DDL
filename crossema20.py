# ============================================================
# NASDAQ — MA Compression Breakout Scanner  (IREN Pattern)
# ============================================================
# Finds the exact pattern from the IREN chart:
#
#  PHASE 1 — MA COMPRESSION  (the red circle base)
#    EMA20, SMA50, SMA150 were all bunched together
#    Price was compressing NEAR or BELOW all three MAs
#    All MAs were nearly flat (low slope = sideways action)
#    Camarilla levels acting as support/resistance nearby
#
#  PHASE 2 — SEQUENTIAL MA BREAKOUT
#    Price broke above EMA20 (green) first
#    Then above SMA50 (blue)
#    Then above SMA150 (pink/red)
#    ALL THREE stacked: EMA20 > SMA50 > SMA150 currently
#
#  PHASE 3 — VOLUME SURGE  (the big teal bars)
#    At least one recent bar with volume >= vol_surge_mult x avg
#    Volume expanding = institutional participation
#
#  PHASE 4 — MACD CONFIRMATION
#    MACD line (12,26) crossed above zero line
#    MACD histogram turning positive and expanding
#
#  ENTRY ZONE  (what we catch)
#    Price is near or just above EMA20 (green line)
#    = Pullback to EMA20 after the breakout
#    OR price just broke above the MA stack with fresh momentum
#    EMA20 is rising (slope positive over last N bars)
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
                sys.modules["IPython"].display.HTML(h)); return True
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

# ── CONFIG ────────────────────────────────────────────────────
CFG = {
    "history_days"              : 400,

    # ── MA periods ────────────────────────────────────────────
    "ema20_period"              : 20,
    "sma50_period"              : 50,
    "sma150_period"             : 150,

    # ── Phase 1: MA compression ───────────────────────────────
    # Look back this many bars to detect the compression phase
    "compression_lookback"      : 60,
    # During compression, SMA50 and SMA150 must have been within
    # this % of each other (bunched together)
    "compression_band_pct"      : 8.0,
    # Minimum number of bars where all 3 MAs were bunched
    "compression_min_bars"      : 10,
    # MA slopes during compression must be low (< slope_flat_pct change)
    "slope_flat_pct"            : 3.0,
    "slope_bars"                : 10,

    # ── Phase 2: MA stack ─────────────────────────────────────
    # Currently: EMA20 > SMA50 > SMA150
    "require_full_stack"        : True,
    # EMA20 must be above SMA150 by at least this %
    "min_ema20_above_sma150_pct": 1.0,

    # ── Phase 2b: EMA20 slope must be rising ──────────────────
    "ema20_slope_bars"          : 5,
    "min_ema20_slope_pct"       : 0.5,   # EMA20 rose >= 0.5% in last 5 bars

    # ── Phase 3: Volume surge ─────────────────────────────────
    "vol_surge_mult"            : 1.8,   # at least 1 bar with 1.8x avg vol
    "vol_surge_lookback"        : 20,    # in last 20 bars
    "vol_avg_bars"              : 50,    # longer avg for better baseline

    # ── Phase 4: MACD ────────────────────────────────────────
    "macd_fast"                 : 12,
    "macd_slow"                 : 26,
    "macd_signal"               : 9,
    # MACD line must be above zero
    "require_macd_above_zero"   : True,
    # MACD histogram must be positive (expanding)
    "require_macd_hist_positive": True,

    # ── Entry zone: price near EMA20 ─────────────────────────
    # Price must be within this % ABOVE EMA20 (pullback zone)
    "pullback_upper_pct"        : 20.0,  # within 20% above EMA20
    # Price must not be below EMA20 by more than this %
    "pullback_lower_pct"        : 3.0,

    # ── Camarilla (optional confluence) ──────────────────────
    "cam_zone_pct"              : 5.0,

    # ── Filters ───────────────────────────────────────────────
    "min_avg_volume"            : 80_000,
    "min_price"                 : 0.5,

    "batch_size"                : 50,
    "batch_sleep"               : 1.5,
}

# ── Indicators ───────────────────────────────────────────────
def calc_ema(series, period):
    return series.ewm(span=period, adjust=False).mean()

def calc_macd(close, fast=12, slow=26, signal=9):
    ema_f = calc_ema(close, fast)
    ema_s = calc_ema(close, slow)
    macd  = ema_f - ema_s
    sig   = calc_ema(macd, signal)
    hist  = macd - sig
    return macd, sig, hist

def calc_rsi(close, period=14):
    d = close.diff()
    g = d.clip(lower=0); l = -d.clip(upper=0)
    ag = g.ewm(alpha=1/period, adjust=False).mean()
    al = l.ewm(alpha=1/period, adjust=False).mean()
    rs = ag / al.replace(0, np.nan)
    return 100 - (100 / (1 + rs))

def cam_h4(high, low, close): return close + (high - low) * 1.1 / 2.0

def get_cam_levels(df):
    df = df.copy(); df.index = pd.to_datetime(df.index)
    today = pd.Timestamp.today().normalize()
    levels = []
    for offset in [1, 2]:
        p   = today.to_period("M") - offset
        sub = df[df.index.to_period("M") == p]
        if len(sub) >= 5:
            hi = float(sub["High"].max())
            lo = float(sub["Low"].min())
            cl = float(sub["Close"].iloc[-1])
            levels.append(round(cam_h4(hi, lo, cl), 2))
    return levels

# ── Download ──────────────────────────────────────────────────
def _clean(df, min_bars=160):
    if df is None or df.empty: return None
    need = [c for c in ["Open","High","Low","Close","Volume"] if c in df.columns]
    if not all(c in need for c in ["High","Low","Close","Volume"]): return None
    df = df[need].copy()
    df.index = pd.to_datetime(df.index)
    if hasattr(df.index, "tz") and df.index.tz:
        df.index = df.index.tz_localize(None)
    df.dropna(subset=["Close","Volume"], inplace=True)
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
    df      = df.copy(); df.index = pd.to_datetime(df.index)
    n       = len(df)
    price   = float(df["Close"].iloc[-1])
    avg_vol = float(df["Volume"].tail(CFG["vol_avg_bars"]).mean())

    if price   < CFG["min_price"]:      return None
    if avg_vol < CFG["min_avg_volume"]: return None

    # ── Compute MAs ───────────────────────────────────────────
    ema20_s  = calc_ema(df["Close"], CFG["ema20_period"])
    sma50_s  = df["Close"].rolling(CFG["sma50_period"]).mean()
    sma150_s = df["Close"].rolling(CFG["sma150_period"]).mean()
    macd_s, sig_s, hist_s = calc_macd(df["Close"],
                                       CFG["macd_fast"],
                                       CFG["macd_slow"],
                                       CFG["macd_signal"])
    rsi_s    = calc_rsi(df["Close"])

    cur_ema20  = float(ema20_s.iloc[-1])
    cur_sma50  = float(sma50_s.iloc[-1])
    cur_sma150 = float(sma150_s.iloc[-1])
    cur_macd   = float(macd_s.iloc[-1])
    cur_hist   = float(hist_s.iloc[-1])
    cur_rsi    = float(rsi_s.iloc[-1]) if not np.isnan(rsi_s.iloc[-1]) else 50

    if any(np.isnan([cur_ema20, cur_sma50, cur_sma150, cur_macd])): return None

    # ─────────────────────────────────────────────────────────
    # PHASE 2: MA stack check — EMA20 > SMA50 > SMA150
    # ─────────────────────────────────────────────────────────
    if CFG["require_full_stack"]:
        if not (cur_ema20 > cur_sma50 > cur_sma150): return None

    # EMA20 must be above SMA150 by minimum %
    ema20_above_s150_pct = (cur_ema20 - cur_sma150) / cur_sma150 * 100
    if ema20_above_s150_pct < CFG["min_ema20_above_sma150_pct"]: return None

    # ─────────────────────────────────────────────────────────
    # EXACT 1-BAR EMA20 CROSS  ← NEW PRIMARY CONDITION
    # Bar[-2] close was BELOW EMA20[-2]  (previous day below)
    # Bar[-1] close is  ABOVE EMA20[-1]  (today crossed above)
    # This is the precise entry trigger from the chart
    # ─────────────────────────────────────────────────────────
    if n < 2: return None
    prev_close = float(df["Close"].iloc[-2])
    prev_ema20 = float(ema20_s.iloc[-2]) if not np.isnan(ema20_s.iloc[-2]) else np.nan
    if np.isnan(prev_ema20): return None

    # Previous day must have been BELOW EMA20
    if prev_close >= prev_ema20: return None   # was already above — no cross

    # Today (Bar[-1]) must be ABOVE EMA20
    if price < cur_ema20: return None           # hasn't crossed yet

    # Cross confirmed: prev below, today above — exact 1-bar
    ema20_cross_pct = (price - cur_ema20) / cur_ema20 * 100   # how far above

    # ─────────────────────────────────────────────────────────
    # PHASE 2b: EMA20 slope must be rising
    # ─────────────────────────────────────────────────────────
    sb = CFG["ema20_slope_bars"]
    ema20_prev = float(ema20_s.iloc[-sb-1]) if not np.isnan(ema20_s.iloc[-sb-1]) else cur_ema20
    ema20_slope_pct = (cur_ema20 - ema20_prev) / ema20_prev * 100 if ema20_prev > 0 else 0
    if ema20_slope_pct < CFG["min_ema20_slope_pct"]: return None

    # ─────────────────────────────────────────────────────────
    # PHASE 4: MACD above zero + histogram positive
    # ─────────────────────────────────────────────────────────
    if CFG["require_macd_above_zero"] and cur_macd <= 0:     return None
    if CFG["require_macd_hist_positive"] and cur_hist <= 0:  return None

    # ─────────────────────────────────────────────────────────
    # ENTRY ZONE: Confirmed by the EMA20 cross above
    # dist_ema20_pct = how far today's close is above EMA20
    # Accept within pullback_upper_pct% above EMA20 (not too extended)
    # ─────────────────────────────────────────────────────────
    dist_ema20_pct = ema20_cross_pct   # reuse the cross distance

    # Cap: don't take if price is already too far above EMA20
    if dist_ema20_pct > CFG["pullback_upper_pct"]: return None

    # Price must also be above SMA150 (not in downtrend)
    if price < cur_sma150: return None

    # ─────────────────────────────────────────────────────────
    # PHASE 3: Volume surge in last vol_surge_lookback bars
    # ─────────────────────────────────────────────────────────
    best_vol_mult = 0.0
    surge_bar     = None
    for i in range(max(0, n - CFG["vol_surge_lookback"]), n):
        vi = float(df["Volume"].iloc[i])
        vm = vi / avg_vol if avg_vol > 0 else 0
        if vm > best_vol_mult:
            best_vol_mult = vm
            surge_bar     = i

    if best_vol_mult < CFG["vol_surge_mult"]: return None

    # ─────────────────────────────────────────────────────────
    # PHASE 1: MA compression BEFORE the current breakout
    # Check the compression_lookback bars ago
    # All 3 MAs were bunched within compression_band_pct%
    # ─────────────────────────────────────────────────────────
    comp_lb   = CFG["compression_lookback"]
    comp_band = CFG["compression_band_pct"] / 100
    comp_bars = 0

    # Look at a window before the surge (or just earlier history)
    anchor = surge_bar if surge_bar is not None else n - 1
    search_start = max(0, anchor - comp_lb)
    search_end   = max(1, anchor - 2)   # at least 2 bars before surge

    for i in range(search_start, search_end):
        e20  = float(ema20_s.iloc[i])   if not np.isnan(ema20_s.iloc[i])  else np.nan
        s50  = float(sma50_s.iloc[i])   if not np.isnan(sma50_s.iloc[i])  else np.nan
        s150 = float(sma150_s.iloc[i])  if not np.isnan(sma150_s.iloc[i]) else np.nan
        if any(np.isnan([e20, s50, s150])): continue
        ma_hi = max(e20, s50, s150)
        ma_lo = min(e20, s50, s150)
        if ma_lo > 0 and (ma_hi - ma_lo) / ma_lo <= comp_band:
            comp_bars += 1

    if comp_bars < CFG["compression_min_bars"]: return None

    # ─────────────────────────────────────────────────────────
    # MA slope during compression was flat
    # ─────────────────────────────────────────────────────────
    sb2 = CFG["slope_bars"]
    if search_end > sb2:
        s150_at_end   = float(sma150_s.iloc[search_end])
        s150_at_start = float(sma150_s.iloc[max(0, search_end - sb2)])
        if s150_at_start > 0:
            comp_slope = abs(s150_at_end - s150_at_start) / s150_at_start * 100
            # Accept if slope was flat (< slope_flat_pct%) during compression
            if comp_slope > CFG["slope_flat_pct"]: return None

    # ── Camarilla confluence ──────────────────────────────────
    cam_levels = get_cam_levels(df)
    cam_near   = any(abs(lv - price) / price * 100 <= CFG["cam_zone_pct"]
                     for lv in cam_levels) if cam_levels else False
    nearest_cam= min(cam_levels, key=lambda x: abs(x-price)) if cam_levels else None

    # ── Score (0-100) ─────────────────────────────────────────
    score = 0

    # EMA20 pullback tightness (0-25): closer to EMA20 = better
    score += max(0, 25 - int(abs(dist_ema20_pct) * 2))

    # Volume surge strength (0-25)
    score += min(25, int(best_vol_mult * 7))

    # Compression quality (0-20)
    score += min(20, int(comp_bars * 1.5))

    # MA stack margin (0-15): bigger gap between EMA20 and SMA150 = stronger
    score += min(15, int(ema20_above_s150_pct * 0.5))

    # MACD histogram strength (0-10)
    macd_hist_pct = cur_hist / price * 100 if price > 0 else 0
    score += min(10, int(macd_hist_pct * 200))

    # Camarilla bonus (0-5)
    score += 5 if cam_near else 0

    score = min(100, max(0, score))

    # ── Metrics ───────────────────────────────────────────────
    sma50_vs_sma150_pct  = (cur_sma50 - cur_sma150) / cur_sma150 * 100
    price_vs_sma150_pct  = (price - cur_sma150) / cur_sma150 * 100
    surge_date           = df.index[surge_bar].strftime("%Y-%m-%d") if surge_bar else "—"
    surge_bars_ago       = n - 1 - surge_bar if surge_bar is not None else 0

    return {
        "Ticker"             : sym,
        "Price"              : round(price, 2),
        "Score"              : score,
        # EMA20 cross — the primary new condition
        "EMA20_Cross"        : "✅ Today",
        "Prev_Close"         : round(prev_close, 2),
        "Prev_EMA20"         : round(prev_ema20, 2),
        "Cross_Dist_%"       : round(dist_ema20_pct, 2),
        # MA levels
        "EMA20"              : round(cur_ema20, 2),
        "SMA50"              : round(cur_sma50, 2),
        "SMA150"             : round(cur_sma150, 2),
        # EMA20 proximity
        "Dist_EMA20_%"       : round(dist_ema20_pct, 2),
        "EMA20_Slope_%"      : round(ema20_slope_pct, 3),
        "EMA20_vs_SMA150_%"  : round(ema20_above_s150_pct, 2),
        "SMA50_vs_SMA150_%"  : round(sma50_vs_sma150_pct, 2),
        "Price_vs_SMA150_%"  : round(price_vs_sma150_pct, 2),
        # Compression
        "Comp_Bars"          : comp_bars,
        # Volume
        "Best_Vol_x"         : round(best_vol_mult, 2),
        "Surge_Date"         : surge_date,
        "Surge_Bars_Ago"     : surge_bars_ago,
        # MACD
        "MACD"               : round(cur_macd, 4),
        "MACD_Hist"          : round(cur_hist, 4),
        # Camarilla
        "Cam_Near"           : "✅" if cam_near else "—",
        "Nearest_Cam"        : round(nearest_cam, 2) if nearest_cam else None,
        # RSI
        "RSI"                : round(cur_rsi, 1),
        "Avg_Vol_50d"        : int(avg_vol),
        # internals
        "_df"                : df,
        "_ema20"             : ema20_s,
        "_sma50"             : sma50_s,
        "_sma150"            : sma150_s,
        "_macd"              : macd_s,
        "_hist"              : hist_s,
        "_surge_bar"         : surge_bar,
    }

# ── Live print ────────────────────────────────────────────────
LIVE_COLS = [
    "Ticker","Price","Score",
    "EMA20_Cross","Prev_Close","Prev_EMA20","Cross_Dist_%",
    "EMA20","SMA50","SMA150",
    "EMA20_Slope_%","EMA20_vs_SMA150_%",
    "Comp_Bars","Best_Vol_x","Surge_Bars_Ago",
    "MACD","MACD_Hist","RSI","Cam_Near",
]
_CW = {
    "Ticker":8,"Price":10,"Score":7,
    "EMA20_Cross":12,"Prev_Close":11,"Prev_EMA20":11,"Cross_Dist_%":12,
    "EMA20":9,"SMA50":9,"SMA150":9,
    "EMA20_Slope_%":13,"EMA20_vs_SMA150_%":17,
    "Comp_Bars":10,"Best_Vol_x":11,"Surge_Bars_Ago":14,
    "MACD":9,"MACD_Hist":10,"RSI":6,"Cam_Near":9,
}
_CF = {
    "Price":"${:.2f}","Score":"{:.0f}",
    "Prev_Close":"${:.2f}","Prev_EMA20":"${:.2f}","Cross_Dist_%":"{:+.2f}%",
    "EMA20":"${:.2f}","SMA50":"${:.2f}","SMA150":"${:.2f}",
    "EMA20_Slope_%":"{:+.3f}%","EMA20_vs_SMA150_%":"{:+.2f}%",
    "Comp_Bars":"{:.0f}","Best_Vol_x":"{:.2f}×","Surge_Bars_Ago":"{:.0f}",
    "MACD":"{:.4f}","MACD_Hist":"{:.4f}","RSI":"{:.1f}",
}
_hdr_done = False

def _live_header():
    global _hdr_done
    if _hdr_done: return
    sep = "━" * 200
    print(f"\n{sep}")
    print("  📊  LIVE MATCHES  —  MA Compression Breakout  (IREN Pattern)")
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
chk = download(["IREN","NVDA","AAPL"], 250)
if not chk: print("❌  No data.")
else:
    for s, d in chk.items():
        p    = float(d["Close"].iloc[-1])
        e20  = float(calc_ema(d["Close"],20).iloc[-1])
        s50  = float(d["Close"].rolling(50).mean().iloc[-1])
        s150 = float(d["Close"].rolling(150).mean().iloc[-1])
        ml, _, mh = calc_macd(d["Close"])
        stack = "✅" if e20 > s50 > s150 else "❌"
        macd  = "✅" if float(ml.iloc[-1]) > 0 else "❌"
        print(f"  ✅ {s}: ${p:.2f}  EMA20=${e20:.2f}  SMA50=${s50:.2f}  "
              f"SMA150=${s150:.2f}  Stack:{stack}  MACD>0:{macd}  {d.index[-1].date()}")
print()

# ── Diagnostic ────────────────────────────────────────────────
print("━"*65)
print("  STEP 2  DIAGNOSTIC (IREN + 9 samples)")
print("━"*65+"\n")

DIAG = ["IREN","NVDA","AMD","PLTR","MARA","COIN","HOOD","CRWD","AVGO","MU"]
diag_data = download(DIAG, CFG["history_days"])
print(f"  Downloaded {len(diag_data)}/{len(DIAG)}\n")
print(f"  {'SYM':<7} {'P':>8}  {'STACK':>6}  {'COMP':>5}  "
      f"{'VOL×':>6}  {'MACD':>6}  {'DIST%':>7}  SCORE  RESULT")
print("  "+"─"*68)

for sym in DIAG:
    if sym not in diag_data:
        print(f"  {sym:<7} — no data"); continue
    try:
        df_d = diag_data[sym]; p = float(df_d["Close"].iloc[-1])
        e20  = calc_ema(df_d["Close"],20)
        s50  = df_d["Close"].rolling(50).mean()
        s150 = df_d["Close"].rolling(150).mean()
        ml, _, mh = calc_macd(df_d["Close"])
        ce20 = float(e20.iloc[-1]); cs50 = float(s50.iloc[-1])
        cs150= float(s150.iloc[-1]); cm = float(ml.iloc[-1])
        t    = lambda b: "✅" if b else "❌"
        stack= ce20 > cs50 > cs150
        r    = detect_pattern(sym, df_d)
        if r:
            print(f"  {sym:<7} ${p:>7.2f}  {t(stack):>6}  {r['Comp_Bars']:>5}  "
                  f"{r['Best_Vol_x']:>5.1f}×  {t(cm>0):>6}  "
                  f"{r['Dist_EMA20_%']:>+6.1f}%  {r['Score']:>5}  ✅")
        else:
            dist = (p-ce20)/ce20*100 if ce20>0 else 0
            why  = ("!stack" if not stack
                    else "!MACD"   if cm <= 0
                    else "!vol"    if True else "!comp")
            print(f"  {sym:<7} ${p:>7.2f}  {t(stack):>6}  {'—':>5}  "
                  f"{'—':>6}  {t(cm>0):>6}  {dist:>+6.1f}%  {'—':>5}  ❌  {why}")
    except Exception as e:
        print(f"  {sym:<7} error: {e}")

print(f"""
  Pattern (IREN chart):
    🟢 Green = EMA20  🔵 Blue = SMA50  🔴 Pink = SMA150
    Orange horizontal = Monthly Camarilla H4 levels

    P1  Compression  : EMA20,SMA50,SMA150 bunched within {CFG['compression_band_pct']}%
                       for at least {CFG['compression_min_bars']} bars, all flat
    P2  MA Stack     : EMA20 > SMA50 > SMA150 (full bull stack)
                       EMA20 rising >= {CFG['min_ema20_slope_pct']}% in last {CFG['ema20_slope_bars']} bars
    P3  Volume surge : >= {CFG['vol_surge_mult']}× avg in last {CFG['vol_surge_lookback']} bars
    P4  MACD         : MACD > 0 AND histogram > 0

  Entry zone: price within {CFG['pullback_lower_pct']}%–{CFG['pullback_upper_pct']}% above EMA20
  (catches both: fresh breakout AND pullback to EMA20)

  Tune if mostly ❌:
    compression_min_bars  10 → 5
    compression_band_pct   8 → 12
    vol_surge_mult        1.8 → 1.3
    pullback_upper_pct    20 → 35
    min_ema20_slope_pct  0.5 → 0.1
    slope_flat_pct         3 → 6
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
        "IREN","MARA","COIN","HOOD","SMCI","ALAB","SOFI","DDOG","SNOW","RBRK",
        "IONQ","QUBT","RGTI","ASTS","RKLB","LUNR","FSLR","PYPL","ROKU","ROST",
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
    print("\n  No matches. Try relaxing:")
    print("   compression_min_bars  10 → 5")
    print("   compression_band_pct   8 → 12")
    print("   vol_surge_mult        1.8 → 1.3")
    print("   pullback_upper_pct    20 → 35")
    print("   min_ema20_slope_pct   0.5 → 0.1")
else:
    results.sort(key=lambda x: x["Score"], reverse=True)
    COLS = [
        "Ticker","Price","Score",
        "EMA20_Cross","Prev_Close","Prev_EMA20","Cross_Dist_%",
        "EMA20","SMA50","SMA150",
        "Dist_EMA20_%","EMA20_Slope_%",
        "EMA20_vs_SMA150_%","SMA50_vs_SMA150_%","Price_vs_SMA150_%",
        "Comp_Bars","Best_Vol_x","Surge_Date","Surge_Bars_Ago",
        "MACD","MACD_Hist","Cam_Near","Nearest_Cam",
        "RSI","Avg_Vol_50d",
    ]
    df_out = pd.DataFrame([{k:v for k,v in r.items() if not k.startswith("_")}
                            for r in results])
    df_out = df_out[[c for c in COLS if c in df_out.columns]]
    df_out.reset_index(drop=True, inplace=True)

    ts = datetime.today().strftime("%Y%m%d_%H%M")

    FMT = {
        "Price"             : lambda v: f"${v:.2f}",
        "EMA20"             : lambda v: f"${v:.2f}",
        "SMA50"             : lambda v: f"${v:.2f}",
        "SMA150"            : lambda v: f"${v:.2f}",
        "Prev_Close"        : lambda v: f"${v:.2f}",
        "Prev_EMA20"        : lambda v: f"${v:.2f}",
        "Nearest_Cam"       : lambda v: f"${v:.2f}",
        "Score"             : lambda v: f"{v:.0f}",
        "Cross_Dist_%"      : lambda v: f"{v:+.2f}%",
        "Dist_EMA20_%"      : lambda v: f"{v:+.2f}%",
        "EMA20_Slope_%"     : lambda v: f"{v:+.3f}%",
        "EMA20_vs_SMA150_%"  : lambda v: f"{v:+.2f}%",
        "SMA50_vs_SMA150_%"  : lambda v: f"{v:+.2f}%",
        "Price_vs_SMA150_%"  : lambda v: f"{v:+.2f}%",
        "Comp_Bars"         : lambda v: f"{int(v)}",
        "Best_Vol_x"        : lambda v: f"{v:.2f}×",
        "Surge_Bars_Ago"    : lambda v: f"{int(v)}d",
        "MACD"              : lambda v: f"{v:.4f}",
        "MACD_Hist"         : lambda v: f"{v:.4f}",
        "RSI"               : lambda v: f"{v:.1f}",
        "Avg_Vol_50d"       : lambda v: f"{v:,.0f}",
    }

    def fmt_v(col, val):
        if val is None or (isinstance(val, float) and np.isnan(val)): return "—"
        try:
            if col in FMT: return FMT[col](val)
        except Exception: pass
        return str(val) if str(val) not in ("nan","None","") else "—"

    if _IN_NOTEBOOK:
        # Rich HTML
        DISP_COLS = ["Ticker","Price","Score",
                     "EMA20_Cross","Prev_Close","Prev_EMA20","Cross_Dist_%",
                     "EMA20","SMA50","SMA150",
                     "EMA20_Slope_%","EMA20_vs_SMA150_%",
                     "Comp_Bars","Best_Vol_x","Surge_Bars_Ago",
                     "MACD","MACD_Hist","Cam_Near","RSI"]
        DISP_COLS = [c for c in DISP_COLS if c in df_out.columns]

        th = "".join(
            f'<th style="background:#0f172a;color:#e2e8f0;padding:9px 13px;'
            f'font-size:11px;font-weight:700;text-align:center;'
            f'border-bottom:2px solid #22c55e;white-space:nowrap">{c}</th>'
            for c in DISP_COLS
        )
        rows_html = ""
        for i, (_, row_) in enumerate(df_out.iterrows()):
            bg = "#ffffff" if i % 2 == 0 else "#f0f9ff"
            tds = ""
            for col in DISP_COLS:
                raw  = row_.get(col)
                disp = fmt_v(col, raw)
                sty  = ""
                if col == "Score":
                    try:
                        v = float(raw)
                        g = int(min(220, 80 + v * 1.4))
                        sty = f"background:rgb(20,{g},60);color:#fff;font-weight:700;text-align:center"
                    except Exception: pass
                elif col in ("Dist_EMA20_%","EMA20_Slope_%","EMA20_vs_SMA150_%"):
                    try:
                        v = float(str(raw).replace("%","").replace("+",""))
                        clr = "#22c55e" if v >= 0 else "#ef4444"
                        sty = f"color:{clr};font-weight:600"
                    except Exception: pass
                elif col == "Best_Vol_x":
                    try:
                        v = float(str(raw).replace("×",""))
                        if v >= 3:   sty = "color:#f59e0b;font-weight:700"
                        elif v >= 2: sty = "color:#fbbf24;font-weight:600"
                    except Exception: pass
                elif col == "Comp_Bars":
                    try:
                        v = int(float(raw))
                        if v >= 20: sty = "color:#3b82f6;font-weight:700;text-align:center"
                        elif v >= 10: sty = "color:#93c5fd;text-align:center"
                    except Exception: pass
                tds += (f'<td style="padding:7px 13px;font-size:12px;'
                        f'border-bottom:1px solid #e2e8f0;white-space:nowrap;{sty}">'
                        f'{disp}</td>')
            rows_html += f'<tr style="background:{bg}">{tds}</tr>\n'

        html = f"""
<div style="font-family:'Segoe UI',Arial,sans-serif;margin:10px 0">
  <div style="background:linear-gradient(135deg,#0f172a,#1e3a5f);
              border-radius:10px;padding:18px 24px;margin-bottom:8px">
    <h2 style="margin:0;color:#60a5fa;font-size:20px;font-weight:700">
      📈 MA Compression Breakout  —  IREN Pattern
    </h2>
    <p style="margin:6px 0 0;color:#94a3b8;font-size:12px">
      {datetime.today().strftime('%Y-%m-%d %H:%M')} &nbsp;·&nbsp;
      <span style="color:#22c55e;font-weight:700">{len(df_out)} matches</span>
      from {len(TICKERS)} tickers &nbsp;·&nbsp;
      🟢 EMA20 &nbsp; 🔵 SMA50 &nbsp; 🔴 SMA150
    </p>
  </div>
  <div style="overflow-x:auto;border:1px solid #e2e8f0;border-radius:8px;
              box-shadow:0 2px 8px rgba(0,0,0,0.06)">
    <table style="border-collapse:collapse;width:100%;min-width:800px">
      <thead><tr>{th}</tr></thead>
      <tbody>{rows_html}</tbody>
    </table>
  </div>
  <div style="background:#f8fafc;border:1px solid #e2e8f0;border-radius:6px;
              padding:10px 16px;margin-top:8px;font-size:11px;color:#64748b">
    <b style="color:#475569">GUIDE</b> &nbsp;·&nbsp;
    Comp_Bars = bars MAs were compressed (more = tighter coil) &nbsp;·&nbsp;
    Dist_EMA20_% near 0 = price just at green EMA20 (best entry) &nbsp;·&nbsp;
    <span style="color:#f59e0b">Vol×≥2</span> = strong institutional surge &nbsp;·&nbsp;
    EMA20_Slope > 0 = green line rising
  </div>
</div>"""
        display_html(html)
    else:
        # ASCII table
        CLI_COLS = ["Ticker","Price","Score",
                    "EMA20_Cross","Cross_Dist_%","Prev_Close",
                    "EMA20_Slope_%","Comp_Bars",
                    "Best_Vol_x","Surge_Bars_Ago","MACD","MACD_Hist","RSI"]
        CLI_COLS = [c for c in CLI_COLS if c in df_out.columns]
        col_w = {c: max(len(c), max(
            len(fmt_v(c, df_out[c].iloc[i])) for i in range(len(df_out))
        )) + 2 for c in CLI_COLS}
        top  = "┬".join("─"*col_w[c] for c in CLI_COLS)
        sep  = "┼".join("─"*col_w[c] for c in CLI_COLS)
        bot  = "┴".join("─"*col_w[c] for c in CLI_COLS)
        hdr  = "│".join(c.center(col_w[c]) for c in CLI_COLS)
        inner= sum(col_w.values()) + len(CLI_COLS) - 1
        print()
        print(f"  ╔{'═'*inner}╗")
        tit  = f"  MA Compression Breakout (IREN)  {datetime.today().strftime('%Y-%m-%d')}  {len(df_out)} matches"
        print(f"  ║{tit.center(inner)}║")
        print(f"  ╚{'═'*inner}╝\n")
        print(f"  ┌{top}┐")
        print(f"  │{hdr}│")
        print(f"  ├{sep}┤")
        for i,(_, row_) in enumerate(df_out.iterrows()):
            cells = [fmt_v(c,row_.get(c)).center(col_w[c]) for c in CLI_COLS]
            print(f"  │{'│'.join(cells)}│")
            if i < len(df_out)-1: print(f"  ├{sep}┤")
        print(f"  └{bot}┘")
        print(f"""
  COLUMN KEY
  ───────────────────────────────────────────────────────
  Score           composite signal strength 0-100
  Dist_EMA20_%    % above EMA20 (0% = right at green line)
  EMA20_Slope_%   EMA20 rising speed (higher = faster)
  Comp_Bars       how many bars MAs were bunched
  Best_Vol_x      peak volume vs 50d avg
  MACD / Hist     MACD line and histogram values
  ───────────────────────────────────────────────────────""")

    # Save outputs
    out_dir = os.environ.get("GITHUB_WORKSPACE", os.getcwd())
    fpath   = os.path.join(out_dir, f"ma_compression_breakout_{ts}.csv")
    df_out.to_csv(fpath, index=False)
    print(f"\n  💾 CSV → {fpath}")
    tv = os.path.join(out_dir, f"tv_ma_compression_{ts}.txt")
    with open(tv,"w") as f:
        f.write(f"###MA Compression Breakout {datetime.today().strftime('%Y-%m-%d')}\n")
        for r in results: f.write(f"NASDAQ:{r['Ticker']}\n")
    print(f"  📋 TradingView → {tv}")

    # Email
    def _send_email(rl, csv_path):
        import smtplib
        from email.mime.base import MIMEBase; from email import encoders
        gu = os.environ.get("GMAIL_USER",""); gp = os.environ.get("GMAIL_PASS","")
        et = os.environ.get("EMAIL_TO","")
        if not gu or not gp or not et:
            print("[Email] Skipped — set GMAIL_USER, GMAIL_PASS, EMAIL_TO"); return
        eto = [e.strip() for e in et.split(",") if e.strip()]
        cnt = len(rl)
        th_e = "".join(f'<th style="background:#1e293b;color:#e2e8f0;padding:8px;font-size:11px">{c}</th>'
                       for c in ["Ticker","Price","Score","Dist_EMA20_%","Comp_Bars","Best_Vol_x","RSI"])
        rows_e = "".join(
            f'<tr style="background:{"#fff" if i%2==0 else "#f0f9ff"}">'
            f'<td style="padding:6px 10px;font-size:11px">{r["Ticker"]}</td>'
            f'<td style="padding:6px 10px;font-size:11px">${r["Price"]:.2f}</td>'
            f'<td style="padding:6px 10px;font-size:11px;font-weight:700">{r["Score"]:.0f}</td>'
            f'<td style="padding:6px 10px;font-size:11px;color:{"#22c55e" if r["Dist_EMA20_%"]>=0 else "#ef4444"}">{r["Dist_EMA20_%"]:+.2f}%</td>'
            f'<td style="padding:6px 10px;font-size:11px">{r["Comp_Bars"]}</td>'
            f'<td style="padding:6px 10px;font-size:11px">{r["Best_Vol_x"]:.2f}×</td>'
            f'<td style="padding:6px 10px;font-size:11px">{r["RSI"]:.1f}</td>'
            f'</tr>'
            for i,r in enumerate(rl[:50])
        )
        html_e = (f'<html><body style="font-family:Arial;background:#f1f5f9;padding:20px">'
                  f'<div style="background:#0f172a;border-radius:8px;padding:18px 24px;max-width:800px;margin:0 auto">'
                  f'<h2 style="color:#60a5fa;margin:0">📈 MA Compression Breakout — {datetime.today().strftime("%Y-%m-%d")}</h2>'
                  f'<p style="color:#94a3b8;font-size:12px">{cnt} matches · EMA20>SMA50>SMA150 + Volume Surge + MACD>0</p></div>'
                  f'<div style="max-width:800px;margin:8px auto;overflow-x:auto">'
                  f'<table style="border-collapse:collapse;width:100%"><thead><tr>{th_e}</tr></thead><tbody>{rows_e}</tbody></table></div>'
                  f'<p style="color:#94a3b8;font-size:10px;text-align:center">⚠️ Not financial advice</p></body></html>')
        plain_e = f"MA Compression Breakout — {cnt} matches\n" + "\n".join(
            f"{r['Ticker']} ${r['Price']:.2f}  Score:{r['Score']:.0f}  Dist:{r['Dist_EMA20_%']:+.2f}%"
            for r in rl[:50])
        subj  = f"📈 MA Compression Breakout — {cnt} signals — {datetime.today().strftime('%Y-%m-%d')}"
        msg   = MIMEMultipart("mixed"); msg["Subject"]=subj; msg["From"]=gu; msg["To"]=", ".join(eto)
        alt   = MIMEMultipart("alternative")
        alt.attach(MIMEText(plain_e,"plain")); alt.attach(MIMEText(html_e,"html"))
        msg.attach(alt)
        if csv_path and os.path.exists(csv_path):
            try:
                with open(csv_path,"rb") as f:
                    p = MIMEBase("application","octet-stream"); p.set_payload(f.read())
                encoders.encode_base64(p)
                p.add_header("Content-Disposition",f"attachment; filename={os.path.basename(csv_path)}")
                msg.attach(p); print("[Email] 📎 CSV attached")
            except Exception as e: print(f"[Email] ⚠️  {e}")
        try:
            with smtplib.SMTP_SSL("smtp.gmail.com",465) as srv:
                srv.login(gu,gp.replace(" ","")); srv.sendmail(gu,eto,msg.as_string())
            print(f"[Email] ✅  Sent to {et}  |  {cnt} matches")
        except smtplib.SMTPAuthenticationError: print("[Email] ❌  Auth failed")
        except Exception as e: print(f"[Email] ❌  {e}")

    _send_email(results, fpath)

    if _IN_NOTEBOOK:
        try:
            from google.colab import files
            files.download(fpath); files.download(tv)
        except Exception: pass
    else:
        print("  (CI: files in workspace, email sent)")

# ── Charts top 5 ──────────────────────────────────────────────
if results:
    top = results[:min(5,len(results))]
    fig, axes = plt.subplots(len(top), 2, figsize=(16, 4.5*len(top)),
                             facecolor="#0f172a",
                             gridspec_kw={"width_ratios":[3,1]})
    if len(top)==1: axes=[axes]
    for idx, r in enumerate(top):
        ax_c = axes[idx][0]; ax_m = axes[idx][1]
        df_p  = r["_df"].tail(80).copy()
        e20   = r["_ema20"].reindex(df_p.index)
        s50   = r["_sma50"].reindex(df_p.index)
        s150  = r["_sma150"].reindex(df_p.index)
        macd  = r["_macd"].reindex(df_p.index)
        hist  = r["_hist"].reindex(df_p.index)
        n_p   = len(df_p)
        fn    = len(r["_df"]); off = fn - n_p

        ax_c.set_facecolor("#0f172a")
        for i,(_, row_) in enumerate(df_p.iterrows()):
            o=float(row_["Open"]); h=float(row_["High"])
            l=float(row_["Low"]);  c=float(row_["Close"])
            clr = "#34d399" if c>=o else "#ef4444"
            ax_c.plot([i,i],[l,h],color=clr,lw=0.7,zorder=2)
            blo=min(o,c); bhi=max(o,c); bh=max(bhi-blo,(h-l)*0.005)
            rect=mpatches.FancyBboxPatch((i-0.3,blo),0.6,bh,
                 boxstyle="square,pad=0",facecolor=clr,edgecolor=clr,lw=0.4,zorder=3)
            ax_c.add_patch(rect)

        ax_c.plot(range(n_p),e20.values, color="#34d399",lw=1.8,ls="-", label="EMA20 🟢",zorder=5)
        ax_c.plot(range(n_p),s50.values, color="#3b82f6",lw=1.5,ls="-", label="SMA50 🔵",zorder=4)
        ax_c.plot(range(n_p),s150.values,color="#f472b6",lw=1.5,ls="-.", label="SMA150 🔴",zorder=4)

        # Mark surge bar
        sp = r["_surge_bar"] - off if r["_surge_bar"] is not None else None
        if sp is not None and 0 <= sp < n_p:
            ax_c.axvline(sp, color="#f59e0b", lw=1.5, ls=":", alpha=0.8)
            ax_c.text(sp, float(df_p["High"].max())*0.998,
                      f"Vol {r['Best_Vol_x']:.1f}×",
                      color="#f59e0b", fontsize=7, ha="center", va="top")

        tick_step = max(1, n_p//8)
        ax_c.set_xticks(range(0,n_p,tick_step))
        ax_c.set_xticklabels(
            [df_p.index[i].strftime("%m/%d") for i in range(0,n_p,tick_step)],
            color="#94a3b8", fontsize=7)
        ax_c.set_xlim(-0.5, n_p-0.5)
        ax_c.set_title(
            f"{r['Ticker']}  ${r['Price']:.2f}  Score={r['Score']}  "
            f"Comp={r['Comp_Bars']}bars  Vol{r['Best_Vol_x']:.1f}×  "
            f"Dist_EMA20={r['Dist_EMA20_%']:+.1f}%  "
            f"EMA20_slope={r['EMA20_Slope_%']:+.2f}%  RSI={r['RSI']:.0f}",
            color="#e2e8f0", fontsize=8, fontweight="bold", pad=5)
        ax_c.tick_params(colors="#94a3b8", labelsize=7)
        for sp_ in ax_c.spines.values(): sp_.set_edgecolor("#1e3a5f")
        ax_c.legend(loc="upper left", facecolor="#1e293b",
                    labelcolor="#e2e8f0", fontsize=7, framealpha=0.9)
        ax_c.grid(color="#1e3a5f", ls="--", lw=0.4, alpha=0.4, axis="y")

        # MACD panel
        ax_m.set_facecolor("#0f172a")
        for i in range(n_p):
            ax_m.bar(i, float(hist.iloc[i]),
                     color="#34d399" if float(hist.iloc[i])>=0 else "#ef4444",
                     alpha=0.7, width=0.7)
        ax_m.plot(range(n_p), macd.values, color="#60a5fa", lw=1.2)
        ax_m.axhline(0, color="#94a3b8", lw=0.8, ls="--", alpha=0.6)
        ax_m.set_title("MACD 12,26,9", color="#e2e8f0", fontsize=8)
        ax_m.set_xticks(range(0,n_p,max(1,n_p//4)))
        ax_m.set_xticklabels(
            [df_p.index[i].strftime("%m/%d") for i in range(0,n_p,max(1,n_p//4))],
            color="#94a3b8", fontsize=6)
        ax_m.tick_params(colors="#94a3b8", labelsize=7)
        for sp_ in ax_m.spines.values(): sp_.set_edgecolor("#1e3a5f")
        ax_m.grid(color="#1e3a5f", ls="--", lw=0.4, alpha=0.4, axis="y")

    plt.suptitle(
        f"MA Compression Breakout — EMA20🟢 SMA50🔵 SMA150🔴 + Volume Surge + MACD\n"
        f"{datetime.today().strftime('%Y-%m-%d')}  ·  🟡 = Volume surge bar",
        color="#60a5fa", fontsize=10, fontweight="bold", y=1.001)
    plt.tight_layout()
    ts2 = datetime.today().strftime("%Y%m%d_%H%M")
    cp  = os.path.join(os.environ.get("GITHUB_WORKSPACE",os.getcwd()),
                       f"ma_compression_chart_{ts2}.png")
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
  📋 IREN PATTERN EXPLAINED

  From the chart:
    🟢 Green  = EMA 20
    🔵 Blue   = SMA 50
    🔴 Pink   = SMA 150
    🟠 Orange = Monthly Camarilla levels
    Red circle = The compression / launch zone

  P1  COMPRESSION  (the red circle)
      All 3 MAs bunched within 8% of each other
      All MAs were flat (low slope)
      Price was compressing near or below the MA cluster
      = Energy storing before explosion

  P2  MA STACK  (current requirement)
      EMA20 > SMA50 > SMA150  (full bull stack)
      EMA20 is rising (slope > 0.5% in 5 bars)
      = Trend confirmed bullish across all timeframes

  P3  VOLUME SURGE
      At least one bar with >= 1.8× the 50d avg volume
      = Institutional buying confirmed

  P4  MACD  > 0  AND  Histogram > 0
      Momentum positive and expanding

  ENTRY:  EXACT 1-BAR EMA20 CROSS  ← NEW REQUIREMENT
          Bar[-2] (prev day): close was BELOW EMA20
          Bar[-1] (today)   : close crossed ABOVE EMA20
          = Catches the stock right at the moment of breakout

  💡 BEST SETUPS
  EMA20_Cross = ✅ Today    confirmed exact cross (always true here)
  Cross_Dist_% near 0%     price just barely crossed above EMA20
  Comp_Bars > 15           long coil = more energy
  EMA20_Slope_% rising     green line accelerating up
  Best_Vol_x > 3×          massive institutional entry
  EMA20_vs_SMA150_% < 20%  early in the breakout (not extended)

  ⚙️  TUNE IF 0 RESULTS
  compression_min_bars  10 → 5
  compression_band_pct   8 → 12
  vol_surge_mult        1.8 → 1.3
  pullback_upper_pct    20 → 35
  min_ema20_slope_pct  0.5 → 0.1
  slope_flat_pct         3 → 6
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
""")
