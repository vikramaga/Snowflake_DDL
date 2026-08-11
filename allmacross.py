# ============================================================
# NASDAQ — MA Compression Breakout + MACD Cross + RSI > 50
# ============================================================
#
# EXACT PATTERN (from ABCL chart — 3 highlighted circles):
#
#  CIRCLE 1 — PRICE CHART:
#    All MAs (JMA, EMA8, SMA21, SMA50) compressed together
#    while price was sideways/below them. Then ONE green
#    candle breaks above ALL of them simultaneously.
#
#  CIRCLE 2 — MACD PANEL:
#    MACD histogram crossed from negative/zero to POSITIVE
#    AND MACD line crossed above the signal line
#    = Momentum confirmation of the price breakout
#
#  CIRCLE 3 — RSI PANEL:
#    RSI crossed from below 50 to ABOVE 50
#    = Buyers now in control of momentum (neutral → bullish)
#
# DETECTION LOGIC:
#
#  C1  MA COMPRESSION + BREAKOUT
#      In last compression_lookback bars, find tightest bar
#      where JMA/EMA8/SMA21 all within compression_pct% of
#      SMA50 while price was AT or BELOW SMA50 (compressed).
#      Then in last cross_lookback bars, price closes above
#      ALL 4 MAs on volume >= vol_mult × 20d average.
#
#  C2  MACD CROSS — histogram turned positive
#      MACD line crossed above Signal line within
#      last macd_cross_lookback bars (was below, now above)
#      AND current histogram > 0 (confirmed positive)
#
#  C3  RSI CROSSED ABOVE 50
#      RSI crossed from below 50 to above 50 within
#      last rsi_cross_lookback bars
#      AND current RSI > 50 (confirmed)
#
# ALL 3 MUST OCCUR WITHIN THE SAME RECENT WINDOW — confirming
# the price breakout is backed by momentum AND buying conviction.
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
import matplotlib.gridspec as gridspec
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
    "history_days"              : 300,

    # ── MA periods ────────────────────────────────────────────
    "jma_period"                : 13,
    "jma_phase"                 : 40,
    "ema8_period"               : 8,
    "sma21_period"              : 21,
    "sma50_period"              : 50,

    # ── C1: MA compression ────────────────────────────────────
    # Fast MAs must be within this % of SMA50
    "compression_pct"           : 8.0,
    # How far back to search for compression
    "compression_lookback"      : 20,
    # How far back to look for the price breakout above all MAs
    "cross_lookback"            : 10,
    # Min volume on breakout bar vs 20d avg
    "min_vol_mult"              : 1.0,
    "vol_avg_bars"              : 20,
    # How many bars before breakout price must have been below SMA50
    "lookback_below_bars"       : 20,

    # ── C2: MACD cross ────────────────────────────────────────
    "macd_fast"                 : 12,
    "macd_slow"                 : 26,
    "macd_signal"               : 9,
    # Look back this many bars for the MACD cross
    "macd_cross_lookback"       : 10,

    # ── C3: RSI cross above 50 ────────────────────────────────
    "rsi_period"                : 14,
    # Look back this many bars for the RSI 50 cross
    "rsi_cross_lookback"        : 10,

    # ── Filters ───────────────────────────────────────────────
    "min_avg_volume"            : 50_000,
    "min_price"                 : 0.5,

    "batch_size"                : 50,
    "batch_sleep"               : 1.5,
}

# ── Indicators ───────────────────────────────────────────────
def calc_jma(series, period=13, phase=40):
    n      = len(series); vals = series.values.astype(float)
    result = np.full(n, np.nan)
    first  = next((i for i in range(n) if not np.isnan(vals[i])), 0)
    phase_ratio = phase / 100.0 + 1.5
    alpha = 2.0 / (period + 1.0); beta = alpha * phase_ratio
    e0 = e1 = e2 = vals[first]; result[first] = e0
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
    gain  = delta.clip(lower=0); loss = -delta.clip(upper=0)
    ag    = gain.ewm(alpha=1/period, adjust=False).mean()
    al    = loss.ewm(alpha=1/period, adjust=False).mean()
    rs    = ag / al.replace(0, np.nan)
    return 100 - (100 / (1 + rs))

def calc_macd(close, fast=12, slow=26, signal=9):
    ef  = calc_ema(close, fast)
    es  = calc_ema(close, slow)
    ml  = ef - es
    sig = calc_ema(ml, signal)
    return ml, sig, ml - sig   # line, signal, histogram

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

# ── Debug counters ────────────────────────────────────────────
_DBG = {
    "total"         : 0,
    "pass_filter"   : 0,
    "fail_c1_comp"  : 0,
    "fail_c1_break" : 0,
    "fail_c2_macd"  : 0,
    "fail_c3_rsi"   : 0,
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
    if n < CFG["sma50_period"] + 30:    return None
    _DBG["pass_filter"] += 1

    # ── Indicators ───────────────────────────────────────────
    jma_s   = calc_jma(df["Close"], CFG["jma_period"], CFG["jma_phase"])
    ema8_s  = calc_ema(df["Close"], CFG["ema8_period"])
    s21_s   = df["Close"].rolling(CFG["sma21_period"]).mean()
    s50_s   = df["Close"].rolling(CFG["sma50_period"]).mean()
    rsi_s   = calc_rsi(df["Close"], CFG["rsi_period"])
    macd_l, macd_sig, macd_hist = calc_macd(
        df["Close"], CFG["macd_fast"], CFG["macd_slow"], CFG["macd_signal"])

    # Current values
    cur_jma   = float(jma_s.iloc[-1])    if not np.isnan(jma_s.iloc[-1])    else np.nan
    cur_ema8  = float(ema8_s.iloc[-1])   if not np.isnan(ema8_s.iloc[-1])   else np.nan
    cur_s21   = float(s21_s.iloc[-1])    if not np.isnan(s21_s.iloc[-1])    else np.nan
    cur_s50   = float(s50_s.iloc[-1])    if not np.isnan(s50_s.iloc[-1])    else np.nan
    cur_rsi   = float(rsi_s.iloc[-1])    if not np.isnan(rsi_s.iloc[-1])    else 50
    cur_macdl = float(macd_l.iloc[-1])   if not np.isnan(macd_l.iloc[-1])   else 0
    cur_macds = float(macd_sig.iloc[-1]) if not np.isnan(macd_sig.iloc[-1]) else 0
    cur_macdh = float(macd_hist.iloc[-1])if not np.isnan(macd_hist.iloc[-1])else 0

    if any(np.isnan([cur_jma, cur_ema8, cur_s21, cur_s50])): return None

    # ─────────────────────────────────────────────────────────
    # C1a: FIND TIGHTEST MA COMPRESSION
    # Search compression_lookback + cross_lookback bars back
    # for a bar where fast MAs were within compression_pct%
    # of SMA50 AND price was at/below SMA50 (suppressed)
    # ─────────────────────────────────────────────────────────
    cl    = CFG["compression_lookback"]
    xc    = CFG["cross_lookback"]
    cpc   = CFG["compression_pct"] / 100
    best  = float("inf")
    comp_bar = None

    for i in range(max(0, n - cl - xc), n):
        j  = float(jma_s.iloc[i])  if not np.isnan(jma_s.iloc[i])  else np.nan
        e  = float(ema8_s.iloc[i]) if not np.isnan(ema8_s.iloc[i]) else np.nan
        s2 = float(s21_s.iloc[i])  if not np.isnan(s21_s.iloc[i])  else np.nan
        s5 = float(s50_s.iloc[i])  if not np.isnan(s50_s.iloc[i])  else np.nan
        pc = float(df["Close"].iloc[i])
        if any(np.isnan([j,e,s2,s5])) or s5 <= 0: continue
        spread = max(abs(j-s5)/s5, abs(e-s5)/s5, abs(s2-s5)/s5)
        # Price must be at or near/below SMA50 during compression
        price_suppressed = pc <= s5 * 1.03
        if spread <= cpc and price_suppressed and spread < best:
            best = spread; comp_bar = i

    if comp_bar is None or best > cpc:
        _DBG["fail_c1_comp"] += 1; return None

    comp_spread_pct = round(best * 100, 2)

    # ─────────────────────────────────────────────────────────
    # C1b: BREAKOUT — green candle above ALL 4 MAs + volume
    # within last cross_lookback bars
    # Price must have been below SMA50 recently before breakout
    # ─────────────────────────────────────────────────────────
    break_bar = None; break_date = None; break_vmul = None

    for i in range(max(1, n - xc), n):
        pc  = float(df["Close"].iloc[i])
        po  = float(df["Open"].iloc[i])
        vol = float(df["Volume"].iloc[i])
        j   = float(jma_s.iloc[i])  if not np.isnan(jma_s.iloc[i])  else np.nan
        e   = float(ema8_s.iloc[i]) if not np.isnan(ema8_s.iloc[i]) else np.nan
        s2  = float(s21_s.iloc[i])  if not np.isnan(s21_s.iloc[i])  else np.nan
        s5  = float(s50_s.iloc[i])  if not np.isnan(s50_s.iloc[i])  else np.nan
        if any(np.isnan([j,e,s2,s5])): continue

        # Green candle
        if pc <= po: continue
        # Above all 4 MAs
        if not (pc > j and pc > e and pc > s2 and pc > s5): continue
        # Volume >= min_vol_mult × avg
        vmul = vol / avg_vol if avg_vol > 0 else 0
        if vmul < CFG["min_vol_mult"]: continue
        # Was below SMA50 recently
        lb = CFG["lookback_below_bars"]
        was_below = any(
            float(df["Close"].iloc[k]) < float(s50_s.iloc[k])
            for k in range(max(0, i-lb), i)
            if not np.isnan(s50_s.iloc[k])
        )
        if not was_below: continue

        if break_bar is None or i > break_bar:
            break_bar = i; break_date = df.index[i]; break_vmul = vmul

    if break_bar is None:
        _DBG["fail_c1_break"] += 1; return None

    # ─────────────────────────────────────────────────────────
    # C2: MACD LINE CROSSED ABOVE SIGNAL within macd_cross_lookback
    # Bar[i-1]: MACD line <= signal line  (was below/at)
    # Bar[i]  : MACD line > signal line   (now above)
    # Current histogram must be > 0 (confirmed positive)
    # ─────────────────────────────────────────────────────────
    mc = CFG["macd_cross_lookback"]
    macd_cross_bar = None

    for i in range(max(1, n - mc), n):
        ml_cur  = float(macd_l.iloc[i])   if not np.isnan(macd_l.iloc[i])   else np.nan
        ms_cur  = float(macd_sig.iloc[i]) if not np.isnan(macd_sig.iloc[i]) else np.nan
        ml_prev = float(macd_l.iloc[i-1]) if not np.isnan(macd_l.iloc[i-1]) else np.nan
        ms_prev = float(macd_sig.iloc[i-1])if not np.isnan(macd_sig.iloc[i-1])else np.nan
        if any(np.isnan([ml_cur, ms_cur, ml_prev, ms_prev])): continue
        # Cross: was below signal, now above
        if ml_prev <= ms_prev and ml_cur > ms_cur:
            if macd_cross_bar is None or i > macd_cross_bar:
                macd_cross_bar = i

    if macd_cross_bar is None or cur_macdh <= 0:
        _DBG["fail_c2_macd"] += 1; return None

    # ─────────────────────────────────────────────────────────
    # C3: RSI CROSSED ABOVE 50 within rsi_cross_lookback bars
    # Bar[i-1]: RSI <= 50   (was at/below midline)
    # Bar[i]  : RSI > 50    (now above midline)
    # Current RSI must be > 50 (confirmed)
    # ─────────────────────────────────────────────────────────
    rc = CFG["rsi_cross_lookback"]
    rsi_cross_bar = None

    for i in range(max(1, n - rc), n):
        r_cur  = float(rsi_s.iloc[i])   if not np.isnan(rsi_s.iloc[i])   else np.nan
        r_prev = float(rsi_s.iloc[i-1]) if not np.isnan(rsi_s.iloc[i-1]) else np.nan
        if np.isnan(r_cur) or np.isnan(r_prev): continue
        if r_prev <= 50 and r_cur > 50:
            if rsi_cross_bar is None or i > rsi_cross_bar:
                rsi_cross_bar = i

    if rsi_cross_bar is None or cur_rsi <= 50:
        _DBG["fail_c3_rsi"] += 1; return None

    _DBG["pass_all"] += 1

    # ── Metrics ───────────────────────────────────────────────
    bars_since_break     = n - 1 - break_bar
    bars_since_macd_cross= n - 1 - macd_cross_bar
    bars_since_rsi_cross = n - 1 - rsi_cross_bar

    # All 3 crosses must be within sync_window bars of each other
    # (they should all happen together — the triple confirmation)
    latest_signal  = max(break_bar, macd_cross_bar, rsi_cross_bar)
    earliest_signal= min(break_bar, macd_cross_bar, rsi_cross_bar)
    signal_spread  = latest_signal - earliest_signal  # bars apart

    dist_s50_pct  = (price - cur_s50) / cur_s50 * 100 if cur_s50 > 0 else 0
    dist_jma_pct  = (price - cur_jma) / cur_jma * 100 if cur_jma > 0 else 0

    # Volume on break bar vs prev bar
    vol_break  = float(df["Volume"].iloc[break_bar])
    vol_prev   = float(df["Volume"].iloc[break_bar-1])
    vol_vs_prev= vol_break / vol_prev if vol_prev > 0 else 0

    # Score (0-100)
    score = 0
    # Compression tightness (0-25)
    score += max(0, 25 - int(comp_spread_pct * 2.5))
    # Break freshness (0-20)
    score += max(0, 20 - bars_since_break * 4)
    # All 3 signals in sync (0-20): smaller spread = better
    score += max(0, 20 - signal_spread * 2)
    # Volume on break (0-15)
    score += min(15, int(break_vmul * 5))
    # MACD histogram strength (0-10)
    score += min(10, max(0, int(abs(cur_macdh) * 20)))
    # RSI level (0-10)
    score += min(10, max(0, int((cur_rsi - 50) * 0.5)))
    score = min(100, max(0, score))

    return {
        "Ticker"              : sym,
        "Price"               : round(price, 2),
        "Score"               : score,
        # C1: compression + breakout
        "Comp_Spread_%"       : comp_spread_pct,
        "Comp_Bar_Ago"        : n - 1 - comp_bar,
        "Break_Date"          : break_date.strftime("%Y-%m-%d"),
        "Bars_Since_Break"    : bars_since_break,
        "Break_Vol_x"         : round(break_vmul, 2),
        "Vol_vs_Prev_x"       : round(vol_vs_prev, 2),
        # C2: MACD
        "MACD_Cross_Ago"      : bars_since_macd_cross,
        "MACD_Line"           : round(cur_macdl, 4),
        "MACD_Hist"           : round(cur_macdh, 4),
        # C3: RSI
        "RSI_Cross_Ago"       : bars_since_rsi_cross,
        "RSI"                 : round(cur_rsi, 1),
        # Sync
        "Signal_Spread_Bars"  : signal_spread,
        # MAs now
        "JMA"                 : round(cur_jma, 2),
        "EMA8"                : round(cur_ema8, 2),
        "SMA21"               : round(cur_s21, 2),
        "SMA50"               : round(cur_s50, 2),
        "Dist_SMA50_%"        : round(dist_s50_pct, 2),
        "Dist_JMA_%"          : round(dist_jma_pct, 2),
        "Avg_Vol_20d"         : int(avg_vol),
        # internals
        "_df"       : df,
        "_jma"      : jma_s,
        "_ema8"     : ema8_s,
        "_sma21"    : s21_s,
        "_sma50"    : s50_s,
        "_rsi"      : rsi_s,
        "_macd_l"   : macd_l,
        "_macd_sig" : macd_sig,
        "_macd_hist": macd_hist,
        "_break_bar": break_bar,
        "_comp_bar" : comp_bar,
        "_macd_cross": macd_cross_bar,
        "_rsi_cross" : rsi_cross_bar,
    }

# ── Live print ────────────────────────────────────────────────
LIVE_COLS = ["Ticker","Price","Score",
             "Comp_Spread_%","Bars_Since_Break","Break_Vol_x",
             "MACD_Cross_Ago","MACD_Hist",
             "RSI_Cross_Ago","RSI","Signal_Spread_Bars"]
_CW = {"Ticker":8,"Price":10,"Score":7,
       "Comp_Spread_%":14,"Bars_Since_Break":17,"Break_Vol_x":13,
       "MACD_Cross_Ago":15,"MACD_Hist":12,
       "RSI_Cross_Ago":14,"RSI":6,"Signal_Spread_Bars":19}
_CF = {"Price":"${:.2f}","Score":"{:.0f}",
       "Comp_Spread_%":"{:.2f}%","Bars_Since_Break":"{:.0f}d",
       "Break_Vol_x":"{:.2f}×","MACD_Cross_Ago":"{:.0f}d",
       "MACD_Hist":"{:.4f}","RSI_Cross_Ago":"{:.0f}d",
       "RSI":"{:.1f}","Signal_Spread_Bars":"{:.0f}bars"}
_hdr_done = False

def _live_header():
    global _hdr_done
    if _hdr_done: return
    sep = "━"*195
    print(f"\n{sep}")
    print("  📊  LIVE — MA Compression Breakout + MACD Cross + RSI > 50")
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
chk = download(["ABCL","NVDA","AMD"], 300)
if not chk: print("❌  No data.")
else:
    for s, d in chk.items():
        p    = float(d["Close"].iloc[-1])
        rsi  = float(calc_rsi(d["Close"]).iloc[-1])
        ml,ms,mh = calc_macd(d["Close"])
        mhv  = float(mh.iloc[-1])
        s50  = float(d["Close"].rolling(50).mean().iloc[-1])
        print(f"  ✅ {s}: ${p:.2f}  SMA50=${s50:.2f}  RSI={rsi:.1f}  MACD_Hist={mhv:.4f}  {d.index[-1].date()}")
print()

# ── Diagnostic ────────────────────────────────────────────────
print("━"*65)
print("  STEP 2  DIAGNOSTIC")
print("━"*65+"\n")
DIAG = ["ABCL","NVDA","AAPL","AMD","PLTR","META","CRWD","MU","SMCI","MXL"]
diag_data = download(DIAG, CFG["history_days"])
print(f"  Downloaded {len(diag_data)}/{len(DIAG)}\n")
print(f"  {'SYM':<6} {'PRICE':>8}  {'COMP%':>7}  {'MACD_H':>8}  {'RSI':>6}  RESULT")
print("  "+"─"*50)

for sym in DIAG:
    if sym not in diag_data:
        print(f"  {sym:<6} — no data"); continue
    try:
        df_d = diag_data[sym]
        p    = float(df_d["Close"].iloc[-1])
        r    = detect_pattern(sym, df_d)
        ml,ms,mh = calc_macd(df_d["Close"])
        mhv  = float(mh.iloc[-1]) if not np.isnan(mh.iloc[-1]) else 0
        rsiv = float(calc_rsi(df_d["Close"]).iloc[-1])
        if r:
            print(f"  {sym:<6} ${p:>7.2f}  "
                  f"{r['Comp_Spread_%']:>6.2f}%  "
                  f"{mhv:>+8.4f}  {rsiv:>5.1f}  "
                  f"✅ Score={r['Score']} Spread={r['Signal_Spread_Bars']}bars")
        else:
            print(f"  {sym:<6} ${p:>7.2f}  {'—':>7}  {mhv:>+8.4f}  {rsiv:>5.1f}  ❌")
    except Exception as e:
        print(f"  {sym:<6} error: {e}")

print(f"""
  Pattern (all 3 must fire together):
    C1  Compression + Breakout:
        Fast MAs within {CFG['compression_pct']}% of SMA50 (suppressed price)
        Then green candle above ALL 4 MAs, vol>={CFG['min_vol_mult']}×avg
    C2  MACD: line crossed above signal within {CFG['macd_cross_lookback']} bars
        AND current histogram > 0
    C3  RSI crossed above 50 within {CFG['rsi_cross_lookback']} bars
        AND current RSI > 50

  Tune if mostly ❌:
    compression_pct         8 → 12
    cross_lookback         10 → 15
    macd_cross_lookback    10 → 15
    rsi_cross_lookback     10 → 15
    min_vol_mult           1.0 → 0.5
    lookback_below_bars     20 → 30
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
            r  = requests.get(url,headers=hdrs,timeout=20); r.raise_for_status()
            df = pd.read_csv(io.StringIO(r.text),sep="|")
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
            headers={**hdrs,"Referer":"https://www.nasdaq.com/"},timeout=25)
        r.raise_for_status()
        rows = r.json()["data"]["rows"]
        t = {row["symbol"].strip() for row in rows
             if row.get("symbol","").strip().isalpha()
             and 1<=len(row["symbol"].strip())<=5}
        b = len(pool); pool |= t
        print(f"  ✅ {'NASDAQ API':<18}: +{len(pool)-b:>4} → {len(pool)}")
    except Exception as e: print(f"  ⚠️  NASDAQ API: {e}")
    static = {
        "AAPL","MSFT","NVDA","AMZN","META","GOOGL","TSLA","AVGO","COST","NFLX",
        "AMD","CSCO","ADBE","QCOM","TXN","AMAT","MU","KLAC","LRCX","MRVL",
        "PANW","CRWD","SNPS","CDNS","TEAM","WDAY","PLTR","DDOG","SNOW","MDB",
        "VRTX","ISRG","LULU","FTNT","SBUX","TMUS","RBRK","NET","AXON","ANET",
        "CAVA","VRT","ELF","GRMN","ON","ENPH","ROST","HOOD","COIN","UPST",
        "SMCI","ABCL","MXL","ACLS","IRTC","MNDY","HUBS","GTLB","GLBE","CELH",
        "DKNG","SMAR","BILL","APPN","ALRM","FIVE","BOOT","INMD","LGIH","UFPT",
        "PRCT","BCAB","IONQ","RGTI","QUBT","ASTS","RKLB","FSLR","PYPL","ROKU",
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
  Total processed            : {_DBG['total']}
  Passed vol/price filter    : {_DBG['pass_filter']}
  ❌ Failed C1 compression    : {_DBG['fail_c1_comp']}
  ❌ Failed C1 breakout       : {_DBG['fail_c1_break']}
  ❌ Failed C2 MACD cross     : {_DBG['fail_c2_macd']}
  ❌ Failed C3 RSI > 50       : {_DBG['fail_c3_rsi']}
  ✅ Passed all               : {_DBG['pass_all']}
""")

if not results:
    print("  Relax the condition with most failures:")
    print("   C1 comp:  compression_pct     8 → 12")
    print("   C1 break: cross_lookback      10 → 15")
    print("             min_vol_mult       1.0 → 0.5")
    print("   C2 MACD:  macd_cross_lookback 10 → 15")
    print("   C3 RSI:   rsi_cross_lookback  10 → 15")

results.sort(key=lambda x: (x["Signal_Spread_Bars"], -x["Score"]))

# ── Build df_out ──────────────────────────────────────────────
ts      = datetime.today().strftime("%Y%m%d_%H%M")
out_dir = os.environ.get("GITHUB_WORKSPACE", os.getcwd())

COLS = [
    "Ticker","Price","Score",
    "Comp_Spread_%","Comp_Bar_Ago","Break_Date","Bars_Since_Break",
    "Break_Vol_x","Vol_vs_Prev_x",
    "MACD_Cross_Ago","MACD_Line","MACD_Hist",
    "RSI_Cross_Ago","RSI",
    "Signal_Spread_Bars",
    "JMA","EMA8","SMA21","SMA50",
    "Dist_SMA50_%","Dist_JMA_%","Avg_Vol_20d",
]
df_out = pd.DataFrame([{k:v for k,v in r.items() if not k.startswith("_")}
                        for r in results]) if results else pd.DataFrame(columns=COLS)
if not df_out.empty:
    df_out = df_out[[c for c in COLS if c in df_out.columns]]
    df_out.reset_index(drop=True, inplace=True)

FMT = {
    "Price"           : lambda v: f"${v:.2f}",
    "Score"           : lambda v: f"{v:.0f}",
    "Comp_Spread_%"   : lambda v: f"{v:.2f}%",
    "Comp_Bar_Ago"    : lambda v: f"{int(v)}d",
    "Bars_Since_Break": lambda v: f"{int(v)}d",
    "Break_Vol_x"     : lambda v: f"{v:.2f}×",
    "Vol_vs_Prev_x"   : lambda v: f"{v:.2f}×",
    "MACD_Cross_Ago"  : lambda v: f"{int(v)}d",
    "MACD_Line"       : lambda v: f"{v:.4f}",
    "MACD_Hist"       : lambda v: f"{v:.4f}",
    "RSI_Cross_Ago"   : lambda v: f"{int(v)}d",
    "RSI"             : lambda v: f"{v:.1f}",
    "Signal_Spread_Bars": lambda v: f"{int(v)}bars",
    "JMA"             : lambda v: f"${v:.2f}",
    "EMA8"            : lambda v: f"${v:.2f}",
    "SMA21"           : lambda v: f"${v:.2f}",
    "SMA50"           : lambda v: f"${v:.2f}",
    "Dist_SMA50_%"    : lambda v: f"{v:+.2f}%",
    "Dist_JMA_%"      : lambda v: f"{v:+.2f}%",
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
            "Comp_Spread_%","Bars_Since_Break","Break_Vol_x",
            "MACD_Cross_Ago","MACD_Hist",
            "RSI_Cross_Ago","RSI","Signal_Spread_Bars"]
    DISP = [c for c in DISP if c in df_out.columns]
    gc = "#22c55e"
    th = "".join(
        f'<th style="background:#0f172a;color:#e2e8f0;padding:9px 12px;'
        f'font-size:11px;font-weight:700;border-bottom:2px solid {gc};white-space:nowrap">'
        f'{c}</th>' for c in DISP)
    rows_html = ""
    for i, r in enumerate(results):
        bg = "#fff" if i%2==0 else "#f0f9ff"
        tds = ""
        for col in DISP:
            raw=r.get(col); disp=fmt_v(col,raw); sty=""
            if col=="Score":
                try:
                    v=float(raw); g=int(min(220,80+v*1.4))
                    sty=f"background:rgb(20,{g},60);color:#fff;font-weight:700;text-align:center"
                except: pass
            elif col=="Signal_Spread_Bars":
                try:
                    v=int(str(raw).replace("bars",""))
                    sty="color:#22c55e;font-weight:700;text-align:center" if v<=2 else "text-align:center"
                except: pass
            elif col=="MACD_Hist":
                try:
                    v=float(raw)
                    sty=f"color:{'#22c55e' if v>0 else '#ef4444'};font-weight:700"
                except: pass
            elif col in ("MACD_Cross_Ago","RSI_Cross_Ago","Bars_Since_Break"):
                try:
                    v=int(str(raw).replace("d",""))
                    sty="color:#22c55e;font-weight:700;text-align:center" if v==0 else "text-align:center"
                except: pass
            tds+=f'<td style="padding:7px 12px;font-size:12px;border-bottom:1px solid #e2e8f0;white-space:nowrap;{sty}">{disp}</td>'
        rows_html+=f'<tr style="background:{bg}">{tds}</tr>\n'

    ticker_csv_str = ",".join(r["Ticker"] for r in results)
    display_html(f"""
<div style="background:linear-gradient(135deg,#0f172a,#1e3a5f);border-radius:10px;
            padding:18px 24px;margin-bottom:8px">
  <h2 style="margin:0;color:#60a5fa;font-size:20px;font-weight:700">
    📈 MA Compression Breakout + MACD Cross + RSI &gt; 50
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
  <b>GUIDE</b> &nbsp;·&nbsp;
  Signal_Spread_Bars = how many bars apart the 3 signals were (0-2 = perfectly synchronized) &nbsp;·&nbsp;
  MACD_Cross_Ago = bars since MACD line crossed above signal &nbsp;·&nbsp;
  RSI_Cross_Ago = bars since RSI crossed above 50 &nbsp;·&nbsp;
  Comp_Spread_% = MA compression tightness (lower = tighter coil)
</div>""")

elif results:
    CLI = ["Ticker","Price","Score","Comp_Spread_%",
           "Bars_Since_Break","MACD_Cross_Ago","RSI","Signal_Spread_Bars"]
    col_w = {c: max(len(c), max(len(fmt_v(c,r.get(c))) for r in results))+2 for c in CLI}
    top="┬".join("─"*col_w[c] for c in CLI)
    sep="┼".join("─"*col_w[c] for c in CLI)
    bot="┴".join("─"*col_w[c] for c in CLI)
    hdr="│".join(c.center(col_w[c]) for c in CLI)
    inner=sum(col_w.values())+len(CLI)-1
    print(f"\n  ╔{'═'*inner}╗")
    print(f"  ║{'  MA Compression + MACD Cross + RSI>50   '+datetime.today().strftime('%Y-%m-%d')+'   '+str(len(df_out))+' matches'.center(inner)}║")
    print(f"  ╚{'═'*inner}╝\n")
    print(f"  ┌{top}┐\n  │{hdr}│\n  ├{sep}┤")
    for i,r in enumerate(results):
        cells=[fmt_v(c,r.get(c)).center(col_w[c]) for c in CLI]
        print(f"  │{'│'.join(cells)}│")
        if i<len(results)-1: print(f"  ├{sep}┤")
    print(f"  └{bot}┘")

# ── Save CSV + TV ──────────────────────────────────────────────
fpath = os.path.join(out_dir, f"ma_comp_macd_rsi_{ts}.csv")
df_out.to_csv(fpath, index=False)
print(f"\n  💾 CSV → {fpath}")
tv = os.path.join(out_dir, f"tv_ma_comp_macd_rsi_{ts}.txt")
with open(tv,"w") as f:
    f.write(f"###MA Comp MACD RSI {datetime.today().strftime('%Y-%m-%d')}\n")
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
        ticker_csv = ",".join(r.get("Ticker","") for r in rl) if rl else "—"
        print(f"[Email] Sending to {et}  ({cnt} results)...")

        ticker_html = f"""
<div style="margin:14px 0;padding:14px 16px;background:#0f172a;
            border-radius:8px;border-left:4px solid #22c55e;">
  <p style="margin:0 0 6px;color:#94a3b8;font-size:11px;font-weight:600;
             letter-spacing:.05em;text-transform:uppercase">
    📋 Stock List — Copy &amp; paste into TradingView / Excel
  </p>
  <p style="margin:0;color:#22c55e;font-size:13px;font-weight:700;
             font-family:'Courier New',monospace;word-break:break-all">
    {ticker_csv}
  </p>
</div>"""

        th_e = "".join(
            f'<th style="background:#1e293b;color:#e2e8f0;padding:8px 11px;'
            f'font-size:11px;font-weight:700;border-bottom:2px solid #3b82f6;'
            f'white-space:nowrap">{c}</th>'
            for c in ["Ticker","Price","Score","Comp_%","Break_Ago",
                      "MACD_Ago","MACD_Hist","RSI_Ago","RSI","Sync"]
        )
        rows_e = ""
        for i,r in enumerate(rl[:50]):
            bg="#fff" if i%2==0 else "#f0f9ff"
            mh=float(r.get("MACD_Hist",0))
            ss=int(r.get("Signal_Spread_Bars",99))
            rows_e += (
                f'<tr style="background:{bg}">'
                f'<td style="padding:6px 11px;font-size:12px;font-weight:700;color:#22c55e">{r.get("Ticker","—")}</td>'
                f'<td style="padding:6px 11px;font-size:12px">${float(r.get("Price",0)):.2f}</td>'
                f'<td style="padding:6px 11px;font-size:12px;font-weight:700;background:#166534;color:#fff;text-align:center">{float(r.get("Score",0)):.0f}</td>'
                f'<td style="padding:6px 11px;font-size:12px">{float(r.get("Comp_Spread_%",0)):.2f}%</td>'
                f'<td style="padding:6px 11px;font-size:12px;text-align:center">{int(r.get("Bars_Since_Break",0))}d</td>'
                f'<td style="padding:6px 11px;font-size:12px;text-align:center">{int(r.get("MACD_Cross_Ago",0))}d</td>'
                f'<td style="padding:6px 11px;font-size:12px;color:{"#22c55e" if mh>0 else "#ef4444"};font-weight:700">{mh:.4f}</td>'
                f'<td style="padding:6px 11px;font-size:12px;text-align:center">{int(r.get("RSI_Cross_Ago",0))}d</td>'
                f'<td style="padding:6px 11px;font-size:12px">{float(r.get("RSI",0)):.1f}</td>'
                f'<td style="padding:6px 11px;font-size:12px;text-align:center;color:{"#22c55e" if ss<=2 else "#94a3b8"};font-weight:{"700" if ss<=2 else "400"}">{ss}b</td>'
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
      📊 MA Compression Breakout + MACD Cross + RSI &gt; 50
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
      📎 CSV and TradingView watchlist attached &nbsp;·&nbsp;
      <b>Sync</b> = bars apart the 3 signals were (0-2b = perfectly synced) &nbsp;·&nbsp;
      <b>Comp_%</b> = MA compression spread at tightest bar &nbsp;·&nbsp;
      <b>MACD_Hist</b> positive = momentum confirmed
    </p>
  </td></tr>
  <tr><td style="background:#f8fafc;padding:12px 28px;border-top:1px solid #e2e8f0;text-align:center">
    <p style="margin:0;color:#94a3b8;font-size:10px">
      ⚠️ Not financial advice &nbsp;·&nbsp; Auto-generated by GitHub Actions
    </p>
  </td></tr>
</table></td></tr></table></body></html>"""

        plain_e="\n".join([
            f"MA Compression + MACD Cross + RSI>50 — {datetime.today().strftime('%Y-%m-%d')}",
            f"{cnt} matches",
            "",
            f"STOCKS: {ticker_csv}",
            "",
            "="*65,
        ]+([
            f"{r.get('Ticker','—'):<7} ${float(r.get('Price',0)):.2f}  "
            f"Score:{float(r.get('Score',0)):.0f}  "
            f"Comp:{float(r.get('Comp_Spread_%',0)):.2f}%  "
            f"MACD:{int(r.get('MACD_Cross_Ago',0))}d ago  "
            f"RSI:{float(r.get('RSI',0)):.1f}  "
            f"Sync:{int(r.get('Signal_Spread_Bars',0))}bars"
            for r in rl[:50]
        ] if rl else ["No matches today"])+["\n📎 CSV + TradingView attached."])

        subj=(f"📊 MA Comp+MACD+RSI — {cnt} signal{'s' if cnt!=1 else ''}"
              f" — {datetime.today().strftime('%Y-%m-%d')}")

        msg=MIMEMultipart("mixed")
        msg["Subject"]=subj; msg["From"]=gu; msg["To"]=", ".join(eto)
        alt=MIMEMultipart("alternative")
        alt.attach(MIMEText(plain_e,"plain")); alt.attach(MIMEText(html_e,"html"))
        msg.attach(alt)

    except Exception as e:
        print(f"[Email] ❌  Body failed: {type(e).__name__}: {e}"); return

    for att in [csv_path, tv]:
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

# ── Charts for top 5 — 3 panels per stock ─────────────────────
if results:
    top = results[:min(5,len(results))]
    n_stocks = len(top)
    fig = plt.figure(figsize=(15, 9*n_stocks), facecolor="#0f172a")
    gs  = gridspec.GridSpec(n_stocks*3, 1,
                             height_ratios=[4,1.5,1]*n_stocks,
                             hspace=0.08)

    for idx, r in enumerate(top):
        ax_p  = fig.add_subplot(gs[idx*3])
        ax_m  = fig.add_subplot(gs[idx*3+1])
        ax_r  = fig.add_subplot(gs[idx*3+2])

        df_p   = r["_df"].tail(80).copy()
        n_p    = len(df_p)
        fn     = len(r["_df"]); off = fn - n_p

        jma    = r["_jma"].reindex(df_p.index)
        ema8   = r["_ema8"].reindex(df_p.index)
        s21    = r["_sma21"].reindex(df_p.index)
        s50    = r["_sma50"].reindex(df_p.index)
        rsi_v  = r["_rsi"].reindex(df_p.index)
        mh_v   = r["_macd_hist"].reindex(df_p.index)
        ml_v   = r["_macd_l"].reindex(df_p.index)
        ms_v   = r["_macd_sig"].reindex(df_p.index)

        for ax in [ax_p, ax_m, ax_r]:
            ax.set_facecolor("#0f172a")
            ax.tick_params(colors="#94a3b8", labelsize=7)
            for sp_ in ax.spines.values(): sp_.set_edgecolor("#1e3a5f")
            ax.grid(color="#1e3a5f", ls="--", lw=0.4, alpha=0.4, axis="y")

        # ── Price panel ───────────────────────────────────────
        for i,(_, row_) in enumerate(df_p.iterrows()):
            o=float(row_["Open"]); h=float(row_["High"])
            l=float(row_["Low"]);  c=float(row_["Close"])
            clr="#34d399" if c>=o else "#ef4444"
            ax_p.plot([i,i],[l,h],color=clr,lw=0.7,zorder=2)
            blo=min(o,c); bhi=max(o,c); bh=max(bhi-blo,(h-l)*0.005)
            rect=mpatches.FancyBboxPatch((i-0.3,blo),0.6,bh,
                 boxstyle="square,pad=0",facecolor=clr,edgecolor=clr,lw=0.3,zorder=3)
            ax_p.add_patch(rect)

        ax_p.plot(range(n_p), jma.values,  color="#22d3ee", lw=1.8, label="JMA", zorder=6)
        ax_p.plot(range(n_p), ema8.values, color="#34d399", lw=1.4, ls="--", label="EMA8", zorder=5)
        ax_p.plot(range(n_p), s21.values,  color="#f59e0b", lw=1.3, ls="-.", label="SMA21", zorder=5)
        ax_p.plot(range(n_p), s50.values,  color="#3b82f6", lw=1.8, label="SMA50", zorder=5)

        # Shade compression zone
        cb = r["_comp_bar"] - off
        if 0 <= cb < n_p:
            ax_p.axvspan(max(0,cb-3), min(cb+3,n_p-1),
                         alpha=0.12, color="#a78bfa", zorder=1, label="Compression")

        # Mark breakout bar
        bb = r["_break_bar"] - off
        if 0 <= bb < n_p:
            ax_p.axvline(bb, color="#22c55e", lw=2.0, ls="--", alpha=0.9)
            ax_p.scatter([bb],[float(df_p["Close"].iloc[bb])],
                         color="#22c55e",s=200,zorder=9,marker="^",
                         label=f"Breakout {r['Break_Date']}")

        tick_step = max(1, n_p//8)
        for ax in [ax_p, ax_m, ax_r]:
            ax.set_xticks(range(0,n_p,tick_step))
            ax.set_xticklabels(
                [df_p.index[i].strftime("%m/%d") for i in range(0,n_p,tick_step)],
                color="#94a3b8", fontsize=7)
            ax.set_xlim(-0.5, n_p-0.5)

        ax_p.set_title(
            f"{r['Ticker']}  ${r['Price']:.2f}  |  Score {r['Score']}/100  |  "
            f"Compression {r['Comp_Spread_%']:.2f}%  |  "
            f"Break {r['Break_Date']} ({r['Bars_Since_Break']}d)  Vol {r['Break_Vol_x']:.1f}×  |  "
            f"MACD cross {r['MACD_Cross_Ago']}d ago  RSI cross {r['RSI_Cross_Ago']}d ago  "
            f"Sync {r['Signal_Spread_Bars']}bars",
            color="#e2e8f0", fontsize=8, fontweight="bold", pad=5)
        ax_p.legend(loc="upper left", facecolor="#1e293b",
                    labelcolor="#e2e8f0", fontsize=7, framealpha=0.9, ncol=3)

        # ── MACD panel ───────────────────────────────────────
        for i in range(n_p):
            v=float(mh_v.iloc[i]) if not np.isnan(mh_v.iloc[i]) else 0
            ax_m.bar(i, v, color="#34d399" if v>=0 else "#ef4444",
                     alpha=0.8, width=0.8, zorder=2)
        ax_m.plot(range(n_p), ml_v.values, color="#3b82f6", lw=1.2, label="MACD")
        ax_m.plot(range(n_p), ms_v.values, color="#f59e0b", lw=1.0, ls="--", label="Signal")
        ax_m.axhline(0, color="#94a3b8", lw=0.8, ls="--")
        # Mark MACD cross bar
        mc = r["_macd_cross"] - off
        if 0 <= mc < n_p:
            ax_m.axvline(mc, color="#fbbf24", lw=1.5, ls=":", alpha=0.8)
        ax_m.set_ylabel("MACD", color="#94a3b8", fontsize=7)
        ax_m.legend(loc="upper left", facecolor="#1e293b",
                    labelcolor="#e2e8f0", fontsize=6, framealpha=0.9)

        # ── RSI panel ────────────────────────────────────────
        ax_r.plot(range(n_p), rsi_v.values, color="#c084fc", lw=1.5, label="RSI")
        ax_r.axhline(50, color="#94a3b8", lw=1.0, ls="--", alpha=0.8)
        ax_r.axhline(70, color="#ef4444", lw=0.7, ls=":", alpha=0.6)
        ax_r.axhline(30, color="#22c55e", lw=0.7, ls=":", alpha=0.6)
        ax_r.fill_between(range(n_p), rsi_v.values, 50,
                          where=(rsi_v.values>50), alpha=0.15, color="#22c55e")
        ax_r.fill_between(range(n_p), rsi_v.values, 50,
                          where=(rsi_v.values<=50), alpha=0.15, color="#ef4444")
        # Mark RSI 50 cross bar
        rc = r["_rsi_cross"] - off
        if 0 <= rc < n_p:
            ax_r.axvline(rc, color="#c084fc", lw=1.5, ls=":", alpha=0.8)
        ax_r.set_ylim(10, 90)
        ax_r.set_ylabel("RSI", color="#94a3b8", fontsize=7)
        ax_r.legend(loc="upper left", facecolor="#1e293b",
                    labelcolor="#e2e8f0", fontsize=6, framealpha=0.9)

    plt.suptitle(
        f"MA Compression Breakout + MACD Cross + RSI > 50  ·  {datetime.today().strftime('%Y-%m-%d')}\n"
        f"🔵 JMA  🟢 EMA8  🟡 SMA21  🔵 SMA50  🟣=compression  ▲=breakout  |dashed=cross bars",
        color="#60a5fa", fontsize=10, fontweight="bold", y=1.001)
    plt.tight_layout()
    cp = os.path.join(out_dir, f"ma_comp_macd_rsi_chart_{ts}.png")
    plt.savefig(cp, dpi=150, bbox_inches="tight", facecolor="#0f172a")
    if _IN_NOTEBOOK: plt.show()
    else: plt.close()
    print(f"  📊 Chart → {cp}")

print("""
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  📋 PATTERN EXPLAINED (from ABCL chart)

  CIRCLE 1 — PRICE CHART (MA Compression + Breakout)
    JMA / EMA8 / SMA21 / SMA50 all bunch together near
    the same price while the stock is compressed/sideways.
    Then ONE green candle breaks above ALL 4 MAs on volume.

  CIRCLE 2 — MACD PANEL (momentum confirmation)
    MACD line crossed above the signal line → histogram
    flipped from negative to positive. This confirms the
    price breakout is backed by accelerating momentum.

  CIRCLE 3 — RSI PANEL (trend shift confirmation)
    RSI crossed from below 50 to above 50 (midline).
    This means buyers now dominate sellers — the trend
    has shifted from bearish/neutral to bullish.

  WHY ALL 3 TOGETHER:
    Price break alone = could be a false breakout
    MACD cross alone = lagging, no price confirmation
    RSI > 50 alone = could be in overbought decline
    ALL 3 TOGETHER = triple confirmation that buyers
    have taken full control at the same moment

  💡 BEST SETUPS
  Signal_Spread_Bars = 0-2   all 3 fired on same bar
  Comp_Spread_% < 3%         extremely tight coil
  Bars_Since_Break = 0        fresh breakout today
  Break_Vol_x >= 2×          strong volume confirmation
  RSI 50-65                  momentum healthy not extreme

  ⚙️  TUNE IF 0 RESULTS
  compression_pct        8 → 12
  cross_lookback        10 → 15
  macd_cross_lookback   10 → 15
  rsi_cross_lookback    10 → 15
  min_vol_mult         1.0 → 0.5
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
""")
