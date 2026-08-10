# ============================================================
# NASDAQ — Minervini Stage 2 + SMA50 Pullback VCP Scanner
# ============================================================
#
# MINERVINI STAGE 2 CRITERIA (all must pass):
#   1. Price > SMA150 > SMA200  (price above both long MAs)
#   2. SMA150 and SMA200 trending UP (rising slopes)
#   3. Price > SMA50  (above medium-term trend)
#   4. Price >= 30% above its 52-week low
#   5. Price within 25% of its 52-week high  (near highs)
#   6. RS Rating >= 70  (relative strength vs market)
#      approximated: stock % change >= SPY % change (proxy)
#
# SMA50 PULLBACK + VOLUME DRY-UP + BUYING VOLUME RETURNING:
#   C1  PULLBACK TO SMA50:
#       Price pulled back from recent highs toward SMA50
#       Current price within sma50_zone_pct% of SMA50
#
#   C2  VOLUME DRY-UP (contraction):
#       During the pullback (last pullback_bars bars),
#       volume CONTRACTED — avg vol in pullback <
#       vol_dryup_pct% of the prior 20-day avg
#       = Sellers dried up — no distribution
#
#   C3  BUYING VOLUME RETURNING:
#       In the last buying_signal_bars bars, at least
#       min_buying_days green days with volume ABOVE
#       the pullback average
#       = Fresh buyers stepping in at SMA50 support
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
    "history_days"              : 400,   # need 200d MA

    # ── Minervini Stage 2 ────────────────────────────────────
    # 1. Price > SMA150 > SMA200
    # 2. SMA150 and SMA200 rising (slope over slope_bars)
    "slope_bars"                : 20,
    # 3. Price > SMA50
    # 4. Price >= 30% above 52-week low
    "min_above_52w_low_pct"     : 30.0,
    # 5. Price within 25% of 52-week high
    "max_below_52w_high_pct"    : 25.0,
    # 6. RS: stock must have outperformed SPY over rs_period
    "rs_period"                 : 52,    # weeks (252 bars approx)
    "min_rs_pct_vs_spy"         : -5.0,  # stock >= SPY perf - 5% (relaxed)

    # ── SMA50 pullback zone ────────────────────────────────────
    # Price must be within this % of SMA50 (above or below)
    "sma50_zone_pct"            : 5.0,

    # ── Volume dry-up ─────────────────────────────────────────
    # Lookback bars for the pullback phase
    "pullback_bars"             : 10,
    # Dry-up: avg vol in pullback must be <= this % of prior avg
    "vol_dryup_pct"             : 75.0,  # <= 75% of prior avg = drying up
    # Prior avg baseline (bars before pullback)
    "baseline_vol_bars"         : 20,

    # ── Buying volume returning ────────────────────────────────
    # Last N bars to check for buying signal
    "buying_signal_bars"        : 5,
    # Need at least this many green up-days with vol > pullback avg
    "min_buying_days"           : 1,

    # ── Filters ─────────────────────────────────────────────
    "min_avg_volume"            : 100_000,
    "min_price"                 : 5.0,

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
    ag    = gain.ewm(alpha=1/period, adjust=False).mean()
    al    = loss.ewm(alpha=1/period, adjust=False).mean()
    rs    = ag / al.replace(0, np.nan)
    return 100 - (100 / (1 + rs))

def calc_adr(high, low, close, period=20):
    """Average Daily Range as % of close — measures volatility."""
    daily_range = (high - low) / close * 100
    return daily_range.rolling(period).mean()

# ── SPY RS cache ──────────────────────────────────────────────
_SPY_PERF = None

def get_spy_perf(rs_period):
    global _SPY_PERF
    if _SPY_PERF is not None:
        return _SPY_PERF
    try:
        end   = datetime.today()
        start = end - timedelta(days=CFG["history_days"])
        spy   = yf.Ticker("SPY").history(
            start=start.strftime("%Y-%m-%d"),
            end=end.strftime("%Y-%m-%d"),
            auto_adjust=True)
        spy.index = pd.to_datetime(spy.index).tz_localize(None)
        if len(spy) > rs_period:
            p_now  = float(spy["Close"].iloc[-1])
            p_then = float(spy["Close"].iloc[-rs_period])
            _SPY_PERF = (p_now - p_then) / p_then * 100
        else:
            _SPY_PERF = 0.0
    except Exception:
        _SPY_PERF = 0.0
    print(f"  SPY {rs_period}d performance: {_SPY_PERF:+.1f}%")
    return _SPY_PERF

# ── Download ──────────────────────────────────────────────────
def _clean(df, min_bars=210):
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
    "fail_stage2"  : 0,
    "fail_pullback": 0,
    "fail_dryup"   : 0,
    "fail_buying"  : 0,
    "pass_all"     : 0,
}

# ── Core detection ────────────────────────────────────────────
def detect_pattern(sym, df, spy_perf):
    global _DBG
    _DBG["total"] += 1

    df      = df.copy(); df.index = pd.to_datetime(df.index)
    n       = len(df)
    price   = float(df["Close"].iloc[-1])
    avg_vol = float(df["Volume"].tail(CFG["baseline_vol_bars"]).mean())

    if price   < CFG["min_price"]:      return None
    if avg_vol < CFG["min_avg_volume"]: return None
    if n < 210: return None

    _DBG["pass_filter"] += 1

    # ── Compute MAs ───────────────────────────────────────────
    sma50_s  = df["Close"].rolling(50).mean()
    sma150_s = df["Close"].rolling(150).mean()
    sma200_s = df["Close"].rolling(200).mean()
    rsi_s    = calc_rsi(df["Close"])

    cur_s50  = float(sma50_s.iloc[-1])  if not np.isnan(sma50_s.iloc[-1])  else np.nan
    cur_s150 = float(sma150_s.iloc[-1]) if not np.isnan(sma150_s.iloc[-1]) else np.nan
    cur_s200 = float(sma200_s.iloc[-1]) if not np.isnan(sma200_s.iloc[-1]) else np.nan
    cur_rsi  = float(rsi_s.iloc[-1])    if not np.isnan(rsi_s.iloc[-1])    else 50

    if any(np.isnan([cur_s50, cur_s150, cur_s200])): return None

    # ─────────────────────────────────────────────────────────
    # MINERVINI STAGE 2 — ALL 6 CRITERIA
    # ─────────────────────────────────────────────────────────
    sb = CFG["slope_bars"]

    # 1. Price > SMA150 > SMA200
    if not (price > cur_s150 > cur_s200):
        _DBG["fail_stage2"] += 1; return None

    # 2. SMA150 and SMA200 trending up
    s150_now  = float(sma150_s.iloc[-1])
    s150_prev = float(sma150_s.iloc[-sb]) if not np.isnan(sma150_s.iloc[-sb]) else np.nan
    s200_now  = float(sma200_s.iloc[-1])
    s200_prev = float(sma200_s.iloc[-sb]) if not np.isnan(sma200_s.iloc[-sb]) else np.nan
    if np.isnan(s150_prev) or np.isnan(s200_prev): _DBG["fail_stage2"] += 1; return None
    if s150_now <= s150_prev or s200_now <= s200_prev:
        _DBG["fail_stage2"] += 1; return None

    # 3. Price > SMA50
    if price <= cur_s50:
        _DBG["fail_stage2"] += 1; return None

    # 4. Price >= 30% above 52-week low
    w52 = min(252, n)
    lo52 = float(df["Low"].tail(w52).min())
    hi52 = float(df["High"].tail(w52).max())
    above_low_pct  = (price - lo52) / lo52 * 100 if lo52 > 0 else 0
    below_high_pct = (hi52 - price) / hi52 * 100 if hi52 > 0 else 0

    if above_low_pct < CFG["min_above_52w_low_pct"]:
        _DBG["fail_stage2"] += 1; return None

    # 5. Price within 25% of 52-week high
    if below_high_pct > CFG["max_below_52w_high_pct"]:
        _DBG["fail_stage2"] += 1; return None

    # 6. RS: stock performance vs SPY over rs_period
    rs_period = CFG["rs_period"]
    if n >= rs_period:
        p_now  = price
        p_then = float(df["Close"].iloc[-rs_period])
        stock_perf = (p_now - p_then) / p_then * 100 if p_then > 0 else 0
        rs_diff    = stock_perf - spy_perf
        if rs_diff < CFG["min_rs_pct_vs_spy"]:
            _DBG["fail_stage2"] += 1; return None
    else:
        stock_perf = 0.0; rs_diff = 0.0

    # ─────────────────────────────────────────────────────────
    # C1: SMA50 PULLBACK — price near SMA50 (within 5%)
    # ─────────────────────────────────────────────────────────
    dist_s50_pct = (price - cur_s50) / cur_s50 * 100 if cur_s50 > 0 else 99
    if abs(dist_s50_pct) > CFG["sma50_zone_pct"]:
        _DBG["fail_pullback"] += 1; return None

    # ─────────────────────────────────────────────────────────
    # C2: VOLUME DRY-UP during pullback
    # Compare avg vol in last pullback_bars vs baseline
    # ─────────────────────────────────────────────────────────
    pb = CFG["pullback_bars"]
    bl = CFG["baseline_vol_bars"]

    vol_pullback  = float(df["Volume"].tail(pb).mean())
    # Baseline: vol from pb+bl to pb bars ago
    baseline_start = max(0, n - pb - bl)
    baseline_end   = max(0, n - pb)
    if baseline_end <= baseline_start:
        _DBG["fail_dryup"] += 1; return None
    vol_baseline = float(df["Volume"].iloc[baseline_start:baseline_end].mean())

    if vol_baseline <= 0:
        _DBG["fail_dryup"] += 1; return None

    vol_dryup_ratio = vol_pullback / vol_baseline * 100
    if vol_dryup_ratio > CFG["vol_dryup_pct"]:
        _DBG["fail_dryup"] += 1; return None   # volume NOT drying up

    # ─────────────────────────────────────────────────────────
    # C3: BUYING VOLUME RETURNING in last buying_signal_bars
    # Need min_buying_days green days where vol > pullback avg
    # ─────────────────────────────────────────────────────────
    bs = CFG["buying_signal_bars"]
    recent = df.tail(bs)
    buying_days = 0
    for _, row in recent.iterrows():
        is_green  = float(row["Close"]) > float(row["Open"])
        vol_above = float(row["Volume"]) > vol_pullback
        if is_green and vol_above:
            buying_days += 1

    if buying_days < CFG["min_buying_days"]:
        _DBG["fail_buying"] += 1; return None

    _DBG["pass_all"] += 1

    # ── Metrics ───────────────────────────────────────────────
    # Number of contracting bars (VCP tightening count)
    # Count how many of last pullback_bars had below-avg volume
    below_avg_bars = sum(
        1 for v in df["Volume"].tail(pb)
        if float(v) < vol_baseline
    )

    # Compute % from 52w high — how tight the setup is
    tightness_pct = below_high_pct   # lower = tighter, nearer to highs

    # Score (0-100)
    score = 0
    # Stage 2 strength: how far above 52w low (max 25)
    score += min(25, int(above_low_pct * 0.15))
    # Near 52w high — tighter setup (max 20)
    score += max(0, 20 - int(below_high_pct * 0.8))
    # Volume dry-up magnitude (max 20): drier = better
    score += max(0, 20 - int(vol_dryup_ratio * 0.2))
    # Buying days in recent window (max 15)
    score += min(15, buying_days * 5)
    # RS vs SPY (max 10)
    score += min(10, max(0, int((rs_diff + 10) * 0.5)))
    # RSI health 40-70 (max 10)
    if 40 <= cur_rsi <= 70: score += 10
    elif 30 <= cur_rsi <= 80: score += 5
    score = min(100, max(0, score))

    return {
        "Ticker"          : sym,
        "Price"           : round(price, 2),
        "Score"           : score,
        # Stage 2 metrics
        "SMA50"           : round(cur_s50, 2),
        "SMA150"          : round(cur_s150, 2),
        "SMA200"          : round(cur_s200, 2),
        "Dist_SMA50_%"    : round(dist_s50_pct, 2),
        "Above_52w_Lo_%"  : round(above_low_pct, 1),
        "Below_52w_Hi_%"  : round(below_high_pct, 1),
        "RS_vs_SPY_%"     : round(rs_diff, 1),
        "Stock_Perf_%"    : round(stock_perf, 1),
        # Volume dry-up
        "Vol_Baseline"    : int(vol_baseline),
        "Vol_Pullback"    : int(vol_pullback),
        "Vol_Dryup_%"     : round(vol_dryup_ratio, 1),
        "Below_Avg_Bars"  : below_avg_bars,
        # Buying signal
        "Buying_Days"     : buying_days,
        "Buying_Days_of"  : bs,
        # Other
        "RSI"             : round(cur_rsi, 1),
        "Avg_Vol_20d"     : int(avg_vol),
        # internals
        "_df"      : df,
        "_sma50"   : sma50_s,
        "_sma150"  : sma150_s,
        "_sma200"  : sma200_s,
    }

# ── Live print ────────────────────────────────────────────────
LIVE_COLS = ["Ticker","Price","Score",
             "Dist_SMA50_%","Below_52w_Hi_%","Above_52w_Lo_%",
             "RS_vs_SPY_%","Vol_Dryup_%","Buying_Days","RSI"]
_CW = {"Ticker":8,"Price":10,"Score":7,
       "Dist_SMA50_%":13,"Below_52w_Hi_%":15,"Above_52w_Lo_%":16,
       "RS_vs_SPY_%":13,"Vol_Dryup_%":12,"Buying_Days":12,"RSI":6}
_CF = {"Price":"${:.2f}","Score":"{:.0f}",
       "Dist_SMA50_%":"{:+.2f}%","Below_52w_Hi_%":"{:.1f}%",
       "Above_52w_Lo_%":"{:.1f}%","RS_vs_SPY_%":"{:+.1f}%",
       "Vol_Dryup_%":"{:.1f}%","Buying_Days":"{:.0f}d","RSI":"{:.1f}"}
_hdr_done = False

def _live_header():
    global _hdr_done
    if _hdr_done: return
    sep = "━"*155
    print(f"\n{sep}")
    print("  📊  LIVE — Minervini Stage 2 + SMA50 Pullback + Volume Dry-Up + Buying Returning")
    print(sep)
    print("".join(f"  {c:<{_CW.get(c,10)}}" for c in LIVE_COLS))
    print("  "+"─"*153)
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
print("  STEP 1  DATA CHECK + SPY BASELINE")
print("━"*65)
spy_perf = get_spy_perf(CFG["rs_period"])
chk = download(["AAPL","NVDA","MSFT"], CFG["history_days"])
if not chk: print("❌  No data.")
else:
    for s, d in chk.items():
        p    = float(d["Close"].iloc[-1])
        s50  = float(d["Close"].rolling(50).mean().iloc[-1])
        s200 = float(d["Close"].rolling(200).mean().iloc[-1])
        print(f"  ✅ {s}: ${p:.2f}  SMA50=${s50:.2f}  SMA200=${s200:.2f}  {d.index[-1].date()}")
print()

# ── Diagnostic ────────────────────────────────────────────────
print("━"*65)
print("  STEP 2  DIAGNOSTIC")
print("━"*65+"\n")
DIAG = ["AAPL","NVDA","MSFT","AMD","PLTR","META","CRWD","AVGO","MU","AXON"]
diag_data = download(DIAG, CFG["history_days"])
print(f"  Downloaded {len(diag_data)}/{len(DIAG)}\n")
print(f"  {'SYM':<7} {'P>S50>S150>S200':>16}  {'DIST_S50':>9}  {'DRYUP%':>7}  {'BUY_D':>6}  RESULT")
print("  "+"─"*60)

for sym in DIAG:
    if sym not in diag_data:
        print(f"  {sym:<7} — no data"); continue
    try:
        df_d = diag_data[sym]
        p    = float(df_d["Close"].iloc[-1])
        r    = detect_pattern(sym, df_d, spy_perf)
        s50  = float(df_d["Close"].rolling(50).mean().iloc[-1])
        s150 = float(df_d["Close"].rolling(150).mean().iloc[-1])
        s200 = float(df_d["Close"].rolling(200).mean().iloc[-1])
        stg2 = "✅" if (p>s50>s150>s200) else "❌"
        if r:
            print(f"  {sym:<7} {stg2:>16}  {r['Dist_SMA50_%']:>+8.2f}%  "
                  f"{r['Vol_Dryup_%']:>6.1f}%  {r['Buying_Days']:>5}d  "
                  f"✅ Score={r['Score']}")
        else:
            dist = (p-s50)/s50*100 if s50>0 else 0
            print(f"  {sym:<7} {stg2:>16}  {dist:>+8.2f}%  "
                  f"{'—':>7}  {'—':>6}  ❌")
    except Exception as e:
        print(f"  {sym:<7} error: {e}")

print(f"""
  Minervini Stage 2:
    Price > SMA50 > SMA150 > SMA200 (trend alignment)
    SMA150 and SMA200 both RISING (slope over {CFG['slope_bars']} bars)
    Price >= {CFG['min_above_52w_low_pct']}% above 52-week low  (left base)
    Price within {CFG['max_below_52w_high_pct']}% of 52-week high  (near highs)
    RS >= SPY {CFG['rs_period']}d performance - {abs(CFG['min_rs_pct_vs_spy'])}%

  SMA50 Pullback + VCP:
    Price within {CFG['sma50_zone_pct']}% of SMA50
    Vol in last {CFG['pullback_bars']} bars <= {CFG['vol_dryup_pct']}% of prior {CFG['baseline_vol_bars']}d avg (drying up)
    Last {CFG['buying_signal_bars']} bars: >= {CFG['min_buying_days']} green day with vol > pullback avg

  Tune if mostly ❌:
    sma50_zone_pct          5 → 8
    vol_dryup_pct          75 → 85
    min_above_52w_low_pct  30 → 20
    max_below_52w_high_pct 25 → 35
    min_buying_days         1 → 1  (already at minimum)
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
        "VRTX","ISRG","LULU","FTNT","IDXX","SBUX","TMUS","RBRK","NET","AXON",
        "ANET","CAVA","VRT","ELF","GRMN","ON","ENPH","ROST","AMGN","GILD",
        "INTU","MCHP","MNST","NXPI","ACLS","IRTC","HOOD","COIN","UPST","SMCI",
        "PAYC","HUBS","VEEV","BILL","GTLB","MNDY","DKNG","CELH","GLBE","SMAR",
        "APPN","ALRM","FIVE","BOOT","BCAB","INMD","LGIH","ICAD","PRCT","UFPT",
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
                r = detect_pattern(sym, data_map[sym], spy_perf)
                if r: results.append(r); live_print(r)
            except Exception: pass
        time.sleep(CFG["batch_sleep"])

got = len(TICKERS) - no_data; pct = got/max(len(TICKERS),1)*100
print(f"\n{'━'*65}")
print(f"  SCAN COMPLETE | {len(TICKERS)} tickers | {got} ({pct:.0f}%) | ✅ {len(results)} matches")
print(f"{'━'*65}")
print(f"""
  📊 DEBUG BREAKDOWN:
  Total processed           : {_DBG['total']}
  Passed vol/price/bars     : {_DBG['pass_filter']}
  ❌ Failed Stage 2          : {_DBG['fail_stage2']}
  ❌ Failed SMA50 pullback   : {_DBG['fail_pullback']}
  ❌ Failed vol dry-up       : {_DBG['fail_dryup']}
  ❌ Failed buying signal    : {_DBG['fail_buying']}
  ✅ Passed all              : {_DBG['pass_all']}
""")

if not results:
    print("  No matches. Tune the condition with most failures above:")
    print("   Stage 2:    max_below_52w_high_pct  25 → 35")
    print("               min_above_52w_low_pct   30 → 20")
    print("   Pullback:   sma50_zone_pct            5 → 8")
    print("   Dry-up:     vol_dryup_pct            75 → 85")
    print("   Buying:     min_buying_days           1 → 1 (already min)")

results.sort(key=lambda x: -x["Score"])

# ── Build df_out ──────────────────────────────────────────────
ts      = datetime.today().strftime("%Y%m%d_%H%M")
out_dir = os.environ.get("GITHUB_WORKSPACE", os.getcwd())

COLS = [
    "Ticker","Price","Score",
    "SMA50","SMA150","SMA200",
    "Dist_SMA50_%","Above_52w_Lo_%","Below_52w_Hi_%",
    "RS_vs_SPY_%","Stock_Perf_%",
    "Vol_Baseline","Vol_Pullback","Vol_Dryup_%","Below_Avg_Bars",
    "Buying_Days","Buying_Days_of",
    "RSI","Avg_Vol_20d",
]
df_out = pd.DataFrame([{k:v for k,v in r.items() if not k.startswith("_")}
                        for r in results]) if results else pd.DataFrame(columns=COLS)
if not df_out.empty:
    df_out = df_out[[c for c in COLS if c in df_out.columns]]
    df_out.reset_index(drop=True, inplace=True)

FMT = {
    "Price"          : lambda v: f"${v:.2f}",
    "Score"          : lambda v: f"{v:.0f}",
    "SMA50"          : lambda v: f"${v:.2f}",
    "SMA150"         : lambda v: f"${v:.2f}",
    "SMA200"         : lambda v: f"${v:.2f}",
    "Dist_SMA50_%"   : lambda v: f"{v:+.2f}%",
    "Above_52w_Lo_%" : lambda v: f"{v:.1f}%",
    "Below_52w_Hi_%" : lambda v: f"{v:.1f}%",
    "RS_vs_SPY_%"    : lambda v: f"{v:+.1f}%",
    "Stock_Perf_%"   : lambda v: f"{v:+.1f}%",
    "Vol_Baseline"   : lambda v: f"{v:,.0f}",
    "Vol_Pullback"   : lambda v: f"{v:,.0f}",
    "Vol_Dryup_%"    : lambda v: f"{v:.1f}%",
    "RSI"            : lambda v: f"{v:.1f}",
    "Avg_Vol_20d"    : lambda v: f"{v:,.0f}",
}

def fmt_v(col, val):
    if val is None or (isinstance(val, float) and np.isnan(val)): return "—"
    try:
        if col in FMT: return FMT[col](val)
    except Exception: pass
    return str(val) if str(val) not in ("nan","None","") else "—"

# ── Display (notebook) ────────────────────────────────────────
if _IN_NOTEBOOK and results:
    DISP = ["Ticker","Price","Score","Dist_SMA50_%",
            "Below_52w_Hi_%","Above_52w_Lo_%","RS_vs_SPY_%",
            "Vol_Dryup_%","Buying_Days","RSI"]
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
            raw=r.get(col); disp=fmt_v(col,raw); sty=""
            if col=="Score":
                try:
                    v=float(raw); g=int(min(220,80+v*1.4))
                    sty=f"background:rgb(20,{g},60);color:#fff;font-weight:700;text-align:center"
                except: pass
            elif col=="Vol_Dryup_%":
                try:
                    v=float(str(raw).replace("%",""))
                    sty="color:#22c55e;font-weight:700" if v<=50 else "color:#86efac" if v<=65 else ""
                except: pass
            elif col=="Dist_SMA50_%":
                try:
                    v=float(str(raw).replace("%","").replace("+",""))
                    sty="color:#22c55e;font-weight:700" if abs(v)<=2 else ""
                except: pass
            elif col=="RS_vs_SPY_%":
                try:
                    v=float(str(raw).replace("%","").replace("+",""))
                    sty=f"color:{'#22c55e' if v>=0 else '#ef4444'};font-weight:600"
                except: pass
            tds+=f'<td style="padding:7px 12px;font-size:12px;border-bottom:1px solid #e2e8f0;white-space:nowrap;{sty}">{disp}</td>'
        rows_html+=f'<tr style="background:{bg}">{tds}</tr>\n'

    ticker_csv_str = ",".join(r["Ticker"] for r in results)
    display_html(f"""
<div style="background:linear-gradient(135deg,#0f172a,#1e3a5f);border-radius:10px;
            padding:18px 24px;margin-bottom:8px">
  <h2 style="margin:0;color:#60a5fa;font-size:20px;font-weight:700">
    📈 Minervini Stage 2 + SMA50 Pullback VCP
  </h2>
  <p style="margin:6px 0 0;color:#94a3b8;font-size:12px">
    {datetime.today().strftime('%Y-%m-%d %H:%M')} &nbsp;·&nbsp;
    <b style="color:{gc}">{len(results)} matches</b> &nbsp;·&nbsp;
    SPY {CFG['rs_period']}d: {spy_perf:+.1f}%
  </p>
</div>
<div style="background:#0f172a;border-radius:8px;padding:14px 16px;margin:8px 0;
            border-left:4px solid {gc};">
  <p style="margin:0 0 4px;color:#94a3b8;font-size:11px;font-weight:600;
             text-transform:uppercase;letter-spacing:.05em">
    📋 Stock List (CSV)
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
  Vol_Dryup_% = pullback vol as % of prior baseline (lower=drier=better) &nbsp;·&nbsp;
  Buying_Days = green days with vol above pullback avg in last 5 bars &nbsp;·&nbsp;
  Dist_SMA50_% near 0 = tightest at SMA50 &nbsp;·&nbsp;
  RS_vs_SPY_% positive = outperforming market
</div>""")

elif results:
    CLI = ["Ticker","Price","Score","Dist_SMA50_%","Below_52w_Hi_%",
           "Vol_Dryup_%","Buying_Days","RS_vs_SPY_%","RSI"]
    col_w = {c: max(len(c), max(len(fmt_v(c,r.get(c))) for r in results))+2 for c in CLI}
    top = "┬".join("─"*col_w[c] for c in CLI)
    sep = "┼".join("─"*col_w[c] for c in CLI)
    bot = "┴".join("─"*col_w[c] for c in CLI)
    hdr = "│".join(c.center(col_w[c]) for c in CLI)
    inner = sum(col_w.values())+len(CLI)-1
    print(f"\n  ╔{'═'*inner}╗")
    print(f"  ║{'  Minervini Stage 2 + SMA50 VCP Pullback   '+datetime.today().strftime('%Y-%m-%d')+'   '+str(len(df_out))+' matches'.center(inner)}║")
    print(f"  ╚{'═'*inner}╝\n")
    print(f"  ┌{top}┐\n  │{hdr}│\n  ├{sep}┤")
    for i,r in enumerate(results):
        cells=[fmt_v(c,r.get(c)).center(col_w[c]) for c in CLI]
        print(f"  │{'│'.join(cells)}│")
        if i<len(results)-1: print(f"  ├{sep}┤")
    print(f"  └{bot}┘")

# ── Save CSV + TV ──────────────────────────────────────────────
fpath = os.path.join(out_dir, f"minervini_stage2_vcp_{ts}.csv")
df_out.to_csv(fpath, index=False)
print(f"\n  💾 CSV → {fpath}")
tv = os.path.join(out_dir, f"tv_stage2_vcp_{ts}.txt")
with open(tv,"w") as f:
    f.write(f"###Minervini Stage2 VCP {datetime.today().strftime('%Y-%m-%d')}\n")
    for r in results: f.write(f"NASDAQ:{r['Ticker']}\n")
print(f"  📋 TradingView → {tv}")

# ── Email ──────────────────────────────────────────────────────
def _send_email(rl, csv_path):
    gu=_GMAIL_USER; gp=_GMAIL_PASS; et=_EMAIL_TO
    if not gu: print("[Email] ❌  GMAIL_USER secret is empty"); return
    if not gp: print("[Email] ❌  GMAIL_PASS secret is empty\n         → myaccount.google.com/apppasswords"); return
    if not et: print("[Email] ❌  EMAIL_TO secret is empty"); return
    eto=[e.strip() for e in et.split(",") if e.strip()]
    cnt=len(rl)

    try:
        ticker_csv = ",".join(r.get("Ticker","") for r in rl) if rl else "—"
        print(f"[Email] Sending to {et}  ({cnt} results)...")

        ticker_html = f"""
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

        th_e = "".join(
            f'<th style="background:#1e293b;color:#e2e8f0;padding:8px 11px;'
            f'font-size:11px;font-weight:700;border-bottom:2px solid #3b82f6;'
            f'white-space:nowrap">{c}</th>'
            for c in ["Ticker","Price","Score","Dist_SMA50_%",
                      "Below_52w_Hi_%","Vol_Dryup_%","Buying_Days",
                      "RS_vs_SPY_%","RSI"]
        )
        rows_e = ""
        for i,r in enumerate(rl[:50]):
            bg = "#fff" if i%2==0 else "#f0f9ff"
            bhi = float(r.get("Below_52w_Hi_%",99))
            dryup = float(r.get("Vol_Dryup_%",100))
            rsvspy= float(r.get("RS_vs_SPY_%",0))
            rows_e += (
                f'<tr style="background:{bg}">'
                f'<td style="padding:6px 11px;font-size:12px;font-weight:700;'
                f'color:#22c55e">{r.get("Ticker","—")}</td>'
                f'<td style="padding:6px 11px;font-size:12px">'
                f'${float(r.get("Price",0)):.2f}</td>'
                f'<td style="padding:6px 11px;font-size:12px;font-weight:700;'
                f'background:#166534;color:#fff;text-align:center">'
                f'{float(r.get("Score",0)):.0f}</td>'
                f'<td style="padding:6px 11px;font-size:12px">'
                f'{float(r.get("Dist_SMA50_%",0)):+.2f}%</td>'
                f'<td style="padding:6px 11px;font-size:12px;color:'
                f'{"#22c55e" if bhi<=10 else "#94a3b8"}">'
                f'{bhi:.1f}%</td>'
                f'<td style="padding:6px 11px;font-size:12px;color:'
                f'{"#22c55e" if dryup<=60 else "#86efac"};font-weight:700">'
                f'{dryup:.1f}%</td>'
                f'<td style="padding:6px 11px;font-size:12px;text-align:center">'
                f'{int(r.get("Buying_Days",0))}d</td>'
                f'<td style="padding:6px 11px;font-size:12px;color:'
                f'{"#22c55e" if rsvspy>=0 else "#ef4444"};font-weight:600">'
                f'{rsvspy:+.1f}%</td>'
                f'<td style="padding:6px 11px;font-size:12px">'
                f'{float(r.get("RSI",0)):.1f}</td>'
                f'</tr>'
            )

        no_res = "" if cnt else (
            '<tr><td colspan="9" style="padding:20px;text-align:center;'
            'color:#64748b;font-size:13px">No Stage 2 VCP setups found today</td></tr>'
        )

        html_e = f"""<!DOCTYPE html><html><body style="margin:0;padding:0;
background:#f1f5f9;font-family:'Segoe UI',Arial,sans-serif">
<table width="100%" cellpadding="0" cellspacing="0" style="padding:20px 0"><tr><td>
<table width="100%" cellpadding="0" cellspacing="0"
   style="max-width:960px;margin:0 auto;background:#fff;border-radius:12px;
          overflow:hidden;box-shadow:0 4px 20px rgba(0,0,0,0.08)">
  <tr><td style="background:linear-gradient(135deg,#0f172a,#1e3a5f);padding:22px 28px">
    <h1 style="margin:0;color:#60a5fa;font-size:20px;font-weight:700">
      📊 Minervini Stage 2 + SMA50 Pullback VCP
    </h1>
    <p style="margin:6px 0 0;color:#94a3b8;font-size:12px">
      {datetime.today().strftime('%Y-%m-%d %H:%M UTC')} &nbsp;·&nbsp;
      {cnt} match{'es' if cnt!=1 else ''} &nbsp;·&nbsp;
      SPY {CFG['rs_period']}d: {spy_perf:+.1f}%
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
      📎 CSV + TradingView file attached &nbsp;·&nbsp;
      <b>Vol_Dryup_%</b> = pullback vol ÷ baseline (lower=drier) &nbsp;·&nbsp;
      <b>Below_52w_Hi_%</b> = distance from 52w high (tighter setup = lower) &nbsp;·&nbsp;
      <b>Buying_Days</b> = green days with vol above pullback avg (last 5 bars)
    </p>
  </td></tr>
  <tr><td style="background:#f8fafc;padding:12px 28px;border-top:1px solid #e2e8f0;text-align:center">
    <p style="margin:0;color:#94a3b8;font-size:10px">
      ⚠️ Not financial advice &nbsp;·&nbsp; Auto-generated by GitHub Actions
    </p>
  </td></tr>
</table></td></tr></table></body></html>"""

        plain_e = "\n".join([
            f"Minervini Stage 2 + SMA50 VCP — {datetime.today().strftime('%Y-%m-%d')}",
            f"{cnt} matches  |  SPY {CFG['rs_period']}d: {spy_perf:+.1f}%",
            "",
            f"STOCKS: {ticker_csv}",
            "",
            "="*65,
        ] + ([
            f"{r.get('Ticker','—'):<7} ${float(r.get('Price',0)):.2f}  "
            f"Score:{float(r.get('Score',0)):.0f}  "
            f"Dist_S50:{float(r.get('Dist_SMA50_%',0)):+.2f}%  "
            f"Dryup:{float(r.get('Vol_Dryup_%',0)):.0f}%  "
            f"BuyDays:{int(r.get('Buying_Days',0))}  "
            f"RS:{float(r.get('RS_vs_SPY_%',0)):+.1f}%"
            for r in rl[:50]
        ] if rl else ["No matches today"]) + ["\n📎 CSV + TradingView file attached."])

        subj = (f"📊 Stage2 VCP — {cnt} setup{'s' if cnt!=1 else ''}"
                f" — {datetime.today().strftime('%Y-%m-%d')}")

        msg = MIMEMultipart("mixed")
        msg["Subject"]=subj; msg["From"]=gu; msg["To"]=", ".join(eto)
        alt = MIMEMultipart("alternative")
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
                msg.attach(part)
                print(f"[Email] 📎 {os.path.basename(att)}")
            except Exception as e: print(f"[Email] ⚠️  Attach failed: {e}")

    try:
        with smtplib.SMTP_SSL("smtp.gmail.com",465) as srv:
            srv.login(gu, gp.replace(" ",""))
            srv.sendmail(gu, eto, msg.as_string())
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

# ── Charts for top 6 ──────────────────────────────────────────
if results:
    top = results[:min(6,len(results))]
    fig, axes = plt.subplots(len(top),1,figsize=(15,5*len(top)),facecolor="#0f172a")
    if len(top)==1: axes=[axes]

    for idx, r in enumerate(top):
        ax   = axes[idx]
        df_p = r["_df"].tail(120).copy()
        n_p  = len(df_p)
        fn   = len(r["_df"]); off = fn - n_p
        s50  = r["_sma50"].reindex(df_p.index)
        s150 = r["_sma150"].reindex(df_p.index)
        s200 = r["_sma200"].reindex(df_p.index)

        ax.set_facecolor("#0f172a")
        for i,(_, row_) in enumerate(df_p.iterrows()):
            o=float(row_["Open"]); h=float(row_["High"])
            l=float(row_["Low"]);  c=float(row_["Close"])
            clr="#34d399" if c>=o else "#ef4444"
            ax.plot([i,i],[l,h],color=clr,lw=0.7,zorder=2)
            blo=min(o,c); bhi=max(o,c); bh=max(bhi-blo,(h-l)*0.005)
            rect=mpatches.FancyBboxPatch((i-0.3,blo),0.6,bh,
                 boxstyle="square,pad=0",facecolor=clr,edgecolor=clr,lw=0.3,zorder=3)
            ax.add_patch(rect)

        ax.plot(range(n_p), s50.values,  color="#3b82f6", lw=1.8, label="SMA50",  zorder=5)
        ax.plot(range(n_p), s150.values, color="#f59e0b", lw=1.4, ls="--", label="SMA150", zorder=4)
        ax.plot(range(n_p), s200.values, color="#f472b6", lw=1.4, ls="-.", label="SMA200", zorder=4)
        ax.axhline(r["SMA50"], color="#3b82f6", lw=0.6, alpha=0.4)

        # Shade pullback zone
        pb = CFG["pullback_bars"]
        ax.axvspan(n_p-pb, n_p-1, alpha=0.1, color="#22c55e", label="Pullback zone")

        tick_step = max(1, n_p//8)
        ax.set_xticks(range(0,n_p,tick_step))
        ax.set_xticklabels(
            [df_p.index[i].strftime("%m/%d") for i in range(0,n_p,tick_step)],
            color="#94a3b8", fontsize=7)
        ax.set_xlim(-0.5, n_p-0.5)
        ax.set_title(
            f"{r['Ticker']}  ${r['Price']:.2f}  |  Score {r['Score']}/100  |  "
            f"Dist_S50 {r['Dist_SMA50_%']:+.2f}%  |  "
            f"Vol_Dryup {r['Vol_Dryup_%']:.0f}%  |  "
            f"BuyDays {r['Buying_Days']}/{r['Buying_Days_of']}  |  "
            f"RS_vs_SPY {r['RS_vs_SPY_%']:+.1f}%  |  "
            f"Below_Hi {r['Below_52w_Hi_%']:.1f}%  |  RSI {r['RSI']:.0f}",
            color="#e2e8f0", fontsize=8, fontweight="bold", pad=5)
        ax.tick_params(colors="#94a3b8", labelsize=7)
        for sp_ in ax.spines.values(): sp_.set_edgecolor("#1e3a5f")
        ax.legend(loc="upper left", facecolor="#1e293b",
                  labelcolor="#e2e8f0", fontsize=7, framealpha=0.9)
        ax.grid(color="#1e3a5f", ls="--", lw=0.4, alpha=0.4, axis="y")

    plt.suptitle(
        f"Minervini Stage 2 + SMA50 Pullback VCP  ·  {datetime.today().strftime('%Y-%m-%d')}\n"
        f"🔵 SMA50  🟠 SMA150  🩷 SMA200  🟢 shaded = pullback zone",
        color="#60a5fa", fontsize=10, fontweight="bold", y=1.001)
    plt.tight_layout()
    cp = os.path.join(out_dir, f"stage2_vcp_chart_{ts}.png")
    plt.savefig(cp, dpi=150, bbox_inches="tight", facecolor="#0f172a")
    if _IN_NOTEBOOK: plt.show()
    else: plt.close()
    print(f"  📊 Chart → {cp}")

print("""
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  📋 PATTERN EXPLAINED

  MINERVINI STAGE 2 (all 6 must pass):
    ① Price > SMA50 > SMA150 > SMA200 (full trend alignment)
    ② SMA150 and SMA200 both RISING (positive slopes)
    ③ Price >= 30% above 52-week low (emerged from base)
    ④ Price within 25% of 52-week high (near highs = stage 2)
    ⑤ Stock RS >= SPY performance (outperforming market)

  SMA50 PULLBACK + VOLUME DRY-UP:
    Price pulls back to within 5% of SMA50 (health check)
    Volume CONTRACTS during pullback vs prior baseline:
      Vol_Dryup_% <= 75% means sellers are not present
      = No distribution, just natural healthy pullback
    This is the "VCP" (Volatility Contraction Pattern):
      each pullback tightens with declining volume

  BUYING VOLUME RETURNING:
    In the last 5 bars, at least 1 GREEN day where
    volume exceeded the pullback average
    = Buyers stepping in at SMA50 support

  💡 BEST SETUPS
  Score >= 70           elite setup quality
  Dist_SMA50_% near 0  right at SMA50 (tightest entry)
  Vol_Dryup_% < 50%    very dry — strong VCP
  Below_52w_Hi_% < 10  very tight, near highs
  RS_vs_SPY_% positive outperforming market

  ⚙️  TUNE IF 0 RESULTS
  sma50_zone_pct          5 → 8
  vol_dryup_pct          75 → 85
  min_above_52w_low_pct  30 → 20
  max_below_52w_high_pct 25 → 35
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
""")
