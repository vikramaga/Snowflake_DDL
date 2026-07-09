# ============================================================
# NASDAQ — SMA50 Compression to SMA150 + Bullish Reclaim
# ============================================================
#
# EXACT PATTERN (from ZGN chart — the red-circled zone):
#
#  C1 — SMA50 APPROACHING SMA150 FROM ABOVE  (the compression)
#      SMA50 is still ABOVE SMA150 (uptrend structurally intact)
#      But the GAP between them has been NARROWING —
#      SMA50 is descending toward SMA150 (converging)
#      Distance now within compression_pct% (very tight coil)
#      = The medium-term trend is decelerating into the
#        long-term trend line — classic "energy compression"
#        before either a breakdown or a renewed breakout
#
#  C2 — PRICE CLOSED ABOVE SMA50  (the reclaim / trigger)
#      EXACT 1-bar cross (most recent bar):
#        Bar[-2] close < SMA50[-2]   ← was below SMA50 yesterday
#        Bar[-1] close >= SMA50[-1]  ← closed above SMA50 today
#      = Price bounced right at the SMA50/SMA150 compression
#        zone and reclaimed the medium-term trend line
#
#  C3 — GOOD VOLUME ON THE RECLAIM BAR
#      Volume on the reclaim bar >= vol_mult × avg volume
#      = Real buying interest confirming the bounce,
#        not just a low-volume drift back above SMA50
#
# WHY THIS MATTERS:
#   When SMA50 compresses down into SMA150 (rather than crossing
#   below it), and price snaps back above SMA50 on volume, it
#   often marks the resumption of the primary uptrend — the
#   "coil released" moment visible in the ZGN chart where price,
#   EMA20, SMA50, and SMA150 all converge before turning up
#   together with rising volume.
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
    "ema20_period"               : 20,
    "sma50_period"                : 50,
    "sma150_period"               : 150,

    # ── C1: SMA50 compression toward SMA150 ───────────────────
    # SMA50 must currently be ABOVE SMA150 (still bullish structure)
    "require_sma50_above_sma150" : True,
    # Current gap between SMA50 and SMA150 must be within this %
    # (tight compression zone)
    "compression_pct"            : 3.0,
    # SMA50 must have been DESCENDING toward SMA150 over the last
    # N bars (confirms "approaching from above", not just flat)
    "convergence_lookback"       : 15,
    # The gap N bars ago must have been at least this much WIDER
    # than the current gap (confirms real convergence happened)
    "min_gap_narrowing_pct"      : 1.0,

    # ── C2: Price closed above SMA50 — exact 1-bar cross ──────
    "cross_lookback"             : 5,

    # ── C3: Volume confirmation ────────────────────────────────
    "vol_avg_bars"                : 20,
    "vol_mult"                    : 1.3,   # reclaim bar >= 1.3x avg volume

    # ── RSI ────────────────────────────────────────────────────
    "rsi_min"                     : 35,

    # ── Filters ────────────────────────────────────────────────
    "min_avg_volume"              : 80_000,
    "min_price"                   : 1.0,

    "batch_size"                  : 50,
    "batch_sleep"                 : 1.5,
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
    C1: SMA50 compression toward SMA150 from above
        - SMA50 currently above SMA150 (still bullish)
        - Gap has been NARROWING over convergence_lookback bars
        - Current gap within compression_pct% (tight coil)
    C2: Price closed above SMA50 — exact 1-bar cross
        - Bar[-2] close < SMA50   (was below yesterday)
        - Bar[-1] close >= SMA50  (closed above today)
    C3: Volume on the reclaim bar >= vol_mult x average
    """
    df      = df.copy(); df.index = pd.to_datetime(df.index)
    n       = len(df)
    price   = float(df["Close"].iloc[-1])
    avg_vol = float(df["Volume"].tail(CFG["vol_avg_bars"]).mean())

    if price   < CFG["min_price"]:      return None
    if avg_vol < CFG["min_avg_volume"]: return None
    if n < CFG["sma150_period"] + CFG["convergence_lookback"] + 5: return None

    # ── Compute MAs ───────────────────────────────────────────
    ema20_s  = calc_ema(df["Close"], CFG["ema20_period"])
    sma50_s  = df["Close"].rolling(CFG["sma50_period"]).mean()
    sma150_s = df["Close"].rolling(CFG["sma150_period"]).mean()
    rsi_s    = calc_rsi(df["Close"])
    macd_s, _, hist_s = calc_macd(df["Close"])

    cur_ema20  = float(ema20_s.iloc[-1])
    cur_sma50  = float(sma50_s.iloc[-1])
    cur_sma150 = float(sma150_s.iloc[-1])
    cur_rsi    = float(rsi_s.iloc[-1]) if not np.isnan(rsi_s.iloc[-1]) else 50
    cur_hist   = float(hist_s.iloc[-1]) if not np.isnan(hist_s.iloc[-1]) else 0

    if any(np.isnan([cur_ema20, cur_sma50, cur_sma150])): return None
    if cur_rsi < CFG["rsi_min"]: return None

    # ─────────────────────────────────────────────────────────
    # C2: PRICE CLOSED ABOVE SMA50 — exact 1-bar cross
    # (check this first — fast gate before the slower C1 search)
    # ─────────────────────────────────────────────────────────
    cl = CFG["cross_lookback"]
    reclaim_bar  = None
    reclaim_date = None

    for i in range(max(1, n - cl), n):
        pc  = float(df["Close"].iloc[i-1])
        cc  = float(df["Close"].iloc[i])
        ps  = float(sma50_s.iloc[i-1]) if not np.isnan(sma50_s.iloc[i-1]) else np.nan
        cs  = float(sma50_s.iloc[i])   if not np.isnan(sma50_s.iloc[i])   else np.nan
        if np.isnan(ps) or np.isnan(cs): continue
        # Exact 1-bar cross above SMA50
        if pc < ps and cc >= cs:
            reclaim_bar  = i
            reclaim_date = df.index[i]

    if reclaim_bar is None: return None   # no SMA50 reclaim found

    # ─────────────────────────────────────────────────────────
    # C3: Volume on the reclaim bar
    # ─────────────────────────────────────────────────────────
    reclaim_vol = float(df["Volume"].iloc[reclaim_bar])
    vol_mult    = reclaim_vol / avg_vol if avg_vol > 0 else 0
    if vol_mult < CFG["vol_mult"]: return None

    # ─────────────────────────────────────────────────────────
    # C1: SMA50 COMPRESSION TOWARD SMA150 FROM ABOVE
    # Checked AS OF the reclaim bar (the moment of the bounce)
    # ─────────────────────────────────────────────────────────
    s50_at_reclaim  = float(sma50_s.iloc[reclaim_bar])  if not np.isnan(sma50_s.iloc[reclaim_bar])  else np.nan
    s150_at_reclaim = float(sma150_s.iloc[reclaim_bar]) if not np.isnan(sma150_s.iloc[reclaim_bar]) else np.nan
    if np.isnan(s50_at_reclaim) or np.isnan(s150_at_reclaim): return None

    # SMA50 must be above SMA150 (structurally still bullish)
    if CFG["require_sma50_above_sma150"] and s50_at_reclaim <= s150_at_reclaim:
        return None

    # Current gap between SMA50 and SMA150 (at reclaim bar)
    gap_now_pct = (s50_at_reclaim - s150_at_reclaim) / s150_at_reclaim * 100
    if gap_now_pct > CFG["compression_pct"]:
        return None   # not tight enough — no real compression

    # ── Confirm SMA50 was DESCENDING toward SMA150 ────────────
    # Compare the gap `convergence_lookback` bars before the
    # reclaim bar vs the gap at the reclaim bar — must have
    # narrowed by at least min_gap_narrowing_pct
    lb = CFG["convergence_lookback"]
    ref_bar = max(0, reclaim_bar - lb)

    s50_ref  = float(sma50_s.iloc[ref_bar])  if not np.isnan(sma50_s.iloc[ref_bar])  else np.nan
    s150_ref = float(sma150_s.iloc[ref_bar]) if not np.isnan(sma150_s.iloc[ref_bar]) else np.nan
    if np.isnan(s50_ref) or np.isnan(s150_ref): return None

    gap_ref_pct = (s50_ref - s150_ref) / s150_ref * 100 if s150_ref > 0 else 0

    # SMA50 must have been meaningfully further above SMA150
    # `convergence_lookback` bars ago than it is now (narrowing)
    gap_narrowed_pct = gap_ref_pct - gap_now_pct
    if gap_narrowed_pct < CFG["min_gap_narrowing_pct"]:
        return None   # gap didn't narrow enough — not a real compression

    # SMA50 must also have been literally DESCENDING (sloping down)
    # over the convergence window — not just naturally spreading
    sma50_slope_pct = (s50_at_reclaim - s50_ref) / s50_ref * 100 if s50_ref > 0 else 0
    if sma50_slope_pct > 0:
        return None   # SMA50 was rising, not compressing down — reject

    # ── Metrics ───────────────────────────────────────────────
    bars_since_reclaim = n - 1 - reclaim_bar
    dist_ema20_pct     = (price - cur_ema20)  / cur_ema20  * 100
    dist_sma50_pct     = (price - cur_sma50)  / cur_sma50  * 100
    dist_sma150_pct    = (price - cur_sma150) / cur_sma150 * 100
    sma50_vs_sma150    = (cur_sma50 - cur_sma150) / cur_sma150 * 100

    prev_close = float(df["Close"].iloc[reclaim_bar - 1])
    prev_sma50 = float(sma50_s.iloc[reclaim_bar - 1])
    reclaim_close = float(df["Close"].iloc[reclaim_bar])

    # ── Score (0-100) ─────────────────────────────────────────
    score = 0

    # Compression tightness (0-30): tighter gap = better coil
    score += max(0, 30 - int(gap_now_pct * 10))

    # Gap narrowing magnitude (0-20): more convergence = stronger setup
    score += min(20, int(gap_narrowed_pct * 4))

    # Reclaim freshness (0-20): today = 20, yesterday = 15...
    score += max(0, 20 - bars_since_reclaim * 5)

    # Volume on reclaim (0-20)
    score += min(20, int(vol_mult * 8))

    # MACD turning positive (0-5)
    score += 5 if cur_hist > 0 else 0

    # RSI health (0-5)
    score += min(5, max(0, int((cur_rsi - 40) / 4)))

    score = min(100, max(0, score))

    return {
        "Ticker"              : sym,
        "Price"               : round(price, 2),
        "Score"               : score,
        # Compression info
        "Gap_Now_%"           : round(gap_now_pct, 2),
        "Gap_Ref_%"           : round(gap_ref_pct, 2),
        "Gap_Narrowed_%"      : round(gap_narrowed_pct, 2),
        "SMA50_Slope_%"       : round(sma50_slope_pct, 3),
        # Reclaim info
        "Reclaim_Date"        : reclaim_date.strftime("%Y-%m-%d"),
        "Bars_Since_Reclaim"  : bars_since_reclaim,
        "Reclaim_Close"       : round(reclaim_close, 2),
        "Prev_Close"          : round(prev_close, 2),
        "Prev_SMA50"          : round(prev_sma50, 2),
        "Vol_x_Avg"           : round(vol_mult, 2),
        # MA levels
        "EMA20"               : round(cur_ema20, 2),
        "SMA50"               : round(cur_sma50, 2),
        "SMA150"              : round(cur_sma150, 2),
        "Dist_EMA20_%"        : round(dist_ema20_pct, 2),
        "Dist_SMA50_%"        : round(dist_sma50_pct, 2),
        "Dist_SMA150_%"       : round(dist_sma150_pct, 2),
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
        "_reclaim_bar"        : reclaim_bar,
    }

# ── Live print ────────────────────────────────────────────────
LIVE_COLS = [
    "Ticker","Price","Score",
    "Gap_Now_%","Gap_Narrowed_%","SMA50_Slope_%",
    "Reclaim_Date","Bars_Since_Reclaim","Vol_x_Avg",
    "EMA20","SMA50","SMA150","Dist_EMA20_%","RSI",
]
_CW = {
    "Ticker":8,"Price":10,"Score":7,
    "Gap_Now_%":12,"Gap_Narrowed_%":16,"SMA50_Slope_%":15,
    "Reclaim_Date":14,"Bars_Since_Reclaim":19,"Vol_x_Avg":11,
    "EMA20":9,"SMA50":9,"SMA150":9,"Dist_EMA20_%":13,"RSI":6,
}
_CF = {
    "Price":"${:.2f}","Score":"{:.0f}",
    "Gap_Now_%":"{:.2f}%","Gap_Narrowed_%":"{:+.2f}%","SMA50_Slope_%":"{:+.3f}%",
    "Bars_Since_Reclaim":"{:.0f}","Vol_x_Avg":"{:.2f}×",
    "EMA20":"${:.2f}","SMA50":"${:.2f}","SMA150":"${:.2f}",
    "Dist_EMA20_%":"{:+.2f}%","RSI":"{:.1f}",
}
_hdr_done = False

def _live_header():
    global _hdr_done
    if _hdr_done: return
    sep = "━" * 195
    print(f"\n{sep}")
    print("  📊  LIVE MATCHES  —  SMA50 Compression to SMA150 + Bullish Reclaim")
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
chk = download(["AAPL","NVDA","MSFT"], 300)
if not chk: print("❌  No data.")
else:
    for s, d in chk.items():
        p    = float(d["Close"].iloc[-1])
        s50  = float(d["Close"].rolling(50).mean().iloc[-1])
        s150 = float(d["Close"].rolling(150).mean().iloc[-1])
        gap  = (s50-s150)/s150*100 if s150>0 else 0
        print(f"  ✅ {s}: ${p:.2f}  SMA50=${s50:.2f}  SMA150=${s150:.2f}  "
              f"Gap={gap:+.2f}%  {d.index[-1].date()}")
print()

# ── Diagnostic ────────────────────────────────────────────────
print("━"*65)
print("  STEP 2  DIAGNOSTIC (10 sample stocks)")
print("━"*65+"\n")

DIAG = ["AAPL","MSFT","NVDA","AMD","PLTR","META","CRWD","AVGO","DDOG","MU"]
diag_data = download(DIAG, CFG["history_days"])
print(f"  Downloaded {len(diag_data)}/{len(DIAG)}\n")
print(f"  {'SYM':<7} {'PRICE':>8}  {'GAP%':>7}  {'SLOPE':>8}  "
      f"{'VOL×':>6}  {'SCORE':>6}  RESULT")
print("  "+"─"*60)

for sym in DIAG:
    if sym not in diag_data:
        print(f"  {sym:<7} — no data"); continue
    try:
        df_d = diag_data[sym]
        p    = float(df_d["Close"].iloc[-1])
        s50  = float(df_d["Close"].rolling(50).mean().iloc[-1])
        s150 = float(df_d["Close"].rolling(150).mean().iloc[-1])
        gap  = (s50-s150)/s150*100 if s150>0 else 0
        r    = detect_pattern(sym, df_d)
        if r:
            print(f"  {sym:<7} ${p:>7.2f}  {r['Gap_Now_%']:>6.2f}%  "
                  f"{r['SMA50_Slope_%']:>+7.2f}%  "
                  f"{r['Vol_x_Avg']:>5.1f}×  {r['Score']:>6}  ✅")
        else:
            print(f"  {sym:<7} ${p:>7.2f}  {gap:>6.2f}%  "
                  f"{'—':>8}  {'—':>6}  {'—':>6}  ❌")
    except Exception as e:
        print(f"  {sym:<7} error: {e}")

print(f"""
  Pattern:
    C1  Compression : SMA50 above SMA150, gap <= {CFG['compression_pct']}%
                      gap narrowed >= {CFG['min_gap_narrowing_pct']}% over last {CFG['convergence_lookback']} bars
                      (SMA50 was descending toward SMA150)
    C2  Reclaim      : Bar[-2] close < SMA50  AND  Bar[-1] close >= SMA50
    C3  Volume       : reclaim bar volume >= {CFG['vol_mult']}x avg

  Tune if mostly ❌:
    compression_pct        {CFG['compression_pct']} → 5    (wider compression zone)
    min_gap_narrowing_pct  {CFG['min_gap_narrowing_pct']} → 0.5  (less strict convergence)
    convergence_lookback   {CFG['convergence_lookback']} → 25   (look further back)
    vol_mult               {CFG['vol_mult']} → 1.0  (any volume)
    rsi_min                {CFG['rsi_min']} → 25
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
            b  = len(pool)
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

got = len(TICKERS) - no_data; pct = got/max(len(TICKERS),1)*100
print(f"\n{'━'*65}")
print(f"  SCAN COMPLETE | {len(TICKERS)} tickers | {got} ({pct:.0f}%) | ✅ {len(results)} matches")
print(f"{'━'*65}")

if not results:
    print("\n  No matches. Try:")
    print("   compression_pct         3 → 5")
    print("   min_gap_narrowing_pct   1 → 0.5")
    print("   convergence_lookback   15 → 25")
    print("   vol_mult              1.3 → 1.0")
    print("   rsi_min                35 → 25")

# Sort by score (always runs, even on empty list)
results.sort(key=lambda x: x["Score"], reverse=True)

# ── Always build df_out and save/email (even if 0 results) ────
ts      = datetime.today().strftime("%Y%m%d_%H%M")
out_dir = os.environ.get("GITHUB_WORKSPACE", os.getcwd())

COLS = [
    "Ticker","Price","Score",
    "Gap_Now_%","Gap_Ref_%","Gap_Narrowed_%","SMA50_Slope_%",
    "Reclaim_Date","Bars_Since_Reclaim","Reclaim_Close",
    "Prev_Close","Prev_SMA50","Vol_x_Avg",
    "EMA20","SMA50","SMA150",
    "Dist_EMA20_%","Dist_SMA50_%","Dist_SMA150_%","SMA50_vs_SMA150_%",
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
    "Reclaim_Close"       : lambda v: f"${v:.2f}",
    "Prev_Close"          : lambda v: f"${v:.2f}",
    "Prev_SMA50"          : lambda v: f"${v:.2f}",
    "Gap_Now_%"           : lambda v: f"{v:.2f}%",
    "Gap_Ref_%"           : lambda v: f"{v:.2f}%",
    "Gap_Narrowed_%"      : lambda v: f"{v:+.2f}%",
    "SMA50_Slope_%"       : lambda v: f"{v:+.3f}%",
    "Bars_Since_Reclaim"  : lambda v: f"{int(v)}",
    "Vol_x_Avg"           : lambda v: f"{v:.2f}×",
    "Dist_EMA20_%"        : lambda v: f"{v:+.2f}%",
    "Dist_SMA50_%"        : lambda v: f"{v:+.2f}%",
    "Dist_SMA150_%"       : lambda v: f"{v:+.2f}%",
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
            "Gap_Now_%","Gap_Narrowed_%","SMA50_Slope_%",
            "Reclaim_Date","Bars_Since_Reclaim","Vol_x_Avg",
            "EMA20","SMA50","SMA150","Dist_EMA20_%","RSI","MACD_Hist"]
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
            elif col == "Gap_Now_%":
                try:
                    v = float(str(raw).replace("%",""))
                    if v <= 1.0: sty = "color:#22c55e;font-weight:700"
                    elif v <= 2.0: sty = "color:#86efac"
                except Exception: pass
            elif col in ("Gap_Narrowed_%","Dist_EMA20_%"):
                try:
                    v = float(str(raw).replace("%","").replace("+",""))
                    clr = "#22c55e" if v >= 0 else "#ef4444"
                    sty = f"color:{clr};font-weight:600"
                except Exception: pass
            elif col == "SMA50_Slope_%":
                sty = "color:#f59e0b;font-weight:600"   # always negative (descending) by design
            elif col == "Bars_Since_Reclaim":
                try:
                    v = int(float(raw))
                    if v == 0: sty = "color:#22c55e;font-weight:700;text-align:center"
                    elif v <= 1: sty = "color:#86efac;text-align:center"
                except Exception: pass
            elif col == "Vol_x_Avg":
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
    <span style="color:#f1f5f9;font-size:15px;font-weight:700">SMA50 Compression → SMA150 + Bullish Reclaim</span>
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
    📈 SMA50 Compression to SMA150 + Bullish Reclaim
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
  SMA50 was descending toward SMA150 (compression) &nbsp;·&nbsp;
  Gap_Now_% = current distance between SMA50/SMA150 (lower = tighter coil) &nbsp;·&nbsp;
  Gap_Narrowed_% = how much the gap shrank over the lookback window &nbsp;·&nbsp;
  Price closed back above SMA50 on volume &nbsp;·&nbsp;
  Bars_Since_Reclaim 0 = today
</div>"""

    display_html(header_html + table_html + legend_html)

elif results:
    # ASCII table (CLI/GitHub Actions mode)
    CLI_COLS = ["Ticker","Price","Score",
                "Gap_Now_%","Gap_Narrowed_%","Reclaim_Date",
                "Bars_Since_Reclaim","Vol_x_Avg","Dist_EMA20_%","RSI"]
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
    tit  = f"  SMA50 Compression + Bullish Reclaim   {datetime.today().strftime('%Y-%m-%d')}   {len(df_out)} matches"
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
  Gap_Now_%          current % distance SMA50 is above SMA150
  Gap_Narrowed_%      how much the gap shrank (compression amount)
  Reclaim_Date        date price closed back above SMA50
  Bars_Since_Reclaim  0 = reclaim happened today
  Vol_x_Avg           reclaim bar volume vs 20d average
  Dist_EMA20_%        how far price is above EMA20 now
  ──────────────────────────────────────────────────────""")

# Save
out_dir = os.environ.get("GITHUB_WORKSPACE", os.getcwd())
fpath   = os.path.join(out_dir, f"sma50_compression_reclaim_{ts}.csv")
df_out.to_csv(fpath, index=False)
print(f"\n  💾 CSV → {fpath}")
tv = os.path.join(out_dir, f"tv_sma50_compression_{ts}.txt")
with open(tv,"w") as f:
    f.write(f"###SMA50 Compression Reclaim {datetime.today().strftime('%Y-%m-%d')}\n")
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
            for c in ["Ticker","Price","Score","Gap_Now_%",
                      "Bars_Since_Reclaim","Vol_x_Avg","Dist_EMA20_%","RSI"]
        )
        rows_e = ""
        for i, r in enumerate(rl[:50]):
            bg  = "#fff" if i % 2 == 0 else "#f0f9ff"
            ticker = r.get("Ticker","—")
            price  = r.get("Price",0) or 0
            score  = r.get("Score",0) or 0
            gapnow = r.get("Gap_Now_%",0) or 0
            bsc    = r.get("Bars_Since_Reclaim",99)
            volx   = r.get("Vol_x_Avg",0) or 0
            edist  = r.get("Dist_EMA20_%",0) or 0
            rsi    = r.get("RSI",0) or 0
            rows_e += (
                f'<tr style="background:{bg}">'
                f'<td style="padding:6px 11px;font-size:12px;font-weight:700">{ticker}</td>'
                f'<td style="padding:6px 11px;font-size:12px">${float(price):.2f}</td>'
                f'<td style="padding:6px 11px;font-size:12px;font-weight:700;'
                f'background:#166534;color:#fff;text-align:center">{float(score):.0f}</td>'
                f'<td style="padding:6px 11px;font-size:12px">{float(gapnow):.2f}%</td>'
                f'<td style="padding:6px 11px;font-size:12px;color:'
                f'{"#22c55e" if bsc==0 else "#94a3b8"};font-weight:700">'
                f'{bsc}d</td>'
                f'<td style="padding:6px 11px;font-size:12px">{float(volx):.2f}×</td>'
                f'<td style="padding:6px 11px;font-size:12px">{float(edist):+.2f}%</td>'
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
  📊 SMA50 Compression to SMA150 + Bullish Reclaim
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
            f"SMA50 Compression to SMA150 + Bullish Reclaim — {datetime.today().strftime('%Y-%m-%d')}",
            f"{cnt} matches",
            "="*60,
        ]
        if rl:
            for r in rl[:50]:
                ticker = r.get("Ticker","—")
                price  = r.get("Price",0) or 0
                score  = r.get("Score",0) or 0
                gapnow = r.get("Gap_Now_%",0) or 0
                bsc    = r.get("Bars_Since_Reclaim",0) or 0
                plain_lines.append(
                    f"{ticker:<7} ${float(price):.2f}  Score:{float(score):.0f}  "
                    f"Gap:{float(gapnow):.2f}%  Reclaim:{bsc}d ago"
                )
        else:
            plain_lines.append("No matches today")
        plain_lines.append("\nFull results in CSV attachment.")
        plain_e = "\n".join(plain_lines)

        subj = (f"📊 SMA50 Compression + Reclaim — {cnt} signal{'s' if cnt!=1 else ''}"
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
        df_p  = r["_df"].tail(80).copy()
        ema20 = r["_ema20"].reindex(df_p.index)
        sma50 = r["_sma50"].reindex(df_p.index)
        sma150= r["_sma150"].reindex(df_p.index)
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

        # MAs (matching chart colours from the ZGN screenshot)
        ax.plot(range(n_p), ema20.values,  color="#34d399", lw=1.6, label="EMA20 🟢", zorder=5)
        ax.plot(range(n_p), sma50.values,  color="#3b82f6", lw=1.8, label="SMA50 🔵", zorder=4)
        ax.plot(range(n_p), sma150.values, color="#f472b6", lw=1.8, ls="-.", label="SMA150 🩷", zorder=4)

        # Shade the SMA50/SMA150 compression zone
        ax.fill_between(range(n_p), sma50.values, sma150.values,
                        where=(sma50.values >= sma150.values),
                        color="#a78bfa", alpha=0.08, zorder=1,
                        label="SMA50-SMA150 gap")

        # Mark the reclaim bar
        rb = r["_reclaim_bar"] - off
        if 0 <= rb < n_p:
            ax.axvline(rb, color="#22c55e", lw=1.8, ls="--", alpha=0.9)
            ax.scatter([rb],[float(df_p["Close"].iloc[rb])],
                       color="#22c55e", s=180, zorder=8, marker="^",
                       label=f"SMA50 Reclaim {r['Reclaim_Date']}")

        tick_step = max(1, n_p//8)
        ax.set_xticks(range(0, n_p, tick_step))
        ax.set_xticklabels(
            [df_p.index[i].strftime("%m/%d") for i in range(0,n_p,tick_step)],
            color="#94a3b8", fontsize=7)
        ax.set_xlim(-0.5, n_p-0.5)
        ax.set_title(
            f"{r['Ticker']}  ${r['Price']:.2f}  |  Score {r['Score']}/100  |  "
            f"Gap {r['Gap_Now_%']:.2f}% (narrowed {r['Gap_Narrowed_%']:+.2f}%)  |  "
            f"Reclaim {r['Reclaim_Date']} ({r['Bars_Since_Reclaim']}d ago)  "
            f"Vol {r['Vol_x_Avg']:.1f}×  |  RSI {r['RSI']:.0f}",
            color="#e2e8f0", fontsize=8, fontweight="bold", pad=6)
        ax.tick_params(colors="#94a3b8", labelsize=7)
        for sp_ in ax.spines.values(): sp_.set_edgecolor("#1e3a5f")
        ax.legend(loc="upper left", facecolor="#1e293b",
                  labelcolor="#e2e8f0", fontsize=7, framealpha=0.9)
        ax.grid(color="#1e3a5f", ls="--", lw=0.4, alpha=0.4, axis="y")

    plt.suptitle(
        f"SMA50 Compression to SMA150 + Bullish Reclaim  ·  "
        f"{datetime.today().strftime('%Y-%m-%d')}\n"
        f"🟢 EMA20  🔵 SMA50  🩷 SMA150  ▲ = SMA50 Reclaim Bar  🟣 shaded = compression zone",
        color="#60a5fa", fontsize=10, fontweight="bold", y=1.001)
    plt.tight_layout()
    cp = os.path.join(out_dir, f"sma50_compression_chart_{ts}.png")
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
  📋 PATTERN EXPLAINED  (ZGN chart example)

  C1  SMA50 COMPRESSION TOWARD SMA150  (the red-circled zone)
      SMA50 (🔵 blue) is still ABOVE SMA150 (🩷 pink)
      But SMA50 has been DESCENDING, narrowing the gap
      Current gap is very tight (<= 3% by default)
      = Medium-term trend decelerating into long-term support
        — a coil that often precedes a fresh breakout

  C2  PRICE CLOSES BACK ABOVE SMA50  (the trigger)
      Bar[-2] close < SMA50   ← was below SMA50 yesterday
      Bar[-1] close >= SMA50  ← closed above SMA50 today
      = Buyers stepped in right at the compression zone

  C3  GOOD VOLUME ON THE RECLAIM
      Reclaim bar volume >= 1.3× the 20-day average
      = Real conviction behind the bounce, not a drift

  WHY THIS WORKS:
      When SMA50 compresses INTO SMA150 (rather than crossing
      below it) and price snaps back above SMA50 on volume,
      it often marks the moment all the moving averages
      (EMA20, SMA50, SMA150) turn up together — exactly the
      structure visible in the ZGN chart's circled zone.

  💡 BEST SETUPS
  Gap_Now_% < 1%          extremely tight coil = biggest potential
  Gap_Narrowed_% large     significant compression occurred
  Bars_Since_Reclaim = 0   fresh reclaim today = earliest entry
  Vol_x_Avg > 2×           strong institutional participation
  RSI 45–65                healthy, not overbought

  ⚙️  TUNE IF 0 RESULTS
  compression_pct         3 → 5
  min_gap_narrowing_pct    1 → 0.5
  convergence_lookback    15 → 25
  vol_mult               1.3 → 1.0
  rsi_min                 35 → 25
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
""")
