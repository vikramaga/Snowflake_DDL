# ============================================================
# NASDAQ — Bullish Engulfing on SMA50/SMA20 Retest + Above SMA150
# ============================================================
#
# EXACT PATTERN:
#
#  C1 — PRICE ABOVE SMA150  (bull structure)
#      Current price > SMA150 (long-term trend intact)
#      SMA50 > SMA150 (medium-term above long-term)
#
#  C2 — RETEST OF SMA50 OR SMA20  (pullback to support)
#      In the last retest_lookback bars, the pattern must
#      occur AT or NEAR SMA50 or SMA20:
#        The red candle low must touch SMA50 or SMA20
#        within touch_pct% of that level
#
#  C3 — BULLISH ENGULFING CANDLE PATTERN
#      Two-candle pattern (within last cross_lookback bars):
#        Bar[i]   = RED candle  (close < open)
#                   Low touched SMA50 or SMA20 (the retest)
#        Bar[i+1] = GREEN candle  (close > open)
#                   Open <= Bar[i] close  (opens at/below red close)
#                   Close >= Bar[i] open  (closes at/above red open)
#                   = Green candle BODY fully engulfs red body
#
#  C4 — VOLUME ON GREEN BAR > RED BAR VOLUME
#      Volume on the engulfing green bar must be greater
#      than the volume on the red candle it engulfs
#      = Strong buying conviction behind the engulf
#
#  LOGIC FLOW:
#    Uptrend (price > SMA150) → Pullback touches SMA50 or SMA20
#    → Red candle forms at the support level
#    → Next green candle FULLY ENGULFS the red body
#    → On HIGHER VOLUME than the red candle
#    = Classic institutional absorption at key support
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
    "history_days"               : 300,

    # ── MA periods ────────────────────────────────────────────
    "sma20_period"               : 20,
    "sma50_period"               : 50,
    "sma150_period"              : 150,

    # ── C1: Bull structure ────────────────────────────────────
    "require_price_above_sma150" : True,
    "require_sma50_above_sma150" : True,

    # ── C2: Retest zone ───────────────────────────────────────
    # Red candle low must come within this % of SMA50 or SMA20
    "touch_pct"                  : 3.0,

    # ── C3: Engulfing pattern ─────────────────────────────────
    # How many recent bars to scan for the pattern
    "lookback_bars"              : 10,
    # How many bars ago the green engulf can be (max)
    "max_bars_since_engulf"      : 5,

    # ── C4: Volume ────────────────────────────────────────────
    # Green bar volume must be > red bar volume
    "require_vol_gt_red"         : True,
    # Additional: green vol must also be above N-day average
    "vol_avg_bars"               : 20,
    "min_vol_mult"               : 0.8,   # green vol >= 0.8x avg (basic filter)

    # ── Filters ───────────────────────────────────────────────
    "min_avg_volume"             : 80_000,
    "min_price"                  : 1.0,

    "batch_size"                 : 50,
    "batch_sleep"                : 1.5,
}

# ── Indicators ───────────────────────────────────────────────
def calc_rsi(close, period=14):
    delta = close.diff()
    gain  = delta.clip(lower=0)
    loss  = -delta.clip(upper=0)
    avg_g = gain.ewm(alpha=1/period, adjust=False).mean()
    avg_l = loss.ewm(alpha=1/period, adjust=False).mean()
    rs    = avg_g / avg_l.replace(0, np.nan)
    return 100 - (100 / (1 + rs))

def calc_macd_hist(close, fast=12, slow=26, signal=9):
    ema_f = close.ewm(span=fast, adjust=False).mean()
    ema_s = close.ewm(span=slow, adjust=False).mean()
    macd  = ema_f - ema_s
    sig   = macd.ewm(span=signal, adjust=False).mean()
    return macd - sig

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
    C1: price > SMA150, SMA50 > SMA150
    C2: red candle low touched SMA50 or SMA20 (within touch_pct%)
    C3: next green candle fully engulfs the red body
    C4: green candle volume > red candle volume
    """
    df      = df.copy(); df.index = pd.to_datetime(df.index)
    n       = len(df)
    price   = float(df["Close"].iloc[-1])
    avg_vol = float(df["Volume"].tail(CFG["vol_avg_bars"]).mean())

    if price   < CFG["min_price"]:      return None
    if avg_vol < CFG["min_avg_volume"]: return None
    if n < CFG["sma150_period"] + 10:   return None

    # ── Compute MAs ───────────────────────────────────────────
    sma20_s  = df["Close"].rolling(CFG["sma20_period"]).mean()
    sma50_s  = df["Close"].rolling(CFG["sma50_period"]).mean()
    sma150_s = df["Close"].rolling(CFG["sma150_period"]).mean()
    rsi_s    = calc_rsi(df["Close"])
    macdh_s  = calc_macd_hist(df["Close"])

    cur_sma20  = float(sma20_s.iloc[-1])
    cur_sma50  = float(sma50_s.iloc[-1])
    cur_sma150 = float(sma150_s.iloc[-1])
    cur_rsi    = float(rsi_s.iloc[-1])  if not np.isnan(rsi_s.iloc[-1])  else 50
    cur_macdh  = float(macdh_s.iloc[-1]) if not np.isnan(macdh_s.iloc[-1]) else 0

    if any(np.isnan([cur_sma20, cur_sma50, cur_sma150])): return None

    # ─────────────────────────────────────────────────────────
    # C1: BULL STRUCTURE
    # ─────────────────────────────────────────────────────────
    if CFG["require_price_above_sma150"] and price < cur_sma150: return None
    if CFG["require_sma50_above_sma150"] and cur_sma50 < cur_sma150: return None

    # ─────────────────────────────────────────────────────────
    # C2 + C3 + C4: SCAN FOR BULLISH ENGULFING AT SMA RETEST
    # ─────────────────────────────────────────────────────────
    lb        = CFG["lookback_bars"]
    touch_pct = CFG["touch_pct"] / 100
    max_bse   = CFG["max_bars_since_engulf"]

    search_start = max(1, n - lb - 1)

    found        = False
    red_bar      = None
    green_bar    = None
    touched_sma  = None
    touched_level= None

    for i in range(search_start, n - 1):
        # ── Red candle ────────────────────────────────────────
        red_o   = float(df["Open"].iloc[i])
        red_c   = float(df["Close"].iloc[i])
        red_l   = float(df["Low"].iloc[i])
        red_vol = float(df["Volume"].iloc[i])

        # Must be RED
        if red_c >= red_o: continue

        s20_i  = float(sma20_s.iloc[i])  if not np.isnan(sma20_s.iloc[i])  else np.nan
        s50_i  = float(sma50_s.iloc[i])  if not np.isnan(sma50_s.iloc[i])  else np.nan
        if np.isnan(s20_i) or np.isnan(s50_i): continue

        # ── C2: Red candle LOW must touch SMA50 or SMA20 ─────
        dist_s50 = abs(red_l - s50_i) / s50_i if s50_i > 0 else np.inf
        dist_s20 = abs(red_l - s20_i) / s20_i if s20_i > 0 else np.inf

        touch_s50 = dist_s50 <= touch_pct
        touch_s20 = dist_s20 <= touch_pct

        if not touch_s50 and not touch_s20: continue

        # Prefer SMA50 touch (stronger support); fall back to SMA20
        if touch_s50:
            tsma  = "SMA50"
            tlvl  = s50_i
        else:
            tsma  = "SMA20"
            tlvl  = s20_i

        # ── C3: Green engulfing candle ────────────────────────
        j       = i + 1
        grn_o   = float(df["Open"].iloc[j])
        grn_c   = float(df["Close"].iloc[j])
        grn_vol = float(df["Volume"].iloc[j])

        # Must be GREEN
        if grn_c <= grn_o: continue

        # ENGULFING: green body fully covers red body
        # green open <= red close  AND  green close >= red open
        if grn_o > red_c: continue   # green opens above red close
        if grn_c < red_o: continue   # green closes below red open

        # ── C4: Volume on green > volume on red ───────────────
        if CFG["require_vol_gt_red"] and grn_vol <= red_vol: continue

        # Additional: green vol >= min_vol_mult × average
        if grn_vol < CFG["min_vol_mult"] * avg_vol: continue

        # Recency check: green bar within max_bars_since_engulf of today
        bars_since = n - 1 - j
        if bars_since > max_bse: continue

        # Keep most recent valid pattern
        if green_bar is None or j > green_bar:
            found         = True
            red_bar       = i
            green_bar     = j
            touched_sma   = tsma
            touched_level = tlvl

    if not found: return None

    # ── Metrics ───────────────────────────────────────────────
    red_o   = float(df["Open"].iloc[red_bar])
    red_c   = float(df["Close"].iloc[red_bar])
    red_h   = float(df["High"].iloc[red_bar])
    red_l   = float(df["Low"].iloc[red_bar])
    red_vol = float(df["Volume"].iloc[red_bar])
    red_date= df.index[red_bar].strftime("%Y-%m-%d")

    grn_o   = float(df["Open"].iloc[green_bar])
    grn_c   = float(df["Close"].iloc[green_bar])
    grn_h   = float(df["High"].iloc[green_bar])
    grn_l   = float(df["Low"].iloc[green_bar])
    grn_vol = float(df["Volume"].iloc[green_bar])
    grn_date= df.index[green_bar].strftime("%Y-%m-%d")

    bars_since_engulf = n - 1 - green_bar

    red_body_size  = abs(red_c  - red_o)  / red_o  * 100
    grn_body_size  = abs(grn_c  - grn_o)  / grn_o  * 100
    vol_ratio      = grn_vol / red_vol if red_vol > 0 else 0
    touch_depth_pct= abs(red_l - touched_level) / touched_level * 100
    engulf_pct     = (grn_c - red_c) / red_body_size if red_body_size > 0 else 0

    dist_sma20_pct = (price - cur_sma20)  / cur_sma20  * 100 if cur_sma20  > 0 else 0
    dist_sma50_pct = (price - cur_sma50)  / cur_sma50  * 100 if cur_sma50  > 0 else 0
    dist_sma150_pct= (price - cur_sma150) / cur_sma150 * 100 if cur_sma150 > 0 else 0

    # Label if both SMA50 and SMA20 were touched
    s20_i  = float(sma20_s.iloc[red_bar]) if not np.isnan(sma20_s.iloc[red_bar]) else np.nan
    s50_i  = float(sma50_s.iloc[red_bar]) if not np.isnan(sma50_s.iloc[red_bar]) else np.nan
    both_touched = (not np.isnan(s20_i) and abs(red_l - s20_i)/s20_i <= touch_pct and
                    not np.isnan(s50_i) and abs(red_l - s50_i)/s50_i <= touch_pct)
    touched_label = "SMA50 + SMA20" if both_touched else touched_sma

    # ── Score (0-100) ─────────────────────────────────────────
    score = 0
    # Volume ratio (0-30): higher green vs red = stronger absorption
    score += min(30, int(vol_ratio * 10))
    # Engulf freshness (0-25): today = 25
    score += max(0, 25 - bars_since_engulf * 5)
    # Touch precision (0-20): tighter = cleaner retest
    score += max(0, 20 - int(touch_depth_pct * 5))
    # Green body size vs red (0-15): larger engulf = stronger
    score += min(15, int(grn_body_size / max(red_body_size, 0.1) * 5))
    # MACD positive (0-5): momentum confirming
    score += 5 if cur_macdh > 0 else 0
    # Both SMA touched (0-5 bonus)
    score += 5 if both_touched else 0
    score = min(100, max(0, score))

    # Tier
    if touched_label == "SMA50 + SMA20":
        tier = 1; tier_label = "🏆 TIER 1 — SMA50+SMA20 Confluence Engulf"
    elif touched_sma == "SMA50":
        tier = 2; tier_label = "🥈 TIER 2 — SMA50 Engulf"
    else:
        tier = 3; tier_label = "🥉 TIER 3 — SMA20 Engulf"

    return {
        "Ticker"           : sym,
        "Price"            : round(price, 2),
        "Score"            : score,
        "Tier"             : tier,
        "Tier_Label"       : tier_label,
        # Retest + engulf
        "Touched_SMA"      : touched_label,
        "Touch_Depth_%"    : round(touch_depth_pct, 2),
        "Red_Date"         : red_date,
        "Red_Open"         : round(red_o, 2),
        "Red_Close"        : round(red_c, 2),
        "Red_Low"          : round(red_l, 2),
        "Red_Vol"          : int(red_vol),
        "Red_Body_%"       : round(red_body_size, 2),
        "Green_Date"       : grn_date,
        "Green_Open"       : round(grn_o, 2),
        "Green_Close"      : round(grn_c, 2),
        "Green_Vol"        : int(grn_vol),
        "Green_Body_%"     : round(grn_body_size, 2),
        "Vol_Ratio"        : round(vol_ratio, 2),
        "Bars_Since_Engulf": bars_since_engulf,
        # MAs
        "SMA20"            : round(cur_sma20, 2),
        "SMA50"            : round(cur_sma50, 2),
        "SMA150"           : round(cur_sma150, 2),
        "Dist_SMA20_%"     : round(dist_sma20_pct, 2),
        "Dist_SMA50_%"     : round(dist_sma50_pct, 2),
        "Dist_SMA150_%"    : round(dist_sma150_pct, 2),
        # Indicators
        "RSI"              : round(cur_rsi, 1),
        "MACD_Hist"        : round(cur_macdh, 4),
        "Avg_Vol_20d"      : int(avg_vol),
        # internals
        "_df"      : df,
        "_sma20"   : sma20_s,
        "_sma50"   : sma50_s,
        "_sma150"  : sma150_s,
        "_red_bar" : red_bar,
        "_grn_bar" : green_bar,
    }

# ── Live print ────────────────────────────────────────────────
LIVE_COLS = ["Ticker","Price","Score","Tier_Label",
             "Touched_SMA","Touch_Depth_%","Red_Date","Green_Date",
             "Vol_Ratio","Bars_Since_Engulf","RSI"]
_CW = {"Ticker":8,"Price":10,"Score":7,"Tier_Label":42,
       "Touched_SMA":16,"Touch_Depth_%":14,"Red_Date":12,"Green_Date":12,
       "Vol_Ratio":11,"Bars_Since_Engulf":18,"RSI":6}
_CF = {"Price":"${:.2f}","Score":"{:.0f}","Touch_Depth_%":"{:.2f}%",
       "Vol_Ratio":"{:.2f}×","Bars_Since_Engulf":"{:.0f}d","RSI":"{:.1f}"}
_hdr_done = False

def _live_header():
    global _hdr_done
    if _hdr_done: return
    sep = "━"*200
    print(f"\n{sep}")
    print("  📊  LIVE MATCHES  —  Bullish Engulfing on SMA50/SMA20 Retest + Above SMA150")
    print(sep)
    print("".join(f"  {c:<{_CW.get(c,10)}}" for c in LIVE_COLS))
    print("  "+"─"*198)
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
        print(f"  ✅ {s}: ${p:.2f}  SMA50=${s50:.2f}  SMA150=${s150:.2f}  {d.index[-1].date()}")
print()

# ── Diagnostic ────────────────────────────────────────────────
print("━"*65)
print("  STEP 2  DIAGNOSTIC")
print("━"*65+"\n")
DIAG = ["AAPL","NVDA","MSFT","AMD","PLTR","META","CRWD","AVGO","MU","AXON"]
diag_data = download(DIAG, CFG["history_days"])
print(f"  Downloaded {len(diag_data)}/{len(DIAG)}\n")
print(f"  {'SYM':<7} {'PRICE':>8}  {'>S150':>6}  {'S50>S150':>9}  RESULT")
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
            print(f"  {sym:<7} ${p:>7.2f}  {t(p>s150):>6}  {t(s50>s150):>9}  "
                  f"✅ {r['Touched_SMA']} {r['Green_Date']}")
        else:
            print(f"  {sym:<7} ${p:>7.2f}  {t(p>s150):>6}  {t(s50>s150):>9}  ❌")
    except Exception as e:
        print(f"  {sym:<7} error: {e}")

print(f"""
  Pattern:
    C1  Bull structure  : price > SMA150  AND  SMA50 > SMA150
    C2  Retest          : red candle LOW within {CFG['touch_pct']}% of SMA50 or SMA20
    C3  Engulfing       : next green candle open <= red close
                          AND green close >= red open
    C4  Volume          : green candle volume > red candle volume

  Tune if mostly ❌:
    touch_pct            3 → 5     (wider retest zone)
    lookback_bars       10 → 15
    max_bars_since_engulf 5 → 8
    min_vol_mult        0.8 → 0.5
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

if not results:
    print("\n  No matches. Try relaxing:")
    print("   touch_pct              3 → 5")
    print("   lookback_bars         10 → 15")
    print("   max_bars_since_engulf  5 → 8")
    print("   min_vol_mult          0.8 → 0.5")

# Sort by tier then score
results.sort(key=lambda x: (x["Tier"], -x["Score"]))

# ── Always build df_out and save/email (even if 0 results) ────
ts      = datetime.today().strftime("%Y%m%d_%H%M")
out_dir = os.environ.get("GITHUB_WORKSPACE", os.getcwd())

COLS = [
    "Ticker","Price","Score","Tier","Tier_Label",
    "Touched_SMA","Touch_Depth_%",
    "Red_Date","Red_Open","Red_Close","Red_Low","Red_Vol","Red_Body_%",
    "Green_Date","Green_Open","Green_Close","Green_Vol","Green_Body_%",
    "Vol_Ratio","Bars_Since_Engulf",
    "SMA20","SMA50","SMA150",
    "Dist_SMA20_%","Dist_SMA50_%","Dist_SMA150_%",
    "RSI","MACD_Hist","Avg_Vol_20d",
]
df_out = pd.DataFrame([{k:v for k,v in r.items() if not k.startswith("_")}
                        for r in results]) if results else pd.DataFrame(columns=COLS)
if not df_out.empty:
    df_out = df_out[[c for c in COLS if c in df_out.columns]]
    df_out.reset_index(drop=True, inplace=True)

FMT = {
    "Price"            : lambda v: f"${v:.2f}",
    "Score"            : lambda v: f"{v:.0f}",
    "Touch_Depth_%"    : lambda v: f"{v:.2f}%",
    "Red_Open"         : lambda v: f"${v:.2f}",
    "Red_Close"        : lambda v: f"${v:.2f}",
    "Red_Low"          : lambda v: f"${v:.2f}",
    "Red_Vol"          : lambda v: f"{v:,.0f}",
    "Red_Body_%"       : lambda v: f"{v:.2f}%",
    "Green_Open"       : lambda v: f"${v:.2f}",
    "Green_Close"      : lambda v: f"${v:.2f}",
    "Green_Vol"        : lambda v: f"{v:,.0f}",
    "Green_Body_%"     : lambda v: f"{v:.2f}%",
    "Vol_Ratio"        : lambda v: f"{v:.2f}×",
    "Bars_Since_Engulf": lambda v: f"{int(v)}d",
    "SMA20"            : lambda v: f"${v:.2f}",
    "SMA50"            : lambda v: f"${v:.2f}",
    "SMA150"           : lambda v: f"${v:.2f}",
    "Dist_SMA20_%"     : lambda v: f"{v:+.2f}%",
    "Dist_SMA50_%"     : lambda v: f"{v:+.2f}%",
    "Dist_SMA150_%"    : lambda v: f"{v:+.2f}%",
    "RSI"              : lambda v: f"{v:.1f}",
    "MACD_Hist"        : lambda v: f"{v:.4f}",
    "Avg_Vol_20d"      : lambda v: f"{v:,.0f}",
}

def fmt_v(col, val):
    if val is None or (isinstance(val, float) and np.isnan(val)): return "—"
    try:
        if col in FMT: return FMT[col](val)
    except Exception: pass
    return str(val) if str(val) not in ("nan","None","") else "—"

TIER_COLORS = {1:"#22c55e", 2:"#3b82f6", 3:"#f59e0b"}
TIER_ICONS  = {1:"🏆", 2:"🥈", 3:"🥉"}

# ── Save CSV ──────────────────────────────────────────────────
fpath = os.path.join(out_dir, f"bullish_engulfing_sma_{ts}.csv")
df_out.to_csv(fpath, index=False)
print(f"\n  💾 CSV → {fpath}")
tv = os.path.join(out_dir, f"tv_bullish_engulfing_{ts}.txt")
with open(tv,"w") as f:
    f.write(f"###Bullish Engulfing SMA Retest {datetime.today().strftime('%Y-%m-%d')}\n")
    for r in results: f.write(f"NASDAQ:{r['Ticker']}\n")
print(f"  📋 TradingView → {tv}")

# ── Display (Notebook) ────────────────────────────────────────
if _IN_NOTEBOOK and results:
    DISP = ["Ticker","Price","Score","Tier_Label","Touched_SMA",
            "Touch_Depth_%","Red_Date","Green_Date",
            "Vol_Ratio","Bars_Since_Engulf","SMA50","SMA150","RSI"]
    DISP = [c for c in DISP if c in df_out.columns]

    def make_tier_table(tier_rows, tier_num):
        if not tier_rows: return ""
        tc = TIER_COLORS[tier_num]; ico = TIER_ICONS[tier_num]
        th = "".join(
            f'<th style="background:#0f172a;color:#e2e8f0;padding:9px 12px;'
            f'font-size:11px;font-weight:700;border-bottom:2px solid {tc};white-space:nowrap">{c}</th>'
            for c in DISP
        )
        rows = ""
        for i, r in enumerate(tier_rows):
            bg = "#fff" if i%2==0 else "#f0f9ff"
            tds = ""
            for col in DISP:
                raw = r.get(col); disp = fmt_v(col, raw); sty = ""
                if col == "Score":
                    try:
                        v = float(raw); g = int(min(220, 80+v*1.4))
                        sty = f"background:rgb(20,{g},60);color:#fff;font-weight:700;text-align:center"
                    except Exception: pass
                elif col == "Vol_Ratio":
                    try:
                        v = float(str(raw).replace("×",""))
                        sty = ("color:#22c55e;font-weight:800" if v>=2 else
                               "color:#86efac;font-weight:600" if v>=1.5 else "")
                    except Exception: pass
                elif col == "Touch_Depth_%":
                    try:
                        v = float(str(raw).replace("%",""))
                        sty = "color:#22c55e;font-weight:700" if v<=1 else ""
                    except Exception: pass
                elif col == "Bars_Since_Engulf":
                    try:
                        v = int(str(raw).replace("d",""))
                        sty = "color:#22c55e;font-weight:700;text-align:center" if v==0 else "text-align:center"
                    except Exception: pass
                tds += f'<td style="padding:7px 12px;font-size:12px;border-bottom:1px solid #e2e8f0;white-space:nowrap;{sty}">{disp}</td>'
            rows += f'<tr style="background:{bg}">{tds}</tr>\n'
        return f"""
<div style="margin:10px 0">
  <div style="background:linear-gradient(90deg,{tc}22,#0f172a);border-left:4px solid {tc};
              border-radius:6px 6px 0 0;padding:10px 18px;display:flex;align-items:center;gap:10px">
    <span style="font-size:18px">{ico}</span>
    <span style="color:#f1f5f9;font-size:14px;font-weight:700">{TIER_ICONS[tier_num]} Tier {tier_num} — {len(tier_rows)} stocks</span>
  </div>
  <div style="overflow-x:auto;border:1px solid #e2e8f0;border-top:none;border-radius:0 0 8px 8px">
    <table style="border-collapse:collapse;width:100%;min-width:700px">
      <thead><tr>{th}</tr></thead><tbody>{rows}</tbody>
    </table>
  </div>
</div>"""

    ticker_csv_str = ",".join(r["Ticker"] for r in results)
    t1=sum(1 for r in results if r["Tier"]==1)
    t2=sum(1 for r in results if r["Tier"]==2)
    t3=sum(1 for r in results if r["Tier"]==3)

    header = f"""
<div style="background:linear-gradient(135deg,#0f172a,#1e3a5f);border-radius:10px;
            padding:18px 24px;margin-bottom:8px;font-family:'Segoe UI',Arial,sans-serif">
  <h2 style="margin:0;color:#60a5fa;font-size:20px;font-weight:700">
    📈 Bullish Engulfing on SMA50/SMA20 Retest + Above SMA150
  </h2>
  <p style="margin:6px 0 0;color:#94a3b8;font-size:12px">
    {datetime.today().strftime('%Y-%m-%d %H:%M')} &nbsp;·&nbsp;
    <b style="color:#22c55e">{len(results)} matches</b> &nbsp;·&nbsp;
    🏆{t1} SMA50+SMA20 &nbsp; 🥈{t2} SMA50 &nbsp; 🥉{t3} SMA20
  </p>
</div>
<div style="background:#0f172a;border-radius:8px;padding:14px 16px;margin:8px 0;
            border-left:4px solid #22c55e;">
  <p style="margin:0 0 4px;color:#94a3b8;font-size:11px;font-weight:600;
             text-transform:uppercase;letter-spacing:0.05em">
    📋 Stock List (CSV) — copy &amp; paste
  </p>
  <p style="margin:0;color:#22c55e;font-size:13px;font-weight:700;
             font-family:'Courier New',monospace;word-break:break-all">
    {ticker_csv_str}
  </p>
</div>"""
    tiers = "".join(make_tier_table([r for r in results if r["Tier"]==t], t) for t in [1,2,3])
    display_html(header + tiers)

elif results:
    for t_num in [1,2,3]:
        t_rows = [r for r in results if r["Tier"]==t_num]
        if not t_rows: continue
        print(f"\n  {TIER_ICONS[t_num]}  Tier {t_num}  ({len(t_rows)} stocks)\n")
        CLI = ["Ticker","Price","Score","Touched_SMA","Touch_Depth_%",
               "Vol_Ratio","Bars_Since_Engulf","RSI"]
        CLI = [c for c in CLI if c in df_out.columns]
        col_w = {c: max(len(c), max(len(fmt_v(c,r.get(c))) for r in t_rows))+2 for c in CLI}
        top = "┬".join("─"*col_w[c] for c in CLI)
        sep = "┼".join("─"*col_w[c] for c in CLI)
        bot = "┴".join("─"*col_w[c] for c in CLI)
        hdr = "│".join(c.center(col_w[c]) for c in CLI)
        print(f"  ┌{top}┐\n  │{hdr}│\n  ├{sep}┤")
        for i,r in enumerate(t_rows):
            cells = [fmt_v(c,r.get(c)).center(col_w[c]) for c in CLI]
            print(f"  │{'│'.join(cells)}│")
            if i<len(t_rows)-1: print(f"  ├{sep}┤")
        print(f"  └{bot}┘")

# ── Email ──────────────────────────────────────────────────────
def _send_email(rl, csv_path):
    gu = _GMAIL_USER; gp = _GMAIL_PASS; et = _EMAIL_TO
    if not gu:
        print("[Email] ❌  GMAIL_USER secret is empty"); return
    if not gp:
        print("[Email] ❌  GMAIL_PASS secret is empty"); return
    if not et:
        print("[Email] ❌  EMAIL_TO secret is empty"); return

    eto = [e.strip() for e in et.split(",") if e.strip()]
    cnt = len(rl)

    try:
        t1 = sum(1 for r in rl if r.get("Tier")==1)
        t2 = sum(1 for r in rl if r.get("Tier")==2)
        t3 = sum(1 for r in rl if r.get("Tier")==3)

        # ── Ticker CSV one-liner ──────────────────────────────
        ticker_csv = ",".join(r.get("Ticker","") for r in rl) if rl else "—"

        ticker_csv_html = f"""
<div style="margin:14px 0;padding:14px 16px;background:#0f172a;
            border-radius:8px;border-left:4px solid #22c55e;">
  <p style="margin:0 0 6px;color:#94a3b8;font-size:11px;font-weight:600;
             letter-spacing:0.05em;text-transform:uppercase">
    📋 Stock List — Copy &amp; paste into TradingView / Excel
  </p>
  <p style="margin:0;color:#22c55e;font-size:13px;font-weight:700;
             font-family:'Courier New',monospace;word-break:break-all;
             letter-spacing:0.04em">
    {ticker_csv}
  </p>
</div>"""

        print(f"[Email] Sending to {et}  ({cnt} results)...")

        th_e = "".join(
            f'<th style="background:#1e293b;color:#e2e8f0;padding:8px 11px;'
            f'font-size:11px;font-weight:700;border-bottom:2px solid #3b82f6;'
            f'white-space:nowrap">{c}</th>'
            for c in ["Ticker","Price","Score","Touched_SMA",
                      "Touch_Depth_%","Vol_Ratio","Bars_Since_Engulf","RSI"]
        )
        rows_e = ""
        for i, r in enumerate(rl[:50]):
            bg   = "#fff" if i%2==0 else "#f0f9ff"
            tc   = TIER_COLORS.get(r.get("Tier",3),"#f59e0b")
            rows_e += (
                f'<tr style="background:{bg}">'
                f'<td style="padding:6px 11px;font-size:12px;font-weight:700;color:{tc}">'
                f'{r.get("Ticker","—")}</td>'
                f'<td style="padding:6px 11px;font-size:12px">'
                f'${float(r.get("Price",0)):.2f}</td>'
                f'<td style="padding:6px 11px;font-size:12px;font-weight:700;'
                f'background:#166534;color:#fff;text-align:center">'
                f'{float(r.get("Score",0)):.0f}</td>'
                f'<td style="padding:6px 11px;font-size:12px;font-weight:600">'
                f'{r.get("Touched_SMA","—")}</td>'
                f'<td style="padding:6px 11px;font-size:12px">'
                f'{float(r.get("Touch_Depth_%",0)):.2f}%</td>'
                f'<td style="padding:6px 11px;font-size:12px;color:#22c55e;font-weight:700">'
                f'{float(r.get("Vol_Ratio",0)):.2f}×</td>'
                f'<td style="padding:6px 11px;font-size:12px;text-align:center;color:'
                f'{"#22c55e" if r.get("Bars_Since_Engulf",99)==0 else "#94a3b8"};font-weight:700">'
                f'{r.get("Bars_Since_Engulf",0)}d</td>'
                f'<td style="padding:6px 11px;font-size:12px">'
                f'{float(r.get("RSI",0)):.1f}</td>'
                f'</tr>'
            )

        no_res = "" if cnt else (
            '<tr><td colspan="8" style="padding:20px;text-align:center;'
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
      📊 Bullish Engulfing on SMA50/SMA20 Retest + Above SMA150
    </h1>
    <p style="margin:6px 0 0;color:#94a3b8;font-size:12px">
      {datetime.today().strftime('%Y-%m-%d %H:%M UTC')} &nbsp;·&nbsp;
      {cnt} match{'es' if cnt!=1 else ''} &nbsp;·&nbsp;
      🏆{t1} &nbsp; 🥈{t2} &nbsp; 🥉{t3}
    </p>
  </td></tr>
  <tr><td style="padding:16px">
    {ticker_csv_html}
    <div style="overflow-x:auto;border-radius:8px;border:1px solid #e2e8f0">
      <table style="border-collapse:collapse;width:100%;min-width:700px">
        <thead><tr>{th_e}</tr></thead>
        <tbody>{rows_e or no_res}</tbody>
      </table>
    </div>
    <p style="font-size:11px;color:#64748b;margin:10px 0 0">
      📎 CSV attached &nbsp;·&nbsp;
      <b>Touch_Depth_%</b> = how close red candle low came to SMA
      (lower=tighter) &nbsp;·&nbsp;
      <b>Vol_Ratio</b> = green ÷ red volume
    </p>
  </td></tr>
  <tr><td style="background:#f8fafc;padding:12px 28px;border-top:1px solid #e2e8f0;text-align:center">
    <p style="margin:0;color:#94a3b8;font-size:10px">
      ⚠️ Not financial advice &nbsp;·&nbsp; Auto-generated by GitHub Actions
    </p>
  </td></tr>
</table></td></tr></table></body></html>"""

        plain_e = "\n".join([
            f"Bullish Engulfing SMA Retest — {datetime.today().strftime('%Y-%m-%d')}",
            f"{cnt} matches  (🏆{t1} 🥈{t2} 🥉{t3})",
            "",
            f"STOCKS: {ticker_csv}",
            "",
            "="*60,
        ] + ([
            f"{r.get('Ticker','—'):<7} ${float(r.get('Price',0)):.2f}  "
            f"Score:{float(r.get('Score',0)):.0f}  "
            f"{r.get('Touched_SMA','—')}  "
            f"Vol:{float(r.get('Vol_Ratio',0)):.1f}x  "
            f"Engulf:{r.get('Bars_Since_Engulf',0)}d ago"
            for r in rl[:50]
        ] if rl else ["No matches today"]) + ["\n📎 CSV attached."])

        subj = (f"📊 Bull Engulf SMA — {cnt} signal{'s' if cnt!=1 else ''}"
                f"  (🏆{t1} 🥈{t2} 🥉{t3}) — {datetime.today().strftime('%Y-%m-%d')}")

        msg = MIMEMultipart("mixed")
        msg["Subject"] = subj; msg["From"] = gu; msg["To"] = ", ".join(eto)
        alt = MIMEMultipart("alternative")
        alt.attach(MIMEText(plain_e,"plain")); alt.attach(MIMEText(html_e,"html"))
        msg.attach(alt)

    except Exception as e:
        print(f"[Email] ❌  Failed to build email body: {type(e).__name__}: {e}"); return

    # Attach CSV + TV file
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
        print("         myaccount.google.com/apppasswords")
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
        df_p = r["_df"].tail(60).copy()
        n_p  = len(df_p)
        fn   = len(r["_df"]); off = fn - n_p

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

        ax.plot(range(n_p), s20.values,  color="#fbbf24", lw=1.3, ls="--", label="SMA20 🟡", zorder=4)
        ax.plot(range(n_p), s50.values,  color="#3b82f6", lw=1.6, label="SMA50 🔵", zorder=4)
        ax.plot(range(n_p), s150.values, color="#f472b6", lw=1.5, ls="-.", label="SMA150 🩷", zorder=4)

        # Shade touched SMA zone
        tsma = r.get("Touched_SMA","")
        vlvl = r.get("SMA50") if "SMA50" in tsma else r.get("SMA20")
        if vlvl:
            ax.axhspan(vlvl*0.97, vlvl*1.03, alpha=0.07,
                       color="#3b82f6" if "SMA50" in tsma else "#fbbf24", zorder=1)

        # Mark red (shakeout) and green (engulf) bars
        rb  = r["_red_bar"] - off
        gb  = r["_grn_bar"] - off
        if 0 <= rb < n_p:
            ax.scatter([rb],[float(df_p["Low"].iloc[rb])],
                       color="#ef4444",s=150,zorder=9,marker="v",label="Red Candle (retest)")
            ax.axvline(rb, color="#ef4444", lw=1.0, ls=":", alpha=0.5)
        if 0 <= gb < n_p:
            ax.scatter([gb],[float(df_p["Close"].iloc[gb])],
                       color="#22c55e",s=180,zorder=9,marker="^",
                       label=f"Bull Engulf {r['Green_Date']} Vol{r['Vol_Ratio']:.1f}×")
            ax.axvline(gb, color="#22c55e", lw=1.5, ls="--", alpha=0.8)

        tick_step = max(1, n_p//8)
        ax.set_xticks(range(0,n_p,tick_step))
        ax.set_xticklabels(
            [df_p.index[i].strftime("%m/%d") for i in range(0,n_p,tick_step)],
            color="#94a3b8", fontsize=7)
        ax.set_xlim(-0.5, n_p-0.5)
        ax.set_title(
            f"{r['Ticker']}  ${r['Price']:.2f}  |  Score {r['Score']}/100  |  "
            f"{r['Tier_Label']}  |  "
            f"Retest {r['Touched_SMA']} ({r['Touch_Depth_%']:.1f}%)  |  "
            f"Engulf {r['Green_Date']} ({r['Bars_Since_Engulf']}d ago)  "
            f"Vol {r['Vol_Ratio']:.1f}×  |  RSI {r['RSI']:.0f}",
            color="#e2e8f0", fontsize=8, fontweight="bold", pad=6)
        ax.tick_params(colors="#94a3b8", labelsize=7)
        for sp_ in ax.spines.values(): sp_.set_edgecolor("#1e3a5f")
        ax.legend(loc="upper left", facecolor="#1e293b",
                  labelcolor="#e2e8f0", fontsize=7, framealpha=0.9)
        ax.grid(color="#1e3a5f", ls="--", lw=0.4, alpha=0.4, axis="y")

    plt.suptitle(
        f"Bullish Engulfing on SMA50/SMA20 Retest + Above SMA150  ·  "
        f"{datetime.today().strftime('%Y-%m-%d')}\n"
        f"🟡 SMA20  🔵 SMA50  🩷 SMA150  ▼ Red Retest  ▲ Bull Engulf",
        color="#60a5fa", fontsize=10, fontweight="bold", y=1.001)
    plt.tight_layout()
    cp = os.path.join(out_dir, f"bullish_engulfing_chart_{ts}.png")
    plt.savefig(cp, dpi=150, bbox_inches="tight", facecolor="#0f172a")
    if _IN_NOTEBOOK: plt.show()
    else: plt.close()
    print(f"  📊 Chart → {cp}")

print("""
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  📋 PATTERN EXPLAINED

  C1  BULL STRUCTURE
      Price above SMA150 (long-term trend intact)
      SMA50 above SMA150 (medium-term bullish)

  C2  RETEST OF SMA50 OR SMA20
      A red candle's LOW came within 3% of SMA50 or SMA20
      = Price pulled back and tested a key support level

  C3  BULLISH ENGULFING  (very next candle)
      Green candle OPEN  <= Red candle CLOSE
      Green candle CLOSE >= Red candle OPEN
      = Green body completely covers/engulfs the red body
      Classic "institutional absorption" signal

  C4  HIGHER VOLUME on the green engulfing candle vs red
      = Strong buying conviction confirmed by volume

  TIERS (by which SMA was retested):
      🏆 Tier 1  Both SMA50 + SMA20 touched (strongest confluence)
      🥈 Tier 2  SMA50 only (strong support)
      🥉 Tier 3  SMA20 only (faster pullback)

  💡 BEST SETUPS
  Bars_Since_Engulf = 0     engulf happened today = fresh entry
  Vol_Ratio >= 2×            strong institutional buying
  Touch_Depth_% < 1%        precise SMA touch = clean setup
  Tier 1 (both SMAs)        maximum support confluence

  ⚙️  TUNE IF 0 RESULTS
  touch_pct              3 → 5
  lookback_bars         10 → 15
  max_bars_since_engulf  5 → 8
  min_vol_mult          0.8 → 0.5
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
""")
