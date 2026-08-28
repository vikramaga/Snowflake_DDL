# ============================================================
# NASDAQ — MA/VWAP Cluster Reclaim + RSI Confirmation Scanner (v1)
# ============================================================
#
# SIGNAL (all required):
#
#   1. STRUCTURE (evaluated on the signal day):
#      SMA50 > SMA150 (bullish structure), AND EMA8, JMA, SMA21,
#      and VWAP are ALL above SMA50 AND compressed tightly together
#      (max-min spread among them within cluster_compression_pct%
#      of price) — a "coiled" short-term cluster riding above the
#      longer-term trend. SMA50 sloping upward is scored as a bonus,
#      not a hard requirement ("preferably increasing slope").
#
#   2. TWO-CANDLE RECLAIM (scanned over the last
#      signal_lookback_days trading days, not just today):
#        Candle A (red)  — closed BELOW all four of JMA, EMA8,
#                           SMA21, AND VWAP
#        Candle B (green)— closed ABOVE all four of the same lines,
#                           with volume >= volume_multiplier (1.3x)
#                           times candle A's volume
#
#   3. RSI CONFIRMATION: RSI(14) just crossed above its own moving
#      average (a signal-line cross) AND RSI is above 50, both on
#      candle B.
#
# DATA — only 1 download per ticker: daily bars.
#
# OUTPUT:
#   Entry_Price = SMA21 value on the signal day (candle B)
#   Stop_Loss   = LOW of candle A (the red day)
#   Target      = Entry + risk_reward_ratio * (Entry - Stop_Loss)
#                 — a fixed 1:3 risk:reward by construction, not a
#                 swing-high search
#   Signal_Date = the exact calendar date the pattern fired (not a
#                 relative "Nd ago" — shown alongside Days_Since_Signal)
#
# SINGLE PASS — purely technical, no fundamentals fetch.
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
import matplotlib.dates as mdates

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
    "history_days"          : 450,   # daily bars — plenty for SMA150 + JMA warmup

    # ── Indicator periods ────────────────────────────────────────
    "ema8_period"            : 8,
    "sma21_period"           : 21,
    "sma50_period"           : 50,
    "sma150_period"          : 150,
    "vwap_period"            : 20,   # rolling volume-weighted moving average
    "jma_period"             : 13,
    "jma_phase"              : 40,

    # ── Step 1: cluster structure ───────────────────────────────
    "cluster_compression_pct": 3.0,  # max-min spread among {EMA8,JMA,SMA21,VWAP}
                                      # as % of price — must be within this
    "slope_lookback"         : 5,    # bars back to check SMA50 rising (scored
                                      # as a bonus, not a hard requirement)

    # ── Step 2: two-candle reclaim ───────────────────────────────
    "volume_multiplier"      : 1.3,  # candle B volume >= this x candle A volume
    "signal_lookback_days"   : 15,   # ~3 trading weeks — how far back to look
                                      # for the reclaim pattern, not just today

    # ── Step 3: RSI confirmation ─────────────────────────────────
    "rsi_period"             : 14,
    "rsi_avg_period"         : 9,    # period of RSI's own moving average
    "rsi_min_level"          : 50,

    # ── Step 4: risk:reward ──────────────────────────────────────
    "risk_reward_ratio"      : 3.0,  # Target = Entry + ratio * (Entry - Stop)

    # ── Filters ─────────────────────────────────────────────────
    "min_avg_volume"         : 80_000,
    "min_price"              : 2.0,

    "batch_size"             : 50,
    "batch_sleep"            : 1.5,
}

# ── Indicators ───────────────────────────────────────────────
def calc_jma(series, period=13, phase=40, power=2):
    """
    JMA (Jurik Moving Average) approximation — adaptive EMA with
    phase-based smoothing, using the corrected e2 update (steady-
    state gain 1.0, tracks price correctly).
    """
    n      = len(series)
    vals   = series.values.astype(float)
    result = np.full(n, np.nan)
    phase_ratio = phase / 100.0 + 1.5
    alpha = 2.0 / (period + 1.0)
    beta  = alpha * phase_ratio
    first_valid = 0
    for i in range(n):
        if not np.isnan(vals[i]):
            first_valid = i
            break
    e0 = e1 = e2 = vals[first_valid]
    result[first_valid] = e0
    for i in range(first_valid + 1, n):
        v   = vals[i]
        e0  = (1 - alpha) * e0 + alpha * v
        e1  = (v - e0) * (1 - beta) + beta * e1
        e2  = (1 - alpha) * e2 + alpha * (e0 + e1)
        result[i] = e2
    return pd.Series(result, index=series.index)

def calc_vwap_rolling(close, volume, period=20):
    """Rolling volume-weighted moving average (VWAP proxy on daily bars)."""
    pv = close * volume
    return pv.rolling(period).sum() / volume.rolling(period).sum()

# ── Uptrend test (reused across all 4 timeframes) ──────────────
def calc_rsi(close, period=14):
    """Standard Wilder-smoothed RSI."""
    delta = close.diff()
    gain  = delta.clip(lower=0)
    loss  = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1/period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1/period, min_periods=period, adjust=False).mean()
    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))

def check_cluster_structure(df, ema8, jma, sma21, vwap, sma50, sma150, idx, cfg):
    """
    STEP 1: at bar `idx` — SMA50 > SMA150, AND EMA8/JMA/SMA21/VWAP
    all above SMA50 and compressed tightly together (max-min spread
    within cluster_compression_pct% of price). SMA50 rising is
    scored as a bonus, not required.

    Returns (passed: bool, details: dict).
    """
    price = float(df["Close"].iloc[idx])
    s50, s150 = float(sma50.iloc[idx]), float(sma150.iloc[idx])
    vals = [float(ema8.iloc[idx]), float(jma.iloc[idx]),
            float(sma21.iloc[idx]), float(vwap.iloc[idx])]
    if any(np.isnan(v) for v in vals + [s50, s150, price]):
        return False, {}

    structure_ok = s50 > s150
    cluster_above_sma50 = all(v > s50 for v in vals)
    compression_pct = (max(vals) - min(vals)) / price * 100 if price > 0 else 999

    passed = structure_ok and cluster_above_sma50 and \
             (compression_pct <= cfg["cluster_compression_pct"])

    sb = cfg["slope_lookback"]
    slope_ok = False
    if idx - sb >= 0 and not np.isnan(sma50.iloc[idx-sb]):
        slope_ok = s50 > float(sma50.iloc[idx-sb])

    return passed, {
        "compression_pct": round(compression_pct, 2),
        "slope_ok": slope_ok,
        "sma50": round(s50, 2), "sma150": round(s150, 2),
    }

def find_cluster_reclaim_rsi_signals(df, ema8, jma, sma21, vwap, sma50, sma150,
                                       rsi, rsi_avg, cfg):
    """
    Scans the last `signal_lookback_days` trading days for the full
    3-step pattern (structure + 2-candle reclaim + RSI confirmation),
    all evaluated at each candidate day. Returns a list of hit dicts,
    most recent first.
    """
    n = len(df)
    lb = cfg["signal_lookback_days"]
    vol_mult = cfg["volume_multiplier"]
    rsi_min = cfg["rsi_min_level"]
    hits = []

    for back in range(0, lb):
        i = (n - 1) - back   # candle B (the green reclaim day)
        j = i - 1             # candle A (the red day)
        if j < 1: break

        # ── Step 1: structure on candle B ──────────────────────────
        struct_ok, struct_details = check_cluster_structure(
            df, ema8, jma, sma21, vwap, sma50, sma150, i, cfg)
        if not struct_ok:
            continue

        # ── Step 2: two-candle reclaim across all 4 lines ──────────
        open_A, close_A = float(df["Open"].iloc[j]), float(df["Close"].iloc[j])
        open_B, close_B = float(df["Open"].iloc[i]), float(df["Close"].iloc[i])
        vol_A, vol_B = float(df["Volume"].iloc[j]), float(df["Volume"].iloc[i])

        line_vals_A = [float(jma.iloc[j]), float(ema8.iloc[j]),
                       float(sma21.iloc[j]), float(vwap.iloc[j])]
        line_vals_B = [float(jma.iloc[i]), float(ema8.iloc[i]),
                       float(sma21.iloc[i]), float(vwap.iloc[i])]
        if any(np.isnan(v) for v in line_vals_A + line_vals_B):
            continue

        candle_A_red   = close_A < open_A
        candle_A_below = all(close_A < v for v in line_vals_A)
        candle_B_green = close_B > open_B
        candle_B_above = all(close_B > v for v in line_vals_B)
        vol_confirmed  = vol_B >= vol_mult * vol_A

        if not (candle_A_red and candle_A_below and candle_B_green
                and candle_B_above and vol_confirmed):
            continue

        # ── Step 3: RSI crosses above its own average, RSI > 50 ────
        r_cur, r_prev = rsi.iloc[i], rsi.iloc[i-1]
        ra_cur, ra_prev = rsi_avg.iloc[i], rsi_avg.iloc[i-1]
        if any(np.isnan(v) for v in [r_cur, r_prev, ra_cur, ra_prev]):
            continue
        rsi_cross = (r_prev <= ra_prev) and (r_cur > ra_cur)
        rsi_above_50 = r_cur > rsi_min
        if not (rsi_cross and rsi_above_50):
            continue

        hits.append({
            "idx": i, "j": j,
            "struct": struct_details,
            "vol_ratio": vol_B / vol_A if vol_A > 0 else 0,
            "rsi_cur": float(r_cur), "rsi_avg_cur": float(ra_cur),
            "close_A": close_A, "low_A": float(df["Low"].iloc[j]),
            "sma21_B": float(sma21.iloc[i]),
        })

    return hits

# ── Technical signal: cluster reclaim + RSI confirmation ─────────
def analyze_cluster_reclaim(sym, df_daily):
    """
    Returns dict with score and setup details, or None if no
    required condition is met anywhere in the lookback window.
    """
    if df_daily is None:
        return None

    price   = float(df_daily["Close"].iloc[-1])
    avg_vol = float(df_daily["Volume"].tail(20).mean())
    if price   < CFG["min_price"]:      return None
    if avg_vol < CFG["min_avg_volume"]: return None

    n = len(df_daily)
    if n < CFG["sma150_period"] + CFG["signal_lookback_days"] + 20:
        return None

    ema8   = df_daily["Close"].ewm(span=CFG["ema8_period"], adjust=False).mean()
    sma21  = df_daily["Close"].rolling(CFG["sma21_period"]).mean()
    sma50  = df_daily["Close"].rolling(CFG["sma50_period"]).mean()
    sma150 = df_daily["Close"].rolling(CFG["sma150_period"]).mean()
    vwap   = calc_vwap_rolling(df_daily["Close"], df_daily["Volume"], CFG["vwap_period"])
    jma    = calc_jma(df_daily["Close"], CFG["jma_period"], CFG["jma_phase"])
    rsi    = calc_rsi(df_daily["Close"], CFG["rsi_period"])
    rsi_avg = rsi.rolling(CFG["rsi_avg_period"]).mean()

    hits = find_cluster_reclaim_rsi_signals(
        df_daily, ema8, jma, sma21, vwap, sma50, sma150, rsi, rsi_avg, CFG)
    if not hits:
        return None

    sig = hits[0]   # most recent
    sig_idx = sig["idx"]
    days_since_signal = (n - 1) - sig_idx
    recent_signals = [
        {"date": df_daily.index[h["idx"]], "bars_ago": (n-1)-h["idx"]}
        for h in hits
    ]

    # ── Entry / Stop / Target ──────────────────────────────────────
    entry_price = sig["sma21_B"]           # SMA21 value on the signal day
    stop_loss   = sig["low_A"]             # low of the red candle
    if entry_price <= stop_loss:
        return None
    risk   = entry_price - stop_loss
    rr     = CFG["risk_reward_ratio"]
    target = entry_price + rr * risk       # fixed 1:3 by construction
    risk_pct = risk / entry_price * 100 if entry_price > 0 else 0

    # ── Score (0-100) ────────────────────────────────────────────
    score = 0
    reasons = []
    comp = sig["struct"]["compression_pct"]
    score += max(0, min(20, 20 - comp * 4))
    reasons.append(f"Compression{comp:.1f}%")
    if sig["struct"]["slope_ok"]:
        score += 10; reasons.append("SMA50Rising")
    vr = sig["vol_ratio"]
    score += min(20, (vr - 1.0) * 20)
    reasons.append(f"Vol{vr:.1f}x")
    rsi_margin = sig["rsi_cur"] - CFG["rsi_min_level"]
    score += min(20, rsi_margin)
    reasons.append(f"RSI{sig['rsi_cur']:.0f}")
    freshness_pts = max(0, 15 - days_since_signal)
    score += freshness_pts
    reasons.append(f"{days_since_signal}dAgo")
    score += 15   # base for clearing every gate
    score = round(min(100, max(0, score)))

    return {
        "Score"          : score,
        "Price"          : round(price, 2),
        "Entry_Price"    : round(entry_price, 2),
        "Stop_Loss"      : round(stop_loss, 2),
        "Target"         : round(target, 2),
        "Risk_%"         : round(risk_pct, 1),
        "Risk_Reward"    : rr,
        "Compression_%"  : comp,
        "Vol_Ratio"      : round(sig["vol_ratio"], 2),
        "RSI"            : round(sig["rsi_cur"], 1),
        "RSI_Avg"        : round(sig["rsi_avg_cur"], 1),
        "SMA50_Rising"   : sig["struct"]["slope_ok"],
        "Signal_Date"    : df_daily.index[sig_idx].strftime("%Y-%m-%d"),
        "Days_Since_Signal"  : days_since_signal,
        "Recent_Signal_Count": len(recent_signals),
        "Recent_Signals" : " | ".join(
            f"{s['date'].strftime('%Y-%m-%d')}" for s in recent_signals),
        "Flags"          : " | ".join(reasons),
        "_df_daily"      : df_daily,
        "_ema8"          : ema8, "_sma21": sma21, "_sma50": sma50,
        "_sma150"        : sma150, "_vwap": vwap, "_jma": jma,
        "_rsi"           : rsi, "_rsi_avg": rsi_avg,
    }

# ── Download: daily (→ also Weekly + Monthly via resample) ──────
def _clean(df, min_bars=200):
    if df is None or df.empty: return None
    need = [c for c in ["Open","High","Low","Close","Volume"] if c in df.columns]
    if not all(c in need for c in ["High","Low","Close","Volume"]): return None
    df = df[need].copy()
    df.index = pd.to_datetime(df.index)
    if hasattr(df.index,"tz") and df.index.tz:
        df.index = df.index.tz_localize(None)
    df.dropna(subset=["Close","Volume"], inplace=True)
    return df if (len(df)>=min_bars and float(df["Close"].iloc[-1])>0) else None

def download_daily(symbols, days):
    end = datetime.today(); start = end - timedelta(days=days)
    out = {}
    try:
        raw = yf.download(symbols, start=start.strftime("%Y-%m-%d"),
                          end=end.strftime("%Y-%m-%d"), group_by="ticker",
                          auto_adjust=True, actions=False,
                          threads=True, progress=False)
        if raw is not None and not raw.empty:
            pf = {"Open","High","Low","Close","Volume","Adj Close"}
            if isinstance(raw.columns, pd.MultiIndex):
                l0 = set(raw.columns.get_level_values(0))
                for sym in symbols:
                    try:
                        df = raw.xs(sym,axis=1,level=1) if l0&pf else raw[sym]
                        df = _clean(df, min_bars=300)
                        if df is not None: out[sym] = df
                    except Exception: pass
            elif len(symbols) == 1:
                df = _clean(raw, min_bars=300)
                if df is not None: out[symbols[0]] = df
    except Exception: pass
    for sym in [s for s in symbols if s not in out]:
        for _ in range(2):
            try:
                df = yf.Ticker(sym).history(
                    start=start.strftime("%Y-%m-%d"),
                    end=end.strftime("%Y-%m-%d"),
                    auto_adjust=True, actions=False)
                df = _clean(df, min_bars=300)
                if df is not None: out[sym] = df; break
            except Exception: time.sleep(0.2)
        time.sleep(0.04)
    return out

# ── Live print ────────────────────────────────────────────────
LIVE_COLS = ["Ticker","Price","Score","Entry_Price","Stop_Loss","Target",
             "Risk_Reward","Signal_Date"]
_CW = {"Ticker":8,"Price":10,"Score":7,"Entry_Price":12,"Stop_Loss":11,
       "Target":10,"Risk_Reward":12,"Signal_Date":13}
_CF = {"Price":"${:.2f}","Score":"{:.0f}","Entry_Price":"${:.2f}",
       "Stop_Loss":"${:.2f}","Target":"${:.2f}","Risk_Reward":"{:.2f}"}
_hdr_done = False

def _live_header():
    global _hdr_done
    if _hdr_done: return
    print("\n" + "━"*95)
    print("  📊  LIVE MATCHES  —  each stock printed the moment it passes all 3 timeframes")
    print("━"*95)
    h = "".join(f"  {c:<{_CW.get(c,12)}}" for c in LIVE_COLS)
    print(h)
    print("  " + "─"*93)
    _hdr_done = True

def live_print(r):
    _live_header()
    row = ""
    for c in LIVE_COLS:
        val = r.get(c,"—")
        w   = _CW.get(c,12)
        fmt = _CF.get(c)
        try:   s = fmt.format(val) if (fmt and val not in("—",None)) else str(val)
        except Exception: s = str(val)
        row += f"  {s:<{w}}"
    print(row)

# ── Health check ──────────────────────────────────────────────
print("━"*65)
print("  STEP 1  DATA CHECK")
print("━"*65)
chk_d = download_daily(["AAPL","MSFT","NVDA"], CFG["history_days"])
if not chk_d:
    print("❌  No data.")
else:
    for s, dd in chk_d.items():
        print(f"  ✅ {s}: daily {len(dd)} bars (${float(dd['Close'].iloc[-1]):.2f}, "
              f"{dd.index[-1].date()})")
print()

# ── Ticker list ───────────────────────────────────────────────
print("━"*65)
print("  STEP 2  FETCH TICKERS")
print("━"*65)

def get_tickers():
    pool = set()
    hdrs = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}
    for url, label in [
        ("https://www.nasdaqtrader.com/dynamic/SymDir/nasdaqlisted.txt","nasdaqlisted"),
        ("https://www.nasdaqtrader.com/dynamic/SymDir/otherlisted.txt", "otherlisted"),
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
        r = requests.get("https://api.nasdaq.com/api/screener/stocks"
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
        "ALAB","SMCI","HOOD","COIN","SOFI","UPST","BILL","ZS","OKTA","DDOG",
        "SNOW","MDB","REGN","VRTX","ALNY","PODD","MPWR","ONTO","ENTG","SWKS",
        "ADSK","ANSS","BIIB","CPRT","ENPH","FAST","FTNT","IDXX","ISRG","LULU",
        "ODFL","ORLY","PAYX","PCAR","SBUX","TMUS","VRSK","ZM","ZBRA","RBRK",
        "SMAR","PSTG","NET","GTLB","CFLT","MNDY","HUBS","VEEV","PCTY","PAYC",
        "BRZE","IONQ","ABNB","DASH","RBLX","KVYO","SOUN","CRWV","MSTR","MARA",
        "QUBT","RGTI","ASTS","RKLB","LUNR","FSLR","PYPL","ROKU","ROST","POOL",
        "ALGN","AMGN","CTAS","DOCU","EA","FISV","GILD","INTU","MCHP","MNST",
        "NXPI","PDD","SIRI","ULTA","XEL","ESTC","QRVO","ACLS","EXAS","IRTC",
    }
    b = len(pool); pool |= static
    print(f"  ✅ {'Static fallback':<18}: +{len(pool)-b:>4} → {len(pool)}")
    clean = sorted({s.upper() for s in pool if isinstance(s,str)
                    and s.isalpha() and 1<=len(s)<=5})
    print(f"\n  🎯 Total: {len(clean)} tickers")
    return clean

TICKERS = get_tickers()
print()

# ── Main scan — single pass (daily download only, then check) ────
print("━"*65)
print(f"  STEP 3  SCANNING {len(TICKERS)} TICKERS")
print("━"*65)
print("  Fetching daily bars (single download per ticker)")
print("  A stock only matches if the cluster structure, 2-candle reclaim,")
print("  and RSI confirmation all fire within the lookback window\n")

_hdr_done = False
results = []
no_daily_data = 0

daily_batches = [TICKERS[i:i+CFG["batch_size"]]
                 for i in range(0, len(TICKERS), CFG["batch_size"])]

daily_map = {}

with tqdm(total=len(TICKERS), desc="Daily fetch", unit="stk",
          bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]") as pbar:
    for batch in daily_batches:
        got = download_daily(batch, CFG["history_days"])
        daily_map.update(got)
        no_daily_data += len(batch) - len(got)
        pbar.update(len(batch))
        time.sleep(CFG["batch_sleep"])

got_daily = len(TICKERS) - no_daily_data
print(f"\n  Daily data : {got_daily}/{len(TICKERS)} tickers")
print()

for sym in tqdm(list(daily_map.keys()), desc="Checking RSI reversal", unit="stk"):
    try:
        r = analyze_cluster_reclaim(sym, daily_map[sym])
        if r is None: continue
        r["Ticker"] = sym
        results.append(r)
        live_print(r)
    except Exception: pass

print(f"\n{'━'*65}")
print(f"  SCAN COMPLETE")
print(f"  Tickers scanned : {len(TICKERS)}")
print(f"  Daily data      : {got_daily}")
print(f"  ✅ Matches       : {len(results)}")
print(f"{'━'*65}")

if not results:
    print("\n  No matches. Try relaxing:")
    print("   cluster_compression_pct    3.0 → 5.0   (allow a looser cluster)")
    print("   volume_multiplier          1.3 → 1.15  (lower the volume bar)")
    print("   signal_lookback_days        15 → 25    (search further back)")
    print("   rsi_min_level                50 → 45")
    print("   rsi_avg_period                9 → 14")
    print("   min_price                     2 → 1")
    print("   min_avg_volume            80000 → 50000")

results.sort(key=lambda x: x["Score"], reverse=True)

# ── Always build df_out and save/email (even if 0 results) ────
ts      = datetime.today().strftime("%Y%m%d_%H%M")
out_dir = os.environ.get("GITHUB_WORKSPACE", os.getcwd())

COLS = [
    "Ticker","Price","Score",
    "Entry_Price","Stop_Loss","Target","Risk_%","Risk_Reward",
    "Compression_%","Vol_Ratio","RSI","RSI_Avg","SMA50_Rising",
    "Signal_Date","Days_Since_Signal","Recent_Signal_Count","Recent_Signals",
    "Flags",
]
df_out = pd.DataFrame([{k:v for k,v in r.items() if not k.startswith("_")}
                        for r in results]) if results else pd.DataFrame(columns=COLS)
if not df_out.empty:
    df_out = df_out[[c for c in COLS if c in df_out.columns]]
    df_out.reset_index(drop=True, inplace=True)

FMT = {
    "Price"       : lambda v: f"${v:.2f}",
    "Score"       : lambda v: f"{v:.0f}",
    "Entry_Price" : lambda v: f"${v:.2f}",
    "Stop_Loss"   : lambda v: f"${v:.2f}",
    "Target"      : lambda v: f"${v:.2f}",
    "Risk_%"      : lambda v: f"{v:.1f}%",
    "Risk_Reward" : lambda v: f"1:{v:.1f}",
    "Compression_%": lambda v: f"{v:.2f}%",
    "Vol_Ratio"   : lambda v: f"{v:.2f}x",
    "RSI"         : lambda v: f"{v:.1f}",
    "RSI_Avg"     : lambda v: f"{v:.1f}",
    "Days_Since_Signal": lambda v: f"{int(v)}d ago",
}

def fmt_v(col, val):
    if val is None or (isinstance(val, float) and np.isnan(val)): return "—"
    try:
        if col in FMT: return FMT[col](val)
    except Exception: pass
    return str(val) if str(val) not in ("nan","None","") else "—"

if _IN_NOTEBOOK and results:
    DISP = ["Ticker","Price","Score","Entry_Price","Stop_Loss","Target",
            "Risk_Reward","Signal_Date","Compression_%","Vol_Ratio","RSI"]
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
            elif col == "Risk_Reward":
                try:
                    v = float(raw)
                    clr = "#22c55e" if v >= 2 else "#f59e0b"
                    sty = f"color:{clr};font-weight:600"
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
    <span style="color:#f1f5f9;font-size:15px;font-weight:700">MA/VWAP Cluster Reclaim + RSI</span>
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
    📈 MA/VWAP Cluster Reclaim + RSI Confirmation
  </h2>
  <p style="margin:6px 0 0;color:#94a3b8;font-size:12px">
    {datetime.today().strftime('%Y-%m-%d %H:%M')} &nbsp;·&nbsp;
    <b style="color:#22c55e">{len(results)} matches</b> from {len(TICKERS)} tickers
    &nbsp;·&nbsp; pattern checked over the last {CFG['signal_lookback_days']} trading days
  </p>
</div>"""

    legend_html = f"""
<div style="background:#f8fafc;border:1px solid #e2e8f0;border-radius:8px;
        padding:12px 18px;margin-top:6px;font-size:11px;color:#64748b;
        font-family:'Segoe UI',Arial,sans-serif">
  <b style="color:#475569">GUIDE</b> &nbsp;·&nbsp;
  SMA50&gt;SMA150, EMA8/JMA/SMA21/VWAP all above SMA50 and tightly
  compressed &nbsp;·&nbsp;
  A red candle closed below all 4 lines, next green candle closed
  above all 4 on 1.3x+ volume &nbsp;·&nbsp;
  RSI crossed above its own average and is above 50 &nbsp;·&nbsp;
  Entry_Price = SMA21 on the signal day &nbsp;·&nbsp;
  Stop_Loss = the red candle's low &nbsp;·&nbsp;
  Target = fixed 1:{CFG['risk_reward_ratio']:.0f} risk:reward &nbsp;·&nbsp;
  Signal_Date is the exact date the pattern fired (checked over the
  last {CFG['signal_lookback_days']} trading days, not just today)
</div>"""

    display_html(header_html + table_html + legend_html)

elif results:
    # ASCII table (CLI/GitHub Actions mode)
    CLI_COLS = ["Ticker","Price","Score","Entry_Price","Stop_Loss","Target",
                "Risk_Reward","Signal_Date","Compression_%","Vol_Ratio","RSI"]
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
    tit = f"  MA/VWAP Cluster Reclaim + RSI   {datetime.today().strftime('%Y-%m-%d')}   {len(df_out)} matches"
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
  Score           0-100 (compression + slope + volume + RSI + freshness)
  Entry_Price     SMA21 value on the signal day
  Stop_Loss       low of the red (candle A) day
  Target          Entry + {CFG['risk_reward_ratio']:.0f} x (Entry - Stop_Loss) — fixed 1:{CFG['risk_reward_ratio']:.0f}
  Signal_Date     exact calendar date the pattern fired
  Days_Since_Signal  how many trading days ago (0 = today; checked
                     over the last signal_lookback_days trading days)
  ──────────────────────────────────────────────────────""")

# Save
fpath = os.path.join(out_dir, f"cluster_reclaim_rsi_{ts}.csv")
df_out.to_csv(fpath, index=False)
print(f"\n  💾 CSV → {fpath}")
tv = os.path.join(out_dir, f"tv_cluster_reclaim_rsi_{ts}.txt")
with open(tv,"w") as f:
    f.write(f"###MA-VWAP Cluster Reclaim + RSI {datetime.today().strftime('%Y-%m-%d')}\n")
    for r in results: f.write(f"NASDAQ:{r['Ticker']}\n")
print(f"  📋 TradingView → {tv}")
if results:
    print(f"\n  📋 Tickers (comma-separated):")
    print(f"  {', '.join(r['Ticker'] for r in results)}")

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
        return
    if not gp:
        print("[Email] ❌  GMAIL_PASS secret is empty")
        print("         → Must be a Gmail App Password (16 chars, no spaces)")
        return
    if not et:
        print("[Email] ❌  EMAIL_TO secret is empty")
        return

    eto = [e.strip() for e in et.split(",") if e.strip()]
    cnt = len(rl)

    try:
        print(f"[Email] Sending to {et}  ({cnt} results)...")

        th_e = "".join(
            f'<th style="background:#1e293b;color:#e2e8f0;padding:8px 11px;'
            f'font-size:11px;font-weight:700;border-bottom:2px solid #3b82f6;'
            f'white-space:nowrap">{c}</th>'
            for c in ["Ticker","Price","Score","Entry_Price","Stop_Loss",
                      "Target","Risk_Reward","Signal_Date"]
        )
        rows_e = ""
        for i, r in enumerate(rl[:50]):
            bg  = "#fff" if i % 2 == 0 else "#f0f9ff"
            ticker = r.get("Ticker","—")
            price  = r.get("Price",0) or 0
            score  = r.get("Score",0) or 0
            entry  = r.get("Entry_Price",0) or 0
            stop   = r.get("Stop_Loss",0) or 0
            target = r.get("Target",0) or 0
            rr     = r.get("Risk_Reward",0) or 0
            sdate  = r.get("Signal_Date","—")
            rows_e += (
                f'<tr style="background:{bg}">'
                f'<td style="padding:6px 11px;font-size:12px;font-weight:700">{ticker}</td>'
                f'<td style="padding:6px 11px;font-size:12px">${float(price):.2f}</td>'
                f'<td style="padding:6px 11px;font-size:12px;font-weight:700;'
                f'background:#166534;color:#fff;text-align:center">{float(score):.0f}</td>'
                f'<td style="padding:6px 11px;font-size:12px;color:#22c55e">${float(entry):.2f}</td>'
                f'<td style="padding:6px 11px;font-size:12px;color:#ef4444">${float(stop):.2f}</td>'
                f'<td style="padding:6px 11px;font-size:12px;color:#3b82f6">${float(target):.2f}</td>'
                f'<td style="padding:6px 11px;font-size:12px;font-weight:600">1:{float(rr):.0f}</td>'
                f'<td style="padding:6px 11px;font-size:12px;text-align:center;'
                f'color:#a78bfa;font-weight:600">{sdate}</td>'
                f'</tr>'
            )
        no_results_msg = ('<tr><td colspan="8" style="padding:20px;text-align:center;'
                           'color:#94a3b8;font-size:13px">No matches today</td></tr>')

        html_e = f"""<!DOCTYPE html><html><body style="margin:0;padding:0;
background:#f1f5f9;font-family:'Segoe UI',Arial,sans-serif">
<table width="100%" cellpadding="0" cellspacing="0"><tr><td align="center" style="padding:20px 10px">
<table width="100%" style="max-width:800px;background:#fff;border-radius:12px;
       overflow:hidden;box-shadow:0 4px 16px rgba(0,0,0,0.08)">
  <tr><td style="background:linear-gradient(135deg,#0f172a,#1e3a5f);padding:22px 28px">
<h1 style="margin:0;color:#60a5fa;font-size:20px;font-weight:700">
  📊 MA/VWAP Cluster Reclaim + RSI
</h1>
<p style="margin:6px 0 0;color:#94a3b8;font-size:12px">
  {datetime.today().strftime('%Y-%m-%d %H:%M UTC')} &nbsp;·&nbsp;
  {cnt} match{'es' if cnt!=1 else ''} found — Monthly RSI cross, Weekly RSI strength,
  Daily RSI reversal, all at once
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
{f'''<div style="margin-top:10px;background:#f8fafc;border:1px solid #e2e8f0;
        border-radius:6px;padding:10px 14px">
  <p style="margin:0 0 4px;font-size:10px;color:#94a3b8;font-weight:700">
    TICKERS (comma-separated, copy/paste)
  </p>
  <p style="margin:0;font-size:12px;color:#1e293b;font-family:monospace;
      word-break:break-all">{", ".join(r.get("Ticker","") for r in rl)}</p>
</div>''' if rl else ''}
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
            f"MA/VWAP Cluster Reclaim + RSI (checked over last {CFG['signal_lookback_days']} trading days) — {datetime.today().strftime('%Y-%m-%d')}",
            f"{cnt} matches (cluster structure + 2-candle reclaim + RSI confirmation)",
            "="*60,
        ]
        if rl:
            for r in rl[:50]:
                ticker = r.get("Ticker","—")
                price  = r.get("Price",0) or 0
                score  = r.get("Score",0) or 0
                entry  = r.get("Entry_Price",0) or 0
                stop   = r.get("Stop_Loss",0) or 0
                target = r.get("Target",0) or 0
                rr     = r.get("Risk_Reward",0) or 0
                sdate  = r.get("Signal_Date","—")
                plain_lines.append(
                    f"{ticker:<7} ${float(price):.2f}  Score:{float(score):.0f}  "
                    f"Entry:${float(entry):.2f}  SL:${float(stop):.2f}  "
                    f"Target:${float(target):.2f}  R:R:1:{float(rr):.0f}  Signal:{sdate}"
                )
            plain_lines.append("")
            plain_lines.append("Tickers (comma-separated):")
            plain_lines.append(", ".join(r.get("Ticker","") for r in rl))
        else:
            plain_lines.append("No matches today")
        plain_lines.append("\nFull results in CSV attachment.")
        plain_e = "\n".join(plain_lines)

        subj = (f"📊 Cluster Reclaim + RSI — {cnt} signal{'s' if cnt!=1 else ''}"
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
    except smtplib.SMTPException as e:
        print(f"[Email] ❌  SMTP error: {e}")
    except Exception as e:
        print(f"[Email] ❌  Unexpected error: {type(e).__name__}: {e}")

try:
    _send_email(results, fpath)
except Exception as e:
    print(f"[Email] ❌  Unexpected top-level error: {type(e).__name__}: {e}")
    print("[Email]    Continuing — CSV is still saved.")

if _IN_NOTEBOOK:
    try:
        from google.colab import files
        files.download(fpath); files.download(tv)
    except Exception: pass
else:
    print("  (CI: files in workspace, email sent)")

# ── Charts for top 5 (price + entry/stop/target + RSI panel) ────
if results:
    top = results[:min(5,len(results))]
    fig, axes = plt.subplots(len(top),2,figsize=(16,4.2*len(top)),facecolor="#0f172a",
                              gridspec_kw={"height_ratios":[1]*len(top), "width_ratios":[3,1]})
    if len(top)==1: axes = axes.reshape(1,2)
    for row, r in zip(axes, top):
        ax, ax_rsi = row[0], row[1]
        df_p = r["_df_daily"].tail(90).copy()
        ema8_p  = r["_ema8"].reindex(df_p.index)
        sma21_p = r["_sma21"].reindex(df_p.index)
        sma50_p = r["_sma50"].reindex(df_p.index)
        vwap_p  = r["_vwap"].reindex(df_p.index)
        jma_p   = r["_jma"].reindex(df_p.index)
        ax.set_facecolor("#0f172a")
        ax.plot(df_p.index, df_p["Close"], color="#60a5fa", lw=1.6, label="Close", zorder=5)
        ax.plot(df_p.index, jma_p,   color="#f472b6", lw=1.1, label="JMA",   zorder=4)
        ax.plot(df_p.index, ema8_p,  color="#38bdf8", lw=1.0, ls="--", label="EMA8",  zorder=3)
        ax.plot(df_p.index, sma21_p, color="#34d399", lw=1.0, ls="--", label="SMA21", zorder=3)
        ax.plot(df_p.index, vwap_p,  color="#fbbf24", lw=1.0, ls=":",  label="VWAP",  zorder=3)
        ax.plot(df_p.index, sma50_p, color="#f87171", lw=1.2, ls="-.", label="SMA50", zorder=3)
        ax.axhline(r["Stop_Loss"], color="#ef4444", lw=1.0, ls="--", alpha=0.85,
                  label=f"Stop ${r['Stop_Loss']:.2f}")
        ax.axhline(r["Target"], color="#3b82f6", lw=1.0, ls="--", alpha=0.85,
                  label=f"Target ${r['Target']:.2f}")
        ax.set_title(
            f"{r['Ticker']}  |  ${r['Price']:.2f}  |  Score {r['Score']}  |  "
            f"Entry ${r['Entry_Price']:.2f}  1:{r['Risk_Reward']:.0f}  |  {r['Signal_Date']}",
            color="#e2e8f0", fontsize=9, fontweight="bold", pad=7)
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))
        ax.tick_params(colors="#94a3b8", labelsize=8)
        for sp in ax.spines.values(): sp.set_edgecolor("#1e3a5f")
        ax.legend(loc="upper left", facecolor="#1e293b", labelcolor="#e2e8f0",
                  fontsize=6, framealpha=0.9, ncol=2)
        ax.grid(color="#1e3a5f", ls="--", lw=0.5, alpha=0.6)

        rsi_p = r["_rsi"].reindex(df_p.index)
        rsi_avg_p = r["_rsi_avg"].reindex(df_p.index)
        ax_rsi.set_facecolor("#0f172a")
        ax_rsi.plot(df_p.index, rsi_p, color="#f472b6", lw=1.3, label="RSI")
        ax_rsi.plot(df_p.index, rsi_avg_p, color="#94a3b8", lw=1.0, ls="--", label="RSI Avg")
        ax_rsi.axhline(50, color="#64748b", lw=0.8, ls=":")
        ax_rsi.set_ylim(0,100)
        ax_rsi.set_title(f"RSI {r['RSI']:.0f} vs Avg {r['RSI_Avg']:.0f}",
                          color="#e2e8f0", fontsize=8, pad=5)
        ax_rsi.xaxis.set_major_formatter(mdates.DateFormatter("%b"))
        ax_rsi.tick_params(colors="#94a3b8", labelsize=7)
        for sp in ax_rsi.spines.values(): sp.set_edgecolor("#1e3a5f")
        ax_rsi.legend(loc="upper left", facecolor="#1e293b", labelcolor="#e2e8f0", fontsize=6)
        ax_rsi.grid(color="#1e3a5f", ls="--", lw=0.5, alpha=0.6)
    plt.suptitle(
        f"MA/VWAP Cluster Reclaim + RSI  ·  {datetime.today().strftime('%Y-%m-%d')}",
        color="#60a5fa", fontsize=12, fontweight="bold", y=1.001)
    plt.tight_layout()
    cp = os.path.join(out_dir, f"cluster_reclaim_rsi_chart_{ts}.png")
    plt.savefig(cp, dpi=150, bbox_inches="tight", facecolor="#0f172a")
    if _IN_NOTEBOOK: plt.show()
    else: plt.close()
    print(f"  📊 Chart → {cp}")
    if _IN_NOTEBOOK:
        try:
            from google.colab import files; files.download(cp)
        except Exception: pass

print(f"""
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  📋 SIGNAL (all required — scanned over the last
  signal_lookback_days trading days, {CFG['signal_lookback_days']}d ≈ 3 weeks,
  not just today)

  1) STRUCTURE: SMA50 > SMA150, AND EMA8/JMA/SMA21/VWAP all above
     SMA50 AND compressed tightly together (max-min spread within
     cluster_compression_pct% of price). SMA50 rising is a scored
     bonus, not a hard requirement.
  2) TWO-CANDLE RECLAIM: candle A (red) closed BELOW all 4 of
     JMA/EMA8/SMA21/VWAP; candle B (green) closed ABOVE all 4 of
     the same lines, with volume >= volume_multiplier (1.3x)
     candle A's volume.
  3) RSI CONFIRMATION: RSI(14) crossed above its own moving
     average (rsi_avg_period-bar SMA of RSI) AND RSI > 50, both
     on candle B.

  If the pattern fired more than once in the window, the MOST
  RECENT occurrence is used for Entry/Stop/Target/scoring.

  📋 DATA SOURCING
  Only 1 download per ticker: daily bars (~450 days).

  📋 OUTPUT
  Entry_Price = SMA21 value on the signal day (candle B)
  Stop_Loss   = LOW of candle A (the red day)
  Target      = Entry + risk_reward_ratio x (Entry - Stop_Loss)
                — a FIXED 1:{CFG['risk_reward_ratio']:.0f} risk:reward by
                construction, not a swing-high search
  Signal_Date = the exact calendar date the pattern fired
  Days_Since_Signal = how many trading days ago (0 = today)
  Recent_Signals    = every date the pattern fired within the window

  📋 SCORE (0-100)
  Compression tightness (0-20) + SMA50-rising bonus (0-10) +
  Volume ratio strength (0-20) + RSI margin above 50 (0-20) +
  Freshness (0-15) + base points for clearing every gate (15)

  💡 BEST SETUPS
  Score > 70          tight cluster, strong volume, healthy RSI
  Vol_Ratio > 1.5       well above the 1.3x minimum
  Compression_% < 1.5   a genuinely tight coil, not a loose cluster
  Days_Since_Signal = 0-3   freshest reclaim

  ⚙️  TUNE IF 0 RESULTS
  cluster_compression_pct    3.0 → 5.0   (allow a looser cluster)
  volume_multiplier          1.3 → 1.15  (lower the volume bar)
  signal_lookback_days        15 → 25    (search further back)
  rsi_min_level                50 → 45
  rsi_avg_period                9 → 14
  min_price                     2 → 1
  min_avg_volume            80000 → 50000
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
""")
