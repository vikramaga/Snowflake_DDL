# ============================================================
# NASDAQ — 3-Candle Reversal Near SMA50 Scanner (v1)
# ============================================================
#
# SIGNAL (all required):
#
#   0. STRUCTURE: SMA50 > SMA150 (bullish structure), AND the low
#      of candle 1 is within near_sma50_pct% of SMA50 — the pattern
#      is testing/touching support, not floating far from it.
#
#   1. CANDLE 1 (the test day): RED (close < open). A HAMMER shape
#      (long lower wick, small body, tiny upper wick) is PREFERRED
#      and scored as a bonus, but not a hard requirement.
#
#   2. CANDLE 2 (next day): GREEN DOJI — a tiny body relative to
#      its range, close > open — AND its LOW is higher than
#      candle 1's low (support holding, no new low made).
#
#   3. CANDLE 3 (next day): a "healthy" GREEN candle — a real body
#      (not a doji), close > open — with volume higher than candle
#      2's volume, AND its LOW is higher than candle 2's low
#      (rising lows continue).
#
# Scanned over the last signal_lookback_days trading days (not just
# today), consistent with the other scanners in this repo.
#
# DATA — only 1 download per ticker: daily bars.
#
# OUTPUT: Entry_Price and Stop_Loss are included as a reasonable
# default (Entry = candle 3's high — a breakout trigger; Stop = 
# candle 2's low — the most recent confirmed higher low) since every
# other scanner in this repo reports actionable levels, but no
# specific entry/stop/target logic was requested for this pattern —
# adjust or ignore these two fields if that default doesn't fit.
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
    "history_days"          : 750,   # ~3 years — gives the backtest a much
                                      # larger sample of historical occurrences
                                      # (was 400, just enough for SMA150+buffer)

    # ── Indicator periods ────────────────────────────────────────
    "sma50_period"           : 50,
    "sma150_period"          : 150,

    # ── Step 0: structure + proximity to SMA50 ──────────────────
    "near_sma50_pct"         : 3.0,  # candle 1's low must be within this %
                                      # of SMA50 (tests support)

    # ── Candle shape thresholds ──────────────────────────────────
    "hammer_lower_wick_mult" : 2.0,  # lower wick >= this x body (bonus, not gate)
    "hammer_upper_wick_pct_of_range": 0.15,
    "doji_body_pct"          : 0.15, # candle 2 body <= this % of its range
    "healthy_body_pct"       : 0.40, # candle 3 body >= this % of its range

    "signal_lookback_days"   : 15,   # ~3 trading weeks — how far back to look
                                      # for the pattern, not just today

    # ── Historical backtest (max favorable price move after the pattern) ──
    "mfe_holding_days"       : 15,   # trading days forward to measure the
                                      # highest price reached after each
                                      # historical occurrence
    "mfe_hit_threshold_pct"  : 5.0,  # "hit rate" = % of historical occurrences
                                      # that reached at least this much upside

    # ── Filters ─────────────────────────────────────────────────
    "min_avg_volume"         : 80_000,
    "min_price"              : 2.0,

    "batch_size"             : 50,
    "batch_sleep"            : 1.5,
}

# ── Candle shape helpers ───────────────────────────────────────
def candle_metrics(o, h, l, c):
    body = abs(c - o)
    rng  = h - l
    upper_wick = h - max(o, c)
    lower_wick = min(o, c) - l
    return body, rng, upper_wick, lower_wick

def hammer_score(o, h, l, c, cfg):
    """Returns (is_hammer: bool, quality 0-1) for a RED candle."""
    body, rng, upper_wick, lower_wick = candle_metrics(o, h, l, c)
    if rng <= 0: return False, 0.0
    body_eff = max(body, rng * 0.02)
    is_hammer = (lower_wick >= cfg["hammer_lower_wick_mult"] * body_eff) and \
                (upper_wick <= cfg["hammer_upper_wick_pct_of_range"] * rng)
    quality = min(1.0, lower_wick / rng) if is_hammer else 0.0
    return is_hammer, quality

def is_doji(o, h, l, c, cfg):
    body, rng, _, _ = candle_metrics(o, h, l, c)
    if rng <= 0: return False
    return body <= cfg["doji_body_pct"] * rng

def is_healthy_green(o, h, l, c, cfg):
    body, rng, _, _ = candle_metrics(o, h, l, c)
    if rng <= 0 or c <= o: return False
    return body >= cfg["healthy_body_pct"] * rng

def check_three_candle_at(df, sma50, sma150, i3, cfg):
    """
    Checks the full 3-candle pattern with candle 3 anchored at bar
    `i3`. This is the SINGLE SOURCE OF TRUTH for the pattern logic —
    both the live scanner (last signal_lookback_days) and the
    historical backtest (full available history) call this exact
    function, so they can never drift apart.

    Returns (passed: bool, details: dict).
    """
    i2, i1 = i3 - 1, i3 - 2
    if i1 < 1 or i3 >= len(df):
        return False, {}

    s50, s150 = sma50.iloc[i1], sma150.iloc[i1]
    if np.isnan(s50) or np.isnan(s150):
        return False, {}

    o1, h1, l1, c1 = (float(df["Open"].iloc[i1]), float(df["High"].iloc[i1]),
                      float(df["Low"].iloc[i1]),  float(df["Close"].iloc[i1]))
    o2, h2, l2, c2 = (float(df["Open"].iloc[i2]), float(df["High"].iloc[i2]),
                      float(df["Low"].iloc[i2]),  float(df["Close"].iloc[i2]))
    o3, h3, l3, c3 = (float(df["Open"].iloc[i3]), float(df["High"].iloc[i3]),
                      float(df["Low"].iloc[i3]),  float(df["Close"].iloc[i3]))
    vol2, vol3 = float(df["Volume"].iloc[i2]), float(df["Volume"].iloc[i3])

    # ── Step 0: structure + proximity to SMA50 (checked at candle 1) ──
    structure_ok = s50 > s150
    near_pct = abs(l1 - s50) / s50 * 100 if s50 > 0 else 999
    near_ok  = near_pct <= cfg["near_sma50_pct"]
    if not (structure_ok and near_ok):
        return False, {}

    # ── Step 1: candle 1 red (hammer preferred, scored not gated) ──
    candle1_red = c1 < o1
    if not candle1_red:
        return False, {}
    is_hammer, hammer_quality = hammer_score(o1, h1, l1, c1, cfg)

    # ── Step 2: candle 2 green doji, higher low than candle 1 ───────
    candle2_green = c2 > o2
    candle2_doji  = is_doji(o2, h2, l2, c2, cfg)
    candle2_higher_low = l2 > l1
    if not (candle2_green and candle2_doji and candle2_higher_low):
        return False, {}

    # ── Step 3: candle 3 healthy green, higher volume, higher low ───
    candle3_healthy = is_healthy_green(o3, h3, l3, c3, cfg)
    candle3_higher_vol = vol3 > vol2
    candle3_higher_low = l3 > l2
    if not (candle3_healthy and candle3_higher_vol and candle3_higher_low):
        return False, {}

    return True, {
        "idx": i3, "i2": i2, "i1": i1,
        "sma50": float(s50), "near_pct": near_pct,
        "is_hammer": is_hammer, "hammer_quality": hammer_quality,
        "low1": l1, "low2": l2, "low3": l3,
        "high3": h3, "close3": c3,
        "vol2": vol2, "vol3": vol3,
    }

def find_three_candle_signals(df, sma50, sma150, cfg):
    """
    Scans the last `signal_lookback_days` trading days for the full
    3-candle pattern, evaluated with candle 3 (the healthy green
    day) as the anchor of each candidate window. Returns a list of
    hit dicts, most recent first.
    """
    n = len(df)
    lb = cfg["signal_lookback_days"]
    hits = []
    for back in range(0, lb):
        i3 = (n - 1) - back
        passed, details = check_three_candle_at(df, sma50, sma150, i3, cfg)
        if passed:
            hits.append(details)
    return hits

def compute_forward_mfe(df, entry_idx, entry_price, holding_days):
    """
    Maximum Favorable Excursion: the highest High reached over the
    next `holding_days` bars after `entry_idx`, expressed as a %
    above `entry_price`. Returns None if there isn't enough forward
    data to evaluate the full window.
    """
    n = len(df)
    end = entry_idx + 1 + holding_days
    if end > n or entry_idx + 1 >= n:
        return None
    max_high = float(df["High"].iloc[entry_idx+1:end].max())
    return (max_high - entry_price) / entry_price * 100 if entry_price > 0 else None

def backtest_three_candle_pattern(df, sma50, sma150, cfg):
    """
    Re-runs check_three_candle_at() over the ENTIRE available
    history (not just the live lookback window) to find every past
    occurrence of the pattern, then measures the maximum favorable
    price move (MFE %) over the following mfe_holding_days for each
    one. Returns a list of trade dicts.
    """
    n = len(df)
    hold = cfg["mfe_holding_days"]
    trades = []
    for i3 in range(3, n - hold):   # need i1=i3-2>=1 and `hold` days after i3
        passed, details = check_three_candle_at(df, sma50, sma150, i3, cfg)
        if not passed:
            continue
        entry_price = details["high3"]
        mfe_pct = compute_forward_mfe(df, i3, entry_price, hold)
        if mfe_pct is None:
            continue
        trades.append({
            "date": df.index[i3], "entry": round(entry_price, 2),
            "mfe_pct": round(mfe_pct, 2), "is_hammer": details["is_hammer"],
        })
    return trades

def summarize_mfe(trades, hit_threshold_pct):
    """Aggregates a list of MFE trade dicts into summary statistics."""
    if not trades:
        return {"count": 0, "avg": None, "median": None,
                "best": None, "worst": None, "hit_rate_pct": None}
    vals = sorted(t["mfe_pct"] for t in trades)
    n = len(vals)
    median = vals[n//2] if n % 2 == 1 else (vals[n//2-1] + vals[n//2]) / 2
    hits = sum(1 for v in vals if v >= hit_threshold_pct)
    return {
        "count": n, "avg": sum(vals)/n, "median": median,
        "best": vals[-1], "worst": vals[0],
        "hit_rate_pct": hits / n * 100,
    }

# ── Technical signal: 3-candle reversal near SMA50 ────────────────
ALL_BACKTEST_TRADES = []   # accumulates trades across the FULL universe scan

def analyze_three_candle_pattern(sym, df_daily):
    """
    Returns dict with score and setup details, or None if no
    required condition is met anywhere in the lookback window.

    Regardless of whether a live signal matches, this ALSO runs the
    full-history backtest for the ticker (if it passes the same
    basic price/volume filters) and appends the trades found to the
    global ALL_BACKTEST_TRADES accumulator, so the full-universe
    backtest summary reflects every liquid ticker scanned — not just
    today's matches.
    """
    global ALL_BACKTEST_TRADES

    if df_daily is None:
        return None

    price   = float(df_daily["Close"].iloc[-1])
    avg_vol = float(df_daily["Volume"].tail(20).mean())
    if price   < CFG["min_price"]:      return None
    if avg_vol < CFG["min_avg_volume"]: return None

    n = len(df_daily)
    if n < CFG["sma150_period"] + CFG["signal_lookback_days"] + 20:
        return None

    sma50  = df_daily["Close"].rolling(CFG["sma50_period"]).mean()
    sma150 = df_daily["Close"].rolling(CFG["sma150_period"]).mean()

    # ── Backtest: full available history, this ticker, regardless of
    #    whether a live signal matches ──────────────────────────────
    bt_trades = backtest_three_candle_pattern(df_daily, sma50, sma150, CFG)
    for t in bt_trades:
        t["ticker"] = sym
    ALL_BACKTEST_TRADES.extend(bt_trades)
    bt_summary = summarize_mfe(bt_trades, CFG["mfe_hit_threshold_pct"])

    hits = find_three_candle_signals(df_daily, sma50, sma150, CFG)
    if not hits:
        return None

    sig = hits[0]   # most recent
    sig_idx = sig["idx"]
    days_since_signal = (n - 1) - sig_idx
    recent_signals = [
        {"date": df_daily.index[h["idx"]], "bars_ago": (n-1)-h["idx"]}
        for h in hits
    ]

    # ── Entry / Stop (reasonable defaults — see header note) ─────────
    entry_price = sig["high3"]      # breakout above candle 3's high
    stop_loss   = sig["low2"]       # candle 2's low (most recent higher low)
    if entry_price <= stop_loss:
        return None
    risk_pct = (entry_price - stop_loss) / entry_price * 100 if entry_price > 0 else 0
    vol_ratio = sig["vol3"] / sig["vol2"] if sig["vol2"] > 0 else 0

    # ── Score (0-100) ────────────────────────────────────────────
    score = 0
    reasons = []
    if sig["is_hammer"]:
        score += round(20 * sig["hammer_quality"])
        reasons.append(f"Hammer({sig['hammer_quality']:.1f})")
    else:
        reasons.append("NoHammer")
    score += max(0, min(15, 15 - sig["near_pct"] * 4))
    reasons.append(f"NearSMA50({sig['near_pct']:.1f}%)")
    score += min(20, (vol_ratio - 1.0) * 20)
    reasons.append(f"Vol{vol_ratio:.1f}x")
    low_progression_pct = (sig["low3"] - sig["low1"]) / sig["low1"] * 100 if sig["low1"] > 0 else 0
    score += min(10, low_progression_pct * 3)
    reasons.append(f"LowsUp{low_progression_pct:+.1f}%")
    freshness_pts = max(0, 10 - days_since_signal)
    score += freshness_pts
    reasons.append(f"{days_since_signal}dAgo")
    if bt_summary["median"] is not None:
        if bt_summary["median"] >= 10: score += 10; reasons.append("StrongHistoricalMFE")
        elif bt_summary["median"] >= 5: score += 5
    score += 10   # base for clearing every gate
    score = round(min(100, max(0, score)))

    return {
        "Score"          : score,
        "Price"          : round(price, 2),
        "Entry_Price"    : round(entry_price, 2),
        "Stop_Loss"      : round(stop_loss, 2),
        "Risk_%"         : round(risk_pct, 1),
        "Is_Hammer"      : sig["is_hammer"],
        "Hammer_Quality" : round(sig["hammer_quality"], 2),
        "Near_SMA50_%"   : round(sig["near_pct"], 2),
        "SMA50"          : round(sig["sma50"], 2),
        "Vol_Ratio"      : round(vol_ratio, 2),
        "Low1"           : round(sig["low1"], 2),
        "Low2"           : round(sig["low2"], 2),
        "Low3"           : round(sig["low3"], 2),
        "Signal_Date"    : df_daily.index[sig_idx].strftime("%Y-%m-%d"),
        "Days_Since_Signal"  : days_since_signal,
        "Recent_Signal_Count": len(recent_signals),
        "Backtest_Occurrences": bt_summary["count"],
        "Backtest_Avg_Max_Gain_%"   : (round(bt_summary["avg"], 1)
                                        if bt_summary["avg"] is not None else None),
        "Backtest_Median_Max_Gain_%": (round(bt_summary["median"], 1)
                                        if bt_summary["median"] is not None else None),
        "Backtest_Best_Max_Gain_%"  : (round(bt_summary["best"], 1)
                                        if bt_summary["best"] is not None else None),
        "Backtest_Worst_Max_Gain_%" : (round(bt_summary["worst"], 1)
                                        if bt_summary["worst"] is not None else None),
        "Backtest_HitRate_%"        : (round(bt_summary["hit_rate_pct"], 1)
                                        if bt_summary["hit_rate_pct"] is not None else None),
        "Recent_Signals" : " | ".join(
            f"{s['date'].strftime('%Y-%m-%d')}" for s in recent_signals),
        "Flags"          : " | ".join(reasons),
        "_df_daily"      : df_daily,
        "_sma50"         : sma50, "_sma150": sma150,
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
LIVE_COLS = ["Ticker","Price","Score","Entry_Price","Stop_Loss",
             "Is_Hammer","Vol_Ratio","Signal_Date"]
_CW = {"Ticker":8,"Price":10,"Score":7,"Entry_Price":12,"Stop_Loss":11,
       "Is_Hammer":11,"Vol_Ratio":11,"Signal_Date":13}
_CF = {"Price":"${:.2f}","Score":"{:.0f}","Entry_Price":"${:.2f}",
       "Stop_Loss":"${:.2f}","Vol_Ratio":"{:.2f}x"}
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
print("  A stock only matches if all 3 candles and the structure gate")
print("  fire together within the lookback window\n")

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

for sym in tqdm(list(daily_map.keys()), desc="Checking 3-candle pattern", unit="stk"):
    try:
        r = analyze_three_candle_pattern(sym, daily_map[sym])
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

# ── Full-universe historical backtest summary ──────────────────
BT_SUMMARY = summarize_mfe(ALL_BACKTEST_TRADES, CFG["mfe_hit_threshold_pct"])
print(f"\n{'━'*65}")
print(f"  📈 HISTORICAL BACKTEST — MAX FAVORABLE MOVE AFTER THE PATTERN")
print(f"  Full NASDAQ universe, {CFG['mfe_holding_days']}-day forward window")
print(f"{'━'*65}")
print(f"  Historical occurrences found : {BT_SUMMARY['count']}")
if BT_SUMMARY["count"] > 0:
    print(f"  Average max gain             : {BT_SUMMARY['avg']:+.1f}%")
    print(f"  Median max gain (most probable): {BT_SUMMARY['median']:+.1f}%")
    print(f"  Best historical outcome       : {BT_SUMMARY['best']:+.1f}%")
    print(f"  Worst historical outcome      : {BT_SUMMARY['worst']:+.1f}%")
    print(f"  Hit rate (>= {CFG['mfe_hit_threshold_pct']:.0f}% gain)       : {BT_SUMMARY['hit_rate_pct']:.1f}%")
else:
    print(f"  (no historical occurrences with enough forward data found)")
print(f"{'━'*65}")

if not results:
    print("\n  No matches. Try relaxing:")
    print("   near_sma50_pct              3.0 → 5.0   (allow a looser test of support)")
    print("   doji_body_pct              0.15 → 0.25  (allow a slightly bigger doji body)")
    print("   healthy_body_pct           0.40 → 0.30  (allow a smaller healthy candle)")
    print("   signal_lookback_days         15 → 25    (search further back)")
    print("   min_price                      2 → 1")
    print("   min_avg_volume             80000 → 50000")

results.sort(key=lambda x: x["Score"], reverse=True)

# ── Always build df_out and save/email (even if 0 results) ────
ts      = datetime.today().strftime("%Y%m%d_%H%M")
out_dir = os.environ.get("GITHUB_WORKSPACE", os.getcwd())

COLS = [
    "Ticker","Price","Score",
    "Entry_Price","Stop_Loss","Risk_%",
    "Is_Hammer","Hammer_Quality","Near_SMA50_%","SMA50","Vol_Ratio",
    "Low1","Low2","Low3",
    "Backtest_Occurrences","Backtest_Avg_Max_Gain_%","Backtest_Median_Max_Gain_%",
    "Backtest_Best_Max_Gain_%","Backtest_Worst_Max_Gain_%","Backtest_HitRate_%",
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
    "Risk_%"      : lambda v: f"{v:.1f}%",
    "Hammer_Quality": lambda v: f"{v:.2f}",
    "Near_SMA50_%": lambda v: f"{v:.2f}%",
    "SMA50"       : lambda v: f"${v:.2f}",
    "Vol_Ratio"   : lambda v: f"{v:.2f}x",
    "Low1"        : lambda v: f"${v:.2f}",
    "Low2"        : lambda v: f"${v:.2f}",
    "Low3"        : lambda v: f"${v:.2f}",
    "Backtest_Avg_Max_Gain_%"   : lambda v: f"{v:+.1f}%",
    "Backtest_Median_Max_Gain_%": lambda v: f"{v:+.1f}%",
    "Backtest_Best_Max_Gain_%"  : lambda v: f"{v:+.1f}%",
    "Backtest_Worst_Max_Gain_%" : lambda v: f"{v:+.1f}%",
    "Backtest_HitRate_%"        : lambda v: f"{v:.1f}%",
    "Days_Since_Signal": lambda v: f"{int(v)}d ago",
}

def fmt_v(col, val):
    if val is None or (isinstance(val, float) and np.isnan(val)): return "—"
    try:
        if col in FMT: return FMT[col](val)
    except Exception: pass
    return str(val) if str(val) not in ("nan","None","") else "—"

if _IN_NOTEBOOK and results:
    DISP = ["Ticker","Price","Score","Entry_Price","Stop_Loss",
            "Is_Hammer","Vol_Ratio","Backtest_Median_Max_Gain_%",
            "Backtest_HitRate_%","Signal_Date"]
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
            elif col == "Is_Hammer":
                clr = "#22c55e" if raw else "#94a3b8"
                sty = f"color:{clr};font-weight:600"
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
    <span style="color:#f1f5f9;font-size:15px;font-weight:700">3-Candle Reversal Near SMA50</span>
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
    📈 3-Candle Reversal Near SMA50
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
  SMA50&gt;SMA150, candle 1's low near SMA50 &nbsp;·&nbsp;
  Candle 1: red (hammer preferred) &nbsp;·&nbsp;
  Candle 2: green doji with a higher low than candle 1 &nbsp;·&nbsp;
  Candle 3: healthy green, higher volume than candle 2, higher low
  than candle 2 &nbsp;·&nbsp;
  Entry_Price = candle 3's high (breakout) &nbsp;·&nbsp;
  Stop_Loss = candle 2's low — reasonable defaults, not explicitly
  requested &nbsp;·&nbsp;
  Signal_Date is the exact date the pattern fired (checked over the
  last {CFG['signal_lookback_days']} trading days, not just today)
</div>"""

    display_html(header_html + table_html + legend_html)

elif results:
    # ASCII table (CLI/GitHub Actions mode)
    CLI_COLS = ["Ticker","Price","Score","Entry_Price","Stop_Loss",
                "Is_Hammer","Vol_Ratio","Backtest_Median_Max_Gain_%",
                "Backtest_HitRate_%","Signal_Date"]
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
    tit = f"  3-Candle Reversal Near SMA50   {datetime.today().strftime('%Y-%m-%d')}   {len(df_out)} matches"
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
  Score           0-100 (hammer quality + proximity to SMA50 +
                  volume increase + rising lows + freshness +
                  historical backtest strength)
  Entry_Price     candle 3's high (breakout trigger — a reasonable
                  default, not explicitly requested)
  Stop_Loss       candle 2's low (most recent confirmed higher low)
  Is_Hammer       whether candle 1 met the hammer shape criteria
                  (preferred but not required to match)
  Near_SMA50_%    how far candle 1's low sits from SMA50
  Vol_Ratio       candle 3 volume / candle 2 volume
  Backtest_Occurrences    how many times this pattern fired historically
                          for THIS ticker (full available history)
  Backtest_Median_Max_Gain_%  the most probable max price increase —
                              median of the historical max gains
  Backtest_Best_Max_Gain_%    the single best historical outcome
  Backtest_HitRate_%          % of historical occurrences that reached
                              at least mfe_hit_threshold_pct upside
  Signal_Date     exact calendar date the pattern fired
  Days_Since_Signal  how many trading days ago (0 = today; checked
                     over the last signal_lookback_days trading days)
  ──────────────────────────────────────────────────────""")

# Save
fpath = os.path.join(out_dir, f"three_candle_hammer_doji_{ts}.csv")
df_out.to_csv(fpath, index=False)
print(f"\n  💾 CSV → {fpath}")
tv = os.path.join(out_dir, f"tv_three_candle_hammer_doji_{ts}.txt")
with open(tv,"w") as f:
    f.write(f"###3-Candle Reversal Near SMA50 {datetime.today().strftime('%Y-%m-%d')}\n")
    for r in results: f.write(f"NASDAQ:{r['Ticker']}\n")
print(f"  📋 TradingView → {tv}")
if results:
    print(f"\n  📋 Tickers (comma-separated):")
    print(f"  {', '.join(r['Ticker'] for r in results)}")

# Save backtest trade log (every historical occurrence found, full universe)
bt_fpath = os.path.join(out_dir, f"three_candle_hammer_doji_backtest_{ts}.csv")
bt_df = pd.DataFrame(ALL_BACKTEST_TRADES) if ALL_BACKTEST_TRADES else pd.DataFrame(
    columns=["ticker","date","entry","mfe_pct","is_hammer"])
bt_df.to_csv(bt_fpath, index=False)
print(f"  💾 Backtest trade log → {bt_fpath}  ({len(ALL_BACKTEST_TRADES)} historical occurrences)")

# ── Email with CSV attached ───────────────────────────────
def _send_email(rl, csv_path, bt_csv_path=None):
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
                      "Is_Hammer","Vol_Ratio","Median_MaxGain","HitRate","Signal_Date"]
        )
        rows_e = ""
        for i, r in enumerate(rl[:50]):
            bg  = "#fff" if i % 2 == 0 else "#f0f9ff"
            ticker = r.get("Ticker","—")
            price  = r.get("Price",0) or 0
            score  = r.get("Score",0) or 0
            entry  = r.get("Entry_Price",0) or 0
            stop   = r.get("Stop_Loss",0) or 0
            hammer = "Yes" if r.get("Is_Hammer") else "No"
            vr     = r.get("Vol_Ratio",0) or 0
            sdate  = r.get("Signal_Date","—")
            med_g  = r.get("Backtest_Median_Max_Gain_%")
            med_g_disp = f"{med_g:+.1f}%" if med_g is not None else "n/a"
            hr     = r.get("Backtest_HitRate_%")
            hr_disp = f"{hr:.0f}%" if hr is not None else "n/a"
            rows_e += (
                f'<tr style="background:{bg}">'
                f'<td style="padding:6px 11px;font-size:12px;font-weight:700">{ticker}</td>'
                f'<td style="padding:6px 11px;font-size:12px">${float(price):.2f}</td>'
                f'<td style="padding:6px 11px;font-size:12px;font-weight:700;'
                f'background:#166534;color:#fff;text-align:center">{float(score):.0f}</td>'
                f'<td style="padding:6px 11px;font-size:12px;color:#22c55e">${float(entry):.2f}</td>'
                f'<td style="padding:6px 11px;font-size:12px;color:#ef4444">${float(stop):.2f}</td>'
                f'<td style="padding:6px 11px;font-size:12px;text-align:center">{hammer}</td>'
                f'<td style="padding:6px 11px;font-size:12px;font-weight:600">{float(vr):.2f}x</td>'
                f'<td style="padding:6px 11px;font-size:12px;text-align:center;'
                f'color:#facc15;font-weight:600">{med_g_disp}</td>'
                f'<td style="padding:6px 11px;font-size:12px;text-align:center">{hr_disp}</td>'
                f'<td style="padding:6px 11px;font-size:12px;text-align:center;'
                f'color:#a78bfa;font-weight:600">{sdate}</td>'
                f'</tr>'
            )
        no_results_msg = ('<tr><td colspan="10" style="padding:20px;text-align:center;'
                           'color:#94a3b8;font-size:13px">No matches today</td></tr>')

        html_e = f"""<!DOCTYPE html><html><body style="margin:0;padding:0;
background:#f1f5f9;font-family:'Segoe UI',Arial,sans-serif">
<table width="100%" cellpadding="0" cellspacing="0"><tr><td align="center" style="padding:20px 10px">
<table width="100%" style="max-width:800px;background:#fff;border-radius:12px;
       overflow:hidden;box-shadow:0 4px 16px rgba(0,0,0,0.08)">
  <tr><td style="background:linear-gradient(135deg,#0f172a,#1e3a5f);padding:22px 28px">
<h1 style="margin:0;color:#60a5fa;font-size:20px;font-weight:700">
  📊 3-Candle Reversal Near SMA50
</h1>
<p style="margin:6px 0 0;color:#94a3b8;font-size:12px">
  {datetime.today().strftime('%Y-%m-%d %H:%M UTC')} &nbsp;·&nbsp;
  {cnt} match{'es' if cnt!=1 else ''} found — red/hammer, green doji
  with a higher low, healthy green with higher volume and a higher low
</p>
  </td></tr>
  <tr><td style="padding:14px 28px 4px;background:#0b1220">
<div style="background:#111827;border:1px solid #1f2937;border-radius:8px;padding:12px 16px">
  <p style="margin:0 0 6px;color:#93c5fd;font-size:12px;font-weight:700">
    📈 HISTORICAL BACKTEST — MAX FAVORABLE MOVE AFTER THE PATTERN (full universe, {CFG['mfe_holding_days']}-day window)
  </p>
  <p style="margin:0;color:#cbd5e1;font-size:12px">
    {BT_SUMMARY['count']} historical occurrences
    {f'''&nbsp;·&nbsp;
    Avg: <span style="color:#22c55e">{BT_SUMMARY['avg']:+.1f}%</span> &nbsp;·&nbsp;
    <b style="color:#facc15">Median (most probable): {BT_SUMMARY['median']:+.1f}%</b> &nbsp;·&nbsp;
    Best: <span style="color:#22c55e">{BT_SUMMARY['best']:+.1f}%</span> &nbsp;·&nbsp;
    Hit rate (&ge;{CFG['mfe_hit_threshold_pct']:.0f}%): {BT_SUMMARY['hit_rate_pct']:.1f}%''' if BT_SUMMARY['count'] > 0 else '(no historical occurrences with enough forward data)'}
  </p>
</div>
  </td></tr>
  <tr><td style="padding:16px">
<div style="overflow-x:auto;border-radius:8px;border:1px solid #e2e8f0">
  <table style="border-collapse:collapse;width:100%;min-width:600px">
    <thead><tr>{th_e}</tr></thead>
    <tbody>{rows_e or no_results_msg}</tbody>
  </table>
</div>
<p style="font-size:11px;color:#64748b;margin:8px 0 0">
  📎 Full results + full backtest trade log attached as CSV
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
            f"3-Candle Reversal Near SMA50 (checked over last {CFG['signal_lookback_days']} trading days) — {datetime.today().strftime('%Y-%m-%d')}",
            f"{cnt} matches (red/hammer + green doji higher-low + healthy green higher-volume higher-low)",
            "="*60,
            f"HISTORICAL BACKTEST (full universe, {CFG['mfe_holding_days']}-day forward window):",
            f"  Occurrences: {BT_SUMMARY['count']}",
        ]
        if BT_SUMMARY["count"] > 0:
            plain_lines.append(
                f"  Avg max gain: {BT_SUMMARY['avg']:+.1f}%  "
                f"Median (most probable): {BT_SUMMARY['median']:+.1f}%  "
                f"Best: {BT_SUMMARY['best']:+.1f}%  "
                f"Hit rate (>={CFG['mfe_hit_threshold_pct']:.0f}%): {BT_SUMMARY['hit_rate_pct']:.1f}%"
            )
        plain_lines.append("="*60)
        if rl:
            for r in rl[:50]:
                ticker = r.get("Ticker","—")
                price  = r.get("Price",0) or 0
                score  = r.get("Score",0) or 0
                entry  = r.get("Entry_Price",0) or 0
                stop   = r.get("Stop_Loss",0) or 0
                hammer = "Hammer" if r.get("Is_Hammer") else "NoHammer"
                vr     = r.get("Vol_Ratio",0) or 0
                sdate  = r.get("Signal_Date","—")
                med_g  = r.get("Backtest_Median_Max_Gain_%")
                med_g_disp = f"{med_g:+.1f}%" if med_g is not None else "n/a"
                plain_lines.append(
                    f"{ticker:<7} ${float(price):.2f}  Score:{float(score):.0f}  "
                    f"Entry:${float(entry):.2f}  SL:${float(stop):.2f}  "
                    f"{hammer}  Vol:{float(vr):.2f}x  OwnMedianMaxGain:{med_g_disp}  Signal:{sdate}"
                )
            plain_lines.append("")
            plain_lines.append("Tickers (comma-separated):")
            plain_lines.append(", ".join(r.get("Ticker","") for r in rl))
        else:
            plain_lines.append("No matches today")
        plain_lines.append("\nFull results + full backtest trade log in CSV attachments.")
        plain_e = "\n".join(plain_lines)

        med_disp = (f"{BT_SUMMARY['median']:+.0f}%"
                    if BT_SUMMARY['median'] is not None else "n/a")
        subj = (f"📊 3-Candle Reversal Near SMA50 — {cnt} signal{'s' if cnt!=1 else ''} "
                f"(median max gain: {med_disp})"
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

    for attach_path in [csv_path, bt_csv_path]:
        if attach_path and os.path.exists(attach_path):
            try:
                with open(attach_path, "rb") as f:
                    part = MIMEBase("application", "octet-stream")
                    part.set_payload(f.read())
                encoders.encode_base64(part)
                part.add_header("Content-Disposition",
                    f"attachment; filename={os.path.basename(attach_path)}")
                msg.attach(part)
                sz = os.path.getsize(attach_path)
                print(f"[Email] 📎 Attached: {os.path.basename(attach_path)} ({sz:,} bytes)")
            except Exception as e:
                print(f"[Email] ⚠️  Attach failed for {attach_path}: {e}")

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
    _send_email(results, fpath, bt_fpath)
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
    fig, axes = plt.subplots(len(top),1,figsize=(15,4.2*len(top)),facecolor="#0f172a")
    if len(top)==1: axes=[axes]
    for ax, r in zip(axes, top):
        df_p = r["_df_daily"].tail(90).copy()
        sma50_p  = r["_sma50"].reindex(df_p.index)
        sma150_p = r["_sma150"].reindex(df_p.index)
        ax.set_facecolor("#0f172a")
        ax.plot(df_p.index, df_p["Close"], color="#60a5fa", lw=1.4, label="Close", zorder=4)
        ax.plot(df_p.index, sma50_p,  color="#fbbf24", lw=1.3, ls="-.", label="SMA50",  zorder=3)
        ax.plot(df_p.index, sma150_p, color="#f87171", lw=1.2, ls=":",  label="SMA150", zorder=3)

        # mark the 3 pattern candles by date using their stored lows
        pat_date = pd.to_datetime(r["Signal_Date"])
        if pat_date in df_p.index:
            pos3 = df_p.index.get_loc(pat_date)
            if pos3 >= 2:
                dates3 = df_p.index[pos3-2:pos3+1]
                lows3  = [r["Low1"], r["Low2"], r["Low3"]]
                ax.scatter(dates3, lows3, color="#22c55e", s=50, zorder=6, marker="^",
                           label="Pattern lows (rising)")
        ax.axhline(r["Stop_Loss"], color="#ef4444", lw=1.0, ls="--", alpha=0.85,
                  label=f"Stop ${r['Stop_Loss']:.2f}")
        ax.axhline(r["Entry_Price"], color="#22c55e", lw=1.0, ls="--", alpha=0.85,
                  label=f"Entry ${r['Entry_Price']:.2f}")
        med_g = r.get("Backtest_Median_Max_Gain_%")
        med_g_disp = f"MedGain:{med_g:+.1f}%" if med_g is not None else "MedGain:n/a"
        ax.set_title(
            f"{r['Ticker']}  |  ${r['Price']:.2f}  |  Score {r['Score']}  |  "
            f"{'Hammer' if r['Is_Hammer'] else 'NoHammer'}  |  Vol {r['Vol_Ratio']:.2f}x  |  "
            f"{med_g_disp} ({r['Backtest_Occurrences']}x)  |  {r['Signal_Date']}",
            color="#e2e8f0", fontsize=9, fontweight="bold", pad=7)
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))
        ax.tick_params(colors="#94a3b8", labelsize=8)
        for sp in ax.spines.values(): sp.set_edgecolor("#1e3a5f")
        ax.legend(loc="upper left", facecolor="#1e293b", labelcolor="#e2e8f0",
                  fontsize=7, framealpha=0.9, ncol=2)
        ax.grid(color="#1e3a5f", ls="--", lw=0.5, alpha=0.6)
    plt.suptitle(
        f"3-Candle Reversal Near SMA50  ·  {datetime.today().strftime('%Y-%m-%d')}",
        color="#60a5fa", fontsize=12, fontweight="bold", y=1.001)
    plt.tight_layout()
    cp = os.path.join(out_dir, f"three_candle_hammer_doji_chart_{ts}.png")
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

  0) STRUCTURE: SMA50 > SMA150, AND candle 1's low is within
     near_sma50_pct% of SMA50 — the pattern is testing/touching
     support, not floating far from it.
  1) CANDLE 1 (test day): RED (close < open). A HAMMER shape (long
     lower wick, small body, tiny upper wick) is PREFERRED and
     scored as a bonus, but not required.
  2) CANDLE 2 (next day): GREEN DOJI — tiny body relative to its
     range, close > open — AND its low is HIGHER than candle 1's
     low (support holding, no new low made).
  3) CANDLE 3 (next day): a "healthy" GREEN candle — a real body
     (not a doji), close > open — with volume HIGHER than candle
     2's volume, AND its low HIGHER than candle 2's low.

  If the pattern fired more than once in the window, the MOST
  RECENT occurrence is used for Entry/Stop/scoring.

  📋 DATA SOURCING
  Only 1 download per ticker: daily bars (~750 days, ~3 years) —
  the longer window also gives the historical backtest below a
  meaningful sample size.

  📋 OUTPUT (reasonable defaults — not explicitly requested)
  Entry_Price = candle 3's HIGH (a breakout trigger)
  Stop_Loss   = candle 2's LOW (the most recent confirmed higher low)
  Signal_Date = the exact calendar date the pattern fired
  Days_Since_Signal = how many trading days ago (0 = today)
  Recent_Signals    = every date the pattern fired within the window

  📋 HISTORICAL BACKTEST (new) — max probable price increase
  Re-runs the exact same pattern check (check_three_candle_at) over
  the ENTIRE available history for every ticker scanned — not just
  the live lookback window — using the identical logic as the live
  scanner so the two can never drift apart. For each historical
  occurrence found, measures the Maximum Favorable Excursion: the
  highest price reached over the following mfe_holding_days trading
  days, as a % above that occurrence's entry price (candle 3's high).
    Backtest_Occurrences         = how many times found, this ticker
    Backtest_Avg_Max_Gain_%      = mean of all historical max gains
    Backtest_Median_Max_Gain_%   = MEDIAN — the most probable/typical
                                    max price increase (robust to
                                    outliers, unlike the average)
    Backtest_Best_Max_Gain_%     = the single best historical outcome
    Backtest_Worst_Max_Gain_%    = the single worst historical outcome
    Backtest_HitRate_%           = % of historical occurrences that
                                    reached at least mfe_hit_threshold_pct
  A scan-wide aggregate across the FULL universe (not just today's
  matches) is printed above and shown in the email header. The full
  trade-by-trade log is saved to
  three_candle_hammer_doji_backtest_<timestamp>.csv and attached to
  the email alongside the main results CSV.

  📋 SCORE (0-100)
  Hammer quality (0-20, 0 if candle 1 wasn't a hammer) + proximity
  to SMA50 (0-15) + volume increase strength (0-20) + rising-lows
  magnitude (0-10) + freshness (0-10) + historical backtest strength
  (0-10, based on this ticker's own median max gain) + base points
  for clearing every gate (10)

  💡 BEST SETUPS
  Score > 70                    hammer present, tight to SMA50, strong
                                 volume, strong own history
  Is_Hammer = True                 candle 1 had genuine hammer geometry
  Vol_Ratio > 1.5                    well above candle 2's volume
  Backtest_Median_Max_Gain_% > 5%      this ticker's own history of the
                                        pattern has worked well
  Backtest_HitRate_% > 50%             reached a meaningful gain most
                                        of the time historically
  Days_Since_Signal = 0-3               freshest pattern

  ⚙️  TUNE IF 0 RESULTS
  near_sma50_pct              3.0 → 5.0   (allow a looser test of support)
  doji_body_pct              0.15 → 0.25  (allow a slightly bigger doji body)
  healthy_body_pct           0.40 → 0.30  (allow a smaller healthy candle)
  signal_lookback_days         15 → 25    (search further back)
  min_price                      2 → 1
  min_avg_volume             80000 → 50000

  ⚙️  BACKTEST TUNING
  mfe_holding_days             15 → 20    (longer forward window)
  mfe_hit_threshold_pct       5.0 → 3.0   (easier "hit" bar)
  history_days                750 → 1500  (deeper backtest sample,
                                            slower fetch)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
""")
