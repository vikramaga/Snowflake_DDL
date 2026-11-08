# ============================================================
# NASDAQ / NYSE — Minervini Stage 2 + Volume Dry-Up Pullback to SMA50
#                 + Volume-Surge Green Candle  (minervini.py, v2)
# ============================================================
#
# STAGE 2 — Mark Minervini's Trend Template (all 8 on the signal day):
#   1. Close above SMA150 and SMA200
#   2. SMA150 above SMA200
#   3. SMA200 rising for at least 1 month (higher than 21 sessions ago)
#   4. SMA50 above SMA150 and SMA200
#   5. Close above SMA50
#   6. Close at least 30% above the 52-week low
#   7. Close within 25% of the 52-week high
#   8. RS Rating >= 70 — IBD-style: 40% weight on the last 3 months' return
#      + 20% on each of the 3 quarters before, ranked as a 1-99 percentile
#      against every stock scanned that day
#
# SETUP (Vik's request):
#   A. PULLBACK TOWARD SMA50: a swing high 3-20 sessions before the signal,
#      then a 2-15 session pullback whose low comes within 3% of the SMA50
#      (above it, or up to 3% under it), without any close more than 3% under
#      the SMA50.
#   B. SELL VOLUME DRIED UP: during the pullback the red candles' average
#      volume is at most 75% of the 50-day average volume (measured at the
#      swing high), and selling volume is falling — the red candles in the
#      second half of the pullback average no more than those in the first.
#   C. GREEN CANDLE WITH VOLUME: the signal day is a green candle (close >
#      open) closing above the SMA50, with volume at least 2x the volume of
#      the previous red candle.
#
# Signal day = one of the last 2 sessions. Entry = signal close; Stop = the
# pullback low (flagged when risk > 8%, Minervini's usual max); Target 1 =
# the swing high, Target 2 = entry + 3R. Stop/targets are reasonable
# defaults, not explicitly requested.
# The earlier version of this file (v1: Stage 2 proxy + SMA50 zone + volume
# dry-up) is in the git history.
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
    # history_days vs bars: 560 calendar days -> ~386 business days
    # (pd.bdate_range); needs 252 for the 52-week range and the RS rating.
    "history_days"            : 560,
    "min_bars"                : 260,

    # ── Trend Template ──────────────────────────────────────────
    "sma200_rising_bars"      : 21,    # SMA200 higher than 1 month ago
    "min_above_52w_low_pct"   : 30.0,
    "max_below_52w_high_pct"  : 25.0,
    "min_rs_rating"           : 70,

    # ── A. Pullback toward SMA50 ───────────────────────────────
    "swing_min_ago"           : 3,     # swing high 3..20 sessions before the signal
    "swing_max_ago"           : 20,
    "pullback_min_days"       : 2,
    "pullback_max_days"       : 15,
    "sma50_near_pct"          : 3.0,   # pullback low within 3% above SMA50 ...
    "sma50_undercut_pct"      : 3.0,   # ... or up to 3% under it
    "max_close_below_sma50_pct": 3.0,  # no pullback close > 3% under SMA50

    # ── B. Sell-volume dry-up ──────────────────────────────────
    "vol_avg_bars"            : 50,
    "red_vol_max_pct"         : 75.0,  # red-candle avg vol <= 75% of 50-day avg

    # ── C. Green candle with volume ────────────────────────────
    "green_vol_mult"          : 2.0,   # volume >= 2x the previous red candle

    "target2_r"               : 3.0,
    "max_risk_pct_flag"       : 8.0,
    "signal_lookback_days"    : 2,

    # ── Filters ─────────────────────────────────────────────────
    "min_avg_volume"          : 200_000,
    "min_price"               : 5.0,

    "batch_size"              : 50,
    "batch_sleep"             : 1.5,
}

# ── Indicators ────────────────────────────────────────────────
def calc_indicators(df, cfg):
    c, v = df["Close"], df["Volume"]
    return {
        "s50":  c.rolling(50).mean().values,
        "s150": c.rolling(150).mean().values,
        "s200": c.rolling(200).mean().values,
        "vavg": v.rolling(cfg["vol_avg_bars"]).mean().values,
        "lo52": df["Low"].rolling(252, min_periods=240).min().values,
        "hi52": df["High"].rolling(252, min_periods=240).max().values,
    }

def rs_raw(close, i):
    """IBD-style weighted 12-month performance at bar i (None if too short)."""
    if i < 252:
        return None
    c = close
    q = lambda a, b: c[i - a] / c[i - b] - 1 if c[i - b] > 0 else 0.0
    return 0.4 * q(0, 63) + 0.2 * q(63, 126) + 0.2 * q(126, 189) + 0.2 * q(189, 252)

def check_pattern_at(df, ind, i, cfg):
    """
    SINGLE SOURCE OF TRUTH. Checks signal day i. Computes EVERY step (no
    short-circuit) so the funnel shows where tickers drop out. The RS Rating
    (template point 8) needs every ticker, so it is applied after the scan.
    Returns (passed, details); details is {} only if data is too short.
    """
    rb = cfg["sma200_rising_bars"]
    if i < max(260, cfg["swing_max_ago"] + 2) or i >= len(df):
        return False, {}
    o, h, l, c, v = (df[k].values for k in ("Open", "High", "Low", "Close", "Volume"))
    s50, s150, s200, vavg, lo52, hi52 = (ind[k] for k in ("s50", "s150", "s200", "vavg", "lo52", "hi52"))
    need = (s50[i], s150[i], s200[i], s200[i - rb], lo52[i], hi52[i])
    if not all(np.isfinite(x) for x in need) or lo52[i] <= 0 or hi52[i] <= 0:
        return False, {}

    # Step 1 — Trend Template points 1-7
    above_low = (c[i] / lo52[i] - 1) * 100
    below_high = (1 - c[i] / hi52[i]) * 100
    tt = {
        "tt1_above_150_200": c[i] > s150[i] and c[i] > s200[i],
        "tt2_150_over_200": s150[i] > s200[i],
        "tt3_200_rising": s200[i] > s200[i - rb],
        "tt4_50_over_150_200": s50[i] > s150[i] and s50[i] > s200[i],
        "tt5_above_50": c[i] > s50[i],
        "tt6_30pct_above_low": above_low >= cfg["min_above_52w_low_pct"],
        "tt7_within_25pct_high": below_high <= cfg["max_below_52w_high_pct"],
    }
    step1 = all(tt.values())

    # Step 2 — swing high, then a pullback toward the SMA50
    hi = max(range(i - cfg["swing_max_ago"], i - cfg["swing_min_ago"] + 1), key=lambda k: h[k])
    pb = list(range(hi + 1, i))
    lo = min(pb, key=lambda k: l[k]) if pb else i - 1
    near = bool(pb) and np.isfinite(s50[lo]) and \
        s50[lo] * (1 - cfg["sma50_undercut_pct"] / 100) <= l[lo] <= s50[lo] * (1 + cfg["sma50_near_pct"] / 100)
    held = all(c[k] >= s50[k] * (1 - cfg["max_close_below_sma50_pct"] / 100) for k in pb)
    step2 = bool(cfg["pullback_min_days"] <= len(pb) <= cfg["pullback_max_days"] and near and held)

    # Step 3 — sell volume dried up (red candles light, and getting lighter)
    reds = [k for k in pb if c[k] < o[k]]
    base_vol = vavg[hi] if np.isfinite(vavg[hi]) and vavg[hi] > 0 else np.nan
    red_avg = float(np.mean([v[k] for k in reds])) if reds else np.nan
    red_pct = red_avg / base_vol * 100 if reds and np.isfinite(base_vol) else np.nan
    half = len(pb) // 2
    r1 = [v[k] for k in reds if k < pb[0] + half] if pb else []
    r2 = [v[k] for k in reds if k >= pb[0] + half] if pb else []
    falling = (not r1 or not r2) or (np.mean(r2) <= np.mean(r1))
    step3 = bool(reds and np.isfinite(red_pct) and red_pct <= cfg["red_vol_max_pct"] and falling)

    # Step 4 — green candle above SMA50 with volume >= 2x the previous red candle
    prev_red = next((k for k in range(i - 1, max(i - 30, 0), -1) if c[k] < o[k]), None)
    vol_mult = float(v[i] / v[prev_red]) if prev_red is not None and v[prev_red] > 0 else 0.0
    step4 = bool(c[i] > o[i] and c[i] > s50[i] and vol_mult >= cfg["green_vol_mult"])

    passed = step1 and step2 and step3 and step4
    return passed, {
        "date": df.index[i], "i": i, "swing_date": df.index[hi], "swing_high": float(h[hi]),
        "low_date": df.index[lo], "pullback_low": float(l[lo]), "pullback_days": len(pb),
        "low_vs_sma50": float((l[lo] / s50[lo] - 1) * 100) if np.isfinite(s50[lo]) else np.nan,
        "red_days": len(reds), "red_vol_pct": float(red_pct) if np.isfinite(red_pct) else np.nan,
        "prev_red_date": df.index[prev_red] if prev_red is not None else None,
        "vol_mult": vol_mult, "close": float(c[i]), "open": float(o[i]),
        "sma50": float(s50[i]), "sma150": float(s150[i]), "sma200": float(s200[i]),
        "above_low": float(above_low), "below_high": float(below_high), "tt": tt,
        "step1": step1, "step2": step2, "step3": step3, "step4": step4,
    }

# ── Diagnostic funnel (per ticker: did ANY of the lookback days pass
#    steps 1..k cumulatively); the RS step is counted after the scan ──────
FUNNEL_COUNTS = {
    "tickers_checked": 0,
    "passed_step1_trend_template": 0,
    "passed_step2_pullback_to_sma50": 0,
    "passed_step3_sell_volume_dried_up": 0,
    "passed_step4_green_2x_volume": 0,
    "passed_step5_rs_rating": 0,
}
_FUNNEL_KEYS = list(FUNNEL_COUNTS)[1:5]
TT_FAILS = {}

def find_signals(df, ind, cfg):
    hits, best = [], [False] * 4
    n = len(df)
    any_checked = False
    for i in range(max(0, n - cfg["signal_lookback_days"]), n):
        ok, d = check_pattern_at(df, ind, i, cfg)
        if not d: continue
        any_checked = True
        if i == n - 1:
            for k_, val in d["tt"].items():
                if not val: TT_FAILS[k_] = TT_FAILS.get(k_, 0) + 1
        cum = True
        for k in range(4):
            cum = cum and d[f"step{k+1}"]
            best[k] |= cum
        if ok: hits.append(d)
    if any_checked:
        FUNNEL_COUNTS["tickers_checked"] += 1
        for k, key in enumerate(_FUNNEL_KEYS):
            FUNNEL_COUNTS[key] += best[k]
    return hits

def analyze_minervini_dryup(sym, df_daily):
    """Returns (result or None, raw RS for this ticker)."""
    if df_daily is None: return None, None
    close = df_daily["Close"].values
    rs = rs_raw(close, len(close) - 1)
    price   = float(close[-1])
    avg_vol = float(df_daily["Volume"].tail(20).mean())
    if price   < CFG["min_price"]:      return None, rs
    if avg_vol < CFG["min_avg_volume"]: return None, rs
    if len(df_daily) < CFG["min_bars"]: return None, rs

    ind = calc_indicators(df_daily, CFG)
    hits = find_signals(df_daily, ind, CFG)
    if not hits: return None, rs
    sig = hits[-1]   # most recent hit wins

    entry = sig["close"]
    stop = sig["pullback_low"]
    if entry <= stop: return None, rs
    risk = entry - stop
    risk_pct = risk / entry * 100
    t1, t2 = sig["swing_high"], entry + CFG["target2_r"] * risk
    r1 = (t1 - entry) / risk
    i = sig["i"]
    days_ago = len(df_daily) - 1 - i
    after = df_daily.iloc[i + 1:]
    flags = [f"pullback {sig['pullback_days']}d to {sig['low_vs_sma50']:+.1f}% vs SMA50",
             f"red vol {sig['red_vol_pct']:.0f}% of avg", f"green vol {sig['vol_mult']:.1f}x prev red"]
    if risk_pct > CFG["max_risk_pct_flag"]: flags.append(f"Risk {risk_pct:.1f}% > {CFG['max_risk_pct_flag']:.0f}%")
    if len(after) and after["High"].max() >= t1: flags.append("Swing high already retested")
    if len(after) and after["Low"].min() <= stop: flags.append("Stop already hit")

    score = 20
    score += min(15, max(0, (sig["vol_mult"] - 2) * 7.5 + 7))          # volume surge on the green candle
    score += min(15, max(0, (75 - sig["red_vol_pct"]) * 0.4))          # drier selling
    score += max(0, 10 - abs(sig["low_vs_sma50"]) * 3)                 # low right at SMA50
    score += max(0, 10 - sig["below_high"] * 0.6)                      # near the 52-week high
    score += max(0, 10 - max(0, risk_pct - 3) * 2)                     # tight stop
    score = round(min(100, max(0, score)))                             # + up to 20 for RS, added later

    return {
        "Score": score, "Price": round(price, 2),
        "Entry_Price": round(entry, 2), "Stop_Loss": round(stop, 2), "Risk_%": round(risk_pct, 1),
        "Target_1_Swing_High": round(t1, 2), "T1_R": round(r1, 2), "Target_2_3R": round(t2, 2),
        "Swing_High_Date": sig["swing_date"].strftime("%Y-%m-%d"),
        "Pullback_Low_Date": sig["low_date"].strftime("%Y-%m-%d"),
        "Pullback_%": round((sig["swing_high"] - stop) / sig["swing_high"] * 100, 1),
        "Pullback_Days": sig["pullback_days"],
        "Low_vs_SMA50_%": round(sig["low_vs_sma50"], 1),
        "Red_Days": sig["red_days"], "Red_Vol_%_of_Avg": round(sig["red_vol_pct"], 0),
        "Prev_Red_Date": sig["prev_red_date"].strftime("%Y-%m-%d") if sig["prev_red_date"] is not None else "",
        "Green_Vol_x_Prev_Red": round(sig["vol_mult"], 2),
        "SMA50": round(sig["sma50"], 2), "SMA150": round(sig["sma150"], 2), "SMA200": round(sig["sma200"], 2),
        "Above_52w_Low_%": round(sig["above_low"], 1), "Below_52w_High_%": round(sig["below_high"], 1),
        "Signal_Date": sig["date"].strftime("%Y-%m-%d"), "Days_Since_Signal": int(days_ago),
        "Flags": " | ".join(flags),
        "_df": df_daily, "_ind": ind,
    }, rs

# ── Download: daily bars ─────────────────────────────────────────────
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
                        df = _clean(df, min_bars=150)
                        if df is not None: out[sym] = df
                    except Exception: pass
            elif len(symbols) == 1:
                df = _clean(raw, min_bars=150)
                if df is not None: out[symbols[0]] = df
    except Exception: pass
    for sym in [s for s in symbols if s not in out]:
        for _ in range(2):
            try:
                df = yf.Ticker(sym).history(
                    start=start.strftime("%Y-%m-%d"),
                    end=end.strftime("%Y-%m-%d"),
                    auto_adjust=True, actions=False)
                df = _clean(df, min_bars=150)
                if df is not None: out[sym] = df; break
            except Exception: time.sleep(0.2)
        time.sleep(0.04)
    return out

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

# ── Main scan ─────────────────────────────────────────────────
print("━"*65)
print(f"  STEP 3  SCANNING {len(TICKERS)} TICKERS")
print("━"*65)

results, no_daily_data, daily_map = [], 0, {}
daily_batches = [TICKERS[i:i+CFG["batch_size"]]
                 for i in range(0, len(TICKERS), CFG["batch_size"])]
with tqdm(total=len(TICKERS), desc="Daily fetch", unit="stk") as pbar:
    for batch in daily_batches:
        got = download_daily(batch, CFG["history_days"])
        daily_map.update(got)
        no_daily_data += len(batch) - len(got)
        pbar.update(len(batch))
        time.sleep(CFG["batch_sleep"])
got_daily = len(TICKERS) - no_daily_data
print(f"\n  Daily data : {got_daily}/{len(TICKERS)} tickers\n")

raw_rs, candidates = {}, []
for sym in tqdm(list(daily_map.keys()), desc="Checking pattern", unit="stk"):
    try:
        r, rs = analyze_minervini_dryup(sym, daily_map[sym])
        if rs is not None and np.isfinite(rs): raw_rs[sym] = rs
        if r is None: continue
        r["Ticker"] = sym
        candidates.append(r)
    except Exception: pass

# Trend Template point 8 — RS Rating 1-99 against every stock scanned today
rs_rank = pd.Series(raw_rs).rank(pct=True) * 98 + 1 if raw_rs else pd.Series(dtype=float)
for r in candidates:
    rating = int(round(rs_rank.get(r["Ticker"], 0)))
    r["RS_Rating"] = rating
    if rating >= CFG["min_rs_rating"]:
        r["Score"] = int(min(100, r["Score"] + max(0, rating - 70) * 20 / 29))
        results.append(r)
        print(f"  ✅ {r['Ticker']:<6} ${r['Price']:.2f}  Score {r['Score']}  RS {rating}  pullback {r['Pullback_%']}% "
              f"({r['Low_vs_SMA50_%']:+}% vs SMA50)  red vol {r['Red_Vol_%_of_Avg']:.0f}%  "
              f"green {r['Green_Vol_x_Prev_Red']}x  {r['Signal_Date']}")
FUNNEL_COUNTS["passed_step5_rs_rating"] = len(results)

fc = FUNNEL_COUNTS
print(f"\n{'━'*65}\n  SCAN COMPLETE — {len(results)} matches from {got_daily} tickers with data "
      f"(RS ranked over {len(raw_rs)})")
print(f"  🔍 FUNNEL (any of the last {CFG['signal_lookback_days']} sessions)")
print(f"  Tickers checked                           : {fc['tickers_checked']}")
print(f"  Step 1 — Trend Template points 1-7          : {fc['passed_step1_trend_template']}")
print(f"  Step 2 — + pullback toward SMA50            : {fc['passed_step2_pullback_to_sma50']}")
print(f"  Step 3 — + sell volume dried up             : {fc['passed_step3_sell_volume_dried_up']}")
print(f"  Step 4 — + green candle, vol >= {CFG['green_vol_mult']:.0f}x prev red : {fc['passed_step4_green_2x_volume']}")
print(f"  Step 5 — + RS Rating >= {CFG['min_rs_rating']}               : {fc['passed_step5_rs_rating']}  (= full pattern)")
if TT_FAILS:
    print("  Trend Template failures (latest session): " +
          ", ".join(f"{k} {v}" for k, v in sorted(TT_FAILS.items(), key=lambda x: -x[1])))
print(f"{'━'*65}")

results.sort(key=lambda x: x["Score"], reverse=True)

# ── Save CSV + TradingView list ───────────────────────────────
ts      = datetime.today().strftime("%Y%m%d_%H%M")
out_dir = os.environ.get("GITHUB_WORKSPACE", os.getcwd())
COLS = ["Ticker","Price","Score","RS_Rating","Entry_Price","Stop_Loss","Risk_%",
        "Target_1_Swing_High","T1_R","Target_2_3R",
        "Swing_High_Date","Pullback_Low_Date","Pullback_%","Pullback_Days","Low_vs_SMA50_%",
        "Red_Days","Red_Vol_%_of_Avg","Prev_Red_Date","Green_Vol_x_Prev_Red",
        "SMA50","SMA150","SMA200","Above_52w_Low_%","Below_52w_High_%",
        "Signal_Date","Days_Since_Signal","Flags"]
df_out = pd.DataFrame([{k: v for k, v in r.items() if not k.startswith("_")}
                       for r in results]) if results else pd.DataFrame(columns=COLS)
df_out = df_out[[c for c in COLS if c in df_out.columns]].reset_index(drop=True)

fpath = os.path.join(out_dir, f"minervini_stage2_vcp_{ts}.csv")
try:
    from position_sizing import add_position_size
    df_out = add_position_size(df_out)
    try: __import__("best_picks").rank_best(df_out)
    except Exception as _bp_err: print(f"  ⚠️  Best 10 skipped: {_bp_err}")
except Exception as _ps_err:
    print(f"  ⚠️  Position sizing skipped: {_ps_err}")
df_out.to_csv(fpath, index=False)
print(f"\n  💾 CSV → {fpath}")
tv = os.path.join(out_dir, f"tv_minervini_stage2_dryup_{ts}.txt")
with open(tv, "w") as f:
    f.write(f"###Minervini Stage2 Dry-up {datetime.today().strftime('%Y-%m-%d')}\n")
    for r in results: f.write(f"NASDAQ:{r['Ticker']}\n")
print(f"  📋 TradingView → {tv}")
if results:
    print(f"\n  📋 Tickers: {', '.join(r['Ticker'] for r in results)}")

# ── Email with CSV attached ───────────────────────────────────
EMAIL_COLS = ["Ticker","Price","Score","RS_Rating","Entry_Price","Stop_Loss","Risk_%",
              "Target_1_Swing_High","Pullback_%","Low_vs_SMA50_%","Red_Vol_%_of_Avg",
              "Green_Vol_x_Prev_Red","Below_52w_High_%","Signal_Date"]

def _fmt_cell(col, v):
    try:
        if col in ("Price","Entry_Price","Stop_Loss","Target_1_Swing_High"): return f"${float(v):.2f}"
        if col in ("Risk_%","Pullback_%","Below_52w_High_%"): return f"{float(v):.1f}%"
        if col == "Low_vs_SMA50_%": return f"{float(v):+.1f}%"
        if col == "Red_Vol_%_of_Avg": return f"{float(v):.0f}%"
        if col == "Green_Vol_x_Prev_Red": return f"{float(v):.1f}x"
        if col in ("Score","RS_Rating"): return f"{float(v):.0f}"
    except Exception: pass
    return "—" if v is None else str(v)

def _send_email(rl, csv_path):
    import smtplib
    from email.mime.multipart import MIMEMultipart
    from email.mime.text      import MIMEText
    from email.mime.base      import MIMEBase
    from email                import encoders

    gu = _GMAIL_USER; gp = _GMAIL_PASS; et = _EMAIL_TO
    if not (gu and gp and et):
        print("[Email] ❌  GMAIL_USER / GMAIL_PASS / EMAIL_TO missing — skipped")
        return
    eto = [e.strip() for e in et.split(",") if e.strip()]
    cnt = len(rl)
    fc = FUNNEL_COUNTS
    try:
        th = "".join(f'<th style="background:#1e293b;color:#e2e8f0;padding:8px 10px;'
                     f'font-size:11px;border-bottom:2px solid #3b82f6;white-space:nowrap">{c}</th>'
                     for c in EMAIL_COLS)
        rows = ""
        for i, r in enumerate(rl[:50]):
            bg = "#fff" if i % 2 == 0 else "#f0f9ff"
            tds = ""
            for c in EMAIL_COLS:
                sty = "font-weight:700" if c == "Ticker" else ""
                if c == "Score": sty = "font-weight:700;background:#166534;color:#fff;text-align:center"
                if c == "Stop_Loss": sty = "color:#ef4444"
                if c == "Entry_Price": sty = "color:#16a34a"
                if c == "Target_1_Swing_High": sty = "color:#2563eb;font-weight:600"
                if c == "Risk_%" and float(r.get(c) or 0) > CFG["max_risk_pct_flag"]: sty = "color:#b45309;font-weight:700"
                tds += f'<td style="padding:6px 10px;font-size:12px;{sty}">{_fmt_cell(c, r.get(c))}</td>'
            rows += f'<tr style="background:{bg}">{tds}</tr>'
        if not rows:
            rows = (f'<tr><td colspan="{len(EMAIL_COLS)}" style="padding:20px;text-align:center;'
                    f'color:#94a3b8">No matches today</td></tr>')
        tick_html = (f'<p style="margin:10px 0 0;font-size:12px;font-family:monospace">'
                     f'{", ".join(r["Ticker"] for r in rl)}</p>') if rl else ""
        html = f"""<!DOCTYPE html><html><body style="margin:0;background:#f1f5f9;font-family:'Segoe UI',Arial,sans-serif">
<table width="100%" cellpadding="0" cellspacing="0"><tr><td align="center" style="padding:20px 10px">
<table width="100%" style="max-width:900px;background:#fff;border-radius:12px;overflow:hidden">
<tr><td style="background:linear-gradient(135deg,#0f172a,#1e3a5f);padding:20px 26px">
<h1 style="margin:0;color:#60a5fa;font-size:19px">📊 Minervini Stage 2 — Volume Dry-Up Pullback to SMA50</h1>
<p style="margin:6px 0 0;color:#94a3b8;font-size:12px">{datetime.today().strftime('%Y-%m-%d %H:%M UTC')} · {cnt} match{'es' if cnt != 1 else ''}
 — all 8 Trend Template points (RS Rating ≥ {CFG['min_rs_rating']}), a pullback toward the SMA50 on drying-up selling volume
 (red candles ≤ {CFG['red_vol_max_pct']:.0f}% of average), then a green candle with ≥ {CFG['green_vol_mult']:.0f}x the previous red candle's volume</p>
</td></tr>
<tr><td style="padding:12px 26px;background:#0b1220"><p style="margin:0;color:#cbd5e1;font-size:12px">
🔍 FUNNEL: {fc['tickers_checked']} checked → {fc['passed_step1_trend_template']} Trend Template 1-7
→ {fc['passed_step2_pullback_to_sma50']} + pullback to SMA50 → {fc['passed_step3_sell_volume_dried_up']} + selling dried up
→ {fc['passed_step4_green_2x_volume']} + green {CFG['green_vol_mult']:.0f}x volume
→ <b style="color:#facc15">{fc['passed_step5_rs_rating']} + RS ≥ {CFG['min_rs_rating']} = full match</b></p></td></tr>
<tr><td style="padding:16px"><div style="overflow-x:auto;border:1px solid #e2e8f0;border-radius:8px">
<table style="border-collapse:collapse;width:100%"><thead><tr>{th}</tr></thead><tbody>{rows}</tbody></table></div>
{tick_html}
<p style="font-size:11px;color:#64748b;margin:10px 0 0">Entry = the signal day's close · Stop = the pullback low
(orange risk = above Minervini's usual {CFG['max_risk_pct_flag']:.0f}% max) · Target 1 = the swing high before the pullback,
Target 2 = entry + {CFG['target2_r']:.0f}R (in the CSV). RS Rating = IBD-style 12-month weighted performance ranked 1-99
against all stocks scanned. 📎 Full results (with Quantity for a fixed max loss) attached as CSV.</p>
</td></tr>
<tr><td style="background:#f8fafc;padding:10px;text-align:center;color:#94a3b8;font-size:10px">
⚠️ Not financial advice · Auto-generated by GitHub Actions</td></tr>
</table></td></tr></table></body></html>"""

        plain = [f"Minervini Stage 2 — Volume Dry-Up Pullback to SMA50 — {datetime.today().strftime('%Y-%m-%d')}",
                 f"{cnt} matches",
                 f"FUNNEL: {fc['tickers_checked']} checked -> {fc['passed_step1_trend_template']} trend template -> "
                 f"{fc['passed_step2_pullback_to_sma50']} pullback to SMA50 -> {fc['passed_step3_sell_volume_dried_up']} selling dried up -> "
                 f"{fc['passed_step4_green_2x_volume']} green 2x volume -> {fc['passed_step5_rs_rating']} RS >= {CFG['min_rs_rating']} (=full match)", "="*60]
        for r in rl[:50]:
            plain.append(f"{r['Ticker']:<6} ${r['Price']:.2f}  Score:{r['Score']}  RS:{r['RS_Rating']}  Entry:${r['Entry_Price']:.2f}  "
                         f"SL:${r['Stop_Loss']:.2f} ({r['Risk_%']}%)  T1:${r['Target_1_Swing_High']:.2f}  "
                         f"red vol {r['Red_Vol_%_of_Avg']:.0f}%  green {r['Green_Vol_x_Prev_Red']}x  {r['Signal_Date']}")
        if rl:
            plain += ["", "Tickers: " + ", ".join(r["Ticker"] for r in rl)]
        else:
            plain.append("No matches today")
        plain.append("\nFull results in CSV attachment.")

        subj = (f"📊 Minervini Stage 2 Dry-Up — "
                f"{cnt} signal{'s' if cnt != 1 else ''} — {datetime.today().strftime('%Y-%m-%d')}")
        msg = MIMEMultipart("mixed")
        msg["Subject"] = subj; msg["From"] = gu; msg["To"] = ", ".join(eto)
        alt = MIMEMultipart("alternative")
        alt.attach(MIMEText("\n".join(plain), "plain"))
        alt.attach(MIMEText(html, "html"))
        msg.attach(alt)
    except Exception as e:
        print(f"[Email] ❌  Failed to build email body: {type(e).__name__}: {e}")
        return

    if csv_path and os.path.exists(csv_path):
        try:
            with open(csv_path, "rb") as f:
                part = MIMEBase("application", "octet-stream"); part.set_payload(f.read())
            encoders.encode_base64(part)
            part.add_header("Content-Disposition", f"attachment; filename={os.path.basename(csv_path)}")
            msg.attach(part)
        except Exception as e:
            print(f"[Email] ⚠️  CSV attach failed: {e}")
    try:
        with smtplib.SMTP_SSL("smtp.gmail.com", 465) as srv:
            srv.login(gu, gp.replace(" ", ""))
            try: __import__("best_picks").attach_to_email(msg)
            except Exception as _bp_err: print(f"  ⚠️  Best 10 not added: {_bp_err}")
            srv.sendmail(gu, eto, msg.as_string())
        print(f"[Email] ✅  Sent to {', '.join(eto)} — {subj}")
    except Exception as e:
        print(f"[Email] ❌  {type(e).__name__}: {e}")

try:
    _send_email(results, fpath)
except Exception as e:
    print(f"[Email] ❌  Unexpected top-level error: {type(e).__name__}: {e}")

# ── Charts for top 5: last 70 daily candles, SMA50/150/200, volume ─
if results:
    top = results[:5]
    fig, axes = plt.subplots(len(top), 1, figsize=(14, 4.6*len(top)), facecolor="#0f172a")
    if len(top) == 1: axes = [axes]
    for ax, r in zip(axes, top):
        df = r["_df"]; ind = r["_ind"]; k = df.index.get_loc(pd.Timestamp(r["Signal_Date"]))
        lo = max(0, k - 64); seg = df.iloc[lo:]
        x = np.arange(len(seg))
        ax.set_facecolor("#0f172a")
        cols = ["#22c55e" if c > o else "#ef4444" for o, c in zip(seg["Open"], seg["Close"])]
        ax.vlines(x, seg["Low"], seg["High"], color=cols, lw=1)
        ax.bar(x, (seg["Close"] - seg["Open"]).abs(), bottom=np.minimum(seg["Open"], seg["Close"]),
               color=cols, width=0.6)
        for key, c_, lab, lw in (("s50", "#86efac", "SMA50", 1.4), ("s150", "#60a5fa", "SMA150", 1.1),
                                 ("s200", "#f87171", "SMA200", 1.1)):
            ax.plot(x, ind[key][lo:], color=c_, lw=lw, label=lab)
        for lvl, c_, lab in ((r["Entry_Price"], "#e2e8f0", "Entry"), (r["Stop_Loss"], "#ef4444", "Stop"),
                             (r["Target_1_Swing_High"], "#a78bfa", "T1")):
            ax.axhline(lvl, color=c_, ls=":", lw=0.8)
            ax.text(len(seg) - 0.5, lvl, f" {lab} {lvl:.2f}", color=c_, fontsize=7, va="center")
        ax.axvline(k - lo, color="#facc15", ls="--", lw=0.8)
        axv = ax.twinx()
        axv.bar(x, seg["Volume"], color=cols, alpha=0.25, width=0.6)
        axv.set_ylim(0, seg["Volume"].max() * 4); axv.set_yticks([])
        ax.legend(loc="upper left", facecolor="#1e293b", labelcolor="#e2e8f0", fontsize=7, ncol=3)
        step = max(1, len(seg) // 8)
        ax.set_xticks(x[::step]); ax.set_xticklabels([d.strftime("%m/%d") for d in seg.index[::step]], fontsize=7)
        ax.set_title(f"{r['Ticker']} | USD {r['Price']:.2f} | Score {r['Score']} | RS {r['RS_Rating']} | "
                     f"red vol {r['Red_Vol_%_of_Avg']:.0f}% | green {r['Green_Vol_x_Prev_Red']}x prev red",
                     color="#e2e8f0", fontsize=9, fontweight="bold")
        ax.tick_params(colors="#94a3b8", labelsize=7)
        for sp in ax.spines.values(): sp.set_edgecolor("#1e3a5f")
        ax.grid(color="#1e3a5f", ls="--", lw=0.5, alpha=0.6)
    plt.suptitle(f"Minervini Stage 2 — Volume Dry-Up Pullback to SMA50 · "
                 f"{datetime.today().strftime('%Y-%m-%d')}", color="#60a5fa", fontsize=12, fontweight="bold")
    plt.tight_layout()
    cp = os.path.join(out_dir, f"minervini_stage2_dryup_chart_{ts}.png")
    plt.savefig(cp, dpi=130, bbox_inches="tight", facecolor="#0f172a")
    plt.close()
    print(f"  📊 Chart → {cp}")

print(f"""
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  📋 MINERVINI TREND TEMPLATE (all 8)
  1) Close > SMA150 and SMA200   2) SMA150 > SMA200   3) SMA200 rising {CFG['sma200_rising_bars']} sessions
  4) SMA50 > SMA150 and SMA200   5) Close > SMA50
  6) >= {CFG['min_above_52w_low_pct']:.0f}% above 52-week low   7) within {CFG['max_below_52w_high_pct']:.0f}% of 52-week high
  8) RS Rating >= {CFG['min_rs_rating']} (IBD-style, ranked against all stocks scanned)
  SETUP (signal day = one of the last {CFG['signal_lookback_days']} sessions)
  A) Swing high {CFG['swing_min_ago']}-{CFG['swing_max_ago']} sessions back, {CFG['pullback_min_days']}-{CFG['pullback_max_days']} day pullback, low within
     +{CFG['sma50_near_pct']:.0f}% / -{CFG['sma50_undercut_pct']:.0f}% of SMA50, no close > {CFG['max_close_below_sma50_pct']:.0f}% under it
  B) Red candles average <= {CFG['red_vol_max_pct']:.0f}% of the {CFG['vol_avg_bars']}-day volume and are getting lighter
  C) Green candle closing above SMA50 with volume >= {CFG['green_vol_mult']:.0f}x the previous red candle
  OUTPUT: Entry = signal close, Stop = pullback low, T1 = swing high, T2 = entry + {CFG['target2_r']:.0f}R
  ⚙️  If 0 results, check the FUNNEL: relax red_vol_max_pct 75 → 85,
     green_vol_mult 2.0 → 1.5, or signal_lookback_days 2 → 5
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
""")
