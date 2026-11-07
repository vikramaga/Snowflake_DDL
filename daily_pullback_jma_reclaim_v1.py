# ============================================================
# NASDAQ / NYSE — Daily Pullback to the 50 → JMA Reclaim (v1)
# ============================================================
#
# Vik's RVTY pattern (14 Sep 2026): a stock in an uptrend pulls back for a
# few days, the JMA rolls over (turns red on the chart), price dips into the
# EMA21 / SMA50 zone and holds, then a strong green candle closes back above
# the JMA and EMA21.
#
# SIGNAL (all required; signal day = one of the last signal_lookback_days
# sessions; the most recent hit wins):
#
#   1. UPTREND: SMA50 above both SMA150 and SMA200 (their own order does not
#      matter), SMA50 and SMA200 higher than 20 sessions ago, close > SMA150.
#
#   2. JMA PULLBACK: the swing high is 3-15 sessions before the signal day,
#      followed by 2-10 pullback sessions in which JMA(13, 40, 2) falls on at
#      least 2 days and is still falling or flat the day before the signal.
#
#   3. REACHED THE 21/50 ZONE: the pullback low is within 0.5% of the higher
#      of EMA21 / SMA50 (or below it).
#
#   4. HELD: the pullback is at least 1.5 ATR(14) deep, no pullback close is
#      more than 3% under the SMA50 and no low is under the SMA150.
#
#   5. RECLAIM CANDLE: green, body >= 0.6 ATR(14), closes back above the JMA
#      (the day before closed at or below it) and above EMA21.
#
# OUTPUT: Entry_Price = signal day's close; Stop_Loss = pullback low;
# Target_1 = the swing high (retest), Target_2 = entry + 2R. Stop and
# targets are reasonable defaults, not explicitly requested.
# The rule was checked against RVTY: it fires on 14 Sep 2026 only, in the
# past year.
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
    # history_days vs bars needed: 560 calendar days -> ~400 business days
    # (checked with pd.bdate_range); the pattern needs 240 (SMA200 + 20 + 20).
    "history_days"            : 560,
    "min_bars"                : 240,

    # ── Moving averages ─────────────────────────────────────────
    "jma_period"              : 13,
    "jma_phase"               : 40,
    "jma_power"               : 2,
    "ema_period"              : 21,
    "sma_fast"                : 50,
    "sma_mid"                 : 150,
    "sma_slow"                : 200,
    "slope_lookback"          : 20,    # SMA50 / SMA200 higher than 20 sessions ago
    "atr_period"              : 14,    # Wilder ATR

    # ── Pullback ────────────────────────────────────────────────
    "swing_min_ago"           : 3,     # swing high 3..15 sessions before the signal
    "swing_max_ago"           : 15,
    "pullback_min_days"       : 2,
    "pullback_max_days"       : 10,
    "jma_min_down_days"       : 2,     # JMA falls on at least 2 pullback days
    "zone_tolerance_pct"      : 0.5,   # low within 0.5% of max(EMA21, SMA50)
    "min_depth_atr"           : 1.5,   # pullback at least 1.5 ATR deep
    "max_close_below_sma50_pct": 3.0,  # no pullback close > 3% under SMA50
    # ── Reclaim candle ──────────────────────────────────────────
    "min_body_atr"            : 0.6,

    "target2_r"               : 2.0,   # Target 2 = entry + 2R
    "signal_lookback_days"    : 2,     # latest session + the one before

    # ── Filters ─────────────────────────────────────────────────
    "min_avg_volume"          : 200_000,
    "min_price"               : 5.0,

    "batch_size"              : 50,
    "batch_sleep"             : 1.5,
}

# ── Indicators ───────────────────────────────────────────────────────────────
def calc_jma(series, period=13, phase=40, power=2):
    """
    Jurik Moving Average — same formula as TradingView's "Jurik Moving
    Average" indicator, so values match the chart:
        beta       = 0.45*(L-1) / (0.45*(L-1) + 2)
        alpha      = beta ** power
        phaseRatio = phase/100 + 1.5   (0.5 if phase < -100, 2.5 if > 100)
        e0  = (1-alpha)*src + alpha*e0[1]
        e1  = (src - e0)*(1-beta) + beta*e1[1]
        e2  = (e0 + phaseRatio*e1 - jma[1])*(1-alpha)^2 + alpha^2*e2[1]
        jma = e2 + jma[1]
    period = Length, phase = Phase, power = Power (TradingView inputs).
    """
    phase_ratio = 0.5 if phase < -100 else (2.5 if phase > 100 else phase / 100.0 + 1.5)
    beta  = 0.45 * (period - 1) / (0.45 * (period - 1) + 2)
    alpha = beta ** power
    vals  = np.asarray(series.values, dtype=float)
    out   = np.full(len(vals), np.nan)
    e0 = e1 = e2 = jma = 0.0
    for i, v in enumerate(vals):
        if np.isnan(v):
            continue
        e0  = (1 - alpha) * v + alpha * e0
        e1  = (v - e0) * (1 - beta) + beta * e1
        e2  = (e0 + phase_ratio * e1 - jma) * (1 - alpha) ** 2 + alpha ** 2 * e2
        jma = e2 + jma
        out[i] = jma
    return pd.Series(out, index=series.index)

def calc_atr(df, period):
    """Wilder ATR (RMA of the true range)."""
    h, l, c = df["High"], df["Low"], df["Close"]
    pc = c.shift(1)
    tr = pd.concat([h - l, (h - pc).abs(), (l - pc).abs()], axis=1).max(axis=1)
    return tr.ewm(alpha=1 / period, adjust=False).mean()

def calc_indicators(df, cfg):
    c = df["Close"]
    return {
        "jma":  calc_jma(c, cfg["jma_period"], cfg["jma_phase"], cfg["jma_power"]).values,
        "ema":  c.ewm(span=cfg["ema_period"], adjust=False).mean().values,
        "s50":  c.rolling(cfg["sma_fast"]).mean().values,
        "s150": c.rolling(cfg["sma_mid"]).mean().values,
        "s200": c.rolling(cfg["sma_slow"]).mean().values,
        "atr":  calc_atr(df, cfg["atr_period"]).values,
    }

def check_pattern_at(df, ind, i, cfg):
    """
    SINGLE SOURCE OF TRUTH. Checks signal day i. Computes EVERY step (no
    short-circuit) so the funnel shows where tickers drop out.
    Returns (passed, details); details is {} only if data is too short.
    """
    lb = cfg["slope_lookback"]
    if i < max(cfg["sma_slow"] + lb, cfg["swing_max_ago"] + 2) or i >= len(df):
        return False, {}
    o, h, l, c = (df[k].values for k in ("Open", "High", "Low", "Close"))
    j, e21, s50, s150, s200, atr = (ind[k] for k in ("jma", "ema", "s50", "s150", "s200", "atr"))
    need = (j[i], j[i - 1], j[i - 2], e21[i], s50[i], s150[i], s200[i], s50[i - lb], s200[i - lb], atr[i])
    if not all(np.isfinite(x) for x in need) or atr[i] <= 0:
        return False, {}

    # Step 1 — uptrend
    step1 = bool(s50[i] > s150[i] and s50[i] > s200[i] and s50[i] > s50[i - lb]
                 and s200[i] > s200[i - lb] and c[i] > s150[i])

    # Step 2 — swing high, then a pullback with the JMA rolling over
    hi = max(range(i - cfg["swing_max_ago"], i - cfg["swing_min_ago"] + 1), key=lambda k: h[k])
    pb = list(range(hi + 1, i))
    falls = sum(1 for k in pb if j[k] < j[k - 1])
    step2 = bool(cfg["pullback_min_days"] <= len(pb) <= cfg["pullback_max_days"]
                 and falls >= cfg["jma_min_down_days"] and j[i - 1] <= j[i - 2])

    # Step 3 — the pullback reached the EMA21 / SMA50 zone
    lo = min(pb, key=lambda k: l[k]) if pb else i - 1
    zone = max(e21[lo], s50[lo])
    step3 = bool(pb and l[lo] <= zone * (1 + cfg["zone_tolerance_pct"] / 100))

    # Step 4 — deep enough, and held near the SMA50 / above the SMA150
    depth_atr = (h[hi] - l[lo]) / atr[i]
    held = all(c[k] >= s50[k] * (1 - cfg["max_close_below_sma50_pct"] / 100) and l[k] >= s150[k] for k in pb)
    step4 = bool(pb and depth_atr >= cfg["min_depth_atr"] and held)

    # Step 5 — strong green candle back above the JMA and EMA21
    body_atr = (c[i] - o[i]) / atr[i]
    step5 = bool(c[i] > o[i] and body_atr >= cfg["min_body_atr"] and c[i] > j[i]
                 and c[i] > e21[i] and c[i - 1] <= j[i - 1])

    passed = step1 and step2 and step3 and step4 and step5
    return passed, {
        "date": df.index[i], "i": i, "swing_date": df.index[hi], "swing_high": float(h[hi]),
        "low_date": df.index[lo], "pullback_low": float(l[lo]), "pullback_days": len(pb),
        "jma_down_days": falls, "depth_atr": float(depth_atr), "body_atr": float(body_atr),
        "low_vs_sma50": float((l[lo] / s50[lo] - 1) * 100),
        "open": float(o[i]), "close": float(c[i]), "jma": float(j[i]), "ema21": float(e21[i]),
        "sma50": float(s50[i]), "sma150": float(s150[i]), "sma200": float(s200[i]),
        "sma50_slope": float((s50[i] / s50[i - lb] - 1) * 100),
        "step1": step1, "step2": step2, "step3": step3, "step4": step4, "step5": step5,
    }

# ── Diagnostic funnel (per ticker: did ANY of the lookback days pass
#    steps 1..k cumulatively) ──────────────────────────────────────────
FUNNEL_COUNTS = {
    "tickers_checked": 0,
    "passed_step1_uptrend": 0,
    "passed_step2_jma_pullback": 0,
    "passed_step3_reached_21_50_zone": 0,
    "passed_step4_held": 0,
    "passed_step5_reclaim_candle": 0,
}
_FUNNEL_KEYS = list(FUNNEL_COUNTS)[1:]

def find_signals(df, ind, cfg):
    hits, best = [], [False] * 5
    n = len(df)
    any_checked = False
    for i in range(max(0, n - cfg["signal_lookback_days"]), n):
        ok, d = check_pattern_at(df, ind, i, cfg)
        if not d: continue
        any_checked = True
        cum = True
        for k in range(5):
            cum = cum and d[f"step{k+1}"]
            best[k] |= cum
        if ok: hits.append(d)
    if any_checked:
        FUNNEL_COUNTS["tickers_checked"] += 1
        for k, key in enumerate(_FUNNEL_KEYS):
            FUNNEL_COUNTS[key] += best[k]
    return hits

def analyze_daily_pullback_jma_reclaim(sym, df_daily):
    if df_daily is None: return None
    price   = float(df_daily["Close"].iloc[-1])
    avg_vol = float(df_daily["Volume"].tail(20).mean())
    if price   < CFG["min_price"]:      return None
    if avg_vol < CFG["min_avg_volume"]: return None
    if len(df_daily) < CFG["min_bars"]: return None

    ind = calc_indicators(df_daily, CFG)
    hits = find_signals(df_daily, ind, CFG)
    if not hits: return None
    sig = hits[-1]   # most recent hit wins

    entry = sig["close"]
    stop = sig["pullback_low"]
    if entry <= stop: return None
    risk = entry - stop
    risk_pct = risk / entry * 100
    t1 = sig["swing_high"]
    t2 = entry + CFG["target2_r"] * risk
    r1 = (t1 - entry) / risk
    i = sig["i"]
    vol = df_daily["Volume"].values
    vol_x = float(vol[i] / np.mean(vol[i - 20:i])) if np.mean(vol[i - 20:i]) > 0 else 0.0
    days_ago = len(df_daily) - 1 - i
    after = df_daily.iloc[i + 1:]
    hit_t1 = bool(len(after) and after["High"].max() >= t1)
    hit_sl = bool(len(after) and after["Low"].min() <= stop)

    score = 20
    score += min(20, sig["body_atr"] * 10)                          # strong reclaim candle
    score += max(0, 15 - abs(sig["low_vs_sma50"]) * 3)               # low close to the SMA50
    score += max(0, 15 - risk_pct * 1.5)                             # tight stop
    score += min(10, max(0, r1) * 5)                                 # room to the swing high
    score += min(10, max(0, vol_x - 1) * 10)                         # volume on the reclaim
    score += min(10, max(0, sig["sma50_slope"]) * 1.5)               # strength of the trend
    score = round(min(100, max(0, score)))

    flags = [f"pullback {sig['pullback_days']}d to {sig['low_vs_sma50']:+.1f}% vs SMA50",
             f"reclaim body {sig['body_atr']:.1f} ATR", f"vol {vol_x:.1f}x"]
    if hit_t1: flags.append("Swing high already retested")
    if hit_sl: flags.append("Stop already hit")

    return {
        "Score": score, "Price": round(price, 2),
        "Entry_Price": round(entry, 2), "Stop_Loss": round(stop, 2), "Risk_%": round(risk_pct, 1),
        "Target_1_Swing_High": round(t1, 2), "T1_R": round(r1, 2), "T1_%": round((t1 - entry) / entry * 100, 1),
        "Target_2_2R": round(t2, 2),
        "Swing_High_Date": sig["swing_date"].strftime("%Y-%m-%d"),
        "Pullback_Low_Date": sig["low_date"].strftime("%Y-%m-%d"),
        "Pullback_%": round((sig["swing_high"] - stop) / sig["swing_high"] * 100, 1),
        "Pullback_Days": sig["pullback_days"], "JMA_Down_Days": sig["jma_down_days"],
        "Low_vs_SMA50_%": round(sig["low_vs_sma50"], 1), "Depth_ATR": round(sig["depth_atr"], 1),
        "Reclaim_Body_ATR": round(sig["body_atr"], 2), "Reclaim_Volume_x": round(vol_x, 2),
        "JMA": round(sig["jma"], 2), "EMA21": round(sig["ema21"], 2), "SMA50": round(sig["sma50"], 2),
        "SMA50_Slope_%": round(sig["sma50_slope"], 1),
        "Signal_Date": sig["date"].strftime("%Y-%m-%d"), "Days_Since_Signal": int(days_ago),
        "Recent_Signal_Count": len(hits),
        "Flags": " | ".join(flags),
        "_df": df_daily, "_ind": ind,
    }

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

for sym in tqdm(list(daily_map.keys()), desc="Checking pattern", unit="stk"):
    try:
        r = analyze_daily_pullback_jma_reclaim(sym, daily_map[sym])
        if r is None: continue
        r["Ticker"] = sym
        results.append(r)
        print(f"  ✅ {sym:<6} ${r['Price']:.2f}  Score {r['Score']}  pullback {r['Pullback_%']}% "
              f"({r['Low_vs_SMA50_%']:+}% vs SMA50)  body {r['Reclaim_Body_ATR']} ATR  {r['Signal_Date']}")
    except Exception: pass

fc = FUNNEL_COUNTS
print(f"\n{'━'*65}\n  SCAN COMPLETE — {len(results)} matches from {got_daily} tickers with data")
print(f"  🔍 FUNNEL (any of the last {CFG['signal_lookback_days']} sessions)")
print(f"  Tickers checked                          : {fc['tickers_checked']}")
print(f"  Step 1 — uptrend (SMA50 over 150/200)     : {fc['passed_step1_uptrend']}")
print(f"  Step 2 — + pullback with JMA rolling over : {fc['passed_step2_jma_pullback']}")
print(f"  Step 3 — + low reached EMA21/SMA50 zone   : {fc['passed_step3_reached_21_50_zone']}")
print(f"  Step 4 — + deep enough and held           : {fc['passed_step4_held']}")
print(f"  Step 5 — + green reclaim of JMA and EMA21 : {fc['passed_step5_reclaim_candle']}  (= full pattern)")
print(f"{'━'*65}")

results.sort(key=lambda x: x["Score"], reverse=True)

# ── Save CSV + TradingView list ───────────────────────────────
ts      = datetime.today().strftime("%Y%m%d_%H%M")
out_dir = os.environ.get("GITHUB_WORKSPACE", os.getcwd())
COLS = ["Ticker","Price","Score","Entry_Price","Stop_Loss","Risk_%",
        "Target_1_Swing_High","T1_R","T1_%","Target_2_2R",
        "Swing_High_Date","Pullback_Low_Date","Pullback_%","Pullback_Days","JMA_Down_Days",
        "Low_vs_SMA50_%","Depth_ATR","Reclaim_Body_ATR","Reclaim_Volume_x",
        "JMA","EMA21","SMA50","SMA50_Slope_%","Signal_Date","Days_Since_Signal",
        "Recent_Signal_Count","Flags"]
df_out = pd.DataFrame([{k: v for k, v in r.items() if not k.startswith("_")}
                       for r in results]) if results else pd.DataFrame(columns=COLS)
df_out = df_out[[c for c in COLS if c in df_out.columns]].reset_index(drop=True)

fpath = os.path.join(out_dir, f"daily_pullback_jma_reclaim_{ts}.csv")
try:
    from position_sizing import add_position_size
    df_out = add_position_size(df_out)
    try: __import__("best_picks").rank_best(df_out)
    except Exception as _bp_err: print(f"  ⚠️  Best 10 skipped: {_bp_err}")
except Exception as _ps_err:
    print(f"  ⚠️  Position sizing skipped: {_ps_err}")
df_out.to_csv(fpath, index=False)
print(f"\n  💾 CSV → {fpath}")
tv = os.path.join(out_dir, f"tv_daily_pullback_jma_reclaim_{ts}.txt")
with open(tv, "w") as f:
    f.write(f"###Daily Pullback JMA Reclaim {datetime.today().strftime('%Y-%m-%d')}\n")
    for r in results: f.write(f"NASDAQ:{r['Ticker']}\n")
print(f"  📋 TradingView → {tv}")
if results:
    print(f"\n  📋 Tickers: {', '.join(r['Ticker'] for r in results)}")

# ── Email with CSV attached ───────────────────────────────────
EMAIL_COLS = ["Ticker","Price","Score","Entry_Price","Stop_Loss","Risk_%",
              "Target_1_Swing_High","T1_R","Pullback_%","Low_vs_SMA50_%","Reclaim_Body_ATR",
              "Reclaim_Volume_x","Signal_Date"]

def _fmt_cell(col, v):
    try:
        if col in ("Price","Entry_Price","Stop_Loss","Target_1_Swing_High"): return f"${float(v):.2f}"
        if col == "T1_R": return f"{float(v):.1f}R"
        if col in ("Risk_%","Pullback_%"): return f"{float(v):.1f}%"
        if col == "Low_vs_SMA50_%": return f"{float(v):+.1f}%"
        if col == "Reclaim_Body_ATR": return f"{float(v):.1f} ATR"
        if col == "Reclaim_Volume_x": return f"{float(v):.1f}x"
        if col == "Score": return f"{float(v):.0f}"
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
                tds += f'<td style="padding:6px 10px;font-size:12px;{sty}">{_fmt_cell(c, r.get(c))}</td>'
            rows += f'<tr style="background:{bg}">{tds}</tr>'
        if not rows:
            rows = (f'<tr><td colspan="{len(EMAIL_COLS)}" style="padding:20px;text-align:center;'
                    f'color:#94a3b8">No matches today</td></tr>')
        tick_html = (f'<p style="margin:10px 0 0;font-size:12px;font-family:monospace">'
                     f'{", ".join(r["Ticker"] for r in rl)}</p>') if rl else ""
        html = f"""<!DOCTYPE html><html><body style="margin:0;background:#f1f5f9;font-family:'Segoe UI',Arial,sans-serif">
<table width="100%" cellpadding="0" cellspacing="0"><tr><td align="center" style="padding:20px 10px">
<table width="100%" style="max-width:860px;background:#fff;border-radius:12px;overflow:hidden">
<tr><td style="background:linear-gradient(135deg,#0f172a,#1e3a5f);padding:20px 26px">
<h1 style="margin:0;color:#60a5fa;font-size:19px">📊 Daily Pullback to the 50 → JMA Reclaim</h1>
<p style="margin:6px 0 0;color:#94a3b8;font-size:12px">{datetime.today().strftime('%Y-%m-%d %H:%M UTC')} · {cnt} match{'es' if cnt != 1 else ''}
 — uptrend, a {CFG['pullback_min_days']}-{CFG['pullback_max_days']} day pullback with the JMA rolling over into the EMA{CFG['ema_period']}/SMA{CFG['sma_fast']} zone, then a strong green candle back above the JMA (the RVTY 14 Sep pattern)</p>
</td></tr>
<tr><td style="padding:12px 26px;background:#0b1220"><p style="margin:0;color:#cbd5e1;font-size:12px">
🔍 FUNNEL: {fc['tickers_checked']} checked → {fc['passed_step1_uptrend']} uptrend
→ {fc['passed_step2_jma_pullback']} + JMA pullback → {fc['passed_step3_reached_21_50_zone']} + reached 21/50 zone
→ {fc['passed_step4_held']} + held
→ <b style="color:#facc15">{fc['passed_step5_reclaim_candle']} + reclaim candle = full match</b></p></td></tr>
<tr><td style="padding:16px"><div style="overflow-x:auto;border:1px solid #e2e8f0;border-radius:8px">
<table style="border-collapse:collapse;width:100%"><thead><tr>{th}</tr></thead><tbody>{rows}</tbody></table></div>
{tick_html}
<p style="font-size:11px;color:#64748b;margin:10px 0 0">Entry = the signal day's close · Stop = the pullback low ·
Target 1 = the swing high before the pullback (R = reward ÷ risk), Target 2 = entry + {CFG['target2_r']:.0f}R (in the CSV). Flags say if the swing high or the stop was already touched since. 📎 Full results (with Quantity for a fixed max loss) attached as CSV.</p>
</td></tr>
<tr><td style="background:#f8fafc;padding:10px;text-align:center;color:#94a3b8;font-size:10px">
⚠️ Not financial advice · Auto-generated by GitHub Actions</td></tr>
</table></td></tr></table></body></html>"""

        plain = [f"Daily Pullback to the 50 → JMA Reclaim — {datetime.today().strftime('%Y-%m-%d')}",
                 f"{cnt} matches",
                 f"FUNNEL: {fc['tickers_checked']} checked -> {fc['passed_step1_uptrend']} uptrend -> "
                 f"{fc['passed_step2_jma_pullback']} JMA pullback -> {fc['passed_step3_reached_21_50_zone']} reached 21/50 zone -> "
                 f"{fc['passed_step4_held']} held -> {fc['passed_step5_reclaim_candle']} reclaim candle (=full match)", "="*60]
        for r in rl[:50]:
            plain.append(f"{r['Ticker']:<6} ${r['Price']:.2f}  Score:{r['Score']}  Entry:${r['Entry_Price']:.2f}  "
                         f"SL:${r['Stop_Loss']:.2f}  T1(swing high):${r['Target_1_Swing_High']:.2f} {r['T1_R']}R  "
                         f"pullback {r['Pullback_%']}% ({r['Low_vs_SMA50_%']:+}% vs SMA50)  {r['Signal_Date']}")
        if rl:
            plain += ["", "Tickers: " + ", ".join(r["Ticker"] for r in rl)]
        else:
            plain.append("No matches today")
        plain.append("\nFull results in CSV attachment.")

        subj = (f"📊 Daily Pullback JMA Reclaim — "
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

# ── Charts for top 5: last 60 daily candles + JMA, EMA21, SMA50/150/200 ─
if results:
    top = results[:5]
    fig, axes = plt.subplots(len(top), 1, figsize=(14, 4.2*len(top)), facecolor="#0f172a")
    if len(top) == 1: axes = [axes]
    for ax, r in zip(axes, top):
        df = r["_df"]; ind = r["_ind"]; k = df.index.get_loc(pd.Timestamp(r["Signal_Date"]))
        lo = max(0, k - 54); seg = df.iloc[lo:]
        x = np.arange(len(seg))
        ax.set_facecolor("#0f172a")
        cols = ["#22c55e" if c > o else "#ef4444" for o, c in zip(seg["Open"], seg["Close"])]
        ax.vlines(x, seg["Low"], seg["High"], color=cols, lw=1)
        ax.bar(x, (seg["Close"] - seg["Open"]).abs(), bottom=np.minimum(seg["Open"], seg["Close"]),
               color=cols, width=0.6)
        for key, c_, lab, lw in (("jma", "#f97316", "JMA", 1.4), ("ema", "#facc15", "EMA21", 1.1),
                                 ("s50", "#86efac", "SMA50", 1.1), ("s150", "#60a5fa", "SMA150", 1.0),
                                 ("s200", "#f87171", "SMA200", 1.0)):
            ax.plot(x, ind[key][lo:], color=c_, lw=lw, label=lab)
        for lvl, c_, lab in ((r["Entry_Price"], "#e2e8f0", "Entry"), (r["Stop_Loss"], "#ef4444", "Stop"),
                             (r["Target_1_Swing_High"], "#a78bfa", "T1")):
            ax.axhline(lvl, color=c_, ls=":", lw=0.8)
            ax.text(len(seg) - 0.5, lvl, f" {lab} {lvl:.2f}", color=c_, fontsize=7, va="center")
        ax.axvline(k - lo, color="#facc15", ls="--", lw=0.8)
        ax.legend(loc="upper left", facecolor="#1e293b", labelcolor="#e2e8f0", fontsize=7, ncol=5)
        step = max(1, len(seg) // 8)
        ax.set_xticks(x[::step]); ax.set_xticklabels([d.strftime("%m/%d") for d in seg.index[::step]], fontsize=7)
        ax.set_title(f"{r['Ticker']} | USD {r['Price']:.2f} | Score {r['Score']} | pullback {r['Pullback_%']}% "
                     f"({r['Low_vs_SMA50_%']:+}% vs SMA50) | T1 {r['T1_R']}R", color="#e2e8f0", fontsize=9, fontweight="bold")
        ax.tick_params(colors="#94a3b8", labelsize=7)
        for sp in ax.spines.values(): sp.set_edgecolor("#1e3a5f")
        ax.grid(color="#1e3a5f", ls="--", lw=0.5, alpha=0.6)
    plt.suptitle(f"Daily Pullback to the 50 → JMA Reclaim · "
                 f"{datetime.today().strftime('%Y-%m-%d')}", color="#60a5fa", fontsize=12, fontweight="bold")
    plt.tight_layout()
    cp = os.path.join(out_dir, f"daily_pullback_jma_reclaim_chart_{ts}.png")
    plt.savefig(cp, dpi=130, bbox_inches="tight", facecolor="#0f172a")
    plt.close()
    print(f"  📊 Chart → {cp}")

print(f"""
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  📋 SIGNAL (all required; signal day = one of the last {CFG['signal_lookback_days']} sessions)
  1) Uptrend: SMA{CFG['sma_fast']} above SMA{CFG['sma_mid']} and SMA{CFG['sma_slow']}, SMA{CFG['sma_fast']} and SMA{CFG['sma_slow']} rising
     ({CFG['slope_lookback']} sessions), close above SMA{CFG['sma_mid']}
  2) Swing high {CFG['swing_min_ago']}-{CFG['swing_max_ago']} sessions back, then a {CFG['pullback_min_days']}-{CFG['pullback_max_days']} day pullback with
     JMA({CFG['jma_period']},{CFG['jma_phase']},{CFG['jma_power']}) falling on >= {CFG['jma_min_down_days']} days and not rising the day before
  3) Pullback low within {CFG['zone_tolerance_pct']}% of the higher of EMA{CFG['ema_period']} / SMA{CFG['sma_fast']} (or below)
  4) Pullback >= {CFG['min_depth_atr']} ATR{CFG['atr_period']} deep; no close > {CFG['max_close_below_sma50_pct']:.0f}% under SMA{CFG['sma_fast']}, no low under SMA{CFG['sma_mid']}
  5) Signal day: green, body >= {CFG['min_body_atr']} ATR, closes back above the JMA and EMA{CFG['ema_period']}
  OUTPUT: Entry = signal day close, Stop = pullback low,
          Target 1 = swing high, Target 2 = entry + {CFG['target2_r']:.0f}R
  ⚙️  If 0 results, check the FUNNEL: relax min_body_atr 0.6 → 0.4,
     zone_tolerance_pct 0.5 → 1.5, or signal_lookback_days 2 → 5
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
""")
