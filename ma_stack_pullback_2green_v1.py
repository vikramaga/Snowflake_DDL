# ============================================================
# NASDAQ / NYSE - MA Stack Pullback + 2 Green Candles on Rising Volume
#                 (Daily AND Weekly)  v1
# ============================================================
#
# From Vik's chart (2026-11-14). Line colours on that chart:
#   JMA(13,40,2) = the green/red line on top, EMA8 = orange, SMA21 = green
#   (the support), SMA50 = blue, SMA150 = red.
#
# Checked separately on DAILY bars and on WEEKLY bars (resampled from the
# daily download; the in-progress week counts, its volume scaled to 5 days):
#   1. TREND STACK today: JMA and EMA8 above SMA21 > SMA50 > SMA150, with
#      SMA50 and SMA150 both higher than 10 bars ago
#   2. RUN TO A HIGH: the pullback high is a fresh high (the highest high of
#      the 40 daily / 26 weekly bars up to it), 3-15 bars before today
#   3. PULLBACK HOLDING SMA21: from that high, at least one bar pulled back
#      >= 1 ATR(14) and reached EMA8, with no close below SMA21 (1% tolerance)
#      and JMA turning down on at least one bar (the red section of the line)
#   4. TWO GREEN CANDLES: yesterday and today both green, today's volume above
#      yesterday's (rising volume)
#   5. RECLAIM: today's close above EMA8 and above JMA
#
# Entry = today's close; Stop = lowest low of the pullback and the two green
# candles; T1 = the pullback high (Reward_R); T2 = 2R. If price already closed
# above the pullback high it is flagged "breakout" and T1 = 2R.
# Thresholds are reasonable defaults, not explicitly requested; all are in CFG.
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


# -- CONFIG
CFG = {
    # 1300 calendar days -> ~185 weekly bars (weekly SMA150 + 10-week slope need 160)
    "history_days"            : 1300,
    "min_bars_D"              : 200,
    "min_bars_W"              : 165,

    "atr_period"              : 14,
    "slope_bars"              : 10,    # SMA50 / SMA150 higher than 10 bars ago
    "high_window_D"           : 40,    # pullback high = highest high of the last 40 daily bars
    "high_window_W"           : 26,    # ... / 26 weekly bars
    "high_min_age"            : 3,     # pullback high 3..15 bars before today
    "high_max_age"            : 15,
    "min_pullback_atr"        : 1.0,   # pullback depth from the high >= 1 ATR
    "ema8_touch_tol"          : 0.005, # pullback low <= EMA8 + 0.5%
    "sma21_close_tol"         : 0.01,  # no pullback close below SMA21 - 1%
    "signal_lookback_days"    : 1,     # the latest bar only

    "min_avg_volume"          : 200_000,
    "min_price"               : 5.0,

    "batch_size"              : 50,
    "batch_sleep"             : 1.5,
}

TFS = ("D", "W")
TF_NAME = {"D": "Daily", "W": "Weekly"}

# -- Indicators
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


def resample_weekly(df):
    """Weekly bars (weeks end Friday). The last, in-progress week's volume is scaled to 5 days."""
    agg = {"Open": "first", "High": "max", "Low": "min", "Close": "last", "Volume": "sum"}
    w = df.resample("W-FRI").agg(agg).dropna(subset=["Close"])
    days = df["Close"].resample("W-FRI").count().reindex(w.index)
    w["Volume_Adj"] = w["Volume"].astype(float)
    nd = int(days.iloc[-1]) if len(days) else 5
    if 0 < nd < 5:
        w.iloc[-1, w.columns.get_loc("Volume_Adj")] = float(w["Volume"].iloc[-1]) * 5.0 / nd
    w["Days_In_Bar"] = days.values
    return w

def calc_indicators(df, cfg):
    h, l, c = (df[k].values.astype(float) for k in ("High", "Low", "Close"))
    pc = np.concatenate([[np.nan], c[:-1]])
    tr = np.nanmax(np.vstack([h - l, np.abs(h - pc), np.abs(l - pc)]), axis=0)
    cs = df["Close"].astype(float)
    return {
        "atr":    pd.Series(tr).rolling(cfg["atr_period"]).mean().values,
        "jma":    calc_jma(cs, 13, 40, 2).values,
        "ema8":   cs.ewm(span=8, adjust=False).mean().values,
        "sma21":  cs.rolling(21).mean().values,
        "sma50":  cs.rolling(50).mean().values,
        "sma150": cs.rolling(150).mean().values,
    }

def check_pattern_at(df, ind, i, cfg, tf):
    """
    SINGLE SOURCE OF TRUTH for one bar i on one timeframe. Computes EVERY step
    (no short-circuit) so the funnel shows where tickers drop out. Of the
    candidate pullback highs (3-15 bars back), reports the one that got furthest.
    Returns (passed, details); {} if data short.
    """
    sb = cfg["slope_bars"]
    if i < 150 + sb + 2 or i >= len(df):
        return False, {}
    o, h, l, c = (df[k].values.astype(float) for k in ("Open", "High", "Low", "Close"))
    v = (df["Volume_Adj"] if "Volume_Adj" in df else df["Volume"]).values.astype(float)
    jma, e8, s21, s50, s150, atr = (ind[k] for k in ("jma", "ema8", "sma21", "sma50", "sma150", "atr"))
    a = atr[i]
    if not np.isfinite(a) or a <= 0 or not np.isfinite(s150[i - sb]):
        return False, {}
    step1 = bool(min(jma[i], e8[i]) > s21[i] > s50[i] > s150[i]
                 and s50[i] > s50[i - sb] and s150[i] > s150[i - sb])
    step4 = bool(c[i - 1] > o[i - 1] and c[i] > o[i] and v[i] > v[i - 1])
    step5 = bool(c[i] > e8[i] and c[i] > jma[i])
    win = cfg[f"high_window_{tf}"]
    best, best_key = None, None
    for hb in range(i - cfg["high_max_age"], i - cfg["high_min_age"] + 1):
        if hb - win + 1 < 0:
            continue
        step2 = bool(h[hb] >= h[hb - win + 1:hb + 1].max() and h[hb] >= h[hb:i - 1].max())
        pb = range(hb + 1, i - 1)                       # pullback bars: after the high, before the 2 greens
        if len(pb) == 0:
            continue
        lows = l[hb + 1:i - 1]
        k_low = hb + 1 + int(np.argmin(lows))
        depth = (h[hb] - l[k_low]) / a
        touch = any(l[k] <= e8[k] * (1 + cfg["ema8_touch_tol"]) for k in pb)
        held = all(c[k] >= s21[k] * (1 - cfg["sma21_close_tol"]) for k in pb)
        jma_down = any(jma[k] < jma[k - 1] for k in pb)
        step3 = bool(depth >= cfg["min_pullback_atr"] and touch and held and jma_down)
        d = {"hb": hb, "k_low": k_low, "depth_atr": float(depth), "touch": touch, "held": held,
             "jma_down": jma_down, "step1": step1, "step2": step2, "step3": step3,
             "step4": step4, "step5": step5}
        dep = 0
        for k in range(1, 6):
            if not d[f"step{k}"]: break
            dep = k
        key = (dep, h[hb])
        if best_key is None or key > best_key:
            best, best_key = d, key
    if best is None:
        best = {"step1": step1, "step2": False, "step3": False, "step4": step4, "step5": step5}
    passed = all(best[f"step{k}"] for k in range(1, 6))
    best.update({"date": df.index[i], "i": i, "atr": float(a), "close": float(c[i])})
    return passed, best

# -- Diagnostic funnel (per ticker and timeframe)
_STEP_NAMES = ["trend_stack", "fresh_high", "pullback_held_sma21", "two_green_rising_vol", "close_above_ema8_jma"]
FUNNEL_COUNTS = {tf: {"tickers_checked": 0, **{f"passed_step{k+1}_{n}": 0 for k, n in enumerate(_STEP_NAMES)}}
                 for tf in TFS}

def find_signals(df, ind, cfg, tf):
    hits, best = [], [False] * 5
    n = len(df)
    any_checked = False
    for i in range(max(0, n - cfg["signal_lookback_days"]), n):
        ok, d = check_pattern_at(df, ind, i, cfg, tf)
        if not d: continue
        any_checked = True
        cum = True
        for k in range(5):
            cum = cum and d[f"step{k+1}"]
            best[k] |= cum
        if ok: hits.append(d)
    if any_checked:
        fc = FUNNEL_COUNTS[tf]
        fc["tickers_checked"] += 1
        for k, key in enumerate(list(fc)[1:]):
            fc[key] += best[k]
    return hits

def analyze_tf(sym, df, tf):
    if df is None or len(df) < CFG[f"min_bars_{tf}"]: return None
    ind = calc_indicators(df, CFG)
    hits = find_signals(df, ind, CFG, tf)
    if not hits: return None
    sig = hits[-1]
    i, a = sig["i"], sig["atr"]
    h, l = df["High"].values.astype(float), df["Low"].values.astype(float)
    vcol = "Volume_Adj" if "Volume_Adj" in df else "Volume"
    v = df[vcol].values.astype(float)
    entry = sig["close"]
    stop = float(l[sig["hb"] + 1:i + 1].min())
    if stop <= 0 or entry <= stop: return None
    risk = entry - stop
    risk_pct = risk / entry * 100
    high = float(h[sig["hb"]])
    breakout = entry >= high
    t1 = entry + 2 * risk if breakout else high
    reward_r = (t1 - entry) / risk
    t2 = entry + 2 * risk
    vol_ratio = v[i] / v[i - 1] if v[i - 1] > 0 else np.nan
    v20 = np.mean(v[i - 20:i]) if i >= 20 else np.nan
    vol_vs_avg = v[i] / v20 if v20 and v20 > 0 else np.nan
    s21, e8 = ind["sma21"][i], ind["ema8"][i]
    k_low = sig["k_low"]
    low_vs_21 = (l[k_low] / ind["sma21"][k_low] - 1) * 100
    flags = [f"pullback {sig['depth_atr']:.1f} ATR over {i - 1 - sig['hb'] - 1} bar(s)",
             f"pullback low {low_vs_21:+.1f}% vs SMA21", f"volume {vol_ratio:.1f}x prior green"]
    if breakout: flags.append("closed above the pullback high (breakout) - T1 = 2R")
    if tf == "W" and int(df["Days_In_Bar"].iloc[-1]) < 5: flags.append(f"week in progress ({int(df['Days_In_Bar'].iloc[-1])} days, volume scaled)")

    score = 40
    score += min(15, max(0, (vol_ratio - 1) * 15)) if np.isfinite(vol_ratio) else 0
    score += min(10, max(0, (vol_vs_avg - 1) * 10)) if np.isfinite(vol_vs_avg) else 0
    score += max(0, 10 - abs(low_vs_21) * 2)                            # pullback low close to SMA21
    score += max(0, 10 - max(0, risk_pct - 4) * 2)                      # tighter stop
    score += min(10, max(0, reward_r) * 4)
    score += 5 if not breakout else 0
    score = round(min(100, max(0, score)))
    di = df.index
    return {
        "Score": score, "Timeframe": TF_NAME[tf], "Price": round(entry, 2),
        "Entry_Price": round(entry, 2), "Stop_Loss": round(stop, 2), "Risk_%": round(risk_pct, 1),
        "Target_1": round(t1, 2), "Reward_R": round(reward_r, 1), "Target_2": round(t2, 2),
        "Pullback_High": round(high, 2), "High_Date": di[sig["hb"]].strftime("%Y-%m-%d"),
        "Pullback_Low": round(float(l[k_low]), 2), "Low_Date": di[k_low].strftime("%Y-%m-%d"),
        "Vol_Ratio": round(vol_ratio, 2), "Vol_vs_20avg": round(vol_vs_avg, 2) if np.isfinite(vol_vs_avg) else np.nan,
        "JMA": round(float(ind["jma"][i]), 2), "EMA8": round(float(e8), 2), "SMA21": round(float(s21), 2),
        "SMA50": round(float(ind["sma50"][i]), 2), "SMA150": round(float(ind["sma150"][i]), 2),
        "Breakout": "Yes" if breakout else "No",
        "Signal_Date": sig["date"].strftime("%Y-%m-%d"),
        "Flags": " | ".join(flags),
        "_df": df, "_ind": ind, "_sig": sig,
    }

def analyze_stock(sym, df_daily):
    if df_daily is None: return []
    price   = float(df_daily["Close"].iloc[-1])
    avg_vol = float(df_daily["Volume"].tail(20).mean())
    if price < CFG["min_price"] or avg_vol < CFG["min_avg_volume"]: return []
    out = []
    for tf, df in (("D", df_daily), ("W", resample_weekly(df_daily))):
        try:
            r = analyze_tf(sym, df, tf)
            if r: out.append(r)
        except Exception:
            pass
    return out

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

# -- Main scan
print("=" * 65)
print(f"  STEP 3  SCANNING {len(TICKERS)} TICKERS")
print("=" * 65)

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
        for r in analyze_stock(sym, daily_map[sym]):
            r["Ticker"] = sym
            results.append(r)
            print(f"  OK {sym:<6} {r['Timeframe']:<6} ${r['Price']:.2f}  Score {r['Score']}  high {r['High_Date']}  "
                  f"vol {r['Vol_Ratio']}x  T1 {r['Target_1']} ({r['Reward_R']}R)")
    except Exception: pass

print(f"\n{'=' * 65}\n  SCAN COMPLETE - {len(results)} matches from {got_daily} tickers with data")
for tf in TFS:
    fc = FUNNEL_COUNTS[tf]; ks = list(fc)
    print(f"  {TF_NAME[tf].upper()} FUNNEL (latest bar)")
    print(f"    Tickers checked                               : {fc[ks[0]]}")
    print(f"    Step 1 - JMA/EMA8 > SMA21 > SMA50 > SMA150    : {fc[ks[1]]}")
    print(f"    Step 2 - + fresh high 3-15 bars ago           : {fc[ks[2]]}")
    print(f"    Step 3 - + pullback to EMA8 holding SMA21     : {fc[ks[3]]}")
    print(f"    Step 4 - + two green candles, rising volume   : {fc[ks[4]]}")
    print(f"    Step 5 - + close above EMA8 and JMA           : {fc[ks[5]]}  (= full pattern)")
print("=" * 65)

results.sort(key=lambda x: (x["Timeframe"] != "Weekly", -x["Score"]))

# -- Save CSV + TradingView list
ts      = datetime.today().strftime("%Y%m%d_%H%M")
out_dir = os.environ.get("GITHUB_WORKSPACE", os.getcwd())
COLS = ["Ticker","Timeframe","Price","Score","Entry_Price","Stop_Loss","Risk_%","Target_1","Reward_R","Target_2",
        "Pullback_High","High_Date","Pullback_Low","Low_Date","Vol_Ratio","Vol_vs_20avg",
        "JMA","EMA8","SMA21","SMA50","SMA150","Breakout","Signal_Date","Flags"]
df_out = pd.DataFrame([{k: v for k, v in r.items() if not k.startswith("_")}
                       for r in results]) if results else pd.DataFrame(columns=COLS)
df_out = df_out[[c for c in COLS if c in df_out.columns]].reset_index(drop=True)

fpath = os.path.join(out_dir, f"ma_stack_pullback_2green_{ts}.csv")
try:
    from position_sizing import add_position_size
    df_out = add_position_size(df_out)
    try: __import__("best_picks").rank_best(df_out)
    except Exception as _bp_err: print(f"  Best 10 skipped: {_bp_err}")
except Exception as _ps_err:
    print(f"  Position sizing skipped: {_ps_err}")
df_out.to_csv(fpath, index=False)
print(f"\n  CSV -> {fpath}")
tv = os.path.join(out_dir, f"tv_ma_stack_pullback_2green_{ts}.txt")
with open(tv, "w") as f:
    for tf in TFS:
        names = [r["Ticker"] for r in results if r["Timeframe"] == TF_NAME[tf]]
        if names:
            f.write(f"###MA Stack 2 Green {TF_NAME[tf]} {datetime.today().strftime('%Y-%m-%d')}\n")
            for s in dict.fromkeys(names): f.write(f"NASDAQ:{s}\n")
print(f"  TradingView -> {tv}")
for tf in TFS:
    names = [r["Ticker"] for r in results if r["Timeframe"] == TF_NAME[tf]]
    if names: print(f"\n  {TF_NAME[tf]}: {', '.join(names)}")

# -- Email with CSV attached
EMAIL_COLS = ["Ticker","Timeframe","Price","Score","Entry_Price","Stop_Loss","Risk_%","Target_1","Reward_R","Target_2",
              "Vol_Ratio","Pullback_High","Pullback_Low","SMA21","Breakout"]

def _fmt_cell(col, v):
    try:
        if col in ("Price","Entry_Price","Stop_Loss","Target_1","Target_2","Pullback_High","Pullback_Low","SMA21"): return f"${float(v):.2f}"
        if col == "Risk_%": return f"{float(v):.1f}%"
        if col == "Reward_R": return f"{float(v):.1f}R"
        if col == "Vol_Ratio": return f"{float(v):.2f}x"
        if col == "Score": return f"{float(v):.0f}"
    except Exception: pass
    return "-" if v is None else str(v)

def _send_email(rl, csv_path):
    import smtplib
    from email.mime.multipart import MIMEMultipart
    from email.mime.text      import MIMEText
    from email.mime.base      import MIMEBase
    from email                import encoders

    gu = _GMAIL_USER; gp = _GMAIL_PASS; et = _EMAIL_TO
    if not (gu and gp and et):
        print("[Email] GMAIL_USER / GMAIL_PASS / EMAIL_TO missing - skipped")
        return
    eto = [e.strip() for e in et.split(",") if e.strip()]
    cnt = len(rl)
    nD = sum(r["Timeframe"] == "Daily" for r in rl); nW = cnt - nD
    try:
        th = "".join(f'<th style="background:#1e293b;color:#e2e8f0;padding:8px 10px;'
                     f'font-size:11px;border-bottom:2px solid #3b82f6;white-space:nowrap">{c}</th>'
                     for c in EMAIL_COLS)
        rows = ""
        for i, r in enumerate(rl[:60]):
            bg = "#fff" if i % 2 == 0 else "#f0f9ff"
            tds = ""
            for c in EMAIL_COLS:
                sty = "font-weight:700" if c == "Ticker" else ""
                if c == "Score": sty = "font-weight:700;background:#166534;color:#fff;text-align:center"
                if c == "Timeframe" and r["Timeframe"] == "Weekly": sty = "font-weight:700;color:#7c3aed"
                if c == "Stop_Loss": sty = "color:#ef4444"
                if c == "Entry_Price": sty = "color:#16a34a"
                if c == "Target_1": sty = "color:#2563eb;font-weight:600"
                tds += f'<td style="padding:6px 10px;font-size:12px;{sty}">{_fmt_cell(c, r.get(c))}</td>'
            rows += f'<tr style="background:{bg}">{tds}</tr>'
        if not rows:
            rows = (f'<tr><td colspan="{len(EMAIL_COLS)}" style="padding:20px;text-align:center;'
                    f'color:#94a3b8">No matches today</td></tr>')
        fun = ""
        for tf in TFS:
            fc = FUNNEL_COUNTS[tf]; ks = list(fc)
            fun += (f"<b>{TF_NAME[tf]}:</b> {fc[ks[0]]} checked &rarr; {fc[ks[1]]} MA stack &rarr; {fc[ks[2]]} + fresh high "
                    f"&rarr; {fc[ks[3]]} + pullback held SMA21 &rarr; {fc[ks[4]]} + 2 green / rising volume "
                    f"&rarr; <b style=\"color:#facc15\">{fc[ks[5]]} + close above EMA8 &amp; JMA = full match</b><br>")
        html = f"""<!DOCTYPE html><html><body style="margin:0;background:#f1f5f9;font-family:'Segoe UI',Arial,sans-serif">
<table width="100%" cellpadding="0" cellspacing="0"><tr><td align="center" style="padding:20px 10px">
<table width="100%" style="max-width:900px;background:#fff;border-radius:12px;overflow:hidden">
<tr><td style="background:linear-gradient(135deg,#0f172a,#1e3a5f);padding:20px 26px">
<h1 style="margin:0;color:#60a5fa;font-size:19px">MA Stack Pullback + 2 Green Candles on Rising Volume</h1>
<p style="margin:6px 0 0;color:#94a3b8;font-size:12px">{datetime.today().strftime('%Y-%m-%d %H:%M UTC')} &middot; {nW} weekly, {nD} daily
 - JMA/EMA8 &gt; SMA21 &gt; SMA50 &gt; SMA150, a pullback from a fresh high held SMA21 while JMA turned down,
 then two green candles with rising volume closing back above EMA8 and JMA</p>
</td></tr>
<tr><td style="padding:12px 26px;background:#0b1220"><p style="margin:0;color:#cbd5e1;font-size:12px;line-height:1.6">{fun}</p></td></tr>
<tr><td style="padding:16px"><div style="overflow-x:auto;border:1px solid #e2e8f0;border-radius:8px">
<table style="border-collapse:collapse;width:100%"><thead><tr>{th}</tr></thead><tbody>{rows}</tbody></table></div>
<p style="font-size:11px;color:#64748b;margin:10px 0 0">Entry = latest close &middot; Stop = lowest low of the pullback and the two green candles &middot;
Target_1 = the pullback high (2R if price already closed above it - Breakout = Yes) &middot; Target_2 = 2R &middot;
Vol_Ratio = today's volume / yesterday's (weekly: this week / last week, an in-progress week scaled to 5 days).
Pullback: at least {CFG['min_pullback_atr']:g} ATR from a high that topped the last {CFG['high_window_D']} daily / {CFG['high_window_W']} weekly bars,
reaching EMA8, no close more than {CFG['sma21_close_tol']*100:g}% below SMA21, JMA falling on at least one bar.
Full results (with Quantity for a fixed max loss) attached as CSV.</p>
</td></tr>
<tr><td style="background:#f8fafc;padding:10px;text-align:center;color:#94a3b8;font-size:10px">
Not financial advice &middot; Auto-generated by GitHub Actions</td></tr>
</table></td></tr></table></body></html>"""

        plain = [f"MA Stack Pullback + 2 Green (Daily + Weekly) - {datetime.today().strftime('%Y-%m-%d')}",
                 f"{nW} weekly, {nD} daily matches"]
        for tf in TFS:
            fc = FUNNEL_COUNTS[tf]; ks = list(fc)
            plain.append(f"{TF_NAME[tf]} FUNNEL: " + " -> ".join(str(fc[k]) for k in ks) + " (=full match)")
        plain.append("=" * 60)
        for r in rl[:60]:
            plain.append(f"{r['Ticker']:<6} {r['Timeframe']:<6} ${r['Price']:.2f}  Score:{r['Score']}  "
                         f"SL:${r['Stop_Loss']:.2f} ({r['Risk_%']}%)  T1:${r['Target_1']:.2f} ({r['Reward_R']}R)  "
                         f"vol {r['Vol_Ratio']}x  breakout {r['Breakout']}")
        if not rl: plain.append("No matches today")
        plain.append("\nFull results in CSV attachment.")

        subj = (f"MA Stack 2 Green - {nW} weekly, {nD} daily - {datetime.today().strftime('%Y-%m-%d')}")
        msg = MIMEMultipart("mixed")
        msg["Subject"] = subj; msg["From"] = gu; msg["To"] = ", ".join(eto)
        alt = MIMEMultipart("alternative")
        alt.attach(MIMEText("\n".join(plain), "plain"))
        alt.attach(MIMEText(html, "html"))
        msg.attach(alt)
    except Exception as e:
        print(f"[Email] Failed to build email body: {type(e).__name__}: {e}")
        return

    if csv_path and os.path.exists(csv_path):
        try:
            with open(csv_path, "rb") as f:
                part = MIMEBase("application", "octet-stream"); part.set_payload(f.read())
            encoders.encode_base64(part)
            part.add_header("Content-Disposition", f"attachment; filename={os.path.basename(csv_path)}")
            msg.attach(part)
        except Exception as e:
            print(f"[Email] CSV attach failed: {e}")
    try:
        with smtplib.SMTP_SSL("smtp.gmail.com", 465) as srv:
            srv.login(gu, gp.replace(" ", ""))
            try: __import__("best_picks").attach_to_email(msg)
            except Exception as _bp_err: print(f"  Best 10 not added: {_bp_err}")
            srv.sendmail(gu, eto, msg.as_string())
        print(f"[Email] Sent to {', '.join(eto)} - {subj}")
    except Exception as e:
        print(f"[Email] {type(e).__name__}: {e}")

try:
    _send_email(results, fpath)
except Exception as e:
    print(f"[Email] Unexpected top-level error: {type(e).__name__}: {e}")

# -- Charts for the top 5 by score: last 90 bars of the signal timeframe with the five lines
if results:
    top = sorted(results, key=lambda x: -x["Score"])[:5]
    fig, axes = plt.subplots(len(top), 1, figsize=(13, 4.2*len(top)), facecolor="#0f172a")
    if len(top) == 1: axes = [axes]
    for ax, r in zip(axes, top):
        full, ind = r["_df"], r["_ind"]
        d = full.iloc[-90:]; n_ = len(d); off = len(full) - n_
        x = np.arange(n_)
        ax.set_facecolor("#0f172a")
        cols = ["#22c55e" if c > o else "#ef4444" for o, c in zip(d["Open"], d["Close"])]
        ax.vlines(x, d["Low"], d["High"], color=cols, lw=0.8)
        ax.bar(x, (d["Close"] - d["Open"]).abs().clip(lower=1e-9), bottom=np.minimum(d["Open"], d["Close"]),
               color=cols, width=0.6)
        for k, c_, lab in (("jma", "#e879f9", "JMA"), ("ema8", "#f97316", "EMA8"), ("sma21", "#16a34a", "SMA21"),
                           ("sma50", "#3b82f6", "SMA50"), ("sma150", "#ef4444", "SMA150")):
            ax.plot(x, ind[k][off:], color=c_, lw=1.1, label=lab)
        for lvl, c_, lab in ((r["Stop_Loss"], "#ef4444", "Stop"), (r["Target_1"], "#22c55e", "T1")):
            ax.axhline(lvl, color=c_, ls=":", lw=0.9)
            ax.text(n_ - 0.4, lvl, f" {lab} {lvl:.2f}", color=c_, fontsize=7, va="center")
        ax.legend(loc="upper left", fontsize=7, facecolor="#0f172a", labelcolor="#cbd5e1")
        ax.set_xticks(x[::10]); ax.set_xticklabels([t.strftime("%y/%m/%d") for t in d.index[::10]], fontsize=7)
        ax.set_title(f"{r['Ticker']} | {r['Timeframe']} | USD {r['Price']:.2f} | Score {r['Score']} | vol {r['Vol_Ratio']}x | "
                     f"T1 {r['Target_1']:.2f} ({r['Reward_R']}R) | {r['Signal_Date']}",
                     color="#e2e8f0", fontsize=9, fontweight="bold")
        ax.tick_params(colors="#94a3b8", labelsize=7)
        for sp in ax.spines.values(): sp.set_edgecolor("#1e3a5f")
        ax.grid(color="#1e3a5f", ls="--", lw=0.5, alpha=0.6)
    plt.suptitle(f"MA Stack Pullback + 2 Green on Rising Volume - {datetime.today().strftime('%Y-%m-%d')}",
                 color="#60a5fa", fontsize=12, fontweight="bold")
    plt.tight_layout(rect=[0, 0, 1, 0.985])
    cp = os.path.join(out_dir, f"ma_stack_pullback_2green_chart_{ts}.png")
    plt.savefig(cp, dpi=130, bbox_inches="tight", facecolor="#0f172a")
    plt.close()
    print(f"  Chart -> {cp}")

print(f"""
{'=' * 55}
  PATTERN (latest daily bar and latest weekly bar)
  1) JMA and EMA8 > SMA21 > SMA50 > SMA150, SMA50 and SMA150 rising
  2) A fresh high ({CFG['high_window_D']} daily / {CFG['high_window_W']} weekly bars) 3-15 bars ago
  3) Pullback >= {CFG['min_pullback_atr']:g} ATR to EMA8, closes held SMA21, JMA turned down
  4) Last two candles green, the second on higher volume
  5) Close above EMA8 and JMA
  OUTPUT: Entry = close, Stop = pullback low, T1 = pullback high, T2 = 2R
{'=' * 55}
""")
