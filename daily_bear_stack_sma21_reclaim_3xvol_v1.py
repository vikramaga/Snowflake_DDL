# ============================================================
# NASDAQ / NYSE - Daily Bearish-Stack SMA21 Reclaim on 3x Volume (v1)
# ============================================================
#
# Finds a stock whose moving averages are stacked bearish and that rips
# from its EMA8 up through its SMA21 on a big volume spike:
#
#   1. BEARISH STACK (previous session): JMA < EMA8 < SMA21 < SMA50
#   2. FROM EMA8: today's low is at or below EMA8 (the move starts at EMA8)
#   3. CROSS ABOVE SMA21: yesterday closed at or below SMA21, today closes
#      above SMA21
#   4. VOLUME: today's volume > 3x yesterday's volume
#
# The stack is checked on the previous session because a strong cross-up day
# itself pulls JMA/EMA8 higher. JMA = Jurik MA (13, phase 40, power 2), the
# TradingView-matching version used by the other scanners.
#
# Entry = today's close; Stop = today's low; Target 1 = SMA50 (the next MA
# overhead), Target 2 = entry + 2R. These and the "from EMA8" reading
# (today's low touched EMA8) are reasonable defaults, not explicitly
# requested; all are in CFG.
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
    # history_days vs bars: 400 calendar days -> ~285 business days (pd.bdate_range);
    # SMA50 needs 50 bars, the JMA settles after ~100, the context SMA200 needs 200.
    "history_days"            : 400,
    "min_bars"                : 210,

    "jma_period"              : 13,
    "jma_phase"               : 40,
    "jma_power"               : 2,
    "ema_fast"                : 8,
    "sma_mid"                 : 21,
    "sma_slow"                : 50,
    "ema8_touch_tol_pct"      : 0.0,   # today's low <= EMA8 * (1 + tol/100)
    "vol_mult_prev_day"       : 3.0,   # today's volume > 3x yesterday's
    "target2_r"               : 2.0,
    "signal_lookback_days"    : 1,     # the latest session only

    "min_avg_volume"          : 200_000,
    "min_price"               : 5.0,

    "batch_size"              : 50,
    "batch_sleep"             : 1.5,
}

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

def calc_indicators(df, cfg):
    c = df["Close"]
    return {
        "jma":  calc_jma(c, cfg["jma_period"], cfg["jma_phase"], cfg["jma_power"]).values,
        "ema8": c.ewm(span=cfg["ema_fast"], adjust=False).mean().values,
        "s21":  c.rolling(cfg["sma_mid"]).mean().values,
        "s50":  c.rolling(cfg["sma_slow"]).mean().values,
        "s200": c.rolling(200).mean().values,
        "vavg": df["Volume"].rolling(20).mean().values,
    }

def check_pattern_at(df, ind, i, cfg):
    """
    SINGLE SOURCE OF TRUTH. Checks signal day i. Computes EVERY step (no
    short-circuit) so the funnel shows where tickers drop out.
    Returns (passed, details); details is {} only if data is too short.
    """
    if i < 100 or i >= len(df):
        return False, {}
    jma, e8, s21, s50 = ind["jma"], ind["ema8"], ind["s21"], ind["s50"]
    p = i - 1
    need = (jma[p], e8[p], s21[p], s50[p], e8[i], s21[i], s50[i])
    if not all(np.isfinite(x) for x in need):
        return False, {}
    o, h, l, c, v = (df[k].values.astype(float) for k in ("Open", "High", "Low", "Close", "Volume"))
    step1 = bool(jma[p] < e8[p] < s21[p] < s50[p])                            # bearish stack
    step2 = bool(l[i] <= e8[i] * (1 + cfg["ema8_touch_tol_pct"] / 100))       # from EMA8
    step3 = bool(c[p] <= s21[p] and c[i] > s21[i])                            # cross above SMA21
    vol_x = float(v[i] / v[p]) if v[p] > 0 else 0.0
    step4 = bool(vol_x > cfg["vol_mult_prev_day"])                            # volume spike
    passed = step1 and step2 and step3 and step4
    return passed, {
        "date": df.index[i], "i": i, "open": float(o[i]), "high": float(h[i]), "low": float(l[i]),
        "close": float(c[i]), "prev_close": float(c[p]), "vol_x": vol_x,
        "jma_p": float(jma[p]), "ema8_p": float(e8[p]), "sma21_p": float(s21[p]), "sma50_p": float(s50[p]),
        "ema8": float(e8[i]), "sma21": float(s21[i]), "sma50": float(s50[i]),
        "step1": step1, "step2": step2, "step3": step3, "step4": step4,
    }

# -- Diagnostic funnel (per ticker: did ANY of the lookback days pass
#    steps 1..k cumulatively)
FUNNEL_COUNTS = {
    "tickers_checked": 0,
    "passed_step1_bearish_stack": 0,
    "passed_step2_low_at_ema8": 0,
    "passed_step3_close_above_sma21": 0,
    "passed_step4_volume_3x_prev_day": 0,
}
_FUNNEL_KEYS = list(FUNNEL_COUNTS)[1:]

def find_signals(df, ind, cfg):
    hits, best = [], [False] * 4
    n = len(df)
    any_checked = False
    for i in range(max(0, n - cfg["signal_lookback_days"]), n):
        ok, d = check_pattern_at(df, ind, i, cfg)
        if not d: continue
        any_checked = True
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

def analyze_bear_stack_reclaim(sym, df_daily):
    if df_daily is None: return None
    price   = float(df_daily["Close"].iloc[-1])
    avg_vol = float(df_daily["Volume"].tail(20).mean())
    if price   < CFG["min_price"]:      return None
    if avg_vol < CFG["min_avg_volume"]: return None
    if len(df_daily) < CFG["min_bars"]: return None

    ind = calc_indicators(df_daily, CFG)
    hits = find_signals(df_daily, ind, CFG)
    if not hits: return None
    sig = hits[-1]
    entry, stop = sig["close"], sig["low"]
    if entry <= stop: return None
    risk = entry - stop
    risk_pct = risk / entry * 100
    t1 = sig["sma50"]
    t1_r = (t1 - entry) / risk
    i = sig["i"]
    s200 = ind["s200"][i]
    vavg = ind["vavg"][i]
    vol_avg_x = float(df_daily["Volume"].iloc[i] / vavg) if np.isfinite(vavg) and vavg > 0 else np.nan
    rng = sig["high"] - sig["low"]
    close_pos = (sig["close"] - sig["low"]) / rng if rng > 0 else 0.0
    above21 = (entry / sig["sma21"] - 1) * 100
    flags = [f"vol {sig['vol_x']:.1f}x prev day ({vol_avg_x:.1f}x 20d avg)",
             f"close {above21:+.1f}% vs SMA21"]
    if t1 <= entry: flags.append("already at/above SMA50")
    elif t1_r < 1: flags.append("SMA50 under 1R away")
    flags.append("above SMA200" if np.isfinite(s200) and entry > s200 else "below SMA200")
    if sig["close"] < sig["open"]: flags.append("red candle")

    score = 30
    score += min(20, (sig["vol_x"] - 3) * 4)                            # bigger volume spike
    score += min(10, max(0, (vol_avg_x - 1) * 4)) if np.isfinite(vol_avg_x) else 0
    score += 15 * min(1, max(0, (close_pos - 0.5) * 2))                 # strong close
    score += max(0, 10 - max(0, risk_pct - 4) * 2)                      # tighter stop
    score += min(10, max(0, t1_r) * 5)                                  # room to SMA50
    score += 5 if np.isfinite(s200) and entry > s200 else 0
    score = round(min(100, max(0, score)))

    return {
        "Score": score, "Price": round(price, 2),
        "Entry_Price": round(entry, 2), "Stop_Loss": round(stop, 2), "Risk_%": round(risk_pct, 1),
        "Target_1_SMA50": round(t1, 2), "T1_R": round(t1_r, 1),
        "Target_2_2R": round(entry + CFG["target2_r"] * risk, 2),
        "Vol_x_PrevDay": round(sig["vol_x"], 2), "Vol_x_20dAvg": round(vol_avg_x, 2),
        "Close_vs_SMA21_%": round(above21, 2), "Close_Position": round(close_pos, 2),
        "JMA_prev": round(sig["jma_p"], 2), "EMA8_prev": round(sig["ema8_p"], 2),
        "SMA21_prev": round(sig["sma21_p"], 2), "SMA50_prev": round(sig["sma50_p"], 2),
        "EMA8": round(sig["ema8"], 2), "SMA21": round(sig["sma21"], 2), "SMA50": round(sig["sma50"], 2),
        "Above_SMA200": "Yes" if np.isfinite(s200) and entry > s200 else "No",
        "Signal_Date": sig["date"].strftime("%Y-%m-%d"),
        "Days_Since_Signal": int(len(df_daily) - 1 - i),
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
        r = analyze_bear_stack_reclaim(sym, daily_map[sym])
        if r is None: continue
        r["Ticker"] = sym
        results.append(r)
        print(f"  OK {sym:<6} ${r['Price']:.2f}  Score {r['Score']}  vol {r['Vol_x_PrevDay']}x prev day  "
              f"close {r['Close_vs_SMA21_%']:+}% vs SMA21  {r['Signal_Date']}")
    except Exception: pass

fc = FUNNEL_COUNTS
print(f"\n{'=' * 65}\n  SCAN COMPLETE - {len(results)} matches from {got_daily} tickers with data")
print(f"  FUNNEL (last {CFG['signal_lookback_days']} session(s))")
print(f"  Tickers checked                                   : {fc['tickers_checked']}")
print(f"  Step 1 - JMA < EMA8 < SMA21 < SMA50 (prev session) : {fc['passed_step1_bearish_stack']}")
print(f"  Step 2 - + today's low at/below EMA8               : {fc['passed_step2_low_at_ema8']}")
print(f"  Step 3 - + close crossed above SMA21               : {fc['passed_step3_close_above_sma21']}")
print(f"  Step 4 - + volume > {CFG['vol_mult_prev_day']:g}x previous day             : {fc['passed_step4_volume_3x_prev_day']}  (= full pattern)")
print("=" * 65)

results.sort(key=lambda x: x["Score"], reverse=True)

# -- Save CSV + TradingView list
ts      = datetime.today().strftime("%Y%m%d_%H%M")
out_dir = os.environ.get("GITHUB_WORKSPACE", os.getcwd())
COLS = ["Ticker","Price","Score","Entry_Price","Stop_Loss","Risk_%","Target_1_SMA50","T1_R","Target_2_2R",
        "Vol_x_PrevDay","Vol_x_20dAvg","Close_vs_SMA21_%","Close_Position",
        "JMA_prev","EMA8_prev","SMA21_prev","SMA50_prev","EMA8","SMA21","SMA50",
        "Above_SMA200","Signal_Date","Days_Since_Signal","Flags"]
df_out = pd.DataFrame([{k: v for k, v in r.items() if not k.startswith("_")}
                       for r in results]) if results else pd.DataFrame(columns=COLS)
df_out = df_out[[c for c in COLS if c in df_out.columns]].reset_index(drop=True)

fpath = os.path.join(out_dir, f"daily_bear_stack_sma21_reclaim_3xvol_{ts}.csv")
try:
    from position_sizing import add_position_size
    df_out = add_position_size(df_out)
    try: __import__("best_picks").rank_best(df_out)
    except Exception as _bp_err: print(f"  Best 10 skipped: {_bp_err}")
except Exception as _ps_err:
    print(f"  Position sizing skipped: {_ps_err}")
df_out.to_csv(fpath, index=False)
print(f"\n  CSV -> {fpath}")
tv = os.path.join(out_dir, f"tv_daily_bear_stack_sma21_reclaim_3xvol_{ts}.txt")
with open(tv, "w") as f:
    f.write(f"###Bear-Stack SMA21 Reclaim 3x Vol {datetime.today().strftime('%Y-%m-%d')}\n")
    for r in results: f.write(f"NASDAQ:{r['Ticker']}\n")
print(f"  TradingView -> {tv}")
if results:
    print(f"\n  Tickers: {', '.join(r['Ticker'] for r in results)}")

# -- Email with CSV attached
EMAIL_COLS = ["Ticker","Price","Score","Entry_Price","Stop_Loss","Risk_%","Target_1_SMA50","T1_R",
              "Vol_x_PrevDay","Vol_x_20dAvg","Close_vs_SMA21_%","Above_SMA200","Signal_Date"]

def _fmt_cell(col, v):
    try:
        if col in ("Price","Entry_Price","Stop_Loss","Target_1_SMA50"): return f"${float(v):.2f}"
        if col in ("Risk_%","Close_vs_SMA21_%"): return f"{float(v):.1f}%"
        if col in ("Vol_x_PrevDay","Vol_x_20dAvg"): return f"{float(v):.1f}x"
        if col == "T1_R": return f"{float(v):.1f}R"
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
                if c == "Target_1_SMA50": sty = "color:#2563eb;font-weight:600"
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
<h1 style="margin:0;color:#60a5fa;font-size:19px">Bearish-Stack SMA21 Reclaim on 3x Volume</h1>
<p style="margin:6px 0 0;color:#94a3b8;font-size:12px">{datetime.today().strftime('%Y-%m-%d %H:%M UTC')} &middot; {cnt} match{'es' if cnt != 1 else ''}
 - previous session JMA &lt; EMA8 &lt; SMA21 &lt; SMA50; today the low touched EMA8 and the close crossed above SMA21
 on more than {CFG['vol_mult_prev_day']:g}x the previous day's volume</p>
</td></tr>
<tr><td style="padding:12px 26px;background:#0b1220"><p style="margin:0;color:#cbd5e1;font-size:12px">
FUNNEL: {fc['tickers_checked']} checked &rarr; {fc['passed_step1_bearish_stack']} bearish stack
&rarr; {fc['passed_step2_low_at_ema8']} + low at EMA8
&rarr; {fc['passed_step3_close_above_sma21']} + close above SMA21
&rarr; <b style="color:#facc15">{fc['passed_step4_volume_3x_prev_day']} + volume &gt; {CFG['vol_mult_prev_day']:g}x prev day = full match</b></p></td></tr>
<tr><td style="padding:16px"><div style="overflow-x:auto;border:1px solid #e2e8f0;border-radius:8px">
<table style="border-collapse:collapse;width:100%"><thead><tr>{th}</tr></thead><tbody>{rows}</tbody></table></div>
{tick_html}
<p style="font-size:11px;color:#64748b;margin:10px 0 0">Entry = today's close &middot; Stop = today's low &middot;
Target 1 = SMA50 (T1_R = distance in R), Target 2 = entry + {CFG['target2_r']:g}R (in the CSV). The MA stack is checked on the previous session.
Full results (with Quantity for a fixed max loss) attached as CSV.</p>
</td></tr>
<tr><td style="background:#f8fafc;padding:10px;text-align:center;color:#94a3b8;font-size:10px">
Not financial advice &middot; Auto-generated by GitHub Actions</td></tr>
</table></td></tr></table></body></html>"""

        plain = [f"Bear-Stack SMA21 Reclaim 3x Vol - {datetime.today().strftime('%Y-%m-%d')}",
                 f"{cnt} matches",
                 f"FUNNEL: {fc['tickers_checked']} checked -> {fc['passed_step1_bearish_stack']} bearish stack -> "
                 f"{fc['passed_step2_low_at_ema8']} low at EMA8 -> {fc['passed_step3_close_above_sma21']} above SMA21 -> "
                 f"{fc['passed_step4_volume_3x_prev_day']} volume spike (=full match)",
                 "=" * 60]
        for r in rl[:50]:
            plain.append(f"{r['Ticker']:<6} ${r['Price']:.2f}  Score:{r['Score']}  Entry:${r['Entry_Price']:.2f}  "
                         f"SL:${r['Stop_Loss']:.2f} ({r['Risk_%']}%)  T1 SMA50:${r['Target_1_SMA50']:.2f} ({r['T1_R']}R)  "
                         f"vol {r['Vol_x_PrevDay']}x prev day")
        if rl:
            plain += ["", "Tickers: " + ", ".join(r["Ticker"] for r in rl)]
        else:
            plain.append("No matches today")
        plain.append("\nFull results in CSV attachment.")

        subj = (f"Bear-Stack SMA21 Reclaim 3x Vol - {cnt} signal{'s' if cnt != 1 else ''} - "
                f"{datetime.today().strftime('%Y-%m-%d')}")
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

# -- Charts for top 5: last 90 daily candles + JMA, EMA8, SMA21, SMA50
if results:
    top = results[:5]
    fig, axes = plt.subplots(len(top), 1, figsize=(13, 4.2*len(top)), facecolor="#0f172a")
    if len(top) == 1: axes = [axes]
    for ax, r in zip(axes, top):
        d = r["_df"].iloc[-90:]
        n_ = len(d)
        x = np.arange(n_)
        ax.set_facecolor("#0f172a")
        cols = ["#22c55e" if c > o else "#ef4444" for o, c in zip(d["Open"], d["Close"])]
        ax.vlines(x, d["Low"], d["High"], color=cols, lw=0.8)
        ax.bar(x, (d["Close"] - d["Open"]).abs().clip(lower=1e-9), bottom=np.minimum(d["Open"], d["Close"]),
               color=cols, width=0.6)
        for key, c_, lab in (("jma", "#f472b6", "JMA"), ("ema8", "#facc15", "EMA8"),
                             ("s21", "#38bdf8", "SMA21"), ("s50", "#a78bfa", "SMA50")):
            ax.plot(x, r["_ind"][key][-n_:], color=c_, lw=1.0, label=lab)
        k_ = np.where(d.index == pd.Timestamp(r["Signal_Date"]))[0]
        if len(k_): ax.axvline(k_[0], color="#334155", ls="--", lw=0.8)
        ax.axhline(r["Stop_Loss"], color="#ef4444", ls=":", lw=0.9)
        ax.text(n_ - 0.4, r["Stop_Loss"], f" Stop {r['Stop_Loss']:.2f}", color="#ef4444", fontsize=7, va="center")
        ax.set_xticks(x[::10]); ax.set_xticklabels([t.strftime("%m/%d") for t in d.index[::10]], fontsize=7)
        ax.set_title(f"{r['Ticker']} | USD {r['Price']:.2f} | Score {r['Score']} | vol {r['Vol_x_PrevDay']}x prev day | "
                     f"T1 SMA50 {r['Target_1_SMA50']:.2f} ({r['T1_R']}R) | {r['Signal_Date']}",
                     color="#e2e8f0", fontsize=9, fontweight="bold")
        ax.tick_params(colors="#94a3b8", labelsize=7)
        for sp in ax.spines.values(): sp.set_edgecolor("#1e3a5f")
        ax.grid(color="#1e3a5f", ls="--", lw=0.5, alpha=0.6)
        ax.legend(loc="upper left", fontsize=7, facecolor="#0f172a", labelcolor="#cbd5e1", edgecolor="#1e3a5f")
    plt.suptitle(f"Bearish-Stack SMA21 Reclaim on 3x Volume - {datetime.today().strftime('%Y-%m-%d')}",
                 color="#60a5fa", fontsize=12, fontweight="bold")
    plt.tight_layout()
    cp = os.path.join(out_dir, f"daily_bear_stack_sma21_reclaim_3xvol_chart_{ts}.png")
    plt.savefig(cp, dpi=130, bbox_inches="tight", facecolor="#0f172a")
    plt.close()
    print(f"  Chart -> {cp}")

print(f"""
{'=' * 55}
  PATTERN (latest session)
  1) Previous session: JMA < EMA8 < SMA21 < SMA50 (bearish stack)
  2) Today's low at or below EMA8 (the move starts from EMA8)
  3) Yesterday closed at/below SMA21, today closes above SMA21
  4) Today's volume > {CFG['vol_mult_prev_day']:g}x yesterday's volume
  OUTPUT: Entry = close, Stop = today's low, T1 = SMA50, T2 = +{CFG['target2_r']:g}R
{'=' * 55}
""")
