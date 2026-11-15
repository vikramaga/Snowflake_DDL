# ============================================================
# NASDAQ / NYSE - SB Reclaim: pullback shakeout + reclaim of JMA / EMA8 / SMA21
#                 (Daily AND Weekly, Strict and Touch modes)  v1
# ============================================================
#
# From Vik's SB chart (2026-11-15; weekly). Lines: JMA(13,40,2), EMA8, SMA21,
# SMA50, SMA150. Checked separately on DAILY bars and on WEEKLY bars (weeks end
# Friday, resampled from the daily download; the in-progress week counts):
#   1. TREND STACK: SMA21 > SMA50 > SMA150, SMA50 and SMA150 higher than 10 bars ago
#   2. FRESH HIGH 3-20 bars ago: the highest high of the last 60 daily / 52 weekly
#      bars, nothing higher since, with JMA and EMA8 above SMA21 at that high
#   3. PULLBACK: JMA fell on at least one bar after the high; every low since the
#      high stayed above today's SMA50
#   4. SHAKEOUT: a red candle in the 3 bars before today, after the high, that
#        Strict: CLOSED below SMA21 (SB)       Touch: only its LOW reached SMA21 (NDSN, 24 Jul 2026)
#   5. RECLAIM today: green candle closing above JMA, EMA8 and SMA21, in the upper
#      half of its range, and
#        Strict: yesterday's close was below EMA8
#        Touch : yesterday's close was below JMA or EMA8 (first close back above both)
#   Mode = Strict if both strict rules pass, else Touch.
#
# Entry = latest close; Stop = lowest low from the shakeout candle to today;
# T1 = the fresh high (Reward_R); T2 = 2R. Liquidity: price >= $2, 20-day average
# volume >= 200k (defaults, in CFG).
# Also creates / replaces a TradingView watchlist "SB Reclaim - YYYY-MM-DD" (IST
# date) with Weekly / Daily, Strict / Touch sections (needs TV_SESSIONID secrets).
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
    # 1400 calendar days -> ~200 weekly bars (weekly SMA150 + 10-week slope need 160)
    "history_days"            : 1400,
    "min_bars_D"              : 200,
    "min_bars_W"              : 165,
    "slope_bars"              : 10,
    "high_window_D"           : 60,    # fresh high = highest high of 60 daily bars
    "high_window_W"           : 52,    # ... / 52 weekly bars
    "high_min_age"            : 3,     # high 3..20 bars before today
    "high_max_age"            : 20,
    "shakeout_bars"           : 3,     # shakeout candle within the 3 bars before today
    "min_price"               : 2.0,
    "min_avg_volume"          : 200_000,
    "signal_lookback_days"    : 1,
    "tv_list_prefix"          : "SB Reclaim",

    "batch_size"              : 50,
    "batch_sleep"             : 1.5,
}

TFS = ("W", "D")
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
    agg = {"Open": "first", "High": "max", "Low": "min", "Close": "last", "Volume": "sum"}
    w = df.resample("W-FRI").agg(agg).dropna(subset=["Close"])
    w["Days"] = df["Close"].resample("W-FRI").count().reindex(w.index).values
    return w

def calc_indicators(df):
    cs = df["Close"].astype(float)
    return {
        "jma":  calc_jma(cs, 13, 40, 2).values,
        "e8":   cs.ewm(span=8, adjust=False).mean().values,
        "s21":  cs.rolling(21).mean().values,
        "s50":  cs.rolling(50).mean().values,
        "s150": cs.rolling(150).mean().values,
    }

def check_at(o, h, l, c, ind, i, cfg, tf):
    """
    SINGLE SOURCE OF TRUTH for bar i. Every step is computed (no short-circuit).
    Returns (passed, details). details["mode"] = "Strict" / "Touch" / None.
    """
    jma, e8, s21, s50, s150 = (ind[k] for k in ("jma", "e8", "s21", "s50", "s150"))
    sb, win = cfg["slope_bars"], cfg[f"high_window_{tf}"]
    if i < max(sb, win) + cfg["high_max_age"] or not np.isfinite(s150[i - sb]):
        return False, {}
    st = {}
    st["step1"] = bool(s21[i] > s50[i] > s150[i] and s50[i] > s50[i - sb] and s150[i] > s150[i - sb])
    hb = None
    for k in range(i - cfg["high_max_age"], i - cfg["high_min_age"] + 1):
        if h[k] >= h[k - win + 1:k + 1].max() and h[k] >= h[k:i + 1].max() and min(jma[k], e8[k]) > s21[k]:
            hb = k
            break
    st["step2"] = hb is not None
    strict_k = touch_k = None
    if hb is not None:
        st["step3"] = bool(any(jma[k] < jma[k - 1] for k in range(hb + 1, i))
                           and l[hb + 1:i + 1].min() > s50[i])
        for k in range(i - cfg["shakeout_bars"], i):
            if k > hb and c[k] < o[k]:
                if c[k] < s21[k]: strict_k = k
                elif l[k] <= s21[k]: touch_k = k
    else:
        st["step3"] = False
    green_reclaim = bool(c[i] > o[i] and c[i] > max(jma[i], e8[i], s21[i]) and c[i] >= (h[i] + l[i]) / 2)
    strict = strict_k is not None and green_reclaim and c[i - 1] < e8[i - 1]
    touch = (strict_k is not None or touch_k is not None) and green_reclaim and c[i - 1] < max(jma[i - 1], e8[i - 1])
    st["step4"] = strict_k is not None or touch_k is not None
    st["step5"] = bool(strict or touch)
    mode = "Strict" if strict else ("Touch" if touch else None)
    sk = strict_k if strict else (strict_k if strict_k is not None else touch_k)
    passed = all(st[f"step{k}"] for k in range(1, 6))
    return passed, {**st, "mode": mode, "hb": hb, "sk": sk, "i": i}

_STEPS = ["trend_stack", "fresh_high", "pullback_jma_down", "shakeout_at_sma21", "reclaim_today"]
FUNNEL_COUNTS = {tf: {"tickers_checked": 0, **{f"passed_step{k+1}_{n}": 0 for k, n in enumerate(_STEPS)}}
                 for tf in TFS}

def analyze_tf(sym, df, tf):
    if df is None or len(df) < CFG[f"min_bars_{tf}"]: return None
    ind = calc_indicators(df)
    o, h, l, c = (df[k].values.astype(float) for k in ("Open", "High", "Low", "Close"))
    i = len(df) - 1
    ok, d = check_at(o, h, l, c, ind, i, CFG, tf)
    if not d: return None
    fc = FUNNEL_COUNTS[tf]; fc["tickers_checked"] += 1
    cum = True
    for k, key in enumerate(list(fc)[1:]):
        cum = cum and d[f"step{k+1}"]; fc[key] += cum
    if not ok: return None
    entry = float(c[i]); stop = float(l[d["sk"]:i + 1].min())
    if stop <= 0 or entry <= stop: return None
    risk = entry - stop; risk_pct = risk / entry * 100
    high = float(h[d["hb"]])
    t1 = high if high > entry else entry + 2 * risk
    rr = (t1 - entry) / risk
    di = df.index
    flags = [f"shakeout {di[d['sk']].strftime('%Y-%m-%d')} {'closed below' if d['mode'] == 'Strict' else 'touched'} SMA21",
             f"high {di[d['hb']].strftime('%Y-%m-%d')}"]
    if high <= entry: flags.append("already above the high - T1 = 2R")
    if tf == "W" and "Days" in df and int(df["Days"].iloc[-1]) < 5:
        flags.append(f"week in progress ({int(df['Days'].iloc[-1])} days)")
    score = 50 + (15 if d["mode"] == "Strict" else 0) + min(15, max(0, rr) * 6) + max(0, 15 - max(0, risk_pct - 3) * 1.5)
    score += 5 if tf == "W" else 0
    return {
        "Score": round(min(100, score)), "Timeframe": TF_NAME[tf], "Mode": d["mode"], "Price": round(entry, 2),
        "Entry_Price": round(entry, 2), "Stop_Loss": round(stop, 2), "Risk_%": round(risk_pct, 1),
        "Target_1": round(t1, 2), "Reward_R": round(rr, 1), "Target_2": round(entry + 2 * risk, 2),
        "Prior_High": round(high, 2), "High_Date": di[d["hb"]].strftime("%Y-%m-%d"),
        "Shakeout_Date": di[d["sk"]].strftime("%Y-%m-%d"),
        "JMA": round(float(ind["jma"][i]), 2), "EMA8": round(float(ind["e8"][i]), 2),
        "SMA21": round(float(ind["s21"][i]), 2), "SMA50": round(float(ind["s50"][i]), 2),
        "Signal_Date": di[i].strftime("%Y-%m-%d"), "Flags": " | ".join(flags),
        "_df": df, "_ind": ind,
    }

def analyze_stock(sym, df_daily):
    if df_daily is None: return []
    if float(df_daily["Close"].iloc[-1]) < CFG["min_price"]: return []
    if float(df_daily["Volume"].tail(20).mean()) < CFG["min_avg_volume"]: return []
    out = []
    for tf, df in (("W", resample_weekly(df_daily)), ("D", df_daily)):
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
            print(f"  OK {sym:<6} {r['Timeframe']:<6} {r['Mode']:<6} ${r['Price']:.2f}  Score {r['Score']}  "
                  f"stop {r['Stop_Loss']} ({r['Risk_%']}%)  T1 {r['Target_1']} ({r['Reward_R']}R)")
    except Exception: pass

both = {r["Ticker"] for r in results if r["Timeframe"] == "Daily"} & {r["Ticker"] for r in results if r["Timeframe"] == "Weekly"}
for r in results:
    r["Both_Timeframes"] = "Yes" if r["Ticker"] in both else "No"
    if r["Ticker"] in both: r["Score"] = min(100, r["Score"] + 10)

print(f"\n{'=' * 65}\n  SCAN COMPLETE - {len(results)} matches from {got_daily} tickers with data")
for tf in TFS:
    fc = FUNNEL_COUNTS[tf]; ks = list(fc)
    print(f"  {TF_NAME[tf].upper()} FUNNEL")
    print(f"    Tickers checked                                : {fc[ks[0]]}")
    print(f"    Step 1 - SMA21 > SMA50 > SMA150, rising        : {fc[ks[1]]}")
    print(f"    Step 2 - + fresh high 3-20 bars ago            : {fc[ks[2]]}")
    print(f"    Step 3 - + JMA turned down, held SMA50         : {fc[ks[3]]}")
    print(f"    Step 4 - + red shakeout at/below SMA21         : {fc[ks[4]]}")
    print(f"    Step 5 - + green reclaim of JMA/EMA8/SMA21     : {fc[ks[5]]}  (= full pattern)")
print("=" * 65)

ORDER = {("Weekly", "Strict"): 0, ("Weekly", "Touch"): 1, ("Daily", "Strict"): 2, ("Daily", "Touch"): 3}
results.sort(key=lambda x: (ORDER[(x["Timeframe"], x["Mode"])], -x["Score"]))

# -- Save CSV + TradingView text list
ts      = datetime.today().strftime("%Y%m%d_%H%M")
out_dir = os.environ.get("GITHUB_WORKSPACE", os.getcwd())
COLS = ["Ticker","Timeframe","Mode","Both_Timeframes","Price","Score","Entry_Price","Stop_Loss","Risk_%","Target_1",
        "Reward_R","Target_2","Prior_High","High_Date","Shakeout_Date","JMA","EMA8","SMA21","SMA50","Signal_Date","Flags"]
df_out = pd.DataFrame([{k: v for k, v in r.items() if not k.startswith("_")}
                       for r in results]) if results else pd.DataFrame(columns=COLS)
df_out = df_out[[c for c in COLS if c in df_out.columns]].reset_index(drop=True)
fpath = os.path.join(out_dir, f"sb_reclaim_{ts}.csv")
try:
    from position_sizing import add_position_size
    df_out = add_position_size(df_out)
    try: __import__("best_picks").rank_best(df_out)
    except Exception as _bp_err: print(f"  Best 10 skipped: {_bp_err}")
except Exception as _ps_err:
    print(f"  Position sizing skipped: {_ps_err}")
df_out.to_csv(fpath, index=False)
print(f"\n  CSV -> {fpath}")
SECTIONS = [("Weekly", "Strict"), ("Weekly", "Touch"), ("Daily", "Strict"), ("Daily", "Touch")]
tv = os.path.join(out_dir, f"tv_sb_reclaim_{ts}.txt")
with open(tv, "w") as f:
    for tf_, md in SECTIONS:
        names = [r["Ticker"] for r in results if r["Timeframe"] == tf_ and r["Mode"] == md]
        if names:
            f.write(f"###{tf_} {md}\n")
            for s in names: f.write(f"NASDAQ:{s}\n")
print(f"  TradingView file -> {tv}")
for tf_, md in SECTIONS:
    names = [r["Ticker"] for r in results if r["Timeframe"] == tf_ and r["Mode"] == md]
    if names: print(f"  {tf_} {md}: {', '.join(names)}")

# -- TradingView watchlist "SB Reclaim - YYYY-MM-DD" (IST date), replaced if it already exists
TV_STATUS = "not created"
def _tv_sync():
    global TV_STATUS
    if not results:
        TV_STATUS = "skipped (no matches)"; print(f"[TradingView] {TV_STATUS}"); return
    if not (os.environ.get("TV_SESSIONID") and os.environ.get("TV_SESSIONID_SIGN")):
        TV_STATUS = "skipped (TV_SESSIONID secrets not set)"; print(f"[TradingView] {TV_STATUS}"); return
    from zoneinfo import ZoneInfo
    name = f"{CFG['tv_list_prefix']} - {datetime.now(ZoneInfo('Asia/Kolkata')):%Y-%m-%d}"
    os.environ["TV_WATCHLIST"] = name
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "claude_run", "tools"))
    import tv_watchlist_sync as tvs
    tvs.NAME = name
    s = tvs.session(); lists = tvs.get_lists(s); ex = tvs.exchange_map()
    symbols, seen = [], set()
    for tf_, md in SECTIONS:
        syms = [tvs.tv_symbol(r["Ticker"], ex) for r in results if r["Timeframe"] == tf_ and r["Mode"] == md]
        new = [x for x in syms if x not in seen]
        if new:
            symbols.append(f"###{tf_} {md}"); symbols.extend(new); seen.update(new)
        elif syms:
            symbols.append((f"###{tf_} {md} - listed above: " + ", ".join(x.split(':')[-1] for x in syms))[:100])
    tvs.replace_list(s, lists, symbols)
    after = next((x for x in tvs.get_lists(s) if x.get("name") == name), None)
    got = [x for x in (after or {}).get("symbols") or [] if not str(x).startswith("###")]
    TV_STATUS = f"'{name}' - {len(got)} symbols"
    print(f"[TradingView] ✅ {TV_STATUS}")
try:
    _tv_sync()
except Exception as e:
    TV_STATUS = f"failed ({type(e).__name__}: {e})"
    print(f"[TradingView] {TV_STATUS}")

# -- Email with CSV attached
EMAIL_COLS = ["Ticker","Timeframe","Mode","Both_Timeframes","Price","Score","Entry_Price","Stop_Loss","Risk_%",
              "Target_1","Reward_R","Shakeout_Date","High_Date"]

def _fmt_cell(col, v):
    try:
        if col in ("Price","Entry_Price","Stop_Loss","Target_1"): return f"${float(v):.2f}"
        if col == "Risk_%": return f"{float(v):.1f}%"
        if col == "Reward_R": return f"{float(v):.1f}R"
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
        print("[Email] GMAIL_USER / GMAIL_PASS / EMAIL_TO missing - skipped"); return
    eto = [e.strip() for e in et.split(",") if e.strip()]
    nW = sum(r["Timeframe"] == "Weekly" for r in rl); nD = len(rl) - nW
    try:
        th = "".join(f'<th style="background:#1e293b;color:#e2e8f0;padding:8px 10px;font-size:11px;'
                     f'border-bottom:2px solid #3b82f6;white-space:nowrap">{c}</th>' for c in EMAIL_COLS)
        rows = ""
        for i, r in enumerate(rl[:80]):
            bg = "#fff" if i % 2 == 0 else "#f0f9ff"
            tds = ""
            for c in EMAIL_COLS:
                sty = "font-weight:700" if c == "Ticker" else ""
                if c == "Score": sty = "font-weight:700;background:#166534;color:#fff;text-align:center"
                if c == "Mode" and r["Mode"] == "Strict": sty = "font-weight:700;color:#7c3aed"
                if c == "Both_Timeframes" and r["Both_Timeframes"] == "Yes": sty = "font-weight:700;color:#16a34a"
                if c == "Stop_Loss": sty = "color:#ef4444"
                tds += f'<td style="padding:6px 10px;font-size:12px;{sty}">{_fmt_cell(c, r.get(c))}</td>'
            rows += f'<tr style="background:{bg}">{tds}</tr>'
        if not rows:
            rows = f'<tr><td colspan="{len(EMAIL_COLS)}" style="padding:20px;text-align:center;color:#94a3b8">No matches today</td></tr>'
        fun = ""
        for tf in TFS:
            fc = FUNNEL_COUNTS[tf]; ks = list(fc)
            fun += (f"<b>{TF_NAME[tf]}:</b> {fc[ks[0]]} checked &rarr; {fc[ks[1]]} MA stack &rarr; {fc[ks[2]]} + fresh high "
                    f"&rarr; {fc[ks[3]]} + JMA down / held SMA50 &rarr; {fc[ks[4]]} + shakeout at SMA21 "
                    f"&rarr; <b style=\"color:#facc15\">{fc[ks[5]]} + reclaim = full match</b><br>")
        html = f"""<!DOCTYPE html><html><body style="margin:0;background:#f1f5f9;font-family:'Segoe UI',Arial,sans-serif">
<table width="100%" cellpadding="0" cellspacing="0"><tr><td align="center" style="padding:20px 10px">
<table width="100%" style="max-width:900px;background:#fff;border-radius:12px;overflow:hidden">
<tr><td style="background:linear-gradient(135deg,#0f172a,#1e3a5f);padding:20px 26px">
<h1 style="margin:0;color:#60a5fa;font-size:19px">SB Reclaim: Pullback Shakeout + Reclaim (Daily + Weekly)</h1>
<p style="margin:6px 0 0;color:#94a3b8;font-size:12px">{datetime.today().strftime('%Y-%m-%d %H:%M UTC')} &middot; {nW} weekly, {nD} daily
 &middot; TradingView: {TV_STATUS}</p></td></tr>
<tr><td style="padding:12px 26px;background:#0b1220"><p style="margin:0;color:#cbd5e1;font-size:12px;line-height:1.6">{fun}</p></td></tr>
<tr><td style="padding:16px"><div style="overflow-x:auto;border:1px solid #e2e8f0;border-radius:8px">
<table style="border-collapse:collapse;width:100%"><thead><tr>{th}</tr></thead><tbody>{rows}</tbody></table></div>
<p style="font-size:11px;color:#64748b;margin:10px 0 0">SMA21 &gt; SMA50 &gt; SMA150 rising; fresh high 3-20 bars ago; JMA turned down, lows held SMA50;
a red shakeout candle in the last 3 bars - <b>Strict</b>: closed below SMA21 (SB), <b>Touch</b>: its low reached SMA21 (NDSN 24 Jul);
today a green candle closes above JMA, EMA8 and SMA21. Entry = close &middot; Stop = lowest low since the shakeout &middot;
Target_1 = the prior high (2R if already above it) &middot; Both_Timeframes = Yes when the stock matched on daily and weekly.
Full results (with Quantity for a fixed max loss) attached as CSV.</p></td></tr>
<tr><td style="background:#f8fafc;padding:10px;text-align:center;color:#94a3b8;font-size:10px">
Not financial advice &middot; Auto-generated by GitHub Actions</td></tr></table></td></tr></table></body></html>"""
        plain = [f"SB Reclaim (Daily + Weekly) - {datetime.today().strftime('%Y-%m-%d')}", f"{nW} weekly, {nD} daily",
                 f"TradingView: {TV_STATUS}", "=" * 60]
        for r in rl[:80]:
            plain.append(f"{r['Ticker']:<6} {r['Timeframe']:<6} {r['Mode']:<6} ${r['Price']:.2f}  SL:${r['Stop_Loss']:.2f} "
                         f"({r['Risk_%']}%)  T1:${r['Target_1']:.2f} ({r['Reward_R']}R)  both:{r['Both_Timeframes']}")
        if not rl: plain.append("No matches today")
        subj = f"SB Reclaim - {nW} weekly, {nD} daily - {datetime.today().strftime('%Y-%m-%d')}"
        msg = MIMEMultipart("mixed"); msg["Subject"] = subj; msg["From"] = gu; msg["To"] = ", ".join(eto)
        alt = MIMEMultipart("alternative")
        alt.attach(MIMEText("\n".join(plain), "plain")); alt.attach(MIMEText(html, "html")); msg.attach(alt)
    except Exception as e:
        print(f"[Email] Failed to build email body: {type(e).__name__}: {e}"); return
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
