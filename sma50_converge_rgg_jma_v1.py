# ============================================================
# NASDAQ / NYSE - SMA50 Closing on SMA150 + RGG JMA Reclaim - daily, v1 (RGG within 7 sessions)
# ============================================================
#
# From Vik's chart (2026-11-18): blue SMA50 rising toward the red SMA150 from
# below, price already above SMA150, a pullback to EMA8 / SMA21, then an RGG
# (red, green, green on rising volume) that closes back above JMA - preferably
# also above the monthly Camarilla H4.
#
# Rules on the latest daily candle (all required):
#   1. SMA50 CLOSING ON SMA150: SMA50 below SMA150 by at most 8% (or crossed above
#      it by at most 1% - "just crossed"), the gap narrower than 10 sessions ago by
#      at least 0.3 points, and SMA50 higher than 10 sessions ago.
#   2. Close above SMA150.
#   3. RGG within the last 7 sessions: red, green, green, the second green's volume
#      above the first green's (the most recent valid RGG is used).
#   4. FROM EMA8 / SMA21: one of the three RGG candles' low reached EMA8 or SMA21
#      (within 0.5%).
#   5. JMA RECLAIM: the second green closed above JMA, and the red or the first green
#      closed at or below JMA.
#   6. LOW HELD: no candle after the RGG traded below the lowest low of the 3 RGG candles.
# Preferred (grouped first, +score): close above the monthly Camarilla H4
#   (H4 = previous month's close + 1.1 x (prev month high - low) / 2);
#   "crossed H4 today" when yesterday's close was at or below it.
# Score extras: RGG today, still above JMA today.
# Entry = today's close | Stop = lowest low of the RGG candles
# T1 = monthly H4 if still above the entry, else the 60-session high if above,
#      else 2R | T2 = 2R. Liquidity $5 / 200k.
# TradingView list "SMA50 Converge RGG - YYYY-MM-DD" (IST). Settings in CFG.
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
    "history_days"     : 550,     # SMA150 + JMA warm-up
    "min_bars"         : 200,
    "atr_period"       : 14,
    "slope_bars"       : 10,
    "gap_max_pct"      : 8.0,     # SMA150 at most 8% above SMA50
    "gap_min_pct"      : -1.0,    # or SMA50 at most 1% above SMA150 (just crossed)
    "gap_narrow_pts"   : 0.3,     # gap narrower than 10 sessions ago by >= 0.3 points
    "touch_tol"        : 0.005,   # low within 0.5% of EMA8 / SMA21
    "high_window"      : 60,
    "rgg_lookback"     : 7,       # RGG may complete in any of the last 7 sessions
    "min_price"        : 5.0,
    "min_avg_volume"   : 200_000,
    "tv_list_prefix"   : "SMA50 Converge RGG",
    "batch_size"       : 50,
    "batch_sleep"      : 1.5,
}

FUNNEL = {"tickers_checked": 0, "sma50_closing": 0, "above_sma150": 0, "rgg": 0,
          "from_ema8_sma21": 0, "jma_reclaim": 0, "low_held": 0, "above_h4": 0}

def calc_jma(series, period=13, phase=40, power=2):
    """Jurik Moving Average, same formula as TradingView's "Jurik Moving Average" indicator."""
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

def monthly_camarilla(df):
    """H3 / H4 from the PREVIOUS completed month, aligned to each daily bar."""
    per = df.index.to_period("M")
    g = df.groupby(per).agg(H=("High", "max"), L=("Low", "min"), C=("Close", "last"))
    p = g.shift(1)
    rng = p.H - p.L
    lv = pd.DataFrame({"H3": p.C + rng * 1.1 / 4, "H4": p.C + rng * 1.1 / 2}, index=g.index).reindex(per)
    return lv["H3"].values.astype(float), lv["H4"].values.astype(float)

def analyze_stock(sym, df):
    if df is None or len(df) < CFG["min_bars"]: return None
    o, h, l, c, v = (df[k].values.astype(float) for k in ("Open", "High", "Low", "Close", "Volume"))
    i = len(df) - 1
    if c[i] < CFG["min_price"] or float(np.mean(v[-20:])) < CFG["min_avg_volume"]: return None
    cs = df["Close"].astype(float)
    s50 = cs.rolling(50).mean().values; s150 = cs.rolling(150).mean().values
    s21 = cs.rolling(21).mean().values; e8 = cs.ewm(span=8, adjust=False).mean().values
    k = CFG["slope_bars"]
    if not (np.isfinite(s150[i - k]) and np.isfinite(s50[i - k])): return None
    FUNNEL["tickers_checked"] += 1
    gap = (s150[i] - s50[i]) / s150[i] * 100
    gap_then = (s150[i - k] - s50[i - k]) / s150[i - k] * 100
    if not (CFG["gap_min_pct"] <= gap <= CFG["gap_max_pct"] and gap <= gap_then - CFG["gap_narrow_pts"]
            and s50[i] > s50[i - k]): return None
    FUNNEL["sma50_closing"] += 1
    if not c[i] > s150[i]: return None
    FUNNEL["above_sma150"] += 1
    # RGG completed on bar g (0-6 sessions ago); the most recent valid one is used
    tol = 1 + CFG["touch_tol"]
    jma = calc_jma(cs).values
    stage, found = 0, None
    for g in range(i, i - CFG["rgg_lookback"], -1):
        if not (c[g - 2] < o[g - 2] and c[g - 1] > o[g - 1] and c[g] > o[g] and v[g] > v[g - 1]): continue
        stage = max(stage, 1)
        t8 = [j for j in (g - 2, g - 1, g) if l[j] <= e8[j] * tol]
        t21 = [j for j in (g - 2, g - 1, g) if l[j] <= s21[j] * tol]
        if not (t8 or t21): continue
        stage = max(stage, 2)
        if not (c[g] > jma[g] and (c[g - 2] <= jma[g - 2] or c[g - 1] <= jma[g - 1])): continue
        stage = max(stage, 3)
        rgg_low = float(min(l[g - 2:g + 1]))
        if g < i and float(np.min(l[g + 1:i + 1])) < rgg_low: continue      # a later candle broke the RGG low
        stage = 4; found = (g, t8, t21, rgg_low); break
    for st, key in ((1, "rgg"), (2, "from_ema8_sma21"), (3, "jma_reclaim"), (4, "low_held")):
        if stage >= st: FUNNEL[key] += 1
    if found is None: return None
    g, t8, t21, rgg_low = found
    days_ago = i - g

    mh3, mh4 = monthly_camarilla(df)
    H4 = mh4[i]
    above_h4 = np.isfinite(H4) and c[i] > H4
    crossed_h4 = above_h4 and c[i - 1] <= mh4[i - 1]
    if above_h4: FUNNEL["above_h4"] += 1
    stop = rgg_low; entry = float(c[i]); risk = entry - stop
    if risk <= 0: return None
    hi = float(np.max(h[max(0, i - CFG["high_window"]):i]))
    if np.isfinite(H4) and H4 > entry: t1, t1_src = float(H4), "monthly H4"
    elif hi > entry: t1, t1_src = hi, f"{CFG['high_window']}-session high"
    else: t1, t1_src = entry + 2 * risk, "2R"
    pc = np.concatenate([[np.nan], c[:-1]])
    tr = np.nanmax(np.vstack([h - l, np.abs(h - pc), np.abs(l - pc)]), axis=0)
    atr = float(pd.Series(tr).rolling(CFG["atr_period"]).mean().values[i])
    vavg = float(np.mean(v[g - 20:g])); vx = v[g] / vavg if vavg > 0 else np.nan
    rng = h[g] - l[g]; cpos = (c[g] - l[g]) / rng if rng > 0 else 0.5
    above_jma_now = c[i] > jma[i]
    touched = "EMA8 + SMA21" if (t8 and t21) else ("EMA8" if t8 else "SMA21")
    sma_state = "just crossed above SMA150" if gap < 0 else f"{gap:.1f}% below SMA150"
    score = 50 + (15 if above_h4 else 0) + (5 if crossed_h4 else 0) + max(0, 10 - max(gap, 0) * 1.25) \
            + (5 if np.isfinite(vx) and vx >= 1.2 else 0) + (5 if cpos >= 0.7 else 0) + (5 if t8 and t21 else 0) \
            + (5 if s150[i] >= s150[i - k] else 0) + (5 if above_jma_now else 0) + (5 if days_ago == 0 else 0)
    flags = [f"SMA50 {sma_state} (was {gap_then:.1f}%)", f"pullback reached {touched}",
             "RGG today" if days_ago == 0 else f"RGG {days_ago} session(s) ago, low held since",
             "red closed below JMA" if c[g - 2] <= jma[g - 2] else "first green closed below JMA",
             "still above JMA" if above_jma_now else "back below JMA now",
             f"2nd green vol x{v[g] / max(v[g - 1], 1):.2f} vs 1st, x{vx:.1f} 20d avg",
             ("crossed monthly H4 today" if crossed_h4 else "above monthly H4") if above_h4 else "below monthly H4",
             "SMA150 rising" if s150[i] >= s150[i - k] else "SMA150 falling"]
    return {"Score": round(min(100, score)), "H4_Status": ("Crossed today" if crossed_h4 else "Above") if above_h4 else "Below",
            "Price": round(entry, 2), "Entry_Price": round(entry, 2), "Stop_Loss": round(stop, 2),
            "RGG_Date": df.index[g].strftime("%Y-%m-%d"), "Days_Since_RGG": days_ago, "RGG_Low": round(rgg_low, 2),
            "Above_JMA_Now": "Yes" if above_jma_now else "No",
            "Risk_%": round(risk / entry * 100, 2), "Target_1": round(t1, 2), "T1_Source": t1_src,
            "Reward_R": round((t1 - entry) / risk, 2), "Target_2R": round(entry + 2 * risk, 2),
            "Monthly_H4": round(float(H4), 2) if np.isfinite(H4) else None,
            "Monthly_H3": round(float(mh3[i]), 2) if np.isfinite(mh3[i]) else None,
            "SMA50_Gap_%": round(gap, 2), "Gap_10d_Ago_%": round(gap_then, 2), "Touched": touched,
            "JMA": round(float(jma[i]), 2), "EMA8": round(float(e8[i]), 2), "SMA21": round(float(s21[i]), 2),
            "SMA50": round(float(s50[i]), 2), "SMA150": round(float(s150[i]), 2),
            "Vol_x_20d": round(vx, 2) if np.isfinite(vx) else None, "ATR_%": round(atr / entry * 100, 2),
            "Signal_Date": df.index[i].strftime("%Y-%m-%d"), "Flags": " | ".join(flags)}

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
        r = analyze_stock(sym, daily_map[sym])
        if r: r["Ticker"] = sym; results.append(r)
    except Exception: pass

results.sort(key=lambda x: -x["Score"])
GROUPS = [("Above monthly Camarilla H4", lambda r: r["H4_Status"] != "Below"),
          ("Below monthly H4", lambda r: True)]
for r in results:
    r["Group"] = next(g for g, f in GROUPS if f(r))
fc = FUNNEL
print(f"\n{'=' * 65}\n  SCAN COMPLETE - {len(results)} matches from {got_daily} tickers with data")
print(f"  checked {fc['tickers_checked']} -> SMA50 closing on SMA150 {fc['sma50_closing']} -> above SMA150 {fc['above_sma150']}"
      f" -> RGG (7 sessions) {fc['rgg']} -> from EMA8/SMA21 {fc['from_ema8_sma21']} -> JMA reclaim {fc['jma_reclaim']}"
      f" -> RGG low held {fc['low_held']}"
      f" (above monthly H4 {fc['above_h4']})")
for g, _ in GROUPS:
    names = [r["Ticker"] for r in results if r["Group"] == g]
    print(f"  {g}: {len(names)}  {', '.join(names[:60])}")
print("=" * 65)

ts      = datetime.today().strftime("%Y%m%d_%H%M")
out_dir = os.environ.get("GITHUB_WORKSPACE", os.getcwd())
COLS = ["Ticker","Group","H4_Status","Score","RGG_Date","Days_Since_RGG","Above_JMA_Now","Price","Entry_Price","Stop_Loss",
        "RGG_Low","Risk_%","Target_1","T1_Source","Reward_R",
        "Target_2R","Monthly_H4","Monthly_H3","SMA50_Gap_%","Gap_10d_Ago_%","Touched","JMA","EMA8","SMA21","SMA50","SMA150",
        "Vol_x_20d","ATR_%","Signal_Date","Flags"]
df_out = pd.DataFrame(results) if results else pd.DataFrame(columns=COLS)
df_out = df_out[[c for c in COLS if c in df_out.columns]].reset_index(drop=True)
fpath = os.path.join(out_dir, f"sma50_converge_rgg_jma_{ts}.csv")
try:
    from position_sizing import add_position_size
    df_out = add_position_size(df_out)
    try: __import__("best_picks").rank_best(df_out)
    except Exception as _bp_err: print(f"  Best 10 skipped: {_bp_err}")
except Exception as _ps_err:
    print(f"  Position sizing skipped: {_ps_err}")
df_out.to_csv(fpath, index=False)
print(f"\n  CSV -> {fpath}")
tv = os.path.join(out_dir, f"tv_sma50_converge_rgg_jma_{ts}.txt")
with open(tv, "w") as f:
    for g, _ in GROUPS:
        names = [r["Ticker"] for r in results if r["Group"] == g]
        if names: f.write(f"###{g}\n"); f.writelines(f"NASDAQ:{s}\n" for s in names)
print(f"  TradingView file -> {tv}")

TV_STATUS = "not created"
def _tv_sync():
    global TV_STATUS
    if not results:
        TV_STATUS = "skipped (no matches)"; return
    if not (os.environ.get("TV_SESSIONID") and os.environ.get("TV_SESSIONID_SIGN")):
        TV_STATUS = "skipped (TV_SESSIONID secrets not set)"; return
    from zoneinfo import ZoneInfo
    name = f"{CFG['tv_list_prefix']} - {datetime.now(ZoneInfo('Asia/Kolkata')):%Y-%m-%d}"
    os.environ["TV_WATCHLIST"] = name
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "claude_run", "tools"))
    import tv_watchlist_sync as tvs
    tvs.NAME = name
    s = tvs.session(); lists = tvs.get_lists(s); ex = tvs.exchange_map()
    symbols = []
    for g, _ in GROUPS:
        syms = [tvs.tv_symbol(r["Ticker"], ex) for r in results if r["Group"] == g]
        if syms: symbols.append(f"###{g.split(' - ')[0]} ({len(syms)})"); symbols += syms
    tvs.replace_list(s, lists, symbols)
    after = next((x for x in tvs.get_lists(s) if x.get("name") == name), None)
    got = [x for x in (after or {}).get("symbols") or [] if not str(x).startswith("###")]
    TV_STATUS = f"'{name}' - {len(got)} symbols"
try:
    _tv_sync()
except Exception as e:
    TV_STATUS = f"failed ({type(e).__name__}: {e})"
print(f"[TradingView] {TV_STATUS}")

EMAIL_COLS = ["Ticker","H4_Status","Score","RGG_Date","Days_Since_RGG","Above_JMA_Now","Price","Stop_Loss","Risk_%",
              "Target_1","T1_Source","Reward_R","Monthly_H4","SMA50_Gap_%","Touched","Vol_x_20d"]
def _table(rows):
    th = "".join(f'<th style="background:#1e293b;color:#e2e8f0;padding:6px 8px;font-size:11px;white-space:nowrap">{c}</th>' for c in EMAIL_COLS)
    body = ""
    for i, r in enumerate(rows):
        bg = "#fff" if i % 2 == 0 else "#f0f9ff"
        body += f'<tr style="background:{bg}">' + "".join(
            f'<td style="padding:5px 8px;font-size:12px;{"font-weight:700" if c in ("Ticker", "H4_Status") else ""}">{r.get(c, "-")}</td>'
            for c in EMAIL_COLS) + "</tr>"
    return f'<table style="border-collapse:collapse;width:100%;margin-bottom:14px"><thead><tr>{th}</tr></thead><tbody>{body}</tbody></table>'

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
    try:
        secs = ""
        for g, _ in GROUPS:
            rows = [r for r in rl if r["Group"] == g]
            secs += f'<h3 style="font-size:14px;margin:12px 0 6px">{g} ({len(rows)})</h3>'
            secs += _table(rows[:80]) if rows else '<p style="color:#94a3b8;font-size:12px">none</p>'
        html = f"""<!DOCTYPE html><html><body style="margin:0;background:#f1f5f9;font-family:'Segoe UI',Arial,sans-serif">
<table width="100%" cellpadding="0" cellspacing="0"><tr><td align="center" style="padding:20px 10px">
<table width="100%" style="max-width:960px;background:#fff;border-radius:12px;overflow:hidden">
<tr><td style="background:linear-gradient(135deg,#0f172a,#1e3a5f);padding:20px 26px">
<h1 style="margin:0;color:#60a5fa;font-size:19px">SMA50 Closing on SMA150 + RGG JMA Reclaim</h1>
<p style="margin:6px 0 0;color:#94a3b8;font-size:12px">{datetime.today().strftime('%Y-%m-%d %H:%M UTC')} &middot; {len(rl)} matches
&middot; TradingView: {TV_STATUS}</p></td></tr>
<tr><td style="padding:16px 22px">{secs}
<p style="font-size:11px;color:#64748b">SMA50 rising toward SMA150 from below (gap &le; {CFG['gap_max_pct']:g}%, or just crossed by &le; {-CFG['gap_min_pct']:g}%; narrower than {CFG['slope_bars']} sessions ago) &middot;
close above SMA150 &middot; within the last {CFG['rgg_lookback']} sessions a red, green, green with the second green's volume above the first's &middot;
one of the three lows reached EMA8 or SMA21 &middot; the second green closed above JMA after the red or first green closed at or below it &middot;
no candle since then below the lowest low of the three. Preferred: close above the monthly Camarilla H4
(previous month's close + 1.1 &times; range / 2). Entry = today's close &middot; Stop = lowest low of the RGG candles &middot;
T1 = monthly H4 if above the entry, else the {CFG['high_window']}-session high, else 2R &middot; T2 = 2R. Funnel: checked {FUNNEL['tickers_checked']}
&rarr; SMA50 closing {FUNNEL['sma50_closing']} &rarr; above SMA150 {FUNNEL['above_sma150']} &rarr; RGG {FUNNEL['rgg']} &rarr; from EMA8/SMA21 {FUNNEL['from_ema8_sma21']}
&rarr; JMA reclaim {FUNNEL['jma_reclaim']} &rarr; RGG low held {FUNNEL['low_held']}. Full results (with Quantity for a fixed max loss) attached.</p></td></tr>
<tr><td style="background:#f8fafc;padding:10px;text-align:center;color:#94a3b8;font-size:10px">Not financial advice &middot; Auto-generated by GitHub Actions</td></tr>
</table></td></tr></table></body></html>"""
        plain = [f"SMA50 Closing on SMA150 + RGG JMA Reclaim - {datetime.today().strftime('%Y-%m-%d')}", f"{len(rl)} matches", f"TradingView: {TV_STATUS}", ""]
        for g, _ in GROUPS:
            plain.append(f"{g}: " + ", ".join(r["Ticker"] for r in rl if r["Group"] == g))
        nH = sum(1 for r in rl if r["H4_Status"] != "Below")
        subj = f"SMA50->SMA150 RGG JMA Reclaim - {len(rl)} signals, {nH} above monthly H4 - {datetime.today().strftime('%Y-%m-%d')}"
        msg = MIMEMultipart("mixed"); msg["Subject"] = subj; msg["From"] = gu; msg["To"] = ", ".join(eto)
        alt = MIMEMultipart("alternative"); alt.attach(MIMEText("\n".join(plain), "plain")); alt.attach(MIMEText(html, "html")); msg.attach(alt)
        if csv_path and os.path.exists(csv_path):
            with open(csv_path, "rb") as f:
                part = MIMEBase("application", "octet-stream"); part.set_payload(f.read())
            encoders.encode_base64(part)
            part.add_header("Content-Disposition", f"attachment; filename={os.path.basename(csv_path)}")
            msg.attach(part)
        with smtplib.SMTP_SSL("smtp.gmail.com", 465) as srv:
            srv.login(gu, gp.replace(" ", ""))
            try: __import__("best_picks").attach_to_email(msg)
            except Exception as _bp_err: print(f"  Best 10 not added: {_bp_err}")
            srv.sendmail(gu, eto, msg.as_string())
        print(f"[Email] Sent to {', '.join(eto)} - {subj}")
    except Exception as e:
        print(f"[Email] {type(e).__name__}: {e}")

_send_email(results, fpath)
