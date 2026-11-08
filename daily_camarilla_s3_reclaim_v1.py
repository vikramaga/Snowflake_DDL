# ============================================================
# NASDAQ / NYSE — Monthly Camarilla S3 Reclaim + SMA50/SMA150 Swing (v1)
# ============================================================
#
# The best bullish swing rule found by claude_run/research/cam_research.py
# (12 Camarilla triggers on monthly and weekly H3 / H4 / S3 x 16 filters x
# 51 exits, picked on the first half of 2023-2026 and confirmed on the
# second half and on 1,436 different stocks):
#
#   1. MONTHLY S3 RECLAIM: today's close is back above the monthly Camarilla
#      S3 after the previous close was at or below it.
#      Monthly S3 = previous month's close - 1.1 x (prev high - prev low) / 4
#   2. UPTREND: SMA50 above SMA150
#   3. AT THE SMA50: close within 3% of the SMA50 (above or below)
#   4. VOLUME: today's volume >= 1.5x the 20-day average
#   (price >= $5, 20-day average volume >= 100k)
#
# ORDERS (exactly as tested): buy at the next open; stop = buy - 2 x ATR(14);
# target = buy + 6 x ATR(14) (3R); sell at the open after 40 sessions if
# neither is hit.
#
# BACKTEST (one trade per stock at a time, 0.10% costs, conservative fills):
#   research sample 2023-25  292 trades  42.8% wins  PF 1.58  +0.35R / trade
#   research sample 2025-26  233 trades  41.2% wins  PF 1.58  +0.31R / trade
#   fresh 1,436 stocks       487 trades  40.0% wins  PF 1.32  +0.27R / trade
#     (avg win +13.3%, avg loss -6.7%); same filters without the S3 reclaim:
#     +0.13R; any Camarilla level event: +0.11R.
# Weekly Camarilla levels did not add an edge in the test; they are shown in
# the output as context (overhead weekly H3 / H4).
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


# ── CONFIG
CFG = {
    # history_days vs bars: 400 calendar days -> ~285 business days
    # (pd.bdate_range); the rule needs 150 (SMA150) + 1 previous month.
    "history_days"            : 400,
    "min_bars"                : 170,

    "sma_fast"                : 50,
    "sma_slow"                : 150,
    "near_sma50_pct"          : 3.0,   # |close / SMA50 - 1| <= 3%
    "vol_avg_bars"            : 20,
    "vol_mult"                : 1.5,   # volume >= 1.5x the 20-day average
    "atr_period"              : 14,    # simple mean of the true range
    "stop_atr"                : 2.0,
    "target_r"                : 3.0,
    "max_hold"                : 40,
    "signal_lookback_days"    : 1,     # the latest session (the order is for the next open)

    "min_avg_volume"          : 100_000,
    "min_price"               : 5.0,

    "batch_size"              : 50,
    "batch_sleep"             : 1.5,
}

# ── Indicators
def camarilla(df, rule):
    """H3 / H4 / S3 from the PREVIOUS completed period ('M' month, 'W-FRI' week)."""
    per = df.index.to_period(rule)
    g = df.groupby(per).agg(H=("High", "max"), L=("Low", "min"), C=("Close", "last"))
    p = g.shift(1)
    rng = p.H - p.L
    lv = pd.DataFrame({"H3": p.C + rng * 1.1 / 4, "H4": p.C + rng * 1.1 / 2,
                       "S3": p.C - rng * 1.1 / 4}, index=g.index).reindex(per)
    return {k: lv[k].values.astype(float) for k in lv.columns}

def calc_indicators(df, cfg):
    c, h, l, v = (df[k].values.astype(float) for k in ("Close", "High", "Low", "Volume"))
    pc = np.concatenate([[np.nan], c[:-1]])
    tr = np.nanmax(np.vstack([h - l, np.abs(h - pc), np.abs(l - pc)]), axis=0)
    m, w = camarilla(df, "M"), camarilla(df, "W-FRI")
    return {
        "s50":  pd.Series(c).rolling(cfg["sma_fast"]).mean().values,
        "s150": pd.Series(c).rolling(cfg["sma_slow"]).mean().values,
        "vavg": pd.Series(v).rolling(cfg["vol_avg_bars"]).mean().values,
        "atr":  pd.Series(tr).rolling(cfg["atr_period"]).mean().values,
        "mS3": m["S3"], "mH3": m["H3"], "mH4": m["H4"],
        "wS3": w["S3"], "wH3": w["H3"], "wH4": w["H4"],
    }

def check_pattern_at(df, ind, i, cfg):
    """
    SINGLE SOURCE OF TRUTH. Checks signal day i. Computes EVERY step (no
    short-circuit) so the funnel shows where tickers drop out.
    Returns (passed, details); details is {} only if data is too short.
    """
    if i < cfg["sma_slow"] + 1 or i >= len(df):
        return False, {}
    c, v = df["Close"].values, df["Volume"].values
    s50, s150, vavg, atr, mS3 = (ind[k] for k in ("s50", "s150", "vavg", "atr", "mS3"))
    need = (s50[i], s150[i], vavg[i], atr[i], mS3[i], c[i - 1])
    if not all(np.isfinite(x) for x in need) or atr[i] <= 0:
        return False, {}
    step1 = bool(c[i] > mS3[i] and c[i - 1] <= mS3[i])                       # monthly S3 reclaim
    step2 = bool(s50[i] > s150[i])                                           # SMA50 above SMA150
    dist = (c[i] / s50[i] - 1) * 100
    step3 = bool(abs(dist) <= cfg["near_sma50_pct"])                         # at the SMA50
    vol_x = float(v[i] / vavg[i]) if vavg[i] > 0 else 0.0
    step4 = bool(vol_x >= cfg["vol_mult"])                                   # volume surge
    passed = step1 and step2 and step3 and step4
    return passed, {
        "date": df.index[i], "i": i, "close": float(c[i]), "prev_close": float(c[i - 1]),
        "mS3": float(mS3[i]), "mH3": float(ind["mH3"][i]), "mH4": float(ind["mH4"][i]),
        "wS3": float(ind["wS3"][i]), "wH3": float(ind["wH3"][i]), "wH4": float(ind["wH4"][i]),
        "sma50": float(s50[i]), "sma150": float(s150[i]), "dist_sma50": float(dist),
        "vol_x": vol_x, "atr": float(atr[i]),
        "step1": step1, "step2": step2, "step3": step3, "step4": step4,
    }

# ── Diagnostic funnel (per ticker: did ANY of the lookback days pass
#    steps 1..k cumulatively)
FUNNEL_COUNTS = {
    "tickers_checked": 0,
    "passed_step1_monthly_s3_reclaim": 0,
    "passed_step2_sma50_over_sma150": 0,
    "passed_step3_near_sma50": 0,
    "passed_step4_volume_surge": 0,
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

def analyze_camarilla_s3(sym, df_daily):
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
    entry = sig["close"]                      # estimate; the order is for the next open
    stop = entry - CFG["stop_atr"] * sig["atr"]
    target = entry + CFG["target_r"] * CFG["stop_atr"] * sig["atr"]
    if stop <= 0: return None
    risk_pct = (entry - stop) / entry * 100
    over = [(n_, lv) for n_, lv in (("weekly H3", sig["wH3"]), ("weekly H4", sig["wH4"]),
                                     ("monthly H3", sig["mH3"]), ("monthly H4", sig["mH4"]))
            if np.isfinite(lv) and entry < lv < target]
    flags = [f"vol {sig['vol_x']:.1f}x", f"{sig['dist_sma50']:+.1f}% vs SMA50"]
    if over:
        n_, lv = min(over, key=lambda x: x[1])
        flags.append(f"first level overhead: {n_} {lv:.2f} ({(lv / entry - 1) * 100:+.1f}%)")

    score = 40
    score += min(20, (sig["vol_x"] - 1.5) * 10)                    # stronger volume
    score += max(0, 15 - abs(sig["dist_sma50"]) * 5)               # closer to SMA50
    score += min(15, max(0, (sig["sma50"] / sig["sma150"] - 1) * 150))  # trend strength
    score += max(0, 10 - max(0, risk_pct - 4) * 2)                 # tighter stop
    score = round(min(100, max(0, score)))

    return {
        "Score": score, "Price": round(price, 2),
        "Entry_Price": round(entry, 2), "Stop_Loss": round(stop, 2), "Risk_%": round(risk_pct, 1),
        "Target_3R": round(target, 2), "Target_%": round((target / entry - 1) * 100, 1),
        "Max_Hold_Days": CFG["max_hold"],
        "Monthly_S3": round(sig["mS3"], 2), "Prev_Close": round(sig["prev_close"], 2),
        "Monthly_H3": round(sig["mH3"], 2), "Monthly_H4": round(sig["mH4"], 2),
        "Weekly_S3": round(sig["wS3"], 2), "Weekly_H3": round(sig["wH3"], 2), "Weekly_H4": round(sig["wH4"], 2),
        "SMA50": round(sig["sma50"], 2), "SMA150": round(sig["sma150"], 2),
        "Dist_SMA50_%": round(sig["dist_sma50"], 1), "Volume_x": round(sig["vol_x"], 2),
        "ATR14": round(sig["atr"], 2),
        "Signal_Date": sig["date"].strftime("%Y-%m-%d"),
        "Days_Since_Signal": int(len(df_daily) - 1 - sig["i"]),
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

# ── Main scan
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
        r = analyze_camarilla_s3(sym, daily_map[sym])
        if r is None: continue
        r["Ticker"] = sym
        results.append(r)
        print(f"  ✅ {sym:<6} ${r['Price']:.2f}  Score {r['Score']}  S3 {r['Monthly_S3']}  "
              f"vol {r['Volume_x']}x  {r['Dist_SMA50_%']:+}% vs SMA50  {r['Signal_Date']}")
    except Exception: pass

fc = FUNNEL_COUNTS
print(f"\n{'━'*65}\n  SCAN COMPLETE — {len(results)} matches from {got_daily} tickers with data")
print(f"  🔍 FUNNEL (the last {CFG['signal_lookback_days']} session)")
print(f"  Tickers checked                       : {fc['tickers_checked']}")
print(f"  Step 1 — closed back above monthly S3  : {fc['passed_step1_monthly_s3_reclaim']}")
print(f"  Step 2 — + SMA50 above SMA150          : {fc['passed_step2_sma50_over_sma150']}")
print(f"  Step 3 — + within {CFG['near_sma50_pct']:.0f}% of SMA50          : {fc['passed_step3_near_sma50']}")
print(f"  Step 4 — + volume >= {CFG['vol_mult']}x 20-day avg    : {fc['passed_step4_volume_surge']}  (= full rule)")
print(f"{'━'*65}")

results.sort(key=lambda x: x["Score"], reverse=True)

# ── Save CSV + TradingView list
ts      = datetime.today().strftime("%Y%m%d_%H%M")
out_dir = os.environ.get("GITHUB_WORKSPACE", os.getcwd())
COLS = ["Ticker","Price","Score","Entry_Price","Stop_Loss","Risk_%","Target_3R","Target_%","Max_Hold_Days",
        "Monthly_S3","Prev_Close","Monthly_H3","Monthly_H4","Weekly_S3","Weekly_H3","Weekly_H4",
        "SMA50","SMA150","Dist_SMA50_%","Volume_x","ATR14","Signal_Date","Days_Since_Signal","Flags"]
df_out = pd.DataFrame([{k: v for k, v in r.items() if not k.startswith("_")}
                       for r in results]) if results else pd.DataFrame(columns=COLS)
df_out = df_out[[c for c in COLS if c in df_out.columns]].reset_index(drop=True)

fpath = os.path.join(out_dir, f"daily_camarilla_s3_reclaim_{ts}.csv")
try:
    from position_sizing import add_position_size
    df_out = add_position_size(df_out)
    try: __import__("best_picks").rank_best(df_out)
    except Exception as _bp_err: print(f"  ⚠️  Best 10 skipped: {_bp_err}")
except Exception as _ps_err:
    print(f"  ⚠️  Position sizing skipped: {_ps_err}")
df_out.to_csv(fpath, index=False)
print(f"\n  💾 CSV → {fpath}")
tv = os.path.join(out_dir, f"tv_daily_camarilla_s3_reclaim_{ts}.txt")
with open(tv, "w") as f:
    f.write(f"###Monthly Camarilla S3 Reclaim {datetime.today().strftime('%Y-%m-%d')}\n")
    for r in results: f.write(f"NASDAQ:{r['Ticker']}\n")
print(f"  📋 TradingView → {tv}")
if results:
    print(f"\n  📋 Tickers: {', '.join(r['Ticker'] for r in results)}")

# ── Email with CSV attached
EMAIL_COLS = ["Ticker","Price","Score","Entry_Price","Stop_Loss","Risk_%","Target_3R",
              "Monthly_S3","Weekly_H3","Weekly_H4","Dist_SMA50_%","Volume_x","Signal_Date"]

def _fmt_cell(col, v):
    try:
        if col in ("Price","Entry_Price","Stop_Loss","Target_3R","Monthly_S3","Weekly_H3","Weekly_H4"):
            return f"${float(v):.2f}"
        if col == "Risk_%": return f"{float(v):.1f}%"
        if col == "Dist_SMA50_%": return f"{float(v):+.1f}%"
        if col == "Volume_x": return f"{float(v):.1f}x"
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
                if c == "Target_3R": sty = "color:#2563eb;font-weight:600"
                tds += f'<td style="padding:6px 10px;font-size:12px;{sty}">{_fmt_cell(c, r.get(c))}</td>'
            rows += f'<tr style="background:{bg}">{tds}</tr>'
        if not rows:
            rows = (f'<tr><td colspan="{len(EMAIL_COLS)}" style="padding:20px;text-align:center;'
                    f'color:#94a3b8">No matches today</td></tr>')
        tick_html = (f'<p style="margin:10px 0 0;font-size:12px;font-family:monospace">'
                     f'{", ".join(r["Ticker"] for r in rl)}</p>') if rl else ""
        html = f"""<!DOCTYPE html><html><body style="margin:0;background:#f1f5f9;font-family:'Segoe UI',Arial,sans-serif">
<table width="100%" cellpadding="0" cellspacing="0"><tr><td align="center" style="padding:20px 10px">
<table width="100%" style="max-width:880px;background:#fff;border-radius:12px;overflow:hidden">
<tr><td style="background:linear-gradient(135deg,#0f172a,#1e3a5f);padding:20px 26px">
<h1 style="margin:0;color:#60a5fa;font-size:19px">📊 Monthly Camarilla S3 Reclaim — SMA50/150 Swing</h1>
<p style="margin:6px 0 0;color:#94a3b8;font-size:12px">{datetime.today().strftime('%Y-%m-%d %H:%M UTC')} · {cnt} match{'es' if cnt != 1 else ''}
 — close back above the monthly S3, SMA50 above SMA150, within {CFG['near_sma50_pct']:.0f}% of SMA50, volume ≥ {CFG['vol_mult']}x average.
 Backtest on fresh stocks: 40% wins, avg win +13%, avg loss -7%, +0.27R per trade.</p>
</td></tr>
<tr><td style="padding:12px 26px;background:#0b1220"><p style="margin:0;color:#cbd5e1;font-size:12px">
🔍 FUNNEL: {fc['tickers_checked']} checked → {fc['passed_step1_monthly_s3_reclaim']} reclaimed monthly S3
→ {fc['passed_step2_sma50_over_sma150']} + SMA50 &gt; SMA150 → {fc['passed_step3_near_sma50']} + at SMA50
→ <b style="color:#facc15">{fc['passed_step4_volume_surge']} + volume surge = full rule</b></p></td></tr>
<tr><td style="padding:16px"><div style="overflow-x:auto;border:1px solid #e2e8f0;border-radius:8px">
<table style="border-collapse:collapse;width:100%"><thead><tr>{th}</tr></thead><tbody>{rows}</tbody></table></div>
{tick_html}
<p style="font-size:11px;color:#64748b;margin:10px 0 0">Orders as tested: buy at the next open; stop = buy − {CFG['stop_atr']:.0f} × ATR(14);
target = buy + {CFG['target_r'] * CFG['stop_atr']:.0f} × ATR(14) (3R); sell after {CFG['max_hold']} sessions if neither is hit.
Entry/stop/target here use the signal close — reset them from your actual fill. Weekly H3/H4 are shown as overhead context only
(they did not improve results in the test). 📎 Full results (with Quantity for a fixed max loss) attached as CSV.</p>
</td></tr>
<tr><td style="background:#f8fafc;padding:10px;text-align:center;color:#94a3b8;font-size:10px">
⚠️ Not financial advice · Auto-generated by GitHub Actions</td></tr>
</table></td></tr></table></body></html>"""

        plain = [f"Monthly Camarilla S3 Reclaim — SMA50/150 Swing — {datetime.today().strftime('%Y-%m-%d')}",
                 f"{cnt} matches",
                 f"FUNNEL: {fc['tickers_checked']} checked -> {fc['passed_step1_monthly_s3_reclaim']} S3 reclaim -> "
                 f"{fc['passed_step2_sma50_over_sma150']} SMA50>SMA150 -> {fc['passed_step3_near_sma50']} at SMA50 -> "
                 f"{fc['passed_step4_volume_surge']} volume surge (=full rule)", "="*60]
        for r in rl[:50]:
            plain.append(f"{r['Ticker']:<6} ${r['Price']:.2f}  Score:{r['Score']}  Entry~${r['Entry_Price']:.2f}  "
                         f"SL:${r['Stop_Loss']:.2f} ({r['Risk_%']}%)  T(3R):${r['Target_3R']:.2f}  "
                         f"S3 ${r['Monthly_S3']:.2f}  vol {r['Volume_x']}x  {r['Signal_Date']}")
        if rl:
            plain += ["", "Tickers: " + ", ".join(r["Ticker"] for r in rl)]
        else:
            plain.append("No matches today")
        plain.append(f"\nOrders: buy next open, stop -{CFG['stop_atr']:.0f} ATR, target +{CFG['target_r'] * CFG['stop_atr']:.0f} ATR, "
                     f"max {CFG['max_hold']} sessions. Full results in CSV attachment.")

        subj = (f"📊 Camarilla S3 Reclaim — "
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

# ── Charts for top 5: last 70 daily candles, SMA50/150, monthly S3, weekly H3/H4
if results:
    top = results[:5]
    fig, axes = plt.subplots(len(top), 1, figsize=(14, 4.4*len(top)), facecolor="#0f172a")
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
        for key, c_, lab, lw, ls in (("s50", "#86efac", "SMA50", 1.4, "-"), ("s150", "#60a5fa", "SMA150", 1.1, "-"),
                                     ("mS3", "#f59e0b", "Monthly S3", 1.2, "-"), ("wH3", "#c084fc", "Weekly H3", 0.9, "--"),
                                     ("wH4", "#e879f9", "Weekly H4", 0.9, "--")):
            ax.plot(x, ind[key][lo:], color=c_, lw=lw, ls=ls, label=lab, drawstyle="steps-mid" if key[0] in "mw" else "default")
        for lvl, c_, lab in ((r["Stop_Loss"], "#ef4444", "Stop"), (r["Target_3R"], "#38bdf8", "3R")):
            ax.axhline(lvl, color=c_, ls=":", lw=0.8)
            ax.text(len(seg) - 0.5, lvl, f" {lab} {lvl:.2f}", color=c_, fontsize=7, va="center")
        ax.axvline(k - lo, color="#facc15", ls="--", lw=0.8)
        ax.legend(loc="upper left", facecolor="#1e293b", labelcolor="#e2e8f0", fontsize=7, ncol=5)
        step = max(1, len(seg) // 8)
        ax.set_xticks(x[::step]); ax.set_xticklabels([d.strftime("%m/%d") for d in seg.index[::step]], fontsize=7)
        ax.set_title(f"{r['Ticker']} | USD {r['Price']:.2f} | Score {r['Score']} | monthly S3 {r['Monthly_S3']:.2f} | "
                     f"vol {r['Volume_x']}x | {r['Dist_SMA50_%']:+}% vs SMA50", color="#e2e8f0", fontsize=9, fontweight="bold")
        ax.tick_params(colors="#94a3b8", labelsize=7)
        for sp in ax.spines.values(): sp.set_edgecolor("#1e3a5f")
        ax.grid(color="#1e3a5f", ls="--", lw=0.5, alpha=0.6)
    plt.suptitle(f"Monthly Camarilla S3 Reclaim — SMA50/150 Swing · "
                 f"{datetime.today().strftime('%Y-%m-%d')}", color="#60a5fa", fontsize=12, fontweight="bold")
    plt.tight_layout()
    cp = os.path.join(out_dir, f"daily_camarilla_s3_reclaim_chart_{ts}.png")
    plt.savefig(cp, dpi=130, bbox_inches="tight", facecolor="#0f172a")
    plt.close()
    print(f"  📊 Chart → {cp}")

print(f"""
{'━'*55}
  📋 RULE (signal = the latest session's close)
  1) Close back above the monthly Camarilla S3 (previous close at or below it)
     S3 = previous month's close - 1.1 x (previous month's high - low) / 4
  2) SMA{CFG['sma_fast']} above SMA{CFG['sma_slow']}
  3) Close within {CFG['near_sma50_pct']:.0f}% of SMA{CFG['sma_fast']}
  4) Volume >= {CFG['vol_mult']}x the {CFG['vol_avg_bars']}-day average
  ORDERS: buy next open, stop = buy - {CFG['stop_atr']:.0f} x ATR{CFG['atr_period']},
          target = buy + {CFG['target_r'] * CFG['stop_atr']:.0f} x ATR{CFG['atr_period']} (3R), sell after {CFG['max_hold']} sessions
{'━'*55}
""")
