# ============================================================
# NASDAQ — Early Stage-2 Breakout Scanner (v1)
# ============================================================
#
# Finds stocks in the EARLY phase of a potential large advance,
# based on the structural template common to big winners.
#
# SINGLE PASS — pure price/volume, no .info calls (fast).
#
#  F1  52w gain inside band          -> mid-move, not extended
#  F2  Price within X% of 52w high
#  F3  SMA50 > SMA150 > SMA200, all rising
#  F4  >= N held SMA50 tests in last 6 months
#  F5  Volume breakout in last 20 sessions (>= 2.5x avg50, close up)
#  F6  RS line vs SPY at / near 6-month high
#  S1  SOFT: ATR% contraction (recent 20d < prior 20d)
#
#  SCORE = RS(0-30) + Support(0-24) + Proximity(0-14)
#        + Volume(0-12) + Stack(0-10) + Slope(0-6) + Contract(0-4)
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
    "history_days"            : 550,
    "benchmark"               : "SPY",

    # ── F1 52w gain band ───────────────────────────────────────
    "min_52w_gain_pct"        : 30.0,
    "max_52w_gain_pct"        : 60.0,

    # ── F2 proximity to 52w high ───────────────────────────────
    "max_off_52w_high_pct"    : 15.0,

    # ── F3 MA stack ────────────────────────────────────────────
    "slope_lookback"          : 20,

    # ── F4 SMA50 support history ───────────────────────────────
    "min_sma50_touches"       : 3,
    "touch_band_pct"          : 3.0,
    "touch_confirm_days"      : 5,
    "touch_debounce_days"     : 10,
    "touch_lookback_months"   : 6,

    # ── F5 volume breakout ─────────────────────────────────────
    "vol_mult"                : 2.5,
    "breakout_window"         : 20,

    # ── F6 relative strength ───────────────────────────────────
    "rs_high_tol_pct"         : 2.0,
    "rs_lookback_months"      : 6,

    # ── Gates ──────────────────────────────────────────────────
    "min_hard_filters"        : 5,      # out of 6  (1 near-miss allowed)
    "min_total_score"         : 55,     # out of 100

    # ── Liquidity ──────────────────────────────────────────────
    "min_price"               : 5.0,
    "min_avg_dollar_vol"      : 5_000_000,

    "batch_size"              : 100,
    "batch_sleep"             : 1.0,
    "max_email_rows"          : 50,
    "chart_top_n"             : 5,
}

# ── Indicators ───────────────────────────────────────────────
def sma(close, n):
    return close.rolling(n, min_periods=n).mean()

def atr_pct(df, period=14):
    high  = df["High"].values.astype(float)
    low   = df["Low"].values.astype(float)
    close = df["Close"].values.astype(float)
    pc    = np.roll(close, 1); pc[0] = close[0]
    tr = np.maximum(high - low,
                    np.maximum(np.abs(high - pc), np.abs(low - pc)))
    a = pd.Series(tr, index=df.index).rolling(period, min_periods=period).mean()
    return (a / df["Close"]) * 100.0

def slope_pct(series, lookback):
    """Return (is_rising, % change implied by the fitted slope)."""
    s = series.dropna()
    if len(s) < lookback + 1: return False, 0.0
    y = s.values[-lookback:].astype(float)
    x = np.arange(lookback, dtype=float)
    den = ((x - x.mean())**2).sum()
    if den == 0: return False, 0.0
    m = (((x - x.mean()) * (y - y.mean())).sum()) / den
    pct = (m * lookback) / y[0] * 100.0 if y[0] != 0 else 0.0
    return bool(m > 0), float(pct)

def count_sma50_touches(df, s50):
    """Pullbacks that tagged SMA50 and reclaimed it without breaking down."""
    n = min(len(df), CFG["touch_lookback_months"] * 21)
    if n < 40: return 0
    lows   = df["Low"].values[-n:].astype(float)
    closes = df["Close"].values[-n:].astype(float)
    ma     = s50.values[-n:].astype(float)
    band   = 1.0 + CFG["touch_band_pct"]/100.0
    conf   = CFG["touch_confirm_days"]
    touches, i = 0, 1
    while i < n - 1:
        if np.isnan(ma[i]) or np.isnan(ma[i-1]):
            i += 1; continue
        near   = lows[i] <= ma[i] * band
        before = closes[i-1] > ma[i-1]
        if near and before:
            e = min(i + conf + 1, n)
            reclaimed = bool(np.any(closes[i:e] > ma[i:e]))
            broke     = bool(np.all(closes[i:e] < ma[i:e] * 0.97))
            if reclaimed and not broke:
                touches += 1
                i += CFG["touch_debounce_days"]
                continue
        i += 1
    return int(touches)

def vol_breakout(df):
    if len(df) < 60: return False, 0.0, None
    vol   = df["Volume"].astype(float)
    avg50 = vol.rolling(50, min_periods=50).mean()
    ratio = vol / avg50
    up    = df["Close"].astype(float) > df["Close"].astype(float).shift(1)
    mask  = (ratio >= CFG["vol_mult"]) & up
    tail  = mask.iloc[-CFG["breakout_window"]:]
    if not tail.any():
        mx = ratio.iloc[-CFG["breakout_window"]:].max()
        return False, float(mx if np.isfinite(mx) else 0.0), None
    idx = tail[tail].index[-1]
    return True, float(ratio.loc[idx]), idx

def rs_vs_bench(close, bench_close):
    """RS line = stock/benchmark. Return (at_high, pct_of_6m_high, rs_series)."""
    j = pd.concat([close, bench_close], axis=1, join="inner").dropna()
    m = CFG["rs_lookback_months"] * 21
    if len(j) < m: return False, 0.0, None
    rs  = j.iloc[:,0] / j.iloc[:,1]
    win = rs.iloc[-m:]
    hi  = float(win.max())
    if hi <= 0: return False, 0.0, rs
    pct = float(rs.iloc[-1]) / hi
    return bool(pct >= 1.0 - CFG["rs_high_tol_pct"]/100.0), pct, rs

def atr_contraction(df):
    a = atr_pct(df).dropna()
    if len(a) < 45: return False, 0.0
    recent = float(a.iloc[-20:].mean()); prior = float(a.iloc[-40:-20].mean())
    if prior <= 0: return False, 0.0
    return bool(recent < prior), float(recent / prior)

# ── Download ──────────────────────────────────────────────────
def _clean(df, min_bars=250):
    if df is None or df.empty: return None
    need = [c for c in ["Open","High","Low","Close","Volume"] if c in df.columns]
    if not all(c in need for c in ["High","Low","Close","Volume"]): return None
    df = df[need].copy()
    df.index = pd.to_datetime(df.index)
    if hasattr(df.index,"tz") and df.index.tz:
        df.index = df.index.tz_localize(None)
    df.dropna(subset=["Close","Volume"], inplace=True)
    return df if (len(df) >= min_bars and float(df["Close"].iloc[-1]) > 0) else None

def download(symbols, days, min_bars=250):
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
                        df = raw.xs(sym,axis=1,level=1) if l0 & pf else raw[sym]
                        df = _clean(df, min_bars)
                        if df is not None: out[sym] = df
                    except Exception: pass
            elif len(symbols) == 1:
                df = _clean(raw, min_bars)
                if df is not None: out[symbols[0]] = df
    except Exception: pass
    for sym in [s for s in symbols if s not in out]:
        for _ in range(2):
            try:
                df = yf.Ticker(sym).history(
                    start=start.strftime("%Y-%m-%d"),
                    end=end.strftime("%Y-%m-%d"),
                    auto_adjust=True, actions=False)
                df = _clean(df, min_bars)
                if df is not None: out[sym] = df; break
            except Exception: time.sleep(0.2)
        time.sleep(0.04)
    return out

# ── Core evaluation ───────────────────────────────────────────
def analyze(sym, df, bench_close):
    close = df["Close"].astype(float)
    price = float(close.iloc[-1])
    if price < CFG["min_price"]: return None

    adv = float((close * df["Volume"].astype(float)).rolling(20).mean().iloc[-1])
    if not np.isfinite(adv) or adv < CFG["min_avg_dollar_vol"]: return None

    s50, s150, s200 = sma(close,50), sma(close,150), sma(close,200)
    if np.isnan(s200.iloc[-1]): return None

    # F1 — 52w gain band
    win  = close.iloc[-252:] if len(close) >= 252 else close
    gain = (price / float(win.iloc[0]) - 1.0) * 100.0
    f1   = CFG["min_52w_gain_pct"] <= gain <= CFG["max_52w_gain_pct"]

    # F2 — proximity to 52w high
    hi52 = float(win.max())
    off  = (hi52 - price) / hi52 * 100.0 if hi52 > 0 else 100.0
    f2   = off <= CFG["max_off_52w_high_pct"]

    # F3 — MA stack + rising
    stacked = price > float(s50.iloc[-1]) > float(s150.iloc[-1]) > float(s200.iloc[-1])
    up50,  sl50  = slope_pct(s50,  CFG["slope_lookback"])
    up150, sl150 = slope_pct(s150, CFG["slope_lookback"])
    up200, sl200 = slope_pct(s200, CFG["slope_lookback"])
    f3 = bool(stacked and up50 and up150 and up200)

    # F4 — SMA50 support history
    touches = count_sma50_touches(df, s50)
    f4 = touches >= CFG["min_sma50_touches"]

    # F5 — volume breakout
    f5, volx, bo_dt = vol_breakout(df)

    # F6 — relative strength
    f6, rs_pct, rs_series = rs_vs_bench(close, bench_close)

    # S1 — soft contraction
    s1, contract = atr_contraction(df)

    hard   = [f1, f2, f3, f4, f5, f6]
    passed = int(sum(hard))

    score  = 0.0
    score += 30.0 * rs_pct                                                   # RS
    score += 24.0 * min(touches / 5.0, 1.0)                                  # support
    score += 14.0 * (1.0 - min(off / CFG["max_off_52w_high_pct"], 1.0))      # proximity
    score += 12.0 * min(volx / CFG["vol_mult"], 1.0)                         # volume
    score += 10.0 * (1.0 if f3 else 0.0)                                     # stack
    score +=  6.0 * min(max(sl50, 0.0) / 5.0, 1.0)                           # slope
    score +=  4.0 * (1.0 if s1 else 0.0)                                     # contraction

    return {
        "Ticker"      : sym,
        "Price"       : round(price, 2),
        "Gain52W_%"   : round(gain, 1),
        "OffHigh_%"   : round(off, 1),
        "Touches"     : touches,
        "VolX"        : round(volx, 2),
        "BO_Date"     : bo_dt.strftime("%Y-%m-%d") if bo_dt is not None else "—",
        "RS_%"        : round(rs_pct * 100.0, 1),
        "SMA50"       : round(float(s50.iloc[-1]), 2),
        "SMA150"      : round(float(s150.iloc[-1]), 2),
        "SMA200"      : round(float(s200.iloc[-1]), 2),
        "SMA50_Slope" : round(sl50, 2),
        "SMA200_Slope": round(sl200, 2),
        "ATR_Contract": round(contract, 2),
        "ADV_$M"      : round(adv / 1e6, 1),
        "F1_Gain"     : int(f1), "F2_NearHigh" : int(f2),
        "F3_Stack"    : int(f3), "F4_Support"  : int(f4),
        "F5_Volume"   : int(f5), "F6_RS"       : int(f6),
        "S1_Contract" : int(s1),
        "Passed"      : passed,
        "Total"       : round(score, 1),
        "Missing"     : ",".join([n for n,v in
                        [("Gain",f1),("NearHigh",f2),("Stack",f3),
                         ("Support",f4),("Volume",f5),("RS",f6)] if not v]) or "—",
        "_df"    : df, "_s50": s50, "_s150": s150, "_s200": s200,
        "_rs"    : rs_series,
    }

# ── Tickers ───────────────────────────────────────────────────
print("━"*65)
print("  STEP 1  FETCH TICKERS")
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
        "ALGN","AMGN","CTAS","DOCU","EA","GILD","INTU","MCHP","MNST","SNDK",
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

# ── Benchmark ─────────────────────────────────────────────────
print("━"*65)
print("  STEP 2  BENCHMARK")
print("━"*65)
_bench_map  = download([CFG["benchmark"]], CFG["history_days"])
_bench_df   = _bench_map.get(CFG["benchmark"])
BENCH_CLOSE = None
if _bench_df is not None and len(_bench_df) >= 130:
    BENCH_CLOSE = _bench_df["Close"].astype(float)
    print(f"  ✅ {CFG['benchmark']}: {len(BENCH_CLOSE)} bars")
else:
    print(f"  ⚠️  {CFG['benchmark']} unavailable — RS filter will be skipped (F6=0)")
print()

# ── Live print ────────────────────────────────────────────────
LIVE_COLS = ["Ticker","Price","Total","Passed","Gain52W_%","OffHigh_%",
             "Touches","VolX","RS_%","Missing"]
_CW = {"Ticker":8,"Price":10,"Total":8,"Passed":8,"Gain52W_%":11,
       "OffHigh_%":11,"Touches":9,"VolX":7,"RS_%":7,"Missing":18}
_CF = {"Price":"${:.2f}","Total":"{:.1f}","Gain52W_%":"{:+.1f}%",
       "OffHigh_%":"{:.1f}%","VolX":"{:.2f}x","RS_%":"{:.1f}%"}
_hdr_done = False

def _live_header():
    global _hdr_done
    if _hdr_done: return
    print("\n" + "━"*112)
    print("  📊  LIVE MATCHES  —  printed the moment a stock passes the gates")
    print("━"*112)
    print("".join(f"  {c:<{_CW.get(c,12)}}" for c in LIVE_COLS))
    print("  " + "─"*110)
    _hdr_done = True

def live_print(r):
    _live_header()
    row = ""
    for c in LIVE_COLS:
        val = r.get(c,"—"); w = _CW.get(c,12)
        if c in _CF and isinstance(val,(int,float)):
            try: val = _CF[c].format(float(val))
            except Exception: val = str(val)
        if c == "Passed": val = f"{int(r.get('Passed',0))}/6"
        row += f"  {str(val):<{w}}"
    print(row)

# ── Main scan ─────────────────────────────────────────────────
print("━"*65)
print("  STEP 3  SCAN")
print("━"*65)

results, errors, scanned = [], 0, 0
bs = CFG["batch_size"]
for i in tqdm(range(0, len(TICKERS), bs), desc="Batches", ncols=88):
    batch = TICKERS[i:i+bs]
    data_map = download(batch, CFG["history_days"])
    for sym in batch:
        scanned += 1
        df = data_map.get(sym)
        if df is None: continue
        try:
            if BENCH_CLOSE is None: continue
            r = analyze(sym, df, BENCH_CLOSE)
            if r is None: continue
            if r["Passed"] >= CFG["min_hard_filters"] and r["Total"] >= CFG["min_total_score"]:
                results.append(r); live_print(r)
        except Exception:
            errors += 1
    time.sleep(CFG["batch_sleep"])

results.sort(key=lambda x: (x["Passed"], x["Total"]), reverse=True)

print()
print("━"*65)
print(f"  Scanned {scanned}  |  Matches {len(results)}  |  Errors {errors}")
print("━"*65)

# ── Outputs ───────────────────────────────────────────────────
ts      = datetime.today().strftime("%Y%m%d_%H%M")
out_dir = os.environ.get("GITHUB_WORKSPACE", os.getcwd())

CSV_COLS = ["Ticker","Price","Total","Passed","Gain52W_%","OffHigh_%","Touches",
            "VolX","BO_Date","RS_%","SMA50","SMA150","SMA200","SMA50_Slope",
            "SMA200_Slope","ATR_Contract","ADV_$M","F1_Gain","F2_NearHigh",
            "F3_Stack","F4_Support","F5_Volume","F6_RS","S1_Contract","Missing"]

df_out = pd.DataFrame([{k:r.get(k) for k in CSV_COLS} for r in results]) \
         if results else pd.DataFrame(columns=CSV_COLS)

fpath = os.path.join(out_dir, f"early_stage2_{ts}.csv")
df_out.to_csv(fpath, index=False)
print(f"\n  💾 CSV → {fpath}")

tv = os.path.join(out_dir, f"tv_early_stage2_{ts}.txt")
with open(tv,"w") as f:
    f.write(f"###Early Stage-2 {datetime.today().strftime('%Y-%m-%d')}\n")
    for r in results: f.write(f"NASDAQ:{r['Ticker']}\n")
print(f"  📋 TradingView → {tv}")

oneline = os.path.join(out_dir, f"early_stage2_oneline_{ts}.txt")
_one = ",".join(r["Ticker"] for r in results)
_tvl = ",".join(f"NASDAQ:{r['Ticker']}" for r in results)
with open(oneline,"w") as f:
    f.write(_one + "\n" + _tvl + "\n")
print(f"  📋 One-line → {oneline}")
if _one: print(f"\n  {_one}")

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
        print("         → Repo → Settings → Secrets → Actions → GMAIL_USER")
        return
    if not gp:
        print("[Email] ❌  GMAIL_PASS secret is empty")
        print("         → Must be a Gmail App Password (16 chars, no spaces)")
        print("         → Get one at: myaccount.google.com/apppasswords")
        return
    if not et:
        print("[Email] ❌  EMAIL_TO secret is empty")
        print("         → Repo → Settings → Secrets → Actions → EMAIL_TO")
        return

    eto = [e.strip() for e in et.split(",") if e.strip()]
    cnt = len(rl)

    try:
        print(f"[Email] Sending to {et}  ({cnt} results)...")

        th_e = "".join(
            f'<th style="background:#1e293b;color:#e2e8f0;padding:8px 11px;'
            f'font-size:11px;font-weight:700;border-bottom:2px solid #3b82f6;'
            f'white-space:nowrap">{c}</th>'
            for c in ["Ticker","Price","Total","Pass","Gain52W","OffHigh",
                      "SMA50 Holds","VolX","RS"]
        )
        rows_e = ""
        for i, r in enumerate(rl[:CFG["max_email_rows"]]):
            bg     = "#fff" if i % 2 == 0 else "#f0f9ff"
            ticker = r.get("Ticker","—")
            price  = r.get("Price",0)     or 0
            total  = r.get("Total",0)     or 0
            passed = r.get("Passed",0)    or 0
            gain   = r.get("Gain52W_%",0) or 0
            off    = r.get("OffHigh_%",0) or 0
            touch  = r.get("Touches",0)   or 0
            volx   = r.get("VolX",0)      or 0
            rs     = r.get("RS_%",0)      or 0
            tot_bg = "#166534" if float(total) >= 70 else "#1e3a5f"
            rows_e += (
                f'<tr style="background:{bg}">'
                f'<td style="padding:6px 11px;font-size:12px;font-weight:700">{ticker}</td>'
                f'<td style="padding:6px 11px;font-size:12px">${float(price):.2f}</td>'
                f'<td style="padding:6px 11px;font-size:12px;font-weight:700;'
                f'background:{tot_bg};color:#fff;text-align:center">{float(total):.0f}</td>'
                f'<td style="padding:6px 11px;font-size:12px;text-align:center;'
                f'color:{"#22c55e" if int(passed)>=6 else "#64748b"};'
                f'font-weight:700">{int(passed)}/6</td>'
                f'<td style="padding:6px 11px;font-size:12px">{float(gain):+.1f}%</td>'
                f'<td style="padding:6px 11px;font-size:12px">-{float(off):.1f}%</td>'
                f'<td style="padding:6px 11px;font-size:12px;text-align:center">{int(touch)}</td>'
                f'<td style="padding:6px 11px;font-size:12px">{float(volx):.2f}x</td>'
                f'<td style="padding:6px 11px;font-size:12px;color:'
                f'{"#22c55e" if float(rs)>=98 else "#334155"}">{float(rs):.1f}%</td>'
                f'</tr>'
            )

        no_results_msg = ""
        if cnt == 0:
            no_results_msg = (
                '<tr><td colspan="9" style="padding:20px;text-align:center;'
                'color:#64748b;font-size:13px">No matches found today — '
                'market conditions did not trigger the pattern</td></tr>'
            )

        one_line = ",".join(r.get("Ticker","") for r in rl) or "—"
        tv_line  = ",".join(f"NASDAQ:{r.get('Ticker','')}" for r in rl) or "—"

        html_e = f"""<!DOCTYPE html><html><body style="margin:0;padding:0;
background:#f1f5f9;font-family:'Segoe UI',Arial,sans-serif">
<table width="100%" cellpadding="0" cellspacing="0" style="padding:20px 0">
<tr><td>
<table width="100%" cellpadding="0" cellspacing="0"
   style="max-width:900px;margin:0 auto;background:#fff;
          border-radius:12px;overflow:hidden;
          box-shadow:0 4px 20px rgba(0,0,0,0.08)">
  <tr><td style="background:linear-gradient(135deg,#0f172a,#1e3a5f);padding:22px 28px">
<h1 style="margin:0;color:#60a5fa;font-size:20px;font-weight:700">
  🚀 Early Stage-2 Breakout Candidates
</h1>
<p style="margin:6px 0 0;color:#94a3b8;font-size:12px">
  {datetime.today().strftime('%Y-%m-%d %H:%M UTC')} &nbsp;·&nbsp;
  {cnt} match{'es' if cnt!=1 else ''} found
</p>
  </td></tr>
  <tr><td style="padding:16px">
<div style="overflow-x:auto;border-radius:8px;border:1px solid #e2e8f0">
  <table style="border-collapse:collapse;width:100%;min-width:640px">
    <thead><tr>{th_e}</tr></thead>
    <tbody>{rows_e or no_results_msg}</tbody>
  </table>
</div>
<p style="font-size:11px;color:#64748b;margin:12px 0 4px;font-weight:700">
  Single-line CSV
</p>
<p style="font-family:monospace;font-size:11px;color:#334155;
          background:#f8fafc;padding:8px;border-radius:6px;
          word-break:break-all;margin:0">{one_line}</p>
<p style="font-size:11px;color:#64748b;margin:12px 0 4px;font-weight:700">
  TradingView import
</p>
<p style="font-family:monospace;font-size:11px;color:#334155;
          background:#f8fafc;padding:8px;border-radius:6px;
          word-break:break-all;margin:0">{tv_line}</p>
<p style="font-size:11px;color:#64748b;margin:12px 0 0">
  📎 Full results attached as CSV
</p>
  </td></tr>
  <tr><td style="background:#f8fafc;padding:12px 28px;
             border-top:1px solid #e2e8f0;text-align:center">
<p style="margin:0;color:#94a3b8;font-size:10px">
  Filters: 52w gain {CFG['min_52w_gain_pct']:.0f}-{CFG['max_52w_gain_pct']:.0f}%
  &nbsp;·&nbsp; within {CFG['max_off_52w_high_pct']:.0f}% of 52w high
  &nbsp;·&nbsp; SMA50&gt;150&gt;200 rising
  &nbsp;·&nbsp; &gt;={CFG['min_sma50_touches']} SMA50 holds
  &nbsp;·&nbsp; vol &gt;={CFG['vol_mult']}x
  &nbsp;·&nbsp; RS near 6m high
</p>
<p style="margin:6px 0 0;color:#94a3b8;font-size:10px">
  ⚠️ Not financial advice &nbsp;·&nbsp; Auto-generated by GitHub Actions
</p>
  </td></tr>
</table>
</td></tr></table>
</body></html>"""

        plain_lines = [
            f"Early Stage-2 Breakout Candidates — {datetime.today().strftime('%Y-%m-%d')}",
            f"{cnt} matches",
            "="*60,
        ]
        if rl:
            for r in rl[:CFG["max_email_rows"]]:
                plain_lines.append(
                    f"{r.get('Ticker','—'):<7} ${float(r.get('Price',0) or 0):.2f}  "
                    f"Total:{float(r.get('Total',0) or 0):.0f}  "
                    f"Pass:{int(r.get('Passed',0) or 0)}/6  "
                    f"RS:{float(r.get('RS_%',0) or 0):.0f}%  "
                    f"Holds:{int(r.get('Touches',0) or 0)}"
                )
            plain_lines.append("")
            plain_lines.append("ONE-LINE: " + ",".join(r.get("Ticker","") for r in rl))
        else:
            plain_lines.append("No matches today")
        plain_lines.append("\nFull results in CSV attachment.")
        plain_e = "\n".join(plain_lines)

        subj = (f"🚀 Early Stage-2 — {cnt} candidate{'s' if cnt!=1 else ''}"
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
        print("         Generate one: myaccount.google.com/apppasswords")
        print("         → Google Account → Security → 2-Step Verification → App Passwords")
    except smtplib.SMTPException as e:
        print(f"[Email] ❌  SMTP error: {e}")
    except Exception as e:
        print(f"[Email] ❌  Unexpected error: {type(e).__name__}: {e}")

try:
    _send_email(results, fpath)
except Exception as e:
    print(f"[Email] ❌  Unexpected top-level error: {type(e).__name__}: {e}")
    print("[Email]    Continuing — CSV and charts are still saved.")

if _IN_NOTEBOOK:
    try:
        from google.colab import files
        files.download(fpath); files.download(tv)
    except Exception: pass
else:
    print("  (CI: files in workspace, email sent)")

# ── Charts for top N ──────────────────────────────────────────
if results:
    top = results[:min(CFG["chart_top_n"], len(results))]
    fig, axes = plt.subplots(len(top),1,figsize=(15,5*len(top)),facecolor="#0f172a")
    if len(top)==1: axes=[axes]
    for ax, r in zip(axes, top):
        df_p = r["_df"].tail(180).copy()
        s50  = r["_s50"].reindex(df_p.index)
        s150 = r["_s150"].reindex(df_p.index)
        s200 = r["_s200"].reindex(df_p.index)
        ax.set_facecolor("#0f172a")
        ax.plot(df_p.index, df_p["Close"], color="#60a5fa", lw=1.8, label="Price", zorder=4)
        ax.plot(df_p.index, s50,  color="#fbbf24", lw=1.3, ls="--", label="SMA50",  zorder=3)
        ax.plot(df_p.index, s150, color="#f87171", lw=1.1, ls="-.", label="SMA150", zorder=3)
        ax.plot(df_p.index, s200, color="#a78bfa", lw=1.0, ls=":",  label="SMA200", zorder=3)

        # ASCII-only marker on the volume-breakout bar
        bod = r.get("BO_Date","—")
        if bod and bod != "—":
            try:
                bts = pd.Timestamp(bod)
                if bts in df_p.index:
                    ax.scatter([bts], [float(df_p.loc[bts,"Close"])], marker="^",
                               s=140, color="#34d399", zorder=6, label="Vol breakout")
            except Exception: pass

        # SMA50 support band
        cs50 = float(s50.dropna().iloc[-1]) if s50.notna().any() else None
        if cs50: ax.axhspan(cs50*0.97, cs50*1.03, alpha=0.07, color="#34d399", zorder=1)

        # RS panel (twin axis)
        rs = r.get("_rs")
        if rs is not None:
            rsp = rs.reindex(df_p.index).dropna()
            if len(rsp) > 10:
                ax2 = ax.twinx()
                ax2.plot(rsp.index, rsp.values, color="#e879f9", lw=1.0,
                         alpha=0.85, label="RS vs SPY")
                ax2.set_yticks([]); ax2.set_facecolor("none")
                for sp in ax2.spines.values(): sp.set_visible(False)

        ax.set_title(
            f"{r['Ticker']}  |  ${r['Price']:.2f}  |  Score {r['Total']:.0f}  "
            f"({r['Passed']}/6 hard)  |  52wGain {r['Gain52W_%']:+.1f}%  "
            f"OffHigh -{r['OffHigh_%']:.1f}%  |  SMA50 holds {r['Touches']}  "
            f"VolX {r['VolX']:.2f}  RS {r['RS_%']:.1f}%",
            color="#e2e8f0", fontsize=9, fontweight="bold", pad=7)
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %y"))
        ax.xaxis.set_major_locator(mdates.MonthLocator())
        ax.tick_params(colors="#94a3b8", labelsize=9)
        for sp in ax.spines.values(): sp.set_edgecolor("#1e3a5f")
        ax.legend(loc="upper left", facecolor="#1e293b", labelcolor="#e2e8f0",
                  fontsize=7, framealpha=0.9, ncol=2)
        ax.grid(color="#1e3a5f", ls="--", lw=0.5, alpha=0.6)
    plt.suptitle(
        f"Early Stage-2 Breakout Candidates  ·  "
        f"{datetime.today().strftime('%Y-%m-%d')}",
        color="#60a5fa", fontsize=12, fontweight="bold", y=1.001)
    plt.tight_layout()
    cp = os.path.join(out_dir, f"early_stage2_chart_{ts}.png")
    plt.savefig(cp, dpi=150, bbox_inches="tight", facecolor="#0f172a")
    if _IN_NOTEBOOK: plt.show()
    else: plt.close()
    print(f"  📊 Chart → {cp}")
    if _IN_NOTEBOOK:
        try:
            from google.colab import files; files.download(cp)
        except Exception: pass

print("""
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  📋 HARD FILTERS  (need 5 of 6 by default)
  F1  52w gain 30-60%        mid-move, not extended
  F2  within 15% of 52w high
  F3  SMA50 > SMA150 > SMA200, all rising
  F4  >= 3 SMA50 holds in last 6 months
  F5  vol >= 2.5x avg50 with up-close, last 20 days
  F6  RS vs SPY within 2% of its 6-month high

  📋 SOFT
  S1  ATR% contraction (recent 20d < prior 20d)

  📋 SCORE  (100 total)
  RS 30  ·  Support 24  ·  Proximity 14  ·  Volume 12
  Stack 10  ·  Slope 6  ·  Contraction 4

  💡 BEST SETUPS
  Passed 6/6      full structural template
  Total > 75      strongest composite
  RS_% >= 99      RS line making new highs now
  Touches >= 4    SMA50 acting as reliable support

  ⚙️  TUNE IF 0 RESULTS
  min_hard_filters     5 → 4
  min_total_score     55 → 45
  max_52w_gain_pct    60 → 80
  min_sma50_touches    3 → 2
  rs_high_tol_pct      2 → 5

  ⚠️  Base rate for any breakout becoming a 100% gainer is low.
      Size positions and keep a stop below the pivot.
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
""")
