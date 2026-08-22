# ============================================================
# NASDAQ — Multi-Timeframe Uptrend Alignment Scanner (v1)
# ============================================================
#
# Finds stocks in a confirmed uptrend on ALL FOUR timeframes AT
# THE SAME TIME: 1-Hour, 4-Hour, Daily, and Weekly.
#
# UPTREND DEFINITION (applied per timeframe, same test each time):
#   1. STACK:  Price > EMA(fast) > SMA(slow)  on that timeframe's
#              own bars
#   2. RISING: SMA(slow) today > SMA(slow) rising_lookback bars ago
#              (on that same timeframe)
#   Both required, on all 4 timeframes independently, for a match.
#
# DATA — only 2 downloads per ticker (not 4):
#   • Daily bars   (~550 days) → Daily uptrend directly, AND
#     resampled to weekly bars → Weekly uptrend
#   • Hourly bars  (~729 days, yfinance's practical ceiling for the
#     60m interval) → 1-Hour uptrend directly, AND resampled to
#     4-hour bars → 4-Hour uptrend
#   Yahoo/yfinance has no native "4h" interval, so 4H bars are built
#   by resampling the fetched hourly bars — same reasoning applies
#   to weekly-from-daily (no extra network call needed).
#
# SINGLE PASS — purely technical, no fundamentals fetch (this
# scanner is about trend alignment, not business quality).
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
    "history_days"        : 550,   # daily fetch window (→ also gives Weekly)
    "intraday_days"       : 729,   # hourly fetch window (yfinance ceiling for 60m)

    # ── Uptrend test (same shape on every timeframe) ────────────
    "ema_period"           : 8,
    "sma_period"           : 20,
    "rising_lookback"      : 5,    # bars back, for 1H/4H/Daily
    "rising_lookback_weekly": 3,   # weekly moves slower — shorter lookback

    # ── Filters ─────────────────────────────────────────────────
    "min_avg_volume"       : 80_000,
    "min_price"            : 2.0,

    "batch_size"           : 50,
    "batch_sleep"          : 1.5,
    "intraday_batch_size"  : 30,   # smaller batches — intraday payloads are heavier
    "intraday_batch_sleep" : 2.0,
}

# ── Uptrend test (reused across all 4 timeframes) ──────────────
def check_uptrend(df, ema_period, sma_period, rising_lookback):
    """
    STACK: price > EMA(fast) > SMA(slow)
    RISING: SMA(slow) now > SMA(slow) rising_lookback bars ago
    Both required. Returns (passed: bool, details: dict).
    """
    n = len(df)
    if n < max(ema_period, sma_period, rising_lookback) + 2:
        return False, {}
    ema = df["Close"].ewm(span=ema_period, adjust=False).mean()
    sma = df["Close"].rolling(sma_period).mean()
    price   = float(df["Close"].iloc[-1])
    cur_ema = float(ema.iloc[-1])
    cur_sma = float(sma.iloc[-1])
    if np.isnan(cur_ema) or np.isnan(cur_sma):
        return False, {}

    stack_ok = price > cur_ema > cur_sma

    sma_prev = (float(sma.iloc[-1-rising_lookback])
                if n > rising_lookback and not np.isnan(sma.iloc[-1-rising_lookback])
                else np.nan)
    rising_ok = (not np.isnan(sma_prev)) and (cur_sma > sma_prev)

    passed = stack_ok and rising_ok
    dist_pct = (price - cur_sma) / cur_sma * 100 if cur_sma > 0 else 0
    slope_pct = ((cur_sma - sma_prev) / sma_prev * 100
                 if (not np.isnan(sma_prev) and sma_prev > 0) else 0)
    return passed, {
        "price": price, "ema": round(cur_ema, 2), "sma": round(cur_sma, 2),
        "stack_ok": stack_ok, "rising_ok": rising_ok,
        "dist_above_sma_pct": round(dist_pct, 2),
        "sma_slope_pct": round(slope_pct, 2),
    }

def resample_ohlcv(df, rule):
    """Resample an OHLCV dataframe to a coarser bar size."""
    agg = {"Open": "first", "High": "max", "Low": "min",
           "Close": "last", "Volume": "sum"}
    out = df.resample(rule).agg(agg)
    out.dropna(subset=["Close"], inplace=True)
    return out

# ── Technical signal: 4-timeframe uptrend alignment ─────────────
def analyze_multi_tf_uptrend(sym, df_daily, df_hourly):
    """
    Returns dict with score and per-timeframe details, or None if
    any of the 4 timeframes is not in an uptrend.
    """
    if df_daily is None or df_hourly is None:
        return None

    price   = float(df_daily["Close"].iloc[-1])
    avg_vol = float(df_daily["Volume"].tail(20).mean())
    if price   < CFG["min_price"]:      return None
    if avg_vol < CFG["min_avg_volume"]: return None

    df_weekly = resample_ohlcv(df_daily, "W")
    df_4h     = resample_ohlcv(df_hourly, "4h")

    tf_specs = [
        ("1H",     df_hourly, CFG["ema_period"], CFG["sma_period"], CFG["rising_lookback"]),
        ("4H",     df_4h,     CFG["ema_period"], CFG["sma_period"], CFG["rising_lookback"]),
        ("Daily",  df_daily,  CFG["ema_period"], CFG["sma_period"], CFG["rising_lookback"]),
        ("Weekly", df_weekly, CFG["ema_period"], CFG["sma_period"], CFG["rising_lookback_weekly"]),
    ]

    tf_results = {}
    for tf_name, tf_df, ema_p, sma_p, rise_lb in tf_specs:
        passed, details = check_uptrend(tf_df, ema_p, sma_p, rise_lb)
        tf_results[tf_name] = details
        if not passed:
            return None   # ALL 4 required — fail fast on the first miss

    # ── Alignment score (0-100): 25 pts/timeframe, scaled by how far
    #    above its own rising SMA each timeframe is ─────────────────
    score = 0
    reasons = []
    for tf_name, d in tf_results.items():
        pts = min(25, 15 + d["dist_above_sma_pct"])
        pts = max(10, pts)   # floor — it already passed the gate
        score += pts
        reasons.append(f"{tf_name}+{d['dist_above_sma_pct']:.1f}%")
    score = round(min(100, max(0, score)))

    return {
        "Score"          : score,
        "Price"          : round(price, 2),
        "1H_Price"       : tf_results["1H"]["price"],
        "1H_EMA8"        : tf_results["1H"]["ema"],
        "1H_SMA20"       : tf_results["1H"]["sma"],
        "1H_Dist_%"      : tf_results["1H"]["dist_above_sma_pct"],
        "4H_Price"       : tf_results["4H"]["price"],
        "4H_EMA8"        : tf_results["4H"]["ema"],
        "4H_SMA20"       : tf_results["4H"]["sma"],
        "4H_Dist_%"      : tf_results["4H"]["dist_above_sma_pct"],
        "Daily_EMA8"     : tf_results["Daily"]["ema"],
        "Daily_SMA20"    : tf_results["Daily"]["sma"],
        "Daily_Dist_%"   : tf_results["Daily"]["dist_above_sma_pct"],
        "Weekly_EMA8"    : tf_results["Weekly"]["ema"],
        "Weekly_SMA20"   : tf_results["Weekly"]["sma"],
        "Weekly_Dist_%"  : tf_results["Weekly"]["dist_above_sma_pct"],
        "Flags"          : " | ".join(reasons),
        "_df_daily"      : df_daily,
        "_df_weekly"     : df_weekly,
    }

# ── Download: daily (→ also Weekly via resample) ────────────────
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
                        df = _clean(df, min_bars=200)
                        if df is not None: out[sym] = df
                    except Exception: pass
            elif len(symbols) == 1:
                df = _clean(raw, min_bars=200)
                if df is not None: out[symbols[0]] = df
    except Exception: pass
    for sym in [s for s in symbols if s not in out]:
        for _ in range(2):
            try:
                df = yf.Ticker(sym).history(
                    start=start.strftime("%Y-%m-%d"),
                    end=end.strftime("%Y-%m-%d"),
                    auto_adjust=True, actions=False)
                df = _clean(df, min_bars=200)
                if df is not None: out[sym] = df; break
            except Exception: time.sleep(0.2)
        time.sleep(0.04)
    return out

# ── Download: hourly (→ also 4H via resample) ───────────────────
def download_hourly(symbols, days):
    end = datetime.today(); start = end - timedelta(days=days)
    out = {}
    try:
        raw = yf.download(symbols, start=start.strftime("%Y-%m-%d"),
                          end=end.strftime("%Y-%m-%d"), interval="60m",
                          group_by="ticker", auto_adjust=True, actions=False,
                          threads=True, progress=False)
        if raw is not None and not raw.empty:
            pf = {"Open","High","Low","Close","Volume","Adj Close"}
            if isinstance(raw.columns, pd.MultiIndex):
                l0 = set(raw.columns.get_level_values(0))
                for sym in symbols:
                    try:
                        df = raw.xs(sym,axis=1,level=1) if l0&pf else raw[sym]
                        df = _clean(df, min_bars=100)
                        if df is not None: out[sym] = df
                    except Exception: pass
            elif len(symbols) == 1:
                df = _clean(raw, min_bars=100)
                if df is not None: out[symbols[0]] = df
    except Exception: pass
    for sym in [s for s in symbols if s not in out]:
        for _ in range(2):
            try:
                df = yf.Ticker(sym).history(
                    start=start.strftime("%Y-%m-%d"),
                    end=end.strftime("%Y-%m-%d"),
                    interval="60m", auto_adjust=True, actions=False)
                df = _clean(df, min_bars=100)
                if df is not None: out[sym] = df; break
            except Exception: time.sleep(0.2)
        time.sleep(0.04)
    return out

# ── Live print ────────────────────────────────────────────────
LIVE_COLS = ["Ticker","Price","Score","1H_Dist_%","4H_Dist_%",
             "Daily_Dist_%","Weekly_Dist_%"]
_CW = {"Ticker":8,"Price":10,"Score":7,"1H_Dist_%":10,"4H_Dist_%":10,
       "Daily_Dist_%":12,"Weekly_Dist_%":13}
_CF = {"Price":"${:.2f}","Score":"{:.0f}","1H_Dist_%":"{:+.1f}%",
       "4H_Dist_%":"{:+.1f}%","Daily_Dist_%":"{:+.1f}%","Weekly_Dist_%":"{:+.1f}%"}
_hdr_done = False

def _live_header():
    global _hdr_done
    if _hdr_done: return
    print("\n" + "━"*95)
    print("  📊  LIVE MATCHES  —  each stock printed the moment it passes all 4 timeframes")
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
chk_d = download_daily(["AAPL","MSFT","NVDA"], 300)
chk_h = download_hourly(["AAPL","MSFT","NVDA"], 60)
if not chk_d or not chk_h:
    print("❌  No data.")
else:
    for s in chk_d:
        dd = chk_d[s]
        hh = chk_h.get(s)
        print(f"  ✅ {s}: daily {len(dd)} bars (${float(dd['Close'].iloc[-1]):.2f}, "
              f"{dd.index[-1].date()})" + (f"  |  hourly {len(hh)} bars" if hh is not None else "  |  hourly ⚠️ missing"))
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

# ── Main scan — single pass (daily+hourly download, then check) ──
print("━"*65)
print(f"  STEP 3  SCANNING {len(TICKERS)} TICKERS")
print("━"*65)
print("  Fetching daily bars (→ Daily + Weekly) and hourly bars (→ 1H + 4H)")
print("  A stock only matches if ALL 4 timeframes are in an uptrend\n")

_hdr_done = False
results = []
no_daily_data  = 0
no_hourly_data = 0

daily_batches = [TICKERS[i:i+CFG["batch_size"]]
                 for i in range(0, len(TICKERS), CFG["batch_size"])]
hourly_batch_size = CFG["intraday_batch_size"]

daily_map  = {}
hourly_map = {}

with tqdm(total=len(TICKERS), desc="Daily fetch", unit="stk",
          bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]") as pbar:
    for batch in daily_batches:
        got = download_daily(batch, CFG["history_days"])
        daily_map.update(got)
        no_daily_data += len(batch) - len(got)
        pbar.update(len(batch))
        time.sleep(CFG["batch_sleep"])

# Only fetch hourly data for tickers that at least survived the daily fetch
# (saves the heavier intraday call for symbols we already know are dead/delisted)
daily_survivors = list(daily_map.keys())
hourly_batches = [daily_survivors[i:i+hourly_batch_size]
                  for i in range(0, len(daily_survivors), hourly_batch_size)]

with tqdm(total=len(daily_survivors), desc="Hourly fetch", unit="stk",
          bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]") as pbar:
    for batch in hourly_batches:
        got = download_hourly(batch, CFG["intraday_days"])
        hourly_map.update(got)
        no_hourly_data += len(batch) - len(got)
        pbar.update(len(batch))
        time.sleep(CFG["intraday_batch_sleep"])

got_daily  = len(TICKERS) - no_daily_data
got_hourly = len(daily_survivors) - no_hourly_data
print(f"\n  Daily data : {got_daily}/{len(TICKERS)} tickers")
print(f"  Hourly data: {got_hourly}/{len(daily_survivors)} daily-survivors")
print()

for sym in tqdm(daily_survivors, desc="Checking alignment", unit="stk"):
    if sym not in hourly_map: continue
    try:
        r = analyze_multi_tf_uptrend(sym, daily_map[sym], hourly_map[sym])
        if r is None: continue
        r["Ticker"] = sym
        results.append(r)
        live_print(r)
    except Exception: pass

print(f"\n{'━'*65}")
print(f"  SCAN COMPLETE")
print(f"  Tickers scanned : {len(TICKERS)}")
print(f"  Daily data      : {got_daily}")
print(f"  Hourly data     : {got_hourly}")
print(f"  ✅ 4-TF Matches  : {len(results)}")
print(f"{'━'*65}")

if not results:
    print("\n  No matches. Try relaxing:")
    print("   sma_period                20 → 10")
    print("   rising_lookback            5 → 3")
    print("   rising_lookback_weekly     3 → 2")
    print("   min_price                  2 → 1")
    print("   min_avg_volume         80000 → 50000")

results.sort(key=lambda x: x["Score"], reverse=True)

# ── Always build df_out and save/email (even if 0 results) ────
ts      = datetime.today().strftime("%Y%m%d_%H%M")
out_dir = os.environ.get("GITHUB_WORKSPACE", os.getcwd())

COLS = [
    "Ticker","Price","Score",
    "1H_Dist_%","4H_Dist_%","Daily_Dist_%","Weekly_Dist_%",
    "1H_EMA8","1H_SMA20","4H_EMA8","4H_SMA20",
    "Daily_EMA8","Daily_SMA20","Weekly_EMA8","Weekly_SMA20",
    "Flags",
]
df_out = pd.DataFrame([{k:v for k,v in r.items() if not k.startswith("_")}
                        for r in results]) if results else pd.DataFrame(columns=COLS)
if not df_out.empty:
    df_out = df_out[[c for c in COLS if c in df_out.columns]]
    df_out.reset_index(drop=True, inplace=True)

FMT = {
    "Price"        : lambda v: f"${v:.2f}",
    "Score"        : lambda v: f"{v:.0f}",
    "1H_Dist_%"    : lambda v: f"{v:+.1f}%",
    "4H_Dist_%"    : lambda v: f"{v:+.1f}%",
    "Daily_Dist_%" : lambda v: f"{v:+.1f}%",
    "Weekly_Dist_%": lambda v: f"{v:+.1f}%",
    "1H_EMA8"      : lambda v: f"${v:.2f}",
    "1H_SMA20"     : lambda v: f"${v:.2f}",
    "4H_EMA8"      : lambda v: f"${v:.2f}",
    "4H_SMA20"     : lambda v: f"${v:.2f}",
    "Daily_EMA8"   : lambda v: f"${v:.2f}",
    "Daily_SMA20"  : lambda v: f"${v:.2f}",
    "Weekly_EMA8"  : lambda v: f"${v:.2f}",
    "Weekly_SMA20" : lambda v: f"${v:.2f}",
}

def fmt_v(col, val):
    if val is None or (isinstance(val, float) and np.isnan(val)): return "—"
    try:
        if col in FMT: return FMT[col](val)
    except Exception: pass
    return str(val) if str(val) not in ("nan","None","") else "—"

if _IN_NOTEBOOK and results:
    DISP = ["Ticker","Price","Score","1H_Dist_%","4H_Dist_%",
            "Daily_Dist_%","Weekly_Dist_%"]
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
            elif col.endswith("_Dist_%"):
                try:
                    v = float(str(raw).replace("%","").replace("+",""))
                    clr = "#22c55e" if v >= 0 else "#ef4444"
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
    <span style="color:#f1f5f9;font-size:15px;font-weight:700">Multi-Timeframe Uptrend Alignment</span>
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
    📈 Multi-Timeframe Uptrend Alignment (1H + 4H + Daily + Weekly)
  </h2>
  <p style="margin:6px 0 0;color:#94a3b8;font-size:12px">
    {datetime.today().strftime('%Y-%m-%d %H:%M')} &nbsp;·&nbsp;
    <b style="color:#22c55e">{len(results)} matches</b> from {len(TICKERS)} tickers
  </p>
</div>"""

    legend_html = """
<div style="background:#f8fafc;border:1px solid #e2e8f0;border-radius:8px;
        padding:12px 18px;margin-top:6px;font-size:11px;color:#64748b;
        font-family:'Segoe UI',Arial,sans-serif">
  <b style="color:#475569">GUIDE</b> &nbsp;·&nbsp;
  Uptrend (per timeframe) = Price &gt; EMA8 &gt; SMA20, AND SMA20 rising &nbsp;·&nbsp;
  ALL FOUR of 1H, 4H, Daily, and Weekly must pass at once &nbsp;·&nbsp;
  Score (0-100) = 25 pts/timeframe, scaled by distance above its own SMA20
</div>"""

    display_html(header_html + table_html + legend_html)

elif results:
    # ASCII table (CLI/GitHub Actions mode)
    CLI_COLS = ["Ticker","Price","Score","1H_Dist_%","4H_Dist_%",
                "Daily_Dist_%","Weekly_Dist_%"]
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
    tit = f"  Multi-TF Uptrend (1H+4H+Daily+Weekly)   {datetime.today().strftime('%Y-%m-%d')}   {len(df_out)} matches"
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
  Score           0-100, 25 pts/timeframe scaled by distance above SMA20
  1H/4H/Daily/Weekly_Dist_%   price's distance above that timeframe's SMA20
  ──────────────────────────────────────────────────────""")

# Save
fpath = os.path.join(out_dir, f"multi_tf_uptrend_{ts}.csv")
df_out.to_csv(fpath, index=False)
print(f"\n  💾 CSV → {fpath}")
tv = os.path.join(out_dir, f"tv_multi_tf_uptrend_{ts}.txt")
with open(tv,"w") as f:
    f.write(f"###Multi-Timeframe Uptrend Alignment {datetime.today().strftime('%Y-%m-%d')}\n")
    for r in results: f.write(f"NASDAQ:{r['Ticker']}\n")
print(f"  📋 TradingView → {tv}")

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
            for c in ["Ticker","Price","Score","1H_Dist_%","4H_Dist_%",
                      "Daily_Dist_%","Weekly_Dist_%"]
        )
        rows_e = ""
        for i, r in enumerate(rl[:50]):
            bg  = "#fff" if i % 2 == 0 else "#f0f9ff"
            ticker = r.get("Ticker","—")
            price  = r.get("Price",0) or 0
            score  = r.get("Score",0) or 0
            d1h    = r.get("1H_Dist_%",0) or 0
            d4h    = r.get("4H_Dist_%",0) or 0
            dd     = r.get("Daily_Dist_%",0) or 0
            dw     = r.get("Weekly_Dist_%",0) or 0
            def _c(v): return "#22c55e" if float(v) >= 0 else "#ef4444"
            rows_e += (
                f'<tr style="background:{bg}">'
                f'<td style="padding:6px 11px;font-size:12px;font-weight:700">{ticker}</td>'
                f'<td style="padding:6px 11px;font-size:12px">${float(price):.2f}</td>'
                f'<td style="padding:6px 11px;font-size:12px;font-weight:700;'
                f'background:#166534;color:#fff;text-align:center">{float(score):.0f}</td>'
                f'<td style="padding:6px 11px;font-size:12px;color:{_c(d1h)}">{float(d1h):+.1f}%</td>'
                f'<td style="padding:6px 11px;font-size:12px;color:{_c(d4h)}">{float(d4h):+.1f}%</td>'
                f'<td style="padding:6px 11px;font-size:12px;color:{_c(dd)}">{float(dd):+.1f}%</td>'
                f'<td style="padding:6px 11px;font-size:12px;color:{_c(dw)}">{float(dw):+.1f}%</td>'
                f'</tr>'
            )
        no_results_msg = ('<tr><td colspan="7" style="padding:20px;text-align:center;'
                           'color:#94a3b8;font-size:13px">No matches today</td></tr>')

        html_e = f"""<!DOCTYPE html><html><body style="margin:0;padding:0;
background:#f1f5f9;font-family:'Segoe UI',Arial,sans-serif">
<table width="100%" cellpadding="0" cellspacing="0"><tr><td align="center" style="padding:20px 10px">
<table width="100%" style="max-width:800px;background:#fff;border-radius:12px;
       overflow:hidden;box-shadow:0 4px 16px rgba(0,0,0,0.08)">
  <tr><td style="background:linear-gradient(135deg,#0f172a,#1e3a5f);padding:22px 28px">
<h1 style="margin:0;color:#60a5fa;font-size:20px;font-weight:700">
  📊 Multi-Timeframe Uptrend Alignment
</h1>
<p style="margin:6px 0 0;color:#94a3b8;font-size:12px">
  {datetime.today().strftime('%Y-%m-%d %H:%M UTC')} &nbsp;·&nbsp;
  {cnt} match{'es' if cnt!=1 else ''} found — 1H + 4H + Daily + Weekly, all at once
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
            f"Multi-Timeframe Uptrend Alignment — {datetime.today().strftime('%Y-%m-%d')}",
            f"{cnt} matches (1H + 4H + Daily + Weekly all in uptrend at once)",
            "="*60,
        ]
        if rl:
            for r in rl[:50]:
                ticker = r.get("Ticker","—")
                price  = r.get("Price",0) or 0
                score  = r.get("Score",0) or 0
                plain_lines.append(
                    f"{ticker:<7} ${float(price):.2f}  Score:{float(score):.0f}  "
                    f"1H:{r.get('1H_Dist_%',0):+.1f}%  4H:{r.get('4H_Dist_%',0):+.1f}%  "
                    f"D:{r.get('Daily_Dist_%',0):+.1f}%  W:{r.get('Weekly_Dist_%',0):+.1f}%"
                )
        else:
            plain_lines.append("No matches today")
        plain_lines.append("\nFull results in CSV attachment.")
        plain_e = "\n".join(plain_lines)

        subj = (f"📊 Multi-TF Uptrend (1H+4H+D+W) — {cnt} signal{'s' if cnt!=1 else ''}"
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

# ── Charts for top 5 (daily view with weekly overlay context) ────
if results:
    top = results[:min(5,len(results))]
    fig, axes = plt.subplots(len(top),1,figsize=(15,5*len(top)),facecolor="#0f172a")
    if len(top)==1: axes=[axes]
    for ax, r in zip(axes, top):
        df_p = r["_df_daily"].tail(180).copy()
        ema8 = df_p["Close"].ewm(span=8, adjust=False).mean()
        sma20 = df_p["Close"].rolling(20).mean()
        ax.set_facecolor("#0f172a")
        ax.plot(df_p.index, df_p["Close"], color="#60a5fa", lw=1.6, label="Daily Close", zorder=5)
        ax.plot(df_p.index, ema8,  color="#38bdf8", lw=1.1, ls="--", label="EMA8", zorder=3)
        ax.plot(df_p.index, sma20, color="#fbbf24", lw=1.2, ls="-.", label="SMA20", zorder=3)
        ax.set_title(
            f"{r['Ticker']}  |  ${r['Price']:.2f}  |  Score {r['Score']}  |  "
            f"1H:{r['1H_Dist_%']:+.1f}%  4H:{r['4H_Dist_%']:+.1f}%  "
            f"D:{r['Daily_Dist_%']:+.1f}%  W:{r['Weekly_Dist_%']:+.1f}%",
            color="#e2e8f0", fontsize=9, fontweight="bold", pad=7)
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %y"))
        ax.tick_params(colors="#94a3b8", labelsize=9)
        for sp in ax.spines.values(): sp.set_edgecolor("#1e3a5f")
        ax.legend(loc="upper left", facecolor="#1e293b", labelcolor="#e2e8f0",
                  fontsize=7, framealpha=0.9)
        ax.grid(color="#1e3a5f", ls="--", lw=0.5, alpha=0.6)
    plt.suptitle(
        f"Multi-Timeframe Uptrend Alignment  ·  {datetime.today().strftime('%Y-%m-%d')}",
        color="#60a5fa", fontsize=12, fontweight="bold", y=1.001)
    plt.tight_layout()
    cp = os.path.join(out_dir, f"multi_tf_uptrend_chart_{ts}.png")
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
  📋 UPTREND TEST (same on every timeframe)
  1) STACK:  Price > EMA8 > SMA20  (on that timeframe's own bars)
  2) RISING: SMA20 now > SMA20 rising_lookback bars ago
  Both required, on 1H, 4H, Daily, AND Weekly simultaneously.

  📋 DATA SOURCING
  Only 2 downloads per ticker:
    Daily bars  → Daily uptrend directly + resampled to Weekly
    Hourly bars → 1H uptrend directly    + resampled to 4H
  (Yahoo/yfinance has no native 4H interval, and no separate
  weekly fetch is needed since daily resamples into it cleanly.)

  📋 SCORE (0-100)
  25 points per timeframe, scaled by how far price sits above
  that timeframe's own rising SMA20 — a stock aligned everywhere
  but only barely above each SMA scores lower than one with a
  strong margin on every timeframe.

  💡 BEST SETUPS
  Score > 80          strong alignment margin on all 4 timeframes
  Weekly_Dist_% high   the long-term trend has real room, not a
                       fresh breakout that could fail
  1H/4H_Dist_% modest  intraday isn't already overextended

  ⚙️  TUNE IF 0 RESULTS
  sma_period                20 → 10   (looser trend definition)
  rising_lookback             5 → 3
  rising_lookback_weekly      3 → 2
  min_price                   2 → 1
  min_avg_volume          80000 → 50000
  intraday_days              729 → 400   (if hourly fetch is failing/slow)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
""")
