# ============================================================
# NASDAQ — JMA Price-Cross + MA Stack Scanner (v1)
# ============================================================
#
# SIGNAL (all required):
#   1. Price closed above JMA today, having been at/below JMA
#      the prior bar — i.e. a fresh price/JMA crossover from below,
#      confirmed by close (not just an intrabar poke).
#   2. JMA is stacked ABOVE EMA8, SMA21, SMA50, and SMA150
#      (JMA leading a fully aligned short/medium/long MA stack).
#   3. Volume today > volume yesterday (fresh interest confirming
#      the cross, not a low-conviction move).
#
# 2-PASS ARCHITECTURE (same as fundamental_v2.py):
#   PASS 1 — technical signal above (fast, no .info calls)
#   PASS 2 — fundamentals (slow, .info calls — pass-1 survivors only)
#
#   FINAL SCORE = Fund (0-50) + Tech (0-30) = 80
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
    "history_days"               : 550,

    # ── Fundamental thresholds ─────────────────────────────────
    "min_revenue_growth_pct"     : 0.0,
    "min_profit_margin_pct"      : 0.0,
    "max_debt_to_equity"         : 2.0,
    "min_current_ratio"          : 1.0,
    "min_roe_pct"                : 10.0,
    "max_pe_ratio"                : 50.0,
    "high_growth_threshold_pct"  : 15.0,

    # ── JMA / MA stack signal ───────────────────────────────────
    "jma_period"                  : 13,   # JMA 13 40 2
    "jma_phase"                   : 40,
    "ema8_period"                  : 8,
    "sma21_period"                : 21,
    "sma50_period"                : 50,
    "sma150_period"               : 150,
    "price_jma_cross_lookback"    : 3,    # bars to look back for a fresh close-above-JMA cross
    "require_price_jma_cross"     : True, # price closed above JMA from below, within lookback
    "require_jma_above_ema8"      : True, # JMA > EMA8
    "require_jma_above_sma21"     : True, # JMA > SMA21
    "require_jma_above_sma50"     : True, # JMA > SMA50
    "require_jma_above_sma150"    : True, # JMA > SMA150
    "require_volume_increase"     : True, # today's volume > yesterday's volume

    # ── Score gates ─────────────────────────────────────────────
    "min_tech_score"              : 10,   # out of 30
    "min_total_score"             : 15,   # out of 80 (Fund 0-50 + Tech 0-30)

    # ── Filters ─────────────────────────────────────────────────
    "min_avg_volume"              : 80_000,
    "min_price"                   : 2.0,

    "batch_size"                  : 50,
    "batch_sleep"                  : 1.5,
    "fund_sleep"                    : 0.3,
}

# ── Indicators ───────────────────────────────────────────────
def calc_jma(series, period=13, phase=40, power=2):
    """
    JMA (Jurik Moving Average) approximation — adaptive EMA with
    phase-based smoothing factor. The true JMA is proprietary;
    this is the widely-used pure-numpy approximation, using the
    corrected e2 update (steady-state gain 1.0, tracks price
    correctly — see fundamental_v2.py for details on the bug in
    the original snippet this is derived from).

    phase : -100..+100 (positive = more responsive)
    power : typically 2
    """
    n      = len(series)
    vals   = series.values.astype(float)
    result = np.full(n, np.nan)

    phase_ratio = phase / 100.0 + 1.5
    alpha = 2.0 / (period + 1.0)
    beta  = alpha * phase_ratio

    first_valid = 0
    for i in range(n):
        if not np.isnan(vals[i]):
            first_valid = i
            break

    e0 = e1 = e2 = vals[first_valid]
    result[first_valid] = e0

    for i in range(first_valid + 1, n):
        v   = vals[i]
        e0  = (1 - alpha) * e0 + alpha * v
        e1  = (v - e0) * (1 - beta) + beta * e1
        e2  = (1 - alpha) * e2 + alpha * (e0 + e1)
        result[i] = e2

    return pd.Series(result, index=series.index)

def find_price_cross_above_jma(close, jma_s, lookback=3):
    """
    Detects a bar where close crosses from at/below JMA to above JMA
    (confirmed by close), within the last `lookback` bars.
    Returns (crossed: bool, bars_since: int|None).
    """
    crossed    = False
    bars_since = None
    n = len(close)
    for back in range(0, lookback + 1):
        i = n - 1 - back
        if i < 1: break
        c_i, c_p = close.iloc[i],  close.iloc[i-1]
        j_i, j_p = jma_s.iloc[i],  jma_s.iloc[i-1]
        if any(np.isnan(v) for v in [c_i, c_p, j_i, j_p]): continue
        if c_p <= j_p and c_i > j_i:
            crossed    = True
            bars_since = back
            break
    return crossed, bars_since

# ── Fundamental data fetch (robust) ────────────────────────────
def get_fundamentals(sym):
    """
    Fetch fundamentals via yf.Ticker(sym).info
    Never returns None — always returns a dict, Fund_Score=0 if
    data unavailable.
    """
    empty = {
        "Company":"—", "Sector":"—", "Industry":"—",
        "Rev_Growth_%":None, "Profit_Margin_%":None,
        "ROE_%":None, "PE_Ratio":None, "EPS":None,
        "Debt_Equity":None, "Current_Ratio":None,
        "Market_Cap_B":None, "Fund_Score":0,
        "Fund_Flags":"No Data",
    }
    try:
        tk   = yf.Ticker(sym)
        info = tk.info
        if not info or len(info) < 10:
            return empty

        def safe(key):
            v = info.get(key)
            if v is None: return None
            try:    return float(v)
            except Exception: return None

        rev_growth     = safe("revenueGrowth")
        profit_margin  = safe("profitMargins")
        roe            = safe("returnOnEquity")
        pe             = safe("trailingPE")
        eps            = safe("trailingEps")
        de             = safe("debtToEquity")
        cr             = safe("currentRatio")
        mktcap         = safe("marketCap")
        company        = info.get("longName", sym)
        sector         = info.get("sector", "—")
        industry       = info.get("industry", "—")

        rg_pct  = rev_growth    * 100 if rev_growth    is not None else None
        pm_pct  = profit_margin * 100 if profit_margin is not None else None
        roe_pct = roe           * 100 if roe           is not None else None

        # ── Score (0-50) ──────────────────────────────────────
        score = 0
        flags = []

        if rg_pct is not None:
            if rg_pct > CFG["min_revenue_growth_pct"]:
                score += 8; flags.append(f"RevG{rg_pct:+.0f}%")
            if rg_pct > CFG["high_growth_threshold_pct"]:
                score += 4; flags.append("HiGrw")
        if eps is not None and eps > 0:
            score += 7; flags.append(f"EPS${eps:.2f}")
        if pm_pct is not None and pm_pct > CFG["min_profit_margin_pct"]:
            score += 6; flags.append(f"Mgn{pm_pct:.0f}%")
        if de is not None and de < CFG["max_debt_to_equity"]:
            score += 6; flags.append(f"DE{de:.1f}")
        if cr is not None and cr > CFG["min_current_ratio"]:
            score += 5; flags.append(f"CR{cr:.1f}")
        if roe_pct is not None and roe_pct > CFG["min_roe_pct"]:
            score += 8; flags.append(f"ROE{roe_pct:.0f}%")
        if pe is not None and 0 < pe < CFG["max_pe_ratio"]:
            score += 6; flags.append(f"PE{pe:.0f}")

        return {
            "Company"          : company,
            "Sector"           : sector,
            "Industry"         : industry,
            "Rev_Growth_%"     : round(rg_pct, 1)  if rg_pct  is not None else None,
            "Profit_Margin_%"  : round(pm_pct, 1)  if pm_pct  is not None else None,
            "ROE_%"            : round(roe_pct, 1) if roe_pct is not None else None,
            "PE_Ratio"         : round(pe, 1)      if pe      is not None else None,
            "EPS"              : round(eps, 2)     if eps     is not None else None,
            "Debt_Equity"      : round(de, 2)      if de      is not None else None,
            "Current_Ratio"    : round(cr, 2)      if cr      is not None else None,
            "Market_Cap_B"     : round(mktcap/1e9,2) if mktcap is not None else None,
            "Fund_Score"       : min(50, score),
            "Fund_Flags"       : " ".join(flags) if flags else "—",
        }
    except Exception:
        return empty

# ── Technical signal: JMA price-cross + MA stack ────────────────
def analyze_jma_stack_cross(sym, df):
    """
    Returns dict with tech_score and details, or None if any
    required condition fails.
    """
    n       = len(df)
    price   = float(df["Close"].iloc[-1])
    vol_today     = float(df["Volume"].iloc[-1])
    vol_yesterday = float(df["Volume"].iloc[-2]) if n >= 2 else 0
    avg_vol = float(df["Volume"].tail(20).mean())

    if price   < CFG["min_price"]:      return None
    if avg_vol < CFG["min_avg_volume"]: return None

    ema8   = df["Close"].ewm(span=CFG["ema8_period"], adjust=False).mean()
    sma21  = df["Close"].rolling(CFG["sma21_period"]).mean()
    sma50  = df["Close"].rolling(CFG["sma50_period"]).mean()
    sma150 = df["Close"].rolling(CFG["sma150_period"]).mean()
    jma_s  = calc_jma(df["Close"], CFG["jma_period"], CFG["jma_phase"])

    cur_ema8   = float(ema8.iloc[-1])
    cur_sma21  = float(sma21.iloc[-1])
    cur_sma50  = float(sma50.iloc[-1])
    cur_sma150 = float(sma150.iloc[-1])
    cur_jma    = float(jma_s.iloc[-1])
    if any(np.isnan(v) for v in
           [cur_ema8, cur_sma21, cur_sma50, cur_sma150, cur_jma]):
        return None

    # ── Condition 1: price closed above JMA from below (fresh cross) ──
    crossed, bars_since = find_price_cross_above_jma(
        df["Close"], jma_s, CFG["price_jma_cross_lookback"])
    if CFG["require_price_jma_cross"] and not crossed:
        return None

    # ── Condition 2: JMA stacked above EMA8/SMA21/SMA50/SMA150 ────────
    jma_above_ema8   = cur_jma > cur_ema8
    jma_above_sma21  = cur_jma > cur_sma21
    jma_above_sma50  = cur_jma > cur_sma50
    jma_above_sma150 = cur_jma > cur_sma150

    if CFG["require_jma_above_ema8"]   and not jma_above_ema8:   return None
    if CFG["require_jma_above_sma21"]  and not jma_above_sma21:  return None
    if CFG["require_jma_above_sma50"]  and not jma_above_sma50:  return None
    if CFG["require_jma_above_sma150"] and not jma_above_sma150: return None

    # ── Condition 3: volume increasing vs previous day ────────────────
    vol_increasing = vol_today > vol_yesterday
    vol_chg_pct = ((vol_today - vol_yesterday) / vol_yesterday * 100
                   if vol_yesterday > 0 else 0)
    if CFG["require_volume_increase"] and not vol_increasing:
        return None

    # ── Technical score (0-30) ────────────────────────────────
    ts = 0
    tr = []
    if crossed:
        pts = 10 if bars_since == 0 else (7 if bars_since <= 1 else 4)
        ts += pts; tr.append(f"JMA_Cross({bars_since}d)")
    stack_count = sum([jma_above_ema8, jma_above_sma21,
                        jma_above_sma50, jma_above_sma150])
    ts += stack_count * 2  # up to 8
    if stack_count == 4:
        ts += 4; tr.append("FullStack")
    if vol_increasing:
        vol_pts = 8 if vol_chg_pct >= 50 else (5 if vol_chg_pct >= 20 else 3)
        ts += vol_pts; tr.append(f"Vol{vol_chg_pct:+.0f}%")
    ts = min(30, ts)

    return {
        "tech_score"     : ts,
        "tech_reasons"   : " | ".join(tr),
        "JMA"            : round(cur_jma, 2),
        "EMA8"           : round(cur_ema8, 2),
        "SMA21"          : round(cur_sma21, 2),
        "SMA50"          : round(cur_sma50, 2),
        "SMA150"         : round(cur_sma150, 2),
        "Bars_Since_Cross": bars_since,
        "Vol_Chg_%"      : round(vol_chg_pct, 1),
        "Vol_Today"      : int(vol_today),
        "Vol_Yesterday"  : int(vol_yesterday),
        "_df"            : df,
        "_jma"           : jma_s,
        "_ema8"          : ema8,
        "_sma21"         : sma21,
        "_sma50"         : sma50,
        "_sma150"        : sma150,
    }

# ── Download ──────────────────────────────────────────────────
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

def download(symbols, days):
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
                        df = _clean(df)
                        if df is not None: out[sym] = df
                    except Exception: pass
            elif len(symbols) == 1:
                df = _clean(raw)
                if df is not None: out[symbols[0]] = df
    except Exception: pass
    for sym in [s for s in symbols if s not in out]:
        for _ in range(2):
            try:
                df = yf.Ticker(sym).history(
                    start=start.strftime("%Y-%m-%d"),
                    end=end.strftime("%Y-%m-%d"),
                    auto_adjust=True, actions=False)
                df = _clean(df)
                if df is not None: out[sym] = df; break
            except Exception: time.sleep(0.2)
        time.sleep(0.04)
    return out

# ── Live print ────────────────────────────────────────────────
LIVE_COLS = ["Ticker","Price","Total","Fund","Tech",
             "Bars_Since_Cross","Vol_Chg_%","Sector"]
_CW = {"Ticker":8,"Price":10,"Total":7,"Fund":6,"Tech":6,
       "Bars_Since_Cross":10,"Vol_Chg_%":10,"Sector":20}
_CF = {"Price":"${:.2f}","Total":"{:.0f}","Fund":"{:.0f}",
       "Tech":"{:.0f}","Vol_Chg_%":"{:+.1f}%"}
_hdr_done = False

def _live_header():
    global _hdr_done
    if _hdr_done: return
    print("\n" + "━"*95)
    print("  📊  LIVE MATCHES  —  each stock printed the moment it passes all filters")
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
chk = download(["AAPL","MSFT","NVDA"], 300)
if not chk:
    print("❌  No data.")
else:
    for s, d in chk.items():
        print(f"  ✅ {s}: {len(d)} bars  ${float(d['Close'].iloc[-1]):.2f}  {d.index[-1].date()}")
    print("\n  Testing .info fetch for AAPL...")
    t0  = time.time()
    fnd = get_fundamentals("AAPL")
    ela = time.time() - t0
    print(f"  ✅ Fund_Score={fnd['Fund_Score']}/50  Flags: {fnd['Fund_Flags']}  ({ela:.1f}s)")
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

# ── Main scan — 2-pass ───────────────────────────────────────
print("━"*65)
print(f"  STEP 3  SCANNING {len(TICKERS)} TICKERS")
print("━"*65)
print("  Pass 1: JMA price-cross + MA stack + volume screening (fast)")
print("  Pass 2: Fundamental fetch for pass-1 stocks only\n")

_hdr_done   = False
results     = []
tech_passes = []
no_data     = 0

batches = [TICKERS[i:i+CFG["batch_size"]]
           for i in range(0, len(TICKERS), CFG["batch_size"])]

# ── PASS 1: Technical signal (no .info calls) ────────────────
with tqdm(total=len(TICKERS), desc="Pass 1 Tech", unit="stk",
          bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]") as pbar:
    for batch in batches:
        data_map = download(batch, CFG["history_days"])
        no_data += len(batch) - len(data_map)
        for sym in batch:
            pbar.update(1)
            if sym not in data_map: continue
            try:
                ts = analyze_jma_stack_cross(sym, data_map[sym])
                if ts is None: continue
                if ts["tech_score"] < CFG["min_tech_score"]: continue
                tech_passes.append({
                    "sym"  : sym,
                    "price": float(data_map[sym]["Close"].iloc[-1]),
                    "ts"   : ts,
                })
            except Exception: pass
        time.sleep(CFG["batch_sleep"])

got = len(TICKERS) - no_data
pct = got / max(len(TICKERS), 1) * 100
print(f"\n  Pass 1 done: {got}/{len(TICKERS)} got data ({pct:.0f}%)")
print(f"  Tech passes: {len(tech_passes)} stocks → fetching fundamentals now\n")

# ── PASS 2: Fundamentals for tech-pass stocks only ──────────────
print("━"*65)
print(f"  PASS 2  FUNDAMENTAL CHECK ({len(tech_passes)} stocks)")
print("━"*65+"\n")

for item in tqdm(tech_passes, desc="Pass 2 Fund", unit="stk"):
    sym   = item["sym"]; price = item["price"]; ts = item["ts"]
    try:
        fund = get_fundamentals(sym)
        time.sleep(CFG["fund_sleep"])

        fs, tscr = fund["Fund_Score"], ts["tech_score"]
        total = fs + tscr
        if total < CFG["min_total_score"]: continue

        result = {
            "Ticker"            : sym,
            "Price"             : round(price, 2),
            "Total"             : total,
            "Fund"              : fs,
            "Tech"              : tscr,
            "Sector"            : fund["Sector"],
            "Company"           : fund["Company"],
            "Industry"          : fund["Industry"],
            "JMA"               : ts["JMA"],
            "EMA8"              : ts["EMA8"],
            "SMA21"             : ts["SMA21"],
            "SMA50"             : ts["SMA50"],
            "SMA150"            : ts["SMA150"],
            "Bars_Since_Cross"  : ts["Bars_Since_Cross"],
            "Vol_Chg_%"         : ts["Vol_Chg_%"],
            "Vol_Today"         : ts["Vol_Today"],
            "Vol_Yesterday"     : ts["Vol_Yesterday"],
            "Tech_Flags"        : ts["tech_reasons"],
            "Rev_Growth_%"      : fund["Rev_Growth_%"],
            "Profit_Margin_%"   : fund["Profit_Margin_%"],
            "ROE_%"             : fund["ROE_%"],
            "PE_Ratio"          : fund["PE_Ratio"],
            "EPS"               : fund["EPS"],
            "Debt_Equity"       : fund["Debt_Equity"],
            "Market_Cap_B"      : fund["Market_Cap_B"],
            "Fund_Flags"        : fund["Fund_Flags"],
            # internals
            "_df"    : ts["_df"],
            "_jma"   : ts["_jma"],
            "_ema8"  : ts["_ema8"],
            "_sma21" : ts["_sma21"],
            "_sma50" : ts["_sma50"],
            "_sma150": ts["_sma150"],
        }
        results.append(result)
        live_print(result)
    except Exception: pass

print(f"\n{'━'*65}")
print(f"  SCAN COMPLETE")
print(f"  Tickers    : {len(TICKERS)}")
print(f"  Got data   : {got}  ({pct:.0f}%)")
print(f"  Tech passes: {len(tech_passes)}")
print(f"  ✅ Matches  : {len(results)}")
print(f"{'━'*65}")

if not results:
    print("\n  No matches. Try relaxing:")
    print("   min_tech_score            10 → 6")
    print("   min_total_score           15 → 5")
    print("   price_jma_cross_lookback   3 → 6")
    print("   require_jma_above_sma150   True → False")
    print("   require_volume_increase    True → False")
    print("   min_price                  2 → 1")
    print("   min_avg_volume         80000 → 50000")

# Sort by total score (always runs, even on empty list)
results.sort(key=lambda x: x["Total"], reverse=True)

# ── Always build df_out and save/email (even if 0 results) ────
ts      = datetime.today().strftime("%Y%m%d_%H%M")
out_dir = os.environ.get("GITHUB_WORKSPACE", os.getcwd())

COLS = [
    "Ticker","Company","Sector","Price",
    "Total","Fund","Tech",
    "Rev_Growth_%","Profit_Margin_%","ROE_%","PE_Ratio","EPS",
    "JMA","EMA8","SMA21","SMA50","SMA150",
    "Bars_Since_Cross","Vol_Chg_%","Vol_Today","Vol_Yesterday",
    "Tech_Flags","Fund_Flags",
]
df_out = pd.DataFrame([{k:v for k,v in r.items() if not k.startswith("_")}
                        for r in results]) if results else pd.DataFrame(columns=COLS)
if not df_out.empty:
    df_out = df_out[[c for c in COLS if c in df_out.columns]]
    df_out.reset_index(drop=True, inplace=True)

FMT = {
    "Price"          : lambda v: f"${v:.2f}",
    "Total"          : lambda v: f"{v:.0f}",
    "Fund"           : lambda v: f"{v:.0f}",
    "Tech"           : lambda v: f"{v:.0f}",
    "Rev_Growth_%"   : lambda v: f"{v:+.1f}%",
    "Profit_Margin_%": lambda v: f"{v:.1f}%",
    "ROE_%"          : lambda v: f"{v:.1f}%",
    "PE_Ratio"       : lambda v: f"{v:.1f}",
    "EPS"            : lambda v: f"${v:.2f}",
    "Market_Cap_B"   : lambda v: f"${v:.2f}B",
    "JMA"            : lambda v: f"${v:.2f}",
    "EMA8"           : lambda v: f"${v:.2f}",
    "SMA21"          : lambda v: f"${v:.2f}",
    "SMA50"          : lambda v: f"${v:.2f}",
    "SMA150"         : lambda v: f"${v:.2f}",
    "Bars_Since_Cross": lambda v: f"{int(v)}d ago",
    "Vol_Chg_%"      : lambda v: f"{v:+.1f}%",
    "Vol_Today"      : lambda v: f"{int(v):,}",
    "Vol_Yesterday"  : lambda v: f"{int(v):,}",
}

def fmt_v(col, val):
    if val is None or (isinstance(val, float) and np.isnan(val)): return "—"
    try:
        if col in FMT: return FMT[col](val)
    except Exception: pass
    return str(val) if str(val) not in ("nan","None","") else "—"

if _IN_NOTEBOOK and results:
    DISP = ["Ticker","Company","Sector","Price",
            "Total","Fund","Tech",
            "Bars_Since_Cross","Vol_Chg_%","JMA","SMA150"]
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
            if col == "Total":
                try:
                    v = float(raw)
                    g = int(min(220, 80 + v*2))
                    sty = f"background:rgb(20,{g},60);color:#fff;font-weight:700;text-align:center"
                except Exception: pass
            elif col == "Fund":
                try:
                    v = float(raw)
                    g = int(min(200, 60 + v*2.5))
                    sty = f"background:rgb(20,{g},80);color:#fff;font-weight:600;text-align:center"
                except Exception: pass
            elif col in ("Vol_Chg_%",):
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
    <span style="color:#f1f5f9;font-size:15px;font-weight:700">JMA Price-Cross + MA Stack</span>
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
    📈 JMA Price-Cross + MA Stack
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
  Total = Fund(0-50) + Tech(0-30) &nbsp;·&nbsp;
  Signal = price closed above JMA from below, JMA above EMA8/SMA21/SMA50/SMA150,
  volume rising vs prior day &nbsp;·&nbsp;
  Bars_Since_Cross = 0d ago means the cross happened on the latest bar
</div>"""

    display_html(header_html + table_html + legend_html)

elif results:
    # ASCII table (CLI/GitHub Actions mode)
    CLI_COLS = ["Ticker","Price","Total","Fund","Tech",
                "Bars_Since_Cross","Vol_Chg_%","Sector"]
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
    tit = f"  JMA Price-Cross + MA Stack   {datetime.today().strftime('%Y-%m-%d')}   {len(df_out)} matches"
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
  Total               Fund(0-50) + Tech(0-30)
  Bars_Since_Cross     bars since price closed above JMA from below
  Vol_Chg_%            today's volume vs yesterday's volume
  ──────────────────────────────────────────────────────""")

# Save
fpath = os.path.join(out_dir, f"jma_stack_cross_{ts}.csv")
df_out.to_csv(fpath, index=False)
print(f"\n  💾 CSV → {fpath}")
tv = os.path.join(out_dir, f"tv_jma_stack_cross_{ts}.txt")
with open(tv,"w") as f:
    f.write(f"###JMA Stack Cross {datetime.today().strftime('%Y-%m-%d')}\n")
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
            for c in ["Ticker","Price","Total","Fund","Tech",
                      "Bars_Since_Cross","Vol_Chg_%"]
        )
        rows_e = ""
        for i, r in enumerate(rl[:50]):
            bg  = "#fff" if i % 2 == 0 else "#f0f9ff"
            ticker = r.get("Ticker","—")
            price  = r.get("Price",0) or 0
            total  = r.get("Total",0) or 0
            fund   = r.get("Fund",0) or 0
            tech   = r.get("Tech",0) or 0
            bcross = r.get("Bars_Since_Cross")
            bcross_disp = f"{int(bcross)}d ago" if bcross is not None else "—"
            volchg = r.get("Vol_Chg_%",0) or 0
            rows_e += (
                f'<tr style="background:{bg}">'
                f'<td style="padding:6px 11px;font-size:12px;font-weight:700">{ticker}</td>'
                f'<td style="padding:6px 11px;font-size:12px">${float(price):.2f}</td>'
                f'<td style="padding:6px 11px;font-size:12px;font-weight:700;'
                f'background:#166534;color:#fff;text-align:center">{float(total):.0f}</td>'
                f'<td style="padding:6px 11px;font-size:12px">{float(fund):.0f}</td>'
                f'<td style="padding:6px 11px;font-size:12px">{float(tech):.0f}</td>'
                f'<td style="padding:6px 11px;font-size:12px;text-align:center;'
                f'color:#a78bfa;font-weight:600">{bcross_disp}</td>'
                f'<td style="padding:6px 11px;font-size:12px;color:'
                f'{"#22c55e" if float(volchg)>=0 else "#ef4444"}">{float(volchg):+.1f}%</td>'
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
  📊 JMA Price-Cross + MA Stack
</h1>
<p style="margin:6px 0 0;color:#94a3b8;font-size:12px">
  {datetime.today().strftime('%Y-%m-%d %H:%M UTC')} &nbsp;·&nbsp;
  {cnt} match{'es' if cnt!=1 else ''} found
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
            f"JMA Price-Cross + MA Stack — {datetime.today().strftime('%Y-%m-%d')}",
            f"{cnt} matches",
            "="*60,
        ]
        if rl:
            for r in rl[:50]:
                ticker = r.get("Ticker","—")
                price  = r.get("Price",0) or 0
                total  = r.get("Total",0) or 0
                bcross = r.get("Bars_Since_Cross")
                bcross_disp = f"{int(bcross)}d ago" if bcross is not None else "—"
                volchg = r.get("Vol_Chg_%",0) or 0
                plain_lines.append(
                    f"{ticker:<7} ${float(price):.2f}  Total:{float(total):.0f}  "
                    f"Cross:{bcross_disp}  VolChg:{float(volchg):+.1f}%"
                )
        else:
            plain_lines.append("No matches today")
        plain_lines.append("\nFull results in CSV attachment.")
        plain_e = "\n".join(plain_lines)

        subj = (f"📊 JMA Price-Cross + MA Stack — {cnt} signal{'s' if cnt!=1 else ''}"
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

# ── Charts for top 5 ──────────────────────────────────────────
if results:
    top = results[:min(5,len(results))]
    fig, axes = plt.subplots(len(top),1,figsize=(15,5*len(top)),facecolor="#0f172a")
    if len(top)==1: axes=[axes]
    for ax, r in zip(axes, top):
        df_p   = r["_df"].tail(120).copy()
        jma_p  = r["_jma"].reindex(df_p.index)
        ema8_p = r["_ema8"].reindex(df_p.index)
        s21_p  = r["_sma21"].reindex(df_p.index)
        s50_p  = r["_sma50"].reindex(df_p.index)
        s150_p = r["_sma150"].reindex(df_p.index)
        ax.set_facecolor("#0f172a")
        ax.plot(df_p.index, df_p["Close"], color="#60a5fa", lw=1.8, label="Price", zorder=5)
        ax.plot(df_p.index, jma_p,  color="#f472b6", lw=1.6, label="JMA(13,40)", zorder=4)
        ax.plot(df_p.index, ema8_p, color="#38bdf8", lw=1.1, ls="--", label="EMA8",   zorder=3)
        ax.plot(df_p.index, s21_p,  color="#34d399", lw=1.1, ls="--", label="SMA21",  zorder=3)
        ax.plot(df_p.index, s50_p,  color="#fbbf24", lw=1.1, ls="-.", label="SMA50",  zorder=3)
        ax.plot(df_p.index, s150_p, color="#f87171", lw=1.0, ls=":",  label="SMA150", zorder=3)
        ax.set_title(
            f"{r['Ticker']}  {r['Company']}  |  ${r['Price']:.2f}  |  "
            f"Score {r['Total']} (F{r['Fund']}+T{r['Tech']})  |  "
            f"Cross: {r['Bars_Since_Cross']}d ago  VolChg: {r['Vol_Chg_%']:+.1f}%",
            color="#e2e8f0", fontsize=9, fontweight="bold", pad=7)
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %y"))
        ax.xaxis.set_major_locator(mdates.MonthLocator())
        ax.tick_params(colors="#94a3b8", labelsize=9)
        for sp in ax.spines.values(): sp.set_edgecolor("#1e3a5f")
        ax.legend(loc="upper left", facecolor="#1e293b", labelcolor="#e2e8f0",
                  fontsize=7, framealpha=0.9, ncol=2)
        ax.grid(color="#1e3a5f", ls="--", lw=0.5, alpha=0.6)
    plt.suptitle(
        f"JMA Price-Cross + MA Stack  ·  "
        f"{datetime.today().strftime('%Y-%m-%d')}",
        color="#60a5fa", fontsize=12, fontweight="bold", y=1.001)
    plt.tight_layout()
    cp = os.path.join(out_dir, f"jma_stack_cross_chart_{ts}.png")
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
  📋 SCORE BREAKDOWN  (80 total)
  Fund  0–50   8 fundamental metrics
  Tech  0–30   cross freshness + MA stack breadth + volume surge

  📋 SIGNAL (all 3 required)
  1) Price closed above JMA from below (fresh cross, within
     price_jma_cross_lookback bars)
  2) JMA sits above EMA8, SMA21, SMA50, AND SMA150
  3) Today's volume > yesterday's volume

  📋 Bars_Since_Cross = 0d ago means the cross happened on the
  latest bar. Vol_Chg_% = today's volume vs yesterday's.

  💡 BEST SETUPS
  Total > 50              elite fundamental + technical combo
  Bars_Since_Cross = 0-1   freshest crosses
  Vol_Chg_% > 50           strong volume confirmation
  Fund > 35                genuinely strong business quality

  ⚙️  TUNE IF 0 RESULTS
  min_tech_score              10 → 6
  min_total_score              15 → 5
  price_jma_cross_lookback      3 → 6   (widen the cross window)
  require_jma_above_sma150   True → False
  require_volume_increase    True → False
  min_price                     2 → 1
  min_avg_volume            80000 → 50000
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
""")
