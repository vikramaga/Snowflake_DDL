# ============================================================
# NASDAQ — Fundamentally + Technically Strong at Support (v2)
# ============================================================
#
# 2-PASS ARCHITECTURE:
#
#  PASS 1 — TECHNICAL + SUPPORT  (fast, no .info calls)
#      Price structure vs SMA50/SMA150/SMA200
#      RSI in healthy range, MACD histogram positive
#      Distance from 52-week high/low
#      Price sitting within support_zone_pct% of a support level
#        (SMA50/150/200, Camarilla S3, swing low, 52w retracement)
#      Cluster score: how many support levels coincide
#
#  PASS 2 — FUNDAMENTALS  (slow, .info calls — only for Pass 1 survivors)
#      Revenue growth, profit margin, ROE, P/E, EPS,
#      debt/equity, current ratio
#
#  FINAL SCORE = Fund (0-50) + Tech (0-30) + Support (0-20) = 100
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

    # ── Technical thresholds ───────────────────────────────────
    "rsi_min"                    : 40,
    "rsi_max"                    : 70,
    "macd_fast"                  : 12,
    "macd_slow"                  : 26,
    "macd_signal"                : 9,
    "pct_below_52w_high"         : 25.0,
    "pct_above_52w_low"          : 30.0,

    # ── Support detection ──────────────────────────────────────
    "support_zone_pct"           : 3.0,
    "support_below_pct"          : 1.0,
    "cluster_tolerance_pct"      : 2.0,
    "swing_low_lookback"         : 60,
    "swing_low_window"           : 10,

    # ── Score gates ─────────────────────────────────────────────
    "min_tech_score"             : 12,    # out of 30
    "min_support_score"          : 5,     # out of 20
    "min_total_score"            : 45,    # out of 100

    # ── Filters ─────────────────────────────────────────────────
    "min_avg_volume"             : 80_000,
    "min_price"                  : 2.0,

    "batch_size"                 : 50,
    "batch_sleep"                 : 1.5,
    "fund_sleep"                   : 0.3,
}

# ── Indicators ───────────────────────────────────────────────
def calc_rsi(close, period=14):
    delta = close.diff()
    gain  = delta.clip(lower=0)
    loss  = -delta.clip(upper=0)
    avg_g = gain.ewm(alpha=1/period, adjust=False).mean()
    avg_l = loss.ewm(alpha=1/period, adjust=False).mean()
    rs    = avg_g / avg_l.replace(0, np.nan)
    return 100 - (100 / (1 + rs))

def calc_macd(close, fast=12, slow=26, signal=9):
    ema_f = close.ewm(span=fast,   adjust=False).mean()
    ema_s = close.ewm(span=slow,   adjust=False).mean()
    macd  = ema_f - ema_s
    sig   = macd.ewm(span=signal,  adjust=False).mean()
    return macd - sig   # histogram only

def cam_s3(high, low, close):
    return close - (high - low) * 1.1 / 4.0

def get_monthly_s3_levels(df):
    df = df.copy()
    df.index = pd.to_datetime(df.index)
    today = pd.Timestamp.today().normalize()
    levels = []
    for offset in [1, 2]:
        p   = today.to_period("M") - offset
        sub = df[df.index.to_period("M") == p]
        if len(sub) >= 5:
            levels.append(round(cam_s3(
                float(sub["High"].max()),
                float(sub["Low"].min()),
                float(sub["Close"].iloc[-1])), 2))
    return levels

def find_swing_low(low_series, lookback=60, window=10):
    vals = low_series.tail(lookback).values.astype(float)
    n    = len(vals)
    for i in range(n - window - 1, window - 1, -1):
        lo = max(0, i - window)
        hi = min(n, i + window + 1)
        if vals[i] == np.min(vals[lo:hi]):
            return round(float(vals[i]), 4)
    return None

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

# ── Technical + support analysis ────────────────────────────────
def analyze_tech_support(sym, df):
    """
    Returns dict with tech_score, support_score, and details.
    Returns None only if basic filters fail.
    """
    n       = len(df)
    price   = float(df["Close"].iloc[-1])
    avg_vol = float(df["Volume"].tail(20).mean())

    if price   < CFG["min_price"]:      return None
    if avg_vol < CFG["min_avg_volume"]: return None

    sma50  = df["Close"].rolling(50).mean()
    sma150 = df["Close"].rolling(150).mean()
    sma200 = df["Close"].rolling(200).mean()
    cs50   = float(sma50.iloc[-1])
    cs150  = float(sma150.iloc[-1])
    cs200  = float(sma200.iloc[-1])
    if any(np.isnan([cs50, cs150, cs200])): return None

    rsi_s   = calc_rsi(df["Close"])
    macdh_s = calc_macd(df["Close"], CFG["macd_fast"], CFG["macd_slow"], CFG["macd_signal"])
    cur_rsi = float(rsi_s.iloc[-1])   if not np.isnan(rsi_s.iloc[-1])   else 50
    cur_mh  = float(macdh_s.iloc[-1]) if not np.isnan(macdh_s.iloc[-1]) else 0

    w52    = min(252, n)
    hi52   = float(df["Close"].tail(w52).max())
    lo52   = float(df["Close"].tail(w52).min())
    pct_hi = (hi52 - price) / hi52 * 100
    pct_lo = (price - lo52) / lo52 * 100

    sma50_prev   = float(sma50.iloc[-6]) if not np.isnan(sma50.iloc[-6]) else cs50
    sma50_rising = cs50 > sma50_prev

    # ── Technical score (0-30) ────────────────────────────────
    ts = 0
    tr = []
    if price > cs50 > cs150 > cs200:
        ts += 10; tr.append("FullStack")
    elif price > cs50 > cs150:
        ts += 6;  tr.append("PartStack")
    elif price > cs50:
        ts += 3;  tr.append("AboveSMA50")
    if sma50_rising:
        ts += 4; tr.append("SMA50↑")
    if CFG["rsi_min"] <= cur_rsi <= CFG["rsi_max"]:
        ts += 5; tr.append(f"RSI{cur_rsi:.0f}")
    if pct_hi <= CFG["pct_below_52w_high"]:
        ts += 5; tr.append("Near52Hi")
    if pct_lo >= CFG["pct_above_52w_low"]:
        ts += 3; tr.append("Strong52Lo")
    if cur_mh > 0:
        ts += 3; tr.append("MACD+")
    ts = min(30, ts)

    # ── Support levels ────────────────────────────────────────
    candidates = {"SMA50": cs50, "SMA150": cs150, "SMA200": cs200}
    for i, lv in enumerate(get_monthly_s3_levels(df), 1):
        candidates[f"Cam_S3_M{i}"] = lv
    sl = find_swing_low(df["Low"], CFG["swing_low_lookback"], CFG["swing_low_window"])
    if sl: candidates["SwingLow"] = sl
    candidates["52W_Retrace"] = hi52 * 0.75

    zone  = CFG["support_zone_pct"]  / 100
    below = CFG["support_below_pct"] / 100
    active = {}
    for name, level in candidates.items():
        if level <= 0: continue
        dist = (price - level) / level * 100
        if -CFG["support_below_pct"] <= dist <= CFG["support_zone_pct"]:
            active[name] = {"level": round(level, 2), "dist": round(dist, 2)}

    cluster_tol  = CFG["cluster_tolerance_pct"] / 100
    best_cluster = 1
    best_level   = price
    best_names   = []

    for name_i, data_i in active.items():
        cluster = [name_i]
        for name_j, data_j in active.items():
            if name_i == name_j: continue
            if abs(data_j["level"] - data_i["level"]) / data_i["level"] <= cluster_tol:
                cluster.append(name_j)
        if len(cluster) > best_cluster:
            best_cluster = len(cluster)
            best_level   = data_i["level"]
            best_names   = cluster

    if not active:
        ss = 0
        best_level = min(candidates.values()) if candidates else price
        best_names = []
    else:
        ss = min(20, best_cluster * 5 + len(active) * 2)
        if not best_names:
            best_names = list(active.keys())[:1]
            best_level = active[best_names[0]]["level"]

    dist_best = (price - best_level) / best_level * 100 if best_level > 0 else 0

    return {
        "tech_score"    : ts,
        "tech_reasons"  : " | ".join(tr),
        "support_score" : ss,
        "active_sup"    : active,
        "best_level"    : round(best_level, 2),
        "best_names"    : " + ".join(best_names),
        "cluster"       : best_cluster,
        "dist_best"     : round(dist_best, 2),
        "RSI"           : round(cur_rsi, 1),
        "MACD_Hist"     : round(cur_mh, 4),
        "SMA50"         : round(cs50, 2),
        "SMA150"        : round(cs150, 2),
        "SMA200"        : round(cs200, 2),
        "Pct_from_Hi52" : round(pct_hi, 1),
        "Pct_from_Lo52" : round(pct_lo, 1),
        "_df"           : df,
        "_s50"          : sma50,
        "_s150"         : sma150,
        "_s200"         : sma200,
        "_active_sup"   : active,
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
LIVE_COLS = ["Ticker","Price","Total","Fund","Tech","Supp",
             "Cluster","Dist_%","Best_Support_Name","Sector"]
_CW = {"Ticker":8,"Price":10,"Total":7,"Fund":6,"Tech":6,"Supp":6,
       "Cluster":8,"Dist_%":8,"Best_Support_Name":22,"Sector":20}
_CF = {"Price":"${:.2f}","Total":"{:.0f}","Fund":"{:.0f}",
       "Tech":"{:.0f}","Supp":"{:.0f}","Dist_%":"{:+.2f}%"}
_hdr_done = False

def _live_header():
    global _hdr_done
    if _hdr_done: return
    print("\n" + "━"*105)
    print("  📊  LIVE MATCHES  —  each stock printed the moment it passes all filters")
    print("━"*105)
    h = "".join(f"  {c:<{_CW.get(c,12)}}" for c in LIVE_COLS)
    print(h)
    print("  " + "─"*103)
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
print("  Pass 1: Technical + Support screening (fast)")
print("  Pass 2: Fundamental fetch for pass-1 stocks only\n")

_hdr_done   = False
results     = []
tech_passes = []
no_data     = 0

batches = [TICKERS[i:i+CFG["batch_size"]]
           for i in range(0, len(TICKERS), CFG["batch_size"])]

# ── PASS 1: Technical + Support (no .info calls) ────────────────
with tqdm(total=len(TICKERS), desc="Pass 1 Tech", unit="stk",
          bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]") as pbar:
    for batch in batches:
        data_map = download(batch, CFG["history_days"])
        no_data += len(batch) - len(data_map)
        for sym in batch:
            pbar.update(1)
            if sym not in data_map: continue
            try:
                ts = analyze_tech_support(sym, data_map[sym])
                if ts is None: continue
                if ts["tech_score"]    < CFG["min_tech_score"]:    continue
                if ts["support_score"] < CFG["min_support_score"]: continue
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

        fs, tscr, sscr = fund["Fund_Score"], ts["tech_score"], ts["support_score"]
        total = fs + tscr + sscr
        if total < CFG["min_total_score"]: continue

        result = {
            "Ticker"            : sym,
            "Price"             : round(price, 2),
            "Total"             : total,
            "Fund"              : fs,
            "Tech"              : tscr,
            "Supp"              : sscr,
            "Cluster"           : ts["cluster"],
            "Dist_%"            : ts["dist_best"],
            "Best_Support_Name" : ts["best_names"],
            "Sector"            : fund["Sector"],
            "Company"           : fund["Company"],
            "Industry"          : fund["Industry"],
            "SMA50"             : ts["SMA50"],
            "SMA150"            : ts["SMA150"],
            "SMA200"            : ts["SMA200"],
            "RSI"               : ts["RSI"],
            "MACD_Hist"         : ts["MACD_Hist"],
            "Pct_from_52Hi"     : ts["Pct_from_Hi52"],
            "Pct_above_52Lo"    : ts["Pct_from_Lo52"],
            "Best_Support_$"    : ts["best_level"],
            "Active_Supports"   : str({k:v["level"] for k,v in ts["active_sup"].items()}),
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
            "_df"   : ts["_df"],
            "_s50"  : ts["_s50"],
            "_s150" : ts["_s150"],
            "_s200" : ts["_s200"],
            "_asupp": ts["_active_sup"],
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
    print("   min_tech_score      12 → 10")
    print("   min_support_score    5 → 3")
    print("   min_total_score     45 → 35")
    print("   support_zone_pct     3 → 5")
    print("   min_price            2 → 1")
    print("   min_avg_volume    80000 → 50000")

# Sort by total score (always runs, even on empty list)
results.sort(key=lambda x: x["Total"], reverse=True)

# ── Always build df_out and save/email (even if 0 results) ────
ts      = datetime.today().strftime("%Y%m%d_%H%M")
out_dir = os.environ.get("GITHUB_WORKSPACE", os.getcwd())

COLS = [
    "Ticker","Company","Sector","Price",
    "Total","Fund","Tech","Supp",
    "Rev_Growth_%","Profit_Margin_%","ROE_%","PE_Ratio","EPS",
    "RSI","MACD_Hist","Pct_from_52Hi",
    "Best_Support_$","Best_Support_Name","Cluster","Dist_%",
    "Active_Supports","Tech_Flags","Fund_Flags",
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
    "Supp"           : lambda v: f"{v:.0f}",
    "Rev_Growth_%"   : lambda v: f"{v:+.1f}%",
    "Profit_Margin_%": lambda v: f"{v:.1f}%",
    "ROE_%"          : lambda v: f"{v:.1f}%",
    "PE_Ratio"       : lambda v: f"{v:.1f}",
    "EPS"            : lambda v: f"${v:.2f}",
    "RSI"            : lambda v: f"{v:.1f}",
    "MACD_Hist"      : lambda v: f"{v:.4f}",
    "Pct_from_52Hi"  : lambda v: f"{v:.1f}%",
    "Best_Support_$" : lambda v: f"${v:.2f}",
    "Dist_%"         : lambda v: f"{v:+.2f}%",
    "Market_Cap_B"   : lambda v: f"${v:.2f}B",
}

def fmt_v(col, val):
    if val is None or (isinstance(val, float) and np.isnan(val)): return "—"
    try:
        if col in FMT: return FMT[col](val)
    except Exception: pass
    return str(val) if str(val) not in ("nan","None","") else "—"

if _IN_NOTEBOOK and results:
    DISP = ["Ticker","Company","Sector","Price",
            "Total","Fund","Tech","Supp",
            "Best_Support_Name","Dist_%","Cluster",
            "RSI","MACD_Hist","Pct_from_52Hi"]
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
                    g = int(min(220, 80 + v*1.4))
                    sty = f"background:rgb(20,{g},60);color:#fff;font-weight:700;text-align:center"
                except Exception: pass
            elif col == "Fund":
                try:
                    v = float(raw)
                    g = int(min(200, 60 + v*2.5))
                    sty = f"background:rgb(20,{g},80);color:#fff;font-weight:600;text-align:center"
                except Exception: pass
            elif col == "Cluster":
                try:
                    v = int(float(raw))
                    if v >= 3: sty = "color:#22c55e;font-weight:800;text-align:center;font-size:14px"
                    elif v == 2: sty = "color:#a78bfa;font-weight:700;text-align:center"
                    else: sty = "text-align:center;color:#94a3b8"
                except Exception: pass
            elif col in ("Dist_%","Rev_Growth_%","MACD_Hist"):
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
    <span style="color:#f1f5f9;font-size:15px;font-weight:700">Fundamentally + Technically Strong at Support</span>
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
    📈 Fundamentally + Technically Strong at Support
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
  Total = Fund(0-50) + Tech(0-30) + Supp(0-20) &nbsp;·&nbsp;
  Cluster = how many support levels coincide (3+ = strong confluence) &nbsp;·&nbsp;
  Dist_% = distance from best support (near 0 = sitting on support now)
</div>"""

    display_html(header_html + table_html + legend_html)

elif results:
    # ASCII table (CLI/GitHub Actions mode)
    CLI_COLS = ["Ticker","Price","Total","Fund","Tech","Supp",
                "Cluster","Dist_%","Best_Support_Name"]
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
    tit = f"  Fund + Tech Strong at Support   {datetime.today().strftime('%Y-%m-%d')}   {len(df_out)} matches"
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
  Total               Fund(0-50) + Tech(0-30) + Supp(0-20)
  Cluster              how many support levels coincide
  Dist_%               distance from best support level
  Best_Support_Name    which support(s) price is testing
  ──────────────────────────────────────────────────────""")

# Save
fpath = os.path.join(out_dir, f"fund_tech_support_{ts}.csv")
df_out.to_csv(fpath, index=False)
print(f"\n  💾 CSV → {fpath}")
tv = os.path.join(out_dir, f"tv_fund_tech_support_{ts}.txt")
with open(tv,"w") as f:
    f.write(f"###Fund+Tech Support {datetime.today().strftime('%Y-%m-%d')}\n")
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
            for c in ["Ticker","Price","Total","Fund","Tech","Supp",
                      "Cluster","Dist_%"]
        )
        rows_e = ""
        for i, r in enumerate(rl[:50]):
            bg  = "#fff" if i % 2 == 0 else "#f0f9ff"
            ticker = r.get("Ticker","—")
            price  = r.get("Price",0) or 0
            total  = r.get("Total",0) or 0
            fund   = r.get("Fund",0) or 0
            tech   = r.get("Tech",0) or 0
            supp   = r.get("Supp",0) or 0
            clust  = r.get("Cluster",0) or 0
            dist   = r.get("Dist_%",0) or 0
            rows_e += (
                f'<tr style="background:{bg}">'
                f'<td style="padding:6px 11px;font-size:12px;font-weight:700">{ticker}</td>'
                f'<td style="padding:6px 11px;font-size:12px">${float(price):.2f}</td>'
                f'<td style="padding:6px 11px;font-size:12px;font-weight:700;'
                f'background:#166534;color:#fff;text-align:center">{float(total):.0f}</td>'
                f'<td style="padding:6px 11px;font-size:12px">{float(fund):.0f}</td>'
                f'<td style="padding:6px 11px;font-size:12px">{float(tech):.0f}</td>'
                f'<td style="padding:6px 11px;font-size:12px">{float(supp):.0f}</td>'
                f'<td style="padding:6px 11px;font-size:12px;text-align:center">{int(clust)}</td>'
                f'<td style="padding:6px 11px;font-size:12px;color:'
                f'{"#22c55e" if float(dist)>=0 else "#ef4444"}">{float(dist):+.2f}%</td>'
                f'</tr>'
            )

        no_results_msg = ""
        if cnt == 0:
            no_results_msg = (
                '<tr><td colspan="8" style="padding:20px;text-align:center;'
                'color:#64748b;font-size:13px">No matches found today — '
                'market conditions did not trigger the pattern</td></tr>'
            )

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
  📊 Fundamentally + Technically Strong at Support
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
            f"Fundamentally + Technically Strong at Support — {datetime.today().strftime('%Y-%m-%d')}",
            f"{cnt} matches",
            "="*60,
        ]
        if rl:
            for r in rl[:50]:
                ticker = r.get("Ticker","—")
                price  = r.get("Price",0) or 0
                total  = r.get("Total",0) or 0
                supp_name = r.get("Best_Support_Name","—")
                plain_lines.append(
                    f"{ticker:<7} ${float(price):.2f}  Total:{float(total):.0f}  "
                    f"Support:{supp_name}"
                )
        else:
            plain_lines.append("No matches today")
        plain_lines.append("\nFull results in CSV attachment.")
        plain_e = "\n".join(plain_lines)

        subj = (f"📊 Fund+Tech Support — {cnt} signal{'s' if cnt!=1 else ''}"
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
        df_p  = r["_df"].tail(120).copy()
        s50   = r["_s50"].reindex(df_p.index)
        s150  = r["_s150"].reindex(df_p.index)
        s200  = r["_s200"].reindex(df_p.index)
        asupp = r["_asupp"]
        ax.set_facecolor("#0f172a")
        ax.plot(df_p.index, df_p["Close"], color="#60a5fa", lw=1.8, label="Price", zorder=4)
        ax.plot(df_p.index, s50,  color="#fbbf24", lw=1.3, ls="--", label="SMA50",  zorder=3)
        ax.plot(df_p.index, s150, color="#f87171", lw=1.1, ls="-.", label="SMA150", zorder=3)
        ax.plot(df_p.index, s200, color="#a78bfa", lw=1.0, ls=":",  label="SMA200", zorder=3)
        clrs = ["#34d399","#fde68a","#fb923c","#38bdf8","#e879f9","#a3e635"]
        for ci, (sn, sd) in enumerate(asupp.items()):
            ax.axhline(sd["level"], color=clrs[ci%len(clrs)], lw=1.3,
                       ls="--", alpha=0.8, label=f"{sn} ${sd['level']:.2f}", zorder=2)
        bs = r["Best_Support_$"]
        ax.axhspan(bs*0.97, bs*1.03, alpha=0.07, color="#34d399", zorder=1)
        ax.set_title(
            f"{r['Ticker']}  {r['Company']}  |  ${r['Price']:.2f}  |  "
            f"Score {r['Total']} (F{r['Fund']}+T{r['Tech']}+S{r['Supp']})  |  "
            f"Support: {r['Best_Support_Name']} ${r['Best_Support_$']:.2f}  "
            f"({r['Dist_%']:+.1f}%)  Cluster={r['Cluster']}",
            color="#e2e8f0", fontsize=9, fontweight="bold", pad=7)
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %y"))
        ax.xaxis.set_major_locator(mdates.MonthLocator())
        ax.tick_params(colors="#94a3b8", labelsize=9)
        for sp in ax.spines.values(): sp.set_edgecolor("#1e3a5f")
        ax.legend(loc="upper left", facecolor="#1e293b", labelcolor="#e2e8f0",
                  fontsize=7, framealpha=0.9, ncol=2)
        ax.grid(color="#1e3a5f", ls="--", lw=0.5, alpha=0.6)
    plt.suptitle(
        f"Fundamental + Technical Strength at Key Support  ·  "
        f"{datetime.today().strftime('%Y-%m-%d')}",
        color="#60a5fa", fontsize=12, fontweight="bold", y=1.001)
    plt.tight_layout()
    cp = os.path.join(out_dir, f"fund_tech_chart_{ts}.png")
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
  📋 SCORE BREAKDOWN  (100 total)
  Fund  0–50   8 fundamental metrics
  Tech  0–30   6 technical conditions
  Supp  0–20   support cluster strength

  📋 SUPPORT TYPES TESTED
  SMA50 / SMA150 / SMA200
  Camarilla S3 (prev 2 months)
  Recent Swing Low  |  52W 25% Retracement

  📋 Cluster = how many support levels coincide at same zone
  1 = single support  |  2 = dual  |  3+ = triple confluence

  💡 BEST SETUPS
  Total > 70     elite fundamental + technical + support
  Cluster >= 3   strongest multi-level support
  Dist_% near 0  price sitting right on support now
  Fund > 35      genuinely strong business quality

  ⚙️  TUNE IF 0 RESULTS
  min_tech_score      12 → 10
  min_support_score    5 → 3
  min_total_score     45 → 35
  support_zone_pct     3 → 5
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
""")
