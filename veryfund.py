# ============================================================
# NASDAQ — Very Strong Fundamentals + SMA50/EMA20 Support
# ============================================================
#
# 2-PASS ARCHITECTURE:
#
#  PASS 1 — TECHNICAL STRUCTURE + SUPPORT  (fast, no .info calls)
#      C1  Price ABOVE SMA50 AND ABOVE SMA150 (bull structure)
#      C2  Price is currently TAKING SUPPORT at SMA50 OR EMA20
#          = candle LOW came within support_zone_pct% of
#            SMA50 or EMA20, price closed back above it
#      RSI healthy, MACD not deeply negative
#
#  PASS 2 — VERY STRONG FUNDAMENTALS  (slow, .info calls —
#           only for Pass 1 survivors)
#      Much stricter thresholds than a general "good fundamentals"
#      scanner — this is a HIGH BAR quality filter:
#        Revenue growth   > 10%
#        Profit margin    > 15%
#        ROE              > 15%
#        Debt/Equity      < 1.0
#        Current ratio    > 1.5
#        EPS              > 0  (profitable)
#        Earnings growth  > 10%
#      Must pass at least min_fund_hits of these 7 (default 5)
#
#  FINAL SCORE = Fund (0-50, weighted for strength) +
#                Tech (0-30) + Support (0-20) = 100
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
import matplotlib.patches as mpatches

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
    "history_days"                : 300,

    # ── MA periods ────────────────────────────────────────────
    "ema20_period"                 : 20,
    "sma50_period"                 : 50,
    "sma150_period"                : 150,

    # ── C1: Bull structure ──────────────────────────────────────
    # Price must be above BOTH SMA50 and SMA150
    "require_above_sma50"          : True,
    "require_above_sma150"         : True,

    # ── C2: Taking support at SMA50 or EMA20 ──────────────────
    # Candle LOW must come within this % of SMA50 or EMA20
    "support_zone_pct"              : 3.0,
    # Look back this many bars for the support touch
    "support_lookback"              : 10,
    # Price must have closed back ABOVE the support level
    # after touching it (confirms the bounce)
    "require_close_above_support"   : True,

    # ── Very strong fundamentals — HIGH BAR thresholds ─────────
    "min_revenue_growth_pct"        : 10.0,   # > 10% YoY
    "min_profit_margin_pct"         : 15.0,   # > 15%
    "min_roe_pct"                   : 15.0,   # > 15%
    "max_debt_to_equity"            : 1.0,    # < 1.0
    "min_current_ratio"             : 1.5,    # > 1.5
    "min_eps"                       : 0.0,    # profitable
    "min_earnings_growth_pct"       : 10.0,   # > 10% YoY
    # Must pass at least this many of the 7 fundamental checks
    "min_fund_hits"                 : 5,
    "fund_sleep"                    : 0.3,

    # ── RSI / MACD ────────────────────────────────────────────
    "rsi_min"                       : 40,
    "rsi_max"                       : 75,

    # ── Filters ───────────────────────────────────────────────
    "min_avg_volume"                : 100_000,
    "min_price"                     : 3.0,

    "batch_size"                    : 50,
    "batch_sleep"                   : 1.5,
}

# ── Indicators ───────────────────────────────────────────────
def calc_ema(series, period):
    return series.ewm(span=period, adjust=False).mean()

def calc_rsi(close, period=14):
    delta = close.diff()
    gain  = delta.clip(lower=0)
    loss  = -delta.clip(upper=0)
    avg_g = gain.ewm(alpha=1/period, adjust=False).mean()
    avg_l = loss.ewm(alpha=1/period, adjust=False).mean()
    rs    = avg_g / avg_l.replace(0, np.nan)
    return 100 - (100 / (1 + rs))

def calc_macd(close, fast=12, slow=26, signal=9):
    ema_f = calc_ema(close, fast)
    ema_s = calc_ema(close, slow)
    macd  = ema_f - ema_s
    sig   = calc_ema(macd, signal)
    return macd, sig, macd - sig

# ── Very strong fundamental data fetch ──────────────────────────
def get_fundamentals(sym):
    """
    Fetch fundamentals via yf.Ticker(sym).info
    Applies HIGH-BAR "very strong" thresholds.
    Never returns None — always returns a dict.
    """
    empty = {
        "Company":"—", "Sector":"—", "Industry":"—",
        "Rev_Growth_%":None, "Profit_Margin_%":None,
        "ROE_%":None, "PE_Ratio":None, "EPS":None,
        "Debt_Equity":None, "Current_Ratio":None,
        "Earnings_Growth_%":None, "Market_Cap_B":None,
        "Fund_Score":0, "Fund_Hits":0, "Fund_Flags":"No Data",
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

        rev_growth    = safe("revenueGrowth")
        profit_margin = safe("profitMargins")
        roe           = safe("returnOnEquity")
        pe            = safe("trailingPE")
        eps           = safe("trailingEps")
        de            = safe("debtToEquity")
        cr            = safe("currentRatio")
        earn_growth   = safe("earningsGrowth")
        mktcap        = safe("marketCap")
        company       = info.get("longName", sym)
        sector        = info.get("sector", "—")
        industry      = info.get("industry", "—")

        rg_pct = rev_growth    * 100 if rev_growth    is not None else None
        pm_pct = profit_margin * 100 if profit_margin is not None else None
        roe_pct= roe           * 100 if roe           is not None else None
        eg_pct = earn_growth   * 100 if earn_growth   is not None else None
        # yfinance sometimes returns debtToEquity as a raw ratio *100
        de_val = de / 100.0 if (de is not None and de > 10) else de

        # ── 7 HIGH-BAR "very strong" checks ────────────────────
        hits  = 0
        score = 0
        flags = []

        if rg_pct is not None and rg_pct > CFG["min_revenue_growth_pct"]:
            hits += 1; score += 8; flags.append(f"RevG{rg_pct:+.0f}%")
        if pm_pct is not None and pm_pct > CFG["min_profit_margin_pct"]:
            hits += 1; score += 8; flags.append(f"Mgn{pm_pct:.0f}%")
        if roe_pct is not None and roe_pct > CFG["min_roe_pct"]:
            hits += 1; score += 8; flags.append(f"ROE{roe_pct:.0f}%")
        if de_val is not None and de_val < CFG["max_debt_to_equity"]:
            hits += 1; score += 7; flags.append(f"DE{de_val:.2f}")
        if cr is not None and cr > CFG["min_current_ratio"]:
            hits += 1; score += 7; flags.append(f"CR{cr:.1f}")
        if eps is not None and eps > CFG["min_eps"]:
            hits += 1; score += 6; flags.append(f"EPS${eps:.2f}")
        if eg_pct is not None and eg_pct > CFG["min_earnings_growth_pct"]:
            hits += 1; score += 6; flags.append(f"EarnG{eg_pct:+.0f}%")

        return {
            "Company"           : company,
            "Sector"            : sector,
            "Industry"          : industry,
            "Rev_Growth_%"      : round(rg_pct, 1)  if rg_pct  is not None else None,
            "Profit_Margin_%"   : round(pm_pct, 1)  if pm_pct  is not None else None,
            "ROE_%"             : round(roe_pct, 1) if roe_pct is not None else None,
            "PE_Ratio"          : round(pe, 1)      if pe      is not None else None,
            "EPS"               : round(eps, 2)     if eps     is not None else None,
            "Debt_Equity"       : round(de_val, 2)  if de_val  is not None else None,
            "Current_Ratio"     : round(cr, 2)      if cr      is not None else None,
            "Earnings_Growth_%" : round(eg_pct, 1)  if eg_pct  is not None else None,
            "Market_Cap_B"      : round(mktcap/1e9,2) if mktcap is not None else None,
            "Fund_Score"        : min(50, score),
            "Fund_Hits"         : hits,
            "Fund_Flags"        : " ".join(flags) if flags else "—",
        }
    except Exception:
        return empty

# ── Technical structure + support detection ──────────────────────
def analyze_tech_support(sym, df):
    """
    C1: price above SMA50 AND SMA150
    C2: candle LOW touched SMA50 or EMA20 within support_lookback
        bars, and price is now closed back above that level
    """
    n       = len(df)
    price   = float(df["Close"].iloc[-1])
    avg_vol = float(df["Volume"].tail(20).mean())

    if price   < CFG["min_price"]:      return None
    if avg_vol < CFG["min_avg_volume"]: return None

    ema20_s  = calc_ema(df["Close"], CFG["ema20_period"])
    sma50_s  = df["Close"].rolling(CFG["sma50_period"]).mean()
    sma150_s = df["Close"].rolling(CFG["sma150_period"]).mean()
    rsi_s    = calc_rsi(df["Close"])
    macd_s, _, hist_s = calc_macd(df["Close"])

    cur_ema20  = float(ema20_s.iloc[-1])
    cur_sma50  = float(sma50_s.iloc[-1])
    cur_sma150 = float(sma150_s.iloc[-1])
    cur_rsi    = float(rsi_s.iloc[-1]) if not np.isnan(rsi_s.iloc[-1]) else 50
    cur_hist   = float(hist_s.iloc[-1]) if not np.isnan(hist_s.iloc[-1]) else 0

    if any(np.isnan([cur_ema20, cur_sma50, cur_sma150])): return None
    if not (CFG["rsi_min"] <= cur_rsi <= CFG["rsi_max"]): return None

    # ─────────────────────────────────────────────────────────
    # C1: BULL STRUCTURE — price above SMA50 AND SMA150
    # ─────────────────────────────────────────────────────────
    if CFG["require_above_sma50"]  and price < cur_sma50:  return None
    if CFG["require_above_sma150"] and price < cur_sma150: return None

    # ─────────────────────────────────────────────────────────
    # C2: TAKING SUPPORT AT SMA50 OR EMA20
    # Search the last support_lookback bars for a touch
    # ─────────────────────────────────────────────────────────
    lb   = CFG["support_lookback"]
    zone = CFG["support_zone_pct"] / 100

    support_sma50 = False
    support_ema20 = False
    support_bar   = None
    support_level = None
    support_name  = None
    best_depth    = float("inf")

    search_start = max(0, n - lb)
    for i in range(search_start, n):
        lo    = float(df["Low"].iloc[i])
        s50_i = float(sma50_s.iloc[i]) if not np.isnan(sma50_s.iloc[i]) else np.nan
        e20_i = float(ema20_s.iloc[i]) if not np.isnan(ema20_s.iloc[i]) else np.nan

        if not np.isnan(s50_i) and s50_i > 0:
            dist50 = abs(lo - s50_i) / s50_i
            if dist50 <= zone:
                support_sma50 = True
                if dist50 < best_depth:
                    best_depth    = dist50
                    support_bar   = i
                    support_level = s50_i
                    support_name  = "SMA50"

        if not np.isnan(e20_i) and e20_i > 0:
            dist20 = abs(lo - e20_i) / e20_i
            if dist20 <= zone:
                support_ema20 = True
                if dist20 < best_depth:
                    best_depth    = dist20
                    support_bar   = i
                    support_level = e20_i
                    support_name  = "EMA20"

    if not support_sma50 and not support_ema20:
        return None   # no support touch found

    # Confirm price has closed back above the support level
    if CFG["require_close_above_support"] and price < support_level:
        return None

    if support_sma50 and support_ema20:
        support_label = "SMA50 + EMA20"
    else:
        support_label = support_name

    bars_since_support = n - 1 - support_bar if support_bar is not None else 0
    support_depth_pct  = round(best_depth * 100, 2)

    # ── Technical score (0-30) ────────────────────────────────
    ts = 0
    tr = []
    if price > cur_sma50 > cur_sma150:
        ts += 10; tr.append("FullStack")
    elif price > cur_sma50:
        ts += 6;  tr.append("AboveSMA50")
    if cur_hist > 0:
        ts += 6; tr.append("MACD+")
    if CFG["rsi_min"] <= cur_rsi <= 60:
        ts += 5; tr.append("RSI-Healthy")
    elif cur_rsi <= CFG["rsi_max"]:
        ts += 3; tr.append("RSI-OK")
    ts = min(30, ts)

    # ── Support score (0-20) ──────────────────────────────────
    ss = 0
    if support_sma50 and support_ema20:
        ss += 14   # both levels = strongest confluence
    else:
        ss += 8
    ss += max(0, 6 - int(support_depth_pct * 2))   # tighter touch = better
    ss = min(20, ss)

    return {
        "tech_score"          : ts,
        "tech_reasons"        : " | ".join(tr),
        "support_score"       : ss,
        "support_label"       : support_label,
        "support_level"       : round(support_level, 2) if support_level else None,
        "support_depth_%"     : support_depth_pct,
        "bars_since_support"  : bars_since_support,
        "support_sma50"       : "✅" if support_sma50 else "—",
        "support_ema20"       : "✅" if support_ema20 else "—",
        "RSI"                 : round(cur_rsi, 1),
        "MACD_Hist"           : round(cur_hist, 4),
        "EMA20"               : round(cur_ema20, 2),
        "SMA50"               : round(cur_sma50, 2),
        "SMA150"              : round(cur_sma150, 2),
        "Dist_SMA50_%"        : round((price-cur_sma50)/cur_sma50*100, 2),
        "Dist_SMA150_%"       : round((price-cur_sma150)/cur_sma150*100, 2),
        "_df"    : df,
        "_ema20" : ema20_s,
        "_sma50" : sma50_s,
        "_sma150": sma150_s,
        "_support_bar": support_bar,
    }

# ── Download ──────────────────────────────────────────────────
def _clean(df, min_bars=160):
    if df is None or df.empty: return None
    need = [c for c in ["Open","High","Low","Close","Volume"] if c in df.columns]
    if not all(c in need for c in ["Open","High","Low","Close","Volume"]): return None
    df = df[need].copy()
    df.index = pd.to_datetime(df.index)
    if hasattr(df.index, "tz") and df.index.tz:
        df.index = df.index.tz_localize(None)
    df.dropna(subset=["Open","High","Low","Close","Volume"], inplace=True)
    return df if (len(df) >= min_bars and float(df["Close"].iloc[-1]) > 0) else None

def download(symbols, days):
    end = datetime.today(); start = end - timedelta(days=days)
    out = {}
    try:
        raw = yf.download(symbols,
                          start=start.strftime("%Y-%m-%d"),
                          end=end.strftime("%Y-%m-%d"),
                          group_by="ticker", auto_adjust=True,
                          actions=False, threads=True, progress=False)
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
             "Fund_Hits","Support_Label","Support_Depth_%","Sector"]
_CW = {"Ticker":8,"Price":10,"Total":7,"Fund":6,"Tech":6,"Supp":6,
       "Fund_Hits":10,"Support_Label":16,"Support_Depth_%":16,"Sector":20}
_CF = {"Price":"${:.2f}","Total":"{:.0f}","Fund":"{:.0f}",
       "Tech":"{:.0f}","Supp":"{:.0f}","Fund_Hits":"{:.0f}/7",
       "Support_Depth_%":"{:.2f}%"}
_hdr_done = False

def _live_header():
    global _hdr_done
    if _hdr_done: return
    print("\n" + "━"*115)
    print("  📊  LIVE MATCHES  —  Very Strong Fundamentals + SMA50/EMA20 Support")
    print("━"*115)
    h = "".join(f"  {c:<{_CW.get(c,12)}}" for c in LIVE_COLS)
    print(h)
    print("  " + "─"*113)
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
chk = download(["AAPL","MSFT","NVDA"], 250)
if not chk:
    print("❌  No data.")
else:
    for s, d in chk.items():
        print(f"  ✅ {s}: {len(d)} bars  ${float(d['Close'].iloc[-1]):.2f}  {d.index[-1].date()}")
    print("\n  Testing .info fetch for AAPL (very strong fund check)...")
    t0  = time.time()
    fnd = get_fundamentals("AAPL")
    ela = time.time() - t0
    print(f"  ✅ Fund_Hits={fnd['Fund_Hits']}/7  Fund_Score={fnd['Fund_Score']}/50  "
          f"Flags: {fnd['Fund_Flags']}  ({ela:.1f}s)")
print()

# ── Diagnostic ────────────────────────────────────────────────
print("━"*65)
print("  STEP 2  DIAGNOSTIC (10 sample stocks)")
print("━"*65+"\n")

DIAG = ["AAPL","MSFT","NVDA","AMD","PLTR","META","CRWD","AVGO","DDOG","MU"]
diag_data = download(DIAG, CFG["history_days"])
print(f"  Downloaded {len(diag_data)}/{len(DIAG)}\n")
print(f"  {'SYM':<7} {'PRICE':>8}  {'>S50&S150':>10}  {'SUPPORT':>16}  RESULT")
print("  "+"─"*55)

for sym in DIAG:
    if sym not in diag_data:
        print(f"  {sym:<7} — no data"); continue
    try:
        df_d = diag_data[sym]
        p    = float(df_d["Close"].iloc[-1])
        s50  = float(df_d["Close"].rolling(50).mean().iloc[-1])
        s150 = float(df_d["Close"].rolling(150).mean().iloc[-1])
        t    = lambda b: "✅" if b else "❌"
        struct_ok = p > s50 and p > s150
        r = analyze_tech_support(sym, df_d)
        if r:
            print(f"  {sym:<7} ${p:>7.2f}  {t(struct_ok):>10}  "
                  f"{r['support_label']:>16}  ✅ (fund check next)")
        else:
            print(f"  {sym:<7} ${p:>7.2f}  {t(struct_ok):>10}  "
                  f"{'no support':>16}  ❌")
    except Exception as e:
        print(f"  {sym:<7} error: {e}")

print(f"""
  Pattern:
    C1  Bull structure : price > SMA50  AND  price > SMA150
    C2  Support touch  : candle LOW within {CFG['support_zone_pct']}% of SMA50 or EMA20
                          in last {CFG['support_lookback']} bars, price closed back above
    C3  VERY STRONG FUNDAMENTALS (high bar — must pass {CFG['min_fund_hits']}/7):
        RevGrowth > {CFG['min_revenue_growth_pct']}%   ProfitMargin > {CFG['min_profit_margin_pct']}%
        ROE > {CFG['min_roe_pct']}%          Debt/Equity < {CFG['max_debt_to_equity']}
        CurrentRatio > {CFG['min_current_ratio']}   EPS > 0
        EarningsGrowth > {CFG['min_earnings_growth_pct']}%

  Tune if mostly ❌:
    support_zone_pct   3 → 5     (wider support zone)
    support_lookback   10 → 15   (look further back)
    min_fund_hits       5 → 4    (relax fundamental bar)
    rsi_min            40 → 30
""")
print("━"*65+"\n")

# ── Ticker list ───────────────────────────────────────────────
print("━"*65)
print("  STEP 3  FETCH TICKERS")
print("━"*65)

def get_tickers():
    pool = set()
    hdrs = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}
    for url, label in [
        ("https://www.nasdaqtrader.com/dynamic/SymDir/nasdaqlisted.txt","nasdaqlisted"),
        ("https://www.nasdaqtrader.com/dynamic/SymDir/otherlisted.txt", "otherlisted"),
    ]:
        try:
            r  = requests.get(url, headers=hdrs, timeout=20); r.raise_for_status()
            df = pd.read_csv(io.StringIO(r.text), sep="|")
            df.columns = [c.strip() for c in df.columns]
            sc = next((c for c in ["Symbol","ACT Symbol","Nasdaq Symbol"] if c in df.columns),None)
            ec = next((c for c in ["ETF","Is ETF"] if c in df.columns),None)
            if not sc: continue
            b = len(pool)
            for _, row in df.iterrows():
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
        r = requests.get(
            "https://api.nasdaq.com/api/screener/stocks"
            "?tableonly=true&limit=10000&exchange=nasdaq&download=true",
            headers={**hdrs,"Referer":"https://www.nasdaq.com/"}, timeout=25)
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
        "ALAB","SMCI","HOOD","COIN","SOFI","UPST","DDOG","SNOW","MDB","REGN",
        "VRTX","ISRG","LULU","FTNT","IDXX","SBUX","TMUS","RBRK","NET","MARA",
        "ADSK","ANSS","BIIB","CPRT","ENPH","FAST","GILD","INTU","MCHP","MNST",
        "NXPI","ODFL","ORLY","PAYX","PCAR","VRSK","ZBRA","PSTG","GTLB","MNDY",
        "HUBS","VEEV","PAYC","ABNB","DASH","RBLX","AMGN","EA","ULTA","XEL",
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
print(f"  STEP 4  SCANNING {len(TICKERS)} TICKERS")
print("━"*65)
print("  Pass 1: Technical structure + SMA50/EMA20 support (fast)")
print("  Pass 2: Very strong fundamentals — only for Pass 1 survivors\n")

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
                r = analyze_tech_support(sym, data_map[sym])
                if r is None: continue
                tech_passes.append({
                    "sym"  : sym,
                    "price": float(data_map[sym]["Close"].iloc[-1]),
                    "r"    : r,
                })
            except Exception: pass
        time.sleep(CFG["batch_sleep"])

got = len(TICKERS) - no_data
pct = got / max(len(TICKERS), 1) * 100
print(f"\n  Pass 1 done: {got}/{len(TICKERS)} got data ({pct:.0f}%)")
print(f"  Tech + support passes: {len(tech_passes)} stocks → checking fundamentals\n")

# ── PASS 2: Very strong fundamentals ────────────────────────────
print("━"*65)
print(f"  PASS 2  VERY STRONG FUNDAMENTAL CHECK ({len(tech_passes)} stocks)")
print("━"*65+"\n")

for item in tqdm(tech_passes, desc="Pass 2 Fund", unit="stk"):
    sym = item["sym"]; price = item["price"]; r = item["r"]
    try:
        fund = get_fundamentals(sym)
        time.sleep(CFG["fund_sleep"])

        if fund["Fund_Hits"] < CFG["min_fund_hits"]:
            continue   # not "very strong" enough

        fs   = fund["Fund_Score"]
        tscr = r["tech_score"]
        sscr = r["support_score"]
        total = fs + tscr + sscr

        result = {
            "Ticker"            : sym,
            "Price"             : round(price, 2),
            "Total"             : total,
            "Fund"              : fs,
            "Tech"              : tscr,
            "Supp"              : sscr,
            "Fund_Hits"         : fund["Fund_Hits"],
            "Support_Label"     : r["support_label"],
            "Support_Level"     : r["support_level"],
            "Support_Depth_%"   : r["support_depth_%"],
            "Bars_Since_Support": r["bars_since_support"],
            "Support_SMA50"     : r["support_sma50"],
            "Support_EMA20"     : r["support_ema20"],
            "Sector"            : fund["Sector"],
            "Company"           : fund["Company"],
            "Industry"          : fund["Industry"],
            "EMA20"             : r["EMA20"],
            "SMA50"             : r["SMA50"],
            "SMA150"            : r["SMA150"],
            "Dist_SMA50_%"      : r["Dist_SMA50_%"],
            "Dist_SMA150_%"     : r["Dist_SMA150_%"],
            "RSI"               : r["RSI"],
            "MACD_Hist"         : r["MACD_Hist"],
            "Tech_Flags"        : r["tech_reasons"],
            "Rev_Growth_%"      : fund["Rev_Growth_%"],
            "Profit_Margin_%"   : fund["Profit_Margin_%"],
            "ROE_%"             : fund["ROE_%"],
            "PE_Ratio"          : fund["PE_Ratio"],
            "EPS"               : fund["EPS"],
            "Debt_Equity"       : fund["Debt_Equity"],
            "Current_Ratio"     : fund["Current_Ratio"],
            "Earnings_Growth_%" : fund["Earnings_Growth_%"],
            "Market_Cap_B"      : fund["Market_Cap_B"],
            "Fund_Flags"        : fund["Fund_Flags"],
            # internals
            "_df"    : r["_df"],
            "_ema20" : r["_ema20"],
            "_sma50" : r["_sma50"],
            "_sma150": r["_sma150"],
            "_support_bar": r["_support_bar"],
        }
        results.append(result)
        live_print(result)
    except Exception: pass

print(f"\n{'━'*65}")
print(f"  SCAN COMPLETE")
print(f"  Tickers            : {len(TICKERS)}")
print(f"  Got data           : {got}  ({pct:.0f}%)")
print(f"  Tech+support passes: {len(tech_passes)}")
print(f"  ✅ Very strong fund : {len(results)}")
print(f"{'━'*65}")

if not results:
    print("\n  No matches. Try relaxing:")
    print("   min_fund_hits       5 → 4")
    print("   support_zone_pct    3 → 5")
    print("   support_lookback   10 → 15")
    print("   min_revenue_growth 10 → 5")
    print("   min_roe_pct        15 → 10")
    print("   rsi_min            40 → 30")

# Sort by total score (always runs, even on empty list)
results.sort(key=lambda x: x["Total"], reverse=True)

# ── Always build df_out and save/email (even if 0 results) ────
ts      = datetime.today().strftime("%Y%m%d_%H%M")
out_dir = os.environ.get("GITHUB_WORKSPACE", os.getcwd())

COLS = [
    "Ticker","Company","Sector","Price",
    "Total","Fund","Tech","Supp","Fund_Hits",
    "Support_Label","Support_Level","Support_Depth_%","Bars_Since_Support",
    "EMA20","SMA50","SMA150","Dist_SMA50_%","Dist_SMA150_%",
    "Rev_Growth_%","Profit_Margin_%","ROE_%","PE_Ratio","EPS",
    "Debt_Equity","Current_Ratio","Earnings_Growth_%","Market_Cap_B",
    "RSI","MACD_Hist","Tech_Flags","Fund_Flags",
]
df_out = pd.DataFrame([{k:v for k,v in r.items() if not k.startswith("_")}
                        for r in results]) if results else pd.DataFrame(columns=COLS)
if not df_out.empty:
    df_out = df_out[[c for c in COLS if c in df_out.columns]]
    df_out.reset_index(drop=True, inplace=True)

FMT = {
    "Price"             : lambda v: f"${v:.2f}",
    "Total"             : lambda v: f"{v:.0f}",
    "Fund"              : lambda v: f"{v:.0f}",
    "Tech"              : lambda v: f"{v:.0f}",
    "Supp"              : lambda v: f"{v:.0f}",
    "Fund_Hits"         : lambda v: f"{v:.0f}/7",
    "Support_Level"     : lambda v: f"${v:.2f}",
    "Support_Depth_%"   : lambda v: f"{v:.2f}%",
    "Bars_Since_Support": lambda v: f"{int(v)}",
    "EMA20"             : lambda v: f"${v:.2f}",
    "SMA50"             : lambda v: f"${v:.2f}",
    "SMA150"            : lambda v: f"${v:.2f}",
    "Dist_SMA50_%"      : lambda v: f"{v:+.2f}%",
    "Dist_SMA150_%"     : lambda v: f"{v:+.2f}%",
    "Rev_Growth_%"      : lambda v: f"{v:+.1f}%",
    "Profit_Margin_%"   : lambda v: f"{v:.1f}%",
    "ROE_%"             : lambda v: f"{v:.1f}%",
    "PE_Ratio"          : lambda v: f"{v:.1f}",
    "EPS"               : lambda v: f"${v:.2f}",
    "Debt_Equity"       : lambda v: f"{v:.2f}",
    "Current_Ratio"     : lambda v: f"{v:.2f}",
    "Earnings_Growth_%" : lambda v: f"{v:+.1f}%",
    "Market_Cap_B"      : lambda v: f"${v:.2f}B",
    "RSI"               : lambda v: f"{v:.1f}",
    "MACD_Hist"         : lambda v: f"{v:.4f}",
}

def fmt_v(col, val):
    if val is None or (isinstance(val, float) and np.isnan(val)): return "—"
    try:
        if col in FMT: return FMT[col](val)
    except Exception: pass
    return str(val) if str(val) not in ("nan","None","") else "—"

if _IN_NOTEBOOK and results:
    DISP = ["Ticker","Company","Sector","Price",
            "Total","Fund","Tech","Supp","Fund_Hits",
            "Support_Label","Support_Depth_%","Bars_Since_Support",
            "Rev_Growth_%","ROE_%","RSI"]
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
            elif col == "Fund_Hits":
                try:
                    v = int(str(raw).split("/")[0])
                    if v >= 6: sty = "color:#22c55e;font-weight:800;text-align:center;font-size:14px"
                    elif v >= 5: sty = "color:#86efac;font-weight:700;text-align:center"
                except Exception: pass
            elif col == "Support_Label":
                sty = f"color:{gc};font-weight:700" if "+" in str(raw) else "color:#3b82f6;font-weight:600"
            elif col == "Support_Depth_%":
                try:
                    v = float(str(raw).replace("%",""))
                    if v <= 1.0: sty = "color:#22c55e;font-weight:700"
                    elif v <= 2.0: sty = "color:#86efac"
                except Exception: pass
            elif col in ("Rev_Growth_%","ROE_%"):
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
    <span style="color:#f1f5f9;font-size:15px;font-weight:700">Very Strong Fundamentals + SMA50/EMA20 Support</span>
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
    📈 Very Strong Fundamentals + SMA50/EMA20 Support
  </h2>
  <p style="margin:6px 0 0;color:#94a3b8;font-size:12px">
    {datetime.today().strftime('%Y-%m-%d %H:%M')} &nbsp;·&nbsp;
    <b style="color:#22c55e">{len(results)} matches</b> from {len(TICKERS)} tickers
  </p>
</div>"""

    legend_html = f"""
<div style="background:#f8fafc;border:1px solid #e2e8f0;border-radius:8px;
        padding:12px 18px;margin-top:6px;font-size:11px;color:#64748b;
        font-family:'Segoe UI',Arial,sans-serif">
  <b style="color:#475569">GUIDE</b> &nbsp;·&nbsp;
  Fund_Hits = how many of 7 "very strong" fundamental checks passed (min {CFG['min_fund_hits']}) &nbsp;·&nbsp;
  Support_Label = which level price is bouncing off (SMA50+EMA20 = strongest) &nbsp;·&nbsp;
  Support_Depth_% = how close the low came to support (lower = tighter) &nbsp;·&nbsp;
  Bars_Since_Support 0 = touched today
</div>"""

    display_html(header_html + table_html + legend_html)

elif results:
    # ASCII table (CLI/GitHub Actions mode)
    CLI_COLS = ["Ticker","Price","Total","Fund","Tech","Supp","Fund_Hits",
                "Support_Label","Support_Depth_%","RSI"]
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
    tit = f"  Very Strong Fund + SMA50/EMA20 Support   {datetime.today().strftime('%Y-%m-%d')}   {len(df_out)} matches"
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
  Total              Fund(0-50) + Tech(0-30) + Supp(0-20)
  Fund_Hits           how many of 7 strict fund checks passed
  Support_Label       SMA50 / EMA20 / SMA50+EMA20 (strongest)
  Support_Depth_%     how close the low came to support
  ──────────────────────────────────────────────────────""")

# Save
fpath = os.path.join(out_dir, f"strong_fund_sma_support_{ts}.csv")
df_out.to_csv(fpath, index=False)
print(f"\n  💾 CSV → {fpath}")
tv = os.path.join(out_dir, f"tv_strong_fund_support_{ts}.txt")
with open(tv,"w") as f:
    f.write(f"###Very Strong Fund + SMA Support {datetime.today().strftime('%Y-%m-%d')}\n")
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
                      "Fund_Hits","Support_Label"]
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
            fhits  = r.get("Fund_Hits",0) or 0
            slabel = r.get("Support_Label","—")
            rows_e += (
                f'<tr style="background:{bg}">'
                f'<td style="padding:6px 11px;font-size:12px;font-weight:700">{ticker}</td>'
                f'<td style="padding:6px 11px;font-size:12px">${float(price):.2f}</td>'
                f'<td style="padding:6px 11px;font-size:12px;font-weight:700;'
                f'background:#166534;color:#fff;text-align:center">{float(total):.0f}</td>'
                f'<td style="padding:6px 11px;font-size:12px">{float(fund):.0f}</td>'
                f'<td style="padding:6px 11px;font-size:12px">{float(tech):.0f}</td>'
                f'<td style="padding:6px 11px;font-size:12px">{float(supp):.0f}</td>'
                f'<td style="padding:6px 11px;font-size:12px;text-align:center;font-weight:700">'
                f'{int(fhits)}/7</td>'
                f'<td style="padding:6px 11px;font-size:12px;color:#3b82f6;font-weight:600">'
                f'{slabel}</td>'
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
  📊 Very Strong Fundamentals + SMA50/EMA20 Support
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
            f"Very Strong Fundamentals + SMA50/EMA20 Support — {datetime.today().strftime('%Y-%m-%d')}",
            f"{cnt} matches",
            "="*60,
        ]
        if rl:
            for r in rl[:50]:
                ticker = r.get("Ticker","—")
                price  = r.get("Price",0) or 0
                total  = r.get("Total",0) or 0
                fhits  = r.get("Fund_Hits",0) or 0
                slabel = r.get("Support_Label","—")
                plain_lines.append(
                    f"{ticker:<7} ${float(price):.2f}  Total:{float(total):.0f}  "
                    f"FundHits:{int(fhits)}/7  Support:{slabel}"
                )
        else:
            plain_lines.append("No matches today")
        plain_lines.append("\nFull results in CSV attachment.")
        plain_e = "\n".join(plain_lines)

        subj = (f"📊 Strong Fund + SMA Support — {cnt} signal{'s' if cnt!=1 else ''}"
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
    top = results[:min(5, len(results))]
    fig, axes = plt.subplots(len(top), 1, figsize=(15, 5*len(top)), facecolor="#0f172a")
    if len(top)==1: axes=[axes]

    for idx, r in enumerate(top):
        ax    = axes[idx]
        df_p  = r["_df"].tail(80).copy()
        ema20 = r["_ema20"].reindex(df_p.index)
        sma50 = r["_sma50"].reindex(df_p.index)
        sma150= r["_sma150"].reindex(df_p.index)
        n_p   = len(df_p)
        fn    = len(r["_df"]); off = fn - n_p

        ax.set_facecolor("#0f172a")

        for i, (_, row_) in enumerate(df_p.iterrows()):
            o=float(row_["Open"]); h=float(row_["High"])
            l=float(row_["Low"]);  c=float(row_["Close"])
            clr="#34d399" if c>=o else "#ef4444"
            ax.plot([i,i],[l,h],color=clr,lw=0.7,zorder=2)
            blo=min(o,c); bhi=max(o,c); bh=max(bhi-blo,(h-l)*0.005)
            rect=mpatches.FancyBboxPatch((i-0.3,blo),0.6,bh,
                 boxstyle="square,pad=0",facecolor=clr,edgecolor=clr,lw=0.4,zorder=3)
            ax.add_patch(rect)

        ax.plot(range(n_p), ema20.values,  color="#34d399", lw=1.6, label="EMA20 🟢", zorder=5)
        ax.plot(range(n_p), sma50.values,  color="#3b82f6", lw=1.6, label="SMA50 🔵", zorder=4)
        ax.plot(range(n_p), sma150.values, color="#f472b6", lw=1.5, ls="-.", label="SMA150 🩷", zorder=4)

        sb = r["_support_bar"] - off if r["_support_bar"] is not None else None
        if sb is not None and 0 <= sb < n_p:
            ax.scatter([sb],[float(df_p["Low"].iloc[sb])],
                       color="#fbbf24", s=150, zorder=8, marker="v",
                       label=f"{r['Support_Label']} Support")
            ax.axvline(sb, color="#fbbf24", lw=1.0, ls=":", alpha=0.6)

        tick_step = max(1, n_p//8)
        ax.set_xticks(range(0, n_p, tick_step))
        ax.set_xticklabels(
            [df_p.index[i].strftime("%m/%d") for i in range(0,n_p,tick_step)],
            color="#94a3b8", fontsize=7)
        ax.set_xlim(-0.5, n_p-0.5)
        ax.set_title(
            f"{r['Ticker']}  {r.get('Company','')}  ${r['Price']:.2f}  |  "
            f"Total {r['Total']}/100 (F{r['Fund']}+T{r['Tech']}+S{r['Supp']})  |  "
            f"FundHits {r['Fund_Hits']}/7  |  "
            f"Support: {r['Support_Label']} (depth {r['Support_Depth_%']:.1f}%)  |  "
            f"RSI {r['RSI']:.0f}",
            color="#e2e8f0", fontsize=8, fontweight="bold", pad=6)
        ax.tick_params(colors="#94a3b8", labelsize=7)
        for sp_ in ax.spines.values(): sp_.set_edgecolor("#1e3a5f")
        ax.legend(loc="upper left", facecolor="#1e293b",
                  labelcolor="#e2e8f0", fontsize=7, framealpha=0.9)
        ax.grid(color="#1e3a5f", ls="--", lw=0.4, alpha=0.4, axis="y")

    plt.suptitle(
        f"Very Strong Fundamentals + SMA50/EMA20 Support  ·  "
        f"{datetime.today().strftime('%Y-%m-%d')}\n"
        f"🟢 EMA20  🔵 SMA50  🩷 SMA150  ▼ = Support Touch",
        color="#60a5fa", fontsize=10, fontweight="bold", y=1.001)
    plt.tight_layout()
    cp = os.path.join(out_dir, f"strong_fund_sma_chart_{ts}.png")
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
  📋 PATTERN EXPLAINED

  C1  BULL STRUCTURE
      Price ABOVE SMA50  AND  Price ABOVE SMA150
      = Confirmed uptrend across both timeframes

  C2  TAKING SUPPORT AT SMA50 OR EMA20
      Candle LOW touched SMA50 or EMA20 within 3%
      in the last 10 bars, price closed back above it
      = Pullback held, buyers defended the level

  C3  VERY STRONG FUNDAMENTALS  (high bar — 5 of 7 required)
      Revenue growth   > 10%
      Profit margin    > 15%
      ROE              > 15%
      Debt/Equity      < 1.0
      Current ratio    > 1.5
      EPS              > 0  (profitable)
      Earnings growth  > 10%

  WHY THIS COMBINATION:
      Only genuinely high-quality businesses (5+/7 strict checks)
      that are also structurally strong (above both MAs) and
      showing real technical support at a key short-term level
      make it through — a small, high-conviction watchlist.

  💡 BEST SETUPS
  Fund_Hits = 7/7           elite business quality
  Support_Label = SMA50+EMA20   double confluence support
  Support_Depth_% < 1%      very tight, precise bounce
  Bars_Since_Support = 0    fresh touch today

  ⚙️  TUNE IF 0 RESULTS
  min_fund_hits        5 → 4
  support_zone_pct      3 → 5
  support_lookback     10 → 15
  min_revenue_growth   10 → 5
  min_roe_pct          15 → 10
  rsi_min              40 → 30
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
""")
