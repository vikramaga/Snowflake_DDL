# ============================================================
# NASDAQ — Daily JMA Reclaim → Saved-High Breakout (v1)
# ============================================================
#
# SETUP (3 consecutive daily candles A, B, C; JMA = JMA(13, phase
# 40, power 2) evaluated on each candle's own bar):
#
#   1. CANDLE A: CLOSE BELOW JMA.
#   2. CANDLE B: CLOSE ABOVE JMA.
#   3. CANDLE C: CLOSE ABOVE JMA and its LOW ABOVE JMA (does not
#      touch JMA). Candle C's HIGH is saved.
#
# TRIGGER (this is what qualifies a stock):
#
#   4. A LATER daily candle CLOSES ABOVE the saved high — the first
#      such close after candle C.
#
# The MOST RECENT setup in the last setup_lookback_days is used.
# Defaults (not explicitly requested):
#   - CANCELLED if any candle after C closes back BELOW JMA before
#     the breakout.
#   - STALE if the breakout has not come within max_wait_bars
#     trading days of candle C.
#   - A stock QUALIFIES when the breakout candle is the most recent
#     daily candle (trigger_lookback_days = 1).
#   - Setups still waiting for their breakout are listed separately
#     as ARMED (with the saved high to set an alert at).
#
# DATA — only 1 download per ticker: daily bars.
#
# OUTPUT: Entry_Price and Stop_Loss are reasonable defaults (Entry =
# the breakout candle's close; Stop = candle C's low), not explicitly
# requested.
#
# SINGLE PASS — purely technical, no fundamentals fetch.
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
    "history_days"          : 550,   # daily bars — ~380 trading days, comfortable
                                      # margin above _clean()'s 220-bar floor

    # ── Indicator periods ────────────────────────────────────────
    "jma_period"             : 13,
    "jma_phase"              : 40,
    "jma_power"              : 2,

    # ── Pattern timing ──────────────────────────────────────────
    "setup_lookback_days"    : 20,   # search this far back for the most
                                      # recent A-B-C setup (candle C date)
    "max_wait_bars"          : 10,   # breakout must come within N trading
                                      # days of candle C, else STALE
    "trigger_lookback_days"  : 1,    # 1 = the breakout candle must be the
                                      # most recent daily candle

    # ── Filters ─────────────────────────────────────────────────
    "min_avg_volume"         : 100_000,
    "min_price"              : 10.0,

    "batch_size"             : 50,
    "batch_sleep"            : 1.5,
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

def check_setup_at(df, close, jma, i):
    """
    Checks the A-B-C setup with candle C at bar `i` (B = i-1,
    A = i-2). SINGLE SOURCE OF TRUTH for the setup logic. Every
    stage's bool is computed regardless of earlier failures.
    Returns (is_setup: bool, details: dict).
    """
    if i < 2 or i >= len(df):
        return False, {}
    cA, cB, cC = close.iloc[i-2], close.iloc[i-1], close.iloc[i]
    lowC = df["Low"].iloc[i]
    jA, jB, jC = jma.iloc[i-2], jma.iloc[i-1], jma.iloc[i]
    if any(np.isnan(v) for v in [cA, cB, cC, lowC, jA, jB, jC]):
        return False, {}

    A_below   = bool(cA < jA)                    # Step 1
    B_above   = bool(cB > jB)                    # Step 2
    C_clean   = bool(cC > jC and lowC > jC)      # Step 3

    return (A_below and B_above and C_clean), {
        "A_below": A_below, "B_above": B_above, "C_clean": C_clean,
    }

def evaluate_ticker(df, close, jma, cfg):
    """
    Finds the MOST RECENT setup (candle C within the last
    setup_lookback_days), saves candle C's high, then walks forward
    from C to classify it:

      TRIGGERED   first close above the saved high, within
                  max_wait_bars of C, with no close below JMA before it
      ARMED       no breakout yet, no close below JMA, still within
                  max_wait_bars — waiting for the breakout
      CANCELLED   a candle closed below JMA before any breakout
      STALE       no breakout within max_wait_bars of C

    Returns a dict, or None if there is no setup in the window.
    """
    n = len(df)
    lb = cfg["setup_lookback_days"]
    c_idx = None
    for i in range(n - 1, max(1, n - 1 - lb), -1):
        ok, _ = check_setup_at(df, close, jma, i)
        if ok:
            c_idx = i
            break
    if c_idx is None:
        return None

    saved_high = float(df["High"].iloc[c_idx])
    status, trig_idx, cancel_idx = "ARMED", None, None
    for t in range(c_idx + 1, n):
        if close.iloc[t] > saved_high:
            trig_idx, status = t, "TRIGGERED"
            break
        if close.iloc[t] < jma.iloc[t]:
            cancel_idx, status = t, "CANCELLED"
            break

    wait_limit = cfg["max_wait_bars"]
    if status == "TRIGGERED" and (trig_idx - c_idx) > wait_limit:
        status = "STALE"
    if status == "ARMED" and ((n - 1) - c_idx) >= wait_limit:
        status = "STALE"

    fresh = (status == "TRIGGERED"
             and ((n - 1) - trig_idx) < cfg["trigger_lookback_days"])

    return {
        "c_idx": c_idx, "saved_high": saved_high, "status": status,
        "trig_idx": trig_idx, "cancel_idx": cancel_idx, "fresh": fresh,
    }

# ── Diagnostic funnel — ticker level, most recent setup per ticker ──
FUNNEL_COUNTS = {
    "tickers_checked": 0,
    "passed_step1_setup_found": 0,          # A<JMA, B>JMA, C clean above JMA
    "passed_step2_not_cancelled": 0,        # no close below JMA after C
    "passed_step3_broke_saved_high": 0,     # closed above C's high in time
    "passed_step4_breakout_fresh": 0,       # breakout on the latest candle
    "armed_waiting": 0,                     # setup valid, breakout pending
}

# ── Technical signal ─────────────────────────────────────────────
def analyze_jma_breakout(sym, df_daily):
    """
    Returns ("match", row) for a fresh breakout, ("armed", row) for a
    setup still waiting for its breakout, or None.
    """
    if df_daily is None:
        return None

    price   = float(df_daily["Close"].iloc[-1])
    avg_vol = float(df_daily["Volume"].tail(20).mean())
    if price   < CFG["min_price"]:      return None
    if avg_vol < CFG["min_avg_volume"]: return None

    n = len(df_daily)
    if n < 60 + CFG["setup_lookback_days"]:   # let JMA settle
        return None

    close = df_daily["Close"]
    jma   = calc_jma(close, CFG["jma_period"], CFG["jma_phase"], CFG["jma_power"])

    global FUNNEL_COUNTS
    FUNNEL_COUNTS["tickers_checked"] += 1
    ev = evaluate_ticker(df_daily, close, jma, CFG)
    if ev is None:
        return None
    FUNNEL_COUNTS["passed_step1_setup_found"] += 1
    if ev["status"] != "CANCELLED":
        FUNNEL_COUNTS["passed_step2_not_cancelled"] += 1
    if ev["status"] == "TRIGGERED":
        FUNNEL_COUNTS["passed_step3_broke_saved_high"] += 1
    if ev["fresh"]:
        FUNNEL_COUNTS["passed_step4_breakout_fresh"] += 1
    if ev["status"] == "ARMED":
        FUNNEL_COUNTS["armed_waiting"] += 1

    ci = ev["c_idx"]
    lowC = float(df_daily["Low"].iloc[ci])
    jC   = float(jma.iloc[ci])
    c_clear_pct = (lowC - jC) / jC * 100 if jC > 0 else 0

    if ev["status"] == "ARMED":
        bars_since = (n - 1) - ci
        return "armed", {
            "Price"            : round(price, 2),
            "Saved_High"       : round(ev["saved_high"], 2),
            "Dist_to_High_%"   : round((ev["saved_high"] - price) / price * 100, 1),
            "JMA"              : round(float(jma.iloc[-1]), 2),
            "Stop_Loss"        : round(lowC, 2),
            "Setup_Date"       : df_daily.index[ci].strftime("%Y-%m-%d"),
            "Bars_Since_Setup" : bars_since,
            "Bars_Left"        : CFG["max_wait_bars"] - bars_since,
        }

    if not ev["fresh"]:
        return None

    ti = ev["trig_idx"]
    entry_price = float(close.iloc[ti])
    stop_loss   = lowC
    if entry_price <= stop_loss:
        return None
    risk_pct      = (entry_price - stop_loss) / entry_price * 100
    breakout_pct  = (entry_price - ev["saved_high"]) / ev["saved_high"] * 100
    above_jma_pct = (entry_price - float(jma.iloc[ti])) / float(jma.iloc[ti]) * 100
    wait_bars     = ti - ci
    vol_T  = float(df_daily["Volume"].iloc[ti])
    avg20  = float(df_daily["Volume"].iloc[max(0, ti-20):ti].mean())
    vol_vs_avg = vol_T / avg20 if avg20 > 0 else 0.0
    days_since = (n - 1) - ti

    # ── Score (0-100) ────────────────────────────────────────────
    score = 20   # base for clearing every gate
    reasons = []
    score += max(0, min(20, breakout_pct * 8))
    reasons.append(f"Breakout+{breakout_pct:.1f}%")
    score += max(0, min(15, c_clear_pct * 5))
    reasons.append(f"C-low clears JMA+{c_clear_pct:.1f}%")
    score += max(0, min(20, (vol_vs_avg - 1) * 20))
    reasons.append(f"Vol/20d x{vol_vs_avg:.2f}")
    score += max(0, 15 - (wait_bars - 1) * 2)
    reasons.append(f"{wait_bars}d C→breakout")
    score += max(0, 10 - days_since * 3)
    if above_jma_pct > 8:
        score -= 10; reasons.append("Extended")
    if risk_pct > 12:
        score -= 10; reasons.append("WideStop")
    score = round(min(100, max(0, score)))

    return "match", {
        "Score"             : score,
        "Price"             : round(price, 2),
        "Entry_Price"       : round(entry_price, 2),
        "Stop_Loss"         : round(stop_loss, 2),
        "Risk_%"            : round(risk_pct, 1),
        "Saved_High"        : round(ev["saved_high"], 2),
        "Breakout_%"        : round(breakout_pct, 1),
        "JMA"               : round(float(jma.iloc[ti]), 2),
        "Above_JMA_%"       : round(above_jma_pct, 1),
        "C_Low_Clear_%"     : round(c_clear_pct, 1),
        "Vol_vs_20d_Avg"    : round(vol_vs_avg, 2),
        "Bars_C_to_Breakout": wait_bars,
        "Setup_Date"        : df_daily.index[ci].strftime("%Y-%m-%d"),
        "Signal_Date"       : df_daily.index[ti].strftime("%Y-%m-%d"),
        "Days_Since_Signal" : days_since,
        "Flags"             : " | ".join(reasons),
        "_df_daily"         : df_daily,
        "_jma"              : jma,
        "_c_idx"            : ci,
        "_trig_idx"         : ti,
    }

# ── Download: daily ───────────────────────────────────────────
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
                        df = _clean(df, min_bars=220)
                        if df is not None: out[sym] = df
                    except Exception: pass
            elif len(symbols) == 1:
                df = _clean(raw, min_bars=220)
                if df is not None: out[symbols[0]] = df
    except Exception: pass
    for sym in [s for s in symbols if s not in out]:
        for _ in range(2):
            try:
                df = yf.Ticker(sym).history(
                    start=start.strftime("%Y-%m-%d"),
                    end=end.strftime("%Y-%m-%d"),
                    auto_adjust=True, actions=False)
                df = _clean(df, min_bars=220)
                if df is not None: out[sym] = df; break
            except Exception: time.sleep(0.2)
        time.sleep(0.04)
    return out

# ── Live print ────────────────────────────────────────────────
LIVE_COLS = ["Ticker","Price","Score","Entry_Price","Stop_Loss",
             "Saved_High","Breakout_%","Signal_Date"]
_CW = {"Ticker":8,"Price":10,"Score":7,"Entry_Price":12,"Stop_Loss":11,
       "Saved_High":11,"Breakout_%":11,"Signal_Date":13}
_CF = {"Price":"${:.2f}","Score":"{:.0f}","Entry_Price":"${:.2f}",
       "Stop_Loss":"${:.2f}","Saved_High":"${:.2f}"}
_hdr_done = False

def _live_header():
    global _hdr_done
    if _hdr_done: return
    print("\n" + "━"*95)
    print("  📊  LIVE MATCHES  —  breakout above the saved candle-C high on the latest candle")
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

# ── Main scan — single pass (daily download only, then check) ────
print("━"*65)
print(f"  STEP 3  SCANNING {len(TICKERS)} TICKERS")
print("━"*65)
print("  Fetching daily bars (single download per ticker)")
print("  Setup: A close < JMA → B close > JMA → C close & low > JMA (save C high)")
print("  Trigger: a later close above the saved high (on the latest candle)\n")

_hdr_done = False
results = []
armed   = []
no_daily_data = 0

daily_batches = [TICKERS[i:i+CFG["batch_size"]]
                 for i in range(0, len(TICKERS), CFG["batch_size"])]

daily_map = {}

with tqdm(total=len(TICKERS), desc="Daily fetch", unit="stk",
          bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]") as pbar:
    for batch in daily_batches:
        got = download_daily(batch, CFG["history_days"])
        daily_map.update(got)
        no_daily_data += len(batch) - len(got)
        pbar.update(len(batch))
        time.sleep(CFG["batch_sleep"])

got_daily = len(TICKERS) - no_daily_data
print(f"\n  Daily data : {got_daily}/{len(TICKERS)} tickers")
print()

for sym in tqdm(list(daily_map.keys()), desc="Checking JMA breakout", unit="stk"):
    try:
        out = analyze_jma_breakout(sym, daily_map[sym])
        if out is None: continue
        kind, r = out
        r["Ticker"] = sym
        if kind == "match":
            results.append(r)
            live_print(r)
        else:
            armed.append(r)
    except Exception: pass

print(f"\n{'━'*65}")
print(f"  SCAN COMPLETE")
print(f"  Tickers scanned : {len(TICKERS)}")
print(f"  Daily data      : {got_daily}")
print(f"  ✅ Matches       : {len(results)}")
print(f"  ⏳ Armed         : {len(armed)}  (setup valid, waiting for breakout)")
print(f"{'━'*65}")

# ── Diagnostic funnel — where tickers were filtered out ──────────
print(f"\n{'━'*65}")
print(f"  🔍 FUNNEL — most recent setup per liquid ticker")
print(f"  (setup searched over the last {CFG['setup_lookback_days']} trading days)")
print(f"{'━'*65}")
fc = FUNNEL_COUNTS
print(f"  Liquid tickers checked                       : {fc['tickers_checked']}")
print(f"  Step 1 — setup found (A<JMA, B>JMA, C clean) : {fc['passed_step1_setup_found']}")
print(f"  Step 2 — + not cancelled (no close < JMA)    : {fc['passed_step2_not_cancelled']}")
print(f"  Step 3 — + closed above saved high in time   : {fc['passed_step3_broke_saved_high']}")
print(f"  Step 4 — + breakout on the latest candle     : {fc['passed_step4_breakout_fresh']}  (= full match)")
print(f"  Armed — valid setup, breakout still pending  : {fc['armed_waiting']}")
print(f"{'━'*65}")

if not results:
    print("\n  No matches. Try relaxing (see the FUNNEL above to see which")
    print("  step is actually the bottleneck before guessing):")
    print("   trigger_lookback_days           1 → 3     (breakouts in the last 3 days)")
    print("   max_wait_bars                  10 → 15")
    print("   min_price                      10 → 5")
    print("   min_avg_volume             100000 → 50000")

results.sort(key=lambda x: x["Score"], reverse=True)
armed.sort(key=lambda x: x["Dist_to_High_%"])

# ── Always build df_out and save/email (even if 0 results) ────
ts      = datetime.today().strftime("%Y%m%d_%H%M")
out_dir = os.environ.get("GITHUB_WORKSPACE", os.getcwd())

COLS = [
    "Ticker","Price","Score",
    "Entry_Price","Stop_Loss","Risk_%",
    "Saved_High","Breakout_%","JMA","Above_JMA_%","C_Low_Clear_%","Vol_vs_20d_Avg",
    "Bars_C_to_Breakout","Setup_Date","Signal_Date","Days_Since_Signal",
    "Flags",
]
ARMED_COLS = ["Ticker","Price","Saved_High","Dist_to_High_%","JMA","Stop_Loss",
              "Setup_Date","Bars_Since_Setup","Bars_Left"]

df_out = pd.DataFrame([{k:v for k,v in r.items() if not k.startswith("_")}
                        for r in results]) if results else pd.DataFrame(columns=COLS)
if not df_out.empty:
    df_out = df_out[[c for c in COLS if c in df_out.columns]]
    df_out.reset_index(drop=True, inplace=True)
df_armed = pd.DataFrame(armed)[ARMED_COLS] if armed else pd.DataFrame(columns=ARMED_COLS)

FMT = {
    "Price"         : lambda v: f"${v:.2f}",
    "Score"         : lambda v: f"{v:.0f}",
    "Entry_Price"   : lambda v: f"${v:.2f}",
    "Stop_Loss"     : lambda v: f"${v:.2f}",
    "Saved_High"    : lambda v: f"${v:.2f}",
    "Risk_%"        : lambda v: f"{v:.1f}%",
    "JMA"           : lambda v: f"${v:.2f}",
    "Breakout_%"    : lambda v: f"{v:+.1f}%",
    "Above_JMA_%"   : lambda v: f"{v:+.1f}%",
    "C_Low_Clear_%" : lambda v: f"{v:+.1f}%",
    "Vol_vs_20d_Avg": lambda v: f"x{v:.2f}",
    "Days_Since_Signal": lambda v: f"{int(v)}d ago",
}

def fmt_v(col, val):
    if val is None or (isinstance(val, float) and np.isnan(val)): return "—"
    try:
        if col in FMT: return FMT[col](val)
    except Exception: pass
    return str(val) if str(val) not in ("nan","None","") else "—"

DISP_COLS = ["Ticker","Price","Score","Entry_Price","Stop_Loss","Risk_%",
             "Saved_High","Breakout_%","Vol_vs_20d_Avg","Signal_Date"]

if _IN_NOTEBOOK and results:
    th = "".join(
        f'<th style="background:#0f172a;color:#e2e8f0;padding:9px 12px;'
        f'font-size:11px;font-weight:700;text-align:center;'
        f'border-bottom:2px solid #22c55e;white-space:nowrap">{c}</th>'
        for c in DISP_COLS
    )
    rows_html = ""
    for i, r in enumerate(results):
        bg  = "#ffffff" if i%2==0 else "#f0f9ff"
        tds = "".join(
            f'<td style="padding:7px 12px;font-size:12px;border-bottom:1px solid #e2e8f0;'
            f'white-space:nowrap">{fmt_v(col, r.get(col))}</td>' for col in DISP_COLS)
        rows_html += f'<tr style="background:{bg}">{tds}</tr>\n'
    display_html(f"""
<div style="background:linear-gradient(135deg,#0f172a,#1e3a5f);
        border-radius:10px;padding:18px 24px;margin-bottom:8px;
        font-family:'Segoe UI',Arial,sans-serif">
  <h2 style="margin:0;color:#60a5fa;font-size:20px;font-weight:700">
    📈 JMA Reclaim → Saved-High Breakout
  </h2>
  <p style="margin:6px 0 0;color:#94a3b8;font-size:12px">
    {datetime.today().strftime('%Y-%m-%d %H:%M')} &nbsp;·&nbsp;
    <b style="color:#22c55e">{len(results)} breakouts</b>, {len(armed)} armed, from {len(TICKERS)} tickers
  </p>
</div>
<div style="overflow-x:auto;border:1px solid #e2e8f0;border-radius:8px">
  <table style="border-collapse:collapse;width:100%;min-width:700px">
    <thead><tr>{th}</tr></thead>
    <tbody>{rows_html}</tbody>
  </table>
</div>""")

elif results:
    # ASCII table (CLI/GitHub Actions mode)
    CLI_COLS = [c for c in DISP_COLS if c in df_out.columns]
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
    tit = f"  JMA Reclaim → Saved-High Breakout   {datetime.today().strftime('%Y-%m-%d')}   {len(df_out)} matches"
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
  Score           0-100 (breakout strength + C's clearance above
                  JMA + breakout volume + speed C→breakout + freshness)
  Entry_Price     breakout candle's close
  Stop_Loss       candle C's low
  Saved_High      candle C's high (the level that had to be cleared)
  Breakout_%      breakout close vs the saved high
  Setup_Date      date of candle C
  Signal_Date     date of the breakout candle
  ──────────────────────────────────────────────────────""")

if armed:
    print(f"\n  ⏳ ARMED (setup valid, waiting for a close above the saved high) — top 20 by distance:")
    print(df_armed.head(20).to_string(index=False))

# Save
fpath = os.path.join(out_dir, f"daily_jma_reclaim_breakout_{ts}.csv")
df_out.to_csv(fpath, index=False)
print(f"\n  💾 CSV → {fpath}")
apath = os.path.join(out_dir, f"daily_jma_reclaim_armed_{ts}.csv")
df_armed.to_csv(apath, index=False)
print(f"  💾 Armed CSV → {apath}")
tv = os.path.join(out_dir, f"tv_daily_jma_reclaim_breakout_{ts}.txt")
with open(tv,"w") as f:
    f.write(f"###JMA Breakout {datetime.today().strftime('%Y-%m-%d')}\n")
    for r in results: f.write(f"NASDAQ:{r['Ticker']}\n")
    f.write(f"###JMA Armed (alert at saved high)\n")
    for r in armed: f.write(f"NASDAQ:{r['Ticker']}\n")
print(f"  📋 TradingView → {tv}")
if results:
    print(f"\n  📋 Tickers (comma-separated):")
    print(f"  {', '.join(r['Ticker'] for r in results)}")

# ── Email with CSVs attached ──────────────────────────────
def _send_email(rl, al, csv_paths):
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
        print(f"[Email] Sending to {et}  ({cnt} results, {len(al)} armed)...")

        def th(cols):
            return "".join(
                f'<th style="background:#1e293b;color:#e2e8f0;padding:8px 11px;'
                f'font-size:11px;font-weight:700;border-bottom:2px solid #3b82f6;'
                f'white-space:nowrap">{c}</th>' for c in cols)

        def td(v, extra=""):
            return f'<td style="padding:6px 11px;font-size:12px;{extra}">{v}</td>'

        heads = ["Ticker","Price","Score","Entry","Stop","Risk %","Saved High",
                 "Breakout %","Vol/20d","Setup","Signal"]
        rows_e = ""
        for i, r in enumerate(rl[:50]):
            bg = "#fff" if i % 2 == 0 else "#f0f9ff"
            rows_e += (
                f'<tr style="background:{bg}">'
                + td(r.get("Ticker","—"), "font-weight:700")
                + td(f'${float(r.get("Price",0) or 0):.2f}')
                + td(f'{float(r.get("Score",0) or 0):.0f}',
                     "font-weight:700;background:#166534;color:#fff;text-align:center")
                + td(f'${float(r.get("Entry_Price",0) or 0):.2f}', "color:#22c55e")
                + td(f'${float(r.get("Stop_Loss",0) or 0):.2f}', "color:#ef4444")
                + td(f'{float(r.get("Risk_%",0) or 0):.1f}%')
                + td(f'${float(r.get("Saved_High",0) or 0):.2f}', "color:#3b82f6")
                + td(f'{float(r.get("Breakout_%",0) or 0):+.1f}%', "font-weight:600")
                + td(f'x{float(r.get("Vol_vs_20d_Avg",0) or 0):.2f}')
                + td(r.get("Setup_Date","—"))
                + td(r.get("Signal_Date","—"), "color:#a78bfa;font-weight:600")
                + '</tr>')
        no_results_msg = ('<tr><td colspan="11" style="padding:20px;text-align:center;'
                          'color:#94a3b8;font-size:13px">No breakouts today</td></tr>')

        armed_heads = ["Ticker","Price","Saved High","Distance","Stop","Setup","Days left"]
        armed_rows = ""
        for i, r in enumerate(al[:30]):
            bg = "#fff" if i % 2 == 0 else "#fefce8"
            armed_rows += (
                f'<tr style="background:{bg}">'
                + td(r.get("Ticker","—"), "font-weight:700")
                + td(f'${float(r.get("Price",0) or 0):.2f}')
                + td(f'${float(r.get("Saved_High",0) or 0):.2f}', "color:#3b82f6;font-weight:600")
                + td(f'{float(r.get("Dist_to_High_%",0) or 0):.1f}%')
                + td(f'${float(r.get("Stop_Loss",0) or 0):.2f}', "color:#ef4444")
                + td(r.get("Setup_Date","—"))
                + td(r.get("Bars_Left","—"))
                + '</tr>')
        armed_block = (f'''
<p style="margin:18px 0 6px;font-size:13px;font-weight:700;color:#854d0e">
  ⏳ ARMED — valid setup, waiting for a close above the saved high
  ({len(al)} total{', closest 30 shown' if len(al) > 30 else ''})
</p>
<div style="overflow-x:auto;border-radius:8px;border:1px solid #fde68a">
  <table style="border-collapse:collapse;width:100%;min-width:500px">
    <thead><tr>{th(armed_heads)}</tr></thead>
    <tbody>{armed_rows}</tbody>
  </table>
</div>''' if al else '')

        fc = FUNNEL_COUNTS
        html_e = f"""<!DOCTYPE html><html><body style="margin:0;padding:0;
background:#f1f5f9;font-family:'Segoe UI',Arial,sans-serif">
<table width="100%" cellpadding="0" cellspacing="0"><tr><td align="center" style="padding:20px 10px">
<table width="100%" style="max-width:820px;background:#fff;border-radius:12px;
       overflow:hidden;box-shadow:0 4px 16px rgba(0,0,0,0.08)">
  <tr><td style="background:linear-gradient(135deg,#0f172a,#1e3a5f);padding:22px 28px">
<h1 style="margin:0;color:#60a5fa;font-size:20px;font-weight:700">
  📊 JMA Reclaim → Saved-High Breakout
</h1>
<p style="margin:6px 0 0;color:#94a3b8;font-size:12px">
  {datetime.today().strftime('%Y-%m-%d %H:%M UTC')} &nbsp;·&nbsp;
  {cnt} breakout{'s' if cnt!=1 else ''}, {len(al)} armed — A closes below JMA, B closes
  above JMA, C closes above JMA without touching it (high saved), then a close above that high
</p>
  </td></tr>
  <tr><td style="padding:14px 28px 4px;background:#0b1220">
<div style="background:#111827;border:1px solid #1f2937;border-radius:8px;padding:12px 16px">
  <p style="margin:0 0 6px;color:#93c5fd;font-size:12px;font-weight:700">
    🔍 FUNNEL — most recent setup per liquid ticker (last {CFG['setup_lookback_days']} trading days)
  </p>
  <p style="margin:0;color:#cbd5e1;font-size:12px">
    {fc['tickers_checked']} tickers &nbsp;→&nbsp;
    {fc['passed_step1_setup_found']} with a setup &nbsp;→&nbsp;
    {fc['passed_step2_not_cancelled']} not cancelled &nbsp;→&nbsp;
    {fc['passed_step3_broke_saved_high']} broke the saved high &nbsp;→&nbsp;
    <b style="color:#facc15">{fc['passed_step4_breakout_fresh']} on the latest candle = full match</b>
    &nbsp;·&nbsp; {fc['armed_waiting']} armed
  </p>
</div>
  </td></tr>
  <tr><td style="padding:16px">
<div style="overflow-x:auto;border-radius:8px;border:1px solid #e2e8f0">
  <table style="border-collapse:collapse;width:100%;min-width:700px">
    <thead><tr>{th(heads)}</tr></thead>
    <tbody>{rows_e or no_results_msg}</tbody>
  </table>
</div>
{armed_block}
<p style="font-size:11px;color:#64748b;margin:8px 0 0">📎 Breakouts and armed setups attached as CSV</p>
{f'''<div style="margin-top:10px;background:#f8fafc;border:1px solid #e2e8f0;
        border-radius:6px;padding:10px 14px">
  <p style="margin:0 0 4px;font-size:10px;color:#94a3b8;font-weight:700">
    BREAKOUT TICKERS (comma-separated, copy/paste)
  </p>
  <p style="margin:0;font-size:12px;color:#1e293b;font-family:monospace;
      word-break:break-all">{", ".join(r.get("Ticker","") for r in rl)}</p>
</div>''' if rl else ''}
  </td></tr>
  <tr><td style="background:#f8fafc;padding:12px 28px;
             border-top:1px solid #e2e8f0;text-align:center">
<p style="margin:0;color:#94a3b8;font-size:10px">
  ⚠️ Not financial advice &nbsp;·&nbsp; Entry = breakout close, Stop = candle C low (defaults) &nbsp;·&nbsp; Auto-generated by GitHub Actions
</p>
  </td></tr>
</table>
</td></tr></table>
</body></html>"""

        plain_lines = [
            f"JMA Reclaim -> Saved-High Breakout — {datetime.today().strftime('%Y-%m-%d')}",
            f"{cnt} breakouts, {len(al)} armed",
            "="*60,
            f"FUNNEL: {fc['tickers_checked']} tickers -> "
            f"{fc['passed_step1_setup_found']} setup -> "
            f"{fc['passed_step2_not_cancelled']} not cancelled -> "
            f"{fc['passed_step3_broke_saved_high']} broke saved high -> "
            f"{fc['passed_step4_breakout_fresh']} on latest candle (=full match); "
            f"{fc['armed_waiting']} armed",
            "="*60,
        ]
        if rl:
            for r in rl[:50]:
                plain_lines.append(
                    f"{r.get('Ticker','—'):<7} ${float(r.get('Price',0) or 0):.2f}  "
                    f"Score:{float(r.get('Score',0) or 0):.0f}  "
                    f"Entry:${float(r.get('Entry_Price',0) or 0):.2f}  "
                    f"SL:${float(r.get('Stop_Loss',0) or 0):.2f}  "
                    f"Risk:{float(r.get('Risk_%',0) or 0):.1f}%  "
                    f"SavedHigh:${float(r.get('Saved_High',0) or 0):.2f}  "
                    f"Breakout:{float(r.get('Breakout_%',0) or 0):+.1f}%  "
                    f"Setup:{r.get('Setup_Date','—')}  Signal:{r.get('Signal_Date','—')}"
                )
            plain_lines.append("")
            plain_lines.append("Tickers (comma-separated):")
            plain_lines.append(", ".join(r.get("Ticker","") for r in rl))
        else:
            plain_lines.append("No breakouts today")
        if al:
            plain_lines.append("")
            plain_lines.append("ARMED (alert at saved high):")
            for r in al[:30]:
                plain_lines.append(
                    f"{r.get('Ticker','—'):<7} ${float(r.get('Price',0) or 0):.2f}  "
                    f"SavedHigh:${float(r.get('Saved_High',0) or 0):.2f}  "
                    f"Dist:{float(r.get('Dist_to_High_%',0) or 0):.1f}%  "
                    f"SL:${float(r.get('Stop_Loss',0) or 0):.2f}  "
                    f"Setup:{r.get('Setup_Date','—')}  DaysLeft:{r.get('Bars_Left','—')}")
        plain_lines.append("\nFull results in CSV attachments.")
        plain_e = "\n".join(plain_lines)

        subj = (f"📊 JMA Breakout — {cnt} signal{'s' if cnt!=1 else ''}, {len(al)} armed"
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

    for csv_path in csv_paths:
        if csv_path and os.path.exists(csv_path):
            try:
                with open(csv_path, "rb") as f:
                    part = MIMEBase("application", "octet-stream")
                    part.set_payload(f.read())
                encoders.encode_base64(part)
                part.add_header("Content-Disposition",
                    f"attachment; filename={os.path.basename(csv_path)}")
                msg.attach(part)
                print(f"[Email] 📎 Attached: {os.path.basename(csv_path)} ({os.path.getsize(csv_path):,} bytes)")
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
    _send_email(results, armed, [fpath, apath])
except Exception as e:
    print(f"[Email] ❌  Unexpected top-level error: {type(e).__name__}: {e}")
    print("[Email]    Continuing — CSV is still saved.")

if _IN_NOTEBOOK:
    try:
        from google.colab import files
        files.download(fpath); files.download(apath); files.download(tv)
    except Exception: pass
else:
    print("  (CI: files in workspace, email sent)")

# ── Charts for top 5: price + JMA + saved high, with a zoomed candle
#    view marking candles A, B, C and the breakout ──────────────────
if results:
    top = results[:min(5,len(results))]
    fig, axes = plt.subplots(len(top),2,figsize=(16,4.2*len(top)),facecolor="#0f172a",
                              gridspec_kw={"width_ratios":[3,1]})
    if len(top)==1: axes = axes.reshape(1,2)
    for row, r in zip(axes, top):
        ax, ax_zoom = row[0], row[1]
        df_all = r["_df_daily"]
        df_p = df_all.tail(90).copy()
        jma_p = r["_jma"].reindex(df_p.index)
        ax.set_facecolor("#0f172a")
        ax.plot(df_p.index, df_p["Close"], color="#60a5fa", lw=1.6, label="Close", zorder=5)
        ax.plot(df_p.index, jma_p, color="#22c55e", lw=1.4, label="JMA 13", zorder=4)
        ax.axhline(r["Saved_High"], color="#3b82f6", lw=1.0, ls=":", alpha=0.9,
                  label=f"Saved high {r['Saved_High']:.2f}")
        ax.axhline(r["Stop_Loss"], color="#ef4444", lw=1.0, ls="--", alpha=0.85,
                  label=f"Stop {r['Stop_Loss']:.2f}")
        ax.set_title(
            f"{r['Ticker']}  |  {r['Price']:.2f} USD  |  Score {r['Score']}  |  "
            f"setup {r['Setup_Date']}  breakout {r['Signal_Date']}",
            color="#e2e8f0", fontsize=9, fontweight="bold", pad=7)
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))
        ax.tick_params(colors="#94a3b8", labelsize=8)
        for sp in ax.spines.values(): sp.set_edgecolor("#1e3a5f")
        ax.legend(loc="upper left", facecolor="#1e293b", labelcolor="#e2e8f0",
                  fontsize=6, framealpha=0.9, ncol=2)
        ax.grid(color="#1e3a5f", ls="--", lw=0.5, alpha=0.6)

        ci, ti = r["_c_idx"], r["_trig_idx"]
        z0 = max(0, ci - 4)
        zoom = df_all.iloc[z0: min(len(df_all), ti + 3)]
        zj   = r["_jma"].iloc[z0: min(len(df_all), ti + 3)]
        x    = np.arange(len(zoom))
        ax_zoom.set_facecolor("#0f172a")
        colors = ["#22c55e" if c >= o else "#ef4444" for o, c in zip(zoom["Open"], zoom["Close"])]
        ax_zoom.vlines(x, zoom["Low"], zoom["High"], color=colors, lw=1)
        ax_zoom.bar(x, (zoom["Close"] - zoom["Open"]).abs().clip(lower=1e-9),
                    bottom=np.minimum(zoom["Open"], zoom["Close"]), color=colors, width=0.6)
        ax_zoom.plot(x, zj.values, color="#22c55e", lw=1.4, alpha=0.9)
        ax_zoom.axhline(r["Saved_High"], color="#3b82f6", lw=1.0, ls=":")
        for lbl, idx in [("A", ci-2), ("B", ci-1), ("C", ci), ("BO", ti)]:
            k = idx - z0
            if 0 <= k < len(zoom):
                ax_zoom.annotate(lbl, (k, float(zoom["High"].iloc[k])), textcoords="offset points",
                                 xytext=(0, 4), ha="center", color="#facc15", fontsize=7, fontweight="bold")
        ax_zoom.set_xticks(x)
        ax_zoom.set_xticklabels([d.strftime("%m/%d") for d in zoom.index], rotation=45, fontsize=6)
        ax_zoom.set_title(f"Breakout {r['Breakout_%']:+.1f}%  Vol x{r['Vol_vs_20d_Avg']:.2f}",
                          color="#e2e8f0", fontsize=8, pad=5)
        ax_zoom.tick_params(colors="#94a3b8", labelsize=7)
        for sp in ax_zoom.spines.values(): sp.set_edgecolor("#1e3a5f")
        ax_zoom.grid(color="#1e3a5f", ls="--", lw=0.5, alpha=0.6)
    plt.suptitle(
        f"JMA Reclaim → Saved-High Breakout  ·  {datetime.today().strftime('%Y-%m-%d')}",
        color="#60a5fa", fontsize=12, fontweight="bold", y=1.001)
    plt.tight_layout()
    cp = os.path.join(out_dir, f"daily_jma_reclaim_breakout_chart_{ts}.png")
    plt.savefig(cp, dpi=150, bbox_inches="tight", facecolor="#0f172a")
    if _IN_NOTEBOOK: plt.show()
    else: plt.close()
    print(f"  📊 Chart → {cp}")
    if _IN_NOTEBOOK:
        try:
            from google.colab import files; files.download(cp)
        except Exception: pass

print(f"""
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  📋 SETUP (3 consecutive daily candles A, B, C; JMA(13, phase 40))
  1) CANDLE A: closes BELOW JMA.
  2) CANDLE B: closes ABOVE JMA.
  3) CANDLE C: closes ABOVE JMA and its LOW stays ABOVE JMA.
     Candle C's HIGH is saved.

  📋 TRIGGER (qualifies the stock)
  4) The first later candle to CLOSE ABOVE the saved high —
     must be the latest candle (trigger_lookback_days = {CFG['trigger_lookback_days']}).

  📋 DEFAULTS (not explicitly requested)
  Most recent setup in the last {CFG['setup_lookback_days']} trading days is used.
  Cancelled if a candle closes below JMA before the breakout.
  Stale if no breakout within {CFG['max_wait_bars']} trading days of candle C.
  Armed list = valid setups still waiting for the breakout.
  Entry_Price = breakout close; Stop_Loss = candle C's low.
  Universe    = price >= 10 USD, 20d avg volume >= 100k

  📋 SCORE (0-100)
  Base 20 + breakout strength (0-20) + C's low clearance above JMA
  (0-15) + breakout volume vs 20d avg (0-20) + speed from C to
  breakout (0-15) + freshness (0-10); -10 if > 8% above JMA,
  -10 if stop risk > 12%

  ⚙️  TUNE IF 0 RESULTS
  trigger_lookback_days           1 → 3
  max_wait_bars                  10 → 15
  min_price                      10 → 5
  min_avg_volume             100000 → 50000
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
""")
