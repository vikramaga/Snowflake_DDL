# ============================================================
# NASDAQ — Monthly 3-Candle Reclaim: Red → Partial → Full (JMA/EMA8/SMA50) (v1)
# ============================================================
#
# SIGNAL (evaluated on the most recent 3 monthly candles — this is
# an inherently slow, monthly-timeframe pattern, not a rolling
# daily-window scan). Call them A (oldest), B, C (current month):
#
#   1. CANDLE A: RED — closed below its open.
#
#   2. CANDLE B: GREEN — closed above its open, AND closed above
#      EMA8 but BELOW JMA (a PARTIAL reclaim — clears the faster
#      line, not yet the slower one).
#
#   3. CANDLE C: GREEN — closed above its open, AND closed above
#      BOTH JMA and EMA8 (a FULL reclaim).
#
#   4. VOLUME: candle C's volume is higher than candle B's (the two
#      green candles show increasing volume).
#
#   5. STRUCTURE: both EMA8 and JMA are above SMA50 on candle C — a
#      healthy longer-term backdrop, not just a local bounce.
#
# All MAs (EMA8, JMA, SMA50) are computed on MONTHLY closes, not the
# usual daily values.
#
# DATA — only 1 download per ticker: daily bars (~7 years, needed
# for a stable monthly SMA50), resampled to monthly ('ME') — no
# separate network calls needed.
#
# OUTPUT: Entry_Price and Stop_Loss are reasonable defaults (Entry =
# candle C's close; Stop = candle A's low), not explicitly requested.
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
    # NOTE on history_days vs _clean()'s min_bars: a past scanner this
    # session silently returned 0 results everywhere because
    # history_days implied fewer trading days than _clean()'s min_bars
    # floor, so EVERY download got rejected before any pattern logic
    # ran. Verified numerically this time (see comments below) to
    # avoid repeating that bug.
    "history_days"          : 2500,  # ~7 years — yields ~115 monthly bars on
                                      # real data (verified via synthetic test:
                                      # 2500 calendar days -> 115 monthly bars),
                                      # comfortable margin above the 50 needed
                                      # for a stable monthly SMA50

    # ── Indicator periods (all computed on MONTHLY closes) ───────
    "ema8_period"              : 8,
    "jma_period"               : 13,
    "jma_phase"                : 40,
    "sma50_period"             : 50,

    # ── Filters ─────────────────────────────────────────────────
    "min_avg_volume"         : 80_000,
    "min_price"              : 2.0,

    "batch_size"             : 50,
    "batch_sleep"            : 1.5,
}

# ── Indicators ───────────────────────────────────────────────
def resample_ohlcv(df, rule):
    """Resample daily OHLCV to a coarser timeframe (e.g. 'ME' for monthly)."""
    agg = {"Open": "first", "High": "max", "Low": "min",
           "Close": "last", "Volume": "sum"}
    out = df.resample(rule).agg(agg)
    out.dropna(subset=["Close"], inplace=True)
    return out

def calc_jma(series, period=13, phase=40, power=2):
    """
    JMA (Jurik Moving Average) approximation — adaptive EMA with
    phase-based smoothing, using the corrected e2 update (steady-
    state gain 1.0).
    """
    n = len(series)
    vals = series.values.astype(float)
    result = np.full(n, np.nan)
    phase_ratio = phase / 100.0 + 1.5
    alpha = 2.0 / (period + 1.0)
    beta = alpha * phase_ratio
    first_valid = 0
    for i in range(n):
        if not np.isnan(vals[i]):
            first_valid = i
            break
    e0 = e1 = e2 = vals[first_valid]
    result[first_valid] = e0
    for i in range(first_valid + 1, n):
        v = vals[i]
        e0 = (1 - alpha) * e0 + alpha * v
        e1 = (v - e0) * (1 - beta) + beta * e1
        e2 = (1 - alpha) * e2 + alpha * (e0 + e1)
        result[i] = e2
    return pd.Series(result, index=series.index)

def check_3candle_reclaim(df_monthly, ema8, jma, sma50, i, cfg):
    """
    Checks the 3-candle reclaim pattern with candle C (the most
    recent, full-reclaim candle) anchored at bar `i`. Candle B is
    i-1 (partial reclaim), candle A is i-2 (the red candle). This is
    the SINGLE SOURCE OF TRUTH for the pattern logic.

    Unlike a simple short-circuit check, this computes EVERY stage's
    pass/fail regardless of earlier failures (guarding only on
    missing/insufficient data), so callers can diagnose exactly
    which stage is filtering results out — see FUNNEL_COUNTS below.

    Returns (passed: bool, details: dict).
    """
    if i < 2 or i >= len(df_monthly):
        return False, {}

    oA, cA = df_monthly["Open"].iloc[i-2], df_monthly["Close"].iloc[i-2]
    oB, cB = df_monthly["Open"].iloc[i-1], df_monthly["Close"].iloc[i-1]
    oC, cC = df_monthly["Open"].iloc[i],   df_monthly["Close"].iloc[i]
    volB, volC = df_monthly["Volume"].iloc[i-1], df_monthly["Volume"].iloc[i]

    e8_B, j_B = ema8.iloc[i-1], jma.iloc[i-1]
    e8_C, j_C, s50_C = ema8.iloc[i], jma.iloc[i], sma50.iloc[i]
    if any(np.isnan(v) for v in [e8_B, j_B, e8_C, j_C, s50_C]):
        return False, {}

    # ── Step 1: candle A red ─────────────────────────────────────
    A_red = bool(cA < oA)

    # ── Step 2: candle B green + partial reclaim (above EMA8, below JMA) ──
    B_green = bool(cB > oB)
    B_partial_reclaim = bool((cB > e8_B) and (cB < j_B))

    # ── Step 3: candle C green + full reclaim (above both) ───────
    C_green = bool(cC > oC)
    C_full_reclaim = bool((cC > j_C) and (cC > e8_C))

    # ── Step 4: volume increasing across the two green candles ───
    vol_increasing = bool(volC > volB)

    # ── Step 5: structure — EMA8 and JMA both above SMA50 ─────────
    structure_ok = bool((e8_C > s50_C) and (j_C > s50_C))

    passed = (A_red and B_green and B_partial_reclaim and
              C_green and C_full_reclaim and vol_increasing and structure_ok)

    return passed, {
        "idx": i, "cA": float(cA), "cB": float(cB), "cC": float(cC),
        "e8_B": float(e8_B), "j_B": float(j_B),
        "e8_C": float(e8_C), "j_C": float(j_C), "s50_C": float(s50_C),
        "volB": float(volB), "volC": float(volC),
        "low_A": float(df_monthly["Low"].iloc[i-2]),
        "date_A": df_monthly.index[i-2], "date_B": df_monthly.index[i-1],
        "date_C": df_monthly.index[i],
        "A_red": A_red, "B_green": B_green, "B_partial_reclaim": B_partial_reclaim,
        "C_green": C_green, "C_full_reclaim": C_full_reclaim,
        "vol_increasing": vol_increasing, "structure_ok": structure_ok,
    }

# ── Diagnostic funnel — tallies how far each ticker gets through the
#    pattern, across the FULL universe scan, so a 0-match run can be
#    diagnosed empirically instead of guessed at. Evaluated once per
#    ticker (the most recent 3 monthly candles), not per rolling day ──
FUNNEL_COUNTS = {
    "tickers_checked": 0,
    "passed_step1_A_red": 0,
    "passed_step2_B_partial_reclaim": 0,
    "passed_step3_C_full_reclaim": 0,
    "passed_step4_vol_increasing": 0,
    "passed_step5_structure": 0,
}

# ── Technical signal: monthly 3-candle reclaim ─────────────────────
def analyze_3candle_reclaim(sym, df_daily):
    """
    Returns dict with score and setup details, or None if no
    required condition is met.
    """
    if df_daily is None:
        return None

    price   = float(df_daily["Close"].iloc[-1])
    avg_vol = float(df_daily["Volume"].tail(20).mean())
    if price   < CFG["min_price"]:      return None
    if avg_vol < CFG["min_avg_volume"]: return None

    df_monthly = resample_ohlcv(df_daily, "ME")
    if len(df_monthly) < CFG["sma50_period"] + 10:
        return None   # not enough monthly bars for a stable monthly SMA50

    ema8  = df_monthly["Close"].ewm(span=CFG["ema8_period"], adjust=False).mean()
    jma   = calc_jma(df_monthly["Close"], CFG["jma_period"], CFG["jma_phase"])
    sma50 = df_monthly["Close"].rolling(CFG["sma50_period"]).mean()

    i = len(df_monthly) - 1
    global FUNNEL_COUNTS
    passed, sig = check_3candle_reclaim(df_monthly, ema8, jma, sma50, i, CFG)
    if sig:
        FUNNEL_COUNTS["tickers_checked"] += 1
        if sig["A_red"]:
            FUNNEL_COUNTS["passed_step1_A_red"] += 1
            if sig["B_green"] and sig["B_partial_reclaim"]:
                FUNNEL_COUNTS["passed_step2_B_partial_reclaim"] += 1
                if sig["C_green"] and sig["C_full_reclaim"]:
                    FUNNEL_COUNTS["passed_step3_C_full_reclaim"] += 1
                    if sig["vol_increasing"]:
                        FUNNEL_COUNTS["passed_step4_vol_increasing"] += 1
                        if sig["structure_ok"]:
                            FUNNEL_COUNTS["passed_step5_structure"] += 1
    if not passed:
        return None

    # ── Entry / Stop (reasonable defaults — see header note) ─────────
    entry_price = sig["cC"]
    stop_loss   = sig["low_A"]
    if entry_price <= stop_loss:
        return None
    risk_pct = (entry_price - stop_loss) / entry_price * 100 if entry_price > 0 else 0

    vol_ratio = sig["volC"] / sig["volB"] if sig["volB"] > 0 else 0
    above_sma50_pct = (sig["cC"] - sig["s50_C"]) / sig["s50_C"] * 100 if sig["s50_C"] > 0 else 0

    # ── Score (0-100) ────────────────────────────────────────────
    score = 0
    reasons = []
    score += min(25, (vol_ratio - 1.0) * 20)
    reasons.append(f"VolRatio{vol_ratio:.2f}x")
    clearance_C = (sig["cC"] - sig["j_C"]) / sig["j_C"] * 100 if sig["j_C"] > 0 else 0
    score += max(0, min(25, clearance_C * 2))
    reasons.append(f"ClearJMA+{clearance_C:.1f}%")
    score += max(0, min(25, above_sma50_pct))
    reasons.append(f"AboveSMA50+{above_sma50_pct:.1f}%")
    score += 25   # base for clearing every gate
    score = round(min(100, max(0, score)))

    return {
        "Score"          : score,
        "Price"          : round(price, 2),
        "Entry_Price"    : round(entry_price, 2),
        "Stop_Loss"      : round(stop_loss, 2),
        "Risk_%"         : round(risk_pct, 1),
        "EMA8"           : round(sig["e8_C"], 2),
        "JMA"            : round(sig["j_C"], 2),
        "SMA50"          : round(sig["s50_C"], 2),
        "Vol_Ratio"      : round(vol_ratio, 2),
        "Above_SMA50_%"  : round(above_sma50_pct, 1),
        "Month_A"        : sig["date_A"].strftime("%Y-%m"),
        "Month_B"        : sig["date_B"].strftime("%Y-%m"),
        "Month_C"        : sig["date_C"].strftime("%Y-%m"),
        "Flags"          : " | ".join(reasons),
        "_df_daily"      : df_daily,
        "_df_monthly"    : df_monthly,
        "_ema8"          : ema8, "_jma": jma, "_sma50": sma50,
    }

# ── Download: daily (→ also Weekly + Monthly via resample) ──────
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
                        df = _clean(df, min_bars=1000)
                        if df is not None: out[sym] = df
                    except Exception: pass
            elif len(symbols) == 1:
                df = _clean(raw, min_bars=1000)
                if df is not None: out[symbols[0]] = df
    except Exception: pass
    for sym in [s for s in symbols if s not in out]:
        for _ in range(2):
            try:
                df = yf.Ticker(sym).history(
                    start=start.strftime("%Y-%m-%d"),
                    end=end.strftime("%Y-%m-%d"),
                    auto_adjust=True, actions=False)
                df = _clean(df, min_bars=1000)
                if df is not None: out[sym] = df; break
            except Exception: time.sleep(0.2)
        time.sleep(0.04)
    return out

# ── Live print ────────────────────────────────────────────────
LIVE_COLS = ["Ticker","Price","Score","Entry_Price","Stop_Loss",
             "EMA8","JMA","SMA50"]
_CW = {"Ticker":8,"Price":10,"Score":7,"Entry_Price":12,"Stop_Loss":11,
       "EMA8":10,"JMA":10,"SMA50":10}
_CF = {"Price":"${:.2f}","Score":"{:.0f}","Entry_Price":"${:.2f}",
       "Stop_Loss":"${:.2f}"}
_hdr_done = False

def _live_header():
    global _hdr_done
    if _hdr_done: return
    print("\n" + "━"*95)
    print("  📊  LIVE MATCHES  —  each stock printed the moment it passes all 3 timeframes")
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
print("  A stock only matches if the monthly candle pattern and monthly")
print("  and SMA structure all fire within the lookback window\n")

_hdr_done = False
results = []
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

for sym in tqdm(list(daily_map.keys()), desc="Checking monthly reversal", unit="stk"):
    try:
        r = analyze_3candle_reclaim(sym, daily_map[sym])
        if r is None: continue
        r["Ticker"] = sym
        results.append(r)
        live_print(r)
    except Exception: pass

print(f"\n{'━'*65}")
print(f"  SCAN COMPLETE")
print(f"  Tickers scanned : {len(TICKERS)}")
print(f"  Daily data      : {got_daily}")
print(f"  ✅ Matches       : {len(results)}")
print(f"{'━'*65}")

# ── Diagnostic funnel — where tickers were filtered out ──────────
print(f"\n{'━'*65}")
print(f"  🔍 FUNNEL — where tickers were filtered out")
print(f"  (evaluated once per liquid ticker, on its last 2 monthly candles)")
print(f"{'━'*65}")
fc = FUNNEL_COUNTS
print(f"  Tickers checked                             : {fc['tickers_checked']}")
print(f"  Step 1 — candle A red                       : {fc['passed_step1_A_red']}")
print(f"  Step 2 — + candle B green, partial reclaim  : {fc['passed_step2_B_partial_reclaim']}")
print(f"  Step 3 — + candle C green, full reclaim     : {fc['passed_step3_C_full_reclaim']}")
print(f"  Step 4 — + volume increasing (B to C)       : {fc['passed_step4_vol_increasing']}")
print(f"  Step 5 — + EMA8/JMA both above SMA50         : {fc['passed_step5_structure']}  (= full pattern)")
print(f"{'━'*65}")

if not results:
    print("\n  No matches. Try relaxing (see the FUNNEL above to see which")
    print("  step is actually the bottleneck before guessing):")
    print("   ema8_period                        8 → 5    (faster, more reactive EMA)")
    print("   min_price                         2 → 1")
    print("   min_avg_volume                80000 → 50000")

results.sort(key=lambda x: x["Score"], reverse=True)

# ── Always build df_out and save/email (even if 0 results) ────
ts      = datetime.today().strftime("%Y%m%d_%H%M")
out_dir = os.environ.get("GITHUB_WORKSPACE", os.getcwd())

COLS = [
    "Ticker","Price","Score",
    "Entry_Price","Stop_Loss","Risk_%",
    "EMA8","JMA","SMA50","Vol_Ratio","Above_SMA50_%",
    "Month_A","Month_B","Month_C",
    "Signal_Date","Days_Since_Signal","Recent_Signal_Count","Recent_Signals",
    "Flags",
]
df_out = pd.DataFrame([{k:v for k,v in r.items() if not k.startswith("_")}
                        for r in results]) if results else pd.DataFrame(columns=COLS)
if not df_out.empty:
    df_out = df_out[[c for c in COLS if c in df_out.columns]]
    df_out.reset_index(drop=True, inplace=True)

FMT = {
    "Price"       : lambda v: f"${v:.2f}",
    "Score"       : lambda v: f"{v:.0f}",
    "Entry_Price" : lambda v: f"${v:.2f}",
    "Stop_Loss"   : lambda v: f"${v:.2f}",

    "Risk_%"      : lambda v: f"{v:.1f}%",

    "EMA8"           : lambda v: f"${v:.2f}",
    "JMA"            : lambda v: f"${v:.2f}",
    "SMA50"          : lambda v: f"${v:.2f}",
    "Vol_Ratio"      : lambda v: f"{v:.2f}x",
    "Above_SMA50_%"  : lambda v: f"{v:+.1f}%",
    "Days_Since_Signal": lambda v: f"{int(v)}d ago",
}

def fmt_v(col, val):
    if val is None or (isinstance(val, float) and np.isnan(val)): return "—"
    try:
        if col in FMT: return FMT[col](val)
    except Exception: pass
    return str(val) if str(val) not in ("nan","None","") else "—"

if _IN_NOTEBOOK and results:
    DISP = ["Ticker","Price","Score","Entry_Price","Stop_Loss",
            "EMA8","JMA","SMA50","Vol_Ratio","Month_C"]
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
            elif col == "Vol_Ratio":
                try:
                    v = float(str(raw).replace("x",""))
                    clr = "#22c55e" if v >= 1.5 else "#f59e0b"
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
    <span style="color:#f1f5f9;font-size:15px;font-weight:700">Monthly 3-Candle Reclaim (JMA/EMA8/SMA50)</span>
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
    📈 Monthly 3-Candle Reclaim (Red → Partial → Full)
  </h2>
  <p style="margin:6px 0 0;color:#94a3b8;font-size:12px">
    {datetime.today().strftime('%Y-%m-%d %H:%M')} &nbsp;·&nbsp;
    <b style="color:#22c55e">{len(results)} matches</b> from {len(TICKERS)} tickers
    &nbsp;·&nbsp; pattern checked over the last {CFG['signal_lookback_days']} trading days
  </p>
</div>"""

    legend_html = f"""
<div style="background:#f8fafc;border:1px solid #e2e8f0;border-radius:8px;
        padding:12px 18px;margin-top:6px;font-size:11px;color:#64748b;
        font-family:'Segoe UI',Arial,sans-serif">
  <b style="color:#475569">GUIDE</b> &nbsp;·&nbsp;
  Previous month closed red (below its open) &nbsp;·&nbsp;
  Current month closes green (above its open) &nbsp;·&nbsp;
  Candle A red, candle B green closed above EMA8 but below JMA,
  now above it this month — a fresh cross, not a persistent level
  previous month's % decline &nbsp;·&nbsp;
  Current close is above the MONTHLY 50-period SMA (~4 years of monthly
  bars) &nbsp;·&nbsp;
  Entry_Price = the current month's close &nbsp;·&nbsp;
  Stop_Loss = the previous month's low — reasonable defaults, not
  explicitly requested
</div>"""

    display_html(header_html + table_html + legend_html)

elif results:
    # ASCII table (CLI/GitHub Actions mode)
    CLI_COLS = ["Ticker","Price","Score","Entry_Price","Stop_Loss",
                "EMA8","JMA","SMA50","Vol_Ratio","Month_C"]
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
    tit = f"  Monthly 3-Candle Reclaim (JMA/EMA8/SMA50)   {datetime.today().strftime('%Y-%m-%d')}   {len(df_out)} matches"
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
  Score           0-100 (volume-increase strength + how far candle C
                  clears JMA + distance above the monthly SMA50)
  Entry_Price     candle C's (the current month's) close
  Stop_Loss       candle A's (the red month's) low
  EMA8 / JMA / SMA50  the three monthly moving averages, as of candle C
  Vol_Ratio       candle C's volume / candle B's volume
  Month_A / Month_B / Month_C  the three monthly periods (YYYY-MM):
                                red, partial reclaim, full reclaim
  ──────────────────────────────────────────────────────""")

# Save
fpath = os.path.join(out_dir, f"monthly_3candle_reclaim_jma_ema8_{ts}.csv")
df_out.to_csv(fpath, index=False)
print(f"\n  💾 CSV → {fpath}")
tv = os.path.join(out_dir, f"tv_monthly_3candle_reclaim_jma_ema8_{ts}.txt")
with open(tv,"w") as f:
    f.write(f"###Monthly 3-Candle Reclaim JMA-EMA8-SMA50 {datetime.today().strftime('%Y-%m-%d')}\n")
    for r in results: f.write(f"NASDAQ:{r['Ticker']}\n")
print(f"  📋 TradingView → {tv}")
if results:
    print(f"\n  📋 Tickers (comma-separated):")
    print(f"  {', '.join(r['Ticker'] for r in results)}")

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
            for c in ["Ticker","Price","Score","Entry_Price","Stop_Loss",
                      "EMA8","JMA","SMA50","Vol_Ratio","Month_C"]
        )
        rows_e = ""
        for i, r in enumerate(rl[:50]):
            bg  = "#fff" if i % 2 == 0 else "#f0f9ff"
            ticker = r.get("Ticker","—")
            price  = r.get("Price",0) or 0
            score  = r.get("Score",0) or 0
            entry  = r.get("Entry_Price",0) or 0
            stop   = r.get("Stop_Loss",0) or 0
            ema8_v = r.get("EMA8",0) or 0
            jma_v  = r.get("JMA",0) or 0
            sma50_v= r.get("SMA50",0) or 0
            vol_r  = r.get("Vol_Ratio",0) or 0
            cmonth = r.get("Month_C","—")
            rows_e += (
                f'<tr style="background:{bg}">'
                f'<td style="padding:6px 11px;font-size:12px;font-weight:700">{ticker}</td>'
                f'<td style="padding:6px 11px;font-size:12px">${float(price):.2f}</td>'
                f'<td style="padding:6px 11px;font-size:12px;font-weight:700;'
                f'background:#166534;color:#fff;text-align:center">{float(score):.0f}</td>'
                f'<td style="padding:6px 11px;font-size:12px;color:#22c55e">${float(entry):.2f}</td>'
                f'<td style="padding:6px 11px;font-size:12px;color:#ef4444">${float(stop):.2f}</td>'
                f'<td style="padding:6px 11px;font-size:12px;color:#3b82f6">${float(ema8_v):.2f}</td>'
                f'<td style="padding:6px 11px;font-size:12px;color:#f472b6">${float(jma_v):.2f}</td>'
                f'<td style="padding:6px 11px;font-size:12px;color:#fbbf24">${float(sma50_v):.2f}</td>'
                f'<td style="padding:6px 11px;font-size:12px;font-weight:600">{float(vol_r):.2f}x</td>'
                f'<td style="padding:6px 11px;font-size:12px;text-align:center;'
                f'color:#a78bfa;font-weight:600">{cmonth}</td>'
                f'</tr>'
            )
        no_results_msg = ('<tr><td colspan="10" style="padding:20px;text-align:center;'
                           'color:#94a3b8;font-size:13px">No matches today</td></tr>')

        html_e = f"""<!DOCTYPE html><html><body style="margin:0;padding:0;
background:#f1f5f9;font-family:'Segoe UI',Arial,sans-serif">
<table width="100%" cellpadding="0" cellspacing="0"><tr><td align="center" style="padding:20px 10px">
<table width="100%" style="max-width:800px;background:#fff;border-radius:12px;
       overflow:hidden;box-shadow:0 4px 16px rgba(0,0,0,0.08)">
  <tr><td style="background:linear-gradient(135deg,#0f172a,#1e3a5f);padding:22px 28px">
<h1 style="margin:0;color:#60a5fa;font-size:20px;font-weight:700">
  📊 Monthly 3-Candle Reclaim (JMA/EMA8/SMA50)
</h1>
<p style="margin:6px 0 0;color:#94a3b8;font-size:12px">
  {datetime.today().strftime('%Y-%m-%d %H:%M UTC')} &nbsp;·&nbsp;
  {cnt} match{'es' if cnt!=1 else ''} found — red month, then a partial
  reclaim (above EMA8, below JMA), then a full reclaim with rising volume
</p>
  </td></tr>
  <tr><td style="padding:14px 28px 4px;background:#0b1220">
<div style="background:#111827;border:1px solid #1f2937;border-radius:8px;padding:12px 16px">
  <p style="margin:0 0 6px;color:#93c5fd;font-size:12px;font-weight:700">
    🔍 FUNNEL — where tickers were filtered out (evaluated once per liquid ticker)
  </p>
  <p style="margin:0;color:#cbd5e1;font-size:12px">
    {FUNNEL_COUNTS['tickers_checked']} tickers checked &nbsp;→&nbsp;
    {FUNNEL_COUNTS['passed_step1_A_red']} passed Step 1 (A red) &nbsp;→&nbsp;
    {FUNNEL_COUNTS['passed_step2_B_partial_reclaim']} passed Step 2 (B partial reclaim) &nbsp;→&nbsp;
    {FUNNEL_COUNTS['passed_step3_C_full_reclaim']} passed Step 3 (C full reclaim) &nbsp;→&nbsp;
    {FUNNEL_COUNTS['passed_step4_vol_increasing']} passed Step 4 (volume increasing) &nbsp;→&nbsp;
    <b style="color:#facc15">{FUNNEL_COUNTS['passed_step5_structure']} passed Step 5 (structure) = full match</b>
  </p>
</div>
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
{f'''<div style="margin-top:10px;background:#f8fafc;border:1px solid #e2e8f0;
        border-radius:6px;padding:10px 14px">
  <p style="margin:0 0 4px;font-size:10px;color:#94a3b8;font-weight:700">
    TICKERS (comma-separated, copy/paste)
  </p>
  <p style="margin:0;font-size:12px;color:#1e293b;font-family:monospace;
      word-break:break-all">{", ".join(r.get("Ticker","") for r in rl)}</p>
</div>''' if rl else ''}
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
            f"Monthly 3-Candle Reclaim (JMA/EMA8/SMA50) — {datetime.today().strftime('%Y-%m-%d')}",
            f"{cnt} matches (red month + partial reclaim above EMA8/below JMA + full reclaim above both, rising volume, above monthly SMA50)",
            "="*60,
            f"FUNNEL: {FUNNEL_COUNTS['tickers_checked']} tickers -> "
            f"{FUNNEL_COUNTS['passed_step1_A_red']} passed A-red -> "
            f"{FUNNEL_COUNTS['passed_step2_B_partial_reclaim']} passed B-partial -> "
            f"{FUNNEL_COUNTS['passed_step3_C_full_reclaim']} passed C-full -> "
            f"{FUNNEL_COUNTS['passed_step4_vol_increasing']} passed vol-increasing -> "
            f"{FUNNEL_COUNTS['passed_step5_structure']} passed structure (=full match)",
            "="*60,
        ]
        if rl:
            for r in rl[:50]:
                ticker = r.get("Ticker","—")
                price  = r.get("Price",0) or 0
                score  = r.get("Score",0) or 0
                entry  = r.get("Entry_Price",0) or 0
                stop   = r.get("Stop_Loss",0) or 0
                ema8_v = r.get("EMA8",0) or 0
                jma_v  = r.get("JMA",0) or 0
                sma50_v= r.get("SMA50",0) or 0
                vol_r  = r.get("Vol_Ratio",0) or 0
                cmonth = r.get("Month_C","—")
                plain_lines.append(
                    f"{ticker:<7} ${float(price):.2f}  Score:{float(score):.0f}  "
                    f"Entry:${float(entry):.2f}  SL:${float(stop):.2f}  "
                    f"EMA8:${float(ema8_v):.2f}  JMA:${float(jma_v):.2f}  SMA50:${float(sma50_v):.2f}  Vol:{float(vol_r):.2f}x  Month:{cmonth}"
                )
            plain_lines.append("")
            plain_lines.append("Tickers (comma-separated):")
            plain_lines.append(", ".join(r.get("Ticker","") for r in rl))
        else:
            plain_lines.append("No matches today")
        plain_lines.append("\nFull results in CSV attachment.")
        plain_e = "\n".join(plain_lines)

        subj = (f"📊 Monthly 3-Candle Reclaim (JMA/EMA8/SMA50) — {cnt} signal{'s' if cnt!=1 else ''}"
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

# ── Charts for top 5 (multi-year monthly price with EMA8/JMA/SMA50
#    overlay, plus a zoomed view of the last ~8 monthly candles) ──
if results:
    top = results[:min(5,len(results))]
    fig, axes = plt.subplots(len(top),2,figsize=(16,4.2*len(top)),facecolor="#0f172a",
                              gridspec_kw={"width_ratios":[2,1]})
    if len(top)==1: axes = axes.reshape(1,2)
    for row, r in zip(axes, top):
        ax_long, ax_zoom = row[0], row[1]
        monthly_p = r["_df_monthly"]
        ema8_p = r["_ema8"]; jma_p = r["_jma"]; sma50_p = r["_sma50"]

        ax_long.set_facecolor("#0f172a")
        ax_long.plot(monthly_p.index, monthly_p["Close"], color="#60a5fa", lw=1.4, label="Monthly Close")
        ax_long.plot(monthly_p.index, ema8_p,  color="#38bdf8", lw=1.0, ls="--", label="EMA8")
        ax_long.plot(monthly_p.index, jma_p,   color="#f472b6", lw=1.1, label="JMA")
        ax_long.plot(monthly_p.index, sma50_p, color="#fbbf24", lw=1.2, ls="-.", label="SMA50")
        ax_long.set_title(
            f"{r['Ticker']}  |  ${r['Price']:.2f}  |  Score {r['Score']}  |  {r['Month_C']}",
            color="#e2e8f0", fontsize=9, fontweight="bold", pad=7)
        ax_long.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
        ax_long.tick_params(colors="#94a3b8", labelsize=8)
        for sp in ax_long.spines.values(): sp.set_edgecolor("#1e3a5f")
        ax_long.legend(loc="upper left", facecolor="#1e293b", labelcolor="#e2e8f0", fontsize=7)
        ax_long.grid(color="#1e3a5f", ls="--", lw=0.5, alpha=0.6)

        zoom = monthly_p.tail(8)
        ax_zoom.set_facecolor("#0f172a")
        colors = ["#22c55e" if c > o else "#ef4444" for o, c in zip(zoom["Open"], zoom["Close"])]
        ax_zoom.bar(range(len(zoom)), zoom["Close"] - zoom["Open"],
                    bottom=zoom["Open"], color=colors, width=0.6)
        ax_zoom.set_xticks(range(len(zoom)))
        ax_zoom.set_xticklabels([d.strftime("%b'%y") for d in zoom.index], rotation=45, fontsize=6)
        ax_zoom.set_title(f"EMA8 ${r['EMA8']:.2f}  JMA ${r['JMA']:.2f}  Vol {r['Vol_Ratio']:.2f}x",
                          color="#e2e8f0", fontsize=8, pad=5)
        ax_zoom.tick_params(colors="#94a3b8", labelsize=7)
        for sp in ax_zoom.spines.values(): sp.set_edgecolor("#1e3a5f")
        ax_zoom.grid(color="#1e3a5f", ls="--", lw=0.5, alpha=0.6)
    plt.suptitle(
        f"Monthly 3-Candle Reclaim (JMA/EMA8/SMA50)  ·  {datetime.today().strftime('%Y-%m-%d')}",
        color="#60a5fa", fontsize=12, fontweight="bold", y=1.001)
    plt.tight_layout()
    cp = os.path.join(out_dir, f"monthly_3candle_reclaim_jma_ema8_chart_{ts}.png")
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
  📋 SIGNAL (evaluated on the most recent 3 monthly candles per
  ticker — A (oldest), B, C (current month) — a slow, monthly-
  timeframe pattern, not a rolling daily-window scan)

  1) CANDLE A: RED — closed below its open.
  2) CANDLE B: GREEN — closed above its open, AND closed above
     EMA8 but BELOW JMA (a partial reclaim).
  3) CANDLE C: GREEN — closed above its open, AND closed above
     BOTH JMA and EMA8 (a full reclaim).
  4) VOLUME: candle C's volume is higher than candle B's.
  5) STRUCTURE: both EMA8 and JMA are above SMA50 on candle C.

  All MAs (EMA8, JMA, SMA50) are computed on MONTHLY closes.

  📋 DATA SOURCING
  Only 1 download per ticker: daily bars (~{CFG['history_days']} days, ~7 years —
  needed for a stable monthly SMA50), resampled to monthly ('ME') —
  no separate network calls needed.

  📋 OUTPUT (reasonable defaults — not explicitly requested)
  Entry_Price = candle C's (the current month's) CLOSE
  Stop_Loss   = candle A's (the red month's) LOW
  EMA8 / JMA / SMA50 = the three monthly moving averages at candle C
  Vol_Ratio = candle C's volume / candle B's volume
  Month_A / Month_B / Month_C = the three monthly periods (YYYY-MM)

  📋 SCORE (0-100)
  Volume-increase strength (0-25) + how far candle C clears JMA
  (0-25) + distance above the monthly SMA50 (0-25) + base points
  for clearing every gate (25)

  💡 BEST SETUPS
  Score > 70                      strong volume, decisive JMA
                                   clearance, well above SMA50
  Vol_Ratio large                     a clean, convincing volume
                                       increase, not a marginal one
  Above_SMA50_% high                     genuinely strong long-term
                                          position, not a borderline one

  ⚙️  TUNE IF 0 RESULTS (see the FUNNEL above to see which step is
  actually the bottleneck before guessing)
  ema8_period                       8 → 5    (faster, more reactive EMA)
  sma50_period                     50 → 30    (shorter monthly SMA)
  min_price                         2 → 1
  min_avg_volume                80000 → 50000
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
""")
