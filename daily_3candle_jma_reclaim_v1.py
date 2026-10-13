# ============================================================
# NASDAQ — Daily 3-Candle JMA Reclaim (v1)
# ============================================================
#
# SIGNAL (all required). Call the last 3 daily candles A (oldest),
# B, C (most recent). JMA = JMA(13, phase 40, power 2), evaluated on
# each candle's own bar:
#
#   1. CANDLE A: CLOSE BELOW JMA.
#
#   2. CANDLE B: CLOSE ABOVE JMA, AND its LOW is ABOVE JMA — the
#      whole candle is clear of JMA (a clean reclaim, not a wick
#      through it).
#
#   3. CANDLE C: CLOSE ABOVE candle B's close — follow-through.
#
#   4. CANDLE C: LOW ABOVE JMA — the follow-through day never touches
#      JMA.
#
# By default candle C must be the most recent daily candle
# (signal_lookback_days = 1). Raise signal_lookback_days to also catch
# patterns that completed a few days ago (Days_Since_Signal shows how
# long ago).
#
# DATA — only 1 download per ticker: daily bars.
#
# OUTPUT: Entry_Price and Stop_Loss are reasonable defaults (Entry =
# candle C's close; Stop = the lower of candle B's and C's lows), not
# explicitly requested.
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

    "signal_lookback_days"   : 1,    # 1 = candle C must be the most recent
                                      # daily candle; raise to catch patterns
                                      # that completed a few days ago

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

def check_pattern_at(df, close, jma, i, cfg):
    """
    Checks the 3-candle JMA reclaim with candle C (the follow-through
    day) anchored at bar `i`. Candle B is i-1, candle A is i-2. This
    is the SINGLE SOURCE OF TRUTH for the pattern logic.

    Computes EVERY stage's pass/fail regardless of earlier failures
    (guarding only on missing data), so callers can diagnose exactly
    which stage is filtering results out — see FUNNEL_COUNTS below.

    Returns (passed: bool, details: dict).
    """
    if i < 2 or i >= len(df):
        return False, {}

    cA, cB, cC = close.iloc[i-2], close.iloc[i-1], close.iloc[i]
    lowA, lowB, lowC = df["Low"].iloc[i-2], df["Low"].iloc[i-1], df["Low"].iloc[i]
    oC = df["Open"].iloc[i] if "Open" in df.columns else np.nan
    jA, jB, jC = jma.iloc[i-2], jma.iloc[i-1], jma.iloc[i]
    if any(np.isnan(v) for v in [cA, cB, cC, lowB, lowC, jA, jB, jC]):
        return False, {}

    # ── Step 1: candle A closes below JMA ────────────────────────
    A_below = bool(cA < jA)

    # ── Step 2: candle B closes above JMA with its low above JMA ──
    B_close_above = bool(cB > jB)
    B_low_above   = bool(lowB > jB)
    B_clean       = B_close_above and B_low_above

    # ── Step 3: candle C closes above candle B's close ────────────
    C_higher_close = bool(cC > cB)

    # ── Step 4: candle C never touches JMA (low above JMA) ────────
    C_no_touch = bool(lowC > jC)

    passed = A_below and B_clean and C_higher_close and C_no_touch

    return passed, {
        "idx": i, "cA": float(cA), "cB": float(cB), "cC": float(cC),
        "oC": float(oC), "lowA": float(lowA), "lowB": float(lowB), "lowC": float(lowC),
        "jA": float(jA), "jB": float(jB), "jC": float(jC),
        "date_A": df.index[i-2], "date_B": df.index[i-1], "date_C": df.index[i],
        "A_below": A_below, "B_close_above": B_close_above, "B_low_above": B_low_above,
        "B_clean": B_clean, "C_higher_close": C_higher_close, "C_no_touch": C_no_touch,
    }

# ── Diagnostic funnel — tallies how far each ticker-day gets through
#    the pattern, across the full universe scan, so a 0-match run can
#    be diagnosed empirically instead of guessed at ─────────────────
FUNNEL_COUNTS = {
    "days_checked": 0,
    "passed_step1_A_close_below_jma": 0,
    "passed_step2_B_clean_above_jma": 0,
    "passed_step3_C_close_above_B": 0,
    "passed_step4_C_no_jma_touch": 0,
}

def find_jma_reclaim_signals(df, close, jma, cfg):
    """
    Checks the last `signal_lookback_days` trading days (1 = the most
    recent candle only) for the full pattern. Returns a list of hit
    dicts, most recent first. Also tallies FUNNEL_COUNTS for every
    day checked, regardless of match.
    """
    global FUNNEL_COUNTS
    n = len(df)
    lb = cfg["signal_lookback_days"]
    hits = []
    for back in range(0, lb):
        i = (n - 1) - back
        passed, details = check_pattern_at(df, close, jma, i, cfg)
        if not details:
            continue
        FUNNEL_COUNTS["days_checked"] += 1
        if details["A_below"]:
            FUNNEL_COUNTS["passed_step1_A_close_below_jma"] += 1
            if details["B_clean"]:
                FUNNEL_COUNTS["passed_step2_B_clean_above_jma"] += 1
                if details["C_higher_close"]:
                    FUNNEL_COUNTS["passed_step3_C_close_above_B"] += 1
                    if details["C_no_touch"]:
                        FUNNEL_COUNTS["passed_step4_C_no_jma_touch"] += 1
        if passed:
            hits.append(details)
    return hits

# ── Technical signal: daily 3-candle JMA reclaim ──────────────────
def analyze_jma_reclaim(sym, df_daily):
    """
    Returns dict with score and setup details, or None if the pattern
    is not present in the lookback window.
    """
    if df_daily is None:
        return None

    price   = float(df_daily["Close"].iloc[-1])
    avg_vol = float(df_daily["Volume"].tail(20).mean())
    if price   < CFG["min_price"]:      return None
    if avg_vol < CFG["min_avg_volume"]: return None

    n = len(df_daily)
    if n < 60 + CFG["signal_lookback_days"]:   # let JMA settle
        return None

    close = df_daily["Close"]
    jma   = calc_jma(close, CFG["jma_period"], CFG["jma_phase"], CFG["jma_power"])

    hits = find_jma_reclaim_signals(df_daily, close, jma, CFG)
    if not hits:
        return None

    sig = hits[0]   # most recent
    sig_idx = sig["idx"]
    days_since_signal = (n - 1) - sig_idx
    recent_signals = [
        {"date": df_daily.index[h["idx"]], "bars_ago": (n-1)-h["idx"]}
        for h in hits
    ]

    # ── Entry / Stop (reasonable defaults — see header note) ─────────
    entry_price = sig["cC"]
    stop_loss   = min(sig["lowB"], sig["lowC"])
    if entry_price <= stop_loss:
        return None
    risk_pct = (entry_price - stop_loss) / entry_price * 100 if entry_price > 0 else 0

    A_below_pct    = (sig["jA"] - sig["cA"]) / sig["jA"] * 100 if sig["jA"] > 0 else 0
    B_above_pct    = (sig["cB"] - sig["jB"]) / sig["jB"] * 100 if sig["jB"] > 0 else 0
    C_gain_pct     = (sig["cC"] - sig["cB"]) / sig["cB"] * 100 if sig["cB"] > 0 else 0
    C_clear_pct    = (sig["lowC"] - sig["jC"]) / sig["jC"] * 100 if sig["jC"] > 0 else 0
    vol_C   = float(df_daily["Volume"].iloc[sig_idx])
    avg20   = float(df_daily["Volume"].iloc[max(0, sig_idx-20):sig_idx].mean())
    vol_vs_avg = vol_C / avg20 if avg20 > 0 else 0.0

    # ── Score (0-100) ────────────────────────────────────────────
    score = 20   # base for clearing every gate
    reasons = []
    score += max(0, min(20, B_above_pct * 5))
    reasons.append(f"B>JMA+{B_above_pct:.1f}%")
    score += max(0, min(20, C_gain_pct * 8))
    reasons.append(f"C>B+{C_gain_pct:.1f}%")
    score += max(0, min(15, C_clear_pct * 5))
    reasons.append(f"C-low clears JMA+{C_clear_pct:.1f}%")
    score += max(0, min(15, (vol_vs_avg - 1) * 15))
    reasons.append(f"Vol/20d x{vol_vs_avg:.2f}")
    if B_above_pct > 8:
        score -= 10; reasons.append("Extended")
    score += max(0, 10 - days_since_signal * 3)
    reasons.append(f"{days_since_signal}dAgo")
    score = round(min(100, max(0, score)))

    return {
        "Score"          : score,
        "Price"          : round(price, 2),
        "Entry_Price"    : round(entry_price, 2),
        "Stop_Loss"      : round(stop_loss, 2),
        "Risk_%"         : round(risk_pct, 1),
        "JMA"            : round(sig["jC"], 2),
        "A_Below_JMA_%"  : round(A_below_pct, 1),
        "B_Above_JMA_%"  : round(B_above_pct, 1),
        "C_Gain_%"       : round(C_gain_pct, 1),
        "C_Low_Clear_%"  : round(C_clear_pct, 1),
        "Vol_vs_20d_Avg" : round(vol_vs_avg, 2),
        "Signal_Date"    : df_daily.index[sig_idx].strftime("%Y-%m-%d"),
        "Days_Since_Signal"  : days_since_signal,
        "Recent_Signal_Count": len(recent_signals),
        "Recent_Signals" : " | ".join(
            f"{s['date'].strftime('%Y-%m-%d')}" for s in recent_signals),
        "Flags"          : " | ".join(reasons),
        "_df_daily"      : df_daily,
        "_jma"           : jma,
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
             "B_Above_JMA_%","C_Gain_%","Signal_Date"]
_CW = {"Ticker":8,"Price":10,"Score":7,"Entry_Price":12,"Stop_Loss":11,
       "B_Above_JMA_%":14,"C_Gain_%":10,"Signal_Date":13}
_CF = {"Price":"${:.2f}","Score":"{:.0f}","Entry_Price":"${:.2f}",
       "Stop_Loss":"${:.2f}"}
_hdr_done = False

def _live_header():
    global _hdr_done
    if _hdr_done: return
    print("\n" + "━"*95)
    print("  📊  LIVE MATCHES  —  each stock printed the moment it passes all 4 steps")
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
print("  Pattern: A closes below JMA → B closes above JMA with low above")
print("  JMA → C closes above B and its low never touches JMA\n")

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

for sym in tqdm(list(daily_map.keys()), desc="Checking JMA reclaim", unit="stk"):
    try:
        r = analyze_jma_reclaim(sym, daily_map[sym])
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

# ── Diagnostic funnel — where ticker-days were filtered out ──────
print(f"\n{'━'*65}")
print(f"  🔍 FUNNEL — where ticker-days were filtered out")
print(f"  (tallied across every liquid ticker's last {CFG['signal_lookback_days']} trading day(s))")
print(f"{'━'*65}")
fc = FUNNEL_COUNTS
print(f"  Ticker-days checked                          : {fc['days_checked']}")
print(f"  Step 1 — candle A close below JMA            : {fc['passed_step1_A_close_below_jma']}")
print(f"  Step 2 — + candle B close & low above JMA    : {fc['passed_step2_B_clean_above_jma']}")
print(f"  Step 3 — + candle C close above B's close    : {fc['passed_step3_C_close_above_B']}")
print(f"  Step 4 — + candle C low never touches JMA    : {fc['passed_step4_C_no_jma_touch']}  (= full pattern)")
print(f"{'━'*65}")

if not results:
    print("\n  No matches. Try relaxing (see the FUNNEL above to see which")
    print("  step is actually the bottleneck before guessing):")
    print("   signal_lookback_days            1 → 3     (patterns completed in the last 3 days)")
    print("   min_price                      10 → 5")
    print("   min_avg_volume             100000 → 50000")

results.sort(key=lambda x: x["Score"], reverse=True)

# ── Always build df_out and save/email (even if 0 results) ────
ts      = datetime.today().strftime("%Y%m%d_%H%M")
out_dir = os.environ.get("GITHUB_WORKSPACE", os.getcwd())

COLS = [
    "Ticker","Price","Score",
    "Entry_Price","Stop_Loss","Risk_%",
    "JMA","A_Below_JMA_%","B_Above_JMA_%","C_Gain_%","C_Low_Clear_%","Vol_vs_20d_Avg",
    "Signal_Date","Days_Since_Signal","Recent_Signal_Count","Recent_Signals",
    "Flags",
]
df_out = pd.DataFrame([{k:v for k,v in r.items() if not k.startswith("_")}
                        for r in results]) if results else pd.DataFrame(columns=COLS)
if not df_out.empty:
    df_out = df_out[[c for c in COLS if c in df_out.columns]]
    df_out.reset_index(drop=True, inplace=True)

FMT = {
    "Price"         : lambda v: f"${v:.2f}",
    "Score"         : lambda v: f"{v:.0f}",
    "Entry_Price"   : lambda v: f"${v:.2f}",
    "Stop_Loss"     : lambda v: f"${v:.2f}",
    "Risk_%"        : lambda v: f"{v:.1f}%",
    "JMA"           : lambda v: f"${v:.2f}",
    "A_Below_JMA_%" : lambda v: f"{v:.1f}%",
    "B_Above_JMA_%" : lambda v: f"{v:+.1f}%",
    "C_Gain_%"      : lambda v: f"{v:+.1f}%",
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

if _IN_NOTEBOOK and results:
    DISP = ["Ticker","Price","Score","Entry_Price","Stop_Loss",
            "B_Above_JMA_%","C_Gain_%","C_Low_Clear_%","Signal_Date"]
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
            tds += (f'<td style="padding:7px 12px;font-size:12px;'
                    f'border-bottom:1px solid #e2e8f0;white-space:nowrap;{sty}">'
                    f'{disp}</td>')
        rows_html += f'<tr style="background:{bg}">{tds}</tr>\n'

    display_html(f"""
<div style="background:linear-gradient(135deg,#0f172a,#1e3a5f);
        border-radius:10px;padding:18px 24px;margin-bottom:8px;
        font-family:'Segoe UI',Arial,sans-serif">
  <h2 style="margin:0;color:#60a5fa;font-size:20px;font-weight:700">
    📈 Daily 3-Candle JMA Reclaim
  </h2>
  <p style="margin:6px 0 0;color:#94a3b8;font-size:12px">
    {datetime.today().strftime('%Y-%m-%d %H:%M')} &nbsp;·&nbsp;
    <b style="color:#22c55e">{len(results)} matches</b> from {len(TICKERS)} tickers
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
    CLI_COLS = ["Ticker","Price","Score","Entry_Price","Stop_Loss",
                "B_Above_JMA_%","C_Gain_%","C_Low_Clear_%","Signal_Date"]
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
    tit = f"  Daily 3-Candle JMA Reclaim   {datetime.today().strftime('%Y-%m-%d')}   {len(df_out)} matches"
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
  Score           0-100 (B's reclaim strength + C's follow-through
                  + C's clearance above JMA + volume + freshness)
  Entry_Price     candle C's close
  Stop_Loss       lower of candle B's and C's lows
  A_Below_JMA_%   how far candle A closed below JMA
  B_Above_JMA_%   how far candle B closed above JMA
  C_Gain_%        candle C's close vs candle B's close
  C_Low_Clear_%   how far candle C's low stayed above JMA
  Signal_Date     date of candle C
  ──────────────────────────────────────────────────────""")

# Save
fpath = os.path.join(out_dir, f"daily_3candle_jma_reclaim_{ts}.csv")
df_out.to_csv(fpath, index=False)
print(f"\n  💾 CSV → {fpath}")
tv = os.path.join(out_dir, f"tv_daily_3candle_jma_reclaim_{ts}.txt")
with open(tv,"w") as f:
    f.write(f"###Daily 3-Candle JMA Reclaim {datetime.today().strftime('%Y-%m-%d')}\n")
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

        heads = ["Ticker","Price","Score","Entry_Price","Stop_Loss",
                 "B_Above_JMA_%","C_Gain_%","C_Low_Clear_%","Signal_Date"]
        th_e = "".join(
            f'<th style="background:#1e293b;color:#e2e8f0;padding:8px 11px;'
            f'font-size:11px;font-weight:700;border-bottom:2px solid #3b82f6;'
            f'white-space:nowrap">{c}</th>'
            for c in heads
        )
        rows_e = ""
        for i, r in enumerate(rl[:50]):
            bg  = "#fff" if i % 2 == 0 else "#f0f9ff"
            rows_e += (
                f'<tr style="background:{bg}">'
                f'<td style="padding:6px 11px;font-size:12px;font-weight:700">{r.get("Ticker","—")}</td>'
                f'<td style="padding:6px 11px;font-size:12px">${float(r.get("Price",0) or 0):.2f}</td>'
                f'<td style="padding:6px 11px;font-size:12px;font-weight:700;'
                f'background:#166534;color:#fff;text-align:center">{float(r.get("Score",0) or 0):.0f}</td>'
                f'<td style="padding:6px 11px;font-size:12px;color:#22c55e">${float(r.get("Entry_Price",0) or 0):.2f}</td>'
                f'<td style="padding:6px 11px;font-size:12px;color:#ef4444">${float(r.get("Stop_Loss",0) or 0):.2f}</td>'
                f'<td style="padding:6px 11px;font-size:12px;color:#3b82f6">{float(r.get("B_Above_JMA_%",0) or 0):+.1f}%</td>'
                f'<td style="padding:6px 11px;font-size:12px">{float(r.get("C_Gain_%",0) or 0):+.1f}%</td>'
                f'<td style="padding:6px 11px;font-size:12px;font-weight:600">{float(r.get("C_Low_Clear_%",0) or 0):+.1f}%</td>'
                f'<td style="padding:6px 11px;font-size:12px;text-align:center;'
                f'color:#a78bfa;font-weight:600">{r.get("Signal_Date","—")}</td>'
                f'</tr>'
            )
        no_results_msg = ('<tr><td colspan="9" style="padding:20px;text-align:center;'
                           'color:#94a3b8;font-size:13px">No matches today</td></tr>')
        more_note = (f'Showing top 50 of {cnt} — full list in the CSV attachment'
                     if cnt > 50 else '📎 Full results attached as CSV')

        fc = FUNNEL_COUNTS
        html_e = f"""<!DOCTYPE html><html><body style="margin:0;padding:0;
background:#f1f5f9;font-family:'Segoe UI',Arial,sans-serif">
<table width="100%" cellpadding="0" cellspacing="0"><tr><td align="center" style="padding:20px 10px">
<table width="100%" style="max-width:800px;background:#fff;border-radius:12px;
       overflow:hidden;box-shadow:0 4px 16px rgba(0,0,0,0.08)">
  <tr><td style="background:linear-gradient(135deg,#0f172a,#1e3a5f);padding:22px 28px">
<h1 style="margin:0;color:#60a5fa;font-size:20px;font-weight:700">
  📊 Daily 3-Candle JMA Reclaim
</h1>
<p style="margin:6px 0 0;color:#94a3b8;font-size:12px">
  {datetime.today().strftime('%Y-%m-%d %H:%M UTC')} &nbsp;·&nbsp;
  {cnt} match{'es' if cnt!=1 else ''} — A closes below JMA, B closes above JMA
  with its low above JMA, C closes above B without touching JMA
</p>
  </td></tr>
  <tr><td style="padding:14px 28px 4px;background:#0b1220">
<div style="background:#111827;border:1px solid #1f2937;border-radius:8px;padding:12px 16px">
  <p style="margin:0 0 6px;color:#93c5fd;font-size:12px;font-weight:700">
    🔍 FUNNEL — last {CFG['signal_lookback_days']} trading day(s), every liquid ticker
  </p>
  <p style="margin:0;color:#cbd5e1;font-size:12px">
    {fc['days_checked']} ticker-days checked &nbsp;→&nbsp;
    {fc['passed_step1_A_close_below_jma']} A close &lt; JMA &nbsp;→&nbsp;
    {fc['passed_step2_B_clean_above_jma']} B close &amp; low &gt; JMA &nbsp;→&nbsp;
    {fc['passed_step3_C_close_above_B']} C close &gt; B close &nbsp;→&nbsp;
    <b style="color:#facc15">{fc['passed_step4_C_no_jma_touch']} C low &gt; JMA = full match</b>
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
<p style="font-size:11px;color:#64748b;margin:8px 0 0">{more_note}</p>
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
  ⚠️ Not financial advice &nbsp;·&nbsp; Entry = candle C close, Stop = lower of B/C lows (defaults) &nbsp;·&nbsp; Auto-generated by GitHub Actions
</p>
  </td></tr>
</table>
</td></tr></table>
</body></html>"""

        plain_lines = [
            f"Daily 3-Candle JMA Reclaim — {datetime.today().strftime('%Y-%m-%d')}",
            f"{cnt} matches (A close < JMA; B close & low > JMA; C close > B close, C low > JMA)",
            "="*60,
            f"FUNNEL: {fc['days_checked']} ticker-days -> "
            f"{fc['passed_step1_A_close_below_jma']} A<JMA -> "
            f"{fc['passed_step2_B_clean_above_jma']} B clean above JMA -> "
            f"{fc['passed_step3_C_close_above_B']} C>B -> "
            f"{fc['passed_step4_C_no_jma_touch']} C no JMA touch (=full match)",
            "="*60,
        ]
        if rl:
            for r in rl[:50]:
                plain_lines.append(
                    f"{r.get('Ticker','—'):<7} ${float(r.get('Price',0) or 0):.2f}  "
                    f"Score:{float(r.get('Score',0) or 0):.0f}  "
                    f"Entry:${float(r.get('Entry_Price',0) or 0):.2f}  "
                    f"SL:${float(r.get('Stop_Loss',0) or 0):.2f}  "
                    f"B>JMA:{float(r.get('B_Above_JMA_%',0) or 0):+.1f}%  "
                    f"C>B:{float(r.get('C_Gain_%',0) or 0):+.1f}%  "
                    f"C-low clear:{float(r.get('C_Low_Clear_%',0) or 0):+.1f}%  "
                    f"Signal:{r.get('Signal_Date','—')}"
                )
            plain_lines.append("")
            plain_lines.append("Tickers (comma-separated):")
            plain_lines.append(", ".join(r.get("Ticker","") for r in rl))
        else:
            plain_lines.append("No matches today")
        plain_lines.append("\nFull results in CSV attachment.")
        plain_e = "\n".join(plain_lines)

        subj = (f"📊 Daily 3-Candle JMA Reclaim — {cnt} signal{'s' if cnt!=1 else ''}"
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

# ── Charts for top 5 (price + JMA, plus a zoomed candle view of the
#    last 10 days showing the 3-candle reclaim against JMA) ────────
if results:
    top = results[:min(5,len(results))]
    fig, axes = plt.subplots(len(top),2,figsize=(16,4.2*len(top)),facecolor="#0f172a",
                              gridspec_kw={"width_ratios":[3,1]})
    if len(top)==1: axes = axes.reshape(1,2)
    for row, r in zip(axes, top):
        ax, ax_zoom = row[0], row[1]
        df_p = r["_df_daily"].tail(90).copy()
        jma_p = r["_jma"].reindex(df_p.index)
        ax.set_facecolor("#0f172a")
        ax.plot(df_p.index, df_p["Close"], color="#60a5fa", lw=1.6, label="Close", zorder=5)
        ax.plot(df_p.index, jma_p, color="#22c55e", lw=1.4, label="JMA 13", zorder=4)
        ax.axhline(r["Stop_Loss"], color="#ef4444", lw=1.0, ls="--", alpha=0.85,
                  label=f"Stop {r['Stop_Loss']:.2f}")
        ax.axhline(r["Entry_Price"], color="#facc15", lw=1.0, ls="--", alpha=0.85,
                  label=f"Entry {r['Entry_Price']:.2f}")
        ax.set_title(
            f"{r['Ticker']}  |  {r['Price']:.2f} USD  |  Score {r['Score']}  |  {r['Signal_Date']}",
            color="#e2e8f0", fontsize=9, fontweight="bold", pad=7)
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))
        ax.tick_params(colors="#94a3b8", labelsize=8)
        for sp in ax.spines.values(): sp.set_edgecolor("#1e3a5f")
        ax.legend(loc="upper left", facecolor="#1e293b", labelcolor="#e2e8f0",
                  fontsize=6, framealpha=0.9, ncol=2)
        ax.grid(color="#1e3a5f", ls="--", lw=0.5, alpha=0.6)

        zoom = df_p.tail(10)
        zj   = jma_p.reindex(zoom.index)
        x    = np.arange(len(zoom))
        ax_zoom.set_facecolor("#0f172a")
        colors = ["#22c55e" if c >= o else "#ef4444" for o, c in zip(zoom["Open"], zoom["Close"])]
        ax_zoom.vlines(x, zoom["Low"], zoom["High"], color=colors, lw=1)
        ax_zoom.bar(x, (zoom["Close"] - zoom["Open"]).abs().clip(lower=1e-9),
                    bottom=np.minimum(zoom["Open"], zoom["Close"]), color=colors, width=0.6)
        ax_zoom.plot(x, zj.values, color="#22c55e", lw=1.4, ls="-", alpha=0.9)
        ax_zoom.set_xticks(x)
        ax_zoom.set_xticklabels([d.strftime("%m/%d") for d in zoom.index], rotation=45, fontsize=6)
        ax_zoom.set_title(f"B>JMA {r['B_Above_JMA_%']:+.1f}%  C>B {r['C_Gain_%']:+.1f}%",
                          color="#e2e8f0", fontsize=8, pad=5)
        ax_zoom.tick_params(colors="#94a3b8", labelsize=7)
        for sp in ax_zoom.spines.values(): sp.set_edgecolor("#1e3a5f")
        ax_zoom.grid(color="#1e3a5f", ls="--", lw=0.5, alpha=0.6)
    plt.suptitle(
        f"Daily 3-Candle JMA Reclaim  ·  {datetime.today().strftime('%Y-%m-%d')}",
        color="#60a5fa", fontsize=12, fontweight="bold", y=1.001)
    plt.tight_layout()
    cp = os.path.join(out_dir, f"daily_3candle_jma_reclaim_chart_{ts}.png")
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
  📋 SIGNAL (all required). The last 3 daily candles:
  A (oldest), B, C (most recent). JMA = JMA(13, phase 40).

  1) CANDLE A: closes BELOW JMA.
  2) CANDLE B: closes ABOVE JMA, and its LOW is ABOVE JMA.
  3) CANDLE C: closes ABOVE candle B's close.
  4) CANDLE C: its LOW stays ABOVE JMA (never touches it).

  signal_lookback_days = {CFG['signal_lookback_days']}  (1 = C must be the latest candle)

  📋 OUTPUT (reasonable defaults — not explicitly requested)
  Entry_Price = candle C's CLOSE
  Stop_Loss   = the lower of candle B's and C's LOWS
  Universe    = price >= 10 USD, 20d avg volume >= 100k

  📋 SCORE (0-100)
  Base 20 + B's close above JMA (0-20) + C's gain over B (0-20)
  + C's low clearance above JMA (0-15) + volume vs 20d avg (0-15)
  + freshness (0-10); -10 if B closed > 8% above JMA (extended)

  ⚙️  TUNE IF 0 RESULTS
  signal_lookback_days            1 → 3
  min_price                      10 → 5
  min_avg_volume             100000 → 50000
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
""")
