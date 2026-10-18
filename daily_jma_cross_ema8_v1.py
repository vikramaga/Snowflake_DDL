# ============================================================
# NASDAQ — Daily JMA Cross Above EMA8 + Confirmation (v1)
# ============================================================
#
# SIGNAL (all required). JMA = repo-standard JMA(13, phase 40, power 2)
# (corrected calc_jma, steady-state gain 1.0); EMA8 = 8-day EMA of close.
#
#   1. CROSS DAY: JMA was at or below EMA8 on the previous day and is
#      ABOVE EMA8 on the cross day (JMA crossed above EMA8 from below).
#
#   2. CONFIRM DAY (the day after the cross): the daily CLOSE is above
#      JMA. The confirm day is the signal day.
#
#   3. VALID STOP: the confirm-day close is above the CROSS PRICE, so
#      the stop sits below the entry.
#
# CROSS PRICE = the price where JMA and EMA8 intersect. The two lines
# cross between the day before the cross and the cross day; the exact
# point is found by straight-line interpolation of both lines between
# those two bars. It is the STOP LOSS (as requested).
#
# By default the confirm day must be the most recent daily candle
# (signal_lookback_days = 1).
#
# DATA — only 1 download per ticker: daily bars.
#
# OUTPUT: Stop_Loss = the cross price (requested). Entry_Price = the
# confirm-day close (reasonable default, not explicitly requested).
# Quantity / Total_Amount come from position_sizing.py (max loss $40).
# SMA50 is shown for trend context and adds a small score bonus; it is
# NOT a gate.
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

# ── CONFIG ────────────────────────────────────────────────────────────
CFG = {
    "history_days"          : 550,   # daily bars — ~380 trading days, comfortable
                                      # margin above _clean()'s 220-bar floor

    # ── Indicator periods ───────────────────────────────────────
    "jma_period"             : 13,
    "jma_phase"              : 40,
    "jma_power"              : 2,
    "ema_period"             : 8,
    "sma_period"             : 50,   # trend context only (score bonus, not a gate)

    "signal_lookback_days"   : 1,    # 1 = the confirm day must be the most
                                      # recent daily candle

    # ── Filters ─────────────────────────────────────────────────
    "min_avg_volume"         : 100_000,
    "min_price"              : 10.0,

    "batch_size"             : 50,
    "batch_sleep"            : 1.5,
}

# ── Indicators ───────────────────────────────────────────────────────
def calc_ema(close, n):
    return close.ewm(span=n, adjust=False).mean()

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

def cross_price(jma, ema, x):
    """
    Price where JMA and EMA8 intersect between bar x-1 and bar x
    (straight-line interpolation of both lines). Returns (price, t)
    with t in [0, 1] = fraction of the way from bar x-1 to bar x.
    """
    j0, j1 = float(jma.iloc[x-1]), float(jma.iloc[x])
    e0, e1 = float(ema.iloc[x-1]), float(ema.iloc[x])
    d0, d1 = j0 - e0, j1 - e1          # d0 <= 0 < d1 on a cross
    t = d0 / (d0 - d1) if d0 != d1 else 1.0
    t = min(1.0, max(0.0, t))
    return e0 + t * (e1 - e0), t

def check_pattern_at(close, jma, ema, i, cfg):
    """
    Checks the JMA-over-EMA8 cross + confirmation with the CONFIRM day
    at bar `i` (the cross day is bar i-1).
    SINGLE SOURCE OF TRUTH for the pattern logic.

    Computes EVERY stage's pass/fail regardless of earlier failures
    (guarding only on missing data), so callers can diagnose exactly
    which stage is filtering results out — see FUNNEL_COUNTS below.

    Returns (passed: bool, details: dict).
    """
    x = i - 1                                   # cross day
    if x - 1 < 0 or i >= len(close):
        return False, {}
    vals = [jma.iloc[x-1], jma.iloc[x], ema.iloc[x-1], ema.iloc[x], jma.iloc[i], close.iloc[i]]
    if any(np.isnan(v) for v in vals):
        return False, {}

    crossed = bool(jma.iloc[x-1] <= ema.iloc[x-1] and jma.iloc[x] > ema.iloc[x])   # Step 1
    cp, t   = cross_price(jma, ema, x) if crossed else (np.nan, np.nan)
    c_i     = float(close.iloc[i])
    confirm = bool(c_i > float(jma.iloc[i]))                                       # Step 2
    valid   = bool(crossed and c_i > cp)                                           # Step 3

    passed = crossed and confirm and valid
    return passed, {
        "idx": i, "cross_idx": x, "cross_price": cp, "cross_t": t,
        "close": c_i, "jma": float(jma.iloc[i]), "ema": float(ema.iloc[i]),
        "crossed": crossed, "confirm": confirm, "valid": valid,
    }

# ── Diagnostic funnel — tallies how far each ticker-day gets through
#    the pattern, across the full universe scan, so a 0-match run can
#    be diagnosed empirically instead of guessed at ─────────────────
FUNNEL_COUNTS = {
    "days_checked": 0,
    "passed_step1_jma_crossed_above_ema8": 0,
    "passed_step2_next_close_above_jma": 0,
    "passed_step3_close_above_cross_price": 0,
}

def find_jma_crosses(close, jma, ema, cfg):
    """
    Checks the last `signal_lookback_days` trading days as the confirm
    day (1 = the most recent candle only). Returns hit dicts, most
    recent first, and tallies FUNNEL_COUNTS for every day checked.
    """
    global FUNNEL_COUNTS
    n = len(close)
    hits = []
    for back in range(0, cfg["signal_lookback_days"]):
        i = (n - 1) - back
        passed, d = check_pattern_at(close, jma, ema, i, cfg)
        if not d:
            continue
        FUNNEL_COUNTS["days_checked"] += 1
        if d["crossed"]:
            FUNNEL_COUNTS["passed_step1_jma_crossed_above_ema8"] += 1
            if d["confirm"]:
                FUNNEL_COUNTS["passed_step2_next_close_above_jma"] += 1
                if d["valid"]:
                    FUNNEL_COUNTS["passed_step3_close_above_cross_price"] += 1
        if passed:
            hits.append(d)
    return hits

# ── Technical signal: JMA cross above EMA8 + confirmation ───────────
def analyze_jma_cross(sym, df_daily):
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
    if n < CFG["sma_period"] + 30:
        return None

    close = df_daily["Close"]
    jma   = calc_jma(close, CFG["jma_period"], CFG["jma_phase"], CFG["jma_power"])
    ema   = calc_ema(close, CFG["ema_period"])

    hits = find_jma_crosses(close, jma, ema, CFG)
    if not hits:
        return None

    s = hits[0]   # most recent
    k, x = s["idx"], s["cross_idx"]
    days_since_signal = (n - 1) - k

    # ── Entry / Stop: Stop = cross price (requested); Entry = confirm close
    entry_price = s["close"]
    stop_loss   = s["cross_price"]
    if price <= stop_loss:              # (lookback > 1) already back below the stop
        return None
    risk_pct = (entry_price - stop_loss) / entry_price * 100

    gap_pct   = (s["jma"] / s["ema"] - 1) * 100                 # JMA above EMA8 on confirm day
    above_jma = (entry_price / s["jma"] - 1) * 100              # close above JMA on confirm day
    sma   = close.rolling(CFG["sma_period"]).mean()
    sma_k = float(sma.iloc[k])
    above_sma = bool(entry_price > sma_k)
    vol_k  = float(df_daily["Volume"].iloc[k])
    avg20  = float(df_daily["Volume"].iloc[max(0, k-20):k].mean())
    vol_vs_avg = vol_k / avg20 if avg20 > 0 else 0.0

    # ── Score (0-100) ────────────────────────────────────────────
    score = 30   # base for clearing every gate
    reasons = []
    score += max(0, min(15, gap_pct * 15))
    reasons.append(f"JMA>EMA8 +{gap_pct:.2f}%")
    if 0 < above_jma <= 3:
        score += 15; reasons.append(f"Close +{above_jma:.1f}% over JMA")
    elif above_jma > 6:
        score -= 5;  reasons.append(f"Extended +{above_jma:.1f}% over JMA")
    else:
        score += 8;  reasons.append(f"Close +{above_jma:.1f}% over JMA")
    score += max(0, min(15, (8 - risk_pct) / 8 * 15))
    reasons.append(f"Risk {risk_pct:.1f}%")
    score += max(0, min(15, (vol_vs_avg - 1) * 15))
    reasons.append(f"Vol/20d x{vol_vs_avg:.2f}")
    if above_sma:
        score += 10; reasons.append(f"Above SMA{CFG['sma_period']}")
    else:
        reasons.append(f"Below SMA{CFG['sma_period']}")
    if risk_pct > 12:
        score -= 10; reasons.append("WideStop")
    score = round(min(100, max(0, score)))

    return {
        "Score"          : score,
        "Price"          : round(price, 2),
        "Entry_Price"    : round(entry_price, 2),
        "Stop_Loss"      : round(stop_loss, 2),
        "Risk_%"         : round(risk_pct, 1),
        "Cross_Price"    : round(stop_loss, 2),
        "Cross_Date"     : df_daily.index[x].strftime("%Y-%m-%d"),
        "Confirm_Date"   : df_daily.index[k].strftime("%Y-%m-%d"),
        "JMA"            : round(s["jma"], 2),
        "EMA8"           : round(s["ema"], 2),
        "Close_vs_JMA_%" : round(above_jma, 2),
        "JMA_EMA8_Gap_%" : round(gap_pct, 2),
        "Vol_vs_20d_Avg" : round(vol_vs_avg, 2),
        "SMA50"          : round(sma_k, 2),
        "Above_SMA50"    : above_sma,
        "Signal_Date"    : df_daily.index[k].strftime("%Y-%m-%d"),
        "Days_Since_Signal": days_since_signal,
        "Flags"          : " | ".join(reasons),
        "_df_daily"      : df_daily,
        "_jma"           : jma,
        "_ema"           : ema,
        "_sma"           : sma,
        "_idx"           : k,
        "_cross_idx"     : x,
        "_cross_t"       : s["cross_t"],
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
             "Risk_%","Cross_Date","Confirm_Date"]
_CW = {"Ticker":8,"Price":10,"Score":7,"Entry_Price":12,"Stop_Loss":11,
       "Risk_%":8,"Cross_Date":12,"Confirm_Date":13}
_CF = {"Price":"${:.2f}","Score":"{:.0f}","Entry_Price":"${:.2f}",
       "Stop_Loss":"${:.2f}","Risk_%":"{:.1f}%"}
_hdr_done = False

def _live_header():
    global _hdr_done
    if _hdr_done: return
    print("\n" + "━"*95)
    print("  📊  LIVE MATCHES  —  each stock printed the moment it passes all 3 steps")
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
print(f"  Signal: JMA({CFG['jma_period']}) crossed above EMA{CFG['ema_period']} from below, and the "
      f"next day closed above JMA (stop = the JMA/EMA{CFG['ema_period']} cross price)\n")

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

for sym in tqdm(list(daily_map.keys()), desc="Checking JMA/EMA8 cross", unit="stk"):
    try:
        r = analyze_jma_cross(sym, daily_map[sym])
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
print(f"  Ticker-days checked                              : {fc['days_checked']}")
print(f"  Step 1 — JMA crossed above EMA{CFG['ema_period']} the day before    : {fc['passed_step1_jma_crossed_above_ema8']}")
print(f"  Step 2 — + next day closed above JMA            : {fc['passed_step2_next_close_above_jma']}")
print(f"  Step 3 — + close above the cross price (stop)   : {fc['passed_step3_close_above_cross_price']}  (= full pattern)")
print(f"{'━'*65}")

if not results:
    print("\n  No matches. Try relaxing (see the FUNNEL above to see which")
    print("  step is actually the bottleneck before guessing):")
    print("   signal_lookback_days            1 → 3     (confirmations in the last 3 days)")
    print("   min_price                      10 → 5")
    print("   min_avg_volume             100000 → 50000")

results.sort(key=lambda x: x["Score"], reverse=True)

# ── Always build df_out and save/email (even if 0 results) ────
ts      = datetime.today().strftime("%Y%m%d_%H%M")
out_dir = os.environ.get("GITHUB_WORKSPACE", os.getcwd())

COLS = [
    "Ticker","Price","Score",
    "Entry_Price","Stop_Loss","Risk_%",
    "Cross_Price","Cross_Date","Confirm_Date",
    "JMA","EMA8","Close_vs_JMA_%","JMA_EMA8_Gap_%",
    "Vol_vs_20d_Avg","SMA50","Above_SMA50",
    "Signal_Date","Days_Since_Signal","Flags",
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
    "Cross_Price"   : lambda v: f"${v:.2f}",
    "JMA"           : lambda v: f"${v:.2f}",
    "EMA8"          : lambda v: f"${v:.2f}",
    "SMA50"         : lambda v: f"${v:.2f}",
    "Risk_%"        : lambda v: f"{v:.1f}%",
    "Close_vs_JMA_%": lambda v: f"{v:+.1f}%",
    "JMA_EMA8_Gap_%": lambda v: f"{v:+.2f}%",
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
             "Cross_Date","Confirm_Date","Close_vs_JMA_%","Vol_vs_20d_Avg"]

if _IN_NOTEBOOK and results:
    th = "".join(
        f'<th style="background:#0f172a;color:#e2e8f0;padding:9px 12px;'
        f'font-size:11px;font-weight:700;text-align:center;'
        f'border-bottom:2px solid #22c55e;white-space:nowrap">{c}</th>'
        for c in DISP_COLS)
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
    📈 Daily JMA Cross Above EMA8
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
    tit = f"  Daily JMA Cross Above EMA8   {datetime.today().strftime('%Y-%m-%d')}   {len(df_out)} matches"
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
  Score           0-100 (JMA/EMA8 gap + close vs JMA + tight stop
                  + volume + above SMA50)
  Cross_Date      day JMA crossed above EMA{CFG['ema_period']}
  Confirm_Date    next day, closed above JMA (= signal)
  Stop_Loss       the price where JMA and EMA{CFG['ema_period']} intersect (Cross_Price)
  Entry_Price     confirm-day close
  ──────────────────────────────────────────────────────""")

# Save
fpath = os.path.join(out_dir, f"daily_jma_cross_ema8_{ts}.csv")
# ── Position sizing: Quantity / Total_Amount for a fixed max loss (see position_sizing.py)
try:
    from position_sizing import add_position_size
    df_out = add_position_size(df_out)
except Exception as _ps_err:
    print(f"  ⚠️  Position sizing skipped: {_ps_err}")
df_out.to_csv(fpath, index=False)
print(f"\n  💾 CSV → {fpath}")
tv = os.path.join(out_dir, f"tv_daily_jma_cross_ema8_{ts}.txt")
with open(tv,"w") as f:
    f.write(f"###Daily JMA Cross Above EMA8 {datetime.today().strftime('%Y-%m-%d')}\n")
    for r in results: f.write(f"NASDAQ:{r['Ticker']}\n")
print(f"  📋 TradingView → {tv}")
if results:
    print(f"\n  📋 Tickers (comma-separated):")
    print(f"  {', '.join(r['Ticker'] for r in results)}")


# Quantity / Total_Amount per ticker for the email (from the sized df_out)
_SIZE = {}
if {"Ticker", "Quantity", "Total_Amount"} <= set(df_out.columns):
    _SIZE = {t: (q, a) for t, q, a in zip(df_out["Ticker"], df_out["Quantity"], df_out["Total_Amount"])}

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

        heads = ["Ticker","Price","Score","Entry","Stop (cross)","Risk %","Cross",
                 "Confirm","Close vs JMA","Vol/20d","Trend","Qty","Amount"]
        th_e = "".join(
            f'<th style="background:#1e293b;color:#e2e8f0;padding:8px 11px;'
            f'font-size:11px;font-weight:700;border-bottom:2px solid #3b82f6;'
            f'white-space:nowrap">{c}</th>' for c in heads)

        def td(v, extra=""):
            return f'<td style="padding:6px 11px;font-size:12px;{extra}">{v}</td>'

        rows_e = ""
        for i, r in enumerate(rl[:50]):
            bg = "#fff" if i % 2 == 0 else "#f0f9ff"
            q, a = _SIZE.get(r.get("Ticker"), ("—", None))
            rows_e += (
                f'<tr style="background:{bg}">'
                + td(r.get("Ticker","—"), "font-weight:700")
                + td(f'${float(r.get("Price",0) or 0):.2f}')
                + td(f'{float(r.get("Score",0) or 0):.0f}',
                     "font-weight:700;background:#166534;color:#fff;text-align:center")
                + td(f'${float(r.get("Entry_Price",0) or 0):.2f}', "color:#22c55e")
                + td(f'${float(r.get("Stop_Loss",0) or 0):.2f}', "color:#ef4444")
                + td(f'{float(r.get("Risk_%",0) or 0):.1f}%')
                + td(r.get("Cross_Date","—"), "color:#7c3aed")
                + td(r.get("Confirm_Date","—"), "color:#a78bfa;font-weight:600")
                + td(f'{float(r.get("Close_vs_JMA_%",0) or 0):+.1f}%')
                + td(f'x{float(r.get("Vol_vs_20d_Avg",0) or 0):.2f}')
                + td("Above SMA50" if r.get("Above_SMA50") else "Below SMA50",
                     "color:#16a34a" if r.get("Above_SMA50") else "color:#b45309")
                + td(f'{q}')
                + td(f'${float(a):,.0f}' if a is not None else "—")
                + '</tr>')
        no_results_msg = ('<tr><td colspan="13" style="padding:20px;text-align:center;'
                          'color:#94a3b8;font-size:13px">No matches today</td></tr>')
        more_note = (f'Showing top 50 of {cnt} — full list in the CSV attachment'
                     if cnt > 50 else '📎 Full results attached as CSV')

        fc = FUNNEL_COUNTS
        html_e = f"""<!DOCTYPE html><html><body style="margin:0;padding:0;
background:#f1f5f9;font-family:'Segoe UI',Arial,sans-serif">
<table width="100%" cellpadding="0" cellspacing="0"><tr><td align="center" style="padding:20px 10px">
<table width="100%" style="max-width:900px;background:#fff;border-radius:12px;
       overflow:hidden;box-shadow:0 4px 16px rgba(0,0,0,0.08)">
  <tr><td style="background:linear-gradient(135deg,#0f172a,#1e3a5f);padding:22px 28px">
<h1 style="margin:0;color:#60a5fa;font-size:20px;font-weight:700">
  📊 Daily JMA Cross Above EMA8
</h1>
<p style="margin:6px 0 0;color:#94a3b8;font-size:12px">
  {datetime.today().strftime('%Y-%m-%d %H:%M UTC')} &nbsp;·&nbsp;
  {cnt} match{'es' if cnt!=1 else ''} — JMA({CFG['jma_period']}) crossed above EMA{CFG['ema_period']} from below and the
  next day closed above JMA; stop = the price where JMA and EMA{CFG['ema_period']} intersect
</p>
  </td></tr>
  <tr><td style="padding:14px 28px 4px;background:#0b1220">
<div style="background:#111827;border:1px solid #1f2937;border-radius:8px;padding:12px 16px">
  <p style="margin:0 0 6px;color:#93c5fd;font-size:12px;font-weight:700">
    🔍 FUNNEL — last {CFG['signal_lookback_days']} trading day(s), every liquid ticker
  </p>
  <p style="margin:0;color:#cbd5e1;font-size:12px">
    {fc['days_checked']} ticker-days &nbsp;→&nbsp;
    {fc['passed_step1_jma_crossed_above_ema8']} JMA crossed above EMA{CFG['ema_period']} &nbsp;→&nbsp;
    {fc['passed_step2_next_close_above_jma']} next close above JMA &nbsp;→&nbsp;
    <b style="color:#facc15">{fc['passed_step3_close_above_cross_price']} close above cross price = full match</b>
  </p>
</div>
  </td></tr>
  <tr><td style="padding:16px">
<div style="overflow-x:auto;border-radius:8px;border:1px solid #e2e8f0">
  <table style="border-collapse:collapse;width:100%;min-width:760px">
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
  ⚠️ Not financial advice &nbsp;·&nbsp; Entry = confirm-day close (default), Stop = JMA/EMA{CFG['ema_period']} cross price &nbsp;·&nbsp; Qty sized for a $40 max loss &nbsp;·&nbsp; Auto-generated by GitHub Actions
</p>
  </td></tr>
</table>
</td></tr></table>
</body></html>"""

        plain_lines = [
            f"Daily JMA Cross Above EMA{CFG['ema_period']} — {datetime.today().strftime('%Y-%m-%d')}",
            f"{cnt} matches (JMA crossed above EMA{CFG['ema_period']}; next day closed above JMA; stop = cross price)",
            "="*60,
            f"FUNNEL: {fc['days_checked']} ticker-days -> "
            f"{fc['passed_step1_jma_crossed_above_ema8']} JMA crossed above EMA{CFG['ema_period']} -> "
            f"{fc['passed_step2_next_close_above_jma']} next close above JMA -> "
            f"{fc['passed_step3_close_above_cross_price']} close above cross price (=full match)",
            "="*60,
        ]
        if rl:
            for r in rl[:50]:
                q, a = _SIZE.get(r.get("Ticker"), ("—", None))
                plain_lines.append(
                    f"{r.get('Ticker','—'):<7} ${float(r.get('Price',0) or 0):.2f}  "
                    f"Score:{float(r.get('Score',0) or 0):.0f}  "
                    f"Entry:${float(r.get('Entry_Price',0) or 0):.2f}  "
                    f"SL:${float(r.get('Stop_Loss',0) or 0):.2f}  "
                    f"Risk:{float(r.get('Risk_%',0) or 0):.1f}%  "
                    f"Cross:{r.get('Cross_Date','—')}  Confirm:{r.get('Confirm_Date','—')}  "
                    f"vsJMA:{float(r.get('Close_vs_JMA_%',0) or 0):+.1f}%  "
                    f"Qty:{q}  Amount:{'$%s' % format(float(a), ',.0f') if a is not None else '—'}")
            plain_lines.append("")
            plain_lines.append("Tickers (comma-separated):")
            plain_lines.append(", ".join(r.get("Ticker","") for r in rl))
        else:
            plain_lines.append("No matches today")
        plain_lines.append("\nFull results in CSV attachment.")
        plain_e = "\n".join(plain_lines)

        subj = (f"📊 JMA Cross Above EMA8 — {cnt} signal{'s' if cnt!=1 else ''}"
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

# ── Charts for top 5: price with JMA and EMA8, the cross point (= stop)
#    and the confirm day marked ─────────────────────────────────────
if results:
    top = results[:min(5,len(results))]
    fig, axes = plt.subplots(len(top), 1, figsize=(13, 3.6*len(top)), facecolor="#0f172a",
                              squeeze=False)
    axes = axes[:,0]
    for p, r in enumerate(top):
        ax = axes[p]
        df_all = r["_df_daily"]
        df_p = df_all.tail(70)
        idx  = df_p.index
        k, x, t = r["_idx"], r["_cross_idx"], r["_cross_t"]

        ax.set_facecolor("#0f172a")
        ax.plot(idx, df_p["Close"], color="#60a5fa", lw=1.4, label="Close", zorder=5)
        ax.plot(idx, r["_jma"].reindex(idx), color="#f472b6", lw=1.8, label="JMA 13")
        ax.plot(idx, r["_ema"].reindex(idx), color="#22c55e", lw=1.2, label="EMA8")
        ax.plot(idx, r["_sma"].reindex(idx), color="#fbbf24", lw=0.9, ls=":", label="SMA50")
        ax.axhline(r["Stop_Loss"], color="#ef4444", lw=1.0, ls="--", alpha=0.85,
                   label=f"Stop {r['Stop_Loss']:.2f} (cross)")
        d0, d1 = df_all.index[x-1], df_all.index[x]
        cross_d = d0 + (d1 - d0) * t
        if d0 in idx:
            ax.scatter([cross_d], [r["Cross_Price"]], color="#ef4444", marker="x", s=60,
                       zorder=8, label="Cross point")
        sig_d = df_all.index[k]
        if sig_d in idx:
            ax.scatter([sig_d], [df_all["Close"].iloc[k]], color="#facc15", s=50, zorder=7,
                       label="Confirm close")
        ax.set_title(f"{r['Ticker']}  |  {r['Price']:.2f} USD  |  Score {r['Score']}  |  "
                     f"cross {r['Cross_Date']}  |  confirm {r['Confirm_Date']}  |  "
                     f"stop {r['Stop_Loss']:.2f} ({r['Risk_%']:.1f}%)",
                     color="#e2e8f0", fontsize=9, fontweight="bold", pad=7)
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))
        ax.tick_params(colors="#94a3b8", labelsize=8)
        for sp in ax.spines.values(): sp.set_edgecolor("#1e3a5f")
        ax.legend(loc="upper left", facecolor="#1e293b", labelcolor="#e2e8f0", fontsize=6, ncol=7)
        ax.grid(color="#1e3a5f", ls="--", lw=0.5, alpha=0.6)
    plt.suptitle(f"Daily JMA Cross Above EMA8  ·  {datetime.today().strftime('%Y-%m-%d')}",
                 color="#60a5fa", fontsize=12, fontweight="bold", y=1.001)
    plt.tight_layout()
    cp = os.path.join(out_dir, f"daily_jma_cross_ema8_chart_{ts}.png")
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
  📋 SIGNAL (all required; confirm day = latest daily candle)
  JMA = JMA({CFG['jma_period']}, phase {CFG['jma_phase']}, power {CFG['jma_power']}); EMA{CFG['ema_period']} on daily closes.

  1) Cross day: JMA was at or below EMA{CFG['ema_period']} the day before and is
     above it on the cross day.
  2) Confirm day (next day): the close is above JMA.
  3) The confirm-day close is above the cross price (valid stop).

  CROSS PRICE = where JMA and EMA{CFG['ema_period']} intersect, interpolated between the
  day before the cross and the cross day. It is the STOP LOSS.

  📋 OUTPUT
  Stop_Loss   = cross price (requested)
  Entry_Price = confirm-day CLOSE (reasonable default)
  Quantity / Total_Amount = sized for a $40 max loss (position_sizing.py)
  Universe    = price >= 10 USD, 20d avg volume >= 100k

  📋 SCORE (0-100)
  Base 30 + JMA above EMA8 gap (0-15) + close 0-3% over JMA (15; 8 if
  further; -5 if > 6%) + tight stop (0-15) + volume vs 20d avg (0-15)
  + above SMA50 (10); -10 if stop risk > 12%

  ⚙️  TUNE IF 0 RESULTS
  signal_lookback_days            1 → 3
  min_price                      10 → 5
  min_avg_volume             100000 → 50000
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
""")
