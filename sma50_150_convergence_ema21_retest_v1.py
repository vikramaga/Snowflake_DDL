# ============================================================
# NASDAQ — SMA50/SMA150 Convergence + EMA21 Retest Scanner (v1)
# ============================================================
#
# SIGNAL (all required, evaluated on the retest day):
#
#   1. CONVERGENCE: SMA50 is still BELOW SMA150 (hasn't crossed
#      yet) but the gap between them is small (within
#      sma_convergence_pct% of SMA150) AND SMA50 is rising toward
#      it (SMA50 today > SMA50 sma50_slope_lookback bars ago) —
#      i.e. SMA50 is approaching SMA150 from below, not just
#      incidentally close to it.
#
#   2. EMA21 ALREADY ABOVE SMA150: the shorter-term EMA21 has
#      already crossed above SMA150, even though the medium-term
#      SMA50 hasn't caught up yet.
#
#   3. RETEST: price pulls back so its LOW comes within
#      retest_tolerance_pct% of EMA21 (a genuine test of that
#      level), AND closes back AT/ABOVE EMA21 (the test held,
#      not a breakdown).
#
#   4. VOLUME: that day's volume is higher than the previous day's.
#
# Scanned over the last signal_lookback_days trading days (not just
# today), consistent with the other scanners in this repo.
#
# DATA — only 1 download per ticker: daily bars.
#
# OUTPUT: Entry_Price and Stop_Loss are included as a reasonable
# default (Entry = the retest day's close; Stop = the retest day's
# low, just under the tested EMA21 level) since every other scanner
# in this repo reports actionable levels, but no specific entry/
# stop/target logic was requested for this pattern — adjust or
# ignore these two fields if that default doesn't fit. No historical
# backtest is included yet either (can be added the same way as
# three_candle_hammer_doji_v1.py's, if wanted).
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
    "history_days"          : 400,   # daily bars — plenty for SMA150 + buffer

    # ── Indicator periods ────────────────────────────────────────
    "ema21_period"           : 21,
    "sma50_period"           : 50,
    "sma150_period"          : 150,

    # ── Step 1: SMA50/SMA150 convergence (SMA50 still below) ───────
    "sma_convergence_pct"    : 15.0, # max gap (as % of SMA150) — SMA50
                                      # must be within this % of SMA150
    "sma50_slope_lookback"   : 5,    # bars back to confirm SMA50 is
                                      # rising toward SMA150, not just
                                      # incidentally close to it

    # ── Step 3: EMA21 retest ────────────────────────────────────────
    "retest_tolerance_pct"   : 1.5,  # how close the day's LOW must come
                                      # to EMA21 to count as a genuine retest
    "require_close_above_ema21": True,  # the retest must hold (close >= EMA21)

    "signal_lookback_days"   : 15,   # ~3 trading weeks — how far back to look
                                      # for the pattern, not just today

    # ── Filters ─────────────────────────────────────────────────
    "min_avg_volume"         : 80_000,
    "min_price"              : 2.0,

    "batch_size"             : 50,
    "batch_sleep"            : 1.5,
}

def check_convergence_retest_at(df, ema21, sma50, sma150, i, cfg):
    """
    Checks the full pattern with the retest day anchored at bar `i`.
    This is the SINGLE SOURCE OF TRUTH for the pattern logic.

    Unlike a simple short-circuit check, this computes EVERY stage's
    pass/fail regardless of earlier failures (guarding only on
    missing/insufficient data), so callers can diagnose exactly
    which stage is filtering results out — see FUNNEL_COUNTS below.

    Returns (passed: bool, details: dict). details is {} only when
    there isn't enough data to evaluate at all; otherwise it always
    contains every stage's boolean, even on failure.
    """
    if i < 1 or i >= len(df):
        return False, {}

    e21  = ema21.iloc[i]
    s50  = sma50.iloc[i]
    s150 = sma150.iloc[i]
    if any(np.isnan(v) for v in [e21, s50, s150]):
        return False, {}

    close_i, low_i = float(df["Close"].iloc[i]), float(df["Low"].iloc[i])
    vol_i, vol_prev = float(df["Volume"].iloc[i]), float(df["Volume"].iloc[i-1])

    # ── Step 1: SMA50 still below SMA150, gap small, SMA50 rising ──
    still_below = s50 < s150
    gap_pct = (s150 - s50) / s150 * 100 if s150 > 0 else 999
    gap_ok  = gap_pct <= cfg["sma_convergence_pct"]

    sb = cfg["sma50_slope_lookback"]
    slope_ok = False
    if i - sb >= 0 and not np.isnan(sma50.iloc[i-sb]):
        slope_ok = s50 > float(sma50.iloc[i-sb])

    step1_ok = still_below and gap_ok and slope_ok

    # ── Step 2: EMA21 already above SMA150 ──────────────────────────
    step2_ok = e21 > s150

    # ── Step 3: price retesting EMA21 (low near it, close holding) ──
    near_pct = abs(low_i - e21) / e21 * 100 if e21 > 0 else 999
    retest_ok = near_pct <= cfg["retest_tolerance_pct"]
    holds_ok  = (close_i >= e21) if cfg["require_close_above_ema21"] else True
    step3_ok = retest_ok and holds_ok

    # ── Step 4: volume higher than the previous day ─────────────────
    step4_ok = vol_i > vol_prev

    passed = step1_ok and step2_ok and step3_ok and step4_ok

    details = {
        "idx": i, "gap_pct": gap_pct, "near_pct": near_pct,
        "ema21": float(e21), "sma50": float(s50), "sma150": float(s150),
        "close": close_i, "low": low_i,
        "vol": vol_i, "vol_prev": vol_prev,
        "step1_ok": step1_ok, "step2_ok": step2_ok,
        "step3_ok": step3_ok, "step4_ok": step4_ok,
        "still_below": still_below, "gap_ok": gap_ok, "slope_ok": slope_ok,
    }
    return passed, details

# ── Diagnostic funnel — tallies how far each ticker-day gets through
#    the pattern, across the FULL universe scan, so a 0-match run can
#    be diagnosed empirically instead of guessed at ─────────────────
FUNNEL_COUNTS = {
    "days_checked": 0,      # every (ticker, day) evaluated at all
    "passed_step1_convergence": 0,   # SMA50 still below SMA150, gap small, rising
    "passed_step2_ema21_above": 0,   # + EMA21 already above SMA150
    "passed_step3_retest": 0,        # + price retesting EMA21 and holding
    "passed_step4_volume": 0,        # + volume higher than prior day (= full match)
}

def find_convergence_retest_signals(df, ema21, sma50, sma150, cfg):
    """
    Scans the last `signal_lookback_days` trading days for the full
    pattern. Returns a list of hit dicts, most recent first. Also
    tallies FUNNEL_COUNTS for every day checked, regardless of match.
    """
    global FUNNEL_COUNTS
    n = len(df)
    lb = cfg["signal_lookback_days"]
    hits = []
    for back in range(0, lb):
        i = (n - 1) - back
        passed, details = check_convergence_retest_at(df, ema21, sma50, sma150, i, cfg)
        if not details:
            continue   # insufficient data that day — not counted in the funnel
        FUNNEL_COUNTS["days_checked"] += 1
        if details["step1_ok"]:
            FUNNEL_COUNTS["passed_step1_convergence"] += 1
            if details["step2_ok"]:
                FUNNEL_COUNTS["passed_step2_ema21_above"] += 1
                if details["step3_ok"]:
                    FUNNEL_COUNTS["passed_step3_retest"] += 1
                    if details["step4_ok"]:
                        FUNNEL_COUNTS["passed_step4_volume"] += 1
        if passed:
            hits.append(details)
    return hits

# ── Technical signal: SMA50/SMA150 convergence + EMA21 retest ────
def analyze_convergence_retest(sym, df_daily):
    """
    Returns dict with score and setup details, or None if no
    required condition is met anywhere in the lookback window.
    """
    if df_daily is None:
        return None

    price   = float(df_daily["Close"].iloc[-1])
    avg_vol = float(df_daily["Volume"].tail(20).mean())
    if price   < CFG["min_price"]:      return None
    if avg_vol < CFG["min_avg_volume"]: return None

    n = len(df_daily)
    if n < CFG["sma150_period"] + CFG["signal_lookback_days"] + 20:
        return None

    ema21  = df_daily["Close"].ewm(span=CFG["ema21_period"], adjust=False).mean()
    sma50  = df_daily["Close"].rolling(CFG["sma50_period"]).mean()
    sma150 = df_daily["Close"].rolling(CFG["sma150_period"]).mean()

    hits = find_convergence_retest_signals(df_daily, ema21, sma50, sma150, CFG)
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
    entry_price = sig["close"]      # the retest day's close
    stop_loss   = sig["low"]        # the retest day's low (just under EMA21)
    if entry_price <= stop_loss:
        return None
    risk_pct = (entry_price - stop_loss) / entry_price * 100 if entry_price > 0 else 0
    vol_ratio = sig["vol"] / sig["vol_prev"] if sig["vol_prev"] > 0 else 0

    # ── Score (0-100) ────────────────────────────────────────────
    score = 0
    reasons = []
    score += max(0, min(25, 25 - sig["gap_pct"] * 8))
    reasons.append(f"Gap{sig['gap_pct']:.1f}%")
    score += max(0, min(25, 25 - sig["near_pct"] * 12))
    reasons.append(f"RetestNear{sig['near_pct']:.2f}%")
    score += min(25, (vol_ratio - 1.0) * 25)
    reasons.append(f"Vol{vol_ratio:.1f}x")
    freshness_pts = max(0, 15 - days_since_signal)
    score += freshness_pts
    reasons.append(f"{days_since_signal}dAgo")
    score += 10   # base for clearing every gate
    score = round(min(100, max(0, score)))

    return {
        "Score"          : score,
        "Price"          : round(price, 2),
        "Entry_Price"    : round(entry_price, 2),
        "Stop_Loss"      : round(stop_loss, 2),
        "Risk_%"         : round(risk_pct, 1),
        "Gap_%"          : round(sig["gap_pct"], 2),
        "Retest_Near_%"  : round(sig["near_pct"], 2),
        "EMA21"          : round(sig["ema21"], 2),
        "SMA50"          : round(sig["sma50"], 2),
        "SMA150"         : round(sig["sma150"], 2),
        "Vol_Ratio"      : round(vol_ratio, 2),
        "Signal_Date"    : df_daily.index[sig_idx].strftime("%Y-%m-%d"),
        "Days_Since_Signal"  : days_since_signal,
        "Recent_Signal_Count": len(recent_signals),
        "Recent_Signals" : " | ".join(
            f"{s['date'].strftime('%Y-%m-%d')}" for s in recent_signals),
        "Flags"          : " | ".join(reasons),
        "_df_daily"      : df_daily,
        "_ema21"         : ema21, "_sma50": sma50, "_sma150": sma150,
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
                        df = _clean(df, min_bars=300)
                        if df is not None: out[sym] = df
                    except Exception: pass
            elif len(symbols) == 1:
                df = _clean(raw, min_bars=300)
                if df is not None: out[symbols[0]] = df
    except Exception: pass
    for sym in [s for s in symbols if s not in out]:
        for _ in range(2):
            try:
                df = yf.Ticker(sym).history(
                    start=start.strftime("%Y-%m-%d"),
                    end=end.strftime("%Y-%m-%d"),
                    auto_adjust=True, actions=False)
                df = _clean(df, min_bars=300)
                if df is not None: out[sym] = df; break
            except Exception: time.sleep(0.2)
        time.sleep(0.04)
    return out

# ── Live print ────────────────────────────────────────────────
LIVE_COLS = ["Ticker","Price","Score","Entry_Price","Stop_Loss",
             "Gap_%","Retest_Near_%","Vol_Ratio","Signal_Date"]
_CW = {"Ticker":8,"Price":10,"Score":7,"Entry_Price":12,"Stop_Loss":11,
       "Gap_%":9,"Retest_Near_%":13,"Vol_Ratio":11,"Signal_Date":13}
_CF = {"Price":"${:.2f}","Score":"{:.0f}","Entry_Price":"${:.2f}",
       "Stop_Loss":"${:.2f}","Vol_Ratio":"{:.2f}x"}
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
print("  A stock only matches if all 3 candles and the structure gate")
print("  fire together within the lookback window\n")

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

for sym in tqdm(list(daily_map.keys()), desc="Checking 3-candle pattern", unit="stk"):
    try:
        r = analyze_convergence_retest(sym, daily_map[sym])
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

# ── Diagnostic funnel — where ticker-days got filtered out ──────
print(f"\n{'━'*65}")
print(f"  🔍 FUNNEL — where ticker-days were filtered out")
print(f"  (tallied across every liquid ticker's last {CFG['signal_lookback_days']} trading days)")
print(f"{'━'*65}")
fc = FUNNEL_COUNTS
print(f"  Ticker-days checked                        : {fc['days_checked']}")
print(f"  Step 1 — SMA50 converging (below, tight, rising) : {fc['passed_step1_convergence']}")
print(f"  Step 2 — + EMA21 already above SMA150      : {fc['passed_step2_ema21_above']}")
print(f"  Step 3 — + retesting EMA21 and holding     : {fc['passed_step3_retest']}")
print(f"  Step 4 — + volume higher than prior day    : {fc['passed_step4_volume']}  (= full pattern)")
print(f"{'━'*65}")

if not results:
    print("\n  No matches. Try relaxing (see the FUNNEL above to see which")
    print("  step is actually the bottleneck before guessing):")
    print("   sma_convergence_pct         2.0 → 3.5   (allow a wider convergence gap)")
    print("   retest_tolerance_pct        1.5 → 2.5   (allow a looser retest)")
    print("   sma50_slope_lookback          5 → 10")
    print("   signal_lookback_days         15 → 25    (search further back)")
    print("   min_price                      2 → 1")
    print("   min_avg_volume             80000 → 50000")

results.sort(key=lambda x: x["Score"], reverse=True)

# ── Always build df_out and save/email (even if 0 results) ────
ts      = datetime.today().strftime("%Y%m%d_%H%M")
out_dir = os.environ.get("GITHUB_WORKSPACE", os.getcwd())

COLS = [
    "Ticker","Price","Score",
    "Entry_Price","Stop_Loss","Risk_%",
    "Gap_%","Retest_Near_%","EMA21","SMA50","SMA150","Vol_Ratio",
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
    "Gap_%"       : lambda v: f"{v:.2f}%",
    "Retest_Near_%": lambda v: f"{v:.2f}%",
    "EMA21"       : lambda v: f"${v:.2f}",
    "SMA50"       : lambda v: f"${v:.2f}",
    "SMA150"      : lambda v: f"${v:.2f}",
    "Vol_Ratio"   : lambda v: f"{v:.2f}x",
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
            "Gap_%","Retest_Near_%","Vol_Ratio","Signal_Date"]
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
                    clr = "#22c55e" if v >= 1.3 else "#f59e0b"
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
    <span style="color:#f1f5f9;font-size:15px;font-weight:700">SMA50/SMA150 Convergence + EMA21 Retest</span>
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
    📈 SMA50/SMA150 Convergence + EMA21 Retest
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
  SMA50 still below SMA150 but converging (small, narrowing gap) &nbsp;·&nbsp;
  EMA21 already above SMA150 &nbsp;·&nbsp;
  Price's low retests EMA21 and closes back at/above it &nbsp;·&nbsp;
  Volume higher than the previous day &nbsp;·&nbsp;
  Entry_Price = the retest day's close &nbsp;·&nbsp;
  Stop_Loss = the retest day's low — reasonable defaults, not explicitly
  requested &nbsp;·&nbsp;
  Signal_Date is the exact date the pattern fired (checked over the
  last {CFG['signal_lookback_days']} trading days, not just today)
</div>"""

    display_html(header_html + table_html + legend_html)

elif results:
    # ASCII table (CLI/GitHub Actions mode)
    CLI_COLS = ["Ticker","Price","Score","Entry_Price","Stop_Loss",
                "Gap_%","Retest_Near_%","Vol_Ratio","Signal_Date"]
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
    tit = f"  SMA50/SMA150 Convergence + EMA21 Retest   {datetime.today().strftime('%Y-%m-%d')}   {len(df_out)} matches"
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
  Score           0-100 (convergence tightness + retest closeness +
                  volume increase + freshness)
  Entry_Price     the retest day's close (a reasonable default,
                  not explicitly requested)
  Stop_Loss       the retest day's low, just under EMA21
  Gap_%           how close SMA50 is to SMA150 (SMA50 still below)
  Retest_Near_%   how close the day's low came to EMA21
  Vol_Ratio       that day's volume / previous day's volume
  Signal_Date     exact calendar date the pattern fired
  Days_Since_Signal  how many trading days ago (0 = today; checked
                     over the last signal_lookback_days trading days)
  ──────────────────────────────────────────────────────""")

# Save
fpath = os.path.join(out_dir, f"sma50_150_convergence_ema21_retest_{ts}.csv")
df_out.to_csv(fpath, index=False)
print(f"\n  💾 CSV → {fpath}")
tv = os.path.join(out_dir, f"tv_sma50_150_convergence_ema21_retest_{ts}.txt")
with open(tv,"w") as f:
    f.write(f"###SMA50/SMA150 Convergence + EMA21 Retest {datetime.today().strftime('%Y-%m-%d')}\n")
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
                      "Gap_%","Retest_Near_%","Vol_Ratio","Signal_Date"]
        )
        rows_e = ""
        for i, r in enumerate(rl[:50]):
            bg  = "#fff" if i % 2 == 0 else "#f0f9ff"
            ticker = r.get("Ticker","—")
            price  = r.get("Price",0) or 0
            score  = r.get("Score",0) or 0
            entry  = r.get("Entry_Price",0) or 0
            stop   = r.get("Stop_Loss",0) or 0
            gap    = r.get("Gap_%",0) or 0
            near   = r.get("Retest_Near_%",0) or 0
            vr     = r.get("Vol_Ratio",0) or 0
            sdate  = r.get("Signal_Date","—")
            rows_e += (
                f'<tr style="background:{bg}">'
                f'<td style="padding:6px 11px;font-size:12px;font-weight:700">{ticker}</td>'
                f'<td style="padding:6px 11px;font-size:12px">${float(price):.2f}</td>'
                f'<td style="padding:6px 11px;font-size:12px;font-weight:700;'
                f'background:#166534;color:#fff;text-align:center">{float(score):.0f}</td>'
                f'<td style="padding:6px 11px;font-size:12px;color:#22c55e">${float(entry):.2f}</td>'
                f'<td style="padding:6px 11px;font-size:12px;color:#ef4444">${float(stop):.2f}</td>'
                f'<td style="padding:6px 11px;font-size:12px;text-align:center">{float(gap):.2f}%</td>'
                f'<td style="padding:6px 11px;font-size:12px;text-align:center">{float(near):.2f}%</td>'
                f'<td style="padding:6px 11px;font-size:12px;font-weight:600">{float(vr):.2f}x</td>'
                f'<td style="padding:6px 11px;font-size:12px;text-align:center;'
                f'color:#a78bfa;font-weight:600">{sdate}</td>'
                f'</tr>'
            )
        no_results_msg = ('<tr><td colspan="9" style="padding:20px;text-align:center;'
                           'color:#94a3b8;font-size:13px">No matches today</td></tr>')

        html_e = f"""<!DOCTYPE html><html><body style="margin:0;padding:0;
background:#f1f5f9;font-family:'Segoe UI',Arial,sans-serif">
<table width="100%" cellpadding="0" cellspacing="0"><tr><td align="center" style="padding:20px 10px">
<table width="100%" style="max-width:800px;background:#fff;border-radius:12px;
       overflow:hidden;box-shadow:0 4px 16px rgba(0,0,0,0.08)">
  <tr><td style="background:linear-gradient(135deg,#0f172a,#1e3a5f);padding:22px 28px">
<h1 style="margin:0;color:#60a5fa;font-size:20px;font-weight:700">
  📊 SMA50/SMA150 Convergence + EMA21 Retest
</h1>
<p style="margin:6px 0 0;color:#94a3b8;font-size:12px">
  {datetime.today().strftime('%Y-%m-%d %H:%M UTC')} &nbsp;·&nbsp;
  {cnt} match{'es' if cnt!=1 else ''} found — SMA50 converging toward
  SMA150 from below, EMA21 already above SMA150, price retesting EMA21
  on rising volume
</p>
  </td></tr>
  <tr><td style="padding:14px 28px 4px;background:#0b1220">
<div style="background:#111827;border:1px solid #1f2937;border-radius:8px;padding:12px 16px">
  <p style="margin:0 0 6px;color:#93c5fd;font-size:12px;font-weight:700">
    🔍 FUNNEL — where ticker-days were filtered out (last {CFG['signal_lookback_days']} trading days, every liquid ticker)
  </p>
  <p style="margin:0;color:#cbd5e1;font-size:12px">
    {FUNNEL_COUNTS['days_checked']} ticker-days checked &nbsp;→&nbsp;
    {FUNNEL_COUNTS['passed_step1_convergence']} passed Step 1 (convergence) &nbsp;→&nbsp;
    {FUNNEL_COUNTS['passed_step2_ema21_above']} passed Step 2 (EMA21&gt;SMA150) &nbsp;→&nbsp;
    {FUNNEL_COUNTS['passed_step3_retest']} passed Step 3 (retest holds) &nbsp;→&nbsp;
    <b style="color:#facc15">{FUNNEL_COUNTS['passed_step4_volume']} passed Step 4 (volume) = full match</b>
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
            f"SMA50/SMA150 Convergence + EMA21 Retest (checked over last {CFG['signal_lookback_days']} trading days) — {datetime.today().strftime('%Y-%m-%d')}",
            f"{cnt} matches (SMA50 converging toward SMA150 from below + EMA21 above SMA150 + price retesting EMA21 on rising volume)",
            "="*60,
            f"FUNNEL: {FUNNEL_COUNTS['days_checked']} ticker-days -> "
            f"{FUNNEL_COUNTS['passed_step1_convergence']} passed convergence -> "
            f"{FUNNEL_COUNTS['passed_step2_ema21_above']} passed EMA21>SMA150 -> "
            f"{FUNNEL_COUNTS['passed_step3_retest']} passed retest -> "
            f"{FUNNEL_COUNTS['passed_step4_volume']} passed volume (=full match)",
            "="*60,
        ]
        if rl:
            for r in rl[:50]:
                ticker = r.get("Ticker","—")
                price  = r.get("Price",0) or 0
                score  = r.get("Score",0) or 0
                entry  = r.get("Entry_Price",0) or 0
                stop   = r.get("Stop_Loss",0) or 0
                gap    = r.get("Gap_%",0) or 0
                near   = r.get("Retest_Near_%",0) or 0
                vr     = r.get("Vol_Ratio",0) or 0
                sdate  = r.get("Signal_Date","—")
                plain_lines.append(
                    f"{ticker:<7} ${float(price):.2f}  Score:{float(score):.0f}  "
                    f"Entry:${float(entry):.2f}  SL:${float(stop):.2f}  "
                    f"Gap:{float(gap):.2f}%  RetestNear:{float(near):.2f}%  "
                    f"Vol:{float(vr):.2f}x  Signal:{sdate}"
                )
            plain_lines.append("")
            plain_lines.append("Tickers (comma-separated):")
            plain_lines.append(", ".join(r.get("Ticker","") for r in rl))
        else:
            plain_lines.append("No matches today")
        plain_lines.append("\nFull results in CSV attachment.")
        plain_e = "\n".join(plain_lines)

        subj = (f"📊 SMA50/SMA150 Convergence + EMA21 Retest — {cnt} signal{'s' if cnt!=1 else ''}"
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

# ── Charts for top 5 (price + EMA21/SMA50/SMA150 + retest marker) ──
if results:
    top = results[:min(5,len(results))]
    fig, axes = plt.subplots(len(top),1,figsize=(15,4.2*len(top)),facecolor="#0f172a")
    if len(top)==1: axes=[axes]
    for ax, r in zip(axes, top):
        df_p = r["_df_daily"].tail(90).copy()
        ema21_p  = r["_ema21"].reindex(df_p.index)
        sma50_p  = r["_sma50"].reindex(df_p.index)
        sma150_p = r["_sma150"].reindex(df_p.index)
        ax.set_facecolor("#0f172a")
        ax.plot(df_p.index, df_p["Close"], color="#60a5fa", lw=1.4, label="Close", zorder=4)
        ax.plot(df_p.index, ema21_p,  color="#34d399", lw=1.2, ls="--", label="EMA21", zorder=3)
        ax.plot(df_p.index, sma50_p,  color="#fbbf24", lw=1.3, ls="-.", label="SMA50",  zorder=3)
        ax.plot(df_p.index, sma150_p, color="#f87171", lw=1.2, ls=":",  label="SMA150", zorder=3)

        # mark the retest day
        pat_date = pd.to_datetime(r["Signal_Date"])
        if pat_date in df_p.index:
            ax.scatter([pat_date], [r["Entry_Price"]], color="#22c55e", s=60, zorder=6,
                       marker="^", label="Retest held")
        ax.axhline(r["Stop_Loss"], color="#ef4444", lw=1.0, ls="--", alpha=0.85,
                  label=f"Stop ${r['Stop_Loss']:.2f}")
        ax.axhline(r["Entry_Price"], color="#22c55e", lw=1.0, ls="--", alpha=0.85,
                  label=f"Entry ${r['Entry_Price']:.2f}")
        ax.set_title(
            f"{r['Ticker']}  |  ${r['Price']:.2f}  |  Score {r['Score']}  |  "
            f"Gap {r['Gap_%']:.2f}%  |  Retest {r['Retest_Near_%']:.2f}%  |  "
            f"Vol {r['Vol_Ratio']:.2f}x  |  {r['Signal_Date']}",
            color="#e2e8f0", fontsize=9, fontweight="bold", pad=7)
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))
        ax.tick_params(colors="#94a3b8", labelsize=8)
        for sp in ax.spines.values(): sp.set_edgecolor("#1e3a5f")
        ax.legend(loc="upper left", facecolor="#1e293b", labelcolor="#e2e8f0",
                  fontsize=7, framealpha=0.9, ncol=2)
        ax.grid(color="#1e3a5f", ls="--", lw=0.5, alpha=0.6)
    plt.suptitle(
        f"SMA50/SMA150 Convergence + EMA21 Retest  ·  {datetime.today().strftime('%Y-%m-%d')}",
        color="#60a5fa", fontsize=12, fontweight="bold", y=1.001)
    plt.tight_layout()
    cp = os.path.join(out_dir, f"sma50_150_convergence_ema21_retest_chart_{ts}.png")
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
  📋 SIGNAL (all required — scanned over the last
  signal_lookback_days trading days, {CFG['signal_lookback_days']}d ≈ 3 weeks,
  not just today)

  1) CONVERGENCE: SMA50 is still BELOW SMA150 (hasn't crossed yet)
     but the gap is small (within sma_convergence_pct% of SMA150)
     AND SMA50 is rising toward it (not just incidentally close).
  2) EMA21 ALREADY ABOVE SMA150: the shorter-term EMA21 has already
     crossed above SMA150, even though SMA50 hasn't caught up yet.
  3) RETEST: that day's LOW comes within retest_tolerance_pct% of
     EMA21 (a genuine test of that level), AND the close holds
     at/above EMA21 (the test held, not a breakdown).
  4) VOLUME: that day's volume is higher than the previous day's.

  If the pattern fired more than once in the window, the MOST
  RECENT occurrence is used for Entry/Stop/scoring.

  📋 DATA SOURCING
  Only 1 download per ticker: daily bars (~400 days).

  📋 OUTPUT (reasonable defaults — not explicitly requested)
  Entry_Price = the retest day's CLOSE
  Stop_Loss   = the retest day's LOW, just under the tested EMA21
  Signal_Date = the exact calendar date the pattern fired
  Days_Since_Signal = how many trading days ago (0 = today)
  Recent_Signals    = every date the pattern fired within the window

  No historical backtest is included in this scanner yet — it can
  be added the same way as three_candle_hammer_doji_v1.py's, if
  wanted (re-run the same check across full history, measure the
  forward price move after each historical occurrence).

  📋 SCORE (0-100)
  Convergence tightness (0-25) + retest closeness to EMA21 (0-25) +
  volume increase strength (0-25) + freshness (0-15) + base points
  for clearing every gate (10)

  💡 BEST SETUPS
  Score > 70              tight convergence, close retest, strong volume
  Gap_% < 1%                 SMA50 is right on the verge of crossing SMA150
  Retest_Near_% < 0.5%         price barely touched EMA21 and held
  Vol_Ratio > 1.5                 well above the prior day's volume
  Days_Since_Signal = 0-3            freshest retest

  ⚙️  TUNE IF 0 RESULTS
  sma_convergence_pct         2.0 → 3.5   (allow a wider convergence gap)
  retest_tolerance_pct        1.5 → 2.5   (allow a looser retest)
  sma50_slope_lookback          5 → 10    (more lenient "rising" check)
  signal_lookback_days         15 → 25    (search further back)
  min_price                      2 → 1
  min_avg_volume             80000 → 50000
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
""")
