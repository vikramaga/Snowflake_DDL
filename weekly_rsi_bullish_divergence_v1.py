# ============================================================
# NASDAQ — Weekly RSI Bullish Divergence (v1)
# ============================================================
#
# Same pattern as daily_rsi_bullish_divergence_v1.py, one timeframe
# up: swing lows and RSI computed on WEEKLY candles instead of daily.
#
# SIGNAL — classic bullish divergence between the two most recent
# confirmed WEEKLY swing lows (fractal lows: strictly lower than
# swing_arm weekly bars on both sides):
#
#   1. PRICE BROKE THE LAST LOW: the most recent swing low's price
#      is LOWER than the swing low before it.
#
#   2. RSI DID NOT BREAK ITS LAST LOW: weekly RSI(14) at the most
#      recent swing low is HIGHER than RSI at the swing low before
#      it — momentum did not confirm the new price low, the classic
#      divergence signature.
#
#   3. FRESHNESS: the most recent swing low must have been confirmed
#      within the last max_bars_since_swing_low WEEKS — not an old,
#      stale divergence.
#
# A swing low needs swing_arm weekly bars AFTER it to be confirmed
# (it's a fractal low), so the most recent usable swing low
# necessarily lags "today" by at least that many weeks — this is
# inherent to detecting a local low, not a scanner limitation.
#
# DATA — only 1 download per ticker: daily bars (resampled to
# weekly — no separate network calls needed).
#
# OUTPUT: Entry_Price and Stop_Loss are reasonable defaults (Entry =
# today's close; Stop = the most recent swing low's price), not
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
    # NOTE: history_days/min_bars sized for WEEKLY swing detection —
    # verified numerically: 900 calendar days -> ~643 business days
    # (~129 weekly bars), comfortable margin above the ~84 weekly
    # bars needed (rsi_period + swing_search_window + buffer).
    "history_days"          : 900,

    # ── Indicator periods (all computed on WEEKLY closes) ─────────
    "rsi_period"              : 14,
    "swing_arm"               : 3,   # WEEKLY bars of strictly-higher lows
                                      # required on each side to confirm a
                                      # fractal swing low (fewer than daily's
                                      # 5, since each bar already spans a week)
    "swing_search_window"     : 60,  # how far back (WEEKLY bars, ~14 months)
                                      # to search for swing lows

    "max_bars_since_swing_low": 8,   # the most recent swing low must have
                                      # been confirmed within this many WEEKS

    # ── Filters ─────────────────────────────────────────────────
    "min_avg_volume"         : 80_000,
    "min_price"              : 2.0,

    "batch_size"             : 50,
    "batch_sleep"            : 1.5,
}

# ── Indicators ───────────────────────────────────────────────
def resample_ohlcv(df, rule):
    """Resample daily OHLCV to a coarser timeframe (e.g. 'W' for weekly)."""
    agg = {"Open": "first", "High": "max", "Low": "min",
           "Close": "last", "Volume": "sum"}
    out = df.resample(rule).agg(agg)
    out.dropna(subset=["Close"], inplace=True)
    return out

def calc_rsi(close, period=14):
    """Standard Wilder-smoothed RSI."""
    delta = close.diff()
    gain  = delta.clip(lower=0)
    loss  = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1/period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1/period, min_periods=period, adjust=False).mean()
    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))

def find_swing_lows(df, arm, window):
    """
    Returns a list of bar indices (within the last `window` bars)
    that are confirmed fractal swing lows: strictly lower than the
    Low of `arm` bars on BOTH sides. Oldest first.
    """
    n = len(df)
    lows = df["Low"].values
    start = max(arm, n - window)
    swing_idx = []
    for i in range(start, n - arm):
        left = lows[i-arm:i]
        right = lows[i+1:i+arm+1]
        if lows[i] < left.min() and lows[i] < right.min():
            swing_idx.append(i)
    return swing_idx

def check_divergence(df, rsi, swing_idx, cfg):
    """
    Checks the two MOST RECENT confirmed swing lows for a bullish
    RSI divergence: price makes a lower low, but RSI does not. This
    is the SINGLE SOURCE OF TRUTH for the pattern logic.

    Returns (passed: bool, details: dict). details is {} only when
    there aren't at least 2 swing lows or indicator data is missing;
    otherwise it always contains every stage's boolean, even on
    failure.
    """
    if len(swing_idx) < 2:
        return False, {}

    i2, i1 = swing_idx[-1], swing_idx[-2]   # i2 = most recent, i1 = prior
    n = len(df)

    low2, low1 = df["Low"].iloc[i2], df["Low"].iloc[i1]
    rsi2, rsi1 = rsi.iloc[i2], rsi.iloc[i1]
    if any(np.isnan(v) for v in [low2, low1, rsi2, rsi1]):
        return False, {}

    price_broke_low = bool(low2 < low1)
    rsi_held_higher = bool(rsi2 > rsi1)
    fresh = bool(((n - 1) - i2) <= cfg["max_bars_since_swing_low"])

    passed = price_broke_low and rsi_held_higher and fresh

    return passed, {
        "i1": i1, "i2": i2, "low1": float(low1), "low2": float(low2),
        "rsi1": float(rsi1), "rsi2": float(rsi2),
        "date1": df.index[i1], "date2": df.index[i2],
        "bars_since_swing_low": (n - 1) - i2,
        "price_broke_low": price_broke_low, "rsi_held_higher": rsi_held_higher,
        "fresh": fresh,
    }

# ── Diagnostic funnel — tallies how far each ticker gets through the
#    pattern, across the FULL universe scan, so a 0-match run can be
#    diagnosed empirically instead of guessed at. Evaluated once per
#    ticker (its two most recent confirmed swing lows) ─────────────
FUNNEL_COUNTS = {
    "tickers_checked": 0,
    "passed_step1_price_broke_low": 0,
    "passed_step2_rsi_held_higher": 0,
    "passed_step3_fresh": 0,
}

# ── Technical signal: daily RSI bullish divergence ─────────────────
def analyze_rsi_divergence(sym, df_daily):
    """
    Returns dict with score and setup details, or None if no
    required condition is met.
    """
    if df_daily is None:
        return None

    price   = float(df_daily["Close"].iloc[-1])   # today's actual price
    avg_vol = float(df_daily["Volume"].tail(20).mean())
    if price   < CFG["min_price"]:      return None
    if avg_vol < CFG["min_avg_volume"]: return None

    df_weekly = resample_ohlcv(df_daily, "W")
    n = len(df_weekly)
    if n < CFG["rsi_period"] + CFG["swing_search_window"] + 10:
        return None

    rsi = calc_rsi(df_weekly["Close"], CFG["rsi_period"])
    swing_idx = find_swing_lows(df_weekly, CFG["swing_arm"], CFG["swing_search_window"])

    global FUNNEL_COUNTS
    passed, sig = check_divergence(df_weekly, rsi, swing_idx, CFG)
    if sig:
        FUNNEL_COUNTS["tickers_checked"] += 1
        if sig["price_broke_low"]:
            FUNNEL_COUNTS["passed_step1_price_broke_low"] += 1
            if sig["rsi_held_higher"]:
                FUNNEL_COUNTS["passed_step2_rsi_held_higher"] += 1
                if sig["fresh"]:
                    FUNNEL_COUNTS["passed_step3_fresh"] += 1
    if not passed:
        return None

    # ── Entry / Stop (reasonable defaults — see header note) ─────────
    entry_price = price
    stop_loss   = sig["low2"]
    if entry_price <= stop_loss:
        return None
    risk_pct = (entry_price - stop_loss) / entry_price * 100 if entry_price > 0 else 0

    price_break_pct = (sig["low1"] - sig["low2"]) / sig["low1"] * 100 if sig["low1"] > 0 else 0
    rsi_divergence_pts = sig["rsi2"] - sig["rsi1"]

    # ── Score (0-100) ────────────────────────────────────────────
    score = 0
    reasons = []
    score += min(30, rsi_divergence_pts * 1.5)
    reasons.append(f"RSIDiverge+{rsi_divergence_pts:.1f}")
    score += min(20, price_break_pct * 10)
    reasons.append(f"PriceBreak-{price_break_pct:.1f}%")
    freshness_pts = max(0, 15 - sig["bars_since_swing_low"] * (15/CFG["max_bars_since_swing_low"]))
    score += freshness_pts
    reasons.append(f"{sig['bars_since_swing_low']}wSinceLow")
    bounce_pct = (price - sig["low2"]) / sig["low2"] * 100 if sig["low2"] > 0 else 0
    score += max(0, min(10, bounce_pct))
    reasons.append(f"Bounce+{bounce_pct:.1f}%")
    score += 25   # base for clearing every gate
    score = round(min(100, max(0, score)))

    return {
        "Score"          : score,
        "Price"          : round(price, 2),
        "Entry_Price"    : round(entry_price, 2),
        "Stop_Loss"      : round(stop_loss, 2),
        "Risk_%"         : round(risk_pct, 1),
        "RSI_At_Low2"    : round(sig["rsi2"], 1),
        "RSI_At_Low1"    : round(sig["rsi1"], 1),
        "RSI_Divergence_Pts": round(rsi_divergence_pts, 1),
        "Price_Low2"     : round(sig["low2"], 2),
        "Price_Low1"     : round(sig["low1"], 2),
        "Price_Break_%"  : round(price_break_pct, 1),
        "Bounce_From_Low_%": round(bounce_pct, 1),
        "Swing_Low_Date" : sig["date2"].strftime("%Y-%m-%d"),
        "Prior_Swing_Low_Date": sig["date1"].strftime("%Y-%m-%d"),
        "Weeks_Since_Swing_Low": sig["bars_since_swing_low"],
        "Flags"          : " | ".join(reasons),
        "_df_daily"      : df_daily,
        "_df_weekly"     : df_weekly,
        "_rsi"           : rsi,
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
                        df = _clean(df, min_bars=400)
                        if df is not None: out[sym] = df
                    except Exception: pass
            elif len(symbols) == 1:
                df = _clean(raw, min_bars=400)
                if df is not None: out[symbols[0]] = df
    except Exception: pass
    for sym in [s for s in symbols if s not in out]:
        for _ in range(2):
            try:
                df = yf.Ticker(sym).history(
                    start=start.strftime("%Y-%m-%d"),
                    end=end.strftime("%Y-%m-%d"),
                    auto_adjust=True, actions=False)
                df = _clean(df, min_bars=400)
                if df is not None: out[sym] = df; break
            except Exception: time.sleep(0.2)
        time.sleep(0.04)
    return out

# ── Live print ────────────────────────────────────────────────
LIVE_COLS = ["Ticker","Price","Score","Entry_Price","Stop_Loss",
             "RSI_At_Low2","RSI_At_Low1","Swing_Low_Date"]
_CW = {"Ticker":8,"Price":10,"Score":7,"Entry_Price":12,"Stop_Loss":11,
       "RSI_At_Low2":12,"RSI_At_Low1":12,"Swing_Low_Date":14}
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
print("  A stock only matches if the divergence and freshness conditions")
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

for sym in tqdm(list(daily_map.keys()), desc="Checking 3-candle pattern", unit="stk"):
    try:
        r = analyze_rsi_divergence(sym, daily_map[sym])
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
print(f"  (evaluated once per liquid ticker, on its 2 most recent confirmed swing lows)")
print(f"{'━'*65}")
fc = FUNNEL_COUNTS
print(f"  Tickers checked                            : {fc['tickers_checked']}")
print(f"  Step 1 — price broke its last swing low     : {fc['passed_step1_price_broke_low']}")
print(f"  Step 2 — + RSI held above its last swing low : {fc['passed_step2_rsi_held_higher']}")
print(f"  Step 3 — + fresh (within {CFG['max_bars_since_swing_low']}w)              : {fc['passed_step3_fresh']}  (= full pattern)")
print(f"{'━'*65}")

if not results:
    print("\n  No matches. Try relaxing (see the FUNNEL above to see which")
    print("  step is actually the bottleneck before guessing):")
    print("   max_bars_since_swing_low         8 → 14    (allow an older divergence)")
    print("   swing_arm                        5 → 3    (less strict swing confirmation)")
    print("   swing_search_window            150 → 250   (search further back)")
    print("   min_price                      2 → 1")
    print("   min_avg_volume              80000 → 50000")

results.sort(key=lambda x: x["Score"], reverse=True)

# ── Always build df_out and save/email (even if 0 results) ────
ts      = datetime.today().strftime("%Y%m%d_%H%M")
out_dir = os.environ.get("GITHUB_WORKSPACE", os.getcwd())

COLS = [
    "Ticker","Price","Score",
    "Entry_Price","Stop_Loss","Risk_%",
    "RSI_At_Low2","RSI_At_Low1","RSI_Divergence_Pts",
    "Price_Low2","Price_Low1","Price_Break_%","Bounce_From_Low_%",
    "Swing_Low_Date","Prior_Swing_Low_Date","Weeks_Since_Swing_Low",
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

    "Above_SMA50_%": lambda v: f"{v:+.1f}%",
    "RSI_At_Low2" : lambda v: f"{v:.1f}",
    "RSI_At_Low1" : lambda v: f"{v:.1f}",
    "RSI_Divergence_Pts": lambda v: f"+{v:.1f}",
    "Price_Low2"  : lambda v: f"${v:.2f}",
    "Price_Low1"  : lambda v: f"${v:.2f}",
    "Price_Break_%": lambda v: f"-{v:.1f}%",
    "Bounce_From_Low_%": lambda v: f"+{v:.1f}%",
    "SMA150"      : lambda v: f"${v:.2f}",
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
            "RSI_At_Low2","RSI_At_Low1","RSI_Divergence_Pts","Swing_Low_Date"]
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
            elif col == "RSI_Divergence_Pts":
                try:
                    v = float(str(raw).replace("+",""))
                    clr = "#22c55e" if v >= 5 else "#f59e0b"
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
    <span style="color:#f1f5f9;font-size:15px;font-weight:700">Weekly RSI Bullish Divergence</span>
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
    📈 Weekly RSI Bullish Divergence
  </h2>
  <p style="margin:6px 0 0;color:#94a3b8;font-size:12px">
    {datetime.today().strftime('%Y-%m-%d %H:%M')} &nbsp;·&nbsp;
    <b style="color:#22c55e">{len(results)} matches</b> from {len(TICKERS)} tickers
    &nbsp;·&nbsp; swing low confirmed within the last {CFG['max_bars_since_swing_low']} weeks
  </p>
</div>"""

    legend_html = f"""
<div style="background:#f8fafc;border:1px solid #e2e8f0;border-radius:8px;
        padding:12px 18px;margin-top:6px;font-size:11px;color:#64748b;
        font-family:'Segoe UI',Arial,sans-serif">
  <b style="color:#475569">GUIDE</b> &nbsp;·&nbsp;
  Price's most recent confirmed swing low is LOWER than the swing
  low before it &nbsp;·&nbsp;
  RSI at that same low is HIGHER than RSI at the prior low (bullish
  divergence — momentum didn't confirm the new price low) &nbsp;·&nbsp;
  Entry_Price = the signal day's close &nbsp;·&nbsp;
  Stop_Loss = a 10-day swing-low proxy — reasonable defaults, not
  explicitly requested &nbsp;·&nbsp;
  Signal_Date is the exact date the pattern fired (checked over the
  last {CFG['max_bars_since_swing_low']} weeks)
</div>"""

    display_html(header_html + table_html + legend_html)

elif results:
    # ASCII table (CLI/GitHub Actions mode)
    CLI_COLS = ["Ticker","Price","Score","Entry_Price","Stop_Loss",
                "RSI_At_Low2","RSI_At_Low1","RSI_Divergence_Pts","Swing_Low_Date"]
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
    tit = f"  Weekly RSI Bullish Divergence   {datetime.today().strftime('%Y-%m-%d')}   {len(df_out)} matches"
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
  Score           0-100 (RSI divergence magnitude + price-break
                  depth + freshness + initial bounce strength)
  Entry_Price     today's close
  Stop_Loss       the most recent swing low's price
  RSI_At_Low2 / RSI_At_Low1  RSI at the recent / prior swing low
  RSI_Divergence_Pts  RSI_At_Low2 - RSI_At_Low1 (must be positive)
  Price_Low2 / Price_Low1    the recent / prior swing low's price
  Swing_Low_Date / Prior_Swing_Low_Date  the two dates compared
  Weeks_Since_Swing_Low  how many weeks since the most recent
                        swing low (checked over the swing_search_window)
  ──────────────────────────────────────────────────────""")

# Save
fpath = os.path.join(out_dir, f"weekly_rsi_bullish_divergence_{ts}.csv")
df_out.to_csv(fpath, index=False)
print(f"\n  💾 CSV → {fpath}")
tv = os.path.join(out_dir, f"tv_weekly_rsi_bullish_divergence_{ts}.txt")
with open(tv,"w") as f:
    f.write(f"###Weekly RSI Bullish Divergence {datetime.today().strftime('%Y-%m-%d')}\n")
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
                      "RSI_At_Low2","RSI_At_Low1","RSI_Divergence_Pts","Swing_Low_Date"]
        )
        rows_e = ""
        for i, r in enumerate(rl[:50]):
            bg  = "#fff" if i % 2 == 0 else "#f0f9ff"
            ticker = r.get("Ticker","—")
            price  = r.get("Price",0) or 0
            score  = r.get("Score",0) or 0
            entry  = r.get("Entry_Price",0) or 0
            stop   = r.get("Stop_Loss",0) or 0
            rsi2v  = r.get("RSI_At_Low2",0) or 0
            rsi1v  = r.get("RSI_At_Low1",0) or 0
            divpts = r.get("RSI_Divergence_Pts",0) or 0
            sdate  = r.get("Swing_Low_Date","—")
            rows_e += (
                f'<tr style="background:{bg}">'
                f'<td style="padding:6px 11px;font-size:12px;font-weight:700">{ticker}</td>'
                f'<td style="padding:6px 11px;font-size:12px">${float(price):.2f}</td>'
                f'<td style="padding:6px 11px;font-size:12px;font-weight:700;'
                f'background:#166534;color:#fff;text-align:center">{float(score):.0f}</td>'
                f'<td style="padding:6px 11px;font-size:12px;color:#22c55e">${float(entry):.2f}</td>'
                f'<td style="padding:6px 11px;font-size:12px;color:#ef4444">${float(stop):.2f}</td>'
                f'<td style="padding:6px 11px;font-size:12px;color:#3b82f6">{float(rsi2v):.1f}</td>'
                f'<td style="padding:6px 11px;font-size:12px">{float(rsi1v):.1f}</td>'
                f'<td style="padding:6px 11px;font-size:12px;font-weight:600">+{float(divpts):.1f}</td>'
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
  📊 Weekly RSI Bullish Divergence
</h1>
<p style="margin:6px 0 0;color:#94a3b8;font-size:12px">
  {datetime.today().strftime('%Y-%m-%d %H:%M UTC')} &nbsp;·&nbsp;
  {cnt} match{'es' if cnt!=1 else ''} found — red, red (higher low), green
  price broke its last swing low, RSI did not
</p>
  </td></tr>
  <tr><td style="padding:14px 28px 4px;background:#0b1220">
<div style="background:#111827;border:1px solid #1f2937;border-radius:8px;padding:12px 16px">
  <p style="margin:0 0 6px;color:#93c5fd;font-size:12px;font-weight:700">
    🔍 FUNNEL — where tickers were filtered out (evaluated once per liquid ticker)
  </p>
  <p style="margin:0;color:#cbd5e1;font-size:12px">
    {FUNNEL_COUNTS['tickers_checked']} tickers checked &nbsp;→&nbsp;
    {FUNNEL_COUNTS['passed_step1_price_broke_low']} passed Step 1 (price broke low) &nbsp;→&nbsp;
    {FUNNEL_COUNTS['passed_step2_rsi_held_higher']} passed Step 2 (RSI held higher) &nbsp;→&nbsp;
    <b style="color:#facc15">{FUNNEL_COUNTS['passed_step3_fresh']} passed Step 3 (fresh) = full match</b>
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
            f"Weekly RSI Bullish Divergence — {datetime.today().strftime('%Y-%m-%d')}",
            f"{cnt} matches (price broke its last swing low, RSI held above its last swing low)",
            "="*60,
            f"FUNNEL: {FUNNEL_COUNTS['tickers_checked']} tickers -> "
            f"{FUNNEL_COUNTS['passed_step1_price_broke_low']} passed price-broke-low -> "
            f"{FUNNEL_COUNTS['passed_step2_rsi_held_higher']} passed rsi-held-higher -> "
            f"{FUNNEL_COUNTS['passed_step3_fresh']} passed fresh (=full match)",
            "="*60,
        ]
        if rl:
            for r in rl[:50]:
                ticker = r.get("Ticker","—")
                price  = r.get("Price",0) or 0
                score  = r.get("Score",0) or 0
                entry  = r.get("Entry_Price",0) or 0
                stop   = r.get("Stop_Loss",0) or 0
                rsi2v  = r.get("RSI_At_Low2",0) or 0
                rsi1v  = r.get("RSI_At_Low1",0) or 0
                divpts = r.get("RSI_Divergence_Pts",0) or 0
                sdate  = r.get("Swing_Low_Date","—")
                plain_lines.append(
                    f"{ticker:<7} ${float(price):.2f}  Score:{float(score):.0f}  "
                    f"Entry:${float(entry):.2f}  SL:${float(stop):.2f}  "
                    f"RSI@Low2:{float(rsi2v):.1f}  RSI@Low1:{float(rsi1v):.1f}  Divergence:+{float(divpts):.1f}  SwingLow:{sdate}"
                )
            plain_lines.append("")
            plain_lines.append("Tickers (comma-separated):")
            plain_lines.append(", ".join(r.get("Ticker","") for r in rl))
        else:
            plain_lines.append("No matches today")
        plain_lines.append("\nFull results in CSV attachment.")
        plain_e = "\n".join(plain_lines)

        subj = (f"📊 Weekly RSI Bullish Divergence — {cnt} signal{'s' if cnt!=1 else ''}"
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

# ── Charts for top 5 (price panel with the 2 swing lows marked,
#    stacked above an RSI panel with the divergence line drawn) ──
if results:
    top = results[:min(5,len(results))]
    fig, axes = plt.subplots(len(top)*2,1,figsize=(13,3.0*len(top)*2),facecolor="#0f172a",
                              gridspec_kw={"height_ratios":[2,1]*len(top)})
    if len(top)==1: axes = [axes[0], axes[1]]
    for p, r in enumerate(top):
        ax = axes[p*2]; ax_rsi = axes[p*2+1]
        df_p = r["_df_weekly"].tail(60).copy()
        rsi_p = r["_rsi"].reindex(df_p.index)
        d1, d2 = pd.Timestamp(r["Prior_Swing_Low_Date"]), pd.Timestamp(r["Swing_Low_Date"])

        ax.set_facecolor("#0f172a")
        ax.plot(df_p.index, df_p["Close"], color="#60a5fa", lw=1.4, label="Close", zorder=5)
        if d1 in df_p.index and d2 in df_p.index:
            ax.plot([d1, d2], [r["Price_Low1"], r["Price_Low2"]],
                    color="#ef4444", lw=1.6, ls="--", marker="v", markersize=7,
                    label="Price lower low", zorder=6)
        ax.axhline(r["Stop_Loss"], color="#ef4444", lw=1.0, ls=":", alpha=0.7,
                  label=f"Stop ${r['Stop_Loss']:.2f}")
        ax.set_title(
            f"{r['Ticker']}  |  ${r['Price']:.2f}  |  Score {r['Score']}  |  "
            f"RSI divergence +{r['RSI_Divergence_Pts']:.1f}pts",
            color="#e2e8f0", fontsize=9, fontweight="bold", pad=7)
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))
        ax.tick_params(colors="#94a3b8", labelsize=8)
        for sp in ax.spines.values(): sp.set_edgecolor("#1e3a5f")
        ax.legend(loc="upper left", facecolor="#1e293b", labelcolor="#e2e8f0", fontsize=6)
        ax.grid(color="#1e3a5f", ls="--", lw=0.5, alpha=0.6)

        ax_rsi.set_facecolor("#0f172a")
        ax_rsi.plot(df_p.index, rsi_p, color="#f472b6", lw=1.3, label="RSI")
        if d1 in df_p.index and d2 in df_p.index:
            ax_rsi.plot([d1, d2], [r["RSI_At_Low1"], r["RSI_At_Low2"]],
                       color="#22c55e", lw=1.6, ls="--", marker="^", markersize=7,
                       label="RSI higher low", zorder=6)
        ax_rsi.axhline(30, color="#64748b", lw=0.8, ls=":")
        ax_rsi.set_ylim(0, 100)
        ax_rsi.xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))
        ax_rsi.tick_params(colors="#94a3b8", labelsize=7)
        for sp in ax_rsi.spines.values(): sp.set_edgecolor("#1e3a5f")
        ax_rsi.legend(loc="upper left", facecolor="#1e293b", labelcolor="#e2e8f0", fontsize=6)
        ax_rsi.grid(color="#1e3a5f", ls="--", lw=0.5, alpha=0.6)
    plt.suptitle(
        f"Weekly RSI Bullish Divergence  ·  {datetime.today().strftime('%Y-%m-%d')}",
        color="#60a5fa", fontsize=12, fontweight="bold", y=1.001)
    plt.tight_layout()
    cp = os.path.join(out_dir, f"weekly_rsi_bullish_divergence_chart_{ts}.png")
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
  📋 SIGNAL (all required — evaluated on the two most recent
  confirmed swing lows per ticker, searched over the last
  swing_search_window ({CFG['swing_search_window']}) weekly bars)

  A swing low is a fractal low: strictly lower than the Low of
  swing_arm ({CFG['swing_arm']}) bars on BOTH sides — it needs that many
  bars AFTER it to be confirmed, so the most recent usable swing low
  necessarily lags "today" by at least that many bars.

  1) PRICE BROKE THE LAST LOW: the most recent confirmed swing low
     is LOWER than the swing low before it.
  2) RSI DID NOT BREAK ITS LAST LOW: RSI(14) at that same swing low
     is HIGHER than RSI at the prior swing low — momentum did not
     confirm the new price low (the divergence).
  3) FRESHNESS: the most recent swing low must have been confirmed
     within the last max_bars_since_swing_low ({CFG['max_bars_since_swing_low']})
     weeks — not an old, stale divergence.

  📋 DATA SOURCING
  Only 1 download per ticker: daily bars (~{CFG['history_days']} days, ~2.5
  years — needed for a stable weekly RSI and enough weekly bars to
  search), resampled to weekly ('W') — no separate network calls
  needed.

  📋 OUTPUT (reasonable defaults — not explicitly requested)
  Entry_Price = today's CLOSE
  Stop_Loss   = the most recent swing low's price
  Swing_Low_Date / Prior_Swing_Low_Date = the two swing lows compared
  Weeks_Since_Swing_Low = how many weeks since the most recent one

  📋 SCORE (0-100)
  RSI divergence magnitude (0-30) + how far price broke below the
  prior low (0-20) + freshness (0-15) + initial bounce off the low
  (0-10) + base points for clearing every gate (25)

  💡 BEST SETUPS
  Score > 70                     strong divergence, fresh, already
                                  bouncing
  RSI_Divergence_Pts large           a decisive, not marginal,
                                      divergence
  Weeks_Since_Swing_Low low             the freshest divergence
  Bounce_From_Low_%  positive              price already reacting off
                                            the low, not just sitting there

  ⚙️  TUNE IF 0 RESULTS
  max_bars_since_swing_low         8 → 14    (allow an older divergence)
  swing_arm                        5 → 3    (less strict swing confirmation)
  swing_search_window            150 → 250   (search further back)
  min_price                        2 → 1
  min_avg_volume               80000 → 50000
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
""")
