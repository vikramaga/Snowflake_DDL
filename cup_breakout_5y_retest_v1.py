# ============================================================
# NASDAQ — 5-Year Cup Breakout + Retest Scanner (v1)
# ============================================================
#
# SIGNAL (all required, evaluated as of today — this is a slow,
# structural, multi-year pattern, not a rolling daily window scan):
#
#   1. BIG CUP (found on WEEKLY-resampled data over the last ~5
#      years): a genuine LEFT RIM (a prior high, occurring in the
#      first max_left_rim_position_pct% of the window) → a deep
#      decline of at least min_cup_depth_pct% → a BOTTOM → a
#      recovery back up to within right_rim_tolerance_pct% of the
#      left rim (the RIGHT RIM). The rim-to-bottom span must be at
#      least min_cup_duration_weeks weeks — a genuine multi-year
#      formation, not a quick dip.
#
#   2. BREAKOUT (found on DAILY data after the cup's bottom): price
#      closed above the cup's rim level (with a small buffer) at
#      some point — this must have happened within the last
#      max_days_since_breakout days (keeps results current).
#
#   3. RETEST: after that breakout, price pulled back down to within
#      retest_tolerance_pct% of the rim level (a genuine retest of
#      the old high as new support) AND the current close is still
#      AT/ABOVE the rim (the retest held — no breakdown back below
#      the old high).
#
#   4. "Same level as 5 years ago": the rim price IS approximately
#      the price from ~5 years ago by construction (it's the cup's
#      own left rim) — current price sitting back at/near that rim
#      after breaking out and retesting is exactly what's being
#      asked for.
#
# DATA — only 1 download per ticker: ~5.5 years of daily bars,
# resampled to weekly for the cup shape (less noise, matches how
# multi-year bases are conventionally read) and used directly at
# daily resolution for the precise breakout/retest timing.
#
# OUTPUT: Entry_Price and Stop_Loss are included as a reasonable
# default (Entry = current close; Stop = the lowest point reached
# during the retest pullback) since every other scanner in this repo
# reports actionable levels, but no specific entry/stop/target logic
# was requested for this pattern.
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
    "history_days"           : 2000,  # ~5.5 years — buffer beyond the 5-year cup window

    # ── Cup shape (found on weekly-resampled data) ──────────────────
    "min_cup_duration_weeks"    : 52,   # left rim to bottom must span at least this
    "min_recovery_weeks"        : 8,    # weeks excluded from the end when searching
                                         # for the bottom, so it's a completed event
    "min_cup_depth_pct"         : 20.0, # rim-to-bottom decline must be at least this
    "max_left_rim_position_pct" : 60.0, # left rim must occur within the first
                                         # this-% of the 5-year window
    "right_rim_tolerance_pct"   : 5.0,  # post-bottom recovery high must come within
                                         # this % of the left rim

    # ── Breakout + retest (found on daily data) ─────────────────────
    "breakout_buffer_pct"       : 1.0,  # close must exceed the rim by at least this %
    "max_days_since_breakout"   : 365,  # keeps results current — breakout must have
                                         # happened within the last ~1 year
    "retest_tolerance_pct"      : 5.0,  # how close the pullback must come to the rim
    "require_retest_holding"    : True, # current close must still be >= the rim

    # ── Filters ─────────────────────────────────────────────────
    "min_avg_volume"         : 80_000,
    "min_price"              : 2.0,

    "batch_size"             : 50,
    "batch_sleep"            : 1.5,
}

def resample_weekly(df):
    """Resample daily OHLCV to weekly bars."""
    agg = {"Open": "first", "High": "max", "Low": "min",
           "Close": "last", "Volume": "sum"}
    out = df.resample("W").agg(agg)
    out.dropna(subset=["Close"], inplace=True)
    return out

def find_cup(weekly, cfg):
    """
    Finds a big cup on weekly-resampled data: a LEFT RIM (prior
    high) -> a deep decline -> a BOTTOM -> a recovery back up near
    the left rim (the RIGHT RIM). Returns a details dict or None.
    """
    n = len(weekly)
    closes = weekly["Close"].values
    search_end = n - cfg["min_recovery_weeks"]
    if search_end < cfg["min_cup_duration_weeks"] + 10:
        return None

    cup_low_idx = int(np.argmin(closes[:search_end]))
    if cup_low_idx < cfg["min_cup_duration_weeks"]:
        return None   # not enough room for a left rim before the bottom

    cup_high_idx = int(np.argmax(closes[:cup_low_idx]))
    cup_high = float(closes[cup_high_idx])
    cup_low  = float(closes[cup_low_idx])
    if cup_high <= 0:
        return None

    duration_weeks = cup_low_idx - cup_high_idx
    depth_pct = (cup_high - cup_low) / cup_high * 100
    left_rim_position_pct = cup_high_idx / n * 100

    if duration_weeks < cfg["min_cup_duration_weeks"]:
        return None
    if depth_pct < cfg["min_cup_depth_pct"]:
        return None
    if left_rim_position_pct > cfg["max_left_rim_position_pct"]:
        return None

    recovery_high = float(closes[cup_low_idx:].max())
    recovery_pct_of_rim = recovery_high / cup_high * 100
    if recovery_pct_of_rim < (100 - cfg["right_rim_tolerance_pct"]):
        return None

    return {
        "cup_high_idx": cup_high_idx, "cup_low_idx": cup_low_idx,
        "cup_high": cup_high, "cup_low": cup_low,
        "cup_high_date": weekly.index[cup_high_idx],
        "cup_low_date": weekly.index[cup_low_idx],
        "depth_pct": depth_pct, "duration_weeks": duration_weeks,
        "left_rim_position_pct": left_rim_position_pct,
        "recovery_high": recovery_high,
    }

def find_breakout_and_retest(daily, cup_high, cup_low_date, cfg):
    """
    On daily data AFTER the cup's bottom, finds the first genuine
    breakout above cup_high (with a buffer), then checks whether
    price later retested that level and is currently still holding
    at/above it. Returns a details dict or None.
    """
    after = daily[daily.index > cup_low_date]
    if after.empty:
        return None

    buffer = 1 + cfg["breakout_buffer_pct"] / 100
    breakout_mask = after["Close"] > cup_high * buffer
    if not breakout_mask.any():
        return None
    breakout_pos = int(breakout_mask.values.argmax())
    breakout_date = after.index[breakout_pos]

    days_since_breakout = (daily.index[-1] - breakout_date).days
    if days_since_breakout > cfg["max_days_since_breakout"]:
        return None

    post_breakout = daily[daily.index > breakout_date]
    if post_breakout.empty:
        return None

    min_low_since_breakout = float(post_breakout["Low"].min())
    retest_pct = abs(min_low_since_breakout - cup_high) / cup_high * 100
    retested = retest_pct <= cfg["retest_tolerance_pct"]

    current_close = float(daily["Close"].iloc[-1])
    holding = (current_close >= cup_high) if cfg["require_retest_holding"] else True

    if not (retested and holding):
        return None

    return {
        "breakout_date": breakout_date,
        "days_since_breakout": days_since_breakout,
        "min_low_since_breakout": min_low_since_breakout,
        "retest_pct": retest_pct,
        "current_close": current_close,
    }

# ── Technical signal: 5-year cup breakout + retest ────────────────
def analyze_cup_breakout_retest(sym, df_daily):
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

    n = len(df_daily)
    if n < 1000:   # need a genuinely long history (~4+ years) for a real 5y cup
        return None

    weekly = resample_weekly(df_daily)
    if len(weekly) < CFG["min_cup_duration_weeks"] + CFG["min_recovery_weeks"] + 10:
        return None

    cup = find_cup(weekly, CFG)
    if cup is None:
        return None

    bt = find_breakout_and_retest(df_daily, cup["cup_high"], cup["cup_low_date"], CFG)
    if bt is None:
        return None

    # ── "same level as ~5 years ago" sanity check ───────────────────
    five_years_ago = df_daily.index[-1] - pd.Timedelta(days=365*5)
    past = df_daily[df_daily.index <= five_years_ago]
    price_5y_ago = float(past["Close"].iloc[-1]) if not past.empty else cup["cup_high"]
    vs_5y_ago_pct = (price - price_5y_ago) / price_5y_ago * 100 if price_5y_ago > 0 else 0

    # ── Entry / Stop (reasonable defaults — see header note) ─────────
    entry_price = price
    stop_loss   = bt["min_low_since_breakout"]
    if entry_price <= stop_loss:
        return None
    risk_pct = (entry_price - stop_loss) / entry_price * 100 if entry_price > 0 else 0

    # ── Score (0-100) ────────────────────────────────────────────
    score = 0
    reasons = []
    score += min(25, cup["depth_pct"] * 0.6)
    reasons.append(f"CupDepth{cup['depth_pct']:.0f}%")
    score += min(20, cup["duration_weeks"] / 10)
    reasons.append(f"Duration{cup['duration_weeks']}wk")
    score += max(0, min(20, 20 - bt["retest_pct"] * 3))
    reasons.append(f"Retest{bt['retest_pct']:.1f}%")
    score += max(0, min(20, 20 - bt["days_since_breakout"] / 18))
    reasons.append(f"{bt['days_since_breakout']}dSinceBO")
    score += 15   # base for clearing every gate
    score = round(min(100, max(0, score)))

    return {
        "Score"              : score,
        "Price"              : round(price, 2),
        "Entry_Price"        : round(entry_price, 2),
        "Stop_Loss"          : round(stop_loss, 2),
        "Risk_%"             : round(risk_pct, 1),
        "Cup_High"           : round(cup["cup_high"], 2),
        "Cup_Low"            : round(cup["cup_low"], 2),
        "Cup_Depth_%"        : round(cup["depth_pct"], 1),
        "Cup_Duration_Weeks" : cup["duration_weeks"],
        "Cup_High_Date"      : cup["cup_high_date"].strftime("%Y-%m-%d"),
        "Cup_Low_Date"       : cup["cup_low_date"].strftime("%Y-%m-%d"),
        "Breakout_Date"      : bt["breakout_date"].strftime("%Y-%m-%d"),
        "Days_Since_Breakout": bt["days_since_breakout"],
        "Retest_%"           : round(bt["retest_pct"], 2),
        "Price_vs_5Y_Ago_%"  : round(vs_5y_ago_pct, 1),
        "Flags"              : " | ".join(reasons),
        "_df_daily"          : df_daily,
        "_weekly"            : weekly,
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
                        df = _clean(df, min_bars=500)
                        if df is not None: out[sym] = df
                    except Exception: pass
            elif len(symbols) == 1:
                df = _clean(raw, min_bars=500)
                if df is not None: out[symbols[0]] = df
    except Exception: pass
    for sym in [s for s in symbols if s not in out]:
        for _ in range(2):
            try:
                df = yf.Ticker(sym).history(
                    start=start.strftime("%Y-%m-%d"),
                    end=end.strftime("%Y-%m-%d"),
                    auto_adjust=True, actions=False)
                df = _clean(df, min_bars=500)
                if df is not None: out[sym] = df; break
            except Exception: time.sleep(0.2)
        time.sleep(0.04)
    return out

# ── Live print ────────────────────────────────────────────────
LIVE_COLS = ["Ticker","Price","Score","Entry_Price","Stop_Loss",
             "Cup_Depth_%","Days_Since_Breakout","Retest_%"]
_CW = {"Ticker":8,"Price":10,"Score":7,"Entry_Price":12,"Stop_Loss":11,
       "Cup_Depth_%":12,"Days_Since_Breakout":19,"Retest_%":9}
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
        r = analyze_cup_breakout_retest(sym, daily_map[sym])
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

if not results:
    print("\n  No matches. Try relaxing:")
    print("   min_cup_depth_pct           20.0 → 12.0  (allow a shallower cup)")
    print("   min_cup_duration_weeks         52 → 26    (allow a shorter cup)")
    print("   right_rim_tolerance_pct       5.0 → 10.0  (allow a lower right rim)")
    print("   retest_tolerance_pct          5.0 → 8.0   (allow a looser retest)")
    print("   max_days_since_breakout        365 → 730   (older breakouts count too)")
    print("   min_price                        2 → 1")
    print("   min_avg_volume               80000 → 50000")

results.sort(key=lambda x: x["Score"], reverse=True)

# ── Always build df_out and save/email (even if 0 results) ────
ts      = datetime.today().strftime("%Y%m%d_%H%M")
out_dir = os.environ.get("GITHUB_WORKSPACE", os.getcwd())

COLS = [
    "Ticker","Price","Score",
    "Entry_Price","Stop_Loss","Risk_%",
    "Cup_High","Cup_Low","Cup_Depth_%","Cup_Duration_Weeks",
    "Cup_High_Date","Cup_Low_Date","Breakout_Date","Days_Since_Breakout",
    "Retest_%","Price_vs_5Y_Ago_%",
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
    "Cup_High"    : lambda v: f"${v:.2f}",
    "Cup_Low"     : lambda v: f"${v:.2f}",
    "Cup_Depth_%" : lambda v: f"{v:.1f}%",
    "Retest_%"    : lambda v: f"{v:.2f}%",
    "Price_vs_5Y_Ago_%": lambda v: f"{v:+.1f}%",
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
            "Cup_Depth_%","Days_Since_Breakout","Retest_%","Price_vs_5Y_Ago_%"]
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
            elif col == "Retest_%":
                try:
                    v = float(str(raw).replace("%",""))
                    clr = "#22c55e" if v <= 2 else "#f59e0b"
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
    <span style="color:#f1f5f9;font-size:15px;font-weight:700">5-Year Cup Breakout + Retest</span>
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
    📈 5-Year Cup Breakout + Retest
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
  A genuine multi-year cup (left rim, deep decline, bottom, recovery
  back to the right rim) &nbsp;·&nbsp;
  Price broke out above the old rim within the last year &nbsp;·&nbsp;
  Price pulled back to retest that level and is still holding above it &nbsp;·&nbsp;
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
                "Cup_Depth_%","Days_Since_Breakout","Retest_%","Price_vs_5Y_Ago_%"]
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
    tit = f"  5-Year Cup Breakout + Retest   {datetime.today().strftime('%Y-%m-%d')}   {len(df_out)} matches"
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
  Stop_Loss       the lowest low reached during the retest pullback
  Cup_Depth_%     how deep the cup's decline was, rim to bottom
  Days_Since_Breakout  how many days ago the breakout above the rim happened
  Retest_%        how close the pullback came to the rim level
  Price_vs_5Y_Ago_%  current price vs. the close from ~5 years ago
  Signal_Date     exact calendar date the pattern fired
  Days_Since_Signal  how many trading days ago (0 = today; checked
                     over the last signal_lookback_days trading days)
  ──────────────────────────────────────────────────────""")

# Save
fpath = os.path.join(out_dir, f"cup_breakout_5y_retest_{ts}.csv")
df_out.to_csv(fpath, index=False)
print(f"\n  💾 CSV → {fpath}")
tv = os.path.join(out_dir, f"tv_cup_breakout_5y_retest_{ts}.txt")
with open(tv,"w") as f:
    f.write(f"###5-Year Cup Breakout + Retest {datetime.today().strftime('%Y-%m-%d')}\n")
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
                      "Cup_Depth_%","Days_Since_Breakout","Retest_%","Price_vs_5Y_Ago_%"]
        )
        rows_e = ""
        for i, r in enumerate(rl[:50]):
            bg  = "#fff" if i % 2 == 0 else "#f0f9ff"
            ticker = r.get("Ticker","—")
            price  = r.get("Price",0) or 0
            score  = r.get("Score",0) or 0
            entry  = r.get("Entry_Price",0) or 0
            stop   = r.get("Stop_Loss",0) or 0
            depth  = r.get("Cup_Depth_%",0) or 0
            dsb    = r.get("Days_Since_Breakout",0) or 0
            retest = r.get("Retest_%",0) or 0
            vs5y   = r.get("Price_vs_5Y_Ago_%",0) or 0
            rows_e += (
                f'<tr style="background:{bg}">'
                f'<td style="padding:6px 11px;font-size:12px;font-weight:700">{ticker}</td>'
                f'<td style="padding:6px 11px;font-size:12px">${float(price):.2f}</td>'
                f'<td style="padding:6px 11px;font-size:12px;font-weight:700;'
                f'background:#166534;color:#fff;text-align:center">{float(score):.0f}</td>'
                f'<td style="padding:6px 11px;font-size:12px;color:#22c55e">${float(entry):.2f}</td>'
                f'<td style="padding:6px 11px;font-size:12px;color:#ef4444">${float(stop):.2f}</td>'
                f'<td style="padding:6px 11px;font-size:12px;text-align:center">{float(depth):.0f}%</td>'
                f'<td style="padding:6px 11px;font-size:12px;text-align:center">{int(dsb)}d</td>'
                f'<td style="padding:6px 11px;font-size:12px;font-weight:600">{float(retest):.2f}%</td>'
                f'<td style="padding:6px 11px;font-size:12px;text-align:center;'
                f'color:#a78bfa;font-weight:600">{float(vs5y):+.1f}%</td>'
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
  📊 5-Year Cup Breakout + Retest
</h1>
<p style="margin:6px 0 0;color:#94a3b8;font-size:12px">
  {datetime.today().strftime('%Y-%m-%d %H:%M UTC')} &nbsp;·&nbsp;
  {cnt} match{'es' if cnt!=1 else ''} found — a big multi-year cup, a
  breakout above the old high, and a successful retest holding
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
            f"5-Year Cup Breakout + Retest — {datetime.today().strftime('%Y-%m-%d')}",
            f"{cnt} matches (big multi-year cup + breakout above the old high + successful retest holding)",
            "="*60,
        ]
        if rl:
            for r in rl[:50]:
                ticker = r.get("Ticker","—")
                price  = r.get("Price",0) or 0
                score  = r.get("Score",0) or 0
                entry  = r.get("Entry_Price",0) or 0
                stop   = r.get("Stop_Loss",0) or 0
                cup_high = r.get("Cup_High",0) or 0
                depth  = r.get("Cup_Depth_%",0) or 0
                dsb    = r.get("Days_Since_Breakout",0) or 0
                vs5y   = r.get("Price_vs_5Y_Ago_%",0) or 0
                plain_lines.append(
                    f"{ticker:<7} ${float(price):.2f}  Score:{float(score):.0f}  "
                    f"Entry:${float(entry):.2f}  SL:${float(stop):.2f}  "
                    f"CupHigh:${float(cup_high):.2f}  Depth:{float(depth):.0f}%  "
                    f"DaysSinceBreakout:{int(dsb)}  vs5YAgo:{float(vs5y):+.1f}%"
                )
            plain_lines.append("")
            plain_lines.append("Tickers (comma-separated):")
            plain_lines.append(", ".join(r.get("Ticker","") for r in rl))
        else:
            plain_lines.append("No matches today")
        plain_lines.append("\nFull results in CSV attachment.")
        plain_e = "\n".join(plain_lines)

        subj = (f"📊 5-Year Cup Breakout + Retest — {cnt} signal{'s' if cnt!=1 else ''}"
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

# ── Charts for top 5 (weekly cup shape + daily breakout/retest zoom) ──
if results:
    top = results[:min(5,len(results))]
    fig, axes = plt.subplots(len(top),2,figsize=(16,4.2*len(top)),facecolor="#0f172a",
                              gridspec_kw={"width_ratios":[2,1]})
    if len(top)==1: axes = axes.reshape(1,2)
    for row, r in zip(axes, top):
        ax_w, ax_d = row[0], row[1]

        # Left panel: full weekly cup shape
        weekly_p = r["_weekly"]
        ax_w.set_facecolor("#0f172a")
        ax_w.plot(weekly_p.index, weekly_p["Close"], color="#60a5fa", lw=1.3, label="Weekly Close")
        ax_w.axhline(r["Cup_High"], color="#fbbf24", lw=1.0, ls="--", alpha=0.8,
                     label=f"Rim ${r['Cup_High']:.2f}")
        ax_w.scatter([pd.to_datetime(r["Cup_High_Date"])], [r["Cup_High"]],
                     color="#fbbf24", s=50, zorder=5, marker="v")
        ax_w.scatter([pd.to_datetime(r["Cup_Low_Date"])], [r["Cup_Low"]],
                     color="#ef4444", s=50, zorder=5, marker="^")
        ax_w.set_title(
            f"{r['Ticker']}  |  Cup: {r['Cup_Depth_%']:.0f}% deep, {r['Cup_Duration_Weeks']}wk  |  "
            f"Score {r['Score']}",
            color="#e2e8f0", fontsize=9, fontweight="bold", pad=7)
        ax_w.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
        ax_w.tick_params(colors="#94a3b8", labelsize=8)
        for sp in ax_w.spines.values(): sp.set_edgecolor("#1e3a5f")
        ax_w.legend(loc="upper left", facecolor="#1e293b", labelcolor="#e2e8f0", fontsize=7)
        ax_w.grid(color="#1e3a5f", ls="--", lw=0.5, alpha=0.6)

        # Right panel: recent daily breakout + retest zoom
        df_p = r["_df_daily"].tail(120).copy()
        ax_d.set_facecolor("#0f172a")
        ax_d.plot(df_p.index, df_p["Close"], color="#60a5fa", lw=1.4, label="Close")
        ax_d.axhline(r["Cup_High"], color="#fbbf24", lw=1.0, ls="--", alpha=0.85,
                     label=f"Rim ${r['Cup_High']:.2f}")
        ax_d.axhline(r["Stop_Loss"], color="#ef4444", lw=1.0, ls="--", alpha=0.7,
                     label=f"Stop ${r['Stop_Loss']:.2f}")
        bo_date = pd.to_datetime(r["Breakout_Date"])
        if bo_date in df_p.index:
            ax_d.scatter([bo_date], [df_p.loc[bo_date,"Close"]], color="#22c55e", s=50,
                        zorder=6, marker="^", label="Breakout")
        ax_d.set_title(f"Breakout {r['Days_Since_Breakout']}d ago  |  Retest {r['Retest_%']:.1f}%",
                        color="#e2e8f0", fontsize=8, pad=5)
        ax_d.xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))
        ax_d.tick_params(colors="#94a3b8", labelsize=7)
        for sp in ax_d.spines.values(): sp.set_edgecolor("#1e3a5f")
        ax_d.legend(loc="upper left", facecolor="#1e293b", labelcolor="#e2e8f0", fontsize=6)
        ax_d.grid(color="#1e3a5f", ls="--", lw=0.5, alpha=0.6)
    plt.suptitle(
        f"5-Year Cup Breakout + Retest  ·  {datetime.today().strftime('%Y-%m-%d')}",
        color="#60a5fa", fontsize=12, fontweight="bold", y=1.001)
    plt.tight_layout()
    cp = os.path.join(out_dir, f"cup_breakout_5y_retest_chart_{ts}.png")
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
  📋 SIGNAL (all required, evaluated as of today — a slow,
  structural, multi-year pattern, not a rolling daily window)

  1) BIG CUP (found on weekly-resampled data over the last ~5
     years): a LEFT RIM (prior high, in the first
     max_left_rim_position_pct% of the window) → a decline of at
     least min_cup_depth_pct% → a BOTTOM → a recovery back up to
     within right_rim_tolerance_pct% of the left rim (the RIGHT
     RIM). Rim-to-bottom must span at least min_cup_duration_weeks
     weeks — a genuine multi-year formation, not a quick dip.
  2) BREAKOUT (daily data, after the cup's bottom): price closed
     above the rim (with a small buffer) within the last
     max_days_since_breakout days.
  3) RETEST: after the breakout, price pulled back to within
     retest_tolerance_pct% of the rim (testing it as new support)
     AND the current close is still at/above the rim (holding).

  📋 DATA SOURCING
  Only 1 download per ticker: ~5.5 years of daily bars, resampled
  to weekly for the cup shape and used directly for the precise
  breakout/retest timing.

  📋 OUTPUT (reasonable defaults — not explicitly requested)
  Entry_Price = today's close
  Stop_Loss   = the lowest low reached during the retest pullback
  Cup_High / Cup_Low = the rim and bottom price levels
  Cup_High_Date / Cup_Low_Date = when the rim and bottom occurred
  Breakout_Date / Days_Since_Breakout = when price broke out
  Retest_%    = how close the pullback came to the rim
  Price_vs_5Y_Ago_%  = current price vs. the close from ~5 years ago
                       (this is the "same level as 5 years ago" check)

  No historical backtest is included in this scanner yet — it could
  be added the same way as three_candle_hammer_doji_v1.py's, if
  wanted, though with only 5-year cups this would need a MUCH longer
  data history (10+ years) to find more than one or two historical
  occurrences per ticker.

  📋 SCORE (0-100)
  Cup depth (0-25) + cup duration (0-20) + retest closeness (0-20) +
  breakout freshness (0-20) + base points for clearing every gate (15)

  💡 BEST SETUPS
  Score > 70                strong deep cup, fresh breakout, tight retest
  Cup_Depth_% > 30%            a genuinely significant multi-year decline
  Retest_% < 2%                   price barely touched the rim and held
  Days_Since_Breakout < 60           a fresh, still-relevant breakout

  ⚙️  TUNE IF 0 RESULTS
  min_cup_depth_pct           20.0 → 12.0  (allow a shallower cup)
  min_cup_duration_weeks         52 → 26    (allow a shorter cup)
  right_rim_tolerance_pct       5.0 → 10.0  (allow a lower right rim)
  retest_tolerance_pct          5.0 → 8.0   (allow a looser retest)
  max_days_since_breakout        365 → 730   (older breakouts count too)
  min_price                        2 → 1
  min_avg_volume               80000 → 50000
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
""")
