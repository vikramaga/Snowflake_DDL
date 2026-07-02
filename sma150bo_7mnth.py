# ============================================================
# NASDAQ — Extended Below SMA150 → SMA150 Breakout (AXON Pattern)
# ============================================================
#
# PATTERN:
#
#  PHASE 1 — BELOW SMA150 FOR 3+ MONTHS  (not necessarily SMA50)
#    Price close was BELOW SMA150 for at least min_bear_bars
#    consecutive bars — price may still be above SMA50
#    AXON example: pulled back to SMA150 zone for several months,
#    never deep bear — just consolidating below the 150-day line
#
#  PHASE 2 — SMA150 BREAKOUT WITH VOLUME
#    Previous close < SMA150  →  Current close >= SMA150
#    Volume on breakout bar >= vol_mult × 20-day avg
#    = Institutional reclaim of the key long-term level
#
#  PHASE 3 — QUALITY FILTERS
#    SMA150 slope not steeply declining (flattening or turning up)
#    Price not too extended above SMA150
#    RSI >= rsi_min
#
# TIERS (by duration below SMA150 + volume):
#    🏆 Tier 1  >= 7 months below SMA150  + vol >= 1.5×
#    🥈 Tier 2  >= 5 months below SMA150  + vol >= 1.2×
#    🥉 Tier 3  >= 3 months below SMA150  + vol >= 0.8×
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
import requests, time, warnings, io, smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text      import MIMEText
from email.mime.base      import MIMEBase
from email                import encoders
from datetime import datetime, timedelta
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

warnings.filterwarnings("ignore")
pd.set_option("display.max_rows", 200)
env = "Colab/Jupyter" if _IN_NOTEBOOK else "Script/CI"
print(f"✅  yfinance {yf.__version__}  |  numpy {np.__version__}  |  [{env}]")

# ── Email secret diagnostic (same as sma_retest script) ───────
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
    print(f"  ℹ️  GitHub → repo → Settings → Secrets → Actions")
    print(f"       Add: GMAIL_USER, GMAIL_PASS (App Password), EMAIL_TO")
    print(f"  ℹ️  Email will be SKIPPED this run")
print("━"*65)
print()

# ── CONFIG ────────────────────────────────────────────────────
CFG = {
    "history_days"              : 600,   # need 2+ years for 7-month check

    # ── MA periods ────────────────────────────────────────────
    "sma50_period"              : 50,
    "sma150_period"             : 150,
    "ema20_period"              : 20,

    # ── Phase 1: Below SMA150 duration ────────────────────────
    # Minimum consecutive bars where close < SMA150
    # (SMA50 condition REMOVED — AXON was above SMA50 while below SMA150)
    # 3 months ≈ 63 trading bars
    "min_bear_bars"             : 63,    # 3 months minimum (was 147)

    # ── Phase 2: SMA150 breakout ─────────────────────────────
    # The SMA150 cross must be within last cross_lookback bars
    # AXON crossed SMA150 recently — allow up to 15 bars lookback
    "cross_lookback"            : 15,
    "pre_cross_below_bars"      : 1,

    # ── Volume confirmation ───────────────────────────────────
    "vol_avg_bars"              : 20,
    "min_vol_mult"              : 0.8,
    # Tier thresholds (by months below SMA150)
    "tier1_bars"                : 147,   # 7 months
    "tier1_vol_mult"            : 1.5,
    "tier2_bars"                : 105,   # 5 months
    "tier2_vol_mult"            : 1.2,
    "tier3_bars"                : 63,    # 3 months
    "tier3_vol_mult"            : 0.8,

    # ── Phase 3: Quality filters ──────────────────────────────
    # Price vs SMA150 — AXON is ~22% above SMA150 after breakout
    "max_above_sma150_pct"      : 50.0,  # very wide — don't reject extended moves
    "rsi_min"                   : 20,
    # SMA150 slope — AXON's SMA150 is still declining during breakout
    # Set very permissive — we care about price breakout, not MA direction
    "sma150_slope_bars"         : 20,
    "max_downslope_pct"         : -99.0, # effectively disabled

    # ── Filters ───────────────────────────────────────────────
    "min_avg_volume"            : 80_000,
    "min_price"                 : 0.5,

    "batch_size"                : 50,
    "batch_sleep"               : 1.5,
}

# ── Indicators ───────────────────────────────────────────────
def calc_ema(series, period):
    return series.ewm(span=period, adjust=False).mean()

def calc_rsi(close, period=14):
    d  = close.diff()
    g  = d.clip(lower=0); l = -d.clip(upper=0)
    ag = g.ewm(alpha=1/period, adjust=False).mean()
    al = l.ewm(alpha=1/period, adjust=False).mean()
    rs = ag / al.replace(0, np.nan)
    return 100 - (100 / (1 + rs))

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

# ── Core detection ────────────────────────────────────────────
def detect_pattern(sym, df):
    df       = df.copy(); df.index = pd.to_datetime(df.index)
    n        = len(df)
    price    = float(df["Close"].iloc[-1])
    avg_vol  = float(df["Volume"].tail(CFG["vol_avg_bars"]).mean())

    if price   < CFG["min_price"]:      return None
    if avg_vol < CFG["min_avg_volume"]: return None

    # ── Compute MAs ───────────────────────────────────────────
    sma50_s  = df["Close"].rolling(CFG["sma50_period"]).mean()
    sma150_s = df["Close"].rolling(CFG["sma150_period"]).mean()
    ema20_s  = calc_ema(df["Close"], CFG["ema20_period"])
    rsi_s    = calc_rsi(df["Close"])

    cur_sma50  = float(sma50_s.iloc[-1])
    cur_sma150 = float(sma150_s.iloc[-1])
    cur_ema20  = float(ema20_s.iloc[-1])
    cur_rsi    = float(rsi_s.iloc[-1]) if not np.isnan(rsi_s.iloc[-1]) else 50

    if any(np.isnan([cur_sma50, cur_sma150, cur_ema20])): return None
    if cur_rsi < CFG["rsi_min"]: return None

    # ─────────────────────────────────────────────────────────
    # PHASE 2: SMA150 breakout (check first — fastest gate)
    # Most recent bar must have closed ABOVE SMA150
    # AND in the cross_lookback bars, find the exact cross point
    # ─────────────────────────────────────────────────────────
    if price <= cur_sma150: return None   # still below — no breakout yet

    # Find the exact cross bar (most recent close < SMA150 → close >= SMA150)
    cl = CFG["cross_lookback"]
    cross_bar  = None
    cross_date = None

    for i in range(max(1, n - cl), n):
        pc   = float(df["Close"].iloc[i-1])
        cc   = float(df["Close"].iloc[i])
        ps150= float(sma150_s.iloc[i-1]) if not np.isnan(sma150_s.iloc[i-1]) else np.nan
        cs150= float(sma150_s.iloc[i])   if not np.isnan(sma150_s.iloc[i])   else np.nan
        if np.isnan(ps150) or np.isnan(cs150): continue
        if pc < ps150 and cc >= cs150:
            cross_bar  = i
            cross_date = df.index[i]

    if cross_bar is None: return None

    bars_since_cross = n - 1 - cross_bar

    # Confirm pre-cross bars were also below SMA150
    pb = CFG["pre_cross_below_bars"]
    for i in range(max(0, cross_bar - pb), cross_bar):
        c_i   = float(df["Close"].iloc[i])
        s150_i= float(sma150_s.iloc[i]) if not np.isnan(sma150_s.iloc[i]) else cur_sma150
        if c_i >= s150_i: return None   # was above before cross — false signal

    # ─────────────────────────────────────────────────────────
    # PHASE 1: Below SMA150 — count consecutive bars
    # AXON pattern: price below SMA150 only (may be above SMA50)
    # Count going backwards from cross_bar-1
    # ─────────────────────────────────────────────────────────
    bear_bars   = 0
    bear_start  = cross_bar - 1

    for i in range(bear_start, -1, -1):
        c_i   = float(df["Close"].iloc[i])
        s150_i= float(sma150_s.iloc[i]) if not np.isnan(sma150_s.iloc[i]) else np.nan
        if np.isnan(s150_i): break
        # Only SMA150 check — price can be above SMA50 (AXON pattern)
        if c_i < s150_i:
            bear_bars += 1
        else:
            break   # first bar above SMA150 = end of streak

    if bear_bars < CFG["min_bear_bars"]: return None

    bear_months = round(bear_bars / 21, 1)

    # ─────────────────────────────────────────────────────────
    # PHASE 2b: Volume on breakout bar
    # ─────────────────────────────────────────────────────────
    cross_vol  = float(df["Volume"].iloc[cross_bar])
    vol_mult   = cross_vol / avg_vol if avg_vol > 0 else 0
    if vol_mult < CFG["min_vol_mult"]: return None

    # ─────────────────────────────────────────────────────────
    # PHASE 3: Quality checks
    # ─────────────────────────────────────────────────────────
    # Price not too far above SMA150 (not chasing)
    dist_sma150_pct = (price - cur_sma150) / cur_sma150 * 100
    if dist_sma150_pct > CFG["max_above_sma150_pct"]: return None

    # SMA150 slope — should not be steeply declining
    sb = CFG["sma150_slope_bars"]
    sma150_prev = float(sma150_s.iloc[-sb-1]) if not np.isnan(sma150_s.iloc[-sb-1]) else cur_sma150
    sma150_slope_pct = (cur_sma150 - sma150_prev) / sma150_prev * 100 if sma150_prev > 0 else 0
    if sma150_slope_pct < CFG["max_downslope_pct"]: return None

    # ─────────────────────────────────────────────────────────
    # TIER CLASSIFICATION  (by duration below SMA150 + volume)
    # ─────────────────────────────────────────────────────────
    if bear_bars >= CFG["tier1_bars"] and vol_mult >= CFG["tier1_vol_mult"]:
        tier       = 1
        tier_label = "🏆 TIER 1 — 7mo+ Below SMA150 + Vol ≥1.5×"
    elif bear_bars >= CFG["tier2_bars"] and vol_mult >= CFG["tier2_vol_mult"]:
        tier       = 2
        tier_label = "🥈 TIER 2 — 5mo+ Below SMA150 + Vol ≥1.2×"
    else:
        tier       = 3
        tier_label = "🥉 TIER 3 — 3mo+ Below SMA150 + Vol ≥0.8×"

    # Price vs SMA50 (next resistance level)
    dist_sma50_pct  = (price - cur_sma50) / cur_sma50 * 100
    # Price vs EMA20
    dist_ema20_pct  = (price - cur_ema20) / cur_ema20 * 100

    # Bear low (lowest close during bear period)
    bear_window = df["Close"].iloc[max(0, cross_bar - bear_bars) : cross_bar]
    bear_low    = round(float(bear_window.min()), 2) if len(bear_window) > 0 else price
    recovery_pct= round((price - bear_low) / bear_low * 100, 2) if bear_low > 0 else 0

    # ── Score (0-100) ─────────────────────────────────────────
    score = 0

    # Bear duration bonus (0-30): longer = bigger potential
    score += min(30, int(bear_months * 2.0))

    # Volume surge quality (0-25)
    score += min(25, int(vol_mult * 7))

    # SMA150 slope (0-15): flat/rising = better
    score += max(0, min(15, int((sma150_slope_pct + 3) * 3)))

    # Distance from SMA150 (0-15): closer = fresher cross
    score += max(0, 15 - int(dist_sma150_pct * 2))

    # RSI momentum (0-10)
    score += min(10, max(0, int((cur_rsi - 30) / 5)))

    # SMA50 proximity bonus (0-5): price near SMA50 = breaking more resistance
    score += 5 if abs(dist_sma50_pct) <= 5 else 0

    score = min(100, max(0, score))

    return {
        "Ticker"             : sym,
        "Price"              : round(price, 2),
        "Score"              : score,
        "Tier"               : tier,
        "Tier_Label"         : tier_label,

        # Bear period
        "Bear_Bars"          : bear_bars,
        "Bear_Months"        : bear_months,
        "Bear_Low"           : bear_low,
        "Recovery_%"         : recovery_pct,

        # Breakout
        "Break_Date"         : cross_date.strftime("%Y-%m-%d"),
        "Bars_Since_Break"   : bars_since_cross,
        "Break_Vol_x"        : round(vol_mult, 2),
        "Break_Dist_SMA150_%" : round(dist_sma150_pct, 2),

        # MAs
        "EMA20"              : round(cur_ema20,  2),
        "SMA50"              : round(cur_sma50,  2),
        "SMA150"             : round(cur_sma150, 2),
        "SMA150_Slope_%"     : round(sma150_slope_pct, 3),
        "Dist_SMA50_%"       : round(dist_sma50_pct, 2),
        "Dist_SMA150_%"      : round(dist_sma150_pct, 2),
        "Dist_EMA20_%"       : round(dist_ema20_pct, 2),

        # Indicators
        "RSI"                : round(cur_rsi, 1),
        "Avg_Vol_50d"        : int(avg_vol),

        # Internals
        "_df"                : df,
        "_sma50"             : sma50_s,
        "_sma150"            : sma150_s,
        "_ema20"             : ema20_s,
        "_cross_bar"         : cross_bar,
        "_bear_start"        : bear_start,
        "_bear_bars"         : bear_bars,
    }

# ── Live print ────────────────────────────────────────────────
LIVE_COLS = [
    "Ticker","Price","Score","Tier_Label",
    "Bear_Months","Bear_Low","Recovery_%",
    "Break_Date","Bars_Since_Break","Break_Vol_x","Break_Dist_SMA150_%",
    "SMA50","SMA150","SMA150_Slope_%","RSI",
]
_CW = {
    "Ticker":8,"Price":10,"Score":7,"Tier_Label":30,
    "Bear_Months":12,"Bear_Low":10,"Recovery_%":12,
    "Break_Date":12,"Bars_Since_Break":16,"Break_Vol_x":12,
    "Break_Dist_SMA150_%":19,
    "SMA50":9,"SMA150":9,"SMA150_Slope_%":14,"RSI":6,
}
_CF = {
    "Price"               : "${:.2f}",
    "Score"               : "{:.0f}",
    "Bear_Months"         : "{:.1f}mo",
    "Bear_Low"            : "${:.2f}",
    "Recovery_%"          : "{:+.1f}%",
    "Bars_Since_Break"    : "{:.0f}d",
    "Break_Vol_x"         : "{:.2f}×",
    "Break_Dist_SMA150_%"  : "{:+.2f}%",
    "SMA50"               : "${:.2f}",
    "SMA150"              : "${:.2f}",
    "SMA150_Slope_%"      : "{:+.3f}%",
    "RSI"                 : "{:.1f}",
}
_hdr_done = False

def _live_header():
    global _hdr_done
    if _hdr_done: return
    sep = "━" * 210
    print(f"\n{sep}")
    print("  📊  LIVE MATCHES  —  7-Month Bear → SMA150 Breakout")
    print(sep)
    print("".join(f"  {c:<{_CW.get(c,10)}}" for c in LIVE_COLS))
    print("  " + "─" * 208)
    _hdr_done = True

def live_print(r):
    _live_header()
    row = ""
    for c in LIVE_COLS:
        val=r.get(c,"—"); w=_CW.get(c,10); fmt=_CF.get(c)
        try:   s=fmt.format(val) if (fmt and val not in("—",None)) else str(val)
        except: s=str(val)
        row += f"  {s:<{w}}"
    print(row)

# ── Health check ──────────────────────────────────────────────
print("━"*65)
print("  STEP 1  DATA CHECK")
print("━"*65)
chk = download(["AAPL","NVDA","MARA"], 550)
if not chk: print("❌  No data.")
else:
    for s, d in chk.items():
        p    = float(d["Close"].iloc[-1])
        s50  = float(d["Close"].rolling(50).mean().iloc[-1])
        s150 = float(d["Close"].rolling(150).mean().iloc[-1])
        print(f"  ✅ {s}: ${p:.2f}  SMA50=${s50:.2f}  SMA150=${s150:.2f}  "
              f"P>S150:{'✅' if p>s150 else '❌'}  {d.index[-1].date()}")
print()

# ── Diagnostic ────────────────────────────────────────────────
print("━"*65)
print("  STEP 2  DIAGNOSTIC (10 sample stocks)")
print("━"*65+"\n")

DIAG = ["MARA","COIN","HOOD","RIOT","SOFI","UPST","PLTR","IONQ","AFRM","SNAP"]
diag_data = download(DIAG, CFG["history_days"])
print(f"  Downloaded {len(diag_data)}/{len(DIAG)}\n")
print(f"  {'SYM':<7} {'PRICE':>8}  {'>S150':>6}  "
      f"{'BEAR_MO':>8}  {'VOL×':>6}  {'SCORE':>6}  RESULT")
print("  " + "─" * 58)

for sym in DIAG:
    if sym not in diag_data:
        print(f"  {sym:<7} — no data"); continue
    try:
        df_d = diag_data[sym]
        p    = float(df_d["Close"].iloc[-1])
        s150 = float(df_d["Close"].rolling(150).mean().iloc[-1])
        t    = lambda b: "✅" if b else "❌"
        r    = detect_pattern(sym, df_d)
        if r:
            print(f"  {sym:<7} ${p:>7.2f}  {t(p>s150):>6}  "
                  f"{r['Bear_Months']:>7.1f}mo  "
                  f"{r['Break_Vol_x']:>5.1f}×  "
                  f"{r['Score']:>6}  ✅ {r['Tier_Label']}")
        else:
            print(f"  {sym:<7} ${p:>7.2f}  {t(p>s150):>6}  "
                  f"{'—':>8}  {'—':>6}  {'—':>6}  ❌")
    except Exception as e:
        print(f"  {sym:<7} error: {e}")

print(f"""
  Pattern:
    Phase 1  Extended bear: close < SMA50 AND < SMA150 for >= {CFG['min_bear_bars']} bars
             (~{CFG['min_bear_bars']//21} months)  consecutive
    Phase 2  SMA150 breakout: previous bar below, current bar closed above
             Volume >= {CFG['min_vol_mult']}× 50d average on breakout bar
    Phase 3  Not too extended (price < SMA150 + {CFG['max_above_sma150_pct']}%)
             SMA150 slope >= {CFG['max_downslope_pct']}% (not steeply falling)

  Tiers:
    🏆 12+ months bear  + vol >= 2.5×  (biggest spring)
    🥈  9+ months bear  + vol >= 2.0×
    🥉  7+ months bear  + vol >= 1.5×

  Tune if mostly ❌:
    min_bear_bars     63 → 42  (2 months instead of 3)
    min_vol_mult     0.8 → 0.5 (any volume is fine)
    max_above_sma150  50 → 80  (allow very extended moves)
    cross_lookback    15 → 20
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
            b  = len(pool)
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
        t_   = {row["symbol"].strip() for row in rows
                if row.get("symbol","").strip().isalpha()
                and 1<=len(row["symbol"].strip())<=5}
        b = len(pool); pool |= t_
        print(f"  ✅ {'NASDAQ API':<18}: +{len(pool)-b:>4} → {len(pool)}")
    except Exception as e: print(f"  ⚠️  NASDAQ API: {e}")
    static = {
        "AAPL","MSFT","NVDA","AMZN","META","GOOGL","GOOG","TSLA","AVGO","COST",
        "NFLX","AMD","INTC","CSCO","ADBE","QCOM","TXN","AMAT","MU","KLAC",
        "LRCX","MRVL","MELI","PANW","CRWD","SNPS","CDNS","TEAM","WDAY","PLTR",
        "MARA","COIN","HOOD","RIOT","SOFI","UPST","AFRM","DDOG","SNOW","RBRK",
        "IONQ","QUBT","RGTI","ASTS","RKLB","FSLR","PYPL","ROKU","SNAP","PINS",
        "AMGN","GILD","INTU","MCHP","MNST","NXPI","XEL","ACLS","IRTC","SMCI",
    }
    b = len(pool); pool |= static
    print(f"  ✅ {'Static fallback':<18}: +{len(pool)-b:>4} → {len(pool)}")
    clean = sorted({s.upper() for s in pool if isinstance(s,str)
                    and s.isalpha() and 1<=len(s)<=5})
    print(f"\n  🎯 Total: {len(clean)} tickers")
    return clean

TICKERS = get_tickers(); print()

# ── Main scan ─────────────────────────────────────────────────
print("━"*65)
print(f"  STEP 4  SCANNING {len(TICKERS)} TICKERS")
print("━"*65+"\n")

_hdr_done = False; results = []; no_data = 0
batches = [TICKERS[i:i+CFG["batch_size"]] for i in range(0,len(TICKERS),CFG["batch_size"])]

with tqdm(total=len(TICKERS), desc="Scanning", unit="stk",
          bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]") as pbar:
    for batch in batches:
        data_map = download(batch, CFG["history_days"])
        no_data += len(batch) - len(data_map)
        for sym in batch:
            pbar.update(1)
            if sym not in data_map: continue
            try:
                r = detect_pattern(sym, data_map[sym])
                if r: results.append(r); live_print(r)
            except Exception: pass
        time.sleep(CFG["batch_sleep"])

got = len(TICKERS) - no_data; pct = got/max(len(TICKERS),1)*100
print(f"\n{'━'*65}")
print(f"  SCAN COMPLETE | {len(TICKERS)} tickers | {got} ({pct:.0f}%) | ✅ {len(results)} matches")
print(f"{'━'*65}")

# ── Results ───────────────────────────────────────────────────
if not results:
    print("\n  No matches. Try relaxing:")
    print("   min_bear_bars     63 → 42  (2 months)")
    print("   min_vol_mult     0.8 → 0.5")
    print("   max_above_sma150  50 → 80")
    print("   cross_lookback    15 → 20")

# Sort by tier first, then score
results.sort(key=lambda x: (x["Tier"], -x["Score"]))

# ── Always save and email (even if 0 results) ─────────────────
ts      = datetime.today().strftime("%Y%m%d_%H%M")
out_dir = os.environ.get("GITHUB_WORKSPACE", os.getcwd())

COLS = [
    "Ticker","Price","Score","Tier","Tier_Label",
    "Bear_Bars","Bear_Months","Bear_Low","Recovery_%",
    "Break_Date","Bars_Since_Break","Break_Vol_x","Break_Dist_SMA150_%",
    "EMA20","SMA50","SMA150","SMA150_Slope_%",
    "Dist_EMA20_%","Dist_SMA50_%","Dist_SMA150_%",
    "RSI","Avg_Vol_50d",
]
df_out = pd.DataFrame([{k:v for k,v in r.items() if not k.startswith("_")}
                        for r in results]) if results else pd.DataFrame(columns=COLS)
if not df_out.empty:
    df_out = df_out[[c for c in COLS if c in df_out.columns]]
    df_out.reset_index(drop=True, inplace=True)

# ── Format helpers ────────────────────────────────────────────
FMT = {
    "Price"                : lambda v: f"${v:.2f}",
    "Score"                : lambda v: f"{v:.0f}",
    "Bear_Months"          : lambda v: f"{v:.1f}mo",
    "Bear_Low"             : lambda v: f"${v:.2f}",
    "Recovery_%"           : lambda v: f"{v:+.1f}%",
    "Break_Vol_x"          : lambda v: f"{v:.2f}×",
    "Break_Dist_SMA150_%"  : lambda v: f"{v:+.2f}%",
    "EMA20"                : lambda v: f"${v:.2f}",
    "SMA50"                : lambda v: f"${v:.2f}",
    "SMA150"               : lambda v: f"${v:.2f}",
    "SMA150_Slope_%"       : lambda v: f"{v:+.3f}%",
    "Dist_EMA20_%"         : lambda v: f"{v:+.2f}%",
    "Dist_SMA50_%"         : lambda v: f"{v:+.2f}%",
    "Dist_SMA150_%"        : lambda v: f"{v:+.2f}%",
    "Bars_Since_Break"     : lambda v: f"{int(v)}d",
    "Bear_Bars"            : lambda v: f"{int(v)}",
    "RSI"                  : lambda v: f"{v:.1f}",
    "Avg_Vol_50d"          : lambda v: f"{v:,.0f}",
}

def fmt_v(col, val):
    if val is None or (isinstance(val, float) and np.isnan(val)): return "—"
    try:
        if col in FMT: return FMT[col](val)
    except Exception: pass
    return str(val) if str(val) not in ("nan","None","") else "—"

# ── Display ───────────────────────────────────────────────────
TIER_COLORS = {1:"#22c55e", 2:"#3b82f6", 3:"#f59e0b"}
TIER_ICONS  = {1:"🏆", 2:"🥈", 3:"🥉"}
TIER_NAMES  = {
    1:"TIER 1 — 12mo+ Bear + Vol ≥2.5×",
    2:"TIER 2 — 9mo+ Bear + Vol ≥2.0×",
    3:"TIER 3 — 7mo+ Bear + Vol ≥1.5×",
}

DISP_COLS = [
    "Ticker","Price","Score",
    "Bear_Months","Bear_Low","Recovery_%",
    "Break_Date","Bars_Since_Break","Break_Vol_x","Break_Dist_SMA150_%",
    "SMA50","SMA150","SMA150_Slope_%","Dist_SMA50_%","RSI",
]
DISP_COLS = [c for c in DISP_COLS if c in df_out.columns or not results]

if _IN_NOTEBOOK and results:
    # ── Rich HTML table grouped by tier ──────────────────────
    def make_tier_block(tier_rows, tier_num):
        tc   = TIER_COLORS[tier_num]
        icon = TIER_ICONS[tier_num]
        name = TIER_NAMES[tier_num]
        if not tier_rows: return ""

        th = "".join(
            f'<th style="background:#0f172a;color:#e2e8f0;padding:9px 12px;'
            f'font-size:11px;font-weight:700;text-align:center;'
            f'border-bottom:2px solid {tc};white-space:nowrap">{c}</th>'
            for c in DISP_COLS
        )
        rows_html = ""
        for i, r in enumerate(tier_rows):
            bg = "#ffffff" if i % 2 == 0 else "#f0f9ff"
            tds = ""
            for col in DISP_COLS:
                raw  = r.get(col)
                disp = fmt_v(col, raw)
                sty  = ""
                if col == "Score":
                    try:
                        v = float(raw)
                        g = int(min(220, 80 + v * 1.4))
                        sty = (f"background:rgb(20,{g},60);color:#fff;"
                               f"font-weight:700;text-align:center")
                    except Exception: pass
                elif col == "Bear_Months":
                    try:
                        v = float(raw)
                        if v >= 12: sty = "color:#ef4444;font-weight:700"
                        elif v >= 9: sty = "color:#f59e0b;font-weight:600"
                        else:        sty = "color:#fbbf24"
                    except Exception: pass
                elif col == "Break_Vol_x":
                    try:
                        v = float(str(raw).replace("×",""))
                        if v >= 3:   sty = "color:#f59e0b;font-weight:700"
                        elif v >= 2: sty = "color:#fbbf24;font-weight:600"
                    except Exception: pass
                elif col in ("Break_Dist_SMA150_%","Dist_SMA50_%",
                             "Recovery_%","SMA150_Slope_%"):
                    try:
                        v = float(str(raw).replace("%","").replace("+",""))
                        clr = "#22c55e" if v >= 0 else "#ef4444"
                        sty = f"color:{clr};font-weight:600"
                    except Exception: pass
                elif col == "Bars_Since_Break":
                    try:
                        v = int(float(str(raw).replace("d","")))
                        if v == 0: sty = "color:#22c55e;font-weight:700;text-align:center"
                        elif v <= 1: sty = "color:#86efac;text-align:center"
                    except Exception: pass
                tds += (f'<td style="padding:7px 12px;font-size:12px;'
                        f'border-bottom:1px solid #e2e8f0;white-space:nowrap;{sty}">'
                        f'{disp}</td>')
            rows_html += f'<tr style="background:{bg}">{tds}</tr>\n'

        return f"""
<div style="margin:12px 0">
  <div style="background:linear-gradient(90deg,{tc}22,#0f172a);
              border-left:4px solid {tc};border-radius:6px 6px 0 0;
              padding:10px 18px;display:flex;align-items:center;gap:10px">
    <span style="font-size:20px">{icon}</span>
    <div>
      <span style="color:#f1f5f9;font-size:15px;font-weight:700">{name}</span>
      <span style="color:{tc};font-size:12px;margin-left:12px;font-weight:600">
        {len(tier_rows)} stock{'s' if len(tier_rows)!=1 else ''}
      </span>
    </div>
  </div>
  <div style="overflow-x:auto;border:1px solid #e2e8f0;border-top:none;
              border-radius:0 0 8px 8px;box-shadow:0 2px 8px rgba(0,0,0,0.05)">
    <table style="border-collapse:collapse;width:100%;min-width:700px">
      <thead><tr>{th}</tr></thead>
      <tbody>{rows_html}</tbody>
    </table>
  </div>
</div>"""

    t1_cnt = sum(1 for r in results if r["Tier"]==1)
    t2_cnt = sum(1 for r in results if r["Tier"]==2)
    t3_cnt = sum(1 for r in results if r["Tier"]==3)

    header_html = f"""
<div style="background:linear-gradient(135deg,#0f172a,#1e3a5f);
            border-radius:10px;padding:18px 24px;margin-bottom:8px;
            font-family:'Segoe UI',Arial,sans-serif">
  <h2 style="margin:0;color:#60a5fa;font-size:20px;font-weight:700">
    📈 7-Month Bear → SMA150 Breakout Scanner
  </h2>
  <p style="margin:6px 0 0;color:#94a3b8;font-size:12px">
    {datetime.today().strftime('%Y-%m-%d %H:%M')} &nbsp;·&nbsp;
    <span style="color:#22c55e;font-weight:700">{len(results)} matches</span>
    from {len(TICKERS)} tickers
    &nbsp;·&nbsp; 🏆{t1_cnt} &nbsp; 🥈{t2_cnt} &nbsp; 🥉{t3_cnt}
  </p>
</div>"""

    tier_blocks = "".join(
        make_tier_block([r for r in results if r["Tier"]==t], t)
        for t in [1,2,3]
    )

    legend = """
<div style="background:#f8fafc;border:1px solid #e2e8f0;border-radius:8px;
            padding:12px 18px;margin-top:8px;font-size:11px;color:#64748b">
  <b style="color:#475569">GUIDE</b> &nbsp;·&nbsp;
  Bear_Months = how long price was below BOTH SMA50 and SMA150 &nbsp;·&nbsp;
  Bear_Low = lowest close during the bear &nbsp;·&nbsp;
  Recovery_% = how much price recovered from the bear low &nbsp;·&nbsp;
  Break_Vol_x = volume on breakout bar vs 50d avg &nbsp;·&nbsp;
  Bars_Since_Break: <span style="color:#22c55e">0=today</span>
</div>"""

    display_html(header_html + tier_blocks + legend)

elif results:
    # ASCII table for CLI/GitHub
    CLI_COLS = ["Ticker","Price","Score","Bear_Months","Bear_Low",
                "Break_Date","Bars_Since_Break","Break_Vol_x",
                "Break_Dist_SMA150_%","SMA150_Slope_%","RSI"]
    CLI_COLS = [c for c in CLI_COLS if c in df_out.columns]

    for t_num in [1,2,3]:
        t_rows = [r for r in results if r["Tier"]==t_num]
        if not t_rows: continue
        print(f"\n  {TIER_ICONS[t_num]}  {TIER_NAMES[t_num]}  ({len(t_rows)} stocks)\n")

        col_w = {c: max(len(c), max(
            len(fmt_v(c, r.get(c))) for r in t_rows
        )) + 2 for c in CLI_COLS}
        top  = "┬".join("─"*col_w[c] for c in CLI_COLS)
        sep  = "┼".join("─"*col_w[c] for c in CLI_COLS)
        bot  = "┴".join("─"*col_w[c] for c in CLI_COLS)
        hdr  = "│".join(c.center(col_w[c]) for c in CLI_COLS)
        print(f"  ┌{top}┐")
        print(f"  │{hdr}│")
        print(f"  ├{sep}┤")
        for i, r in enumerate(t_rows):
            cells = [fmt_v(c,r.get(c)).center(col_w[c]) for c in CLI_COLS]
            print(f"  │{'│'.join(cells)}│")
            if i < len(t_rows)-1: print(f"  ├{sep}┤")
        print(f"  └{bot}┘")

    print(f"""
  COLUMN KEY
  ─────────────────────────────────────────────────────────
  Bear_Months    consecutive months price was below SMA50+SMA150
  Bear_Low       lowest price during the bear period
  Break_Vol_x    breakout bar volume vs 50-day average
  Break_Dist_%   how far above SMA150 at breakout (lower = fresher)
  SMA150_Slope_% SMA150 direction (positive = turning up)
  ─────────────────────────────────────────────────────────""")

# ── Save CSV ──────────────────────────────────────────────────
fpath = os.path.join(out_dir, f"bear_sma150_breakout_{ts}.csv")
df_out.to_csv(fpath, index=False)
print(f"\n  💾 CSV → {fpath}")

tv = os.path.join(out_dir, f"tv_bear_breakout_{ts}.txt")
with open(tv, "w") as f:
    f.write(f"###7mo Bear SMA150 Breakout {datetime.today().strftime('%Y-%m-%d')}\n")
    for r in results:
        f.write(f"NASDAQ:{r['Ticker']}\n")
print(f"  📋 TradingView → {tv}")

# ── Email with CSV attached (same hardened pattern as sma_retest) ──
def _send_email(rl, csv_path):
    gu = _GMAIL_USER; gp = _GMAIL_PASS; et = _EMAIL_TO

    if not gu:
        print("[Email] ❌  GMAIL_USER secret is empty")
        print("         → Repo → Settings → Secrets → Actions → GMAIL_USER")
        return
    if not gp:
        print("[Email] ❌  GMAIL_PASS secret is empty")
        print("         → Must be Gmail App Password (16 chars)")
        print("         → myaccount.google.com/apppasswords")
        return
    if not et:
        print("[Email] ❌  EMAIL_TO secret is empty")
        print("         → Repo → Settings → Secrets → Actions → EMAIL_TO")
        return

    eto  = [e.strip() for e in et.split(",") if e.strip()]
    cnt  = len(rl)
    t1   = sum(1 for r in rl if r["Tier"]==1)
    t2   = sum(1 for r in rl if r["Tier"]==2)
    t3   = sum(1 for r in rl if r["Tier"]==3)

    print(f"[Email] Sending to {et}  ({cnt} results)...")

    SHOW = ["Ticker","Price","Score","Bear_Months","Bear_Low","Recovery_%",
            "Break_Date","Bars_Since_Break","Break_Vol_x",
            "Break_Dist_SMA150_%","SMA150_Slope_%","RSI"]

    th_e = "".join(
        f'<th style="background:#1e293b;color:#e2e8f0;padding:8px 11px;'
        f'font-size:11px;font-weight:700;border-bottom:2px solid #3b82f6;'
        f'white-space:nowrap">{c}</th>' for c in SHOW
    )

    rows_e = ""
    for i, r in enumerate(rl[:50]):
        bg  = "#fff" if i%2==0 else "#f0f9ff"
        tc  = TIER_COLORS.get(r["Tier"],"#64748b")
        tds = "".join(
            f'<td style="padding:6px 11px;font-size:11px;'
            f'border-bottom:1px solid #e2e8f0;white-space:nowrap;'
            f'{"color:"+tc+";font-weight:700" if c=="Score" else ""}">'
            f'{fmt_v(c,r.get(c))}</td>'
            for c in SHOW
        )
        rows_e += f'<tr style="background:{bg}">{tds}</tr>\n'

    no_results_msg = ""
    if cnt == 0:
        no_results_msg = (
            f'<tr><td colspan="{len(SHOW)}" style="padding:20px;text-align:center;'
            f'color:#64748b;font-size:13px">No matches found today — '
            f'no stock emerging from a 7-month bear today</td></tr>'
        )

    html_e = f"""<!DOCTYPE html><html><body style="margin:0;padding:0;
background:#f1f5f9;font-family:'Segoe UI',Arial,sans-serif">
<table width="100%" cellpadding="0" cellspacing="0" style="padding:20px 0">
<tr><td>
<table width="100%" cellpadding="0" cellspacing="0"
       style="max-width:960px;margin:0 auto;background:#fff;border-radius:12px;
              overflow:hidden;box-shadow:0 4px 20px rgba(0,0,0,0.08)">
  <tr><td style="background:linear-gradient(135deg,#0f172a,#1e3a5f);padding:22px 28px">
    <h1 style="margin:0;color:#60a5fa;font-size:20px;font-weight:700">
      📈 7-Month Bear → SMA150 Breakout
    </h1>
    <p style="margin:6px 0 0;color:#94a3b8;font-size:12px">
      {datetime.today().strftime('%Y-%m-%d %H:%M UTC')} &nbsp;·&nbsp;
      {cnt} match{'es' if cnt!=1 else ''} &nbsp;·&nbsp;
      🏆 Tier1:{t1} &nbsp; 🥈 Tier2:{t2} &nbsp; 🥉 Tier3:{t3}
    </p>
  </td></tr>
  <tr><td style="padding:16px">
    <div style="overflow-x:auto;border-radius:8px;border:1px solid #e2e8f0">
      <table style="border-collapse:collapse;width:100%;min-width:700px">
        <thead><tr>{th_e}</tr></thead>
        <tbody>{rows_e or no_results_msg}</tbody>
      </table>
    </div>
    <p style="font-size:11px;color:#64748b;margin:8px 0 0">
      📎 Full results attached as CSV &nbsp;·&nbsp;
      Bear_Months = months price was below SMA50+SMA150 &nbsp;·&nbsp;
      Break_Vol_x = volume vs 50d avg
    </p>
  </td></tr>
  <tr><td style="background:#f8fafc;padding:12px 28px;
                 border-top:1px solid #e2e8f0;text-align:center">
    <p style="margin:0;color:#94a3b8;font-size:10px">
      ⚠️ Not financial advice &nbsp;·&nbsp; Auto-generated by GitHub Actions
    </p>
  </td></tr>
</table></td></tr></table>
</body></html>"""

    plain_e = (
        f"7-Month Bear → SMA150 Breakout — {datetime.today().strftime('%Y-%m-%d')}\n"
        f"{cnt} matches  (🏆T1:{t1}  🥈T2:{t2}  🥉T3:{t3})\n"
        + "="*60 + "\n"
        + ("\n".join(
            f"{r['Ticker']:<7} ${r.get('Price',0):.2f}  "
            f"Score:{r.get('Score',0):.0f}  "
            f"Bear:{r.get('Bear_Months',0):.1f}mo  "
            f"Vol:{r.get('Break_Vol_x',0):.1f}×  "
            f"Break:{r.get('Break_Date','—')}"
            for r in rl[:50]
          ) if rl else "No matches today")
        + "\n\nFull results in CSV attachment. Not financial advice."
    )

    subj = (f"📈 Bear→SMA150 Break — {cnt} signal{'s' if cnt!=1 else ''}"
            f"  (🏆{t1} 🥈{t2} 🥉{t3}) — "
            f"{datetime.today().strftime('%Y-%m-%d')}")

    msg = MIMEMultipart("mixed")
    msg["Subject"] = subj; msg["From"] = gu; msg["To"] = ", ".join(eto)
    alt = MIMEMultipart("alternative")
    alt.attach(MIMEText(plain_e,"plain")); alt.attach(MIMEText(html_e,"html"))
    msg.attach(alt)

    if csv_path and os.path.exists(csv_path):
        try:
            with open(csv_path,"rb") as f:
                part = MIMEBase("application","octet-stream")
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
            srv.login(gu, gp.replace(" ",""))
            srv.sendmail(gu, eto, msg.as_string())
        print(f"[Email] ✅  Sent to: {', '.join(eto)}")
        print(f"[Email]    Subject : {subj}")
    except smtplib.SMTPAuthenticationError:
        print("[Email] ❌  AUTHENTICATION FAILED")
        print("         Use Gmail App Password, NOT your login password")
        print("         Generate: myaccount.google.com/apppasswords")
    except smtplib.SMTPException as e:
        print(f"[Email] ❌  SMTP error: {e}")
    except Exception as e:
        print(f"[Email] ❌  {type(e).__name__}: {e}")

_send_email(results, fpath)

if _IN_NOTEBOOK:
    try:
        from google.colab import files
        files.download(fpath); files.download(tv)
    except Exception: pass
else:
    print("  (CI: files saved to workspace, email sent)")

# ── Charts for top 5 ──────────────────────────────────────────
if results:
    top  = results[:min(5, len(results))]
    rows = len(top)
    fig, axes = plt.subplots(rows, 1, figsize=(15, 5.5*rows), facecolor="#0f172a")
    if rows == 1: axes = [axes]

    for idx, r in enumerate(top):
        ax = axes[idx]
        ax.set_facecolor("#0f172a")

        df_p   = r["_df"].tail(120).copy()   # show more history for bear period
        sma50  = r["_sma50"].reindex(df_p.index)
        sma150 = r["_sma150"].reindex(df_p.index)
        ema20  = r["_ema20"].reindex(df_p.index)
        n_p    = len(df_p)
        fn     = len(r["_df"]); off = fn - n_p

        # Candlestick
        for i, (_, row_) in enumerate(df_p.iterrows()):
            o=float(row_["Open"]); h=float(row_["High"])
            l=float(row_["Low"]);  c=float(row_["Close"])
            clr = "#34d399" if c >= o else "#ef4444"
            ax.plot([i,i],[l,h], color=clr, lw=0.6, zorder=2)
            blo=min(o,c); bhi=max(o,c); bh=max(bhi-blo,(h-l)*0.005)
            rect=mpatches.FancyBboxPatch((i-0.3,blo),0.6,bh,
                 boxstyle="square,pad=0",facecolor=clr,edgecolor=clr,lw=0.4,zorder=3)
            ax.add_patch(rect)

        # MA lines
        ax.plot(range(n_p), ema20.values,
                color="#34d399", lw=1.4, ls="--", label="EMA20", zorder=5)
        ax.plot(range(n_p), sma50.values,
                color="#3b82f6", lw=1.5, ls="-",  label="SMA50 🔵", zorder=4)
        ax.plot(range(n_p), sma150.values,
                color="#f472b6", lw=2.0, ls="-",  label="SMA150 🔴", zorder=4)

        # Shade bear zone
        cb  = r["_cross_bar"] - off
        bbs = max(0, cb - min(r["_bear_bars"], n_p))
        if bbs < cb:
            ax.axvspan(bbs, cb, alpha=0.07, color="#ef4444", zorder=1,
                       label=f"Bear zone ({r['Bear_Months']:.1f}mo)")

        # Mark breakout bar
        if 0 <= cb < n_p:
            ax.axvline(cb, color="#22c55e", lw=2.0, ls="--", alpha=0.9)
            ax.scatter([cb],[float(df_p["Close"].iloc[cb])],
                       color="#22c55e", s=200, zorder=9, marker="^",
                       label=f"SMA150 Break {r['Break_Date']} Vol {r['Break_Vol_x']:.1f}×")

        tick_step = max(1, n_p//8)
        ax.set_xticks(range(0, n_p, tick_step))
        ax.set_xticklabels(
            [df_p.index[i].strftime("%m/%d") for i in range(0, n_p, tick_step)],
            color="#94a3b8", fontsize=7)
        ax.set_xlim(-0.5, n_p-0.5)
        ax.set_title(
            f"{r['Ticker']}  ${r['Price']:.2f}  |  Score {r['Score']}/100  |  "
            f"{r['Tier_Label']}  |  "
            f"Bear: {r['Bear_Months']:.1f}mo  Low ${r['Bear_Low']:.2f}  "
            f"Recovery {r['Recovery_%']:+.1f}%  |  "
            f"Break: {r['Break_Date']} ({r['Bars_Since_Break']}d ago)  "
            f"Vol {r['Break_Vol_x']:.1f}×  |  RSI {r['RSI']:.0f}",
            color="#e2e8f0", fontsize=8, fontweight="bold", pad=7)
        ax.tick_params(colors="#94a3b8", labelsize=7)
        for sp_ in ax.spines.values(): sp_.set_edgecolor("#1e3a5f")
        ax.legend(loc="upper left", facecolor="#1e293b",
                  labelcolor="#e2e8f0", fontsize=7, framealpha=0.9, ncol=2)
        ax.grid(color="#1e3a5f", ls="--", lw=0.4, alpha=0.4, axis="y")

    plt.suptitle(
        f"7-Month Bear → SMA150 Breakout  ·  {datetime.today().strftime('%Y-%m-%d')}\n"
        f"🔴 SMA150  🔵 SMA50  🟢 EMA20  🟥 Bear zone  ▲ Breakout bar",
        color="#60a5fa", fontsize=10, fontweight="bold", y=1.001)
    plt.tight_layout()
    cp = os.path.join(out_dir, f"bear_sma150_breakout_chart_{ts}.png")
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

  PHASE 1 — EXTENDED BEAR (7+ months)
    Close < SMA50  AND  Close < SMA150  for 147+ trading days
    = Stock was in deep downtrend or base-building mode
    = Bigger the bear, bigger the potential spring

  PHASE 2 — SMA150 BREAKOUT WITH VOLUME
    Yesterday: close < SMA150  (still below)
    Today:     close >= SMA150  (first close above!)
    Volume on today's bar >= 1.5× 50-day average
    = Institutions stepped in to support the move

  PHASE 3 — QUALITY FILTERS
    SMA150 slope not steeply declining (>= -3%)
    Price not too extended (< 15% above SMA150)
    RSI >= 30 (not exhausted)

  RESULT TIERS
    🏆 Tier 1  12mo+ bear + Vol ≥2.5×  (massive coil released)
    🥈 Tier 2   9mo+ bear + Vol ≥2.0×  (strong setup)
    🥉 Tier 3   7mo+ bear + Vol ≥1.5×  (confirmed pattern)

  💡 BEST SETUPS
  Bear_Months > 12       bigger compression = bigger potential
  Break_Vol_x > 3×       huge institutional buying
  SMA150_Slope_% > 0    SMA150 turning up = structural change
  Break_Dist_%   < 5%   fresh cross, not extended
  Bars_Since_Break = 0   catching it on the actual breakout day

  ⚙️  TUNE IF 0 RESULTS
  min_bear_bars    63 → 42  (2 months)
  min_vol_mult    0.8 → 0.5
  max_above_sma150 50 → 80
  cross_lookback   15 → 20
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
""")
