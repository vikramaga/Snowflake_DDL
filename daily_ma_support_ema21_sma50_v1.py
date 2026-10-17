# ============================================================
# NASDAQ — Daily MA Support: EMA21 / SMA50 (v1)
# ============================================================
#
# IDEA: find stocks that have been taking support on the same moving
# average — EMA21 or SMA50 — most of the time over the last ~2 months,
# and are back near that MA today (a pullback to proven support).
#
# For EACH of EMA21 and SMA50, over the last 42 trading days before
# today (support_window_days):
#   • BOUNCE: a "touch" day has its LOW within 1.5% above the MA or
#     below it (touch_pct), while the CLOSE holds no more than 1% below
#     the MA (hold_tol_pct). Touch days within 3 bars of each other are
#     one event (min_touch_gap). The event counts as a bounce if price
#     then CLOSES at least 2% above the MA within 5 days (bounce_pct,
#     bounce_days).
#   • HELD %: share of days whose close is no more than 1% below the MA.
# The SUPPORT MA is the one with more bounces (tie → higher Held %;
# tie → EMA21).
#
# SIGNAL (all required, on the most recent daily candle):
#   1. SUPPORT FOUND: the support MA has >= 3 bounces in the window.
#   2. HELD MOST OF THE TIME: Held % on that MA >= 80%.
#   3. RISING MA: that MA is higher than 10 days ago.
#   4. NEAR IT NOW: today's close is between 1% below and 3% above
#      that same MA.
#
# DATA — only 1 download per ticker: daily bars.
#
# OUTPUT: Entry_Price and Stop_Loss are reasonable defaults (Entry =
# today's close; Stop = 2% below the support MA), not explicitly
# requested. Quantity / Total_Amount come from position_sizing.py
# (max loss $40 per trade).
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
    "ema_period"             : 21,
    "sma_period"             : 50,

    # ── Support analysis (last ~2 months) ───────────────────────
    "support_window_days"    : 42,   # trading days before today (~2 months)
    "touch_pct"              : 1.5,  # low within this % ABOVE the MA (or below it) = touch
    "hold_tol_pct"           : 1.0,  # close may be at most this % BELOW the MA and still "hold"
    "min_touch_gap"          : 3,    # touch days closer than this are one event
    "bounce_pct"             : 2.0,  # after a touch, a close this % above the MA ...
    "bounce_days"            : 5,    # ... within this many days = a bounce (support worked)
    "min_bounces"            : 3,    # step 1
    "min_held_ratio"         : 0.80, # step 2: share of window closes held vs the MA
    "slope_days"             : 10,   # step 3: MA today > MA this many days ago
    "require_rising_ma"      : True,
    "near_below_pct"         : 1.0,  # step 4: today's close from 1% below ...
    "near_above_pct"         : 3.0,  # ... to 3% above the support MA

    "stop_buffer_pct"        : 2.0,  # Stop = support MA x (1 - 2%)
    "signal_lookback_days"   : 1,    # 1 = the setup must be on the most recent candle

    # ── Filters ─────────────────────────────────────────────────
    "min_avg_volume"         : 100_000,
    "min_price"              : 10.0,

    "batch_size"             : 50,
    "batch_sleep"            : 1.5,
}

# ── Indicators ───────────────────────────────────────────────────────
def calc_ema(close, n):
    return close.ewm(span=n, adjust=False).mean()

def calc_sma(close, n):
    return close.rolling(n).mean()

def ma_support_stats(close, low, ma, i, cfg):
    """
    Support statistics of `ma` over the window [i - W, i - 1] (the days
    BEFORE bar i). Returns dict with bounces, held_ratio, touch events and
    the bar of the last bounce. Bounce confirmation may use closes up to
    bar i (today) — today itself is not a touch candidate.
    """
    W   = cfg["support_window_days"]
    s, e = i - W, i                       # window = s .. e-1
    tp  = cfg["touch_pct"] / 100.0
    ht  = cfg["hold_tol_pct"] / 100.0
    bp  = cfg["bounce_pct"] / 100.0
    c, l, m = close.values, low.values, ma.values

    held = 0
    touch_bars = []
    for t in range(s, e):
        if np.isnan(m[t]):
            continue
        holds = c[t] >= m[t] * (1 - ht)
        held += holds
        if holds and l[t] <= m[t] * (1 + tp):
            touch_bars.append(t)

    # group touch days into events
    events = []
    for t in touch_bars:
        if events and t - events[-1][1] <= cfg["min_touch_gap"]:
            events[-1][1] = t
        else:
            events.append([t, t])

    bounces, last_bounce = 0, None
    for a, b in events:
        hi = min(b + cfg["bounce_days"], i)
        if any(c[k] >= m[k] * (1 + bp) for k in range(b + 1, hi + 1) if not np.isnan(m[k])):
            bounces += 1
            last_bounce = b
    return {"bounces": bounces, "held_ratio": held / W, "touch_events": len(events),
            "last_bounce": last_bounce}

def check_pattern_at(close, low, ema, sma, i, cfg):
    """
    Checks the MA-support setup with "today" at bar `i`.
    SINGLE SOURCE OF TRUTH for the pattern logic.

    Computes EVERY stage's pass/fail regardless of earlier failures
    (guarding only on missing data), so callers can diagnose exactly
    which stage is filtering results out — see FUNNEL_COUNTS below.

    Returns (passed: bool, details: dict).
    """
    W = cfg["support_window_days"]
    if i - W - cfg["slope_days"] < 0 or i >= len(close):
        return False, {}
    if np.isnan(sma.iloc[i - W]) or np.isnan(ema.iloc[i - W]):
        return False, {}

    st = {"EMA21": ma_support_stats(close, low, ema, i, cfg),
          "SMA50": ma_support_stats(close, low, sma, i, cfg)}
    # support MA: more bounces, then higher held ratio, then EMA21
    name = max(["EMA21", "SMA50"],
               key=lambda k: (st[k]["bounces"], st[k]["held_ratio"], k == "EMA21"))
    ma   = ema if name == "EMA21" else sma
    s    = st[name]
    m_now  = float(ma.iloc[i])
    m_prev = float(ma.iloc[i - cfg["slope_days"]])
    c_now  = float(close.iloc[i])
    dist   = (c_now / m_now - 1) * 100

    found   = s["bounces"] >= cfg["min_bounces"]                                  # Step 1
    held    = s["held_ratio"] >= cfg["min_held_ratio"]                            # Step 2
    rising  = (m_now > m_prev) or not cfg["require_rising_ma"]                     # Step 3
    near    = -cfg["near_below_pct"] <= dist <= cfg["near_above_pct"]              # Step 4

    passed = found and held and rising and near
    return passed, {
        "idx": i, "ma_name": name, "ma": m_now, "ma_prev": m_prev, "dist": dist,
        "bounces": s["bounces"], "held_ratio": s["held_ratio"],
        "touch_events": s["touch_events"], "last_bounce": s["last_bounce"],
        "stats": st,
        "found": found, "held": held, "rising": rising, "near": near,
    }

# ── Diagnostic funnel — tallies how far each ticker-day gets through
#    the pattern, across the full universe scan, so a 0-match run can
#    be diagnosed empirically instead of guessed at ─────────────────
FUNNEL_COUNTS = {
    "days_checked": 0,
    "passed_step1_support_found": 0,
    "passed_step2_held_most_of_time": 0,
    "passed_step3_rising_ma": 0,
    "passed_step4_near_ma_now": 0,
}

def find_ma_support(close, low, ema, sma, cfg):
    """
    Checks the last `signal_lookback_days` trading days (1 = the most
    recent candle only). Returns hit dicts, most recent first, and
    tallies FUNNEL_COUNTS for every day checked.
    """
    global FUNNEL_COUNTS
    n = len(close)
    hits = []
    for back in range(0, cfg["signal_lookback_days"]):
        i = (n - 1) - back
        passed, d = check_pattern_at(close, low, ema, sma, i, cfg)
        if not d:
            continue
        FUNNEL_COUNTS["days_checked"] += 1
        if d["found"]:
            FUNNEL_COUNTS["passed_step1_support_found"] += 1
            if d["held"]:
                FUNNEL_COUNTS["passed_step2_held_most_of_time"] += 1
                if d["rising"]:
                    FUNNEL_COUNTS["passed_step3_rising_ma"] += 1
                    if d["near"]:
                        FUNNEL_COUNTS["passed_step4_near_ma_now"] += 1
        if passed:
            hits.append(d)
    return hits

# ── Technical signal: daily MA support ───────────────────────────
def analyze_ma_support(sym, df_daily):
    """
    Returns dict with score and setup details, or None if the setup
    is not present today.
    """
    if df_daily is None:
        return None

    price   = float(df_daily["Close"].iloc[-1])
    avg_vol = float(df_daily["Volume"].tail(20).mean())
    if price   < CFG["min_price"]:      return None
    if avg_vol < CFG["min_avg_volume"]: return None

    n = len(df_daily)
    if n < CFG["sma_period"] + CFG["support_window_days"] + CFG["slope_days"] + 10:
        return None

    close = df_daily["Close"]
    low   = df_daily["Low"]
    ema   = calc_ema(close, CFG["ema_period"])
    sma   = calc_sma(close, CFG["sma_period"])

    hits = find_ma_support(close, low, ema, sma, CFG)
    if not hits:
        return None

    s = hits[0]   # most recent
    k = s["idx"]
    days_since_signal = (n - 1) - k

    # ── Entry / Stop (reasonable defaults — see header note) ─────────
    entry_price = float(close.iloc[k])
    stop_loss   = s["ma"] * (1 - CFG["stop_buffer_pct"] / 100.0)
    if entry_price <= stop_loss:
        return None
    risk_pct = (entry_price - stop_loss) / entry_price * 100

    slope_pct = (s["ma"] / s["ma_prev"] - 1) * 100
    vol_k  = float(df_daily["Volume"].iloc[k])
    avg20  = float(df_daily["Volume"].iloc[max(0, k-20):k].mean())
    vol_vs_avg = vol_k / avg20 if avg20 > 0 else 0.0
    other = "SMA50" if s["ma_name"] == "EMA21" else "EMA21"
    st_o  = s["stats"][other]

    # ── Score (0-100) ────────────────────────────────────────────
    score = 20   # base for clearing every gate
    reasons = [f"Support {s['ma_name']}"]
    score += max(0, min(20, (s["bounces"] - CFG["min_bounces"] + 1) * 5))
    reasons.append(f"{s['bounces']} bounces")
    score += max(0, min(20, (s["held_ratio"] - CFG["min_held_ratio"])
                         / (1 - CFG["min_held_ratio"]) * 20))
    reasons.append(f"Held {s['held_ratio']*100:.0f}%")
    score += max(0, 20 * (1 - abs(s["dist"]) / CFG["near_above_pct"]))
    reasons.append(f"{s['dist']:+.1f}% from MA")
    score += max(0, min(10, slope_pct * 2))
    reasons.append(f"MA slope {slope_pct:+.1f}%/{CFG['slope_days']}d")
    if vol_vs_avg < 1.0:
        score += 10; reasons.append(f"Quiet pullback (vol x{vol_vs_avg:.2f})")
    else:
        reasons.append(f"Vol x{vol_vs_avg:.2f}")
    if st_o["bounces"] >= CFG["min_bounces"]:
        reasons.append(f"{other} also {st_o['bounces']} bounces")
    if risk_pct > 8:
        score -= 10; reasons.append("WideStop")
    score = round(min(100, max(0, score)))

    lb = s["last_bounce"]
    return {
        "Score"          : score,
        "Price"          : round(price, 2),
        "Entry_Price"    : round(entry_price, 2),
        "Stop_Loss"      : round(stop_loss, 2),
        "Risk_%"         : round(risk_pct, 1),
        "Support_MA"     : s["ma_name"],
        "MA_Value"       : round(s["ma"], 2),
        "Dist_to_MA_%"   : round(s["dist"], 2),
        "Bounces"        : s["bounces"],
        "Held_%"         : round(s["held_ratio"] * 100, 1),
        "EMA21_Bounces"  : s["stats"]["EMA21"]["bounces"],
        "EMA21_Held_%"   : round(s["stats"]["EMA21"]["held_ratio"] * 100, 1),
        "SMA50_Bounces"  : s["stats"]["SMA50"]["bounces"],
        "SMA50_Held_%"   : round(s["stats"]["SMA50"]["held_ratio"] * 100, 1),
        "MA_Slope_%"     : round(slope_pct, 2),
        "Last_Bounce"    : df_daily.index[lb].strftime("%Y-%m-%d") if lb is not None else "",
        "Vol_vs_20d_Avg" : round(vol_vs_avg, 2),
        "Signal_Date"    : df_daily.index[k].strftime("%Y-%m-%d"),
        "Days_Since_Signal": days_since_signal,
        "Flags"          : " | ".join(reasons),
        "_df_daily"      : df_daily,
        "_ema"           : ema,
        "_sma"           : sma,
        "_idx"           : k,
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
             "Support_MA","Bounces","Held_%","Dist_to_MA_%"]
_CW = {"Ticker":8,"Price":10,"Score":7,"Entry_Price":12,"Stop_Loss":11,
       "Support_MA":11,"Bounces":8,"Held_%":8,"Dist_to_MA_%":12}
_CF = {"Price":"${:.2f}","Score":"{:.0f}","Entry_Price":"${:.2f}",
       "Stop_Loss":"${:.2f}","Held_%":"{:.0f}%","Dist_to_MA_%":"{:+.1f}%"}
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
print(f"  Signal: price has taken support on EMA{CFG['ema_period']} or SMA{CFG['sma_period']} "
      f"(>= {CFG['min_bounces']} bounces, held >= {CFG['min_held_ratio']*100:.0f}% of the last "
      f"{CFG['support_window_days']} days), MA rising, and today's close is near that MA\n")

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

for sym in tqdm(list(daily_map.keys()), desc="Checking MA support", unit="stk"):
    try:
        r = analyze_ma_support(sym, daily_map[sym])
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
print(f"  Ticker-days checked                                : {fc['days_checked']}")
print(f"  Step 1 — support MA with >= {CFG['min_bounces']} bounces              : {fc['passed_step1_support_found']}")
print(f"  Step 2 — + held >= {CFG['min_held_ratio']*100:.0f}% of the last {CFG['support_window_days']} days       : {fc['passed_step2_held_most_of_time']}")
print(f"  Step 3 — + MA rising over {CFG['slope_days']} days                  : {fc['passed_step3_rising_ma']}")
print(f"  Step 4 — + close {-CFG['near_below_pct']:+.0f}% to +{CFG['near_above_pct']:.0f}% from that MA today   : {fc['passed_step4_near_ma_now']}  (= full pattern)")
print(f"{'━'*65}")

if not results:
    print("\n  No matches. Try relaxing (see the FUNNEL above to see which")
    print("  step is actually the bottleneck before guessing):")
    print("   min_bounces                     3 → 2")
    print("   min_held_ratio               0.80 → 0.70")
    print("   near_above_pct                3.0 → 4.0")
    print("   min_price                      10 → 5")

results.sort(key=lambda x: x["Score"], reverse=True)

# ── Always build df_out and save/email (even if 0 results) ────
ts      = datetime.today().strftime("%Y%m%d_%H%M")
out_dir = os.environ.get("GITHUB_WORKSPACE", os.getcwd())

COLS = [
    "Ticker","Price","Score",
    "Entry_Price","Stop_Loss","Risk_%",
    "Support_MA","MA_Value","Dist_to_MA_%","Bounces","Held_%",
    "EMA21_Bounces","EMA21_Held_%","SMA50_Bounces","SMA50_Held_%",
    "MA_Slope_%","Last_Bounce","Vol_vs_20d_Avg",
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
    "MA_Value"      : lambda v: f"${v:.2f}",
    "Risk_%"        : lambda v: f"{v:.1f}%",
    "Dist_to_MA_%"  : lambda v: f"{v:+.1f}%",
    "Held_%"        : lambda v: f"{v:.0f}%",
    "MA_Slope_%"    : lambda v: f"{v:+.1f}%",
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
             "Support_MA","MA_Value","Dist_to_MA_%","Bounces","Held_%","Last_Bounce"]

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
    📈 Daily MA Support (EMA21 / SMA50)
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
    tit = f"  Daily MA Support (EMA21/SMA50)   {datetime.today().strftime('%Y-%m-%d')}   {len(df_out)} matches"
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
  Score           0-100 (bounces + held % + closeness to the MA + MA slope
                  + quiet-volume pullback)
  Support_MA      the MA price has respected most (EMA21 or SMA50)
  Bounces         confirmed bounces off that MA in the last {CFG['support_window_days']} days
  Held_%          share of those days that closed no more than {CFG['hold_tol_pct']:.0f}% below it
  Dist_to_MA_%    today's close vs that MA
  Entry_Price     today's close
  Stop_Loss       {CFG['stop_buffer_pct']:.0f}% below the support MA
  ──────────────────────────────────────────────────────""")

# Save
fpath = os.path.join(out_dir, f"daily_ma_support_{ts}.csv")
# ── Position sizing: Quantity / Total_Amount for a fixed max loss (see position_sizing.py)
try:
    from position_sizing import add_position_size
    df_out = add_position_size(df_out)
except Exception as _ps_err:
    print(f"  ⚠️  Position sizing skipped: {_ps_err}")
df_out.to_csv(fpath, index=False)
print(f"\n  💾 CSV → {fpath}")
tv = os.path.join(out_dir, f"tv_daily_ma_support_{ts}.txt")
with open(tv,"w") as f:
    f.write(f"###Daily MA Support EMA21-SMA50 {datetime.today().strftime('%Y-%m-%d')}\n")
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

        heads = ["Ticker","Price","Score","Entry","Stop","Risk %","Support",
                 "MA","vs MA","Bounces","Held","Qty","Amount"]
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
                + td(r.get("Support_MA","—"), "font-weight:700;color:#7c3aed")
                + td(f'${float(r.get("MA_Value",0) or 0):.2f}')
                + td(f'{float(r.get("Dist_to_MA_%",0) or 0):+.1f}%')
                + td(f'{r.get("Bounces","—")}')
                + td(f'{float(r.get("Held_%",0) or 0):.0f}%')
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
  📊 Daily MA Support — EMA21 / SMA50
</h1>
<p style="margin:6px 0 0;color:#94a3b8;font-size:12px">
  {datetime.today().strftime('%Y-%m-%d %H:%M UTC')} &nbsp;·&nbsp;
  {cnt} match{'es' if cnt!=1 else ''} — price took support on the same MA (EMA{CFG['ema_period']} or
  SMA{CFG['sma_period']}) most of the last {CFG['support_window_days']} days and is back near it today
</p>
  </td></tr>
  <tr><td style="padding:14px 28px 4px;background:#0b1220">
<div style="background:#111827;border:1px solid #1f2937;border-radius:8px;padding:12px 16px">
  <p style="margin:0 0 6px;color:#93c5fd;font-size:12px;font-weight:700">
    🔍 FUNNEL — last {CFG['signal_lookback_days']} trading day(s), every liquid ticker
  </p>
  <p style="margin:0;color:#cbd5e1;font-size:12px">
    {fc['days_checked']} ticker-days &nbsp;→&nbsp;
    {fc['passed_step1_support_found']} support MA with &ge;{CFG['min_bounces']} bounces &nbsp;→&nbsp;
    {fc['passed_step2_held_most_of_time']} held &ge;{CFG['min_held_ratio']*100:.0f}% &nbsp;→&nbsp;
    {fc['passed_step3_rising_ma']} MA rising &nbsp;→&nbsp;
    <b style="color:#facc15">{fc['passed_step4_near_ma_now']} near the MA today = full match</b>
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
  ⚠️ Not financial advice &nbsp;·&nbsp; Entry = today's close, Stop = {CFG['stop_buffer_pct']:.0f}% below the support MA (defaults) &nbsp;·&nbsp; Qty sized for a $40 max loss &nbsp;·&nbsp; Auto-generated by GitHub Actions
</p>
  </td></tr>
</table>
</td></tr></table>
</body></html>"""

        plain_lines = [
            f"Daily MA Support (EMA21/SMA50) — {datetime.today().strftime('%Y-%m-%d')}",
            f"{cnt} matches (support on the same MA most of the last {CFG['support_window_days']} days; near it today)",
            "="*60,
            f"FUNNEL: {fc['days_checked']} ticker-days -> "
            f"{fc['passed_step1_support_found']} support MA >={CFG['min_bounces']} bounces -> "
            f"{fc['passed_step2_held_most_of_time']} held >={CFG['min_held_ratio']*100:.0f}% -> "
            f"{fc['passed_step3_rising_ma']} MA rising -> "
            f"{fc['passed_step4_near_ma_now']} near MA today (=full match)",
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
                    f"{r.get('Support_MA','—')}:${float(r.get('MA_Value',0) or 0):.2f} "
                    f"({float(r.get('Dist_to_MA_%',0) or 0):+.1f}%)  "
                    f"Bounces:{r.get('Bounces','—')}  Held:{float(r.get('Held_%',0) or 0):.0f}%  "
                    f"Qty:{q}  Amount:{'$%s' % format(float(a), ',.0f') if a is not None else '—'}")
            plain_lines.append("")
            plain_lines.append("Tickers (comma-separated):")
            plain_lines.append(", ".join(r.get("Ticker","") for r in rl))
        else:
            plain_lines.append("No matches today")
        plain_lines.append("\nFull results in CSV attachment.")
        plain_e = "\n".join(plain_lines)

        subj = (f"📊 MA Support EMA21/SMA50 — {cnt} signal{'s' if cnt!=1 else ''}"
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

# ── Charts for top 5: price with EMA21 and SMA50, the support MA
#    highlighted, the support window shaded and bounce days marked ────
if results:
    top = results[:min(5,len(results))]
    fig, axes = plt.subplots(len(top), 1, figsize=(13, 3.6*len(top)), facecolor="#0f172a",
                              squeeze=False)
    axes = axes[:,0]
    for p, r in enumerate(top):
        ax = axes[p]
        df_all = r["_df_daily"]
        df_p = df_all.tail(90)
        idx  = df_p.index
        k = r["_idx"]
        W = CFG["support_window_days"]
        sup_is_ema = r["Support_MA"] == "EMA21"

        ax.set_facecolor("#0f172a")
        ax.plot(idx, df_p["Close"], color="#60a5fa", lw=1.5, label="Close", zorder=5)
        ax.plot(idx, r["_ema"].reindex(idx), color="#22c55e",
                lw=2.0 if sup_is_ema else 0.9, ls="-" if sup_is_ema else ":", label="EMA21")
        ax.plot(idx, r["_sma"].reindex(idx), color="#fbbf24",
                lw=2.0 if not sup_is_ema else 0.9, ls="-" if not sup_is_ema else ":", label="SMA50")
        ax.axhline(r["Stop_Loss"], color="#ef4444", lw=1.0, ls="--", alpha=0.85,
                   label=f"Stop {r['Stop_Loss']:.2f}")
        w0 = df_all.index[max(0, k - W)]
        ax.axvspan(max(w0, idx[0]), df_all.index[k], color="#1e3a5f", alpha=0.35, lw=0)
        ma_s = r["_ema"] if sup_is_ema else r["_sma"]
        tp = CFG["touch_pct"] / 100.0
        touch = [d for d in idx if df_all.index.get_loc(d) < k and d >= w0
                 and df_all.loc[d, "Low"] <= ma_s.loc[d] * (1 + tp)
                 and df_all.loc[d, "Close"] >= ma_s.loc[d] * (1 - CFG["hold_tol_pct"]/100.0)]
        if touch:
            ax.scatter(touch, [df_all.loc[d, "Low"] for d in touch], color="#a78bfa", s=14,
                       zorder=6, label="Touches")
        sig_d = df_all.index[k]
        if sig_d in idx:
            ax.scatter([sig_d], [df_all["Close"].iloc[k]], color="#facc15", s=50, zorder=7, label="Today")
        ax.set_title(f"{r['Ticker']}  |  {r['Price']:.2f} USD  |  Score {r['Score']}  |  "
                     f"{r['Support_MA']} support: {r['Bounces']} bounces, held {r['Held_%']:.0f}%  |  "
                     f"{r['Dist_to_MA_%']:+.1f}% from MA",
                     color="#e2e8f0", fontsize=9, fontweight="bold", pad=7)
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))
        ax.tick_params(colors="#94a3b8", labelsize=8)
        for sp in ax.spines.values(): sp.set_edgecolor("#1e3a5f")
        ax.legend(loc="upper left", facecolor="#1e293b", labelcolor="#e2e8f0", fontsize=6, ncol=6)
        ax.grid(color="#1e3a5f", ls="--", lw=0.5, alpha=0.6)
    plt.suptitle(f"Daily MA Support (EMA21 / SMA50)  ·  {datetime.today().strftime('%Y-%m-%d')}",
                 color="#60a5fa", fontsize=12, fontweight="bold", y=1.001)
    plt.tight_layout()
    cp = os.path.join(out_dir, f"daily_ma_support_chart_{ts}.png")
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
  📋 SIGNAL (all required, on the latest daily candle)
  For EMA{CFG['ema_period']} and SMA{CFG['sma_period']}, over the last {CFG['support_window_days']} trading days:
   BOUNCE = low within {CFG['touch_pct']}% above the MA (or below it), close no more
            than {CFG['hold_tol_pct']}% below it, then a close {CFG['bounce_pct']}%+ above the MA within
            {CFG['bounce_days']} days (touches within {CFG['min_touch_gap']} bars = one event)
   HELD % = share of days closing no more than {CFG['hold_tol_pct']}% below the MA
  Support MA = the one with more bounces (tie: higher Held %).

  1) Support MA has >= {CFG['min_bounces']} bounces.
  2) Held % >= {CFG['min_held_ratio']*100:.0f}%.
  3) That MA is rising (higher than {CFG['slope_days']} days ago).
  4) Today's close is {CFG['near_below_pct']}% below to {CFG['near_above_pct']}% above that MA.

  📋 OUTPUT (reasonable defaults — not explicitly requested)
  Entry_Price = today's CLOSE
  Stop_Loss   = {CFG['stop_buffer_pct']}% below the support MA
  Quantity / Total_Amount = sized for a $40 max loss (position_sizing.py)
  Universe    = price >= 10 USD, 20d avg volume >= 100k

  📋 SCORE (0-100)
  Base 20 + bounces (0-20) + Held % above the minimum (0-20) + closeness
  to the MA (0-20) + MA slope (0-10) + quiet-volume pullback (10);
  -10 if stop risk > 8%

  ⚙️  TUNE IF 0 RESULTS
  min_bounces                     3 → 2
  min_held_ratio               0.80 → 0.70
  near_above_pct                3.0 → 4.0
  min_price                      10 → 5
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
""")
