# ============================================================
# NASDAQ — Weekly Red Doji Below JMA → Green Reclaim (v1)
# ============================================================
#
# Two-candle WEEKLY pattern: last week a small red doji closed below
# the weekly JMA; this week a green candle closes back above the JMA
# and above last week's high, with weekly RSI sitting near its own
# average (not stretched).
#
# SIGNAL (all required; checked on the last signal_lookback_weeks
# weekly candles as the "current" week; the most recent hit wins):
#
#   1. PREVIOUS WEEK SMALL RED DOJI: closed below its open, body
#      <= doji_max_body_range_pct % of its high-low range AND body
#      <= doji_max_body_atr x weekly ATR(10).
#
#   2. PREVIOUS WEEK BELOW JMA: its close is below the weekly
#      JMA(13, 40, 2) (TradingView Jurik MA formula).
#
#   3. CURRENT WEEK GREEN ABOVE JMA: closes above its open and above
#      the weekly JMA.
#
#   4. ABOVE PREVIOUS HIGH: current week's close > previous week's high.
#
#   5. RSI NEAR AVERAGE: weekly RSI(14) within rsi_near_pts points of
#      its 14-week SMA (TradingView's RSI-based MA).
#
# DATA — one download per ticker (~4 years of daily bars), resampled
# to weekly ('W'). A week still in progress is evaluated on its
# week-to-date candle and flagged Week_Complete = No.
#
# OUTPUT: Entry_Price = current week's close; Stop_Loss = the lower of
# the two weeks' lows (the doji low in practice). The doji, RSI
# "near" and stop definitions are reasonable defaults, not
# explicitly requested.
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
    # history_days vs _clean(min_bars=400): 1500 calendar days -> ~1070
    # business days (checked with pd.bdate_range), well above 400. ~200
    # weekly bars give the weekly JMA, RSI and RSI average time to settle.
    "history_days"            : 1500,

    # ── Weekly JMA (TradingView Jurik MA inputs) ─────────────────
    "jma_period"              : 13,
    "jma_phase"               : 40,
    "jma_power"               : 2,

    # ── Previous week: small red doji (default definition) ──────
    "doji_max_body_range_pct" : 35.0,  # body <= 35% of the week's high-low range
    "doji_max_body_atr"       : 0.5,   # and body <= 0.5 x weekly ATR(10) (a SMALL candle)
    "atr_period"              : 10,

    # ── Weekly RSI near its average ──────────────────────────────
    "rsi_period"              : 14,
    "rsi_avg_period"          : 14,    # SMA of the weekly RSI (TradingView's RSI-based MA default)
    "rsi_near_pts"            : 5.0,   # |RSI - RSI average| <= 5 points

    "signal_lookback_weeks"   : 2,     # current week + last completed week

    # ── Filters ─────────────────────────────────────────────────
    "min_avg_volume"          : 200_000,
    "min_price"               : 5.0,

    "batch_size"              : 50,
    "batch_sleep"             : 1.5,
}

# ── Indicators ───────────────────────────────────────────────
def resample_ohlcv(df, rule):
    """Resample daily OHLCV to a coarser timeframe ('W' weekly)."""
    agg = {"Open": "first", "High": "max", "Low": "min",
           "Close": "last", "Volume": "sum"}
    out = df.resample(rule).agg(agg)
    out.dropna(subset=["Close"], inplace=True)
    return out

def calc_jma(series, period=13, phase=40, power=2):
    """
    Jurik Moving Average — same formula as TradingView's "Jurik Moving
    Average" indicator, so values match the chart:
        beta       = 0.45*(L-1) / (0.45*(L-1) + 2)
        alpha      = beta ** power
        phaseRatio = phase/100 + 1.5   (0.5 if phase < -100, 2.5 if > 100)
        e0  = (1-alpha)*src + alpha*e0[1]
        e1  = (src - e0)*(1-beta) + beta*e1[1]
        e2  = (e0 + phaseRatio*e1 - jma[1])*(1-alpha)^2 + alpha^2*e2[1]
        jma = e2 + jma[1]
    period = Length, phase = Phase, power = Power (TradingView inputs).
    """
    phase_ratio = 0.5 if phase < -100 else (2.5 if phase > 100 else phase / 100.0 + 1.5)
    beta  = 0.45 * (period - 1) / (0.45 * (period - 1) + 2)
    alpha = beta ** power
    vals  = np.asarray(series.values, dtype=float)
    out   = np.full(len(vals), np.nan)
    e0 = e1 = e2 = jma = 0.0
    for i, v in enumerate(vals):
        if np.isnan(v):
            continue
        e0  = (1 - alpha) * v + alpha * e0
        e1  = (v - e0) * (1 - beta) + beta * e1
        e2  = (e0 + phase_ratio * e1 - jma) * (1 - alpha) ** 2 + alpha ** 2 * e2
        jma = e2 + jma
        out[i] = jma
    return pd.Series(out, index=series.index)

def calc_rsi(close, period=14):
    """Standard Wilder-smoothed RSI."""
    delta = close.diff()
    gain  = delta.clip(lower=0)
    loss  = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1/period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1/period, min_periods=period, adjust=False).mean()
    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))

def weekly_atr(df_weekly, period):
    h, l, c = df_weekly["High"], df_weekly["Low"], df_weekly["Close"]
    pc = c.shift(1)
    tr = pd.concat([h - l, (h - pc).abs(), (l - pc).abs()], axis=1).max(axis=1)
    return tr.rolling(period).mean()

def check_pattern_at(df_weekly, jma, rsi, rsi_avg, atr_w, i, cfg):
    """
    SINGLE SOURCE OF TRUTH. Checks the weekly pair (i-1 = previous week,
    i = current week). Computes EVERY step (no short-circuit) so the
    funnel shows where tickers drop out.
    Returns (passed, details); details is {} only if data is too short.
    """
    if i < cfg["atr_period"] + 2 or i >= len(df_weekly):
        return False, {}
    p, w = df_weekly.iloc[i - 1], df_weekly.iloc[i]
    po, ph, pl, pc = (float(p[k]) for k in ("Open", "High", "Low", "Close"))
    co, ch, cl, cc = (float(w[k]) for k in ("Open", "High", "Low", "Close"))
    j_prev, j_curr = float(jma.iloc[i - 1]), float(jma.iloc[i])
    r, ra = float(rsi.iloc[i]), float(rsi_avg.iloc[i])
    atr = float(atr_w.iloc[i - 1])
    if not all(np.isfinite(x) for x in (j_prev, j_curr, r, ra, atr)) or atr <= 0 or po <= 0:
        return False, {}

    # Step 1 — previous week: small red doji
    p_body = po - pc
    p_rng = ph - pl
    p_body_rng = p_body / p_rng * 100 if p_rng > 0 else 0.0
    p_body_atr = p_body / atr
    step1 = bool(p_body > 0 and p_body_rng <= cfg["doji_max_body_range_pct"]
                 and p_body_atr <= cfg["doji_max_body_atr"])

    # Step 2 — previous week closed below the weekly JMA
    step2 = bool(pc < j_prev)

    # Step 3 — current week green and closes above the weekly JMA
    step3 = bool(cc > co and cc > j_curr)

    # Step 4 — current week closes above the previous week's high
    step4 = bool(cc > ph)

    # Step 5 — weekly RSI near its own average
    step5 = bool(abs(r - ra) <= cfg["rsi_near_pts"])

    passed = step1 and step2 and step3 and step4 and step5
    return passed, {
        "week_date": df_weekly.index[i], "prev_date": df_weekly.index[i - 1],
        "prev_open": po, "prev_high": ph, "prev_low": pl, "prev_close": pc,
        "open": co, "high": ch, "low": cl, "close": cc,
        "jma_prev": j_prev, "jma_curr": j_curr, "rsi": r, "rsi_avg": ra,
        "p_body_rng": p_body_rng, "p_body_atr": p_body_atr,
        "week_pct": (cc - co) / co * 100 if co > 0 else 0.0,
        "step1": step1, "step2": step2, "step3": step3, "step4": step4, "step5": step5,
    }

# ── Diagnostic funnel (per ticker: did ANY of the lookback weeks pass
#    steps 1..k cumulatively) ──────────────────────────────────────────
FUNNEL_COUNTS = {
    "tickers_checked": 0,
    "passed_step1_prev_red_doji": 0,
    "passed_step2_prev_below_jma": 0,
    "passed_step3_curr_green_above_jma": 0,
    "passed_step4_above_prev_high": 0,
    "passed_step5_rsi_near_avg": 0,
}
_FUNNEL_KEYS = list(FUNNEL_COUNTS)[1:]

def find_signals(df_weekly, jma, rsi, rsi_avg, atr_w, cfg):
    hits, best = [], [False] * 5
    n = len(df_weekly)
    any_checked = False
    for i in range(max(0, n - cfg["signal_lookback_weeks"]), n):
        ok, d = check_pattern_at(df_weekly, jma, rsi, rsi_avg, atr_w, i, cfg)
        if not d: continue
        any_checked = True
        cum = True
        for k in range(5):
            cum = cum and d[f"step{k+1}"]
            best[k] |= cum
        if ok: hits.append(d)
    if any_checked:
        FUNNEL_COUNTS["tickers_checked"] += 1
        for k, key in enumerate(_FUNNEL_KEYS):
            FUNNEL_COUNTS[key] += best[k]
    return hits

def analyze_weekly_doji_jma_reclaim(sym, df_daily):
    if df_daily is None: return None
    price   = float(df_daily["Close"].iloc[-1])
    avg_vol = float(df_daily["Volume"].tail(20).mean())
    if price   < CFG["min_price"]:      return None
    if avg_vol < CFG["min_avg_volume"]: return None

    df_weekly = resample_ohlcv(df_daily, "W")
    if len(df_weekly) < 60:
        return None
    jma = calc_jma(df_weekly["Close"], CFG["jma_period"], CFG["jma_phase"], CFG["jma_power"])
    rsi = calc_rsi(df_weekly["Close"], CFG["rsi_period"])
    rsi_avg = rsi.rolling(CFG["rsi_avg_period"]).mean()
    atr_w = weekly_atr(df_weekly, CFG["atr_period"])

    hits = find_signals(df_weekly, jma, rsi, rsi_avg, atr_w, CFG)
    if not hits: return None
    sig = hits[-1]   # most recent hit wins

    entry = sig["close"]
    stop = min(sig["prev_low"], sig["low"])
    if entry <= stop: return None
    risk_pct = (entry - stop) / entry * 100
    weeks_ago = len(df_weekly) - 1 - df_weekly.index.get_loc(sig["week_date"])
    week_days = int(df_daily.loc[df_daily.index > sig["prev_date"]].loc[:sig["week_date"]].shape[0])
    complete = week_days >= 5 or weeks_ago > 0
    above_jma_pct = (sig["close"] - sig["jma_curr"]) / sig["jma_curr"] * 100
    above_ph_pct = (sig["close"] - sig["prev_high"]) / sig["prev_high"] * 100

    score = 30
    score += max(0, 20 - sig["p_body_rng"] * 0.5)                 # tighter doji
    score += min(20, above_ph_pct * 4)                             # decisive close above prev high
    score += max(0, 15 - abs(sig["rsi"] - sig["rsi_avg"]) * 3)     # RSI closest to its average
    score += max(0, 15 - abs(above_jma_pct - 3) * 2)               # just above JMA, not extended
    score = round(min(100, max(0, score)))

    flags = [f"Doji body {sig['p_body_rng']:.0f}% of range",
             f"Close +{above_ph_pct:.1f}% above prev high",
             f"RSI {sig['rsi']:.1f} vs avg {sig['rsi_avg']:.1f}"]
    if not complete:
        flags.append(f"Week in progress ({week_days}d)")

    return {
        "Score": score, "Price": round(price, 2),
        "Entry_Price": round(entry, 2), "Stop_Loss": round(stop, 2), "Risk_%": round(risk_pct, 1),
        "Weekly_JMA": round(sig["jma_curr"], 2), "Above_JMA_%": round(above_jma_pct, 1),
        "Prev_Week_High": round(sig["prev_high"], 2), "Above_Prev_High_%": round(above_ph_pct, 1),
        "Doji_Body_of_Range_%": round(sig["p_body_rng"], 0), "Doji_Body_xATR": round(sig["p_body_atr"], 2),
        "Weekly_RSI": round(sig["rsi"], 1), "RSI_Avg": round(sig["rsi_avg"], 1),
        "RSI_Gap": round(sig["rsi"] - sig["rsi_avg"], 1), "Week_%": round(sig["week_pct"], 1),
        "Week_Complete": "Yes" if complete else "No",
        "Signal_Date": sig["week_date"].strftime("%Y-%m-%d"), "Weeks_Since_Signal": int(weeks_ago),
        "Recent_Signal_Count": len(hits),
        "Recent_Signals": ", ".join(h["week_date"].strftime("%Y-%m-%d") for h in hits),
        "Flags": " | ".join(flags),
        "_df_weekly": df_weekly, "_jma": jma, "_rsi": rsi, "_rsi_avg": rsi_avg,
    }

# ── Download: daily (→ weekly via resample) ──────────────────────
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

# ── Main scan ─────────────────────────────────────────────────
print("━"*65)
print(f"  STEP 3  SCANNING {len(TICKERS)} TICKERS")
print("━"*65)

results, no_daily_data, daily_map = [], 0, {}
daily_batches = [TICKERS[i:i+CFG["batch_size"]]
                 for i in range(0, len(TICKERS), CFG["batch_size"])]
with tqdm(total=len(TICKERS), desc="Daily fetch", unit="stk") as pbar:
    for batch in daily_batches:
        got = download_daily(batch, CFG["history_days"])
        daily_map.update(got)
        no_daily_data += len(batch) - len(got)
        pbar.update(len(batch))
        time.sleep(CFG["batch_sleep"])
got_daily = len(TICKERS) - no_daily_data
print(f"\n  Daily data : {got_daily}/{len(TICKERS)} tickers\n")

for sym in tqdm(list(daily_map.keys()), desc="Checking pattern", unit="stk"):
    try:
        r = analyze_weekly_doji_jma_reclaim(sym, daily_map[sym])
        if r is None: continue
        r["Ticker"] = sym
        results.append(r)
        print(f"  ✅ {sym:<6} ${r['Price']:.2f}  Score {r['Score']}  RSI {r['Weekly_RSI']}/{r['RSI_Avg']}  "
              f"+{r['Above_Prev_High_%']}% over prev high  Week {r['Signal_Date']}")
    except Exception: pass

fc = FUNNEL_COUNTS
print(f"\n{'━'*65}\n  SCAN COMPLETE — {len(results)} matches from {got_daily} tickers with data")
print(f"  🔍 FUNNEL (any of the last {CFG['signal_lookback_weeks']} weekly candles)")
print(f"  Tickers checked                         : {fc['tickers_checked']}")
print(f"  Step 1 — previous week small red doji    : {fc['passed_step1_prev_red_doji']}")
print(f"  Step 2 — + previous close below JMA      : {fc['passed_step2_prev_below_jma']}")
print(f"  Step 3 — + current green, close above JMA: {fc['passed_step3_curr_green_above_jma']}")
print(f"  Step 4 — + close above previous high     : {fc['passed_step4_above_prev_high']}")
print(f"  Step 5 — + RSI within {CFG['rsi_near_pts']:.0f} pts of its avg  : {fc['passed_step5_rsi_near_avg']}  (= full pattern)")
print(f"{'━'*65}")

results.sort(key=lambda x: x["Score"], reverse=True)

# ── Save CSV + TradingView list ───────────────────────────────
ts      = datetime.today().strftime("%Y%m%d_%H%M")
out_dir = os.environ.get("GITHUB_WORKSPACE", os.getcwd())
COLS = ["Ticker","Price","Score","Entry_Price","Stop_Loss","Risk_%",
        "Weekly_JMA","Above_JMA_%","Prev_Week_High","Above_Prev_High_%",
        "Doji_Body_of_Range_%","Doji_Body_xATR","Weekly_RSI","RSI_Avg","RSI_Gap",
        "Week_%","Week_Complete","Signal_Date",
        "Weeks_Since_Signal","Recent_Signal_Count","Recent_Signals","Flags"]
df_out = pd.DataFrame([{k: v for k, v in r.items() if not k.startswith("_")}
                       for r in results]) if results else pd.DataFrame(columns=COLS)
df_out = df_out[[c for c in COLS if c in df_out.columns]].reset_index(drop=True)

fpath = os.path.join(out_dir, f"weekly_doji_jma_reclaim_{ts}.csv")
try:
    from position_sizing import add_position_size
    df_out = add_position_size(df_out)
except Exception as _ps_err:
    print(f"  ⚠️  Position sizing skipped: {_ps_err}")
df_out.to_csv(fpath, index=False)
print(f"\n  💾 CSV → {fpath}")
tv = os.path.join(out_dir, f"tv_weekly_doji_jma_reclaim_{ts}.txt")
with open(tv, "w") as f:
    f.write(f"###Weekly Doji JMA Reclaim {datetime.today().strftime('%Y-%m-%d')}\n")
    for r in results: f.write(f"NASDAQ:{r['Ticker']}\n")
print(f"  📋 TradingView → {tv}")
if results:
    print(f"\n  📋 Tickers: {', '.join(r['Ticker'] for r in results)}")

# ── Email with CSV attached ───────────────────────────────────
EMAIL_COLS = ["Ticker","Price","Score","Entry_Price","Stop_Loss","Risk_%","Weekly_JMA",
              "Above_Prev_High_%","Weekly_RSI","RSI_Avg","Signal_Date"]

def _fmt_cell(col, v):
    try:
        if col in ("Price","Entry_Price","Stop_Loss","Weekly_JMA"): return f"${float(v):.2f}"
        if col in ("Above_Prev_High_%",): return f"+{float(v):.1f}%"
        if col in ("Risk_%",): return f"{float(v):.1f}%"
        if col == "Score": return f"{float(v):.0f}"
    except Exception: pass
    return "—" if v is None else str(v)

def _send_email(rl, csv_path):
    import smtplib
    from email.mime.multipart import MIMEMultipart
    from email.mime.text      import MIMEText
    from email.mime.base      import MIMEBase
    from email                import encoders

    gu = _GMAIL_USER; gp = _GMAIL_PASS; et = _EMAIL_TO
    if not (gu and gp and et):
        print("[Email] ❌  GMAIL_USER / GMAIL_PASS / EMAIL_TO missing — skipped")
        return
    eto = [e.strip() for e in et.split(",") if e.strip()]
    cnt = len(rl)
    fc = FUNNEL_COUNTS
    try:
        th = "".join(f'<th style="background:#1e293b;color:#e2e8f0;padding:8px 10px;'
                     f'font-size:11px;border-bottom:2px solid #3b82f6;white-space:nowrap">{c}</th>'
                     for c in EMAIL_COLS)
        rows = ""
        for i, r in enumerate(rl[:50]):
            bg = "#fff" if i % 2 == 0 else "#f0f9ff"
            tds = ""
            for c in EMAIL_COLS:
                sty = "font-weight:700" if c == "Ticker" else ""
                if c == "Score": sty = "font-weight:700;background:#166534;color:#fff;text-align:center"
                if c == "Stop_Loss": sty = "color:#ef4444"
                if c == "Entry_Price": sty = "color:#16a34a"
                tds += f'<td style="padding:6px 10px;font-size:12px;{sty}">{_fmt_cell(c, r.get(c))}</td>'
            rows += f'<tr style="background:{bg}">{tds}</tr>'
        if not rows:
            rows = (f'<tr><td colspan="{len(EMAIL_COLS)}" style="padding:20px;text-align:center;'
                    f'color:#94a3b8">No matches today</td></tr>')
        tick_html = (f'<p style="margin:10px 0 0;font-size:12px;font-family:monospace">'
                     f'{", ".join(r["Ticker"] for r in rl)}</p>') if rl else ""
        html = f"""<!DOCTYPE html><html><body style="margin:0;background:#f1f5f9;font-family:'Segoe UI',Arial,sans-serif">
<table width="100%" cellpadding="0" cellspacing="0"><tr><td align="center" style="padding:20px 10px">
<table width="100%" style="max-width:860px;background:#fff;border-radius:12px;overflow:hidden">
<tr><td style="background:linear-gradient(135deg,#0f172a,#1e3a5f);padding:20px 26px">
<h1 style="margin:0;color:#60a5fa;font-size:19px">📊 Weekly Red Doji Below JMA → Green Reclaim</h1>
<p style="margin:6px 0 0;color:#94a3b8;font-size:12px">{datetime.today().strftime('%Y-%m-%d %H:%M UTC')} · {cnt} match{'es' if cnt != 1 else ''}
 — last week a small red doji below the weekly JMA, this week green, above the JMA and last week's high, RSI within {CFG['rsi_near_pts']:.0f} pts of its average</p>
</td></tr>
<tr><td style="padding:12px 26px;background:#0b1220"><p style="margin:0;color:#cbd5e1;font-size:12px">
🔍 FUNNEL: {fc['tickers_checked']} checked → {fc['passed_step1_prev_red_doji']} prev red doji
→ {fc['passed_step2_prev_below_jma']} + below JMA → {fc['passed_step3_curr_green_above_jma']} + green above JMA
→ {fc['passed_step4_above_prev_high']} + above prev high
→ <b style="color:#facc15">{fc['passed_step5_rsi_near_avg']} + RSI near avg = full match</b></p></td></tr>
<tr><td style="padding:16px"><div style="overflow-x:auto;border:1px solid #e2e8f0;border-radius:8px">
<table style="border-collapse:collapse;width:100%"><thead><tr>{th}</tr></thead><tbody>{rows}</tbody></table></div>
{tick_html}
<p style="font-size:11px;color:#64748b;margin:10px 0 0">Entry = the signal week's close · Stop = the lower of the two
weeks' lows (defaults). Week_Complete = No means the week is still in progress — confirm at the Friday close. 📎 Full results (with Quantity for a fixed max loss) attached as CSV.</p>
</td></tr>
<tr><td style="background:#f8fafc;padding:10px;text-align:center;color:#94a3b8;font-size:10px">
⚠️ Not financial advice · Auto-generated by GitHub Actions</td></tr>
</table></td></tr></table></body></html>"""

        plain = [f"Weekly Red Doji Below JMA → Green Reclaim — {datetime.today().strftime('%Y-%m-%d')}",
                 f"{cnt} matches",
                 f"FUNNEL: {fc['tickers_checked']} checked -> {fc['passed_step1_prev_red_doji']} prev red doji -> "
                 f"{fc['passed_step2_prev_below_jma']} below JMA -> {fc['passed_step3_curr_green_above_jma']} green above JMA -> "
                 f"{fc['passed_step4_above_prev_high']} above prev high -> {fc['passed_step5_rsi_near_avg']} RSI near avg (=full match)", "="*60]
        for r in rl[:50]:
            plain.append(f"{r['Ticker']:<6} ${r['Price']:.2f}  Score:{r['Score']}  Entry:${r['Entry_Price']:.2f}  "
                         f"SL:${r['Stop_Loss']:.2f}  JMA:${r['Weekly_JMA']:.2f}  RSI:{r['Weekly_RSI']}/{r['RSI_Avg']}  "
                         f"+{r['Above_Prev_High_%']}% over prev high  {r['Signal_Date']}"
                         f"{'' if r['Week_Complete'] == 'Yes' else ' (week in progress)'}")
        if rl:
            plain += ["", "Tickers: " + ", ".join(r["Ticker"] for r in rl)]
        else:
            plain.append("No matches today")
        plain.append("\nFull results in CSV attachment.")

        subj = (f"📊 Weekly Doji JMA Reclaim — "
                f"{cnt} signal{'s' if cnt != 1 else ''} — {datetime.today().strftime('%Y-%m-%d')}")
        msg = MIMEMultipart("mixed")
        msg["Subject"] = subj; msg["From"] = gu; msg["To"] = ", ".join(eto)
        alt = MIMEMultipart("alternative")
        alt.attach(MIMEText("\n".join(plain), "plain"))
        alt.attach(MIMEText(html, "html"))
        msg.attach(alt)
    except Exception as e:
        print(f"[Email] ❌  Failed to build email body: {type(e).__name__}: {e}")
        return

    if csv_path and os.path.exists(csv_path):
        try:
            with open(csv_path, "rb") as f:
                part = MIMEBase("application", "octet-stream"); part.set_payload(f.read())
            encoders.encode_base64(part)
            part.add_header("Content-Disposition", f"attachment; filename={os.path.basename(csv_path)}")
            msg.attach(part)
        except Exception as e:
            print(f"[Email] ⚠️  CSV attach failed: {e}")
    try:
        with smtplib.SMTP_SSL("smtp.gmail.com", 465) as srv:
            srv.login(gu, gp.replace(" ", ""))
            srv.sendmail(gu, eto, msg.as_string())
        print(f"[Email] ✅  Sent to {', '.join(eto)} — {subj}")
    except Exception as e:
        print(f"[Email] ❌  {type(e).__name__}: {e}")

try:
    _send_email(results, fpath)
except Exception as e:
    print(f"[Email] ❌  Unexpected top-level error: {type(e).__name__}: {e}")

# ── Charts for top 5: last 30 weekly candles + JMA, weekly RSI + its average ─
if results:
    top = results[:5]
    fig, axes = plt.subplots(len(top), 2, figsize=(16, 4.2*len(top)), facecolor="#0f172a",
                             gridspec_kw={"width_ratios": [3, 1.3]})
    if len(top) == 1: axes = axes.reshape(1, 2)
    for row, r in zip(axes, top):
        ax_w, ax_r = row
        wk = r["_df_weekly"].tail(30)
        ax_w.set_facecolor("#0f172a")
        x = np.arange(len(wk))
        cols = ["#22c55e" if c > o else "#ef4444" for o, c in zip(wk["Open"], wk["Close"])]
        ax_w.vlines(x, wk["Low"], wk["High"], color=cols, lw=1)
        ax_w.bar(x, (wk["Close"] - wk["Open"]).abs(), bottom=np.minimum(wk["Open"], wk["Close"]),
                 color=cols, width=0.6)
        sig_ts = pd.Timestamp(r["Signal_Date"])
        if sig_ts in wk.index:
            k = list(wk.index).index(sig_ts)
            ax_w.axvline(k, color="#facc15", ls="--", lw=0.8)
        ax_w.set_xticks(x[::3]); ax_w.set_xticklabels([d.strftime("%m/%d") for d in wk.index[::3]], fontsize=7)
        ax_w.plot(x, r["_jma"].reindex(wk.index).values, color="#f97316", lw=1.4, label="Weekly JMA")
        ax_w.legend(loc="upper left", facecolor="#1e293b", labelcolor="#e2e8f0", fontsize=7)
        ax_w.set_title(f"{r['Ticker']} | ${r['Price']:.2f} | Score {r['Score']} | Week {r['Week_%']:+.1f}% "
                       f"| +{r['Above_Prev_High_%']}% over prev high", color="#e2e8f0", fontsize=9, fontweight="bold")
        rs = r["_rsi"].reindex(wk.index); ra = r["_rsi_avg"].reindex(wk.index)
        ax_r.set_facecolor("#0f172a")
        ax_r.plot(rs.index, rs.values, color="#60a5fa", lw=1.3, label="RSI(14)")
        ax_r.plot(ra.index, ra.values, color="#facc15", lw=1.1, label="RSI avg")
        ax_r.legend(loc="upper left", facecolor="#1e293b", labelcolor="#e2e8f0", fontsize=6)
        ax_r.set_title(f"Weekly RSI {r['Weekly_RSI']} vs avg {r['RSI_Avg']}", color="#e2e8f0", fontsize=8)
        ax_r.xaxis.set_major_formatter(mdates.DateFormatter("%m/%d"))
        for ax in (ax_w, ax_r):
            ax.tick_params(colors="#94a3b8", labelsize=7)
            for sp in ax.spines.values(): sp.set_edgecolor("#1e3a5f")
            ax.grid(color="#1e3a5f", ls="--", lw=0.5, alpha=0.6)
    plt.suptitle(f"Weekly Red Doji Below JMA → Green Reclaim · "
                 f"{datetime.today().strftime('%Y-%m-%d')}", color="#60a5fa", fontsize=12, fontweight="bold")
    plt.tight_layout()
    cp = os.path.join(out_dir, f"weekly_doji_jma_reclaim_chart_{ts}.png")
    plt.savefig(cp, dpi=130, bbox_inches="tight", facecolor="#0f172a")
    plt.close()
    print(f"  📊 Chart → {cp}")

print(f"""
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  📋 SIGNAL (all required; checked on the last {CFG['signal_lookback_weeks']} weekly candles)
  1) Previous week small red doji: body <= {CFG['doji_max_body_range_pct']:.0f}% of range and <= {CFG['doji_max_body_atr']} x weekly ATR({CFG['atr_period']})
  2) Previous week closed below weekly JMA({CFG['jma_period']},{CFG['jma_phase']},{CFG['jma_power']})
  3) Current week green and closes above the weekly JMA
  4) Current week closes above the previous week's high
  5) Weekly RSI({CFG['rsi_period']}) within {CFG['rsi_near_pts']:.0f} points of its {CFG['rsi_avg_period']}-week average
  OUTPUT defaults: Entry = current week close, Stop = lower of the two weeks' lows
  ⚙️  If 0 results, check the FUNNEL, then relax: doji_max_body_range_pct 35 → 45,
     doji_max_body_atr 0.5 → 0.7, rsi_near_pts 5 → 8
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
""")
