# ============================================================
# NASDAQ — Weekly Green Reversal Candle + Monthly RSI 38-44 (v1)
# ============================================================
#
# Looks for a stock that has been falling for several weeks, then
# prints a strong green WEEKLY candle on heavy volume, while its
# MONTHLY RSI(14) is still low (between 38 and 44) at that same moment.
#
# SIGNAL (all required, checked on each of the last
# signal_lookback_weeks weekly candles; the most recent hit wins):
#
#   1. MONTHLY RSI 38-44: Wilder RSI(14) of monthly closes, measured
#      at the close of the signal week (the month in progress uses
#      that week's close as its month-to-date close), is between
#      rsi_min and rsi_max.
#
#   2. PRIOR WEEKLY DOWNTREND: before the signal week, the close fell
#      at least downtrend_min_drop_pct % over downtrend_weeks weeks,
#      AND at least downtrend_min_down_weeks of those weeks closed
#      lower than the week before.
#
#   3. STRONG GREEN WEEK: the signal week closes above its open, the
#      body is at least body_min_atr x the weekly ATR(10) of the
#      prior weeks, and the body is at least body_min_range_pct % of
#      the week's high-low range (no long upper-wick candles).
#
#   4. GOOD VOLUME: the signal week's volume is at least
#      volume_mult x the average of the prior 10 weeks. For a week
#      still in progress the volume is projected to a full 5-day week
#      (volume / days traded x 5), so a Tuesday run is not penalised.
#
# DATA — one download per ticker (~7 years of daily bars), resampled
# to weekly ('W') and monthly ('ME').
#
# OUTPUT: Entry_Price = signal week's close; Stop_Loss = signal week's
# low. These are reasonable defaults, not explicitly requested.
# The downtrend, body, and volume thresholds are also defaults chosen
# to define "downtrend", "good green candle" and "good volume".
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
    # history_days vs _clean(min_bars=750): 2600 calendar days -> ~1856
    # business days (checked with pd.bdate_range), well above 750. A
    # stock needs ~3 years of history (36+ monthly bars) for a stable
    # monthly RSI(14).
    "history_days"            : 2600,

    # ── Monthly RSI ──────────────────────────────────────────────
    "rsi_period"              : 14,
    "rsi_min"                 : 38.0,
    "rsi_max"                 : 44.0,
    "min_monthly_bars"        : 30,

    # ── Prior weekly downtrend (default definition) ─────────────
    "downtrend_weeks"         : 5,     # look back this many weeks before the signal week
    "downtrend_min_drop_pct"  : 5.0,   # close fell at least this % over those weeks
    "downtrend_min_down_weeks": 3,     # at least this many of them closed lower

    # ── Strong green weekly candle (default definition) ─────────
    "atr_period"              : 10,    # weekly ATR of the weeks before the signal week
    "body_min_atr"            : 0.8,   # body >= 0.8 x weekly ATR
    "body_min_range_pct"      : 50.0,  # body >= 50% of the week's high-low range

    # ── Volume (default definition) ──────────────────────────────
    "volume_avg_weeks"        : 10,
    "volume_mult"             : 1.5,   # (projected) week volume >= 1.5 x prior 10-week avg

    "signal_lookback_weeks"   : 2,     # current week + last completed week

    # ── Filters ─────────────────────────────────────────────────
    "min_avg_volume"          : 200_000,
    "min_price"               : 5.0,

    "batch_size"              : 50,
    "batch_sleep"             : 1.5,
}

# ── Indicators ───────────────────────────────────────────────
def resample_ohlcv(df, rule):
    """Resample daily OHLCV to a coarser timeframe ('W' weekly, 'ME' monthly)."""
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

def monthly_rsi_at(monthly_close, week_end, week_close, period):
    """Monthly RSI as it stood at the close of the week ending week_end:
    completed months before that week's month, plus the week's close
    as the month-to-date close of the month in progress."""
    # A week belongs to the month of its last trading day (week_end).
    month_end = (pd.Timestamp(week_end) + pd.offsets.MonthEnd(0)).normalize()
    prior = monthly_close[monthly_close.index < month_end]
    series = pd.concat([prior, pd.Series([float(week_close)], index=[month_end])])
    r = calc_rsi(series, period)
    return float(r.iloc[-1]) if len(r) else float("nan")

def weekly_atr(df_weekly, period):
    h, l, c = df_weekly["High"], df_weekly["Low"], df_weekly["Close"]
    pc = c.shift(1)
    tr = pd.concat([h - l, (h - pc).abs(), (l - pc).abs()], axis=1).max(axis=1)
    return tr.rolling(period).mean()

def check_pattern_at(df_weekly, monthly_close, week_days, week_last, atr_w, i, cfg):
    """
    SINGLE SOURCE OF TRUTH. Checks weekly bar i. Computes EVERY step
    (no short-circuit) so the funnel shows where tickers drop out.
    Returns (passed, details); details is {} only if data is too short.
    week_days[i] = number of daily bars in week i (for volume projection);
    week_last[i] = last trading day of week i (decides which month it belongs to).
    """
    n_dn, n_vol = cfg["downtrend_weeks"], cfg["volume_avg_weeks"]
    if i < max(n_dn, n_vol, cfg["atr_period"]) + 1 or i >= len(df_weekly):
        return False, {}
    w = df_weekly.iloc[i]
    o, h, l, c, v = (float(w[k]) for k in ("Open", "High", "Low", "Close", "Volume"))
    atr = float(atr_w.iloc[i - 1])
    if not (o > 0 and np.isfinite(atr) and atr > 0):
        return False, {}

    # Step 1 — monthly RSI at the signal week's close
    rsi_m = monthly_rsi_at(monthly_close, week_last[i], c, cfg["rsi_period"])
    step1 = bool(np.isfinite(rsi_m) and cfg["rsi_min"] <= rsi_m <= cfg["rsi_max"])

    # Step 2 — prior weekly downtrend (weeks i-n_dn .. i-1)
    closes = df_weekly["Close"].values.astype(float)
    start_c, end_c = closes[i - 1 - n_dn], closes[i - 1]
    drop_pct = float((start_c - end_c) / start_c * 100) if start_c > 0 else 0.0
    down_weeks = int(sum(closes[k] < closes[k - 1] for k in range(i - n_dn, i)))
    step2 = bool(drop_pct >= cfg["downtrend_min_drop_pct"]
                 and down_weeks >= cfg["downtrend_min_down_weeks"])

    # Step 3 — strong green weekly candle
    body = c - o
    rng = h - l
    body_atr = body / atr
    body_rng_pct = body / rng * 100 if rng > 0 else 0.0
    step3 = bool(body > 0 and body_atr >= cfg["body_min_atr"]
                 and body_rng_pct >= cfg["body_min_range_pct"])

    # Step 4 — volume vs prior 10-week average (in-progress week projected to 5 days)
    days = int(week_days[i]) if week_days is not None else 5
    proj_vol = v * 5.0 / days if 0 < days < 5 else v
    avg_vol = float(df_weekly["Volume"].iloc[i - n_vol:i].mean())
    vol_ratio = proj_vol / avg_vol if avg_vol > 0 else 0.0
    step4 = bool(vol_ratio >= cfg["volume_mult"])

    passed = step1 and step2 and step3 and step4
    return passed, {
        "week_date": df_weekly.index[i], "open": o, "high": h, "low": l, "close": c,
        "week_pct": (c - o) / o * 100, "rsi_m": rsi_m, "drop_pct": drop_pct,
        "down_weeks": down_weeks, "body_atr": body_atr, "body_rng_pct": body_rng_pct,
        "vol_ratio": vol_ratio, "week_days": days, "week_complete": days >= 5,
        "step1": step1, "step2": step2, "step3": step3, "step4": step4,
    }

# ── Diagnostic funnel (per ticker: did ANY of the lookback weeks pass
#    steps 1..k cumulatively) ──────────────────────────────────────────
FUNNEL_COUNTS = {
    "tickers_checked": 0,
    "passed_step1_monthly_rsi": 0,
    "passed_step2_downtrend": 0,
    "passed_step3_green_candle": 0,
    "passed_step4_volume": 0,
}

def find_signals(df_weekly, monthly_close, week_days, week_last, atr_w, cfg):
    hits, best = [], {k: False for k in ("s1", "s2", "s3", "s4")}
    n = len(df_weekly)
    any_checked = False
    for i in range(max(0, n - cfg["signal_lookback_weeks"]), n):
        ok, d = check_pattern_at(df_weekly, monthly_close, week_days, week_last, atr_w, i, cfg)
        if not d: continue
        any_checked = True
        c1 = d["step1"]; c2 = c1 and d["step2"]; c3 = c2 and d["step3"]; c4 = c3 and d["step4"]
        best["s1"] |= c1; best["s2"] |= c2; best["s3"] |= c3; best["s4"] |= c4
        if ok: hits.append(d)
    if any_checked:
        FUNNEL_COUNTS["tickers_checked"] += 1
        FUNNEL_COUNTS["passed_step1_monthly_rsi"] += best["s1"]
        FUNNEL_COUNTS["passed_step2_downtrend"] += best["s2"]
        FUNNEL_COUNTS["passed_step3_green_candle"] += best["s3"]
        FUNNEL_COUNTS["passed_step4_volume"] += best["s4"]
    return hits

def analyze_monthly_rsi_weekly_green(sym, df_daily):
    if df_daily is None: return None
    price   = float(df_daily["Close"].iloc[-1])
    avg_vol = float(df_daily["Volume"].tail(20).mean())
    if price   < CFG["min_price"]:      return None
    if avg_vol < CFG["min_avg_volume"]: return None

    df_weekly  = resample_ohlcv(df_daily, "W")
    df_monthly = resample_ohlcv(df_daily, "ME")
    if len(df_monthly) < CFG["min_monthly_bars"] or len(df_weekly) < 30:
        return None
    week_days = df_daily["Close"].resample("W").count()
    week_days = week_days[week_days > 0].reindex(df_weekly.index).fillna(5).values
    _dates = pd.Series(df_daily.index, index=df_daily.index)
    week_last = _dates.resample("W").max().reindex(df_weekly.index)
    week_last = [pd.Timestamp(x) if pd.notna(x) else df_weekly.index[k] for k, x in enumerate(week_last)]
    atr_w = weekly_atr(df_weekly, CFG["atr_period"])

    hits = find_signals(df_weekly, df_monthly["Close"], week_days, week_last, atr_w, CFG)
    if not hits: return None
    sig = hits[-1]   # most recent hit wins

    entry, stop = sig["close"], sig["low"]
    if entry <= stop: return None
    risk_pct = (entry - stop) / entry * 100
    weeks_ago = len(df_weekly) - 1 - df_weekly.index.get_loc(sig["week_date"])

    score = 20
    score += min(30, (sig["vol_ratio"] - CFG["volume_mult"]) * 15 + 10)
    score += min(25, (sig["body_atr"] - CFG["body_min_atr"]) * 20 + 5)
    score += min(15, sig["drop_pct"] - CFG["downtrend_min_drop_pct"] + 5)
    score += 10 - min(10, abs(sig["rsi_m"] - 41) * 3)   # nearer the middle of 38-44
    score = round(min(100, max(0, score)))

    flags = [f"MonthlyRSI {sig['rsi_m']:.1f}", f"Vol {sig['vol_ratio']:.1f}x",
             f"Body {sig['body_atr']:.1f}xATR", f"Down {sig['drop_pct']:.0f}% / {sig['down_weeks']}wk"]
    if not sig["week_complete"]:
        flags.append(f"Week in progress ({sig['week_days']}d, volume projected)")
    rsi_series = calc_rsi(df_monthly["Close"], CFG["rsi_period"])

    return {
        "Score": score, "Price": round(price, 2),
        "Entry_Price": round(entry, 2), "Stop_Loss": round(stop, 2), "Risk_%": round(risk_pct, 1),
        "Monthly_RSI": round(sig["rsi_m"], 1), "Week_%": round(sig["week_pct"], 1),
        "Body_xATR": round(sig["body_atr"], 2), "Body_of_Range_%": round(sig["body_rng_pct"], 0),
        "Volume_x": round(sig["vol_ratio"], 2), "Prior_Drop_%": round(sig["drop_pct"], 1),
        "Down_Weeks": sig["down_weeks"], "Week_Complete": "Yes" if sig["week_complete"] else "No",
        "Signal_Date": sig["week_date"].strftime("%Y-%m-%d"), "Weeks_Since_Signal": int(weeks_ago),
        "Recent_Signal_Count": len(hits),
        "Recent_Signals": ", ".join(h["week_date"].strftime("%Y-%m-%d") for h in hits),
        "Flags": " | ".join(flags),
        "_df_weekly": df_weekly, "_rsi_monthly": rsi_series,
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
                        df = _clean(df, min_bars=750)
                        if df is not None: out[sym] = df
                    except Exception: pass
            elif len(symbols) == 1:
                df = _clean(raw, min_bars=750)
                if df is not None: out[symbols[0]] = df
    except Exception: pass
    for sym in [s for s in symbols if s not in out]:
        for _ in range(2):
            try:
                df = yf.Ticker(sym).history(
                    start=start.strftime("%Y-%m-%d"),
                    end=end.strftime("%Y-%m-%d"),
                    auto_adjust=True, actions=False)
                df = _clean(df, min_bars=750)
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
        r = analyze_monthly_rsi_weekly_green(sym, daily_map[sym])
        if r is None: continue
        r["Ticker"] = sym
        results.append(r)
        print(f"  ✅ {sym:<6} ${r['Price']:.2f}  Score {r['Score']}  RSI(M) {r['Monthly_RSI']}  "
              f"Vol {r['Volume_x']}x  Week {r['Signal_Date']}")
    except Exception: pass

fc = FUNNEL_COUNTS
print(f"\n{'━'*65}\n  SCAN COMPLETE — {len(results)} matches from {got_daily} tickers with data")
print(f"  🔍 FUNNEL (any of the last {CFG['signal_lookback_weeks']} weekly candles)")
print(f"  Tickers checked                         : {fc['tickers_checked']}")
print(f"  Step 1 — monthly RSI {CFG['rsi_min']:.0f}-{CFG['rsi_max']:.0f}              : {fc['passed_step1_monthly_rsi']}")
print(f"  Step 2 — + prior weekly downtrend        : {fc['passed_step2_downtrend']}")
print(f"  Step 3 — + strong green weekly candle    : {fc['passed_step3_green_candle']}")
print(f"  Step 4 — + volume >= {CFG['volume_mult']}x 10-wk avg     : {fc['passed_step4_volume']}  (= full pattern)")
print(f"{'━'*65}")

results.sort(key=lambda x: x["Score"], reverse=True)

# ── Save CSV + TradingView list ───────────────────────────────
ts      = datetime.today().strftime("%Y%m%d_%H%M")
out_dir = os.environ.get("GITHUB_WORKSPACE", os.getcwd())
COLS = ["Ticker","Price","Score","Entry_Price","Stop_Loss","Risk_%",
        "Monthly_RSI","Week_%","Body_xATR","Body_of_Range_%","Volume_x",
        "Prior_Drop_%","Down_Weeks","Week_Complete","Signal_Date",
        "Weeks_Since_Signal","Recent_Signal_Count","Recent_Signals","Flags"]
df_out = pd.DataFrame([{k: v for k, v in r.items() if not k.startswith("_")}
                       for r in results]) if results else pd.DataFrame(columns=COLS)
df_out = df_out[[c for c in COLS if c in df_out.columns]].reset_index(drop=True)

fpath = os.path.join(out_dir, f"monthly_rsi_weekly_green_reversal_{ts}.csv")
try:
    from position_sizing import add_position_size
    df_out = add_position_size(df_out)
except Exception as _ps_err:
    print(f"  ⚠️  Position sizing skipped: {_ps_err}")
df_out.to_csv(fpath, index=False)
print(f"\n  💾 CSV → {fpath}")
tv = os.path.join(out_dir, f"tv_monthly_rsi_weekly_green_reversal_{ts}.txt")
with open(tv, "w") as f:
    f.write(f"###Monthly RSI 38-44 Weekly Green Reversal {datetime.today().strftime('%Y-%m-%d')}\n")
    for r in results: f.write(f"NASDAQ:{r['Ticker']}\n")
print(f"  📋 TradingView → {tv}")
if results:
    print(f"\n  📋 Tickers: {', '.join(r['Ticker'] for r in results)}")

# ── Email with CSV attached ───────────────────────────────────
EMAIL_COLS = ["Ticker","Price","Score","Entry_Price","Stop_Loss","Monthly_RSI",
              "Week_%","Volume_x","Prior_Drop_%","Signal_Date"]

def _fmt_cell(col, v):
    try:
        if col in ("Price","Entry_Price","Stop_Loss"): return f"${float(v):.2f}"
        if col in ("Week_%",): return f"{float(v):+.1f}%"
        if col in ("Prior_Drop_%",): return f"-{float(v):.1f}%"
        if col == "Volume_x": return f"{float(v):.2f}x"
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
<h1 style="margin:0;color:#60a5fa;font-size:19px">📊 Monthly RSI {CFG['rsi_min']:.0f}-{CFG['rsi_max']:.0f} + Weekly Green Reversal Candle</h1>
<p style="margin:6px 0 0;color:#94a3b8;font-size:12px">{datetime.today().strftime('%Y-%m-%d %H:%M UTC')} · {cnt} match{'es' if cnt != 1 else ''}
 — falling for {CFG['downtrend_weeks']} weeks, then a strong green week on {CFG['volume_mult']}x volume, with monthly RSI {CFG['rsi_min']:.0f}-{CFG['rsi_max']:.0f}</p>
</td></tr>
<tr><td style="padding:12px 26px;background:#0b1220"><p style="margin:0;color:#cbd5e1;font-size:12px">
🔍 FUNNEL: {fc['tickers_checked']} checked → {fc['passed_step1_monthly_rsi']} monthly RSI {CFG['rsi_min']:.0f}-{CFG['rsi_max']:.0f}
→ {fc['passed_step2_downtrend']} + prior downtrend → {fc['passed_step3_green_candle']} + strong green week
→ <b style="color:#facc15">{fc['passed_step4_volume']} + volume = full match</b></p></td></tr>
<tr><td style="padding:16px"><div style="overflow-x:auto;border:1px solid #e2e8f0;border-radius:8px">
<table style="border-collapse:collapse;width:100%"><thead><tr>{th}</tr></thead><tbody>{rows}</tbody></table></div>
{tick_html}
<p style="font-size:11px;color:#64748b;margin:10px 0 0">Entry = the signal week's close · Stop = the signal week's low
(defaults). Week_Complete = No means the week is still in progress and its volume is projected to 5 days —
confirm at the Friday close. 📎 Full results (with Quantity for a fixed max loss) attached as CSV.</p>
</td></tr>
<tr><td style="background:#f8fafc;padding:10px;text-align:center;color:#94a3b8;font-size:10px">
⚠️ Not financial advice · Auto-generated by GitHub Actions</td></tr>
</table></td></tr></table></body></html>"""

        plain = [f"Monthly RSI {CFG['rsi_min']:.0f}-{CFG['rsi_max']:.0f} + Weekly Green Reversal Candle — {datetime.today().strftime('%Y-%m-%d')}",
                 f"{cnt} matches",
                 f"FUNNEL: {fc['tickers_checked']} checked -> {fc['passed_step1_monthly_rsi']} monthly RSI -> "
                 f"{fc['passed_step2_downtrend']} downtrend -> {fc['passed_step3_green_candle']} green week -> "
                 f"{fc['passed_step4_volume']} volume (=full match)", "="*60]
        for r in rl[:50]:
            plain.append(f"{r['Ticker']:<6} ${r['Price']:.2f}  Score:{r['Score']}  Entry:${r['Entry_Price']:.2f}  "
                         f"SL:${r['Stop_Loss']:.2f}  RSI(M):{r['Monthly_RSI']}  Week:{r['Week_%']:+.1f}%  "
                         f"Vol:{r['Volume_x']}x  Drop:-{r['Prior_Drop_%']}%  {r['Signal_Date']}"
                         f"{'' if r['Week_Complete'] == 'Yes' else ' (week in progress)'}")
        if rl:
            plain += ["", "Tickers: " + ", ".join(r["Ticker"] for r in rl)]
        else:
            plain.append("No matches today")
        plain.append("\nFull results in CSV attachment.")

        subj = (f"📊 Monthly RSI {CFG['rsi_min']:.0f}-{CFG['rsi_max']:.0f} Weekly Green Reversal — "
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

# ── Charts for top 5: last 30 weekly candles + monthly RSI ─────
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
        ax_w.set_title(f"{r['Ticker']} | ${r['Price']:.2f} | Score {r['Score']} | Week {r['Week_%']:+.1f}% "
                       f"| Vol {r['Volume_x']}x", color="#e2e8f0", fontsize=9, fontweight="bold")
        rs = r["_rsi_monthly"].dropna().tail(36)
        ax_r.set_facecolor("#0f172a")
        ax_r.plot(rs.index, rs.values, color="#60a5fa", lw=1.3)
        ax_r.axhspan(CFG["rsi_min"], CFG["rsi_max"], color="#facc15", alpha=0.15)
        ax_r.set_title(f"Monthly RSI(14) {r['Monthly_RSI']}", color="#e2e8f0", fontsize=8)
        ax_r.xaxis.set_major_formatter(mdates.DateFormatter("%b %y"))
        for ax in (ax_w, ax_r):
            ax.tick_params(colors="#94a3b8", labelsize=7)
            for sp in ax.spines.values(): sp.set_edgecolor("#1e3a5f")
            ax.grid(color="#1e3a5f", ls="--", lw=0.5, alpha=0.6)
    plt.suptitle(f"Monthly RSI {CFG['rsi_min']:.0f}-{CFG['rsi_max']:.0f} + Weekly Green Reversal · "
                 f"{datetime.today().strftime('%Y-%m-%d')}", color="#60a5fa", fontsize=12, fontweight="bold")
    plt.tight_layout()
    cp = os.path.join(out_dir, f"monthly_rsi_weekly_green_reversal_chart_{ts}.png")
    plt.savefig(cp, dpi=130, bbox_inches="tight", facecolor="#0f172a")
    plt.close()
    print(f"  📊 Chart → {cp}")

print(f"""
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  📋 SIGNAL (all required; checked on the last {CFG['signal_lookback_weeks']} weekly candles)
  1) Monthly RSI({CFG['rsi_period']}) at the signal week's close between {CFG['rsi_min']:.0f} and {CFG['rsi_max']:.0f}
  2) Prior downtrend: close fell >= {CFG['downtrend_min_drop_pct']:.0f}% over {CFG['downtrend_weeks']} weeks, with
     >= {CFG['downtrend_min_down_weeks']} lower weekly closes
  3) Strong green week: body >= {CFG['body_min_atr']} x weekly ATR({CFG['atr_period']}) and >= {CFG['body_min_range_pct']:.0f}% of range
  4) Volume >= {CFG['volume_mult']}x the prior {CFG['volume_avg_weeks']}-week average (in-progress week projected)
  OUTPUT defaults: Entry = signal week close, Stop = signal week low
  ⚙️  If 0 results, check the FUNNEL, then relax: volume_mult 1.5 → 1.3,
     body_min_atr 0.8 → 0.6, downtrend_min_drop_pct 5 → 3
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
""")
