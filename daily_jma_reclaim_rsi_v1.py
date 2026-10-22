# ============================================================
# NASDAQ — Daily Reclaim of JMA / EMA8 / SMA21 / SMA50 with RSI 57-63 (v1)
# ============================================================
#
# SIGNAL (all required; the reclaim candle is the latest daily candle)
#
#   1. UPTREND: SMA50 is rising (today's SMA50 > SMA50 10 trading days ago).
#
#   2. PULLBACK: in the 10 trading days before today, at least one daily
#      close was BELOW the SMA21 (the dip / consolidation under the fast
#      averages).
#
#   3. RECLAIM: today's close is ABOVE all four averages — JMA(13, 40, 2;
#      TradingView formula), EMA8, SMA21 and SMA50 — and yesterday's close
#      was NOT above all four (today is the first close back above them).
#
#   4. RSI(14) (Wilder) on today's candle is between 57 and 63.
#
# OUTPUT: Entry_Price and Stop_Loss are reasonable defaults, not
# explicitly requested (Entry = today's close; Stop = the lowest low of
# the pullback, from its first close below SMA21 to today).
# Quantity / Total_Amount come from position_sizing.py (max loss $40).
# Stocks whose ATR14 is under 1% of price are skipped (min_atr_pct).
#
# SINGLE PASS — 1 daily download per ticker, purely technical.
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
    "history_days"          : 550,   # daily bars — ~380 trading days, above _clean()'s 220-bar floor

    # ── Indicators ─────────────────────────────────────────────────
    "jma_period"             : 13,
    "jma_phase"              : 40,
    "jma_power"              : 2,
    "ema_period"             : 8,
    "sma_mid_period"         : 21,
    "sma_period"             : 50,
    "rsi_period"             : 14,

    # ── Pattern ────────────────────────────────────────────────────
    "sma_rising_days"        : 10,    # step 1: SMA50 today > SMA50 this many days ago
    "dip_window"             : 10,    # step 2: a close below SMA21 within this many days before today
    "rsi_min"                : 57.0,  # step 4
    "rsi_max"                : 63.0,

    "signal_lookback_days"   : 1,     # 1 = the reclaim must be the latest candle

    # ── Universe ───────────────────────────────────────────────────
    "min_avg_volume"         : 100_000,
    "min_price"              : 10.0,
    "min_atr_pct"            : 1.0,   # skip stocks moving < 1% a day (e.g. pinned by a takeover)

    "batch_size"             : 50,
    "batch_sleep"            : 1.5,
}

# ── Indicators ──────────────────────────────────────────────────────
def calc_jma(series, period=13, phase=40, power=2):
    """
    Jurik Moving Average — same formula as TradingView's "Jurik Moving
    Average" indicator (identical to the repo's other JMA scanners).
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

# ── Pattern — single source of truth ────────────────────────────────
def check_pattern_at(c, l, jma, ema, sma21, sma50, rsi, i, cfg):
    """
    Checks the reclaim with "today" at bar `i`.
    SINGLE SOURCE OF TRUTH for the pattern logic.

    Computes EVERY step's pass/fail regardless of earlier failures
    (guarding only on missing data), so callers can diagnose exactly
    which step is filtering results out — see FUNNEL_COUNTS below.

    Returns (passed: bool, details: dict).
    """
    w, rd = cfg["dip_window"], cfg["sma_rising_days"]
    if i - max(w, rd) - 1 < 0 or i >= len(c):
        return False, {}
    seg = slice(i - w - 1, i + 1)
    if not all(np.isfinite(x[seg]).all() for x in (jma, ema, sma21, sma50)) \
            or not np.isfinite(sma50[i - rd]) or not np.isfinite(rsi[i]):
        return False, {}

    rising = bool(sma50[i] > sma50[i - rd])                                  # Step 1

    below = [j for j in range(i - w, i) if c[j] < sma21[j]]
    dipped = len(below) > 0                                                  # Step 2

    top_i  = max(jma[i], ema[i], sma21[i], sma50[i])
    top_p  = max(jma[i-1], ema[i-1], sma21[i-1], sma50[i-1])
    reclaim = bool(c[i] > top_i and not (c[i-1] > top_p))                    # Step 3

    rsi_ok = bool(cfg["rsi_min"] <= rsi[i] <= cfg["rsi_max"])               # Step 4

    passed = rising and dipped and reclaim and rsi_ok
    dip_start = below[0] if dipped else i
    return passed, {
        "idx": i, "dip_start": dip_start, "days_below": len(below),
        "dip_low": float(np.min(l[dip_start:i + 1])),
        "rsi": float(rsi[i]), "top": float(top_i),
        "slope_pct": float((sma50[i] / sma50[i - rd] - 1) * 100),
        "rising": rising, "dipped": dipped, "reclaim": reclaim, "rsi_ok": rsi_ok,
    }

# ── Diagnostic funnel — tallies how far each ticker-day gets through
#    the pattern, across the full universe scan, so a 0-match run can
#    be diagnosed empirically instead of guessed at ─────────────────
FUNNEL_COUNTS = {
    "days_checked": 0,
    "passed_step1_sma50_rising": 0,
    "passed_step2_dip_below_sma21": 0,
    "passed_step3_first_close_above_all": 0,
    "passed_step4_rsi_57_63": 0,
}

def find_reclaim_signals(c, l, jma, ema, sma21, sma50, rsi, cfg):
    """
    Checks the last `signal_lookback_days` trading days (1 = the most
    recent candle only). Returns hit dicts, most recent first, and
    tallies FUNNEL_COUNTS for every day checked.
    """
    global FUNNEL_COUNTS
    n = len(c)
    hits = []
    for back in range(0, cfg["signal_lookback_days"]):
        i = (n - 1) - back
        passed, d = check_pattern_at(c, l, jma, ema, sma21, sma50, rsi, i, cfg)
        if not d:
            continue
        FUNNEL_COUNTS["days_checked"] += 1
        if d["rising"]:
            FUNNEL_COUNTS["passed_step1_sma50_rising"] += 1
            if d["dipped"]:
                FUNNEL_COUNTS["passed_step2_dip_below_sma21"] += 1
                if d["reclaim"]:
                    FUNNEL_COUNTS["passed_step3_first_close_above_all"] += 1
                    if d["rsi_ok"]:
                        FUNNEL_COUNTS["passed_step4_rsi_57_63"] += 1
        if passed:
            hits.append(d)
    return hits

# ── Technical signal: reclaim of all four averages, RSI 57-63 ─────────
def analyze_jma_reclaim_rsi(sym, df_daily):
    """
    Returns dict with score and setup details, or None if the pattern
    is not present on the checked candle(s).
    """
    if df_daily is None:
        return None

    price   = float(df_daily["Close"].iloc[-1])
    avg_vol = float(df_daily["Volume"].tail(20).mean())
    if price   < CFG["min_price"]:      return None
    if avg_vol < CFG["min_avg_volume"]: return None
    if len(df_daily) < CFG["sma_period"] + CFG["sma_rising_days"] + 30:
        return None

    close = df_daily["Close"]
    c, l = close.values.astype(float), df_daily["Low"].values.astype(float)
    jma   = calc_jma(close, CFG["jma_period"], CFG["jma_phase"], CFG["jma_power"])
    ema   = close.ewm(span=CFG["ema_period"], adjust=False).mean()
    sma21 = close.rolling(CFG["sma_mid_period"]).mean()
    sma50 = close.rolling(CFG["sma_period"]).mean()
    rsi   = calc_rsi(close, CFG["rsi_period"])
    h  = df_daily["High"].values.astype(float)
    pc = np.concatenate([[np.nan], c[:-1]])
    tr = np.nanmax(np.vstack([h - l, np.abs(h - pc), np.abs(l - pc)]), axis=0)
    atr14 = pd.Series(tr).rolling(14).mean().values[-1]
    if not np.isfinite(atr14) or atr14 / price * 100 < CFG["min_atr_pct"]:
        return None

    hits = find_reclaim_signals(c, l, jma.values, ema.values, sma21.values, sma50.values,
                                rsi.values, CFG)
    if not hits:
        return None

    s = hits[0]   # most recent
    k, ds = s["idx"], s["dip_start"]
    days_since_signal = (len(c) - 1) - k

    # ── Entry / Stop (reasonable defaults — see header note) ─────────
    entry_price = float(c[k])
    stop_loss   = s["dip_low"]                    # lowest low of the pullback
    if entry_price <= stop_loss:
        return None
    risk_pct = (entry_price - stop_loss) / entry_price * 100

    vol   = df_daily["Volume"].values.astype(float)
    avg20 = float(np.mean(vol[max(0, k - 20):k])) if k > 0 else 0.0
    vol_vs_avg = vol[k] / avg20 if avg20 > 0 else 0.0
    above_top_pct = (entry_price / s["top"] - 1) * 100

    # ── Score (0-100) ─────────────────────────────────────────────────
    score = 45
    reasons = [f"RSI {s['rsi']:.1f}"]
    score += max(0, 15 - abs(s["rsi"] - 60) * 5)                  # RSI near the middle of 57-63
    score += max(0, min(15, s["slope_pct"] * 5))                  # SMA50 slope
    reasons.append(f"SMA50 +{s['slope_pct']:.1f}%/{CFG['sma_rising_days']}d")
    if vol_vs_avg >= 1.0:
        score += min(15, (vol_vs_avg - 1) * 15 + 5); reasons.append(f"Vol x{vol_vs_avg:.1f}")
    reasons.append(f"{s['days_below']}d below SMA21")
    if above_top_pct > 4:
        score -= 10; reasons.append("Extended")
    if risk_pct > 12:
        score -= 10; reasons.append("WideStop")
    score = round(min(100, max(0, score)))

    return {
        "Score"          : score,
        "Price"          : round(price, 2),
        "Entry_Price"    : round(entry_price, 2),
        "Stop_Loss"      : round(stop_loss, 2),
        "Risk_%"         : round(risk_pct, 1),
        "RSI"            : round(s["rsi"], 1),
        "JMA"            : round(float(jma.iloc[k]), 2),
        "EMA8"           : round(float(ema.iloc[k]), 2),
        "SMA21"          : round(float(sma21.iloc[k]), 2),
        "SMA50"          : round(float(sma50.iloc[k]), 2),
        "SMA50_Slope_%"  : round(s["slope_pct"], 2),
        "Above_All_%"    : round(above_top_pct, 2),
        "Dip_Start"      : df_daily.index[ds].strftime("%Y-%m-%d"),
        "Days_Below_SMA21": s["days_below"],
        "Vol_vs_20d_Avg" : round(vol_vs_avg, 2),
        "Signal_Date"    : df_daily.index[k].strftime("%Y-%m-%d"),
        "Days_Since_Signal": days_since_signal,
        "Flags"          : " | ".join(reasons),
        "_df_daily"      : df_daily,
        "_jma"           : jma,
        "_ema"           : ema,
        "_sma21"         : sma21,
        "_sma50"         : sma50,
        "_idx"           : k,
        "_dip_start"     : ds,
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
LIVE_COLS = ["Ticker","Price","Score","Entry_Price","Stop_Loss","RSI","Dip_Start","Days_Below_SMA21"]
_CW = {"Ticker":8,"Price":10,"Score":7,"Entry_Price":12,"Stop_Loss":11,
       "RSI":7,"Dip_Start":12,"Days_Below_SMA21":8}
_CF = {"Price":"${:.2f}","Score":"{:.0f}","Entry_Price":"${:.2f}",
       "Stop_Loss":"${:.2f}","RSI":"{:.1f}"}
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
print(f"  Signal: SMA{CFG['sma_period']} rising, a close below SMA{CFG['sma_mid_period']} in the last "
      f"{CFG['dip_window']} days, today the first close above JMA / EMA{CFG['ema_period']} / "
      f"SMA{CFG['sma_mid_period']} / SMA{CFG['sma_period']}, RSI {CFG['rsi_min']:g}-{CFG['rsi_max']:g}\n")

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

for sym in tqdm(list(daily_map.keys()), desc="Checking reclaim + RSI", unit="stk"):
    try:
        r = analyze_jma_reclaim_rsi(sym, daily_map[sym])
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
print(f"  Step 1 — SMA{CFG['sma_period']} rising ({CFG['sma_rising_days']}d)                  : {fc['passed_step1_sma50_rising']}")
print(f"  Step 2 — + close below SMA{CFG['sma_mid_period']} in last {CFG['dip_window']}d       : {fc['passed_step2_dip_below_sma21']}")
print(f"  Step 3 — + first close above all 4 MAs today  : {fc['passed_step3_first_close_above_all']}")
print(f"  Step 4 — + RSI {CFG['rsi_min']:g}-{CFG['rsi_max']:g}                          : {fc['passed_step4_rsi_57_63']}  (= full pattern)")
print(f"{'━'*65}")

if not results:
    print("\n  No matches. Try relaxing (see the FUNNEL above to see which")
    print("  step is actually the bottleneck before guessing):")
    print("   signal_lookback_days            1 → 3     (reclaim in the last 3 days)")
    print("   rsi_min / rsi_max         57-63 → 55-65")
    print("   dip_window                     10 → 15")
    print("   min_price                      10 → 5")

results.sort(key=lambda x: x["Score"], reverse=True)

# ── Always build df_out and save/email (even if 0 results) ────
ts      = datetime.today().strftime("%Y%m%d_%H%M")
out_dir = os.environ.get("GITHUB_WORKSPACE", os.getcwd())

COLS = [
    "Ticker","Price","Score",
    "Entry_Price","Stop_Loss","Risk_%",
    "RSI","JMA","EMA8","SMA21","SMA50","SMA50_Slope_%","Above_All_%",
    "Dip_Start","Days_Below_SMA21","Vol_vs_20d_Avg",
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
    "JMA"           : lambda v: f"${v:.2f}",
    "EMA8"          : lambda v: f"${v:.2f}",
    "SMA21"         : lambda v: f"${v:.2f}",
    "SMA50"         : lambda v: f"${v:.2f}",
    "RSI"           : lambda v: f"{v:.1f}",
    "SMA50_Slope_%" : lambda v: f"{v:+.2f}%",
    "Above_All_%"   : lambda v: f"{v:+.2f}%",
    "Risk_%"        : lambda v: f"{v:.1f}%",
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
             "RSI","Dip_Start","Days_Below_SMA21","SMA50_Slope_%","Vol_vs_20d_Avg"]

if results:
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
    tit = f"  Daily JMA/EMA8/SMA21/SMA50 Reclaim, RSI 57-63   {datetime.today().strftime('%Y-%m-%d')}   {len(df_out)} matches"
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

# Save
fpath = os.path.join(out_dir, f"daily_jma_reclaim_rsi_{ts}.csv")
# ── Position sizing: Quantity / Total_Amount for a fixed max loss (see position_sizing.py)
try:
    from position_sizing import add_position_size
    df_out = add_position_size(df_out)
except Exception as _ps_err:
    print(f"  ⚠️  Position sizing skipped: {_ps_err}")
df_out.to_csv(fpath, index=False)
print(f"\n  💾 CSV → {fpath}")
tv = os.path.join(out_dir, f"tv_daily_jma_reclaim_rsi_{ts}.txt")
with open(tv,"w") as f:
    f.write(f"###Daily JMA/EMA8/SMA21/SMA50 Reclaim, RSI 57-63 {datetime.today().strftime('%Y-%m-%d')}\n")
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

        heads = ["Ticker","Price","Score","Entry","Stop","Risk %","RSI",
                 "Dip since","Days below","SMA50 slope","Qty","Amount"]
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
                + td(f'{float(r.get("RSI",0) or 0):.1f}', "font-weight:700;color:#7c3aed")
                + td(r.get("Dip_Start","—"))
                + td(f'{r.get("Days_Below_SMA21","—")}')
                + td(f'{float(r.get("SMA50_Slope_%",0) or 0):+.2f}%', "color:#16a34a;font-weight:600")
                + td(f'{q}')
                + td(f'${float(a):,.0f}' if a is not None else "—")
                + '</tr>')
        no_results_msg = ('<tr><td colspan="12" style="padding:20px;text-align:center;'
                          'color:#94a3b8;font-size:13px">No matches today</td></tr>')
        more_note = (f'Showing top 50 of {cnt} — full list in the CSV attachment'
                     if cnt > 50 else '📎 Full results attached as CSV')
        tick_box = ('<div style="margin-top:10px;background:#f8fafc;border:1px solid #e2e8f0;'
                    'border-radius:6px;padding:10px 14px">'
                    '<p style="margin:0 0 4px;font-size:10px;color:#94a3b8;font-weight:700">'
                    'TICKERS (comma-separated, copy/paste)</p>'
                    '<p style="margin:0;font-size:12px;color:#1e293b;font-family:monospace;word-break:break-all">'
                    + ", ".join(r.get("Ticker","") for r in rl) + '</p></div>') if rl else ''

        fc = FUNNEL_COUNTS
        html_e = f"""<!DOCTYPE html><html><body style="margin:0;padding:0;
background:#f1f5f9;font-family:'Segoe UI',Arial,sans-serif">
<table width="100%" cellpadding="0" cellspacing="0"><tr><td align="center" style="padding:20px 10px">
<table width="100%" style="max-width:860px;background:#fff;border-radius:12px;
       overflow:hidden;box-shadow:0 4px 16px rgba(0,0,0,0.08)">
  <tr><td style="background:linear-gradient(135deg,#0f172a,#1e3a5f);padding:22px 28px">
<h1 style="margin:0;color:#60a5fa;font-size:20px;font-weight:700">
  📊 Daily JMA/EMA8/SMA21/SMA50 Reclaim, RSI 57-63
</h1>
<p style="margin:6px 0 0;color:#94a3b8;font-size:12px">
  {datetime.today().strftime('%Y-%m-%d %H:%M UTC')} &nbsp;·&nbsp;
  {cnt} match{'es' if cnt!=1 else ''} — SMA{CFG['sma_period']} rising, a dip below SMA{CFG['sma_mid_period']}, today the first close
  above JMA / EMA{CFG['ema_period']} / SMA{CFG['sma_mid_period']} / SMA{CFG['sma_period']} with RSI {CFG['rsi_min']:g}-{CFG['rsi_max']:g}
</p>
  </td></tr>
  <tr><td style="padding:14px 28px 4px;background:#0b1220">
<div style="background:#111827;border:1px solid #1f2937;border-radius:8px;padding:12px 16px">
  <p style="margin:0 0 6px;color:#93c5fd;font-size:12px;font-weight:700">
    🔍 FUNNEL — last {CFG['signal_lookback_days']} trading day(s), every liquid ticker
  </p>
  <p style="margin:0;color:#cbd5e1;font-size:12px">
    {fc['days_checked']} ticker-days &nbsp;→&nbsp;
    {fc['passed_step1_sma50_rising']} SMA{CFG['sma_period']} rising &nbsp;→&nbsp;
    {fc['passed_step2_dip_below_sma21']} dipped below SMA{CFG['sma_mid_period']} &nbsp;→&nbsp;
    {fc['passed_step3_first_close_above_all']} first close above all 4 &nbsp;→&nbsp;
    <b style="color:#facc15">{fc['passed_step4_rsi_57_63']} RSI {CFG['rsi_min']:g}-{CFG['rsi_max']:g} = full match</b>
  </p>
</div>
  </td></tr>
  <tr><td style="padding:16px">
<div style="overflow-x:auto;border-radius:8px;border:1px solid #e2e8f0">
  <table style="border-collapse:collapse;width:100%;min-width:720px">
    <thead><tr>{th_e}</tr></thead>
    <tbody>{rows_e or no_results_msg}</tbody>
  </table>
</div>
<p style="font-size:11px;color:#64748b;margin:8px 0 0">{more_note}</p>
{tick_box}
  </td></tr>
  <tr><td style="background:#f8fafc;padding:12px 28px;
             border-top:1px solid #e2e8f0;text-align:center">
<p style="margin:0;color:#94a3b8;font-size:10px">
  ⚠️ Not financial advice &nbsp;·&nbsp; Entry = today's close, Stop = lowest low of the pullback (defaults) &nbsp;·&nbsp; Qty sized for a $40 max loss &nbsp;·&nbsp; Auto-generated by GitHub Actions
</p>
  </td></tr>
</table>
</td></tr></table>
</body></html>"""

        plain_lines = [
            f"Daily JMA/EMA8/SMA21/SMA50 Reclaim, RSI 57-63 — {datetime.today().strftime('%Y-%m-%d')}",
            f"{cnt} matches (SMA{CFG['sma_period']} rising, dip below SMA{CFG['sma_mid_period']}, first close above all 4 MAs, RSI {CFG['rsi_min']:g}-{CFG['rsi_max']:g})",
            "="*60,
            f"FUNNEL: {fc['days_checked']} ticker-days -> "
            f"{fc['passed_step1_sma50_rising']} SMA50 rising -> "
            f"{fc['passed_step2_dip_below_sma21']} dipped -> "
            f"{fc['passed_step3_first_close_above_all']} reclaimed -> "
            f"{fc['passed_step4_rsi_57_63']} RSI ok (=full match)",
            "="*60,
        ]
        if rl:
            for r in rl[:50]:
                plain_lines.append(
                    f"{r.get('Ticker','—'):<7} ${float(r.get('Price',0) or 0):.2f}  "
                    f"Score:{float(r.get('Score',0) or 0):.0f}  "
                    f"Entry:${float(r.get('Entry_Price',0) or 0):.2f}  "
                    f"SL:${float(r.get('Stop_Loss',0) or 0):.2f}  "
                    f"Risk:{float(r.get('Risk_%',0) or 0):.1f}%  "
                    f"RSI:{float(r.get('RSI',0) or 0):.1f}  Dip:{r.get('Dip_Start','—')}  "
                    f"Qty:{_SIZE.get(r.get('Ticker'), ('—', None))[0]}")
            plain_lines.append("")
            plain_lines.append("Tickers (comma-separated):")
            plain_lines.append(", ".join(r.get("Ticker","") for r in rl))
        else:
            plain_lines.append("No matches today")
        plain_lines.append("\nFull results in CSV attachment.")
        plain_e = "\n".join(plain_lines)

        subj = (f"📊 MA Reclaim + RSI 57-63 — {cnt} signal{'s' if cnt!=1 else ''}"
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

# ── Charts for top 5: last 60 daily candles with JMA / EMA8 / SMA21 / SMA50,
#    the pullback start and the reclaim candle marked ─────────────────
if results:
    top = results[:min(5,len(results))]
    fig, axes = plt.subplots(len(top), 1, figsize=(13, 4.2*len(top)), facecolor="#0f172a", squeeze=False)
    axes = axes[:,0]
    for p, r in enumerate(top):
        ax = axes[p]
        df_all = r["_df_daily"]
        df_p = df_all.tail(60)
        idx  = df_p.index
        k, a = r["_idx"], r["_dip_start"]
        sig_d, sw_d = df_all.index[k], df_all.index[a]
        x = np.arange(len(df_p))
        up = (df_p["Close"] >= df_p["Open"]).values
        col = np.where(up, "#22c55e", "#ef4444")

        ax.set_facecolor("#0f172a")
        ax.vlines(x, df_p["Low"], df_p["High"], color=col, lw=0.8)
        ax.bar(x, (df_p["Close"] - df_p["Open"]).abs().clip(lower=df_p["Close"]*0.0005),
               bottom=np.minimum(df_p["Open"], df_p["Close"]), color=col, width=0.6)
        ax.plot(x, r["_jma"].reindex(idx).values, color="#ef4444", lw=1.1, label="JMA")
        ax.plot(x, r["_ema"].reindex(idx).values, color="#f97316", lw=1.1, label="EMA8")
        ax.plot(x, r["_sma21"].reindex(idx).values, color="#22c55e", lw=1.1, label="SMA21")
        ax.plot(x, r["_sma50"].reindex(idx).values, color="#3b82f6", lw=1.3, label="SMA50")
        ax.axhline(r["Stop_Loss"], color="#ef4444", lw=0.9, ls="--", alpha=0.8, label=f"Stop {r['Stop_Loss']:.2f}")
        pos = {d: i for i, d in enumerate(idx)}
        if sw_d in pos:
            ax.scatter([pos[sw_d]], [df_all["Low"].iloc[a]], marker="^", color="#a78bfa", s=60, zorder=7, label="Dip start")
        if sig_d in pos:
            ax.scatter([pos[sig_d]], [df_all["High"].iloc[k]], marker="v", color="#facc15", s=60, zorder=7, label="Reclaim")
        step = max(1, len(idx) // 8)
        ax.set_xticks(x[::step]); ax.set_xticklabels([d.strftime("%b %d") for d in idx[::step]])
        ax.set_title(f"{r['Ticker']}  |  {r['Price']:.2f} USD  |  Score {r['Score']}  |  RSI {r['RSI']:.1f}  |  "
                     f"dip since {r['Dip_Start']}  |  reclaim {r['Signal_Date']}",
                     color="#e2e8f0", fontsize=9, fontweight="bold", pad=7)
        ax.tick_params(colors="#94a3b8", labelsize=8)
        for sp in ax.spines.values(): sp.set_edgecolor("#1e3a5f")
        ax.legend(loc="upper left", facecolor="#1e293b", labelcolor="#e2e8f0", fontsize=6, ncol=5)
        ax.grid(color="#1e3a5f", ls="--", lw=0.5, alpha=0.6)
    plt.suptitle(f"Daily JMA/EMA8/SMA21/SMA50 Reclaim, RSI 57-63  ·  {datetime.today().strftime('%Y-%m-%d')}",
                 color="#60a5fa", fontsize=12, fontweight="bold", y=1.001)
    plt.tight_layout()
    cp = os.path.join(out_dir, f"daily_jma_reclaim_rsi_chart_{ts}.png")
    plt.savefig(cp, dpi=150, bbox_inches="tight", facecolor="#0f172a")
    if _IN_NOTEBOOK: plt.show()
    else: plt.close()
    print(f"  📊 Chart → {cp}")

print(f"""
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  📋 SIGNAL (all required; the reclaim candle is the latest daily candle)
  1) SMA{CFG['sma_period']} rising: today > {CFG['sma_rising_days']} trading days ago.
  2) Pullback: a close below SMA{CFG['sma_mid_period']} within the {CFG['dip_window']} days before today.
  3) Today the FIRST close above JMA({CFG['jma_period']},{CFG['jma_phase']},{CFG['jma_power']}), EMA{CFG['ema_period']},
     SMA{CFG['sma_mid_period']} and SMA{CFG['sma_period']} (yesterday was not above all four).
  4) RSI({CFG['rsi_period']}) today between {CFG['rsi_min']:g} and {CFG['rsi_max']:g}.

  📋 OUTPUT (reasonable defaults — not explicitly requested)
  Entry_Price = today's CLOSE
  Stop_Loss   = lowest LOW of the pullback (first close below SMA21 to today)
  Universe    = price >= 10 USD, 20d avg volume >= 100k

  📋 SCORE (0-100)
  Base 45 + RSI near 60 (0-15) + SMA50 slope (0-15) + volume above avg (0-15)
  ; -10 if > 4% above the highest average (extended); -10 if stop risk > 12%
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
""")
