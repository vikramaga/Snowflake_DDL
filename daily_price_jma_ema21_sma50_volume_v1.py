# ============================================================
# NASDAQ — Daily Price > JMA > EMA21 & SMA50, Volume > Prior Day (v1)
# ============================================================
#
# SIGNAL — all seven must hold on the same daily bar:
#
#   1. PRICE ABOVE JMA:      Close > JMA(13, phase 40)
#   2. NOT EXTENDED:         Close <= JMA * (1 + max_above_jma_pct/100)
#   3. JMA ABOVE EMA 21:     JMA   > EMA(21)
#   4. JMA ABOVE SMA 50:     JMA   > SMA(50)
#   5. VOLUME EXPANSION:     Volume > previous day's Volume
#   6. VOLUME ABOVE AVERAGE: Volume > 1.5 x 20-day average volume (prior 20 bars)
#   7. RISK CAP:             (Close - Stop) / Close <= max_risk_pct (12%),
#                            Stop = lowest low of the prior 10 bars
#
# Universe filter: last close >= min_price ($10) and 20-day avg volume
# >= min_avg_volume.
#
# The last signal_lookback_days bars are checked and the most recent
# bar that satisfies all seven is reported (Signal_Date /
# Days_Since_Signal), so a signal firing yesterday is not missed.
#
# DATA — only 1 download per ticker: daily bars.
#
# OUTPUT: Entry_Price and Stop_Loss are reasonable defaults (Entry =
# signal-day close; Stop = lowest low of the prior 10 bars), not
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
    "history_days"          : 550,   # ~390 trading days; floor in _clean() is 220

    "jma_period"            : 13,
    "jma_phase"             : 40,
    "ema_period"            : 21,
    "sma_period"            : 50,

    "signal_lookback_days"  : 3,     # check the last N bars, report most recent hit

    "max_above_jma_pct"     : 8.0,   # reject bars where Close is > 8% above JMA (extended)
    "vol_avg_period"        : 20,    # signal volume must exceed this average (prior bars)...
    "vol_avg_mult"          : 1.5,   # ...by at least this multiple
    "stop_lookback"         : 10,    # stop = lowest low of the prior N bars
    "max_risk_pct"          : 12.0,  # reject setups whose entry-to-stop risk exceeds this

    "min_avg_volume"        : 100_000,
    "min_price"             : 10.0,

    "batch_size"            : 50,
    "batch_sleep"           : 1.5,
}

# ── Indicators ───────────────────────────────────────────────
def calc_jma(series, period=13, phase=40):
    """Jurik-style moving average approximation (3-stage filter)."""
    v = np.asarray(series, dtype=float)
    alpha = 2.0 / (period + 1)
    beta  = alpha * (phase / 100.0 + 1.5)
    out = np.empty(len(v)); out[:] = np.nan
    if len(v) == 0: return pd.Series(out, index=getattr(series, "index", None))
    e0 = e1 = e2 = v[0]
    for i in range(len(v)):
        x = v[i]
        if np.isnan(x):
            out[i] = out[i-1] if i else np.nan
            continue
        e0 = (1 - alpha) * e0 + alpha * x
        e1 = (x - e0) * (1 - beta) + beta * e1
        e2 = (1 - alpha) * e2 + alpha * (e0 + e1)
        out[i] = e2
    return pd.Series(out, index=getattr(series, "index", None))

def compute_flags(df, cfg):
    """
    SINGLE SOURCE OF TRUTH for the pattern. Returns a DataFrame aligned to
    df with every stage's boolean computed unconditionally on every bar.
    """
    close, vol = df["Close"], df["Volume"]
    jma   = calc_jma(close, cfg["jma_period"], cfg["jma_phase"])
    jma.index = df.index
    ema21 = close.ewm(span=cfg["ema_period"], adjust=False).mean()
    sma50 = close.rolling(cfg["sma_period"]).mean()
    vavg  = vol.rolling(cfg["vol_avg_period"]).mean().shift(1)   # prior N bars, excludes today
    f = pd.DataFrame(index=df.index)
    f["jma"], f["ema21"], f["sma50"], f["vol_avg"] = jma, ema21, sma50, vavg
    f["valid"]         = sma50.notna() & jma.notna() & vol.shift(1).notna() & vavg.notna()
    f["price_gt_jma"]  = (close > jma) & f["valid"]
    f["not_extended"]  = (close <= jma * (1 + cfg["max_above_jma_pct"] / 100.0)) & f["valid"]
    f["jma_gt_ema21"]  = (jma > ema21) & f["valid"]
    f["jma_gt_sma50"]  = (jma > sma50) & f["valid"]
    f["vol_gt_prev"]   = (vol > vol.shift(1)) & f["valid"]
    f["vol_gt_avg"]    = (vol > vavg * cfg["vol_avg_mult"]) & f["valid"]
    # stop = lowest low of the prior N bars; if that is not below close, fall back to 5%
    stop  = df["Low"].rolling(cfg["stop_lookback"], min_periods=1).min().shift(1)
    stop  = stop.where(stop < close, close * 0.95)
    f["stop"]          = stop
    f["risk_pct"]      = (close - stop) / close * 100
    f["risk_ok"]       = (f["risk_pct"] <= cfg["max_risk_pct"]) & f["valid"]
    f["full"] = (f["price_gt_jma"] & f["not_extended"] & f["jma_gt_ema21"]
                 & f["jma_gt_sma50"] & f["vol_gt_prev"] & f["vol_gt_avg"]
                 & f["risk_ok"])
    return f

# ── Diagnostic funnel — tallied per ticker-day over the lookback window ──
FUNNEL_COUNTS = {
    "ticker_days_checked": 0,
    "passed_step1_price_gt_jma": 0,
    "passed_step2_not_extended": 0,
    "passed_step3_jma_gt_ema21": 0,
    "passed_step4_jma_gt_sma50": 0,
    "passed_step5_vol_gt_prev": 0,
    "passed_step6_vol_gt_avg": 0,
    "passed_step7_risk_ok": 0,
}

# ── Technical signal ───────────────────────────────────────────
def analyze_pattern(sym, df_daily):
    """Returns dict with score and setup details, or None if no signal."""
    if df_daily is None:
        return None

    price   = float(df_daily["Close"].iloc[-1])
    avg_vol = float(df_daily["Volume"].tail(20).mean())
    if price   < CFG["min_price"]:      return None
    if avg_vol < CFG["min_avg_volume"]: return None
    if len(df_daily) < CFG["sma_period"] + 30: return None

    f = compute_flags(df_daily, CFG)
    n = len(df_daily)
    lb = min(CFG["signal_lookback_days"], n)
    w = f.iloc[n - lb:]

    global FUNNEL_COUNTS
    FUNNEL_COUNTS["ticker_days_checked"] += int(w["valid"].sum())
    s1 = w["price_gt_jma"]
    s2 = s1 & w["not_extended"]
    s3 = s2 & w["jma_gt_ema21"]
    s4 = s3 & w["jma_gt_sma50"]
    s5 = s4 & w["vol_gt_prev"]
    s6 = s5 & w["vol_gt_avg"]
    s7 = s6 & w["risk_ok"]
    FUNNEL_COUNTS["passed_step1_price_gt_jma"] += int(s1.sum())
    FUNNEL_COUNTS["passed_step2_not_extended"] += int(s2.sum())
    FUNNEL_COUNTS["passed_step3_jma_gt_ema21"] += int(s3.sum())
    FUNNEL_COUNTS["passed_step4_jma_gt_sma50"] += int(s4.sum())
    FUNNEL_COUNTS["passed_step5_vol_gt_prev"] += int(s5.sum())
    FUNNEL_COUNTS["passed_step6_vol_gt_avg"]  += int(s6.sum())
    FUNNEL_COUNTS["passed_step7_risk_ok"]     += int(s7.sum())

    hits = [i for i in range(n - lb, n) if bool(f["full"].iloc[i])]
    if not hits:
        return None
    k = hits[-1]                       # most recent signal bar
    days_since = (n - 1) - k

    close_k = float(df_daily["Close"].iloc[k])
    vol_k   = float(df_daily["Volume"].iloc[k])
    vol_p   = float(df_daily["Volume"].iloc[k-1])
    jma_k, ema_k, sma_k = float(f["jma"].iloc[k]), float(f["ema21"].iloc[k]), float(f["sma50"].iloc[k])

    entry_price = close_k
    stop_loss = float(f["stop"].iloc[k])
    risk_pct  = float(f["risk_pct"].iloc[k])

    above_jma_pct  = (close_k / jma_k - 1) * 100
    jma_ema_pct    = (jma_k / ema_k - 1) * 100
    jma_sma_pct    = (jma_k / sma_k - 1) * 100
    vol_ratio      = vol_k / vol_p if vol_p > 0 else 0.0
    avg20          = float(df_daily["Volume"].iloc[max(0, k-20):k].mean())
    vol_vs_avg     = vol_k / avg20 if avg20 > 0 else 0.0

    # ── Score (0-100) ────────────────────────────────────────────
    score = 15   # base for clearing every gate
    reasons = []
    score += min(20, jma_ema_pct * 4);  reasons.append(f"JMA>EMA21+{jma_ema_pct:.1f}%")
    score += min(20, jma_sma_pct * 2);  reasons.append(f"JMA>SMA50+{jma_sma_pct:.1f}%")
    score += min(15, above_jma_pct * 3); reasons.append(f"Px>JMA+{above_jma_pct:.1f}%")
    score += min(15, max(0.0, (vol_ratio - 1) * 10)); reasons.append(f"Vol x{vol_ratio:.2f}")
    score += min(10, max(0.0, (vol_vs_avg - 1.5) * 5)); reasons.append(f"Vol/20d x{vol_vs_avg:.2f}")
    score += min(5, max(0.0, (CFG["max_risk_pct"] - risk_pct) / 2)); reasons.append(f"Risk {risk_pct:.1f}%")
    score += max(0, 5 - days_since * 2)
    score = round(min(100, max(0, score)))

    return {
        "Score": score,
        "Price": round(price, 2),
        "Entry_Price": round(entry_price, 2),
        "Stop_Loss": round(stop_loss, 2),
        "Risk_%": round(risk_pct, 1),
        "JMA": round(jma_k, 2),
        "EMA21": round(ema_k, 2),
        "SMA50": round(sma_k, 2),
        "Above_JMA_%": round(above_jma_pct, 1),
        "JMA_vs_EMA21_%": round(jma_ema_pct, 1),
        "JMA_vs_SMA50_%": round(jma_sma_pct, 1),
        "Volume": int(vol_k),
        "Prev_Volume": int(vol_p),
        "Vol_Ratio": round(vol_ratio, 2),
        "Vol_vs_20d_Avg": round(vol_vs_avg, 2),
        "Signal_Date": df_daily.index[k].strftime("%Y-%m-%d"),
        "Days_Since_Signal": days_since,
        "Flags": " | ".join(reasons),
        "_df_daily": df_daily,
        "_flags": f,
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
             "Above_JMA_%","Vol_Ratio","Signal_Date"]
_CW = {"Ticker":8,"Price":10,"Score":7,"Entry_Price":12,"Stop_Loss":11,
       "Above_JMA_%":12,"Vol_Ratio":10,"Signal_Date":13}
_CF = {"Price":"${:.2f}","Score":"{:.0f}","Entry_Price":"${:.2f}",
       "Stop_Loss":"${:.2f}"}
_hdr_done = False

def _live_header():
    global _hdr_done
    if _hdr_done: return
    print("\n" + "━"*95)
    print("  📊  LIVE MATCHES  —  each stock printed the moment it passes all 7 conditions")
    print("━"*95)
    print("".join(f"  {c:<{_CW.get(c,12)}}" for c in LIVE_COLS))
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


# ── Main scan — single pass ─────────────────────────────────────
print("━"*65)
print(f"  STEP 3  SCANNING {len(TICKERS)} TICKERS")
print("━"*65)
print("  Fetching daily bars (single download per ticker)")
print(f"  Signal: JMA < Close <= JMA+{CFG['max_above_jma_pct']:.0f}%, JMA > EMA21, JMA > SMA50, Vol > prior day & > {CFG['vol_avg_mult']}x {CFG['vol_avg_period']}d avg, risk <= {CFG['max_risk_pct']:.0f}%\n")

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

for sym in tqdm(list(daily_map.keys()), desc="Checking pattern", unit="stk"):
    try:
        r = analyze_pattern(sym, daily_map[sym])
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

print(f"\n{'━'*65}")
print(f"  🔍 FUNNEL — ticker-days (last {CFG['signal_lookback_days']} bars per liquid ticker)")
print(f"{'━'*65}")
fc = FUNNEL_COUNTS
print(f"  Ticker-days checked                  : {fc['ticker_days_checked']}")
print(f"  Step 1 — Close > JMA                 : {fc['passed_step1_price_gt_jma']}")
print(f"  Step 2 — + not > {CFG['max_above_jma_pct']:.0f}% above JMA     : {fc['passed_step2_not_extended']}")
print(f"  Step 3 — + JMA > EMA21               : {fc['passed_step3_jma_gt_ema21']}")
print(f"  Step 4 — + JMA > SMA50               : {fc['passed_step4_jma_gt_sma50']}")
print(f"  Step 5 — + Volume > prior day        : {fc['passed_step5_vol_gt_prev']}")
print(f"  Step 6 — + Volume > {CFG['vol_avg_mult']}x {CFG['vol_avg_period']}d avg     : {fc['passed_step6_vol_gt_avg']}")
print(f"  Step 7 — + Risk <= {CFG['max_risk_pct']:.0f}%              : {fc['passed_step7_risk_ok']}  (= full pattern)")
print(f"{'━'*65}")

if not results:
    print("\n  No matches. Check the FUNNEL above for the bottleneck, then try:")
    print("   signal_lookback_days   3 → 5")
    print("   min_price             10 → 5")
    print("   max_above_jma_pct      8 → 12")
    print("   vol_avg_mult         1.5 → 1.2")
    print("   max_risk_pct          12 → 15")
    print("   min_avg_volume    100000 → 50000")

results.sort(key=lambda x: x["Score"], reverse=True)

ts      = datetime.today().strftime("%Y%m%d_%H%M")
out_dir = os.environ.get("GITHUB_WORKSPACE", os.getcwd())

COLS = [
    "Ticker","Price","Score",
    "Entry_Price","Stop_Loss","Risk_%",
    "JMA","EMA21","SMA50",
    "Above_JMA_%","JMA_vs_EMA21_%","JMA_vs_SMA50_%",
    "Volume","Prev_Volume","Vol_Ratio","Vol_vs_20d_Avg",
    "Signal_Date","Days_Since_Signal","Flags",
]
df_out = pd.DataFrame([{k:v for k,v in r.items() if not k.startswith("_")}
                        for r in results]) if results else pd.DataFrame(columns=COLS)
if not df_out.empty:
    df_out = df_out[[c for c in COLS if c in df_out.columns]]
    df_out.reset_index(drop=True, inplace=True)

if results:
    CLI_COLS = ["Ticker","Price","Score","Entry_Price","Stop_Loss",
                "Above_JMA_%","JMA_vs_SMA50_%","Vol_Ratio","Signal_Date"]
    print()
    print(df_out.head(40)[CLI_COLS].to_string(index=False))
    if len(df_out) > 40: print(f"  ... {len(df_out)-40} more in CSV")

fpath = os.path.join(out_dir, f"daily_price_jma_ema21_sma50_volume_{ts}.csv")
df_out.to_csv(fpath, index=False)
print(f"\n  💾 CSV → {fpath}")
tv = os.path.join(out_dir, f"tv_daily_price_jma_ema21_sma50_volume_{ts}.txt")
with open(tv,"w") as f:
    f.write(f"###Daily Price>JMA>EMA21/SMA50 Vol {datetime.today().strftime('%Y-%m-%d')}\n")
    for r in results: f.write(f"NASDAQ:{r['Ticker']}\n")
print(f"  📋 TradingView → {tv}")

# ── Email with CSV attached ───────────────────────────────
def _send_email(rl, csv_path):
    import smtplib
    from email.mime.multipart import MIMEMultipart
    from email.mime.text      import MIMEText
    from email.mime.base      import MIMEBase
    from email                import encoders

    gu = _GMAIL_USER; gp = _GMAIL_PASS; et = _EMAIL_TO

    if not gu:
        print("[Email] ❌  GMAIL_USER secret is empty"); return
    if not gp:
        print("[Email] ❌  GMAIL_PASS secret is empty"); return
    if not et:
        print("[Email] ❌  EMAIL_TO secret is empty"); return

    eto = [e.strip() for e in et.split(",") if e.strip()]
    cnt = len(rl)

    try:
        print(f"[Email] Sending to {et}  ({cnt} results)...")
        heads = ["Ticker","Price","Score","Entry_Price","Stop_Loss",
                 "Above_JMA_%","JMA_vs_SMA50_%","Vol_Ratio","Signal_Date"]
        th_e = "".join(
            f'<th style="background:#1e293b;color:#e2e8f0;padding:8px 11px;'
            f'font-size:11px;font-weight:700;border-bottom:2px solid #3b82f6;'
            f'white-space:nowrap">{c}</th>' for c in heads)
        rows_e = ""
        for i, r in enumerate(rl[:50]):
            bg = "#fff" if i % 2 == 0 else "#f0f9ff"
            rows_e += (
                f'<tr style="background:{bg}">'
                f'<td style="padding:6px 11px;font-size:12px;font-weight:700">{r.get("Ticker","—")}</td>'
                f'<td style="padding:6px 11px;font-size:12px">${float(r.get("Price",0) or 0):.2f}</td>'
                f'<td style="padding:6px 11px;font-size:12px;font-weight:700;'
                f'background:#166534;color:#fff;text-align:center">{float(r.get("Score",0) or 0):.0f}</td>'
                f'<td style="padding:6px 11px;font-size:12px;color:#22c55e">${float(r.get("Entry_Price",0) or 0):.2f}</td>'
                f'<td style="padding:6px 11px;font-size:12px;color:#ef4444">${float(r.get("Stop_Loss",0) or 0):.2f}</td>'
                f'<td style="padding:6px 11px;font-size:12px">+{float(r.get("Above_JMA_%",0) or 0):.1f}%</td>'
                f'<td style="padding:6px 11px;font-size:12px">+{float(r.get("JMA_vs_SMA50_%",0) or 0):.1f}%</td>'
                f'<td style="padding:6px 11px;font-size:12px;font-weight:600">x{float(r.get("Vol_Ratio",0) or 0):.2f}</td>'
                f'<td style="padding:6px 11px;font-size:12px;text-align:center;'
                f'color:#a78bfa;font-weight:600">{r.get("Signal_Date","—")}</td>'
                f'</tr>')
        no_results_msg = ('<tr><td colspan="9" style="padding:20px;text-align:center;'
                          'color:#94a3b8;font-size:13px">No matches today</td></tr>')
        shown = min(cnt, 50)
        more_note = (f'<p style="font-size:11px;color:#64748b;margin:8px 0 0">'
                     f'Showing top {shown} of {cnt} — full list in the CSV attachment</p>'
                     if cnt > 50 else
                     '<p style="font-size:11px;color:#64748b;margin:8px 0 0">📎 Full results attached as CSV</p>')
        tick_block = ""
        if rl:
            tick_block = (
                '<div style="margin-top:10px;background:#f8fafc;border:1px solid #e2e8f0;'
                'border-radius:6px;padding:10px 14px">'
                '<p style="margin:0 0 4px;font-size:10px;color:#94a3b8;font-weight:700">'
                'TOP TICKERS (comma-separated, copy/paste)</p>'
                '<p style="margin:0;font-size:12px;color:#1e293b;font-family:monospace;'
                'word-break:break-all">' + ", ".join(r.get("Ticker","") for r in rl[:100]) + '</p></div>')

        fc = FUNNEL_COUNTS
        html_e = f"""<!DOCTYPE html><html><body style="margin:0;padding:0;
background:#f1f5f9;font-family:'Segoe UI',Arial,sans-serif">
<table width="100%" cellpadding="0" cellspacing="0"><tr><td align="center" style="padding:20px 10px">
<table width="100%" style="max-width:800px;background:#fff;border-radius:12px;
       overflow:hidden;box-shadow:0 4px 16px rgba(0,0,0,0.08)">
  <tr><td style="background:linear-gradient(135deg,#0f172a,#1e3a5f);padding:22px 28px">
<h1 style="margin:0;color:#60a5fa;font-size:20px;font-weight:700">
  📈 Daily Price &gt; JMA &gt; EMA21 &amp; SMA50 + Volume Up
</h1>
<p style="margin:6px 0 0;color:#94a3b8;font-size:12px">
  {datetime.today().strftime('%Y-%m-%d %H:%M UTC')} &nbsp;·&nbsp;
  {cnt} match{'es' if cnt!=1 else ''} — Close &gt; JMA (≤8% above), JMA &gt; EMA21, JMA &gt; SMA50, Volume &gt; prior day &amp; 1.5× 20d avg, risk ≤12%
</p>
  </td></tr>
  <tr><td style="padding:14px 28px 4px;background:#0b1220">
<div style="background:#111827;border:1px solid #1f2937;border-radius:8px;padding:12px 16px">
  <p style="margin:0 0 6px;color:#93c5fd;font-size:12px;font-weight:700">
    🔍 FUNNEL — ticker-days over last {CFG['signal_lookback_days']} bars
  </p>
  <p style="margin:0;color:#cbd5e1;font-size:12px">
    {fc['ticker_days_checked']} checked &nbsp;→&nbsp;
    {fc['passed_step1_price_gt_jma']} Close&gt;JMA &nbsp;→&nbsp;
    {fc['passed_step2_not_extended']} not extended &nbsp;→&nbsp;
    {fc['passed_step3_jma_gt_ema21']} JMA&gt;EMA21 &nbsp;→&nbsp;
    {fc['passed_step4_jma_gt_sma50']} JMA&gt;SMA50 &nbsp;→&nbsp;
    {fc['passed_step5_vol_gt_prev']} Vol&gt;prior day &nbsp;→&nbsp;
    {fc['passed_step6_vol_gt_avg']} Vol&gt;1.5× 20d avg &nbsp;→&nbsp;
    <b style="color:#facc15">{fc['passed_step7_risk_ok']} risk ≤12% = full match</b>
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
{more_note}
{tick_block}
  </td></tr>
  <tr><td style="background:#f8fafc;padding:12px 28px;
             border-top:1px solid #e2e8f0;text-align:center">
<p style="margin:0;color:#94a3b8;font-size:10px">
  ⚠️ Not financial advice &nbsp;·&nbsp; Entry = signal close, Stop = prior 10-bar low (defaults) &nbsp;·&nbsp; Auto-generated by GitHub Actions
</p>
  </td></tr>
</table>
</td></tr></table>
</body></html>"""

        plain_lines = [
            f"Daily Price > JMA > EMA21 & SMA50 + Volume Up — {datetime.today().strftime('%Y-%m-%d')}",
            f"{cnt} matches (Close>JMA <=8% above, JMA>EMA21, JMA>SMA50, Volume>prior day & 1.5x 20d avg, risk<=12%)",
            "="*60,
            f"FUNNEL: {fc['ticker_days_checked']} ticker-days -> "
            f"{fc['passed_step1_price_gt_jma']} Close>JMA -> "
            f"{fc['passed_step2_not_extended']} not extended -> "
            f"{fc['passed_step3_jma_gt_ema21']} JMA>EMA21 -> "
            f"{fc['passed_step4_jma_gt_sma50']} JMA>SMA50 -> "
            f"{fc['passed_step5_vol_gt_prev']} Vol>prev -> "
            f"{fc['passed_step6_vol_gt_avg']} Vol>1.5x 20d avg -> "
            f"{fc['passed_step7_risk_ok']} risk<=12% (=full match)",
            "="*60,
        ]
        if rl:
            for r in rl[:50]:
                plain_lines.append(
                    f"{r.get('Ticker','—'):<7} ${float(r.get('Price',0) or 0):.2f}  "
                    f"Score:{float(r.get('Score',0) or 0):.0f}  "
                    f"Entry:${float(r.get('Entry_Price',0) or 0):.2f}  "
                    f"SL:${float(r.get('Stop_Loss',0) or 0):.2f}  "
                    f"AboveJMA:+{float(r.get('Above_JMA_%',0) or 0):.1f}%  "
                    f"JMAvsSMA50:+{float(r.get('JMA_vs_SMA50_%',0) or 0):.1f}%  "
                    f"Vol:x{float(r.get('Vol_Ratio',0) or 0):.2f}  "
                    f"Signal:{r.get('Signal_Date','—')}")
            plain_lines.append("")
            plain_lines.append("Tickers (comma-separated):")
            plain_lines.append(", ".join(r.get("Ticker","") for r in rl[:100]))
        else:
            plain_lines.append("No matches today")
        plain_lines.append("\nFull results in CSV attachment.")
        plain_e = "\n".join(plain_lines)

        subj = (f"📈 Daily Price>JMA>EMA21/SMA50 + Vol — {cnt} signal{'s' if cnt!=1 else ''}"
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
            print(f"[Email] 📎 Attached: {os.path.basename(csv_path)} ({os.path.getsize(csv_path):,} bytes)")
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
        print("[Email] ❌  AUTHENTICATION FAILED (use a Gmail App Password)")
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

# ── Charts for top 5: price + JMA/EMA21/SMA50, volume panel ──
if results:
    top = results[:min(5,len(results))]
    fig, axes = plt.subplots(len(top)*2,1,figsize=(13,3.0*len(top)*2),facecolor="#0f172a",
                              gridspec_kw={"height_ratios":[3,1]*len(top)}, squeeze=False)
    axes = axes[:,0]
    for p, r in enumerate(top):
        ax = axes[p*2]; axv = axes[p*2+1]
        df_p = r["_df_daily"].tail(120).copy()
        fl = r["_flags"].reindex(df_p.index)
        ax.set_facecolor("#0f172a")
        ax.plot(df_p.index, df_p["Close"], color="#e2e8f0", lw=1.3, label="Close", zorder=5)
        ax.plot(df_p.index, fl["jma"],   color="#22c55e", lw=1.4, label="JMA 13")
        ax.plot(df_p.index, fl["ema21"], color="#f59e0b", lw=1.2, label="EMA 21")
        ax.plot(df_p.index, fl["sma50"], color="#60a5fa", lw=1.2, label="SMA 50")
        sd = pd.Timestamp(r["Signal_Date"])
        if sd in df_p.index:
            ax.scatter([sd],[df_p.loc[sd,"Close"]], color="#facc15", s=60, zorder=7, label="Signal")
        ax.axhline(r["Stop_Loss"], color="#ef4444", lw=1.0, ls=":", alpha=0.7,
                   label=f"Stop ${r['Stop_Loss']:.2f}")
        ax.set_title(f"{r['Ticker']}  |  ${r['Price']:.2f}  |  Score {r['Score']}  |  "
                     f"Vol x{r['Vol_Ratio']:.2f} vs prior day",
                     color="#e2e8f0", fontsize=9, fontweight="bold", pad=7)
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))
        ax.tick_params(colors="#94a3b8", labelsize=8)
        for sp in ax.spines.values(): sp.set_edgecolor("#1e3a5f")
        ax.legend(loc="upper left", facecolor="#1e293b", labelcolor="#e2e8f0", fontsize=6, ncol=3)
        ax.grid(color="#1e3a5f", ls="--", lw=0.5, alpha=0.6)
        axv.set_facecolor("#0f172a")
        up = df_p["Close"].diff() >= 0
        axv.bar(df_p.index, df_p["Volume"], color=np.where(up, "#22c55e", "#ef4444"), width=0.8)
        axv.xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))
        axv.tick_params(colors="#94a3b8", labelsize=7)
        for sp in axv.spines.values(): sp.set_edgecolor("#1e3a5f")
        axv.grid(color="#1e3a5f", ls="--", lw=0.5, alpha=0.4)
    plt.suptitle(f"Daily Price > JMA > EMA21 & SMA50 + Volume Up  ·  {datetime.today().strftime('%Y-%m-%d')}",
                 color="#60a5fa", fontsize=12, fontweight="bold", y=1.001)
    plt.tight_layout()
    cp = os.path.join(out_dir, f"daily_price_jma_ema21_sma50_volume_chart_{ts}.png")
    plt.savefig(cp, dpi=150, bbox_inches="tight", facecolor="#0f172a")
    if _IN_NOTEBOOK: plt.show()
    else: plt.close()
    print(f"  📊 Chart → {cp}")

print("""
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  📋 SIGNAL (all required, same daily bar)
  1) Close > JMA(13, phase 40)
  2) Close no more than 8% above JMA (not extended)
  3) JMA > EMA(21)
  4) JMA > SMA(50)
  5) Volume > previous day's volume
  6) Volume > 1.5 x 20-day average volume
  7) Risk to stop (prior 10-bar low) <= 12%
  Universe: price >= $10, 20d avg volume >= 100k
  Last signal_lookback_days bars are checked; most recent hit reported.

  📋 SCORE (0-100): base 15 + JMA/EMA21 gap + JMA/SMA50 gap + price
  above JMA + volume vs prior day + volume vs 20d avg + low risk + freshness
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
""")
