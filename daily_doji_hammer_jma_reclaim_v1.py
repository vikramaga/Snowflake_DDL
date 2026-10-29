# ============================================================
# NASDAQ — Daily Red Doji / Hammer Below JMA → Reclaim (v1)
# ============================================================
#
# Two-candle DAILY pattern: yesterday a small red doji OR a red hammer
# closed below the JMA; today price closes above that red candle's
# high AND back above the JMA. Targets: T1 = EMA8, T2 = SMA21.
#
# SIGNAL (all required; signal day = one of the last
# signal_lookback_days sessions; the most recent hit wins):
#
#   1. RED DOJI OR HAMMER (the day before the signal day):
#      closes below its open, and EITHER
#        doji  : body <= doji_max_body_range_pct % of its high-low range
#                AND body <= doji_max_body_atr x ATR(14)
#        hammer: lower wick >= hammer_min_wick_body x body AND upper
#                wick <= hammer_max_upper_pct % of the range
#
#   2. RED CANDLE BELOW JMA: its close is below JMA(13, 40, 2)
#      (TradingView Jurik MA formula).
#
#   3. CLOSE ABOVE RED HIGH: signal day closes above the red candle's high.
#
#   4. CLOSE ABOVE JMA: signal day closes above the JMA.
#
#   5. TARGETS AHEAD: EMA8 (T1) and SMA21 (T2) are both still above the
#      signal day's close, so both targets are actually upside.
#
# OUTPUT: Entry_Price = signal day's close; Stop_Loss = the red
# candle's low (the lower of the two lows); Target_1 = EMA8,
# Target_2 = SMA21 on the signal day. The doji/hammer shape limits, the
# stop and step 5 are reasonable defaults, not explicitly requested.
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
    # history_days vs _clean(min_bars=150): 500 calendar days -> ~358
    # business days (checked with pd.bdate_range), well above 150.
    "history_days"            : 500,

    # ── JMA (TradingView Jurik MA inputs) and targets ─────────────
    "jma_period"              : 13,
    "jma_phase"               : 40,
    "jma_power"               : 2,
    "ema_period"              : 8,     # Target 1
    "sma_period"              : 21,    # Target 2

    # ── Red candle shape (default definitions) ──────────────────
    "atr_period"              : 14,
    "doji_max_body_range_pct" : 35.0,  # doji: body <= 35% of the high-low range
    "doji_max_body_atr"       : 0.5,   #       and body <= 0.5 x ATR(14)
    "hammer_min_wick_body"    : 2.0,   # hammer: lower wick >= 2 x body
    "hammer_max_upper_pct"    : 25.0,  #         upper wick <= 25% of the range

    "require_targets_above"   : True,  # step 5: EMA8 and SMA21 above the entry
    "signal_lookback_days"    : 2,     # today + yesterday

    # ── Filters ─────────────────────────────────────────────────
    "min_avg_volume"          : 200_000,
    "min_price"               : 5.0,

    "batch_size"              : 50,
    "batch_sleep"             : 1.5,
}

# ── Indicators ───────────────────────────────────────────────
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

def calc_atr(df, period):
    h, l, c = df["High"], df["Low"], df["Close"]
    pc = c.shift(1)
    tr = pd.concat([h - l, (h - pc).abs(), (l - pc).abs()], axis=1).max(axis=1)
    return tr.rolling(period).mean()

def check_pattern_at(df, jma, ema, sma, atr, i, cfg):
    """
    SINGLE SOURCE OF TRUTH. Checks the daily pair (i-1 = red doji/hammer,
    i = signal day). Computes EVERY step (no short-circuit) so the
    funnel shows where tickers drop out.
    Returns (passed, details); details is {} only if data is too short.
    """
    if i < cfg["sma_period"] + 2 or i >= len(df):
        return False, {}
    p, w = df.iloc[i - 1], df.iloc[i]
    po, ph, pl, pc = (float(p[k]) for k in ("Open", "High", "Low", "Close"))
    co, ch, cl, cc = (float(w[k]) for k in ("Open", "High", "Low", "Close"))
    j_prev, j_curr = float(jma.iloc[i - 1]), float(jma.iloc[i])
    t1, t2 = float(ema.iloc[i]), float(sma.iloc[i])
    a = float(atr.iloc[i - 1])
    if not all(np.isfinite(x) for x in (j_prev, j_curr, t1, t2, a)) or a <= 0 or po <= 0:
        return False, {}

    # Step 1 — red doji or red hammer
    body = po - pc
    rng = ph - pl
    lower_wick = pc - pl                      # red candle: body bottom is the close
    upper_wick = ph - po                      #             body top is the open
    body_rng = body / rng * 100 if rng > 0 else 0.0
    is_doji = bool(body > 0 and body_rng <= cfg["doji_max_body_range_pct"]
                   and body / a <= cfg["doji_max_body_atr"])
    is_hammer = bool(body > 0 and rng > 0 and lower_wick >= cfg["hammer_min_wick_body"] * body
                     and upper_wick / rng * 100 <= cfg["hammer_max_upper_pct"])
    step1 = is_doji or is_hammer

    # Step 2 — red candle closed below the JMA
    step2 = bool(pc < j_prev)

    # Step 3 — signal day closes above the red candle's high
    step3 = bool(cc > ph)

    # Step 4 — signal day closes above the JMA
    step4 = bool(cc > j_curr)

    # Step 5 — both targets still above the entry
    step5 = bool(t1 > cc and t2 > cc) if cfg["require_targets_above"] else True

    passed = step1 and step2 and step3 and step4 and step5
    ctype = "Doji+Hammer" if (is_doji and is_hammer) else ("Doji" if is_doji else ("Hammer" if is_hammer else "—"))
    return passed, {
        "date": df.index[i], "red_date": df.index[i - 1], "candle_type": ctype,
        "red_open": po, "red_high": ph, "red_low": pl, "red_close": pc,
        "open": co, "high": ch, "low": cl, "close": cc,
        "jma_prev": j_prev, "jma_curr": j_curr, "ema8": t1, "sma21": t2,
        "body_rng": body_rng, "body_atr": body / a, "wick_body": lower_wick / body if body > 0 else 0.0,
        "step1": step1, "step2": step2, "step3": step3, "step4": step4, "step5": step5,
    }

# ── Diagnostic funnel (per ticker: did ANY of the lookback days pass
#    steps 1..k cumulatively) ──────────────────────────────────────────
FUNNEL_COUNTS = {
    "tickers_checked": 0,
    "passed_step1_red_doji_or_hammer": 0,
    "passed_step2_red_below_jma": 0,
    "passed_step3_close_above_red_high": 0,
    "passed_step4_close_above_jma": 0,
    "passed_step5_targets_above": 0,
}
_FUNNEL_KEYS = list(FUNNEL_COUNTS)[1:]

def find_signals(df, jma, ema, sma, atr, cfg):
    hits, best = [], [False] * 5
    n = len(df)
    any_checked = False
    for i in range(max(0, n - cfg["signal_lookback_days"]), n):
        ok, d = check_pattern_at(df, jma, ema, sma, atr, i, cfg)
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

def analyze_daily_doji_hammer_jma_reclaim(sym, df_daily):
    if df_daily is None: return None
    price   = float(df_daily["Close"].iloc[-1])
    avg_vol = float(df_daily["Volume"].tail(20).mean())
    if price   < CFG["min_price"]:      return None
    if avg_vol < CFG["min_avg_volume"]: return None
    if len(df_daily) < 60: return None

    close = df_daily["Close"]
    jma = calc_jma(close, CFG["jma_period"], CFG["jma_phase"], CFG["jma_power"])
    ema = close.ewm(span=CFG["ema_period"], adjust=False).mean()
    sma = close.rolling(CFG["sma_period"]).mean()
    atr = calc_atr(df_daily, CFG["atr_period"])

    hits = find_signals(df_daily, jma, ema, sma, atr, CFG)
    if not hits: return None
    sig = hits[-1]   # most recent hit wins

    entry = sig["close"]
    stop = min(sig["red_low"], sig["low"])
    if entry <= stop: return None
    risk = entry - stop
    risk_pct = risk / entry * 100
    t1, t2 = sig["ema8"], sig["sma21"]
    r1, r2 = (t1 - entry) / risk, (t2 - entry) / risk
    days_ago = len(df_daily) - 1 - df_daily.index.get_loc(sig["date"])
    # if the signal was yesterday, has today already reached a target?
    after = df_daily.iloc[df_daily.index.get_loc(sig["date"]) + 1:]
    hit_t1 = bool(len(after) and after["High"].max() >= t1)
    hit_t2 = bool(len(after) and after["High"].max() >= t2)
    hit_sl = bool(len(after) and after["Low"].min() <= stop)

    score = 30
    score += min(20, max(0, r1) * 10)                               # room to Target 1
    score += min(15, max(0, r2) * 5)                                # room to Target 2
    score += max(0, 15 - risk_pct * 2)                              # tight stop
    score += 10 if sig["candle_type"] in ("Hammer", "Doji+Hammer") else 5
    score += min(10, (entry - sig["red_high"]) / sig["red_high"] * 100 * 4)   # decisive close over red high
    score = round(min(100, max(0, score)))

    flags = [f"{sig['candle_type']} on {sig['red_date'].strftime('%m-%d')}",
             f"T1 EMA8 {r1:.1f}R", f"T2 SMA21 {r2:.1f}R"]
    if hit_t1: flags.append("T1 already reached")
    if hit_t2: flags.append("T2 already reached")
    if hit_sl: flags.append("Stop already hit")

    return {
        "Score": score, "Price": round(price, 2),
        "Entry_Price": round(entry, 2), "Stop_Loss": round(stop, 2), "Risk_%": round(risk_pct, 1),
        "Target_1_EMA8": round(t1, 2), "T1_R": round(r1, 2), "T1_%": round((t1 - entry) / entry * 100, 1),
        "Target_2_SMA21": round(t2, 2), "T2_R": round(r2, 2), "T2_%": round((t2 - entry) / entry * 100, 1),
        "Candle_Type": sig["candle_type"], "Red_Candle_Date": sig["red_date"].strftime("%Y-%m-%d"),
        "Red_High": round(sig["red_high"], 2), "Red_Low": round(sig["red_low"], 2),
        "JMA": round(sig["jma_curr"], 2), "Body_of_Range_%": round(sig["body_rng"], 0),
        "Lower_Wick_x_Body": round(sig["wick_body"], 1),
        "Signal_Date": sig["date"].strftime("%Y-%m-%d"), "Days_Since_Signal": int(days_ago),
        "Recent_Signal_Count": len(hits),
        "Flags": " | ".join(flags),
        "_df": df_daily, "_jma": jma, "_ema": ema, "_sma": sma,
    }

# ── Download: daily bars ─────────────────────────────────────────────
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
                        df = _clean(df, min_bars=150)
                        if df is not None: out[sym] = df
                    except Exception: pass
            elif len(symbols) == 1:
                df = _clean(raw, min_bars=150)
                if df is not None: out[symbols[0]] = df
    except Exception: pass
    for sym in [s for s in symbols if s not in out]:
        for _ in range(2):
            try:
                df = yf.Ticker(sym).history(
                    start=start.strftime("%Y-%m-%d"),
                    end=end.strftime("%Y-%m-%d"),
                    auto_adjust=True, actions=False)
                df = _clean(df, min_bars=150)
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
        r = analyze_daily_doji_hammer_jma_reclaim(sym, daily_map[sym])
        if r is None: continue
        r["Ticker"] = sym
        results.append(r)
        print(f"  ✅ {sym:<6} ${r['Price']:.2f}  Score {r['Score']}  {r['Candle_Type']}  "
              f"T1 ${r['Target_1_EMA8']} ({r['T1_R']}R)  T2 ${r['Target_2_SMA21']} ({r['T2_R']}R)  {r['Signal_Date']}")
    except Exception: pass

fc = FUNNEL_COUNTS
print(f"\n{'━'*65}\n  SCAN COMPLETE — {len(results)} matches from {got_daily} tickers with data")
print(f"  🔍 FUNNEL (any of the last {CFG['signal_lookback_days']} sessions)")
print(f"  Tickers checked                          : {fc['tickers_checked']}")
print(f"  Step 1 — red doji or hammer               : {fc['passed_step1_red_doji_or_hammer']}")
print(f"  Step 2 — + red candle closed below JMA    : {fc['passed_step2_red_below_jma']}")
print(f"  Step 3 — + next day closed above red high : {fc['passed_step3_close_above_red_high']}")
print(f"  Step 4 — + next day closed above JMA      : {fc['passed_step4_close_above_jma']}")
print(f"  Step 5 — + EMA8 & SMA21 above entry       : {fc['passed_step5_targets_above']}  (= full pattern)")
print(f"{'━'*65}")

results.sort(key=lambda x: x["Score"], reverse=True)

# ── Save CSV + TradingView list ───────────────────────────────
ts      = datetime.today().strftime("%Y%m%d_%H%M")
out_dir = os.environ.get("GITHUB_WORKSPACE", os.getcwd())
COLS = ["Ticker","Price","Score","Entry_Price","Stop_Loss","Risk_%",
        "Target_1_EMA8","T1_R","T1_%","Target_2_SMA21","T2_R","T2_%",
        "Candle_Type","Red_Candle_Date","Red_High","Red_Low","JMA",
        "Body_of_Range_%","Lower_Wick_x_Body","Signal_Date","Days_Since_Signal",
        "Recent_Signal_Count","Flags"]
df_out = pd.DataFrame([{k: v for k, v in r.items() if not k.startswith("_")}
                       for r in results]) if results else pd.DataFrame(columns=COLS)
df_out = df_out[[c for c in COLS if c in df_out.columns]].reset_index(drop=True)

fpath = os.path.join(out_dir, f"daily_doji_hammer_jma_reclaim_{ts}.csv")
try:
    from position_sizing import add_position_size
    df_out = add_position_size(df_out)
except Exception as _ps_err:
    print(f"  ⚠️  Position sizing skipped: {_ps_err}")
df_out.to_csv(fpath, index=False)
print(f"\n  💾 CSV → {fpath}")
tv = os.path.join(out_dir, f"tv_daily_doji_hammer_jma_reclaim_{ts}.txt")
with open(tv, "w") as f:
    f.write(f"###Daily Doji Hammer JMA Reclaim {datetime.today().strftime('%Y-%m-%d')}\n")
    for r in results: f.write(f"NASDAQ:{r['Ticker']}\n")
print(f"  📋 TradingView → {tv}")
if results:
    print(f"\n  📋 Tickers: {', '.join(r['Ticker'] for r in results)}")

# ── Email with CSV attached ───────────────────────────────────
EMAIL_COLS = ["Ticker","Price","Score","Candle_Type","Entry_Price","Stop_Loss","Risk_%",
              "Target_1_EMA8","T1_R","Target_2_SMA21","T2_R","Signal_Date"]

def _fmt_cell(col, v):
    try:
        if col in ("Price","Entry_Price","Stop_Loss","Target_1_EMA8","Target_2_SMA21"): return f"${float(v):.2f}"
        if col in ("T1_R","T2_R"): return f"{float(v):.1f}R"
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
                if c in ("Target_1_EMA8","Target_2_SMA21"): sty = "color:#2563eb;font-weight:600"
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
<h1 style="margin:0;color:#60a5fa;font-size:19px">📊 Daily Red Doji / Hammer Below JMA → Reclaim</h1>
<p style="margin:6px 0 0;color:#94a3b8;font-size:12px">{datetime.today().strftime('%Y-%m-%d %H:%M UTC')} · {cnt} match{'es' if cnt != 1 else ''}
 — yesterday a red doji or hammer closed below the JMA, today closed above its high and above the JMA · T1 = EMA{CFG['ema_period']}, T2 = SMA{CFG['sma_period']}</p>
</td></tr>
<tr><td style="padding:12px 26px;background:#0b1220"><p style="margin:0;color:#cbd5e1;font-size:12px">
🔍 FUNNEL: {fc['tickers_checked']} checked → {fc['passed_step1_red_doji_or_hammer']} red doji/hammer
→ {fc['passed_step2_red_below_jma']} + below JMA → {fc['passed_step3_close_above_red_high']} + close above red high
→ {fc['passed_step4_close_above_jma']} + above JMA
→ <b style="color:#facc15">{fc['passed_step5_targets_above']} + targets above entry = full match</b></p></td></tr>
<tr><td style="padding:16px"><div style="overflow-x:auto;border:1px solid #e2e8f0;border-radius:8px">
<table style="border-collapse:collapse;width:100%"><thead><tr>{th}</tr></thead><tbody>{rows}</tbody></table></div>
{tick_html}
<p style="font-size:11px;color:#64748b;margin:10px 0 0">Entry = the signal day's close · Stop = the red candle's low ·
Target 1 = EMA{CFG['ema_period']}, Target 2 = SMA{CFG['sma_period']} on the signal day (R = reward ÷ risk). Flags say if a target or the stop was already touched since. 📎 Full results (with Quantity for a fixed max loss) attached as CSV.</p>
</td></tr>
<tr><td style="background:#f8fafc;padding:10px;text-align:center;color:#94a3b8;font-size:10px">
⚠️ Not financial advice · Auto-generated by GitHub Actions</td></tr>
</table></td></tr></table></body></html>"""

        plain = [f"Daily Red Doji / Hammer Below JMA → Reclaim — {datetime.today().strftime('%Y-%m-%d')}",
                 f"{cnt} matches",
                 f"FUNNEL: {fc['tickers_checked']} checked -> {fc['passed_step1_red_doji_or_hammer']} red doji/hammer -> "
                 f"{fc['passed_step2_red_below_jma']} below JMA -> {fc['passed_step3_close_above_red_high']} above red high -> "
                 f"{fc['passed_step4_close_above_jma']} above JMA -> {fc['passed_step5_targets_above']} targets above (=full match)", "="*60]
        for r in rl[:50]:
            plain.append(f"{r['Ticker']:<6} ${r['Price']:.2f}  Score:{r['Score']}  {r['Candle_Type']}  Entry:${r['Entry_Price']:.2f}  "
                         f"SL:${r['Stop_Loss']:.2f}  T1(EMA8):${r['Target_1_EMA8']:.2f} {r['T1_R']}R  "
                         f"T2(SMA21):${r['Target_2_SMA21']:.2f} {r['T2_R']}R  {r['Signal_Date']}")
        if rl:
            plain += ["", "Tickers: " + ", ".join(r["Ticker"] for r in rl)]
        else:
            plain.append("No matches today")
        plain.append("\nFull results in CSV attachment.")

        subj = (f"📊 Daily Doji/Hammer JMA Reclaim — "
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

# ── Charts for top 5: last 40 daily candles + JMA, EMA8, SMA21, entry/stop/targets ─
if results:
    top = results[:5]
    fig, axes = plt.subplots(len(top), 1, figsize=(14, 4.2*len(top)), facecolor="#0f172a")
    if len(top) == 1: axes = [axes]
    for ax, r in zip(axes, top):
        df = r["_df"]; k = df.index.get_loc(pd.Timestamp(r["Signal_Date"]))
        lo = max(0, k - 34); seg = df.iloc[lo:]
        x = np.arange(len(seg))
        ax.set_facecolor("#0f172a")
        cols = ["#22c55e" if c > o else "#ef4444" for o, c in zip(seg["Open"], seg["Close"])]
        ax.vlines(x, seg["Low"], seg["High"], color=cols, lw=1)
        ax.bar(x, (seg["Close"] - seg["Open"]).abs(), bottom=np.minimum(seg["Open"], seg["Close"]),
               color=cols, width=0.6)
        ax.plot(x, r["_jma"].iloc[lo:].values, color="#f97316", lw=1.4, label="JMA")
        ax.plot(x, r["_ema"].iloc[lo:].values, color="#a78bfa", lw=1.2, label="EMA8 (T1)")
        ax.plot(x, r["_sma"].iloc[lo:].values, color="#22d3ee", lw=1.2, label="SMA21 (T2)")
        for lvl, c_, lab in ((r["Entry_Price"], "#e2e8f0", "Entry"), (r["Stop_Loss"], "#ef4444", "Stop"),
                             (r["Target_1_EMA8"], "#a78bfa", "T1"), (r["Target_2_SMA21"], "#22d3ee", "T2")):
            ax.axhline(lvl, color=c_, ls=":", lw=0.8)
            ax.text(len(seg) - 0.5, lvl, f" {lab} {lvl:.2f}", color=c_, fontsize=7, va="center")
        ax.axvline(k - lo, color="#facc15", ls="--", lw=0.8)
        ax.legend(loc="upper left", facecolor="#1e293b", labelcolor="#e2e8f0", fontsize=7)
        step = max(1, len(seg) // 8)
        ax.set_xticks(x[::step]); ax.set_xticklabels([d.strftime("%m/%d") for d in seg.index[::step]], fontsize=7)
        ax.set_title(f"{r['Ticker']} | ${r['Price']:.2f} | Score {r['Score']} | {r['Candle_Type']} | "
                     f"T1 {r['T1_R']}R  T2 {r['T2_R']}R", color="#e2e8f0", fontsize=9, fontweight="bold")
        ax.tick_params(colors="#94a3b8", labelsize=7)
        for sp in ax.spines.values(): sp.set_edgecolor("#1e3a5f")
        ax.grid(color="#1e3a5f", ls="--", lw=0.5, alpha=0.6)
    plt.suptitle(f"Daily Red Doji / Hammer Below JMA → Reclaim · "
                 f"{datetime.today().strftime('%Y-%m-%d')}", color="#60a5fa", fontsize=12, fontweight="bold")
    plt.tight_layout()
    cp = os.path.join(out_dir, f"daily_doji_hammer_jma_reclaim_chart_{ts}.png")
    plt.savefig(cp, dpi=130, bbox_inches="tight", facecolor="#0f172a")
    plt.close()
    print(f"  📊 Chart → {cp}")

print(f"""
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  📋 SIGNAL (all required; signal day = one of the last {CFG['signal_lookback_days']} sessions)
  1) Day before: red DOJI (body <= {CFG['doji_max_body_range_pct']:.0f}% of range and <= {CFG['doji_max_body_atr']} x ATR{CFG['atr_period']})
     or red HAMMER (lower wick >= {CFG['hammer_min_wick_body']:.0f} x body, upper wick <= {CFG['hammer_max_upper_pct']:.0f}% of range)
  2) That red candle closed below JMA({CFG['jma_period']},{CFG['jma_phase']},{CFG['jma_power']})
  3) Signal day closed above the red candle's high
  4) Signal day closed above the JMA
  5) EMA{CFG['ema_period']} and SMA{CFG['sma_period']} are both above the entry (targets ahead)
  OUTPUT: Entry = signal day close, Stop = red candle low,
          Target 1 = EMA{CFG['ema_period']}, Target 2 = SMA{CFG['sma_period']} (R = reward / risk)
  ⚙️  If 0 results, check the FUNNEL: relax doji_max_body_range_pct 35 → 45,
     hammer_min_wick_body 2 → 1.5, or set require_targets_above False
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
""")
