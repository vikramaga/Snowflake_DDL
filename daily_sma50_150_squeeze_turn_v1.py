# ============================================================
# NASDAQ — SMA50 / SMA150 Squeeze → SMA50 Turns Up (Daily) (v1)
# ============================================================
#
# Finds stocks whose SMA50 came very close to the SMA150 and stayed
# glued to it for at least two weeks, and has now started to lift
# away upward, with price above the EMA8.
#
# SIGNAL (all required; checked on the latest completed session).
# gap% = (SMA50 - SMA150) / SMA150 x 100
#
#   1. SQUEEZE: |gap%| <= squeeze_max_gap_pct on EVERY one of the
#      squeeze_days sessions ending turn_days sessions before the
#      signal day (the two "glued" weeks before the turn), and the gap
#      moved no more than squeeze_max_gap_change points over them (so
#      SMA50 stayed glued instead of just passing through the band).
#
#   2. TURNING UP: on the signal day SMA50 is above SMA150, the gap
#      has widened on each of the last turn_days sessions, and SMA50
#      itself is higher than turn_days sessions ago.
#
#   3. PRICE ABOVE EMA8: the signal day closes above the EMA8.
#
# OUTPUT: Entry_Price = signal day's close; Stop_Loss = the lower of
# SMA50 / SMA150 minus stop_atr_buffer x ATR(14) (below the MA
# cluster), or close - 1.5 x ATR(14) if price is under the cluster.
# The squeeze, turn and stop definitions are reasonable defaults, not
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
    # history_days vs _clean(min_bars=200): 500 calendar days -> ~358
    # business days (checked with pd.bdate_range), well above 200, and
    # enough for SMA150 plus the squeeze window.
    "history_days"            : 500,

    "fast_sma"                : 50,
    "slow_sma"                : 150,
    "ema_period"              : 8,

    # ── Squeeze (default definition) ─────────────────────────────
    "squeeze_max_gap_pct"     : 1.5,   # SMA50 within 1.5% of SMA150 ...
    "squeeze_days"            : 10,    # ... on every one of 10 sessions (2 weeks)
    "squeeze_max_gap_change"  : 0.75,  # and the gap moved <= 0.75 pts over those sessions
                                       # (stayed glued, not just passing through the band)

    # ── Turning up (default definition) ──────────────────────────
    "turn_days"               : 3,     # gap widened on each of the last 3 sessions

    "signal_lookback_days"    : 1,     # latest session only, so "price above EMA8" is current

    # ── Stop (default) ───────────────────────────────────────────
    "atr_period"              : 14,
    "stop_atr_buffer"         : 0.5,   # stop = lower of SMA50/SMA150 - 0.5 x ATR

    # ── Filters ─────────────────────────────────────────────────
    "min_avg_volume"          : 200_000,
    "min_price"               : 5.0,

    "batch_size"              : 50,
    "batch_sleep"             : 1.5,
}

# ── Indicators ───────────────────────────────────────────────
def calc_atr(df, period):
    h, l, c = df["High"], df["Low"], df["Close"]
    pc = c.shift(1)
    tr = pd.concat([h - l, (h - pc).abs(), (l - pc).abs()], axis=1).max(axis=1)
    return tr.rolling(period).mean()

def check_pattern_at(close, sma_f, sma_s, ema, i, cfg):
    """
    SINGLE SOURCE OF TRUTH. Checks signal day i. Computes EVERY step
    (no short-circuit) so the funnel shows where tickers drop out.
    Returns (passed, details); details is {} only if data is too short.
    """
    sq, td = cfg["squeeze_days"], cfg["turn_days"]
    lo = i - td - sq + 1                       # first squeeze day
    if lo < 1 or i >= len(close):
        return False, {}
    f = sma_f.iloc[lo - 1:i + 1].values.astype(float)
    s = sma_s.iloc[lo - 1:i + 1].values.astype(float)
    if not (np.all(np.isfinite(f)) and np.all(np.isfinite(s)) and np.all(s > 0)):
        return False, {}
    gap = (f - s) / s * 100                    # gap[0] = day lo-1 ... gap[-1] = day i
    sq_gap = gap[1:1 + sq]                     # the squeeze_days sessions before the turn
    turn_gap = gap[-(td + 1):]                 # day i-td .. day i

    # Step 1 — squeeze: every squeeze day within the band
    sq_max = float(np.max(np.abs(sq_gap)))
    sq_change = float(np.max(sq_gap) - np.min(sq_gap))
    step1 = bool(sq_max <= cfg["squeeze_max_gap_pct"] and sq_change <= cfg["squeeze_max_gap_change"])

    # Step 2 — turning up: SMA50 above SMA150, gap widening each day, SMA50 rising
    widening = bool(np.all(np.diff(turn_gap) > 0))
    f_now, f_then = float(sma_f.iloc[i]), float(sma_f.iloc[i - td])
    step2 = bool(gap[-1] > 0 and widening and f_now > f_then)

    # Step 3 — price above EMA8
    c, e = float(close.iloc[i]), float(ema.iloc[i])
    step3 = bool(np.isfinite(e) and c > e)

    passed = step1 and step2 and step3
    return passed, {
        "date": close.index[i], "close": c, "ema8": e,
        "sma_fast": f_now, "sma_slow": float(sma_s.iloc[i]),
        "gap_now": float(gap[-1]), "gap_turn_start": float(turn_gap[0]),
        "sq_max_gap": sq_max, "sq_avg_gap": float(np.mean(np.abs(sq_gap))), "sq_change": sq_change,
        "sma_fast_slope_pct": (f_now - f_then) / f_then * 100 if f_then > 0 else 0.0,
        "step1": step1, "step2": step2, "step3": step3,
    }

# ── Diagnostic funnel (per ticker: did ANY of the lookback days pass
#    steps 1..k cumulatively) ──────────────────────────────────────────
FUNNEL_COUNTS = {
    "tickers_checked": 0,
    "passed_step1_squeeze": 0,
    "passed_step2_turning_up": 0,
    "passed_step3_above_ema8": 0,
}
_FUNNEL_KEYS = list(FUNNEL_COUNTS)[1:]

def find_signals(close, sma_f, sma_s, ema, cfg):
    hits, best = [], [False] * 3
    n = len(close)
    any_checked = False
    for i in range(max(0, n - cfg["signal_lookback_days"]), n):
        ok, d = check_pattern_at(close, sma_f, sma_s, ema, i, cfg)
        if not d: continue
        any_checked = True
        cum = True
        for k in range(3):
            cum = cum and d[f"step{k+1}"]
            best[k] |= cum
        if ok:
            d["i"] = i
            hits.append(d)
    if any_checked:
        FUNNEL_COUNTS["tickers_checked"] += 1
        for k, key in enumerate(_FUNNEL_KEYS):
            FUNNEL_COUNTS[key] += best[k]
    return hits

def analyze_sma50_150_squeeze_turn(sym, df_daily):
    if df_daily is None: return None
    price   = float(df_daily["Close"].iloc[-1])
    avg_vol = float(df_daily["Volume"].tail(20).mean())
    if price   < CFG["min_price"]:      return None
    if avg_vol < CFG["min_avg_volume"]: return None
    if len(df_daily) < CFG["slow_sma"] + CFG["squeeze_days"] + CFG["turn_days"] + 5: return None

    close = df_daily["Close"]
    sma_f = close.rolling(CFG["fast_sma"]).mean()
    sma_s = close.rolling(CFG["slow_sma"]).mean()
    ema = close.ewm(span=CFG["ema_period"], adjust=False).mean()
    atr = calc_atr(df_daily, CFG["atr_period"])

    hits = find_signals(close, sma_f, sma_s, ema, CFG)
    if not hits: return None
    sig = hits[-1]   # most recent hit wins
    i = sig["i"]

    entry = sig["close"]
    a = float(atr.iloc[i])
    if not np.isfinite(a) or a <= 0: return None
    stop = min(sig["sma_fast"], sig["sma_slow"]) - CFG["stop_atr_buffer"] * a
    stop_basis = "MA cluster"
    if stop >= entry:
        stop, stop_basis = entry - 1.5 * a, "1.5 ATR"
    if stop <= 0: return None
    risk_pct = (entry - stop) / entry * 100
    days_ago = len(close) - 1 - i
    sma_s_slope = (float(sma_s.iloc[i]) - float(sma_s.iloc[i - 10])) / float(sma_s.iloc[i - 10]) * 100
    above_ema_pct = (entry - sig["ema8"]) / sig["ema8"] * 100

    score = 30
    score += max(0, 25 - sig["sq_avg_gap"] * 15)                    # tighter squeeze
    score += min(20, sig["sma_fast_slope_pct"] * 20)                 # SMA50 lifting
    score += 10 if sma_s_slope >= 0 else 0                           # SMA150 flat/rising
    score += max(0, 15 - risk_pct * 1.5)                             # tight stop
    score = round(min(100, max(0, score)))

    flags = [f"Squeeze {CFG['squeeze_days']}d max gap {sig['sq_max_gap']:.2f}%",
             f"Gap now +{sig['gap_now']:.2f}%",
             f"SMA150 {'rising' if sma_s_slope >= 0 else 'falling'} ({sma_s_slope:+.2f}%/10d)"]
    if stop_basis != "MA cluster": flags.append("Price under MA cluster, ATR stop")

    return {
        "Score": score, "Price": round(price, 2),
        "Entry_Price": round(entry, 2), "Stop_Loss": round(stop, 2), "Risk_%": round(risk_pct, 1),
        "Stop_Basis": stop_basis,
        "SMA50": round(sig["sma_fast"], 2), "SMA150": round(sig["sma_slow"], 2),
        "Gap_Now_%": round(sig["gap_now"], 2), "Squeeze_Max_Gap_%": round(sig["sq_max_gap"], 2),
        "Squeeze_Avg_Gap_%": round(sig["sq_avg_gap"], 2), "Squeeze_Gap_Change": round(sig["sq_change"], 2),
        "SMA50_Slope_%": round(sig["sma_fast_slope_pct"], 2), "SMA150_Slope_10d_%": round(sma_s_slope, 2),
        "EMA8": round(sig["ema8"], 2), "Above_EMA8_%": round(above_ema_pct, 1),
        "Signal_Date": sig["date"].strftime("%Y-%m-%d"), "Days_Since_Signal": int(days_ago),
        "Recent_Signal_Count": len(hits),
        "Flags": " | ".join(flags),
        "_df": df_daily, "_sma_f": sma_f, "_sma_s": sma_s, "_ema": ema, "_i": i,
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
                        df = _clean(df, min_bars=200)
                        if df is not None: out[sym] = df
                    except Exception: pass
            elif len(symbols) == 1:
                df = _clean(raw, min_bars=200)
                if df is not None: out[symbols[0]] = df
    except Exception: pass
    for sym in [s for s in symbols if s not in out]:
        for _ in range(2):
            try:
                df = yf.Ticker(sym).history(
                    start=start.strftime("%Y-%m-%d"),
                    end=end.strftime("%Y-%m-%d"),
                    auto_adjust=True, actions=False)
                df = _clean(df, min_bars=200)
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
        r = analyze_sma50_150_squeeze_turn(sym, daily_map[sym])
        if r is None: continue
        r["Ticker"] = sym
        results.append(r)
        print(f"  ✅ {sym:<6} ${r['Price']:.2f}  Score {r['Score']}  squeeze max {r['Squeeze_Max_Gap_%']}%  "
              f"gap now +{r['Gap_Now_%']}%  {r['Signal_Date']}")
    except Exception: pass

fc = FUNNEL_COUNTS
print(f"\n{'━'*65}\n  SCAN COMPLETE — {len(results)} matches from {got_daily} tickers with data")
print(f"  🔍 FUNNEL (any of the last {CFG['signal_lookback_days']} sessions)")
print(f"  Tickers checked                               : {fc['tickers_checked']}")
print(f"  Step 1 — SMA50 glued within {CFG['squeeze_max_gap_pct']}% of SMA150 for {CFG['squeeze_days']}d : {fc['passed_step1_squeeze']}")
print(f"  Step 2 — + SMA50 now lifting away upward       : {fc['passed_step2_turning_up']}")
print(f"  Step 3 — + close above EMA8                    : {fc['passed_step3_above_ema8']}  (= full pattern)")
print(f"{'━'*65}")

results.sort(key=lambda x: x["Score"], reverse=True)

# ── Save CSV + TradingView list ───────────────────────────────
ts      = datetime.today().strftime("%Y%m%d_%H%M")
out_dir = os.environ.get("GITHUB_WORKSPACE", os.getcwd())
COLS = ["Ticker","Price","Score","Entry_Price","Stop_Loss","Risk_%","Stop_Basis",
        "SMA50","SMA150","Gap_Now_%","Squeeze_Max_Gap_%","Squeeze_Avg_Gap_%","Squeeze_Gap_Change",
        "SMA50_Slope_%","SMA150_Slope_10d_%","EMA8","Above_EMA8_%",
        "Signal_Date","Days_Since_Signal","Recent_Signal_Count","Flags"]
df_out = pd.DataFrame([{k: v for k, v in r.items() if not k.startswith("_")}
                       for r in results]) if results else pd.DataFrame(columns=COLS)
df_out = df_out[[c for c in COLS if c in df_out.columns]].reset_index(drop=True)

fpath = os.path.join(out_dir, f"daily_sma50_150_squeeze_turn_{ts}.csv")
try:
    from position_sizing import add_position_size
    df_out = add_position_size(df_out)
except Exception as _ps_err:
    print(f"  ⚠️  Position sizing skipped: {_ps_err}")
df_out.to_csv(fpath, index=False)
print(f"\n  💾 CSV → {fpath}")
tv = os.path.join(out_dir, f"tv_daily_sma50_150_squeeze_turn_{ts}.txt")
with open(tv, "w") as f:
    f.write(f"###SMA50 SMA150 Squeeze Turn {datetime.today().strftime('%Y-%m-%d')}\n")
    for r in results: f.write(f"NASDAQ:{r['Ticker']}\n")
print(f"  📋 TradingView → {tv}")
if results:
    print(f"\n  📋 Tickers: {', '.join(r['Ticker'] for r in results)}")

# ── Email with CSV attached ───────────────────────────────────
EMAIL_COLS = ["Ticker","Price","Score","Entry_Price","Stop_Loss","Risk_%","SMA50","SMA150",
              "Squeeze_Max_Gap_%","Gap_Now_%","SMA150_Slope_10d_%","Signal_Date"]

def _fmt_cell(col, v):
    try:
        if col in ("Price","Entry_Price","Stop_Loss","SMA50","SMA150"): return f"${float(v):.2f}"
        if col in ("Squeeze_Max_Gap_%","Gap_Now_%","SMA150_Slope_10d_%"): return f"{float(v):+.2f}%"
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
<h1 style="margin:0;color:#60a5fa;font-size:19px">📊 SMA50 / SMA150 Squeeze → SMA50 Turns Up (Daily)</h1>
<p style="margin:6px 0 0;color:#94a3b8;font-size:12px">{datetime.today().strftime('%Y-%m-%d %H:%M UTC')} · {cnt} match{'es' if cnt != 1 else ''}
 — SMA{CFG['fast_sma']} within {CFG['squeeze_max_gap_pct']}% of SMA{CFG['slow_sma']} for {CFG['squeeze_days']} sessions, now lifting away upward, close above EMA{CFG['ema_period']}</p>
</td></tr>
<tr><td style="padding:12px 26px;background:#0b1220"><p style="margin:0;color:#cbd5e1;font-size:12px">
🔍 FUNNEL: {fc['tickers_checked']} checked → {fc['passed_step1_squeeze']} SMA50/SMA150 squeeze
→ {fc['passed_step2_turning_up']} + SMA50 turning up
→ <b style="color:#facc15">{fc['passed_step3_above_ema8']} + close above EMA8 = full match</b></p></td></tr>
<tr><td style="padding:16px"><div style="overflow-x:auto;border:1px solid #e2e8f0;border-radius:8px">
<table style="border-collapse:collapse;width:100%"><thead><tr>{th}</tr></thead><tbody>{rows}</tbody></table></div>
{tick_html}
<p style="font-size:11px;color:#64748b;margin:10px 0 0">Entry = the signal day's close · Stop = below the SMA50/SMA150 cluster
(lower of the two − {CFG['stop_atr_buffer']} × ATR{CFG['atr_period']}; defaults). Gap = (SMA50 − SMA150) ÷ SMA150. 📎 Full results (with Quantity for a fixed max loss) attached as CSV.</p>
</td></tr>
<tr><td style="background:#f8fafc;padding:10px;text-align:center;color:#94a3b8;font-size:10px">
⚠️ Not financial advice · Auto-generated by GitHub Actions</td></tr>
</table></td></tr></table></body></html>"""

        plain = [f"SMA50 / SMA150 Squeeze → SMA50 Turns Up (Daily) — {datetime.today().strftime('%Y-%m-%d')}",
                 f"{cnt} matches",
                 f"FUNNEL: {fc['tickers_checked']} checked -> {fc['passed_step1_squeeze']} squeeze -> "
                 f"{fc['passed_step2_turning_up']} turning up -> {fc['passed_step3_above_ema8']} above EMA8 (=full match)", "="*60]
        for r in rl[:50]:
            plain.append(f"{r['Ticker']:<6} ${r['Price']:.2f}  Score:{r['Score']}  Entry:${r['Entry_Price']:.2f}  "
                         f"SL:${r['Stop_Loss']:.2f} ({r['Risk_%']}%)  SMA50 ${r['SMA50']:.2f} / SMA150 ${r['SMA150']:.2f}  "
                         f"squeeze max {r['Squeeze_Max_Gap_%']}%  gap now +{r['Gap_Now_%']}%  {r['Signal_Date']}")
        if rl:
            plain += ["", "Tickers: " + ", ".join(r["Ticker"] for r in rl)]
        else:
            plain.append("No matches today")
        plain.append("\nFull results in CSV attachment.")

        subj = (f"📊 SMA50/150 Squeeze Turn Up — "
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

# ── Charts for top 5: last 90 daily closes + SMA50, SMA150, EMA8, gap % below ─
if results:
    top = results[:5]
    fig, axes = plt.subplots(len(top), 2, figsize=(16, 4.2*len(top)), facecolor="#0f172a",
                             gridspec_kw={"width_ratios": [2.4, 1]})
    if len(top) == 1: axes = axes.reshape(1, 2)
    for row, r in zip(axes, top):
        ax, ax_g = row
        df, i = r["_df"], r["_i"]
        lo = max(0, i - 89); seg = df.iloc[lo:]
        x = np.arange(len(seg))
        f_ = r["_sma_f"].iloc[lo:].values; s_ = r["_sma_s"].iloc[lo:].values
        ax.set_facecolor("#0f172a")
        ax.plot(x, seg["Close"].values, color="#60a5fa", lw=1.2, label="Close")
        ax.plot(x, r["_ema"].iloc[lo:].values, color="#a78bfa", lw=1.0, label="EMA8")
        ax.plot(x, f_, color="#facc15", lw=1.6, label="SMA50")
        ax.plot(x, s_, color="#f97316", lw=1.6, label="SMA150")
        sq0 = i - lo - CFG["turn_days"] - CFG["squeeze_days"] + 1
        ax.axvspan(sq0, i - lo - CFG["turn_days"], color="#94a3b8", alpha=0.12)
        ax.axvline(i - lo, color="#22c55e", ls="--", lw=0.9)
        ax.axhline(r["Stop_Loss"], color="#ef4444", ls=":", lw=0.8)
        ax.legend(loc="upper left", facecolor="#1e293b", labelcolor="#e2e8f0", fontsize=7)
        ax.set_title(f"{r['Ticker']} | {r['Price']:.2f} USD | Score {r['Score']} | stop {r['Stop_Loss']:.2f}",
                     color="#e2e8f0", fontsize=9, fontweight="bold")
        gap = (f_ - s_) / s_ * 100
        ax_g.set_facecolor("#0f172a")
        ax_g.plot(x, gap, color="#facc15", lw=1.4)
        ax_g.axhspan(-CFG["squeeze_max_gap_pct"], CFG["squeeze_max_gap_pct"], color="#94a3b8", alpha=0.15)
        ax_g.axhline(0, color="#475569", lw=0.6)
        ax_g.axvline(i - lo, color="#22c55e", ls="--", lw=0.9)
        ax_g.set_title(f"SMA50 vs SMA150 gap % (now +{r['Gap_Now_%']}%)", color="#e2e8f0", fontsize=8)
        step = max(1, len(seg) // 6)
        for a_ in (ax, ax_g):
            a_.set_xticks(x[::step]); a_.set_xticklabels([d.strftime("%m/%d") for d in seg.index[::step]], fontsize=7)
            a_.tick_params(colors="#94a3b8", labelsize=7)
            for sp in a_.spines.values(): sp.set_edgecolor("#1e3a5f")
            a_.grid(color="#1e3a5f", ls="--", lw=0.5, alpha=0.6)
    plt.suptitle(f"SMA50 / SMA150 Squeeze → SMA50 Turns Up · "
                 f"{datetime.today().strftime('%Y-%m-%d')}", color="#60a5fa", fontsize=12, fontweight="bold")
    plt.tight_layout()
    cp = os.path.join(out_dir, f"daily_sma50_150_squeeze_turn_chart_{ts}.png")
    plt.savefig(cp, dpi=130, bbox_inches="tight", facecolor="#0f172a")
    plt.close()
    print(f"  📊 Chart → {cp}")

print(f"""
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  📋 SIGNAL (all required; signal day = one of the last {CFG['signal_lookback_days']} sessions)
  gap% = (SMA{CFG['fast_sma']} - SMA{CFG['slow_sma']}) / SMA{CFG['slow_sma']} x 100
  1) SQUEEZE: |gap%| <= {CFG['squeeze_max_gap_pct']}% on each of {CFG['squeeze_days']} sessions before the turn,
     and the gap moved <= {CFG['squeeze_max_gap_change']} pts over them (glued, not passing through)
  2) TURNING UP: SMA{CFG['fast_sma']} above SMA{CFG['slow_sma']}, gap widened on each of the last
     {CFG['turn_days']} sessions, and SMA{CFG['fast_sma']} higher than {CFG['turn_days']} sessions ago
  3) Close above EMA{CFG['ema_period']}
  OUTPUT defaults: Entry = signal day close,
     Stop = lower of SMA{CFG['fast_sma']}/SMA{CFG['slow_sma']} - {CFG['stop_atr_buffer']} x ATR{CFG['atr_period']}
  ⚙️  Too few results: squeeze_max_gap_pct 1.5 → 2.5, turn_days 3 → 2
     Too many: squeeze_max_gap_pct 1.5 → 1.0, squeeze_days 10 → 15
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
""")
