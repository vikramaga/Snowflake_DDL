# ============================================================
# NASDAQ / NYSE - Multi-Timeframe: Daily HH/HL uptrend -> 1H retracement ->
#                 5-min CHoCH + FVG retest entry   (v1)
# ============================================================
#
# From Vik's strategy (2026-11-15):
#   1. DAILY UPTREND: the last two daily swing highs AND the last two daily swing
#      lows are rising (HH + HL; swing = highest/lowest of 3 bars each side), the
#      close is above the latest higher low and above SMA50
#   2. 1H RETRACEMENT: the highest 1-hour high of the last 35 hourly bars is at
#      least 2 bars old; price has pulled back 23.6%-78.6% of the 1H up-leg into it
#      (and at least 1 hourly ATR) without breaking the daily higher low
#   3. 5-MIN CHoCH: after the pullback low (lowest 5-min low since the 1H high), a
#      5-min close above the last 5-min lower swing high (swing = 2 bars each side)
#   4. FVG: the up-move from the pullback low through the CHoCH left a bullish
#      fair value gap (candle k-1 high < candle k+1 low)
#   5. SUPPORT + ENTRY: price came back into the FVG (low <= FVG top) without a
#      close below its bottom, then a 5-min candle closed above that retest
#      candle's high = ENTRY (must be in the latest session)
#
# Stages reported: Triggered (all 5) | Retest holding (1-4 + retest, no move up yet)
#                  | CHoCH, waiting retest (1-4) | Armed (1-2: watch on 5-min next session)
# Entry = trigger candle close; Stop = the 5-min pullback low (the swing the CHoCH
# came from) - 0.1%; Aggressive_Stop = min(FVG bottom, retest low) - 0.1%;
# T1 = the 1H swing high (2R if already above it); T2 = 2R. After-entry outcome
# (high / low / last price, T1 or stop hit) is shown when run after the session.
# Data: Yahoo daily (400 days), 1-hour (60 days), 5-min (5 days); regular hours.
# Creates TradingView watchlist "MTF FVG - YYYY-MM-DD" (IST). Thresholds in CFG.
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


# -- CONFIG
CFG = {
    "history_days"         : 400,
    "min_price"            : 5.0,
    "min_avg_volume"       : 500_000,   # intraday setups need liquid names
    "daily_wing"           : 3,
    "daily_lookback"       : 120,
    "h1_window"            : 35,        # 1H swing high = highest high of the last 35 hourly bars
    "h1_min_age"           : 2,         # ... at least 2 hourly bars old
    "h1_leg_lookback"      : 35,        # up-leg low = lowest low of the 35 hourly bars before the high
    "retrace_min"          : 0.236,
    "retrace_max"          : 0.786,
    "retrace_min_atr"      : 1.0,
    "m5_wing"              : 2,
    "stop_buffer"          : 0.001,
    "armed_cap"            : 150,       # most armed names kept in the email / list
    "tv_list_prefix"       : "MTF FVG",
    "batch_size"           : 50,
    "batch_sleep"          : 1.5,
}

def _pivots(h, l, wing, start, end):
    """Confirmed swing highs / lows between start and end (inclusive), wing bars each side."""
    sh, sl = [], []
    for j in range(max(start, wing), min(end, len(h) - 1 - wing) + 1):
        if h[j] == h[j - wing:j + wing + 1].max(): sh.append(j)
        if l[j] == l[j - wing:j + wing + 1].min(): sl.append(j)
    return sh, sl

def daily_trend(df, cfg):
    h, l, c = (df[k].values.astype(float) for k in ("High", "Low", "Close"))
    n = len(df)
    s50 = df["Close"].rolling(50).mean().values
    sh, sl = _pivots(h, l, cfg["daily_wing"], n - cfg["daily_lookback"], n - 1)
    if len(sh) < 2 or len(sl) < 2: return None
    hh = h[sh[-1]] > h[sh[-2]]; hl = l[sl[-1]] > l[sl[-2]]
    ok = bool(hh and hl and c[-1] > l[sl[-1]] and np.isfinite(s50[-1]) and c[-1] > s50[-1])
    return {"ok": ok, "hl": float(l[sl[-1]]), "hl_date": df.index[sl[-1]], "hh": float(h[sh[-1]]),
            "hh_date": df.index[sh[-1]], "sma50": float(s50[-1])}

def h1_retrace(d1, cfg, daily_hl):
    h, l, c = (d1[k].values.astype(float) for k in ("High", "Low", "Close"))
    n = len(d1)
    if n < cfg["h1_window"] + cfg["h1_leg_lookback"] + 15: return None
    pc = np.concatenate([[np.nan], c[:-1]])
    atr = pd.Series(np.nanmax(np.vstack([h - l, abs(h - pc), abs(l - pc)]), axis=0)).rolling(14).mean().values
    w0 = n - cfg["h1_window"]
    hb = w0 + int(np.argmax(h[w0:]))
    age = n - 1 - hb
    if age < cfg["h1_min_age"]: return {"ok": False}
    lo0 = max(0, hb - cfg["h1_leg_lookback"])
    leg_low = float(l[lo0:hb].min()); H = float(h[hb])
    pull_low = float(l[hb + 1:].min())
    leg = H - leg_low
    if leg <= 0: return {"ok": False}
    retr = (H - pull_low) / leg
    ok = bool(cfg["retrace_min"] <= retr <= cfg["retrace_max"] and (H - pull_low) >= cfg["retrace_min_atr"] * atr[hb]
              and pull_low > daily_hl)
    return {"ok": ok, "high": H, "high_time": d1.index[hb], "leg_low": leg_low, "pull_low": pull_low,
            "retrace": float(retr), "atr": float(atr[hb])}

def m5_entry(d5, high_time, cfg):
    """5-min CHoCH -> FVG -> retest -> move up, on bars after the 1H swing high."""
    d = d5[d5.index >= high_time]
    out = {"stage": "Armed"}
    if len(d) < 10: return out
    o, h, l, c = (d[k].values.astype(float) for k in ("Open", "High", "Low", "Close"))
    n, wg = len(d), cfg["m5_wing"]
    m = int(np.argmin(l))
    out.update({"pull_low_5m": float(l[m]), "pull_low_time": d.index[m]})
    sh = []
    for w in (wg, 1):                                  # fall back to 1-bar swings in a smooth decline
        sh = [j for j in _pivots(h, l, w, 0, m - 1)[0] if j + w < m]
        if sh: break
    if not sh: return out
    lh = sh[-1]; lh_v = float(h[lh])
    x = next((j for j in range(m + 1, n) if c[j] > lh_v), None)
    if x is None: return out
    gaps = [k for k in range(m + 1, min(x + 2, n - 1)) if l[k + 1] > h[k - 1]]
    out.update({"stage": "CHoCH, no FVG", "choch_time": d.index[x], "choch_level": lh_v})
    if not gaps: return out
    k = gaps[-1]; fb, ft = float(h[k - 1]), float(l[k + 1])
    out.update({"stage": "CHoCH, waiting retest", "fvg_bot": fb, "fvg_top": ft, "fvg_time": d.index[k]})
    r = None
    for j in range(k + 2, n):
        if c[j] < fb: return {**out, "stage": "FVG lost"}
        if l[j] <= ft: r = j; break
    if r is None: return out
    out.update({"stage": "Retest holding", "retest_time": d.index[r], "retest_low": float(l[r]), "retest_high": float(h[r])})
    e = None
    for j in range(r + 1, n):
        if c[j] < fb: return {**out, "stage": "FVG lost"}
        if c[j] > h[r]: e = j; break
    if e is None: return out
    stop = min(fb, float(l[r])) * (1 - cfg["stop_buffer"])
    after = d.iloc[e + 1:]
    out.update({"stage": "Triggered", "entry_time": d.index[e], "entry": float(c[e]), "stop": float(stop),
                "stop_struct": float(l[m]) * (1 - cfg["stop_buffer"]),
                "post_high": float(after["High"].max()) if len(after) else np.nan,
                "post_low": float(after["Low"].min()) if len(after) else np.nan,
                "last": float(c[-1]), "last_time": d.index[-1], "_after": after})
    return out

def _intraday(symbols, period, interval):
    out = {}
    for b in range(0, len(symbols), 80):
        batch = symbols[b:b + 80]
        try:
            raw = yf.download(batch, period=period, interval=interval, group_by="ticker", auto_adjust=False,
                              prepost=False, threads=True, progress=False)
        except Exception as e:
            print(f"  intraday batch failed: {e}"); continue
        for s in batch:
            try:
                df = raw[s] if isinstance(raw.columns, pd.MultiIndex) else raw
                df = df.dropna(subset=["Close"])
                if df.index.tz is not None: df.index = df.index.tz_convert("America/New_York")
                if len(df): out[s] = df
            except Exception: pass
        time.sleep(1)
    return out

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

# -- Main scan
print("=" * 65)
print(f"  STEP 3  SCANNING {len(TICKERS)} TICKERS")
print("=" * 65)

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

# -- Stage 1: daily uptrend
up = {}
for sym, df in daily_map.items():
    try:
        if float(df["Close"].iloc[-1]) < CFG["min_price"] or float(df["Volume"].tail(20).mean()) < CFG["min_avg_volume"]:
            continue
        t = daily_trend(df, CFG)
        if t and t["ok"]: up[sym] = t
    except Exception: pass
print(f"\n  Stage 1 - daily HH + HL uptrend: {len(up)}")

# -- Stage 2: 1-hour retracement
h1 = _intraday(sorted(up), "60d", "60m")
arm = {}
for s, d1 in h1.items():
    try:
        r = h1_retrace(d1, CFG, up[s]["hl"])
        if r and r["ok"]: arm[s] = r
    except Exception: pass
print(f"  Stage 2 - 1H retracement in progress: {len(arm)} (1H data for {len(h1)})")

# -- Stage 3: 5-minute CHoCH + FVG + retest entry
m5 = _intraday(sorted(arm), "5d", "5m")
last_session = max((d.index[-1].date() for d in m5.values()), default=None)
print(f"  5-min data for {len(m5)}; latest session {last_session}")
results = []
for s, a in arm.items():
    d5 = m5.get(s)
    try:
        r = m5_entry(d5, a["high_time"], CFG) if d5 is not None else {"stage": "Armed"}
    except Exception:
        r = {"stage": "Armed"}
    if r["stage"] == "Triggered" and r["entry_time"].date() != last_session:
        continue                                          # triggered in an earlier session - stale
    if r["stage"] in ("FVG lost", "CHoCH, no FVG"):
        continue
    t = up[s]
    row = {"Ticker": s, "Stage": r["stage"], "Price": round(float(daily_map[s]["Close"].iloc[-1]), 2),
           "Daily_HL": round(t["hl"], 2), "Daily_HH": round(t["hh"], 2),
           "H1_High": round(a["high"], 2), "H1_High_Time": a["high_time"].strftime("%Y-%m-%d %H:%M"),
           "Retrace_%": round(a["retrace"] * 100, 1), "Pullback_Low": round(a["pull_low"], 2)}
    if "fvg_bot" in r:
        row.update({"FVG_Bottom": round(r["fvg_bot"], 2), "FVG_Top": round(r["fvg_top"], 2),
                    "CHoCH_Level": round(r["choch_level"], 2), "CHoCH_Time": r["choch_time"].strftime("%m-%d %H:%M")})
    if r["stage"] == "Triggered":
        entry, stop, aggr = r["entry"], r["stop_struct"], r["stop"]
        risk = entry - stop
        if risk <= 0: continue
        t1 = a["high"] if a["high"] > entry else entry + 2 * risk
        def _outcome(sl):
            for _, b in r["_after"].iterrows():
                if b["Low"] <= sl: return "stopped"
                if b["High"] >= t1: return "T1 hit"
            return "open"
        outcome = _outcome(stop)
        row.update({"Aggressive_Stop": round(aggr, 2), "Outcome_Aggr": _outcome(aggr),
                    "Reward_R_Aggr": round((t1 - entry) / (entry - aggr), 1) if entry > aggr else None})
        row.update({"Entry_Price": round(entry, 2), "Stop_Loss": round(stop, 2), "Risk_%": round(risk / entry * 100, 2),
                    "Target_1": round(t1, 2), "Reward_R": round((t1 - entry) / risk, 1), "Target_2": round(entry + 2 * risk, 2),
                    "Entry_Time_ET": r["entry_time"].strftime("%Y-%m-%d %H:%M"), "Outcome": outcome,
                    "Last_5m": round(r["last"], 2), "Open_R": round((r["last"] - entry) / risk, 2)})
    results.append(row)

ORDER = {"Triggered": 0, "Retest holding": 1, "CHoCH, waiting retest": 2, "Armed": 3}
results.sort(key=lambda x: (ORDER[x["Stage"]], -x.get("Reward_R", 0) if x["Stage"] == "Triggered" else abs(x["Retrace_%"] - 50)))
armed_all = [x for x in results if x["Stage"] == "Armed"]
results = [x for x in results if x["Stage"] != "Armed"] + armed_all[:CFG["armed_cap"]]
cnt = {k: sum(1 for x in results if x["Stage"] == k) for k in ORDER}
print(f"\n{'=' * 65}\n  SCAN COMPLETE - latest session {last_session}")
print(f"  Daily uptrend {len(up)} -> 1H retracement {len(arm)} -> CHoCH waiting retest {cnt['CHoCH, waiting retest']}"
      f" -> retest holding {cnt['Retest holding']} -> TRIGGERED {cnt['Triggered']}  (armed shown: {cnt['Armed']} of {len(armed_all)})")
for x in results:
    if x["Stage"] == "Triggered":
        print(f"  ENTRY {x['Ticker']:<6} {x['Entry_Time_ET']} ET  {x['Entry_Price']}  SL {x['Stop_Loss']}  T1 {x['Target_1']} ({x['Reward_R']}R)  -> {x['Outcome']}")
print("=" * 65)

# -- Save CSV + TradingView text list
ts      = datetime.today().strftime("%Y%m%d_%H%M")
out_dir = os.environ.get("GITHUB_WORKSPACE", os.getcwd())
COLS = ["Ticker","Stage","Price","Entry_Price","Stop_Loss","Risk_%","Target_1","Reward_R","Target_2","Entry_Time_ET",
        "Outcome","Aggressive_Stop","Reward_R_Aggr","Outcome_Aggr","Last_5m","Open_R","FVG_Bottom","FVG_Top","CHoCH_Level","CHoCH_Time","H1_High","H1_High_Time",
        "Retrace_%","Pullback_Low","Daily_HL","Daily_HH"]
df_out = pd.DataFrame(results) if results else pd.DataFrame(columns=COLS)
df_out = df_out[[c for c in COLS if c in df_out.columns]].reset_index(drop=True)
fpath = os.path.join(out_dir, f"mtf_choch_fvg_{ts}.csv")
try:
    from position_sizing import add_position_size
    df_out = add_position_size(df_out)
    try: __import__("best_picks").rank_best(df_out[df_out.Stage == "Triggered"].reset_index(drop=True) if len(df_out) else df_out)
    except Exception as _bp_err: print(f"  Best 10 skipped: {_bp_err}")
except Exception as _ps_err:
    print(f"  Position sizing skipped: {_ps_err}")
df_out.to_csv(fpath, index=False)
print(f"\n  CSV -> {fpath}")
STAGES = ["Triggered", "Retest holding", "CHoCH, waiting retest", "Armed"]
tv = os.path.join(out_dir, f"tv_mtf_choch_fvg_{ts}.txt")
with open(tv, "w") as f:
    for st in STAGES:
        names = [x["Ticker"] for x in results if x["Stage"] == st]
        if names:
            f.write(f"###{st}\n"); f.writelines(f"NASDAQ:{s}\n" for s in names)
print(f"  TradingView file -> {tv}")

TV_STATUS = "not created"
def _tv_sync():
    global TV_STATUS
    if not results:
        TV_STATUS = "skipped (no matches)"; return
    if not (os.environ.get("TV_SESSIONID") and os.environ.get("TV_SESSIONID_SIGN")):
        TV_STATUS = "skipped (TV_SESSIONID secrets not set)"; return
    from zoneinfo import ZoneInfo
    name = f"{CFG['tv_list_prefix']} - {datetime.now(ZoneInfo('Asia/Kolkata')):%Y-%m-%d}"
    os.environ["TV_WATCHLIST"] = name
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "claude_run", "tools"))
    import tv_watchlist_sync as tvs
    tvs.NAME = name
    s = tvs.session(); lists = tvs.get_lists(s); ex = tvs.exchange_map()
    symbols = []
    for st in STAGES:
        syms = [tvs.tv_symbol(x["Ticker"], ex) for x in results if x["Stage"] == st]
        if syms: symbols.append(f"###{st} ({len(syms)})"); symbols += syms
    tvs.replace_list(s, lists, symbols)
    after = next((x for x in tvs.get_lists(s) if x.get("name") == name), None)
    got = [x for x in (after or {}).get("symbols") or [] if not str(x).startswith("###")]
    TV_STATUS = f"'{name}' - {len(got)} symbols"
try:
    _tv_sync()
except Exception as e:
    TV_STATUS = f"failed ({type(e).__name__}: {e})"
print(f"[TradingView] {TV_STATUS}")

# -- Email
EMAIL_COLS = {"Triggered": ["Ticker","Entry_Time_ET","Entry_Price","Stop_Loss","Risk_%","Target_1","Reward_R","Outcome","Open_R","Aggressive_Stop","Outcome_Aggr","FVG_Bottom","FVG_Top","Retrace_%"],
              "Retest holding": ["Ticker","Price","FVG_Bottom","FVG_Top","CHoCH_Level","CHoCH_Time","H1_High","Retrace_%","Daily_HL"],
              "CHoCH, waiting retest": ["Ticker","Price","FVG_Bottom","FVG_Top","CHoCH_Level","CHoCH_Time","H1_High","Retrace_%","Daily_HL"],
              "Armed": ["Ticker","Price","H1_High","H1_High_Time","Pullback_Low","Retrace_%","Daily_HL","Daily_HH"]}

def _table(rows, cols):
    th = "".join(f'<th style="background:#1e293b;color:#e2e8f0;padding:6px 8px;font-size:11px;white-space:nowrap">{c}</th>' for c in cols)
    body = ""
    for i, r in enumerate(rows):
        bg = "#fff" if i % 2 == 0 else "#f0f9ff"
        body += f'<tr style="background:{bg}">' + "".join(
            f'<td style="padding:5px 8px;font-size:12px;{"font-weight:700" if c == "Ticker" else ""}">{r.get(c, "-")}</td>' for c in cols) + "</tr>"
    return f'<table style="border-collapse:collapse;width:100%;margin-bottom:14px"><thead><tr>{th}</tr></thead><tbody>{body}</tbody></table>'

def _send_email(rl, csv_path):
    import smtplib
    from email.mime.multipart import MIMEMultipart
    from email.mime.text      import MIMEText
    from email.mime.base      import MIMEBase
    from email                import encoders
    gu = _GMAIL_USER; gp = _GMAIL_PASS; et = _EMAIL_TO
    if not (gu and gp and et):
        print("[Email] GMAIL_USER / GMAIL_PASS / EMAIL_TO missing - skipped"); return
    eto = [e.strip() for e in et.split(",") if e.strip()]
    try:
        secs = ""
        for st in STAGES:
            rows = [x for x in rl if x["Stage"] == st]
            title = {"Triggered": f"Entries triggered in the {last_session} session",
                     "Retest holding": "Retest holding the FVG - waiting for a 5-min close above the retest candle",
                     "CHoCH, waiting retest": "5-min CHoCH + FVG done - waiting for price to come back into the FVG",
                     "Armed": f"Armed - daily uptrend + 1H pullback; watch the 5-min chart (top {CFG['armed_cap']})"}[st]
            secs += f'<h3 style="font-size:14px;margin:12px 0 6px">{title} ({len(rows)})</h3>'
            secs += _table(rows, EMAIL_COLS[st]) if rows else '<p style="color:#94a3b8;font-size:12px">none</p>'
        html = f"""<!DOCTYPE html><html><body style="margin:0;background:#f1f5f9;font-family:'Segoe UI',Arial,sans-serif">
<table width="100%" cellpadding="0" cellspacing="0"><tr><td align="center" style="padding:20px 10px">
<table width="100%" style="max-width:960px;background:#fff;border-radius:12px;overflow:hidden">
<tr><td style="background:linear-gradient(135deg,#0f172a,#1e3a5f);padding:20px 26px">
<h1 style="margin:0;color:#60a5fa;font-size:19px">Daily Uptrend &rarr; 1H Pullback &rarr; 5-min CHoCH + FVG Entry</h1>
<p style="margin:6px 0 0;color:#94a3b8;font-size:12px">{datetime.today().strftime('%Y-%m-%d %H:%M UTC')} &middot; session {last_session}
&middot; {len(up)} daily uptrends &rarr; {len(arm)} 1H pullbacks &rarr; {cnt['Triggered']} entries &middot; TradingView: {TV_STATUS}</p></td></tr>
<tr><td style="padding:16px 22px">{secs}
<p style="font-size:11px;color:#64748b">Daily: last two swing highs and lows rising (3 bars each side), close above the higher low and SMA50.
1H: pullback of {CFG['retrace_min']*100:.1f}-{CFG['retrace_max']*100:.1f}% of the up-leg into the highest high of the last {CFG['h1_window']} hourly bars, above the daily higher low.
5-min: close above the last lower swing high after the pullback low (CHoCH), bullish FVG in that move, price back into the FVG without a close below it,
then a close above the retest candle's high = entry. Stop = the 5-min pullback low - 0.1% (Aggressive_Stop = min(FVG bottom, retest low) - 0.1%); T1 = the 1H high (2R if already above). Times are US Eastern.
Outcome = what happened after the entry in the rest of the data (T1 hit / stopped / open).</p></td></tr>
<tr><td style="background:#f8fafc;padding:10px;text-align:center;color:#94a3b8;font-size:10px">Not financial advice &middot; Auto-generated by GitHub Actions</td></tr>
</table></td></tr></table></body></html>"""
        plain = [f"MTF CHoCH + FVG - session {last_session}", f"{len(up)} uptrends -> {len(arm)} 1H pullbacks -> {cnt['Triggered']} entries",
                 f"TradingView: {TV_STATUS}", ""]
        for st in STAGES:
            plain.append(f"{st}: " + ", ".join(x["Ticker"] for x in rl if x["Stage"] == st))
        subj = f"MTF CHoCH+FVG - {cnt['Triggered']} entries, {cnt['Retest holding'] + cnt['CHoCH, waiting retest']} setting up, {len(armed_all)} armed - {datetime.today().strftime('%Y-%m-%d')}"
        msg = MIMEMultipart("mixed"); msg["Subject"] = subj; msg["From"] = gu; msg["To"] = ", ".join(eto)
        alt = MIMEMultipart("alternative"); alt.attach(MIMEText("\n".join(plain), "plain")); alt.attach(MIMEText(html, "html")); msg.attach(alt)
        if csv_path and os.path.exists(csv_path):
            with open(csv_path, "rb") as f:
                part = MIMEBase("application", "octet-stream"); part.set_payload(f.read())
            encoders.encode_base64(part)
            part.add_header("Content-Disposition", f"attachment; filename={os.path.basename(csv_path)}")
            msg.attach(part)
        with smtplib.SMTP_SSL("smtp.gmail.com", 465) as srv:
            srv.login(gu, gp.replace(" ", ""))
            try: __import__("best_picks").attach_to_email(msg)
            except Exception as _bp_err: print(f"  Best 10 not added: {_bp_err}")
            srv.sendmail(gu, eto, msg.as_string())
        print(f"[Email] Sent to {', '.join(eto)} - {subj}")
    except Exception as e:
        print(f"[Email] {type(e).__name__}: {e}")

_send_email(results, fpath)
