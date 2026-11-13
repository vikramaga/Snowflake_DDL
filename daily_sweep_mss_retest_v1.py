# ============================================================
# NASDAQ / NYSE - Daily Liquidity Sweep + MSS + Retest/FVG Entry (v1)
# ============================================================
#
# Smart-money style long setup:
#   1. LIQUIDITY SWEEP: within the last 30 sessions a bar's low went below the
#      latest swing low (lowest low of 3 bars each side, formed up to 60
#      sessions earlier) - the stop-loss hunt - and price closed back above
#      that old low on the same bar or within the next 2 bars
#   2. BULLISH MSS (market structure shift): within 15 sessions of the sweep,
#      a close above the last swing high before the sweep
#   3. SWEEP LOW HELD: no low since the sweep below the sweep low
#   4. FIRST PULLBACK INTO THE ENTRY ZONE TODAY: today's candle trades into
#      the RETEST zone (the broken swing high +/- 0.25 ATR) or into the
#      bullish FVG (fair value gap: candle k-1 high < candle k+1 low) left by
#      the up-move - the first time back: after the MSS move topped out at least
#      0.5 ATR above the zone, no candle came back into it before today
#   5. HOLDING: today's close is at or above the bottom of the entry zone
#
# Entry = today's close; Stop = the sweep low - 0.25 ATR (just below the
# swing low that was swept); Target = the nearest resistance above: the high
# the MSS move reached before pulling back, or an earlier swing high in the
# last 120 sessions, whichever is closer (2R if none). Flagged when the target
# is under 1.5R. Thresholds are reasonable defaults, not explicitly requested;
# all are in CFG.
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
    # history_days vs bars: 400 calendar days -> ~285 business days (pd.bdate_range);
    # the pattern looks back ~30 + 60 + 120 bars at most, the context SMA200 needs 200.
    "history_days"            : 400,
    "min_bars"                : 210,

    "atr_period"              : 14,
    "pivot_wing"              : 3,     # swing low/high = lowest/highest of 3 bars each side
    "sweep_lookback"          : 30,    # the sweep happened within the last 30 sessions
    "swing_max_age"           : 60,    # the swept swing low formed at most 60 sessions before the sweep
    "reclaim_bars"            : 3,     # sweep bar or the next 2 close back above the old low
    "mss_bars"                : 15,    # MSS close above the swing high within 15 sessions of the sweep
    "retest_tol_atr"          : 0.25,  # retest zone = broken swing high +/- 0.25 ATR
    "stop_buffer_atr"         : 0.25,  # stop = sweep low - 0.25 ATR
    "resistance_lookback"     : 120,   # nearest swing high above entry in the last 120 sessions
    "min_reward_r"            : 1.5,   # flag targets under 1.5R
    "signal_lookback_days"    : 1,     # the latest session only (the retest is today)

    "min_avg_volume"          : 200_000,
    "min_price"               : 5.0,

    "batch_size"              : 50,
    "batch_sleep"             : 1.5,
}

# -- Indicators
def calc_indicators(df, cfg):
    h, l, c = (df[k].values.astype(float) for k in ("High", "Low", "Close"))
    pc = np.concatenate([[np.nan], c[:-1]])
    tr = np.nanmax(np.vstack([h - l, np.abs(h - pc), np.abs(l - pc)]), axis=0)
    w = 2 * cfg["pivot_wing"] + 1
    lows, highs = pd.Series(l), pd.Series(h)
    return {
        "atr":  pd.Series(tr).rolling(cfg["atr_period"]).mean().values,
        "pl":   (lows == lows.rolling(w, center=True).min()).values,     # swing lows
        "ph":   (highs == highs.rolling(w, center=True).max()).values,   # swing highs
        "s50":  pd.Series(c).rolling(50).mean().values,
        "s200": pd.Series(c).rolling(200).mean().values,
    }

def _setup(df, ind, i, s, cfg):
    """Steps 1..3 for a candidate sweep bar s, seen from signal day i."""
    o, h, l, c = (df[k].values.astype(float) for k in ("Open", "High", "Low", "Close"))
    wg = cfg["pivot_wing"]
    # most recent swing low that was confirmed before the sweep bar (pivot + wing bars < s)
    lows_before = [j for j in range(max(0, s - cfg["swing_max_age"]), s - wg) if ind["pl"][j]]
    if not lows_before:
        return None
    j0 = lows_before[-1]
    L0 = l[j0]
    took = l[s] < L0
    k_end = min(s + cfg["reclaim_bars"], i + 1)
    reclaimed = took and any(c[k] > L0 for k in range(s, k_end))
    # the sweep low = lowest low from the sweep bar until the reclaim window ends
    sweep_low = float(l[s:k_end].min())
    step1 = bool(took and reclaimed and l[s] == sweep_low)
    # swing high to break: the last confirmed swing high before the sweep, after the swept low
    highs_before = [j for j in range(j0, s - wg + 1) if ind["ph"][j]]
    if not highs_before:
        highs_before = [j for j in range(max(0, j0 - cfg["swing_max_age"]), s - wg + 1) if ind["ph"][j]]
    H, jh = (float(h[highs_before[-1]]), highs_before[-1]) if highs_before else (np.nan, None)
    m = None
    if np.isfinite(H):
        for k in range(s + 1, min(s + 1 + cfg["mss_bars"], i)):      # MSS before today
            if c[k] > H:
                m = k; break
    step2 = bool(m is not None)
    held = bool(m is not None and l[s + 1:i + 1].min() > sweep_low and c[i] > sweep_low)
    step3 = bool(step2 and held)
    return {"s": s, "j0": j0, "L0": float(L0), "sweep_low": sweep_low, "H": H, "jh": jh, "m": m,
            "step1": step1, "step2": step2, "step3": step3}

def _fvg(df, s, m):
    """Most recent bullish fair value gap in the up-move from the sweep to the MSS bar:
    candle k-1 high < candle k+1 low. Returns (bottom, top) or None."""
    h, l = df["High"].values.astype(float), df["Low"].values.astype(float)
    best = None
    for k in range(s + 1, m + 1):
        if k + 1 > m + 1 or k + 1 >= len(h):
            break
        if l[k + 1] > h[k - 1]:
            best = (float(h[k - 1]), float(l[k + 1]))
    return best

def check_pattern_at(df, ind, i, cfg):
    """
    SINGLE SOURCE OF TRUTH. Checks signal day i against every sweep bar in the
    last `sweep_lookback` sessions. Computes EVERY step (no short-circuit) so the
    funnel shows where tickers drop out; reports the candidate that got furthest
    (ties: the deepest sweep low, so the stop sits under the real stop-hunt wick).
    Returns (passed, details); {} if data short.
    """
    if i < cfg["swing_max_age"] + cfg["sweep_lookback"] + 10 or i >= len(df):
        return False, {}
    a = ind["atr"][i]
    if not np.isfinite(a) or a <= 0:
        return False, {}
    o, h, l, c = (df[k].values.astype(float) for k in ("Open", "High", "Low", "Close"))
    best, best_key = None, None
    for s in range(i - cfg["sweep_lookback"], i - 1):
        d = _setup(df, ind, i, s, cfg)
        if d is None:
            continue
        fvg = _fvg(df, s, d["m"]) if d["m"] is not None else None
        lo_rt, hi_rt = (d["H"] - cfg["retest_tol_atr"] * a, d["H"] + cfg["retest_tol_atr"] * a) if np.isfinite(d["H"]) else (np.nan, np.nan)
        in_retest = bool(np.isfinite(lo_rt) and l[i] <= hi_rt and h[i] >= lo_rt)
        in_fvg = bool(fvg is not None and l[i] <= fvg[1] and h[i] >= fvg[0])
        # first return: after the MSS move topped out (at least 0.5 ATR above the zone),
        # no low came back into the zone before today
        zone_top = max([x for x in (hi_rt, fvg[1] if fvg else np.nan) if np.isfinite(x)], default=np.nan)
        first = False
        if d["m"] is not None and np.isfinite(zone_top) and i > d["m"]:
            p = d["m"] + int(np.argmax(h[d["m"]:i]))
            first = bool(h[p] >= zone_top + 0.5 * a and (l[p:i] > zone_top).all())
        step4 = bool(d["step3"] and (in_retest or in_fvg) and first)
        zone_bot = min([x for x in (lo_rt, fvg[0] if fvg else np.nan) if np.isfinite(x)], default=np.nan)
        step5 = bool(np.isfinite(zone_bot) and c[i] >= zone_bot and c[i] > d["sweep_low"])
        d.update({"fvg": fvg, "in_retest": in_retest, "in_fvg": in_fvg, "first_return": first,
                  "zone_bot": zone_bot, "zone_top": zone_top, "step4": step4, "step5": step5})
        depth = 0
        for k in range(1, 6):
            if not d[f"step{k}"]: break
            depth = k
        key = (depth, -d["sweep_low"], s)
        if best_key is None or key > best_key:
            best, best_key = d, key
    if best is None:
        best = {f"step{k}": False for k in range(1, 6)}
    passed = all(best[f"step{k}"] for k in range(1, 6))
    best.update({"date": df.index[i], "i": i, "atr": float(a), "close": float(c[i]), "low": float(l[i])})
    return passed, best

# -- Diagnostic funnel (per ticker, cumulative over the lookback days)
FUNNEL_COUNTS = {
    "tickers_checked": 0,
    "passed_step1_liquidity_sweep": 0,
    "passed_step2_bullish_mss": 0,
    "passed_step3_sweep_low_held": 0,
    "passed_step4_first_retest_or_fvg": 0,
    "passed_step5_holding_the_zone": 0,
}
_FUNNEL_KEYS = list(FUNNEL_COUNTS)[1:]

def find_signals(df, ind, cfg):
    hits, best = [], [False] * 5
    n = len(df)
    any_checked = False
    for i in range(max(0, n - cfg["signal_lookback_days"]), n):
        ok, d = check_pattern_at(df, ind, i, cfg)
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

def analyze_sweep_mss_retest(sym, df_daily):
    if df_daily is None: return None
    price   = float(df_daily["Close"].iloc[-1])
    avg_vol = float(df_daily["Volume"].tail(20).mean())
    if price   < CFG["min_price"]:      return None
    if avg_vol < CFG["min_avg_volume"]: return None
    if len(df_daily) < CFG["min_bars"]: return None

    ind = calc_indicators(df_daily, CFG)
    hits = find_signals(df_daily, ind, CFG)
    if not hits: return None
    sig = hits[-1]
    a, i = sig["atr"], sig["i"]
    entry = sig["close"]
    stop = sig["sweep_low"] - CFG["stop_buffer_atr"] * a
    if stop <= 0 or entry <= stop: return None
    risk = entry - stop
    risk_pct = risk / entry * 100
    h = df_daily["High"].values.astype(float)
    # target = the nearest resistance above entry: the high the MSS move reached before
    # pulling back, or a swing high from before the sweep (last 120 sessions), whichever is closer
    peak = float(h[sig["m"]:i].max()) if sig["m"] is not None and i > sig["m"] else np.nan
    cands = [float(h[q]) for q in range(max(0, i - CFG["resistance_lookback"]), sig["s"])
             if ind["ph"][q] and h[q] > entry]                         # swing highs before the sweep
    if np.isfinite(peak) and peak > entry:
        cands.append(peak)
    target = min(cands) if cands else np.nan
    reward_r = (target - entry) / risk if np.isfinite(target) else np.nan
    s200 = ind["s200"][i]
    mss_pct = (df_daily["Close"].iloc[sig["m"]] / sig["H"] - 1) * 100
    sweep_depth_atr = (sig["L0"] - sig["sweep_low"]) / a
    zone = "FVG + retest" if sig["in_fvg"] and sig["in_retest"] else ("FVG" if sig["in_fvg"] else "Retest")
    flags = [f"entry zone: {zone}", f"sweep {sweep_depth_atr:.1f} ATR below the old low"]
    if not np.isfinite(target): flags.append("no resistance found above - use 2R")
    elif reward_r < CFG["min_reward_r"]: flags.append(f"target under {CFG['min_reward_r']:g}R")
    flags.append("above SMA200" if np.isfinite(s200) and entry > s200 else "below SMA200")

    score = 30
    score += 15 if sig["in_fvg"] else 0                                 # FVG entry
    score += 10 if sig["in_retest"] else 0                              # retest of the broken high
    score += min(10, max(0, sweep_depth_atr) * 10)                      # a clear stop-run
    score += max(0, 10 - max(0, risk_pct - 4) * 2)                      # tighter stop
    score += min(15, max(0, reward_r) * 5) if np.isfinite(reward_r) else 5
    score += 5 if np.isfinite(s200) and entry > s200 else 0
    score += min(5, max(0, mss_pct) * 2)                                # decisive MSS close
    score = round(min(100, max(0, score)))
    t2 = entry + 2 * risk

    return {
        "Score": score, "Price": round(price, 2),
        "Entry_Price": round(entry, 2), "Stop_Loss": round(stop, 2), "Risk_%": round(risk_pct, 1),
        "Target_Resistance": round(target, 2) if np.isfinite(target) else round(t2, 2),
        "Reward_R": round(reward_r, 1) if np.isfinite(reward_r) else 2.0,
        "Entry_Zone": zone,
        "Old_Swing_Low": round(sig["L0"], 2), "Old_Low_Date": df_daily.index[sig["j0"]].strftime("%Y-%m-%d"),
        "Sweep_Low": round(sig["sweep_low"], 2), "Sweep_Date": df_daily.index[sig["s"]].strftime("%Y-%m-%d"),
        "MSS_Level": round(sig["H"], 2), "MSS_Date": df_daily.index[sig["m"]].strftime("%Y-%m-%d"),
        "FVG_Bottom": round(sig["fvg"][0], 2) if sig["fvg"] else np.nan,
        "FVG_Top": round(sig["fvg"][1], 2) if sig["fvg"] else np.nan,
        "Above_SMA200": "Yes" if np.isfinite(s200) and entry > s200 else "No",
        "Signal_Date": sig["date"].strftime("%Y-%m-%d"),
        "Days_Since_Signal": int(len(df_daily) - 1 - i),
        "Flags": " | ".join(flags),
        "_df": df_daily, "_ind": ind, "_sig": sig,
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

for sym in tqdm(list(daily_map.keys()), desc="Checking pattern", unit="stk"):
    try:
        r = analyze_sweep_mss_retest(sym, daily_map[sym])
        if r is None: continue
        r["Ticker"] = sym
        results.append(r)
        print(f"  OK {sym:<6} ${r['Price']:.2f}  Score {r['Score']}  sweep {r['Sweep_Date']}  MSS {r['MSS_Date']}  "
              f"{r['Entry_Zone']}  target {r['Target_Resistance']} ({r['Reward_R']}R)")
    except Exception: pass

fc = FUNNEL_COUNTS
print(f"\n{'=' * 65}\n  SCAN COMPLETE - {len(results)} matches from {got_daily} tickers with data")
print(f"  FUNNEL (last {CFG['signal_lookback_days']} session(s))")
print(f"  Tickers checked                                   : {fc['tickers_checked']}")
print(f"  Step 1 - liquidity sweep of a swing low (30d)      : {fc['passed_step1_liquidity_sweep']}")
print(f"  Step 2 - + bullish MSS (close above swing high)    : {fc['passed_step2_bullish_mss']}")
print(f"  Step 3 - + sweep low held since                    : {fc['passed_step3_sweep_low_held']}")
print(f"  Step 4 - + first pullback into retest / FVG today  : {fc['passed_step4_first_retest_or_fvg']}")
print(f"  Step 5 - + close holding the zone                  : {fc['passed_step5_holding_the_zone']}  (= full pattern)")
print("=" * 65)

results.sort(key=lambda x: x["Score"], reverse=True)

# -- Save CSV + TradingView list
ts      = datetime.today().strftime("%Y%m%d_%H%M")
out_dir = os.environ.get("GITHUB_WORKSPACE", os.getcwd())
COLS = ["Ticker","Price","Score","Entry_Price","Stop_Loss","Risk_%","Target_Resistance","Reward_R","Entry_Zone",
        "Old_Swing_Low","Old_Low_Date","Sweep_Low","Sweep_Date","MSS_Level","MSS_Date","FVG_Bottom","FVG_Top",
        "Above_SMA200","Signal_Date","Days_Since_Signal","Flags"]
df_out = pd.DataFrame([{k: v for k, v in r.items() if not k.startswith("_")}
                       for r in results]) if results else pd.DataFrame(columns=COLS)
df_out = df_out[[c for c in COLS if c in df_out.columns]].reset_index(drop=True)

fpath = os.path.join(out_dir, f"daily_sweep_mss_retest_{ts}.csv")
try:
    from position_sizing import add_position_size
    df_out = add_position_size(df_out)
    try: __import__("best_picks").rank_best(df_out)
    except Exception as _bp_err: print(f"  Best 10 skipped: {_bp_err}")
except Exception as _ps_err:
    print(f"  Position sizing skipped: {_ps_err}")
df_out.to_csv(fpath, index=False)
print(f"\n  CSV -> {fpath}")
tv = os.path.join(out_dir, f"tv_daily_sweep_mss_retest_{ts}.txt")
with open(tv, "w") as f:
    f.write(f"###Sweep + MSS + Retest {datetime.today().strftime('%Y-%m-%d')}\n")
    for r in results: f.write(f"NASDAQ:{r['Ticker']}\n")
print(f"  TradingView -> {tv}")
if results:
    print(f"\n  Tickers: {', '.join(r['Ticker'] for r in results)}")

# -- Email with CSV attached
EMAIL_COLS = ["Ticker","Price","Score","Entry_Price","Stop_Loss","Risk_%","Target_Resistance","Reward_R",
              "Entry_Zone","Sweep_Date","MSS_Date","MSS_Level","Above_SMA200"]

def _fmt_cell(col, v):
    try:
        if col in ("Price","Entry_Price","Stop_Loss","Target_Resistance","MSS_Level"): return f"${float(v):.2f}"
        if col == "Risk_%": return f"{float(v):.1f}%"
        if col == "Reward_R": return f"{float(v):.1f}R"
        if col == "Score": return f"{float(v):.0f}"
    except Exception: pass
    return "-" if v is None else str(v)

def _send_email(rl, csv_path):
    import smtplib
    from email.mime.multipart import MIMEMultipart
    from email.mime.text      import MIMEText
    from email.mime.base      import MIMEBase
    from email                import encoders

    gu = _GMAIL_USER; gp = _GMAIL_PASS; et = _EMAIL_TO
    if not (gu and gp and et):
        print("[Email] GMAIL_USER / GMAIL_PASS / EMAIL_TO missing - skipped")
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
                if c == "Target_Resistance": sty = "color:#2563eb;font-weight:600"
                tds += f'<td style="padding:6px 10px;font-size:12px;{sty}">{_fmt_cell(c, r.get(c))}</td>'
            rows += f'<tr style="background:{bg}">{tds}</tr>'
        if not rows:
            rows = (f'<tr><td colspan="{len(EMAIL_COLS)}" style="padding:20px;text-align:center;'
                    f'color:#94a3b8">No matches today</td></tr>')
        tick_html = (f'<p style="margin:10px 0 0;font-size:12px;font-family:monospace">'
                     f'{", ".join(r["Ticker"] for r in rl)}</p>') if rl else ""
        html = f"""<!DOCTYPE html><html><body style="margin:0;background:#f1f5f9;font-family:'Segoe UI',Arial,sans-serif">
<table width="100%" cellpadding="0" cellspacing="0"><tr><td align="center" style="padding:20px 10px">
<table width="100%" style="max-width:900px;background:#fff;border-radius:12px;overflow:hidden">
<tr><td style="background:linear-gradient(135deg,#0f172a,#1e3a5f);padding:20px 26px">
<h1 style="margin:0;color:#60a5fa;font-size:19px">Liquidity Sweep + MSS + Retest/FVG Entry</h1>
<p style="margin:6px 0 0;color:#94a3b8;font-size:12px">{datetime.today().strftime('%Y-%m-%d %H:%M UTC')} &middot; {cnt} match{'es' if cnt != 1 else ''}
 - a swing low was swept (stop hunt) and reclaimed, a close broke the last swing high (bullish MSS),
 and today price pulled back for the first time into the retest level or the FVG and held it</p>
</td></tr>
<tr><td style="padding:12px 26px;background:#0b1220"><p style="margin:0;color:#cbd5e1;font-size:12px">
FUNNEL: {fc['tickers_checked']} checked &rarr; {fc['passed_step1_liquidity_sweep']} liquidity sweep
&rarr; {fc['passed_step2_bullish_mss']} + bullish MSS
&rarr; {fc['passed_step3_sweep_low_held']} + sweep low held
&rarr; {fc['passed_step4_first_retest_or_fvg']} + first pullback into retest/FVG today
&rarr; <b style="color:#facc15">{fc['passed_step5_holding_the_zone']} + holding the zone = full match</b></p></td></tr>
<tr><td style="padding:16px"><div style="overflow-x:auto;border:1px solid #e2e8f0;border-radius:8px">
<table style="border-collapse:collapse;width:100%"><thead><tr>{th}</tr></thead><tbody>{rows}</tbody></table></div>
{tick_html}
<p style="font-size:11px;color:#64748b;margin:10px 0 0">Entry = today's close &middot; Stop = sweep low - {CFG['stop_buffer_atr']:g} ATR &middot;
Target = nearest resistance above (the MSS move's high or an earlier swing high), Reward_R in R multiples.
Retest zone = broken swing high +/- {CFG['retest_tol_atr']:g} ATR; FVG = gap between candle k-1's high and candle k+1's low in the up-move.
Full results (with Quantity for a fixed max loss) attached as CSV.</p>
</td></tr>
<tr><td style="background:#f8fafc;padding:10px;text-align:center;color:#94a3b8;font-size:10px">
Not financial advice &middot; Auto-generated by GitHub Actions</td></tr>
</table></td></tr></table></body></html>"""

        plain = [f"Sweep + MSS + Retest - {datetime.today().strftime('%Y-%m-%d')}",
                 f"{cnt} matches",
                 f"FUNNEL: {fc['tickers_checked']} checked -> {fc['passed_step1_liquidity_sweep']} sweep -> "
                 f"{fc['passed_step2_bullish_mss']} MSS -> {fc['passed_step3_sweep_low_held']} held -> "
                 f"{fc['passed_step4_first_retest_or_fvg']} retest/FVG today -> {fc['passed_step5_holding_the_zone']} holding (=full match)",
                 "=" * 60]
        for r in rl[:50]:
            plain.append(f"{r['Ticker']:<6} ${r['Price']:.2f}  Score:{r['Score']}  Entry:${r['Entry_Price']:.2f}  "
                         f"SL:${r['Stop_Loss']:.2f} ({r['Risk_%']}%)  Target:${r['Target_Resistance']:.2f} ({r['Reward_R']}R)  "
                         f"{r['Entry_Zone']}  sweep {r['Sweep_Date']} MSS {r['MSS_Date']}")
        if rl:
            plain += ["", "Tickers: " + ", ".join(r["Ticker"] for r in rl)]
        else:
            plain.append("No matches today")
        plain.append("\nFull results in CSV attachment.")

        subj = (f"Sweep + MSS + Retest - {cnt} signal{'s' if cnt != 1 else ''} - "
                f"{datetime.today().strftime('%Y-%m-%d')}")
        msg = MIMEMultipart("mixed")
        msg["Subject"] = subj; msg["From"] = gu; msg["To"] = ", ".join(eto)
        alt = MIMEMultipart("alternative")
        alt.attach(MIMEText("\n".join(plain), "plain"))
        alt.attach(MIMEText(html, "html"))
        msg.attach(alt)
    except Exception as e:
        print(f"[Email] Failed to build email body: {type(e).__name__}: {e}")
        return

    if csv_path and os.path.exists(csv_path):
        try:
            with open(csv_path, "rb") as f:
                part = MIMEBase("application", "octet-stream"); part.set_payload(f.read())
            encoders.encode_base64(part)
            part.add_header("Content-Disposition", f"attachment; filename={os.path.basename(csv_path)}")
            msg.attach(part)
        except Exception as e:
            print(f"[Email] CSV attach failed: {e}")
    try:
        with smtplib.SMTP_SSL("smtp.gmail.com", 465) as srv:
            srv.login(gu, gp.replace(" ", ""))
            try: __import__("best_picks").attach_to_email(msg)
            except Exception as _bp_err: print(f"  Best 10 not added: {_bp_err}")
            srv.sendmail(gu, eto, msg.as_string())
        print(f"[Email] Sent to {', '.join(eto)} - {subj}")
    except Exception as e:
        print(f"[Email] {type(e).__name__}: {e}")

try:
    _send_email(results, fpath)
except Exception as e:
    print(f"[Email] Unexpected top-level error: {type(e).__name__}: {e}")

# -- Charts for top 5: last 90 daily candles, old low, sweep low, MSS level, FVG, stop, target
if results:
    top = results[:5]
    fig, axes = plt.subplots(len(top), 1, figsize=(13, 4.2*len(top)), facecolor="#0f172a")
    if len(top) == 1: axes = [axes]
    for ax, r in zip(axes, top):
        d = r["_df"].iloc[-90:]
        n_ = len(d)
        x = np.arange(n_)
        ax.set_facecolor("#0f172a")
        cols = ["#22c55e" if c > o else "#ef4444" for o, c in zip(d["Open"], d["Close"])]
        ax.vlines(x, d["Low"], d["High"], color=cols, lw=0.8)
        ax.bar(x, (d["Close"] - d["Open"]).abs().clip(lower=1e-9), bottom=np.minimum(d["Open"], d["Close"]),
               color=cols, width=0.6)
        if np.isfinite(r["FVG_Bottom"]):
            ax.axhspan(r["FVG_Bottom"], r["FVG_Top"], color="#38bdf8", alpha=0.15)
        for when, lab in ((r["Sweep_Date"], "sweep"), (r["MSS_Date"], "MSS")):
            k_ = np.where(d.index == pd.Timestamp(when))[0]
            if len(k_): ax.axvline(k_[0], color="#334155", ls="--", lw=0.8); ax.text(k_[0], d["High"].max(), lab, color="#cbd5e1", fontsize=7, ha="center", va="bottom")
        for lvl, c_, lab in ((r["Old_Swing_Low"], "#94a3b8", "Old low"), (r["MSS_Level"], "#facc15", "MSS level"),
                             (r["Stop_Loss"], "#ef4444", "Stop"), (r["Target_Resistance"], "#22c55e", "Target")):
            ax.axhline(lvl, color=c_, ls=":", lw=0.9)
            ax.text(n_ - 0.4, lvl, f" {lab} {lvl:.2f}", color=c_, fontsize=7, va="center")
        ax.set_xticks(x[::10]); ax.set_xticklabels([t.strftime("%m/%d") for t in d.index[::10]], fontsize=7)
        ax.set_title(f"{r['Ticker']} | USD {r['Price']:.2f} | Score {r['Score']} | {r['Entry_Zone']} | "
                     f"target {r['Target_Resistance']:.2f} ({r['Reward_R']}R) | {r['Signal_Date']}",
                     color="#e2e8f0", fontsize=9, fontweight="bold")
        ax.tick_params(colors="#94a3b8", labelsize=7)
        for sp in ax.spines.values(): sp.set_edgecolor("#1e3a5f")
        ax.grid(color="#1e3a5f", ls="--", lw=0.5, alpha=0.6)
    plt.suptitle(f"Liquidity Sweep + MSS + Retest/FVG - {datetime.today().strftime('%Y-%m-%d')}",
                 color="#60a5fa", fontsize=12, fontweight="bold")
    plt.tight_layout()
    cp = os.path.join(out_dir, f"daily_sweep_mss_retest_chart_{ts}.png")
    plt.savefig(cp, dpi=130, bbox_inches="tight", facecolor="#0f172a")
    plt.close()
    print(f"  Chart -> {cp}")

print(f"""
{'=' * 55}
  PATTERN (latest session)
  1) Liquidity sweep: a low below the last swing low (last {CFG['sweep_lookback']} sessions), reclaimed within {CFG['reclaim_bars']} bars
  2) Bullish MSS: a close above the last swing high within {CFG['mss_bars']} sessions
  3) The sweep low held since
  4) Today: first pullback into the retest zone (+/- {CFG['retest_tol_atr']:g} ATR) or the FVG
  5) Today's close at or above the bottom of that zone
  OUTPUT: Entry = close, Stop = sweep low - {CFG['stop_buffer_atr']:g} ATR, Target = nearest resistance
{'=' * 55}
""")
