# ============================================================
# NASDAQ / NYSE - Daily Major Support Retest (v1)
# ============================================================
#
# Finds stocks back at a "major support": a price where, in the last 1-3
# months, the stock made a very strong reversal, and today it is back at the
# same level and showing the first sign of bouncing again.
#
#   1. MAJOR LOW: a swing low 25-65 sessions ago (~1.5-3 months) that is the
#      lowest low of the 5 bars each side AND of the 40 bars before it
#   2. STRONG REVERSAL: within 15 sessions of that low the stock rallied at
#      least +10% AND at least 4 ATR (ATR = 14-day average true range)
#   3. SUPPORT HELD: no close since then below support - 0.5 ATR
#   4. BACK AT SUPPORT: today's low is within support - 0.5 ATR .. + 1.0 ATR
#      and today's close is not below support - 0.25 ATR
#   5. REJECTION: today's close is in the upper half of its range and the
#      candle is green or has a lower wick at least as long as its body
#
# When several old lows qualify, the one that gets furthest through the
# steps (then the strongest reversal) is used.
#
# Entry = today's close; Stop = the lower of today's low and the support,
# minus 0.25 ATR; Target = the high the last reversal reached.
# Thresholds, stop and target are reasonable defaults, not explicitly
# requested; all are in CFG. Score ranks reversal strength and volume, how
# precisely the level was retested, the rejection close, earlier tests of the
# level, SMA200 position and reward:risk.
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
    # the pattern looks back up to 65 + 40 + 5 bars, the context SMA200 needs 200.
    "history_days"            : 400,
    "min_bars"                : 210,

    "atr_period"              : 14,
    "pivot_min_age"           : 25,    # the support low formed 25..65 sessions ago
    "pivot_max_age"           : 65,    # (~1.5 to 3 months)
    "pivot_wing"              : 5,     # swing low = lowest low of the 5 bars each side
    "major_lookback"          : 40,    # ... and the lowest low of the 40 bars before it
    "rally_bars"              : 15,    # the reversal: rally within 15 sessions of the low
    "rally_min_pct"           : 10.0,  # ... of at least +10%
    "rally_min_atr"           : 4.0,   # ... and at least 4 ATR
    "break_tol_atr"           : 0.5,   # support "held": no close below S - 0.5 ATR since
    "zone_above_atr"          : 1.0,   # today's low within S - 0.5 ATR .. S + 1.0 ATR
    "zone_below_atr"          : 0.5,
    "close_below_tol_atr"     : 0.25,  # today's close not below S - 0.25 ATR
    "reject_close_pos"        : 0.5,   # today's close in the upper half of its range
    "stop_buffer_atr"         : 0.25,  # stop = min(today's low, S) - 0.25 ATR
    "touch_lookback"          : 120,   # count earlier swing lows near S (extra tests)
    "vol_avg_bars"            : 20,
    "signal_lookback_days"    : 1,     # the latest session only ("today")

    "min_avg_volume"          : 200_000,
    "min_price"               : 5.0,

    "batch_size"              : 50,
    "batch_sleep"             : 1.5,
}

# -- Indicators
def calc_indicators(df, cfg):
    h, l, c, v = (df[k].values.astype(float) for k in ("High", "Low", "Close", "Volume"))
    pc = np.concatenate([[np.nan], c[:-1]])
    tr = np.nanmax(np.vstack([h - l, np.abs(h - pc), np.abs(l - pc)]), axis=0)
    w = cfg["pivot_wing"]
    lows = pd.Series(l)
    is_piv = (lows == lows.rolling(2 * w + 1, center=True).min()).values
    return {
        "atr":  pd.Series(tr).rolling(cfg["atr_period"]).mean().values,
        "vavg": pd.Series(v).rolling(cfg["vol_avg_bars"]).mean().values,
        "piv":  is_piv,
        "s50":  pd.Series(c).rolling(50).mean().values,
        "s200": pd.Series(c).rolling(200).mean().values,
    }

def _candidate(df, ind, i, j, cfg):
    """Steps 1..5 for support pivot j seen from signal day i (no short-circuit)."""
    o, h, l, c, v = (df[k].values.astype(float) for k in ("Open", "High", "Low", "Close", "Volume"))
    atr, vavg = ind["atr"], ind["vavg"]
    S, a = l[j], atr[i]
    lb = max(0, j - cfg["major_lookback"])
    step1 = bool(ind["piv"][j] and S <= l[lb:j + 1].min())                    # major swing low
    k_end = min(j + 1 + cfg["rally_bars"], i)
    k_pk = j + 1 + int(np.argmax(h[j + 1:k_end])) if k_end > j + 1 else j
    peak = h[k_pk] if k_pk > j else S
    rally_pct = (peak / S - 1) * 100 if S > 0 else 0.0
    rally_atr = (peak - S) / atr[j] if np.isfinite(atr[j]) and atr[j] > 0 else 0.0
    step2 = bool(rally_pct >= cfg["rally_min_pct"] and rally_atr >= cfg["rally_min_atr"])  # strong reversal
    between = c[j + 1:i]
    step3 = bool(len(between) and between.min() >= S - cfg["break_tol_atr"] * a)  # support held since
    step4 = bool(S - cfg["zone_below_atr"] * a <= l[i] <= S + cfg["zone_above_atr"] * a
                 and c[i] >= S - cfg["close_below_tol_atr"] * a)              # back at support today
    rng = h[i] - l[i]
    pos = (c[i] - l[i]) / rng if rng > 0 else 0.0
    lower = min(o[i], c[i]) - l[i]
    body = abs(c[i] - o[i])
    step5 = bool(pos >= cfg["reject_close_pos"] and (c[i] > o[i] or lower >= body))  # rejection candle
    rev_vol = float(v[j:j + 3].max() / vavg[j]) if np.isfinite(vavg[j]) and vavg[j] > 0 else np.nan
    t0 = max(0, j - cfg["touch_lookback"])
    tests = int(sum(1 for q in range(t0, j - cfg["pivot_wing"])
                    if ind["piv"][q] and abs(l[q] - S) <= 1.0 * atr[j]))
    return {
        "j": j, "support": float(S), "support_date": df.index[j], "peak": float(peak),
        "peak_date": df.index[k_pk], "rally_pct": float(rally_pct), "rally_atr": float(rally_atr),
        "rev_vol": rev_vol, "prior_tests": tests, "close_pos": float(pos),
        "low_vs_s_atr": float((l[i] - S) / a), "today_vol": float(v[i] / vavg[i]) if vavg[i] > 0 else np.nan,
        "step1": step1, "step2": step2, "step3": step3, "step4": step4, "step5": step5,
    }

def check_pattern_at(df, ind, i, cfg):
    """
    SINGLE SOURCE OF TRUTH. Checks signal day i against every candidate support
    low 25..65 sessions earlier. Computes EVERY step (no short-circuit) so the
    funnel shows where tickers drop out; reports the candidate that got furthest
    (ties: the strongest reversal). Returns (passed, details); {} if data short.
    """
    lo, hi = i - cfg["pivot_max_age"], i - cfg["pivot_min_age"]
    if lo < cfg["major_lookback"] or i >= len(df):
        return False, {}
    a = ind["atr"][i]
    if not np.isfinite(a) or a <= 0 or not np.isfinite(ind["vavg"][i]):
        return False, {}
    best, best_key = None, None
    for j in range(lo, hi + 1):
        if not ind["piv"][j]: continue
        d = _candidate(df, ind, i, j, cfg)
        depth = 0
        for k in range(1, 6):
            if not d[f"step{k}"]: break
            depth = k
        key = (depth, d["rally_atr"])
        if best_key is None or key > best_key:
            best, best_key = d, key
    if best is None:
        best = {f"step{k}": False for k in range(1, 6)}
    passed = all(best[f"step{k}"] for k in range(1, 6))
    best.update({"date": df.index[i], "i": i, "atr": float(a), "close": float(df["Close"].iloc[i]),
                 "low": float(df["Low"].iloc[i])})
    return passed, best

# -- Diagnostic funnel (per ticker, cumulative over the lookback days)
FUNNEL_COUNTS = {
    "tickers_checked": 0,
    "passed_step1_major_low_2_3_months_ago": 0,
    "passed_step2_strong_reversal": 0,
    "passed_step3_support_held": 0,
    "passed_step4_back_at_support": 0,
    "passed_step5_rejection_candle": 0,
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

def analyze_support_retest(sym, df_daily):
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
    S, a = sig["support"], sig["atr"]
    entry = sig["close"]
    stop = min(sig["low"], S) - CFG["stop_buffer_atr"] * a
    if stop <= 0 or entry <= stop: return None
    risk = entry - stop
    risk_pct = risk / entry * 100
    target = sig["peak"]                                 # the high of the last reversal
    reward_r = (target - entry) / risk
    i = sig["i"]
    s50, s200 = ind["s50"][i], ind["s200"][i]
    flags = [f"reversal +{sig['rally_pct']:.0f}% ({sig['rally_atr']:.1f} ATR) in {CFG['rally_bars']}d",
             f"reversal vol {sig['rev_vol']:.1f}x"]
    if sig["prior_tests"]: flags.append(f"{sig['prior_tests']} earlier test(s) near this level")
    flags.append("above SMA200" if np.isfinite(s200) and entry > s200 else "below SMA200")
    if np.isfinite(sig["today_vol"]) and sig["today_vol"] < 0.8: flags.append("quiet volume on the retest")
    if reward_r < 2: flags.append("target under 2R")

    score = 30
    score += min(20, (sig["rally_atr"] - 4) * 3 + (sig["rally_pct"] - 10) * 0.5)   # stronger reversal
    score += min(10, max(0, (sig["rev_vol"] - 1) * 5)) if np.isfinite(sig["rev_vol"]) else 0
    score += max(0, 10 - abs(sig["low_vs_s_atr"]) * 10)                       # precise retest
    score += 10 * min(1, max(0, (sig["close_pos"] - 0.5) * 2))                # strong rejection close
    score += min(10, sig["prior_tests"] * 5)                                  # tested before
    score += 5 if np.isfinite(s200) and entry > s200 else 0
    score += min(5, max(0, reward_r - 1) * 2)
    score = round(min(100, max(0, score)))

    return {
        "Score": score, "Price": round(price, 2),
        "Entry_Price": round(entry, 2), "Stop_Loss": round(stop, 2), "Risk_%": round(risk_pct, 1),
        "Target_Prior_High": round(target, 2), "Reward_R": round(reward_r, 1),
        "Support": round(S, 2), "Support_Date": sig["support_date"].strftime("%Y-%m-%d"),
        "Reversal_High": round(sig["peak"], 2), "Reversal_High_Date": sig["peak_date"].strftime("%Y-%m-%d"),
        "Reversal_%": round(sig["rally_pct"], 1), "Reversal_ATR": round(sig["rally_atr"], 1),
        "Reversal_Vol_x": round(sig["rev_vol"], 2), "Prior_Tests": sig["prior_tests"],
        "Low_vs_Support_ATR": round(sig["low_vs_s_atr"], 2), "Close_Position": round(sig["close_pos"], 2),
        "Today_Vol_x": round(sig["today_vol"], 2),
        "Above_SMA50": "Yes" if np.isfinite(s50) and entry > s50 else "No",
        "Above_SMA200": "Yes" if np.isfinite(s200) and entry > s200 else "No",
        "Signal_Date": sig["date"].strftime("%Y-%m-%d"),
        "Days_Since_Signal": int(len(df_daily) - 1 - i),
        "Flags": " | ".join(flags),
        "_df": df_daily, "_ind": ind,
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
        r = analyze_support_retest(sym, daily_map[sym])
        if r is None: continue
        r["Ticker"] = sym
        results.append(r)
        print(f"  OK {sym:<6} ${r['Price']:.2f}  Score {r['Score']}  support {r['Support']} ({r['Support_Date']})  "
              f"reversal +{r['Reversal_%']}% / {r['Reversal_ATR']} ATR  {r['Signal_Date']}")
    except Exception: pass

fc = FUNNEL_COUNTS
print(f"\n{'=' * 65}\n  SCAN COMPLETE - {len(results)} matches from {got_daily} tickers with data")
print(f"  FUNNEL (last {CFG['signal_lookback_days']} session(s))")
print(f"  Tickers checked                                   : {fc['tickers_checked']}")
print(f"  Step 1 - major swing low 25-65 sessions ago        : {fc['passed_step1_major_low_2_3_months_ago']}")
print(f"  Step 2 - + strong reversal off it                  : {fc['passed_step2_strong_reversal']}")
print(f"  Step 3 - + support held since                      : {fc['passed_step3_support_held']}")
print(f"  Step 4 - + today's low back at the support         : {fc['passed_step4_back_at_support']}")
print(f"  Step 5 - + rejection candle today                  : {fc['passed_step5_rejection_candle']}  (= full pattern)")
print("=" * 65)

results.sort(key=lambda x: x["Score"], reverse=True)

# -- Save CSV + TradingView list
ts      = datetime.today().strftime("%Y%m%d_%H%M")
out_dir = os.environ.get("GITHUB_WORKSPACE", os.getcwd())
COLS = ["Ticker","Price","Score","Entry_Price","Stop_Loss","Risk_%","Target_Prior_High","Reward_R",
        "Support","Support_Date","Reversal_High","Reversal_High_Date","Reversal_%","Reversal_ATR",
        "Reversal_Vol_x","Prior_Tests","Low_vs_Support_ATR","Close_Position","Today_Vol_x",
        "Above_SMA50","Above_SMA200","Signal_Date","Days_Since_Signal","Flags"]
df_out = pd.DataFrame([{k: v for k, v in r.items() if not k.startswith("_")}
                       for r in results]) if results else pd.DataFrame(columns=COLS)
df_out = df_out[[c for c in COLS if c in df_out.columns]].reset_index(drop=True)

fpath = os.path.join(out_dir, f"daily_major_support_retest_{ts}.csv")
try:
    from position_sizing import add_position_size
    df_out = add_position_size(df_out)
    try: __import__("best_picks").rank_best(df_out)
    except Exception as _bp_err: print(f"  Best 10 skipped: {_bp_err}")
except Exception as _ps_err:
    print(f"  Position sizing skipped: {_ps_err}")
df_out.to_csv(fpath, index=False)
print(f"\n  CSV -> {fpath}")
tv = os.path.join(out_dir, f"tv_daily_major_support_retest_{ts}.txt")
with open(tv, "w") as f:
    f.write(f"###Major Support Retest {datetime.today().strftime('%Y-%m-%d')}\n")
    for r in results: f.write(f"NASDAQ:{r['Ticker']}\n")
print(f"  TradingView -> {tv}")
if results:
    print(f"\n  Tickers: {', '.join(r['Ticker'] for r in results)}")

# -- Email with CSV attached
EMAIL_COLS = ["Ticker","Price","Score","Entry_Price","Stop_Loss","Risk_%","Target_Prior_High","Reward_R",
              "Support","Support_Date","Reversal_%","Reversal_Vol_x","Prior_Tests","Above_SMA200"]

def _fmt_cell(col, v):
    try:
        if col in ("Price","Entry_Price","Stop_Loss","Target_Prior_High","Support"): return f"${float(v):.2f}"
        if col in ("Risk_%","Reversal_%"): return f"{float(v):.1f}%"
        if col in ("Reversal_Vol_x",): return f"{float(v):.1f}x"
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
                if c == "Target_Prior_High": sty = "color:#2563eb;font-weight:600"
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
<h1 style="margin:0;color:#60a5fa;font-size:19px">Major Support Retest - back at a level that reversed hard 1-3 months ago</h1>
<p style="margin:6px 0 0;color:#94a3b8;font-size:12px">{datetime.today().strftime('%Y-%m-%d %H:%M UTC')} &middot; {cnt} match{'es' if cnt != 1 else ''}
 - a major swing low 25-65 sessions ago that rallied at least +{CFG['rally_min_pct']:g}% and {CFG['rally_min_atr']:g} ATR within {CFG['rally_bars']} sessions,
 support held since, today's low back at that level and a rejection candle</p>
</td></tr>
<tr><td style="padding:12px 26px;background:#0b1220"><p style="margin:0;color:#cbd5e1;font-size:12px">
FUNNEL: {fc['tickers_checked']} checked &rarr; {fc['passed_step1_major_low_2_3_months_ago']} major low 1-3 months ago
&rarr; {fc['passed_step2_strong_reversal']} + strong reversal
&rarr; {fc['passed_step3_support_held']} + support held
&rarr; {fc['passed_step4_back_at_support']} + back at support today
&rarr; <b style="color:#facc15">{fc['passed_step5_rejection_candle']} + rejection candle = full match</b></p></td></tr>
<tr><td style="padding:16px"><div style="overflow-x:auto;border:1px solid #e2e8f0;border-radius:8px">
<table style="border-collapse:collapse;width:100%"><thead><tr>{th}</tr></thead><tbody>{rows}</tbody></table></div>
{tick_html}
<p style="font-size:11px;color:#64748b;margin:10px 0 0">Entry = today's close &middot; Stop = the lower of today's low and the support,
minus {CFG['stop_buffer_atr']:g} ATR &middot; Target = the high the last reversal reached (Reward_R in R multiples).
Support zone = support - {CFG['zone_below_atr']:g} ATR to + {CFG['zone_above_atr']:g} ATR; rejection = close in the upper half of the day's range and
green or a lower wick at least the body. Prior_Tests = earlier swing lows within 1 ATR of the level (last {CFG['touch_lookback']} sessions).
Full results (with Quantity for a fixed max loss) attached as CSV.</p>
</td></tr>
<tr><td style="background:#f8fafc;padding:10px;text-align:center;color:#94a3b8;font-size:10px">
Not financial advice &middot; Auto-generated by GitHub Actions</td></tr>
</table></td></tr></table></body></html>"""

        plain = [f"Major Support Retest - {datetime.today().strftime('%Y-%m-%d')}",
                 f"{cnt} matches",
                 f"FUNNEL: {fc['tickers_checked']} checked -> {fc['passed_step1_major_low_2_3_months_ago']} major low -> "
                 f"{fc['passed_step2_strong_reversal']} strong reversal -> {fc['passed_step3_support_held']} held -> "
                 f"{fc['passed_step4_back_at_support']} at support -> {fc['passed_step5_rejection_candle']} rejection (=full match)",
                 "=" * 60]
        for r in rl[:50]:
            plain.append(f"{r['Ticker']:<6} ${r['Price']:.2f}  Score:{r['Score']}  Entry:${r['Entry_Price']:.2f}  "
                         f"SL:${r['Stop_Loss']:.2f} ({r['Risk_%']}%)  Target:${r['Target_Prior_High']:.2f} ({r['Reward_R']}R)  "
                         f"support {r['Support']} ({r['Support_Date']}) reversal +{r['Reversal_%']}%")
        if rl:
            plain += ["", "Tickers: " + ", ".join(r["Ticker"] for r in rl)]
        else:
            plain.append("No matches today")
        plain.append("\nFull results in CSV attachment.")

        subj = (f"Major Support Retest - {cnt} signal{'s' if cnt != 1 else ''} - "
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

# -- Charts for top 5: last 90 daily candles, support line, reversal high, stop
if results:
    top = results[:5]
    fig, axes = plt.subplots(len(top), 1, figsize=(13, 4.2*len(top)), facecolor="#0f172a")
    if len(top) == 1: axes = [axes]
    for ax, r in zip(axes, top):
        d = r["_df"].iloc[-90:]
        x = np.arange(len(d))
        ax.set_facecolor("#0f172a")
        cols = ["#22c55e" if c > o else "#ef4444" for o, c in zip(d["Open"], d["Close"])]
        ax.vlines(x, d["Low"], d["High"], color=cols, lw=0.8)
        ax.bar(x, (d["Close"] - d["Open"]).abs().clip(lower=1e-9), bottom=np.minimum(d["Open"], d["Close"]),
               color=cols, width=0.6)
        ax.plot(x, r["_ind"]["s50"][-len(d):], color="#a78bfa", lw=0.9, label="SMA50")
        for when, lab in ((r["Support_Date"], "support low"), (r["Reversal_High_Date"], "reversal high")):
            k_ = np.where(d.index == pd.Timestamp(when))[0]
            if len(k_): ax.axvline(k_[0], color="#334155", ls="--", lw=0.8); ax.text(k_[0], d["High"].max(), lab, color="#cbd5e1", fontsize=7, ha="center", va="bottom")
        for lvl, c_, lab in ((r["Support"], "#22c55e", "Support"), (r["Stop_Loss"], "#ef4444", "Stop"),
                             (r["Target_Prior_High"], "#38bdf8", "Target")):
            ax.axhline(lvl, color=c_, ls=":", lw=0.9)
            ax.text(len(d) - 0.4, lvl, f" {lab} {lvl:.2f}", color=c_, fontsize=7, va="center")
        ax.set_xticks(x[::10]); ax.set_xticklabels([t.strftime("%m/%d") for t in d.index[::10]], fontsize=7)
        ax.set_title(f"{r['Ticker']} | USD {r['Price']:.2f} | Score {r['Score']} | support {r['Support']:.2f} ({r['Support_Date']}) | "
                     f"reversal +{r['Reversal_%']}% | {r['Reward_R']}R to prior high", color="#e2e8f0", fontsize=9, fontweight="bold")
        ax.tick_params(colors="#94a3b8", labelsize=7)
        for sp in ax.spines.values(): sp.set_edgecolor("#1e3a5f")
        ax.grid(color="#1e3a5f", ls="--", lw=0.5, alpha=0.6)
        ax.legend(loc="upper left", fontsize=7, facecolor="#0f172a", labelcolor="#cbd5e1", edgecolor="#1e3a5f")
    plt.suptitle(f"Major Support Retest - {datetime.today().strftime('%Y-%m-%d')}",
                 color="#60a5fa", fontsize=12, fontweight="bold")
    plt.tight_layout()
    cp = os.path.join(out_dir, f"daily_major_support_retest_chart_{ts}.png")
    plt.savefig(cp, dpi=130, bbox_inches="tight", facecolor="#0f172a")
    plt.close()
    print(f"  Chart -> {cp}")

print(f"""
{'=' * 55}
  PATTERN (latest session)
  1) A major swing low {CFG['pivot_min_age']}-{CFG['pivot_max_age']} sessions ago (lowest of {CFG['pivot_wing']} bars each side
     and of the {CFG['major_lookback']} bars before it) = the support
  2) A strong reversal off it: +{CFG['rally_min_pct']:g}% and {CFG['rally_min_atr']:g} ATR within {CFG['rally_bars']} sessions
  3) Support held since: no close below support - {CFG['break_tol_atr']:g} ATR
  4) Today's low back in the zone support - {CFG['zone_below_atr']:g} ATR .. + {CFG['zone_above_atr']:g} ATR,
     close not below support - {CFG['close_below_tol_atr']:g} ATR
  5) Rejection candle: close in the upper half of the range, green or lower wick >= body
  OUTPUT: Entry = close, Stop = min(low, support) - {CFG['stop_buffer_atr']:g} ATR, Target = the reversal high
{'=' * 55}
""")
