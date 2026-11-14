# ============================================================
# NASDAQ / NYSE - Daily Triangle Breakout + FVG Entry (v1)
# ============================================================
#
# From the "Triangle Breakout + FVG Entry" chart (2026-11-14):
#   1. IMPULSE LEG WITH AN FVG: a rally of at least +15% and 4 ATR within
#      20 sessions, from the leg low to a swing high (the leg high), that left
#      at least one bullish fair value gap (candle k-1 high < candle k+1 low)
#   2. TRIANGLE: from the leg high, 6-40 sessions of converging consolidation:
#      the upper line (leg high -> lower swing highs) is falling or flat, the
#      lower line (first swing low -> later swing lows) is rising or flat,
#      the range narrows by at least 25%, no high pokes above the upper line
#      and no close below the lower line (0.3 ATR tolerance), and the
#      consolidation gives back at most 61.8% of the leg
#   3. FAKE DROP: within the last 10 sessions a candle CLOSED below the lower
#      triangle line and its low reached the FVG (the highest gap of the leg
#      that sits below the triangle)
#   4. FVG HELD: no close below the FVG bottom since the drop, and no low
#      below the stop (FVG bottom - 0.25 ATR)
#   5. TODAY'S TRIGGER, either
#      - FVG ENTRY: price is back in the FVG today (low at or below its top),
#        closes in the upper half of its range or green, and has not broken
#        out of the triangle yet (within 5 sessions of the drop), or
#      - BREAKOUT: today's close crosses above the upper triangle line for the
#        first time since the drop
#
# Entry = today's close; Stop = below the FVG (FVG bottom - 0.25 ATR);
# Target_Measured = the upper triangle line today (the breakout point) + the
# leg height (leg high - leg low) = "same height" in the chart.
# Target_Resistance = the leg high (top of the triangle), the first
# resistance; if price is already above it, the measured target is used.
# Thresholds are reasonable defaults, not explicitly requested; all are in CFG.
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
    # history_days vs bars: 400 calendar days -> ~285 business days; the pattern
    # looks back at most ~10 + 40 + 20 bars, the context SMA200 needs 200.
    "history_days"            : 400,
    "min_bars"                : 210,

    "atr_period"              : 14,
    "pivot_wing"              : 2,     # swing high/low = highest/lowest of 2 bars each side
    "leg_max_bars"            : 20,    # the impulse leg took at most 20 sessions
    "leg_min_pct"             : 15.0,  # ... and gained at least 15%
    "leg_min_atr"             : 4.0,   # ... and at least 4 ATR
    "tri_min_bars"            : 6,     # triangle (leg high -> bar before the drop): 6-40 sessions
    "tri_max_bars"            : 40,
    "tri_tol_atr"             : 0.3,   # highs / closes may stray 0.3 ATR outside the lines
    "tri_flat_atr"            : 0.02,  # a line within +/-0.02 ATR per bar counts as flat
    "tri_min_narrow"          : 0.25,  # the range narrows by at least 25%
    "max_retrace"             : 0.618, # the triangle gives back at most 61.8% of the leg
    "drop_lookback"           : 10,    # the fake drop happened within the last 10 sessions
    "entry_window"            : 5,     # FVG entry: within 5 sessions of the drop
    "fvg_touch_tol_atr"       : 0.1,   # "in the FVG": low <= FVG top + 0.1 ATR
    "stop_buffer_atr"         : 0.25,  # stop = FVG bottom - 0.25 ATR
    "min_reward_r"            : 1.5,   # flag measured targets under 1.5R
    "signal_lookback_days"    : 1,     # the latest session only

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

def _leg(df, ind, p, cfg):
    """Step 1 for leg high p: the impulse leg into it and the FVGs it left."""
    h, l = df["High"].values.astype(float), df["Low"].values.astype(float)
    a = ind["atr"][p]
    if not np.isfinite(a) or a <= 0:
        return None
    q0 = max(1, p - cfg["leg_max_bars"])
    q = q0 + int(np.argmin(l[q0:p]))
    height = h[p] - l[q]
    if l[q] <= 0:
        return None
    ok = bool(p - q >= 2 and (h[p] / l[q] - 1) * 100 >= cfg["leg_min_pct"] and height >= cfg["leg_min_atr"] * a)
    gaps = [(float(h[k - 1]), float(l[k + 1]), k) for k in range(q + 1, p) if l[k + 1] > h[k - 1]]
    return {"p": p, "q": q, "leg_low": float(l[q]), "leg_high": float(h[p]), "height": float(height),
            "leg_pct": float((h[p] / l[q] - 1) * 100), "gaps": gaps, "ok": ok and bool(gaps)}

def _triangle(df, ind, p, d, a, cfg):
    """Step 2: converging lines over the bars p .. d-1 (leg high .. bar before the drop)."""
    h, l, c = (df[k].values.astype(float) for k in ("High", "Low", "Close"))
    wg, tol, flat = cfg["pivot_wing"], cfg["tri_tol_atr"] * a, cfg["tri_flat_atr"] * a
    ph = [j for j in range(p + 1, d - wg) if ind["ph"][j] and h[j] < h[p]]
    pl = [j for j in range(p + wg + 1, d - wg) if ind["pl"][j]]          # first low: after the top bar's wing
    if not ph or len(pl) < 2:
        return None
    su = max((h[j] - h[p]) / (j - p) for j in ph)            # upper line: leg high -> swing highs
    j1 = pl[0]
    sl = min((l[j] - l[j1]) / (j - j1) for j in pl[1:])      # lower line: first swing low -> later lows
    U = lambda k: h[p] + su * (k - p)
    L = lambda k: l[j1] + sl * (k - j1)
    w0, w1 = U(j1) - L(j1), U(d - 1) - L(d - 1)
    ok = bool(su <= flat and sl >= -flat and (su < -flat or sl > flat)
              and w0 > 0 and w1 > 0 and w1 <= (1 - cfg["tri_min_narrow"]) * w0
              and all(h[k] <= U(k) + tol for k in range(p + 1, d))
              and all(c[k] >= L(k) - tol for k in range(j1, d)))
    return {"su": float(su), "sl": float(sl), "j1": j1, "U": U, "L": L, "tri_low": float(l[p:d].min()), "ok": ok}

def check_pattern_at(df, ind, i, cfg):
    """
    SINGLE SOURCE OF TRUTH. Checks signal day i against every fake-drop bar in the
    last `drop_lookback` sessions and every leg high that could start its triangle.
    Records how far each candidate got (funnel); reports the candidate that got
    furthest (ties: today's trigger, then the bigger leg).
    Returns (passed, details); {} if data short.
    """
    need = cfg["drop_lookback"] + cfg["tri_max_bars"] + cfg["leg_max_bars"] + 5
    if i < need or i >= len(df):
        return False, {}
    a = ind["atr"][i]
    if not np.isfinite(a) or a <= 0:
        return False, {}
    o, h, l, c = (df[k].values.astype(float) for k in ("Open", "High", "Low", "Close"))
    wg = cfg["pivot_wing"]
    best, best_key, legs = None, None, {}
    for d in range(i - cfg["drop_lookback"], i):
        for p in range(max(wg, d - cfg["tri_max_bars"]), d - cfg["tri_min_bars"] + 1):
            if not ind["ph"][p] or h[p] < h[p:d].max():
                continue                                    # the leg high tops the whole triangle
            if p not in legs:
                legs[p] = _leg(df, ind, p, cfg)
            lg = legs[p]
            if lg is None:
                continue
            r = {"d": d, "p": p, "leg": lg, "step1": lg["ok"]}
            tri = _triangle(df, ind, p, d, a, cfg)
            r["tri"] = tri
            r["step2"] = bool(r["step1"] and tri is not None and tri["ok"]
                              and tri["tri_low"] >= lg["leg_high"] - cfg["max_retrace"] * lg["height"])
            # the FVG = the highest gap of the leg that sits below the triangle
            fv = [g for g in lg["gaps"] if g[1] <= (tri["tri_low"] if tri else lg["leg_high"])]
            fvg = max(fv, key=lambda g: g[1]) if fv else None
            r["fvg"] = fvg
            r["step3"] = bool(r["step2"] and fvg is not None and c[d] < tri["L"](d)
                              and l[d] <= fvg[1] + cfg["fvg_touch_tol_atr"] * a)
            stop = fvg[0] - cfg["stop_buffer_atr"] * a if fvg else np.nan
            r["stop"] = float(stop)
            r["step4"] = bool(r["step3"] and c[d:i + 1].min() >= fvg[0] and l[d:i + 1].min() > stop)
            stage = None
            if r["step4"]:
                U = tri["U"]
                broke_before = any(c[k] > U(k) for k in range(d, i))
                if (not broke_before and c[i] > U(i)):
                    stage = "Breakout"
                elif (not broke_before and c[i] <= U(i) and i - d <= cfg["entry_window"]
                      and l[i] <= fvg[1] + cfg["fvg_touch_tol_atr"] * a
                      and (c[i] > o[i] or c[i] >= (h[i] + l[i]) / 2)):
                    stage = "FVG entry"
            r["stage"] = stage
            r["step5"] = stage is not None
            depth = 0
            for k in range(1, 6):
                if not r[f"step{k}"]: break
                depth = k
            key = (depth, lg["height"], -d)
            if best_key is None or key > best_key:
                best, best_key = r, key
    if best is None:
        best = {f"step{k}": False for k in range(1, 6)}
    passed = all(best[f"step{k}"] for k in range(1, 6))
    best.update({"date": df.index[i], "i": i, "atr": float(a), "close": float(c[i]), "low": float(l[i])})
    return passed, best

# -- Diagnostic funnel (per ticker, cumulative over the lookback days)
FUNNEL_COUNTS = {
    "tickers_checked": 0,
    "passed_step1_impulse_leg_fvg": 0,
    "passed_step2_triangle": 0,
    "passed_step3_fake_drop_into_fvg": 0,
    "passed_step4_fvg_held": 0,
    "passed_step5_trigger_today": 0,
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

def analyze_triangle_fvg(sym, df_daily):
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
    lg, tri, fvg = sig["leg"], sig["tri"], sig["fvg"]
    entry = sig["close"]
    stop = sig["stop"]
    if stop <= 0 or entry <= stop: return None
    risk = entry - stop
    risk_pct = risk / entry * 100
    upper, lower = float(tri["U"](i)), float(tri["L"](i))
    t_mm = max(upper, entry) + lg["height"]                      # breakout point + leg height
    rr_mm = (t_mm - entry) / risk
    target = lg["leg_high"] if lg["leg_high"] > entry else t_mm  # first resistance = top of the triangle
    reward_r = (target - entry) / risk
    s200 = ind["s200"][i]
    vol = df_daily["Volume"].values.astype(float)
    vol_x = vol[i] / np.mean(vol[i - 20:i]) if np.mean(vol[i - 20:i]) > 0 else np.nan
    flags = [f"stage: {sig['stage']}",
             f"leg +{lg['leg_pct']:.0f}% in {lg['p'] - lg['q']} sessions",
             f"triangle {sig['d'] - sig['p']} sessions",
             f"drop {(lower - sig['low']) / a:.1f} ATR below the lower line" if sig["stage"] == "FVG entry" else
             f"breakout {(entry - upper) / a:.1f} ATR above the upper line"]
    if sig["stage"] == "Breakout" and np.isfinite(vol_x): flags.append(f"breakout volume {vol_x:.1f}x avg")
    if rr_mm < CFG["min_reward_r"]: flags.append(f"measured target under {CFG['min_reward_r']:g}R")
    flags.append("above SMA200" if np.isfinite(s200) and entry > s200 else "below SMA200")

    score = 30
    score += 15 if sig["stage"] == "FVG entry" else 10                  # the chart's entry is in the FVG
    score += min(15, max(0, rr_mm) * 4)                                 # reward:risk to the measured target
    score += max(0, 10 - max(0, risk_pct - 4) * 2)                      # tighter stop
    score += min(10, max(0, lg["leg_pct"] - 15) / 2)                    # stronger impulse leg
    score += 10 if (sig["stage"] == "Breakout" and np.isfinite(vol_x) and vol_x >= 1.5) else 0
    score += 10 if np.isfinite(s200) and entry > s200 else 0
    score = round(min(100, max(0, score)))
    di = df_daily.index

    return {
        "Score": score, "Price": round(price, 2), "Stage": sig["stage"],
        "Entry_Price": round(entry, 2), "Stop_Loss": round(stop, 2), "Risk_%": round(risk_pct, 1),
        "Target_Resistance": round(target, 2), "Reward_R": round(reward_r, 1),
        "Target_Measured": round(t_mm, 2), "Reward_R_Measured": round(rr_mm, 1),
        "Leg_Low": round(lg["leg_low"], 2), "Leg_Low_Date": di[lg["q"]].strftime("%Y-%m-%d"),
        "Leg_High": round(lg["leg_high"], 2), "Leg_High_Date": di[lg["p"]].strftime("%Y-%m-%d"),
        "Leg_%": round(lg["leg_pct"], 1), "Triangle_Bars": int(sig["d"] - sig["p"]),
        "Upper_Line": round(upper, 2), "Lower_Line": round(lower, 2),
        "Fake_Drop_Date": di[sig["d"]].strftime("%Y-%m-%d"),
        "FVG_Bottom": round(fvg[0], 2), "FVG_Top": round(fvg[1], 2),
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
        r = analyze_triangle_fvg(sym, daily_map[sym])
        if r is None: continue
        r["Ticker"] = sym
        results.append(r)
        print(f"  OK {sym:<6} ${r['Price']:.2f}  Score {r['Score']}  {r['Stage']}  leg +{r['Leg_%']}%  "
              f"drop {r['Fake_Drop_Date']}  target {r['Target_Measured']} ({r['Reward_R_Measured']}R)")
    except Exception: pass

fc = FUNNEL_COUNTS
print(f"\n{'=' * 65}\n  SCAN COMPLETE - {len(results)} matches from {got_daily} tickers with data")
print(f"  FUNNEL (last {CFG['signal_lookback_days']} session(s))")
print(f"  Tickers checked                                 : {fc['tickers_checked']}")
print(f"  Step 1 - impulse leg (+{CFG['leg_min_pct']:g}%) with an FVG     : {fc['passed_step1_impulse_leg_fvg']}")
print(f"  Step 2 - + converging triangle after the leg    : {fc['passed_step2_triangle']}")
print(f"  Step 3 - + fake drop below the triangle into FVG: {fc['passed_step3_fake_drop_into_fvg']}")
print(f"  Step 4 - + FVG held since                       : {fc['passed_step4_fvg_held']}")
print(f"  Step 5 - + FVG entry or breakout today          : {fc['passed_step5_trigger_today']}  (= full pattern)")
print("=" * 65)

results.sort(key=lambda x: x["Score"], reverse=True)

# -- Save CSV + TradingView list
ts      = datetime.today().strftime("%Y%m%d_%H%M")
out_dir = os.environ.get("GITHUB_WORKSPACE", os.getcwd())
COLS = ["Ticker","Price","Score","Stage","Entry_Price","Stop_Loss","Risk_%","Target_Resistance","Reward_R",
        "Target_Measured","Reward_R_Measured","Leg_Low","Leg_Low_Date","Leg_High","Leg_High_Date","Leg_%",
        "Triangle_Bars","Upper_Line","Lower_Line","Fake_Drop_Date","FVG_Bottom","FVG_Top",
        "Above_SMA200","Signal_Date","Days_Since_Signal","Flags"]
df_out = pd.DataFrame([{k: v for k, v in r.items() if not k.startswith("_")}
                       for r in results]) if results else pd.DataFrame(columns=COLS)
df_out = df_out[[c for c in COLS if c in df_out.columns]].reset_index(drop=True)

fpath = os.path.join(out_dir, f"daily_triangle_fvg_breakout_{ts}.csv")
try:
    from position_sizing import add_position_size
    df_out = add_position_size(df_out)
    try: __import__("best_picks").rank_best(df_out)
    except Exception as _bp_err: print(f"  Best 10 skipped: {_bp_err}")
except Exception as _ps_err:
    print(f"  Position sizing skipped: {_ps_err}")
df_out.to_csv(fpath, index=False)
print(f"\n  CSV -> {fpath}")
tv = os.path.join(out_dir, f"tv_daily_triangle_fvg_breakout_{ts}.txt")
with open(tv, "w") as f:
    f.write(f"###Triangle + FVG {datetime.today().strftime('%Y-%m-%d')}\n")
    for r in results: f.write(f"NASDAQ:{r['Ticker']}\n")
print(f"  TradingView -> {tv}")
if results:
    print(f"\n  Tickers: {', '.join(r['Ticker'] for r in results)}")

# -- Email with CSV attached
EMAIL_COLS = ["Ticker","Price","Score","Stage","Entry_Price","Stop_Loss","Risk_%","Target_Resistance","Reward_R",
              "Target_Measured","Reward_R_Measured","Leg_%","Triangle_Bars","FVG_Bottom","FVG_Top","Fake_Drop_Date"]

def _fmt_cell(col, v):
    try:
        if col in ("Price","Entry_Price","Stop_Loss","Target_Resistance","Target_Measured","FVG_Bottom","FVG_Top"): return f"${float(v):.2f}"
        if col == "Risk_%": return f"{float(v):.1f}%"
        if col == "Leg_%": return f"+{float(v):.0f}%"
        if col in ("Reward_R","Reward_R_Measured"): return f"{float(v):.1f}R"
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
                if c == "Target_Measured": sty = "color:#2563eb;font-weight:600"
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
<h1 style="margin:0;color:#60a5fa;font-size:19px">Triangle Breakout + FVG Entry</h1>
<p style="margin:6px 0 0;color:#94a3b8;font-size:12px">{datetime.today().strftime('%Y-%m-%d %H:%M UTC')} &middot; {cnt} match{'es' if cnt != 1 else ''}
 - an impulse leg left a fair value gap, price coiled in a triangle, a fake drop below the triangle tagged the FVG and held,
 and today price is back in the FVG (FVG entry) or closed above the triangle (Breakout)</p>
</td></tr>
<tr><td style="padding:12px 26px;background:#0b1220"><p style="margin:0;color:#cbd5e1;font-size:12px">
FUNNEL: {fc['tickers_checked']} checked &rarr; {fc['passed_step1_impulse_leg_fvg']} impulse leg with FVG
&rarr; {fc['passed_step2_triangle']} + triangle
&rarr; {fc['passed_step3_fake_drop_into_fvg']} + fake drop into FVG
&rarr; {fc['passed_step4_fvg_held']} + FVG held
&rarr; <b style="color:#facc15">{fc['passed_step5_trigger_today']} + FVG entry / breakout today = full match</b></p></td></tr>
<tr><td style="padding:16px"><div style="overflow-x:auto;border:1px solid #e2e8f0;border-radius:8px">
<table style="border-collapse:collapse;width:100%"><thead><tr>{th}</tr></thead><tbody>{rows}</tbody></table></div>
{tick_html}
<p style="font-size:11px;color:#64748b;margin:10px 0 0">Entry = today's close &middot; Stop = below the FVG (FVG bottom - {CFG['stop_buffer_atr']:g} ATR) &middot;
Target_Measured = the upper triangle line today (breakout point) + the leg height (leg high - leg low) &middot;
Target_Resistance = the leg high (top of the triangle), the first resistance.
Stage: FVG entry = back in the FVG after the fake drop, not broken out yet (earlier, tighter stop); Breakout = first close above the
upper triangle line after the drop (confirmed, wider stop). Leg: +{CFG['leg_min_pct']:g}% and {CFG['leg_min_atr']:g} ATR within {CFG['leg_max_bars']} sessions;
triangle {CFG['tri_min_bars']}-{CFG['tri_max_bars']} sessions, narrowing at least {CFG['tri_min_narrow']*100:.0f}%.
Full results (with Quantity for a fixed max loss) attached as CSV.</p>
</td></tr>
<tr><td style="background:#f8fafc;padding:10px;text-align:center;color:#94a3b8;font-size:10px">
Not financial advice &middot; Auto-generated by GitHub Actions</td></tr>
</table></td></tr></table></body></html>"""

        plain = [f"Triangle Breakout + FVG Entry - {datetime.today().strftime('%Y-%m-%d')}",
                 f"{cnt} matches",
                 f"FUNNEL: {fc['tickers_checked']} checked -> {fc['passed_step1_impulse_leg_fvg']} leg+FVG -> "
                 f"{fc['passed_step2_triangle']} triangle -> {fc['passed_step3_fake_drop_into_fvg']} fake drop -> "
                 f"{fc['passed_step4_fvg_held']} held -> {fc['passed_step5_trigger_today']} trigger today (=full match)",
                 "=" * 60]
        for r in rl[:50]:
            plain.append(f"{r['Ticker']:<6} ${r['Price']:.2f}  Score:{r['Score']}  {r['Stage']}  Entry:${r['Entry_Price']:.2f}  "
                         f"SL:${r['Stop_Loss']:.2f} ({r['Risk_%']}%)  T1:${r['Target_Resistance']:.2f} ({r['Reward_R']}R)  "
                         f"Measured:${r['Target_Measured']:.2f} ({r['Reward_R_Measured']}R)  leg +{r['Leg_%']}%  drop {r['Fake_Drop_Date']}")
        if rl:
            plain += ["", "Tickers: " + ", ".join(r["Ticker"] for r in rl)]
        else:
            plain.append("No matches today")
        plain.append("\nFull results in CSV attachment.")

        subj = (f"Triangle + FVG Entry - {cnt} signal{'s' if cnt != 1 else ''} - "
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

# -- Charts for top 5: last 90 daily candles, triangle lines, FVG, stop, targets
if results:
    top = results[:5]
    fig, axes = plt.subplots(len(top), 1, figsize=(13, 4.2*len(top)), facecolor="#0f172a")
    if len(top) == 1: axes = [axes]
    for ax, r in zip(axes, top):
        full = r["_df"]
        d = full.iloc[-90:]
        n_ = len(d)
        off = len(full) - n_
        x = np.arange(n_)
        ax.set_facecolor("#0f172a")
        cols = ["#22c55e" if c > o else "#ef4444" for o, c in zip(d["Open"], d["Close"])]
        ax.vlines(x, d["Low"], d["High"], color=cols, lw=0.8)
        ax.bar(x, (d["Close"] - d["Open"]).abs().clip(lower=1e-9), bottom=np.minimum(d["Open"], d["Close"]),
               color=cols, width=0.6)
        ax.axhspan(r["FVG_Bottom"], r["FVG_Top"], color="#ef4444", alpha=0.15)
        sg = r["_sig"]; tri = sg["tri"]; last = len(full) - 1
        p0, j1 = sg["p"], tri["j1"]
        if p0 >= off:
            ax.plot([p0 - off, last - off], [tri["U"](p0), tri["U"](last)], color="#e2e8f0", lw=1.0)
        if j1 >= off:
            ax.plot([j1 - off, last - off], [tri["L"](j1), tri["L"](last)], color="#e2e8f0", lw=1.0)
        k_ = np.where(d.index == pd.Timestamp(r["Fake_Drop_Date"]))[0]
        if len(k_): ax.axvline(k_[0], color="#334155", ls="--", lw=0.8); ax.text(k_[0], d["High"].max(), "fake drop", color="#cbd5e1", fontsize=7, ha="center", va="bottom")
        for lvl, c_, lab in ((r["Stop_Loss"], "#ef4444", "Stop"), (r["Leg_High"], "#facc15", "Leg high"),
                             (r["Target_Measured"], "#38bdf8", "Measured")):
            ax.axhline(lvl, color=c_, ls=":", lw=0.9)
            ax.text(n_ - 0.4, lvl, f" {lab} {lvl:.2f}", color=c_, fontsize=7, va="center")
        ax.set_xticks(x[::10]); ax.set_xticklabels([t.strftime("%m/%d") for t in d.index[::10]], fontsize=7)
        ax.set_title(f"{r['Ticker']} | USD {r['Price']:.2f} | Score {r['Score']} | {r['Stage']} | leg +{r['Leg_%']:.0f}% | "
                     f"target {r['Target_Measured']:.2f} ({r['Reward_R_Measured']}R) | {r['Signal_Date']}",
                     color="#e2e8f0", fontsize=9, fontweight="bold")
        ax.tick_params(colors="#94a3b8", labelsize=7)
        for sp in ax.spines.values(): sp.set_edgecolor("#1e3a5f")
        ax.grid(color="#1e3a5f", ls="--", lw=0.5, alpha=0.6)
    plt.suptitle(f"Triangle Breakout + FVG Entry - {datetime.today().strftime('%Y-%m-%d')}",
                 color="#60a5fa", fontsize=12, fontweight="bold")
    plt.tight_layout()
    cp = os.path.join(out_dir, f"daily_triangle_fvg_breakout_chart_{ts}.png")
    plt.savefig(cp, dpi=130, bbox_inches="tight", facecolor="#0f172a")
    plt.close()
    print(f"  Chart -> {cp}")

print(f"""
{'=' * 55}
  PATTERN (latest session)
  1) Impulse leg: +{CFG['leg_min_pct']:g}% and {CFG['leg_min_atr']:g} ATR within {CFG['leg_max_bars']} sessions, leaving a bullish FVG
  2) Triangle after the leg high: {CFG['tri_min_bars']}-{CFG['tri_max_bars']} sessions, falling/flat top, rising/flat bottom, narrowing
  3) Fake drop: a close below the lower line (last {CFG['drop_lookback']} sessions) with the low in the FVG
  4) FVG held: no close below the FVG bottom since
  5) Today: back in the FVG (FVG entry) or first close above the upper line (Breakout)
  OUTPUT: Entry = close, Stop = FVG bottom - {CFG['stop_buffer_atr']:g} ATR,
          Target_Measured = breakout point + leg height, Target_Resistance = leg high
{'=' * 55}
""")
