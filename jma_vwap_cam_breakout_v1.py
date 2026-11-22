# ============================================================
# NASDAQ / NYSE - JMA Turn -> VWAP + Weekly Camarilla Breakout (G-R-G-Doji-G) - daily, v1
# ============================================================
#
# From Vik's chart (2026-11-22): yellow = 20-day VWAP, green/red line = JMA, orange =
# weekly Camarilla levels. The exact 5-candle sequence, c4 = the latest candle:
#   c0  BIG GREEN below VWAP: body >= 0.5 ATR(14) and >= 50% of its range, closes above
#       JMA, and JMA turns from falling to rising at c0, c1 or c2 (red -> green) and keeps
#       rising to c4.
#   c1  SMALL RED pause: body <= half of c0's body, low above c0's open, close in the
#       top half of c0; still below VWAP.
#   c2  BIG GREEN BREAKOUT: body >= 0.5 ATR and >= 50% of its range, at least twice c1's
#       body, closes above VWAP (opened below it) and above c0's high, and its body
#       crosses at least one weekly Camarilla level (opens below it, closes above it).
#   c3  DOJI hold: body <= 35% of its range and <= 0.5 ATR, close above VWAP and above
#       that Camarilla level, low above c2's midpoint.
#   c4  ENTRY: green, closes above c3's high and above VWAP.
# Weekly Camarilla from the previous completed Friday week (C, H, L, range = H - L):
#   H1..H4 = C + 1.1 x range / 12, /6, /4, /2;  L1..L4 = C - the same.
# VWAP = rolling 20-day VWAP.
# Entry = c4 close | Stop = min(c3 low, c4 low) - 0.1 ATR | T1 = the next weekly Camarilla
# level above the entry (else 2R) | T2 = 2R. Liquidity $5 / 200k.
# TradingView list "JMA VWAP Cam - YYYY-MM-DD" (IST). Settings in CFG.
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
    "history_days"     : 400,
    "min_bars"         : 120,
    "big_body_atr"     : 0.5,     # big candle: body >= 0.5 ATR(14) ...
    "big_body_range"   : 0.5,     # ... and >= 50% of its range
    "pause_max"        : 0.5,     # c1 body <= half of c0 body
    "breakout_vs_pause": 2.0,     # c2 body >= 2x c1 body
    "doji_body_range"  : 0.35,
    "doji_body_atr"    : 0.5,
    "vwap_period"      : 20,
    "atr_period"       : 14,
    "stop_buffer_atr"  : 0.1,
    "min_price"        : 5.0,
    "min_avg_volume"   : 200_000,
    "tv_list_prefix"   : "JMA VWAP Cam",
    "batch_size"       : 50,
    "batch_sleep"      : 1.5,
}

FUNNEL = {"tickers_checked": 0, "c4_entry": 0, "c3_doji": 0, "c2_breakout": 0, "c2_camarilla": 0,
          "c1_pause": 0, "c0_jma_turn": 0}

def calc_jma(series, period=13, phase=40, power=2):
    """Jurik Moving Average, same formula as TradingView's "Jurik Moving Average" indicator."""
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

def wilder_rsi(close, n=14):
    d = close.diff()
    up = d.clip(lower=0).ewm(alpha=1 / n, adjust=False).mean()
    dn = (-d.clip(upper=0)).ewm(alpha=1 / n, adjust=False).mean()
    rs = up / dn.replace(0, np.nan)
    return (100 - 100 / (1 + rs)).fillna(100).values

CAM_LEVELS = (("L4", -2), ("L3", -4), ("L2", -6), ("L1", -12), ("H1", 12), ("H2", 6), ("H3", 4), ("H4", 2))

def weekly_camarilla(df):
    """L4..L1, H1..H4 from the previous completed Friday week, aligned to each daily bar."""
    per = df.index.to_period("W-FRI")
    g = df.groupby(per).agg(H=("High", "max"), L=("Low", "min"), C=("Close", "last"))
    p = g.shift(1); rng = p.H - p.L
    lv = pd.DataFrame({k: p.C + np.sign(d) * rng * 1.1 / abs(d) for k, d in CAM_LEVELS}, index=g.index).reindex(per)
    return {k: lv[k].values.astype(float) for k in lv.columns}

def analyze_stock(sym, df):
    if df is None or len(df) < CFG["min_bars"]: return None
    o, h, l, c, v = (df[k].values.astype(float) for k in ("Open", "High", "Low", "Close", "Volume"))
    i = len(df) - 1
    if c[i] < CFG["min_price"] or float(np.mean(v[-20:])) < CFG["min_avg_volume"]: return None
    cs = df["Close"].astype(float)
    tp = (df["High"] + df["Low"] + df["Close"]) / 3
    vw = ((tp * df["Volume"]).rolling(CFG["vwap_period"]).sum() / df["Volume"].rolling(CFG["vwap_period"]).sum()).values
    pc = np.concatenate([[np.nan], c[:-1]])
    tr = np.nanmax(np.vstack([h - l, np.abs(h - pc), np.abs(l - pc)]), axis=0)
    atr = pd.Series(tr).rolling(CFG["atr_period"]).mean().values
    if not (np.isfinite(vw[i - 5]) and np.isfinite(atr[i - 5])): return None
    FUNNEL["tickers_checked"] += 1
    k0, k1, k2, k3, k4 = i - 4, i - 3, i - 2, i - 1, i
    body = np.abs(c - o); rng = h - l
    big = lambda k: c[k] > o[k] and body[k] >= CFG["big_body_atr"] * atr[k] and rng[k] > 0 and body[k] >= CFG["big_body_range"] * rng[k]
    # c4 entry
    if not (c[k4] > o[k4] and c[k4] > h[k3] and c[k4] > vw[k4]): return None
    FUNNEL["c4_entry"] += 1
    # c3 doji holding above VWAP and c2's midpoint
    mid2 = (o[k2] + c[k2]) / 2
    if not (rng[k3] > 0 and body[k3] <= CFG["doji_body_range"] * rng[k3] and body[k3] <= CFG["doji_body_atr"] * atr[k3]
            and c[k3] > vw[k3] and l[k3] > mid2): return None
    FUNNEL["c3_doji"] += 1
    # c2 big green breakout through VWAP and above c0's high
    if not (big(k2) and o[k2] < vw[k2] < c[k2] and c[k2] > h[k0]): return None
    FUNNEL["c2_breakout"] += 1
    cam = weekly_camarilla(df)
    crossed = [n for n, _ in CAM_LEVELS if np.isfinite(cam[n][k2]) and o[k2] < cam[n][k2] < c[k2]
               and c[k3] > cam[n][k3] and c[k4] > cam[n][k4]]
    if not crossed: return None
    FUNNEL["c2_camarilla"] += 1
    # c1 small red pause in the top half of c0, below VWAP
    if not (c[k1] < o[k1] and body[k1] <= CFG["pause_max"] * body[k0] and l[k1] > o[k0]
            and c[k1] >= (o[k0] + c[k0]) / 2 and c[k1] < vw[k1] and body[k2] >= CFG["breakout_vs_pause"] * body[k1]): return None
    FUNNEL["c1_pause"] += 1
    # c0 big green below VWAP, above JMA, JMA turning up at c0 or c1 and rising since
    jma = calc_jma(cs).values
    tturn = next((t for t in (k0, k1, k2) if jma[t] > jma[t - 1] and jma[t - 1] <= jma[t - 2]), None)
    if not (big(k0) and c[k0] < vw[k0] and c[k0] > jma[k0] and tturn is not None
            and all(jma[t] > jma[t - 1] for t in range(max(tturn, k1), k4 + 1))): return None
    FUNNEL["c0_jma_turn"] += 1

    a = float(atr[i]); entry = float(c[i]); stop = float(min(l[k3], l[k4])) - CFG["stop_buffer_atr"] * a
    risk = entry - stop
    if risk <= 0: return None
    above = sorted((cam[n][i], n) for n, _ in CAM_LEVELS if np.isfinite(cam[n][i]) and cam[n][i] > entry)
    t1, src = (float(above[0][0]), f"weekly {above[0][1]}") if above else (entry + 2 * risk, "2R")
    vol_up = v[k2] > v[k1] and v[k4] > v[k3]
    vavg = float(np.mean(v[i - 20:i])); vx2 = v[k2] / vavg if vavg > 0 else np.nan
    score = 55 + (10 if vol_up else 0) + (min(10, (vx2 - 1) * 10) if np.isfinite(vx2) and vx2 > 1 else 0) \
            + min(10, (t1 - entry) / risk * 3) + (5 if len(crossed) > 1 else 0) + (5 if risk / entry < 0.05 else 0)
    flags = [f"breakout candle crossed VWAP {vw[k2]:.2f} and weekly {'/'.join(crossed)}",
             f"breakout volume x{vx2:.1f} 20d avg" if np.isfinite(vx2) else "volume n/a",
             "volume up on both green pushes" if vol_up else "volume not up on both pushes",
             f"JMA turned up on c{tturn - k0}"]
    return {"Score": round(min(100, score)), "Levels_Crossed": "/".join(crossed),
            "Price": round(entry, 2), "Entry_Price": round(entry, 2), "Stop_Loss": round(stop, 2),
            "Risk_%": round(risk / entry * 100, 2), "Target_1": round(t1, 2), "T1_Source": src,
            "Reward_R": round((t1 - entry) / risk, 2), "Target_2R": round(entry + 2 * risk, 2),
            "VWAP20": round(float(vw[i]), 2), "JMA": round(float(jma[i]), 2),
            "W_H3": round(float(cam["H3"][i]), 2), "W_H4": round(float(cam["H4"][i]), 2),
            "Breakout_Date": df.index[k2].strftime("%Y-%m-%d"), "Breakout_Vol_x": round(vx2, 2) if np.isfinite(vx2) else None,
            "Signal_Date": df.index[i].strftime("%Y-%m-%d"), "Flags": " | ".join(flags)}

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
        r = analyze_stock(sym, daily_map[sym])
        if r: r["Ticker"] = sym; results.append(r)
    except Exception: pass

results.sort(key=lambda x: -x["Score"])
GROUPS = [("JMA turn -> VWAP + weekly Camarilla breakout", lambda r: True)]
for r in results:
    r["Group"] = next(g for g, f in GROUPS if f(r))
fc = FUNNEL
print(f"\n{'=' * 65}\n  SCAN COMPLETE - {len(results)} matches from {got_daily} tickers with data")
print(f"  checked {fc['tickers_checked']} -> c4 entry {fc['c4_entry']} -> c3 doji {fc['c3_doji']} -> c2 VWAP breakout {fc['c2_breakout']}"
      f" -> c2 crossed weekly Camarilla {fc['c2_camarilla']} -> c1 pause {fc['c1_pause']} -> c0 + JMA turn {fc['c0_jma_turn']}")
for g, _ in GROUPS:
    names = [r["Ticker"] for r in results if r["Group"] == g]
    print(f"  {g}: {len(names)}  {', '.join(names[:60])}")
print("=" * 65)

ts      = datetime.today().strftime("%Y%m%d_%H%M")
out_dir = os.environ.get("GITHUB_WORKSPACE", os.getcwd())
COLS = ["Ticker","Group","Levels_Crossed","Score","Price","Entry_Price","Stop_Loss","Risk_%","Target_1","T1_Source","Reward_R",
        "Target_2R","VWAP20","JMA","W_H3","W_H4","Breakout_Date","Breakout_Vol_x","Signal_Date","Flags"]
df_out = pd.DataFrame(results) if results else pd.DataFrame(columns=COLS)
df_out = df_out[[c for c in COLS if c in df_out.columns]].reset_index(drop=True)
fpath = os.path.join(out_dir, f"jma_vwap_cam_breakout_{ts}.csv")
try:
    from position_sizing import add_position_size
    df_out = add_position_size(df_out)
    try: __import__("best_picks").rank_best(df_out)
    except Exception as _bp_err: print(f"  Best 10 skipped: {_bp_err}")
except Exception as _ps_err:
    print(f"  Position sizing skipped: {_ps_err}")
df_out.to_csv(fpath, index=False)
print(f"\n  CSV -> {fpath}")
tv = os.path.join(out_dir, f"tv_jma_vwap_cam_breakout_{ts}.txt")
with open(tv, "w") as f:
    for g, _ in GROUPS:
        names = [r["Ticker"] for r in results if r["Group"] == g]
        if names: f.write(f"###{g}\n"); f.writelines(f"NASDAQ:{s}\n" for s in names)
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
    for g, _ in GROUPS:
        syms = [tvs.tv_symbol(r["Ticker"], ex) for r in results if r["Group"] == g]
        if syms: symbols.append(f"###{g.split(' - ')[0]} ({len(syms)})"); symbols += syms
    tvs.replace_list(s, lists, symbols)
    after = next((x for x in tvs.get_lists(s) if x.get("name") == name), None)
    got = [x for x in (after or {}).get("symbols") or [] if not str(x).startswith("###")]
    TV_STATUS = f"'{name}' - {len(got)} symbols"
try:
    _tv_sync()
except Exception as e:
    TV_STATUS = f"failed ({type(e).__name__}: {e})"
print(f"[TradingView] {TV_STATUS}")

EMAIL_COLS = ["Ticker","Levels_Crossed","Score","Price","Stop_Loss","Risk_%","Target_1","T1_Source","Reward_R",
              "VWAP20","Breakout_Date","Breakout_Vol_x"]
def _table(rows):
    th = "".join(f'<th style="background:#1e293b;color:#e2e8f0;padding:6px 8px;font-size:11px;white-space:nowrap">{c}</th>' for c in EMAIL_COLS)
    body = ""
    for i, r in enumerate(rows):
        bg = "#fff" if i % 2 == 0 else "#f0f9ff"
        body += f'<tr style="background:{bg}">' + "".join(
            f'<td style="padding:5px 8px;font-size:12px;{"font-weight:700" if c in ("Ticker", "Levels_Crossed") else ""}">{r.get(c, "-")}</td>'
            for c in EMAIL_COLS) + "</tr>"
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
        for g, _ in GROUPS:
            rows = [r for r in rl if r["Group"] == g]
            secs += f'<h3 style="font-size:14px;margin:12px 0 6px">{g} ({len(rows)})</h3>'
            secs += _table(rows[:80]) if rows else '<p style="color:#94a3b8;font-size:12px">none</p>'
        html = f"""<!DOCTYPE html><html><body style="margin:0;background:#f1f5f9;font-family:'Segoe UI',Arial,sans-serif">
<table width="100%" cellpadding="0" cellspacing="0"><tr><td align="center" style="padding:20px 10px">
<table width="100%" style="max-width:960px;background:#fff;border-radius:12px;overflow:hidden">
<tr><td style="background:linear-gradient(135deg,#0f172a,#1e3a5f);padding:20px 26px">
<h1 style="margin:0;color:#60a5fa;font-size:19px">JMA Turn &rarr; VWAP + Weekly Camarilla Breakout (G-R-G-Doji-G)</h1>
<p style="margin:6px 0 0;color:#94a3b8;font-size:12px">{datetime.today().strftime('%Y-%m-%d %H:%M UTC')} &middot; {len(rl)} matches
&middot; TradingView: {TV_STATUS}</p></td></tr>
<tr><td style="padding:16px 22px">{secs}
<p style="font-size:11px;color:#64748b">Five candles, the last one today: c0 big green below VWAP closing above JMA (JMA turns up by c2) &middot;
c1 small red pause in the top half of c0 &middot; c2 big green breakout that opens below and closes above the {CFG['vwap_period']}-day VWAP and above c0's high, crossing
a weekly Camarilla level &middot; c3 doji holding above VWAP and that level &middot; c4 green closing above the doji's high = entry.
Entry = today's close &middot; Stop = lower of the doji's and today's low - {CFG['stop_buffer_atr']:g} ATR &middot; T1 = next weekly Camarilla level above (else 2R) &middot; T2 = 2R.
Funnel: checked {FUNNEL['tickers_checked']} &rarr; c4 entry {FUNNEL['c4_entry']} &rarr; c3 doji {FUNNEL['c3_doji']} &rarr; c2 VWAP breakout {FUNNEL['c2_breakout']}
&rarr; c2 Camarilla cross {FUNNEL['c2_camarilla']} &rarr; c1 pause {FUNNEL['c1_pause']} &rarr; c0 + JMA turn {FUNNEL['c0_jma_turn']}.
Full results (with Quantity for a fixed max loss) attached.</p></td></tr>
<tr><td style="background:#f8fafc;padding:10px;text-align:center;color:#94a3b8;font-size:10px">Not financial advice &middot; Auto-generated by GitHub Actions</td></tr>
</table></td></tr></table></body></html>"""
        plain = [f"JMA VWAP Camarilla Breakout - {datetime.today().strftime('%Y-%m-%d')}", f"{len(rl)} matches", f"TradingView: {TV_STATUS}", ""]
        for g, _ in GROUPS:
            plain.append(f"{g}: " + ", ".join(r["Ticker"] for r in rl if r["Group"] == g))
        subj = f"JMA + VWAP + Weekly Camarilla Breakout - {len(rl)} signals - {datetime.today().strftime('%Y-%m-%d')}"
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
