# ============================================================
# NASDAQ / NYSE - RGGRRG / RGGGRRG / GRRG Pullback Reclaim - Daily + 4H, v1
# ============================================================
#
# Vik's candle patterns (G = green: close > open, R = red: close < open),
# ending on the latest candle, checked on DAILY and on 4-HOUR bars:
#
#   RGGRRG   red, green, green (2nd green volume > 1st green volume),
#            red, red, green whose CLOSE is above the HIGH of the red candle before it
#   RGGGRRG  red, green, green, green (2nd green volume > 1st green volume),
#            red, red, green whose CLOSE is above the HIGH of the red candle before it
#   GRRG     STRONG green on HIGH volume (body >= 60% of range and >= 0.8 ATR(14),
#            volume >= 1.5x the 20-bar average), red, red, green whose CLOSE is
#            above the HIGH of the red candle before it
#
# If a candle sequence fits more than one, the longest pattern is reported.
# Preferred: close above SMA50 of that timeframe (grouped first, +20 score) -
# below-SMA50 matches are listed separately, not dropped.
#
# Recency: daily = the final green is the latest daily candle;
#          4H    = the final green is one of the latest session's 4H candles, and the
#                  latest close is still above that red candle's high.
# 4H bars = regular-session 60-minute bars from Yahoo grouped like TradingView:
#   09:30-13:30 and 13:30-16:00 ET. 4H volume is compared per hour (the second
#   bar is only 2.5 hours long). An unfinished 4H bar (run during US hours) is dropped.
#
# Entry = final green close | Stop = lowest low of the two reds and the final green
# T1 = high of the green run (Reward_R) | T2 = 2R. Liquidity $5 / 200k.
# TradingView list "RGGRRG - YYYY-MM-DD" (IST). Settings in CFG.
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
    "min_bars"         : 60,
    "atr_period"       : 14,
    "sma_period"       : 50,
    "strong_body_rng"  : 0.60,    # GRRG first green: body >= 60% of its range
    "strong_body_atr"  : 0.80,    #                   body >= 0.8 ATR(14)
    "strong_vol_mult"  : 1.5,     #                   volume >= 1.5x 20-bar average
    "vol_avg_bars"     : 20,
    "intraday_period"  : "60d",   # 60-minute history for 4H bars
    "patterns"         : ("RGGGRRG", "RGGRRG", "GRRG"),
    "min_price"        : 5.0,
    "min_avg_volume"   : 200_000,
    "tv_list_prefix"   : "RGGRRG",
    "batch_size"       : 50,
    "batch_sleep"      : 1.5,
}

FUNNEL = {"daily_checked": 0, "daily_hits": 0, "h4_data": 0, "h4_hits": 0,
          "RGGGRRG": 0, "RGGRRG": 0, "GRRG": 0}

def _colors(o, c):
    return np.where(c > o, "G", np.where(c < o, "R", "D"))

def detect(df, recent=1, tf="Daily"):
    """Return the pattern dict for the latest candles of df, or None."""
    if df is None or len(df) < CFG["min_bars"]: return None
    o, h, l, c = (df[k].values.astype(float) for k in ("Open", "High", "Low", "Close"))
    hours = df["Hours"].values.astype(float) if "Hours" in df.columns else np.ones(len(df))
    v = df["Volume"].values.astype(float) / np.maximum(hours, 0.5)      # 4H: volume per hour
    n = len(df)
    pc = np.concatenate([[np.nan], c[:-1]])
    tr = np.nanmax(np.vstack([h - l, np.abs(h - pc), np.abs(l - pc)]), axis=0)
    atr = pd.Series(tr).rolling(CFG["atr_period"]).mean().values
    sma = pd.Series(c).rolling(CFG["sma_period"]).mean().values
    vavg = pd.Series(v).rolling(CFG["vol_avg_bars"]).mean().shift(1).values   # prior bars only
    col = _colors(o, c)
    for e in range(n - 1, n - 1 - recent, -1):
        if e < 8: break
        for pat in CFG["patterns"]:
            k = len(pat); s = e - k + 1
            if s < 1 or "".join(col[s:e + 1]) != pat: continue
            r2 = e - 1                                     # red candle just before the final green
            if not c[e] > h[r2]: continue
            if c[n - 1] <= h[r2]: continue                 # 4H: still above it now
            greens = [i for i in range(s, e - 2) if col[i] == "G"]
            g1 = greens[0]
            if pat.startswith("R"):
                if not v[greens[1]] > v[greens[0]]: continue
                vol_note = f"GG vol x{v[greens[1]] / max(v[greens[0]], 1):.2f}"
                if len(greens) == 3:
                    vol_note += " (3rd green also higher)" if v[greens[2]] > v[greens[1]] else " (3rd green lower)"
            else:
                body = c[g1] - o[g1]; rng = h[g1] - l[g1]
                if not (np.isfinite(atr[g1]) and np.isfinite(vavg[g1])): continue
                if not (rng > 0 and body >= CFG["strong_body_rng"] * rng and body >= CFG["strong_body_atr"] * atr[g1]
                        and v[g1] >= CFG["strong_vol_mult"] * vavg[g1]): continue
                vol_note = f"strong G: body {body / atr[g1]:.1f} ATR, vol x{v[g1] / vavg[g1]:.1f} avg"
            if not np.isfinite(sma[e]) or not np.isfinite(atr[e]): continue
            stop = float(min(l[e - 2:e + 1]))
            entry = float(c[e]); risk = entry - stop
            if risk <= 0: continue
            run_high = float(max(h[i] for i in greens))
            t1 = run_high if run_high > entry else entry + 2 * risk
            above = c[e] > sma[e]
            sma_up = sma[e] > sma[max(0, e - 10)]
            g_vol = np.mean([v[i] for i in greens]); rr_vol = (v[e - 2] + v[e - 1]) / 2
            light_pullback = rr_vol < g_vol
            above_both = c[e] > max(h[e - 2], h[e - 1])
            rng_e = h[e] - l[e]
            close_pos = (c[e] - l[e]) / rng_e if rng_e > 0 else 0.5
            fv = v[e] / vavg[e] if np.isfinite(vavg[e]) and vavg[e] > 0 else np.nan
            score = 45 + (20 if above else 0) + (10 if sma_up else 0) + (10 if light_pullback else 0) \
                    + (5 if above_both else 0) + (5 if close_pos >= 0.7 else 0) \
                    + (min(5, max(0, (fv - 1) * 5)) if np.isfinite(fv) else 0)
            flags = [vol_note, "RR volume lighter than greens" if light_pullback else "RR volume heavier",
                     "closed above both red highs" if above_both else "above last red high only",
                     f"close in top {100 - close_pos * 100:.0f}% of range", "SMA50 rising" if sma_up else "SMA50 falling"]
            fmt = "%Y-%m-%d" if tf == "Daily" else "%Y-%m-%d %H:%M"
            return {"TF": tf, "Pattern": pat, "Score": round(min(100, score)),
                    "Signal_Bar": df.index[e].strftime(fmt), "Price": round(float(c[n - 1]), 2),
                    "Entry_Price": round(entry, 2), "Stop_Loss": round(stop, 2), "Risk_%": round(risk / entry * 100, 2),
                    "Target_1": round(t1, 2), "Reward_R": round((t1 - entry) / risk, 2), "Target_2R": round(entry + 2 * risk, 2),
                    "Red_High": round(float(h[r2]), 2), "Above_SMA50": "Yes" if above else "No",
                    "vs_SMA50_%": round((c[e] / sma[e] - 1) * 100, 1), "Final_G_Vol_x": round(fv, 2) if np.isfinite(fv) else None,
                    "Flags": " | ".join(flags)}
    return None

def build_4h(d1):
    """60-minute regular-session bars -> TradingView-style 4H bars (09:30-13:30, 13:30-16:00 ET)."""
    if d1 is None or d1.empty: return None
    d = d1.copy()
    if d.index.tz is None: d.index = d.index.tz_localize("America/New_York")
    else: d.index = d.index.tz_convert("America/New_York")
    mins = d.index.hour * 60 + d.index.minute
    d = d[(mins >= 570) & (mins < 960)]
    if d.empty: return None
    mins = d.index.hour * 60 + d.index.minute
    key = d.index.tz_localize(None).normalize() + pd.to_timedelta(np.where(mins < 810, 570, 810), unit="m")
    g = d.groupby(key)
    out = pd.DataFrame({"Open": g["Open"].first(), "High": g["High"].max(), "Low": g["Low"].min(),
                        "Close": g["Close"].last(), "Volume": g["Volume"].sum(), "Hours": g["Close"].count()})
    out.index = pd.DatetimeIndex(out.index)            # ET wall-clock time, tz-naive
    if out.empty: return None
    # drop an unfinished last bar while the US session is open
    now_et = pd.Timestamp.now(tz="America/New_York").tz_localize(None)
    last = out.index[-1]
    exp = 4 if last.hour * 60 + last.minute == 570 else 3
    if out["Hours"].iloc[-1] < exp and last.date() == now_et.date() and now_et.hour * 60 + now_et.minute < 960:
        out = out.iloc[:-1]
    return out.dropna(subset=["Close"])

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

# -- Daily scan
results = []
liquid = []
for sym, df in tqdm(list(daily_map.items()), desc="Daily pattern", unit="stk"):
    try:
        if float(df["Close"].iloc[-1]) < CFG["min_price"] or float(df["Volume"].tail(20).mean()) < CFG["min_avg_volume"]:
            continue
        liquid.append(sym); FUNNEL["daily_checked"] += 1
        r = detect(df, recent=1, tf="Daily")
        if r: r["Ticker"] = sym; results.append(r); FUNNEL["daily_hits"] += 1; FUNNEL[r["Pattern"]] += 1
    except Exception: pass
print(f"  Daily: {FUNNEL['daily_checked']} liquid tickers -> {FUNNEL['daily_hits']} matches")

# -- 4H scan (60-minute bars grouped into TradingView-style 4H bars)
h1 = _intraday(sorted(liquid), CFG["intraday_period"], "60m")
h4 = {}
for s, d in h1.items():
    try:
        b = build_4h(d)
        if b is not None and len(b) >= CFG["min_bars"]: h4[s] = b
    except Exception: pass
FUNNEL["h4_data"] = len(h4)
last_day = max((b.index[-1].date() for b in h4.values()), default=None)
for s, b in tqdm(list(h4.items()), desc="4H pattern", unit="stk"):
    try:
        r = detect(b, recent=2, tf="4H")
        if r and pd.Timestamp(r["Signal_Bar"]).date() == last_day:
            r["Ticker"] = s; results.append(r); FUNNEL["h4_hits"] += 1; FUNNEL[r["Pattern"]] += 1
    except Exception: pass
print(f"  4H: data for {len(h4)} (latest session {last_day}) -> {FUNNEL['h4_hits']} matches")

tf_of = {}
for r in results: tf_of.setdefault(r["Ticker"], set()).add(r["TF"])
for r in results:
    r["Both_TF"] = "Yes" if len(tf_of[r["Ticker"]]) == 2 else "No"
    if r["Both_TF"] == "Yes": r["Score"] = min(100, r["Score"] + 10)
results.sort(key=lambda x: (-x["Score"], x["TF"]))
GROUPS = [("Both timeframes - daily and 4H", lambda r: r["Both_TF"] == "Yes"),
          ("Daily - above SMA50", lambda r: r["TF"] == "Daily" and r["Above_SMA50"] == "Yes"),
          ("4H - above SMA50", lambda r: r["TF"] == "4H" and r["Above_SMA50"] == "Yes"),
          ("Below SMA50", lambda r: True)]
for r in results:
    r["Group"] = next(g for g, f in GROUPS if f(r))
def _group_names(g):
    """Tickers of group g, each ticker listed only in the first group it appears in."""
    earlier = set()
    for gg, _ in GROUPS:
        if gg == g: break
        earlier |= {r["Ticker"] for r in results if r["Group"] == gg}
    seen, out = set(earlier), []
    for r in results:
        if r["Group"] == g and r["Ticker"] not in seen: seen.add(r["Ticker"]); out.append(r["Ticker"])
    return out
fc = FUNNEL
print(f"\n{'=' * 65}\n  SCAN COMPLETE - {len(results)} matches ({len(tf_of)} stocks) from {got_daily} tickers with data")
print(f"  daily {fc['daily_hits']} | 4H {fc['h4_hits']} | RGGGRRG {fc['RGGGRRG']} | RGGRRG {fc['RGGRRG']} | GRRG {fc['GRRG']}")
for g, _ in GROUPS:
    names = _group_names(g)
    print(f"  {g}: {len(names)}  {', '.join(names[:60])}")
print("=" * 65)

ts      = datetime.today().strftime("%Y%m%d_%H%M")
out_dir = os.environ.get("GITHUB_WORKSPACE", os.getcwd())
COLS = ["Ticker","TF","Pattern","Group","Score","Both_TF","Price","Entry_Price","Stop_Loss","Risk_%","Target_1","Reward_R",
        "Target_2R","Red_High","Above_SMA50","vs_SMA50_%","Final_G_Vol_x","Signal_Bar","Flags"]
df_out = pd.DataFrame(results) if results else pd.DataFrame(columns=COLS)
df_out = df_out[[c for c in COLS if c in df_out.columns]].reset_index(drop=True)
fpath = os.path.join(out_dir, f"rggrrg_pattern_{ts}.csv")
try:
    from position_sizing import add_position_size
    df_out = add_position_size(df_out)
    try: __import__("best_picks").rank_best(df_out.drop_duplicates("Ticker").reset_index(drop=True))
    except Exception as _bp_err: print(f"  Best 10 skipped: {_bp_err}")
except Exception as _ps_err:
    print(f"  Position sizing skipped: {_ps_err}")
df_out.to_csv(fpath, index=False)
print(f"\n  CSV -> {fpath}")
tv = os.path.join(out_dir, f"tv_rggrrg_pattern_{ts}.txt")
with open(tv, "w") as f:
    for g, _ in GROUPS:
        names = _group_names(g)
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
        syms = [tvs.tv_symbol(t, ex) for t in _group_names(g)]
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

EMAIL_COLS = ["Ticker","TF","Pattern","Score","Price","Stop_Loss","Risk_%","Target_1","Reward_R","vs_SMA50_%",
              "Final_G_Vol_x","Signal_Bar"]
def _table(rows):
    th = "".join(f'<th style="background:#1e293b;color:#e2e8f0;padding:6px 8px;font-size:11px;white-space:nowrap">{c}</th>' for c in EMAIL_COLS)
    body = ""
    for i, r in enumerate(rows):
        bg = "#fff" if i % 2 == 0 else "#f0f9ff"
        body += f'<tr style="background:{bg}">' + "".join(
            f'<td style="padding:5px 8px;font-size:12px;{"font-weight:700" if c in ("Ticker", "Pattern") else ""}">{r.get(c, "-")}</td>'
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
<h1 style="margin:0;color:#60a5fa;font-size:19px">RGGRRG / RGGGRRG / GRRG Reclaim (Daily + 4H)</h1>
<p style="margin:6px 0 0;color:#94a3b8;font-size:12px">{datetime.today().strftime('%Y-%m-%d %H:%M UTC')} &middot; {len(rl)} matches
&middot; daily {FUNNEL['daily_hits']} &middot; 4H {FUNNEL['h4_hits']} &middot; TradingView: {TV_STATUS}</p></td></tr>
<tr><td style="padding:16px 22px">{secs}
<p style="font-size:11px;color:#64748b">G = green candle, R = red. <b>RGGRRG</b> and <b>RGGGRRG</b>: the second green's volume is above the first green's.
<b>GRRG</b>: the first green is strong (body &ge; {CFG['strong_body_rng']:.0%} of its range and &ge; {CFG['strong_body_atr']:g} ATR) on volume &ge; {CFG['strong_vol_mult']:g}&times; the 20-bar average.
In all three the final green closes above the high of the red candle before it. Daily: final green = latest session. 4H (09:30-13:30 / 13:30-16:00 ET,
volume compared per hour): final green in the latest session and price still above that red high. Above SMA50 of that timeframe is preferred (listed first, +20 score);
both timeframes +10. Entry = final green close &middot; Stop = lowest low of the two reds and the final green &middot; T1 = high of the green run (2R if already above) &middot; T2 = 2R.
Full results (with Quantity for a fixed max loss) attached.</p></td></tr>
<tr><td style="background:#f8fafc;padding:10px;text-align:center;color:#94a3b8;font-size:10px">Not financial advice &middot; Auto-generated by GitHub Actions</td></tr>
</table></td></tr></table></body></html>"""
        plain = [f"RGGRRG / RGGGRRG / GRRG - {datetime.today().strftime('%Y-%m-%d')}", f"{len(rl)} matches", f"TradingView: {TV_STATUS}", ""]
        for g, _ in GROUPS:
            plain.append(f"{g}: " + ", ".join(f"{r['Ticker']} ({r['TF']} {r['Pattern']})" for r in rl if r["Group"] == g))
        nUp = len({r["Ticker"] for r in rl if r["Above_SMA50"] == "Yes"})
        subj = f"RGGRRG Reclaim (Daily+4H) - {len(rl)} signals, {nUp} above SMA50 - {datetime.today().strftime('%Y-%m-%d')}"
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
