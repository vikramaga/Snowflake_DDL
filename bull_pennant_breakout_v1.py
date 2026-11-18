# ============================================================
# NASDAQ / NYSE - Bull Pennant + Breakout Entry - Daily + Weekly, v1
# ============================================================
#
# From the "Bull Pennant + Breakout Entry" chart (2026-11-18). The signal fires on
# the EXACT BREAKOUT CANDLE - the first close above the pennant.
#
#   1. POLE: a sharp advance of 3-12 bars from the pole low to the pole top -
#      gain >= 15% daily / 20% weekly and >= 3 ATR(14), at least 60% green candles.
#   2. PENNANT: from the pole top, 4-15 daily / 3-10 weekly bars of converging
#      consolidation - the upper line (fitted to the highs, shifted so every high is
#      on or under it) slopes DOWN, the lower line (fitted to the lows, shifted so
#      every low is on or above it) slopes up or less steeply down, and the pennant
#      is narrower at its end than at its start (<= 75%). It holds the upper half of
#      the pole (lowest low above the pole midpoint) and its range is at most 50% of
#      the pole height.
#   3. BREAKOUT = ENTRY: the latest bar is green and CLOSES ABOVE the upper pennant
#      line AND above the highs of the last 3 pennant bars; no earlier pennant bar
#      closed above the line or above its own 3 prior highs (first breakout only).
#
# Entry  = the breakout close
# Stop   = below the last pennant low: lowest low of the last 3 pennant bars - 0.1 ATR
# Target = pole height added to the breakout: entry + (pole top - pole low)
# Weekly bars end Friday and only COMPLETED weeks are used (the in-progress week
# is dropped). Volume dry-up in the pennant and a breakout-volume surge add score.
# Liquidity $5 / 200k. TradingView list "Bull Pennant - YYYY-MM-DD" (IST).
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
    "history_days"     : 500,
    "atr_period"       : 14,
    "pole_min_bars"    : 3,
    "pole_max_bars"    : 12,
    "pole_min_gain"    : {"Daily": 15.0, "Weekly": 20.0},   # % from pole low to pole top
    "pole_min_atr"     : 3.0,
    "pole_green_share" : 0.6,
    "pen_min_bars"     : {"Daily": 4, "Weekly": 3},          # pennant bars incl. the pole-top bar
    "pen_max_bars"     : {"Daily": 15, "Weekly": 10},
    "pen_max_range"    : 0.5,     # pennant high-low <= 50% of the pole height
    "pen_end_width"    : 0.75,    # width at the end <= 75% of the width at the start
    "stop_last_bars"   : 3,
    "stop_buffer_atr"  : 0.1,
    "max_risk_pct"     : 25.0,
    "min_price"        : 5.0,
    "min_avg_volume"   : 200_000,
    "tv_list_prefix"   : "Bull Pennant",
    "batch_size"       : 50,
    "batch_sleep"      : 1.5,
}

FUNNEL = {"Daily": {"checked": 0, "pole": 0, "pennant": 0, "breakout": 0},
          "Weekly": {"checked": 0, "pole": 0, "pennant": 0, "breakout": 0}}

def to_weekly_completed(df):
    """Friday weekly bars; the in-progress week is dropped unless the last bar is a Friday."""
    w = df.resample("W-FRI").agg({"Open": "first", "High": "max", "Low": "min", "Close": "last",
                                  "Volume": "sum"}).dropna(subset=["Close"])
    if len(df) and df.index[-1].weekday() != 4 and len(w):
        w = w.iloc[:-1]
    return w

def _lines(h, l, t, e):
    """Upper / lower pennant lines over bars t..e (inclusive) -> (u_slope, u_icpt, l_slope, l_icpt), x = bar index."""
    x = np.arange(t, e + 1, dtype=float)
    us, ui = np.polyfit(x, h[t:e + 1], 1); ls, li = np.polyfit(x, l[t:e + 1], 1)
    ui += max(0.0, float(np.max(h[t:e + 1] - (us * x + ui))))     # every high on/under the line
    li -= max(0.0, float(np.max((ls * x + li) - l[t:e + 1])))     # every low on/above the line
    return us, ui, ls, li

def detect(df, tf):
    F = FUNNEL[tf]
    o, h, l, c, v = (df[k].values.astype(float) for k in ("Open", "High", "Low", "Close", "Volume"))
    n = len(df); i = n - 1
    if n < 40: return None
    pc = np.concatenate([[np.nan], c[:-1]])
    tr = np.nanmax(np.vstack([h - l, np.abs(h - pc), np.abs(l - pc)]), axis=0)
    atr = pd.Series(tr).rolling(CFG["atr_period"]).mean().values
    F["checked"] += 1
    stage = 0; best = None
    pmin, pmax = CFG["pen_min_bars"][tf], CFG["pen_max_bars"][tf]
    for t in range(i - pmin, i - pmax - 1, -1):                    # pole top = first pennant bar
        if t < CFG["pole_max_bars"] + 2: break
        if h[t] < h[t:i].max(): continue                           # pole top is the pennant's highest high
        lo_a, lo_b = t - CFG["pole_max_bars"], t - CFG["pole_min_bars"]
        b = lo_a + int(np.argmin(l[lo_a:lo_b + 1]))                # pole low
        height = h[t] - l[b]
        if not (np.isfinite(atr[t]) and l[b] > 0 and h[t] >= h[b:t + 1].max()): continue   # top = highest high since the pole low
        gain = height / l[b] * 100
        greens = np.mean(c[b + 1:t + 1] > o[b + 1:t + 1]) if t > b else 0
        if not (gain >= CFG["pole_min_gain"][tf] and height >= CFG["pole_min_atr"] * atr[t]
                and greens >= CFG["pole_green_share"]): continue
        stage = max(stage, 1)
        e = i - 1                                                  # last pennant bar
        pen_lo = l[t:e + 1].min()
        if not (pen_lo > l[b] + 0.5 * height and h[t] - pen_lo <= CFG["pen_max_range"] * height): continue
        us, ui, ls, li = _lines(h, l, t, e)
        w0 = (us * t + ui) - (ls * t + li); w1 = (us * e + ui) - (ls * e + li)
        if not (us < 0 and ls > us and w0 > 0 and w1 <= CFG["pen_end_width"] * w0): continue
        upper = lambda k: us * k + ui
        if any(c[k] > upper(k) or (k >= t + 3 and c[k] > h[k - 3:k].max()) for k in range(t + 1, e + 1)):
            continue                                               # an earlier close already broke out
        stage = max(stage, 2)
        if not (c[i] > o[i] and c[i] > upper(i) and c[i] > h[max(t, e - 2):e + 1].max()): continue
        stage = 3
        best = (t, b, e, height, gain, greens, us, ui, ls, li); break
    if stage >= 1: F["pole"] += 1
    if stage >= 2: F["pennant"] += 1
    if stage < 3: return None
    F["breakout"] += 1
    t, b, e, height, gain, greens, us, ui, ls, li = best
    entry = float(c[i]); a = float(atr[i]) if np.isfinite(atr[i]) else 0.0
    stop = float(l[max(t, e - CFG["stop_last_bars"] + 1):e + 1].min()) - CFG["stop_buffer_atr"] * a
    risk = entry - stop
    if risk <= 0 or risk / entry * 100 > CFG["max_risk_pct"]: return None
    target = entry + height
    pole_vol = float(v[b + 1:t + 1].mean()); pen_vol = float(v[t + 1:e + 1].mean()) if e > t else float(v[t])
    dry = pen_vol < pole_vol
    bvx = v[i] / pen_vol if pen_vol > 0 else np.nan
    tight = (h[t] - l[t:e + 1].min()) / height
    score = 40 + min(15, (gain - CFG["pole_min_gain"][tf]) * 0.5) + (10 if dry else 0) \
            + (min(15, (bvx - 1) * 10) if np.isfinite(bvx) and bvx > 1 else 0) + max(0, 10 - tight * 20) \
            + min(10, (height / risk) * 2)
    fmt = "%Y-%m-%d"
    flags = [f"pole +{gain:.1f}% in {t - b} bars ({greens:.0%} green)", f"pennant {e - t + 1} bars, range {tight:.0%} of pole",
             "volume dried up in the pennant" if dry else "pennant volume not lighter than the pole",
             f"breakout volume x{bvx:.1f} pennant avg" if np.isfinite(bvx) else "breakout volume n/a",
             f"upper line {us * i + ui:.2f} at breakout"]
    return {"TF": tf, "Score": round(min(100, score)), "Price": round(entry, 2), "Entry_Price": round(entry, 2),
            "Stop_Loss": round(stop, 2), "Risk_%": round(risk / entry * 100, 2), "Target": round(target, 2),
            "Reward_R": round(height / risk, 2), "Pole_Low": round(float(l[b]), 2), "Pole_Top": round(float(h[t]), 2),
            "Pole_Gain_%": round(gain, 1), "Pole_Bars": int(t - b), "Pennant_Bars": int(e - t + 1),
            "Pennant_Range_%_of_Pole": round(tight * 100, 1), "Upper_Line": round(float(us * i + ui), 2),
            "Volume_Dry_Up": "Yes" if dry else "No", "Breakout_Vol_x": round(bvx, 2) if np.isfinite(bvx) else None,
            "Pole_Start": df.index[b].strftime(fmt), "Pole_Top_Date": df.index[t].strftime(fmt),
            "Signal_Bar": df.index[i].strftime(fmt), "Flags": " | ".join(flags)}

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
        d = daily_map[sym]
        if float(d["Close"].iloc[-1]) < CFG["min_price"] or float(d["Volume"].tail(20).mean()) < CFG["min_avg_volume"]:
            continue
        for tf, bars in (("Daily", d), ("Weekly", to_weekly_completed(d))):
            r = detect(bars, tf)
            if r: r["Ticker"] = sym; results.append(r)
    except Exception: pass

results.sort(key=lambda x: (x["TF"] != "Daily", -x["Score"]))
GROUPS = [("Daily - breakout today", lambda r: r["TF"] == "Daily"),
          ("Weekly - breakout last completed week", lambda r: True)]
for r in results:
    r["Group"] = next(g for g, f in GROUPS if f(r))
print(f"\n{'=' * 65}\n  SCAN COMPLETE - {len(results)} matches from {got_daily} tickers with data")
for tf in ("Daily", "Weekly"):
    f = FUNNEL[tf]
    print(f"  {tf}: checked {f['checked']} -> pole {f['pole']} -> pennant {f['pennant']} -> breakout {f['breakout']}")
for g, _ in GROUPS:
    names = [r["Ticker"] for r in results if r["Group"] == g]
    print(f"  {g}: {len(names)}  {', '.join(names[:60])}")
print("=" * 65)

ts      = datetime.today().strftime("%Y%m%d_%H%M")
out_dir = os.environ.get("GITHUB_WORKSPACE", os.getcwd())
COLS = ["Ticker","TF","Group","Score","Price","Entry_Price","Stop_Loss","Risk_%","Target","Reward_R","Pole_Low","Pole_Top",
        "Pole_Gain_%","Pole_Bars","Pennant_Bars","Pennant_Range_%_of_Pole","Upper_Line","Volume_Dry_Up","Breakout_Vol_x",
        "Pole_Start","Pole_Top_Date","Signal_Bar","Flags"]
df_out = pd.DataFrame(results) if results else pd.DataFrame(columns=COLS)
df_out = df_out[[c for c in COLS if c in df_out.columns]].reset_index(drop=True)
fpath = os.path.join(out_dir, f"bull_pennant_breakout_{ts}.csv")
try:
    from position_sizing import add_position_size
    df_out = add_position_size(df_out)
    try: __import__("best_picks").rank_best(df_out.drop_duplicates("Ticker").reset_index(drop=True))
    except Exception as _bp_err: print(f"  Best 10 skipped: {_bp_err}")
except Exception as _ps_err:
    print(f"  Position sizing skipped: {_ps_err}")
df_out.to_csv(fpath, index=False)
print(f"\n  CSV -> {fpath}")
tv = os.path.join(out_dir, f"tv_bull_pennant_breakout_{ts}.txt")
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

EMAIL_COLS = ["Ticker","TF","Score","Price","Stop_Loss","Risk_%","Target","Reward_R","Pole_Gain_%","Pole_Bars",
              "Pennant_Bars","Volume_Dry_Up","Breakout_Vol_x","Signal_Bar"]
def _table(rows):
    th = "".join(f'<th style="background:#1e293b;color:#e2e8f0;padding:6px 8px;font-size:11px;white-space:nowrap">{c}</th>' for c in EMAIL_COLS)
    body = ""
    for i, r in enumerate(rows):
        bg = "#fff" if i % 2 == 0 else "#f0f9ff"
        body += f'<tr style="background:{bg}">' + "".join(
            f'<td style="padding:5px 8px;font-size:12px;{"font-weight:700" if c in ("Ticker", "TF") else ""}">{r.get(c, "-")}</td>'
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
<h1 style="margin:0;color:#60a5fa;font-size:19px">Bull Pennant + Breakout Entry (Daily + Weekly)</h1>
<p style="margin:6px 0 0;color:#94a3b8;font-size:12px">{datetime.today().strftime('%Y-%m-%d %H:%M UTC')} &middot; {len(rl)} matches
&middot; TradingView: {TV_STATUS}</p></td></tr>
<tr><td style="padding:16px 22px">{secs}
<p style="font-size:11px;color:#64748b">Pole: 3-12 bars, +{CFG['pole_min_gain']['Daily']:g}% daily / +{CFG['pole_min_gain']['Weekly']:g}% weekly and &ge; {CFG['pole_min_atr']:g} ATR, mostly green.
Pennant: from the pole top, a converging range (falling upper line, lower line rising or falling less steeply, narrower at the end) holding the upper half of the pole.
Signal = the breakout candle: green, closing above the upper line and the last 3 pennant highs, the first close out of the pennant.
Entry = breakout close &middot; Stop = lowest low of the last {CFG['stop_last_bars']} pennant bars - {CFG['stop_buffer_atr']:g} ATR &middot; Target = entry + pole height.
Weekly uses completed Friday weeks only. Daily: pole {FUNNEL['Daily']['pole']} &rarr; pennant {FUNNEL['Daily']['pennant']} &rarr; breakout {FUNNEL['Daily']['breakout']};
weekly: pole {FUNNEL['Weekly']['pole']} &rarr; pennant {FUNNEL['Weekly']['pennant']} &rarr; breakout {FUNNEL['Weekly']['breakout']}.
Full results (with Quantity for a fixed max loss) attached.</p></td></tr>
<tr><td style="background:#f8fafc;padding:10px;text-align:center;color:#94a3b8;font-size:10px">Not financial advice &middot; Auto-generated by GitHub Actions</td></tr>
</table></td></tr></table></body></html>"""
        plain = [f"Bull Pennant Breakout - {datetime.today().strftime('%Y-%m-%d')}", f"{len(rl)} matches", f"TradingView: {TV_STATUS}", ""]
        for g, _ in GROUPS:
            plain.append(f"{g}: " + ", ".join(r["Ticker"] for r in rl if r["Group"] == g))
        nD = sum(1 for r in rl if r["TF"] == "Daily")
        subj = f"Bull Pennant Breakout - {nD} daily, {len(rl) - nD} weekly - {datetime.today().strftime('%Y-%m-%d')}"
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
