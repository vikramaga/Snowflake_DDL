# ============================================================
# NASDAQ — RSI Average Turns Up After Flat (Daily + Weekly) (v1)
# ============================================================
#
# Finds stocks whose RSI average (the 14-bar SMA of RSI(14), the
# TradingView "RSI-based MA") has been FLAT for a while and has just
# STARTED SLOPING UP. Checked on both the DAILY and WEEKLY timeframes;
# each match row says which timeframe fired.
#
# SIGNAL (all required; signal bar = one of the last
# signal_lookback_bars bars of that timeframe; most recent hit wins):
#
#   1. FLAT: over the flat_bars bars before the signal bar, every
#      one-bar change of the RSI average is within +/- flat_max_slope
#      points, the RSI average stayed inside a band no wider than
#      flat_max_range points, AND its net change across those bars is
#      within +/- flat_max_drift points (a slow steady climb is not flat).
#
#   2. TURN UP: on the signal bar the RSI average rises by at least
#      up_min_slope points versus the bar before. Because step 1 keeps
#      every earlier bar flat, the signal bar is the FIRST bar of the
#      upturn.
#
# DATA — one download per ticker (~4 years of daily bars); weekly bars
# are resampled from daily ('W'). A week still in progress is
# evaluated on its week-to-date candle and flagged Bar_Complete = No.
#
# OUTPUT: Entry_Price = signal bar's close; Stop_Loss = close minus
# stop_atr_mult x ATR(14) of that timeframe. The flat/turn thresholds
# and the stop are reasonable defaults, not explicitly requested.
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
    # history_days vs _clean(min_bars=400): 1500 calendar days -> ~1070
    # business days (checked with pd.bdate_range), well above 400. That
    # also gives ~200 weekly bars so the weekly RSI average has settled.
    "history_days"            : 1500,

    "timeframes"              : ["Daily", "Weekly"],

    # ── RSI and its average ──────────────────────────────────────
    "rsi_period"              : 14,
    "rsi_avg_period"          : 14,    # SMA of RSI (TradingView's RSI-based MA default)

    # ── Flat, then turn up (default definitions) ─────────────────
    "flat_bars"               : 8,     # bars the RSI average must be flat before the turn
    "flat_max_slope"          : 0.4,   # each one-bar change within +/- 0.4 RSI points
    "flat_max_range"          : 2.5,   # and the average stays inside a 2.5-point band
    "flat_max_drift"          : 1.0,   # and its net change across the flat bars is <= 1 point
                                       # (rules out a slow steady climb that never was flat)
    "up_min_slope"            : 0.5,   # signal bar: RSI average up >= 0.5 points

    # signal bar may be any of the last N bars (Daily: 3 days, Weekly: 2 weeks)
    "signal_lookback_bars"    : {"Daily": 3, "Weekly": 2},

    # ── Entry / stop (defaults) ─────────────────────────────────
    "atr_period"              : 14,
    "stop_atr_mult"           : 1.5,

    # ── Filters ─────────────────────────────────────────────────
    "min_avg_volume"          : 200_000,
    "min_price"               : 5.0,

    "batch_size"              : 50,
    "batch_sleep"             : 1.5,
}

# ── Indicators ───────────────────────────────────────────────
def resample_ohlcv(df, rule):
    """Resample daily OHLCV to a coarser timeframe ('W' weekly)."""
    agg = {"Open": "first", "High": "max", "Low": "min",
           "Close": "last", "Volume": "sum"}
    out = df.resample(rule).agg(agg)
    out.dropna(subset=["Close"], inplace=True)
    return out

def calc_rsi(close, period=14):
    """Standard Wilder-smoothed RSI."""
    delta = close.diff()
    gain  = delta.clip(lower=0)
    loss  = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1/period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1/period, min_periods=period, adjust=False).mean()
    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))

def calc_atr(df, period):
    h, l, c = df["High"], df["Low"], df["Close"]
    pc = c.shift(1)
    tr = pd.concat([h - l, (h - pc).abs(), (l - pc).abs()], axis=1).max(axis=1)
    return tr.rolling(period).mean()

def check_pattern_at(rsi_avg, i, cfg):
    """
    SINGLE SOURCE OF TRUTH. Checks signal bar i of one timeframe.
    Computes EVERY step (no short-circuit) so the funnel shows where
    tickers drop out. Returns (passed, details); details is {} only if
    there is not enough data.
    """
    fb = cfg["flat_bars"]
    if i < fb + 1 or i >= len(rsi_avg):
        return False, {}
    win = rsi_avg.iloc[i - fb - 1:i].values.astype(float)   # flat window incl. the bar before it
    now, prev = float(rsi_avg.iloc[i]), float(rsi_avg.iloc[i - 1])
    if not (np.all(np.isfinite(win)) and np.isfinite(now)):
        return False, {}

    slopes = np.diff(win)                                    # fb one-bar changes before bar i
    flat_slope = float(np.max(np.abs(slopes)))
    flat_range = float(win[1:].max() - win[1:].min())
    flat_drift = float(win[-1] - win[0])                     # net change across the flat window
    step1 = bool(flat_slope <= cfg["flat_max_slope"] and flat_range <= cfg["flat_max_range"]
                 and abs(flat_drift) <= cfg["flat_max_drift"])

    slope_now = now - prev
    step2 = bool(slope_now >= cfg["up_min_slope"])

    passed = step1 and step2
    return passed, {
        "rsi_avg": now, "rsi_avg_prev": prev, "slope_now": slope_now,
        "flat_slope": flat_slope, "flat_range": flat_range, "flat_drift": flat_drift,
        "flat_level": float(win[1:].mean()),
        "step1": step1, "step2": step2,
    }

# ── Diagnostic funnel per timeframe (per ticker: did ANY of the
#    lookback bars pass steps 1..k cumulatively) ──────────────────────
FUNNEL_COUNTS = {
    "tickers_checked": 0,
    "Daily_checked": 0,  "Daily_step1_flat": 0,  "Daily_step2_turn_up": 0,
    "Weekly_checked": 0, "Weekly_step1_flat": 0, "Weekly_step2_turn_up": 0,
}

def find_signals(rsi_avg, tf, cfg):
    hits, b1, b2, checked = [], False, False, False
    n = len(rsi_avg)
    for i in range(max(0, n - cfg["signal_lookback_bars"][tf]), n):
        ok, d = check_pattern_at(rsi_avg, i, cfg)
        if not d: continue
        checked = True
        b1 |= d["step1"]; b2 |= (d["step1"] and d["step2"])
        if ok:
            d["i"] = i
            hits.append(d)
    if checked:
        FUNNEL_COUNTS[f"{tf}_checked"] += 1
        FUNNEL_COUNTS[f"{tf}_step1_flat"] += b1
        FUNNEL_COUNTS[f"{tf}_step2_turn_up"] += b2
    return hits

def analyze_rsi_avg_turn_up(sym, df_daily):
    """Returns a list of result dicts (one per timeframe that fired)."""
    if df_daily is None: return []
    price   = float(df_daily["Close"].iloc[-1])
    avg_vol = float(df_daily["Volume"].tail(20).mean())
    if price   < CFG["min_price"]:      return []
    if avg_vol < CFG["min_avg_volume"]: return []

    FUNNEL_COUNTS["tickers_checked"] += 1
    out = []
    for tf in CFG["timeframes"]:
        df = df_daily if tf == "Daily" else resample_ohlcv(df_daily, "W")
        if len(df) < CFG["rsi_period"] + CFG["rsi_avg_period"] + CFG["flat_bars"] + 20:
            continue
        rsi = calc_rsi(df["Close"], CFG["rsi_period"])
        rsi_avg = rsi.rolling(CFG["rsi_avg_period"]).mean()
        atr = calc_atr(df, CFG["atr_period"])

        hits = find_signals(rsi_avg, tf, CFG)
        if not hits: continue
        sig = hits[-1]   # most recent hit wins
        i = sig["i"]
        entry = float(df["Close"].iloc[i])
        a = float(atr.iloc[i])
        if not np.isfinite(a) or a <= 0: continue
        stop = entry - CFG["stop_atr_mult"] * a
        if stop <= 0: continue
        risk_pct = (entry - stop) / entry * 100
        bars_ago = len(df) - 1 - i
        bar_date = df.index[i]
        if tf == "Weekly":
            n_days = int(((df_daily.index > bar_date - pd.Timedelta(days=7)) & (df_daily.index <= bar_date)).sum())
            complete = bars_ago > 0 or n_days >= 5
        else:
            complete = True
        r_now = float(rsi.iloc[i])

        score = 30
        score += min(30, (sig["slope_now"] - CFG["up_min_slope"]) * 15 + 10)     # stronger turn
        score += max(0, 20 - sig["flat_range"] * 6)                              # tighter flat base
        score += max(0, 10 - abs(sig["flat_level"] - 50) * 0.5)                   # base near the midline
        score += 10 if r_now > sig["rsi_avg"] else 0                             # RSI above its average
        score = round(min(100, max(0, score)))

        flags = [f"Flat {CFG['flat_bars']} bars in {sig['flat_range']:.1f}-pt band",
                 f"Turn +{sig['slope_now']:.2f}/bar",
                 f"RSI {r_now:.1f} {'>' if r_now > sig['rsi_avg'] else '<'} avg {sig['rsi_avg']:.1f}"]
        if not complete:
            flags.append("Week in progress")

        out.append({
            "Timeframe": tf, "Score": score, "Price": round(price, 2),
            "Entry_Price": round(entry, 2), "Stop_Loss": round(stop, 2), "Risk_%": round(risk_pct, 1),
            "RSI": round(r_now, 1), "RSI_Avg": round(sig["rsi_avg"], 1),
            "RSI_Avg_Slope": round(sig["slope_now"], 2),
            "Flat_Level": round(sig["flat_level"], 1), "Flat_Range": round(sig["flat_range"], 2),
            "Flat_Max_Slope": round(sig["flat_slope"], 2),
            "Bar_Complete": "Yes" if complete else "No",
            "Signal_Date": bar_date.strftime("%Y-%m-%d"), "Bars_Since_Signal": int(bars_ago),
            "Recent_Signal_Count": len(hits),
            "Flags": " | ".join(flags),
            "_df": df, "_rsi": rsi, "_rsi_avg": rsi_avg, "_i": i,
        })
    return out

# ── Download: daily (→ weekly via resample) ──────────────────────
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
                        df = _clean(df, min_bars=400)
                        if df is not None: out[sym] = df
                    except Exception: pass
            elif len(symbols) == 1:
                df = _clean(raw, min_bars=400)
                if df is not None: out[symbols[0]] = df
    except Exception: pass
    for sym in [s for s in symbols if s not in out]:
        for _ in range(2):
            try:
                df = yf.Ticker(sym).history(
                    start=start.strftime("%Y-%m-%d"),
                    end=end.strftime("%Y-%m-%d"),
                    auto_adjust=True, actions=False)
                df = _clean(df, min_bars=400)
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
        for r in analyze_rsi_avg_turn_up(sym, daily_map[sym]):
            r["Ticker"] = sym
            results.append(r)
            print(f"  ✅ {sym:<6} [{r['Timeframe']}] ${r['Price']:.2f}  Score {r['Score']}  "
                  f"RSI avg {r['RSI_Avg']} (+{r['RSI_Avg_Slope']}/bar)  {r['Signal_Date']}")
    except Exception: pass

fc = FUNNEL_COUNTS
print(f"\n{'━'*65}\n  SCAN COMPLETE — {len(results)} matches from {got_daily} tickers with data")
print(f"  🔍 FUNNEL (liquid tickers checked: {fc['tickers_checked']})")
for _tf in CFG["timeframes"]:
    print(f"  {_tf:<7} checked {fc[_tf + '_checked']:>5}  →  step 1 RSI avg flat {fc[_tf + '_step1_flat']:>5}"
          f"  →  step 2 + turned up {fc[_tf + '_step2_turn_up']:>4}  (= full pattern)")
print(f"{'━'*65}")

results.sort(key=lambda x: (x["Timeframe"] != "Weekly", -x["Score"]))   # weekly first, then by score

# ── Save CSV + TradingView list ───────────────────────────────
ts      = datetime.today().strftime("%Y%m%d_%H%M")
out_dir = os.environ.get("GITHUB_WORKSPACE", os.getcwd())
COLS = ["Ticker","Timeframe","Price","Score","Entry_Price","Stop_Loss","Risk_%",
        "RSI","RSI_Avg","RSI_Avg_Slope","Flat_Level","Flat_Range","Flat_Max_Slope",
        "Bar_Complete","Signal_Date","Bars_Since_Signal","Recent_Signal_Count","Flags"]
df_out = pd.DataFrame([{k: v for k, v in r.items() if not k.startswith("_")}
                       for r in results]) if results else pd.DataFrame(columns=COLS)
df_out = df_out[[c for c in COLS if c in df_out.columns]].reset_index(drop=True)

fpath = os.path.join(out_dir, f"rsi_avg_turn_up_{ts}.csv")
try:
    from position_sizing import add_position_size
    df_out = add_position_size(df_out)
except Exception as _ps_err:
    print(f"  ⚠️  Position sizing skipped: {_ps_err}")
df_out.to_csv(fpath, index=False)
print(f"\n  💾 CSV → {fpath}")
tv = os.path.join(out_dir, f"tv_rsi_avg_turn_up_{ts}.txt")
with open(tv, "w") as f:
    f.write(f"###RSI Avg Turn Up {datetime.today().strftime('%Y-%m-%d')}\n")
    for t in dict.fromkeys(r["Ticker"] for r in results): f.write(f"NASDAQ:{t}\n")
print(f"  📋 TradingView → {tv}")
if results:
    print(f"\n  📋 Tickers: {', '.join(dict.fromkeys(r['Ticker'] for r in results))}")

# ── Email with CSV attached ───────────────────────────────────
EMAIL_COLS = ["Ticker","Timeframe","Price","Score","Entry_Price","Stop_Loss","Risk_%",
              "RSI","RSI_Avg","RSI_Avg_Slope","Flat_Level","Signal_Date"]

def _fmt_cell(col, v):
    try:
        if col in ("Price","Entry_Price","Stop_Loss"): return f"${float(v):.2f}"
        if col == "RSI_Avg_Slope": return f"+{float(v):.2f}/bar"
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
                     f'{", ".join(dict.fromkeys(r["Ticker"] for r in rl))}</p>') if rl else ""
        html = f"""<!DOCTYPE html><html><body style="margin:0;background:#f1f5f9;font-family:'Segoe UI',Arial,sans-serif">
<table width="100%" cellpadding="0" cellspacing="0"><tr><td align="center" style="padding:20px 10px">
<table width="100%" style="max-width:860px;background:#fff;border-radius:12px;overflow:hidden">
<tr><td style="background:linear-gradient(135deg,#0f172a,#1e3a5f);padding:20px 26px">
<h1 style="margin:0;color:#60a5fa;font-size:19px">📊 RSI Average Turns Up After Flat (Daily + Weekly)</h1>
<p style="margin:6px 0 0;color:#94a3b8;font-size:12px">{datetime.today().strftime('%Y-%m-%d %H:%M UTC')} · {cnt} match{'es' if cnt != 1 else ''}
 — RSI average (SMA{CFG['rsi_avg_period']} of RSI{CFG['rsi_period']}) flat for {CFG['flat_bars']} bars, now rising ≥ {CFG['up_min_slope']} pts/bar</p>
</td></tr>
<tr><td style="padding:12px 26px;background:#0b1220"><p style="margin:0;color:#cbd5e1;font-size:12px">
🔍 FUNNEL: {fc['tickers_checked']} liquid tickers.
Daily: {fc['Daily_step1_flat']} flat → <b style="color:#facc15">{fc['Daily_step2_turn_up']} turned up</b> ·
Weekly: {fc['Weekly_step1_flat']} flat → <b style="color:#facc15">{fc['Weekly_step2_turn_up']} turned up</b></p></td></tr>
<tr><td style="padding:16px"><div style="overflow-x:auto;border:1px solid #e2e8f0;border-radius:8px">
<table style="border-collapse:collapse;width:100%"><thead><tr>{th}</tr></thead><tbody>{rows}</tbody></table></div>
{tick_html}
<p style="font-size:11px;color:#64748b;margin:10px 0 0">Entry = the signal bar's close · Stop = close − {CFG['stop_atr_mult']} × ATR({CFG['atr_period']})
of that timeframe (defaults). Bar_Complete = No means the week is still in progress — confirm at the Friday close. 📎 Full results (with Quantity for a fixed max loss) attached as CSV.</p>
</td></tr>
<tr><td style="background:#f8fafc;padding:10px;text-align:center;color:#94a3b8;font-size:10px">
⚠️ Not financial advice · Auto-generated by GitHub Actions</td></tr>
</table></td></tr></table></body></html>"""

        plain = [f"RSI Average Turns Up After Flat (Daily + Weekly) — {datetime.today().strftime('%Y-%m-%d')}",
                 f"{cnt} matches",
                 f"FUNNEL: {fc['tickers_checked']} liquid tickers | Daily: {fc['Daily_step1_flat']} flat -> "
                 f"{fc['Daily_step2_turn_up']} turned up | Weekly: {fc['Weekly_step1_flat']} flat -> "
                 f"{fc['Weekly_step2_turn_up']} turned up", "="*60]
        for r in rl[:50]:
            plain.append(f"{r['Ticker']:<6} [{r['Timeframe']}] ${r['Price']:.2f}  Score:{r['Score']}  Entry:${r['Entry_Price']:.2f}  "
                         f"SL:${r['Stop_Loss']:.2f}  RSI:{r['RSI']}  RSI avg:{r['RSI_Avg']} (+{r['RSI_Avg_Slope']}/bar, "
                         f"flat near {r['Flat_Level']})  {r['Signal_Date']}"
                         f"{'' if r['Bar_Complete'] == 'Yes' else ' (week in progress)'}")
        if rl:
            plain += ["", "Tickers: " + ", ".join(dict.fromkeys(r["Ticker"] for r in rl))]
        else:
            plain.append("No matches today")
        plain.append("\nFull results in CSV attachment.")

        subj = (f"📊 RSI Avg Turn Up (D+W) — "
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

# ── Charts for top 5: price (last 60 bars) + RSI with its average, flat window shaded ─
if results:
    top = sorted(results, key=lambda x: -x["Score"])[:5]
    fig, axes = plt.subplots(len(top), 2, figsize=(16, 4.2*len(top)), facecolor="#0f172a",
                             gridspec_kw={"width_ratios": [1.4, 1.6]})
    if len(top) == 1: axes = axes.reshape(1, 2)
    for row, r in zip(axes, top):
        ax_p, ax_r = row
        df, i = r["_df"], r["_i"]
        lo = max(0, i - 59); seg = df.iloc[lo:i + 1]
        x = np.arange(len(seg))
        ax_p.set_facecolor("#0f172a")
        ax_p.plot(x, seg["Close"].values, color="#60a5fa", lw=1.3)
        ax_p.set_title(f"{r['Ticker']} [{r['Timeframe']}] | ${r['Price']:.2f} | Score {r['Score']}",
                       color="#e2e8f0", fontsize=9, fontweight="bold")
        rs = r["_rsi"].iloc[lo:i + 1].values; ra = r["_rsi_avg"].iloc[lo:i + 1].values
        ax_r.set_facecolor("#0f172a")
        ax_r.plot(x, rs, color="#60a5fa", lw=1.0, alpha=0.7, label="RSI(14)")
        ax_r.plot(x, ra, color="#facc15", lw=1.8, label=f"RSI avg (SMA{CFG['rsi_avg_period']})")
        ax_r.axvspan(len(seg) - 1 - CFG["flat_bars"], len(seg) - 1, color="#94a3b8", alpha=0.12)
        ax_r.axvline(len(seg) - 1, color="#22c55e", ls="--", lw=0.9)
        ax_r.axhline(50, color="#475569", lw=0.6)
        ax_r.legend(loc="upper left", facecolor="#1e293b", labelcolor="#e2e8f0", fontsize=6)
        ax_r.set_title(f"RSI avg {r['RSI_Avg']} (+{r['RSI_Avg_Slope']}/bar after flat near {r['Flat_Level']})",
                       color="#e2e8f0", fontsize=8)
        step = max(1, len(seg) // 6)
        for ax in (ax_p, ax_r):
            ax.set_xticks(x[::step]); ax.set_xticklabels([d.strftime("%m/%d") for d in seg.index[::step]], fontsize=7)
            ax.tick_params(colors="#94a3b8", labelsize=7)
            for sp in ax.spines.values(): sp.set_edgecolor("#1e3a5f")
            ax.grid(color="#1e3a5f", ls="--", lw=0.5, alpha=0.6)
    plt.suptitle(f"RSI Average Turns Up After Flat · "
                 f"{datetime.today().strftime('%Y-%m-%d')}", color="#60a5fa", fontsize=12, fontweight="bold")
    plt.tight_layout()
    cp = os.path.join(out_dir, f"rsi_avg_turn_up_chart_{ts}.png")
    plt.savefig(cp, dpi=130, bbox_inches="tight", facecolor="#0f172a")
    plt.close()
    print(f"  📊 Chart → {cp}")

print(f"""
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  📋 SIGNAL (both required; checked on Daily and Weekly separately)
  1) FLAT: for {CFG['flat_bars']} bars, the RSI average (SMA{CFG['rsi_avg_period']} of RSI{CFG['rsi_period']}) moved
     at most {CFG['flat_max_slope']} pts per bar, stayed in a {CFG['flat_max_range']}-pt band and drifted
     no more than {CFG['flat_max_drift']} pt net
  2) TURN UP: on the signal bar the RSI average rose >= {CFG['up_min_slope']} pts (first rising bar)
  Signal bar = one of the last {CFG['signal_lookback_bars']['Daily']} daily / {CFG['signal_lookback_bars']['Weekly']} weekly bars
  OUTPUT defaults: Entry = signal bar close, Stop = close - {CFG['stop_atr_mult']} x ATR({CFG['atr_period']})
  ⚙️  Too few results: flat_max_slope 0.4 → 0.6, flat_max_drift 1 → 2, flat_bars 8 → 6
     Too many: up_min_slope 0.5 → 0.8, flat_bars 8 → 10
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
""")
