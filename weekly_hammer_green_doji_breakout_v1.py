# ============================================================
# NASDAQ / NYSE - Weekly Hammer/Doji -> Green -> Inside Doji -> Daily Breakout (v1)
# ============================================================
#
# Weeks run Monday-Friday. "Current week" = the week of the latest daily bar
# (on a weekend that is the week just finished). W-1, W-2, W-3 = the three
# completed weeks before it.
#
#   1. W-3: RED candle that is a HAMMER or a DOJI
#   2. W-2: GREEN candle whose low is above W-3's low, with MORE volume than W-3
#   3. W-1: a DOJI or a SMALL HAMMER whose low stays above the middle of W-2's
#      green body ((open + close) / 2)
#   4. Current week: a DAILY candle closes above W-1's high (the first such
#      close is the signal day)
#
# Candle definitions (weekly bars, ATR = 10-week average true range):
#   doji   = body <= 35% of the range and <= 0.5 ATR
#   hammer = lower wick >= 2x body and upper wick <= 25% of the range
#   small  = range <= 1.0 ATR
#
# Entry = the signal day's close; Stop = W-1's low; Target 1 = entry + 2R,
# Target 2 = entry + 3R. "Half of the green candle" read as its body midpoint.
# Stop/targets and candle thresholds are reasonable defaults, not explicitly
# requested; all are in CFG.
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
    # history_days vs bars: 560 calendar days -> ~386 business days (pd.bdate_range);
    # the pattern needs 4 weeks + a 10-week ATR, the context SMA200 needs 200 days.
    "history_days"            : 560,
    "min_bars"                : 210,

    "atr_weeks"               : 10,
    "doji_body_pct"           : 35.0,  # body <= 35% of the range ...
    "doji_body_atr"           : 0.5,   # ... and <= 0.5 weekly ATR
    "hammer_wick_x_body"      : 2.0,   # lower wick >= 2x body
    "hammer_upper_wick_pct"   : 25.0,  # upper wick <= 25% of the range
    "small_range_atr"         : 1.0,   # small = range <= 1 weekly ATR
    "w1_low_vs_w2_body"       : 0.5,   # W-1 low above W-2 body's midpoint

    "target1_r"               : 2.0,
    "target2_r"               : 3.0,

    "min_avg_volume"          : 200_000,
    "min_price"               : 5.0,

    "batch_size"              : 50,
    "batch_sleep"             : 1.5,
}

# -- Indicators / candle shapes
def weekly_bars(df):
    """Mon-Fri weeks labelled by Friday; the last row is the current (maybe partial) week."""
    agg = {"Open": "first", "High": "max", "Low": "min", "Close": "last", "Volume": "sum"}
    w = df.resample("W-FRI").agg(agg).dropna(subset=["Open", "Close"])
    pc = w["Close"].shift(1)
    tr = pd.concat([w["High"] - w["Low"], (w["High"] - pc).abs(), (w["Low"] - pc).abs()], axis=1).max(axis=1)
    w["ATR"] = tr.rolling(CFG["atr_weeks"]).mean()
    return w

def shape(o, h, l, c, atr, cfg):
    rng = h - l
    body = abs(c - o)
    lower = min(o, c) - l
    upper = h - max(o, c)
    doji = rng > 0 and body <= rng * cfg["doji_body_pct"] / 100 and body <= cfg["doji_body_atr"] * atr
    hammer = rng > 0 and lower >= cfg["hammer_wick_x_body"] * max(body, rng * 0.02) \
        and upper <= rng * cfg["hammer_upper_wick_pct"] / 100
    small = rng <= cfg["small_range_atr"] * atr
    return {"doji": bool(doji), "hammer": bool(hammer), "small": bool(small),
            "range": rng, "body": body, "lower": lower, "upper": upper}

def check_pattern_at(df, wk, cfg):
    """
    SINGLE SOURCE OF TRUTH. Checks the latest week of `wk` (the current week)
    against the three weeks before it, and the daily bars of the current week.
    Computes EVERY step (no short-circuit) so the funnel shows where tickers
    drop out. Returns (passed, details); details is {} if data is too short.
    """
    if len(wk) < cfg["atr_weeks"] + 5:
        return False, {}
    w0, w1, w2, w3 = (wk.iloc[-k] for k in (1, 2, 3, 4))
    if not all(np.isfinite(x) for x in (w1.ATR, w2.ATR, w3.ATR)):
        return False, {}
    s3 = shape(w3.Open, w3.High, w3.Low, w3.Close, w3.ATR, cfg)
    s1 = shape(w1.Open, w1.High, w1.Low, w1.Close, w1.ATR, cfg)

    step1 = bool(w3.Close < w3.Open and (s3["doji"] or s3["hammer"]))
    step2 = bool(w2.Close > w2.Open and w2.Low > w3.Low and w2.Volume > w3.Volume)
    mid2 = w2.Open + (w2.Close - w2.Open) * cfg["w1_low_vs_w2_body"]
    step3 = bool((s1["doji"] or (s1["hammer"] and s1["small"])) and w1.Low > mid2)

    week_start = wk.index[-2] + pd.Timedelta(days=1)          # day after W-1's Friday
    days = df[df.index >= week_start]
    above = days[days["Close"] > w1.High]
    step4 = bool(len(above) > 0)
    sig_day = above.index[0] if step4 else None

    passed = step1 and step2 and step3 and step4
    return passed, {
        "w3": w3, "w2": w2, "w1": w1, "w0": w0, "s3": s3, "s1": s1, "mid2": float(mid2),
        "w3_type": "doji" if s3["doji"] else ("hammer" if s3["hammer"] else "-"),
        "w1_type": "doji" if s1["doji"] else ("small hammer" if s1["hammer"] and s1["small"] else "-"),
        "sig_day": sig_day, "sig_close": float(above["Close"].iloc[0]) if step4 else np.nan,
        "step1": step1, "step2": step2, "step3": step3, "step4": step4,
    }

# -- Diagnostic funnel (per ticker, cumulative)
FUNNEL_COUNTS = {
    "tickers_checked": 0,
    "passed_step1_w3_red_hammer_doji": 0,
    "passed_step2_w2_green_higher_low_volume": 0,
    "passed_step3_w1_doji_small_hammer": 0,
    "passed_step4_daily_close_above_w1_high": 0,
}
_FUNNEL_KEYS = list(FUNNEL_COUNTS)[1:]

def analyze_weekly_hdgd(sym, df_daily):
    if df_daily is None: return None
    price   = float(df_daily["Close"].iloc[-1])
    avg_vol = float(df_daily["Volume"].tail(20).mean())
    if price   < CFG["min_price"]:      return None
    if avg_vol < CFG["min_avg_volume"]: return None
    if len(df_daily) < CFG["min_bars"]: return None

    wk = weekly_bars(df_daily)
    ok, d = check_pattern_at(df_daily, wk, CFG)
    if not d: return None
    FUNNEL_COUNTS["tickers_checked"] += 1
    cum = True
    for k, key in enumerate(_FUNNEL_KEYS):
        cum = cum and d[f"step{k+1}"]
        FUNNEL_COUNTS[key] += cum
    if not ok: return None

    w3, w2, w1 = d["w3"], d["w2"], d["w1"]
    entry = d["sig_close"]
    stop = float(w1.Low)
    if entry <= stop: return None
    risk = entry - stop
    risk_pct = risk / entry * 100
    c = df_daily["Close"]
    s50 = float(c.rolling(50).mean().iloc[-1]); s200 = float(c.rolling(200).mean().iloc[-1])
    vol_ratio = float(w2.Volume / w3.Volume) if w3.Volume > 0 else np.nan
    w1_tight = float(d["s1"]["range"] / w1.ATR) if w1.ATR > 0 else np.nan
    brk = (entry / w1.High - 1) * 100
    after = df_daily[df_daily.index > d["sig_day"]]
    flags = [f"W-3 red {d['w3_type']}", f"W-2 vol {vol_ratio:.1f}x W-3", f"W-1 {d['w1_type']} ({w1_tight:.2f} ATR)"]
    flags.append("above SMA50 & SMA200" if price > s50 and price > s200 else
                 ("above SMA200" if price > s200 else "below SMA200"))
    if len(after) and after["Low"].min() <= stop: flags.append("Stop already hit")
    if price < w1.High: flags.append("Now back below W-1 high")

    score = 30
    score += min(15, max(0, (vol_ratio - 1) * 15))                 # stronger W-2 volume
    score += max(0, 15 - w1_tight * 10)                            # tighter W-1
    score += 10 if price > s50 else 0
    score += 10 if price > s200 else 0
    score += max(0, 10 - max(0, risk_pct - 4) * 2)                 # tighter stop
    score += min(10, brk * 3)                                      # clean breakout
    score = round(min(100, max(0, score)))

    return {
        "Score": score, "Price": round(price, 2),
        "Entry_Price": round(entry, 2), "Stop_Loss": round(stop, 2), "Risk_%": round(risk_pct, 1),
        "Target_1_2R": round(entry + CFG["target1_r"] * risk, 2), "Target_2_3R": round(entry + CFG["target2_r"] * risk, 2),
        "W3_Week": (wk.index[-4] - pd.Timedelta(days=4)).strftime("%Y-%m-%d"), "W3_Type": d["w3_type"],
        "W3_Low": round(float(w3.Low), 2),
        "W2_Low": round(float(w2.Low), 2), "W2_Body_Mid": round(d["mid2"], 2), "W2_Vol_x_W3": round(vol_ratio, 2),
        "W1_Type": d["w1_type"], "W1_High": round(float(w1.High), 2), "W1_Low": round(float(w1.Low), 2),
        "W1_Range_ATR": round(w1_tight, 2), "Breakout_%": round(brk, 2),
        "Above_SMA50": "Yes" if price > s50 else "No", "Above_SMA200": "Yes" if price > s200 else "No",
        "Signal_Date": d["sig_day"].strftime("%Y-%m-%d"),
        "Days_Since_Signal": int(len(after)),
        "Flags": " | ".join(flags),
        "_df": df_daily, "_wk": wk,
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
        r = analyze_weekly_hdgd(sym, daily_map[sym])
        if r is None: continue
        r["Ticker"] = sym
        results.append(r)
        print(f"  ✅ {sym:<6} ${r['Price']:.2f}  Score {r['Score']}  W-3 {r['W3_Type']}  W-2 vol {r['W2_Vol_x_W3']}x  "
              f"W-1 {r['W1_Type']}  close > W-1 high {r['W1_High']} on {r['Signal_Date']}")
    except Exception: pass

fc = FUNNEL_COUNTS
print(f"\n{'=' * 65}\n  SCAN COMPLETE - {len(results)} matches from {got_daily} tickers with data")
print("  FUNNEL (current week vs the 3 weeks before)")
print(f"  Tickers checked                                 : {fc['tickers_checked']}")
print(f"  Step 1 - W-3 red hammer / doji                   : {fc['passed_step1_w3_red_hammer_doji']}")
print(f"  Step 2 - + W-2 green, higher low, more volume    : {fc['passed_step2_w2_green_higher_low_volume']}")
print(f"  Step 3 - + W-1 doji / small hammer above W-2 mid : {fc['passed_step3_w1_doji_small_hammer']}")
print(f"  Step 4 - + daily close above W-1 high this week  : {fc['passed_step4_daily_close_above_w1_high']}  (= full pattern)")
print("=" * 65)

results.sort(key=lambda x: x["Score"], reverse=True)

# -- Save CSV + TradingView list
ts      = datetime.today().strftime("%Y%m%d_%H%M")
out_dir = os.environ.get("GITHUB_WORKSPACE", os.getcwd())
COLS = ["Ticker","Price","Score","Entry_Price","Stop_Loss","Risk_%","Target_1_2R","Target_2_3R",
        "W3_Week","W3_Type","W3_Low","W2_Low","W2_Body_Mid","W2_Vol_x_W3",
        "W1_Type","W1_High","W1_Low","W1_Range_ATR","Breakout_%","Above_SMA50","Above_SMA200",
        "Signal_Date","Days_Since_Signal","Flags"]
df_out = pd.DataFrame([{k: v for k, v in r.items() if not k.startswith("_")}
                       for r in results]) if results else pd.DataFrame(columns=COLS)
df_out = df_out[[c for c in COLS if c in df_out.columns]].reset_index(drop=True)

fpath = os.path.join(out_dir, f"weekly_hammer_green_doji_breakout_{ts}.csv")
try:
    from position_sizing import add_position_size
    df_out = add_position_size(df_out)
    try: __import__("best_picks").rank_best(df_out)
    except Exception as _bp_err: print(f"  ⚠️  Best 10 skipped: {_bp_err}")
except Exception as _ps_err:
    print(f"  ⚠️  Position sizing skipped: {_ps_err}")
df_out.to_csv(fpath, index=False)
print(f"\n  CSV -> {fpath}")
tv = os.path.join(out_dir, f"tv_weekly_hammer_green_doji_breakout_{ts}.txt")
with open(tv, "w") as f:
    f.write(f"###Weekly Hammer-Green-Doji Breakout {datetime.today().strftime('%Y-%m-%d')}\n")
    for r in results: f.write(f"NASDAQ:{r['Ticker']}\n")
print(f"  TradingView -> {tv}")
if results:
    print(f"\n  Tickers: {', '.join(r['Ticker'] for r in results)}")

# -- Email with CSV attached
EMAIL_COLS = ["Ticker","Price","Score","Entry_Price","Stop_Loss","Risk_%","Target_1_2R",
              "W3_Type","W2_Vol_x_W3","W1_Type","W1_High","Above_SMA200","Signal_Date"]

def _fmt_cell(col, v):
    try:
        if col in ("Price","Entry_Price","Stop_Loss","Target_1_2R","W1_High"): return f"${float(v):.2f}"
        if col == "Risk_%": return f"{float(v):.1f}%"
        if col == "W2_Vol_x_W3": return f"{float(v):.1f}x"
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
                if c == "Target_1_2R": sty = "color:#2563eb;font-weight:600"
                tds += f'<td style="padding:6px 10px;font-size:12px;{sty}">{_fmt_cell(c, r.get(c))}</td>'
            rows += f'<tr style="background:{bg}">{tds}</tr>'
        if not rows:
            rows = (f'<tr><td colspan="{len(EMAIL_COLS)}" style="padding:20px;text-align:center;'
                    f'color:#94a3b8">No matches this week</td></tr>')
        tick_html = (f'<p style="margin:10px 0 0;font-size:12px;font-family:monospace">'
                     f'{", ".join(r["Ticker"] for r in rl)}</p>') if rl else ""
        html = f"""<!DOCTYPE html><html><body style="margin:0;background:#f1f5f9;font-family:'Segoe UI',Arial,sans-serif">
<table width="100%" cellpadding="0" cellspacing="0"><tr><td align="center" style="padding:20px 10px">
<table width="100%" style="max-width:880px;background:#fff;border-radius:12px;overflow:hidden">
<tr><td style="background:linear-gradient(135deg,#0f172a,#1e3a5f);padding:20px 26px">
<h1 style="margin:0;color:#60a5fa;font-size:19px">\U0001F4CA Weekly Hammer/Doji → Green → Doji → Daily Breakout</h1>
<p style="margin:6px 0 0;color:#94a3b8;font-size:12px">{datetime.today().strftime('%Y-%m-%d %H:%M UTC')} · {cnt} match{'es' if cnt != 1 else ''}
 — W-3 red hammer or doji, W-2 green with a higher low and more volume, W-1 doji or small hammer holding above the middle of W-2's body,
 and a daily close above W-1's high this week</p>
</td></tr>
<tr><td style="padding:12px 26px;background:#0b1220"><p style="margin:0;color:#cbd5e1;font-size:12px">
\U0001F50D FUNNEL: {fc['tickers_checked']} checked → {fc['passed_step1_w3_red_hammer_doji']} W-3 red hammer/doji
→ {fc['passed_step2_w2_green_higher_low_volume']} + W-2 green, higher low, more volume
→ {fc['passed_step3_w1_doji_small_hammer']} + W-1 doji / small hammer
→ <b style="color:#facc15">{fc['passed_step4_daily_close_above_w1_high']} + daily close above W-1 high = full match</b></p></td></tr>
<tr><td style="padding:16px"><div style="overflow-x:auto;border:1px solid #e2e8f0;border-radius:8px">
<table style="border-collapse:collapse;width:100%"><thead><tr>{th}</tr></thead><tbody>{rows}</tbody></table></div>
{tick_html}
<p style="font-size:11px;color:#64748b;margin:10px 0 0">Entry = the close of the first day this week that closed above W-1's high ·
Stop = W-1's low · Target 1 = entry + {CFG['target1_r']:g}R, Target 2 = entry + {CFG['target2_r']:g}R (in the CSV).
Doji = body ≤ {CFG['doji_body_pct']:.0f}% of the range and ≤ {CFG['doji_body_atr']} weekly ATR; hammer = lower wick ≥ {CFG['hammer_wick_x_body']:g}x body,
upper wick ≤ {CFG['hammer_upper_wick_pct']:.0f}% of range; small = range ≤ {CFG['small_range_atr']:g} weekly ATR.
\U0001F4CE Full results (with Quantity for a fixed max loss) attached as CSV.</p>
</td></tr>
<tr><td style="background:#f8fafc;padding:10px;text-align:center;color:#94a3b8;font-size:10px">
⚠️ Not financial advice · Auto-generated by GitHub Actions</td></tr>
</table></td></tr></table></body></html>"""

        plain = [f"Weekly Hammer/Doji -> Green -> Doji -> Daily Breakout — {datetime.today().strftime('%Y-%m-%d')}",
                 f"{cnt} matches",
                 f"FUNNEL: {fc['tickers_checked']} checked -> {fc['passed_step1_w3_red_hammer_doji']} W-3 red hammer/doji -> "
                 f"{fc['passed_step2_w2_green_higher_low_volume']} W-2 green -> {fc['passed_step3_w1_doji_small_hammer']} W-1 doji -> "
                 f"{fc['passed_step4_daily_close_above_w1_high']} breakout (=full match)", "=" * 60]
        for r in rl[:50]:
            plain.append(f"{r['Ticker']:<6} ${r['Price']:.2f}  Score:{r['Score']}  Entry:${r['Entry_Price']:.2f}  "
                         f"SL:${r['Stop_Loss']:.2f} ({r['Risk_%']}%)  T1:${r['Target_1_2R']:.2f}  "
                         f"W-3 {r['W3_Type']}  W-2 vol {r['W2_Vol_x_W3']}x  W-1 {r['W1_Type']}  {r['Signal_Date']}")
        if rl:
            plain += ["", "Tickers: " + ", ".join(r["Ticker"] for r in rl)]
        else:
            plain.append("No matches this week")
        plain.append("\nFull results in CSV attachment.")

        subj = (f"\U0001F4CA Weekly Hammer-Green-Doji Breakout — "
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
            try: __import__("best_picks").attach_to_email(msg)
            except Exception as _bp_err: print(f"  ⚠️  Best 10 not added: {_bp_err}")
            srv.sendmail(gu, eto, msg.as_string())
        print(f"[Email] ✅  Sent to {', '.join(eto)} — {subj}")
    except Exception as e:
        print(f"[Email] ❌  {type(e).__name__}: {e}")

try:
    _send_email(results, fpath)
except Exception as e:
    print(f"[Email] ❌  Unexpected top-level error: {type(e).__name__}: {e}")

# -- Charts for top 5: last 16 weekly candles, W-3..W-1 shaded, W-1 high / stop
if results:
    top = results[:5]
    fig, axes = plt.subplots(len(top), 1, figsize=(13, 4.2*len(top)), facecolor="#0f172a")
    if len(top) == 1: axes = [axes]
    for ax, r in zip(axes, top):
        wk = r["_wk"].iloc[-16:]
        x = np.arange(len(wk))
        ax.set_facecolor("#0f172a")
        cols = ["#22c55e" if c > o else "#ef4444" for o, c in zip(wk["Open"], wk["Close"])]
        ax.vlines(x, wk["Low"], wk["High"], color=cols, lw=1.2)
        ax.bar(x, (wk["Close"] - wk["Open"]).abs().clip(lower=1e-9), bottom=np.minimum(wk["Open"], wk["Close"]),
               color=cols, width=0.6)
        n = len(wk)
        for k_, lab in ((n - 4, "W-3"), (n - 3, "W-2"), (n - 2, "W-1"), (n - 1, "now")):
            ax.axvspan(k_ - 0.45, k_ + 0.45, color="#1e3a5f", alpha=0.35, zorder=0)
            ax.text(k_, wk["High"].max(), lab, color="#cbd5e1", fontsize=7, ha="center", va="bottom")
        for lvl, c_, lab in ((r["W1_High"], "#facc15", "W-1 high"), (r["Stop_Loss"], "#ef4444", "Stop"),
                             (r["Target_1_2R"], "#38bdf8", "2R")):
            ax.axhline(lvl, color=c_, ls=":", lw=0.8)
            ax.text(n - 0.4, lvl, f" {lab} {lvl:.2f}", color=c_, fontsize=7, va="center")
        ax.set_xticks(x[::2]); ax.set_xticklabels([d.strftime("%m/%d") for d in wk.index[::2]], fontsize=7)
        ax.set_title(f"{r['Ticker']} | USD {r['Price']:.2f} | Score {r['Score']} | W-3 {r['W3_Type']} | W-2 vol {r['W2_Vol_x_W3']}x | "
                     f"W-1 {r['W1_Type']} | breakout {r['Signal_Date']}", color="#e2e8f0", fontsize=9, fontweight="bold")
        ax.tick_params(colors="#94a3b8", labelsize=7)
        for sp in ax.spines.values(): sp.set_edgecolor("#1e3a5f")
        ax.grid(color="#1e3a5f", ls="--", lw=0.5, alpha=0.6)
    plt.suptitle(f"Weekly Hammer/Doji -> Green -> Doji -> Daily Breakout - "
                 f"{datetime.today().strftime('%Y-%m-%d')}", color="#60a5fa", fontsize=12, fontweight="bold")
    plt.tight_layout()
    cp = os.path.join(out_dir, f"weekly_hammer_green_doji_breakout_chart_{ts}.png")
    plt.savefig(cp, dpi=130, bbox_inches="tight", facecolor="#0f172a")
    plt.close()
    print(f"  Chart -> {cp}")

print(f"""
{'=' * 55}
  PATTERN (current week vs the 3 completed weeks before it)
  1) W-3: red hammer or doji
  2) W-2: green, low above W-3 low, volume above W-3 volume
  3) W-1: doji or small hammer, low above the middle of W-2's body
  4) This week: a daily close above W-1's high
  Doji = body <= {CFG['doji_body_pct']:.0f}% of range and <= {CFG['doji_body_atr']} weekly ATR
  Hammer = lower wick >= {CFG['hammer_wick_x_body']:g}x body, upper wick <= {CFG['hammer_upper_wick_pct']:.0f}% of range
  Small = range <= {CFG['small_range_atr']:g} weekly ATR
  OUTPUT: Entry = signal-day close, Stop = W-1 low, T1 = +{CFG['target1_r']:g}R, T2 = +{CFG['target2_r']:g}R
{'=' * 55}
""")
