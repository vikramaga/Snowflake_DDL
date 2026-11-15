# ============================================================
# NASDAQ / NYSE - Weekly Close Crosses Above a Rising 30-Week SMA
#                 + Daily Volume at a 21-Day High   (v1)
# ============================================================
#
# Port of Vik's Chartink screen (2026-11-15) to US stocks. Weekly bars end
# Friday and are resampled from daily bars; the current (in-progress) week
# counts, as on Chartink. SMA30 = 30-week simple average of weekly closes.
# All of these must pass:
#   1. 1 week ago: weekly close <= weekly SMA30
#   2. Weekly close crossed above weekly SMA30 (this week's close > SMA30
#      after last week's close was at or below it)
#   3. Daily volume (latest session) >= the highest daily volume of the 21
#      sessions before it ("1 day ago Max(21, Daily Volume)")
#   4. Weekly SMA30 > 1 week ago SMA30
#   5. 3 weeks ago SMA30 > 4 weeks ago SMA30
#   6. 5 weeks ago SMA30 > 7 weeks ago SMA30
# (The screen's "Weekly Low < Weekly SMA30" row is switched off there, so it
#  is not used here either.)
#
# Entry = latest close; Stop = this week's low (reasonable default, not part
# of the screen); Target = 2R. Liquidity filters $5 / 200k avg volume (the
# screen had none) - all in CFG.
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
    # 500 calendar days -> ~345 daily bars -> ~72 weekly bars (SMA30 + 7 weeks back needs 38)
    "history_days"            : 500,
    "min_weekly_bars"         : 40,
    "sma_weeks"               : 30,
    "vol_lookback_days"       : 21,    # latest volume >= max of the 21 sessions before it

    "min_avg_volume"          : 200_000,
    "min_price"               : 5.0,

    "batch_size"              : 50,
    "batch_sleep"             : 1.5,
}

# -- Indicators
def resample_weekly(df):
    """Weekly bars ending Friday; the last week may be in progress."""
    agg = {"Open": "first", "High": "max", "Low": "min", "Close": "last", "Volume": "sum"}
    w = df.resample("W-FRI").agg(agg).dropna(subset=["Close"])
    w["Days"] = df["Close"].resample("W-FRI").count().reindex(w.index).values
    return w

def check_pattern(df_daily, wk, cfg):
    """
    SINGLE SOURCE OF TRUTH. Evaluates every condition (no short-circuit) so the
    funnel shows where tickers drop out. Returns (passed, details); {} if data short.
    """
    if len(wk) < cfg["min_weekly_bars"] or len(df_daily) < cfg["vol_lookback_days"] + 2:
        return False, {}
    c = wk["Close"].values.astype(float)
    sma = wk["Close"].rolling(cfg["sma_weeks"]).mean().values
    s = lambda k: sma[-1 - k]                       # SMA30 k weeks ago
    if not np.isfinite(s(7)):
        return False, {}
    v = df_daily["Volume"].values.astype(float)
    vmax = v[-1 - cfg["vol_lookback_days"]:-1].max()
    d = {
        "step1": bool(c[-2] <= s(1)),                       # 1 week ago close <= SMA30
        "step2": bool(c[-1] > s(0) and c[-2] <= s(1)),      # crossed above
        "step3": bool(v[-1] >= vmax),                       # volume >= 21-day max (prior sessions)
        "step4": bool(s(0) > s(1)),
        "step5": bool(s(3) > s(4)),
        "step6": bool(s(5) > s(7)),
        "sma0": float(s(0)), "sma1": float(s(1)), "vmax": float(vmax), "vol": float(v[-1]),
    }
    return all(d[f"step{k}"] for k in range(1, 7)), d

# -- Diagnostic funnel: cumulative (each count = passed this AND all earlier steps)
FUNNEL_COUNTS = {
    "tickers_checked": 0,
    "passed_step1_prev_week_close_le_sma30": 0,
    "passed_step2_close_crossed_above_sma30": 0,
    "passed_step3_volume_ge_21d_max": 0,
    "passed_step4_sma30_up_this_week": 0,
    "passed_step5_sma30_up_3v4_weeks_ago": 0,
    "passed_step6_sma30_up_5v7_weeks_ago": 0,
}
_FK = list(FUNNEL_COUNTS)[1:]

def analyze_stock(sym, df_daily):
    if df_daily is None: return None
    price   = float(df_daily["Close"].iloc[-1])
    avg_vol = float(df_daily["Volume"].tail(20).mean())
    if price < CFG["min_price"] or avg_vol < CFG["min_avg_volume"]: return None
    wk = resample_weekly(df_daily)
    ok, d = check_pattern(df_daily, wk, CFG)
    if not d: return None
    FUNNEL_COUNTS["tickers_checked"] += 1
    cum = True
    for k, key in enumerate(_FK):
        cum = cum and d[f"step{k+1}"]
        FUNNEL_COUNTS[key] += cum
    if not ok: return None

    entry = price
    stop = float(wk["Low"].iloc[-1])
    if stop <= 0 or entry <= stop: return None
    risk = entry - stop
    risk_pct = risk / entry * 100
    above_pct = (entry / d["sma0"] - 1) * 100
    vol_x = d["vol"] / d["vmax"] if d["vmax"] > 0 else np.nan
    sma_slope = (d["sma0"] / d["sma1"] - 1) * 100
    s200 = df_daily["Close"].rolling(200).mean().iloc[-1]
    wk_chg = (entry / float(wk["Close"].iloc[-2]) - 1) * 100

    score = 40
    score += min(20, max(0, (vol_x - 1) * 20)) if np.isfinite(vol_x) else 0      # volume beyond the 21-day max
    score += max(0, 15 - max(0, above_pct - 2) * 2)                              # not stretched above SMA30
    score += max(0, 10 - max(0, risk_pct - 5) * 2)                               # tighter stop
    score += min(10, max(0, sma_slope) * 10)                                     # SMA30 slope
    score += 5 if np.isfinite(s200) and entry > s200 else 0
    score = round(min(100, max(0, score)))
    flags = [f"{above_pct:+.1f}% above SMA30", f"volume {vol_x:.2f}x the 21-day max",
             f"week {wk_chg:+.1f}%"]
    if int(wk["Days"].iloc[-1]) < 5: flags.append(f"week in progress ({int(wk['Days'].iloc[-1])} days)")
    if np.isfinite(s200): flags.append("above SMA200" if entry > s200 else "below SMA200")

    return {
        "Score": score, "Price": round(price, 2),
        "Entry_Price": round(entry, 2), "Stop_Loss": round(stop, 2), "Risk_%": round(risk_pct, 1),
        "Target_2R": round(entry + 2 * risk, 2),
        "SMA30w": round(d["sma0"], 2), "Above_SMA30_%": round(above_pct, 1),
        "SMA30w_Slope_%": round(sma_slope, 2), "Week_Chg_%": round(wk_chg, 1),
        "Volume": int(d["vol"]), "Vol_21d_Max": int(d["vmax"]), "Vol_x_21d_Max": round(vol_x, 2),
        "Week_Ending": wk.index[-1].strftime("%Y-%m-%d"),
        "Signal_Date": df_daily.index[-1].strftime("%Y-%m-%d"),
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
        r = analyze_stock(sym, daily_map[sym])
        if r is None: continue
        r["Ticker"] = sym
        results.append(r)
        print(f"  OK {sym:<6} ${r['Price']:.2f}  Score {r['Score']}  SMA30w {r['SMA30w']}  "
              f"vol {r['Vol_x_21d_Max']}x 21d max  stop {r['Stop_Loss']} ({r['Risk_%']}%)")
    except Exception: pass

fc = FUNNEL_COUNTS
print(f"\n{'=' * 65}\n  SCAN COMPLETE - {len(results)} matches from {got_daily} tickers with data")
print(f"  Tickers checked                                   : {fc['tickers_checked']}")
print(f"  Step 1 - last week's close <= its SMA30           : {fc['passed_step1_prev_week_close_le_sma30']}")
print(f"  Step 2 - + this week's close crossed above SMA30  : {fc['passed_step2_close_crossed_above_sma30']}")
print(f"  Step 3 - + daily volume >= prior 21-day max       : {fc['passed_step3_volume_ge_21d_max']}")
print(f"  Step 4 - + SMA30 up this week                     : {fc['passed_step4_sma30_up_this_week']}")
print(f"  Step 5 - + SMA30 3 wks ago > 4 wks ago            : {fc['passed_step5_sma30_up_3v4_weeks_ago']}")
print(f"  Step 6 - + SMA30 5 wks ago > 7 wks ago            : {fc['passed_step6_sma30_up_5v7_weeks_ago']}  (= full screen)")
print("=" * 65)

results.sort(key=lambda x: x["Score"], reverse=True)

# -- Save CSV + TradingView list
ts      = datetime.today().strftime("%Y%m%d_%H%M")
out_dir = os.environ.get("GITHUB_WORKSPACE", os.getcwd())
COLS = ["Ticker","Price","Score","Entry_Price","Stop_Loss","Risk_%","Target_2R","SMA30w","Above_SMA30_%",
        "SMA30w_Slope_%","Week_Chg_%","Volume","Vol_21d_Max","Vol_x_21d_Max","Week_Ending","Signal_Date","Flags"]
df_out = pd.DataFrame([{k: v for k, v in r.items() if not k.startswith("_")}
                       for r in results]) if results else pd.DataFrame(columns=COLS)
df_out = df_out[[c for c in COLS if c in df_out.columns]].reset_index(drop=True)

fpath = os.path.join(out_dir, f"weekly_sma30_cross_vol21_{ts}.csv")
try:
    from position_sizing import add_position_size
    df_out = add_position_size(df_out)
    try: __import__("best_picks").rank_best(df_out)
    except Exception as _bp_err: print(f"  Best 10 skipped: {_bp_err}")
except Exception as _ps_err:
    print(f"  Position sizing skipped: {_ps_err}")
df_out.to_csv(fpath, index=False)
print(f"\n  CSV -> {fpath}")
tv = os.path.join(out_dir, f"tv_weekly_sma30_cross_vol21_{ts}.txt")
with open(tv, "w") as f:
    f.write(f"###SMA30w Cross + Vol {datetime.today().strftime('%Y-%m-%d')}\n")
    for r in results: f.write(f"NASDAQ:{r['Ticker']}\n")
print(f"  TradingView -> {tv}")
if results:
    print(f"\n  Tickers: {', '.join(r['Ticker'] for r in results)}")

# -- Email with CSV attached
EMAIL_COLS = ["Ticker","Price","Score","Entry_Price","Stop_Loss","Risk_%","Target_2R","SMA30w","Above_SMA30_%",
              "SMA30w_Slope_%","Week_Chg_%","Vol_x_21d_Max"]

def _fmt_cell(col, v):
    try:
        if col in ("Price","Entry_Price","Stop_Loss","Target_2R","SMA30w"): return f"${float(v):.2f}"
        if col in ("Risk_%","Above_SMA30_%","Week_Chg_%"): return f"{float(v):+.1f}%" if col != "Risk_%" else f"{float(v):.1f}%"
        if col == "SMA30w_Slope_%": return f"{float(v):+.2f}%"
        if col == "Vol_x_21d_Max": return f"{float(v):.2f}x"
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
        for i, r in enumerate(rl[:60]):
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
<table width="100%" style="max-width:900px;background:#fff;border-radius:12px;overflow:hidden">
<tr><td style="background:linear-gradient(135deg,#0f172a,#1e3a5f);padding:20px 26px">
<h1 style="margin:0;color:#60a5fa;font-size:19px">Weekly Close Crosses Above Rising SMA30 + 21-Day Volume High</h1>
<p style="margin:6px 0 0;color:#94a3b8;font-size:12px">{datetime.today().strftime('%Y-%m-%d %H:%M UTC')} &middot; {cnt} match{'es' if cnt != 1 else ''}
 - US port of the Chartink screen: last week closed at or below its 30-week SMA, this week's close is above it,
 the 30-week SMA is rising (this week, 3 vs 4 weeks ago, 5 vs 7 weeks ago), and the latest daily volume is the highest in 22 sessions</p>
</td></tr>
<tr><td style="padding:12px 26px;background:#0b1220"><p style="margin:0;color:#cbd5e1;font-size:12px">
FUNNEL: {fc['tickers_checked']} checked &rarr; {fc['passed_step1_prev_week_close_le_sma30']} last week at/below SMA30
&rarr; {fc['passed_step2_close_crossed_above_sma30']} + crossed above
&rarr; {fc['passed_step3_volume_ge_21d_max']} + volume &ge; 21-day max
&rarr; {fc['passed_step4_sma30_up_this_week']} + SMA30 up this week
&rarr; {fc['passed_step5_sma30_up_3v4_weeks_ago']} + up 3 vs 4 weeks ago
&rarr; <b style="color:#facc15">{fc['passed_step6_sma30_up_5v7_weeks_ago']} + up 5 vs 7 weeks ago = full match</b></p></td></tr>
<tr><td style="padding:16px"><div style="overflow-x:auto;border:1px solid #e2e8f0;border-radius:8px">
<table style="border-collapse:collapse;width:100%"><thead><tr>{th}</tr></thead><tbody>{rows}</tbody></table></div>
{tick_html}
<p style="font-size:11px;color:#64748b;margin:10px 0 0">Weekly bars end Friday; the current week counts while in progress (as on Chartink).
Entry = latest close &middot; Stop = this week's low &middot; Target_2R = entry + 2 &times; risk (defaults, not part of the screen).
Filters: price &ge; ${CFG['min_price']:g}, 20-day avg volume &ge; {CFG['min_avg_volume']:,}.
Full results (with Quantity for a fixed max loss) attached as CSV.</p>
</td></tr>
<tr><td style="background:#f8fafc;padding:10px;text-align:center;color:#94a3b8;font-size:10px">
Not financial advice &middot; Auto-generated by GitHub Actions</td></tr>
</table></td></tr></table></body></html>"""

        plain = [f"Weekly SMA30 Cross + 21-Day Volume High - {datetime.today().strftime('%Y-%m-%d')}",
                 f"{cnt} matches",
                 "FUNNEL: " + " -> ".join(str(fc[k]) for k in fc) + " (=full match)",
                 "=" * 60]
        for r in rl[:60]:
            plain.append(f"{r['Ticker']:<6} ${r['Price']:.2f}  Score:{r['Score']}  SL:${r['Stop_Loss']:.2f} ({r['Risk_%']}%)  "
                         f"SMA30w:${r['SMA30w']:.2f} ({r['Above_SMA30_%']:+.1f}%)  vol {r['Vol_x_21d_Max']}x 21d max")
        if rl:
            plain += ["", "Tickers: " + ", ".join(r["Ticker"] for r in rl)]
        else:
            plain.append("No matches today")
        plain.append("\nFull results in CSV attachment.")

        subj = (f"Weekly SMA30 Cross + Vol High - {cnt} signal{'s' if cnt != 1 else ''} - "
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

# -- Charts for the top 5: last 60 weekly candles with SMA30, stop
if results:
    top = results[:5]
    fig, axes = plt.subplots(len(top), 1, figsize=(13, 4.2*len(top)), facecolor="#0f172a")
    if len(top) == 1: axes = [axes]
    for ax, r in zip(axes, top):
        wk = r["_wk"]
        sma = wk["Close"].rolling(CFG["sma_weeks"]).mean()
        d = wk.iloc[-60:]; s_ = sma.iloc[-60:]; n_ = len(d); x = np.arange(n_)
        ax.set_facecolor("#0f172a")
        cols = ["#22c55e" if c > o else "#ef4444" for o, c in zip(d["Open"], d["Close"])]
        ax.vlines(x, d["Low"], d["High"], color=cols, lw=0.8)
        ax.bar(x, (d["Close"] - d["Open"]).abs().clip(lower=1e-9), bottom=np.minimum(d["Open"], d["Close"]),
               color=cols, width=0.6)
        ax.plot(x, s_.values, color="#facc15", lw=1.3, label="SMA30 (weekly)")
        ax.axhline(r["Stop_Loss"], color="#ef4444", ls=":", lw=0.9)
        ax.text(n_ - 0.4, r["Stop_Loss"], f" Stop {r['Stop_Loss']:.2f}", color="#ef4444", fontsize=7, va="center")
        ax.legend(loc="upper left", fontsize=7, facecolor="#0f172a", labelcolor="#cbd5e1")
        ax.set_xticks(x[::6]); ax.set_xticklabels([t.strftime("%y/%m/%d") for t in d.index[::6]], fontsize=7)
        ax.set_title(f"{r['Ticker']} | USD {r['Price']:.2f} | Score {r['Score']} | {r['Above_SMA30_%']:+.1f}% vs SMA30w | "
                     f"vol {r['Vol_x_21d_Max']}x 21d max | {r['Signal_Date']}",
                     color="#e2e8f0", fontsize=9, fontweight="bold")
        ax.tick_params(colors="#94a3b8", labelsize=7)
        for sp in ax.spines.values(): sp.set_edgecolor("#1e3a5f")
        ax.grid(color="#1e3a5f", ls="--", lw=0.5, alpha=0.6)
    plt.suptitle(f"Weekly SMA30 Cross + 21-Day Volume High - {datetime.today().strftime('%Y-%m-%d')}",
                 color="#60a5fa", fontsize=12, fontweight="bold")
    plt.tight_layout(rect=[0, 0, 1, 0.985])
    cp = os.path.join(out_dir, f"weekly_sma30_cross_vol21_chart_{ts}.png")
    plt.savefig(cp, dpi=130, bbox_inches="tight", facecolor="#0f172a")
    plt.close()
    print(f"  Chart -> {cp}")
