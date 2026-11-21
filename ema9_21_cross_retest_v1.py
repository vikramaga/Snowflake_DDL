# ============================================================
# NASDAQ / NYSE - 9/21 EMA Cross -> Retest -> Hold (entry on the hold) - daily, v1
# ============================================================
#
# From the "CROSS -> RETEST -> HOLD" chart (2026-11-21): the cross is not the entry.
#
#   1. DOWNTREND BEFORE: EMA9 below EMA21 on at least 15 of the 20 sessions before the cross.
#   2. CROSS: EMA9 crossed above EMA21 5-40 sessions ago and has stayed above it since.
#   3. RALLY: after the cross price ran to a swing high at least 2 ATR(14) above the
#      cross-day close.
#   4. RETEST: price pulled back from that high into the 9/21 zone - a candle in the
#      last 3 sessions (before today) with its low at or below EMA9 - and
#   5. HOLD: no close since the cross more than 0.5% below EMA21.
#   6. ENTRY (today): a green candle that closes back above EMA9 and above the previous
#      candle's high, after yesterday closed at or below EMA9 (first reclaim = entry).
# Groups: "Tagged EMA21" (retest low within 0.5% of EMA21 or below it) / "Held above EMA21".
# Entry = today's close | Stop = lowest retest low - 0.1 ATR | T1 = the post-cross swing
# high (2R if already above) | T2 = 2R. Liquidity $5 / 200k.
# TradingView list "EMA 9-21 Retest - YYYY-MM-DD" (IST). Settings in CFG.
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
    "history_days"     : 300,
    "min_bars"         : 80,
    "fast"             : 9,
    "slow"             : 21,
    "down_window"      : 20,      # EMA9 < EMA21 on >= 15 of the 20 sessions before the cross
    "down_min"         : 15,
    "cross_min_ago"    : 5,
    "cross_max_ago"    : 40,
    "rally_atr"        : 2.0,     # swing high >= cross close + 2 ATR
    "retest_window"    : 3,       # retest candle within the 3 sessions before today
    "hold_tol"         : 0.005,   # no close more than 0.5% under EMA21 since the cross
    "tag_tol"          : 0.005,   # "tagged EMA21": retest low <= EMA21 + 0.5%
    "atr_period"       : 14,
    "stop_buffer_atr"  : 0.1,
    "min_price"        : 5.0,
    "min_avg_volume"   : 200_000,
    "tv_list_prefix"   : "EMA 9-21 Retest",
    "batch_size"       : 50,
    "batch_sleep"      : 1.5,
}

FUNNEL = {"tickers_checked": 0, "cross": 0, "rally": 0, "retest": 0, "hold": 0, "entry": 0}

def analyze_stock(sym, df):
    if df is None or len(df) < CFG["min_bars"]: return None
    o, h, l, c, v = (df[k].values.astype(float) for k in ("Open", "High", "Low", "Close", "Volume"))
    i = len(df) - 1
    if c[i] < CFG["min_price"] or float(np.mean(v[-20:])) < CFG["min_avg_volume"]: return None
    cs = df["Close"].astype(float)
    e9 = cs.ewm(span=CFG["fast"], adjust=False).mean().values
    e21 = cs.ewm(span=CFG["slow"], adjust=False).mean().values
    pc = np.concatenate([[np.nan], c[:-1]])
    tr = np.nanmax(np.vstack([h - l, np.abs(h - pc), np.abs(l - pc)]), axis=0)
    atr = pd.Series(tr).rolling(CFG["atr_period"]).mean().values
    FUNNEL["tickers_checked"] += 1

    # 1-2. the latest bullish cross, 5-40 sessions ago, EMA9 above EMA21 ever since, downtrend before it
    above = e9 > e21
    xs = [x for x in range(i - CFG["cross_max_ago"], i - CFG["cross_min_ago"] + 1) if not above[x - 1] and above[x]]
    if not xs: return None
    x = xs[-1]
    if not above[x:i + 1].all(): return None
    dw = CFG["down_window"]
    if int(np.sum(~above[x - dw:x])) < CFG["down_min"]: return None
    FUNNEL["cross"] += 1

    # 3. rally to a swing high >= 2 ATR above the cross close
    rw = CFG["retest_window"]
    r0 = i - rw
    p = x + int(np.argmax(h[x:r0]))                             # post-cross swing high before the retest window
    if not (p < r0 and np.isfinite(atr[x]) and h[p] >= c[x] + CFG["rally_atr"] * atr[x]): return None
    if h[p] < h[x:i].max(): return None                         # no higher high after the swing high
    FUNNEL["rally"] += 1

    # 4. retest into the 9/21 zone within the last 3 sessions before today
    rr = [r for r in range(r0, i) if l[r] <= e9[r]]
    if not rr: return None
    FUNNEL["retest"] += 1

    # 5. hold: no close since the cross more than 0.5% under EMA21
    if any(c[j] < e21[j] * (1 - CFG["hold_tol"]) for j in range(x, i + 1)): return None
    FUNNEL["hold"] += 1

    # 6. entry today: first close back above EMA9, green, above the previous high
    if not (c[i] > o[i] and c[i] > e9[i] and c[i] > h[i - 1] and c[i - 1] <= e9[i - 1]): return None
    FUNNEL["entry"] += 1

    rlow_i = min(rr, key=lambda r: l[r]); rlow = float(l[rlow_i])
    tagged = rlow <= e21[rlow_i] * (1 + CFG["tag_tol"])
    a = float(atr[i]); entry = float(c[i]); stop = rlow - CFG["stop_buffer_atr"] * a; risk = entry - stop
    if risk <= 0: return None
    t1, t1_src = (float(h[p]), "post-cross swing high") if h[p] > entry else (entry + 2 * risk, "2R")
    rally_atr = (h[p] - c[x]) / atr[x]
    pull_pct = (h[p] - rlow) / (h[p] - c[x]) * 100 if h[p] > c[x] else np.nan
    vol_up = v[i] > v[i - 1]
    vavg = float(np.mean(v[i - 20:i])); vx = v[i] / vavg if vavg > 0 else np.nan
    pull_vol = float(np.mean(v[p + 1:i])) if i - p > 1 else np.nan
    rally_vol = float(np.mean(v[x:p + 1]))
    dry = np.isfinite(pull_vol) and pull_vol < rally_vol
    score = 45 + (10 if tagged else 5) + (10 if vol_up else 0) + (10 if dry else 0) \
            + min(10, rally_atr * 2) + (min(10, (t1 - entry) / risk * 3)) \
            + (5 if e21[i] > e21[i - 5] else 0)
    flags = [f"9/21 cross {i - x} sessions ago after {int(np.sum(~above[x - dw:x]))}/{dw} sessions below",
             f"rally {rally_atr:.1f} ATR to {h[p]:.2f} on {df.index[p]:%Y-%m-%d}",
             f"retest low {rlow:.2f} ({'tagged EMA21' if tagged else 'held above EMA21'}), retraced {pull_pct:.0f}% of the rally",
             "pullback volume lighter than the rally" if dry else "pullback volume not lighter",
             f"entry volume x{v[i] / max(v[i - 1], 1):.2f} vs yesterday, x{vx:.1f} 20d avg"]
    return {"Score": round(min(100, score)), "Retest": "Tagged EMA21" if tagged else "Held above EMA21",
            "Price": round(entry, 2), "Entry_Price": round(entry, 2), "Stop_Loss": round(stop, 2),
            "Risk_%": round(risk / entry * 100, 2), "Target_1": round(t1, 2), "T1_Source": t1_src,
            "Reward_R": round((t1 - entry) / risk, 2), "Target_2R": round(entry + 2 * risk, 2),
            "EMA9": round(float(e9[i]), 2), "EMA21": round(float(e21[i]), 2),
            "Cross_Date": df.index[x].strftime("%Y-%m-%d"), "Swing_High": round(float(h[p]), 2),
            "Swing_High_Date": df.index[p].strftime("%Y-%m-%d"), "Rally_ATR": round(rally_atr, 1),
            "Retest_Low": round(rlow, 2), "Retest_Date": df.index[rlow_i].strftime("%Y-%m-%d"),
            "Retrace_%": round(pull_pct, 0) if np.isfinite(pull_pct) else None,
            "Volume_Up_Today": "Yes" if vol_up else "No", "Pullback_Vol_Lighter": "Yes" if dry else "No",
            "Vol_x_20d": round(vx, 2) if np.isfinite(vx) else None,
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
GROUPS = [("Retest tagged EMA21", lambda r: r["Retest"] == "Tagged EMA21"),
          ("Retest held above EMA21", lambda r: True)]
for r in results:
    r["Group"] = next(g for g, f in GROUPS if f(r))
fc = FUNNEL
print(f"\n{'=' * 65}\n  SCAN COMPLETE - {len(results)} matches from {got_daily} tickers with data")
print(f"  checked {fc['tickers_checked']} -> 9/21 cross {fc['cross']} -> rally {fc['rally']} -> retest {fc['retest']}"
      f" -> hold {fc['hold']} -> entry today {fc['entry']}")
for g, _ in GROUPS:
    names = [r["Ticker"] for r in results if r["Group"] == g]
    print(f"  {g}: {len(names)}  {', '.join(names[:60])}")
print("=" * 65)

ts      = datetime.today().strftime("%Y%m%d_%H%M")
out_dir = os.environ.get("GITHUB_WORKSPACE", os.getcwd())
COLS = ["Ticker","Group","Retest","Score","Price","Entry_Price","Stop_Loss","Risk_%","Target_1","T1_Source","Reward_R",
        "Target_2R","EMA9","EMA21","Cross_Date","Swing_High","Swing_High_Date","Rally_ATR","Retest_Low","Retest_Date",
        "Retrace_%","Volume_Up_Today","Pullback_Vol_Lighter","Vol_x_20d","Signal_Date","Flags"]
df_out = pd.DataFrame(results) if results else pd.DataFrame(columns=COLS)
df_out = df_out[[c for c in COLS if c in df_out.columns]].reset_index(drop=True)
fpath = os.path.join(out_dir, f"ema9_21_cross_retest_{ts}.csv")
try:
    from position_sizing import add_position_size
    df_out = add_position_size(df_out)
    try: __import__("best_picks").rank_best(df_out)
    except Exception as _bp_err: print(f"  Best 10 skipped: {_bp_err}")
except Exception as _ps_err:
    print(f"  Position sizing skipped: {_ps_err}")
df_out.to_csv(fpath, index=False)
print(f"\n  CSV -> {fpath}")
tv = os.path.join(out_dir, f"tv_ema9_21_cross_retest_{ts}.txt")
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

EMAIL_COLS = ["Ticker","Score","Price","Stop_Loss","Risk_%","Target_1","Reward_R","Cross_Date","Rally_ATR",
              "Retest_Date","Retrace_%","Volume_Up_Today","Pullback_Vol_Lighter"]
def _table(rows):
    th = "".join(f'<th style="background:#1e293b;color:#e2e8f0;padding:6px 8px;font-size:11px;white-space:nowrap">{c}</th>' for c in EMAIL_COLS)
    body = ""
    for i, r in enumerate(rows):
        bg = "#fff" if i % 2 == 0 else "#f0f9ff"
        body += f'<tr style="background:{bg}">' + "".join(
            f'<td style="padding:5px 8px;font-size:12px;{"font-weight:700" if c in ("Ticker", "Reward_R") else ""}">{r.get(c, "-")}</td>'
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
<h1 style="margin:0;color:#60a5fa;font-size:19px">9/21 EMA Cross &rarr; Retest &rarr; Hold</h1>
<p style="margin:6px 0 0;color:#94a3b8;font-size:12px">{datetime.today().strftime('%Y-%m-%d %H:%M UTC')} &middot; {len(rl)} matches
&middot; TradingView: {TV_STATUS}</p></td></tr>
<tr><td style="padding:16px 22px">{secs}
<p style="font-size:11px;color:#64748b">The cross is not the entry. EMA9 crossed above EMA21 {CFG['cross_min_ago']}-{CFG['cross_max_ago']} sessions ago (after being below it on
&ge; {CFG['down_min']} of the {CFG['down_window']} sessions before) and stayed above. Price rallied &ge; {CFG['rally_atr']:g} ATR above the cross close, pulled back into the
9/21 zone (a low at or below EMA9 in the last {CFG['retest_window']} sessions) without closing more than {CFG['hold_tol'] * 100:g}% under EMA21, and today a green candle
closed back above EMA9 and the previous high (yesterday closed at or below EMA9) = entry. Stop = lowest retest low - {CFG['stop_buffer_atr']:g} ATR &middot;
T1 = post-cross swing high (2R if above) &middot; T2 = 2R. Funnel: checked {FUNNEL['tickers_checked']} &rarr; cross {FUNNEL['cross']} &rarr; rally {FUNNEL['rally']}
&rarr; retest {FUNNEL['retest']} &rarr; hold {FUNNEL['hold']} &rarr; entry {FUNNEL['entry']}. Full results (with Quantity for a fixed max loss) attached.</p></td></tr>
<tr><td style="background:#f8fafc;padding:10px;text-align:center;color:#94a3b8;font-size:10px">Not financial advice &middot; Auto-generated by GitHub Actions</td></tr>
</table></td></tr></table></body></html>"""
        plain = [f"9/21 EMA Cross -> Retest -> Hold - {datetime.today().strftime('%Y-%m-%d')}", f"{len(rl)} matches", f"TradingView: {TV_STATUS}", ""]
        for g, _ in GROUPS:
            plain.append(f"{g}: " + ", ".join(r["Ticker"] for r in rl if r["Group"] == g))
        nT = sum(1 for r in rl if r["Retest"] == "Tagged EMA21")
        subj = f"9/21 EMA Cross-Retest-Hold - {len(rl)} entries, {nT} tagged EMA21 - {datetime.today().strftime('%Y-%m-%d')}"
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
