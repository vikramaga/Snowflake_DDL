# ============================================================
# NASDAQ / NYSE - LuxAlgo Support & Resistance: on last support / crossed
#                 above last resistance  (daily, v1)
# ============================================================
#
# Recreates LuxAlgo's "Support and Resistance Levels with Breaks" (TradingView):
#   Resistance = fixnan(pivothigh(15, 15)[1])  - the last pivot high (15 bars each
#   Support    = fixnan(pivotlow(15, 15)[1])     side), known 16 bars after the pivot
#   Volume osc = 100 * (EMA5(volume) - EMA10(volume)) / EMA10(volume)
#   Break "B"  = close crosses above resistance, volume osc > 20, and NOT a bull wick
#   Bull Wick  = close crosses above resistance with (open - low) > (close - open)
#
# Lists (latest daily session):
#   RESISTANCE BREAK  - today's close crossed above the last resistance
#                       (Signal = "B" / "Bull Wick" / "Cross, volume osc < 20")
#   BREAK HOLDING     - crossed above it 2-5 sessions ago and every close since held above it
#   ON SUPPORT        - today's low came within 0.5 ATR(14) of the last support (or below
#                       it) and the close is at or above support, within 1 ATR of it;
#                       not a close below support
#
# Breakout: Entry = close, Stop = breakout candle low (or resistance - 0.5 ATR if lower),
#           Target = 2R.  Support: Entry = close, Stop = support - 0.5 ATR,
#           Target = the last resistance (Reward_R; 2R if resistance is below the entry).
# Liquidity $5 / 200k average volume. Also creates TradingView list
# "LuxAlgo SR - YYYY-MM-DD" (IST). Settings in CFG.
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
    "history_days"        : 500,
    "min_bars"            : 120,
    "left_bars"           : 15,
    "right_bars"          : 15,
    "volume_thresh"       : 20,
    "atr_period"          : 14,
    "support_touch_atr"   : 0.5,     # low within 0.5 ATR above support (or below it)
    "support_close_atr"   : 1.0,     # close at most 1 ATR above support
    "stop_buffer_atr"     : 0.5,
    "hold_lookback"       : 5,       # "break holding": crossed 2-5 sessions ago
    "min_price"           : 5.0,
    "min_avg_volume"      : 200_000,
    "tv_list_prefix"      : "LuxAlgo SR",
    "batch_size"          : 50,
    "batch_sleep"         : 1.5,
}

def lux_levels(df, cfg):
    """LuxAlgo levels: fixnan(pivothigh(L, R)[1]) / fixnan(pivotlow(L, R)[1])."""
    L, R = cfg["left_bars"], cfg["right_bars"]
    h, l = df["High"].astype(float), df["Low"].astype(float)
    hmax = h[::-1].rolling(R + 1).max()[::-1]           # max of bar .. bar+R
    hmax = pd.concat([h.rolling(L + 1).max(), hmax], axis=1).max(axis=1)
    lmin = pd.concat([l.rolling(L + 1).min(), l[::-1].rolling(R + 1).min()[::-1]], axis=1).min(axis=1)
    ok = h.rolling(L + 1).count().ge(L + 1) & h[::-1].rolling(R + 1).count()[::-1].ge(R + 1)
    ph = h.where((h >= hmax) & ok)
    pl = l.where((l <= lmin) & ok)
    res = ph.shift(R + 1).ffill().values                  # known R bars later, plus [1]
    sup = pl.shift(R + 1).ffill().values
    v = df["Volume"].astype(float)
    e5, e10 = v.ewm(span=5, adjust=False).mean(), v.ewm(span=10, adjust=False).mean()
    osc = (100 * (e5 - e10) / e10.replace(0, np.nan)).values
    return res, sup, osc

def atr(df, n):
    h, l, c = (df[k].values.astype(float) for k in ("High", "Low", "Close"))
    pc = np.concatenate([[np.nan], c[:-1]])
    tr = np.nanmax(np.vstack([h - l, np.abs(h - pc), np.abs(l - pc)]), axis=0)
    return pd.Series(tr).rolling(n).mean().values

FUNNEL = {"tickers_checked": 0, "resistance_break": 0, "break_holding": 0, "on_support": 0}

def analyze_stock(sym, df):
    if df is None or len(df) < CFG["min_bars"]: return []
    if float(df["Close"].iloc[-1]) < CFG["min_price"] or float(df["Volume"].tail(20).mean()) < CFG["min_avg_volume"]:
        return []
    res, sup, osc = lux_levels(df, CFG)
    a = atr(df, CFG["atr_period"])
    o, h, l, c = (df[k].values.astype(float) for k in ("Open", "High", "Low", "Close"))
    i = len(df) - 1
    if not (np.isfinite(res[i]) and np.isfinite(sup[i]) and np.isfinite(a[i]) and a[i] > 0): return []
    FUNNEL["tickers_checked"] += 1
    s200 = df["Close"].rolling(200).mean().values[i]
    common = {"Price": round(c[i], 2), "Resistance": round(res[i], 2), "Support": round(sup[i], 2),
              "Vol_Osc": round(osc[i], 1) if np.isfinite(osc[i]) else None,
              "Above_SMA200": "Yes" if np.isfinite(s200) and c[i] > s200 else "No",
              "Signal_Date": df.index[i].strftime("%Y-%m-%d"), "_df": df, "_res": res, "_sup": sup}
    out = []
    # 1) resistance break today
    if c[i] > res[i] and c[i - 1] <= res[i - 1]:
        wick = (o[i] - l[i]) > (c[i] - o[i])
        sig = "Bull Wick" if wick else ("B" if osc[i] > CFG["volume_thresh"] else "Cross, vol osc < 20")
        stop = min(l[i], res[i] - CFG["stop_buffer_atr"] * a[i])
        risk = c[i] - stop
        if risk > 0:
            FUNNEL["resistance_break"] += 1
            score = 50 + (25 if sig == "B" else 10 if sig == "Bull Wick" else 0) \
                    + min(15, max(0, osc[i]) / 4) + (10 if common["Above_SMA200"] == "Yes" else 0)
            out.append({**common, "List": "Resistance break", "Signal": sig, "Score": round(min(100, score)),
                        "Entry_Price": round(c[i], 2), "Stop_Loss": round(stop, 2), "Risk_%": round(risk / c[i] * 100, 1),
                        "Target_1": round(c[i] + 2 * risk, 2), "Reward_R": 2.0,
                        "Dist_%": round((c[i] / res[i] - 1) * 100, 1)})
    # 2) break 2-5 sessions ago, still holding
    else:
        for j in range(i - 1, i - CFG["hold_lookback"] - 1, -1):
            if c[j] > res[j] and c[j - 1] <= res[j - 1]:
                lvl = res[j]
                if all(c[k] > lvl for k in range(j, i + 1)) and res[i] == lvl:
                    stop = lvl - CFG["stop_buffer_atr"] * a[i]; risk = c[i] - stop
                    wick = (o[j] - l[j]) > (c[j] - o[j])
                    sig = "Bull Wick" if wick else ("B" if osc[j] > CFG["volume_thresh"] else "Cross, vol osc < 20")
                    FUNNEL["break_holding"] += 1
                    score = 40 + (25 if sig == "B" else 10 if sig == "Bull Wick" else 0) + max(0, 15 - (c[i] / lvl - 1) * 300) \
                            + (10 if common["Above_SMA200"] == "Yes" else 0)
                    out.append({**common, "List": "Break holding", "Signal": f"{sig} on {df.index[j]:%Y-%m-%d}",
                                "Score": round(min(100, score)), "Entry_Price": round(c[i], 2), "Stop_Loss": round(stop, 2),
                                "Risk_%": round(risk / c[i] * 100, 1), "Target_1": round(c[i] + 2 * risk, 2), "Reward_R": 2.0,
                                "Dist_%": round((c[i] / lvl - 1) * 100, 1)})
                break
    # 3) on support
    S = sup[i]
    touch = l[i] <= S + CFG["support_touch_atr"] * a[i]
    hold = S <= c[i] <= S + CFG["support_close_atr"] * a[i]
    if touch and hold and not (c[i] < S):
        stop = S - CFG["stop_buffer_atr"] * a[i]; risk = c[i] - stop
        t1 = res[i] if res[i] > c[i] else c[i] + 2 * risk
        rr = (t1 - c[i]) / risk
        bounce = c[i] > o[i] or (min(o[i], c[i]) - l[i]) >= abs(c[i] - o[i])
        pierced = l[i] < S
        sig = ("Swept below, closed back above" if pierced else "Tested") + (" + bounce candle" if bounce else "")
        FUNNEL["on_support"] += 1
        score = 45 + (15 if bounce else 0) + (10 if pierced else 0) + min(20, rr * 5) + (10 if common["Above_SMA200"] == "Yes" else 0)
        out.append({**common, "List": "On support", "Signal": sig, "Score": round(min(100, score)),
                    "Entry_Price": round(c[i], 2), "Stop_Loss": round(stop, 2), "Risk_%": round(risk / c[i] * 100, 1),
                    "Target_1": round(t1, 2), "Reward_R": round(rr, 1), "Dist_%": round((c[i] / S - 1) * 100, 1)})
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

for sym in tqdm(list(daily_map.keys()), desc="Checking levels", unit="stk"):
    try:
        for r in analyze_stock(sym, daily_map[sym]):
            r["Ticker"] = sym
            results.append(r)
    except Exception: pass

LISTS = ["Resistance break", "Break holding", "On support"]
results.sort(key=lambda x: (LISTS.index(x["List"]), -x["Score"]))
cnt = {k: sum(1 for x in results if x["List"] == k) for k in LISTS}
print(f"\n{'=' * 65}\n  SCAN COMPLETE - {got_daily} tickers with data, {FUNNEL['tickers_checked']} checked")
for k in LISTS: print(f"  {k:<18}: {cnt[k]}")
for k in LISTS:
    names = [x["Ticker"] for x in results if x["List"] == k]
    if names: print(f"\n  {k}: {', '.join(names[:80])}")
print("=" * 65)

# -- Save CSV + TradingView text list
ts      = datetime.today().strftime("%Y%m%d_%H%M")
out_dir = os.environ.get("GITHUB_WORKSPACE", os.getcwd())
COLS = ["Ticker","List","Signal","Score","Price","Entry_Price","Stop_Loss","Risk_%","Target_1","Reward_R",
        "Resistance","Support","Dist_%","Vol_Osc","Above_SMA200","Signal_Date"]
df_out = pd.DataFrame([{k: v for k, v in r.items() if not k.startswith("_")} for r in results]) if results else pd.DataFrame(columns=COLS)
df_out = df_out[[c for c in COLS if c in df_out.columns]].reset_index(drop=True)
fpath = os.path.join(out_dir, f"luxalgo_sr_{ts}.csv")
try:
    from position_sizing import add_position_size
    df_out = add_position_size(df_out)
    try: __import__("best_picks").rank_best(df_out)
    except Exception as _bp_err: print(f"  Best 10 skipped: {_bp_err}")
except Exception as _ps_err:
    print(f"  Position sizing skipped: {_ps_err}")
df_out.to_csv(fpath, index=False)
print(f"\n  CSV -> {fpath}")
tv = os.path.join(out_dir, f"tv_luxalgo_sr_{ts}.txt")
with open(tv, "w") as f:
    for k in LISTS:
        names = [x["Ticker"] for x in results if x["List"] == k]
        if names:
            f.write(f"###{k}\n"); f.writelines(f"NASDAQ:{s}\n" for s in names)
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
    symbols, seen = [], set()
    for k in LISTS:
        syms = [tvs.tv_symbol(x["Ticker"], ex) for x in results if x["List"] == k]
        new = [x for x in syms if x not in seen]
        if new: symbols.append(f"###{k} ({len(new)})"); symbols += new; seen.update(new)
    tvs.replace_list(s, lists, symbols)
    after = next((x for x in tvs.get_lists(s) if x.get("name") == name), None)
    got = [x for x in (after or {}).get("symbols") or [] if not str(x).startswith("###")]
    TV_STATUS = f"'{name}' - {len(got)} symbols"
try:
    _tv_sync()
except Exception as e:
    TV_STATUS = f"failed ({type(e).__name__}: {e})"
print(f"[TradingView] {TV_STATUS}")

# -- Email
EMAIL_COLS = ["Ticker","Signal","Score","Price","Entry_Price","Stop_Loss","Risk_%","Target_1","Reward_R",
              "Resistance","Support","Dist_%","Vol_Osc","Above_SMA200"]

def _table(rows):
    th = "".join(f'<th style="background:#1e293b;color:#e2e8f0;padding:6px 8px;font-size:11px;white-space:nowrap">{c}</th>' for c in EMAIL_COLS)
    body = ""
    for i, r in enumerate(rows):
        bg = "#fff" if i % 2 == 0 else "#f0f9ff"
        body += f'<tr style="background:{bg}">' + "".join(
            f'<td style="padding:5px 8px;font-size:12px;{"font-weight:700" if c in ("Ticker", "Signal") else ""}">{r.get(c, "-")}</td>'
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
        titles = {"Resistance break": "Crossed above the last resistance today",
                  "Break holding": f"Crossed above it 2-{CFG['hold_lookback']} sessions ago and still holding",
                  "On support": "On the last support today"}
        secs = ""
        for k in LISTS:
            rows = [x for x in rl if x["List"] == k]
            secs += f'<h3 style="font-size:14px;margin:12px 0 6px">{titles[k]} ({len(rows)})</h3>'
            secs += _table(rows[:80]) if rows else '<p style="color:#94a3b8;font-size:12px">none</p>'
        html = f"""<!DOCTYPE html><html><body style="margin:0;background:#f1f5f9;font-family:'Segoe UI',Arial,sans-serif">
<table width="100%" cellpadding="0" cellspacing="0"><tr><td align="center" style="padding:20px 10px">
<table width="100%" style="max-width:960px;background:#fff;border-radius:12px;overflow:hidden">
<tr><td style="background:linear-gradient(135deg,#0f172a,#1e3a5f);padding:20px 26px">
<h1 style="margin:0;color:#60a5fa;font-size:19px">LuxAlgo Support &amp; Resistance - Breaks and Support Tests</h1>
<p style="margin:6px 0 0;color:#94a3b8;font-size:12px">{datetime.today().strftime('%Y-%m-%d %H:%M UTC')} &middot; {cnt['Resistance break']} breaks,
{cnt['Break holding']} holding, {cnt['On support']} on support &middot; TradingView: {TV_STATUS}</p></td></tr>
<tr><td style="padding:16px 22px">{secs}
<p style="font-size:11px;color:#64748b">Levels as LuxAlgo "Support and Resistance Levels with Breaks": resistance / support = last pivot high / low with
{CFG['left_bars']} bars each side (shown once confirmed). B = close crossed above resistance with volume oscillator
(EMA5 vs EMA10 of volume) &gt; {CFG['volume_thresh']}%; Bull Wick = the crossing candle's lower wick is longer than its body.
On support = today's low within {CFG['support_touch_atr']:g} ATR of support (or below it) with the close at or above support.
Breaks: stop = breakout low (or resistance - {CFG['stop_buffer_atr']:g} ATR), target 2R. Support: stop = support - {CFG['stop_buffer_atr']:g} ATR,
target = the last resistance. Full results (with Quantity for a fixed max loss) attached.</p></td></tr>
<tr><td style="background:#f8fafc;padding:10px;text-align:center;color:#94a3b8;font-size:10px">Not financial advice &middot; Auto-generated by GitHub Actions</td></tr>
</table></td></tr></table></body></html>"""
        plain = [f"LuxAlgo S/R - {datetime.today().strftime('%Y-%m-%d')}", f"TradingView: {TV_STATUS}", ""]
        for k in LISTS:
            plain.append(f"{k} ({cnt[k]}): " + ", ".join(x["Ticker"] for x in rl if x["List"] == k))
        subj = (f"LuxAlgo S/R - {cnt['Resistance break']} breaks, {cnt['On support']} on support - "
                f"{datetime.today().strftime('%Y-%m-%d')}")
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
