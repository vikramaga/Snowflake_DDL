# ============================================================
# NASDAQ / NYSE - RSI 30 -> 50 Recovery + SMA50 Reclaim on Steady Buying - daily, v1
# ============================================================
#
# Vik's idea (2026-11-18): RSI was oversold (< 30) about 2 months ago, has now
# crossed back above 50, buying volume has been steadily larger than selling
# volume, and price has crossed above SMA50 and is holding / going higher.
#
# Rules on the latest daily candle (all required):
#   1. OVERSOLD 2 MONTHS AGO: RSI(14) below 30 on at least one day 25-60 sessions ago.
#   2. RSI ABOVE 50: RSI crossed above 50 within the last 10 sessions (a day with
#      RSI <= 50 followed by > 50) and is above 50 today.
#   3. BUYING > SELLING (last 20 sessions): volume on up-close days >= 1.2x volume on
#      down-close days, at least as many up days as down days, and no single day
#      carries more than 35% of the up-day volume (steady buying, not one spike).
#   4. SMA50 RECLAIM: a close crossed above SMA50 within the last 15 sessions; today's
#      close above SMA50 and above that cross-day close; none of the last 3 closes
#      below SMA50.
#   5. GOING GOOD: close above EMA21, EMA21 rising over 5 sessions, and close no more
#      than 15% above SMA50.
# Groups: "Fresh" = SMA50 cross in the last 5 sessions, "Holding" = 6-15 sessions ago.
# Entry = today's close | Stop = lowest low since the SMA50 cross, but not below
# SMA50 - 1 ATR(14) | T1 = the highest high of the 120 sessions before the
# oversold low if above the entry, else 2R | T2 = 2R. Liquidity $5 / 200k.
# TradingView list "RSI Recovery - YYYY-MM-DD" (IST). Settings in CFG.
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
    "min_bars"         : 200,
    "rsi_period"       : 14,
    "oversold"         : 30,
    "oversold_from"    : 25,      # RSI < 30 between 25 ...
    "oversold_to"      : 60,      # ... and 60 sessions ago
    "rsi_mid"          : 50,
    "rsi_cross_within" : 10,
    "vol_window"       : 20,
    "buy_sell_ratio"   : 1.2,
    "max_day_share"    : 0.35,    # one up day <= 35% of all up-day volume
    "sma_cross_within" : 15,
    "fresh_sessions"   : 5,
    "max_above_sma50"  : 15.0,
    "atr_period"       : 14,
    "min_price"        : 5.0,
    "min_avg_volume"   : 200_000,
    "tv_list_prefix"   : "RSI Recovery",
    "batch_size"       : 50,
    "batch_sleep"      : 1.5,
}

FUNNEL = {"tickers_checked": 0, "oversold_2m": 0, "rsi_above_50": 0, "buying_volume": 0,
          "sma50_reclaim": 0, "going_good": 0}

def wilder_rsi(close, n=14):
    d = close.diff()
    up = d.clip(lower=0).ewm(alpha=1 / n, adjust=False).mean()
    dn = (-d.clip(upper=0)).ewm(alpha=1 / n, adjust=False).mean()
    rs = up / dn.replace(0, np.nan)
    return (100 - 100 / (1 + rs)).fillna(100).values

def analyze_stock(sym, df):
    if df is None or len(df) < CFG["min_bars"]: return None
    o, h, l, c, v = (df[k].values.astype(float) for k in ("Open", "High", "Low", "Close", "Volume"))
    i = len(df) - 1
    if c[i] < CFG["min_price"] or float(np.mean(v[-20:])) < CFG["min_avg_volume"]: return None
    cs = df["Close"].astype(float)
    rsi = wilder_rsi(cs, CFG["rsi_period"])
    s50 = cs.rolling(50).mean().values
    e21 = cs.ewm(span=21, adjust=False).mean().values
    if not np.isfinite(s50[i - CFG["sma_cross_within"] - 1]): return None
    FUNNEL["tickers_checked"] += 1

    # 1. oversold 25-60 sessions ago
    a, b = i - CFG["oversold_to"], i - CFG["oversold_from"]
    win = rsi[a:b + 1]
    if not (win.min() < CFG["oversold"]): return None
    os_idx = a + int(np.argmin(win)); os_rsi = float(rsi[os_idx])
    FUNNEL["oversold_2m"] += 1

    # 2. RSI crossed above 50 recently, above 50 now
    if not rsi[i] > CFG["rsi_mid"]: return None
    rc = [j for j in range(i - CFG["rsi_cross_within"] + 1, i + 1) if rsi[j - 1] <= CFG["rsi_mid"] < rsi[j]]
    if not rc: return None
    rsi_cross = rc[-1]
    FUNNEL["rsi_above_50"] += 1

    # 3. buying volume vs selling volume
    w = CFG["vol_window"]; pc = c[i - w:i]; cc = c[i - w + 1:i + 1]; vv = v[i - w + 1:i + 1]
    upm = cc > pc; dnm = cc < pc
    up_vol, dn_vol = float(vv[upm].sum()), float(vv[dnm].sum())
    ratio = up_vol / dn_vol if dn_vol > 0 else 99.0
    share = float(vv[upm].max() / up_vol) if up_vol > 0 else 1.0
    if not (ratio >= CFG["buy_sell_ratio"] and upm.sum() >= dnm.sum() and share <= CFG["max_day_share"]): return None
    FUNNEL["buying_volume"] += 1

    # 4. SMA50 reclaim
    xs = [j for j in range(i - CFG["sma_cross_within"] + 1, i + 1) if c[j - 1] <= s50[j - 1] and c[j] > s50[j]]
    if not xs: return None
    x = xs[-1]
    if not (c[i] > s50[i] and (x == i or c[i] > c[x])): return None
    if any(c[j] <= s50[j] for j in range(i - 2, i + 1)): return None
    FUNNEL["sma50_reclaim"] += 1

    # 5. going good
    above50 = (c[i] / s50[i] - 1) * 100
    if not (c[i] > e21[i] and e21[i] > e21[i - 5] and above50 <= CFG["max_above_sma50"]): return None
    FUNNEL["going_good"] += 1

    prev = np.concatenate([[np.nan], c[:-1]])
    tr = np.nanmax(np.vstack([h - l, np.abs(h - prev), np.abs(l - prev)]), axis=0)
    atr = float(pd.Series(tr).rolling(CFG["atr_period"]).mean().values[i])
    stop = max(float(l[x:i + 1].min()), float(s50[i] - atr))
    entry = float(c[i]); risk = entry - stop
    if risk <= 0: return None
    res = float(np.max(h[max(0, os_idx - 120):os_idx])) if os_idx > 0 else np.nan
    t1, t1_src = (res, "pre-selloff high") if np.isfinite(res) and res > entry else (entry + 2 * risk, "2R")
    days_x = i - x
    s50_up = s50[i] > s50[i - 5]
    vavg = float(np.mean(v[i - 50:i])); vx_up = float(vv[upm].mean()) / vavg if vavg > 0 else np.nan
    score = 40 + min(20, (ratio - 1) * 20) + (10 if days_x <= CFG["fresh_sessions"] else 5) \
            + (10 if 55 <= rsi[i] <= 70 else 5 if rsi[i] < 55 else 0) + min(10, (CFG["oversold"] - os_rsi) * 1.0) \
            + (5 if s50_up else 0) + max(0, 5 - above50 / 2)
    flags = [f"RSI low {os_rsi:.1f} on {df.index[os_idx]:%Y-%m-%d}", f"RSI crossed 50 on {df.index[rsi_cross]:%Y-%m-%d}, now {rsi[i]:.1f}",
             f"buy/sell volume x{ratio:.2f} ({int(upm.sum())} up / {int(dnm.sum())} down days)",
             f"largest up day {share:.0%} of up volume", f"SMA50 cross {days_x} session(s) ago",
             "SMA50 turning up" if s50_up else "SMA50 still falling"]
    return {"Score": round(min(100, score)), "Stage": "Fresh" if days_x <= CFG["fresh_sessions"] else "Holding",
            "Price": round(entry, 2), "Entry_Price": round(entry, 2), "Stop_Loss": round(stop, 2),
            "Risk_%": round(risk / entry * 100, 2), "Target_1": round(t1, 2), "T1_Source": t1_src,
            "Reward_R": round((t1 - entry) / risk, 2), "Target_2R": round(entry + 2 * risk, 2),
            "RSI_Now": round(float(rsi[i]), 1), "RSI_Low": round(os_rsi, 1), "RSI_Low_Date": df.index[os_idx].strftime("%Y-%m-%d"),
            "RSI50_Cross_Date": df.index[rsi_cross].strftime("%Y-%m-%d"), "Buy_Sell_Vol_Ratio": round(ratio, 2),
            "Up_Days": int(upm.sum()), "Down_Days": int(dnm.sum()), "Max_Day_Share_%": round(share * 100, 1),
            "Up_Day_Vol_x_50d": round(vx_up, 2) if np.isfinite(vx_up) else None,
            "SMA50_Cross_Date": df.index[x].strftime("%Y-%m-%d"), "Days_Since_Cross": days_x,
            "Above_SMA50_%": round(above50, 1), "SMA50": round(float(s50[i]), 2), "EMA21": round(float(e21[i]), 2),
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
GROUPS = [("Fresh - SMA50 cross in the last 5 sessions", lambda r: r["Stage"] == "Fresh"),
          ("Holding - SMA50 cross 6-15 sessions ago", lambda r: True)]
for r in results:
    r["Group"] = next(g for g, f in GROUPS if f(r))
fc = FUNNEL
print(f"\n{'=' * 65}\n  SCAN COMPLETE - {len(results)} matches from {got_daily} tickers with data")
print(f"  checked {fc['tickers_checked']} -> RSI < 30 2 months ago {fc['oversold_2m']} -> RSI crossed 50 {fc['rsi_above_50']}"
      f" -> buying > selling {fc['buying_volume']} -> SMA50 reclaim {fc['sma50_reclaim']} -> going good {fc['going_good']}")
for g, _ in GROUPS:
    names = [r["Ticker"] for r in results if r["Group"] == g]
    print(f"  {g}: {len(names)}  {', '.join(names[:60])}")
print("=" * 65)

ts      = datetime.today().strftime("%Y%m%d_%H%M")
out_dir = os.environ.get("GITHUB_WORKSPACE", os.getcwd())
COLS = ["Ticker","Stage","Group","Score","Price","Entry_Price","Stop_Loss","Risk_%","Target_1","T1_Source","Reward_R","Target_2R",
        "RSI_Now","RSI_Low","RSI_Low_Date","RSI50_Cross_Date","Buy_Sell_Vol_Ratio","Up_Days","Down_Days","Max_Day_Share_%",
        "Up_Day_Vol_x_50d","SMA50_Cross_Date","Days_Since_Cross","Above_SMA50_%","SMA50","EMA21","Signal_Date","Flags"]
df_out = pd.DataFrame(results) if results else pd.DataFrame(columns=COLS)
df_out = df_out[[c for c in COLS if c in df_out.columns]].reset_index(drop=True)
fpath = os.path.join(out_dir, f"rsi30_recovery_sma50_{ts}.csv")
try:
    from position_sizing import add_position_size
    df_out = add_position_size(df_out)
    try: __import__("best_picks").rank_best(df_out)
    except Exception as _bp_err: print(f"  Best 10 skipped: {_bp_err}")
except Exception as _ps_err:
    print(f"  Position sizing skipped: {_ps_err}")
df_out.to_csv(fpath, index=False)
print(f"\n  CSV -> {fpath}")
tv = os.path.join(out_dir, f"tv_rsi30_recovery_sma50_{ts}.txt")
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

EMAIL_COLS = ["Ticker","Score","Price","Stop_Loss","Risk_%","Target_1","Reward_R","RSI_Now","RSI_Low","RSI_Low_Date",
              "Buy_Sell_Vol_Ratio","Up_Days","Down_Days","Days_Since_Cross","Above_SMA50_%"]
def _table(rows):
    th = "".join(f'<th style="background:#1e293b;color:#e2e8f0;padding:6px 8px;font-size:11px;white-space:nowrap">{c}</th>' for c in EMAIL_COLS)
    body = ""
    for i, r in enumerate(rows):
        bg = "#fff" if i % 2 == 0 else "#f0f9ff"
        body += f'<tr style="background:{bg}">' + "".join(
            f'<td style="padding:5px 8px;font-size:12px;{"font-weight:700" if c in ("Ticker", "Buy_Sell_Vol_Ratio") else ""}">{r.get(c, "-")}</td>'
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
<h1 style="margin:0;color:#60a5fa;font-size:19px">RSI 30 &rarr; 50 Recovery + SMA50 Reclaim</h1>
<p style="margin:6px 0 0;color:#94a3b8;font-size:12px">{datetime.today().strftime('%Y-%m-%d %H:%M UTC')} &middot; {len(rl)} matches
&middot; TradingView: {TV_STATUS}</p></td></tr>
<tr><td style="padding:16px 22px">{secs}
<p style="font-size:11px;color:#64748b">RSI(14) below {CFG['oversold']} on a day {CFG['oversold_from']}-{CFG['oversold_to']} sessions ago &middot;
RSI crossed above {CFG['rsi_mid']} in the last {CFG['rsi_cross_within']} sessions and is above it today &middot;
last {CFG['vol_window']} sessions: up-day volume &ge; {CFG['buy_sell_ratio']:g}&times; down-day volume, at least as many up days as down days,
no single up day over {CFG['max_day_share']:.0%} of the up-day volume &middot; close crossed above SMA50 in the last {CFG['sma_cross_within']} sessions,
now above it and above the cross-day close, last 3 closes above SMA50 &middot; close above a rising EMA21, at most {CFG['max_above_sma50']:g}% above SMA50.
Entry = today's close &middot; Stop = lowest low since the SMA50 cross (not below SMA50 - 1 ATR) &middot; T1 = the high of the 120 sessions before the
oversold low (2R if already above) &middot; T2 = 2R. Funnel: checked {FUNNEL['tickers_checked']} &rarr; oversold 2 months ago {FUNNEL['oversold_2m']}
&rarr; RSI above 50 {FUNNEL['rsi_above_50']} &rarr; buying &gt; selling {FUNNEL['buying_volume']} &rarr; SMA50 reclaim {FUNNEL['sma50_reclaim']}
&rarr; going good {FUNNEL['going_good']}. Full results (with Quantity for a fixed max loss) attached.</p></td></tr>
<tr><td style="background:#f8fafc;padding:10px;text-align:center;color:#94a3b8;font-size:10px">Not financial advice &middot; Auto-generated by GitHub Actions</td></tr>
</table></td></tr></table></body></html>"""
        plain = [f"RSI 30 -> 50 Recovery + SMA50 Reclaim - {datetime.today().strftime('%Y-%m-%d')}", f"{len(rl)} matches", f"TradingView: {TV_STATUS}", ""]
        for g, _ in GROUPS:
            plain.append(f"{g}: " + ", ".join(r["Ticker"] for r in rl if r["Group"] == g))
        nF = sum(1 for r in rl if r["Stage"] == "Fresh")
        subj = f"RSI 30->50 Recovery + SMA50 - {len(rl)} signals, {nF} fresh - {datetime.today().strftime('%Y-%m-%d')}"
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
