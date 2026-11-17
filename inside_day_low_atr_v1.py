# ============================================================
# NASDAQ / NYSE - Quiet Inside Day (low ATR) - daily, v1
# ============================================================
#
# Built from the pattern mining of 2026-11-17 (claude/pattern-mining-2026-11.md):
# across 27,629 scanner picks (9 Oct - 9 Nov 2026) an INSIDE DAY on a LOW-VOLATILITY
# stock was the pattern most often followed by a rise:
#   inside day + ATR < 2.5% + lagging SPY (3m):  76% up after 5 days (avg +1.6%), 67% after 10
#   inside day + NR7 + ATR < 2.5%:               76% up after 5 days (avg +1.4%)
#   inside day + ATR < 2.5% + price > $50:       74% up after 5 days (avg +1.3%)
# (base rate of all picks: 52.5%). Five weeks of data - unproven out of sample.
#
# Rules on the latest daily candle:
#   1. INSIDE DAY: today's high < yesterday's high AND today's low > yesterday's low
#   2. LOW VOLATILITY: ATR(14) < 2.5% of the close
#   3. AVOID (weak in the mining): close > 15% above SMA50, price < $10
# Tier (ranked by how many of the mined extras it has):
#   A = lagging SPY over 3 months (the best combination)   B = NR7 (narrowest range of 7)
#   C = price > $50     -> Tier "A+B+C", "A+B", "A", "B+C", ... or "Base"
# Entry = today's close; Stop = yesterday's (mother bar) low - 0.1 ATR;
# Exit = close 5 sessions later (Exit_5d_Date), or 10 sessions (the mining's horizons);
# Target_2R shown for reference. Liquidity $10 / 200k. TradingView list
# "Inside Day - YYYY-MM-DD" (IST). Settings in CFG.
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
    "min_bars"         : 70,
    "atr_period"       : 14,
    "max_atr_pct"      : 2.5,     # ATR(14) < 2.5% of price
    "max_above_sma50"  : 15.0,    # skip if close > 15% above SMA50
    "rs_days"          : 63,      # 3-month relative strength vs SPY
    "nr_days"          : 7,       # NR7
    "tier_c_price"     : 50.0,
    "stop_buffer_atr"  : 0.1,
    "hold_days"        : (5, 10),
    "min_price"        : 10.0,
    "min_avg_volume"   : 200_000,
    "tv_list_prefix"   : "Inside Day",
    "batch_size"       : 50,
    "batch_sleep"      : 1.5,
}

FUNNEL = {"tickers_checked": 0, "inside_day": 0, "low_atr": 0, "not_extended": 0}

def analyze_stock(sym, df, spy):
    if df is None or len(df) < CFG["min_bars"]: return None
    o, h, l, c = (df[k].values.astype(float) for k in ("Open", "High", "Low", "Close"))
    i = len(df) - 1
    if c[i] < CFG["min_price"] or float(df["Volume"].tail(20).mean()) < CFG["min_avg_volume"]: return None
    pc = np.concatenate([[np.nan], c[:-1]])
    tr = np.nanmax(np.vstack([h - l, np.abs(h - pc), np.abs(l - pc)]), axis=0)
    atr = pd.Series(tr).rolling(CFG["atr_period"]).mean().values[i]
    s50 = df["Close"].rolling(50).mean().values[i]
    if not (np.isfinite(atr) and np.isfinite(s50)): return None
    FUNNEL["tickers_checked"] += 1
    inside = h[i] < h[i - 1] and l[i] > l[i - 1]
    atr_pct = atr / c[i] * 100
    above50 = (c[i] / s50 - 1) * 100
    if not inside: return None
    FUNNEL["inside_day"] += 1
    if atr_pct >= CFG["max_atr_pct"]: return None
    FUNNEL["low_atr"] += 1
    if above50 > CFG["max_above_sma50"]: return None
    FUNNEL["not_extended"] += 1

    rs = np.nan
    sp = spy.reindex(df.index).ffill().values
    k = CFG["rs_days"]
    if i >= k and np.isfinite(sp[i]) and np.isfinite(sp[i - k]) and sp[i - k] > 0:
        rs = (c[i] / c[i - k]) / (sp[i] / sp[i - k])
    rng = h - l
    nr = rng[i] <= rng[i - CFG["nr_days"] + 1:i + 1].min()
    tags = []
    if np.isfinite(rs) and rs < 1: tags.append("A")
    if nr: tags.append("B")
    if c[i] > CFG["tier_c_price"]: tags.append("C")
    tier = "+".join(tags) if tags else "Base"

    stop = l[i - 1] - CFG["stop_buffer_atr"] * atr
    risk = c[i] - stop
    if risk <= 0: return None
    bd = pd.offsets.BDay
    exits = {n: (df.index[i] + bd(n)).strftime("%Y-%m-%d") for n in CFG["hold_days"]}
    score = 40 + (25 if "A" in tags else 0) + (15 if "B" in tags else 0) + (10 if "C" in tags else 0) \
            + max(0, 10 - atr_pct * 4)
    flags = [f"ATR {atr_pct:.2f}%", f"{above50:+.1f}% vs SMA50",
             f"RS vs SPY (3m) {rs:.2f}" if np.isfinite(rs) else "RS n/a",
             "green" if c[i] > o[i] else "red" if c[i] < o[i] else "doji"]
    return {"Score": round(min(100, score)), "Tier": tier, "Price": round(c[i], 2),
            "Entry_Price": round(c[i], 2), "Stop_Loss": round(stop, 2), "Risk_%": round(risk / c[i] * 100, 2),
            "Target_2R": round(c[i] + 2 * risk, 2), "Exit_5d_Date": exits[5], "Exit_10d_Date": exits[10],
            "ATR_%": round(atr_pct, 2), "Above_SMA50_%": round(above50, 1),
            "RS_vs_SPY_3m": round(rs, 3) if np.isfinite(rs) else None, "NR7": "Yes" if nr else "No",
            "Mother_High": round(h[i - 1], 2), "Mother_Low": round(l[i - 1], 2),
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

# ── Ticker list ─────────────────────────────────────────────────
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

spy_map = download_daily(["SPY"], CFG["history_days"])
spy = spy_map["SPY"]["Close"].astype(float) if "SPY" in spy_map else pd.Series(dtype=float)
print(f"  SPY bars: {len(spy)}")

for sym in tqdm(list(daily_map.keys()), desc="Checking pattern", unit="stk"):
    try:
        r = analyze_stock(sym, daily_map[sym], spy)
        if r: r["Ticker"] = sym; results.append(r)
    except Exception: pass

results.sort(key=lambda x: -x["Score"])
GROUPS = [("Tier A - lags SPY (best mined combo)", lambda r: "A" in r["Tier"]),
          ("Tier B/C - NR7 or price > $50", lambda r: "A" not in r["Tier"] and r["Tier"] != "Base"),
          ("Base - inside day + low ATR only", lambda r: r["Tier"] == "Base")]
for r in results:
    r["Group"] = next(g for g, f in GROUPS if f(r))
fc = FUNNEL
print(f"\n{'=' * 65}\n  SCAN COMPLETE - {len(results)} matches from {got_daily} tickers with data")
print(f"  checked {fc['tickers_checked']} -> inside day {fc['inside_day']} -> ATR < {CFG['max_atr_pct']}% {fc['low_atr']}"
      f" -> not extended {fc['not_extended']}")
for g, _ in GROUPS:
    names = [r["Ticker"] for r in results if r["Group"] == g]
    print(f"  {g}: {len(names)}  {', '.join(names[:60])}")
print("=" * 65)

ts      = datetime.today().strftime("%Y%m%d_%H%M")
out_dir = os.environ.get("GITHUB_WORKSPACE", os.getcwd())
COLS = ["Ticker","Tier","Group","Score","Price","Entry_Price","Stop_Loss","Risk_%","Target_2R","Exit_5d_Date",
        "Exit_10d_Date","ATR_%","Above_SMA50_%","RS_vs_SPY_3m","NR7","Mother_High","Mother_Low","Signal_Date","Flags"]
df_out = pd.DataFrame(results) if results else pd.DataFrame(columns=COLS)
df_out = df_out[[c for c in COLS if c in df_out.columns]].reset_index(drop=True)
fpath = os.path.join(out_dir, f"inside_day_low_atr_{ts}.csv")
try:
    from position_sizing import add_position_size
    df_out = add_position_size(df_out)
    try: __import__("best_picks").rank_best(df_out)
    except Exception as _bp_err: print(f"  Best 10 skipped: {_bp_err}")
except Exception as _ps_err:
    print(f"  Position sizing skipped: {_ps_err}")
df_out.to_csv(fpath, index=False)
print(f"\n  CSV -> {fpath}")
tv = os.path.join(out_dir, f"tv_inside_day_low_atr_{ts}.txt")
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

EMAIL_COLS = ["Ticker","Tier","Score","Price","Stop_Loss","Risk_%","Exit_5d_Date","Exit_10d_Date","ATR_%",
              "Above_SMA50_%","RS_vs_SPY_3m","NR7"]
def _table(rows):
    th = "".join(f'<th style="background:#1e293b;color:#e2e8f0;padding:6px 8px;font-size:11px;white-space:nowrap">{c}</th>' for c in EMAIL_COLS)
    body = ""
    for i, r in enumerate(rows):
        bg = "#fff" if i % 2 == 0 else "#f0f9ff"
        body += f'<tr style="background:{bg}">' + "".join(
            f'<td style="padding:5px 8px;font-size:12px;{"font-weight:700" if c in ("Ticker", "Tier") else ""}">{r.get(c, "-")}</td>'
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
<h1 style="margin:0;color:#60a5fa;font-size:19px">Quiet Inside Day (ATR &lt; {CFG['max_atr_pct']:g}%)</h1>
<p style="margin:6px 0 0;color:#94a3b8;font-size:12px">{datetime.today().strftime('%Y-%m-%d %H:%M UTC')} &middot; {len(rl)} matches
&middot; TradingView: {TV_STATUS}</p></td></tr>
<tr><td style="padding:16px 22px">{secs}
<p style="font-size:11px;color:#64748b">Inside day (high and low inside yesterday's range) on a stock with ATR(14) under {CFG['max_atr_pct']:g}% of price,
not more than {CFG['max_above_sma50']:g}% above SMA50, price &ge; ${CFG['min_price']:g}. Tier A = lagging SPY over 3 months, B = NR7, C = price &gt; ${CFG['tier_c_price']:g}.
Mined 9 Oct - 9 Nov 2026: inside day + ATR &lt; 2.5% + lagging SPY rose after 5 sessions 76% of the time (avg +1.6%; 67% after 10),
vs 52.5% for all scanner picks - five weeks of data, not yet proven out of sample.
Entry = today's close &middot; Stop = yesterday's low - {CFG['stop_buffer_atr']:g} ATR &middot; Exit = close on Exit_5d_Date (or Exit_10d_Date).
Full results (with Quantity for a fixed max loss) attached.</p></td></tr>
<tr><td style="background:#f8fafc;padding:10px;text-align:center;color:#94a3b8;font-size:10px">Not financial advice &middot; Auto-generated by GitHub Actions</td></tr>
</table></td></tr></table></body></html>"""
        plain = [f"Quiet Inside Day - {datetime.today().strftime('%Y-%m-%d')}", f"{len(rl)} matches", f"TradingView: {TV_STATUS}", ""]
        for g, _ in GROUPS:
            plain.append(f"{g}: " + ", ".join(r["Ticker"] for r in rl if r["Group"] == g))
        nA = sum(1 for r in rl if "A" in r["Tier"])
        subj = f"Inside Day (low ATR) - {len(rl)} signals, {nA} Tier A - {datetime.today().strftime('%Y-%m-%d')}"
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
