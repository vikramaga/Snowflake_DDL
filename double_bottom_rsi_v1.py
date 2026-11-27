# ============================================================
# NASDAQ / NYSE - Double Bottom + RSI Bullish Divergence, neckline breakout - daily, v1
# ============================================================
#
# From the "Double Bottom + RSI" chart (2026-11-27). Signal = the latest candle is the entry:
#   1. LOW 1 and LOW 2: two swing lows (lowest low of 3 bars each side), 8-60 bars apart, LOW 2
#      3-30 bars before today, at about the same level (within 0.75 ATR(14) or 3%); nothing in
#      between or since traded below the lower of the two.
#   2. NECKLINE = the highest high between the two lows, at least 1.5 ATR above them.
#   3. Prior decline: the highest high of the 60 bars before LOW 1 (= TARGET) is at least 10% and
#      3 ATR above LOW 1 and at least half the pattern height above the neckline.
#   4. RSI(14) bullish divergence: RSI at LOW 1 <= 35, RSI at LOW 2 at least 2 points higher.
#   5. ENTRY: today is a green candle and the first close above the neckline since LOW 2.
#   6. Room: (TARGET - entry) >= 1.5 x risk. "RR >= 3" (as in the chart) listed first.
# Entry = today's close | Stop = lower of the two lows - 0.1 ATR | T1 = the prior high (TARGET) |
# T2 = 2R. Liquidity $5 / 200k. TradingView list "Double Bottom RSI - YYYY-MM-DD" (IST).
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
    "min_bars"         : 150,
    "pivot_k"          : 3,       # swing low = lowest low of 3 bars each side
    "lookback"         : 120,     # bars searched for LOW 1
    "low2_max_age"     : 30,      # LOW 2 at most 30 bars before today (and at least 3, to be confirmed)
    "gap_min"          : 8,       # bars between LOW 1 and LOW 2
    "gap_max"          : 60,
    "low_match_atr"    : 0.75,    # LOW 2 within 0.75 ATR (or 3%) of LOW 1
    "low_match_pct"    : 3.0,
    "min_height_atr"   : 1.5,     # neckline - lows >= 1.5 ATR
    "decline_min_pct"  : 10.0,    # prior high -> LOW 1 at least -10% ...
    "decline_min_atr"  : 3.0,     # ... and at least 3 ATR
    "prior_high_lookback": 60,    # bars before LOW 1 searched for the prior high (TARGET)
    "rsi_low1_max"     : 35.0,    # RSI(14) at LOW 1 near/below 30
    "rsi_div_min"      : 2.0,     # RSI at LOW 2 at least 2 points higher (bullish divergence)
    "min_rr"           : 1.5,     # target / risk at entry
    "rsi_period"       : 14,
    "atr_period"       : 14,
    "stop_buffer_atr"  : 0.1,
    "min_price"        : 5.0,
    "min_avg_volume"   : 200_000,
    "tv_list_prefix"   : "Double Bottom RSI",
    "batch_size"       : 50,
    "batch_sleep"      : 1.5,
}

STAGES = ["tickers_checked", "low_pairs", "equal_lows", "prior_decline", "rsi_divergence", "neckline_break", "room_to_target"]
FUNNEL = {k: 0 for k in STAGES}

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
    pc = np.concatenate([[np.nan], c[:-1]])
    tr = np.nanmax(np.vstack([h - l, np.abs(h - pc), np.abs(l - pc)]), axis=0)
    atr = pd.Series(tr).rolling(CFG["atr_period"]).mean().values
    if not np.isfinite(atr[i]): return None
    FUNNEL["tickers_checked"] += 1
    # cheap pre-check: today must be a green close above the prior bar's high region; neckline test below
    if not (c[i] > o[i]): return None
    rsi = wilder_rsi(df["Close"].astype(float), CFG["rsi_period"])
    k = CFG["pivot_k"]; lo0 = max(k, i - CFG["lookback"])
    piv = [j for j in range(lo0, i - k + 1) if l[j] == l[j - k:j + k + 1].min()]
    best, stage = None, 0
    for j2 in reversed([p for p in piv if i - CFG["low2_max_age"] <= p <= i - 3]):
        for j1 in reversed([p for p in piv if CFG["gap_min"] <= j2 - p <= CFG["gap_max"]]):
            st = 1
            a = atr[j2]
            if not (abs(l[j2] - l[j1]) <= max(CFG["low_match_atr"] * a, CFG["low_match_pct"] / 100 * l[j1])): stage = max(stage, st); continue
            neck_j = j1 + int(np.argmax(h[j1:j2 + 1])); neck = h[neck_j]
            base = min(l[j1], l[j2]); height = neck - base
            if not (height >= CFG["min_height_atr"] * a and l[j1:j2 + 1].min() >= base and l[j2:i + 1].min() >= l[j2]): stage = max(stage, st); continue
            st = 2
            s = max(0, j1 - CFG["prior_high_lookback"]); tj = s + int(np.argmax(h[s:j1 + 1])); target = h[tj]
            if not (target >= neck + 0.5 * height and (target / l[j1] - 1) * 100 >= CFG["decline_min_pct"]
                    and target - l[j1] >= CFG["decline_min_atr"] * atr[j1]): stage = max(stage, st); continue
            st = 3
            if not (rsi[j1] <= CFG["rsi_low1_max"] and rsi[j2] >= rsi[j1] + CFG["rsi_div_min"]): stage = max(stage, st); continue
            st = 4
            # entry: today's close is the FIRST close above the neckline since LOW 2
            if not (c[i] > neck and (c[j2:i] <= neck).all()): stage = max(stage, st); continue
            st = 5
            stop = base - CFG["stop_buffer_atr"] * atr[i]; risk = c[i] - stop
            rr = (target - c[i]) / risk if risk > 0 else 0
            if rr < CFG["min_rr"]: stage = max(stage, st); continue
            best = (j1, j2, neck_j, tj, neck, base, target, stop, rr); stage = 6; break
        if best: break
    for n in range(1, stage + 1):
        FUNNEL[STAGES[n]] += 1
    if not best: return None
    j1, j2, neck_j, tj, neck, base, target, stop, rr = best
    entry = float(c[i]); risk = entry - stop
    vavg = float(np.mean(v[i - 20:i])); vx = v[i] / vavg if vavg > 0 else np.nan
    div = rsi[j2] - rsi[j1]
    score = 50 + min(20, rr * 5) + min(10, div) + (10 if rsi[j1] <= 30 else 0) \
            + (min(10, (vx - 1) * 10) if np.isfinite(vx) and vx > 1 else 0) + (5 if l[j2] <= l[j1] else 0)
    flags = [f"LOW1 {l[j1]:.2f} ({df.index[j1]:%d %b}, RSI {rsi[j1]:.0f}) / LOW2 {l[j2]:.2f} ({df.index[j2]:%d %b}, RSI {rsi[j2]:.0f})",
             f"neckline {neck:.2f} broken today", f"target = prior high {target:.2f} ({df.index[tj]:%d %b})",
             f"breakout volume x{vx:.1f}" if np.isfinite(vx) else "volume n/a",
             "LOW2 undercut LOW1" if l[j2] < l[j1] else "LOW2 held above LOW1"]
    return {"Score": round(min(100, score)), "RR_Group": "RR >= 3" if rr >= 3 else "RR 1.5-3",
            "Price": round(entry, 2), "Entry_Price": round(entry, 2), "Stop_Loss": round(stop, 2),
            "Risk_%": round(risk / entry * 100, 2), "Target_1": round(target, 2), "Reward_R": round(rr, 2),
            "Target_2R": round(entry + 2 * risk, 2), "Neckline": round(neck, 2),
            "Low1": round(l[j1], 2), "Low1_Date": df.index[j1].strftime("%Y-%m-%d"), "RSI_Low1": round(rsi[j1], 1),
            "Low2": round(l[j2], 2), "Low2_Date": df.index[j2].strftime("%Y-%m-%d"), "RSI_Low2": round(rsi[j2], 1),
            "Breakout_Vol_x": round(vx, 2) if np.isfinite(vx) else None,
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
GROUPS = [("RR >= 3 (as in the chart)", lambda r: r["Reward_R"] >= 3), ("RR 1.5-3", lambda r: True)]
for r in results:
    r["Group"] = next(g for g, f in GROUPS if f(r))
fc = FUNNEL
print(f"\n{'=' * 65}\n  SCAN COMPLETE - {len(results)} matches from {got_daily} tickers with data")
print(f"  checked {fc['tickers_checked']} -> low pairs {fc['low_pairs']} -> equal lows {fc['equal_lows']} -> prior decline {fc['prior_decline']}"
      f" -> RSI divergence {fc['rsi_divergence']} -> neckline break today {fc['neckline_break']} -> room to target {fc['room_to_target']}")
for g, _ in GROUPS:
    names = [r["Ticker"] for r in results if r["Group"] == g]
    print(f"  {g}: {len(names)}  {', '.join(names[:60])}")
print("=" * 65)

ts      = datetime.today().strftime("%Y%m%d_%H%M")
out_dir = os.environ.get("GITHUB_WORKSPACE", os.getcwd())
COLS = ["Ticker","Group","Score","Price","Entry_Price","Stop_Loss","Risk_%","Target_1","Reward_R","Target_2R","Neckline",
        "Low1","Low1_Date","RSI_Low1","Low2","Low2_Date","RSI_Low2","Breakout_Vol_x","Signal_Date","Flags"]
df_out = pd.DataFrame(results) if results else pd.DataFrame(columns=COLS)
df_out = df_out[[c for c in COLS if c in df_out.columns]].reset_index(drop=True)
fpath = os.path.join(out_dir, f"double_bottom_rsi_{ts}.csv")
try:
    from position_sizing import add_position_size
    df_out = add_position_size(df_out)
    try: __import__("best_picks").rank_best(df_out)
    except Exception as _bp_err: print(f"  Best 10 skipped: {_bp_err}")
except Exception as _ps_err:
    print(f"  Position sizing skipped: {_ps_err}")
df_out.to_csv(fpath, index=False)
print(f"\n  CSV -> {fpath}")
tv = os.path.join(out_dir, f"tv_double_bottom_rsi_{ts}.txt")
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

EMAIL_COLS = ["Ticker","Score","Price","Stop_Loss","Risk_%","Target_1","Reward_R","Neckline","Low1","RSI_Low1","Low2","RSI_Low2",
              "Breakout_Vol_x"]
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
<h1 style="margin:0;color:#60a5fa;font-size:19px">Double Bottom + RSI Divergence &rarr; Neckline Breakout</h1>
<p style="margin:6px 0 0;color:#94a3b8;font-size:12px">{datetime.today().strftime('%Y-%m-%d %H:%M UTC')} &middot; {len(rl)} matches
&middot; TradingView: {TV_STATUS}</p></td></tr>
<tr><td style="padding:16px 22px">{secs}
<p style="font-size:11px;color:#64748b">Two swing lows at about the same level after a 10%+ decline, RSI(14) at LOW 1 &lt;= 35 and higher at LOW 2 (bullish divergence),
today = first green close above the neckline (the high between the lows). Target = the prior high where the decline started.
Entry = today's close &middot; Stop = the lower of the two lows - {CFG['stop_buffer_atr']:g} ATR &middot; T1 = prior high &middot; T2 = 2R.
Funnel: checked {FUNNEL['tickers_checked']} &rarr; low pairs {FUNNEL['low_pairs']} &rarr; equal lows {FUNNEL['equal_lows']} &rarr; prior decline {FUNNEL['prior_decline']}
&rarr; RSI divergence {FUNNEL['rsi_divergence']} &rarr; neckline break {FUNNEL['neckline_break']} &rarr; room to target {FUNNEL['room_to_target']}.
Full results (with Quantity for a fixed max loss) attached.</p></td></tr>
<tr><td style="background:#f8fafc;padding:10px;text-align:center;color:#94a3b8;font-size:10px">Not financial advice &middot; Auto-generated by GitHub Actions</td></tr>
</table></td></tr></table></body></html>"""
        plain = [f"Double Bottom + RSI - {datetime.today().strftime('%Y-%m-%d')}", f"{len(rl)} matches", f"TradingView: {TV_STATUS}", ""]
        for g, _ in GROUPS:
            plain.append(f"{g}: " + ", ".join(r["Ticker"] for r in rl if r["Group"] == g))
        subj = f"Double Bottom + RSI Divergence Breakout - {len(rl)} signals - {datetime.today().strftime('%Y-%m-%d')}"
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
