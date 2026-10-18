# ============================================================
# NASDAQ — Liquidity Sweep + Rising SMA50 — 0.75R strategy (v1)
# ============================================================
#
# SIGNAL (all required). Call the last 3 daily candles A (3rd day
# back), B, C (current / most recent):
#
#   1. CANDLE A: GREEN (closed above its open).
#
#   2. CANDLE B — the SWEEP: its HIGH went ABOVE candle A's high
#      (took the liquidity resting above A)...
#
#   3. ...but it came back down and CLOSED BELOW candle A's close.
#
#   4. CANDLE C — the BREAKOUT: CLOSES ABOVE candle B's high...
#
#   5. ...and is GREEN (closed above its open).
#
#   6. SMA50 RISING: today's SMA50 is above the SMA50 of 20 trading days
#      ago (sma_rise_days).
#
#   7. REAL STOP: the stop (lower of candle B's and C's lows) is at least
#      1% below the entry (min_stop_pct) — tighter stops are noise.
#
# TRADE PLAN (from the 2-year research backtest, Oct 2024 – Oct 2026):
#   Entry  = candle C's close.   Stop = lower of candle B's and C's lows.
#   Target = entry + 0.75 x (entry - stop)   (target_r).
#   Exit   = target, stop, or the close of the 20th trading day (max_hold_days).
#   Backtest, 600-stock sample: year 1 352 trades, 62.2% wins, +0.14R/trade;
#   year 2 (unseen) 387 trades, 61.8% wins, +0.11R/trade. Breakeven win
#   rate at 0.75R is 57.1%. No costs/slippage; today's listings only.
#
# By default candle C must be the most recent daily candle
# (signal_lookback_days = 1).
#
# DATA — only 1 download per ticker: daily bars.
#
# OUTPUT: Entry_Price, Stop_Loss, Target_Price and Exit_By follow the trade
# plan above. Quantity / Total_Amount come from position_sizing.py
# (max loss $40 per trade).
#
# SINGLE PASS — purely technical, no fundamentals fetch.
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
    "history_days"          : 550,   # daily bars — ~380 trading days, comfortable
                                      # margin above _clean()'s 220-bar floor

    "sma_period"             : 50,
    "sma_rise_days"          : 20,   # step 6: SMA50 today > SMA50 this many days ago
    "min_stop_pct"           : 1.0,  # step 7: stop at least this % below the entry
    "target_r"               : 0.75, # target = entry + target_r x risk
    "max_hold_days"          : 20,   # exit at the close of this trading day

    "signal_lookback_days"   : 1,    # 1 = candle C must be the most recent
                                      # daily candle; raise to catch patterns
                                      # that completed a few days ago

    # ── Filters ─────────────────────────────────────────────────
    "min_avg_volume"         : 100_000,
    "min_price"              : 10.0,

    "batch_size"             : 50,
    "batch_sleep"            : 1.5,
}

# ── Pattern ──────────────────────────────────────────────────
def check_pattern_at(df, i, cfg=None):
    """
    Checks the liquidity-sweep breakout with candle C (the breakout
    day) anchored at bar `i`. Candle B is i-1, candle A is i-2. This
    is the SINGLE SOURCE OF TRUTH for the pattern logic.

    Computes EVERY stage's pass/fail regardless of earlier failures
    (guarding only on missing data), so callers can diagnose exactly
    which stage is filtering results out — see FUNNEL_COUNTS below.

    Returns (passed: bool, details: dict).
    """
    if i < 2 or i >= len(df):
        return False, {}

    oA, hA, lA, cA = (float(df[k].iloc[i-2]) for k in ("Open","High","Low","Close"))
    oB, hB, lB, cB = (float(df[k].iloc[i-1]) for k in ("Open","High","Low","Close"))
    oC, hC, lC, cC = (float(df[k].iloc[i])   for k in ("Open","High","Low","Close"))
    if any(np.isnan(v) for v in [oA, hA, cA, hB, lB, cB, oC, lC, cC]):
        return False, {}

    A_green        = bool(cA > oA)          # Step 1
    B_swept_high   = bool(hB > hA)          # Step 2
    B_closed_below = bool(cB < cA)          # Step 3
    C_above_B_high = bool(cC > hB)          # Step 4
    C_green        = bool(cC > oC)          # Step 5

    cfg = cfg or CFG
    closes = df["Close"].iloc[:i+1]
    n_sma, back = cfg["sma_period"], cfg["sma_rise_days"]
    if len(closes) >= n_sma + back:
        sma_now  = float(closes.iloc[-n_sma:].mean())
        sma_prev = float(closes.iloc[-(n_sma + back):-back].mean())
    else:
        sma_now = sma_prev = float("nan")
    stop     = min(lB, lC)
    risk_pct = (cC - stop) / cC * 100 if cC > 0 else 0.0
    SMA50_rising = bool(sma_now > sma_prev)                     # Step 6
    stop_ok      = bool(risk_pct >= cfg["min_stop_pct"])        # Step 7

    passed = (A_green and B_swept_high and B_closed_below and C_above_B_high and C_green
              and SMA50_rising and stop_ok)

    return passed, {
        "idx": i,
        "oA": oA, "hA": hA, "cA": cA,
        "oB": oB, "hB": hB, "lB": lB, "cB": cB,
        "oC": oC, "hC": hC, "lC": lC, "cC": cC,
        "A_green": A_green, "B_swept_high": B_swept_high,
        "B_closed_below": B_closed_below, "C_above_B_high": C_above_B_high,
        "C_green": C_green,
        "sma_now": sma_now, "sma_prev": sma_prev, "risk_pct": risk_pct,
        "SMA50_rising": SMA50_rising, "stop_ok": stop_ok,
    }

# ── Diagnostic funnel — tallies how far each ticker-day gets through
#    the pattern, across the full universe scan, so a 0-match run can
#    be diagnosed empirically instead of guessed at ─────────────────
FUNNEL_COUNTS = {
    "days_checked": 0,
    "passed_step1_A_green": 0,
    "passed_step2_B_swept_A_high": 0,
    "passed_step3_B_closed_below_A_close": 0,
    "passed_step4_C_closed_above_B_high": 0,
    "passed_step5_C_green": 0,
    "passed_step6_SMA50_rising": 0,
    "passed_step7_stop_at_least_1pct": 0,
}

def find_sweep_signals(df, cfg):
    """
    Checks the last `signal_lookback_days` trading days (1 = the most
    recent candle only) for the full pattern. Returns a list of hit
    dicts, most recent first. Also tallies FUNNEL_COUNTS for every
    day checked, regardless of match.
    """
    global FUNNEL_COUNTS
    n = len(df)
    hits = []
    for back in range(0, cfg["signal_lookback_days"]):
        i = (n - 1) - back
        passed, d = check_pattern_at(df, i, cfg)
        if not d:
            continue
        FUNNEL_COUNTS["days_checked"] += 1
        if d["A_green"]:
            FUNNEL_COUNTS["passed_step1_A_green"] += 1
            if d["B_swept_high"]:
                FUNNEL_COUNTS["passed_step2_B_swept_A_high"] += 1
                if d["B_closed_below"]:
                    FUNNEL_COUNTS["passed_step3_B_closed_below_A_close"] += 1
                    if d["C_above_B_high"]:
                        FUNNEL_COUNTS["passed_step4_C_closed_above_B_high"] += 1
                        if d["C_green"]:
                            FUNNEL_COUNTS["passed_step5_C_green"] += 1
                            if d["SMA50_rising"]:
                                FUNNEL_COUNTS["passed_step6_SMA50_rising"] += 1
                                if d["stop_ok"]:
                                    FUNNEL_COUNTS["passed_step7_stop_at_least_1pct"] += 1
        if passed:
            hits.append(d)
    return hits

# ── Technical signal: daily liquidity sweep breakout ──────────────
def analyze_sweep_breakout(sym, df_daily):
    """
    Returns dict with score and setup details, or None if the pattern
    is not present in the lookback window.
    """
    if df_daily is None or "Open" not in df_daily.columns:
        return None

    price   = float(df_daily["Close"].iloc[-1])
    avg_vol = float(df_daily["Volume"].tail(20).mean())
    if price   < CFG["min_price"]:      return None
    if avg_vol < CFG["min_avg_volume"]: return None

    n = len(df_daily)
    if n < CFG["sma_period"] + CFG["signal_lookback_days"] + 5:
        return None

    hits = find_sweep_signals(df_daily, CFG)
    if not hits:
        return None

    s = hits[0]   # most recent
    k = s["idx"]
    days_since_signal = (n - 1) - k

    # ── Entry / Stop (reasonable defaults — see header note) ─────────
    entry_price = s["cC"]
    stop_loss   = min(s["lB"], s["lC"])
    if entry_price <= stop_loss:
        return None
    risk_pct = (entry_price - stop_loss) / entry_price * 100
    target_price = entry_price + CFG["target_r"] * (entry_price - stop_loss)
    exit_by = pd.bdate_range(df_daily.index[k], periods=CFG["max_hold_days"] + 1)[-1]

    sweep_pct     = (s["hB"] - s["hA"]) / s["hA"] * 100       # how far B poked above A's high
    reject_pct    = (s["cA"] - s["cB"]) / s["cA"] * 100       # how far B closed back below A's close
    breakout_pct  = (s["cC"] - s["hB"]) / s["hB"] * 100       # C's close above B's high
    c_body_pct    = (s["cC"] - s["oC"]) / s["oC"] * 100
    sma = df_daily["Close"].rolling(CFG["sma_period"]).mean()
    sma_k = float(sma.iloc[k])
    above_sma = bool(s["cC"] > sma_k)
    vol_C  = float(df_daily["Volume"].iloc[k])
    avg20  = float(df_daily["Volume"].iloc[max(0, k-20):k].mean())
    vol_vs_avg = vol_C / avg20 if avg20 > 0 else 0.0

    # ── Score (0-100) ────────────────────────────────────────────
    score = 20   # base for clearing every gate
    reasons = []
    score += max(0, min(20, breakout_pct * 8))
    reasons.append(f"C>B-high+{breakout_pct:.1f}%")
    score += max(0, min(15, c_body_pct * 4))
    reasons.append(f"C body+{c_body_pct:.1f}%")
    score += max(0, min(10, reject_pct * 4))
    reasons.append(f"B rejected-{reject_pct:.1f}%")
    score += max(0, min(15, (vol_vs_avg - 1) * 15))
    reasons.append(f"Vol/20d x{vol_vs_avg:.2f}")
    if above_sma:
        score += 10; reasons.append(f"Above SMA{CFG['sma_period']}")
    else:
        reasons.append(f"Below SMA{CFG['sma_period']}")
    score += max(0, 10 - days_since_signal * 3)
    if risk_pct > 12:
        score -= 10; reasons.append("WideStop")
    score = round(min(100, max(0, score)))

    return {
        "Score"          : score,
        "Price"          : round(price, 2),
        "Entry_Price"    : round(entry_price, 2),
        "Stop_Loss"      : round(stop_loss, 2),
        "Risk_%"         : round(risk_pct, 1),
        "Target_Price"   : round(target_price, 2),
        "Target_R"       : CFG["target_r"],
        "Exit_By"        : exit_by.strftime("%Y-%m-%d"),
        "A_High"         : round(s["hA"], 2),
        "B_High"         : round(s["hB"], 2),
        "Sweep_%"        : round(sweep_pct, 1),
        "B_Reject_%"     : round(reject_pct, 1),
        "Breakout_%"     : round(breakout_pct, 1),
        "C_Body_%"       : round(c_body_pct, 1),
        "Vol_vs_20d_Avg" : round(vol_vs_avg, 2),
        "SMA50"          : round(sma_k, 2),
        "Above_SMA50"    : above_sma,
        "SMA50_20d_Ago"  : round(s["sma_prev"], 2),
        "Signal_Date"    : df_daily.index[k].strftime("%Y-%m-%d"),
        "Days_Since_Signal": days_since_signal,
        "Flags"          : " | ".join(reasons),
        "_df_daily"      : df_daily,
        "_sma"           : sma,
        "_idx"           : k,
    }

# ── Download: daily ───────────────────────────────────────────
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
                        df = _clean(df, min_bars=220)
                        if df is not None: out[sym] = df
                    except Exception: pass
            elif len(symbols) == 1:
                df = _clean(raw, min_bars=220)
                if df is not None: out[symbols[0]] = df
    except Exception: pass
    for sym in [s for s in symbols if s not in out]:
        for _ in range(2):
            try:
                df = yf.Ticker(sym).history(
                    start=start.strftime("%Y-%m-%d"),
                    end=end.strftime("%Y-%m-%d"),
                    auto_adjust=True, actions=False)
                df = _clean(df, min_bars=220)
                if df is not None: out[sym] = df; break
            except Exception: time.sleep(0.2)
        time.sleep(0.04)
    return out

# ── Live print ────────────────────────────────────────────────
LIVE_COLS = ["Ticker","Price","Score","Entry_Price","Stop_Loss",
             "Target_Price","Risk_%","Exit_By"]
_CW = {"Ticker":8,"Price":10,"Score":7,"Entry_Price":12,"Stop_Loss":11,
       "Target_Price":13,"Risk_%":8,"Exit_By":12}
_CF = {"Price":"${:.2f}","Score":"{:.0f}","Entry_Price":"${:.2f}",
       "Stop_Loss":"${:.2f}","Target_Price":"${:.2f}","Risk_%":"{:.1f}%"}
_hdr_done = False

def _live_header():
    global _hdr_done
    if _hdr_done: return
    print("\n" + "━"*95)
    print("  📊  LIVE MATCHES  —  each stock printed the moment it passes all 7 steps")
    print("━"*95)
    h = "".join(f"  {c:<{_CW.get(c,12)}}" for c in LIVE_COLS)
    print(h)
    print("  " + "─"*93)
    _hdr_done = True

def live_print(r):
    _live_header()
    row = ""
    for c in LIVE_COLS:
        val = r.get(c,"—")
        w   = _CW.get(c,12)
        fmt = _CF.get(c)
        try:   s = fmt.format(val) if (fmt and val not in("—",None)) else str(val)
        except Exception: s = str(val)
        row += f"  {s:<{w}}"
    print(row)

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

# ── Main scan — single pass (daily download only, then check) ────
print("━"*65)
print(f"  STEP 3  SCANNING {len(TICKERS)} TICKERS")
print("━"*65)
print("  Fetching daily bars (single download per ticker)")
print("  Pattern: A green → B sweeps above A's high but closes below A's close")
print("  → C closes above B's high and is green\n")

_hdr_done = False
results = []
no_daily_data = 0

daily_batches = [TICKERS[i:i+CFG["batch_size"]]
                 for i in range(0, len(TICKERS), CFG["batch_size"])]

daily_map = {}

with tqdm(total=len(TICKERS), desc="Daily fetch", unit="stk",
          bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]") as pbar:
    for batch in daily_batches:
        got = download_daily(batch, CFG["history_days"])
        daily_map.update(got)
        no_daily_data += len(batch) - len(got)
        pbar.update(len(batch))
        time.sleep(CFG["batch_sleep"])

got_daily = len(TICKERS) - no_daily_data
print(f"\n  Daily data : {got_daily}/{len(TICKERS)} tickers")
print()

for sym in tqdm(list(daily_map.keys()), desc="Checking sweep breakout", unit="stk"):
    try:
        r = analyze_sweep_breakout(sym, daily_map[sym])
        if r is None: continue
        r["Ticker"] = sym
        results.append(r)
        live_print(r)
    except Exception: pass

print(f"\n{'━'*65}")
print(f"  SCAN COMPLETE")
print(f"  Tickers scanned : {len(TICKERS)}")
print(f"  Daily data      : {got_daily}")
print(f"  ✅ Matches       : {len(results)}")
print(f"{'━'*65}")

# ── Diagnostic funnel — where ticker-days were filtered out ──────
print(f"\n{'━'*65}")
print(f"  🔍 FUNNEL — where ticker-days were filtered out")
print(f"  (tallied across every liquid ticker's last {CFG['signal_lookback_days']} trading day(s))")
print(f"{'━'*65}")
fc = FUNNEL_COUNTS
print(f"  Ticker-days checked                          : {fc['days_checked']}")
print(f"  Step 1 — candle A green                      : {fc['passed_step1_A_green']}")
print(f"  Step 2 — + candle B high above A's high      : {fc['passed_step2_B_swept_A_high']}")
print(f"  Step 3 — + candle B closed below A's close   : {fc['passed_step3_B_closed_below_A_close']}")
print(f"  Step 4 — + candle C closed above B's high    : {fc['passed_step4_C_closed_above_B_high']}")
print(f"  Step 5 — + candle C green                    : {fc['passed_step5_C_green']}")
print(f"  Step 6 — + SMA50 above its level {CFG['sma_rise_days']}d ago   : {fc['passed_step6_SMA50_rising']}")
print(f"  Step 7 — + stop >= {CFG['min_stop_pct']:.0f}% below entry         : {fc['passed_step7_stop_at_least_1pct']}  (= full pattern)")
print(f"{'━'*65}")

if not results:
    print("\n  No matches. Try relaxing (see the FUNNEL above to see which")
    print("  step is actually the bottleneck before guessing):")
    print("   signal_lookback_days            1 → 3     (patterns completed in the last 3 days)")
    print("   min_price                      10 → 5")
    print("   min_avg_volume             100000 → 50000")

results.sort(key=lambda x: x["Score"], reverse=True)

# ── Always build df_out and save/email (even if 0 results) ────
ts      = datetime.today().strftime("%Y%m%d_%H%M")
out_dir = os.environ.get("GITHUB_WORKSPACE", os.getcwd())

COLS = [
    "Ticker","Price","Score",
    "Entry_Price","Stop_Loss","Risk_%","Target_Price","Target_R","Exit_By",
    "A_High","B_High","Sweep_%","B_Reject_%","Breakout_%","C_Body_%",
    "Vol_vs_20d_Avg","SMA50","SMA50_20d_Ago","Above_SMA50",
    "Signal_Date","Days_Since_Signal","Flags",
]
df_out = pd.DataFrame([{k:v for k,v in r.items() if not k.startswith("_")}
                        for r in results]) if results else pd.DataFrame(columns=COLS)
if not df_out.empty:
    df_out = df_out[[c for c in COLS if c in df_out.columns]]
    df_out.reset_index(drop=True, inplace=True)

FMT = {
    "Price"         : lambda v: f"${v:.2f}",
    "Score"         : lambda v: f"{v:.0f}",
    "Entry_Price"   : lambda v: f"${v:.2f}",
    "Stop_Loss"     : lambda v: f"${v:.2f}",
    "Target_Price"  : lambda v: f"${v:.2f}",
    "A_High"        : lambda v: f"${v:.2f}",
    "B_High"        : lambda v: f"${v:.2f}",
    "SMA50"         : lambda v: f"${v:.2f}",
    "Risk_%"        : lambda v: f"{v:.1f}%",
    "Sweep_%"       : lambda v: f"{v:+.1f}%",
    "B_Reject_%"    : lambda v: f"{v:.1f}%",
    "Breakout_%"    : lambda v: f"{v:+.1f}%",
    "C_Body_%"      : lambda v: f"{v:+.1f}%",
    "Vol_vs_20d_Avg": lambda v: f"x{v:.2f}",
    "Days_Since_Signal": lambda v: f"{int(v)}d ago",
}

def fmt_v(col, val):
    if val is None or (isinstance(val, float) and np.isnan(val)): return "—"
    try:
        if col in FMT: return FMT[col](val)
    except Exception: pass
    return str(val) if str(val) not in ("nan","None","") else "—"

DISP_COLS = ["Ticker","Price","Score","Entry_Price","Stop_Loss","Target_Price","Risk_%",
             "Exit_By","Vol_vs_20d_Avg","Signal_Date"]

if _IN_NOTEBOOK and results:
    th = "".join(
        f'<th style="background:#0f172a;color:#e2e8f0;padding:9px 12px;'
        f'font-size:11px;font-weight:700;text-align:center;'
        f'border-bottom:2px solid #22c55e;white-space:nowrap">{c}</th>'
        for c in DISP_COLS)
    rows_html = ""
    for i, r in enumerate(results):
        bg  = "#ffffff" if i%2==0 else "#f0f9ff"
        tds = "".join(
            f'<td style="padding:7px 12px;font-size:12px;border-bottom:1px solid #e2e8f0;'
            f'white-space:nowrap">{fmt_v(col, r.get(col))}</td>' for col in DISP_COLS)
        rows_html += f'<tr style="background:{bg}">{tds}</tr>\n'
    display_html(f"""
<div style="background:linear-gradient(135deg,#0f172a,#1e3a5f);
        border-radius:10px;padding:18px 24px;margin-bottom:8px;
        font-family:'Segoe UI',Arial,sans-serif">
  <h2 style="margin:0;color:#60a5fa;font-size:20px;font-weight:700">
    📈 Liquidity Sweep + Rising SMA50
  </h2>
  <p style="margin:6px 0 0;color:#94a3b8;font-size:12px">
    {datetime.today().strftime('%Y-%m-%d %H:%M')} &nbsp;·&nbsp;
    <b style="color:#22c55e">{len(results)} matches</b> from {len(TICKERS)} tickers
  </p>
</div>
<div style="overflow-x:auto;border:1px solid #e2e8f0;border-radius:8px">
  <table style="border-collapse:collapse;width:100%;min-width:700px">
    <thead><tr>{th}</tr></thead>
    <tbody>{rows_html}</tbody>
  </table>
</div>""")

elif results:
    # ASCII table (CLI/GitHub Actions mode)
    CLI_COLS = [c for c in DISP_COLS if c in df_out.columns]
    col_w = {c: max(len(c), max(
        len(fmt_v(c, df_out[c].iloc[i])) for i in range(len(df_out))
    ))+2 for c in CLI_COLS}
    top  = "┬".join("─"*col_w[c] for c in CLI_COLS)
    sep  = "┼".join("─"*col_w[c] for c in CLI_COLS)
    bot  = "┴".join("─"*col_w[c] for c in CLI_COLS)
    hdr  = "│".join(c.center(col_w[c]) for c in CLI_COLS)
    inner= sum(col_w.values()) + len(CLI_COLS) - 1
    print()
    print(f"  ╔{'═'*inner}╗")
    tit = f"  Liquidity Sweep + Rising SMA50   {datetime.today().strftime('%Y-%m-%d')}   {len(df_out)} matches"
    print(f"  ║{tit.center(inner)}║")
    print(f"  ╚{'═'*inner}╝\n")
    print(f"  ┌{top}┐")
    print(f"  │{hdr}│")
    print(f"  ├{sep}┤")
    for i,(_, row_) in enumerate(df_out.iterrows()):
        cells=[fmt_v(c,row_.get(c)).center(col_w[c]) for c in CLI_COLS]
        print(f"  │{'│'.join(cells)}│")
        if i<len(df_out)-1: print(f"  ├{sep}┤")
    print(f"  └{bot}┘")
    print(f"""
  COLUMN KEY
  ──────────────────────────────────────────────────────
  Score           0-100 (breakout above B's high + C's body + how
                  hard B was rejected + volume + above SMA50 + freshness)
  Entry_Price     candle C's close
  Stop_Loss       lower of candle B's and C's lows (>= 1% below entry)
  Target_Price    entry + 0.75 x (entry - stop)
  Exit_By         sell at this day's close if neither stop nor target hit
  A_High          candle A's high (the liquidity level B swept)
  B_High          candle B's high (the level C had to close above)
  Sweep_%         how far B's high went above A's high
  B_Reject_%      how far B closed below A's close
  Breakout_%      C's close vs B's high
  Signal_Date     date of candle C
  ──────────────────────────────────────────────────────""")

# Save
fpath = os.path.join(out_dir, f"daily_sweep_sma50_strategy_{ts}.csv")
# ── Position sizing: Quantity / Total_Amount for a fixed max loss (see position_sizing.py)
try:
    from position_sizing import add_position_size
    df_out = add_position_size(df_out)
except Exception as _ps_err:
    print(f"  ⚠️  Position sizing skipped: {_ps_err}")
df_out.to_csv(fpath, index=False)
print(f"\n  💾 CSV → {fpath}")
tv = os.path.join(out_dir, f"tv_daily_sweep_sma50_strategy_{ts}.txt")
with open(tv,"w") as f:
    f.write(f"###Liquidity Sweep + Rising SMA50 {datetime.today().strftime('%Y-%m-%d')}\n")
    for r in results: f.write(f"NASDAQ:{r['Ticker']}\n")
print(f"  📋 TradingView → {tv}")
if results:
    print(f"\n  📋 Tickers (comma-separated):")
    print(f"  {', '.join(r['Ticker'] for r in results)}")

# Quantity / Total_Amount per ticker for the email (from the sized df_out)
_SIZE = {}
if {"Ticker", "Quantity", "Total_Amount"} <= set(df_out.columns):
    _SIZE = {t: (q, a) for t, q, a in zip(df_out["Ticker"], df_out["Quantity"], df_out["Total_Amount"])}

# ── Email with CSV attached ───────────────────────────────
def _send_email(rl, csv_path):
    import smtplib
    from email.mime.multipart import MIMEMultipart
    from email.mime.text      import MIMEText
    from email.mime.base      import MIMEBase
    from email                import encoders

    gu = _GMAIL_USER; gp = _GMAIL_PASS; et = _EMAIL_TO

    if not gu:
        print("[Email] ❌  GMAIL_USER secret is empty")
        return
    if not gp:
        print("[Email] ❌  GMAIL_PASS secret is empty")
        print("         → Must be a Gmail App Password (16 chars, no spaces)")
        return
    if not et:
        print("[Email] ❌  EMAIL_TO secret is empty")
        return

    eto = [e.strip() for e in et.split(",") if e.strip()]
    cnt = len(rl)

    try:
        print(f"[Email] Sending to {et}  ({cnt} results)...")

        heads = ["Ticker","Price","Score","Entry","Stop","Risk %","Target",
                 "Qty","Amount","Vol/20d","Exit by","Signal"]
        th_e = "".join(
            f'<th style="background:#1e293b;color:#e2e8f0;padding:8px 11px;'
            f'font-size:11px;font-weight:700;border-bottom:2px solid #3b82f6;'
            f'white-space:nowrap">{c}</th>' for c in heads)

        def td(v, extra=""):
            return f'<td style="padding:6px 11px;font-size:12px;{extra}">{v}</td>'

        rows_e = ""
        for i, r in enumerate(rl[:50]):
            bg = "#fff" if i % 2 == 0 else "#f0f9ff"
            q, a = _SIZE.get(r.get("Ticker"), ("—", None))
            rows_e += (
                f'<tr style="background:{bg}">'
                + td(r.get("Ticker","—"), "font-weight:700")
                + td(f'${float(r.get("Price",0) or 0):.2f}')
                + td(f'{float(r.get("Score",0) or 0):.0f}',
                     "font-weight:700;background:#166534;color:#fff;text-align:center")
                + td(f'${float(r.get("Entry_Price",0) or 0):.2f}', "color:#22c55e")
                + td(f'${float(r.get("Stop_Loss",0) or 0):.2f}', "color:#ef4444")
                + td(f'{float(r.get("Risk_%",0) or 0):.1f}%')
                + td(f'${float(r.get("Target_Price",0) or 0):.2f}', "color:#16a34a;font-weight:700")
                + td(f'{q}')
                + td(f'${float(a):,.0f}' if a is not None else "—")
                + td(f'x{float(r.get("Vol_vs_20d_Avg",0) or 0):.2f}')
                + td(r.get("Exit_By","—"))
                + td(r.get("Signal_Date","—"), "color:#a78bfa;font-weight:600")
                + '</tr>')
        no_results_msg = ('<tr><td colspan="12" style="padding:20px;text-align:center;'
                          'color:#94a3b8;font-size:13px">No matches today</td></tr>')
        more_note = (f'Showing top 50 of {cnt} — full list in the CSV attachment'
                     if cnt > 50 else '📎 Full results attached as CSV')

        fc = FUNNEL_COUNTS
        html_e = f"""<!DOCTYPE html><html><body style="margin:0;padding:0;
background:#f1f5f9;font-family:'Segoe UI',Arial,sans-serif">
<table width="100%" cellpadding="0" cellspacing="0"><tr><td align="center" style="padding:20px 10px">
<table width="100%" style="max-width:820px;background:#fff;border-radius:12px;
       overflow:hidden;box-shadow:0 4px 16px rgba(0,0,0,0.08)">
  <tr><td style="background:linear-gradient(135deg,#0f172a,#1e3a5f);padding:22px 28px">
<h1 style="margin:0;color:#60a5fa;font-size:20px;font-weight:700">
  📊 Liquidity Sweep + Rising SMA50
</h1>
<p style="margin:6px 0 0;color:#94a3b8;font-size:12px">
  {datetime.today().strftime('%Y-%m-%d %H:%M UTC')} &nbsp;·&nbsp;
  {cnt} match{'es' if cnt!=1 else ''} — liquidity sweep breakout (A green; B swept A's high, closed below
  A's close; C closed above B's high, green) + SMA50 rising + stop &ge; 1%<br>
  Plan: target {CFG['target_r']}R, stop, or exit after {CFG['max_hold_days']} days &nbsp;·&nbsp;
  2-yr backtest: ~62% wins, +0.11 to +0.14R per trade (before costs)
</p>
  </td></tr>
  <tr><td style="padding:14px 28px 4px;background:#0b1220">
<div style="background:#111827;border:1px solid #1f2937;border-radius:8px;padding:12px 16px">
  <p style="margin:0 0 6px;color:#93c5fd;font-size:12px;font-weight:700">
    🔍 FUNNEL — last {CFG['signal_lookback_days']} trading day(s), every liquid ticker
  </p>
  <p style="margin:0;color:#cbd5e1;font-size:12px">
    {fc['days_checked']} ticker-days &nbsp;→&nbsp;
    {fc['passed_step1_A_green']} A green &nbsp;→&nbsp;
    {fc['passed_step2_B_swept_A_high']} B swept A's high &nbsp;→&nbsp;
    {fc['passed_step3_B_closed_below_A_close']} B closed below A's close &nbsp;→&nbsp;
    {fc['passed_step4_C_closed_above_B_high']} C closed above B's high &nbsp;→&nbsp;
    {fc['passed_step5_C_green']} C green &nbsp;→&nbsp;
    {fc['passed_step6_SMA50_rising']} SMA50 rising &nbsp;→&nbsp;
    <b style="color:#facc15">{fc['passed_step7_stop_at_least_1pct']} stop &ge; 1% = full match</b>
  </p>
</div>
  </td></tr>
  <tr><td style="padding:16px">
<div style="overflow-x:auto;border-radius:8px;border:1px solid #e2e8f0">
  <table style="border-collapse:collapse;width:100%;min-width:700px">
    <thead><tr>{th_e}</tr></thead>
    <tbody>{rows_e or no_results_msg}</tbody>
  </table>
</div>
<p style="font-size:11px;color:#64748b;margin:8px 0 0">{more_note}</p>
{f'''<div style="margin-top:10px;background:#f8fafc;border:1px solid #e2e8f0;
        border-radius:6px;padding:10px 14px">
  <p style="margin:0 0 4px;font-size:10px;color:#94a3b8;font-weight:700">
    TICKERS (comma-separated, copy/paste)
  </p>
  <p style="margin:0;font-size:12px;color:#1e293b;font-family:monospace;
      word-break:break-all">{", ".join(r.get("Ticker","") for r in rl)}</p>
</div>''' if rl else ''}
  </td></tr>
  <tr><td style="background:#f8fafc;padding:12px 28px;
             border-top:1px solid #e2e8f0;text-align:center">
<p style="margin:0;color:#94a3b8;font-size:10px">
  ⚠️ Not financial advice &nbsp;·&nbsp; Entry = C close · Stop = lower of B/C lows · Target = entry + 0.75R · max hold 20 days · Qty sized for a $40 max loss &nbsp;·&nbsp; Auto-generated by GitHub Actions
</p>
  </td></tr>
</table>
</td></tr></table>
</body></html>"""

        plain_lines = [
            f"Liquidity Sweep + Rising SMA50 — {datetime.today().strftime('%Y-%m-%d')}",
            f"{cnt} matches (liquidity sweep breakout + SMA50 rising + stop >= 1%; target 0.75R, max hold 20 days)",
            "="*60,
            f"FUNNEL: {fc['days_checked']} ticker-days -> "
            f"{fc['passed_step1_A_green']} A green -> "
            f"{fc['passed_step2_B_swept_A_high']} B swept A high -> "
            f"{fc['passed_step3_B_closed_below_A_close']} B closed below A close -> "
            f"{fc['passed_step4_C_closed_above_B_high']} C > B high -> "
            f"{fc['passed_step5_C_green']} C green -> "
            f"{fc['passed_step6_SMA50_rising']} SMA50 rising -> "
            f"{fc['passed_step7_stop_at_least_1pct']} stop >= 1% (=full match)",
            "="*60,
        ]
        if rl:
            for r in rl[:50]:
                plain_lines.append(
                    f"{r.get('Ticker','—'):<7} ${float(r.get('Price',0) or 0):.2f}  "
                    f"Score:{float(r.get('Score',0) or 0):.0f}  "
                    f"Entry:${float(r.get('Entry_Price',0) or 0):.2f}  "
                    f"SL:${float(r.get('Stop_Loss',0) or 0):.2f}  "
                    f"Risk:{float(r.get('Risk_%',0) or 0):.1f}%  "
                    f"Target:${float(r.get('Target_Price',0) or 0):.2f}  "
                    f"Qty:{_SIZE.get(r.get('Ticker'), ('—', None))[0]}  "
                    f"ExitBy:{r.get('Exit_By','—')}  "
                    f"Signal:{r.get('Signal_Date','—')}")
            plain_lines.append("")
            plain_lines.append("Tickers (comma-separated):")
            plain_lines.append(", ".join(r.get("Ticker","") for r in rl))
        else:
            plain_lines.append("No matches today")
        plain_lines.append("\nFull results in CSV attachment.")
        plain_e = "\n".join(plain_lines)

        subj = (f"🎯 Sweep + SMA50 Strategy (0.75R) — {cnt} signal{'s' if cnt!=1 else ''}"
                f" — {datetime.today().strftime('%Y-%m-%d')}")

        msg = MIMEMultipart("mixed")
        msg["Subject"] = subj
        msg["From"]    = gu
        msg["To"]      = ", ".join(eto)

        alt = MIMEMultipart("alternative")
        alt.attach(MIMEText(plain_e, "plain"))
        alt.attach(MIMEText(html_e,  "html"))
        msg.attach(alt)

    except Exception as e:
        print(f"[Email] ❌  Failed to build email body: {type(e).__name__}: {e}")
        return

    if csv_path and os.path.exists(csv_path):
        try:
            with open(csv_path, "rb") as f:
                part = MIMEBase("application", "octet-stream")
                part.set_payload(f.read())
            encoders.encode_base64(part)
            part.add_header("Content-Disposition",
                f"attachment; filename={os.path.basename(csv_path)}")
            msg.attach(part)
            sz = os.path.getsize(csv_path)
            print(f"[Email] 📎 Attached: {os.path.basename(csv_path)} ({sz:,} bytes)")
        except Exception as e:
            print(f"[Email] ⚠️  CSV attach failed: {e}")

    try:
        print(f"[Email] Connecting to smtp.gmail.com:465 ...")
        with smtplib.SMTP_SSL("smtp.gmail.com", 465) as srv:
            srv.login(gu, gp.replace(" ", ""))
            srv.sendmail(gu, eto, msg.as_string())
        print(f"[Email] ✅  Sent successfully to: {', '.join(eto)}")
        print(f"[Email]    Subject: {subj}")
    except smtplib.SMTPAuthenticationError:
        print("[Email] ❌  AUTHENTICATION FAILED")
        print("         GMAIL_PASS must be a Gmail App Password, NOT your login password")
    except smtplib.SMTPException as e:
        print(f"[Email] ❌  SMTP error: {e}")
    except Exception as e:
        print(f"[Email] ❌  Unexpected error: {type(e).__name__}: {e}")

try:
    _send_email(results, fpath)
except Exception as e:
    print(f"[Email] ❌  Unexpected top-level error: {type(e).__name__}: {e}")
    print("[Email]    Continuing — CSV is still saved.")

if _IN_NOTEBOOK:
    try:
        from google.colab import files
        files.download(fpath); files.download(tv)
    except Exception: pass
else:
    print("  (CI: files in workspace, email sent)")

# ── Charts for top 5: price + SMA50 with B's high / stop, plus a
#    zoomed candle view marking candles A, B, C ─────────────────────
if results:
    top = results[:min(5,len(results))]
    fig, axes = plt.subplots(len(top),2,figsize=(16,4.2*len(top)),facecolor="#0f172a",
                              gridspec_kw={"width_ratios":[3,1]})
    if len(top)==1: axes = axes.reshape(1,2)
    for row, r in zip(axes, top):
        ax, ax_zoom = row[0], row[1]
        df_all = r["_df_daily"]
        df_p = df_all.tail(90).copy()
        sma_p = r["_sma"].reindex(df_p.index)
        ax.set_facecolor("#0f172a")
        ax.plot(df_p.index, df_p["Close"], color="#60a5fa", lw=1.6, label="Close", zorder=5)
        ax.plot(df_p.index, sma_p, color="#fbbf24", lw=1.2, ls="-.", label="SMA50", zorder=3)
        ax.axhline(r["B_High"], color="#3b82f6", lw=1.0, ls=":", alpha=0.9,
                  label=f"B high {r['B_High']:.2f}")
        ax.axhline(r["Stop_Loss"], color="#ef4444", lw=1.0, ls="--", alpha=0.85,
                  label=f"Stop {r['Stop_Loss']:.2f}")
        ax.axhline(r["Target_Price"], color="#22c55e", lw=1.0, ls="--", alpha=0.85,
                  label=f"Target {r['Target_Price']:.2f}")
        ax.set_title(
            f"{r['Ticker']}  |  {r['Price']:.2f} USD  |  Score {r['Score']}  |  {r['Signal_Date']}",
            color="#e2e8f0", fontsize=9, fontweight="bold", pad=7)
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))
        ax.tick_params(colors="#94a3b8", labelsize=8)
        for sp in ax.spines.values(): sp.set_edgecolor("#1e3a5f")
        ax.legend(loc="upper left", facecolor="#1e293b", labelcolor="#e2e8f0",
                  fontsize=6, framealpha=0.9, ncol=2)
        ax.grid(color="#1e3a5f", ls="--", lw=0.5, alpha=0.6)

        k  = r["_idx"]
        z0 = max(0, k - 7)
        zoom = df_all.iloc[z0: k + 1]
        x    = np.arange(len(zoom))
        ax_zoom.set_facecolor("#0f172a")
        colors = ["#22c55e" if c >= o else "#ef4444" for o, c in zip(zoom["Open"], zoom["Close"])]
        ax_zoom.vlines(x, zoom["Low"], zoom["High"], color=colors, lw=1)
        ax_zoom.bar(x, (zoom["Close"] - zoom["Open"]).abs().clip(lower=1e-9),
                    bottom=np.minimum(zoom["Open"], zoom["Close"]), color=colors, width=0.6)
        ax_zoom.axhline(r["A_High"], color="#a78bfa", lw=0.9, ls=":")
        ax_zoom.axhline(r["B_High"], color="#3b82f6", lw=0.9, ls=":")
        for lbl, idx in [("A", k-2), ("B", k-1), ("C", k)]:
            j = idx - z0
            if 0 <= j < len(zoom):
                ax_zoom.annotate(lbl, (j, float(zoom["High"].iloc[j])), textcoords="offset points",
                                 xytext=(0, 4), ha="center", color="#facc15", fontsize=7, fontweight="bold")
        ax_zoom.set_xticks(x)
        ax_zoom.set_xticklabels([d.strftime("%m/%d") for d in zoom.index], rotation=45, fontsize=6)
        ax_zoom.set_title(f"Sweep {r['Sweep_%']:+.1f}%  Breakout {r['Breakout_%']:+.1f}%",
                          color="#e2e8f0", fontsize=8, pad=5)
        ax_zoom.tick_params(colors="#94a3b8", labelsize=7)
        for sp in ax_zoom.spines.values(): sp.set_edgecolor("#1e3a5f")
        ax_zoom.grid(color="#1e3a5f", ls="--", lw=0.5, alpha=0.6)
    plt.suptitle(
        f"Liquidity Sweep + Rising SMA50  ·  {datetime.today().strftime('%Y-%m-%d')}",
        color="#60a5fa", fontsize=12, fontweight="bold", y=1.001)
    plt.tight_layout()
    cp = os.path.join(out_dir, f"daily_sweep_sma50_strategy_chart_{ts}.png")
    plt.savefig(cp, dpi=150, bbox_inches="tight", facecolor="#0f172a")
    if _IN_NOTEBOOK: plt.show()
    else: plt.close()
    print(f"  📊 Chart → {cp}")
    if _IN_NOTEBOOK:
        try:
            from google.colab import files; files.download(cp)
        except Exception: pass

print(f"""
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  📋 SIGNAL (all required). The last 3 daily candles:
  A (3rd day back), B, C (current).

  1) CANDLE A: GREEN.
  2) CANDLE B: HIGH goes ABOVE candle A's high (liquidity sweep)...
  3) ...but B CLOSES BELOW candle A's close.
  4) CANDLE C: CLOSES ABOVE candle B's high...
  5) ...and is GREEN.
  6) SMA50 today > SMA50 {CFG['sma_rise_days']} trading days ago.
  7) Stop (lower of B/C lows) at least {CFG['min_stop_pct']}% below the entry.

  signal_lookback_days = {CFG['signal_lookback_days']}  (1 = C must be the latest candle)

  📋 TRADE PLAN (from the 2-year research backtest)
  Entry_Price  = candle C's CLOSE
  Stop_Loss    = the lower of candle B's and C's LOWS
  Target_Price = entry + {CFG['target_r']} x (entry - stop)
  Exit_By      = close of the {CFG['max_hold_days']}th trading day if neither is hit
  Backtest: ~62% wins, +0.11 to +0.14R per trade (before costs)
  Universe    = price >= 10 USD, 20d avg volume >= 100k

  📋 SCORE (0-100)
  Base 20 + C's close above B's high (0-20) + C's body (0-15)
  + how far B closed below A's close (0-10) + volume vs 20d avg
  (0-15) + above SMA50 (10) + freshness (0-10); -10 if stop risk > 12%

  ⚙️  TUNE IF 0 RESULTS
  signal_lookback_days            1 → 3
  min_price                      10 → 5
  min_avg_volume             100000 → 50000
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
""")
