# ============================================================
# NASDAQ — EMA8/SMA21/SMA50/SMA150 Retest + 2-Candle Reclaim Scanner (v3)
# ============================================================
#
# SIGNAL (all required) — checked against EMA8, SMA21, SMA50, AND
# SMA150 independently; a stock matches if ANY of these MAs satisfies
# the full pattern (a stock can also match on more than one MA at
# once — that confluence is scored as a bonus):
#
#   0. PRIOR UPTREND: price was above the MA a few bars before the
#      retest (confirms this is a pullback within an uptrend, not
#      a breakdown).
#   1. MA RISING: SMA50 has started increasing (SMA50 today >
#      SMA50 sma_rising_lookback bars ago) — confirms the trend
#      driving this MA has turned back up. Checked once, regardless
#      of which MA is being retested (SMA50 acts as the broader
#      trend-context filter for all four).
#   2. CANDLE A (yesterday, the retest candle): RED (close < open)
#      AND closed BELOW the MA under test (EMA8/SMA21/SMA50/SMA150)
#      — the pullback actually touches/breaches that MA.
#   3. CANDLE B (today, the reclaim candle): GREEN (close > open)
#      AND closed ABOVE the SAME MA AND above EMA8 simultaneously
#      — a clean one-day reclaim of support. (When the MA under test
#      IS EMA8 itself, this is simply "closed above EMA8".)
#   4. VOLUME: today's volume (candle B) > yesterday's volume
#      (candle A) — confirms real buying interest on the reclaim.
#
# OUTPUT:
#   Price      = close of candle B (today, the signal/entry price)
#   Stop_Loss  = low of candle A (yesterday's candle low)
#   Retested_MA = the shortest-period MA that fired (EMA8 preferred
#                 over SMA21 over SMA50 over SMA150 if more than one
#                 matches — tightest support reported first)
#
# 2-PASS ARCHITECTURE (same as fundamental_v2.py / jma_stack_cross_v1.py):
#   PASS 1 — technical signal above (fast, no .info calls)
#   PASS 2 — fundamentals (slow, .info calls — pass-1 survivors only)
#
#   FINAL SCORE = Fund (0-50) + Tech (0-30) = 80
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
    "history_days"               : 550,

    # ── Fundamental thresholds ─────────────────────────────────
    "min_revenue_growth_pct"     : 0.0,
    "min_profit_margin_pct"      : 0.0,
    "max_debt_to_equity"         : 2.0,
    "min_current_ratio"          : 1.0,
    "min_roe_pct"                : 10.0,
    "max_pe_ratio"                : 50.0,
    "high_growth_threshold_pct"  : 15.0,

    # ── EMA8/SMA21/SMA50/SMA150 retest + 2-candle reclaim signal ──
    "ema8_period"                  : 8,
    "sma21_period"                 : 21,
    "sma50_period"                 : 50,
    "sma150_period"                : 150,

    "prior_uptrend_lookback"       : 3,    # bars before candle A that must show
                                            # price above the MA (confirms pullback
                                            # in an uptrend, not a breakdown)
    "sma_rising_lookback"          : 5,    # bars back to confirm SMA50 has turned up
    "require_prior_uptrend"        : True,
    "require_sma50_rising"         : True,
    "require_volume_confirmation"  : True, # candle B volume > candle A volume

    # ── Backtest (last 3 months, full universe) ─────────────────
    "backtest_lookback_days"       : 63,   # ~3 trading months
    "backtest_holding_days"        : 15,   # max bars to hold before "timeout"
    "backtest_reward_r"            : 2.0,  # win = hits 2:1 reward before stop-loss


    # ── Score gates ─────────────────────────────────────────────
    "min_tech_score"              : 10,   # out of 30
    "min_total_score"             : 15,   # out of 80 (Fund 0-50 + Tech 0-30)

    # ── Filters ─────────────────────────────────────────────────
    "min_avg_volume"              : 80_000,
    "min_price"                   : 2.0,

    "batch_size"                  : 50,
    "batch_sleep"                  : 1.5,
    "fund_sleep"                    : 0.3,
}

# ── Indicators / signal detection ─────────────────────────────
def check_two_candle_retest_reclaim(df, ma_series, ema8, prior_uptrend_lookback,
                                     sma_rising_lookback, sma50_for_rising,
                                     require_prior_uptrend, require_sma50_rising,
                                     require_volume_confirmation, end_idx=None):
    """
    Checks the exact 2-candle retest + reclaim pattern against ONE
    moving average series (EMA8, SMA21, SMA50, or SMA150).

    candle B (the reclaim bar) = row at position `end_idx` (default:
    the LAST row of df, i.e. "today"). candle A (the retest bar) =
    the row immediately before it. Passing an explicit end_idx lets
    the exact same logic be reused to backtest any historical day,
    not just the live/latest one.

    Every condition is computed explicitly and returned in the trace
    dict, so failures can be inspected step by step.

    Returns (passed: bool, trace: dict).
    """
    n = len(df)
    if end_idx is None:
        end_idx = n - 1
    trace = {}
    if end_idx < 1 or end_idx >= n:
        trace["fail_reason"] = "bad_end_idx"
        return False, trace
    if end_idx < max(prior_uptrend_lookback, sma_rising_lookback) + 2:
        trace["fail_reason"] = "not_enough_history"
        return False, trace

    i_B, i_A = end_idx, end_idx - 1

    open_A,  close_A  = float(df["Open"].iloc[i_A]),  float(df["Close"].iloc[i_A])
    low_A                                              = float(df["Low"].iloc[i_A])
    open_B,  close_B  = float(df["Open"].iloc[i_B]),  float(df["Close"].iloc[i_B])
    vol_A = float(df["Volume"].iloc[i_A])
    vol_B = float(df["Volume"].iloc[i_B])

    ma_A = float(ma_series.iloc[i_A])
    ma_B = float(ma_series.iloc[i_B])
    ema8_B = float(ema8.iloc[i_B])

    if any(np.isnan(v) for v in [ma_A, ma_B, ema8_B]):
        trace["fail_reason"] = "nan_indicator"
        return False, trace

    # ── Step 0: prior uptrend — price was above the MA a few bars
    #    before the retest candle ────────────────────────────────
    prior_i = i_A - prior_uptrend_lookback   # bar index before candle A
    prior_ok = False
    if prior_i >= 0:
        prior_close = float(df["Close"].iloc[prior_i])
        prior_ma    = float(ma_series.iloc[prior_i])
        if not (np.isnan(prior_close) or np.isnan(prior_ma)):
            prior_ok = prior_close > prior_ma
    trace["prior_uptrend_ok"] = prior_ok

    # ── Step 1: MA (SMA50) has started rising ──────────────────────
    sma_rising_ok = False
    if i_B - sma_rising_lookback >= 0:
        s_now  = float(sma50_for_rising.iloc[i_B])
        s_prev = float(sma50_for_rising.iloc[i_B-sma_rising_lookback])
        if not (np.isnan(s_now) or np.isnan(s_prev)):
            sma_rising_ok = s_now > s_prev
    trace["sma50_rising_ok"] = sma_rising_ok

    # ── Step 2: candle A — red, closed below the MA ────────────────
    candle_A_red      = close_A < open_A
    candle_A_below_ma = close_A < ma_A
    trace["candle_A_red"]      = candle_A_red
    trace["candle_A_below_ma"] = candle_A_below_ma

    # ── Step 3: candle B — green, closed above the MA AND above EMA8 ──
    candle_B_green      = close_B > open_B
    candle_B_above_ma   = close_B > ma_B
    candle_B_above_ema8 = close_B > ema8_B
    trace["candle_B_green"]      = candle_B_green
    trace["candle_B_above_ma"]   = candle_B_above_ma
    trace["candle_B_above_ema8"] = candle_B_above_ema8

    # ── Step 4: volume confirmation ────────────────────────────────
    vol_ok = vol_B > vol_A
    trace["vol_confirmed"] = vol_ok
    trace["vol_chg_pct"] = ((vol_B - vol_A) / vol_A * 100) if vol_A > 0 else 0.0

    trace.update({
        "end_idx": end_idx,
        "open_A": open_A, "close_A": close_A, "low_A": low_A,
        "open_B": open_B, "close_B": close_B,
        "ma_A": ma_A, "ma_B": ma_B, "ema8_B": ema8_B,
        "vol_A": vol_A, "vol_B": vol_B,
    })

    checks = [
        candle_A_red, candle_A_below_ma,
        candle_B_green, candle_B_above_ma, candle_B_above_ema8,
    ]
    if require_prior_uptrend:       checks.append(prior_ok)
    if require_sma50_rising:        checks.append(sma_rising_ok)
    if require_volume_confirmation: checks.append(vol_ok)

    passed = all(checks)
    trace["passed"] = passed
    return passed, trace

def simulate_trade_outcome(df, entry_idx, entry_price, stop_loss,
                            reward_r, holding_days):
    """
    Forward-simulates a single backtested trade from entry_idx+1
    onward, up to `holding_days` bars.

    Win  = High >= target (entry + reward_r * risk) before the stop
           is hit.
    Loss = Low <= stop_loss before the target is hit. If both the
           stop and target are touched on the SAME bar, conservatively
           counted as a loss (can't know intraday sequencing from
           daily OHLC).
    Timeout = neither hit within holding_days bars — excluded from
              the win-rate calculation, reported separately.

    Returns (outcome: "win"|"loss"|"timeout"|"invalid", bars_to_resolve: int|None).
    """
    risk = entry_price - stop_loss
    if risk <= 0:
        return "invalid", None
    target = entry_price + reward_r * risk

    n = len(df)
    end = min(entry_idx + 1 + holding_days, n)
    for j in range(entry_idx + 1, end):
        lo = float(df["Low"].iloc[j])
        hi = float(df["High"].iloc[j])
        hit_stop   = lo <= stop_loss
        hit_target = hi >= target
        if hit_stop and hit_target:
            return "loss", j - entry_idx     # conservative: stop assumed first
        if hit_stop:
            return "loss", j - entry_idx
        if hit_target:
            return "win", j - entry_idx
    return "timeout", None

def backtest_ticker(sym, df, sma21, sma50, sma150, ema8, cfg):
    """
    Re-runs the exact same 2-candle retest+reclaim check on every day
    over the last `backtest_lookback_days` trading days (checked
    against EMA8, SMA21, SMA50, AND SMA150 independently), simulating
    the forward outcome of each signal found.

    Returns a list of trade dicts:
      {ticker, date, ma, entry, stop, risk_pct, outcome, bars_to_resolve}
    """
    n = len(df)
    lb = cfg["backtest_lookback_days"]
    hold = cfg["backtest_holding_days"]
    # need enough forward bars to resolve a trade, and enough backward
    # history for SMA150/rising checks to be valid
    start_idx = max(160, n - lb)
    end_idx_max = n - 1   # can still include recent signals; they may resolve as "timeout"
                          # if not enough forward data exists yet — that's expected and fine

    ma_candidates = [
        ("EMA8",   ema8),
        ("SMA21",  sma21),
        ("SMA50",  sma50),
        ("SMA150", sma150),
    ]

    trades = []
    for i in range(start_idx, end_idx_max + 1):
        for ma_name, ma_series in ma_candidates:
            passed, trace = check_two_candle_retest_reclaim(
                df, ma_series, ema8,
                cfg["prior_uptrend_lookback"], cfg["sma_rising_lookback"], sma50,
                cfg["require_prior_uptrend"], cfg["require_sma50_rising"],
                cfg["require_volume_confirmation"], end_idx=i,
            )
            if not passed:
                continue
            entry = trace["close_B"]
            stop  = trace["low_A"]
            outcome, bars = simulate_trade_outcome(
                df, i, entry, stop, cfg["backtest_reward_r"], hold)
            if outcome == "invalid":
                continue
            risk_pct = (entry - stop) / entry * 100 if entry > 0 else 0
            trades.append({
                "ticker": sym, "date": df.index[i], "ma": ma_name,
                "entry": round(entry, 2), "stop": round(stop, 2),
                "risk_pct": round(risk_pct, 1),
                "outcome": outcome, "bars_to_resolve": bars,
            })
    return trades

def summarize_trades(trades):
    """Aggregates a list of trade dicts into win/loss/timeout counts + win rate %."""
    wins    = sum(1 for t in trades if t["outcome"] == "win")
    losses  = sum(1 for t in trades if t["outcome"] == "loss")
    timeouts= sum(1 for t in trades if t["outcome"] == "timeout")
    resolved = wins + losses
    win_rate = (wins / resolved * 100) if resolved > 0 else None
    return {
        "signals": len(trades), "wins": wins, "losses": losses,
        "timeouts": timeouts, "resolved": resolved, "win_rate_pct": win_rate,
    }

# ── Fundamental data fetch (robust) ────────────────────────────
def get_fundamentals(sym):
    """
    Fetch fundamentals via yf.Ticker(sym).info
    Never returns None — always returns a dict, Fund_Score=0 if
    data unavailable.
    """
    empty = {
        "Company":"—", "Sector":"—", "Industry":"—",
        "Rev_Growth_%":None, "Profit_Margin_%":None,
        "ROE_%":None, "PE_Ratio":None, "EPS":None,
        "Debt_Equity":None, "Current_Ratio":None,
        "Market_Cap_B":None, "Fund_Score":0,
        "Fund_Flags":"No Data",
    }
    try:
        tk   = yf.Ticker(sym)
        info = tk.info
        if not info or len(info) < 10:
            return empty

        def safe(key):
            v = info.get(key)
            if v is None: return None
            try:    return float(v)
            except Exception: return None

        rev_growth     = safe("revenueGrowth")
        profit_margin  = safe("profitMargins")
        roe            = safe("returnOnEquity")
        pe             = safe("trailingPE")
        eps            = safe("trailingEps")
        de             = safe("debtToEquity")
        cr             = safe("currentRatio")
        mktcap         = safe("marketCap")
        company        = info.get("longName", sym)
        sector         = info.get("sector", "—")
        industry       = info.get("industry", "—")

        rg_pct  = rev_growth    * 100 if rev_growth    is not None else None
        pm_pct  = profit_margin * 100 if profit_margin is not None else None
        roe_pct = roe           * 100 if roe           is not None else None

        # ── Score (0-50) ──────────────────────────────────────
        score = 0
        flags = []

        if rg_pct is not None:
            if rg_pct > CFG["min_revenue_growth_pct"]:
                score += 8; flags.append(f"RevG{rg_pct:+.0f}%")
            if rg_pct > CFG["high_growth_threshold_pct"]:
                score += 4; flags.append("HiGrw")
        if eps is not None and eps > 0:
            score += 7; flags.append(f"EPS${eps:.2f}")
        if pm_pct is not None and pm_pct > CFG["min_profit_margin_pct"]:
            score += 6; flags.append(f"Mgn{pm_pct:.0f}%")
        if de is not None and de < CFG["max_debt_to_equity"]:
            score += 6; flags.append(f"DE{de:.1f}")
        if cr is not None and cr > CFG["min_current_ratio"]:
            score += 5; flags.append(f"CR{cr:.1f}")
        if roe_pct is not None and roe_pct > CFG["min_roe_pct"]:
            score += 8; flags.append(f"ROE{roe_pct:.0f}%")
        if pe is not None and 0 < pe < CFG["max_pe_ratio"]:
            score += 6; flags.append(f"PE{pe:.0f}")

        return {
            "Company"          : company,
            "Sector"           : sector,
            "Industry"         : industry,
            "Rev_Growth_%"     : round(rg_pct, 1)  if rg_pct  is not None else None,
            "Profit_Margin_%"  : round(pm_pct, 1)  if pm_pct  is not None else None,
            "ROE_%"            : round(roe_pct, 1) if roe_pct is not None else None,
            "PE_Ratio"         : round(pe, 1)      if pe      is not None else None,
            "EPS"              : round(eps, 2)     if eps     is not None else None,
            "Debt_Equity"      : round(de, 2)      if de      is not None else None,
            "Current_Ratio"    : round(cr, 2)      if cr      is not None else None,
            "Market_Cap_B"     : round(mktcap/1e9,2) if mktcap is not None else None,
            "Fund_Score"       : min(50, score),
            "Fund_Flags"       : " ".join(flags) if flags else "—",
        }
    except Exception:
        return empty

# ── Technical signal: EMA8/SMA21/SMA50/SMA150 retest + 2-candle reclaim ──
ALL_BACKTEST_TRADES = []   # accumulates trades across the FULL universe scan

def analyze_retest_reclaim(sym, df):
    """
    Returns dict with tech_score and details, or None if no
    required condition is met against any of EMA8/SMA21/SMA50/SMA150.

    Regardless of whether today's live signal matches, this ALSO
    runs the last-3-months backtest for the ticker (if it passes the
    same basic price/volume filters) and appends the trades found to
    the global ALL_BACKTEST_TRADES accumulator, so the full-universe
    backtest summary reflects every liquid ticker scanned — not just
    today's matches.
    """
    global ALL_BACKTEST_TRADES

    n       = len(df)
    price   = float(df["Close"].iloc[-1])
    avg_vol = float(df["Volume"].tail(20).mean())

    if price   < CFG["min_price"]:      return None
    if avg_vol < CFG["min_avg_volume"]: return None
    if n < 160: return None   # need enough history for SMA150 + rising checks

    sma21  = df["Close"].rolling(CFG["sma21_period"]).mean()
    sma50  = df["Close"].rolling(CFG["sma50_period"]).mean()
    sma150 = df["Close"].rolling(CFG["sma150_period"]).mean()
    ema8   = df["Close"].ewm(span=CFG["ema8_period"], adjust=False).mean()

    # ── Backtest: last 3 months, this ticker, regardless of live match ──
    bt_trades = backtest_ticker(sym, df, sma21, sma50, sma150, ema8, CFG)
    ALL_BACKTEST_TRADES.extend(bt_trades)
    bt_summary = summarize_trades(bt_trades)

    # Order = shortest period first, so the tightest/most immediate
    # support is reported as the primary Retested_MA when more than
    # one fires at once.
    ma_candidates = [
        ("EMA8",   ema8),
        ("SMA21",  sma21),
        ("SMA50",  sma50),
        ("SMA150", sma150),
    ]
    matches = []
    for ma_name, ma_series in ma_candidates:
        passed, trace = check_two_candle_retest_reclaim(
            df, ma_series, ema8,
            CFG["prior_uptrend_lookback"], CFG["sma_rising_lookback"], sma50,
            CFG["require_prior_uptrend"], CFG["require_sma50_rising"],
            CFG["require_volume_confirmation"],
        )
        if passed:
            matches.append((ma_name, trace))

    if not matches:
        return None

    ma_name, trace = matches[0]
    matched_names = [m[0] for m in matches]
    matched_count = len(matches)

    close_A, low_A   = trace["close_A"], trace["low_A"]
    close_B          = trace["close_B"]
    ma_A, ma_B       = trace["ma_A"], trace["ma_B"]
    ema8_B           = trace["ema8_B"]
    vol_chg_pct      = trace["vol_chg_pct"]

    signal_price = close_B          # today's (candle B) close — the entry price
    stop_loss    = low_A            # yesterday's (candle A) low

    risk_pct = ((signal_price - stop_loss) / signal_price * 100
                if signal_price > 0 else 0)

    # ── Technical score (0-30) ────────────────────────────────
    ts = 0
    tr = [f"Retest+Reclaim({ma_name})"]
    if matched_count > 1:
        conf_pts = min(6, 2 * (matched_count - 1))
        ts += conf_pts; tr.append(f"Confluence({'+'.join(matched_names)})")

    dist_below_ma_A = (ma_A - close_A) / ma_A * 100 if ma_A > 0 else 0
    ts += 6; tr.append(f"CandleA_Below{ma_name}{dist_below_ma_A:+.1f}%")

    dist_above_ma_B = (close_B - ma_B) / ma_B * 100 if ma_B > 0 else 0
    ma_pts = 6 if dist_above_ma_B >= 2 else 4
    ts += ma_pts; tr.append(f"CandleB_Above{ma_name}{dist_above_ma_B:+.1f}%")

    dist_above_ema8 = (close_B - ema8_B) / ema8_B * 100 if ema8_B > 0 else 0
    ema_pts = 6 if dist_above_ema8 >= 2 else 4
    ts += ema_pts; tr.append(f"CandleB_AboveEMA8{dist_above_ema8:+.1f}%")

    vol_pts = 8 if vol_chg_pct >= 50 else (5 if vol_chg_pct >= 20 else 3)
    ts += vol_pts; tr.append(f"Vol{vol_chg_pct:+.0f}%")

    if bt_summary["win_rate_pct"] is not None:
        if bt_summary["win_rate_pct"] >= 60: ts += 4; tr.append("StrongBacktest")
        elif bt_summary["win_rate_pct"] >= 45: ts += 2

    ts = min(30, ts)

    return {
        "tech_score"     : ts,
        "tech_reasons"   : " | ".join(tr),
        "Retested_MA"    : ma_name,
        "Matched_MAs"    : "+".join(matched_names),
        "Price"          : round(signal_price, 2),
        "Stop_Loss"      : round(stop_loss, 2),
        "Risk_%"         : round(risk_pct, 1),
        "Candle_A_Close" : round(close_A, 2),
        "Candle_A_Low"   : round(low_A, 2),
        "Candle_B_Close" : round(close_B, 2),
        "EMA8"           : round(ema8_B, 2),
        "SMA21"          : round(float(sma21.iloc[-1]), 2),
        "SMA50"          : round(float(sma50.iloc[-1]), 2),
        "SMA150"         : round(float(sma150.iloc[-1]), 2),
        "Vol_Chg_%"      : round(vol_chg_pct, 1),
        "Backtest_Signals_3M": bt_summary["signals"],
        "Backtest_Wins"      : bt_summary["wins"],
        "Backtest_Losses"    : bt_summary["losses"],
        "Backtest_Timeouts"  : bt_summary["timeouts"],
        "Backtest_WinRate_%" : (round(bt_summary["win_rate_pct"], 1)
                                 if bt_summary["win_rate_pct"] is not None else None),
        "_df"            : df,
        "_sma21"         : sma21,
        "_sma50"         : sma50,
        "_sma150"        : sma150,
        "_ema8"          : ema8,
    }

# ── Download ──────────────────────────────────────────────────
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

def download(symbols, days):
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
                        df = _clean(df)
                        if df is not None: out[sym] = df
                    except Exception: pass
            elif len(symbols) == 1:
                df = _clean(raw)
                if df is not None: out[symbols[0]] = df
    except Exception: pass
    for sym in [s for s in symbols if s not in out]:
        for _ in range(2):
            try:
                df = yf.Ticker(sym).history(
                    start=start.strftime("%Y-%m-%d"),
                    end=end.strftime("%Y-%m-%d"),
                    auto_adjust=True, actions=False)
                df = _clean(df)
                if df is not None: out[sym] = df; break
            except Exception: time.sleep(0.2)
        time.sleep(0.04)
    return out

# ── Live print ────────────────────────────────────────────────
LIVE_COLS = ["Ticker","Price","Stop_Loss","Total","Fund","Tech",
             "Retested_MA","Risk_%","Sector"]
_CW = {"Ticker":8,"Price":10,"Stop_Loss":11,"Total":7,"Fund":6,"Tech":6,
       "Retested_MA":12,"Risk_%":9,"Sector":20}
_CF = {"Price":"${:.2f}","Stop_Loss":"${:.2f}","Total":"{:.0f}","Fund":"{:.0f}",
       "Tech":"{:.0f}","Risk_%":"{:+.1f}%"}
_hdr_done = False

def _live_header():
    global _hdr_done
    if _hdr_done: return
    print("\n" + "━"*95)
    print("  📊  LIVE MATCHES  —  each stock printed the moment it passes all filters")
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
chk = download(["AAPL","MSFT","NVDA"], 300)
if not chk:
    print("❌  No data.")
else:
    for s, d in chk.items():
        print(f"  ✅ {s}: {len(d)} bars  ${float(d['Close'].iloc[-1]):.2f}  {d.index[-1].date()}")
    print("\n  Testing .info fetch for AAPL...")
    t0  = time.time()
    fnd = get_fundamentals("AAPL")
    ela = time.time() - t0
    print(f"  ✅ Fund_Score={fnd['Fund_Score']}/50  Flags: {fnd['Fund_Flags']}  ({ela:.1f}s)")
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

# ── Main scan — 2-pass ───────────────────────────────────────
print("━"*65)
print(f"  STEP 3  SCANNING {len(TICKERS)} TICKERS")
print("━"*65)
print("  Pass 1: EMA8/SMA21/SMA50/SMA150 retest + 2-candle reclaim screening (fast)")
print("  Pass 2: Fundamental fetch for pass-1 stocks only\n")

_hdr_done   = False
results     = []
tech_passes = []
no_data     = 0

batches = [TICKERS[i:i+CFG["batch_size"]]
           for i in range(0, len(TICKERS), CFG["batch_size"])]

# ── PASS 1: Technical signal (no .info calls) ────────────────
with tqdm(total=len(TICKERS), desc="Pass 1 Tech", unit="stk",
          bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]") as pbar:
    for batch in batches:
        data_map = download(batch, CFG["history_days"])
        no_data += len(batch) - len(data_map)
        for sym in batch:
            pbar.update(1)
            if sym not in data_map: continue
            try:
                ts = analyze_retest_reclaim(sym, data_map[sym])
                if ts is None: continue
                if ts["tech_score"] < CFG["min_tech_score"]: continue
                tech_passes.append({
                    "sym"  : sym,
                    "price": float(data_map[sym]["Close"].iloc[-1]),
                    "ts"   : ts,
                })
            except Exception: pass
        time.sleep(CFG["batch_sleep"])

got = len(TICKERS) - no_data
pct = got / max(len(TICKERS), 1) * 100
print(f"\n  Pass 1 done: {got}/{len(TICKERS)} got data ({pct:.0f}%)")
print(f"  Tech passes: {len(tech_passes)} stocks → fetching fundamentals now\n")

# ── PASS 2: Fundamentals for tech-pass stocks only ──────────────
print("━"*65)
print(f"  PASS 2  FUNDAMENTAL CHECK ({len(tech_passes)} stocks)")
print("━"*65+"\n")

for item in tqdm(tech_passes, desc="Pass 2 Fund", unit="stk"):
    sym   = item["sym"]; price = item["price"]; ts = item["ts"]
    try:
        fund = get_fundamentals(sym)
        time.sleep(CFG["fund_sleep"])

        fs, tscr = fund["Fund_Score"], ts["tech_score"]
        total = fs + tscr
        if total < CFG["min_total_score"]: continue

        result = {
            "Ticker"            : sym,
            "Price"             : ts["Price"],
            "Stop_Loss"         : ts["Stop_Loss"],
            "Risk_%"            : ts["Risk_%"],
            "Total"             : total,
            "Fund"              : fs,
            "Tech"              : tscr,
            "Sector"            : fund["Sector"],
            "Company"           : fund["Company"],
            "Industry"          : fund["Industry"],
            "Retested_MA"       : ts["Retested_MA"],
            "Matched_MAs"       : ts["Matched_MAs"],
            "Candle_A_Close"    : ts["Candle_A_Close"],
            "Candle_A_Low"      : ts["Candle_A_Low"],
            "Candle_B_Close"    : ts["Candle_B_Close"],
            "EMA8"              : ts["EMA8"],
            "SMA21"             : ts["SMA21"],
            "SMA50"             : ts["SMA50"],
            "SMA150"            : ts["SMA150"],
            "Vol_Chg_%"         : ts["Vol_Chg_%"],
            "Backtest_Signals_3M": ts["Backtest_Signals_3M"],
            "Backtest_Wins"      : ts["Backtest_Wins"],
            "Backtest_Losses"    : ts["Backtest_Losses"],
            "Backtest_Timeouts"  : ts["Backtest_Timeouts"],
            "Backtest_WinRate_%" : ts["Backtest_WinRate_%"],
            "Tech_Flags"        : ts["tech_reasons"],
            "Rev_Growth_%"      : fund["Rev_Growth_%"],
            "Profit_Margin_%"   : fund["Profit_Margin_%"],
            "ROE_%"             : fund["ROE_%"],
            "PE_Ratio"          : fund["PE_Ratio"],
            "EPS"               : fund["EPS"],
            "Debt_Equity"       : fund["Debt_Equity"],
            "Market_Cap_B"      : fund["Market_Cap_B"],
            "Fund_Flags"        : fund["Fund_Flags"],
            # internals
            "_df"    : ts["_df"],
            "_sma21" : ts["_sma21"],
            "_sma50" : ts["_sma50"],
            "_sma150": ts["_sma150"],
            "_ema8"  : ts["_ema8"],
        }
        results.append(result)
        live_print(result)
    except Exception: pass

print(f"\n{'━'*65}")
print(f"  SCAN COMPLETE")
print(f"  Tickers    : {len(TICKERS)}")
print(f"  Got data   : {got}  ({pct:.0f}%)")
print(f"  Tech passes: {len(tech_passes)}")
print(f"  ✅ Matches  : {len(results)}")
print(f"{'━'*65}")

# ── Full-universe 3-month backtest summary ─────────────────────
BT_SUMMARY = summarize_trades(ALL_BACKTEST_TRADES)
print(f"\n{'━'*65}")
print(f"  📈 BACKTEST — LAST {CFG['backtest_lookback_days']} TRADING DAYS (~3 MONTHS)")
print(f"  Full NASDAQ universe, {CFG['backtest_reward_r']:.0f}:1 reward vs stop-loss, "
      f"{CFG['backtest_holding_days']}-day max hold")
print(f"{'━'*65}")
print(f"  Total signals found : {BT_SUMMARY['signals']}")
print(f"  Wins                : {BT_SUMMARY['wins']}")
print(f"  Losses              : {BT_SUMMARY['losses']}")
print(f"  Timeouts (excluded) : {BT_SUMMARY['timeouts']}")
if BT_SUMMARY["win_rate_pct"] is not None:
    print(f"  ✅ SUCCESS RATE      : {BT_SUMMARY['win_rate_pct']:.1f}%  "
          f"({BT_SUMMARY['wins']}/{BT_SUMMARY['resolved']} resolved trades)")
else:
    print(f"  ✅ SUCCESS RATE      : n/a (no resolved trades yet)")
print(f"{'━'*65}")

if not results:
    print("\n  No matches. Try relaxing:")
    print("   min_tech_score                 10 → 6")
    print("   min_total_score                15 → 5")
    print("   require_prior_uptrend        True → False")
    print("   require_sma50_rising         True → False")
    print("   require_volume_confirmation  True → False")
    print("   prior_uptrend_lookback          3 → 5")
    print("   min_price                        2 → 1")
    print("   min_avg_volume               80000 → 50000")

# Sort by total score (always runs, even on empty list)
results.sort(key=lambda x: x["Total"], reverse=True)

# ── Always build df_out and save/email (even if 0 results) ────
ts      = datetime.today().strftime("%Y%m%d_%H%M")
out_dir = os.environ.get("GITHUB_WORKSPACE", os.getcwd())

COLS = [
    "Ticker","Company","Sector","Price","Stop_Loss","Risk_%",
    "Total","Fund","Tech",
    "Rev_Growth_%","Profit_Margin_%","ROE_%","PE_Ratio","EPS",
    "Retested_MA","Matched_MAs","Candle_A_Close","Candle_A_Low","Candle_B_Close",
    "EMA8","SMA21","SMA50","SMA150","Vol_Chg_%",
    "Backtest_Signals_3M","Backtest_Wins","Backtest_Losses",
    "Backtest_Timeouts","Backtest_WinRate_%",
    "Tech_Flags","Fund_Flags",
]
df_out = pd.DataFrame([{k:v for k,v in r.items() if not k.startswith("_")}
                        for r in results]) if results else pd.DataFrame(columns=COLS)
if not df_out.empty:
    df_out = df_out[[c for c in COLS if c in df_out.columns]]
    df_out.reset_index(drop=True, inplace=True)

FMT = {
    "Price"          : lambda v: f"${v:.2f}",
    "Stop_Loss"      : lambda v: f"${v:.2f}",
    "Risk_%"         : lambda v: f"{v:+.1f}%",
    "Total"          : lambda v: f"{v:.0f}",
    "Fund"           : lambda v: f"{v:.0f}",
    "Tech"           : lambda v: f"{v:.0f}",
    "Rev_Growth_%"   : lambda v: f"{v:+.1f}%",
    "Profit_Margin_%": lambda v: f"{v:.1f}%",
    "ROE_%"          : lambda v: f"{v:.1f}%",
    "PE_Ratio"       : lambda v: f"{v:.1f}",
    "EPS"            : lambda v: f"${v:.2f}",
    "Market_Cap_B"   : lambda v: f"${v:.2f}B",
    "Candle_A_Close" : lambda v: f"${v:.2f}",
    "Candle_A_Low"   : lambda v: f"${v:.2f}",
    "Candle_B_Close" : lambda v: f"${v:.2f}",
    "Backtest_WinRate_%": lambda v: f"{v:.1f}%",
    "EMA8"           : lambda v: f"${v:.2f}",
    "SMA21"          : lambda v: f"${v:.2f}",
    "SMA50"          : lambda v: f"${v:.2f}",
    "SMA150"         : lambda v: f"${v:.2f}",
    "Vol_Chg_%"      : lambda v: f"{v:+.1f}%",
}

def fmt_v(col, val):
    if val is None or (isinstance(val, float) and np.isnan(val)): return "—"
    try:
        if col in FMT: return FMT[col](val)
    except Exception: pass
    return str(val) if str(val) not in ("nan","None","") else "—"

if _IN_NOTEBOOK and results:
    DISP = ["Ticker","Company","Sector","Price","Stop_Loss",
            "Total","Fund","Tech",
            "Retested_MA","Risk_%","Backtest_WinRate_%"]
    DISP = [c for c in DISP if c in df_out.columns]

    gc = "#22c55e"

    th = "".join(
        f'<th style="background:#0f172a;color:#e2e8f0;padding:9px 12px;'
        f'font-size:11px;font-weight:700;text-align:center;'
        f'border-bottom:2px solid {gc};white-space:nowrap">{c}</th>'
        for c in DISP
    )
    rows_html = ""
    for i, r in enumerate(results):
        bg  = "#ffffff" if i%2==0 else "#f0f9ff"
        tds = ""
        for col in DISP:
            raw  = r.get(col)
            disp = fmt_v(col, raw)
            sty  = ""
            if col == "Total":
                try:
                    v = float(raw)
                    g = int(min(220, 80 + v*2))
                    sty = f"background:rgb(20,{g},60);color:#fff;font-weight:700;text-align:center"
                except Exception: pass
            elif col == "Fund":
                try:
                    v = float(raw)
                    g = int(min(200, 60 + v*2.5))
                    sty = f"background:rgb(20,{g},80);color:#fff;font-weight:600;text-align:center"
                except Exception: pass
            elif col in ("Risk_%","Vol_Chg_%"):
                try:
                    v = float(str(raw).replace("%","").replace("+",""))
                    clr = "#22c55e" if v >= 0 else "#ef4444"
                    sty = f"color:{clr};font-weight:600"
                except Exception: pass
            elif col == "Stop_Loss":
                sty = "color:#ef4444;font-weight:600"
            tds += (f'<td style="padding:7px 12px;font-size:12px;'
                    f'border-bottom:1px solid #e2e8f0;white-space:nowrap;{sty}">'
                    f'{disp}</td>')
        rows_html += f'<tr style="background:{bg}">{tds}</tr>\n'

    table_html = f"""
<div style="margin:14px 0">
  <div style="background:linear-gradient(90deg,{gc}22,#0f172a);
          border-left:4px solid {gc};border-radius:6px 6px 0 0;
          padding:10px 18px;display:flex;align-items:center;gap:10px">
    <span style="font-size:18px">🎯</span>
    <span style="color:#f1f5f9;font-size:15px;font-weight:700">EMA8/SMA21/SMA50/SMA150 Retest + 2-Candle Reclaim</span>
    <span style="color:{gc};font-size:12px;margin-left:8px">{len(results)} stock{'s' if len(results)!=1 else ''}</span>
  </div>
  <div style="overflow-x:auto;border:1px solid #e2e8f0;border-top:none;
          border-radius:0 0 8px 8px;box-shadow:0 2px 8px rgba(0,0,0,0.05)">
    <table style="border-collapse:collapse;width:100%;min-width:700px">
      <thead><tr>{th}</tr></thead>
      <tbody>{rows_html}</tbody>
    </table>
  </div>
</div>"""

    header_html = f"""
<div style="background:linear-gradient(135deg,#0f172a,#1e3a5f);
        border-radius:10px;padding:18px 24px;margin-bottom:8px;
        font-family:'Segoe UI',Arial,sans-serif">
  <h2 style="margin:0;color:#60a5fa;font-size:20px;font-weight:700">
    📈 EMA8/SMA21/SMA50/SMA150 Retest + 2-Candle Reclaim
  </h2>
  <p style="margin:6px 0 0;color:#94a3b8;font-size:12px">
    {datetime.today().strftime('%Y-%m-%d %H:%M')} &nbsp;·&nbsp;
    <b style="color:#22c55e">{len(results)} matches</b> from {len(TICKERS)} tickers
  </p>
</div>"""

    legend_html = """
<div style="background:#f8fafc;border:1px solid #e2e8f0;border-radius:8px;
        padding:12px 18px;margin-top:6px;font-size:11px;color:#64748b;
        font-family:'Segoe UI',Arial,sans-serif">
  <b style="color:#475569">GUIDE</b> &nbsp;·&nbsp;
  Total = Fund(0-50) + Tech(0-30) &nbsp;·&nbsp;
  Signal = candle A (yesterday) red &amp; closed below EMA8/SMA21/SMA50/SMA150, candle B
  (today) green &amp; closed above the same MA and EMA8, volume up vs
  yesterday, SMA50 rising, in a prior uptrend &nbsp;·&nbsp;
  Price = candle B close &nbsp;·&nbsp; Stop_Loss = candle A low
</div>"""

    display_html(header_html + table_html + legend_html)

elif results:
    # ASCII table (CLI/GitHub Actions mode)
    CLI_COLS = ["Ticker","Price","Stop_Loss","Total","Fund","Tech",
                "Retested_MA","Risk_%","Backtest_WinRate_%","Sector"]
    CLI_COLS = [c for c in CLI_COLS if c in df_out.columns]
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
    tit = f"  EMA8/SMA21/SMA50/SMA150 Retest + 2-Candle Reclaim   {datetime.today().strftime('%Y-%m-%d')}   {len(df_out)} matches"
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
  Total          Fund(0-50) + Tech(0-30)
  Price           candle B (today) close — the entry price
  Stop_Loss       candle A (yesterday) low
  Retested_MA     which MA (EMA8/SMA21/SMA50/SMA150) this pattern formed against
  Matched_MAs     all MAs that matched at once (confluence), if more than one
  Risk_%          (Price - Stop_Loss) / Price
  Backtest_WinRate_%  this ticker's own win rate over the last 3 months
                      (n/a if it had no resolved historical signals)
  ──────────────────────────────────────────────────────""")

# Save
fpath = os.path.join(out_dir, f"sma150_base_reclaim_{ts}.csv")
df_out.to_csv(fpath, index=False)
print(f"\n  💾 CSV → {fpath}")
tv = os.path.join(out_dir, f"tv_sma150_base_reclaim_{ts}.txt")
with open(tv,"w") as f:
    f.write(f"###EMA8/SMA21/SMA50/SMA150 Retest Reclaim {datetime.today().strftime('%Y-%m-%d')}\n")
    for r in results: f.write(f"NASDAQ:{r['Ticker']}\n")
print(f"  📋 TradingView → {tv}")

# Save backtest trade log (every signal found in the last 3 months, full universe)
bt_fpath = os.path.join(out_dir, f"sma150_base_reclaim_backtest_{ts}.csv")
bt_df = pd.DataFrame(ALL_BACKTEST_TRADES) if ALL_BACKTEST_TRADES else pd.DataFrame(
    columns=["ticker","date","ma","entry","stop","risk_pct","outcome","bars_to_resolve"])
bt_df.to_csv(bt_fpath, index=False)
print(f"  💾 Backtest trade log → {bt_fpath}  ({len(ALL_BACKTEST_TRADES)} trades)")

# ── Email with CSV attached ───────────────────────────────
def _send_email(rl, csv_path, bt_csv_path=None):
    import smtplib
    from email.mime.multipart import MIMEMultipart
    from email.mime.text      import MIMEText
    from email.mime.base      import MIMEBase
    from email                import encoders

    gu = _GMAIL_USER; gp = _GMAIL_PASS; et = _EMAIL_TO

    if not gu:
        print("[Email] ❌  GMAIL_USER secret is empty")
        print("         → Repo → Settings → Secrets → Actions → GMAIL_USER")
        return
    if not gp:
        print("[Email] ❌  GMAIL_PASS secret is empty")
        print("         → Must be a Gmail App Password (16 chars, no spaces)")
        print("         → Get one at: myaccount.google.com/apppasswords")
        return
    if not et:
        print("[Email] ❌  EMAIL_TO secret is empty")
        print("         → Repo → Settings → Secrets → Actions → EMAIL_TO")
        return

    eto = [e.strip() for e in et.split(",") if e.strip()]
    cnt = len(rl)

    try:
        print(f"[Email] Sending to {et}  ({cnt} results)...")

        th_e = "".join(
            f'<th style="background:#1e293b;color:#e2e8f0;padding:8px 11px;'
            f'font-size:11px;font-weight:700;border-bottom:2px solid #3b82f6;'
            f'white-space:nowrap">{c}</th>'
            for c in ["Ticker","Price","Stop_Loss","Total","Fund","Tech",
                      "Retested_MA","Risk_%","Backtest_WinRate_%"]
        )
        rows_e = ""
        for i, r in enumerate(rl[:50]):
            bg  = "#fff" if i % 2 == 0 else "#f0f9ff"
            ticker = r.get("Ticker","—")
            price  = r.get("Price",0) or 0
            sl     = r.get("Stop_Loss",0) or 0
            total  = r.get("Total",0) or 0
            fund   = r.get("Fund",0) or 0
            tech   = r.get("Tech",0) or 0
            ma     = r.get("Retested_MA","—")
            risk   = r.get("Risk_%",0) or 0
            bwr    = r.get("Backtest_WinRate_%")
            bwr_disp = f"{bwr:.1f}%" if bwr is not None else "n/a"
            rows_e += (
                f'<tr style="background:{bg}">'
                f'<td style="padding:6px 11px;font-size:12px;font-weight:700">{ticker}</td>'
                f'<td style="padding:6px 11px;font-size:12px">${float(price):.2f}</td>'
                f'<td style="padding:6px 11px;font-size:12px;color:#ef4444;font-weight:600">'
                f'${float(sl):.2f}</td>'
                f'<td style="padding:6px 11px;font-size:12px;font-weight:700;'
                f'background:#166534;color:#fff;text-align:center">{float(total):.0f}</td>'
                f'<td style="padding:6px 11px;font-size:12px">{float(fund):.0f}</td>'
                f'<td style="padding:6px 11px;font-size:12px">{float(tech):.0f}</td>'
                f'<td style="padding:6px 11px;font-size:12px;text-align:center;'
                f'color:#a78bfa;font-weight:600">{ma}</td>'
                f'<td style="padding:6px 11px;font-size:12px;color:'
                f'{"#22c55e" if float(risk)>=0 else "#ef4444"}">{float(risk):+.1f}%</td>'
                f'<td style="padding:6px 11px;font-size:12px;text-align:center;'
                f'color:#facc15;font-weight:600">{bwr_disp}</td>'
                f'</tr>'
            )
        no_results_msg = ('<tr><td colspan="9" style="padding:20px;text-align:center;'
                           'color:#94a3b8;font-size:13px">No matches today</td></tr>')

        html_e = f"""<!DOCTYPE html><html><body style="margin:0;padding:0;
background:#f1f5f9;font-family:'Segoe UI',Arial,sans-serif">
<table width="100%" cellpadding="0" cellspacing="0"><tr><td align="center" style="padding:20px 10px">
<table width="100%" style="max-width:800px;background:#fff;border-radius:12px;
       overflow:hidden;box-shadow:0 4px 16px rgba(0,0,0,0.08)">
  <tr><td style="background:linear-gradient(135deg,#0f172a,#1e3a5f);padding:22px 28px">
<h1 style="margin:0;color:#60a5fa;font-size:20px;font-weight:700">
  📊 EMA8/SMA21/SMA50/SMA150 Retest + 2-Candle Reclaim
</h1>
<p style="margin:6px 0 0;color:#94a3b8;font-size:12px">
  {datetime.today().strftime('%Y-%m-%d %H:%M UTC')} &nbsp;·&nbsp;
  {cnt} match{'es' if cnt!=1 else ''} found
</p>
  </td></tr>
  <tr><td style="padding:14px 28px 4px;background:#0b1220">
<div style="background:#111827;border:1px solid #1f2937;border-radius:8px;padding:12px 16px">
  <p style="margin:0 0 6px;color:#93c5fd;font-size:12px;font-weight:700">
    📈 BACKTEST — LAST {CFG['backtest_lookback_days']} TRADING DAYS (~3 MONTHS), FULL UNIVERSE
  </p>
  <p style="margin:0;color:#cbd5e1;font-size:12px">
    {BT_SUMMARY['signals']} signals &nbsp;·&nbsp;
    <span style="color:#22c55e">{BT_SUMMARY['wins']} wins</span> &nbsp;·&nbsp;
    <span style="color:#ef4444">{BT_SUMMARY['losses']} losses</span> &nbsp;·&nbsp;
    {BT_SUMMARY['timeouts']} timeouts (excluded) &nbsp;·&nbsp;
    <b style="color:#facc15">Success rate: {(f"{BT_SUMMARY['win_rate_pct']:.1f}%" if BT_SUMMARY['win_rate_pct'] is not None else 'n/a')}</b>
  </p>
  <p style="margin:6px 0 0;color:#64748b;font-size:10px">
    Win = hit {CFG['backtest_reward_r']:.0f}:1 reward before stop-loss, within {CFG['backtest_holding_days']} trading days
  </p>
</div>
  </td></tr>
  <tr><td style="padding:16px">
<div style="overflow-x:auto;border-radius:8px;border:1px solid #e2e8f0">
  <table style="border-collapse:collapse;width:100%;min-width:600px">
    <thead><tr>{th_e}</tr></thead>
    <tbody>{rows_e or no_results_msg}</tbody>
  </table>
</div>
<p style="font-size:11px;color:#64748b;margin:8px 0 0">
  📎 Full results + full backtest trade log attached as CSV
</p>
  </td></tr>
  <tr><td style="background:#f8fafc;padding:12px 28px;
             border-top:1px solid #e2e8f0;text-align:center">
<p style="margin:0;color:#94a3b8;font-size:10px">
  ⚠️ Not financial advice &nbsp;·&nbsp; Auto-generated by GitHub Actions
</p>
  </td></tr>
</table>
</td></tr></table>
</body></html>"""

        plain_lines = [
            f"EMA8/SMA21/SMA50/SMA150 Retest + 2-Candle Reclaim — {datetime.today().strftime('%Y-%m-%d')}",
            f"{cnt} matches",
            "="*60,
            f"BACKTEST (last {CFG['backtest_lookback_days']} trading days, full universe):",
            f"  Signals: {BT_SUMMARY['signals']}  Wins: {BT_SUMMARY['wins']}  "
            f"Losses: {BT_SUMMARY['losses']}  Timeouts: {BT_SUMMARY['timeouts']}",
            f"  Success rate: " + (f"{BT_SUMMARY['win_rate_pct']:.1f}%"
                                    if BT_SUMMARY['win_rate_pct'] is not None else "n/a"),
            "="*60,
        ]
        if rl:
            for r in rl[:50]:
                ticker = r.get("Ticker","—")
                price  = r.get("Price",0) or 0
                sl     = r.get("Stop_Loss",0) or 0
                total  = r.get("Total",0) or 0
                ma     = r.get("Retested_MA","—")
                risk   = r.get("Risk_%",0) or 0
                bwr    = r.get("Backtest_WinRate_%")
                bwr_disp = f"{bwr:.1f}%" if bwr is not None else "n/a"
                plain_lines.append(
                    f"{ticker:<7} Entry:${float(price):.2f}  SL:${float(sl):.2f}  "
                    f"Total:{float(total):.0f}  MA:{ma}  Risk:{float(risk):+.1f}%  "
                    f"OwnBacktestWinRate:{bwr_disp}"
                )
        else:
            plain_lines.append("No matches today")
        plain_lines.append("\nFull results + full backtest trade log in CSV attachments.")
        plain_e = "\n".join(plain_lines)

        wr_disp = (f"{BT_SUMMARY['win_rate_pct']:.0f}%"
                   if BT_SUMMARY['win_rate_pct'] is not None else "n/a")
        subj = (f"📊 SMA Retest+Reclaim — {cnt} signal{'s' if cnt!=1 else ''} "
                f"(3M backtest: {wr_disp})"
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

    for attach_path in [csv_path, bt_csv_path]:
        if attach_path and os.path.exists(attach_path):
            try:
                with open(attach_path, "rb") as f:
                    part = MIMEBase("application", "octet-stream")
                    part.set_payload(f.read())
                encoders.encode_base64(part)
                part.add_header("Content-Disposition",
                    f"attachment; filename={os.path.basename(attach_path)}")
                msg.attach(part)
                sz = os.path.getsize(attach_path)
                print(f"[Email] 📎 Attached: {os.path.basename(attach_path)} ({sz:,} bytes)")
            except Exception as e:
                print(f"[Email] ⚠️  Attach failed for {attach_path}: {e}")

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
        print("         Generate one: myaccount.google.com/apppasswords")
        print("         → Google Account → Security → 2-Step Verification → App Passwords")
    except smtplib.SMTPException as e:
        print(f"[Email] ❌  SMTP error: {e}")
    except Exception as e:
        print(f"[Email] ❌  Unexpected error: {type(e).__name__}: {e}")

try:
    _send_email(results, fpath, bt_fpath)
except Exception as e:
    print(f"[Email] ❌  Unexpected top-level error: {type(e).__name__}: {e}")
    print("[Email]    Continuing — CSV and charts are still saved.")

if _IN_NOTEBOOK:
    try:
        from google.colab import files
        files.download(fpath); files.download(tv)
    except Exception: pass
else:
    print("  (CI: files in workspace, email sent)")

# ── Charts for top 5 ──────────────────────────────────────────
if results:
    top = results[:min(5,len(results))]
    fig, axes = plt.subplots(len(top),1,figsize=(15,5*len(top)),facecolor="#0f172a")
    if len(top)==1: axes=[axes]
    for ax, r in zip(axes, top):
        df_p    = r["_df"].tail(90).copy()  # shorter window — this is a short setup
        sma21_p  = r["_sma21"].reindex(df_p.index)
        sma50_p  = r["_sma50"].reindex(df_p.index)
        sma150_p = r["_sma150"].reindex(df_p.index)
        ema8_p   = r["_ema8"].reindex(df_p.index)
        ax.set_facecolor("#0f172a")
        ax.plot(df_p.index, df_p["Close"], color="#60a5fa", lw=1.6, label="Price", zorder=5)
        ax.plot(df_p.index, ema8_p,   color="#38bdf8", lw=1.1, ls="--", label="EMA8",   zorder=3)
        ax.plot(df_p.index, sma21_p,  color="#34d399", lw=1.1, ls="--", label="SMA21",  zorder=3)
        ax.plot(df_p.index, sma50_p,  color="#fbbf24", lw=1.3, ls="-.", label="SMA50",  zorder=3)
        ax.plot(df_p.index, sma150_p, color="#f87171", lw=1.3, ls=":",  label="SMA150", zorder=3)
        # mark candle B (today, signal bar) and the stop-loss level
        ax.scatter([df_p.index[-1]], [r["Price"]], color="#22c55e", s=60, zorder=6,
                   marker="^", label="Entry (Candle B close)")
        ax.axhline(r["Stop_Loss"], color="#ef4444", lw=1.0, ls="--", alpha=0.8,
                  label=f"Stop Loss ${r['Stop_Loss']:.2f}")
        ax.set_title(
            f"{r['Ticker']}  {r['Company']}  |  Entry ${r['Price']:.2f}  |  "
            f"SL ${r['Stop_Loss']:.2f} ({r['Risk_%']:+.1f}%)  |  "
            f"Score {r['Total']} (F{r['Fund']}+T{r['Tech']})  |  "
            f"Retested: {r['Retested_MA']}",
            color="#e2e8f0", fontsize=9, fontweight="bold", pad=7)
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))
        ax.tick_params(colors="#94a3b8", labelsize=9)
        for sp in ax.spines.values(): sp.set_edgecolor("#1e3a5f")
        ax.legend(loc="upper left", facecolor="#1e293b", labelcolor="#e2e8f0",
                  fontsize=7, framealpha=0.9, ncol=2)
        ax.grid(color="#1e3a5f", ls="--", lw=0.5, alpha=0.6)
    plt.suptitle(
        f"EMA8/SMA21/SMA50/SMA150 Retest + 2-Candle Reclaim  ·  "
        f"{datetime.today().strftime('%Y-%m-%d')}",
        color="#60a5fa", fontsize=12, fontweight="bold", y=1.001)
    plt.tight_layout()
    cp = os.path.join(out_dir, f"sma150_base_reclaim_chart_{ts}.png")
    plt.savefig(cp, dpi=150, bbox_inches="tight", facecolor="#0f172a")
    if _IN_NOTEBOOK: plt.show()
    else: plt.close()
    print(f"  📊 Chart → {cp}")
    if _IN_NOTEBOOK:
        try:
            from google.colab import files; files.download(cp)
        except Exception: pass

print("""
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  📋 SCORE BREAKDOWN  (80 total)
  Fund  0–50   8 fundamental metrics
  Tech  0–30   candle A/B distance from MA + EMA8 + volume surge

  📋 SIGNAL (checked against EMA8, SMA21, SMA50, AND SMA150 independently —
  matches if either satisfies it; all steps required)
  0) PRIOR UPTREND: price was above the MA a few bars before the
     retest (confirms pullback within an uptrend, not a breakdown)
  1) SMA50 RISING: SMA50 today > SMA50 sma_rising_lookback bars ago
  2) CANDLE A (yesterday): RED, closed BELOW the MA
  3) CANDLE B (today): GREEN, closed ABOVE the SAME MA AND above
     EMA8, simultaneously
  4) VOLUME: candle B volume > candle A volume

  📋 OUTPUT
  Price      = candle B (today) close — the entry price
  Stop_Loss  = candle A (yesterday) low
  Risk_%     = (Price - Stop_Loss) / Price
  Retested_MA = which MA (EMA8/SMA21/SMA50/SMA150) this fired against
  Matched_MAs = all MAs that matched at once, if more than one (confluence)

  📋 BACKTEST (new)
  Every ticker scanned (full NASDAQ universe, not just today's
  matches) is re-checked day-by-day over the last
  backtest_lookback_days (~3 months) for this same signal. Each
  historical signal found is simulated forward up to
  backtest_holding_days bars:
    WIN     = price hits backtest_reward_r : 1 reward (vs. the
              signal's own risk) before hitting its stop-loss
    LOSS    = stop-loss hit first
    TIMEOUT = neither hit within the holding window — excluded
              from the win-rate calc, reported separately
  Success rate % = wins / (wins + losses), i.e. resolved trades only.
  The scan-wide aggregate is printed at the end and in the email
  header. Each matching ticker also shows its OWN historical
  Backtest_WinRate_% (n/a if it had no resolved signals recently).
  Full trade-by-trade log is saved to
  sma150_base_reclaim_backtest_<timestamp>.csv and attached to the
  email alongside the main results CSV.

  💡 BEST SETUPS
  Total > 50           elite fundamental + technical combo
  Risk_% < 5            tight stop relative to entry
  Vol_Chg_% > 50         strong volume confirmation on the reclaim
  Fund > 35              genuinely strong business quality
  Backtest_WinRate_% > 50  this ticker's own pattern has worked recently

  ⚙️  TUNE IF 0 RESULTS
  min_tech_score                   10 → 6
  min_total_score                  15 → 5
  prior_uptrend_lookback             3 → 5
  sma_rising_lookback                5 → 10
  require_prior_uptrend           True → False
  require_sma50_rising            True → False
  require_volume_confirmation     True → False
  min_price                         2 → 1
  min_avg_volume                80000 → 50000

  ⚙️  BACKTEST TUNING
  backtest_lookback_days   63 → 126   (6 months instead of 3)
  backtest_holding_days    15 → 20
  backtest_reward_r        2.0 → 1.5  (easier win condition)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
""")
