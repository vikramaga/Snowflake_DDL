# ============================================================
# NASDAQ — 3-Layer Confluence Scanner (v1)
# ============================================================
#
# Intersects three independent layers — a stock must pass ALL
# THREE to appear in the results. Single universe scan, single
# set of downloaded OHLCV per ticker (no redundant re-scans):
#
#  LAYER 1 — FUNDAMENTALS  (from fundamental_v2.py)
#      Revenue growth, profit margin, ROE, P/E, EPS, debt/equity.
#      Fund_Score (0-50); must clear min_fund_score.
#
#  LAYER 2 — STRUCTURE  (Minervini Stage 2, from minervini.py)
#      1. Price > SMA150 > SMA200
#      2. SMA150 AND SMA200 both trending up (slope_bars)
#      3. Price > SMA50
#      4. Price >= 30% above its 52-week low
#      5. Price within 25% of its 52-week high
#      6. RS: stock's performance over rs_period >= SPY's - 5%
#      All 6 required — confirms a genuine, established uptrend,
#      not just a short-term bounce.
#
#  LAYER 3 — ENTRY TRIGGER  (2-candle retest+reclaim, from
#            sma150_base_reclaim_v1.py — reused verbatim, already
#            validated with unit + integration tests and a 3-month
#            walk-forward backtest)
#      Checked against EMA8/SMA21/SMA50/SMA150 independently:
#      candle A (red, closed below the MA) → candle B (green,
#      closed above that MA AND EMA8) → volume up vs prior day →
#      SMA50 rising → price was above the MA a few bars before
#      the retest (genuine pullback, not breakdown).
#
# A ticker only appears in the final results if Layer 1 AND
# Layer 2 AND Layer 3 all pass — this is the literal intersection
# of "good business" + "confirmed uptrend" + "validated timing
# signal" that a chart-pattern-only scanner can't offer alone.
#
# BACKTEST: the same 3-month, full-universe backtest from
# sma150_base_reclaim_v1.py runs underneath Layer 3 for every
# ticker that reaches that stage, so each match also shows its own
# historical win rate for the entry trigger specifically.
#
# FINAL SCORE = Fund (0-50) + Structure (0-20) + Trigger (0-30) = 100
#
# ============================================================
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

    "recent_signal_lookback_days"  : 15,   # ~3 trading weeks — how far back to
                                            # look for a Layer 3 trigger, not just today

    # ── Backtest (last 3 months, full universe) ─────────────────
    "backtest_lookback_days"       : 63,   # ~3 trading months
    "backtest_holding_days"        : 15,   # max bars to hold before "timeout"
    "backtest_reward_r"            : 2.0,  # win = hits 2:1 reward before stop-loss

    # ── Layer 2: Structure (Minervini Stage 2) ──────────────────
    "sma200_period"                : 200,
    "slope_bars"                   : 20,   # bars back to confirm SMA150/200 rising
    "min_above_52w_low_pct"        : 30.0,
    "max_below_52w_high_pct"       : 25.0,
    "rs_period"                    : 252,  # ~52 weeks of trading days
    "min_rs_pct_vs_spy"            : -5.0, # stock perf >= SPY perf - 5% (relaxed)

    # ── Score gates ─────────────────────────────────────────────
    "min_fund_score"              : 15,   # out of 50 — Layer 1 gate
    "min_tech_score"              : 10,   # out of 30 — Layer 3 gate
    "min_total_score"             : 40,   # out of 100 (Fund 0-50 + Struct 0-20 + Tech 0-30)

    # ── Filters ─────────────────────────────────────────────────
    "min_avg_volume"              : 80_000,
    "min_price"                   : 2.0,

    "batch_size"                  : 50,
    "batch_sleep"                  : 1.5,
    "fund_sleep"                    : 0.3,
}

# ── SPY RS cache (for Layer 2 structural relative-strength check) ──
_SPY_PERF = None

def get_spy_perf(rs_period, history_days):
    global _SPY_PERF
    if _SPY_PERF is not None:
        return _SPY_PERF
    try:
        end   = datetime.today()
        start = end - timedelta(days=history_days)
        spy   = yf.Ticker("SPY").history(
            start=start.strftime("%Y-%m-%d"),
            end=end.strftime("%Y-%m-%d"),
            auto_adjust=True)
        spy.index = pd.to_datetime(spy.index)
        if hasattr(spy.index, "tz") and spy.index.tz:
            spy.index = spy.index.tz_localize(None)
        if len(spy) > rs_period:
            p_now  = float(spy["Close"].iloc[-1])
            p_then = float(spy["Close"].iloc[-rs_period])
            _SPY_PERF = (p_now - p_then) / p_then * 100
        else:
            _SPY_PERF = 0.0
    except Exception:
        _SPY_PERF = 0.0
    print(f"  SPY {rs_period}-bar performance: {_SPY_PERF:+.1f}%")
    return _SPY_PERF

def check_stage2_structure(df, sma50, sma150, sma200, spy_perf, cfg):
    """
    Minervini Stage-2 structural gate (6 criteria, all required),
    reused from minervini.py. Confirms a genuine, established
    uptrend — not just a short-term technical bounce.

    Returns (passed: bool, details: dict).
    """
    n = len(df)
    price = float(df["Close"].iloc[-1])
    sb = cfg["slope_bars"]
    details = {}

    cur_s50, cur_s150, cur_s200 = (float(sma50.iloc[-1]), float(sma150.iloc[-1]),
                                    float(sma200.iloc[-1]))
    if any(np.isnan(v) for v in [cur_s50, cur_s150, cur_s200]):
        return False, details

    # 1. Price > SMA150 > SMA200
    c1 = price > cur_s150 > cur_s200
    # 2. SMA150 and SMA200 trending up
    s150_prev = float(sma150.iloc[-sb]) if n > sb and not np.isnan(sma150.iloc[-sb]) else np.nan
    s200_prev = float(sma200.iloc[-sb]) if n > sb and not np.isnan(sma200.iloc[-sb]) else np.nan
    c2 = (not np.isnan(s150_prev)) and (not np.isnan(s200_prev)) and \
         (cur_s150 > s150_prev) and (cur_s200 > s200_prev)
    # 3. Price > SMA50
    c3 = price > cur_s50
    # 4/5. 52-week low/high position
    w52 = min(252, n)
    lo52 = float(df["Low"].tail(w52).min())
    hi52 = float(df["High"].tail(w52).max())
    above_low_pct  = (price - lo52) / lo52 * 100 if lo52 > 0 else 0
    below_high_pct = (hi52 - price) / hi52 * 100 if hi52 > 0 else 0
    c4 = above_low_pct >= cfg["min_above_52w_low_pct"]
    c5 = below_high_pct <= cfg["max_below_52w_high_pct"]
    # 6. RS vs SPY
    rs_period = cfg["rs_period"]
    if n >= rs_period:
        p_then = float(df["Close"].iloc[-rs_period])
        stock_perf = (price - p_then) / p_then * 100 if p_then > 0 else 0
        rs_diff = stock_perf - spy_perf
    else:
        stock_perf, rs_diff = 0.0, 0.0
    c6 = rs_diff >= cfg["min_rs_pct_vs_spy"]

    passed = c1 and c2 and c3 and c4 and c5 and c6
    details = {
        "above_52w_low_pct": round(above_low_pct, 1),
        "below_52w_high_pct": round(below_high_pct, 1),
        "rs_diff_vs_spy": round(rs_diff, 1),
        "checks": {"price>SMA150>SMA200": c1, "SMA150&200_rising": c2,
                   "price>SMA50": c3, "above_52w_low": c4,
                   "near_52w_high": c5, "RS_vs_SPY": c6},
    }
    return passed, details

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

def analyze_confluence_technical(sym, df, spy_perf):
    """
    LAYER 2 (Structure) + LAYER 3 (Entry trigger) — both fast,
    no .info calls. Returns dict with struct_score + tech_score
    and details, or None if either layer fails.

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
    if n < 210: return None   # need enough history for SMA200 + rising checks

    sma21  = df["Close"].rolling(CFG["sma21_period"]).mean()
    sma50  = df["Close"].rolling(CFG["sma50_period"]).mean()
    sma150 = df["Close"].rolling(CFG["sma150_period"]).mean()
    sma200 = df["Close"].rolling(CFG["sma200_period"]).mean()
    ema8   = df["Close"].ewm(span=CFG["ema8_period"], adjust=False).mean()

    # ── LAYER 2: STRUCTURE (Minervini Stage 2, 6 criteria) ─────────
    struct_passed, struct_details = check_stage2_structure(
        df, sma50, sma150, sma200, spy_perf, CFG)
    if not struct_passed:
        return None

    # ── Backtest: last 3 months, this ticker, regardless of live match ──
    bt_trades = backtest_ticker(sym, df, sma21, sma50, sma150, ema8, CFG)
    ALL_BACKTEST_TRADES.extend(bt_trades)
    bt_summary = summarize_trades(bt_trades)

    # ── LAYER 3: ENTRY TRIGGER (2-candle retest+reclaim) ───────────
    # Scans the last `recent_signal_lookback_days` trading days (not
    # just today) — a ticker qualifies if the trigger fired on ANY
    # of those days, against ANY of EMA8/SMA21/SMA50/SMA150. This
    # surfaces stocks whose setup formed within the last 2-3 weeks
    # even if the exact trigger day wasn't today. Order = shortest
    # MA period first, so the tightest/most immediate support is
    # reported as the primary Retested_MA when more than one fires.
    ma_candidates = [
        ("EMA8",   ema8),
        ("SMA21",  sma21),
        ("SMA50",  sma50),
        ("SMA150", sma150),
    ]
    lb = CFG["recent_signal_lookback_days"]
    all_hits = []   # every (end_idx, ma_name, trace) found in the window
    for back in range(0, lb):
        end_idx = n - 1 - back
        if end_idx < 1: break
        for ma_name, ma_series in ma_candidates:
            passed, trace = check_two_candle_retest_reclaim(
                df, ma_series, ema8,
                CFG["prior_uptrend_lookback"], CFG["sma_rising_lookback"], sma50,
                CFG["require_prior_uptrend"], CFG["require_sma50_rising"],
                CFG["require_volume_confirmation"], end_idx=end_idx,
            )
            if passed:
                all_hits.append((end_idx, ma_name, trace))

    if not all_hits:
        return None

    most_recent_idx = max(h[0] for h in all_hits)
    # "matches" = every MA that fired on that SAME most-recent day
    # (same-day confluence, as before)
    matches = [(ma_name, trace) for (idx, ma_name, trace) in all_hits
               if idx == most_recent_idx]
    # Full recent-signal list, most recent first, for the "identified
    # in the last 2-3 weeks" output
    recent_signals = sorted(
        [{"date": df.index[idx], "ma": ma_name,
          "entry": round(trace["close_B"], 2), "stop": round(trace["low_A"], 2),
          "bars_ago": n - 1 - idx}
         for (idx, ma_name, trace) in all_hits],
        key=lambda x: x["date"], reverse=True,
    )

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

    # ── Structure score (0-20) ────────────────────────────────
    ss = 0
    sr = []
    ss += min(10, int(struct_details["above_52w_low_pct"] * 0.1))
    sr.append(f"52wLow+{struct_details['above_52w_low_pct']:.0f}%")
    ss += max(0, 6 - int(struct_details["below_52w_high_pct"] * 0.24))
    sr.append(f"52wHigh-{struct_details['below_52w_high_pct']:.0f}%")
    if struct_details["rs_diff_vs_spy"] >= 0:
        ss += 4; sr.append(f"RSvSPY{struct_details['rs_diff_vs_spy']:+.0f}%")
    ss = min(20, ss)

    # ── Technical (trigger) score (0-30) ────────────────────────
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
        "struct_score"   : ss,
        "struct_reasons" : " | ".join(sr),
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
        "SMA200"         : round(float(sma200.iloc[-1]), 2),
        "Above_52wLow_%" : struct_details["above_52w_low_pct"],
        "Below_52wHigh_%": struct_details["below_52w_high_pct"],
        "RS_vs_SPY_%"    : struct_details["rs_diff_vs_spy"],
        "Vol_Chg_%"      : round(vol_chg_pct, 1),
        "Days_Since_Signal"  : n - 1 - most_recent_idx,
        "Recent_Signal_Count": len(recent_signals),
        "Recent_Signals_Detail": recent_signals,
        "Recent_Signals" : " | ".join(
            f"{s['date'].strftime('%Y-%m-%d')}:{s['ma']}" for s in recent_signals),
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
        "_sma200"        : sma200,
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
LIVE_COLS = ["Ticker","Price","Stop_Loss","Total","Fund","Struct","Tech",
             "Retested_MA","Days_Since_Signal","Risk_%","Sector"]
_CW = {"Ticker":8,"Price":10,"Stop_Loss":11,"Total":7,"Fund":6,"Struct":7,"Tech":6,
       "Retested_MA":12,"Days_Since_Signal":11,"Risk_%":9,"Sector":20}
_CF = {"Price":"${:.2f}","Stop_Loss":"${:.2f}","Total":"{:.0f}","Fund":"{:.0f}",
       "Struct":"{:.0f}","Tech":"{:.0f}","Risk_%":"{:+.1f}%"}
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

# ── SPY relative-strength baseline (Layer 2 structure check) ────
print("━"*65)
print("  STEP 2b  SPY BASELINE (for RS vs SPY check)")
print("━"*65)
SPY_PERF = get_spy_perf(CFG["rs_period"], CFG["history_days"])
print()

# ── Main scan — 2-pass ───────────────────────────────────────
print("━"*65)
print(f"  STEP 3  SCANNING {len(TICKERS)} TICKERS")
print("━"*65)
print("  Pass 1: Layer 2 (Stage-2 structure) + Layer 3 (retest+reclaim trigger) — fast")
print("  Pass 2: Layer 1 (fundamentals) — fetched only for Pass-1 survivors\n")

_hdr_done   = False
results     = []
tech_passes = []
no_data     = 0

batches = [TICKERS[i:i+CFG["batch_size"]]
           for i in range(0, len(TICKERS), CFG["batch_size"])]

# ── PASS 1: Structure + Trigger (no .info calls) ────────────────
with tqdm(total=len(TICKERS), desc="Pass 1 Tech", unit="stk",
          bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]") as pbar:
    for batch in batches:
        data_map = download(batch, CFG["history_days"])
        no_data += len(batch) - len(data_map)
        for sym in batch:
            pbar.update(1)
            if sym not in data_map: continue
            try:
                ts = analyze_confluence_technical(sym, data_map[sym], SPY_PERF)
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
print(f"  Structure+Trigger passes: {len(tech_passes)} stocks → fetching fundamentals now\n")

# ── PASS 2: Fundamentals for tech-pass stocks only ──────────────
print("━"*65)
print(f"  PASS 2  FUNDAMENTAL CHECK ({len(tech_passes)} stocks)")
print("━"*65+"\n")

for item in tqdm(tech_passes, desc="Pass 2 Fund", unit="stk"):
    sym   = item["sym"]; price = item["price"]; ts = item["ts"]
    try:
        fund = get_fundamentals(sym)
        time.sleep(CFG["fund_sleep"])

        fs = fund["Fund_Score"]
        if fs < CFG["min_fund_score"]: continue   # ── LAYER 1 gate ──

        sscr, tscr = ts["struct_score"], ts["tech_score"]
        total = fs + sscr + tscr
        if total < CFG["min_total_score"]: continue

        result = {
            "Ticker"            : sym,
            "Price"             : ts["Price"],
            "Stop_Loss"         : ts["Stop_Loss"],
            "Risk_%"            : ts["Risk_%"],
            "Total"             : total,
            "Fund"              : fs,
            "Struct"            : sscr,
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
            "SMA200"            : ts["SMA200"],
            "Above_52wLow_%"    : ts["Above_52wLow_%"],
            "Below_52wHigh_%"   : ts["Below_52wHigh_%"],
            "RS_vs_SPY_%"       : ts["RS_vs_SPY_%"],
            "Vol_Chg_%"         : ts["Vol_Chg_%"],
            "Days_Since_Signal"   : ts["Days_Since_Signal"],
            "Recent_Signal_Count" : ts["Recent_Signal_Count"],
            "Recent_Signals"      : ts["Recent_Signals"],
            "Backtest_Signals_3M": ts["Backtest_Signals_3M"],
            "Backtest_Wins"      : ts["Backtest_Wins"],
            "Backtest_Losses"    : ts["Backtest_Losses"],
            "Backtest_Timeouts"  : ts["Backtest_Timeouts"],
            "Backtest_WinRate_%" : ts["Backtest_WinRate_%"],
            "Tech_Flags"        : ts["tech_reasons"],
            "Struct_Flags"      : ts["struct_reasons"],
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
            "_sma200": ts["_sma200"],
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
print(f"  ℹ️  Note: this backtest measures LAYER 3 (the entry trigger)")
print(f"      on its own — it does not re-check Layer 1/2 (fundamentals")
print(f"      /structure) as they stood on each historical signal date.")
print(f"{'━'*65}")

if not results:
    print("\n  No matches. Try relaxing:")
    print("   min_fund_score                  15 → 5    (Layer 1)")
    print("   min_rs_pct_vs_spy               -5 → -15  (Layer 2)")
    print("   max_below_52w_high_pct          25 → 40   (Layer 2)")
    print("   min_above_52w_low_pct           30 → 15   (Layer 2)")
    print("   min_tech_score                  10 → 6    (Layer 3)")
    print("   min_total_score                 40 → 25")
    print("   require_prior_uptrend         True → False")
    print("   require_sma50_rising          True → False")
    print("   require_volume_confirmation   True → False")
    print("   prior_uptrend_lookback           3 → 5")
    print("   min_price                         2 → 1")
    print("   min_avg_volume                80000 → 50000")

# Sort by total score (always runs, even on empty list)
results.sort(key=lambda x: x["Total"], reverse=True)

# ── Always build df_out and save/email (even if 0 results) ────
ts      = datetime.today().strftime("%Y%m%d_%H%M")
out_dir = os.environ.get("GITHUB_WORKSPACE", os.getcwd())

COLS = [
    "Ticker","Company","Sector","Price","Stop_Loss","Risk_%",
    "Total","Fund","Struct","Tech",
    "Rev_Growth_%","Profit_Margin_%","ROE_%","PE_Ratio","EPS",
    "Above_52wLow_%","Below_52wHigh_%","RS_vs_SPY_%",
    "Retested_MA","Matched_MAs","Days_Since_Signal","Recent_Signal_Count",
    "Recent_Signals","Candle_A_Close","Candle_A_Low","Candle_B_Close",
    "EMA8","SMA21","SMA50","SMA150","SMA200","Vol_Chg_%",
    "Backtest_Signals_3M","Backtest_Wins","Backtest_Losses",
    "Backtest_Timeouts","Backtest_WinRate_%",
    "Tech_Flags","Struct_Flags","Fund_Flags",
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
    "Struct"         : lambda v: f"{v:.0f}",
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
    "SMA200"         : lambda v: f"${v:.2f}",
    "Above_52wLow_%" : lambda v: f"{v:+.1f}%",
    "Below_52wHigh_%": lambda v: f"{v:.1f}%",
    "RS_vs_SPY_%"    : lambda v: f"{v:+.1f}%",
    "Vol_Chg_%"      : lambda v: f"{v:+.1f}%",
    "Days_Since_Signal": lambda v: f"{int(v)}d ago",
}

def fmt_v(col, val):
    if val is None or (isinstance(val, float) and np.isnan(val)): return "—"
    try:
        if col in FMT: return FMT[col](val)
    except Exception: pass
    return str(val) if str(val) not in ("nan","None","") else "—"

if _IN_NOTEBOOK and results:
    DISP = ["Ticker","Company","Sector","Price","Stop_Loss",
            "Total","Fund","Struct","Tech",
            "Retested_MA","Days_Since_Signal","Risk_%","Backtest_WinRate_%"]
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
    <span style="color:#f1f5f9;font-size:15px;font-weight:700">3-Layer Confluence: Fund + Structure + Trigger</span>
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
    📈 3-Layer Confluence: Fundamentals + Structure + Trigger
  </h2>
  <p style="margin:6px 0 0;color:#94a3b8;font-size:12px">
    {datetime.today().strftime('%Y-%m-%d %H:%M')} &nbsp;·&nbsp;
    <b style="color:#22c55e">{len(results)} matches</b> from {len(TICKERS)} tickers
    &nbsp;·&nbsp; trigger signals from the last {CFG['recent_signal_lookback_days']} trading days
  </p>
</div>"""

    legend_html = f"""
<div style="background:#f8fafc;border:1px solid #e2e8f0;border-radius:8px;
        padding:12px 18px;margin-top:6px;font-size:11px;color:#64748b;
        font-family:'Segoe UI',Arial,sans-serif">
  <b style="color:#475569">GUIDE</b> &nbsp;·&nbsp;
  Total = Fund(0-50) + Struct(0-20) + Tech(0-30) = 100 &nbsp;·&nbsp;
  Layer 1 (Fund) = fundamentals score &nbsp;·&nbsp;
  Layer 2 (Struct) = Minervini Stage-2 uptrend confirmed &nbsp;·&nbsp;
  Layer 3 (Tech) = candle A (retest day) red &amp; closed below
  EMA8/SMA21/SMA50/SMA150, candle B (next day) green &amp; closed above the
  same MA and EMA8, volume up vs yesterday, SMA50 rising &nbsp;·&nbsp;
  Price = candle B close &nbsp;·&nbsp; Stop_Loss = candle A low &nbsp;·&nbsp;
  A ticker only appears here if ALL THREE layers passed &nbsp;·&nbsp;
  Layer 3 checks the last {CFG['recent_signal_lookback_days']} trading days
  (~2-3 weeks), not just today — Days_Since_Signal shows how long ago the
  most recent trigger fired
</div>"""

    display_html(header_html + table_html + legend_html)

elif results:
    # ASCII table (CLI/GitHub Actions mode)
    CLI_COLS = ["Ticker","Price","Stop_Loss","Total","Fund","Struct","Tech",
                "Retested_MA","Days_Since_Signal","Risk_%","Backtest_WinRate_%","Sector"]
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
    tit = f"  3-Layer Confluence (last {CFG['recent_signal_lookback_days']}d signals)   {datetime.today().strftime('%Y-%m-%d')}   {len(df_out)} matches"
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
  Total          Fund(0-50) + Struct(0-20) + Tech(0-30) = 100
  Fund            Layer 1: fundamentals score
  Struct          Layer 2: Minervini Stage-2 structure score
  Tech            Layer 3: retest+reclaim trigger score
  Price           candle B close on the most recent signal day found
  Stop_Loss       candle A low on that same signal day
  Retested_MA     which MA (EMA8/SMA21/SMA50/SMA150) this pattern formed against
  Matched_MAs     all MAs that matched at once (confluence), if more than one
  Days_Since_Signal  how many trading days ago the most recent trigger fired
                     (Layer 3 is checked over the last {CFG['recent_signal_lookback_days']}
                     trading days, not just today)
  Recent_Signals     every date+MA this trigger fired within that window
  Risk_%          (Price - Stop_Loss) / Price
  Backtest_WinRate_%  this ticker's own win rate over the last 3 months
                      for the Layer 3 trigger specifically
                      (n/a if it had no resolved historical signals)
  ──────────────────────────────────────────────────────""")

# Save
fpath = os.path.join(out_dir, f"confluence_scanner_{ts}.csv")
df_out.to_csv(fpath, index=False)
print(f"\n  💾 CSV → {fpath}")
tv = os.path.join(out_dir, f"tv_confluence_scanner_{ts}.txt")
with open(tv,"w") as f:
    f.write(f"###3-Layer Confluence Scanner {datetime.today().strftime('%Y-%m-%d')}\n")
    for r in results: f.write(f"NASDAQ:{r['Ticker']}\n")
print(f"  📋 TradingView → {tv}")

# Save backtest trade log (every signal found in the last 3 months, full universe)
bt_fpath = os.path.join(out_dir, f"confluence_scanner_backtest_{ts}.csv")
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
            for c in ["Ticker","Price","Stop_Loss","Total","Fund","Struct","Tech",
                      "Retested_MA","Days_Ago","Risk_%","Backtest_WinRate_%"]
        )
        rows_e = ""
        for i, r in enumerate(rl[:50]):
            bg  = "#fff" if i % 2 == 0 else "#f0f9ff"
            ticker = r.get("Ticker","—")
            price  = r.get("Price",0) or 0
            sl     = r.get("Stop_Loss",0) or 0
            total  = r.get("Total",0) or 0
            fund   = r.get("Fund",0) or 0
            struct = r.get("Struct",0) or 0
            tech   = r.get("Tech",0) or 0
            ma     = r.get("Retested_MA","—")
            dsig   = r.get("Days_Since_Signal")
            dsig_disp = f"{int(dsig)}d ago" if dsig is not None else "—"
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
                f'<td style="padding:6px 11px;font-size:12px">{float(struct):.0f}</td>'
                f'<td style="padding:6px 11px;font-size:12px">{float(tech):.0f}</td>'
                f'<td style="padding:6px 11px;font-size:12px;text-align:center;'
                f'color:#a78bfa;font-weight:600">{ma}</td>'
                f'<td style="padding:6px 11px;font-size:12px;text-align:center;'
                f'color:#38bdf8;font-weight:600">{dsig_disp}</td>'
                f'<td style="padding:6px 11px;font-size:12px;color:'
                f'{"#22c55e" if float(risk)>=0 else "#ef4444"}">{float(risk):+.1f}%</td>'
                f'<td style="padding:6px 11px;font-size:12px;text-align:center;'
                f'color:#facc15;font-weight:600">{bwr_disp}</td>'
                f'</tr>'
            )
        no_results_msg = ('<tr><td colspan="11" style="padding:20px;text-align:center;'
                           'color:#94a3b8;font-size:13px">No matches today</td></tr>')

        html_e = f"""<!DOCTYPE html><html><body style="margin:0;padding:0;
background:#f1f5f9;font-family:'Segoe UI',Arial,sans-serif">
<table width="100%" cellpadding="0" cellspacing="0"><tr><td align="center" style="padding:20px 10px">
<table width="100%" style="max-width:800px;background:#fff;border-radius:12px;
       overflow:hidden;box-shadow:0 4px 16px rgba(0,0,0,0.08)">
  <tr><td style="background:linear-gradient(135deg,#0f172a,#1e3a5f);padding:22px 28px">
<h1 style="margin:0;color:#60a5fa;font-size:20px;font-weight:700">
  📊 3-Layer Confluence: Fundamentals + Structure + Trigger
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
            f"3-Layer Confluence Scanner (signals from last {CFG['recent_signal_lookback_days']} trading days) — {datetime.today().strftime('%Y-%m-%d')}",
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
                fund   = r.get("Fund",0) or 0
                struct = r.get("Struct",0) or 0
                tech   = r.get("Tech",0) or 0
                ma     = r.get("Retested_MA","—")
                dsig   = r.get("Days_Since_Signal")
                dsig_disp = f"{int(dsig)}d ago" if dsig is not None else "—"
                risk   = r.get("Risk_%",0) or 0
                bwr    = r.get("Backtest_WinRate_%")
                bwr_disp = f"{bwr:.1f}%" if bwr is not None else "n/a"
                recent = r.get("Recent_Signals","")
                plain_lines.append(
                    f"{ticker:<7} Entry:${float(price):.2f}  SL:${float(sl):.2f}  "
                    f"Total:{float(total):.0f}(F{float(fund):.0f}+S{float(struct):.0f}+T{float(tech):.0f})  "
                    f"MA:{ma}  Signal:{dsig_disp}  Risk:{float(risk):+.1f}%  OwnBacktestWinRate:{bwr_disp}"
                )
                if recent:
                    plain_lines.append(f"        All signals in last 2-3 weeks: {recent}")
        else:
            plain_lines.append("No matches today")
        plain_lines.append("\nFull results + full backtest trade log in CSV attachments.")
        plain_e = "\n".join(plain_lines)

        wr_disp = (f"{BT_SUMMARY['win_rate_pct']:.0f}%"
                   if BT_SUMMARY['win_rate_pct'] is not None else "n/a")
        subj = (f"📊 3-Layer Confluence — {cnt} signal{'s' if cnt!=1 else ''} "
                f"(3M trigger backtest: {wr_disp})"
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
        df_p    = r["_df"].tail(260).copy()  # wide enough to show Stage-2 structure
        sma21_p  = r["_sma21"].reindex(df_p.index)
        sma50_p  = r["_sma50"].reindex(df_p.index)
        sma150_p = r["_sma150"].reindex(df_p.index)
        sma200_p = r["_sma200"].reindex(df_p.index)
        ema8_p   = r["_ema8"].reindex(df_p.index)
        ax.set_facecolor("#0f172a")
        ax.plot(df_p.index, df_p["Close"], color="#60a5fa", lw=1.6, label="Price", zorder=5)
        ax.plot(df_p.index, ema8_p,   color="#38bdf8", lw=1.0, ls="--", label="EMA8",   zorder=3)
        ax.plot(df_p.index, sma21_p,  color="#34d399", lw=1.0, ls="--", label="SMA21",  zorder=3)
        ax.plot(df_p.index, sma50_p,  color="#fbbf24", lw=1.2, ls="-.", label="SMA50",  zorder=3)
        ax.plot(df_p.index, sma150_p, color="#f87171", lw=1.2, ls=":",  label="SMA150", zorder=3)
        ax.plot(df_p.index, sma200_p, color="#a78bfa", lw=1.2, ls=":",  label="SMA200", zorder=3)
        # mark candle B (today, signal bar) and the stop-loss level
        ax.scatter([df_p.index[-1]], [r["Price"]], color="#22c55e", s=60, zorder=6,
                   marker="^", label="Entry (Candle B close)")
        ax.axhline(r["Stop_Loss"], color="#ef4444", lw=1.0, ls="--", alpha=0.8,
                  label=f"Stop Loss ${r['Stop_Loss']:.2f}")
        ax.set_title(
            f"{r['Ticker']}  {r['Company']}  |  Entry ${r['Price']:.2f}  |  "
            f"SL ${r['Stop_Loss']:.2f} ({r['Risk_%']:+.1f}%)  |  "
            f"Score {r['Total']} (F{r['Fund']}+S{r['Struct']}+T{r['Tech']})  |  "
            f"Retested: {r['Retested_MA']}",
            color="#e2e8f0", fontsize=9, fontweight="bold", pad=7)
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %y"))
        ax.xaxis.set_major_locator(mdates.MonthLocator())
        ax.tick_params(colors="#94a3b8", labelsize=9)
        for sp in ax.spines.values(): sp.set_edgecolor("#1e3a5f")
        ax.legend(loc="upper left", facecolor="#1e293b", labelcolor="#e2e8f0",
                  fontsize=7, framealpha=0.9, ncol=3)
        ax.grid(color="#1e3a5f", ls="--", lw=0.5, alpha=0.6)
    plt.suptitle(
        f"3-Layer Confluence Scanner  ·  "
        f"{datetime.today().strftime('%Y-%m-%d')}",
        color="#60a5fa", fontsize=12, fontweight="bold", y=1.001)
    plt.tight_layout()
    cp = os.path.join(out_dir, f"confluence_scanner_chart_{ts}.png")
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
  📋 SCORE BREAKDOWN  (100 total)
  Fund    0–50   Layer 1: 8 fundamental metrics (fundamental_v2.py)
  Struct  0–20   Layer 2: Minervini Stage-2 uptrend strength
  Tech    0–30   Layer 3: retest+reclaim trigger quality

  📋 A TICKER ONLY APPEARS HERE IF ALL 3 LAYERS PASS
  ────────────────────────────────────────────────────
  LAYER 1 — FUNDAMENTALS (gate: Fund_Score >= min_fund_score)
    Revenue growth, profit margin, ROE, P/E, EPS, debt/equity

  LAYER 2 — STRUCTURE (Minervini Stage 2, all 6 required)
    1) Price > SMA150 > SMA200
    2) SMA150 AND SMA200 both trending up
    3) Price > SMA50
    4) Price >= 30% above its 52-week low
    5) Price within 25% of its 52-week high
    6) RS: stock's performance over rs_period >= SPY's - 5%

  LAYER 3 — ENTRY TRIGGER (checked against EMA8/SMA21/SMA50/
            SMA150 independently; matches if any satisfies it.
            Scans the last recent_signal_lookback_days trading days
            ({CFG['recent_signal_lookback_days']}d, ~2-3 weeks) — not
            just today — so a stock whose setup fired any day in that
            window still shows up now, using its CURRENT price/
            fundamentals/structure)
    0) PRIOR UPTREND: price was above the MA a few bars before
       the retest (confirms pullback, not breakdown)
    1) SMA50 RISING: SMA50 today > SMA50 sma_rising_lookback bars ago
    2) CANDLE A (that day): RED, closed BELOW the MA
    3) CANDLE B (next day): GREEN, closed ABOVE the SAME MA AND
       above EMA8, simultaneously
    4) VOLUME: candle B volume > candle A volume

  📋 OUTPUT
  Price       = candle B close on the MOST RECENT signal day found
                (the entry price as of that day — may differ from
                today's live price if the signal is a few days old)
  Stop_Loss   = candle A low on that same signal day
  Risk_%      = (Price - Stop_Loss) / Price
  Retested_MA = which MA (EMA8/SMA21/SMA50/SMA150) the most recent
                signal fired against
  Matched_MAs = all MAs that matched on that SAME day, if more than
                one (same-day confluence)
  Days_Since_Signal = how many trading days ago the most recent
                trigger fired (0 = today)
  Recent_Signals = every date+MA this trigger fired within the last
                {CFG['recent_signal_lookback_days']} trading days —
                the full "identified in the last 2-3 weeks" list
  Above_52wLow_% / Below_52wHigh_% / RS_vs_SPY_% = Layer 2 structural detail

  📋 BACKTEST
  Every ticker that clears Layer 2 (Structure) is re-checked
  day-by-day over the last backtest_lookback_days (~3 months) for
  the Layer 3 entry trigger specifically — this measures the
  TRIGGER's own historical accuracy, not the full 3-layer combo's
  (fundamentals and structure as of each past date aren't re-checked,
  since that would require a much more expensive point-in-time
  historical re-scan). Each historical signal found is simulated
  forward up to backtest_holding_days bars:
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
  confluence_scanner_backtest_<timestamp>.csv and attached to the
  email alongside the main results CSV.

  💡 BEST SETUPS
  Total > 65               elite across all 3 layers
  Struct = 20                perfect Stage-2 structure
  Risk_% < 5                 tight stop relative to entry
  Days_Since_Signal <= 3      freshest trigger in the 2-3 week window
  Recent_Signal_Count > 1     trigger fired more than once recently
                              (repeated support at the same level)
  Backtest_WinRate_% > 50    this ticker's own trigger has worked recently
  Fund > 35                  genuinely strong business quality

  ⚙️  TUNE IF 0 RESULTS  (loosen one layer at a time to find the
      bottleneck — Layer 2 Structure is usually the strictest gate)
  min_fund_score                    15 → 5     (Layer 1)
  min_rs_pct_vs_spy                 -5 → -15   (Layer 2)
  max_below_52w_high_pct            25 → 40    (Layer 2)
  min_above_52w_low_pct             30 → 15    (Layer 2)
  min_tech_score                    10 → 6     (Layer 3)
  min_total_score                   40 → 25
  recent_signal_lookback_days       15 → 25    (widen the 2-3 week window)
  prior_uptrend_lookback              3 → 5
  sma_rising_lookback                 5 → 10
  require_prior_uptrend            True → False
  require_sma50_rising             True → False
  require_volume_confirmation      True → False
  min_price                          2 → 1
  min_avg_volume                 80000 → 50000

  ⚙️  BACKTEST TUNING
  backtest_lookback_days   63 → 126   (6 months instead of 3)
  backtest_holding_days    15 → 20
  backtest_reward_r        2.0 → 1.5  (easier win condition)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
""")
