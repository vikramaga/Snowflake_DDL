# ============================================================
# NASDAQ — Master Technical Scanner (v1)
# ============================================================
#
# Synthesizes the best-validated signal families from across this
# entire repo into ONE ranked scan — so a single run replaces
# needing to run the individual specialist scanners separately.
#
# Unlike most of the specialist scanners (which AND-gate a handful
# of narrow conditions and can legitimately return 0 matches), this
# is a WEIGHTED COMPOSITE SCORE (0-100): every stock gets partial
# credit for each signal family it satisfies, then the highest-
# scoring stocks are the "best technical setups" — no single missing
# condition zeroes out an otherwise strong stock.
#
# SIGNAL FAMILIES (each contributes points independently):
#
#   1. LONG-TERM TREND HEALTH (0-20) — from stagev2.py/minervini.py:
#      SMA50 > SMA150 > SMA200, all rising; price within 25% of its
#      52-week high; price at least 20% above its 52-week low.
#
#   2. RELATIVE STRENGTH vs SPY (0-15) — from stagev2.py: 6-month
#      performance vs SPY's own 6-month performance.
#
#   3. SHORT-TERM MA STACK (0-15) — from macompress.py/smasupport.py:
#      JMA > EMA8 > SMA21 > SMA50, tightly compressed and/or rising —
#      a coiled short-term structure riding above the long-term trend.
#
#   4. ENTRY TRIGGER (0-20) — the validated 2-candle retest+reclaim
#      pattern from sma150_base_reclaim_v1.py, reused verbatim,
#      checked against EMA8/SMA21/SMA50/SMA150, scanned over the
#      last entry_trigger_lookback_days (not just today).
#
#   5. MOMENTUM (0-10) — RSI(14) above 50 and rising vs its own
#      9-period moving average (from cluster_reclaim_rsi_v1.py /
#      rsi_multi_tf_reversal_v1.py).
#
#   6. VOLUME CONFIRMATION (0-10) — recent volume above its own
#      20-day average (broad volume-interest confirmation, echoing
#      the "volume > prior day/avg" requirement present in nearly
#      every specialist scanner in this repo).
#
#   7. FUNDAMENTALS (0-10, bonus, Pass 2 only) — the validated
#      get_fundamentals() from fundamental_v2.py, reused verbatim,
#      fetched only for technical survivors to keep this efficient.
#      Business quality is a bonus on top of technical strength here,
#      not a hard gate — this scanner is about "best technical setup"
#      first and foremost.
#
# DATA — only 1 download per ticker (~750 days), single indicator
# pass. Fundamentals (the only *args, i.e. .info fetch) run in a
# quick Pass 2, for technical survivors only.
#
# OUTPUT: Entry_Price / Stop_Loss reuse the entry-trigger candle's
# levels when a trigger fired; otherwise fall back to today's
# close / a recent swing-low proxy, since a stock can score well
# here even without a fresh trigger today.
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
    "history_days"               : 750,   # gives enough room for 52w metrics + RS

    # ── Fundamental thresholds (reused from fundamental_v2.py) ──────
    "min_revenue_growth_pct"     : 0.0,
    "min_profit_margin_pct"      : 0.0,
    "max_debt_to_equity"         : 2.0,
    "min_current_ratio"          : 1.0,
    "min_roe_pct"                : 10.0,
    "max_pe_ratio"                : 50.0,
    "high_growth_threshold_pct"  : 15.0,

    # ── Indicator periods ────────────────────────────────────────
    "ema8_period"                  : 8,
    "sma21_period"                 : 21,
    "sma50_period"                 : 50,
    "sma150_period"                : 150,
    "sma200_period"                : 200,
    "jma_period"                   : 13,
    "jma_phase"                    : 40,
    "rsi_period"                   : 14,
    "rsi_avg_period"               : 9,

    # ── Layer 1: Long-term trend health (0-20) ──────────────────────
    "trend_slope_lookback"         : 20,
    "max_off_52w_high_pct"         : 25.0,
    "min_above_52w_low_pct"        : 20.0,

    # ── Layer 2: Relative strength vs SPY (0-15) ─────────────────────
    "rs_lookback_days"             : 126,   # ~6 months

    # ── Layer 3: Short-term MA stack (0-15) ──────────────────────────
    "stack_compression_pct"        : 5.0,   # max-min spread among
                                             # {JMA,EMA8,SMA21} as % of price

    # ── Layer 4: Entry trigger (0-20) — same pattern as
    #    sma150_base_reclaim_v1.py, reused verbatim ──────────────────
    "prior_uptrend_lookback"       : 3,
    "sma_rising_lookback"          : 5,
    "require_prior_uptrend"        : True,
    "require_sma50_rising"         : True,
    "require_volume_confirmation"  : True,
    "entry_trigger_lookback_days"  : 15,

    # ── Layer 5: Momentum (0-10) ──────────────────────────────────
    "rsi_min_level"                 : 50,

    # ── Layer 6: Volume confirmation (0-10) ──────────────────────────
    "volume_surge_lookback"        : 20,

    # ── Score gates ─────────────────────────────────────────────
    "min_total_score"             : 45,   # out of 100 — only show genuinely
                                           # strong composite setups

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
def calc_jma(series, period=13, phase=40, power=2):
    """JMA approximation — adaptive EMA with corrected steady-state gain."""
    n      = len(series)
    vals   = series.values.astype(float)
    result = np.full(n, np.nan)
    phase_ratio = phase / 100.0 + 1.5
    alpha = 2.0 / (period + 1.0)
    beta  = alpha * phase_ratio
    first_valid = 0
    for i in range(n):
        if not np.isnan(vals[i]):
            first_valid = i
            break
    e0 = e1 = e2 = vals[first_valid]
    result[first_valid] = e0
    for i in range(first_valid + 1, n):
        v   = vals[i]
        e0  = (1 - alpha) * e0 + alpha * v
        e1  = (v - e0) * (1 - beta) + beta * e1
        e2  = (1 - alpha) * e2 + alpha * (e0 + e1)
        result[i] = e2
    return pd.Series(result, index=series.index)

def calc_rsi(close, period=14):
    """Standard Wilder-smoothed RSI."""
    delta = close.diff()
    gain  = delta.clip(lower=0)
    loss  = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1/period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1/period, min_periods=period, adjust=False).mean()
    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))

# ── Layer 1: Long-term trend health (0-20) ────────────────────────
def score_trend_health(df, sma50, sma150, sma200, cfg):
    price = float(df["Close"].iloc[-1])
    s50, s150, s200 = float(sma50.iloc[-1]), float(sma150.iloc[-1]), float(sma200.iloc[-1])
    if any(np.isnan(v) for v in [s50, s150, s200]):
        return 0, [], {}

    n = len(df); sb = cfg["trend_slope_lookback"]
    pts, reasons = 0, []
    details = {"sma50": s50, "sma150": s150, "sma200": s200}

    if s50 > s150 > s200:
        pts += 8; reasons.append("MAStack")
    if n > sb and not np.isnan(sma150.iloc[-sb]) and not np.isnan(sma200.iloc[-sb]):
        if s150 > float(sma150.iloc[-sb]) and s200 > float(sma200.iloc[-sb]):
            pts += 5; reasons.append("MAsRising")

    w52 = min(252, n)
    hi52 = float(df["High"].tail(w52).max())
    lo52 = float(df["Low"].tail(w52).min())
    off_high_pct = (hi52 - price) / hi52 * 100 if hi52 > 0 else 999
    above_low_pct = (price - lo52) / lo52 * 100 if lo52 > 0 else 0
    details["off_52w_high_pct"] = off_high_pct
    details["above_52w_low_pct"] = above_low_pct

    if off_high_pct <= cfg["max_off_52w_high_pct"]:
        pts += 4; reasons.append(f"Near52wHigh(-{off_high_pct:.0f}%)")
    if above_low_pct >= cfg["min_above_52w_low_pct"]:
        pts += 3; reasons.append(f"Above52wLow(+{above_low_pct:.0f}%)")

    return min(20, pts), reasons, details

# ── Layer 2: Relative strength vs SPY (0-15) ───────────────────────
def score_relative_strength(df, spy_perf_pct, cfg):
    n = len(df); lb = cfg["rs_lookback_days"]
    if n <= lb:
        return 0, [], {}
    price = float(df["Close"].iloc[-1])
    price_then = float(df["Close"].iloc[-lb])
    stock_perf = (price - price_then) / price_then * 100 if price_then > 0 else 0
    rs_diff = stock_perf - spy_perf_pct
    pts = 0; reasons = []
    if rs_diff >= 0:
        pts += min(15, 7 + rs_diff * 0.3)
        reasons.append(f"RSvSPY{rs_diff:+.0f}%")
    return round(min(15, max(0, pts))), reasons, {"stock_perf_pct": stock_perf, "rs_diff_pct": rs_diff}

# ── Layer 3: Short-term MA stack, compressed and rising (0-15) ─────
def score_ma_stack(df, jma, ema8, sma21, sma50, cfg):
    price = float(df["Close"].iloc[-1])
    j, e8, s21, s50 = jma.iloc[-1], ema8.iloc[-1], sma21.iloc[-1], sma50.iloc[-1]
    if any(np.isnan(v) for v in [j, e8, s21, s50]):
        return 0, [], {}
    j, e8, s21, s50 = float(j), float(e8), float(s21), float(s50)
    pts = 0; reasons = []
    stacked = j > e8 > s21 > s50
    if stacked:
        pts += 7; reasons.append("StackedBullish")
    vals = [j, e8, s21]
    compression_pct = (max(vals) - min(vals)) / price * 100 if price > 0 else 999
    if compression_pct <= cfg["stack_compression_pct"]:
        comp_pts = max(0, min(8, 8 - compression_pct * (8/cfg["stack_compression_pct"])))
        pts += comp_pts
        reasons.append(f"Compressed({compression_pct:.1f}%)")
    return round(min(15, pts)), reasons, {"compression_pct": compression_pct, "stacked": stacked}

# ── Layer 4: Entry trigger (0-20) — reuses check_two_candle_retest_reclaim verbatim ──
def score_entry_trigger(df, ema8, sma21, sma50, sma150, cfg):
    n = len(df)
    lb = cfg["entry_trigger_lookback_days"]
    ma_candidates = [("EMA8", ema8), ("SMA21", sma21), ("SMA50", sma50), ("SMA150", sma150)]
    for back in range(0, lb):
        end_idx = (n - 1) - back
        if end_idx < 5: break
        for ma_name, ma_series in ma_candidates:
            passed, trace = check_two_candle_retest_reclaim(
                df, ma_series, ema8,
                cfg["prior_uptrend_lookback"], cfg["sma_rising_lookback"], sma50,
                cfg["require_prior_uptrend"], cfg["require_sma50_rising"],
                cfg["require_volume_confirmation"], end_idx=end_idx,
            )
            if passed:
                pts = 12
                days_ago = (n - 1) - end_idx
                pts += max(0, 8 - days_ago)   # freshness bonus, up to +8
                reasons = [f"Retest+Reclaim({ma_name},{days_ago}dAgo)"]
                return min(20, round(pts)), reasons, {
                    "ma_name": ma_name, "days_ago": days_ago,
                    "entry_price": trace["close_B"], "stop_loss": trace["low_A"],
                }
    return 0, [], {}

# ── Layer 5: Momentum (0-10) ────────────────────────────────────────
def score_momentum(rsi, rsi_avg, cfg):
    r, ra = rsi.iloc[-1], rsi_avg.iloc[-1]
    if np.isnan(r) or np.isnan(ra):
        return 0, [], {}
    r, ra = float(r), float(ra)
    pts = 0; reasons = []
    if r > cfg["rsi_min_level"]:
        pts += 5; reasons.append(f"RSI{r:.0f}")
    if r > ra:
        pts += 5; reasons.append("RSI>Avg")
    return min(10, pts), reasons, {"rsi": r, "rsi_avg": ra}

# ── Layer 6: Volume confirmation (0-10) ─────────────────────────────
def score_volume(df, cfg):
    vol = df["Volume"]
    cur_vol = float(vol.iloc[-1])
    avg_vol = float(vol.tail(cfg["volume_surge_lookback"]).mean())
    if avg_vol <= 0:
        return 0, [], {}
    ratio = cur_vol / avg_vol
    pts = 0; reasons = []
    if ratio >= 1.0:
        pts += min(10, (ratio - 1.0) * 10 + 3)
        reasons.append(f"Vol{ratio:.1f}xAvg")
    return round(min(10, pts)), reasons, {"vol_ratio": ratio}

# ── Master composite analysis ───────────────────────────────────────
def analyze_master(sym, df, spy_perf_pct):
    """
    Returns dict with composite tech_score (0-100) and full details,
    or None if basic liquidity filters fail.
    """
    n       = len(df)
    price   = float(df["Close"].iloc[-1])
    avg_vol = float(df["Volume"].tail(20).mean())

    if price   < CFG["min_price"]:      return None
    if avg_vol < CFG["min_avg_volume"]: return None
    if n < 210: return None   # need room for SMA200 + 52w metrics

    sma21  = df["Close"].rolling(CFG["sma21_period"]).mean()
    sma50  = df["Close"].rolling(CFG["sma50_period"]).mean()
    sma150 = df["Close"].rolling(CFG["sma150_period"]).mean()
    sma200 = df["Close"].rolling(CFG["sma200_period"]).mean()
    ema8   = df["Close"].ewm(span=CFG["ema8_period"], adjust=False).mean()
    jma    = calc_jma(df["Close"], CFG["jma_period"], CFG["jma_phase"])
    rsi    = calc_rsi(df["Close"], CFG["rsi_period"])
    rsi_avg = rsi.rolling(CFG["rsi_avg_period"]).mean()

    trend_pts, trend_r, trend_d       = score_trend_health(df, sma50, sma150, sma200, CFG)
    rs_pts, rs_r, rs_d                = score_relative_strength(df, spy_perf_pct, CFG)
    stack_pts, stack_r, stack_d       = score_ma_stack(df, jma, ema8, sma21, sma50, CFG)
    trigger_pts, trigger_r, trigger_d = score_entry_trigger(df, ema8, sma21, sma50, sma150, CFG)
    mom_pts, mom_r, mom_d             = score_momentum(rsi, rsi_avg, CFG)
    vol_pts, vol_r, vol_d             = score_volume(df, CFG)

    total = trend_pts + rs_pts + stack_pts + trigger_pts + mom_pts + vol_pts
    if total < CFG["min_total_score"]:
        return None

    # ── Entry / Stop: use the trigger's levels if one fired,
    #    otherwise fall back to today's close / recent swing low ────
    if trigger_d:
        entry_price = trigger_d["entry_price"]
        stop_loss   = trigger_d["stop_loss"]
    else:
        entry_price = price
        stop_loss   = float(df["Low"].tail(10).min())
    risk_pct = ((entry_price - stop_loss) / entry_price * 100
                if entry_price > 0 and entry_price > stop_loss else None)

    all_reasons = trend_r + rs_r + stack_r + trigger_r + mom_r + vol_r

    return {
        "tech_score"       : round(min(100, total)),
        "Trend_Score"      : trend_pts,
        "RS_Score"         : rs_pts,
        "Stack_Score"      : stack_pts,
        "Trigger_Score"    : trigger_pts,
        "Momentum_Score"   : mom_pts,
        "Volume_Score"     : vol_pts,
        "Price"            : round(price, 2),
        "Entry_Price"      : round(entry_price, 2),
        "Stop_Loss"        : round(stop_loss, 2),
        "Risk_%"           : round(risk_pct, 1) if risk_pct is not None else None,
        "Off_52wHigh_%"    : round(trend_d.get("off_52w_high_pct", 0), 1),
        "Above_52wLow_%"   : round(trend_d.get("above_52w_low_pct", 0), 1),
        "RS_vs_SPY_%"      : round(rs_d.get("rs_diff_pct", 0), 1),
        "Stack_Compression_%": round(stack_d.get("compression_pct", 0), 2),
        "Trigger_MA"       : trigger_d.get("ma_name", "—"),
        "Trigger_Days_Ago" : trigger_d.get("days_ago"),
        "RSI"              : round(mom_d.get("rsi", 0), 1),
        "Vol_Ratio"        : round(vol_d.get("vol_ratio", 0), 2),
        "SMA50"            : round(trend_d.get("sma50", 0), 2),
        "SMA150"           : round(trend_d.get("sma150", 0), 2),
        "SMA200"           : round(trend_d.get("sma200", 0), 2),
        "tech_reasons"     : " | ".join(all_reasons),
        "_df"    : df,
        "_sma21" : sma21, "_sma50": sma50, "_sma150": sma150, "_sma200": sma200,
        "_ema8"  : ema8, "_jma": jma,
    }

def get_spy_perf(rs_lookback_days, history_days):
    try:
        end   = datetime.today()
        start = end - timedelta(days=history_days)
        spy   = yf.Ticker("SPY").history(
            start=start.strftime("%Y-%m-%d"), end=end.strftime("%Y-%m-%d"),
            auto_adjust=True)
        spy.index = pd.to_datetime(spy.index)
        if hasattr(spy.index, "tz") and spy.index.tz:
            spy.index = spy.index.tz_localize(None)
        if len(spy) > rs_lookback_days:
            p_now  = float(spy["Close"].iloc[-1])
            p_then = float(spy["Close"].iloc[-rs_lookback_days])
            return (p_now - p_then) / p_then * 100
    except Exception: pass
    return 0.0

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
LIVE_COLS = ["Ticker","Price","Total","Tech","Fund_Bonus",
             "Trigger_MA","Risk_%","Sector"]
_CW = {"Ticker":8,"Price":10,"Total":7,"Tech":6,"Fund_Bonus":11,
       "Trigger_MA":12,"Risk_%":9,"Sector":20}
_CF = {"Price":"${:.2f}","Total":"{:.0f}","Tech":"{:.0f}",
       "Fund_Bonus":"{:.0f}","Risk_%":"{:+.1f}%"}
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

# ── SPY relative-strength baseline (Layer 2) ─────────────────────
print("━"*65)
print("  STEP 2b  SPY BASELINE (for RS vs SPY check)")
print("━"*65)
SPY_PERF = get_spy_perf(CFG["rs_lookback_days"], CFG["history_days"])
print(f"  SPY {CFG['rs_lookback_days']}-bar performance: {SPY_PERF:+.1f}%")
print()

# ── Main scan — 2-pass ───────────────────────────────────────
print("━"*65)
print(f"  STEP 3  SCANNING {len(TICKERS)} TICKERS")
print("━"*65)
print("  Pass 1: composite technical score across 6 signal families (fast)")
print("  Pass 2: fundamentals fetch for pass-1 survivors only (bonus points)\n")

_hdr_done   = False
results     = []
tech_passes = []
no_data     = 0

batches = [TICKERS[i:i+CFG["batch_size"]]
           for i in range(0, len(TICKERS), CFG["batch_size"])]

# ── PASS 1: Composite technical score (no .info calls) ────────────
with tqdm(total=len(TICKERS), desc="Pass 1 Tech", unit="stk",
          bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]") as pbar:
    for batch in batches:
        data_map = download(batch, CFG["history_days"])
        no_data += len(batch) - len(data_map)
        for sym in batch:
            pbar.update(1)
            if sym not in data_map: continue
            try:
                ts = analyze_master(sym, data_map[sym], SPY_PERF)
                if ts is None: continue
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

# ── PASS 2: Fundamentals for tech-pass stocks only (bonus points) ──
print("━"*65)
print(f"  PASS 2  FUNDAMENTAL CHECK ({len(tech_passes)} stocks)")
print("━"*65+"\n")

for item in tqdm(tech_passes, desc="Pass 2 Fund", unit="stk"):
    sym   = item["sym"]; price = item["price"]; ts = item["ts"]
    try:
        fund = get_fundamentals(sym)
        time.sleep(CFG["fund_sleep"])

        fund_bonus = round(min(10, fund["Fund_Score"] / 5))
        total = min(100, ts["tech_score"] + fund_bonus)

        result = {
            "Ticker"            : sym,
            "Price"             : ts["Price"],
            "Entry_Price"       : ts["Entry_Price"],
            "Stop_Loss"         : ts["Stop_Loss"],
            "Risk_%"            : ts["Risk_%"],
            "Total"             : total,
            "Tech"              : ts["tech_score"],
            "Fund_Bonus"        : fund_bonus,
            "Trend_Score"       : ts["Trend_Score"],
            "RS_Score"          : ts["RS_Score"],
            "Stack_Score"       : ts["Stack_Score"],
            "Trigger_Score"     : ts["Trigger_Score"],
            "Momentum_Score"    : ts["Momentum_Score"],
            "Volume_Score"      : ts["Volume_Score"],
            "Sector"            : fund["Sector"],
            "Company"           : fund["Company"],
            "Industry"          : fund["Industry"],
            "Off_52wHigh_%"     : ts["Off_52wHigh_%"],
            "Above_52wLow_%"    : ts["Above_52wLow_%"],
            "RS_vs_SPY_%"       : ts["RS_vs_SPY_%"],
            "Stack_Compression_%": ts["Stack_Compression_%"],
            "Trigger_MA"        : ts["Trigger_MA"],
            "Trigger_Days_Ago"  : ts["Trigger_Days_Ago"],
            "RSI"               : ts["RSI"],
            "Vol_Ratio"         : ts["Vol_Ratio"],
            "SMA50"             : ts["SMA50"],
            "SMA150"            : ts["SMA150"],
            "SMA200"            : ts["SMA200"],
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
            "_sma200": ts["_sma200"],
            "_ema8"  : ts["_ema8"],
            "_jma"   : ts["_jma"],
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

if not results:
    print("\n  No matches. Try relaxing:")
    print("   min_total_score                 45 → 30")
    print("   max_off_52w_high_pct             25 → 40")
    print("   min_above_52w_low_pct            20 → 10")
    print("   stack_compression_pct           5.0 → 8.0")
    print("   require_prior_uptrend         True → False")
    print("   require_sma50_rising          True → False")
    print("   require_volume_confirmation   True → False")
    print("   min_price                         2 → 1")
    print("   min_avg_volume                80000 → 50000")

# Sort by total score (always runs, even on empty list)
results.sort(key=lambda x: x["Total"], reverse=True)

# ── Always build df_out and save/email (even if 0 results) ────
ts      = datetime.today().strftime("%Y%m%d_%H%M")
out_dir = os.environ.get("GITHUB_WORKSPACE", os.getcwd())

COLS = [
    "Ticker","Company","Sector","Price","Entry_Price","Stop_Loss","Risk_%",
    "Total","Tech","Fund_Bonus",
    "Trend_Score","RS_Score","Stack_Score","Trigger_Score","Momentum_Score","Volume_Score",
    "Off_52wHigh_%","Above_52wLow_%","RS_vs_SPY_%","Stack_Compression_%",
    "Trigger_MA","Trigger_Days_Ago","RSI","Vol_Ratio",
    "SMA50","SMA150","SMA200",
    "Rev_Growth_%","Profit_Margin_%","ROE_%","PE_Ratio","EPS",
    "Tech_Flags","Fund_Flags",
]
df_out = pd.DataFrame([{k:v for k,v in r.items() if not k.startswith("_")}
                        for r in results]) if results else pd.DataFrame(columns=COLS)
if not df_out.empty:
    df_out = df_out[[c for c in COLS if c in df_out.columns]]
    df_out.reset_index(drop=True, inplace=True)

FMT = {
    "Price"          : lambda v: f"${v:.2f}",
    "Entry_Price"    : lambda v: f"${v:.2f}",
    "Stop_Loss"      : lambda v: f"${v:.2f}",
    "Risk_%"         : lambda v: f"{v:+.1f}%",
    "Total"          : lambda v: f"{v:.0f}",
    "Tech"           : lambda v: f"{v:.0f}",
    "Fund_Bonus"     : lambda v: f"{v:.0f}",
    "Trend_Score"    : lambda v: f"{v:.0f}",
    "RS_Score"       : lambda v: f"{v:.0f}",
    "Stack_Score"    : lambda v: f"{v:.0f}",
    "Trigger_Score"  : lambda v: f"{v:.0f}",
    "Momentum_Score" : lambda v: f"{v:.0f}",
    "Volume_Score"   : lambda v: f"{v:.0f}",
    "Off_52wHigh_%"  : lambda v: f"{v:.1f}%",
    "Above_52wLow_%" : lambda v: f"{v:+.1f}%",
    "RS_vs_SPY_%"    : lambda v: f"{v:+.1f}%",
    "Stack_Compression_%": lambda v: f"{v:.2f}%",
    "Trigger_Days_Ago": lambda v: f"{int(v)}d ago",
    "RSI"            : lambda v: f"{v:.1f}",
    "Vol_Ratio"      : lambda v: f"{v:.2f}x",
    "Rev_Growth_%"   : lambda v: f"{v:+.1f}%",
    "Profit_Margin_%": lambda v: f"{v:.1f}%",
    "ROE_%"          : lambda v: f"{v:.1f}%",
    "PE_Ratio"       : lambda v: f"{v:.1f}",
    "EPS"            : lambda v: f"${v:.2f}",
    "Market_Cap_B"   : lambda v: f"${v:.2f}B",
    "SMA50"          : lambda v: f"${v:.2f}",
    "SMA150"         : lambda v: f"${v:.2f}",
    "SMA200"         : lambda v: f"${v:.2f}",
}

def fmt_v(col, val):
    if val is None or (isinstance(val, float) and np.isnan(val)): return "—"
    try:
        if col in FMT: return FMT[col](val)
    except Exception: pass
    return str(val) if str(val) not in ("nan","None","") else "—"

if _IN_NOTEBOOK and results:
    DISP = ["Ticker","Company","Sector","Price","Entry_Price","Stop_Loss",
            "Total","Tech","Fund_Bonus","Risk_%"]
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
            elif col == "Fund_Bonus":
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
    CLI_COLS = ["Ticker","Price","Entry_Price","Stop_Loss","Total","Tech",
                "Fund_Bonus","Risk_%","Sector"]
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
    tit = f"  Master Technical Scanner   {datetime.today().strftime('%Y-%m-%d')}   {len(df_out)} matches"
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
  Total           Tech(0-90) + Fund_Bonus(0-10) = 0-100
  Tech            sum of the 6 technical layers below
  Fund_Bonus      bonus points from business fundamentals (0-10)
  Trend_Score     0-20: SMA50/150/200 stacked + rising, 52w position
  RS_Score        0-15: relative strength vs SPY, 6 months
  Stack_Score     0-15: JMA/EMA8/SMA21 bullish + compressed
  Trigger_Score   0-20: 2-candle retest+reclaim (any of EMA8/SMA21/SMA50/SMA150)
  Momentum_Score  0-10: RSI > 50 and rising vs its own average
  Volume_Score    0-10: today's volume vs its own 20-day average
  Entry_Price     the retest+reclaim entry if a trigger fired,
                  otherwise today's close
  Stop_Loss       the retest+reclaim stop if a trigger fired,
                  otherwise a 10-day swing-low proxy
  Risk_%          (Entry_Price - Stop_Loss) / Entry_Price
  ──────────────────────────────────────────────────────""")

# Save
fpath = os.path.join(out_dir, f"master_technical_scanner_{ts}.csv")
df_out.to_csv(fpath, index=False)
print(f"\n  💾 CSV → {fpath}")
tv = os.path.join(out_dir, f"tv_master_technical_scanner_{ts}.txt")
with open(tv,"w") as f:
    f.write(f"###Master Technical Scanner {datetime.today().strftime('%Y-%m-%d')}\n")
    for r in results: f.write(f"NASDAQ:{r['Ticker']}\n")
print(f"  📋 TradingView → {tv}")
if results:
    print(f"\n  📋 Tickers (comma-separated):")
    print(f"  {', '.join(r['Ticker'] for r in results)}")

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
            for c in ["Ticker","Price","Entry_Price","Stop_Loss","Total","Tech",
                      "Fund_Bonus","Risk_%"]
        )
        rows_e = ""
        for i, r in enumerate(rl[:50]):
            bg  = "#fff" if i % 2 == 0 else "#f0f9ff"
            ticker = r.get("Ticker","—")
            price  = r.get("Price",0) or 0
            entry  = r.get("Entry_Price",0) or 0
            sl     = r.get("Stop_Loss",0) or 0
            total  = r.get("Total",0) or 0
            tech   = r.get("Tech",0) or 0
            fund   = r.get("Fund_Bonus",0) or 0
            risk   = r.get("Risk_%",0) or 0
            rows_e += (
                f'<tr style="background:{bg}">'
                f'<td style="padding:6px 11px;font-size:12px;font-weight:700">{ticker}</td>'
                f'<td style="padding:6px 11px;font-size:12px">${float(price):.2f}</td>'
                f'<td style="padding:6px 11px;font-size:12px;color:#22c55e">${float(entry):.2f}</td>'
                f'<td style="padding:6px 11px;font-size:12px;color:#ef4444;font-weight:600">'
                f'${float(sl):.2f}</td>'
                f'<td style="padding:6px 11px;font-size:12px;font-weight:700;'
                f'background:#166534;color:#fff;text-align:center">{float(total):.0f}</td>'
                f'<td style="padding:6px 11px;font-size:12px">{float(tech):.0f}</td>'
                f'<td style="padding:6px 11px;font-size:12px">{float(fund):.0f}</td>'
                f'<td style="padding:6px 11px;font-size:12px;color:'
                f'{"#22c55e" if float(risk)>=0 else "#ef4444"}">{float(risk):+.1f}%</td>'
                f'</tr>'
            )
        no_results_msg = ('<tr><td colspan="8" style="padding:20px;text-align:center;'
                           'color:#94a3b8;font-size:13px">No matches today</td></tr>')

        html_e = f"""<!DOCTYPE html><html><body style="margin:0;padding:0;
background:#f1f5f9;font-family:'Segoe UI',Arial,sans-serif">
<table width="100%" cellpadding="0" cellspacing="0"><tr><td align="center" style="padding:20px 10px">
<table width="100%" style="max-width:800px;background:#fff;border-radius:12px;
       overflow:hidden;box-shadow:0 4px 16px rgba(0,0,0,0.08)">
  <tr><td style="background:linear-gradient(135deg,#0f172a,#1e3a5f);padding:22px 28px">
<h1 style="margin:0;color:#60a5fa;font-size:20px;font-weight:700">
  📊 Master Technical Scanner — Best Setups
</h1>
<p style="margin:6px 0 0;color:#94a3b8;font-size:12px">
  {datetime.today().strftime('%Y-%m-%d %H:%M UTC')} &nbsp;·&nbsp;
  {cnt} match{'es' if cnt!=1 else ''} found &nbsp;·&nbsp;
  composite of trend, relative strength, MA stack, entry trigger, momentum,
  volume, and fundamentals
</p>
  </td></tr>
  <tr><td style="padding:16px">
<div style="overflow-x:auto;border-radius:8px;border:1px solid #e2e8f0">
  <table style="border-collapse:collapse;width:100%;min-width:600px">
    <thead><tr>{th_e}</tr></thead>
    <tbody>{rows_e or no_results_msg}</tbody>
  </table>
</div>
<p style="font-size:11px;color:#64748b;margin:8px 0 0">
  📎 Full results (with per-layer score breakdown) attached as CSV
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
            f"Master Technical Scanner — {datetime.today().strftime('%Y-%m-%d')}",
            f"{cnt} matches",
            "="*60,
        ]
        if rl:
            for r in rl[:50]:
                ticker = r.get("Ticker","—")
                price  = r.get("Price",0) or 0
                entry  = r.get("Entry_Price",0) or 0
                sl     = r.get("Stop_Loss",0) or 0
                total  = r.get("Total",0) or 0
                tech   = r.get("Tech",0) or 0
                fund   = r.get("Fund_Bonus",0) or 0
                risk   = r.get("Risk_%",0) or 0
                plain_lines.append(
                    f"{ticker:<7} ${float(price):.2f}  Entry:${float(entry):.2f}  SL:${float(sl):.2f}  "
                    f"Total:{float(total):.0f}(Tech{float(tech):.0f}+Fund{float(fund):.0f})  "
                    f"Risk:{float(risk):+.1f}%"
                )
            plain_lines.append("")
            plain_lines.append("Tickers (comma-separated):")
            plain_lines.append(", ".join(r.get("Ticker","") for r in rl))
        else:
            plain_lines.append("No matches today")
        plain_lines.append("\nFull results (with per-layer score breakdown) in CSV attachment.")
        plain_e = "\n".join(plain_lines)

        subj = (f"📊 Master Technical Scanner — {cnt} best setup{'s' if cnt!=1 else ''}"
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
        print("         Generate one: myaccount.google.com/apppasswords")
        print("         → Google Account → Security → 2-Step Verification → App Passwords")
    except smtplib.SMTPException as e:
        print(f"[Email] ❌  SMTP error: {e}")
    except Exception as e:
        print(f"[Email] ❌  Unexpected error: {type(e).__name__}: {e}")

try:
    _send_email(results, fpath)
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
        df_p     = r["_df"].tail(180).copy()
        jma_p    = r["_jma"].reindex(df_p.index)
        ema8_p   = r["_ema8"].reindex(df_p.index)
        sma21_p  = r["_sma21"].reindex(df_p.index)
        sma50_p  = r["_sma50"].reindex(df_p.index)
        sma150_p = r["_sma150"].reindex(df_p.index)
        sma200_p = r["_sma200"].reindex(df_p.index)
        ax.set_facecolor("#0f172a")
        ax.plot(df_p.index, df_p["Close"], color="#60a5fa", lw=1.6, label="Price", zorder=5)
        ax.plot(df_p.index, jma_p,    color="#f472b6", lw=1.0, label="JMA",    zorder=4)
        ax.plot(df_p.index, ema8_p,   color="#38bdf8", lw=1.0, ls="--", label="EMA8",   zorder=3)
        ax.plot(df_p.index, sma21_p,  color="#34d399", lw=1.0, ls="--", label="SMA21",  zorder=3)
        ax.plot(df_p.index, sma50_p,  color="#fbbf24", lw=1.2, ls="-.", label="SMA50",  zorder=3)
        ax.plot(df_p.index, sma150_p, color="#f87171", lw=1.1, ls=":",  label="SMA150", zorder=3)
        ax.plot(df_p.index, sma200_p, color="#a78bfa", lw=1.1, ls=":",  label="SMA200", zorder=3)
        ax.scatter([df_p.index[-1]], [r["Entry_Price"]], color="#22c55e", s=60, zorder=6,
                   marker="^", label="Entry")
        ax.axhline(r["Stop_Loss"], color="#ef4444", lw=1.0, ls="--", alpha=0.8,
                  label=f"Stop ${r['Stop_Loss']:.2f}")
        ax.set_title(
            f"{r['Ticker']}  {r['Company']}  |  Entry ${r['Entry_Price']:.2f}  |  "
            f"SL ${r['Stop_Loss']:.2f}  |  "
            f"Score {r['Total']} (Tech{r['Tech']}+Fund{r['Fund_Bonus']})  |  "
            f"Trend{r['Trend_Score']} RS{r['RS_Score']} Stack{r['Stack_Score']} "
            f"Trig{r['Trigger_Score']} Mom{r['Momentum_Score']} Vol{r['Volume_Score']}",
            color="#e2e8f0", fontsize=8, fontweight="bold", pad=7)
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %y"))
        ax.tick_params(colors="#94a3b8", labelsize=9)
        for sp in ax.spines.values(): sp.set_edgecolor("#1e3a5f")
        ax.legend(loc="upper left", facecolor="#1e293b", labelcolor="#e2e8f0",
                  fontsize=6, framealpha=0.9, ncol=4)
        ax.grid(color="#1e3a5f", ls="--", lw=0.5, alpha=0.6)
    plt.suptitle(
        f"Master Technical Scanner — Best Setups  ·  "
        f"{datetime.today().strftime('%Y-%m-%d')}",
        color="#60a5fa", fontsize=12, fontweight="bold", y=1.001)
    plt.tight_layout()
    cp = os.path.join(out_dir, f"master_technical_scanner_chart_{ts}.png")
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
  📋 WHAT THIS SCANNER IS
  A synthesis of this repo's best-validated signal families into
  ONE ranked scan, so you don't need to run each specialist scanner
  separately. Instead of AND-gating every condition from every
  script (which would return ~0 matches), each stock earns partial
  credit across 6 independent technical layers, plus a fundamentals
  bonus — the highest TOTAL scores are the "best technical setups."

  📋 SCORE BREAKDOWN  (100 total)
  Trend_Score      0–20   SMA50>150>200 stacked+rising, 52w position
  RS_Score         0–15   relative strength vs SPY, 6 months
  Stack_Score      0–15   JMA/EMA8/SMA21 bullish + compressed
  Trigger_Score    0–20   2-candle retest+reclaim (EMA8/SMA21/SMA50/SMA150)
  Momentum_Score   0–10   RSI > 50 and rising vs its own average
  Volume_Score     0–10   today's volume vs its own 20-day average
  Tech             = sum of the 6 layers above (0-90)
  Fund_Bonus       0–10   business fundamentals (Pass 2 only, bonus
                          not a gate — this scanner is about
                          technical setup first)
  Total            = Tech + Fund_Bonus (0-100)

  📋 ENTRY TRIGGER DETAIL (reused verbatim from
  sma150_base_reclaim_v1.py's validated logic)
  Candle A (red, closed below the MA) → Candle B (green, closed
  above that MA AND EMA8, on higher volume) — scanned over the last
  entry_trigger_lookback_days trading days, not just today. A stock
  can score well here even without a fresh trigger — the other 5
  layers still contribute.

  📋 OUTPUT
  Entry_Price / Stop_Loss = the retest+reclaim trigger's levels if
  one fired recently, otherwise today's close / a 10-day swing-low
  proxy — a stock can rank highly on trend/RS/stack/momentum/volume
  alone even without an active entry trigger right now.

  💡 BEST SETUPS
  Total > 70                  elite across nearly every layer
  Trigger_Score > 0              an active, recent entry signal
  RS_Score > 10                     genuinely outperforming SPY
  Fund_Bonus > 7                       strong business quality too

  ⚙️  TUNE IF 0 RESULTS
  min_total_score              45 → 30
  max_off_52w_high_pct         25 → 40
  min_above_52w_low_pct        20 → 10
  stack_compression_pct       5.0 → 8.0
  require_prior_uptrend      True → False
  require_sma50_rising       True → False
  require_volume_confirmation True → False
  min_price                      2 → 1
  min_avg_volume             80000 → 50000
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
""")
