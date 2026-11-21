# ============================================================
# Scanner backtest engine (claude_run/research/backtest_engine.py)
# ============================================================
# Replays each scheduled scanner's OWN per-stock function (analyze_*(sym, df_daily))
# day by day over the last ~12 months on a fixed random sample of liquid US stocks,
# exactly as the live scan would have seen the data on each day, and simulates the trade:
#   entry  = next session's OPEN (the email arrives before the US open)
#   stop   = the scanner's Stop_Loss       (skip if the open is already below it)
#   target = the scanner's first target (Target_1 / Target / T1 ...), else 2R
#   exit   = stop or target, whichever is hit first (both in one bar -> stop),
#            gaps fill at the open, otherwise the close after 20 sessions.
# One open trade per scanner per stock; a setup re-reported on later days is ignored.
# Each scanner's file is loaded with its imports, settings and functions only - no
# downloads, emails or TradingView calls run. Output: bt_trades_<shard>.csv,
# bt_status_<shard>.csv. Shards: bt_shard_NN.py call main(shard, nshards).
# ============================================================
import ast, contextlib, io, os, random, re, sys, time, traceback, warnings
import subprocess
subprocess.call([sys.executable, "-m", "pip", "install", "-q", "yfinance", "pandas", "numpy", "requests", "tqdm", "matplotlib"],
                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
from datetime import datetime, timedelta
import numpy as np, pandas as pd

warnings.filterwarnings("ignore")
ROOT = os.getcwd()
SKIP = {"mtf_choch_fvg_v1.py": "needs 1H/5-min history (Yahoo keeps 60 days)",
        "india_weekly_reversal_2x_above_sma50_v1.py": "NSE universe",
        "claude_run/tools/scanner_returns.py": "not a scanner"}
BAD_CALLS = ("pip_install", "get_tickers", "_send_email", "send_email", "_tv_sync", "rank_best", "tqdm",
             "attach_to_email", "add_position_size")
BAD_NAMES = ("TICKERS", "daily_map", "results", "fpath", "df_out", "requests", "smtplib", "spy_map", "chk_d")
STOP_KEYS = ("Stop_Loss", "Stop", "SL", "Stop_Price")
TGT_KEYS = ("Target_1", "Target", "T1", "Target1", "Target_Price", "Target_Resistance", "Target_Prior_High",
            "Target_1_Swing_High", "Target_1_EMA8", "Target_1_SMA50", "Target_1_2R", "Take_Profit",
            "Target_Measured", "Target_2R", "T2", "Target_2")
HOLD = 20
REPLAY_DAYS = 250
N_TICKERS = int(os.environ.get("BT_TICKERS", 600))
SEED = 42
TIME_LIMIT = 130 * 60

def scheduled():
    out = []
    for line in open(os.path.join(ROOT, "claude_run", "schedule.txt")):
        s = line.strip()
        if s and not s.startswith("#") and s.endswith(".py"): out.append(s)
    return out

def load_scanner(path):
    """Exec only imports, defs and simple assignments; return (namespace, analyze_fn, errors)."""
    src = open(os.path.join(ROOT, path), encoding="utf-8").read()
    tree = ast.parse(src)
    ns = {"__name__": "bt_" + os.path.basename(path)[:-3], "__file__": os.path.join(ROOT, path)}
    errs = []
    for node in tree.body:
        if isinstance(node, (ast.Import, ast.ImportFrom, ast.FunctionDef, ast.ClassDef)):
            pass
        elif isinstance(node, (ast.Assign, ast.AnnAssign)):
            bad = False
            for sub in ast.walk(node):
                if isinstance(sub, ast.Call):
                    f = sub.func
                    fname = f.id if isinstance(f, ast.Name) else (f.attr if isinstance(f, ast.Attribute) else "")
                    if fname.startswith("download") or fname in BAD_CALLS: bad = True
                if isinstance(sub, ast.Name) and sub.id in BAD_NAMES: bad = True
            if bad: continue
        else:
            continue
        try:
            with contextlib.redirect_stdout(io.StringIO()):
                exec(compile(ast.Module(body=[node], type_ignores=[]), path, "exec"), ns)
        except Exception as e:
            errs.append(f"{type(e).__name__}: {e}"[:200])
    fn = None
    names = [n.name for n in tree.body if isinstance(n, ast.FunctionDef)]
    for cand in ("analyze_stock", "detect_pattern") + tuple(n for n in names if n.startswith("analyze") and n not in ("analyze_tf",)):
        if cand in ns and callable(ns[cand]): fn = ns[cand]; break
    if fn is None and "detect" in ns:
        d = ns["detect"]
        fn = (lambda sym, df, _d=d: _d(df, "Daily")) if "tf" in d.__code__.co_varnames[:d.__code__.co_argcount] \
             and d.__code__.co_argcount == 2 else (lambda sym, df, _d=d: _d(df))
    hist = 500
    cfg = ns.get("CFG") or {}
    for k in ("history_days", "lookback_days", "days", "HISTORY_DAYS"):
        if isinstance(cfg, dict) and isinstance(cfg.get(k), (int, float)): hist = int(cfg[k]); break
        if isinstance(ns.get(k), (int, float)): hist = int(ns[k]); break
    return ns, fn, errs, hist

def _first(d, keys):
    keys = tuple(keys) + tuple(k for k in d if str(k).startswith("Target_1") and k not in keys)
    for k in keys:
        v = d.get(k)
        try:
            v = float(v)
            if np.isfinite(v) and v > 0: return v, k
        except Exception: pass
    return None, None

def normalize(r):
    if r is None: return []
    if isinstance(r, dict): return [r]
    if isinstance(r, (list, tuple)): return [x for x in r if isinstance(x, dict)]
    return []

def simulate(df, pos, stop, target):
    """Enter at the open of bar pos+1. Returns dict or None."""
    o, h, l, c = (df[k].values for k in ("Open", "High", "Low", "Close"))
    e = pos + 1
    if e >= len(df): return None
    entry = float(o[e])
    if not (np.isfinite(entry) and entry > stop): return {"skip": "opened below stop"}
    risk = entry - stop
    tgt, tsrc = (target, "scanner") if target and target > entry else (entry + 2 * risk, "2R")
    last = min(len(df) - 1, e + HOLD - 1)
    for j in range(e, last + 1):
        if j > e and o[j] <= stop:   ex, why = float(o[j]), "stop (gap)"; break
        if j > e and o[j] >= tgt:    ex, why = float(o[j]), "target (gap)"; break
        if l[j] <= stop:             ex, why = stop, "stop"; break
        if h[j] >= tgt:              ex, why = tgt, "target"; break
    else:
        j = last; ex = float(c[j]); why = "time" if last == e + HOLD - 1 else "open"
    return {"entry_date": df.index[e].strftime("%Y-%m-%d"), "entry": round(entry, 4), "stop": round(stop, 4),
            "target": round(tgt, 4), "target_src": tsrc, "exit_date": df.index[j].strftime("%Y-%m-%d"),
            "exit": round(ex, 4), "exit_reason": why, "bars": int(j - e + 1),
            "R": round((ex - entry) / risk, 3), "ret_pct": round((ex / entry - 1) * 100, 3),
            "risk_pct": round(risk / entry * 100, 3)}

# measured ms per call (synthetic data) - used only to balance the shards
COST_MS = {
    "bull_pennant_breakout_v1.py": 0.4,
    "daily_3candle_higherlow_above_sma50_v1.py": 2.8,
    "daily_3candle_jma_reclaim_v1.py": 1.6,
    "daily_bear_stack_sma21_reclaim_3xvol_v1.py": 1.2,
    "daily_doji_hammer_jma_reclaim_v1.py": 2.2,
    "daily_jma_cross_ema8_v1.py": 0.9,
    "daily_jma_reclaim_rsi_v1.py": 2.4,
    "daily_liquidity_sweep_breakout_v1.py": 0.6,
    "daily_ma_support_ema21_sma50_v1.py": 0.6,
    "daily_ma_sweep_4x_v1.py": 0.7,
    "daily_major_support_retest_v1.py": 1.3,
    "daily_penny_jma_reclaim_rsi_v1.py": 0.2,
    "daily_penny_liquidity_sweep_breakout_v1.py": 0.2,
    "daily_penny_ma_support_ema21_sma50_v1.py": 0.2,
    "daily_price_jma_ema21_sma50_volume_v1.py": 6.3,
    "daily_pullback_jma_reclaim_v1.py": 2.5,
    "daily_rsi2_rs_1to1_v1.py": 0.3,
    "daily_rsi2_snapback_v1.py": 4.3,
    "daily_rsi_avg_bounce_v1.py": 1.5,
    "daily_rsi_bullish_divergence_v1.py": 1.9,
    "daily_rsi_high_breakout_v1.py": 1.8,
    "daily_sma50_150_squeeze_turn_v1.py": 1.6,
    "daily_sweep_mss_retest_v1.py": 4.1,
    "daily_sweep_sma50_strategy_v1.py": 0.5,
    "ema9_21_cross_retest_v1.py": 0.5,
    "inside_day_low_atr_v1.py": 0.6,
    "luxalgo_sr_v1.py": 5.0,
    "ma_stack_pullback_2green_v1.py": 9.8,
    "minervini.py": 1.3,
    "monthly_3candle_reclaim_jma_ema8_v1.py": 3.8,
    "monthly_reversal_2x_above_sma50_v1.py": 2.9,
    "monthly_rsi_cross_above_avg_v1.py": 3.8,
    "monthly_rsi_weekly_green_reversal_v1.py": 18.6,
    "rggrrg_pattern_v1.py": 0.6,
    "rsi30_recovery_sma50_v1.py": 2.2,
    "rsi_avg_turn_up_v1.py": 8.4,
    "sb_reclaim_v1.py": 12.8,
    "sma50_converge_rgg_jma_v1.py": 0.6,
    "smaretest.py": 2.4,
    "vwap20_touch_bounce_v1.py": 0.7,
    "weekly_doji_jma_reclaim_v1.py": 6.8,
    "weekly_hammer_green_doji_breakout_v1.py": 4.5,
    "weekly_reversal_2x_above_sma50_v1.py": 3.4,
    "weekly_rsi_bullish_divergence_v1.py": 4.8,
    "weekly_sma30_cross_vol21_v1.py": 5.3,
}

def shard_files(files, shard, nshards):
    """Longest-processing-time-first split of the scanners across the shards."""
    loads = [0.0] * nshards; assign = [[] for _ in range(nshards)]
    for f in sorted(files, key=lambda f: -COST_MS.get(f, 5.0)):
        k = loads.index(min(loads)); assign[k].append(f); loads[k] += COST_MS.get(f, 5.0) + 1.0
    return assign[shard]

# ---------------------------------------------------------------- data
def universe():
    import requests
    hdr = {"User-Agent": "Mozilla/5.0", "Referer": "https://www.nasdaq.com/"}
    rows = []
    for ex in ("nasdaq", "nyse", "amex"):
        try:
            r = requests.get(f"https://api.nasdaq.com/api/screener/stocks?tableonly=true&limit=10000&exchange={ex}&download=true",
                             headers=hdr, timeout=30)
            rows += r.json()["data"]["rows"]
        except Exception as e: print("universe", ex, e)
    pool = []
    for row in rows:
        s = str(row.get("symbol", "")).strip().upper()
        try:
            px = float(str(row.get("lastsale", "")).replace("$", "").replace(",", "")); vol = float(row.get("volume") or 0)
        except Exception: continue
        if s.isalpha() and 1 <= len(s) <= 5 and px >= 2 and vol >= 300_000: pool.append(s)
    pool = sorted(set(pool))
    print(f"universe pool {len(pool)}")
    rnd = random.Random(SEED)
    return sorted(rnd.sample(pool, min(N_TICKERS, len(pool))))

def download(symbols):
    import yfinance as yf
    out = {}
    for b in range(0, len(symbols), 100):
        batch = symbols[b:b + 100]
        for attempt in range(3):
            try:
                raw = yf.download(batch, period="7y", interval="1d", auto_adjust=True, actions=False,
                                  group_by="ticker", threads=True, progress=False)
                break
            except Exception as e:
                print("download retry", e); time.sleep(5); raw = None
        if raw is None: continue
        for s in batch:
            try:
                d = raw[s] if isinstance(raw.columns, pd.MultiIndex) else raw
                d = d[["Open", "High", "Low", "Close", "Volume"]].dropna(subset=["Close"])
                d.index = pd.to_datetime(d.index).tz_localize(None)
                if len(d) > 300: out[s] = d
            except Exception: pass
        time.sleep(1)
    return out

# ---------------------------------------------------------------- main
def main(shard=0, nshards=1, quick=False):
    t0 = time.time()
    files = [f for f in scheduled() if f not in SKIP and os.path.exists(os.path.join(ROOT, f))]
    mine = shard_files(files, shard, nshards)
    print(f"shard {shard}/{nshards}: {mine}")
    syms = universe()
    if quick: syms = syms[:40]
    data = download(syms + ["SPY"])
    spy = data.pop("SPY", None)
    print(f"data for {len(data)} stocks, SPY {None if spy is None else len(spy)} bars")
    cal = spy.index if spy is not None else max(data.values(), key=len).index
    days = REPLAY_DAYS if not quick else 15
    replay = list(cal[-days - 1:-1])                          # each needs a next bar for the entry
    spy_c = spy["Close"] if spy is not None else None
    regime = (spy_c > spy_c.rolling(50).mean()) if spy_c is not None else None
    trades, status = [], []
    for f in mine:
        ns, fn, errs, hist = load_scanner(f)
        if spy is not None:                                    # globals some scanners fill from a SPY download
            ns.setdefault("SPY_CLOSE", spy_c); ns.setdefault("_spy", spy)
        name = os.path.basename(f)[:-3]
        st = {"scanner": name, "load_errors": " | ".join(errs[:3]), "calls": 0, "call_errors": 0,
              "first_error": "", "signals": 0, "no_stop": 0, "trades": 0, "seconds": 0.0, "keys": ""}
        if fn is None:
            st["first_error"] = "no analyze function"; status.append(st); print(name, "SKIP no fn"); continue
        argn = fn.__code__.co_varnames[:fn.__code__.co_argcount] if hasattr(fn, "__code__") else ()
        wants_spy = "spy" in argn
        ts = time.time()
        for s, df in data.items():
            if time.time() - t0 > TIME_LIMIT: break
            busy_until = None; seen = {}
            idx = df.index
            for d in replay:
                if d not in idx: continue
                pos = idx.get_loc(d)
                if busy_until is not None and pos <= busy_until: continue
                sub = df.loc[d - timedelta(days=hist + 5):d]
                st["calls"] += 1
                try:
                    with contextlib.redirect_stdout(io.StringIO()):
                        r = fn(s, sub, spy_c.loc[:d]) if wants_spy else fn(s, sub)
                except Exception as e:
                    st["call_errors"] += 1
                    if not st["first_error"]: st["first_error"] = f"{type(e).__name__}: {e}"[:200]
                    continue
                for sig in normalize(r):
                    if not st["keys"]: st["keys"] = ",".join(map(str, sig.keys()))[:300]
                    stop, _ = _first(sig, STOP_KEYS)
                    if stop is None: st["no_stop"] += 1; continue
                    key = round(stop, 2)
                    if key in seen and pos - seen[key] <= 15: continue
                    seen[key] = pos
                    st["signals"] += 1
                    tgt, tk = _first(sig, TGT_KEYS)
                    res = simulate(df, pos, stop, tgt)
                    if not res or "skip" in res: continue
                    res.update({"scanner": name, "ticker": s, "signal_date": d.strftime("%Y-%m-%d"),
                                "target_key": tk or "", "tf": sig.get("TF", sig.get("Timeframe", "")),
                                "spy_above_50": bool(regime.loc[:d].iloc[-1]) if regime is not None else None})
                    trades.append(res); st["trades"] += 1
                    busy_until = idx.get_loc(pd.Timestamp(res["exit_date"]))
                    break
        st["seconds"] = round(time.time() - ts, 1)
        status.append(st)
        print(f"{name}: calls {st['calls']} errors {st['call_errors']} signals {st['signals']} trades {st['trades']} "
              f"{st['seconds']}s {st['first_error'][:120]}")
        pd.DataFrame(trades).to_csv(f"bt_trades_{shard:02d}.csv", index=False)
        pd.DataFrame(status).to_csv(f"bt_status_{shard:02d}.csv", index=False)
    pd.DataFrame(trades).to_csv(f"bt_trades_{shard:02d}.csv", index=False)
    pd.DataFrame(status).to_csv(f"bt_status_{shard:02d}.csv", index=False)
    print(f"done in {time.time() - t0:.0f}s: {len(trades)} trades")

if __name__ == "__main__":
    main(int(os.environ.get("BT_SHARD", 0)), int(os.environ.get("BT_NSHARDS", 1)), bool(os.environ.get("BT_QUICK")))
