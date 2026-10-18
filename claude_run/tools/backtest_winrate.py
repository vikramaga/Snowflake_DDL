# Win-rate backtest for one scanner, run by ClaudeRunner through a small
# wrapper in claude_run/backtest/ that sets SCANNER (the scanner's file name).
#
# Method (point-in-time replay of the scanner's OWN code):
#   • The scanner's functions and CFG are loaded with ast — its full scan is
#     NOT run. Every CFG "*lookback*" signal window is set to 1 so a signal
#     is only counted on the day it first appears.
#   • For each of the last BT_DAYS trading days, each sampled ticker's daily
#     history is cut at that day and the scanner's analyze_*() is called.
#   • Trade: enter at Entry_Price (the signal close; if Entry_Price is above
#     the close it is treated as a buy-stop that must trigger within 5 days),
#     stop = Stop_Loss, target = entry + R_MULT x risk. Walk forward up to
#     HOLD_DAYS: gap through stop/target fills at the open; a day that hits
#     both counts as a loss; no hit = exit at the last close (timeout).
#   • WIN = target reached before the stop. One open trade per ticker at a time.
#   • Universe: a fixed random sample (seed 42) of the scanner's own ticker
#     list — today's listings, so delisted names are missing (survivorship).
import ast, os, sys, time, json, math, random, warnings
import numpy as np, pandas as pd

warnings.filterwarnings("ignore")
SCANNER   = globals().get("SCANNER") or (sys.argv[1] if len(sys.argv) > 1 else None)
BT_DAYS   = int(os.environ.get("BT_DAYS", "260"))
HOLD_DAYS = int(os.environ.get("HOLD_DAYS", "20"))
R_MULT    = float(os.environ.get("R_MULT", "2"))
SAMPLE    = int(os.environ.get("BT_SAMPLE", "600"))
TRIGGER_DAYS = 5


def load_scanner(path):
    """Exec the scanner's imports, functions, classes and literal constants."""
    tree = ast.parse(open(path).read())
    keep = []
    for n in tree.body:
        if isinstance(n, (ast.Import, ast.ImportFrom, ast.FunctionDef, ast.ClassDef)):
            keep.append(n)
        elif isinstance(n, ast.Assign) and all(isinstance(t, ast.Name) for t in n.targets):
            if isinstance(n.value, (ast.Dict, ast.List, ast.Tuple, ast.Set, ast.Constant)):
                keep.append(n)
    ns = {"__name__": "scanner_bt", "_IN_NOTEBOOK": False}
    exec(compile(ast.Module(body=keep, type_ignores=[]), path, "exec"), ns)
    return ns


def find_analyze(ns):
    names = [k for k, v in ns.items() if k.startswith("analyze_") and callable(v)]
    if len(names) != 1:
        raise RuntimeError(f"expected one analyze_* function, found {names}")
    return names[0], ns[names[0]]


def download(tickers, days):
    import yfinance as yf
    from datetime import datetime, timedelta
    start = (datetime.today() - timedelta(days=days)).strftime("%Y-%m-%d")
    out = {}
    for i in range(0, len(tickers), 100):
        batch = tickers[i:i+100]
        try:
            raw = yf.download(batch, start=start, group_by="ticker", auto_adjust=True,
                              actions=False, threads=True, progress=False)
        except Exception as e:
            print("download error", e); continue
        for t in batch:
            try:
                df = raw[t] if isinstance(raw.columns, pd.MultiIndex) else raw
                df = df[["Open", "High", "Low", "Close", "Volume"]].dropna(subset=["Close"])
                df.index = pd.to_datetime(df.index).tz_localize(None)
                if len(df) > 260:
                    out[t] = df
            except Exception:
                pass
        time.sleep(1.0)
    return out


def simulate(df, k, entry, stop):
    """Return dict with outcome for a signal on bar k (or None if not filled)."""
    n = len(df)
    o, h, l, c = (df[x].values for x in ("Open", "High", "Low", "Close"))
    start = k + 1
    if entry > c[k] * 1.001:                      # buy-stop above the signal close
        fill = None
        for j in range(k + 1, min(k + 1 + TRIGGER_DAYS, n)):
            if h[j] >= entry:
                fill = j; entry = max(entry, o[j]); break
        if fill is None:
            return None if k + TRIGGER_DAYS < n else {"status": "open"}
        start = fill                               # the fill day itself counts
    risk = entry - stop
    if risk <= 0:
        return None
    target = entry + R_MULT * risk
    last = min(start + HOLD_DAYS, n) if start == k + 1 else min(start + HOLD_DAYS, n)
    for j in range(start, last):
        if o[j] <= stop:  return {"status": "loss", "R": (o[j] - entry) / risk, "days": j - k}
        if o[j] >= target: return {"status": "win", "R": (o[j] - entry) / risk, "days": j - k}
        hit_s, hit_t = l[j] <= stop, h[j] >= target
        if hit_s:  return {"status": "loss", "R": -1.0, "days": j - k}
        if hit_t:  return {"status": "win", "R": R_MULT, "days": j - k}
    if start + HOLD_DAYS > n:
        return {"status": "open"}
    j = start + HOLD_DAYS - 1
    return {"status": "timeout", "R": (c[j] - entry) / risk, "days": j - k}


def main():
    path = SCANNER
    name = os.path.basename(path)[:-3]
    t0 = time.time()
    ns = load_scanner(path)
    cfg = ns["CFG"]
    for key in list(cfg):
        if "lookback" in key and key.startswith(("signal_", "trigger_")):
            cfg[key] = 1
    aname, analyze = find_analyze(ns)
    print(f"{name}: {aname}()  lookbacks -> 1: {[k for k in cfg if 'lookback' in k and k.startswith(('signal_','trigger_'))]}")

    tickers = ns["get_tickers"]() if "get_tickers" in ns else []
    random.Random(42).shuffle(tickers)
    tickers = sorted(tickers[:SAMPLE])
    hist = int(cfg.get("history_days", 550)) + int(BT_DAYS * 1.5) + 10
    data = download(tickers, hist)
    print(f"tickers sampled {len(tickers)}, with data {len(data)}, history {hist} days")

    trades = []
    for t, df in data.items():
        n = len(df)
        first = max(1, n - BT_DAYS)
        busy_until = -1
        seen = set()
        for k in range(first, n):
            if k <= busy_until:
                continue
            try:
                r = analyze(t, df.iloc[:k+1])
            except Exception:
                r = None
            if isinstance(r, tuple):                 # ("match"|"armed", row) style
                kind, r = r
                if kind != "match":
                    continue
            if not r:
                continue
            sig = r.get("Signal_Date") or str(df.index[k].date())
            if sig in seen:
                continue
            seen.add(sig)
            entry, stop = r.get("Entry_Price"), r.get("Stop_Loss")
            try:
                entry, stop = float(entry), float(stop)
            except (TypeError, ValueError):
                continue
            if not (math.isfinite(entry) and math.isfinite(stop)) or stop >= entry:
                continue
            res = simulate(df, k, entry, stop)
            if res is None:
                continue
            row = {"Ticker": t, "Date": str(df.index[k].date()), "Signal_Date": sig,
                   "Entry": round(entry, 4), "Stop": round(stop, 4),
                   "Risk_%": round((entry - stop) / entry * 100, 2), "Score": r.get("Score")}
            row.update(res)
            trades.append(row)
            if res["status"] != "open":
                busy_until = k + res["days"]
            else:
                busy_until = n
    tr = pd.DataFrame(trades)
    tr.to_csv(f"bt_trades_{name}.csv", index=False)
    done = tr[tr.status != "open"] if len(tr) else tr
    nT = len(done)
    w = int((done.status == "win").sum()) if nT else 0
    l = int((done.status == "loss").sum()) if nT else 0
    to = int((done.status == "timeout").sum()) if nT else 0
    summ = {
        "Scanner": name, "Trades": nT, "Wins": w, "Losses": l, "Timeouts": to,
        "Win_Rate_%": round(w / nT * 100, 1) if nT else None,
        "Win_Rate_excl_Timeouts_%": round(w / (w + l) * 100, 1) if (w + l) else None,
        "Avg_R": round(done.R.mean(), 3) if nT else None,
        "Total_R": round(done.R.sum(), 1) if nT else None,
        "Avg_Days": round(done.days.mean(), 1) if nT else None,
        "Median_Risk_%": round(done["Risk_%"].median(), 2) if nT else None,
        "Open_Trades": int((tr.status == "open").sum()) if len(tr) else 0,
        "Tickers": len(data), "BT_Days": BT_DAYS, "Hold_Days": HOLD_DAYS, "R_Mult": R_MULT,
        "Period_Start": str(min(df.index[max(1, len(df)-BT_DAYS)] for df in data.values()).date()) if data else "",
        "Period_End": str(max(df.index[-1] for df in data.values()).date()) if data else "",
        "Runtime_min": round((time.time() - t0) / 60, 1),
    }
    pd.DataFrame([summ]).to_csv(f"bt_summary_{name}.csv", index=False)
    print(json.dumps(summ, indent=1))


if __name__ == "__main__" or SCANNER:
    main()
