# Research backtest for one scanner (run by ClaudeRunner via a wrapper in
# claude_run/research/ that sets SCANNER). Reuses the point-in-time replay
# from backtest_winrate.py, but for every signal it records the market /
# stock context at the signal bar and the outcome at several R targets, so
# filters and targets can be searched afterwards (and validated out-of-sample).
#
# Output: rs_trades_<scanner>.csv — one row per signal with features and,
# for each R in R_LIST, status_<R> (win / loss / timeout / open) and R_<R>.
import os, sys, time, math, random, warnings, runpy
import numpy as np, pandas as pd

warnings.filterwarnings("ignore")
SCANNER  = globals().get("SCANNER") or (sys.argv[1] if len(sys.argv) > 1 else None)
BT_DAYS  = int(os.environ.get("BT_DAYS", "500"))
HOLD     = int(os.environ.get("HOLD_DAYS", "20"))
SAMPLE   = int(os.environ.get("BT_SAMPLE", "600"))
R_LIST   = [0.5, 0.75, 1.0, 1.5, 2.0]

base = runpy.run_path("claude_run/tools/backtest_winrate.py", run_name="lib")
load_scanner, find_analyze, download = base["load_scanner"], base["find_analyze"], base["download"]


def outcome(o, h, l, c, k, entry, stop, rmult, hold):
    """Same rules as backtest_winrate.simulate (entry = signal close)."""
    n = len(c); risk = entry - stop; target = entry + rmult * risk
    for j in range(k + 1, min(k + 1 + hold, n)):
        if o[j] <= stop:   return "loss", (o[j] - entry) / risk, j - k
        if o[j] >= target: return "win", (o[j] - entry) / risk, j - k
        if l[j] <= stop:   return "loss", -1.0, j - k
        if h[j] >= target: return "win", rmult, j - k
    if k + hold >= n:
        return "open", np.nan, np.nan
    j = k + hold
    return "timeout", (c[j] - entry) / risk, j - k


def features(df, k, spy):
    c, h, l, v = (df[x].values for x in ("Close", "High", "Low", "Volume"))
    cs = pd.Series(c[:k+1])
    def sma(n): return cs.tail(n).mean() if k + 1 >= n else np.nan
    sma20, sma50, sma200 = sma(20), sma(50), sma(200)
    sma50_prev = cs.iloc[-70:-20].mean() if k + 1 >= 70 else np.nan
    d = cs.diff(); g = d.clip(lower=0).ewm(alpha=1/14, adjust=False).mean(); ls = (-d.clip(upper=0)).ewm(alpha=1/14, adjust=False).mean()
    rsi = float(100 - 100 / (1 + g.iloc[-1] / ls.iloc[-1])) if ls.iloc[-1] > 0 else 100.0
    tr = np.maximum(h[1:k+1] - l[1:k+1], np.maximum(abs(h[1:k+1] - c[:k]), abs(l[1:k+1] - c[:k])))
    atr = tr[-14:].mean() if len(tr) >= 14 else np.nan
    hi252 = h[max(0, k-251):k+1].max()
    vol20 = v[max(0, k-20):k].mean()
    f = {
        "above_sma20": c[k] > sma20, "above_sma50": c[k] > sma50, "above_sma200": c[k] > sma200,
        "sma50_rising": sma50 > sma50_prev if not np.isnan(sma50_prev) else np.nan,
        "sma50_above_sma200": sma50 > sma200 if not np.isnan(sma200) else np.nan,
        "rsi14": round(rsi, 1), "atr_pct": round(atr / c[k] * 100, 2) if atr == atr else np.nan,
        "dist_52w_high_pct": round((c[k] / hi252 - 1) * 100, 1),
        "vol_ratio": round(v[k] / vol20, 2) if vol20 > 0 else np.nan,
        "ret_5d": round((c[k] / c[k-5] - 1) * 100, 1) if k >= 5 else np.nan,
        "ret_20d": round((c[k] / c[k-20] - 1) * 100, 1) if k >= 20 else np.nan,
        "ret_60d": round((c[k] / c[k-60] - 1) * 100, 1) if k >= 60 else np.nan,
        "price": round(c[k], 2),
    }
    d0 = df.index[k]
    if spy is not None and d0 in spy.index:
        f.update({kk: bool(vv) for kk, vv in spy.loc[d0].items()})
    return f


def main():
    name = os.path.basename(SCANNER)[:-3]
    t0 = time.time()
    ns = load_scanner(SCANNER)
    cfg = ns["CFG"]
    for key in list(cfg):
        if "lookback" in key and key.startswith(("signal_", "trigger_")):
            cfg[key] = 1
    aname, analyze = find_analyze(ns)
    tickers = ns["get_tickers"]() if "get_tickers" in ns else []
    random.Random(42).shuffle(tickers)
    tickers = sorted(tickers[:SAMPLE])
    hist = int(cfg.get("history_days", 550)) + int(BT_DAYS * 1.5) + 10
    data = download(tickers, hist)
    bench = "^NSEI" if any(t.endswith(".NS") for t in tickers) else "SPY"
    spyd = download([bench], hist + 400).get(bench)
    spy = None
    if spyd is not None:
        sc = spyd["Close"]
        spy = pd.DataFrame({"mkt_above_sma50": sc > sc.rolling(50).mean(),
                            "mkt_above_sma200": sc > sc.rolling(200).mean(),
                            "mkt_sma50_rising": sc.rolling(50).mean() > sc.rolling(50).mean().shift(20)})
    print(f"{name}: {aname}() | tickers {len(data)} | benchmark {bench} | history {hist}d | bt days {BT_DAYS}")

    rows = []
    for t, df in data.items():
        n = len(df)
        o, h, l, c = (df[x].values for x in ("Open", "High", "Low", "Close"))
        busy, seen = -1, set()
        for k in range(max(1, n - BT_DAYS), n):
            if k <= busy:
                continue
            try:
                r = analyze(t, df.iloc[:k+1])
            except Exception:
                r = None
            if isinstance(r, tuple):
                kind, r = r
                if kind != "match":
                    continue
            if not r:
                continue
            sig = r.get("Signal_Date") or str(df.index[k].date())
            if sig in seen:
                continue
            seen.add(sig)
            try:
                entry, stop = float(r.get("Entry_Price")), float(r.get("Stop_Loss"))
            except (TypeError, ValueError):
                continue
            if not (math.isfinite(entry) and math.isfinite(stop)) or stop >= entry or entry > c[k] * 1.001:
                continue                                  # (all scheduled scanners enter at the close)
            row = {"Scanner": name, "Ticker": t, "Date": str(df.index[k].date()),
                   "Entry": entry, "Stop": stop, "risk_pct": round((entry - stop) / entry * 100, 2),
                   "Score": r.get("Score")}
            row.update(features(df, k, spy))
            longest = 0
            for rm in R_LIST:
                st, rv, days = outcome(o, h, l, c, k, entry, stop, rm, HOLD)
                row[f"status_{rm}"], row[f"R_{rm}"], row[f"days_{rm}"] = st, rv, days
                if st != "open":
                    longest = max(longest, days)
            rows.append(row)
            busy = k + int(longest) if longest else n           # no overlapping trades per ticker
    out = pd.DataFrame(rows)
    out.to_csv(f"rs_trades_{name}.csv", index=False)
    print(f"{len(out)} signals in {(time.time()-t0)/60:.1f} min")
    for rm in R_LIST:
        d = out[out[f"status_{rm}"] != "open"]
        if len(d):
            print(f"  R {rm}: n={len(d)} win%={(d[f'status_{rm}']=='win').mean()*100:.1f} avgR={d[f'R_{rm}'].mean():+.3f}")


if __name__ == "__main__" or SCANNER:
    main()
