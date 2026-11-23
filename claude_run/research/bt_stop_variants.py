# Stop-loss review, step 2: would a wider stop have done better than the scanner's own stop?
# Replays every trade of the 1-year backtest (same entries, same engine: next-day open, stop or
# target first, 20-session limit) with the stop moved down to at least k x ATR(14) below the
# entry when the scanner's stop was closer than that. A scanner price target is kept; a 2R target
# is recomputed from the new risk. R is measured on the new risk, as with a fixed $ max loss.
# Output: sr_stop_variants.csv (one row per trade, R for each variant)
import glob, os, sys, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, pandas as pd
import backtest_engine as be

ROOT = "claude_run/results"
BT_RUNS = ["44937025710", "45026911281", "45048442468"]
FLOORS = [0.0, 1.0, 1.5, 2.0, 2.5]

bt = pd.concat([pd.read_csv(f) for r in BT_RUNS for f in glob.glob(f"{ROOT}/{r}/*/bt_trades_*.csv")
                if os.path.getsize(f) > 200], ignore_index=True)
bt = bt[bt.exit_reason != "open"]
print(f"trades {len(bt)}")
data = be.download(sorted(bt.ticker.astype(str).unique()))
print(f"data {len(data)} tickers")
out = []
for t, g in bt.groupby("ticker"):
    d = data.get(t)
    if d is None: continue
    h, l, c = d.High.values, d.Low.values, d.Close.values
    pc = np.r_[np.nan, c[:-1]]
    atr = pd.Series(np.nanmax(np.vstack([h - l, np.abs(h - pc), np.abs(l - pc)]), axis=0)).rolling(14).mean().values
    idx = d.index
    for _, x in g.iterrows():
        k = idx.searchsorted(pd.Timestamp(x.signal_date))
        if k >= len(idx) - 1 or idx[k] != pd.Timestamp(x.signal_date) or not np.isfinite(atr[k]): continue
        entry = float(d.Open.values[k + 1]); a = float(atr[k])
        tgt = float(x.target) if x.target_src == "scanner" else None
        r = {"scanner": x.scanner, "ticker": t, "signal_date": x.signal_date, "risk_atr0": (entry - x.stop) / a,
             "risk_pct0": (entry - x.stop) / entry * 100}
        for f in FLOORS:
            stop = min(float(x.stop), entry - f * a) if f else float(x.stop)
            s = be.simulate(d, k, stop, tgt)
            if not s or "R" not in s: r[f"R_{f}"] = np.nan; continue
            r[f"R_{f}"] = s["R"]; r[f"stop_{f}"] = s["exit_reason"].startswith("stop"); r[f"riskpct_{f}"] = round((entry - stop) / entry * 100, 2)
        out.append(r)
o = pd.DataFrame(out)
o.round(3).to_csv("sr_stop_variants.csv", index=False)
print(f"sr_stop_variants.csv: {len(o)} trades")
for f in FLOORS:
    R = o[f"R_{f}"].clip(-10, 10)
    print(f"floor {f} ATR: avg R {R.mean():+.3f}  win {(R > 0).mean():.0%}  stopped {o[f'stop_{f}'].mean():.0%}  avg risk {o[f'riskpct_{f}'].mean():.1f}%")
