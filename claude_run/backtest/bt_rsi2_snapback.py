# Independent replay of daily_rsi2_snapback_v1.py (run by ClaudeRunner).
# The scanner's own analyze function is called point-in-time on every
# candidate day (df cut at that day), and the email's order plan is then
# traded with code written separately from the research harness:
#   buy  : limit Entry_Price on the next session only (fill = min(open, limit))
#   exit : target fill + Target_Offset, stop fill - Stop_Offset, else sell at
#          the open of the first session on/after Exit_By.
#   Conservative: gap through a level fills at the open; a day touching both
#   counts as the stop; on the fill day the target counts only if the close
#   is at/above it. Costs 0.10% round trip. One position per ticker at a time.
# Different random ticker sample (seed 99) from the research run (seed 7).
import os, sys, time, random, runpy, subprocess, warnings
import numpy as np, pandas as pd

warnings.filterwarnings("ignore")
SAMPLE, YEARS, COST = int(os.environ.get("BT_SAMPLE", "1500")), 3.6, 0.10
t0 = time.time()
subprocess.check_call([sys.executable, "claude_run/tools/gen_rsi2_snapback.py", "_bt_snap.py"])
base = runpy.run_path("claude_run/tools/backtest_winrate.py", run_name="lib")
ns = base["load_scanner"]("_bt_snap.py")
analyze, CFG = ns["analyze_rsi2_snapback"], ns["CFG"]
tickers = ns["get_tickers"]()
random.Random(99).shuffle(tickers)
tickers = sorted(tickers[:SAMPLE])
data = base["download"](tickers, int(YEARS * 365))
print(f"tickers {len(data)} | download {(time.time()-t0)/60:.1f} min", flush=True)

trades, signals, unfilled, date_mismatch = [], 0, 0, 0
for t, df in data.items():
    o, h, l, c = (df[x].values for x in ("Open", "High", "Low", "Close"))
    rsi2 = ns["wilder_rsi"](c.astype(float), CFG["rsi_period"])
    n, busy = len(df), -1
    for k in np.where(rsi2 < CFG["rsi_max"] + 0.5)[0]:
        if k < 250 or k >= n - 2 or k <= busy:
            continue
        r = analyze(t, df.iloc[:k + 1])
        if not r:
            continue
        signals += 1
        f = k + 1
        if str(df.index[f].date()) != r["Buy_Date"]:
            date_mismatch += 1                      # data gap or holiday-calendar difference
        if l[f] > r["Entry_Price"]:
            unfilled += 1
            continue
        fill = min(o[f], r["Entry_Price"])
        tgt, stp = fill + r["Target_Offset"], fill - r["Stop_Offset"]
        exit_by = pd.Timestamp(r["Exit_By"])
        px = xb = None
        for j in range(f, n):
            if df.index[j] >= exit_by and j > f:
                px, xb, why = o[j], j, "time"; break
            if j == f:
                if l[j] <= stp: px, xb, why = stp, j, "stop"; break
                if c[j] >= tgt: px, xb, why = tgt, j, "target"; break
            else:
                if o[j] <= stp: px, xb, why = o[j], j, "stop"; break
                if o[j] >= tgt: px, xb, why = o[j], j, "target"; break
                if l[j] <= stp: px, xb, why = stp, j, "stop"; break
                if h[j] >= tgt: px, xb, why = tgt, j, "target"; break
        if px is None:
            busy = n; continue                      # still open at the end of the data
        trades.append({"Ticker": t, "Signal": df.index[k], "Fill_Date": df.index[f], "Exit": df.index[xb],
                       "Why": why, "Fill": round(float(fill), 4), "Exit_Px": round(float(px), 4),
                       "Stop_Offset": r["Stop_Offset"], "Score": r["Score"], "RSI2": r["RSI2"],
                       "Ret_%": (px / fill - 1) * 100 - COST, "Risk_%": r["Risk_%"]})
        busy = xb

T = pd.DataFrame(trades)
T.to_csv("bt_rsi2_snapback_trades.csv", index=False)
def line(d, lab):
    w = d["Ret_%"][d["Ret_%"] > 0].sum(); ls = -d["Ret_%"][d["Ret_%"] <= 0].sum()
    print(f"  {lab:12s} n={len(d):5d}  win={(d['Ret_%']>0).mean()*100:5.1f}%  avg={d['Ret_%'].mean():+.2f}%  "
          f"PF={w/ls if ls else float('inf'):.2f}  avgWin={d['Ret_%'][d['Ret_%']>0].mean():+.2f}%  "
          f"avgLoss={d['Ret_%'][d['Ret_%']<=0].mean():+.2f}%  worst={d['Ret_%'].min():+.1f}%")
print(f"\nsignals {signals} | unfilled {unfilled} ({unfilled/max(1,signals)*100:.0f}%) | trades {len(T)} | "
      f"buy-date mismatches {date_mismatch}")
line(T, "ALL")
for y, d in T.groupby(T.Signal.dt.year):
    line(d, str(y))
cut = T.Signal.min() + (T.Signal.max() - T.Signal.min()) / 2
line(T[T.Signal < cut], "1st half"); line(T[T.Signal >= cut], "2nd half")
print("  exits:", T.Why.value_counts().to_dict())
mo = T.Signal.dt.to_period("M")
m = T.groupby(mo)["Ret_%"].agg(lambda s: (s > 0).mean() * 100)
print(f"  months {len(m)} | months with win rate < 70%: {(m < 70).sum()} | worst month {m.min():.0f}%")
print(f"  trades per month in this {len(data)}-ticker sample: {len(T)/len(m):.1f}")
# equity at $40 risk per trade (risk measured from fill to stop = Risk_%)
pnl = 40 * T["Ret_%"] / T["Risk_%"]
print(f"  P&L at $40 risk per trade: total ${pnl.sum():,.0f} over {len(T)} trades, avg ${pnl.mean():+.2f}, worst ${pnl.min():,.0f}")
print(f"done in {(time.time()-t0)/60:.1f} min")
