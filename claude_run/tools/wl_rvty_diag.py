# Diagnostic: why the RVTY example does / does not match wl_pullback_reclaim_check.
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, pandas as pd
import wl_pullback_reclaim_check as w
d = w.indicators(w.download(["RVTY"])["RVTY"])
o, h, l, c = (d[k].values for k in ("Open", "High", "Low", "Close"))
j, e21, s50, s150, s200, atr = (d[k].values for k in ("JMA", "EMA21", "SMA50", "SMA150", "SMA200", "ATR14"))
n = len(d)
print("last close", c[-1], d.index[-1].date())
for t in range(n - 200, n):
    body = (c[t] - o[t]) / atr[t]
    if c[t] > o[t] and body >= 0.5:
        why = []
        if not (s50[t] > s150[t] > s200[t]): why.append("MA order")
        if not s50[t] > s50[t-20]: why.append("SMA50 not rising")
        if not s200[t] > s200[t-20]: why.append("SMA200 not rising")
        if not c[t] > j[t]: why.append("close<=JMA")
        if not c[t-1] <= j[t-1]: why.append("prev close above JMA")
        if not c[t] > e21[t]: why.append("close<=EMA21")
        hi = max(range(t-15, t-2), key=lambda i: h[i]); pb = list(range(hi+1, t))
        falls = sum(1 for i in pb if j[i] < j[i-1])
        lo = min(pb, key=lambda i: l[i]) if pb else t-1
        print(f"{d.index[t].date()} C {c[t]:.2f} body {body:.2f}ATR | JMA {j[t]:.2f} prevC {c[t-1]:.2f} prevJMA {j[t-1]:.2f} "
              f"EMA21 {e21[t]:.2f} SMA50 {s50[t]:.2f} SMA150 {s150[t]:.2f} SMA200 {s200[t]:.2f} | pb {len(pb)}d falls {falls} "
              f"lo {l[lo]:.2f} (EMA21 {e21[lo]:.2f} SMA50 {s50[lo]:.2f}) depth {(h[hi]-l[lo])/atr[t]:.1f}ATR "
              f"jfall_prev {j[t-1] <= j[t-2]} | {'MATCH' if w.check_at(d, t) else ', '.join(why) or 'pullback rules'}")
