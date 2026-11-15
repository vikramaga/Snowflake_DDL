# Research (2026-11-15): which stocks from today's scanner results show the same pattern
# as Vik's SB chart (WEEKLY bars, weeks end Friday)? Collects every ticker from today's result CSVs, downloads daily bars,
# and keeps only the exact pattern:
#   1. Trend stack: SMA21 > SMA50 > SMA150, SMA50 and SMA150 rising (10 weeks)
#   2. Run to a fresh high 3-20 weeks ago (highest high of 52 weeks), with
#      JMA and EMA8 above SMA21 at that high
#   3. Pullback: JMA turned down after the high; the pullback held above SMA50
#   4. Shakeout: a red candle in the 3 weeks before this week CLOSED below SMA21
#   5. Reclaim this week: green candle closing above JMA, EMA8 and SMA21 (last week's
#      close was below EMA8), close in the upper half of the range
import glob, os, sys, time, subprocess
subprocess.call([sys.executable, "-m", "pip", "install", "-q", "yfinance", "matplotlib"])
import numpy as np, pandas as pd, yfinance as yf
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt

def calc_jma(series, period=13, phase=40, power=2):
    """
    Jurik Moving Average — same formula as TradingView's "Jurik Moving
    Average" indicator, so values match the chart:
        beta       = 0.45*(L-1) / (0.45*(L-1) + 2)
        alpha      = beta ** power
        phaseRatio = phase/100 + 1.5   (0.5 if phase < -100, 2.5 if > 100)
        e0  = (1-alpha)*src + alpha*e0[1]
        e1  = (src - e0)*(1-beta) + beta*e1[1]
        e2  = (e0 + phaseRatio*e1 - jma[1])*(1-alpha)^2 + alpha^2*e2[1]
        jma = e2 + jma[1]
    period = Length, phase = Phase, power = Power (TradingView inputs).
    """
    phase_ratio = 0.5 if phase < -100 else (2.5 if phase > 100 else phase / 100.0 + 1.5)
    beta  = 0.45 * (period - 1) / (0.45 * (period - 1) + 2)
    alpha = beta ** power
    vals  = np.asarray(series.values, dtype=float)
    out   = np.full(len(vals), np.nan)
    e0 = e1 = e2 = jma = 0.0
    for i, v in enumerate(vals):
        if np.isnan(v):
            continue
        e0  = (1 - alpha) * v + alpha * e0
        e1  = (v - e0) * (1 - beta) + beta * e1
        e2  = (e0 + phase_ratio * e1 - jma) * (1 - alpha) ** 2 + alpha ** 2 * e2
        jma = e2 + jma
        out[i] = jma
    return pd.Series(out, index=series.index)


def check(df):
    o, h, l, c = (df[k].values.astype(float) for k in ("Open", "High", "Low", "Close"))
    cs = df["Close"].astype(float)
    jma = calc_jma(cs, 13, 40, 2).values
    e8 = cs.ewm(span=8, adjust=False).mean().values
    s21, s50, s150 = (cs.rolling(n).mean().values for n in (21, 50, 150))
    i = len(df) - 1
    st = {}
    st["1_stack"] = bool(s21[i] > s50[i] > s150[i] and s50[i] > s50[i-10] and s150[i] > s150[i-10])
    hb = None
    for k in range(i - 20, i - 2):
        if h[k] >= h[k-51:k+1].max() and h[k] >= h[k:i+1].max() and min(jma[k], e8[k]) > s21[k]:
            hb = k; break
    st["2_fresh_high"] = hb is not None
    if hb is not None:
        st["3_pullback"] = bool(any(jma[k] < jma[k-1] for k in range(hb+1, i)) and l[hb+1:i+1].min() > s50[i])
        sk = [k for k in range(i-3, i) if c[k] < o[k] and c[k] < s21[k] and k > hb]
        st["4_shakeout_close_below_sma21"] = bool(sk)
    else:
        st["3_pullback"] = st["4_shakeout_close_below_sma21"] = False
        sk = []
    st["5_reclaim_today"] = bool(c[i] > o[i] and c[i] > max(jma[i], e8[i], s21[i]) and c[i-1] < e8[i-1]
                                and c[i] >= (h[i] + l[i]) / 2)
    info = {"hb": hb, "shake": sk[-1] if sk else None, "jma": jma, "e8": e8, "s21": s21, "s50": s50, "s150": s150}
    return st, info


raw = yf.download("NDSN", period="7y", auto_adjust=True, progress=False)
if isinstance(raw.columns, pd.MultiIndex): raw.columns = raw.columns.get_level_values(0)
w = raw.resample("W-FRI").agg({"Open": "first", "High": "max", "Low": "min", "Close": "last",
                               "Volume": "sum"}).dropna(subset=["Close"])
print("weekly bars:", len(w), w.index[0].date(), "->", w.index[-1].date())
for end in pd.date_range("2026-06-05", "2026-09-04", freq="W-FRI"):
    sub = w.loc[:end]
    st, info = check(sub)
    i = len(sub) - 1
    print(end.date(), "PASS" if all(st.values()) else "    ", {k.split("_",1)[1][:12]: int(v) for k, v in st.items()},
          "high", sub.index[info["hb"]].date() if info["hb"] is not None else None,
          "shake", sub.index[info["shake"]].date() if info["shake"] is not None else None)
end = pd.Timestamp("2026-07-24")
sub = w.loc[:end]; st, info = check(sub)
t = sub[["Open","High","Low","Close"]].copy()
for k in ("jma","e8","s21","s50","s150"): t[k] = info[k]
print("\nNDSN weekly bars to 2026-07-24:")
print(t.tail(22).round(2).to_string())
print("\nAfter 2026-07-24:")
print(w.loc["2026-07-25":, ["Open","High","Low","Close"]].round(2).to_string())
