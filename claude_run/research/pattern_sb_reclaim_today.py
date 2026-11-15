# Research (2026-11-15): which stocks from today's scanner results show the same pattern
# as Vik's SB chart? Collects every ticker from today's result CSVs, downloads daily bars,
# and keeps only the exact pattern:
#   1. Trend stack: SMA21 > SMA50 > SMA150, SMA50 and SMA150 rising (10 sessions)
#   2. Run to a fresh high 3-20 sessions ago (highest high of 60 sessions), with
#      JMA and EMA8 above SMA21 at that high
#   3. Pullback: JMA turned down after the high; the pullback held above SMA50
#   4. Shakeout: a red candle in the last 3 sessions before today CLOSED below SMA21
#   5. Reclaim today: green candle closing above JMA, EMA8 and SMA21 (yesterday's
#      close was below EMA8), close in the upper half of the range
import glob, os, sys, time, subprocess
subprocess.call([sys.executable, "-m", "pip", "install", "-q", "yfinance", "matplotlib"])
import numpy as np, pandas as pd, yfinance as yf
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt

RUNS = ["43665094951", "43668286377", "43676212528"]   # today's (2026-11-15 IST) scanner runs
tick = {}
for run in RUNS:
    for f in glob.glob(f"claude_run/results/{run}/*/*.csv"):
        if os.path.basename(f).startswith("best10"): continue
        try: d = pd.read_csv(f)
        except Exception: continue
        col = next((c for c in d.columns if c.lower() in ("ticker", "symbol")), None)
        if not col: continue
        sc = f.split("/")[3]
        for t in d[col].dropna().astype(str):
            t = t.strip().upper()
            if t.isalpha() and len(t) <= 5: tick.setdefault(t, set()).add(sc)
print(f"{len(tick)} tickers from {len(RUNS)} runs")

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
        if h[k] >= h[k-59:k+1].max() and h[k] >= h[k:i+1].max() and min(jma[k], e8[k]) > s21[k]:
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

syms = sorted(tick)
data = {}
for b in range(0, len(syms), 100):
    batch = syms[b:b+100]
    try:
        raw = yf.download(batch, period="400d", group_by="ticker", auto_adjust=True, progress=False, threads=True)
        for s in batch:
            try:
                df = raw[s].dropna(subset=["Close"]) if len(batch) > 1 else raw.dropna(subset=["Close"])
                if len(df) >= 200: data[s] = df
            except Exception: pass
    except Exception as e: print("batch fail", e)
    time.sleep(1)
print(f"data for {len(data)} tickers, last bar {max(d.index[-1] for d in data.values()).date()}")

rows, near, funnel = [], [], [0]*5
for s, df in data.items():
    try: st, info = check(df)
    except Exception: continue
    vals = list(st.values())
    cum = True
    for k, v in enumerate(vals):
        cum = cum and v; funnel[k] += cum
    i = len(df) - 1
    base = {"Ticker": s, "Close": round(float(df.Close.iloc[-1]), 2),
            "Stop_shakeout_low": round(float(df.Low.iloc[info["shake"]]), 2) if info["shake"] is not None else None,
            "High": round(float(df.High.iloc[info["hb"]]), 2) if info["hb"] is not None else None,
            "SMA21": round(float(info["s21"][i]), 2), "SMA50": round(float(info["s50"][i]), 2),
            "Scanners": ", ".join(sorted(tick[s]))}
    if all(vals): rows.append((base, df, info))
    elif sum(vals) == 4: near.append({**base, "Missing": [k for k, v in st.items() if not v][0]})
print("FUNNEL:", dict(zip(["stack", "fresh_high", "pullback", "shakeout", "reclaim"], funnel)))
print(f"\n=== EXACT MATCHES: {len(rows)} ===")
out = []
for b, df, info in rows:
    r = b.copy()
    if r["Stop_shakeout_low"]:
        risk = r["Close"] - r["Stop_shakeout_low"]
        r["Risk_%"] = round(risk / r["Close"] * 100, 1)
        r["R_to_high"] = round((r["High"] - r["Close"]) / risk, 1) if risk > 0 else None
    out.append(r); print(r)
pd.DataFrame(out).to_csv("pattern_sb_matches.csv", index=False)
print(f"\n=== NEAR MISSES (4 of 5): {len(near)} ===")
for r in near[:40]: print(r)
pd.DataFrame(near).to_csv("pattern_sb_near.csv", index=False)

if rows:
    n = min(12, len(rows))
    fig, axes = plt.subplots(n, 1, figsize=(12, 3.6*n), facecolor="#0f172a")
    axes = np.atleast_1d(axes)
    for ax, (b, df, info) in zip(axes, rows[:n]):
        d = df.iloc[-70:]; off = len(df) - 70; x = np.arange(len(d))
        ax.set_facecolor("#0f172a")
        cols = ["#22c55e" if cc > oo else "#ef4444" for oo, cc in zip(d.Open, d.Close)]
        ax.vlines(x, d.Low, d.High, color=cols, lw=0.8)
        ax.bar(x, (d.Close - d.Open).abs().clip(lower=1e-9), bottom=np.minimum(d.Open, d.Close), color=cols, width=0.6)
        for k, cl, lab in (("jma", "#e879f9", "JMA"), ("e8", "#f97316", "EMA8"), ("s21", "#16a34a", "SMA21"),
                           ("s50", "#3b82f6", "SMA50"), ("s150", "#ef4444", "SMA150")):
            ax.plot(x, info[k][off:], color=cl, lw=1.1, label=lab)
        ax.legend(loc="upper left", fontsize=6, facecolor="#0f172a", labelcolor="#cbd5e1")
        ax.set_title(f"{b['Ticker']}  {b['Close']}  | {b['Scanners'][:90]}", color="#e2e8f0", fontsize=9)
        ax.tick_params(colors="#94a3b8", labelsize=6)
    plt.tight_layout(); plt.savefig("pattern_sb_matches.png", dpi=110, facecolor="#0f172a")
