# Chart review of the latest Best 10 watchlist (all scanners' Best 10 from the newest scheduled run).
# For every stock: trend, pullback/reclaim trigger, extension, relative strength vs SPY, volume,
# room to resistance, earnings date. Each stock gets BUY TODAY / WATCH (with the trigger price) / AVOID
# and the reasons. Stops follow the Best 10 stop rules (at least 2 ATR below the close).
# Outputs: chart_review_YYYYMMDD.csv, chart_review_buys_YYYYMMDD.png (charts of the buy list)
import glob, os, re, subprocess, sys, time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
subprocess.call([sys.executable, "-m", "pip", "install", "-q", "yfinance", "matplotlib", "lxml"],
                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
import numpy as np, pandas as pd, yfinance as yf
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt

ROOT = "claude_run/results"
STOP_ATR = 2.0


def latest_best10():
    runs = {}
    for f in glob.glob(f"{ROOT}/[0-9]*/*/best10_*.csv"):
        runs.setdefault(f.split("/")[2], set()).add(f.split("/")[3])
    run = max((r for r, s in runs.items() if len(s) >= 10), key=int)
    fs = glob.glob(f"{ROOT}/{run}/*/best10_*.csv")
    a = pd.concat([pd.read_csv(f).assign(Scanner=f.split("/")[3]) for f in fs], ignore_index=True)
    a["Ticker"] = a.Ticker.astype(str).str.upper()
    g = a.groupby("Ticker").agg(Scanners=("Scanner", lambda x: ", ".join(sorted(set(x)))), N_Scanners=("Scanner", "nunique"),
                                Best_Score=("Best_Score", "max"), Scanner_Stop=("Stop_Loss", "min"),
                                Analyst_Upside=("Analyst_Upside_%", "max"), Headline=("Catalyst_Headline", "first")).reset_index()
    return run, g


def calc_jma(c, period=13, phase=40, power=2):
    pr = phase / 100 + 1.5; beta = 0.45 * (period - 1) / (0.45 * (period - 1) + 2); alpha = beta ** power
    out = np.full(len(c), np.nan); e0 = e1 = e2 = j = 0.0
    for i, v in enumerate(c):
        if np.isnan(v): continue
        e0 = (1 - alpha) * v + alpha * e0; e1 = (v - e0) * (1 - beta) + beta * e1
        e2 = (e0 + pr * e1 - j) * (1 - alpha) ** 2 + alpha ** 2 * e2; j = e2 + j; out[i] = j
    return out


def download(tickers):
    out = {}
    for i in range(0, len(tickers), 60):
        b = tickers[i:i + 60]
        try:
            raw = yf.download(b, period="2y", group_by="ticker", auto_adjust=True, actions=False, threads=True, progress=False)
        except Exception:
            continue
        for t in b:
            try:
                d = raw[t] if isinstance(raw.columns, pd.MultiIndex) else raw
                d = d.dropna(subset=["Close"]); d.index = pd.to_datetime(d.index).tz_localize(None)
                if len(d) > 220: out[t] = d[["Open", "High", "Low", "Close", "Volume"]].astype(float)
            except Exception:
                pass
    return out


def next_earn(t):
    try:
        cal = yf.Ticker(t).calendar
        ds = cal.get("Earnings Date") if isinstance(cal, dict) else None
        if ds:
            ds = [pd.Timestamp(x) for x in ds]
            ds = [x for x in ds if x >= pd.Timestamp.today().normalize() - pd.Timedelta(days=1)]
            if ds: return t, min(ds)
    except Exception:
        pass
    return t, None


def ind(d):
    o, h, l, c, v = (d[k].values for k in ("Open", "High", "Low", "Close", "Volume"))
    s = pd.Series(c)
    pc = np.r_[np.nan, c[:-1]]
    atr = pd.Series(np.nanmax(np.vstack([h - l, np.abs(h - pc), np.abs(l - pc)]), axis=0)).rolling(14).mean().values
    dd = s.diff(); up = dd.clip(lower=0).ewm(alpha=1 / 14, adjust=False).mean(); dn = (-dd.clip(upper=0)).ewm(alpha=1 / 14, adjust=False).mean()
    rsi = (100 - 100 / (1 + up / dn.replace(0, np.nan))).fillna(100).values
    macd = s.ewm(span=12, adjust=False).mean() - s.ewm(span=26, adjust=False).mean(); sig = macd.ewm(span=9, adjust=False).mean()
    return dict(o=o, h=h, l=l, c=c, v=v, atr=atr, rsi=rsi, macd=macd.values, sig=sig.values, jma=calc_jma(c),
                e8=s.ewm(span=8, adjust=False).mean().values, e21=s.ewm(span=21, adjust=False).mean().values,
                s50=s.rolling(50).mean().values, s200=s.rolling(200).mean().values, vavg=pd.Series(v).rolling(20).mean().shift(1).values)


def review(t, d, spy_ret20, row, earn):
    x = ind(d); i = len(d) - 1
    c, h, l, o = x["c"], x["h"], x["l"], x["o"]
    a = x["atr"][i]; close = c[i]
    if not np.isfinite(a) or a <= 0: return None
    reasons_bad, reasons_good = [], []
    stage2 = close > x["s200"][i] and x["s50"][i] > x["s200"][i]
    s200_up = x["s200"][i] > x["s200"][i - 20]
    s50_up = x["s50"][i] > x["s50"][i - 10]
    ret20 = (close / c[i - 20] - 1) * 100; rs20 = ret20 - spy_ret20
    ext = (close - x["e21"][i]) / a
    green = close > o[i]; clv = (close - l[i]) / (h[i] - l[i]) if h[i] > l[i] else 0.5
    above_short = close > x["jma"][i] and close > x["e8"][i]
    was_below = c[i - 1] <= x["jma"][i - 1] or c[i - 1] <= x["e8"][i - 1] or c[i - 2] <= x["e8"][i - 2]
    touched = min(l[i - 4:i + 1]) <= max(x["e21"][i], x["s50"][i]) * 1.01
    jma_up = x["jma"][i] > x["jma"][i - 1]
    volx = x["v"][i] / x["vavg"][i] if x["vavg"][i] > 0 else np.nan
    dollar_vol = x["vavg"][i] * close / 1e6
    hi60 = h[i - 60:i].max()
    # stop: the Best 10 rule (at least 2 ATR below), but no deeper than 3 ATR (an old weekly/monthly stop can be far away)
    sc = row.Scanner_Stop if np.isfinite(row.Scanner_Stop) and row.Scanner_Stop < close else np.inf
    stop = max(min(sc, close - STOP_ATR * a), close - 3 * a)
    risk = close - stop
    target = hi60 if hi60 > close + 0.5 * a else close + 2 * risk
    rr = (target - close) / risk
    bd = np.busday_count(pd.Timestamp.today().date(), earn.date()) if earn is not None else None
    trig = max(x["jma"][i], x["e8"][i])

    # avoid
    if bd is not None and 0 <= bd <= 5: reasons_bad.append(f"earnings {earn:%d %b}")
    if close < 3 or dollar_vol < 5: reasons_bad.append(f"thin/penny (${dollar_vol:.1f}M/day)")
    if close < x["s200"][i] and not s200_up and x["s50"][i] < x["s200"][i]: reasons_bad.append("downtrend (below falling SMA200, SMA50 < SMA200)")
    if ext > 3 or x["rsi"][i] > 75: reasons_bad.append(f"extended ({ext:.1f} ATR above EMA21, RSI {x['rsi'][i]:.0f})")
    if close < x["s50"][i] and not s50_up and close < x["e21"][i]: reasons_bad.append("below falling SMA50 and EMA21")
    # buy-today checklist
    checks = {
        "uptrend": stage2 or (close > x["s200"][i] and s200_up),
        "trigger": green and above_short and (was_below or touched) and clv >= 0.5,
        "not_extended": ext <= 2 and 45 <= x["rsi"][i] <= 70,
        "rel_strength": rs20 >= 0,
        "room": rr >= 1.5,
        "risk_ok": risk / close <= 0.10,
    }
    if checks["uptrend"]: reasons_good.append("Stage 2 uptrend (above SMA200, SMA50 > SMA200)" if stage2 else "above rising SMA200")
    if checks["trigger"]: reasons_good.append("green close back above JMA & EMA8 after a pullback" + (" to EMA21/SMA50" if touched else "")
                                              + f", closed in top {100 - clv * 100:.0f}% of range")
    if checks["rel_strength"]: reasons_good.append(f"held up vs SPY (20d {ret20:+.1f}% vs SPY {spy_ret20:+.1f}%)")
    if volx >= 1.2 and green: reasons_good.append(f"volume {volx:.1f}x average on the up day")
    if x["macd"][i] > x["sig"][i] and x["macd"][i - 1] <= x["sig"][i - 1]: reasons_good.append("MACD just crossed up")
    if reasons_bad: verdict = "AVOID"
    elif all(checks.values()): verdict = "BUY TODAY"
    elif checks["uptrend"] and checks["not_extended"] and not checks["trigger"]: verdict = "WATCH"
    else: verdict = "PASS"
    missing = [k for k, ok in checks.items() if not ok]
    score = (25 * checks["uptrend"] + 25 * checks["trigger"] + 15 * checks["not_extended"] + min(15, max(0, rs20)) +
             min(10, rr * 3) + (5 if volx >= 1.2 and green else 0) + 3 * min(row.N_Scanners - 1, 3) + np.nan_to_num(row.Best_Score) / 20)
    return {"Ticker": t, "Verdict": verdict, "Score": round(score, 1), "Close": round(close, 2), "Stop": round(stop, 2),
            "Risk_%": round(risk / close * 100, 1), "Target": round(target, 2), "Reward_R": round(rr, 2),
            "Buy_Trigger": round(trig, 2) if verdict == "WATCH" else np.nan,
            "Why_Buy": "; ".join(reasons_good), "Why_Not": "; ".join(reasons_bad) or ("missing: " + ", ".join(missing) if missing else ""),
            "RSI": round(x["rsi"][i], 1), "Ext_ATR": round(ext, 2), "RS20_vs_SPY": round(rs20, 1), "Vol_x": round(volx, 2),
            "ATR_%": round(a / close * 100, 2), "Next_Earnings": earn.strftime("%Y-%m-%d") if earn is not None else "",
            "N_Scanners": row.N_Scanners, "Scanners": row.Scanners, "Best_Score": row.Best_Score,
            "Analyst_Upside": row.Analyst_Upside, "Headline": row.Headline, "Date": d.index[i].strftime("%Y-%m-%d")}


def charts(buys, data, path):
    n = len(buys)
    if not n: return
    cols = 3; rows = int(np.ceil(n / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(6 * cols, 3.6 * rows), squeeze=False)
    for ax, (_, r) in zip(axes.flat, buys.iterrows()):
        d = data[r.Ticker].iloc[-130:]; x = ind(data[r.Ticker]); k = len(d)
        idx = np.arange(k)
        for j in range(k):
            col = "#16a34a" if d.Close.iloc[j] >= d.Open.iloc[j] else "#dc2626"
            ax.vlines(j, d.Low.iloc[j], d.High.iloc[j], color=col, lw=0.7)
            ax.vlines(j, min(d.Open.iloc[j], d.Close.iloc[j]), max(d.Open.iloc[j], d.Close.iloc[j]), color=col, lw=2.6)
        for key, col, lab in (("jma", "#22c55e", "JMA"), ("e8", "#f97316", "EMA8"), ("e21", "#facc15", "EMA21"),
                              ("s50", "#2563eb", "SMA50"), ("s200", "#dc2626", "SMA200")):
            ax.plot(idx, x[key][-k:], color=col, lw=1, label=lab)
        ax.axhline(r.Stop, color="#991b1b", ls="--", lw=0.9); ax.axhline(r.Target, color="#15803d", ls="--", lw=0.9)
        ax.set_title(f"{r.Ticker}  {r.Close}  stop {r.Stop}  target {r.Target} ({r.Reward_R}R)", fontsize=9)
        ax.tick_params(labelsize=7); ax.set_xticks([])
    for ax in axes.flat[n:]: ax.axis("off")
    axes.flat[0].legend(fontsize=6, loc="upper left")
    fig.tight_layout(); fig.savefig(path, dpi=90); plt.close(fig)


def main():
    run, g = latest_best10()
    print(f"Best 10 run {run}: {len(g)} stocks")
    data = download(sorted(set(g.Ticker) | {"SPY"}))
    spy = data["SPY"].Close; spy_ret20 = (spy.iloc[-1] / spy.iloc[-21] - 1) * 100
    print(f"SPY {spy.iloc[-1]:.2f} ({spy.index[-1]:%Y-%m-%d}): 20d {spy_ret20:+.1f}%, vs SMA50 {(spy.iloc[-1] / spy.rolling(50).mean().iloc[-1] - 1) * 100:+.1f}%, "
          f"vs SMA200 {(spy.iloc[-1] / spy.rolling(200).mean().iloc[-1] - 1) * 100:+.1f}%")
    with ThreadPoolExecutor(8) as ex:
        earn = dict(ex.map(next_earn, [t for t in g.Ticker if t in data]))
    out = []
    for _, row in g.iterrows():
        if row.Ticker not in data: continue
        try:
            r = review(row.Ticker, data[row.Ticker], spy_ret20, row, earn.get(row.Ticker))
            if r: out.append(r)
        except Exception as e:
            print(row.Ticker, e)
    o = pd.DataFrame(out)
    o["rank"] = o.Verdict.map({"BUY TODAY": 0, "WATCH": 1, "PASS": 2, "AVOID": 3})
    o = o.sort_values(["rank", "Score"], ascending=[True, False]).drop(columns="rank")
    ds = datetime.today().strftime("%Y%m%d")
    o.to_csv(f"chart_review_{ds}.csv", index=False)
    print(o.Verdict.value_counts().to_string())
    pd.set_option("display.width", 250); pd.set_option("display.max_colwidth", 120)
    print(o[o.Verdict == "BUY TODAY"][["Ticker", "Score", "Close", "Stop", "Risk_%", "Target", "Reward_R", "RSI", "RS20_vs_SPY", "Vol_x", "Next_Earnings", "N_Scanners"]].to_string(index=False))
    for _, r in o[o.Verdict == "BUY TODAY"].iterrows():
        print(f"\n{r.Ticker}: {r.Why_Buy} | scanners: {r.Scanners} | {r.Headline}")
    print("\nWATCH (top 15):")
    print(o[o.Verdict == "WATCH"].head(15)[["Ticker", "Score", "Close", "Buy_Trigger", "Stop", "RSI", "RS20_vs_SPY", "Why_Not"]].to_string(index=False))
    charts(o[o.Verdict == "BUY TODAY"].head(12), data, f"chart_review_buys_{ds}.png")
    charts(o[o.Verdict == "WATCH"].head(9), data, f"chart_review_watch_{ds}.png")


if __name__ == "__main__":
    main()
