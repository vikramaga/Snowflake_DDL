# Stop-loss review, data step (runs in ClaudeRunner; reads the saved results in the repo).
#
# 1. Every Best 10 pick (= the daily "Best 10 - YYYY-MM-DD" TradingView list) from the scheduled
#    runs in claude_run/results/: what happened after the signal day - stop touched or not, on
#    which session, gap through the stop, best move before that (MFE in R), latest result in R.
# 2. The same signal-day features for every trade of the 1-year backtest (run 44937025710 plus
#    the later one-scanner backtests), so a rule found on two weeks of live picks can be checked
#    on a full year before it is used.
# Features: risk vs ATR, ATR %, RSI(14)/RSI(2), distance from EMA21/SMA20/50/200, 5/20-day return,
# signal-day gap / range / close location / upper wick / volume, 52-week-high distance, SPY trend,
# days to the next earnings report.
# Outputs: sr_best10_outcomes.csv, sr_backtest_features.csv
import glob, os, re, sys, time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta
import subprocess
subprocess.call([sys.executable, "-m", "pip", "install", "-q", "lxml"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
import numpy as np, pandas as pd, yfinance as yf

ROOT = "claude_run/results"
BT_RUNS = ["44937025710", "45026911281", "45048442468"]
TS = re.compile(r"_(20\d{6})_(\d{4})\.csv$")
T0 = time.time()


def best10_picks():
    rows = []
    for rd in sorted(glob.glob(f"{ROOT}/[0-9]*/")):
        subs = [s for s in glob.glob(rd + "*/") if glob.glob(s + "best10_*.csv")]
        if len(subs) < 10:
            continue
        run = os.path.basename(rd.rstrip("/"))
        for sd in subs:
            scn = os.path.basename(sd.rstrip("/"))
            if scn in ("build_best10_email",):
                continue
            for f in glob.glob(sd + "best10_*.csv"):
                m = TS.search(f)
                if not m or "all_scanners" in f:
                    continue
                try:
                    x = pd.read_csv(f)
                except Exception:
                    continue
                x["Scanner"], x["Run"] = scn, run
                x["File_TS"] = pd.Timestamp(datetime.strptime(m.group(1) + m.group(2), "%Y%m%d%H%M"))
                rows.append(x)
    a = pd.concat(rows, ignore_index=True)
    a = a[pd.to_numeric(a.Entry_Price, errors="coerce").gt(0) & pd.to_numeric(a.Stop_Loss, errors="coerce").gt(0)]
    a["Ticker"] = a.Ticker.astype(str).str.upper()
    return a


def download(tickers, start):
    out = {}
    for i in range(0, len(tickers), 80):
        b = tickers[i:i + 80]
        try:
            raw = yf.download(b, start=start, group_by="ticker", auto_adjust=True, actions=False,
                              threads=True, progress=False)
        except Exception:
            continue
        for t in b:
            try:
                d = raw[t] if isinstance(raw.columns, pd.MultiIndex) else raw
                d = d.dropna(subset=["Close"])
                d.index = pd.to_datetime(d.index).tz_localize(None)
                if len(d) > 60:
                    out[t] = d[["Open", "High", "Low", "Close", "Volume"]].astype(float)
            except Exception:
                pass
        time.sleep(0.5)
    return out


def earnings(t):
    try:
        e = yf.Ticker(t).get_earnings_dates(limit=16)
        if e is None or len(e.index) == 0:
            return t, []
        return t, sorted({pd.Timestamp(x).tz_localize(None).normalize() for x in e.index})
    except Exception:
        return t, []


def feats(d):
    """Signal-day features for every bar of one ticker (DataFrame, same index)."""
    o, h, l, c, v = (d[k].values for k in ("Open", "High", "Low", "Close", "Volume"))
    pc = np.r_[np.nan, c[:-1]]
    tr = np.nanmax(np.vstack([h - l, np.abs(h - pc), np.abs(l - pc)]), axis=0)
    atr = pd.Series(tr).rolling(14).mean().values
    cs = pd.Series(c)
    def rsi(n):
        dd = cs.diff(); up = dd.clip(lower=0).ewm(alpha=1 / n, adjust=False).mean()
        dn = (-dd.clip(upper=0)).ewm(alpha=1 / n, adjust=False).mean()
        return (100 - 100 / (1 + up / dn.replace(0, np.nan))).fillna(100).values
    ema21 = cs.ewm(span=21, adjust=False).mean().values
    s20, s50, s200 = (cs.rolling(n).mean().values for n in (20, 50, 200))
    rng = np.where(h - l > 0, h - l, np.nan)
    vavg = pd.Series(v).rolling(20).mean().shift(1).values
    hi52 = pd.Series(h).rolling(252, min_periods=120).max().values
    up = (cs.diff() > 0).astype(int)
    streak = up.groupby((up != up.shift()).cumsum()).cumsum().where(up == 1, 0).values
    return pd.DataFrame({
        "atr_pct": atr / c * 100, "rsi14": rsi(14), "rsi2": rsi(2),
        "ext_ema21_atr": (c - ema21) / atr, "d_sma20": (c / s20 - 1) * 100, "d_sma50": (c / s50 - 1) * 100,
        "d_sma200": (c / s200 - 1) * 100, "sma50_slope10": (s50 / np.r_[np.full(10, np.nan), s50[:-10]] - 1) * 100,
        "ret5": (c / np.r_[np.full(5, np.nan), c[:-5]] - 1) * 100, "ret20": (c / np.r_[np.full(20, np.nan), c[:-20]] - 1) * 100,
        "gap_pct": (o / pc - 1) * 100, "range_atr": (h - l) / atr, "clv": (c - l) / rng,
        "upper_wick": (h - np.maximum(o, c)) / rng, "vol_ratio": v / vavg, "dollar_vol_m": vavg * c / 1e6,
        "off_52w_high": (c / hi52 - 1) * 100, "up_streak": streak, "atr": atr,
    }, index=d.index)


def spy_feats(spy):
    c = spy.Close
    return pd.DataFrame({"spy_above50": c > c.rolling(50).mean(), "spy_above20": c > c.rolling(20).mean(),
                         "spy_ret5": (c / c.shift(5) - 1) * 100}, index=spy.index)


def days_to_earn(dates, sig, idx=None):
    """Business days from the signal day to the next earnings report (after it), and since the last one."""
    nxt = next((x for x in dates if x > sig), None)
    prv = next((x for x in reversed(dates) if x <= sig), None)
    tn = int(np.busday_count(sig.date(), nxt.date())) if nxt is not None else np.nan
    tp = int(np.busday_count(prv.date(), sig.date())) if prv is not None else np.nan
    return tn, tp


def main():
    a = best10_picks()
    print(f"Best 10 picks: {len(a)} rows, {a.Run.nunique()} runs, {a.Ticker.nunique()} tickers ({time.time()-T0:.0f}s)")
    bt = []
    for r in BT_RUNS:
        for f in glob.glob(f"{ROOT}/{r}/*/bt_trades_*.csv"):
            try:
                x = pd.read_csv(f)
                if len(x): bt.append(x)
            except Exception:
                pass
    bt = pd.concat(bt, ignore_index=True) if bt else pd.DataFrame()
    print(f"Backtest trades: {len(bt)}")
    tick = sorted(set(a.Ticker) | (set(bt.ticker.astype(str)) if len(bt) else set()) | {"SPY"})
    data = download(tick, "2024-06-01")
    print(f"price data: {len(data)}/{len(tick)} ({time.time()-T0:.0f}s)")
    with ThreadPoolExecutor(8) as ex:
        earn = dict(ex.map(earnings, [t for t in tick if t != "SPY"]))
    print(f"earnings dates for {sum(1 for v in earn.values() if v)} tickers ({time.time()-T0:.0f}s)")
    spy = spy_feats(data["SPY"]); F = {t: feats(d) for t, d in data.items()}

    # ---- Best 10 outcomes
    out = []
    for _, p in a.iterrows():
        t = p.Ticker
        if t not in data: continue
        d, f = data[t], F[t]; idx = d.index
        # signal = the last session that had closed when the file was written (US close 21:00 UTC)
        cut = p.File_TS - pd.Timedelta(hours=21)
        k = int(idx.searchsorted(cut.normalize(), side="right")) - 1
        if k < 200: continue
        sig = idx[k]; e, s = float(p.Entry_Price), float(p.Stop_Loss)
        if s >= e: continue
        risk = e - s; fut = d.iloc[k + 1:]
        r = {"Run": p.Run, "Scanner": p.Scanner, "Ticker": t, "Signal_Date": sig.date(), "Best_Rank": p.get("Best_Rank"),
             "Best_Score": p.get("Best_Score"), "Trend_Score": p.get("Trend_Score"), "Catalyst_Score": p.get("Catalyst_Score"),
             "Value_Score": p.get("Value_Score"), "Analyst_Upside": p.get("Analyst_Upside_%"), "Entry": e, "Stop": s,
             "close_vs_entry_pct": (d.Close.iloc[k] / e - 1) * 100, "risk_pct": risk / e * 100,
             "risk_atr": risk / f.atr.iloc[k] if f.atr.iloc[k] > 0 else np.nan, "sessions": len(fut)}
        r.update({c: f[c].iloc[k] for c in f.columns if c != "atr"})
        r.update({c: spy[c].get(sig, np.nan) for c in spy.columns})
        r["earn_next"], r["earn_prev"] = days_to_earn(earn.get(t, []), sig, idx)
        if len(fut):
            hit = np.where(fut.Low.values <= s)[0]
            j = int(hit[0]) if len(hit) else None
            r["stopped"] = j is not None
            r["stop_session"] = j + 1 if j is not None else np.nan
            r["stop_gap"] = bool(j is not None and fut.Open.values[j] <= s)
            upto = fut.iloc[: (j + 1 if j is not None else len(fut))]
            r["mfe_R"] = (upto.High.max() - e) / risk
            r["mae_R"] = (upto.Low.min() - e) / risk
            r["day1_gap_pct"] = (fut.Open.values[0] / e - 1) * 100
            r["day1_R"] = (fut.Close.values[0] - e) / risk
            r["last_R"] = (min(fut.Open.values[j], s) - e) / risk if j is not None else (fut.Close.values[-1] - e) / risk
            r["max_R_before_stop"] = (fut.High.values[:j].max() - e) / risk if j else (np.nan if j == 0 else r["mfe_R"])
            sp = data["SPY"].Close
            end = fut.index[j] if j is not None else fut.index[-1]
            r["spy_ret_window"] = (sp.get(end, np.nan) / sp.get(sig, np.nan) - 1) * 100
            ex_ = earn.get(t, [])
            r["earn_in_window"] = any(sig < x <= end for x in ex_)
        out.append(r)
    o = pd.DataFrame(out)
    # how many scanners had the same ticker in the same run
    o["n_scanners"] = o.groupby(["Run", "Ticker"]).Scanner.transform("nunique")
    o.round(4).to_csv("sr_best10_outcomes.csv", index=False)
    print(f"sr_best10_outcomes.csv: {len(o)} picks ({time.time()-T0:.0f}s)")

    # ---- backtest features
    if len(bt):
        bt["signal_date"] = pd.to_datetime(bt.signal_date)
        rows = []
        for t, g in bt.groupby("ticker"):
            if t not in F: continue
            f = F[t]; idx = f.index; ex_ = earn.get(t, [])
            for _, x in g.iterrows():
                k = idx.searchsorted(x.signal_date)
                if k >= len(idx) or idx[k] != x.signal_date: continue
                sig = idx[k]
                r = {"scanner": x.scanner, "ticker": t, "signal_date": sig.date(), "R": x.R, "exit_reason": x.exit_reason,
                     "bars": x.bars, "risk_pct": x.risk_pct, "entry": x.entry, "stop": x.stop,
                     "risk_atr": (x.entry - x.stop) / f.atr.iloc[k] if f.atr.iloc[k] > 0 else np.nan,
                     "day1_gap_pct": (x.entry / data[t].Close.iloc[k] - 1) * 100}
                r.update({c: f[c].iloc[k] for c in f.columns if c != "atr"})
                r.update({c: spy[c].get(sig, np.nan) for c in spy.columns})
                r["earn_next"], r["earn_prev"] = days_to_earn(ex_, sig, idx)
                end = idx[min(k + int(x.bars), len(idx) - 1)]
                r["earn_in_window"] = any(sig < y <= end for y in ex_)
                rows.append(r)
        b = pd.DataFrame(rows)
        for c in b.select_dtypes("float").columns: b[c] = b[c].round(3)
        b.to_csv("sr_backtest_features.csv", index=False)
        print(f"sr_backtest_features.csv: {len(b)} trades ({time.time()-T0:.0f}s)")


if __name__ == "__main__":
    main()
