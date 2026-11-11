# Scanner return tracker (run on demand by ClaudeRunner).
#
# For every pick in the saved results of the daily runs (claude_run/results/<run>/<scanner>/),
# measures what happened after the scan was published:
#   entry  = close of the last US session before the run (what the email showed)
#   now    = latest price (includes today's bar, live if the market is open)
#   managed return = exit at Stop_Loss if a later low touched it (at the open if it gapped
#                    below), otherwise the return to now
#   vs SPY = the same window's SPY return subtracted
# A stock repeated by the same scanner within 10 days counts once (first appearance).
#
# Three views, per scanner:
#   1. TODAY'S WATCHLIST - the Best 10 lists of the latest run (= today's TradingView list)
#   2. BEST 10 OVERALL   - every Best 10 pick since Best 10 started
#   3. ALL MATCHES       - every match since the oldest saved run
# Outputs: scanner_returns_picks.csv, scanner_returns_summary.csv, scanner_returns.html
import glob, os, re, subprocess, sys, time
from datetime import datetime, timedelta, timezone

subprocess.call([sys.executable, "-m", "pip", "install", "-q", "yfinance", "pandas", "numpy"],
                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
import numpy as np
import pandas as pd
import yfinance as yf

ROOT = "claude_run/results"
DEDUPE_DAYS = 10
TS = re.compile(r"_(20\d{6})_(\d{4})\.csv$")


def runs():
    out = []
    for d in sorted(glob.glob(f"{ROOT}/[0-9]*/")):
        subs = [s for s in glob.glob(d + "*/") if glob.glob(s + "*.csv")]
        if len(subs) >= 10:
            out.append(d.rstrip("/"))
    return out


def picks():
    rows = []
    for rd in runs():
        run = os.path.basename(rd)
        for sd in sorted(glob.glob(rd + "/*/")):
            scn = os.path.basename(sd.rstrip("/"))
            files = [f for f in glob.glob(sd + "*.csv") if not os.path.basename(f).startswith(("best10_", "tv_"))]
            b10 = glob.glob(sd + "best10_*.csv")
            best = set()
            for f in b10:
                try:
                    best |= set(pd.read_csv(f)["Ticker"].astype(str).str.strip().str.upper())
                except Exception:
                    pass
            for f in files:
                m = TS.search(f)
                if not m:
                    continue
                ts = datetime.strptime(m.group(1) + m.group(2), "%Y%m%d%H%M")
                try:
                    df = pd.read_csv(f)
                except Exception:
                    continue
                tcol = "Ticker" if "Ticker" in df.columns else ("Symbol" if "Symbol" in df.columns else None)
                if tcol is None or "Price" not in df.columns or df.empty:
                    continue
                for _, r in df.iterrows():
                    t = str(r[tcol]).strip().upper()
                    if not t or t == "NAN":
                        continue
                    stop = r.get("Stop_Loss", np.nan)
                    rows.append({"Run": run, "Run_UTC": ts, "Scanner": scn, "Ticker": t,
                                 "CSV_Price": float(r["Price"]) if pd.notna(r["Price"]) else np.nan,
                                 "Stop_Loss": float(stop) if pd.notna(stop) else np.nan,
                                 "In_Best10": t in best, "Has_Best10": bool(b10)})
    return pd.DataFrame(rows)


def download(tickers, start):
    data = {}
    tickers = sorted(set(tickers))
    end = (datetime.now(timezone.utc) + timedelta(days=2)).strftime("%Y-%m-%d")
    for i in range(0, len(tickers), 100):
        batch = tickers[i:i + 100]
        try:
            raw = yf.download(batch, start=start, end=end, group_by="ticker", auto_adjust=True,
                              actions=False, threads=True, progress=False)
        except Exception as err:
            print("download error", err); raw = None
        for t in batch:
            try:
                d = raw[t] if isinstance(raw.columns, pd.MultiIndex) else raw
                d = d.dropna(subset=["Close"])
                if len(d):
                    d.index = pd.to_datetime(d.index).tz_localize(None).normalize()
                    data[t] = d
            except Exception:
                pass
        time.sleep(1)
        print(f"  prices {min(i + 100, len(tickers))}/{len(tickers)}", flush=True)
    return data


def measure(p, px, spy):
    out = []
    for _, r in p.iterrows():
        d = px.get(r.Ticker)
        if d is None:
            continue
        base_day = pd.Timestamp(r.Run_UTC.date())
        before = d[d.index < base_day]
        if before.empty:
            continue
        b = before.index[-1]
        entry = float(before.Close.iloc[-1])
        if np.isfinite(r.CSV_Price) and r.CSV_Price > 0 and abs(entry / r.CSV_Price - 1) > 0.03:
            entry = r.CSV_Price                       # data mismatch: use the price the email showed
        after = d[d.index > b]
        last = float(d.Close.iloc[-1])
        ret = last / entry - 1
        r5 = float(after.Close.iloc[4] / entry - 1) if len(after) >= 5 else np.nan
        stop, hit, managed = r.Stop_Loss, False, ret
        if np.isfinite(stop) and 0 < stop < entry and len(after):
            lows = after.Low.values
            k = np.argmax(lows <= stop) if (lows <= stop).any() else -1
            if k >= 0:
                hit = True
                managed = min(float(after.Open.iloc[k]), stop) / entry - 1
        sp = spy[spy.index >= b]
        spy_ret = float(sp.Close.iloc[-1] / sp.Close.iloc[0] - 1) if len(sp) else np.nan
        out.append({**r.to_dict(), "Base_Date": b.date(), "Entry": round(entry, 4), "Last": round(last, 4),
                    "Sessions": len(after), "Ret_Now_%": ret * 100, "Ret_5d_%": r5 * 100,
                    "Stop_Hit": hit, "Ret_Managed_%": managed * 100, "SPY_%": spy_ret * 100,
                    "Excess_%": (managed - spy_ret) * 100,
                    "Max_Up_%": (float(after.High.max()) / entry - 1) * 100 if len(after) else 0.0})
    return pd.DataFrame(out)


def dedupe(m):
    m = m.sort_values("Base_Date")
    keep, last = [], {}
    for idx, r in m.iterrows():
        k = (r.Scanner, r.Ticker)
        if k in last and (pd.Timestamp(r.Base_Date) - pd.Timestamp(last[k])).days < DEDUPE_DAYS:
            continue
        last[k] = r.Base_Date
        keep.append(idx)
    return m.loc[keep]


def summarize(m, view):
    if m.empty:
        return pd.DataFrame()
    g = m.groupby("Scanner")
    s = pd.DataFrame({
        "Picks": g.size(),
        "Avg_Ret_%": g["Ret_Now_%"].mean(),
        "Median_Ret_%": g["Ret_Now_%"].median(),
        "Win_%": g["Ret_Now_%"].apply(lambda x: (x > 0).mean() * 100),
        "Avg_Managed_%": g["Ret_Managed_%"].mean(),
        "Stop_Hit_%": g["Stop_Hit"].mean() * 100,
        "Avg_vs_SPY_%": g["Excess_%"].mean(),
        "Avg_5d_%": g["Ret_5d_%"].mean(),
        "N_5d": g["Ret_5d_%"].count(),
        "Avg_Max_Up_%": g["Max_Up_%"].mean(),
        "Avg_Sessions": g["Sessions"].mean(),
    }).round(2).sort_values("Avg_Managed_%", ascending=False)
    allrow = pd.DataFrame([{"Picks": len(m), "Avg_Ret_%": m["Ret_Now_%"].mean(), "Median_Ret_%": m["Ret_Now_%"].median(),
                            "Win_%": (m["Ret_Now_%"] > 0).mean() * 100, "Avg_Managed_%": m["Ret_Managed_%"].mean(),
                            "Stop_Hit_%": m["Stop_Hit"].mean() * 100, "Avg_vs_SPY_%": m["Excess_%"].mean(),
                            "Avg_5d_%": m["Ret_5d_%"].mean(), "N_5d": m["Ret_5d_%"].count(),
                            "Avg_Max_Up_%": m["Max_Up_%"].mean(), "Avg_Sessions": m["Sessions"].mean()}],
                          index=["ALL SCANNERS"]).round(2)
    s = pd.concat([s, allrow])
    s.insert(0, "View", view)
    return s.reset_index().rename(columns={"index": "Scanner"})


def main():
    p = picks()
    if p.empty:
        sys.exit("no saved scanner results found")
    latest = p.Run_UTC.max()
    latest_run = p.loc[p.Run_UTC == latest, "Run"].iloc[0]
    print(f"{len(p)} pick rows from {p.Run.nunique()} runs; oldest {p.Run_UTC.min():%Y-%m-%d}, latest run {latest_run}")
    start = (p.Run_UTC.min() - timedelta(days=10)).strftime("%Y-%m-%d")
    px = download(list(p.Ticker.unique()) + ["SPY"], start)
    spy = px["SPY"]
    m = measure(p, px, spy)
    print(f"priced {len(m)} of {len(p)} rows; latest bar {spy.index[-1].date()}")

    today = m[(m.Run == latest_run) & m.In_Best10]
    t2 = today.groupby("Ticker").Scanner.nunique()
    multi = today[today.Ticker.isin(t2[t2 >= 2].index)].drop_duplicates("Ticker").assign(Scanner="(found by 2+ scanners)")
    best_all = dedupe(m[m.In_Best10])
    allm = dedupe(m)
    summ = pd.concat([summarize(pd.concat([today, multi]), "1 Today's watchlist"),
                      summarize(best_all, "2 Best 10 overall"),
                      summarize(allm, "3 All matches overall")])
    m.to_csv("scanner_returns_picks.csv", index=False)
    summ.to_csv("scanner_returns_summary.csv", index=False)
    pd.set_option("display.width", 250); pd.set_option("display.max_rows", 500)
    for v, g in summ.groupby("View"):
        print("\n" + "=" * 100 + f"\n{v}\n" + "=" * 100)
        print(g.drop(columns="View").to_string(index=False))
    meta = (f"Latest price bar: {spy.index[-1].date()} (live if the market is open) | "
            f"runs {p.Run_UTC.min():%Y-%m-%d} to {latest:%Y-%m-%d} | today's list = run {latest_run}")
    html = ["<html><body style='font-family:Arial;font-size:12px'><h2>Scanner returns</h2><p>" + meta + "</p>"]
    for v, g in summ.groupby("View"):
        html.append(f"<h3>{v}</h3>" + g.drop(columns="View").to_html(index=False, border=1))
    open("scanner_returns.html", "w").write("".join(html) + "</body></html>")
    print("\n" + meta)


if __name__ == "__main__":
    main()
