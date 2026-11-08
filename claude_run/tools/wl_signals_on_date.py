# Which stocks in your TradingView watchlists got a buy signal from any of the
# scanners on a given day (default: Friday 6 Nov 2026).
#
# Reads the scanners' saved results (claude_run/results/<run>/.../*.csv; for
# each scanner the newest file is used) and keeps the rows whose signal fired
# on SIGNAL_DATE:
#   daily scanners           Signal_Date == SIGNAL_DATE
#   RSI-average turn (daily) Signal_Date == SIGNAL_DATE
#   weekly scanners          the signal week is the week that ended on
#                            SIGNAL_DATE (weekly bars labelled by their Sunday)
# Not counted (no single-day signal): the RSI divergence scanners (swing-low
# based) and the monthly scanners (month still in progress).
# Then matches those tickers against every TradingView watchlist.
#
# Env: SIGNAL_DATE (YYYY-MM-DD), TV_SESSIONID / TV_SESSIONID_SIGN
# Writes wl_signals_<date>.csv (one row per stock) and
#        wl_signals_<date>_detail.csv (one row per stock and scanner).
import glob, os, re, sys
from datetime import datetime, timedelta

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
DATE = os.environ.get("SIGNAL_DATE", "2026-11-06")
D = pd.Timestamp(DATE)
WEEK_LABELS = {(D + timedelta(days=k)).strftime("%Y-%m-%d") for k in range(-4, 3)}   # Mon..Sun of that week
SKIP = ("daily_rsi_bullish_divergence", "weekly_rsi_bullish_divergence")


def latest_files():
    """scanner name -> newest results CSV for it (name taken from the file name)."""
    best = {}
    for f in glob.glob("claude_run/results/*/*/*.csv"):
        b = os.path.basename(f)
        if b.startswith(("best10_", "tv_", "wl_", "rsi_")):
            continue
        m = re.match(r"(.+?)_(\d{8})_(\d{4})\.csv$", b)
        if not m:
            continue
        name, ts = m.group(1), m.group(2) + m.group(3)
        if name not in best or ts > best[name][0]:
            best[name] = (ts, f)
    return {k: v[1] for k, v in best.items()}


def fired_rows(name, df):
    if name.startswith(SKIP) or name.startswith("monthly_reversal") or name.startswith("monthly_rsi_cross") \
            or name.startswith("monthly_3candle"):
        return df.iloc[0:0], ""
    if "Timeframe" in df.columns and "Signal_Date" in df.columns:          # RSI average turn (daily + weekly)
        sd = df["Signal_Date"].astype(str)
        daily = (df["Timeframe"] == "Daily") & (sd == DATE)
        weekly = (df["Timeframe"] == "Weekly") & sd.isin(WEEK_LABELS)
        out = df[daily | weekly].copy()
        out["_tf"] = out["Timeframe"]
        return out, ""
    if "Signal_Date" in df.columns:
        sd = df["Signal_Date"].astype(str)
        if name.startswith(("weekly_", "monthly_rsi_weekly")):
            return df[sd.isin(WEEK_LABELS)], "Weekly"
        return df[sd == DATE], "Daily"
    if "Curr_Week" in df.columns:                                          # weekly reversal scanners
        cw = df["Curr_Week"].astype(str).str[:10]
        return df[cw.isin(WEEK_LABELS)], "Weekly"
    return df.iloc[0:0], ""


def main():
    import tv_watchlist_sync as tv
    import wl_pullback_reclaim_check as w
    lists = tv.get_lists(tv.session())
    in_lists = {}
    for wl in lists:
        for x in wl.get("symbols") or []:
            x = str(x)
            if x.startswith("###"):
                continue
            ex = x.partition(":")[0] if ":" in x else ""
            if ex and ex not in w.KNOWN:
                continue
            t = w.to_yahoo(x)
            in_lists.setdefault(t, [])
            if wl.get("name") not in in_lists[t]:
                in_lists[t].append(wl.get("name"))
    print(f"{len(lists)} watchlists, {len(in_lists)} stocks; signals on {DATE}")

    rows, used = [], []
    for name, f in sorted(latest_files().items()):
        try:
            df = pd.read_csv(f)
        except Exception:
            continue
        if not len(df) or "Ticker" not in df.columns:
            used.append((name, 0, 0, f)); continue
        hit, tf = fired_rows(name, df)
        mine = hit[hit["Ticker"].astype(str).isin(in_lists)]
        used.append((name, len(hit), len(mine), f))
        for _, r in mine.iterrows():
            rows.append({"Ticker": r["Ticker"], "Scanner": name.replace("_v1", ""),
                         "Timeframe": r.get("_tf", tf) or tf, "Score": r.get("Score"),
                         "Price": r.get("Price"), "Entry_Price": r.get("Entry_Price"),
                         "Stop_Loss": r.get("Stop_Loss"), "Risk_%": r.get("Risk_%"),
                         "Signal_Date": r.get("Signal_Date", r.get("Curr_Week", DATE))})
    print("\nScanner results used (signals that day / in your watchlists):")
    for name, n, m, f in used:
        print(f"  {name:48s} {n:4d} / {m:3d}   {f.split('/')[2]}")
    det = pd.DataFrame(rows)
    tag = DATE.replace("-", "")
    if not len(det):
        print("\nNo watchlist stock had a signal that day."); pd.DataFrame().to_csv(f"wl_signals_{tag}.csv"); return
    det["Watchlists"] = det["Ticker"].map(lambda t: "; ".join(in_lists.get(t, [])))
    det.to_csv(f"wl_signals_{tag}_detail.csv", index=False)
    g = det.sort_values("Score", ascending=False).groupby("Ticker")
    summ = pd.DataFrame({
        "Signals": g["Scanner"].nunique(),
        "Scanners": g["Scanner"].apply(lambda s: ", ".join(dict.fromkeys(s))),
        "Best_Score": g["Score"].max(),
        "Price": g["Price"].first(), "Entry_Price": g["Entry_Price"].first(),
        "Stop_Loss": g["Stop_Loss"].first(), "Risk_%": g["Risk_%"].first(),
        "Watchlist_Count": g["Ticker"].first().map(lambda t: len(in_lists.get(t, []))),
        "Watchlists": g["Ticker"].first().map(lambda t: "; ".join(in_lists.get(t, []))),
    }).reset_index().sort_values(["Signals", "Best_Score"], ascending=False)
    summ.to_csv(f"wl_signals_{tag}.csv", index=False)
    print(f"\n{len(summ)} watchlist stocks had a buy signal on {DATE} ({len(det)} scanner signals)")
    with pd.option_context("display.width", 250, "display.max_colwidth", 90):
        print(summ.drop(columns=["Watchlists"]).head(80).to_string(index=False))


if __name__ == "__main__":
    main()
