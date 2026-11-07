# Checks the stocks in a TradingView watchlist for the "pullback to the 50 and
# JMA reclaim" pattern (Vik's RVTY example) within the last 10 sessions.
#
# Pattern on signal day t (daily bars):
#   1. Uptrend: SMA50 > SMA150 > SMA200, SMA50 and SMA200 higher than 20
#      sessions ago, close > SMA150.
#   2. Pullback: swing high in the 3-15 sessions before t, then 2-10 sessions
#      of pullback in which the JMA(13,40,2) falls on at least 2 days (the line
#      turns red on the chart) and is still falling or flat the day before t.
#   3. The pullback reaches the EMA21 / SMA50 zone (low within 0.5% of the
#      higher of the two, or below), dips at least 1.5 ATR14 from the high,
#      and holds: no close more than 3% under the SMA50 and no low under SMA150.
#   4. Reclaim day t: green candle, body >= 0.6 ATR14, close back above the
#      JMA (the day before closed at or below it) and above EMA21.
#
# Usage (ClaudeRunner, needs TV_SESSIONID / TV_SESSIONID_SIGN):
#   python claude_run/tools/wl_pullback_reclaim_check.py
# Env: TV_WATCHLIST (default "Best 10 - all scanners"), LOOKBACK (default 10)
# Writes wl_pullback_reclaim_<date>.csv and prints the matches.
import os, sys
from datetime import datetime, timedelta

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
LOOKBACK = int(os.environ.get("LOOKBACK", "10"))
CONTROL = ["RVTY"]          # the example chart, to confirm the rule sees it


def calc_jma(series, period=13, phase=40, power=2):
    phase_ratio = 0.5 if phase < -100 else (2.5 if phase > 100 else phase / 100.0 + 1.5)
    beta = 0.45 * (period - 1) / (0.45 * (period - 1) + 2)
    alpha = beta ** power
    vals = np.asarray(series.values, dtype=float)
    out = np.full(len(vals), np.nan)
    e0 = e1 = e2 = jma = 0.0
    for i, v in enumerate(vals):
        if np.isnan(v):
            continue
        e0 = (1 - alpha) * v + alpha * e0
        e1 = (v - e0) * (1 - beta) + beta * e1
        e2 = (e0 + phase_ratio * e1 - jma) * (1 - alpha) ** 2 + alpha ** 2 * e2
        jma = e2 + jma
        out[i] = jma
    return pd.Series(out, index=series.index)


def indicators(d):
    d = d.copy()
    c = d["Close"]
    d["JMA"] = calc_jma(c)
    d["EMA21"] = c.ewm(span=21, adjust=False).mean()
    d["SMA50"], d["SMA150"], d["SMA200"] = c.rolling(50).mean(), c.rolling(150).mean(), c.rolling(200).mean()
    pc = c.shift(1)
    tr = pd.concat([d["High"] - d["Low"], (d["High"] - pc).abs(), (d["Low"] - pc).abs()], axis=1).max(axis=1)
    d["ATR14"] = tr.ewm(alpha=1 / 14, adjust=False).mean()
    return d


def check_at(d, t):
    """Returns a dict with the pattern details if day index t is a signal, else None."""
    if t < 220:
        return None
    r = d.iloc[t]
    o, h, l, c = d["Open"].values, d["High"].values, d["Low"].values, d["Close"].values
    j, e21, s50, s150, s200, atr = (d[k].values for k in ("JMA", "EMA21", "SMA50", "SMA150", "SMA200", "ATR14"))
    # 1. uptrend
    if not (s50[t] > s150[t] > s200[t] and s50[t] > s50[t - 20] and s200[t] > s200[t - 20] and c[t] > s150[t]):
        return None
    # 4. reclaim day
    if not (c[t] > o[t] and (c[t] - o[t]) >= 0.6 * atr[t] and c[t] > j[t] and c[t] > e21[t] and c[t - 1] <= j[t - 1]):
        return None
    # 2. swing high 3-15 sessions back, pullback window after it
    win = range(t - 15, t - 2)
    hi = max(win, key=lambda i: h[i])
    pb = list(range(hi + 1, t))
    if not (2 <= len(pb) <= 10):
        return None
    falls = sum(1 for i in pb if j[i] < j[i - 1])
    if falls < 2 or j[t - 1] > j[t - 2]:
        return None
    # 3. reached the EMA21 / SMA50 zone, deep enough, and held
    lo = min(pb, key=lambda i: l[i])
    zone = max(e21[lo], s50[lo])
    if l[lo] > zone * 1.005:
        return None
    if (h[hi] - l[lo]) < 1.5 * atr[t]:
        return None
    if any(c[i] < s50[i] * 0.97 or l[i] < s150[i] for i in pb):
        return None
    return {"Signal_Date": d.index[t].strftime("%Y-%m-%d"), "Swing_High": round(h[hi], 2),
            "Pullback_Low": round(l[lo], 2), "Pullback_%": round((h[hi] - l[lo]) / h[hi] * 100, 1),
            "Pullback_Days": len(pb), "Low_vs_SMA50_%": round((l[lo] / s50[lo] - 1) * 100, 1),
            "Reclaim_Close": round(c[t], 2), "Reclaim_Body_ATR": round((c[t] - o[t]) / atr[t], 2),
            "Stop_(pullback_low)": round(l[lo], 2)}


def scan(d, last_n):
    d = indicators(d)
    n = len(d)
    hits = []
    for t in range(max(220, n - last_n), n):
        x = check_at(d, t)
        if x:
            x["Sessions_Ago"] = n - 1 - t
            x["Close_Now"] = round(float(d["Close"].iloc[-1]), 2)
            x["Change_Since_%"] = round((d["Close"].iloc[-1] / d["Close"].iloc[t] - 1) * 100, 1)
            x["Above_Pullback_Low"] = "Yes" if d["Close"].iloc[-1] > x["Pullback_Low"] else "No"
            hits.append(x)
    return hits


def to_yahoo(sym):
    ex, _, t = sym.partition(":")
    t = t or ex
    if ex == "NSE":
        return t + ".NS"
    if ex == "BSE":
        return t + ".BO"
    return t.replace(".", "-")


def download(tickers):
    import yfinance as yf
    out = {}
    end = datetime.today() + timedelta(days=1)
    start = end - timedelta(days=560)
    for i in range(0, len(tickers), 50):
        b = tickers[i:i + 50]
        raw = yf.download(b, start=start.strftime("%Y-%m-%d"), end=end.strftime("%Y-%m-%d"), group_by="ticker",
                          auto_adjust=True, actions=False, threads=True, progress=False)
        for t in b:
            try:
                x = (raw[t] if isinstance(raw.columns, pd.MultiIndex) else raw).dropna(subset=["Close"])
                if len(x) > 230:
                    out[t] = x
            except Exception:
                pass
    return out


def main():
    import tv_watchlist_sync as tv
    name = os.environ.get("TV_WATCHLIST", "Best 10 - all scanners")
    s = tv.session()
    wl = next((x for x in tv.get_lists(s) if x.get("name") == name), None)
    if wl is None:
        sys.exit(f"watchlist {name!r} not found")
    syms = [x for x in wl.get("symbols") or [] if not str(x).startswith("###")]
    section, sec_of = None, {}
    for x in wl.get("symbols") or []:
        if str(x).startswith("###"):
            section = x[3:]
        else:
            sec_of.setdefault(to_yahoo(x), section)
    tickers = list(dict.fromkeys(to_yahoo(x) for x in syms))
    print(f"Watchlist {name!r}: {len(tickers)} stocks; checking the last {LOOKBACK} sessions")
    data = download(tickers + [t for t in CONTROL if t not in tickers])
    print(f"price data for {len(data)} tickers")

    for t in CONTROL:
        if t in data:
            hits = scan(data[t], 260)
            print(f"\nControl {t} - signals in the past year: " +
                  (", ".join(f"{h['Signal_Date']} (pullback {h['Pullback_%']}%, low {h['Low_vs_SMA50_%']:+}% vs SMA50)" for h in hits) or "none"))

    rows = []
    for t in tickers:
        if t not in data:
            continue
        for h in scan(data[t], LOOKBACK):
            rows.append({"Ticker": t, "Watchlist_Section": sec_of.get(t, ""), **h})
    df = pd.DataFrame(rows)
    out = f"wl_pullback_reclaim_{datetime.today():%Y%m%d}.csv"
    if len(df):
        df = df.sort_values(["Sessions_Ago", "Reclaim_Body_ATR"], ascending=[True, False]).drop_duplicates("Ticker")
    df.to_csv(out, index=False)
    print(f"\n{len(df)} of {len(data)} stocks show the pattern in the last {LOOKBACK} sessions -> {out}")
    if len(df):
        with pd.option_context("display.width", 250, "display.max_columns", 30):
            print(df.to_string(index=False))


if __name__ == "__main__":
    main()
