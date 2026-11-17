# Research (2026-11-17): which chart patterns, among stocks that appeared in the scanner results,
# were most often followed by a rising price?
# 1. Picks: every US scanner CSV in claude_run/results (file name ..._YYYYMMDD_HHMM.csv = run time UTC).
#    Signal date = last US session that had closed before the run. One row per ticker + signal date.
# 2. Features on the signal date (daily bars): trend / MA stack, JMA, RSI, candles, volume, distance
#    from highs and MAs, momentum, volatility, relative strength vs SPY, market regime.
# 3. Outcome from the signal-date close: return after 5 and 10 sessions, and vs SPY over the same days.
# 4. Win rate (5-day return > 0) and average return for every single feature and every pair /
#    triple of features (min sample + min distinct signal dates), with a first-half / second-half check.
import glob, os, re, sys, time, itertools, subprocess
subprocess.call([sys.executable, "-m", "pip", "install", "-q", "yfinance"])
import numpy as np, pandas as pd, yfinance as yf

TS = re.compile(r"_(20\d{6})_(\d{4})\.csv$")
SKIP = ("best10", "scanner_scorecard", "rsi_", "pattern_", "rgg_", "tri_lookback", "diag", "india_",
        "monday_red", "sma150_base_reclaim_backtest")
picks = []
for f in glob.glob("claude_run/results/*/*/*.csv"):
    b = os.path.basename(f); m = TS.search(b)
    if not m or b.startswith(SKIP): continue
    try: d = pd.read_csv(f, usecols=lambda c: c.lower() in ("ticker", "symbol"))
    except Exception: continue
    if d.empty: continue
    run = pd.Timestamp(m.group(1) + m.group(2), tz="UTC")
    for t in d.iloc[:, 0].dropna().astype(str):
        t = t.strip().upper()
        if t.isalpha() and len(t) <= 5: picks.append((t, run, b[:m.start()]))
P = pd.DataFrame(picks, columns=["Ticker", "Run", "Scanner"])
print(f"{len(P)} scanner rows, {P.Ticker.nunique()} tickers, {P.Scanner.nunique()} scanners")

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


def rsi(c, n=14):
    d = c.diff(); up = d.clip(lower=0).ewm(alpha=1/n, adjust=False).mean(); dn = (-d.clip(upper=0)).ewm(alpha=1/n, adjust=False).mean()
    return 100 - 100 / (1 + up / dn.replace(0, np.nan))

def features(df, spy):
    o, h, l, c, v = (df[k].astype(float) for k in ("Open", "High", "Low", "Close", "Volume"))
    F = pd.DataFrame(index=df.index)
    e8 = c.ewm(span=8, adjust=False).mean(); s21 = c.rolling(21).mean(); s50 = c.rolling(50).mean()
    s150 = c.rolling(150).mean(); s200 = c.rolling(200).mean(); j = calc_jma(c, 13, 40, 2)
    r = rsi(c); pc = c.shift()
    tr = pd.concat([h - l, (h - pc).abs(), (l - pc).abs()], axis=1).max(axis=1); atr = tr.rolling(14).mean()
    body = (c - o).abs(); rng = (h - l).replace(0, np.nan); lw = np.minimum(o, c) - l; uw = h - np.maximum(o, c)
    g = c > o; vr = v / v.rolling(20).mean().shift()
    F["Trend: above SMA50"] = c > s50
    F["Trend: above SMA200"] = c > s200
    F["Trend: SMA50 > SMA150"] = s50 > s150
    F["Trend: full stack EMA8>SMA21>SMA50>SMA150"] = (e8 > s21) & (s21 > s50) & (s50 > s150)
    F["Trend: SMA50 rising (10d)"] = s50 > s50.shift(10)
    F["Trend: below SMA50 (downtrend)"] = (c < s50) & (s50 < s150)
    F["JMA: close above JMA"] = c > j
    F["JMA: JMA rising"] = j > j.shift()
    F["JMA: first close above JMA"] = (c > j) & (pc <= j.shift())
    F["RSI < 40"] = r < 40
    F["RSI 40-50"] = (r >= 40) & (r < 50)
    F["RSI 50-60"] = (r >= 50) & (r < 60)
    F["RSI 60-70"] = (r >= 60) & (r < 70)
    F["RSI >= 70"] = r >= 70
    F["RSI rising 3d"] = (r > r.shift()) & (r.shift() > r.shift(2))
    F["Candle: green"] = g
    F["Candle: red"] = c < o
    F["Candle: hammer"] = (lw >= 2 * body) & (uw <= 0.3 * rng)
    F["Candle: doji"] = body <= 0.1 * rng
    F["Candle: strong green (body > 1 ATR)"] = g & (body > atr)
    F["Candle: close in top 25% of range"] = (c - l) >= 0.75 * rng
    F["Candle: bullish engulfing"] = g & (c.shift() < o.shift()) & (c >= o.shift()) & (o <= c.shift())
    F["Candle: inside day"] = (h < h.shift()) & (l > l.shift())
    F["Candle: gap up"] = l > h.shift()
    F["Candle: NR7 (narrowest range of 7)"] = (h - l) <= (h - l).rolling(7).min()
    F["Seq: 2 green in a row"] = g & g.shift(fill_value=False)
    F["Seq: 3 green in a row"] = g & g.shift(fill_value=False) & g.shift(2, fill_value=False)
    F["Seq: red then green (RG)"] = g & (c.shift() < o.shift())
    F["Seq: 3 red in a row"] = (c < o) & (c.shift() < o.shift()) & (c.shift(2) < o.shift(2))
    F["Volume: < 0.8x avg"] = vr < 0.8
    F["Volume: 0.8-1.5x avg"] = (vr >= 0.8) & (vr < 1.5)
    F["Volume: 1.5-3x avg"] = (vr >= 1.5) & (vr < 3)
    F["Volume: >= 3x avg"] = vr >= 3
    F["Volume: rising 2 days"] = (v > v.shift()) & (v.shift() > v.shift(2))
    hi52 = h.rolling(252, min_periods=120).max(); d52 = c / hi52 - 1
    F["High: within 5% of 52w high"] = d52 >= -0.05
    F["High: 5-15% below 52w high"] = (d52 < -0.05) & (d52 >= -0.15)
    F["High: 15-30% below 52w high"] = (d52 < -0.15) & (d52 >= -0.30)
    F["High: > 30% below 52w high"] = d52 < -0.30
    F["High: new 20-day high"] = h >= h.rolling(20).max()
    d50 = c / s50 - 1
    F["MA: within 2% of SMA21"] = (c / s21 - 1).abs() <= 0.02
    F["MA: within 3% of SMA50"] = d50.abs() <= 0.03
    F["MA: > 15% above SMA50 (extended)"] = d50 > 0.15
    F["MA: low touched SMA21/SMA50, closed above"] = ((l <= s21) & (c > s21)) | ((l <= s50) & (c > s50))
    r5 = c / c.shift(5) - 1; r20 = c / c.shift(20) - 1
    F["Mom: 5d return < -5%"] = r5 < -0.05
    F["Mom: 5d return -5..0%"] = (r5 >= -0.05) & (r5 < 0)
    F["Mom: 5d return 0..5%"] = (r5 >= 0) & (r5 < 0.05)
    F["Mom: 5d return > 5%"] = r5 >= 0.05
    F["Mom: 20d return > 10%"] = r20 > 0.10
    F["Mom: 20d return < -10%"] = r20 < -0.10
    ap = atr / c
    F["Vol: ATR < 2.5% of price"] = ap < 0.025
    F["Vol: ATR 2.5-5%"] = (ap >= 0.025) & (ap < 0.05)
    F["Vol: ATR > 5%"] = ap >= 0.05
    F["Price: < $10"] = c < 10
    F["Price: $10-50"] = (c >= 10) & (c < 50)
    F["Price: > $50"] = c >= 50
    sp = spy.reindex(df.index).ffill()
    rs = (c / c.shift(63)) / (sp / sp.shift(63))
    F["RS: beats SPY over 3 months"] = rs > 1
    F["RS: lags SPY over 3 months"] = rs < 1
    return F.fillna(False).astype(bool), c

print("downloading ...")
tick = sorted(P.Ticker.unique()); data = {}
for b in range(0, len(tick), 100):
    batch = tick[b:b + 100]
    try:
        raw = yf.download(batch, period="420d", group_by="ticker", auto_adjust=True, progress=False, threads=True)
        for s in batch:
            try:
                df = raw[s].dropna(subset=["Close"])
                if len(df) >= 210: data[s] = df
            except Exception: pass
    except Exception as e: print("batch fail", e)
    time.sleep(1)
spyd = yf.download("SPY", period="420d", auto_adjust=True, progress=False)
if isinstance(spyd.columns, pd.MultiIndex): spyd.columns = spyd.columns.get_level_values(0)
spy = spyd["Close"].astype(float)
spy_s50 = spy.rolling(50).mean()
print(f"data for {len(data)} tickers; last bar {spy.index[-1].date()}")

sessions = spy.index
_close_utc = (pd.DatetimeIndex(sessions).tz_localize(None).normalize() + pd.Timedelta(hours=21)).tz_localize("UTC")
def sig_date(run):
    ok = _close_utc <= run                        # sessions that had closed (21:00 UTC) before the run
    return sessions[ok][-1] if ok.any() else None
P["Date"] = P.Run.map(sig_date)
P = P.dropna(subset=["Date"])
rows = []
for s, grp in P.groupby("Ticker"):
    df = data.get(s)
    if df is None: continue
    F, c = features(df, spy)
    idx = {d: k for k, d in enumerate(df.index)}
    for d, g2 in grp.groupby("Date"):
        k = idx.get(d)
        if k is None: continue
        rec = {"Ticker": s, "Date": d, "Scanners": "|".join(sorted(set(g2.Scanner))), "N_Scanners": g2.Scanner.nunique()}
        for n in (5, 10):
            if k + n < len(df):
                rec[f"R{n}"] = c.iloc[k + n] / c.iloc[k] - 1
                sk = sessions.get_loc(d)
                rec[f"X{n}"] = rec[f"R{n}"] - (spy.iloc[sk + n] / spy.iloc[sk] - 1) if sk + n < len(spy) else np.nan
        rec["Market: SPY above SMA50"] = bool(spy.loc[d] > spy_s50.loc[d])
        rec.update(F.iloc[k].to_dict())
        rows.append(rec)
D = pd.DataFrame(rows)
D = D[D.R5.notna()].copy()
FEATS = [c for c in D.columns if c not in ("Ticker", "Date", "Scanners", "N_Scanners", "R5", "R10", "X5", "X10")]
D["Multi-scanner (2+)"] = D.N_Scanners >= 2
FEATS.append("Multi-scanner (2+)")
dates = sorted(D.Date.unique()); mid = dates[len(dates) // 2]
print(f"\n{len(D)} ticker-days with a 5-day outcome, {D.Ticker.nunique()} tickers, {len(dates)} signal dates "
      f"{pd.Timestamp(dates[0]).date()} .. {pd.Timestamp(dates[-1]).date()}")
base = {"n": len(D), "win5": (D.R5 > 0).mean(), "avg5": D.R5.mean(), "med5": D.R5.median(), "xwin5": (D.X5 > 0).mean(),
        "avg10": D.R10.mean(), "win10": (D.R10 > 0).mean()}
print("BASE RATE (all picks): " + ", ".join(f"{k} {v:.3f}" for k, v in base.items()))

def stats(mask, label):
    s = D[mask]
    if len(s) == 0: return None
    a, b2 = s[s.Date < mid], s[s.Date >= mid]
    return {"Pattern": label, "N": len(s), "Dates": s.Date.nunique(), "Win5_%": round((s.R5 > 0).mean() * 100, 1),
            "Avg5_%": round(s.R5.mean() * 100, 2), "Med5_%": round(s.R5.median() * 100, 2),
            "BeatSPY5_%": round((s.X5 > 0).mean() * 100, 1), "AvgX5_%": round(s.X5.mean() * 100, 2),
            "Win10_%": round((s.R10 > 0).mean() * 100, 1) if s.R10.notna().any() else None,
            "Avg10_%": round(s.R10.mean() * 100, 2) if s.R10.notna().any() else None,
            "Win5_1stHalf_%": round((a.R5 > 0).mean() * 100, 1) if len(a) else None,
            "Win5_2ndHalf_%": round((b2.R5 > 0).mean() * 100, 1) if len(b2) else None,
            "N_1st": len(a), "N_2nd": len(b2)}

out = [stats(D[f], f) for f in FEATS]
singles = pd.DataFrame([o for o in out if o and o["N"] >= 150]).sort_values("Win5_%", ascending=False)
singles.to_csv("pattern_mining_single.csv", index=False)
pd.set_option("display.width", 250); pd.set_option("display.max_columns", 20)
print("\n=== SINGLE FEATURES (N >= 150), by 5-day win rate ===")
print(singles.to_string(index=False))

cand = [f for f in FEATS if D[f].sum() >= 150]
combo = []
for k in (2, 3):
    for fs in itertools.combinations(cand, k):
        m = D[list(fs)].all(axis=1)
        n = int(m.sum())
        if n < (120 if k == 2 else 80): continue
        st = stats(m, " + ".join(fs))
        if st["Dates"] < 6: continue
        combo.append(st)
C = pd.DataFrame(combo) if combo else pd.DataFrame(columns=list(singles.columns))
C["Stable"] = (C["Win5_1stHalf_%"] >= base["win5"] * 100 + 5) & (C["Win5_2ndHalf_%"] >= base["win5"] * 100 + 5) \
              & (C["N_1st"] >= 25) & (C["N_2nd"] >= 25)
C = C.sort_values(["Win5_%", "Avg5_%"], ascending=False)
C.to_csv("pattern_mining_combos.csv", index=False)
print(f"\n=== TOP 40 COMBOS (pairs N>=120, triples N>=80, >=6 dates) by 5-day win rate - {len(C)} tested ===")
print(C.head(40).to_string(index=False))
print("\n=== TOP 25 STABLE COMBOS (win rate >= base+5 pts in BOTH halves) ===")
print(C[C.Stable].head(25).to_string(index=False))
print("\n=== TOP 25 by avg 5-day excess return vs SPY (N>=150 combos) ===")
print(C[C.N >= 150].sort_values("AvgX5_%", ascending=False).head(25).to_string(index=False))

sc = []
for name in sorted({x for s in D.Scanners for x in s.split("|")}):
    m = D.Scanners.str.contains(rf"(?:^|\|){re.escape(name)}(?:\||$)")
    if m.sum() >= 40: sc.append(stats(m, name))
S = pd.DataFrame(sc).sort_values("Win5_%", ascending=False)
S.to_csv("pattern_mining_scanners.csv", index=False)
print("\n=== BY SCANNER (N >= 40) ===")
print(S.to_string(index=False))
