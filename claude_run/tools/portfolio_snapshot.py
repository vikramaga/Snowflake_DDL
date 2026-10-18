# Portfolio snapshot for a ticker list (used by Claude for position sizing).
# Reads tickers from claude_run/tools/snapshot_tickers.txt, writes
# portfolio_snapshot_<ts>.csv to the workspace root. No email.
import subprocess, sys, os
subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "--upgrade",
                       "yfinance", "pandas", "numpy"],
                      stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
import yfinance as yf, pandas as pd, numpy as np
from datetime import datetime, timedelta

here = os.path.dirname(os.path.abspath(__file__))
tickers = [t.strip().upper() for t in open(os.path.join(here, "snapshot_tickers.txt"))
           if t.strip() and not t.startswith("#")]
bench = ["SPY", "QQQ"]

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

raw = yf.download(tickers + bench, period="400d", group_by="ticker",
                  auto_adjust=True, progress=False, threads=True)

def get(sym):
    try:
        df = raw[sym].dropna(subset=["Close"])
        return df if len(df) > 80 else None
    except Exception:
        return None

spy = get("SPY")
spy_r63 = float(spy["Close"].iloc[-1] / spy["Close"].iloc[-64] - 1) if spy is not None else np.nan

rows = []
for t in tickers:
    df = get(t)
    if df is None:
        rows.append({"Ticker": t, "Note": "no data"}); continue
    c, h, l, v = df["Close"], df["High"], df["Low"], df["Volume"]
    jma = calc_jma(c); ema21 = c.ewm(span=21, adjust=False).mean(); sma50 = c.rolling(50).mean()
    sma200 = c.rolling(200).mean()
    tr = pd.concat([h - l, (h - c.shift()).abs(), (l - c.shift()).abs()], axis=1).max(axis=1)
    atr14 = tr.rolling(14).mean()
    stop10 = float(l.iloc[-11:-1].min())
    last = float(c.iloc[-1])
    try:
        on_0909 = float(c.loc[:"2026-10-09"].iloc[-1])
    except Exception:
        on_0909 = np.nan
    r63 = float(last / c.iloc[-64] - 1) if len(c) > 64 else np.nan
    earn = ""
    try:
        cal = yf.Ticker(t).calendar
        if isinstance(cal, dict):
            ed = cal.get("Earnings Date")
            if ed: earn = str(ed[0] if isinstance(ed, (list, tuple)) else ed)
    except Exception as e:
        earn = f"err:{type(e).__name__}"
    if not earn or earn.startswith("err"):
        try:
            edf = yf.Ticker(t).get_earnings_dates(limit=6)
            fut = [d for d in edf.index if d.tz_localize(None) >= pd.Timestamp.today().normalize() - pd.Timedelta(days=1)]
            if fut: earn = str(min(fut).date())
        except Exception:
            pass
    sector = industry = qtype = ""
    try:
        info = yf.Ticker(t).info
        sector, industry, qtype = info.get("sector", ""), info.get("industry", ""), info.get("quoteType", "")
    except Exception:
        pass
    rows.append({
        "Ticker": t, "Last_Date": str(c.index[-1].date()), "Close": round(last, 2),
        "Close_0909": round(on_0909, 2), "Chg_since_0909_%": round((last / on_0909 - 1) * 100, 1) if on_0909 == on_0909 else np.nan,
        "JMA": round(float(jma.iloc[-1]), 2), "EMA21": round(float(ema21.iloc[-1]), 2),
        "SMA50": round(float(sma50.iloc[-1]), 2), "SMA200": round(float(sma200.iloc[-1]), 2) if not np.isnan(sma200.iloc[-1]) else np.nan,
        "Close_gt_JMA": bool(last > jma.iloc[-1]), "JMA_gt_EMA21": bool(jma.iloc[-1] > ema21.iloc[-1]),
        "JMA_gt_SMA50": bool(jma.iloc[-1] > sma50.iloc[-1]),
        "Above_SMA200": bool(last > sma200.iloc[-1]) if not np.isnan(sma200.iloc[-1]) else None,
        "Above_JMA_%": round((last / jma.iloc[-1] - 1) * 100, 1),
        "Stop_10d_low": round(stop10, 2), "Risk_%": round((last - stop10) / last * 100, 1),
        "ATR14_%": round(float(atr14.iloc[-1]) / last * 100, 1),
        "Ret_63d_%": round(r63 * 100, 1), "RS_vs_SPY_63d": round((r63 - spy_r63) * 100, 1),
        "AvgDollarVol_20d_M": round(float((c * v).tail(20).mean()) / 1e6, 1),
        "Vol_last_vs_20d": round(float(v.iloc[-1] / v.iloc[-21:-1].mean()), 2),
        "Earnings_Date": earn, "Sector": sector, "Industry": industry, "QuoteType": qtype,
    })

out = pd.DataFrame(rows)
ts = datetime.utcnow().strftime("%Y%m%d_%H%M")
path = os.path.join(os.environ.get("GITHUB_WORKSPACE", os.getcwd()), f"portfolio_snapshot_{ts}.csv")
out.to_csv(path, index=False)
print(out.to_string(index=False))
print(f"SPY 63d return: {spy_r63*100:.1f}%")
print("saved", path)
