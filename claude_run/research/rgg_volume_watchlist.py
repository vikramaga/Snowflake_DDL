# Research / one-off watchlist (2026-11-15): across every stock in today's scanner results, find
#   RGG : red candle, green, green - the last green's volume above the first green's
#   RGGG: red candle, green, green, green - volume rising on at least the last 2 greens
# on the latest daily candles. Preferred (ranked first): the red candle is a small doji or
# hammer, and the higher-volume last green candle bounced on a moving average (its low at or
# within 0.5% of JMA / EMA8 / EMA21 / SMA21 / SMA50 / SMA150 / SMA200, close above it).
# Creates TradingView watchlist "RGG Volume - YYYY-MM-DD" (IST) with priority sections.
import glob, os, sys, time, subprocess
subprocess.call([sys.executable, "-m", "pip", "install", "-q", "yfinance"])
import numpy as np, pandas as pd, yfinance as yf
from datetime import datetime
from zoneinfo import ZoneInfo

RUNS = ["43665094951", "43668286377", "43676212528", "43681013594"]   # today's scanner runs (IST 2026-11-15)
tick = {}
for run in RUNS:
    for f in glob.glob(f"claude_run/results/{run}/*/*.csv"):
        if os.path.basename(f).startswith("best10"): continue
        try: d = pd.read_csv(f)
        except Exception: continue
        col = next((c for c in d.columns if c.lower() in ("ticker", "symbol")), None)
        if not col: continue
        for t in d[col].dropna().astype(str):
            t = t.strip().upper()
            if t.isalpha() and len(t) <= 5: tick.setdefault(t, set()).add(f.split("/")[3])
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


MAS = ("JMA", "EMA8", "EMA21", "SMA21", "SMA50", "SMA150", "SMA200")
def analyze(df):
    o, h, l, c, v = (df[k].values.astype(float) for k in ("Open", "High", "Low", "Close", "Volume"))
    cs = df["Close"].astype(float)
    ma = {"JMA": calc_jma(cs, 13, 40, 2).values, "EMA8": cs.ewm(span=8, adjust=False).mean().values,
          "EMA21": cs.ewm(span=21, adjust=False).mean().values, "SMA21": cs.rolling(21).mean().values,
          "SMA50": cs.rolling(50).mean().values, "SMA150": cs.rolling(150).mean().values,
          "SMA200": cs.rolling(200).mean().values}
    pc = np.concatenate([[np.nan], c[:-1]])
    atr = pd.Series(np.nanmax(np.vstack([h - l, abs(h - pc), abs(l - pc)]), axis=0)).rolling(14).mean().values
    i = len(df) - 1
    G = lambda k: c[k] > o[k]
    R = lambda k: c[k] < o[k]
    pat = None
    if R(i-2) and G(i-1) and G(i) and v[i] > v[i-1]:
        pat, r = "RGG", i - 2
    elif R(i-3) and G(i-2) and G(i-1) and G(i) and v[i] > v[i-1]:
        pat, r = "RGGG", i - 3
    if pat is None: return None
    rng = h[r] - l[r]; body = abs(c[r] - o[r]); lw = min(o[r], c[r]) - l[r]; uw = h[r] - max(o[r], c[r])
    small = rng <= 1.2 * atr[r] if np.isfinite(atr[r]) else True
    if rng > 0 and body <= 0.3 * rng and small: rtype = "Doji"
    elif rng > 0 and lw >= 2 * max(body, 0.02 * rng) and uw <= 0.3 * rng and small: rtype = "Hammer"
    else: rtype = "Plain"
    on_ma = [m for m in MAS if np.isfinite(ma[m][i]) and l[i] <= ma[m][i] * 1.005 and c[i] > ma[m][i]]
    vol3 = pat == "RGGG" and v[i-1] > v[i-2]
    return {"Pattern": pat, "Red_Candle": rtype, "On_MA": "/".join(on_ma), "Vol_Rising_All_Greens": "Yes" if (vol3 or pat == "RGG") else "No",
            "Vol_x_prev": round(v[i] / v[i-1], 2) if v[i-1] > 0 else None, "Close": round(c[i], 2),
            "Stop_RedLow": round(min(l[r:i+1]), 2), "Risk_%": round((c[i] - min(l[r:i+1])) / c[i] * 100, 1),
            "Above_SMA50": "Yes" if np.isfinite(ma["SMA50"][i]) and c[i] > ma["SMA50"][i] else "No"}

syms = sorted(tick); data = {}
for b in range(0, len(syms), 100):
    batch = syms[b:b+100]
    try:
        raw = yf.download(batch, period="400d", group_by="ticker", auto_adjust=True, progress=False, threads=True)
        for s in batch:
            try:
                df = raw[s].dropna(subset=["Close"])
                if len(df) >= 30: data[s] = df
            except Exception: pass
    except Exception as e: print("batch fail", e)
    time.sleep(1)
last = max(d.index[-1] for d in data.values())
print(f"data for {len(data)} tickers, last bar {last.date()}")

rows = []
for s, df in data.items():
    if df.index[-1] != last: continue
    try: r = analyze(df)
    except Exception: continue
    if r:
        r["Ticker"] = s; r["Scanners"] = len(tick[s]); rows.append(r)
res = pd.DataFrame(rows)
def section(r):
    nice = r.Red_Candle in ("Doji", "Hammer"); ma = bool(r.On_MA)
    if r.Pattern == "RGG":
        return 1 if nice and ma else 2 if ma else 3 if nice else 6
    return 4 if (ma and r.Vol_Rising_All_Greens == "Yes") else 5 if r.Vol_Rising_All_Greens == "Yes" else 7
NAMES = {1: "RGG doji-hammer + vol up on MA", 2: "RGG vol up on MA", 3: "RGG doji-hammer + vol up",
         4: "RGGG vol rising on MA", 5: "RGGG vol rising", 6: "RGG vol up", 7: "RGGG vol up last 2"}
if len(res):
    res["Section"] = res.apply(section, axis=1)
    res = res.sort_values(["Section", "Scanners", "Vol_x_prev"], ascending=[True, False, False])
    res["Section_Name"] = res.Section.map(NAMES)
    cols = ["Ticker", "Section_Name", "Pattern", "Red_Candle", "On_MA", "Vol_x_prev", "Vol_Rising_All_Greens",
            "Close", "Stop_RedLow", "Risk_%", "Above_SMA50", "Scanners"]
    res[cols].to_csv("rgg_volume_watchlist.csv", index=False)
    for k in sorted(res.Section.unique()):
        part = res[res.Section == k]
        print(f"\n=== {NAMES[k]}: {len(part)} ===")
        print(part[cols].head(60).to_string(index=False))

# -- TradingView list
try:
    sys.path.insert(0, "claude_run/tools")
    name = f"RGG Volume - {datetime.now(ZoneInfo('Asia/Kolkata')):%Y-%m-%d}"
    os.environ["TV_WATCHLIST"] = name
    import tv_watchlist_sync as tvs
    tvs.NAME = name
    s = tvs.session(); lists = tvs.get_lists(s); ex = tvs.exchange_map()
    symbols = []
    for k in sorted(res.Section.unique()):
        part = res[res.Section == k]
        symbols.append(f"###{NAMES[k]} ({len(part)})")
        symbols += [tvs.tv_symbol(t, ex) for t in part.Ticker]
    tvs.replace_list(s, lists, symbols)
    after = next((x for x in tvs.get_lists(s) if x.get("name") == name), None)
    got = [x for x in (after or {}).get("symbols") or [] if not str(x).startswith("###")]
    print(f"\n[TradingView] '{name}' - {len(got)} of {len(res)} symbols")
except Exception as e:
    print(f"\n[TradingView] failed: {type(e).__name__}: {e}")
