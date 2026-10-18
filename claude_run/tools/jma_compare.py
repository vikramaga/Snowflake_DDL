# Compares the repo's JMA approximation with the standard TradingView
# "Jurik Moving Average" formula for the tickers in claude_run/explain.txt,
# for several (length, phase, power) settings, over the last few days.
import ast, numpy as np, pandas as pd, yfinance as yf

def jma_tv(src, length=7, phase=50, power=2):
    """TradingView Jurik MA (everget) — the common Pine implementation."""
    phase_ratio = 0.5 if phase < -100 else (2.5 if phase > 100 else phase / 100 + 1.5)
    beta  = 0.45 * (length - 1) / (0.45 * (length - 1) + 2)
    alpha = beta ** power
    x = src.values.astype(float)
    out = np.full(len(x), np.nan)
    e0 = e1 = e2 = 0.0; j = 0.0
    for i, v in enumerate(x):
        if np.isnan(v):
            continue
        e0 = (1 - alpha) * v + alpha * e0
        e1 = (v - e0) * (1 - beta) + beta * e1
        e2 = (e0 + phase_ratio * e1 - j) * (1 - alpha) ** 2 + alpha ** 2 * e2
        j  = e2 + j
        out[i] = j
    return pd.Series(out, index=src.index)

# repo version, extracted from the scanner (no scan is run)
SRC = "daily_jma_cross_ema8_v1.py"
tree = ast.parse(open(SRC).read())
ns = {"np": np, "pd": pd}
exec(compile(ast.Module(body=[n for n in tree.body if isinstance(n, ast.FunctionDef)
                              and n.name in ("calc_jma",)], type_ignores=[]), SRC, "exec"), ns)

pd.set_option("display.width", 250)
for t in [l.strip().upper() for l in open("claude_run/explain.txt") if l.strip() and not l.startswith("#")]:
    for adj in (True, False):
        df = yf.Ticker(t).history(period="2y", auto_adjust=adj, actions=False)
        df.index = df.index.tz_localize(None).date
        c = df["Close"]
        tab = pd.DataFrame({"Close": c, "EMA8": c.ewm(span=8, adjust=False).mean(),
                            "repo JMA(13,40,2)": ns["calc_jma"](c, 13, 40, 2)})
        for L, ph, pw in [(13, 40, 2), (7, 50, 2), (7, 0, 2), (13, 50, 2), (14, 50, 2), (10, 50, 2), (9, 50, 2), (8, 50, 2), (5, 50, 2)]:
            tab[f"TV JMA({L},{ph},{pw})"] = jma_tv(c, L, ph, pw)
        print(f"\n===== {t}  (auto_adjust={adj}) =====")
        print(tab.tail(6).round(3).T.to_string())
