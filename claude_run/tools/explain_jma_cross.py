# Prints the last few days of Close / JMA / EMA8 for the tickers in
# claude_run/explain.txt, using the exact functions from
# daily_jma_cross_ema8_v1.py (extracted with ast, so no scan is run),
# and shows the cross day, confirm day and cross price (stop).
import ast, numpy as np, pandas as pd, yfinance as yf

SRC = "daily_jma_cross_ema8_v1.py"
tree = ast.parse(open(SRC).read())
want = {"CFG", "calc_jma", "calc_ema", "cross_price", "check_pattern_at"}
nodes = [n for n in tree.body if (isinstance(n, ast.FunctionDef) and n.name in want) or
         (isinstance(n, ast.Assign) and any(getattr(t, "id", None) in want for t in n.targets))]
ns = {"np": np, "pd": pd}
exec(compile(ast.Module(body=nodes, type_ignores=[]), SRC, "exec"), ns)
CFG = ns["CFG"]

tickers = [l.strip().upper() for l in open("claude_run/explain.txt") if l.strip() and not l.startswith("#")]
for t in tickers:
    df = yf.Ticker(t).history(period="2y", auto_adjust=True, actions=False)
    df.index = df.index.tz_localize(None)
    close = df["Close"]
    jma = ns["calc_jma"](close, CFG["jma_period"], CFG["jma_phase"], CFG["jma_power"])
    ema = ns["calc_ema"](close, CFG["ema_period"])
    i = len(close) - 1
    ok, d = ns["check_pattern_at"](close, jma, ema, i, CFG)
    tab = pd.DataFrame({"Close": close, "JMA": jma, "EMA8": ema})
    tab["JMA-EMA8"] = tab.JMA - tab.EMA8
    tab["JMA vs EMA8"] = np.where(tab["JMA-EMA8"] > 0, "JMA above", "JMA below/equal")
    tab["Close vs JMA"] = np.where(tab.Close > tab.JMA, "above JMA", "below JMA")
    print(f"\n===== {t} =====")
    print(tab.tail(6).round(3).to_string())
    if d.get("crossed"):
        x = d["cross_idx"]
        print(f"\nCross day   : {close.index[x].date()}  (JMA {jma.iloc[x-1]:.3f} <= EMA8 {ema.iloc[x-1]:.3f} on "
              f"{close.index[x-1].date()}, then JMA {jma.iloc[x]:.3f} > EMA8 {ema.iloc[x]:.3f})")
        print(f"Cross price : {d['cross_price']:.4f}  (t = {d['cross_t']:.3f} of the way from "
              f"{close.index[x-1].date()} to {close.index[x].date()})")
        print(f"Confirm day : {close.index[i].date()}  close {close.iloc[i]:.3f} vs JMA {jma.iloc[i]:.3f}")
        print(f"Signal      : {ok}   steps crossed={d['crossed']} confirm={d['confirm']} valid={d['valid']}")
    else:
        print("\nNo cross on the day before the latest candle.")
