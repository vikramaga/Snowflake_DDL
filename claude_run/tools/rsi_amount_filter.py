# Filter the latest ClaudeRunner screener results by current daily RSI(14)
# and position cost.  Reads every sized CSV in the newest results folder that
# has Quantity/Total_Amount, computes Wilder RSI(14) on daily closes (same
# formula as the scanners), and keeps rows with
#   RSI_MIN <= RSI <= RSI_MAX  and  0 < Total_Amount < MAX_AMOUNT.
# Usage (env overrides): RESULTS_RUN=<run_id> RSI_MIN=58 RSI_MAX=64 MAX_AMOUNT=550
import glob, os, re
import pandas as pd
import yfinance as yf

RSI_MIN = float(os.environ.get("RSI_MIN", "58"))
RSI_MAX = float(os.environ.get("RSI_MAX", "64"))
MAX_AMOUNT = float(os.environ.get("MAX_AMOUNT", "550"))
ROOT = "claude_run/results"

run = os.environ.get("RESULTS_RUN") or max(
    (d for d in os.listdir(ROOT) if d.isdigit() and glob.glob(f"{ROOT}/{d}/daily_*/*.csv")),
    key=int)
print(f"Results run: {run}   filter: RSI {RSI_MIN}-{RSI_MAX}, Total_Amount < ${MAX_AMOUNT:,.0f}")


def calc_rsi(close, period=14):
    delta = close.diff()
    gain, loss = delta.clip(lower=0), -delta.clip(upper=0)
    ag = gain.ewm(alpha=1/period, min_periods=period, adjust=False).mean()
    al = loss.ewm(alpha=1/period, min_periods=period, adjust=False).mean()
    return 100 - 100 / (1 + ag / al)


frames = []
for c in sorted(glob.glob(f"{ROOT}/{run}/*/*.csv")):
    b = os.path.basename(c)
    if b.startswith("tv_"):
        continue
    df = pd.read_csv(c)
    if not {"Ticker", "Quantity", "Total_Amount"} <= set(df.columns):
        continue
    entry = "Saved_High" if "armed" in b else ("Entry_Price" if "Entry_Price" in df else "Price")
    keep = ["Ticker"] + (["Score"] if "Score" in df else []) + [entry, "Stop_Loss", "Quantity", "Total_Amount"]
    if "RSI" in df:
        keep.append("RSI")
    d = df[keep].rename(columns={entry: "Entry", "RSI": "Scanner_RSI"})
    d.insert(0, "Scanner", re.sub(r"_\d{8}_\d{4}\.csv$", "", b))
    frames.append(d)
    print(f"  {b}: {len(df)} rows")

allr = pd.concat(frames, ignore_index=True)
tickers = sorted(allr.Ticker.dropna().unique())
print(f"Downloading daily closes for {len(tickers)} tickers ...")
raw = yf.download(tickers, period="2y", interval="1d", auto_adjust=True,
                  actions=False, progress=False, group_by="ticker", threads=True)
rsi = {}
for t in tickers:
    try:
        cl = (raw[t]["Close"] if isinstance(raw.columns, pd.MultiIndex) else raw["Close"]).dropna()
        if len(cl) > 30:
            rsi[t] = float(calc_rsi(cl).iloc[-1])
    except Exception:
        pass
allr["RSI"] = allr.Ticker.map(rsi).round(1)
if "Scanner_RSI" in allr:
    chk = allr.dropna(subset=["Scanner_RSI", "RSI"])
    if len(chk):
        print(f"RSI check vs scanner's own RSI: max diff {(chk.RSI - chk.Scanner_RSI).abs().max():.2f} over {len(chk)} rows")
    allr = allr.drop(columns="Scanner_RSI")

hit = allr[(allr.RSI >= RSI_MIN) & (allr.RSI <= RSI_MAX)
           & (allr.Quantity > 0) & (allr.Total_Amount < MAX_AMOUNT)]
hit = hit.sort_values(["Ticker", "Scanner"])
agg = (hit.groupby("Ticker")
          .agg(Scanners=("Scanner", lambda s: ", ".join(sorted(set(s)))),
               N=("Scanner", "nunique"), RSI=("RSI", "first"))
          .sort_values(["N", "RSI"], ascending=[False, True]))
print(f"\n{len(hit)} rows, {hit.Ticker.nunique()} unique tickers\n")
pd.set_option("display.width", 250)
print(hit.to_string(index=False))
print("\nBy ticker:\n" + agg.to_string())
hit.to_csv(f"rsi_{int(RSI_MIN)}_{int(RSI_MAX)}_under_{int(MAX_AMOUNT)}.csv", index=False)
