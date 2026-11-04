# Independent check of the top rules from rr1_highwin_research.py on a FRESH
# random sample of stocks (seed 99, excluding every ticker the search used),
# same period, same execution engine (imported from the research script),
# one trade per ticker at a time. Nothing here was used to pick the rules.
# Output: rr_validate.csv (per rule: trades, win rate, payoff, expectancy,
# per-year win rate, worst trade) and rr_validate_trades.csv (best rule).
import os, re, time, random, runpy
import numpy as np, pandas as pd

SRC = "claude_run/research/rr1_highwin_research.py"
code = open(SRC).read()
code = re.sub(r"\nmain\(\)\s*$", "\n", code)            # load the engine without running the search
E = {"__name__": "rr1_engine"}
exec(compile(code, SRC, "exec"), E)

# (trigger, filters, entry index, stop ATR, target R, max hold) — from the search results
RULES = [
    ("T_rsi2_lt10", ["sma50_gt_sma200", "atr_ge3", "rs_pos_63d"], 3, 1.0, 1.0, 10),
    ("T_rsi2_lt10", ["sma200_rising", "atr_ge3", "rs_pos_63d"], 3, 1.0, 1.0, 10),
    ("T_rsi2_lt10", ["atr_ge3", "rs_pos_63d", "rsi14_lt40"], 3, 1.0, 1.0, 10),
    ("T_rsi2_lt10", ["atr_ge3", "price_ge20", "rs_pos_63d"], 3, 1.0, 1.0, 10),
    ("T_rsi2_lt5", ["sma200_rising", "price_ge20", "rs_pos_63d"], 3, 1.5, 1.0, 10),
    ("T_bb_lower", ["atr_ge3", "price_ge20", "rs_pos_63d"], 3, 1.0, 1.0, 10),
    ("T_bb_lower", ["above_sma200", "atr_ge3", "price_ge20"], 3, 2.0, 1.0, 10),
    ("T_ibs_lt15_below_sma5", ["sma50_gt_sma200", "rs_pos_63d", "rsi14_lt40"], 3, 3.0, 1.0, 10),
    ("T_drop_3atr_5d", ["price_ge20", "rs_pos_63d", "rsi14_lt40"], 2, 1.0, 1.0, 20),
]
SAMPLE = int(os.environ.get("RV_SAMPLE", "2000"))


def main():
    t0 = time.time()
    base = runpy.run_path("claude_run/tools/backtest_winrate.py", run_name="lib")
    ns = base["load_scanner"]("daily_sweep_sma50_strategy_v1.py")
    tickers = ns["get_tickers"]()
    used = list(tickers); random.Random(E["SEED"]).shuffle(used)
    used = set(used[:E["SAMPLE"]])
    fresh = [t for t in tickers if t not in used]
    random.Random(99).shuffle(fresh)
    fresh = sorted(fresh[:SAMPLE])
    days = int(E["YEARS"] * 365)
    data = base["download"](fresh, days)
    spy = base["download"](["SPY"], days + 400)["SPY"]["Close"]
    mkt = pd.DataFrame({"mkt_above_sma200": spy > spy.rolling(200).mean(),
                        "mkt_above_sma50": spy > spy.rolling(50).mean(),
                        "spy_ret63": spy / spy.shift(63) - 1})
    print(f"fresh tickers {len(data)} of {len(fresh)} (none used by the search) | {(time.time()-t0)/60:.1f} min", flush=True)

    feats, rets, risks, xds = [], [], [], []
    for t, df in data.items():
        ind = E["indicators"](df)
        fr = E["signal_rows"](t, df, ind, mkt)
        if fr is None:
            continue
        r, rk, x = E["simulate"](ind, fr["k"].values)
        feats.append(fr); rets.append(r); risks.append(rk); xds.append(x)
    F = pd.concat(feats, ignore_index=True)
    R = np.vstack(rets); RK = np.vstack(risks); X = np.vstack(xds)
    F["Date"] = pd.to_datetime(F["Date"])
    print(f"signals {len(F)} | {F.Date.min().date()} .. {F.Date.max().date()}", flush=True)

    keyorder = F.sort_values(["Ticker", "k"]).index.values
    out, best = [], None
    for tg, flt, e, s, rr, hold in RULES:
        v = E["VARIANTS"].index((e, s, rr, hold))
        m = F[tg].values.astype(bool)
        for x in flt:
            m &= F[x].values.astype(bool)
        taken, busy = [], {}
        for i in keyorder:
            if not m[i] or not np.isfinite(R[i, v]):
                continue
            t, k = F.at[i, "Ticker"], F.at[i, "k"]
            if k <= busy.get(t, -1):
                continue
            busy[t] = X[i, v]
            taken.append(i)
        tk = np.array(taken, dtype=int)
        r_, rk_ = R[tk, v].astype(float), RK[tk, v].astype(float)
        st = E["stats"](r_, rk_)
        dd = F.loc[tk, "Date"]
        yrs = dd.dt.year.values
        mo = dd.dt.to_period("M").values
        mwr = pd.Series(r_ > 0).groupby(mo).mean() if len(tk) else pd.Series(dtype=float)
        h = len(tk) // 2
        row = {"Trigger": tg, "Filters": " & ".join(flt), "Entry": E["entry_name"](e), "StopATR": s,
               "TargetR": rr, "MaxHold": hold, "n": st["n"], "WR": round(st["wr"], 1),
               "Payoff": round(st["payoff"], 2), "ExpR": round(st["expR"], 3), "Avg%": round(st["avg"], 3),
               "AvgWin%": round(st["aw"], 2), "AvgLoss%": round(st["al"], 2), "Worst%": round(st["worst"], 1),
               "WR_first_half": round((r_[:h] > 0).mean() * 100, 1) if h else np.nan,
               "WR_second_half": round((r_[h:] > 0).mean() * 100, 1) if h else np.nan,
               "Months": len(mwr), "Months_WR_lt50": int((mwr < 0.5).sum()),
               "Trades_per_month": round(len(tk) / max(1, len(mwr)), 1),
               "Years_WR": " ".join(f"{y}:{(r_[yrs==y]>0).mean()*100:.0f}%/{(yrs==y).sum()}" for y in sorted(set(yrs)))}
        out.append(row)
        print(f"{tg:22s} {' & '.join(flt):45s} stop {s}ATR {rr}R {hold}d | n={row['n']} WR={row['WR']}% "
              f"payoff={row['Payoff']} expR={row['ExpR']:+.3f} worst={row['Worst%']}% | {row['Years_WR']}", flush=True)
        if best is None:
            best = F.loc[tk, ["Ticker", "Date", "atr_pct"]].assign(Ret_pct=r_, Risk_pct=rk_, R_mult=r_ / rk_)
    pd.DataFrame(out).to_csv("rr_validate.csv", index=False)
    if best is not None:
        best.to_csv("rr_validate_trades.csv", index=False)
    print(f"done in {(time.time()-t0)/60:.1f} min")


main()
