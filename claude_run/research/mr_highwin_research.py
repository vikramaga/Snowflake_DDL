# High-win-rate research: long-only pullback (mean-reversion) rules with
# mechanical exits, searched for >= 80% winning trades AND positive average
# net return, trained on the first half of the period and tested on the
# second half. Run by ClaudeRunner (needs Yahoo data).
#
# Realistic execution (the scanner runs at 06:00 IST, after the US close):
#   signal on day k's close -> buy on day k+1 (at the open, or with a limit
#   order below the signal close) -> exit rule checked on each close -> sell
#   at the NEXT open. Stop (if any) is intraday: fill at min(open, stop).
#   Costs: 0.10% round trip. Win = net return > 0.
#
# Outputs: mr_candidates.csv (every rule passing train), mr_final.csv
# (top rules re-checked with one trade per ticker at a time),
# mr_trades_best.csv (trade list of the best rule).
import os, sys, time, random, itertools, warnings, runpy
import numpy as np, pandas as pd

warnings.filterwarnings("ignore")
SAMPLE = int(os.environ.get("MR_SAMPLE", "2000"))
YEARS = float(os.environ.get("MR_YEARS", "3.6"))
COST = 0.10            # % round trip
HMAX = 10
SEED = 7


def wilder_rsi(c, n):
    d = np.diff(c, prepend=np.nan)
    g = pd.Series(np.where(d > 0, d, 0.0)).ewm(alpha=1/n, adjust=False).mean().values
    l = pd.Series(np.where(d < 0, -d, 0.0)).ewm(alpha=1/n, adjust=False).mean().values
    with np.errstate(divide="ignore", invalid="ignore"):
        r = 100 - 100 / (1 + g / l)
    r[l == 0] = 100.0
    r[:n] = np.nan
    return r


def roll_mean(x, n):
    return pd.Series(x).rolling(n).mean().values


TRIGGERS = ["T_rsi2_lt10", "T_rsi2_lt5", "T_3down", "T_bb_lower", "T_ibs_lt15_below_sma5",
            "T_cumrsi2_lt20", "T_drop_3atr_5d"]
FILTERS = ["above_sma200", "sma50_gt_sma200", "sma200_rising", "above_sma50", "mkt_above_sma200",
           "mkt_above_sma50", "liquid_20m", "atr_lt3", "atr_ge3", "price_ge20", "ret5_gt_m10",
           "rsi2_lt3"]
ENTRIES = ["open", "limit05"]                    # next open / limit at close - 0.5 ATR10 (next day only)
EXITS = ["close_gt_sma5", "rsi2_gt70", "close_gt_prev_high", "first_up_close", "close_gt_sma10"]
HOLDS = [5, 10]
STOPS = [0.0, 2.5, 4.0]                          # ATR10 multiples below the fill (0 = no stop)
VARIANTS = list(itertools.product(ENTRIES, EXITS, HOLDS, STOPS))


def indicators(df):
    o, h, l, c, v = (df[x].values.astype(float) for x in ("Open", "High", "Low", "Close", "Volume"))
    ind = {"o": o, "h": h, "l": l, "c": c}
    ind["rsi2"] = wilder_rsi(c, 2)
    ind["sma5"], ind["sma10"] = roll_mean(c, 5), roll_mean(c, 10)
    ind["sma20"], ind["sma50"], ind["sma200"] = roll_mean(c, 20), roll_mean(c, 50), roll_mean(c, 200)
    sd20 = pd.Series(c).rolling(20).std().values
    pc = np.concatenate([[np.nan], c[:-1]])
    tr = np.nanmax(np.vstack([h - l, np.abs(h - pc), np.abs(l - pc)]), axis=0)
    ind["atr"] = roll_mean(tr, 10)
    ind["bbl"] = ind["sma20"] - 2 * sd20
    rng = h - l
    with np.errstate(divide="ignore", invalid="ignore"):
        ind["ibs"] = np.where(rng > 0, (c - l) / rng, 0.5)
    ind["vol20"] = roll_mean(v, 20)
    ind["dvol20"] = roll_mean(c * v, 20)
    ind["prev_high"] = np.concatenate([[np.nan], h[:-1]])
    return ind


def signal_rows(t, df, ind, mkt):
    c, n = ind["c"], len(ind["c"])
    k = np.arange(n)
    c1, c2, c3 = (np.concatenate([[np.nan] * s, c[:-s]]) for s in (1, 2, 3))
    c5 = np.concatenate([[np.nan] * 5, c[:-5]])
    hi5 = pd.Series(ind["h"]).rolling(5).max().values
    rsi2 = ind["rsi2"]
    rsi2p = np.concatenate([[np.nan], rsi2[:-1]])
    trig = {
        "T_rsi2_lt10": rsi2 < 10,
        "T_rsi2_lt5": rsi2 < 5,
        "T_3down": (c < c1) & (c1 < c2) & (c2 < c3),
        "T_bb_lower": c < ind["bbl"],
        "T_ibs_lt15_below_sma5": (ind["ibs"] < 0.15) & (c < ind["sma5"]),
        "T_cumrsi2_lt20": (rsi2 + rsi2p) < 20,
        "T_drop_3atr_5d": (hi5 - c) > 3 * ind["atr"],
    }
    anyt = np.zeros(n, bool)
    for v in trig.values():
        anyt |= np.nan_to_num(v, nan=0).astype(bool)
    ok = anyt & (k >= 210) & (k <= n - HMAX - 3) & (c >= 10) & (ind["vol20"] >= 100_000) \
        & np.isfinite(ind["atr"]) & np.isfinite(ind["sma200"])
    ks = np.where(ok)[0]
    if len(ks) == 0:
        return None
    sma200_20 = np.concatenate([[np.nan] * 20, ind["sma200"][:-20]])
    atrp = ind["atr"] / c * 100
    dates = df.index[ks]
    f = pd.DataFrame({"Ticker": t, "Date": dates, "k": ks})
    for name, v in trig.items():
        f[name] = v[ks]
    f["above_sma200"] = c[ks] > ind["sma200"][ks]
    f["sma50_gt_sma200"] = ind["sma50"][ks] > ind["sma200"][ks]
    f["sma200_rising"] = ind["sma200"][ks] > sma200_20[ks]
    f["above_sma50"] = c[ks] > ind["sma50"][ks]
    m = mkt.reindex(dates)
    f["mkt_above_sma200"] = m["mkt_above_sma200"].fillna(False).values.astype(bool)
    f["mkt_above_sma50"] = m["mkt_above_sma50"].fillna(False).values.astype(bool)
    f["liquid_20m"] = ind["dvol20"][ks] >= 20e6
    f["atr_lt3"] = atrp[ks] < 3
    f["atr_ge3"] = atrp[ks] >= 3
    f["price_ge20"] = c[ks] >= 20
    f["ret5_gt_m10"] = (c[ks] / c5[ks] - 1) > -0.10
    f["rsi2_lt3"] = rsi2[ks] < 3
    f["atr_pct"] = atrp[ks]
    f["rsi2"] = rsi2[ks]
    return f


def simulate(ind, ks):
    """Returns ret[N, V] (net %, NaN = limit not filled), xday[N, V] (exit bar index)."""
    o, h, l, c, atr = ind["o"], ind["h"], ind["l"], ind["c"], ind["atr"]
    N, V = len(ks), len(VARIANTS)
    ret = np.full((N, V), np.nan)
    xday = np.full((N, V), -1, dtype=int)
    f = ks + 1                                            # fill day
    J = np.arange(HMAX + 1)
    idx = f[:, None] + J[None, :]                         # (N, HMAX+1), all < n by construction
    O, Hh, L, C = o[idx], h[idx], l[idx], c[idx]
    conds = {
        "close_gt_sma5": C > ind["sma5"][idx],
        "rsi2_gt70": ind["rsi2"][idx] > 70,
        "close_gt_prev_high": C > ind["prev_high"][idx],
        "close_gt_sma10": C > ind["sma10"][idx],
    }
    for e_i, entry in enumerate(ENTRIES):
        if entry == "open":
            fill = o[f].copy(); filled = np.isfinite(fill)
        else:
            lim = c[ks] - 0.5 * atr[ks]
            filled = l[f] <= lim
            fill = np.minimum(o[f], lim)
        conds["first_up_close"] = C > fill[:, None]
        for v, (en, ex, hold, sm) in enumerate(VARIANTS):
            if en != entry:
                continue
            cond = conds[ex][:, :hold].copy()
            if sm > 0:
                stop = fill - sm * atr[ks]
                hit = L[:, :hold] <= stop[:, None]
            else:
                stop = np.full(N, -np.inf); hit = np.zeros((N, hold), bool)
            ev = hit | cond
            has = ev.any(axis=1)
            j = np.where(has, ev.argmax(axis=1), hold)
            rows = np.arange(N)
            jj = np.minimum(j, hold - 1)
            is_stop = has & hit[rows, jj]
            px = np.where(is_stop,
                          np.where(j == 0, stop, np.minimum(O[rows, jj], stop)),
                          O[rows, np.minimum(j + 1, HMAX)])   # exit next open (time exit: open of day `hold`)
            px = np.where(~has, O[rows, hold], px)
            r = (px / fill - 1) * 100 - COST
            ret[:, v] = np.where(filled, r, np.nan)
            xday[:, v] = np.where(is_stop, f + j, f + np.where(has, j + 1, hold))
    return ret, xday


def stats(r):
    r = r[np.isfinite(r)]
    n = len(r)
    if n == 0:
        return n, np.nan, np.nan, np.nan, np.nan
    w = r[r > 0].sum(); lo = -r[r <= 0].sum()
    return n, (r > 0).mean() * 100, r.mean(), (w / lo if lo > 0 else np.inf), r.min()


def main():
    t0 = time.time()
    base = runpy.run_path("claude_run/tools/backtest_winrate.py", run_name="lib")
    ns = base["load_scanner"]("daily_sweep_sma50_strategy_v1.py")
    tickers = ns["get_tickers"]()
    random.Random(SEED).shuffle(tickers)
    tickers = sorted(tickers[:SAMPLE])
    days = int(YEARS * 365)
    data = base["download"](tickers, days)
    spy = base["download"](["SPY"], days + 400)["SPY"]["Close"]
    mkt = pd.DataFrame({"mkt_above_sma200": spy > spy.rolling(200).mean(),
                        "mkt_above_sma50": spy > spy.rolling(50).mean()})
    print(f"tickers {len(data)} of {len(tickers)} | download {(time.time()-t0)/60:.1f} min", flush=True)

    feats, rets, xds = [], [], []
    for t, df in data.items():
        ind = indicators(df)
        fr = signal_rows(t, df, ind, mkt)
        if fr is None:
            continue
        r, x = simulate(ind, fr["k"].values)
        feats.append(fr); rets.append(r); xds.append(x)
    F = pd.concat(feats, ignore_index=True)
    R = np.vstack(rets); X = np.vstack(xds)
    F["Date"] = pd.to_datetime(F["Date"])
    d0, d1 = F["Date"].min(), F["Date"].max()
    cut = d0 + (d1 - d0) / 2
    tq = cut + (d1 - cut) / 2
    tr_m = (F["Date"] < cut).values
    te1 = ((F["Date"] >= cut) & (F["Date"] < tq)).values
    te2 = (F["Date"] >= tq).values
    print(f"signals {len(F)} | {d0.date()} .. {d1.date()} | train < {cut.date()} | sim {(time.time()-t0)/60:.1f} min",
          flush=True)

    # baseline per trigger (no filter)
    for tg in TRIGGERS:
        m = F[tg].values
        v = VARIANTS.index(("open", "close_gt_sma5", 10, 0.0))
        n, w, a, pf, mn = stats(R[m, v])
        print(f"  baseline {tg:24s} open/close>sma5/10d/nostop  n={n} win={w:.1f}% avg={a:+.2f}%")

    fcols = {k: F[k].values.astype(bool) for k in FILTERS}
    rows = []
    fcombos = [()] + [(a,) for a in FILTERS] + list(itertools.combinations(FILTERS, 2)) \
        + list(itertools.combinations(FILTERS, 3))
    for tg in TRIGGERS:
        tm = F[tg].values.astype(bool)
        Rt, trt, t1t, t2t = R[tm], tr_m[tm], te1[tm], te2[tm]
        fct = {x: fcols[x][tm] for x in FILTERS}
        for fc in fcombos:
            if "atr_lt3" in fc and "atr_ge3" in fc:
                continue
            m = np.ones(len(Rt), bool)
            for x in fc:
                m &= fct[x]
            if m.sum() < 400:
                continue
            Rtr, Rte1, Rte2 = Rt[m & trt], Rt[m & t1t], Rt[m & t2t]
            # quick vectorised train screen over all variants (70%+ kept so near misses are visible)
            ntr = np.isfinite(Rtr).sum(0)
            with np.errstate(invalid="ignore"):
                wtr = (Rtr > 0).sum(0) / np.maximum(ntr, 1) * 100
                atr_ = np.nanmean(Rtr, 0)
            for v in np.where((ntr >= 150) & (wtr >= 70) & (atr_ > 0.05))[0]:
                n1, w1, a1, p1, mn1 = stats(Rtr[:, v])
                n2, w2, a2, p2, mn2 = stats(Rte1[:, v])
                n3, w3, a3, p3, mn3 = stats(Rte2[:, v])
                nt, wt, at, pt, mnt = stats(np.concatenate([Rte1[:, v], Rte2[:, v]]))
                en, ex, hold, sm = VARIANTS[v]
                rows.append({"Trigger": tg, "Filters": " & ".join(fc) or "(none)", "Entry": en, "Exit": ex,
                             "MaxHold": hold, "StopATR": sm,
                             "Train_n": n1, "Train_WR": round(w1, 1), "Train_Avg%": round(a1, 3),
                             "Train_PF": round(p1, 2),
                             "Test_n": nt, "Test_WR": round(wt, 1), "Test_Avg%": round(at, 3),
                             "Test_PF": round(pt, 2), "Test_Worst%": round(mnt, 1),
                             "TestH1_WR": round(w2, 1), "TestH1_Avg%": round(a2, 3),
                             "TestH2_WR": round(w3, 1), "TestH2_Avg%": round(a3, 3), "_v": v})
    C = pd.DataFrame(rows)
    print(f"train-passing candidates: {len(C)} | search {(time.time()-t0)/60:.1f} min", flush=True)
    if C.empty:
        return
    C["Robust"] = (C.Train_WR >= 80) & (C["Train_Avg%"] > 0.15) \
        & (C.Test_n >= 150) & (C.Test_WR >= 80) & (C["Test_Avg%"] > 0) & (C.TestH1_WR >= 78) \
        & (C.TestH2_WR >= 78) & (C["TestH1_Avg%"] > 0) & (C["TestH2_Avg%"] > 0)
    C["Score"] = C[["Test_WR", "Train_WR"]].min(1) + 20 * C[["Test_Avg%", "Train_Avg%"]].min(1).clip(upper=1.5)
    C = C.sort_values(["Robust", "Score"], ascending=[False, False])
    C.drop(columns="_v").head(3000).to_csv("mr_candidates.csv", index=False)
    pd.set_option("display.width", 260); pd.set_option("display.max_colwidth", 60)
    show = ["Trigger", "Filters", "Entry", "Exit", "MaxHold", "StopATR", "Train_n", "Train_WR", "Train_Avg%",
            "Test_n", "Test_WR", "Test_Avg%", "Test_PF", "Test_Worst%", "TestH1_WR", "TestH2_WR", "Robust"]
    print(C[show].head(40).to_string(index=False))

    # re-check the top distinct robust rules with ONE trade per ticker at a time
    pool = C[C.Robust] if C.Robust.any() else C.head(300)        # near misses if nothing is robust
    top = pool.drop_duplicates(["Entry", "Exit", "MaxHold", "Train_n", "Train_WR", "Test_n", "Test_WR"]).head(40)
    fin = []
    keyorder = F.sort_values(["Ticker", "k"]).index.values
    best_trades = None
    for _, rw in top.iterrows():
        m = F[rw.Trigger].values.astype(bool)
        if rw.Filters != "(none)":
            for x in rw.Filters.split(" & "):
                m &= fcols[x]
        v = int(rw._v)
        taken = []
        busy = {}
        for i in keyorder:
            if not m[i] or not np.isfinite(R[i, v]):
                continue
            t, k = F.at[i, "Ticker"], F.at[i, "k"]
            if k <= busy.get(t, -1):
                continue
            busy[t] = X[i, v]
            taken.append(i)
        tk = np.array(taken)
        rr = R[tk, v]; dd = F.loc[tk, "Date"]
        out = dict(rw.drop(labels=["_v", "Score", "Robust"]))
        for lab, mm in [("NO_Train", (dd < cut).values), ("NO_Test", (dd >= cut).values), ("NO_All", np.ones(len(tk), bool))]:
            n, w, a, pf, mn = stats(rr[mm])
            out.update({f"{lab}_n": n, f"{lab}_WR": round(w, 1), f"{lab}_Avg%": round(a, 3),
                        f"{lab}_PF": round(pf, 2), f"{lab}_Worst%": round(mn, 1)})
        yrs = dd.dt.year.values
        out["Years_WR"] = " ".join(f"{y}:{(rr[yrs==y]>0).mean()*100:.0f}%/{len(rr[yrs==y])}" for y in sorted(set(yrs)))
        mo = dd.dt.to_period("M").values
        mwr = pd.Series(rr > 0).groupby(mo).mean()
        out["Months"] = len(mwr); out["Months_WR_lt70"] = int((mwr < 0.70).sum())
        out["AvgWin%"] = round(rr[rr > 0].mean(), 2); out["AvgLoss%"] = round(rr[rr <= 0].mean(), 2)
        out["Trades_per_month"] = round(len(tk) / max(1, len(mwr)), 1)
        fin.append(out)
        if best_trades is None:
            best_trades = F.loc[tk, ["Ticker", "Date", "atr_pct", "rsi2"]].assign(Ret_pct=rr, ExitBar=X[tk, v],
                                                                                  Rule=f"{rw.Trigger} | {rw.Filters} | {VARIANTS[v]}")
    FN = pd.DataFrame(fin)
    if len(FN):
        FN = FN.sort_values(["NO_Test_WR"], ascending=False)
        FN.to_csv("mr_final.csv", index=False)
        cols = ["Trigger", "Filters", "Entry", "Exit", "MaxHold", "StopATR", "NO_Train_n", "NO_Train_WR",
                "NO_Train_Avg%", "NO_Test_n", "NO_Test_WR", "NO_Test_Avg%", "NO_Test_PF", "NO_All_Worst%",
                "AvgWin%", "AvgLoss%", "Months_WR_lt70", "Trades_per_month", "Years_WR"]
        print("\nNON-OVERLAPPING re-check (one trade per ticker at a time):")
        print(FN[cols].head(25).to_string(index=False))
    if best_trades is not None:
        best_trades.to_csv("mr_trades_best.csv", index=False)
    print(f"done in {(time.time()-t0)/60:.1f} min")


main()
