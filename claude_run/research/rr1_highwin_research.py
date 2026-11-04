# Highest win rate at >= 1:1 reward/risk.
# Long-only rules: a trigger (pullback OR trend/momentum) + up to 3 filters,
# an entry (next open, or a limit below the signal close), and a BRACKET
# exit: stop = fill - s*ATR10, target = fill + R*s*ATR10 with R >= 1, plus a
# time exit. Rules are picked on the first half of the period and checked on
# the second half (both quarters), then re-checked with one trade per ticker
# at a time. Run by ClaudeRunner (needs Yahoo data).
#
# Execution (the scanner runs at 06:00 IST, after the US close):
#   signal on day k's close -> order for day k+1 -> bracket rests from the
#   fill. Conservative intraday order: a gap through the stop/target fills at
#   the open; a day touching both counts as the STOP; on the fill day the
#   target counts only if the close is at/above it. Time exit = next open
#   after MaxHold sessions. Costs: 0.10% round trip. Win = net return > 0.
#
# "1:1" is checked on what actually happened, not just the order sizes:
#   Payoff = average win / average loss (both net %). A rule only counts if
#   Payoff >= 1.0 on train AND test. Expectancy is reported in R
#   (net % / risk %, risk = s*ATR10 / fill).
#
# Outputs: rr_candidates.csv, rr_final.csv (non-overlapping re-check),
# rr_frontier.csv (best robust win rate per stop/R), rr_trades_best.csv.
import os, time, random, itertools, warnings, runpy
import numpy as np, pandas as pd

warnings.filterwarnings("ignore")
SAMPLE = int(os.environ.get("RR_SAMPLE", "2000"))
YEARS = float(os.environ.get("RR_YEARS", "3.6"))
COST = 0.10            # % round trip
HMAX = 20
SEED = 11              # different random sample from the 80% research (seed 7)


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


def shift(x, s):
    return np.concatenate([[np.nan] * s, x[:-s]])


TRIGGERS = [
    # pullback / mean reversion
    "T_rsi2_lt10", "T_rsi2_lt5", "T_3down", "T_bb_lower", "T_ibs_lt15_below_sma5", "T_drop_3atr_5d",
    # trend / momentum
    "T_breakout20", "T_sma20_touch_hold", "T_rsi14_cross50", "T_ema8_reclaim", "T_nr7_inside",
    "T_higher_low_above_sma50",
]
FILTERS = ["above_sma200", "sma50_gt_sma200", "sma200_rising", "above_sma50", "mkt_above_sma200",
           "mkt_above_sma50", "liquid_20m", "atr_lt3", "atr_ge3", "price_ge20", "rs_pos_63d",
           "vol_surge", "rsi14_lt40", "rsi14_gt60"]
ENTRIES = [("open", 0.0), ("lim", 0.25), ("lim", 0.5), ("lim", 1.0)]   # limit at close - x*ATR10, next day only
STOPS = [0.5, 1.0, 1.5, 2.0, 3.0]                                      # ATR10 multiples
RRS = [1.0, 1.5]                                                       # target = RR x stop distance
HOLDS = [5, 10, 20]
VARIANTS = list(itertools.product(range(len(ENTRIES)), STOPS, RRS, HOLDS))


def entry_name(e):
    kind, x = ENTRIES[e]
    return "next_open" if kind == "open" else f"limit_close-{x}ATR"


def indicators(df):
    o, h, l, c, v = (df[x].values.astype(float) for x in ("Open", "High", "Low", "Close", "Volume"))
    ind = {"o": o, "h": h, "l": l, "c": c, "v": v}
    ind["rsi2"], ind["rsi14"] = wilder_rsi(c, 2), wilder_rsi(c, 14)
    ind["sma5"], ind["sma20"] = roll_mean(c, 5), roll_mean(c, 20)
    ind["sma50"], ind["sma200"] = roll_mean(c, 50), roll_mean(c, 200)
    ind["ema8"] = pd.Series(c).ewm(span=8, adjust=False).mean().values
    sd20 = pd.Series(c).rolling(20).std().values
    pc = shift(c, 1)
    tr = np.nanmax(np.vstack([h - l, np.abs(h - pc), np.abs(l - pc)]), axis=0)
    ind["atr"] = roll_mean(tr, 10)
    ind["bbl"] = ind["sma20"] - 2 * sd20
    rng = h - l
    with np.errstate(divide="ignore", invalid="ignore"):
        ind["ibs"] = np.where(rng > 0, (c - l) / rng, 0.5)
    ind["vol20"] = roll_mean(v, 20)
    ind["dvol20"] = roll_mean(c * v, 20)
    return ind


def signal_rows(t, df, ind, mkt):
    o, h, l, c, v = ind["o"], ind["h"], ind["l"], ind["c"], ind["v"]
    n = len(c); k = np.arange(n)
    c1, c2, c3 = shift(c, 1), shift(c, 2), shift(c, 3)
    rsi2, rsi14 = ind["rsi2"], ind["rsi14"]
    hi5 = pd.Series(h).rolling(5).max().values
    hi20p = shift(pd.Series(h).rolling(20).max().values, 1)
    rng = h - l
    nr7 = rng <= pd.Series(rng).rolling(7).min().values
    inside = (h < shift(h, 1)) & (l > shift(l, 1))
    sma50, sma200 = ind["sma50"], ind["sma200"]
    sma50_20 = shift(sma50, 20)
    l1, l2 = shift(l, 1), shift(l, 2)
    trig = {
        "T_rsi2_lt10": rsi2 < 10,
        "T_rsi2_lt5": rsi2 < 5,
        "T_3down": (c < c1) & (c1 < c2) & (c2 < c3),
        "T_bb_lower": c < ind["bbl"],
        "T_ibs_lt15_below_sma5": (ind["ibs"] < 0.15) & (c < ind["sma5"]),
        "T_drop_3atr_5d": (hi5 - c) > 3 * ind["atr"],
        "T_breakout20": c > hi20p,
        "T_sma20_touch_hold": (l <= ind["sma20"]) & (c > ind["sma20"]) & (ind["sma20"] > shift(ind["sma20"], 5)),
        "T_rsi14_cross50": (rsi14 > 50) & (shift(rsi14, 1) <= 50),
        "T_ema8_reclaim": (c > ind["ema8"]) & (c1 <= shift(ind["ema8"], 1)) & (sma50 > sma50_20),
        "T_nr7_inside": nr7 & inside,
        "T_higher_low_above_sma50": (l > l1) & (l1 > l2) & (c > sma50) & (c > c1),
    }
    anyt = np.zeros(n, bool)
    for x in trig.values():
        anyt |= np.nan_to_num(x, nan=0).astype(bool)
    ok = anyt & (k >= 210) & (k <= n - HMAX - 3) & (c >= 5) & (ind["vol20"] >= 100_000) \
        & np.isfinite(ind["atr"]) & (ind["atr"] > 0) & np.isfinite(sma200)
    ks = np.where(ok)[0]
    if len(ks) == 0:
        return None
    sma200_20 = shift(sma200, 20)
    atrp = ind["atr"] / c * 100
    dates = df.index[ks]
    f = pd.DataFrame({"Ticker": t, "Date": dates, "k": ks})
    for name, x in trig.items():
        f[name] = np.nan_to_num(x[ks], nan=0).astype(bool)
    f["above_sma200"] = c[ks] > sma200[ks]
    f["sma50_gt_sma200"] = sma50[ks] > sma200[ks]
    f["sma200_rising"] = sma200[ks] > sma200_20[ks]
    f["above_sma50"] = c[ks] > sma50[ks]
    m = mkt.reindex(dates)
    f["mkt_above_sma200"] = m["mkt_above_sma200"].fillna(False).values.astype(bool)
    f["mkt_above_sma50"] = m["mkt_above_sma50"].fillna(False).values.astype(bool)
    ret63 = c / shift(c, 63) - 1
    f["rs_pos_63d"] = (ret63[ks] - m["spy_ret63"].values) > 0
    f["liquid_20m"] = ind["dvol20"][ks] >= 20e6
    f["atr_lt3"] = atrp[ks] < 3
    f["atr_ge3"] = atrp[ks] >= 3
    f["price_ge20"] = c[ks] >= 20
    f["vol_surge"] = v[ks] >= 1.5 * ind["vol20"][ks]
    f["rsi14_lt40"] = rsi14[ks] < 40
    f["rsi14_gt60"] = rsi14[ks] > 60
    f["atr_pct"] = atrp[ks]
    return f


def simulate(ind, ks):
    """ret[N,V] net %, riskp[N,V] risk % of fill (NaN = limit not filled), xday[N,V] exit bar."""
    o, h, l, c, atr = ind["o"], ind["h"], ind["l"], ind["c"], ind["atr"]
    N, V = len(ks), len(VARIANTS)
    ret = np.full((N, V), np.nan, dtype=np.float32)
    riskp = np.full((N, V), np.nan, dtype=np.float32)
    xday = np.full((N, V), -1, dtype=np.int32)
    f = ks + 1
    J = np.arange(HMAX + 1)
    idx = f[:, None] + J[None, :]
    O, Hh, L, C = o[idx], h[idx], l[idx], c[idx]
    a = atr[ks]
    rows = np.arange(N)
    for e, (kind, x) in enumerate(ENTRIES):
        if kind == "open":
            fill = O[:, 0].copy(); filled = np.isfinite(fill) & (fill > 0)
        else:
            lim = c[ks] - x * a
            filled = (L[:, 0] <= lim) & (lim > 0)
            fill = np.minimum(O[:, 0], lim)
        for v, (ev_, s, rr, hold) in enumerate(VARIANTS):
            if ev_ != e:
                continue
            stop = fill - s * a
            tgt = fill + rr * s * a
            Oh, Lh, Hh_, Ch = O[:, :hold], L[:, :hold], Hh[:, :hold], C[:, :hold]
            gs = Oh <= stop[:, None]; gt = Oh >= tgt[:, None]
            sh = Lh <= stop[:, None]; th = Hh_ >= tgt[:, None]
            gs[:, 0] = False; gt[:, 0] = False             # day 0 open is the fill itself
            th[:, 0] = Ch[:, 0] >= tgt                      # high may have come before the fill
            evm = gs | gt | sh | th
            has = evm.any(axis=1)
            j = np.where(has, evm.argmax(axis=1), hold)
            jj = np.minimum(j, hold - 1)
            px = np.where(gs[rows, jj], Oh[rows, jj],
                 np.where(gt[rows, jj], Oh[rows, jj],
                 np.where(sh[rows, jj], stop, tgt)))
            px = np.where(has, px, O[rows, hold])
            r = (px / fill - 1) * 100 - COST
            ok = filled & (stop > 0)
            ret[:, v] = np.where(ok, r, np.nan)
            riskp[:, v] = np.where(ok, s * a / fill * 100, np.nan)
            xday[:, v] = f + np.where(has, j, hold)
    return ret, riskp, xday


def stats(r, rk):
    m = np.isfinite(r)
    r, rk = r[m].astype(float), rk[m].astype(float)
    n = len(r)
    if n == 0:
        return dict(n=0, wr=np.nan, avg=np.nan, payoff=np.nan, expR=np.nan, worst=np.nan, aw=np.nan, al=np.nan)
    wins, losses = r[r > 0], r[r <= 0]
    aw = wins.mean() if len(wins) else 0.0
    al = -losses.mean() if len(losses) else np.nan
    return dict(n=n, wr=(r > 0).mean() * 100, avg=r.mean(),
                payoff=(aw / al) if (al and al > 0) else np.inf,
                expR=(r / rk).mean(), worst=r.min(), aw=aw, al=-al if np.isfinite(al) else np.nan)


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
                        "mkt_above_sma50": spy > spy.rolling(50).mean(),
                        "spy_ret63": spy / spy.shift(63) - 1})
    print(f"tickers {len(data)} of {len(tickers)} | download {(time.time()-t0)/60:.1f} min", flush=True)

    feats, rets, risks, xds = [], [], [], []
    for t, df in data.items():
        ind = indicators(df)
        fr = signal_rows(t, df, ind, mkt)
        if fr is None:
            continue
        r, rk, x = simulate(ind, fr["k"].values)
        feats.append(fr); rets.append(r); risks.append(rk); xds.append(x)
    F = pd.concat(feats, ignore_index=True)
    R = np.vstack(rets); RK = np.vstack(risks); X = np.vstack(xds)
    del rets, risks, xds
    F["Date"] = pd.to_datetime(F["Date"])
    d0, d1 = F["Date"].min(), F["Date"].max()
    cut = d0 + (d1 - d0) / 2
    tq = cut + (d1 - cut) / 2
    tr_m = (F["Date"] < cut).values
    te1 = ((F["Date"] >= cut) & (F["Date"] < tq)).values
    te2 = (F["Date"] >= tq).values
    print(f"signals {len(F)} | variants {len(VARIANTS)} | {d0.date()} .. {d1.date()} | train < {cut.date()} | "
          f"sim {(time.time()-t0)/60:.1f} min", flush=True)

    # baseline per trigger: next open, 1 ATR stop, 1R target, 10 days
    vb = VARIANTS.index((0, 1.0, 1.0, 10))
    for tg in TRIGGERS:
        m = F[tg].values
        s = stats(R[m, vb], RK[m, vb])
        print(f"  baseline {tg:26s} open / 1ATR stop / 1R / 10d  n={s['n']} win={s['wr']:.1f}% "
              f"payoff={s['payoff']:.2f} expR={s['expR']:+.3f}")

    fcols = {k: F[k].values.astype(bool) for k in FILTERS}
    rows = []
    fcombos = [()] + [(a,) for a in FILTERS] + list(itertools.combinations(FILTERS, 2)) \
        + list(itertools.combinations(FILTERS, 3))
    bad_pairs = [("atr_lt3", "atr_ge3"), ("rsi14_lt40", "rsi14_gt60")]
    for tg in TRIGGERS:
        tm = F[tg].values.astype(bool)
        Rt, RKt, trt, t1t, t2t = R[tm], RK[tm], tr_m[tm], te1[tm], te2[tm]
        fct = {x: fcols[x][tm] for x in FILTERS}
        for fc in fcombos:
            if any(a in fc and b in fc for a, b in bad_pairs):
                continue
            m = np.ones(len(Rt), bool)
            for x in fc:
                m &= fct[x]
            if m.sum() < 400:
                continue
            mtr = m & trt
            Rtr = Rt[mtr]
            ntr = np.isfinite(Rtr).sum(0)
            with np.errstate(invalid="ignore", divide="ignore"):
                wtr = (Rtr > 0).sum(0) / np.maximum(ntr, 1) * 100
                atr_ = np.nanmean(Rtr, 0)
            cand = np.where((ntr >= 150) & (wtr >= 55) & (atr_ > 0))[0]
            if len(cand) == 0:
                continue
            RKtr = RKt[mtr]
            m1, m2 = m & t1t, m & t2t
            R1, RK1, R2, RK2 = Rt[m1], RKt[m1], Rt[m2], RKt[m2]
            for v in cand:
                s1 = stats(Rtr[:, v], RKtr[:, v])
                if not (s1["payoff"] >= 1.0):
                    continue
                s2 = stats(R1[:, v], RK1[:, v])
                s3 = stats(R2[:, v], RK2[:, v])
                st = stats(np.concatenate([R1[:, v], R2[:, v]]), np.concatenate([RK1[:, v], RK2[:, v]]))
                e, s, rr, hold = VARIANTS[v]
                rows.append({"Trigger": tg, "Filters": " & ".join(fc) or "(none)", "Entry": entry_name(e),
                             "StopATR": s, "TargetR": rr, "MaxHold": hold,
                             "Train_n": s1["n"], "Train_WR": round(s1["wr"], 1), "Train_Payoff": round(s1["payoff"], 2),
                             "Train_ExpR": round(s1["expR"], 3),
                             "Test_n": st["n"], "Test_WR": round(st["wr"], 1), "Test_Payoff": round(st["payoff"], 2),
                             "Test_ExpR": round(st["expR"], 3), "Test_Avg%": round(st["avg"], 3),
                             "Test_Worst%": round(st["worst"], 1),
                             "TestH1_WR": round(s2["wr"], 1), "TestH1_ExpR": round(s2["expR"], 3),
                             "TestH2_WR": round(s3["wr"], 1), "TestH2_ExpR": round(s3["expR"], 3), "_v": v})
    C = pd.DataFrame(rows)
    print(f"train-passing candidates (WR>=55, payoff>=1): {len(C)} | search {(time.time()-t0)/60:.1f} min", flush=True)
    if C.empty:
        return
    C["Robust"] = (C.Test_n >= 150) & (C.Test_Payoff >= 1.0) & (C.Test_ExpR > 0) \
        & (C.TestH1_ExpR > 0) & (C.TestH2_ExpR > 0) & (C.Train_ExpR > 0)
    C["MinWR"] = C[["Train_WR", "Test_WR", "TestH1_WR", "TestH2_WR"]].min(1)
    C = C.sort_values(["Robust", "MinWR", "Test_ExpR"], ascending=[False, False, False])
    C.drop(columns="_v").head(5000).to_csv("rr_candidates.csv", index=False)
    pd.set_option("display.width", 280); pd.set_option("display.max_colwidth", 55)
    show = ["Trigger", "Filters", "Entry", "StopATR", "TargetR", "MaxHold", "Train_n", "Train_WR",
            "Train_Payoff", "Test_n", "Test_WR", "Test_Payoff", "Test_ExpR", "TestH1_WR", "TestH2_WR", "MinWR"]
    print("\nTOP ROBUST CANDIDATES (sorted by worst of train / test / test halves win rate):")
    print(C[C.Robust][show].head(40).to_string(index=False))

    # frontier: best robust MinWR per stop size and target R (the honest ceiling)
    fr = C[C.Robust].groupby(["TargetR", "StopATR"]).head(1)[
        ["TargetR", "StopATR", "Trigger", "Filters", "Entry", "MaxHold", "Train_WR", "Test_WR", "MinWR",
         "Test_Payoff", "Test_ExpR", "Test_n"]].sort_values(["TargetR", "StopATR"])
    fr.to_csv("rr_frontier.csv", index=False)
    print("\nFRONTIER (best robust rule per stop size / target):")
    print(fr.to_string(index=False))

    # non-overlapping re-check: one trade per ticker at a time
    pool = C[C.Robust]
    top = pool.drop_duplicates(["Entry", "StopATR", "TargetR", "MaxHold", "Train_n", "Train_WR", "Test_n"]).head(40)
    fin, best_trades = [], None
    keyorder = F.sort_values(["Ticker", "k"]).index.values
    for _, rw in top.iterrows():
        m = F[rw.Trigger].values.astype(bool)
        if rw.Filters != "(none)":
            for x in rw.Filters.split(" & "):
                m &= fcols[x]
        v = int(rw._v)
        taken, busy = [], {}
        for i in keyorder:
            if not m[i] or not np.isfinite(R[i, v]):
                continue
            t, k = F.at[i, "Ticker"], F.at[i, "k"]
            if k <= busy.get(t, -1):
                continue
            busy[t] = X[i, v]
            taken.append(i)
        tk = np.array(taken)
        rr_, rk_ = R[tk, v].astype(float), RK[tk, v].astype(float)
        dd = F.loc[tk, "Date"]
        out = dict(rw.drop(labels=["_v", "Robust"]))
        for lab, mm in [("NO_Train", (dd < cut).values), ("NO_Test", (dd >= cut).values),
                        ("NO_All", np.ones(len(tk), bool))]:
            s = stats(rr_[mm], rk_[mm])
            out.update({f"{lab}_n": s["n"], f"{lab}_WR": round(s["wr"], 1), f"{lab}_Payoff": round(s["payoff"], 2),
                        f"{lab}_ExpR": round(s["expR"], 3), f"{lab}_Worst%": round(s["worst"], 1)})
        yrs = dd.dt.year.values
        out["Years_WR"] = " ".join(f"{y}:{(rr_[yrs==y]>0).mean()*100:.0f}%/{(yrs==y).sum()}" for y in sorted(set(yrs)))
        mo = dd.dt.to_period("M").values
        mwr = pd.Series(rr_ > 0).groupby(mo).mean()
        out["Months"] = len(mwr); out["Months_WR_lt50"] = int((mwr < 0.50).sum())
        out["AvgWin%"] = round(rr_[rr_ > 0].mean(), 2); out["AvgLoss%"] = round(rr_[rr_ <= 0].mean(), 2)
        out["Trades_per_month"] = round(len(tk) / max(1, len(mwr)), 1)
        fin.append(out)
        if best_trades is None:
            best_trades = F.loc[tk, ["Ticker", "Date", "atr_pct"]].assign(
                Ret_pct=rr_, Risk_pct=rk_, R_mult=rr_ / rk_, ExitBar=X[tk, v],
                Rule=f"{rw.Trigger} | {rw.Filters} | {rw.Entry} | stop {rw.StopATR}ATR | {rw.TargetR}R | {rw.MaxHold}d")
    FN = pd.DataFrame(fin)
    if len(FN):
        FN["NO_MinWR"] = FN[["NO_Train_WR", "NO_Test_WR"]].min(1)
        FN = FN.sort_values(["NO_MinWR", "NO_Test_ExpR"], ascending=False)
        FN.to_csv("rr_final.csv", index=False)
        cols = ["Trigger", "Filters", "Entry", "StopATR", "TargetR", "MaxHold", "NO_Train_n", "NO_Train_WR",
                "NO_Train_Payoff", "NO_Test_n", "NO_Test_WR", "NO_Test_Payoff", "NO_Test_ExpR", "NO_All_Worst%",
                "AvgWin%", "AvgLoss%", "Months_WR_lt50", "Trades_per_month", "Years_WR"]
        print("\nNON-OVERLAPPING re-check (one trade per ticker at a time):")
        print(FN[cols].head(25).to_string(index=False))
    if best_trades is not None:
        best_trades.to_csv("rr_trades_best.csv", index=False)
    print(f"done in {(time.time()-t0)/60:.1f} min")


main()
