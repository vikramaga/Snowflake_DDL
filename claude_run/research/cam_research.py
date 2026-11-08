# Best bullish swing strategy from Camarilla levels (monthly + weekly H3, H4,
# S3) and SMA50 / SMA150. Run by ClaudeRunner (needs Yahoo data).
#
# Levels (classic Camarilla, from the PREVIOUS completed period):
#   range = prevHigh - prevLow
#   H3 = prevClose + 1.1*range/4   H4 = prevClose + 1.1*range/2
#   S3 = prevClose - 1.1*range/4
# Weekly levels = previous week, monthly levels = previous calendar month.
#
# Search: 12 triggers (H4 breakouts / retests, H3 reclaims, S3 sweeps and
# reclaims, S3 + SMA50 confluence) x up to 3 of 16 filters (SMA50/SMA150
# trend, monthly level position, room to H4, narrow week, market) x exits
# (ATR bracket: stop 1-3 ATR, target 1-3R, 10/20/40 days; or swing exit: ATR
# stop and sell when a close is below SMA50, max 60 days).
# Signal on day k's close -> buy next open. Conservative fills (gap through a
# level fills at the open, a day touching stop and target = stop), 0.10%
# costs. Expectancy in R = net % / initial risk %.
#
# Picked on sample A (first half of the period), must hold on both quarters of
# the second half, then the top rules are re-tested ONE TRADE PER TICKER AT A
# TIME on sample A and on a fresh sample B of different stocks.
# Outputs: cam_candidates.csv, cam_final.csv, cam_trades_best.csv
import os, time, random, itertools, warnings, runpy
import numpy as np, pandas as pd

warnings.filterwarnings("ignore")
SAMPLE = int(os.environ.get("CAM_SAMPLE", "1800"))
YEARS = float(os.environ.get("CAM_YEARS", "4.0"))
COST = 0.10
HMAX = 60
SEED_A, SEED_B = 21, 99

TRIGGERS = ["T_wH4_break", "T_wH4_retest", "T_wH3_reclaim", "T_wS3_sweep", "T_wS3_reclaim",
            "T_mH4_break", "T_mH4_retest", "T_mH3_reclaim", "T_mS3_sweep", "T_mS3_reclaim",
            "T_wS3_sma50_confluence", "T_wH3_hold"]
FILTERS = ["above_sma50", "above_sma150", "sma50_gt_sma150", "sma50_rising", "sma150_rising",
           "near_sma50", "not_extended", "above_mH3", "above_mH4", "above_mS3", "below_wH4",
           "w_narrow", "m_narrow", "mkt_above_sma50", "vol_surge", "wH3_above_mH3"]
BAD = [("above_mH4", "below_wH4")]
BRACKETS = [(s, r, h) for s in (1.0, 1.5, 2.0, 3.0) for r in (1.0, 1.5, 2.0, 3.0) for h in (10, 20, 40)]
TRAILS = [(s, 0, 60) for s in (1.5, 2.0, 3.0)]
VARIANTS = [("bracket",) + b for b in BRACKETS] + [("sma50_exit",) + t for t in TRAILS]


def vname(v):
    kind, s, r, h = VARIANTS[v]
    return f"stop {s}ATR, target {r}R, max {h}d" if kind == "bracket" else f"stop {s}ATR, sell on close < SMA50, max {h}d"


def shift(x, s):
    return np.concatenate([[np.nan] * s, x[:-s]])


def roll_mean(x, n):
    return pd.Series(x).rolling(n).mean().values


def camarilla(df, rule):
    per = df.index.to_period(rule)
    g = df.groupby(per).agg(H=("High", "max"), L=("Low", "min"), C=("Close", "last"))
    p = g.shift(1)
    rng = p.H - p.L
    width = rng / p.C
    lv = pd.DataFrame({"H3": p.C + rng * 1.1 / 4, "H4": p.C + rng * 1.1 / 2, "S3": p.C - rng * 1.1 / 4,
                       "narrow": width < width.rolling(20, min_periods=10).median().shift(1)}, index=g.index)
    lv = lv.reindex(per)
    return {k: lv[k].values for k in lv.columns}


def indicators(df):
    o, h, l, c, v = (df[x].values.astype(float) for x in ("Open", "High", "Low", "Close", "Volume"))
    ind = {"o": o, "h": h, "l": l, "c": c, "v": v}
    ind["sma50"], ind["sma150"] = roll_mean(c, 50), roll_mean(c, 150)
    pc = shift(c, 1)
    tr = np.nanmax(np.vstack([h - l, np.abs(h - pc), np.abs(l - pc)]), axis=0)
    ind["atr"] = roll_mean(tr, 14)
    ind["vol20"] = roll_mean(v, 20)
    w, m = camarilla(df, "W-FRI"), camarilla(df, "M")
    for k in ("H3", "H4", "S3"):
        ind["w" + k], ind["m" + k] = w[k].astype(float), m[k].astype(float)
    ind["w_narrow"] = np.asarray(w["narrow"], dtype=float) > 0.5
    ind["m_narrow"] = np.asarray(m["narrow"], dtype=float) > 0.5
    return ind


def signal_rows(t, df, ind, mkt):
    o, h, l, c, v, a = ind["o"], ind["h"], ind["l"], ind["c"], ind["v"], ind["atr"]
    n = len(c); k = np.arange(n)
    c1 = shift(c, 1)
    s50, s150 = ind["sma50"], ind["sma150"]
    wH3, wH4, wS3, mH3, mH4, mS3 = (ind[x] for x in ("wH3", "wH4", "wS3", "mH3", "mH4", "mS3"))
    with np.errstate(invalid="ignore"):
        above_w4_5 = pd.Series((c > wH4).astype(float)).shift(1).rolling(5).max().values > 0
        above_m4_10 = pd.Series((c > mH4).astype(float)).shift(1).rolling(10).max().values > 0
        trig = {
            "T_wH4_break": (c > wH4) & (c1 <= wH4),
            "T_wH4_retest": above_w4_5 & (l <= wH4 + 0.25 * a) & (c > wH4) & (c1 > wH4 - 0.25 * a),
            "T_wH3_reclaim": (c > wH3) & (c1 <= wH3) & (c < wH4),
            "T_wS3_sweep": (l < wS3) & (c > wS3),
            "T_wS3_reclaim": (c > wS3) & (c1 <= wS3),
            "T_mH4_break": (c > mH4) & (c1 <= mH4),
            "T_mH4_retest": above_m4_10 & (l <= mH4 + 0.25 * a) & (c > mH4),
            "T_mH3_reclaim": (c > mH3) & (c1 <= mH3) & (c < mH4),
            "T_mS3_sweep": (l < mS3) & (c > mS3),
            "T_mS3_reclaim": (c > mS3) & (c1 <= mS3),
            "T_wS3_sma50_confluence": (np.abs(wS3 - s50) <= a) & (l <= np.maximum(wS3, s50) + 0.1 * a)
                                      & (c > np.maximum(wS3, s50)),
            "T_wH3_hold": (l <= wH3 + 0.15 * a) & (c > wH3) & (c1 > wH3) & (c < wH4),
        }
    anyt = np.zeros(n, bool)
    for x in trig.values():
        anyt |= np.nan_to_num(x, nan=0).astype(bool)
    ok = anyt & (k >= 170) & (k <= n - HMAX - 3) & (c >= 5) & (ind["vol20"] >= 100_000) \
        & np.isfinite(a) & (a > 0) & np.isfinite(s150) & np.isfinite(mS3) & np.isfinite(wS3)
    ks = np.where(ok)[0]
    if len(ks) == 0:
        return None
    dates = df.index[ks]
    f = pd.DataFrame({"Ticker": t, "Date": dates, "k": ks})
    for name, x in trig.items():
        f[name] = np.nan_to_num(x[ks], nan=0).astype(bool)
    cc = c[ks]
    f["above_sma50"] = cc > s50[ks]
    f["above_sma150"] = cc > s150[ks]
    f["sma50_gt_sma150"] = s50[ks] > s150[ks]
    f["sma50_rising"] = s50[ks] > shift(s50, 10)[ks]
    f["sma150_rising"] = s150[ks] > shift(s150, 20)[ks]
    f["near_sma50"] = np.abs(cc / s50[ks] - 1) <= 0.03
    f["not_extended"] = cc / s50[ks] - 1 <= 0.10
    f["above_mH3"] = cc > mH3[ks]
    f["above_mH4"] = cc > mH4[ks]
    f["above_mS3"] = cc > mS3[ks]
    f["below_wH4"] = cc < wH4[ks]
    f["w_narrow"] = ind["w_narrow"][ks]
    f["m_narrow"] = ind["m_narrow"][ks]
    f["mkt_above_sma50"] = mkt.reindex(dates)["mkt_above_sma50"].fillna(False).values.astype(bool)
    f["vol_surge"] = v[ks] >= 1.5 * ind["vol20"][ks]
    f["wH3_above_mH3"] = wH3[ks] > mH3[ks]
    f["atr_pct"] = a[ks] / cc * 100
    return f


def simulate(ind, ks):
    o, h, l, c, a, s50 = ind["o"], ind["h"], ind["l"], ind["c"], ind["atr"], ind["sma50"]
    N, V = len(ks), len(VARIANTS)
    ret = np.full((N, V), np.nan, np.float32)
    riskp = np.full((N, V), np.nan, np.float32)
    xday = np.full((N, V), -1, np.int32)
    f = ks + 1
    idx = f[:, None] + np.arange(HMAX + 1)[None, :]
    O, H, L, C, S = o[idx], h[idx], l[idx], c[idx], s50[idx]
    fill = O[:, 0]
    ak = a[ks]
    rows = np.arange(N)
    good = np.isfinite(fill) & (fill > 0)
    for vi, (kind, s, rr, hold) in enumerate(VARIANTS):
        stop = fill - s * ak
        Oh, Lh, Hh, Ch = O[:, :hold], L[:, :hold], H[:, :hold], C[:, :hold]
        gs = Oh <= stop[:, None]; gs[:, 0] = False
        sh = Lh <= stop[:, None]
        if kind == "bracket":
            tgt = fill + rr * s * ak
            gt = Oh >= tgt[:, None]; gt[:, 0] = False
            th = Hh >= tgt[:, None]; th[:, 0] = Ch[:, 0] >= tgt
            ev = gs | gt | sh | th
            has = ev.any(1)
            j = np.where(has, ev.argmax(1), hold)
            jj = np.minimum(j, hold - 1)
            px = np.where(gs[rows, jj], Oh[rows, jj], np.where(gt[rows, jj], Oh[rows, jj],
                          np.where(sh[rows, jj], stop, tgt)))
            px = np.where(has, px, O[rows, hold])
            xd = f + np.where(has, j, hold)
        else:
            tx = Ch < S[:, :hold]                       # close below SMA50 -> sell next open
            evs = gs | sh
            js = np.where(evs.any(1), evs.argmax(1), 10 ** 6)
            jt = np.where(tx.any(1), tx.argmax(1), 10 ** 6)
            stop_first = js <= jt
            j = np.where(stop_first, js, jt)
            has = j < 10 ** 6
            jj = np.minimum(j, hold - 1)
            pstop = np.where(gs[rows, jj], Oh[rows, jj], stop)
            ptrail = O[rows, np.minimum(jj + 1, HMAX)]
            px = np.where(has, np.where(stop_first, pstop, ptrail), O[rows, hold])
            xd = f + np.where(has, np.where(stop_first, j, j + 1), hold)
        r = (px / fill - 1) * 100 - COST
        okm = good & (stop > 0)
        ret[:, vi] = np.where(okm, r, np.nan)
        riskp[:, vi] = np.where(okm, s * ak / fill * 100, np.nan)
        xday[:, vi] = xd
    return ret, riskp, xday


def stats(r, rk):
    m = np.isfinite(r)
    r, rk = r[m].astype(float), rk[m].astype(float)
    n = len(r)
    if n == 0:
        return dict(n=0, wr=np.nan, pf=np.nan, expR=np.nan, avg=np.nan, worst=np.nan)
    gw, gl = r[r > 0].sum(), -r[r <= 0].sum()
    return dict(n=n, wr=(r > 0).mean() * 100, pf=gw / gl if gl > 0 else np.inf,
                expR=(r / rk).mean(), avg=r.mean(), worst=r.min())


def load_sample(seed, base, tickers_all, mkt, exclude=()):
    tk = [t for t in tickers_all if t not in set(exclude)]
    random.Random(seed).shuffle(tk)
    tk = sorted(tk[:SAMPLE])
    data = base["download"](tk, int(YEARS * 365))
    feats, rets, risks, xds = [], [], [], []
    for t, df in data.items():
        try:
            ind = indicators(df)
            fr = signal_rows(t, df, ind, mkt)
            if fr is None:
                continue
            r, rk, x = simulate(ind, fr["k"].values)
        except Exception as e:
            print("skip", t, type(e).__name__, e); continue
        feats.append(fr); rets.append(r); risks.append(rk); xds.append(x)
    F = pd.concat(feats, ignore_index=True)
    F["Date"] = pd.to_datetime(F["Date"])
    return F, np.vstack(rets), np.vstack(risks), np.vstack(xds), list(data)


def mask(F, trig, filters):
    m = F[trig].values.astype(bool).copy()
    for x in filters:
        m &= F[x].values.astype(bool)
    return m


def non_overlap(F, R, RK, X, m, v):
    order = F.sort_values(["Ticker", "k"]).index.values
    taken, busy = [], {}
    for i in order:
        if not m[i] or not np.isfinite(R[i, v]):
            continue
        t, k = F.at[i, "Ticker"], F.at[i, "k"]
        if k <= busy.get(t, -1):
            continue
        busy[t] = X[i, v]
        taken.append(i)
    return np.array(taken, dtype=int)


def describe(F, R, RK, tk, v, cut=None):
    rr, rk = R[tk, v].astype(float), RK[tk, v].astype(float)
    dd = F.loc[tk, "Date"]
    out = {}
    parts = [("All", np.ones(len(tk), bool))]
    if cut is not None:
        parts += [("Train", (dd < cut).values), ("Test", (dd >= cut).values)]
    for lab, mm in parts:
        s = stats(rr[mm], rk[mm])
        out.update({f"{lab}_n": s["n"], f"{lab}_WR": round(s["wr"], 1), f"{lab}_PF": round(s["pf"], 2),
                    f"{lab}_ExpR": round(s["expR"], 3), f"{lab}_Avg%": round(s["avg"], 2)})
    yrs = dd.dt.year.values
    out["Years"] = " ".join(f"{y}:{np.mean(rr[yrs == y] / rk[yrs == y]):+.2f}R/{(yrs == y).sum()}" for y in sorted(set(yrs)))
    out["AvgWin%"] = round(rr[rr > 0].mean(), 2) if (rr > 0).any() else 0
    out["AvgLoss%"] = round(rr[rr <= 0].mean(), 2) if (rr <= 0).any() else 0
    return out


def run():
    t0 = time.time()
    base = runpy.run_path("claude_run/tools/backtest_winrate.py", run_name="lib")
    ns = base["load_scanner"]("daily_sweep_sma50_strategy_v1.py")
    tickers_all = ns["get_tickers"]()
    spy = base["download"](["SPY"], int(YEARS * 365) + 400)["SPY"]["Close"]
    mkt = pd.DataFrame({"mkt_above_sma50": spy > spy.rolling(50).mean()})
    A = load_sample(SEED_A, base, tickers_all, mkt)
    short, cut = search(A, t0)
    if short is None:
        return
    F, R, RK, X, usedA = A
    fin = []
    for _, rw in short.iterrows():
        fl = [] if rw.Filters == "(none)" else rw.Filters.split(" & ")
        v = int(rw._v)
        tk = non_overlap(F, R, RK, X, mask(F, rw.Trigger, fl), v)
        d = describe(F, R, RK, tk, v, cut)
        fin.append({"Trigger": rw.Trigger, "Filters": rw.Filters, "Exit": rw.Exit, "_v": v,
                    **{f"A_{k}": val for k, val in d.items()}})
    del A, F, R, RK, X
    B = load_sample(SEED_B, base, tickers_all, mkt, exclude=usedA)
    FB, RB_, RKB, XB, usedB = B
    print(f"\nsample B: {len(usedB)} different tickers, {len(FB)} signal rows | {(time.time()-t0)/60:.1f} min", flush=True)
    best_trades = None
    anyrow = np.ones(len(FB), bool)                 # benchmark: any level event, same exit
    base_cache = {}
    for row in fin:
        fl = [] if row["Filters"] == "(none)" else row["Filters"].split(" & ")
        v = row["_v"]
        tk = non_overlap(FB, RB_, RKB, XB, mask(FB, row["Trigger"], fl), v)
        d = describe(FB, RB_, RKB, tk, v)
        row.update({f"B_{k}": val for k, val in d.items()})
        if v not in base_cache:
            tb = non_overlap(FB, RB_, RKB, XB, anyrow, v)
            base_cache[v] = stats(RB_[tb, v], RKB[tb, v])["expR"]
        row["B_Benchmark_ExpR"] = round(base_cache[v], 3)
        fkey = (v, row["Filters"])
        if fkey not in base_cache:                  # same filters + same exit, any level event
            mf = anyrow.copy()
            for x in fl:
                mf &= FB[x].values.astype(bool)
            tf = non_overlap(FB, RB_, RKB, XB, mf, v)
            base_cache[fkey] = stats(RB_[tf, v], RKB[tf, v])["expR"]
        row["B_SameFilters_ExpR"] = round(base_cache[fkey], 3)
        row["B_Edge_R"] = round(row["B_All_ExpR"] - base_cache[fkey], 3)
    FN = pd.DataFrame(fin)
    FN["Score"] = FN[["A_Train_ExpR", "A_Test_ExpR", "B_All_ExpR"]].min(1)
    FN = FN.sort_values(["Score"], ascending=False)
    FN = pd.concat([FN[FN.B_Edge_R > 0], FN[~(FN.B_Edge_R > 0)]])
    FN.drop(columns="_v").to_csv("cam_final.csv", index=False)
    cols = ["Trigger", "Filters", "Exit", "A_Train_n", "A_Train_ExpR", "A_Test_n", "A_Test_WR", "A_Test_PF",
            "A_Test_ExpR", "B_All_n", "B_All_WR", "B_All_PF", "B_All_ExpR", "B_Benchmark_ExpR", "B_SameFilters_ExpR", "B_Edge_R", "Score"]
    print("\nONE TRADE PER TICKER AT A TIME — sample A (train/test) and fresh sample B:")
    print(FN[cols].to_string(index=False))
    print("\nYEARS (sample B) for the top 8:")
    for _, r in FN.head(8).iterrows():
        print(f"  {r.Trigger} | {r.Filters} | {r.Exit}\n     A {r.A_Years}\n     B {r.B_Years}")
    top = FN.iloc[0]
    fl = [] if top.Filters == "(none)" else top.Filters.split(" & ")
    tk = non_overlap(FB, RB_, RKB, XB, mask(FB, top.Trigger, fl), int(top._v))
    v = int(top._v)
    FB.loc[tk, ["Ticker", "Date", "atr_pct"]].assign(Ret_pct=RB_[tk, v], Risk_pct=RKB[tk, v],
        R_mult=RB_[tk, v] / RKB[tk, v]).to_csv("cam_trades_best.csv", index=False)
    print(f"done in {(time.time()-t0)/60:.1f} min")


def search(A, t0):
    F, R, RK, X, usedA = A
    d0, d1 = F.Date.min(), F.Date.max()
    cut = d0 + (d1 - d0) / 2
    tq = cut + (d1 - cut) / 2
    tr_m = (F.Date < cut).values; te1 = ((F.Date >= cut) & (F.Date < tq)).values; te2 = (F.Date >= tq).values
    print(f"sample A: {len(usedA)} tickers, {len(F)} signal rows, {len(VARIANTS)} exits, {d0.date()}..{d1.date()}, "
          f"train < {cut.date()} | {(time.time()-t0)/60:.1f} min", flush=True)
    vb = VARIANTS.index(("bracket", 1.5, 2.0, 20))
    vt = VARIANTS.index(("sma50_exit", 2.0, 0, 60))
    print("\nBASELINE per trigger, no filter (stop 1.5 ATR / 2R / 20d   |   2 ATR stop + SMA50 exit):")
    for tg in TRIGGERS:
        m = F[tg].values
        s, s2 = stats(R[m, vb], RK[m, vb]), stats(R[m, vt], RK[m, vt])
        print(f"  {tg:26s} n={s['n']:6d} win={s['wr']:5.1f}% PF={s['pf']:.2f} expR={s['expR']:+.3f}   |   "
              f"win={s2['wr']:5.1f}% PF={s2['pf']:.2f} expR={s2['expR']:+.3f}")
    fcols = {x: F[x].values.astype(bool) for x in FILTERS}
    combos = [()] + [(a,) for a in FILTERS] + list(itertools.combinations(FILTERS, 2)) \
        + list(itertools.combinations(FILTERS, 3))
    rows = []
    for tg in TRIGGERS:
        tm = F[tg].values.astype(bool)
        Rt, RKt = R[tm], RK[tm]
        trt, t1t, t2t = tr_m[tm], te1[tm], te2[tm]
        fct = {x: fcols[x][tm] for x in FILTERS}
        with np.errstate(invalid="ignore", divide="ignore"):
            ERt = Rt / RKt
        for fc in combos:
            if any(a in fc and b in fc for a, b in BAD):
                continue
            m = np.ones(len(Rt), bool)
            for x in fc:
                m &= fct[x]
            if m.sum() < 500:
                continue
            mtr = m & trt
            ntr = np.isfinite(ERt[mtr]).sum(0)
            with np.errstate(invalid="ignore"):
                ER = np.nanmean(ERt[mtr], 0)
            cand = np.where((ntr >= 200) & (ER >= 0.10))[0]
            if len(cand) == 0:
                continue
            m1, m2 = m & t1t, m & t2t
            for v in cand:
                s1 = stats(Rt[mtr, v], RKt[mtr, v])
                if s1["pf"] < 1.15:
                    continue
                a1, a2 = stats(Rt[m1, v], RKt[m1, v]), stats(Rt[m2, v], RKt[m2, v])
                st = stats(np.concatenate([Rt[m1, v], Rt[m2, v]]), np.concatenate([RKt[m1, v], RKt[m2, v]]))
                rows.append({"Trigger": tg, "Filters": " & ".join(fc) or "(none)", "Exit": vname(v),
                             "Train_n": s1["n"], "Train_WR": round(s1["wr"], 1), "Train_PF": round(s1["pf"], 2),
                             "Train_ExpR": round(s1["expR"], 3),
                             "Test_n": st["n"], "Test_WR": round(st["wr"], 1), "Test_PF": round(st["pf"], 2),
                             "Test_ExpR": round(st["expR"], 3),
                             "H1_ExpR": round(a1["expR"], 3), "H2_ExpR": round(a2["expR"], 3), "_v": v})
    C = pd.DataFrame(rows)
    print(f"\ntrain candidates (expR >= 0.10, PF >= 1.15): {len(C)} | {(time.time()-t0)/60:.1f} min", flush=True)
    if C.empty:
        return None, cut
    C["Robust"] = (C.Test_n >= 150) & (C.Test_PF >= 1.10) & (C.H1_ExpR > 0) & (C.H2_ExpR > 0)
    C["MinExpR"] = C[["Train_ExpR", "H1_ExpR", "H2_ExpR"]].min(1)
    C = C.sort_values(["Robust", "MinExpR"], ascending=False)
    C.drop(columns="_v").head(5000).to_csv("cam_candidates.csv", index=False)
    pd.set_option("display.width", 300); pd.set_option("display.max_colwidth", 60)
    show = ["Trigger", "Filters", "Exit", "Train_n", "Train_WR", "Train_PF", "Train_ExpR", "Test_n", "Test_WR",
            "Test_PF", "Test_ExpR", "H1_ExpR", "H2_ExpR"]
    rb = C[C.Robust]
    print(f"robust: {len(rb)}\n\nTOP ROBUST (sorted by the worst of train / test-half expectancy):")
    print(rb[show].head(30).to_string(index=False))
    print("\nBEST ROBUST RULE PER TRIGGER:")
    print(rb.groupby("Trigger").head(1)[show].to_string(index=False))
    if rb.empty:
        return None, cut
    short = pd.concat([rb.head(15), rb.groupby("Trigger").head(2)]).drop_duplicates(["Trigger", "Filters", "Exit"]).head(36)
    return short, cut


if __name__ == "__main__":
    run()
