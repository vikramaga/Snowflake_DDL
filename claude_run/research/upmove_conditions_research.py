# Event study (run by ClaudeRunner): which chart conditions are most often
# followed by a strong move up?
#
# For every stock-day (random ~2,000 US stocks, ~3.6 years of daily data) it
# records ~45 point-in-time conditions (trend, position vs highs, momentum,
# volatility contraction, volume, price action, market regime, price/liquidity)
# and the outcome:
#   GOOD = from the NEXT day's open, the high reaches +15% within 20 sessions
#          before the low touches -7% (a day touching both counts as a fail).
# Signals are counted only on the FIRST day a combination becomes true
# ("fresh"), so a condition that stays true for weeks is not counted 20 times.
# Single conditions and every combination of 2 and 3 are scored; rules are
# chosen on the first half of the period and checked on the second half
# (both quarters), and must appear in at least 10 different months.
#
# Outputs: up_conditions_all.csv (every combo with enough signals),
# up_conditions_best.csv (robust winners), up_signals_best.csv (signal list
# of the top rule).
import os, sys, time, random, itertools, warnings, runpy
import numpy as np, pandas as pd

warnings.filterwarnings("ignore")
SAMPLE = int(os.environ.get("UP_SAMPLE", "2000"))
YEARS = float(os.environ.get("UP_YEARS", "3.6"))
UP, DN, H = 0.15, 0.07, 20          # +15% before -7% within 20 sessions
ALT_UP = 0.10                       # second label: close after 20 sessions >= +10%
# UP_MODE=atr: volatility-normalised target instead: +3 x ATR14 before -1.5 x ATR14
# (a 2:1 trade, breakeven hit rate 33.3%); R per trade is also reported.
ATR_MODE = os.environ.get("UP_MODE", "") == "atr"
UPA, DNA = 3.0, 1.5
TAG = "_atr" if ATR_MODE else ""
SEED = 11


def sma(x, n): return pd.Series(x).rolling(n).mean().values
def ema(x, n): return pd.Series(x).ewm(span=n, adjust=False).mean().values
def rmax(x, n): return pd.Series(x).rolling(n).max().values
def rmin(x, n): return pd.Series(x).rolling(n).min().values
def shift(x, k=1):
    out = np.full(len(x), np.nan); out[k:] = x[:-k]; return out


def rsi14(c):
    d = np.diff(c, prepend=np.nan)
    g = pd.Series(np.where(d > 0, d, 0.0)).ewm(alpha=1/14, adjust=False).mean().values
    l = pd.Series(np.where(d < 0, -d, 0.0)).ewm(alpha=1/14, adjust=False).mean().values
    with np.errstate(divide="ignore", invalid="ignore"):
        r = 100 - 100 / (1 + g / l)
    r[l == 0] = 100.0; r[:15] = np.nan
    return r


def outcome(o, h, l, c, atr=None):
    """GOOD[k]: entry = open[k+1]; high >= target within H sessions before low <= stop
    (target/stop = +15%/-7%, or +3/-1.5 x ATR14 of day k in ATR mode).
    ALT[k]: close[k+H] >= open[k+1] * 1.10. RR[k]: result in R (risk = entry - stop):
    target = +UP/DN R, stop = -1 R, otherwise the close after H sessions (capped).
    NaN where the window is incomplete."""
    n = len(c); good = np.full(n, np.nan); alt = np.full(n, np.nan); rr = np.full(n, np.nan)
    for k in range(n - H - 1):
        e = o[k + 1]
        if not np.isfinite(e) or e <= 0:
            continue
        if ATR_MODE:
            if atr is None or not np.isfinite(atr[k]) or atr[k] <= 0:
                continue
            up, dn = e + UPA * atr[k], e - DNA * atr[k]
        else:
            up, dn = e * (1 + UP), e * (1 - DN)
        risk = e - dn
        res, r = 0.0, None
        for j in range(k + 1, k + 1 + H):
            if l[j] <= dn: res, r = 0.0, -1.0; break
            if h[j] >= up: res, r = 1.0, (up - e) / risk; break
        if r is None:
            r = max(-1.0, min((up - e) / risk, (c[k + H] - e) / risk))
        good[k] = res; rr[k] = r
        alt[k] = float(c[k + H] >= e * (1 + ALT_UP))
    return good, alt, rr


def features(df, spy):
    o, h, l, c, v = (df[x].values.astype(float) for x in ("Open", "High", "Low", "Close", "Volume"))
    n = len(c)
    s20, s50, s150, s200 = sma(c, 20), sma(c, 50), sma(c, 150), sma(c, 200)
    e8, e21 = ema(c, 8), ema(c, 21)
    pc = shift(c)
    tr = np.nanmax(np.vstack([h - l, np.abs(h - pc), np.abs(l - pc)]), axis=0)
    atr10, atr50, atr14 = sma(tr, 10), sma(tr, 50), sma(tr, 14)
    r = rsi14(c); ravg = sma(r, 9)
    hi252 = rmax(h, 252); hi20p = shift(rmax(h, 20)); hi55p = shift(rmax(h, 55)); hi252p = shift(hi252)
    v10, v50 = sma(v, 10), sma(v, 50)
    rng = h - l
    ret = lambda k: c / shift(c, k) - 1
    spy_c = spy.reindex(df.index)["Close"].ffill().values
    spy63 = spy_c / shift(spy_c, 63) - 1
    sp50, sp200 = sma(spy_c, 50), sma(spy_c, 200)
    with np.errstate(invalid="ignore", divide="ignore"):
        F = {
            # trend
            "t_above_sma50": c > s50,
            "t_sma50_rising": s50 > shift(s50, 10),
            "t_above_sma200": c > s200,
            "t_sma50_gt_sma200": s50 > s200,
            "t_stage2": (c > s50) & (s50 > s150) & (s150 > s200) & (s200 > shift(s200, 20)),
            # position vs highs / averages
            "p_within10_52wh": c >= 0.90 * hi252,
            "p_within25_52wh": c >= 0.75 * hi252,
            "p_new_52wh": c > hi252p,
            "p_40below_52wh": c < 0.60 * hi252,
            "p_0to5_above_sma50": (c >= s50) & (c <= 1.05 * s50),
            "p_gt20_above_sma50": c > 1.20 * s50,
            # momentum
            "m_rsi_50_60": (r >= 50) & (r < 60),
            "m_rsi_57_63": (r >= 57) & (r <= 63),
            "m_rsi_60_70": (r >= 60) & (r < 70),
            "m_rsi_gt70": r >= 70,
            "m_rsi_lt35": r < 35,
            "m_rsi_above_avg": r > ravg,
            "m_ret20_gt20": ret(20) > 0.20,
            "m_ret60_gt30": ret(60) > 0.30,
            "m_rs63_gt20": (ret(63) - spy63) > 0.20,
            "m_rs63_pos": (ret(63) - spy63) > 0,
            # volatility contraction
            "v_atr_contract": atr10 / atr50 < 0.75,
            "v_tight10": (rmax(h, 10) - rmin(l, 10)) / c < 0.08,
            "v_nr7": rng <= rmin(rng, 7),
            "v_inside_day": (h < shift(h)) & (l > shift(l)),
            "v_atr_pct_gt5": atr14 / c > 0.05,
            # volume
            "vol_dryup": v10 / v50 < 0.7,
            "vol_surge2": v > 2 * shift(v50),
            "vol_surge3": v > 3 * shift(v50),
            # price action
            "a_breakout20": c > hi20p,
            "a_breakout55": c > hi55p,
            "a_gap_up": o > shift(h),
            "a_close_top20": (c - l) >= 0.8 * rng,
            "a_up4": ret(1) > 0.04,
            "a_pullback_ema21": (l <= e21) & (c > e21) & (s50 > shift(s50, 10)),
            "a_3down": (c < pc) & (pc < shift(c, 2)) & (shift(c, 2) < shift(c, 3)),
            "a_reclaim_all": (c > np.fmax(np.fmax(e8, s20), s50)) & ~(pc > shift(np.fmax(np.fmax(e8, s20), s50))),
            "a_green": c > o,
            # market
            "mkt_above_sma50": spy_c > sp50,
            "mkt_above_sma200": spy_c > sp200,
            # price / liquidity
            "pr_lt10": c < 10,
            "pr_10_50": (c >= 10) & (c < 50),
            "pr_gt50": c >= 50,
            "liq_gt50m": sma(c * v, 20) > 50e6,
        }
    F = {k: np.nan_to_num(val, nan=0).astype(bool) for k, val in F.items()}
    valid = np.isfinite(s200) & np.isfinite(hi252) & (np.arange(n) >= 252) & (c >= 2) & (sma(c * v, 20) >= 2e6)
    return F, valid, atr14


def main():
    t0 = time.time()
    base = runpy.run_path("claude_run/tools/backtest_winrate.py", run_name="lib")
    ns = base["load_scanner"]("daily_sweep_sma50_strategy_v1.py")
    tickers = ns["get_tickers"]()
    random.Random(SEED).shuffle(tickers)
    tickers = sorted(tickers[:SAMPLE])
    days = int(YEARS * 365)
    data = base["download"](tickers, days)
    spy = base["download"](["SPY"], days + 400)["SPY"]
    print(f"tickers {len(data)} | download {(time.time()-t0)/60:.1f} min", flush=True)

    names = None; cols = {}; prevs = {}; Y, Y2, D, T, RR = [], [], [], [], []
    for t, df in data.items():
        F, valid, atr14 = features(df, spy)
        g, a, rr = outcome(*(df[x].values.astype(float) for x in ("Open", "High", "Low", "Close")), atr=atr14)
        keep = valid & np.isfinite(g)
        if keep.sum() == 0:
            continue
        if names is None:
            names = list(F); cols = {k: [] for k in names}; prevs = {k: [] for k in names}
        for k in names:
            cols[k].append(F[k][keep])
            prevs[k].append(np.r_[False, F[k][:-1]][keep])     # same condition the day before
        Y.append(g[keep]); Y2.append(a[keep]); D.append(df.index[keep].values); T.append(np.full(keep.sum(), t))
        RR.append(rr[keep])
    C = {k: np.concatenate(cols[k]) for k in names}
    P = {k: np.concatenate(prevs[k]) for k in names}
    y = np.concatenate(Y).astype(bool); y2 = np.concatenate(Y2).astype(bool)
    dates = pd.to_datetime(np.concatenate(D)); tick = np.concatenate(T); RRa = np.concatenate(RR)
    N = len(y)
    d0, d1 = dates.min(), dates.max(); cut = d0 + (d1 - d0) / 2; q3 = cut + (d1 - cut) / 2
    tr_m, te1, te2 = np.asarray(dates < cut), np.asarray((dates >= cut) & (dates < q3)), np.asarray(dates >= q3)
    month = np.asarray(dates.to_period("M").astype(str))
    base_tr, base_te = y[tr_m].mean(), y[~tr_m].mean()
    print(f"stock-days {N:,} | {d0.date()} .. {d1.date()} | train < {cut.date()} | "
          f"base rate GOOD: train {base_tr*100:.1f}% test {base_te*100:.1f}% | ALT {y2.mean()*100:.1f}% | "
          f"{(time.time()-t0)/60:.1f} min", flush=True)

    # packed bitsets for fast AND / popcount
    pk = lambda b: np.packbits(b)
    POP = np.array([bin(i).count("1") for i in range(256)], dtype=np.int64)
    cnt = lambda p: int(POP[p].sum())
    Cp = {k: pk(C[k]) for k in names}; Pp = {k: pk(P[k]) for k in names}
    Yp, Y2p = pk(y), pk(y2)
    TRp, TE1p, TE2p = pk(tr_m), pk(te1), pk(te2)

    rows = []
    combos = [(a,) for a in names] + list(itertools.combinations(names, 2)) + list(itertools.combinations(names, 3))
    excl = [{"pr_lt10", "pr_10_50", "pr_gt50"}, {"m_rsi_50_60", "m_rsi_60_70", "m_rsi_gt70", "m_rsi_lt35"},
            {"p_within10_52wh", "p_40below_52wh"}, {"p_0to5_above_sma50", "p_gt20_above_sma50"}]
    for cb in combos:
        if any(len(set(cb) & e) > 1 for e in excl):
            continue
        cur = Cp[cb[0]].copy(); prv = Pp[cb[0]].copy()
        for k in cb[1:]:
            cur &= Cp[k]; prv &= Pp[k]
        fresh = cur & ~prv
        n_tr = cnt(fresh & TRp)
        if n_tr < 150:
            continue
        n1, n2 = cnt(fresh & TE1p), cnt(fresh & TE2p)
        if n1 + n2 < 150:
            continue
        g_tr = cnt(fresh & TRp & Yp); g1 = cnt(fresh & TE1p & Yp); g2 = cnt(fresh & TE2p & Yp)
        a_all = cnt(fresh & Y2p)
        rows.append({"Conditions": " & ".join(cb), "k": len(cb),
                     "Train_n": n_tr, "Train_P%": round(g_tr / n_tr * 100, 1),
                     "Test_n": n1 + n2, "Test_P%": round((g1 + g2) / (n1 + n2) * 100, 1),
                     "TestH1_P%": round(g1 / n1 * 100, 1) if n1 else np.nan,
                     "TestH2_P%": round(g2 / n2 * 100, 1) if n2 else np.nan,
                     "TestH1_n": n1, "TestH2_n": n2,
                     "Lift_test": round(((g1 + g2) / (n1 + n2)) / base_te, 2),
                     "Close20_+10%_P%": round(a_all / (n_tr + n1 + n2) * 100, 1)})
    R = pd.DataFrame(rows)
    print(f"combos scored: {len(R):,} | {(time.time()-t0)/60:.1f} min", flush=True)
    R["MinP"] = R[["Train_P%", "Test_P%", "TestH1_P%", "TestH2_P%"]].min(axis=1)
    R = R.sort_values("MinP", ascending=False)
    R.head(5000).to_csv(f"up_conditions_all{TAG}.csv", index=False)

    # months check + signal list for the most robust rules
    best = []
    for _, rw in R[(R.TestH1_n >= 40) & (R.TestH2_n >= 40)].head(60).iterrows():
        cb = rw.Conditions.split(" & ")
        cur = np.ones(N, bool); prv = np.ones(N, bool)
        for k in cb:
            cur &= C[k]; prv &= P[k]
        fresh = cur & ~prv
        mo = pd.Series(y[fresh]).groupby(month[fresh]).agg(["size", "mean"])
        out = dict(rw); out["Months"] = len(mo); out["Months_P_lt_base"] = int((mo["mean"] < base_te).sum())
        out["Signals_per_month"] = round(fresh.sum() / len(mo), 1)
        out["AvgR_train"] = round(float(np.mean(RRa[fresh & tr_m])), 3)
        out["AvgR_test"] = round(float(np.mean(RRa[fresh & ~tr_m])), 3)
        best.append(out)
    B = pd.DataFrame(best)
    B = B[B.Months >= 10].sort_values(["MinP", "Test_P%"], ascending=False)
    B.to_csv(f"up_conditions_best{TAG}.csv", index=False)
    pd.set_option("display.width", 250); pd.set_option("display.max_colwidth", 80)
    print(f"\nMODE: {'+3 / -1.5 x ATR14 (breakeven 33.3%)' if ATR_MODE else '+15% / -7%'} within {H} sessions")
    print(f"BASE RATE (any stock-day): train {base_tr*100:.1f}%  test {base_te*100:.1f}%  | avg R all days "
          f"train {np.mean(RRa[tr_m]):+.3f} test {np.mean(RRa[~tr_m]):+.3f}")
    print("\nSINGLE CONDITIONS (fresh signals):")
    print(R[R.k == 1].sort_values("Test_P%", ascending=False)[["Conditions", "Train_n", "Train_P%", "Test_n", "Test_P%", "Lift_test"]].to_string(index=False))
    print("\nMOST ROBUST COMBINATIONS (min of train / test / test halves):")
    print(B.head(30)[["Conditions", "Train_n", "Train_P%", "Test_n", "Test_P%", "TestH1_P%", "TestH2_P%", "Lift_test",
                       "AvgR_train", "AvgR_test", "Months", "Signals_per_month"]].to_string(index=False))
    print("\nBEST BY AVERAGE R (both halves positive):")
    print(B[(B.AvgR_train > 0) & (B.AvgR_test > 0)].sort_values("AvgR_test", ascending=False).head(15)[
        ["Conditions", "Train_n", "Train_P%", "AvgR_train", "Test_n", "Test_P%", "AvgR_test", "Signals_per_month"]].to_string(index=False))
    if len(B):
        cb = B.iloc[0].Conditions.split(" & ")
        cur = np.ones(N, bool); prv = np.ones(N, bool)
        for k in cb:
            cur &= C[k]; prv &= P[k]
        fresh = cur & ~prv
        pd.DataFrame({"Ticker": tick[fresh], "Date": dates[fresh], "GOOD": y[fresh], "Close20_+10%": y2[fresh]}) \
            .to_csv(f"up_signals_best{TAG}.csv", index=False)
    print(f"done in {(time.time()-t0)/60:.1f} min")


main()
