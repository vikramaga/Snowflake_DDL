# Chart review of the Best 10 watchlists: every stock that appeared in any scanner's Best 10 in the
# scheduled runs of the last CHART_REVIEW_DAYS days (default 90; 0 = all), not only the newest list.
# Runs automatically after every scheduled 06:00 IST run: the RSI Digest workflow calls
# rsi_amount_filter.py --email, which runs best10_digest.py and then this script with the run id.
# For every stock: trend, pullback/reclaim trigger, extension, relative strength vs SPY, volume,
# room to resistance, earnings date. Each stock gets BUY TODAY / WATCH (with the trigger price) /
# PASS / AVOID and the reasons. Stops follow the Best 10 stop rules (at least 2 ATR below the close,
# no deeper than 3 ATR).
# BUY TODAY = all of: uptrend (above SMA200 and SMA50 > SMA200, or above a rising SMA200); green close
#   back above JMA and EMA8 after a pullback, in the top half of the day's range; not extended
#   (<= 2 ATR above EMA21, RSI 45-70); 20-day return >= SPY's; >= 1.5R to the 60-day high; stop <= 10%.
# AVOID = earnings within 5 business days, < $3 or < $5M/day, ATR < 1% of price (pinned by a pending deal /
#   SPAC), downtrend, extended (> 3 ATR / RSI > 75), or below a falling SMA50 and EMA21.
# Usage: python claude_run/tools/chart_review.py [run_id] [--email] [--no-tv]
#   run_id = the newest run to include (lists after it are ignored); env CHART_REVIEW_DAYS=90.
# Columns First_Seen / Last_Seen / Times_Listed / In_Latest_List / Chg_Since_First_% show each stock's
# history on the lists. Earnings dates are looked up only for stocks that pass the chart checks.
# Outputs: chart_review_YYYYMMDD.csv, chart_review_buys/watch_YYYYMMDD.png (email attachments),
#   TradingView list "Chart Review - YYYY-MM-DD" (Buy today + Watch), email.
import glob, os, re, subprocess, sys, time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
subprocess.call([sys.executable, "-m", "pip", "install", "-q", "yfinance", "matplotlib", "lxml"],
                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
import numpy as np, pandas as pd, yfinance as yf
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt

ROOT = "claude_run/results"
STOP_ATR = 2.0


DAYS = int(os.environ.get("CHART_REVIEW_DAYS", "90"))
TS = re.compile(r"_(20\d{6})_\d{4}\.csv$")


def load_best10(want=""):
    """All Best 10 lists of the scheduled runs (>= 10 scanners) up to run `want`, last DAYS days."""
    runs = {}
    for f in glob.glob(f"{ROOT}/[0-9]*/*/best10_*.csv"):
        runs.setdefault(f.split("/")[2], set()).add(f.split("/")[3])
    sched = sorted((r for r, s in runs.items() if len(s) >= 10), key=int)
    if want and want in runs:
        sched = [r for r in sched if int(r) <= int(want)] + ([] if want in sched else [want])
    rows = []
    for r in sched:
        for f in glob.glob(f"{ROOT}/{r}/*/best10_*.csv"):
            m = TS.search(f)
            if not m: continue
            try: x = pd.read_csv(f)
            except Exception: continue
            if not len(x) or "Ticker" not in x: continue
            rows.append(x.assign(Scanner=f.split("/")[3], Run=r, Listed=pd.Timestamp(m.group(1))))
    a = pd.concat(rows, ignore_index=True)
    a["Ticker"] = a.Ticker.astype(str).str.upper()
    if DAYS > 0:
        a = a[a.Listed >= a.Listed.max() - pd.Timedelta(days=DAYS)]
    latest_run = max(a.Run, key=int)
    a["_px"] = pd.to_numeric(a.get("Entry_Price", a.get("Price")), errors="coerce")
    a = a.sort_values(["Listed", "Run"])
    last = a.groupby("Ticker").tail(1).set_index("Ticker")
    g = a.groupby("Ticker").agg(Scanners=("Scanner", lambda x: ", ".join(sorted(set(x)))), N_Scanners=("Scanner", "nunique"),
                                Best_Score=("Best_Score", "max"), Times_Listed=("Run", "nunique"),
                                First_Seen=("Listed", "min"), Last_Seen=("Listed", "max"), Price_First=("_px", "first"))
    # stop / news / upside from the stock's most recent listing (lowest stop if several scanners listed it that day)
    lr = a.merge(last[["Run"]].rename(columns={"Run": "_lr"}), left_on="Ticker", right_index=True)
    lr = lr[lr.Run == lr._lr]
    g["Scanner_Stop"] = lr.groupby("Ticker").Stop_Loss.min()
    g["Analyst_Upside"] = lr.groupby("Ticker")["Analyst_Upside_%"].max() if "Analyst_Upside_%" in lr else np.nan
    g["Headline"] = last.get("Catalyst_Headline")
    g["In_Latest_List"] = g.index.isin(a[a.Run == latest_run].Ticker)
    g = g.reset_index()
    print(f"Best 10 lists: {a.Run.nunique()} runs ({a.Listed.min():%d %b} - {a.Listed.max():%d %b}), {len(a)} rows, {len(g)} stocks, "
          f"{int(g.In_Latest_List.sum())} in the newest list (run {latest_run})")
    return latest_run, g


def calc_jma(c, period=13, phase=40, power=2):
    pr = phase / 100 + 1.5; beta = 0.45 * (period - 1) / (0.45 * (period - 1) + 2); alpha = beta ** power
    out = np.full(len(c), np.nan); e0 = e1 = e2 = j = 0.0
    for i, v in enumerate(c):
        if np.isnan(v): continue
        e0 = (1 - alpha) * v + alpha * e0; e1 = (v - e0) * (1 - beta) + beta * e1
        e2 = (e0 + pr * e1 - j) * (1 - alpha) ** 2 + alpha ** 2 * e2; j = e2 + j; out[i] = j
    return out


def download(tickers):
    out = {}
    for i in range(0, len(tickers), 60):
        b = tickers[i:i + 60]
        try:
            raw = yf.download(b, period="2y", group_by="ticker", auto_adjust=True, actions=False, threads=True, progress=False)
        except Exception:
            continue
        for t in b:
            try:
                d = raw[t] if isinstance(raw.columns, pd.MultiIndex) else raw
                d = d.dropna(subset=["Close"]); d.index = pd.to_datetime(d.index).tz_localize(None)
                if len(d) > 220: out[t] = d[["Open", "High", "Low", "Close", "Volume"]].astype(float)
            except Exception:
                pass
    return out


def next_earn(t):
    try:
        cal = yf.Ticker(t).calendar
        ds = cal.get("Earnings Date") if isinstance(cal, dict) else None
        if ds:
            ds = [pd.Timestamp(x) for x in ds]
            ds = [x for x in ds if x >= pd.Timestamp.today().normalize() - pd.Timedelta(days=1)]
            if ds: return t, min(ds)
    except Exception:
        pass
    return t, None


def ind(d):
    o, h, l, c, v = (d[k].values for k in ("Open", "High", "Low", "Close", "Volume"))
    s = pd.Series(c)
    pc = np.r_[np.nan, c[:-1]]
    atr = pd.Series(np.nanmax(np.vstack([h - l, np.abs(h - pc), np.abs(l - pc)]), axis=0)).rolling(14).mean().values
    dd = s.diff(); up = dd.clip(lower=0).ewm(alpha=1 / 14, adjust=False).mean(); dn = (-dd.clip(upper=0)).ewm(alpha=1 / 14, adjust=False).mean()
    rsi = (100 - 100 / (1 + up / dn.replace(0, np.nan))).fillna(100).values
    macd = s.ewm(span=12, adjust=False).mean() - s.ewm(span=26, adjust=False).mean(); sig = macd.ewm(span=9, adjust=False).mean()
    return dict(o=o, h=h, l=l, c=c, v=v, atr=atr, rsi=rsi, macd=macd.values, sig=sig.values, jma=calc_jma(c),
                e8=s.ewm(span=8, adjust=False).mean().values, e21=s.ewm(span=21, adjust=False).mean().values,
                s50=s.rolling(50).mean().values, s200=s.rolling(200).mean().values, vavg=pd.Series(v).rolling(20).mean().shift(1).values)


def review(t, d, spy_ret20, row, earn):
    x = ind(d); i = len(d) - 1
    c, h, l, o = x["c"], x["h"], x["l"], x["o"]
    a = x["atr"][i]; close = c[i]
    if not np.isfinite(a) or a <= 0: return None
    reasons_bad, reasons_good = [], []
    stage2 = close > x["s200"][i] and x["s50"][i] > x["s200"][i]
    s200_up = x["s200"][i] > x["s200"][i - 20]
    s50_up = x["s50"][i] > x["s50"][i - 10]
    ret20 = (close / c[i - 20] - 1) * 100; rs20 = ret20 - spy_ret20
    ext = (close - x["e21"][i]) / a
    green = close > o[i]; clv = (close - l[i]) / (h[i] - l[i]) if h[i] > l[i] else 0.5
    above_short = close > x["jma"][i] and close > x["e8"][i]
    was_below = c[i - 1] <= x["jma"][i - 1] or c[i - 1] <= x["e8"][i - 1] or c[i - 2] <= x["e8"][i - 2]
    touched = min(l[i - 4:i + 1]) <= max(x["e21"][i], x["s50"][i]) * 1.01
    jma_up = x["jma"][i] > x["jma"][i - 1]
    volx = x["v"][i] / x["vavg"][i] if x["vavg"][i] > 0 else np.nan
    dollar_vol = x["vavg"][i] * close / 1e6
    hi60 = h[i - 60:i].max()
    # stop: the Best 10 rule (at least 2 ATR below), but no deeper than 3 ATR (an old weekly/monthly stop can be far away)
    sc = row.Scanner_Stop if np.isfinite(row.Scanner_Stop) and row.Scanner_Stop < close else np.inf
    stop = max(min(sc, close - STOP_ATR * a), close - 3 * a)
    risk = close - stop
    target = hi60 if hi60 > close + 0.5 * a else close + 2 * risk
    rr = (target - close) / risk
    bd = np.busday_count(pd.Timestamp.today().date(), earn.date()) if earn is not None else None
    trig = max(x["jma"][i], x["e8"][i])

    # avoid
    if bd is not None and 0 <= bd <= 5: reasons_bad.append(f"earnings {earn:%d %b}")
    if close < 3 or dollar_vol < 5: reasons_bad.append(f"thin/penny (${dollar_vol:.1f}M/day)")
    if a / close < 0.01: reasons_bad.append(f"pinned (ATR {a / close * 100:.2f}% of price: likely a pending deal or SPAC)")
    if close < x["s200"][i] and not s200_up and x["s50"][i] < x["s200"][i]: reasons_bad.append("downtrend (below falling SMA200, SMA50 < SMA200)")
    if ext > 3 or x["rsi"][i] > 75: reasons_bad.append(f"extended ({ext:.1f} ATR above EMA21, RSI {x['rsi'][i]:.0f})")
    if close < x["s50"][i] and not s50_up and close < x["e21"][i]: reasons_bad.append("below falling SMA50 and EMA21")
    # buy-today checklist
    checks = {
        "uptrend": stage2 or (close > x["s200"][i] and s200_up),
        "trigger": green and above_short and (was_below or touched) and clv >= 0.5,
        "not_extended": ext <= 2 and 45 <= x["rsi"][i] <= 70,
        "rel_strength": rs20 >= 0,
        "room": rr >= 1.5,
        "risk_ok": risk / close <= 0.10,
    }
    if checks["uptrend"]: reasons_good.append("Stage 2 uptrend (above SMA200, SMA50 > SMA200)" if stage2 else "above rising SMA200")
    if checks["trigger"]: reasons_good.append("green close back above JMA & EMA8 after a pullback" + (" to EMA21/SMA50" if touched else "")
                                              + f", closed in top {100 - clv * 100:.0f}% of range")
    if checks["rel_strength"]: reasons_good.append(f"held up vs SPY (20d {ret20:+.1f}% vs SPY {spy_ret20:+.1f}%)")
    if volx >= 1.2 and green: reasons_good.append(f"volume {volx:.1f}x average on the up day")
    if x["macd"][i] > x["sig"][i] and x["macd"][i - 1] <= x["sig"][i - 1]: reasons_good.append("MACD just crossed up")
    if reasons_bad: verdict = "AVOID"
    elif all(checks.values()): verdict = "BUY TODAY"
    elif checks["uptrend"] and checks["not_extended"] and not checks["trigger"]: verdict = "WATCH"
    else: verdict = "PASS"
    missing = [k for k, ok in checks.items() if not ok]
    score = (25 * checks["uptrend"] + 25 * checks["trigger"] + 15 * checks["not_extended"] + min(15, max(0, rs20)) +
             min(10, rr * 3) + (5 if volx >= 1.2 and green else 0) + 3 * min(row.N_Scanners - 1, 3) + np.nan_to_num(row.Best_Score) / 20
             + (3 if row.In_Latest_List else 0) + min(4, row.Times_Listed - 1))
    return {"Ticker": t, "Verdict": verdict, "Score": round(score, 1), "Close": round(close, 2), "Stop": round(stop, 2),
            "Risk_%": round(risk / close * 100, 1), "Target": round(target, 2), "Reward_R": round(rr, 2),
            "Buy_Trigger": round(trig, 2) if verdict == "WATCH" else np.nan,
            "Why_Buy": "; ".join(reasons_good), "Why_Not": "; ".join(reasons_bad) or ("missing: " + ", ".join(missing) if missing else ""),
            "RSI": round(x["rsi"][i], 1), "Ext_ATR": round(ext, 2), "RS20_vs_SPY": round(rs20, 1), "Vol_x": round(volx, 2),
            "ATR_%": round(a / close * 100, 2), "Next_Earnings": earn.strftime("%Y-%m-%d") if earn is not None else "",
            "N_Scanners": row.N_Scanners, "Times_Listed": row.Times_Listed, "First_Seen": row.First_Seen.strftime("%Y-%m-%d"),
            "Last_Seen": row.Last_Seen.strftime("%Y-%m-%d"), "In_Latest_List": "Yes" if row.In_Latest_List else "No",
            "Chg_Since_First_%": round((close / row.Price_First - 1) * 100, 1) if np.isfinite(row.Price_First) and row.Price_First > 0 else np.nan,
            "Scanners": row.Scanners, "Best_Score": row.Best_Score,
            "Analyst_Upside": row.Analyst_Upside, "Headline": row.Headline, "Date": d.index[i].strftime("%Y-%m-%d")}


def charts(buys, data, path):
    n = len(buys)
    if not n: return
    cols = 3; rows = int(np.ceil(n / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(6 * cols, 3.6 * rows), squeeze=False)
    for ax, (_, r) in zip(axes.flat, buys.iterrows()):
        d = data[r.Ticker].iloc[-130:]; x = ind(data[r.Ticker]); k = len(d)
        idx = np.arange(k)
        for j in range(k):
            col = "#16a34a" if d.Close.iloc[j] >= d.Open.iloc[j] else "#dc2626"
            ax.vlines(j, d.Low.iloc[j], d.High.iloc[j], color=col, lw=0.7)
            ax.vlines(j, min(d.Open.iloc[j], d.Close.iloc[j]), max(d.Open.iloc[j], d.Close.iloc[j]), color=col, lw=2.6)
        for key, col, lab in (("jma", "#22c55e", "JMA"), ("e8", "#f97316", "EMA8"), ("e21", "#facc15", "EMA21"),
                              ("s50", "#2563eb", "SMA50"), ("s200", "#dc2626", "SMA200")):
            ax.plot(idx, x[key][-k:], color=col, lw=1, label=lab)
        ax.axhline(r.Stop, color="#991b1b", ls="--", lw=0.9); ax.axhline(r.Target, color="#15803d", ls="--", lw=0.9)
        ax.set_title(f"{r.Ticker}  {r.Close}  stop {r.Stop}  target {r.Target} ({r.Reward_R}R)", fontsize=9)
        ax.tick_params(labelsize=7); ax.set_xticks([])
    for ax in axes.flat[n:]: ax.axis("off")
    axes.flat[0].legend(fontsize=6, loc="upper left")
    fig.tight_layout(); fig.savefig(path, dpi=90); plt.close(fig)


def tv_list(o):
    if not (os.environ.get("TV_SESSIONID") and os.environ.get("TV_SESSIONID_SIGN")):
        return "skipped (TV_SESSIONID secrets not set)"
    from zoneinfo import ZoneInfo
    name = f"Chart Review - {datetime.now(ZoneInfo('Asia/Kolkata')):%Y-%m-%d}"
    os.environ["TV_WATCHLIST"] = name
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import tv_watchlist_sync as tvs
    tvs.NAME = name
    s = tvs.session(); lists = tvs.get_lists(s); ex = tvs.exchange_map()
    syms = []
    for v, lab in (("BUY TODAY", "Buy today"), ("WATCH", "Watch - buy above trigger")):
        t = [tvs.tv_symbol(x, ex) for x in o[o.Verdict == v].Ticker]
        if t: syms += [f"###{lab} ({len(t)})"] + t
    if not syms:
        return "skipped (no buy or watch stocks)"
    tvs.replace_list(s, lists, syms)
    return f"'{name}' - {sum(1 for x in syms if not x.startswith('###'))} symbols"


def email(o, run, spy_line, files, tv_status):
    import html, smtplib
    from email.mime.multipart import MIMEMultipart
    from email.mime.text import MIMEText
    from email.mime.base import MIMEBase
    from email import encoders
    gu = os.environ.get("GMAIL_USER") or os.environ.get("EMAIL_FROM", "")
    gp = os.environ.get("GMAIL_PASS") or os.environ.get("EMAIL_PASSWORD", "")
    et = os.environ.get("EMAIL_TO", "")
    if not (gu and gp and et):
        print("[Chart review email] secrets missing - skipped"); return
    e = lambda v: html.escape("" if v is None or (isinstance(v, float) and np.isnan(v)) else str(v))
    css = ("<style>body{font-family:Segoe UI,Arial;font-size:13px;color:#0f172a}table{border-collapse:collapse;margin:6px 0 14px}"
           "td,th{border:1px solid #e2e8f0;padding:4px 7px;font-size:12px;vertical-align:top}th{background:#1e293b;color:#fff}</style>")
    b = o[o.Verdict == "BUY TODAY"]; w = o[o.Verdict == "WATCH"]
    buy_rows = "".join(f"<tr><td><b>{e(r.Ticker)}</b></td><td>{e(r.Close)}</td><td>{e(r.Stop)} ({e(r['Risk_%'])}%)</td><td>{e(r.Target)} ({e(r.Reward_R)}R)</td>"
                       f"<td>{round(r.Close * 0.98, 2)}</td><td>{e(r.Why_Buy)}</td><td>{e(r.Scanners)}</td><td>{e(r.Headline)}</td>"
                       f"<td>{e(r.First_Seen)} ({e(r.Times_Listed)}x){' · new today' if r.In_Latest_List == 'Yes' and r.Times_Listed == 1 else ''}</td></tr>" for _, r in b.iterrows())
    watch_rows = "".join(f"<tr><td><b>{e(r.Ticker)}</b></td><td>{e(r.Close)}</td><td><b>{e(r.Buy_Trigger)}</b></td><td>{e(r.Stop)}</td><td>{e(r.RSI)}</td>"
                         f"<td>{e(r.RS20_vs_SPY)}</td><td>{e(r.Why_Not)}</td><td>{e(r.Next_Earnings)}</td><td>{e(r.First_Seen)}</td></tr>" for _, r in w.head(40).iterrows())
    avoid = o[o.Verdict == "AVOID"].Why_Not.str.split(";").explode().str.strip().str.replace(r" ?\(.*", "", regex=True).value_counts()
    body = (f"<html><head>{css}</head><body><h2>Chart review of all Best 10 watchlists · {datetime.today():%d %b %Y}</h2>"
            f"<p>{len(o)} stocks from every Best 10 list of the last {DAYS or 'all'} days (newest run {run}). {e(spy_line)}<br>Buy today <b>{len(b)}</b> · Watch {len(w)} · Pass {(o.Verdict == 'PASS').sum()} · "
            f"Avoid {(o.Verdict == 'AVOID').sum()} · TradingView: {e(tv_status)}</p>"
            f"<h3>Buy today ({len(b)})</h3>" + (f"<table><tr><th>Ticker</th><th>Close</th><th>Stop</th><th>Target</th><th>Skip if opens below</th>"
            f"<th>Why</th><th>Scanners</th><th>News</th><th>Listed</th></tr>{buy_rows}</table>" if len(b) else "<p>None today.</p>") +
            f"<h3>Watch: buy only on a daily close above the trigger ({len(w)}, top 40)</h3><table><tr><th>Ticker</th><th>Close</th><th>Buy above</th>"
            f"<th>Stop</th><th>RSI</th><th>vs SPY 20d</th><th>Missing</th><th>Earnings</th><th>First listed</th></tr>{watch_rows}</table>"
            f"<h3>Avoid reasons</h3><p>{'<br>'.join(f'{e(k)}: {v}' for k, v in avoid.items())}</p>"
            f"<p style='color:#64748b;font-size:11px'>Buy today = uptrend + green close back above JMA/EMA8 after a pullback (top half of the range) + not extended "
            f"(<= 2 ATR above EMA21, RSI 45-70) + beating SPY over 20 days + >= 1.5R to the 60-day high + stop <= 10%. Stop = at least 2 ATR below the close. "
            f"Indian (.NS) bars are the last NSE close. Full list and reasons in the CSV. Not financial advice.</p></body></html>")
    plain = [f"Chart review {datetime.today():%Y-%m-%d}: buy today {', '.join(b.Ticker) or 'none'}",
             "Watch: " + ", ".join(f"{t} > {x}" for t, x in zip(w.Ticker.head(40), w.Buy_Trigger.head(40)))]
    msg = MIMEMultipart("mixed")
    msg["Subject"] = f"Chart review (all Best 10 lists) · Buy today: {', '.join(b.Ticker.head(5)) or 'none'} · {len(w)} to watch · {datetime.today():%d %b}"
    msg["From"], msg["To"] = gu, et
    alt = MIMEMultipart("alternative"); alt.attach(MIMEText("\n".join(plain), "plain")); alt.attach(MIMEText(body, "html", "utf-8")); msg.attach(alt)
    for f in files:
        if os.path.exists(f):
            part = MIMEBase("application", "octet-stream"); part.set_payload(open(f, "rb").read())
            encoders.encode_base64(part); part.add_header("Content-Disposition", f"attachment; filename={os.path.basename(f)}"); msg.attach(part)
    with smtplib.SMTP_SSL("smtp.gmail.com", 465) as srv:
        srv.login(gu, gp.replace(" ", "")); srv.sendmail(gu, [x.strip() for x in et.split(",") if x.strip()], msg.as_string())
    print(f"[Chart review email] sent: {msg['Subject']}")


def main(argv):
    want = next((x for x in argv if x.isdigit()), "")
    run, g = load_best10(want)
    data = download(sorted(set(g.Ticker) | {"SPY"}))
    spy = data["SPY"].Close; spy_ret20 = (spy.iloc[-1] / spy.iloc[-21] - 1) * 100
    spy_line = (f"SPY {spy.iloc[-1]:.2f} ({spy.index[-1]:%Y-%m-%d}): 20d {spy_ret20:+.1f}%, vs SMA50 {(spy.iloc[-1] / spy.rolling(50).mean().iloc[-1] - 1) * 100:+.1f}%, "
                f"vs SMA200 {(spy.iloc[-1] / spy.rolling(200).mean().iloc[-1] - 1) * 100:+.1f}%")
    print(spy_line)
    out = {}
    for _, row in g.iterrows():
        if row.Ticker not in data: continue
        try:
            r = review(row.Ticker, data[row.Ticker], spy_ret20, row, None)
            if r: out[row.Ticker] = r
        except Exception as e:
            print(row.Ticker, e)
    # earnings dates only for the stocks that pass the chart checks (keeps the run fast as the lists grow)
    cand = [t for t, r in out.items() if r["Verdict"] in ("BUY TODAY", "WATCH")]
    with ThreadPoolExecutor(8) as ex:
        earn = dict(ex.map(next_earn, cand))
    rows = g.set_index("Ticker")
    for t in cand:
        try: out[t] = review(t, data[t], spy_ret20, rows.loc[t], earn.get(t)) or out[t]
        except Exception as e: print(t, e)
    print(f"earnings checked for {len(cand)} buy/watch candidates ({sum(1 for v in earn.values() if v is not None)} dates found)")
    o = pd.DataFrame(list(out.values()))
    o["rank"] = o.Verdict.map({"BUY TODAY": 0, "WATCH": 1, "PASS": 2, "AVOID": 3})
    o = o.sort_values(["rank", "Score"], ascending=[True, False]).drop(columns="rank")
    ds = datetime.today().strftime("%Y%m%d")
    o.to_csv(f"chart_review_{ds}.csv", index=False)
    print(o.Verdict.value_counts().to_string())
    pd.set_option("display.width", 250); pd.set_option("display.max_colwidth", 120)
    print(o[o.Verdict == "BUY TODAY"][["Ticker", "Score", "Close", "Stop", "Risk_%", "Target", "Reward_R", "RSI", "RS20_vs_SPY", "Vol_x", "Next_Earnings", "N_Scanners"]].to_string(index=False))
    for _, r in o[o.Verdict == "BUY TODAY"].iterrows():
        print(f"\n{r.Ticker}: {r.Why_Buy} | scanners: {r.Scanners} | {r.Headline}")
    print("\nWATCH (top 15):")
    print(o[o.Verdict == "WATCH"].head(15)[["Ticker", "Score", "Close", "Buy_Trigger", "Stop", "RSI", "RS20_vs_SPY", "Why_Not"]].to_string(index=False))
    files = [f"chart_review_{ds}.csv"]
    for v, f, k in (("BUY TODAY", f"chart_review_buys_{ds}.png", 12), ("WATCH", f"chart_review_watch_{ds}.png", 9)):
        try:
            charts(o[o.Verdict == v].head(k), data, f); files.append(f)
        except Exception as err:
            print(f"chart {f} skipped: {err}")
    tv = "not created"
    if "--no-tv" not in argv:
        try: tv = tv_list(o)
        except Exception as err: tv = f"failed ({type(err).__name__}: {err})"
    print(f"[TradingView] {tv}")
    if "--email" in argv:
        try: email(o, run, spy_line, files, tv)
        except Exception as err: print(f"[Chart review email] {type(err).__name__}: {err}")


if __name__ == "__main__":
    main(sys.argv[1:])
