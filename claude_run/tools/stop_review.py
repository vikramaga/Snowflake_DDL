# Best 10 stop-loss review - runs with the 06:00 IST schedule; emails a report on Saturdays (and
# whenever it changes a rule, or on its first run).
#
# For every Best 10 pick saved in claude_run/results/ (= the daily "Best 10 - YYYY-MM-DD"
# TradingView watchlists), first appearance per scanner within 10 days:
#   * replays the trade the way the backtest does: buy at the next session's open, exit at the
#     stop or 2R target, whichever comes first, or at the close after 20 sessions;
#   * with the scanner's own stop and with the stop floored at 1.0 / 1.5 / 2.0 / 2.5 ATR(14);
#   * with and without the "skip if the next open is more than X% below the signal close" rule;
#   * labels each stop-out: gap through the stop, stop inside normal noise (< 1 ATR, hit within
#     3 sessions), next open > 2% lower, market (SPY) fell > 3% during the trade, gave back a
#     1R+ gain, other.
# ONGOING ADJUSTMENT (guard-railed): the minimum stop distance used by best_picks.py moves one
# step (0.5 ATR) toward the floor that did best on the live picks of the last 90 days, only when
# >= 150 live trades with 5+ sessions are available, the gain is >= 0.03R per trade, and the new value was
# also positive in the 1-year backtest (+0.06R or better). The gap rule is switched off only if
# live gap-down opens did >= 0.15R better than the rest over >= 40 trades, and back on if they
# then do >= 0.15R worse. Earnings skip stays at 5 business days (live earnings dates are not
# reliable enough to tune it).
# Writes stop_rules_YYYYMMDD.csv (read by best_picks.py from the newest run) and
# stop_review_YYYYMMDD.html; picks with outcomes are in the email only.
import glob, os, re, subprocess, sys, time
from datetime import datetime, timedelta, timezone

subprocess.call([sys.executable, "-m", "pip", "install", "-q", "yfinance", "pandas", "numpy"],
                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
import numpy as np, pandas as pd, yfinance as yf

ROOT = "claude_run/results"
TS = re.compile(r"_(20\d{6})_(\d{4})\.csv$")
HOLD, DEDUPE_DAYS, LOOKBACK_DAYS = 20, 10, 90
FLOORS = [0.0, 1.0, 1.5, 2.0, 2.5]
BACKTEST_R = {0.0: 0.050, 1.0: 0.061, 1.5: 0.072, 2.0: 0.073, 2.5: 0.069}   # 1-year backtest, 77k trades
DEFAULTS = {"stop_floor_atr": 1.5, "earnings_skip_days": 5, "gap_down_skip_pct": 2.0}
MIN_TRADES, MIN_GAIN, GAP_MIN_N, GAP_DIFF = 150, 0.03, 40, 0.15
LIMIT_SCANNERS = {"daily_rsi2_snapback_v1", "daily_rsi2_rs_1to1_v1"}
IST = timezone(timedelta(hours=5, minutes=30))
TODAY = datetime.now(IST)


def current_rules():
    rules, src = dict(DEFAULTS), "defaults"
    fs = glob.glob(f"{ROOT}/*/stop_review/stop_rules_*.csv")
    if fs:
        f = max(fs, key=lambda x: (os.path.basename(x), x))
        try:
            for _, r in pd.read_csv(f).iterrows():
                if r["rule"] in rules: rules[r["rule"]] = float(r["value"])
            src = f
        except Exception:
            pass
    return rules, src, bool(fs)


def picks():
    rows = []
    for rd in sorted(glob.glob(f"{ROOT}/[0-9]*/")):
        subs = [s for s in glob.glob(rd + "*/") if glob.glob(s + "best10_*.csv")]
        if len(subs) < 10: continue
        for sd in subs:
            scn = os.path.basename(sd.rstrip("/"))
            for f in glob.glob(sd + "best10_*.csv"):
                m = TS.search(f)
                if not m or "all_scanners" in f: continue
                try: x = pd.read_csv(f)
                except Exception: continue
                x["Scanner"] = scn
                x["File_TS"] = pd.Timestamp(datetime.strptime(m.group(1) + m.group(2), "%Y%m%d%H%M"))
                rows.append(x)
    a = pd.concat(rows, ignore_index=True)
    a["Ticker"] = a.Ticker.astype(str).str.upper()
    if "Scanner_Stop" not in a: a["Scanner_Stop"] = np.nan
    a["Stop0"] = pd.to_numeric(a.Scanner_Stop, errors="coerce").fillna(pd.to_numeric(a.Stop_Loss, errors="coerce"))
    a["Entry_Price"] = pd.to_numeric(a.get("Entry_Price", a.get("Price")), errors="coerce")
    return a[a.Stop0.gt(0) & a.Entry_Price.gt(0)]


def download(tickers, start):
    out = {}
    for i in range(0, len(tickers), 80):
        b = tickers[i:i + 80]
        try:
            raw = yf.download(b, start=start, group_by="ticker", auto_adjust=True, actions=False, threads=True, progress=False)
        except Exception:
            continue
        for t in b:
            try:
                d = raw[t] if isinstance(raw.columns, pd.MultiIndex) else raw
                d = d.dropna(subset=["Close"]); d.index = pd.to_datetime(d.index).tz_localize(None).normalize()
                if len(d) > 20: out[t] = d[["Open", "High", "Low", "Close"]].astype(float)
            except Exception:
                pass
    return out


def simulate(o, h, l, c, e, stop, entry):
    """From bar e (entry at its open). Returns (R, reason, bars) or None if still open with < 1 bar."""
    risk = entry - stop
    tgt = entry + 2 * risk
    last = min(len(o) - 1, e + HOLD - 1)
    for j in range(e, last + 1):
        if j > e and o[j] <= stop: return (o[j] - entry) / risk, "stop (gap)", j - e + 1
        if j > e and o[j] >= tgt:  return (o[j] - entry) / risk, "target", j - e + 1
        if l[j] <= stop: return -1.0, "stop", j - e + 1
        if h[j] >= tgt:  return 2.0, "target", j - e + 1
    return (c[last] - entry) / risk, ("time" if last == e + HOLD - 1 else "open"), last - e + 1


def main():
    rules, src, had = current_rules()
    a = picks()
    data = download(sorted(set(a.Ticker) | {"SPY"}), (TODAY - timedelta(days=LOOKBACK_DAYS + 80)).strftime("%Y-%m-%d"))
    spy = data.get("SPY")
    print(f"{len(a)} Best 10 rows, price data for {len(data)} tickers; current rules {rules} from {src}")
    rows = []
    for _, p in a.iterrows():
        d = data.get(p.Ticker)
        if d is None: continue
        idx = d.index
        k = int(idx.searchsorted((p.File_TS - pd.Timedelta(hours=21)).normalize(), side="right")) - 1
        if k < 15 or k >= len(idx) - 1: continue
        o, h, l, c = (d[x].values for x in ("Open", "High", "Low", "Close"))
        pc = np.r_[np.nan, c[:-1]]
        atr = pd.Series(np.nanmax(np.vstack([h - l, np.abs(h - pc), np.abs(l - pc)]), axis=0)).rolling(14).mean().values[k]
        e = k + 1
        lim = p.Scanner in LIMIT_SCANNERS and p.Entry_Price < c[k] * 0.995
        if lim:
            if l[e] > p.Entry_Price: continue          # limit order not filled
            entry = min(o[e], p.Entry_Price)
        else:
            entry = o[e]
        if not (np.isfinite(atr) and entry > p.Stop0): continue
        r = {"Scanner": p.Scanner, "Ticker": p.Ticker, "Signal": idx[k], "risk_atr": (entry - p.Stop0) / atr,
             "open_gap_pct": (o[e] / c[k] - 1) * 100, "sessions": len(idx) - e}
        for f in FLOORS:
            stop = min(p.Stop0, entry - f * atr) if f else p.Stop0
            R, why, bars = simulate(o, h, l, c, e, stop, entry)
            r[f"R_{f}"], r[f"why_{f}"], r[f"bars_{f}"] = R, why, bars
        # scenario for the stop that applies today
        fc = rules["stop_floor_atr"]; why, bars = r[f"why_{fc}"], r[f"bars_{fc}"]
        if why.startswith("stop"):
            stop = min(p.Stop0, entry - fc * atr) if fc else p.Stop0
            end = idx[min(e + bars - 1, len(idx) - 1)]
            spy_ret = (spy.Close.get(end, np.nan) / spy.Close.get(idx[k], np.nan) - 1) * 100 if spy is not None else np.nan
            mfe = (h[e:e + bars].max() - entry) / (entry - stop)
            r["scenario"] = ("Gapped through the stop" if why == "stop (gap)" else
                             "Stop inside normal noise (<1 ATR), hit within 3 sessions" if (entry - stop) / atr < 1 and bars <= 3 else
                             "Next open >2% below signal close" if r["open_gap_pct"] < -2 else
                             "Market (SPY) fell >3% during the trade" if spy_ret < -3 else
                             "Was up 1R+ first, then reversed" if mfe >= 1 else "Weak follow-through")
        rows.append(r)
    t = pd.DataFrame(rows).sort_values("Signal")
    if "scenario" not in t: t["scenario"] = np.nan
    t = t[~t.duplicated(["Scanner", "Ticker"]) | (t.groupby(["Scanner", "Ticker"]).Signal.diff() > pd.Timedelta(days=DEDUPE_DAYS))]
    # scored = at least 5 sessions since the entry; trades still open are valued at the latest close
    closed = t[(t.sessions >= 5) & (t.Signal >= pd.Timestamp(TODAY.date()) - pd.Timedelta(days=LOOKBACK_DAYS))]
    print(f"{len(t)} trades, {len(closed)} scored (5+ sessions) in the last {LOOKBACK_DAYS} days")

    # ---- rule decisions
    new, notes = dict(rules), []
    gap = rules["gap_down_skip_pct"] or DEFAULTS["gap_down_skip_pct"]
    keep = closed[~(closed.open_gap_pct < -gap)] if rules["gap_down_skip_pct"] else closed
    avg = {f: keep[f"R_{f}"].clip(-10, 10).mean() for f in FLOORS}
    cur = rules["stop_floor_atr"]
    best = max(avg, key=lambda f: avg[f] if np.isfinite(avg[f]) else -9)
    if len(keep) >= MIN_TRADES and best != cur and avg[best] - avg.get(cur, np.nan) >= MIN_GAIN and BACKTEST_R.get(best, 0) >= 0.06:
        step = cur + 0.5 * np.sign(best - cur)
        new["stop_floor_atr"] = float(min(max(step, 1.0), 2.5))
        notes.append(f"Minimum stop moved {cur:g} → {new['stop_floor_atr']:g} ATR: on {len(keep)} live trades the "
                     f"{best:g} ATR floor averaged {avg[best]:+.3f}R vs {avg[cur]:+.3f}R for {cur:g} ATR (one 0.5 step at a time).")
    else:
        why = (f"only {len(keep)} live trades (5+ sessions) (need {MIN_TRADES})" if len(keep) < MIN_TRADES else
               f"the best live floor ({best:g} ATR, {avg[best]:+.3f}R) does not beat {cur:g} ATR by {MIN_GAIN}R" if best != cur else
               f"{cur:g} ATR is still the best live floor")
        notes.append(f"Minimum stop kept at {cur:g} ATR: {why}.")
    cf = f"R_{new['stop_floor_atr']}"
    gd = closed[closed.open_gap_pct < -gap][cf].clip(-10, 10); rest = closed[~(closed.open_gap_pct < -gap)][cf].clip(-10, 10)
    if len(gd) >= GAP_MIN_N and rules["gap_down_skip_pct"] and gd.mean() - rest.mean() >= GAP_DIFF:
        new["gap_down_skip_pct"] = 0.0
        notes.append(f"Gap-down skip switched OFF: {len(gd)} live gap-down opens averaged {gd.mean():+.3f}R vs {rest.mean():+.3f}R.")
    elif len(gd) >= GAP_MIN_N and not rules["gap_down_skip_pct"] and rest.mean() - gd.mean() >= GAP_DIFF:
        new["gap_down_skip_pct"] = DEFAULTS["gap_down_skip_pct"]
        notes.append(f"Gap-down skip switched back ON: gap-down opens {gd.mean():+.3f}R vs {rest.mean():+.3f}R.")
    else:
        notes.append(f"Gap-down skip {'kept at ' + format(rules['gap_down_skip_pct'], 'g') + '%' if rules['gap_down_skip_pct'] else 'kept off'}: "
                     f"{len(gd)} live gap-down opens ({gd.mean():+.3f}R) vs the rest ({rest.mean():+.3f}R)." if len(gd) else
                     "Gap-down skip kept: no live gap-down opens yet.")
    changed = new != rules
    ds = TODAY.strftime("%Y%m%d")
    pd.DataFrame([{"rule": k, "value": v, "updated": TODAY.strftime("%Y-%m-%d"), "note": " ".join(notes)[:400]}
                  for k, v in new.items()]).to_csv(f"stop_rules_{ds}.csv", index=False)

    # ---- report
    def pct(x): return f"{x:.0%}" if np.isfinite(x) else "—"
    fl = new["stop_floor_atr"]
    st = closed[f"why_{fl}"].str.startswith("stop"); st0 = closed["why_0.0"].str.startswith("stop")
    scen = closed[st].scenario.value_counts()
    fl_rows = "".join(f"<tr><td>{'scanner stop' if f == 0 else f'{f:g} ATR'}</td><td>{closed[f'R_{f}'].clip(-10, 10).mean():+.3f}R</td>"
                      f"<td>{pct(closed[f'why_{f}'].str.startswith('stop').mean())}</td><td>{BACKTEST_R[f]:+.3f}R</td></tr>" for f in FLOORS)
    by = closed.groupby("Scanner").agg(n=("Ticker", "size"), stop0=("why_0.0", lambda x: x.str.startswith("stop").mean()),
                                       stopf=(f"why_{fl}", lambda x: x.str.startswith("stop").mean()),
                                       risk=("risk_atr", "median"), R=(f"R_{fl}", "mean")).query("n >= 8").sort_values("stopf", ascending=False)
    sc_rows = "".join(f"<tr><td>{s}</td><td>{r.n}</td><td>{r.risk:.1f}</td><td>{pct(r.stop0)}</td><td>{pct(r.stopf)}</td><td>{r.R:+.2f}R</td></tr>"
                      for s, r in by.iterrows())
    sn_rows = "".join(f"<tr><td>{k}</td><td>{v}</td><td>{v / st.sum():.0%}</td></tr>" for k, v in scen.items())
    css = ("<style>body{font-family:Segoe UI,Arial;font-size:13px;color:#0f172a}table{border-collapse:collapse;margin:6px 0 14px}"
           "td,th{border:1px solid #e2e8f0;padding:4px 8px;font-size:12px}th{background:#1e293b;color:#fff}</style>")
    html = (f"<html><head>{css}</head><body><h2>Best 10 stop-loss review · {TODAY:%d %b %Y}</h2>"
            f"<p>{len(closed)} live Best 10 trades with 5+ sessions since entry (signals in the last {LOOKBACK_DAYS} days, first appearance "
            f"per scanner), bought at the next open, 2R target, 20-session limit; trades still open are valued at the latest close. Stop hit: <b>{pct(st0.mean())}</b> with the scanners' own stops, "
            f"<b>{pct(st.mean())}</b> with the {fl:g} ATR minimum.</p>"
            f"<h3>Rules for the next runs</h3><p>{'<br>'.join(notes)}</p>"
            f"<p>Stop at least <b>{fl:g} ATR</b> below entry · no earnings within <b>{int(new['earnings_skip_days'])}</b> business days · "
            f"skip if the next open is more than <b>{new['gap_down_skip_pct']:g}%</b> below the signal close"
            f"{' (off)' if not new['gap_down_skip_pct'] else ''}.{' <b>Changed this week.</b>' if changed else ''}</p>"
            f"<h3>Stop distance</h3><table><tr><th>Stop</th><th>Live avg</th><th>Live stop hit</th><th>1-yr backtest avg</th></tr>{fl_rows}</table>"
            f"<h3>Why stops were hit ({st.sum()} trades)</h3><table><tr><th>Scenario</th><th>Trades</th><th>Share</th></tr>{sn_rows}</table>"
            f"<h3>By scanner (8+ trades; highest stop rate first)</h3><table><tr><th>Scanner</th><th>Trades</th><th>Own stop, ATR</th>"
            f"<th>Stop hit (own)</th><th>Stop hit ({fl:g} ATR)</th><th>Avg</th></tr>{sc_rows}</table>"
            f"<p style='color:#64748b;font-size:11px'>Rules file: stop_rules_{ds}.csv (best_picks.py reads the newest). Guard rails: the minimum stop "
            f"moves 0.5 ATR at a time, needs {MIN_TRADES}+ live trades (5+ sessions) and a {MIN_GAIN}R gain that the backtest also supports. "
            f"Not financial advice.</p></body></html>")
    open(f"stop_review_{ds}.html", "w").write(html)
    print("\n".join(notes))
    send = TODAY.weekday() == 5 or changed or not had
    if send:
        email(html, changed, len(closed), st.mean(), t)


def email(html, changed, n, rate, t):
    import smtplib
    from email.mime.multipart import MIMEMultipart
    from email.mime.text import MIMEText
    from email.mime.base import MIMEBase
    from email import encoders
    gu, gp, et = (os.environ.get(k, "") for k in ("GMAIL_USER", "GMAIL_PASS", "EMAIL_TO"))
    if not (gu and gp and et):
        print("[Email] secrets missing - skipped"); return
    msg = MIMEMultipart("mixed")
    msg["Subject"] = f"Best 10 stop-loss review · {n} trades · stop hit {rate:.0%}{' · RULE CHANGED' if changed else ''} · {TODAY:%d %b}"
    msg["From"], msg["To"] = gu, et
    msg.attach(MIMEText(html, "html", "utf-8"))
    part = MIMEBase("application", "octet-stream"); part.set_payload(t.round(3).to_csv(index=False).encode())
    encoders.encode_base64(part); part.add_header("Content-Disposition", f"attachment; filename=stop_review_trades_{TODAY:%Y%m%d}.csv")
    msg.attach(part)
    with smtplib.SMTP_SSL("smtp.gmail.com", 465) as s:
        s.login(gu, gp.replace(" ", "")); s.sendmail(gu, [x.strip() for x in et.split(",") if x.strip()], msg.as_string())
    print(f"[Email] sent: {msg['Subject']}")


if __name__ == "__main__":
    main()
