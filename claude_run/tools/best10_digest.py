# Consolidated Best 10 email: one email with the Best 10 of every scanner in a
# ClaudeRunner results run (the best10_<scanner>_*.csv files that best_picks.py
# saves), plus the stocks found by two or more scanners and the top 20 scores.
#
# Runs automatically after the daily 06:00 IST run (called at the end of
# claude_run/tools/rsi_amount_filter.py --email by the RSI Digest workflow).
# On demand: put  claude_run/tools/best10_digest.py  in claude_run/run.txt
# after a run has committed its results; it uses the newest run.
#
# Usage: python claude_run/tools/best10_digest.py [run_id] [--no-email]
# Env:   RESULTS_RUN=<run_id> (else the newest run with best10 files)
#        GMAIL_USER / GMAIL_PASS / EMAIL_TO (or EMAIL_FROM / EMAIL_PASSWORD)
# Writes best10_all_scanners_<date>.csv in the current folder and attaches it.
import glob, html, os, re, sys
from datetime import datetime

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
ROOT = "claude_run/results"
NAMES = {
    "daily_price_jma_ema21_sma50_volume_v1": "Daily price above JMA/EMA21/SMA50 + volume",
    "daily_3candle_jma_reclaim_v1": "Daily 3-candle JMA reclaim",
    "daily_liquidity_sweep_breakout_v1": "Daily liquidity sweep breakout",
    "daily_rsi_bullish_divergence_v1": "Daily RSI bullish divergence",
    "daily_rsi_high_breakout_v1": "Daily RSI high breakout",
    "daily_3candle_higherlow_above_sma50_v1": "Daily 3-candle higher low above SMA50",
    "weekly_rsi_bullish_divergence_v1": "Weekly RSI bullish divergence",
    "weekly_reversal_2x_above_sma50_v1": "Weekly 2x reversal above SMA50",
    "monthly_reversal_2x_above_sma50_v1": "Monthly 2x reversal above SMA50",
    "monthly_rsi_cross_above_avg_v1": "Monthly RSI cross above average",
    "monthly_3candle_reclaim_jma_ema8_v1": "Monthly 3-candle reclaim JMA/EMA8",
    "india_weekly_reversal_2x_above_sma50_v1": "India (NSE) weekly 2x reversal above SMA50",
    "daily_ma_support_ema21_sma50_v1": "Daily MA support EMA21/SMA50",
    "daily_jma_cross_ema8_v1": "Daily JMA cross EMA8",
    "daily_sweep_sma50_strategy_v1": "Daily sweep of SMA50",
    "daily_rsi_avg_bounce_v1": "Daily RSI average bounce",
    "daily_rsi2_snapback_v1": "Daily RSI(2) snapback",
    "daily_ma_sweep_4x_v1": "Daily MA sweep 4x",
    "daily_jma_reclaim_rsi_v1": "Daily JMA reclaim + RSI",
    "daily_penny_ma_support_ema21_sma50_v1": "Penny: MA support EMA21/SMA50",
    "daily_penny_liquidity_sweep_breakout_v1": "Penny: liquidity sweep breakout",
    "daily_penny_jma_reclaim_rsi_v1": "Penny: JMA reclaim + RSI",
    "monthly_rsi_weekly_green_reversal_v1": "Monthly RSI 38-44 + weekly green reversal",
    "weekly_doji_jma_reclaim_v1": "Weekly doji JMA reclaim",
    "rsi_avg_turn_up_v1": "RSI average turning up",
    "daily_doji_hammer_jma_reclaim_v1": "Daily doji/hammer JMA reclaim",
    "daily_sma50_150_squeeze_turn_v1": "Daily SMA50/150 squeeze turn",
    "daily_rsi2_rs_1to1_v1": "RSI(2) leader pullback 1:1",
}


def nice(s):
    return NAMES.get(s, s.replace("_v1", "").replace("_", " "))


def schedule():
    try:
        return [l.strip()[:-3] for l in open("claude_run/schedule.txt") if l.strip().endswith(".py") and not l.startswith("#")]
    except OSError:
        return list(NAMES)


def pick_run(argv):
    want = next((a for a in argv if a.isdigit()), "") or os.environ.get("RESULTS_RUN", "").strip()
    if want and glob.glob(f"{ROOT}/{want}/*/best10_*.csv"):
        return want
    runs = [d for d in os.listdir(ROOT) if d.isdigit() and glob.glob(f"{ROOT}/{d}/*/best10_*.csv")] if os.path.isdir(ROOT) else []
    return max(runs, key=int) if runs else None


def load(run):
    frames, ran = [], []
    for d in sorted(glob.glob(f"{ROOT}/{run}/*/")):
        s = os.path.basename(d.rstrip("/"))
        ran.append(s)
        fs = sorted(glob.glob(f"{d}best10_*.csv"))
        if not fs:
            continue
        try:
            x = pd.read_csv(fs[-1])
        except Exception:
            continue
        if not len(x):
            continue
        full = [c for c in glob.glob(f"{d}*.csv") if not os.path.basename(c).startswith(("best10_", "tv_"))]
        try:
            q = len(pd.read_csv(full[0])) if full else None
        except Exception:
            q = None
        x.insert(0, "Scanner", s)
        x.insert(1, "Qualified", q)
        frames.append(x)
    return (pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()), ran


def e(v):
    return html.escape("" if v is None or (isinstance(v, float) and pd.isna(v)) else str(v))


def money(v, t):
    if v is None or pd.isna(v):
        return "—"
    return ("₹" if str(t).endswith((".NS", ".BO")) else "$") + f"{float(v):,.2f}"


def upside(r):
    u = r.get("Analyst_Upside_%")
    if u is None or pd.isna(u):
        return '<span class="nd">no data</span>'
    return f'<span class="{"neg" if u < 0 else ""}">{u:+.0f}%</span>'


def pe(r):
    v = r.get("Fwd_PE")
    return "—" if v is None or pd.isna(v) else f"{v:.1f}"


def score(v):
    return f'<span class="{"sc" if v >= 50 else "sl"}">{int(v)}</span>'


def short(t, n=95):
    t = "" if t is None or (isinstance(t, float) and pd.isna(t)) else str(t)
    return t if len(t) <= n else t[:n - 1].rstrip() + "…"


CSS = """
body{margin:0;background:#f5f7f6;font-family:Arial,Helvetica,sans-serif;color:#14201b}
.w{max-width:980px;margin:0 auto;padding:18px 12px}
h1{font-size:22px;margin:0 0 4px} h2{font-size:15px;margin:22px 0 6px}
.sub{color:#5b6b64;font-size:12px;margin:0}
.note{background:#fbf0e0;border-left:3px solid #a15c00;padding:8px 12px;font-size:12px;margin:12px 0}
table{border-collapse:collapse;width:100%;background:#fff;border:1px solid #dde4e0;font-size:12px}
th{text-align:left;color:#5b6b64;font-size:10px;text-transform:uppercase;padding:6px 8px;border-bottom:1px solid #dde4e0;white-space:nowrap}
td{padding:5px 8px;border-bottom:1px solid #eef2f0;vertical-align:top}
.n{font-family:Menlo,Consolas,monospace;white-space:nowrap}
.t{font-family:Menlo,Consolas,monospace;font-weight:bold}
.sc{background:#e3f2ea;color:#0f7a4f;font-weight:bold;padding:1px 6px;border-radius:4px;font-family:Menlo,Consolas,monospace}
.sl{color:#5b6b64;border:1px solid #dde4e0;padding:0 5px;border-radius:4px;font-family:Menlo,Consolas,monospace}
.nd{color:#a15c00;font-size:11px} .neg{color:#b4232c} .m{color:#5b6b64}
"""


def build(a, run, ran):
    date = datetime.today().strftime("%d %b %Y")
    sched = schedule()
    nsc = a.Scanner.nunique()
    nd = int(a["Analyst_Upside_%"].isna().sum()) if "Analyst_Upside_%" in a else len(a)
    by = a.groupby("Ticker")
    multi = (by.agg(n=("Scanner", "nunique"), best=("Best_Score", "max"))
               .query("n >= 2").sort_values(["n", "best"], ascending=False))
    top = a.sort_values(["Best_Score", "Trend_Score"], ascending=False).drop_duplicates("Ticker").head(20)
    missing = [s for s in sched if s in ran and s not in set(a.Scanner)]
    not_run = [s for s in sched if s not in ran]

    h = [f'<html><head><meta charset="utf-8"><style>{CSS}</style></head><body><div class="w">',
         f"<h1>🏅 Best 10 — all scanners</h1>",
         f'<p class="sub">{date} · run {e(run)} · {nsc} scanners · {len(a)} picks · {a.Ticker.nunique()} stocks · '
         f'{len(multi)} found by 2+ scanners</p>',
         f'<p class="note">Best score = Trend 35 + Catalyst 30 + Value 35. {nd} of {len(a)} picks have no analyst '
         f'upside / forward P/E from Yahoo ("no data") and may rank lower than they should. Headlines are '
         f'keyword-matched — read them before acting. Full list attached as CSV.</p>']

    if len(multi):
        h.append("<h2>Found by two or more scanners</h2><table><tr><th>Ticker</th><th>#</th><th>Best</th>"
                 "<th>Upside</th><th>Fwd P/E</th><th>Scanners (rank)</th></tr>")
        for t, m in multi.iterrows():
            l = a[a.Ticker == t]
            v = l.dropna(subset=["Analyst_Upside_%"]).head(1)
            v = (v if len(v) else l.head(1)).iloc[0]
            sc = "; ".join(f"{nice(s)} #{int(k)}" for s, k in zip(l.Scanner, l.Best_Rank))
            h.append(f'<tr><td class=t>{e(t)}</td><td class=n>{int(m.n)}</td><td>{score(m.best)}</td>'
                     f'<td class=n>{upside(v)}</td><td class=n>{pe(v)}</td><td class=m>{e(sc)}</td></tr>')
        h.append("</table>")

    h.append("<h2>Top 20 scores across all scanners</h2><table><tr><th>#</th><th>Ticker</th><th>Best</th>"
             "<th>T/C/V</th><th>Entry</th><th>Stop</th><th>Upside</th><th>Fwd P/E</th><th>Scanner</th>"
             "<th>Catalyst headline</th></tr>")
    for i, (_, r) in enumerate(top.iterrows(), 1):
        h.append(f'<tr><td class=n>{i}</td><td class=t>{e(r.Ticker)}</td><td>{score(r.Best_Score)}</td>'
                 f'<td class=n>{int(r.Trend_Score)}/{int(r.Catalyst_Score)}/{int(r.Value_Score)}</td>'
                 f'<td class=n>{money(r.get("Entry_Price", r.get("Price")), r.Ticker)}</td>'
                 f'<td class=n>{money(r.get("Stop_Loss"), r.Ticker)}</td><td class=n>{upside(r)}</td>'
                 f'<td class=n>{pe(r)}</td><td class=m>{e(nice(r.Scanner))}</td>'
                 f'<td>{e(short(r.get("Catalyst_Headline"))) or "—"}</td></tr>')
    h.append("</table>")

    order = [s for s in sched if s in set(a.Scanner)] + sorted(set(a.Scanner) - set(sched))
    for s in order:
        l = a[a.Scanner == s].sort_values("Best_Rank")
        q = l.Qualified.iloc[0]
        h.append(f'<h2>{e(nice(s))} <span class="sub">— {len(l)} of {"?" if pd.isna(q) else int(q)} qualified</span></h2>'
                 '<table><tr><th>#</th><th>Ticker</th><th>Best</th><th>T/C/V</th><th>Entry</th><th>Stop</th>'
                 '<th>Qty</th><th>Upside</th><th>Fwd P/E</th><th>Catalyst headline</th></tr>')
        for _, r in l.iterrows():
            qty = r.get("Quantity")
            h.append(f'<tr><td class=n>{int(r.Best_Rank)}</td><td class=t>{e(r.Ticker)}</td><td>{score(r.Best_Score)}</td>'
                     f'<td class=n>{int(r.Trend_Score)}/{int(r.Catalyst_Score)}/{int(r.Value_Score)}</td>'
                     f'<td class=n>{money(r.get("Entry_Price", r.get("Price")), r.Ticker)}</td>'
                     f'<td class=n>{money(r.get("Stop_Loss"), r.Ticker)}</td>'
                     f'<td class=n>{"—" if qty is None or pd.isna(qty) else int(qty)}</td>'
                     f'<td class=n>{upside(r)}</td><td class=n>{pe(r)}</td>'
                     f'<td>{e(short(r.get("Catalyst_Headline"), 60)) or "—"}</td></tr>')
        h.append("</table>")
    if missing:
        h.append(f'<p class="sub" style="margin-top:14px">No qualified stocks (or no Best 10 file): {e(", ".join(nice(s) for s in missing))}</p>')
    if not_run:
        h.append(f'<p class="sub">Not in this run: {e(", ".join(nice(s) for s in not_run))}</p>')
    h.append('<p class="sub" style="margin-top:14px">T/C/V = Trend / Catalyst / Value points. Sent by '
             'claude_run/tools/best10_digest.py.</p></div></body></html>')

    p = [f"BEST 10 — ALL SCANNERS  {date}  run {run}",
         f"{nsc} scanners, {len(a)} picks, {a.Ticker.nunique()} stocks; {nd} picks without analyst data", ""]
    if len(multi):
        p.append("FOUND BY 2+ SCANNERS")
        for t, m in multi.iterrows():
            p.append(f"  {t:<10} {int(m.n)} scanners  best {int(m.best)}")
        p.append("")
    p.append("TOP 20")
    for i, (_, r) in enumerate(top.iterrows(), 1):
        p.append(f"  #{i:<2} {r.Ticker:<10} {int(r.Best_Score):>3}  {nice(r.Scanner)}")
    for s in order:
        l = a[a.Scanner == s].sort_values("Best_Rank")
        p += ["", nice(s).upper()] + [f"  #{int(r.Best_Rank):<2} {r.Ticker:<10} {int(r.Best_Score):>3}" for _, r in l.iterrows()]

    lead = ", ".join(top.Ticker.head(3))
    subj = f"🏅 Best 10 — all scanners · {date} · {len(a)} picks · top: {lead}"
    return subj, "".join(h), "\n".join(p) + "\n"


def send(subj, body_html, plain, csv_path):
    import smtplib
    from email.mime.multipart import MIMEMultipart
    from email.mime.text import MIMEText
    from email.mime.base import MIMEBase
    from email import encoders
    gu = os.environ.get("GMAIL_USER") or os.environ.get("EMAIL_FROM", "")
    gp = os.environ.get("GMAIL_PASS") or os.environ.get("EMAIL_PASSWORD", "")
    et = os.environ.get("EMAIL_TO", "")
    if not (gu and gp and et):
        print("[Best 10 email] ❌  GMAIL_USER / GMAIL_PASS / EMAIL_TO not set — email skipped")
        return False
    eto = [x.strip() for x in et.split(",") if x.strip()]
    msg = MIMEMultipart("mixed")
    msg["Subject"], msg["From"], msg["To"] = subj, gu, ", ".join(eto)
    alt = MIMEMultipart("alternative")
    alt.attach(MIMEText(plain, "plain", "utf-8"))
    alt.attach(MIMEText(body_html, "html", "utf-8"))
    msg.attach(alt)
    with open(csv_path, "rb") as f:
        part = MIMEBase("application", "octet-stream"); part.set_payload(f.read())
    encoders.encode_base64(part)
    part.add_header("Content-Disposition", f"attachment; filename={os.path.basename(csv_path)}")
    msg.attach(part)
    try:
        with smtplib.SMTP_SSL("smtp.gmail.com", 465) as srv:
            srv.login(gu, gp.replace(" ", ""))
            srv.sendmail(gu, eto, msg.as_string())
        print(f"[Best 10 email] ✅  Sent to {', '.join(eto)} — {subj}")
        return True
    except smtplib.SMTPException as err:
        print(f"[Best 10 email] ❌  SMTP error: {err}")
        return False


def main(argv):
    run = pick_run(argv)
    if not run:
        print("[Best 10 email] no results run with best10 files — nothing to send")
        return 0
    a, ran = load(run)
    if a.empty:
        print(f"[Best 10 email] run {run} has no Best 10 rows — nothing to send")
        return 0
    csv_path = f"best10_all_scanners_{datetime.today().strftime('%Y%m%d')}.csv"
    a.to_csv(csv_path, index=False)
    subj, body_html, plain = build(a, run, ran)
    print(f"[Best 10 email] run {run}: {a.Scanner.nunique()} scanners, {len(a)} picks → {csv_path} "
          f"(html {len(body_html.encode()) // 1024} KB)")
    if "--no-email" not in argv:
        send(subj, body_html, plain, csv_path)
    else:
        open(csv_path.replace(".csv", ".html"), "w").write(body_html)
    # TradingView watchlist (only when the session secrets are configured)
    if os.environ.get("TV_SESSIONID") and "--no-tv" not in argv:
        try:
            import tv_watchlist_sync
            tv_watchlist_sync.main([str(run)])
        except Exception as err:
            print(f"[TradingView] not synced: {err}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
