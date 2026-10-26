# ============================================================
# NASDAQ — Monday intraday: 5-min close above a quiet red candle (v1)
# ============================================================
#
# Runs every 30 minutes on Mondays from 18:30 IST (13:00 UTC, before the US
# open) to the close (.github/workflows/monday_red_breakout.yml). The 18:30
# IST run emails the watchlist with trigger prices before the open, so a late
# GitHub start never misses the setup. Each run:
#
#   1. WATCHLIST: every ticker in the latest results of the MA screeners
#      (MA support, MA sweep + 4x, MA reclaim + RSI, and the penny MA
#      support / MA reclaim versions) — i.e. the latest "MA" emails.
#
#   2. QUIET RED CANDLE: keeps a ticker only if its last completed daily
#      candle before today (normally Friday's) was
#        - RED          close < open
#        - SMALL BODY   |close - open| <= 0.5 x ATR14            (default)
#        - LOW VOLUME   volume < its 20-day average volume       (default)
#      Trigger = that red candle's HIGH.
#
#   3. BREAKOUT: on today's 5-minute chart (regular session), the first
#      COMPLETED 5-min candle whose CLOSE is above the trigger.
#
# EMAIL: the first run of the day sends the watchlist with trigger prices
# (plus any breakouts already). Later runs email only when there are NEW
# breakouts. "Already emailed" is rebuilt from today's earlier outputs:
# claude_run/results/*/monday_red_candle_breakout_v1/monday_red_breakout_<date>_*.csv
# (runs via ClaudeRunner, started every 30 min on Mondays by a scheduled task)
# plus claude_run/intraday_state/ (runs via the GitHub workflow). Entry / Stop are defaults: Entry = the breakout 5-min
# close, Stop = the red candle's low; Quantity sized for a $40 max loss.
#
# Data: Yahoo 5-min bars are delayed ~15 min, and GitHub schedules can
# start late, so alerts are for awareness, not exact fills.
#
# Env: FORCE=1 runs outside Monday hours (uses the latest session as
# "today"); TEST=1 also marks the email [TEST] and does not save state;
# NOW_ET (ISO time, tests only) overrides the clock.
# ============================================================
import os, sys, json, glob, time, warnings
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo
import numpy as np, pandas as pd

warnings.filterwarnings("ignore")
ET, IST = ZoneInfo("America/New_York"), ZoneInfo("Asia/Kolkata")

CFG = {
    "sources": [   # scanner name -> label (latest results folder of each is used)
        ("daily_ma_support_ema21_sma50_v1",       "MA Support"),
        ("daily_ma_sweep_4x_v1",                  "MA Sweep 4x"),
        ("daily_jma_reclaim_rsi_v1",              "MA Reclaim RSI"),
        ("daily_penny_ma_support_ema21_sma50_v1", "PENNY MA Support"),
        ("daily_penny_jma_reclaim_rsi_v1",        "PENNY MA Reclaim RSI"),
    ],
    "max_body_atr"   : 0.5,    # small body: |close-open| <= this x ATR14
    "atr_period"     : 14,
    "vol_avg_days"   : 20,     # low volume: volume < average of the previous N days
    "max_loss_usd"   : 40.0,
    "results_dir"    : "claude_run/results",
    "state_dir"      : "claude_run/intraday_state",
}
FORCE = os.environ.get("FORCE", "") in ("1", "true", "True")
TEST = os.environ.get("TEST", "") in ("1", "true", "True")
FORCE = FORCE or TEST


# ── 1. Watchlist from the latest MA screener results ─────────────────────
def latest_csv(scanner):
    folders = [d for d in glob.glob(os.path.join(CFG["results_dir"], "*", scanner)) if os.path.isdir(d)]
    folders = [d for d in folders if os.path.basename(os.path.dirname(d)).isdigit()]
    if not folders:
        return None, None
    d = max(folders, key=lambda p: int(os.path.basename(os.path.dirname(p))))
    csvs = [f for f in glob.glob(os.path.join(d, "*.csv")) if not os.path.basename(f).startswith("tv_")]
    return (max(csvs) if csvs else None), d


def load_watchlist():
    wl = {}
    for scanner, label in CFG["sources"]:
        f, d = latest_csv(scanner)
        if not f:
            print(f"  ⚠️  no results for {scanner}"); continue
        try:
            df = pd.read_csv(f)
        except Exception as e:
            print(f"  ⚠️  {f}: {e}"); continue
        tick = [str(t).strip().upper() for t in df.get("Ticker", []) if str(t).strip()]
        print(f"  ✅ {label:<22} {len(tick):>3} tickers  ({os.path.relpath(f)})")
        for t in tick:
            wl.setdefault(t, []).append(label)
    return wl


# ── 2. Quiet red candle check (single source of truth) ──────────────────
def check_red_candle(d, cfg):
    """d = daily OHLCV up to and including the candle to test (last row).
    Computes every condition (no short-circuit). Returns (passed, details)."""
    if d is None or len(d) < max(cfg["atr_period"], cfg["vol_avg_days"]) + 2:
        return False, {}
    o, h, l, c, v = (d[x].values.astype(float) for x in ("Open", "High", "Low", "Close", "Volume"))
    pc = np.concatenate([[np.nan], c[:-1]])
    tr = np.nanmax(np.vstack([h - l, np.abs(h - pc), np.abs(l - pc)]), axis=0)
    atr = float(np.nanmean(tr[-cfg["atr_period"]:]))
    vavg = float(np.mean(v[-cfg["vol_avg_days"] - 1:-1]))
    body = abs(c[-1] - o[-1])
    red = bool(c[-1] < o[-1])
    small = bool(atr > 0 and body <= cfg["max_body_atr"] * atr)
    quiet = bool(vavg > 0 and v[-1] < vavg)
    return (red and small and quiet), {
        "date": d.index[-1].strftime("%Y-%m-%d"), "open": o[-1], "high": h[-1], "low": l[-1],
        "close": c[-1], "body_atr": body / atr if atr > 0 else np.nan,
        "vol_ratio": v[-1] / vavg if vavg > 0 else np.nan,
        "red": red, "small": small, "quiet": quiet,
    }


# ── 3. First completed 5-min close above the trigger ────────────────────
def first_breakout(m5, trigger, now_et):
    """m5 = today's regular-session 5-min bars (tz ET). Only bars that have
    finished (start + 5 min <= now) count. Returns (bar, last_close) or (None, last_close)."""
    if m5 is None or m5.empty:
        return None, np.nan
    done = m5[m5.index + pd.Timedelta(minutes=5) <= now_et]
    if done.empty:
        return None, np.nan
    hit = done[done["Close"] > trigger]
    return (hit.iloc[0:1] if len(hit) else None), float(done["Close"].iloc[-1])


def _frame(raw, t, single):
    try:
        df = raw if single else raw[t]
        df = df[["Open", "High", "Low", "Close", "Volume"]].dropna(subset=["Close"])
        return df if len(df) else None
    except Exception:
        return None


def download(tickers, **kw):
    import yfinance as yf
    out = {}
    for i in range(0, len(tickers), 50):
        b = tickers[i:i + 50]
        try:
            raw = yf.download(b, group_by="ticker", auto_adjust=False, actions=False,
                              threads=True, progress=False, **kw)
        except Exception as e:
            print("  download error", e); continue
        single = not isinstance(raw.columns, pd.MultiIndex)
        for t in b:
            f = _frame(raw, t, single and len(b) == 1)
            if f is not None:
                out[t] = f
        time.sleep(1)
    return out


def main():
    now_et = datetime.fromisoformat(os.environ["NOW_ET"]).replace(tzinfo=ET) if os.environ.get("NOW_ET") \
        else datetime.now(ET)
    print("━" * 65)
    print(f"  Monday red-candle 5-min breakout   {now_et:%Y-%m-%d %H:%M} ET / {now_et.astimezone(IST):%H:%M} IST"
          + ("   [FORCE]" if FORCE else ""))
    print("━" * 65)
    # window: from 07:30 ET (18:30 IST is 08:00 EST / 09:00 EDT, before the 09:30 open) to 16:05 ET
    start_t, close_t = now_et.replace(hour=7, minute=30, second=0, microsecond=0), now_et.replace(hour=16, minute=5, second=0, microsecond=0)
    if not FORCE and (now_et.weekday() != 0 or not (start_t <= now_et <= close_t)):
        print("  Not Monday 07:30-16:05 ET — nothing to do."); return
    if not FORCE and now_et < now_et.replace(hour=9, minute=30, second=0, microsecond=0):
        print("  Pre-market run: building the watchlist (no 5-min candles yet).")

    print("\n  STEP 1  WATCHLIST (latest MA screener results)")
    wl = load_watchlist()
    tickers = sorted(wl)
    print(f"  → {len(tickers)} unique tickers")
    if not tickers:
        return

    print("\n  STEP 2  QUIET RED CANDLE (last completed daily candle)")
    m5_all = download(tickers, period="5d", interval="5m", prepost=False)
    daily = download(tickers, period="3mo", interval="1d")
    # "today" = the latest session in the 5-min data (live: today; FORCE: last session)
    sess = sorted({ix.astimezone(ET).date() for df in m5_all.values() for ix in df.index})
    if FORCE and not sess:
        print("  ❌ no 5-min data"); return
    today = sess[-1] if FORCE else now_et.date()          # live: always the real date (pre-market: no bars yet)
    if FORCE:
        now_et = datetime.combine(today, datetime.min.time(), ET).replace(hour=16, minute=5)
    print(f"  Session checked: {today}  (5-min bars up to {now_et:%H:%M} ET)")

    funnel = {"watchlist": len(tickers), "red": 0, "small_body": 0, "low_volume": 0}
    cands = []
    for t in tickers:
        d = daily.get(t)
        if d is None:
            continue
        d = d[[ix.date() < today for ix in d.index]]          # last COMPLETED daily candle before today
        ok, x = check_red_candle(d, CFG)
        if not x:
            continue
        funnel["red"] += x["red"]
        funnel["small_body"] += x["red"] and x["small"]
        funnel["low_volume"] += ok
        if ok:
            cands.append((t, x))
    print(f"  Funnel: {funnel['watchlist']} watchlist → {funnel['red']} red → "
          f"{funnel['small_body']} small body → {funnel['low_volume']} low volume (= watch for breakout)")

    print("\n  STEP 3  5-MIN CLOSE ABOVE THE RED CANDLE HIGH")
    rows = []
    for t, x in cands:
        m5 = m5_all.get(t)
        if m5 is not None:
            m5 = m5.copy(); m5.index = m5.index.tz_convert(ET)
            m5 = m5[[ix.date() == today for ix in m5.index]]
        bar, last = first_breakout(m5, x["high"], now_et)
        r = {"Ticker": t, "Source": " + ".join(wl[t]), "Red_Candle_Date": x["date"],
             "Trigger_High": round(x["high"], 2), "Red_Low": round(x["low"], 2),
             "Body_ATR": round(x["body_atr"], 2), "Vol_vs_Avg": round(x["vol_ratio"], 2),
             "Last_Price": round(last, 2) if np.isfinite(last) else np.nan,
             "Broke_Out": bar is not None}
        if bar is not None:
            ts = bar.index[0]
            entry = float(bar["Close"].iloc[0])
            r.update({"Breakout_Time_ET": ts.strftime("%H:%M"), "Breakout_Time_IST": ts.astimezone(IST).strftime("%H:%M"),
                      "Breakout_Close": round(entry, 2), "Pct_Above_Trigger": round((entry / x["high"] - 1) * 100, 2),
                      "Entry_Price": round(entry, 2), "Stop_Loss": round(x["low"], 2)})
        rows.append(r)
    df = pd.DataFrame(rows)
    if df.empty:
        df = pd.DataFrame(columns=["Ticker", "Source", "Red_Candle_Date", "Trigger_High", "Red_Low", "Broke_Out"])
    hits = df[df["Broke_Out"] == True].copy()
    if len(hits):
        try:
            sys.path.insert(0, os.getcwd())
            from position_sizing import add_position_size
            hits = add_position_size(hits, max_loss_usd=CFG["max_loss_usd"])
        except Exception as e:
            print(f"  ⚠️  sizing skipped: {e}")
        hits = hits.sort_values("Breakout_Time_ET")
    for _, h in hits.iterrows():
        print(f"  🔔 {h.Ticker:<6} {h.Breakout_Time_ET} ET ({h.Breakout_Time_IST} IST)  close {h.Breakout_Close} > high {h.Trigger_High}")
    print(f"  Breakouts so far: {len(hits)} of {len(df)}")

    # ── state: which breakouts were already emailed today ──
    os.makedirs(CFG["state_dir"], exist_ok=True)
    spath = os.path.join(CFG["state_dir"], f"red_breakout_{today}.json")
    state = {"watchlist_sent": False, "alerted": []}
    if not TEST:
        if os.path.exists(spath):
            try: state = json.load(open(spath))
            except Exception: pass
        # earlier runs today (ClaudeRunner results or workflow state folder)
        day = today.strftime("%Y%m%d")
        prev = glob.glob(os.path.join(CFG["results_dir"], "*", "monday_red_candle_breakout_v1", f"monday_red_breakout_{day}_*.csv")) \
            + glob.glob(os.path.join(CFG["state_dir"], f"monday_red_breakout_{day}_*.csv"))
        for f in prev:
            try:
                p = pd.read_csv(f)
            except Exception:
                continue
            state["watchlist_sent"] = True
            if "Broke_Out" in p.columns:
                done = p[p["Broke_Out"].astype(str).str.lower() == "true"]["Ticker"].astype(str).tolist()
                state["alerted"] = sorted(set(state["alerted"]) | set(done))
        if prev:
            print(f"  Earlier runs today: {len(prev)} — already emailed: {', '.join(state['alerted']) or 'none'}")
    new = [t for t in hits["Ticker"] if t not in state["alerted"]] if len(hits) else []
    first = not state["watchlist_sent"]
    ts = now_et.strftime("%Y%m%d_%H%M")
    csv_path = f"monday_red_breakout_{ts}.csv"
    out = df.drop(columns=[c for c in ("Entry_Price", "Stop_Loss") if c in df.columns]).merge(
        hits[[c for c in ("Ticker", "Entry_Price", "Stop_Loss", "Quantity", "Total_Amount") if c in hits.columns]],
        on="Ticker", how="left") if len(hits) else df
    out.to_csv(csv_path, index=False)
    print(f"  💾 {csv_path}")

    if first or new:
        send_email(df, hits, new, first, funnel, today, now_et, csv_path)
    else:
        print("  No new breakouts since the last email — no email this run.")
    if not TEST:
        state["watchlist_sent"] = True
        state["alerted"] = sorted(set(state["alerted"]) | set(new))
        json.dump(state, open(spath, "w"), indent=1)
        print(f"  state → {spath}")


def send_email(df, hits, new, first, funnel, today, now_et, csv_path):
    import smtplib
    from email.mime.multipart import MIMEMultipart
    from email.mime.text import MIMEText
    from email.mime.base import MIMEBase
    from email import encoders
    gu, gp, et = os.environ.get("GMAIL_USER", ""), os.environ.get("GMAIL_PASS", ""), os.environ.get("EMAIL_TO", "")
    if not (gu and gp and et):
        print("[Email] ⚠️  GMAIL_USER / GMAIL_PASS / EMAIL_TO missing — email skipped"); return

    def td(v, s=""):
        return f'<td style="padding:6px 10px;font-size:12px;border-bottom:1px solid #e2e8f0;{s}">{v}</td>'
    def th(cols):
        return "".join(f'<th style="background:#1e293b;color:#e2e8f0;padding:7px 10px;font-size:11px;'
                       f'white-space:nowrap">{c}</th>' for c in cols)

    hrows = ""
    for _, h in hits.iterrows():
        isnew = h.Ticker in new
        q = h.get("Quantity", "—"); a = h.get("Total_Amount", None)
        hrows += ("<tr style='background:%s'>" % ("#fef9c3" if isnew else "#fff")
                  + td(("🆕 " if isnew else "") + h.Ticker, "font-weight:700")
                  + td(h.Source) + td(f"{h.Breakout_Time_ET} ET / {h.Breakout_Time_IST} IST", "font-weight:700")
                  + td(f"${h.Trigger_High:.2f}") + td(f"${h.Breakout_Close:.2f}", "color:#16a34a;font-weight:700")
                  + td(f"{h.Pct_Above_Trigger:+.2f}%") + td(f"${h.Last_Price:.2f}" if pd.notna(h.Last_Price) else "—")
                  + td(f"${h.Stop_Loss:.2f}", "color:#ef4444") + td(q)
                  + td(f"${float(a):,.0f}" if a is not None and pd.notna(a) else "—") + "</tr>")
    waiting = df[df["Broke_Out"] != True] if len(df) else df
    wrows = "".join("<tr>" + td(w.Ticker, "font-weight:700") + td(w.Source) + td(w.Red_Candle_Date)
                    + td(f"${w.Trigger_High:.2f}", "font-weight:700") + td(f"${w.Red_Low:.2f}")
                    + td(f"{w.Body_ATR:.2f}") + td(f"x{w.Vol_vs_Avg:.2f}")
                    + td(f"${w.Last_Price:.2f}" if pd.notna(w.Last_Price) else "—") + "</tr>"
                    for _, w in waiting.iterrows())
    tag = "[TEST] " if TEST else ""
    subj = (f"{tag}🔔 Monday 5-min breakout above red candle — {len(new)} new, {len(hits)} total"
            if new else f"{tag}🔔 Monday red-candle watchlist — {len(df)} stocks, {len(hits)} broke out")
    html = f"""<html><body style="font-family:'Segoe UI',Arial,sans-serif;background:#f1f5f9;padding:16px">
<div style="max-width:900px;margin:auto;background:#fff;border-radius:10px;overflow:hidden">
<div style="background:linear-gradient(135deg,#0f172a,#1e3a5f);padding:18px 24px">
<h2 style="margin:0;color:#60a5fa">🔔 Monday: 5-min close above the quiet red candle</h2>
<p style="margin:6px 0 0;color:#94a3b8;font-size:12px">Session {today} · checked {now_et:%H:%M} ET ({now_et.astimezone(IST):%H:%M} IST) ·
watchlist {funnel['watchlist']} → red {funnel['red']} → small body {funnel['small_body']} → low volume {funnel['low_volume']}</p></div>
<div style="padding:16px 20px">
{'<p style="margin:0 0 12px;font-size:12px;color:#b45309"><b>Pre-market watchlist</b> — the US market opens at 09:30 ET (' + now_et.replace(hour=9, minute=30).astimezone(IST).strftime('%H:%M') + ' IST). Breakouts will be emailed as they happen.</p>' if now_et < now_et.replace(hour=9, minute=30, second=0, microsecond=0) else ''}
<h3 style="margin:0 0 8px;font-size:14px">Breakouts ({len(hits)}{', ' + str(len(new)) + ' new' if new else ''})</h3>
<table style="border-collapse:collapse;width:100%"><tr>{th(['Ticker','From email','5-min candle','Red high','5-min close','Above','Last','Stop (red low)','Qty','Amount'])}</tr>
{hrows or '<tr><td colspan="10" style="padding:12px;color:#94a3b8;font-size:12px">No breakout yet</td></tr>'}</table>
<h3 style="margin:18px 0 8px;font-size:14px">Still waiting ({len(waiting)}) — trigger = red candle high</h3>
<table style="border-collapse:collapse;width:100%"><tr>{th(['Ticker','From email','Red candle','Trigger (high)','Red low','Body/ATR','Vol vs avg','Last'])}</tr>
{wrows or '<tr><td colspan="8" style="padding:12px;color:#94a3b8;font-size:12px">—</td></tr>'}</table>
<p style="font-size:11px;color:#64748b;margin-top:14px">Red candle = last completed daily candle before {today}: red, body ≤ {CFG['max_body_atr']} ATR14,
volume below its {CFG['vol_avg_days']}-day average. Breakout = first finished 5-min candle closing above its high. Entry = that 5-min close,
Stop = red candle low (defaults), Qty for a ${CFG['max_loss_usd']:.0f} max loss. Yahoo 5-min data is ~15 min delayed. Not financial advice.</p>
</div></div></body></html>"""
    plain = [subj, f"Session {today}, checked {now_et:%H:%M} ET", ""]
    for _, h in hits.iterrows():
        plain.append(f"{'NEW ' if h.Ticker in new else ''}{h.Ticker}: {h.Breakout_Time_ET} ET close {h.Breakout_Close} > red high {h.Trigger_High}, stop {h.Stop_Loss}, qty {h.get('Quantity','-')}")
    plain.append("")
    for _, w in waiting.iterrows():
        plain.append(f"waiting {w.Ticker}: trigger {w.Trigger_High} (red {w.Red_Candle_Date})")
    msg = MIMEMultipart("mixed"); msg["Subject"] = subj; msg["From"] = gu
    eto = [e.strip() for e in et.split(",") if e.strip()]; msg["To"] = ", ".join(eto)
    alt = MIMEMultipart("alternative"); alt.attach(MIMEText("\n".join(plain), "plain")); alt.attach(MIMEText(html, "html"))
    msg.attach(alt)
    try:
        part = MIMEBase("application", "octet-stream"); part.set_payload(open(csv_path, "rb").read())
        encoders.encode_base64(part); part.add_header("Content-Disposition", f"attachment; filename={os.path.basename(csv_path)}")
        msg.attach(part)
    except Exception as e:
        print(f"[Email] ⚠️  attach failed: {e}")
    try:
        with smtplib.SMTP_SSL("smtp.gmail.com", 465) as s:
            s.login(gu, gp.replace(" ", "")); s.sendmail(gu, eto, msg.as_string())
        print(f"[Email] ✅ sent: {subj}")
    except Exception as e:
        print(f"[Email] ❌ {type(e).__name__}: {e}")


if __name__ == "__main__":
    main()
