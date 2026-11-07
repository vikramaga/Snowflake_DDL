# Replace a TradingView watchlist with the day's Best 10 picks.
#
# Called at the end of claude_run/tools/best10_digest.py (so it runs after the
# daily 06:00 IST run) when the TradingView session secrets are present.
# Standalone: python claude_run/tools/tv_watchlist_sync.py [run_id] [--probe]
#   --probe  only signs in and lists your watchlists (changes nothing)
#
# Env: TV_SESSIONID, TV_SESSIONID_SIGN  - the "sessionid" and "sessionid_sign"
#        cookies of tradingview.com from a signed-in browser (repo secrets)
#      TV_WATCHLIST  - watchlist name (default "Best 10 - all scanners")
#
# TradingView has no public watchlist API; this uses the same endpoints as the
# TradingView website (/api/v1/symbols_list/custom/). If they change, or the
# cookies expire, the sync fails, says why in the log and emails a short alert;
# nothing else in the pipeline is affected. The cookies are never printed.
#
# List layout: "###" lines are TradingView section headers.
#   ### Found by 2+ scanners   (most scanners first)
#   ### Top 20 scores
#   ### <each scanner>         (its Best 10, minus symbols already listed above)
import glob, json, os, sys
from datetime import datetime

import pandas as pd
import requests

BASE = "https://www.tradingview.com"
API = f"{BASE}/api/v1/symbols_list/custom/"
NAME = os.environ.get("TV_WATCHLIST", "Best 10 - all scanners")
UA = ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
      "(KHTML, like Gecko) Chrome/124.0 Safari/537.36")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


class SyncError(Exception):
    pass


def session():
    sid, sign = os.environ.get("TV_SESSIONID", "").strip(), os.environ.get("TV_SESSIONID_SIGN", "").strip()
    if not sid or not sign:
        raise SyncError("TV_SESSIONID / TV_SESSIONID_SIGN are not set")
    s = requests.Session()
    for k, v in (("sessionid", sid), ("sessionid_sign", sign)):
        s.cookies.set(k, v, domain=".tradingview.com", path="/")
    s.headers.update({"User-Agent": UA, "Origin": BASE, "Referer": f"{BASE}/chart/",
                      "Accept": "application/json", "X-Requested-With": "XMLHttpRequest"})
    return s


def _check(r, what):
    if r.status_code in (401, 403):
        raise SyncError(f"{what}: HTTP {r.status_code} - TradingView refused the session; refresh the cookies")
    if r.status_code >= 400:
        raise SyncError(f"{what}: HTTP {r.status_code} {r.text[:200]!r}")
    try:
        return r.json()
    except ValueError:
        raise SyncError(f"{what}: not JSON (HTTP {r.status_code}) {r.text[:120]!r}")


def get_lists(s):
    data = _check(s.get(API, timeout=20), "read watchlists")
    if isinstance(data, dict):
        data = data.get("results") or data.get("lists") or [data]
    lists = [x for x in data if isinstance(x, dict)]
    if not lists or all(x.get("id") is None for x in lists):
        raise SyncError("TradingView answered as signed out - the session cookies have expired; refresh them")
    return lists


def exchange_map():
    """Ticker -> TradingView exchange for NYSE / AMEX stocks (everything else is NASDAQ)."""
    out = {}
    for ex, tv in (("nyse", "NYSE"), ("amex", "AMEX")):
        try:
            r = requests.get("https://api.nasdaq.com/api/screener/stocks"
                             f"?tableonly=true&limit=10000&exchange={ex}&download=true",
                             headers={"User-Agent": UA, "Accept": "application/json"}, timeout=30)
            for row in r.json()["data"]["rows"]:
                out[str(row["symbol"]).strip().upper()] = tv
        except Exception as err:
            print(f"[TradingView] exchange list {ex} unavailable ({err}); those tickers default to NASDAQ")
    return out


def tv_symbol(t, ex):
    t = str(t).strip().upper()
    if t.endswith(".NS"):
        return "NSE:" + t[:-3]
    if t.endswith(".BO"):
        return "BSE:" + t[:-3]
    base = t.replace("-", ".").replace("/", ".")
    return f"{ex.get(t, ex.get(base, 'NASDAQ'))}:{base}"


def build_symbols(a, ex, names):
    out, seen = [], set()

    def add(section, tickers):
        syms = [tv_symbol(t, ex) for t in tickers]
        syms = [x for x in syms if x not in seen]
        if syms:
            out.append("###" + section)
            out.extend(syms)
            seen.update(syms)

    multi = (a.groupby("Ticker").agg(n=("Scanner", "nunique"), best=("Best_Score", "max"))
              .query("n >= 2").sort_values(["n", "best"], ascending=False))
    add("Found by 2+ scanners", multi.index)
    top = a.sort_values(["Best_Score", "Trend_Score"], ascending=False).drop_duplicates("Ticker").head(20)
    add("Top 20 scores", top.Ticker)
    for scn in names:
        l = a[a.Scanner == scn].sort_values("Best_Rank")
        add(names[scn], l.Ticker)
    return out


def replace_list(s, lists, symbols):
    s.headers["Content-Type"] = "application/json"
    mine = next((x for x in lists if x.get("name") == NAME), None)
    if mine is None:
        r = s.post(API, data=json.dumps({"name": NAME, "symbols": symbols}), timeout=30)
        created = _check(r, "create watchlist")
        print(f"[TradingView] created watchlist '{NAME}' (id {created.get('id') if isinstance(created, dict) else '?'})")
        return
    lid = mine["id"]
    last = None
    for url in (f"{API}{lid}/replace/?unsafe=true", f"{API}{lid}/replace/"):
        r = s.post(url, data=json.dumps(symbols), timeout=30)
        if r.status_code < 400:
            print(f"[TradingView] replaced watchlist '{NAME}' (id {lid}) via {url.split(str(lid))[1]}")
            return
        last = r
    _check(last, "replace watchlist")


def sync(run=None, probe=False):
    s = session()
    lists = get_lists(s)
    print(f"[TradingView] signed in; {len(lists)} watchlist(s): "
          + ", ".join(f"{x.get('name')!r} ({len(x.get('symbols') or [])})" for x in lists))
    if probe:
        return True
    import best10_digest as bd
    run = run or bd.pick_run([])
    if not run:
        raise SyncError("no results run with Best 10 files")
    a, _ = bd.load(run)
    if a.empty:
        raise SyncError(f"run {run} has no Best 10 rows")
    order = [x for x in bd.schedule() if x in set(a.Scanner)] + sorted(set(a.Scanner) - set(bd.schedule()))
    symbols = build_symbols(a, exchange_map(), {x: bd.nice(x) for x in order})
    n = sum(1 for x in symbols if not x.startswith("###"))
    print(f"[TradingView] {n} symbols in {sum(1 for x in symbols if x.startswith('###'))} sections from run {run}")
    replace_list(s, lists, symbols)
    after = next((x for x in get_lists(s) if x.get("name") == NAME), None)
    got = [x for x in (after or {}).get("symbols") or [] if not str(x).startswith("###")]
    if after is None or not got:
        raise SyncError("TradingView accepted the request but the watchlist is still empty - endpoint may have changed")
    if len(got) < n:
        print(f"[TradingView] ⚠️  {len(got)} of {n} symbols kept (plan limit or unknown symbols)")
    print(f"[TradingView] ✅  '{NAME}' now has {len(got)} symbols")
    return True


def alert(reason):
    """Short email so an expired cookie doesn't go unnoticed."""
    import smtplib
    from email.mime.text import MIMEText
    gu = os.environ.get("GMAIL_USER") or os.environ.get("EMAIL_FROM", "")
    gp = os.environ.get("GMAIL_PASS") or os.environ.get("EMAIL_PASSWORD", "")
    et = os.environ.get("EMAIL_TO", "")
    if not (gu and gp and et):
        return
    body = (f"The TradingView watchlist '{NAME}' was not updated today.\n\nReason: {reason}\n\n"
            "If the session expired: sign in to tradingview.com in your browser, copy the cookies "
            "'sessionid' and 'sessionid_sign' (Developer tools > Application > Cookies) and paste them "
            "into the GitHub repo secrets TV_SESSIONID and TV_SESSIONID_SIGN.\n\n"
            "The Best 10 email and the scanners are not affected.")
    msg = MIMEText(body, "plain", "utf-8")
    msg["Subject"] = f"⚠️ TradingView watchlist not updated · {datetime.today():%d %b %Y}"
    msg["From"], msg["To"] = gu, et
    try:
        with smtplib.SMTP_SSL("smtp.gmail.com", 465) as srv:
            srv.login(gu, gp.replace(" ", ""))
            srv.sendmail(gu, [x.strip() for x in et.split(",") if x.strip()], msg.as_string())
        print("[TradingView] alert email sent")
    except Exception as err:
        print(f"[TradingView] alert email failed: {err}")


def main(argv):
    probe = "--probe" in argv
    run = next((x for x in argv if x.isdigit()), None)
    try:
        sync(run, probe)
        return 0
    except SyncError as err:
        reason = str(err)
    except Exception as err:
        reason = f"{type(err).__name__}: {err}"
    print(f"[TradingView] ❌  {reason}")
    if not probe:
        alert(reason)
    return 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
