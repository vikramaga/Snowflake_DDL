# best_picks.py — "Best 10" shortlist for every scanner.
#
# Each scheduled scanner calls rank_best(df_out) right after position sizing
# and attach_to_email(msg) just before it sends its email. The email then
# opens with a Best 10 table (the full list stays below it, unchanged) and a
# second CSV, best10_<scanner>_<timestamp>.csv, is attached and saved next
# to the scanner's own CSV.
#
# BEST SCORE (0-100) = Trend 35 + Catalyst 30 + Value 35
#   Trend (daily, 1 year of data)
#     12  price above SMA200
#     10  SMA50 rising (higher than 20 sessions ago)
#      8  SMA200 rising (higher than 20 sessions ago)
#      5  SMA200 just STARTING to rise: rising now, and it was flat/falling
#         or rising more slowly over the 20 sessions before that
#   Catalyst (Yahoo Finance news, analyst actions and earnings)
#     12  a headline in the last 30 days about a contract / order / award /
#         deal / partnership (the "large contracts or orders" check)
#      4  each other positive headline (upgrade, beat, record, approval,
#         raises guidance, buyback ...), up to 12
#     -6  each negative headline (downgrade, miss, offering, lawsuit ...)
#      4  each net analyst upgrade in the last 60 days, up to 8
#      6  last reported quarter beat the EPS estimate
#   Value
#     fundamental (20): analyst mean target upside >=25% -> 10, >=10% -> 6,
#       >0 -> 3; PEG <= 1.5 or forward P/E <= 20 (positive) -> 6;
#       analyst rating Buy or better (recommendationMean <= 2.0) -> 4
#     entry (15): close 0-8% above SMA50 -> 8 (8-15% -> 4); close no more
#       than 30% above SMA200 -> 7 (30-50% -> 3)
#
# News and analyst data come from Yahoo through yfinance and can be missing
# or late; a stock with no data simply scores 0 on that part. Headlines are
# matched by keywords, so read the headline before acting on it. Everything
# is wrapped so a data problem never stops the scanner or its email.
import os, re, sys, time
from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd

TOP_N = int(os.environ.get("BEST10_N", "10"))
MAX_FUNDAMENTAL = int(os.environ.get("BEST10_MAX_CANDIDATES", "30"))   # deep look-ups for the 30 best by trend/entry
TIME_BUDGET_S = float(os.environ.get("BEST10_TIME_BUDGET", "240"))
NEWS_DAYS = 30
ANALYST_DAYS = 60

CONTRACT_WORDS = r"\b(contract|contracts|order|orders|awarded|award|wins|won|selected|deal|agreement|partnership|purchase order|backlog)\b"
POSITIVE_WORDS = (r"\b(upgrade|upgraded|upgrades|beat|beats|tops|record|surge|soars|approval|approved|"
                  r"clearance|raises|raised|boosts|lifts|buyback|repurchase|outperform|strong demand|"
                  r"expands|expansion|launch|breakthrough|accelerat\w*|guidance raise|price target raised)\b")
NEGATIVE_WORDS = (r"\b(downgrade|downgraded|downgrades|miss|misses|missed|cut|cuts|lowers|lowered|offering|dilution|"
                  r"lawsuit|probe|investigation|recall|plunge|plunges|tumbles|slumps|warns|warning|delay|delayed|"
                  r"halt|halted|bankruptcy|default|fraud|resigns|layoffs)\b")

LAST = None          # last result of rank_best (DataFrame) — used by attach_to_email
LAST_CSV = None      # path of the saved best-10 CSV
LAST_NOTE = ""       # shown in the email when there is nothing to rank
_SCANNER = None


def _scanner_name():
    main = sys.modules.get("__main__")
    f = getattr(main, "__file__", None) or (sys.argv[0] if sys.argv and sys.argv[0] else "scanner")
    return os.path.splitext(os.path.basename(f))[0] or "scanner"


def _sma(x, n):
    return pd.Series(x).rolling(n).mean().values


def _trend(df):
    """Trend and entry scores from daily OHLC. Returns dict or None."""
    c = df["Close"].astype(float).values
    if len(c) < 245 or not np.isfinite(c[-1]) or c[-1] <= 0:
        return None
    s50, s200 = _sma(c, 50), _sma(c, 200)
    i = len(c) - 1
    if not (np.isfinite(s50[i]) and np.isfinite(s200[i]) and np.isfinite(s200[i - 40])):
        return None
    above200 = c[i] > s200[i]
    s50_up = s50[i] > s50[i - 20]
    slope_now = (s200[i] - s200[i - 20]) / s200[i - 20] * 100
    slope_prev = (s200[i - 20] - s200[i - 40]) / s200[i - 40] * 100
    s200_up = slope_now > 0
    s200_turning = bool(s200_up and (slope_prev <= 0 or slope_now > slope_prev))
    t = 12 * above200 + 10 * s50_up + 8 * s200_up + 5 * s200_turning
    d50 = (c[i] - s50[i]) / s50[i] * 100
    d200 = (c[i] - s200[i]) / s200[i] * 100
    entry = (8 if 0 <= d50 <= 8 else 4 if 8 < d50 <= 15 else 0) + (7 if 0 <= d200 <= 30 else 3 if 30 < d200 <= 50 else 0)
    return {"trend": int(t), "entry": int(entry), "above200": bool(above200), "s50_up": bool(s50_up),
            "s200_up": bool(s200_up), "s200_turning": s200_turning, "d50": float(d50), "d200": float(d200),
            "s200_slope20": float(slope_now)}


def _news_items(tk):
    out = []
    try:
        items = tk.news or []
    except Exception:
        return out
    cutoff = datetime.now(timezone.utc) - timedelta(days=NEWS_DAYS)
    for it in items[:20]:
        try:
            c = it.get("content", it) if isinstance(it, dict) else {}
            title = c.get("title") or it.get("title") or ""
            ts = c.get("pubDate") or c.get("displayTime")
            if ts:
                when = pd.Timestamp(ts).tz_convert("UTC") if pd.Timestamp(ts).tzinfo else pd.Timestamp(ts).tz_localize("UTC")
            elif it.get("providerPublishTime"):
                when = pd.Timestamp(int(it["providerPublishTime"]), unit="s", tz="UTC")
            else:
                when = None
            if when is not None and when.to_pydatetime() < cutoff:
                continue
            if title:
                out.append((when, title.strip()))
        except Exception:
            continue
    return out


def _catalyst(tk):
    score, best_head, notes = 0, "", []
    news = _news_items(tk)
    contract_hit, pos_pts = False, 0
    for when, title in news:
        low = title.lower()
        neg = re.search(NEGATIVE_WORDS, low) is not None
        if neg:
            score -= 6
            continue
        if not contract_hit and re.search(CONTRACT_WORDS, low):
            contract_hit = True; score += 12; best_head = best_head or title
            notes.append("contract/order news")
        elif re.search(POSITIVE_WORDS, low) and pos_pts < 12:
            pos_pts += 4; score += 4; best_head = best_head or title
    if pos_pts:
        notes.append("positive news")
    try:
        ud = tk.upgrades_downgrades
        if ud is not None and len(ud):
            idx = pd.to_datetime(ud.index)
            idx = idx.tz_localize(None) if getattr(idx, "tz", None) is not None else idx
            recent = ud[idx >= pd.Timestamp.now() - pd.Timedelta(days=ANALYST_DAYS)]
            act = recent["Action"].astype(str).str.lower() if "Action" in recent else pd.Series(dtype=str)
            net = int((act == "up").sum() - (act == "down").sum())
            if net > 0:
                score += min(8, 4 * net); notes.append(f"{net} net upgrade{'s' if net > 1 else ''}")
    except Exception:
        pass
    try:
        ed = tk.get_earnings_dates(limit=8)
        if ed is not None and len(ed) and "Surprise(%)" in ed:
            rep = ed.dropna(subset=["Reported EPS"]) if "Reported EPS" in ed else ed.dropna(subset=["Surprise(%)"])
            if len(rep) and float(rep["Surprise(%)"].iloc[0]) > 0:
                score += 6; notes.append("beat last EPS")
    except Exception:
        pass
    return max(0, min(30, score)), best_head, notes


def _fundamental(tk, price):
    pts, notes, up, fpe, peg, rec = 0, [], np.nan, np.nan, np.nan, np.nan
    try:
        info = tk.info or {}
    except Exception:
        info = {}
    try:
        tgt = info.get("targetMeanPrice")
        if tgt and price and price > 0 and (info.get("numberOfAnalystOpinions") or 0) >= 1:
            up = (float(tgt) / float(price) - 1) * 100
            pts += 10 if up >= 25 else 6 if up >= 10 else 3 if up > 0 else 0
        fpe = info.get("forwardPE")
        peg = info.get("trailingPegRatio") or info.get("pegRatio")
        fpe = float(fpe) if fpe not in (None, "Infinity") else np.nan
        peg = float(peg) if peg not in (None, "Infinity") else np.nan
        if (np.isfinite(peg) and 0 < peg <= 1.5) or (np.isfinite(fpe) and 0 < fpe <= 20):
            pts += 6; notes.append("fair P/E" if not (np.isfinite(peg) and 0 < peg <= 1.5) else "PEG <= 1.5")
        rec = info.get("recommendationMean")
        rec = float(rec) if rec is not None else np.nan
        if np.isfinite(rec) and rec <= 2.0:
            pts += 4; notes.append("analysts Buy")
        if np.isfinite(up) and up >= 10:
            notes.append(f"target +{up:.0f}%")
    except Exception:
        pass
    return min(20, pts), up, fpe, peg, rec, notes


def _download(tickers):
    import yfinance as yf
    out = {}
    end = datetime.today() + timedelta(days=1)
    start = end - timedelta(days=420)
    for i in range(0, len(tickers), 50):
        batch = tickers[i:i + 50]
        try:
            raw = yf.download(batch, start=start.strftime("%Y-%m-%d"), end=end.strftime("%Y-%m-%d"),
                              group_by="ticker", auto_adjust=True, actions=False, threads=True, progress=False)
        except Exception:
            continue
        if raw is None or raw.empty:
            continue
        for t in batch:
            try:
                d = raw[t] if isinstance(raw.columns, pd.MultiIndex) else raw
                d = d.dropna(subset=["Close"])
                if len(d):
                    out[t] = d
            except Exception:
                pass
    return out


def rank_best(df_out, n=None, csv_dir=None):
    """Rank the scanner's qualified stocks and return the best n as a DataFrame.
    Never raises; on any problem returns an empty DataFrame."""
    global LAST, LAST_CSV, LAST_NOTE, _SCANNER
    n = TOP_N if n is None else n
    t0 = time.time()
    _SCANNER = _scanner_name()
    LAST, LAST_CSV, LAST_NOTE = pd.DataFrame(), None, "No qualified stocks to rank today."
    try:
        if df_out is None or len(df_out) == 0 or "Ticker" not in df_out.columns:
            print("  🏅 Best 10: no qualified stocks")
            return LAST
        import yfinance as yf
        base = df_out.drop_duplicates("Ticker").reset_index(drop=True)
        tickers = [str(t) for t in base["Ticker"]]
        print(f"  🏅 Best 10: ranking {len(tickers)} qualified stocks (trend, catalyst, value) ...")
        data = _download(tickers)
        if not data:
            LAST_NOTE = "Price data for the ranking was unavailable this run (Yahoo); see the full list below."
            print("  ⚠️  Best 10: no price data returned — ranking skipped")
            return LAST
        rows = []
        for t in tickers:
            tr = _trend(data[t]) if t in data else None
            rows.append({"Ticker": t, "_tr": tr,
                         "_pre": (tr["trend"] + tr["entry"]) if tr else -1,
                         "Price_now": float(data[t]["Close"].iloc[-1]) if t in data else np.nan})
        rows.sort(key=lambda r: r["_pre"], reverse=True)
        deep = set(r["Ticker"] for r in rows[:MAX_FUNDAMENTAL] if r["_pre"] >= 0)
        out = []
        for r in rows:
            t, tr = r["Ticker"], r["_tr"]
            cat, head, cnotes, fund, up, fpe, peg, rec, fnotes = 0, "", [], 0, np.nan, np.nan, np.nan, np.nan, []
            if t in deep and time.time() - t0 < TIME_BUDGET_S:
                tk = yf.Ticker(t)
                cat, head, cnotes = _catalyst(tk)
                fund, up, fpe, peg, rec, fnotes = _fundamental(tk, r["Price_now"])
            trend = tr["trend"] if tr else 0
            value = fund + (tr["entry"] if tr else 0)
            why = []
            if tr:
                if tr["above200"]: why.append("above SMA200")
                if tr["s50_up"]: why.append("SMA50 rising")
                if tr["s200_turning"]: why.append("SMA200 turning up")
                elif tr["s200_up"]: why.append("SMA200 rising")
            why += cnotes + fnotes
            out.append({
                "Ticker": t, "Best_Score": int(trend + cat + value), "Trend_Score": int(trend),
                "Catalyst_Score": int(cat), "Value_Score": int(value),
                "Above_SMA200": "Yes" if tr and tr["above200"] else "No",
                "SMA50_Rising": "Yes" if tr and tr["s50_up"] else "No",
                "SMA200_Rising": ("Turning up" if tr and tr["s200_turning"] else "Yes" if tr and tr["s200_up"] else "No"),
                "Pct_Above_SMA50": round(tr["d50"], 1) if tr else np.nan,
                "Pct_Above_SMA200": round(tr["d200"], 1) if tr else np.nan,
                "Analyst_Upside_%": round(up, 1) if np.isfinite(up) else np.nan,
                "Fwd_PE": round(fpe, 1) if np.isfinite(fpe) else np.nan,
                "PEG": round(peg, 2) if np.isfinite(peg) else np.nan,
                "Analyst_Rating": round(rec, 2) if np.isfinite(rec) else np.nan,
                "Catalyst_Headline": head[:140],
                "Why": ", ".join(why),
            })
        keep = [c for c in ("Price", "Entry_Price", "Stop_Loss", "Quantity", "Total_Amount", "Score") if c in base.columns]
        res = pd.DataFrame(out).merge(base[["Ticker"] + keep].rename(columns={"Score": "Scanner_Score"}),
                                      on="Ticker", how="left")
        order = ["Best_Score", "Trend_Score"] + (["Scanner_Score"] if "Scanner_Score" in res else [])
        res = res.sort_values(order, ascending=False, na_position="last").head(n)
        res.insert(0, "Best_Rank", range(1, len(res) + 1))
        LAST = res.reset_index(drop=True)
        try:
            d = csv_dir or os.environ.get("GITHUB_WORKSPACE", os.getcwd())
            ts = datetime.today().strftime("%Y%m%d_%H%M")
            LAST_CSV = os.path.join(d, f"best10_{_SCANNER}_{ts}.csv")
            LAST.to_csv(LAST_CSV, index=False)
            print(f"  🏅 Best 10 → {LAST_CSV}  ({time.time()-t0:.0f}s)")
        except Exception as e:
            LAST_CSV = None
            print(f"  ⚠️  Best 10 CSV not saved: {e}")
        for _, r in LAST.iterrows():
            print(f"     #{r.Best_Rank:<2} {r.Ticker:<10} {r.Best_Score:>3}  (trend {r.Trend_Score}, catalyst "
                  f"{r.Catalyst_Score}, value {r.Value_Score})  {r.Why}")
        return LAST
    except Exception as e:
        print(f"  ⚠️  Best 10 skipped: {type(e).__name__}: {e}")
        LAST = pd.DataFrame()
        return LAST


def _fmt_money(v, inr):
    try:
        return ("₹" if inr else "$") + f"{float(v):,.2f}"
    except Exception:
        return "—"


def _html(best):
    if best is None or len(best) == 0:
        body = f'<p style="margin:0;color:#64748b;font-size:12px">{LAST_NOTE or "No qualified stocks to rank today."}</p>'
    else:
        inr = best["Ticker"].astype(str).str.endswith(".NS").any()
        cols = ["#", "Ticker", "Best", "Trend", "Catalyst", "Value", "Entry", "Stop", "Qty", "Analyst upside", "Why", "Catalyst headline"]
        th = "".join(f'<th style="background:#14532d;color:#ecfdf5;padding:6px 8px;font-size:11px;text-align:left;'
                     f'white-space:nowrap">{c}</th>' for c in cols)
        rows = ""
        for k, r in best.iterrows():
            bg = "#ffffff" if k % 2 == 0 else "#f0fdf4"
            up = r.get("Analyst_Upside_%")
            vals = [r["Best_Rank"], f"<b>{r['Ticker']}</b>", f"<b>{r['Best_Score']}</b>", r["Trend_Score"],
                    r["Catalyst_Score"], r["Value_Score"],
                    _fmt_money(r.get("Entry_Price", r.get("Price")), inr), _fmt_money(r.get("Stop_Loss"), inr),
                    "" if pd.isna(r.get("Quantity", np.nan)) else int(r.get("Quantity")),
                    "—" if pd.isna(up) else f"{up:+.0f}%", r.get("Why", ""),
                    (r.get("Catalyst_Headline") or "—")]
            rows += f'<tr style="background:{bg}">' + "".join(
                f'<td style="padding:5px 8px;font-size:11px;vertical-align:top">{v}</td>' for v in vals) + "</tr>"
        body = (f'<div style="overflow-x:auto"><table style="border-collapse:collapse;width:100%">'
                f'<thead><tr>{th}</tr></thead><tbody>{rows}</tbody></table></div>')
    return (f'<table width="100%" cellpadding="0" cellspacing="0"><tr><td align="center" style="padding:16px 10px 0">'
            f'<table width="100%" style="max-width:900px;background:#fff;border:2px solid #16a34a;border-radius:12px">'
            f'<tr><td style="padding:12px 16px">'
            f'<h2 style="margin:0 0 4px;color:#166534;font-size:16px">🏅 Best {TOP_N} of today\'s qualified stocks</h2>'
            f'<p style="margin:0 0 8px;color:#475569;font-size:11px">Score = Trend 35 (above SMA200, SMA50 rising, SMA200 rising / turning up) '
            f'+ Catalyst 30 (contract/order and positive news, analyst upgrades, EPS beat) + Value 35 (analyst upside, P/E or PEG, '
            f'rating, entry not extended above SMA50/SMA200). News is keyword-matched from Yahoo — read the headline. '
            f'Full list below; Best 10 CSV attached.</p>{body}</td></tr></table></td></tr></table>')


def _text(best):
    lines = [f"BEST {TOP_N} (trend 35 + catalyst 30 + value 35)", "-" * 60]
    if best is None or len(best) == 0:
        lines.append(LAST_NOTE or "No qualified stocks to rank today.")
    else:
        for _, r in best.iterrows():
            up = r.get("Analyst_Upside_%")
            lines.append(f"#{r['Best_Rank']:<2} {r['Ticker']:<10} {r['Best_Score']:>3} (T{r['Trend_Score']} C{r['Catalyst_Score']} "
                         f"V{r['Value_Score']})  upside {'—' if pd.isna(up) else f'{up:+.0f}%'}  {r.get('Why', '')}")
            if r.get("Catalyst_Headline"):
                lines.append(f"      news: {r['Catalyst_Headline']}")
    lines += ["-" * 60, "FULL LIST", ""]
    return "\n".join(lines) + "\n"


def attach_to_email(msg, best=None, csv_path=None):
    """Put the Best 10 table at the top of the email's HTML and plain-text
    bodies and attach the Best 10 CSV. Modifies msg in place; never raises."""
    try:
        from email.mime.text import MIMEText
        from email.mime.base import MIMEBase
        from email import encoders
        best = LAST if best is None else best
        csv_path = LAST_CSV if csv_path is None else csv_path
        if best is None:
            return msg
        done = {"html": False, "plain": False}

        def walk(part):
            if not part.is_multipart():
                return
            payload = part.get_payload()
            for k, sub in enumerate(payload):
                if sub.is_multipart():
                    walk(sub); continue
                ctype = sub.get_content_type()
                if sub.get_filename() or ctype not in ("text/html", "text/plain"):
                    continue
                kind = "html" if ctype == "text/html" else "plain"
                if done[kind]:
                    continue
                cs = sub.get_content_charset() or "utf-8"
                txt = sub.get_payload(decode=True).decode(cs, errors="replace")
                if kind == "html":
                    block = _html(best)
                    m = re.search(r"<body[^>]*>", txt, flags=re.I)
                    txt = (txt[:m.end()] + block + txt[m.end():]) if m else (block + txt)
                else:
                    txt = _text(best) + txt
                payload[k] = MIMEText(txt, kind, "utf-8")
                done[kind] = True
        walk(msg)
        if csv_path and os.path.exists(csv_path) and msg.is_multipart():
            with open(csv_path, "rb") as f:
                att = MIMEBase("application", "octet-stream"); att.set_payload(f.read())
            encoders.encode_base64(att)
            att.add_header("Content-Disposition", f"attachment; filename={os.path.basename(csv_path)}")
            msg.attach(att)
        subj = msg.get("Subject")
        if subj and len(best) and "🏅" not in str(subj):
            del msg["Subject"]; msg["Subject"] = f"{subj} · 🏅 Best: {', '.join(best['Ticker'].head(3))}"
        print(f"  🏅 Best 10 added to the email ({'html ' if done['html'] else ''}{'text' if done['plain'] else ''})")
    except Exception as e:
        print(f"  ⚠️  Best 10 not added to the email: {type(e).__name__}: {e}")
    return msg
