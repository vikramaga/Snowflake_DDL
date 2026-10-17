
# Quantity / Total_Amount per ticker for the email (from the sized df_out)
_SIZE = {}
if {"Ticker", "Quantity", "Total_Amount"} <= set(df_out.columns):
    _SIZE = {t: (q, a) for t, q, a in zip(df_out["Ticker"], df_out["Quantity"], df_out["Total_Amount"])}

# ── Email with CSV attached ───────────────────────────────
def _send_email(rl, csv_path):
    import smtplib
    from email.mime.multipart import MIMEMultipart
    from email.mime.text      import MIMEText
    from email.mime.base      import MIMEBase
    from email                import encoders

    gu = _GMAIL_USER; gp = _GMAIL_PASS; et = _EMAIL_TO

    if not gu:
        print("[Email] ❌  GMAIL_USER secret is empty")
        return
    if not gp:
        print("[Email] ❌  GMAIL_PASS secret is empty")
        print("         → Must be a Gmail App Password (16 chars, no spaces)")
        return
    if not et:
        print("[Email] ❌  EMAIL_TO secret is empty")
        return

    eto = [e.strip() for e in et.split(",") if e.strip()]
    cnt = len(rl)

    try:
        print(f"[Email] Sending to {et}  ({cnt} results)...")

        heads = ["Ticker","Price","Score","Entry","Stop","Risk %","Support",
                 "MA","vs MA","Bounces","Held","Qty","Amount"]
        th_e = "".join(
            f'<th style="background:#1e293b;color:#e2e8f0;padding:8px 11px;'
            f'font-size:11px;font-weight:700;border-bottom:2px solid #3b82f6;'
            f'white-space:nowrap">{c}</th>' for c in heads)

        def td(v, extra=""):
            return f'<td style="padding:6px 11px;font-size:12px;{extra}">{v}</td>'

        rows_e = ""
        for i, r in enumerate(rl[:50]):
            bg = "#fff" if i % 2 == 0 else "#f0f9ff"
            q, a = _SIZE.get(r.get("Ticker"), ("—", None))
            rows_e += (
                f'<tr style="background:{bg}">'
                + td(r.get("Ticker","—"), "font-weight:700")
                + td(f'${float(r.get("Price",0) or 0):.2f}')
                + td(f'{float(r.get("Score",0) or 0):.0f}',
                     "font-weight:700;background:#166534;color:#fff;text-align:center")
                + td(f'${float(r.get("Entry_Price",0) or 0):.2f}', "color:#22c55e")
                + td(f'${float(r.get("Stop_Loss",0) or 0):.2f}', "color:#ef4444")
                + td(f'{float(r.get("Risk_%",0) or 0):.1f}%')
                + td(r.get("Support_MA","—"), "font-weight:700;color:#7c3aed")
                + td(f'${float(r.get("MA_Value",0) or 0):.2f}')
                + td(f'{float(r.get("Dist_to_MA_%",0) or 0):+.1f}%')
                + td(f'{r.get("Bounces","—")}')
                + td(f'{float(r.get("Held_%",0) or 0):.0f}%')
                + td(f'{q}')
                + td(f'${float(a):,.0f}' if a is not None else "—")
                + '</tr>')
        no_results_msg = ('<tr><td colspan="13" style="padding:20px;text-align:center;'
                          'color:#94a3b8;font-size:13px">No matches today</td></tr>')
        more_note = (f'Showing top 50 of {cnt} — full list in the CSV attachment'
                     if cnt > 50 else '📎 Full results attached as CSV')

        fc = FUNNEL_COUNTS
        html_e = f"""<!DOCTYPE html><html><body style="margin:0;padding:0;
background:#f1f5f9;font-family:'Segoe UI',Arial,sans-serif">
<table width="100%" cellpadding="0" cellspacing="0"><tr><td align="center" style="padding:20px 10px">
<table width="100%" style="max-width:900px;background:#fff;border-radius:12px;
       overflow:hidden;box-shadow:0 4px 16px rgba(0,0,0,0.08)">
  <tr><td style="background:linear-gradient(135deg,#0f172a,#1e3a5f);padding:22px 28px">
<h1 style="margin:0;color:#60a5fa;font-size:20px;font-weight:700">
  📊 Daily MA Support — EMA21 / SMA50
</h1>
<p style="margin:6px 0 0;color:#94a3b8;font-size:12px">
  {datetime.today().strftime('%Y-%m-%d %H:%M UTC')} &nbsp;·&nbsp;
  {cnt} match{'es' if cnt!=1 else ''} — price took support on the same MA (EMA{CFG['ema_period']} or
  SMA{CFG['sma_period']}) most of the last {CFG['support_window_days']} days and is back near it today
</p>
  </td></tr>
  <tr><td style="padding:14px 28px 4px;background:#0b1220">
<div style="background:#111827;border:1px solid #1f2937;border-radius:8px;padding:12px 16px">
  <p style="margin:0 0 6px;color:#93c5fd;font-size:12px;font-weight:700">
    🔍 FUNNEL — last {CFG['signal_lookback_days']} trading day(s), every liquid ticker
  </p>
  <p style="margin:0;color:#cbd5e1;font-size:12px">
    {fc['days_checked']} ticker-days &nbsp;→&nbsp;
    {fc['passed_step1_support_found']} support MA with &ge;{CFG['min_bounces']} bounces &nbsp;→&nbsp;
    {fc['passed_step2_held_most_of_time']} held &ge;{CFG['min_held_ratio']*100:.0f}% &nbsp;→&nbsp;
    {fc['passed_step3_rising_ma']} MA rising &nbsp;→&nbsp;
    <b style="color:#facc15">{fc['passed_step4_near_ma_now']} near the MA today = full match</b>
  </p>
</div>
  </td></tr>
  <tr><td style="padding:16px">
<div style="overflow-x:auto;border-radius:8px;border:1px solid #e2e8f0">
  <table style="border-collapse:collapse;width:100%;min-width:760px">
    <thead><tr>{th_e}</tr></thead>
    <tbody>{rows_e or no_results_msg}</tbody>
  </table>
</div>
<p style="font-size:11px;color:#64748b;margin:8px 0 0">{more_note}</p>
{f'''<div style="margin-top:10px;background:#f8fafc;border:1px solid #e2e8f0;
        border-radius:6px;padding:10px 14px">
  <p style="margin:0 0 4px;font-size:10px;color:#94a3b8;font-weight:700">
    TICKERS (comma-separated, copy/paste)
  </p>
  <p style="margin:0;font-size:12px;color:#1e293b;font-family:monospace;
      word-break:break-all">{", ".join(r.get("Ticker","") for r in rl)}</p>
</div>''' if rl else ''}
  </td></tr>
  <tr><td style="background:#f8fafc;padding:12px 28px;
             border-top:1px solid #e2e8f0;text-align:center">
<p style="margin:0;color:#94a3b8;font-size:10px">
  ⚠️ Not financial advice &nbsp;·&nbsp; Entry = today's close, Stop = {CFG['stop_buffer_pct']:.0f}% below the support MA (defaults) &nbsp;·&nbsp; Qty sized for a $40 max loss &nbsp;·&nbsp; Auto-generated by GitHub Actions
</p>
  </td></tr>
</table>
</td></tr></table>
</body></html>"""

        plain_lines = [
            f"Daily MA Support (EMA21/SMA50) — {datetime.today().strftime('%Y-%m-%d')}",
            f"{cnt} matches (support on the same MA most of the last {CFG['support_window_days']} days; near it today)",
            "="*60,
            f"FUNNEL: {fc['days_checked']} ticker-days -> "
            f"{fc['passed_step1_support_found']} support MA >={CFG['min_bounces']} bounces -> "
            f"{fc['passed_step2_held_most_of_time']} held >={CFG['min_held_ratio']*100:.0f}% -> "
            f"{fc['passed_step3_rising_ma']} MA rising -> "
            f"{fc['passed_step4_near_ma_now']} near MA today (=full match)",
            "="*60,
        ]
        if rl:
            for r in rl[:50]:
                q, a = _SIZE.get(r.get("Ticker"), ("—", None))
                plain_lines.append(
                    f"{r.get('Ticker','—'):<7} ${float(r.get('Price',0) or 0):.2f}  "
                    f"Score:{float(r.get('Score',0) or 0):.0f}  "
                    f"Entry:${float(r.get('Entry_Price',0) or 0):.2f}  "
                    f"SL:${float(r.get('Stop_Loss',0) or 0):.2f}  "
                    f"Risk:{float(r.get('Risk_%',0) or 0):.1f}%  "
                    f"{r.get('Support_MA','—')}:${float(r.get('MA_Value',0) or 0):.2f} "
                    f"({float(r.get('Dist_to_MA_%',0) or 0):+.1f}%)  "
                    f"Bounces:{r.get('Bounces','—')}  Held:{float(r.get('Held_%',0) or 0):.0f}%  "
                    f"Qty:{q}  Amount:{'$%s' % format(float(a), ',.0f') if a is not None else '—'}")
            plain_lines.append("")
            plain_lines.append("Tickers (comma-separated):")
            plain_lines.append(", ".join(r.get("Ticker","") for r in rl))
        else:
            plain_lines.append("No matches today")
        plain_lines.append("\nFull results in CSV attachment.")
        plain_e = "\n".join(plain_lines)

        subj = (f"📊 MA Support EMA21/SMA50 — {cnt} signal{'s' if cnt!=1 else ''}"
                f" — {datetime.today().strftime('%Y-%m-%d')}")

        msg = MIMEMultipart("mixed")
        msg["Subject"] = subj
        msg["From"]    = gu
        msg["To"]      = ", ".join(eto)

        alt = MIMEMultipart("alternative")
        alt.attach(MIMEText(plain_e, "plain"))
        alt.attach(MIMEText(html_e,  "html"))
        msg.attach(alt)

    except Exception as e:
        print(f"[Email] ❌  Failed to build email body: {type(e).__name__}: {e}")
        return

    if csv_path and os.path.exists(csv_path):
        try:
            with open(csv_path, "rb") as f:
                part = MIMEBase("application", "octet-stream")
                part.set_payload(f.read())
            encoders.encode_base64(part)
            part.add_header("Content-Disposition",
                f"attachment; filename={os.path.basename(csv_path)}")
            msg.attach(part)
            sz = os.path.getsize(csv_path)
            print(f"[Email] 📎 Attached: {os.path.basename(csv_path)} ({sz:,} bytes)")
        except Exception as e:
            print(f"[Email] ⚠️  CSV attach failed: {e}")

    try:
        print(f"[Email] Connecting to smtp.gmail.com:465 ...")
        with smtplib.SMTP_SSL("smtp.gmail.com", 465) as srv:
            srv.login(gu, gp.replace(" ", ""))
            srv.sendmail(gu, eto, msg.as_string())
        print(f"[Email] ✅  Sent successfully to: {', '.join(eto)}")
        print(f"[Email]    Subject: {subj}")
    except smtplib.SMTPAuthenticationError:
        print("[Email] ❌  AUTHENTICATION FAILED")
        print("         GMAIL_PASS must be a Gmail App Password, NOT your login password")
    except smtplib.SMTPException as e:
        print(f"[Email] ❌  SMTP error: {e}")
    except Exception as e:
        print(f"[Email] ❌  Unexpected error: {type(e).__name__}: {e}")

try:
    _send_email(results, fpath)
except Exception as e:
    print(f"[Email] ❌  Unexpected top-level error: {type(e).__name__}: {e}")
    print("[Email]    Continuing — CSV is still saved.")

if _IN_NOTEBOOK:
    try:
        from google.colab import files
        files.download(fpath); files.download(tv)
    except Exception: pass
else:
    print("  (CI: files in workspace, email sent)")

# ── Charts for top 5: price with EMA21 and SMA50, the support MA
#    highlighted, the support window shaded and bounce days marked ────
if results:
    top = results[:min(5,len(results))]
    fig, axes = plt.subplots(len(top), 1, figsize=(13, 3.6*len(top)), facecolor="#0f172a",
                              squeeze=False)
    axes = axes[:,0]
    for p, r in enumerate(top):
        ax = axes[p]
        df_all = r["_df_daily"]
        df_p = df_all.tail(90)
        idx  = df_p.index
        k = r["_idx"]
        W = CFG["support_window_days"]
        sup_is_ema = r["Support_MA"] == "EMA21"

        ax.set_facecolor("#0f172a")
        ax.plot(idx, df_p["Close"], color="#60a5fa", lw=1.5, label="Close", zorder=5)
        ax.plot(idx, r["_ema"].reindex(idx), color="#22c55e",
                lw=2.0 if sup_is_ema else 0.9, ls="-" if sup_is_ema else ":", label="EMA21")
        ax.plot(idx, r["_sma"].reindex(idx), color="#fbbf24",
                lw=2.0 if not sup_is_ema else 0.9, ls="-" if not sup_is_ema else ":", label="SMA50")
        ax.axhline(r["Stop_Loss"], color="#ef4444", lw=1.0, ls="--", alpha=0.85,
                   label=f"Stop {r['Stop_Loss']:.2f}")
        w0 = df_all.index[max(0, k - W)]
        ax.axvspan(max(w0, idx[0]), df_all.index[k], color="#1e3a5f", alpha=0.35, lw=0)
        ma_s = r["_ema"] if sup_is_ema else r["_sma"]
        tp = CFG["touch_pct"] / 100.0
        touch = [d for d in idx if df_all.index.get_loc(d) < k and d >= w0
                 and df_all.loc[d, "Low"] <= ma_s.loc[d] * (1 + tp)
                 and df_all.loc[d, "Close"] >= ma_s.loc[d] * (1 - CFG["hold_tol_pct"]/100.0)]
        if touch:
            ax.scatter(touch, [df_all.loc[d, "Low"] for d in touch], color="#a78bfa", s=14,
                       zorder=6, label="Touches")
        sig_d = df_all.index[k]
        if sig_d in idx:
            ax.scatter([sig_d], [df_all["Close"].iloc[k]], color="#facc15", s=50, zorder=7, label="Today")
        ax.set_title(f"{r['Ticker']}  |  {r['Price']:.2f} USD  |  Score {r['Score']}  |  "
                     f"{r['Support_MA']} support: {r['Bounces']} bounces, held {r['Held_%']:.0f}%  |  "
                     f"{r['Dist_to_MA_%']:+.1f}% from MA",
                     color="#e2e8f0", fontsize=9, fontweight="bold", pad=7)
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))
        ax.tick_params(colors="#94a3b8", labelsize=8)
        for sp in ax.spines.values(): sp.set_edgecolor("#1e3a5f")
        ax.legend(loc="upper left", facecolor="#1e293b", labelcolor="#e2e8f0", fontsize=6, ncol=6)
        ax.grid(color="#1e3a5f", ls="--", lw=0.5, alpha=0.6)
    plt.suptitle(f"Daily MA Support (EMA21 / SMA50)  ·  {datetime.today().strftime('%Y-%m-%d')}",
                 color="#60a5fa", fontsize=12, fontweight="bold", y=1.001)
    plt.tight_layout()
    cp = os.path.join(out_dir, f"daily_ma_support_chart_{ts}.png")
    plt.savefig(cp, dpi=150, bbox_inches="tight", facecolor="#0f172a")
    if _IN_NOTEBOOK: plt.show()
    else: plt.close()
    print(f"  📊 Chart → {cp}")
    if _IN_NOTEBOOK:
        try:
            from google.colab import files; files.download(cp)
        except Exception: pass

print(f"""
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  📋 SIGNAL (all required, on the latest daily candle)
  For EMA{CFG['ema_period']} and SMA{CFG['sma_period']}, over the last {CFG['support_window_days']} trading days:
   BOUNCE = low within {CFG['touch_pct']}% above the MA (or below it), close no more
            than {CFG['hold_tol_pct']}% below it, then a close {CFG['bounce_pct']}%+ above the MA within
            {CFG['bounce_days']} days (touches within {CFG['min_touch_gap']} bars = one event)
   HELD % = share of days closing no more than {CFG['hold_tol_pct']}% below the MA
  Support MA = the one with more bounces (tie: higher Held %).

  1) Support MA has >= {CFG['min_bounces']} bounces.
  2) Held % >= {CFG['min_held_ratio']*100:.0f}%.
  3) That MA is rising (higher than {CFG['slope_days']} days ago).
  4) Today's close is {CFG['near_below_pct']}% below to {CFG['near_above_pct']}% above that MA.

  📋 OUTPUT (reasonable defaults — not explicitly requested)
  Entry_Price = today's CLOSE
  Stop_Loss   = {CFG['stop_buffer_pct']}% below the support MA
  Quantity / Total_Amount = sized for a $40 max loss (position_sizing.py)
  Universe    = price >= 10 USD, 20d avg volume >= 100k

  📋 SCORE (0-100)
  Base 20 + bounces (0-20) + Held % above the minimum (0-20) + closeness
  to the MA (0-20) + MA slope (0-10) + quiet-volume pullback (10);
  -10 if stop risk > 8%

  ⚙️  TUNE IF 0 RESULTS
  min_bounces                     3 → 2
  min_held_ratio               0.80 → 0.70
  near_above_pct                3.0 → 4.0
  min_price                      10 → 5
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
""")
