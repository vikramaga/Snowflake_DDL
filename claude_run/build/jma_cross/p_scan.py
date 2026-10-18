
# ── Main scan — single pass (daily download only, then check) ────
print("━"*65)
print(f"  STEP 3  SCANNING {len(TICKERS)} TICKERS")
print("━"*65)
print("  Fetching daily bars (single download per ticker)")
print(f"  Signal: JMA({CFG['jma_period']}) crossed above EMA{CFG['ema_period']} from below, and the "
      f"next day closed above JMA (stop = the JMA/EMA{CFG['ema_period']} cross price)\n")

_hdr_done = False
results = []
no_daily_data = 0

daily_batches = [TICKERS[i:i+CFG["batch_size"]]
                 for i in range(0, len(TICKERS), CFG["batch_size"])]

daily_map = {}

with tqdm(total=len(TICKERS), desc="Daily fetch", unit="stk",
          bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]") as pbar:
    for batch in daily_batches:
        got = download_daily(batch, CFG["history_days"])
        daily_map.update(got)
        no_daily_data += len(batch) - len(got)
        pbar.update(len(batch))
        time.sleep(CFG["batch_sleep"])

got_daily = len(TICKERS) - no_daily_data
print(f"\n  Daily data : {got_daily}/{len(TICKERS)} tickers")
print()

for sym in tqdm(list(daily_map.keys()), desc="Checking JMA/EMA8 cross", unit="stk"):
    try:
        r = analyze_jma_cross(sym, daily_map[sym])
        if r is None: continue
        r["Ticker"] = sym
        results.append(r)
        live_print(r)
    except Exception: pass

print(f"\n{'━'*65}")
print(f"  SCAN COMPLETE")
print(f"  Tickers scanned : {len(TICKERS)}")
print(f"  Daily data      : {got_daily}")
print(f"  ✅ Matches       : {len(results)}")
print(f"{'━'*65}")

# ── Diagnostic funnel — where ticker-days were filtered out ──────
print(f"\n{'━'*65}")
print(f"  🔍 FUNNEL — where ticker-days were filtered out")
print(f"  (tallied across every liquid ticker's last {CFG['signal_lookback_days']} trading day(s))")
print(f"{'━'*65}")
fc = FUNNEL_COUNTS
print(f"  Ticker-days checked                              : {fc['days_checked']}")
print(f"  Step 1 — JMA crossed above EMA{CFG['ema_period']} the day before    : {fc['passed_step1_jma_crossed_above_ema8']}")
print(f"  Step 2 — + next day closed above JMA            : {fc['passed_step2_next_close_above_jma']}")
print(f"  Step 3 — + close above the cross price (stop)   : {fc['passed_step3_close_above_cross_price']}  (= full pattern)")
print(f"{'━'*65}")

if not results:
    print("\n  No matches. Try relaxing (see the FUNNEL above to see which")
    print("  step is actually the bottleneck before guessing):")
    print("   signal_lookback_days            1 → 3     (confirmations in the last 3 days)")
    print("   min_price                      10 → 5")
    print("   min_avg_volume             100000 → 50000")

results.sort(key=lambda x: x["Score"], reverse=True)

# ── Always build df_out and save/email (even if 0 results) ────
ts      = datetime.today().strftime("%Y%m%d_%H%M")
out_dir = os.environ.get("GITHUB_WORKSPACE", os.getcwd())

COLS = [
    "Ticker","Price","Score",
    "Entry_Price","Stop_Loss","Risk_%",
    "Cross_Price","Cross_Date","Confirm_Date",
    "JMA","EMA8","Close_vs_JMA_%","JMA_EMA8_Gap_%",
    "Vol_vs_20d_Avg","SMA50","Above_SMA50",
    "Signal_Date","Days_Since_Signal","Flags",
]
df_out = pd.DataFrame([{k:v for k,v in r.items() if not k.startswith("_")}
                        for r in results]) if results else pd.DataFrame(columns=COLS)
if not df_out.empty:
    df_out = df_out[[c for c in COLS if c in df_out.columns]]
    df_out.reset_index(drop=True, inplace=True)

FMT = {
    "Price"         : lambda v: f"${v:.2f}",
    "Score"         : lambda v: f"{v:.0f}",
    "Entry_Price"   : lambda v: f"${v:.2f}",
    "Stop_Loss"     : lambda v: f"${v:.2f}",
    "Cross_Price"   : lambda v: f"${v:.2f}",
    "JMA"           : lambda v: f"${v:.2f}",
    "EMA8"          : lambda v: f"${v:.2f}",
    "SMA50"         : lambda v: f"${v:.2f}",
    "Risk_%"        : lambda v: f"{v:.1f}%",
    "Close_vs_JMA_%": lambda v: f"{v:+.1f}%",
    "JMA_EMA8_Gap_%": lambda v: f"{v:+.2f}%",
    "Vol_vs_20d_Avg": lambda v: f"x{v:.2f}",
    "Days_Since_Signal": lambda v: f"{int(v)}d ago",
}

def fmt_v(col, val):
    if val is None or (isinstance(val, float) and np.isnan(val)): return "—"
    try:
        if col in FMT: return FMT[col](val)
    except Exception: pass
    return str(val) if str(val) not in ("nan","None","") else "—"

DISP_COLS = ["Ticker","Price","Score","Entry_Price","Stop_Loss","Risk_%",
             "Cross_Date","Confirm_Date","Close_vs_JMA_%","Vol_vs_20d_Avg"]

if _IN_NOTEBOOK and results:
    th = "".join(
        f'<th style="background:#0f172a;color:#e2e8f0;padding:9px 12px;'
        f'font-size:11px;font-weight:700;text-align:center;'
        f'border-bottom:2px solid #22c55e;white-space:nowrap">{c}</th>'
        for c in DISP_COLS)
    rows_html = ""
    for i, r in enumerate(results):
        bg  = "#ffffff" if i%2==0 else "#f0f9ff"
        tds = "".join(
            f'<td style="padding:7px 12px;font-size:12px;border-bottom:1px solid #e2e8f0;'
            f'white-space:nowrap">{fmt_v(col, r.get(col))}</td>' for col in DISP_COLS)
        rows_html += f'<tr style="background:{bg}">{tds}</tr>\n'
    display_html(f"""
<div style="background:linear-gradient(135deg,#0f172a,#1e3a5f);
        border-radius:10px;padding:18px 24px;margin-bottom:8px;
        font-family:'Segoe UI',Arial,sans-serif">
  <h2 style="margin:0;color:#60a5fa;font-size:20px;font-weight:700">
    📈 Daily JMA Cross Above EMA8
  </h2>
  <p style="margin:6px 0 0;color:#94a3b8;font-size:12px">
    {datetime.today().strftime('%Y-%m-%d %H:%M')} &nbsp;·&nbsp;
    <b style="color:#22c55e">{len(results)} matches</b> from {len(TICKERS)} tickers
  </p>
</div>
<div style="overflow-x:auto;border:1px solid #e2e8f0;border-radius:8px">
  <table style="border-collapse:collapse;width:100%;min-width:700px">
    <thead><tr>{th}</tr></thead>
    <tbody>{rows_html}</tbody>
  </table>
</div>""")

elif results:
    # ASCII table (CLI/GitHub Actions mode)
    CLI_COLS = [c for c in DISP_COLS if c in df_out.columns]
    col_w = {c: max(len(c), max(
        len(fmt_v(c, df_out[c].iloc[i])) for i in range(len(df_out))
    ))+2 for c in CLI_COLS}
    top  = "┬".join("─"*col_w[c] for c in CLI_COLS)
    sep  = "┼".join("─"*col_w[c] for c in CLI_COLS)
    bot  = "┴".join("─"*col_w[c] for c in CLI_COLS)
    hdr  = "│".join(c.center(col_w[c]) for c in CLI_COLS)
    inner= sum(col_w.values()) + len(CLI_COLS) - 1
    print()
    print(f"  ╔{'═'*inner}╗")
    tit = f"  Daily JMA Cross Above EMA8   {datetime.today().strftime('%Y-%m-%d')}   {len(df_out)} matches"
    print(f"  ║{tit.center(inner)}║")
    print(f"  ╚{'═'*inner}╝\n")
    print(f"  ┌{top}┐")
    print(f"  │{hdr}│")
    print(f"  ├{sep}┤")
    for i,(_, row_) in enumerate(df_out.iterrows()):
        cells=[fmt_v(c,row_.get(c)).center(col_w[c]) for c in CLI_COLS]
        print(f"  │{'│'.join(cells)}│")
        if i<len(df_out)-1: print(f"  ├{sep}┤")
    print(f"  └{bot}┘")
    print(f"""
  COLUMN KEY
  ──────────────────────────────────────────────────────
  Score           0-100 (JMA/EMA8 gap + close vs JMA + tight stop
                  + volume + above SMA50)
  Cross_Date      day JMA crossed above EMA{CFG['ema_period']}
  Confirm_Date    next day, closed above JMA (= signal)
  Stop_Loss       the price where JMA and EMA{CFG['ema_period']} intersect (Cross_Price)
  Entry_Price     confirm-day close
  ──────────────────────────────────────────────────────""")

# Save
fpath = os.path.join(out_dir, f"daily_jma_cross_ema8_{ts}.csv")
# ── Position sizing: Quantity / Total_Amount for a fixed max loss (see position_sizing.py)
try:
    from position_sizing import add_position_size
    df_out = add_position_size(df_out)
except Exception as _ps_err:
    print(f"  ⚠️  Position sizing skipped: {_ps_err}")
df_out.to_csv(fpath, index=False)
print(f"\n  💾 CSV → {fpath}")
tv = os.path.join(out_dir, f"tv_daily_jma_cross_ema8_{ts}.txt")
with open(tv,"w") as f:
    f.write(f"###Daily JMA Cross Above EMA8 {datetime.today().strftime('%Y-%m-%d')}\n")
    for r in results: f.write(f"NASDAQ:{r['Ticker']}\n")
print(f"  📋 TradingView → {tv}")
if results:
    print(f"\n  📋 Tickers (comma-separated):")
    print(f"  {', '.join(r['Ticker'] for r in results)}")

