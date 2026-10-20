# Generates daily_ma_sweep_4x_v1.py: boilerplate, download and ticker code are
# copied unchanged from daily_rsi2_snapback_v1.py; pattern, output, email and
# chart code are below. Usage: python gen_ma_sweep_4x.py [out]
import sys, py_compile
SRC = "daily_rsi2_snapback_v1.py"
OUT = sys.argv[1] if len(sys.argv) > 1 else "daily_ma_sweep_4x_v1.py"
L = open(SRC).read().split("\n")
assert L[37].startswith("import subprocess") and L[117].startswith("# ── CONFIG")
assert L[321].startswith("# ── Download: daily") and L[368].startswith("# ── Live print")
assert L[399].startswith("# ── Health check") and L[475] == "TICKERS = get_tickers()"

HEADER = '''# ============================================================
# NASDAQ — Daily MA Sweep + 4x Green Candle (v1)
# ============================================================
#
# SIGNAL (all required; the strong green candle is the latest daily candle)
#
#   1. SWEEP DAY (previous candle): a long lower wick (at least 0.75 x
#      ATR14) whose LOW touches the SMA50 or the EMA21 (at or below it, or
#      within 0.1 x ATR above it), while the candle CLOSES above that same
#      moving average (e.g. the body holds up near the EMA8).
#
#   2. STRONG GREEN DAY (today): close > open, and today's body
#      (close - open) is MORE THAN 4x the sweep day's body |close - open|.
#      To keep a tiny previous body from making a small candle qualify,
#      today's body must also be at least 0.5 x ATR14 (as of the sweep day).
#
#   3. Today's close is still above the moving average that was swept.
#
#   If both averages were swept, the row says "SMA50+EMA21".
#
# OUTPUT: Entry_Price and Stop_Loss are reasonable defaults, not
# explicitly requested (Entry = today's close; Stop = the sweep-day low).
# Quantity / Total_Amount come from position_sizing.py (max loss $40).
# Stocks whose ATR14 is under 1% of price are skipped (min_atr_pct).
#
# SINGLE PASS — 1 daily download per ticker, purely technical.
#
# ============================================================
'''

LOGIC = r'''# ── CONFIG ────────────────────────────────────────────────────────────
CFG = {
    "history_days"          : 550,   # daily bars — ~380 trading days, above _clean()'s 220-bar floor

    # ── Indicators ─────────────────────────────────────────────────
    "sma_period"             : 50,
    "ema_period"             : 21,
    "atr_period"             : 14,

    # ── Pattern ────────────────────────────────────────────────────
    "touch_tol_atr"          : 0.1,   # step 1: low at/below the MA, or within this x ATR above it
    "min_wick_atr"           : 0.75,  # step 1: lower wick (body bottom - low) at least this x ATR
    "body_mult"              : 4.0,   # step 2: today's body MORE than this x the sweep-day body
    "min_body_atr"           : 0.5,   # step 2: today's body at least this x ATR (default guard)

    "signal_lookback_days"   : 1,     # 1 = the strong green candle must be the latest candle

    # ── Universe ───────────────────────────────────────────────────
    "min_avg_volume"         : 100_000,
    "min_price"              : 10.0,
    "min_atr_pct"            : 1.0,   # skip stocks moving < 1% a day (e.g. pinned by a takeover)

    "batch_size"             : 50,
    "batch_sleep"            : 1.5,
}

# ── Indicators ──────────────────────────────────────────────────────
def calc_atr(h, l, c, n):
    """Simple average of True Range over n days."""
    pc = np.concatenate([[np.nan], c[:-1]])
    tr = np.nanmax(np.vstack([h - l, np.abs(h - pc), np.abs(l - pc)]), axis=0)
    return pd.Series(tr).rolling(n).mean().values

# ── Pattern — single source of truth ────────────────────────────────
def check_pattern_at(o, h, l, c, sma, ema, atr, i, cfg):
    """
    Checks the MA sweep + 4x green candle with "today" (the green candle)
    at bar `i` and the sweep day at bar i-1.
    SINGLE SOURCE OF TRUTH for the pattern logic.

    Computes EVERY step's pass/fail regardless of earlier failures
    (guarding only on missing data), so callers can diagnose exactly
    which step is filtering results out — see FUNNEL_COUNTS below.

    Returns (passed: bool, details: dict).
    """
    a = i - 1
    if a < 0 or i >= len(c):
        return False, {}
    vals = [sma[a], ema[a], sma[i], ema[i], atr[a], atr[i]]
    if not all(np.isfinite(v) for v in vals) or atr[a] <= 0:
        return False, {}

    swept = []                                                          # Step 1
    depth = {}
    wick_atr = (min(o[a], c[a]) - l[a]) / atr[a]
    long_wick = wick_atr >= cfg["min_wick_atr"]
    for name, m in (("SMA50", sma), ("EMA21", ema)):
        if long_wick and l[a] <= m[a] + cfg["touch_tol_atr"] * atr[a] and c[a] > m[a]:
            swept.append(name)
            depth[name] = (m[a] - l[a]) / atr[a]       # how far the low went past the MA (ATR)
    has_sweep = len(swept) > 0

    body_a = abs(c[a] - o[a])
    body_i = c[i] - o[i]
    green  = bool(c[i] > o[i])
    ratio  = (body_i / body_a) if body_a > 0 else (np.inf if body_i > 0 else 0.0)
    strong = bool(green and ratio > cfg["body_mult"]
                  and body_i >= cfg["min_body_atr"] * atr[a])              # Step 2

    ma_now = {"SMA50": sma[i], "EMA21": ema[i]}
    holds  = [nm for nm in swept if c[i] > ma_now[nm]]
    above  = bool(has_sweep and len(holds) > 0)                            # Step 3

    passed = has_sweep and strong and above
    return passed, {
        "idx": i, "sweep_idx": a, "swept": holds if holds else swept,
        "depth": max(depth[nm] for nm in (holds if holds else swept)) if has_sweep else np.nan,
        "wick_atr": float(wick_atr),
        "body_ratio": float(ratio), "body_atr": float(body_i / atr[a]),
        "close_pos": float((c[i] - l[i]) / (h[i] - l[i])) if h[i] > l[i] else 1.0,
        "has_sweep": has_sweep, "strong": strong, "above": above,
    }

# ── Diagnostic funnel — tallies how far each ticker-day gets through
#    the pattern, across the full universe scan, so a 0-match run can
#    be diagnosed empirically instead of guessed at ─────────────────
FUNNEL_COUNTS = {
    "days_checked": 0,
    "passed_step1_sweep_of_ma": 0,
    "passed_step2_green_body_4x": 0,
    "passed_step3_close_above_ma": 0,
}

def find_ma_sweep_signals(o, h, l, c, sma, ema, atr, cfg):
    """
    Checks the last `signal_lookback_days` trading days (1 = the most
    recent candle only). Returns hit dicts, most recent first, and
    tallies FUNNEL_COUNTS for every day checked.
    """
    global FUNNEL_COUNTS
    n = len(c)
    hits = []
    for back in range(0, cfg["signal_lookback_days"]):
        i = (n - 1) - back
        passed, d = check_pattern_at(o, h, l, c, sma, ema, atr, i, cfg)
        if not d:
            continue
        FUNNEL_COUNTS["days_checked"] += 1
        if d["has_sweep"]:
            FUNNEL_COUNTS["passed_step1_sweep_of_ma"] += 1
            if d["strong"]:
                FUNNEL_COUNTS["passed_step2_green_body_4x"] += 1
                if d["above"]:
                    FUNNEL_COUNTS["passed_step3_close_above_ma"] += 1
        if passed:
            hits.append(d)
    return hits

# ── Technical signal: MA sweep + 4x green candle ─────────────────────
def analyze_ma_sweep_4x(sym, df_daily):
    """
    Returns dict with score and setup details, or None if the pattern
    is not present on the checked candle(s).
    """
    if df_daily is None:
        return None

    price   = float(df_daily["Close"].iloc[-1])
    avg_vol = float(df_daily["Volume"].tail(20).mean())
    if price   < CFG["min_price"]:      return None
    if avg_vol < CFG["min_avg_volume"]: return None
    if len(df_daily) < CFG["sma_period"] + 20:
        return None

    o, h, l, c = (df_daily[x].values.astype(float) for x in ("Open", "High", "Low", "Close"))
    close = df_daily["Close"]
    sma = close.rolling(CFG["sma_period"]).mean()
    ema = close.ewm(span=CFG["ema_period"], adjust=False).mean()
    atr = calc_atr(h, l, c, CFG["atr_period"])
    if not np.isfinite(atr[-1]) or atr[-1] / price * 100 < CFG["min_atr_pct"]:
        return None

    hits = find_ma_sweep_signals(o, h, l, c, sma.values, ema.values, atr, CFG)
    if not hits:
        return None

    s = hits[0]   # most recent
    k, a = s["idx"], s["sweep_idx"]
    days_since_signal = (len(c) - 1) - k

    # ── Entry / Stop (reasonable defaults — see header note) ─────────
    entry_price = float(c[k])
    stop_loss   = float(l[a])                     # the sweep-day low
    if entry_price <= stop_loss:
        return None
    risk_pct = (entry_price - stop_loss) / entry_price * 100

    vol   = df_daily["Volume"].values.astype(float)
    avg20 = float(np.mean(vol[max(0, k - 20):k])) if k > 0 else 0.0
    vol_vs_avg = vol[k] / avg20 if avg20 > 0 else 0.0
    ratio = min(99.0, s["body_ratio"])

    # ── Score (0-100) ─────────────────────────────────────────────────
    score = 40
    reasons = [f"Swept {'+'.join(s['swept'])}"]
    score += max(0, min(20, (s["wick_atr"] - CFG["min_wick_atr"]) * 15 + 5))
    reasons.append(f"Wick {s['wick_atr']:.1f} ATR")
    score += min(15, (min(ratio, 12) - CFG["body_mult"]) * 2 + 5)
    reasons.append(f"Body x{ratio:.1f}")
    if len(s["swept"]) == 2:
        score += 10; reasons.append("Both MAs")
    score += s["close_pos"] * 10
    if vol_vs_avg >= 1.5:
        score += 5; reasons.append(f"Vol x{vol_vs_avg:.1f}")
    if risk_pct > 12:
        score -= 10; reasons.append("WideStop")
    score = round(min(100, max(0, score)))

    ma_name = "+".join(s["swept"])
    return {
        "Score"          : score,
        "Price"          : round(price, 2),
        "Entry_Price"    : round(entry_price, 2),
        "Stop_Loss"      : round(stop_loss, 2),
        "Risk_%"         : round(risk_pct, 1),
        "MA_Swept"       : ma_name,
        "Sweep_Date"     : df_daily.index[a].strftime("%Y-%m-%d"),
        "Sweep_Low"      : round(float(l[a]), 2),
        "Wick_ATR"       : round(s["wick_atr"], 2),
        "Low_vs_MA_ATR"  : round(-s["depth"], 2),
        "Body_Ratio"     : round(ratio, 1),
        "Body_ATR"       : round(s["body_atr"], 2),
        "Close_In_Range" : round(s["close_pos"], 2),
        "Vol_vs_20d_Avg" : round(vol_vs_avg, 2),
        "SMA50"          : round(float(sma.iloc[k]), 2),
        "EMA21"          : round(float(ema.iloc[k]), 2),
        "Signal_Date"    : df_daily.index[k].strftime("%Y-%m-%d"),
        "Days_Since_Signal": days_since_signal,
        "Flags"          : " | ".join(reasons),
        "_df_daily"      : df_daily,
        "_sma"           : sma,
        "_ema"           : ema,
        "_idx"           : k,
        "_sweep_idx"     : a,
    }
'''

LIVE = r'''# ── Live print ────────────────────────────────────────────────
LIVE_COLS = ["Ticker","Price","Score","Entry_Price","Stop_Loss","MA_Swept","Body_Ratio","Sweep_Date"]
_CW = {"Ticker":8,"Price":10,"Score":7,"Entry_Price":12,"Stop_Loss":11,
       "MA_Swept":13,"Body_Ratio":11,"Sweep_Date":12}
_CF = {"Price":"${:.2f}","Score":"{:.0f}","Entry_Price":"${:.2f}",
       "Stop_Loss":"${:.2f}","Body_Ratio":"x{:.1f}"}
_hdr_done = False

def _live_header():
    global _hdr_done
    if _hdr_done: return
    print("\n" + "━"*95)
    print("  📊  LIVE MATCHES  —  each stock printed the moment it passes all 3 steps")
    print("━"*95)
    h = "".join(f"  {c:<{_CW.get(c,12)}}" for c in LIVE_COLS)
    print(h)
    print("  " + "─"*93)
    _hdr_done = True

def live_print(r):
    _live_header()
    row = ""
    for c in LIVE_COLS:
        val = r.get(c,"—")
        w   = _CW.get(c,12)
        fmt = _CF.get(c)
        try:   s = fmt.format(val) if (fmt and val not in("—",None)) else str(val)
        except Exception: s = str(val)
        row += f"  {s:<{w}}"
    print(row)
'''

TAIL = r'''
# ── Main scan — single pass (daily download only, then check) ────
print("━"*65)
print(f"  STEP 3  SCANNING {len(TICKERS)} TICKERS")
print("━"*65)
print("  Fetching daily bars (single download per ticker)")
print(f"  Signal: yesterday a long lower wick (>= {CFG['min_wick_atr']} ATR) touched SMA{CFG['sma_period']} or "
      f"EMA{CFG['ema_period']} and closed above it; today a green candle with a body "
      f"> {CFG['body_mult']:g}x yesterday's\n")

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

for sym in tqdm(list(daily_map.keys()), desc="Checking MA sweep + 4x candle", unit="stk"):
    try:
        r = analyze_ma_sweep_4x(sym, daily_map[sym])
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
print(f"  Ticker-days checked                          : {fc['days_checked']}")
print(f"  Step 1 — sweep of SMA{CFG['sma_period']}/EMA{CFG['ema_period']} yesterday       : {fc['passed_step1_sweep_of_ma']}")
print(f"  Step 2 — + green body > {CFG['body_mult']:g}x yesterday's       : {fc['passed_step2_green_body_4x']}")
print(f"  Step 3 — + close still above the swept MA    : {fc['passed_step3_close_above_ma']}  (= full pattern)")
print(f"{'━'*65}")

if not results:
    print("\n  No matches. Try relaxing (see the FUNNEL above to see which")
    print("  step is actually the bottleneck before guessing):")
    print("   signal_lookback_days            1 → 3     (green candle in the last 3 days)")
    print("   min_wick_atr                 0.75 → 0.5")
    print("   touch_tol_atr                 0.1 → 0.25")
    print("   body_mult                     4.0 → 3.0")
    print("   min_price                      10 → 5")

results.sort(key=lambda x: x["Score"], reverse=True)

# ── Always build df_out and save/email (even if 0 results) ────
ts      = datetime.today().strftime("%Y%m%d_%H%M")
out_dir = os.environ.get("GITHUB_WORKSPACE", os.getcwd())

COLS = [
    "Ticker","Price","Score",
    "Entry_Price","Stop_Loss","Risk_%",
    "MA_Swept","Sweep_Date","Sweep_Low","Wick_ATR","Low_vs_MA_ATR",
    "Body_Ratio","Body_ATR","Close_In_Range","Vol_vs_20d_Avg",
    "SMA50","EMA21","Signal_Date","Days_Since_Signal","Flags",
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
    "Sweep_Low"     : lambda v: f"${v:.2f}",
    "SMA50"         : lambda v: f"${v:.2f}",
    "EMA21"         : lambda v: f"${v:.2f}",
    "Risk_%"        : lambda v: f"{v:.1f}%",
    "Wick_ATR"      : lambda v: f"{v:.2f}",
    "Low_vs_MA_ATR" : lambda v: f"{v:+.2f}",
    "Body_Ratio"    : lambda v: f"x{v:.1f}",
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
             "MA_Swept","Sweep_Date","Wick_ATR","Body_Ratio","Vol_vs_20d_Avg"]

if results:
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
    tit = f"  Daily MA Sweep + 4x Green Candle   {datetime.today().strftime('%Y-%m-%d')}   {len(df_out)} matches"
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

# Save
fpath = os.path.join(out_dir, f"daily_ma_sweep_4x_{ts}.csv")
# ── Position sizing: Quantity / Total_Amount for a fixed max loss (see position_sizing.py)
try:
    from position_sizing import add_position_size
    df_out = add_position_size(df_out)
except Exception as _ps_err:
    print(f"  ⚠️  Position sizing skipped: {_ps_err}")
df_out.to_csv(fpath, index=False)
print(f"\n  💾 CSV → {fpath}")
tv = os.path.join(out_dir, f"tv_daily_ma_sweep_4x_{ts}.txt")
with open(tv,"w") as f:
    f.write(f"###Daily MA Sweep + 4x Green Candle {datetime.today().strftime('%Y-%m-%d')}\n")
    for r in results: f.write(f"NASDAQ:{r['Ticker']}\n")
print(f"  📋 TradingView → {tv}")
if results:
    print(f"\n  📋 Tickers (comma-separated):")
    print(f"  {', '.join(r['Ticker'] for r in results)}")

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

        heads = ["Ticker","Price","Score","Entry","Stop","Risk %","MA swept",
                 "Sweep day","Wick (ATR)","Body x","Qty","Amount"]
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
                + td(r.get("MA_Swept","—"), "font-weight:700;color:#7c3aed")
                + td(r.get("Sweep_Date","—"))
                + td(f'{float(r.get("Wick_ATR",0) or 0):.2f}')
                + td(f'x{float(r.get("Body_Ratio",0) or 0):.1f}', "color:#16a34a;font-weight:600")
                + td(f'{q}')
                + td(f'${float(a):,.0f}' if a is not None else "—")
                + '</tr>')
        no_results_msg = ('<tr><td colspan="12" style="padding:20px;text-align:center;'
                          'color:#94a3b8;font-size:13px">No matches today</td></tr>')
        more_note = (f'Showing top 50 of {cnt} — full list in the CSV attachment'
                     if cnt > 50 else '📎 Full results attached as CSV')
        tick_box = ('<div style="margin-top:10px;background:#f8fafc;border:1px solid #e2e8f0;'
                    'border-radius:6px;padding:10px 14px">'
                    '<p style="margin:0 0 4px;font-size:10px;color:#94a3b8;font-weight:700">'
                    'TICKERS (comma-separated, copy/paste)</p>'
                    '<p style="margin:0;font-size:12px;color:#1e293b;font-family:monospace;word-break:break-all">'
                    + ", ".join(r.get("Ticker","") for r in rl) + '</p></div>') if rl else ''

        fc = FUNNEL_COUNTS
        html_e = f"""<!DOCTYPE html><html><body style="margin:0;padding:0;
background:#f1f5f9;font-family:'Segoe UI',Arial,sans-serif">
<table width="100%" cellpadding="0" cellspacing="0"><tr><td align="center" style="padding:20px 10px">
<table width="100%" style="max-width:860px;background:#fff;border-radius:12px;
       overflow:hidden;box-shadow:0 4px 16px rgba(0,0,0,0.08)">
  <tr><td style="background:linear-gradient(135deg,#0f172a,#1e3a5f);padding:22px 28px">
<h1 style="margin:0;color:#60a5fa;font-size:20px;font-weight:700">
  📊 Daily MA Sweep + 4x Green Candle
</h1>
<p style="margin:6px 0 0;color:#94a3b8;font-size:12px">
  {datetime.today().strftime('%Y-%m-%d %H:%M UTC')} &nbsp;·&nbsp;
  {cnt} match{'es' if cnt!=1 else ''} — yesterday a long lower wick touched SMA{CFG['sma_period']} or EMA{CFG['ema_period']}
  and closed above it; today a green candle with a body &gt; {CFG['body_mult']:g}x yesterday's
</p>
  </td></tr>
  <tr><td style="padding:14px 28px 4px;background:#0b1220">
<div style="background:#111827;border:1px solid #1f2937;border-radius:8px;padding:12px 16px">
  <p style="margin:0 0 6px;color:#93c5fd;font-size:12px;font-weight:700">
    🔍 FUNNEL — last {CFG['signal_lookback_days']} trading day(s), every liquid ticker
  </p>
  <p style="margin:0;color:#cbd5e1;font-size:12px">
    {fc['days_checked']} ticker-days &nbsp;→&nbsp;
    {fc['passed_step1_sweep_of_ma']} swept an MA &nbsp;→&nbsp;
    {fc['passed_step2_green_body_4x']} green body &gt; {CFG['body_mult']:g}x &nbsp;→&nbsp;
    <b style="color:#facc15">{fc['passed_step3_close_above_ma']} still above the MA = full match</b>
  </p>
</div>
  </td></tr>
  <tr><td style="padding:16px">
<div style="overflow-x:auto;border-radius:8px;border:1px solid #e2e8f0">
  <table style="border-collapse:collapse;width:100%;min-width:720px">
    <thead><tr>{th_e}</tr></thead>
    <tbody>{rows_e or no_results_msg}</tbody>
  </table>
</div>
<p style="font-size:11px;color:#64748b;margin:8px 0 0">{more_note}</p>
{tick_box}
  </td></tr>
  <tr><td style="background:#f8fafc;padding:12px 28px;
             border-top:1px solid #e2e8f0;text-align:center">
<p style="margin:0;color:#94a3b8;font-size:10px">
  ⚠️ Not financial advice &nbsp;·&nbsp; Entry = today's close, Stop = sweep-day low (defaults) &nbsp;·&nbsp; Qty sized for a $40 max loss &nbsp;·&nbsp; Auto-generated by GitHub Actions
</p>
  </td></tr>
</table>
</td></tr></table>
</body></html>"""

        plain_lines = [
            f"Daily MA Sweep + 4x Green Candle — {datetime.today().strftime('%Y-%m-%d')}",
            f"{cnt} matches (sweep of SMA{CFG['sma_period']}/EMA{CFG['ema_period']} yesterday, green body > {CFG['body_mult']:g}x today)",
            "="*60,
            f"FUNNEL: {fc['days_checked']} ticker-days -> "
            f"{fc['passed_step1_sweep_of_ma']} swept -> "
            f"{fc['passed_step2_green_body_4x']} body 4x -> "
            f"{fc['passed_step3_close_above_ma']} above MA (=full match)",
            "="*60,
        ]
        if rl:
            for r in rl[:50]:
                plain_lines.append(
                    f"{r.get('Ticker','—'):<7} ${float(r.get('Price',0) or 0):.2f}  "
                    f"Score:{float(r.get('Score',0) or 0):.0f}  "
                    f"Entry:${float(r.get('Entry_Price',0) or 0):.2f}  "
                    f"SL:${float(r.get('Stop_Loss',0) or 0):.2f}  "
                    f"Risk:{float(r.get('Risk_%',0) or 0):.1f}%  "
                    f"{r.get('MA_Swept','—')} swept {r.get('Sweep_Date','—')}  "
                    f"Body:x{float(r.get('Body_Ratio',0) or 0):.1f}  "
                    f"Qty:{_SIZE.get(r.get('Ticker'), ('—', None))[0]}")
            plain_lines.append("")
            plain_lines.append("Tickers (comma-separated):")
            plain_lines.append(", ".join(r.get("Ticker","") for r in rl))
        else:
            plain_lines.append("No matches today")
        plain_lines.append("\nFull results in CSV attachment.")
        plain_e = "\n".join(plain_lines)

        subj = (f"📊 MA Sweep + 4x Candle — {cnt} signal{'s' if cnt!=1 else ''}"
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

# ── Charts for top 5: last 60 daily candles with SMA50 / EMA21, the sweep
#    day and the 4x green candle marked ───────────────────────────────
if results:
    top = results[:min(5,len(results))]
    fig, axes = plt.subplots(len(top), 1, figsize=(13, 4.2*len(top)), facecolor="#0f172a", squeeze=False)
    axes = axes[:,0]
    for p, r in enumerate(top):
        ax = axes[p]
        df_all = r["_df_daily"]
        df_p = df_all.tail(60)
        idx  = df_p.index
        k, a = r["_idx"], r["_sweep_idx"]
        sig_d, sw_d = df_all.index[k], df_all.index[a]
        x = np.arange(len(df_p))
        up = (df_p["Close"] >= df_p["Open"]).values
        col = np.where(up, "#22c55e", "#ef4444")

        ax.set_facecolor("#0f172a")
        ax.vlines(x, df_p["Low"], df_p["High"], color=col, lw=0.8)
        ax.bar(x, (df_p["Close"] - df_p["Open"]).abs().clip(lower=df_p["Close"]*0.0005),
               bottom=np.minimum(df_p["Open"], df_p["Close"]), color=col, width=0.6)
        ax.plot(x, r["_sma"].reindex(idx).values, color="#fbbf24", lw=1.1, label="SMA50")
        ax.plot(x, r["_ema"].reindex(idx).values, color="#60a5fa", lw=1.1, label="EMA21")
        ax.axhline(r["Stop_Loss"], color="#ef4444", lw=0.9, ls="--", alpha=0.8, label=f"Stop {r['Stop_Loss']:.2f}")
        pos = {d: i for i, d in enumerate(idx)}
        if sw_d in pos:
            ax.scatter([pos[sw_d]], [df_all["Low"].iloc[a]], marker="^", color="#a78bfa", s=60, zorder=7, label="Sweep")
        if sig_d in pos:
            ax.scatter([pos[sig_d]], [df_all["High"].iloc[k]], marker="v", color="#facc15", s=60, zorder=7, label="4x green")
        step = max(1, len(idx) // 8)
        ax.set_xticks(x[::step]); ax.set_xticklabels([d.strftime("%b %d") for d in idx[::step]])
        ax.set_title(f"{r['Ticker']}  |  {r['Price']:.2f} USD  |  Score {r['Score']}  |  swept {r['MA_Swept']} on "
                     f"{r['Sweep_Date']}  |  body x{r['Body_Ratio']:.1f}",
                     color="#e2e8f0", fontsize=9, fontweight="bold", pad=7)
        ax.tick_params(colors="#94a3b8", labelsize=8)
        for sp in ax.spines.values(): sp.set_edgecolor("#1e3a5f")
        ax.legend(loc="upper left", facecolor="#1e293b", labelcolor="#e2e8f0", fontsize=6, ncol=5)
        ax.grid(color="#1e3a5f", ls="--", lw=0.5, alpha=0.6)
    plt.suptitle(f"Daily MA Sweep + 4x Green Candle  ·  {datetime.today().strftime('%Y-%m-%d')}",
                 color="#60a5fa", fontsize=12, fontweight="bold", y=1.001)
    plt.tight_layout()
    cp = os.path.join(out_dir, f"daily_ma_sweep_4x_chart_{ts}.png")
    plt.savefig(cp, dpi=150, bbox_inches="tight", facecolor="#0f172a")
    if _IN_NOTEBOOK: plt.show()
    else: plt.close()
    print(f"  📊 Chart → {cp}")

print(f"""
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  📋 SIGNAL (all required; the green candle is the latest daily candle)
  1) Sweep day (previous candle): lower wick at least {CFG['min_wick_atr']} x ATR{CFG['atr_period']}, low
     touching SMA{CFG['sma_period']} or EMA{CFG['ema_period']} (at/below it, or within {CFG['touch_tol_atr']} ATR above),
     close ABOVE that same average.
  2) Today: green candle whose body is MORE than {CFG['body_mult']:g}x the sweep day's
     body (and at least {CFG['min_body_atr']} x ATR, so a doji the day before is not enough).
  3) Today's close is still above the swept average.

  📋 OUTPUT (reasonable defaults — not explicitly requested)
  Entry_Price = today's CLOSE
  Stop_Loss   = sweep-day LOW
  Universe    = price >= 10 USD, 20d avg volume >= 100k

  📋 SCORE (0-100)
  Base 40 + wick length (0-20) + body ratio (0-15) + both MAs swept (10)
  + close near the high (0-10) + volume >= 1.5x avg (5); -10 if stop risk > 12%
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
""")
'''

src = (HEADER + "\n" + "\n".join(L[37:117]) + "\n" + LOGIC + "\n" + "\n".join(L[321:368]) + "\n"
       + LIVE + "\n" + "\n".join(L[399:477]) + "\n" + TAIL)
for bad in ("rsi2", "RSI(2)", "Snap-back", "snapback", "Exit_By"):
    assert bad not in src, f"template leftover: {bad}"
open(OUT, "w").write(src)
py_compile.compile(OUT, doraise=True)
print(f"wrote {OUT} ({len(src.splitlines())} lines)")
