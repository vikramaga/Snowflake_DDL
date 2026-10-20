# Generates daily_rsi2_snapback_v1.py: the 80%+ win-rate pullback strategy
# found by claude_run/research/mr_highwin_research.py, as a screener whose
# email gives the complete order plan (limit buy, target, stop, exit-by date,
# quantity). Boilerplate, download and ticker code are copied unchanged from
# daily_rsi_avg_bounce_v1.py. Usage: python gen_rsi2_snapback.py [out]
import py_compile, sys

SRC = "daily_rsi_avg_bounce_v1.py"
OUT = sys.argv[1] if len(sys.argv) > 1 else "daily_rsi2_snapback_v1.py"
L = open(SRC).read().split("\n")
assert L[36].startswith("import subprocess"), "template layout changed (boilerplate)"
assert L[116].startswith("# ── CONFIG"), "template layout changed (config)"
assert L[355].startswith("# ── Download: daily") and L[400].strip() == "return out", "template layout changed (download)"
assert L[434].startswith("# ── Health check") and L[510] == "TICKERS = get_tickers()", "template layout changed (tickers)"

HEADER = '''# ============================================================
# NASDAQ — Daily RSI(2) Snap-back, high win rate (v1)
# ============================================================
#
# A fully mechanical pullback strategy: every number you need to trade
# it is in the email, so no chart reading is required.
#
# SIGNAL (all required, on the latest daily close)
#   1. RSI(2) < 3            (Wilder RSI, 2 periods: a sharp short-term drop)
#   2. ATR(10) >= 3% of price (enough daily range for the bounce target)
#   3. 20-day average traded value >= $20M (orders fill cleanly)
#   Universe: price >= $10, 20-day average volume >= 100k.
#
# ORDER PLAN (all prices computed by the scanner)
#   Buy    : LIMIT at close - 0.5 x ATR10, valid for the next session only.
#            Not filled -> cancel, no trade.
#   Target : LIMIT SELL at fill + 0.5 x ATR10 (good till cancelled).
#   Stop   : STOP at fill - 2.5 x ATR10.
#   Time   : if neither has hit after 10 sessions (fill day = session 1),
#            sell at the open on the Exit_By date.
#   If the buy fills BELOW the limit (gap down), move target and stop by
#   the same dollar offsets from the actual fill (Target_Offset / Stop_Offset).
#
# BACKTEST (claude_run/research/mr_highwin_research.py, ~1,700 random US
# stocks, Jan 2024 - Oct 2026, next-day execution, 0.10% costs, one trade
# per stock at a time): train year 85.6% wins / +1.07% per trade, unseen
# test year 83.1% wins / +0.95% per trade. Average win about +2.9%,
# average loss about -8.7% (one loss costs about three wins); worst single
# trade -29% (gap through the stop). Take every signal, not a selection.
#
# Quantity / Total_Amount come from position_sizing.py (max loss $40,
# measured from the limit buy to the stop).
#
# SINGLE PASS — 1 daily download per ticker, purely technical.
#
# ============================================================
'''

LOGIC = r'''# ── CONFIG ────────────────────────────────────────────────────────────
CFG = {
    "history_days"          : 550,   # daily bars — ~380 trading days, above _clean()'s 220-bar floor

    # ── Signal (values tested by the research backtest — change = untested) ──
    "rsi_period"             : 2,
    "rsi_max"                : 3.0,    # step 1: RSI(2) below this
    "atr_period"             : 10,
    "min_atr_pct"            : 3.0,    # step 2: ATR10 at least this % of price
    "min_dollar_vol"         : 20e6,   # step 3: 20-day average close x volume

    # ── Order plan ─────────────────────────────────────────────────
    "buy_atr_below"          : 0.5,    # limit buy = close - 0.5 x ATR
    "target_atr"             : 0.5,    # target    = fill + 0.5 x ATR
    "stop_atr"               : 2.5,    # stop      = fill - 2.5 x ATR
    "max_hold_sessions"      : 10,     # sell at the open after 10 sessions

    "signal_lookback_days"   : 1,      # 1 = latest candle only (the order is for the next session)

    # ── Universe ───────────────────────────────────────────────────
    "min_avg_volume"         : 100_000,
    "min_price"              : 10.0,

    "batch_size"             : 50,
    "batch_sleep"            : 1.5,
}

# ── Indicators (identical to the research backtest) ──────────────────────
def wilder_rsi(c, n):
    """Wilder RSI on a numpy array of closes."""
    d = np.diff(c, prepend=np.nan)
    g = pd.Series(np.where(d > 0, d, 0.0)).ewm(alpha=1/n, adjust=False).mean().values
    l = pd.Series(np.where(d < 0, -d, 0.0)).ewm(alpha=1/n, adjust=False).mean().values
    with np.errstate(divide="ignore", invalid="ignore"):
        r = 100 - 100 / (1 + g / l)
    r[l == 0] = 100.0
    r[:n] = np.nan
    return r

def calc_atr(h, l, c, n):
    """Simple average of True Range over n days."""
    pc = np.concatenate([[np.nan], c[:-1]])
    tr = np.nanmax(np.vstack([h - l, np.abs(h - pc), np.abs(l - pc)]), axis=0)
    return pd.Series(tr).rolling(n).mean().values

# ── NYSE sessions (for the buy date and the exit-by date) ──────────────────
from pandas.tseries.holiday import (AbstractHolidayCalendar, Holiday, nearest_workday,
                                    USMartinLutherKingJr, USPresidentsDay, GoodFriday,
                                    USMemorialDay, USLaborDay, USThanksgivingDay)
from pandas.tseries.offsets import CustomBusinessDay

class _NYSECalendar(AbstractHolidayCalendar):
    rules = [
        Holiday("NewYearsDay", month=1, day=1, observance=nearest_workday),
        USMartinLutherKingJr, USPresidentsDay, GoodFriday, USMemorialDay,
        Holiday("Juneteenth", month=6, day=19, start_date="2022-01-01", observance=nearest_workday),
        Holiday("IndependenceDay", month=7, day=4, observance=nearest_workday),
        USLaborDay, USThanksgivingDay,
        Holiday("Christmas", month=12, day=25, observance=nearest_workday),
    ]

def sessions_after(date, n):
    """The NYSE session n sessions after `date`."""
    return (pd.Timestamp(date) + n * CustomBusinessDay(calendar=_NYSECalendar())).normalize()

# ── Pattern — single source of truth ─────────────────────────────────────────────
def check_pattern_at(rsi2, atr, dvol20, close, i, cfg):
    """
    Checks the RSI(2) snap-back signal with "today" at bar `i`.
    SINGLE SOURCE OF TRUTH for the signal logic.

    Computes EVERY step's pass/fail regardless of earlier failures
    (guarding only on missing data), so callers can diagnose exactly
    which step is filtering results out — see FUNNEL_COUNTS below.

    Returns (passed: bool, details: dict).
    """
    if i < 0 or i >= len(close) or not np.isfinite(rsi2[i]) or not np.isfinite(atr[i]) \
            or not np.isfinite(dvol20[i]) or close[i] <= 0:
        return False, {}
    atr_pct = atr[i] / close[i] * 100
    oversold = bool(rsi2[i] < cfg["rsi_max"])                    # Step 1
    volatile = bool(atr_pct >= cfg["min_atr_pct"])               # Step 2
    liquid   = bool(dvol20[i] >= cfg["min_dollar_vol"])          # Step 3
    passed = oversold and volatile and liquid
    return passed, {
        "idx": i, "rsi2": float(rsi2[i]), "atr": float(atr[i]), "atr_pct": float(atr_pct),
        "dvol20": float(dvol20[i]), "close": float(close[i]),
        "oversold": oversold, "volatile": volatile, "liquid": liquid,
    }

# ── Diagnostic funnel — tallies how far each ticker-day gets through
#    the signal, across the full universe scan, so a 0-match run can
#    be diagnosed empirically instead of guessed at ─────────────────
FUNNEL_COUNTS = {
    "days_checked": 0,
    "passed_step1_rsi2_below_3": 0,
    "passed_step2_atr_at_least_3pct": 0,
    "passed_step3_traded_value_20m": 0,
}

def find_snapback_signals(rsi2, atr, dvol20, close, cfg):
    """
    Checks the last `signal_lookback_days` trading days (1 = the most
    recent candle only). Returns hit dicts, most recent first, and
    tallies FUNNEL_COUNTS for every day checked.
    """
    global FUNNEL_COUNTS
    n = len(close)
    hits = []
    for back in range(0, cfg["signal_lookback_days"]):
        i = (n - 1) - back
        passed, d = check_pattern_at(rsi2, atr, dvol20, close, i, cfg)
        if not d:
            continue
        FUNNEL_COUNTS["days_checked"] += 1
        if d["oversold"]:
            FUNNEL_COUNTS["passed_step1_rsi2_below_3"] += 1
            if d["volatile"]:
                FUNNEL_COUNTS["passed_step2_atr_at_least_3pct"] += 1
                if d["liquid"]:
                    FUNNEL_COUNTS["passed_step3_traded_value_20m"] += 1
        if passed:
            hits.append(d)
    return hits

# ── Technical signal: RSI(2) snap-back with the full order plan ────────
def analyze_rsi2_snapback(sym, df_daily):
    """
    Returns dict with score and the order plan, or None if there is no
    signal on the checked candle(s).
    """
    if df_daily is None:
        return None

    price   = float(df_daily["Close"].iloc[-1])
    avg_vol = float(df_daily["Volume"].tail(20).mean())
    if price   < CFG["min_price"]:      return None
    if avg_vol < CFG["min_avg_volume"]: return None
    if len(df_daily) < 60:
        return None

    o, h, l, c, v = (df_daily[x].values.astype(float) for x in ("Open", "High", "Low", "Close", "Volume"))
    rsi2   = wilder_rsi(c, CFG["rsi_period"])
    atr    = calc_atr(h, l, c, CFG["atr_period"])
    dvol20 = pd.Series(c * v).rolling(20).mean().values

    hits = find_snapback_signals(rsi2, atr, dvol20, c, CFG)
    if not hits:
        return None

    s = hits[0]   # most recent
    k = s["idx"]
    days_since_signal = (len(c) - 1) - k
    sig_date = df_daily.index[k]

    # ── Order plan ──────────────────────────────────────────────────────
    a = s["atr"]
    buy    = s["close"] - CFG["buy_atr_below"] * a
    t_off  = CFG["target_atr"] * a
    s_off  = CFG["stop_atr"] * a
    target = buy + t_off
    stop   = buy - s_off
    if stop <= 0 or buy <= 0:
        return None
    buy_date  = sessions_after(sig_date, 1)
    exit_date = sessions_after(buy_date, CFG["max_hold_sessions"])
    risk_pct   = s_off / buy * 100
    reward_pct = t_off / buy * 100

    # ── Score (0-100) — for ORDERING only; the research did not test ──
    #    whether a higher score wins more often. Lower RSI(2) and more
    #    liquidity rank first.
    score = 70 + (CFG["rsi_max"] - s["rsi2"]) / CFG["rsi_max"] * 20 \
            + min(10, max(0, (np.log10(s["dvol20"]) - np.log10(CFG["min_dollar_vol"])) * 10))
    score = round(min(100, max(0, score)))
    flags = [f"RSI2 {s['rsi2']:.1f}", f"ATR {s['atr_pct']:.1f}%", f"Value ${s['dvol20']/1e6:,.0f}M/day"]
    if risk_pct > 15:
        flags.append("WideStop")

    return {
        "Score"          : score,
        "Price"          : round(price, 2),
        "Entry_Price"    : round(buy, 2),
        "Stop_Loss"      : round(stop, 2),
        "Target"         : round(target, 2),
        "Risk_%"         : round(risk_pct, 1),
        "Reward_%"       : round(reward_pct, 1),
        "Buy_Date"       : buy_date.strftime("%Y-%m-%d"),
        "Exit_By"        : exit_date.strftime("%Y-%m-%d"),
        "Target_Offset"  : round(t_off, 2),
        "Stop_Offset"    : round(s_off, 2),
        "RSI2"           : round(s["rsi2"], 2),
        "ATR10"          : round(a, 2),
        "ATR_%"          : round(s["atr_pct"], 1),
        "Value_20d_$M"   : round(s["dvol20"] / 1e6, 1),
        "Signal_Date"    : sig_date.strftime("%Y-%m-%d"),
        "Days_Since_Signal": days_since_signal,
        "Flags"          : " | ".join(flags),
        "_df_daily"      : df_daily,
        "_rsi2"          : pd.Series(rsi2, index=df_daily.index),
        "_idx"           : k,
    }
'''

LIVE = r'''# ── Live print ────────────────────────────────────────────────
LIVE_COLS = ["Ticker","Price","Score","Entry_Price","Target","Stop_Loss","RSI2","ATR_%","Exit_By"]
_CW = {"Ticker":8,"Price":10,"Score":7,"Entry_Price":12,"Target":10,"Stop_Loss":11,
       "RSI2":7,"ATR_%":7,"Exit_By":12}
_CF = {"Price":"${:.2f}","Score":"{:.0f}","Entry_Price":"${:.2f}","Target":"${:.2f}",
       "Stop_Loss":"${:.2f}","RSI2":"{:.1f}","ATR_%":"{:.1f}"}
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
print(f"  Signal: RSI({CFG['rsi_period']}) < {CFG['rsi_max']:g}, ATR{CFG['atr_period']} >= {CFG['min_atr_pct']:g}% of price, "
      f"20d traded value >= ${CFG['min_dollar_vol']/1e6:,.0f}M\n")

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

for sym in tqdm(list(daily_map.keys()), desc="Checking RSI(2) snap-back", unit="stk"):
    try:
        r = analyze_rsi2_snapback(sym, daily_map[sym])
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
print(f"  Step 1 — RSI({CFG['rsi_period']}) below {CFG['rsi_max']:g}                     : {fc['passed_step1_rsi2_below_3']}")
print(f"  Step 2 — + ATR{CFG['atr_period']} >= {CFG['min_atr_pct']:g}% of price              : {fc['passed_step2_atr_at_least_3pct']}")
print(f"  Step 3 — + traded value >= ${CFG['min_dollar_vol']/1e6:,.0f}M/day         : {fc['passed_step3_traded_value_20m']}  (= full signal)")
print(f"{'━'*65}")

if not results:
    print("\n  No signals today. That is normal for this strategy: RSI(2) below 3")
    print("  with enough range and liquidity does not happen every day. The")
    print("  thresholds are the tested ones — relaxing them is untested.")

results.sort(key=lambda x: x["Score"], reverse=True)

# ── Always build df_out and save/email (even if 0 results) ────
ts      = datetime.today().strftime("%Y%m%d_%H%M")
out_dir = os.environ.get("GITHUB_WORKSPACE", os.getcwd())

COLS = [
    "Ticker","Price","Score",
    "Entry_Price","Stop_Loss","Target","Risk_%","Reward_%",
    "Buy_Date","Exit_By","Target_Offset","Stop_Offset",
    "RSI2","ATR10","ATR_%","Value_20d_$M",
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
    "Target"        : lambda v: f"${v:.2f}",
    "Risk_%"        : lambda v: f"{v:.1f}%",
    "Reward_%"      : lambda v: f"{v:.1f}%",
    "RSI2"          : lambda v: f"{v:.1f}",
    "ATR_%"         : lambda v: f"{v:.1f}%",
    "Value_20d_$M"  : lambda v: f"${v:,.0f}M",
    "Days_Since_Signal": lambda v: f"{int(v)}d ago",
}

def fmt_v(col, val):
    if val is None or (isinstance(val, float) and np.isnan(val)): return "—"
    try:
        if col in FMT: return FMT[col](val)
    except Exception: pass
    return str(val) if str(val) not in ("nan","None","") else "—"

DISP_COLS = ["Ticker","Price","Score","Entry_Price","Target","Stop_Loss","Risk_%",
             "Buy_Date","Exit_By","RSI2","ATR_%"]

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
    tit = f"  Daily RSI(2) Snap-back   {datetime.today().strftime('%Y-%m-%d')}   {len(df_out)} signals"
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
fpath = os.path.join(out_dir, f"daily_rsi2_snapback_{ts}.csv")
# ── Position sizing: Quantity / Total_Amount for a fixed max loss (see position_sizing.py)
try:
    from position_sizing import add_position_size
    df_out = add_position_size(df_out)
except Exception as _ps_err:
    print(f"  ⚠️  Position sizing skipped: {_ps_err}")
df_out.to_csv(fpath, index=False)
print(f"\n  💾 CSV → {fpath}")
tv = os.path.join(out_dir, f"tv_daily_rsi2_snapback_{ts}.txt")
with open(tv,"w") as f:
    f.write(f"###Daily RSI(2) Snap-back {datetime.today().strftime('%Y-%m-%d')}\n")
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

        heads = ["Ticker","Close","Buy limit","Target","Stop","Exit by (open)","Qty","Amount",
                 "Target +$","Stop −$","RSI2","ATR %"]
        th_e = "".join(
            f'<th style="background:#1e293b;color:#e2e8f0;padding:8px 11px;'
            f'font-size:11px;font-weight:700;border-bottom:2px solid #3b82f6;'
            f'white-space:nowrap">{c}</th>' for c in heads)

        def td(v, extra=""):
            return f'<td style="padding:6px 11px;font-size:12px;{extra}">{v}</td>'

        rows_e = ""
        for i, r in enumerate(rl[:60]):
            bg = "#fff" if i % 2 == 0 else "#f0f9ff"
            q, a = _SIZE.get(r.get("Ticker"), ("—", None))
            rows_e += (
                f'<tr style="background:{bg}">'
                + td(r.get("Ticker","—"), "font-weight:700")
                + td(f'${float(r.get("Price",0) or 0):.2f}')
                + td(f'${float(r.get("Entry_Price",0) or 0):.2f}', "color:#16a34a;font-weight:700")
                + td(f'${float(r.get("Target",0) or 0):.2f}', "color:#2563eb;font-weight:700")
                + td(f'${float(r.get("Stop_Loss",0) or 0):.2f}', "color:#ef4444;font-weight:700")
                + td(r.get("Exit_By","—"))
                + td(f'{q}', "font-weight:700")
                + td(f'${float(a):,.0f}' if a is not None else "—")
                + td(f'{float(r.get("Target_Offset",0) or 0):.2f}')
                + td(f'{float(r.get("Stop_Offset",0) or 0):.2f}')
                + td(f'{float(r.get("RSI2",0) or 0):.1f}', "color:#7c3aed")
                + td(f'{float(r.get("ATR_%",0) or 0):.1f}%')
                + '</tr>')
        no_results_msg = ('<tr><td colspan="12" style="padding:20px;text-align:center;'
                          'color:#94a3b8;font-size:13px">No signals today — no orders to place</td></tr>')
        more_note = (f'Showing top 60 of {cnt} — full list in the CSV attachment'
                     if cnt > 60 else '📎 Full results attached as CSV')
        buy_day = rl[0].get("Buy_Date", "the next session") if rl else "the next session"
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
<table width="100%" style="max-width:900px;background:#fff;border-radius:12px;
       overflow:hidden;box-shadow:0 4px 16px rgba(0,0,0,0.08)">
  <tr><td style="background:linear-gradient(135deg,#0f172a,#1e3a5f);padding:22px 28px">
<h1 style="margin:0;color:#60a5fa;font-size:20px;font-weight:700">
  📊 Daily RSI(2) Snap-back — order plan
</h1>
<p style="margin:6px 0 0;color:#94a3b8;font-size:12px">
  {datetime.today().strftime('%Y-%m-%d %H:%M UTC')} &nbsp;·&nbsp;
  {cnt} signal{'s' if cnt!=1 else ''} — RSI({CFG['rsi_period']}) &lt; {CFG['rsi_max']:g}, ATR{CFG['atr_period']} &ge; {CFG['min_atr_pct']:g}% of price,
  traded value &ge; ${CFG['min_dollar_vol']/1e6:,.0f}M/day
</p>
  </td></tr>
  <tr><td style="padding:14px 28px 4px">
<div style="background:#f0fdf4;border:1px solid #bbf7d0;border-radius:8px;padding:12px 16px;font-size:12px;color:#14532d">
  <b>What to do (no chart reading needed)</b><br>
  1. For <b>{buy_day}</b> only: place a <b>LIMIT BUY</b> at "Buy limit" for "Qty" shares. Not filled by the close → cancel it.<br>
  2. Once filled: place a <b>LIMIT SELL</b> at "Target" (GTC) and a <b>STOP</b> at "Stop" (an OCO bracket if your broker has one).<br>
  3. If the buy filled <b>below</b> the limit (gap down): Target = fill + "Target +$", Stop = fill − "Stop −$".<br>
  4. If neither has hit, <b>sell at the open</b> on "Exit by". Take every signal — the win rate is for all of them, not a selection.
</div>
  </td></tr>
  <tr><td style="padding:10px 28px 4px;background:#0b1220">
<div style="background:#111827;border:1px solid #1f2937;border-radius:8px;padding:12px 16px">
  <p style="margin:0 0 6px;color:#93c5fd;font-size:12px;font-weight:700">
    🔍 FUNNEL — last {CFG['signal_lookback_days']} trading day(s), every liquid ticker
  </p>
  <p style="margin:0;color:#cbd5e1;font-size:12px">
    {fc['days_checked']} ticker-days &nbsp;→&nbsp;
    {fc['passed_step1_rsi2_below_3']} RSI(2) &lt; {CFG['rsi_max']:g} &nbsp;→&nbsp;
    {fc['passed_step2_atr_at_least_3pct']} ATR &ge; {CFG['min_atr_pct']:g}% &nbsp;→&nbsp;
    <b style="color:#facc15">{fc['passed_step3_traded_value_20m']} traded value &ge; ${CFG['min_dollar_vol']/1e6:,.0f}M = full signal</b>
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
{tick_box}
<p style="font-size:11px;color:#475569;margin:10px 0 0">
  Backtest (≈1,700 US stocks, Jan 2024 – Oct 2026, next-day orders, 0.10% costs): 85.6% wins / +1.07% per trade
  in the training year, 83.1% / +0.95% in the unseen year. Average win ≈ +2.9%, average loss ≈ −8.7%;
  worst trade −29% (gap through the stop). Past results do not guarantee future results.
</p>
  </td></tr>
  <tr><td style="background:#f8fafc;padding:12px 28px;
             border-top:1px solid #e2e8f0;text-align:center">
<p style="margin:0;color:#94a3b8;font-size:10px">
  ⚠️ Not financial advice &nbsp;·&nbsp; Qty sized for a $40 loss from the buy limit to the stop (a gap can lose more) &nbsp;·&nbsp; Auto-generated by GitHub Actions
</p>
  </td></tr>
</table>
</td></tr></table>
</body></html>"""

        plain_lines = [
            f"Daily RSI(2) Snap-back — order plan — {datetime.today().strftime('%Y-%m-%d')}",
            f"{cnt} signals (RSI({CFG['rsi_period']}) < {CFG['rsi_max']:g}, ATR >= {CFG['min_atr_pct']:g}%, value >= ${CFG['min_dollar_vol']/1e6:,.0f}M/day)",
            "="*60,
            f"1. {buy_day} only: LIMIT BUY at Buy; cancel if not filled.",
            "2. Once filled: LIMIT SELL at Target (GTC) + STOP at Stop.",
            "3. Filled below the limit: Target = fill + T+, Stop = fill - S-.",
            "4. Neither hit: sell at the open on Exit-by. Take every signal.",
            "="*60,
            f"FUNNEL: {fc['days_checked']} ticker-days -> "
            f"{fc['passed_step1_rsi2_below_3']} RSI2<{CFG['rsi_max']:g} -> "
            f"{fc['passed_step2_atr_at_least_3pct']} ATR ok -> "
            f"{fc['passed_step3_traded_value_20m']} liquid (=full signal)",
            "="*60,
        ]
        if rl:
            for r in rl[:60]:
                plain_lines.append(
                    f"{r.get('Ticker','—'):<7} Close:${float(r.get('Price',0) or 0):.2f}  "
                    f"Buy:${float(r.get('Entry_Price',0) or 0):.2f}  "
                    f"Target:${float(r.get('Target',0) or 0):.2f}  "
                    f"Stop:${float(r.get('Stop_Loss',0) or 0):.2f}  "
                    f"ExitBy:{r.get('Exit_By','—')}  "
                    f"Qty:{_SIZE.get(r.get('Ticker'), ('—', None))[0]}  "
                    f"T+:{float(r.get('Target_Offset',0) or 0):.2f}  S-:{float(r.get('Stop_Offset',0) or 0):.2f}")
            plain_lines.append("")
            plain_lines.append("Tickers (comma-separated):")
            plain_lines.append(", ".join(r.get("Ticker","") for r in rl))
        else:
            plain_lines.append("No signals today — no orders to place")
        plain_lines.append("\nFull results in CSV attachment.")
        plain_e = "\n".join(plain_lines)

        subj = (f"📊 RSI(2) Snap-back — {cnt} order{'s' if cnt!=1 else ''}"
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

# ── Charts for top 5: price with buy / target / stop (top), RSI(2) (bottom)
if results:
    top = results[:min(5,len(results))]
    fig, axes = plt.subplots(len(top)*2, 1, figsize=(13, 3.0*len(top)*2), facecolor="#0f172a",
                              gridspec_kw={"height_ratios":[2,1]*len(top)}, squeeze=False)
    axes = axes[:,0]
    for p, r in enumerate(top):
        ax, axr = axes[p*2], axes[p*2+1]
        df_all = r["_df_daily"]
        df_p = df_all.tail(90)
        idx  = df_p.index
        k = r["_idx"]
        sig_d = df_all.index[k]

        ax.set_facecolor("#0f172a")
        ax.plot(idx, df_p["Close"], color="#60a5fa", lw=1.5, label="Close", zorder=5)
        ax.axhline(r["Entry_Price"], color="#22c55e", lw=1.0, ls="--", label=f"Buy limit {r['Entry_Price']:.2f}")
        ax.axhline(r["Target"], color="#3b82f6", lw=1.0, ls="--", label=f"Target {r['Target']:.2f}")
        ax.axhline(r["Stop_Loss"], color="#ef4444", lw=1.0, ls="--", label=f"Stop {r['Stop_Loss']:.2f}")
        if sig_d in idx:
            ax.scatter([sig_d], [df_all["Close"].iloc[k]], color="#facc15", s=50, zorder=7, label="Signal")
        ax.set_title(f"{r['Ticker']}  |  close {r['Price']:.2f}  |  buy {r['Entry_Price']:.2f}  target {r['Target']:.2f}  "
                     f"stop {r['Stop_Loss']:.2f}  |  exit by {r['Exit_By']}",
                     color="#e2e8f0", fontsize=9, fontweight="bold", pad=7)
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))
        ax.tick_params(colors="#94a3b8", labelsize=8)
        for sp in ax.spines.values(): sp.set_edgecolor("#1e3a5f")
        ax.legend(loc="upper left", facecolor="#1e293b", labelcolor="#e2e8f0", fontsize=6, ncol=5)
        ax.grid(color="#1e3a5f", ls="--", lw=0.5, alpha=0.6)

        axr.set_facecolor("#0f172a")
        axr.plot(idx, r["_rsi2"].reindex(idx), color="#a78bfa", lw=1.2, label="RSI 2")
        axr.axhline(CFG["rsi_max"], color="#facc15", lw=0.8, ls="--", label=f"{CFG['rsi_max']:g}")
        for lvl in (50, 90):
            axr.axhline(lvl, color="#334155", lw=0.6, ls="--")
        if sig_d in idx:
            axr.scatter([sig_d], [r["RSI2"]], color="#facc15", s=40, zorder=7)
        axr.set_ylim(-2, 102)
        axr.xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))
        axr.tick_params(colors="#94a3b8", labelsize=7)
        for sp in axr.spines.values(): sp.set_edgecolor("#1e3a5f")
        axr.legend(loc="upper left", facecolor="#1e293b", labelcolor="#e2e8f0", fontsize=6, ncol=2)
        axr.grid(color="#1e3a5f", ls="--", lw=0.5, alpha=0.4)
    plt.suptitle(f"Daily RSI(2) Snap-back  ·  {datetime.today().strftime('%Y-%m-%d')}",
                 color="#60a5fa", fontsize=12, fontweight="bold", y=1.001)
    plt.tight_layout()
    cp = os.path.join(out_dir, f"daily_rsi2_snapback_chart_{ts}.png")
    plt.savefig(cp, dpi=150, bbox_inches="tight", facecolor="#0f172a")
    if _IN_NOTEBOOK: plt.show()
    else: plt.close()
    print(f"  📊 Chart → {cp}")

print(f"""
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  📋 SIGNAL (all required, on the latest daily close)
  1) RSI({CFG['rsi_period']}) below {CFG['rsi_max']:g}  (Wilder RSI)
  2) ATR{CFG['atr_period']} at least {CFG['min_atr_pct']:g}% of price
  3) 20-day average traded value at least ${CFG['min_dollar_vol']/1e6:,.0f}M
  Universe: price >= {CFG['min_price']:g} USD, 20d avg volume >= {CFG['min_avg_volume']:,}

  📋 ORDER PLAN
  Buy    = LIMIT at close - {CFG['buy_atr_below']} x ATR, next session only
  Target = LIMIT SELL at fill + {CFG['target_atr']} x ATR (GTC)
  Stop   = fill - {CFG['stop_atr']} x ATR
  Time   = sell at the open on Exit_By ({CFG['max_hold_sessions']} sessions after the fill day)

  📋 SCORE: ordering only (lower RSI(2) and more liquidity first);
  not tested as a predictor.
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
""")
'''

src = (HEADER + "\n" + "\n".join(L[36:116]) + "\n" + LOGIC + "\n" + "\n".join(L[355:401]) + "\n\n"
       + LIVE + "\n" + "\n".join(L[434:512]) + "\n" + TAIL)
for bad in ("RSI Bounce", "rsi_avg_bounce", "Touch_", "Cross_Date", "analyze_rsi_avg"):
    assert bad not in src, f"template leftover: {bad}"
open(OUT, "w").write(src)
py_compile.compile(OUT, doraise=True)
print(f"✅ wrote {OUT} ({len(src.splitlines())} lines) — compiles")
