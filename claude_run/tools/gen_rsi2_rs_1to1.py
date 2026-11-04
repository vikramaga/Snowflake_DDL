# Generates daily_rsi2_rs_1to1_v1.py: the best >= 1:1 reward/risk rule from
# claude_run/research/rr1_highwin_research.py (confirmed on fresh stocks by
# rr1_validate.py), as a screener whose email gives the full order plan.
# Built from daily_rsi2_snapback_v1.py (same order-plan email, NYSE calendar,
# position sizing) with checked text replacements.
# Usage: python gen_rsi2_rs_1to1.py [out]
import py_compile, sys

SRC = "daily_rsi2_snapback_v1.py"
OUT = sys.argv[1] if len(sys.argv) > 1 else "daily_rsi2_rs_1to1_v1.py"
s = open(SRC).read()


def rep(a, b, n=1):
    global s
    k = s.count(a)
    if k != n:
        sys.exit(f"❌ expected {n} match(es), found {k}: {a[:90]!r}")
    s = s.replace(a, b)


# ── Header ───────────────────────────────────────────────────────────────────────
a = s.index("# ============================================================\n# NASDAQ")
b = s.index("# ============================================================\n", s.index("# SINGLE PASS")) \
    + len("# ============================================================\n")
s = s[:a] + '''# ============================================================
# NASDAQ — Daily RSI(2) Pullback in a Market Leader, 1:1 (v1)
# ============================================================
#
# The highest win rate found at a true 1:1 reward/risk. A fully mechanical
# pullback strategy: every number you need is in the email.
#
# SIGNAL (all required, on the latest daily close)
#   1. RSI(2) < 10                (Wilder RSI, 2 periods: a sharp short-term drop)
#   2. ATR(10) >= 3% of price      (enough daily range)
#   3. Stronger than SPY over 63 sessions (stock % change - SPY % change > 0)
#   4. RSI(14) < 40                (the pullback is real, not a one-day dip)
#   Universe: price >= $5, 20-day average volume >= 100k.
#
# ORDER PLAN (all prices computed by the scanner)
#   Buy    : LIMIT at close - 1.0 x ATR10, valid for the next session only.
#            Not filled -> cancel, no trade.
#   Target : LIMIT SELL at fill + 1.0 x ATR10 (good till cancelled).
#   Stop   : STOP at fill - 1.0 x ATR10.          -> reward = risk (1:1)
#   Time   : if neither has hit after 10 sessions (fill day = session 1),
#            sell at the open on the Exit_By date.
#   If the buy fills BELOW the limit (gap down), move target and stop by
#   the same dollar offsets from the actual fill (Target_Offset / Stop_Offset).
#
# BACKTEST (claude_run/research/rr1_highwin_research.py, Feb 2024 - Oct 2026,
# next-day execution, conservative fills, 0.10% costs, one trade per stock
# at a time):
#   search sample (1,721 stocks): 540 trades, 65.2% wins, avg win +5.75% /
#     avg loss -5.24%, +0.34R per trade.
#   FRESH sample never used to pick the rule (rr1_validate.py, 1,709 other
#     stocks): 561 trades, 66.0% wins, avg win +6.10% / avg loss -5.62%,
#     +0.35R per trade. By year 61% / 72% / 59% (2024 / 2025 / 2026).
#   Worst single trade about -33% (gap through the stop). 90% at 1:1 was not
#   found anywhere in the search; about 60-66% is the realistic ceiling.
#
# Quantity / Total_Amount come from position_sizing.py (max loss $40,
# measured from the limit buy to the stop).
#
# SINGLE PASS — 1 daily download per ticker (+ SPY once), purely technical.
#
# ============================================================
''' + s[b:]

# ── CONFIG ───────────────────────────────────────────────────────────────────────
rep('''    "rsi_max"                : 3.0,    # step 1: RSI(2) below this
    "atr_period"             : 10,
    "min_atr_pct"            : 3.0,    # step 2: ATR10 at least this % of price
    "min_dollar_vol"         : 20e6,   # step 3: 20-day average close x volume
''', '''    "rsi_max"                : 10.0,   # step 1: RSI(2) below this
    "atr_period"             : 10,
    "min_atr_pct"            : 3.0,    # step 2: ATR10 at least this % of price
    "rs_days"                : 63,     # step 3: stock % change - SPY % change over 63 sessions > 0
    "rsi14_max"              : 40.0,   # step 4: RSI(14) below this
''')
rep('''    "buy_atr_below"          : 0.5,    # limit buy = close - 0.5 x ATR
    "target_atr"             : 0.5,    # target    = fill + 0.5 x ATR
    "stop_atr"               : 2.5,    # stop      = fill - 2.5 x ATR''',
'''    "buy_atr_below"          : 1.0,    # limit buy = close - 1.0 x ATR
    "target_atr"             : 1.0,    # target    = fill + 1.0 x ATR
    "stop_atr"               : 1.0,    # stop      = fill - 1.0 x ATR   (1:1)''')
rep('''    "min_price"              : 10.0,''', '''    "min_price"              : 5.0,''')

# ── Signal ───────────────────────────────────────────────────────────────────────
a = s.index("# ── Pattern — single source of truth")
b = s.index("# ── Technical signal: RSI(2) snap-back with the full order plan")
s = s[:a] + '''# ── Pattern — single source of truth ─────────────────────────────────────────────
def check_pattern_at(rsi2, rsi14, atr, rs63, close, i, cfg):
    """
    Checks the signal with "today" at bar `i`. SINGLE SOURCE OF TRUTH.
    rs63[i] = stock % change over rs_days minus SPY % change over rs_days.

    Computes EVERY step's pass/fail regardless of earlier failures
    (guarding only on missing data), so callers can diagnose exactly
    which step is filtering results out — see FUNNEL_COUNTS below.

    Returns (passed: bool, details: dict).
    """
    if i < 0 or i >= len(close) or not np.isfinite(rsi2[i]) or not np.isfinite(atr[i]) \\
            or not np.isfinite(rsi14[i]) or not np.isfinite(rs63[i]) or close[i] <= 0:
        return False, {}
    atr_pct = atr[i] / close[i] * 100
    oversold = bool(rsi2[i] < cfg["rsi_max"])                    # Step 1
    volatile = bool(atr_pct >= cfg["min_atr_pct"])               # Step 2
    leader   = bool(rs63[i] > 0)                                 # Step 3
    pulled   = bool(rsi14[i] < cfg["rsi14_max"])                 # Step 4
    passed = oversold and volatile and leader and pulled
    return passed, {
        "idx": i, "rsi2": float(rsi2[i]), "rsi14": float(rsi14[i]), "atr": float(atr[i]),
        "atr_pct": float(atr_pct), "rs63": float(rs63[i]), "close": float(close[i]),
        "oversold": oversold, "volatile": volatile, "leader": leader, "pulled": pulled,
    }

# ── Diagnostic funnel — tallies how far each ticker-day gets through
#    the signal, across the full universe scan, so a 0-match run can
#    be diagnosed empirically instead of guessed at ─────────────────
FUNNEL_COUNTS = {
    "days_checked": 0,
    "passed_step1_rsi2_below_10": 0,
    "passed_step2_atr_at_least_3pct": 0,
    "passed_step3_beats_spy_63d": 0,
    "passed_step4_rsi14_below_40": 0,
}

def find_signals(rsi2, rsi14, atr, rs63, close, cfg):
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
        passed, d = check_pattern_at(rsi2, rsi14, atr, rs63, close, i, cfg)
        if not d:
            continue
        FUNNEL_COUNTS["days_checked"] += 1
        if d["oversold"]:
            FUNNEL_COUNTS["passed_step1_rsi2_below_10"] += 1
            if d["volatile"]:
                FUNNEL_COUNTS["passed_step2_atr_at_least_3pct"] += 1
                if d["leader"]:
                    FUNNEL_COUNTS["passed_step3_beats_spy_63d"] += 1
                    if d["pulled"]:
                        FUNNEL_COUNTS["passed_step4_rsi14_below_40"] += 1
        if passed:
            hits.append(d)
    return hits

SPY_CLOSE = None   # SPY daily closes, loaded once before the scan (needed for step 3)

''' + s[b:]

rep('''# ── Technical signal: RSI(2) snap-back with the full order plan ────────
def analyze_rsi2_snapback(sym, df_daily):''', '''# ── Technical signal: RSI(2) pullback in a leader, with the order plan ──
def analyze_rsi2_rs_1to1(sym, df_daily):''')
rep('''    rsi2   = wilder_rsi(c, CFG["rsi_period"])
    atr    = calc_atr(h, l, c, CFG["atr_period"])
    dvol20 = pd.Series(c * v).rolling(20).mean().values

    hits = find_snapback_signals(rsi2, atr, dvol20, c, CFG)''', '''    if SPY_CLOSE is None:
        return None
    rsi2   = wilder_rsi(c, CFG["rsi_period"])
    rsi14  = wilder_rsi(c, 14)
    atr    = calc_atr(h, l, c, CFG["atr_period"])
    dvol20 = pd.Series(c * v).rolling(20).mean().values
    nd = CFG["rs_days"]
    ret_s = c / np.concatenate([[np.nan] * nd, c[:-nd]]) - 1
    ret_spy = (SPY_CLOSE / SPY_CLOSE.shift(nd) - 1).reindex(df_daily.index).values
    rs63 = ret_s - ret_spy

    hits = find_signals(rsi2, rsi14, atr, rs63, c, CFG)''')
rep('''    # ── Score (0-100) — for ORDERING only; the research did not test ──
    #    whether a higher score wins more often. Lower RSI(2) and more
    #    liquidity rank first.
    score = 70 + (CFG["rsi_max"] - s["rsi2"]) / CFG["rsi_max"] * 20 \\
            + min(10, max(0, (np.log10(s["dvol20"]) - np.log10(CFG["min_dollar_vol"])) * 10))
    score = round(min(100, max(0, score)))
    flags = [f"RSI2 {s['rsi2']:.1f}", f"ATR {s['atr_pct']:.1f}%", f"Value ${s['dvol20']/1e6:,.0f}M/day"]''',
'''    # ── Score (0-100) — for ORDERING only; the research did not test ──
    #    whether a higher score wins more often. Lower RSI(2) and more
    #    strength vs SPY rank first.
    score = 60 + (CFG["rsi_max"] - s["rsi2"]) / CFG["rsi_max"] * 20 + min(20, s["rs63"] * 100 / 2)
    score = round(min(100, max(0, score)))
    dv = float(dvol20[k]) if np.isfinite(dvol20[k]) else 0.0
    flags = [f"RSI2 {s['rsi2']:.1f}", f"RSI14 {s['rsi14']:.0f}", f"vs SPY +{s['rs63']*100:.0f}% (63d)",
             f"ATR {s['atr_pct']:.1f}%", f"Value ${dv/1e6:,.0f}M/day"]
    if dv < 20e6:
        flags.append("Thin: check the spread")''')
rep('''        "RSI2"           : round(s["rsi2"], 2),
        "ATR10"          : round(a, 2),
        "ATR_%"          : round(s["atr_pct"], 1),
        "Value_20d_$M"   : round(s["dvol20"] / 1e6, 1),''', '''        "RSI2"           : round(s["rsi2"], 2),
        "RSI14"          : round(s["rsi14"], 1),
        "vs_SPY_63d_%"   : round(s["rs63"] * 100, 1),
        "ATR10"          : round(a, 2),
        "ATR_%"          : round(s["atr_pct"], 1),
        "Value_20d_$M"   : round(dv / 1e6, 1),''')

# ── Load SPY before the scan ─────────────────────────────────────────────
rep('''TICKERS = get_tickers()
print()
''', '''TICKERS = get_tickers()
print()

# SPY for step 3 (relative strength) — loaded once
_spy = download_daily(["SPY"], CFG["history_days"]).get("SPY")
SPY_CLOSE = _spy["Close"] if _spy is not None else None
if SPY_CLOSE is None:
    print("  ❌ SPY data unavailable — step 3 cannot be checked, so no signals this run")
else:
    print(f"  ✅ SPY: {len(SPY_CLOSE)} bars to {SPY_CLOSE.index[-1].date()}")
''')

# ── Scan, funnel and text ────────────────────────────────────────────────────────
rep('''print(f"  Signal: RSI({CFG['rsi_period']}) < {CFG['rsi_max']:g}, ATR{CFG['atr_period']} >= {CFG['min_atr_pct']:g}% of price, "
      f"20d traded value >= ${CFG['min_dollar_vol']/1e6:,.0f}M\\n")''',
'''print(f"  Signal: RSI({CFG['rsi_period']}) < {CFG['rsi_max']:g}, ATR{CFG['atr_period']} >= {CFG['min_atr_pct']:g}% of price, "
      f"beats SPY over {CFG['rs_days']} sessions, RSI(14) < {CFG['rsi14_max']:g}\\n")''')
rep('''for sym in tqdm(list(daily_map.keys()), desc="Checking RSI(2) snap-back", unit="stk"):
    try:
        r = analyze_rsi2_snapback(sym, daily_map[sym])''', '''for sym in tqdm(list(daily_map.keys()), desc="Checking RSI(2) leader pullback", unit="stk"):
    try:
        r = analyze_rsi2_rs_1to1(sym, daily_map[sym])''')
rep('''print(f"  Step 1 — RSI({CFG['rsi_period']}) below {CFG['rsi_max']:g}                     : {fc['passed_step1_rsi2_below_3']}")
print(f"  Step 2 — + ATR{CFG['atr_period']} >= {CFG['min_atr_pct']:g}% of price              : {fc['passed_step2_atr_at_least_3pct']}")
print(f"  Step 3 — + traded value >= ${CFG['min_dollar_vol']/1e6:,.0f}M/day         : {fc['passed_step3_traded_value_20m']}  (= full signal)")''',
'''print(f"  Step 1 — RSI({CFG['rsi_period']}) below {CFG['rsi_max']:g}                    : {fc['passed_step1_rsi2_below_10']}")
print(f"  Step 2 — + ATR{CFG['atr_period']} >= {CFG['min_atr_pct']:g}% of price              : {fc['passed_step2_atr_at_least_3pct']}")
print(f"  Step 3 — + beats SPY over {CFG['rs_days']} sessions             : {fc['passed_step3_beats_spy_63d']}")
print(f"  Step 4 — + RSI(14) below {CFG['rsi14_max']:g}                      : {fc['passed_step4_rsi14_below_40']}  (= full signal)")''')
rep('''    print("\\n  No signals today. That is normal for this strategy: RSI(2) below 3")
    print("  with enough range and liquidity does not happen every day. The")''',
'''    print("\\n  No signals today. That is normal for this strategy: a sharp pullback")
    print("  in a stock that is beating SPY does not happen every day. The")''')
rep('''    "RSI2","ATR10","ATR_%","Value_20d_$M",''', '''    "RSI2","RSI14","vs_SPY_63d_%","ATR10","ATR_%","Value_20d_$M",''')
rep('''    "Value_20d_$M"  : lambda v: f"${v:,.0f}M",''', '''    "Value_20d_$M"  : lambda v: f"${v:,.0f}M",
    "RSI14"         : lambda v: f"{v:.0f}",
    "vs_SPY_63d_%"  : lambda v: f"{v:+.0f}%",''')
s = s.replace("Daily RSI(2) Snap-back", "Daily RSI(2) Leader Pullback 1:1")
s = s.replace("daily_rsi2_snapback_", "daily_rsi2_rs_1to1_")
rep('''  {cnt} signal{'s' if cnt!=1 else ''} — RSI({CFG['rsi_period']}) &lt; {CFG['rsi_max']:g}, ATR{CFG['atr_period']} &ge; {CFG['min_atr_pct']:g}% of price,
  traded value &ge; ${CFG['min_dollar_vol']/1e6:,.0f}M/day''',
'''  {cnt} signal{'s' if cnt!=1 else ''} — RSI({CFG['rsi_period']}) &lt; {CFG['rsi_max']:g}, ATR{CFG['atr_period']} &ge; {CFG['min_atr_pct']:g}% of price,
  beating SPY over {CFG['rs_days']} sessions, RSI(14) &lt; {CFG['rsi14_max']:g} · target = stop distance (1:1)''')
rep('''    {fc['passed_step1_rsi2_below_3']} RSI(2) &lt; {CFG['rsi_max']:g} &nbsp;→&nbsp;
    {fc['passed_step2_atr_at_least_3pct']} ATR &ge; {CFG['min_atr_pct']:g}% &nbsp;→&nbsp;
    <b style="color:#facc15">{fc['passed_step3_traded_value_20m']} traded value &ge; ${CFG['min_dollar_vol']/1e6:,.0f}M = full signal</b>''',
'''    {fc['passed_step1_rsi2_below_10']} RSI(2) &lt; {CFG['rsi_max']:g} &nbsp;→&nbsp;
    {fc['passed_step2_atr_at_least_3pct']} ATR &ge; {CFG['min_atr_pct']:g}% &nbsp;→&nbsp;
    {fc['passed_step3_beats_spy_63d']} beats SPY &nbsp;→&nbsp;
    <b style="color:#facc15">{fc['passed_step4_rsi14_below_40']} RSI(14) &lt; {CFG['rsi14_max']:g} = full signal</b>''')
rep('''  Backtest (≈1,700 US stocks, Jan 2024 – Oct 2026, next-day orders, 0.10% costs): 85.6% wins / +1.07% per trade
  in the training year, 83.1% / +0.95% in the unseen year. Average win ≈ +2.9%, average loss ≈ −8.7%;
  worst trade −29% (gap through the stop). Past results do not guarantee future results.''',
'''  Backtest (Feb 2024 – Oct 2026, next-day orders, conservative fills, 0.10% costs, one trade per stock):
  1,721 stocks used to find the rule: 540 trades, 65.2% wins, avg win +5.8% / avg loss −5.2%;
  1,709 fresh stocks never used to pick it: 561 trades, 66.0% wins, avg win +6.1% / avg loss −5.6%
  (by year 61% / 72% / 59%). Worst trade about −33% (gap through the stop). Past results do not guarantee future results.''')
rep('''            f"{cnt} signals (RSI({CFG['rsi_period']}) < {CFG['rsi_max']:g}, ATR >= {CFG['min_atr_pct']:g}%, value >= ${CFG['min_dollar_vol']/1e6:,.0f}M/day)",''',
'''            f"{cnt} signals (RSI({CFG['rsi_period']}) < {CFG['rsi_max']:g}, ATR >= {CFG['min_atr_pct']:g}%, beats SPY {CFG['rs_days']}d, RSI14 < {CFG['rsi14_max']:g}; 1:1)",''')
rep('''            f"{fc['passed_step1_rsi2_below_3']} RSI2<{CFG['rsi_max']:g} -> "
            f"{fc['passed_step2_atr_at_least_3pct']} ATR ok -> "
            f"{fc['passed_step3_traded_value_20m']} liquid (=full signal)",''',
'''            f"{fc['passed_step1_rsi2_below_10']} RSI2<{CFG['rsi_max']:g} -> "
            f"{fc['passed_step2_atr_at_least_3pct']} ATR ok -> "
            f"{fc['passed_step3_beats_spy_63d']} beats SPY -> "
            f"{fc['passed_step4_rsi14_below_40']} RSI14<{CFG['rsi14_max']:g} (=full signal)",''')
rep('''        subj = (f"📊 RSI(2) Snap-back — {cnt} order{'s' if cnt!=1 else ''}"''',
    '''        subj = (f"📊 RSI(2) Leader Pullback 1:1 — {cnt} order{'s' if cnt!=1 else ''}"''')
rep('''  3) 20-day average traded value at least ${CFG['min_dollar_vol']/1e6:,.0f}M''',
    '''  3) Stock % change minus SPY % change over {CFG['rs_days']} sessions > 0
  4) RSI(14) below {CFG['rsi14_max']:g}''')
rep('''  📋 SCORE: ordering only (lower RSI(2) and more liquidity first);''',
    '''  📋 SCORE: ordering only (lower RSI(2) and more strength vs SPY first);''')

for word in ("min_dollar_vol", "passed_step3_traded_value_20m", "passed_step1_rsi2_below_3'",
             "find_snapback_signals", "analyze_rsi2_snapback", "Snap-back"):
    if word in s:
        sys.exit(f"❌ leftover: {word}")
open(OUT, "w").write(s)
py_compile.compile(OUT, doraise=True)
print(f"✅ wrote {OUT}")
