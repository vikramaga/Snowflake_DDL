# Generates daily_sweep_sma50_strategy_v1.py from daily_liquidity_sweep_breakout_v1.py
# (the 0.75R strategy found by the Oct-2026 research backtest). Every edit must
# match exactly once; the result is compiled. Usage: python gen_sweep_sma50.py [out]
import py_compile, sys

SRC = "daily_liquidity_sweep_breakout_v1.py"
OUT = sys.argv[1] if len(sys.argv) > 1 else "daily_sweep_sma50_strategy_v1.py"
s = open(SRC).read()


def rep(a, b, count=1):
    global s
    n = s.count(a)
    if (count == "all" and n == 0) or (count != "all" and n != count):
        raise SystemExit(f"edit expected {count} match(es), found {n}: {a[:70]!r}")
    s = s.replace(a, b)


# ── header ───────────────────────────────────────────────────────────────────
rep("# NASDAQ — Daily Liquidity Sweep Breakout (v1)",
    "# NASDAQ — Liquidity Sweep + Rising SMA50 — 0.75R strategy (v1)")
rep("""#   5. ...and is GREEN (closed above its open).
#
# By default""", """#   5. ...and is GREEN (closed above its open).
#
#   6. SMA50 RISING: today's SMA50 is above the SMA50 of 20 trading days
#      ago (sma_rise_days).
#
#   7. REAL STOP: the stop (lower of candle B's and C's lows) is at least
#      1% below the entry (min_stop_pct) — tighter stops are noise.
#
# TRADE PLAN (from the 2-year research backtest, Oct 2024 – Oct 2026):
#   Entry  = candle C's close.   Stop = lower of candle B's and C's lows.
#   Target = entry + 0.75 x (entry - stop)   (target_r).
#   Exit   = target, stop, or the close of the 20th trading day (max_hold_days).
#   Backtest, 600-stock sample: year 1 352 trades, 62.2% wins, +0.14R/trade;
#   year 2 (unseen) 387 trades, 61.8% wins, +0.11R/trade. Breakeven win
#   rate at 0.75R is 57.1%. No costs/slippage; today's listings only.
#
# By default""")
rep("""# OUTPUT: Entry_Price and Stop_Loss are reasonable defaults (Entry =
# candle C's close; Stop = the lower of candle B's and C's lows), not
# explicitly requested. SMA50 is shown for trend context and adds a
# small score bonus; it is NOT a gate.""", """# OUTPUT: Entry_Price, Stop_Loss, Target_Price and Exit_By follow the trade
# plan above. Quantity / Total_Amount come from position_sizing.py
# (max loss $40 per trade).""")

# ── config ───────────────────────────────────────────────────────────────────
rep('''    "sma_period"             : 50,   # trend context only (score bonus, not a gate)''',
    '''    "sma_period"             : 50,
    "sma_rise_days"          : 20,   # step 6: SMA50 today > SMA50 this many days ago
    "min_stop_pct"           : 1.0,  # step 7: stop at least this % below the entry
    "target_r"               : 0.75, # target = entry + target_r x risk
    "max_hold_days"          : 20,   # exit at the close of this trading day''')

# ── pattern: steps 6 and 7 ──────────────────────────────────────────────
rep("def check_pattern_at(df, i):", "def check_pattern_at(df, i, cfg=None):")
rep('''    C_green        = bool(cC > oC)          # Step 5

    passed = A_green and B_swept_high and B_closed_below and C_above_B_high and C_green
''', '''    C_green        = bool(cC > oC)          # Step 5

    cfg = cfg or CFG
    closes = df["Close"].iloc[:i+1]
    n_sma, back = cfg["sma_period"], cfg["sma_rise_days"]
    if len(closes) >= n_sma + back:
        sma_now  = float(closes.iloc[-n_sma:].mean())
        sma_prev = float(closes.iloc[-(n_sma + back):-back].mean())
    else:
        sma_now = sma_prev = float("nan")
    stop     = min(lB, lC)
    risk_pct = (cC - stop) / cC * 100 if cC > 0 else 0.0
    SMA50_rising = bool(sma_now > sma_prev)                     # Step 6
    stop_ok      = bool(risk_pct >= cfg["min_stop_pct"])        # Step 7

    passed = (A_green and B_swept_high and B_closed_below and C_above_B_high and C_green
              and SMA50_rising and stop_ok)
''')
rep('''        "C_green": C_green,
    }''', '''        "C_green": C_green,
        "sma_now": sma_now, "sma_prev": sma_prev, "risk_pct": risk_pct,
        "SMA50_rising": SMA50_rising, "stop_ok": stop_ok,
    }''')
rep('''    "passed_step5_C_green": 0,
}''', '''    "passed_step5_C_green": 0,
    "passed_step6_SMA50_rising": 0,
    "passed_step7_stop_at_least_1pct": 0,
}''')
rep("        passed, d = check_pattern_at(df, i)", "        passed, d = check_pattern_at(df, i, cfg)")
rep('''                            FUNNEL_COUNTS["passed_step5_C_green"] += 1''',
    '''                            FUNNEL_COUNTS["passed_step5_C_green"] += 1
                            if d["SMA50_rising"]:
                                FUNNEL_COUNTS["passed_step6_SMA50_rising"] += 1
                                if d["stop_ok"]:
                                    FUNNEL_COUNTS["passed_step7_stop_at_least_1pct"] += 1''')

# ── analyze: target / exit date ──────────────────────────────────────────────
rep('''    risk_pct = (entry_price - stop_loss) / entry_price * 100
''', '''    risk_pct = (entry_price - stop_loss) / entry_price * 100
    target_price = entry_price + CFG["target_r"] * (entry_price - stop_loss)
    exit_by = pd.bdate_range(df_daily.index[k], periods=CFG["max_hold_days"] + 1)[-1]
''')
rep('''        "Risk_%"         : round(risk_pct, 1),
        "A_High"''', '''        "Risk_%"         : round(risk_pct, 1),
        "Target_Price"   : round(target_price, 2),
        "Target_R"       : CFG["target_r"],
        "Exit_By"        : exit_by.strftime("%Y-%m-%d"),
        "A_High"''')
rep('''        "Above_SMA50"    : above_sma,
        "Signal_Date"''', '''        "Above_SMA50"    : above_sma,
        "SMA50_20d_Ago"  : round(s["sma_prev"], 2),
        "Signal_Date"''')

# ── live print / funnel print ────────────────────────────────────────
rep('''LIVE_COLS = ["Ticker","Price","Score","Entry_Price","Stop_Loss",
             "B_High","Breakout_%","Signal_Date"]
_CW = {"Ticker":8,"Price":10,"Score":7,"Entry_Price":12,"Stop_Loss":11,
       "B_High":10,"Breakout_%":11,"Signal_Date":13}
_CF = {"Price":"${:.2f}","Score":"{:.0f}","Entry_Price":"${:.2f}",
       "Stop_Loss":"${:.2f}","B_High":"${:.2f}"}''', '''LIVE_COLS = ["Ticker","Price","Score","Entry_Price","Stop_Loss",
             "Target_Price","Risk_%","Exit_By"]
_CW = {"Ticker":8,"Price":10,"Score":7,"Entry_Price":12,"Stop_Loss":11,
       "Target_Price":13,"Risk_%":8,"Exit_By":12}
_CF = {"Price":"${:.2f}","Score":"{:.0f}","Entry_Price":"${:.2f}",
       "Stop_Loss":"${:.2f}","Target_Price":"${:.2f}","Risk_%":"{:.1f}%"}''')
rep("passes all 5 steps", "passes all 7 steps")
rep('''print(f"  Step 5 — + candle C green                    : {fc['passed_step5_C_green']}  (= full pattern)")''',
    '''print(f"  Step 5 — + candle C green                    : {fc['passed_step5_C_green']}")
print(f"  Step 6 — + SMA50 above its level {CFG['sma_rise_days']}d ago   : {fc['passed_step6_SMA50_rising']}")
print(f"  Step 7 — + stop >= {CFG['min_stop_pct']:.0f}% below entry         : {fc['passed_step7_stop_at_least_1pct']}  (= full pattern)")''')

# ── output columns ───────────────────────────────────────────────────────────
rep('''    "Entry_Price","Stop_Loss","Risk_%",
    "A_High","B_High","Sweep_%","B_Reject_%","Breakout_%","C_Body_%",
    "Vol_vs_20d_Avg","SMA50","Above_SMA50",''', '''    "Entry_Price","Stop_Loss","Risk_%","Target_Price","Target_R","Exit_By",
    "A_High","B_High","Sweep_%","B_Reject_%","Breakout_%","C_Body_%",
    "Vol_vs_20d_Avg","SMA50","SMA50_20d_Ago","Above_SMA50",''')
rep('''    "Stop_Loss"     : lambda v: f"${v:.2f}",
    "A_High"''', '''    "Stop_Loss"     : lambda v: f"${v:.2f}",
    "Target_Price"  : lambda v: f"${v:.2f}",
    "A_High"''')
rep('''DISP_COLS = ["Ticker","Price","Score","Entry_Price","Stop_Loss","Risk_%",
             "B_High","Breakout_%","Vol_vs_20d_Avg","Signal_Date"]''',
    '''DISP_COLS = ["Ticker","Price","Score","Entry_Price","Stop_Loss","Target_Price","Risk_%",
             "Exit_By","Vol_vs_20d_Avg","Signal_Date"]''')
rep('''  Stop_Loss       lower of candle B's and C's lows''', '''  Stop_Loss       lower of candle B's and C's lows (>= 1% below entry)
  Target_Price    entry + 0.75 x (entry - stop)
  Exit_By         sell at this day's close if neither stop nor target hit''')

# ── names / files ─────────────────────────────────────────────────────────────
rep("Daily Liquidity Sweep Breakout", "Liquidity Sweep + Rising SMA50", "all")
rep("daily_liquidity_sweep_breakout_", "daily_sweep_sma50_strategy_", "all")

# ── email ────────────────────────────────────────────────────────────────────
rep('''# ── Email with CSV attached ───────────────────────────────''', '''# Quantity / Total_Amount per ticker for the email (from the sized df_out)
_SIZE = {}
if {"Ticker", "Quantity", "Total_Amount"} <= set(df_out.columns):
    _SIZE = {t: (q, a) for t, q, a in zip(df_out["Ticker"], df_out["Quantity"], df_out["Total_Amount"])}

# ── Email with CSV attached ───────────────────────────────''')
rep('''        heads = ["Ticker","Price","Score","Entry","Stop","Risk %","B High",
                 "Breakout %","Vol/20d","Trend","Signal"]''', '''        heads = ["Ticker","Price","Score","Entry","Stop","Risk %","Target",
                 "Qty","Amount","Vol/20d","Exit by","Signal"]''')
rep('''            trend = "Above SMA50" if r.get("Above_SMA50") else "Below SMA50"
''', '''            q, a = _SIZE.get(r.get("Ticker"), ("—", None))
''')
rep('''                + td(f'${float(r.get("B_High",0) or 0):.2f}', "color:#3b82f6")
                + td(f'{float(r.get("Breakout_%",0) or 0):+.1f}%', "font-weight:600")
                + td(f'x{float(r.get("Vol_vs_20d_Avg",0) or 0):.2f}')
                + td(trend, "color:#16a34a" if r.get("Above_SMA50") else "color:#b45309")''',
    '''                + td(f'${float(r.get("Target_Price",0) or 0):.2f}', "color:#16a34a;font-weight:700")
                + td(f'{q}')
                + td(f'${float(a):,.0f}' if a is not None else "—")
                + td(f'x{float(r.get("Vol_vs_20d_Avg",0) or 0):.2f}')
                + td(r.get("Exit_By","—"))''')
rep('''<td colspan="11" ''', '''<td colspan="12" ''')
rep('''  {cnt} match{'es' if cnt!=1 else ''} — A green; B swept above A's high but closed below
  A's close; C closed above B's high and is green''', '''  {cnt} match{'es' if cnt!=1 else ''} — liquidity sweep breakout (A green; B swept A's high, closed below
  A's close; C closed above B's high, green) + SMA50 rising + stop &ge; 1%<br>
  Plan: target {CFG['target_r']}R, stop, or exit after {CFG['max_hold_days']} days &nbsp;·&nbsp;
  2-yr backtest: ~62% wins, +0.11 to +0.14R per trade (before costs)''')
rep('''    <b style="color:#facc15">{fc['passed_step5_C_green']} C green = full match</b>''',
    '''    {fc['passed_step5_C_green']} C green &nbsp;→&nbsp;
    {fc['passed_step6_SMA50_rising']} SMA50 rising &nbsp;→&nbsp;
    <b style="color:#facc15">{fc['passed_step7_stop_at_least_1pct']} stop &ge; 1% = full match</b>''')
rep('''Entry = candle C close, Stop = lower of B/C lows (defaults)''',
    '''Entry = C close · Stop = lower of B/C lows · Target = entry + 0.75R · max hold 20 days · Qty sized for a $40 max loss''')
rep('''            f"{cnt} matches (A green; B high > A high, B close < A close; C close > B high, C green)",''',
    '''            f"{cnt} matches (liquidity sweep breakout + SMA50 rising + stop >= 1%; target 0.75R, max hold 20 days)",''')
rep('''            f"{fc['passed_step5_C_green']} C green (=full match)",''', '''            f"{fc['passed_step5_C_green']} C green -> "
            f"{fc['passed_step6_SMA50_rising']} SMA50 rising -> "
            f"{fc['passed_step7_stop_at_least_1pct']} stop >= 1% (=full match)",''')
rep('''                    f"BHigh:${float(r.get('B_High',0) or 0):.2f}  "
                    f"Breakout:{float(r.get('Breakout_%',0) or 0):+.1f}%  "
                    f"{'AboveSMA50' if r.get('Above_SMA50') else 'BelowSMA50'}  "''',
    '''                    f"Target:${float(r.get('Target_Price',0) or 0):.2f}  "
                    f"Qty:{_SIZE.get(r.get('Ticker'), ('—', None))[0]}  "
                    f"ExitBy:{r.get('Exit_By','—')}  "''')
rep('''        subj = (f"📊 Liquidity Sweep Breakout — {cnt} signal''', '''        subj = (f"🎯 Sweep + SMA50 Strategy (0.75R) — {cnt} signal''')

# ── chart: target line ──────────────────────────────────────────────────────────
rep('''        ax.axhline(r["Stop_Loss"], color="#ef4444", lw=1.0, ls="--", alpha=0.85,
                  label=f"Stop {r['Stop_Loss']:.2f}")''', '''        ax.axhline(r["Stop_Loss"], color="#ef4444", lw=1.0, ls="--", alpha=0.85,
                  label=f"Stop {r['Stop_Loss']:.2f}")
        ax.axhline(r["Target_Price"], color="#22c55e", lw=1.0, ls="--", alpha=0.85,
                  label=f"Target {r['Target_Price']:.2f}")''')

# ── guide ────────────────────────────────────────────────────────────────────
rep('''  5) ...and is GREEN.
''', '''  5) ...and is GREEN.
  6) SMA50 today > SMA50 {CFG['sma_rise_days']} trading days ago.
  7) Stop (lower of B/C lows) at least {CFG['min_stop_pct']}% below the entry.
''')
rep('''  📋 OUTPUT (reasonable defaults — not explicitly requested)
  Entry_Price = candle C's CLOSE
  Stop_Loss   = the lower of candle B's and C's LOWS
  SMA50       = trend context (score bonus only, not a gate)''', '''  📋 TRADE PLAN (from the 2-year research backtest)
  Entry_Price  = candle C's CLOSE
  Stop_Loss    = the lower of candle B's and C's LOWS
  Target_Price = entry + {CFG['target_r']} x (entry - stop)
  Exit_By      = close of the {CFG['max_hold_days']}th trading day if neither is hit
  Backtest: ~62% wins, +0.11 to +0.14R per trade (before costs)''')

open(OUT, "w").write(s)
py_compile.compile(OUT, doraise=True)
print(f"✅ wrote {OUT} ({len(s.splitlines())} lines) — compiles")
