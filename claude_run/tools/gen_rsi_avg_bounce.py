# Generates daily_rsi_avg_bounce_v1.py from daily_rsi_high_breakout_v1.py:
# new header + new pattern logic, the template's boilerplate / download /
# ticker list / output / email / chart code with checked edits (each edit must
# match exactly once). Usage: python gen_rsi_avg_bounce.py [out]
import py_compile, sys

SRC = "daily_rsi_high_breakout_v1.py"
OUT = sys.argv[1] if len(sys.argv) > 1 else "daily_rsi_avg_bounce_v1.py"
L = open(SRC).read().split("\n")
assert L[113].startswith("# ── CONFIG") and L[330].startswith("# ── Download: daily"), "template layout changed"

HEADER = '''# ============================================================
# NASDAQ — Daily RSI Bounce off its Average (v1)
# ============================================================
#
# SIGNAL (all required). RSI = Wilder RSI(14) on daily closes; RSI
# average = 9-day simple average of RSI (same definitions as the repo's
# other RSI scanners). "Gap" = RSI minus its average.
#
#   1. RECENT CROSS UP: RSI crossed above its average 3-10 trading days
#      ago (the day before it was at/below, that day above).
#
#   2. MOVED AWAY: after the cross, RSI rose at least 3 points above
#      its average (a real separation, not hugging the line).
#
#   3. PULLED BACK TO THE AVERAGE: RSI then came back to within 1.5
#      points of its average (a dip of up to 2 points below it is
#      allowed), without breaking more than 2 points below at any time
#      since the cross.
#
#   4. JUST BOUNCED: that touch was in the last 2 days, and today RSI is
#      rising, above its average and moving away from it again.
#
# By default the bounce must be on the most recent daily candle
# (signal_lookback_days = 1).
#
# DATA — only 1 download per ticker: daily bars.
#
# OUTPUT: Entry_Price and Stop_Loss are reasonable defaults (Entry =
# today's close; Stop = the lowest low of the pullback, from RSI's peak
# to today), not explicitly requested. Quantity / Total_Amount come from
# position_sizing.py (max loss $40). SMA50 is trend context (score bonus).
#
# SINGLE PASS — purely technical, no fundamentals fetch.
#
# ============================================================
'''

LOGIC = '''# ── CONFIG ────────────────────────────────────────────────────────────
CFG = {
    "history_days"          : 550,   # daily bars — ~380 trading days, comfortable
                                      # margin above _clean()'s 220-bar floor

    # ── Indicator periods ───────────────────────────────────────
    "rsi_period"             : 14,
    "rsi_avg_period"         : 9,    # simple average of RSI
    "sma_period"             : 50,   # trend context only (score bonus, not a gate)

    # ── Pattern ──────────────────────────────────────────────────
    "cross_min_days_ago"     : 3,    # step 1: cross 3..10 trading days before today
    "cross_max_days_ago"     : 10,
    "min_separation_pts"     : 3.0,  # step 2: RSI got this far above its average
    "touch_pts"              : 1.5,  # step 3: back within this many points of the average
    "max_break_pts"          : 2.0,  # step 3: never more than this far below it
    "bounce_within_days"     : 2,    # step 4: the touch was in the last N days

    "signal_lookback_days"   : 1,    # 1 = the bounce must be on the most recent candle

    # ── Filters ─────────────────────────────────────────────────
    "min_avg_volume"         : 100_000,
    "min_price"              : 10.0,

    "batch_size"             : 50,
    "batch_sleep"            : 1.5,
}

# ── Indicators ───────────────────────────────────────────────────────
def calc_rsi(close, period=14):
    """Standard Wilder-smoothed RSI."""
    delta = close.diff()
    gain  = delta.clip(lower=0)
    loss  = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1/period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1/period, min_periods=period, adjust=False).mean()
    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))

def check_pattern_at(rsi, rsi_avg, i, cfg):
    """
    Checks the RSI-average bounce with "today" at bar `i`.
    SINGLE SOURCE OF TRUTH for the pattern logic.

    The cross is the most recent bar X in [i - cross_max_days_ago,
    i - cross_min_days_ago] with gap[X-1] <= 0 < gap[X]. The touch is the
    most recent bar T in (X, i) with gap[T] <= touch_pts, after RSI had
    first moved at least min_separation_pts above its average.

    Computes EVERY stage's pass/fail regardless of earlier failures
    (guarding only on missing data), so callers can diagnose exactly
    which stage is filtering results out — see FUNNEL_COUNTS below.

    Returns (passed: bool, details: dict).
    """
    lo, hi = cfg["cross_min_days_ago"], cfg["cross_max_days_ago"]
    if i - hi - 1 < 0 or i >= len(rsi):
        return False, {}
    gap = (rsi - rsi_avg).values
    r = rsi.values
    if np.isnan(gap[i - hi - 1:i + 1]).any():
        return False, {}

    x = None
    for j in range(i - lo, i - hi - 1, -1):                  # most recent first
        if gap[j - 1] <= 0 < gap[j]:
            x = j
            break
    has_cross = x is not None                                         # Step 1

    peak_idx = touch_idx = None
    peak_gap = touch_gap = min_gap = np.nan
    if has_cross:
        min_gap = float(gap[x:i + 1].min())
        run_peak, run_peak_idx = -np.inf, None
        for t in range(x, i):
            if gap[t] > run_peak:
                run_peak, run_peak_idx = gap[t], t
            if t > x and gap[t] <= cfg["touch_pts"] and run_peak >= cfg["min_separation_pts"]:
                touch_idx, peak_idx = t, run_peak_idx
        pk = range(x, i) if touch_idx is None else range(x, touch_idx)
        peak_gap = float(max(gap[t] for t in pk)) if len(pk) else float(gap[x])
        if peak_idx is None:
            peak_idx = x + int(np.argmax(gap[x:i])) if i > x else x
        if touch_idx is not None:
            touch_gap = float(gap[touch_idx])

    separated = bool(has_cross and peak_gap >= cfg["min_separation_pts"])         # Step 2
    touched   = bool(separated and touch_idx is not None
                     and min_gap >= -cfg["max_break_pts"])                        # Step 3
    bounced   = bool(touched and touch_idx >= i - cfg["bounce_within_days"]
                     and r[i] > r[i - 1] and gap[i] > 0 and gap[i] > gap[i - 1]) # Step 4

    passed = has_cross and separated and touched and bounced
    return passed, {
        "idx": i, "cross_idx": x, "peak_idx": peak_idx, "touch_idx": touch_idx,
        "rsi": float(r[i]), "rsi_avg": float(rsi_avg.values[i]), "gap": float(gap[i]),
        "gap_prev": float(gap[i - 1]), "peak_gap": peak_gap, "touch_gap": touch_gap,
        "min_gap": min_gap,
        "has_cross": has_cross, "separated": separated, "touched": touched, "bounced": bounced,
    }

# ── Diagnostic funnel — tallies how far each ticker-day gets through
#    the pattern, across the full universe scan, so a 0-match run can
#    be diagnosed empirically instead of guessed at ─────────────────
FUNNEL_COUNTS = {
    "days_checked": 0,
    "passed_step1_recent_cross_up": 0,
    "passed_step2_moved_away": 0,
    "passed_step3_touched_average": 0,
    "passed_step4_bounced_today": 0,
}

def find_rsi_bounces(rsi, rsi_avg, cfg):
    """
    Checks the last `signal_lookback_days` trading days (1 = the most
    recent candle only). Returns hit dicts, most recent first, and
    tallies FUNNEL_COUNTS for every day checked.
    """
    global FUNNEL_COUNTS
    n = len(rsi)
    hits = []
    for back in range(0, cfg["signal_lookback_days"]):
        i = (n - 1) - back
        passed, d = check_pattern_at(rsi, rsi_avg, i, cfg)
        if not d:
            continue
        FUNNEL_COUNTS["days_checked"] += 1
        if d["has_cross"]:
            FUNNEL_COUNTS["passed_step1_recent_cross_up"] += 1
            if d["separated"]:
                FUNNEL_COUNTS["passed_step2_moved_away"] += 1
                if d["touched"]:
                    FUNNEL_COUNTS["passed_step3_touched_average"] += 1
                    if d["bounced"]:
                        FUNNEL_COUNTS["passed_step4_bounced_today"] += 1
        if passed:
            hits.append(d)
    return hits

# ── Technical signal: RSI bounce off its average ─────────────────
def analyze_rsi_avg_bounce(sym, df_daily):
    """
    Returns dict with score and setup details, or None if the pattern
    is not present in the lookback window.
    """
    if df_daily is None:
        return None

    price   = float(df_daily["Close"].iloc[-1])
    avg_vol = float(df_daily["Volume"].tail(20).mean())
    if price   < CFG["min_price"]:      return None
    if avg_vol < CFG["min_avg_volume"]: return None

    n = len(df_daily)
    if n < CFG["sma_period"] + CFG["rsi_period"] + CFG["rsi_avg_period"] + CFG["cross_max_days_ago"] + 10:
        return None

    close   = df_daily["Close"]
    rsi     = calc_rsi(close, CFG["rsi_period"])
    rsi_avg = rsi.rolling(CFG["rsi_avg_period"]).mean()

    hits = find_rsi_bounces(rsi, rsi_avg, CFG)
    if not hits:
        return None

    s = hits[0]   # most recent
    k, x, pk, tk = s["idx"], s["cross_idx"], s["peak_idx"], s["touch_idx"]
    days_since_signal = (n - 1) - k

    # ── Entry / Stop (reasonable defaults — see header note) ─────────
    entry_price = float(close.iloc[k])
    stop_loss   = float(df_daily["Low"].iloc[pk:k + 1].min())   # lowest low of the pullback
    if entry_price <= stop_loss:
        return None
    risk_pct = (entry_price - stop_loss) / entry_price * 100

    sma   = close.rolling(CFG["sma_period"]).mean()
    sma_k = float(sma.iloc[k])
    above_sma = bool(entry_price > sma_k)
    vol_k  = float(df_daily["Volume"].iloc[k])
    avg20  = float(df_daily["Volume"].iloc[max(0, k-20):k].mean())
    vol_vs_avg = vol_k / avg20 if avg20 > 0 else 0.0
    price_chg_pct = (entry_price / float(close.iloc[k-1]) - 1) * 100
    bounce_pts = s["gap"] - s["gap_prev"]

    # ── Score (0-100) ────────────────────────────────────────────
    score = 30   # base for clearing every gate
    reasons = []
    score += max(0, min(15, (s["peak_gap"] - CFG["min_separation_pts"]) * 2 + 5))
    reasons.append(f"Peak gap +{s['peak_gap']:.1f}")
    score += max(0, min(15, (CFG["touch_pts"] - abs(s["touch_gap"])) / CFG["touch_pts"] * 15))
    reasons.append(f"Touch gap {s['touch_gap']:+.1f}")
    score += max(0, min(15, bounce_pts * 5))
    reasons.append(f"Bounce +{bounce_pts:.1f} pts")
    if 50 <= s["rsi"] <= 65:
        score += 10; reasons.append(f"RSI {s['rsi']:.0f} (sweet spot)")
    elif s["rsi"] > 72:
        score -= 10; reasons.append(f"RSI {s['rsi']:.0f} (overbought)")
    else:
        reasons.append(f"RSI {s['rsi']:.0f}")
    if above_sma:
        score += 10; reasons.append(f"Above SMA{CFG['sma_period']}")
    else:
        reasons.append(f"Below SMA{CFG['sma_period']}")
    if risk_pct > 10:
        score -= 10; reasons.append("WideStop")
    score = round(min(100, max(0, score)))

    return {
        "Score"          : score,
        "Price"          : round(price, 2),
        "Entry_Price"    : round(entry_price, 2),
        "Stop_Loss"      : round(stop_loss, 2),
        "Risk_%"         : round(risk_pct, 1),
        "RSI"            : round(s["rsi"], 1),
        "RSI_Avg"        : round(s["rsi_avg"], 1),
        "Gap_Today"      : round(s["gap"], 1),
        "Cross_Date"     : df_daily.index[x].strftime("%Y-%m-%d"),
        "Days_Since_Cross": k - x,
        "Peak_Gap"       : round(s["peak_gap"], 1),
        "Touch_Date"     : df_daily.index[tk].strftime("%Y-%m-%d"),
        "Touch_Gap"      : round(s["touch_gap"], 1),
        "Price_Chg_%"    : round(price_chg_pct, 1),
        "Vol_vs_20d_Avg" : round(vol_vs_avg, 2),
        "SMA50"          : round(sma_k, 2),
        "Above_SMA50"    : above_sma,
        "Signal_Date"    : df_daily.index[k].strftime("%Y-%m-%d"),
        "Days_Since_Signal": days_since_signal,
        "Flags"          : " | ".join(reasons),
        "_df_daily"      : df_daily,
        "_rsi"           : rsi,
        "_rsi_avg"       : rsi_avg,
        "_sma"           : sma,
        "_idx"           : k,
        "_cross_idx"     : x,
        "_touch_idx"     : tk,
    }
'''

tail = "\n".join(L[330:])


def rep(a, b, count=1):
    global tail
    n = tail.count(a)
    if (count == "all" and n == 0) or (count != "all" and n != count):
        raise SystemExit(f"edit expected {count} match(es), found {n}: {a[:70]!r}")
    tail = tail.replace(a, b)


rep('''LIVE_COLS = ["Ticker","Price","Score","Entry_Price","Stop_Loss",
             "RSI","RSI_Peak","Signal_Date"]
_CW = {"Ticker":8,"Price":10,"Score":7,"Entry_Price":12,"Stop_Loss":11,
       "RSI":7,"RSI_Peak":9,"Signal_Date":13}
_CF = {"Price":"${:.2f}","Score":"{:.0f}","Entry_Price":"${:.2f}",
       "Stop_Loss":"${:.2f}","RSI":"{:.1f}","RSI_Peak":"{:.1f}"}''', '''LIVE_COLS = ["Ticker","Price","Score","Entry_Price","Stop_Loss",
             "RSI","RSI_Avg","Touch_Date"]
_CW = {"Ticker":8,"Price":10,"Score":7,"Entry_Price":12,"Stop_Loss":11,
       "RSI":7,"RSI_Avg":9,"Touch_Date":13}
_CF = {"Price":"${:.2f}","Score":"{:.0f}","Entry_Price":"${:.2f}",
       "Stop_Loss":"${:.2f}","RSI":"{:.1f}","RSI_Avg":"{:.1f}"}''')
rep('''print(f"  Signal: RSI({CFG['rsi_period']}) breaks its most recent peak from the last "
      f"{CFG['peak_window_days']} days (first break), and RSI > its {CFG['rsi_avg_period']}-day average\\n")''',
    '''print(f"  Signal: RSI({CFG['rsi_period']}) crossed above its {CFG['rsi_avg_period']}-day average "
      f"{CFG['cross_min_days_ago']}-{CFG['cross_max_days_ago']} days ago, pulled back to touch it, "
      f"and bounced today\\n")''')
rep('''desc="Checking RSI breakout"''', '''desc="Checking RSI bounce"''')
rep("        r = analyze_rsi_breakout(sym, daily_map[sym])", "        r = analyze_rsi_avg_bounce(sym, daily_map[sym])")
rep('''print(f"  Step 1 — RSI peak in last {CFG['peak_window_days']} days             : {fc['passed_step1_rsi_peak_in_window']}")
print(f"  Step 2 — + RSI above that peak today         : {fc['passed_step2_rsi_above_peak']}")
print(f"  Step 3 — + first break (not broken earlier)  : {fc['passed_step3_first_break']}")
print(f"  Step 4 — + RSI above its {CFG['rsi_avg_period']}-day average        : {fc['passed_step4_rsi_above_avg']}  (= full pattern)")''',
    '''print(f"  Step 1 — RSI crossed above avg {CFG['cross_min_days_ago']}-{CFG['cross_max_days_ago']} days ago : {fc['passed_step1_recent_cross_up']}")
print(f"  Step 2 — + moved >= {CFG['min_separation_pts']:.0f} pts above the avg     : {fc['passed_step2_moved_away']}")
print(f"  Step 3 — + pulled back to touch the avg     : {fc['passed_step3_touched_average']}")
print(f"  Step 4 — + bounced today                     : {fc['passed_step4_bounced_today']}  (= full pattern)")''')
rep('''    print("   signal_lookback_days            1 → 3     (breaks in the last 3 days)")
    print("   peak_window_days                7 → 10")''', '''    print("   signal_lookback_days            1 → 3     (bounces in the last 3 days)")
    print("   touch_pts                     1.5 → 2.5")
    print("   min_separation_pts            3.0 → 2.0")''')
rep('''    "RSI","RSI_Avg","RSI_Peak","Peak_Date","Days_Since_Peak","Break_Pts","RSI_Above_Avg",''',
    '''    "RSI","RSI_Avg","Gap_Today","Cross_Date","Days_Since_Cross","Peak_Gap","Touch_Date","Touch_Gap",''')
rep('''    "RSI_Peak"      : lambda v: f"{v:.1f}",
    "Break_Pts"     : lambda v: f"{v:+.1f}",
    "RSI_Above_Avg" : lambda v: f"{v:+.1f}",''', '''    "Gap_Today"     : lambda v: f"{v:+.1f}",
    "Peak_Gap"      : lambda v: f"{v:+.1f}",
    "Touch_Gap"     : lambda v: f"{v:+.1f}",''')
rep('''             "RSI","RSI_Peak","RSI_Avg","Vol_vs_20d_Avg","Signal_Date"]''',
    '''             "RSI","RSI_Avg","Cross_Date","Touch_Date","Vol_vs_20d_Avg"]''')
rep("Daily RSI High Breakout", "Daily RSI Bounce off Average", "all")
rep("daily_rsi_high_breakout_", "daily_rsi_avg_bounce_", "all")
rep('''  Score           0-100 (how far RSI broke its peak + RSI above its
                  average + RSI in 50-70 + volume + above SMA50 + freshness)
  Entry_Price     signal day's close
  Stop_Loss       lowest low since the RSI peak
  RSI_Peak        the RSI high that was broken (Peak_Date)
  Break_Pts       RSI points above that peak
  RSI_Avg         {CFG['rsi_avg_period']}-day simple average of RSI
  Signal_Date     date RSI broke the peak''', '''  Score           0-100 (separation + how close the touch was + bounce
                  strength + RSI in 50-65 + above SMA50)
  Entry_Price     today's close
  Stop_Loss       lowest low of the pullback (from RSI's peak to today)
  RSI_Avg         {CFG['rsi_avg_period']}-day simple average of RSI
  Cross_Date      day RSI crossed above its average
  Peak_Gap        how far RSI got above its average after the cross
  Touch_Date      day RSI came back to its average (Touch_Gap = RSI - avg)''')
rep('''# Quantity''', '''# Quantity''', 0)  # (template has no sizing map yet)
rep('''# ── Email with CSV attached ───────────────────────────────''', '''# Quantity / Total_Amount per ticker for the email (from the sized df_out)
_SIZE = {}
if {"Ticker", "Quantity", "Total_Amount"} <= set(df_out.columns):
    _SIZE = {t: (q, a) for t, q, a in zip(df_out["Ticker"], df_out["Quantity"], df_out["Total_Amount"])}

# ── Email with CSV attached ───────────────────────────────''')
rep('''        heads = ["Ticker","Price","Score","Entry","Stop","Risk %","RSI",
                 "RSI peak","RSI avg","Vol/20d","Trend","Signal"]''', '''        heads = ["Ticker","Price","Score","Entry","Stop","Risk %","RSI",
                 "RSI avg","Cross","Touch","Qty","Amount"]''')
rep('''            trend = "Above SMA50" if r.get("Above_SMA50") else "Below SMA50"
''', '''            q, a = _SIZE.get(r.get("Ticker"), ("—", None))
''')
rep('''                + td(f'{float(r.get("RSI_Peak",0) or 0):.1f} ({r.get("Peak_Date","—")[5:]})')
                + td(f'{float(r.get("RSI_Avg",0) or 0):.1f}')
                + td(f'x{float(r.get("Vol_vs_20d_Avg",0) or 0):.2f}')
                + td(trend, "color:#16a34a" if r.get("Above_SMA50") else "color:#b45309")
                + td(r.get("Signal_Date","—"), "color:#a78bfa;font-weight:600")''',
    '''                + td(f'{float(r.get("RSI_Avg",0) or 0):.1f}')
                + td(f'{r.get("Cross_Date","—")[5:]} (+{float(r.get("Peak_Gap",0) or 0):.1f})')
                + td(f'{r.get("Touch_Date","—")[5:]} ({float(r.get("Touch_Gap",0) or 0):+.1f})', "color:#a78bfa;font-weight:600")
                + td(f'{q}')
                + td(f'${float(a):,.0f}' if a is not None else "—")''')
rep('''  {cnt} match{'es' if cnt!=1 else ''} — RSI({CFG['rsi_period']}) broke its most recent peak from the
  last {CFG['peak_window_days']} days for the first time, and RSI is above its {CFG['rsi_avg_period']}-day average''',
    '''  {cnt} match{'es' if cnt!=1 else ''} — RSI({CFG['rsi_period']}) crossed above its {CFG['rsi_avg_period']}-day average
  {CFG['cross_min_days_ago']}-{CFG['cross_max_days_ago']} days ago, pulled back to touch it and bounced today''')
rep('''    {fc['passed_step1_rsi_peak_in_window']} RSI peak in last {CFG['peak_window_days']} days &nbsp;→&nbsp;
    {fc['passed_step2_rsi_above_peak']} RSI above peak &nbsp;→&nbsp;
    {fc['passed_step3_first_break']} first break &nbsp;→&nbsp;
    <b style="color:#facc15">{fc['passed_step4_rsi_above_avg']} RSI above avg = full match</b>''',
    '''    {fc['passed_step1_recent_cross_up']} crossed above avg &nbsp;→&nbsp;
    {fc['passed_step2_moved_away']} moved away &nbsp;→&nbsp;
    {fc['passed_step3_touched_average']} touched avg &nbsp;→&nbsp;
    <b style="color:#facc15">{fc['passed_step4_bounced_today']} bounced today = full match</b>''')
rep('''Entry = signal close, Stop = lowest low since the RSI peak (defaults)''',
    '''Entry = today's close, Stop = lowest low of the pullback (defaults) &nbsp;·&nbsp; Qty sized for a $40 max loss''')
rep('''            f"{cnt} matches (RSI first break of its last peak within {CFG['peak_window_days']} days; RSI > {CFG['rsi_avg_period']}-day avg)",''',
    '''            f"{cnt} matches (RSI crossed above its avg {CFG['cross_min_days_ago']}-{CFG['cross_max_days_ago']} days ago, touched it, bounced today)",''')
rep('''            f"{fc['passed_step1_rsi_peak_in_window']} RSI peak in window -> "
            f"{fc['passed_step2_rsi_above_peak']} above peak -> "
            f"{fc['passed_step3_first_break']} first break -> "
            f"{fc['passed_step4_rsi_above_avg']} RSI > avg (=full match)",''',
    '''            f"{fc['passed_step1_recent_cross_up']} crossed up -> "
            f"{fc['passed_step2_moved_away']} moved away -> "
            f"{fc['passed_step3_touched_average']} touched avg -> "
            f"{fc['passed_step4_bounced_today']} bounced (=full match)",''')
rep('''                    f"Peak:{float(r.get('RSI_Peak',0) or 0):.1f}@{r.get('Peak_Date','—')}  "
                    f"Avg:{float(r.get('RSI_Avg',0) or 0):.1f}  "
                    f"{'AboveSMA50' if r.get('Above_SMA50') else 'BelowSMA50'}  "
                    f"Signal:{r.get('Signal_Date','—')}")''',
    '''                    f"Avg:{float(r.get('RSI_Avg',0) or 0):.1f}  "
                    f"Cross:{r.get('Cross_Date','—')}  Touch:{r.get('Touch_Date','—')}  "
                    f"Qty:{_SIZE.get(r.get('Ticker'), ('—', None))[0]}")''')
rep('''        subj = (f"📊 RSI High Breakout — {cnt} signal''', '''        subj = (f"📊 RSI Bounce off Average — {cnt} signal''')
# chart: RSI panel marks cross and touch instead of the broken peak
rep('''        k, pk = r["_idx"], r["_peak_idx"]
        sig_d, peak_d = df_all.index[k], df_all.index[pk]''', '''        k, pk = r["_idx"], r["_touch_idx"]
        sig_d, peak_d = df_all.index[k], df_all.index[pk]
        cross_d = df_all.index[r["_cross_idx"]]''')
rep('''                     f"RSI {r['RSI']:.1f} broke {r['RSI_Peak']:.1f}  |  {r['Signal_Date']}",''',
    '''                     f"RSI {r['RSI']:.1f} vs avg {r['RSI_Avg']:.1f}  |  cross {r['Cross_Date']}  touch {r['Touch_Date']}",''')
rep('''        axr.axhline(r["RSI_Peak"], color="#3b82f6", lw=0.9, ls=":", label=f"Peak {r['RSI_Peak']:.1f}")
''', '')
rep('''        if peak_d in idx:
            axr.scatter([peak_d], [r["RSI_Peak"]], color="#3b82f6", s=30, zorder=6)''', '''        if cross_d in idx:
            axr.scatter([cross_d], [r["_rsi"].loc[cross_d]], color="#22c55e", s=30, zorder=6, label="Cross")
        if peak_d in idx:
            axr.scatter([peak_d], [r["_rsi"].loc[peak_d]], color="#3b82f6", s=30, zorder=6, label="Touch")''')
rep('''  1) RSI made a peak (higher than the day before and after) within
     the last {CFG['peak_window_days']} trading days — the most recent one is used.
  2) Today's RSI is ABOVE that peak.
  3) First break — no day since the peak already closed above it.
  4) Today's RSI is ABOVE its {CFG['rsi_avg_period']}-day average.''', '''  1) RSI crossed ABOVE its average {CFG['cross_min_days_ago']}-{CFG['cross_max_days_ago']} trading days ago.
  2) RSI then rose at least {CFG['min_separation_pts']} points above its average.
  3) RSI pulled back to within {CFG['touch_pts']} points of its average (never more
     than {CFG['max_break_pts']} points below it since the cross).
  4) That touch was in the last {CFG['bounce_within_days']} days and TODAY RSI is rising,
     above its average and moving away from it.''')
rep('''  Entry_Price = signal day's CLOSE
  Stop_Loss   = lowest LOW since the RSI peak''', '''  Entry_Price = today's CLOSE
  Stop_Loss   = lowest LOW of the pullback (from RSI's peak to today)''')
rep('''  Base 20 + RSI points above the peak (0-20) + RSI above its average
  (0-15) + RSI in 50-70 (+10; -10 if above 75) + volume vs 20d avg
  (0-15) + above SMA50 (10) + freshness (0-10); -10 if stop risk > 12%''',
    '''  Base 30 + separation after the cross (0-15) + how close the touch was
  (0-15) + bounce strength (0-15) + RSI in 50-65 (+10; -10 above 72)
  + above SMA50 (10); -10 if stop risk > 10%''')
rep('''  signal_lookback_days            1 → 3
  peak_window_days                7 → 10''', '''  signal_lookback_days            1 → 3
  touch_pts                     1.5 → 2.5
  min_separation_pts            3.0 → 2.0''')

src = HEADER + "\n".join(L[32:113]) + "\n" + LOGIC + "\n" + tail
open(OUT, "w").write(src)
py_compile.compile(OUT, doraise=True)
print(f"✅ wrote {OUT} ({len(src.splitlines())} lines) — compiles")
