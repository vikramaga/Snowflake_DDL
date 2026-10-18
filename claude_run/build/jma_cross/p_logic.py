
def cross_price(jma, ema, x):
    """
    Price where JMA and EMA8 intersect between bar x-1 and bar x
    (straight-line interpolation of both lines). Returns (price, t)
    with t in [0, 1] = fraction of the way from bar x-1 to bar x.
    """
    j0, j1 = float(jma.iloc[x-1]), float(jma.iloc[x])
    e0, e1 = float(ema.iloc[x-1]), float(ema.iloc[x])
    d0, d1 = j0 - e0, j1 - e1          # d0 <= 0 < d1 on a cross
    t = d0 / (d0 - d1) if d0 != d1 else 1.0
    t = min(1.0, max(0.0, t))
    return e0 + t * (e1 - e0), t

def check_pattern_at(close, jma, ema, i, cfg):
    """
    Checks the JMA-over-EMA8 cross + confirmation with the CONFIRM day
    at bar `i` (the cross day is bar i-1).
    SINGLE SOURCE OF TRUTH for the pattern logic.

    Computes EVERY stage's pass/fail regardless of earlier failures
    (guarding only on missing data), so callers can diagnose exactly
    which stage is filtering results out — see FUNNEL_COUNTS below.

    Returns (passed: bool, details: dict).
    """
    x = i - 1                                   # cross day
    if x - 1 < 0 or i >= len(close):
        return False, {}
    vals = [jma.iloc[x-1], jma.iloc[x], ema.iloc[x-1], ema.iloc[x], jma.iloc[i], close.iloc[i]]
    if any(np.isnan(v) for v in vals):
        return False, {}

    crossed = bool(jma.iloc[x-1] <= ema.iloc[x-1] and jma.iloc[x] > ema.iloc[x])   # Step 1
    cp, t   = cross_price(jma, ema, x) if crossed else (np.nan, np.nan)
    c_i     = float(close.iloc[i])
    confirm = bool(c_i > float(jma.iloc[i]))                                       # Step 2
    valid   = bool(crossed and c_i > cp)                                           # Step 3

    passed = crossed and confirm and valid
    return passed, {
        "idx": i, "cross_idx": x, "cross_price": cp, "cross_t": t,
        "close": c_i, "jma": float(jma.iloc[i]), "ema": float(ema.iloc[i]),
        "crossed": crossed, "confirm": confirm, "valid": valid,
    }

# ── Diagnostic funnel — tallies how far each ticker-day gets through
#    the pattern, across the full universe scan, so a 0-match run can
#    be diagnosed empirically instead of guessed at ─────────────────
FUNNEL_COUNTS = {
    "days_checked": 0,
    "passed_step1_jma_crossed_above_ema8": 0,
    "passed_step2_next_close_above_jma": 0,
    "passed_step3_close_above_cross_price": 0,
}

def find_jma_crosses(close, jma, ema, cfg):
    """
    Checks the last `signal_lookback_days` trading days as the confirm
    day (1 = the most recent candle only). Returns hit dicts, most
    recent first, and tallies FUNNEL_COUNTS for every day checked.
    """
    global FUNNEL_COUNTS
    n = len(close)
    hits = []
    for back in range(0, cfg["signal_lookback_days"]):
        i = (n - 1) - back
        passed, d = check_pattern_at(close, jma, ema, i, cfg)
        if not d:
            continue
        FUNNEL_COUNTS["days_checked"] += 1
        if d["crossed"]:
            FUNNEL_COUNTS["passed_step1_jma_crossed_above_ema8"] += 1
            if d["confirm"]:
                FUNNEL_COUNTS["passed_step2_next_close_above_jma"] += 1
                if d["valid"]:
                    FUNNEL_COUNTS["passed_step3_close_above_cross_price"] += 1
        if passed:
            hits.append(d)
    return hits

# ── Technical signal: JMA cross above EMA8 + confirmation ───────────
def analyze_jma_cross(sym, df_daily):
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
    if n < CFG["sma_period"] + 30:
        return None

    close = df_daily["Close"]
    jma   = calc_jma(close, CFG["jma_period"], CFG["jma_phase"], CFG["jma_power"])
    ema   = calc_ema(close, CFG["ema_period"])

    hits = find_jma_crosses(close, jma, ema, CFG)
    if not hits:
        return None

    s = hits[0]   # most recent
    k, x = s["idx"], s["cross_idx"]
    days_since_signal = (n - 1) - k

    # ── Entry / Stop: Stop = cross price (requested); Entry = confirm close
    entry_price = s["close"]
    stop_loss   = s["cross_price"]
    if price <= stop_loss:              # (lookback > 1) already back below the stop
        return None
    risk_pct = (entry_price - stop_loss) / entry_price * 100

    gap_pct   = (s["jma"] / s["ema"] - 1) * 100                 # JMA above EMA8 on confirm day
    above_jma = (entry_price / s["jma"] - 1) * 100              # close above JMA on confirm day
    sma   = close.rolling(CFG["sma_period"]).mean()
    sma_k = float(sma.iloc[k])
    above_sma = bool(entry_price > sma_k)
    vol_k  = float(df_daily["Volume"].iloc[k])
    avg20  = float(df_daily["Volume"].iloc[max(0, k-20):k].mean())
    vol_vs_avg = vol_k / avg20 if avg20 > 0 else 0.0

    # ── Score (0-100) ────────────────────────────────────────────
    score = 30   # base for clearing every gate
    reasons = []
    score += max(0, min(15, gap_pct * 15))
    reasons.append(f"JMA>EMA8 +{gap_pct:.2f}%")
    if 0 < above_jma <= 3:
        score += 15; reasons.append(f"Close +{above_jma:.1f}% over JMA")
    elif above_jma > 6:
        score -= 5;  reasons.append(f"Extended +{above_jma:.1f}% over JMA")
    else:
        score += 8;  reasons.append(f"Close +{above_jma:.1f}% over JMA")
    score += max(0, min(15, (8 - risk_pct) / 8 * 15))
    reasons.append(f"Risk {risk_pct:.1f}%")
    score += max(0, min(15, (vol_vs_avg - 1) * 15))
    reasons.append(f"Vol/20d x{vol_vs_avg:.2f}")
    if above_sma:
        score += 10; reasons.append(f"Above SMA{CFG['sma_period']}")
    else:
        reasons.append(f"Below SMA{CFG['sma_period']}")
    if risk_pct > 12:
        score -= 10; reasons.append("WideStop")
    score = round(min(100, max(0, score)))

    return {
        "Score"          : score,
        "Price"          : round(price, 2),
        "Entry_Price"    : round(entry_price, 2),
        "Stop_Loss"      : round(stop_loss, 2),
        "Risk_%"         : round(risk_pct, 1),
        "Cross_Price"    : round(stop_loss, 2),
        "Cross_Date"     : df_daily.index[x].strftime("%Y-%m-%d"),
        "Confirm_Date"   : df_daily.index[k].strftime("%Y-%m-%d"),
        "JMA"            : round(s["jma"], 2),
        "EMA8"           : round(s["ema"], 2),
        "Close_vs_JMA_%" : round(above_jma, 2),
        "JMA_EMA8_Gap_%" : round(gap_pct, 2),
        "Vol_vs_20d_Avg" : round(vol_vs_avg, 2),
        "SMA50"          : round(sma_k, 2),
        "Above_SMA50"    : above_sma,
        "Signal_Date"    : df_daily.index[k].strftime("%Y-%m-%d"),
        "Days_Since_Signal": days_since_signal,
        "Flags"          : " | ".join(reasons),
        "_df_daily"      : df_daily,
        "_jma"           : jma,
        "_ema"           : ema,
        "_sma"           : sma,
        "_idx"           : k,
        "_cross_idx"     : x,
        "_cross_t"       : s["cross_t"],
    }

