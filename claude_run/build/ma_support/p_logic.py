
# ── CONFIG ────────────────────────────────────────────────────────────
CFG = {
    "history_days"          : 550,   # daily bars — ~380 trading days, comfortable
                                      # margin above _clean()'s 220-bar floor

    # ── Indicator periods ───────────────────────────────────────
    "ema_period"             : 21,
    "sma_period"             : 50,

    # ── Support analysis (last ~2 months) ───────────────────────
    "support_window_days"    : 42,   # trading days before today (~2 months)
    "touch_pct"              : 1.5,  # low within this % ABOVE the MA (or below it) = touch
    "hold_tol_pct"           : 1.0,  # close may be at most this % BELOW the MA and still "hold"
    "min_touch_gap"          : 3,    # touch days closer than this are one event
    "bounce_pct"             : 2.0,  # after a touch, a close this % above the MA ...
    "bounce_days"            : 5,    # ... within this many days = a bounce (support worked)
    "min_bounces"            : 3,    # step 1
    "min_held_ratio"         : 0.80, # step 2: share of window closes held vs the MA
    "slope_days"             : 10,   # step 3: MA today > MA this many days ago
    "require_rising_ma"      : True,
    "near_below_pct"         : 1.0,  # step 4: today's close from 1% below ...
    "near_above_pct"         : 3.0,  # ... to 3% above the support MA

    "stop_buffer_pct"        : 2.0,  # Stop = support MA x (1 - 2%)
    "signal_lookback_days"   : 1,    # 1 = the setup must be on the most recent candle

    # ── Filters ─────────────────────────────────────────────────
    "min_avg_volume"         : 100_000,
    "min_price"              : 10.0,

    "batch_size"             : 50,
    "batch_sleep"            : 1.5,
}

# ── Indicators ───────────────────────────────────────────────────────
def calc_ema(close, n):
    return close.ewm(span=n, adjust=False).mean()

def calc_sma(close, n):
    return close.rolling(n).mean()

def ma_support_stats(close, low, ma, i, cfg):
    """
    Support statistics of `ma` over the window [i - W, i - 1] (the days
    BEFORE bar i). Returns dict with bounces, held_ratio, touch events and
    the bar of the last bounce. Bounce confirmation may use closes up to
    bar i (today) — today itself is not a touch candidate.
    """
    W   = cfg["support_window_days"]
    s, e = i - W, i                       # window = s .. e-1
    tp  = cfg["touch_pct"] / 100.0
    ht  = cfg["hold_tol_pct"] / 100.0
    bp  = cfg["bounce_pct"] / 100.0
    c, l, m = close.values, low.values, ma.values

    held = 0
    touch_bars = []
    for t in range(s, e):
        if np.isnan(m[t]):
            continue
        holds = c[t] >= m[t] * (1 - ht)
        held += holds
        if holds and l[t] <= m[t] * (1 + tp):
            touch_bars.append(t)

    # group touch days into events
    events = []
    for t in touch_bars:
        if events and t - events[-1][1] <= cfg["min_touch_gap"]:
            events[-1][1] = t
        else:
            events.append([t, t])

    bounces, last_bounce = 0, None
    for a, b in events:
        hi = min(b + cfg["bounce_days"], i)
        if any(c[k] >= m[k] * (1 + bp) for k in range(b + 1, hi + 1) if not np.isnan(m[k])):
            bounces += 1
            last_bounce = b
    return {"bounces": bounces, "held_ratio": held / W, "touch_events": len(events),
            "last_bounce": last_bounce}

def check_pattern_at(close, low, ema, sma, i, cfg):
    """
    Checks the MA-support setup with "today" at bar `i`.
    SINGLE SOURCE OF TRUTH for the pattern logic.

    Computes EVERY stage's pass/fail regardless of earlier failures
    (guarding only on missing data), so callers can diagnose exactly
    which stage is filtering results out — see FUNNEL_COUNTS below.

    Returns (passed: bool, details: dict).
    """
    W = cfg["support_window_days"]
    if i - W - cfg["slope_days"] < 0 or i >= len(close):
        return False, {}
    if np.isnan(sma.iloc[i - W]) or np.isnan(ema.iloc[i - W]):
        return False, {}

    st = {"EMA21": ma_support_stats(close, low, ema, i, cfg),
          "SMA50": ma_support_stats(close, low, sma, i, cfg)}
    # support MA: more bounces, then higher held ratio, then EMA21
    name = max(["EMA21", "SMA50"],
               key=lambda k: (st[k]["bounces"], st[k]["held_ratio"], k == "EMA21"))
    ma   = ema if name == "EMA21" else sma
    s    = st[name]
    m_now  = float(ma.iloc[i])
    m_prev = float(ma.iloc[i - cfg["slope_days"]])
    c_now  = float(close.iloc[i])
    dist   = (c_now / m_now - 1) * 100

    found   = s["bounces"] >= cfg["min_bounces"]                                  # Step 1
    held    = s["held_ratio"] >= cfg["min_held_ratio"]                            # Step 2
    rising  = (m_now > m_prev) or not cfg["require_rising_ma"]                     # Step 3
    near    = -cfg["near_below_pct"] <= dist <= cfg["near_above_pct"]              # Step 4

    passed = found and held and rising and near
    return passed, {
        "idx": i, "ma_name": name, "ma": m_now, "ma_prev": m_prev, "dist": dist,
        "bounces": s["bounces"], "held_ratio": s["held_ratio"],
        "touch_events": s["touch_events"], "last_bounce": s["last_bounce"],
        "stats": st,
        "found": found, "held": held, "rising": rising, "near": near,
    }

# ── Diagnostic funnel — tallies how far each ticker-day gets through
#    the pattern, across the full universe scan, so a 0-match run can
#    be diagnosed empirically instead of guessed at ─────────────────
FUNNEL_COUNTS = {
    "days_checked": 0,
    "passed_step1_support_found": 0,
    "passed_step2_held_most_of_time": 0,
    "passed_step3_rising_ma": 0,
    "passed_step4_near_ma_now": 0,
}

def find_ma_support(close, low, ema, sma, cfg):
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
        passed, d = check_pattern_at(close, low, ema, sma, i, cfg)
        if not d:
            continue
        FUNNEL_COUNTS["days_checked"] += 1
        if d["found"]:
            FUNNEL_COUNTS["passed_step1_support_found"] += 1
            if d["held"]:
                FUNNEL_COUNTS["passed_step2_held_most_of_time"] += 1
                if d["rising"]:
                    FUNNEL_COUNTS["passed_step3_rising_ma"] += 1
                    if d["near"]:
                        FUNNEL_COUNTS["passed_step4_near_ma_now"] += 1
        if passed:
            hits.append(d)
    return hits

# ── Technical signal: daily MA support ───────────────────────────
def analyze_ma_support(sym, df_daily):
    """
    Returns dict with score and setup details, or None if the setup
    is not present today.
    """
    if df_daily is None:
        return None

    price   = float(df_daily["Close"].iloc[-1])
    avg_vol = float(df_daily["Volume"].tail(20).mean())
    if price   < CFG["min_price"]:      return None
    if avg_vol < CFG["min_avg_volume"]: return None

    n = len(df_daily)
    if n < CFG["sma_period"] + CFG["support_window_days"] + CFG["slope_days"] + 10:
        return None

    close = df_daily["Close"]
    low   = df_daily["Low"]
    ema   = calc_ema(close, CFG["ema_period"])
    sma   = calc_sma(close, CFG["sma_period"])

    hits = find_ma_support(close, low, ema, sma, CFG)
    if not hits:
        return None

    s = hits[0]   # most recent
    k = s["idx"]
    days_since_signal = (n - 1) - k

    # ── Entry / Stop (reasonable defaults — see header note) ─────────
    entry_price = float(close.iloc[k])
    stop_loss   = s["ma"] * (1 - CFG["stop_buffer_pct"] / 100.0)
    if entry_price <= stop_loss:
        return None
    risk_pct = (entry_price - stop_loss) / entry_price * 100

    slope_pct = (s["ma"] / s["ma_prev"] - 1) * 100
    vol_k  = float(df_daily["Volume"].iloc[k])
    avg20  = float(df_daily["Volume"].iloc[max(0, k-20):k].mean())
    vol_vs_avg = vol_k / avg20 if avg20 > 0 else 0.0
    other = "SMA50" if s["ma_name"] == "EMA21" else "EMA21"
    st_o  = s["stats"][other]

    # ── Score (0-100) ────────────────────────────────────────────
    score = 20   # base for clearing every gate
    reasons = [f"Support {s['ma_name']}"]
    score += max(0, min(20, (s["bounces"] - CFG["min_bounces"] + 1) * 5))
    reasons.append(f"{s['bounces']} bounces")
    score += max(0, min(20, (s["held_ratio"] - CFG["min_held_ratio"])
                         / (1 - CFG["min_held_ratio"]) * 20))
    reasons.append(f"Held {s['held_ratio']*100:.0f}%")
    score += max(0, 20 * (1 - abs(s["dist"]) / CFG["near_above_pct"]))
    reasons.append(f"{s['dist']:+.1f}% from MA")
    score += max(0, min(10, slope_pct * 2))
    reasons.append(f"MA slope {slope_pct:+.1f}%/{CFG['slope_days']}d")
    if vol_vs_avg < 1.0:
        score += 10; reasons.append(f"Quiet pullback (vol x{vol_vs_avg:.2f})")
    else:
        reasons.append(f"Vol x{vol_vs_avg:.2f}")
    if st_o["bounces"] >= CFG["min_bounces"]:
        reasons.append(f"{other} also {st_o['bounces']} bounces")
    if risk_pct > 8:
        score -= 10; reasons.append("WideStop")
    score = round(min(100, max(0, score)))

    lb = s["last_bounce"]
    return {
        "Score"          : score,
        "Price"          : round(price, 2),
        "Entry_Price"    : round(entry_price, 2),
        "Stop_Loss"      : round(stop_loss, 2),
        "Risk_%"         : round(risk_pct, 1),
        "Support_MA"     : s["ma_name"],
        "MA_Value"       : round(s["ma"], 2),
        "Dist_to_MA_%"   : round(s["dist"], 2),
        "Bounces"        : s["bounces"],
        "Held_%"         : round(s["held_ratio"] * 100, 1),
        "EMA21_Bounces"  : s["stats"]["EMA21"]["bounces"],
        "EMA21_Held_%"   : round(s["stats"]["EMA21"]["held_ratio"] * 100, 1),
        "SMA50_Bounces"  : s["stats"]["SMA50"]["bounces"],
        "SMA50_Held_%"   : round(s["stats"]["SMA50"]["held_ratio"] * 100, 1),
        "MA_Slope_%"     : round(slope_pct, 2),
        "Last_Bounce"    : df_daily.index[lb].strftime("%Y-%m-%d") if lb is not None else "",
        "Vol_vs_20d_Avg" : round(vol_vs_avg, 2),
        "Signal_Date"    : df_daily.index[k].strftime("%Y-%m-%d"),
        "Days_Since_Signal": days_since_signal,
        "Flags"          : " | ".join(reasons),
        "_df_daily"      : df_daily,
        "_ema"           : ema,
        "_sma"           : sma,
        "_idx"           : k,
    }

