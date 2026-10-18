
# ── CONFIG ────────────────────────────────────────────────────────────
CFG = {
    "history_days"          : 550,   # daily bars — ~380 trading days, comfortable
                                      # margin above _clean()'s 220-bar floor

    # ── Indicator periods ───────────────────────────────────────
    "jma_period"             : 13,
    "jma_phase"              : 40,
    "jma_power"              : 2,
    "ema_period"             : 8,
    "sma_period"             : 50,   # trend context only (score bonus, not a gate)

    "signal_lookback_days"   : 1,    # 1 = the confirm day must be the most
                                      # recent daily candle

    # ── Filters ─────────────────────────────────────────────────
    "min_avg_volume"         : 100_000,
    "min_price"              : 10.0,

    "batch_size"             : 50,
    "batch_sleep"            : 1.5,
}

# ── Indicators ───────────────────────────────────────────────────────
def calc_ema(close, n):
    return close.ewm(span=n, adjust=False).mean()

