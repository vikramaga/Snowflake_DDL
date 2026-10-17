# ============================================================
# NASDAQ — Daily MA Support: EMA21 / SMA50 (v1)
# ============================================================
#
# IDEA: find stocks that have been taking support on the same moving
# average — EMA21 or SMA50 — most of the time over the last ~2 months,
# and are back near that MA today (a pullback to proven support).
#
# For EACH of EMA21 and SMA50, over the last 42 trading days before
# today (support_window_days):
#   • BOUNCE: a "touch" day has its LOW within 1.5% above the MA or
#     below it (touch_pct), while the CLOSE holds no more than 1% below
#     the MA (hold_tol_pct). Touch days within 3 bars of each other are
#     one event (min_touch_gap). The event counts as a bounce if price
#     then CLOSES at least 2% above the MA within 5 days (bounce_pct,
#     bounce_days).
#   • HELD %: share of days whose close is no more than 1% below the MA.
# The SUPPORT MA is the one with more bounces (tie → higher Held %;
# tie → EMA21).
#
# SIGNAL (all required, on the most recent daily candle):
#   1. SUPPORT FOUND: the support MA has >= 3 bounces in the window.
#   2. HELD MOST OF THE TIME: Held % on that MA >= 80%.
#   3. RISING MA: that MA is higher than 10 days ago.
#   4. NEAR IT NOW: today's close is between 1% below and 3% above
#      that same MA.
#
# DATA — only 1 download per ticker: daily bars.
#
# OUTPUT: Entry_Price and Stop_Loss are reasonable defaults (Entry =
# today's close; Stop = 2% below the support MA), not explicitly
# requested. Quantity / Total_Amount come from position_sizing.py
# (max loss $40 per trade).
#
# SINGLE PASS — purely technical, no fundamentals fetch.
#
# ============================================================

