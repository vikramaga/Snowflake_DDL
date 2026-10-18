# ============================================================
# NASDAQ — Daily JMA Cross Above EMA8 + Confirmation (v1)
# ============================================================
#
# SIGNAL (all required). JMA = repo-standard JMA(13, phase 40, power 2)
# (corrected calc_jma, steady-state gain 1.0); EMA8 = 8-day EMA of close.
#
#   1. CROSS DAY: JMA was at or below EMA8 on the previous day and is
#      ABOVE EMA8 on the cross day (JMA crossed above EMA8 from below).
#
#   2. CONFIRM DAY (the day after the cross): the daily CLOSE is above
#      JMA. The confirm day is the signal day.
#
#   3. VALID STOP: the confirm-day close is above the CROSS PRICE, so
#      the stop sits below the entry.
#
# CROSS PRICE = the price where JMA and EMA8 intersect. The two lines
# cross between the day before the cross and the cross day; the exact
# point is found by straight-line interpolation of both lines between
# those two bars. It is the STOP LOSS (as requested).
#
# By default the confirm day must be the most recent daily candle
# (signal_lookback_days = 1).
#
# DATA — only 1 download per ticker: daily bars.
#
# OUTPUT: Stop_Loss = the cross price (requested). Entry_Price = the
# confirm-day close (reasonable default, not explicitly requested).
# Quantity / Total_Amount come from position_sizing.py (max loss $40).
# SMA50 is shown for trend context and adds a small score bonus; it is
# NOT a gate.
#
# SINGLE PASS — purely technical, no fundamentals fetch.
#
# ============================================================

