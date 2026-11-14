# Research (2026-11-14): how often does the Triangle Breakout + FVG Entry pattern fire?
# Runs daily_triangle_fvg_breakout_v1.py unchanged except signal_lookback_days = 60,
# with the email secrets removed (no email). Matches = tickers that fired on any of
# the last 60 sessions (the latest signal per ticker is reported).
import os
for k in ("GMAIL_USER", "GMAIL_PASS", "EMAIL_TO"):
    os.environ.pop(k, None)
src = open("daily_triangle_fvg_breakout_v1.py").read()
old = '"signal_lookback_days"    : 1,'
assert src.count(old) == 1
src = src.replace(old, '"signal_lookback_days"    : 60,')
src = src.replace('f"daily_triangle_fvg_breakout_{ts}.csv"', 'f"tri_lookback60_{ts}.csv"')
exec(compile(src, "daily_triangle_fvg_breakout_v1.py", "exec"), {"__name__": "__main__"})
