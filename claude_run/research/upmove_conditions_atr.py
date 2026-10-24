# Volatility-normalised run of upmove_conditions_research.py:
# GOOD = +3 x ATR14 before -1.5 x ATR14 within 20 sessions (2:1, breakeven 33.3%).
import os, runpy
os.environ["UP_MODE"] = "atr"
runpy.run_path("claude_run/research/upmove_conditions_research.py", run_name="__main__")
