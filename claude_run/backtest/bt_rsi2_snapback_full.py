# Full-universe run of the RSI(2) snap-back replay (for the $5,000 portfolio simulation).
import os, runpy
os.environ["BT_SAMPLE"] = "100000"
runpy.run_path("claude_run/backtest/bt_rsi2_snapback.py", run_name="__main__")
