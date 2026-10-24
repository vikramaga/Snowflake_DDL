# Test run of monday_red_candle_breakout_v1.py outside Monday hours (via ClaudeRunner):
# FORCE uses the latest completed session as "today"; TEST marks the email [TEST]
# and saves no state.
import os, runpy
os.environ["FORCE"] = "1"
os.environ["TEST"] = "1"
runpy.run_path("monday_red_candle_breakout_v1.py", run_name="__main__")
