# Research backtest wrapper (see claude_run/tools/backtest_research.py)
import runpy
runpy.run_path("claude_run/tools/backtest_research.py", run_name="rs",
               init_globals={"SCANNER": "daily_rsi_high_breakout_v1.py"})
