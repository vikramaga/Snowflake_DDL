# Win-rate backtest wrapper for the strategy scanner: 2 years, 0.75R target,
# 20-day max hold (see claude_run/tools/backtest_winrate.py)
import os, runpy
os.environ.setdefault("R_MULT", "0.75")
os.environ.setdefault("BT_DAYS", "500")
runpy.run_path("claude_run/tools/backtest_winrate.py", run_name="bt",
               init_globals={"SCANNER": "daily_sweep_sma50_strategy_v1.py"})
