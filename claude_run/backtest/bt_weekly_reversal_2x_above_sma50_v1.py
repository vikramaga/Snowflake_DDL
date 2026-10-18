# Win-rate backtest wrapper (see claude_run/tools/backtest_winrate.py)
import runpy
runpy.run_path("claude_run/tools/backtest_winrate.py", run_name="bt",
               init_globals={"SCANNER": "weekly_reversal_2x_above_sma50_v1.py"})
