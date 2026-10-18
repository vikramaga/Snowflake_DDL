# Win-rate backtest wrapper (see claude_run/tools/backtest_winrate.py)
import runpy
runpy.run_path("claude_run/tools/backtest_winrate.py", run_name="bt",
               init_globals={"SCANNER": "daily_rsi_high_breakout_v1.py"})
