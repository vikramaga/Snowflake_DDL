# Win-rate backtest wrapper (see claude_run/tools/backtest_winrate.py)
import runpy
runpy.run_path("claude_run/tools/backtest_winrate.py", run_name="bt",
               init_globals={"SCANNER": "daily_3candle_jma_reclaim_v1.py"})
