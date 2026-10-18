# Win-rate backtest wrapper (see claude_run/tools/backtest_winrate.py)
import runpy
runpy.run_path("claude_run/tools/backtest_winrate.py", run_name="bt",
               init_globals={"SCANNER": "monthly_3candle_reclaim_jma_ema8_v1.py"})
