# Research backtest wrapper (see claude_run/tools/backtest_research.py)
import runpy
runpy.run_path("claude_run/tools/backtest_research.py", run_name="rs",
               init_globals={"SCANNER": "daily_price_jma_ema21_sma50_volume_v1.py"})
