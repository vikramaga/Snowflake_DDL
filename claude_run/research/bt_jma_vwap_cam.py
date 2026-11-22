# One-scanner backtest of jma_vwap_cam_breakout_v1.py - same engine, universe and rules as the 22 Nov full backtest
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import backtest_engine as be
be.N_TICKERS = 1000
be.scheduled = lambda: ["jma_vwap_cam_breakout_v1.py"]
be.main(0, 1)
