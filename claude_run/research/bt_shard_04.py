# Backtest shard 04 of 12 - see backtest_engine.py
import os, sys, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import backtest_engine as be
be.N_TICKERS = 1000
time.sleep(4 * 15)
be.main(4, 12)
