# Backtest smoke test: every scanner, 40 stocks, 15 days - checks loading and real-data calls
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import backtest_engine as be
be.main(0, 1, quick=True)
