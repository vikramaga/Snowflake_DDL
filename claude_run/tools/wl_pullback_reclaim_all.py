# Runs wl_pullback_reclaim_check.py over every watchlist in the TradingView account.
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.environ["TV_WATCHLIST"] = "*"
import wl_pullback_reclaim_check as w
w.main()
