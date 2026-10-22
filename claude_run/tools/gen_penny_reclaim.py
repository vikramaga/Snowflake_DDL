# Generates the penny-stock ($1-$10) copy of the JMA/EMA8/SMA21/SMA50 reclaim +
# RSI 57-63 scanner:
#   daily_jma_reclaim_rsi_v1.py -> daily_penny_jma_reclaim_rsi_v1.py
# The pattern logic and the existing 1% minimum-ATR filter are unchanged.
# Edits (each checked): price universe 1 <= close <= 10, "Penny ($1-10)" in
# titles, "PENNY" in the email subject, daily_penny_* output file names.
# Usage: python gen_penny_reclaim.py [outdir]
import os, sys, py_compile

OUTDIR = sys.argv[1] if len(sys.argv) > 1 else "."
SRC, OUT = "daily_jma_reclaim_rsi_v1.py", "daily_penny_jma_reclaim_rsi_v1.py"
s = open(SRC).read()

def rep(a, b, n=1):
    global s
    k = s.count(a)
    if (n == "any" and k == 0) or (n != "any" and k != n):
        sys.exit(f"❌ {SRC}: expected {n} match(es), found {k}: {a[:70]!r}")
    s = s.replace(a, b)

rep('    "min_price"              : 10.0,\n',
    '    "min_price"              : 1.0,    # penny universe: $1 - $10\n'
    '    "max_price"              : 10.0,\n')
rep('    if price   < CFG["min_price"]:      return None\n',
    '    if price   < CFG["min_price"]:      return None\n'
    '    if price   > CFG["max_price"]:      return None\n')
rep("# NASDAQ — Daily Reclaim of", "# NASDAQ — Daily Penny ($1-10) Reclaim of")
rep("Daily JMA/EMA8/SMA21/SMA50 Reclaim", "Daily Penny ($1-10) JMA/EMA8/SMA21/SMA50 Reclaim", 5)
rep('f"daily_jma_reclaim_rsi_', 'f"daily_penny_jma_reclaim_rsi_', 2)
rep('f"tv_daily_jma_reclaim_rsi_', 'f"tv_daily_penny_jma_reclaim_rsi_')
rep('subj = (f"📊 MA Reclaim + RSI', 'subj = (f"📊 PENNY MA Reclaim + RSI')
rep("price >= 10 USD", "price 1-10 USD, ATR >= 1% of price")
rep('min_price                      10 → 5', 'min_atr_pct                   1.0 → 0.5')

path = os.path.join(OUTDIR, OUT)
open(path, "w").write(s)
py_compile.compile(path, doraise=True)
print(f"✅ wrote {path}")
