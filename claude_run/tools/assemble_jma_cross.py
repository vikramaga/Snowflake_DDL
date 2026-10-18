# One-off build run by ClaudeRunner: assembles daily_jma_cross_ema8_v1.py
# from the new sections in claude_run/build/jma_cross/ plus the unchanged
# boilerplate / download / ticker-list sections of daily_rsi_high_breakout_v1.py
# and calc_jma from daily_3candle_jma_reclaim_v1.py,
# checks it is byte-identical to the locally tested file (SHA-256), compiles it,
# adds it to the 06:00 schedule and commits.
import hashlib, py_compile, subprocess, sys
sys.path.insert(0, "claude_run/build/jma_cross")
from gen_email import build_email_section

EXPECTED = "4734539fefd5853f4a2c07380a044ee4c070a4a9f96cba8b74b25202ee1952ca"
OUT = "daily_jma_cross_ema8_v1.py"
B = "claude_run/build/jma_cross/"

tpl = open("daily_rsi_high_breakout_v1.py").read().split("\n")
jtp = open("daily_3candle_jma_reclaim_v1.py").read().split("\n")
def lines(src_lines, a, b):           # 1-based inclusive, like sed -n a,bp
    return "\n".join(src_lines[a-1:b]) + "\n"

parts = [open(B + "p_head.py").read(), lines(tpl, 34, 112), open(B + "p_cfg.py").read(),
         lines(jtp, 146, 182), open(B + "p_logic.py").read(),
         lines(tpl, 331, 376), open(B + "p_live.py").read(), lines(tpl, 410, 487),
         open(B + "p_scan.py").read(), build_email_section(open("claude_run/build/ma_support/p_email.py").read()), open(B + "p_chart.py").read()]
src = "".join(parts)
h = hashlib.sha256(src.encode()).hexdigest()
print("sha256", h)
if h != EXPECTED:
    sys.exit("❌ assembled file differs from the tested one — nothing committed")
open(OUT, "w").write(src)
py_compile.compile(OUT, doraise=True)
print(f"✅ {OUT} identical to the tested file and compiles")

sched = open("claude_run/schedule.txt").read()
if OUT not in sched:
    open("claude_run/schedule.txt", "w").write(sched.rstrip("\n") + "\n" + OUT + "\n")
    print("✅ added to claude_run/schedule.txt")

def git(*a):
    print("+ git", " ".join(a)); subprocess.check_call(["git", *a])
git("config", "user.name", "Scanner Bot")
git("config", "user.email", "vikram-scanner-bot@users.noreply.github.com")
git("add", OUT, "claude_run/schedule.txt")
git("commit", "-m", """Add daily JMA-cross-above-EMA8 scanner (stop = cross price) and schedule it at 06:00 IST

Signal (confirm day = latest daily candle):
1. Cross day: JMA(13,40,2) was at or below EMA8 the day before and is above it
2. Confirm day (next day): close above JMA
3. Confirm close above the cross price
Cross price = JMA/EMA8 intersection, interpolated between the day before the
cross and the cross day. It is the stop loss (requested). Entry = confirm close
(default). Quantity/Total_Amount for a $40 max loss. Note: with the repo JMA
(slower than EMA8) this is a dip-and-reclaim setup; the user chose the literal rule.

Template: daily_rsi_high_breakout_v1.py; calc_jma from daily_3candle_jma_reclaim_v1.py.

Verified: positive synthetic dip-and-reclaim fires and an independent segment
intersection matches the stop (129.6594); negative controls fail at step 1
(no fresh cross; JMA already above), step 2 (next close below JMA) and step 3
(close above JMA but below the cross price); full script run with mocked
data/SMTP for 6 matches and 0 matches (CSV sizing <= $40, chart, TradingView
list, decoded email); py_compile; no template leftovers.

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01WG6bhVDgntdGBGqtFYno57""")
for i in range(5):
    if subprocess.call(["git", "pull", "--rebase", "origin", "master"]) == 0 and \
       subprocess.call(["git", "push", "origin", "HEAD:master"]) == 0:
        print("✅ pushed"); break
else:
    sys.exit("❌ push failed")
