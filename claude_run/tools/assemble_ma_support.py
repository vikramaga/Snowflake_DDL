# One-off build run by ClaudeRunner: assembles daily_ma_support_ema21_sma50_v1.py
# from the new sections in claude_run/build/ma_support/ plus the unchanged
# boilerplate / download / ticker-list sections of daily_rsi_high_breakout_v1.py,
# checks it is byte-identical to the locally tested file (SHA-256), compiles it,
# adds it to the 06:00 schedule and commits.
import hashlib, py_compile, subprocess, sys

EXPECTED = "7d67299e76f5d72dc5cf0965751bf9288c6286314ff02b378d757c2fe522843e"
OUT = "daily_ma_support_ema21_sma50_v1.py"
B = "claude_run/build/ma_support/"

tpl = open("daily_rsi_high_breakout_v1.py").read().split("\n")
def lines(a, b):                      # 1-based inclusive, like sed -n a,bp
    return "\n".join(tpl[a-1:b]) + "\n"

parts = [open(B + "p_head.py").read(), lines(34, 112), open(B + "p_logic.py").read(),
         lines(331, 376), open(B + "p_live.py").read(), lines(410, 487),
         open(B + "p_scan.py").read(), open(B + "p_email.py").read()]
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
git("commit", "-m", """Add daily MA support scanner (EMA21 / SMA50) and schedule it at 06:00 IST

Signal on the latest daily candle, over the last 42 trading days:
1. Support MA (EMA21 or SMA50, whichever has more bounces) has >= 3 bounces
   (low within 1.5% above / below the MA, close held within 1%, then a close
   2%+ above the MA within 5 days; touches within 3 bars = one event)
2. Closes held on that MA (no more than 1% below) on >= 80% of the days
3. That MA is rising (higher than 10 days ago)
4. Today's close is 1% below to 3% above that same MA

Template: daily_rsi_high_breakout_v1.py. Defaults (not requested): Entry = close,
Stop = 2% below the support MA, Quantity/Total_Amount for a $40 max loss.

Verified: positive synthetic case fires with independent re-check of held %,
distance and slope; negative controls fail at step 1 (runaway trend, choppy),
step 2 (held 76%), step 3 (falling MA) and step 4 (7% above MA); full script
run with mocked data/SMTP for 6 matches and 0 matches (CSV sizing <= $40,
chart, TradingView list, decoded email); py_compile; no template leftovers.

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01WG6bhVDgntdGBGqtFYno57""")
for i in range(5):
    if subprocess.call(["git", "pull", "--rebase", "origin", "master"]) == 0 and \
       subprocess.call(["git", "push", "origin", "HEAD:master"]) == 0:
        print("✅ pushed"); break
else:
    sys.exit("❌ push failed")
