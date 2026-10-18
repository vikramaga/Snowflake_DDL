# One-off build run by ClaudeRunner: generates daily_rsi_avg_bounce_v1.py
# with gen_rsi_avg_bounce.py, checks it is byte-identical to the locally tested
# file (SHA-256), adds it to the 06:00 schedule and commits.
import hashlib, subprocess, sys

EXPECTED = "2a9b3aa4a7c0432c8df13a6d4254d6e769d6ba5789b903b4f000c186ae82c4d3"
OUT = "daily_rsi_avg_bounce_v1.py"
subprocess.check_call([sys.executable, "claude_run/tools/gen_rsi_avg_bounce.py", OUT])
h = hashlib.sha256(open(OUT, "rb").read()).hexdigest()
print("sha256", h)
if h != EXPECTED:
    sys.exit("❌ generated file differs from the tested one — nothing committed")
sched = open("claude_run/schedule.txt").read()
if OUT not in sched:
    open("claude_run/schedule.txt", "w").write(sched.rstrip("\n") + "\n" + OUT + "\n")
    print("✅ added to claude_run/schedule.txt")

def git(*a):
    print("+ git", " ".join(a)); subprocess.check_call(["git", *a])
git("config", "user.name", "Scanner Bot")
git("config", "user.email", "vikram-scanner-bot@users.noreply.github.com")
git("add", OUT, "claude_run/schedule.txt")
git("commit", "-m", """Add daily RSI bounce-off-average scanner and schedule it at 06:00 IST

Signal (latest daily candle; Wilder RSI(14), RSI average = 9-day SMA of RSI):
1. RSI crossed above its average 3-10 trading days ago
2. RSI then rose at least 3 points above its average
3. RSI pulled back to within 1.5 points of its average (never > 2 below since the cross)
4. That touch was in the last 2 days and today RSI is rising, above and moving away from its average
Defaults (not requested): Entry = close, Stop = lowest low of the pullback,
Quantity/Total_Amount for a $40 max loss.

Generated from daily_rsi_high_breakout_v1.py by claude_run/tools/gen_rsi_avg_bounce.py.

Verified: 2 positive and 7 negative RSI paths (each negative fails at the
intended step); 434 signals on random price paths independently re-verified
condition by condition; full run with mocked data/SMTP for 6 and 0 matches
(sizing <= $40, chart, decoded email); py_compile; no template leftovers.

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01WG6bhVDgntdGBGqtFYno57""")
for i in range(5):
    if subprocess.call(["git", "pull", "--rebase", "origin", "master"]) == 0 and \
       subprocess.call(["git", "push", "origin", "HEAD:master"]) == 0:
        print("✅ pushed"); break
else:
    sys.exit("❌ push failed")
