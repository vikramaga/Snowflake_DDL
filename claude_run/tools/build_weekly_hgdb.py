# One-off build run by ClaudeRunner: assembles weekly_hammer_green_doji_breakout_v1.py
# from the template daily_pullback_jma_reclaim_v1.py (boilerplate, data download,
# ticker list) plus the new parts in claude_run/tools/parts_w3/, checks it is
# byte-identical to the locally tested file (SHA-256), adds it to the 06:00
# schedule, removes the parts, commits, then runs it once.
import hashlib, subprocess, sys

OUT = "weekly_hammer_green_doji_breakout_v1.py"
WANT = "40e2ba0a87edfc6b96b239d7e2cba23f4c4b70a728bb38ec738590a23ad988e4"
P = "claude_run/tools/parts_w3/"
t = open("daily_pullback_jma_reclaim_v1.py").read()
a, b = t.index("import subprocess, sys, os"), t.index("# ── CONFIG")
c, d = t.index("# ── Download: daily bars"), t.index("# ── Main scan")
src = open(P + "head.txt").read() + t[a:b] + open(P + "core.txt").read() + t[c:d] + open(P + "tail.txt").read()
got = hashlib.sha256(src.encode()).hexdigest()
print("sha256", got)
if got != WANT:
    sys.exit("assembled file differs from the tested one - nothing committed")
open(OUT, "w").write(src)
sched = open("claude_run/schedule.txt").read()
if OUT not in sched:
    open("claude_run/schedule.txt", "w").write(sched.rstrip("\n") + "\n" + OUT + "\n")
    print("added to claude_run/schedule.txt")

def git(*a):
    print("+ git", " ".join(a)); subprocess.check_call(["git", *a])
git("config", "user.name", "Scanner Bot")
git("config", "user.email", "vikram-scanner-bot@users.noreply.github.com")
git("rm", "-r", "-q", P)
git("add", OUT, "claude_run/schedule.txt")
git("commit", "-m", """Add weekly hammer/doji -> green -> doji -> daily breakout scanner; schedule at 06:00 IST

Current week = week of the latest daily bar; W-1..W-3 = the 3 completed weeks before.
  1. W-3: red candle that is a hammer or a doji
  2. W-2: green, low above W-3's low, volume above W-3's volume
  3. W-1: doji or small hammer, low above the middle of W-2's green body
  4. Current week: a daily close above W-1's high (first such close = signal)
Doji = body <= 35% of range and <= 0.5 weekly ATR10; hammer = lower wick >=
2x body, upper wick <= 25% of range; small = range <= 1 weekly ATR.
Entry = signal-day close, Stop = W-1 low, T1 = +2R, T2 = +3R (defaults).
Includes the Best 10 shortlist in its email.

Verified: synthetic pattern fires and is re-checked independently on the
weekly bars; each step fails on its own negative (W-3 green -> step 1; W-2
lower volume or red -> step 2; W-1 low under W-2 mid or a big W-1 -> step 3;
no close above W-1 high -> step 4); candle shapes checked by hand; mocked full
runs with and without matches (subject, funnel, CSV + Best 10 CSV, chart).

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01WG6bhVDgntdGBGqtFYno57""")
for i in range(5):
    ok = subprocess.call(["git", "pull", "--rebase", "origin", "master"]) == 0 and subprocess.call(["git", "push", "origin", "HEAD:master"]) == 0
    if ok:
        print("pushed"); break
else:
    sys.exit("push failed")
sys.exit(subprocess.call([sys.executable, OUT]))
