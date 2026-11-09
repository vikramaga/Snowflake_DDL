# One-off build run by ClaudeRunner: assembles daily_major_support_retest_v1.py
# from the template daily_pullback_jma_reclaim_v1.py (boilerplate, data download,
# ticker list) plus the new parts in claude_run/tools/parts_sup/, checks it is
# byte-identical to the locally tested file (SHA-256), adds it to the 06:00
# schedule, removes the parts, commits, then runs it once.
import hashlib, subprocess, sys

OUT = "daily_major_support_retest_v1.py"
WANT = "b22fb67908112e1cc39caa92a72a8e49d57a08fdc899492d6485baee607b4b41"
P = "claude_run/tools/parts_sup/"
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
git("commit", "-m", """Add daily major support retest scanner; schedule at 06:00 IST

Finds stocks back at a level where they reversed hard 1.5-3 months ago:
  1. Major swing low 25-65 sessions ago (lowest of 5 bars each side and of the 40 bars before)
  2. Strong reversal off it: +10% and 4 ATR within 15 sessions
  3. Support held since: no close below support - 0.5 ATR
  4. Today's low back at support (-0.5 .. +1.0 ATR), close not below support - 0.25 ATR
  5. Rejection candle: close in the upper half of the range, green or lower wick >= body
Entry = close, Stop = min(low, support) - 0.25 ATR, Target = the reversal high (defaults).
Template: daily_pullback_jma_reclaim_v1.py (boilerplate, download, tickers). Includes Best 10.

Verified: synthetic setup fires and is re-checked independently; negatives fail at
their own step (red close -> 5, today far above -> 4, support broken -> 3, weak
rally -> 2); mocked full runs with and without matches (subject, funnel, CSV,
Best 10 CSV, chart).

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01WG6bhVDgntdGBGqtFYno57""")
for i in range(5):
    ok = subprocess.call(["git", "pull", "--rebase", "origin", "master"]) == 0 and subprocess.call(["git", "push", "origin", "HEAD:master"]) == 0
    if ok:
        print("pushed"); break
else:
    sys.exit("push failed")
sys.exit(subprocess.call([sys.executable, OUT]))
