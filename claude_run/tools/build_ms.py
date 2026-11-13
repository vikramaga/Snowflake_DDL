# One-off build run by ClaudeRunner: assembles daily_sweep_mss_retest_v1.py
# from the template daily_pullback_jma_reclaim_v1.py (boilerplate, data download,
# ticker list) plus the new parts in claude_run/tools/parts_ms/, checks it is
# byte-identical to the locally tested file (SHA-256), adds it to the 06:00
# schedule, removes the parts, commits, then runs it once.
import hashlib, subprocess, sys

OUT = "daily_sweep_mss_retest_v1.py"
WANT = "3a4b5419aee202f3b34330dc36fdd817788fff1ab77339308ef983b93ad79422"
P = "claude_run/tools/parts_ms/"
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
git("commit", "-m", """Add daily liquidity sweep + MSS + retest/FVG entry scanner; schedule at 06:00 IST

  1. Liquidity sweep: a low below the last swing low (3-bar pivots) in the last 30
     sessions, closed back above it within 3 bars
  2. Bullish MSS: a close above the last swing high before the sweep within 15 sessions
  3. Sweep low held since
  4. Today: first pullback into the retest zone (broken swing high +/- 0.25 ATR) or the
     bullish FVG of the up-move, after the move topped out 0.5 ATR above the zone
  5. Today's close at or above the bottom of that zone
Entry = close, Stop = sweep low - 0.25 ATR, Target = nearest resistance (MSS move high
or a swing high from before the sweep), flagged under 1.5R. Defaults; all in CFG.
Template: daily_pullback_jma_reclaim_v1.py (boilerplate, download, tickers). Includes Best 10.

Verified: synthetic setup fires; negatives fail at their own step (no sweep -> 1,
no MSS -> 2, sweep low broken -> 3, zone already touched earlier -> 4, close under
the zone -> 5); mocked full runs with and without matches (subject, funnel, CSV,
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
