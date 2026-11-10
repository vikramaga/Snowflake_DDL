# One-off build run by ClaudeRunner: assembles daily_bear_stack_sma21_reclaim_3xvol_v1.py
# from the template daily_pullback_jma_reclaim_v1.py (boilerplate, calc_jma, data
# download, ticker list) plus the new parts in claude_run/tools/parts_bs/, checks it is
# byte-identical to the locally tested file (SHA-256), adds it to the 06:00
# schedule, removes the parts, commits, then runs it once.
import hashlib, subprocess, sys

OUT = "daily_bear_stack_sma21_reclaim_3xvol_v1.py"
WANT = "7eb634175b82f7c1147e733840052e15dac83b0312ce901e0a374623776d4ea0"
P = "claude_run/tools/parts_bs/"
t = open("daily_pullback_jma_reclaim_v1.py").read()
a, b = t.index("import subprocess, sys, os"), t.index("# ── CONFIG")
c, d = t.index("# ── Download: daily bars"), t.index("# ── Main scan")
e, f = t.index("def calc_jma("), t.index("def calc_atr(")
src = (open(P + "head.txt").read() + t[a:b] + open(P + "core_a.txt").read() + t[e:f]
       + open(P + "core_b.txt").read() + t[c:d] + open(P + "tail.txt").read())
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
git("commit", "-m", """Add daily bearish-stack SMA21 reclaim on 3x volume scanner; schedule at 06:00 IST

  1. Previous session: JMA < EMA8 < SMA21 < SMA50 (bearish stack)
  2. Today's low at or below EMA8 (the move starts from EMA8)
  3. Yesterday closed at/below SMA21, today closes above SMA21
  4. Today's volume > 3x yesterday's volume
Entry = close, Stop = today's low, T1 = SMA50, T2 = +2R (defaults). Latest session only.
Template: daily_pullback_jma_reclaim_v1.py (boilerplate, calc_jma, download, tickers). Includes Best 10.

Verified: synthetic setup fires and is re-checked independently; negatives fail at
their own step (rising stack -> 1, gap open above EMA8 -> 2, close under SMA21 -> 3,
2.5x volume -> 4); mocked full runs with and without matches (subject, funnel, CSV,
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
