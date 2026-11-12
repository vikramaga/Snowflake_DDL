# One-off patch run by ClaudeRunner: loosens the bearish-stack SMA21 reclaim scanner
# from > 3x to > 2x the previous day's volume (file name kept so history and the
# watchlist/scorecard stay linked), commits, then runs it once.
import py_compile, subprocess, sys

F = "daily_bear_stack_sma21_reclaim_3xvol_v1.py"
s = open(F).read()
reps = [
    ('"vol_mult_prev_day"       : 3.0,   # today\'s volume > 3x yesterday\'s',
     '"vol_mult_prev_day"       : 2.0,   # today\'s volume > 2x yesterday\'s (was 3x until 2026-11-12)', 1),
    ("#   4. VOLUME: today's volume > 3x yesterday's volume",
     "#   4. VOLUME: today's volume > 2x yesterday's volume (was 3x until 2026-11-12)", 1),
    ('score += min(20, (sig["vol_x"] - 3) * 4)', 'score += min(20, (sig["vol_x"] - CFG["vol_mult_prev_day"]) * 4)', 1),
    ("Reclaim on 3x Volume", "Reclaim on 2x Volume", 3),
    ("Reclaim 3x Vol", "Reclaim 2x Vol", 3),
]
for old, new, n in reps:
    c = s.count(old)
    if c != n:
        sys.exit(f"expected {n} match(es), found {c}: {old[:60]!r} - nothing changed")
    s = s.replace(old, new)
open(F, "w").write(s)
py_compile.compile(F, doraise=True)
print("patched")

def git(*a):
    print("+ git", " ".join(a)); subprocess.check_call(["git", *a])
git("config", "user.name", "Scanner Bot")
git("config", "user.email", "vikram-scanner-bot@users.noreply.github.com")
git("add", F)
git("commit", "-m", """Bearish-stack SMA21 reclaim: volume rule loosened from > 3x to > 2x the previous day

Title/subject now say 2x; the volume part of the score starts from the CFG value.
File name kept (..._3xvol_v1.py) so its results history stays linked.

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01WG6bhVDgntdGBGqtFYno57""")
for i in range(5):
    ok = subprocess.call(["git", "pull", "--rebase", "origin", "master"]) == 0 and subprocess.call(["git", "push", "origin", "HEAD:master"]) == 0
    if ok:
        print("pushed"); break
else:
    sys.exit("push failed")
sys.exit(subprocess.call([sys.executable, F]))
