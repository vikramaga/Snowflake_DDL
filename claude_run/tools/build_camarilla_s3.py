# One-off build run by ClaudeRunner: assembles daily_camarilla_s3_reclaim_v1.py
# from the template daily_pullback_jma_reclaim_v1.py (boilerplate, data
# download, ticker list) plus the new parts in claude_run/tools/parts_cam/,
# checks it is byte-identical to the locally tested file (SHA-256), adds it to
# the 06:00 schedule, removes the parts, commits, then runs it once.
import hashlib, subprocess, sys

OUT = "daily_camarilla_s3_reclaim_v1.py"
WANT = "235edd6ce29af43e7f31c2a2d88d9df8775cedf7fd878fc5af1bcbab144fde00"
P = "claude_run/tools/parts_cam/"
t = open("daily_pullback_jma_reclaim_v1.py").read()
a, b = t.index("import subprocess, sys, os"), t.index("# ── CONFIG")
c, d = t.index("# ── Download: daily bars"), t.index("# ── Main scan")
src = open(P + "head.txt").read() + t[a:b] + open(P + "core.txt").read() + t[c:d] + open(P + "tail.txt").read()
got = hashlib.sha256(src.encode()).hexdigest()
print("sha256", got)
if got != WANT:
    sys.exit("❌ assembled file differs from the tested one — nothing committed")
open(OUT, "w").write(src)
sched = open("claude_run/schedule.txt").read()
if OUT not in sched:
    open("claude_run/schedule.txt", "w").write(sched.rstrip("\n") + "\n" + OUT + "\n")
    print("✅ added to claude_run/schedule.txt")

def git(*a):
    print("+ git", " ".join(a)); subprocess.check_call(["git", *a])
git("config", "user.name", "Scanner Bot")
git("config", "user.email", "vikram-scanner-bot@users.noreply.github.com")
git("rm", "-r", "-q", P)
git("add", OUT, "claude_run/schedule.txt")
git("commit", "-m", """Add monthly Camarilla S3 reclaim swing scanner (best Camarilla + SMA50/150 rule); schedule at 06:00 IST

Rule (latest session): close back above the monthly Camarilla S3 (previous
close at or below it), SMA50 > SMA150, close within 3% of SMA50, volume >=
1.5x the 20-day average; price >= $5, 20-day avg volume >= 100k.
Orders as tested: buy next open, stop = buy - 2 ATR14, target = buy + 6 ATR14
(3R), sell after 40 sessions.

Found by claude_run/research/cam_research.py (run 42464283358): 12 monthly /
weekly H3-H4-S3 triggers x 16 filters x 51 exits; one trade per stock at a
time: 2023-25 292 trades +0.35R, 2025-26 233 trades +0.31R, fresh 1,436
stocks 487 trades 40.0% wins PF 1.32 +0.27R (avg win +13.3%, loss -6.7%);
same filters without the S3 reclaim +0.13R.

Verified: signals identical to the research engine on 70,200 synthetic days
(0 mismatches, 98 signals); negatives fail at the right step; mocked full
runs with and without matches (subject, funnel, CSV + Best 10 CSV, chart).

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01WG6bhVDgntdGBGqtFYno57""")
for i in range(5):
    ok = subprocess.call(["git", "pull", "--rebase", "origin", "master"]) == 0 and subprocess.call(["git", "push", "origin", "HEAD:master"]) == 0
    if ok:
        print("✅ pushed"); break
else:
    sys.exit("❌ push failed")
sys.exit(subprocess.call([sys.executable, OUT]))
