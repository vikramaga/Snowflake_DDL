# One-off build run by ClaudeRunner: rebuilds minervini.py (v2) from the
# template daily_pullback_jma_reclaim_v1.py (boilerplate, data download,
# ticker list) plus the new parts in claude_run/tools/parts_minv/, checks it
# is byte-identical to the locally tested file (SHA-256), adds it to the 06:00
# schedule, removes the parts, commits, then runs it once.
import hashlib, subprocess, sys

OUT = "minervini.py"
WANT = "87586c4c4744c981aed177496bf62c9c7b5a20aca875051a8ea7c59b00e632ac"
P = "claude_run/tools/parts_minv/"
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
git("commit", "-m", """minervini.py v2: full Trend Template + volume dry-up pullback to SMA50 + 2x-volume green candle

Stage 2 = all 8 points of Mark Minervini's Trend Template on the signal day:
close > SMA150 & SMA200, SMA150 > SMA200, SMA200 rising 1 month, SMA50 >
SMA150 & SMA200, close > SMA50, >= 30% above the 52-week low, within 25% of
the 52-week high, RS Rating >= 70 (IBD-style weighted 12-month performance,
ranked 1-99 against every stock scanned). v1 missed SMA50 > SMA150/200 and
used a 52-bar SPY comparison with a -5% allowance as the RS proxy.
Setup: swing high 3-20 sessions back; 2-15 day pullback whose low comes within
+3%/-3% of SMA50 (no close > 3% under it); red candles average <= 75% of the
50-day volume and get lighter; signal = green candle closing above SMA50 with
volume >= 2x the previous red candle (latest 2 sessions).
Entry = signal close, Stop = pullback low (flag > 8% risk), T1 = swing high,
T2 = entry + 3R (defaults). Same CSV name (minervini_stage2_vcp_*), Best 10
in the email, added to the 06:00 IST schedule. Built on the current daily
scanner template (single check_pattern_at, funnel, position sizing).

Verified: synthetic Stage 2 pullback fires and every condition re-checked
independently; negatives fail where expected (downtrend -> step 1, pullback
not reaching SMA50 -> step 2, heavy or rising red volume -> step 3, green
volume 1.5x -> step 4); full offline run with mocked SMTP for matches and for
no matches (subject, body, funnel, CSV + Best 10 CSV, chart).

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01WG6bhVDgntdGBGqtFYno57""")
for i in range(5):
    if subprocess.call(["git", "pull", "--rebase", "origin", "master"]) == 0 and \
       subprocess.call(["git", "push", "origin", "HEAD:master"]) == 0:
        print("✅ pushed"); break
else:
    sys.exit("❌ push failed")
sys.exit(subprocess.call([sys.executable, OUT]))
