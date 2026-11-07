# One-off build run by ClaudeRunner: assembles daily_pullback_jma_reclaim_v1.py
# from the template daily_doji_hammer_jma_reclaim_v1.py (boilerplate, data
# download and ticker list kept as is) plus the new parts in
# claude_run/tools/parts_pbr/, checks it is byte-identical to the locally
# tested file (SHA-256), adds it to the 06:00 schedule, removes the parts and
# commits.
import hashlib, os, shutil, subprocess, sys

OUT = "daily_pullback_jma_reclaim_v1.py"
WANT = "9bb09a427f3c5142d6f96e442574685cb82f94f05f84eff4854aa3048a575b30"
P = "claude_run/tools/parts_pbr/"
t = open("daily_doji_hammer_jma_reclaim_v1.py").read()
a, b = t.index("import subprocess, sys, os"), t.index("# ── CONFIG")
c, d = t.index("# ── Download: daily bars"), t.index("# ── Main scan")
src = (open(P + "head.txt").read() + t[a:b] + open(P + "core.txt").read() + t[c:d] + open(P + "tail.txt").read())
open(OUT, "w").write(src)
got = hashlib.sha256(src.encode()).hexdigest()
print("sha256", got)
if got != WANT:
    sys.exit("❌ assembled file differs from the tested one — nothing committed")
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
git("commit", "-m", """Add daily pullback-to-the-50 JMA reclaim scanner (RVTY pattern); schedule at 06:00 IST

Full NASDAQ + NYSE/AMEX universe, price >= $5, 20d avg volume >= 200k.
Signal (latest 2 sessions):
  1. Uptrend: SMA50 above SMA150 and SMA200, SMA50 and SMA200 rising (20d),
     close above SMA150.
  2. Swing high 3-15 sessions back, then a 2-10 day pullback with JMA(13,40,2)
     falling on >= 2 days and not rising the day before.
  3. Pullback low within 0.5% of max(EMA21, SMA50) or below.
  4. Pullback >= 1.5 ATR14 deep; no close > 3% under SMA50, no low under SMA150.
  5. Green reclaim candle, body >= 0.6 ATR14, closes back above JMA (prior day
     at or below it) and above EMA21.
Entry = signal close, Stop = pullback low, T1 = swing high, T2 = entry + 2R
(defaults, not requested). Template: daily_doji_hammer_jma_reclaim_v1.py.
Includes the Best 10 shortlist in its email.

Verified: rule identical to claude_run/tools/wl_pullback_reclaim_check.py on
12,080 synthetic days (0 mismatches), which fires on RVTY 14 Sep 2026 only in
the past year; positive case re-checked condition by condition; negatives fail
at steps 1, 2 and 5 per the funnel; full offline run with mocked SMTP for
matches and for no matches (decoded subject, body, funnel, CSV + Best 10 CSV
attached, chart); no template leftovers.

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01WG6bhVDgntdGBGqtFYno57""")
for i in range(5):
    if subprocess.call(["git", "pull", "--rebase", "origin", "master"]) == 0 and \
       subprocess.call(["git", "push", "origin", "HEAD:master"]) == 0:
        print("✅ pushed"); break
else:
    sys.exit("❌ push failed")
