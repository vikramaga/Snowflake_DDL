# One-off build run by ClaudeRunner: lets smaretest.py scan penny stocks
# (price floor $1 -> $0.10), adds the Best 10 shortlist to its output and email,
# puts it on the 06:00 IST schedule, commits, then runs it once.
import hashlib, py_compile, subprocess, sys

F = "smaretest.py"
s = open(F).read()
reps = [
    ('    "min_price"                 : 1.0,\n',
     '    "min_price"                 : 0.10,  # includes penny stocks (was 1.0)\n'),
    ('df_out.to_csv(fpath, index=False)\nprint(f"\\n  \U0001F4BE CSV',
     'try: __import__("best_picks").rank_best(df_out)\nexcept Exception as _bp_err: print(f"  Best 10 skipped: {_bp_err}")\n'
     'df_out.to_csv(fpath, index=False)\nprint(f"\\n  \U0001F4BE CSV'),
    ('            srv.login(gu, gp.replace(" ", ""))\n            srv.sendmail(gu, eto, msg.as_string())\n',
     '            srv.login(gu, gp.replace(" ", ""))\n'
     '            try: __import__("best_picks").attach_to_email(msg)\n'
     '            except Exception as _bp_err: print(f"  Best 10 not added: {_bp_err}")\n'
     '            srv.sendmail(gu, eto, msg.as_string())\n'),
]
for old, new in reps:
    n = s.count(old)
    if n != 1:
        sys.exit(f"expected 1 match, found {n}: {old[:60]!r} - nothing changed")
    s = s.replace(old, new)
open(F, "w").write(s)
py_compile.compile(F, doraise=True)
print("patched; sha256", hashlib.sha256(s.encode()).hexdigest())
sched = open("claude_run/schedule.txt").read()
if F not in sched.split():
    open("claude_run/schedule.txt", "w").write(sched.rstrip("\n") + "\n" + F + "\n")
    print("added to claude_run/schedule.txt")
if "--no-git" in sys.argv:
    sys.exit(0)

def git(*a):
    print("+ git", " ".join(a)); subprocess.check_call(["git", *a])
git("config", "user.name", "Scanner Bot")
git("config", "user.email", "vikram-scanner-bot@users.noreply.github.com")
git("add", F, "claude_run/schedule.txt")
git("commit", "-m", """Schedule smaretest.py (SMA50/SMA150 retest + EMA20 cross) daily at 06:00 IST, penny stocks included

- min_price 1.0 -> 0.10 so sub-$1 stocks are scanned (volume floor unchanged at 80k)
- Best 10 shortlist added to its results and email, like the other scanners

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01WG6bhVDgntdGBGqtFYno57""")
for i in range(5):
    ok = subprocess.call(["git", "pull", "--rebase", "origin", "master"]) == 0 and subprocess.call(["git", "push", "origin", "HEAD:master"]) == 0
    if ok:
        print("pushed"); break
else:
    sys.exit("push failed")
sys.exit(subprocess.call([sys.executable, F]))
