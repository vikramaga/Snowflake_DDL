# One-off patch run by ClaudeRunner (2026-11-15): the three "MA Compression Breakout (IREN)"
# scanners found matches but crashed before emailing (NameError: MIMEMultipart / MIMEText
# were never imported). Adds the two imports, commits, then runs all three once.
import py_compile, subprocess, sys

FILES = ["crossema20.py", "ema20.py", "nasdaq_ma_compression_breakout.py"]
OLD = "from email.mime.base import MIMEBase; from email import encoders"
NEW = OLD + "; from email.mime.multipart import MIMEMultipart; from email.mime.text import MIMEText"
for F in FILES:
    s = open(F).read()
    if NEW in s:
        print(f"{F}: already patched"); continue
    c = s.count(OLD)
    if c != 1:
        sys.exit(f"{F}: expected 1 match, found {c} - nothing changed")
    open(F, "w").write(s.replace(OLD, NEW))
    py_compile.compile(F, doraise=True)
    print(f"{F}: patched")

def git(*a):
    print("+ git", " ".join(a)); subprocess.check_call(["git", *a])
git("config", "user.name", "Scanner Bot")
git("config", "user.email", "vikram-scanner-bot@users.noreply.github.com")
git("add", *FILES)
if subprocess.call(["git", "diff", "--cached", "--quiet"]) != 0:
    git("commit", "-m", """IREN MA compression scanners: import MIMEMultipart and MIMEText (email crashed)

crossema20.py, ema20.py and nasdaq_ma_compression_breakout.py raised NameError
when building the email, so their matches were never sent.

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01WG6bhVDgntdGBGqtFYno57""")
    for i in range(5):
        if subprocess.call(["git", "pull", "--rebase", "origin", "master"]) == 0 and \
           subprocess.call(["git", "push", "origin", "HEAD:master"]) == 0:
            print("pushed"); break
    else:
        sys.exit("push failed")

procs = [(F, subprocess.Popen([sys.executable, F], stdout=open(F + ".out", "w"), stderr=subprocess.STDOUT)) for F in FILES]
rc = 0
for F, p in procs:
    r = p.wait(); rc |= r
    out = open(F + ".out").read().splitlines()
    print(f"\n===== {F} exit {r} =====")
    print("\n".join([l for l in out if "SCAN COMPLETE" in l or "[Email]" in l or "Sent" in l or "Error" in l or "Traceback" in l][-12:]))
    print("\n".join(out[-6:]))
sys.exit(rc)
