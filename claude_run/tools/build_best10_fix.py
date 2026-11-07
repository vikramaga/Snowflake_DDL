# One-off build run by ClaudeRunner: best10_digest.py skips the combined
# best10_all_scanners_*.csv when it picks a run and loads scanner lists
# (an earlier run's combined CSV was being read as a scanner list).
# Checks the result against the locally tested copy (SHA-256) and commits.
import hashlib, subprocess, sys

F = "claude_run/tools/best10_digest.py"
WANT = "76da53be1675b8a950dc001ce4ef7113fb7ef2199992de593ab09b4333ab2eed"
EDITS = [
    ('def pick_run(argv):\n'
     '    want = next((a for a in argv if a.isdigit()), "") or os.environ.get("RESULTS_RUN", "").strip()\n'
     '    if want and glob.glob(f"{ROOT}/{want}/*/best10_*.csv"):\n'
     '        return want\n'
     '    runs = [d for d in os.listdir(ROOT) if d.isdigit() and glob.glob(f"{ROOT}/{d}/*/best10_*.csv")] if os.path.isdir(ROOT) else []\n',
     'def scanner_files(pattern):\n'
     '    """best10_<scanner>_*.csv files, leaving out the combined best10_all_scanners_*.csv."""\n'
     '    return sorted(f for f in glob.glob(pattern) if not os.path.basename(f).startswith("best10_all_scanners"))\n'
     '\n\n'
     'def pick_run(argv):\n'
     '    want = next((a for a in argv if a.isdigit()), "") or os.environ.get("RESULTS_RUN", "").strip()\n'
     '    if want and scanner_files(f"{ROOT}/{want}/*/best10_*.csv"):\n'
     '        return want\n'
     '    runs = [d for d in os.listdir(ROOT) if d.isdigit() and scanner_files(f"{ROOT}/{d}/*/best10_*.csv")] if os.path.isdir(ROOT) else []\n'),
    ('        fs = sorted(glob.glob(f"{d}best10_*.csv"))\n',
     '        fs = scanner_files(f"{d}best10_*.csv")\n'),
    ('        if not len(x):\n            continue\n        full',
     '        if not len(x) or "Scanner" in x.columns:\n            continue\n        full'),
]
s = open(F).read()
if "def scanner_files" not in s:
    for old, new in EDITS:
        if s.count(old) != 1:
            sys.exit(f"❌ expected 1 match: {old[:60]!r}")
        s = s.replace(old, new)
    open(F, "w").write(s)
got = hashlib.sha256(open(F, "rb").read()).hexdigest()
print(("OK  " if got == WANT else "BAD ") + F)
if got != WANT:
    sys.exit("❌ differs from the tested copy — nothing committed")

def git(*a):
    print("+ git", " ".join(a)); return subprocess.call(["git", *a])
git("config", "user.name", "Scanner Bot")
git("config", "user.email", "vikram-scanner-bot@users.noreply.github.com")
git("add", F)
if git("diff", "--cached", "--quiet") != 0:
    git("commit", "-m", """Best 10 digest: ignore the combined best10_all_scanners CSV

The combined CSV saved by an earlier run matched best10_*.csv, so picking the
newest run chose that run and read the file as a scanner list. Scanner files
are now listed without it (and any file that already has a Scanner column).

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01WG6bhVDgntdGBGqtFYno57""")
    for i in range(5):
        if git("pull", "--rebase", "origin", "master") == 0 and git("push", "origin", "HEAD:master") == 0:
            print("✅ pushed"); break
    else:
        sys.exit("❌ push failed")
sys.exit(subprocess.call([sys.executable, F, "--no-email", "--no-tv"]))
