# One-off build run by ClaudeRunner: makes best10_digest.py replace the
# TradingView watchlist after sending the Best 10 email (only when the
# TV_SESSIONID / TV_SESSIONID_SIGN secrets reach it), checks the files are
# byte-identical to the locally tested copies (SHA-256) and commits.
import hashlib, subprocess, sys

F = "claude_run/tools/best10_digest.py"
EXPECTED = {
    F: "3772b00357796540ce39d4306e5bf708aa2dd6e935f446ad3c8005e444c8db44",
    "claude_run/tools/tv_watchlist_sync.py": "92627c6225c00331b66accad79e418d02fe3b2115dbb916b0023ca3533a063af",
}
EDITS = [
    ('import pandas as pd\n\nROOT = "claude_run/results"\n',
     'import pandas as pd\n\nsys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))\nROOT = "claude_run/results"\n'),
    ('        open(csv_path.replace(".csv", ".html"), "w").write(body_html)\n    return 0\n',
     '        open(csv_path.replace(".csv", ".html"), "w").write(body_html)\n'
     '    # TradingView watchlist (only when the session secrets are configured)\n'
     '    if os.environ.get("TV_SESSIONID") and "--no-tv" not in argv:\n'
     '        try:\n'
     '            import tv_watchlist_sync\n'
     '            tv_watchlist_sync.main([str(run)])\n'
     '        except Exception as err:\n'
     '            print(f"[TradingView] not synced: {err}")\n'
     '    return 0\n'),
]
s = open(F).read()
if "tv_watchlist_sync" not in s:
    for old, new in EDITS:
        if s.count(old) != 1:
            sys.exit(f"❌ expected 1 match in {F}: {old[:60]!r}")
        s = s.replace(old, new)
    open(F, "w").write(s)
bad = []
for f, want in EXPECTED.items():
    got = hashlib.sha256(open(f, "rb").read()).hexdigest()
    print(("OK  " if got == want else "BAD ") + f)
    if got != want:
        bad.append(f)
if bad:
    sys.exit("❌ files differ from the tested copies — nothing committed")

def git(*a):
    print("+ git", " ".join(a)); return subprocess.call(["git", *a])
git("config", "user.name", "Scanner Bot")
git("config", "user.email", "vikram-scanner-bot@users.noreply.github.com")
git("add", F)
if git("diff", "--cached", "--quiet") != 0:
    git("commit", "-m", """Best 10 digest also replaces the TradingView watchlist

After the Best 10 email, best10_digest.py runs tv_watchlist_sync.py when the
TV_SESSIONID secret is available: the watchlist "Best 10 - all scanners" is
replaced with the day's picks (sections: found by 2+ scanners, top 20, then
each scanner). A failed sync emails a short alert and affects nothing else.

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01WG6bhVDgntdGBGqtFYno57""")
    for i in range(5):
        if git("pull", "--rebase", "origin", "master") == 0 and git("push", "origin", "HEAD:master") == 0:
            print("✅ pushed"); break
    else:
        sys.exit("❌ push failed")
# sanity: the digest still builds today's email without sending or syncing
sys.exit(subprocess.call([sys.executable, F, "--no-email", "--no-tv"]))
