# One-off build run by ClaudeRunner: makes the daily RSI digest also send the
# consolidated Best 10 email (claude_run/tools/best10_digest.py), checks the
# edited files are byte-identical to the locally tested copies (SHA-256),
# commits, then sends today's Best 10 email for the newest results run.
import hashlib, subprocess, sys

F = "claude_run/tools/rsi_amount_filter.py"
EXPECTED = {
    F: "9864ccceea6affb43d46e4a0ccf8779be5c25053546a71d4a5098e6d05625622",
    "claude_run/tools/save_digest.sh": "bcfbd4a2cad4e2ceb9f446e37037ec3973c9b6bb6d4a0ca32eec1ae2b38abda2",
    "claude_run/tools/best10_digest.py": "240932f0ed71fccdab21bb4006b006375d7a62af891fccd2a2243918c267d266",
}
s = open(F).read()
EDITS = [
    ("# Run daily by .github/workflows/rsi_digest.yml after the 06:00 IST ClaudeRunner run.\n",
     "# Run daily by .github/workflows/rsi_digest.yml after the 06:00 IST ClaudeRunner run.\n"
     "# With --email it then runs best10_digest.py: one email with every scanner's Best 10.\n"),
    ("    if email:\n        send_email(*build_email(hit, by_tkr, run, files, asof), out_csv)\n",
     "    if email:\n        send_email(*build_email(hit, by_tkr, run, files, asof), out_csv)\n"
     "        # second email: every scanner's Best 10 in one place (never stops the digest)\n"
     "        try:\n"
     "            import subprocess\n"
     "            subprocess.call([sys.executable, os.path.join(os.path.dirname(os.path.abspath(__file__)), \"best10_digest.py\"), str(run)])\n"
     "        except Exception as err:\n"
     "            print(f\"[Best 10 email] not sent: {err}\")\n"),
]
if "best10_digest.py" not in s:
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
    git("commit", "-m", """Daily digest also emails every scanner's Best 10 in one email

After the 06:00 IST run, the RSI Digest workflow now runs
claude_run/tools/best10_digest.py as well: one email with the stocks found by
two or more scanners, the top 20 Best scores and each scanner's Best 10, with
the combined CSV attached (saved to claude_run/results/digest/<date>/).

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01WG6bhVDgntdGBGqtFYno57""")
    for i in range(5):
        if git("pull", "--rebase", "origin", "master") == 0 and git("push", "origin", "HEAD:master") == 0:
            print("✅ pushed"); break
    else:
        sys.exit("❌ push failed")

# send today's consolidated email now (newest run with Best 10 files)
sys.exit(subprocess.call([sys.executable, "claude_run/tools/best10_digest.py"]))
