# One-off build run by ClaudeRunner. On 8 Nov the scheduled run's results were
# not committed, so the digest fell back to a one-scanner run: the Best 10
# email had 9 picks and the TradingView watchlist was replaced with 9 symbols.
# This build:
#   - best10_digest.py: with RESULTS_RUN set (daily digest) uses exactly that
#     run or nothing; without it, the newest run covering >= 10 scanners.
#   - tv_watchlist_sync.py: refuses to replace the watchlist from a run with
#     fewer than 10 scanners (alert email instead).
# checks both files against the locally tested copies (SHA-256), commits, and
# restores the watchlist from the full 6 Nov run (42364661033).
import hashlib, os, subprocess, sys

T = "claude_run/tools/"
WANT = {T + "best10_digest.py": "60694a5a78a9ce266076a7fbbe95c0afe9fc3e7eec872ed18087ee1c6dc861ab",
        T + "tv_watchlist_sync.py": "ff4d32d6b1d5cbbf639ac4ae2afd8c4dffd6174039abf1becf4913a479bd7bc1"}
EDITS = {
    T + "best10_digest.py": [(
        'def pick_run(argv):\n'
        '    want = next((a for a in argv if a.isdigit()), "") or os.environ.get("RESULTS_RUN", "").strip()\n'
        '    if want and scanner_files(f"{ROOT}/{want}/*/best10_*.csv"):\n'
        '        return want\n'
        '    runs = [d for d in os.listdir(ROOT) if d.isdigit() and scanner_files(f"{ROOT}/{d}/*/best10_*.csv")] if os.path.isdir(ROOT) else []\n'
        '    return max(runs, key=int) if runs else None\n',
        'MIN_SCANNERS = int(os.environ.get("BEST10_MIN_SCANNERS", "10"))\n'
        '\n\n'
        'def n_scanners(run):\n'
        '    return len({os.path.dirname(f) for f in scanner_files(f"{ROOT}/{run}/*/best10_*.csv")})\n'
        '\n\n'
        'def pick_run(argv):\n'
        '    """The daily digest passes its run in RESULTS_RUN: use exactly that run or\n'
        '    nothing (never an older or partial run). Otherwise an explicit run id from\n'
        '    argv, else the newest run that covers at least MIN_SCANNERS scanners."""\n'
        '    sched = os.environ.get("RESULTS_RUN", "").strip()\n'
        '    if sched:\n'
        '        if scanner_files(f"{ROOT}/{sched}/*/best10_*.csv"):\n'
        '            return sched\n'
        '        print(f"[Best 10 email] results of run {sched} are not in the repo - nothing sent, watchlist unchanged")\n'
        '        return None\n'
        '    want = next((a for a in argv if a.isdigit()), "")\n'
        '    if want:\n'
        '        return want if scanner_files(f"{ROOT}/{want}/*/best10_*.csv") else None\n'
        '    runs = [d for d in os.listdir(ROOT) if d.isdigit() and n_scanners(d) >= MIN_SCANNERS] if os.path.isdir(ROOT) else []\n'
        '    return max(runs, key=int) if runs else None\n')],
    T + "tv_watchlist_sync.py": [(
        '    if a.empty:\n'
        '        raise SyncError(f"run {run} has no Best 10 rows")\n',
        '    if a.empty:\n'
        '        raise SyncError(f"run {run} has no Best 10 rows")\n'
        '    need = int(os.environ.get("TV_MIN_SCANNERS", "10"))\n'
        '    if a.Scanner.nunique() < need:\n'
        '        raise SyncError(f"run {run} has only {a.Scanner.nunique()} scanner(s) (need {need}) - "\n'
        '                        "watchlist left as it was")\n')],
}
for f, edits in EDITS.items():
    s = open(f).read()
    for old, new in edits:
        if new in s:
            continue
        if s.count(old) != 1:
            sys.exit(f"❌ {f}: expected 1 match: {old[:60]!r}")
        s = s.replace(old, new)
    open(f, "w").write(s)
bad = [f for f, w in WANT.items() if hashlib.sha256(open(f, "rb").read()).hexdigest() != w]
for f in WANT:
    print(("BAD " if f in bad else "OK  ") + f)
if bad:
    sys.exit("❌ files differ from the tested copies — nothing committed")

def git(*a):
    print("+ git", " ".join(a)); return subprocess.call(["git", *a])
git("config", "user.name", "Scanner Bot")
git("config", "user.email", "vikram-scanner-bot@users.noreply.github.com")
git("add", *WANT)
if git("diff", "--cached", "--quiet") != 0:
    git("commit", "-m", """Best 10 email / TradingView sync: never use an older or partial run

On 8 Nov the scheduled run's results were not committed; the digest fell back
to a one-scanner run, so the Best 10 email had 9 picks and the TradingView
watchlist was replaced with 9 symbols.
- best10_digest.py: with RESULTS_RUN (daily digest) uses exactly that run or
  sends nothing; otherwise the newest run covering >= 10 scanners.
- tv_watchlist_sync.py: will not replace the watchlist from a run with fewer
  than 10 scanners; emails the usual "not updated" alert instead.

Verified locally: missing scheduled run -> nothing; full run 42364661033
(28 scanners) chosen when no run is given; one-scanner run refused by the sync.

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01WG6bhVDgntdGBGqtFYno57""")
    for i in range(5):
        if git("pull", "--rebase", "origin", "master") == 0 and git("push", "origin", "HEAD:master") == 0:
            print("✅ pushed"); break
    else:
        sys.exit("❌ push failed")
# restore the watchlist from the last full run (Friday 6 Nov close)
sys.path.insert(0, T)
import tv_watchlist_sync
sys.exit(tv_watchlist_sync.main(["42364661033"]))
