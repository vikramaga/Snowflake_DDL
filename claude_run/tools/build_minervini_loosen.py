# One-off build run by ClaudeRunner (2026-11-08): loosens the Minervini Stage 2
# dry-up scanner for more matches (pullback low within 5% above SMA50, red-candle
# volume <= 90% of average and not rising by > 15%, green candle >= 1.5x the
# previous red, last 3 sessions), takes the Camarilla S3 scanner off the 06:00
# schedule (file kept), checks minervini.py against the tested copy (SHA-256),
# commits, then runs minervini.py.
import hashlib, subprocess, sys

EDITS = [
 ("#      then a 2-15 session pullback whose low comes within 3% of the SMA50\n#      (above it, or up to 3% under it), without any close more than 3% under\n#      the SMA50.\n",
  "#      then a 2-15 session pullback whose low comes within 5% above the SMA50\n#      or up to 3% under it, without any close more than 3% under the SMA50.\n"),
 ("#      volume is at most 75% of the 50-day average volume (measured at the\n#      swing high), and selling volume is falling — the red candles in the\n#      second half of the pullback average no more than those in the first.\n",
  "#      volume is at most 90% of the 50-day average volume (measured at the\n#      swing high), and selling volume is not rising — the red candles in the\n#      second half of the pullback average at most 15% more than in the first.\n"),
 ("#      open) closing above the SMA50, with volume at least 2x the volume of\n#      the previous red candle.\n",
  "#      open) closing above the SMA50, with volume at least 1.5x the volume of\n#      the previous red candle.\n#   (Loosened on 2026-11-08 from 3% / 75% / falling / 2x / 2 sessions for\n#   more matches.)\n"),
 ("# Signal day = one of the last 2 sessions.", "# Signal day = one of the last 3 sessions."),
 ('    "sma50_near_pct"          : 3.0,   # pullback low within 3% above SMA50 ...\n',
  '    "sma50_near_pct"          : 5.0,   # pullback low within 5% above SMA50 ...\n'),
 ('    "red_vol_max_pct"         : 75.0,  # red-candle avg vol <= 75% of 50-day avg\n',
  '    "red_vol_max_pct"         : 90.0,  # red-candle avg vol <= 90% of 50-day avg\n    "red_vol_rise_tol"        : 1.15,  # late red candles may average up to 15% more than early ones\n'),
 ('    "green_vol_mult"          : 2.0,   # volume >= 2x the previous red candle\n',
  '    "green_vol_mult"          : 1.5,   # volume >= 1.5x the previous red candle\n'),
 ('    "signal_lookback_days"    : 2,\n', '    "signal_lookback_days"    : 3,\n'),
 ('    falling = (not r1 or not r2) or (np.mean(r2) <= np.mean(r1))\n',
  '    falling = (not r1 or not r2) or (np.mean(r2) <= np.mean(r1) * cfg["red_vol_rise_tol"])\n'),
 ("  B) Red candles average <= {CFG['red_vol_max_pct']:.0f}% of the {CFG['vol_avg_bars']}-day volume and are getting lighter\n",
  "  B) Red candles average <= {CFG['red_vol_max_pct']:.0f}% of the {CFG['vol_avg_bars']}-day volume and are not getting heavier\n"),
 ("  ⚙️  If 0 results, check the FUNNEL: relax red_vol_max_pct 75 → 85,\n     green_vol_mult 2.0 → 1.5, or signal_lookback_days 2 → 5\n",
  "  ⚙️  Loosened 2026-11-08 (was 3% / 75% / 2x / 2 sessions). To widen further:\n     sma50_near_pct 5 → 8, red_vol_max_pct 90 → 100, or signal_lookback_days 3 → 5\n"),
]
REPLACE_ALL = [("{CFG['green_vol_mult']:.0f}x", "{CFG['green_vol_mult']:g}x"),
               (" green 2x volume -> ", " green volume -> ")]

def apply(s):
    for old, new in EDITS:
        if s.count(old) != 1:
            raise SystemExit(f"expected 1 match: {old[:70]!r}")
        s = s.replace(old, new)
    for old, new in REPLACE_ALL:
        if old not in s:
            raise SystemExit(f"missing: {old!r}")
        s = s.replace(old, new)
    return s

F = "minervini.py"
WANT = "26d0b4b1785f7aa33f8017c09ab1888eb419fde766f8452609b346f588da609a"
s = open(F).read()
if '"red_vol_rise_tol"' not in s:
    s = apply(s)
got = hashlib.sha256(s.encode()).hexdigest()
print("sha256", got)
if got != WANT:
    sys.exit("differs from the tested copy - nothing committed")
open(F, "w").write(s)
S = "claude_run/schedule.txt"
lines = open(S).read().split("\n")
keep = [l for l in lines if l.strip() != "daily_camarilla_s3_reclaim_v1.py"]
open(S, "w").write("\n".join(keep))
print("schedule:", len(lines) - len(keep), "line(s) removed")

def git(*a):
    print("+ git", " ".join(a)); return subprocess.call(["git", *a])
git("config", "user.name", "Scanner Bot")
git("config", "user.email", "vikram-scanner-bot@users.noreply.github.com")
git("add", F, S)
if git("diff", "--cached", "--quiet") != 0:
    git("commit", "-m", """Minervini dry-up scanner: looser criteria; Camarilla S3 scanner off the schedule

minervini.py (Trend Template unchanged, RS >= 70 kept):
- pullback low within 5% above SMA50 (was 3%), still up to 3% under it
- red-candle volume <= 90% of the 50-day average (was 75%), and the late
  red candles may average up to 15% more than the early ones (was: lower)
- green candle volume >= 1.5x the previous red candle (was 2x)
- signal in the last 3 sessions (was 2)
The first live run had 444 Trend Template stocks -> 160 at SMA50 -> 15 dry-up
-> 1 match; these steps were the bottleneck.
daily_camarilla_s3_reclaim_v1.py removed from claude_run/schedule.txt at the
user's request (file kept).

Verified: synthetic positive passes; 1.5x green passes, 1.2x fails; heavy and
rising red volume fail; downtrend fails; mocked full run sends the email.

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01WG6bhVDgntdGBGqtFYno57""")
    for i in range(5):
        ok = git("pull", "--rebase", "origin", "master") == 0 and git("push", "origin", "HEAD:master") == 0
        if ok:
            print("pushed"); break
    else:
        sys.exit("push failed")
sys.exit(subprocess.call([sys.executable, F]))
