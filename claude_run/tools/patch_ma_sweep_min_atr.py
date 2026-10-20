# One-off (run by ClaudeRunner): adds a minimum-volatility filter to the MA sweep
# + 4x candle scanner and its generator. Stocks whose ATR14 is below 1% of price
# (e.g. pinned by a pending takeover, like TACO on 2026-10-20) are skipped: on
# those a "4x body" is meaningless and the stop sits cents away, which made the
# $40 sizing ask for thousands of shares. Patches both files identically,
# regenerates the scanner from the generator to prove they match, then commits.
import subprocess, sys, filecmp

FILES = ["daily_ma_sweep_4x_v1.py", "claude_run/tools/gen_ma_sweep_4x.py"]
EDITS = [
    ('    "min_price"              : 10.0,\n',
     '    "min_price"              : 10.0,\n'
     '    "min_atr_pct"            : 1.0,   # skip stocks moving < 1% a day (e.g. pinned by a takeover)\n'),
    ('    atr = calc_atr(h, l, c, CFG["atr_period"])\n\n    hits = find_ma_sweep_signals(',
     '    atr = calc_atr(h, l, c, CFG["atr_period"])\n'
     '    if not np.isfinite(atr[-1]) or atr[-1] / price * 100 < CFG["min_atr_pct"]:\n'
     '        return None\n\n    hits = find_ma_sweep_signals('),
    ('# Quantity / Total_Amount come from position_sizing.py (max loss $40).\n',
     '# Quantity / Total_Amount come from position_sizing.py (max loss $40).\n'
     '# Stocks whose ATR14 is under 1% of price are skipped (min_atr_pct).\n'),
]
for f in FILES:
    s = open(f).read()
    for a, b in EDITS:
        n = s.count(a)
        if n != 1:
            sys.exit(f"❌ {f}: expected 1 match, found {n}: {a[:50]!r}")
        s = s.replace(a, b)
    open(f, "w").write(s)
    print(f"✅ patched {f}")

subprocess.check_call([sys.executable, "claude_run/tools/gen_ma_sweep_4x.py", "_regen.py"])
if not filecmp.cmp("_regen.py", FILES[0], shallow=False):
    sys.exit("❌ regenerated scanner differs from the patched one — nothing committed")
print("✅ generator output == patched scanner")

def git(*a):
    print("+ git", " ".join(a)); subprocess.check_call(["git", *a])
git("config", "user.name", "Scanner Bot")
git("config", "user.email", "vikram-scanner-bot@users.noreply.github.com")
git("add", *FILES)
git("commit", "-m", """MA sweep + 4x scanner: skip stocks whose ATR14 is under 1% of price

First live run (2026-10-20) matched TACO, which barely moves (range ~0.2%,
typical of a pending takeover): its previous body was ~0 so any green candle
was ">4x", and the stop was 2 cents away, so $40 sizing asked for 2,000 shares
($20,960). New CFG min_atr_pct = 1.0 filters these out before the pattern
check. Scanner and generator patched identically; regenerated output verified
byte-identical.

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01WG6bhVDgntdGBGqtFYno57""")
for i in range(5):
    if subprocess.call(["git", "pull", "--rebase", "origin", "master"]) == 0 and \
       subprocess.call(["git", "push", "origin", "HEAD:master"]) == 0:
        print("✅ pushed"); break
else:
    sys.exit("❌ push failed")
