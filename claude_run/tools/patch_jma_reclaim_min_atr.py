# One-off (run by ClaudeRunner): adds a minimum-volatility filter to the
# JMA/EMA8/SMA21/SMA50 reclaim + RSI scanner and its generator. Stocks whose
# ATR14 is below 1% of price (e.g. pinned by a pending takeover, like KOYN/BWIN
# on 2026-10-22) are skipped: on those the stop sits cents away, which made the
# $40 sizing ask for thousands of shares. Patches both files identically,
# regenerates the scanner from the generator to prove they match, then commits.
import subprocess, sys, filecmp

FILES = ["daily_jma_reclaim_rsi_v1.py", "claude_run/tools/gen_jma_reclaim_rsi.py"]
EDITS = [
    ('    "min_price"              : 10.0,\n',
     '    "min_price"              : 10.0,\n'
     '    "min_atr_pct"            : 1.0,   # skip stocks moving < 1% a day (e.g. pinned by a takeover)\n'),
    ('    rsi   = calc_rsi(close, CFG["rsi_period"])\n',
     '    rsi   = calc_rsi(close, CFG["rsi_period"])\n'
     '    h  = df_daily["High"].values.astype(float)\n'
     '    pc = np.concatenate([[np.nan], c[:-1]])\n'
     '    tr = np.nanmax(np.vstack([h - l, np.abs(h - pc), np.abs(l - pc)]), axis=0)\n'
     '    atr14 = pd.Series(tr).rolling(14).mean().values[-1]\n'
     '    if not np.isfinite(atr14) or atr14 / price * 100 < CFG["min_atr_pct"]:\n'
     '        return None\n'),
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

subprocess.check_call([sys.executable, "claude_run/tools/gen_jma_reclaim_rsi.py", "_regen.py"])
if not filecmp.cmp("_regen.py", FILES[0], shallow=False):
    sys.exit("❌ regenerated scanner differs from the patched one — nothing committed")
print("✅ generator output == patched scanner")

def git(*a):
    print("+ git", " ".join(a)); subprocess.check_call(["git", *a])
git("config", "user.name", "Scanner Bot")
git("config", "user.email", "vikram-scanner-bot@users.noreply.github.com")
git("add", *FILES)
git("commit", "-m", """Reclaim + RSI scanner: skip stocks whose ATR14 is under 1% of price

First live run (2026-10-22) matched KOYN, BWIN and STRC, which barely move
(stops 0.2-1.7% away, typical of a pending takeover or a preferred-like
instrument), so $40 sizing asked for up to 2,000 shares ($20,520). New CFG
min_atr_pct = 1.0 filters these out before the pattern check (same fix as the
MA sweep scanner). Scanner and generator patched identically; regenerated
output verified byte-identical.

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01WG6bhVDgntdGBGqtFYno57""")
for i in range(5):
    if subprocess.call(["git", "pull", "--rebase", "origin", "master"]) == 0 and \
       subprocess.call(["git", "push", "origin", "HEAD:master"]) == 0:
        print("✅ pushed"); break
else:
    sys.exit("❌ push failed")
