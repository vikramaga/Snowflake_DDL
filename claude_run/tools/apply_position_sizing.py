# One-off patch run by ClaudeRunner: inserts the position-sizing hook
# (see position_sizing.py) into every scanner that writes Entry/Stop_Loss,
# right before its df_out.to_csv(...) line, then compiles and commits.
# Idempotent: files that already contain the hook are left alone.
import re, subprocess, sys, py_compile, os

TARGETS = """cluster_reclaim_rsi_v1 confluence_scanner_v1 cup_breakout_5y_retest_v1 daily_3candle_higherlow_above_sma50_v1
daily_3candle_jma_reclaim_v1 daily_liquidity_sweep_breakout_v1 daily_price_jma_ema21_sma50_volume_v1 daily_rsi_bullish_divergence_v1
daily_rsi_high_breakout_v1 highlighted_pattern_653_scan_v1 india_weekly_reversal_2x_above_sma50_v1 jma_ema8_convergence_breakout_v1
master_technical_scanner_v1 monthly_3candle_reclaim_jma_ema8_v1 monthly_reversal_2x_above_sma50_v1 monthly_rsi_cross_above_avg_v1
rsi_daily_pop_weekly_monthly_confirm_v1 rsi_multi_tf_reversal_v1 rsi_near_avg_cross_sma_structure_v1 sma150_base_reclaim_v1
sma50_150_convergence_ema21_retest_v1 three_candle_hammer_doji_v1 weekly_reversal_2x_above_sma50_v1 weekly_rsi_bullish_divergence_v1""".split()

def hook(ind, var, extra=""):
    return (f"{ind}# ── Position sizing: Quantity / Total_Amount for a fixed max loss (see position_sizing.py)\n"
            f"{ind}try:\n"
            f"{ind}    from position_sizing import add_position_size\n"
            f"{ind}    {var} = add_position_size({var}{extra})\n"
            f"{ind}except Exception as _ps_err:\n"
            f"{ind}    print(f\"  ⚠️  Position sizing skipped: {{_ps_err}}\")\n")

def insert_before(lines, pattern, block):
    hits = [i for i, l in enumerate(lines) if re.match(pattern, l)]
    if len(hits) != 1:
        raise RuntimeError(f"expected 1 match for {pattern!r}, found {len(hits)}")
    i = hits[0]
    ind = re.match(r"^(\s*)", lines[i]).group(1)
    lines[i:i] = block(ind).rstrip("\n").split("\n")

changed = []
for t in TARGETS:
    p = f"{t}.py"
    s = open(p).read()
    if "add_position_size" in s:
        print(f"skip   {p} (already has hook)")
        continue
    lines = s.split("\n")
    extra = ', currency="INR"' if t.startswith("india_") else ""
    insert_before(lines, r"^(\s*)df_out\.to_csv\(", lambda ind: hook(ind, "df_out", extra))
    if t == "daily_3candle_jma_reclaim_v1":
        insert_before(lines, r"^(\s*)df_armed\.to_csv\(",
                      lambda ind: hook(ind, "df_armed", ', entry_col="Saved_High"'))
    open(p, "w").write("\n".join(lines))
    py_compile.compile(p, doraise=True)
    changed.append(p)
    print(f"patched {p}")

if not changed:
    print("nothing to change"); sys.exit(0)

def git(*a):
    print("+ git", " ".join(a)); subprocess.check_call(["git", *a])

git("config", "user.name", "github-actions[bot]")
git("config", "user.email", "41898+github-actions[bot]@users.noreply.github.com")
git("add", *changed)
git("commit", "-m",
    f"Add Quantity/Total_Amount position sizing ($40 max loss) to {len(changed)} scanners\n\n"
    "Inserted by claude_run/tools/apply_position_sizing.py via ClaudeRunner.\n\n"
    "Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>\n"
    "Claude-Session: https://claude.ai/code/session_01WG6bhVDgntdGBGqtFYno57")
for attempt in range(5):
    try:
        git("pull", "--rebase", "origin", "master")
        git("push", "origin", "HEAD:master")
        break
    except subprocess.CalledProcessError:
        if attempt == 4: raise
print(f"done: {len(changed)} files committed")
