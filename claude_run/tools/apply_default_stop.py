# One-off patch run by ClaudeRunner: for scanners that don't compute a stop,
# inserts a hook before df_out.to_csv(...) that adds Stop_Loss (10-day low)
# and then Quantity / Total_Amount (see position_sizing.py). Compiles and commits.
# Idempotent: files that already contain add_position_size are left alone.
import re, subprocess, sys, py_compile

TARGETS = """allmacross basebreakouth4 crossema20 fundamental_v2 jma_stack_cross_v1 macompress
minervini multi_tf_uptrend_v1 nasdaq_h4_pullback_recovery retestbullengulf sma150bo_7mnth
sma8withh4 smareclaim smaretest2 smasupport stagev2 trendline_h4 veryfund""".split()

def hook(ind):
    return (f"{ind}# ── Default stop (10-day low) + Quantity / Total_Amount for a fixed max loss (see position_sizing.py)\n"
            f"{ind}try:\n"
            f"{ind}    from position_sizing import add_default_stop, add_position_size\n"
            f"{ind}    df_out = add_default_stop(df_out)\n"
            f"{ind}    df_out = add_position_size(df_out)\n"
            f"{ind}except Exception as _ps_err:\n"
            f"{ind}    print(f\"  ⚠️  Position sizing skipped: {{_ps_err}}\")\n")

changed = []
for t in TARGETS:
    p = f"{t}.py"
    s = open(p).read()
    if "add_position_size" in s:
        print(f"skip   {p} (already has hook)")
        continue
    lines = s.split("\n")
    hits = [i for i, l in enumerate(lines) if re.match(r"^(\s*)df_out\s*\.to_csv\(", l)]
    if len(hits) != 1:
        raise RuntimeError(f"{p}: expected 1 df_out.to_csv, found {len(hits)}")
    i = hits[0]
    ind = re.match(r"^(\s*)", lines[i]).group(1)
    lines[i:i] = hook(ind).rstrip("\n").split("\n")
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
    f"Add 10-day-low default stop + Quantity/Total_Amount ($40 max loss) to {len(changed)} scanners\n\n"
    "Inserted by claude_run/tools/apply_default_stop.py via ClaudeRunner.\n\n"
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
