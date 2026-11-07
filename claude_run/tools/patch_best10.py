# Adds the Best 10 shortlist (best_picks.py) to every scanner in
# claude_run/schedule.txt, with two inserted lines per scanner:
#   1. right after  `df_out = add_position_size(df_out...)`
#        -> best_picks.rank_best(df_out)      (ranks, saves best10_*.csv)
#   2. right before `srv.sendmail(gu, eto, msg.as_string())`
#        -> best_picks.attach_to_email(msg)   (Best 10 table on top + CSV)
# Each insertion is guarded so a problem can never stop the scanner.
# Also makes the RSI digest and the Monday breakout scanner ignore the new
# best10_*.csv files. Every edit is checked (exact match count); files that
# are already patched are left alone. Usage: python patch_best10.py [repo_dir]
import os, re, sys, py_compile

ROOT = sys.argv[1] if len(sys.argv) > 1 else "."
MARK = "best_picks"


def patch_scanner(path):
    s = open(path).read()
    if MARK in s:
        return "already patched"
    lines = s.split("\n")
    a = [i for i, l in enumerate(lines) if re.match(r"^\s*df_out = add_position_size\(df_out\b", l)]
    b = [i for i, l in enumerate(lines) if l.strip() == "srv.sendmail(gu, eto, msg.as_string())"]
    if len(a) != 1 or len(b) != 1:
        sys.exit(f"❌ {path}: expected 1 sizing line and 1 sendmail line, found {len(a)} / {len(b)}")
    ia, ib = a[0], b[0]
    ind_a = re.match(r"^(\s*)", lines[ia]).group(1)
    ind_b = re.match(r"^(\s*)", lines[ib]).group(1)
    ins_b = [f"{ind_b}try: __import__(\"best_picks\").attach_to_email(msg)",
             f"{ind_b}except Exception as _bp_err: print(f\"  ⚠️  Best 10 not added: {{_bp_err}}\")"]
    ins_a = [f"{ind_a}try: __import__(\"best_picks\").rank_best(df_out)",
             f"{ind_a}except Exception as _bp_err: print(f\"  ⚠️  Best 10 skipped: {{_bp_err}}\")"]
    # insert the later one first so the earlier index stays valid
    if ib > ia:
        lines[ib:ib] = ins_b
        lines[ia + 1:ia + 1] = ins_a
    else:
        lines[ia + 1:ia + 1] = ins_a
        lines[ib:ib] = ins_b
    open(path, "w").write("\n".join(lines))
    py_compile.compile(path, doraise=True)
    return "patched"


def patch_text(path, old, new):
    s = open(path).read()
    if new in s:
        return "already patched"
    if s.count(old) != 1:
        sys.exit(f"❌ {path}: expected 1 match, found {s.count(old)}: {old[:70]!r}")
    open(path, "w").write(s.replace(old, new))
    py_compile.compile(path, doraise=True)
    return "patched"


sched = [l.strip() for l in open(os.path.join(ROOT, "claude_run/schedule.txt")) if l.strip() and not l.startswith("#")]
for f in sched:
    print(f"{patch_scanner(os.path.join(ROOT, f)):16s} {f}")
print(patch_text(os.path.join(ROOT, "claude_run/tools/rsi_amount_filter.py"),
                 '        if b.startswith("tv_"):\n            continue\n        df = pd.read_csv(c)',
                 '        if b.startswith(("tv_", "best10_")):\n            continue\n        df = pd.read_csv(c)'),
      "claude_run/tools/rsi_amount_filter.py")
print(patch_text(os.path.join(ROOT, "monday_red_candle_breakout_v1.py"),
                 'if not os.path.basename(f).startswith("tv_")]',
                 'if not os.path.basename(f).startswith(("tv_", "best10_"))]'),
      "monday_red_candle_breakout_v1.py")
print("✅ done")
