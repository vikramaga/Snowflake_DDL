# One-off patch run by ClaudeRunner (2026-11-15): MTF CHoCH + FVG scanner - main stop moves to the
# 5-min pullback low (structure); the FVG stop is kept as Aggressive_Stop with its own outcome.
# Applies exact replacements, checks the result's SHA-256, commits, then runs the scanner.
import hashlib, json, py_compile, subprocess, sys
F = "mtf_choch_fvg_v1.py"
REPS = json.loads("[[\"# Entry = trigger candle close; Stop = min(FVG bottom, retest low) - 0.1%;\\n# T1 = the 1H swing high (2R if already above it); T2 = 2R. After-entry outcome\", \"# Entry = trigger candle close; Stop = the 5-min pullback low (the swing the CHoCH\\n# came from) - 0.1%; Aggressive_Stop = min(FVG bottom, retest low) - 0.1%;\\n# T1 = the 1H swing high (2R if already above it); T2 = 2R. After-entry outcome\"], [\"    out.update({\\\"stage\\\": \\\"Triggered\\\", \\\"entry_time\\\": d.index[e], \\\"entry\\\": float(c[e]), \\\"stop\\\": float(stop),\", \"    out.update({\\\"stage\\\": \\\"Triggered\\\", \\\"entry_time\\\": d.index[e], \\\"entry\\\": float(c[e]), \\\"stop\\\": float(stop),\\n                \\\"stop_struct\\\": float(l[m]) * (1 - cfg[\\\"stop_buffer\\\"]),\"], [\"        entry, stop = r[\\\"entry\\\"], r[\\\"stop\\\"]\\n        risk = entry - stop\\n        if risk <= 0: continue\\n        t1 = a[\\\"high\\\"] if a[\\\"high\\\"] > entry else entry + 2 * risk\\n        outcome = \\\"open\\\"\\n        for _, b in r[\\\"_after\\\"].iterrows():\\n            if b[\\\"Low\\\"] <= stop: outcome = \\\"stopped\\\"; break\\n            if b[\\\"High\\\"] >= t1: outcome = \\\"T1 hit\\\"; break\\n        row.update({\\\"Entry_Price\\\": round(entry, 2), \\\"Stop_Loss\\\": round(stop, 2), \\\"Risk_%\\\": round(risk / entry * 100, 2),\", \"        entry, stop, aggr = r[\\\"entry\\\"], r[\\\"stop_struct\\\"], r[\\\"stop\\\"]\\n        risk = entry - stop\\n        if risk <= 0: continue\\n        t1 = a[\\\"high\\\"] if a[\\\"high\\\"] > entry else entry + 2 * risk\\n        def _outcome(sl):\\n            for _, b in r[\\\"_after\\\"].iterrows():\\n                if b[\\\"Low\\\"] <= sl: return \\\"stopped\\\"\\n                if b[\\\"High\\\"] >= t1: return \\\"T1 hit\\\"\\n            return \\\"open\\\"\\n        outcome = _outcome(stop)\\n        row.update({\\\"Aggressive_Stop\\\": round(aggr, 2), \\\"Outcome_Aggr\\\": _outcome(aggr),\\n                    \\\"Reward_R_Aggr\\\": round((t1 - entry) / (entry - aggr), 1) if entry > aggr else None})\\n        row.update({\\\"Entry_Price\\\": round(entry, 2), \\\"Stop_Loss\\\": round(stop, 2), \\\"Risk_%\\\": round(risk / entry * 100, 2),\"], [\"COLS = [\\\"Ticker\\\",\\\"Stage\\\",\\\"Price\\\",\\\"Entry_Price\\\",\\\"Stop_Loss\\\",\\\"Risk_%\\\",\\\"Target_1\\\",\\\"Reward_R\\\",\\\"Target_2\\\",\\\"Entry_Time_ET\\\",\\n        \\\"Outcome\\\",\\\"Last_5m\\\",\", \"COLS = [\\\"Ticker\\\",\\\"Stage\\\",\\\"Price\\\",\\\"Entry_Price\\\",\\\"Stop_Loss\\\",\\\"Risk_%\\\",\\\"Target_1\\\",\\\"Reward_R\\\",\\\"Target_2\\\",\\\"Entry_Time_ET\\\",\\n        \\\"Outcome\\\",\\\"Aggressive_Stop\\\",\\\"Reward_R_Aggr\\\",\\\"Outcome_Aggr\\\",\\\"Last_5m\\\",\"], [\"EMAIL_COLS = {\\\"Triggered\\\": [\\\"Ticker\\\",\\\"Entry_Time_ET\\\",\\\"Entry_Price\\\",\\\"Stop_Loss\\\",\\\"Risk_%\\\",\\\"Target_1\\\",\\\"Reward_R\\\",\\\"Outcome\\\",\\\"Open_R\\\",\\\"FVG_Bottom\\\",\\\"FVG_Top\\\",\\\"Retrace_%\\\"],\", \"EMAIL_COLS = {\\\"Triggered\\\": [\\\"Ticker\\\",\\\"Entry_Time_ET\\\",\\\"Entry_Price\\\",\\\"Stop_Loss\\\",\\\"Risk_%\\\",\\\"Target_1\\\",\\\"Reward_R\\\",\\\"Outcome\\\",\\\"Open_R\\\",\\\"Aggressive_Stop\\\",\\\"Outcome_Aggr\\\",\\\"FVG_Bottom\\\",\\\"FVG_Top\\\",\\\"Retrace_%\\\"],\"], [\"then a close above the retest candle's high = entry. Stop = min(FVG bottom, retest low) - 0.1%; T1 = the 1H high (2R if already above). Times are US Eastern.\", \"then a close above the retest candle's high = entry. Stop = the 5-min pullback low - 0.1% (Aggressive_Stop = min(FVG bottom, retest low) - 0.1%); T1 = the 1H high (2R if already above). Times are US Eastern.\"]]")
WANT = "2f2c07964139038e4bc6623b5af1d8ae7ff42e184dc7420f6041745a1b025fe3"
s = open(F).read()
for a, b in REPS:
    if s.count(a) != 1: sys.exit(f"expected 1 match, found {s.count(a)}: {a[:60]!r} - nothing changed")
    s = s.replace(a, b)
open(F, "w").write(s)
got = hashlib.sha256(open(F, "rb").read()).hexdigest()
if got != WANT: sys.exit(f"sha mismatch {got} != {WANT}")
py_compile.compile(F, doraise=True)
print("patched", got)
def git(*a):
    print("+ git", " ".join(a)); subprocess.check_call(["git", *a])
git("config", "user.name", "Scanner Bot")
git("config", "user.email", "vikram-scanner-bot@users.noreply.github.com")
git("add", F)
git("commit", "-m", """MTF CHoCH + FVG: stop at the 5-min pullback low; FVG stop kept as Aggressive_Stop

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01WG6bhVDgntdGBGqtFYno57""")
for i in range(5):
    if subprocess.call(["git", "pull", "--rebase", "origin", "master"]) == 0 and \
       subprocess.call(["git", "push", "origin", "HEAD:master"]) == 0:
        print("pushed"); break
else:
    sys.exit("push failed")
sys.exit(subprocess.call([sys.executable, F]))
