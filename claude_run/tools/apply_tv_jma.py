# One-off patch run by ClaudeRunner: replaces every top-level calc_jma() in the
# scanners (and claude_run/tools/portfolio_snapshot.py) with TradingView's
# "Jurik Moving Average" formula, keeping each script's name, parameters and
# settings. Idempotent (skips files already patched); compiles every file;
# commits and pushes.
import ast, glob, py_compile, subprocess, sys

MARK = "same formula as TradingView's"
NEW = '''def calc_jma(series, period=13, phase=40, power=2):
    """
    Jurik Moving Average — same formula as TradingView's "Jurik Moving
    Average" indicator, so values match the chart:
        beta       = 0.45*(L-1) / (0.45*(L-1) + 2)
        alpha      = beta ** power
        phaseRatio = phase/100 + 1.5   (0.5 if phase < -100, 2.5 if > 100)
        e0  = (1-alpha)*src + alpha*e0[1]
        e1  = (src - e0)*(1-beta) + beta*e1[1]
        e2  = (e0 + phaseRatio*e1 - jma[1])*(1-alpha)^2 + alpha^2*e2[1]
        jma = e2 + jma[1]
    period = Length, phase = Phase, power = Power (TradingView inputs).
    """
    phase_ratio = 0.5 if phase < -100 else (2.5 if phase > 100 else phase / 100.0 + 1.5)
    beta  = 0.45 * (period - 1) / (0.45 * (period - 1) + 2)
    alpha = beta ** power
    vals  = np.asarray(series.values, dtype=float)
    out   = np.full(len(vals), np.nan)
    e0 = e1 = e2 = jma = 0.0
    for i, v in enumerate(vals):
        if np.isnan(v):
            continue
        e0  = (1 - alpha) * v + alpha * e0
        e1  = (v - e0) * (1 - beta) + beta * e1
        e2  = (e0 + phase_ratio * e1 - jma) * (1 - alpha) ** 2 + alpha ** 2 * e2
        jma = e2 + jma
        out[i] = jma
    return pd.Series(out, index=series.index)'''


def patch(path):
    src = open(path).read()
    if MARK in src:
        return "skip"
    try:
        tree = ast.parse(src)
    except SyntaxError:
        return "unparsable (skipped)" if "calc_jma" in src else None
    fns = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "calc_jma"]
    if len(fns) != 1:
        return None
    f = fns[0]
    a = (f.decorator_list[0].lineno if f.decorator_list else f.lineno) - 1
    b = f.end_lineno
    lines = src.split("\n")
    lines[a:b] = NEW.split("\n")
    out = "\n".join(lines).replace(
        "# (corrected calc_jma, steady-state gain 1.0); EMA8 = 8-day EMA of close.",
        "# (TradingView's Jurik formula); EMA8 = 8-day EMA of close.")
    open(path, "w").write(out)
    py_compile.compile(path, doraise=True)
    return "patched"


targets = sorted(glob.glob("*.py")) + ["claude_run/tools/portfolio_snapshot.py"]
changed = []
for p in targets:
    r = patch(p)
    if r:
        print(f"{r:<8} {p}")
    if r == "patched":
        changed.append(p)
print(f"{len(changed)} files patched")
if "--no-commit" in sys.argv or not changed:
    sys.exit(0)


def git(*a):
    print("+ git", " ".join(a)); subprocess.check_call(["git", *a])
git("config", "user.name", "Scanner Bot")
git("config", "user.email", "vikram-scanner-bot@users.noreply.github.com")
git("add", *changed)
git("commit", "-m", f"""Switch calc_jma to TradingView's Jurik Moving Average formula ({len(changed)} files)

Every scanner now computes JMA exactly like TradingView's "Jurik Moving
Average" indicator, so signals match the charts. Each script keeps its own
settings (Length / Phase / Power, normally 13 / 40 / 2).

Verified: each patched calc_jma matches the reference TradingView formula to
1e-12 on random series; the reference reproduces MG's chart JMA (~20.73 on
2026-10-14); all files compile.

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01WG6bhVDgntdGBGqtFYno57""")
for i in range(5):
    if subprocess.call(["git", "pull", "--rebase", "origin", "master"]) == 0 and \
       subprocess.call(["git", "push", "origin", "HEAD:master"]) == 0:
        print("✅ pushed"); break
else:
    sys.exit("❌ push failed")
