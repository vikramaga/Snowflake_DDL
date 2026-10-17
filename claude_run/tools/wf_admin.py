# Workflow admin run by ClaudeRunner, using the WORKFLOW_TOKEN repo secret
# (fine-grained PAT: Contents, Workflows, Actions = read/write on this repo).
#
# Reads commands from claude_run/wf_admin.txt, one per line:
#   check                 token works? list workflows (state, schedule, script) + recent runs
#   runs [N]              last N runs (default 15)
#   sync                  copy claude_run/workflows/*.yml -> .github/workflows/ and push
#   disable <file.yml>    disable a workflow (stops its schedule; file is kept)
#   enable  <file.yml>    re-enable it
#   dispatch <file.yml>   start a run now (workflow must have workflow_dispatch)
#   disable-scheduled     disable every workflow whose script is in claude_run/schedule.txt
#                         (ClaudeRunner already runs those), except ClaudeRunner / RSI Digest
import json, os, re, subprocess, sys, urllib.request, urllib.error, glob, filecmp, shutil

REPO = os.environ.get("GITHUB_REPOSITORY", "vikramaga/Snowflake_DDL")
TOKEN = os.environ.get("WORKFLOW_TOKEN", "").strip()
API = f"https://api.github.com/repos/{REPO}"
KEEP = {"Claude_runner.yml", "rsi_digest.yml"}
fail = 0


def api(method, path, body=None):
    req = urllib.request.Request(API + path, method=method,
                                 data=json.dumps(body).encode() if body is not None else None)
    req.add_header("Authorization", f"Bearer {TOKEN}")
    req.add_header("Accept", "application/vnd.github+json")
    req.add_header("X-GitHub-Api-Version", "2022-11-28")
    try:
        with urllib.request.urlopen(req, timeout=30) as r:
            txt = r.read().decode()
            return r.status, (json.loads(txt) if txt.strip() else {})
    except urllib.error.HTTPError as e:
        return e.code, {"message": e.read().decode()[:300]}


def workflows():
    st, d = api("GET", "/actions/workflows?per_page=100")
    return st, d.get("workflows", [])


def wf_scripts(fname):
    try:
        s = open(f".github/workflows/{fname}").read()
    except OSError:
        return [], ""
    scripts = sorted(set(re.findall(r"python3?\s+([A-Za-z0-9_./-]+\.py)", s)))
    cron = ", ".join(re.findall(r"cron:\s*['\"]([^'\"]+)['\"]", s))
    return scripts, cron


def cmd_check():
    global fail
    st, d = api("GET", "")
    if st != 200:
        print(f"❌ Token check failed: HTTP {st} {d.get('message','')}"); fail += 1; return
    print(f"✅ Token works for {d.get('full_name')} (push access: {d.get('permissions',{}).get('push')})")
    st, wfs = workflows()
    if st != 200:
        print(f"❌ Listing workflows failed: HTTP {st}"); fail += 1; return
    print(f"\n{len(wfs)} workflows:")
    for w in sorted(wfs, key=lambda w: w["path"]):
        f = os.path.basename(w["path"])
        scripts, cron = wf_scripts(f)
        print(f"  {w['state']:<20} {f:<55} cron[{cron}]  {' '.join(scripts)}")
    cmd_runs(10)


def cmd_runs(n=15):
    st, d = api("GET", f"/actions/runs?per_page={int(n)}")
    if st != 200:
        print(f"❌ Listing runs failed: HTTP {st}"); return
    print(f"\nLast {n} runs:")
    for r in d.get("workflow_runs", []):
        print(f"  {r['created_at']}  {r['name'][:40]:<40} {r['event']:<18} {r['status']:<11} {r.get('conclusion') or '':<10} {r['id']}")


def set_state(fname, on):
    global fail
    st, d = api("PUT", f"/actions/workflows/{fname}/{'enable' if on else 'disable'}")
    ok = st in (200, 204)
    fail += not ok
    print(f"  {'✅' if ok else '❌'} {'enable ' if on else 'disable'} {fname}" + ("" if ok else f"  HTTP {st} {d.get('message','')}"))


def cmd_disable_scheduled():
    sched = {l.strip() for l in open("claude_run/schedule.txt") if l.strip() and not l.startswith("#")}
    st, wfs = workflows()
    print(f"Disabling workflows already covered by ClaudeRunner ({len(sched)} scheduled scripts):")
    hits = 0
    for w in sorted(wfs, key=lambda w: w["path"]):
        f = os.path.basename(w["path"])
        scripts, _ = wf_scripts(f)
        if f in KEEP or not scripts or not set(scripts) <= sched:
            continue
        hits += 1
        if w["state"] == "active":
            set_state(f, False)
        else:
            print(f"  –  {f} already {w['state']}")
    print(f"{hits} matching workflows")


def git(*a):
    print("+ git", " ".join(a)); subprocess.check_call(["git", *a])


def cmd_sync():
    src = sorted(glob.glob("claude_run/workflows/*.yml"))
    changed = []
    for s in src:
        dst = os.path.join(".github/workflows", os.path.basename(s))
        if not os.path.exists(dst) or not filecmp.cmp(s, dst, shallow=False):
            shutil.copyfile(s, dst); changed.append(dst)
    if not changed:
        print("sync: nothing to change"); return
    # push with the PAT (GITHUB_TOKEN can't write workflow files)
    subprocess.call(["git", "config", "--unset-all", "http.https://github.com/.extraheader"])
    git("remote", "set-url", "origin", f"https://x-access-token:{TOKEN}@github.com/{REPO}.git")
    git("config", "user.name", "github-actions[bot]")
    git("config", "user.email", "41898+github-actions[bot]@users.noreply.github.com")
    git("add", *changed)
    git("commit", "-m", "Sync workflows: " + ", ".join(os.path.basename(c) for c in changed) +
        "\n\nCo-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>\n"
        "Claude-Session: https://claude.ai/code/session_01WG6bhVDgntdGBGqtFYno57")
    for i in range(5):
        if subprocess.call(["git", "pull", "--rebase", "origin", "master"]) == 0 and \
           subprocess.call(["git", "push", "origin", "HEAD:master"]) == 0:
            print(f"✅ synced {len(changed)} workflow file(s)"); return
    raise SystemExit("❌ push failed")


def main():
    if not TOKEN:
        print("❌ WORKFLOW_TOKEN is empty — add the repo secret and pass it in Claude_runner.yml"); sys.exit(1)
    lines = [l.strip() for l in open("claude_run/wf_admin.txt") if l.strip() and not l.lstrip().startswith("#")]
    for l in lines:
        p = l.split()
        print(f"\n▶ {l}")
        if p[0] == "check": cmd_check()
        elif p[0] == "runs": cmd_runs(p[1] if len(p) > 1 else 15)
        elif p[0] == "sync": cmd_sync()
        elif p[0] in ("disable", "enable"): set_state(p[1], p[0] == "enable")
        elif p[0] == "dispatch":
            st, d = api("POST", f"/actions/workflows/{p[1]}/dispatches", {"ref": "master"})
            print(f"  {'✅ dispatched' if st == 204 else '❌ HTTP ' + str(st) + ' ' + d.get('message','')} {p[1]}")
        elif p[0] == "disable-scheduled": cmd_disable_scheduled()
        else: print(f"  unknown command: {l}")
    sys.exit(1 if fail else 0)


if __name__ == "__main__":
    main()
