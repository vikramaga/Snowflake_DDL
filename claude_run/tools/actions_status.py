# Lists recent runs of the Monday red-candle workflow (and the repo's latest runs)
# through the public GitHub API (no token needed for a public repo).
import json, urllib.request
API = "https://api.github.com/repos/vikramaga/Snowflake_DDL"
def get(path):
    req = urllib.request.Request(API + path, headers={"Accept": "application/vnd.github+json"})
    with urllib.request.urlopen(req, timeout=30) as r:
        return json.loads(r.read().decode())
for wf in ("monday_red_breakout.yml",):
    try:
        d = get(f"/actions/workflows/{wf}")
        print(f"{wf}: state={d.get('state')} path={d.get('path')} created={d.get('created_at')} updated={d.get('updated_at')}")
        runs = get(f"/actions/workflows/{wf}/runs?per_page=30").get("workflow_runs", [])
        print(f"  {len(runs)} runs")
        for r in runs:
            print(f"  {r['created_at']}  {r['event']:<18} {r['status']:<11} {str(r.get('conclusion')):<10} {r['id']}")
            if r.get("conclusion") not in (None, "success"):
                jobs = get(f"/actions/runs/{r['id']}/jobs").get("jobs", [])
                for j in jobs:
                    for s in j.get("steps", []):
                        if s.get("conclusion") not in ("success", "skipped", None):
                            print(f"      step failed: {s['name']} -> {s['conclusion']}")
    except Exception as e:
        print(f"{wf}: {e}")
print("\nLatest repo runs:")
for r in get("/actions/runs?per_page=15").get("workflow_runs", []):
    print(f"  {r['created_at']}  {r['name'][:32]:<32} {r['event']:<18} {r['status']:<11} {r.get('conclusion')}")
