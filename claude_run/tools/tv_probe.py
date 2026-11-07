# Read-only TradingView check: signs in with TV_SESSIONID / TV_SESSIONID_SIGN
# and prints your watchlists and the shape of the API answer. Changes nothing;
# never prints the cookies.
import json, os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import tv_watchlist_sync as tv

print("secrets present:", bool(os.environ.get("TV_SESSIONID")), bool(os.environ.get("TV_SESSIONID_SIGN")),
      "lengths", len(os.environ.get("TV_SESSIONID", "")), len(os.environ.get("TV_SESSIONID_SIGN", "")))
try:
    s = tv.session()
    r = s.get(tv.API, timeout=20)
    print("GET status", r.status_code, "content-type", r.headers.get("content-type"))
    try:
        data = r.json()
        print("type", type(data).__name__, "len", len(data) if hasattr(data, "__len__") else "-")
        items = data if isinstance(data, list) else [data]
        for x in items[:12]:
            if isinstance(x, dict):
                print("  keys", sorted(x.keys()))
                print("  id", x.get("id"), "name", repr(x.get("name")), "symbols", len(x.get("symbols") or []),
                      "first", (x.get("symbols") or [])[:4])
    except ValueError:
        print("not JSON:", r.text[:200])
except Exception as err:
    print("error:", type(err).__name__, err)
print("--- probe via sync module ---")
sys.exit(tv.main(["--probe"]))
