"""Quick smoke test for all API endpoints. Run while the server is up."""
import httpx, json, sys

BASE = "http://localhost:8000"

results = []
def check(label, method, path, **kwargs):
    try:
        if method == "POST":
            r = httpx.post(f"{BASE}{path}", timeout=30, **kwargs)
        else:
            r = httpx.get(f"{BASE}{path}", timeout=30, **kwargs)
        ok = r.status_code < 400
        results.append((ok, f"{r.status_code}  {method} {path}"))
    except Exception as e:
        results.append((False, f"ERR  {method} {path}: {e}"))

check("health",    "GET",  "/health")
check("readiness", "GET",  "/readiness/Dumbbell Incline Bench Press")
check("plan",      "GET",  "/plan")
check("history",   "GET",  "/history", params={"last_n": 3})
check("search",    "GET",  "/search", params={"q": "hypertrophy volume", "k": 2})
check("explain",   "GET",  "/explain", params={"topic": "sleep and recovery"})
check("block",     "GET",  "/block", params={"weeks": 4})
check("ask",       "POST", "/ask", json={"question": "What did I bench recently?"})

for ok, msg in results:
    print(("OK " if ok else "FAIL") + "  " + msg)

failed = sum(1 for ok, _ in results if not ok)
print(f"\n{len(results) - failed}/{len(results)} passed")
sys.exit(0 if failed == 0 else 1)
