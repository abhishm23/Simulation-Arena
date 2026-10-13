"""Independent retest of every CORRECTNESS CONTRACT invariant across all 6 use cases.

This is the evidence the QA workbook cites. QA can re-run it as-is.

    cd backend
    "C:/Users/Abhishek Mishra/.workbuddy-ai/binaries/python/envs/simarena/Scripts/python.exe" qa_retest.py

Needs the API running (`python -m app.main`). Exits 0 only if all 12 runs pass
every invariant and determinism holds.
"""
import json, http.client, math, re, time, sys

HOST, PORT = "127.0.0.1", 8000
PCT = re.compile(r"(-?\d+(?:\.\d+)?)\s*%")


def req(m, p, b=None, raw=False):
    d = None if b is None else json.dumps(b).encode()
    c = http.client.HTTPConnection(HOST, PORT, timeout=300)
    c.request(m, p, body=d, headers={"Content-Type": "application/json"})
    r = c.getresponse()
    x = r.read()
    c.close()
    return x if raw else json.loads(x)


def wait(rid, limit=300):
    t0 = time.time()
    while time.time() - t0 < limit:
        r = req("GET", f"/api/runs/{rid}/result")
        if r.get("headline") is not None or r.get("status") == "failed":
            return r
        time.sleep(0.3)
    return {"status": "timeout"}


def scan_numbers(obj, path="", out=None):
    """Collect every numeric leaf and every %-formatted string."""
    if out is None:
        out = {"nums": [], "pcts": [], "bad": []}
    if isinstance(obj, dict):
        for k, v in obj.items():
            scan_numbers(v, f"{path}.{k}", out)
    elif isinstance(obj, list):
        for i, v in enumerate(obj):
            scan_numbers(v, f"{path}[{i}]", out)
    elif isinstance(obj, str):
        for m in PCT.finditer(obj):
            v = float(m.group(1))
            out["pcts"].append((f"{path}={obj[:60]!r}", v))
    elif isinstance(obj, bool):
        pass
    elif isinstance(obj, (int, float)):
        out["nums"].append((path, float(obj)))
    return out


results = {}
ucs = [u["id"] for u in req("GET", "/api/usecases")]
samples = {s["use_case_id"]: s["name"] for s in req("GET", "/api/samples")}
print(f"use cases: {len(ucs)}  samples: {len(samples)}\n")

fails = []
for uid in ucs:
    for mode, ds in (("manual", None), ("sample", samples.get(uid))):
        body = {"use_case_id": uid, "seed": 42,
                "output_formats": ["html", "xlsx", "csv", "pdf"]}
        if ds:
            prof = req("POST", f"/api/datasets/sample/{ds}")
            am = req("POST", f"/api/usecases/{uid}/automap", {"dataset_id": prof["dataset_id"]})
            body["dataset_id"] = prof["dataset_id"]
            body["role_mapping"] = am["mapping"]
        rid = req("POST", "/api/runs", body)["run_id"]
        r = wait(rid)
        label = f"{uid}/{mode}"
        if r.get("status") != "succeeded":
            fails.append(f"{label}: status={r.get('status')} {r.get('headline','')[:80]}")
            print(f"FAIL {label}: {r.get('status')}")
            continue

        blob = json.dumps(r)
        checks = {}

        # 0. warning hygiene. Found the hard way: a numeric role auto-mapped to a
        #    category column produced 4 warnings claiming 450 unreadable cells
        #    for data the model never needed, one warning per distinct value.
        warns = r.get("warnings") or []
        checks["no_dupe_warnings"] = len(warns) == len(set(warns))
        checks["no_coercion_noise"] = len([w for w in warns if re.search(
            r"could not be read|are not numbers|are not dates|are not yes/no",
            str(w), re.I)]) <= 1

        # 0b. mapping sanity: a numeric role must never be bound to a column
        #     that holds no numbers, and nothing should map below the floor.
        if ds:
            colmap = {c["name"]: c for c in prof["columns"]}
            rdtype = {}
            for u in req("GET", "/api/usecases"):
                if u["id"] == uid:
                    rdtype = {x["key"]: x.get("dtype", "")
                              for x in u.get("input_roles", [])}
            bad_map = []
            for role, col in am["mapping"].items():
                if not col:
                    continue
                p = colmap.get(col, {})
                has_num = (p.get("numeric_stats") is not None
                           or p.get("dtype") in ("numeric", "identifier"))
                if rdtype.get(role) == "numeric" and not has_num:
                    bad_map.append(f"{role}->{col}({p.get('dtype')})")
                if am["confidence"].get(role, 0) < 0.45:
                    bad_map.append(f"{role}->{col}(low confidence)")
            checks["mapping_sane"] = not bad_map
            if bad_map:
                print(f"      !! bad mapping: {bad_map[:4]}")

        # 1. no NaN/Infinity tokens anywhere
        checks["no_nan"] = not re.search(r"\b(NaN|Infinity|-Infinity)\b", blob)

        # 2. every PROBABILITY is in 0..100. Other percentages legitimately go
        #    negative (a price cut of -19.9%, a loss-making margin of -2.6%), so
        #    only probability-bearing fields are checked - by label, not by regex.
        PROB = re.compile(r"chance|probab|odds|share of|risk score|% of deals|"
                          r"likely to|how often", re.I)
        bad_pct = []

        for k in r["kpis"]:
            if PROB.search(k["label"]):
                for m in PCT.finditer(str(k["value"])):
                    v = float(m.group(1))
                    if v < 0 or v > 100:
                        bad_pct.append((f"kpi:{k['label']}", v))

        for c in r["charts"]:
            for s in c["series"]:
                if PROB.search(str(s.get("name", ""))):
                    for v in (s.get("values") or []):
                        if isinstance(v, (int, float)) and (v < 0 or v > 100):
                            bad_pct.append((f"chart:{c['id']}/{s.get('name')}", v))

        for t in r["tables"]:
            prob_cols = [i for i, h in enumerate(t["columns"]) if PROB.search(str(h))]
            for row in t["rows"]:
                for i in prob_cols:
                    if i >= len(row):
                        continue
                    for m in PCT.finditer(str(row[i])):
                        v = float(m.group(1))
                        if v < 0 or v > 100:
                            bad_pct.append((f"table:{t['name']}[{t['columns'][i]}]", v))

        # probability-labelled series must also never be negative
        checks["pct_in_range"] = not bad_pct
        sc = scan_numbers(r)

        # 3. table rows match headers
        mis = [(t["name"], len(t["columns"]), sorted({len(x) for x in t["rows"]}))
               for t in r["tables"] if t["rows"] and any(len(x) != len(t["columns"]) for x in t["rows"])]
        checks["tables_aligned"] = not mis

        # 4. no non-finite numbers in the raw model fields
        checks["finite"] = all(math.isfinite(v) for _, v in sc["nums"])

        # 5. charts present, each series length matches x where both exist
        chart_bad = []
        for c in r["charts"]:
            for s in c["series"]:
                vals = s.get("values") or []
                if c.get("x") and len(c["x"]) != len(vals):
                    chart_bad.append((c["id"], s.get("name"), len(c["x"]), len(vals)))
        checks["charts_aligned"] = not chart_bad

        # 6. highlight rows valid
        hl_bad = [t["name"] for t in r["tables"]
                  if any(h < 0 or h >= len(t["rows"]) for h in (t.get("highlight_rows") or []))]
        checks["highlight_valid"] = not hl_bad

        # 7. artifacts all four, non-trivial
        arts = {}
        for f in ("html", "xlsx", "csv", "pdf"):
            try:
                arts[f] = len(req("GET", f"/api/runs/{rid}/artifact/{f}", raw=True))
            except Exception as e:
                arts[f] = 0
        checks["artifacts"] = all(v > 3000 for v in arts.values())

        # 8. HTML report stays offline
        html = req("GET", f"/api/runs/{rid}/artifact/html", raw=True).decode("utf-8", "ignore")
        checks["html_offline"] = not (("http://" in html.replace("http://www.w3.org", ""))
                                      or ("https://" in html))

        ok = all(checks.values())
        results[label] = (ok, checks, r)
        if not ok:
            fails.append(f"{label}: {[k for k, v in checks.items() if not v]}")

        flag = "PASS" if ok else "FAIL"
        print(f"{flag} {label:<28} {r.get('runtime_ms',0):>5}ms  "
              f"kpi={len(r['kpis'])} charts={len(r['charts'])} tables={len(r['tables'])} "
              f"story={len(r['story'])} warn={len(r['warnings'])}")
        if bad_pct:
            print(f"      !! out-of-range %: {bad_pct[:4]}")
        if mis:
            print(f"      !! misaligned tables: {mis}")
        if chart_bad:
            print(f"      !! chart mismatch: {chart_bad[:3]}")

# determinism through the API
def headline_of(uid):
    rid = req("POST", "/api/runs", {"use_case_id": uid, "seed": 42,
                                    "output_formats": []})["run_id"]
    r = wait(rid)
    return r.get("headline"), [k["value"] for k in r.get("kpis", [])]

det_ok = True
for uid in ("margin_floor", "pipeline_hopper", "deal_approval"):
    a, b = headline_of(uid), headline_of(uid)
    if a != b:
        det_ok = False
        print(f"DETERMINISM FAIL {uid}")

print(f"\nDETERMINISM: {'PASS' if det_ok else 'FAIL'}")
print(f"RUNS: {sum(1 for v in results.values() if v[0])}/{len(results)} passed")
print(f"FAILURES: {fails if fails else 'none'}")
sys.exit(1 if (fails or not det_ok) else 0)
