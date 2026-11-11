#!/usr/bin/env python3
"""Recompute the published paired statistics and all retained timing flags."""
from pathlib import Path
import csv
import hashlib
import json
import math
from paired_statistics import summarize

root = Path(__file__).resolve().parent
pairs = list(csv.DictReader((root / "paired.csv").open()))
requests = json.loads((root / "request-timings.json").read_text())
expected = json.loads((root / "analysis.json").read_text())
assert len(pairs) == 16 and len(requests) == 48
assert len({(x["phase"], x["case"], x["rep"]) for x in requests}) == 48
by_key = {(x["phase"], x["case"], x["rep"]): x for x in requests}
workload = json.loads((root / "workload.json").read_text())
assert len(workload["model_files"]) == 4
assert hashlib.sha256(workload["prompt_2048"].encode()).hexdigest() == next(
    x["prompt_sha256"] for x in requests if x["case"] == "pp2048-tg128")
flags = []
for pair in range(1, 9):
    for case, rep in [("warmup", 0), ("pp2048-tg128", 1), ("pp2048-tg128", 2)]:
        baseline = by_key[(f"pair{pair:02}-baseline", case, rep)]
        candidate = by_key[(f"pair{pair:02}-candidate", case, rep)]
        for key in ("prompt_tokens", "output_tokens", "prompt_sha256", "completion_sha256",
                    "drafted", "accepted", "physical_width", "rounds"):
            assert baseline[key] == candidate[key], (pair, case, rep, key)
        for metric, old in baseline["metrics"].items():
            new = candidate["metrics"].get(metric)
            if new is not None and new - old > 3 and new > old * 1.05:
                flags.append((pair, case, rep, metric))
published = list(csv.DictReader((root / "phase-flags.csv").open()))
assert set(flags) == {(int(x["pair"]), x["case"], int(x["rep"]), x["metric"])
                      for x in published}
assert len(flags) == len(published) == 13
for rep in (1, 2):
    rows = [x for x in pairs if int(x["rep"]) == rep]
    logs = [float(x["log_ratio"]) for x in rows]
    for row, log in zip(rows, logs):
        pair = int(row["pair"])
        a = by_key[(f"pair{pair:02}-baseline", "pp2048-tg128", rep)]["metrics"]["decode_ms"]
        b = by_key[(f"pair{pair:02}-candidate", "pp2048-tg128", rep)]["metrics"]["decode_ms"]
        assert math.isclose(log, math.log(a / b), abs_tol=1e-14)
    result = summarize(logs, [1 if x["order"].startswith("candidate") else -1 for x in rows])
    reference = next(x for x in expected["results"] if x["rep"] == rep)
    for key, value in result.items():
        assert value == reference[key], (rep, key, value, reference[key])
    print(json.dumps({"rep": rep, "statistics": result}, indent=2))
print("PASS: 48 output/work records, both paired summaries and all 13 timing flags")
