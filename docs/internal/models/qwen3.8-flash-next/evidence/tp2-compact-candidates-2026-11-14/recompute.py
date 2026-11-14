"""Recompute the fixed primary from paired CSV; no GPU or lab paths needed."""
from pathlib import Path
import csv
import itertools
import json
import math
import random
import statistics

p = Path(__file__).resolve().parent
design = json.loads((p / "DESIGN.json").read_text())
expected = json.loads((p / "analysis.json").read_text())
with (p / "case-pairs.csv").open() as f:
    rows = list(csv.DictReader(f))
pairs = design["pairs"]
effects = []
for pair in pairs:
    sampled = [r for r in rows if int(r["pair"]) == pair["pair"]
               and r["cohort"] != "control-greedy"]
    assert len(sampled) == 3
    effects.append(statistics.mean(float(r["log_effect"]) for r in sampled))
observed = statistics.mean(effects)
chrono = [effect if pair["baseline_first"] else -effect
          for pair, effect in zip(pairs, effects, strict=True)]
extreme = total = 0
for positive in itertools.combinations(range(20), 10):
    positive = set(positive)
    value = statistics.mean((1 if i in positive else -1) * x
                            for i, x in enumerate(chrono))
    extreme += abs(value) >= abs(observed) - 1e-14
    total += 1
strata = [[e for pair, e in zip(pairs, effects, strict=True)
           if pair["baseline_first"] == order] for order in (False, True)]
rng = random.Random(design["bootstrap_seed"])
boot = sorted(statistics.mean([rng.choice(group) for group in strata
                               for _ in range(10)]) for _ in range(20000))
def quantile(q):
    at = (len(boot) - 1) * q
    lo = math.floor(at)
    return boot[lo] * (1 - (at - lo)) + boot[math.ceil(at)] * (at - lo)
result = {"primary_sampled_decode_change_pct": math.expm1(observed) * 100,
          "order_stratified_bootstrap95_pct": [math.expm1(quantile(q)) * 100
                                                for q in (.025, .975)],
          "exact_balanced_order_two_sided_p": extreme / total,
          "randomizations": total, "positive_pairs": sum(e > 0 for e in effects)}
for key, value in result.items():
    if isinstance(value, list):
        assert all(abs(x-y) < 1e-12 for x,y in zip(value, expected[key], strict=True))
    else:
        assert abs(value - expected[key]) < 1e-12, key
print(json.dumps(result, indent=2))
