"""Reproduce the fixed ten-pair sampled-decode result from per-case rates."""
from pathlib import Path
import csv
import itertools
import json
import math
import random
import statistics


def reproduce(path: Path) -> dict:
    groups = {}
    with path.open(encoding="utf-8", newline="") as source:
        for row in csv.DictReader(source):
            groups.setdefault(int(row["pair"]), []).append(row)
    assert sorted(groups) == list(range(1, 11))
    effects, chronological, strata = [], [], {False: [], True: []}
    for number in sorted(groups):
        rows = groups[number]
        assert len(rows) == 4
        assert {(r["cohort"], r["case"]) for r in rows} == {
            ("control-greedy", "scene"), ("c1", "scene"),
            ("c1", "explain"), ("pp2k", "scene"),
        }
        assert len({r["baseline_first"] for r in rows}) == 1
        assert rows[0]["baseline_first"] in ("True", "False")
        first = rows[0]["baseline_first"] == "True"
        values = []
        for row in rows:
            baseline, candidate = float(row["baseline_tps"]), float(row["candidate_tps"])
            assert all(math.isfinite(x) and x > 0 for x in (baseline, candidate))
            value = math.log(candidate / baseline)
            assert abs(value - float(row["log_effect"])) < 1e-14
            if row["cohort"] != "control-greedy":
                values.append(value)
        effect = statistics.mean(values)
        effects.append(effect)
        chronological.append(effect if first else -effect)
        strata[first].append(effect)
    assert all(len(values) == 5 for values in strata.values())
    observed = statistics.mean(effects)
    permutations = []
    for plus in itertools.combinations(range(10), 5):
        positive = set(plus)
        permutations.append(statistics.mean(
            (1 if i in positive else -1) * x for i, x in enumerate(chronological)))
    probability = sum(abs(x) >= abs(observed) - 1e-14 for x in permutations) / 252
    rng = random.Random(202611141914)
    boot = sorted(statistics.mean([rng.choice(strata[order])
                                  for order in (False, True) for _ in range(5)])
                  for _ in range(20000))

    def quantile(q):
        position = (len(boot) - 1) * q
        low = math.floor(position)
        return boot[low] * (1 - (position - low)) + boot[math.ceil(position)] * (position - low)

    return {
        "pairs": 10,
        "sampled_decode_change_pct": math.expm1(observed) * 100,
        "order_stratified_bootstrap95_pct": [math.expm1(quantile(q)) * 100 for q in (.025, .975)],
        "exact_balanced_order_two_sided_p": probability,
        "positive_pairs": sum(x > 0 for x in effects),
        "scope": "One fixed three-case sampled endpoint. Greedy is secondary; no pilot pooling or exclusions. Prefill flags remain unresolved.",
    }


if __name__ == "__main__":
    print(json.dumps(reproduce(Path(__file__).with_name("deferred-target-verification-pairs.csv")), indent=2))
