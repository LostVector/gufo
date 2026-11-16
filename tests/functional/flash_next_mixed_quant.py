#!/usr/bin/env python3
"""Qualify native MTP, copied-token verification and cache history on Flash Next.

Use the original ISTA-DASLab GSQ-RCO IQ3_S artifact for mixed-format coverage;
UD-Q4_K_XL is an unaffected-format control. This is an opt-in running-server
regression, not a missing-model skip or a model intelligence benchmark.
"""
import argparse
import json
from pathlib import Path
import time
import urllib.error
import urllib.request


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("base_url")
    parser.add_argument("output", type=Path)
    parser.add_argument("--model")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    base = args.base_url.rstrip("/")
    with urllib.request.urlopen(base + "/v1/models", timeout=15) as response:
        models = json.load(response)
    (args.output / "models.json").write_text(json.dumps(models, indent=2) + "\n")
    if args.model:
        model = args.model
    else:
        assert len(models["data"]) == 1, "Select --model on a multi-model server"
        model = models["data"][0]["id"]
    fixture = Path(__file__).parent / "fixtures/flash-next-mixed-quant"
    prefix = json.loads((fixture / "prefix.json").read_text())
    continuation = json.loads((fixture / "continuation.json").read_text())
    replies = []
    cases = [("prefix-cold", prefix, False), ("prefix-warm", prefix, True),
             ("prefix-repeat", prefix, True),
             ("continuation-cached", continuation, True),
             ("continuation-cold", continuation, False)]
    for name, template, cache in cases:
        body = {**template, "model": model, "cache_prompt": cache}
        (args.output / (name + "-request.json")).write_text(json.dumps(body, indent=2) + "\n")
        request = urllib.request.Request(base + "/v1/chat/completions",
                                         data=json.dumps(body).encode(),
                                         headers={"Content-Type": "application/json"})
        started = time.monotonic()
        try:
            with urllib.request.urlopen(request, timeout=180) as response:
                reply = json.load(response)
        except urllib.error.HTTPError as error:
            (args.output / (name + "-http-error.txt")).write_bytes(error.read())
            raise
        (args.output / (name + "-response.json")).write_text(json.dumps(reply, indent=2) + "\n")
        assert reply["choices"][0]["message"]["content"], name
        assert reply["timings"]["draft_n"] > 0, "Native draft execution was not exercised"
        print(name, "wall_seconds=", time.monotonic() - started,
              "timings=", reply["timings"], flush=True)
        replies.append(reply)
    assert replies[0]["choices"] == replies[1]["choices"] == replies[2]["choices"], "Prefix replay changed"
    assert replies[1]["timings"]["cache_n"] > 0, "Expected actual prompt reuse"
    assert replies[3]["timings"]["cache_n"] > 0, "Expected actual continuation prefix reuse"
    assert replies[3]["choices"] == replies[4]["choices"], "Continuation changed under cache"
    proposed = replies[3]["timings"].get("lookup_n", 0)
    assert proposed > 8, "Copied-token workload was not exercised"
    quality = dict(prefix_repeat_equal=True, continuation_cached_uncached_equal=True,
                   lookup_proposed=proposed,
                   lookup_accepted=replies[3]["timings"].get("lookup_n_accepted", 0),
                   scope="Pinned request that exposed TP2 wide-copy bounds and stride failures; exact output checks, no retries")
    (args.output / "quality.json").write_text(json.dumps(quality, indent=2) + "\n")
    print("PASS Flash Next native MTP mixed-format continuation and copied-token regression")


if __name__ == "__main__":
    main()
