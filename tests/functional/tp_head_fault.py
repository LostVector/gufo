#!/usr/bin/env python3
"""Prepare or verify an exact target-head copy fault on a private TP2 server.

This tool does not load a driver or start a cluster. Run the generated launcher
on the selected rank, using the ordinary two-host server arguments. It arms one
simulated HIP copy API error or deliberate process exit. Use a fresh output
folder for every run; never point it at a production server.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess


def prepare(args):
    binary = args.binary.resolve(strict=True)
    hip = args.hip.resolve(strict=True)
    source = Path(__file__).with_name("tp_head_copy_fault.cpp")
    rows = [line.split() for line in subprocess.check_output(
        ["nm", "-S", "--defined-only", str(binary)], text=True).splitlines()]
    rows = [row for row in rows if len(row) == 4 and row[2] in ("t", "T")]
    names = subprocess.check_output(
        ["c++filt"], input="\n".join(row[3] for row in rows) + "\n",
        text=True).splitlines()
    found = [(row, name) for row, name in zip(rows, names)
             if "Executor::TwoRankTargetHeadGather(" in name and "::_M_invoke(" in name]
    if len(found) != 1:
        raise RuntimeError("requires one unstripped target-head callback symbol")
    row, name = found[0]
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    library = output / "copy-fault.so"
    compiler = args.cxx or str(hip / "bin" / "amdclang++")
    command = [compiler, "-std=c++20", "-O2", "-D__HIP_PLATFORM_AMD__",
               "-I" + str(hip / "include"), "-fPIC", "-shared", str(source),
               "-ldl", "-o", str(library)]
    subprocess.run(command, check=True)
    target = {"binary": str(binary), "binary_sha256": hashlib.sha256(binary.read_bytes()).hexdigest(),
              "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
              "library_sha256": hashlib.sha256(library.read_bytes()).hexdigest(),
              "begin": int(row[0], 16), "end": int(row[0], 16) + int(row[1], 16),
              "symbol": row[3], "demangled": name, "rank": args.rank, "mode": args.mode,
              "width": 496640, "height": 2, "source_pitch": 496640,
              "destination_pitch": 993280, "compile_command": command}
    (output / "target.json").write_text(json.dumps(target, indent=2) + "\n")
    # The preload is confined to exec of the verified binary, not the shell.
    launcher = '''#!/usr/bin/env python3
from pathlib import Path
import hashlib,json,os,sys
out=Path(__file__).resolve().parent
t=json.loads((out/'target.json').read_text())
binary=Path(t['binary']);library=out/'copy-fault.so';args=sys.argv[1:]
assert hashlib.sha256(binary.read_bytes()).hexdigest()==t['binary_sha256']
assert hashlib.sha256(library.read_bytes()).hexdigest()==t['library_sha256']
assert args[:2]==['serve','llm'],'supply ordinary gufo serve llm arguments'
for key,value in [('--tp-world-size','2'),('--tp-rank',str(t['rank']))]:
 assert args.count(key)==1 and args[args.index(key)+1]==value,'wrong TP role'
assert not(out/'owner.json').exists()and not(out/'armed').exists()and not(out/'injected.jsonl').exists(),'use a fresh prepared directory'
p=Path('/proc')/str(os.getpid())
(out/'owner.json').write_text(json.dumps(dict(pid=os.getpid(),start_ticks=(p/'stat').read_text().rsplit(')',1)[1].split()[19],rank=t['rank'])))
(out/'armed').touch(exist_ok=False)
env=dict(os.environ,LD_PRELOAD=str(library),STRIX_COPY_ARM=str(out/'armed'),STRIX_COPY_LOG=str(out/'injected.jsonl'),STRIX_COPY_MODE=t['mode'],STRIX_COPY_BEGIN=str(t['begin']),STRIX_COPY_END=str(t['end']))
os.execve(str(binary),[str(binary),*args],env)
'''
    path = output / "launch.py"
    path.write_text(launcher)
    path.chmod(0o755)
    print(json.dumps({"launcher": str(path), "rank": args.rank, "mode": args.mode}))


def verify(args):
    output = args.output.resolve(strict=True)
    target = json.loads((output / "target.json").read_text())
    owner = json.loads((output / "owner.json").read_text())
    rows = [json.loads(line) for line in (output / "injected.jsonl").read_text().splitlines()
            if line.strip()]
    if len(rows) != 1:
        raise RuntimeError("expected exactly one injected fault")
    row = rows[0]
    assert row["pid"] == owner["pid"] and owner["rank"] == target["rank"]
    assert row["event"] == target["mode"]
    assert target["begin"] <= row["caller_ip"] - row["image_base"] < target["end"]
    assert all(row[key] == target[key] for key in
               ("width", "height", "source_pitch", "destination_pitch"))
    result = {"exact_fault_verified": True, "pid": row["pid"], "mode": row["event"]}
    if args.response:
        events = []
        for line in args.response.read_text().splitlines():
            if line.startswith("data:") and line[5:].strip() != "[DONE]":
                events.append(json.loads(line[5:].strip()))
        errors = [event["error"] for event in events if event.get("error")]
        normal = any(choice.get("finish_reason") is not None
                     for event in events for choice in event.get("choices", []))
        assert errors and not normal, "require an explicit non-normal stream response"
        assert not any(word in json.dumps(errors).lower() for word in ("timeout", "timed out"))
        result["explicit_non_timeout_stream_error"] = errors
    print(json.dumps(result, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    prep = commands.add_parser("prepare")
    prep.add_argument("--binary", type=Path, required=True)
    prep.add_argument("--hip", type=Path, default=Path(os.environ.get("HIP_PATH", "/opt/rocm")))
    prep.add_argument("--cxx")
    prep.add_argument("--output", type=Path, required=True)
    prep.add_argument("--rank", type=int, choices=(0, 1), required=True)
    prep.add_argument("--mode", choices=("copy-error", "peer-exit"), default="copy-error")
    prep.set_defaults(function=prepare)
    check = commands.add_parser("verify")
    check.add_argument("--output", type=Path, required=True)
    check.add_argument("--response", type=Path)
    check.set_defaults(function=verify)
    args = parser.parse_args()
    args.function(args)


if __name__ == "__main__":
    main()
