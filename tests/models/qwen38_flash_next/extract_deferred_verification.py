"""Compile actual state-method bodies in small CPU-only, simulated-HIP fixtures.

This is scoped state/order coverage, not an Executor/GPU integration test.
The GPU operator and two-node probe cover the separate math/transport contracts.
"""
from pathlib import Path
import argparse
import re


def extract(source: str, signature: str, strip_const: bool) -> str:
    if source.count(signature) != 1:
        raise ValueError(f"expected one method definition: {signature}")
    start = source.index(signature)
    body = source.index("{", start)
    # Ignore ordinary quoted literals and comments while balancing braces.
    tokens = re.compile(r'//[^\n]*|/\*[\s\S]*?\*/|"(?:\\.|[^"\\])*"|\'(?:\\.|[^\'\\])*\'|[{}]')
    depth = 0
    end = None
    for match in tokens.finditer(source, body):
        token = match.group()
        if token == "{":
            depth += 1
        elif token == "}":
            depth -= 1
            if depth == 0:
                end = match.end()
                break
    if end is None:
        raise ValueError(f"unclosed method: {signature}")
    declaration = source[start:body].replace("Executor::", "Fixture::")
    if strip_const:
        declaration = re.sub(r"\bconst\s*$", "", declaration)
    return declaration + source[body:end]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--mode", choices=("state", "hook"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    methods = ["bool Executor::FailPartialVerification(",
               "void Executor::ClearPartialVerification("]
    if args.mode == "state":
        methods.append("bool Executor::EnsureFullVerificationRows(")
    else:
        methods.insert(0, "bool Executor::RunLayerHook(")
    text = args.source.read_text(encoding="utf-8")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("\n\n".join(extract(text, name, args.mode == "hook")
                                      for name in methods) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
