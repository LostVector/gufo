#!/usr/bin/env python3
"""Trace Gufo against pinned, unmodified upstream PyTorch DFlash2 operators.

Consume --trace output from qwen_dflash_gpu_test or the Flash Next dflash_trace
probe. Weights come from the same GGUF artifact: this checks execution and
binding, not the GGUF converter or equivalence to an unquantized target
checkpoint. No model download is implicit. Optional BF16 modes are diagnostic
and do not alter the strict packed-weight/FP32 execution gate.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import importlib.metadata
import json
import os
import re
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
UPSTREAM_SHA256 = "f55b7fe0a4c0b3073e0f9cdce547cce29f4b8e2168c4d2818760007c43b7651e"
UPSTREAM_REVISION = "07ebd93db9f472af339b644bb70221ad8428328a"


def module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


class Weights:
    def __init__(self, path: Path, codec, device="cuda", *, include_tokenizer=False):
        self.path, self.codec, self.device = path, codec, device
        self.state = codec.parse_gguf(path, include_tokenizer=include_tokenizer)
        self.tensors = {t["name"]: t for t in self.state["tensors"]}

    def read(self, name: str) -> torch.Tensor:
        info = self.tensors[name]
        data = np.memmap(
            self.path, mode="r", dtype=np.uint8,
            offset=self.state["data_offset"] + info["offset"],
            shape=(self.codec.tensor_bytes(self.state, info),),
        )
        values = self.codec.DEQUANT[info["type"]](data).reshape(info["shape"])
        return torch.from_numpy(np.array(values, dtype=np.float32)).to(self.device)

    def rows(self, name: str, indices: list[int]) -> torch.Tensor:
        info = self.tensors[name]
        row_bytes = self.codec.tensor_bytes(self.state, info) // info["shape"][0]
        decoded = {}
        for index in set(indices):
            if not 0 <= index < info["shape"][0]:
                raise ValueError("embedding token is outside the target vocabulary")
            data = np.memmap(
                self.path, mode="r", dtype=np.uint8,
                offset=self.state["data_offset"] + info["offset"] + index * row_bytes,
                shape=(row_bytes,))
            decoded[index] = torch.from_numpy(
                np.array(self.codec.DEQUANT[info["type"]](data), dtype=np.float32)).to(self.device)
        return torch.stack([decoded[index] for index in indices])[None]


def weight_name(name: str) -> str:
    shared = {
        "fc.weight": "fc.weight",
        "hidden_norm.weight": "enc.output_norm.weight",
        "norm.weight": "output_norm.weight",
        "candidate_selector.predecessor_codebook.weight": "selector_predecessor.weight",
        "candidate_selector.successor_codebook.weight": "selector_successor.weight",
        "candidate_selector.hidden_projection.weight": "selector_hidden.weight",
    }
    if name in shared:
        return shared[name]
    _, index, suffix = name.split(".", 2)
    suffixes = {
        "input_layernorm.weight": "attn_norm.weight",
        "post_attention_layernorm.weight": "ffn_norm.weight",
        "self_attn.q_proj.weight": "attn_q.weight",
        "self_attn.k_proj.weight": "attn_k.weight",
        "self_attn.v_proj.weight": "attn_v.weight",
        "self_attn.o_proj.weight": "attn_output.weight",
        "self_attn.q_norm.weight": "attn_q_norm.weight",
        "self_attn.k_norm.weight": "attn_k_norm.weight",
        "mlp.gate_proj.weight": "ffn_gate.weight",
        "mlp.up_proj.weight": "ffn_up.weight",
        "mlp.down_proj.weight": "ffn_down.weight",
        "attention_conv.base_kernel": "attn_conv_base",
        "attention_conv.kernel_projection.weight": "attn_conv_proj.weight",
        "mlp_conv.base_kernel": "ffn_conv_base",
        "mlp_conv.kernel_projection.weight": "ffn_conv_proj.weight",
    }
    return f"blk.{index}.{suffixes[suffix]}"


def packed_bf16(name: str, gguf_type: int) -> bool:
    return (gguf_type in (0, 30) and not name.endswith("norm.weight")
            and "layernorm" not in name and "base_kernel" not in name) or (
        name == "fc.weight" or ".self_attn.k_proj." in name
        or ".self_attn.v_proj." in name or "_codebook." in name)


@torch.inference_mode()
def compare(args) -> dict:
    if hashlib.sha256(args.upstream.read_bytes()).hexdigest() != UPSTREAM_SHA256:
        raise ValueError("upstream model.py does not match the pinned source")
    upstream = module("dflash_upstream", args.upstream)
    codec = module("gufo_gguf", ROOT / "tools/gufo/gguf.py")
    device = args.device
    weights = Weights(args.draft, codec, device, include_tokenizer=True)
    meta = json.loads((args.trace / "trace.json").read_text())
    config = upstream.Qwen3Config(**json.loads(args.config.read_text()))
    config._attn_implementation = "sdpa"
    if list(config.dflash_config["target_layer_ids"]) != meta["target_layer_ids"]:
        raise ValueError("target feature taps differ from the original configuration")
    if config.dflash_config["selector_top_k"] != 16 or config.is_causal:
        raise ValueError("unexpected DFlash2 topology")
    with torch.device("meta"):
        model = upstream.DFlash2DraftModel(config)
    state = {}
    rounded = set()
    for name, parameter in model.state_dict().items():
        original = weight_name(name)
        value = weights.read(original).reshape(parameter.shape)
        if packed_bf16(name, weights.tensors[original]["type"]):
            value = value.bfloat16().float()
            rounded.add(name.removesuffix(".weight"))
        state[name] = value
    model.load_state_dict(state, strict=True, assign=True)
    model.rotary_emb = upstream.Qwen3RotaryEmbedding(config, device=device)
    model.eval()
    del state
    target = Weights(args.target, codec, device)
    if target.tensors["output.weight"]["type"] != int(meta["head_type"]):
        raise ValueError("trace uses a repacked target head; export that head before comparing")
    head = torch.nn.Linear(config.hidden_size, config.vocab_size, bias=False, device="meta")
    head.weight = torch.nn.Parameter(target.read("output.weight"))
    position, block = int(meta["position"]), int(meta["draft_count"]) + 1
    presence_penalty = float(meta.get("presence_penalty", 0))
    penalty_history = set(meta.get("penalty_history", []))
    policy_candidates, policy_row, policy_prefix = None, 0, 0
    original_sampling_probs = upstream._sampling_probs

    def policy_probs(scores, temperature, top_p=1.0, top_k=0):
        nonlocal policy_row
        ids = policy_candidates[:, policy_row:policy_row + 1]
        seen = penalty_history | set(meta["proposed"][:policy_prefix + policy_row])
        mask = torch.tensor(
            [token in seen for token in ids.reshape(-1).cpu().tolist()],
            dtype=torch.float32, device=device).reshape(ids.shape)
        policy_row += 1
        return original_sampling_probs(scores.float() - presence_penalty * mask,
                                       temperature, top_p, top_k)

    if presence_penalty:
        # Gufo policy extension, independently applied to the author's scores.
        # Candidate membership remains the original unpenalized unary top-k.
        upstream._sampling_probs = policy_probs
    target_rows = []
    if "target_sampling" in meta:
        target_sampling = meta["target_sampling"]
        for row in range(block - 1):
            raw = np.fromfile(args.trace / f"target_logits.{row}.f32", dtype="<f4")
            if raw.size != config.vocab_size or not np.isfinite(raw).all():
                raise ValueError(f"target row {row}: malformed logits")
            scores = torch.from_numpy(raw).to(device)
            seen = penalty_history | set(meta["proposed"][:row])
            if presence_penalty:
                scores[torch.tensor(sorted(seen), dtype=torch.long, device=device)] -= presence_penalty
            p = original_sampling_probs(scores[None, None], target_sampling["temperature"],
                top_p=target_sampling["top_p"], top_k=target_sampling["top_k"])[0, 0]
            target_rows.append(p)

    def trace_values(name: str, width: int):
        if name.startswith("context.0."):
            suffix = name.removeprefix("context.0.")
            chunks = []
            for path in args.trace.glob("context.*.f32"):
                match = re.fullmatch(r"context\.(\d+)\." + re.escape(suffix) + r"\.f32", path.name)
                if match:
                    chunks.append((int(match[1]), path))
            cursor, values = 0, []
            for offset, path in sorted(chunks):
                chunk = np.fromfile(path, dtype="<f4")
                if offset != cursor or chunk.size % width:
                    raise ValueError(f"{name}: incomplete/discontinuous context trace at {offset}")
                values.append(chunk)
                cursor += chunk.size // width
            if cursor != position:
                raise ValueError(f"{name}: context trace has {cursor} rows, expected {position}")
            return np.concatenate(values)
        return np.fromfile(args.trace / f"{name}.f32", dtype="<f4")

    def captured(name: str, width: int) -> torch.Tensor:
        values = trace_values(name, width)
        return torch.from_numpy(values.reshape(1, -1, width)).to(device)

    features = captured("target_features", config.hidden_size *
                        len(config.dflash_config["target_layer_ids"]))
    embedding = target.rows(
        "token_embd.weight",
        [int(meta["anchor"])] + [int(config.dflash_config["mask_token_id"])] * (block - 1))
    embedding *= float(upstream._draft_value(config, "input_embedding_scale", 1.0))
    records = {}
    mode = "packed"
    active_layer = 0

    def record(name, tensor):
        reference = tensor.detach().float().cpu().numpy().reshape(-1)
        width = reference.size // position if name.startswith("context.0.") else tensor.shape[-1]
        actual = trace_values(name, width)
        if actual.shape != reference.shape or not np.isfinite(reference).all():
            raise ValueError(f"{name}: Gufo shape={actual.shape}, reference shape={reference.shape}, "
                             f"reference finite={np.isfinite(reference).sum()}/{reference.size}")
        if not np.isfinite(actual).all():
            raise ValueError(f"{name}: nonfinite Gufo output")
        error = actual.astype(np.float64) - reference
        norm = float(np.sqrt(np.mean(reference.astype(np.float64) ** 2)))
        rmse = float(np.sqrt(np.mean(error ** 2)))
        records.setdefault(mode, {})[name] = {
            "max_abs": float(np.max(np.abs(error))), "rmse": rmse,
            "relative_rmse": rmse / max(norm, 1e-30),
        }
    record("embedding", embedding)

    # Match Gufo's documented packed-BF16 input rounding without changing any
    # upstream operator equations. Feature injection retains FP32 activations.
    for name, submodule in model.named_modules():
        if (meta.get("block_activation_dtype", "bf16") == "bf16"
                and isinstance(submodule, torch.nn.Linear) and name in rounded and name != "fc"):
            def round_input(_, inputs, name=name):
                if (".self_attn.k_proj" in name or ".self_attn.v_proj" in name):
                    if inputs[0].shape[1] == position:
                        return inputs
                return (inputs[0].bfloat16().float(),)
            submodule.register_forward_pre_hook(round_input)
    model.hidden_norm.register_forward_hook(
        lambda m, i, o: record("context.0.normalized", o))
    model.norm.register_forward_hook(lambda m, i, o: record("normalized", o))
    original_rope = upstream.apply_rotary_pos_emb

    def rope(*args, **kwargs):
        q, k = original_rope(*args, **kwargs)
        prefix = f"layer.{active_layer}."
        record(prefix + "q", q.transpose(1, 2))
        record(prefix + "k", k[:, :, position:].transpose(1, 2))
        record(f"context.0.{active_layer}.k", k[:, :, :position].transpose(1, 2))
        return q, k
    upstream.apply_rotary_pos_emb = rope

    for index, layer in enumerate(model.layers):
        prefix = f"layer.{index}."

        def enter(_, inputs, kwargs, index=index):
            nonlocal active_layer
            active_layer = index
            if mode == "local":
                kwargs["hidden_states"] = captured(
                    "embedding" if index == 0 else f"layer.{index - 1}.output",
                    config.hidden_size)
                kwargs["target_hidden"] = captured("context.0.normalized", config.hidden_size)
            return inputs, kwargs
        layer.register_forward_pre_hook(enter, with_kwargs=True)
        for name, stage in [
            ("input_layernorm", "attn_norm"),
            ("post_attention_layernorm", "ffn_norm"),
            ("self_attn.o_proj", "attn_output"),
            ("mlp.down_proj", "ffn_down"),
        ]:
            layer.get_submodule(name).register_forward_hook(
                lambda m, i, o, stage=prefix + stage: record(stage, o))
        layer.self_attn.o_proj.register_forward_pre_hook(
            lambda m, i, stage=prefix + "attention": record(stage, i[0]),
            prepend=True)
        layer.post_attention_layernorm.register_forward_pre_hook(
            lambda m, i, stage=prefix + "attn_residual": record(stage, i[0]))
        layer.self_attn.v_proj.register_forward_hook(
            lambda m, i, o, index=index: record(
                f"context.0.{index}.v" if o.shape[1] == position
                else f"layer.{index}.v", o))
        layer.register_forward_hook(
            lambda m, i, o, stage=prefix + "output": record(stage, o))
        for name, label in [("attention_conv", "attn"), ("mlp_conv", "ffn")]:
            conv = getattr(layer, name)
            original_prepare, original_finish = conv.prepare, conv.finish

            def prepare(x, original=original_prepare, stage=prefix + label + "_conv_in"):
                result = original(x)
                record(stage, result[0])
                return result

            def finish(x, dynamic, original=original_finish, stage=prefix + label + "_conv_out"):
                result = original(x, dynamic)
                record(stage, result)
                return result
            conv.prepare, conv.finish = prepare, finish

    outputs, output_logits = {}, {}
    for mode in ("packed", "local"):
        outputs[mode] = model(
            position_ids=torch.arange(position + block, device=device)[None],
            noise_embedding=embedding, target_hidden=features, use_cache=False,
        )
        record("selector_hidden", model.candidate_selector.hidden_projection(outputs[mode][:, 1:]))
        output_logits[mode] = model.compute_logits(outputs[mode][:, 1:], head)
        record("logits", output_logits[mode])
    mode = "head"
    record("logits", model.compute_logits(captured("normalized", config.hidden_size)[:, 1:], head))
    k = config.dflash_config["selector_top_k"]
    actual_ids = np.array(meta["candidates"], dtype=np.int64).reshape(-1, k)
    actual_q = np.fromfile(args.trace / "probabilities.f32", dtype="<f4").reshape(-1, k)
    uniforms = np.fromfile(args.trace / "uniforms.f32", dtype="<f4")
    selected_q = np.fromfile(args.trace / "selected_probabilities.f32", dtype="<f4")
    if (actual_q.shape != (block - 1, k) or uniforms.shape != (block - 1,)
            or selected_q.shape != (block - 1,) or not np.isfinite(actual_q).all()
            or (actual_q < 0).any()
            or not np.allclose(actual_q.sum(axis=1), 1.0, rtol=0, atol=1e-5)):
        raise ValueError("trace has malformed proposal probabilities")
    for row in range(block - 1):
        # Gufo samples in its own candidate order. Upstream topk is unsorted,
        # so a shared random seed need not produce the same categorical draw.
        choice = int(np.searchsorted(
            np.cumsum(actual_q[row], dtype=np.float64), uniforms[row], side="right"))
        if (choice >= k or int(actual_ids[row, choice]) != int(meta["proposed"][row])
                or abs(float(selected_q[row]) - float(actual_q[row, choice])) > 1e-7):
            raise ValueError(f"row {row}: token/confidence differs from the sampled distribution")
    upstream._sample_probs = lambda q: torch.argmax(q, dim=-1)
    full_tv, full_sets = [], 0
    for row in range(block - 1):
        predecessor = int(meta["anchor"] if row == 0 else meta["proposed"][row - 1])
        policy_candidates = torch.topk(output_logits["packed"][:, row:row + 1], k,
                                       dim=-1, sorted=False).indices
        policy_row, policy_prefix = 0, row
        _, ids, probs = model.candidate_selector.select(
            outputs["packed"][:, row + 1:row + 2],
            output_logits["packed"][:, row:row + 1],
            torch.tensor([predecessor], device=device), float(meta["temperature"]))
        expected = dict(zip(ids[0, 0].cpu().tolist(), probs[0, 0].cpu().tolist()))
        actual = dict(zip(actual_ids[row], actual_q[row]))
        full_sets += set(expected) == set(actual)
        full_tv.append(float(sum(abs(expected.get(token, 0) - actual.get(token, 0))
                                 for token in expected.keys() | actual.keys()) * 0.5))
    # Check the original selector independently of accumulated layer/GEMM
    # differences, using captured unary logits and projected hidden vectors.
    # Force its sampled predecessor path to Gufo's so every conditional q row
    # refers to exactly the same prefix. Candidate order is not a distribution.
    selector = model.candidate_selector
    original_hidden_projection = selector.hidden_projection
    selector.hidden_projection = torch.nn.Identity()
    logits = captured("logits", config.vocab_size)
    observed_logits = logits
    projected = captured("selector_hidden", config.dflash_config["selector_rank"])
    reference_candidates = torch.topk(logits, k, dim=-1, sorted=False).indices
    policy_candidates, policy_row, policy_prefix = reference_candidates, 0, 0
    draw = 0

    def forced_draw(q):
        nonlocal draw
        choice = (reference_candidates[:, draw] == int(meta["proposed"][draw])).nonzero()
        if choice.shape != (1, 2):
            raise ValueError(f"proposal {draw} is outside upstream unary top-k")
        draw += 1
        return choice[:, 1]
    upstream._sample_probs = forced_draw
    path, ids, probs = selector.select(
        projected, logits, torch.tensor([int(meta["anchor"])], device=device),
        float(meta["temperature"]),
    )
    max_error = 0.0
    for row in range(block - 1):
        expected = dict(zip(ids[0, row].cpu().tolist(), probs[0, row].cpu().tolist()))
        if len(set(actual_ids[row])) != k or set(actual_ids[row]) != set(expected):
            raise ValueError(f"row {row}: candidate set differs from upstream top-k")
        for token, probability in zip(actual_ids[row], actual_q[row]):
            max_error = max(max_error, float(abs(probability - expected[token])))
    if max_error > 5e-6 or draw != block - 1:
        raise ValueError(f"selector probability mismatch: {max_error}")
    report = {
        "upstream_revision": UPSTREAM_REVISION, "upstream_sha256": UPSTREAM_SHA256,
        "config_sha256": hashlib.sha256(args.config.read_bytes()).hexdigest(),
        "torch": torch.__version__, "device": device,
        "transformers": importlib.metadata.version("transformers"),
        "environment": "nix" if os.environ.get("IN_NIX_SHELL") else "external",
        "trace": str(args.trace),
        "target": str(args.target), "draft": str(args.draft),
        "target_layer_ids": meta["target_layer_ids"],
        "block_activation_dtype": meta.get("block_activation_dtype", "bf16"),
        "presence_penalty": presence_penalty,
        "selector": {"rows": draw, "top_k": k, "max_probability_error": max_error,
                     "uniform_draws_match": True,
                     "draw_scope": "exported Gufo q and candidate order; upstream order need not give identical same-seed draws"},
        "full_forward": {"same_candidate_sets": full_sets, "rows": block - 1,
                         "max_total_variation": max(full_tv),
                         "mean_total_variation": float(np.mean(full_tv))},
        "strict_contract": "same converted weights and supplied features; packed-weight FP32 execution",
        "stages": {name: dict(stages) for name, stages in records.items()},
    }
    # Numerical tolerances permit different FP32 reduction trees. They are not
    # learned from each run; a new discrepancy must be investigated.
    report["passed"] = (
        all(s["relative_rmse"] <= 1e-4 for stages in records.values() for s in stages.values())
        and records["packed"]["logits"]["max_abs"] <= 1e-3
        and full_sets == block - 1
        and max(full_tv) <= 1e-4
    )
    def conditional_delta(hidden, logits):
        nonlocal policy_candidates, policy_row, policy_prefix
        variation, fixed_variation, details, sets = [], [], [], 0
        for row in range(block - 1):
            predecessor = int(meta["anchor"] if row == 0 else meta["proposed"][row - 1])
            policy_candidates = torch.topk(logits[:, row:row + 1], k,
                                           dim=-1, sorted=False).indices
            policy_row, policy_prefix = 0, row
            _, ids, probs = selector.select(
                hidden[:, row + 1:row + 2], logits[:, row:row + 1],
                torch.tensor([predecessor], device=device), float(meta["temperature"]))
            expected = dict(zip(ids[0, 0].cpu().tolist(), probs[0, 0].cpu().tolist()))
            actual = dict(zip(actual_ids[row], actual_q[row]))
            sets += set(expected) == set(actual)
            distance = float(sum(abs(expected.get(token, 0) - actual.get(token, 0))
                                 for token in expected.keys() | actual.keys()) * .5)
            variation.append(distance)
            forced_ids = torch.tensor(actual_ids[row], device=device)[None, None]
            masked = torch.full_like(logits[:, row:row + 1], -torch.inf)
            masked.scatter_(-1, forced_ids, logits[:, row:row + 1].gather(-1, forced_ids))
            policy_candidates = torch.topk(masked, k, dim=-1, sorted=False).indices
            policy_row, policy_prefix = 0, row
            _, fixed_ids, fixed_probs = selector.select(
                hidden[:, row + 1:row + 2], masked,
                torch.tensor([predecessor], device=device), float(meta["temperature"]))
            fixed = dict(zip(fixed_ids[0, 0].cpu().tolist(), fixed_probs[0, 0].cpu().tolist()))
            fixed_distance = float(sum(abs(fixed.get(token, 0) - actual.get(token, 0))
                                       for token in fixed.keys() | actual.keys()) * .5)
            fixed_variation.append(fixed_distance)
            changed = []
            for token in sorted(actual.keys() ^ expected.keys()):
                baseline = observed_logits[0, row]
                alternative = logits[0, row]
                changed.append(dict(token=int(token),
                    baseline_rank=int((baseline > baseline[int(token)]).sum()) + 1,
                    alternative_rank=int((alternative > alternative[int(token)]).sum()) + 1,
                    baseline_logit=float(baseline[int(token)]),
                    alternative_logit=float(alternative[int(token)]),
                    baseline_q=float(actual.get(token, 0)),
                    alternative_q=float(expected.get(token, 0))))
            details.append(dict(row=row, predecessor=predecessor,
                total_variation=distance, fixed_candidate_total_variation=fixed_distance,
                changed_members=changed))
            if target_rows:
                target_p = target_rows[row]
                details[-1]["conditional_expected_acceptance"] = {
                    "baseline": float(sum(min(float(target_p[int(token)]), float(q))
                                          for token, q in actual.items())),
                    "alternative": float(sum(min(float(target_p[int(token)]), float(q))
                                             for token, q in expected.items())),
                    "fixed_candidates_alternative": float(sum(min(float(target_p[int(token)]), float(q))
                                             for token, q in fixed.items())),
                }
                for member in changed:
                    member["target_p"] = float(target_p[member["token"]])
        return dict(same_candidate_sets=sets, rows=block - 1,
                    max_total_variation=max(variation),
                    mean_total_variation=float(np.mean(variation)),
                    fixed_candidate_max_total_variation=max(fixed_variation),
                    conditional_rows=details, stages=records[mode])

    if args.bf16_diagnostic or args.alternative_draft:
        selector.hidden_projection = original_hidden_projection
        upstream._sample_probs = lambda q: torch.argmax(q, dim=-1)
    if args.bf16_diagnostic:
        # Deliberately diagnostic: these modes change the precision contract,
        # not the strict FP32 comparison or its pre-existing pass thresholds.
        # Keep RoPE frequencies in F32 as initialized by the author model.
        diagnostic = {}
        for mode in ("bf16_weights", "bf16_execution"):
            model.to(dtype=torch.bfloat16)
            head.to(dtype=torch.bfloat16)
            if mode == "bf16_weights":
                model.float()
                head.float()
            model.rotary_emb = upstream.Qwen3RotaryEmbedding(config, device=device)
            dtype = torch.float32 if mode == "bf16_weights" else torch.bfloat16
            hidden = model(
                position_ids=torch.arange(position + block, device=device)[None],
                noise_embedding=embedding.to(dtype), target_hidden=features.to(dtype),
                use_cache=False,
            )
            logits = model.compute_logits(hidden[:, 1:], head)
            record("logits", logits)
            diagnostic[mode] = conditional_delta(hidden, logits)
        report["precision_diagnostic"] = diagnostic
    if args.alternative_draft:
        mode = "alternative_draft_weights"
        alternative = Weights(args.alternative_draft, codec, device, include_tokenizer=True)
        original_metadata = {**weights.state["metadata"], **weights.state["user_kv"]}
        alternative_metadata = {**alternative.state["metadata"], **alternative.state["user_kv"]}
        config_keys = {key for key in original_metadata.keys() | alternative_metadata.keys()
                       if not key.startswith("general.") or key == "general.architecture"}
        mismatched = [key for key in config_keys
                      if original_metadata.get(key) != alternative_metadata.get(key)]
        if mismatched:
            raise ValueError(f"alternative draft changes configuration: {sorted(mismatched)}")
        model.float()
        state = {}
        for name, parameter in model.state_dict().items():
            original = weight_name(name)
            value = alternative.read(original).reshape(parameter.shape)
            if packed_bf16(name, alternative.tensors[original]["type"]):
                value = value.bfloat16().float()
            state[name] = value
        model.load_state_dict(state, strict=True, assign=True)
        model.rotary_emb = upstream.Qwen3RotaryEmbedding(config, device=device)
        head.weight = torch.nn.Parameter(target.read("output.weight"))
        hidden = model(position_ids=torch.arange(position + block, device=device)[None],
                       noise_embedding=embedding, target_hidden=features, use_cache=False)
        logits = model.compute_logits(hidden[:, 1:], head)
        record("logits", logits)
        report["draft_quantization_diagnostic"] = {
            "alternative_draft": str(args.alternative_draft),
            "contract": "alternative converted draft weights; same supplied features, donor tables and FP32 execution",
            **conditional_delta(hidden, logits),
        }
        if args.alternative_draft.resolve() == args.draft.resolve():
            diagnostic = report["draft_quantization_diagnostic"]
            diagnostic["same_weight_control_passed"] = (
                diagnostic["same_candidate_sets"] == block - 1
                and diagnostic["max_total_variation"] <= 1e-4)
            if not diagnostic["same_weight_control_passed"]:
                raise ValueError("identical-weight reload control exceeded strict q tolerances")
        if target_rows:
            report["draft_quantization_diagnostic"]["target_sampling"] = meta["target_sampling"]
            report["draft_quantization_diagnostic"]["target_comparison_scope"] = (
                "sum min(p,q) at identical observed prefixes, not a trajectory-averaged acceptance rate")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("upstream", "config", "target", "draft", "trace", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--external-env", action="store_true",
                        help="Explicitly use a recorded environment outside nix develop")
    parser.add_argument("--bf16-diagnostic", action="store_true",
                        help="Also measure weight-only and full BF16 drift without changing FP32 pass criteria")
    parser.add_argument("--alternative-draft", type=Path,
                        help="Diagnose alternative GGUF draft weights while holding features, donor tables and FP32 execution fixed")
    args = parser.parse_args()
    if not os.environ.get("IN_NIX_SHELL") and not args.external_env:
        parser.error("run inside nix develop")
    torch.set_num_threads(8)
    torch.set_float32_matmul_precision("highest")
    report = compare(args)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report["selector"]))
    print(json.dumps(report["full_forward"]))
    if "precision_diagnostic" in report:
        print("precision_diagnostic", json.dumps({
            mode: {key: value for key, value in result.items() if key != "stages"}
            for mode, result in report["precision_diagnostic"].items()}))
    for mode, stages in report["stages"].items():
        worst = sorted(stages.items(), key=lambda row: row[1]["relative_rmse"], reverse=True)[:8]
        print(mode, json.dumps(dict(worst)))
    if not report["passed"]:
        raise SystemExit("upstream comparison exceeded fixed FP32 tolerances; see report")


if __name__ == "__main__":
    main()
