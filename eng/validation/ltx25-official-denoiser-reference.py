#!/usr/bin/env python3
"""Compare a TensorSharp denoiser probe with unchanged official LTX CPU PyTorch.

Uses deterministic probe inputs or an integrity-checked real forward capture. GGUF weights are
independently dequantized by gguf-py. Full or bounded diagnostic depth is a
numerical component reference, not rendered-quality or GPU performance evidence.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
from importlib.metadata import PackageNotFoundError, version
import json
import math
from pathlib import Path
import resource
import struct
import subprocess
import sys
import threading
import time


def load_captured_inputs(probe, probe_path):
    """Validate captured Float32 arrays before importing torch or loading weights.

    Relative array paths are relative to the manifest. Null optional fields retain
    their actual denoiser semantics; no positions/masks are inferred from geometry.
    """
    if probe.get("input_mode") != "captured":
        raise ValueError("Expected input_mode=captured")
    if probe.get("use_reference_lora", False) is not False:
        raise ValueError("Captured Ingredients/LoRA forwards are not supported by this oracle")
    config, tokens = probe["config"], probe["tokens"]
    def positive(value):
        if type(value) is not int or value < 1:
            raise ValueError("Captured dimensions must be positive integers")
        return value
    v, a = positive(tokens["video"]), positive(tokens["audio"])
    vc = positive(tokens.get("video_context", tokens.get("context")))
    ac = positive(tokens.get("audio_context", tokens.get("context")))
    dimensions = {name: positive(config[name]) for name in
                  ("VideoChannels", "AudioChannels", "VideoContextDim", "AudioContextDim")}
    shapes = dict(video=[v, dimensions["VideoChannels"]], audio=[a, dimensions["AudioChannels"]],
                  video_context=[vc, dimensions["VideoContextDim"]], audio_context=[ac, dimensions["AudioContextDim"]],
                  video_positions=[v, 3, 2], audio_positions=[a, 1, 2],
                  video_timesteps=[v], audio_timesteps=[a], keyframe_mask=[v],
                  video_denoise_mask=[v], clean_video=[v, dimensions["VideoChannels"]])
    optional = {"video_timesteps", "audio_timesteps", "keyframe_mask", "video_denoise_mask", "clean_video"}
    specs = probe["inputs"]
    if set(specs) - set(shapes):
        raise ValueError("Unsupported captured input fields: " + ", ".join(sorted(set(specs) - set(shapes))))
    result = {}
    for name, expected_shape in shapes.items():
        spec = specs.get(name)
        if spec is None:
            if name not in optional:
                raise ValueError("Missing captured input: " + name)
            result[name] = None
            continue
        shape = spec.get("shape")
        if (not isinstance(shape, list) or any(type(value) is not int or value < 1 for value in shape) or
                shape != expected_shape and not (name.endswith("_timesteps") and shape == [1])):
            raise ValueError("Captured input shape differs: " + name)
        if spec.get("dtype") != "float32-le":
            raise ValueError("Captured input must use float32-le: " + name)
        path = Path(spec["path"])
        if not path.is_absolute():
            path = Path(probe_path).parent / path
        if path.stat().st_size != math.prod(shape) * 4:
            raise ValueError("Captured input byte length differs: " + name)
        payload = path.read_bytes()
        if hashlib.sha256(payload).hexdigest() != spec.get("sha256"):
            raise ValueError("Captured input SHA256 differs: " + name)
        previous = None
        for index, (value,) in enumerate(struct.iter_unpack("<f", payload)):
            if not math.isfinite(value):
                raise ValueError("Captured input is nonfinite: " + name)
            if (name.endswith("_timesteps") or name == "video_denoise_mask") and not 0 <= value <= 1:
                raise ValueError("Captured timestep/mask is outside [0,1]: " + name)
            if name.endswith("_positions"):
                if index % 2 and value < previous:
                    raise ValueError("Captured position boundaries are reversed: " + name)
                previous = value
        result[name] = payload
    for name in ("video_sigma", "audio_sigma"):
        value = probe.get(name)
        if type(value) not in (float, int) or not math.isfinite(value) or not 0 <= value <= 1:
            raise ValueError("Captured sigma must be finite [0,1]: " + name)
    return result


def captured_modalities(probe, payloads, np, torch, modality):
    """Map exact capture storage to the unchanged official Modality contract."""
    def tensor(name):
        data = payloads[name]
        return None if data is None else torch.from_numpy(np.frombuffer(data, dtype="<f4").copy())
    result = []
    for name, axes in (("video", 3), ("audio", 1)):
        tokens = probe["tokens"][name]
        context_tokens = probe["tokens"].get(name + "_context", probe["tokens"].get("context"))
        sigma = probe[name + "_sigma"]
        timesteps = tensor(name + "_timesteps")
        if timesteps is None:
            timesteps = torch.full((1, tokens), sigma, dtype=torch.float32)
        else:
            timesteps = timesteps.reshape(1, -1).expand(1, tokens)
        keyframes = tensor("keyframe_mask") if name == "video" else None
        result.append(modality(
            latent=tensor(name).reshape(1, tokens, -1), sigma=torch.tensor([sigma], dtype=torch.float32),
            timesteps=timesteps,
            positions=tensor(name + "_positions").reshape(1, tokens, axes, 2).transpose(1, 2),
            context=tensor(name + "_context").reshape(1, context_tokens, -1),
            keyframes_mask=keyframes.reshape(1, tokens, 1) if keyframes is not None else None))
    return tuple(result)


def validate_captured_outputs(probe):
    """Reject incomplete/tampered native velocity captures before the costly oracle."""
    for name, channels in (("video", "VideoChannels"), ("audio", "AudioChannels")):
        path = Path(probe["outputs"][name])
        if not path.is_absolute():
            raise ValueError("Captured output paths must be absolute")
        if path.stat().st_size != probe["tokens"][name] * probe["config"][channels] * 4:
            raise ValueError("Captured output byte length differs: " + name)
        payload = path.read_bytes()
        if hashlib.sha256(payload).hexdigest() != probe["output_sha256"][name]:
            raise ValueError("Captured output SHA256 differs: " + name)
        if any(not math.isfinite(value) for (value,) in struct.iter_unpack("<f", payload)):
            raise ValueError("Captured output is nonfinite: " + name)


def memory_constraints():
    """Read actual cgroup limits; host RAM is not a container residency budget."""
    paths = ["/sys/fs/cgroup/memory/memory.limit_in_bytes", "/sys/fs/cgroup/memory.max"]
    observed = {path: Path(path).read_text().strip() for path in paths if Path(path).is_file()}
    limits = [int(value) for value in observed.values() if value.isdigit() and int(value) < 2**60]
    return dict(observed=observed, effective_limit_bytes=min(limits) if limits else None)


def load_reference_weights(model, tensors, prefix, gguf, np, torch, stream_blocks=False, dequant_workers=1):
    """Attach parameter-loading hooks without changing the original forward code."""
    if not 1 <= dequant_workers <= 32:
        raise ValueError("--dequant-workers must be between 1 and 32")
    stats = dict(dequantized_tensor_types={}, dequantize_task_seconds=0.0, streamed_blocks=[], dequant_workers=dequant_workers)
    stats_lock = threading.Lock()

    def decode(name, shape):
        started = time.perf_counter()
        tensor = tensors[prefix + name]
        array = gguf.dequantize(tensor.data, tensor.tensor_type).reshape(tuple(shape))
        result = torch.from_numpy(np.array(array, dtype=np.float32, copy=True))
        with stats_lock:
            kinds = stats["dequantized_tensor_types"]
            kinds[str(tensor.tensor_type)] = kinds.get(str(tensor.tensor_type), 0) + 1
            stats["dequantize_task_seconds"] += time.perf_counter() - started
        return result

    def decoded_state(shapes):
        def item(pair):
            name, shape = pair
            return name, decode(name, shape)
        if dequant_workers == 1:
            return dict(item(pair) for pair in shapes.items())
        with ThreadPoolExecutor(max_workers=dequant_workers) as pool:
            return dict(pool.map(item, shapes.items()))

    state_shapes = {name: value.shape for name, value in model.state_dict().items()}
    stats["full_dequantized_parameter_bytes"] = sum(int(np.prod(shape)) * 4 for shape in state_shapes.values())
    if not stream_blocks:
        state = decoded_state(state_shapes)
        model.load_state_dict(state, assign=True, strict=True)
        return stats

    persistent = decoded_state({name: shape for name, shape in state_shapes.items()
                                if not name.startswith("transformer_blocks.")})
    expected_missing = {name for name in state_shapes if name.startswith("transformer_blocks.")}
    loaded = model.load_state_dict(persistent, assign=True, strict=False)
    if set(loaded.missing_keys) != expected_missing or loaded.unexpected_keys:
        raise ValueError("Streamed reference did not load exactly the non-block parameters")
    persistent_bytes = sum(int(np.prod(shape)) * 4 for name, shape in state_shapes.items() if name not in expected_missing)
    block_bytes = [sum(parameter.numel() * 4 for parameter in block.parameters()) for block in model.transformer_blocks]
    stats["estimated_dequantized_residency_bytes"] = persistent_bytes + max(block_bytes, default=0)

    def hooks(index):
        block_prefix = "transformer_blocks." + str(index) + "."

        def before(module, _args):
            state = decoded_state({block_prefix + name: parameter.shape for name, parameter in module.state_dict().items()})
            state = {name.removeprefix(block_prefix): value for name, value in state.items()}
            module.load_state_dict(state, assign=True, strict=True)
            print(f"Loaded original transformer block {index + 1}/{len(model.transformer_blocks)}", file=sys.stderr, flush=True)

        def after(module, _args, _output):
            # Inference outputs contain no autograd graph retaining weights. Move
            # only this completed block's parameters back to meta before the next
            # block loads; its original module forward implementation is unchanged.
            module.to_empty(device="meta")
            stats["streamed_blocks"].append(index)

        return before, after

    for index, block in enumerate(model.transformer_blocks):
        before, after = hooks(index)
        block.register_forward_pre_hook(before)
        block.register_forward_hook(after, always_call=True)
    return stats


def validate_probe(probe, max_blocks=4):
    """Refuse ambiguous reports before importing/allocating the model runtime."""
    if probe.get("completed") is not True:
        raise ValueError("The TensorSharp probe did not complete")
    if probe.get("schedule") != "forward" or probe.get("output_kind") != "velocity":
        raise ValueError("Official forward oracle requires a forward/velocity probe; sampled latents are not velocities")
    if probe.get("frequency_precision") not in ("Float32", "Float64"):
        raise ValueError("Probe must explicitly record frequency_precision; rerun legacy probes")
    blocks = probe.get("executed_blocks")
    if type(max_blocks) is not int or not 1 <= max_blocks <= 128:
        raise ValueError("--max-blocks must be between 1 and 128")
    if type(blocks) is not int or not 1 <= blocks <= max_blocks:
        raise ValueError("Probe depth exceeds the explicit CPU oracle --max-blocks residency limit")
    return blocks


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--probe", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--max-blocks", type=int, default=4,
                        help="Maximum transformer depth; the pinned 48-block denoiser has over 70 GiB of Float32 parameters")
    parser.add_argument("--stream-block-weights", action="store_true",
                        help="Keep original forward code and load/evict one block's parameters through standard module hooks")
    parser.add_argument("--dequant-workers", type=int, default=1,
                        help="Independent gguf-py tensor dequantization jobs, 1..32; more workers use extra temporary RAM")
    args = parser.parse_args()
    if args.output.exists() or any(Path(str(args.output) + "." + stream + ".f32").exists() for stream in ("video", "audio")):
        raise SystemExit("Refusing to replace existing oracle evidence")
    probe = json.loads(args.probe.read_text())
    blocks = validate_probe(probe, args.max_blocks)
    input_mode = probe.get("input_mode", "synthetic")
    if input_mode not in ("synthetic", "captured"):
        raise SystemExit("Unknown input_mode: " + str(input_mode))
    captured = load_captured_inputs(probe, args.probe) if input_mode == "captured" else None
    if captured is not None:
        validate_captured_outputs(probe)
    if args.threads < 1:
        raise SystemExit("--threads must be positive")
    probe_sha256 = hashlib.sha256(args.probe.read_bytes()).hexdigest()
    checkpoint_sha256 = probe.get("model_sha256")
    if not checkpoint_sha256:
        raise SystemExit("Probe must record model_sha256; rerun legacy probes")
    digest = hashlib.sha256()
    with Path(probe["model"]).open("rb") as stream:
        for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    if digest.hexdigest() != checkpoint_sha256:
        raise SystemExit("Checkpoint identity differs from completed probe")
    revision = subprocess.check_output(["git", "-C", str(args.reference), "rev-parse", "HEAD"], text=True).strip()
    if revision != "be6d6d0a69d11a0a399c462ed92606692e5f8697":
        raise SystemExit("Unexpected official LTX revision: " + revision)
    if subprocess.check_output(["git", "-C", str(args.reference), "status", "--porcelain"], text=True).strip():
        raise SystemExit("Official LTX source must be unchanged")
    sys.path.insert(0, str(args.reference.resolve() / "packages/ltx-core/src"))
    import numpy as np
    import torch
    import gguf
    from ltx_core.model.transformer.attention import AttentionOps, PytorchAttention
    from ltx_core.model.transformer.model import LTXModel
    from ltx_core.model.transformer.modality import Modality

    config = probe["config"]
    torch.set_num_threads(args.threads)
    attention = PytorchAttention([torch.nn.attention.SDPBackend.MATH])
    reader = gguf.GGUFReader(probe["model"])
    # Read the multipliers as LTXModelConfigurator.from_metadata does. The keyword
    # default av_ca_timestep_scale_multiplier=1 differs from the published 1000.
    field = reader.fields.get("config")
    checkpoint_config = json.loads(bytes(field.parts[field.data[0]]).decode()) if field is not None else {}
    transformer_config = checkpoint_config.get("transformer", {})
    timestep_scale = transformer_config.get("timestep_scale_multiplier", 1000)
    av_ca_timestep_scale = transformer_config.get("av_ca_timestep_scale_multiplier", 1)
    with torch.device("meta"):
        model = LTXModel(num_attention_heads=config["VideoHeads"], attention_head_dim=config["VideoHeadDim"],
                         in_channels=config["VideoChannels"], out_channels=config["VideoChannels"], num_layers=blocks,
                         cross_attention_dim=config["VideoContextDim"], audio_num_attention_heads=config["AudioHeads"],
                         audio_attention_head_dim=config["AudioHeadDim"], audio_in_channels=config["AudioChannels"],
                         audio_out_channels=config["AudioChannels"], audio_cross_attention_dim=config["AudioContextDim"],
                         double_precision_rope=probe["frequency_precision"] == "Float64",
                         apply_gated_attention=True, cross_attention_adaln=True,
                         ff_bias=config["VideoFfnBias"], audio_ff_bias=config["AudioFfnBias"],
                         use_keyframes_abs_pos_embedding=config["KeyframeEmbedding"],
                         timestep_scale_multiplier=timestep_scale,
                         av_ca_timestep_scale_multiplier=av_ca_timestep_scale,
                         attention_ops=AttentionOps(attention, attention)).eval()
    tensors = {tensor.name: tensor for tensor in reader.tensors}
    prefix = config["Prefix"]
    constraints = memory_constraints()
    estimated_full_bytes = sum(parameter.numel() * 4 for parameter in model.parameters())
    if not args.stream_block_weights and constraints["effective_limit_bytes"] is not None and estimated_full_bytes >= constraints["effective_limit_bytes"]:
        raise SystemExit("Full Float32 parameters exceed the measured cgroup limit; use --stream-block-weights")
    start = time.perf_counter()
    weight_stats = load_reference_weights(model, tensors, prefix, gguf, np, torch, args.stream_block_weights, args.dequant_workers)
    load_seconds = time.perf_counter() - start

    def values(count, seed):
        result = np.empty(count, dtype=np.float32)
        for i in range(count):
            seed ^= (seed << 13) & 0xffffffff
            seed ^= seed >> 17
            seed ^= (seed << 5) & 0xffffffff
            result[i] = (np.float32(seed >> 8) * np.float32(1 / 16777216) - np.float32(.5)) * np.float32(.2)
        return result

    if captured is not None:
        vm, am = captured_modalities(probe, captured, np, torch, Modality)
    else:
        tokens = probe["tokens"]
        v, a, c = tokens["video"], tokens["audio"], tokens["context"]
        arrays = {"video": values(v * config["VideoChannels"], 1), "audio": values(a * config["AudioChannels"], 2),
                  "video_context": values(c * config["VideoContextDim"], 3), "audio_context": values(c * config["AudioContextDim"], 4)}
        for name, array in arrays.items():
            digest = hashlib.sha256(array.tobytes()).hexdigest()
            if digest != probe["input_sha256"][name]:
                raise SystemExit("Deterministic input SHA256 differs for " + name)
        width, height, frames = probe["geometry"]
        ft, ht, wt = (frames - 1) // 8 + 1, height // 32, width // 32
        video_positions = []
        for t in range(ft):
            for h in range(ht):
                for w in range(wt):
                    video_positions.append([np.float32(max(0, t * 8 - 7)) / np.float32(24),
                                            np.float32(max(0, (t + 1) * 8 - 7)) / np.float32(24),
                                            h * 32, (h + 1) * 32, w * 32, (w + 1) * 32])
        audio_positions = [[np.float32(max(0, t * 4 - 3)) * np.float32(160) / np.float32(16000),
                            np.float32(max(0, (t + 1) * 4 - 3)) * np.float32(160) / np.float32(16000)] for t in range(a)]
        vp = torch.tensor(video_positions, dtype=torch.float32).reshape(1, v, 3, 2).transpose(1, 2)
        ap = torch.tensor(audio_positions, dtype=torch.float32).reshape(1, a, 1, 2).transpose(1, 2)
        sigma = probe["sigma"]
        vm = Modality(torch.from_numpy(arrays["video"]).reshape(1, v, -1), torch.tensor([sigma]),
                      torch.full((1, v), sigma), vp, torch.from_numpy(arrays["video_context"]).reshape(1, c, -1))
        am = Modality(torch.from_numpy(arrays["audio"]).reshape(1, a, -1), torch.tensor([sigma]),
                      torch.full((1, a), sigma), ap, torch.from_numpy(arrays["audio_context"]).reshape(1, c, -1))
    start = time.perf_counter()
    with torch.inference_mode():
        outputs = model(vm, am, None)
    if args.stream_block_weights and weight_stats["streamed_blocks"] != list(range(blocks)):
        raise SystemExit("Streamed reference did not execute every original transformer block in order")
    elapsed = time.perf_counter() - start
    comparisons = {}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    for name, output in zip(("video", "audio"), outputs):
        expected = output.detach().numpy().flatten()
        if hashlib.sha256(Path(probe["outputs"][name]).read_bytes()).hexdigest() != probe["output_sha256"][name]:
            raise SystemExit("Output identity differs from completed probe: " + name)
        actual = np.fromfile(probe["outputs"][name], dtype="<f4")
        if len(actual) != len(expected) or not np.isfinite(actual).all() or not np.isfinite(expected).all():
            raise SystemExit("Invalid output arrays: " + name)
        difference = actual.astype(np.float64) - expected.astype(np.float64)
        norm = np.linalg.norm(expected.astype(np.float64))
        comparisons[name] = dict(elements=len(actual), max_abs=float(np.abs(difference).max()),
                                 rmse=float(np.sqrt(np.mean(difference ** 2))),
                                 relative_l2=float(np.linalg.norm(difference) / max(norm, 1e-30)),
                                 cosine=float(np.dot(actual.astype(np.float64), expected.astype(np.float64)) /
                                              max(np.linalg.norm(actual.astype(np.float64)) * norm, 1e-30)))
        expected.tofile(str(args.output) + "." + name + ".f32")
    try:
        gguf_version = version("gguf")
    except PackageNotFoundError:
        gguf_version = "unavailable"
    full_depth = blocks == config["Layers"]
    result = dict(completed=True, reference_revision=revision, torch=torch.__version__, gguf=gguf_version,
                  checkpoint=probe["model"], checkpoint_sha256=checkpoint_sha256, probe=str(args.probe),
                  probe_sha256=probe_sha256, frequency_precision=probe["frequency_precision"],
                  probe_backend=probe.get("backend", "unrecorded"), probe_environment=probe.get("environment", {}),
                  reference_device="cpu", reference_threads=args.threads,
                  input_mode=input_mode, tokens=probe["tokens"],
                  captured_input_sha256={name: spec["sha256"] if spec is not None else None
                                         for name, spec in probe["inputs"].items()} if captured is not None else None,
                  captured_input_bytes=sum(len(payload) for payload in captured.values() if payload is not None) if captured is not None else None,
                  video_sigma=probe.get("video_sigma", probe.get("sigma")),
                  audio_sigma=probe.get("audio_sigma", probe.get("sigma")),
                  tool_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                  executed_blocks=blocks, max_blocks=args.max_blocks, full_depth=full_depth,
                  timestep_scale_multiplier=timestep_scale, av_ca_timestep_scale_multiplier=av_ca_timestep_scale,
                  dequantized_tensor_types=weight_stats["dequantized_tensor_types"], memory_constraints=constraints,
                  weight_residency_mode="one streamed block plus persistent non-block weights" if args.stream_block_weights else "all weights materialized",
                  weight_loading=weight_stats, reference_forward_includes_streamed_weight_loading=args.stream_block_weights,
                  peak_process_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * (1 if sys.platform == "darwin" else 1024),
                  memory_scope="Process peak RSS includes mapped file pages, weights, activations and dequantization temporaries; estimated_dequantized_residency_bytes counts retained weights only.",
                  one_video_attention_score_matrix_bytes=config["VideoHeads"] * probe["tokens"]["video"] ** 2 * 4,
                  load_dequantize_seconds=load_seconds, reference_forward_seconds=elapsed, comparisons=comparisons,
                  scope="Official LTX component graph with " + ("captured actual projected contexts/latents, positions, keyframe markers and token timesteps" if captured is not None else "synthetic projected contexts/latents") + " and real checkpoint weights. " +
                        ("Full denoiser depth; " if full_depth else "Partial denoiser depth; ") +
                        "excludes text encoder/connectors/sampler/VAEs/full generation. CPU PyTorch timing is not a sdcpp performance comparison.")
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
