#!/usr/bin/env python3
"""Compare captured-position CPU BuildRope with unchanged official Float64 RoPE.

Runs small array operations only; no denoiser weights or GPU runtime are loaded.
"""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--probe", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--fps", type=float, required=True, help="Actual capture FPS; older forward manifests omit it")
    args = parser.parse_args()
    if args.output.exists():
        raise SystemExit("Refusing to overwrite evidence")
    revision = subprocess.check_output(["git", "-C", str(args.reference), "rev-parse", "HEAD"], text=True).strip()
    if revision != "be6d6d0a69d11a0a399c462ed92606692e5f8697" or subprocess.check_output(
            ["git", "-C", str(args.reference), "status", "--porcelain"], text=True).strip():
        raise SystemExit("Expected clean pinned official source")
    sys.path.insert(0, str(args.reference.resolve() / "packages/ltx-core/src"))
    import numpy as np
    import torch
    from ltx_core.model.transformer.rope import (
        generate_freq_grid_np, generate_freqs, get_fractional_positions, precompute_freqs_cis)
    from ltx_core.components.patchifiers import VideoLatentPatchifier, AudioPatchifier, get_pixel_coords
    from ltx_core.types import VideoLatentShape, AudioLatentShape, SpatioTemporalScaleFactors
    torch.set_num_threads(2)
    probe = json.loads(args.probe.read_text())
    capture_path = Path(probe["capture"])
    if hashlib.sha256(capture_path.read_bytes()).hexdigest() != probe["capture_sha256"]:
        raise SystemExit("Capture identity differs")
    capture = json.loads(capture_path.read_text())
    if probe["frequency_precision"] != "Float64":
        raise SystemExit("Expected Float64 grid")

    def read(spec):
        path = Path(spec["path"])
        data = path.read_bytes()
        if hashlib.sha256(data).hexdigest() != spec["sha256"] or len(data) != int(np.prod(spec["shape"])) * 4:
            raise ValueError("Array identity differs: " + str(path))
        return np.frombuffer(data, dtype="<f4").reshape(spec["shape"]).copy()

    def compare(actual, expected):
        a, e = np.asarray(actual), np.asarray(expected)
        difference = a.astype(np.float64) - e.astype(np.float64)
        def ordered_float_bits(value):
            bits = value.astype(np.float32, copy=False).view(np.uint32)
            return np.where(bits & 0x80000000, ~bits, bits ^ 0x80000000).astype(np.int64)
        ulps = np.abs(ordered_float_bits(a) - ordered_float_bits(e))
        ulps[a == e] = 0  # +0 and -0 compare equally.
        return dict(shape=list(a.shape), differing_elements=int(np.count_nonzero(a != e)),
                    max_float32_ulps=int(ulps.max()),
                    max_abs=float(np.abs(difference).max()), rmse=float(np.sqrt(np.mean(difference ** 2))),
                    relative_l2=float(np.linalg.norm(difference.ravel()) / max(np.linalg.norm(e.astype(np.float64).ravel()), 1e-30)))

    positions = {name: read(capture["inputs"][name + "_positions"]) for name in ("video", "audio")}
    if not np.isfinite(args.fps) or args.fps <= 0:
        raise ValueError("FPS must be positive and finite")
    width, height, frames = capture["stage_geometry"]
    latent_shape = VideoLatentShape(1, capture["config"]["VideoChannels"], (frames - 1) // 8 + 1, height // 32, width // 32)
    pixel_bounds = get_pixel_coords(VideoLatentPatchifier(1).get_patch_grid_bounds(latent_shape),
                                   SpatioTemporalScaleFactors(8, 32, 32), causal_fix=True).float()
    pixel_bounds[:, 0] /= args.fps
    audio_bounds = AudioPatchifier(1).get_patch_grid_bounds(AudioLatentShape(1, 8, capture["tokens"]["audio"], 16))
    position_comparisons = dict(video=compare(positions["video"], pixel_bounds.transpose(1, 2)[0].numpy()),
                                audio=compare(positions["audio"], audio_bounds.transpose(1, 2)[0].numpy()))
    streams = {}
    for name, native in probe["streams"].items():
        source_positions = positions["video"][:, :1] if name == "cross_video" else positions[name]
        tokens, axes = source_positions.shape[:2]
        grid = torch.from_numpy(source_positions).unsqueeze(0).transpose(1, 2)
        max_pos = [20, 2048, 2048] if axes == 3 else [20]
        indices = generate_freq_grid_np(10000, axes, native["dimensions"], torch.device("cpu"))
        coordinates = get_fractional_positions((grid[..., 0] + grid[..., 1]) / 2.0, max_pos) * 2 - 1
        freqs = generate_freqs(indices, grid, max_pos, True)
        pad = native["dimensions"] // 2 - freqs.shape[-1]
        phases = torch.nn.functional.pad(freqs, (pad, 0))[0].numpy()
        cosine, sine = precompute_freqs_cis(grid, native["dimensions"], torch.float32, max_pos=max_pos,
                                          use_middle_indices_grid=True, num_attention_heads=native["heads"],
                                          freq_grid_generator=generate_freq_grid_np)
        cosine = cosine.transpose(1, 2).reshape(tokens, -1).numpy()
        sine = sine.transpose(1, 2).reshape(tokens, -1).numpy()
        expected = dict(frequency_grid=indices.numpy(), coordinates=coordinates[0].numpy(), phases=phases,
                        cosine=cosine, sine=sine)
        arrays = {field: read(native[field]) for field in expected}
        result = {field: compare(arrays[field], expected[field]) for field in expected}
        # The same native phases evaluated by torch isolate trigonometry from
        # phase construction. Padding phase zero naturally gives cos=1,sin=0.
        native_phases = torch.from_numpy(arrays["phases"])
        result["native_phase_torch_cosine"] = compare(arrays["cosine"], native_phases.cos().numpy())
        result["native_phase_torch_sine"] = compare(arrays["sine"], native_phases.sin().numpy())
        if name != "audio":
            frame_tokens = (capture["stage_geometry"][0] // 32) * (capture["stage_geometry"][1] // 32)
            result["per_latent_frame"] = []
            for start in range(0, tokens, frame_tokens):
                selected = slice(start, start + frame_tokens)
                result["per_latent_frame"].append(dict(latent_frame=start // frame_tokens,
                    native_temporal_coordinate=float(arrays["coordinates"][start, 0]),
                    official_temporal_coordinate=float(expected["coordinates"][start, 0]),
                    phase=compare(arrays["phases"][selected], phases[selected]),
                    cosine=compare(arrays["cosine"][selected], cosine[selected])))
        streams[name] = result
    result = dict(completed=True, reference_revision=revision, torch=torch.__version__, numpy=np.__version__,
                  tool_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                  native_probe=str(args.probe), native_probe_sha256=hashlib.sha256(args.probe.read_bytes()).hexdigest(),
                  native_geometry_source_sha256=hashlib.sha256(Path(probe["source"]).read_bytes()).hexdigest(),
                  capture_sha256=probe["capture_sha256"], fps=args.fps, position_comparisons=position_comparisons,
                  frequency_precision="Float64", streams=streams,
                  scope="CPU-only captured positions and actual native BuildRope. Official NumPy Float64 frequency generation, Float32 coordinates/phases/trig; text context has no RoPE in this denoiser. No model forward or rendered-quality claim.")
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({name: {field: metrics for field, metrics in result.items() if field != "per_latent_frame"}
                      for name, result in streams.items()}, indent=2))


if __name__ == "__main__":
    main()
