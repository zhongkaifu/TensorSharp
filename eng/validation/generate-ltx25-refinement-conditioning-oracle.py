#!/usr/bin/env python3
"""Record refinement conditioning from unchanged official LTX on CPU, without weights.

The fixture supplies deterministic prior latents and noise, then invokes the
official conditioning classes and GaussianNoiser. It covers partial, zero and
full image strength without claiming RNG, GPU or perceptual generation parity.
"""
import argparse
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import subprocess
import sys

REVISION = "be6d6d0a69d11a0a399c462ed92606692e5f8697"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    source = args.reference.resolve()
    assert subprocess.check_output(["git", "-C", str(source), "rev-parse", "HEAD"], text=True).strip() == REVISION
    assert not subprocess.check_output(["git", "-C", str(source), "status", "--porcelain"], text=True).strip()
    sys.path.insert(0, str(source / "packages/ltx-core/src"))
    import torch
    from ltx_core.components.noisers import GaussianNoiser
    from ltx_core.components.patchifiers import VideoLatentPatchifier
    from ltx_core.conditioning.types.keyframe_cond import VideoConditionByKeyframeIndex
    from ltx_core.conditioning.types.latent_cond import VideoConditionByLatentIndex
    from ltx_core.tools import VideoLatentTools
    from ltx_core.types import VideoLatentShape

    torch.set_num_threads(1)
    tools = VideoLatentTools(VideoLatentPatchifier(1), VideoLatentShape(1, 128, 3, 1, 1), 24)
    template = tools.create_initial_state("cpu", torch.float32)
    prior = torch.linspace(-2.3, 1.4, template.latent.numel()).reshape_as(template.latent)
    noise = torch.linspace(-1.3, 1.7, 5 * 128).reshape(1, 5, 128)
    scale = 0.909375
    flat = lambda tensor: tensor.detach().flatten().tolist()
    images = [torch.linspace(-.7, .9, 128).reshape(1, 128, 1, 1, 1) + index * .03 for index in (0, 7, 16)]

    class FixedNoiser(GaussianNoiser):
        def _sample_noise(self, latent_state):
            assert latent_state.latent.shape == noise.shape
            return noise

    cases = []
    for strengths in ((.75, .5, 1.0), (0.0, 0.0, 1.0), (1.0, 1.0, 1.0)):
        state = replace(template.clone(), latent=prior.clone())
        for index, latent, strength in zip((0, 7, 16), images, strengths, strict=True):
            conditioning = (VideoConditionByLatentIndex(latent, strength, 0) if index == 0 else
                            VideoConditionByKeyframeIndex(latent, index, strength))
            state = conditioning.apply_to(state, tools)
        state = FixedNoiser(torch.Generator(device="cpu"))(state, noise_scale=scale)
        cases.append({"strengths": strengths, "output": flat(state.latent), "mask": flat(state.denoise_mask)})
    report = {
        "source_revision": REVISION, "torch_version": torch.__version__, "device": "cpu", "dtype": "float32",
        "noise_scale": scale, "prior": flat(prior), "noise": flat(noise),
        "images": [{"frame_index": index, "latent": flat(latent)} for index, latent in zip((0, 7, 16), images, strict=True)],
        "cases": cases,
        "source_sha256": {name: hashlib.sha256((source / "packages/ltx-core/src/ltx_core" / name).read_bytes()).hexdigest()
                          for name in ["conditioning/types/latent_cond.py", "conditioning/types/keyframe_cond.py",
                                       "components/noisers.py", "components/patchifiers.py", "tools.py"]},
        "limitations": "CPU Float32 conditioning and GaussianNoiser with supplied noise only; no RNG parity, model weights, GPU or video generation.",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
