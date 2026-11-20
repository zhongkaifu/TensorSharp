#!/usr/bin/env python3
# Copyright (c) Zhongkai Fu. All rights reserved.
# Licensed under the BSD-3-Clause license in the repository root.
"""Write the stored CPU torch.randn outputs the diffusion runtime's TorchCpuRandn stream is tested against.

Each case seeds one CPU generator (torch.Generator().manual_seed(seed), what torch.manual_seed does to
the default generator) and makes successive float32 torch.randn calls with it, in order: the state is
carried across calls, which is the property the stateful stream reproduces (diffusers draws an edit's
reference posteriors, then the target noise, then the reference augmentation from one generator).

Small draws are stored whole; large ones as their first and last 32 values plus sums. Counts below 16
are not used: torch draws those through its scalar normal_distribution<double>, which the stream does
not reproduce (it refuses them).

    python3 eng/torch-randn-stream-fixture.py InferenceWeb.Tests/Fixtures/MiniMaxH3Image/torch-randn-stream.json

Needs torch (any CPU build). The file records torch's version and the platform, because PyTorch's x86
AVX2 build evaluates log/sin/cos with SLEEF and its other builds with libm: a value may differ from
another build's in the last bit, which the test's tolerance covers.
"""
import json
import platform
import sys

import torch

# (seed, [count of each successive randn call])
CASES = [
    # Not multiples of 16 (the last block of 16 is drawn again), a multiple, exactly 16.
    (42, [360, 384, 16, 1000, 17]),
    # The H3-Image edit order at a small geometry: a reference posterior over its uncropped latent
    # (24 x 9 x 7), the target noise (24 x 8 x 6), then the reference augmentation over the cropped
    # latent (24 x 8 x 6).
    (123456789, [24 * 9 * 7, 24 * 8 * 6, 24 * 8 * 6]),
    # manual_seed keeps the low 32 bits for the Mersenne Twister: 2^32 + 5 draws what 5 draws.
    (5, [100, 200]),
    (4294967301, [100, 200]),
    # A text-to-image draw (24 x 128 x 128) and one more call after it, so the state after a long
    # draw (630 twists) is checked too.
    (7001, [24 * 128 * 128, 50]),
]

WHOLE_LIMIT = 4096
EDGE = 32


def main() -> int:
    if len(sys.argv) != 2:
        print(__doc__, file=sys.stderr)
        return 2
    cases = []
    for seed, counts in CASES:
        generator = torch.Generator(device="cpu").manual_seed(seed)
        calls = []
        for count in counts:
            if count < 16:
                raise SystemExit("counts below 16 take torch's scalar path, which the stream does not reproduce")
            values = torch.randn(count, generator=generator, dtype=torch.float32)
            call = {"count": count}
            if count <= WHOLE_LIMIT:
                call["values"] = values.tolist()
            else:
                call["first"] = values[:EDGE].tolist()
                call["last"] = values[-EDGE:].tolist()
                call["sum"] = float(values.double().sum())
                call["absSum"] = float(values.double().abs().sum())
            calls.append(call)
        cases.append({"seed": str(seed), "calls": calls})
    document = {
        "generator": f"torch {torch.__version__}, {platform.system()} {platform.machine()}, python {platform.python_version()}",
        "script": "eng/torch-randn-stream-fixture.py",
        "cases": cases,
    }
    with open(sys.argv[1], "w", encoding="utf-8") as f:
        json.dump(document, f, separators=(",", ":"))
        f.write("\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
