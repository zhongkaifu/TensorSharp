"""Compare optional Qwen first-forward tensor dumps without NumPy dependencies."""
import argparse
from array import array
import json
import math
from pathlib import Path
import sys

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("reference", type=Path)
parser.add_argument("candidate", type=Path)
parser.add_argument("--reference-prefix", default="fused.")
parser.add_argument("--candidate-prefix", default="tp.")
args = parser.parse_args()

def read(path):
    values = array("f")
    values.frombytes(path.read_bytes())
    if sys.byteorder != "little":
        values.byteswap()
    return values

results = []
invalid = False
for source in sorted(args.reference.glob(args.reference_prefix + "*.f32")):
    name = source.name[len(args.reference_prefix):]
    target = args.candidate / (args.candidate_prefix + name)
    if not target.exists():
        results.append({"tensor": name, "missing_candidate": True})
        invalid = True
        continue
    try:
        x, y = read(source), read(target)
    except (OSError, ValueError) as error:
        results.append({"tensor": name, "read_error": str(error)})
        invalid = True
        continue
    if not x or len(x) != len(y):
        results.append({"tensor": name, "reference_length": len(x), "candidate_length": len(y)})
        invalid = True
        continue
    if not all(math.isfinite(v) for values in (x, y) for v in values):
        results.append({"tensor": name, "elements": len(x), "finite": False})
        invalid = True
        continue
    square_error = math.fsum((a - b) ** 2 for a, b in zip(x, y))
    norm_x = math.fsum(a * a for a in x)
    norm_y = math.fsum(b * b for b in y)
    results.append({
        "tensor": name, "elements": len(x),
        "finite": True,
        "equal_elements": sum(a == b for a, b in zip(x, y)),
        "max_abs": max((abs(a - b) for a, b in zip(x, y)), default=0),
        "relative_l2": math.sqrt(square_error / max(norm_x, 1e-300)),
        "cosine": math.fsum(a * b for a, b in zip(x, y)) / math.sqrt(max(norm_x * norm_y, 1e-300)),
    })
print(json.dumps(results, indent=2, allow_nan=False))
if not results:
    raise SystemExit("No matching reference tensors")
raise SystemExit(1 if invalid else 0)
