#!/usr/bin/env python3
"""A/B Qwen-Image-2.1 builds or settings with qwen-image21-bench.py, in rotated order.

Each arm is a TensorSharp CLI plus optional environment overrides. Every round runs
each arm once in a fresh process; the arm order rotates between rounds so no arm
always runs first or last (a fixed order pins thermal drift on one arm). The summary
compares every arm's output PNG with the first arm's: identical SHA-256 proves a
bit-exact change, PSNR quantifies an intentional numerical one.

Example (a baseline build against the current one, with the prefix cache on and off):
  python3 eng/validation/qwen-image21-ab.py --rounds 2 \\
      --arm base=artifacts/baseline/TensorSharp.Cli/bin/TensorSharp.Cli.dll \\
      --arm cache=TensorSharp.Cli/bin/TensorSharp.Cli.dll \\
      --arm nocache=TensorSharp.Cli/bin/TensorSharp.Cli.dll,TS_QWEN21_PREFIX_CACHE=0 \\
      -- --mode edit --image ref.png --width 1024 --height 1024 --steps 40

Arguments after `--` go to qwen-image21-bench.py unchanged (--engine and --cli are
set per arm). Generated records belong in ignored artifacts/ or docs/validation/.

Edits: the bench runs TensorSharp's default edit noise, keyed to the reference images. A
build from before that change drew an edit's noise from the seed alone, so pass
`--edit-noise seed` after `--` when one arm is such a build, or every edit differs from it
for that reason alone; an arm can also set TS_QWEN21_EDIT_NOISE itself.
"""
import argparse
import json
import math
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]
BENCH = ROOT / "eng/validation/qwen-image21-bench.py"


def parse_arm(text):
    name, _, spec = text.partition("=")
    if not name or not spec:
        raise argparse.ArgumentTypeError(f"arm '{text}' must be NAME=CLI[,ENV=VALUE...]")
    cli, *overrides = spec.split(",")
    env = {}
    for item in overrides:
        key, sep, value = item.partition("=")
        if not sep or not key:
            raise argparse.ArgumentTypeError(f"arm '{name}': override '{item}' must be ENV=VALUE")
        env[key] = value
    return {"name": name, "cli": str(Path(cli).resolve()), "env": env}


def psnr(path_a, path_b):
    import numpy as np
    from PIL import Image
    a = np.asarray(Image.open(path_a).convert("RGBA"), dtype=np.float64)
    b = np.asarray(Image.open(path_b).convert("RGBA"), dtype=np.float64)
    if a.shape != b.shape:
        return None, None
    mse = float(((a - b) ** 2).mean())
    mae = float(np.abs(a - b).mean())
    return (math.inf if mse == 0 else 10 * math.log10(255.0 ** 2 / mse)), mae


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--arm", type=parse_arm, action="append", required=True)
    parser.add_argument("--rounds", type=int, default=1)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("bench_args", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    bench_args = args.bench_args[1:] if args.bench_args[:1] == ["--"] else args.bench_args
    if any(a in ("--engine", "--cli", "--output", "--repeat") or a.startswith(("--engine=", "--cli=", "--output=", "--repeat="))
           for a in bench_args):
        parser.error("--engine, --cli, --output and --repeat are set by this driver")
    names = [arm["name"] for arm in args.arm]
    if len(set(names)) != len(names):
        parser.error("arm names must be unique")
    args.output.mkdir(parents=True, exist_ok=True)

    records = []
    for round_index in range(args.rounds):
        shift = round_index % len(args.arm)
        order = args.arm[shift:] + args.arm[:shift]
        if round_index % 2:
            order = list(reversed(order))
        for arm in order:
            out = args.output / f"r{round_index + 1}-{arm['name']}"
            env = dict(os.environ, **arm["env"])
            command = [sys.executable, str(BENCH), "--engine", "tensorsharp", "--cli", arm["cli"],
                       "--output", str(out), *bench_args]
            print(f"[round {round_index + 1}] {arm['name']}: {' '.join(f'{k}={v}' for k, v in arm['env'].items())}", flush=True)
            completed = subprocess.run(command, env=env, capture_output=True, text=True)
            (out.parent / f"{out.name}.driver.log").write_text(completed.stdout + completed.stderr)
            report = json.loads((out / "benchmark.json").read_text()) if (out / "benchmark.json").exists() else None
            run = report["runs"][0]["engines"]["tensorsharp"] if report else {}
            steps = [step["seconds"] for step in run.get("steps") or []]
            record = {
                "round": round_index + 1, "arm": arm["name"], "env": arm["env"], "cli": arm["cli"],
                "status": run.get("status", f"driver exit {completed.returncode}"),
                "wall_seconds": run.get("wall_seconds"),
                "phases_seconds": run.get("phases_seconds"),
                "first_step_seconds": steps[0] if steps else None,
                "later_step_mean_seconds": (sum(steps[1:]) / (len(steps) - 1)) if len(steps) > 1 else None,
                "max_rss_bytes": run.get("max_rss_bytes"),
                # nvidia-smi samples whole devices, other processes included.
                "gpu_peak_memory_mib": {str(p.get("index")): p.get("peak_memory_used_mib")
                                        for p in (run.get("gpu_sampling") or {}).get("peaks", [])},
                "image": (run.get("image") or {}).get("path"),
                "sha256": (run.get("image") or {}).get("sha256"),
            }
            records.append(record)
            print(f"    {record['status']}: wall {record['wall_seconds']}, steps {record['first_step_seconds']} then "
                  f"{record['later_step_mean_seconds']}, sha {str(record['sha256'])[:16]}", flush=True)

    reference = next((r for r in records if r["arm"] == names[0] and r["image"]), None)
    for record in records:
        if reference and record["image"]:
            record["identical_to_" + names[0]] = record["sha256"] == reference["sha256"]
            record["psnr_db_vs_" + names[0]], record["mae_vs_" + names[0]] = psnr(reference["image"], record["image"])

    summary = {}
    for name in names:
        rows = [r for r in records if r["arm"] == name and r["status"] == "passed"]
        if not rows:
            summary[name] = {"passed": 0}
            continue
        mean = lambda key: sum(r[key] for r in rows) / len(rows)
        summary[name] = {
            "passed": len(rows),
            "wall_seconds_mean": mean("wall_seconds"),
            "denoise_seconds_mean": sum(r["phases_seconds"]["denoise"] for r in rows) / len(rows),
            "first_step_seconds_mean": mean("first_step_seconds"),
            "later_step_seconds_mean": mean("later_step_mean_seconds") if all(r["later_step_mean_seconds"] for r in rows) else None,
            "max_rss_gib_max": max(r["max_rss_bytes"] or 0 for r in rows) / 2 ** 30,
            "gpu_peak_memory_mib_max": {index: max(r["gpu_peak_memory_mib"].get(index) or 0 for r in rows)
                                        for index in rows[0]["gpu_peak_memory_mib"]},
            "distinct_sha256": sorted({r["sha256"] for r in rows}),
        }
    (args.output / "ab.json").write_text(json.dumps({"bench_args": bench_args, "arms": args.arm,
                                                      "records": records, "summary": summary}, indent=2, default=str))
    print(json.dumps(summary, indent=2, default=str))


if __name__ == "__main__":
    main()
