#!/usr/bin/env python3
"""Qwen-Image-2.1-Turbo validation: parity with stable-diffusion.cpp, speed against the base
checkpoint, the CLI's peak footprint, and the style plug-ins TensorAgent offers on Turbo, each
run through qwen-image21-bench.py.

Every measurement is a fresh process run one after another (one GPU job at a time), and every
bench run writes its own benchmark.json, logs and PNGs under --output. Afterwards this script
prints the parity and performance tables from those reports; --summarize prints them again
without running anything.

  parity    TensorSharp and sd.cpp on the same Turbo GGUF, encoder, VAE, prompt, seed, size,
            Philox noise (--rng cuda) and the published 8-step sigmas: three text-to-image
            prompts (the card's neon sign, a Chinese sign, a photograph) and one edit of
            --edit-image, and the neon sign once more on --turbo-q8 when it is given. The edit
            draws sd.cpp's seed-only noise on both sides (the bench's --edit-noise seed, its
            default when both engines run), not TensorSharp's reference-keyed edit noise.
  perf      TensorSharp only, 1024x1024 at seed 42: base 2.1 at 40 steps, Turbo (and Turbo Q8_0),
            and base 2.1 with --viggle at its 6-step recipe, in A-B-C-D-D-C-B-A order.
  footprint TensorSharp only: the peak memory footprint (macOS /usr/bin/time -l) of an edit at
            1248x832 per transformer, the measurement the TensorAgent catalog tiers use.
  style     TensorSharp only: each style plug-in validated on Turbo (Film Stills, Grainscape; its
            config/lora/ plug-in and the weights found under --lora-dir) at the plug-in's own
            strength, against the same prompt and seed without it, at every --style-seeds seed,
            1024x1024 on --turbo. The pairs are for the eye: the summary lists them with the
            per-step cost of the plug-in.

Example (the companions are Qwen-Image-2.1's; sd.cpp needs c150a6b or newer for --sigmas):
  python3 eng/validation/qwen-image21-turbo.py --suite parity,perf,footprint \\
      --turbo models/qwen-image-2.1/Qwen-Image-2.1-Turbo-AD-Q4_K.gguf \\
      --turbo-q8 models/qwen-image-2.1/Qwen-Image-2.1-Turbo-Q8_0.gguf \\
      --base models/qwen-image-2.1/qwen_image_2.1_Q4_K_M.gguf --companions models/qwen-image-2.1 \\
      --viggle-weights loras/Qwen-Image-2.1-viggle-turbo-v0.2.1-6step-lora-r128.safetensors \\
      --viggle-config config/lora/qwen-image-2.1-viggle-turbo.json \\
      --edit-image boat-1728x608.png --edit-prompt-file boat-prompt.txt --sd-cli sd-build/bin/sd-cli
  python3 eng/validation/qwen-image21-turbo.py --suite style \\
      --turbo models/qwen-image-2.1/Qwen-Image-2.1-Turbo-AD-Q4_K.gguf --companions models/qwen-image-2.1 \\
      --lora-dir models/qwen-image-2.1/loras --style-seeds 42,7
"""
import argparse
import glob
import json
import os
from pathlib import Path
import re
import statistics
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
BENCH = Path(__file__).resolve().parent / "qwen-image21-bench.py"
NEON = 'a neon sign that reads "OPEN LATE", rainy night'
PROMPTS = {
    "neon": NEON,
    "zh": "一家老茶馆门口挂着木质招牌，招牌上用毛笔字写着“清风茶社”四个大字，屋檐下挂着红灯笼，黄昏暖光，写实摄影",
    "portrait": "A close-up portrait photograph of an elderly fisherman with a weathered face and a knitted wool cap, "
                "harbor and boats softly blurred behind him, golden hour light, 85mm lens",
}
# The style plug-ins TensorAgent offers on Turbo (LoraCatalog AlsoFor), each with a prompt that
# names nothing of the style, so the plug-in's look is what changes.
STYLES = {
    "film": ("config/lora/qwen-image-2.1-film-stills.json",
             "A woman waiting alone at a tram stop at night, wet street, neon reflections, cinematic"),
    "grain": ("config/lora/qwen-image-2.1-grainscape.json",
              "A mountain lake at dawn with a small wooden rowing boat at the shore"),
}
COMPANIONS = {"vae": "qwen_image_2.1_vae_bf16.safetensors", "text-encoder": "Qwen3VL-8B-Instruct-Q4_K_M.gguf",
              "mmproj": "mmproj-Qwen3VL-8B-Instruct-F16.gguf"}


def bench(args, extra, output, seed=None):
    """One qwen-image21-bench.py invocation; a cooldown follows it, outside any measurement."""
    if (output / "benchmark.json").exists() and not args.dry_run:
        print(f"skip {output}: a report is already there")
        return
    command = [sys.executable, str(BENCH)] + [f"--{n}={args.companions / f}" for n, f in COMPANIONS.items()] + [
        f"--cli={args.cli}", f"--backend={args.backend}", f"--seed={args.seed if seed is None else seed}",
        f"--output={output}"] + extra
    if args.sd_cli:
        command.append(f"--sd-cli={args.sd_cli}")
    if args.sd_repo:
        command.append(f"--sd-repo={args.sd_repo}")
    if args.dry_run:
        command.append("--dry-run")
    print("+", " ".join(command[2:]), flush=True)
    subprocess.run(command, check=False)
    if not args.dry_run:
        time.sleep(args.cooldown)


def parity(args):
    out = args.output / "parity"
    common = ["--engine=both", "--variant=turbo", "--steps=8", "--cfg=1", f"--cooldown-seconds={args.cooldown}"]
    for name, prompt in PROMPTS.items():
        bench(args, common + [f"--dit={args.turbo}", "--mode=t2i", f"--width={args.size}", f"--height={args.size}",
                              f"--prompt={prompt}"], out / name)
    if args.edit_image:
        prompt = args.edit_prompt_file.read_text().strip() if args.edit_prompt_file else args.edit_prompt
        width, height = args.edit_size
        bench(args, common + [f"--dit={args.turbo}", "--mode=edit", f"--image={args.edit_image}",
                              f"--width={width}", f"--height={height}", f"--prompt={prompt}"], out / "edit")
    if args.turbo_q8:
        bench(args, common + [f"--dit={args.turbo_q8}", "--mode=t2i", f"--width={args.size}", f"--height={args.size}",
                              f"--prompt={NEON}"], out / "neon-q8")


def perf(args):
    out = args.output / "perf"
    common = ["--engine=tensorsharp", f"--width={args.size}", f"--height={args.size}", f"--prompt={NEON}"]
    arms = [("base40", ["--variant=base", "--steps=40", f"--dit={args.base}"]),
            ("turbo", ["--variant=turbo", "--steps=8", f"--dit={args.turbo}"])]
    if args.turbo_q8:
        arms.append(("turbo-q8", ["--variant=turbo", "--steps=8", f"--dit={args.turbo_q8}"]))
    if args.viggle_weights and args.viggle_config:
        arms.append(("viggle6", ["--variant=base", "--steps=6", f"--dit={args.base}", f"--lora={args.viggle_weights}",
                                 f"--lora-config={args.viggle_config}"]))
    # A-B-C-D, then D-C-B-A: each arm runs once early and once late, so drift cancels in the median.
    for repeat, order in ((1, arms), (2, list(reversed(arms)))):
        for name, extra in order:
            bench(args, common + extra, out / f"{name}-{repeat}")


def footprint(args):
    out = args.output / "footprint"
    if not args.edit_image:
        print("footprint needs --edit-image")
        return
    for name, dit, variant in (("base", args.base, "base"), ("turbo", args.turbo, "turbo"), ("turbo-q8", args.turbo_q8, "turbo")):
        if dit:
            bench(args, ["--engine=tensorsharp", "--mode=edit", f"--image={args.edit_image}", "--width=1248", "--height=832",
                         f"--variant={variant}", "--steps=8", f"--dit={dit}",
                         "--prompt=Turn the sketch into a photograph of the ship at sea."], out / name)


def plugin(config_path, lora_dir):
    """A plug-in config's weights file under lora_dir (searched below it) and its own strength."""
    config = json.loads(re.sub(r"(?m)^\s*//.*$", "", config_path.read_text()))  # full-line // comments only
    name = Path(config["weights"]["path"]).name
    found = sorted(lora_dir.rglob(name))
    if not found:
        raise SystemExit(f"{name} ({config_path.name}'s weights) is not under {lora_dir}")
    return found[0], config.get("scale", 1.0)


def style(args):
    out = args.output / "style"
    for name, (config, prompt) in STYLES.items():
        config = ROOT / config
        weights, scale = plugin(config, args.lora_dir)
        common = ["--engine=tensorsharp", "--variant=turbo", "--steps=8", f"--dit={args.turbo}",
                  f"--width={args.size}", f"--height={args.size}", f"--prompt={prompt}"]
        for seed in args.style_seeds:
            bench(args, common, out / f"{name}-none-{seed}", seed)
            bench(args, common + [f"--lora={weights}", f"--lora-config={config}", f"--lora-scale={scale}"],
                  out / f"{name}-lora-{seed}", seed)


def steady(engine):
    steps = [s["seconds"] for s in engine["steps"]]
    return statistics.mean(steps[1:]) if len(steps) > 1 else None


def summarize(output):
    print("\nParity (TensorSharp vs sd.cpp, steady-state seconds per step, wall seconds):")
    for report in sorted(glob.glob(str(output / "parity" / "*" / "benchmark.json"))):
        run = json.loads(Path(report).read_text())["runs"][0]
        engines, pixels = run["engines"], run.get("pixel_comparison", {})
        ts, sd = engines.get("tensorsharp"), engines.get("sd_cpp")
        if not (ts and sd and pixels.get("available")):
            print(f"  {Path(report).parent.name}: incomplete")
            continue
        print(f"  {Path(report).parent.name}: PSNR {pixels['raw_rgba']['psnr_db']:.1f} dB, "
              f"TensorSharp {steady(ts):.2f} s/step {ts['wall_seconds']:.1f} s, sd.cpp {steady(sd):.2f} s/step {sd['wall_seconds']:.1f} s")
    print("\nPerformance (TensorSharp, median of the runs):")
    arms = {}
    for report in sorted(glob.glob(str(output / "perf" / "*" / "benchmark.json"))):
        name = Path(report).parent.name.rsplit("-", 1)[0]
        runs = json.loads(Path(report).read_text())["runs"]
        if runs and "tensorsharp" in runs[0]["engines"]:
            arms.setdefault(name, []).append(runs[0]["engines"]["tensorsharp"])
    for name, engines in arms.items():
        print(f"  {name}: {statistics.median(steady(e) for e in engines):.2f} s/step, "
              f"{statistics.median(e['wall_seconds'] for e in engines):.1f} s wall ({len(engines)} runs)")
    print("\nStyle plug-ins on Turbo (pairs for the eye; steady seconds per step without / with):")
    for none in sorted(glob.glob(str(output / "style" / "*-none-*" / "benchmark.json"))):
        name, seed = Path(none).parent.name.split("-none-")
        lora = output / "style" / f"{name}-lora-{seed}" / "benchmark.json"
        if not lora.exists():
            print(f"  {name} seed {seed}: no plug-in run")
            continue
        a, b = (json.loads(Path(p).read_text())["runs"][0]["engines"].get("tensorsharp") for p in (none, lora))
        if not (a and b and a.get("steps") and b.get("steps")):
            print(f"  {name} seed {seed}: incomplete")
            continue
        print(f"  {name} seed {seed}: {steady(a):.2f} / {steady(b):.2f} s/step, "
              f"{Path(none).parent / 'tensorsharp-1.png'} vs {lora.parent / 'tensorsharp-1.png'}")
    print("\nPeak memory footprint (edit at 1248x832):")
    for log in sorted(glob.glob(str(output / "footprint" / "*" / "tensorsharp-1.log"))):
        match = re.search(r"(\d+)\s+peak memory footprint", Path(log).read_text(errors="replace"))
        print(f"  {Path(log).parent.name}: {int(match.group(1)) / 1e9:.2f} GB" if match else f"  {Path(log).parent.name}: not reported")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--suite", default="parity,perf,footprint")
    parser.add_argument("--turbo", type=Path, help="Turbo transformer GGUF (e.g. AD-Q4_K).")
    parser.add_argument("--turbo-q8", type=Path, help="A second Turbo GGUF (e.g. Q8_0), optional.")
    parser.add_argument("--base", type=Path, help="Base Qwen-Image-2.1 transformer GGUF (perf, footprint).")
    parser.add_argument("--companions", type=Path, help="Folder with the 2.1 VAE, Qwen3-VL-8B encoder and mmproj.")
    parser.add_argument("--viggle-weights", type=Path)
    parser.add_argument("--viggle-config", type=Path)
    parser.add_argument("--lora-dir", type=Path, help="Folder holding the style plug-ins' weights (style).")
    parser.add_argument("--style-seeds", default="42,7", help="Comma-separated seeds for each style pair (style).")
    parser.add_argument("--edit-image", type=Path, help="Reference for the edit, already at its conditioning size.")
    parser.add_argument("--edit-prompt", default="Create a photorealistic photograph of the object in the sketch.")
    parser.add_argument("--edit-prompt-file", type=Path)
    parser.add_argument("--edit-size", type=int, nargs=2, default=(1728, 608), metavar=("WIDTH", "HEIGHT"))
    parser.add_argument("--size", type=int, default=1024)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--backend", default="ggml_metal")
    parser.add_argument("--cli", type=Path, default=ROOT / "TensorSharp.Cli/bin/TensorSharp.Cli.dll")
    parser.add_argument("--sd-cli", type=Path)
    parser.add_argument("--sd-repo", type=Path)
    parser.add_argument("--cooldown", type=float, default=20)
    parser.add_argument("--output", type=Path, default=ROOT / "artifacts/qwen-image-2.1-turbo")
    parser.add_argument("--summarize", action="store_true", help="Only print the tables from an earlier --output.")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    args.output = args.output.resolve()
    if not args.summarize:
        suites = [s.strip() for s in args.suite.split(",") if s.strip()]
        unknown = set(suites) - {"parity", "perf", "footprint", "style"}
        if unknown:
            parser.error(f"unknown suite(s): {', '.join(sorted(unknown))}")
        if not args.turbo or not args.companions:
            parser.error("--turbo and --companions are required.")
        if ("perf" in suites or "footprint" in suites) and not args.base:
            parser.error("perf and footprint compare against --base.")
        if "parity" in suites and not args.sd_cli:
            parser.error("parity needs --sd-cli.")
        if "style" in suites and not args.lora_dir:
            parser.error("style needs --lora-dir.")
        args.style_seeds = [int(v) for v in args.style_seeds.split(",") if v.strip()]
        for suite in suites:
            {"parity": parity, "perf": perf, "footprint": footprint, "style": style}[suite](args)
    if not args.dry_run:
        summarize(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
