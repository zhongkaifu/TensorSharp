#!/usr/bin/env python3
"""Rebind archived full DeepSeek sparse/reference campaigns without running them.

All nine original suites and their order remain intact. TP7 uses reviewed
CPU12/context64K planning bounds; allocation, quality and performance still
require execution. Native pins and archived inputs are immutable prerequisites.
"""
import argparse
import copy
import hashlib
import json
from pathlib import Path
from native_release_evidence import require_sha256

REPORT_ROOT = Path(__file__).resolve().parents[2] / "docs/validation/ggml-no-patch-2026-09-15"
SPARSE_SOURCE = REPORT_ROOT / "sparse-attention-candidate/sparse-attention-build/full-profile.json"
REFERENCE_SOURCE = REPORT_ROOT / "decomposed-f32-reference/raw/profile.json"
SPARSE_SHA256 = "0a29cf8c4b1bd6c4567848314ec246e813f6f145ae8d7c40b14253d7f831639d"
REFERENCE_SHA256 = "c76e800c01bbb4864f9e753ca0543c3718f271d48596d2b1b2773bdc1945039d"
SUITE_OUTPUTS = ["quality.json", "structured-tool-results.json", "serial-tool-workflows.json",
                 "long-context.json", "decode-benchmark.json", "tool-policies.json", "media.json",
                 "invalid-images.json", "audio-rejection.json"]


def read_archived(path, expected_sha256, report=False):
    data = path.read_bytes()
    if hashlib.sha256(data).hexdigest() != expected_sha256:
        raise ValueError("Archived profile SHA256 mismatch: " + str(path))
    value = json.loads(data)
    profile = value["profile"] if report else value
    outputs = [command[command.index("--output") + 1].rsplit("/", 1)[-1] for command in profile["suites"]]
    if outputs != SUITE_OUTPUTS:
        raise ValueError("Archived profile must retain the exact nine-suite order")
    return profile


def normalized_suites(profile):
    suites = copy.deepcopy(profile["suites"])
    for command in suites:
        if "--profile" in command:
            command[command.index("--profile") + 1] = "<profile-label>"
    return suites


def build_profiles(native_sha256, sparse_source=SPARSE_SOURCE, reference_source=REFERENCE_SOURCE):
    require_sha256(native_sha256)
    sparse = read_archived(sparse_source, SPARSE_SHA256)
    reference = read_archived(reference_source, REFERENCE_SHA256, report=True)
    if normalized_suites(sparse) != normalized_suites(reference):
        raise ValueError("Sparse/reference archived workloads differ beyond profile labels")
    if sparse["env"]["TS_DSV4_FA"] != "1" or sparse["env"]["TS_DSV41_SPARSE_FA"] != "1" or reference["env"]["TS_DSV4_FA"] != "0":
        raise ValueError("Archived attention dispatch settings are inconsistent")
    sources = {"sparse": {"path": str(sparse_source), "sha256": SPARSE_SHA256},
               "reference": {"path": str(reference_source), "sha256": REFERENCE_SHA256,
                             "previous_execution": "Failed at 64K after CUDA OOM; five later suites unrun."}}
    profiles = {}
    for name, source, source_key in (("accurate-layer7", sparse, "sparse"),
                                     ("decomposed-control-layer7", reference, "reference"),
                                     ("accurate-expert-tp7", sparse, "sparse")):
        profile = copy.deepcopy(source)
        profile["id"] = "deepseek41-q4km-" + name + "-full-cpumoe12-f16"
        profile["expected_native_sha256"] = native_sha256
        profile["native_policy"] = "exact"
        profile["plan_status"] = "prepared-not-executed"
        profile["archived_input"] = copy.deepcopy(sources[source_key])
        profile["archived_input"]["rebound_fields"] = ["id", "expected_native_sha256", "suite --profile labels"]
        for command in profile["suites"]:
            if "--profile" in command:
                command[command.index("--profile") + 1] = profile["id"]
        if name == "accurate-expert-tp7":
            profile["env"]["TS_DSV41_TP"] = "7"
            profile["archived_input"]["rebound_fields"] += ["env.TS_DSV41_TP"]
            profile["capacity_review"] = {
                "status": "reviewed-for-first-run-runtime-capacity-unqualified",
                "gpu_count": 7, "cpu_expert_layers": 12, "max_context": 65536,
                "max_running_sequences": 4, "native_microbatch": 512,
                "basis": "Prior layer-split resident weights 218.1 GiB average 31.16 GiB/rank; routed-expert strips only. Attention/shared/KV remain layer-placed, and loader prices exact typed strips before packing. This is a planning estimate, not measured TP residency.",
                "execution_gates": ["strict seven-GPU native test", "actual expert-strip and layer-placement banners",
                                    "per-rank free/resident/peak memory", "cgroup 326.9 GiB bound", "all nine original suites"],
            }
            profile["scope"] = "All original nine sparse suites, now routed-expert TP7/CPU12. Native KV remains F16; learned DSpark support is not inferred."
        profiles[name] = profile
    return profiles, sources


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--native-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--sparse-source-profile", type=Path, default=SPARSE_SOURCE,
                        help="Exact archived input, verified against its immutable digest")
    parser.add_argument("--decomposed-source-report", type=Path, default=REFERENCE_SOURCE,
                        help="Exact archived lifecycle report containing the full reference profile")
    args = parser.parse_args()
    try:
        profiles, sources = build_profiles(args.native_sha256, args.sparse_source_profile, args.decomposed_source_report)
    except (ValueError, KeyError) as error:
        parser.error(str(error))
    if args.output.exists():
        parser.error("Refusing to overwrite an existing profile directory")
    args.output.mkdir(parents=True)
    for name, profile in profiles.items():
        path = args.output / (name + ".json")
        path.write_text(json.dumps(profile, indent=2) + "\n", encoding="utf-8")
        print(path)
    manifest = {"status": "prepared-not-executed", "qualified": False, "native_sha256": args.native_sha256,
                "archived_inputs": sources, "profile_names": list(profiles), "suite_count_per_profile": 9}
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
