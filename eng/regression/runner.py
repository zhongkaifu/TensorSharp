"""Sequential orchestration; planning never loads models or downloads payloads."""
from __future__ import annotations

import argparse
from dataclasses import asdict, is_dataclass, replace
from datetime import datetime, timezone
import fnmatch
import json
import os
from pathlib import Path
import shutil
import sys
import time
import uuid

from .audit import audit_catalog
from .configs import read_config
from .downloads import DownloadPolicy, PrepareError, prepare_files
from .hardware import detect_hardware, plan_resources
from .profiles import CPU_MOE_BACKENDS, annotate_resource_plan, residency_profile
from .reports import aggregate_exit, atomic_json, file_hash, runtime_identity, sanitize, source_identity, write_reports
from .scenarios import availability, run_scenario
from .processes import (UNSAFE_RESOURCES_REASON, ProcessContainmentError,
                        containment_failure_evidence, unsafe_process_state)

GIB = 1024 ** 3
SAFE_OPTIONS = {"model", "mmproj", "draft-model", "tokenizer", "kv-cache-dtype", "embedding-threads",
                "embeddings", "model-type", "qwen-image-variant", "qwen-image-vae", "qwen-image-vl",
                "qwen-image-mmproj", "video-dit2", "video-vae", "video-text-encoder", "audio-vae",
                "video-frames", "video-fps", "video-upscaler", "spec"}
MEDIA_SCENARIOS = {"cli_image", "cli_audio", "cli_video"}
BACKEND_PREFERENCE = ("ggml_cuda", "cuda", "ggml_metal", "mlx", "ggml_vulkan", "ggml_cpu", "cpu")


def plain(value):
    if is_dataclass(value):
        return plain(asdict(value))
    if isinstance(value, dict):
        return {k: plain(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [plain(v) for v in value]
    return str(value) if isinstance(value, Path) else value


def command_for(path):
    if not path:
        return None
    path = Path(path).resolve()
    if not path.is_file():
        return None
    if path.suffix.lower() == ".dll":
        dotnet = shutil.which("dotnet")
        return [dotnet, str(path)] if dotnet else None
    return [str(path)]


def find_binary(repo, explicit, project, name):
    if explicit:
        return command_for(explicit)
    for folder in (repo / project / "bin", repo / project / "bin" / "Release" / "net10.0"):
        for extension in ((".exe", ".dll") if os.name == "nt" else ("", ".dll")):
            found = command_for(folder / (name + extension))
            if found:
                return found
    return None


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("audit", "plan", "run"), default="plan")
    parser.add_argument("--repo", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument("--engine-repo", type=Path, help="Source checkout used to build explicitly supplied hosts; recorded separately from harness source")
    parser.add_argument("--catalog", type=Path)
    parser.add_argument("--scenario-profiles", type=Path, help="Typed SHA-pinned fixture/companion profiles for extended scenarios; never arbitrary commands")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--models", default="*", help="Comma-separated preset ID glob patterns")
    parser.add_argument("--scenarios", default="*", help="Comma-separated scenario ID glob patterns")
    parser.add_argument("--backend", default="auto", help="auto, all, or comma-separated exact backend IDs")
    parser.add_argument("--model-root", type=Path, default=None)
    parser.add_argument("--existing-root", action="append", type=Path, default=[], help="Search existing weights by unique filename; never write/delete here")
    parser.add_argument("--cli", type=Path)
    parser.add_argument("--server", type=Path)
    parser.add_argument("--http-client", type=Path, help="Prebuilt TensorSharp.LocalChat example for live HTTP SDK scenarios; never built implicitly")
    parser.add_argument("--logit-probe", type=Path)
    parser.add_argument("--benchmark", type=Path, help="Prebuilt AgentTurnBench for explicitly supported n-gram parity cells")
    parser.add_argument("--runtime-dir", action="append", type=Path, default=[], help="Additional native runtime/build metadata directories")
    parser.add_argument("--media-dir", type=Path)
    parser.add_argument("--allow-download", action="store_true")
    parser.add_argument("--probe-network", action="store_true", help="Resolve missing file sizes by HEAD; plan still downloads no payloads")
    parser.add_argument("--max-download-gib", type=float, default=20)
    parser.add_argument("--disk-headroom-gib", type=float, default=5)
    parser.add_argument("--ram-budget-gib", type=float)
    parser.add_argument("--vram-budget-gib", type=float)
    parser.add_argument("--ram-required-gib", type=float)
    parser.add_argument("--vram-required-gib", type=float)
    parser.add_argument("--cpu-moe", action="store_true", help="Explicit GGML host-expert profile for qwen35/moe or qwen4exp/moe; requires both --ram-required-gib and --vram-required-gib")
    parser.add_argument("--gpu-index", type=int, default=0)
    parser.add_argument("--assume-backend-available", action="store_true", help="Override detection only; actual runtime backend still must match")
    parser.add_argument("--max-models", type=int, default=0, help="0 means no count limit; download budget always applies")
    parser.add_argument("--timeout-seconds", type=float, default=600)
    parser.add_argument("--context-tokens", type=int, default=2048, help="Explicit model context, 1..2048 (default 2048); insufficient fixed fixtures are skipped without resizing")
    parser.add_argument("--strict", action="store_true", help="Exit 3 for any selected skip/untested coverage")
    parser.add_argument("--include-suites", action="store_true", help="Include existing local .NET and client correctness suites")
    parser.add_argument("--native-test-build-dir", type=Path, help="Explicit prebuilt CTest directory from this checkout; no implicit native build")
    parser.add_argument("--native-test-backend", choices=("cpu", "cuda", "metal", "vulkan"), default="cpu")
    parser.add_argument("--native-test-filter", help="Optional regex narrowing native test names; unverified device tests remain untested")
    parser.add_argument("--allow-builds", action="store_true", help="Allow managed suite builds/restores only; native backends must be prebuilt")
    args = parser.parse_args(argv)
    from .hardware import BACKENDS
    if args.backend not in {"auto", "all"} and any(b not in BACKENDS for b in args.backend.split(",")):
        parser.error("--backend contains an unknown backend ID")
    for key in ("max_download_gib", "disk_headroom_gib", "ram_budget_gib", "vram_budget_gib", "ram_required_gib", "vram_required_gib", "timeout_seconds"):
        value = getattr(args, key)
        if value is not None and (not 0 <= value < float("inf") or (key == "timeout_seconds" and value == 0)):
            parser.error(f"--{key.replace('_', '-')} must be finite and nonnegative")
    if args.max_models < 0 or args.gpu_index < 0:
        parser.error("--max-models and --gpu-index must be nonnegative")
    if not 1 <= args.context_tokens <= 2048:
        parser.error("--context-tokens must be an integer in 1..2048")
    for key in ("ram_budget_gib", "vram_budget_gib", "ram_required_gib", "vram_required_gib"):
        if getattr(args, key) == 0:
            parser.error(f"--{key.replace('_', '-')} must be positive")
    if args.cpu_moe and (args.ram_required_gib is None or args.vram_required_gib is None):
        parser.error("--cpu-moe requires explicit --ram-required-gib and --vram-required-gib (including on ggml_cpu)")
    return args


def selected(identifier, patterns):
    return any(fnmatch.fnmatchcase(identifier, pattern.strip()) for pattern in patterns.split(",") if pattern.strip())


def select_backends(requested, supported, hardware):
    if requested == "all":
        return list(supported)
    if requested != "auto":
        return requested.split(",")
    candidates = [backend for backend in BACKEND_PREFERENCE
                  if backend in supported and hardware.get("backends", {}).get(backend, {}).get("available")]
    # Auto chooses one available backend, and records the choice. Explicit requests never fall back.
    return candidates[:1] or list(supported[:1])


def scenario_supports_backend(scenario, backend):
    return "backends" not in scenario or backend in scenario["backends"]


def existing_index(roots):
    index = {}
    for root in roots:
        if not root.is_dir():
            continue
        for path in root.rglob("*"):
            if path.is_file() and path.suffix.lower() in {".gguf", ".safetensors", ".json", ".txt"}:
                index.setdefault(path.name, []).append(path.resolve())
    return index


def use_existing(resolved, index):
    replacements, files = {}, []
    for spec in resolved.files:
        candidates = index.get(spec.path.name, [])
        if not spec.path.exists() and len(candidates) == 1:
            replacements[str(spec.path)] = str(candidates[0])
            spec = replace(spec, path=candidates[0])
        files.append(spec)
    options = {key: replacements.get(value, value) if isinstance(value, str) else value
               for key, value in resolved.options.items()}
    # Split models must remain co-located, including the primary shard.
    groups = {}
    for spec in files:
        if "-of-" in spec.path.name and spec.path.suffix == ".gguf":
            groups.setdefault(spec.option, set()).add(spec.path.parent)
    if any(len(parents) != 1 for parents in groups.values()):
        raise ValueError("split model shards resolved to different directories")
    return options, files


def effective_options(options, scenario_ids, backend):
    result = {key: value for key, value in options.items() if key in SAFE_OPTIONS}
    if not any(s in MEDIA_SCENARIOS for s in scenario_ids):
        result.pop("mmproj", None)
    # Drafting is an independent coverage recipe, never silently enabled by a text test.
    if not any(s != "speculation_ngram" and ("spec" in s or "mtp" in s) for s in scenario_ids):
        result.pop("draft-model", None)
        result.pop("spec", None)
    if "image_generation_smoke" in scenario_ids and "image_editing" not in scenario_ids:
        result.pop("qwen-image-mmproj", None)
    return result


def model_files(files, options):
    return [spec for spec in files if spec.option in options]


def required_scenario_files(files, scenarios):
    # A projector optional for text becomes mandatory when a selected fixture
    # actually exercises media. Its full memory/download cost must be planned.
    needs_media = any(s["id"] in MEDIA_SCENARIOS for s in scenarios)
    needs_image = any(s["id"] == "image_generation_smoke" for s in scenarios)
    return [replace(spec, optional=False) if (needs_media and spec.option == "mmproj") or
            (needs_image and spec.option in {"qwen-image-vae", "qwen-image-vl"}) else spec for spec in files]


def catalog_file_sizes(files, preset):
    sizes = preset.get("file_sizes", {})
    return [replace(spec, size_bytes=sizes[spec.path.name])
            if spec.size_bytes is None and spec.path.name in sizes else spec for spec in files]


def preset_budgets(preset, overrides):
    declared = preset.get("resource_requirements", {})
    values = {}
    for source, target in (("ram_gib", "ram_required_gib"), ("vram_gib", "vram_required_gib")):
        if source in declared:
            values[target] = declared[source]
    if "scratch_gib" in declared:
        values["scratch_bytes"] = int(declared["scratch_gib"] * GIB)
    return {**values, **overrides}


def offload_profile(args, preset=None, backend=None):
    return residency_profile(args.cpu_moe, args.ram_required_gib, args.vram_required_gib, preset, backend)


def plan_model_resources(args, preset, weight_bytes, backend, hardware, overrides, scenarios=()):
    profile = offload_profile(args, preset, backend)
    if not profile["supported"]:
        return {"eligible": False, "reason": profile["reason"], "backend": backend,
                "offload_policy": profile, "estimates": {}, "warnings": []}
    resource_overrides = dict(overrides)
    resident_models = None
    if (preset.get("id") == "embedding-minilm" and backend == "cpu"
            and any(s["id"] == "retrieval_vector_cpu" for s in scenarios)):
        known_size = type(weight_bytes) is int and weight_bytes > 0
        explicit_ram = "ram_required_gib" in resource_overrides
        if known_size and not explicit_ram:
            # The primary embedding endpoint and knowledge provider own separate
            # models, even though both load the same verified file. Keep unique
            # on-disk/transfer bytes unchanged and reserve workspace per instance.
            resource_overrides["ram_required_gib"] = 2 * (weight_bytes * 1.25 + 2 * GIB) / GIB
        resident_models = {
            "scenario": "retrieval_vector_cpu", "instances": 2, "backend": "cpu",
            "unique_weight_bytes": weight_bytes,
            "resident_weight_bytes": 2 * weight_bytes if known_size else None,
            "file_copies_for_download_and_disk": 1,
            "ram_estimate_source": "explicit_total_override" if explicit_ram else "per_instance_conservative_default",
            "ram_formula": "2 * (1.25 * unique_weight_bytes + 2 GiB)",
            "ram_override_scope": "total for both independently resident CPU encoders and their workspace",
            "source_evidence": ["TensorSharp.Server.Host/Program.cs",
                                "TensorSharp.Chat/Knowledge/KnowledgeRuntime.cs",
                                "TensorSharp.Chat/Knowledge/GgufKnowledgeEmbeddingProvider.cs"],
        }
    result = annotate_resource_plan(plan_resources(weight_bytes, backend, hardware, overrides=resource_overrides), profile)
    if resident_models is not None:
        result["resident_models"] = resident_models
    return result


def sizes_for(files, policy, network=False):
    """Return (final weight footprint, remaining transfer/additional disk bytes).

    Existing partial bytes already occupy disk and do not need transferring again.
    Only validated suite sidecars may reduce the remaining-byte budget. The full
    final footprint still sizes RAM/VRAM; all final contents require SHA checking.
    """
    from .downloads import probe_files
    files = list(files)
    sizes, remaining = [], 0
    for spec, probe in zip(files, probe_files(files, policy, network=network)):
        if probe.get("status") == "blocked":
            failure = probe.get("reason") or {}
            raise PrepareError(failure.get("stage", "probe"), failure.get("code", "probe_failed"), spec.path,
                               failure.get("message", "Model metadata probe failed"))
        size, needed = probe.get("size_bytes"), probe.get("remaining_bytes")
        if size is None or size <= 0 or needed is None:
            return None, None
        sizes.append(size)
        remaining += needed
    return sum(sizes), remaining


def result_stub(preset, backend, scenario, status, reason, **details):
    return {"id": f"{preset['id']}::{backend}::{scenario['id']}", "preset": preset["id"],
            "family": preset.get("family"), "backend": backend, "scenario": scenario["id"],
            "status": status, "reason": reason, **details}


def unbound_catalog_recipes(catalog, patterns, profiles=()):
    """Disclose selected recipes with no model binding, without inventing one."""
    bound = {scenario for preset in catalog["presets"] for scenario in preset.get("scenarios", [])}
    profiled = {profile["scenario"] for profile in profiles}
    return [{"id": "catalog::" + scenario["id"], "scope": "catalog", "scenario": scenario["id"],
             "status": "untested", "stage": "coverage_inventory",
             "reason": ("Typed executor needs a selected --scenario-profiles entry with approved fixtures and resource requirements."
                        if scenario.get("execution") == "profile" else
                        "Catalog recipe has no preset or global execution binding; its documented requirements need separate qualification."),
             "catalog_status": scenario.get("status"), "entry_point": scenario.get("entry_point"),
             "recipe": scenario.get("recipe"), "requirements": scenario.get("requirements", []),
             "source_evidence": scenario.get("evidence", []),
             "unexecuted_assertions": scenario.get("assertions", [])}
            for scenario in catalog["scenarios"]
            if ((scenario.get("status") == "recipe" and scenario["id"] not in bound)
                or (scenario.get("execution") == "profile" and scenario["id"] not in profiled and scenario["id"] not in bound))
            and selected(scenario["id"], patterns)]


def missing_profile_cases(preset, scenarios, args, profiles):
    """A profile for one model/device never hides another selected matrix gap."""
    results = []
    backends = ["auto"] if args.backend == "auto" else select_backends(args.backend, preset.get("backends", []), {})
    for identifier in preset.get("scenarios", []):
        scenario = scenarios[identifier]
        if scenario.get("execution") != "profile" or not selected(identifier, args.scenarios):
            continue
        for backend in backends:
            matching = [p for p in profiles if p["preset"] == preset["id"] and p["scenario"] == identifier
                        and (backend == "auto" or p["backend"] == backend)]
            if not matching:
                results.append(result_stub(preset, backend, scenario, "untested",
                    "No selected typed profile for this model/backend/scenario; provide --scenario-profiles with approved fixtures and total peak budgets",
                    stage="profile_requirements", entry_point=scenario.get("entry_point"),
                    unexecuted_assertions=scenario.get("assertions", [])))
    return results


def record_execution_halt(report, result):
    evidence = containment_failure_evidence(result)
    if evidence is None:
        return None
    result.update(status="fail", containment_failure=True)
    result["reason"] = str(result.get("reason", "")) + "; " + UNSAFE_RESOURCES_REASON
    report.setdefault("execution_halt", {"reason": UNSAFE_RESOURCES_REASON,
        "trigger_case": result["id"], "process_evidence": evidence})
    return report["execution_halt"]


def append_halted_cases(report, preset, backends, cases, halt):
    existing = {result["id"] for result in report["results"]}
    for backend in backends:
        for scenario in cases:
            result = result_stub(preset, backend, scenario, "untested", UNSAFE_RESOURCES_REASON,
                                 stage="execution_halted", halt_trigger=halt["trigger_case"])
            if result["id"] not in existing:
                report["results"].append(result)
                existing.add(result["id"])


def reproduce_args(args, repo, root, preset, backend, scenario):
    command = [sys.executable, str(repo / "eng/run-local-regression.py"), "--mode", "run",
               "--models", preset["id"], "--backend", backend, "--scenarios", scenario["id"], "--model-root", str(root)]
    if args.output is not None:
        # Keep diagnostics on the caller's chosen volume without reusing a
        # nonempty evidence directory. The generated argv is saved per result.
        fresh = args.output.resolve().parent / ("repro-" + preset["id"] + "-" + scenario["id"] + "-" + uuid.uuid4().hex[:8])
        command += ["--output", str(fresh)]
    for key in ("cli", "server", "http_client", "logit_probe", "benchmark", "engine_repo", "media_dir", "ram_budget_gib", "vram_budget_gib", "ram_required_gib", "vram_required_gib", "gpu_index", "disk_headroom_gib", "max_download_gib", "timeout_seconds", "context_tokens", "catalog", "scenario_profiles"):
        value = getattr(args, key)
        if value is not None:
            command += ["--" + key.replace("_", "-"), str(value)]
    for key in ("existing_root", "runtime_dir"):
        for value in getattr(args, key):
            command += ["--" + key.replace("_", "-"), str(value)]
    for key in ("assume_backend_available", "allow_download", "probe_network", "cpu_moe"):
        if getattr(args, key):
            command.append("--" + key.replace("_", "-"))
    return command


def main(argv=None):
    args = parse_args(argv)
    repo = args.repo.resolve()
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "-" + uuid.uuid4().hex[:8]
    output = (args.output or repo / "artifacts" / "local-regression" / run_id).resolve()
    if output.exists() and any(output.iterdir()):
        print("Output must be empty: each run has independent evidence and cannot reuse stale passes.", file=sys.stderr)
        return 2
    output.mkdir(parents=True, exist_ok=True)
    report = {"schema_version": 1, "run_id": run_id, "mode": args.mode,
              "started_utc": datetime.now(timezone.utc).isoformat(), "source": source_identity(repo),
              "engine_source": {"identity": source_identity(args.engine_repo.resolve()) if args.engine_repo else None,
                                "provenance": "caller-declared build checkout; binary hashes are recorded independently" if args.engine_repo else "unspecified; harness checkout is not proof of deployed host build source"},
              "invocation": sanitize([sys.executable, str(repo / "eng/run-local-regression.py"), *(argv if argv is not None else sys.argv[1:])]),
              "results": [], "limitations": [
                  "Only this local OS and selected backend can be tested; other platforms are unrun.",
                  "Availability detection is not backend coverage. Runtime cases require positive backend identity.",
                  "CPU MoE planning uses explicit caller-supplied peak requirements; runtime residency and memory use remain workload-dependent." if args.cpu_moe else
                  "Memory planning reserves workspace/KV headroom and charges all weight bytes; the harness requests no automatic partial offload.",
                  "Generation correctness checks use deterministic task fixtures, not a comprehensive language quality benchmark.",
                  "Typed profile executors require their explicit approved fixtures, dependencies and hardware; missing conditions remain skip/untested.",
                  "Model payloads are never downloaded in audit/plan modes. No model files are deleted or published."]}
    cancelled = False
    try:
        catalog_path = args.catalog or repo / "config/regression/catalog.json"
        catalog = json.loads(catalog_path.read_text(encoding="utf-8"))
        from .scenario_profiles import load_profiles
        scenario_profiles = load_profiles(args.scenario_profiles, catalog)
        report["scenario_profiles"] = scenario_profiles
        report["catalog"] = {"path": str(catalog_path), "sha256": file_hash(catalog_path)}
        audit = audit_catalog(repo, catalog)
        report["audit"] = audit
        report["results"].append({"id": "config-coverage-audit", "status": audit["status"],
                                  "reason": "; ".join(audit["errors"]) or f"Inventoried {audit['config_count']} configs and {audit['scenario_count']} scenarios"})
        if audit["status"] == "fail":
            report["exit_code"] = 2
            write_reports(output, report)
            print(f"Config audit failed. See {output / 'report.md'}")
            return 2
        cli = find_binary(repo, args.cli, "TensorSharp.Cli", "TensorSharp.Cli")
        server = find_binary(repo, args.server, "TensorSharp.Server.Host", "TensorSharp.Server.Host")
        http_client = find_binary(repo, args.http_client, "examples/dotnet-local-chat", "TensorSharp.LocalChat")
        logit = find_binary(repo, args.logit_probe, "eng/ForcedLogitProbe", "ForcedLogitProbe")
        benchmark = find_binary(repo, args.benchmark, "benchmarks/AgentTurnBench", "AgentTurnBench")
        commands = {"cli_command": cli, "server_command": server, "http_client_command": http_client, "logit_command": logit, "benchmark_command": benchmark}
        fixture_runtime_dirs = {str(Path(fixture["path"]).parent) for profile in scenario_profiles
            if selected(profile["preset"], args.models) and selected(profile["scenario"], args.scenarios)
            for fixture in profile["fixtures"].values() if Path(fixture["path"]).suffix.lower() in {".dll", ".exe"}}
        runtime_dirs = sorted({str(Path(command[-1]).parent) for command in commands.values() if command}
            | {str(p.resolve()) for p in args.runtime_dir} | fixture_runtime_dirs)
        root = (args.model_root or Path(os.environ.get("TENSORSHARP_MODELS", str(repo / "models")))).resolve()
        hardware = detect_hardware(root, runtime_dirs=runtime_dirs)
        report["hardware"] = hardware
        if args.mode == "run":
            report["runtime_assets"] = runtime_identity(hardware)
        report["binaries"] = {key: {"command": command, "sha256": file_hash(command[-1])}
                              if command else None for key, command in commands.items()}
        if args.mode == "audit":
            report["exit_code"] = aggregate_exit(report["results"], strict=args.strict)
            write_reports(output, report)
            print(f"Audit complete: {output / 'report.md'}")
            return report["exit_code"]
        policy = DownloadPolicy(cache_root=root, max_download_bytes=int(args.max_download_gib * GIB),
                                disk_headroom_bytes=int(args.disk_headroom_gib * GIB),
                                allow_download=args.allow_download and args.mode == "run", token=os.environ.get("HF_TOKEN"))
        index = existing_index(args.existing_root)
        scenarios = {s["id"]: s for s in catalog["scenarios"]}
        overrides = {key: getattr(args, key) for key in ("ram_budget_gib", "vram_budget_gib", "ram_required_gib", "vram_required_gib", "gpu_index", "disk_headroom_gib", "assume_backend_available")
                     if getattr(args, key) is not None}
        report["policy"] = {"model_root": str(root), "max_download_bytes": int(args.max_download_gib * GIB),
                            "requested_backend": args.backend, "overrides": overrides, "allow_download": args.allow_download,
                            "context_tokens": args.context_tokens, "model_environment": {"MAX_CONTEXT": str(args.context_tokens)},
                            "context_policy": "fixed explicit context; fixtures unchanged; no automatic memory estimate reduction",
                            "offload": offload_profile(args), "sampling": {"seed": 42, "temperature": 0}}
        presets = [p for p in catalog["presets"] if selected(p["id"], args.models)]
        if not presets:
            raise ValueError("model selector matched no catalog presets")
        selected_profiles = [p for p in scenario_profiles if selected(p["preset"], args.models)
            and selected(p["scenario"], args.scenarios)
            and (args.backend in {"auto", "all"} or p["backend"] in args.backend.split(","))]
        report["results"].extend(unbound_catalog_recipes(catalog, args.scenarios, selected_profiles))
        download_reserved, executed_models = 0, set()
        halt = None
        if args.mode == "run" and unsafe_process_state() is not None:
            previous = {"id": "prior-containment-failure", "status": "fail", "reason": "Earlier process cleanup remains unverified",
                        "containment_failure": True, "teardown": unsafe_process_state().get("teardown")}
            report["results"].append(previous)
            halt = record_execution_halt(report, previous)
        for preset in presets:
            report["results"].extend(missing_profile_cases(preset, scenarios, args, selected_profiles))
            model_overrides = preset_budgets(preset, overrides)
            case_defs = [scenarios[s] for s in preset.get("scenarios", []) if selected(s, args.scenarios)
                         and scenarios[s].get("execution") != "profile"]
            if not case_defs:
                continue
            if halt:
                # Do not resolve configs, probe metadata, hash/download weights or
                # select another model based on resources which may still be owned.
                remaining_backends = ["auto"] if args.backend == "auto" else select_backends(args.backend, preset.get("backends", []), hardware) or ["none"]
                append_halted_cases(report, preset, remaining_backends, case_defs, halt)
                continue
            profile = offload_profile(args, preset)
            if not profile["supported"]:
                rejected_backends = ["auto"] if args.backend == "auto" else select_backends(args.backend, preset.get("backends", []), hardware) or ["none"]
                for backend in rejected_backends:
                    for scenario in case_defs:
                        report["results"].append(result_stub(preset, backend, scenario, "unsupported", profile["reason"], offload_policy=profile))
                continue
            supported_backends = preset.get("backends", [])
            if args.cpu_moe and args.backend == "auto":
                supported_backends = [b for b in supported_backends if b in CPU_MOE_BACKENDS]
                if not supported_backends:
                    for scenario in case_defs:
                        report["results"].append(result_stub(preset, "auto", scenario, "unsupported",
                            "--cpu-moe has no source-supported GGML backend in this preset", offload_policy=profile))
                    continue
            backends = select_backends(args.backend, supported_backends, hardware) or ["none"]
            if args.backend == "auto" and preset.get("status") != "unsupported" and any(s.get("status") == "automated" for s in case_defs):
                # Evaluate resource fit before choosing an automatic backend. This is
                # a recorded planning decision, never a runtime fallback.
                preliminary_files = []
                prerequisite_failures = {}
                planning_context = {"backend": None, "stage": "config_resolution"}
                try:
                    preliminary = read_config(repo / preset["config"], root)
                    if not preliminary.unavailable:
                        all_options, all_files = use_existing(preliminary, index)
                        attempts, size_cache = [], {}
                        for candidate_backend in (b for b in BACKEND_PREFERENCE if b in preset.get("backends", [])):
                            planning_context = {"backend": candidate_backend, "stage": "scenario_prerequisites"}
                            preliminary_files = []
                            preliminary_context = {"repo": repo, "backend": candidate_backend, "preparing_model": True,
                                "preset_id": preset["id"], "model_family": preset["family"],
                                "context_tokens": args.context_tokens,
                                "model_options": all_options, "model_path": all_options.get("model"),
                                "media_dir": args.media_dir, "offload_policy": offload_profile(args, preset, candidate_backend), **commands}
                            preliminary_cases, case_rejections = [], {}
                            prerequisite_failures = case_rejections
                            for scenario in case_defs:
                                if scenario.get("status") != "automated":
                                    continue
                                if not scenario_supports_backend(scenario, candidate_backend):
                                    case_rejections[scenario["id"]] = "outside this scenario's adapter qualification backend scope"
                                    continue
                                ready, why = availability(scenario, preliminary_context)
                                if ready:
                                    preliminary_cases.append(scenario)
                                else:
                                    case_rejections[scenario["id"]] = why
                            planning_context["stage"] = "file_planning"
                            preliminary_options = effective_options(all_options, [s["id"] for s in preliminary_cases], candidate_backend)
                            preliminary_files = catalog_file_sizes(model_files(all_files, preliminary_options), preset)
                            preliminary_files = required_scenario_files(preliminary_files, preliminary_cases)
                            if preliminary_cases:
                                file_key = tuple(preliminary_files)
                                if file_key not in size_cache:
                                    size_cache[file_key] = sizes_for(preliminary_files, policy,
                                        network=args.probe_network or (args.mode == "run" and args.allow_download))
                                total, missing = size_cache[file_key]
                            else:
                                total, missing = None, None
                            planning_context["stage"] = "resource_planning"
                            attempt = plan_model_resources(args, preset, total, candidate_backend, hardware,
                                overrides={**model_overrides, "download_bytes": missing or 0}, scenarios=preliminary_cases)
                            attempts.append({**attempt, "runnable_scenarios": [s["id"] for s in preliminary_cases],
                                "scenario_rejections": case_rejections, "coverage_count": len(preliminary_cases)})
                        eligible = [attempt for attempt in attempts if attempt["eligible"]]
                        # Maximize selected scenario coverage, then retain backend
                        # preference on ties. A CPU-only recipe cannot disappear
                        # merely because the preset also supports a GPU backend.
                        choices = eligible or [attempt for attempt in attempts if attempt.get("offload_policy", {}).get("supported")] or attempts
                        if choices:
                            backends = [max(choices, key=lambda attempt: attempt["coverage_count"])["backend"]]
                        report.setdefault("backend_selections", []).append({"preset": preset["id"], "requested": "auto",
                            "selected": backends[0], "selection_policy": "most runnable selected scenarios within resource budgets; backend preference breaks ties",
                            "attempts": attempts})
                except (ValueError, OSError, RuntimeError) as error:
                    # A single bad source or preset must not prevent all other
                    # selected models from being planned/tested. Do not retry it
                    # for each backend or turn untouched recipes into failures.
                    details = error.as_dict() if isinstance(error, PrepareError) else {
                        "stage": "planning", "code": "planning_error", "path": str(repo / preset["config"]), "message": str(error)}
                    report.setdefault("backend_selections", []).append({"preset": preset["id"], "requested": "auto",
                        "selected": None, "error": details, "planning_context": planning_context,
                        "scenario_rejections": prerequisite_failures})
                    for scenario in case_defs:
                        capability = scenario.get("status")
                        if capability == "unsupported":
                            status, reason = "unsupported", scenario.get("reason", "Implementation unavailable")
                        elif capability != "automated":
                            status, reason = "untested", "Supported recipe needs a dedicated fixture/adapter: " + str(scenario.get("recipe", "see coverage catalog"))
                        elif scenario["id"] in prerequisite_failures:
                            status, reason = "skip", prerequisite_failures[scenario["id"]]
                        else:
                            status, reason = "fail", str(error)
                        report["results"].append(result_stub(preset, "auto", scenario, status, reason,
                            stage=details["stage"], error_code=details["code"], error=details,
                            planning_context=planning_context,
                            required_files=[plain(spec) for spec in preliminary_files],
                            reproduce=reproduce_args(args, repo, root, preset, "auto", scenario)))
                    write_reports(output, report, bundle=False)
                    continue
            for backend in backends:
                if halt:
                    append_halted_cases(report, preset, [backend], case_defs, halt)
                    continue
                profile = offload_profile(args, preset, backend)
                model_env = {"CUDA_VISIBLE_DEVICES": str(args.gpu_index)} if backend in {"cuda", "ggml_cuda"} else {}
                model_env.update(profile["environment"])
                model_env["MAX_CONTEXT"] = str(args.context_tokens)
                base_context = {"repo": repo, "backend": backend, "artifact_dir": output, "preset_id": preset["id"], "model_family": preset["family"],
                                "timeout_seconds": args.timeout_seconds, "media_dir": args.media_dir,
                                "disk_headroom_gib": args.disk_headroom_gib,
                                "context_tokens": args.context_tokens,
                                "model_layout": preset.get("layout"), "offload_policy": profile,
                                "seed": 42, "env": model_env, **commands}
                candidates = []
                for scenario in case_defs:
                    if scenario.get("status") == "unsupported" or preset.get("kind") == "unsupported" or preset.get("status") == "unsupported":
                        status, reason = "unsupported", preset.get("reason") or scenario.get("reason", "Implementation unavailable")
                    elif backend not in preset.get("backends", []):
                        status, reason = "unsupported", f"{backend} is outside this preset's source-supported backend matrix"
                    elif not scenario_supports_backend(scenario, backend):
                        status, reason = "unsupported", f"{backend} is outside this scenario's adapter qualification backend scope"
                    elif not profile["supported"]:
                        status, reason = "unsupported", profile["reason"]
                    elif scenario.get("status") != "automated":
                        status, reason = "untested", "Supported recipe needs a dedicated fixture/adapter: " + str(scenario.get("recipe", "see coverage catalog"))
                    else:
                        candidates.append(scenario)
                        continue
                    report["results"].append(result_stub(preset, backend, scenario, status, reason, offload_policy=profile))
                if not candidates:
                    continue
                files, config_path, resources, active_scenario = [], None, None, None
                try:
                    resolved = read_config(repo / preset["config"], root)
                    if resolved.unavailable:
                        for scenario in candidates:
                            report["results"].append(result_stub(preset, backend, scenario, "unsupported", resolved.unavailable))
                        continue
                    options, files = use_existing(resolved, index)
                    # Resolve fixture/binary prerequisites before choosing files.
                    # Missing image/audio/video fixtures must not fetch a projector
                    # or prevent otherwise eligible text tests from running.
                    prerequisite_context = {**base_context, "preparing_model": True,
                        "model_path": options.get("model"), "model_options": options}
                    available_candidates = []
                    for scenario in candidates:
                        ready, why = availability(scenario, prerequisite_context)
                        if ready:
                            available_candidates.append(scenario)
                        else:
                            report["results"].append(result_stub(preset, backend, scenario, "skip", why,
                                reproduce=reproduce_args(args, repo, root, preset, backend, scenario)))
                    candidates = available_candidates
                    if not candidates:
                        continue
                    options = effective_options(options, [s["id"] for s in candidates], backend)
                    files = catalog_file_sizes(model_files(files, options), preset)
                    files = required_scenario_files(files, candidates)
                    if not files:
                        raise ValueError("preset has no verifiable model files")
                    model_dir = output / "cases" / f"{preset['id']}--{backend}"
                    model_dir.mkdir(parents=True, exist_ok=True)
                    config_path = model_dir / "effective-config.json"
                    atomic_json(config_path, options)
                    context = {**base_context, "config_path": config_path, "effective_config_path": config_path,
                               "model_path": options.get("model"), "model_options": options}
                    weight_bytes, missing_bytes = sizes_for(files, policy, network=args.probe_network or (args.mode == "run" and args.allow_download))
                    if args.mode == "run":
                        hardware = detect_hardware(root, runtime_dirs=runtime_dirs)
                    resources = plan_model_resources(args, preset, weight_bytes, backend, hardware,
                                                    overrides={**model_overrides, "download_bytes": missing_bytes or 0}, scenarios=candidates)
                    reason = None
                    if args.max_models and preset["id"] not in executed_models and len(executed_models) >= args.max_models:
                        reason = "selected model count limit reached"
                    elif not resources.get("eligible"):
                        reason = resources.get("reason", "resource budget unknown or exceeded")
                    elif any(not spec.path.is_file() for spec in files) and not args.allow_download:
                        reason = "weights missing; --allow-download is disabled"
                    elif missing_bytes and max(download_reserved, policy.downloaded_bytes) + missing_bytes > args.max_download_gib * GIB:
                        reason = "remaining whole-run download budget insufficient"
                    runnable = []
                    for scenario in candidates:
                        ok, available_reason = availability(scenario, {**context, "preparing_model": True})
                        why = reason or (None if ok else available_reason)
                        if why or args.mode != "run":
                            report["results"].append(result_stub(preset, backend, scenario, "skip" if why else "untested",
                                why or "planned; model payloads and tests were not executed", resource_plan=resources,
                                inputs=scenario.get("fixture"), required_files=[plain(spec) for spec in files]))
                        else:
                            runnable.append(scenario)
                    if not runnable:
                        continue
                    # Reserve across models before starting; no concurrent downloads or GPU tests.
                    download_reserved += missing_bytes or 0
                    executed_models.add(preset["id"])
                    print(f"Preparing {preset['id']} / {backend}", flush=True)
                    identities = prepare_files(files, policy)
                    atomic_json(model_dir / "model-identity.json", plain(identities))
                    skipped_files = {item["path"]: item for item in plain(identities) if item.get("status") == "optional-skipped"}
                    dropped_options = {key for key, value in options.items() if isinstance(value, str) and value in skipped_files}
                    if dropped_options:
                        options = {key: value for key, value in options.items() if key not in dropped_options}
                        atomic_json(config_path, options)
                        context = {**context, "model_path": options.get("model"), "model_options": options}
                    for scenario in runnable:
                        if halt:
                            append_halted_cases(report, preset, [backend], [scenario], halt)
                            continue
                        active_scenario = scenario
                        dependencies = {"model"}
                        if scenario["id"] in MEDIA_SCENARIOS:
                            dependencies.add("mmproj")
                        if "spec" in scenario["id"] or "mtp" in scenario["id"]:
                            dependencies.add("draft-model")
                        missing_dependencies = dependencies & dropped_options
                        if missing_dependencies:
                            report["results"].append(result_stub(preset, backend, scenario, "skip",
                                "Selected scenario companion unavailable after preparation: " + ", ".join(sorted(missing_dependencies)),
                                optional_companions=list(skipped_files.values()), model_identity=plain(identities),
                                resource_plan=resources, reproduce=reproduce_args(args, repo, root, preset, backend, scenario)))
                            continue
                        case_context = {**context, "artifact_dir": model_dir / scenario["id"]}
                        print(f"Running {preset['id']} / {backend} / {scenario['id']}", flush=True)
                        start = time.monotonic()
                        outcome = run_scenario(scenario, case_context)
                        outcome.pop("id", None)
                        result = result_stub(preset, backend, scenario, outcome.pop("status"), outcome.pop("reason", ""),
                                             **outcome)
                        result["resource_plan"] = resources
                        result["context_tokens"] = args.context_tokens
                        result["offload_policy"] = profile
                        result["model_identity"] = plain(identities)
                        result["duration_seconds"] = time.monotonic() - start
                        result["reproduce"] = reproduce_args(args, repo, root, preset, backend, scenario)
                        unreported_cleanup = unsafe_process_state()
                        if unreported_cleanup is not None and containment_failure_evidence(result) is None:
                            # Even an adapter accidentally omitting its process
                            # records cannot clear the central sticky launch guard.
                            result.update(containment_failure=True, teardown=unreported_cleanup.get("teardown"),
                                          containment_evidence=unreported_cleanup)
                        report["results"].append(result)
                        halt = record_execution_halt(report, result) or halt
                        write_reports(output, report, bundle=False)
                except PrepareError as error:
                    for scenario in candidates:
                        if not any(r["id"] == f"{preset['id']}::{backend}::{scenario['id']}" for r in report["results"]):
                            report["results"].append(result_stub(preset, backend, scenario, "fail", str(error), stage=error.stage, error_code=error.code,
                                error=error.as_dict(), required_files=[plain(spec) for spec in files], effective_config=config_path,
                                resource_plan=resources, reproduce=reproduce_args(args, repo, root, preset, backend, scenario)))
                except (ValueError, OSError, RuntimeError) as error:
                    containment = error.evidence if isinstance(error, ProcessContainmentError) else unsafe_process_state()
                    if containment is not None:
                        scenario = active_scenario or candidates[0]
                        failure = result_stub(preset, backend, scenario, "fail", str(error), stage="containment",
                            containment_failure=True, teardown=containment.get("teardown"), process_evidence=containment,
                            required_files=[plain(spec) for spec in files], effective_config=config_path,
                            resource_plan=resources, reproduce=reproduce_args(args, repo, root, preset, backend, scenario))
                        report["results"].append(failure)
                        halt = record_execution_halt(report, failure)
                    else:
                        for scenario in candidates:
                            if not any(r["id"] == f"{preset['id']}::{backend}::{scenario['id']}" for r in report["results"]):
                                report["results"].append(result_stub(preset, backend, scenario, "fail", str(error), stage="orchestration",
                                    required_files=[plain(spec) for spec in files], effective_config=config_path,
                                    resource_plan=resources, reproduce=reproduce_args(args, repo, root, preset, backend, scenario)))
                if halt:
                    append_halted_cases(report, preset, [backend], case_defs, halt)
                write_reports(output, report, bundle=False)
        if selected_profiles:
            from .profile_execution import run_profiles
            halt = run_profiles(selected_profiles, args=args, repo=repo, root=root, output=output,
                catalog=catalog, report=report, policy=policy, index=index, commands=commands,
                runtime_dirs=runtime_dirs, executed_models=executed_models,
                download_reserved=download_reserved, halt=halt)
        if args.include_suites:
            from .suites import list_suites, run_suites
            if halt:
                report["results"].extend({"id": spec["id"], "status": "untested", "reason": UNSAFE_RESOURCES_REASON,
                    "stage": "execution_halted", "halt_trigger": halt["trigger_case"]} for spec in list_suites(repo))
            else:
                suite_results = run_suites(repo, output / "suites", args.mode, args.allow_builds, args.timeout_seconds)
                for result in suite_results:
                    halt = record_execution_halt(report, result) or halt
                report["results"].extend(suite_results)
        if args.native_test_build_dir:
            from .ctest import run_native_tests
            if halt:
                report["results"].append({"id": "native_ctest", "status": "untested", "reason": UNSAFE_RESOURCES_REASON,
                    "stage": "execution_halted", "halt_trigger": halt["trigger_case"], "build_dir": str(args.native_test_build_dir)})
            else:
                native_results = run_native_tests(repo, args.native_test_build_dir, output / "native-ctest", args.mode,
                    args.native_test_backend, args.native_test_filter, args.timeout_seconds, args.gpu_index)
                for result in native_results:
                    halt = record_execution_halt(report, result) or halt
                report["results"].extend(native_results)
        if len(report["results"]) == 1:
            raise ValueError("scenario selector matched no cases")
        # An inventory pass alone must not turn a zero-execution run into an inference pass.
        runtime_results = [r for r in report["results"] if r["id"] != "config-coverage-audit"]
        report["exit_code"] = aggregate_exit(runtime_results, strict=args.strict)
        model_results = [r for r in runtime_results if r.get("preset")]
        if model_results and not any(r["status"] in {"pass", "fail"} for r in model_results) and report["exit_code"] == 0:
            report["exit_code"] = 3
        if args.mode == "plan":
            report["exit_code"] = 1 if any(r["status"] == "fail" for r in runtime_results) else 0
    except KeyboardInterrupt:
        cancelled = True
        cleanup = unsafe_process_state()
        if cleanup is None:
            report["results"].append({"id": "run-cancelled", "status": "untested", "reason": "Interrupted by user; remaining cases were not executed"})
            report["exit_code"] = 130
        else:
            failure = {"id": "run-cancelled", "status": "fail", "reason": "Interrupted with unverified owned-process cleanup",
                       "containment_failure": True, "teardown": cleanup.get("teardown"), "process_evidence": cleanup}
            report["results"].append(failure)
            halt = record_execution_halt(report, failure)
            # Cancellation must still disclose untouched selected coverage when
            # resources may remain owned. All operations below are inventory-only.
            for pending in locals().get("presets", []):
                pending_cases = [scenarios[s] for s in pending.get("scenarios", []) if selected(s, args.scenarios)]
                decisions = [d.get("selected") for d in report.get("backend_selections", []) if d["preset"] == pending["id"] and d.get("selected")]
                pending_backends = (decisions[-1:] or ["auto"]) if args.backend == "auto" else select_backends(args.backend, pending.get("backends", []), hardware) or ["none"]
                append_halted_cases(report, pending, pending_backends, pending_cases, halt)
            existing_ids = {row["id"] for row in report["results"]}
            if args.include_suites:
                from .suites import list_suites
                report["results"].extend({"id": spec["id"], "status": "untested", "reason": UNSAFE_RESOURCES_REASON,
                    "stage": "execution_halted", "halt_trigger": halt["trigger_case"]} for spec in list_suites(repo) if spec["id"] not in existing_ids)
            if args.native_test_build_dir and "native_ctest" not in existing_ids:
                report["results"].append({"id": "native_ctest", "status": "untested", "reason": UNSAFE_RESOURCES_REASON,
                    "stage": "execution_halted", "halt_trigger": halt["trigger_case"]})
            report["exit_code"] = 1
    except (ValueError, OSError, KeyError, TypeError, RuntimeError) as error:
        report["results"].append({"id": "harness-error", "status": "fail", "reason": str(error), "stage": "configuration"})
        report["exit_code"] = 2
    report["finished_utc"] = datetime.now(timezone.utc).isoformat()
    report["cancelled"] = cancelled
    write_reports(output, report)
    print(f"Report: {output / 'report.md'} (exit {report['exit_code']})", flush=True)
    return report["exit_code"]
