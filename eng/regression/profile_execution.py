"""Sequential model/fixture preparation for typed scenario profile executors."""
from __future__ import annotations

from dataclasses import replace
from pathlib import Path
import shutil
import time

from .configs import read_config
from .downloads import PrepareError, prepare_files
from .hardware import detect_hardware, plan_resources
from .processes import unsafe_process_state
from .reports import atomic_json, write_reports
from .scenario_profiles import adapter, per_device_budget, verify_fixtures
from .scenarios import availability, run_scenario


def same_volume(first, second):
    def device(path):
        path = Path(path).resolve()
        while not path.exists():
            if path == path.parent:
                raise OSError("profile cache/output volume is unavailable: " + str(path))
            path = path.parent
        return path.stat().st_dev
    return device(first) == device(second)


def run_profiles(profiles, *, args, repo, root, output, catalog, report, policy, index,
                 commands, runtime_dirs, executed_models, download_reserved, halt):
    # Imported here to keep the CLI's orchestration helpers single-sourced.
    from .runner import (GIB, SAFE_OPTIONS, catalog_file_sizes, plain, record_execution_halt,
                         reproduce_args, selected, sizes_for, use_existing)
    presets = {preset["id"]: preset for preset in catalog["presets"]}
    scenarios = {scenario["id"]: scenario for scenario in catalog["scenarios"]}
    for profile in profiles:
        if not (selected(profile["preset"], args.models) and selected(profile["scenario"], args.scenarios)
                and (args.backend in {"auto", "all"} or profile["backend"] in args.backend.split(","))):
            continue
        preset, backend = presets[profile["preset"]], profile["backend"]
        scenario = scenarios[profile["scenario"]]
        result = {"id": "profile::" + profile["id"], "preset": preset["id"], "family": preset["family"],
                  "scenario": scenario["id"], "backend": backend, "status": "untested",
                  "profile": profile, "stage": "profile_preflight"}
        report["results"].append(result)
        result["reproduce"] = reproduce_args(args, repo, root, preset, backend, scenario)
        if halt:
            result.update(reason=halt["reason"], stage="execution_halted", halt_trigger=halt["trigger_case"])
            continue
        try:
            if backend not in preset["backends"] or preset.get("status") == "unsupported":
                result.update(status="unsupported", reason="Profile backend is outside the primary preset's source-supported matrix")
                continue
            # Offload modes require a separate verified runtime contract. Profiles
            # currently use their declared full-residency total peak requirements.
            if args.cpu_moe:
                result.update(status="unsupported", reason="Typed profiles currently require full-residency mode; --cpu-moe is not implicitly applied")
                continue
            identities, missing = verify_fixtures(profile, hash_files=args.mode == "run")
            result["fixture_identity"] = identities
            if missing:
                result.update(status="skip", reason="Supply SHA-pinned local fixtures: " + "; ".join(missing))
                continue
            model_free = profile["scenario"] == "native_unit"
            result.update(model_qualification=not model_free,
                          qualification_scope="model-free backend contract" if model_free else "model scenario")
            roles = {} if model_free else {"primary": preset["id"], **profile["models"]}
            profile_models, files_by_path = {}, {}
            module = adapter(scenario["id"])
            required_options = module.required_options(profile) if hasattr(module, "required_options") else set()
            for role, preset_id in roles.items():
                item = presets[preset_id]
                resolved = read_config(repo / item["config"], root)
                if resolved.unavailable:
                    raise ValueError(role + ": " + resolved.unavailable)
                options, files = use_existing(resolved, index)
                files = catalog_file_sizes(files, item)
                omitted_specs = [spec for spec in files if spec.optional and
                                 (role != "primary" or spec.option not in required_options)]
                files = [replace(spec, optional=False) for spec in files if spec not in omitted_specs]
                omitted = {spec.option for spec in omitted_specs} - {spec.option for spec in files}
                options = {key: value for key, value in options.items() if key not in omitted and key in SAFE_OPTIONS}
                if not files:
                    raise ValueError(role + ": catalog role has no verifiable weight files")
                for spec in files:
                    key = str(spec.path.resolve())
                    if key in files_by_path and (files_by_path[key].sha256 != spec.sha256
                            or files_by_path[key].size_bytes != spec.size_bytes):
                        raise ValueError("Conflicting immutable identities for shared companion path")
                    files_by_path[key] = spec
                profile_models[role] = {"preset": item, "model_path": options.get("model"), "options": options,
                                        "files": files, "identity": []}
            files = list(files_by_path.values())
            primary = profile_models.get("primary", {"preset": preset, "model_path": None, "options": {}, "identity": []})
            model_dir = output / "profiles" / profile["id"]
            hardware = detect_hardware(root, runtime_dirs=runtime_dirs)
            context = {"repo": repo, "backend": backend, "artifact_dir": model_dir,
                "hardware": hardware,
                "preset_id": preset["id"], "model_family": preset["family"], "model_layout": preset.get("layout"),
                "scenario_profile": profile, "profile_models": profile_models,
                "model_path": primary["model_path"], "model_options": primary["options"],
                "context_tokens": args.context_tokens, "timeout_seconds": args.timeout_seconds,
                "disk_headroom_gib": args.disk_headroom_gib, "seed": profile["parameters"].get("seed", 42),
                "env": {"MAX_CONTEXT": str(args.context_tokens)}, **commands}
            if backend in {"cuda", "ggml_cuda"}:
                devices = profile["parameters"].get("devices") if profile["scenario"] == "multi_gpu" else None
                context["env"]["CUDA_VISIBLE_DEVICES"] = ",".join(map(str, devices or [args.gpu_index]))
            ready, reason = availability(scenario, {**context, "preparing_model": True})
            if not ready:
                result.update(status="skip", reason=reason)
                continue
            hardware = detect_hardware(root, runtime_dirs=runtime_dirs)
            total, remaining = (1, 0) if model_free else sizes_for(files, policy, network=args.probe_network or (args.mode == "run" and args.allow_download))
            resources = profile["resources"]
            overrides = {key: getattr(args, key) for key in ("ram_budget_gib", "vram_budget_gib", "gpu_index", "disk_headroom_gib", "assume_backend_available")
                         if getattr(args, key) is not None}
            overrides.update({key: resources[key] for key in ("ram_required_gib", "vram_required_gib") if key in resources})
            scratch = int(resources.get("scratch_gib", 0) * GIB)
            # Cache transfer bytes and output artifacts may occupy different
            # drives. Charge their sum only when they share the same volume.
            overrides.update(download_bytes=remaining or 0,
                             scratch_bytes=scratch if same_volume(root, output) else 0)
            if profile["scenario"] in {"multi_gpu", "distributed"}:
                overrides["gpu_index"] = profile["parameters"]["devices"][0]
            budget = plan_resources(total, backend, hardware, overrides)
            if model_free:
                budget["model_free"] = True
                budget["estimates"]["weight_bytes"] = 0
            result["resource_plan"] = budget
            result["required_files"] = plain(files)
            output_free = shutil.disk_usage(output).free
            scratch_required = int(resources.get("scratch_gib", 0) * GIB)
            output_required = scratch_required + int(args.disk_headroom_gib * GIB)
            result["artifact_disk_plan"] = {"path": str(output), "free_bytes": output_free,
                "scratch_bytes": scratch_required, "headroom_bytes": int(args.disk_headroom_gib * GIB),
                "required_bytes": output_required}
            fits_devices, device_reason = per_device_budget(profile, hardware)
            absent_roles = set(roles.values()) - executed_models
            reason = None
            if not budget["eligible"]:
                reason = budget["reason"]
            elif not fits_devices:
                reason = device_reason
            elif output_free < output_required:
                reason = "profile artifact volume has insufficient scratch space/headroom; choose --output on a volume with capacity"
            elif args.max_models and len(executed_models | set(roles.values())) > args.max_models:
                reason = "profile including companion models exceeds --max-models"
            elif remaining and max(download_reserved, policy.downloaded_bytes) + remaining > policy.max_download_bytes:
                reason = "remaining whole-run transfer budget insufficient for profile and companion weights"
            elif any(not spec.path.is_file() for spec in files) and not args.allow_download:
                reason = "profile weights missing; --allow-download is disabled"
            if reason or args.mode != "run":
                result.update(status="skip" if reason else "untested", reason=reason or "profile planned; fixtures and models were not executed")
                continue
            # Refresh resource evidence above immediately before sequential preparation.
            download_reserved += remaining or 0
            executed_models.update(absent_roles)
            model_dir.mkdir(parents=True, exist_ok=True)
            identities = plain(prepare_files(files, policy))
            result["model_identity"] = identities
            by_path = {entry["path"]: entry for entry in identities}
            for role in profile_models.values():
                role["identity"] = [by_path[str(spec.path)] for spec in role["files"]]
                role["files"] = plain(role["files"])
            config = model_dir / "effective-config.json"
            atomic_json(config, primary["options"])
            atomic_json(model_dir / "profile.json", profile)
            context.update(config_path=config, effective_config_path=config)
            # Recheck fixture hashes after weight preparation to close the usual
            # editing-during-download window. Executors also bound their reads.
            result["fixture_identity"], missing = verify_fixtures(profile, hash_files=True)
            if missing:
                raise ValueError("profile fixture disappeared during preparation")
            # Downloads and hashing can be long. Recheck physical headroom before
            # a model is started, preserving the completed cache on rejection.
            fresh_hardware = detect_hardware(root, runtime_dirs=runtime_dirs)
            context["hardware"] = fresh_hardware
            fresh_budget = plan_resources(total, backend, fresh_hardware,
                                          {**overrides, "download_bytes": 0})
            fresh_devices, fresh_reason = per_device_budget(profile, fresh_hardware)
            result["execution_resource_plan"] = fresh_budget
            result["artifact_disk_plan"]["free_bytes_before_execution"] = shutil.disk_usage(output).free
            if (not fresh_budget["eligible"] or not fresh_devices or
                    result["artifact_disk_plan"]["free_bytes_before_execution"] < output_required):
                result.update(status="skip", reason="Resources changed after preparation: " +
                    (fresh_budget["reason"] if not fresh_budget["eligible"] else fresh_reason if not fresh_devices
                     else "artifact volume lacks scratch/headroom"))
                continue
            print(f"Running profile {profile['id']} / {backend} / {scenario['id']}", flush=True)
            start = time.monotonic()
            outcome = run_scenario(scenario, context)
            outcome.pop("id", None)
            result.update(outcome, stage="profile_execution", duration_seconds=time.monotonic() - start)
        except (ValueError, OSError, RuntimeError, KeyError, TypeError) as error:
            result.update(status="fail", reason=type(error).__name__ + ": " + str(error))
            if isinstance(error, PrepareError):
                result.update(stage=error.stage, error_code=error.code, error=error.as_dict())
        finally:
            cleanup = unsafe_process_state()
            if cleanup is not None:
                result.update(containment_failure=True, teardown=cleanup.get("teardown"), containment_evidence=cleanup)
            halt = record_execution_halt(report, result) or halt
            write_reports(output, report, bundle=False)
    return halt
