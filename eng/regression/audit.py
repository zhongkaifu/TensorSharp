"""Fail closed when repository config inventory and executable coverage drift."""
from pathlib import Path
import json
import re
import math

from .configs import read_config, loads_jsonc
from .reports import file_hash
from .hardware import BACKENDS


def audit_catalog(repo, catalog):
    repo = Path(repo)
    errors, warnings, inventory = [], [], []
    if not isinstance(catalog, dict) or any(not isinstance(catalog.get(key), list) for key in ("presets", "scenarios")) or any(not isinstance(entry, dict) for key in ("presets", "scenarios") for entry in catalog.get(key, [])):
        return {"status": "fail", "errors": ["catalog must contain presets/scenarios arrays of objects"], "warnings": [], "config_count": 0, "preset_count": 0, "scenario_count": 0, "inventory": [], "families_without_presets": []}
    if catalog.get("schema_version") != 1:
        errors.append("catalog.schema_version must be 1")
    presets, scenarios = catalog.get("presets", []), catalog.get("scenarios", [])
    scenario_ids = {s.get("id") for s in scenarios}
    if len(scenario_ids) != len(scenarios):
        errors.append("duplicate scenario IDs")
    preset_ids = {p.get("id") for p in presets}
    if len(preset_ids) != len(presets):
        errors.append("duplicate preset IDs")
    paths = {p.get("config") for p in presets}
    actual = {p.relative_to(repo).as_posix() for p in (repo / "config").rglob("*.json")
              if "regression" not in p.relative_to(repo / "config").parts}
    for path in sorted(actual - paths):
        errors.append(f"config missing coverage metadata: {path}")
    for path in sorted(paths - actual):
        errors.append(f"catalog references missing config: {path}")
    for preset in presets:
        name = preset.get("id", "?")
        if not isinstance(name, str) or not re.fullmatch(r"[A-Za-z0-9_.-]+", name):
            errors.append(f"unsafe preset ID: {name!r}")
        if name in {".", ".."}:
            errors.append(f"unsafe preset ID: {name!r}")
        if not isinstance(preset.get("backends"), list) or any(b not in BACKENDS for b in preset.get("backends", [])):
            errors.append(f"{name}: unknown or malformed backend list")
        sizes = preset.get("file_sizes", {})
        if not isinstance(sizes, dict) or any(type(v) is not int or v <= 0 for v in sizes.values()):
            errors.append(f"{name}: file_sizes must map filenames to positive integer bytes")
        resources = preset.get("resource_requirements", {})
        if not isinstance(resources, dict) or any(not isinstance(v, (int, float)) or isinstance(v, bool) or not math.isfinite(v) or v < 0
                                                for k, v in resources.items() if k.endswith("_gib")):
            errors.append(f"{name}: resource requirements must be finite nonnegative numbers")
        if not isinstance(preset.get("scenarios"), list):
            errors.append(f"{name}: scenarios must be an array")
            continue
        for scenario in preset.get("scenarios", []):
            if scenario not in scenario_ids:
                errors.append(f"{name}: unknown scenario {scenario}")
        path = repo / str(preset.get("config", ""))
        if not path.is_file() or not path.resolve().is_relative_to((repo / "config").resolve()):
            errors.append(f"{name}: config must be a file within config/")
            continue
        try:
            raw = loads_jsonc(path.read_text(encoding="utf-8-sig"))
            if not isinstance(raw, dict):
                raise ValueError("config root is not an object")
            # Plugins have a different public schema and are not host options.
            if preset.get("kind") == "plugin":
                if not raw.get("type"):
                    raise ValueError("plugin must declare type")
                inventory.append({"config": preset["config"], "sha256": file_hash(path), "kind": "plugin"})
                continue
            resolved = read_config(path, repo / "models", environ={})
            files = resolved.files
            if not resolved.unavailable and "model" not in resolved.options and preset.get("kind") != "overlay":
                raise ValueError("runnable host preset has no model")
            for base in preset.get("requires_configs", []):
                if base not in actual:
                    raise ValueError(f"composition references missing config: {base}")
            if resolved.unavailable and preset.get("kind") != "unsupported" and not preset.get("reason"):
                warnings.append(f"{name}: unavailable preset should record reason")
            for spec in files:
                if spec.urls and not spec.sha256:
                    warnings.append(f"{name}: {spec.path.name} has no SHA256; automatic download refused")
            inventory.append({"config": preset["config"], "sha256": file_hash(path),
                              "file_count": len(files), "unavailable": resolved.unavailable,
                              "options": sorted(resolved.options)})
        except (ValueError, OSError, KeyError, TypeError) as error:
            errors.append(f"{name}: {error}")
    for scenario in scenarios:
        name = scenario.get("id")
        if not isinstance(name, str) or not re.fullmatch(r"[A-Za-z0-9_-][A-Za-z0-9_.-]*", name):
            errors.append(f"unsafe scenario ID: {name!r}")
        if scenario.get("status") not in {"automated", "recipe", "unsupported"}:
            errors.append(f"{scenario.get('id')}: invalid capability status")
        if not scenario.get("assertions"):
            errors.append(f"{scenario.get('id')}: no meaningful assertions declared")
        if scenario.get("status") == "automated":
            from .scenarios import ADAPTER_IDS
            if scenario.get("adapter", name) not in ADAPTER_IDS:
                errors.append(f"{name}: declared automated but no adapter exists")
        for evidence in scenario.get("evidence", []):
            # Evidence accepts repo paths with a line number or an external authoritative URL.
            if isinstance(evidence, str) and not evidence.startswith("https://"):
                path = evidence.split(":", 1)[0].split("#", 1)[0]
                if not (repo / path).exists():
                    errors.append(f"{scenario.get('id')}: missing source evidence {path}")
    return {"status": "fail" if errors else "pass", "errors": errors, "warnings": warnings,
            "config_count": len(actual), "preset_count": len(presets),
            "scenario_count": len(scenarios), "inventory": inventory,
            "families_without_presets": [f for f in catalog.get("families", []) if not f.get("presets")]}
