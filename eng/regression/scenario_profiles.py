"""Typed, local scenario profiles. A profile never supplies an arbitrary command.

Companion weights are catalog presets; user fixtures are local SHA256-pinned files.
Model payload preparation and process containment stay in the normal runner.
"""
from __future__ import annotations

import importlib
import math
from pathlib import Path, PureWindowsPath
import os
import re

from .configs import loads_jsonc
from .hardware import BACKENDS
from .reports import file_hash

PROFILE_MODULES = {
    "embedding_media": "image_profiles", "image_generation": "image_profiles",
    "image_editing": "image_profiles", "video_generation": "video_profiles",
    "video_audio_generation": "video_profiles", "video_reference": "video_profiles",
    "speculation": "scaling_profiles", "lora": "scaling_profiles",
    "multi_gpu": "scaling_profiles", "distributed": "scaling_profiles",
    "retrieval": "knowledge_profiles", "jev_media": "knowledge_profiles",
    "client_live": "extended_profiles", "native_unit": "extended_profiles",
}
MAX_FIXTURE_BYTES = 512 * 1024 ** 2
_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,95}\Z")


def adapter(identifier):
    return importlib.import_module("." + PROFILE_MODULES[identifier], __package__)


def _object(value, name, allowed=None):
    if not isinstance(value, dict):
        raise ValueError(name + " must be an object")
    if allowed is not None and set(value) - allowed:
        raise ValueError(name + " has unknown keys: " + ", ".join(sorted(set(value) - allowed)))
    return value


def _positive(value, name):
    if type(value) not in (int, float) or not math.isfinite(value) or value <= 0:
        raise ValueError(name + " must be finite and positive")


def load_profiles(path, catalog):
    """Validate structure/typed parameters without reading large fixture payloads."""
    if path is None:
        return []
    path = Path(path).resolve()
    if path.stat().st_size > 1024 * 1024:
        raise ValueError("scenario profile JSON exceeds 1 MiB")
    raw = _object(loads_jsonc(path.read_text(encoding="utf-8-sig")), "profile document", {"schema_version", "profiles"})
    if type(raw.get("schema_version")) is not int or raw["schema_version"] != 1 or not isinstance(raw.get("profiles"), list):
        raise ValueError("profile document requires schema_version=1 and profiles array")
    if len(raw["profiles"]) > 128:
        raise ValueError("at most 128 profiles per document")
    presets = {p["id"]: p for p in catalog["presets"]}
    scenarios = {s["id"] for s in catalog["scenarios"]}
    ids, result = set(), []
    for source in raw["profiles"]:
        profile = dict(_object(source, "profile", {"id", "scenario", "preset", "backend", "models", "fixtures", "parameters", "expected", "resources"}))
        identifier = profile.get("id")
        if not isinstance(identifier, str) or not _ID.fullmatch(identifier) or identifier in ids:
            raise ValueError("profile id must be unique and safe: " + str(identifier))
        ids.add(identifier)
        if profile.get("scenario") not in PROFILE_MODULES or profile["scenario"] not in scenarios:
            raise ValueError(identifier + ": scenario has no typed profile executor")
        if profile.get("preset") not in presets:
            raise ValueError(identifier + ": unknown primary catalog preset")
        if profile.get("backend") not in BACKENDS:
            raise ValueError(identifier + ": backend must be an exact backend ID")
        models = _object(profile.setdefault("models", {}), identifier + ".models")
        for role, preset in models.items():
            if not _ID.fullmatch(role) or preset not in presets:
                raise ValueError(identifier + ": model roles must reference known catalog presets")
        if "primary" in models and models["primary"] != profile["preset"]:
            raise ValueError(identifier + ": primary role must match preset")
        fixtures = _object(profile.setdefault("fixtures", {}), identifier + ".fixtures")
        resolved = {}
        for name, descriptor in fixtures.items():
            if not _ID.fullmatch(name):
                raise ValueError(identifier + ": unsafe fixture name")
            descriptor = dict(_object(descriptor, "fixture " + name, {"path", "sha256"}))
            filename, digest = descriptor.get("path"), descriptor.get("sha256")
            if not isinstance(filename, str) or not filename.strip() or "\x00" in filename:
                raise ValueError(identifier + ": fixture path must be a local path")
            if "://" in filename or filename.replace("\\", "/").startswith("//"):
                raise ValueError(identifier + ": remote fixture paths are not allowed")
            if os.name != "nt" and PureWindowsPath(filename).drive:
                raise ValueError(identifier + ": Windows drive fixture paths are unavailable on this platform; use a local relative path")
            if not isinstance(digest, str) or re.fullmatch(r"[a-fA-F0-9]{64}", digest) is None:
                raise ValueError(identifier + ": fixture SHA256 is required")
            candidate = Path(filename).expanduser()
            descriptor.update(path=str((path.parent / candidate).resolve() if not candidate.is_absolute() else candidate.resolve()), sha256=digest.lower())
            resolved[name] = descriptor
        profile["fixtures"] = resolved
        _object(profile.setdefault("parameters", {}), identifier + ".parameters")
        _object(profile.setdefault("expected", {}), identifier + ".expected")
        resources = _object(profile.get("resources"), identifier + ".resources", {"ram_required_gib", "vram_required_gib", "per_gpu_required_gib", "scratch_gib"})
        _positive(resources.get("ram_required_gib"), identifier + ".resources.ram_required_gib")
        if profile["backend"] not in {"cpu", "ggml_cpu"}:
            _positive(resources.get("vram_required_gib"), identifier + ".resources.vram_required_gib")
        for key in ("vram_required_gib", "scratch_gib"):
            if key in resources:
                _positive(resources[key], identifier + ".resources." + key)
        per_gpu = _object(resources.get("per_gpu_required_gib", {}), identifier + ".resources.per_gpu_required_gib")
        for device, required in per_gpu.items():
            if not re.fullmatch(r"0|[1-9][0-9]*", device):
                raise ValueError("per_gpu_required_gib keys must be nonnegative device indices")
            _positive(required, "per_gpu_required_gib." + device)
        adapter(profile["scenario"]).validate_profile(profile)
        profile["profile_source"] = {"path": str(path), "sha256": file_hash(path)}
        result.append(profile)
    return result


def verify_fixtures(profile, *, hash_files):
    """Return explicit missing requirements; changed/corrupt fixtures fail closed."""
    identities, missing = {}, []
    for name, descriptor in profile["fixtures"].items():
        path = Path(descriptor["path"])
        if not path.is_file():
            missing.append(name + ": " + str(path))
            continue
        size = path.stat().st_size
        if not 0 < size <= MAX_FIXTURE_BYTES:
            raise ValueError(name + ": fixture must be nonempty and at most 512 MiB")
        digest = file_hash(path) if hash_files else None
        if hash_files and digest != descriptor["sha256"]:
            raise ValueError(name + ": fixture SHA256 mismatch")
        identities[name] = {**descriptor, "bytes": size, "actual_sha256": digest, "verified": hash_files}
    return identities, missing


def per_device_budget(profile, hardware):
    requirements = profile["resources"].get("per_gpu_required_gib", {})
    if profile["scenario"] in {"multi_gpu", "distributed"}:
        devices = profile["parameters"].get("devices", [])
        if set(requirements) != {str(device) for device in devices}:
            return False, "multi_gpu/distributed requires a separate VRAM budget for every selected device"
        if profile["scenario"] == "distributed":
            # Only the single-device numerical control runs on this controller.
            # The fixed remote supervisor measures every declared GPU per node.
            requirements = {str(devices[0]): requirements[str(devices[0])]} if devices else {}
    gpus = {str(gpu["index"]): gpu for gpu in hardware.get("gpus", [])}
    for device, required in requirements.items():
        available = gpus.get(device, {}).get("memory_free_bytes")
        if available is None or required * 1024 ** 3 + 2 * 1024 ** 3 > available:
            return False, "device " + device + " has insufficient/unknown free VRAM including 2 GiB headroom; VRAM is not pooled"
    return True, "each declared GPU fits independently"
