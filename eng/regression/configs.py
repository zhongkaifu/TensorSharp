"""Read host JSONC presets without allowing the host to fetch files implicitly.

This intentionally does not implement arbitrary host CLI parsing. ConfigFileArgs.cs
is the authority for host options; the coverage catalog records the host contract.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import json
import os
from pathlib import Path
import re
from typing import Any, Mapping


class ConfigError(ValueError):
    pass


@dataclass(frozen=True)
class FileSpec:
    option: str
    path: Path
    urls: tuple[str, ...] = ()
    sha256: str | None = None
    size_bytes: int | None = None
    group: str | None = None
    optional: bool = False


@dataclass
class ResolvedConfig:
    source: Path
    options: dict[str, Any]
    files: list[FileSpec]
    unavailable: str | None = None
    # Raw host metadata: only the running host knows appBaseDirectory. The
    # offline planner never expands these values or changes os.environ.
    environment_defaults: dict[str, str] = field(default_factory=dict)


def loads_jsonc(text: str) -> Any:
    """Strip comments and trailing commas only outside JSON strings."""
    chars = list(text.lstrip("\ufeff"))
    i, quoted = 0, False
    while i < len(chars):
        if quoted:
            if chars[i] == "\\":
                i += 2
                continue
            if chars[i] == '"':
                quoted = False
            i += 1
            continue
        if chars[i] == '"':
            quoted = True
        elif chars[i:i + 2] == ["/", "/"]:
            while i < len(chars) and chars[i] not in "\r\n":
                chars[i] = " "
                i += 1
            continue
        elif chars[i:i + 2] == ["/", "*"]:
            end = i + 2
            while end + 1 < len(chars) and chars[end:end + 2] != ["*", "/"]:
                end += 1
            if end + 1 >= len(chars):
                raise ConfigError("Unterminated block comment")
            for n in range(i, end + 2):
                if chars[n] not in "\r\n":
                    chars[n] = " "
            i = end + 2
            continue
        i += 1
    i, quoted = 0, False
    while i < len(chars):
        if quoted:
            if chars[i] == "\\":
                i += 2
                continue
            if chars[i] == '"':
                quoted = False
        elif chars[i] == '"':
            quoted = True
        elif chars[i] == ",":
            n = i + 1
            while n < len(chars) and chars[n].isspace():
                n += 1
            if n < len(chars) and chars[n] in "}]":
                chars[i] = " "
        i += 1

    def pairs(values):
        result = {}
        for key, value in values:
            if key in result:
                raise ConfigError(f"Duplicate JSON property: {key}")
            result[key] = value
        return result

    def invalid(value):
        raise ConfigError(f"Nonfinite JSON number: {value}")

    try:
        return json.loads("".join(chars), object_pairs_hook=pairs, parse_constant=invalid)
    except json.JSONDecodeError as exc:
        raise ConfigError(f"Invalid JSONC at line {exc.lineno}, column {exc.colno}: {exc.msg}") from exc


def _expand(text: str, variables: Mapping[str, Any], environ: Mapping[str, str], stack=()) -> str:
    if len(stack) > 32:
        raise ConfigError("Variable expansion exceeded 32 levels")
    result, i = [], 0
    while i < len(text):
        start = text.find("${", i)
        if start < 0:
            result.append(text[i:])
            break
        result.append(text[i:start])
        depth, end = 1, start + 2
        while end < len(text) and depth:
            if text.startswith("${", end):
                depth += 1
                end += 2
                continue
            if text[end] == "}":
                depth -= 1
            end += 1
        if depth:
            raise ConfigError("Unclosed variable reference")
        expression = text[start + 2:end - 1]
        name, separator, fallback = expression.partition(":-")
        if not re.fullmatch(r"[A-Za-z0-9_.-]+", name):
            raise ConfigError(f"Invalid variable name: {name}")
        if name in stack:
            raise ConfigError("Variable cycle: " + " -> ".join((*stack, name)))
        value = variables.get(name, environ.get(name))
        if value is None or str(value) == "":
            if not separator:
                raise ConfigError(f"Undefined variable: {name}")
            value = fallback
        if not isinstance(value, (str, int, float, bool)):
            raise ConfigError(f"Variable {name} must be scalar")
        result.append(_expand(str(value), variables, environ, (*stack, name)))
        i = end
    return "".join(result)


# String values for these options refer to files too, but are never made into URLs.
FILE_OPTIONS = frozenset({
    "model", "mmproj", "draft-model", "qwen-image-vae", "qwen-image-vl",
    "qwen-image-mmproj", "video-dit2", "video-vae", "audio-vae",
    "video-text-encoder", "lora", "lora-config", "embedding-model",
})
_SHARD = re.compile(r"^(.*)-(\d{5})-of-(\d{5})(\.gguf)$", re.I)


def read_config(path: str | Path, model_root: str | Path | None = None,
                overrides: Mapping[str, Any] | None = None,
                environ: Mapping[str, str] | None = None) -> ResolvedConfig:
    """Resolve a single preset, replacing options before variable/file expansion.

    Paths in file objects resolve relative to the config, matching the hosts.
    Plain string file paths retain host working-directory semantics (the suite
    launches from the repository root). Overrides may remove an option with None.
    The runner can serialize ``options`` as a download-free effective config.
    """
    source = Path(path).resolve()
    raw = loads_jsonc(source.read_text(encoding="utf-8-sig"))
    if not isinstance(raw, dict):
        raise ConfigError("Host config must be a JSON object")
    normalized = {}
    for key, value in raw.items():
        name = key.removeprefix("--").lower()
        if name in normalized:
            raise ConfigError(f"Duplicate normalized host option: {name}")
        normalized[name] = value
    unavailable = normalized.get("$unavailable")
    if "$unavailable" in normalized:
        if not isinstance(unavailable, str) or not unavailable.strip():
            raise ConfigError("$unavailable must be a nonempty string")
        return ResolvedConfig(source, {}, [], unavailable)
    variables = normalized.pop("variables", normalized.pop("vars", {}))
    if not isinstance(variables, dict):
        raise ConfigError("variables must be an object")
    variables = dict(variables)
    if "appBaseDirectory" in variables:
        raise ConfigError("appBaseDirectory is reserved for the running host")
    environment = dict(os.environ if environ is None else environ)
    if model_root is not None:
        environment["TENSORSHARP_MODELS"] = str(Path(model_root).resolve())
        # Explicit suite root also wins if the preset defines the env name itself.
        variables["TENSORSHARP_MODELS"] = environment["TENSORSHARP_MODELS"]
    normalized.pop("$schema", None)
    normalized.pop("vars", None)
    normalized.update({k.removeprefix("--").lower(): v for k, v in (overrides or {}).items()})
    defaults = normalized.pop("environment", {})
    if not isinstance(defaults, dict):
        raise ConfigError("environment must be an object of string values")
    for name, value in defaults.items():
        if (not name.strip() or len(name) >= 32767 or "=" in name
                or any(ord(c) < 32 or 0x7f <= ord(c) <= 0x9f for c in name)):
            raise ConfigError("Invalid environment variable name")
        if not isinstance(value, str) or "\x00" in value:
            raise ConfigError(f"Environment default {name} must be a string without NUL")
    files: list[FileSpec] = []

    def expand(value):
        return _expand(value, variables, environment) if isinstance(value, str) else value

    def local_path(value: str, object_path: bool) -> Path:
        value = expand(value)
        if not value or "\x00" in value:
            raise ConfigError("File path is empty or contains NUL")
        if os.name != "nt" and re.match(r"^[A-Za-z]:[\\/]", value):
            raise ConfigError("Windows absolute path is not portable on this host")
        p = Path(value).expanduser()
        if not p.is_absolute():
            # Host string file options are working-directory relative.
            p = (source.parent if object_path else source.parent.parent) / p
        return p.resolve()

    def file_object(option, value, group=None):
        allowed = {"path", "urls", "url", "sha256", "files", "optional", "size_bytes"}
        if set(value) - allowed:
            raise ConfigError(f"Unknown download keys for {option}: {sorted(set(value) - allowed)}")
        if not isinstance(value.get("path"), str):
            raise ConfigError(f"Download {option} requires a path string")
        target = local_path(value["path"], True)
        urls = value.get("urls", [value["url"]] if "url" in value else [])
        if not isinstance(urls, list) or any(not isinstance(u, str) for u in urls):
            raise ConfigError(f"Download {option} urls must be a string array")
        sha = expand(value.get("sha256"))
        if sha is not None and (not isinstance(sha, str) or not re.fullmatch(r"[0-9a-fA-F]{64}", sha)):
            raise ConfigError(f"Download {option} sha256 must contain 64 hex digits")
        size = value.get("size_bytes")
        if size is not None and (type(size) is not int or size <= 0):
            raise ConfigError(f"Download {option} size_bytes must be positive")
        optional = value.get("optional", False)
        if not isinstance(optional, bool):
            raise ConfigError(f"Download {option} optional must be boolean")
        group = group or str(target)
        files.append(FileSpec(option, target, tuple(expand(u) for u in urls),
                              sha.lower() if sha else None, size, group, optional))
        children = value.get("files", [])
        if not isinstance(children, list) or any(not isinstance(child, dict) for child in children):
            raise ConfigError(f"Download {option} files must be an object array")
        for child in children:
            file_object(option, child, group)
        return str(target)

    def option_value(option, value):
        if isinstance(value, dict):
            return file_object(option, value)
        if isinstance(value, list):
            return [option_value(option, item) for item in value]
        if not isinstance(value, (str, int, float, bool, type(None))):
            raise ConfigError(f"Unsupported value for option {option}")
        resolved = expand(value)
        if option in FILE_OPTIONS and isinstance(resolved, str) and resolved.lower() not in {"none", "auto", ""}:
            target = local_path(resolved, False)
            files.append(FileSpec(option, target, group=str(target)))
            return str(target)
        return resolved

    options = {key: option_value(key, value) for key, value in normalized.items() if value is not None}
    # A first-shard override still requires all siblings; never invent their URLs.
    known = {item.path for item in files}
    for item in list(files):
        match = _SHARD.match(item.path.name)
        if match:
            prefix, index, count, suffix = match.groups()
            if int(count) < 1 or int(index) < 1 or int(index) > int(count) or int(count) > 10000:
                raise ConfigError(f"Invalid GGUF shard numbering: {item.path.name}")
            for number in range(1, int(count) + 1):
                sibling = item.path.with_name(f"{prefix}-{number:05d}-of-{count}{suffix}")
                if sibling not in known:
                    files.append(FileSpec(item.option, sibling, group=item.group))
                    known.add(sibling)
    return ResolvedConfig(source, options, files, environment_defaults=dict(defaults))


def flatten(config: ResolvedConfig) -> list[FileSpec]:
    """Deduplicate shared files while refusing conflicting identities."""
    result = {}
    for item in config.files:
        existing = result.get(item.path)
        if existing and (existing.sha256 != item.sha256 or existing.size_bytes != item.size_bytes):
            raise ConfigError(f"Conflicting file identities for {item.path}")
        if existing is None or (existing.optional and not item.optional):
            result[item.path] = item
    return list(result.values())
