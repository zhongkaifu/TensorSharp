#!/usr/bin/env python3
"""
Configuration loader for the cross-engine inference benchmark.

Three engines (TensorSharp, llama.cpp, vLLM) are compared on the *same* GGUF
files, the *same* host, through one uniform OpenAI `/v1/chat/completions`
surface, across text / image / audio / video / single-turn / multi-turn /
function-call / structured-output scenarios, on any compute backend declared
in the config's `backends` registry (ggml_cuda / ggml_vulkan / ggml_metal /
ggml_cpu / cpu / ...).

Nothing here is hardcoded: every setting — host paths, the model / scenario /
engine / backend registries, and the run defaults — is read from a JSON config
file (default `benchmark_config.json` beside this module). This module loads
that file, resolves `${var}` path placeholders, applies the documented
environment-variable overrides for host retargeting, and exposes the same
registries (`MODELS`, `SCENARIOS`, `ENGINES`, ...) the rest of the harness
consumes.

Precedence (highest first):
    command-line flags (run_matrix.py / report.py)
  > environment variables (host paths only; see the `paths` section)
  > the JSON config file
  > built-in fallbacks

Select a different config file with `--config PATH` (run_matrix.py / report.py)
or the `BENCH_CONFIG` environment variable.
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

# ---------------------------------------------------------------------------
# Locate + load the config file
# ---------------------------------------------------------------------------
HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[1]                      # .../TensorSharp
ASSETS_DIR = HERE / "assets"

CONFIG_PATH = Path(os.environ.get("BENCH_CONFIG") or (HERE / "benchmark_config.json"))


def _load(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(
            f"benchmark config file not found: {path}\n"
            f"Pass --config PATH or set BENCH_CONFIG to point at a config file.")
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)


_CFG = _load(CONFIG_PATH)


# ---------------------------------------------------------------------------
# Path-variable substitution + environment-variable overrides
# ---------------------------------------------------------------------------
def _env_or(env_key: Optional[str], value):
    """An environment variable (if set and non-empty) wins over the file value."""
    if env_key:
        ev = os.environ.get(env_key)
        if ev not in (None, ""):
            return ev
    return value


_paths = _CFG.get("paths", {}) or {}

# Substitution variables usable in any path string as `${name}`.
_VARS: dict = {
    "here": str(HERE),
    "repo_root": str(REPO_ROOT),
}


def _subst(s: str) -> str:
    out = s
    for k, v in _VARS.items():
        out = out.replace("${" + k + "}", v)
    return out


# model_root and gemma4_qat_dir are themselves substitution variables (and the
# two big host-retargeting knobs), resolved first so model paths can reference
# them. Each is overridable by an environment variable for retargeting.
MODEL_ROOT = Path(_subst(str(_env_or(
    "BENCH_MODEL_ROOT", _paths.get("model_root", r"C:/Works/models")))))
_VARS["model_root"] = str(MODEL_ROOT)

GEMMA4_QAT_DIR = Path(_subst(str(_env_or(
    "BENCH_GEMMA4_QAT_DIR",
    _paths.get("gemma4_qat_dir") or "${model_root}/gemma_mtp/qat"))))
_VARS["gemma4_qat_dir"] = str(GEMMA4_QAT_DIR)


def _path(value, env_key: Optional[str] = None) -> Optional[Path]:
    """Resolve a path-valued config entry. `value` is either a plain string or
    an object `{"path": ..., "env": "BENCH_..."}` whose `env` var (if set)
    overrides it. Applies `${var}` substitution. Returns None for null/absent."""
    if isinstance(value, dict):
        env_key = value.get("env", env_key)
        value = value.get("path")
    value = _env_or(env_key, value)
    if value in (None, ""):
        return None
    return Path(_subst(str(value)))


# Result directory + engine binaries / endpoints / media (env-overridable).
RESULTS_DIR = _path(_paths.get("results_dir"), "BENCH_RESULTS") or (HERE / "results")

TENSORSHARP_SERVER_DLL = (
    _path(_paths.get("tensorsharp_server_dll"), "BENCH_TS_SERVER_DLL")
    or (REPO_ROOT / "TensorSharp.Server" / "bin" / "TensorSharp.Server.dll"))
# TensorSharp.Server hard-codes its listen address to 0.0.0.0:5000.
TENSORSHARP_PORT = int(_env_or("BENCH_TS_PORT", _paths.get("tensorsharp_port", 5000)))

LLAMA_SERVER_EXE = (
    _path(_paths.get("llama_server_exe"), "BENCH_LLAMA_SERVER")
    or Path(r"C:/Works/llama.cpp/build-cuda/bin/Release/llama-server.exe"))
LLAMA_PORT = int(_env_or("BENCH_LLAMA_PORT", _paths.get("llama_port", 5001)))

# vLLM is connect-only (we never launch it): point this at a running
# OpenAI-compatible vLLM endpoint. When unreachable, every vLLM cell is
# recorded as skipped(engine unavailable) rather than failing the run.
VLLM_BASE_URL = str(_env_or("BENCH_VLLM_URL",
                            _paths.get("vllm_base_url", "http://127.0.0.1:8000")))

# Media assets used by the multimodal scenarios.
_media = _paths.get("media", {}) or {}
MEDIA_IMAGE = _path(_media.get("image"), "BENCH_IMAGE") or Path(r"C:/Works/test.jpg")
MEDIA_AUDIO = _path(_media.get("audio"), "BENCH_AUDIO") or Path(r"C:/Works/obama_first_45_secs.mp3")
MEDIA_VIDEO = _path(_media.get("video"), "BENCH_VIDEO") or Path(r"C:/Works/concert.mp4")


# ---------------------------------------------------------------------------
# Model registry
# ---------------------------------------------------------------------------
@dataclass
class ModelSpec:
    short_id: str
    display: str
    family: str                       # gemma4 | qwen36 | diffusiongemma
    gguf: Path
    mmproj: Optional[Path]
    modalities: set                   # subset of {text,image,audio,video}
    size_class: str                   # small | medium | large
    is_diffusion: bool = False
    diffusion_steps: int = 32
    # MTP / NextN speculative decoding (TensorSharp only).
    mtp_supported: bool = False       # model ships a draft head we can engage
    mtp_draft: Optional[Path] = None  # separate draft GGUF (Gemma 4); None = embedded (Qwen 3.6)

    def exists(self) -> bool:
        return self.gguf.exists()

    def mtp_available(self) -> bool:
        """MTP can actually run: supported, and (if it needs a separate draft
        GGUF) that file is present on disk."""
        if not self.mtp_supported:
            return False
        return self.mtp_draft is None or self.mtp_draft.exists()


def _build_models(cfg: dict) -> dict:
    out: dict = {}
    for short_id, m in (cfg.get("models") or {}).items():
        is_diffusion = bool(m.get("is_diffusion", False))
        steps = m.get("diffusion_steps", 32)
        if is_diffusion:                     # DIFFUSION_STEPS env override
            steps = _env_or("DIFFUSION_STEPS", steps)
        out[short_id] = ModelSpec(
            short_id=short_id,
            display=m["display"],
            family=m["family"],
            gguf=_path(m.get("gguf")),
            mmproj=_path(m.get("mmproj")),
            modalities=set(m.get("modalities", [])),
            size_class=m.get("size_class", "medium"),
            is_diffusion=is_diffusion,
            diffusion_steps=int(steps),
            mtp_supported=bool(m.get("mtp_supported", False)),
            mtp_draft=_path(m.get("mtp_draft")),
        )
    return out


MODELS: dict = _build_models(_CFG)


# ---------------------------------------------------------------------------
# Scenario registry
# ---------------------------------------------------------------------------
@dataclass
class ScenarioSpec:
    short_id: str
    kind: str                 # text | multi_turn | function_call | json_mode | image | audio | video
    description: str
    modality: Optional[str] = None    # required model modality, if any
    max_tokens: int = 128

    @property
    def is_text_only(self) -> bool:
        return self.modality is None


def _build_scenarios(cfg: dict) -> dict:
    out: dict = {}
    for sid, s in (cfg.get("scenarios") or {}).items():
        out[sid] = ScenarioSpec(
            short_id=sid,
            kind=s["kind"],
            description=s.get("description", ""),
            modality=s.get("modality"),
            max_tokens=int(s.get("max_tokens", 128)),
        )
    return out


def synth_scenario(sid: str) -> Optional[ScenarioSpec]:
    """Synthesize a ScenarioSpec for a generic `prefill_<N>` / `prefill_<N>k` id
    (the long-prompt prefill dataset, e.g. `prefill_8k`, `prefill_512`). These do
    not need to be declared in a config's `scenarios` block, so they work against
    any config file. Returns None for anything that is not a prefill id."""
    if not isinstance(sid, str) or not sid.startswith("prefill_"):
        return None
    suffix = sid.split("prefill_", 1)[-1].strip().lower()
    try:
        target = (int(round(float(suffix[:-1]) * 1024))
                  if suffix.endswith("k") else int(suffix))
    except ValueError:
        return None
    if target <= 0:
        return None
    return ScenarioSpec(
        short_id=sid,
        kind="text",
        description=f"Prefill throughput @ ~{target} prompt tokens",
        modality=None,
        max_tokens=8,            # prefill is timed by TTFT; barely decode
    )


class _ScenarioRegistry(dict):
    """Scenario registry that also resolves on-the-fly `prefill_<N>` ids so the
    prefill dataset can be requested against any config without being declared in
    it. Iteration / `.keys()` still yield only the declared scenarios."""

    def __missing__(self, key):
        spec = synth_scenario(key)
        if spec is None:
            raise KeyError(key)
        return spec

    def __contains__(self, key):
        return super().__contains__(key) or synth_scenario(key) is not None


SCENARIOS: dict = _ScenarioRegistry(_build_scenarios(_CFG))


# ---------------------------------------------------------------------------
# Engine + backend registry
# ---------------------------------------------------------------------------
@dataclass
class EngineSpec:
    engine_id: str
    display: str
    transport: str                 # "server" (we launch it) | "connect" (external)
    backends: tuple                # backend ids and/or kinds ("gpu"/"cpu") this
                                   # engine may run on; empty = no restriction


def _build_engines(cfg: dict) -> dict:
    out: dict = {}
    for eid, e in (cfg.get("engines") or {}).items():
        out[eid] = EngineSpec(
            engine_id=eid,
            display=e.get("display", eid),
            transport=e.get("transport", "server"),
            backends=tuple(e.get("backends", [])),
        )
    return out


ENGINES: dict = _build_engines(_CFG)


@dataclass
class BackendSpec:
    """One concrete compute backend the matrix can run on (one column in the
    report). `kind` drives the generic gating rules (large models skip
    cpu-kind backends; vLLM columns are gpu-kind); the per-engine fields say
    how each engine is launched on this backend — an engine with no mapping
    cannot run it and its cells are recorded as skipped."""
    backend_id: str                    # e.g. "ggml_cuda", "ggml_vulkan", "cpu"
    display: str                       # column label in the report
    kind: str                          # "gpu" | "cpu"
    aliases: tuple = ()                # alternate --backends names (e.g. legacy "gpu")
    # TensorSharp.Server mapping (ts_backend None = TensorSharp cannot run it).
    ts_backend: Optional[str] = None   # value passed to `--backend`
    ts_extra_args: tuple = ()          # extra server CLI args (e.g. --gpu-device 1)
    ts_env: dict = field(default_factory=dict)   # extra env vars for the server process
    # llama.cpp mapping (llama_ngl None = llama.cpp cannot run it).
    llama_ngl: Optional[int] = None    # value passed to `-ngl`
    llama_server_exe: Optional[Path] = None      # per-backend build (e.g. a Vulkan
                                                 # llama-server); None = paths.llama_server_exe
    llama_extra_args: tuple = ()
    llama_env: dict = field(default_factory=dict)
    # vLLM is connect-only: nothing is launched, this flag just says the
    # external endpoint's numbers belong in this backend's column.
    vllm: bool = False


def _build_backend(bid: str, b: dict) -> BackendSpec:
    ts = b.get("tensorsharp")
    if isinstance(ts, str):                  # shorthand: "tensorsharp": "ggml_vulkan"
        ts = {"backend": ts}
    ts = ts or {}
    llama = b.get("llamacpp")
    if isinstance(llama, (int, float)):      # shorthand: "llamacpp": 999  (the -ngl value)
        llama = {"ngl": int(llama)}
    llama = llama or {}
    return BackendSpec(
        backend_id=bid,
        display=b.get("display", bid),
        kind=b.get("kind", "gpu"),
        aliases=tuple(str(a) for a in b.get("aliases", [])),
        ts_backend=ts.get("backend"),
        ts_extra_args=tuple(str(a) for a in ts.get("extra_args", [])),
        ts_env={str(k): str(v) for k, v in (ts.get("env") or {}).items()},
        llama_ngl=(int(llama["ngl"]) if llama.get("ngl") is not None else None),
        llama_server_exe=_path(llama.get("server_exe")),
        llama_extra_args=tuple(str(a) for a in llama.get("extra_args", [])),
        llama_env={str(k): str(v) for k, v in (llama.get("env") or {}).items()},
        vllm=bool(b.get("vllm", False)),
    )


def _build_backends(cfg: dict) -> dict:
    raw = cfg.get("backends")
    if isinstance(raw, dict) and raw:
        return {bid: _build_backend(bid, b or {}) for bid, b in raw.items()
                if not bid.startswith("_")}     # "_comment" etc. are not backends
    # Legacy config form: `backends` is a list of abstract ids (["gpu", "cpu"])
    # mapped per-engine through the `maps` section. Synthesize the equivalent
    # registry so old config files keep working unchanged.
    ids = [str(x) for x in (raw or ["gpu", "cpu"])]
    maps = cfg.get("maps", {}) or {}
    ts_map = dict(maps.get("tensorsharp_backend",
                           {"gpu": "ggml_cuda", "cpu": "ggml_cpu"}))
    ngl_map = {k: int(v) for k, v in
               (maps.get("llama_ngl", {"gpu": 999, "cpu": 0})).items()}
    out: dict = {}
    for bid in ids:
        kind = "cpu" if "cpu" in bid.lower() else "gpu"
        out[bid] = BackendSpec(
            backend_id=bid, display=bid.upper(), kind=kind,
            ts_backend=ts_map.get(bid),
            llama_ngl=ngl_map.get(bid),
            vllm=(kind == "gpu"))
    return out


BACKENDS: dict = _build_backends(_CFG)

_BACKEND_ALIASES: dict = {}
for _b in BACKENDS.values():
    for _a in _b.aliases:
        _BACKEND_ALIASES.setdefault(_a.lower(), _b.backend_id)


def resolve_backend(name: str) -> Optional[str]:
    """Canonical backend id for a --backends token (case-insensitive, resolves
    aliases like the legacy `gpu`). Returns None when unknown."""
    if not name:
        return None
    if name in BACKENDS:
        return name
    n = name.strip().lower()
    for bid in BACKENDS:
        if bid.lower() == n:
            return bid
    return _BACKEND_ALIASES.get(n)


def _resolve_backend_list(names) -> list:
    """Alias-resolve a backend selection, preserving order and dropping dupes.
    Unknown names pass through unchanged so the caller can report them."""
    out: list = []
    for n in names:
        rid = resolve_backend(str(n)) or str(n)
        if rid not in out:
            out.append(rid)
    return out


def llama_server_exe_for(backend: str) -> Path:
    """The llama-server binary for a backend: its per-backend build when
    configured (e.g. a Vulkan build), else the global paths.llama_server_exe."""
    spec = BACKENDS.get(backend)
    exe = spec.llama_server_exe if spec is not None else None
    return exe or LLAMA_SERVER_EXE


# llama.cpp server launch options.
_llama = _CFG.get("llama", {}) or {}
LLAMA_CONTEXT_SIZE = int(_llama.get("context_size", 8192))
LLAMA_EXTRA_ARGS = list(_llama.get("extra_args", ["--jinja"]))

# Per-size-class server readiness timeout (seconds).
READY_TIMEOUT_S = {k: float(v) for k, v in
                   (_CFG.get("ready_timeout_s", {"small": 600, "medium": 900, "large": 1200})).items()}

# ---------------------------------------------------------------------------
# Run defaults (used when a command-line flag is not supplied)
# ---------------------------------------------------------------------------
_defaults = _CFG.get("defaults", {}) or {}
DEFAULT_ENGINES = list(_defaults.get("engines") or list(ENGINES.keys()))
DEFAULT_MODELS = list(_defaults.get("models") or list(MODELS.keys()))
DEFAULT_SCENARIOS = list(_defaults.get("scenarios") or list(SCENARIOS.keys()))
DEFAULT_BACKENDS = _resolve_backend_list(_defaults.get("backends") or list(BACKENDS.keys()))

# Extra benchmark axes (default to the single baseline point so existing
# invocations are unchanged):
#   * mtp         - whether MTP/NextN speculative decoding is engaged (TensorSharp)
#   * concurrency - number of identical requests fired in parallel at one server
DEFAULT_MTP_MODES = [bool(x) for x in _defaults.get("mtp_modes", [False])]
DEFAULT_CONCURRENCY = [int(x) for x in _defaults.get("concurrency", [1])]

DEFAULT_MAX_TOKENS = int(_defaults.get("max_tokens", 128))
DEFAULT_WARMUP = int(_defaults.get("warmup", 1))
# Server max-tokens headroom (must cover the busiest cell).
SERVER_MAX_TOKENS = int(_defaults.get("server_max_tokens", 512))


# ---------------------------------------------------------------------------
# Applicability gating
# ---------------------------------------------------------------------------
def applies(engine: str, backend: str, model: ModelSpec,
            scenario: ScenarioSpec, mtp: bool = False) -> tuple[bool, str]:
    """Return (runnable, skip_reason). A non-runnable combination is recorded
    as a skip in the result set rather than silently dropped."""
    eng = ENGINES[engine]

    b = BACKENDS.get(backend) or BACKENDS.get(resolve_backend(backend) or "")
    if b is None:
        return False, f"unknown backend '{backend}' (known: {', '.join(BACKENDS)})"

    # An engine may restrict itself to backend ids and/or kinds ("gpu"/"cpu")
    # via `engines.*.backends`; on top of that the backend entry itself must
    # carry a launch mapping for the engine.
    if eng.backends and not ({b.backend_id, b.kind} & set(eng.backends)):
        return False, f"{eng.display} is not configured for the {b.backend_id} backend"
    if engine == "tensorsharp" and not b.ts_backend:
        return False, f"{b.backend_id} has no TensorSharp launch mapping"
    if engine == "llamacpp" and b.llama_ngl is None:
        return False, f"{b.backend_id} has no llama.cpp launch mapping"
    if engine == "vllm" and not b.vllm:
        return False, f"vLLM endpoint is not comparable on the {b.backend_id} backend"

    # MTP / NextN speculative decoding is a TensorSharp feature (Qwen 3.6's
    # embedded NextN block, or a Gemma 4 gemma4-assistant draft GGUF). When MTP
    # is requested, gate out engines/models that cannot engage it.
    if mtp:
        if engine != "tensorsharp":
            return False, f"{eng.display} has no MTP/NextN speculative decode path"
        if not model.mtp_supported:
            return False, f"{model.short_id} has no MTP draft head"
        if model.mtp_draft is not None and not model.mtp_draft.exists():
            return False, f"MTP draft GGUF not found: {model.mtp_draft}"

    # CPU-kind backends are restricted to small/medium models (large MoE on
    # CPU is impractically slow).
    if b.kind == "cpu" and model.size_class == "large":
        return False, "CPU skipped for large model (impractically slow)"

    # Scenario needs a modality the model does not have.
    if scenario.modality is not None and scenario.modality not in model.modalities:
        return False, f"{model.short_id} has no {scenario.modality} input"

    # DiffusionGemma is a text-diffusion model: only TensorSharp can run it, and
    # only on text scenarios (no tools / json-mode / multimodal).
    if model.is_diffusion:
        if engine != "tensorsharp":
            return False, "diffusion model only supported by TensorSharp"
        if scenario.kind not in ("text", "multi_turn"):
            return False, "diffusion model is text-only (no tools/json/multimodal)"

    # llama.cpp has no video CLI/endpoint path.
    if engine == "llamacpp" and scenario.kind == "video":
        return False, "llama.cpp has no video input path"

    # vLLM: connect-only, GPU-only, text-style scenarios. Multimodal/custom-arch
    # cells are gated out here; whatever remains is still subject to the
    # connector reporting the endpoint unreachable at runtime.
    if engine == "vllm":
        if scenario.modality is not None or scenario.kind in ("video",):
            return False, "vLLM multimodal not wired in this harness"

    return True, ""


def resolve_media() -> dict:
    """Surface which media assets are present (used for warnings / reporting)."""
    return {
        "image": MEDIA_IMAGE.exists(),
        "audio": MEDIA_AUDIO.exists(),
        "video": MEDIA_VIDEO.exists(),
    }
