#!/usr/bin/env python3
"""
Configuration loader for the cross-engine inference benchmark.

Three engines (TensorSharp, llama.cpp, vLLM) are compared on the *same* GGUF
files, the *same* host, through one uniform OpenAI `/v1/chat/completions`
surface, across text / image / audio / video / single-turn / multi-turn /
function-call / structured-output / agentic-tool-loop / code-edit scenarios, on
any compute backend declared in the config's `backends` registry (ggml_cuda /
ggml_vulkan / ggml_metal / ggml_cpu / cpu / ...).

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
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional
from urllib.parse import quote

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
    an object `{"path": ..., "env": "BENCH_...", "url": ...}` whose `env` var
    (if set) overrides the path. Applies `${var}` substitution. Returns None for
    null/absent."""
    if isinstance(value, dict):
        env_key = value.get("env", env_key)
        value = value.get("path")
    value = _env_or(env_key, value)
    if value in (None, ""):
        return None
    return Path(_subst(str(value)))


def _entry_url(value) -> Optional[str]:
    """The per-file download `url` of a path entry written in object form
    (`{"path": ..., "url": ...}`), or None. May be absolute or a name/relative
    path resolved against the model's `source` base URL."""
    if isinstance(value, dict):
        u = value.get("url")
        if u not in (None, ""):
            return _subst(str(u))
    return None


# Result directory + engine binaries / endpoints / media (env-overridable).
RESULTS_DIR = _path(_paths.get("results_dir"), "BENCH_RESULTS") or (HERE / "results")

# The server's entry point is TensorSharp.Server.Host; TensorSharp.Server is the
# library it builds on and has no Main.
TENSORSHARP_SERVER_DLL = (
    _path(_paths.get("tensorsharp_server_dll"), "BENCH_TS_SERVER_DLL")
    or (REPO_ROOT / "TensorSharp.Server.Host" / "bin" / "TensorSharp.Server.Host.dll"))
# Passed to the server as --port (with --host 127.0.0.1); the health checks poll this port.
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
# Split-GGUF naming, e.g. `Foo-00001-of-00005.gguf`. A config entry points at
# the FIRST shard (what every engine is handed); the remaining shards must be
# present next to it, and are downloaded alongside it.
_SHARD_RE = re.compile(r"^(?P<stem>.*)-(?P<idx>\d{5})-of-(?P<total>\d{5})\.gguf$",
                       re.IGNORECASE)


def expand_shards(p: Optional[Path]) -> list:
    """All files a GGUF path actually needs: `[p]`, or every shard of the set
    when `p` names one part of a split GGUF."""
    if p is None:
        return []
    m = _SHARD_RE.match(p.name)
    if not m:
        return [p]
    total = int(m.group("total"))
    stem = m.group("stem")
    return [p.parent / f"{stem}-{i:05d}-of-{total:05d}.gguf" for i in range(1, total + 1)]


def _shard_url(url: str, index: int) -> str:
    """`url` with its split-GGUF shard index set to `index` (unchanged when the
    URL does not name a shard). Lets a config point one `url` at the first shard
    and have every other shard resolve from it, even when the remote file names
    differ from the local ones."""
    head, _, name = url.rpartition("/")
    m = _SHARD_RE.match(name)
    if not m:
        return url
    name = f"{m.group('stem')}-{index:05d}-of-{m.group('total')}.gguf"
    return f"{head}/{name}" if head else name


def _hf_base(repo: str, revision: str = "main") -> str:
    return f"https://huggingface.co/{repo.strip('/')}/resolve/{revision}/"


def _source_base(src) -> Optional[str]:
    """Base URL a model's files are fetched from. `source` may be:
      * "https://host/dir/"                    - direct base URL
      * "owner/repo"                           - Hugging Face repo id
      * {"hf_repo": ..., "revision": ...}      - Hugging Face repo (+ branch/tag)
      * {"base_url": ...}                      - direct base URL
    Returns None when the model declares no source."""
    if src in (None, ""):
        return None
    if isinstance(src, str):
        s = _subst(src.strip())
        if s.startswith("http://") or s.startswith("https://"):
            return s if s.endswith("/") else s + "/"
        return _hf_base(s)
    if isinstance(src, dict):
        base = src.get("base_url") or src.get("url")
        if base:
            base = _subst(str(base))
            return base if base.endswith("/") else base + "/"
        repo = src.get("hf_repo") or src.get("repo") or src.get("hf")
        if repo:
            return _hf_base(str(repo), str(src.get("revision", "main") or "main"))
    return None


def _join_url(base: Optional[str], name: str) -> Optional[str]:
    """Resolve a file's download URL: an absolute `name` wins, otherwise it is
    appended to the model's source base URL."""
    if name.startswith("http://") or name.startswith("https://"):
        return name
    if not base:
        return None
    return base + quote(name.lstrip("/"))


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
    # Whether the model has routed experts at all, i.e. whether the MoE
    # CPU-offload axis (`--n-cpu-moe`) means anything for it. Tri-state on
    # purpose: `false` gates the model out of that axis with a reason naming the
    # config key, `true` opts it in, and an absent key (the case for every
    # config written before the axis existed) runs the cell and lets the engine
    # be the one to complain — the harness does not guess an architecture fact
    # the config never stated.
    is_moe: Optional[bool] = None
    # Tensor parallelism range this model can be hosted at. Weights that do not
    # fit one GPU declare e.g. `"min_tp": 4`; architectures whose shards must
    # divide a small head count declare e.g. `"max_tp": 2` (Gemma 4 26B-A4B has
    # 2 KV heads, so TensorSharp rejects `--tp 4` at load time). Cells outside
    # the range are recorded as skipped rather than OOMing or failing the box.
    min_tp: int = 1
    max_tp: int = 0                   # 0 = no engine-side limit
    # Download sources. `source_base` is the model's base URL (a Hugging Face
    # repo resolves to https://huggingface.co/<repo>/resolve/<rev>/); `urls`
    # holds any per-file override, keyed by the same role names as `files()`.
    source_base: Optional[str] = None
    urls: dict = field(default_factory=dict)

    def exists(self) -> bool:
        return all(p.exists() for p in expand_shards(self.gguf))

    def mtp_available(self) -> bool:
        """MTP can actually run: supported, and (if it needs a separate draft
        GGUF) that file is present on disk."""
        if not self.mtp_supported:
            return False
        return self.mtp_draft is None or self.mtp_draft.exists()

    def files(self) -> list:
        """Every local file this model needs, as (role, path, source_url).

        Roles are `gguf` / `mmproj` / `mtp_draft`; split GGUFs expand to one
        entry per shard (role `gguf#2`, ...). `source_url` is None when the
        config declares no source for that file, in which case it must already
        exist locally."""
        out: list = []

        def _add(role: str, p: Optional[Path]):
            if p is None:
                return
            shards = expand_shards(p)
            explicit = self.urls.get(role)
            for i, sp in enumerate(shards):
                # An explicit URL names the FIRST shard; the remaining shards
                # advance the shard index inside that same URL (the remote file
                # names may differ from the local ones).
                if explicit:
                    url = _join_url(self.source_base, _shard_url(explicit, i + 1))
                else:
                    url = _join_url(self.source_base, sp.name)
                out.append((role if i == 0 else f"{role}#{i + 1}", sp, url))

        _add("gguf", self.gguf)
        _add("mmproj", self.mmproj)
        _add("mtp_draft", self.mtp_draft)
        return out

    def missing_files(self) -> list:
        return [(role, p, url) for role, p, url in self.files() if not p.exists()]


def _build_models(cfg: dict) -> dict:
    out: dict = {}
    for short_id, m in (cfg.get("models") or {}).items():
        is_diffusion = bool(m.get("is_diffusion", False))
        steps = m.get("diffusion_steps", 32)
        if is_diffusion:                     # DIFFUSION_STEPS env override
            steps = _env_or("DIFFUSION_STEPS", steps)
        # `_hf` (the legacy documentation-only repo pointer) is accepted as a
        # source declaration so older configs download without being rewritten.
        base = _source_base(m.get("source") if m.get("source") is not None else m.get("_hf"))
        urls = {}
        for role, key in (("gguf", "gguf"), ("mmproj", "mmproj"), ("mtp_draft", "mtp_draft")):
            u = _entry_url(m.get(key))
            if u:
                urls[role] = u
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
            is_moe=(bool(m["is_moe"]) if m.get("is_moe") is not None else None),
            mtp_supported=bool(m.get("mtp_supported", False)),
            mtp_draft=_path(m.get("mtp_draft")),
            min_tp=int(m.get("min_tp", 1) or 1),
            max_tp=int(m.get("max_tp", 0) or 0),
            source_base=base,
            urls=urls,
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


# Scenarios that are the same for every model and every host, and therefore do
# not need to be declared in a config's `scenarios` block: the long-prompt
# prefill dataset (`prefill_<N>`), and the two client-driven multi-turn
# workflows. Synthesizing them is what makes `--scenarios agentic` work against
# a config written before they existed, without editing that config — and
# without adding anything to its `defaults.scenarios`, so an existing run is
# unchanged.
_SYNTH_SCENARIOS = {
    "agentic": ScenarioSpec(
        short_id="agentic",
        kind="agentic",
        description=("Multi-step tool use: two dependent tool calls plus a final "
                     "answer that can only come from the tool results"),
        modality=None,
        max_tokens=512),
    "code_edit": ScenarioSpec(
        short_id="code_edit",
        kind="code_edit",
        description=("Code generation then a specific edit to the generated "
                     "program, checked by parsing the edited source"),
        modality=None,
        max_tokens=512),
}


def synth_scenario(sid: str) -> Optional[ScenarioSpec]:
    """Synthesize a ScenarioSpec for an id that needs no config declaration: a
    generic `prefill_<N>` / `prefill_<N>k` (the long-prompt prefill dataset, e.g.
    `prefill_8k`, `prefill_512`), or one of the model-independent multi-turn
    workflows in `_SYNTH_SCENARIOS`. Returns None for anything else."""
    if not isinstance(sid, str):
        return None
    if sid in _SYNTH_SCENARIOS:
        return _SYNTH_SCENARIOS[sid]
    if not sid.startswith("prefill_"):
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
    ts_tp: Optional[bool] = None       # TensorSharp can tensor-parallelize here
                                       # (None = infer from ts_backend)
    ts_tp_arg: str = "--tp"            # flag carrying the TP degree
    # MoE CPU offload (`--n-cpu-moe N` / `--cpu-moe-threads M`). None = infer
    # from ts_backend + the backend kind; the arg names are configurable so a
    # build that spells them differently needs no code change.
    ts_cpu_moe: Optional[bool] = None
    ts_cpu_moe_arg: str = "--n-cpu-moe"
    ts_cpu_moe_threads_arg: str = "--cpu-moe-threads"
    # llama.cpp mapping (llama_ngl None = llama.cpp cannot run it).
    llama_ngl: Optional[int] = None    # value passed to `-ngl`
    llama_server_exe: Optional[Path] = None      # per-backend build (e.g. a Vulkan
                                                 # llama-server); None = paths.llama_server_exe
    llama_extra_args: tuple = ()
    llama_env: dict = field(default_factory=dict)
    llama_tp: Optional[bool] = None    # llama.cpp can tensor-parallelize here
                                       # (None = infer: any gpu-kind backend with -ngl > 0)
    # llama.cpp's tensor-parallel mode. `tensor` splits weights AND KV across the
    # GPUs (its true TP path, NCCL-backed); `row` is the older weight-only row
    # split, which current CUDA builds no longer provide ("device CUDA0 does not
    # support split buffers") — override per backend for an engine build that
    # only has the old mode.
    llama_tp_extra_args: tuple = ("--split-mode", "tensor")
    # llama.cpp's MoE CPU offload. It spells a layer COUNT the same way
    # (`--n-cpu-moe N` / `-ncmoe N`), but `all` is a TensorSharp-only value for
    # that flag: llama.cpp parses `-ncmoe`'s argument as an integer and spells
    # "every layer" as a separate switch (`--cpu-moe` / `-cmoe`, the mapping
    # this repo's FEATURES.md states). So the `all` axis point emits
    # `llama_cpu_moe_all_arg` instead of `--n-cpu-moe all`, which llama-server
    # would reject at startup.
    # llama.cpp also has no MoE-specific worker pool, so the host-thread knob
    # lands on its global `--threads`: that is the nearest equivalent, not the
    # same thing, and it moves every CPU-side op. Leave `--cpu-moe-threads` at 0
    # (the default) to send no thread count at all.
    llama_cpu_moe: Optional[bool] = None
    llama_cpu_moe_arg: str = "--n-cpu-moe"
    llama_cpu_moe_all_arg: str = "--cpu-moe"
    llama_cpu_moe_threads_arg: str = "--threads"
    # Multi-GPU device selection: the env var that restricts a launched engine to
    # the first N devices (so a `--tp N` cell really uses N GPUs and a tp=1 cell
    # really uses one). None = infer from the backend id / TensorSharp backend.
    visible_devices_env: Optional[str] = None
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
        ts_tp=(bool(ts["tp"]) if ts.get("tp") is not None else None),
        ts_tp_arg=str(ts.get("tp_arg", "--tp")),
        ts_cpu_moe=(bool(ts["cpu_moe"]) if ts.get("cpu_moe") is not None else None),
        ts_cpu_moe_arg=str(ts.get("cpu_moe_arg", "--n-cpu-moe")),
        ts_cpu_moe_threads_arg=str(ts.get("cpu_moe_threads_arg", "--cpu-moe-threads")),
        llama_ngl=(int(llama["ngl"]) if llama.get("ngl") is not None else None),
        llama_server_exe=_path(llama.get("server_exe")),
        llama_extra_args=tuple(str(a) for a in llama.get("extra_args", [])),
        llama_env={str(k): str(v) for k, v in (llama.get("env") or {}).items()},
        llama_tp=(bool(llama["tp"]) if llama.get("tp") is not None else None),
        llama_tp_extra_args=tuple(str(a) for a in llama.get(
            "tp_extra_args", ["--split-mode", "row"])),
        llama_cpu_moe=(bool(llama["cpu_moe"]) if llama.get("cpu_moe") is not None else None),
        llama_cpu_moe_arg=str(llama.get("cpu_moe_arg", "--n-cpu-moe")),
        llama_cpu_moe_all_arg=str(llama.get("cpu_moe_all_arg", "--cpu-moe")),
        llama_cpu_moe_threads_arg=str(llama.get("cpu_moe_threads_arg", "--threads")),
        visible_devices_env=b.get("visible_devices_env"),
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


# ---------------------------------------------------------------------------
# Tensor parallelism (multi-GPU) helpers
# ---------------------------------------------------------------------------
# TensorSharp's `--tp N` splits one model across N local GPUs; llama.cpp's
# equivalent is `--split-mode row` over N devices. Both are restricted to the
# selected devices through the backend's visible-devices env var, so a `tp=N`
# cell really occupies N GPUs and the `tp=1` baseline really occupies one.
_TS_TP_BACKENDS = ("cuda", "ggml_cuda", "ggml_vulkan")


def ts_tp_supported(spec: BackendSpec) -> bool:
    if spec.ts_tp is not None:
        return spec.ts_tp
    return spec.ts_backend in _TS_TP_BACKENDS


def llama_tp_supported(spec: BackendSpec) -> bool:
    if spec.llama_tp is not None:
        return spec.llama_tp
    return spec.kind == "gpu" and bool(spec.llama_ngl)


def visible_devices_env_for(spec: BackendSpec) -> Optional[str]:
    """Env var that restricts a launched engine to a subset of GPUs on this
    backend (explicit config value, else inferred from the backend id)."""
    if spec.visible_devices_env is not None:
        return spec.visible_devices_env or None
    tag = f"{spec.backend_id} {spec.ts_backend or ''}".lower()
    if "vulkan" in tag:
        return "GGML_VK_VISIBLE_DEVICES"
    if "cuda" in tag:
        return "CUDA_VISIBLE_DEVICES"
    return None


def tp_device_env(backend: str, tp: int) -> dict:
    """`{VISIBLE_DEVICES_VAR: "0,1,..."}` pinning a launched engine to the first
    `tp` devices of the configured pool. Empty when the backend has no
    visible-devices var, or when no device pool is configured and tp == 1 (the
    historical single-GPU behaviour: let the engine pick)."""
    spec = BACKENDS.get(backend) or BACKENDS.get(resolve_backend(backend) or "")
    if spec is None:
        return {}
    var = visible_devices_env_for(spec)
    if not var:
        return {}
    pool = TP_DEVICES
    if not pool:
        if tp <= 1:
            return {}
        pool = list(range(tp))
    if len(pool) < tp:
        return {}
    return {var: ",".join(str(d) for d in pool[:tp])}


# ---------------------------------------------------------------------------
# MoE CPU offload (`--n-cpu-moe` / `--cpu-moe-threads`)
# ---------------------------------------------------------------------------
# `--n-cpu-moe N` keeps the routed experts of the first N layers in system RAM
# and multiplies them on the host; attention, the norms, the router and the
# always-active shared expert stay on the accelerator. `--cpu-moe-threads M`
# sizes that host matmul. Both are launch options, so a cell that changes them
# relaunches the server, exactly like `--tp`.
#
# This is a FIT knob, not a speed one: it is what makes a checkpoint that does
# not fit run at all, and it costs decode throughput to do it. That is precisely
# why it belongs on the matrix — the price has to be measurable, not folklore.
CPU_MOE_ALL = -1                  # `--n-cpu-moe all`: every layer


@dataclass(frozen=True)
class CpuMoeSpec:
    """One point on the MoE-CPU-offload axis (layers offloaded + host threads).

    The default instance is the baseline "off" point: it emits no launch flags
    at all and keeps a cell byte-identical to a run that never knew about this
    axis."""
    layers: int = 0               # 0 = off; CPU_MOE_ALL = every layer
    threads: int = 0              # 0 = let the engine pick its own default

    @property
    def active(self) -> bool:
        return self.layers != 0

    @property
    def layers_arg(self) -> str:
        """The value handed to `--n-cpu-moe`."""
        return "all" if self.layers == CPU_MOE_ALL else str(self.layers)

    @property
    def tag(self) -> str:
        """Result-filename / log-name suffix. Empty for the baseline point, so
        existing result files keep their historical names."""
        if not self.active:
            return ""
        return f"__ncmoe{self.layers_arg}" + (f"t{self.threads}" if self.threads > 0 else "")

    @property
    def label(self) -> str:
        """Human-readable axis point for console + report output."""
        if not self.active:
            return "off"
        return self.layers_arg + (f"/{self.threads}t" if self.threads > 0 else "")


def parse_cpu_moe_layers(value) -> int:
    """`off`/`0` -> 0, `all` -> CPU_MOE_ALL, `8` -> 8. Raises on anything else.

    The vocabulary is the server flag's own (a non-negative integer, or `all`)
    plus the single word `off`, which is this axis's name for the baseline point
    and the only spelling the harness adds. There is deliberately no second word
    for it: `none` and friends are rejected rather than quietly accepted, so a
    typo in a sweep is an error instead of a silently un-offloaded cell."""
    tok = str(value).strip().lower()
    if tok in ("off", "0"):
        return 0
    if tok == "all":
        return CPU_MOE_ALL
    n = int(tok)                                  # ValueError propagates
    if n < 0:
        raise ValueError(f"negative --n-cpu-moe layer count: {value}")
    return n


def cpu_moe_axis(layer_values, thread_values) -> list:
    """The `(layers, threads)` axis points for a run, ordered and deduped.

    Threads are only meaningful where something is actually offloaded, so the
    off point collapses to a single baseline cell however many thread counts
    were requested — otherwise `--n-cpu-moe off --cpu-moe-threads 32,64` would
    silently run the identical baseline twice."""
    out: list = []
    for lv in layer_values:
        layers = lv if isinstance(lv, int) else parse_cpu_moe_layers(lv)
        for tv in (thread_values or [0]):
            threads = int(tv)
            if threads < 0:
                raise ValueError(f"negative --cpu-moe-threads value: {tv}")
            spec = CpuMoeSpec(layers=layers, threads=threads if layers != 0 else 0)
            if spec not in out:
                out.append(spec)
    return out or [CpuMoeSpec()]


def ts_cpu_moe_supported(spec: BackendSpec) -> bool:
    """TensorSharp can offload this backend's routed experts to the host.
    Inferred as "any GPU-kind backend TensorSharp can launch": offloading to the
    CPU is only meaningful when the experts would otherwise live on an
    accelerator."""
    if spec.ts_cpu_moe is not None:
        return spec.ts_cpu_moe
    return spec.ts_backend is not None and spec.kind == "gpu"


def llama_cpu_moe_supported(spec: BackendSpec) -> bool:
    if spec.llama_cpu_moe is not None:
        return spec.llama_cpu_moe
    return spec.kind == "gpu" and bool(spec.llama_ngl)


# Every spelling of the offload knobs an engine understands, so a backend entry
# that already pins one in its own `extra_args` / `env` can be detected. These
# are the ENGINE's aliases, not new harness options: the harness still emits
# exactly one spelling (the backend's `*_arg`), it just has to recognise the
# others to see a collision.
_TS_CPU_MOE_ALIASES = ("--n-cpu-moe", "-ncmoe", "--cpu-moe", "-cmoe")
_TS_CPU_MOE_THREAD_ALIASES = ("--cpu-moe-threads",)
_TS_CPU_MOE_ENV = ("TS_N_CPU_MOE", "TS_CPU_MOE")
_TS_CPU_MOE_THREAD_ENV = ("TS_CPU_MOE_THREADS",)
_LLAMA_CPU_MOE_ALIASES = ("--n-cpu-moe", "-ncmoe", "--cpu-moe", "-cmoe")
_LLAMA_CPU_MOE_THREAD_ALIASES = ("--threads", "-t")


def cpu_moe_pinned(spec: BackendSpec, engine: str, threads: int = 0) -> str:
    """The offload setting this backend entry already PINS, or "".

    A backend entry may hardcode the offload itself — that is how this axis was
    measured before it existed, as a cloned backend per point (see
    `ggml_cuda_layer_cpu_moe4` in benchmark_config_deepseek41.json). The axis
    would then append a SECOND copy of the same flag and every engine here keeps
    whichever it parsed last, so the cell would quietly measure one offload
    while its record claimed the other. Report the collision instead; the caller
    turns it into a skip with a reason.

    `threads` > 0 also checks the host-thread knob, which is only a conflict
    when the axis is actually going to send one."""
    if engine == "tensorsharp":
        args, flags = spec.ts_extra_args, _TS_CPU_MOE_ALIASES
        env, env_keys = spec.ts_env, _TS_CPU_MOE_ENV
        if threads > 0:
            flags = flags + _TS_CPU_MOE_THREAD_ALIASES
            env_keys = env_keys + _TS_CPU_MOE_THREAD_ENV
    elif engine == "llamacpp":
        args, flags = spec.llama_extra_args, _LLAMA_CPU_MOE_ALIASES
        env, env_keys = spec.llama_env, ()
        if threads > 0:
            flags = flags + _LLAMA_CPU_MOE_THREAD_ALIASES
    else:
        return ""
    for a in args:
        token = str(a).split("=", 1)[0]
        if token in flags:
            return token
    for key in env_keys:
        if key in (env or {}):
            return key
    return ""


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
#   * tp          - tensor-parallel degree (GPUs one model is split across)
#   * cpu_moe     - routed-expert CPU offload (`--n-cpu-moe` / `--cpu-moe-threads`)
DEFAULT_MTP_MODES = [bool(x) for x in _defaults.get("mtp_modes", [False])]
DEFAULT_CONCURRENCY = [int(x) for x in _defaults.get("concurrency", [1])]
DEFAULT_TP_DEGREES = [int(x) for x in _defaults.get("tp_degrees", [1])]
DEFAULT_CPU_MOE_LAYERS = [parse_cpu_moe_layers(x)
                          for x in _defaults.get("cpu_moe_layers", ["off"])]
DEFAULT_CPU_MOE_THREADS = [int(x) for x in _defaults.get("cpu_moe_threads", [0])]

# Ordered pool of GPU device ids the TP axis may use (`defaults.tp_devices`).
# When set, every launched engine is pinned to the first `tp` of them; when
# empty, tp=1 leaves device selection to the engine and tp>1 uses devices
# 0..tp-1. Overridable per host with BENCH_TP_DEVICES="0,1,2,3".
def _parse_devices(v) -> list:
    if v in (None, ""):
        return []
    if isinstance(v, str):
        v = [x for x in v.replace(";", ",").split(",") if x.strip()]
    return [int(str(x).strip()) for x in v]


TP_DEVICES = _parse_devices(_env_or("BENCH_TP_DEVICES", _defaults.get("tp_devices")))

# Reasoning ("thinking") mode, applied identically to every engine (each spells
# it differently and they do NOT default the same way — see engines.thinking_body).
# Off by default: the answer then fits the per-scenario token budget on both
# sides, which is what makes the output-quality comparison meaningful.
THINKING = bool(_defaults.get("thinking", False))

DEFAULT_MAX_TOKENS = int(_defaults.get("max_tokens", 128))
DEFAULT_WARMUP = int(_defaults.get("warmup", 1))
# Server max-tokens headroom (must cover the busiest cell).
SERVER_MAX_TOKENS = int(_defaults.get("server_max_tokens", 512))


# ---------------------------------------------------------------------------
# Applicability gating
# ---------------------------------------------------------------------------
def applies(engine: str, backend: str, model: ModelSpec,
            scenario: ScenarioSpec, mtp: bool = False,
            tp: int = 1, cpu_moe: Optional[CpuMoeSpec] = None) -> tuple[bool, str]:
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

    # Tensor parallelism: one model split across `tp` GPUs. Both server engines
    # can do it (TensorSharp `--tp N`, llama.cpp `--split-mode row` over N
    # devices); the connect-only vLLM engine cannot be driven into it here.
    # A model whose weights do not fit a single GPU declares `min_tp`, and cells
    # below that degree are skipped rather than left to OOM.
    tp = int(tp or 1)
    if tp > 1:
        if b.kind != "gpu":
            return False, f"tensor parallelism needs a GPU backend ({b.backend_id} is {b.kind})"
        if engine == "tensorsharp" and not ts_tp_supported(b):
            return False, f"TensorSharp has no tensor-parallel path on {b.backend_id}"
        if engine == "llamacpp" and not llama_tp_supported(b):
            return False, f"llama.cpp has no tensor-parallel path on {b.backend_id}"
        if engine == "vllm":
            return False, f"{eng.display} is not driven into tensor parallelism by this harness"
        if TP_DEVICES and len(TP_DEVICES) < tp:
            return False, (f"only {len(TP_DEVICES)} GPU(s) configured "
                           f"(defaults.tp_devices), need {tp}")
    if tp < model.min_tp:
        return False, (f"{model.short_id} needs --tp {model.min_tp} "
                       f"(does not fit {tp} GPU(s))")
    # `max_tp` is a TensorSharp-side shard-divisibility limit; llama.cpp splits
    # the same model differently and is not bound by it.
    if engine == "tensorsharp" and model.max_tp and tp > model.max_tp:
        return False, (f"{model.short_id} cannot be split across more than "
                       f"{model.max_tp} GPU(s) by TensorSharp")

    # MoE CPU offload: the routed experts of `layers` layers stay in system RAM
    # and are multiplied on the host. Only the two server engines are launched
    # by this harness, so only they can be driven into it.
    cm = cpu_moe or CpuMoeSpec()
    if cm.active:
        if b.kind != "gpu":
            return False, (f"MoE CPU offload is meaningless on the {b.backend_id} "
                           f"backend (its experts are already host-resident)")
        if engine == "tensorsharp" and not ts_cpu_moe_supported(b):
            return False, f"TensorSharp has no MoE CPU-offload path on {b.backend_id}"
        if engine == "llamacpp" and not llama_cpu_moe_supported(b):
            return False, f"llama.cpp has no MoE CPU-offload path on {b.backend_id}"
        if engine == "vllm":
            return False, f"{eng.display} is not driven into MoE CPU offload by this harness"
        # A backend that already pins the offload in its own extra_args/env is
        # the pre-axis way of measuring this (one cloned backend per point).
        # Sweeping the axis on top of it would hand the engine the flag twice
        # and silently keep whichever it parsed last, so refuse the cell and
        # name what to fix rather than record a number under the wrong label.
        pinned = cpu_moe_pinned(b, engine, cm.threads)
        if pinned:
            return False, (f"{b.backend_id} already pins {pinned} for {eng.display}; "
                           f"sweep --n-cpu-moe on a backend that does not")
        # Only an explicit `"is_moe": false` gates a model out; an absent key
        # means the config never said, and the cell runs so the engine — not a
        # guess here — is what reports a dense checkpoint.
        if model.is_moe is False:
            return False, f"{model.short_id} is declared dense (`is_moe: false`): no routed experts"

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
        # code_edit is plain multi-turn text, so it stays; agentic needs tool
        # declarations, which this family cannot carry.
        if scenario.kind not in ("text", "multi_turn", "code_edit"):
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
