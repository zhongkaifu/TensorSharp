# Copyright (c) Zhongkai Fu. All rights reserved.
# https://github.com/zhongkaifu/TensorSharp
#
# This file is part of TensorSharp.
#
# TensorSharp is licensed under the BSD-3-Clause license found in the LICENSE file in the root directory of this source tree.
#
# TensorSharp is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the BSD-3-Clause License for more details.
"""Start, reuse and stop TensorSharp servers from launch profiles.

A TensorSharp server hosts exactly the one model it was started with, so switching
models means a new process. A profile is a JSON file naming a preset from the
repository's `config/` folder (or a model) and the server's arguments; this module starts
a server for it on the port the workflow chose, reuses one already hosting that model,
and stops the ones it started.

A started server is tied to ComfyUI's lifetime where the platform allows it: a Windows
job object, or `setpriv --pdeathsig` on Linux. It imports nothing from ComfyUI.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import re
import shutil
import signal
import subprocess
import sys
import threading
import time
from dataclasses import dataclass, field
from urllib.parse import urlparse

import aiohttp

log = logging.getLogger("comfyui_tensorsharp")

# The real folder of this pack, through any link ComfyUI's custom_nodes holds to it.
PACK_DIR = os.path.dirname(os.path.realpath(__file__))
# The TensorSharp checkout the pack ships in (examples/comfyui).
REPO_ROOT = os.path.normpath(os.path.join(PACK_DIR, "..", ".."))
PROFILE_DIR = os.path.join(PACK_DIR, "profiles")
EXTRA_PROFILES_ENV = "TENSORSHARP_COMFYUI_PROFILES"
SERVER_DLL_ENV = "TENSORSHARP_SERVER_DLL"
CONNECT_ONLY = "none (connect to a running server)"

_UNSET = re.compile(r"\$\{(\w+)[^}]*\}|\$(\w+)|%(\w+)%")


def default_server_dll(repo_root: str = REPO_ROOT) -> str:
    return os.path.join(repo_root, "TensorSharp.Server.Host", "bin", "TensorSharp.Server.Host.dll")


@dataclass
class Profile:
    name: str
    model: str = ""
    config: str = ""
    args: list[str] = field(default_factory=list)
    env: dict[str, str] = field(default_factory=dict)
    path_prepend: list[str] = field(default_factory=list)
    server_dll: str = ""
    dotnet: str = "dotnet"
    service_name: str = "tensorsharp-comfyui"
    startup_timeout: float = 900.0
    description: str = ""

    def resolved_server_dll(self) -> str:
        return self.server_dll or os.environ.get(SERVER_DLL_ENV, "") or default_server_dll()

    def model_name(self) -> str:
        """The file name of the model this profile hosts, for matching a running server."""
        if self.model:
            return os.path.basename(self.model.replace("\\", "/"))
        return config_model_name(self.config) if self.config else ""

    def command(self, host: str, port: int) -> list[str]:
        dll = self.resolved_server_dll()
        if not os.path.isfile(dll):
            raise FileNotFoundError(
                f"TensorSharp server not found at {dll}. Build TensorSharp.Server.Host, or set "
                f"{SERVER_DLL_ENV} or the profile's server_dll.")
        command = [self.dotnet, dll]
        if self.config:
            if not os.path.isfile(self.config):
                raise FileNotFoundError(f"profile {self.name}: config not found: {self.config}")
            command += ["--config", self.config]
        if self.model:
            if not os.path.isfile(self.model):
                raise FileNotFoundError(f"profile {self.name}: model not found: {self.model}")
            command += ["--model", self.model]
        return command + ["--host", host, "--port", str(port), *self.args]

    def environment(self, base: dict[str, str] | None = None) -> dict[str, str]:
        env = dict(os.environ if base is None else base)
        for key, value in self.env.items():
            env[key] = _expand(value, f"profile {self.name}, env {key}")
        if self.path_prepend:
            extra = os.pathsep.join(_expand(p, f"profile {self.name}, path_prepend") for p in self.path_prepend)
            env["PATH"] = extra + os.pathsep + env.get("PATH", "")
        # The server's own service name, never the one ComfyUI's telemetry may have set.
        env["OTEL_SERVICE_NAME"] = self.service_name
        return env


def _expand(value: str, where: str) -> str:
    """Expand environment variables, refusing any that are not set."""
    expanded = os.path.expandvars(value)
    missing = _UNSET.search(expanded)
    if missing:
        name = next(g for g in missing.groups() if g)
        raise ValueError(f"{where} uses ${name}, which is not set")
    return expanded


def config_model_name(config_path: str) -> str:
    """The model file name a TensorSharp config preset names (`"model": "..."` or
    `"model": { "path": "..." }`), read without evaluating the preset."""
    try:
        with open(config_path, encoding="utf-8") as f:
            text = f.read()
    except OSError:
        return ""
    match = re.search(r'"model"\s*:\s*(?:\{[^{}]*?"path"\s*:\s*)?"([^"]+)"', text)
    return os.path.basename(match.group(1).replace("\\", "/")) if match else ""


def profile_dirs() -> list[str]:
    """Folders searched for profiles: those in TENSORSHARP_COMFYUI_PROFILES first, then the pack's own."""
    extra = [d for d in os.environ.get(EXTRA_PROFILES_ENV, "").split(os.pathsep) if d]
    return [*extra, PROFILE_DIR]


def list_profiles(directories: list[str] | None = None) -> list[str]:
    names = set()
    for directory in directories or profile_dirs():
        if os.path.isdir(directory):
            names.update(f[:-5] for f in os.listdir(directory) if f.endswith(".json") and not f.startswith("_"))
    return sorted(names)


def load_profile(name: str, directories: list[str] | None = None, repo_root: str = REPO_ROOT) -> Profile:
    """Read a profile from the first folder that has it. A relative `config` path is taken
    from the TensorSharp checkout; a relative `model` path from the profile's folder."""
    for directory in directories or profile_dirs():
        path = os.path.join(directory, name + ".json")
        if os.path.isfile(path):
            break
    else:
        raise FileNotFoundError(f"no profile named {name}")
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    known = {k: v for k, v in data.items() if k in Profile.__dataclass_fields__ and k != "name"}
    where = f"profile {name}"
    for key in ("model", "config", "server_dll", "dotnet"):
        if known.get(key):
            known[key] = _expand(known[key], f"{where}, {key}")
    if known.get("args"):
        known["args"] = [_expand(a, f"{where}, args") for a in known["args"]]
    if not known.get("model") and not known.get("config"):
        raise ValueError(f"{where} names neither a config preset nor a model")
    if known.get("config") and not os.path.isabs(known["config"]):
        known["config"] = os.path.normpath(os.path.join(repo_root, known["config"]))
    if known.get("model") and not os.path.isabs(known["model"]):
        known["model"] = os.path.normpath(os.path.join(directory, known["model"]))
    return Profile(name=name, **known)


def hosted_model(models_payload: dict) -> str:
    """The path or name of the model a server reports it hosts, or '' if none."""
    for key in ("loaded", "model", "modelPath", "loadedModel"):
        value = models_payload.get(key)
        if isinstance(value, str) and value:
            return value
        if isinstance(value, dict):
            for inner in ("path", "name", "file"):
                if isinstance(value.get(inner), str) and value[inner]:
                    return value[inner]
    models = models_payload.get("models")
    if isinstance(models, list) and len(models) == 1:
        only = models[0]
        if isinstance(only, str):
            return only
        if isinstance(only, dict):
            return str(only.get("path") or only.get("name") or "")
    return ""


def same_model(reported: str, model_name: str) -> bool:
    if not reported or not model_name:
        return False
    a = os.path.basename(reported.replace("\\", "/")).lower()
    b = os.path.basename(model_name.replace("\\", "/")).lower()
    return a == b or a == os.path.splitext(b)[0]


# ---- Tie each server's lifetime to this process -------------------------------------

_job_handle = None


def _windows_kill_with_parent(proc: subprocess.Popen) -> None:
    global _job_handle
    try:
        import ctypes
        from ctypes import wintypes

        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)

        class IO_COUNTERS(ctypes.Structure):
            _fields_ = [(n, ctypes.c_ulonglong) for n in (
                "ReadOperationCount", "WriteOperationCount", "OtherOperationCount",
                "ReadTransferCount", "WriteTransferCount", "OtherTransferCount")]

        class JOBOBJECT_BASIC_LIMIT_INFORMATION(ctypes.Structure):
            _fields_ = [
                ("PerProcessUserTimeLimit", ctypes.c_int64),
                ("PerJobUserTimeLimit", ctypes.c_int64),
                ("LimitFlags", wintypes.DWORD),
                ("MinimumWorkingSetSize", ctypes.c_size_t),
                ("MaximumWorkingSetSize", ctypes.c_size_t),
                ("ActiveProcessLimit", wintypes.DWORD),
                ("Affinity", ctypes.c_size_t),
                ("PriorityClass", wintypes.DWORD),
                ("SchedulingClass", wintypes.DWORD),
            ]

        class JOBOBJECT_EXTENDED_LIMIT_INFORMATION(ctypes.Structure):
            _fields_ = [
                ("BasicLimitInformation", JOBOBJECT_BASIC_LIMIT_INFORMATION),
                ("IoInfo", IO_COUNTERS),
                ("ProcessMemoryLimit", ctypes.c_size_t),
                ("JobMemoryLimit", ctypes.c_size_t),
                ("PeakProcessMemoryUsed", ctypes.c_size_t),
                ("PeakJobMemoryUsed", ctypes.c_size_t),
            ]

        kernel32.CreateJobObjectW.restype = wintypes.HANDLE
        kernel32.CreateJobObjectW.argtypes = (ctypes.c_void_p, wintypes.LPCWSTR)
        kernel32.SetInformationJobObject.argtypes = (wintypes.HANDLE, ctypes.c_int, ctypes.c_void_p, wintypes.DWORD)
        kernel32.AssignProcessToJobObject.argtypes = (wintypes.HANDLE, wintypes.HANDLE)

        if _job_handle is None:
            job = kernel32.CreateJobObjectW(None, None)
            if not job:
                raise ctypes.WinError(ctypes.get_last_error())
            info = JOBOBJECT_EXTENDED_LIMIT_INFORMATION()
            info.BasicLimitInformation.LimitFlags = 0x2000  # JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE
            if not kernel32.SetInformationJobObject(job, 9, ctypes.byref(info), ctypes.sizeof(info)):
                raise ctypes.WinError(ctypes.get_last_error())
            _job_handle = job  # kept open for this process's lifetime on purpose
        if not kernel32.AssignProcessToJobObject(_job_handle, wintypes.HANDLE(int(proc._handle))):
            raise ctypes.WinError(ctypes.get_last_error())
    except Exception as ex:  # the server still runs; it just will not die with ComfyUI
        log.warning("TensorSharp: could not tie server pid %s to ComfyUI's lifetime: %s", proc.pid, ex)


def launch_prefix() -> list[str]:
    """On Linux, `setpriv --pdeathsig TERM` makes the kernel stop the server when the
    thread that started it ends, which for ComfyUI's prompt worker is when ComfyUI ends."""
    if sys.platform.startswith("linux"):
        setpriv = shutil.which("setpriv")
        if setpriv:
            return [setpriv, "--pdeathsig", "TERM", "--"]
        log.warning("TensorSharp: setpriv (util-linux) not found; a server will outlive a crashed ComfyUI")
    return []


# ---- Managed servers ------------------------------------------------------------------

@dataclass
class _Managed:
    proc: subprocess.Popen
    profile: Profile
    log_path: str


_managed: dict[int, _Managed] = {}
_lock = threading.Lock()


def _host_port(url: str) -> tuple[str, int]:
    parsed = urlparse(url)
    if parsed.scheme not in ("http", "https") or not parsed.hostname:
        raise ValueError(f"not an http URL: {url}")
    return parsed.hostname, parsed.port or (443 if parsed.scheme == "https" else 80)


async def probe(url: str, timeout: float = 3.0) -> dict | None:
    """The server's /api/models reply, or None when nothing answers."""
    try:
        async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=timeout)) as s:
            async with s.get(url.rstrip("/") + "/api/models") as resp:
                if resp.status != 200:
                    return None
                return await resp.json(content_type=None)
    except (aiohttp.ClientError, asyncio.TimeoutError, OSError, ValueError):
        return None


def _tail(path: str, lines: int = 25) -> str:
    try:
        with open(path, "rb") as f:
            f.seek(0, os.SEEK_END)
            f.seek(max(0, f.tell() - 16384))
            return "\n".join(f.read().decode("utf-8", "replace").splitlines()[-lines:])
    except OSError:
        return ""


def owned(url: str) -> bool:
    _, port = _host_port(url)
    m = _managed.get(port)
    return m is not None and m.proc.poll() is None


async def ensure_server(url: str, profile_name: str, log_dir: str, before_start=None) -> dict:
    """Make `url` serve `profile_name`'s model and return its /api/models reply.

    With CONNECT_ONLY the server must already be running. Otherwise a server already
    hosting the profile's model is reused; one this module started for another profile is
    stopped and replaced; one it did not start is never touched. `before_start` runs just
    before a new server is launched (ComfyUI uses it to free its own GPU memory)."""
    info = await probe(url)
    if profile_name in ("", CONNECT_ONLY):
        if info is None:
            raise RuntimeError(f"no TensorSharp server answers at {url}")
        return info

    profile = load_profile(profile_name)
    host, port = _host_port(url)
    with _lock:
        current = _managed.get(port)
    if current is not None and current.proc.poll() is not None:
        with _lock:
            _managed.pop(port, None)
        current = None
    if info is not None:
        if current is not None and current.profile.name == profile.name:
            return info
        if current is None:
            if same_model(hosted_model(info), profile.model_name()):
                return info
            raise RuntimeError(
                f"{url} is a TensorSharp server this ComfyUI did not start, hosting "
                f"{hosted_model(info) or 'another model'}; stop it or use another port for {profile_name}")
    if current is not None:
        await stop_server(url)

    if before_start is not None:
        before_start()
    command = launch_prefix() + profile.command(host, port)
    os.makedirs(log_dir, exist_ok=True)
    log_path = os.path.join(log_dir, f"tensorsharp-{port}.log")
    log_file = open(log_path, "ab")
    log_file.write(f"\n==== {time.strftime('%Y-%m-%d %H:%M:%S')} profile {profile.name}: "
                   f"{subprocess.list2cmdline(command)}\n".encode())
    log_file.flush()
    flags = subprocess.CREATE_NEW_PROCESS_GROUP if sys.platform == "win32" else 0
    env = profile.environment()
    # One Video Studio store per port: a server refuses to start on a store another
    # running server owns, and two profiles on two ports otherwise share the default one.
    env.setdefault("TS_STUDIO_DIRECTORY", os.path.join(log_dir, f"tensorsharp-studio-{port}"))
    proc = subprocess.Popen(
        command, stdout=log_file, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL,
        env=env, cwd=os.path.dirname(profile.resolved_server_dll()), creationflags=flags)
    log_file.close()  # the child holds its own handle
    if sys.platform == "win32":
        _windows_kill_with_parent(proc)
    with _lock:
        _managed[port] = _Managed(proc, profile, log_path)
    log.info("TensorSharp: started pid %s for profile %s on %s (log %s)", proc.pid, profile.name, url, log_path)

    deadline = time.monotonic() + profile.startup_timeout
    while time.monotonic() < deadline:
        if proc.poll() is not None:
            with _lock:
                _managed.pop(port, None)
            raise RuntimeError(
                f"TensorSharp server for {profile.name} exited with code {proc.returncode}:\n{_tail(log_path)}")
        info = await probe(url)
        if info is not None:
            if profile.model_name() and not same_model(hosted_model(info), profile.model_name()):
                log.warning("TensorSharp: %s reports model %r for profile %s", url, hosted_model(info), profile.name)
            return info
        await asyncio.sleep(1.0)
    await stop_server(url)
    raise TimeoutError(f"TensorSharp server for {profile.name} was not ready after {profile.startup_timeout:.0f} s")


async def stop_server(url: str, grace_seconds: float = 20.0) -> bool:
    """Stop the server this module started on `url`'s port. Returns False if it started none.

    It first asks the server to shut down (Ctrl+Break on Windows, SIGTERM elsewhere), so
    its telemetry is flushed, and kills it only if it is still running after the grace period."""
    _, port = _host_port(url)
    with _lock:
        managed = _managed.pop(port, None)
    if managed is None:
        return False
    proc = managed.proc
    if proc.poll() is None:
        try:
            proc.send_signal(signal.CTRL_BREAK_EVENT if sys.platform == "win32" else signal.SIGTERM)
        except (OSError, ValueError) as ex:
            log.warning("TensorSharp: could not ask pid %s to stop: %s", proc.pid, ex)
        deadline = time.monotonic() + grace_seconds
        while proc.poll() is None and time.monotonic() < deadline:
            await asyncio.sleep(0.5)
        if proc.poll() is None:
            log.warning("TensorSharp: pid %s did not stop in %.0f s; killing it", proc.pid, grace_seconds)
            proc.kill()
            proc.wait(timeout=30)
    log.info("TensorSharp: stopped pid %s (profile %s), exit code %s", proc.pid, managed.profile.name, proc.returncode)
    return True


def stop_all() -> None:
    """Kill every server this module started; for interpreter exit, where nothing can wait."""
    with _lock:
        items = list(_managed.values())
        _managed.clear()
    for m in items:
        if m.proc.poll() is None:
            m.proc.kill()
