"""Owned-process lifecycle for the local regression harness (standard library only)."""
from __future__ import annotations

import os
import ctypes
import json
from pathlib import Path
import re
import signal
import subprocess
import threading
import time
import sys

_SECRET_KEY = re.compile(r"(?:TOKEN|PASSWORD|SECRET|API[_-]?KEY|CREDENTIAL|AUTHORIZATION)", re.I)
_PREFIXES = ("TS_", "TENSORSHARP_", "GDN_", "QWEN35_", "FUSED_", "DIFFUSION_")
_SETTING_KEYS = {"KV_CACHE_DTYPE", "MAX_CONTEXT", "MAX_TOKENS", "VIDEO_MAX_FRAMES", "VIDEO_SAMPLE_FPS"}
UNSAFE_RESOURCES_REASON = "Unsafe owned resources: process cleanup was not verified; remaining execution halted"
_unsafe_resources = None


def containment_failure_evidence(value):
    """Find explicit failure or attempted-but-unverified teardown, even in a pass."""
    if isinstance(value, dict):
        teardown = value.get("teardown")
        if value.get("containment_failure") or (isinstance(teardown, dict) and teardown.get("attempted") and teardown.get("verified") is not True):
            return {key: value[key] for key in ("stage", "command", "owned_pid", "exit_code", "parent_exit_code", "containment_failure", "teardown", "stdout_path", "stderr_path") if key in value}
        for child in value.values():
            found = containment_failure_evidence(child)
            if found is not None:
                return found
    elif isinstance(value, (list, tuple)):
        for child in value:
            found = containment_failure_evidence(child)
            if found is not None:
                return found
    return None


class ProcessContainmentError(RuntimeError):
    def __init__(self, evidence):
        super().__init__(UNSAFE_RESOURCES_REASON)
        self.evidence = evidence


def unsafe_process_state():
    return _unsafe_resources


def _record_unsafe_resources(result):
    global _unsafe_resources
    evidence = containment_failure_evidence(result)
    if evidence is not None and _unsafe_resources is None:
        _unsafe_resources = evidence


def require_safe_process_launch():
    # Sticky for this interpreter: a later parent exit or a new model must never
    # silently clear uncertainty about previously owned descendants.
    if _unsafe_resources is not None:
        raise ProcessContainmentError(_unsafe_resources)


def redact(value: str, secrets=()) -> str:
    """Remove known credentials and common credential forms before persistence."""
    value = str(value)
    for secret in sorted({str(x) for x in secrets if x and len(str(x)) >= 4}, key=len, reverse=True):
        value = value.replace(secret, "[REDACTED]")
    value = re.sub(r"(?i)(Bearer\s+)[A-Za-z0-9._~+/=-]+", r"\1[REDACTED]", value)
    value = re.sub(r"(?i)(https?://)[^\s/:@]+:[^\s/@]+@", r"\1[REDACTED]@", value)
    value = re.sub(r"(?i)([?&](?:token|access_token|api_key|key|signature)=)[^&#\s]+", r"\1[REDACTED]", value)
    value = re.sub(r'(?i)((?:password|api[_-]?key|authorization|access[_-]?token)\s*[=:]\s*["\']?)[^\s,"\'}]+', r"\1[REDACTED]", value)
    return value


def secret_values(env=None):
    return [v for k, v in (os.environ if env is None else env).items() if _SECRET_KEY.search(k)]


def clean_environment(overrides=None, artifact_dir=None, *, environment_mode="model_host"):
    """Scrub ambient settings in every mode; bound actual inference hosts by default.

    Only the fixed, model-free component suites opt out of host-policy defaults:
    those tests construct their own permission, context and storage fixtures.
    """
    if environment_mode not in {"model_host", "model_free_suite"}:
        raise ValueError("unknown process environment mode")
    result = {k: v for k, v in os.environ.items()
              if not k.upper().startswith(_PREFIXES) and k.upper() not in _SETTING_KEYS
              and not _SECRET_KEY.search(k)}
    for k, v in (overrides or {}).items():
        if _SECRET_KEY.search(k):
            continue
        result[str(k)] = str(v)
    if environment_mode == "model_host":
        # Actual inference fixtures never require model-generated host commands
        # or discovered skills, even when an override attempts to enable them.
        result.update(TS_CODE_EXEC="0", TS_SKILLS_ALLOW_EXEC="0", TS_SKILLS_ALLOW_NETWORK="0", TS_NO_SKILLS="1")
        if artifact_dir:
            for key, folder in (("TENSORSHARP_PREFIX_CACHE_DIR", "prefix-cache"),
                                ("TENSORSHARP_UPLOAD_DIR", "uploads"), ("TENSORSHARP_LOG_DIR", "host-logs"),
                                ("TS_STUDIO_DIRECTORY", "studio")):
                result[key] = str(Path(artifact_dir).resolve() / folder)
        result.setdefault("MAX_CONTEXT", "2048")
    return result


class _Capture:
    def __init__(self, stream, limit=4 * 1024 * 1024):
        self.stream, self.limit = stream, limit
        self.head, self.tail, self.total = bytearray(), bytearray(), 0
        self.lock = threading.Lock()
        self.thread = threading.Thread(target=self._read, daemon=True)
        self.thread.start()

    def _read(self):
        try:
            while True:
                chunk = self.stream.read1(8192)
                if not chunk:
                    break
                with self.lock:
                    self.total += len(chunk)
                    take = min(len(chunk), self.limit // 2 - len(self.head))
                    self.head.extend(chunk[:take])
                    self.tail.extend(chunk[take:])
                    if len(self.tail) > self.limit // 2:
                        del self.tail[:-self.limit // 2]
        finally:
            self.stream.close()

    def read(self):
        with self.lock:
            truncated = self.total > self.limit
            data = bytes(self.head) + (b"\n[OUTPUT TRUNCATED]\n" if truncated else b"") + bytes(self.tail)
        return data.decode("utf-8", "replace"), truncated


class _WindowsJob:
    """Kernel-owned descendant containment, assigned before the target can start.

    The job disallows breakaway and kills members when its sole handle closes.
    Unlike taskkill /T, it remains usable after the original parent has exited.
    """
    def __init__(self):
        from ctypes import wintypes
        class BasicLimits(ctypes.Structure):
            _fields_ = [("process_time", ctypes.c_int64), ("job_time", ctypes.c_int64),
                        ("flags", wintypes.DWORD), ("min_working_set", ctypes.c_size_t),
                        ("max_working_set", ctypes.c_size_t), ("active_limit", wintypes.DWORD),
                        ("affinity", ctypes.c_size_t), ("priority", wintypes.DWORD),
                        ("scheduling", wintypes.DWORD)]
        class IoCounters(ctypes.Structure):
            _fields_ = [(name, ctypes.c_uint64) for name in
                        ("read_ops", "write_ops", "other_ops", "read_bytes", "write_bytes", "other_bytes")]
        class ExtendedLimits(ctypes.Structure):
            _fields_ = [("basic", BasicLimits), ("io", IoCounters),
                        ("process_memory", ctypes.c_size_t), ("job_memory", ctypes.c_size_t),
                        ("peak_process_memory", ctypes.c_size_t), ("peak_job_memory", ctypes.c_size_t)]
        class Accounting(ctypes.Structure):
            _fields_ = [(name, ctypes.c_int64) for name in ("user", "kernel", "period_user", "period_kernel")] + [
                (name, wintypes.DWORD) for name in ("page_faults", "total_processes", "active_processes", "terminated_processes")]
        self.accounting_type = Accounting
        self.kernel = ctypes.WinDLL("kernel32", use_last_error=True)
        self.kernel.CreateJobObjectW.argtypes = [ctypes.c_void_p, wintypes.LPCWSTR]
        self.kernel.CreateJobObjectW.restype = wintypes.HANDLE
        self.kernel.SetInformationJobObject.argtypes = [wintypes.HANDLE, ctypes.c_int, ctypes.c_void_p, wintypes.DWORD]
        self.kernel.SetInformationJobObject.restype = wintypes.BOOL
        self.kernel.AssignProcessToJobObject.argtypes = [wintypes.HANDLE, wintypes.HANDLE]
        self.kernel.AssignProcessToJobObject.restype = wintypes.BOOL
        self.kernel.QueryInformationJobObject.argtypes = [wintypes.HANDLE, ctypes.c_int, ctypes.c_void_p, wintypes.DWORD, ctypes.c_void_p]
        self.kernel.QueryInformationJobObject.restype = wintypes.BOOL
        self.kernel.TerminateJobObject.argtypes = [wintypes.HANDLE, wintypes.UINT]
        self.kernel.TerminateJobObject.restype = wintypes.BOOL
        self.kernel.CloseHandle.argtypes = [wintypes.HANDLE]
        self.kernel.CloseHandle.restype = wintypes.BOOL
        self.kernel.OpenProcess.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
        self.kernel.OpenProcess.restype = wintypes.HANDLE
        self.kernel.WaitForSingleObject.argtypes = [wintypes.HANDLE, wintypes.DWORD]
        self.kernel.WaitForSingleObject.restype = wintypes.DWORD
        self.handle = self.kernel.CreateJobObjectW(None, None)
        if not self.handle:
            raise ctypes.WinError(ctypes.get_last_error())
        limits = ExtendedLimits()
        limits.basic.flags = 0x00002000  # JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE, no breakaway flags.
        if not self.kernel.SetInformationJobObject(self.handle, 9, ctypes.byref(limits), ctypes.sizeof(limits)):
            error = ctypes.WinError(ctypes.get_last_error())
            self.close()
            raise error

    def assign(self, process):
        if not self.kernel.AssignProcessToJobObject(self.handle, int(process._handle)):
            raise ctypes.WinError(ctypes.get_last_error())

    def active(self):
        info = self.accounting_type()
        if not self.kernel.QueryInformationJobObject(self.handle, 1, ctypes.byref(info), ctypes.sizeof(info), None):
            raise ctypes.WinError(ctypes.get_last_error())
        return int(info.active_processes)

    def terminate(self):
        if not self.kernel.TerminateJobObject(self.handle, 1):
            raise ctypes.WinError(ctypes.get_last_error())

    def member_handles(self):
        """Retain process identities to verify termination, never to kill by PID."""
        size = 4096
        while True:
            buffer = ctypes.create_string_buffer(size)
            if self.kernel.QueryInformationJobObject(self.handle, 3, buffer, size, None):
                break
            error = ctypes.get_last_error()
            if error != 234 or size >= 1024 * 1024:  # ERROR_MORE_DATA
                raise ctypes.WinError(error)
            size *= 2
        count = ctypes.c_uint32.from_buffer(buffer, 4).value
        pids = (ctypes.c_size_t * count).from_buffer(buffer, 8)
        handles = []
        try:
            for pid in pids:
                handle = self.kernel.OpenProcess(0x00100000, False, int(pid))  # SYNCHRONIZE
                if handle:
                    handles.append(handle)
                elif ctypes.get_last_error() not in (87, 1168):  # exited process / not found
                    raise ctypes.WinError(ctypes.get_last_error())
            return handles
        except BaseException:
            self.close_members(handles)
            raise

    def members_exited(self, handles):
        states = [self.kernel.WaitForSingleObject(handle, 0) for handle in handles]
        if any(state not in (0, 258) for state in states):
            raise ctypes.WinError(ctypes.get_last_error())
        return all(state == 0 for state in states)

    def close_members(self, handles):
        for handle in handles:
            self.kernel.CloseHandle(handle)

    def close(self):
        if getattr(self, "handle", None):
            self.kernel.CloseHandle(self.handle)
            self.handle = None

    def __del__(self):
        self.close()


# Isolated Python does no user-site startup and waits for assignment before it can
# launch the target. Closing stdin on an assignment failure makes it exit without
# executing the command. This avoids the assign-after-spawn descendant escape race.
_WINDOWS_LAUNCHER = ("import sys, subprocess\n"
                     "if sys.stdin.buffer.readline() != b'GO\\n': raise SystemExit(126)\n"
                     "raise SystemExit(subprocess.call(sys.argv[1:], stdin=subprocess.DEVNULL))\n")


def _posix_group_state(pgid):
    """Observe the owned group independently of the original parent's liveness."""
    try:
        os.killpg(pgid, 0)
    except ProcessLookupError:
        return {"active": False, "live_pids": [], "observation": "process group absent"}
    except PermissionError:
        return {"active": True, "live_pids": None, "observation": "process group exists but is not inspectable"}
    # Linux containers can leave adopted zombies unreaped. Zombies cannot run or
    # hold GPU allocations; distinguish them from a child ignoring SIGTERM.
    proc = Path("/proc")
    if sys.platform.startswith("linux") and proc.is_dir():
        live, zombies, uncertain = [], [], False
        try:
            for entry in proc.iterdir():
                if not entry.name.isdigit():
                    continue
                try:
                    fields = (entry / "stat").read_text().rsplit(")", 1)[1].split()
                    if int(fields[2]) == pgid:
                        (zombies if fields[0] == "Z" else live).append(int(entry.name))
                except FileNotFoundError:
                    pass  # Process exited during the inventory.
                except (OSError, ValueError, IndexError):
                    uncertain = True
            if live or zombies or not uncertain:
                return {"active": bool(live) or uncertain, "live_pids": live, "zombie_pids": zombies,
                        "incomplete_inventory": uncertain,
                        "observation": "Linux process-group members (zombies are not running)"}
        except OSError:
            pass
    return {"active": True, "live_pids": None, "observation": "process group exists"}


class ManagedProcess:
    """Only terminates this process and its descendants; never searches process names."""

    def __init__(self, command, *, cwd, artifact_dir, stage, env=None, capture_limit=4 * 1024 * 1024,
                 environment_mode="model_host"):
        if not command or isinstance(command, str):
            raise ValueError("command must be a nonempty argv list, never a shell command")
        if not re.fullmatch(r"[A-Za-z0-9_.-]+", stage):
            raise ValueError("invalid artifact stage")
        require_safe_process_launch()
        self.command = [str(x) for x in command]
        self.artifact_dir, self.stage = Path(artifact_dir), stage
        self.artifact_dir.mkdir(parents=True, exist_ok=True)
        self.secrets = secret_values() + secret_values(env or {})
        self.started = time.monotonic()
        self.timed_out = False
        self.cancelled = False
        self._job = None
        self.teardown = {"attempted": False, "verified": False,
                         "mechanism": "windows-job" if os.name == "nt" else "posix-process-group",
                         "reason": "process lifecycle has not completed"}
        self.env = clean_environment(env, artifact_dir, environment_mode=environment_mode)
        self.environment_mode = environment_mode
        kwargs = {"creationflags": subprocess.CREATE_NO_WINDOW | subprocess.CREATE_NEW_PROCESS_GROUP} if os.name == "nt" else {"start_new_session": True}
        launch_command = self.command
        if os.name == "nt":
            self._job = _WindowsJob()
            launch_command = [sys.executable, "-I", "-S", "-u", "-c", _WINDOWS_LAUNCHER, *self.command]
        try:
            self.process = subprocess.Popen(launch_command, cwd=str(cwd), env=self.env,
                                            stdin=subprocess.PIPE if self._job else subprocess.DEVNULL,
                                            stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=False, **kwargs)
            if self._job:
                try:
                    self._job.assign(self.process)
                    self.process.stdin.write(b"GO\n")
                    self.process.stdin.flush()
                except BaseException:
                    self.process.stdin.close()
                    self.process.kill()
                    self.process.wait(timeout=5)
                    self.process.stdout.close()
                    self.process.stderr.close()
                    raise
                self.process.stdin.close()
        except BaseException:
            if self._job:
                self._job.close()
            raise
        self.pgid = self.process.pid if os.name != "nt" else None
        self.stdout = _Capture(self.process.stdout, capture_limit)
        self.stderr = _Capture(self.process.stderr, capture_limit)

    def stop(self):
        if self.teardown["verified"]:
            return self.teardown
        self.teardown.update(attempted=True, actions=[], verified=False)
        member_handles = []
        try:
            if self._job:
                # Query/terminate the job even when its original process is gone.
                before = self._job.active()
                self.teardown["active_before"] = before
                member_handles = self._job.member_handles()
                self.teardown["observed_member_handles"] = len(member_handles)
                if before:
                    self._job.terminate()
                    self.teardown["actions"].append("TerminateJobObject")
                deadline = time.monotonic() + 5
                active = self._job.active()
                members_exited = self._job.members_exited(member_handles)
                while (active or not members_exited) and time.monotonic() < deadline:
                    self.process.poll()
                    time.sleep(0.02)
                    active = self._job.active()
                    members_exited = self._job.members_exited(member_handles)
                verified = active == 0 and members_exited
                self.teardown.update(verified=verified, active_after=active, observed_members_exited=members_exited,
                    reason="job reports zero active processes and observed members have exited" if verified else "job members have not all exited")
                if verified:
                    self._job.close()
            else:
                self.process.poll()  # Reap an exited parent; do not use it as the group verdict.
                state = _posix_group_state(self.pgid)
                self.teardown["before"] = state
                if state["active"]:
                    try:
                        os.killpg(self.pgid, signal.SIGTERM)
                        self.teardown["actions"].append("SIGTERM owned process group")
                    except ProcessLookupError:
                        pass
                    deadline = time.monotonic() + 0.5
                    while time.monotonic() < deadline:
                        self.process.poll()
                        state = _posix_group_state(self.pgid)
                        if not state["active"]:
                            break
                        time.sleep(0.02)
                    if state["active"]:
                        try:
                            os.killpg(self.pgid, signal.SIGKILL)
                            self.teardown["actions"].append("SIGKILL owned process group")
                        except ProcessLookupError:
                            pass
                deadline = time.monotonic() + 5
                while True:
                    self.process.poll()
                    state = _posix_group_state(self.pgid)
                    if not state["active"] or time.monotonic() >= deadline:
                        break
                    time.sleep(0.02)
                self.teardown.update(verified=not state["active"], after=state,
                    reason="owned process group has no live members" if not state["active"] else "owned process group still has live or unobservable members",
                    scope="owned process group; descendants deliberately creating new sessions are outside POSIX group containment")
            self.process.wait(timeout=5)
        except (OSError, subprocess.TimeoutExpired) as error:
            self.teardown.update(verified=False, reason=type(error).__name__ + ": " + str(error))
            if self._job:
                # Fail closed; closing this private handle requests kill-on-close.
                self._job.close()
                self.teardown["actions"].append("job handle closed; termination outcome unverified")
        finally:
            if self._job:
                self._job.close_members(member_handles)
        return self.teardown

    def wait(self, timeout_seconds):
        try:
            self.process.wait(timeout=max(0.01, timeout_seconds - (time.monotonic() - self.started)))
        except subprocess.TimeoutExpired:
            self.timed_out = True
            self.stop()
        except KeyboardInterrupt:
            self.cancelled = True
            self.stop()
            self.snapshot()
            raise
        # A successful parent exit does not imply its children exited. Always
        # finish containment before returning a completed test to the caller.
        self.stop()
        return self.snapshot()

    def snapshot(self):
        if self.process.poll() is not None:
            self.stdout.thread.join(timeout=2)
            self.stderr.thread.join(timeout=2)
        out, out_cut = self.stdout.read()
        err, err_cut = self.stderr.read()
        out, err = redact(out, self.secrets), redact(err, self.secrets)
        out_file, err_file = self.artifact_dir / (self.stage + ".stdout.log"), self.artifact_dir / (self.stage + ".stderr.log")
        out_file.write_text(out, encoding="utf-8")
        err_file.write_text(err, encoding="utf-8")
        parent_exit = self.process.poll()
        containment_failure = self.teardown["attempted"] and not self.teardown["verified"]
        result = {"command": [redact(v, self.secrets) for v in self.command], "stage": self.stage,
                "environment_mode": self.environment_mode,
                "exit_code": 125 if containment_failure else parent_exit, "parent_exit_code": parent_exit,
                "containment_failure": containment_failure,
                "timed_out": self.timed_out, "cancelled": self.cancelled,
                "teardown": self.teardown, "owned_pid": self.process.pid,
                "duration_seconds": round(time.monotonic() - self.started, 4),
                "stdout": out, "stderr": err, "output_truncated": out_cut or err_cut,
                "stdout_path": str(out_file), "stderr_path": str(err_file)}
        _record_unsafe_resources(result)
        # In particular, cancellation must leave teardown evidence even though
        # run_process re-raises KeyboardInterrupt instead of returning a result.
        metadata = {k: v for k, v in result.items() if k not in {"stdout", "stderr"}}
        text = redact(json.dumps(metadata, ensure_ascii=False, allow_nan=False, indent=2), self.secrets)
        (self.artifact_dir / (self.stage + ".process.json")).write_text(text + "\n", encoding="utf-8")
        return result


def run_process(command, *, cwd, artifact_dir, stage, timeout_seconds=600, env=None, environment_mode="model_host"):
    return ManagedProcess(command, cwd=cwd, artifact_dir=artifact_dir, stage=stage, env=env,
                          environment_mode=environment_mode).wait(timeout_seconds)
