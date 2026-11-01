"""Owned-process lifecycle for the local regression harness (standard library only)."""
from __future__ import annotations

import os
from pathlib import Path
import re
import signal
import subprocess
import threading
import time

_SECRET_KEY = re.compile(r"(?:TOKEN|PASSWORD|SECRET|API[_-]?KEY|CREDENTIAL|AUTHORIZATION)", re.I)
_PREFIXES = ("TS_", "TENSORSHARP_", "GDN_", "QWEN35_", "FUSED_", "DIFFUSION_")
_SETTING_KEYS = {"KV_CACHE_DTYPE", "MAX_CONTEXT", "MAX_TOKENS", "VIDEO_MAX_FRAMES", "VIDEO_SAMPLE_FPS"}


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


def clean_environment(overrides=None, artifact_dir=None):
    """Do not inherit tuning, host execution permissions or remote credentials."""
    result = {k: v for k, v in os.environ.items()
              if not k.upper().startswith(_PREFIXES) and k.upper() not in _SETTING_KEYS
              and not _SECRET_KEY.search(k)}
    for k, v in (overrides or {}).items():
        if _SECRET_KEY.search(k):
            continue
        result[str(k)] = str(v)
    # Local fixtures never require model-generated host commands or discovered skills.
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


class ManagedProcess:
    """Only terminates this process and its descendants; never searches process names."""

    def __init__(self, command, *, cwd, artifact_dir, stage, env=None, capture_limit=4 * 1024 * 1024):
        if not command or isinstance(command, str):
            raise ValueError("command must be a nonempty argv list, never a shell command")
        if not re.fullmatch(r"[A-Za-z0-9_.-]+", stage):
            raise ValueError("invalid artifact stage")
        self.command = [str(x) for x in command]
        self.artifact_dir, self.stage = Path(artifact_dir), stage
        self.artifact_dir.mkdir(parents=True, exist_ok=True)
        self.secrets = secret_values() + secret_values(env or {})
        self.started = time.monotonic()
        self.timed_out = False
        self.cancelled = False
        self.env = clean_environment(env, artifact_dir)
        kwargs = {"creationflags": subprocess.CREATE_NO_WINDOW | subprocess.CREATE_NEW_PROCESS_GROUP} if os.name == "nt" else {"start_new_session": True}
        self.process = subprocess.Popen(self.command, cwd=str(cwd), env=self.env, stdin=subprocess.DEVNULL,
                                        stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=False, **kwargs)
        self.stdout = _Capture(self.process.stdout, capture_limit)
        self.stderr = _Capture(self.process.stderr, capture_limit)

    def stop(self):
        if self.process.poll() is not None:
            return
        if os.name == "nt":
            # taskkill receives an exact owned PID, not a generated shell expression.
            try:
                subprocess.run(["taskkill", "/PID", str(self.process.pid), "/T", "/F"],
                               stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                               timeout=10, creationflags=subprocess.CREATE_NO_WINDOW, check=False)
            except (OSError, subprocess.TimeoutExpired):
                # Still terminate the owned parent if taskkill itself is unavailable.
                self.process.kill()
        else:
            try:
                os.killpg(self.process.pid, signal.SIGTERM)
                self.process.wait(timeout=3)
            except (ProcessLookupError, subprocess.TimeoutExpired):
                try:
                    os.killpg(self.process.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
        if self.process.poll() is None:
            self.process.kill()
        self.process.wait(timeout=10)

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
        return {"command": [redact(v, self.secrets) for v in self.command], "stage": self.stage,
                "exit_code": self.process.poll(), "timed_out": self.timed_out, "cancelled": self.cancelled,
                "duration_seconds": round(time.monotonic() - self.started, 4),
                "stdout": out, "stderr": err, "output_truncated": out_cut or err_cut,
                "stdout_path": str(out_file), "stderr_path": str(err_file)}


def run_process(command, *, cwd, artifact_dir, stage, timeout_seconds=600, env=None):
    return ManagedProcess(command, cwd=cwd, artifact_dir=artifact_dir, stage=stage, env=env).wait(timeout_seconds)
