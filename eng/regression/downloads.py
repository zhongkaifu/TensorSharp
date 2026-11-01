"""Sequential, checksum-verified model preparation for the local regression suite.

Dry probes never hash models or use the network. Network probes fetch headers only.
Only commit-pinned HTTPS Hugging Face sources already present in a selected preset
are eligible for automatic download; a checksum remains mandatory. No host's
implicit downloader is used, and existing user files are never overwritten.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import http.client
import json
import os
from pathlib import Path
import re
import shutil
import socket
from typing import Callable, Iterable
import urllib.error
import urllib.parse
import urllib.request

try:
    from .configs import FileSpec
except ImportError:
    from configs import FileSpec


class PrepareError(RuntimeError):
    def __init__(self, stage: str, code: str, path: str | Path, message: str):
        super().__init__(message)
        self.stage, self.code, self.path = stage, code, str(path)

    def as_dict(self):
        return {"stage": self.stage, "code": self.code, "path": self.path, "message": str(self)}


@dataclass
class DownloadPolicy:
    cache_root: Path
    max_download_bytes: int
    disk_headroom_bytes: int = 2 * 1024 ** 3
    allow_download: bool = False
    timeout_seconds: float = 30.0
    chunk_bytes: int = 1024 * 1024
    # Test-only injection: disabled by the runner; never resolves arbitrary hosts.
    allow_loopback_http: bool = False
    token: str | None = field(default=None, repr=False)
    downloaded_bytes: int = 0
    # Immutable source metadata may be reused by auto-backend planning and execution.
    # Neither auth headers nor failures are cached; partial/local state is re-read.
    metadata_cache: dict = field(default_factory=dict, repr=False)

    def __post_init__(self):
        self.cache_root = Path(self.cache_root).resolve()
        if self.max_download_bytes < 0 or self.disk_headroom_bytes < 0:
            raise ValueError("Download and disk budgets must be nonnegative")
        if self.timeout_seconds <= 0 or self.chunk_bytes < 1:
            raise ValueError("Download timeout and chunk size must be positive")


@dataclass(frozen=True)
class FileIdentity:
    path: str
    size_bytes: int
    sha256: str | None
    expected_sha256: str | None
    status: str
    source: str | None = None
    integrity: str = "verified-sha256"
    reason: str | None = None


def _cancelled(cancel, path):
    if cancel and cancel():
        raise PrepareError("prepare", "cancelled", path, "Model preparation cancelled; partial files retained")


def _inside(path: Path, root: Path) -> bool:
    try:
        path.resolve().relative_to(root.resolve())
        return True
    except ValueError:
        return False


def _target(spec: FileSpec, policy: DownloadPolicy):
    target = Path(spec.path)
    if not _inside(target, policy.cache_root) or target.resolve() == policy.cache_root:
        raise PrepareError("preflight", "unsafe_path", target,
                           "Download target must remain inside the explicitly selected model root")
    for sibling in (target, Path(str(target) + ".part"), Path(str(target) + ".part.json"),
                    Path(str(target) + ".download.lock")):
        if sibling.is_symlink() or not _inside(sibling, policy.cache_root):
            raise PrepareError("preflight", "unsafe_path", target, "Download target or sidecar traverses a symlink")
    if target.exists() and not target.is_file():
        raise PrepareError("preflight", "unsafe_path", target, "Model target is not a regular file")


def _source_allowed(url: str, policy: DownloadPolicy) -> bool:
    parsed = urllib.parse.urlsplit(url)
    if parsed.username or parsed.password or parsed.fragment or parsed.query:
        return False
    if policy.allow_loopback_http and parsed.scheme == "http" and parsed.hostname in {"127.0.0.1", "::1", "localhost"}:
        return True
    return (parsed.scheme == "https" and parsed.hostname == "huggingface.co"
            and parsed.port in (None, 443)
            and re.fullmatch(r"/[^/]+/[^/]+/resolve/[0-9a-fA-F]{40}/.+", parsed.path) is not None
            and ".." not in parsed.path.split("/"))


def _validate_source(spec, policy):
    if not spec.sha256 or not re.fullmatch(r"[0-9a-fA-F]{64}", spec.sha256):
        raise PrepareError("preflight", "unverified_source", spec.path,
                           "Automatic downloads require a configured SHA-256")
    if not spec.urls:
        raise PrepareError("preflight", "missing_source", spec.path,
                           "Required local file is missing and the preset declares no download source")
    for url in spec.urls:
        if not _source_allowed(url, policy):
            raise PrepareError("preflight", "unverified_source", spec.path,
                               "Automatic downloads require a commit-pinned Hugging Face HTTPS URL without credentials")


class _Redirect(urllib.request.HTTPRedirectHandler):
    def __init__(self, policy):
        self.policy = policy

    def redirect_request(self, req, fp, code, msg, headers, newurl):
        parsed = urllib.parse.urlsplit(newurl)
        host = parsed.hostname or ""
        allowed = parsed.scheme == "https" and (host == "huggingface.co" or host.endswith(".huggingface.co")
                                                    or host == "hf.co" or host.endswith(".hf.co"))
        if self.policy.allow_loopback_http and parsed.scheme == "http" and host in {"127.0.0.1", "::1", "localhost"}:
            allowed = True
        if not allowed or parsed.username or parsed.password:
            raise urllib.error.URLError("Download redirect left the approved model hosting domains")
        result = super().redirect_request(req, fp, code, msg, headers, newurl)
        if result is not None:
            # Never forward a private HF access token to a CDN/signed redirect.
            if host != urllib.parse.urlsplit(req.full_url).hostname:
                result.remove_header("Authorization")
            if req.get_method() == "HEAD":
                result.method = "HEAD"
        return result


def _open(url, policy, method="GET", headers=None):
    request_headers = {"User-Agent": "TensorSharp-local-regression/1", "Accept-Encoding": "identity"}
    request_headers.update(headers or {})
    if policy.token and urllib.parse.urlsplit(url).hostname == "huggingface.co":
        request_headers["Authorization"] = "Bearer " + policy.token
    request = urllib.request.Request(url, method=method, headers=request_headers)
    return urllib.request.build_opener(_Redirect(policy)).open(request, timeout=policy.timeout_seconds)


def _network_error(exc, spec, stage):
    if isinstance(exc, urllib.error.HTTPError) and exc.code in (401, 403):
        return PrepareError(stage, "credentials_or_license", spec.path,
                            "Model host refused access; credentials or an already accepted license may be required. No agreement was accepted.")
    if isinstance(exc, (TimeoutError, socket.timeout)) or (isinstance(exc, urllib.error.URLError)
                                                          and isinstance(exc.reason, (TimeoutError, socket.timeout))):
        return PrepareError(stage, "timeout", spec.path, "Model host request timed out")
    status = f" HTTP {exc.code}" if isinstance(exc, urllib.error.HTTPError) else ""
    return PrepareError(stage, "download_failed", spec.path, "Model host request failed" + status)


def _part_size(spec, size):
    partial = Path(str(spec.path) + ".part")
    metadata = Path(str(spec.path) + ".part.json")
    if not partial.exists() and not metadata.exists():
        return 0
    try:
        record = json.loads(metadata.read_text(encoding="utf-8"))
        valid = record == {"sha256": spec.sha256, "size_bytes": size, "version": 1}
    except (OSError, ValueError):
        valid = False
    if not valid or not partial.is_file() or partial.is_symlink() or metadata.is_symlink():
        raise PrepareError("preflight", "partial_unowned", spec.path,
                           "Existing partial file lacks matching suite metadata; retained without modification")
    current = partial.stat().st_size
    if current > size:
        raise PrepareError("preflight", "partial_oversized", spec.path, "Partial model exceeds expected size; retained for inspection")
    return current


def probe_files(files: Iterable[FileSpec], policy: DownloadPolicy, network: bool = False,
                cancel: Callable[[], bool] | None = None) -> list[dict]:
    """Get file sizes without reading model payloads. ``network=True`` allows HEAD.

    Missing unknown sizes remain None offline; they cannot be certified to fit.
    Optional failures are represented explicitly rather than silently disappearing.
    """
    result = []
    for spec in files:
        _cancelled(cancel, spec.path)
        target = Path(spec.path)
        record = {"path": str(target), "option": spec.option, "group": spec.group,
                  "optional": spec.optional, "exists": target.is_file(), "size_bytes": spec.size_bytes,
                  "remaining_bytes": None, "sha256": spec.sha256, "source": None,
                  "status": "missing", "reason": None}
        if target.is_file():
            record.update(size_bytes=target.stat().st_size, remaining_bytes=0, status="local-unverified")
        elif network:
            try:
                _target(spec, policy)
                _validate_source(spec, policy)
                cache_key = (tuple(spec.urls), spec.sha256, spec.size_bytes)
                cached = policy.metadata_cache.get(cache_key)
                if cached:
                    size, source = cached
                    record.update(size_bytes=size, source=source, status="available",
                                  remaining_bytes=size - _part_size(spec, size))
                    result.append(record)
                    continue
                errors = []
                for url in spec.urls:
                    try:
                        with _open(url, policy, "HEAD") as response:
                            length = response.headers.get("Content-Length")
                            size = int(length) if length is not None else spec.size_bytes
                            if size is None or size <= 0:
                                raise PrepareError("probe", "unknown_size", target, "Model host omitted a positive content length")
                            if spec.size_bytes is not None and size != spec.size_bytes:
                                raise PrepareError("probe", "size_mismatch", target, "Model host size differs from the configured identity")
                            record.update(size_bytes=size, source=url, status="available",
                                          remaining_bytes=size - _part_size(spec, size))
                            policy.metadata_cache[cache_key] = (size, url)
                            break
                    except (OSError, ValueError) as exc:
                        errors.append(_network_error(exc, spec, "probe"))
                else:
                    raise errors[-1]
            except PrepareError as exc:
                record.update(status="blocked", reason=exc.as_dict())
        else:
            if spec.size_bytes:
                record["remaining_bytes"] = spec.size_bytes
        result.append(record)
    return result


def _hash(path, cancel=None, progress=None):
    digest = hashlib.sha256()
    read = 0
    with Path(path).open("rb") as stream:
        while True:
            _cancelled(cancel, path)
            block = stream.read(4 * 1024 * 1024)
            if not block:
                break
            digest.update(block)
            read += len(block)
            if progress:
                progress({"stage": "hash", "path": str(path), "bytes": read})
    return digest.hexdigest()


def _verify(spec, path, size, cancel=None, progress=None):
    actual = Path(path).stat().st_size
    if actual <= 0 or (size is not None and actual != size):
        raise PrepareError("verify", "size_mismatch", spec.path, "Model file size is empty or differs from the expected length")
    digest = _hash(path, cancel, progress)
    if spec.sha256 and digest.lower() != spec.sha256.lower():
        raise PrepareError("verify", "checksum_mismatch", spec.path,
                           f"SHA-256 mismatch: expected {spec.sha256.lower()}, actual {digest}; file retained")
    if Path(spec.path).suffix.lower() == ".gguf":
        with Path(path).open("rb") as stream:
            if stream.read(4) != b"GGUF":
                raise PrepareError("verify", "invalid_gguf", spec.path, "Model does not have the GGUF file signature")
    return digest


def _free_bytes(target):
    ancestor = Path(target).parent
    while not ancestor.exists():
        ancestor = ancestor.parent
    return shutil.disk_usage(ancestor).free


def _download(spec, probe, policy, cancel, progress):
    _target(spec, policy)
    target, size = Path(spec.path), probe["size_bytes"]
    target.parent.mkdir(parents=True, exist_ok=True)
    lock = Path(str(target) + ".download.lock")
    try:
        fd = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    except FileExistsError as exc:
        raise PrepareError("download", "concurrent_download", target,
                           "Download lock exists; check the owning process before removing a stale lock") from exc
    os.close(fd)
    try:
        if target.exists():
            digest = _verify(spec, target, size, cancel, progress)
            return FileIdentity(str(target), size, digest, spec.sha256, "local")
        partial, metadata = Path(str(target) + ".part"), Path(str(target) + ".part.json")
        offset = _part_size(spec, size)
        if not partial.exists():
            with partial.open("xb"):
                pass
            metadata.write_text(json.dumps({"sha256": spec.sha256, "size_bytes": size, "version": 1}), encoding="utf-8")
        selected_source = probe["source"] or spec.urls[0]
        if offset < size:
            errors = []
            for url in (selected_source, *(u for u in spec.urls if u != selected_source)):
                _cancelled(cancel, target)
                offset = partial.stat().st_size
                headers = {"Range": f"bytes={offset}-"} if offset else {}
                try:
                    with _open(url, policy, headers=headers) as response:
                        status = response.status
                        mode = "ab"
                        if offset:
                            if status == 200:
                                # A range-ignorant host may restart only this suite-owned partial.
                                offset, mode = 0, "wb"
                            elif status == 206:
                                expected = f"bytes {offset}-{size - 1}/{size}"
                                if response.headers.get("Content-Range") != expected:
                                    raise PrepareError("download", "invalid_range", target, "Resume response does not match the requested byte range")
                            else:
                                raise PrepareError("download", "invalid_range", target, "Resume response has an unexpected status")
                        elif status != 200:
                            raise PrepareError("download", "invalid_response", target, "Initial transfer requires a complete HTTP 200 response")
                        length = response.headers.get("Content-Length")
                        if length is not None and int(length) != size - offset:
                            raise PrepareError("download", "size_mismatch", target, "Transfer length differs from the planned size")
                        remaining = size - offset
                        if policy.downloaded_bytes + remaining > policy.max_download_bytes:
                            raise PrepareError("download", "download_budget", target, "Transfer or restart exceeds the total download budget")
                        if _free_bytes(target) < remaining + policy.disk_headroom_bytes:
                            raise PrepareError("download", "disk_budget", target, "Insufficient free disk space after required headroom")
                        with partial.open(mode) as stream:
                            while True:
                                _cancelled(cancel, target)
                                block = response.read(min(policy.chunk_bytes, remaining + 1))
                                if not block:
                                    break
                                policy.downloaded_bytes += len(block)
                                if len(block) > remaining or policy.downloaded_bytes > policy.max_download_bytes:
                                    raise PrepareError("download", "download_budget", target, "Server exceeded the declared file size or download budget")
                                if _free_bytes(target) < len(block) + policy.disk_headroom_bytes:
                                    raise PrepareError("download", "disk_budget", target, "Free disk space fell below the required headroom")
                                stream.write(block)
                                remaining -= len(block)
                                if progress:
                                    progress({"stage": "download", "path": str(target), "bytes": size - remaining, "total_bytes": size})
                            stream.flush()
                            os.fsync(stream.fileno())
                        if remaining:
                            raise PrepareError("download", "partial_file", target, "Transfer ended early; partial bytes retained for resume")
                        selected_source = url
                        break
                except (urllib.error.URLError, OSError, http.client.HTTPException) as exc:
                    errors.append(_network_error(exc, spec, "download"))
            else:
                raise errors[-1]
        digest = _verify(spec, partial, size, cancel, progress)
        _cancelled(cancel, target)
        if target.exists():
            raise PrepareError("download", "target_appeared", target, "Model target appeared during download; retained without overwrite")
        # Hard-link installation is atomic and fails if another process created target.
        # Both paths share a directory/filesystem; never replace an existing user model.
        try:
            os.link(partial, target)
        except FileExistsError as exc:
            raise PrepareError("download", "target_appeared", target, "Model target appeared before installation; retained without overwrite") from exc
        except OSError as exc:
            raise PrepareError("download", "atomic_install_unavailable", target,
                               "Filesystem refused atomic no-overwrite hard-link installation; verified partial retained") from exc
        partial.unlink()
        metadata.unlink()
        return FileIdentity(str(target), size, digest, spec.sha256, "downloaded", selected_source)
    finally:
        lock.unlink(missing_ok=True)


def prepare_files(files: Iterable[FileSpec], policy: DownloadPolicy,
                  cancel: Callable[[], bool] | None = None,
                  progress: Callable[[dict], None] | None = None) -> list[FileIdentity]:
    """Verify local files or fetch selected missing files sequentially.

    The entire required download set is size/budget checked before transferring.
    A missing optional companion yields ``optional-skipped`` with a reason. Required
    failures raise PrepareError. Callers must not claim optional scenario coverage.
    """
    unique = {}
    for spec in files:
        key = Path(spec.path).resolve()
        if key in unique and (unique[key].sha256 != spec.sha256 or unique[key].size_bytes != spec.size_bytes):
            raise PrepareError("preflight", "conflicting_identity", key, "Selected presets disagree on the identity of a shared file")
        if key not in unique or (unique[key].optional and not spec.optional):
            unique[key] = spec
    # Optional companions must never consume a required file's budget reservation.
    specs = sorted(unique.values(), key=lambda spec: spec.optional)
    probes = probe_files(specs, policy, network=policy.allow_download, cancel=cancel)
    selected, result, missing_total = [], [], 0
    for spec, probe in zip(specs, probes):
        if probe["exists"]:
            selected.append((spec, probe))
            continue
        error = None
        if not policy.allow_download:
            error = PrepareError("preflight", "download_disabled", spec.path, "Required file is missing; downloads are disabled")
        elif probe["status"] == "blocked":
            reason = probe["reason"]
            error = PrepareError(reason["stage"], reason["code"], spec.path, reason["message"])
        elif probe["remaining_bytes"] is None:
            error = PrepareError("preflight", "unknown_size", spec.path, "Cannot reserve an unknown model download size")
        elif missing_total + probe["remaining_bytes"] + policy.downloaded_bytes > policy.max_download_bytes:
            error = PrepareError("preflight", "download_budget", spec.path, "Selected files exceed the total download budget")
        if error:
            if spec.optional:
                result.append(FileIdentity(str(spec.path), 0, None, spec.sha256, "optional-skipped", integrity="unverified", reason=f"{error.code}: {error}"))
                continue
            raise error
        missing_total += probe["remaining_bytes"]
        selected.append((spec, probe))
    if missing_total and _free_bytes(policy.cache_root / "placeholder") < missing_total + policy.disk_headroom_bytes:
        raise PrepareError("preflight", "disk_budget", policy.cache_root, "Selected downloads exceed free disk space after required headroom")
    for spec, probe in selected:
        _cancelled(cancel, spec.path)
        if probe["exists"]:
            digest = _verify(spec, spec.path, spec.size_bytes, cancel, progress)
            result.append(FileIdentity(str(spec.path), Path(spec.path).stat().st_size, digest, spec.sha256,
                                       "local", integrity="verified-sha256" if spec.sha256 else "computed-no-reference"))
        else:
            try:
                result.append(_download(spec, probe, policy, cancel, progress))
            except PrepareError as exc:
                if not spec.optional or exc.code == "cancelled":
                    raise
                result.append(FileIdentity(str(spec.path), 0, None, spec.sha256, "optional-skipped", integrity="unverified", reason=f"{exc.code}: {exc}"))
    return result
