#!/usr/bin/env python3
"""Copy verified files; optionally retire an entirely verified source scope.

Default is COPY ONLY. Never overwrites a conflicting destination, recursively
deletes, follows reparse points, or equates logical bytes with freed disk space.
All roots and the receipt directory are explicit. Receipts are durable recovery
records, not a claim of protection against a hostile process swapping ancestor
directories concurrently: keep the selected trees quiescent while this runs.

Example (review paths first):
  python eng/validation/relocate-verified-files.py --source-root C:/data \
    --source-directory C:/data/old --destination-root D:/archive \
    --destination-directory D:/archive/old --receipt-root D:/receipts \
    --receipt-directory D:/receipts/old-v1 --retire-source --junction --progress
Repeat the IDENTICAL command with --resume after an interruption. Single-file
mode uses --source-file/--destination-file; it cannot create a junction. Repeat
--include-file relative/name for a copy-only subset of a directory.
"""
from __future__ import annotations

import argparse
import ctypes
import errno
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import socket
import stat
import subprocess
import sys
import time
import uuid
from dataclasses import dataclass
from typing import Callable


SCHEMA = "tensorsharp.verified-relocation.v1"
REPARSE_POINT = 0x400
REPO = Path(__file__).resolve().parents[2]


class Refusal(RuntimeError):
    """A safety prerequisite was not satisfied; source data is retained."""


def utc() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def lexical(path: Path | str) -> Path:
    return Path(os.path.abspath(os.fspath(path)))


def inside(path: Path, root: Path, *, strict: bool = True) -> bool:
    try:
        rel = path.relative_to(root)
        return not strict or rel != Path(".")
    except ValueError:
        return False


def reparse(info: os.stat_result) -> bool:
    return stat.S_ISLNK(info.st_mode) or bool(getattr(info, "st_file_attributes", 0) & REPARSE_POINT)


def checked_path(path: Path, *, allow_leaf_alias: bool = False) -> None:
    """Inspect every existing component without traversing a symlink/junction."""
    for node in [*reversed(path.parents), path]:
        try:
            info = node.lstat()
        except FileNotFoundError:
            continue
        if reparse(info) and not (allow_leaf_alias and node == path):
            raise Refusal(f"Symlink/reparse component is forbidden: {node}")


def signature(path: Path) -> dict:
    checked_path(path)
    info = path.lstat()
    if reparse(info) or not stat.S_ISREG(info.st_mode):
        raise Refusal(f"Not an ordinary file: {path}")
    return {"size": info.st_size, "mtime_ns": info.st_mtime_ns,
            "ctime_ns": info.st_ctime_ns, "device": info.st_dev,
            "inode": info.st_ino, "mode": info.st_mode}


def assert_signature(path: Path, expected: dict) -> None:
    if signature(path) != expected:
        raise Refusal(f"File changed since snapshot: {path}")


def scan(directory: Path) -> tuple[dict[str, dict], list[str]]:
    checked_path(directory)
    if not directory.is_dir():
        raise Refusal(f"Not an ordinary directory: {directory}")
    files: dict[str, dict] = {}
    directories = ["."]
    stack = [directory]
    while stack:
        parent = stack.pop()
        checked_path(parent)
        with os.scandir(parent) as entries:
            for entry in entries:
                path = Path(entry.path)
                info = entry.stat(follow_symlinks=False)
                if reparse(info):
                    raise Refusal(f"Symlink/reparse tree entry is forbidden: {path}")
                relative = path.relative_to(directory).as_posix()
                if stat.S_ISDIR(info.st_mode):
                    directories.append(relative)
                    stack.append(path)
                elif stat.S_ISREG(info.st_mode):
                    files[relative] = signature(path)
                else:
                    raise Refusal(f"Special file is forbidden: {path}")
    return dict(sorted(files.items())), sorted(directories)


def fsync_directory(path: Path) -> None:
    if os.name != "nt":
        descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)


def replace_own_receipt(temporary: Path, destination: Path) -> None:
    """Retry only Windows sharing/access interference with our metadata rename.

    A reader may briefly omit FILE_SHARE_DELETE (including a monitoring shell).
    Keep publication atomic, with a five-second deadline; every data-file
    publication/conflict and non-Windows/non-sharing failure stays fail-closed.
    """
    deadline = time.monotonic() + 5.0
    while True:
        try:
            os.replace(temporary, destination)
            return
        except OSError as error:
            if os.name != "nt" or getattr(error, "winerror", None) not in (5, 32, 33):
                raise
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                error._receipt_retry_exhausted = True
                raise
            time.sleep(min(0.1, remaining))


def publish_no_replace(part: Path, destination: Path) -> None:
    """Atomic *exclusive* rename; plain POSIX rename could overwrite a race."""
    checked_path(part)
    checked_path(destination)
    if os.name == "nt":
        os.rename(part, destination)  # Windows refuses an existing destination.
    else:
        libc = ctypes.CDLL(None, use_errno=True)
        if sys.platform.startswith("linux") and hasattr(libc, "renameat2"):
            fn = libc.renameat2
            fn.argtypes = [ctypes.c_int, ctypes.c_char_p, ctypes.c_int, ctypes.c_char_p, ctypes.c_uint]
            result = fn(-100, os.fsencode(part), -100, os.fsencode(destination), 1)  # RENAME_NOREPLACE
        elif sys.platform == "darwin" and hasattr(libc, "renamex_np"):
            fn = libc.renamex_np
            fn.argtypes = [ctypes.c_char_p, ctypes.c_char_p, ctypes.c_uint]
            result = fn(os.fsencode(part), os.fsencode(destination), 4)  # RENAME_EXCL
        else:
            raise Refusal("This platform has no supported atomic no-replace rename")
        if result:
            error = ctypes.get_errno()
            raise OSError(error, os.strerror(error), str(destination))
    fsync_directory(destination.parent)


@dataclass
class Config:
    source_root: Path
    source: Path
    destination_root: Path
    destination: Path
    receipt_root: Path
    receipt_directory: Path
    kind: str = "directory"
    include_files: tuple[str, ...] = ()
    retire_source: bool = False
    junction: bool = False
    minimum_free_bytes: int = 512 * 1024 * 1024
    chunk_bytes: int = 8 * 1024 * 1024
    progress_bytes: int = 256 * 1024 * 1024
    progress: bool = False

    def normalize(self) -> None:
        for field in ("source_root", "source", "destination_root", "destination",
                      "receipt_root", "receipt_directory"):
            setattr(self, field, lexical(getattr(self, field)))
        if self.kind not in ("directory", "file"):
            raise Refusal("Source kind must be directory or file")
        if self.junction and (not self.retire_source or self.kind != "directory"):
            raise Refusal("--junction requires an entire directory and --retire-source")
        if self.include_files and (self.retire_source or self.kind != "directory"):
            raise Refusal("--include-file is copy-only directory mode; partial scopes cannot retire")
        normalized = []
        for name in self.include_files:
            relative = Path(name)
            if relative.is_absolute() or relative.anchor or ".." in relative.parts or str(relative) in ("", "."):
                raise Refusal(f"Include must be a relative exact file, without traversal: {name}")
            value = relative.as_posix()
            if value in normalized:
                raise Refusal(f"Duplicate include file: {name}")
            normalized.append(value)
        self.include_files = tuple(sorted(normalized))
        if self.minimum_free_bytes < 0 or self.chunk_bytes <= 0 or self.progress_bytes <= 0:
            raise Refusal("Free-space floor must be nonnegative; chunk/progress sizes must be positive")
        for root in (self.source_root, self.destination_root, self.receipt_root):
            checked_path(root)
            if not root.is_dir() or root.resolve() != root:
                raise Refusal(f"Allowed root must be an existing canonical ordinary directory: {root}")
        for path, root in ((self.source, self.source_root),
                           (self.destination, self.destination_root),
                           (self.receipt_directory, self.receipt_root)):
            if not inside(path, root):
                raise Refusal(f"Path must be a strict descendant of its named root: {path}; root={root}")
            checked_path(path, allow_leaf_alias=(path == self.source and self.junction))
            if path != self.source or not path.is_symlink():
                # A completed Windows junction is also allowed only by receipt validation below.
                if not (path == self.source and self.junction and path.exists() and reparse(path.lstat())):
                    if not inside(path.resolve(), root):
                        raise Refusal(f"Resolved path escapes its named root: {path}")
        scopes = (self.source, self.destination, self.receipt_directory)
        for index, one in enumerate(scopes):
            for other in scopes[index + 1:]:
                if inside(one, other, strict=False) or inside(other, one, strict=False):
                    raise Refusal(f"Source, destination and receipt scopes must not overlap: {one}, {other}")
        if inside(self.receipt_directory, REPO, strict=False):
            if not any(inside(self.receipt_directory, REPO / sub) for sub in ("artifacts", "docs/validation")):
                raise Refusal("Receipts inside this repository must be under ignored artifacts/ or docs/validation/")

    def identity(self) -> dict:
        return {"source_root": str(self.source_root), "source": str(self.source),
                "destination_root": str(self.destination_root), "destination": str(self.destination),
                "receipt_root": str(self.receipt_root), "receipt_directory": str(self.receipt_directory),
                "kind": self.kind, "include_files": list(self.include_files),
                "retire_source": self.retire_source, "junction": self.junction}


class Relocator:
    def __init__(self, config: Config, *, resume: bool = False,
                 reporter: Callable[[dict], None] | None = None):
        self.config = config
        config.normalize()
        self.resume = resume
        self.reporter = reporter
        self.receipt: dict = {}
        self.pending_progress = 0
        self.lock_owned = False
        self.receipt_path = config.receipt_directory / "receipt.json"
        self.lock_path = config.receipt_directory / "operation.lock"

    def save(self) -> None:
        self.receipt["updated_utc"] = utc()
        checked_path(self.config.receipt_directory)
        checked_path(self.receipt_path)
        temporary = self.receipt_path.with_suffix(".json.tmp")
        checked_path(temporary)
        with open(temporary, "w", encoding="utf-8", newline="\n") as output:
            json.dump(self.receipt, output, ensure_ascii=False, indent=2, sort_keys=True)
            output.write("\n")
            output.flush()
            os.fsync(output.fileno())
        replace_own_receipt(temporary, self.receipt_path)  # Only our receipt, never user data.
        fsync_directory(self.config.receipt_directory)

    def emit(self, phase: str, path: Path | None = None, count: int = 0, *, force: bool = False) -> None:
        self.pending_progress += count
        self.receipt["bytes_read_or_copied"] = self.receipt.get("bytes_read_or_copied", 0) + count
        self.receipt["progress"] = {"phase": phase, "path": str(path) if path else None,
                                    "utc": utc(), "bytes_read_or_copied": self.receipt["bytes_read_or_copied"]}
        if force or self.pending_progress >= self.config.progress_bytes:
            self.pending_progress = 0
            self.save()
            if self.reporter:
                self.reporter(self.receipt["progress"])
            elif self.config.progress:
                print(json.dumps(self.receipt["progress"], ensure_ascii=False), flush=True)

    def disk_sample(self, phase: str) -> None:
        samples = {}
        for name, path in (("source", self.config.source_root),
                           ("destination", self.config.destination_root),
                           ("receipt", self.config.receipt_root)):
            disk = shutil.disk_usage(path)
            samples[name] = {"path": str(path), "total": disk.total, "used": disk.used, "free": disk.free}
        self.receipt.setdefault("disk_samples", []).append({"phase": phase, "utc": utc(), "volumes": samples})

    def guard_space(self, remaining: int) -> None:
        free = shutil.disk_usage(self.config.destination_root).free
        if free < remaining + self.config.minimum_free_bytes:
            raise Refusal(f"Insufficient destination free space: free={free}, remaining={remaining}, "
                          f"required_margin={self.config.minimum_free_bytes}")

    def source_file(self, relative: str) -> Path:
        self.validate_relative(relative)
        return self.config.source if self.config.kind == "file" else self.config.source / relative

    def destination_file(self, relative: str) -> Path:
        self.validate_relative(relative)
        return self.config.destination if self.config.kind == "file" else self.config.destination / relative

    @staticmethod
    def validate_relative(relative: str, *, directory: bool = False) -> None:
        if directory and relative == ".":
            return
        path = Path(relative)
        if not isinstance(relative, str) or path.is_absolute() or path.anchor or ".." in path.parts \
                or relative in ("", ".") or path.as_posix() != relative:
            raise Refusal(f"Invalid relative path in receipt: {relative!r}")

    def validate_receipt(self) -> None:
        identity = self.receipt.get("operation_id", "")
        if not re.fullmatch(r"[0-9a-f]{32}", identity):
            raise Refusal("Invalid receipt operation identity")
        for relative, item in self.receipt["files"].items():
            self.validate_relative(relative)
            if self.config.kind == "file" and relative != self.config.source.name:
                raise Refusal("File-mode receipt has an unexpected source key")
            if item.get("sha256") and not re.fullmatch(r"[0-9a-f]{64}", item["sha256"]):
                raise Refusal("Invalid full SHA256 in receipt")
        if self.config.kind == "file" and len(self.receipt["files"]) != 1:
            raise Refusal("File-mode receipt must contain exactly one file")
        for relative in self.receipt["directories"]:
            self.validate_relative(relative, directory=True)

    def current_inventory(self) -> tuple[dict[str, dict], list[str]]:
        if self.config.kind == "file":
            return {self.config.source.name: signature(self.config.source)}, []
        return scan(self.config.source)

    def hash_file(self, path: Path, expected: dict, phase: str) -> str:
        assert_signature(path, expected)
        digest = hashlib.sha256()
        descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_BINARY", 0) | getattr(os, "O_NOFOLLOW", 0))
        try:
            info = os.fstat(descriptor)
            if (info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns) != (
                    expected["device"], expected["inode"], expected["size"], expected["mtime_ns"]):
                raise Refusal(f"File identity changed while opening: {path}")
            with os.fdopen(descriptor, "rb", closefd=False) as source:
                while True:
                    chunk = source.read(self.config.chunk_bytes)
                    if not chunk:
                        break
                    digest.update(chunk)
                    self.emit(phase, path, len(chunk))
        finally:
            os.close(descriptor)
        assert_signature(path, expected)
        return digest.hexdigest()

    def acquire_lock(self) -> None:
        if self.lock_path.exists():
            checked_path(self.lock_path)
            prior = json.loads(self.lock_path.read_text(encoding="utf-8"))
            if not self.resume or prior.get("host") != socket.gethostname() or process_alive(prior["pid"]):
                raise Refusal(f"Another process or unrecognized lock owns receipt: {self.lock_path}")
            self.lock_path.unlink()  # Recognized stale metadata lock only.
        descriptor = os.open(self.lock_path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        with os.fdopen(descriptor, "w", encoding="utf-8") as output:
            json.dump({"pid": os.getpid(), "host": socket.gethostname(), "utc": utc()}, output)
            output.flush()
            os.fsync(output.fileno())
        self.lock_owned = True

    def initialize(self) -> None:
        config = self.config
        if self.resume:
            if not config.receipt_directory.is_dir():
                raise Refusal("--resume requires an existing receipt directory")
            checked_path(self.receipt_path)
            self.receipt = json.loads(self.receipt_path.read_text(encoding="utf-8"))
            if self.receipt.get("schema") != SCHEMA or self.receipt.get("config") != config.identity():
                raise Refusal("Receipt schema/config differs; resume requires identical source/destination/actions")
            self.validate_receipt()
        else:
            config.receipt_directory.mkdir(parents=True, exist_ok=False)
        self.acquire_lock()
        if not self.resume:
            checked_path(config.source)
            files, directories = self.current_inventory()
            if config.include_files:
                missing = set(config.include_files) - files.keys()
                if missing:
                    raise Refusal(f"Included exact source file is missing: {sorted(missing)}")
                files = {key: files[key] for key in config.include_files}
                directories = sorted({".", *(str(Path(key).parent).replace("\\", "/") for key in files)})
            self.receipt = {"schema": SCHEMA, "operation_id": uuid.uuid4().hex,
                            "created_utc": utc(), "config": config.identity(), "status": "copying",
                            "files": {key: {"source_snapshot": value, "state": "pending"} for key, value in files.items()},
                            "directories": directories, "retired_directories": [], "deletion_intents": [],
                            "errors": [], "logical_source_bytes": sum(value["size"] for value in files.values()),
                            "logical_retired_bytes": 0,
                            "physical_release_note": "Logical byte totals are not measured physical disk release; disk_samples are observations."}
        self.disk_sample("begin_or_resume")
        self.save()

    def validate_destination_tree(self) -> None:
        destination = self.config.destination
        checked_path(destination)
        if self.config.kind == "directory":
            if destination.exists():
                scan(destination)  # Reject reparse entries even outside the selected subset.
            else:
                destination.mkdir(parents=True)
            for relative in self.receipt["directories"]:
                path = destination / relative
                checked_path(path)
                path.mkdir(parents=True, exist_ok=True)
        else:
            destination.parent.mkdir(parents=True, exist_ok=True)
            if destination.exists():
                signature(destination)

    def source_is_expected_alias(self) -> bool:
        source = self.config.source
        return bool(self.config.junction and source.exists() and reparse(source.lstat())
                    and source.resolve() == self.config.destination.resolve()
                    and self.receipt.get("retirement_started")
                    and all(item["state"] == "retired" for item in self.receipt["files"].values())
                    and set(self.receipt["directories"]) == set(self.receipt["retired_directories"])
                    and self.receipt.get("status") in ("alias_pending", "complete", "failed"))

    def check_inventory(self, *, retiring: bool = False) -> None:
        expected = self.receipt["files"]
        if self.source_is_expected_alias():
            return
        checked_path(self.config.source)
        if not self.config.source.exists():
            if retiring and all(item["state"] in ("retire_pending", "retired") for item in expected.values()) \
                    and self.receipt.get("retirement_started"):
                return
            raise Refusal("Source disappeared outside recorded retirement")
        files, directories = self.current_inventory()
        if self.config.include_files:
            files = {key: files[key] for key in expected if key in files}
        if set(files) - expected.keys():
            raise Refusal("Source scope gained files after snapshot")
        for relative, item in expected.items():
            if relative not in files:
                if retiring and item["state"] in ("retire_pending", "retired") and self.receipt.get("retirement_started"):
                    continue
                raise Refusal(f"Source file disappeared: {relative}")
            if item["state"] == "retired":
                raise Refusal(f"Retired source path was recreated: {relative}")
            if files[relative] != item["source_snapshot"]:
                raise Refusal(f"Source changed since initial snapshot: {relative}")
        if self.config.kind == "directory" and not self.config.include_files:
            original_dirs = set(self.receipt["directories"])
            if set(directories) - original_dirs:
                raise Refusal("Source scope gained directories after snapshot")
            if not retiring and set(directories) != original_dirs:
                raise Refusal("Source directory inventory changed after snapshot")

    def copy_one(self, relative: str, item: dict) -> None:
        source = self.source_file(relative)
        target = self.destination_file(relative)
        assert_signature(source, item["source_snapshot"])
        checked_path(target)
        if target.exists():
            target_stat = signature(target)
            if target_stat["size"] != item["source_snapshot"]["size"]:
                raise Refusal(f"Conflicting existing destination length (never overwritten): {target}")
            source_hash = self.hash_file(source, item["source_snapshot"], "hash_existing_source")
            if item.get("sha256") and source_hash != item["sha256"]:
                raise Refusal(f"Source content differs from original verified content: {source}")
            if self.hash_file(target, target_stat, "hash_existing_destination") != source_hash:
                raise Refusal(f"Conflicting existing destination content (never overwritten): {target}")
            item.update(state="verified", sha256=source_hash, destination_snapshot=target_stat,
                        disposition=item.get("disposition", "existing_full_hash_match"))
            self.emit("verified_existing", target, force=True)
            return
        if item.get("sha256") or item["state"] != "pending":
            # A crash before publication may leave an owned part; a missing verified destination is not repairable silently.
            if item["state"] not in ("copying", "part_verified", "publishing"):
                raise Refusal(f"Previously verified destination disappeared: {target}")
        relative_hash = hashlib.sha256(relative.encode("utf-8")).hexdigest()[:24]
        part = target.parent / f".tensorsharp-relocate-{self.receipt['operation_id']}-{relative_hash}.part"
        checked_path(part)
        if part.exists():
            if item.get("part") != str(part) or item["state"] not in ("copying", "part_verified", "publishing"):
                raise Refusal(f"Unowned partial destination exists: {part}")
            signature(part)
            part.unlink()  # Restart only our recorded incomplete regular file.
        self.guard_space(item["source_snapshot"]["size"])
        item.update(state="copying", part=str(part))
        self.save()
        digest = hashlib.sha256()
        descriptor = os.open(source, os.O_RDONLY | getattr(os, "O_BINARY", 0) | getattr(os, "O_NOFOLLOW", 0))
        try:
            opened = os.fstat(descriptor)
            expected = item["source_snapshot"]
            if (opened.st_dev, opened.st_ino, opened.st_size, opened.st_mtime_ns) != (
                    expected["device"], expected["inode"], expected["size"], expected["mtime_ns"]):
                raise Refusal(f"Source identity changed while opening: {source}")
            with os.fdopen(descriptor, "rb", closefd=False) as input_file, open(part, "xb") as output:
                copied = 0
                while True:
                    chunk = input_file.read(self.config.chunk_bytes)
                    if not chunk:
                        break
                    output.write(chunk)
                    digest.update(chunk)
                    copied += len(chunk)
                    self.guard_space(max(0, expected["size"] - copied))
                    self.emit("copy", source, len(chunk))
                output.flush()
                os.fsync(output.fileno())
        finally:
            os.close(descriptor)
        assert_signature(source, item["source_snapshot"])
        source_hash = digest.hexdigest()
        if item.get("sha256") and source_hash != item["sha256"]:
            raise Refusal(f"Resumed source content differs from original verified bytes: {source}")
        if copied != item["source_snapshot"]["size"]:
            raise Refusal(f"Copied source length differs from snapshot: {source}")
        os.utime(part, ns=(item["source_snapshot"]["mtime_ns"], item["source_snapshot"]["mtime_ns"]))
        if os.name != "nt":
            os.chmod(part, stat.S_IMODE(item["source_snapshot"]["mode"]))
        part_stat = signature(part)
        if self.hash_file(part, part_stat, "verify_full_partial_destination") != source_hash:
            raise Refusal(f"Full destination verification failed: {part}")
        item.update(state="part_verified", sha256=source_hash, part_snapshot=part_stat)
        self.save()
        assert_signature(source, item["source_snapshot"])
        assert_signature(part, part_stat)
        item["state"] = "publishing"
        self.save()
        publish_no_replace(part, target)
        item.update(state="verified", destination_snapshot=signature(target), disposition="copied_and_full_hash_verified")
        self.disk_sample("file_verified")
        self.emit("verified_copy", target, force=True)

    def retirement_preflight(self) -> None:
        self.check_inventory(retiring=bool(self.receipt.get("retirement_started")))
        for relative, item in self.receipt["files"].items():
            target = self.destination_file(relative)
            assert_signature(target, item["destination_snapshot"])
            if self.hash_file(target, item["destination_snapshot"], "retirement_reverify_destination") != item["sha256"]:
                raise Refusal(f"Verified destination changed: {target}")
            source = self.source_file(relative)
            if source.exists() and not self.source_is_expected_alias():
                if self.hash_file(source, item["source_snapshot"], "retirement_reverify_source") != item["sha256"]:
                    raise Refusal(f"Source changed before retirement: {source}")
            elif item["state"] not in ("retire_pending", "retired"):
                raise Refusal(f"Unrecorded source disappearance: {source}")
        self.check_inventory(retiring=bool(self.receipt.get("retirement_started")))

    def deletion_intent(self, path: Path, kind: str) -> None:
        checked_path(path)
        resolved = path.resolve()
        if not inside(resolved, self.config.source_root) or not inside(resolved, self.config.source, strict=False):
            raise Refusal(f"Deletion boundary is not inside recorded source and root: {resolved}")
        self.receipt["deletion_intents"].append({"absolute_resolved_path": str(resolved), "kind": kind,
                                                "source_root": str(self.config.source_root), "utc": utc()})
        self.save()  # Durable intent BEFORE unlink/rmdir; never a computed recursive removal.

    def unlink_source_file(self, path: Path) -> None:
        path.unlink()
        fsync_directory(path.parent)

    def retire(self) -> None:
        self.retirement_preflight()  # Entire scope, not per-file copy-then-delete.
        self.receipt.update(retirement_started=True, status="retiring")
        self.save()
        for relative, item in self.receipt["files"].items():
            source = self.source_file(relative)
            target = self.destination_file(relative)
            assert_signature(target, item["destination_snapshot"])
            if source.exists():
                assert_signature(source, item["source_snapshot"])
                item["state"] = "retire_pending"
                self.deletion_intent(source, "file")
                assert_signature(source, item["source_snapshot"])
                assert_signature(target, item["destination_snapshot"])
                self.unlink_source_file(source)
            elif item["state"] not in ("retire_pending", "retired"):
                raise Refusal(f"Unrecorded missing source: {source}")
            item["state"] = "retired"
            self.receipt["logical_retired_bytes"] = sum(value["source_snapshot"]["size"]
                for value in self.receipt["files"].values() if value["state"] == "retired")
            self.emit("retired_verified_file", source, force=True)
        if self.config.kind == "directory":
            for relative in sorted(self.receipt["directories"], key=lambda value: (len(Path(value).parts), value), reverse=True):
                path = self.config.source / relative
                checked_path(path)
                if path.exists():
                    self.deletion_intent(path, "empty_directory")
                    path.rmdir()  # Fails on anything unexpected; deliberately no recursive delete.
                    fsync_directory(path.parent)
                if relative not in self.receipt["retired_directories"]:
                    self.receipt["retired_directories"].append(relative)
                    self.save()
            if self.config.junction:
                self.receipt["status"] = "alias_pending"
                self.save()
                self.create_alias()

    def create_alias(self) -> None:
        source, destination = self.config.source, self.config.destination
        checked_path(source.parent)
        checked_path(destination)
        if os.path.lexists(source):
            raise Refusal(f"Alias path already exists: {source}")
        if os.name == "nt":
            powershell = shutil.which("pwsh") or shutil.which("powershell")
            if not powershell:
                raise Refusal("PowerShell is required for a Windows junction; verified destination is retained")
            helper = self.config.receipt_directory / "create-junction.ps1"
            checked_path(helper)
            helper.write_text("param([Parameter(Mandatory=$true)][string]$Link,"
                              "[Parameter(Mandatory=$true)][string]$Target)\n"
                              "$ErrorActionPreference='Stop'\n"
                              "New-Item -ItemType Junction -Path $Link -Target $Target | Out-Null\n", encoding="utf-8")
            result = subprocess.run([powershell, "-NoProfile", "-NonInteractive", "-File", str(helper),
                                     "-Link", str(source), "-Target", str(destination)],
                                    capture_output=True, text=True, shell=False)
            self.receipt["alias_command"] = {"argv": result.args, "exit_code": result.returncode,
                                              "stdout": result.stdout, "stderr": result.stderr}
            self.save()
            if result.returncode:
                raise Refusal(f"Junction creation failed: {result.stderr.strip()}")
        else:
            os.symlink(destination, source, target_is_directory=True)
            fsync_directory(source.parent)
        if not self.source_is_expected_alias():
            raise Refusal("Created alias did not resolve to the verified destination")
        self.receipt["alias"] = {"path": str(source), "resolved_target": str(source.resolve()),
                                  "kind": "junction" if os.name == "nt" else "symlink"}
        self.save()

    def execute(self) -> dict:
        try:
            self.initialize()
            if self.source_is_expected_alias():
                # Recovery after alias creation/receipt fsync: never traverse/delete alias contents.
                for relative, item in self.receipt["files"].items():
                    target = self.destination_file(relative)
                    if self.hash_file(target, item["destination_snapshot"], "resume_verify_alias_destination") != item["sha256"]:
                        raise Refusal(f"Aliased destination content changed: {target}")
                self.receipt["status"] = "complete"
            elif self.receipt.get("retirement_started"):
                self.retire()
                self.receipt["status"] = "complete"
            else:
                self.check_inventory()
                self.validate_destination_tree()
                for relative, item in self.receipt["files"].items():
                    self.copy_one(relative, item)
                self.check_inventory()
                self.receipt["status"] = "verified"
                self.save()
                if self.config.retire_source:
                    self.retire()
                self.receipt["status"] = "complete" if self.config.retire_source else "copy_complete"
            self.disk_sample("finished")
            self.emit("finished", force=True)
            return self.receipt
        except BaseException as error:
            if self.receipt and self.lock_owned:
                self.receipt.setdefault("errors", []).append({"utc": utc(), "type": type(error).__name__, "message": str(error)})
                self.receipt["status"] = "failed"
                try:
                    # Do not begin a second five-second retry budget when the
                    # atomic receipt rename itself just exhausted its deadline.
                    if not getattr(error, "_receipt_retry_exhausted", False):
                        self.disk_sample("failed")
                        self.save()
                except Exception:
                    pass  # Never hide original failure or delete source to make cleanup succeed.
            raise
        finally:
            if self.lock_owned:
                checked_path(self.lock_path)
                self.lock_path.unlink(missing_ok=True)
                self.lock_owned = False


def process_alive(pid: int) -> bool:
    if not isinstance(pid, int) or pid <= 0:
        return True
    if os.name == "nt":
        kernel = ctypes.WinDLL("kernel32", use_last_error=True)
        kernel.OpenProcess.argtypes = [ctypes.c_uint, ctypes.c_int, ctypes.c_uint]
        kernel.OpenProcess.restype = ctypes.c_void_p
        kernel.WaitForSingleObject.argtypes = [ctypes.c_void_p, ctypes.c_uint]
        kernel.CloseHandle.argtypes = [ctypes.c_void_p]
        handle = kernel.OpenProcess(0x100000, False, pid)  # SYNCHRONIZE, never terminate/query unrelated processes.
        if not handle:
            return ctypes.get_last_error() != 87  # ERROR_INVALID_PARAMETER means no such process.
        try:
            return kernel.WaitForSingleObject(handle, 0) != 0
        finally:
            kernel.CloseHandle(handle)
    try:
        os.kill(pid, 0)
        return True
    except ProcessLookupError:
        return False
    except PermissionError:
        return True


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--source-root", type=Path, required=True)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--source-directory", type=Path)
    group.add_argument("--source-file", type=Path)
    parser.add_argument("--destination-root", type=Path, required=True)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--destination-directory", type=Path)
    group.add_argument("--destination-file", type=Path)
    parser.add_argument("--receipt-root", type=Path, required=True)
    parser.add_argument("--receipt-directory", type=Path, required=True)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--include-file", action="append", default=[])
    parser.add_argument("--retire-source", action="store_true")
    parser.add_argument("--junction", action="store_true")
    parser.add_argument("--minimum-free-bytes", type=int, default=512 * 1024 * 1024)
    parser.add_argument("--chunk-bytes", type=int, default=8 * 1024 * 1024)
    parser.add_argument("--progress-bytes", type=int, default=256 * 1024 * 1024)
    parser.add_argument("--progress", action="store_true")
    args = parser.parse_args(argv)
    if bool(args.source_directory) != bool(args.destination_directory):
        parser.error("Source and destination must both be directory mode or both file mode")
    config = Config(args.source_root, args.source_directory or args.source_file,
                    args.destination_root, args.destination_directory or args.destination_file,
                    args.receipt_root, args.receipt_directory,
                    kind="directory" if args.source_directory else "file",
                    include_files=tuple(args.include_file), retire_source=args.retire_source,
                    junction=args.junction, minimum_free_bytes=args.minimum_free_bytes,
                    chunk_bytes=args.chunk_bytes, progress_bytes=args.progress_bytes, progress=args.progress)
    try:
        receipt = Relocator(config, resume=args.resume).execute()
        print(json.dumps({"status": receipt["status"], "receipt": str(config.receipt_directory / "receipt.json"),
                          "files": len(receipt["files"]), "logical_source_bytes": receipt["logical_source_bytes"],
                          "logical_retired_bytes": receipt["logical_retired_bytes"],
                          "physical_release_note": receipt["physical_release_note"]}, ensure_ascii=False), flush=True)
        return 0
    except Exception as error:
        print(json.dumps({"status": "refused_or_failed", "type": type(error).__name__, "error": str(error),
                          "receipt": str(config.receipt_directory / "receipt.json")}, ensure_ascii=False), file=sys.stderr, flush=True)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
