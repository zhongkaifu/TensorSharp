#!/usr/bin/env python3
"""Back up an explicit plan without restoring or changing its source files.

Member verification proves captured bytes/link identities, not a successful
fresh-filesystem restore. Recheck restored links, ancestor directory aliases,
external dependencies and interpreter/ELF/driver ABI before using a snapshot.
Version2 records every followed link hop; old v1 archives are not reclassified.
"""
from __future__ import annotations

import argparse
from collections import deque
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import stat
import tarfile
import time

TOOL_VERSION = "qwen38-recovery-archive/2"


def sha_stream(stream):
    value = hashlib.sha256()
    while data := stream.read(8 * 1024 * 1024):
        value.update(data)
    return value.hexdigest()


def sha(path):
    with Path(path).open("rb") as stream:
        return sha_stream(stream)


def under(path, roots):
    return any(path == root or root in path.parents for root in roots)


def absolute(name):
    path = Path(name)
    if not path.is_absolute() or ".." in path.parts:
        raise ValueError(f"Plan paths must be absolute without parent traversal: {name}")
    return path


def resolve(path):
    try:
        return path.resolve(strict=True)
    except RuntimeError as error:
        raise ValueError(f"Symlink cycle while resolving {path}") from error


def link_target_chain(link):
    """Resolve target components in POSIX order, retaining intermediate links.

    Do not normpath an entire target first: alias/../file must process alias
    before '..'. The existing source-parent filesystem is an explicit restore
    prerequisite; this is not an archive of all system ancestor aliases.
    """
    current = resolve(link.parent)
    pending = deque(Path(os.readlink(link)).parts)
    if pending and pending[0] == "/":
        current = Path("/")
        pending.popleft()
    hops, expansions = [], 0
    while pending:
        part = pending.popleft()
        if part in ("", "."):
            continue
        if part == "..":
            current = current.parent
            continue
        candidate = current / part
        info = candidate.lstat()
        if stat.S_ISLNK(info.st_mode):
            expansions += 1
            if expansions > 40:
                raise ValueError(f"Symlink cycle or excessive chain while resolving {link}")
            target = os.readlink(candidate)
            hops.append((candidate, target))
            components = deque(Path(target).parts)
            if components and components[0] == "/":
                current = Path("/")
                components.popleft()
            pending.extendleft(reversed(components))
        else:
            current = candidate
    return current, hops


def inventory(plan):
    roots = [absolute(p) for p in plan["paths"]]
    resolved_roots = [resolve(p) for p in roots]
    def prefixes(names):
        paths = [absolute(p) for p in names]
        # A declared alias such as /bin is itself an allowed link node, while
        # /usr/bin is the resolved prefix containing its actual targets.
        return paths + [p.resolve() for p in paths]
    owned = prefixes(plan.get("include_symlink_targets_under", []))
    external_allowed = prefixes(plan.get("external_dependency_prefixes", ["/usr", "/lib", "/bin", "/sbin"]))
    forbidden = prefixes(plan.get("forbidden_prefixes", []))
    rows, external = {}, {}

    def require_not_forbidden(path):
        if under(path, forbidden) or under(resolve(path), forbidden):
            raise ValueError(f"Forbidden source (models must not be copied): {path}")

    def include_dependency(path, required_by):
        require_not_forbidden(path)
        if under(path, roots + resolved_roots):
            add(path)
        elif under(path, owned):
            if not path.is_file():
                raise ValueError(f"External owned directory link needs an explicit plan root: {required_by}")
            add(path)
        elif under(path, external_allowed) and (path.is_symlink() or path.is_file()):
            if path.is_symlink():
                external[str(path)] = dict(path=str(path), kind="symlink", target=os.readlink(path),
                                           resolved_target=str(resolve(path)),
                                           required_by_link=str(required_by), copied=False)
            else:
                external[str(path)] = dict(path=str(path), kind="file", bytes=path.stat().st_size,
                                           sha256=sha(path), required_by_link=str(required_by), copied=False)
        elif path.is_symlink() and path.is_dir() and any(path in prefix.parents for prefix in roots + owned + external_allowed):
            # For example macOS /var -> private/var can be an ancestor of the
            # explicitly approved /var/.../build prefix. Record that prerequisite,
            # never turn approval of one leaf into copying a system directory.
            external[str(path)] = dict(path=str(path), kind="symlink", target=os.readlink(path),
                                       resolved_target=str(resolve(path)), required_by_link=str(required_by),
                                       dependency_scope="ancestor_of_explicitly_approved_prefix", copied=False)
        else:
            raise ValueError(f"Uncovered runtime symlink: {required_by} -> {path}")

    def add(path):
        require_not_forbidden(path)
        key = str(path).lstrip("/")
        if key in rows:
            return
        s = path.lstat()
        row = dict(source_path=str(path), archive_path=key, mode=stat.S_IMODE(s.st_mode),
                   uid=s.st_uid, gid=s.st_gid, mtime_ns=s.st_mtime_ns)
        rows[key] = row
        if stat.S_ISLNK(s.st_mode):
            path.stat()  # OS traversal must work too (trailing slash / ELOOP).
            target, hops = link_target_chain(path)
            row.update(kind="symlink", target=os.readlink(path), resolved_target=str(target))
            for hop, _ in hops:
                include_dependency(hop, path)
            include_dependency(target, path)
        elif stat.S_ISREG(s.st_mode):
            row.update(kind="file", bytes=s.st_size, sha256=sha(path))
        elif stat.S_ISDIR(s.st_mode):
            row.update(kind="directory")
            for child in sorted(path.iterdir()):
                add(child)
        else:
            raise ValueError(f"Special files cannot be recovery archive members: {path}")

    for root in roots:
        add(root)
    return [rows[k] for k in sorted(rows)], [external[k] for k in sorted(external)]


def verify_archive(path, rows):
    expected = {row["archive_path"]: row for row in rows}
    seen, verified = set(), 0
    with tarfile.open(path, "r:") as archive:
        for member in archive:
            name = member.name.rstrip("/")
            if name not in expected or name in seen:
                raise ValueError(f"Unexpected or duplicate archive member: {name}")
            seen.add(name)
            row = expected[name]
            if member.mode != row["mode"] or member.uid != row["uid"] or member.gid != row["gid"]:
                raise ValueError(f"Archive permissions/owner differ: {name}")
            if row["kind"] == "file":
                if not (member.isfile() or member.islnk()):
                    raise ValueError(f"Archive file type differs: {name}")
                if member.islnk() and member.linkname not in expected:
                    raise ValueError(f"Uncovered archive hardlink: {name}")
                with archive.extractfile(member) as stream:
                    if sha_stream(stream) != row["sha256"]:
                        raise ValueError(f"Archive content differs: {name}")
                if not member.islnk() and member.size != row["bytes"]:
                    raise ValueError(f"Archive size differs: {name}")
                verified += 1
            elif row["kind"] == "symlink":
                if not member.issym() or member.linkname != row["target"]:
                    raise ValueError(f"Archive symlink differs: {name}")
            elif not member.isdir():
                raise ValueError(f"Archive directory differs: {name}")
    if seen != set(expected):
        raise ValueError("Archive has missing plan members")
    return verified


def save(path, value):
    temporary = path.with_suffix(path.suffix + ".incoming")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    temporary.replace(path)


def archive_plan(plan, destination):
    tool_sha256 = sha(Path(__file__))
    destination = absolute(destination)
    roots = [absolute(p) for p in plan["paths"]]
    if not roots or any(root == Path("/") for root in roots):
        raise ValueError("An explicit bounded non-root source plan is required")
    if any(under(destination.resolve(), [root.resolve()]) for root in roots):
        raise ValueError("Archive destination cannot be inside any source")
    if destination.exists():
        raise FileExistsError(f"Refusing to overwrite evidence: {destination}")
    destination.mkdir(parents=True)
    started = datetime.now(timezone.utc).isoformat()
    t0 = time.monotonic()
    save(destination / "plan.json", plan)
    rows, external = inventory(plan)
    manifest = destination / "files.json"
    save(manifest, dict(schema="qwen38-recovery-files/2", tool_version=TOOL_VERSION, members=rows,
                        required_external_dependencies=external))
    archive = destination / "payload.tar"
    with tarfile.open(archive, "w", format=tarfile.PAX_FORMAT, dereference=False) as output:
        for row in rows:
            output.add(row["source_path"], arcname=row["archive_path"], recursive=False)
    verified = verify_archive(archive, rows)
    archive_sha256 = sha(archive)
    if inventory(plan) != (rows, external):
        raise ValueError("Source or required external dependencies changed during backup")
    if sha(Path(__file__)) != tool_sha256:
        raise ValueError("Archive tool changed during backup")
    result = dict(schema="qwen38-recovery-verified/2", tool_version=TOOL_VERSION, status="VERIFIED_SOURCE_UNCHANGED",
                  started_utc=started, finished_utc=datetime.now(timezone.utc).isoformat(),
                  elapsed_seconds=time.monotonic() - t0,
                  archive_path=str(archive), archive_sha256=archive_sha256,
                  archive_bytes=archive.stat().st_size, manifest_path=str(manifest),
                  manifest_sha256=sha(manifest), plan_sha256=sha(destination / "plan.json"),
                  archived_members=len(rows), verified_regular_file_members=verified,
                  source_regular_file_bytes=sum(r.get("bytes", 0) for r in rows),
                  sources_deleted_or_modified=False, source_unchanged_after_archive=True,
                  required_external_dependencies=external,
                  restore_argv=["tar", "-xf", str(archive), "-C", "/"],
                  restoration_requirement="Backup only; no restore was performed. Restore to original absolute paths and independently resolve every restored link, including existing ancestor aliases. Check external interpreter/system/ELF/driver dependencies against recorded identities. Archive verification is not a fresh-filesystem restore or ABI validation.",
                  tool_sha256=tool_sha256)
    save(destination / "verification.json", result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(archive_plan(json.loads(args.plan.read_text()), args.output)))


if __name__ == "__main__":
    main()
