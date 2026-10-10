#!/usr/bin/env python3
"""Read-only cgroup/RSS inspection before a constrained model validation run.

This never creates a cgroup, moves a process, drops caches, or changes a limit.
Writable mount/permission checks are evidence of a candidate, not proof that
the container/host permits creation or controller delegation.
"""
import argparse
import json
import os
from pathlib import Path, PurePosixPath
import re
import sys


def read(path):
    try:
        return path.read_text().strip()
    except (OSError, UnicodeError):
        return None


def unescape_mount(value):
    return re.sub(r"\\([0-7]{3})", lambda match: chr(int(match[1], 8)), value)


def visible_group(mount_root, mount_point, membership):
    """Translate a host cgroup path into the visible mount without duplication."""
    try:
        relative = PurePosixPath(membership).relative_to(PurePosixPath(mount_root))
    except ValueError:
        return None
    return Path(mount_point).joinpath(*relative.parts)


def memberships(text):
    result = {}
    for line in (text or "").splitlines():
        _, controllers, path = line.split(":", 2)
        result[controllers] = path
    return result


def memory_mounts(mountinfo, groups):
    result = []
    for line in mountinfo.splitlines():
        left, right = line.split(" - ", 1)
        fields, super_fields = left.split(), right.split()
        filesystem = super_fields[0]
        if filesystem == "cgroup2":
            membership = groups.get("")
            version = 2
        elif filesystem == "cgroup" and "memory" in super_fields[2].split(","):
            membership = next((path for names, path in groups.items() if "memory" in names.split(",")), None)
            version = 1
        else:
            continue
        root, point = map(unescape_mount, fields[3:5])
        group = visible_group(root, point, membership) if membership else None
        result.append({"version": version, "mountRoot": root, "mountPoint": point,
                       "mountOptions": fields[5], "superOptions": super_fields[2],
                       "membership": membership, "visibleGroup": str(group) if group else None})
    return result


def key_values(text):
    result = {}
    for line in (text or "").splitlines():
        parts = line.split()
        if len(parts) == 2:
            try:
                result[parts[0]] = int(parts[1])
            except ValueError:
                result[parts[0]] = parts[1]
    return result


def inspect_mount(mount):
    group = Path(mount["visibleGroup"]) if mount["visibleGroup"] else None
    if group is None or not group.is_dir():
        return dict(mount, status="membership-not-visible", childCreationVerified=False)
    version = mount["version"]
    names = (["cgroup.type", "cgroup.controllers", "cgroup.subtree_control", "memory.max",
              "memory.high", "memory.current", "memory.peak", "memory.swap.max",
              "memory.swap.current", "memory.events", "memory.stat"] if version == 2 else
             ["memory.limit_in_bytes", "memory.usage_in_bytes", "memory.max_usage_in_bytes",
              "memory.memsw.limit_in_bytes", "memory.memsw.usage_in_bytes", "memory.failcnt",
              "memory.oom_control", "memory.use_hierarchy", "memory.stat"])
    files = {name: value for name in names if (value := read(group / name)) is not None}
    limit_name = "memory.max" if version == 2 else "memory.limit_in_bytes"
    ancestor_limits = []
    ancestor = group
    mount_point = Path(mount["mountPoint"])
    while ancestor == mount_point or mount_point in ancestor.parents:
        limit = read(ancestor / limit_name)
        if limit is not None:
            ancestor_limits.append({"path": str(ancestor), "limit": limit})
        if ancestor == mount_point:
            break
        ancestor = ancestor.parent
    readonly = "ro" in mount["mountOptions"].split(",") or "ro" in mount["superOptions"].split(",")
    writable = not readonly and os.access(group, os.W_OK | os.X_OK)
    memory_visible = version == 1 or "memory" in files.get("cgroup.controllers", "").split()
    candidate = writable and memory_visible
    return dict(mount, status="observed", files=files,
                memoryStat=key_values(files.get("memory.stat")), visibleAncestorLimits=ancestor_limits,
                ancestorsAboveMountVisible=mount["mountRoot"] == "/",
                memoryControllerAvailableForChildren=memory_visible,
                childCreationCandidate=candidate, childCreationVerified=False,
                candidateReason="Writable memory-cgroup directory; creation/delegation was not attempted."
                if candidate else ("Memory controller is not available for child groups."
                                   if not memory_visible else "Memory-cgroup mount or directory is not writable."))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pid", type=int, default=os.getpid(), help="Process to inspect (default: this inspector).")
    parser.add_argument("--model", type=Path, help="Optional GGUF file; stat only, does not read its payload.")
    parser.add_argument("--proposed-limit-bytes", type=int, help="Record a proposed child limit without applying it.")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if not sys.platform.startswith("linux"):
        parser.error("Linux /proc and cgroup mounts are required.")
    if args.pid <= 0 or (args.proposed_limit_bytes is not None and args.proposed_limit_bytes <= 0):
        parser.error("PID and proposed limit must be positive.")
    proc = Path("/proc") / str(args.pid)
    status = read(proc / "status")
    cgroup = read(proc / "cgroup")
    mountinfo = read(Path("/proc/self/mountinfo"))
    if status is None or cgroup is None or mountinfo is None:
        parser.error("Cannot read the requested process status/cgroup or this process's mount namespace.")
    memory_fields = {}
    for line in status.splitlines():
        key, _, value = line.partition(":")
        if key.startswith("Vm") or key.startswith("Rss"):
            memory_fields[key] = value.strip()
    model_bytes = args.model.stat().st_size if args.model else None
    report = {
        "readOnly": True, "pid": args.pid, "uid": os.getuid(),
        "processMemory": memory_fields,
        "memoryCgroups": [inspect_mount(m) for m in memory_mounts(mountinfo, memberships(cgroup))],
        "modelFile": str(args.model.resolve()) if args.model else None,
        "modelFileBytes": model_bytes, "proposedLimitBytes": args.proposed_limit_bytes,
        "proposedLimitBelowModelFile": args.proposed_limit_bytes < model_bytes
        if args.proposed_limit_bytes is not None and model_bytes is not None else None,
        "limitations": [
            "No limit has been installed and no constrained inference has been run.",
            "Process RSS, cgroup memory (including charged file cache), and managed payload budgets are different measurements.",
            "CUDA VRAM is not constrained by this host-memory cgroup inspection.",
            "A limit above the model file size cannot establish operation with host capacity smaller than that file.",
            "Reserve headroom for the runtime, live KV, activations, native pools, libraries and file-cache effects.",
            "Writable permissions do not prove cgroup delegation; unseen ancestor limits may still constrain the process."]}
    rendered = json.dumps(report, indent=2) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered)
    else:
        print(rendered, end="")


if __name__ == "__main__":
    main()
