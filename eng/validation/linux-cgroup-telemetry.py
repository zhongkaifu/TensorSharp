#!/usr/bin/env python3
"""Read process cgroup telemetry through its visible mounts (v1, v2 or hybrid).

The normalized memory.current/max/peak keys are byte-valued strings for existing
report readers. Raw filenames, controller versions and visible ancestor limits
are preserved; v1 failcnt is never mislabeled as a v2 OOM event. This observes
usage, not a reservation or an assertion that hidden ancestors are unlimited.
"""
from pathlib import Path, PurePosixPath
import os
import re


def unescape(value):
    return re.sub(r"\\([0-7]{3})", lambda m: chr(int(m[1], 8)), value)


def absolute(value):
    if not value.startswith('/') or '\0' in value or any(p in ('.', '..') for p in value.split('/')):
        raise ValueError('Invalid cgroup path')
    return PurePosixPath(value)


def controller_mounts(membership, mountinfo):
    groups = {}
    for line in membership.splitlines():
        _, controllers, path = line.split(':', 2)
        for controller in controllers.split(','):
            if controller in groups:
                raise ValueError('Duplicate cgroup membership')
            groups[controller] = absolute(path)
    result = {}
    for line in mountinfo.splitlines():
        left, right = line.split(' - ', 1)
        fields, super_fields = left.split(), right.split()
        fs = super_fields[0]
        if fs not in ('cgroup', 'cgroup2'):
            continue
        root, point = (absolute(unescape(s)) for s in fields[3:5])
        for controller in ('memory', 'cpu'):
            # A v1 membership owns that controller even when 0:: is present.
            version = 1 if controller in groups else 2
            if version == 1:
                if fs != 'cgroup' or controller not in super_fields[2].split(','):
                    continue
                member = groups[controller]
            else:
                if fs != 'cgroup2' or '' not in groups:
                    continue
                member = groups['']
            try:
                relative = member.relative_to(root)
            except ValueError:
                continue
            candidate = dict(version=version, mount_root=str(root), mount_point=str(point),
                             path=str(point / relative), membership=str(member))
            if controller not in result or len(root.parts) < len(PurePosixPath(result[controller]['mount_root']).parts):
                result[controller] = candidate
    return result


def capture(pid=None, proc_root=Path('/proc'), fs_root=Path('/')):
    proc = Path(proc_root) / str(pid if pid is not None else os.getpid())
    mounts = controller_mounts((proc/'cgroup').read_text(), (proc/'mountinfo').read_text())
    result = {'controllers': {}, 'errors': []}

    def read(path, required=False):
        local = Path(fs_root).joinpath(*absolute(path).parts[1:])
        try:
            return local.read_text().strip()
        except OSError as error:
            if required or not isinstance(error, FileNotFoundError):
                result['errors'].append(f'{path}: {type(error).__name__}')
            return None

    for name, mount in mounts.items():
        version, directory = mount['version'], PurePosixPath(mount['path'])
        if name == 'memory':
            names = ('memory.current', 'memory.max', 'memory.peak', 'memory.events', 'memory.stat') if version == 2 else (
                'memory.usage_in_bytes', 'memory.limit_in_bytes', 'memory.max_usage_in_bytes',
                'memory.failcnt', 'memory.oom_control', 'memory.stat', 'memory.use_hierarchy')
        else:
            names = ('cpu.stat', 'cpu.max') if version == 2 else ('cpu.stat', 'cpu.cfs_quota_us', 'cpu.cfs_period_us')
        files = {key: value for key in names if (value := read(str(directory/key), required=key == names[0])) is not None}
        result['controllers'][name] = dict(mount, files=files, ancestors_above_mount_visible=mount['mount_root'] == '/')
        if name == 'memory':
            aliases = dict(zip(names[:3], ('memory.current', 'memory.max', 'memory.peak')))
            for key, value in files.items():
                result[aliases.get(key, key)] = value
            ancestors = []
            parent = directory
            while True:
                limit = read(str(parent/names[1]))
                used = read(str(parent/names[0]))
                hierarchy = read(str(parent/'memory.use_hierarchy')) if version == 1 else None
                ancestors.append(dict(path=str(parent), limit=limit, current=used,
                                      use_hierarchy=hierarchy))
                if str(parent) == mount['mount_point']:
                    break
                parent = parent.parent
            result['controllers'][name]['visible_ancestors'] = ancestors
        else:
            # Keep raw units: v1 throttled_time is ns; v2 throttled_usec is us.
            result.update(files)
    if 'memory' not in mounts:
        result['errors'].append('No visible memory controller mount for the process')
    return result
