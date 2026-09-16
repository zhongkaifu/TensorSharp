#!/usr/bin/env python3
"""Snapshot identical HTTP applications after an existing model owner finishes.

Copies are fresh and independent; only their top-level native bridge differs.
This prepares inputs and never starts a model or establishes release quality.
"""
import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import time

from release_application_identity import (application_files, checked_build, digest,
    make_manifest, read_pinned_json, require_sha)


def require(condition, message):
    if not condition:
        raise ValueError(message)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('application', 'managed-build', 'baseline', 'candidate', 'owner-profile', 'output'):
        parser.add_argument('--' + name, type=Path, required=True)
    for name in ('managed-build-sha256', 'baseline-sha256', 'candidate-sha256', 'owner-profile-sha256'):
        parser.add_argument('--' + name, required=True)
    args = parser.parse_args()
    require(os.name == 'posix' and Path('/proc').is_dir(), 'Execution requires the Linux validation machine')
    require(not args.output.exists(), 'Use a fresh output directory')
    owner = read_pinned_json(args.owner_profile, args.owner_profile_sha256, 'Finished owner profile')
    require(owner.get('finished_at_unix') and owner.get('status') != 'running', 'The model owner must finish first')
    pid = owner.get('server_pid')
    require(isinstance(pid, int) and not Path(f'/proc/{pid}').exists(), 'The recorded model server must be gone')
    gpu = subprocess.run(['nvidia-smi', '--query-compute-apps=pid', '--format=csv,noheader,nounits'],
                         capture_output=True, text=True, timeout=30, check=True)
    require(not gpu.stdout.strip(), 'GPU compute is active; wait for the assigned quiet lane')
    expected_managed = checked_build(args.managed_build, args.managed_build_sha256)
    source_manifest = make_manifest(args.application, args.managed_build, args.managed_build_sha256)
    require(len(expected_managed) == 12, 'This HTTP comparison requires the independently recorded twelve-assembly build')
    for label in ('baseline', 'candidate'):
        path = getattr(args, label)
        require(path.is_file() and not path.is_symlink(), label + ' native must be a regular file')
        require(digest(path) == require_sha(getattr(args, label + '_sha256')), label + ' native changed')
    args.output.mkdir(parents=True)
    record = {'status': 'copying', 'release_qualified': False, 'started_at_unix': time.time(),
        'source_application': str(args.application.resolve()), 'owner_profile': str(args.owner_profile.resolve()),
        'owner_profile_sha256': args.owner_profile_sha256,
        'managed_build': source_manifest['managed_build'], 'gpu_processes_before': gpu.stdout,
        'applications': {}, 'scope': 'Input identity only; no model execution, correctness, placement or performance qualification.'}
    report = args.output / 'staging.json'
    def save():
        report.write_text(json.dumps(record, indent=2) + '\n')
    save()
    try:
        for label in ('baseline', 'candidate'):
            destination = args.output / label
            destination.mkdir()
            for name, expected in source_manifest['non_native_files_sha256'].items():
                source = args.application / name
                target = destination / name
                require(not source.is_symlink() and source.is_file(), 'Source member changed type: ' + name)
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(source, target)
                require(digest(target) == expected, 'Copied application member changed: ' + name)
            native = destination / 'libGgmlOps.so'
            shutil.copy2(getattr(args, label), native)
            require(digest(native) == getattr(args, label + '_sha256'), 'Copied native changed')
            require(application_files(destination) == source_manifest['non_native_files_sha256'],
                    'Copied non-native application differs')
            record['applications'][label] = {'directory': str(destination.resolve()),
                'server_assembly': str((destination / 'TensorSharp.Server.Host.dll').resolve()),
                'native_sha256': digest(native), 'non_native_file_count': len(source_manifest['non_native_files_sha256'])}
            save()
        require(application_files(args.application) == source_manifest['non_native_files_sha256'],
                'Source application changed during staging')
        require(checked_build(args.managed_build, args.managed_build_sha256) == expected_managed,
                'Managed build changed during staging')
        require(digest(args.owner_profile) == args.owner_profile_sha256, 'Finished owner evidence changed')
        manifest = args.output / 'application-manifest.json'
        with manifest.open('x') as stream:
            stream.write(json.dumps(source_manifest, indent=2) + '\n')
        record.update(status='prepared', application_manifest=str(manifest.resolve()),
                      application_manifest_sha256=digest(manifest))
    except Exception as error:
        record.update(status='failed', error=repr(error))
        raise
    finally:
        record['finished_at_unix'] = time.time()
        save()
    print(json.dumps({key: record[key] for key in ('status', 'release_qualified', 'application_manifest_sha256')}))


if __name__ == '__main__':
    main()
