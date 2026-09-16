#!/usr/bin/env python3
"""Record one complete application and require every comparison peer to match."""
import argparse
import json
from pathlib import Path
from release_application_identity import application_files, make_manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--application', type=Path, required=True)
    parser.add_argument('--peer', type=Path, action='append', default=[])
    parser.add_argument('--managed-build', type=Path, required=True)
    parser.add_argument('--managed-build-sha256', required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    manifest = make_manifest(args.application, args.managed_build, args.managed_build_sha256)
    for peer in args.peer:
        if application_files(peer) != manifest['non_native_files_sha256']:
            raise ValueError('Comparison peer differs outside libGgmlOps.so: ' + str(peer))
    # Never replace prior identity evidence or silently create a new baseline.
    with args.output.open('x', encoding='utf-8') as stream:
        stream.write(json.dumps(manifest, indent=2) + '\n')


if __name__ == '__main__':
    main()
