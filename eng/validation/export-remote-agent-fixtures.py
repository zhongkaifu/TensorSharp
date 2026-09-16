#!/usr/bin/env python3
"""Export existing release-agent prompts/oracles without running any workload."""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path


def export(smoke=False):
    source = Path(__file__).with_name('validate-release-agent-workflows.py')
    spec = importlib.util.spec_from_file_location('release_agent_cases', source)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    cases = []
    distinct_scenarios = list(module.CASES) if smoke else [name for name, case in module.CASES.items() if case.get('execution')]
    for variant, degrees, scenarios in (('original', [1] if smoke else [1, 4], list(module.CASES)),
                                         ('distinct', [1] if smoke else [4], distinct_scenarios)):
        for name in scenarios:
            for degree in degrees:
                for index in range(degree):
                    trial = f'c{degree}-i{index}'
                    cases.append({'variant': variant, 'scenario': name, 'trial': trial,
                                  'wave': f'{variant}-c{degree}-{name}', 'concurrency': degree,
                                  'spec': module.case_spec(name, trial, distinct_inputs=variant == 'distinct')})
    return {'status': 'prepared-not-executed', 'source': str(source),
            'source_sha256': hashlib.sha256(source.read_bytes()).hexdigest(),
            'scope': '12-case-smoke-only' if smoke else '46-case-matched-release-campaign', 'cases': cases}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--smoke', action='store_true', help='Prepare 12 serial smoke cases, not the required 46-case campaign')
    args = parser.parse_args()
    if args.output.exists():
        parser.error('Use a new fixture path; existing exported fixtures are immutable')
    result = export(args.smoke)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + '\n')
    print(f'Prepared {len(result["cases"])} cases; executed none.')


if __name__ == '__main__':
    main()
