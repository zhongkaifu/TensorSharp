#!/usr/bin/env python3
"""Cancel real Studio inference after denoising starts; verify worker recovery."""
import argparse
import json
import pathlib
import time

from validate import Host, uid


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--connection', type=pathlib.Path, required=True)
    parser.add_argument('--project', required=True)
    parser.add_argument('--output', type=pathlib.Path, required=True)
    args = parser.parse_args()
    root = pathlib.Path(__file__).resolve().parents[2]
    output = args.output.resolve()
    assert any(output.is_relative_to(root / folder) for folder in ('artifacts', 'docs/validation'))
    output.mkdir(parents=True, exist_ok=True)
    connection = json.loads(args.connection.read_text())
    host = Host(connection['baseUrl'], connection['cookie'])
    project = host.api('projects/get', {'projectId': args.project})
    generated = next(a for a in project['assets'] if a.get('generation'))
    request = {**generated['generation']['request'], 'requestId': uid(), 'seed': 43}
    job = host.api('jobs/generate', request)
    states = []
    deadline = time.monotonic() + 600
    cancelled_at = None
    while time.monotonic() < deadline:
        current = next(j for j in host.api('jobs/list', {'projectId': args.project}) if j['id'] == job['id'])
        marker = [current['state'], current['stage']]
        if not states or states[-1] != marker:
            states.append(marker)
            print(*marker, flush=True)
        if current['stage'] == 'denoise' and cancelled_at is None:
            result = host.api('jobs/cancel', {'jobId': job['id']})
            cancelled_at = time.monotonic()
            assert result['state'] in ('cancelling', 'cancelled'), result
        if current['state'] in ('failed', 'cancelled', 'completed', 'interrupted'):
            assert current['state'] == 'cancelled' and cancelled_at is not None, current
            break
        time.sleep(.2)
    else:
        raise TimeoutError('Native cancellation did not complete')
    cancellation_seconds = time.monotonic() - cancelled_at
    after = host.api('projects/get', {'projectId': args.project})
    assert len(after['assets']) == len(project['assets']), 'Cancelled inference must not add a phantom asset'
    render = host.api('jobs/render', {'projectId': args.project, 'expectedRevision': after['revision'], 'requestId': uid()})
    finished = host.poll(args.project, render['id'])
    result = {'passed': True, 'job': current, 'states': states, 'cancellationSeconds': cancellation_seconds,
        'subsequentRender': finished, 'scope': 'Real LTX native job cancelled during denoise; terminal state waits for native return, no asset created, worker renders successfully afterward.'}
    (output / 'report.json').write_text(json.dumps(result, indent=2))
    print('NATIVE CANCELLATION PASSED', job['id'], flush=True)


if __name__ == '__main__':
    main()
