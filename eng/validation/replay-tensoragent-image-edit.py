#!/usr/bin/env python3
"""Compare a captured TensorAgent edit with its equivalent direct image API call.

Use an isolated host that still retains the captured uploads. This uses the same
ordered source/references, mask, TensorAgent's 1024x1024 target area and its
keepSourceSize (every TensorAgent edit sends it). It checks
exact output parity and protected pixels; a single pair is not a latency study.

The parity holds for one edit noise: a capture made by a build whose edits drew
their noise from the seed alone replays only on a host started with
TS_QWEN21_EDIT_NOISE=seed.
"""
import argparse
import base64
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import time

import requests

MASK_KEYS = ('maskPath', 'maskMode', 'maskInvert', 'maskFeather', 'maskCrop', 'maskCropPadding')


def canonical_payload(request):
    message = next(message for message in reversed(request['messages']) if message['role'] == 'user')
    return {'prompt': message['content'].strip(), 'imagePaths': list(message['stillImagePaths']),
            'targetArea': 1024 * 1024, 'keepSourceSize': True,
            **{key: message[key] for key in MASK_KEYS if key in message}}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--connection', type=Path, required=True)
    parser.add_argument('--report', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    connection = json.loads(args.connection.read_text(encoding='utf-8-sig'))
    captured = json.loads(args.report.read_text(encoding='utf-8-sig'))
    assert captured['status'] == 'passed'
    assert not any(frame.get('image_loras') for frame in captured['frames']), 'Replay requires the no-LoRA fixture'
    payload = canonical_payload(captured['request'])
    client = requests.Session()
    client.headers['Cookie'] = connection['cookie']

    def download(url):
        response = client.get(connection['baseUrl'] + url, timeout=30)
        response.raise_for_status()
        return response.content

    source_file = payload['imagePaths'][0]
    message = next(message for message in reversed(captured['request']['messages']) if message['role'] == 'user')
    attachment = next((item for item in message.get('attachments', []) if item.get('file') == source_file), {})
    # HEIC pixels are measured against the exact original-size PNG made by the
    # runtime decoder, rather than a second codec's potentially different rounding.
    source = download('/uploads/' + (attachment.get('editFile') or source_file))
    mask = download('/uploads/' + payload['maskPath'])
    started = time.perf_counter()
    response = client.post(connection['baseUrl'] + '/api/image-edit/stream', json=payload, timeout=1200)
    response.raise_for_status()
    elapsed = time.perf_counter() - started
    frames = [json.loads(line[6:]) for line in response.text.splitlines() if line.startswith('data: ')]
    assert not any(frame.get('error') for frame in frames), 'Canonical image API failed'
    done = next(frame for frame in frames if frame.get('done'))
    output = download(done['url'])
    (args.out / 'result.png').write_bytes(output)
    previous = (args.report.parent / 'result.png').read_bytes()
    spec = importlib.util.spec_from_file_location('mask_bench', Path(__file__).with_name('qwen-image21-mask-bench.py'))
    metrics = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(metrics)
    pixels = metrics.measure_pixels(io.BytesIO(source), io.BytesIO(output), io.BytesIO(mask))
    previews = [metrics.measure_pixels(io.BytesIO(source), io.BytesIO(base64.b64decode(frame['image'].split(',', 1)[1])), io.BytesIO(mask))
                for frame in frames if frame.get('image')]
    parity = output == previous
    report = {'status': 'passed' if parity else 'failed', 'request': payload, 'seconds': elapsed,
              'captured_ui_seconds': captured['seconds'], 'byte_identical': parity,
              'output_sha256': hashlib.sha256(output).hexdigest(), 'pixels': pixels,
              'source_pixels_sha256': hashlib.sha256(source).hexdigest(),
              'preview_preservation': previews,
              'limitations': ['One canonical replay, same host and ordered references; thermal/cache conditions are not controlled.',
                              'Checks deterministic output and preservation, not a statistical latency regression or general quality score.']}
    (args.out / 'summary.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps({key: report[key] for key in ('status', 'seconds', 'captured_ui_seconds', 'byte_identical', 'pixels')}))
    assert parity, 'Promoted UI request differs from equivalent canonical source-first request'


if __name__ == '__main__':
    main()
