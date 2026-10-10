#!/usr/bin/env python3
"""Measure real Qwen 2.1 masked edits and enforce exact protected RGBA pixels.

Requires Pillow, numpy and requests. CLI and HTTP/SSE use the same image, seed,
steps and grayscale mask. Crop changes the model's context/workload: timings are
not a quality-equivalent speedup. Generated reports/images/logs stay ignored.
Repeat --reference to add conditioning pictures after the source --image; only
the source is edited and checked for protected pixels.
Example:
  python eng/validation/qwen-image21-mask-bench.py --image photo.png \
      --model-dir C:/Works/models/qwen-image-2.1 --steps 40 --crop both
  python eng/validation/qwen-image21-mask-bench.py --image photo.png \
      --server-url http://127.0.0.1:5050 --steps 40 --crop both
"""
import argparse
import base64
import hashlib
import io
import json
import os
from pathlib import Path
import subprocess
import time

import numpy as np
from PIL import Image, ImageDraw, ImageOps
import requests

ROOT = Path(__file__).resolve().parents[2]


def revision(path):
    def git(*args):
        return subprocess.check_output(['git', '-C', str(path), *args], text=True).strip()
    return {'revision': git('rev-parse', 'HEAD'), 'changes': git('status', '--short')}


def measure_pixels(source, output, mask):
    expected = np.asarray(ImageOps.exif_transpose(Image.open(source)).convert('RGBA')).astype(np.int16)
    actual = np.asarray(Image.open(output).convert('RGBA')).astype(np.int16)
    # The runtime uses positive float luminance, not rounded 8-bit luminance.
    editable = np.any(np.asarray(ImageOps.exif_transpose(Image.open(mask)).convert('RGB')) > 0, axis=-1)
    if expected.shape != actual.shape:
        raise AssertionError(f'Output shape {actual.shape} differs from source {expected.shape}')
    difference = np.abs(actual - expected)
    changed = np.any(difference != 0, axis=-1)
    outside = int(np.count_nonzero(changed & ~editable))
    result = {
        'width': expected.shape[1], 'height': expected.shape[0],
        'protected_pixels': int(np.count_nonzero(~editable)),
        'changed_protected_pixels': outside,
        'editable_pixels': int(np.count_nonzero(editable)),
        'changed_editable_pixels': int(np.count_nonzero(changed & editable)),
        'editable_rgba_mean_absolute_error_255': float(difference[editable].mean()) if editable.any() else 0,
        'protected_rgba_max_error_255': int(difference[~editable].max()) if (~editable).any() else 0,
    }
    if outside:
        raise AssertionError(f'{outside} protected RGBA pixels changed: {result}')
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--image', type=Path, required=True)
    parser.add_argument('--reference', type=Path, action='append', default=[],
                        help='Additional conditioning image, repeatable; source --image stays first')
    parser.add_argument('--mask', type=Path)
    parser.add_argument('--model-dir', type=Path, default=ROOT.parent / 'models/qwen-image-2.1')
    parser.add_argument('--server-url')
    parser.add_argument('--backend', default='ggml_cuda')
    parser.add_argument('--prompt', default='Change the red cube to a blue cube. Keep the same lighting and shape.')
    parser.add_argument('--steps', type=int, default=40)
    parser.add_argument('--width', type=int, default=512)
    parser.add_argument('--height', type=int, default=512)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--reps', type=int, default=1)
    parser.add_argument('--crop', choices=['off', 'on', 'both'], default='both')
    parser.add_argument('--padding', type=int, default=32)
    parser.add_argument('--feather', type=int, default=8)
    parser.add_argument('--timeout', type=int, default=1200)
    parser.add_argument('--out', type=Path, default=ROOT / 'docs/validation/qwen-image21-mask/benchmark')
    args = parser.parse_args()
    if args.reps < 1 or args.steps < 1 or args.timeout < 1:
        parser.error('reps, steps and timeout must be positive')
    if min(args.width, args.height) < 32 or args.width % 32 or args.height % 32:
        parser.error('width and height must be positive multiples of 32')
    args.out.mkdir(parents=True, exist_ok=True)
    source = args.image.resolve()
    references = [path.resolve() for path in args.reference]
    mask = args.mask
    if mask is None:
        width, height = ImageOps.exif_transpose(Image.open(source)).size
        selection = Image.new('L', (width, height))
        ImageDraw.Draw(selection).rectangle((width//4, height//4, 3*width//4-1, 3*height//4-1), fill=255)
        mask = args.out / 'mask.png'
        selection.save(mask)
    mask = mask.resolve()
    report = {
        'parameters': {k: str(v) if isinstance(v, Path) else [str(p) for p in v] if k == 'reference' else v
                       for k, v in vars(args).items()},
        'source_sha256': hashlib.sha256(source.read_bytes()).hexdigest(),
        'references': [{'path': str(path), 'sha256': hashlib.sha256(path.read_bytes()).hexdigest()}
                       for path in references],
        'mask_sha256': hashlib.sha256(mask.read_bytes()).hexdigest(),
        # The initial noise of an edit follows the seed and, by default, the source and
        # reference images (TS_QWEN21_EDIT_NOISE); output hashes only compare like with like.
        'edit_noise': ('set by the server process' if args.server_url else
                       (os.environ.get('TS_QWEN21_EDIT_NOISE') or '').strip().lower() or 'references'),
        'dependencies': {'tensorsharp': revision(ROOT), 'ggml': revision(ROOT / 'ExternalProjects/ggml')},
        'limitations': [
            'Exact protected pixels and nonzero edits do not measure prompt adherence or perceptual quality.',
            'Crop changes model context, token count and quality; this is not a quality-equivalent speedup claim.',
            'CLI wall time includes model loading; HTTP wall time excludes server startup and uploads but includes SSE previews.',
            'Additional references change conditioning work; compare timings only with the same ordered references.',
            'OS cache, thermal state and other GPU activity are not controlled; no cross-device conclusion is implied.',
        ], 'runs': []}
    report_path = args.out / 'report.json'
    crops = [False, True] if args.crop == 'both' else [args.crop == 'on']
    upload_paths = None
    if args.server_url:
        upload_paths = []
        for path in (source, *references, mask):
            with path.open('rb') as stream:
                response = requests.post(args.server_url + '/api/upload', files={'file': (path.name, stream, 'image/png')}, timeout=60)
            response.raise_for_status()
            upload_paths.append(response.json()['file'])
    for crop in crops:
        for repetition in range(args.reps):
            name = f'{"api" if args.server_url else "cli"}-crop-{str(crop).lower()}-{repetition + 1}'
            output = (args.out / (name + '.png')).resolve()
            run = {'name': name, 'crop': crop, 'status': 'failed'}
            started = time.perf_counter()
            try:
                if args.server_url:
                    payload = dict(imagePaths=upload_paths[:-1], maskPath=upload_paths[-1], maskMode='grayscale',
                        prompt=args.prompt, steps=args.steps, width=args.width, height=args.height, seed=args.seed,
                        cfg=1, maskCrop=crop, maskCropPadding=args.padding, maskFeather=args.feather)
                    run['request'] = payload
                    frames = []
                    final = None
                    with requests.post(args.server_url + '/api/image-edit/stream', json=payload, stream=True, timeout=args.timeout) as response:
                        response.raise_for_status()
                        for line in response.iter_lines():
                            if not line.startswith(b'data: '):
                                continue
                            frame = json.loads(line[6:])
                            if frame.get('error'):
                                raise RuntimeError(frame['error'])
                            if frame.get('image'):
                                preview = io.BytesIO(base64.b64decode(frame['image'].split(',', 1)[1]))
                                frame['preservation'] = measure_pixels(source, preview, mask)
                                del frame['image']
                            frames.append(frame)
                            if frame.get('done'):
                                final = frame
                    if not final or not final.get('url'):
                        raise RuntimeError('SSE ended without a successful final image')
                    image = requests.get(args.server_url + final['url'], timeout=60)
                    image.raise_for_status()
                    output.write_bytes(image.content)
                    run['frames'] = frames
                else:
                    models = args.model_dir
                    command = ['dotnet', str(ROOT / 'TensorSharp.Cli/bin/TensorSharp.Cli.dll'),
                        '--model', str(models / 'qwen_image_2.1_Q4_K_M.gguf'), '--backend', args.backend,
                        '--qwen-image-vae', str(models / 'qwen_image_2.1_vae_bf16.safetensors'),
                        '--qwen-image-vl', str(models / 'Qwen3VL-8B-Instruct-Q4_K_M.gguf'),
                        '--qwen-image-mmproj', str(models / 'mmproj-Qwen3VL-8B-Instruct-F16.gguf'),
                        '--image', str(source), '--mask', str(mask), '--mask-mode', 'grayscale',
                        '--mask-feather', str(args.feather), '--mask-crop-padding', str(args.padding),
                        '--prompt', args.prompt, '--width', str(args.width), '--height', str(args.height),
                        '--diffusion-steps', str(args.steps), '--cfg', '1', '--diffusion-seed', str(args.seed),
                        '--output', str(output)]
                    if crop:
                        command.append('--mask-crop')
                    for reference in references:
                        command.extend(['--image', str(reference)])
                    run['command'] = command
                    with (args.out / (name + '.log')).open('w', encoding='utf8') as log:
                        process = subprocess.run(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, timeout=args.timeout)
                    if process.returncode:
                        raise RuntimeError(f'CLI exited {process.returncode}; see {name}.log')
                run['seconds'] = time.perf_counter() - started
                run['pixels'] = measure_pixels(source, output, mask)
                if run['pixels']['editable_pixels'] and not run['pixels']['changed_editable_pixels']:
                    raise AssertionError('The selected region did not change')
                run['output_sha256'] = hashlib.sha256(output.read_bytes()).hexdigest()
                run['status'] = 'passed'
            except Exception as error:
                run['seconds'] = time.perf_counter() - started
                run['error'] = str(error)
            report['runs'].append(run)
            report_path.write_text(json.dumps(report, indent=2), encoding='utf8')
            print(json.dumps(run), flush=True)
    print(f'Report: {report_path}')
    return int(any(run['status'] != 'passed' for run in report['runs']))


if __name__ == '__main__':
    raise SystemExit(main())
