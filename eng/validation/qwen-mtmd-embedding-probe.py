#!/usr/bin/env python3
"""Prepare/run a pinned, bounded projection export through unchanged mtmd.

Build only compiles TensorSharp's tiny main.cpp, against existing import libs.
Run is explicitly serial with other GPU work; its measurements are diagnostics.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import struct
import subprocess
import sys
import time

HERE = Path(__file__).resolve().parent
EXPECTED_REVISION = '4ebdf2c74acce30883d8e34b7c70b3eb8146f2fe'
EXPECTED_IMAGE = '10145e33904f59e8f4e61cdd035cebbf45d7e5591d8ded6cdedd5aa82e022b18'


def sha(path):
    value = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1 << 20), b''):
            value.update(block)
    return value.hexdigest()


def pin(path):
    path = Path(path).resolve()
    return {'Path': str(path), 'Bytes': path.stat().st_size, 'Sha256': sha(path)}


def metadata(path):
    path = Path(path).resolve(); stat = path.stat()
    return {'Path': str(path), 'Bytes': stat.st_size, 'MtimeNs': stat.st_mtime_ns}


def save(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')


def fresh(path):
    path = Path(path).resolve()
    if path.exists():
        raise ValueError('Fresh output directory required: ' + str(path))
    path.mkdir(parents=True)
    return path


def source_identity(source):
    source = Path(source).resolve()
    revision = subprocess.check_output(['git', '-C', str(source), 'rev-parse', 'HEAD'], text=True).strip()
    status = subprocess.check_output(['git', '-C', str(source), 'status', '--porcelain', '--untracked-files=no'], text=True).splitlines()
    if revision != EXPECTED_REVISION or status:
        raise ValueError('Require unchanged llama.cpp baseline revision ' + EXPECTED_REVISION)
    return {'Path': str(source), 'Revision': revision, 'TrackedStatus': status}


def verify_pins(values):
    for value in values:
        if pin(value['Path']) != value:
            raise ValueError('Pinned input changed: ' + value['Path'])


def build(args):
    out = fresh(args.output)
    source = Path(args.source).resolve(); baseline = Path(args.baseline_build).resolve()
    identity = source_identity(source)
    code = HERE / 'QwenMtmdEmbeddingProbe' / 'main.cpp'
    headers = [source / 'include/llama.h', source / 'tools/mtmd/mtmd.h',
               source / 'ggml/include/ggml.h', source / 'ggml/include/ggml-backend.h']
    policy_sources = [source / 'src/llama.cpp', source / 'src/llama-model.cpp', source / 'src/llama-model-loader.cpp']
    libs = [baseline / 'tools/mtmd/mtmd.lib', baseline / 'src/llama.lib',
            baseline / 'ggml/src/ggml.lib', baseline / 'ggml/src/ggml-base.lib']
    inputs = [pin(path) for path in [code, Path(__file__), *headers, *policy_sources, *libs]]
    vcvars = Path(args.vcvars).resolve()
    executable = out / 'QwenMtmdEmbeddingProbe.exe'
    command = ['cl.exe', '/nologo', '/std:c++17', '/EHsc', '/W4', '/MD', '/O2', '/DLLAMA_SHARED', '/DGGML_SHARED',
               '/I' + str(source / 'include'), '/I' + str(source / 'tools/mtmd'), '/I' + str(source / 'ggml/include'),
               str(code), '/Fe:' + str(executable), '/Fo:' + str(out / 'probe.obj'), '/link', *map(str, libs)]
    # This cmd file is exclusively a compiler invocation, never a filesystem action.
    for value in [str(vcvars), *command]:
        if any(char in value for char in '\r\n&|<>^%!"'):
            raise ValueError('Unsupported command metacharacter in build path/argument')
    cmd = out / 'build.cmd'
    cmd.write_text('@call "' + str(vcvars) + '" >nul\r\n@if errorlevel 1 exit /b %errorlevel%\r\n'
                   + subprocess.list2cmdline(command) + '\r\n@exit /b %errorlevel%\r\n', encoding='utf-8')
    started = time.time()
    with (out / 'build.log').open('wb') as log:
        result = subprocess.run(['cmd.exe', '/d', '/c', str(cmd)], cwd=out, stdout=log, stderr=subprocess.STDOUT, timeout=120)
    verify_pins(inputs)
    report = {'Schema': 'qwen-mtmd-probe-build.v1', 'Passed': result.returncode == 0,
              'CompilerCommand': command, 'ExitCode': result.returncode, 'ElapsedSeconds': time.time() - started,
              'Source': identity, 'Inputs': inputs, 'Executable': pin(executable) if executable.exists() else None,
              'UpstreamCompiledOrModified': False, 'ModelOrDeviceValidated': False}
    save(out / 'build.json', report)
    return 0 if report['Passed'] else 1


def prepare(args):
    from PIL import Image
    out = fresh(args.output)
    build_record = json.loads(Path(args.build_record).read_text(encoding='utf-8'))
    if not build_record.get('Passed') or build_record['Schema'] != 'qwen-mtmd-probe-build.v1':
        raise ValueError('Require successful owned-probe build record')
    verify_pins(build_record['Inputs'] + [build_record['Executable']])
    source_identity(build_record['Source']['Path'])
    control_path = Path(args.control_plan).resolve()
    control = json.loads(control_path.read_text(encoding='utf-8'))
    if control['SourceRevision'] != EXPECTED_REVISION or control['SourceStatus']:
        raise ValueError('Unqualified baseline source identity')
    image = Path(args.image).resolve()
    if sha(image) != EXPECTED_IMAGE or control['ImageSha256'] != EXPECTED_IMAGE:
        raise ValueError('Require exact original opaque card-0.png')
    with Image.open(image) as original:
        rgba = original.convert('RGBA')
        if rgba.getchannel('A').getextrema() != (255, 255):
            raise ValueError('This control requires opaque RGB; alpha composition is not inferred')
        width, height = original.size
        rgb = original.convert('RGB').tobytes()
    rgb_path = out / 'card.rgb'; rgb_path.write_bytes(rgb)
    prompt_path = Path(args.prompt).resolve()
    prompt = prompt_path.read_bytes()
    if prompt.count(b'<__media__>') != 1 or prompt.startswith(b'\xef\xbb\xbf') or b'\r' in prompt:
        raise ValueError('Require canonical BOM-free LF native prompt with one marker')
    local_prompt = out / 'prompt.txt'; local_prompt.write_bytes(prompt)
    request = next(row for row in control['Requests'] if row['Turn'] == args.turn)
    expected = request['ExpectedRawInputIds'] + [271]  # Preserve the official final double newline.
    if len(expected) != (318 if args.turn == 1 else 354):
        raise ValueError('Unexpected official generation-boundary token geometry')
    baseline_bin = Path(control['Command'][0]).parent
    dlls = sorted(baseline_bin.glob('*.dll'))
    if not all((baseline_bin / name).is_file() for name in ['mtmd.dll', 'llama.dll', 'ggml-base.dll', 'ggml-cpu.dll', 'ggml-cuda.dll']):
        raise ValueError('Incomplete existing native DLL closure')
    model_files = [metadata(item['Path']) for item in control['ModelMetadataOnly']]
    if model_files != control['ModelMetadataOnly']:
        raise ValueError('Original model metadata changed')
    model = control['Command'][control['Command'].index('--model') + 1]
    projector = control['Command'][control['Command'].index('--mmproj') + 1]
    plan = {'Schema': 'qwen-mtmd-embedding-plan.v2', 'Source': build_record['Source'],
            'Executable': build_record['Executable'], 'ToolAndBuildInputs': build_record['Inputs'],
            'Inputs': [pin(image), pin(local_prompt), pin(rgb_path), pin(control_path), pin(Path(args.build_record))],
            'NativeClosure': [pin(path) for path in dlls], 'ModelMetadataOnly': model_files,
            'Model': model, 'Projector': projector, 'RuntimeDirectory': str(baseline_bin),
            'Rgb': str(rgb_path), 'Prompt': str(local_prompt), 'Width': width, 'Height': height,
            'Rows': 300, 'Dimension': 2560, 'ExpectedExpandedIds': expected, 'ImageToken': 248056,
            'Threads': 8, 'Gpu': bool(args.gpu), 'TimeoutSeconds': 180,
            'LanguageModelLoadPolicy': {'VocabularyOnly': False, 'NoAlloc': True, 'LoadMode': 'none', 'LazyMode': 'off'},
            'Limitations': ['Vision projection diagnostic, no language-model execution or generation quality qualification.',
                            'Model identity is path/size/mtime, not a new hash of the 74.5 GB checkpoint.',
                            'Flash attention explicitly matches the existing llama control; numerical output is not declared ground truth.',
                            'DLL and input hashes happen outside the encode interval; timings are diagnostic only.']}
    save(out / 'plan.json', plan)
    return 0


def verify_model(plan):
    if [metadata(item['Path']) for item in plan['ModelMetadataOnly']] != plan['ModelMetadataOnly']:
        raise ValueError('Original model metadata changed')


def inspect_payload(output, plan):
    result = json.loads((output / 'native-result.json').read_text(encoding='utf-8'))
    if (result['Schema'] != 'qwen-mtmd-native-capture.v2' or not result['Completed']
            or result['VocabularyOnly'] or not result['ModelNoAlloc'] or result['ModelLoadMode'] != 'none'
            or result['ModelLazyMode'] != 'off' or result['LanguageModelExecuted']):
        raise ValueError('Native capture did not complete with metadata-only no-alloc LM')
    if (result['ImageRows'] != plan['Rows'] or result['Dimension'] != plan['Dimension']
            or result['GpuRequested'] != plan['Gpu'] or not result['FlashAttentionRequested']
            or result['Warmup'] or not result['AllFinite'] or result['PerformanceQualified']
            or result['AnswerQualityQualified']):
        raise ValueError('Native capture policy/geometry differs from the plan')
    ids = json.loads((output / 'expanded-ids.json').read_text())
    positions = json.loads((output / 'positions-thw.json').read_text())
    image_positions = json.loads((output / 'image-positions-thw.json').read_text())
    if ids != plan['ExpectedExpandedIds'] or len(positions) != 3 * len(ids):
        raise ValueError('Actual expanded native IDs/position geometry differs from repaired TS input')
    if any(type(value) is not int or not 0 <= value <= 65536 for value in positions):
        raise ValueError('Invalid absolute MRoPE coordinates')
    if len(image_positions) != 3 * plan['Rows']:
        raise ValueError('Image position geometry mismatch')
    start = result['ImageStartToken']
    if ids[start:start + plan['Rows']] != [plan['ImageToken']] * plan['Rows']:
        raise ValueError('Image placeholder span mismatch')
    if positions[3 * start:3 * (start + plan['Rows'])] != image_positions:
        raise ValueError('Full/image position slice mismatch')
    payload = (output / 'projected.f32').read_bytes()
    if len(payload) != plan['Rows'] * plan['Dimension'] * 4:
        raise ValueError('Projected F32 byte count mismatch')
    import math
    if not all(math.isfinite(value[0]) for value in struct.iter_unpack('<f', payload)):
        raise ValueError('Nonfinite projected F32 payload')
    return {'ExactExpandedInputIds': True, 'Projection': pin(output / 'projected.f32'),
            'Positions': pin(output / 'positions-thw.json'), 'Native': result}


def run(args):
    out = fresh(args.output); plan_path = Path(args.plan).resolve()
    plan = json.loads(plan_path.read_text(encoding='utf-8'))
    if plan['Schema'] != 'qwen-mtmd-embedding-plan.v2':
        raise ValueError('Unknown plan schema')
    report = {'Schema': 'qwen-mtmd-embedding-run.v2', 'Completed': False,
              'QualityPassed': None, 'PerformanceQualified': False, 'Plan': pin(plan_path)}
    command = [plan['Executable']['Path'], '--model', plan['Model'], '--mmproj', plan['Projector'],
               '--rgb', plan['Rgb'], '--width', str(plan['Width']), '--height', str(plan['Height']),
               '--prompt', plan['Prompt'], '--output', str(out / 'capture'), '--runtime-dir', plan['RuntimeDirectory'],
               '--rows', str(plan['Rows']), '--dimension', str(plan['Dimension']),
               '--prompt-tokens', str(len(plan['ExpectedExpandedIds'])), '--image-token', str(plan['ImageToken']),
               '--threads', str(plan['Threads']), '--gpu', '1' if plan['Gpu'] else '0']
    all_pins = [plan['Executable'], *plan['ToolAndBuildInputs'], *plan['Inputs'], *plan['NativeClosure']]
    try:
        verify_pins(all_pins); verify_model(plan); source_identity(plan['Source']['Path'])
        overrides = [key for key, value in os.environ.items() if value and
                     key.startswith(('TS_', 'TENSORSHARP_', 'GGML_', 'LLAMA_'))]
        if overrides:
            raise ValueError('Inherited numerical/runtime overrides: ' + ', '.join(overrides))
        env = os.environ.copy(); env['PATH'] = plan['RuntimeDirectory'] + os.pathsep + env.get('PATH', '')
        env['LLAMA_MEDIA_MARKER'] = '<__media__>'
        report['Command'] = command
        with (out / 'stdout.log').open('wb') as stdout, (out / 'stderr.log').open('wb') as stderr:
            with subprocess.Popen(command, cwd=out, env=env, stdout=stdout, stderr=stderr) as child:
                report['OwnedPid'] = child.pid
                try:
                    report['ExitCode'] = child.wait(timeout=plan['TimeoutSeconds'])
                except subprocess.TimeoutExpired:
                    child.kill(); child.wait(); report['TimedOut'] = True; raise
                finally:
                    report['OwnedChildExited'] = child.poll() is not None
        if report['ExitCode'] != 0:
            raise ValueError('Native capture failed; inspect preserved stderr.log')
        report['Capture'] = inspect_payload(out / 'capture', plan)
        verify_pins(all_pins); verify_model(plan); source_identity(plan['Source']['Path'])
        report['InputsAndNativeClosureUnchanged'] = True; report['Completed'] = True
    except Exception as error:
        report['Error'] = str(error)
    save(out / 'report.json', report)
    return 0 if report['Completed'] else 1


def main():
    parser = argparse.ArgumentParser(description=__doc__); modes = parser.add_subparsers(dest='mode', required=True)
    p = modes.add_parser('build'); p.add_argument('--source', type=Path, required=True); p.add_argument('--baseline-build', type=Path, required=True)
    p.add_argument('--vcvars', type=Path, required=True); p.add_argument('--output', type=Path, required=True); p.set_defaults(action=build)
    p = modes.add_parser('prepare'); p.add_argument('--build-record', type=Path, required=True); p.add_argument('--control-plan', type=Path, required=True)
    p.add_argument('--image', type=Path, required=True); p.add_argument('--prompt', type=Path, required=True)
    p.add_argument('--turn', type=int, choices=(1,2), required=True); p.add_argument('--gpu', action='store_true')
    p.add_argument('--output', type=Path, required=True); p.set_defaults(action=prepare)
    p = modes.add_parser('run'); p.add_argument('--plan', type=Path, required=True); p.add_argument('--output', type=Path, required=True); p.set_defaults(action=run)
    args = parser.parse_args(); return args.action(args)


if __name__ == '__main__':
    sys.exit(main())
