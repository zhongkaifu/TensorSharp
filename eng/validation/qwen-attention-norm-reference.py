#!/usr/bin/env python3
"""CPU mathematical reference for captured Qwen text Q/K RMSNorm + NEOX RoPE.

Reads GGUF headers and exactly two small F32 norm tensors, never bulk weights.
FP64 and explicitly rounded sequential FP32 are references, not a CUDA emulator:
CUDA reduction order, rsqrtf/powf/trigonometry and contraction may differ.
The diagnostic log supplies tight-packed dump geometry, not the GPU nb strides.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path
import re
import struct


SIZES = {0: 1, 1: 1, 2: 2, 3: 2, 4: 4, 5: 4, 6: 4, 7: 1, 10: 8, 11: 8, 12: 8}
FORMATS = {0: 'B', 1: 'b', 2: 'H', 3: 'h', 4: 'I', 5: 'i', 6: 'f', 7: '?', 10: 'Q', 11: 'q', 12: 'd'}
PREFIX = 'qwen4exp.'
KEYS = {'general.alignment', PREFIX + 'attention.layer_norm_rms_epsilon',
        PREFIX + 'attention.head_count', PREFIX + 'attention.head_count_kv',
        PREFIX + 'attention.key_length', PREFIX + 'rope.dimension_count',
        PREFIX + 'rope.freq_base', PREFIX + 'rope.scaling.factor', PREFIX + 'rope.dimension_sections'}
CAPTURE = re.compile(r'\[q4e-attn-diagnostic\] path=(\S+) call=(\d+) layer=(\d+) pos=(\d+) caller=(\d+) slot=(\d+) tensor=(\S+) type=(\S+) ne=([\d,]+) bytes=(\d+) hash=([0-9a-f]+) file=(.+)$')


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        while block := stream.read(65536):
            digest.update(block)
    return digest.hexdigest()


def identity(path):
    path = Path(path).resolve()
    stat = path.stat()
    return {'path': str(path), 'size': stat.st_size, 'mtime_ns': stat.st_mtime_ns}


class HeaderReader:
    def __init__(self, stream):
        self.stream = stream
        self.size = stream.seek(0, 2)
        stream.seek(0)

    def read(self, count):
        if not 0 <= count <= 1024 * 1024:
            raise ValueError('header read exceeds bounded size')
        value = self.stream.read(count)
        if len(value) != count:
            raise ValueError('truncated GGUF header')
        return value

    def unpack(self, fmt):
        return struct.unpack('<' + fmt, self.read(struct.calcsize('<' + fmt)))

    def skip(self, count):
        if count < 0 or self.stream.tell() + count > self.size:
            raise ValueError('GGUF header extent outside file')
        self.stream.seek(count, 1)

    def string(self):
        return self.read(self.unpack('Q')[0]).decode('utf-8')

    def value(self, kind, keep, depth=0):
        if depth > 4:
            raise ValueError('nested GGUF array')
        if kind in SIZES:
            if keep:
                return self.unpack(FORMATS[kind])[0]
            self.skip(SIZES[kind])
        elif kind == 8:
            count = self.unpack('Q')[0]
            if keep:
                return self.read(count).decode('utf-8')
            self.skip(count)
        elif kind == 9:
            element, count = self.unpack('I')[0], self.unpack('Q')[0]
            if count > 2_000_000 or (keep and count > 1024):
                raise ValueError('GGUF array count exceeds bounded header parser')
            if not keep and element in SIZES:
                self.skip(count * SIZES[element])
            else:
                result = []
                for _ in range(count):
                    value = self.value(element, keep, depth + 1)
                    if keep:
                        result.append(value)
                return result if keep else None
        else:
            raise ValueError('unsupported GGUF metadata type')
        return None


def headers(path, names):
    with Path(path).open('rb') as stream:
        reader = HeaderReader(stream)
        if reader.read(4) != b'GGUF' or reader.unpack('I')[0] != 3:
            raise ValueError('requires GGUF v3')
        tensors, pairs = reader.unpack('QQ')
        if tensors > 100_000 or pairs > 100_000:
            raise ValueError('excessive header entries')
        metadata, found = {}, []
        for _ in range(pairs):
            key, kind = reader.string(), reader.unpack('I')[0]
            value = reader.value(kind, key in KEYS)
            if key in KEYS:
                if key in metadata:
                    raise ValueError('duplicate selected metadata key')
                metadata[key] = value
        for _ in range(tensors):
            name, dimensions = reader.string(), reader.unpack('I')[0]
            if not 1 <= dimensions <= 4:
                raise ValueError('invalid tensor dimensions')
            shape = reader.unpack('Q' * dimensions)
            kind, offset = reader.unpack('I')[0], reader.unpack('Q')[0]
            if name in names:
                found.append({'name': name, 'shape': list(shape), 'type': kind, 'relative_offset': offset})
        alignment = metadata.get('general.alignment', 32)
        if not isinstance(alignment, int) or alignment < 1 or alignment > 65536 or alignment & (alignment - 1):
            raise ValueError('invalid GGUF alignment')
        base = (stream.tell() + alignment - 1) // alignment * alignment
        for tensor in found:
            tensor['absolute_offset'] = base + tensor['relative_offset']
        return metadata, found


def norms(first_shard, layer):
    first_shard = Path(first_shard)
    match = re.fullmatch(r'(.*)-\d{5}-of-(\d{5})\.gguf', str(first_shard))
    count = int(match[2]) if match else 1
    if not 1 <= count <= 64:
        raise ValueError('invalid or excessive GGUF shard count')
    paths = [Path(f'{match[1]}-{i:05d}-of-{count:05d}.gguf')
             for i in range(1, count + 1)] if match else [first_shard]
    names = {f'blk.{layer}.attn_{part}_norm.weight' for part in ('q', 'k')}
    before = [identity(path) for path in paths]
    metadata, descriptors, weights = {}, {}, {}
    for path in paths:
        meta, tensors = headers(path, names)
        for key, value in meta.items():
            if key in metadata and metadata[key] != value:
                raise ValueError('conflicting shard metadata')
            metadata[key] = value
        for tensor in tensors:
            name = tensor['name']
            if name in descriptors:
                raise ValueError('duplicate norm tensor')
            descriptors[name] = {**tensor, 'path': str(path.resolve())}
    dimension = metadata.get(PREFIX + 'attention.key_length')
    if not isinstance(dimension, int) or not 1 <= dimension <= 4096 or set(descriptors) != names:
        raise ValueError('missing or invalid Q/K geometry/tensors')
    for name, descriptor in descriptors.items():
        count = dimension * 4
        if descriptor['type'] != 0 or descriptor['shape'] != [dimension]:
            raise ValueError('norm tensor must be F32[head_dimension]')
        if descriptor['absolute_offset'] + count > Path(descriptor['path']).stat().st_size:
            raise ValueError('norm tensor outside shard')
        with open(descriptor['path'], 'rb') as stream:
            stream.seek(descriptor['absolute_offset'])
            raw = stream.read(count)
        if len(raw) != count:
            raise ValueError('short norm tensor')
        values = struct.unpack('<' + 'f' * dimension, raw)
        if not all(map(math.isfinite, values)):
            raise ValueError('nonfinite norm weight')
        weights[name] = raw
        descriptor.update(bytes=count, sha256=hashlib.sha256(raw).hexdigest())
    if [identity(path) for path in paths] != before:
        raise ValueError('model shard stat changed during read')
    return metadata, descriptors, weights, before


def f32(value):
    return struct.unpack('<f', struct.pack('<f', value))[0]


def transform(values, weights, position, rotary, base, scale, epsilon, mode):
    dimension = len(weights)
    if (not dimension or len(values) % dimension or not 0 < rotary <= dimension or rotary % 2
            or position < 0 or base <= 0 or scale <= 0 or epsilon < 0
            or not all(map(math.isfinite, [*values, *weights, base, scale, epsilon]))):
        raise ValueError('invalid norm/NEOX geometry or nonfinite inputs')
    if mode not in ('fp64', 'fp32_sequential'):
        raise ValueError('unknown arithmetic reference')
    rounder = float if mode == 'fp64' else f32
    theta_scale = rounder(math.pow(base, rounder(-2.0 / rotary)))
    output, normalized = [], []
    for start in range(0, len(values), dimension):
        row = values[start:start + dimension]
        if mode == 'fp64':
            total = math.fsum(x * x for x in row)
        else:
            total = 0.0
            for x in row:
                total = f32(total + f32(x * x))
        inverse = rounder(1.0 / math.sqrt(rounder(rounder(total / dimension) + epsilon)))
        result = [rounder(rounder(inverse * x) * weight) for x, weight in zip(row, weights)]
        normalized.extend(result)
        for i in range(rotary // 2):
            # FP64 uses the mathematical frequency. FP32 preserves the two pow
            # stages in rope.cu, but libm is not CUDA powf/cosf/sinf/rsqrtf.
            frequency = math.pow(base, -2.0 * i / rotary) if mode == 'fp64' else f32(math.pow(theta_scale, float(i)))
            angle = rounder(rounder(position * frequency) * scale)
            cosine, sine = rounder(math.cos(angle)), rounder(math.sin(angle))
            a, b = result[i], result[i + rotary // 2]
            result[i] = rounder(rounder(a * cosine) - rounder(b * sine))
            result[i + rotary // 2] = rounder(rounder(a * sine) + rounder(b * cosine))
        if not all(map(math.isfinite, result)):
            raise ValueError('nonfinite reference result')
        output.extend(result)
    return output, normalized


def half_word(value):
    return struct.unpack('<H', struct.pack('<e', value))[0]


def half_value(word):
    return struct.unpack('<e', struct.pack('<H', word))[0]


def ordered_half(word):
    return 0x8000 - (word & 0x7fff) if word & 0x8000 else 0x8000 + word


def compare(actual_words, reference, dimension, rotary):
    if len(actual_words) != len(reference) or not reference:
        raise ValueError('comparison dimensions differ')
    differences, abs_errors, half_errors, ulps = [], [], [], []
    for index, (word, value) in enumerate(zip(actual_words, reference)):
        actual, rounded = half_value(word), half_word(value)
        if not math.isfinite(actual) or not math.isfinite(value) or not math.isfinite(half_value(rounded)):
            raise ValueError('nonfinite comparison')
        error = abs(actual - value)
        distance = abs(ordered_half(word) - ordered_half(rounded))
        abs_errors.append(error)
        half_errors.append(abs(actual - half_value(rounded)))
        ulps.append(distance)
        if word != rounded:
            differences.append({'head': index // dimension, 'dimension': index % dimension,
                'rotated': index % dimension < rotary, 'actual': actual, 'actual_half_bits': word,
                'reference': value, 'reference_half': half_value(rounded), 'reference_half_bits': rounded,
                'absolute_error_to_unrounded': error, 'half_ulp_distance': distance})
    worst = max(range(len(abs_errors)), key=abs_errors.__getitem__)
    return {'elements': len(reference), 'half_bit_exact': not differences,
        'different_half_elements': len(differences), 'max_half_ulp_distance': max(ulps),
        'max_absolute_half_difference': max(half_errors),
        'max_absolute_error_to_unrounded': max(abs_errors),
        'rms_error_to_unrounded': math.sqrt(math.fsum(x * x for x in abs_errors) / len(abs_errors)),
        'max_error_coordinate': {'head': worst // dimension, 'dimension': worst % dimension},
        'different_coordinates': differences}


def capture_records(log, path, call, layer, position, caller):
    found = {}
    for line in Path(log).read_text(encoding='utf-8-sig', errors='strict').splitlines():
        match = CAPTURE.search(line)
        if not match:
            continue
        p, c, l, pos, who, slot, name, dtype, shape, size, fnv, file = match.groups()
        if (p, int(c), int(l), int(pos), int(who)) != (path, call, layer, position, caller):
            continue
        if name in found:
            raise ValueError('duplicate capture record')
        found[name] = {'path': file, 'type': dtype, 'shape': [int(v) for v in shape.split(',')],
            'size': int(size), 'fnv1a64': fnv, 'slot': int(slot)}
    if not {'q_pre_norm', 'k_pre_norm', 'k_cache'} <= found.keys():
        raise ValueError('missing readable Q/K prenorm or K cache capture')
    return found


def verify_capture(record):
    path = Path(record['path'])
    if path.stat().st_size != record['size'] or record['size'] > 64 * 1024 * 1024:
        raise ValueError('capture length differs or exceeds bounded size')
    digest, fnv = hashlib.sha256(), 1469598103934665603
    with path.open('rb') as stream:
        while block := stream.read(65536):
            digest.update(block)
            for byte in block:
                fnv = ((fnv ^ byte) * 1099511628211) & 0xffffffffffffffff
    if f'{fnv:016x}' != record['fnv1a64']:
        raise ValueError('capture differs from native diagnostic hash')
    return {**record, 'sha256': digest.hexdigest()}


def run(args):
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=False)
    records = capture_records(args.log, args.path, args.call, args.layer, args.position, args.caller)
    captures = {name: verify_capture(records[name]) for name in ('q_pre_norm', 'k_pre_norm', 'k_cache')}
    metadata, descriptors, raw_weights, model_stats = norms(args.model, args.layer)
    dimension = metadata[PREFIX + 'attention.key_length']
    rotary, base = metadata[PREFIX + 'rope.dimension_count'], metadata[PREFIX + 'rope.freq_base']
    scale = f32(1.0 / metadata.get(PREFIX + 'rope.scaling.factor', 1.0))
    epsilon = metadata[PREFIX + 'attention.layer_norm_rms_epsilon']
    kv_heads = metadata[PREFIX + 'attention.head_count_kv']
    for part, heads in [('q', metadata[PREFIX + 'attention.head_count']), ('k', kv_heads)]:
        record = captures[part + '_pre_norm']
        if record['type'] != 'f32' or record['shape'] != [dimension, heads, 1, 1] or record['size'] != dimension * heads * 4:
            raise ValueError('requires tight single-token prenorm capture')
    cache = captures['k_cache']
    if cache['type'] != 'f16' or cache['shape'] != [dimension, args.position + 1, kv_heads, 1] or cache['size'] != math.prod(cache['shape']) * 2:
        raise ValueError('requires tight F16 live-prefix K cache capture')
    actual_raw = bytearray()
    with open(cache['path'], 'rb') as stream:
        for head in range(kv_heads):
            stream.seek((head * (args.position + 1) + args.position) * dimension * 2)
            actual_raw.extend(stream.read(dimension * 2))
    actual_words = struct.unpack('<' + 'H' * (dimension * kv_heads), actual_raw)
    (output / 'k-current-row.f16.bin').write_bytes(actual_raw)
    comparisons, artifacts = {}, [{'path': str((output / 'k-current-row.f16.bin').resolve()),
        'sha256': sha(output / 'k-current-row.f16.bin'), 'type': 'f16', 'shape': [dimension, kv_heads]}]
    for name, raw in raw_weights.items():
        target = output / (name + '.f32.bin')
        target.write_bytes(raw)
        artifacts.append({'path': str(target.resolve()), 'sha256': sha(target)})
    for part in ('q', 'k'):
        raw = Path(captures[part + '_pre_norm']['path']).read_bytes()
        values = struct.unpack('<' + 'f' * (len(raw) // 4), raw)
        weights = struct.unpack('<' + 'f' * dimension, raw_weights[f'blk.{args.layer}.attn_{part}_norm.weight'])
        for mode, fmt in [('fp64', 'd'), ('fp32_sequential', 'f')]:
            transformed, _ = transform(values, weights, args.position, rotary, base, scale, epsilon, mode)
            target = output / f'{part}-{mode}-rope.bin'
            target.write_bytes(struct.pack('<' + fmt * len(transformed), *transformed))
            artifacts.append({'path': str(target.resolve()), 'sha256': sha(target),
                'type': 'f64' if fmt == 'd' else 'f32', 'shape': captures[part + '_pre_norm']['shape']})
            if part == 'k':
                comparisons[mode] = compare(actual_words, transformed, dimension, rotary)
    if [identity(row['path']) for row in model_stats] != model_stats:
        raise ValueError('model shard stat changed during analysis')
    for record in captures.values():
        if sha(record['path']) != record['sha256']:
            raise ValueError('capture changed during analysis')
    report = {'schema_version': 1, 'completed': True, 'model_quality_passed': None,
        'scope': 'CPU mathematical reference from existing prenorm captures; not CUDA bit-exact simulation, whole-model quality or performance qualification.',
        'assumptions': ['Text NEOX mode, all position axes equal to the selected position; no multi-axis image reference.',
            'Checkpoint norm tensors are the production direct weights, without an additive weight offset.',
            'Log hashes authenticate saved dump bytes; the live log hash is a snapshot and may change as the parent run continues.'],
        'arithmetic': {'fp64': 'math.fsum squares, FP64 RMS/weighted norm and mathematical NEOX frequency; final F16 nearest-even comparison',
            'fp32_sequential': 'F32-rounded sequential sum, host sqrt reciprocal and pow/cos/sin rounded to F32, separate rounded multiply/add, no FMA; not CUDA reduction/transcendental emulation'},
        'source': {'path': str(Path(__file__).resolve()), 'sha256': sha(__file__)},
        'capture_log': str(Path(args.log).resolve()), 'capture_log_sha256_snapshot': sha(args.log),
        'selection': {'path': args.path, 'call': args.call, 'layer': args.layer, 'position': args.position, 'caller': args.caller},
        'metadata': metadata, 'model_shard_stats': model_stats,
        'model_identity_scope': 'headers plus two selected norm tensor hashes; no new full model hash',
        'weights': descriptors, 'captures': captures, 'artifacts': artifacts, 'k_current_row': comparisons}
    (output / 'report.json').write_text(json.dumps(report, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', required=True)
    parser.add_argument('--log', required=True)
    parser.add_argument('--output', required=True, help='fresh evidence directory')
    parser.add_argument('--path', choices=['solo', 'arena'], default='solo')
    parser.add_argument('--call', type=int, default=0)
    parser.add_argument('--layer', type=int, default=39)
    parser.add_argument('--position', type=int, default=1501)
    parser.add_argument('--caller', type=int, default=0)
    args = parser.parse_args()
    if min(args.call, args.layer, args.position, args.caller) < 0:
        parser.error('selection values must be nonnegative')
    report = run(args)
    print(json.dumps({'completed': report['completed'], 'k_current_row': {
        name: {key: value for key, value in result.items() if key != 'different_coordinates'}
        for name, result in report['k_current_row'].items()}}, indent=2))


if __name__ == '__main__':
    main()
