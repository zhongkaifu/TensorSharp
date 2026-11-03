#!/usr/bin/env python3
"""Read-only bounded NInfer inventory. Validates v2 structure and NVFP4 divisors.

v3/VI are identified but their schemas are deliberately UNSUPPORTED. No GPU,
conversion, full-payload hash, quantization or inference support is implied.
"""
import argparse
import collections
import hashlib
import json
import math
import os
from pathlib import Path
import re
import struct
import stat
import sys

VERSION = 'tensorsharp-ninfer-metadata-inspection/1'
U64_MAX = (1 << 64) - 1
MAGICS = {b'NINFER\0\x02': 'NINFER_V2', b'NINFER\0\x03': 'NINFER_V3',
          b'VELOXVI\x01': 'VELOX_VI_V1', b'NINFER\0\x01': 'NINFER_V1'}
SOURCE = {'repository': 'https://github.com/gpillon/ignis',
          'revision': 'e600a7d9e0d5921d767a6cb464373583b3cdcaf4',
          'path': 'crates/artifact/src/lib.rs',
          'captured_source_sha256': 'e264fe3522e4fa07b21f66bf31e9877987396b59c55ae71e7a639b79ad434a73',
          'scope': 'Own Python implementation of published v2 header/schema/registry/geometry; extra positive-finite divisor gate; no copied kernel'}
DIRECT = {'BF16': 2, 'FP32': 4, 'I32': 4, 'I64': 8}
QUANT = {'Q4G64_F16S': (64, 32, 0), 'Q5G64_F16S': (64, 32, 8),
         'Q6G64_F16S': (64, 32, 16), 'W8G32_F16S': (32, 32, 0)}
TRELLIS = {'TRELLIS_MUL1_K2', 'TRELLIS_MUL1_K2P5', 'TRELLIS_MUL1_K3',
           'TRELLIS_MUL1_K4', 'TRELLIS_MUL1_K5', 'TRELLIS_MUL1_K6'}
FORMATS = set(DIRECT) | set(QUANT) | TRELLIS | {'NVFP4', 'FP8_E4M3FN_ROW_BF16S', 'Q4G32_F16S'}
LAYOUTS = {'contiguous-le-v1', 'row-split-k128-v1', 'blockscale-k16-m128x4-v1',
           'row-scale-v1', 'trellis-tile16-v1', 'row-interleaved-v1'}

class Refusal(ValueError):
    def __init__(self, message, status='INVALID_V2_STRUCTURE'):
        super().__init__(message)
        self.status = status

def require(value, message):
    if not value:
        raise Refusal(message)

def uint(value, label, positive=False):
    require(type(value) is int and 0 <= value <= U64_MAX and (not positive or value > 0),
            label + ' must be a bounded unsigned integer' + (' >0' if positive else ''))
    return value

def add(a, b, label):
    uint(a, label); uint(b, label)
    require(a <= U64_MAX - b, label + ' overflows u64')
    return a + b

def mul(a, b, label):
    uint(a, label); uint(b, label)
    require(a == 0 or b <= U64_MAX // a, label + ' overflows u64')
    return a * b

def align(value, alignment, label):
    return add(value, alignment - 1, label) // alignment * alignment

def members(value, fields, label):
    require(isinstance(value, dict) and set(value) == set(fields), label + ' missing/extra members')

def string(value, label):
    require(isinstance(value, str) and value != '', label + ' must be a nonempty string')
    return value

def strict_json(data):
    def pairs(rows):
        value = {}
        for k, v in rows:
            require(k not in value, 'duplicate JSON member: ' + k)
            value[k] = v
        return value
    def constant(value):
        raise Refusal('nonfinite JSON constant: ' + value)
    try:
        return json.loads(data, object_pairs_hook=pairs, parse_constant=constant)
    except (UnicodeError, json.JSONDecodeError, RecursionError) as error:
        raise Refusal('invalid UTF8/JSON directory: ' + str(error)) from error

def geometry(fmt, layout, shape):
    """Exact supported v2 size rules; unsupported known layouts never get guessed."""
    require(fmt in FORMATS, 'unknown v2 format: ' + str(fmt))
    require(layout in LAYOUTS, 'unknown v2 layout: ' + str(layout))
    require(isinstance(shape, list) and len(shape) <= 16, 'shape must have rank0..16')
    for d in shape:
        uint(d, 'shape dimension', True)
    if layout == 'contiguous-le-v1':
        require(fmt in DIRECT, 'contiguous layout requires BF16/FP32/I32/I64')
        elements = 1
        for d in shape:
            elements = mul(elements, d, 'element count')
        return {'validated': True, 'alignment': 256, 'expected_bytes': mul(elements, DIRECT[fmt], 'direct bytes')}
    require(len(shape) == 2, layout + ' requires positive rank2 [N,K]')
    n, k = shape
    if layout == 'row-split-k128-v1':
        require(fmt in QUANT, 'row split requires Q4/Q5/Q6G64 or W8G32')
        group, low, high = QUANT[fmt]
        kp = align(k, 128, 'padded K')
        groups = mul(n, kp // group, 'physical groups')
        low_bytes, high_bytes = mul(groups, low, 'low plane'), mul(groups, high, 'high plane')
        high_offset = align(low_bytes, 256, 'high offset')
        scale_offset = add(high_offset, align(high_bytes, 256, 'high plane aligned'), 'scale offset')
        size = add(scale_offset, mul(groups, 2, 'scale bytes'), 'row split size')
        return dict(validated=True, alignment=256, expected_bytes=size, padded_K=kp,
                    group_size=group, low_bytes=low_bytes, high_offset=high_offset,
                    high_bytes=high_bytes, scale_offset=scale_offset, scale_bytes=groups * 2)
    if layout == 'blockscale-k16-m128x4-v1':
        require(fmt == 'NVFP4' and n % 128 == 0 and k % 64 == 0,
                'NVFP4 requires NVFP4 format, N%128=0 and K%64=0')
        elements = mul(n, k, 'NVFP4 elements')
        codes = elements // 2
        scales = elements // 16
        scale_offset = align(codes, 256, 'NVFP4 scale offset')
        divisor_offset = add(scale_offset, scales, 'NVFP4 divisor offset')
        return dict(validated=True, alignment=256, expected_bytes=add(divisor_offset, 4, 'NVFP4 size'),
                    code_bytes=codes, scale_offset=scale_offset, scale_bytes=scales,
                    divisor_offset=divisor_offset)
    if layout == 'row-scale-v1':
        require(fmt == 'FP8_E4M3FN_ROW_BF16S', 'row scale requires FP8_E4M3FN_ROW_BF16S')
        codes = mul(n, k, 'FP8 codes')
        scale_offset = align(codes, 256, 'FP8 scale offset')
        scales = mul(n, 2, 'FP8 scale bytes')
        return dict(validated=True, alignment=256, expected_bytes=add(scale_offset, scales, 'FP8 size'),
                    code_bytes=codes, scale_offset=scale_offset, scale_bytes=scales)
    require((layout == 'trellis-tile16-v1' and fmt in TRELLIS) or
            (layout == 'row-interleaved-v1' and fmt == 'Q4G32_F16S'), 'incompatible known unsupported format/layout')
    return dict(validated=False, alignment=4096, expected_bytes=None,
                reason='Known geometry deliberately unsupported; bounded opaque inventory only, not importable')

def validate_directory(directory, file_bytes, payload_start, read_at=None, max_objects=100000):
    """Pure metadata validator. Missing read_at explicitly leaves divisors unverified."""
    uint(file_bytes, 'file bytes'); uint(payload_start, 'payload start')
    require(payload_start <= file_bytes and payload_start % 4096 == 0, 'invalid payload start')
    members(directory, ('identity', 'objects'), 'directory root')
    members(directory['identity'], ('model_id', 'weights_id'), 'identity')
    for key in ('model_id', 'weights_id'):
        string(directory['identity'][key], key)
    objects = directory['objects']
    require(isinstance(objects, list) and objects, 'objects must be nonempty array')
    if len(objects) > max_objects:
        raise Refusal('object count exceeds inspection limit', 'INSPECTION_LIMIT')
    cursor = 0; seen = set(); inventory = []; unsupported = []; divisors = 0; pending = 0
    for row in objects:
        require(isinstance(row, dict), 'object must be dictionary')
        kind = row.get('kind')
        if kind == 'tensor':
            members(row, ('name', 'kind', 'shape', 'format', 'layout', 'offset', 'bytes'), 'tensor')
            string(row['format'], 'format'); string(row['layout'], 'layout')
            g = geometry(row['format'], row['layout'], row['shape'])
            if not g['validated']:
                unsupported.append(row['name'])
        elif kind == 'resource':
            members(row, ('name', 'kind', 'encoding', 'offset', 'bytes'), 'resource')
            require(row['encoding'] == 'raw-bytes-v1', 'unknown v2 resource encoding')
            g = dict(validated=True, alignment=1, expected_bytes=row['bytes'])
        else:
            raise Refusal('unknown object kind')
        name = string(row['name'], 'object name')
        require(name not in seen, 'duplicate object name: ' + name); seen.add(name)
        offset = uint(row['offset'], 'object offset')
        size = uint(row['bytes'], 'object bytes', True)
        require(offset >= cursor, 'object overlap or out of order: ' + name)
        require(offset % g['alignment'] == 0, 'object alignment: ' + name)
        end = add(offset, size, 'object end')
        require(end <= file_bytes - payload_start, 'object extends past EOF: ' + name)
        if g['validated']:
            require(size == g['expected_bytes'], 'encoded tensor bytes mismatch: ' + name)
        absolute = add(payload_start, offset, 'absolute object offset')
        item = dict(row, absolute_offset=absolute, geometry=g,
                    geometry_validation='VALIDATED' if g['validated'] else 'UNSUPPORTED_GEOMETRY',
                    payload_content_validation='NOT_READ')
        if kind == 'tensor' and row['format'] == 'NVFP4':
            if read_at is None:
                item['divisor_validation'] = 'NOT_READ_METADATA_ONLY'; pending += 1
            else:
                position = add(absolute, g['divisor_offset'], 'absolute divisor offset')
                raw = read_at(position, 4)
                require(len(raw) == 4, 'truncated divisor read')
                value = struct.unpack('<f', raw)[0]
                require(math.isfinite(value) and value > 0, 'non-positive/nonfinite NVFP4 divisor: ' + name)
                item.update(divisor_validation='POSITIVE_FINITE', weight_divisor=value,
                            divisor_raw_hex=raw.hex(), payload_content_validation='FOUR_DIVISOR_BYTES_ONLY')
                divisors += 1
        inventory.append(item); cursor = end
    return dict(identity=directory['identity'], objects=inventory, object_count=len(inventory),
                tensor_count=sum(r['kind'] == 'tensor' for r in inventory),
                resource_count=sum(r['kind'] == 'resource' for r in inventory),
                format_counts=dict(collections.Counter(r['format'] for r in inventory if r['kind'] == 'tensor')),
                unsupported_geometry_names=unsupported, divisor_count_validated=divisors,
                divisor_count_not_read=pending, trailing_unclaimed_payload_bytes=file_bytes - payload_start - cursor,
                structure_and_geometry_validated=not unsupported,
                structure_geometry_and_divisors_validated=not unsupported and pending == 0)

def snapshot(st):
    return {k: getattr(st, k) for k in ('st_dev', 'st_ino', 'st_size', 'st_mtime_ns', 'st_ctime_ns')}

def inspect(path, max_json_bytes=16 << 20, max_objects=100000, known_sha256=None):
    require(type(max_json_bytes) is int and max_json_bytes > 0 and type(max_objects) is int and max_objects > 0,
            'positive inspection limits required')
    require(known_sha256 is None or re.fullmatch('[0-9a-fA-F]{64}', known_sha256) is not None,
            'known SHA must be exactly64 hexadecimal characters')
    requested = Path(path).absolute(); resolved = requested.resolve(strict=True)
    require(stat.S_ISREG(resolved.stat().st_mode), 'artifact must be a regular file')
    result = dict(schema=VERSION, requested_path=str(requested), resolved_path=str(resolved),
                  inference_support='NOT_IMPLEMENTED_BY_INSPECTOR', import_compatible=False,
                  full_weight_payload_read=False, full_artifact_sha256_recomputed=False,
                  operator_supplied_full_sha256=known_sha256, operator_SHA_status='UNVERIFIED_BY_THIS_TOOL',
                  geometry_source=SOURCE)
    with resolved.open('rb') as f:
        opened = os.fstat(f.fileno())
        require(stat.S_ISREG(opened.st_mode), 'opened artifact must be a regular file')
        pre = snapshot(opened); reads = dict(calls=0, bytes=0, divisor_reads=0)
        def read_at(offset, size):
            require(type(size) is int and size >= 0 and add(offset, size, 'read range') <= pre['st_size'], 'read beyond EOF')
            f.seek(offset); raw = f.read(size); reads['calls'] += 1; reads['bytes'] += len(raw)
            require(len(raw) == size, 'short file read')
            return raw
        require(pre['st_size'] >= 8, 'file shorter than magic')
        magic = read_at(0, 8); fmt = MAGICS.get(magic)
        result.update(format=fmt or 'UNKNOWN', magic_hex=magic.hex(), file_bytes=pre['st_size'], file_stat_pre=pre)
        if fmt != 'NINFER_V2':
            result.update(inspection_status='UNSUPPORTED_SCHEMA' if fmt else 'UNSUPPORTED_MAGIC',
                          schema_validated=False, geometry_validation='NOT_RUN',
                          reason='Only NInfer v2 schema is implemented; recognized magic is not validation')
        else:
            require(pre['st_size'] >= 16, 'file shorter than v2 prefix')
            suffix = read_at(8, 8); length = struct.unpack('<Q', suffix)[0]
            uint(length, 'JSON length', True)
            end = add(16, length, 'metadata end'); start = align(end, 4096, 'payload offset')
            require(end <= pre['st_size'] and start <= pre['st_size'], 'directory/payload start beyond EOF')
            if length > max_json_bytes:
                raise Refusal('JSON exceeds bounded inspection limit', 'INSPECTION_LIMIT')
            data = read_at(16, length); directory = strict_json(data)
            def divisor_read(offset, size):
                reads['divisor_reads'] += 1
                return read_at(offset, size)
            inventory = validate_directory(directory, pre['st_size'], start, divisor_read, max_objects)
            result.update(schema_validated=True, metadata_bytes=length, payload_start=start,
                          prefix_sha256=hashlib.sha256(magic + suffix).hexdigest(),
                          directory_sha256=hashlib.sha256(data).hexdigest(), inventory=inventory,
                          inspection_status='VALIDATED_V2_STRUCTURE_AND_DIVISORS' if inventory['structure_geometry_and_divisors_validated'] else 'UNSUPPORTED_TENSOR_GEOMETRY',
                          scale_code_values_validation='NOT_READ; only geometry and per-NVFP4 F32divisors checked')
        post = snapshot(os.fstat(f.fileno()))
        require(requested.resolve(strict=True) == resolved and pre == post == snapshot(resolved.stat()),
                'file/path changed during inspection')
        result.update(file_stat_post=post, identities_unchanged=True, read_statistics=reads)
    return result

def write_output(path, result):
    p = Path(path).absolute()
    require(str(p) == os.path.normpath(str(p)), 'normalized output required')
    require('artifacts' in p.parts or any(p.parts[i:i+2] == ('docs', 'validation') for i in range(len(p.parts)-1)),
            'output must be in ignored artifacts/ or docs/validation/')
    require(not p.exists() and not p.is_symlink() and all(not a.is_symlink() for a in p.parents), 'fresh nonalias output required')
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open('x') as f:
        f.write(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False) + '\n')

def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('artifact', type=Path); p.add_argument('--output', type=Path)
    p.add_argument('--known-sha256', help='Operator evidence only; NEVER represented as recomputed')
    p.add_argument('--max-json-bytes', type=int, default=16 << 20); p.add_argument('--max-objects', type=int, default=100000)
    a = p.parse_args(argv)
    try:
        result = inspect(a.artifact, a.max_json_bytes, a.max_objects, a.known_sha256)
        code = 0 if result['inspection_status'] == 'VALIDATED_V2_STRUCTURE_AND_DIVISORS' else 2
    except (OSError, ValueError, TypeError, OverflowError) as error:
        result = dict(schema=VERSION, requested_path=str(a.artifact), inspection_status=getattr(error, 'status', 'INSPECTION_ERROR'),
                      error=str(error), schema_validated=False, import_compatible=False, inference_support='NOT_IMPLEMENTED_BY_INSPECTOR')
        code = 1
    if a.output:
        write_output(a.output, result)
    else:
        print(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False))
    return code

if __name__ == '__main__':
    raise SystemExit(main())
