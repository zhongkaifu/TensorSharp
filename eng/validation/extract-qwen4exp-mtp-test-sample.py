#!/usr/bin/env python3
"""Extract a tiny, explicitly reduced subspace from an already verified MTP head.

No full model load or full tensor copy. Source identity comes from the preserved
download inspection; each actually read segment has its own digest. The reduced
weights are useful arithmetic fixtures, not an oracle for complete head logits.
"""
import argparse
import hashlib
import json
import struct
from pathlib import Path


def extract(head, inspection):
    tensors = {item['name']: item for item in inspection['tensors']}
    if head.stat().st_size != inspection['whole_file_bytes']:
        raise ValueError('Head size differs from the prior verified inspection')
    hidden, streams, width, rank = 2560, 4, 8, 3
    wide = [c * hidden + i for c in range(streams) for i in range(width)]
    segments = {}
    with head.open('rb') as source:
        def read(offset, count):
            key = (offset, count)
            if key not in segments:
                source.seek(offset)
                raw = source.read(count)
                if len(raw) != count:
                    raise ValueError('Short tensor segment')
                segments[key] = raw
            return segments[key]

        def norm(name, indices):
            tensor = tensors[name]
            if tensor['type'] != 'F32' or len(tensor['shape']) != 1:
                raise ValueError('Expected a one-dimensional F32 gamma')
            return [struct.unpack('<f', read(tensor['offset'] + i * 4, 4))[0] for i in indices]

        def matrix(name, rows, columns):
            tensor = tensors[name]
            if tensor['type'] != 'Q8_0' or len(tensor['shape']) != 2 or tensor['shape'][0] % 32:
                raise ValueError('Expected row-aligned Q8_0 matrix')
            cols = tensor['shape'][0]
            result = []
            for row in rows:
                for column in columns:
                    if not 0 <= row < tensor['shape'][1] or not 0 <= column < cols:
                        raise ValueError('Sample index exceeds tensor shape')
                    block = read(tensor['offset'] + (row * (cols // 32) + column // 32) * 34, 34)
                    scale = struct.unpack_from('<e', block)[0]
                    quant = struct.unpack_from('<b', block, 2 + column % 32)[0]
                    result.append(scale * quant)
            return result

        prefix = 'blk.48.nextn.'
        arrays = {
            'embedding_norm': norm(prefix + 'enorm.weight', range(width)),
            'hidden_norm': norm(prefix + 'hnorm.weight', wide),
            'eh': matrix(prefix + 'eh_proj.weight', range(width), list(range(width)) + [hidden + i for i in range(width)]),
            'head_norm': norm(prefix + 'hc_head_norm.weight', wide),
            'head_down': matrix(prefix + 'hc_head_down.weight', range(rank), wide),
            'head_up': matrix(prefix + 'hc_head_up.weight', wide, range(rank)),
        }
    return {
        'schema_version': 1,
        'scope': 'Selected actual weight subspace with synthetic test activations; not complete-head inference.',
        'original_file_sha256_previously_verified': inspection['whole_file_sha256'],
        'original_header_sha256': inspection['header_sha256'],
        'original_hidden': hidden, 'streams': streams, 'hidden': width, 'rank': rank,
        'epsilon': inspection['metadata']['qwen4exp.attention.layer_norm_rms_epsilon'],
        'layout': 'Matrices flattened as output rows then input columns. Wide stream order is stream-major.',
        'arrays': arrays,
        'segments': [{'offset': off, 'bytes': count, 'sha256': hashlib.sha256(raw).hexdigest()}
                     for (off, count), raw in sorted(segments.items())],
        'unique_bytes_read': sum(len(raw) for raw in segments.values()),
    }


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--head', type=Path, required=True)
    parser.add_argument('--inspection', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    result = extract(args.head, json.loads(args.inspection.read_text()))
    with args.output.open('x', encoding='utf-8') as output:
        json.dump(result, output, indent=2)
        output.write('\n')
    print(json.dumps({'output': str(args.output), 'unique_bytes_read': result['unique_bytes_read']}))
