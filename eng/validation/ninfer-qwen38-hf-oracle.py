#!/usr/bin/env python3
"""Offline HF tokenizer oracle for a complete, preserved NinferQwen38ImportProbe export.

PLAN by default. No downloads, model calls, CUDA, weights reads or source changes.
"""
import argparse
import base64
import hashlib
import importlib
import json
import os
import platform
import re
import struct
import sys
from dataclasses import dataclass
from pathlib import Path

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
RESOURCE_PINS = {
    'tokenizer.json': (12809320, '0997f410c57a1f4e53b09e4be8f4a172d90edd9564368fb0847030937229b9f3'),
    'tokenizer_config.json': (17928, 'b11349aafa7cdc6a320767cf7ceb29ed82f7eda5d65e8e0819e76f0ce947bf27'),
    'chat_template.jinja': (8952, 'c3cf9e34abf4f9e36c2d72165aa9c132d3e2a725b6c2586aaa3a8af9d7a81041'),
    'generation_config.json': (202, 'e70c136c1b78ddc1fb0905bac8e733a4dc448d4f852a5dd75143fffc70be550e'),
    'preprocessor_config.json': (390, '27225450ac9c6529872ee1924fcb0962ff5634834f817040f444118116f4e516'),
    'video_preprocessor_config.json': (385, '7768af27c1fafa9cc9011c1dc20067e03f8915e03b63504550e11d5066986d13'),
}
PROBE_FILES = ('result.json', 'tokenizer-export.json', 'tokenizer-cases.input.json',
               'tokenizer-cases.output.json', 'token-bytes.jsonl', 'single-byte-token-ids.json')
SHA = re.compile(r'[a-f0-9]{64}\Z')


@dataclass(frozen=True)
class Profile:
    base: int = 248044
    vocab: int = 248077
    head: int = 248320
    cases: int = 22
    directory_sha: str = '222b7540e961a6455fff9f03cce451860e4d248785a4ef0a805f538c8eb82658'
    cases_sha: str = 'f7bf984d997c0daebeaf2be6be20d4ec2b7c695d29b23358a89baa125678b292'


def require(value, message):
    if not value:
        raise ValueError(message)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def strict_object(pairs):
    result = {}
    for key, value in pairs:
        require(key not in result, 'Duplicate JSON property: ' + key)
        result[key] = value
    return result


def parse(text):
    return json.loads(text, object_pairs_hook=strict_object,
                      parse_constant=lambda value: (_ for _ in ()).throw(ValueError('Nonfinite JSON ' + value)))


def read(path):
    return parse(Path(path).read_text(encoding='utf-8'))


def physical(path):
    path = Path(os.path.abspath(path))
    require(path == path.resolve() and not path.is_symlink(), 'Nonaliased physical input required: ' + str(path))
    return path


def identities(paths):
    result = {}
    for path in paths:
        path = physical(path)
        require(path.is_file(), 'Regular input missing: ' + str(path))
        result[str(path)] = dict(bytes=path.stat().st_size, sha256=sha(path))
    return result


def raw64(value):
    require(type(value) is str, 'Base64 string required')
    return base64.b64decode(value, validate=True)


def ids(value, size):
    require(type(value) is list and all(type(x) is int and 0 <= x < size for x in value), 'Typed in-vocabulary ID array required')
    return value


def byte_inverse():
    # Independent specification: GPT2 preserves these byte ranges, then assigns
    # remaining bytes to U+0100 onward in increasing byte order, including zero.
    preserved = list(range(33, 127)) + list(range(161, 173)) + list(range(174, 256))
    result = {chr(b): b for b in preserved}
    extra = 256
    for b in range(256):
        if b not in preserved:
            result[chr(extra)] = b
            extra += 1
    require(len(result) == 256 and result['\u0100'] == 0, 'Internal byte specification incomplete')
    return result


def evaluate(probe, resources, cases_file, expected_result_sha, hf_factory,
             profile=Profile(), resource_pins=RESOURCE_PINS):
    """hf_factory is the actual library in execute; tests inject a labelled CPU mock."""
    probe, resources = physical(probe), physical(resources)
    case_path = physical(cases_file)
    paths = [probe / name for name in PROBE_FILES] + [resources / name for name in resource_pins] + [case_path]
    before = identities(paths)
    require(SHA.fullmatch(expected_result_sha or '') and sha(probe / 'result.json') == expected_result_sha, 'Explicit probe-result SHA differs')
    result, export = read(probe / 'result.json'), read(probe / 'tokenizer-export.json')
    require(result.get('schema') == 'ninfer-qwen38-import-format-probe/1' and result.get('complete') is True,
            'Complete actual public-reader import probe required; incomplete exports are not qualified')
    require(result.get('directorySha256') == profile.directory_sha and result.get('fullArtifactSha256Recomputed') is False,
            'Directory identity or original full-SHA scope differs')
    require(set(result.get('execution', {})) == {'tool', 'runtime', 'models'}, 'Actual execution identities missing')
    for identity in result['execution'].values():
        require(type(identity.get('path')) is str and SHA.fullmatch(identity.get('sha256', '')), 'Assembly identity malformed')
    declared_resources = result.get('resources')
    require(type(declared_resources) is list and len(declared_resources) == len(resource_pins), 'Six resource identities missing')
    resource_rows = {}
    for row in declared_resources:
        require(set(row) == {'Name', 'ByteLength', 'Sha256'} and type(row['Name']) is str and row['Name'] not in resource_rows,
                'Malformed/duplicate original resource identity')
        resource_rows[row['Name']] = row
    require(set(resource_rows) == {'frontend/' + name for name in resource_pins}, 'Original resource names differ')
    for name, (length, digest) in resource_pins.items():
        require((resources / name).stat().st_size == length and sha(resources / name) == digest, 'Original resource SHA/size differs: ' + name)
        row = resource_rows['frontend/' + name]
        require(type(row['ByteLength']) is int and (row['ByteLength'], row['Sha256']) == (length, digest), 'Export resource identity differs: ' + name)
    require(result.get('tokenizer') == export, 'Result tokenizer metadata must exactly equal standalone export')
    expected_counts = dict(validVocabulary=profile.vocab, headRows=profile.head, baseVocabulary=profile.base,
                           atomicAdded=33, HF_special_true=21, caseCount=profile.cases)
    for key, value in expected_counts.items():
        require(type(export.get(key)) is int and export[key] == value, 'Typed export count differs: ' + key)
    require(export.get('bosAutomaticallyAdded') is False and export.get('BosTokenId') == -1 and
            export.get('internalNfcRoundtripAndNoBosInjection') is True, 'Tokenizer BOS/NFC export incomplete')
    require(export.get('caseInputSha256') == sha(probe / 'tokenizer-cases.input.json') == sha(case_path) == profile.cases_sha and
            export.get('caseOutputSha256') == sha(probe / 'tokenizer-cases.output.json'), 'Fixed input or output case SHA differs')
    cases, outputs = read(case_path), read(probe / 'tokenizer-cases.output.json')
    require(type(cases) is list and type(outputs) is list and len(cases) == len(outputs) == profile.cases, 'Exact case matrix missing')
    require(all(set(row) == {'id', 'text'} and type(row['id']) is str and row['id'] and type(row['text']) is str for row in cases) and
            len({row['id'] for row in cases}) == profile.cases, 'Unique typed fixed cases required')
    vocabulary = read(resources / 'tokenizer.json')
    base, added = vocabulary['model']['vocab'], vocabulary['added_tokens']
    require(type(base) is dict and len(base) == profile.base and type(added) is list and len(added) == 33,
            'Original vocabulary geometry differs')
    tokens = {}
    for token, index in base.items():
        require(type(index) is int and 0 <= index < profile.base and index not in tokens, 'Duplicate/invalid source vocabulary ID')
        tokens[index] = token
    special = set()
    for ordinal, row in enumerate(added):
        require(type(row['id']) is int and row['id'] == profile.base + ordinal and type(row['content']) is str and type(row['special']) is bool,
                'Typed contiguous added vocabulary differs')
        tokens[row['id']] = row['content']
        if row['special']:
            special.add(row['id'])
    require(set(tokens) == set(range(profile.vocab)) and len(special) == 21, 'Complete vocabulary/special set differs')
    inverse = byte_inverse()
    expected_bytes = {i: (tokens[i].encode('utf-8') if i >= profile.base else bytes(inverse[c] for c in tokens[i])) for i in tokens}
    hf = hf_factory(resources / 'tokenizer.json')
    require(hf.get_vocab_size(with_added_tokens=False) == profile.base and hf.get_vocab_size(with_added_tokens=True) == profile.vocab,
            'HF actual vocabulary counts differ')
    require(hf.get_vocab(with_added_tokens=True) == {token: i for i, token in tokens.items()}, 'HF actual source token/ID map differs')
    checks = []
    def record(name, passed, detail=None):
        require(type(passed) is bool, 'Internal typed check required')
        checks.append(dict(check=name, passed=passed, **({'detail': detail} if detail is not None else {})))
    case_results = []
    for original, row in zip(cases, outputs):
        require(row.get('id') == original['id'] and row.get('text') == original['text'], 'Case identity/order/text differs')
        require(row.get('addSpecialFalse') is True and type(row.get('noBosEosInjection')) is bool and row.get('decodedEqualsNFC') is True,
                'Original execution flags are not typed')
        require(raw64(row.get('utf8Base64')) == original['text'].encode('utf-8'), 'Original UTF8 text bytes differ')
        candidate, candidate_special = ids(row.get('ids'), profile.vocab), ids(row.get('addSpecialTrueIds'), profile.vocab)
        candidate_raw = raw64(row.get('decodedTokenBytesBase64'))
        require(candidate_raw == b''.join(expected_bytes[i] for i in candidate), 'Case AppendTokenBytes inconsistent with original vocabulary')
        require(type(row.get('decoded')) is str and row.get('HF_ID_oracle') == 'NOT_RUN', 'Original decoded output/oracle scope differs')
        hf_false = hf.encode(original['text'], add_special_tokens=False).ids
        hf_true = hf.encode(original['text'], add_special_tokens=True).ids
        decoded = hf.decode(candidate, skip_special_tokens=False)
        values = dict(encode_add_special_false=candidate == hf_false, encode_add_special_true=candidate_special == hf_true,
                      candidate_no_BOS_EOS_injection=row['noBosEosInjection'] is True and candidate == candidate_special,
                      HF_decode_candidate_ids=row['decoded'] == decoded)
        for name, passed in values.items():
            record('case:' + original['id'] + ':' + name, passed)
        case_results.append(dict(id=original['id'], text=original['text'], candidateIds=candidate, HFIds=hf_false,
                                 candidateAddSpecialTrueIds=candidate_special, HFAddSpecialTrueIds=hf_true,
                                 candidateDecoded=row['decoded'], HFDecoded=decoded, checks=values))
    byte_export = export.get('allValidTokenBytes', {})
    require(byte_export.get('file') == 'token-bytes.jsonl' and byte_export.get('rows') == profile.vocab and
            byte_export.get('fileSha256') == sha(probe / 'token-bytes.jsonl'), 'Full token-byte export identity differs')
    framed = hashlib.sha256(); total = 0; mismatches = []; single = {}; count = 0
    with (probe / 'token-bytes.jsonl').open(encoding='utf-8') as stream:
        for index, line in enumerate(stream):
            row = parse(line)
            require(set(row) == {'id', 'byteLength', 'bytesBase64', 'sha256'} and type(row['id']) is int and row['id'] == index < profile.vocab,
                    'Token-byte IDs must be exact, typed, unique, ordered, complete; no head padding IDs')
            raw = raw64(row['bytesBase64'])
            require(type(row['byteLength']) is int and row['byteLength'] == len(raw) and row['sha256'] == hashlib.sha256(raw).hexdigest(),
                    'Token-byte length/SHA differs')
            framed.update(struct.pack('<II', index, len(raw))); framed.update(raw); total += len(raw); count += 1
            if len(raw) == 1:
                require(raw[0] not in single, 'Duplicate single-byte entry')
                single[raw[0]] = index
            if raw != expected_bytes[index]:
                mismatches.append(dict(id=index, candidateBase64=base64.b64encode(raw).decode(), expectedBase64=base64.b64encode(expected_bytes[index]).decode()))
    require(count == profile.vocab and type(byte_export.get('totalTokenBytes')) is int and byte_export['totalTokenBytes'] == total and
            byte_export.get('framedSha256') == framed.hexdigest(), 'Missing rows or aggregate token-byte framing differs')
    record('all_valid_token_Append_bytes', not mismatches, dict(rows=count, mismatches=len(mismatches)))
    byte_map = read(probe / 'single-byte-token-ids.json'); single_meta = export.get('singleByteTokens', {})
    require(single_meta == dict(rows=256, file='single-byte-token-ids.json', sha256=sha(probe / 'single-byte-token-ids.json')) and
            type(byte_map) is list and len(byte_map) == 256 and set(single) == set(range(256)), 'Full 256 byte map missing')
    for index, row in enumerate(byte_map):
        require(set(row) == {'byteValue', 'id'} and type(row['byteValue']) is int and row['byteValue'] == index and
                type(row['id']) is int and row['id'] == single[index], 'Byte map not bound to actual all-token export')
    record('all_256_bytes_including_zero', all(expected_bytes[row['id']] == bytes([row['byteValue']]) for row in byte_map))
    added_results = []
    for row in added:
        index, text = row['id'], row['content']
        hf_encoded = hf.encode(text, add_special_tokens=False).ids
        retained = hf.decode([index], skip_special_tokens=False)
        skipped = hf.decode([index], skip_special_tokens=True)
        values = dict(HF_added_atomic=hf_encoded == [index], HF_added_decode_retained=retained == text,
                      HF_special_policy=skipped == ('' if index in special else text))
        for name, passed in values.items():
            record('added:' + str(index) + ':' + name, passed)
        added_results.append(dict(id=index, special=row['special'], checks=values))
    after = identities(paths)
    require(before == after, 'Original probe/resources/fixed cases changed during CPU oracle')
    failed = [row for row in checks if not row['passed']]
    return dict(schema='ninfer-qwen38-HF-tokenizer-oracle/1', complete=True, oracle_passed=not failed,
                status='PASS' if not failed else 'FAIL', input_files=before, input_files_unchanged=True,
                probe_execution=result['execution'], expected_probe_result_sha256=expected_result_sha,
                directorySha256=result['directorySha256'], HF_special_true=len(special),
                actual_CSharp_special_ID_list='NOT_EXPORTED; count plus source-qualified construction only',
                cases=case_results, added=added_results, checks=checks,
                counts=dict(case_vectors=len(case_results), encode_comparisons=2 * len(case_results),
                            actual_CSharp_decode_comparisons=len(case_results), all_valid_token_byte_rows=count,
                            byte_values=256, atomic_added=33, failed_checks=len(failed)),
                token_byte_mismatches=mismatches[:128], token_byte_mismatches_total=len(mismatches),
                byte_oracle='Original HF token strings + independent GPT2 inverse, not lossy single-token UTF8 decoding.',
                cleanup='Low-level tokenizers decoder; no transformers cleanup layer or unsupported cleanup keyword.',
                model_quality='NOT_RUN', GPU='NOT_RUN', inference_supported='NOT_ESTABLISHED',
                limitations=['Exact HF encode/decode qualification is limited to the22fixed public input vectors, not all possible texts.',
                             'The full248077 matrix qualifies actual AppendTokenBytes only; CSharp Decode was executed/exported for22sequences, not every arbitrary ID sequence.',
                             'Only original valid vocabulary IDs are compared; padded head rows have no words, and no sampler masking policy is invented.',
                             'Probe source/runtime/whole artifact identities remain external controller responsibilities; this oracle binds their preserved result/assembly declarations and resources, not physical VM DLLs or19GB weights.',
                             'No scores, original exports, model/Factory/KV/sampling settings are modified.'])


def output_path(path, protected):
    out = physical(path)
    parts = out.parts
    require('artifacts' in parts or any(parts[i:i + 2] == ('docs', 'validation') for i in range(len(parts) - 1)), 'Ignored output path required')
    require(not out.exists() and not out.is_symlink(), 'Fresh output required')
    require(all(not out.is_relative_to(root) and not root.is_relative_to(out) for root in protected), 'Output overlaps read-only input roots')
    return out


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--execute', action='store_true')
    parser.add_argument('--probe-output', type=Path, required=True)
    parser.add_argument('--resources', type=Path, required=True)
    parser.add_argument('--cases-file', type=Path, default=HERE / 'NinferQwen38ImportProbe/tokenizer-cases.json')
    parser.add_argument('--expect-probe-result-sha')
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args(argv)
    if not args.execute:
        print(json.dumps(dict(plan_only=True, complete=False, oracle_passed=False, probe_output=str(args.probe_output),
                              resources=str(args.resources), fixed_cases=str(args.cases_file),
                              expected_actual=dict(encode_vectors=22, decode_vectors=22, token_byte_rows=248077, byte_values=256, added=33),
                              dependency='tokenizers must already be installed; no install/download performed', GPU='NOT_RUN')))
        return 0
    out = output_path(args.out, [physical(args.probe_output), physical(args.resources), physical(args.cases_file).parent])
    report = dict(schema='ninfer-qwen38-HF-tokenizer-oracle/1', complete=False, oracle_passed=False, status='ERROR', GPU='NOT_RUN', model_quality='NOT_RUN')
    try:
        os.environ.setdefault('TOKENIZERS_PARALLELISM', 'false')
        library = importlib.import_module('tokenizers')
        package = Path(library.__file__).resolve().parent
        package_files = [Path(library.__file__).resolve(), *sorted(package.glob('*.so')), *sorted(package.glob('*.pyd'))]
        require(callable(getattr(library.Tokenizer, 'from_file', None)), 'Installed library lacks required local-file API')
        package_before = identities(package_files)
        report = evaluate(args.probe_output, args.resources, args.cases_file, args.expect_probe_result_sha,
                          lambda file: library.Tokenizer.from_file(str(file)))
        require(package_before == identities(package_files), 'HF library files changed during oracle')
        report['oracle_runtime'] = dict(python=sys.version, executable=str(Path(sys.executable).resolve()),
                                        executable_sha256=sha(Path(sys.executable).resolve()), platform=platform.platform(),
                                        tokenizers_version=library.__version__, package_files=package_before)
    except ModuleNotFoundError as error:
        report.update(status='NOT_RUN_DEPENDENCY_UNAVAILABLE', error=repr(error))
    except Exception as error:
        report.update(complete=False, oracle_passed=False, status='ERROR', error=repr(error))
    report['oracle_tool_sha256'] = sha(__file__)
    out.parent.mkdir(parents=True, exist_ok=True)
    output_path(out, [physical(args.probe_output), physical(args.resources), physical(args.cases_file).parent])
    with out.open('x', encoding='utf-8') as stream:
        stream.write(json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False) + '\n')
    print(json.dumps(dict(status=report['status'], complete=report['complete'], oracle_passed=report['oracle_passed'], output=str(out))))
    return 0 if report['complete'] and report['oracle_passed'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
