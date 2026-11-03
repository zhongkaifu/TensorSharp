"""CPU contract tests with an explicitly mocked HF library, not real HF oracle coverage."""
import base64
import copy
import hashlib
import importlib.util
import io
import json
import struct
import sys
import tempfile
import unittest
import unicodedata
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import patch

SOURCE = Path(__file__).with_name('ninfer-qwen38-hf-oracle.py')
SPEC = importlib.util.spec_from_file_location('ninfer_hf_oracle_contract', SOURCE)
O = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = O
SPEC.loader.exec_module(O)
ADDED = ['<|endoftext|>', '<|im_start|>', '<|im_end|>', '<|object_ref_start|>', '<|object_ref_end|>',
         '<|box_start|>', '<|box_end|>', '<|quad_start|>', '<|quad_end|>', '<|vision_start|>', '<|vision_end|>',
         '<|vision_pad|>', '<|image_pad|>', '<|video_pad|>', '<tool_call>', '</tool_call>', '<|fim_prefix|>',
         '<|fim_middle|>', '<|fim_suffix|>', '<|fim_pad|>', '<|repo_name|>', '<|file_sep|>', '<tool_response>',
         '</tool_response>', '<think>', '</think>', '<|audio_start|>', '<|audio_end|>', '<tts_pad>',
         '<tts_text_bos>', '<tts_text_eod>', '<tts_text_bos_single>', '<|audio_pad|>']


class MockHF:
    """Independent tiny alphabet fake for checking oracle control flow only."""
    def __init__(self, fixture):
        self.f = fixture

    def get_vocab_size(self, with_added_tokens=True):
        return 291 if with_added_tokens else 258

    def get_vocab(self, with_added_tokens=True):
        return {text: i for i, text in self.f.tokens.items() if with_added_tokens or i < 258}

    def encode(self, text, add_special_tokens=True):
        if text in ADDED:
            result = [258 + ADDED.index(text)]
        elif text == 'hi':
            result = [256]
        elif text == ' hi':
            result = [257]
        else:
            result = list(unicodedata.normalize('NFC', text).encode('utf-8'))
        return type('Encoding', (), {'ids': result})()

    def decode(self, ids, skip_special_tokens=True):
        return b''.join(self.f.raw[i] for i in ids if not skip_special_tokens or i not in self.f.special).decode('utf-8', 'replace')


class Fixture:
    def __init__(self, root):
        self.root = Path(root).resolve()
        self.probe = self.root / 'probe'; self.probe.mkdir()
        self.resources = self.root / 'resources'; self.resources.mkdir()
        self.fixed = self.root / 'fixed/cases.json'; self.fixed.parent.mkdir()
        def char(b):
            return chr(b + 256 if b <= 32 else b + 162 if 127 <= b <= 160 else 323 if b == 173 else b)
        self.tokens = {b: char(b) for b in range(256)}
        self.tokens.update({256: 'hi', 257: char(32) + 'hi'})
        self.tokens.update({258 + i: text for i, text in enumerate(ADDED)})
        self.raw = {b: bytes([b]) for b in range(256)}
        self.raw.update({256: b'hi', 257: b' hi'})
        self.raw.update({258 + i: text.encode() for i, text in enumerate(ADDED)})
        self.special = {258 + i for i in range(33) if i < 14 or i >= 26}
        source = dict(model={'vocab': {text: i for i, text in self.tokens.items() if i < 258}},
                      added_tokens=[dict(id=258 + i, content=text, special=258 + i in self.special) for i, text in enumerate(ADDED)])
        self.write(self.resources / 'tokenizer.json', source)
        for name in O.RESOURCE_PINS:
            if name != 'tokenizer.json':
                (self.resources / name).write_bytes(b'{}')
        self.pins = {name: ((self.resources / name).stat().st_size, O.sha(self.resources / name)) for name in O.RESOURCE_PINS}
        texts = ['hi', ' hi', '', '\0', 'e\u0301', 'é', '<think>', '<tool_call>'] + [chr(65 + i) for i in range(14)]
        cases = [dict(id='case-' + str(i), text=text) for i, text in enumerate(texts)]
        self.write(self.fixed, cases); (self.probe / 'tokenizer-cases.input.json').write_bytes(self.fixed.read_bytes())
        self.profile = O.Profile(base=258, vocab=291, head=320, directory_sha='d' * 64, cases_sha=O.sha(self.fixed))
        hf = MockHF(self)
        outputs = []
        for case in cases:
            ids = hf.encode(case['text'], False).ids
            raw = b''.join(self.raw[i] for i in ids)
            outputs.append(dict(**case, utf8Base64=base64.b64encode(case['text'].encode()).decode(), ids=ids,
                                addSpecialFalse=True, addSpecialTrueIds=ids, noBosEosInjection=True,
                                decoded=raw.decode('utf-8', 'replace'), decodedTokenBytesBase64=base64.b64encode(raw).decode(),
                                decodedEqualsNFC=True, HF_ID_oracle='NOT_RUN'))
        self.write(self.probe / 'tokenizer-cases.output.json', outputs)
        self.write_token_rows([dict(id=i, byteLength=len(raw), bytesBase64=base64.b64encode(raw).decode(),
                                    sha256=hashlib.sha256(raw).hexdigest()) for i, raw in self.raw.items()])
        self.write(self.probe / 'single-byte-token-ids.json', [dict(byteValue=b, id=b) for b in range(256)])
        self.export = dict(validVocabulary=291, headRows=320, baseVocabulary=258, atomicAdded=33,
                           HF_special_true=21, caseCount=22, bosAutomaticallyAdded=False, BosTokenId=-1,
                           internalNfcRoundtripAndNoBosInjection=True)
        self.result = dict(schema='ninfer-qwen38-import-format-probe/1', complete=True,
                           directorySha256='d' * 64, fullArtifactSha256Recomputed=False,
                           execution={name: dict(path='/declared/' + name, sha256='a' * 64) for name in ('tool', 'models', 'runtime')},
                           resources=[dict(Name='frontend/' + name, ByteLength=size, Sha256=digest) for name, (size, digest) in self.pins.items()])
        self.refresh()

    @staticmethod
    def write(path, value):
        path.write_text(json.dumps(value, ensure_ascii=False) + '\n', encoding='utf-8')

    def write_token_rows(self, rows):
        (self.probe / 'token-bytes.jsonl').write_text(''.join(json.dumps(row) + '\n' for row in rows))

    def refresh(self):
        rows = [json.loads(line) for line in (self.probe / 'token-bytes.jsonl').read_text().splitlines()]
        framed = hashlib.sha256(); total = 0
        for row in rows:
            raw = base64.b64decode(row['bytesBase64']); framed.update(struct.pack('<II', row['id'], len(raw))); framed.update(raw); total += len(raw)
        self.export.update(caseInputSha256=O.sha(self.probe / 'tokenizer-cases.input.json'),
                           caseOutputSha256=O.sha(self.probe / 'tokenizer-cases.output.json'),
                           singleByteTokens=dict(rows=256, file='single-byte-token-ids.json', sha256=O.sha(self.probe / 'single-byte-token-ids.json')),
                           allValidTokenBytes=dict(file='token-bytes.jsonl', fileSha256=O.sha(self.probe / 'token-bytes.jsonl'),
                                                  rows=291, totalTokenBytes=total, framedSha256=framed.hexdigest()))
        self.write(self.probe / 'tokenizer-export.json', self.export)
        self.result['tokenizer'] = copy.deepcopy(self.export)
        self.write(self.probe / 'result.json', self.result)

    def run(self, factory=None):
        return O.evaluate(self.probe, self.resources, self.fixed, O.sha(self.probe / 'result.json'),
                          factory or (lambda _: MockHF(self)), self.profile, self.pins)


class Tests(unittest.TestCase):
    def test_complete_mock_matrix_and_originals_unchanged_no_real_HF_claim(self):
        with tempfile.TemporaryDirectory() as tmp:
            f = Fixture(tmp); result = f.run()
            self.assertTrue(result['complete']); self.assertTrue(result['oracle_passed'])
            self.assertEqual(result['counts']['all_valid_token_byte_rows'], 291)
            self.assertEqual(result['counts']['actual_CSharp_decode_comparisons'], 22)
            self.assertEqual(result['counts']['encode_comparisons'], 44)
            self.assertTrue(result['input_files_unchanged']); self.assertEqual(result['GPU'], 'NOT_RUN')

    def test_same_decoded_text_but_different_BPE_ids_fails(self):
        with tempfile.TemporaryDirectory() as tmp:
            f = Fixture(tmp); path = f.probe / 'tokenizer-cases.output.json'; rows = O.read(path)
            rows[0]['ids'] = rows[0]['addSpecialTrueIds'] = [104, 105]
            f.write(path, rows); f.refresh(); result = f.run()
            self.assertTrue(result['complete']); self.assertFalse(result['oracle_passed'])
            self.assertTrue(result['cases'][0]['checks']['HF_decode_candidate_ids'])
            self.assertFalse(result['cases'][0]['checks']['encode_add_special_false'])

    def test_full_framing_does_not_hide_one_changed_token_byte(self):
        with tempfile.TemporaryDirectory() as tmp:
            f = Fixture(tmp); rows = [json.loads(x) for x in (f.probe / 'token-bytes.jsonl').read_text().splitlines()]
            rows[256].update(bytesBase64=base64.b64encode(b'ha').decode(), sha256=hashlib.sha256(b'ha').hexdigest())
            f.write_token_rows(rows); f.refresh(); result = f.run()
            self.assertFalse(result['oracle_passed']); self.assertEqual(result['token_byte_mismatches_total'], 1)

    def test_missing_duplicate_bool_or_extra_token_ID_never_completes(self):
        for mutation in ('missing', 'duplicate', 'boolean', 'extra'):
            with self.subTest(mutation=mutation), tempfile.TemporaryDirectory() as tmp:
                f = Fixture(tmp); rows = [json.loads(x) for x in (f.probe / 'token-bytes.jsonl').read_text().splitlines()]
                if mutation == 'missing': rows.pop()
                elif mutation == 'duplicate': rows[1]['id'] = 0
                elif mutation == 'boolean': rows[1]['id'] = True
                else: rows.append(dict(rows[-1], id=291))
                f.write_token_rows(rows); f.refresh()
                with self.assertRaises(ValueError): f.run()

    def test_zero_byte_mapping_is_not_silently_skipped(self):
        with tempfile.TemporaryDirectory() as tmp:
            f = Fixture(tmp); rows = O.read(f.probe / 'single-byte-token-ids.json'); rows[0]['id'] = 1
            f.write(f.probe / 'single-byte-token-ids.json', rows); f.refresh()
            with self.assertRaises(ValueError): f.run()

    def test_case_identity_duplicate_missing_or_boolean_flags_rejected(self):
        for mutation in ('duplicate', 'missing', 'boolean'):
            with self.subTest(mutation=mutation), tempfile.TemporaryDirectory() as tmp:
                f = Fixture(tmp); rows = O.read(f.probe / 'tokenizer-cases.output.json')
                if mutation == 'duplicate': rows[1]['id'] = rows[0]['id']
                elif mutation == 'missing': rows.pop()
                else: rows[0]['noBosEosInjection'] = 1
                f.write(f.probe / 'tokenizer-cases.output.json', rows); f.refresh()
                with self.assertRaises(ValueError): f.run()

    def test_incomplete_probe_bad_resource_or_wrong_expected_result_SHA_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            f = Fixture(tmp); f.result['complete'] = False; f.refresh()
            with self.assertRaises(ValueError): f.run()
            f.result['complete'] = True; f.refresh()
            with self.assertRaises(ValueError): O.evaluate(f.probe, f.resources, f.fixed, '0' * 64, lambda _: MockHF(f), f.profile, f.pins)
            (f.resources / 'generation_config.json').write_text('{"changed":true}')
            with self.assertRaises(ValueError): f.run()

    def test_input_change_during_HF_load_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            f = Fixture(tmp)
            def factory(_):
                (f.resources / 'generation_config.json').write_text('changed')
                return MockHF(f)
            with self.assertRaises(ValueError): f.run(factory)

    def test_HF_special_behavior_mismatch_not_count_netted(self):
        with tempfile.TemporaryDirectory() as tmp:
            f = Fixture(tmp)
            class BadHF(MockHF):
                def decode(self, ids, skip_special_tokens=True):
                    if skip_special_tokens and ids == [282]: return ''  # <think> is non-special.
                    return super().decode(ids, skip_special_tokens)
            result = f.run(lambda _: BadHF(f))
            self.assertFalse(result['oracle_passed']); self.assertEqual(result['counts']['failed_checks'], 1)

    def test_strict_duplicate_json_and_nonfinite_number_rejected(self):
        for text in ('{"a":1,"a":2}', '{"a":NaN}'):
            with self.assertRaises(ValueError): O.parse(text)

    def test_PLAN_never_imports_HF_or_writes_output(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp).resolve(); out = root / 'artifacts/plan.json'
            with patch.object(O.importlib, 'import_module', side_effect=AssertionError('must not import')), redirect_stdout(io.StringIO()):
                self.assertEqual(O.main(['--probe-output', str(root / 'missing'), '--resources', str(root / 'missing2'), '--out', str(out)]), 0)
            self.assertFalse(out.exists())

    def test_missing_dependency_is_NOT_RUN_not_PASS(self):
        with tempfile.TemporaryDirectory() as tmp:
            f = Fixture(tmp); out = f.root / 'artifacts/report.json'
            with patch.object(O.importlib, 'import_module', side_effect=ModuleNotFoundError('tokenizers absent')), redirect_stdout(io.StringIO()):
                code = O.main(['--execute', '--probe-output', str(f.probe), '--resources', str(f.resources), '--cases-file', str(f.fixed), '--expect-probe-result-sha', O.sha(f.probe / 'result.json'), '--out', str(out)])
            report = O.read(out); self.assertEqual(code, 1); self.assertFalse(report['complete']); self.assertFalse(report['oracle_passed'])
            self.assertEqual(report['status'], 'NOT_RUN_DEPENDENCY_UNAVAILABLE')

    def test_output_foreign_dangling_or_parent_alias_never_written(self):
        with tempfile.TemporaryDirectory() as tmp:
            f = Fixture(tmp); artifacts = f.root / 'artifacts'; artifacts.mkdir(); foreign = f.root / 'foreign'; foreign.mkdir()
            sentinel = foreign / 'report.json'; sentinel.write_text('preserve')
            (artifacts / 'alias').symlink_to(foreign, target_is_directory=True)
            (artifacts / 'dangling').symlink_to(f.root / 'never-created')
            for out in (sentinel, artifacts / 'alias/report.json', artifacts / 'dangling', f.probe / 'artifacts/report.json'):
                with self.assertRaises(ValueError): O.output_path(out, [f.probe, f.resources])
            self.assertEqual(sentinel.read_text(), 'preserve'); self.assertFalse((f.root / 'never-created').exists())


if __name__ == '__main__':
    unittest.main()
