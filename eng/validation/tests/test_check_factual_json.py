import copy
import importlib.util
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch


SPEC = importlib.util.spec_from_file_location('check_factual_json_tests',
    Path(__file__).parents[1] / 'check-factual-json.py')
M = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(M)
FIXTURE = 'ff8-credits-en'
RENDERED = M.PREFIX + M.FIXTURES[FIXTURE] + M.SUFFIX
ANSWER = json.dumps(M.EXPECTED)
RAW = '</think>' + ANSWER + '<|user|>'


def pin(path):
    return {'Path': path, 'Bytes': 10, 'Sha256': 'a' * 64}


def tensorsharp():
    native = pin('D:/test/GgmlOps.dll')
    source = [pin('C:/repo/' + name) for name in ('Program.cs', 'ProbePolicy.cs', 'GlmUnifiedSessionProbe.csproj')]
    managed = [pin('D:/test/' + name) for name in ('TensorSharp.Core.dll', 'TensorSharp.Memory.dll',
        'TensorSharp.Models.dll', 'TensorSharp.Runtime.dll', 'TensorSharp.Backends.GGML.dll',
        'TensorSharp.Backends.Cuda.dll', 'TensorSharp.Backends.MLX.dll', 'GlmUnifiedSessionProbe.dll')]
    row = {'Round': 1, 'RequestId': 'request-1', 'RawText': RAW, 'Content': ANSWER, 'ToolCalls': None, 'ThinkingClosed': True,
        'OutputTokens': [1, 2, 9], 'PublishedTokens': [1, 2], 'ComputedTokens': 5,
        'NaturalEos': True, 'StatusBeforeDispose': 'FinishedStopped', 'SemanticPassed': None,
        'Passed': False, 'Status': 'completed-quality-unknown', 'FinalLogitsFinite': None,
        'Completion': {'Status': 'FinishedStopped', 'FinishReason': 'eos', 'PromptTokenCount': 2,
            'OutputTokenCount': 3, 'PrefixCacheReusedTokens': 0}}
    stats = {'C:/model.gguf': {'Path': 'C:/model.gguf', 'Bytes': 10, 'LastWriteTimeUtcTicks': 621355968000000001}}
    report = {'Status': 'completed-quality-unknown', 'Passed': False, 'ExitCode': 1,
        'Arguments': {'task': 'unknown', 'max-new': '96', 'native-sha256': 'a' * 64, 'model': 'C:/model.gguf'},
        'Scheduler': {'Enabled': False, 'EnablePrefixCaching': False},
        'PromptTokens': [101, 102], 'DecodedPrompt': RENDERED, 'CheckpointEosIds': [9],
        'Errors': [], 'SequenceError': None, 'EngineCompletionObserved': True,
        'SourceToolsBefore': source, 'SourceToolsAfter': copy.deepcopy(source), 'SourceToolsUnchanged': True,
        'ManagedBefore': managed, 'ManagedAfter': copy.deepcopy(managed), 'ManagedUnchanged': True,
        'IntendedNative': native, 'IntendedNativeAfter': copy.deepcopy(native), 'NativeUnchanged': True,
        'LoadedNativeAfterLoad': [copy.deepcopy(native)], 'LoadedNativeAfter': [copy.deepcopy(native)],
        'InputFile': pin('C:/tokens.json'), 'InputFileAfter': pin('C:/tokens.json'), 'InputFileUnchanged': True,
        'ModelIdentityLinked': True, 'ModelStatsBefore': stats, 'ModelStatsAfter': copy.deepcopy(stats),
        'ModelStatsUnchanged': True, 'HistoricalModelIdentity': pin('C:/manifest.json'),
        'HistoricalModelIdentityAfter': pin('C:/manifest.json'), 'HistoricalModelIdentityUnchanged': True,
        'HistoricalModelManifest': {'files': [{'path': 'C:/model.gguf', 'bytes': 10, 'mtime_ns': 100, 'sha256': 'b' * 64}]},
        'EngineDisposed': True, 'SessionDisposed': True, 'LedgerZeroAfterCleanup': True,
        'BudgetAfterSessionDisposed': {'Current': [{'Reserved': 0, 'Committed': 0}]},
        'EngineCountersBeforeDispose': {'TotalSubmitted': 1, 'TotalCompleted': 1,
            'RunningCount': 0, 'WaitingCount': 0, 'PendingRequestCount': 0},
        'RequestedRepetitions': 1, 'CompletedRepetitions': 1, 'Rounds': [row],
        'OutputTokensAfterCleanup': [1, 2, 9], 'ComputedTokensAfterCleanup': 5,
        'RawTextAfterCleanup': RAW, 'NaturalEosAfterCleanup': True,
        'SequenceStatusAfterCleanup': 'FinishedStopped', 'FinishReasonAfterCleanup': 'eos'}
    # The prompt array is deliberately not a real checkpoint tokenizer fixture;
    # tests here exercise the checker, never assert model/device acceptance.
    row['ComputedTokens'] = report['ComputedTokensAfterCleanup'] = 5
    return report


class Tokenizer:
    def decode(self, ids):
        if ids == [101, 102]: return RENDERED
        if ids == [1, 2, 9]: return RAW
        if ids == [1, 9, 2, 9]: return RAW
        if ids == [1, 2]: return '</think>' + ANSWER
        raise ValueError('Unexpected fake tokens')

    def encode(self, text, parse_special):
        return [101, 102] if text == RENDERED and parse_special else []


class Parser:
    def __init__(self, thinking, tools, stream_tools): pass
    def feed(self, raw):
        content = raw.removeprefix('</think>')
        return [SimpleNamespace(kind='content', text=content)]
    def finish(self): return []


def maya():
    return {'ExecutionComplete': True, 'ExitCode': 0, 'Errors': [], 'CleanupErrors': [],
        'TimedOut': False, 'Tokens': [101, 102], 'DecodedPrompt': RENDERED,
        'RawOutputIds': [1, 2, 9], 'RawText': RAW, 'RawTokenWallMs': [10, 20, 30],
        'MaxNewTokens': 96, 'Done': {'generated': 3, 'prompt_tokens': 2, 'finish': 'stop',
            'reused': 0, 'drafts_offered': 0, 'drafts_accepted': 0}}


class FactualJsonSemanticTests(unittest.TestCase):
    def test_exact_object_and_ordinary_json_whitespace(self):
        self.assertTrue(M.semantic(ANSWER))
        self.assertTrue(M.semantic(' \n' + json.dumps(M.EXPECTED, indent=2) + '\t'))

    def test_wrong_names_extra_keys_and_wrong_structure_fail(self):
        for answer in ('null', '[]', json.dumps({'director': 'Yoshinori Kitase'}),
                json.dumps(M.EXPECTED | {'producer': 'Shinji Hashimoto'}),
                json.dumps(M.EXPECTED | {'director': 'Hironobu Sakaguchi'}),
                json.dumps(M.EXPECTED | {'director': 'yoshinori kitase'}),
                json.dumps(M.EXPECTED | {'character_designer': ['Tetsuya Nomura']})):
            with self.subTest(answer=answer): self.assertFalse(M.semantic(answer))

    def test_wrapper_embedded_correct_answer_and_duplicate_keys_fail(self):
        for answer in ('```json\n' + ANSWER + '\n```', 'Answer: ' + ANSWER,
                ANSWER + '\nExplanation', ANSWER + ANSWER,
                '{"director":"Wrong","director":"Yoshinori Kitase","character_designer":"Tetsuya Nomura"}',
                '{"director":NaN,"character_designer":"Tetsuya Nomura"}'):
            with self.subTest(answer=answer): self.assertFalse(M.semantic(answer))

    def test_report_parser_rejects_duplicate_and_nonfinite_metadata(self):
        for raw in ('{"Passed":true,"Passed":false}', '{"x":NaN}', '{"x":1e999}'):
            with self.subTest(raw=raw), self.assertRaises(ValueError): M.strict_json(raw)

    def test_thinking_reopened_at_end_is_incomplete(self):
        self.assertTrue(M.thinking_closed(RENDERED, '</think>' + ANSWER))
        self.assertFalse(M.thinking_closed(RENDERED, '</think>' + ANSWER + '<think>'))
        self.assertFalse(M.thinking_closed(RENDERED, ANSWER))

    def test_embedded_factual_diagnostic_never_repairs_strict_format(self):
        wrapped = '```json\n' + ANSWER + '\n```'
        diagnostic = M.content_diagnostics(wrapped)
        self.assertTrue(diagnostic['FactualValuesMatch'])
        self.assertEqual(diagnostic['FactualValuesObserved'], M.EXPECTED)
        self.assertFalse(diagnostic['StrictFormatPassed'])
        self.assertFalse(M.semantic(wrapped))
        duplicate = '{"director":"Wrong","director":"Yoshinori Kitase","character_designer":"Tetsuya Nomura"}'
        self.assertIsNone(M.content_diagnostics(duplicate)['FactualValuesObserved'])
        self.assertFalse(M.content_diagnostics(duplicate)['StrictFormatPassed'])

    def test_public_probe_csv_and_json_token_files_keep_exact_ids(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'input.ids'
            for raw in ('101,102\n', ' 101 ,\t102\r\n', '101\n102', '[101,102]'):
                path.write_text(raw, encoding='utf-8')
                with self.subTest(raw=raw): self.assertEqual(M.load_token_file(path), [101, 102])
            for raw in ('', '1,,2', ',1', '1,', '01,2', '-1,2', '١,2', '2147483648',
                    '1.0,2', 'true', '[true,2]', '[1.0,2]', '[]', '[1,NaN]'):
                path.write_text(raw, encoding='utf-8')
                with self.subTest(raw=raw), self.assertRaises(ValueError): M.load_token_file(path)

    def test_cli_can_grade_actual_public_probe_csv_without_false_json_failure(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            tokens, manifest, report_path = root/'input.ids', root/'manifest.json', root/'report.json'
            report = tensorsharp()
            tokens.write_text('101,102\n', encoding='utf-8')
            manifest.write_text(json.dumps(report['HistoricalModelManifest']), encoding='utf-8')
            for before, after, path in (('InputFile', 'InputFileAfter', tokens),
                    ('HistoricalModelIdentity', 'HistoricalModelIdentityAfter', manifest)):
                identity = {'Path': str(path), 'Bytes': path.stat().st_size, 'Sha256': M.digest(path)}
                report[before] = identity; report[after] = copy.deepcopy(identity)
            report_path.write_text(json.dumps(report), encoding='utf-8')
            # Use a fake script home so the complete CLI still enforces its
            # ignored-output rule while this unit test leaves no repo evidence.
            home = root/'eng'/'validation'; home.mkdir(parents=True)
            output = root/'artifacts'/'quality.json'
            argv = ['check-factual-json.py', '--engine', 'tensorsharp', '--fixture', FIXTURE,
                '--report', str(report_path), '--output', str(output)]
            with patch.object(M, 'HERE', home), patch.object(M.sys, 'argv', argv):
                self.assertEqual(M.main(), 0)
            result = json.loads(output.read_text('utf-8'))
            self.assertTrue(result['ExecutionChecks']['OriginalInputFile'])
            self.assertTrue(result['ExecutionChecks']['OriginalModelManifest'])
            self.assertTrue(result['QualityPassed'])


class TensorSharpFactualEvidenceTests(unittest.TestCase):
    def grade(self, report): return M.inspect_tensorsharp(report, FIXTURE)

    def test_unknown_task_false_pass_and_exit1_can_qualify(self):
        result = self.grade(tensorsharp())
        self.assertTrue(result['ExecutionPassed'], result['ExecutionChecks'])
        self.assertTrue(result['SemanticPassed'])
        self.assertTrue(result['QualityPassed'])

    def test_semantic_failure_is_separate_from_clean_execution(self):
        report = tensorsharp()
        wrong = json.dumps(M.EXPECTED | {'director': 'Wrong Person'})
        report['Rounds'][0]['Content'] = wrong
        report['Rounds'][0]['RawText'] = report['RawTextAfterCleanup'] = '</think>' + wrong + '<|user|>'
        result = self.grade(report)
        self.assertTrue(result['ExecutionPassed'])
        self.assertFalse(result['SemanticPassed'])
        self.assertFalse(result['QualityPassed'])

    def test_correct_answer_does_not_override_failed_cleanup_or_identity(self):
        mutations = [lambda r: r.update(SessionDisposed=False),
            lambda r: r.update(Errors=['native disposal failed']),
            lambda r: r['BudgetAfterSessionDisposed']['Current'][0].update(Committed=1),
            lambda r: r['EngineCountersBeforeDispose'].update(PendingRequestCount=1),
            lambda r: r['SourceToolsAfter'][0].update(Sha256='c' * 64),
            lambda r: r.pop('SourceToolsUnchanged'),
            lambda r: r['LoadedNativeAfter'][0].update(Path='D:/other/GgmlOps.dll'),
            lambda r: r['HistoricalModelManifest']['files'][0].update(mtime_ns=200)]
        for mutate in mutations:
            report = tensorsharp(); mutate(report)
            with self.subTest(mutate=mutate):
                result = self.grade(report)
                self.assertTrue(result['SemanticPassed'])
                self.assertFalse(result['ExecutionPassed'])
                self.assertFalse(result['QualityPassed'])

    def test_never_accepts_length_cap_early_eos_or_missing_forward(self):
        for mutate in (lambda r: r['Rounds'][0]['Completion'].update(FinishReason='max_tokens'),
                lambda r: r['Rounds'][0].update(OutputTokens=[1, 9, 2, 9]),
                lambda r: r['Rounds'][0].update(ComputedTokens=4),
                lambda r: r['Rounds'][0].update(PublishedTokens=[1]),
                lambda r: r['Rounds'][0]['Completion'].update(OutputTokenCount=True)):
            report = tensorsharp(); mutate(report)
            with self.subTest(mutate=mutate): self.assertFalse(self.grade(report)['QualityPassed'])

    def test_repetitions_cannot_hide_a_wrong_or_missing_round(self):
        report = tensorsharp()
        report.update(RequestedRepetitions=2, CompletedRepetitions=2)
        report['EngineCountersBeforeDispose'].update(TotalSubmitted=2, TotalCompleted=2)
        report['Rounds'].append(copy.deepcopy(report['Rounds'][0]))
        self.assertFalse(self.grade(report)['ExecutionPassed'])
        report['Rounds'][1].update(Round=2, RequestId='request-2')
        self.assertTrue(self.grade(report)['QualityPassed'])
        report['Rounds'][0]['Content'] = '{}'
        self.assertFalse(self.grade(report)['SemanticPassed'])
        report['Rounds'].pop(0)
        self.assertFalse(self.grade(report)['ExecutionPassed'])

    def test_fixture_mismatch_and_forced_capture_cannot_qualify(self):
        for mutate in (lambda r: r.update(DecodedPrompt=RENDERED + ' '),
                lambda r: r['Arguments'].update({'capture-token-id': '1'}),
                lambda r: r.update(DiagnosticForcedCaptureToken=1),
                lambda r: r['Scheduler'].update(Enabled=True)):
            report = tensorsharp(); mutate(report)
            with self.subTest(mutate=mutate): self.assertFalse(self.grade(report)['QualityPassed'])

    def test_real_backend_assembly_name_and_complete_memory_closure_required(self):
        report = tensorsharp()
        self.assertTrue(self.grade(report)['ExecutionChecks']['ManagedIdentity'])
        for name in ('TensorSharp.Core.dll', 'TensorSharp.Memory.dll', 'TensorSharp.Backends.GGML.dll',
                'TensorSharp.Backends.Cuda.dll'):
            broken = copy.deepcopy(report)
            broken['ManagedBefore'] = [r for r in broken['ManagedBefore'] if not r['Path'].endswith(name)]
            broken['ManagedAfter'] = copy.deepcopy(broken['ManagedBefore'])
            with self.subTest(missing=name):
                self.assertFalse(self.grade(broken)['ExecutionChecks']['ManagedIdentity'])
        wrong_name = copy.deepcopy(report)
        for rows in (wrong_name['ManagedBefore'], wrong_name['ManagedAfter']):
            for row in rows:
                row['Path'] = row['Path'].replace('TensorSharp.Backends.GGML.dll', 'TensorSharp.GGML.dll')
        self.assertFalse(self.grade(wrong_name)['ExecutionChecks']['ManagedIdentity'])

    def test_exact_loaded_split_shards_can_use_manifest_with_unused_mmproj(self):
        report = tensorsharp()
        report['Arguments']['model'] = 'C:/models/GLM-5.3-Flash-UD-IQ1_S-00001-of-00003.gguf'
        stats, files = {}, []
        for i in range(1, 4):
            path = f'C:/models/GLM-5.3-Flash-UD-IQ1_S-{i:05d}-of-00003.gguf'
            files.append({'path': path, 'bytes': 100 + i, 'mtime_ns': 100 * i, 'sha256': 'b' * 64})
            stats[path] = {'Path': path, 'Bytes': 100 + i, 'LastWriteTimeUtcTicks': 621355968000000000 + i}
        files.append({'path': 'C:/models/mmproj-F16.gguf', 'bytes': 222,
            'mtime_ns': 99900, 'sha256': 'c' * 64})
        report.update(ModelStatsBefore=stats, ModelStatsAfter=copy.deepcopy(stats), HistoricalModelManifest={'files': files})
        self.assertTrue(self.grade(report)['ExecutionChecks']['ModelIdentity'])
        for mutate in (lambda r: r['ModelStatsBefore'].pop(next(iter(r['ModelStatsBefore']))),
                lambda r: r['HistoricalModelManifest']['files'].pop(1),
                lambda r: r['Arguments'].update(model='C:/models/GLM-5.3-Flash-UD-IQ1_S-00001-of-00004.gguf'),
                lambda r: r['HistoricalModelManifest']['files'].append(copy.deepcopy(r['HistoricalModelManifest']['files'][0])),
                lambda r: r['HistoricalModelManifest']['files'][0].update(bytes=999),
                lambda r: r['HistoricalModelManifest']['files'][0].update(mtime_ns=200)):
            broken = copy.deepcopy(report); mutate(broken)
            broken['ModelStatsAfter'] = copy.deepcopy(broken['ModelStatsBefore'])
            with self.subTest(mutate=mutate):
                self.assertFalse(self.grade(broken)['ExecutionChecks']['ModelIdentity'])

    def test_identity_stat_ticks_match_actual_ns_without_float_rounding(self):
        report = tensorsharp()
        ticks = 639272497709848349
        report['ModelStatsBefore']['C:/model.gguf']['LastWriteTimeUtcTicks'] = ticks
        report['ModelStatsAfter'] = copy.deepcopy(report['ModelStatsBefore'])
        report['HistoricalModelManifest']['files'][0]['mtime_ns'] = 1791652970984834900
        self.assertTrue(self.grade(report)['ExecutionChecks']['ModelIdentity'])
        report['HistoricalModelManifest']['files'][0]['mtime_ns'] += 100
        self.assertFalse(self.grade(report)['ExecutionChecks']['ModelIdentity'])


class MayaFactualEvidenceTests(unittest.TestCase):
    def grade(self, report, identity=None, parser=Parser):
        return M.inspect_maya(report, FIXTURE, Tokenizer(), parser, [9],
            {'SourceIdentity': True, 'PackBookends': True} if identity is None else identity)

    def test_full_wire_ids_and_production_parser_can_qualify(self):
        result = self.grade(maya())
        self.assertTrue(result['ExecutionPassed'], result['ExecutionChecks'])
        self.assertTrue(result['SemanticPassed'])
        self.assertTrue(result['QualityPassed'])

    def test_only_verified_terminal_checkpoint_eos_is_excluded_from_parser(self):
        fed = []
        class ObservingParser(Parser):
            def feed(self, raw):
                fed.append(raw)
                return super().feed(raw)
        report = maya()
        result = self.grade(report, parser=ObservingParser)
        self.assertEqual(fed, ['</think>' + ANSWER])
        self.assertEqual(report['RawText'], RAW)
        self.assertEqual(report['RawOutputIds'], [1, 2, 9])
        self.assertTrue(result['QualityPassed'])
        self.assertTrue(result['Rounds'][0]['TerminalEosExcludedFromParser'])
        self.assertEqual(result['Rounds'][0]['ParserInputTokenCount'], 2)
        for output, finish in (([1, 2, 9], 'length'), ([1, 9, 2, 9], 'stop'), ([1, 2], 'length')):
            report = maya(); report['RawOutputIds'] = output; report['Done']['finish'] = finish
            fed.clear()
            result = self.grade(report, parser=ObservingParser)
            self.assertFalse(result['Rounds'][0]['TerminalEosExcludedFromParser'])
            self.assertEqual(fed, [Tokenizer().decode(output)])
            self.assertFalse(result['QualityPassed'])

    def test_fenced_correct_names_still_fail_format_after_eos_exclusion(self):
        wrapped = '```json\n' + ANSWER + '\n```'
        class WrappedTokenizer(Tokenizer):
            def decode(self, ids):
                if ids == [1, 2, 9]: return '</think>' + wrapped + '<|user|>'
                if ids == [1, 2]: return '</think>' + wrapped
                return super().decode(ids)
        report = maya(); report['RawText'] = '</think>' + wrapped + '<|user|>'
        result = M.inspect_maya(report, FIXTURE, WrappedTokenizer(), Parser, [9], {'SourceIdentity': True})
        self.assertTrue(result['ExecutionPassed'])
        self.assertTrue(result['Rounds'][0]['FactualValuesMatch'])
        self.assertFalse(result['StrictFormatPassed'])
        self.assertFalse(result['SemanticPassed'])
        self.assertFalse(result['QualityPassed'])

    def test_correct_content_never_bypasses_source_or_cleanup(self):
        report = maya(); report['CleanupErrors'] = ['reader still live']
        result = self.grade(report)
        self.assertTrue(result['SemanticPassed']); self.assertFalse(result['QualityPassed'])
        result = self.grade(maya(), {'SourceIdentity': False})
        self.assertTrue(result['SemanticPassed']); self.assertFalse(result['QualityPassed'])

    def test_stop_label_does_not_replace_actual_eos_or_counts(self):
        for mutate in (lambda r: r.update(RawOutputIds=[1, 2]),
                lambda r: r.update(RawOutputIds=[1, 9, 2, 9]),
                lambda r: r['Done'].update(generated=2),
                lambda r: r['Done'].update(prompt_tokens=True),
                lambda r: r.update(RawTokenWallMs=[10, 30, 20]),
                lambda r: r['Done'].update(drafts_offered=1),
                lambda r: r.update(RawText='invented answer')):
            report = maya(); mutate(report)
            with self.subTest(mutate=mutate): self.assertFalse(self.grade(report)['QualityPassed'])

    def test_correct_json_with_tool_call_does_not_qualify(self):
        class ToolParser(Parser):
            def finish(self): return [SimpleNamespace(kind='tool_call')]
        result = self.grade(maya(), parser=ToolParser)
        self.assertTrue(result['SemanticPassed']); self.assertFalse(result['QualityPassed'])

    def test_missing_unsafe_parser_identity_reports_semantics_unavailable(self):
        result = M.finish('maya', FIXTURE, {'SourceIdentity': False}, [])
        self.assertFalse(result['ExecutionPassed'])
        self.assertIsNone(result['SemanticPassed'])
        self.assertFalse(result['QualityPassed'])

    def test_small_evidence_hash_rejects_oversized_input_without_reading(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'file'
            path.write_bytes(b'12345')
            with self.assertRaises(ValueError): M.digest(path, maximum=4)

    def test_maya_source_plan_and_tokenizer_identity_cannot_be_skipped(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source, pack = root/'source', root/'pack'
            source.mkdir(); (pack/'tokenizer').mkdir(parents=True)
            rows = []
            for name in ('chat_template.jinja', 'merges.txt', 'token_type.json', 'tokenizer.json', 'vocab.json'):
                path = pack/'tokenizer'/name
                path.write_text('{}', encoding='utf-8')
                rows.append({'path': str(path), 'bytes': path.stat().st_size, 'sha256': M.digest(path)})
            evidence = root/'evidence.json'
            evidence.write_text(json.dumps({'exit_code': 0, 'original_checkpoint_stat_unchanged': True,
                'source_unchanged': True}), encoding='utf-8')
            harness = root/'harness.py'; harness.write_text('pass\n', encoding='utf-8')
            plan = {'source_path': str(source), 'source_revision': '1'*40, 'pack': str(pack),
                'exe': str(root/'strata.exe'), 'pack_evidence': str(evidence),
                'pack_evidence_sha256': M.digest(evidence), 'eos_ids': [9],
                'current_harness_pins': {'harness.py': {'path': str(harness), 'sha256': M.digest(harness)}}}
            plan_path = root/'plan.json'; plan_path.write_text(json.dumps(plan), encoding='utf-8')
            report = {'PlanSha256': M.digest(plan_path), 'ToolSha256': M.digest(M.HERE/'maya-token-observation.py'),
                'VerifiedPackBefore': rows, 'VerifiedPackAfter': copy.deepcopy(rows),
                'ContextTokens': 32768, 'PrefillRows': 512,
                'Command': [plan['exe'], '--glm-pack', str(pack), '--max-context', '32768', '--prefill', '512', '--serve']}
            cases = [
                ('PlanIdentity', lambda r: r.update(PlanSha256='f'*64), ''),
                ('SourceIdentity', lambda r: None, ' M tools/strata_tokenizer.py\n'),
                ('TokenizerIdentity', lambda r: r['VerifiedPackBefore'][0].update(sha256='f'*64), ''),
                ('PackBookends', lambda r: r['VerifiedPackAfter'][0].update(bytes=3), ''),
            ]
            for name, mutate, dirty in cases:
                candidate = copy.deepcopy(report); mutate(candidate)
                with self.subTest(check=name), patch.object(M.subprocess, 'check_output', side_effect=['1'*40, dirty]):
                    with self.assertRaisesRegex(ValueError, name):
                        M.maya_identity(candidate, plan_path)


if __name__ == '__main__':
    unittest.main()
