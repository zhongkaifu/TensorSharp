"""Model-free tests: scripted native rows choose baseline, never a TP model."""
import copy
import contextlib
import importlib.util
import io
import json
from pathlib import Path
import struct
import tempfile
import unittest
from unittest import mock

import test_deepseek_teacher_capture as capture_fixtures

capture = capture_fixtures.capture
fixture = capture_fixtures.fixture
FakeNative = capture_fixtures.FakeNative

SPEC = importlib.util.spec_from_file_location('teacher_baseline', Path(__file__).parents[1] / 'generate-deepseek-teacher-baseline.py')
baseline = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(baseline)


class GreedyNative(FakeNative):
    def __init__(self, winners):
        super().__init__()
        self.winners = list(winners)
        self.count = 0
    def forward(self, raw, output):
        super().forward(raw, output)
        winner = self.winners[min(self.count, len(self.winners)-1)]
        self.count += 1
        for i in range(4): output[i] = 5 if i in winner else -5
        return 0


class BaselineTests(unittest.TestCase):
    def prepare(self, root):
        plan, schedule, exported, _, report = fixture(root / 'tokens')
        output = root / 'baseline'
        output.mkdir()
        report.update(phase='independent-baseline-continuation', variant='non-tp')
        calls = baseline.GreedyCalls(capture, plan, 'b' * 64, schedule['shared_identity'], exported,
                                    root / 'tokens', output, report)
        return plan, schedule, exported, report, output, calls

    def test_helper_executes_only_checked_bytes(self):
        raw = b'marker = 7\n'
        with mock.patch.object(Path, 'read_bytes', return_value=raw):
            self.assertEqual(baseline.load_helper('unused.py', capture.sha(raw)).marker, 7)
            with self.assertRaises(ValueError): baseline.load_helper('unused.py', '0' * 64)

    def test_complete_producer_cli_freezes_only_audited_non_tp_rows(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            old_config, _, source_report = capture_fixtures.CliTests().fixture(root / 'inputs')
            config = json.loads(Path(old_config).read_text())
            schedule = json.loads(Path(config['schedule_path']).read_text())
            identity_path = root / 'shared.json'
            identity_path.write_text(json.dumps(schedule['shared_identity']))
            config.update(capture_helper_path=capture.__file__, capture_helper_sha256=capture.file_sha(capture.__file__),
                          shared_identity_path=str(identity_path), shared_identity_sha256=capture.file_sha(identity_path),
                          producer_sha256=capture.file_sha(baseline.__file__), maximum_tokens=64)
            config_path = root / 'producer.json'
            config_path.write_text(json.dumps(config))
            output = root / 'output'
            native = GreedyNative([[0], [1, 2], [3]])
            native.load = lambda *_: (output / 'native.log').write_text(capture_fixtures.PlacementTests.log(False))
            events = []
            native.free = lambda: events.append('free')
            observation = copy.deepcopy(source_report['observations']['before'])
            def audit():
                events.append('audit')
                return copy.deepcopy(observation)
            argv = ['producer.py', '--config', str(config_path), '--config-sha256', capture.file_sha(config_path), '--output', str(output)]
            with mock.patch.object(baseline.sys, 'argv', argv), mock.patch.object(baseline.sys, 'platform', 'linux'), \
                    mock.patch.dict(capture.os.environ, {}, clear=True), mock.patch.object(baseline, 'load_helper', return_value=capture), \
                    mock.patch.object(capture, 'Native', return_value=native), \
                    mock.patch.object(capture, 'Observer', return_value=type('Observer', (), {'audit': staticmethod(audit)})()), \
                    contextlib.redirect_stdout(io.StringIO()):
                baseline.main()
            self.assertEqual(events, ['audit', 'audit', 'free'])
            frozen = json.loads((output / 'frozen.json').read_text())
            self.assertEqual(frozen['status'], 'frozen')
            self.assertFalse(frozen['release_qualified'])
            source = json.loads((output / 'rows/capture.json').read_text())
            continuation = json.loads((output / 'baseline-continuation.json').read_text())
            final_schedule = json.loads((output / 'teacher-schedule.json').read_text())
            capture.validate_baseline_trace(source, continuation, final_schedule, frozen['schedule_sha256'], json.loads(Path(config['plan_path']).read_text()))
            self.assertEqual(continuation['requests'][0]['tokens'], [1, 3])

    def test_full_rows_select_lowest_tie_then_eos_and_freeze_independent_history(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            plan, schedule, exported, report, output, calls = self.prepare(root)
            native = GreedyNative([[0], [1, 2], [3]])
            capture.capture_rows(native, calls, output, output / 'rows', report)
            self.assertEqual(calls.requests, [{'id': 'request0', 'tokens': [1, 3], 'stop_reason': 'eos'}])
            self.assertEqual(native.calls, [('reset',), ('forward', [0, 1]), ('forward', [2]), ('forward', [1])])
            frozen = baseline.freeze_schedule(capture, plan, 'b' * 64, schedule['shared_identity'], report,
                calls.requests, exported, root / 'tokens', output, '9' * 64)
            self.assertEqual(frozen['rows'], 3)
            self.assertEqual(frozen['requests'], 1)
            saved = json.loads((output / 'teacher-schedule.json').read_text())
            continuation = json.loads((output / 'baseline-continuation.json').read_text())
            self.assertEqual(continuation['source_capture_sha256'], capture.file_sha(output / 'rows/capture.json'))
            rows = capture.validate_serial_schedule(plan, 'b' * 64, saved, output, exported, root / 'tokens', continuation)
            self.assertEqual([row['next_forced_token_id'] for row in rows], [None, 1, 3])
            # A deliberately different replay argmax must still feed token1
            # from the frozen baseline, never candidate winner0.
            replay = FakeNative()
            replay_report = copy.deepcopy(report)
            capture.capture_rows(replay, rows, output, root / 'teacher', replay_report)
            self.assertEqual(replay.calls[-1], ('forward', [1]))

    def test_immediate_eos_has_no_continuation_forward(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            _, _, _, report, output, calls = self.prepare(root)
            native = GreedyNative([[0], [3]])
            capture.capture_rows(native, calls, output, output / 'rows', report)
            self.assertEqual(calls.requests[0]['tokens'], [3])
            self.assertEqual(len(report['rows']), 2)
            self.assertEqual(native.position, 3)

    def test_64_predictions_feed_only_63_tokens(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            plan, schedule, exported, report, output, calls = self.prepare(root)
            native = GreedyNative([[2]])
            capture.capture_rows(native, calls, output, output / 'rows', report)
            self.assertEqual(calls.requests[0]['tokens'], [2] * 64)
            self.assertEqual(calls.requests[0]['stop_reason'], 'maximum_tokens')
            self.assertEqual(native.position, 3 + 63)
            self.assertEqual(len(report['rows']), 65)
            baseline.freeze_schedule(capture, plan, 'b' * 64, schedule['shared_identity'], report,
                calls.requests, exported, root / 'tokens', output, '9' * 64)

    def test_failed_forward_never_selects_tokens_or_freezes_teacher_schedule(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            plan, schedule, exported, report, output, calls = self.prepare(root)
            with self.assertRaises(ValueError): capture.capture_rows(FakeNative('status'), calls, output, output / 'rows', report)
            self.assertEqual(calls.requests, [])
            self.assertEqual((output / 'rows/logits.f32').stat().st_size, 16)
            with self.assertRaises(ValueError):
                baseline.freeze_schedule(capture, plan, 'b' * 64, schedule['shared_identity'], report,
                    calls.requests, exported, root / 'tokens', output, '9' * 64)
            self.assertFalse((output / 'teacher-schedule.json').exists())

    def test_tp_source_or_changed_greedy_choice_cannot_define_baseline(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            plan, schedule, exported, report, output, calls = self.prepare(root)
            capture.capture_rows(GreedyNative([[3]]), calls, output, output / 'rows', report)
            invalid = copy.deepcopy(report)
            invalid['variant'] = 'expert-tp7'
            with self.assertRaisesRegex(ValueError, 'nonTP baseline'):
                baseline.freeze_schedule(capture, plan, 'b' * 64, schedule['shared_identity'], invalid,
                    calls.requests, exported, root / 'tokens', output, '9' * 64)
            invalid = copy.deepcopy(report)
            invalid['rows'][-1]['next_forced_token_id'] = 1
            with self.assertRaisesRegex(ValueError, 'greedy argmax'):
                baseline.freeze_schedule(capture, plan, 'b' * 64, schedule['shared_identity'], invalid,
                    calls.requests, exported, root / 'tokens', output, '9' * 64)


if __name__ == '__main__': unittest.main()
