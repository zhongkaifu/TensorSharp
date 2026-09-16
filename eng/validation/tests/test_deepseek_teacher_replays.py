"""Model-free checks for immutable teacher replay derivation."""
import copy
import importlib.util
from pathlib import Path
import tempfile
import unittest
from unittest import mock

SPEC = importlib.util.spec_from_file_location('teacher_replays', Path(__file__).parents[1] / 'prepare-deepseek-teacher-replays.py')
replays = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(replays)

def fixture():
    producer = {key: 'fixture-' + key for key in ['scope', 'plan_path', 'plan_sha256', 'native_library', 'native_source_root',
        'original_failed_build_path', 'original_failed_build_sha256', 'token_export_path', 'model_attestation_path', 'reset_api']}
    producer.update(release_qualified=False, variant='non-tp', maximum_tokens=64, model_layers=40)
    frozen = dict(status='frozen', release_qualified=False, requests=113, schedule_sha256='a' * 64)
    return producer, frozen

class ReplayPreparationTests(unittest.TestCase):
    def test_only_variants_differ_and_continuation_is_shared(self):
        producer, frozen = fixture()
        rows = replays.derive_configs(producer, frozen, 'baseline', 'captures')
        self.assertEqual(['non-tp-r1', 'tp7-r1', 'tp7-r2', 'non-tp-r2'], [r[0] for r in rows])
        common = [{k: v for k, v in row[1].items() if k != 'variant'} for row in rows]
        self.assertTrue(all(c == common[0] for c in common))
        self.assertTrue(all(row[1]['release_qualified'] is False for row in rows))
        self.assertEqual(producer['original_failed_build_sha256'], common[0]['original_failed_build_sha256'])
        self.assertEqual(frozen['schedule_sha256'], common[0]['schedule_sha256'])
    def test_reject_tp_or_qualified_or_different_token_cap_producer(self):
        for key, value in [('variant', 'expert-tp7'), ('release_qualified', True), ('maximum_tokens', 63)]:
            with self.subTest(key=key):
                producer, frozen = fixture();producer[key] = value
                with self.assertRaises(ValueError):replays.derive_configs(producer, frozen, 'x', 'y')
    def test_reject_incomplete_or_qualified_or_missing_request_frozen_input(self):
        for key, value in [('status', 'running'), ('release_qualified', True), ('requests', 112)]:
            with self.subTest(key=key):
                producer, frozen = fixture();frozen[key] = value
                with self.assertRaises(ValueError):replays.derive_configs(producer, frozen, 'x', 'y')
    def test_helper_executes_only_checked_bytes(self):
        raw = b'marker = 123\n'
        with mock.patch.object(Path, 'read_bytes', return_value=raw) as read:
            result = replays.load_helper('not-a-real-file.py', replays.sha(raw))
            self.assertEqual(123, result.marker);self.assertEqual(1, read.call_count)
        with mock.patch.object(Path, 'read_bytes', return_value=raw):
            with self.assertRaises(ValueError):replays.load_helper('x.py', '0' * 64)

if __name__ == '__main__':unittest.main()
