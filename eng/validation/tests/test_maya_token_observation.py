import importlib.util
from pathlib import Path
import unittest


SPEC = importlib.util.spec_from_file_location('maya_token_observation_profile_test',
    Path(__file__).parents[1] / 'maya-token-observation.py')
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class MayaObservationPolicyTests(unittest.TestCase):
    def test_supported_bounded_widths(self):
        for value in (32, 64, 128, 256, 512):
            self.assertEqual(MODULE.prefill_rows(value), value)
        for value in (True, '128', 0, -1, 1024):
            with self.subTest(value=value), self.assertRaises(ValueError):
                MODULE.prefill_rows(value)

    def test_actual_progress_keeps_engine_count_and_attribution(self):
        value = MODULE.progress_record('PP 128 1340 1234.5 6.7')
        self.assertEqual(value['ComputedPromptTokens'], 128)
        self.assertEqual(value['TotalPromptTokens'], 1340)
        self.assertEqual(value['EngineReportedCumulativeMs'], 1234.5)

    def test_partial_invalid_or_nonfinite_progress_rejected(self):
        for value in ('PP 0 1340 1 2', 'PP 1400 1340 1 2', 'PP 128 1340 nan 2',
                      'PP 128 1340 -1 2', 'PP 128 1340 1 2 extra', 'PP 128 1340'):
            with self.subTest(value=value), self.assertRaises(ValueError):
                MODULE.progress_record(value)
        with self.assertRaises(ValueError):
            MODULE.progress_record('PP 128 1340 '+('9'*400)+' 2')


if __name__ == '__main__':
    unittest.main()
