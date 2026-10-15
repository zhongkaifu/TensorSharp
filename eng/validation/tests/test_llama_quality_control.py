import importlib.util
from pathlib import Path
import unittest

spec = importlib.util.spec_from_file_location('llama_control', Path(__file__).resolve().parents[1] / 'llama-quality-control.py')
control = importlib.util.module_from_spec(spec)
spec.loader.exec_module(control)


class LlamaTimingTests(unittest.TestCase):
    def test_uses_actual_engine_counts_and_durations(self):
        data = dict(timings=dict(cache_n=0, prompt_n=100, prompt_ms=200, predicted_n=10, predicted_ms=250,
                                prompt_per_second=500, predicted_per_second=36))
        self.assertEqual(control.llama_metrics(data)['prefill_tps'], 500)
        self.assertEqual(control.llama_metrics(data)['decode_tps'], 36)
        self.assertEqual(control.llama_metrics(data)['decode_tokens'], 9)
        self.assertEqual(control.llama_metrics(data)['decode_reported_tokens'], 10)
        for key, value in [('predicted_n', 0), ('prompt_ms', 0), ('predicted_n', True), ('prompt_ms', float('nan')),
                           ('cache_n', 1), ('predicted_per_second', 40)]:
            with self.subTest(key=key), self.assertRaises(ValueError):
                control.llama_metrics(dict(timings={**data['timings'], key: value}))


if __name__ == '__main__':
    unittest.main()
