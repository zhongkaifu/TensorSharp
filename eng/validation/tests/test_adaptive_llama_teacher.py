import copy
import hashlib
import importlib.util
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch


spec = importlib.util.spec_from_file_location("adaptive_llama_teacher",
    Path(__file__).resolve().parents[1] / "AdaptiveMemoryProbe" / "llama-teacher.py")
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def completion():
    return {"truncated": False, "stop_type": "limit", "tokens_predicted": 1, "tokens": [0],
            "generation_settings": {"temperature": 0, "repeat_penalty": 1, "presence_penalty": 0,
                                    "frequency_penalty": 0, "dry_multiplier": 0, "ignore_eos": True},
            "completion_probabilities": [{"top_logprobs": [
                {"id": 0, "logprob": -7.}, {"id": 1, "logprob": -9.}, {"id": 2, "logprob": -12.}]}]}


class IndependentTeacher(unittest.TestCase):
    def test_fixed_history_request_disables_penalties_and_requests_every_log_probability(self):
        body = module.request_body({"InputTokens": [1, 8, 3], "Elements": 12})
        self.assertEqual([1, 8, 3], body["prompt"])
        self.assertEqual(12, body["n_probs"])
        self.assertFalse(body["post_sampling_probs"])
        self.assertTrue(body["ignore_eos"])
        self.assertEqual("none", body["speculative.type"])

    def test_offset_does_not_hide_error_or_argmax_change(self):
        self.assertTrue(module.compare([3., 1., -2.], completion())["numerical_gate_passed"])
        self.assertFalse(module.compare([3., 1., 0.], completion())["numerical_gate_passed"])
        self.assertFalse(module.compare([3., 4., -2.], completion())["numerical_gate_passed"])

    def test_partial_modified_or_non_argmax_predictions_are_rejected(self):
        for key, value in (("truncated", True), ("stop_type", "eos"), ("tokens", [1]),
                           ("tokens_predicted", 2), ("generation_settings", {})):
            with self.subTest(key=key), self.assertRaises(ValueError):
                module.compare([3., 1., -2.], {**completion(), key: value})

    def test_identity_histories_and_full_data_are_bound_while_changed_binaries_are_allowed(self):
        with tempfile.TemporaryDirectory() as directory:
            data = Path(directory) / "capture.f32"
            data.write_bytes(b"source bytes")
            report = {"ModelSha256": "a" * 64, "ModelBytes": 12, "ModelGeometry": {"layers": 3},
                      "Steps": 2, "Repeats": 1, "ModelsAssemblySha256": "b" * 64,
                      "LogitCapture": {"Sha256": hashlib.sha256(data.read_bytes()).hexdigest()}}
            rows = [{"InputTokens": [1, 2]}]
            second = copy.deepcopy(report)
            second["ModelsAssemblySha256"] = "c" * 64
            loaded = [(report, rows, data), (second, copy.deepcopy(rows), data)]
            identity = {"model_sha256": "a" * 64, "binary_sha256": "d" * 64,
                        "source_revision": "revision", "command": ["llama-server"], "startup_evidence": "server.log"}
            with patch.object(module.capture, "load", side_effect=loaded):
                self.assertEqual(2, len(module.validate_inputs([Path("old"), Path("new")], identity)))
            for mutation in ("checkpoint", "history", "capture"):
                invalid = copy.deepcopy(loaded)
                if mutation == "checkpoint": invalid[1][0]["ModelSha256"] = "f" * 64
                if mutation == "history": invalid[1][1][0]["InputTokens"] = [1, 3]
                if mutation == "capture": invalid[1][0]["LogitCapture"]["Sha256"] = "0" * 64
                with self.subTest(mutation=mutation), patch.object(module.capture, "load", side_effect=invalid), self.assertRaises(ValueError):
                    module.validate_inputs([Path("old"), Path("new")], identity)
            with patch.object(module.capture, "load", side_effect=loaded), self.assertRaises(ValueError):
                module.validate_inputs([Path("old"), Path("new")], {**identity, "model_sha256": "f" * 64})


if __name__ == "__main__":
    unittest.main()
