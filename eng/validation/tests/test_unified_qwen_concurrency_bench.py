"""Overlap is generation progress before terminal frames, never socket lifetime."""
import importlib.util
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

SPEC = importlib.util.spec_from_file_location("bench", Path(__file__).parents[1]/"unified-qwen-concurrency-bench.py")
BENCH = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(BENCH)


def row(tokens, terminal_at, **extra):
    return dict(completed=True, tokens=[[t, "x"] for t in tokens], terminal_at=terminal_at, ended=100, **extra)


class OverlapTests(unittest.TestCase):
    def test_summary_keeps_engagement_separate_from_strict_continuity(self):
        report = {"passed": True}
        evidence = {"passed": False, "actual_joint_engagement_verified": True,
                    "full_interval_continuity_verified": False, "joint_steps": 148,
                    "continuity_gaps": [{"line": 121}, {"line": 130}],
                    "contiguous_joint_runs": [{"steps": 47}, {"steps": 9}, {"steps": 92}],
                    "generation_timings": {"all_current_requests_measured": False}}
        BENCH.attach_fused_evidence(report, evidence)
        self.assertTrue(report["actual_joint_engagement_verified"])
        self.assertFalse(report["batched_fused_graph_verified"])
        self.assertFalse(report["full_interval_continuity_verified"])
        self.assertEqual(148, report["fused_joint_steps"])
        self.assertEqual(2, len(report["fused_continuity_gaps"]))
        self.assertFalse(report["generation_timings"]["all_current_requests_measured"])
        # The caller combines the strict graph result with independent HTTP and
        # greedy parity results. Merely attaching a diagnostic does not pass it.
        report["passed"] &= report["batched_fused_graph_verified"]
        self.assertFalse(report["passed"])

    def test_true_interleaving(self):
        self.assertTrue(BENCH.overlap_gate([row([1,3,5],6),row([2,4,7],8)])["passed"])

    def test_open_socket_is_not_active_generation(self):
        self.assertFalse(BENCH.overlap_gate([row([1,2,3],4),row([5,6,7],8)])["passed"])

    def test_one_frame_cannot_qualify(self):
        self.assertFalse(BENCH.overlap_gate([row([1,3],6),row([2,4,7],8)])["passed"])

    def test_error_and_failed_cleanup_refuse(self):
        for key in ("error", "cleanup_error"):
            self.assertFalse(BENCH.overlap_gate([row([1,3,5],6,**{key:"failed"}),row([2,4,7],8)])["passed"])

    def test_nonfinite_missing_or_out_of_order_refuse(self):
        for invalid in (row([1,3,5],float("nan")),row([1,3,5],None),row([1,5,3],6),row([1,3,7],6)):
            self.assertFalse(BENCH.overlap_gate([invalid,row([2,4,7],8)])["passed"])

    def test_phase_observations_do_not_call_frames_token_throughput(self):
        measured = row([2, 3, 4], 5, started=1, headers_at=1.5,
                       terminal={"tokenCount": 19, "elapsed": 3.9, "promptTokens": 41,
                                 "kvReusedTokens": 10, "truncated": True})
        phases = BENCH.phase_observations(measured)
        self.assertEqual(1, phases["dispatch_to_first_content_seconds"])
        self.assertEqual(3, phases["content_frames"])
        self.assertEqual(19, phases["server_token_count"])
        self.assertEqual(10, phases["server_kv_reused_tokens"])
        self.assertNotIn("decode_tokens_per_second", phases)

    def test_missing_invalid_phase_times_remain_unknown(self):
        phases = BENCH.phase_observations({"tokens": [], "started": float("nan"), "headers_at": 3})
        self.assertIsNone(phases["dispatch_to_headers_seconds"])
        self.assertIsNone(phases["dispatch_to_first_content_seconds"])
        self.assertIsNone(phases["server_prompt_tokens"])

    def test_integrated_verifier_preserves_source_log_and_exact_hash(self):
        fixture_spec = importlib.util.spec_from_file_location("fused_fixture", Path(__file__).with_name("test_verify_fused_batch_evidence.py"))
        fixture = importlib.util.module_from_spec(fixture_spec); fixture_spec.loader.exec_module(fixture)
        report, records = fixture.fixture()
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            source = output / "server.jsonl"
            raw = ("\n".join(json.dumps(record) for record in records)+"\n").encode()
            source.write_bytes(raw)
            result = BENCH.capture_fused_evidence(report, source, output)
            self.assertTrue(result["passed"], result)
            self.assertEqual(raw, (output/"fused-server-log.jsonl").read_bytes())
            self.assertEqual(hashlib.sha256(raw).hexdigest(), result["server_log"]["sha256"])

    def test_malformed_trace_preserved_and_never_qualified(self):
        with tempfile.TemporaryDirectory() as directory, patch.object(BENCH.time, "sleep"):
            output = Path(directory); source = output / "server.jsonl"
            source.write_bytes(b'{"partial":')
            with self.assertRaises(json.JSONDecodeError):
                BENCH.capture_fused_evidence({}, source, output)
            self.assertEqual(source.read_bytes(), (output / "fused-server-log.jsonl").read_bytes())
            self.assertFalse((output / "fused-evidence.json").exists())


if __name__ == "__main__": unittest.main()
