"""Native acceptance, request identity and managed publication are separate gates."""
import copy
import importlib.util
import json
from pathlib import Path
import unittest

SPEC = importlib.util.spec_from_file_location("fused_evidence", Path(__file__).parents[1] / "verify-fused-batch-evidence.py")
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def fixture():
    report = {"completed": True, "passed": True, "model": {"unifiedMemory": True}, "concurrent": [
        {"request_id": "http-a", "session": {"sessionId": "session-a"}, "completed": True,
         "status": 200, "terminal": {"done": True, "truncated": True}},
        {"request_id": "http-b", "session": {"sessionId": "session-b"}, "completed": True,
         "status": 200, "terminal": {"done": True, "truncated": True}}]}
    maps = [{"template": "generation.sequence.bound internal={InternalRequestId} session={SessionId}",
             "scope": {"RequestId": "http-" + key},
             "props": {"InternalRequestId": "chat-" + key, "SessionId": "session-" + key}}
            for key in ("a", "b")]
    batches = [{"template": "batch.fused.accepted id={BatchExecutionId}", "props": {
        "BatchExecutionId": i + 1, "NativeAcceptedExecutionId": i + 11,
        "SequenceRequestIdsJson": json.dumps(["chat-a", "chat-b"]),
        "PositionsBeforeJson": json.dumps([100 + i, 200 + i]),
        "PositionsAfterJson": json.dumps([101 + i, 201 + i]), "ActiveSequenceCount": 2,
        "NativeNodeCount": 4000, "NativeGraphCount": 1, "NativeHostSegmentCount": 48, "NativeExecutionKind": 1,
        "ManagedPublishCompleted": True}} for i in range(2)]
    return report, maps + batches


class FusedEvidenceTests(unittest.TestCase):
    def test_malformed_http_session_records_fail_closed_without_parser_exception(self):
        for invalid in (None, [], "session-a", 0):
            report, records = fixture()
            report["concurrent"][0]["session"] = invalid
            timing = MODULE.generation_timing_evidence(report, records)
            self.assertFalse(timing["all_current_requests_measured"])
            self.assertTrue(timing["errors"])
            self.assertFalse(MODULE.verify(report, records)["passed"])

    def test_measured_later_generation_cannot_hide_missing_earlier_completion(self):
        report, records = fixture()
        first = {"template": "generation.timings", "scope": {"RequestId": "http-a"}, "props": {
            "InternalRequestId": "first", "SessionId": "session-a", "PromptTokens": 10,
            "KvReusedTokens": 0, "GeneratedTokens": 2, "EngineOutputTokens": None,
            "PromptNs": None, "EvalNs": None, "TotalNs": 3, "FinishReason": "cancelled",
            "EngineCompletionAvailable": False, "TimingBasis": "attributed_forward"}}
        later = copy.deepcopy(first)
        later["props"].update(InternalRequestId="second", PromptNs=1, EvalNs=2,
                               EngineOutputTokens=2, EngineCompletionAvailable=True, FinishReason="stop")
        second_http = copy.deepcopy(later)
        second_http["scope"]["RequestId"] = "http-b"
        second_http["props"].update(InternalRequestId="third", SessionId="session-b")
        for ordered in ([first, later, second_http], [second_http, later, first]):
            timing = MODULE.generation_timing_evidence(report, records + ordered)
            self.assertFalse(timing["all_current_requests_measured"])
            self.assertEqual(["http-a"], timing["missing_http_request_ids"])
            self.assertEqual(["http-a"], timing["unavailable_completion_http_request_ids"])
            self.assertEqual(3, len(timing["rows"]))

    def test_forward_timing_and_unknown_completion_are_separate(self):
        report, records = fixture()
        timings = [{"template": "generation.timings internal={InternalRequestId}",
                    "scope": {"RequestId": "http-" + key}, "props": {
                    "InternalRequestId": "chat-" + key, "SessionId": "session-" + key,
                    "PromptTokens": 100, "KvReusedTokens": 90, "GeneratedTokens": 14,
                    "EngineOutputTokens": 15, "PromptNs": 1000, "EvalNs": 2000,
                    "TotalNs": 9000, "FinishReason": "length", "EngineCompletionAvailable": True,
                    "TimingBasis": "attributed_forward"}} for key in ("a", "b")]
        result = MODULE.verify(report, records + timings)
        self.assertTrue(result["passed"])
        evidence = result["generation_timings"]
        self.assertTrue(evidence["all_current_requests_measured"])
        self.assertEqual(14, evidence["rows"][0]["recorded_pipeline_tokens"])
        self.assertEqual(15, evidence["rows"][0]["engine_output_tokens"])
        self.assertEqual(1000, evidence["rows"][0]["attributed_prefill_ns"])
        self.assertEqual(9000, evidence["rows"][0]["server_generation_total_ns"])
        timings[1]["props"].update(EngineCompletionAvailable=False, EngineOutputTokens=None,
                                     PromptNs=None, EvalNs=None, FinishReason="cancelled")
        result = MODULE.verify(report, records + timings)
        self.assertTrue(result["passed"])
        self.assertFalse(result["generation_timings"]["all_current_requests_measured"])
        self.assertIsNone(result["generation_timings"]["rows"][1]["attributed_decode_ns"])
        self.assertEqual(["http-b"], result["generation_timings"]["missing_http_request_ids"])
        timings[1]["props"]["EvalNs"] = 0
        self.assertTrue(MODULE.verify(report, records + timings)["generation_timings"]["errors"])

    def test_missing_or_child_timing_does_not_fabricate_root_time(self):
        report, records = fixture()
        records.append({"template": "generation.timings", "scope": {"RequestId": "http-a"},
                        "props": {"SessionId": "agent-child", "InternalRequestId": "child"}})
        result = MODULE.verify(report, records)
        self.assertTrue(result["passed"])
        self.assertFalse(result["generation_timings"]["all_current_requests_measured"])
        self.assertEqual([], result["generation_timings"]["rows"])
        self.assertEqual(1, result["generation_timings"]["ignored_other_session_rows"])
        self.assertEqual(["http-a", "http-b"], result["generation_timings"]["missing_http_request_ids"])

    def test_timing_duplicate_unknown_basis_and_reused_overflow_rejected(self):
        report, records = fixture()
        timing = {"template": "generation.timings", "scope": {"RequestId": "http-a"}, "props": {
            "InternalRequestId": "chat-a", "SessionId": "session-a", "PromptTokens": 10,
            "KvReusedTokens": 0, "GeneratedTokens": 2, "EngineOutputTokens": 2,
            "PromptNs": 1, "EvalNs": 2, "TotalNs": 3, "FinishReason": "stop",
            "EngineCompletionAvailable": True, "TimingBasis": "attributed_forward"}}
        for mutation in (lambda p: p.update(TimingBasis="wall"), lambda p: p.update(KvReusedTokens=11),
                         lambda p: p.update(EvalNs=True), lambda p: p.update(EngineCompletionAvailable=None)):
            changed = copy.deepcopy(timing)
            mutation(changed["props"])
            result = MODULE.verify(report, records + [changed])
            self.assertTrue(result["passed"])
            self.assertTrue(result["generation_timings"]["errors"])
        self.assertTrue(MODULE.verify(report, records + [timing, timing])["generation_timings"]["errors"])

    def test_actual_v3_logger_uint64_strings_and_startup_warmup(self):
        path = Path(__file__).parents[1] / "fixtures/fused-batch/actual-server-v3-excerpt.jsonl"
        records = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
        report, _ = fixture()
        maps = records[2:4]
        for row, mapping in zip(report["concurrent"], maps):
            row["request_id"] = mapping["scope"]["RequestId"]
            row["session"]["sessionId"] = mapping["props"]["SessionId"]
        result = MODULE.verify(report, records)
        self.assertTrue(result["passed"], result)
        self.assertEqual(2, result["ignored_unrelated_mapping_count"])
        self.assertEqual([1, 2], [r["native_accepted_execution_id"] for r in result["matched_executions"]])
        self.assertTrue(all(r["native_node_count"] == 10456 for r in result["matched_executions"]))
        self.assertTrue(all(r["native_host_segment_count"] == 48 for r in result["matched_executions"]))

    def test_missing_http_owner_for_current_generation_is_not_warmup(self):
        report, records = fixture()
        records[0]["scope"].pop("RequestId")
        self.assertFalse(MODULE.verify(report, records)["passed"])

    def test_unscoped_mapping_reusing_target_internal_id_cannot_be_ignored(self):
        report, records = fixture()
        conflict = copy.deepcopy(records[0])
        conflict["scope"] = {}
        conflict["props"]["SessionId"] = "different-session"
        records.insert(0, conflict)
        self.assertFalse(MODULE.verify(report, records)["passed"])

    def test_uint64_strings_compare_numerically_across_decimal_width(self):
        report, records = fixture()
        records[2]["props"]["NativeAcceptedExecutionId"] = "9"
        records[3]["props"]["NativeAcceptedExecutionId"] = "10"
        self.assertTrue(MODULE.verify(report, records)["passed"])
        records[3]["props"]["NativeAcceptedExecutionId"] = 9
        self.assertFalse(MODULE.verify(report, records)["passed"])

    def test_native_uint64_identity_rejects_coercions_and_overflow(self):
        for invalid in (True, False, 1.0, "0", "01", "+1", " 1", "1 ", "1e0", "1.0", "١", "-1",
                        str(1 << 64), 1 << 64, "9" * 100):
            with self.subTest(value=invalid):
                report, records = fixture()
                records[2]["props"]["NativeAcceptedExecutionId"] = invalid
                self.assertFalse(MODULE.verify(report, records)["passed"])
        self.assertEqual((1 << 64) - 1, MODULE.native_execution_id(str((1 << 64) - 1)))

    def test_two_native_joint_steps_bind_current_http_and_allow_segmented_execution(self):
        report, records = fixture()
        result = MODULE.verify(report, records)
        self.assertTrue(result["passed"], result)
        self.assertEqual(2, result["joint_steps"])
        self.assertTrue(result["native_segmented_execution_observed"])
        self.assertIn("not", MODULE.__doc__)

    def test_http_success_or_sse_overlap_alone_never_qualifies(self):
        report, records = fixture()
        for selected in ([], records[:2], records[:3]):
            self.assertFalse(MODULE.verify(report, selected)["passed"])

    def test_plan_and_try_events_are_not_native_completion(self):
        report, records = fixture()
        for record in records[2:]:
            record["template"] = "batch.fused.planned id={BatchExecutionId}"
        self.assertFalse(MODULE.verify(report, records)["passed"])

    def test_unrelated_warmup_members_cannot_qualify_current_requests(self):
        report, records = fixture()
        for record in records[2:]:
            record["props"]["SequenceRequestIdsJson"] = '["warm-a","warm-b"]'
        self.assertFalse(MODULE.verify(report, records)["passed"])

    def test_unpublished_native_result_is_rejected(self):
        report, records = fixture()
        records[2]["props"]["ManagedPublishCompleted"] = False
        self.assertFalse(MODULE.verify(report, records)["passed"])

    def test_native_actual_graph_values_cannot_be_inferred_or_bool(self):
        for key, values in {"NativeNodeCount": (0, None, True), "NativeGraphCount": (0, 2, True),
                            "NativeHostSegmentCount": (-1, None, True),
                            "NativeExecutionKind": (-1, 0, None, True),
                            "NativeAcceptedExecutionId": (0, None, True)}.items():
            for value in values:
                with self.subTest(key=key, value=value):
                    report, records = fixture()
                    records[2]["props"][key] = value
                    self.assertFalse(MODULE.verify(report, records)["passed"])

    def test_repeated_or_reversed_native_and_batch_identity_rejected(self):
        for key, value in (("BatchExecutionId", 1), ("NativeAcceptedExecutionId", 11), ("NativeAcceptedExecutionId", 9)):
            report, records = fixture()
            records[3]["props"][key] = value
            self.assertFalse(MODULE.verify(report, records)["passed"])

    def test_single_member_or_duplicated_sequence_cannot_be_a_joint_graph(self):
        for ids in ('["chat-a"]', '["chat-a","chat-a"]'):
            report, records = fixture()
            records[2]["props"]["SequenceRequestIdsJson"] = ids
            self.assertFalse(MODULE.verify(report, records)["passed"])

    def test_position_width_and_exact_one_step_advance(self):
        for value in ('[101]', '[102,201]', '[true,201]', '[100,201]'):
            report, records = fixture()
            records[2]["props"]["PositionsAfterJson"] = value
            self.assertFalse(MODULE.verify(report, records)["passed"])

    def test_serial_step_between_claimed_joint_steps_fails(self):
        report, records = fixture()
        records[3]["props"]["PositionsBeforeJson"] = '[102,202]'
        records[3]["props"]["PositionsAfterJson"] = '[103,203]'
        self.assertFalse(MODULE.verify(report, records)["passed"])

    def test_real_engagement_and_continuity_fail_remain_separate(self):
        report, records = fixture()
        later = copy.deepcopy(records[-1])
        later["props"].update(BatchExecutionId=3, NativeAcceptedExecutionId="13",
                              PositionsBeforeJson='[103,203]', PositionsAfterJson='[104,204]')
        records.append(later)
        final = copy.deepcopy(later)
        final["props"].update(BatchExecutionId=4, NativeAcceptedExecutionId="14",
                              PositionsBeforeJson='[104,204]', PositionsAfterJson='[105,205]')
        records.append(final)
        result = MODULE.verify(report, records)
        self.assertTrue(result["actual_joint_engagement_verified"])
        self.assertFalse(result["full_interval_continuity_verified"])
        self.assertFalse(result["batched_fused_graph_verified"])
        self.assertFalse(result["passed"])
        self.assertEqual(4, result["joint_steps"])
        self.assertEqual([2, 2], [run["steps"] for run in result["contiguous_joint_runs"]])
        self.assertEqual(1, len(result["errors"]))
        self.assertEqual(1, len(result["continuity_gaps"]))

    def test_replayed_generation_position_is_invalid_not_a_new_interval(self):
        report, records = fixture()
        records[-1]["props"].update(PositionsBeforeJson='[100,200]', PositionsAfterJson='[101,201]')
        result = MODULE.verify(report, records)
        self.assertFalse(result["actual_joint_engagement_verified"])
        self.assertFalse(result["passed"])

    def test_one_internal_request_cannot_map_to_two_http_owners(self):
        report, records = fixture()
        conflict = copy.deepcopy(records[0])
        conflict["scope"]["RequestId"] = "unrelated"
        records.insert(1, conflict)
        self.assertFalse(MODULE.verify(report, records)["passed"])

    def test_current_session_identity_must_match_mapping(self):
        report, records = fixture()
        records[0]["props"]["SessionId"] = "wrong-session"
        self.assertFalse(MODULE.verify(report, records)["passed"])

    def test_legacy_scope_supported_but_invalid_canonical_is_not_overridden(self):
        report, records = fixture()
        records[0]["scopes"] = [records[0].pop("scope")]
        self.assertTrue(MODULE.verify(report, records)["passed"])
        records[0]["scope"] = []
        self.assertFalse(MODULE.verify(report, records)["passed"])

    def test_failed_http_or_nonunified_cannot_qualify(self):
        for mutate in (lambda r: r.update(completed=False), lambda r: r["model"].update(unifiedMemory=False),
                       lambda r: r["concurrent"][0]["terminal"].update(aborted=True),
                       lambda r: r["concurrent"][0].update(cleanup_error="not removed")):
            report, records = fixture()
            mutate(report)
            self.assertFalse(MODULE.verify(report, records)["passed"])

    def test_graph_engagement_is_separate_from_output_parity(self):
        report, records = fixture()
        report.update(passed=False, greedy_text_parity=[False, True])
        result = MODULE.verify(report, records)
        self.assertTrue(result["passed"])
        self.assertFalse(result["source_report_passed"])
        self.assertEqual([False, True], result["source_greedy_text_parity"])

    def test_json_arrays_supported_and_same_http_for_both_rejected(self):
        report, records = fixture()
        for record in records[2:]:
            for key in ("SequenceRequestIdsJson", "PositionsBeforeJson", "PositionsAfterJson"):
                record["props"][key] = json.loads(record["props"][key])
        self.assertTrue(MODULE.verify(report, records)["passed"])
        report["concurrent"][1]["request_id"] = report["concurrent"][0]["request_id"]
        self.assertFalse(MODULE.verify(report, records)["passed"])

    def test_minimum_two_and_zero_host_segments_supported(self):
        report, records = fixture()
        for record in records[2:]:
            record["props"]["NativeHostSegmentCount"] = 0
            record["props"]["NativeExecutionKind"] = 0
        self.assertTrue(MODULE.verify(report, records)["passed"])
        self.assertFalse(MODULE.verify(report, records, 1)["passed"])
        self.assertFalse(MODULE.verify(report, records, 3)["passed"])

    def test_different_generation_pairs_cannot_pool_single_steps(self):
        report, records = fixture()
        for key in ("a", "b"):
            records.insert(0, {"template": "generation.sequence.bound", "scope": {"RequestId": "http-" + key},
                               "props": {"InternalRequestId": "next-" + key, "SessionId": "session-" + key}})
        records[-1]["props"]["SequenceRequestIdsJson"] = '["next-a","next-b"]'
        self.assertFalse(MODULE.verify(report, records)["passed"])


if __name__ == "__main__":
    unittest.main()
