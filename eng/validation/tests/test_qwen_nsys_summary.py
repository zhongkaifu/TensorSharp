import importlib.util
from pathlib import Path
import sqlite3
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("qwen_nsys_summary", ROOT / "qwen-nsys-summary.py")
summary = importlib.util.module_from_spec(spec)
spec.loader.exec_module(summary)
spec = importlib.util.spec_from_file_location("qwen_nsys_profile", ROOT / "qwen-nsys-profile.py")
profile = importlib.util.module_from_spec(spec)
spec.loader.exec_module(profile)


class ProfileTests(unittest.TestCase):
    def test_union_handles_overlap_nested_and_invalid_intervals(self):
        self.assertEqual(15, summary.union_ns([(0, 10), (3, 5), (8, 12), (20, 23), (50, 49)]))
        self.assertEqual(0, summary.union_ns([]))

    def test_gpu_union_copy_bytes_and_cpu_api_are_separate(self):
        db = sqlite3.connect(":memory:")
        db.executescript('''
          CREATE TABLE StringIds (id INTEGER, value TEXT);
          INSERT INTO StringIds VALUES (1, 'nvfp4_gemm'), (2, 'gated_delta_net'), (3, 'cudaStreamSynchronize');
          CREATE TABLE CUPTI_ACTIVITY_KIND_KERNEL (start INTEGER, end INTEGER, demangledName INTEGER);
          INSERT INTO CUPTI_ACTIVITY_KIND_KERNEL VALUES (1000000,5000000,1), (3000000,6000000,2);
          CREATE TABLE CUPTI_ACTIVITY_KIND_MEMCPY (start INTEGER, end INTEGER, copyKind INTEGER, bytes INTEGER);
          INSERT INTO CUPTI_ACTIVITY_KIND_MEMCPY VALUES (7000000,9000000,2,4096);
          CREATE TABLE CUPTI_ACTIVITY_KIND_RUNTIME (start INTEGER, end INTEGER, nameId INTEGER);
          INSERT INTO CUPTI_ACTIVITY_KIND_RUNTIME VALUES (0,10000000,3);
        ''')
        r = summary.summarize(db, {"result": {"e2e_ms": 15}, "server_process_cpu": {"user_ms": 3, "system_ms": 1}})
        self.assertEqual(7, r["gpu_activity_union_ms"])
        self.assertEqual(8, r["gpu_first_to_last_span_ms"])
        self.assertEqual(1, r["gpu_gap_between_first_and_last_ms"])
        self.assertEqual(4096, r["gpu_categories"]["memcpy_DtoH"]["bytes"])
        self.assertEqual(4, r["gpu_categories"]["matrix_multiply"]["sum_ms"])
        self.assertEqual(10, r["host_api_tables"]["CUPTI_ACTIVITY_KIND_RUNTIME"]["inclusive_ms"])
        self.assertEqual(8, r["request_wall_minus_gpu_union_ms"])
        db.close()

    def test_whole_graph_spans_are_unioned_with_nested_kernels_and_transfers(self):
        with sqlite3.connect(":memory:") as db:
            db.executescript('''
              CREATE TABLE StringIds (id INTEGER, value TEXT);
              INSERT INTO StringIds VALUES (1,'nvfp4_gemm'),(2,'gdn_scan'),(3,'cuGraphLaunch');
              CREATE TABLE CUPTI_ACTIVITY_KIND_KERNEL (start INTEGER,end INTEGER,demangledName INTEGER,graphId INTEGER,graphNodeId INTEGER);
              INSERT INTO CUPTI_ACTIVITY_KIND_KERNEL VALUES (0,4000000,1,NULL,NULL),(6000000,8000000,2,NULL,NULL);
              CREATE TABLE CUPTI_ACTIVITY_KIND_GRAPH_TRACE (start INTEGER,end INTEGER,deviceId INTEGER,contextId INTEGER,
                greenContextId INTEGER,streamId INTEGER,correlationId INTEGER,globalPid INTEGER,graphId INTEGER,graphExecId INTEGER);
              INSERT INTO CUPTI_ACTIVITY_KIND_GRAPH_TRACE VALUES
                (2000000,12000000,0,1,0,13,1,123,1,2),(5000000,9000000,0,1,0,13,2,123,1,2),
                (10000000,20000000,0,1,0,13,3,123,1,2),(30000000,31000000,0,1,0,13,4,123,1,2);
              CREATE TABLE CUPTI_ACTIVITY_KIND_MEMCPY (start INTEGER,end INTEGER,copyKind INTEGER,bytes INTEGER);
              INSERT INTO CUPTI_ACTIVITY_KIND_MEMCPY VALUES (19000000,22000000,1,4096);
              CREATE TABLE CUPTI_ACTIVITY_KIND_MEMSET (start INTEGER,end INTEGER,bytes INTEGER);
              INSERT INTO CUPTI_ACTIVITY_KIND_MEMSET VALUES (28000000,30000000,64);
              CREATE TABLE CUPTI_ACTIVITY_KIND_DRIVER (start INTEGER,end INTEGER,nameId INTEGER);
              INSERT INTO CUPTI_ACTIVITY_KIND_DRIVER VALUES (0,500000,3);
            ''')
            r = summary.summarize(db, {"result": {"e2e_ms": 28}})
        self.assertEqual(25, r["gpu_activity_union_ms"])
        self.assertEqual(11, r["gpu_detailed_activity_union_ms"])
        self.assertEqual(6, r["gpu_gap_between_first_and_last_ms"])
        self.assertEqual(3, r["request_wall_minus_gpu_union_ms"])
        self.assertEqual(8, r["gpu_events"])
        self.assertEqual({"kernels": 2, "copies_and_memsets": 2, "graph_aggregates": 4}, r["gpu_event_record_counts"])
        graph = r["cuda_graph_trace"]
        self.assertEqual(25, graph["aggregate_inclusive_sum_ms"])
        self.assertEqual(19, graph["aggregate_interval_union_ms"])
        self.assertEqual(14, graph["aggregate_interval_without_individual_activity_ms"])
        self.assertTrue(graph["aggregate_only"])
        self.assertEqual("UNAVAILABLE_AGGREGATE_ONLY", graph["per_kernel_body_attribution"])
        self.assertIs(graph["per_kernel_body_attribution_complete"], False)
        self.assertEqual(1, graph["host_graph_launch_api_records"])
        self.assertTrue(r["warnings"])
        category = r["gpu_categories"]["cuda_graph_aggregate"]
        self.assertEqual(25, category["sum_ms"])
        self.assertIs(category["additive_with_kernel_or_copy_categories"], False)
        self.assertEqual(2, len(r["top_kernels"]))
        self.assertIn("not CPU execution", r["request_wall_minus_gpu_union_interpretation"])

    def test_graph_only_capture_has_device_coverage_and_missing_kernel_attribution(self):
        with sqlite3.connect(":memory:") as db:
            db.executescript('''
              CREATE TABLE CUPTI_ACTIVITY_KIND_GRAPH_TRACE (start INTEGER,end INTEGER);
              INSERT INTO CUPTI_ACTIVITY_KIND_GRAPH_TRACE VALUES (1000000,6000000);
            ''')
            r = summary.summarize(db)
        self.assertEqual(5, r["gpu_activity_union_ms"])
        self.assertIsNone(r["gpu_detailed_activity_union_ms"])
        self.assertTrue(r["cuda_graph_trace"]["aggregate_only"])
        self.assertEqual(5, r["cuda_graph_trace"]["aggregate_interval_without_individual_activity_ms"])
        self.assertIn("GPU kernel table absent", r["missing"])

    def test_some_identified_node_records_do_not_certify_complete_graph_attribution(self):
        with sqlite3.connect(":memory:") as db:
            db.executescript('''
              CREATE TABLE CUPTI_ACTIVITY_KIND_GRAPH_TRACE (start INTEGER,end INTEGER);
              INSERT INTO CUPTI_ACTIVITY_KIND_GRAPH_TRACE VALUES (0,10000000);
              CREATE TABLE CUPTI_ACTIVITY_KIND_KERNEL (start INTEGER,end INTEGER,graphId INTEGER,graphNodeId INTEGER);
              INSERT INTO CUPTI_ACTIVITY_KIND_KERNEL VALUES (1000000,3000000,5,7),(4000000,5000000,5,0);
            ''')
            r = summary.summarize(db)
        self.assertEqual(10, r["gpu_activity_union_ms"])
        self.assertEqual(1, r["cuda_graph_trace"]["kernel_records_with_positive_graph_node_id"])
        self.assertFalse(r["cuda_graph_trace"]["aggregate_only"])
        self.assertEqual("NODE_RECORDS_PRESENT_COMPLETENESS_NOT_ESTABLISHED", r["cuda_graph_trace"]["per_kernel_body_attribution"])
        self.assertIsNone(r["cuda_graph_trace"]["per_kernel_body_attribution_complete"])

    def test_host_graph_launch_without_device_records_warns_about_missing_body(self):
        with sqlite3.connect(":memory:") as db:
            db.executescript('''
              CREATE TABLE StringIds (id INTEGER,value TEXT);
              INSERT INTO StringIds VALUES (1,'cuGraphLaunch'),(2,'w8_head');
              CREATE TABLE CUPTI_ACTIVITY_KIND_DRIVER (start INTEGER,end INTEGER,nameId INTEGER);
              INSERT INTO CUPTI_ACTIVITY_KIND_DRIVER VALUES (0,500000,1);
              CREATE TABLE CUPTI_ACTIVITY_KIND_KERNEL (start INTEGER,end INTEGER,demangledName INTEGER);
              INSERT INTO CUPTI_ACTIVITY_KIND_KERNEL VALUES (2000000,3000000,2);
            ''')
            r = summary.summarize(db, {"result": {"e2e_ms": 20}})
        self.assertEqual(1, r["gpu_activity_union_ms"])
        self.assertEqual("MISSING_GRAPH_DEVICE_COVERAGE", r["cuda_graph_trace"]["per_kernel_body_attribution"])
        self.assertIs(r["cuda_graph_trace"]["per_kernel_body_attribution_complete"], False)
        self.assertTrue(r["warnings"])
        self.assertIn("not CPU execution", r["request_wall_minus_gpu_union_interpretation"])

    def test_unsupported_or_invalid_graph_intervals_remain_explicitly_unavailable(self):
        with sqlite3.connect(":memory:") as db:
            db.execute("CREATE TABLE CUPTI_ACTIVITY_KIND_GRAPH_TRACE (begin INTEGER,finish INTEGER)")
            r = summary.summarize(db)
        self.assertIsNone(r["gpu_activity_union_ms"])
        self.assertEqual("UNSUPPORTED_GRAPH_INTERVAL_SCHEMA", r["cuda_graph_trace"]["per_kernel_body_attribution"])
        self.assertTrue(r["missing"]); self.assertTrue(r["warnings"])
        with sqlite3.connect(":memory:") as db:
            db.executescript('''
              CREATE TABLE CUPTI_ACTIVITY_KIND_GRAPH_TRACE (start INTEGER,end INTEGER);
              INSERT INTO CUPTI_ACTIVITY_KIND_GRAPH_TRACE VALUES (0,0),(3,2),(NULL,5),(1000000,4000000);
            ''')
            r = summary.summarize(db)
        self.assertEqual(3, r["gpu_activity_union_ms"])
        self.assertEqual(3, r["cuda_graph_trace"]["invalid_intervals_omitted"])
        self.assertTrue(r["warnings"])

    def test_missing_gpu_is_unavailable_not_zero(self):
        with sqlite3.connect(":memory:") as db:
            r = summary.summarize(db)
        self.assertIsNone(r["gpu_activity_union_ms"])
        self.assertTrue(r["missing"])
        self.assertEqual("quantization", summary.category("dequantize_row_nvfp4_cuda"))

    def test_proc_counters_keep_cpu_time_and_count_minor_major_faults(self):
        fields = ["0"] * 22
        fields[0] = "R"
        fields[7], fields[9] = "420", "3"
        fields[11], fields[12] = "12", "7"
        with patch.object(profile.Path, "read_text", return_value="123 (worker name) " + " ".join(fields)), \
                patch.object(profile.os, "sysconf", return_value=100):
            counters = profile.process_counters(123)
        self.assertEqual({"user_ms": 120.0, "system_ms": 70.0, "minor_faults": 420, "major_faults": 3}, counters)

    def test_fixture_selection_checks_digest_and_records_cap_changes(self):
        body = {"max_tokens": 128, "messages": []}
        row = {"scenario": "repeat19k", "lane": 0, "step": 1, "body": body,
               "request_sha256": profile.replay.digest(body)}
        selected = profile.select_row([row], "repeat19k", 0, 1, 1)
        self.assertEqual(128, row["body"]["max_tokens"])
        self.assertEqual(1, selected["body"]["max_tokens"])
        self.assertNotEqual(row["request_sha256"], selected["request_sha256"])
        row["request_sha256"] = "wrong"
        with self.assertRaises(ValueError):
            profile.select_row([row], "repeat19k", 0, 1)
        with self.assertRaises(ValueError):
            profile.select_row([], "repeat19k", 0, 1)


if __name__ == "__main__":
    unittest.main()
