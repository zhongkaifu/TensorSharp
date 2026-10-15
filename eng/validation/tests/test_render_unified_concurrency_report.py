import hashlib
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest


spec = importlib.util.spec_from_file_location("render_concurrency", Path(__file__).parents[1] /
                                              "render-unified-concurrency-report.py")
renderer = importlib.util.module_from_spec(spec)
spec.loader.exec_module(renderer)


class ReportTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        (self.root / "concurrency").mkdir()
        self.report = {"passed": True, "actual_joint_engagement_verified": True,
            "full_interval_continuity_verified": True, "greedy_text_parity": [True, True],
            "fused_joint_steps": 150, "fused_continuity_gaps": [], "concurrent": [], "serial": [],
            "model": {"loaded": "model", "startupWarmup": {"completed": True,
                "elapsedMilliseconds": 1200, "mode": "scheduled"}}}
        self.summary = {"schema_version": 2, "acceptance_passed": True,
            "actual_joint_engagement_verified": True, "full_interval_continuity_verified": True,
            "greedy_text_parity": [True, True], "requests": []}
        for mode in ("concurrent", "serial"):
            for index in range(2):
                name = f"{mode}-{index}"
                text = f"{name}: <script>alert('not code')</script> & 完整原文\n尾部"
                raw = {"name": name, "request_id": name, "text": text,
                    "request": {"messages": [{"content": "<img src=x onerror=alert(1)>"}], "maxTokens": 160},
                    "terminal": {"truncated": True},
                    "phase_observations": {"content_frames": 999}}
                self.report[mode].append(raw)
                self.summary["requests"].append({"mode": mode, "name": name, "text": text,
                    "ttft_seconds": 1.5, "uncached_prompt_tokens": 17, "reused_prompt_tokens": 1462,
                    "recorded_pipeline_output_tokens": 160, "attributed_prefill_seconds": 2,
                    "attributed_prefill_tokens_per_second": 8.5,
                    "pipeline_tokens_per_first_content_to_terminal_second": 10,
                    "attributed_decode_tokens_per_second": 20, "attributed_decode_seconds": 8,
                    "elapsed_seconds": 18, "truncated": True, "finish_reason": "max_tokens"})
        self.audit = {"passed": False, "errors": ["Prefix checker failed: 2"],
                      "checkerExit": 0, "prefixCheckerExit": 2, "cleanupComplete": True,
                      "binariesBefore": {"model.dll": "a"}, "binariesAfter": {"model.dll": "a"}}

    def write(self):
        raw = json.dumps(self.report, ensure_ascii=False).encode("utf-8")
        (self.root / "concurrency" / "report.json").write_bytes(raw)
        self.summary["source_sha256"] = hashlib.sha256(raw).hexdigest()
        (self.root / "summary.json").write_text(json.dumps(self.summary, ensure_ascii=False), encoding="utf-8")
        (self.root / "launch-audit.json").write_text(json.dumps(self.audit, ensure_ascii=False), encoding="utf-8-sig")

    def render(self):
        self.write()
        return renderer.render(self.root)

    def test_core_pass_is_separate_from_wrapper_failure_and_unrun_http(self):
        result = self.render()
        self.assertIn('核心并发文本验收</td><td><span class="pass">通过', result)
        self.assertIn('启动器及所有附加检查总结果</td><td><span class="fail">未通过', result)
        self.assertIn('附加 HTTP checker</td><td>未执行或未记录', result)
        self.assertIn("Prefix checker failed: 2", result)

    def test_four_complete_answers_and_prompts_are_escaped(self):
        result = self.render()
        self.assertEqual(4, result.count('<pre class="model-output">'))
        self.assertEqual(4, result.count("完整原文\n尾部"))
        self.assertNotIn("<script>", result)
        self.assertNotIn("<img ", result)
        self.assertIn("&lt;script&gt;", result)
        self.assertIn("max_tokens", result)
        self.assertIn("不评分回答事实", result)

    def test_uncached_pp_and_two_decode_rates_are_distinct(self):
        result = self.render()
        self.assertIn("17／1,462", result)
        self.assertIn("2.000／8.50", result)
        self.assertIn("<td>160</td><td>10.00</td><td>20.00</td>", result)
        self.assertNotIn("<td>999</td>", result)  # Hashes may incidentally contain 999.

    def test_unknown_resource_observations_stay_unknown(self):
        result = self.render()
        self.assertEqual(2, result.count("未知 GiB"))
        self.assertIn("未知 MiB", result)
        self.assertNotIn("0.000 GiB", result)
        for value in (None, True, float("nan"), float("inf"), -1):
            self.assertEqual("未知", renderer.number(value))
        self.assertEqual("0.00", renderer.number(0))

    def test_missing_identity_is_unknown_not_equal_empty_maps(self):
        self.audit["binariesBefore"] = self.audit["binariesAfter"] = {}
        self.assertIn('部署二进制前后哈希一致</td><td><span class="unknown">', self.render())

    def test_changed_source_fails_before_render(self):
        self.write()
        (self.root / "concurrency" / "report.json").write_text("{}", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "SHA-256"):
            renderer.render(self.root)

    def test_summary_text_mismatch_or_omission_rejected(self):
        self.summary["requests"][0]["text"] = "stale answer"
        with self.assertRaisesRegex(ValueError, "answer differ"):
            self.render()
        self.summary["requests"] = self.summary["requests"][1:]
        with self.assertRaisesRegex(ValueError, "omits"):
            self.render()

    def test_acceptance_mismatch_rejected(self):
        self.summary["greedy_text_parity"] = [False, True]
        with self.assertRaisesRegex(ValueError, "acceptance mismatch"):
            self.render()

    def test_missing_positive_proof_remains_unknown(self):
        del self.report["actual_joint_engagement_verified"]
        self.assertIn('本次请求实际联合逻辑图</td><td><span class="unknown">', self.render())

    def test_extended_shared_window_disclosure_and_no_input_mutation(self):
        self.audit["extendedScope"] = "same process: startup, short, extended, checks"
        self.write()
        paths = [self.root / "summary.json", self.root / "concurrency" / "report.json",
                 self.root / "launch-audit.json"]
        before = [p.read_bytes() for p in paths]
        renderer.main([str(self.root)])
        result = (self.root / "report.html").read_text(encoding="utf-8")
        self.assertIn(self.audit["extendedScope"], result)
        self.assertIn("本进程还有扩展请求", result)
        self.assertIn("本页请求表格只展示自身原始报告", result)
        self.assertNotIn("本页扩展请求", result)
        self.assertEqual(before, [p.read_bytes() for p in paths])
        self.assertIn(renderer.digest(renderer.__file__), result)

    def test_short_and_extended_checker_exit_codes_stay_separate(self):
        self.audit["extendedCheckerExit"] = 1
        result = self.render()
        self.assertIn('主要短请求 checker</td><td><span class="pass">通过</span>（退出码 0）', result)
        self.assertIn('扩展请求 checker</td><td><span class="fail">未通过</span>（退出码 1）', result)
        self.assertNotIn("核心 checker 退出码", result)

    def test_unrun_extended_checker_remains_unknown(self):
        self.assertIn('扩展请求 checker</td><td>未执行或未记录', self.render())

    def test_continuity_is_limited_to_observed_joint_interval(self):
        result = self.render()
        self.assertIn('首末已记录联合步之间连续</td><td><span class="pass">通过', result)
        self.assertIn('不能证明整个共同生命周期均在合批', result)
        self.assertNotIn('共同解码区间连续性</td>', result)


if __name__ == "__main__":
    unittest.main()
