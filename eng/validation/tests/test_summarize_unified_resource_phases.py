import importlib.util
from pathlib import Path
import unittest

spec = importlib.util.spec_from_file_location("resource_phases", Path(__file__).parents[1] /
    "summarize-unified-resource-phases.py")
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


class ResourcePhaseTests(unittest.TestCase):
    def test_http_boundaries_use_exact_ids_not_other_traffic(self):
        report = {"concurrent": [{"name": key, "request_id": key} for key in ("a", "b")]}
        records = []
        for key in ("a", "b", "unrelated"):
            for event, second in (("HttpRequestStarted", "01"), ("HttpRequestCompleted", "05")):
                records.append({"scope": {"RequestId": key}, "event": {"name": event},
                    "ts": f"2026-10-11T00:00:{second}+00:00",
                    "props": {"Method": "POST", "Path": "/api/chat", "StatusCode": 200}})
        self.assertEqual(["a", "b"], [r["request_id"] for r in module.request_windows(report, records, "concurrent")])
        records.append(records[0])
        with self.assertRaisesRegex(ValueError, "duplicate"):
            module.request_windows(report, records, "concurrent")

    def test_missing_end_or_unzoned_time_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "timezone"):
            module.timestamp("2026-10-11T00:00:01")
        with self.assertRaisesRegex(ValueError, "Missing"):
            module.request_windows({"serial": [{"request_id": "a"}, {"request_id": "b"}]}, [], "serial")

    def test_dotnet_seven_digit_timestamps_work_on_python310(self):
        self.assertEqual(module.timestamp("2026-10-11T11:21:58.988467+00:00"),
                         module.timestamp("2026-10-11T11:21:58.9884679+00:00"))

    def test_only_current_samples_are_used_not_lifetime_peak_or_neighbors(self):
        samples = [{"unix": t, "rss": rss, "private": rss + 1, "peak_rss": 99999,
                    "board_gpu": f"GPU-one, {rss}, 1000, 2"} for t, rss in ((0, 900), (2, 20), (3, 30), (5, 800))]
        result = module.observations(samples, 1, 4)
        self.assertEqual(2, result["sample_count"])
        self.assertEqual(30, result["process_sampled_peak_rss_bytes"])
        self.assertEqual(30, result["whole_board_sampled_peak_used_mib"])
        self.assertEqual(1, result["first_sample_after_start_seconds"])
        self.assertEqual(1, result["last_sample_before_end_seconds"])

    def test_unknown_resources_remain_null_and_zero_is_measured(self):
        result = module.observations([{"unix": 2, "rss": None, "private": float("nan"), "error": "unavailable"}], 1, 3)
        self.assertIsNone(result["process_sampled_peak_rss_bytes"])
        self.assertIsNone(result["process_sampled_peak_private_bytes"])
        self.assertIsNone(result["whole_board_sampled_peak_used_mib"])
        self.assertEqual(0, module.observations([{"unix": 2, "rss": 0}], 1, 3)["process_sampled_peak_rss_bytes"])

    def test_empty_interval_or_multiple_boards_does_not_invent_single_board_peak(self):
        self.assertIsNone(module.observations([], 1, 3)["whole_board_sampled_peak_used_mib"])
        result = module.observations([{"unix": 2, "board_gpu": "GPU-a, 20, 0, 0\nGPU-b, 30, 0, 0"}], 1, 3)
        self.assertIsNone(result["whole_board_sampled_peak_used_mib"])
        self.assertEqual(2, len(result["whole_board_devices"]))

    def test_reused_pid_or_nonmonotonic_samples_are_rejected(self):
        owner = {"pid": 10, "startTicks": 621355968000000000 + 10000000}
        samples = [{"unix": 2, "created": 1, "pid": 10}]
        module.validate_samples(samples, owner)
        samples[0]["created"] = 1.1
        with self.assertRaisesRegex(ValueError, "identity"):
            module.validate_samples(samples, owner)
        samples = [{"unix": t, "created": 1, "pid": 10} for t in (3, 2)]
        with self.assertRaisesRegex(ValueError, "nonmonotonic"):
            module.validate_samples(samples, owner)


if __name__ == "__main__":
    unittest.main()
