import copy
import importlib.util
import json
from pathlib import Path
import unittest

spec = importlib.util.spec_from_file_location("report", Path(__file__).resolve().parents[1] / "multimodel-quality-report.py")
report = importlib.util.module_from_spec(spec)
spec.loader.exec_module(report)


def fixture():
    response = dict(done=True, done_reason="stop", message=dict(content="262"),
                    prompt_eval_count=100, eval_count=3, prompt_eval_duration=10_000_000,
                    eval_duration=20_000_000, total_duration=40_000_000, prompt_cache_hit_tokens=0)
    request = dict(messages=[dict(role="user", content=report.bench.PROMPTS["long_extract"])],
                   options=dict(temperature=0, top_k=0, top_p=1, min_p=0, repeat_penalty=1,
                                presence_penalty=0, frequency_penalty=0, seed=17, stop=[]),
                   think=False, stream=False, multi_agent=False, skills=[], skills_discovery=False)
    row = dict(case="long_extract", repetition=0, quality_passed=True, http_status=200,
               response=response, raw_response=json.dumps(response), metrics=report.bench.metrics(response), request=request)
    return dict(run_complete=True, loaded_native=dict(sha256="a"*64), binaries={"GgmlOps.dll": dict(sha256="a"*64)}, records=[row])


class ReportTests(unittest.TestCase):
    def test_linux_native_identity_is_checked(self):
        data = fixture()
        data['native_name'] = 'libGgmlOps.so'
        data['binaries'] = {'libGgmlOps.so': data['binaries']['GgmlOps.dll']}
        self.assertEqual(report.verify(data, ['long_extract'], 1)['long_extract']['quality_passes'], 1)
        data['loaded_native']['sha256'] = 'b' * 64
        with self.assertRaisesRegex(ValueError, 'native identity'):
            report.verify(data, ['long_extract'], 1)

    def test_recomputes_quality_and_timings(self):
        result = report.verify(fixture(), ["long_extract"], 1)
        self.assertEqual(result["long_extract"]["quality_passes"], 1)
        self.assertEqual(result["long_extract"]["first"]["prefill_tps"], 10000)

    def test_rejects_incomplete_or_duplicate_evidence(self):
        for mutation in ("duplicate", "missing", "incomplete", "native"):
            data = fixture()
            if mutation == "duplicate": data["records"] *= 2
            if mutation == "missing": data["records"] = []
            if mutation == "incomplete": data["run_complete"] = False
            if mutation == "native": data["loaded_native"]["sha256"] = "b"*64
            with self.subTest(mutation=mutation), self.assertRaises(ValueError):
                report.verify(data, ["long_extract"], 1)

    def test_rejects_forged_quality_timing_prompt_and_sampler(self):
        for mutation in ("quality", "timing", "prompt", "sampler", "raw"):
            data = fixture(); row = data["records"][0]
            if mutation == "quality": row["quality_passed"] = False
            if mutation == "timing": row["metrics"]["decode_tps"] *= 2
            if mutation == "prompt": row["request"]["messages"][0]["content"] = "Say 262"
            if mutation == "sampler": row["request"]["options"]["repeat_penalty"] = 2
            if mutation == "raw": row["raw_response"] = "{}"
            with self.subTest(mutation=mutation), self.assertRaises(ValueError):
                report.verify(data, ["long_extract"], 1)

    def test_validated_evidence_can_still_fail_quality(self):
        data = fixture(); row = data["records"][0]
        row["response"]["done_reason"] = "length"
        row["raw_response"] = json.dumps(row["response"])
        row["quality_passed"] = False
        result = report.verify(data, ["long_extract"], 1)
        self.assertEqual(result["long_extract"]["quality_passes"], 0)


if __name__ == "__main__":
    unittest.main()
