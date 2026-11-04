"""Historical fixture, original-scorer and fail-fast contracts. No live HTTP/GPU."""
import copy
import http.client
import importlib.util
import io
import json
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import patch

SOURCE = Path(__file__).resolve().parents[1] / "ninfer-api-regression-smoke.py"
spec = importlib.util.spec_from_file_location("ninfer_regression_smoke_test", SOURCE)
smoke = importlib.util.module_from_spec(spec)
spec.loader.exec_module(smoke)


def observation(content):
    return {"httpStatus": 200, "httpBodyComplete": True, "doneSeen": True,
            "finishReasons": ["stop"], "content": content, "reasoning": "", "toolCallsSeen": False,
            "multipleChoices": False, "error": None}


class Fixtures(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.fixture, cls.bench, cls.audit = smoke.load_fixture()

    def test_every_prompt_and_setting_replays_the_original_generator(self):
        args = SimpleNamespace(nonce_seed=20261101, quality_tasks=8, answer_tokens=512,
                               temperature=0.0, top_p=1.0, sampling_seed=1234, keep_text=65536)
        captures = {}
        bench = self.bench
        for repeat in (1, 2):
            class Client:
                engine = "tensorsharp"
                model = self.fixture["source"]["baselineModel"]
                extra = {}

                def __init__(self):
                    self.args = args
                    self.bodies = []

                def chat(self, messages, maximum, extra=None):
                    self.bodies.append(bench.Client.body(self, messages, maximum, extra))
                    return {"content": "{}", "finish_reason": "stop", "error": None}

            client = Client()
            result = bench.run_quality_tasks(bench.Ctx(args, client, {}), repeat)
            captures[repeat] = (client.bodies, result["notes"]["expected"])
        for row in self.fixture["cases"]:
            bodies, expected = captures[row["repeat"]]
            self.assertTrue(self.audit.strict_equal(row["requestBody"], bodies[row["taskIndex"]]))
            truth = row["selectedOldTruth"]
            self.assertTrue(self.audit.strict_equal(truth["expected"], expected[row["taskIndex"]][truth["field"]]))

    def test_exact_five_old_truths_include_partial_fields_of_failed_tasks(self):
        self.assertEqual([513, 1470, [-787, -432, -365, -48, 80, 108, 324, 901, 926], 4649, 2440],
                         [r["selectedOldTruth"]["expected"] for r in self.fixture["cases"]])
        self.assertEqual(4, sum(not r["baselineProof"]["wholeTaskCheck"]["ok"] for r in self.fixture["cases"]))
        for row in self.fixture["cases"]:
            result = smoke.assess(observation(row["baselineProof"]["content"]), row["selectedOldTruth"], self.audit)
            self.assertTrue(result["passed"])

    def test_wrong_field_type_missing_field_and_json_repair_do_not_pass(self):
        truth = self.fixture["cases"][0]["selectedOldTruth"]
        for content in ('{"result":512}', '{"result":"513"}', '{"result":513.0}', '{}',
                        '```json\n{"result":513}\n```', 'The answer is 513.', '{"result":NaN}'):
            result = smoke.assess(observation(content), truth, self.audit)
            self.assertFalse(result["passed"])
            json.dumps(result, allow_nan=False)

    def test_original_generated_json_parser_behavior_and_selected_only_scope(self):
        truth = self.fixture["cases"][0]["selectedOldTruth"]
        result = smoke.assess(observation('{"result":0,"result":513,"sum":-999}'), truth, self.audit)
        self.assertTrue(result["passed"])
        sorted_truth = self.fixture["cases"][2]["selectedOldTruth"]
        wrong = list(sorted_truth["expected"]); wrong[0], wrong[1] = wrong[1], wrong[0]
        self.assertFalse(smoke.assess(observation(json.dumps({"sorted": wrong})), sorted_truth, self.audit)["passed"])

    def test_transport_thinking_or_nonstop_cannot_preserve_old_truth(self):
        truth = self.fixture["cases"][0]["selectedOldTruth"]
        for key, value in (("httpStatus", 500), ("httpBodyComplete", False), ("doneSeen", False),
                           ("finishReasons", ["length"]), ("error", "connection reset"),
                           ("reasoning", "hidden reasoning"), ("toolCallsSeen", True), ("multipleChoices", True)):
            current = observation('{"result":513}'); current[key] = value
            self.assertFalse(smoke.assess(current, truth, self.audit)["passed"], key)


class Transport(unittest.TestCase):
    class Response(io.BytesIO):
        status = 200

    def stream(self, chunks, done=True):
        frames = [json.dumps({"choices": [{"index": 0, "delta": {"content": c}, "finish_reason": None}]}, ensure_ascii=False)
                  for c in chunks]
        frames += [json.dumps({"choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]})]
        return ("\n\n".join("data: " + x for x in frames) + ("\n\ndata: [DONE]" if done else "") + "\n\n").encode()

    def test_complete_split_unicode_content_is_preserved_without_timings(self):
        wire = self.stream(['{"result":', '513,"extra":"中"}'])
        observed, raw = smoke.consume_sse(self.Response(wire), 10)
        self.assertEqual(wire, raw)
        self.assertEqual('{"result":513,"extra":"中"}', observed["content"])
        self.assertTrue(observed["httpBodyComplete"] and observed["doneSeen"])
        self.assertEqual(["stop"], observed["finishReasons"])
        self.assertNotIn("ttft_s", observed)

    def test_missing_done_invalid_frame_and_oversized_line_are_not_complete_success(self):
        for wire in (self.stream(['{"result":513}'], done=False), b'data: {broken}\n\n',
                     b'data: ' + b'x' * (smoke.MAX_LINE + 1)):
            observed, _ = smoke.consume_sse(self.Response(wire), 10)
            self.assertFalse(observed["httpBodyComplete"] and observed["doneSeen"] and observed["error"] is None)

    def test_reasoning_and_multiple_choices_are_retained_as_rejections(self):
        frame = {"choices": [{"index": 0, "delta": {"reasoning_content": "thinking", "content": "{}"}},
                             {"index": 1, "delta": {}}]}
        wire = ("data: " + json.dumps(frame) + "\n\ndata: [DONE]\n\n").encode()
        observed, _ = smoke.consume_sse(self.Response(wire), 10)
        self.assertEqual("thinking", observed["reasoning"])
        self.assertTrue(observed["multipleChoices"])

    def test_real_http_response_requires_complete_declared_length_even_after_done(self):
        wire = self.stream(['{"result":513}'])
        class Socket:
            def __init__(self, raw):
                self.raw = io.BytesIO(raw)
            def makefile(self, *args):
                return self.raw
        fixture, _, audit = smoke.load_fixture()
        for missing in (0, 10):
            header = f"HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nContent-Length: {len(wire)+missing}\r\n\r\n".encode()
            response = http.client.HTTPResponse(Socket(header + wire)); response.begin()
            observed, raw = smoke.consume_sse(response, 10)
            self.assertEqual(wire, raw)
            self.assertTrue(observed["doneSeen"])
            self.assertEqual(missing, response.length)
            result = smoke.assess(observed, fixture["cases"][0]["selectedOldTruth"], audit)
            self.assertTrue(result["checks"]["selectedOldTruth"])
            self.assertEqual(missing == 0, result["passed"])
            self.assertEqual(missing == 0, observed["httpBodyComplete"])


class Execution(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.fixture, cls.bench, cls.audit = smoke.load_fixture()

    def manifest(self, backend="cuda"):
        return {"schema": "ninfer-target-server/1", "engine": "TensorSharp/native-cuda", "model": "candidate",
                "argv": ["host", "--model", "/model.ninfer", "--backend", backend, "--port", "5088"],
                "modelsResponse": {"data": [{"id": "candidate"}]}}

    def test_native_backend_model_and_port_are_fail_closed(self):
        for change in (lambda x: x.update(engine="ggml_cuda"), lambda x: x["argv"].__setitem__(4, "ggml_cuda"),
                       lambda x: x.update(model="other"), lambda x: x["argv"].__setitem__(6, "9999")):
            value = self.manifest(); change(value)
            with self.assertRaises(ValueError):
                smoke.validate_manifest(value, "candidate", smoke.endpoint("http://127.0.0.1:5088"))
        smoke.validate_manifest(self.manifest(), "candidate", smoke.endpoint("http://127.0.0.1:5088"))

    def test_plan_never_reads_manifest_or_calls_http(self):
        with tempfile.TemporaryDirectory() as tmp, patch.object(smoke, "query", side_effect=AssertionError("HTTP forbidden")):
            out = Path(tmp).resolve() / "artifacts/plan"
            self.assertEqual(0, smoke.main(["--url", "http://127.0.0.1:5088", "--model", "candidate",
                                            "--manifest", "/unavailable/manifest.json", "--out", str(out)]))
            self.assertFalse(out.exists())

    def run_campaign(self, fail_at=None):
        temporary = tempfile.TemporaryDirectory(); self.addCleanup(temporary.cleanup)
        root = Path(temporary.name); manifest = root / "manifest.json"; raw_manifest = json.dumps(self.manifest()).encode(); manifest.write_bytes(raw_manifest)
        args = SimpleNamespace(out=root / "artifacts/smoke", manifest=manifest, model="candidate",
                               url="http://127.0.0.1:5088", timeout=10)
        requests = []
        def query(url, wire, timeout):
            index = len(requests); requests.append(json.loads(wire)); row = self.fixture["cases"][index]
            content = row["baselineProof"]["content"]
            if index == fail_at:
                parsed = json.loads(content); parsed[row["selectedOldTruth"]["field"]] = None; content = json.dumps(parsed)
            return observation(content), b"stored actual SSE bytes"
        with patch.object(smoke, "query", side_effect=query):
            result = smoke.execute(args, self.fixture, self.bench, self.audit, smoke.endpoint(args.url), raw_manifest)
        final = args.out / "final-result.json"
        self.assertEqual(smoke.sha(final.read_bytes()) + "  final-result.json\n", Path(str(final) + ".sha256").read_text())
        for index, request in enumerate(requests):
            expected = copy.deepcopy(self.fixture["cases"][index]["requestBody"]); expected["model"] = "candidate"
            self.assertTrue(self.audit.strict_equal(request, expected))
        return result, requests

    def test_first_regression_stops_before_remaining_requests(self):
        result, requests = self.run_campaign(fail_at=0)
        self.assertEqual(1, len(requests))
        self.assertFalse(result["complete"] or result["smokePassed"])
        self.assertEqual(["FAIL"] + ["NOT_RUN_FAIL_FAST"] * 4, [r["status"] for r in result["cases"]])

    def test_middle_regression_does_not_run_or_count_remaining_cases_as_passing(self):
        result, requests = self.run_campaign(fail_at=2)
        self.assertEqual(3, len(requests))
        self.assertEqual(["PASS", "PASS", "FAIL", "NOT_RUN_FAIL_FAST", "NOT_RUN_FAIL_FAST"],
                         [r["status"] for r in result["cases"]])
        self.assertFalse(result["allFiveRequestsExecuted"] or result["smokePassed"])

    def test_passing_all_five_does_not_claim_quality_or_performance(self):
        result, requests = self.run_campaign()
        self.assertEqual(5, len(requests))
        self.assertTrue(result["smokePassed"] and result["allFiveRequestsExecuted"])
        self.assertEqual("NOT_RUN", result["qualityQualification"])
        self.assertEqual("NOT_RUN", result["performance"])
        self.assertTrue(result["fullSuiteStillRequired"])


if __name__ == "__main__":
    unittest.main()
