#!/usr/bin/env python3
"""Offline tests: fake HTTP response iterators only; no sockets, server or GPU."""
import copy
import importlib.util
import io
import json
from pathlib import Path
import tempfile
import threading
import types
import unittest
from unittest.mock import patch
import urllib.error

HERE = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("qwen38_cua_control_tested", HERE / "qwen38-cua-control.py")
cua = importlib.util.module_from_spec(spec)
spec.loader.exec_module(cua)


class Response(io.BytesIO):
    status = 200
    headers = {"Content-Type": "text/event-stream"}


def event(delta=None, finish=None, usage=None):
    obj = {"choices": [{"index": 0, "delta": delta or {}, "finish_reason": finish}]}
    if usage is not None: obj["usage"] = usage
    return b"data: " + json.dumps(obj).encode() + b"\n\n"


def response_bytes(arguments, *, tool="submit_observation", usage=True, finish="tool_calls", done=True, content=""):
    args = cua.canonical(arguments)
    data = event({"role": "assistant", "tool_calls": [{"index": 0, "id": "runtime-random-id", "type": "function"}]})
    data += event({"tool_calls": [{"index": 0, "function": {"name": tool}}]})
    for part in (args[:len(args)//2], args[len(args)//2:]):
        data += event({"tool_calls": [{"index": 0, "function": {"arguments": part}}]})
    if content: data += event({"content": content})
    data += event(finish=finish, usage={"prompt_tokens": 19000, "completion_tokens": 61,
        "prompt_tokens_details": {"cached_tokens": 16602}} if usage else None)
    if done: data += b"data:[DONE]\n\n" # space after ':' is optional SSE syntax
    return data


class FixtureMixin:
    @classmethod
    def setUpClass(cls):
        cls.bench = cua.load_bench()
        with patch("urllib.request.urlopen", side_effect=AssertionError("fixture preparation used network")):
            cls.fixture, cls.images = cua.build_fixture(cls.bench, steps=7, prefix_chars=512, window=2)

    def rows(self, scenario, lanes=1, lane=0):
        return [r for r in self.fixture["requests"] if (r["scenario"],r["lanes"],r["lane"]) == (scenario,lanes,lane)]


class FixtureTests(FixtureMixin,unittest.TestCase):
    def test_complete_1_and_2_lane_plan_and_fixed_budget_contract(self):
        rows = self.fixture["requests"]
        self.assertEqual(len(rows),7*3*3)
        self.assertEqual(len({(r["scenario"],r["lanes"],r["lane"],r["step"]) for r in rows}),len(rows))
        self.assertEqual(len(self.fixture["images"]),14)
        for r in rows:
            common=r["common"]
            self.assertEqual(common["max_tokens"],192)
            self.assertEqual(common["temperature"],0)
            self.assertEqual(common["tool_choice"],"auto")
            self.assertEqual(cua.digest(common),r["common_sha256"])
            self.assertEqual(len(common["tools"]),34)
        self.assertFalse(self.fixture["standard_30step_protocol"])

    def test_append_is_exact_prescribed_history_extension(self):
        rows=self.rows("append")
        self.assertEqual({r["immutable_system_sha256"] for r in rows},{rows[0]["immutable_system_sha256"]})
        for previous,current in zip(rows,rows[1:]):
            old=previous["common"]["messages"]
            self.assertEqual(current["common"]["messages"][:len(old)],old)
            self.assertEqual(current["history"]["retained_complete_turns"],current["step"])
        self.assertEqual(sum(isinstance(m.get("content"),list) for m in rows[-1]["common"]["messages"]),7)

    def test_truncation_drops_complete_oldest_turn_not_dangling_tool_messages(self):
        rows=self.rows("truncate")
        self.assertEqual(rows[3]["history"]["dropped_steps"],[0])
        self.assertEqual(rows[6]["history"]["dropped_steps"],[3])
        for r in rows:
            msgs=r["common"]["messages"]
            self.assertEqual(r["history"]["retained_complete_turns"],min(r["step"],2))
            for i in range(1,len(msgs)-1,3):
                self.assertEqual([m["role"] for m in msgs[i:i+3]],["user","assistant","tool"])
                self.assertEqual(msgs[i+1]["tool_calls"][0]["id"],msgs[i+2]["tool_call_id"])

    def test_edit_changes_early_private_history_and_prior_rows_are_immutable(self):
        rows=self.rows("edit")
        self.assertEqual(rows[4]["expected"]["revision"],"R000")
        self.assertEqual(rows[5]["expected"]["revision"],"R001")
        self.assertIn("revision=R000",rows[4]["common"]["messages"][1]["content"][0]["text"])
        self.assertIn("revision=R001",rows[5]["common"]["messages"][1]["content"][0]["text"])
        self.assertTrue(rows[5]["history"]["policy_edit_this_step"])
        self.assertEqual(rows[5]["history"]["equal_leading_messages_vs_prior_request"],1)
        self.assertEqual(rows[4]["immutable_system_sha256"],rows[5]["immutable_system_sha256"])

    def test_private_scopes_differ_by_case_and_lane_but_images_are_reused(self):
        rows=[r for r in self.fixture["requests"] if r["step"]==0]
        self.assertEqual(len({r["immutable_system_sha256"] for r in rows}),9)
        lane0=[r for r in rows if r["lane"]==0]
        self.assertEqual(len({r["current_image"] for r in lane0}),1)
        self.assertEqual(len({r["common"]["messages"][-1]["content"][-1]["image_url"]["url"] for r in lane0}),1)

    def test_reproducible_public_images_and_hashes(self):
        f,images=cua.build_fixture(self.bench,steps=7,prefix_chars=512,window=2)
        self.assertEqual(cua.digest(f),cua.digest(self.fixture))
        self.assertEqual(images,self.images)
        for data in images.values(): self.assertTrue(data.startswith(b"\x89PNG\r\n\x1a\n"))
        self.assertTrue(all(m["width"]==1344 and m["height"]==768 for m in f["images"].values()))
        for lane in (0,1):
            self.assertTrue(all(f["images"][f"lane{lane}-step{s:03d}.png"]["sha256"] !=
                f["images"][f"lane{lane}-step{s+1:03d}.png"]["sha256"] for s in range(6)))


class StreamAndQualityTests(FixtureMixin,unittest.TestCase):
    def client(self,engine="tensorsharp",model="route-a"):
        args=types.SimpleNamespace(engine=engine,url="http://NO-NETWORK.invalid",model=model,extra_body=None,
            metrics_url="",tokenizer=None,server_log=None,temperature=0,top_p=1,sampling_seed=1234)
        with patch("urllib.request.urlopen",side_effect=AssertionError("client attempted model discovery")):
            return self.bench.Client(args)

    def measure(self,row=None,engine="tensorsharp",model="route-a",**response_options):
        row=row or self.rows("edit")[5]
        payload=response_bytes(row["expected"],**response_options)
        return cua.measure(self.bench,self.client(engine,model),row,10,lambda request,**kw:Response(payload))

    def test_raw_wire_full_response_and_actual_usage_are_preserved(self):
        row=self.rows("edit")[5]
        result=self.measure(row)
        self.assertEqual(result["quality"]["verdict"],"PASS")
        self.assertTrue(result["quality"]["latency_qualified"])
        self.assertEqual(result["response"]["raw_usage"]["completion_tokens"],61)
        self.assertEqual(result["response"]["cached_tokens"],16602)
        self.assertTrue(result["raw_transport"]["done"])
        import base64,hashlib
        wire=base64.b64decode(result["raw_transport"]["body_base64"])
        self.assertTrue(response_bytes(row["expected"]).startswith(wire))
        self.assertTrue(wire.endswith(b"data:[DONE]\n"))
        self.assertEqual(hashlib.sha256(wire).hexdigest(),result["raw_transport"]["body_sha256"])
        self.assertIsNone(result["response"]["decode_tok_s"])
        self.assertEqual(result["output"]["completion_tokens"],61)
        self.assertEqual(result["raw_observation"]["n_tool_metadata_deltas"],1)

    def test_only_model_removed_from_canonical_input_full_body_keeps_it(self):
        a,b=self.measure(model="route-a"),self.measure(model="route-b")
        self.assertNotEqual(a["full_body_sha256"],b["full_body_sha256"])
        self.assertNotEqual(a["wire_request_body_sha256"],b["wire_request_body_sha256"])
        self.assertEqual(a["canonical_input_sha256"],b["canonical_input_sha256"])
        body=copy.deepcopy(a["request_body"]);body.pop("model")
        self.assertEqual(cua.digest(body),a["canonical_input_sha256"])
        body["messages"][2]["tool_calls"][0]["id"]="changed-history-tool-id"
        self.assertNotEqual(cua.digest(body),a["canonical_input_sha256"])

    def test_engine_adapters_remain_explicit_no_forced_tool_choice(self):
        results=[self.measure(engine=e) for e in ("tensorsharp","velox","ninfer")]
        self.assertEqual(len({r["common_sha256"] for r in results}),1)
        self.assertNotEqual(results[0]["canonical_input_sha256"],results[1]["canonical_input_sha256"])
        for r in results:
            self.assertEqual(r["request_body"]["tool_choice"],"auto")
            self.assertEqual(r["request_body"]["temperature"],0)
        self.assertIs(results[0]["request_body"]["think"],False)
        self.assertIs(results[1]["request_body"]["chat_template_kwargs"]["enable_thinking"],False)

    def test_semantic_failures_keep_raw_and_withhold_speed(self):
        row=copy.deepcopy(self.rows("edit")[5])
        for changed in ({"code":"0000"},{"revision":"R000"},{"lane":1},{"step":True},{"extra":1}):
            args={**row["expected"],**changed}
            payload=response_bytes(args)
            result=cua.measure(self.bench,self.client(),row,10,lambda request,**kw:Response(payload))
            self.assertEqual(result["quality"]["verdict"],"FAIL",changed)
            self.assertTrue(result["raw_transport"]["body_base64"])
            summary=cua.summarise([result],[{"wall_s":1}],1)
            self.assertIsNone(summary["qualified_ttft_median_s"])
            self.assertIsNone(summary["qualified_wave_wall_total_s"])
        for options in ({"tool":"desktop_action_00"},{"content":"<think>guess</think>"},
                        {"finish":"length"},{"done":False},{"content":"hello"}):
            result=self.measure(**options)
            self.assertEqual(result["quality"]["verdict"],"FAIL",options)

    def test_missing_usage_cannot_use_tokenizer_or_sse_event_count(self):
        result=self.measure(usage=False)
        self.assertEqual(result["quality"]["verdict"],"PASS")
        self.assertEqual(result["quality"]["measurement_verdict"],"UNVERIFIED")
        self.assertIsNone(result["output"]["completion_tokens"])
        self.assertFalse(result["quality"]["latency_qualified"])
        self.assertIsNone(cua.summarise([result],[{"wall_s":1}],1)["qualified_e2e_median_s"])

    def test_complete_semantic_http_error_body_retained_beyond_client_error_excerpt(self):
        payload=b"SERVER_ERROR:"+b"x"*4096
        def failed(request,**kwargs):
            raise urllib.error.HTTPError(request.full_url,503,"Unavailable",{},io.BytesIO(payload))
        result=cua.measure(self.bench,self.client(),self.rows("append")[0],10,failed)
        self.assertEqual(result["quality"]["verdict"],"FAIL")
        self.assertEqual(result["raw_transport"]["bytes"],len(payload))
        self.assertEqual(result["raw_observation"]["http_status"],503)
        self.assertIn("SERVER_ERROR",result["raw_observation"]["error"])

    def test_malformed_usage_and_extra_choices_are_failed_with_raw_preserved(self):
        row=self.rows("append")[0]
        payload=response_bytes(row["expected"]).replace(b'"completion_tokens": 61',b'"completion_tokens": true')
        result=cua.measure(self.bench,self.client(),row,10,lambda request,**kw:Response(payload))
        self.assertEqual(result["quality"]["measurement_verdict"],"UNVERIFIED")
        payload=response_bytes(row["expected"])+b"unread tail"
        # A second metadata-only choice must not let a semantically valid first
        # choice turn an unsupported multiple-choice stream into a PASS.
        extra=b'data: {"choices":[{"index":1,"delta":{"role":"assistant"}}]}\n\n'
        payload=extra+payload
        result=cua.measure(self.bench,self.client(),row,10,lambda request,**kw:Response(payload))
        self.assertFalse(result["quality"]["checks"]["single_choice"])
        self.assertEqual(result["quality"]["verdict"],"FAIL")
        malformed=event({"content":"diagnostic"},finish="stop",usage=["invalid-usage-type"])+b"data: [DONE]\n\n"
        result=cua.measure(self.bench,self.client(),row,10,lambda request,**kw:Response(malformed))
        self.assertEqual(result["quality"]["verdict"],"FAIL")
        self.assertTrue(result["raw_transport"]["body_base64"])
        self.assertIn("frozen_client_normalization",result["response"]["error"])

    def test_incomplete_or_duplicate_run_never_qualifies(self):
        row=self.measure()
        self.assertFalse(cua.summarise([row],[],2)["latency_qualified"])
        self.assertFalse(cua.summarise([row,row],[],2)["latency_qualified"])
        summary=cua.summarise([row],[{"wall_s":2.0}],1)
        self.assertTrue(summary["latency_qualified"])
        self.assertEqual(summary["qualified_server_completion_tokens"],61)
        self.assertEqual(summary["qualified_wave_wall_total_s"],2)
        self.assertIn("correlated",summary["CI"])

    def test_two_lane_waves_are_sequential_and_arrive_together(self):
        selected=[r for r in self.fixture["requests"] if r["scenario"]=="append" and r["lanes"]==2]
        selected=[r for r in selected if r["step"]<2]
        lock=threading.Lock();arrived={};finished=set();writes=[]
        second_barrier=threading.Barrier(2)
        def fake(bench,client,row,timeout):
            with lock:
                if row["step"]==1:self.assertEqual(finished,{(0,0),(1,0)})
                arrived.setdefault(row["step"],set()).add(row["lane"])
            second_barrier.wait(timeout=5)
            r=self.measure(row)
            with lock:finished.add((row["lane"],row["step"]))
            return r
        result=cua.execute_case(self.bench,self.client(),selected,2,2,10,writes.append,measure_fn=fake)
        self.assertTrue(result["latency_qualified"])
        self.assertEqual(arrived,{0:{0,1},1:{0,1}})
        self.assertEqual(len([r for r in writes if r["kind"]=="request"]),4)
        self.assertEqual(len([r for r in writes if r["kind"]=="wave"]),2)


class OptionalMediaProbeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.bench=cua.load_bench()
        cls.fixture,cls.images=cua.build_fixture(cls.bench,steps=6,prefix_chars=64,media_probe=True)

    def test_opt_in_is_separate_short_protocol_default_scenarios_unchanged(self):
        self.assertEqual(cua.SCENARIOS,("append","truncate","edit"))
        self.assertEqual(self.fixture["scenarios"],["edit_media"])
        self.assertFalse(self.fixture["standard_30step_protocol"])
        self.assertEqual(len(self.fixture["requests"]),18)
        with self.assertRaises(ValueError):cua.build_fixture(self.bench,steps=30,media_probe=True)

    def test_only_old_image_changes_earlier_text_stays_identical_and_anchor_ocr_changes(self):
        rows=[r for r in self.fixture["requests"] if r["lanes"]==1]
        before,after=rows[4],rows[5]
        self.assertEqual(before["common"]["messages"][1]["content"][0],after["common"]["messages"][1]["content"][0])
        self.assertNotEqual(before["common"]["messages"][1]["content"][1],after["common"]["messages"][1]["content"][1])
        self.assertEqual(before["expected"]["revision"],after["expected"]["revision"])
        self.assertNotEqual(before["expected"]["anchor_code"],after["expected"]["anchor_code"])
        self.assertFalse(after["history"]["policy_edit_this_step"])
        self.assertTrue(after["history"]["old_media_edit_this_step"])
        self.assertFalse(after["history"]["edited_media"]["early_user_text_changed"])
        self.assertEqual(before["immutable_system_sha256"],after["immutable_system_sha256"])
        for r in rows[:5]:
            self.assertEqual(r["common"]["messages"][1]["content"][1],rows[0]["common"]["messages"][1]["content"][1])

    def test_stale_anchor_ocr_fails_even_with_correct_current_image_and_tool(self):
        rows=[r for r in self.fixture["requests"] if r["lanes"]==1]
        row=rows[5]
        args=types.SimpleNamespace(engine="tensorsharp",url="http://NO-NETWORK.invalid",model="fixture-route",extra_body=None,
            metrics_url="",tokenizer=None,server_log=None,temperature=0,top_p=1,sampling_seed=1234)
        client=self.bench.Client(args)
        good=response_bytes(row["expected"])
        result=cua.measure(self.bench,client,row,10,lambda request,**kw:Response(good))
        self.assertTrue(result["quality"]["latency_qualified"])
        stale={**row["expected"],"anchor_code":rows[4]["expected"]["anchor_code"]}
        payload=response_bytes(stale)
        result=cua.measure(self.bench,client,row,10,lambda request,**kw:Response(payload))
        self.assertEqual(result["quality"]["verdict"],"FAIL")
        self.assertFalse(result["quality"]["latency_qualified"])


class PreparationAdmissionTests(unittest.TestCase):
    def test_full_cli_mock_replay_writes_raw_manifest_counts_and_final_status(self):
        parent=cua.ROOT/"artifacts/qwen38-regression/cua-cpu-tests"
        parent.mkdir(parents=True,exist_ok=True)
        with tempfile.TemporaryDirectory(dir=parent) as tmp:
            root=Path(tmp);out=root/"prepared"
            with patch("urllib.request.urlopen",side_effect=AssertionError("prepare used network")):
                self.assertEqual(cua.main(["prepare","--out",str(out),"--steps","2","--prefix-chars","64"]),0)
                fixture,_=cua.load_fixture(out)
            expected={cua.digest(r["common"]["messages"]):r["expected"] for r in fixture["requests"]}
            launch=root/"launch.json"
            launch.write_text(json.dumps({"engine":"tensorsharp","context_tokens":65536,"lanes":2,
                "environment":{"TS_PREFIX_CACHE_ISOLATION":None},"argv":["MOCK_ONLY_NO_SERVER"]}))
            def fake(request,**kwargs):
                self.assertIsNotNone(request.data)
                body=json.loads(request.data)
                self.assertEqual(body["tool_choice"],"auto")
                return Response(response_bytes(expected[cua.digest(body["messages"])]))
            result=root/"mock.jsonl"
            with patch("urllib.request.urlopen",side_effect=fake):
                self.assertEqual(cua.main(["run","--fixtures",str(out),"--engine","tensorsharp",
                    "--url","http://NO-NETWORK.invalid","--model","mock-route","--manifest",str(launch),
                    "--label","MOCK_ONLY","--lanes","2","--cache-profile","default-lineage","--out",str(result)]),0)
            records=[json.loads(line) for line in result.read_text().splitlines()]
            self.assertEqual(records[0]["launch_manifest"]["argv"],["MOCK_ONLY_NO_SERVER"])
            self.assertFalse(records[0]["standard_30step_protocol"])
            requests=[r for r in records if r["kind"]=="request"]
            self.assertEqual(len(requests),12)
            self.assertTrue(all(r["raw_transport"]["body_base64"] and r["raw_observation"]["usage"] for r in requests))
            self.assertEqual(len([r for r in records if r["kind"]=="case_summary"]),3)
            self.assertEqual(records[-1]["kind"],"run_end")
            self.assertTrue(records[-1]["qualified"])
            self.assertFalse(records[-1]["original_benchmark_scores_changed"])

    def test_context_lane_and_lineage_deployment_mismatch_rejected_before_network(self):
        fixture={"declared_context_capacity":65536}
        args=types.SimpleNamespace(engine="tensorsharp",context_capacity=65536,lanes=2,cache_profile="default-lineage")
        launch={"engine":"tensorsharp","context_tokens":65536,"lanes":2,
                "environment":{"TS_PREFIX_CACHE_ISOLATION":None}}
        cua.validate_deployment(launch,args,fixture)
        for patch_value in ({"context_tokens":40960},{"lanes":1},{"environment":{}},
                            {"environment":{"TS_PREFIX_CACHE_ISOLATION":"shared"}}):
            with self.assertRaises(ValueError):cua.validate_deployment({**launch,**patch_value},args,fixture)
        args.cache_profile="trusted-single-tenant-shared"
        with self.assertRaises(ValueError):cua.validate_deployment(launch,args,fixture)
        launch["environment"]["TS_PREFIX_CACHE_ISOLATION"]="shared"
        cua.validate_deployment(launch,args,fixture)

    def test_prepare_and_load_no_network_tamper_rejected(self):
        parent=cua.ROOT/"artifacts/qwen38-regression/cua-cpu-tests"
        parent.mkdir(parents=True,exist_ok=True)
        with tempfile.TemporaryDirectory(dir=parent) as tmp:
            out=Path(tmp)/"fixture"
            with patch("urllib.request.urlopen",side_effect=AssertionError("prepare used network")):
                self.assertEqual(cua.main(["prepare","--out",str(out),"--steps","2","--prefix-chars","64"]),0)
                fixture,manifest=cua.load_fixture(out)
            self.assertEqual(manifest["requests"],18)
            self.assertEqual(manifest["status"],"PREPARED_NOT_RUN")
            image=out/"images"/next(iter(fixture["images"]))
            image.write_bytes(b"changed")
            with self.assertRaisesRegex(ValueError,"image changed"):cua.load_fixture(out)


if __name__=="__main__":
    unittest.main()
