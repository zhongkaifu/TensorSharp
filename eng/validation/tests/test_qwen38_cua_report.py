#!/usr/bin/env python3
"""Stdlib-only mock records; no model, socket, matplotlib or GPU required."""
import base64
import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest.mock import patch

HERE = Path(__file__).resolve().parents[1]
def module(name, path):
    spec=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m
report=module("cua_report_tested",HERE/"qwen38-cua-report.py")
control=module("cua_frozen_for_report_mocks",HERE/"qwen38-cua-control.py")
bench=control.load_bench()


def header(lanes=1, scenario="append", profile="default-lineage", label="MOCK_ONLY", engine="tensorsharp", tag="one", steps=None):
    media=scenario=="edit_media";steps=steps if steps is not None else 6 if media else 30
    launch={"engine":engine,"mode":"plain","context_tokens":65536,"lanes":2,"argv":["MOCK_ONLY_NO_SERVER"],
        "environment":{"TS_PREFIX_CACHE_ISOLATION":"shared" if profile=="trusted-single-tenant-shared" else None},
        "binaries":[{"path":"MOCK_ONLY.dll","sha256":"a"*64}],"models":{"target_sha256":"b"*64}}
    sources=control.sources()
    return {"kind":"run","schema":control.VERSION,"created_utc":"MOCK_"+tag,"engine":engine,"mode":"plain","label":label,"model":"MOCK_ROUTE",
        "lanes":lanes,"steps":steps,"selected_scenarios":[scenario],"context_capacity_operator_declaration":65536,
        "cache_profile_operator_declaration":profile,"cache_comparison_group":report.cache_group(profile),
        "standard_30step_protocol":not media,"protocol":"opt_in_6_to_10_step_old_media_probe" if media else "default_three_scenarios",
        "sources":sources,"fixture_manifest":{"sources":sources,"fixture_sha256":"f"*64},"launch_manifest":launch,
        "launch_manifest_sha256":control.digest(launch),"settings":{"temperature":0,"top_p":1,"seed":1234,"thinking":False,
            "tool_choice":"auto","max_tokens":192,"warmup":"NONE","prior_history":"prescribed"}}


def request(h, step, lane):
    scenario=h["selected_scenarios"][0]
    expected={"code":"1234","color":"red","revision":"BASE","step":step,"lane":lane}
    if scenario=="edit_media":expected["anchor_code"]="5678"
    common={"messages":[{"role":"user","content":f"MOCK ONLY step{step} lane{lane} 中文"}],"tools":control.tools(scenario=="edit_media"),
        "tool_choice":"auto","max_tokens":192,"temperature":0,"top_p":1,"presence_penalty":0,"frequency_penalty":0,
        "seed":1234,"stream":True,"stream_options":{"include_usage":True}}
    # Real prepare() writes canonical fixture.json; run() reads that ordering
    # before the actual frozen Client.body inserts base fields and adapters.
    common=json.loads(control.canonical(common))
    client=bench.Client(types.SimpleNamespace(engine=h["engine"],model=h["model"],url="http://offline.invalid",
        extra_body=None,metrics_url="",tokenizer=None,temperature=0,top_p=1,sampling_seed=1234))
    body=client.body(common["messages"],common["max_tokens"],{k:v for k,v in common.items()if k not in("messages","max_tokens")})
    usage={"prompt_tokens":19000+step,"completion_tokens":60+lane,"prompt_tokens_details":{"cached_tokens":16602+step}}
    payload=b"data: MOCK_ONLY_TOOL_RESPONSE\n\ndata: [DONE]\n"
    wire_sha=hashlib.sha256(json.dumps(body).encode()).hexdigest()
    response={"content":"","reasoning":"","tool_calls":[{"name":"submit_observation","parsed":expected,"arguments":json.dumps(expected)}],
        "http_status":200,"error":None,"finish_reason":"tool_calls","raw_usage":usage,"prompt_tokens":usage["prompt_tokens"],
        "completion_tokens":usage["completion_tokens"],"completion_tokens_source":"usage","cached_tokens":16602+step,
        "ttft_s":.1+step*.001+lane*.01,"total_s":.4+step*.001+lane*.01,"response_sha256":"c"*64,
        "content_sha256":hashlib.sha256(b"").hexdigest(),"decode_tok_s":None}
    return {"kind":"request","scenario":scenario,"lanes":h["lanes"],"step":step,"lane":lane,"scope":"MOCK_SCOPE",
        "history":{"MOCK_ONLY":True},"expected":expected,"request_body":body,"common_sha256":control.digest(common),
        "full_body_sha256":control.digest(body),"canonical_input_sha256":control.digest({k:v for k,v in body.items() if k!="model"}),
        "wire_request_body_sha256":wire_sha,"frozen_normalized_request_sha256":"d"*64,
        "raw_observation":{"usage":usage,"error":None},"raw_transport":{"done":True,"closed_normally":True,
            "body_base64":base64.b64encode(payload).decode(),"body_sha256":hashlib.sha256(payload).hexdigest(),"bytes":len(payload),"request_body_sha256":wire_sha},
        "response":response,"quality":{"verdict":"PASS","measurement_verdict":"PASS","latency_qualified":True,"schema_errors":[],
            "checks":{k:True for k in report.QUALITY_CHECKS},"measurement_checks":{k:True for k in report.MEASUREMENT_CHECKS}},
        "output":{"completion_tokens":usage["completion_tokens"],"token_source":"actual_server_usage_only","tool_calls":1,"tool_arguments_utf8_bytes":100}}


def records(h=None):
    h=h or header();rows=[h];requests=[];waves=[];scenario=h["selected_scenarios"][0]
    for step in range(h["steps"]):
        for lane in range(h["lanes"]):
            r=request(h,step,lane);rows.append(r);requests.append(r)
        wave={"kind":"wave","scenario":scenario,"lanes":h["lanes"],"step":step,"wall_s":.6+step*.001,
            "arrival_policy":"barrier_each_step_no_client_think_time"};rows.append(wave);waves.append(wave)
    summary=control.summarise(requests,waves,h["steps"]*h["lanes"])
    summary.update(kind="case_summary",scenario=scenario,lanes=h["lanes"]);rows.append(summary)
    rows.append({"kind":"run_end","complete":True,"qualified":summary["latency_qualified"],"summaries":[summary]})
    return rows


def refresh(rows):
    h=rows[0];requests=[r for r in rows if r["kind"]=="request"];waves=[r for r in rows if r["kind"]=="wave"]
    s=control.summarise(requests,waves,h["steps"]*h["lanes"]);s.update(kind="case_summary",scenario=h["selected_scenarios"][0],lanes=h["lanes"])
    return [r for r in rows if r["kind"] not in ("case_summary","run_end")]+[s,{"kind":"run_end","complete":True,"qualified":s["latency_qualified"],"summaries":[s]}]


class CuaReportTests(unittest.TestCase):
    def setUp(self):
        parent=report.ROOT/"artifacts/qwen38-regression/cua-report-tests";parent.mkdir(parents=True,exist_ok=True)
        self.tmp=tempfile.TemporaryDirectory(dir=parent);self.addCleanup(self.tmp.cleanup);self.root=Path(self.tmp.name)
    def write(self,rows,name="mock.jsonl"):
        p=self.root/name;p.write_text("\n".join(control.canonical(r) for r in rows)+"\n");return p
    def data(self,rows,**kwargs):return report.build_data([self.write(rows)],**kwargs)

    def test_full30_by_two_lanes_and_raw_token_cache_hashes_preserved(self):
        data=self.data(records(header(lanes=2)));g=data["groups"][0];c=g["run_cases"][0]
        self.assertTrue(g["qualified"]);self.assertEqual(c["observed_requests"],60);self.assertEqual(c["observed_waves"],30)
        self.assertEqual(g["metrics"]["completion_tokens_total"],30*121)
        self.assertAlmostEqual(g["metrics"]["wave_wall_total_s"],sum(.6+i*.001 for i in range(30)))
        self.assertEqual(c["observations"][1]["raw_usage"]["completion_tokens"],61)
        self.assertEqual(c["observations"][0]["server_cached_tokens"],16602)
        self.assertTrue(all(c["observations"][0]["request_identity_checks"].values()))
        self.assertIn("correlated",c["sample_note"]);self.assertIsNone(g["decode_tok_s"])

    def test_canonical_jsonl_order_is_rebuilt_by_the_real_frozen_client_offline(self):
        for engine in("tensorsharp","velox","ninfer"):
            h=header(engine=engine,profile="default-lineage"if engine=="tensorsharp"else"engine-native-global")
            rows=records(h);path=self.write(rows)
            parsed=[json.loads(line)for line in path.read_text().splitlines()]
            r=next(r for r in parsed if r["kind"]=="request")
            self.assertNotEqual(hashlib.sha256(json.dumps(r["request_body"]).encode()).hexdigest(),r["wire_request_body_sha256"])
            with patch("urllib.request.urlopen",side_effect=AssertionError("offline report cannot contact any server")):
                data=report.build_data([path])
            self.assertTrue(data["groups"][0]["qualified"],engine)
            checks=data["groups"][0]["run_cases"][0]["observations"][0]["request_identity_checks"]
            self.assertTrue(checks["frozen_body_reconstruction"]);self.assertTrue(checks["wire_request"])

    def test_matching_reported_wire_hashes_do_not_replace_exact_byte_verification(self):
        rows=records();r=next(r for r in rows if r["kind"]=="request")
        reordered=json.loads(control.canonical(r["request_body"]))
        wrong=hashlib.sha256(json.dumps(reordered).encode()).hexdigest()
        self.assertNotEqual(wrong,r["wire_request_body_sha256"])
        r["wire_request_body_sha256"]=r["raw_transport"]["request_body_sha256"]=wrong
        data=self.data(rows);self.assertFalse(data["groups"][0]["qualified"])
        checks=data["groups"][0]["run_cases"][0]["observations"][0]["request_identity_checks"]
        self.assertTrue(checks["full_body"]);self.assertTrue(checks["canonical_input"]);self.assertTrue(checks["portable_common"])
        self.assertTrue(checks["frozen_body_reconstruction"]);self.assertFalse(checks["wire_request"])

    def test_complete_body_adapter_or_model_context_changes_fail_closed(self):
        for change in("adapter","model_header","builder_sha"):
            rows=records();r=next(r for r in rows if r["kind"]=="request")
            if change=="adapter":
                r["request_body"]["skills"]=["unrequested"]
                r["full_body_sha256"]=control.digest(r["request_body"])
                r["canonical_input_sha256"]=control.digest({k:v for k,v in r["request_body"].items()if k!="model"})
                wire=hashlib.sha256(json.dumps(r["request_body"]).encode()).hexdigest()
                r["wire_request_body_sha256"]=r["raw_transport"]["request_body_sha256"]=wire
            elif change=="model_header":rows[0]["model"]="wrong-route"
            else:rows[0]["sources"]["frozen_bench"]="0"*64
            self.assertFalse(self.data(rows)["groups"][0]["qualified"],change)

    def test_original_fail_or_unverified_never_shows_any_performance(self):
        for change in ("quality","measurement","check","flag"):
            rows=records();r=next(r for r in rows if r["kind"]=="request")
            if change=="quality":r["quality"].update(verdict="FAIL",latency_qualified=False);r["quality"]["checks"]["typed_exact_arguments"]=False
            elif change=="measurement":r["quality"].update(measurement_verdict="UNVERIFIED",latency_qualified=False);r["quality"]["measurement_checks"]["actual_server_usage"]=False
            elif change=="check":r["quality"]["checks"]["typed_exact_arguments"]=False
            else:r["quality"]["latency_qualified"]=False
            data=self.data(refresh(rows));g=data["groups"][0]
            self.assertFalse(g["qualified"],change);self.assertTrue(all(v is None for v in g["metrics"].values()))
            self.assertEqual(len(g["run_cases"][0]["observations"]),30)

    def test_missing_duplicate_bool_outofrange_ids_and_wave_fail_closed(self):
        for change in ("missing","duplicate","bool","outofrange","wave_missing","wave_duplicate","wave_bad_range"):
            rows=records();req=next(r for r in rows if r["kind"]=="request")
            if change=="missing":rows.remove(req)
            elif change=="duplicate":rows.insert(2,copy.deepcopy(req))
            elif change=="bool":req["step"]=False
            elif change=="outofrange":req["step"]=30 # still30 unique; producer's old len/set count alone is insufficient
            else:
                wave=next(r for r in rows if r["kind"]=="wave")
                if change=="wave_missing":rows.remove(wave)
                elif change=="wave_duplicate":rows.insert(2,copy.deepcopy(wave))
                else:wave["step"]=30
            g=self.data(rows)["groups"][0];self.assertFalse(g["qualified"],change);self.assertIsNone(g["metrics"]["ttft_median_s"])

    def test_partial_runs_cannot_stitch_into_complete_case(self):
        a,b=records(header(tag="a")),records(header(tag="b"))
        a=refresh([r for r in a if r["kind"] not in ("request","wave") or r["step"]<15])
        b=refresh([r for r in b if r["kind"] not in ("request","wave") or r["step"]>=15])
        data=self.data(a+b);self.assertEqual(len(data["attempts"]),2)
        g=data["groups"][0];self.assertEqual(g["run_case_count"],2);self.assertFalse(g["qualified"])
        self.assertTrue(all(c["observed_requests"]==15 for c in g["run_cases"]))

    def test_failed_attempt_is_not_dropped_for_passing_same_configuration(self):
        a,b=records(header(tag="a")),records(header(tag="b"));r=next(r for r in b if r["kind"]=="request")
        r["quality"].update(verdict="FAIL",latency_qualified=False);r["quality"]["checks"]["one_correct_tool"]=False
        g=self.data(a+refresh(b))["groups"][0]
        self.assertEqual(g["run_case_count"],2);self.assertFalse(g["qualified"]);self.assertIsNone(g["metrics"]["completion_tokens_total"])

    def test_run_end_and_original_summary_required_and_crosschecked(self):
        for change in ("missing_end","missing_summary","fake_metric","duplicate_summary","after_end"):
            rows=records()
            if change=="missing_end":rows.pop()
            elif change=="missing_summary":rows=[r for r in rows if r["kind"]!="case_summary"]
            elif change=="fake_metric":rows[-2]["qualified_ttft_median_s"]=999
            elif change=="duplicate_summary":rows.insert(-1,copy.deepcopy(rows[-2]))
            else:rows.append(copy.deepcopy(next(r for r in rows if r["kind"]=="request")))
            self.assertFalse(self.data(rows)["groups"][0]["qualified"],change)

    def test_wave_wall_cannot_be_shorter_than_its_actual_requests(self):
        rows=records();wave=next(r for r in rows if r["kind"]=="wave");wave["wall_s"]=.01
        g=self.data(refresh(rows))["groups"][0]
        self.assertFalse(g["qualified"])
        self.assertTrue(any("wave wall shorter" in e for e in g["run_cases"][0]["errors"]))

    def test_selected_single_scenario_is_not_claimed_as_full_default_matrix(self):
        data=self.data(records());coverage=data["attempts"][0]["selected_only_coverage"]
        self.assertTrue(data["groups"][0]["qualified"])
        self.assertEqual(coverage["unselected_NOT_RUN"],["truncate","edit"])
        self.assertTrue(coverage["full_default_matrix_NOT_inferred"])

    def test_media6_qualifies_independently_but_cannot_mix_with_default30(self):
        data=self.data(records()+records(header(scenario="edit_media",tag="media")))
        self.assertEqual(len(data["groups"]),2);self.assertTrue(all(g["qualified"] for g in data["groups"]))
        self.assertEqual({g["steps"] for g in data["groups"]},{6,30})
        rows=records(header(scenario="edit_media",steps=5));self.assertFalse(self.data(rows)["groups"][0]["qualified"])

    def test_cache_profiles_context_fixture_and_deployment_do_not_pool(self):
        a=records(header(tag="a"));b=records(header(profile="trusted-single-tenant-shared",tag="b"))
        c=records(header(profile="engine-native-global",engine="velox",tag="c"))
        data=self.data(a+b+c);self.assertEqual(len(data["groups"]),3)
        self.assertEqual({g["cache_group"] for g in data["groups"]},{"default-lineage-diagnostic","trusted-single-tenant-prefix"})
        d=records(header(tag="d"));d[0]["fixture_manifest"]["fixture_sha256"]="e"*64
        e=records(header(tag="e"));e[0]["launch_manifest"]["models"]["target_sha256"]="e"*64
        self.assertEqual(len(self.data(a+d+e)["groups"]),3)

    def test_actual_usage_invalid_counts_or_body_transport_hashes_withhold(self):
        for change in ("no_usage","bool_count","event_count","negative_cache","nan_time","body_sha","wire_sha"):
            rows=records();r=next(r for r in rows if r["kind"]=="request")
            if change=="no_usage":r["response"]["raw_usage"]=None
            elif change=="bool_count":r["response"]["raw_usage"]["completion_tokens"]=True
            elif change=="event_count":r["response"]["completion_tokens_source"]="sse_events"
            elif change=="negative_cache":r["response"]["raw_usage"]["prompt_tokens_details"]["cached_tokens"]=-1
            elif change=="nan_time":r["response"]["ttft_s"]=float('nan')
            elif change=="body_sha":r["request_body"]["model"]="changed-route-without-hash-update"
            else:r["raw_transport"]["body_sha256"]="0"*64
            self.assertFalse(self.data(rows)["groups"][0]["qualified"],change)

    def test_tool_event_decode_rate_is_discarded_even_in_an_eligible_run(self):
        rows=records()
        for r in rows:
            if r["kind"]=="request":r["response"]["decode_tok_s"]=999999
        g=self.data(rows)["groups"][0]
        self.assertTrue(g["qualified"]);self.assertIsNone(g["decode_tok_s"])
        self.assertTrue(g["run_cases"][0]["observations"][0]["tool_decode_rate_discarded"])

    def test_malformed_tail_or_conflicting_snapshots_cannot_promote_older_pass(self):
        path=self.write(records());path.write_text(path.read_text()+'{"kind":\n')
        data=report.build_data([path]);self.assertEqual(data["sources"][0]["status"],"ERROR");self.assertFalse(data["groups"][0]["qualified"])
        a=records();b=records();b.pop()
        data=self.data(a+b);self.assertTrue(all(not g["qualified"] for g in data["groups"]))

    def test_identical_file_or_segment_copies_are_not_independent_runs(self):
        rows=records();a=self.write(rows,"a.jsonl");b=self.write(rows,"b.jsonl")
        data=report.build_data([a,b]);self.assertEqual(data["groups"][0]["run_case_count"],1);self.assertEqual(len(data["sources"][0]["aliases"]),1)
        data=self.data(rows+rows);self.assertEqual(data["groups"][0]["run_case_count"],1);self.assertEqual(len(data["duplicate_attempts"]),1)

    def test_clean_copy_cannot_hide_corrupt_copy_and_hardware_never_pools(self):
        rows=records();a=self.write(rows,"clean.jsonl");b=self.write(rows,"bad.jsonl");b.write_text(b.read_text()+'{"bad":\n')
        self.assertFalse(report.build_data([a,b])["groups"][0]["qualified"])
        other=records(header(tag="otherGPU"));other[0]["launch_manifest"]["gpu"]={"name":"MOCK_DIFFERENT_GPU"}
        self.assertEqual(len(self.data(rows+other)["groups"]),2)

    def test_not_run_plan_and_missing_file_have_null_performance(self):
        slot={"engine":"ninfer","mode":"plain","label":"future","cache_profile":"engine-native-global","lanes":2,"scenario":"append"}
        data=report.build_data([self.root/"absent.jsonl"],plan=[slot]);g=data["groups"][0]
        self.assertEqual(g["status"],"NOT_RUN");self.assertTrue(all(v is None for v in g["metrics"].values()))
        self.assertEqual(data["sources"][0]["status"],"NOT_RUN")
        self.assertEqual(report.build_data([])["summary_zh"]["状态"],"NOT_RUN")

    def test_data_only_cli_imports_no_matplotlib_and_writes_chinese_summary(self):
        p=self.write(records());out=self.root/"result"
        with patch.dict(sys.modules,{"matplotlib":None,"matplotlib.pyplot":None}):
            self.assertEqual(report.main([str(p),"--output-dir",str(out),"--data-only"]),0)
        data=json.loads((out/"data.json").read_text());self.assertTrue(data["groups"][0]["qualified"])
        self.assertEqual(json.loads((out/"context-figures.json").read_text()),[])
        self.assertIn("合格场景组",json.loads((out/"summary-zh.json").read_text()))

    def test_malformed_header_or_extra_broken_source_is_not_clean_qualified_summary(self):
        bad={"kind":"run","schema":"wrong","lanes":"two","steps":1000000000,"selected_scenarios":[],
            "sources":[],"fixture_manifest":[],"launch_manifest":[],"cache_profile_operator_declaration":{}}
        data=self.data([bad]);self.assertEqual(data["summary_zh"]["状态"],"ERROR")
        clean=self.write(records(),"clean.jsonl");broken=self.root/"broken.jsonl";broken.write_text('{"bad":\n')
        data=report.build_data([clean,broken]);self.assertTrue(data["groups"][0]["qualified"])
        self.assertNotEqual(data["summary_zh"]["状态"],"QUALIFIED_DATA")
        self.assertEqual(data["summary_zh"]["输入ERROR文件"],1)


if __name__=="__main__":unittest.main()
