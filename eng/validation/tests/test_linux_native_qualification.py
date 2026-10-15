"""Model-free strict acceptance and immutable matrix tests; no native loading."""
import copy
import contextlib
import importlib.util
import io
import json
from pathlib import Path
import unittest
from unittest.mock import patch

source = Path(__file__).resolve().parents[1] / "linux-native-qualification.py"
spec = importlib.util.spec_from_file_location("linux_native_qualification", source)
tool = importlib.util.module_from_spec(spec)
spec.loader.exec_module(tool)


def evidence(selected=False, flags=None):
    flags = flags or []
    graph = {"ExecutionId": 1, "Rows": 3, "Nodes": 40, "BackendSplits": 1,
             "Slots": [0, 1, 2], "PositionsBefore": [8, 8, 8], "PositionsAfter": [9, 9, 9]}
    pair = {"ExecutionId": 5, "Rows": 2, "Nodes": 30, "BackendSplits": 1,
            "Slots": [1, 2], "PositionsBefore": [1, 1], "PositionsAfter": [2, 2]}
    actual = dict(passed=True, full_logits_bit_exact=True, shared_graph_owner=True, backend="CUDA",
                  logit_elements=2432, logits_sha256="a" * 64, checks=200,
                  selected_experts_tested=selected, dispatch_faults_tested=False,
                  prefill_rows=None, quantized_source=None, swiglu_clamp=0, expert_groups=1,
                  native=[dict(path="/mnt/e/prod/libGgmlOps.so", sha256="b" * 64)],
                  final_budget=[dict(Pool="ram", Committed=0, Reserved=0),
                                dict(Pool="cuda", Committed=0, Reserved=0)],
                  executions=[dict(actual=graph) for _ in range(4)] + [dict(row_departure=pair)])
    if selected:
        actual["executions"].append(dict(minimum_cache_bytes=1000))
    compare = dict(passed=True, differing_elements=0, max_abs=0, rmse=0, logit_elements=2432,
                   logits_sha256="a" * 64, reference_sha256="a" * 64)
    return actual, compare


def final_drain_proof(seams=2):
    return dict(selected_final_drain_failure=True, stage=5, kind=1, participating_rows=3,
                expert_seams=seams, completed_helper_rows=3 * seams, successful_graphs_unchanged=True,
                logits_unpublished=True, heads_unpublished=True, slots_fenced=True,
                reset_recovered_n3=True, primary_departure_recovered_n2=True)


class LinuxNativeQualificationTests(unittest.TestCase):
    def accept(self, actual, comparison, flags=None):
        tool.validate_glm_report(actual, comparison, Path("/mnt/e/prod/libGgmlOps.so"),
                                 "b" * 64, "CUDA", flags or [])

    def test_adapter7_capability_required_without_loading_native(self):
        native = Path('/mnt/e/prod/libGgmlOps.so')
        expected = {'TSGgml_GlmUnifiedBudgetVersion': 2, 'TSGgml_GlmSelectedExpertAdapterVersion': 7,
                    'TSGgml_HostMoeFilePipelineVersion': 2, 'TSGgml_Qwen4ExpArenaBudgetKind': 5,
                    'TSGgml_HostMoeExpertCacheBudgetKind': 4}
        with patch.object(tool, 'require_elf', return_value=native), patch.object(tool, 'sha', return_value='a' * 64), \
                patch.object(tool.subprocess, 'check_output', return_value=json.dumps(expected)) as query:
            self.assertEqual(expected, tool.abi_evidence(native, {})['capabilities'])
            self.assertEqual(expected, json.loads(query.call_args.args[0][-1]))
            for bad in [expected | {'TSGgml_GlmSelectedExpertAdapterVersion': 6},
                        expected | {'TSGgml_GlmSelectedExpertAdapterVersion': 8}]:
                query.return_value = json.dumps(bad)
                with self.assertRaises(RuntimeError):
                    tool.abi_evidence(native, {})

    def test_exact_case_counts_and_native16_new_regressions(self):
        self.assertEqual(12, len(tool.cpu_cases()))
        self.assertEqual(21, len(tool.cuda_cases()))
        self.assertEqual(56, len(tool.glm_cases("/mnt/e/model.gguf")))
        names = {c[0]: c[2] for c in tool.cuda_cases()}
        self.assertEqual(["--glm-scalar-file-window"], names["scalar-large-file-q8"])
        self.assertEqual(["--pooled-quota", "--quantization", "iq1_m"], names["pooled-largest-quota-iq1"])
        self.assertEqual(["--mixed", "--pipeline-scope", "0"], dict((c[0], c[2]) for c in tool.cpu_cases())["mixed-file-batch-cpu"])

    def test_glm_matrix_keeps_rows_original_iq_clamp_and_fault_modes(self):
        cases = tool.glm_cases("/mnt/e/model with spaces.gguf")
        for mode in ("production", "hooks"):
            for kind in ("q8", "iq"):
                for rows in (8, 9, 10, 16, 128, 256, 384, 512, 1024):
                    case = next(c for c in cases if c[0] == f"{mode}-{kind}-rows{rows}")
                    self.assertIn("--prefill-rows=" + str(rows), case[3])
                    self.assertEqual(mode == "hooks", "--dispatch-faults" in case[3])
                    quant = [f for f in case[3] if f.startswith("--quantized-source=")]
                    self.assertEqual(["--quantized-source=/mnt/e/model with spaces.gguf"] if kind == "iq" else [], quant)

    def test_bf16_sensitive_matrix_appends_eight_distinct_cases_without_changing_historical_48(self):
        cases = tool.glm_cases('/mnt/e/model.gguf')
        self.assertEqual(48, len([case for case in cases if '--bf16-sensitive' not in case[3]]))
        self.assertTrue(all('--bf16-sensitive' not in case[3] for case in cases[:48]))
        self.assertEqual(8, len(cases[48:]))
        for mode in ('production', 'hooks'):
            for kind in ('q8', 'iq'):
                for rows in (512, 809):
                    case = next(case for case in cases if case[0] == f'{mode}-{kind}-bf16-sensitive-rows{rows}')
                    self.assertIn('--bf16-sensitive', case[3])
                    self.assertIn(f'--prefill-rows={rows}', case[3])
                    self.assertEqual(mode == 'hooks', '--dispatch-faults' in case[3])
                    self.assertEqual(kind == 'iq', any(flag.startswith('--quantized-source=') for flag in case[3]))

    def test_bf16_sensitive_flag_requires_actual_tensor_type_count_in_both_reports(self):
        actual, comparison = evidence(True)
        actual['prefill_rows'] = 809
        flags = ['--selected-experts', '--bf16-sensitive', '--prefill-rows=809']
        with self.assertRaises(RuntimeError): self.accept(actual, comparison, flags)
        for report in (actual, comparison):
            report.update(bf16_sensitive_projections=True, bf16_sensitive_tensor_count=18)
        self.accept(actual, comparison, flags)
        for report_name in ('actual', 'comparison'):
            for key, value in [('bf16_sensitive_projections', False), ('bf16_sensitive_projections', 1),
                               ('bf16_sensitive_tensor_count', True), ('bf16_sensitive_tensor_count', 17),
                               ('bf16_sensitive_tensor_count', 19)]:
                reports = {'actual': copy.deepcopy(actual), 'comparison': copy.deepcopy(comparison)}
                reports[report_name][key] = value
                with self.subTest(report=report_name, key=key, value=value), self.assertRaises(RuntimeError):
                    self.accept(reports['actual'], reports['comparison'], flags)
        with self.assertRaises(RuntimeError): self.accept(actual, comparison, ['--selected-experts', '--prefill-rows=809'])

    def test_explicit_case_filter_is_subset_coverage_and_rejects_unknown_names(self):
        base = ["--mode", "glm", "--native-root", "/mnt/e/native", "--output", "/mnt/e/new",
                "--managed-source", "/mnt/e/managed", "--managed-manifest", "/mnt/e/pins.json",
                "--probe-sha256", "a" * 64, "--quantized-source", "/mnt/e/model.gguf", "--plan"]
        with contextlib.redirect_stdout(io.StringIO()) as capture:
            self.assertEqual(0, tool.main(base + ["--only-case", "production-cpu", "--only-case", "hooks-cuda-faults"]))
        plan = json.loads(capture.getvalue())
        self.assertEqual(2, plan["count"])
        self.assertFalse(plan["executed"])
        self.assertEqual(["production-cpu", "hooks-cuda-faults"], [c[0] for c in plan["cases"]])
        with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            tool.main(base + ["--only-case", "nonexistent"])

    def test_exit_zero_ordinary_arena_cannot_certify_windows_only_file_extent_checks(self):
        for name in ("arena-file-q8", "arena-file-iq1"):
            case = dict(exit_code=0, passed=True, skipped_or_unavailable=False)
            tool.annotate_requested_native_coverage(name, case, "PASS ordinary arena precision checks")
            self.assertFalse(case["passed"])
            self.assertTrue(case["skipped_or_unavailable"])
            self.assertFalse(case["registered_file_extents_exercised"])
            portable = dict(exit_code=0, passed=True, skipped_or_unavailable=False)
            tool.annotate_requested_native_coverage(name, portable,
                "PASS: exact file extents, cached reuse, out-of-order staging, partial-read recovery and source release")
            self.assertTrue(portable["passed"])
            self.assertTrue(portable["registered_file_extents_exercised"])
        ordinary = dict(exit_code=0, passed=True)
        tool.annotate_requested_native_coverage("arena-q8", ordinary, "PASS ordinary arena precision checks")
        self.assertTrue(ordinary["passed"])

    def test_accepts_complete_exact_native_and_ledger_evidence(self):
        self.accept(*evidence())
        self.accept(*evidence(True), flags=["--selected-experts"])

    def test_selected_hook_run_requires_unique_id_download_failure_proof(self):
        actual, comparison = evidence(True)
        actual['dispatch_faults_tested'] = True
        flags = ['--selected-experts', '--dispatch-faults']
        actual['executions'].append(final_drain_proof())
        with self.assertRaises(RuntimeError): self.accept(actual, comparison, flags)
        proof = dict(selected_ids_download_failure=True, stage=4, kind=1,
                     cache_unchanged=True, logits_unpublished=True, heads_unpublished=True)
        actual['executions'].append(proof)
        self.accept(actual, comparison, flags)
        for key, value in [('selected_ids_download_failure', False), ('stage', 3), ('kind', True),
                           ('cache_unchanged', False), ('logits_unpublished', False), ('heads_unpublished', False)]:
            bad = copy.deepcopy(actual); bad['executions'][-1][key] = value
            with self.subTest(key=key), self.assertRaises(RuntimeError): self.accept(bad, comparison, flags)
        bad = copy.deepcopy(actual); bad['executions'].append(proof)
        with self.assertRaises(RuntimeError): self.accept(bad, comparison, flags)

    def test_selected_hook_final_drain_proof_requires_last_seam_complete_and_reset_recovery(self):
        actual, comparison = evidence(True)
        actual['dispatch_faults_tested'] = True
        flags = ['--selected-experts', '--dispatch-faults']
        actual['executions'].append(dict(selected_ids_download_failure=True, stage=4, kind=1,
                                         cache_unchanged=True, logits_unpublished=True, heads_unpublished=True))
        with self.assertRaises(RuntimeError): self.accept(actual, comparison, flags)
        proof = final_drain_proof()
        actual['executions'].append(proof)
        self.accept(actual, comparison, flags)
        for key in proof:
            bad = copy.deepcopy(actual); del bad['executions'][-1][key]
            with self.subTest(missing=key), self.assertRaises(RuntimeError): self.accept(bad, comparison, flags)
            value = proof[key]
            for invalid in [False, True] if type(value) is int else [False, 1]:
                bad = copy.deepcopy(actual); bad['executions'][-1][key] = invalid
                with self.subTest(key=key, value=invalid), self.assertRaises(RuntimeError): self.accept(bad, comparison, flags)
        for key, value in [('stage', 4), ('kind', 2), ('participating_rows', 2), ('expert_seams', 1),
                           ('completed_helper_rows', 3), ('completed_helper_rows', 7)]:
            bad = copy.deepcopy(actual); bad['executions'][-1][key] = value
            with self.subTest(key=key, value=value), self.assertRaises(RuntimeError): self.accept(bad, comparison, flags)
        bad = copy.deepcopy(actual); bad['executions'].append(proof)
        with self.assertRaises(RuntimeError): self.accept(bad, comparison, flags)

    def test_quantized_final_drain_proof_requires_all_three_expert_seams(self):
        actual, comparison = evidence(True)
        actual.update(dispatch_faults_tested=True, quantized_source='/mnt/e/model.gguf', expert_groups=3, swiglu_clamp=10)
        flags = ['--selected-experts', '--dispatch-faults', '--quantized-source=/mnt/e/model.gguf']
        actual['executions'].append(dict(selected_ids_download_failure=True, stage=4, kind=1,
                                         cache_unchanged=True, logits_unpublished=True, heads_unpublished=True))
        actual['executions'].append(final_drain_proof())
        with self.assertRaises(RuntimeError): self.accept(actual, comparison, flags)
        actual['executions'][-1] = final_drain_proof(3)
        self.accept(actual, comparison, flags)

    def test_rejects_missing_or_tolerance_based_logit_acceptance(self):
        actual, comparison = evidence()
        for key, value in (("passed", False), ("differing_elements", 1), ("max_abs", 1e-12),
                           ("rmse", 1e-12), ("reference_sha256", "c" * 64), ("logit_elements", 0)):
            with self.subTest(key=key):
                bad = comparison | {key: value}
                with self.assertRaises(RuntimeError):
                    self.accept(actual, bad)
        with self.assertRaises(RuntimeError):
            self.accept(actual, None)

    def test_rejects_unsettled_owners_and_wrong_loaded_native(self):
        actual, comparison = evidence()
        for key in ("Committed", "Reserved"):
            bad = copy.deepcopy(actual)
            bad["final_budget"][0][key] = 1
            with self.assertRaises(RuntimeError):
                self.accept(bad, comparison)
        for key, value in (("path", "/mnt/e/old/libGgmlOps.so"), ("sha256", "c" * 64)):
            bad = copy.deepcopy(actual)
            bad["native"][0][key] = value
            with self.assertRaises(RuntimeError):
                self.accept(bad, comparison)

    def test_rejects_stale_head_even_without_selected_mode(self):
        actual, comparison = evidence()
        bad = copy.deepcopy(actual)
        bad["executions"][-1]["row_departure"]["PositionsAfter"] = [1, 2]
        with self.assertRaises(RuntimeError):
            self.accept(bad, comparison)
    def test_accepts_plain_n3_without_unrequested_selected_departure(self):
        actual, comparison = evidence()
        actual["executions"].pop()
        self.accept(actual, comparison)
        actual["dispatch_faults_tested"] = True
        self.accept(actual, comparison, ["--dispatch-faults"])

    def test_selected_n3_requires_actual_primary_departure_n2(self):
        actual, comparison = evidence(True)
        bad = copy.deepcopy(actual)
        bad["executions"] = [e for e in bad["executions"] if "row_departure" not in e]
        with self.assertRaises(RuntimeError):
            self.accept(bad, comparison, ["--selected-experts"])

    def test_all_modes_require_at_least_four_actual_n3_executions(self):
        for selected in (False, True):
            actual, comparison = evidence(selected)
            actual["executions"].pop(0)
            with self.assertRaises(RuntimeError):
                self.accept(actual, comparison, ["--selected-experts"] if selected else [])

    def test_rejects_silent_coverage_and_minimum_floor_omissions(self):
        actual, comparison = evidence()
        with self.assertRaises(RuntimeError):
            self.accept(actual, comparison, ["--selected-experts"])
        actual["selected_experts_tested"] = True
        with self.assertRaises(RuntimeError):
            self.accept(actual, comparison, ["--selected-experts"])
        actual["executions"].append(dict(minimum_cache_bytes=1))
        with self.assertRaises(RuntimeError):
            self.accept(actual, comparison, ["--selected-experts", "--dispatch-faults"])


if __name__ == "__main__":
    unittest.main()
