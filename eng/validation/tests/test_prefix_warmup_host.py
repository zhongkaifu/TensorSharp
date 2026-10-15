"""Receipt/lifecycle controller tests only; no actual Host/model/GPU."""
import copy
import importlib.util
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

spec = importlib.util.spec_from_file_location('prefix_warmup_host_tests', Path(__file__).parents[1] / 'prefix-warmup-host.py')
host = importlib.util.module_from_spec(spec)
spec.loader.exec_module(host)


class Fixture:
    def __init__(self, root, version=22):
        self.managed, self.native, self.stage = (root / key for key in ('managed', 'native', 'stage'))
        self.plan = dict(schema='tensorsharp.prefix-warmup-host-plan.v1', activation='ROOT-CONTROLLER-ONLY', sdk_version=version,
            require_publication_complete=version >= 22, managed_directory=str(self.managed), native_directory=str(self.native),
            stage_directory=str(self.stage), upstream_revision='clean-revision', production_capabilities={'adapter': 7},
            source_pins={str(path.resolve()): host.pin(path) for path in host.SOURCE_FILES})
        publications = {}
        for name in ('host', 'probe'):
            folder = self.managed / ('host-publish' if name == 'host' else 'glm-session-sdk-publish')
            publish = folder / 'publish'; publish.mkdir(parents=True)
            (publish / ('TensorSharp.Server.Host.exe' if name == 'host' else 'probe.dll')).write_bytes((name + '-compiled').encode())
            publications[name] = host.closure(publish)
            receipt = dict(publish_complete=True, exit_code=0, compiled_unchanged=True, user_bin_unchanged=True,
                project_unchanged=True, source_unchanged=True, native_injected=False, publish=str(publish),
                published_files={key: dict(bytes=value['Bytes'], sha256=value['Sha256']) for key, value in publications[name].items()})
            self.write(folder / 'receipt.json', receipt)
        probe_receipt = self.managed / 'glm-session-sdk-publish/receipt.json'
        tested = {key: dict(bytes=value['Bytes'], sha256=value['Sha256']) for key, value in publications['probe'].items()}
        self.write(self.managed / 'published-probe-self-test-receipt.json', dict(passed=True, published_unchanged=True,
            published_before=tested, published_after=tested, publish_receipt_sha256=host.bench.sha(probe_receipt),
            cases=[dict(platform=platform, exit_code=0, owned_process_reaped=True) for platform in ('windows', 'wsl')]))
        self.write(self.managed / 'final-summary.json', dict(status='passed'))
        delivery = dict(passed=True, cpu_version=version, source_dll_control_userbin_bookends_equal=True,
            published_probe_windows_wsl_selftest_passed=True, host_publish=str(self.managed / 'host-publish/publish'),
            probe_publish=str(self.managed / 'glm-session-sdk-publish/publish'),
            receipt_pins={name: host.bench.sha(self.managed / name) for name in
                          ('final-summary.json', 'host-publish/receipt.json', 'glm-session-sdk-publish/receipt.json', 'published-probe-self-test-receipt.json')})
        if version >= 22: delivery['publication_complete'] = True
        self.write(self.managed / 'final-delivery.json', delivery)
        (self.native / 'build-production').mkdir(parents=True)
        binary = self.native / 'build-production/GgmlOps.dll'; binary.write_bytes(b'production-native')
        bookend = dict(upstream=dict(revision='clean-revision', status=''))
        self.write(self.native / 'build-summary.json', dict(passed=True, before=bookend, after=bookend,
            no_upstream_compilation=True, production_constant_capability_queries=dict(adapter=7),
            production_artifacts={'GgmlOps.dll': host.bench.sha(binary)}))
        server = self.stage / 'host'; server.mkdir(parents=True)
        for p in (self.managed / 'host-publish/publish').iterdir(): (server / p.name).write_bytes(p.read_bytes())
        (server / 'GgmlOps.dll').write_bytes(binary.read_bytes())
        self.write(self.stage / 'deployment-receipt.json', [])
        self.write(self.stage / 'stage-summary.json', dict(Passed=True, HostDirectory=str(server),
            Original=dict(host=publications['host']), HostFiles=host.closure(server),
            CompleteDeliverySha256=host.bench.sha(self.managed / 'final-delivery.json'),
            PublishReceiptSha256={name: host.bench.sha(self.managed / ('host-publish/receipt.json' if name == 'host' else 'glm-session-sdk-publish/receipt.json')) for name in ('host', 'probe')},
            PublishedSelfTestSha256=host.bench.sha(self.managed / 'published-probe-self-test-receipt.json'),
            NativeBuildSummarySha256=host.bench.sha(self.native / 'build-summary.json'),
            ManagedSummarySha256=host.bench.sha(self.managed / 'final-summary.json')))

    def write(self, path, value):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(value), encoding='utf-8')

    def mutate(self, path, transform):
        value = host.read(path); transform(value); self.write(path, value)


class ReceiptTests(unittest.TestCase):
    def test_complete_candidate_and_legacy_baseline_are_qualified_without_process_launch(self):
        for version in (20, 22):
            with self.subTest(version=version), TemporaryDirectory() as directory, patch.object(host.subprocess, 'Popen', side_effect=AssertionError('must not launch')):
                fixture = Fixture(Path(directory), version)
                result = host.qualify(fixture.plan)
                self.assertEqual(host.closure(fixture.stage / 'host'), result['expected_closure'])
                self.assertIn('production_native', result['receipt_and_tool_pins'])

    def test_cpu_summary_without_complete_delivery_never_qualifies(self):
        with TemporaryDirectory() as directory:
            fixture = Fixture(Path(directory)); (fixture.managed / 'final-delivery.json').unlink()
            with self.assertRaisesRegex(ValueError, 'complete delivery'): host.qualify(fixture.plan)

    def test_publication_complete_missing_or_false_is_rejected_for_candidate(self):
        for value in (None, False):
            with self.subTest(value=value), TemporaryDirectory() as directory:
                fixture = Fixture(Path(directory))
                fixture.mutate(fixture.managed / 'final-delivery.json', lambda d: d.update(publication_complete=value))
                with self.assertRaisesRegex(ValueError, 'publication is not complete'): host.qualify(fixture.plan)

    def test_wrong_sdk_version_is_rejected(self):
        with TemporaryDirectory() as directory:
            fixture = Fixture(Path(directory)); fixture.plan['sdk_version'] = 20
            with self.assertRaisesRegex(ValueError, 'version differs'): host.qualify(fixture.plan)

    def test_mutated_completed_receipt_is_rejected(self):
        with TemporaryDirectory() as directory:
            fixture = Fixture(Path(directory))
            fixture.mutate(fixture.managed / 'host-publish/receipt.json', lambda d: d.update(publish_complete=False))
            with self.assertRaisesRegex(ValueError, 'receipt pin differs'): host.qualify(fixture.plan)

    def test_staged_managed_or_native_payload_mutation_is_rejected(self):
        for name in ('TensorSharp.Server.Host.exe', 'GgmlOps.dll'):
            with self.subTest(name=name), TemporaryDirectory() as directory:
                fixture = Fixture(Path(directory)); (fixture.stage / 'host' / name).write_bytes(b'changed')
                with self.assertRaisesRegex(ValueError, 'staged Host closure differs'): host.qualify(fixture.plan)

    def test_unexpected_static_binary_is_rejected_but_run_logs_are_not_sdk_members(self):
        with TemporaryDirectory() as directory:
            fixture = Fixture(Path(directory)); logs = fixture.stage / 'host/logs'; logs.mkdir(); (logs / 'dynamic.jsonl').write_text('{}')
            host.qualify(fixture.plan)
            (fixture.stage / 'host/unqualified.dll').write_bytes(b'new')
            with self.assertRaisesRegex(ValueError, 'staged Host closure differs'): host.qualify(fixture.plan)

    def test_mutated_published_host_payload_is_rejected(self):
        with TemporaryDirectory() as directory:
            fixture = Fixture(Path(directory)); (fixture.managed / 'host-publish/publish/TensorSharp.Server.Host.exe').write_bytes(b'changed')
            with self.assertRaisesRegex(ValueError, 'published closure differs'): host.qualify(fixture.plan)

    def test_native_clean_revision_and_capability_are_required(self):
        for key, value, pattern in (('upstream_revision', 'different', 'dependency revision'), ('production_capabilities', {'adapter': 8}, 'capability differs')):
            with self.subTest(key=key), TemporaryDirectory() as directory:
                fixture = Fixture(Path(directory)); fixture.plan[key] = value
                with self.assertRaisesRegex(ValueError, pattern): host.qualify(fixture.plan)

    def test_missing_or_mutated_executing_controller_and_helpers_are_rejected(self):
        for source in (host.SOURCE_FILES[0], host.SOURCE_FILES[1], host.SOURCE_FILES[-1]):
            for mutation in ('remove', 'digest'):
                with self.subTest(source=source, mutation=mutation), TemporaryDirectory() as directory:
                    fixture = Fixture(Path(directory)); key = str(source.resolve())
                    if mutation == 'remove': fixture.plan['source_pins'].pop(key)
                    else: fixture.plan['source_pins'][key]['sha256'] = '0' * 64
                    with self.assertRaisesRegex(ValueError, 'validation controller/helper|validation source differs'): host.qualify(fixture.plan)

    def test_receipt_parent_traversal_and_absolute_paths_are_rejected(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            for value in ('../secret.json', str(root / 'absolute.json')):
                with self.assertRaises(ValueError): host.safe_relative(root, value)


class LifecycleTests(unittest.TestCase):
    def test_child_only_context_log_and_prefix_storage_overrides_preserve_other_environment(self):
        inherited = {'PATH': 'test', 'OTHER': 'kept'}
        actual, settings = host.child_environment(inherited, Path('run/log'), Path('run/prefix'))
        self.assertEqual('32768', actual['MAX_CONTEXT']); self.assertEqual('kept', actual['OTHER'])
        self.assertEqual({'PATH': 'test', 'OTHER': 'kept'}, inherited)
        self.assertEqual({'MAX_CONTEXT', 'TENSORSHARP_LOG_DIR', 'TENSORSHARP_PREFIX_CACHE_DIR'}, set(settings))

    def test_inherited_inference_overrides_fail_without_silently_sanitizing_them(self):
        for key in ('TS_HOST_MOE_TIMING', 'TENSORSHARP_LOG_LEVEL', 'GGML_CUDA_FORCE_MMQ', 'MAX_CONTEXT', 'OMP_NUM_THREADS', 'CUDA_VISIBLE_DEVICES'):
            with self.subTest(key=key), self.assertRaisesRegex(ValueError, 'Inherited inference override'):
                host.child_environment({key: '1'}, Path('log'), Path('prefix'))

    def test_off_arm_sets_child_only_explicit_startup_preference(self):
        inherited = {'PATH': 'test', 'OTHER': 'kept'}
        actual, settings = host.child_environment(inherited, Path('log'), Path('prefix'), '0')
        self.assertEqual(actual['TS_STARTUP_WARMUP_THINKING'], '0')
        self.assertEqual(settings['TS_STARTUP_WARMUP_THINKING'], '0')
        self.assertNotIn('TS_STARTUP_WARMUP_THINKING', inherited)

    def test_default_on_arm_leaves_startup_override_absent(self):
        actual, settings = host.child_environment({'PATH': 'test'}, Path('log'), Path('prefix'))
        self.assertNotIn('TS_STARTUP_WARMUP_THINKING', actual)
        self.assertNotIn('TS_STARTUP_WARMUP_THINKING', settings)

    def test_token_trace_is_explicit_and_changes_only_owned_child_environment(self):
        inherited = {'PATH': 'test', 'OTHER': 'kept'}
        actual, settings = host.child_environment(inherited, Path('log'), Path('prefix'), '0', True)
        self.assertEqual(actual['TS_VALIDATION_TOKEN_TRACE'], '1')
        self.assertEqual(settings['TS_VALIDATION_TOKEN_TRACE'], '1')
        self.assertEqual(inherited, {'PATH': 'test', 'OTHER': 'kept'})
        self.assertTrue(host.token_trace_control(True)['trace_disqualifies_performance_comparison'])
        self.assertNotIn('performance_comparison_qualified', host.token_trace_control(False))

    def test_token_trace_default_stays_absent_and_inherited_trace_is_refused(self):
        actual, settings = host.child_environment({'PATH': 'test'}, Path('log'), Path('prefix'))
        self.assertNotIn('TS_VALIDATION_TOKEN_TRACE', actual)
        self.assertNotIn('TS_VALIDATION_TOKEN_TRACE', settings)
        self.assertFalse(host.token_trace_control(False)['trace_disqualifies_performance_comparison'])
        with self.assertRaisesRegex(ValueError, 'Inherited inference override'):
            host.child_environment({'TS_VALIDATION_TOKEN_TRACE': '1'}, Path('log'), Path('prefix'), validation_token_trace=True)

    def test_token_trace_refuses_nonboolean_plan_values(self):
        for value in (None, 0, 1, 'true', '1', ''):
            with self.subTest(value=value), self.assertRaisesRegex(ValueError, 'explicit boolean'):
                host.child_environment({}, Path('log'), Path('prefix'), validation_token_trace=value)

    def test_long_new_chat_configuration_is_bounded_and_separate_from_continuations(self):
        self.assertEqual(host.long_new_chat_limit({}), (None, None))
        plan = dict(thinking='on', output_gate='stream', long_new_chat_max_tokens=256, long_new_chat_expected_cached_tokens=1332)
        self.assertEqual(host.long_new_chat_limit(plan), (256, 1332))
        self.assertEqual(host.http_timeout({}), 240)
        self.assertEqual(host.http_timeout(dict(http_timeout_seconds=600)), 600)
        for timeout in (True, '600', 0, -1, 1201, float('nan'), float('inf')):
            with self.subTest(timeout=timeout), self.assertRaisesRegex(ValueError, 'finite and bounded'):
                host.http_timeout(dict(http_timeout_seconds=timeout))
        for changes in (dict(thinking='off'), dict(output_gate='visible'), dict(cancellation=True),
                        dict(client_visible_follow_up=True), dict(long_new_chat_max_tokens=True),
                        dict(long_new_chat_max_tokens=512), dict(long_new_chat_expected_cached_tokens=0),
                        dict(long_new_chat_expected_cached_tokens=True), dict(long_new_chat_expected_cached_tokens='1332')):
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                host.long_new_chat_limit(dict(plan, **changes))

    def test_fresh_host_log_selection_accepts_normal_size_and_utc_rotation(self):
        with TemporaryDirectory() as directory:
            root=Path(directory)
            for name in ('tensorsharp-server-20261014.jsonl','tensorsharp-server-20261014-0001.jsonl','tensorsharp-server-20261015.jsonl'):
                (root/name).write_text('{}\n','utf-8')
            self.assertEqual(host.server_log_path(root),root/'tensorsharp-server-20261014.jsonl')
            (root/'tensorsharp-server-20261017.jsonl').write_text('{}\n','utf-8')
            with self.assertRaisesRegex(ValueError,'adjacent2'):host.server_log_path(root)

    def test_startup_preference_requires_explicit_string_boolean(self):
        for value in (True, False, 0, 1, '', 'true', 'off', ' 0'):
            with self.subTest(value=value), self.assertRaisesRegex(ValueError, 'explicit0/1'):
                host.child_environment({}, Path('log'), Path('prefix'), value)

    def test_inherited_startup_preference_is_not_silently_overridden(self):
        with self.assertRaisesRegex(ValueError, 'Inherited inference override'):
            host.child_environment({'TS_STARTUP_WARMUP_THINKING': '1'}, Path('log'), Path('prefix'), '0')

    def test_single_variant_ready_state_binds_default_on_and_explicit_off(self):
        for override, expected in ((None, True), ('0', False), ('1', True)):
            with self.subTest(override=override):
                plan = dict(require_single_preferred_variant=True, startup_warmup_thinking=override)
                state = dict(startupWarmup=dict(completed=True, preferredThinking=expected,
                    singleVariantReason='single-public-checkpoint-budget',
                    stages=[dict(thinking=expected, completed=True)]))
                host.thinking_startup(state, plan)
                state['startupWarmup']['stages'][0]['thinking'] = not expected
                with self.assertRaisesRegex(ValueError, 'single preferred'):
                    host.thinking_startup(state, plan)

    def test_single_variant_gate_refuses_missing_or_multiple_stage_state(self):
        plan = dict(require_single_preferred_variant=True)
        for state in ({}, dict(startupWarmup=dict(completed=True, preferredThinking=True,
            singleVariantReason='single-public-checkpoint-budget', stages=[dict(thinking=True, completed=True)] * 2))):
            with self.subTest(state=state), self.assertRaisesRegex(ValueError, 'single preferred'):
                host.thinking_startup(state, plan)

    def test_separate_followup_limit_is_bounded_and_requires_a_real_requested_check(self):
        self.assertIsNone(host.followup_token_limit({}))
        for value in (8, 16, 64, 128):
            self.assertEqual(host.followup_token_limit(dict(client_visible_follow_up=True, follow_up_max_tokens=value)), value)
        with self.assertRaisesRegex(ValueError, 'explicit client-visible'):
            host.followup_token_limit(dict(follow_up_max_tokens=128))

    def test_followup_limit_rejects_unbounded_boolean_and_string_values(self):
        for value in (True, False, '128', 0, -1, 32, 129, 20000):
            with self.subTest(value=value), self.assertRaisesRegex(ValueError, 'bounded followup'):
                host.followup_token_limit(dict(client_visible_follow_up=True, follow_up_max_tokens=value))

    def test_text_only_readiness_requires_correct_model_unified_architecture_and_completed_warmup(self):
        model = Path('test.gguf')
        state = dict(unifiedMemory=True, architecture='architecture', loaded=model.name, visionReady=False, startupWarmup=dict(completed=True))
        self.assertTrue(host.startup_state(state, model, 'architecture'))
        for key, value in (('visionReady', True), ('unifiedMemory', False), ('loaded', 'other.gguf'), ('startupWarmup', {})):
            wrong = copy.deepcopy(state); wrong[key] = value
            self.assertFalse(host.startup_state(wrong, model, 'architecture'))

    def test_identity_change_is_rejected_before_listener_or_native_claim(self):
        p = Mock(); p.is_running.return_value = True; p.create_time.return_value = 20; p.exe.return_value = str(Path('Host.exe').resolve())
        with self.assertRaisesRegex(ValueError, 'identity changed'): host.process_identity(p, Path('Host.exe'), 10)

    def test_foreign_listener_is_rejected(self):
        row = SimpleNamespace(status=host.bench.psutil.CONN_LISTEN, laddr=SimpleNamespace(port=12345, ip='127.0.0.1'), pid=99)
        with patch.object(host.bench.psutil, 'net_connections', return_value=[row]):
            with self.assertRaisesRegex(ValueError, 'exclusively owned'): host.listener(SimpleNamespace(pid=98), 12345)

    def test_occupied_gpu_is_rejected_without_stopping_anything(self):
        with patch.object(host.bench.psutil, 'net_connections', return_value=[]), patch.object(host.bench.psutil, 'process_iter', return_value=[]), \
             patch.object(host, 'gpu_background', return_value=([dict(pid=5678)], {})), patch.object(host.subprocess, 'Popen', side_effect=AssertionError('must not launch')):
            with self.assertRaisesRegex(ValueError, 'GPU owners differ'): host.quiet_lane(12345, [])

    def test_explicit_wddm_desktop_allowance_is_identity_bound_not_an_idle_claim(self):
        row = dict(pid=23, birth=12.5, name='desktop.exe', executable=str(Path('desktop.exe').resolve()))
        with patch.object(host, 'gpu_background', return_value=([row], {'stdout': '23, [Insufficient Permissions]'})):
            result = host.gpu_allowance([row])
        self.assertEqual([row], result['owners']); self.assertIn('not an idle or exclusive GPU claim', result['scope'])

    def test_new_or_reused_background_gpu_owner_fails_exact_snapshot(self):
        row = dict(pid=23, birth=12.5, name='desktop.exe', executable=str(Path('desktop.exe').resolve()))
        for actual in ([{**row, 'birth': 99}], [row, {**row, 'pid': 24}], []):
            with self.subTest(actual=actual), patch.object(host, 'gpu_background', return_value=(actual, {})):
                with self.assertRaisesRegex(ValueError, 'GPU owners differ'): host.gpu_allowance([row])

    def test_model_or_missing_identity_cannot_be_allowed_as_desktop_background(self):
        row = dict(pid=23, birth=12.5, name='GlmSessionBudgetProbe.exe', executable=str(Path('probe.exe').resolve()))
        for value in ([row], [{**row, 'name': 'desktop.exe', 'birth': None}], None):
            with self.subTest(value=value), self.assertRaises(ValueError): host.gpu_allowance(value)

    def test_activation_mismatch_cannot_launch_a_process(self):
        with TemporaryDirectory() as directory:
            plan = Path(directory) / 'plan.json'; plan.write_text(json.dumps(dict(activation='ROOT-ACTUAL')))
            with patch.object(host.subprocess, 'Popen', side_effect=AssertionError('must not launch')):
                with self.assertRaisesRegex(ValueError, 'exact root activation'): host.main(['--plan', str(plan), '--activation', 'wrong'])

    def test_cleanup_freezes_live_owned_tree_then_stops_only_proven_descendants(self):
        root, child, unrelated = Mock(), Mock(), Mock()
        for process, pid, exe in ((root, 1, 'Host.exe'), (child, 2, 'owned-tool.exe'), (unrelated, 3, 'user.exe')):
            process.pid = pid; process.is_running.return_value = True; process.create_time.return_value = pid * 10
            process.exe.return_value = str(Path(exe).resolve())
        root.children.return_value = [child]
        tracker = host.OwnedDescendants(root, Path('Host.exe'), 10)
        with patch.object(host.bench.psutil, 'wait_procs', return_value=([], [])):
            result = tracker.finish()
        self.assertTrue(result['passed']); self.assertTrue(result['root_frozen_for_cleanup'])
        root.suspend.assert_called_once(); child.suspend.assert_called_once(); child.terminate.assert_called_once()
        root.terminate.assert_not_called(); unrelated.terminate.assert_not_called(); unrelated.suspend.assert_not_called()
        self.assertEqual([2], [row['pid'] for row in result['observed']])

    def test_reused_descendant_identity_is_never_stopped_and_cleanup_fails(self):
        root, child = Mock(), Mock()
        root.pid = 1; root.is_running.return_value = True; root.create_time.return_value = 10
        root.exe.return_value = str(Path('Host.exe').resolve()); root.children.return_value = []
        child.pid = 2; child.is_running.return_value = True; child.create_time.return_value = 99
        child.exe.return_value = str(Path('user.exe').resolve())
        tracker = host.OwnedDescendants(root, Path('Host.exe'), 10)
        tracker.seen[(2, 20)] = (child, dict(pid=2, birth=20, executable=str(Path('owned-tool.exe').resolve())))
        with patch.object(host.bench.psutil, 'wait_procs', return_value=([], [])):
            result = tracker.finish()
        self.assertFalse(result['passed']); self.assertIn('refused to stop PID 2', ' '.join(result['errors']))
        child.suspend.assert_not_called(); child.terminate.assert_not_called(); child.kill.assert_not_called()

    def test_dead_root_cannot_establish_exhaustive_descendant_cleanup(self):
        root = Mock(); root.is_running.return_value = False
        tracker = host.OwnedDescendants(root, Path('Host.exe'), 10)
        with patch.object(host.bench.psutil, 'wait_procs', return_value=([], [])):
            result = tracker.finish()
        self.assertFalse(result['passed']); self.assertFalse(result['root_frozen_for_cleanup'])
        root.suspend.assert_not_called(); root.children.assert_not_called()


if __name__ == '__main__': unittest.main()
