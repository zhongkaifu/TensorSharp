"""Offline gates for the owner; no SSH, process creation or inference."""
import copy
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

spec = importlib.util.spec_from_file_location('remote_owner', Path(__file__).parents[1] / 'run-remote-agent-over-ssh.py')
owner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(owner)


class RemoteAgentOwnerTests(unittest.TestCase):
    def test_remote_helper_sources_compile_without_execution(self):
        compile(owner.REMOTE_OBSERVER, '<remote-observer>', 'exec')
        compile(owner.REMOTE_COMPLETE, '<remote-completion>', 'exec')

    def identity(self):
        return {'remote_pid': 123, 'remote_start_ticks': 456, 'remote_boot_id': 'boot-fixture',
                'mapped_native_libraries': {'/fixture/libGgmlOps.so': 'a' * 64},
                'mapped_managed_libraries': {'/fixture/TensorSharp.Models.dll': 'b' * 64},
                'owned_listening_sockets': [{'inode': '42'}], 'remote_port': 5100}

    def test_actual_binding_accepts_same_process_libraries_and_endpoint(self):
        identity = self.identity()
        owner.verify_identity(identity, copy.deepcopy(identity), 'a' * 64, 5100)

    def test_wrong_native_pin_is_rejected_even_when_ready_file_matches_actual(self):
        identity = self.identity()
        with self.assertRaisesRegex(ValueError, 'native pin'):
            owner.verify_identity(identity, copy.deepcopy(identity), 'c' * 64, 5100)

    def test_ready_file_binds_the_exact_campaign_nonce_before_any_ssh(self):
        identity = {**self.identity(), 'nonce': 'fresh-campaign'}
        owner.verify_ready_identity(identity, 'fresh-campaign')
        for nonce in (None, 'stale-campaign'):
            with self.subTest(nonce=nonce), self.assertRaisesRegex(ValueError, 'nonce'):
                owner.verify_ready_identity({**identity, 'nonce': nonce}, 'fresh-campaign')
        with self.assertRaisesRegex(ValueError, 'Actual'):
            owner.verify_ready_identity({**identity, 'scripted_fixture': True}, 'fresh-campaign')

    def test_process_library_and_endpoint_changes_fail(self):
        expected = self.identity()
        changes = {'remote_pid': 124, 'remote_start_ticks': 457, 'remote_boot_id': 'rebooted',
                   'mapped_native_libraries': {'/fixture/libGgmlOps.so': 'c' * 64},
                   'mapped_managed_libraries': {'/fixture/TensorSharp.Models.dll': 'c' * 64},
                   'owned_listening_sockets': [], 'remote_port': 5101}
        for key, value in changes.items():
            with self.subTest(key=key):
                actual = {**expected, key: value}
                with self.assertRaises(ValueError):
                    owner.verify_identity(actual, expected, 'a' * 64, 5100)

    def test_lazy_managed_load_requires_the_pinned_application_file_hash(self):
        expected = self.identity()
        expected['available_managed_libraries'] = {
            **expected['mapped_managed_libraries'], '/fixture/TensorSharp.AgentHost.dll': 'c' * 64}
        actual = copy.deepcopy(expected)
        actual['mapped_managed_libraries']['/fixture/TensorSharp.AgentHost.dll'] = 'c' * 64
        owner.verify_identity(actual, expected, 'a' * 64, 5100)
        actual['mapped_managed_libraries']['/fixture/TensorSharp.AgentHost.dll'] = 'd' * 64
        with self.assertRaisesRegex(ValueError, 'pinned application'):
            owner.verify_identity(actual, expected, 'a' * 64, 5100)

    def test_46_scope_requires_original_and_distinct_trial_identities(self):
        original = {'skill_selection', 'skill_run', 'skill_script_run', 'shell_run', 'code_generation_run', 'code_edit_run'}
        distinct = original - {'skill_selection', 'skill_run'}
        cases = [{'variant': 'original', 'scenario': name, 'trial': trial} for name in original
                 for trial in ('c1-i0', 'c4-i0', 'c4-i1', 'c4-i2', 'c4-i3')]
        cases += [{'variant': 'distinct', 'scenario': name, 'trial': f'c4-i{i}'} for name in distinct for i in range(4)]
        owner.verify_release_fixtures({'cases': cases})
        with self.assertRaises(ValueError):
            owner.verify_release_fixtures({'cases': cases[:-1]})
        replaced = copy.deepcopy(cases)
        replaced[-1] = replaced[0]
        with self.assertRaises(ValueError):
            owner.verify_release_fixtures({'cases': replaced})

    def test_failed_or_malformed_workflow_still_captures_remote_postcheck(self):
        for raw in ('{"run_complete":false,"cases":[]}', '{truncated'):
            with self.subTest(raw=raw), tempfile.TemporaryDirectory() as temporary:
                output = Path(temporary)
                (output / 'client').mkdir()
                (output / 'client/profile.json').write_text(json.dumps({'client_exit_code': 0, 'independent_code_exit_code': 0}))
                (output / 'client/actual-agent-workflows.json').write_text(raw)
                observations = []

                def observe():
                    observations.append('captured')
                    return self.identity()

                completion, details = owner.capture_completion(output, 'nonce-fixture', self.identity(), 'a' * 64, 5100, observe)
                self.assertEqual(['captured'], observations)
                self.assertTrue(completion['remote_binding_postcheck_passed'])
                self.assertNotEqual(0, completion['client_exit_code'])
                self.assertRegex(completion['client_report_sha256'], '^[0-9a-f]{64}$')
                self.assertIn('workflow_validation_error', details)
                self.assertTrue((output / 'remote-after.json').is_file())


if __name__ == '__main__':
    unittest.main()
