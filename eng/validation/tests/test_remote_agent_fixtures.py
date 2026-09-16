import importlib.util
from pathlib import Path
import unittest


def load(name):
    script = Path(__file__).resolve().parents[1] / (name + '.py')
    spec = importlib.util.spec_from_file_location(name.replace('-', '_'), script)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class RemoteAgentFixtureTests(unittest.TestCase):
    def test_original_prompts_and_oracles_are_exactly_shared(self):
        original = load('validate-release-agent-workflows')
        result = load('export-remote-agent-fixtures').export()
        originals = [case for case in result['cases'] if case['variant'] == 'original']
        self.assertEqual(30, len(originals))
        for case in originals:
            self.assertEqual(original.CASES[case['scenario']], case['spec'])
        self.assertEqual('prepared-not-executed', result['status'])
        for case in result['cases']:
            expected = original.case_spec(case['scenario'], case['trial'], distinct_inputs=case['variant'] == 'distinct')
            self.assertEqual(expected, case['spec'])
            self.assertEqual(expected['prompt'].encode('utf-8'), case['spec']['prompt'].encode('utf-8'))
            self.assertRegex(case['trial'], r'^c[14]-i[0-3]$')

    def test_concurrent_inputs_and_artifact_probes_are_distinct(self):
        result = load('export-remote-agent-fixtures').export()
        self.assertEqual(46, len(result['cases']))
        distinct = [case for case in result['cases'] if case['variant'] == 'distinct']
        self.assertEqual(16, len(distinct))
        self.assertEqual({'skill_script_run', 'shell_run', 'code_generation_run', 'code_edit_run'}, {case['scenario'] for case in distinct})
        waves = {}
        for case in result['cases']:
            waves.setdefault(case['wave'], []).append(case)
        for cases in waves.values():
            self.assertEqual(cases[0]['concurrency'], len(cases))
            if cases[0]['variant'] == 'distinct':
                self.assertEqual(len(cases), len({case['spec']['prompt'] for case in cases}))
            if cases[0]['variant'] == 'distinct' and cases[0]['concurrency'] == 4 and 'artifact' in cases[0]['spec']:
                self.assertEqual(4, len({case['spec']['artifact']['probe'] for case in cases}))

    def test_bounded_start_has_six_original_and_six_distinct_cases(self):
        result = load('export-remote-agent-fixtures').export(smoke=True)
        self.assertEqual(12, len(result['cases']))
        self.assertTrue(all(case['concurrency'] == 1 for case in result['cases']))


if __name__ == '__main__':
    unittest.main()
