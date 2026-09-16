"""Final artifacts must replace drafts without accepting a stale passing copy."""
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace
import unittest

spec = importlib.util.spec_from_file_location('agent_workflows', Path(__file__).with_name('validate-release-agent-workflows.py'))
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


class ArtifactVersions(unittest.TestCase):
    expected = {'n': 37, 'sum': 703, 'tests_passed': True}

    def check_versions(self, contents, names=None):
        artifacts, replies = {}, {}
        for index, content in enumerate(contents):
            url = f'/api/code/artifacts/version-{index}/result.json'
            artifacts[url] = SimpleNamespace(name=names[index] if names else 'result.json', url=url)
            replies[url] = (200, 'application/json', json.dumps(content).encode())
        result = {'artifacts': []}
        module.validate_result_artifact(SimpleNamespace(download=lambda url, *_: replies[url]), artifacts, self.expected, result)
        return result

    def test_corrected_final_file_passes_and_keeps_history(self):
        result = self.check_versions([703, self.expected])
        self.assertEqual(len(result['result_artifact_versions']), 2)
        self.assertEqual(result['artifacts'][0]['url'], '/api/code/artifacts/version-1/result.json')

    def test_later_wrong_file_cannot_pass_using_earlier_correct_copy(self):
        with self.assertRaisesRegex(RuntimeError, 'final result artifact failed'):
            self.check_versions([self.expected, 703])

    def test_similar_name_cannot_satisfy_required_result(self):
        with self.assertRaisesRegex(RuntimeError, 'No result.json'):
            self.check_versions([self.expected], ['unrelated_result.json'])

    def test_unrelated_later_file_does_not_replace_result(self):
        result = self.check_versions([self.expected, 703], ['result.json', 'debug.json'])
        self.assertEqual(result['artifacts'][0]['content'], self.expected)


if __name__ == '__main__':
    unittest.main()
