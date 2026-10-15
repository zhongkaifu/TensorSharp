import copy
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

SPEC = importlib.util.spec_from_file_location('maya_rescore', Path(__file__).parents[1] / 'rescore-maya-report.py')
MOD = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MOD)


class MayaRescoreTests(unittest.TestCase):
    def report(self):
        return dict(completed=True, process_success=True, exit_code=0, identities_unchanged=True,
            errors=[], timed_out=False, prompt_tokenizer_exact=True, quality_passed=False,
            plan=dict(repetitions=1, eos_ids=[9], case='extraction'),
            reference=dict(name='extraction', prompt=MOD.reporter.fixed_tasks.PROMPTS['extraction'],
                rendered_prompt='<|assistant|><think>',
                prompt_tokens=[1], generated_tokens=[2, 9], passed=True, eos=True, parsed_text='cedar, maple'),
            rows=[dict(tokens=[3, 9], generated=2, prompt_tokens=1, reused=0, drafts_offered=0,
                drafts_accepted=0, finish='stop', quality=dict(passed=False),
                semantic=dict(passed=False, content='cedar,maple', raw_text='reason</think>cedar,maple', thinking_closed=True, tool_calls=[]))])

    def test_original_is_unchanged_and_pinned_alongside_new_conclusion(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / 'report.json'
            raw = json.dumps(self.report()).encode()
            path.write_bytes(raw)
            result = MOD.audit_report(path)
            self.assertEqual(raw, path.read_bytes())
            self.assertEqual(MOD.sha_bytes(raw), result['source_report']['sha256'])
            self.assertFalse(result['original']['quality_passed'])
            self.assertTrue(result['independent']['independent_quality_passed'])
            self.assertEqual(3, len(result['checker_pins']))

    def test_missing_or_incomplete_cannot_pass_offline(self):
        for change in [dict(completed=False), dict(timed_out=True), dict(identities_unchanged=False)]:
            data = self.report(); data.update(change)
            with tempfile.TemporaryDirectory() as temporary:
                path = Path(temporary) / 'report.json'; path.write_text(json.dumps(data))
                self.assertFalse(MOD.audit_report(path)['independent']['independent_quality_passed'])


if __name__ == '__main__':
    unittest.main()
