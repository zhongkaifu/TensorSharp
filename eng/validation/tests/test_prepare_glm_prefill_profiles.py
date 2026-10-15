import importlib.util
from pathlib import Path
import unittest


SPEC = importlib.util.spec_from_file_location('glm_prefill_profiles_preparation',
    Path(__file__).parents[1] / 'prepare-glm-prefill-profiles.py')
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class ActualInputBindingTests(unittest.TestCase):
    LINE = 'generation.validation.input internal=chat-test session=session-test architecture=glm5-next compacted=False thinking=False maxNewTokens=20000 tokens=154822,1,154841'

    def test_exact_actual_source_and_line(self):
        result = MODULE.extract('noise\n  '+self.LINE+'\n', 'chat-test', 3)
        self.assertEqual(result['Line'], 2)
        self.assertEqual(result['Tokens'], [154822, 1, 154841])
        self.assertEqual(result['SourceMaxNewTokens'], 20000)

    def test_duplicates_missing_or_different_source_shape_rejected(self):
        for text in (self.LINE+'\n'+self.LINE, '', self.LINE.replace('thinking=False', 'thinking=True'),
                     self.LINE.replace('maxNewTokens=20000', 'maxNewTokens=32'),
                     self.LINE.replace('glm5-next', 'qwen4exp')):
            with self.subTest(text=text), self.assertRaises(ValueError):
                MODULE.extract(text, 'chat-test', 3)

    def test_count_vocabulary_and_empty_fields_rejected(self):
        for text, count in ((self.LINE, 4), (self.LINE.replace('1,154841', '154880,154841'), 3),
                            (self.LINE.replace('1,154841', '1,,154841'), 3),
                            (self.LINE.replace('1,154841', '01,154841'), 3)):
            with self.subTest(text=text), self.assertRaises(ValueError):
                MODULE.extract(text, 'chat-test', count)

    def test_probe_arguments_are_unique_pairs(self):
        self.assertEqual(MODULE.arguments(['dotnet','Probe.dll','--task','unknown']), {'--task':'unknown'})
        for command in (['dotnet','Probe.dll','--task'], ['dotnet','Probe.dll','--task','a','--task','b']):
            with self.subTest(command=command), self.assertRaises(ValueError):
                MODULE.arguments(command)

    def test_default_keeps_legacy_command_and_does_not_mutate_template(self):
        original={'--selected':'true','--prefill':'128'}
        result=MODULE.adaptive_options(original,False,False)
        self.assertEqual(result,original)
        self.assertNotIn('--adaptive-prefill',result)
        self.assertIsNot(result,original)

    def test_adaptive_needs_explicit_new_sdk_and_selected_experts(self):
        original={'--selected':'true','--prefill':'512'}
        result=MODULE.adaptive_options(original,True,True)
        self.assertEqual(result['--adaptive-prefill'],'true')
        self.assertEqual(result['--prefill'],'512')
        self.assertNotIn('--adaptive-prefill',original)
        for options,capability in [(original,False),({'--selected':'false'},True)]:
            with self.subTest(options=options,capability=capability),self.assertRaises(ValueError):
                MODULE.adaptive_options(options,True,capability)

    def test_adaptive_template_cannot_enable_it_by_default_or_malformed_flag(self):
        for value in ('true','TRUE','1'):
            with self.subTest(value=value),self.assertRaises(ValueError):
                MODULE.adaptive_options({'--selected':'true','--adaptive-prefill':value},False,True)

    def test_legacy_widths_keep_the_old_sdk_contract(self):
        MODULE.require_wide_adaptive_policy([128, 256, 512], False, {})
        MODULE.require_wide_adaptive_policy([512], True, {'CliAndSessionSetterPresent': True})

    def test_wider_shape_needs_explicit_policy_and_actual_new_setter(self):
        valid = {'CliAndSessionSetterPresent': True, 'WideAdaptiveSessionSetterPresent': True}
        MODULE.require_wide_adaptive_policy([1024], True, valid)
        for enabled, evidence in [(False, valid), (True, {}),
                (True, valid | {'WideAdaptiveSessionSetterPresent': False}),
                (True, valid | {'CliAndSessionSetterPresent': False})]:
            with self.subTest(enabled=enabled, evidence=evidence), self.assertRaises(ValueError):
                MODULE.require_wide_adaptive_policy([512, 1024], enabled, evidence)


if __name__ == '__main__':
    unittest.main()
