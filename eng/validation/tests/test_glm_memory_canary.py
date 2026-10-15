import copy
import importlib.util
from pathlib import Path
import unittest

spec = importlib.util.spec_from_file_location('glm_canary', Path(__file__).parents[1]/'run-glm-memory-canary.py')
canary = importlib.util.module_from_spec(spec)
spec.loader.exec_module(canary)


class ParentLimitProofTests(unittest.TestCase):
    def setUp(self):
        self.before = {level: {'memory.events.local': 'max 0\noom 0\noom_kill 0', 'memory.swap.current': '0'}
                       for level in ('parent', 'child')}
        self.after = copy.deepcopy(self.before)
        self.after['parent']['memory.events.local'] = 'max 23\noom 1\noom_kill 0'
        self.after['child']['memory.events.local'] = 'max 0\noom 0\noom_kill 1'

    def test_local_parent_limit_kill(self):
        self.assertTrue(canary.proves_parent_limit(self.before, self.after, -9))

    def test_global_or_ancestor_kill_is_not_local_limit_proof(self):
        self.after['parent']['memory.events.local'] = 'max 0\noom 0\noom_kill 1'
        self.assertFalse(canary.proves_parent_limit(self.before, self.after, -9))

    def test_child_limit_hit_is_not_parent_only_proof(self):
        self.after['child']['memory.events.local'] = 'max 1\noom 1\noom_kill 1'
        self.assertFalse(canary.proves_parent_limit(self.before, self.after, -9))

    def test_old_counters_do_not_prove_new_kill(self):
        self.assertFalse(canary.proves_parent_limit(self.after, self.after, -9))

    def test_success_or_swapping_is_not_requested_enforcement(self):
        self.assertFalse(canary.proves_parent_limit(self.before, self.after, 0))
        self.after['parent']['memory.swap.current'] = '4096'
        self.assertFalse(canary.proves_parent_limit(self.before, self.after, -9))


if __name__ == '__main__':
    unittest.main()
