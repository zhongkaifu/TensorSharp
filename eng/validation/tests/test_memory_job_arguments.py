"""Pure CLI checks; actual Job enforcement requires the Windows integration run."""
import importlib.util
from pathlib import Path
import unittest

path = Path(__file__).resolve().parents[1] / 'run-memory-job.py'
spec = importlib.util.spec_from_file_location('memory_job', path)
memory_job = importlib.util.module_from_spec(spec)
spec.loader.exec_module(memory_job)


class MemoryJobArgumentsTests(unittest.TestCase):
    def arguments(self, *extra):
        return memory_job.argument_parser().parse_args([
            '--commit-bytes', '536870912', '--output', 'artifacts/example',
            *extra, '--', 'dotnet', 'Probe.dll'])

    def test_default_retains_suspended_no_window(self):
        options = self.arguments()
        self.assertFalse(options.detached)
        self.assertEqual(0x08000004, memory_job.creation_flags(options.detached))
        self.assertEqual(['--', 'dotnet', 'Probe.dll'], options.command)

    def test_detached_is_explicit_and_remains_suspended(self):
        options = self.arguments('--detached')
        self.assertTrue(options.detached)
        self.assertEqual(0xC, memory_job.creation_flags(options.detached))
        self.assertEqual(0, memory_job.creation_flags(True) & 0x08000000)

    def test_detached_after_command_is_child_argument(self):
        options = memory_job.argument_parser().parse_args([
            '--commit-bytes', '536870912', '--output', 'artifacts/example',
            '--', 'Probe', '--detached'])
        self.assertFalse(options.detached)
        self.assertEqual(['--', 'Probe', '--detached'], options.command)


if __name__ == '__main__':
    unittest.main()
