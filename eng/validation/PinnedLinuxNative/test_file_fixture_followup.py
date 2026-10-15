"""Model-free negative controls for the strict file fixture runner."""
import importlib.util
from pathlib import Path, PurePosixPath
import unittest

spec = importlib.util.spec_from_file_location("file_fixture_followup", Path(__file__).with_name("file_fixture_followup.py"))
runner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)


class FileFixtureFollowupTests(unittest.TestCase):
    def test_two_only_test_deltas(self):
        expected = {name: "old" for name in runner.ALLOWED_DELTAS}
        actual = {name: "new" for name in expected}
        self.assertEqual(set(runner.source_deltas(expected, actual, True)), runner.ALLOWED_DELTAS)

    def test_production_change_refused(self):
        expected = {"TensorSharp.GGML.Native/ggml_ops_host_moe_cache.cpp": "old"}
        with self.assertRaises(RuntimeError):
            runner.source_deltas(expected, {next(iter(expected)): "new"}, False)

    def test_extra_input_refused(self):
        with self.assertRaises(RuntimeError):
            runner.source_deltas({}, {"TensorSharp.GGML.Native/extra.h": "new"}, False)

    def test_half_port_refused(self):
        expected = {name: "old" for name in runner.ALLOWED_DELTAS}
        actual = dict(expected)
        actual[next(iter(expected))] = "new"
        with self.assertRaises(RuntimeError):
            runner.source_deltas(expected, actual, True)

    def test_compile_and_noop_link_wrapper(self):
        raw = "/usr/bin/g++-12 -o a.o -c /old/test.cpp\n/usr/bin/g++-12 -o b.o -c /old/cache.cpp\n: && /usr/bin/g++-12 a.o b.o -o test && :\n"
        rows = runner.parse_commands(raw)
        self.assertEqual(rows[2], ["/usr/bin/g++-12", "a.o", "b.o", "-o", "test"])

    def test_shell_chaining_refused(self):
        raw = "/usr/bin/g++-12 -c a\n/usr/bin/g++-12 -c b\n/usr/bin/g++-12 a.o b.o && touch bad\n"
        with self.assertRaises(RuntimeError):
            runner.parse_commands(raw)

    def test_mapping_changes_only_owned_absolute_paths(self):
        original = ["/usr/bin/g++-12", "-DTSG_GGML_TEST_HOOKS", "-I/old/driver",
                    "-I/frozen/ggml/src", "-o", "CMakeFiles/test/a.o", "-c", "/old/driver/tests/a.cpp",
                    "/frozen/libs/libggml.a"]
        actual = runner.mapped_arguments(original, PurePosixPath("/old/driver"), PurePosixPath("/fresh/source"))
        expected = list(original)
        expected[2], expected[7] = "-I/fresh/source", "/fresh/source/tests/a.cpp"
        self.assertEqual(actual, expected)

    def test_dependency_flags_only_removed_for_database_comparison(self):
        actual = ["cc", "-O3", "-MD", "-MT", "a.o", "-MF", "a.o.d", "-o", "a.o", "-c", "a.cpp"]
        self.assertEqual(runner.without_dependency_flags(actual), ["cc", "-O3", "-o", "a.o", "-c", "a.cpp"])

    def test_skip_never_passes(self):
        with self.assertRaises(RuntimeError):
            runner.validate_output(77, "SKIP: CUDA unavailable")

    def test_unexercised_old_linux_file_switch_never_passes(self):
        with self.assertRaises(RuntimeError):
            runner.validate_output(0, "PASS: CUDA compact expert cache, warm hits, eviction, invalidation, reload and budget")

    def test_both_actual_file_and_microgroup_markers_required(self):
        markers = ("PASS: exact file extents, cached reuse, out-of-order staging, partial-read recovery and source release\n"
                   "PASS: scoped N3 prefill microgroups for 2/3/4/5/8/17/32 tokens, K10, scalar CUDA bit-exact, 3 bridges")
        runner.validate_output(0, markers)
        with self.assertRaises(RuntimeError):
            runner.validate_output(0, markers + "\nFAIL: injected")


if __name__ == "__main__":
    unittest.main()
