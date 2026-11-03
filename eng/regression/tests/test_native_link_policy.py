"""Evaluate owned test-link policy in CMake script mode, without native builds."""
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest

REPO = Path(__file__).resolve().parents[3]
CMAKE_FILE = REPO / "TensorSharp.GGML.Native/CMakeLists.txt"


class NativeLinkPolicyTests(unittest.TestCase):
    def test_vision_fixture_uses_gguf_writer_not_exported_by_wrapper(self):
        source = (REPO / "TensorSharp.GGML.Native/tests/dsv41_vision_clip_test.cpp").read_text()
        exports = set((REPO / "TensorSharp.GGML.Native/ggml_api_exports.def").read_text().split())
        self.assertIn("gguf_write_to_file(", source)
        self.assertNotIn("gguf_write_to_file", exports)
        self.assertIn("ggml_init", exports)
        self.assertIn('TSGgml_Dsv41VisionLoad(path.c_str(), "CPU"', source)

    @unittest.skipUnless(shutil.which("cmake"), "CMake link-policy contract requires local cmake; no configure/compiler step")
    def test_windows_static_fixture_resolves_before_wrapper_nonwindows_order_unchanged(self):
        source = CMAKE_FILE.read_text()
        start = source.index("    set(TSG_DSV41_VISION_CLIP_LIBRARIES GgmlOps ggml-base)")
        tail = "    target_link_libraries(GgmlOpsDsv41VisionClipTest PRIVATE ${TSG_DSV41_VISION_CLIP_LIBRARIES})"
        stop = source.index(tail, start) + len(tail)
        policy = source[start:stop]
        with tempfile.TemporaryDirectory() as directory:
            script = Path(directory) / "link-policy.cmake"
            for windows, expected in (("TRUE", "PRIVATE;ggml-base;GgmlOps"),
                                      ("FALSE", "PRIVATE;GgmlOps;ggml-base")):
                with self.subTest(windows=windows):
                    script.write_text('cmake_minimum_required(VERSION 3.20)\n'
                        'function(target_link_libraries target)\n'
                        '  if(NOT target STREQUAL "GgmlOpsDsv41VisionClipTest")\n'
                        '    message(FATAL_ERROR "unexpected target: ${target}")\n'
                        '  endif()\n'
                        '  set(actual "${ARGN}" PARENT_SCOPE)\n'
                        'endfunction()\n'
                        f'set(WIN32 {windows})\n{policy}\n'
                        f'if(NOT actual STREQUAL "{expected}")\n'
                        '  message(FATAL_ERROR "wrong link order: ${actual}")\n'
                        'endif()\n')
                    result = subprocess.run([shutil.which("cmake"), "-P", str(script)],
                        text=True, capture_output=True, timeout=10, check=False)
                    self.assertEqual(0, result.returncode, result.stdout + result.stderr)


if __name__ == "__main__":
    unittest.main()
