"""Model-free tests: injected Win32 calls, no files/devices/benchmarks opened."""

import contextlib
import ctypes
import importlib.util
import io
import json
from pathlib import Path
import unittest


spec = importlib.util.spec_from_file_location(
    "file_read_bandwidth_probe", Path(__file__).resolve().parents[1] / "file-read-bandwidth-probe.py")
probe = importlib.util.module_from_spec(spec)
spec.loader.exec_module(probe)


class Call:
    def __init__(self, callback):
        self.callback = callback
        self.calls = []

    def __call__(self, *args):
        self.calls.append(args)
        return self.callback(*args)


class FakeKernel:
    def __init__(self, *, storage_failure=False, alloc_failure=False, read_failure=False,
                 short_read=False, wrong_seek=False, mutate=False, free_failure=False):
        self.storage_failure = storage_failure
        self.alloc_failure = alloc_failure
        self.read_failure = read_failure
        self.short_read = short_read
        self.wrong_seek = wrong_seek
        self.mutate = mutate
        self.free_failure = free_failure
        self.size = 256 * 1024
        self.buffers = []
        self.metadata_queries = 0
        self.CreateFileW = Call(lambda *args: 123)
        self.GetFileInformationByHandle = Call(self.metadata)
        self.GetFileInformationByHandleEx = Call(self.storage)
        self.GetFileSizeEx = Call(self.get_size)
        self.SetFilePointerEx = Call(self.seek)
        self.ReadFile = Call(self.read)
        self.CloseHandle = Call(lambda *args: 1)
        self.VirtualAlloc = Call(self.allocate)
        self.VirtualFree = Call(lambda *args: 0 if self.free_failure else 1)

    def metadata(self, handle, pointer):
        self.metadata_queries += 1
        info = ctypes.cast(pointer, ctypes.POINTER(probe.FileInformation)).contents
        info.volume_serial, info.index_low = 12, 34
        info.size_high, info.size_low = self.size >> 32, self.size & 0xFFFFFFFF
        info.written.low = self.metadata_queries if self.mutate else 55
        return 1

    def storage(self, handle, info_class, pointer, byte_count):
        if self.storage_failure:
            return 0
        assert info_class == 16 and byte_count == 28
        info = ctypes.cast(pointer, ctypes.POINTER(probe.FileStorageInfo)).contents
        info.logical_bytes_per_sector = 512
        info.physical_bytes_per_sector_for_atomicity = 4096
        info.physical_bytes_per_sector_for_performance = 4096
        info.filesystem_effective_physical_bytes_per_sector_for_atomicity = 4096
        return 1

    def get_size(self, handle, pointer):
        ctypes.cast(pointer, ctypes.POINTER(ctypes.c_int64))[0] = self.size
        return 1

    def seek(self, handle, offset, pointer, method):
        ctypes.cast(pointer, ctypes.POINTER(ctypes.c_int64))[0] = offset + (512 if self.wrong_seek else 0)
        return 1

    def read(self, handle, buffer, byte_count, pointer, overlapped):
        ctypes.cast(pointer, ctypes.POINTER(probe.DWORD))[0] = byte_count - 512 if self.short_read else (0 if self.read_failure else byte_count)
        return 0 if self.read_failure else 1

    def allocate(self, pointer, byte_count, flags, protection):
        if self.alloc_failure:
            return 0
        buffer = ctypes.create_string_buffer(byte_count)
        self.buffers.append(buffer)
        return ctypes.addressof(buffer)

    def reader(self, path):
        return probe.WindowsUnbufferedReader(
            path, kernel=self, last_error=lambda: 5, format_error=lambda code: "mock error " + str(code))


class ReadBandwidthProbeTests(unittest.TestCase):
    def run_mock(self, kernel, **kwargs):
        return probe.probe_file("not-opened-model.gguf", samples=8, window_bytes=8192,
                                block_bytes=4096, seed=1729, reader_factory=kernel.reader, **kwargs)

    def test_schedule_covers_endpoints_without_overlap_or_alignment_loss(self):
        schedule = probe.sample_schedule(10**12 + 123, 32 * probe.MIB, 16, 4096, 1729)
        by_index = sorted(schedule, key=lambda item: item["sample_index"])
        self.assertEqual(0, by_index[0]["offset"])
        self.assertEqual(probe.align_down(10**12 + 123 - 32 * probe.MIB, 4096), by_index[-1]["offset"])
        self.assertTrue(all(item["offset"] % 4096 == 0 for item in by_index))
        self.assertTrue(all(b["offset"] - a["offset"] >= 32 * probe.MIB for a, b in zip(by_index, by_index[1:])))
        other = probe.sample_schedule(20**12, 32 * probe.MIB, 16, 512, 1729)
        self.assertEqual([x["sample_index"] for x in schedule], [x["sample_index"] for x in other])
        self.assertNotEqual(list(range(16)), [x["sample_index"] for x in schedule])

    def test_invalid_sampling_refuses_instead_of_repeated_cached_region(self):
        for parameters in [(4096, 4096, 8, 4096), (65536, 4095, 8, 4096), (65536, 4096, 1, 4096)]:
            with self.subTest(parameters=parameters), self.assertRaises(ValueError):
                probe.sample_schedule(*parameters, seed=1)

    def test_sector_information_is_required_not_guessed(self):
        base = dict(logical_bytes_per_sector=512, physical_bytes_per_sector_for_atomicity=4096,
                    physical_bytes_per_sector_for_performance=4096,
                    filesystem_effective_physical_bytes_per_sector_for_atomicity=4096)
        self.assertEqual(4096, probe.alignment_from_storage(base))
        for physical in (0, 0xFFFFFFFF, 1234, 2 * probe.MIB):
            invalid = dict(base, **{name: physical for name in base if name != "logical_bytes_per_sector"})
            with self.subTest(physical=physical), self.assertRaises(probe.ProbeError):
                probe.alignment_from_storage(invalid)

    def test_success_uses_read_only_no_buffering_exact_counts_and_cleanup(self):
        kernel = FakeKernel()
        result = self.run_mock(kernel)
        self.assertTrue(result["passed"], result)
        self.assertEqual(16, len(result["reads"]))
        self.assertEqual(65536, result["summary"]["read_bytes"])
        self.assertEqual(0x80000000, kernel.CreateFileW.calls[0][1])
        self.assertEqual(3, kernel.CreateFileW.calls[0][4])
        self.assertEqual(0x20000080, kernel.CreateFileW.calls[0][5])
        self.assertEqual(0, result["buffer"]["address"] % 4096)
        self.assertTrue(result["buffer"]["prefaulted_before_read_timings"])
        self.assertEqual(result["identity_before"], result["identity_after"])
        self.assertTrue(all(row["actual_offset"] == row["requested_offset"] and row["read_bytes"] == 4096 for row in result["reads"]))
        self.assertEqual(1, len(kernel.VirtualFree.calls))
        self.assertEqual(1, len(kernel.CloseHandle.calls))

    def test_read_error_keeps_raw_error_and_never_retries_cached(self):
        kernel = FakeKernel(read_failure=True)
        result = self.run_mock(kernel)
        self.assertFalse(result["passed"])
        self.assertEqual(1, len(kernel.CreateFileW.calls))
        self.assertEqual(1, len(kernel.ReadFile.calls))
        self.assertEqual(5, result["reads"][0]["error"]["win32_error"])
        self.assertEqual("ReadFile", result["reads"][0]["error"]["operation"])
        self.assertEqual(0, result["reads"][0]["read_bytes"])
        self.assertTrue(result["summary"]["partial"])
        self.assertEqual(1, len(kernel.CloseHandle.calls))

    def test_short_success_is_not_a_throughput_pass(self):
        kernel = FakeKernel(short_read=True)
        result = self.run_mock(kernel)
        self.assertFalse(result["passed"])
        self.assertEqual(3584, result["reads"][0]["read_bytes"])
        self.assertEqual(0, result["reads"][0]["error"]["win32_error"])
        self.assertEqual("ReadFileIncomplete", result["reads"][0]["error"]["operation"])
        self.assertEqual(1, len(kernel.ReadFile.calls))

    def test_wrong_seek_cannot_read_a_different_offset(self):
        kernel = FakeKernel(wrong_seek=True)
        result = self.run_mock(kernel)
        self.assertFalse(result["passed"])
        self.assertEqual([], kernel.ReadFile.calls)
        row = result["reads"][0]
        self.assertEqual(row["requested_offset"] + 512, row["actual_offset"])

    def test_storage_api_failure_is_explicit_and_releases_handle(self):
        kernel = FakeKernel(storage_failure=True)
        result = self.run_mock(kernel)
        self.assertFalse(result["passed"])
        self.assertEqual(5, result["error"]["win32_error"])
        self.assertEqual("GetFileInformationByHandleEx(FileStorageInfo)", result["error"]["operation"])
        self.assertEqual([], kernel.VirtualAlloc.calls)
        self.assertEqual([], kernel.ReadFile.calls)
        self.assertEqual(1, len(kernel.CloseHandle.calls))

    def test_allocation_failure_cannot_trigger_a_cached_reader(self):
        kernel = FakeKernel(alloc_failure=True)
        result = self.run_mock(kernel)
        self.assertFalse(result["passed"])
        self.assertEqual("VirtualAlloc", result["error"]["operation"])
        self.assertEqual(5, result["error"]["win32_error"])
        self.assertEqual([], kernel.ReadFile.calls)
        self.assertEqual(1, len(kernel.CloseHandle.calls))

    def test_cleanup_error_fails_an_otherwise_complete_probe(self):
        kernel = FakeKernel(free_failure=True)
        result = self.run_mock(kernel)
        self.assertFalse(result["passed"])
        self.assertEqual("VirtualFree", result["cleanup_errors"][0]["operation"])
        self.assertEqual(1, len(kernel.CloseHandle.calls))

    def test_source_change_is_not_compared_as_a_stable_file(self):
        result = self.run_mock(FakeKernel(mutate=True))
        self.assertFalse(result["passed"])
        self.assertEqual("file_identity", result["error"]["operation"])

    def test_plan_does_not_open_any_file_or_native_api(self):
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            self.assertEqual(0, probe.main(["nonexistent:C-file", "nonexistent:E-file", "--plan"]))
        result = json.loads(output.getvalue())
        self.assertEqual("planned", result["status"])
        self.assertEqual([], result["files"])
        self.assertEqual(256 * probe.MIB, result["policy"]["planned_read_bytes_per_file"])
        self.assertEqual(4 * probe.MIB, result["policy"]["block_bytes"])
        self.assertFalse(result["physical_cold_state_proven"])
        self.assertFalse(result["cached_fallback"])


if __name__ == "__main__":
    unittest.main()
