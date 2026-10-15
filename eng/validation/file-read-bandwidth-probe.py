#!/usr/bin/env python3
"""Compare bounded, spread read windows with Windows unbuffered synchronous I/O.

Example (run only when other model/benchmark activity has stopped)::

    python eng/validation/file-read-bandwidth-probe.py C:/models/example.gguf \
        E:/models/example.gguf --samples 8 --window-mib 32 --block-mib 4 \
        --output artifacts/storage-read-probe/report.json

The default reads exactly 256 MiB per file in eight widely spaced windows.
--samples 16 reads 512 MiB per file. The same normalized positions and shuffled
issue order apply to every file. No payload hash, mmap, writes, cache flush, or
cached retry is performed. FILE_FLAG_NO_BUFFERING bypasses Windows file-data
caching; it does NOT establish an absolutely cold physical device/controller.
This Windows-only diagnostic does not constrain TensorSharp runtime platforms.

Primary API references:
https://learn.microsoft.com/en-us/windows/win32/fileio/file-buffering
https://learn.microsoft.com/en-us/windows/win32/api/winbase/ns-winbase-file_storage_info
https://learn.microsoft.com/en-us/windows/win32/api/fileapi/nf-fileapi-setfilepointerex
https://learn.microsoft.com/en-us/windows/win32/api/fileapi/nf-fileapi-readfile
"""

import argparse
import ctypes
import datetime
import json
import math
import os
from pathlib import Path
import platform
import random
import statistics
import time


MIB = 1024 * 1024
DWORD = ctypes.c_uint32
BOOL = ctypes.c_int32
HANDLE = ctypes.c_void_p
INVALID_HANDLE_VALUE = ctypes.c_void_p(-1).value
GENERIC_READ = 0x80000000
FILE_SHARE_ALL = 0x00000001 | 0x00000002 | 0x00000004
OPEN_EXISTING = 3
FILE_FLAG_NO_BUFFERING = 0x20000000
FILE_ATTRIBUTE_NORMAL = 0x00000080
FILE_BEGIN = 0
FILE_STORAGE_INFO_CLASS = 16
MEM_COMMIT_RESERVE = 0x1000 | 0x2000
MEM_RELEASE = 0x8000
PAGE_READWRITE = 0x04


class FileTime(ctypes.Structure):
    _fields_ = [("low", DWORD), ("high", DWORD)]


class FileInformation(ctypes.Structure):
    _fields_ = [
        ("attributes", DWORD), ("created", FileTime), ("accessed", FileTime),
        ("written", FileTime), ("volume_serial", DWORD),
        ("size_high", DWORD), ("size_low", DWORD), ("links", DWORD),
        ("index_high", DWORD), ("index_low", DWORD),
    ]


class FileStorageInfo(ctypes.Structure):
    _fields_ = [(name, DWORD) for name in (
        "logical_bytes_per_sector", "physical_bytes_per_sector_for_atomicity",
        "physical_bytes_per_sector_for_performance",
        "filesystem_effective_physical_bytes_per_sector_for_atomicity", "flags",
        "byte_offset_for_sector_alignment", "byte_offset_for_partition_alignment",
    )]


class ProbeError(Exception):
    def __init__(self, operation, message, *, win32_error=None, **details):
        super().__init__(message)
        self.record = dict(operation=operation, message=message,
                           win32_error=win32_error, **details)


def align_down(value, alignment):
    return value // alignment * alignment


def alignment_from_storage(storage):
    logical = storage["logical_bytes_per_sector"]
    physical = [storage[name] for name in (
        "physical_bytes_per_sector_for_atomicity",
        "physical_bytes_per_sector_for_performance",
        "filesystem_effective_physical_bytes_per_sector_for_atomicity",
    ) if storage[name] not in (0, 0xFFFFFFFF)]
    sizes = [logical] + physical
    if not physical or any(size < logical or size > MIB or size <= 0
                           or size & (size - 1) for size in sizes):
        raise ProbeError("FileStorageInfo", "Unknown or invalid sector alignment",
                         storage=storage)
    return math.lcm(*sizes)


def sample_schedule(file_bytes, window_bytes, samples, alignment, seed):
    """Disjoint windows, both endpoints, same normalized issue order per file."""
    if not 2 <= samples <= 64 or alignment <= 0 or window_bytes <= 0:
        raise ValueError("Need 2..64 samples and positive window/alignment")
    if window_bytes % alignment or file_bytes < samples * window_bytes:
        raise ValueError("File too small for disjoint windows or window is unaligned")
    last = align_down(file_bytes - window_bytes, alignment)
    offsets = [align_down(last * index // (samples - 1), alignment)
               for index in range(samples)]
    if any(right - left < window_bytes for left, right in zip(offsets, offsets[1:])):
        raise ValueError("Aligned sample windows overlap")
    order = list(range(samples))
    random.Random(seed).shuffle(order)
    return [dict(sample_index=index, offset=offsets[index],
                 normalized_position_numerator=index,
                 normalized_position_denominator=samples - 1) for index in order]


class WindowsUnbufferedReader:
    """No file-data fallback. Kernel injection is for model-free API tests only."""

    def __init__(self, path, *, kernel=None, last_error=None, format_error=None):
        if kernel is None:
            if os.name != "nt":
                raise ProbeError("platform", "Windows unbuffered reader unsupported")
            kernel = ctypes.WinDLL("kernel32", use_last_error=True)
        self.kernel = kernel
        self.last_error = last_error or ctypes.get_last_error
        self.format_error = format_error or ctypes.FormatError
        self.handle = None
        self.allocation = None
        self.buffer = None
        self.allocation_bytes = 0
        self.buffer_bytes = 0
        self.path = str(path)
        self.storage = None
        self.alignment = None
        self._bind()

    def _bind(self):
        signatures = {
            "CreateFileW": ([ctypes.c_wchar_p, DWORD, DWORD, ctypes.c_void_p,
                             DWORD, DWORD, HANDLE], HANDLE),
            "GetFileInformationByHandle": ([HANDLE, ctypes.POINTER(FileInformation)], BOOL),
            "GetFileInformationByHandleEx": ([HANDLE, ctypes.c_int32,
                                              ctypes.c_void_p, DWORD], BOOL),
            "GetFileSizeEx": ([HANDLE, ctypes.POINTER(ctypes.c_int64)], BOOL),
            "SetFilePointerEx": ([HANDLE, ctypes.c_int64,
                                  ctypes.POINTER(ctypes.c_int64), DWORD], BOOL),
            "ReadFile": ([HANDLE, ctypes.c_void_p, DWORD,
                          ctypes.POINTER(DWORD), ctypes.c_void_p], BOOL),
            "CloseHandle": ([HANDLE], BOOL),
            "VirtualAlloc": ([ctypes.c_void_p, ctypes.c_size_t, DWORD, DWORD], ctypes.c_void_p),
            "VirtualFree": ([ctypes.c_void_p, ctypes.c_size_t, DWORD], BOOL),
        }
        for name, (arguments, result) in signatures.items():
            call = getattr(self.kernel, name)
            call.argtypes, call.restype = arguments, result

    def _error(self, operation, **details):
        code = int(self.last_error())
        return ProbeError(operation, self.format_error(code).strip(),
                          win32_error=code, **details)

    def open(self):
        handle = self.kernel.CreateFileW(
            self.path, GENERIC_READ, FILE_SHARE_ALL, None, OPEN_EXISTING,
            FILE_ATTRIBUTE_NORMAL | FILE_FLAG_NO_BUFFERING, None)
        if handle is None or handle == INVALID_HANDLE_VALUE:
            raise self._error("CreateFileW", path=self.path,
                              flags=FILE_ATTRIBUTE_NORMAL | FILE_FLAG_NO_BUFFERING)
        self.handle = handle
        info = FileStorageInfo()
        if not self.kernel.GetFileInformationByHandleEx(
                handle, FILE_STORAGE_INFO_CLASS, ctypes.byref(info), ctypes.sizeof(info)):
            raise self._error("GetFileInformationByHandleEx(FileStorageInfo)", path=self.path)
        self.storage = {name: int(getattr(info, name)) for name, _ in info._fields_}
        self.alignment = alignment_from_storage(self.storage)
        return self.identity()

    def identity(self):
        info = FileInformation()
        if not self.kernel.GetFileInformationByHandle(self.handle, ctypes.byref(info)):
            raise self._error("GetFileInformationByHandle", path=self.path)
        size = ctypes.c_int64()
        if not self.kernel.GetFileSizeEx(self.handle, ctypes.byref(size)):
            raise self._error("GetFileSizeEx", path=self.path)
        metadata_size = (info.size_high << 32) | info.size_low
        if size.value != metadata_size or size.value <= 0:
            raise ProbeError("file_identity", "Inconsistent/empty source size",
                             size_ex=size.value, metadata_size=metadata_size)
        return dict(volume_serial=int(info.volume_serial),
                    file_index=(info.index_high << 32) | info.index_low,
                    last_write_filetime=(info.written.high << 32) | info.written.low,
                    file_bytes=size.value)

    def allocate(self, block_bytes):
        if block_bytes <= 0 or block_bytes > 128 * MIB or block_bytes % self.alignment:
            raise ProbeError("buffer_alignment", "Invalid/unbounded read-block size",
                             block_bytes=block_bytes, alignment=self.alignment)
        if self.allocation is not None:
            raise ProbeError("VirtualAlloc", "Reader buffer already allocated")
        self.allocation_bytes = block_bytes + self.alignment - 1
        allocation = self.kernel.VirtualAlloc(None, self.allocation_bytes,
                                              MEM_COMMIT_RESERVE, PAGE_READWRITE)
        if not allocation:
            raise self._error("VirtualAlloc", allocation_bytes=self.allocation_bytes)
        self.allocation = allocation
        self.buffer = (allocation + self.alignment - 1) // self.alignment * self.alignment
        self.buffer_bytes = block_bytes
        # Pay first-touch/zero-page costs before read timings, not in the first ReadFile.
        ctypes.memset(self.buffer, 0, block_bytes)

    def read_at(self, offset, requested_bytes):
        row = dict(requested_offset=offset, actual_offset=None,
                   requested_bytes=requested_bytes, read_bytes=0,
                   seek_ns=None, read_ns=None, error=None)
        if (self.buffer is None or offset < 0 or offset % self.alignment
                or requested_bytes <= 0 or requested_bytes > self.buffer_bytes
                or requested_bytes % self.alignment):
            row["error"] = dict(operation="read_alignment", message="Unaligned/out-of-bounds read")
            return row
        actual = ctypes.c_int64()
        started = time.perf_counter_ns()
        success = self.kernel.SetFilePointerEx(self.handle, offset, ctypes.byref(actual), FILE_BEGIN)
        row["seek_ns"] = time.perf_counter_ns() - started
        if not success:
            row["error"] = self._error("SetFilePointerEx", offset=offset).record
            return row
        row["actual_offset"] = actual.value
        if actual.value != offset:
            row["error"] = dict(operation="SetFilePointerEx", message="Returned offset differs from requested")
            return row
        count = DWORD()
        started = time.perf_counter_ns()
        success = self.kernel.ReadFile(self.handle, self.buffer, requested_bytes,
                                       ctypes.byref(count), None)
        row["read_ns"] = time.perf_counter_ns() - started
        row["read_bytes"] = int(count.value)
        if not success:
            row["error"] = self._error("ReadFile", offset=offset,
                                       requested_bytes=requested_bytes,
                                       read_bytes=count.value).record
        elif count.value != requested_bytes:
            row["error"] = dict(operation="ReadFileIncomplete", win32_error=0,
                                 message="Successful ReadFile returned a short/invalid byte count")
        return row

    def close(self):
        errors = []
        if self.allocation is not None:
            if not self.kernel.VirtualFree(self.allocation, 0, MEM_RELEASE):
                errors.append(self._error("VirtualFree").record)
            self.allocation = self.buffer = None
        if self.handle is not None:
            if not self.kernel.CloseHandle(self.handle):
                errors.append(self._error("CloseHandle").record)
            self.handle = None
        return errors


def summarize_reads(rows):
    complete = [row for row in rows if row["error"] is None]
    if not complete:
        return dict(complete_reads=0, read_bytes=0, partial=True)
    byte_count = sum(row["read_bytes"] for row in complete)
    read_ns = sum(row["read_ns"] for row in complete)
    seek_read_ns = sum(row["read_ns"] + row["seek_ns"] for row in complete)
    times = sorted(row["read_ns"] / 1e6 for row in complete)
    return dict(complete_reads=len(complete), read_bytes=byte_count,
                partial=len(complete) != len(rows), total_read_ns=read_ns,
                total_seek_read_ns=seek_read_ns,
                read_call_mib_per_s=byte_count * 1e9 / read_ns / MIB if read_ns else None,
                seek_read_mib_per_s=byte_count * 1e9 / seek_read_ns / MIB if seek_read_ns else None,
                read_latency_ms=dict(min=times[0], median=statistics.median(times),
                                     p95=times[math.ceil(len(times) * 0.95) - 1], max=times[-1]))


def probe_file(path, *, samples, window_bytes, block_bytes, seed, reader_factory=WindowsUnbufferedReader):
    result = dict(path=str(path), passed=False, status="failed", reads=[], cleanup_errors=[],
                  open_access=GENERIC_READ, open_flags=FILE_ATTRIBUTE_NORMAL | FILE_FLAG_NO_BUFFERING,
                  share_mode=FILE_SHARE_ALL, creation_disposition=OPEN_EXISTING)
    reader = None
    started = time.perf_counter_ns()
    try:
        reader = reader_factory(path)
        before = reader.open()
        result.update(identity_before=before, storage=reader.storage, alignment=reader.alignment)
        if window_bytes % block_bytes:
            raise ValueError("Window must be an exact multiple of the block size")
        schedule = sample_schedule(before["file_bytes"], window_bytes, samples, reader.alignment, seed)
        result["schedule"] = schedule
        reader.allocate(block_bytes)
        result["buffer"] = dict(address=reader.buffer, bytes=reader.buffer_bytes,
                                allocation_bytes=reader.allocation_bytes,
                                alignment_verified=reader.buffer % reader.alignment == 0,
                                prefaulted_before_read_timings=True)
        if not result["buffer"]["alignment_verified"]:
            raise ProbeError("buffer_alignment", "Actual allocation is not aligned")
        for issue_order, sample in enumerate(schedule):
            for delta in range(0, window_bytes, block_bytes):
                row = reader.read_at(sample["offset"] + delta, block_bytes)
                row.update(sample_index=sample["sample_index"], issue_order=issue_order,
                           window_relative_offset=delta)
                result["reads"].append(row)
                if row["error"] is not None:
                    raise ProbeError("read_failed", "Unbuffered read failed; no cached retry",
                                     failed_read_index=len(result["reads"]) - 1)
        after = reader.identity()
        result["identity_after"] = after
        if before != after:
            raise ProbeError("file_identity", "Source identity/size/mtime changed during probe")
        if sum(row["read_bytes"] for row in result["reads"]) != samples * window_bytes:
            raise ProbeError("read_total", "Actual read-byte total differs from the plan")
        result.update(passed=True, status="passed")
    except Exception as error:
        result["error"] = (error.record if isinstance(error, ProbeError)
                           else dict(operation="probe", message=repr(error)))
    finally:
        if reader is not None:
            result["cleanup_errors"] = reader.close()
            if result["cleanup_errors"]:
                result.update(passed=False, status="failed")
        result["elapsed_ns"] = time.perf_counter_ns() - started
    result["summary"] = summarize_reads(result["reads"])
    result["summary"]["partial"] = not result["passed"]
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("paths", nargs="+", help="Existing regular model/data files; read only")
    parser.add_argument("--samples", type=int, default=8)
    parser.add_argument("--window-mib", type=int, default=32)
    parser.add_argument("--block-mib", type=int, default=4)
    parser.add_argument("--seed", type=int, default=1729)
    parser.add_argument("--max-read-mib", type=int, default=512, help="Hard per-file read cap, at most 1024 MiB")
    parser.add_argument("--output", type=Path, help="Fresh JSON report path; existing output is refused")
    parser.add_argument("--plan", action="store_true", help="Show byte budget/paths without opening a file or loading Win32 APIs")
    args = parser.parse_args(argv)
    if (not 2 <= args.samples <= 64 or not 1 <= args.block_mib <= args.window_mib <= 128
            or args.window_mib % args.block_mib or not 1 <= args.max_read_mib <= 1024
            or args.samples * args.window_mib > args.max_read_mib):
        parser.error("Invalid geometry or read budget (2..64 samples, block divides window<=128 MiB, cap<=1024 MiB)")
    report = dict(schema_version=1, passed=False, status="planned",
                  utc_start=datetime.datetime.now(datetime.timezone.utc).isoformat(),
                  platform=platform.platform(), process_id=os.getpid(),
                  cache_mode="Windows CreateFileW FILE_FLAG_NO_BUFFERING; synchronous ReadFile",
                  physical_cold_state_proven=False, cached_fallback=False, writes_to_sources=False,
                  policy=dict(samples=args.samples, window_bytes=args.window_mib * MIB,
                              block_bytes=args.block_mib * MIB, seed=args.seed, queue_depth=1,
                              schedule="evenly_spread_including_endpoints_then_seeded_shuffle",
                              planned_read_bytes_per_file=args.samples * args.window_mib * MIB,
                              max_read_bytes_per_file=args.max_read_mib * MIB),
                  requested_paths=args.paths, files=[],
                  limitations=["Bypasses Windows file-data cache; hardware/controller/virtualization caches are not flushed or proven cold.",
                               "Spread sampling is not the application's mixed-size expert-read workload or an inference benchmark.",
                               "One sequential reader per file; no overlapped I/O, GPU transfers or runtime-platform requirement."])
    if args.plan:
        print(json.dumps(report, indent=2))
        return 0
    if args.output is None:
        parser.error("--output is required for an actual run")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    # Refuse an occupied output before opening any source file.
    with args.output.open("x", encoding="utf-8") as stream:
        if os.name != "nt":
            report.update(status="unsupported", error=dict(operation="platform", message="Windows NO_BUFFERING probe is unsupported on this platform; no fallback attempted"))
        else:
            for path in args.paths:
                report["files"].append(probe_file(
                    path, samples=args.samples, window_bytes=args.window_mib * MIB,
                    block_bytes=args.block_mib * MIB, seed=args.seed))
            report["passed"] = all(item["passed"] for item in report["files"])
            report["status"] = "passed" if report["passed"] else "failed"
        report["utc_end"] = datetime.datetime.now(datetime.timezone.utc).isoformat()
        json.dump(report, stream, indent=2)
        stream.write("\n")
    print(json.dumps(dict(passed=report["passed"], status=report["status"],
                          output=str(args.output), files=[dict(path=x["path"], passed=x["passed"],
                          summary=x["summary"]) for x in report["files"]])))
    return 0 if report["passed"] else 2 if report["status"] == "unsupported" else 1


if __name__ == "__main__":
    raise SystemExit(main())
