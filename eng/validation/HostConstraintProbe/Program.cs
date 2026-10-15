// Copyright (c) Zhongkai Fu. Licensed under the BSD-3-Clause license.
using System.Diagnostics;
using System.Reflection;
using System.Runtime.InteropServices;
using System.Security.Cryptography;
using System.Text.Json;
using TensorSharp.Memory;

if (args.SequenceEqual(new[] { "--self-test" })) return SelfTests.Run();
ProbeOptions options;
try { options = ProbeOptions.Parse(args); }
catch (ArgumentException error) { Console.Error.WriteLine(error.Message); return 2; }
if (File.Exists(options.Output) || Directory.Exists(options.Output))
    throw new IOException("Evidence already exists; choose a fresh output JSON path.");
Directory.CreateDirectory(Path.GetDirectoryName(options.Output)!);
// CreateNew makes accidental concurrent evidence overwrite fail before allocation.
using var evidence = new FileStream(options.Output, FileMode.CreateNew, FileAccess.Write, FileShare.Read);
var errors = new List<string>();
var samples = new List<object>();
bool observationsPassed = true, allocated = false, allocationFreed = false;
long pageStrideWrites = 0;
nint allocation = 0;
void Capture(string phase)
{
    var before = DateTimeOffset.UtcNow;
    var snapshot = HostMemoryInfo.CaptureAllocationConstraints();
    var after = DateTimeOffset.UtcNow;
    var check = ObservationCheck.Evaluate(snapshot, OperatingSystem.IsWindows(), options.ExpectedCommitCeiling);
    observationsPassed &= check.Passed;
    using var process = Process.GetCurrentProcess();
    process.Refresh();
    samples.Add(new { phase, sampling_started_utc = before, sampling_finished_utc = after,
        physical_total_bytes = snapshot.Total, physical_available_bytes = snapshot.Available,
        process_commit_limit_bytes = snapshot.CommitLimit, process_commit_available_bytes = snapshot.CommitAvailable,
        commit_constraint_scope = snapshot.CommitConstraintScope,
        process_private_bytes_counter = process.PrivateMemorySize64, process_working_set_bytes_counter = process.WorkingSet64,
        requested_commit_ceiling_observed = check.RequestedCommitCeilingObserved, commit_unknown = check.CommitUnknown,
        passed = check.Passed, failures = check.Failures });
}
try
{
    Capture("before-allocation");
    if (options.AllocationBytes > 0)
    {
        unsafe
        {
            allocation = (nint)NativeMemory.Alloc((nuint)options.AllocationBytes);
            if (allocation == 0) throw new OutOfMemoryException("The bounded probe allocation failed.");
            allocated = true;
            int stride = Environment.SystemPageSize;
            if (stride <= 0) throw new InvalidOperationException("Invalid operating-system page size.");
            byte* data = (byte*)allocation;
            for (long offset = 0; offset < options.AllocationBytes; offset += stride)
            { data[offset] = 0x5a; pageStrideWrites++; }
            data[options.AllocationBytes - 1] = 0x5a;
        }
        Capture("while-allocation-is-live");
    }
}
catch (Exception error) { errors.Add(error.ToString()); }
finally
{
    if (allocation != 0)
    {
        unsafe { NativeMemory.Free((void*)allocation); }
        allocation = 0;
        allocationFreed = true;
    }
    try { Capture("after-free"); }
    catch (Exception error) { errors.Add("After-free observation: " + error); }
}
bool passed = errors.Count == 0 && observationsPassed
    && samples.Count == (options.AllocationBytes == 0 ? 2 : 3)
    && (options.AllocationBytes == 0 || allocated && allocationFreed);
string Hash(string path) { using var input = File.OpenRead(path); return Convert.ToHexString(SHA256.HashData(input)); }
var identities = new[] { Assembly.GetExecutingAssembly().Location, typeof(HostMemoryInfo).Assembly.Location }
    .Select(path => new { path, sha256 = Hash(path) }).ToArray();
var report = new
{
    passed, os = RuntimeInformation.OSDescription, process_id = Environment.ProcessId,
    is_64_bit = Environment.Is64BitProcess, requested_allocation_bytes = options.AllocationBytes,
    requested_commit_ceiling_bytes = options.ExpectedCommitCeiling, allocated, allocation_freed = allocationFreed,
    page_stride_writes = pageStrideWrites, final_live_payload_bytes = allocation == 0 ? 0 : options.AllocationBytes,
    samples, errors, identities,
    operating_system_enforcement_verified_by_this_probe = false,
    scope = "CPU API-to-constraint observation only. A requested ceiling is a caller-supplied expectation, not proof of an installed limit. Pair with the separately verified Windows Job launcher report. No model, GPU, RSS/file-cache hard cap or cgroup/Apple equivalence is established.",
    allocation_scope = "At most 64 MiB NativeMemory payload, one byte per system page and the final byte touched; finally NativeMemory.Free. Allocator retention and unrelated memory activity can prevent immediate OS-counter recovery; no exact before/after delta is required.",
    commit_scope = "Windows API allowance intersected with observable immediate-Job constraints; inspect each sample's constraint scope, and do not infer hidden-ancestor coverage. Other current providers explicitly report commit unknown. Commit is not added to physical RAM. Linux private/working-set counters are observations, not a Windows commit equivalent."
};
JsonSerializer.Serialize(evidence, report, new JsonSerializerOptions { WriteIndented = true });
evidence.Flush();
Console.WriteLine($"HostConstraintProbe passed={passed}; samples={samples.Count}; allocation_freed={allocationFreed}; report={options.Output}");
return passed ? 0 : 1;
