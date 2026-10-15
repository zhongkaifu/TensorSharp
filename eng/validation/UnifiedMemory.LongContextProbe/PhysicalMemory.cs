// Copyright (c) Zhongkai Fu. Licensed under the BSD-3-Clause license.
using System.Diagnostics;
using System.Globalization;

/// <summary>OS observations separate from allocation credit. RSS/PSS/GC and
/// ledger payload overlap and must not be summed as separate physical owners.</summary>
internal static class PhysicalMemory
{
    internal static Dictionary<string, long>? LinuxStatus()
    {
        if (!OperatingSystem.IsLinux()) return null;
        try { return Kilobytes(File.ReadLines("/proc/self/status")); }
        catch (IOException) { return null; }
        catch (UnauthorizedAccessException) { return null; }
    }

    internal static object Capture(string modelPath)
    {
        using var process = Process.GetCurrentProcess();
        var gc = GC.GetGCMemoryInfo();
        Dictionary<string, long>? status = null, rollup = null, mapping = null;
        string? error = null;
        if (OperatingSystem.IsLinux())
        {
            try
            {
                status = Kilobytes(File.ReadLines("/proc/self/status"));
                rollup = Kilobytes(File.ReadLines("/proc/self/smaps_rollup"));
                mapping = new();
                bool selected = false;
                foreach (string line in File.ReadLines("/proc/self/smaps"))
                {
                    // Map headers have an address range before the first space.
                    int space = line.IndexOf(' ');
                    if (space > 0 && line.AsSpan(0, space).Contains('-'))
                    {
                        var fields = line.Split(' ', 6, StringSplitOptions.RemoveEmptyEntries);
                        selected = fields.Length == 6 && fields[5] == modelPath;
                    }
                    else if (selected && TryKilobytes(line, out var key, out var value))
                        mapping[key] = checked(mapping.GetValueOrDefault(key) + value);
                }
            }
            catch (Exception ex) when (ex is IOException or UnauthorizedAccessException) { error = ex.Message; }
        }
        return new
        {
            Source = OperatingSystem.IsLinux() ? "proc-status/smaps" : "Process/GC",
            process.WorkingSet64, process.PeakWorkingSet64, process.PrivateMemorySize64,
            ManagedLiveBytes = GC.GetTotalMemory(false), GcHeapSizeBytes = gc.HeapSizeBytes,
            GcCommittedBytes = gc.TotalCommittedBytes, StatusBytes = status, RollupBytes = rollup,
            ModelMappingBytes = mapping, Error = error,
            Limitation = "RSS/PSS, mapped pages, managed heap and allocation payload overlap. Measurements are not additive or hard limits; phase samples may miss peaks. Windows mapping residency is unavailable here.",
        };
    }

    private static Dictionary<string, long> Kilobytes(IEnumerable<string> lines)
    {
        var values = new Dictionary<string, long>();
        foreach (string line in lines)
            if (TryKilobytes(line, out var key, out var value)) values[key] = value;
        return values;
    }
    private static bool TryKilobytes(string line, out string key, out long bytes)
    {
        key = ""; bytes = 0;
        var fields = line.Split((char[]?)null, StringSplitOptions.RemoveEmptyEntries);
        if (fields.Length != 3 || !fields[0].EndsWith(':') || fields[2] != "kB"
            || !long.TryParse(fields[1], NumberStyles.None, CultureInfo.InvariantCulture, out long kb)) return false;
        key = fields[0][..^1]; bytes = checked(kb * 1024); return true;
    }
}
