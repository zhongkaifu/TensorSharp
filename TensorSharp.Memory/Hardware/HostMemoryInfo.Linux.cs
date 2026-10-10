// Copyright (c) Zhongkai Fu. Licensed under the BSD-3-Clause license.
#nullable enable
using System;
using System.Collections.Generic;
using System.Globalization;
using System.IO;
using System.Linq;

namespace TensorSharp.Memory;

public static partial class HostMemoryInfo
{
    /// <summary>Linux host availability constrained by this process's visible memory
    /// cgroup hierarchy. This is an observation for admission forecasts, not an atomic
    /// reservation. A cgroup namespace can hide further ancestors; their usage cannot
    /// be discovered through the namespace-private filesystem. Visible ancestors are
    /// all checked. Observed clean, unmapped inactive file cache may be reclaimed;
    /// active/mapped/dirty/unevictable pages are not treated as free. Missing optional
    /// cache details give no credit; read/parse failures never mean unlimited capacity.</summary>
    internal static (long Total, long Available) CaptureLinux() => ReadLinux(File.ReadAllText);

    // A pure filesystem seam keeps parsing tests independent of the test host's
    // operating system and does not create or mutate any cgroup.
    internal static (long Total, long Available) ReadLinux(Func<string, string> readText)
    {
        ArgumentNullException.ThrowIfNull(readText);
        long total = -1, available = -1;
        foreach (string line in Lines(Read("/proc/meminfo")))
        {
            string[] parts = line.Split((char[]?)null, StringSplitOptions.RemoveEmptyEntries);
            if (parts.Length == 0 || parts[0] is not ("MemTotal:" or "MemAvailable:")) continue;
            if (parts.Length != 3 || parts[2] != "kB") throw Invalid("/proc/meminfo");
            long value = Number(parts[1], "/proc/meminfo");
            if (value > long.MaxValue / 1024) throw Invalid("/proc/meminfo");
            if (parts[0] == "MemTotal:") total = value * 1024;
            else available = value * 1024;
        }
        if (total <= 0 || available < 0 || available > total) throw Invalid("/proc/meminfo");

        string? unified = null, memory = null;
        foreach (string line in Lines(Read("/proc/self/cgroup")))
        {
            string[] parts = line.Split(':', 3);
            if (parts.Length != 3 || !int.TryParse(parts[0], NumberStyles.None, CultureInfo.InvariantCulture, out _))
                throw Invalid("/proc/self/cgroup");
            if (parts[0] == "0" && parts[1].Length == 0)
            {
                if (unified != null) throw Invalid("/proc/self/cgroup");
                unified = Path(parts[2], "/proc/self/cgroup");
            }
            else if (Words(parts[1], ',').Contains("memory", StringComparer.Ordinal))
            {
                if (memory != null) throw Invalid("/proc/self/cgroup");
                memory = Path(parts[2], "/proc/self/cgroup");
            }
        }

        // A hybrid installation can mount v2 while the memory controller still
        // belongs to v1. Its explicit memory membership takes precedence.
        bool v2 = memory == null;
        string? member = memory ?? unified;
        if (member == null)
        {
            // No membership is only unbounded when the kernel explicitly says
            // its memory controller is disabled. Missing information is unknown.
            foreach (string line in Lines(Read("/proc/cgroups")))
            {
                string[] fields = line.Split((char[]?)null, StringSplitOptions.RemoveEmptyEntries);
                if (fields.Length == 4 && fields[0] == "memory" && fields[3] == "0") return (total, available);
            }
            throw new IOException("The process memory cgroup could not be determined.");
        }

        var mounts = new List<(string Root, string Point)>();
        foreach (string line in Lines(Read("/proc/self/mountinfo")))
        {
            string[] fields = Words(line, ' ');
            int separator = Array.IndexOf(fields, "-");
            if (separator < 6 || fields.Length <= separator + 3) throw Invalid("/proc/self/mountinfo");
            string fs = fields[separator + 1];
            if (v2 ? fs != "cgroup2" : fs != "cgroup"
                || !Words(fields[separator + 3], ',').Contains("memory", StringComparer.Ordinal)) continue;
            string root = Path(Unescape(fields[3]), "/proc/self/mountinfo");
            string point = Path(Unescape(fields[4]), "/proc/self/mountinfo");
            if (Within(member, root)) mounts.Add((root, point));
        }
        if (mounts.Count == 0) throw new IOException("No readable mount maps the process memory cgroup.");
        // Prefer the widest view of the hierarchy, not a convenient bind mount
        // of only the leaf, so accessible parent quotas are not accidentally lost.
        var mount = mounts.OrderBy(m => m.Root.Length).First();
        string suffix = mount.Root == "/" ? member : member[mount.Root.Length..];
        string leaf = suffix is "" or "/" ? mount.Point : Join(mount.Point, suffix.TrimStart('/'));
        long reclaimable = ReclaimableLeafCache(leaf);
        string directory = leaf;
        while (true)
        {
            bool root = directory == mount.Point;
            if (v2)
            {
                string? limit = Optional(Join(directory, "memory.max"));
                string? current = Optional(Join(directory, "memory.current"));
                if (limit == null || current == null)
                {
                    if (limit != null || current != null) throw Invalid(directory + "/memory.{max,current}");
                    // The real v2 root has no memory.max, and descendants for
                    // which memory is not enabled also have no controller files.
                    // Their nearest enabled ancestor is still visited below.
                    string[] controllers = Read(Join(directory, "cgroup.controllers"))
                        .Split((char[]?)null, StringSplitOptions.RemoveEmptyEntries);
                    if (!root && controllers.Contains("memory", StringComparer.Ordinal))
                        throw new IOException($"Missing memory controller observation at '{directory}'.");
                }
                else
                {
                    long used = Number(current.Trim(), directory + "/memory.current");
                    if (limit.Trim() != "max") Apply(Number(limit.Trim(), directory + "/memory.max"), used);
                }
            }
            else
            {
                string hierarchy = Read(Join(directory, "memory.use_hierarchy")).Trim();
                if (hierarchy is not ("0" or "1")) throw Invalid(directory + "/memory.use_hierarchy");
                // v1 permits a non-hierarchical ancestor. Its private usage and
                // limit do not constrain this descendant; the leaf always does.
                if (directory == leaf || hierarchy == "1")
                    Apply(Number(Read(Join(directory, "memory.limit_in_bytes")).Trim(), directory + "/memory.limit_in_bytes"),
                        Number(Read(Join(directory, "memory.usage_in_bytes")).Trim(), directory + "/memory.usage_in_bytes"));
            }
            if (root) break;
            int slash = directory.LastIndexOf('/');
            directory = slash == 0 ? "/" : directory[..slash];
            if (!Within(directory, mount.Point)) throw new IOException("Cgroup hierarchy escaped its visible mount.");
        }
        return (total, Math.Min(total, available));

        void Apply(long limit, long used)
        {
            total = Math.Min(total, limit);
            // Only this leaf's observed clean inactive file cache is credited
            // at every ancestor. Do not borrow siblings' protected working sets.
            decimal remaining = (decimal)limit - used + Math.Min(used, reclaimable);
            available = Math.Min(available, (long)Math.Clamp(remaining, 0, limit));
        }
        long ReclaimableLeafCache(string leafDirectory)
        {
            var stats = ParseStats(Optional(Join(leafDirectory, "memory.stat")));
            if (stats == null) return 0;
            if (v2)
            {
                // v2 memory.stat aggregates descendants, which may have hard
                // reclaim protection. With no local-only counter, credit only a
                // proven leaf; missing observations retain the strict old bound.
                var tree = ParseStats(Optional(Join(leafDirectory, "cgroup.stat")));
                if (tree == null || !tree.TryGetValue("nr_descendants", out long children) || children != 0
                    || !tree.TryGetValue("nr_dying_descendants", out long dying) || dying != 0) return 0;
            }
            string[] exclusions = v2
                ? ["file_dirty", "file_writeback", "file_mapped", "shmem", "unevictable"]
                : ["dirty", "writeback", "mapped_file", "shmem", "unevictable"];
            if (!stats.TryGetValue("inactive_file", out long inactive) || exclusions.Any(k => !stats.ContainsKey(k))) return 0;
            decimal clean = inactive;
            foreach (string key in exclusions) clean -= stats[key];
            if (v2)
            {
                string? min = Optional(Join(leafDirectory, "memory.min"));
                string? low = Optional(Join(leafDirectory, "memory.low"));
                if (min == null || low == null) return 0;
                clean -= Math.Max(Number(min.Trim(), leafDirectory + "/memory.min"), Number(low.Trim(), leafDirectory + "/memory.low"));
            }
            return (long)Math.Max(0, clean);
        }
        Dictionary<string, long>? ParseStats(string? text)
        {
            if (text == null) return null;
            var values = new Dictionary<string, long>(StringComparer.Ordinal);
            foreach (string line in Lines(text))
            {
                var fields = line.Split((char[]?)null, StringSplitOptions.RemoveEmptyEntries);
                if (fields.Length != 2 || !values.TryAdd(fields[0], Number(fields[1], "cgroup statistics")))
                    throw Invalid("cgroup statistics");
            }
            return values;
        }
        string Read(string path)
        {
            try { return readText(path); }
            catch (Exception ex) when (ex is IOException or UnauthorizedAccessException)
            { throw new IOException($"Unable to observe host memory at '{path}'.", ex); }
        }
        string? Optional(string path)
        {
            try { return readText(path); }
            catch (Exception ex) when (ex is FileNotFoundException or DirectoryNotFoundException) { return null; }
            catch (Exception ex) when (ex is IOException or UnauthorizedAccessException)
            { throw new IOException($"Unable to observe host memory at '{path}'.", ex); }
        }
    }

    private static IEnumerable<string> Lines(string value) => value.Split('\n').Select(l => l.TrimEnd('\r')).Where(l => !string.IsNullOrWhiteSpace(l));
    private static string[] Words(string value, char separator) => value.Split(separator, StringSplitOptions.RemoveEmptyEntries);
    private static IOException Invalid(string path) => new($"Invalid host-memory observation at '{path}'.");
    private static long Number(string value, string path) => long.TryParse(value, NumberStyles.None,
        CultureInfo.InvariantCulture, out long number) && number >= 0 ? number : throw Invalid(path);
    private static bool Within(string path, string root) => root == "/" || path == root || path.StartsWith(root + "/", StringComparison.Ordinal);
    private static string Join(string root, string suffix) => root == "/" ? "/" + suffix : root + "/" + suffix;
    private static string Path(string value, string source)
    {
        if (!value.StartsWith('/') || value.Contains('\0') || value.Contains("//", StringComparison.Ordinal)
            || value.Split('/').Any(p => p is "." or "..")) throw Invalid(source);
        return value.Length > 1 ? value.TrimEnd('/') : value;
    }
    private static string Unescape(string value)
    {
        // mountinfo's octal escapes are decoded once; literal "\\040" after an
        // escaped backslash must not be interpreted as a second escape.
        var result = new System.Text.StringBuilder(value.Length);
        for (int i = 0; i < value.Length; i++)
        {
            if (value[i] != '\\') { result.Append(value[i]); continue; }
            if (i + 3 >= value.Length) throw Invalid("/proc/self/mountinfo");
            result.Append(value.Substring(i + 1, 3) switch
            { "040" => ' ', "011" => '\t', "012" => '\n', "134" => '\\', _ => throw Invalid("/proc/self/mountinfo") });
            i += 3;
        }
        return result.ToString();
    }
}
