// Copyright (c) Zhongkai Fu. Licensed under the BSD-3-Clause license.
using System.Globalization;
using TensorSharp.Models;
using Xunit;

namespace InferenceWeb.Tests;

public sealed class HostMemoryAvailabilityTests
{
    private const long GiB = 1L << 30;

    [Fact]
    public void V1CreditsOnlyOwnCleanInactiveCacheNotHierarchicalOrActiveCache()
    {
        var f = V1();
        V1Limit(f, "/cg/team/job", 4 * GiB, 4 * GiB);
        V1Limit(f, "/cg/team", 8 * GiB, 8 * GiB);
        f["/cg/team/job/memory.stat"] = $"inactive_file {3 * GiB}\ndirty {GiB / 4}\nwriteback {GiB / 4}\nmapped_file {GiB / 4}\nshmem {GiB / 4}\nunevictable 0\nactive_file {8 * GiB}\ntotal_inactive_file {16 * GiB}\n";
        Assert.Equal((4 * GiB, 2 * GiB), Read(f));
        f["/cg/team/job/memory.stat"] = "inactive_file 100\ndirty 200\nwriteback 0\nmapped_file 0\nshmem 0\nunevictable 0";
        Assert.Equal((4 * GiB, 0L), Read(f));
    }

    [Fact]
    public void V2CreditsVerifiedLeafCacheWhileRespectingProtectionAndHostAvailability()
    {
        var f = V2("/job"); Limit(f, "/cg/job", 4 * GiB, 4 * GiB);
        f["/cg/job/cgroup.stat"] = "nr_descendants 0\nnr_dying_descendants 0";
        f["/cg/job/memory.stat"] = $"inactive_file {3 * GiB}\nfile_dirty 0\nfile_writeback 0\nfile_mapped 0\nshmem 0\nunevictable 0";
        f["/cg/job/memory.min"] = GiB.ToString(); f["/cg/job/memory.low"] = "0";
        Assert.Equal((4 * GiB, 2 * GiB), Read(f));
        f["/proc/meminfo"] = MemInfo(16 * GiB, GiB);
        Assert.Equal((4 * GiB, GiB), Read(f));
        f["/cg/job/cgroup.stat"] = "nr_descendants 1\nnr_dying_descendants 0";
        Assert.Equal((4 * GiB, 0L), Read(f));
    }

    [Fact]
    public void MissingCacheDetailsNeverRelaxQuotaAndMalformedStatsFailClosed()
    {
        var f = V1(); V1Limit(f, "/cg/team/job", GiB, GiB); V1Limit(f, "/cg/team", 2 * GiB, GiB);
        f["/cg/team/job/memory.stat"] = $"inactive_file {GiB}";
        Assert.Equal((GiB, 0L), Read(f));
        f["/cg/team/job/memory.stat"] = "inactive_file -1";
        Assert.Throws<IOException>(() => Read(f));
    }

    [Fact]
    public void V2UsesTheTightestRemainingAncestorQuotaIncludingSiblingUsage()
    {
        var f = V2("/team/job");
        Limit(f, "/cg/team/job", 4 * GiB, GiB);
        Limit(f, "/cg/team", 8 * GiB, 7 * GiB);
        Assert.Equal((4 * GiB, GiB), Read(f));
    }

    [Fact]
    public void HardwareAvailabilityIsAlreadyNetOfHostUsageAndIsNotSubtractedAgain()
    {
        var f = V2("/job");
        f["/proc/meminfo"] = MemInfo(16 * GiB, 2 * GiB);
        Limit(f, "/cg/job", 8 * GiB, 3 * GiB);
        Assert.Equal((8 * GiB, 2 * GiB), Read(f));
    }

    [Theory]
    [InlineData(0, 0)]
    [InlineData(100, 150)]
    public void ZeroOrAlreadyExceededQuotaCannotAdmitAnotherByte(long limit, long current)
    {
        var f = V2("/job"); Limit(f, "/cg/job", limit, current);
        Assert.Equal((limit, 0L), Read(f));
    }

    [Fact]
    public void UnlimitedVisibleHierarchyStillUsesMemAvailable()
    {
        var f = V2("/job");
        f["/cg/job/memory.max"] = "max\n";
        f["/cg/job/memory.current"] = (2 * GiB).ToString(CultureInfo.InvariantCulture);
        Assert.Equal((16 * GiB, 12 * GiB), Read(f));
    }

    [Fact]
    public void RealUnifiedRootHasNoControllerLimitFiles()
    {
        var f = V2("/");
        f.Remove("/cg/memory.max"); f.Remove("/cg/memory.current");
        f["/cg/cgroup.controllers"] = "cpu io memory pids";
        Assert.Equal((16 * GiB, 12 * GiB), Read(f));
    }

    [Fact]
    public void DisabledLeafControllerStillInheritsVisibleAncestorPressure()
    {
        var f = V2("/team/job");
        f["/cg/team/job/cgroup.controllers"] = "cpu pids";
        Limit(f, "/cg/team", 4 * GiB, 3 * GiB);
        Assert.Equal((4 * GiB, GiB), Read(f));
    }

    [Fact]
    public void BindMountedSubtreeMapsMembershipRelativeToItsMountRoot()
    {
        var f = V2("/docker/container/job");
        f["/proc/self/mountinfo"] = Mount(1, "/docker/container", "/cg", "cgroup2", "rw");
        Limit(f, "/cg/job", 6 * GiB, 2 * GiB);
        Limit(f, "/cg", 8 * GiB, 7 * GiB);
        Assert.Equal((6 * GiB, GiB), Read(f));
    }

    [Fact]
    public void NamespaceRootUsesItsVisibleLimitWithoutPretendingHiddenAncestorsWereObserved()
    {
        var f = V2("/"); Limit(f, "/cg", 2 * GiB, GiB);
        Assert.Equal((2 * GiB, GiB), Read(f));
    }

    [Fact]
    public void WidestMountWinsOverAConvenientLeafBindMount()
    {
        var f = V2("/team/job");
        f["/proc/self/mountinfo"] = Mount(2, "/team/job", "/leaf", "cgroup2", "rw")
            + "\n" + Mount(1, "/", "/cg", "cgroup2", "rw");
        Limit(f, "/leaf", 8 * GiB, 0);
        Limit(f, "/cg/team/job", 8 * GiB, 0);
        Limit(f, "/cg/team", 4 * GiB, 3 * GiB);
        Assert.Equal((4 * GiB, GiB), Read(f));
    }

    [Fact]
    public void MountInfoEscapesAreDecodedOnlyOnce()
    {
        var f = V2("/job");
        f["/proc/self/mountinfo"] = Mount(1, "/", "/memory\\040space\\134040", "cgroup2", "rw");
        Limit(f, "/memory space\\040/job", GiB, 0);
        Limit(f, "/memory space\\040", 2 * GiB, 0);
        Assert.Equal((GiB, GiB), Read(f));
    }

    [Fact]
    public void HybridInstallationUsesItsV1MemoryControllerAndEveryHierarchicalAncestor()
    {
        var f = V1();
        f["/proc/self/cgroup"] += "\n0::/unified-job";
        f["/proc/self/mountinfo"] += "\n" + Mount(2, "/", "/unified", "cgroup2", "rw");
        V1Limit(f, "/cg/team/job", 4 * GiB, GiB);
        V1Limit(f, "/cg/team", 8 * GiB, 7 * GiB);
        Assert.Equal((4 * GiB, GiB), Read(f));
    }

    [Fact]
    public void V1NonHierarchicalAncestorDoesNotChargeItsPrivateUsageToDescendants()
    {
        var f = V1();
        V1Limit(f, "/cg/team/job", 4 * GiB, GiB, hierarchy: false);
        V1Limit(f, "/cg/team", GiB, GiB, hierarchy: false);
        Assert.Equal((4 * GiB, 3 * GiB), Read(f));
    }

    [Fact]
    public void V1UnlimitedNumericSentinelDoesNotOverflowOrHideHostPressure()
    {
        var f = V1();
        V1Limit(f, "/cg/team/job", 9223372036854771712, 2 * GiB);
        V1Limit(f, "/cg/team", 9223372036854771712, 3 * GiB);
        Assert.Equal((16 * GiB, 12 * GiB), Read(f));
    }

    [Theory]
    [InlineData("missing-ancestor-limit")]
    [InlineData("missing-ancestor-pair")]
    [InlineData("malformed-limit")]
    [InlineData("negative-usage")]
    [InlineData("overflow-limit")]
    [InlineData("missing-mount")]
    [InlineData("outside-mount")]
    [InlineData("parent-traversal")]
    [InlineData("missing-host-available")]
    [InlineData("wrong-host-unit")]
    [InlineData("overflow-host-kib")]
    public void UnknownObservationsFailClosedInsteadOfBecomingUnlimited(string mutation)
    {
        var f = V2("/team/job");
        Limit(f, "/cg/team/job", 4 * GiB, GiB);
        Limit(f, "/cg/team", 8 * GiB, 2 * GiB);
        switch (mutation)
        {
            case "missing-ancestor-limit": f.Remove("/cg/team/memory.max"); break;
            case "missing-ancestor-pair":
                f.Remove("/cg/team/memory.max"); f.Remove("/cg/team/memory.current");
                f["/cg/team/cgroup.controllers"] = "memory cpu"; break;
            case "malformed-limit": f["/cg/team/memory.max"] = "unlimited"; break;
            case "negative-usage": f["/cg/team/memory.current"] = "-1"; break;
            case "overflow-limit": f["/cg/team/memory.max"] = "18446744073709551615"; break;
            case "missing-mount": f["/proc/self/mountinfo"] = Mount(1, "/", "/x", "tmpfs", "rw"); break;
            case "outside-mount": f["/proc/self/mountinfo"] = Mount(1, "/team2", "/cg", "cgroup2", "rw"); break;
            case "parent-traversal": f["/proc/self/cgroup"] = "0::/../host"; break;
            case "missing-host-available": f["/proc/meminfo"] = "MemTotal: 16777216 kB"; break;
            case "wrong-host-unit": f["/proc/meminfo"] = "MemTotal: 16 GB\nMemAvailable: 12 GB"; break;
            case "overflow-host-kib": f["/proc/meminfo"] = "MemTotal: 9223372036854775807 kB\nMemAvailable: 1 kB"; break;
        }
        Assert.Throws<IOException>(() => Read(f));
    }

    [Fact]
    public void PermissionFailureCannotBeMistakenForAnAbsentController()
    {
        var f = V2("/job"); Limit(f, "/cg/job", GiB, 0);
        IOException error = Assert.Throws<IOException>(() => HostMemoryAvailability.ReadLinux(path =>
            path == "/cg/job/memory.current" ? throw new UnauthorizedAccessException("test") : Get(f, path)));
        Assert.IsType<UnauthorizedAccessException>(error.InnerException);
    }

    [Fact]
    public void MissingMembershipRequiresExplicitKernelEvidenceThatMemoryControllerIsDisabled()
    {
        var f = V2("/");
        f["/proc/self/cgroup"] = "2:cpu:/job";
        f["/proc/cgroups"] = "#subsys_name hierarchy num_cgroups enabled\nmemory 0 1 0";
        Assert.Equal((16 * GiB, 12 * GiB), Read(f));
        f["/proc/cgroups"] = "memory 0 1 1";
        Assert.Throws<IOException>(() => Read(f));
    }

    private static Dictionary<string, string> V2(string member) => new(StringComparer.Ordinal)
    {
        ["/proc/meminfo"] = MemInfo(16 * GiB, 12 * GiB),
        ["/proc/self/cgroup"] = "0::" + member,
        ["/proc/self/mountinfo"] = Mount(1, "/", "/cg", "cgroup2", "rw"),
        ["/cg/memory.max"] = "max", ["/cg/memory.current"] = "0"
    };
    private static Dictionary<string, string> V1()
    {
        var result = V2("/");
        result["/proc/self/cgroup"] = "4:cpu,cpuacct:/other\n8:memory:/team/job";
        result["/proc/self/mountinfo"] = Mount(1, "/", "/cg", "cgroup", "rw,memory");
        V1Limit(result, "/cg", 9223372036854771712, 0);
        return result;
    }
    private static string Mount(int id, string root, string point, string fs, string options)
        => $"{id} 0 0:30 {root} {point} rw,nosuid,nodev - {fs} cgroup {options}";
    private static string MemInfo(long total, long available) => $"MemTotal: {total / 1024} kB\nMemAvailable: {available / 1024} kB\n";
    private static void Limit(Dictionary<string, string> f, string directory, long limit, long used)
    { f[directory + "/memory.max"] = limit.ToString(CultureInfo.InvariantCulture); f[directory + "/memory.current"] = used.ToString(CultureInfo.InvariantCulture); }
    private static void V1Limit(Dictionary<string, string> f, string directory, long limit, long used, bool hierarchy = true)
    {
        f[directory + "/memory.limit_in_bytes"] = limit.ToString(CultureInfo.InvariantCulture);
        f[directory + "/memory.usage_in_bytes"] = used.ToString(CultureInfo.InvariantCulture);
        f[directory + "/memory.use_hierarchy"] = hierarchy ? "1" : "0";
    }
    private static string Get(Dictionary<string, string> f, string path) => f.TryGetValue(path, out string? value)
        ? value : throw new FileNotFoundException("fixture path missing", path);
    private static (long Total, long Available) Read(Dictionary<string, string> f) => HostMemoryAvailability.ReadLinux(path => Get(f, path));
}
