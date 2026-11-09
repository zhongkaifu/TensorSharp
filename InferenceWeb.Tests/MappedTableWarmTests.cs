// Copyright (c) Zhongkai Fu. All rights reserved.
// https://github.com/zhongkaifu/TensorSharp
//
// This file is part of TensorSharp.
//
// TensorSharp is licensed under the BSD-3-Clause license found in the LICENSE file in the root directory of this source tree.
//
// TensorSharp is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the BSD-3-Clause License for more details.
using System;
using System.IO;
using TensorSharp.Models;
using Xunit;

namespace InferenceWeb.Tests;

/// <summary>
/// A lookup-table warm is skipped when the host cannot keep the table cached. Inside a
/// container that limit is the memory cgroup's, not the host's MemAvailable (a 93 GB
/// cgroup on a 187 GB host reported 138 GB available), so these pin the cgroup reading.
/// </summary>
public sealed class MappedTableWarmTests : IDisposable
{
    private readonly string _dir = Path.Combine(Path.GetTempPath(), "ts-cgroup-" + Guid.NewGuid().ToString("N"));

    public MappedTableWarmTests() => Directory.CreateDirectory(_dir);

    public void Dispose()
    {
        try { Directory.Delete(_dir, recursive: true); } catch (IOException) { }
    }

    private void Write(string name, string text) => File.WriteAllText(Path.Combine(_dir, name), text);

    [Fact]
    public void CgroupV2_LimitLessUsagePlusFileCache()
    {
        Write("memory.max", "92999999488\n");
        Write("memory.current", "46522081280\n");
        Write("memory.stat", "anon 1220341760\nfile 44898598912\nshmem 28898205696\n" +
            "active_anon 30118408192\ninactive_file 12474654720\nactive_file 3525566464\n");

        Assert.True(MappedTableWarm.TryCgroup(_dir, "memory.max", "memory.current", "active_file", "inactive_file",
            out long available));
        // shmem stays charged: tmpfs pages sit on the anon lists, not the file ones.
        Assert.Equal(92999999488L - 46522081280L + 12474654720L + 3525566464L, available);
    }

    [Fact]
    public void CgroupV2_UnlimitedLeavesTheHostFigure()
    {
        Write("memory.max", "max\n");
        Write("memory.current", "1000\n");
        Write("memory.stat", "active_file 10\ninactive_file 10\n");

        Assert.False(MappedTableWarm.TryCgroup(_dir, "memory.max", "memory.current", "active_file", "inactive_file", out _));
    }

    [Fact]
    public void CgroupV1_UsesTotalFileCounters()
    {
        Write("memory.limit_in_bytes", "8589934592\n");
        Write("memory.usage_in_bytes", "6442450944\n");
        Write("memory.stat", "cache 100\nactive_file 999\ntotal_active_file 1073741824\ntotal_inactive_file 536870912\n");

        Assert.True(MappedTableWarm.TryCgroup(_dir, "memory.limit_in_bytes", "memory.usage_in_bytes",
            "total_active_file", "total_inactive_file", out long available));
        Assert.Equal(8589934592L - 6442450944L + 1073741824L + 536870912L, available);
    }

    [Fact]
    public void CgroupV1_UnlimitedSentinelLeavesTheHostFigure()
    {
        Write("memory.limit_in_bytes", "9223372036854771712\n");
        Write("memory.usage_in_bytes", "1000\n");
        Write("memory.stat", "total_active_file 1\ntotal_inactive_file 1\n");

        Assert.False(MappedTableWarm.TryCgroup(_dir, "memory.limit_in_bytes", "memory.usage_in_bytes",
            "total_active_file", "total_inactive_file", out _));
    }
}
