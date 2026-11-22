// Copyright (c) Zhongkai Fu. All rights reserved.
// https://github.com/zhongkaifu/TensorSharp
//
// This file is part of TensorSharp.
//
// TensorSharp is licensed under the BSD-3-Clause license found in the LICENSE file in the root directory of this source tree.
//
// Device memory claimed ahead of use (GpuMemoryBudget.Reserve): a loaded vision tower's weights before the first image uploads
// them, and the compute of the largest image it accepts, stay out of the free bytes the shared decoder sizes its slots and
// their growth from. A KV cache that grows while no image has come then leaves the first image its room. The managed CPU
// executor runs the direct-CUDA room code through the model's device stand-in (DeviceRoomForTests), on a device number no real
// reading uses, so these reservations reach no other test.
using TensorSharp;
using TensorSharp.Models;
using TensorSharp.Runtime.Scheduling;

namespace InferenceWeb.Tests;

public sealed class DeviceReservationTests : IDisposable
{
    private const long GB = 1L << 30;

    /// <summary>A device number no reading of a real device uses.</summary>
    private const int TestDevice = 1 << 20;

    private readonly string _dir = Path.Combine(Path.GetTempPath(), "ts-device-reservation-" + Guid.NewGuid().ToString("N"));
    private readonly int[] _prompt = Enumerable.Range(0, 20).Select(i => 40 + i % 7).ToArray();

    public DeviceReservationTests() => Directory.CreateDirectory(_dir);

    public void Dispose()
    {
        try { Directory.Delete(_dir, recursive: true); } catch (IOException) { }
    }

    private Qwen35RuntimeModel Create()
    {
        using var dtype = KvCacheDtypeConfig.ForCurrentFlowInTests(KvCacheDtype.F32);
        string path = new Qwen35DenseSyntheticModelBuilder { Layers = 8, UntiedOutput = true }.Write(Path.Combine(_dir, "q35-dense.gguf"));
        var model = new Qwen35RuntimeModel(path, BackendType.Cpu, 1, null, 1);
        model.ReservationDeviceForTests = (BackendType.Cuda, TestDevice);
        return model;
    }

    /// <summary>The claims on a device add up per device and family, follow their owner's updates and go when disposed; the
    /// budget's spare is the free bytes less the headroom and what is claimed.</summary>
    [Fact]
    public void AReservationStaysOutOfTheSpareUntilItsOwnerGivesItUp()
    {
        Assert.Equal(0, GpuMemoryBudget.ReservedBytes(BackendType.Cuda, TestDevice));
        using (DeviceReservation tower = GpuMemoryBudget.Reserve(BackendType.Cuda, TestDevice, "a tower", 3 * GB))
        {
            using DeviceReservation other = GpuMemoryBudget.Reserve(BackendType.Cuda, TestDevice, "another owner", GB);
            Assert.Equal(4 * GB, GpuMemoryBudget.ReservedBytes(BackendType.Cuda, TestDevice));
            Assert.Equal(0, GpuMemoryBudget.ReservedBytes(BackendType.Cuda, TestDevice + 1));
            Assert.Equal(0, GpuMemoryBudget.ReservedBytes(BackendType.GgmlCuda, TestDevice));
            tower.Bytes = GB / 2;   // its weights are uploaded: only the compute is still to come
            Assert.Equal(GB + GB / 2, GpuMemoryBudget.ReservedBytes(BackendType.Cuda, TestDevice));
            tower.Bytes = -1;
            Assert.Equal(0, tower.Bytes);
        }
        Assert.Equal(0, GpuMemoryBudget.ReservedBytes(BackendType.Cuda, TestDevice));

        long headroom = GpuMemoryBudget.ResolveHeadroomBytes(32 * GB);
        Assert.Equal(10 * GB - headroom - 3 * GB, GpuMemoryBudget.SpareBytes(10 * GB, 32 * GB, 3 * GB));
        Assert.Equal(10 * GB - headroom, GpuMemoryBudget.SpareBytes(10 * GB, 32 * GB, 0));
        Assert.Equal(0, GpuMemoryBudget.SpareBytes(headroom + GB, 32 * GB, 3 * GB));
    }

    /// <summary>A new request's slot and a running request's growth leave the reserved bytes free: where the device has room
    /// beside its headroom only with the reservation counted as free, the new slot waits and the growth fails with what holds
    /// the device; once the claim is given up both go ahead.</summary>
    [Fact]
    public void ASlotAndItsGrowthLeaveTheReservedBytesFree()
    {
        using Qwen35RuntimeModel model = Create();
        var unified = (IUnifiedBatchModel)model;
        // 1.5 GB free: a slot fits beside the 1 GB headroom (DeviceRoomTests), and not once 1 GB of it is claimed.
        model.DeviceRoomForTests = () => (GB + GB / 2, 16 * GB);
        Assert.Equal(0, unified.EnsureSequenceSlot("run", 64));
        Assert.True(unified.TryForwardUnified(new[] { "run" }, new[] { _prompt }, new[] { 0 }, new float[1][]));
        int capacity = model.SlotCapacityForTests("run").Value;

        using (GpuMemoryBudget.Reserve(BackendType.Cuda, TestDevice, "a vision tower", GB))
        {
            // Beside a running request, a new one waits for it (alone it would be refused before any of it ran).
            Assert.Throws<SequenceSlotUnavailableException>(() => unified.EnsureSequenceSlot("second", 2048));
            Assert.False(model.HasFusedSequenceCache("second"));

            var exhausted = Assert.Throws<InvalidOperationException>(() => unified.EnsureSequenceSlot("run", 4 * capacity, 4 * capacity));
            Assert.Contains("device memory exhausted", exhausted.Message);
            Assert.Equal(capacity, model.SlotCapacityForTests("run"));
        }

        Assert.Equal(_prompt.Length, unified.EnsureSequenceSlot("run", 4 * capacity, 4 * capacity));
        Assert.True(model.SlotCapacityForTests("run") >= 4 * capacity);
        model.OnSequenceReleased("run");
        Assert.Equal(0, unified.EnsureSequenceSlot("second", 2048));
        Assert.True(unified.TryForwardUnified(new[] { "second" }, new[] { _prompt }, new[] { 0 }, new float[1][]));
    }

    /// <summary>The prefix cache is told what the device lacks of its headroom with the claim counted, so it releases finished
    /// conversations for it; a model standing in for its device sees no claim but on the device it names.</summary>
    [Fact]
    public void TheDeviceDeficitCountsTheReservedBytes()
    {
        using Qwen35RuntimeModel model = Create();
        model.DeviceRoomForTests = () => (2 * GB, 16 * GB);
        Assert.Equal(0, model.DeviceSlotDeficitBytes);
        using (GpuMemoryBudget.Reserve(BackendType.Cuda, TestDevice, "a vision tower", GB + GB / 2))
        {
            Assert.Equal(GB / 2, model.DeviceSlotDeficitBytes);
            model.ReservationDeviceForTests = (BackendType.Cuda, TestDevice + 1);
            Assert.Equal(0, model.DeviceSlotDeficitBytes);
            model.ReservationDeviceForTests = null;   // a stand-in device names none
            Assert.Equal(0, model.DeviceSlotDeficitBytes);
        }
    }
}
