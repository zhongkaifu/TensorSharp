// Copyright (c) Zhongkai Fu. All rights reserved.
// https://github.com/zhongkaifu/TensorSharp
//
// This file is part of TensorSharp.
//
// TensorSharp is licensed under the BSD-3-Clause license found in the LICENSE file in the root directory of this source tree.
using System;
using System.Linq;
using TensorSharp;
using TensorSharp.Cuda;
using TensorSharp.Cuda.Interop;
using Xunit;

namespace InferenceWeb.Tests;

/// <summary>
/// A doorbell layer's misses staged for the device (ts_graph_moe_route_split, ts_graph_moe_stage_copy, ts_graph_moe_wait_add):
/// the split gives the device the first distinct host experts it may read (onDevice 2) up to the call's capacity, points their
/// table entries at staging slots and leaves the rest to the host; the copy brings each one's three matrices over byte for byte;
/// the wait restores the entries and clears the plan.
/// </summary>
public sealed unsafe class CudaMoeStagingKernelTests
{
    private const int Experts = 16, PerRow = 4, Rows = 2, Hidden = 64;
    private static readonly long[] MatrixBytes = { 256, 512, 768 };

    [CudaTheory]
    [InlineData(4)]   // every host route the device may read is staged: nothing rings
    [InlineData(2)]   // two staged, the rest rung
    public void StagedMissesCopyByteForByteAndTheTablesComeBack(int capacity)
    {
        using var allocator = new CudaAllocator();
        allocator.Context.MakeCurrent();
        Assert.True(CudaGraphKernels.TryCreate(out CudaGraphKernels kernels, out string reason), reason);
        using (kernels)
        using (var box = new CudaHostExpertMailbox(Rows, Hidden, PerRow))
        {
            // Experts 1 and 5 on the device, 3, 7, 9 and 12 mapped host experts, the rest the host's alone.
            int[] sel = { 3, 7, 3, 9, 1, 7, 12, 5 };
            var onDevice = new byte[Experts];
            onDevice[1] = onDevice[5] = 1;
            foreach (int e in new[] { 3, 7, 9, 12 }) onDevice[e] = 2;
            if (capacity == 4) { sel[3] = 1; }   // no host route beyond the staged ones (9 dropped)

            // Each expert's matrices in one mapped host allocation; device experts in device memory.
            long perExpert = MatrixBytes.Sum();
            Assert.Equal(0, CudaDriverApi.cuMemHostAlloc(out IntPtr host, new UIntPtr((ulong)(Experts * perExpert)), 0x02 /* DEVICEMAP */));
            try
            {
                var rng = new Random(5 + capacity);
                var bytes = new byte[Experts * perExpert];
                rng.NextBytes(bytes);
                System.Runtime.InteropServices.Marshal.Copy(bytes, 0, host, bytes.Length);
                Assert.Equal(0, CudaDriverApi.cuMemHostGetDevicePointer(out IntPtr hostDev, host, 0));
                using var deviceExperts = new Tensor(allocator, DType.UInt8, Experts * perExpert);
                ((CudaStorage)deviceExperts.Storage).EnsureDeviceCurrent();
                long Address(int e, int m)
                {
                    long off = e * perExpert + MatrixBytes.Take(m).Sum();
                    return onDevice[e] == 1 ? ((CudaStorage)deviceExperts.Storage).DevicePtrAtElement(0).ToInt64() + off
                        : onDevice[e] == 2 ? hostDev.ToInt64() + off : 0;
                }
                long[][] words = Enumerable.Range(0, 3).Select(m => Enumerable.Range(0, Experts).Select(e => Address(e, m)).ToArray()).ToArray();
                Tensor Table(long[] w) => Tensor.FromArray(allocator, w.SelectMany(v => new[] { unchecked((int)v), unchecked((int)(v >> 32)) }).ToArray());
                using Tensor gate = Table(words[0]), up = Table(words[1]), down = Table(words[2]);
                using Tensor restGate = Table(words[0]), restUp = Table(words[1]), restDown = Table(words[2]);
                using var mask = new Tensor(allocator, DType.UInt8, Experts);
                using var selT = Tensor.FromArray(allocator, sel);
                using var x = Tensor.FromArray(allocator, Enumerable.Range(0, Rows * Hidden).Select(i => (float)i).ToArray());
                using var w = Tensor.FromArray(allocator, Enumerable.Repeat(0.25f, Rows * PerRow).ToArray());
                using var devSel = new Tensor(allocator, DType.Int32, Rows * PerRow);
                using var plan = Tensor.FromArray(allocator, new int[32]);
                using var missDev = Tensor.FromArray(allocator, new int[1]);
                long stride = 2048, offUp = 256, offDown = 768;
                using var slots = new Tensor(allocator, DType.UInt8, capacity * stride);
                foreach (Tensor t in new[] { gate, up, down, restGate, restUp, restDown, mask, selT, x, w, devSel, plan, missDev, slots })
                    ((CudaStorage)t.Storage).EnsureDeviceCurrent();
                IntPtr P(Tensor t) => ((CudaStorage)t.Storage).DevicePtrAtElement(t.StorageOffset);
                fixed (byte* m = onDevice)
                    Assert.Equal(0, CudaDriverApi.cuMemcpyHtoD(P(mask), (IntPtr)m, new UIntPtr(Experts)));
                var staging = new CudaGraphKernels.MoeStaging
                {
                    Capacity = capacity, Gate = P(gate), Up = P(up), Down = P(down), SrcGate = P(restGate), SrcUp = P(restUp), SrcDown = P(restDown),
                    Slots = P(slots), Stride = stride, OffUp = offUp, OffDown = offDown, BytesGate = MatrixBytes[0], BytesUp = MatrixBytes[1],
                    BytesDown = MatrixBytes[2], List = P(plan), Count = P(plan) + 16 * sizeof(int),
                };
                IntPtr stream = allocator.Stream.Handle;
                kernels.LaunchMoeRouteSplit(P(x), P(selT), P(w), P(mask), Hidden, Rows, PerRow, 3, 0, P(devSel), IntPtr.Zero, box, P(missDev), stream, staging);
                kernels.LaunchMoeStageCopy(staging, stream);
                allocator.Stream.Synchronize();
                foreach (Tensor t in new[] { gate, up, down, devSel, plan, missDev, slots }) ((CudaStorage)t.Storage).MarkDeviceModified();

                int[] staged = capacity == 4 ? new[] { 3, 7, 12 } : new[] { 3, 7 };
                int[] planInts = plan.GetElementsAsInt(32);
                Assert.Equal(staged.Length, planInts[16]);
                Assert.Equal(staged, planInts.Take(staged.Length));
                int[] want = sel.Select(e => onDevice[e] == 1 || staged.Contains(e) ? e : -1).ToArray();
                Assert.Equal(want, devSel.GetElementsAsInt(sel.Length));
                bool rang = sel.Any(e => onDevice[e] != 1 && !staged.Contains(e));
                Assert.Equal(rang ? 1 : 0, missDev.GetElementsAsInt(1)[0]);
                if (rang)
                    for (int i = 0; i < sel.Length; i++)
                        Assert.Equal(want[i] >= 0 ? -(sel[i] + 2) : sel[i], box.Ids[i]);

                // The staged experts' entries at their slots, and the slots holding their bytes.
                long slotBase = P(slots).ToInt64();
                long[] gateNow = Words(gate), upNow = Words(up), downNow = Words(down);
                byte[] slotBytes = new byte[capacity * stride];
                slots.CopyToArray(slotBytes);
                for (int k = 0; k < staged.Length; k++)
                {
                    int e = staged[k];
                    Assert.Equal(slotBase + k * stride, gateNow[e]);
                    Assert.Equal(slotBase + k * stride + offUp, upNow[e]);
                    Assert.Equal(slotBase + k * stride + offDown, downNow[e]);
                    long[] at = { 0, offUp, offDown };
                    for (int m = 0; m < 3; m++)
                    {
                        long src = e * perExpert + MatrixBytes.Take(m).Sum();
                        Assert.True(slotBytes.AsSpan((int)(k * stride + at[m]), (int)MatrixBytes[m]).SequenceEqual(bytes.AsSpan((int)src, (int)MatrixBytes[m])),
                            $"expert {e} matrix {m} differs in slot {k}");
                    }
                }
                for (int e = 0; e < Experts; e++)
                    if (!staged.Contains(e)) Assert.Equal(words[0][e], gateNow[e]);

                if (rang) return;   // the wait would wait for the host's answer
                using var output = Tensor.FromArray(allocator, new float[Rows * Hidden]);
                ((CudaStorage)output.Storage).EnsureDeviceCurrent();
                kernels.LaunchMoeWaitAdd(box, P(output), Rows * Hidden, P(missDev), stream, staging);
                allocator.Stream.Synchronize();
                foreach (Tensor t in new[] { gate, up, down, plan }) ((CudaStorage)t.Storage).MarkDeviceModified();
                Assert.Equal(words[0], Words(gate));
                Assert.Equal(words[1], Words(up));
                Assert.Equal(words[2], Words(down));
                Assert.Equal(0, plan.GetElementsAsInt(32)[16]);
            }
            finally
            {
                CudaDriverApi.cuMemFreeHost(host);
            }
        }
    }

    private static long[] Words(Tensor table)
    {
        int[] ints = table.GetElementsAsInt((int)table.ElementCount());
        return Enumerable.Range(0, ints.Length / 2).Select(i => (long)(uint)ints[2 * i] | ((long)ints[2 * i + 1] << 32)).ToArray();
    }
}
