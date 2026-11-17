// Copyright (c) Zhongkai Fu. All rights reserved.
// https://github.com/zhongkaifu/TensorSharp
//
// This file is part of TensorSharp.
//
// TensorSharp is licensed under the BSD-3-Clause license found in the LICENSE file in the root directory of this source tree.
using System;
using System.Collections.Generic;
using System.Runtime.InteropServices;
using TensorSharp;
using TensorSharp.Cuda;
using Xunit;
using Xunit.Abstractions;

namespace InferenceWeb.Tests;

/// <summary>
/// The grouped MoE kernels (CudaMoe.Experts: the routes grouped by expert on the device, tensor-core or staged grouped
/// GEMMs, the weighted scatter) over per-expert pointer tables, as the CUDA executor runs a prompt chunk's staged and tier
/// experts, against the batched per-row kernels on the same routes: every row agrees to the activations' rounding.
/// </summary>
public sealed class CudaGroupedMoeTablesTests
{
    private readonly ITestOutputHelper _output;

    public CudaGroupedMoeTablesTests(ITestOutputHelper output) => _output = output;

    private static (int Bytes, int Values) Block(int type) => type switch
    {
        23 => (136, 256),   // IQ4_XS
        21 => (110, 256),   // IQ3_S
        17 => (74, 256),    // IQ2_XS
        _ => throw new ArgumentOutOfRangeException(nameof(type)),
    };

    private static byte[] RandomRows(int type, int rows, int cols, Random rng)
    {
        (int bytes, int values) = Block(type);
        int blocks = rows * (cols / values);
        var data = new byte[blocks * bytes];
        rng.NextBytes(data);
        // Scales that keep the products a checkpoint's size: random sub-block scales and codes reach ~30x the block
        // scale, and the tensor-core route stages the SwiGLU hidden as F16 (a real model's stays far below its range).
        for (int b = 0; b < blocks; b++)
            BitConverter.TryWriteBytes(data.AsSpan(b * bytes, 2), (System.Half)(0.0001 + rng.NextDouble() * 0.0003));
        return data;
    }

    [CudaTheory]
    [InlineData(23, 23, 17)]
    [InlineData(23, 23, 300)]
    [InlineData(23, 23, 129)]
    [InlineData(23, 23, 200)]
    [InlineData(21, 23, 17)]
    [InlineData(17, 23, 130)]
    public void TablesMatchThePerRowKernels(int gateType, int downType, int rows)
    {
        const int hidden = 256, ff = 256, experts = 16, k = 4;
        using var allocator = new CudaAllocator();
        allocator.Context.MakeCurrent();
        Dsv4Kernels dk;
        try { dk = Dsv4Kernels.Create(); }
        catch (InvalidOperationException ex)
        {
            _output.WriteLine($"no grouped MoE module: {ex.Message}");
            return;
        }
        using (dk)
        {
            if (!CudaMoe.GroupedSupportsType(gateType, dk.DecodesLowBitIqExperts, dk.DecodesLegacyQuants)
                || !CudaMoe.GroupedSupportsType(downType, dk.DecodesLowBitIqExperts, dk.DecodesLegacyQuants))
            {
                _output.WriteLine("the loaded module's grouped kernels do not take these types");
                return;
            }
            var rng = new Random(41 + gateType + rows);
            var hosts = new List<IntPtr>();
            var tables = new IntPtr[3];
            var owned = new List<Tensor>();
            try
            {
                var resident = new IntPtr[3, experts];
                long[] rowBytes = new long[3];
                for (int m = 0; m < 3; m++)
                    for (int e = 0; e < experts; e++)
                    {
                        int type = m < 2 ? gateType : downType, inDim = m < 2 ? hidden : ff, outDim = m < 2 ? ff : hidden;
                        byte[] w = RandomRows(type, outDim, inDim, rng);
                        rowBytes[m] = w.Length / outDim;
                        IntPtr host = Marshal.AllocHGlobal(w.Length);
                        hosts.Add(host);
                        Marshal.Copy(w, 0, host, w.Length);
                        var key = new IntPtr(0x7b0000 + 1000 * rows + 64 * gateType + 16 * m + e);
                        CudaQuantizedOps.PreloadQuantizedWeight(allocator, key, host, type, inDim, outDim, w.Length);
                        Assert.True(CudaQuantizedOps.TryGetResidentDevicePtr(allocator, key, out resident[m, e], out _));
                    }
                for (int m = 0; m < 3; m++)
                {
                    var row = new IntPtr[experts];
                    for (int e = 0; e < experts; e++) row[e] = resident[m, e];
                    tables[m] = CudaQuantizedOps.CreateDevicePointerTable(allocator, row);
                }

                var x = new float[rows, hidden];
                var logits = new float[rows, experts];
                for (int r = 0; r < rows; r++)
                {
                    for (int i = 0; i < hidden; i++) x[r, i] = (float)(rng.NextDouble() * 2 - 1);
                    for (int e = 0; e < experts; e++) logits[r, e] = (float)(rng.NextDouble() * 4 - 2);
                }
                Tensor T(Tensor t) { owned.Add(t); return t; }
                Tensor xt = T(Tensor.FromArray(allocator, x)), lt = T(Tensor.FromArray(allocator, logits));
                Tensor expected = T(new Tensor(allocator, DType.Float32, rows, hidden));
                Tensor sel = T(new Tensor(allocator, DType.Int32, rows * k)), wts = T(new Tensor(allocator, DType.Float32, rows * k));
                Tensor gateOut = T(new Tensor(allocator, DType.Float32, rows * k, ff)), upOut = T(new Tensor(allocator, DType.Float32, rows * k, ff));
                Assert.True(CudaFusedOps.TryMoEExpertFFNPrefillSwiGLU(lt, xt, expected, sel, wts, gateOut, upOut, IntPtr.Zero,
                    tables[0], tables[1], tables[2], gateType, downType, experts, k, hidden, ff, rows, null, IntPtr.Zero));

                // The same routes through the grouped kernels, as the executor hands them over.
                var scratch = new CudaMoeScratch((type, sizes) => T(new Tensor(allocator, type, sizes)), rows, experts, k, hidden, ff);
                using (Tensor sd = scratch.Sel.View(rows * k), wd = scratch.SelW.View(rows * k))
                {
                    Ops.Copy(sd, sel);
                    Ops.Copy(wd, wts);
                }
                Tensor got = T(new Tensor(allocator, DType.Float32, rows, hidden));
                Ops.Fill(got, 0f);
                var gate = new DeviceWeight { Type = gateType, Ne0 = hidden, Ne1 = ff, RowBytes = rowBytes[0], Table = tables[0] };
                var up = new DeviceWeight { Type = gateType, Ne0 = hidden, Ne1 = ff, RowBytes = rowBytes[1], Table = tables[1] };
                var down = new DeviceWeight { Type = downType, Ne0 = ff, Ne1 = hidden, RowBytes = rowBytes[2], Table = tables[2] };
                foreach (Tensor t in new[] { xt, got }) ((CudaStorage)t.Storage).EnsureDeviceCurrent();
                allocator.Context.MakeCurrent();
                CudaMoe.Experts(dk, allocator.Kernels, scratch, gate, up, down, xt, null, got, rows, k, experts, hidden, ff, 0f, 16,
                    allocator.Stream.Handle);
                allocator.Stream.Synchronize();
                ((CudaStorage)got.Storage).MarkDeviceModified();
                float[] a = expected.GetElementsAsFloat(rows * hidden), b = got.GetElementsAsFloat(rows * hidden);
                int badA = 0, badB = 0, firstBad = -1;
                for (int i = 0; i < a.Length; i++)
                {
                    if (!float.IsFinite(a[i])) badA++;
                    if (!float.IsFinite(b[i])) { badB++; if (firstBad < 0) firstBad = i / hidden; }
                }
                _output.WriteLine($"non-finite: per-row {badA}, grouped {badB} (first grouped row {firstBad})");
                double worst = 0;
                for (int r = 0; r < rows; r++)
                {
                    double num = 0, den = 0;
                    for (int i = 0; i < hidden; i++)
                    {
                        double d = b[r * hidden + i] - a[r * hidden + i];
                        num += d * d;
                        den += a[r * hidden + i] * (double)a[r * hidden + i];
                    }
                    worst = Math.Max(worst, Math.Sqrt(num / Math.Max(den, 1e-30)));
                }
                CudaMoe.ExpertRoute route = CudaMoe.SelectRoute(rows, 16, dk.HasMoeMma, dk.DecodesLowBitIqExperts, gateType, downType, hidden, ff,
                    dk.DecodesLegacyQuants);
                _output.WriteLine($"types {gateType}/{downType}, {rows} rows, route {route}: worst row relative L2 {worst:E2}");
                Assert.True(worst < 3e-2, $"route {route}: worst row relative L2 {worst:E2}");
            }
            finally
            {
                allocator.Synchronize();
                foreach (Tensor t in owned) t.Dispose();
                foreach (IntPtr table in tables)
                    if (table != IntPtr.Zero) CudaQuantizedOps.FreeDeviceBuffer(allocator, table);
                foreach (IntPtr host in hosts) Marshal.FreeHGlobal(host);
            }
        }
    }
}
