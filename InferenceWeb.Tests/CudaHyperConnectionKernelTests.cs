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
using Xunit;

namespace InferenceWeb.Tests;

/// <summary>
/// The hyper-connection read's kernels write the q8_1 copies their outputs' products read (the normed streams, the
/// bottleneck, the mix): byte for byte what the row quantizer writes from the same values, and the values as without.
/// </summary>
public sealed class CudaHyperConnectionKernelTests
{
    private static IntPtr P(Tensor t) => ((CudaStorage)t.Storage).DevicePtrAtElement(t.StorageOffset);

    private static void Upload(params Tensor[] tensors)
    {
        foreach (Tensor t in tensors) ((CudaStorage)t.Storage).EnsureDeviceCurrent();
    }

    private static void Written(params Tensor[] tensors)
    {
        foreach (Tensor t in tensors) ((CudaStorage)t.Storage).MarkDeviceModified();
    }

    [CudaTheory]
    [InlineData(4, 4, 2560, 64)]
    [InlineData(1, 4, 256, 32)]
    [InlineData(32, 2, 512, 128)]
    public void FusedQuantizedCopiesMatchTheRowQuantizer(int rows, int hc, int n, int lr)
    {
        using var allocator = new CudaAllocator();
        allocator.Context.MakeCurrent();
        Assert.True(CudaGraphKernels.TryCreate(out CudaGraphKernels kernels, out string reason), reason);
        using (kernels)
        {
            var rng = new Random(29);
            float[] R(int count, double scale) => Enumerable.Range(0, count).Select(_ => (float)((rng.NextDouble() * 2 - 1) * scale)).ToArray();
            int width = hc * n;
            using var x = Tensor.FromArray(allocator, R(rows * width, 3));
            using var gamma = Tensor.FromArray(allocator, R(width, 1.5));
            using var lo = Tensor.FromArray(allocator, R(rows * lr, 4));
            using var gates = Tensor.FromArray(allocator, R(rows * width, 5));
            using var xn = new Tensor(allocator, DType.Float32, rows * width);
            using var xn2 = new Tensor(allocator, DType.Float32, rows * width);
            using var silu = new Tensor(allocator, DType.Float32, rows * lr);
            using var silu2 = new Tensor(allocator, DType.Float32, rows * lr);
            using var mix = new Tensor(allocator, DType.Float32, rows * n);
            using var mix2 = new Tensor(allocator, DType.Float32, rows * n);
            Tensor Q(long values) => new Tensor(allocator, DType.UInt8, values / 32 * CudaKernels.Q81BlockBytes);
            using Tensor xnq = Q(rows * width), xnRef = Q(rows * width), siluq = Q(rows * lr), siluRef = Q(rows * lr), mixq = Q(rows * n), mixRef = Q(rows * n);
            Upload(x, gamma, lo, gates, xn, xn2, silu, silu2, mix, mix2, xnq, xnRef, siluq, siluRef, mixq, mixRef);
            IntPtr stream = allocator.Stream.Handle;
            allocator.Context.MakeCurrent();
            kernels.LaunchHcNorm(P(x), P(gamma), P(xn), rows, n, hc, 1e-6f, stream, P(xnq));
            kernels.LaunchHcNorm(P(x), P(gamma), P(xn2), rows, n, hc, 1e-6f, stream);
            kernels.LaunchScaleSilu(P(lo), P(silu), rows * lr, 1f / hc, stream, P(siluq));
            kernels.LaunchScaleSilu(P(lo), P(silu2), rows * lr, 1f / hc, stream);
            kernels.LaunchHcGateMix(P(xn), P(gates), P(mix), rows, n, hc, stream, P(mixq));
            kernels.LaunchHcGateMix(P(xn), P(gates), P(mix2), rows, n, hc, stream);
            CudaQuantizedOps.QuantizeRowsQ81(allocator, P(xn2), P(xnRef), width, rows);
            CudaQuantizedOps.QuantizeRowsQ81(allocator, P(silu2), P(siluRef), lr, rows);
            CudaQuantizedOps.QuantizeRowsQ81(allocator, P(mix2), P(mixRef), n, rows);
            allocator.Stream.Synchronize();
            Written(xn, xn2, silu, silu2, mix, mix2, xnq, xnRef, siluq, siluRef, mixq, mixRef);
            void Same(Tensor a, Tensor b, string what)
            {
                byte[] x1 = new byte[a.ElementCount()], x2 = new byte[b.ElementCount()];
                a.CopyToArray(x1);
                b.CopyToArray(x2);
                Assert.True(x2.Count(v => v != 0) > x2.Length / 4, $"{what}: the reference read back as zeros");
                Assert.True(x1.AsSpan().SequenceEqual(x2), $"{what} differs ({rows} rows, {hc}x{n}, rank {lr})");
            }
            void SameF(Tensor a, Tensor b, string what)
            {
                float[] f1 = a.GetElementsAsFloat((int)a.ElementCount()), f2 = b.GetElementsAsFloat((int)b.ElementCount());
                Assert.True(f1.Select(BitConverter.SingleToInt32Bits).SequenceEqual(f2.Select(BitConverter.SingleToInt32Bits)), $"{what} values differ");
            }
            SameF(xn, xn2, "normed streams");
            SameF(silu, silu2, "bottleneck");
            SameF(mix, mix2, "mix");
            Same(xnq, xnRef, "normed streams' q8_1");
            Same(siluq, siluRef, "bottleneck's q8_1");
            Same(mixq, mixRef, "mix's q8_1");
        }
    }
    /// <summary>The injection logits folded into the norm and the mix: a double reference's values, and each row of a
    /// window the same bits as that row read alone (a verify window's rows are its decode steps').</summary>
    [CudaTheory]
    [InlineData(4, 4, 2560)]
    [InlineData(3, 2, 512)]
    [InlineData(20, 8, 256)]   // a prompt chunk's narrow blocks
    public void FoldedInjectionMatchesTheReferenceAndEachRowAlone(int rows, int hc, int n)
    {
        using var allocator = new CudaAllocator();
        allocator.Context.MakeCurrent();
        Assert.True(CudaGraphKernels.TryCreate(out CudaGraphKernels kernels, out string reason), reason);
        using (kernels)
        {
            var rng = new Random(31);
            float[] R(int count, double scale) => Enumerable.Range(0, count).Select(_ => (float)((rng.NextDouble() * 2 - 1) * scale)).ToArray();
            int width = hc * n;
            float[] xs = R(rows * width, 3), gs = R(width, 1.5), ws = R(hc * width, 0.05);
            using var x = Tensor.FromArray(allocator, xs);
            using var gamma = Tensor.FromArray(allocator, gs);
            using var injW = Tensor.FromArray(allocator, ws);
            using var gates = Tensor.FromArray(allocator, R(rows * width, 5));
            using var xn = new Tensor(allocator, DType.Float32, rows * width);
            using var part = new Tensor(allocator, DType.Float32, rows * hc * hc);
            using var mix = new Tensor(allocator, DType.Float32, rows * n);
            using var inject = new Tensor(allocator, DType.Float32, rows * hc);
            Upload(x, gamma, injW, gates, xn, part, mix, inject);
            IntPtr stream = allocator.Stream.Handle;
            kernels.LaunchHcNorm(P(x), P(gamma), P(xn), rows, n, hc, 1e-6f, stream, IntPtr.Zero, P(injW), P(part));
            kernels.LaunchHcGateMix(P(xn), P(gates), P(mix), rows, n, hc, stream, IntPtr.Zero, P(part), P(inject));
            allocator.Stream.Synchronize();
            Written(xn, part, mix, inject);
            float[] got = inject.GetElementsAsFloat(rows * hc);
            for (int r = 0; r < rows; r++)
            {
                for (int o = 0; o < hc; o++)
                {
                    double want = 0, mag = 0;
                    for (int c = 0; c < hc; c++)
                    {
                        double ss = 0;
                        for (int i = 0; i < n; i++) ss += (double)xs[(r * hc + c) * n + i] * xs[(r * hc + c) * n + i];
                        double scale = 1 / Math.Sqrt(ss / n + 1e-6);
                        for (int i = 0; i < n; i++)
                        {
                            double v = xs[(r * hc + c) * n + i] * scale * gs[c * n + i] * ws[(long)o * width + c * n + i];
                            want += v;
                            mag += Math.Abs(v);
                        }
                    }
                    Assert.True(Math.Abs(got[r * hc + o] - want) <= 1e-5 * mag + 1e-6, $"row {r} output {o}: {got[r * hc + o]} against {want}");
                }
                // The row alone.
                using var x1 = Tensor.FromArray(allocator, xs.Skip(r * width).Take(width).ToArray());
                using var g1 = Tensor.FromArray(allocator, gates.GetElementsAsFloat(rows * width).Skip(r * width).Take(width).ToArray());
                using var xn1 = new Tensor(allocator, DType.Float32, width);
                using var part1 = new Tensor(allocator, DType.Float32, hc * hc);
                using var mix1 = new Tensor(allocator, DType.Float32, n);
                using var inject1 = new Tensor(allocator, DType.Float32, hc);
                Upload(x1, g1, xn1, part1, mix1, inject1);
                kernels.LaunchHcNorm(P(x1), P(gamma), P(xn1), 1, n, hc, 1e-6f, stream, IntPtr.Zero, P(injW), P(part1));
                kernels.LaunchHcGateMix(P(xn1), P(g1), P(mix1), 1, n, hc, stream, IntPtr.Zero, P(part1), P(inject1));
                allocator.Stream.Synchronize();
                Written(xn1, part1, mix1, inject1);
                float[] alone = inject1.GetElementsAsFloat(hc);
                if (rows * hc < 64)   // the norm's block width is the same for the row alone
                    Assert.Equal(alone.Select(BitConverter.SingleToInt32Bits), got.Skip(r * hc).Take(hc).Select(BitConverter.SingleToInt32Bits));
            }
        }
    }
    /// <summary>The decode-size norm in two passes over slices of 256 values (LaunchHcPartials, LaunchHcApply): the normed
    /// streams and the injection logits against a double reference, the q8_1 copy byte for byte the row quantizer's, and each
    /// row of a window the same bits as that row alone.</summary>
    [CudaTheory]
    [InlineData(4, 4, 2560, true)]
    [InlineData(1, 4, 256, true)]
    [InlineData(32, 8, 512, true)]
    [InlineData(7, 2, 1024, false)]
    public void SplitNormMatchesTheReferenceAndEachRowAlone(int rows, int hc, int n, bool withInjection)
    {
        using var allocator = new CudaAllocator();
        allocator.Context.MakeCurrent();
        Assert.True(CudaGraphKernels.TryCreate(out CudaGraphKernels kernels, out string reason), reason);
        using (kernels)
        {
            var rng = new Random(37 + rows);
            float[] R(int count, double scale) => Enumerable.Range(0, count).Select(_ => (float)((rng.NextDouble() * 2 - 1) * scale)).ToArray();
            int width = hc * n, slices = width / CudaGraphKernels.HcSlice;
            float[] xs = R(rows * width, 3), gs = R(width, 1.5), ws = R(hc * width, 0.05);
            using var gamma = Tensor.FromArray(allocator, gs);
            using var injW = Tensor.FromArray(allocator, ws);
            IntPtr stream = allocator.Stream.Handle;
            (float[] Xn, byte[] Xnq, float[] Inject, byte[] Reference) Run(float[] input, int count)
            {
                using var x = Tensor.FromArray(allocator, input);
                using var partials = new Tensor(allocator, DType.Float32, count * slices * CudaGraphKernels.HcPartialFloats);
                using var xn = new Tensor(allocator, DType.Float32, count * width);
                using var xnq = new Tensor(allocator, DType.UInt8, count * width / 32 * CudaKernels.Q81BlockBytes);
                using var xnRef = new Tensor(allocator, DType.UInt8, count * width / 32 * CudaKernels.Q81BlockBytes);
                using var inject = new Tensor(allocator, DType.Float32, count * hc);
                Upload(x, gamma, injW, partials, xn, xnq, xnRef, inject);
                IntPtr w = withInjection ? P(injW) : IntPtr.Zero;
                kernels.LaunchHcPartials(P(x), P(gamma), w, P(partials), count, n, hc, stream);
                kernels.LaunchHcApply(P(x), P(gamma), P(partials), P(xn), P(xnq), withInjection ? P(inject) : IntPtr.Zero, count, n, hc, 1e-6f, stream);
                CudaQuantizedOps.QuantizeRowsQ81(allocator, P(xn), P(xnRef), width, count);
                allocator.Stream.Synchronize();
                Written(partials, xn, xnq, xnRef, inject);
                byte[] q = new byte[xnq.ElementCount()], qr = new byte[xnRef.ElementCount()];
                xnq.CopyToArray(q);
                xnRef.CopyToArray(qr);
                return (xn.GetElementsAsFloat(count * width), q, withInjection ? inject.GetElementsAsFloat(count * hc) : null, qr);
            }
            var all = Run(xs, rows);
            Assert.True(all.Xnq.AsSpan().SequenceEqual(all.Reference), $"the q8_1 copy differs from the row quantizer's ({rows} rows, {hc}x{n})");
            for (int r = 0; r < rows; r++)
            {
                for (int c = 0; c < hc; c++)
                {
                    double ss = 0;
                    for (int i = 0; i < n; i++) ss += (double)xs[(r * hc + c) * n + i] * xs[(r * hc + c) * n + i];
                    double scale = 1 / Math.Sqrt(ss / n + 1e-6);
                    for (int i = 0; i < n; i++)
                    {
                        double want = xs[(r * hc + c) * n + i] * scale * gs[c * n + i];
                        float got = all.Xn[(r * hc + c) * n + i];
                        Assert.True(Math.Abs(got - want) <= 1e-5 * Math.Abs(want) + 1e-6, $"row {r} stream {c} value {i}: {got} against {want}");
                    }
                }
                if (withInjection)
                    for (int o = 0; o < hc; o++)
                    {
                        double want = 0, mag = 0;
                        for (int c = 0; c < hc; c++)
                        {
                            double ss = 0;
                            for (int i = 0; i < n; i++) ss += (double)xs[(r * hc + c) * n + i] * xs[(r * hc + c) * n + i];
                            double scale = 1 / Math.Sqrt(ss / n + 1e-6);
                            for (int i = 0; i < n; i++)
                            {
                                double v = xs[(r * hc + c) * n + i] * scale * gs[c * n + i] * ws[(long)o * width + c * n + i];
                                want += v;
                                mag += Math.Abs(v);
                            }
                        }
                        Assert.True(Math.Abs(all.Inject[r * hc + o] - want) <= 1e-5 * mag + 1e-6, $"row {r} output {o}: {all.Inject[r * hc + o]} against {want}");
                    }
                var alone = Run(xs.Skip(r * width).Take(width).ToArray(), 1);
                Assert.Equal(alone.Xn.Select(BitConverter.SingleToInt32Bits), all.Xn.Skip(r * width).Take(width).Select(BitConverter.SingleToInt32Bits));
                if (withInjection)
                    Assert.Equal(alone.Inject.Select(BitConverter.SingleToInt32Bits), all.Inject.Skip(r * hc).Take(hc).Select(BitConverter.SingleToInt32Bits));
            }
        }
    }

    /// <summary>The gated norm with a sigmoid gate (Qwen3.8's delta-net output) against a double reference.</summary>
    [CudaFact]
    public void SigmoidGatedNormMatchesTheReference()
    {
        using var allocator = new CudaAllocator();
        allocator.Context.MakeCurrent();
        Assert.True(CudaGraphKernels.TryCreate(out CudaGraphKernels kernels, out string reason), reason);
        using (kernels)
        {
            const int d = 128, heads = 3, tokens = 4, rows = heads * tokens;
            var rng = new Random(43);
            float[] R(int count, double scale) => Enumerable.Range(0, count).Select(_ => (float)((rng.NextDouble() * 2 - 1) * scale)).ToArray();
            float[] ys = R(rows * d, 2), zs = R(rows * d, 3), wsN = R(d, 1);
            using var y = Tensor.FromArray(allocator, ys);
            using var z = Tensor.FromArray(allocator, zs);
            using var w = Tensor.FromArray(allocator, wsN);
            using var output = new Tensor(allocator, DType.Float32, rows * d);
            Upload(y, z, w, output);
            kernels.LaunchGatedRmsNorm(P(y), P(z), P(w), P(output), rows, d, 1e-6f, heads, (long)heads * d, allocator.Stream.Handle, sigmoidGate: true);
            allocator.Stream.Synchronize();
            Written(output);
            float[] got = output.GetElementsAsFloat(rows * d);
            for (int r = 0; r < rows; r++)
            {
                double ss = 0;
                for (int i = 0; i < d; i++) ss += (double)ys[r * d + i] * ys[r * d + i];
                double scale = 1 / Math.Sqrt(ss / d + 1e-6);
                for (int i = 0; i < d; i++)
                {
                    double zz = zs[r * d + i];
                    double want = ys[r * d + i] * scale * wsN[i] / (1 + Math.Exp(-zz));
                    Assert.True(Math.Abs(got[r * d + i] - want) <= 1e-5 * Math.Abs(want) + 1e-6, $"row {r} value {i}: {got[r * d + i]} against {want}");
                }
            }
        }
    }

    /// <summary>A scatter fused with the next read's first pass (ts_graph_hc_post_partials) writes the streams and the
    /// partials the scatter and the first pass write apart, bit for bit.</summary>
    [CudaTheory]
    [InlineData(4, 4, 2560, true)]
    [InlineData(1, 4, 256, false)]
    [InlineData(8, 8, 512, true)]
    public void ScatterWithTheNextFirstPassMatchesThemApart(int rows, int hc, int n, bool withInjection)
    {
        using var allocator = new CudaAllocator();
        allocator.Context.MakeCurrent();
        Assert.True(CudaGraphKernels.TryCreate(out CudaGraphKernels kernels, out string reason), reason);
        using (kernels)
        {
            var rng = new Random(91 + rows);
            float[] R(int count, double scale) => Enumerable.Range(0, count).Select(_ => (float)((rng.NextDouble() * 2 - 1) * scale)).ToArray();
            int width = hc * n, slices = width / CudaGraphKernels.HcSlice;
            using var streams = Tensor.FromArray(allocator, R(rows * width, 3));
            using var block = Tensor.FromArray(allocator, R(rows * n, 2));
            using var inject = Tensor.FromArray(allocator, R(rows * hc, 4));
            using var gamma = Tensor.FromArray(allocator, R(width, 1.5));
            using var injW = Tensor.FromArray(allocator, R(hc * width, 0.05));
            using var apart = new Tensor(allocator, DType.Float32, rows * width);
            using var fused = new Tensor(allocator, DType.Float32, rows * width);
            using var partialsApart = new Tensor(allocator, DType.Float32, rows * slices * CudaGraphKernels.HcPartialFloats);
            using var partialsFused = new Tensor(allocator, DType.Float32, rows * slices * CudaGraphKernels.HcPartialFloats);
            Upload(streams, block, inject, gamma, injW, apart, fused, partialsApart, partialsFused);
            IntPtr stream = allocator.Stream.Handle, w = withInjection ? P(injW) : IntPtr.Zero;
            kernels.LaunchHcScatter(P(streams), P(block), P(inject), P(apart), rows, n, hc, stream);
            kernels.LaunchHcPartials(P(apart), P(gamma), w, P(partialsApart), rows, n, hc, stream);
            kernels.LaunchHcPostPartials(P(streams), P(block), P(inject), P(fused), P(gamma), w, P(partialsFused), rows, n, hc, stream);
            allocator.Stream.Synchronize();
            Written(apart, fused, partialsApart, partialsFused);
            Assert.Equal(apart.GetElementsAsFloat(rows * width), fused.GetElementsAsFloat(rows * width));
            int count = rows * slices * CudaGraphKernels.HcPartialFloats;
            Assert.Equal(partialsApart.GetElementsAsFloat(count), partialsFused.GetElementsAsFloat(count));
        }
    }

    /// <summary>The gated norm's and the sigmoid gate's q8_1 copies (written beside their values for the out-projection) are
    /// the row quantizer's bytes, and the values are those the kernels write without them.</summary>
    [CudaTheory]
    [InlineData(1, 32, 128)]
    [InlineData(4, 32, 128)]
    [InlineData(8, 16, 256)]
    public void NormAndGateQuantizedCopiesAreTheQuantizers(int tokens, int heads, int d)
    {
        using var allocator = new CudaAllocator();
        allocator.Context.MakeCurrent();
        Assert.True(CudaGraphKernels.TryCreate(out CudaGraphKernels kernels, out string reason), reason);
        using (kernels)
        {
            var rng = new Random(17 + tokens);
            float[] R(int count, double scale) => Enumerable.Range(0, count).Select(_ => (float)((rng.NextDouble() * 2 - 1) * scale)).ToArray();
            int n = tokens * heads * d, blocks = n / 32 * CudaKernels.Q81BlockBytes;
            using var y = Tensor.FromArray(allocator, R(n, 2));
            using var z = Tensor.FromArray(allocator, R(n, 3));
            using var w = Tensor.FromArray(allocator, R(d, 1));
            using var plain = new Tensor(allocator, DType.Float32, n);
            using var withQ = new Tensor(allocator, DType.Float32, n);
            using var q = new Tensor(allocator, DType.UInt8, blocks);
            using var reference = new Tensor(allocator, DType.UInt8, blocks);
            Upload(y, z, w, plain, withQ, q, reference);
            IntPtr stream = allocator.Stream.Handle;
            void Check(string what)
            {
                CudaQuantizedOps.QuantizeRowsQ81(allocator, P(withQ), P(reference), heads * d, tokens);
                allocator.Stream.Synchronize();
                Written(plain, withQ, q, reference);
                Assert.Equal(plain.GetElementsAsFloat(n), withQ.GetElementsAsFloat(n));
                byte[] a = new byte[blocks], b = new byte[blocks];
                q.CopyToArray(a);
                reference.CopyToArray(b);
                Assert.True(a.AsSpan().SequenceEqual(b), $"{what}: the q8_1 copy differs from the row quantizer's");
            }
            kernels.LaunchGatedRmsNorm(P(y), P(z), P(w), P(plain), tokens * heads, d, 1e-6f, heads, (long)heads * d, stream);
            kernels.LaunchGatedRmsNorm(P(y), P(z), P(w), P(withQ), tokens * heads, d, 1e-6f, heads, (long)heads * d, stream, outputQ81: P(q));
            Check("gated norm");
            kernels.LaunchSigmoidGate(P(y), P(z), P(plain), tokens, heads, d, (long)heads * d, d, stream);
            kernels.LaunchSigmoidGate(P(y), P(z), P(withQ), tokens, heads, d, (long)heads * d, d, stream, P(q));
            Check("sigmoid gate");
        }
    }

    /// <summary>The strided rotary (a fused projection's q block read where it lies) writes the rows ts_rope_ex_f32 writes over
    /// a contiguous copy, bit for bit: NeoX and normal pairs, a partial rotary dimension.</summary>
    [CudaTheory]
    [InlineData(1, 16, 256, 256, 2)]
    [InlineData(4, 16, 256, 64, 2)]
    [InlineData(4, 2, 128, 128, 0)]
    public void StridedRopeMatchesTheContiguousOne(int tokens, int heads, int d, int ropeDim, int mode)
    {
        using var allocator = new CudaAllocator();
        allocator.Context.MakeCurrent();
        Assert.True(CudaGraphKernels.TryCreate(out CudaGraphKernels kernels, out string reason), reason);
        using (kernels)
        {
            var rng = new Random(5 + tokens * heads);
            int pitch = 2 * heads * d, rows = tokens * heads;   // q | gate per token
            float[] packed = Enumerable.Range(0, tokens * pitch).Select(_ => (float)(rng.NextDouble() * 4 - 2)).ToArray();
            var contiguous = new float[rows * d];
            for (int t = 0; t < tokens; t++)
                for (int h = 0; h < heads; h++)
                    Array.Copy(packed, t * pitch + h * d, contiguous, (t * heads + h) * d, d);
            int[] pos = Enumerable.Range(0, rows).Select(r => 100 + r / heads * 7).ToArray();
            using var src = Tensor.FromArray(allocator, packed);
            using var flat = Tensor.FromArray(allocator, contiguous);
            using var positions = Tensor.FromArray(allocator, pos);
            using var expected = new Tensor(allocator, DType.Float32, rows * d);
            using var got = new Tensor(allocator, DType.Float32, rows * d);
            Upload(src, flat, positions, expected, got);
            IntPtr stream = allocator.Stream.Handle;
            allocator.Kernels.LaunchRoPEExF32(P(flat), P(positions), P(expected), rows, d, ropeDim, mode, 1, 32768, 1e7f, 1f, 0f, 1f, 32f, 1f, 0, stream);
            kernels.LaunchRopeExStrided(P(src), pitch, d, heads, P(positions), P(got), rows, d, ropeDim, mode, true, 32768, 1e7f, 1f, 0f, 1f, 32f, 1f, stream);
            allocator.Stream.Synchronize();
            Written(expected, got);
            Assert.Equal(expected.GetElementsAsFloat(rows * d), got.GetElementsAsFloat(rows * d));
        }
    }

    /// <summary>The fused shared-expert gate writes moe + shared * sigmoid(x . w) as the four separate launches it replaces
    /// write it (the small matvec, the sigmoid, the broadcast product, the add), bit for bit.</summary>
    [CudaTheory]
    [InlineData(1, 2048)]
    [InlineData(4, 4096)]
    [InlineData(13, 1024)]
    public void SharedExpertGateMatchesItsOps(int rows, int n)
    {
        using var allocator = new CudaAllocator();
        allocator.Context.MakeCurrent();
        Assert.True(CudaGraphKernels.TryCreate(out CudaGraphKernels kernels, out string reason), reason);
        using (kernels)
        {
            var rng = new Random(29 + rows);
            float[] R(int count, double scale) => Enumerable.Range(0, count).Select(_ => (float)((rng.NextDouble() * 2 - 1) * scale)).ToArray();
            using var moe = Tensor.FromArray(allocator, R(rows * n, 2));
            using var shared = Tensor.FromArray(allocator, R(rows * n, 2));
            using var x = Tensor.FromArray(allocator, R(rows * n, 1));
            using var w = Tensor.FromArray(allocator, R(n, 0.05));
            using var g = new Tensor(allocator, DType.Float32, rows, 1);
            using var fused = new Tensor(allocator, DType.Float32, rows * n);
            Upload(moe, shared, x, w, g, fused);
            IntPtr stream = allocator.Stream.Handle;
            kernels.LaunchSmallMatvec(P(x), P(w), P(g), rows, n, 1, n, stream);
            kernels.LaunchSharedGateAdd(P(moe), P(shared), P(x), P(w), P(fused), rows, n, n, n, stream);
            allocator.Stream.Synchronize();
            Written(g, fused);
            using Tensor s = Ops.Sigmoid(null, g);
            using var scaled = new Tensor(allocator, DType.Float32, rows, n);
            Upload(scaled);
            using (Tensor sh = shared.View(rows, n))
                kernels.LaunchBinaryBroadcast(P(sh), P(s), P(scaled), new long[] { n, rows, 1, 1 }, new long[] { 1, rows, 1, 1 }, 2, stream);
            allocator.Stream.Synchronize();
            Written(scaled);
            using Tensor m2 = moe.View(rows, n);
            using Tensor sum = Ops.Add(null, m2, scaled);
            Assert.Equal(sum.GetElementsAsFloat(rows * n), fused.GetElementsAsFloat(rows * n));
        }
    }
}
