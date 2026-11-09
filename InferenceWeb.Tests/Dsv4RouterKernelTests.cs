// Copyright (c) Zhongkai Fu. All rights reserved.
// Licensed under the BSD-3-Clause license in the repository root.
using System;
using System.Collections.Generic;
using System.Security.Cryptography;
using TensorSharp;
using TensorSharp.Cuda;
using TensorSharp.Cuda.Interop;
using Xunit;
using Xunit.Abstractions;
using static InferenceWeb.Tests.Dsv4ExpertKernelTests;

namespace InferenceWeb.Tests;

/// <summary>
/// The direct engines' MoE router select: the keyed warp select and the float-argmax one against a host reference and
/// against each other bit for bit (on rows built to hit every rule of the order), fingerprinted so two builds can be
/// compared, and a graph-captured timing of the router (BF16 GEMV, then select) at Flash Next's shapes (512 experts over
/// 2560, top 10).
/// </summary>
public sealed class Dsv4RouterKernelTests
{
    private readonly ITestOutputHelper _output;

    public Dsv4RouterKernelTests(ITestOutputHelper output) => _output = output;

    private static IntPtr Ptr(Tensor t) => Dsv4CudaEngine.Ptr(t);

    private static ushort[] RandomBf16(Random rnd, int n)
    {
        var w = new ushort[n];
        for (int i = 0; i < n; i++)
        {
            float v = (float)((rnd.NextDouble() * 2 - 1) * 0.05);
            w[i] = (ushort)(BitConverter.SingleToUInt32Bits(v) >> 16);
        }
        return w;
    }

    private static byte[] Bytes(ushort[] w)
    {
        var b = new byte[w.Length * 2];
        Buffer.BlockCopy(w, 0, b, 0, b.Length);
        return b;
    }

    /// <summary>The wide-output GEMVs (the routers' block kernels, BF16 and F32) at 1..16 rows: against a double reference
    /// of the widened weights, the BF16 and F32 forms bit-equal, and fingerprinted so two builds can be compared.</summary>
    [CudaTheory]
    [InlineData(2560, 512)]
    [InlineData(4096, 256)]
    public void WideGemvs_Fingerprint(int inDim, int outDim)
    {
        using var allocator = new CudaAllocator(0);
        allocator.Context.MakeCurrent();
        using var dk = Dsv4Kernels.Create();
        var rnd = new Random(inDim * 3 + outDim);
        const int maxRows = Dsv4Kernels.GemvMaxRows;
        var x = new float[maxRows * inDim];
        for (int i = 0; i < x.Length; i++)
            x[i] = (float)((rnd.NextDouble() * 2 - 1) * (i % 89 == 5 ? 6 : 1));
        ushort[] w = RandomBf16(rnd, inDim * outDim);
        var wF = new float[w.Length];
        for (int i = 0; i < w.Length; i++)
            wF[i] = BitConverter.UInt32BitsToSingle((uint)w[i] << 16);
        using Tensor xT = Up(allocator, x);
        using Tensor wT = Up(allocator, Bytes(w));
        using Tensor wFT = Up(allocator, wF);
        using Tensor y = new Tensor(allocator, DType.Float32, maxRows, outDim);
        using Tensor scratch = new Tensor(allocator, DType.Float32, Dsv4Kernels.GemvScratchFloats);
        IntPtr stream = allocator.Stream.Handle;
        string line = $"fingerprint gemv {inDim} -> {outDim}:";
        for (int rows = 1; rows <= maxRows; rows++)
        {
            dk.GemvBf16(Ptr(xT), Ptr(wT), Ptr(y), inDim, outDim, rows, stream);
            allocator.Stream.Synchronize();
            float[] got = Down(y, rows * outDim);
            dk.Gemv(Ptr(xT), Ptr(wFT), Ptr(y), inDim, outDim, rows, stream, Ptr(scratch));
            allocator.Stream.Synchronize();
            float[] gotF = Down(y, rows * outDim);
            double worst = 0;
            for (int r = 0; r < rows; r++)
                for (int o = 0; o < outDim; o++)
                {
                    int i = r * outDim + o;
                    Assert.True(BitConverter.SingleToInt32Bits(got[i]) == BitConverter.SingleToInt32Bits(gotF[i]),
                        $"{rows} rows, output {i}: BF16 {got[i]:R}, F32 {gotF[i]:R}");
                    double s = 0, mag = 0;
                    for (int k = 0; k < inDim; k++)
                    {
                        double p = (double)x[r * inDim + k] * wF[(long)o * inDim + k];
                        s += p;
                        mag += Math.Abs(p);
                    }
                    worst = Math.Max(worst, Math.Abs(got[i] - s) / Math.Max(mag, 1e-30));
                }
            Assert.True(worst < 1e-5, $"{rows} rows: error {worst:E2} of the absolute sum");
            var bytes = new byte[got.Length * 4];
            Buffer.BlockCopy(got, 0, bytes, 0, bytes.Length);
            line += $" {rows}r {Convert.ToHexString(SHA256.HashData(bytes))[..12]}";
        }
        _output.WriteLine(line);
    }

    // Router logits for nt rows: random, with rows whose top scores tie exactly (the lower expert first) and rows with
    // -inf entries (a zero score, still a candidate).
    private static float[] Logits(int nt, int nExpert, int seed)
    {
        var rnd = new Random(seed);
        var l = new float[nt * nExpert];
        for (int t = 0; t < nt; t++)
            for (int e = 0; e < nExpert; e++)
            {
                float v = (float)(rnd.NextDouble() * 8 - 4);
                if (t % 3 == 1 && e % 7 == 0)
                    v = 4.5f;   // ties among the top picks
                if (t % 4 == 2 && e % 5 == 0)
                    v = float.NegativeInfinity;
                l[t * nExpert + e] = v;
            }
        return l;
    }

    // The host's selection: softmax (gating 2) or sqrt(softplus) (gating 0) scores, plus the bias for the ranking, the
    // higher score first and the lower expert on ties; the weights are the unbiased scores, normalized by their sum.
    private static (int[] Sel, double[] W) ReferenceSelect(float[] logits, float[] bias, int t, int nExpert, int nUsed, int gating)
    {
        var raw = new double[nExpert];
        if (gating == Dsv4Kernels.RouterSoftmax)
        {
            double mx = double.NegativeInfinity, sum = 0;
            for (int e = 0; e < nExpert; e++)
                mx = Math.Max(mx, logits[t * nExpert + e]);
            for (int e = 0; e < nExpert; e++)
                sum += raw[e] = Math.Exp(logits[t * nExpert + e] - mx);
            for (int e = 0; e < nExpert; e++)
                raw[e] /= sum;
        }
        else
            for (int e = 0; e < nExpert; e++)
            {
                double l = logits[t * nExpert + e];
                raw[e] = Math.Sqrt(l > 20 ? l : Math.Log(1 + Math.Exp(l)));
            }
        var order = new List<int>();
        for (int e = 0; e < nExpert; e++)
            order.Add(e);
        double Score(int e) => (float)raw[e] + (bias != null ? bias[e] : 0f);
        order.Sort((a, b) => Score(a) != Score(b) ? Score(b).CompareTo(Score(a)) : a.CompareTo(b));
        var sel = order.GetRange(0, nUsed).ToArray();
        var w = new double[nUsed];
        double ws = 0;
        for (int k = 0; k < nUsed; k++)
            ws += raw[sel[k]];
        for (int k = 0; k < nUsed; k++)
            w[k] = raw[sel[k]] / ws;
        return (sel, w);
    }

    [CudaTheory]
    [InlineData(512, 10, 2, false)]
    [InlineData(512, 16, 2, false)]
    [InlineData(512, 8, 0, true)]
    [InlineData(256, 8, 0, true)]
    [InlineData(384, 6, 2, false)]
    public void MoeSelect_MatchesTheHost_Fingerprint(int nExpert, int nUsed, int gating, bool withBias)
    {
        using var allocator = new CudaAllocator(0);
        allocator.Context.MakeCurrent();
        using var dk = Dsv4Kernels.Create();
        const int nt = 8;
        float[] logits = Logits(nt, nExpert, nExpert * 31 + nUsed + gating);
        float[] bias = null;
        if (withBias)
        {
            var rb = new Random(nExpert);
            bias = new float[nExpert];
            for (int e = 0; e < nExpert; e++)
                bias[e] = (float)(rb.NextDouble() * 0.5);
        }
        using Tensor lT = Up(allocator, logits);
        using Tensor bT = bias != null ? Up(allocator, bias) : null;
        using Tensor sel = new Tensor(allocator, DType.Int32, nt * nUsed);
        using Tensor wOut = new Tensor(allocator, DType.Float32, nt * nUsed);
        IntPtr stream = allocator.Stream.Handle;
        foreach (bool keys in new[] { false, true })
        {
            string line = $"fingerprint select {(keys ? "keys" : "argmax")} {nExpert} experts top {nUsed} gating {gating}{(withBias ? " bias" : "")}:";
            for (int rows = 1; rows <= nt; rows++)
            {
                dk.MoeSelect(lT, bT != null ? Ptr(bT) : IntPtr.Zero, IntPtr.Zero, null, sel, wOut, nExpert, nUsed, 1, 1.0f, rows,
                    stream, gating, keys);
                allocator.Stream.Synchronize();
                ((CudaStorage)sel.Storage).MarkDeviceModified();
                int[] s = sel.GetElementsAsInt(rows * nUsed);
                float[] w = Down(wOut, rows * nUsed);
                for (int t = 0; t < rows; t++)
                {
                    var (rs, rw) = ReferenceSelect(logits, bias, t, nExpert, nUsed, gating);
                    for (int k = 0; k < nUsed; k++)
                    {
                        Assert.True(rs[k] == s[t * nUsed + k], $"keys {keys} row {t} pick {k}: expert {s[t * nUsed + k]}, the host picked {rs[k]}");
                        Assert.True(Math.Abs(w[t * nUsed + k] - rw[k]) <= 4e-6 * rw[k] + 1e-9,
                            $"keys {keys} row {t} pick {k}: weight {w[t * nUsed + k]:R}, the host gave {rw[k]:R}");
                    }
                }
                var bytes = new byte[rows * nUsed * 8];
                Buffer.BlockCopy(s, 0, bytes, 0, rows * nUsed * 4);
                Buffer.BlockCopy(w, 0, bytes, rows * nUsed * 4, rows * nUsed * 4);
                line += $" {rows}r {Convert.ToHexString(SHA256.HashData(bytes))[..12]}";
            }
            _output.WriteLine(line);
        }
    }

    // Rows built to hit every rule of the order: exact ties, -inf, NaN, +inf, signed zeros, rows with no candidate at all.
    private static float[] HardLogits(int nt, int nExpert, int seed)
    {
        var rnd = new Random(seed);
        var l = new float[nt * nExpert];
        for (int t = 0; t < nt; t++)
        {
            int kind = t % 8;
            for (int e = 0; e < nExpert; e++)
            {
                float v = (float)(rnd.NextDouble() * 12 - 6);
                switch (kind)
                {
                    case 1: if (rnd.Next(3) == 0) v = 2.0f; break;                          // many exact ties
                    case 2: if (rnd.Next(4) == 0) v = float.NegativeInfinity; break;
                    case 3: if (rnd.Next(50) == 0) v = float.NaN; break;
                    case 4: if (rnd.Next(40) == 0) v = float.PositiveInfinity; break;
                    case 5: v = rnd.Next(2) == 0 ? 0.0f : -0.0f; break;                     // signed zeros only
                    case 6: v = float.NegativeInfinity; break;                               // no candidate
                    case 7: v = (float)Math.Round(v, 1); break;                              // coarse values, ties at every rank
                }
                l[t * nExpert + e] = v;
            }
        }
        return l;
    }

    [CudaTheory]
    [InlineData(512, 10, 2, false)]
    [InlineData(512, 16, 2, true)]
    [InlineData(512, 32, 2, false)]
    [InlineData(512, 8, 0, true)]
    [InlineData(256, 8, 0, true)]
    [InlineData(256, 8, 1, true)]
    [InlineData(300, 6, 1, false)]
    [InlineData(64, 6, 2, false)]
    public void MoeSelect_KeysAndArgmax_GiveTheSameBits(int nExpert, int nUsed, int gating, bool withBias)
    {
        using var allocator = new CudaAllocator(0);
        allocator.Context.MakeCurrent();
        using var dk = Dsv4Kernels.Create();
        const int nt = 2048;
        float[] logits = HardLogits(nt, nExpert, nExpert * 7 + nUsed * 3 + gating);
        float[] bias = null;
        if (withBias)
        {
            var rb = new Random(nExpert + 1);
            bias = new float[nExpert];
            for (int e = 0; e < nExpert; e++)
                bias[e] = e % 9 == 0 ? 0.0f : (float)Math.Round(rb.NextDouble() * 0.5, 2);
        }
        using Tensor lT = Up(allocator, logits);
        using Tensor bT = bias != null ? Up(allocator, bias) : null;
        using Tensor sel = new Tensor(allocator, DType.Int32, nt * nUsed);
        using Tensor wOut = new Tensor(allocator, DType.Float32, nt * nUsed);
        IntPtr stream = allocator.Stream.Handle;
        (int[] S, float[] W) Run(bool keys)
        {
            dk.MoeSelect(lT, bT != null ? Ptr(bT) : IntPtr.Zero, IntPtr.Zero, null, sel, wOut, nExpert, nUsed, 1, 0.75f, nt,
                stream, gating, keys);
            allocator.Stream.Synchronize();
            ((CudaStorage)sel.Storage).MarkDeviceModified();
            return (sel.GetElementsAsInt(nt * nUsed), Down(wOut, nt * nUsed));
        }
        var (s0, w0) = Run(false);
        var (s1, w1) = Run(true);
        for (int i = 0; i < s0.Length; i++)
        {
            Assert.True(s0[i] == s1[i], $"row {i / nUsed} (kind {i / nUsed % 8}) pick {i % nUsed}: keys took {s1[i]}, the argmax {s0[i]}");
            Assert.True(BitConverter.SingleToInt32Bits(w0[i]) == BitConverter.SingleToInt32Bits(w1[i]),
                $"row {i / nUsed} (kind {i / nUsed % 8}) pick {i % nUsed}: keys weighed {w1[i]:R}, the argmax {w0[i]:R}");
        }
    }

    // Microseconds a node of a graph of n launches, replayed reps times (the decode step's own setting: captured).
    private static double GraphMicros(IntPtr stream, Action<int> launch, int n, int reps)
    {
        CudaDriverApi.cuEventCreate(out IntPtr start, 0).ThrowOnError();
        CudaDriverApi.cuEventCreate(out IntPtr end, 0).ThrowOnError();
        try
        {
            CudaDriverApi.cuStreamBeginCapture(stream, CudaDriverApi.CU_STREAM_CAPTURE_MODE_RELAXED).ThrowOnError();
            for (int i = 0; i < n; i++)
                launch(i);
            CudaDriverApi.cuStreamEndCapture(stream, out IntPtr graph).ThrowOnError();
            CudaDriverApi.cuGraphInstantiateWithFlags(out IntPtr exec, graph, 0).ThrowOnError();
            try
            {
                for (int r = 0; r < 3; r++)
                    CudaDriverApi.cuGraphLaunch(exec, stream).ThrowOnError();
                CudaDriverApi.cuStreamSynchronize(stream).ThrowOnError();
                CudaDriverApi.cuEventRecord(start, stream).ThrowOnError();
                for (int r = 0; r < reps; r++)
                    CudaDriverApi.cuGraphLaunch(exec, stream).ThrowOnError();
                CudaDriverApi.cuEventRecord(end, stream).ThrowOnError();
                CudaDriverApi.cuEventSynchronize(end).ThrowOnError();
                CudaDriverApi.cuEventElapsedTime(out float ms, start, end).ThrowOnError();
                return ms * 1000.0 / ((double)n * reps);
            }
            finally
            {
                CudaDriverApi.cuGraphExecDestroy(exec);
                CudaDriverApi.cuGraphDestroy(graph);
            }
        }
        finally
        {
            CudaDriverApi.cuEventDestroy(start);
            CudaDriverApi.cuEventDestroy(end);
        }
    }

    [CudaFact]
    [Trait("Category", "Bench")]
    public void FlashNextRouter_CapturedTiming()
    {
        const int inDim = 2560, nExpert = 512, nUsed = 10, layers = 48;
        using var allocator = new CudaAllocator(0);
        allocator.Context.MakeCurrent();
        using var dk = Dsv4Kernels.Create();
        IntPtr stream = allocator.Stream.Handle;
        var rnd = new Random(11);
        // One router matrix a layer (2.5 MiB each, 120 MiB in all: past the L2, as a decode step reads them).
        var mats = new List<Tensor>();
        try
        {
            for (int l = 0; l < layers; l++)
                mats.Add(Up(allocator, Bytes(RandomBf16(rnd, inDim * nExpert))));
            var x = new float[4 * inDim];
            for (int i = 0; i < x.Length; i++)
                x[i] = (float)(rnd.NextDouble() * 2 - 1);
            using Tensor xT = Up(allocator, x);
            using Tensor y = new Tensor(allocator, DType.Float32, 4, nExpert);
            using Tensor sel = new Tensor(allocator, DType.Int32, 4 * nUsed);
            using Tensor wOut = new Tensor(allocator, DType.Float32, 4 * nUsed);
            for (int rows = 1; rows <= 4; rows++)
            {
                string line = $"router {rows}r:";
                double gus = GraphMicros(stream, i => dk.GemvBf16(Ptr(xT), Ptr(mats[i % layers]), Ptr(y), inDim, nExpert, rows,
                    stream), 4 * layers, 20);
                line += $" gemv {gus:F2} us;";
                foreach (bool keys in new[] { false, true })
                {
                    double sus = GraphMicros(stream, i => dk.MoeSelect(y, IntPtr.Zero, IntPtr.Zero, null, sel, wOut, nExpert, nUsed, 1,
                        1.0f, rows, stream, Dsv4Kernels.RouterSoftmax, keys), 4 * layers, 20);
                    double pair = GraphMicros(stream, i =>
                    {
                        dk.GemvBf16(Ptr(xT), Ptr(mats[i % layers]), Ptr(y), inDim, nExpert, rows, stream);
                        dk.MoeSelect(y, IntPtr.Zero, IntPtr.Zero, null, sel, wOut, nExpert, nUsed, 1, 1.0f, rows, stream,
                            Dsv4Kernels.RouterSoftmax, keys);
                    }, 4 * layers, 20);
                    line += $" select {(keys ? "keys" : "argmax")} {sus:F2} us, gemv + select {pair:F2} us;";
                }
                _output.WriteLine(line);
            }
        }
        finally
        {
            allocator.Synchronize();
            foreach (Tensor t in mats)
                t.Dispose();
        }
    }
}
