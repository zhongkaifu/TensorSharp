// Copyright (c) Zhongkai Fu. All rights reserved.
// https://github.com/zhongkaifu/TensorSharp
//
// This file is part of TensorSharp.
//
// TensorSharp is licensed under the BSD-3-Clause license found in the LICENSE file in the root directory of this source tree.
//
// The pure-C# decode attention kept each block's (query heads per KV head) x (keys) score
// scratch on the stack. The blocks run on CpuWorkerPool workers, whose stacks are 512 KiB, and
// running out of stack is not an exception - the runtime aborts the process. Nemotron-3.5
// Lightning (32 heads over 2 KV heads) decoding a 13,844-token prompt on --backend cuda asked a
// worker for 16 x 13,844 x 4 B = 865 KiB and died with exit 134. Past CpuScratch.StackFloatLimit
// the scratch is now rented. Every test that runs a long context does it on a thread whose stack
// is exactly a worker's, so whichever thread ends up running a block - the submitter or a pool
// worker - has a worker-sized stack: without the limit these tests abort the test host instead
// of failing. The rented results are checked against a double-precision reference, and bit for
// bit against the stack path, which is what short contexts still take. Rented buffers are
// NaN-filled while these tests run (stack scratch is zeroed, pooled arrays are not), so a kernel
// that read scratch it had not written fails the bit-for-bit comparison instead of passing by luck.
using System.Runtime.ExceptionServices;
using TensorSharp;
using TensorSharp.Cpu;
using TensorSharp.Models;
using Xunit;

namespace InferenceWeb.Tests;

public sealed unsafe class CpuAttentionStackScratchTests : IDisposable
{
    private static readonly IAllocator Alloc = new CpuAllocator(BlasEnum.DotNet);

    // xUnit runs this class's tests one at a time; other classes that rent meanwhile only see NaN
    // in slots they overwrite anyway.
    public CpuAttentionStackScratchTests() => CpuScratch.PoisonRentedBuffers = true;

    public void Dispose() => CpuScratch.PoisonRentedBuffers = false;

    [Fact]
    public void RentedScratch_IsNaNFilled_WhileTheseTestsRun()
    {
        // Leave values behind as a previous renter would, then rent the same size class again
        // (normally the same array, back from this thread's pool slot).
        float[] first = CpuScratch.RentFloats(CpuScratch.MaxStackFloats + 1);
        Array.Fill(first, 1f);
        CpuScratch.Return(first);

        float[] rented = CpuScratch.RentFloats(CpuScratch.MaxStackFloats + 1);
        try
        {
            for (int i = 0; i < rented.Length; i++)
                Assert.True(float.IsNaN(rented[i]), $"rented[{i}] = {rented[i]:R}, not NaN");
        }
        finally
        {
            CpuScratch.Return(rented);
        }
    }

    // ------------------------------------------------------------------------------------
    //  ModelBase.AttentionDecodeGrouped / AttentionDecodeGroupedF16
    // ------------------------------------------------------------------------------------

    [Fact]
    public void LightningShape_F16_OnWorkerSizedStacks_RentsTheScratch_AndMatchesReference()
    {
        // The crashing call: 32 heads / 2 KV heads / head dim 128 / 13,844 keys, F16 cache, one
        // block per KV head on the shared pool.
        var c = GroupedCase.Create(heads: 32, kvHeads: 2, headDim: 128, keys: 13_844, f16: true, seed: 1);
        Assert.True((long)c.Group * c.Keys * sizeof(float) > CpuWorkerPool.WorkerStackBytes,
            "the shape must not fit a worker's stack, or this test proves nothing");

        float[] actual = OnThread(CpuWorkerPool.WorkerStackBytes, () => c.Run(Environment.ProcessorCount));

        AssertMatchesReference(actual, c.Reference());
    }

    public static IEnumerable<object[]> LongSchedules()
    {
        // heads, kvHeads, keys, f16, processorCount (pins the F32 K-split), scratch keys per block
        yield return new object[] { 16, 1, 13_844, false, 1, 13_844 };   // F32 serial, caller's thread
        yield return new object[] { 32, 2, 13_844, false, 1, 13_844 };   // F32 one block per KV head
        yield return new object[] { 32, 2, 20_000, false, 4, 10_000 };   // F32 K-split: 2 chunks of 10,000
        yield return new object[] { 16, 1, 13_844, true, 1, 13_844 };    // F16 serial, caller's thread
    }

    [Theory]
    [MemberData(nameof(LongSchedules))]
    public void EverySchedule_OnWorkerSizedStacks_RentsTheScratch_AndMatchesReference(
        int heads, int kvHeads, int keys, bool f16, int processorCount, int scratchKeys)
    {
        var c = GroupedCase.Create(heads, kvHeads, headDim: 128, keys, f16, seed: heads + keys);
        Assert.True((long)c.Group * scratchKeys * sizeof(float) > CpuWorkerPool.WorkerStackBytes);

        float[] actual = OnThread(CpuWorkerPool.WorkerStackBytes, () => c.Run(processorCount));

        AssertMatchesReference(actual, c.Reference());
    }

    public static IEnumerable<object[]> FittingShapes()
    {
        // heads, kvHeads, headDim, keys, f16, processorCount: every schedule and both the
        // specialized group-of-4 loops and the generic ones, at sizes the stack path can hold.
        foreach (bool f16 in new[] { false, true })
        {
            yield return new object[] { 8, 2, 64, 100, f16, 1 };        // serial (too little work to fork)
            yield return new object[] { 16, 1, 128, 2_000, f16, 1 };    // serial, group 16
            yield return new object[] { 16, 4, 128, 3_000, f16, 1 };    // one block per KV head, group 4
            yield return new object[] { 32, 2, 128, 2_000, f16, 1 };    // one block per KV head, group 16 (125 KiB)
            yield return new object[] { 12, 2, 64, 1_500, f16, 1 };     // group 6
        }
        yield return new object[] { 16, 4, 128, 3_000, false, 64 };     // K-split, 5 chunks of 600
        yield return new object[] { 32, 2, 128, 2_000, false, 4 };      // K-split, 2 chunks of 1,000
    }

    [Theory]
    [MemberData(nameof(FittingShapes))]
    public void StackAndRentedScratch_GiveBitIdenticalResults(
        int heads, int kvHeads, int headDim, int keys, bool f16, int processorCount)
    {
        var c = GroupedCase.Create(heads, kvHeads, headDim, keys, f16, seed: 7 * heads + keys);

        float[] onStack, rented;
        using (CpuScratch.OverrideStackFloatLimit(int.MaxValue))
            onStack = c.Run(processorCount);
        using (CpuScratch.OverrideStackFloatLimit(0))
            rented = c.Run(processorCount);

        AssertBitIdentical(onStack, rented);
        AssertMatchesReference(rented, c.Reference());
    }

    [Theory]
    [InlineData(false)]
    [InlineData(true)]
    public void AtTheCrashingLength_StackAndRentedScratch_AreBitIdentical(bool f16)
    {
        // 865 KiB of scratch: the stack run needs a thread big enough to hold it, the rented run
        // gets a worker-sized one.
        var c = GroupedCase.Create(heads: 16, kvHeads: 1, headDim: 128, keys: 13_844, f16, seed: 3);

        float[] onStack = OnThread(64 << 20, () =>
        {
            using (CpuScratch.OverrideStackFloatLimit(int.MaxValue))
                return c.Run(1);
        });
        float[] rented = OnThread(CpuWorkerPool.WorkerStackBytes, () => c.Run(1));

        AssertBitIdentical(onStack, rented);
    }

    [Fact]
    public void RentedScratch_IsReusedAcrossCalls_NotAllocatedPerCall()
    {
        // Serial path, so the whole call runs on this thread and its allocations are countable.
        var c = GroupedCase.Create(heads: 16, kvHeads: 1, headDim: 128, keys: 13_844, f16: true, seed: 5);
        Assert.True(c.Group * c.Keys > CpuScratch.StackFloatLimit);
        float[] result = new float[c.Heads * c.HeadDim];
        c.Run(1, result);
        c.Run(1, result);

        long before = GC.GetAllocatedBytesForCurrentThread();
        c.Run(1, result);
        long allocated = GC.GetAllocatedBytesForCurrentThread() - before;

        // The scratch is 865 KiB; anything near that means it was allocated, not reused.
        Assert.True(allocated < 16 * 1024, $"a decode call allocated {allocated} bytes");
    }

    [Fact]
    public void DefaultLimit_KeepsShortContextsOnTheStack_AndLeavesMostOfAWorkerStack()
    {
        Assert.Equal(CpuScratch.MaxStackFloats, CpuScratch.StackFloatLimit);
        Assert.True(CpuScratch.MaxStackBytes <= CpuWorkerPool.WorkerStackBytes / 4);
        // Unchanged fast path for the contexts the limit was sized for.
        Assert.True(4 * 8_192 <= CpuScratch.StackFloatLimit);    // group 4 to 8K keys
        Assert.True(16 * 2_048 <= CpuScratch.StackFloatLimit);   // group 16 to 2K keys
    }

    [Fact]
    public void LimitOverride_AppliesToTheOverridingThreadOnly_AndRestores()
    {
        using (CpuScratch.OverrideStackFloatLimit(0))
        {
            Assert.Equal(0, CpuScratch.StackFloatLimit);
            Assert.Equal(CpuScratch.MaxStackFloats, OnThread(256 * 1024, () => CpuScratch.StackFloatLimit));
            using (CpuScratch.OverrideStackFloatLimit(int.MaxValue))
                Assert.Equal(int.MaxValue - 1, CpuScratch.StackFloatLimit);
            Assert.Equal(0, CpuScratch.StackFloatLimit);
        }
        Assert.Equal(CpuScratch.MaxStackFloats, CpuScratch.StackFloatLimit);
    }

    // ------------------------------------------------------------------------------------
    //  The per-model host attention kernels (one score row per head over the attended range)
    // ------------------------------------------------------------------------------------

    public static IEnumerable<object[]> RowKernels()
    {
        foreach (string k in new[] { "gpt-oss", "gpt-oss-tp" })
        {
            yield return new object[] { k, false };
            if (k != "gpt-oss-tp")   // the TP kernel reads an F32 cache only
                yield return new object[] { k, true };
        }
    }

    [Theory]
    [MemberData(nameof(RowKernels))]
    public void ModelRowKernels_AtALongContext_OnWorkerSizedStacks_MatchReference(string kernel, bool f16)
    {
        // 400,000 keys = 1.6 MB of scores per head: past a worker's stack and a ThreadPool
        // thread's, so whichever thread runs a head would have overflowed.
        var c = RowCase.Create(kernel, keys: 400_000, f16, seed: 11);
        try
        {
            OnThread(CpuWorkerPool.WorkerStackBytes, () => { c.Run(); return 0; });
            AssertMatchesReference(c.Result(), c.Reference());
        }
        finally
        {
            c.Dispose();
        }
    }

    [Theory]
    [MemberData(nameof(RowKernels))]
    public void ModelRowKernels_StackAndRentedScratch_GiveBitIdenticalResults(string kernel, bool f16)
    {
        var c = RowCase.Create(kernel, keys: 20_000, f16, seed: 13);
        try
        {
            using (CpuScratch.OverrideStackFloatLimit(int.MaxValue))
                c.Run();
            float[] onStack = c.Result();
            using (CpuScratch.OverrideStackFloatLimit(0))
                c.Run();
            AssertBitIdentical(onStack, c.Result());
        }
        finally
        {
            c.Dispose();
        }
    }

    // ------------------------------------------------------------------------------------
    //  Helpers
    // ------------------------------------------------------------------------------------

    private sealed class GroupedCase
    {
        public int Heads, KvHeads, HeadDim, Keys, MaxSeq;
        public float Scale;
        public float[] Q, K, V;        // K/V hold the cached values (widened from F16 when F16)
        public ushort[] KHalf, VHalf;  // the F16 cache itself; null for an F32 cache

        public int Group => Heads / KvHeads;

        public static GroupedCase Create(int heads, int kvHeads, int headDim, int keys, bool f16, int seed)
        {
            var rnd = new Random(seed);
            // Headroom past the live keys, as a KV cache has: the kernels must stride by MaxSeq.
            int maxSeq = keys + 7;
            var c = new GroupedCase
            {
                Heads = heads, KvHeads = kvHeads, HeadDim = headDim, Keys = keys, MaxSeq = maxSeq,
                Scale = 1f / MathF.Sqrt(headDim),
                Q = Random(rnd, heads * headDim),
                K = Random(rnd, kvHeads * maxSeq * headDim),
                V = Random(rnd, kvHeads * maxSeq * headDim),
            };
            if (f16)
            {
                c.KHalf = ToHalf(c.K);
                c.VHalf = ToHalf(c.V);
            }
            return c;
        }

        public float[] Run(int processorCount)
        {
            var result = new float[Heads * HeadDim];
            Run(processorCount, result);
            return result;
        }

        public void Run(int processorCount, float[] result)
        {
            fixed (float* q = Q, r = result)
            {
                if (KHalf != null)
                {
                    fixed (ushort* k = KHalf, v = VHalf)
                        ModelBase.AttentionDecodeGroupedF16(q, k, v, r, Heads, KvHeads, HeadDim, MaxSeq, Keys, Scale);
                }
                else
                {
                    fixed (float* k = K, v = V)
                        ModelBase.AttentionDecodeGrouped(q, k, v, r, Heads, KvHeads, HeadDim, MaxSeq, Keys, Scale, processorCount);
                }
            }
        }

        public double[] Reference()
        {
            var output = new double[Heads * HeadDim];
            for (int h = 0; h < Heads; h++)
            {
                long kvBase = (long)(h / Group) * MaxSeq * HeadDim;
                ReferenceRow(Q, h * HeadDim, K, V, kvBase, HeadDim, HeadDim, Keys, Scale, sink: null,
                    output, h * HeadDim);
            }
            return output;
        }
    }

    /// <summary>One of the per-model kernels that keeps one score row per head.</summary>
    private sealed class RowCase : IDisposable
    {
        private const int Heads = 2, KvHeads = 1, Dim = 8;
        private readonly string _kernel;
        private readonly int _keys;
        private readonly float _scale;
        private readonly float[] _q, _k, _v, _sinks;
        private readonly Tensor _qT, _kT, _vT, _rT;

        private RowCase(string kernel, int keys, bool f16, int seed)
        {
            _kernel = kernel;
            _keys = keys;
            _scale = 1f / MathF.Sqrt(Dim);
            var rnd = new Random(seed);
            int maxSeq = keys + 5;
            _q = Random(rnd, Heads * Dim);
            _k = Random(rnd, KvHeads * maxSeq * Dim);
            _v = Random(rnd, KvHeads * maxSeq * Dim);
            _sinks = kernel.StartsWith("gpt-oss", StringComparison.Ordinal) ? new[] { 0.25f, -0.5f } : null;

            _qT = FromFloats(_q, 1, Heads * Dim);
            _rT = new Tensor(Alloc, DType.Float32, 1, Heads * Dim);
            if (f16)
            {
                ushort[] kh = ToHalf(_k), vh = ToHalf(_v);   // widens _k/_v in place to the cached values
                _kT = FromHalves(kh, KvHeads, maxSeq, Dim);
                _vT = FromHalves(vh, KvHeads, maxSeq, Dim);
            }
            else
            {
                _kT = FromFloats(_k, KvHeads, maxSeq, Dim);
                _vT = FromFloats(_v, KvHeads, maxSeq, Dim);
            }
        }

        public static RowCase Create(string kernel, int keys, bool f16, int seed) => new(kernel, keys, f16, seed);

        public void Run()
        {
            switch (_kernel)
            {
                case "gpt-oss":
                    GptOssModel.AttentionDecodeWithSinks(_qT, _kT, _vT, _rT, Heads, KvHeads, Dim, _keys, _scale,
                        _sinks, isSWA: false, slidingWindow: 128);
                    break;
                case "gpt-oss-tp":
                    GptOssModel.AttentionDecodeWithSinksTPHost(_qT, _kT, _vT, _rT, Heads, KvHeads, Dim, 0, _keys, _scale, _sinks);
                    break;
                default:
                    throw new ArgumentException(_kernel);
            }
        }

        public float[] Result() => new ReadOnlySpan<float>(TensorComputePrimitives.GetFloatPointer(_rT), Heads * Dim).ToArray();

        public double[] Reference()
        {
            var output = new double[Heads * Dim];
            int maxSeq = _keys + 5;
            for (int h = 0; h < Heads; h++)
            {
                long kvBase = (long)(h / (Heads / KvHeads)) * maxSeq * Dim;
                ReferenceRow(_q, h * Dim, _k, _v, kvBase, Dim, Dim, _keys, _scale, _sinks?[h], output, h * Dim);
            }
            return output;
        }

        public void Dispose()
        {
            _qT.Dispose(); _kT.Dispose(); _vT.Dispose(); _rT.Dispose();
        }
    }

    /// <summary>Softmax(q.k * scale) . v over keys [0, keys) of one KV head, in double, with an
    /// optional attention sink (a virtual key that only enters the max and the denominator).</summary>
    private static void ReferenceRow(float[] q, int qOffset, float[] k, float[] v, long kvBase, int keyDim, int valDim,
        int keys, float scale, float? sink, double[] output, int outOffset)
    {
        var s = new double[keys];
        double max = sink ?? double.NegativeInfinity;
        for (int t = 0; t < keys; t++)
        {
            double dot = 0;
            long kRow = kvBase + (long)t * keyDim;
            for (int d = 0; d < keyDim; d++) dot += (double)q[qOffset + d] * k[kRow + d];
            s[t] = dot * scale;
            if (s[t] > max) max = s[t];
        }
        double sum = sink.HasValue ? Math.Exp(sink.Value - max) : 0;
        for (int t = 0; t < keys; t++) { s[t] = Math.Exp(s[t] - max); sum += s[t]; }
        for (int t = 0; t < keys; t++)
        {
            double w = s[t] / sum;
            long vRow = kvBase + (long)t * valDim;
            for (int d = 0; d < valDim; d++) output[outOffset + d] += w * v[vRow + d];
        }
    }

    private static void AssertMatchesReference(float[] actual, double[] expected)
    {
        Assert.Equal(expected.Length, actual.Length);
        double scale = 0;
        foreach (double e in expected) scale = Math.Max(scale, Math.Abs(e));
        for (int i = 0; i < actual.Length; i++)
        {
            Assert.False(float.IsNaN(actual[i]), $"output {i} is NaN");
            double err = Math.Abs(actual[i] - expected[i]);
            Assert.True(err <= 1e-4 * scale + 1e-3 * Math.Abs(expected[i]),
                $"output {i}: {actual[i]:R} vs reference {expected[i]:R} (row scale {scale:R})");
        }
    }

    private static void AssertBitIdentical(float[] expected, float[] actual)
    {
        Assert.Equal(expected.Length, actual.Length);
        for (int i = 0; i < expected.Length; i++)
        {
            int a = BitConverter.SingleToInt32Bits(expected[i]), b = BitConverter.SingleToInt32Bits(actual[i]);
            Assert.True(a == b, $"output {i}: stack {expected[i]:R} (0x{a:x8}) vs rented {actual[i]:R} (0x{b:x8})");
        }
    }

    /// <summary>Runs <paramref name="body"/> on a new thread with exactly <paramref name="stackBytes"/>
    /// of stack and rethrows its failure here. A stack overflow aborts the process instead.</summary>
    private static T OnThread<T>(int stackBytes, Func<T> body)
    {
        T result = default!;
        ExceptionDispatchInfo error = null;
        var thread = new Thread(() =>
        {
            try { result = body(); }
            catch (Exception e) { error = ExceptionDispatchInfo.Capture(e); }
        }, stackBytes)
        { IsBackground = true, Name = "cpu-attention-stack-test" };
        thread.Start();
        thread.Join();
        error?.Throw();
        return result;
    }

    private static float[] Random(Random rnd, int n)
    {
        var a = new float[n];
        for (int i = 0; i < n; i++) a[i] = (float)(rnd.NextDouble() * 2 - 1);
        return a;
    }

    /// <summary>F16 copy of <paramref name="values"/>; also rounds <paramref name="values"/> to what the cache holds.</summary>
    private static ushort[] ToHalf(float[] values)
    {
        var h = new ushort[values.Length];
        for (int i = 0; i < values.Length; i++)
        {
            System.Half half = (System.Half)values[i];
            h[i] = BitConverter.HalfToUInt16Bits(half);
            values[i] = (float)half;
        }
        return h;
    }

    private static Tensor FromFloats(float[] data, params long[] sizes)
    {
        var t = new Tensor(Alloc, DType.Float32, sizes);
        data.AsSpan().CopyTo(new Span<float>(TensorComputePrimitives.GetFloatPointer(t), data.Length));
        return t;
    }

    private static Tensor FromHalves(ushort[] data, params long[] sizes)
    {
        var t = new Tensor(Alloc, DType.Float16, sizes);
        data.AsSpan().CopyTo(new Span<ushort>(TensorComputePrimitives.GetHalfPointer(t), data.Length));
        return t;
    }
}
