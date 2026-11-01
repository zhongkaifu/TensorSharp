// Copyright (c) Zhongkai Fu. All rights reserved.
// Licensed under the BSD-3-Clause license in the repository root.
using System.Diagnostics;
using TensorSharp.Cuda;
using TensorSharp.Cuda.Interop;
using Xunit.Abstractions;

namespace InferenceWeb.Tests;

/// <summary>Exact selection tests over fixed logits, isolated from the drafter's atomic
/// split-K GEMMs. Values are copied, not recomputed: signed zero and every selected float bit
/// must match the legacy kernel and an independent CPU sort, including invalid-value sentinels.</summary>
public sealed unsafe class Qwen35PartitionedTopKTests
{
    private const int Partitions = 32;
    private readonly ITestOutputHelper _output;

    public Qwen35PartitionedTopKTests(ITestOutputHelper output) => _output = output;

    [Fact]
    public void InputFixture_ActuallyContainsEveryNamedSpecialDistribution()
    {
        const int n = 257;
        float[] input = Inputs(7, n);
        Assert.All(input[..n], x => Assert.True(float.IsFinite(x)));
        Assert.Equal(13, input[n..(2 * n)].Distinct().Count());
        Assert.Equal(new[] { int.MinValue, 0 }, input[(2 * n)..(3 * n)]
            .Select(BitConverter.SingleToInt32Bits).Distinct().Order().ToArray());
        Assert.All(input[(3 * n)..(4 * n)], x => Assert.Equal(float.NegativeInfinity, x));
        Assert.All(input[(4 * n)..(5 * n)], x => Assert.True(float.IsNaN(x)));
        Assert.Contains(float.PositiveInfinity, input[(5 * n)..(6 * n)]);
        Assert.Contains(float.MaxValue, input[(5 * n)..(6 * n)]);
        Assert.Contains(float.MinValue, input[(5 * n)..(6 * n)]);
        Assert.Contains(input[(6 * n)..], float.IsNaN);
        Assert.Contains(float.NegativeInfinity, input[(6 * n)..]);
        Assert.Contains(float.PositiveInfinity, input[(6 * n)..]);
    }

    [CudaFact(NvccOutput = "tensorsharp_q35_kernels.ptx")]
    public void PartitionedSelection_MatchesLegacyAndCpuSort_BitForBit()
    {
        using var gpu = new Kernels();
        _output.WriteLine($"device={CudaDevice.GetDevice(0)} module_sha256={gpu.ImageHash}");
        // Includes empty partitions, n < k, thread/tile/partition boundaries and a full Qwen
        // vocabulary with a ragged tail. Each row uses a different adversarial distribution.
        foreach (int n in new[] { 1, 7, 17, 31, 33, 257, 16381, 16387, 248320, 248323 })
        {
            const int rows = 7;
            float[] input = Inputs(rows, n);
            foreach (int k in new[] { 1, 5, 16 })
            {
                using var buffers = new Buffers(gpu, input, rows, n, k);
                buffers.Issue(partitioned: false);
                var legacy = buffers.Read();
                buffers.Issue(partitioned: true);
                var candidate = buffers.Read();
                var oracle = Oracle(input, rows, n, k);
                Check(oracle, legacy, $"legacy n={n} k={k}");
                Check(oracle, candidate, $"partitioned n={n} k={k}");
                Check(legacy, candidate, $"paired n={n} k={k}");
            }
        }
        _output.WriteLine("30 shapes, 210 rows: random, plateaus, signed zeros, all -Inf, all NaN, sparse extrema and mixed invalid values; exact indices and selected float bits");
    }

    [CudaFact(NvccOutput = "tensorsharp_q35_kernels.ptx")]
    public void PartitionedSelection_GraphReplayReadsUpdatedInputs_Exactly()
    {
        using var gpu = new Kernels();
        const int n = 16387, rows = 7, k = 16;
        using var buffers = new Buffers(gpu, Inputs(rows, n), rows, n, k);
        IntPtr graph = IntPtr.Zero, executable = IntPtr.Zero;
        CudaDriverApi.cuStreamBeginCapture(gpu.Stream, CudaDriverApi.CU_STREAM_CAPTURE_MODE_THREAD_LOCAL).ThrowOnError();
        try
        {
            buffers.Issue(partitioned: true);
            CudaDriverApi.cuStreamEndCapture(gpu.Stream, out graph).ThrowOnError();
            CudaDriverApi.cuGraphInstantiateWithFlags(out executable, graph, 0).ThrowOnError();
            for (int replay = 0; replay < 4; replay++)
            {
                float[] input = Inputs(rows, n, seed: 77 + replay);
                // Move the winner across partition boundaries after capture. A graph that baked
                // input values or the wrong scratch offsets must fail its independent oracle.
                for (int row = 0; row < rows; row++)
                    input[row * n + (replay * 8191 + row * 257) % n] = float.PositiveInfinity;
                gpu.Upload(buffers.Input, input);
                CudaDriverApi.cuGraphLaunch(executable, gpu.Stream).ThrowOnError();
                Check(Oracle(input, rows, n, k), buffers.Read(), $"graph replay {replay}");
            }
        }
        finally
        {
            // End an invalid capture as well, so test failures do not poison later GPU tests.
            if (graph == IntPtr.Zero)
                CudaDriverApi.cuStreamEndCapture(gpu.Stream, out graph);
            CudaDriverApi.cuStreamSynchronize(gpu.Stream).ThrowOnError();
            if (executable != IntPtr.Zero) CudaDriverApi.cuGraphExecDestroy(executable);
            if (graph != IntPtr.Zero) CudaDriverApi.cuGraphDestroy(graph);
        }
    }

    [CudaFact(NvccOutput = "tensorsharp_q35_kernels.ptx")]
    public void PartitionedSelection_Microbenchmark_ReportsBothPaths()
    {
        using var gpu = new Kernels();
        _output.WriteLine($"device={CudaDevice.GetDevice(0)} module_sha256={gpu.ImageHash}; timing is exploratory, without a pass/fail speed threshold");
        const int n = 248320, k = 16, iterations = 24;
        foreach (int rows in new[] { 7, 14, 28 })
        {
            var rng = new Random(103 + rows);
            float[] input = Enumerable.Range(0, rows * n).Select(_ => (float)(rng.NextDouble() * 40 - 20)).ToArray();
            using var buffers = new Buffers(gpu, input, rows, n, k);
            for (int warm = 0; warm < 3; warm++)
            {
                buffers.Issue(partitioned: false);
                buffers.Issue(partitioned: true);
            }
            CudaDriverApi.cuStreamSynchronize(gpu.Stream).ThrowOnError();
            // Alternate order to make a consistently second-running arm visible in the logs.
            for (int repeat = 0; repeat < 3; repeat++)
            {
                foreach (bool partitioned in repeat % 2 == 0 ? new[] { false, true } : new[] { true, false })
                {
                    var time = gpu.Time(() => buffers.Issue(partitioned), iterations);
                    _output.WriteLine($"n={n} rows={rows} k={k} repeat={repeat} path={(partitioned ? "partitioned" : "legacy")} iterations={iterations} gpu_ms_per_call={time.GpuMs:F6} wall_ms_per_call={time.WallMs:F6}");
                }
            }
            buffers.Issue(partitioned: false);
            var legacy = buffers.Read();
            buffers.Issue(partitioned: true);
            Check(legacy, buffers.Read(), $"timed n={n} rows={rows}");
        }
    }

    private static float[] Inputs(int rows, int n, int seed = 719)
    {
        var rng = new Random(seed);
        var input = new float[rows * n];
        for (int row = 0; row < rows; row++)
            for (int i = 0; i < n; i++)
                input[row * n + i] = (row % 7) switch
                {
                    0 => (float)(rng.NextDouble() * 40 - 20),
                    1 => (i * 17 + seed) % 13 - 6,
                    2 => (i & 1) == 0 ? 0f : BitConverter.Int32BitsToSingle(unchecked((int)0x80000000)),
                    3 => float.NegativeInfinity,
                    4 => float.NaN,
                    5 => i == n - 1 ? float.PositiveInfinity : i == n / 2 ? float.MaxValue
                        : i == 0 ? float.MinValue : float.NegativeInfinity,
                    _ => i % 17 == 0 ? float.NaN : i % 113 == 0 ? float.PositiveInfinity
                        : i % 19 == 0 ? float.NegativeInfinity : i % 29 == 0 ? -0f : (float)(rng.NextDouble() * 6 - 3)
                };
        return input;
    }

    private sealed record Selection(int[] Indices, int[] FloatBits);

    private static Selection Oracle(float[] input, int rows, int n, int k)
    {
        var indices = new int[rows * k];
        var bits = new int[rows * k];
        for (int row = 0; row < rows; row++)
        {
            int offset = row * n;
            // Legacy excludes NaN and -Infinity when filling its initially empty lists. Ties
            // (including +/-0) prefer the lower original index and preserve that input's bits.
            var best = Enumerable.Range(0, n).Where(i => input[offset + i] > float.NegativeInfinity)
                .OrderByDescending(i => input[offset + i]).ThenBy(i => i).Take(k).ToArray();
            for (int j = 0; j < k; j++)
            {
                indices[row * k + j] = j < best.Length ? best[j] : int.MaxValue;
                bits[row * k + j] = BitConverter.SingleToInt32Bits(j < best.Length ? input[offset + best[j]] : float.NegativeInfinity);
            }
        }
        return new Selection(indices, bits);
    }

    private static void Check(Selection expected, Selection actual, string label)
    {
        Assert.True(expected.Indices.SequenceEqual(actual.Indices), $"{label}: selected indices differ");
        Assert.True(expected.FloatBits.SequenceEqual(actual.FloatBits), $"{label}: selected float bits differ");
    }

    private sealed class Buffers : IDisposable
    {
        private const int Guard = 16, Canary = 0x4a517c33;
        private readonly Kernels _gpu;
        private readonly int _rows, _n, _k, _count;
        private readonly List<IntPtr> _owned = new();
        private readonly IntPtr _indices, _values, _partialIndices, _partialValues;
        public IntPtr Input { get; }

        public Buffers(Kernels gpu, float[] input, int rows, int n, int k)
        {
            _gpu = gpu; _rows = rows; _n = n; _k = k; _count = rows * k;
            Input = Alloc(input.LongLength);
            _indices = Alloc(_count + 2 * Guard);
            _values = Alloc(_count + 2 * Guard);
            _partialIndices = Alloc((long)rows * Partitions * k);
            _partialValues = Alloc((long)rows * Partitions * k);
            _gpu.Upload(Input, input);
            _gpu.Upload(_indices, Enumerable.Repeat(Canary, _count + 2 * Guard).ToArray());
            _gpu.Upload(_values, Enumerable.Repeat(Canary, _count + 2 * Guard).ToArray());
        }

        private IntPtr Alloc(long elements)
        {
            CudaDriverApi.cuMemAlloc(out IntPtr p, new UIntPtr(checked((ulong)elements * 4))).ThrowOnError();
            _owned.Add(p);
            return p;
        }

        public void Issue(bool partitioned) => _gpu.Issue(Input, _rows, _n, _k,
            _indices + Guard * 4, _values + Guard * 4, _partialIndices, _partialValues, partitioned);

        public Selection Read()
        {
            CudaDriverApi.cuStreamSynchronize(_gpu.Stream).ThrowOnError();
            int[] indices = _gpu.Read(_indices, _count + 2 * Guard);
            int[] values = _gpu.Read(_values, _count + 2 * Guard);
            foreach (var buffer in new[] { indices, values })
            {
                Assert.All(buffer.Take(Guard), value => Assert.Equal(Canary, value));
                Assert.All(buffer.Skip(Guard + _count), value => Assert.Equal(Canary, value));
            }
            return new Selection(indices[Guard..(Guard + _count)], values[Guard..(Guard + _count)]);
        }

        public void Dispose()
        {
            CudaDriverApi.cuStreamSynchronize(_gpu.Stream).ThrowOnError();
            foreach (IntPtr p in _owned) CudaDriverApi.cuMemFree(p);
        }
    }

    private sealed class Kernels : IDisposable
    {
        private readonly CudaAllocator _allocator = new(0);
        private readonly CudaModule _module;
        private readonly IntPtr _legacy, _partial, _merge;
        public IntPtr Stream => _allocator.Stream.Handle;
        public string ImageHash => _module.ImageSha256;

        public Kernels()
        {
            _allocator.Context.MakeCurrent();
            _module = CudaModule.LoadFromFile(CudaKernels.LocatePtxPath("tensorsharp_q35_kernels.ptx"), identify: true);
            _legacy = _module.GetFunction("ts_q35_topk_f32");
            _partial = _module.GetFunction("ts_q35_topk_partial_f32");
            _merge = _module.GetFunction("ts_q35_topk_merge_f32");
        }

        public void Upload<T>(IntPtr dst, T[] values) where T : unmanaged
        {
            fixed (T* p = values)
                CudaDriverApi.cuMemcpyHtoD(dst, (IntPtr)p, new UIntPtr((ulong)values.LongLength * (ulong)sizeof(T))).ThrowOnError();
        }

        public int[] Read(IntPtr src, int count)
        {
            var values = new int[count];
            fixed (int* p = values)
                CudaDriverApi.cuMemcpyDtoH((IntPtr)p, src, new UIntPtr((ulong)count * 4)).ThrowOnError();
            return values;
        }

        public void Issue(IntPtr input, int rows, int n, int k, IntPtr idx, IntPtr val, IntPtr partialIdx, IntPtr partialVal, bool partitioned)
        {
            int partitions = Partitions;
            if (partitioned)
            {
                void** args = stackalloc void*[] { &input, &n, &k, &partitions, &partialIdx, &partialVal };
                Launch(_partial, Partitions, (uint)rows, args);
                void** mergeArgs = stackalloc void*[] { &partialIdx, &partialVal, &k, &partitions, &idx, &val };
                Launch(_merge, (uint)rows, 1, mergeArgs);
            }
            else
            {
                void** args = stackalloc void*[] { &input, &n, &k, &idx, &val };
                Launch(_legacy, (uint)rows, 1, args);
            }
        }

        private void Launch(IntPtr fn, uint x, uint y, void** args) =>
            CudaDriverApi.cuLaunchKernel(fn, x, y, 1, 256, 1, 1, 0, Stream, (IntPtr)args, IntPtr.Zero).ThrowOnError();

        public (double GpuMs, double WallMs) Time(Action issue, int iterations)
        {
            CudaDriverApi.cuEventCreate(out IntPtr start, 0).ThrowOnError();
            CudaDriverApi.cuEventCreate(out IntPtr end, 0).ThrowOnError();
            try
            {
                var wall = Stopwatch.StartNew();
                CudaDriverApi.cuEventRecord(start, Stream).ThrowOnError();
                for (int i = 0; i < iterations; i++) issue();
                CudaDriverApi.cuEventRecord(end, Stream).ThrowOnError();
                CudaDriverApi.cuEventSynchronize(end).ThrowOnError();
                wall.Stop();
                CudaDriverApi.cuEventElapsedTime(out float elapsed, start, end).ThrowOnError();
                return (elapsed / iterations, wall.Elapsed.TotalMilliseconds / iterations);
            }
            finally
            {
                CudaDriverApi.cuEventDestroy(start);
                CudaDriverApi.cuEventDestroy(end);
            }
        }

        public void Dispose()
        {
            CudaDriverApi.cuStreamSynchronize(Stream).ThrowOnError();
            _module.Dispose();
            _allocator.Dispose();
        }
    }
}
