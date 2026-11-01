// Copyright (c) Zhongkai Fu. All rights reserved.
// Licensed under the BSD-3-Clause license in the repository root.
using System.Diagnostics;
using TensorSharp.Cuda;
using TensorSharp.Cuda.Interop;
using Xunit.Abstractions;

namespace InferenceWeb.Tests;

[Collection(EngineEnvironmentCollection.Name)]
public sealed unsafe class Qwen35PreparedGdnTests
{
    private readonly ITestOutputHelper _output;
    public Qwen35PreparedGdnTests(ITestOutputHelper output) => _output = output;

    [Theory]
    [InlineData(16, 1, 366592)]
    [InlineData(17, 1, 733184)]
    [InlineData(33, 3, 3299328)]
    [InlineData(4096, 1, 93847552)]
    public void PacketAllocation_AccountsForTailAndEverySequence(int rows, int sequences, long bytes)
        => Assert.Equal(bytes, Q35Kernels.GdnPreparedScratchBytes(rows, sequences, 16, 48));

    [Fact]
    public void Capacity_CoversIndependentlyPaddedSequencesAtEveryLaneWidth()
    {
        long capacity = Q35Kernels.GdnPreparedScratchCapacityBytes(4160, 16, 16, 48);
        Assert.Equal(100812800, capacity);
        for (int sequences = 1; sequences <= 16; sequences++)
            for (int rows = 32; rows * sequences <= 4160; rows += 17)
                Assert.True(Q35Kernels.GdnPreparedScratchBytes(rows, sequences, 16, 48) <= capacity);
    }

    [CudaFact(NvccOutput = "tensorsharp_q35_kernels.ptx", TensorCoreModule = "tensorsharp_q35_kernels")]
    public void PreparedGdn_MatchesEveryOutputAndFinalStateBit_WithFiniteInputsAndPartialChunks()
    {
        using var gpu = new Harness();
        _output.WriteLine($"device={CudaDevice.GetDevice(0)} module_sha256={gpu.ModuleHash}");
        var shapes = new[] {
            (1, 1, 1, 1), (15, 2, 3, 2), (16, 3, 5, 1), (17, 2, 6, 3),
            (31, 2, 6, 2), (32, 16, 48, 1), (33, 3, 7, 3), (47, 1, 3, 2),
            (63, 2, 6, 1), (64, 16, 48, 2), (65, 3, 7, 3), (127, 2, 6, 2),
            (129, 16, 48, 1), (257, 2, 6, 3)
        };
        foreach (var (rows, hk, hv, sequences) in shapes)
            foreach (bool boundaries in new[] { false, true })
            {
                using var data = new Data(gpu, rows, sequences, hk, hv, boundaries);
                string label = $"rows={rows} seq={sequences} hk={hk} hv={hv} boundaries={boundaries}";
                data.Reset();
                data.Issue(prepared: false);
                var first = data.Read();
                data.Issue(prepared: false);
                var continuation = data.Read();
                data.Reset();
                data.Issue(prepared: true);
                Exact(first, data.Read(), label + " initial");
                data.Issue(prepared: true);
                Exact(continuation, data.Read(), label + " continued-state");
                data.Reset();
                data.Issue(prepared: true);
                Exact(first, data.Read(), label + " reset/reuse");
            }
        _output.WriteLine("28 cases: every output and final-state bit equal, including continued state, zero/one beta, zero/negative log decay, grouped heads, padding and partial chunks");
    }

    [CudaFact(NvccOutput = "tensorsharp_q35_kernels.ptx", TensorCoreModule = "tensorsharp_q35_kernels")]
    public void PreparedGdn_Microbenchmark_ReportsLegacyAndAllThreePreparedPasses()
    {
        using var gpu = new Harness();
        _output.WriteLine($"device={CudaDevice.GetDevice(0)} module_sha256={gpu.ModuleHash}; exploratory timings, no speed pass/fail threshold");
        foreach (int rows in new[] { 256, 1024, 4096 })
        {
            using var data = new Data(gpu, rows, 1, 16, 48, boundaries: false);
            data.Reset(); data.Issue(false); var oracle = data.Read();
            data.Reset(); data.Issue(true); Exact(oracle, data.Read(), $"timed rows={rows}");
            for (int repeat = 0; repeat < 3; repeat++)
                foreach (bool prepared in repeat % 2 == 0 ? new[] { false, true } : new[] { true, false })
                {
                    data.Reset();
                    for (int warm = 0; warm < 2; warm++) data.Issue(prepared);
                    CudaDriverApi.cuStreamSynchronize(gpu.Stream).ThrowOnError();
                    var time = gpu.Time(() => data.Issue(prepared), 8);
                    _output.WriteLine($"rows={rows} hk=16 hv=48 seq=1 repeat={repeat} prepared={prepared} scratch_bytes={data.ScratchBytes} gpu_ms_per_call={time.Gpu:F6} wall_ms_per_call={time.Wall:F6}");
                }
        }
    }

    private void Exact(Result expected, Result actual, string label)
    {
        Compare(expected.Output, actual.Output, label + " output");
        Compare(expected.State, actual.State, label + " state");
    }
    private void Compare(float[] expected, float[] actual, string label)
    {
        long differences = 0;
        int first = -1;
        double max = 0;
        for (int i = 0; i < expected.Length; i++)
        {
            if (!float.IsFinite(expected[i]) || !float.IsFinite(actual[i]))
                Assert.Fail($"{label}: non-finite value at {i}");
            if (BitConverter.SingleToInt32Bits(expected[i]) != BitConverter.SingleToInt32Bits(actual[i]))
            {
                if (first < 0) first = i;
                differences++;
                max = Math.Max(max, Math.Abs((double)expected[i] - actual[i]));
            }
        }
        if (differences > 0)
            _output.WriteLine($"{label}: {differences}/{expected.Length} differing floats, first={first}, max_abs={max:E8}");
        Assert.True(differences == 0, $"{label}: not bit-identical; do not qualify this optimization with a relaxed tolerance");
    }
    private sealed record Result(float[] Output, float[] State);

    private sealed class Data : IDisposable
    {
        private const int S = 128, Guard = 16, Canary = 0x537ab19d;
        private readonly Harness _gpu;
        private readonly int _rows, _sequences, _hk, _hv, _conv, _outputCount, _stateCount;
        private readonly List<IntPtr> _owned = new();
        private readonly IntPtr _qkv, _g, _beta, _stateRaw, _outputRaw, _statePtrs, _scratchRaw;
        private readonly float[] _initialState;
        private IntPtr State => _stateRaw + Guard * 4;
        private IntPtr Output => _outputRaw + Guard * 4;
        private IntPtr Scratch => _scratchRaw + Guard * 4;
        public long ScratchBytes { get; }

        public Data(Harness gpu, int rows, int sequences, int hk, int hv, bool boundaries)
        {
            _gpu = gpu; _rows = rows; _sequences = sequences; _hk = hk; _hv = hv;
            _conv = (2 * hk + hv) * S;
            _outputCount = rows * sequences * hv * S;
            _stateCount = sequences * hv * S * S;
            ScratchBytes = Q35Kernels.GdnPreparedScratchBytes(rows, sequences, hk, hv);
            try
            {
                var rng = new Random(971 + rows + 13 * hk + sequences);
                var qkv = new float[rows * sequences * _conv];
                var decay = new float[rows * sequences * hv];
                var beta = new float[decay.Length];
                for (int row = 0; row < rows * sequences; row++)
                {
                    int at = row * _conv;
                    for (int d = 0; d < _conv; d++) qkv[at + d] = (float)(rng.NextDouble() * 2 - 1);
                    for (int h = 0; h < 2 * hk; h++)
                    {
                        double squared = 0;
                        for (int d = 0; d < S; d++) squared += (double)qkv[at + h * S + d] * qkv[at + h * S + d];
                        float scale = (float)(1 / Math.Sqrt(squared));
                        for (int d = 0; d < S; d++)
                            qkv[at + h * S + d] = boundaries && (row + h) % 11 == 0 ? (d == (h % S) ? 1f : 0f)
                                : boundaries && (row + h) % 13 == 0 ? (d % 2 == 0 ? 0f : BitConverter.Int32BitsToSingle(int.MinValue))
                                : qkv[at + h * S + d] * scale;
                    }
                    for (int h = 0; h < hv; h++)
                    {
                        int i = row * hv + h;
                        decay[i] = boundaries ? ((i % 7) switch { 0 => 0f, 1 => -0f, 2 => -0.00000095367431640625f,
                            3 => -0.125f, 4 => -4f, 5 => -16f, _ => -80f }) : (float)(-0.01 - rng.NextDouble() * 1.5);
                        beta[i] = boundaries ? ((i % 5) switch { 0 => 0f, 1 => 1f, 2 => float.BitDecrement(1f),
                            3 => float.Epsilon, _ => 0.5f }) : (float)(0.01 + rng.NextDouble() * 0.98);
                        if (row / rows > 0 && row % rows >= Math.Max(0, rows - row / rows))
                            decay[i] = beta[i] = 0f; // masked tail of a padded sequence
                    }
                }
                _initialState = Enumerable.Range(0, _stateCount).Select(_ => (float)(rng.NextDouble() * 0.1 - 0.05)).ToArray();
                _qkv = Alloc(qkv.LongLength * 4); _gpu.Upload(_qkv, qkv);
                _g = Alloc(decay.LongLength * 4); _gpu.Upload(_g, decay);
                _beta = Alloc(beta.LongLength * 4); _gpu.Upload(_beta, beta);
                _stateRaw = Alloc(((long)_stateCount + 2 * Guard) * 4);
                _outputRaw = Alloc(((long)_outputCount + 2 * Guard) * 4);
                _scratchRaw = Alloc(ScratchBytes + 2 * Guard * 4);
                _statePtrs = Alloc((long)sequences * IntPtr.Size);
                _gpu.Upload(_statePtrs, Enumerable.Range(0, sequences).Select(i => State + (nint)((long)i * hv * S * S * 4)).ToArray());
                foreach (var (ptr, count) in new[] { (_stateRaw, (long)_stateCount), (_outputRaw, (long)_outputCount), (_scratchRaw, ScratchBytes / 4) })
                {
                    _gpu.Upload(ptr, Enumerable.Repeat(Canary, Guard).ToArray());
                    _gpu.Upload(ptr + (nint)((count + Guard) * 4), Enumerable.Repeat(Canary, Guard).ToArray());
                }
            }
            catch { Dispose(); throw; }
        }
        private IntPtr Alloc(long bytes)
        {
            CudaDriverApi.cuMemAlloc(out IntPtr ptr, new UIntPtr((ulong)bytes)).ThrowOnError();
            try { _owned.Add(ptr); }
            catch { CudaDriverApi.cuMemFree(ptr); throw; }
            return ptr;
        }
        public void Reset() => _gpu.Upload(State, _initialState);
        public void Issue(bool prepared) => _gpu.Issue(_qkv, _conv, _hk, _hv, _rows, _sequences, _g, _beta, _statePtrs, Output, Scratch, prepared);
        public Result Read()
        {
            CudaDriverApi.cuStreamSynchronize(_gpu.Stream).ThrowOnError();
            foreach (var (ptr, count) in new[] { (_stateRaw, (long)_stateCount), (_outputRaw, (long)_outputCount), (_scratchRaw, ScratchBytes / 4) })
            {
                Assert.All(_gpu.Read<int>(ptr, Guard), x => Assert.Equal(Canary, x));
                Assert.All(_gpu.Read<int>(ptr + (nint)((count + Guard) * 4), Guard), x => Assert.Equal(Canary, x));
            }
            return new Result(_gpu.Read<float>(Output, _outputCount), _gpu.Read<float>(State, _stateCount));
        }
        public void Dispose()
        {
            CudaDriverApi.cuStreamSynchronize(_gpu.Stream).ThrowOnError();
            foreach (IntPtr ptr in _owned) CudaDriverApi.cuMemFree(ptr);
            _owned.Clear();
        }
    }

    private sealed class Harness : IDisposable
    {
        private readonly CudaAllocator _allocator = new(0);
        private readonly CudaModule _module;
        private readonly IntPtr _legacy, _qk, _control, _prepared;
        public IntPtr Stream => _allocator.Stream.Handle;
        public string ModuleHash => _module.ImageSha256;
        public Harness()
        {
            try
            {
                _allocator.Context.MakeCurrent();
                _module = CudaModule.LoadFromFile(CudaKernels.LocatePtxPath("tensorsharp_q35_kernels.ptx"), identify: true);
                _legacy = _module.GetFunction("ts_q35_gdn_chunk_f32");
                _qk = _module.GetFunction("ts_q35_gdn_prepare_qk_f32");
                _control = _module.GetFunction("ts_q35_gdn_prepare_control_f32");
                _prepared = _module.GetFunction("ts_q35_gdn_chunk_prepared_f32");
            }
            catch { _module?.Dispose(); _allocator.Dispose(); throw; }
        }
        public void Upload<T>(IntPtr dst, T[] values) where T : unmanaged
        {
            fixed (T* ptr = values)
                CudaDriverApi.cuMemcpyHtoD(dst, (IntPtr)ptr, new UIntPtr((ulong)values.LongLength * (ulong)sizeof(T))).ThrowOnError();
        }
        public T[] Read<T>(IntPtr src, int count) where T : unmanaged
        {
            var values = new T[count];
            fixed (T* ptr = values)
                CudaDriverApi.cuMemcpyDtoH((IntPtr)ptr, src, new UIntPtr((ulong)count * (ulong)sizeof(T))).ThrowOnError();
            return values;
        }
        public void Issue(IntPtr qkv, int conv, int hk, int hv, int rows, int sequences, IntPtr g, IntPtr beta,
            IntPtr states, IntPtr output, IntPtr scratch, bool prepared)
        {
            if (prepared)
            {
                int chunks = (rows + 15) / 16;
                IntPtr control = scratch + (nint)((long)sequences * hk * chunks * Q35Kernels.GdnQkPacketBytes);
                void** a = stackalloc void*[] { &qkv, &conv, &hk, &rows, &scratch };
                Launch(_qk, (uint)chunks, (uint)hk, (uint)sequences, a);
                void** b = stackalloc void*[] { &scratch, &g, &beta, &hk, &hv, &rows, &control };
                Launch(_control, (uint)chunks, (uint)hv, (uint)sequences, b);
                void** c = stackalloc void*[] { &qkv, &conv, &hk, &hv, &rows, &g, &beta, &states, &output, &scratch, &control };
                Launch(_prepared, (uint)hv * 4, (uint)sequences, 1, c);
            }
            else
            {
                void** a = stackalloc void*[] { &qkv, &conv, &hk, &hv, &rows, &g, &beta, &states, &output };
                Launch(_legacy, (uint)hv * 4, (uint)sequences, 1, a);
            }
        }
        private void Launch(IntPtr fn, uint x, uint y, uint z, void** args) =>
            CudaDriverApi.cuLaunchKernel(fn, x, y, z, 256, 1, 1, 0, Stream, (IntPtr)args, IntPtr.Zero).ThrowOnError();
        public (double Gpu, double Wall) Time(Action action, int iterations)
        {
            CudaDriverApi.cuEventCreate(out IntPtr start, 0).ThrowOnError();
            IntPtr end = IntPtr.Zero;
            try
            {
                CudaDriverApi.cuEventCreate(out end, 0).ThrowOnError();
                var wall = Stopwatch.StartNew();
                CudaDriverApi.cuEventRecord(start, Stream).ThrowOnError();
                for (int i = 0; i < iterations; i++) action();
                CudaDriverApi.cuEventRecord(end, Stream).ThrowOnError();
                CudaDriverApi.cuEventSynchronize(end).ThrowOnError();
                wall.Stop();
                CudaDriverApi.cuEventElapsedTime(out float gpu, start, end).ThrowOnError();
                return (gpu / iterations, wall.Elapsed.TotalMilliseconds / iterations);
            }
            finally { CudaDriverApi.cuEventDestroy(start); if (end != IntPtr.Zero) CudaDriverApi.cuEventDestroy(end); }
        }
        public void Dispose()
        {
            CudaDriverApi.cuStreamSynchronize(Stream).ThrowOnError();
            _module.Dispose(); _allocator.Dispose();
        }
    }
}
