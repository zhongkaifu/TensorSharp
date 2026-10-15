// Copyright (c) Zhongkai Fu. Licensed under the BSD-3-Clause license.
using System.Runtime.InteropServices;
using TensorSharp;
using TensorSharp.GGML;
using TensorSharp.Memory;
using TensorSharp.Models;

namespace InferenceWeb.Tests;

// Run in the CUDA lane against TENSORSHARP_GGML_NATIVE_BUILD_TESTS=ON.
// Missing native fault hooks fail explicitly rather than simulating an allocation.
// The historical class name now covers both Q8_0 and F16 streaming sessions.
public sealed class Q8StreamingSessionLifetimeTests
{
    private static readonly string[] Pools = ["shared", "gpu0"];

    [GgmlTheory(BackendType.GgmlCuda)]
    [InlineData(8, false)]
    [InlineData(1, true)]
    [Trait("Requires", "NativeTestHooks")]
    public void ModelResetRetriesRetainedWorkspaceBeforeResettingState(int weightType, bool refill)
    {
        string path = Path.Combine(Path.GetTempPath(), $"ts-stream-release-{Guid.NewGuid():N}.gguf");
        SyntheticGguf.Write(path,
            [new SyntheticGguf.Str { Key = "general.architecture", V = "probe" }],
            [new SyntheticGguf.Tensor { Name = "projection.weight", Dims = [32, 4],
                Type = (SyntheticGguf.GgmlType)weightType, Data = Enumerable.Repeat(1f, 128).ToArray() }]);
        using var native = new NativeFixture(2, weightType);
        var budget = Budget(4096);
        using var competitor = budget.Reserve(Pools.Select(pool => new MemoryCharge(pool, 64)));
        competitor.Commit();
        StreamingProjectionModel? model = null;
        try
        {
            model = new StreamingProjectionModel(path,
                new WeightStreamingOptions(budget, "shared", Pools, tileBytes: 256, tokenTileRows: 2));
            {
                long workspace = GgmlWeightStreamingSession.GetPayloadBytes(weightType, 32, 4, 2);
                native.FailNext(2); // Real execution completes; its first cudaFree handoff is refused.
                Assert.Throws<InvalidOperationException>(() => Run());
                Assert.Equal(1, model.State);
                Assert.Equal(0, model.Resets);
                AssertCharges(workspace);
                Assert.Contains("ResetKVCache", Assert.Throws<InvalidOperationException>(() => Run()).Message);
                Assert.Equal(1, model.State);

                native.FailNext(2); // Reset's release retry must also fail closed.
                Assert.Throws<InvalidOperationException>(model.ResetKVCache);
                Assert.Equal(0, model.Resets); // KV reset never ran ahead of unreleased CUDA work.
                Assert.Equal(1, model.State);
                AssertCharges(workspace);
                Assert.Contains("ResetKVCache", Assert.Throws<InvalidOperationException>(() => Run()).Message);

                model.ResetKVCache();
                Assert.Equal(1, model.Resets);
                Assert.Equal(0, model.State);
                AssertCharges(0);
                Assert.All(Run(), value => Assert.InRange(value, 31f, 33f));
                Assert.Equal(1, model.State);
                AssertCharges(0);
                float[] Run() => refill ? model!.ForwardRefill([1, 2]) : model!.Forward([1, 2]);
            }
            model.Dispose();
            model = null;
            Assert.All(budget.Snapshot(), pool => Assert.Equal(64, pool.Committed));
            competitor.Dispose();
            AssertEmpty(budget);
        }
        finally { native.FailNext(0); model?.Dispose(); File.Delete(path); }

        void AssertCharges(long workspace)
        {
            Assert.All(budget.Snapshot(), pool => Assert.Equal(0, pool.Reserved));
            Assert.Equal(64 + workspace, budget.Snapshot().Single(pool => pool.Pool == "gpu0").Committed);
            Assert.Equal(64 + 256 + workspace, budget.Snapshot().Single(pool => pool.Pool == "shared").Committed);
        }
    }

    [GgmlTheory(BackendType.GgmlCuda)]
    [InlineData(8)]
    [InlineData(1)]
    [Trait("Requires", "NativeTestHooks")]
    public void ConstructionAndCleanupFailure_RetainsOwnerAndQuotaUntilRetry(int weightType)
    {
        using var native = new NativeFixture(2, weightType);
        long bytes = GgmlWeightStreamingSession.GetPayloadBytes(weightType, 32, 4, 2);
        var budget = Budget(bytes);
        GgmlWeightStreamingSession? retained = null;
        try
        {
            native.FailNext(3); // Real cudaMalloc succeeds; construction and first cleanup fail.
            var error = Assert.Throws<GgmlWeightStreamingAllocationException>(() =>
                new GgmlWeightStreamingSession(budget, Pools, 0, weightType, 32, 4, 2, native.Input));
            retained = error.UnreleasedSession;
            Assert.NotNull(retained);
            Assert.Equal(bytes, retained.PayloadBytes);
            Assert.All(budget.Snapshot(), pool => Assert.Equal(bytes, pool.Reserved + pool.Committed));
            Assert.Null(budget.TryReserve([new MemoryCharge("gpu0", 1)]));
            retained.Dispose();
            AssertEmpty(budget);
            retained.Dispose();
            AssertEmpty(budget);
        }
        finally
        {
            native.FailNext(0);
            retained?.Dispose();
        }
    }

    [GgmlTheory(BackendType.GgmlCuda)]
    [InlineData(8)]
    [InlineData(1)]
    [Trait("Requires", "NativeTestHooks")]
    public void FailedDispose_PreservesCommittedCreditAndSupportsRetry(int weightType)
    {
        using var native = new NativeFixture(2, weightType);
        long bytes = GgmlWeightStreamingSession.GetPayloadBytes(weightType, 32, 4, 2);
        var budget = Budget(bytes);
        using var session = new GgmlWeightStreamingSession(budget, Pools, 0, weightType, 32, 4, 2, native.Input);
        try
        {
            native.FailNext(2);
            Assert.Throws<InvalidOperationException>(session.Dispose);
            Assert.All(budget.Snapshot(), pool =>
            {
                Assert.Equal(0, pool.Reserved);
                Assert.Equal(bytes, pool.Committed);
            });
            Assert.Null(budget.TryReserve([new MemoryCharge("shared", 1)]));
            session.Dispose();
            AssertEmpty(budget);
            using var replacement = new GgmlWeightStreamingSession(budget, Pools, 0, weightType, 32, 4, 2, native.Input);
            Assert.All(budget.Snapshot(), pool => Assert.Equal(bytes, pool.Committed));
        }
        finally { native.FailNext(0); }
        AssertEmpty(budget);
    }

    [GgmlTheory(BackendType.GgmlCuda)]
    [InlineData(8)]
    [InlineData(1)]
    [Trait("Requires", "NativeTestHooks")]
    public void ExternalQuotaRefusal_HappensBeforeNativeAllocation(int weightType)
    {
        using var native = new NativeFixture(2, weightType);
        long bytes = GgmlWeightStreamingSession.GetPayloadBytes(weightType, 32, 4, 2);
        var budget = Budget(bytes);
        using var external = budget.Reserve([new MemoryCharge("shared", bytes)]);
        external.Commit();
        try
        {
            native.FailNext(1);
            Assert.Throws<MemoryPressureException>(() =>
                new GgmlWeightStreamingSession(budget, Pools, 0, weightType, 32, 4, 2, native.Input));
            Assert.Equal(0, budget.Snapshot().Single(pool => pool.Pool == "gpu0").Committed);
            external.Dispose();
            // The pending native fault must still fire after quota is released:
            // refusal above never entered native creation or consumed the hook.
            var failure = Assert.Throws<InvalidOperationException>(() =>
                new GgmlWeightStreamingSession(budget, Pools, 0, weightType, 32, 4, 2, native.Input));
            Assert.Contains("injected", failure.Message, StringComparison.OrdinalIgnoreCase);
            AssertEmpty(budget);
            using (var retry = new GgmlWeightStreamingSession(budget, Pools, 0, weightType, 32, 4, 2, native.Input))
                Assert.All(budget.Snapshot(), pool => Assert.Equal(bytes, pool.Committed));
            AssertEmpty(budget);
        }
        finally { native.FailNext(0); }
    }

    [GgmlTheory(BackendType.GgmlCuda)]
    [InlineData(8)]
    [InlineData(1)]
    [Trait("Requires", "NativeTestHooks")]
    public void InputUpload_ReusesMaximumReservationAndRejectsArgumentsWithoutLosingInput(int weightType)
    {
        using var native = new NativeFixture(33, weightType);
        long bytes = GgmlWeightStreamingSession.GetPayloadBytes(weightType, 32, 4, 33);
        var budget = Budget(bytes);
        using (var session = new GgmlWeightStreamingSession(budget, Pools, 0, weightType, 32, 4, 33, native.Input))
        {
            Assert.Equal(33, session.MaxTokenCount);
            int revision = 0;
            foreach (int tokens in new[] { 33, 1, 17, 8, 9, 16, 32, 33 })
            {
                float input = ++revision;
                native.FillInput(input);
                session.UploadInput(native.Input, tokens);
                native.FillInput(float.NaN); // No CUDA read may outlive the upload call.
                Assert.Equal(tokens, session.TokenCount);
                Assert.Equal(bytes, session.PayloadBytes);
                AssertCommitted(budget, bytes);
                Assert.Null(budget.TryReserve([new MemoryCharge("shared", 1)]));
                Assert.Throws<ArgumentException>(() => session.UploadInput(IntPtr.Zero, tokens));
                Assert.Throws<ArgumentOutOfRangeException>(() => session.UploadInput(native.Input, 0));
                Assert.Throws<ArgumentOutOfRangeException>(() => session.UploadInput(native.Input, -1));
                Assert.Throws<ArgumentOutOfRangeException>(() => session.UploadInput(native.Input, 34));
                Assert.Equal(tokens, session.TokenCount);
                native.FillOutput(-12345.0f);
                session.Execute(native.Weights, 4, native.Output);
                float[] actual = native.ReadOutput();
                Assert.All(actual.Take(tokens * 4), value => Assert.Equal(input * 32, value));
                Assert.All(actual.Skip(tokens * 4), value => Assert.Equal(-12345.0f, value));
                AssertCommitted(budget, bytes);
            }
        }
        AssertEmpty(budget);
    }

    [GgmlTheory(BackendType.GgmlCuda)]
    [InlineData(8)]
    [InlineData(1)]
    [Trait("Requires", "NativeTestHooks")]
    public void InputUploadAndCleanupFailure_RetainsMaximumQuotaAndRequiresDisposal(int weightType)
    {
        using var native = new NativeFixture(33, weightType);
        long bytes = GgmlWeightStreamingSession.GetPayloadBytes(weightType, 32, 4, 33);
        var budget = Budget(bytes);
        using var session = new GgmlWeightStreamingSession(budget, Pools, 0, weightType, 32, 4, 33, native.Input);
        try
        {
            native.FillInput(2);
            native.FailNext(6); // Replace device input successfully, then fail upload and first cleanup.
            var failure = Assert.Throws<InvalidOperationException>(() => session.UploadInput(native.Input, 1));
            Assert.Contains("injected", failure.Message, StringComparison.OrdinalIgnoreCase);
            Assert.Equal(33, session.TokenCount); // Failed upload never publishes a new active count.
            Assert.Throws<InvalidOperationException>(() => session.Execute(native.Weights, 4, native.Output));
            Assert.Throws<InvalidOperationException>(() => session.UploadInput(native.Input, 1));
            AssertCommitted(budget, bytes);
            Assert.Throws<InvalidOperationException>(session.Dispose);
            AssertCommitted(budget, bytes);
            Assert.Null(budget.TryReserve([new MemoryCharge("gpu0", 1)]));
            session.Dispose();
            AssertEmpty(budget);
        }
        finally { native.FailNext(0); }
    }

    // These are real supported resident-dispatch shapes, not the tiny K=32,
    // four-row FullPrecision fixtures above. The logical 36-token operation is
    // split into a maximum 32-token input workspace. Its 37-row tile also tests
    // packed downloads from native padded output/storage.
    private const int ResidentWidth = 2560;
    private const int ResidentTileRows = 37;
    private const int ResidentLogicalRows = 128;

    [GgmlTheory(BackendType.GgmlCuda)]
    [InlineData(36)]
    [InlineData(640)]
    [Trait("Requires", "NativeTestHooks")]
    public void ResidentF16LogicalShape_IsFullyChargedEvenForTinyActiveTile(int logicalTokens)
    {
        const int logicalRows = 10752, tileRows = 4, capacity = 32;
        using var native = new NativeFixture(capacity, 1, ResidentWidth, tileRows);
        long bytes = GgmlWeightStreamingSession.GetPayloadBytes(1, ResidentWidth, tileRows, capacity,
            GgmlWeightStreamingArithmetic.ResidentCuda, logicalTokens, logicalRows);
        long logicalPayload = checked(2L * (ResidentWidth * (long)logicalRows
            + ResidentWidth * (long)logicalTokens + logicalRows * (long)logicalTokens));
        Assert.True(bytes >= logicalPayload + 4L * 1024 * 1024,
            "The cuBLAS compatibility payload must charge complete logical F16 weights/input/output and workspace.");
        Assert.True(bytes > 32L * 1024 * 1024,
            "A tiny active row tile must not conceal original-shape cuBLAS storage.");
        var budget = new MemoryBudget([new MemoryCharge("shared", 32L * 1024 * 1024), new MemoryCharge("gpu0", bytes)]);
        GgmlWeightStreamingSession Create() => new(budget, Pools, 0, 1, ResidentWidth, tileRows, capacity,
            native.Input, GgmlWeightStreamingArithmetic.ResidentCuda, logicalTokens, logicalRows);
        try
        {
            native.FailNext(1);
            Assert.Throws<MemoryPressureException>(() => Create());
            AssertEmpty(budget);
            Assert.True(budget.TrySetCapacity("shared", bytes));
            var failure = Assert.Throws<InvalidOperationException>(() => Create());
            Assert.Contains("injected", failure.Message, StringComparison.OrdinalIgnoreCase);
            AssertEmpty(budget);
            using (var session = Create())
            {
                Assert.Equal(bytes, session.PayloadBytes);
                AssertCommitted(budget, bytes);
                Assert.Null(budget.TryReserve([new MemoryCharge("gpu0", 1)]));
                native.FillOutput(-12345.0f);
                session.Execute(native.Weights, tileRows, native.Output);
                float[] output = native.ReadOutput();
                Assert.All(output.Take(capacity * tileRows), value => Assert.Equal((float)ResidentWidth, value));
                Assert.All(output.Skip(capacity * tileRows), value => Assert.Equal(-12345.0f, value));
            }
            AssertEmpty(budget);
        }
        finally { native.FailNext(0); }
    }

    [GgmlTheory(BackendType.GgmlCuda)]
    [InlineData(8)]
    [InlineData(1)]
    [Trait("Requires", "NativeTestHooks")]
    public void ResidentConstructionAndCleanupFailure_RetainsCompleteScratchCharge(int weightType)
    {
        using var native = new NativeFixture(32, weightType, ResidentWidth, ResidentTileRows);
        long bytes = ResidentBytes(weightType, 32, 36);
        long fullPrecisionBytes = GgmlWeightStreamingSession.GetPayloadBytes(weightType, ResidentWidth, ResidentTileRows, 32);
        Assert.True(bytes > fullPrecisionBytes);
        if (weightType == 1) Assert.True(bytes - fullPrecisionBytes >= 4L * 1024 * 1024,
            "F16 compatibility omitted the minimum resident cuBLAS workspace from the reservation.");
        var budget = Budget(bytes);
        GgmlWeightStreamingSession? retained = null;
        try
        {
            native.FailNext(3);
            var error = Assert.Throws<GgmlWeightStreamingAllocationException>(() =>
                ResidentSession(budget, weightType, 32, 36, native.Input));
            retained = error.UnreleasedSession;
            Assert.Equal(GgmlWeightStreamingArithmetic.ResidentCuda, retained.Arithmetic);
            Assert.Equal(bytes, retained.PayloadBytes);
            Assert.All(budget.Snapshot(), pool => Assert.Equal(bytes, pool.Reserved + pool.Committed));
            Assert.Null(budget.TryReserve([new MemoryCharge("shared", 1)]));
            Assert.Throws<InvalidOperationException>(() => retained.Execute(native.Weights, ResidentTileRows, native.Output));
            retained.Dispose();
            AssertEmpty(budget);
            retained.Dispose();
            AssertEmpty(budget);
        }
        finally { native.FailNext(0); retained?.Dispose(); }
    }

    [GgmlTheory(BackendType.GgmlCuda)]
    [InlineData(8)]
    [InlineData(1)]
    [Trait("Requires", "NativeTestHooks")]
    public void ResidentQuotaOneByteShort_DoesNotEnterNativeAllocation(int weightType)
    {
        using var native = new NativeFixture(32, weightType, ResidentWidth, ResidentTileRows);
        long bytes = ResidentBytes(weightType, 32, 36);
        var budget = new MemoryBudget([new MemoryCharge("shared", bytes - 1), new MemoryCharge("gpu0", bytes)]);
        try
        {
            native.FailNext(1);
            Assert.Throws<MemoryPressureException>(() => ResidentSession(budget, weightType, 32, 36, native.Input));
            AssertEmpty(budget);
            Assert.True(budget.TrySetCapacity("shared", bytes));
            // The fault remains pending: a one-byte deficit rejected before the
            // native cudaMalloc, including the compatibility scratch charge.
            var failure = Assert.Throws<InvalidOperationException>(() => ResidentSession(budget, weightType, 32, 36, native.Input));
            Assert.Contains("injected", failure.Message, StringComparison.OrdinalIgnoreCase);
            AssertEmpty(budget);
            using (var retry = ResidentSession(budget, weightType, 32, 36, native.Input))
                AssertCommitted(budget, bytes);
            AssertEmpty(budget);
        }
        finally { native.FailNext(0); }
    }

    [GgmlTheory(BackendType.GgmlCuda)]
    [InlineData(8)]
    [InlineData(1)]
    [Trait("Requires", "NativeTestHooks")]
    public void ResidentInputUploadFailure_PoisonsSessionAndRetainsScratchUntilReleaseRetry(int weightType)
    {
        using var native = new NativeFixture(32, weightType, ResidentWidth, ResidentTileRows);
        long bytes = ResidentBytes(weightType, 32, 36);
        var budget = Budget(bytes);
        using var session = ResidentSession(budget, weightType, 32, 36, native.Input);
        try
        {
            native.FillInput(2);
            native.FailNext(6);
            var failure = Assert.Throws<InvalidOperationException>(() => session.UploadInput(native.Input, 4));
            Assert.Contains("injected", failure.Message, StringComparison.OrdinalIgnoreCase);
            native.FillInput(float.NaN);
            Assert.Equal(32, session.TokenCount);
            Assert.Throws<InvalidOperationException>(() => session.Execute(native.Weights, ResidentTileRows, native.Output));
            Assert.Throws<InvalidOperationException>(() => session.UploadInput(native.Input, 1));
            AssertCommitted(budget, bytes);
            Assert.Throws<InvalidOperationException>(session.Dispose);
            AssertCommitted(budget, bytes);
            Assert.Null(budget.TryReserve([new MemoryCharge("gpu0", 1)]));
            session.Dispose();
            AssertEmpty(budget);
        }
        finally { native.FailNext(0); }
    }

    [GgmlTheory(BackendType.GgmlCuda)]
    [InlineData(8, 36, 32)]
    [InlineData(1, 36, 32)]
    [InlineData(8, 8, 8)]
    [InlineData(1, 8, 8)]
    [Trait("Requires", "NativeTestHooks")]
    public void ResidentInputReuse_PreservesLogicalDispatchAndMaximumCharge(int weightType, int logicalTokens, int capacity)
    {
        using var native = new NativeFixture(capacity, weightType, ResidentWidth, ResidentTileRows);
        long bytes = ResidentBytes(weightType, capacity, logicalTokens);
        var budget = Budget(bytes);
        try
        {
            using (var session = ResidentSession(budget, weightType, capacity, logicalTokens, native.Input))
            {
                session.Execute(native.Weights, ResidentTileRows, native.Output);
                float[] first = native.ReadOutput().Take(ResidentTileRows).ToArray();
                Assert.All(first, value => Assert.True(float.IsFinite(value) && value > 0));
                native.FailNext(1); // Input replacement must not construct another native workspace.
                int revision = 0;
                foreach (int tokens in new[] { capacity, 1, 4, 8, 9, 16, 17, capacity }.Where(n => n <= capacity))
                {
                    // Powers of two preserve the resident kernels' rounding, so
                    // this tests input refresh exactly without inventing a relaxed
                    // numerical oracle for Q8_1 or half-precision accumulation.
                    float scale = 1 << (revision++ % 4);
                    native.FillInput(scale);
                    session.UploadInput(native.Input, tokens);
                    native.FillInput(float.NaN);
                    Assert.Equal(tokens, session.TokenCount);
                    Assert.Equal(bytes, session.PayloadBytes);
                    AssertCommitted(budget, bytes);
                    Assert.Null(budget.TryReserve([new MemoryCharge("shared", 1)]));
                    Assert.Throws<ArgumentException>(() => session.UploadInput(IntPtr.Zero, tokens));
                    Assert.Throws<ArgumentOutOfRangeException>(() => session.UploadInput(native.Input, capacity + 1));
                    native.FillOutput(-12345.0f);
                    session.Execute(native.Weights, ResidentTileRows, native.Output);
                    float[] actual = native.ReadOutput();
                    for (int index = 0; index < tokens * ResidentTileRows; index++)
                        Assert.Equal(first[index % ResidentTileRows] * scale, actual[index]);
                    Assert.All(actual.Skip(tokens * ResidentTileRows), value => Assert.Equal(-12345.0f, value));
                    AssertCommitted(budget, bytes);
                }
            }
            AssertEmpty(budget);
            native.FillInput(1);
            var failure = Assert.Throws<InvalidOperationException>(() => ResidentSession(budget, weightType, capacity, logicalTokens, native.Input));
            Assert.Contains("injected", failure.Message, StringComparison.OrdinalIgnoreCase);
            AssertEmpty(budget);
        }
        finally { native.FailNext(0); }
    }

    private static long ResidentBytes(int type, int capacity, int logicalTokens)
        => GgmlWeightStreamingSession.GetPayloadBytes(type, ResidentWidth, ResidentTileRows, capacity,
            GgmlWeightStreamingArithmetic.ResidentCuda, logicalTokens, ResidentLogicalRows);

    private static GgmlWeightStreamingSession ResidentSession(MemoryBudget budget, int type, int capacity,
        int logicalTokens, IntPtr input)
        => new(budget, Pools, 0, type, ResidentWidth, ResidentTileRows, capacity, input,
            GgmlWeightStreamingArithmetic.ResidentCuda, logicalTokens, ResidentLogicalRows);

    private static MemoryBudget Budget(long bytes)
        => new(Pools.Select(pool => new MemoryCharge(pool, bytes)));

    private static void AssertCommitted(MemoryBudget budget, long bytes)
        => Assert.All(budget.Snapshot(), pool =>
        {
            Assert.Equal(0, pool.Reserved);
            Assert.Equal(bytes, pool.Committed);
        });

    private static void AssertEmpty(MemoryBudget budget)
        => Assert.All(budget.Snapshot(), pool =>
        {
            Assert.Equal(0, pool.Reserved);
            Assert.Equal(0, pool.Committed);
        });

    private sealed class StreamingProjectionModel : ModelBase
    {
        public int State { get; private set; }
        public int Resets { get; private set; }
        public StreamingProjectionModel(string path, WeightStreamingOptions options)
            : base(path, BackendType.GgmlCuda, weightStreaming: options)
        {
            try { LoadWeights(); }
            catch { Dispose(); throw; }
        }
        protected override float[] ForwardCore(int[] tokens)
        {
            State++;
            using var input = CreateFloatTensor(Enumerable.Repeat(1f, tokens.Length * 32).ToArray(), tokens.Length, 32);
            using var result = LinearForward(input, "projection.weight");
            return result.GetElementsAsFloat(checked(tokens.Length * 4));
        }
        protected override void ResetKVCacheCore() { State = 0; Resets++; }
    }

    private sealed class NativeFixture : IDisposable
    {
        [UnmanagedFunctionPointer(CallingConvention.Cdecl)]
        private delegate void InjectFailure(int flags);
        private readonly IntPtr _module;
        private readonly InjectFailure _inject;
        private readonly int _tokenCapacity;
        private readonly int _width;
        private readonly int _rows;
        public IntPtr Input { get; private set; }
        public IntPtr Weights { get; private set; }
        public IntPtr Output { get; private set; }

        public NativeFixture(int tokenCapacity, int weightType, int width = 32, int rows = 4)
        {
            _tokenCapacity = tokenCapacity;
            _width = width;
            _rows = rows;
            GgmlBasicOps.EnsureBackendAvailable(GgmlBackendType.Cuda);
            _module = NativeLibrary.Load(TestGates.MappedNativeGgmlOpsPath());
            try
            {
                _inject = Marshal.GetDelegateForFunctionPointer<InjectFailure>(NativeLibrary.GetExport(
                    _module, "TSGgml_TestQ8StreamingFailNext"));
                Input = Marshal.AllocHGlobal(checked(width * tokenCapacity * sizeof(float)));
                int rowBytes = weightType == 8 ? checked(width / 32 * 34) : checked(width * 2);
                Weights = Marshal.AllocHGlobal(checked(rowBytes * rows));
                Output = Marshal.AllocHGlobal(checked((rows * tokenCapacity + 8) * sizeof(float)));
                FillInput(1);
                byte[] weights = new byte[checked(rowBytes * rows)];
                for (int row = 0; row < rows; row++)
                {
                    if (weightType == 8)
                    {
                        for (int block = 0; block < width / 32; block++)
                        {
                            weights[row * rowBytes + block * 34 + 1] = 0x3c; // Exact FP16 scale=1, all Q8 values=1.
                            Array.Fill(weights, (byte)1, row * rowBytes + block * 34 + 2, 32);
                        }
                    }
                    else
                    {
                        for (int k = 0; k < width; k++) weights[row * rowBytes + 2 * k + 1] = 0x3c; // F16 1.
                    }
                }
                Marshal.Copy(weights, 0, Weights, weights.Length);
            }
            catch
            {
                if (Input != IntPtr.Zero) Marshal.FreeHGlobal(Input);
                if (Weights != IntPtr.Zero) Marshal.FreeHGlobal(Weights);
                if (Output != IntPtr.Zero) Marshal.FreeHGlobal(Output);
                NativeLibrary.Free(_module);
                throw;
            }
        }
        public void FailNext(int flags) => _inject(flags);
        public void FillInput(float value)
            => Marshal.Copy(Enumerable.Repeat(value, _width * _tokenCapacity).ToArray(), 0, Input, _width * _tokenCapacity);
        public void FillOutput(float value)
            => Marshal.Copy(Enumerable.Repeat(value, _rows * _tokenCapacity + 8).ToArray(), 0, Output, _rows * _tokenCapacity + 8);
        public float[] ReadOutput()
        {
            var result = new float[_rows * _tokenCapacity + 8];
            Marshal.Copy(Output, result, 0, result.Length);
            return result;
        }
        public void Dispose()
        {
            _inject(0);
            Marshal.FreeHGlobal(Input);
            Marshal.FreeHGlobal(Weights);
            Marshal.FreeHGlobal(Output);
            Input = IntPtr.Zero;
            Weights = IntPtr.Zero;
            Output = IntPtr.Zero;
            NativeLibrary.Free(_module);
        }
    }
}
