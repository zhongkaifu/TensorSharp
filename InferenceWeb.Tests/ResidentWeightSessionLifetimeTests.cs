// Copyright (c) Zhongkai Fu. Licensed under the BSD-3-Clause license.
using System.Runtime.InteropServices;
using TensorSharp;
using TensorSharp.GGML;
using TensorSharp.Memory;
using TensorSharp.Models;

namespace InferenceWeb.Tests;

// Run with TS_TEST_GGML_BACKEND=cuda and TENSORSHARP_GGML_NATIVE_BUILD_TESTS=ON.
// These cases require real CUDA allocations and the native fault hooks. A missing
// hook is a failure, not a simulated allocation or successful skipped scenario.
public sealed class ResidentWeightSessionLifetimeTests
{
    private static readonly string[] Pools = ["shared", "gpu0"];

    [GgmlTheory(BackendType.GgmlCuda)]
    [InlineData(8, 96, 129, 9)]
    [InlineData(8, 2560, 384, 36)]
    [InlineData(1, 2560, 128, 36)]
    [Trait("Requires", "NativeTestHooks")]
    public void SegmentedUploadsAndInputReuseMatchUnchangedResidentProjectionExactly(int type, int width, int rows, int tokens)
    {
        using var native = new Fixture(type, width, rows, tokens);
        // Both references use the unchanged ordinary GGML path, before the
        // caller's upload buffers are deliberately overwritten with poison.
        float[] first = native.Reference(native.FirstInput);
        float[] second = native.Reference(native.SecondInput);
        long bytes = GgmlResidentWeightSession.GetPayloadBytes(0, type, width, rows, tokens);
        var budget = Budget(bytes);
        GgmlResidentWeightSession? session = null;
        try
        {
            session = new(budget, Pools, 0, type, width, rows, tokens);
            Assert.Equal(bytes, session.PayloadBytes);
            AssertCommitted(budget, bytes);
            native.FillOutput();
            Assert.Throws<InvalidOperationException>(session.Project);
            Assert.Throws<InvalidOperationException>(() => session.Download(native.OutputData, 0, tokens, 0, rows));

            Assert.ThrowsAny<ArgumentException>(() => session.UploadWeightRows(IntPtr.Zero, 0, 1));
            Assert.ThrowsAny<ArgumentException>(() => session.UploadWeightRows(native.Weights, -1, 1));
            Assert.ThrowsAny<ArgumentException>(() => session.UploadWeightRows(native.Weights, rows, 1));
            Assert.ThrowsAny<ArgumentException>(() => session.UploadWeightRows(native.Weights, 0, 0));
            session.UploadWeightRows(native.Weights, 0, 63);
            Assert.Throws<InvalidOperationException>(session.Project);
            Assert.ThrowsAny<ArgumentException>(() => session.UploadWeightRows(native.WeightRow(64), 64, 1));
            Assert.ThrowsAny<ArgumentException>(() => session.UploadWeightRows(native.Weights, 0, 1));
            session.UploadWeightRows(native.WeightRow(63), 63, rows - 63);
            native.PoisonWeights(); // No host weight pointer may be retained.

            native.SetInput(native.FirstInput);
            Assert.ThrowsAny<ArgumentException>(() => session.UploadInputTokens(IntPtr.Zero, 0, 1));
            Assert.ThrowsAny<ArgumentException>(() => session.UploadInputTokens(native.Input, tokens, 1));
            Assert.ThrowsAny<ArgumentException>(() => session.UploadInputTokens(native.Input, 0, 0));
            session.UploadInputTokens(native.Input, 0, 5);
            // Starting at zero replaces an unfinished input epoch. The old
            // five-token progress must not conceal a gap after the new prefix.
            session.UploadInputTokens(native.Input, 0, 3);
            Assert.ThrowsAny<ArgumentException>(() => session.UploadInputTokens(native.InputToken(4), 4, 1));
            Assert.Throws<InvalidOperationException>(session.Project);
            native.PoisonInput(0, 3);
            session.UploadInputTokens(native.InputToken(3), 3, tokens - 3);
            native.PoisonInput(3, tokens - 3);
            Assert.Throws<InvalidOperationException>(() => session.Download(native.OutputData, 0, tokens, 0, rows));
            Assert.All(native.ReadOutput(), value => Assert.Equal(Fixture.Canary, value));

            session.Project();
            session.Project(); // Completed projection is reusable, not an input reset.
            native.AssertDownload(session, first, 0, tokens, 0, rows);
            native.AssertDownload(session, first, 2, tokens - 3, 61, 63);
            Assert.ThrowsAny<ArgumentException>(() => session.Download(native.OutputData, tokens, 1, 0, 1));
            Assert.ThrowsAny<ArgumentException>(() => session.Download(native.OutputData, 0, 1, rows, 1));
            // A rejected download leaves the completed result usable.
            native.AssertDownload(session, first, tokens - 1, 1, rows - 1, 1);

            native.FailNext(1); // Reusing inputs/projecting must not create another arena.
            native.SetInput(native.SecondInput);
            session.UploadInputTokens(native.Input, 0, 4);
            native.PoisonInput(0, 4);
            Assert.Throws<InvalidOperationException>(() => session.Download(native.OutputData, 0, tokens, 0, rows));
            Assert.Throws<InvalidOperationException>(session.Project);
            session.UploadInputTokens(native.InputToken(4), 4, tokens - 4);
            native.PoisonInput(4, tokens - 4);
            session.Project();
            native.AssertDownload(session, second, 0, tokens, 0, rows);
            AssertCommitted(budget, bytes);
            Assert.Null(budget.TryReserve([new MemoryCharge("gpu0", 1)]));
            session.Dispose();
            session = null;
            AssertEmpty(budget);
            // The pending constructor fault proves that refresh/project never
            // reconstructed a session or discarded the already uploaded weights.
            var failure = Assert.Throws<InvalidOperationException>(() =>
                new GgmlResidentWeightSession(budget, Pools, 0, type, width, rows, tokens));
            Assert.Contains("injected", failure.Message, StringComparison.OrdinalIgnoreCase);
            AssertEmpty(budget);
        }
        finally { native.FailNext(0); session?.Dispose(); }
    }

    [GgmlTheory(BackendType.GgmlCuda)]
    [InlineData(4)]
    [InlineData(8)]
    [Trait("Requires", "NativeTestHooks")]
    public void RealUploadFailurePoisonsWorkAndRetainsQuotaUntilReleaseSucceeds(int uploadFault)
    {
        using var native = new Fixture(8, 96, 129, 9);
        long bytes = GgmlResidentWeightSession.GetPayloadBytes(0, 8, 96, 129, 9);
        var budget = Budget(bytes);
        using var session = new GgmlResidentWeightSession(budget, Pools, 0, 8, 96, 129, 9);
        try
        {
            if (uploadFault == 4) session.UploadWeightRows(native.Weights, 0, 129);
            native.FailNext(uploadFault | 2);
            var failure = Assert.Throws<InvalidOperationException>(() =>
            {
                if (uploadFault == 4) session.UploadInputTokens(native.Input, 0, 9);
                else session.UploadWeightRows(native.Weights, 0, 63);
            });
            Assert.Contains("injected", failure.Message, StringComparison.OrdinalIgnoreCase);
            native.PoisonWeights();
            native.PoisonInput(0, 9);
            Assert.Throws<InvalidOperationException>(session.Project);
            Assert.Throws<InvalidOperationException>(() => session.Download(native.OutputData, 0, 9, 0, 129));
            Assert.Throws<InvalidOperationException>(() => session.UploadInputTokens(native.Input, 0, 9));
            AssertCommitted(budget, bytes);
            Assert.Throws<InvalidOperationException>(session.Dispose);
            AssertCommitted(budget, bytes);
            Assert.Null(budget.TryReserve([new MemoryCharge("shared", 1)]));
            session.Dispose();
            AssertEmpty(budget);
        }
        finally { native.FailNext(0); }
    }

    [GgmlTheory(BackendType.GgmlCuda)]
    [InlineData(8)]
    [InlineData(1)]
    [Trait("Requires", "NativeTestHooks")]
    public void ConstructorAndCleanupFailureExposeTheChargedOwnerForRetry(int type)
    {
        using var native = new Fixture(type, 2560, 128, 36);
        long bytes = GgmlResidentWeightSession.GetPayloadBytes(0, type, 2560, 128, 36);
        var budget = Budget(bytes);
        GgmlResidentWeightSession? retained = null;
        try
        {
            native.FailNext(3);
            var failure = Assert.Throws<GgmlResidentWeightAllocationException>(() =>
                new GgmlResidentWeightSession(budget, Pools, 0, type, 2560, 128, 36));
            retained = failure.UnreleasedSession;
            Assert.NotNull(retained);
            Assert.Equal(bytes, retained.PayloadBytes);
            Assert.All(budget.Snapshot(), pool => Assert.Equal(bytes, pool.Reserved + pool.Committed));
            Assert.Null(budget.TryReserve([new MemoryCharge("gpu0", 1)]));
            Assert.Throws<InvalidOperationException>(retained.Project);
            Assert.Throws<InvalidOperationException>(() => retained.UploadInputTokens(native.Input, 0, 36));
            retained.Dispose();
            AssertEmpty(budget);
            retained.Dispose();
            AssertEmpty(budget);
        }
        finally { native.FailNext(0); retained?.Dispose(); }
    }

    [GgmlFact(BackendType.GgmlCuda)]
    [Trait("Requires", "NativeTestHooks")]
    public void ActualHostStagingSharesCapacityAndRejectsBeforeNativeAllocation()
    {
        using var native = new Fixture(8, 96, 129, 9);
        long bytes = GgmlResidentWeightSession.GetPayloadBytes(0, 8, 96, 129, 9);
        var budget = new MemoryBudget([new MemoryCharge("shared", bytes + 63), new MemoryCharge("gpu0", bytes)]);
        var options = new WeightStreamingOptions(budget, "shared", Pools, tileBytes: 64);
        using (var host = new StreamingHostBuffer(options, 64))
        {
            host.GetSpan().Fill(0x4d); // A real separately owned host allocation.
            try
            {
                native.FailNext(1);
                Assert.Throws<MemoryPressureException>(() =>
                    new GgmlResidentWeightSession(budget, Pools, 0, 8, 96, 129, 9));
                Assert.Equal(64, budget.Snapshot().Single(pool => pool.Pool == "shared").Committed);
                Assert.Equal(0, budget.Snapshot().Single(pool => pool.Pool == "gpu0").Committed);
                Assert.All(budget.Snapshot(), pool => Assert.Equal(0, pool.Reserved));
                Assert.True(budget.TrySetCapacity("shared", bytes + 64));
                var failure = Assert.Throws<InvalidOperationException>(() =>
                    new GgmlResidentWeightSession(budget, Pools, 0, 8, 96, 129, 9));
                Assert.Contains("injected", failure.Message, StringComparison.OrdinalIgnoreCase);
                using (var session = new GgmlResidentWeightSession(budget, Pools, 0, 8, 96, 129, 9))
                {
                    Assert.Equal(bytes + 64, budget.Snapshot().Single(pool => pool.Pool == "shared").Committed);
                    Assert.Equal(bytes, budget.Snapshot().Single(pool => pool.Pool == "gpu0").Committed);
                    Assert.Null(budget.TryReserve([new MemoryCharge("shared", 1)]));
                    Assert.Null(budget.TryReserve([new MemoryCharge("gpu0", 1)]));
                }
                Assert.All(host.GetSpan().ToArray(), value => Assert.Equal((byte)0x4d, value));
                Assert.Equal(64, budget.Snapshot().Single(pool => pool.Pool == "shared").Committed);
                Assert.Equal(0, budget.Snapshot().Single(pool => pool.Pool == "gpu0").Committed);
            }
            finally { native.FailNext(0); }
        }
        AssertEmpty(budget);
    }

    private static MemoryBudget Budget(long bytes)
        => new(Pools.Select(pool => new MemoryCharge(pool, bytes)));
    private static void AssertCommitted(MemoryBudget budget, long bytes)
        => Assert.All(budget.Snapshot(), pool => { Assert.Equal(0, pool.Reserved); Assert.Equal(bytes, pool.Committed); });
    private static void AssertEmpty(MemoryBudget budget)
        => Assert.All(budget.Snapshot(), pool => { Assert.Equal(0, pool.Reserved); Assert.Equal(0, pool.Committed); });

    private sealed class Fixture : IDisposable
    {
        [UnmanagedFunctionPointer(CallingConvention.Cdecl)]
        private delegate void InjectFailure(int flags);
        private readonly IntPtr _module;
        private readonly InjectFailure _inject;
        private readonly GgmlContext _context;
        private readonly GgmlAllocator _allocator;
        private readonly int _type, _width, _rows, _tokens, _rowBytes;
        private const int Guard = 16;
        internal const float Canary = -98765.25f;
        internal IntPtr Weights { get; private set; }
        internal IntPtr Input { get; private set; }
        private IntPtr Output { get; set; }
        internal IntPtr OutputData => Output + Guard * sizeof(float);
        internal float[] FirstInput { get; }
        internal float[] SecondInput { get; }

        internal Fixture(int type, int width, int rows, int tokens)
        {
            _type = type; _width = width; _rows = rows; _tokens = tokens;
            _rowBytes = type == 8 ? width / 32 * 34 : width * 2;
            GgmlBasicOps.EnsureBackendAvailable(GgmlBackendType.Cuda);
            _context = new GgmlContext([0], GgmlBackendType.Cuda);
            _allocator = new GgmlAllocator(_context, 0);
            _module = NativeLibrary.Load(TestGates.MappedNativeGgmlOpsPath());
            try
            {
                _inject = Marshal.GetDelegateForFunctionPointer<InjectFailure>(NativeLibrary.GetExport(
                    _module, "TSGgml_TestQ8StreamingFailNext"));
                Weights = Marshal.AllocHGlobal(checked(rows * _rowBytes));
                Input = Marshal.AllocHGlobal(checked(tokens * width * sizeof(float)));
                Output = Marshal.AllocHGlobal(checked((tokens * rows + 2 * Guard) * sizeof(float)));
                byte[] packed = new byte[checked(rows * _rowBytes)];
                for (int row = 0; row < rows; row++)
                {
                    if (type == 8)
                    {
                        for (int block = 0; block < width / 32; block++)
                        {
                            int offset = row * _rowBytes + block * 34;
                            ushort scale = BitConverter.HalfToUInt16Bits((System.Half)(1f / (1 << (1 + (row + block) % 5))));
                            packed[offset] = (byte)scale; packed[offset + 1] = (byte)(scale >> 8);
                            for (int k = 0; k < 32; k++) packed[offset + 2 + k] = unchecked((byte)((row * 17 + block * 7 + k * 13) % 53 - 26));
                        }
                    }
                    else for (int k = 0; k < width; k++)
                    {
                        ushort value = BitConverter.HalfToUInt16Bits((System.Half)(0.17f * MathF.Sin(row * 0.13f + k * 0.07f)));
                        int offset = row * _rowBytes + k * 2;
                        packed[offset] = (byte)value; packed[offset + 1] = (byte)(value >> 8);
                    }
                }
                Marshal.Copy(packed, 0, Weights, packed.Length);
                FirstInput = Enumerable.Range(0, tokens * width).Select(i => 0.21f * MathF.Sin(i * 0.031f + i / width * 0.2f)).ToArray();
                SecondInput = Enumerable.Range(0, tokens * width).Select(i => 0.19f * MathF.Cos(i * 0.019f - i / width * 0.3f)).ToArray();
                SetInput(FirstInput);
                FillOutput();
            }
            catch
            {
                Marshal.FreeHGlobal(Weights); Marshal.FreeHGlobal(Input); Marshal.FreeHGlobal(Output);
                _context.ReleasePooledMemory();
                NativeLibrary.Free(_module);
                throw;
            }
        }

        internal float[] Reference(float[] input)
        {
            using var source = new Tensor(_allocator, DType.Float32, _tokens, _width);
            source.SetElementsAsFloat(input);
            using var result = new Tensor(_allocator, DType.Float32, _tokens, _rows);
            GgmlBasicOps.AddmmQuant(result, source, Weights, _type, _width, _rows, (long)_rows * _rowBytes);
            return result.GetElementsAsFloat(_tokens * _rows);
        }
        internal IntPtr WeightRow(int row) => Weights + checked(row * _rowBytes);
        internal IntPtr InputToken(int token) => Input + checked(token * _width * sizeof(float));
        internal void FailNext(int flags) => _inject(flags);
        internal void SetInput(float[] values) => Marshal.Copy(values, 0, Input, values.Length);
        internal void PoisonInput(int first, int count)
            => Marshal.Copy(Enumerable.Repeat(float.NaN, count * _width).ToArray(), 0, InputToken(first), count * _width);
        internal void PoisonWeights()
            => Marshal.Copy(Enumerable.Repeat((byte)0x7e, _rows * _rowBytes).ToArray(), 0, Weights, _rows * _rowBytes);
        internal void FillOutput()
            => Marshal.Copy(Enumerable.Repeat(Canary, _tokens * _rows + 2 * Guard).ToArray(), 0, Output, _tokens * _rows + 2 * Guard);
        internal float[] ReadOutput()
        {
            var values = new float[_tokens * _rows + 2 * Guard];
            Marshal.Copy(Output, values, 0, values.Length);
            return values;
        }
        internal void AssertDownload(GgmlResidentWeightSession session, float[] expected,
            int firstToken, int tokenCount, int firstRow, int rowCount)
        {
            FillOutput();
            session.Download(OutputData, firstToken, tokenCount, firstRow, rowCount);
            float[] actual = ReadOutput();
            Assert.All(actual.Take(Guard), value => Assert.Equal(Canary, value));
            for (int token = 0; token < tokenCount; token++)
            for (int row = 0; row < rowCount; row++)
                Assert.Equal(BitConverter.SingleToInt32Bits(expected[(firstToken + token) * _rows + firstRow + row]),
                    BitConverter.SingleToInt32Bits(actual[Guard + token * rowCount + row]));
            Assert.All(actual.Skip(Guard + tokenCount * rowCount), value => Assert.Equal(Canary, value));
        }
        public void Dispose()
        {
            _inject(0);
            // The reference API may cache the immutable caller weight key.
            // Clear that ownership before the fixture's host address can recycle.
            GgmlBasicOps.ClearHostBufferCache();
            _context.ReleasePooledMemory();
            Marshal.FreeHGlobal(Weights); Marshal.FreeHGlobal(Input); Marshal.FreeHGlobal(Output);
            Weights = Input = Output = IntPtr.Zero;
            NativeLibrary.Free(_module);
        }
    }
}
