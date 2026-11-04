// Copyright (c) Zhongkai Fu. All rights reserved.
// Licensed under the BSD-3-Clause license in the repository root.
#nullable enable
using System;
using System.Linq;
using TensorSharp.Cuda.Interop;

namespace TensorSharp.Cuda;

/// <summary>Explicit BF16 prefill candidate, using original Ignis Br64/Bc64
/// arithmetic with an owned pointer-page accessor. Callers own scratch/cache
/// buffers and completion. No runtime allocation or KV conversion; the separate
/// T8/B1 Verify entry is module-only until its independent qualification.</summary>
internal sealed unsafe class CudaNinferPrefillAttentionModule : IDisposable
{
    internal const string EnvironmentVariable = "TS_Q35_NINFER_PREFILL_ATTN_PTX";
    internal const string Policy = "ignis-e600-bf16-br64-bc64-s1s4-ts256-pointerpages-v1";
    internal const int HeadDim = 256, QueryHeads = 24, KvHeads = 4, Width = HeadDim * QueryHeads;
    internal const int MaximumRows = 8192, MaximumContext = 262144, DynamicSharedBytes = 98304;
    // Separate module-only qualification seam. Do not broaden ordinary-prefill
    // Supports/Plan, or select this route without the B1/full-valid Verify guard.
    // S4 is the original 170-SM Prompt dispatch for this exact T8 shape.
    internal const int Verify8Rows = 8, Verify8Context = 40960, Verify8Splits = 4;
    internal const string Verify8Policy = "ignis-e600-bf16-br64-bc64-verify8-b1-s4-c40960-devicebase-v1";
    internal const long KvPlaneBlockBytes = 4L * 256 * 256 * 2, KvBlockBytes = 16 * 2 * KvPlaneBlockBytes;
    private readonly CudaModule _module;
    private readonly IntPtr _context, _pack, _gate, _owned, _original, _reduce;
    private bool _disposed;
    internal string ImageSha256 => _module.ImageSha256;
    internal string[] KernelSymbols { get; }
    // Host submissions (including capture), not completion, elapsed GPU time or
    // graph replays. Reference launches stay separate; replay cannot increment these.
    internal long PagedCoreLaunches { get; private set; }
    internal long OriginalCoreLaunches { get; private set; }

    internal readonly record struct Layout(long Query, long Output, long PartialAcc, long PartialM,
        long PartialL, long Planes, long Positions, long Table, long Bytes, int Splits, int Pages);

    internal CudaNinferPrefillAttentionModule(string path)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(path);
        CudaDriverApi.cuCtxGetCurrent(out _context).ThrowOnError();
        if (_context == IntPtr.Zero || IntPtr.Size != 8) throw new PlatformNotSupportedException("The prefill adapter requires its current64-bit CUDA context.");
        _module = CudaModule.LoadFromFile(path, identify: true);
        try
        {
            var symbols = _module.EnumerateFunctions();
            if (symbols.Length != 5) throw new InvalidOperationException("Prefill module requires two cores, shared reducer and two owned bridges.");
            KernelSymbols = symbols.Select(x => x.Name).OrderBy(x => x, StringComparer.Ordinal).ToArray();
            _pack = Bind(symbols.Single(x => x.Name == "ts_ni_prefill_pack").Function,
                new ulong[] { 0, 8, 16, 24, 32, 40, 48, 56, 64, 72, 80 }, new ulong[] { 8, 8, 8, 8, 8, 8, 8, 8, 4, 8, 4 });
            _gate = Bind(symbols.Single(x => x.Name == "ts_ni_prefill_gate").Function,
                new ulong[] { 0, 8, 16, 24 }, new ulong[] { 8, 8, 8, 8 });
            ulong[] offsets = { 0, 8, 16, 24, 32, 40, 48, 56, 60, 64, 68, 72, 80, 88, 96, 104, 112, 120, 128 };
            ulong[] sizes = { 8, 8, 8, 8, 8, 4, 8, 4, 4, 4, 4, 8, 8, 8, 4, 8, 8, 8, 4 };
            _owned = Bind(Find(symbols, "ts_ni_prefill_bf16_kernel"), offsets, sizes);
            _original = Bind(Find(symbols, "gqa_attention_prefill_bf16_kernel"), offsets, sizes);
            _reduce = Bind(Find(symbols, "gqa_attention_prefill_reduce_kernel"),
                new ulong[] { 0, 8, 16, 24, 28, 32, 40 }, new ulong[] { 8, 8, 8, 4, 4, 4, 8 });
            foreach (IntPtr kernel in new[] { _owned, _original })
            {
                // Exactly the original load-time opt-in, before any graph capture.
                CudaDriverApi.cuFuncSetAttribute(kernel, 8, DynamicSharedBytes).ThrowOnError();
                CudaDriverApi.cuFuncGetAttribute(out int shared, 8, kernel).ThrowOnError();
                CudaDriverApi.cuFuncGetAttribute(out int threads, 0, kernel).ThrowOnError();
                if (shared < DynamicSharedBytes || threads < 128) throw new InvalidOperationException("Original96KiB prefill launch is unavailable.");
            }
        }
        catch { _module.Dispose(); throw; }
    }
    private static IntPtr Find(CudaFunctionSymbol[] symbols, string name)
        => symbols.Single(x => x.Name.Contains(name, StringComparison.Ordinal)).Function;
    private static IntPtr Bind(IntPtr kernel, ulong[] offsets, ulong[] sizes)
    {
        CudaDriverApi.cuFuncLoad(kernel).ThrowOnError();
        for (uint i = 0; i < offsets.Length; i++)
        {
            CudaDriverApi.cuFuncGetParamInfo(kernel, (UIntPtr)i, out var offset, out var size).ThrowOnError();
            if (offset.ToUInt64() != offsets[i] || size.ToUInt64() != sizes[i]) throw new InvalidOperationException("Prefill kernel deviceABI differs.");
        }
        if (CudaDriverApi.cuFuncGetParamInfo(kernel, (UIntPtr)(uint)offsets.Length, out _, out _) != 1)
            throw new InvalidOperationException("Unexpected extra prefill kernel argument.");
        return kernel;
    }

    internal static bool Supports(int rows, int sequences, bool isVerify, bool allRowsValid = true)
        => rows is >= 64 and <= MaximumRows && sequences == 1 && !isVerify && allRowsValid;
    internal static int SplitCount(int rows, int sms)
    {
        if (rows < 64 || rows > MaximumRows || sms < 1) throw new ArgumentOutOfRangeException(nameof(rows));
        long items = ((long)rows + 63) / 64 * QueryHeads;
        double first = (double)((items + sms - 1) / sms), bestRatio = first; int best = 1;
        for (int split = 2; split <= 4; split++)
        {
            double ratio = (double)((items * split + sms - 1) / sms) / split;
            if (ratio < bestRatio) { bestRatio = ratio; best = split; }
        }
        return bestRatio > 0.9 * first ? 1 : best;
    }
    private static Layout AllocateLayout(int rows, int context, int splits)
    {
        long cursor = 0;
        long Take(long bytes) { long at = cursor; cursor = checked((cursor + bytes + 255) & ~255L); return at; }
        long elements = checked((long)rows * Width), stats = checked((long)rows * QueryHeads * splits);
        long query = Take(elements * 2), output = Take(elements * 2);
        long acc = Take(elements * splits * 4), m = Take(stats * 4), l = Take(stats * 4);
        long planes = Take(32), positions = Take(4); int pages = checked((context + 63) / 64);
        long table = Take((long)pages * 4);
        return new(query, output, acc, m, l, planes, positions, table, cursor, splits, pages);
    }
    internal static long WorkspaceCapacity(int maximumRows, int maximumContext)
    {
        if (maximumRows < 64 || maximumRows > MaximumRows || maximumContext < maximumRows || maximumContext > MaximumContext)
            throw new ArgumentOutOfRangeException(nameof(maximumRows));
        return AllocateLayout(maximumRows, maximumContext, 4).Bytes;
    }
    internal static Layout Plan(int rows, int maxVisibleKeys, int maxBlocks, int sms)
    {
        if (!Supports(rows, 1, false) || maxVisibleKeys < rows || maxVisibleKeys > MaximumContext
            || maxBlocks < (maxVisibleKeys + 255) / 256 || maxBlocks > (MaximumContext + 255) / 256 + 1)
            throw new ArgumentOutOfRangeException(nameof(rows), "Prefill rows/history must fit the declared native BF16 pointer pages.");
        return AllocateLayout(rows, maxVisibleKeys, SplitCount(rows, sms));
    }
    internal static bool SupportsVerify8(int rows, int sequences, bool isVerify, bool allRowsValid, int maxVisibleKeys)
        => rows == Verify8Rows && sequences == 1 && isVerify && allRowsValid
            && maxVisibleKeys >= Verify8Rows && maxVisibleKeys <= Verify8Context;

    /// <summary>Fixed graph-stable capacity. Visible history is admission metadata,
    /// never a captured page count, base position, split count or kernel argument.</summary>
    internal static Layout PlanVerify8(int maxVisibleKeys, int maxBlocks)
    {
        if (!SupportsVerify8(Verify8Rows, 1, true, true, maxVisibleKeys)
            || maxBlocks < (maxVisibleKeys + 255) / 256 || maxBlocks > (MaximumContext + 255) / 256 + 1)
            throw new ArgumentOutOfRangeException(nameof(maxVisibleKeys), "Verify8 must fit its fixed40960-token native BF16 cache capacity.");
        return AllocateLayout(Verify8Rows, Verify8Context, Verify8Splits);
    }
    internal static long WorkspaceVerify8() => AllocateLayout(Verify8Rows, Verify8Context, Verify8Splits).Bytes;

    // Pure host admission is separately testable without loading a CUDA device.
    // The caller owns the pointees of blocks and must keep them alive until all
    // graph executions complete. On each replay it must admit current physical
    // base+8 <=40960; positionsBaseDev is not a RoPE-coordinate array.
    internal static Layout ValidateVerify8Buffers(IntPtr qF32, IntPtr gateF32, IntPtr blocks, int maxBlocks,
        long kOff, long vOff, IntPtr positionsBaseDev, int maxVisibleKeys, float scale,
        IntPtr outputF32, IntPtr workspace, long workspaceBytes)
    {
        var plan = PlanVerify8(maxVisibleKeys, maxBlocks); ValidatePlaneOffsets(kOff, vOff); Scale(scale);
        const long bytes = (long)Verify8Rows * Width * 4;
        long tableBytes = (long)maxBlocks * 8;
        if (workspaceBytes < plan.Bytes) throw new ArgumentException("Verify8 workspace is too small.");
        Pointer(qF32, bytes, 4); Pointer(gateF32, bytes, 4); Pointer(blocks, tableBytes, 8);
        Pointer(positionsBaseDev, 4, 4); Pointer(outputF32, bytes, 4); Pointer(workspace, workspaceBytes, 256);
        Separate(workspace, workspaceBytes, qF32, bytes); Separate(workspace, workspaceBytes, gateF32, bytes);
        Separate(workspace, workspaceBytes, blocks, tableBytes); Separate(workspace, workspaceBytes, positionsBaseDev, 4);
        Separate(workspace, workspaceBytes, outputF32, bytes);
        Separate(outputF32, bytes, qF32, bytes); Separate(outputF32, bytes, gateF32, bytes);
        Separate(outputF32, bytes, blocks, tableBytes); Separate(outputF32, bytes, positionsBaseDev, 4);
        Separate(positionsBaseDev, 4, qF32, bytes); Separate(positionsBaseDev, 4, gateF32, bytes);
        Separate(positionsBaseDev, 4, blocks, tableBytes);
        return plan;
    }
    internal static void ValidatePlaneOffsets(long kOff, long vOff)
    {
        if (kOff < 0 || vOff < 0 || kOff % 16 != 0 || vOff % 16 != 0
            || kOff > KvBlockBytes - KvPlaneBlockBytes || vOff > KvBlockBytes - KvPlaneBlockBytes
            || kOff < vOff + KvPlaneBlockBytes && vOff < kOff + KvPlaneBlockBytes)
            throw new ArgumentException("Prefill requires disjoint aligned BF16 K/V planes within a native cache block.");
    }
    private void Current()
    {
        ObjectDisposedException.ThrowIf(_disposed, this);
        CudaDriverApi.cuCtxGetCurrent(out var current).ThrowOnError();
        if (current != _context) throw new InvalidOperationException("Make the prefill module's owning CUDA context current.");
    }
    private static void Pointer(IntPtr pointer, long bytes, int alignment)
    {
        if (pointer.ToInt64() <= 0 || bytes <= 0 || pointer.ToInt64() % alignment != 0 || pointer.ToInt64() > long.MaxValue - bytes)
            throw new ArgumentException("Invalid or overflowing prefill buffer span/alignment.");
    }
    private static void Separate(IntPtr a, long aBytes, IntPtr b, long bBytes)
    {
        if (a.ToInt64() < b.ToInt64() + bBytes && b.ToInt64() < a.ToInt64() + aBytes)
            throw new ArgumentException("Prefill simultaneously live spans cannot overlap.");
    }
    private static void Scale(float scale)
    {
        if (!float.IsFinite(scale) || scale != 0.0625f) throw new ArgumentOutOfRangeException(nameof(scale), "HD256 attention uses exactly1/sqrt256.");
    }
    private Layout PagedAdmission(IntPtr input, int inputBytesPerElement, IntPtr blocks, int maxBlocks, long kOff, long vOff,
        int rows, int basePosition, float scale, IntPtr output, int outputBytesPerElement, IntPtr workspace, long workspaceBytes, int sms)
    {
        if (basePosition < 0 || basePosition > MaximumContext - rows) throw new ArgumentOutOfRangeException(nameof(basePosition));
        var plan = Plan(rows, checked(basePosition + rows), maxBlocks, sms); ValidatePlaneOffsets(kOff, vOff); Scale(scale);
        long inputBytes = checked((long)rows * Width * inputBytesPerElement), outputBytes = checked((long)rows * Width * outputBytesPerElement);
        if (workspaceBytes < plan.Bytes) throw new ArgumentException("Prefill workspace is too small.");
        Pointer(input, inputBytes, inputBytesPerElement == 2 ? 16 : 4); Pointer(output, outputBytes, 4);
        Pointer(blocks, (long)maxBlocks * 8, 8); Pointer(workspace, workspaceBytes, 256);
        Separate(workspace, workspaceBytes, input, inputBytes); Separate(workspace, workspaceBytes, output, outputBytes);
        Separate(workspace, workspaceBytes, blocks, (long)maxBlocks * 8); Separate(output, outputBytes, input, inputBytes);
        Separate(output, outputBytes, blocks, (long)maxBlocks * 8); Current(); return plan;
    }
    private void Pack(IntPtr input, IntPtr output, ulong count, IntPtr workspace, in Layout plan, IntPtr blocks,
        long kOff, long vOff, int basePosition, IntPtr stream)
    {
        IntPtr planes = workspace + (nint)plan.Planes, positions = workspace + (nint)plan.Positions, table = workspace + (nint)plan.Table;
        int pages = plan.Pages;
        void** args = stackalloc void*[] { &input, &output, &count, &planes, &blocks, &kOff, &vOff, &positions, &basePosition, &table, &pages };
        uint grid = checked((uint)((Math.Max(count, (ulong)pages) + 255) / 256));
        CudaDriverApi.cuLaunchKernel(_pack, grid, 1, 1, 256, 1, 1, 0, stream, (IntPtr)args, IntPtr.Zero).ThrowOnError();
    }
    internal void RunF32(IntPtr qF32, IntPtr gateF32, IntPtr blocks, int maxBlocks, long kOff, long vOff,
        int rows, int basePosition, float scale, IntPtr outputF32, IntPtr workspace, long workspaceBytes, int sms, IntPtr stream)
    {
        var plan = PagedAdmission(qF32, 4, blocks, maxBlocks, kOff, vOff, rows, basePosition, scale, outputF32, 4, workspace, workspaceBytes, sms);
        long bytes = checked((long)rows * Width * 4); Pointer(gateF32, bytes, 4);
        Separate(workspace, workspaceBytes, gateF32, bytes); Separate(outputF32, bytes, gateF32, bytes);
        IntPtr query = workspace + (nint)plan.Query, output = workspace + (nint)plan.Output;
        ulong count = checked((ulong)rows * Width);
        Pack(qF32, query, count, workspace, plan, blocks, kOff, vOff, basePosition, stream);
        Core(_owned, query, workspace + (nint)plan.Planes, workspace + (nint)(plan.Planes + 16), workspace + (nint)plan.Table,
            workspace + (nint)plan.Positions, output, rows, scale, plan.Splits, workspace + (nint)plan.PartialAcc,
            workspace + (nint)plan.PartialM, workspace + (nint)plan.PartialL, stream);
        void** args = stackalloc void*[] { &output, &gateF32, &outputF32, &count };
        CudaDriverApi.cuLaunchKernel(_gate, checked((uint)((count + 255) / 256)), 1, 1, 256, 1, 1, 0, stream, (IntPtr)args, IntPtr.Zero).ThrowOnError();
    }
    /// <summary>Module-only T8/B1/full-valid Verify candidate. Core causality reads
    /// positionsBaseDev directly on every graph replay. maxVisibleKeys is checked
    /// only on the host and never enters a captured launch argument. No allocation,
    /// synchronous readback or CPU-derived cache position occurs here.</summary>
    internal void RunF32Verify8(IntPtr qF32, IntPtr gateF32, IntPtr blocks, int maxBlocks, long kOff, long vOff,
        IntPtr positionsBaseDev, int maxVisibleKeys, float scale, IntPtr outputF32,
        IntPtr workspace, long workspaceBytes, IntPtr stream)
    {
        var plan = ValidateVerify8Buffers(qF32, gateF32, blocks, maxBlocks, kOff, vOff,
            positionsBaseDev, maxVisibleKeys, scale, outputF32, workspace, workspaceBytes);
        Current();
        IntPtr query = workspace + (nint)plan.Query, output = workspace + (nint)plan.Output;
        ulong count = (ulong)Verify8Rows * Width;
        // Pack's CPU position writes only the unused workspace.Positions cell.
        // Its page count is always640, including capture at a short history.
        Pack(qF32, query, count, workspace, plan, blocks, kOff, vOff, 0, stream);
        Core(_owned, query, workspace + (nint)plan.Planes, workspace + (nint)(plan.Planes + 16),
            workspace + (nint)plan.Table, positionsBaseDev, output, Verify8Rows, scale, Verify8Splits,
            workspace + (nint)plan.PartialAcc, workspace + (nint)plan.PartialM, workspace + (nint)plan.PartialL, stream);
        void** args = stackalloc void*[] { &output, &gateF32, &outputF32, &count };
        CudaDriverApi.cuLaunchKernel(_gate, checked((uint)((count + 255) / 256)), 1, 1, 256, 1, 1, 0,
            stream, (IntPtr)args, IntPtr.Zero).ThrowOnError();
    }
    internal void RunPagedBf16(IntPtr qBf16, IntPtr blocks, int maxBlocks, long kOff, long vOff,
        int rows, int basePosition, float scale, IntPtr outputBf16, IntPtr workspace, long workspaceBytes, int sms, IntPtr stream)
    {
        var plan = PagedAdmission(qBf16, 2, blocks, maxBlocks, kOff, vOff, rows, basePosition, scale, outputBf16, 2, workspace, workspaceBytes, sms);
        Pack(IntPtr.Zero, IntPtr.Zero, 0, workspace, plan, blocks, kOff, vOff, basePosition, stream);
        Core(_owned, qBf16, workspace + (nint)plan.Planes, workspace + (nint)(plan.Planes + 16), workspace + (nint)plan.Table,
            workspace + (nint)plan.Positions, outputBf16, rows, scale, plan.Splits, workspace + (nint)plan.PartialAcc,
            workspace + (nint)plan.PartialM, workspace + (nint)plan.PartialL, stream);
    }
    /// <summary>Unchanged original64-token/I32-page kernel, only for the independent
    /// component oracle. Caller owns complete contiguous physical cache planes,
    /// valid original page indices/positions and exact-sized partial allocations.</summary>
    internal void RunOriginalBf16(IntPtr qBf16, IntPtr cacheK, IntPtr cacheV, IntPtr table, IntPtr positions,
        IntPtr outputBf16, int rows, float scale, int splits, IntPtr partialAcc, IntPtr partialM, IntPtr partialL, IntPtr stream)
    {
        if (!Supports(rows, 1, false) || splits < 1 || splits > 4) throw new ArgumentOutOfRangeException(nameof(rows));
        RunOriginal(qBf16, cacheK, cacheV, table, positions, outputBf16, rows, scale, splits, partialAcc, partialM, partialL, stream);
    }
    /// <summary>Unchanged original64-page T8/S4 oracle only. This does not relax
    /// the ordinary-prefill API or provide engine dispatch/graph policy.</summary>
    internal void RunOriginalVerify8Bf16(IntPtr qBf16, IntPtr cacheK, IntPtr cacheV, IntPtr table, IntPtr positionsBaseDev,
        IntPtr outputBf16, float scale, IntPtr partialAcc, IntPtr partialM, IntPtr partialL, IntPtr stream)
        => RunOriginal(qBf16, cacheK, cacheV, table, positionsBaseDev, outputBf16, Verify8Rows, scale,
            Verify8Splits, partialAcc, partialM, partialL, stream);

    private void RunOriginal(IntPtr qBf16, IntPtr cacheK, IntPtr cacheV, IntPtr table, IntPtr positions,
        IntPtr outputBf16, int rows, float scale, int splits, IntPtr partialAcc, IntPtr partialM, IntPtr partialL, IntPtr stream)
    {
        Scale(scale); long elements = checked((long)rows * Width);
        Pointer(qBf16, elements * 2, 16); Pointer(cacheK, 16, 16); Pointer(cacheV, 16, 16);
        Pointer(table, 4, 4); Pointer(positions, 4, 4); Pointer(outputBf16, elements * 2, 4);
        if (splits > 1) { Pointer(partialAcc, elements * splits * 4, 4); Pointer(partialM, (long)rows * QueryHeads * splits * 4, 4); Pointer(partialL, (long)rows * QueryHeads * splits * 4, 4); }
        Current(); Core(_original, qBf16, cacheK, cacheV, table, positions, outputBf16, rows, scale, splits, partialAcc, partialM, partialL, stream);
    }
    private void Core(IntPtr kernel, IntPtr q, IntPtr k, IntPtr v, IntPtr table, IntPtr positions,
        IntPtr output, int rows, float scale, int splits, IntPtr partialAcc, IntPtr partialM, IntPtr partialL, IntPtr stream)
    {
        int zero = 0, limit = int.MaxValue; IntPtr none = IntPtr.Zero;
        void** args = stackalloc void*[] { &q, &k, &v, &table, &positions, &scale, &output, &rows,
            &zero, &zero, &limit, &none, &none, &none, &zero, &partialAcc, &partialM, &partialL, &splits };
        CudaDriverApi.cuLaunchKernel(kernel, checked((uint)((rows + 63) / 64)), QueryHeads, checked((uint)splits), 128, 1, 1,
            DynamicSharedBytes, stream, (IntPtr)args, IntPtr.Zero).ThrowOnError();
        if (kernel == _owned) PagedCoreLaunches++; else OriginalCoreLaunches++;
        if (splits <= 1) return;
        void** reduce = stackalloc void*[] { &partialAcc, &partialM, &partialL, &scale, &rows, &splits, &output };
        CudaDriverApi.cuLaunchKernel(_reduce, checked((uint)rows), QueryHeads, 1, 256, 1, 1, 0,
            stream, (IntPtr)reduce, IntPtr.Zero).ThrowOnError();
    }
    public void Dispose()
    {
        if (_disposed) return; Current(); _module.Dispose(); _disposed = true;
    }
}
