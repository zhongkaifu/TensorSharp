// Copyright (c) Zhongkai Fu. All rights reserved.
// Licensed under the BSD-3-Clause license in the repository root.
using System;
using System.IO;
using TensorSharp.Cuda.Interop;

namespace TensorSharp.Cuda;

/// <summary>
/// DFlash2 components of the NInfer v3 artifact. Activations, plain weights and cyclic KV are
/// BF16; projections are q8_g32 (INT8 codes, F16 group-32 scales) or BF16 on the small-T BF16
/// tensor-core kernels; the candidate codebooks are BF16.
/// Device metadata values must be validated by the owning scheduler before submission.
/// </summary>
internal sealed unsafe class CudaNinferDraftModule : IDisposable
{
    internal const string EnvironmentVariable = "TS_Q35_NINFER_DRAFT_PTX";
    private readonly CudaModule _module;
    // [tile 8/16/32]: q8_g32 projection, q8_g32 gate|up SwiGLU, BF16-weight projection.
    private readonly IntPtr[] _w8 = new IntPtr[3], _w8SwiGlu = new IntPtr[3], _bf16w = new IntPtr[3];
    private readonly IntPtr[] _swaPartial = new IntPtr[8], _swaDirect = new IntPtr[8], _swaReduce = new IntPtr[8];
    private readonly IntPtr _rms128, _rms5120, _conv, _residual, _tap, _pack, _cast, _prepare, _rope, _append;
    private readonly IntPtr _topPartial, _topMerge, _pred, _scores, _walk;
    private bool _disposed;
    internal string ImageSha256 => _module.ImageSha256;
    internal CudaNinferDraftModule(string path)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(path);
        _module = CudaModule.LoadFromFile(Path.GetFullPath(path), identify: true);
        try
        {
            for (int i = 0; i < 3; i++)
            {
                int tile = 8 << i;
                _w8[i] = Function($"ts_nidf_w8_t{tile}"); _w8SwiGlu[i] = Function($"ts_nidf_w8_swiglu_t{tile}"); _bf16w[i] = Function($"ts_nidf_bf16w_t{tile}");
            }
            for (int t = 1; t <= 8; t++)
            {
                _swaPartial[t - 1] = Function($"ts_nidf_swa_partial_t{t}");
                _swaDirect[t - 1] = Function($"ts_nidf_swa_direct_t{t}"); _swaReduce[t - 1] = Function($"ts_nidf_swa_reduce_t{t}");
            }
            _rms128 = Function("ts_nidf_rms128"); _rms5120 = Function("ts_nidf_rms5120");
            _conv = Function("ts_nidf_dynamic_conv"); _residual = Function("ts_nidf_residual_add");
            _tap = Function("ts_nidf_copy_target_tap"); _pack = Function("ts_nidf_pack_hidden");
            _cast = Function("ts_nidf_cast_bf16_f32"); _prepare = Function("ts_nidf_prepare_block");
            _rope = Function("ts_nidf_rope"); _append = Function("ts_nidf_ring_append");
            _topPartial = Function("ts_nidf_top16_partial"); _topMerge = Function("ts_nidf_top16_merge");
            _pred = Function("ts_nidf_selector_predecessors"); _scores = Function("ts_nidf_selector_scores_bf16"); _walk = Function("ts_nidf_selector_walk");
        }
        catch { _module.Dispose(); throw; }
    }
    private IntPtr Function(string name) => _module.GetFunction(name);

    // Warps per CTA of each small-T instance (tensorsharp_q35_ninfer_draft.cu TS_NIDF_SMALL).
    private static int Tile(int tokens) => tokens <= 8 ? 0 : tokens <= 16 ? 1 : 2;
    private static int Threads(int tile, bool bf16Weights) => (tile == 0 || (tile == 1 && !bf16Weights) ? 4 : 2) * 32;
    /// <summary>Weight rows per CTA (16 x RT) of each instance.</summary>
    private static int WeightRows(int tile, bool bf16Weights) => tile == 0 ? 16 : 32;

    /// <summary>y[tokens][n] = x W^T for a q8_g32 (scales: F16 [n][k/32]) or BF16-weight matrix,
    /// BF16 activations and outputs; any token count (32-row grid tiles).</summary>
    internal void ProjectSmall(bool w8, IntPtr weights, IntPtr scales, IntPtr input, IntPtr output, int n, int k, int tokens, IntPtr stream)
    {
        int tile = Tile(tokens), tt = 8 << tile, rowsPerCta = WeightRows(tile, !w8);
        if (n % rowsPerCta != 0 || k % 256 != 0 || tokens < 1) throw new ArgumentOutOfRangeException(nameof(n), "Draft projections need whole weight-row tiles and k % 256 == 0.");
        long wb = (long)n * k * (w8 ? 1 : 2), ib = (long)tokens * k * 2, ob = (long)tokens * n * 2;
        Pointer(weights, wb, 16); Pointer(input, ib, 16); Pointer(output, ob, 2);
        Separate(output, ob, weights, wb); Separate(output, ob, input, ib);
        void** args = stackalloc void*[] { &weights, &scales, &input, &output, &n, &k, &tokens, &n };
        Launch(w8 ? _w8[tile] : _bf16w[tile], n / rowsPerCta, (tokens + tt - 1) / tt, 1, Threads(tile, !w8), 0, stream, args);
    }

    /// <summary>BF16(silu(gate) * up) [tokens][n / 2] from a q8_g32 gate|up matrix interleaved by 8 rows.</summary>
    internal void SwiGluW8(IntPtr weights, IntPtr scales, IntPtr input, IntPtr output, int n, int k, int tokens, IntPtr stream)
    {
        if (n % 16 != 0 || k % 256 != 0 || tokens < 1) throw new ArgumentOutOfRangeException(nameof(n));
        long wb = (long)n * k, ib = (long)tokens * k * 2, ob = (long)tokens * (n / 2) * 2;
        Pointer(weights, wb, 16); Pointer(input, ib, 16); Pointer(output, ob, 2);
        Separate(output, ob, weights, wb); Separate(output, ob, input, ib);
        int tile = Tile(tokens), tt = 8 << tile, ldy = n / 2, rowsPerCta = WeightRows(tile, false);
        if (n % rowsPerCta != 0) throw new ArgumentOutOfRangeException(nameof(n));
        void** args = stackalloc void*[] { &weights, &scales, &input, &output, &n, &k, &tokens, &ldy };
        Launch(_w8SwiGlu[tile], n / rowsPerCta, (tokens + tt - 1) / tt, 1, Threads(tile, false), 0, stream, args);
    }

    internal void PlainRms(IntPtr input, IntPtr weight, IntPtr output, int width, int rows, float eps, IntPtr stream)
    {
        if (width != 128 && width != 5120 || rows < 1 || !float.IsFinite(eps) || eps <= 0) throw new ArgumentOutOfRangeException(nameof(width));
        long bytes = checked((long)width * rows * 2); Pointer(input, bytes, 4); Pointer(weight, width * 2L, 4); Pointer(output, bytes, 4);
        Separate(output, bytes, input, bytes); Separate(output, bytes, weight, width * 2L);
        void** args = stackalloc void*[] { &input, &weight, &output, &rows, &eps };
        Launch(width == 128 ? _rms128 : _rms5120, width == 128 ? checked((rows + 3) / 4) : rows, 1, 1, width == 128 ? 128 : 512, 0, stream, args);
    }
    internal void DynamicConv(IntPtr input, IntPtr dynamic, IntPtr basis, IntPtr output, int hidden, int block, int lanes, int side, IntPtr stream)
    {
        Block(block, lanes); if (hidden < 16 || hidden > 5120 || hidden % 16 != 0 || (uint)side > 1) throw new ArgumentOutOfRangeException(nameof(hidden));
        int columns = checked(block * lanes); long bytes = checked((long)hidden * columns * 2), db = checked((long)hidden / 16 * 4 * columns * 2), bb = hidden * 4L * 2;
        Pointer(input, bytes, 2); Pointer(dynamic, db, 2); Pointer(basis, bb, 2); Pointer(output, bytes, 2);
        Separate(output, bytes, input, bytes); Separate(output, bytes, dynamic, db); Separate(output, bytes, basis, bb);
        void** args = stackalloc void*[] { &input, &dynamic, &basis, &hidden, &columns, &block, &side, &output };
        Launch(_conv, (int)Math.Min(65535, ((long)hidden * columns + 255) / 256), 1, 1, 256, 0, stream, args);
    }
    internal void ResidualAdd(IntPtr input, IntPtr residual, long count, IntPtr stream)
    {
        long bytes = ElementBytes(count, 2); Pointer(input, bytes, 2); Pointer(residual, bytes, 2); Separate(input, bytes, residual, bytes);
        void** args = stackalloc void*[] { &input, &residual, &count }; Launch(_residual, Grid(count), 1, 1, 256, 0, stream, args);
    }
    internal void CopyTargetTap(IntPtr inputF32, IntPtr concatenatedBf16, int rows, int tap, IntPtr stream, int inputStride = 5120, int outputStride = 25600)
    {
        if (rows < 1 || (uint)tap >= 5 || inputStride < 5120 || outputStride < 25600) throw new ArgumentOutOfRangeException(nameof(tap));
        long ib = checked(((long)(rows - 1) * inputStride + 5120) * 4), ob = checked(((long)(rows - 1) * outputStride + 25600) * 2);
        Pointer(inputF32, ib, 4); Pointer(concatenatedBf16, ob, 2); Separate(inputF32, ib, concatenatedBf16, ob);
        void** args = stackalloc void*[] { &inputF32, &concatenatedBf16, &rows, &tap, &inputStride, &outputStride };
        Launch(_tap, 20, rows, 1, 256, 0, stream, args);
    }
    internal void PackDraftHidden(IntPtr input, IntPtr packed, int block, int lanes, IntPtr stream)
    {
        Block(block, lanes); if (block < 2) throw new ArgumentOutOfRangeException(nameof(block));
        long count = checked((long)lanes * (block - 1) * 5120), ib = (long)lanes * block * 5120 * 2, ob = count * 2;
        Pointer(input, ib, 2); Pointer(packed, ob, 2); Separate(input, ib, packed, ob);
        void** args = stackalloc void*[] { &input, &packed, &block, &lanes }; Launch(_pack, Grid(count), 1, 1, 256, 0, stream, args);
    }
    internal void CastBf16ToF32(IntPtr input, IntPtr output, long count, IntPtr stream)
    {
        long ib = ElementBytes(count, 2), ob = checked(count * 4); Pointer(input, ib, 2); Pointer(output, ob, 4); Separate(input, ib, output, ob);
        void** args = stackalloc void*[] { &input, &output, &count }; Launch(_cast, Grid(count), 1, 1, 256, 0, stream, args);
    }
    internal void PrepareBlock(IntPtr anchors, IntPtr basePositions, IntPtr valid, IntPtr ids, IntPtr positions, int block, int lanes, IntPtr stream)
    {
        Block(block, lanes); long mb = lanes * 4L, ob = block * lanes * 4L;
        Pointer(anchors, mb, 4); Pointer(basePositions, mb, 4); Pointer(valid, mb, 4); Pointer(ids, ob, 4); Pointer(positions, ob, 4);
        Separate(ids, ob, positions, ob); Separate(ids, ob, anchors, mb); Separate(ids, ob, basePositions, mb); Separate(ids, ob, valid, mb);
        Separate(positions, ob, anchors, mb); Separate(positions, ob, basePositions, mb); Separate(positions, ob, valid, mb);
        int vocab = 248320; void** args = stackalloc void*[] { &anchors, &basePositions, &valid, &ids, &positions, &block, &vocab };
        Launch(_prepare, lanes, 1, 1, 32, 0, stream, args);
    }
    internal static double[] RopeFrequencies()
    {
        var result = new double[64]; for (int i = 0; i < result.Length; i++) result[i] = Math.Pow(10000000.0, -2.0 * i / 128.0); return result;
    }
    internal void QkRope(IntPtr query, IntPtr key, IntPtr positions, IntPtr inverseFrequencies, int tokens, IntPtr stream)
    {
        if (tokens < 1 || query == IntPtr.Zero && key == IntPtr.Zero) throw new ArgumentOutOfRangeException(nameof(tokens));
        long qb = checked((long)tokens * 4096 * 2), kb = checked((long)tokens * 1024 * 2);
        if (query != IntPtr.Zero) Pointer(query, qb, 4); if (key != IntPtr.Zero) Pointer(key, kb, 4);
        Pointer(positions, tokens * 4L, 4); Pointer(inverseFrequencies, 512, 8);
        if (query != IntPtr.Zero && key != IntPtr.Zero) Separate(query, qb, key, kb);
        if (query != IntPtr.Zero) { Separate(query, qb, positions, tokens * 4L); Separate(query, qb, inverseFrequencies, 512); }
        if (key != IntPtr.Zero) { Separate(key, kb, positions, tokens * 4L); Separate(key, kb, inverseFrequencies, 512); }
        void** args = stackalloc void*[] { &query, &key, &positions, &inverseFrequencies, &tokens }; Launch(_rope, tokens, 40, 1, 64, 0, stream, args);
    }

    internal void RingAppend(IntPtr key, IntPtr value, IntPtr positions, IntPtr counts, IntPtr lanes,
        IntPtr ringKey, IntPtr ringValue, int width, int batch, int capacitySlots, IntPtr stream)
    {
        if (width < 1 || width > 2048 || batch < 1 || batch > 8 || capacitySlots < batch || capacitySlots > 16) throw new ArgumentOutOfRangeException(nameof(width));
        long ib = (long)width * batch * 1024 * 2, rb = (long)capacitySlots * 2048 * 1024 * 2, pb = (long)width * batch * 4, mb = batch * 4L;
        Pointer(key, ib, 16); Pointer(value, ib, 16); Pointer(positions, pb, 4); Pointer(counts, mb, 4); Pointer(lanes, mb, 4);
        Pointer(ringKey, rb, 16); Pointer(ringValue, rb, 16); Separate(ringKey, rb, ringValue, rb);
        Separate(ringKey, rb, key, ib); Separate(ringKey, rb, value, ib); Separate(ringValue, rb, key, ib); Separate(ringValue, rb, value, ib);
        Separate(ringKey, rb, positions, pb); Separate(ringKey, rb, counts, mb); Separate(ringKey, rb, lanes, mb);
        Separate(ringValue, rb, positions, pb); Separate(ringValue, rb, counts, mb); Separate(ringValue, rb, lanes, mb);
        void** args = stackalloc void*[] { &key, &value, &positions, &counts, &lanes, &ringKey, &ringValue, &width, &batch, &capacitySlots };
        Launch(_append, Grid(ib / 2), 1, 1, 256, 0, stream, args);
    }

    internal readonly record struct SwaLayout(bool Direct, int Splits, long QueryBytes, long KeyValueBytes, long RingBytes, long PartialBytes, long StatBytes);
    internal static SwaLayout ValidateSwa(int maxContext, int block, int batch, int capacitySlots)
    {
        Block(block, batch); if (maxContext < 0 || maxContext > int.MaxValue - block || capacitySlots < batch || capacitySlots > 16) throw new ArgumentOutOfRangeException(nameof(maxContext));
        bool direct = maxContext <= 96; int splits = direct ? 1 : Math.Min(32, Math.Max(1, (Math.Min(maxContext, 2047) + 31) / 32));
        long qb = (long)block * batch * 4096 * 2;
        return new(direct, splits, qb, (long)block * batch * 1024 * 2, (long)capacitySlots * 2048 * 1024 * 2,
            direct ? 0 : qb * splits, direct ? 0 : (long)block * batch * 32 * splits * 4);
    }
    internal void Swa(IntPtr query, IntPtr key, IntPtr value, IntPtr positions, IntPtr valid, IntPtr lanes,
        IntPtr ringKey, IntPtr ringValue, IntPtr partial, IntPtr maxima, IntPtr denominators, IntPtr output,
        int maxContext, int block, int batch, int capacitySlots, IntPtr stream)
    {
        var p = ValidateSwa(maxContext, block, batch, capacitySlots);
        Pointer(query, p.QueryBytes, 16); Pointer(key, p.KeyValueBytes, 16); Pointer(value, p.KeyValueBytes, 16); Pointer(output, p.QueryBytes, 16);
        Pointer(positions, (long)block * batch * 4, 4); Pointer(valid, batch * 4L, 4); Pointer(lanes, batch * 4L, 4);
        Pointer(ringKey, p.RingBytes, 16); Pointer(ringValue, p.RingBytes, 16);
        Separate(output, p.QueryBytes, query, p.QueryBytes); Separate(output, p.QueryBytes, key, p.KeyValueBytes); Separate(output, p.QueryBytes, value, p.KeyValueBytes);
        Separate(output, p.QueryBytes, ringKey, p.RingBytes); Separate(output, p.QueryBytes, ringValue, p.RingBytes);
        if (!p.Direct)
        {
            Pointer(partial, p.PartialBytes, 16); Pointer(maxima, p.StatBytes, 4); Pointer(denominators, p.StatBytes, 4);
            Separate(partial, p.PartialBytes, output, p.QueryBytes); Separate(partial, p.PartialBytes, maxima, p.StatBytes); Separate(partial, p.PartialBytes, denominators, p.StatBytes);
            Separate(maxima, p.StatBytes, denominators, p.StatBytes); Separate(output, p.QueryBytes, maxima, p.StatBytes); Separate(output, p.QueryBytes, denominators, p.StatBytes);
            SwaScratchSeparate(partial, p.PartialBytes, p, query, key, value, ringKey, ringValue, positions, valid, lanes, block, batch);
            SwaScratchSeparate(maxima, p.StatBytes, p, query, key, value, ringKey, ringValue, positions, valid, lanes, block, batch);
            SwaScratchSeparate(denominators, p.StatBytes, p, query, key, value, ringKey, ringValue, positions, valid, lanes, block, batch);
        }
        Separate(output, p.QueryBytes, positions, (long)block * batch * 4); Separate(output, p.QueryBytes, valid, batch * 4L); Separate(output, p.QueryBytes, lanes, batch * 4L);
        int splits = p.Splits;
        void** args = stackalloc void*[] { &query, &key, &value, &positions, &valid, &lanes, &ringKey, &ringValue, &maxContext, &splits, &partial, &maxima, &denominators, &output };
        Launch(p.Direct ? _swaDirect[block - 1] : _swaPartial[block - 1], 8, splits, batch, (block + 3) / 4 * 32, 16384, stream, args);
        if (!p.Direct)
        {
            void** reduceArgs = stackalloc void*[] { &partial, &maxima, &denominators, &positions, &valid, &maxContext, &splits, &output };
            Launch(_swaReduce[block - 1], block * 32, 1, batch, 32, 0, stream, reduceArgs);
        }
    }
    private static void SwaScratchSeparate(IntPtr ptr, long bytes, SwaLayout p, IntPtr query, IntPtr key, IntPtr value,
        IntPtr ringKey, IntPtr ringValue, IntPtr positions, IntPtr valid, IntPtr lanes, int block, int batch)
    {
        Separate(ptr, bytes, query, p.QueryBytes); Separate(ptr, bytes, key, p.KeyValueBytes); Separate(ptr, bytes, value, p.KeyValueBytes);
        Separate(ptr, bytes, ringKey, p.RingBytes); Separate(ptr, bytes, ringValue, p.RingBytes);
        Separate(ptr, bytes, positions, (long)block * batch * 4); Separate(ptr, bytes, valid, batch * 4L); Separate(ptr, bytes, lanes, batch * 4L);
    }

    internal static long Top16WorkspaceBytes(int vocabulary, int columns)
    {
        if (vocabulary < 16 || vocabulary > 248320 || columns < 1 || columns > 56) throw new ArgumentOutOfRangeException(nameof(vocabulary));
        return checked((long)TopSplits(vocabulary) * columns * 16 * 8);
    }
    private static int TopSplits(int vocabulary) => Math.Clamp((vocabulary + 2047) / 2048, 1, 1024);
    internal void Top16(IntPtr logits, IntPtr ids, IntPtr unary, int vocabulary, int columns, IntPtr workspace, IntPtr stream)
    {
        long wb = Top16WorkspaceBytes(vocabulary, columns), lb = (long)vocabulary * columns * 2, ob = columns * 16L * 4;
        Pointer(logits, lb, 2); Pointer(ids, ob, 4); Pointer(unary, ob, 4); Pointer(workspace, wb, 8);
        Separate(ids, ob, unary, ob); Separate(ids, ob, logits, lb); Separate(unary, ob, logits, lb);
        Separate(workspace, wb, logits, lb); Separate(workspace, wb, ids, ob); Separate(workspace, wb, unary, ob);
        int splits = TopSplits(vocabulary);
        void** first = stackalloc void*[] { &logits, &vocabulary, &splits, &workspace }; Launch(_topPartial, splits, columns, 1, 128, 0, stream, first);
        void** second = stackalloc void*[] { &workspace, &logits, &vocabulary, &splits, &ids, &unary }; Launch(_topMerge, columns, 1, 1, 128, 0, stream, second);
    }
    internal void SelectorPredecessors(IntPtr candidates, IntPtr anchors, IntPtr output, int positions, int lanes, IntPtr stream)
    {
        SelectorShape(positions, lanes); int top = 16; long bytes = (long)top * positions * lanes * 4;
        Pointer(candidates, bytes, 4); Pointer(anchors, lanes * 4L, 4); Pointer(output, bytes, 4); Separate(output, bytes, candidates, bytes); Separate(output, bytes, anchors, lanes * 4L);
        void** args = stackalloc void*[] { &candidates, &anchors, &top, &positions, &lanes, &output }; Launch(_pred, Grid(bytes / 4), 1, 1, 256, 0, stream, args);
    }
    internal void SelectorScoresBf16(IntPtr candidates, IntPtr predecessors, IntPtr unary, IntPtr hidden,
        IntPtr successor, IntPtr predecessor, IntPtr output, int positions, int lanes, IntPtr stream)
    {
        SelectorShape(positions, lanes);
        int rank = 256, top = 16; long cb = 248320L * rank * 2, ib = (long)positions * lanes * 16 * 4, hb = (long)positions * lanes * 256 * 4, ob = ib * 16;
        Pointer(candidates, ib, 4); Pointer(predecessors, ib, 4); Pointer(unary, ib, 4); Pointer(hidden, hb, 4);
        Pointer(successor, cb, 2); Pointer(predecessor, cb, 2); Pointer(output, ob, 4);
        Separate(output, ob, candidates, ib); Separate(output, ob, predecessors, ib); Separate(output, ob, unary, ib); Separate(output, ob, hidden, hb);
        void** args = stackalloc void*[] { &candidates, &predecessors, &unary, &hidden, &successor, &predecessor, &rank, &top, &positions, &lanes, &output };
        Launch(_scores, checked(16 * 16 * positions * lanes), 1, 1, 256, 0, stream, args);
    }
    internal void SelectorWalk(IntPtr scores, IntPtr candidates, IntPtr output, int positions, int lanes, IntPtr stream)
    {
        SelectorShape(positions, lanes); int top = 16; long ib = (long)positions * lanes * top * 4, ob = (long)positions * lanes * 4;
        Pointer(scores, ib * top, 4); Pointer(candidates, ib, 4); Pointer(output, ob, 4); Separate(output, ob, scores, ib * top); Separate(output, ob, candidates, ib);
        void** args = stackalloc void*[] { &scores, &candidates, &top, &positions, &lanes, &output }; Launch(_walk, 1, 1, 1, 64, 0, stream, args);
    }

    private void Launch(IntPtr function, int gx, int gy, int gz, int threads, uint shared, IntPtr stream, void** args)
    {
        ObjectDisposedException.ThrowIf(_disposed, this);
        CudaDriverApi.cuLaunchKernel(function, checked((uint)gx), checked((uint)gy), checked((uint)gz), checked((uint)threads), 1, 1, shared, stream, (IntPtr)args, IntPtr.Zero).ThrowOnError();
    }
    private static void Block(int block, int lanes) { if (block < 1 || block > 8 || lanes < 1 || lanes > 8) throw new ArgumentOutOfRangeException(nameof(block)); }
    private static void SelectorShape(int positions, int lanes) { if (positions < 1 || positions > 7 || lanes < 1 || lanes > 8) throw new ArgumentOutOfRangeException(nameof(positions)); }
    private static void Factor(float value) { if (!float.IsFinite(value) || value <= 0) throw new ArgumentOutOfRangeException(nameof(value)); }
    private static long ElementBytes(long count, int width) { if (count < 1 || count > (long)int.MaxValue * 256) throw new ArgumentOutOfRangeException(nameof(count)); return checked(count * width); }
    private static int Grid(long count) => checked((int)((count + 255) / 256));
    private static void Pointer(IntPtr p, long bytes, int alignment)
    {
        if (bytes <= 0 || p.ToInt64() <= 0 || p.ToInt64() % alignment != 0 || (ulong)p.ToInt64() > (ulong)long.MaxValue - (ulong)bytes)
            throw new ArgumentException("Original draft component requires an aligned, nonoverflowing device range.");
    }
    private static void Separate(IntPtr a, long ab, IntPtr b, long bb)
    {
        if (a.ToInt64() < b.ToInt64() + bb && b.ToInt64() < a.ToInt64() + ab) throw new ArgumentException("Original draft output/scratch must not alias its inputs.");
    }
    public void Dispose() { if (_disposed) return; _disposed = true; _module.Dispose(); }
}
