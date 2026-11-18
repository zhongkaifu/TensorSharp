// Copyright (c) Zhongkai Fu. All rights reserved.
// https://github.com/zhongkaifu/TensorSharp
//
// This file is part of TensorSharp.
//
// TensorSharp is licensed under the BSD-3-Clause license found in the LICENSE file in the root directory of this source tree.
using System;
using System.Linq;
using System.Runtime.InteropServices;
using System.Threading.Tasks;
using TensorSharp;
using TensorSharp.Cuda;
using Xunit;
using Xunit.Abstractions;

namespace InferenceWeb.Tests;

/// <summary>
/// The 8-bit prompt attention (tensorsharp_flash_prefill's ts_flash_prompt_band_f16_d256 + ts_flash_prompt_i8e4_d256), on an
/// RTX Blackwell image (skipped elsewhere): within its precision budget of an F64 attention over Qwen3.8-27B's prompt shapes
/// (24 query heads over 4 KV heads of 256) and others (a floor, GQA groups of 4 and 8, a split launch), on Gaussian data, keys
/// with outlier channels and peaked rows; deterministic; its key splits agreeing with the unsplit launch; a row's result
/// independent of where its chunk ends outside the last partial key tile; and its band exactly what the band's definition says.
/// </summary>
public sealed class CudaFlashPromptAttention8Tests
{
    private readonly ITestOutputHelper _output;

    public CudaFlashPromptAttention8Tests(ITestOutputHelper output) => _output = output;

    private const int D = 256;

    private static IntPtr P(Tensor t) => ((CudaStorage)t.Storage).DevicePtrAtElement(t.StorageOffset);

    /// <summary>How the operands are drawn.</summary>
    public enum Data
    {
        /// <summary>q, k, v ~ N(0, 1).</summary>
        Gaussian,
        /// <summary>Gaussian, four K channels 20 times larger (per-key INT8 scales set by them).</summary>
        KeyOutliers,
        /// <summary>Gaussian plus a direction every query shares, which eight evenly spaced keys hold 12-fold: those keys score
        /// about 12 above the rest (a peaked softmax).</summary>
        Peaked,
    }

    /// <summary>One sequence's operands: q [heads][nq][256] F32 (host and device), the head-first F16 caches [kvHeads][cap][256]
    /// (host values exactly the F16 ones).</summary>
    private sealed class Chunk : IDisposable
    {
        public readonly float[] Q, K, V;
        public readonly Tensor Qt, Kt, Vt;
        public readonly int Heads, KvHeads, Nq, Len, Floor, Cap;

        public Chunk(CudaAllocator allocator, int heads, int kvHeads, int nq, int len, int floor, Data data, int seed = 0)
            : this(allocator, heads, kvHeads, nq, len, floor, Draw(heads, kvHeads, nq, len + 29, len, data, seed)) { }

        private Chunk(CudaAllocator allocator, int heads, int kvHeads, int nq, int len, int floor, (float[] Q, float[] K, float[] V) host)
        {
            (Heads, KvHeads, Nq, Len, Floor, Cap) = (heads, kvHeads, nq, len, floor, len + 29);
            (Q, K, V) = host;
            Qt = Tensor.FromArray(allocator, Q);
            Kt = ToHalf(allocator, K);
            Vt = ToHalf(allocator, V);
            ((CudaStorage)Qt.Storage).EnsureDeviceCurrent();
        }

        /// <summary>A chunk of rows [first, first + nq) of a sequence whose operands <paramref name="whole"/> holds (its caches
        /// shared; its q the rows' own), attending keys [floor, first + nq).</summary>
        public Chunk(CudaAllocator allocator, Chunk whole, int first, int nq)
        {
            (Heads, KvHeads, Nq, Len, Floor, Cap) = (whole.Heads, whole.KvHeads, nq, first + nq, whole.Floor, whole.Cap);
            K = whole.K;
            V = whole.V;
            Q = new float[Heads * nq * D];
            for (int h = 0; h < Heads; h++)
                Array.Copy(whole.Q, ((long)h * whole.Nq + first) * D, Q, (long)h * nq * D, (long)nq * D);
            Qt = Tensor.FromArray(allocator, Q);
            ((CudaStorage)Qt.Storage).EnsureDeviceCurrent();
            Kt = whole.Kt.CopyRef();
            Vt = whole.Vt.CopyRef();
        }

        private static Tensor ToHalf(CudaAllocator allocator, float[] values)
        {
            using var f32 = Tensor.FromArray(allocator, values);
            var f16 = new Tensor(allocator, DType.Float16, values.Length);
            ((CudaStorage)f32.Storage).EnsureDeviceCurrent();
            ((CudaStorage)f16.Storage).EnsureDeviceCurrent();
            allocator.Context.MakeCurrent();
            allocator.Kernels.LaunchConvertF32F16(P(f32), P(f16), values.Length, allocator.Stream.Handle);
            allocator.Stream.Synchronize();
            ((CudaStorage)f16.Storage).MarkDeviceModified();
            return f16;
        }

        private static (float[] Q, float[] K, float[] V) Draw(int heads, int kvHeads, int nq, int cap, int len, Data data, int seed)
        {
            var rng = new Random(heads * 7919 + nq * 31 + len + (int)data * 104729 + seed);
            double Gauss()
            {
                double u = 1.0 - rng.NextDouble(), w = rng.NextDouble();
                return Math.Sqrt(-2.0 * Math.Log(u)) * Math.Cos(2.0 * Math.PI * w);
            }
            var q = new float[heads * nq * D];
            var k = new float[kvHeads * cap * D];
            var v = new float[kvHeads * cap * D];
            for (int i = 0; i < q.Length; i++) q[i] = (float)Gauss();
            for (int i = 0; i < k.Length; i++) k[i] = (float)Gauss();
            for (int i = 0; i < v.Length; i++) v[i] = (float)Gauss();
            if (data == Data.KeyOutliers)
                foreach (int c in new[] { 3, 77, 150, 201 })
                    for (int r = 0; r < kvHeads * cap; r++) k[(long)r * D + c] *= 20f;
            if (data == Data.Peaked)
            {
                var e = new double[D];
                for (int x = 0; x < D; x++) e[x] = Gauss();
                double norm = Math.Sqrt(e.Sum(x => x * x));
                for (int x = 0; x < D; x++) e[x] /= norm;
                for (int r = 0; r < heads * nq; r++)
                    for (int x = 0; x < D; x++) q[(long)r * D + x] += (float)(16 * e[x]);
                int every = Math.Max(1, len / 8);
                for (int h = 0; h < kvHeads; h++)
                    for (int j = every - 1; j < len; j += every)
                        for (int x = 0; x < D; x++) k[((long)h * cap + j) * D + x] = (float)(12 * e[x]);
            }
            for (int i = 0; i < k.Length; i++) k[i] = (float)(System.Half)k[i];
            for (int i = 0; i < v.Length; i++) v[i] = (float)(System.Half)v[i];
            return (q, k, v);
        }

        /// <summary>The 8-bit launch (its band in a scratch tensor), out [nq][heads][256].</summary>
        public float[] Run8(CudaAllocator allocator, CudaFlashPrefillKernels flash, int splits, Action<byte[]> band = null)
        {
            int bandKeys = CudaFlashPrefillKernels.Attn8BandKeys(Len);
            long bytes = CudaFlashPrefillKernels.Attn8BandBytes(bandKeys, KvHeads);
            using var bandT = new Tensor(allocator, DType.Float32, bytes / 4);
            using var outT = new Tensor(allocator, DType.Float32, (long)Nq * Heads * D);
            using Tensor partial = splits > 1 ? new Tensor(allocator, DType.Float32, CudaFlashPrefillKernels.PartialFloats(Nq, Heads, D, splits)) : null;
            foreach (Tensor t in new[] { bandT, outT, partial }) if (t != null) ((CudaStorage)t.Storage).EnsureDeviceCurrent();
            allocator.Context.MakeCurrent();
            flash.LaunchPromptAttention8(P(Qt), P(Kt), P(Vt), Cap, Nq, Len, Floor, Heads, KvHeads, 1f / MathF.Sqrt(D), splits, P(bandT), bandKeys,
                partial == null ? IntPtr.Zero : P(partial), P(outT), allocator.Stream.Handle);
            allocator.Stream.Synchronize();
            ((CudaStorage)outT.Storage).MarkDeviceModified();
            if (band != null)
            {
                ((CudaStorage)bandT.Storage).MarkDeviceModified();
                band(MemoryMarshal.AsBytes(bandT.GetElementsAsFloat((int)(bytes / 4)).AsSpan()).ToArray());
            }
            return outT.GetElementsAsFloat(Nq * Heads * D);
        }

        /// <summary>The 16-bit prompt kernel over the same rows.</summary>
        public float[] Run16(CudaAllocator allocator, CudaFlashPrefillKernels flash, int splits)
        {
            using var outT = new Tensor(allocator, DType.Float32, (long)Nq * Heads * D);
            using Tensor partial = splits > 1 ? new Tensor(allocator, DType.Float32, CudaFlashPrefillKernels.PartialFloats(Nq, Heads, D, splits)) : null;
            foreach (Tensor t in new[] { outT, partial }) if (t != null) ((CudaStorage)t.Storage).EnsureDeviceCurrent();
            allocator.Context.MakeCurrent();
            flash.LaunchAttention(P(Qt), P(Kt), P(Vt), Cap, Nq, Len, Floor, Heads, KvHeads, D, 1f / MathF.Sqrt(D), splits,
                partial == null ? IntPtr.Zero : P(partial), P(outT), allocator.Stream.Handle, prompt: true);
            allocator.Stream.Synchronize();
            ((CudaStorage)outT.Storage).MarkDeviceModified();
            return outT.GetElementsAsFloat(Nq * Heads * D);
        }

        /// <summary>F64 attention of row t (all heads), [heads][256].</summary>
        public double[] Reference(int t)
        {
            int G = Heads / KvHeads, end = Len - Nq + t + 1;
            var result = new double[Heads * D];
            var s = new double[end];
            for (int h = 0; h < Heads; h++)
            {
                int kh = h / G;
                long qo = ((long)h * Nq + t) * D;
                double max = double.NegativeInfinity;
                for (int j = Floor; j < end; j++)
                {
                    long ko = ((long)kh * Cap + j) * D;
                    double dot = 0;
                    for (int x = 0; x < D; x++) dot += (double)Q[qo + x] * K[ko + x];
                    s[j] = dot / Math.Sqrt(D);
                    max = Math.Max(max, s[j]);
                }
                double sum = 0;
                for (int j = Floor; j < end; j++) sum += s[j] = Math.Exp(s[j] - max);
                for (int j = Floor; j < end; j++)
                {
                    long vo = ((long)kh * Cap + j) * D;
                    double p = s[j] / sum;
                    for (int x = 0; x < D; x++) result[h * D + x] += p * V[vo + x];
                }
            }
            return result;
        }

        public void Dispose()
        {
            foreach (Tensor t in new[] { Qt, Kt, Vt }) t?.Dispose();
        }
    }

    /// <summary>Relative RMS of a against b over the given rows (all heads), and the largest absolute difference.</summary>
    private static (double Rel, double MaxAbs) Compare(float[] a, Func<int, double[]> b, int[] rows, int heads)
    {
        var sums = new (double Diff, double Norm, double Max)[rows.Length];
        Parallel.For(0, rows.Length, i =>
        {
            double[] reference = b(rows[i]);
            double diff = 0, norm = 0, max = 0;
            for (int x = 0; x < heads * D; x++)
            {
                double e = a[(long)rows[i] * heads * D + x] - reference[x];
                diff += e * e;
                norm += reference[x] * reference[x];
                max = Math.Max(max, Math.Abs(e));
            }
            sums[i] = (diff, norm, max);
        });
        return (Math.Sqrt(sums.Sum(s => s.Diff) / sums.Sum(s => s.Norm)), sums.Max(s => s.Max));
    }

    private static double[] Row(float[] values, int t, int heads)
        => values.AsSpan(t * heads * D, heads * D).ToArray().Select(x => (double)x).ToArray();

    /// <summary>The rows an F64 reference is computed for: the first and last eight and 48 spread between.</summary>
    private static int[] SampledRows(int nq)
        => Enumerable.Range(0, Math.Min(8, nq)).Concat(Enumerable.Range(Math.Max(0, nq - 8), Math.Min(8, nq)))
            .Concat(Enumerable.Range(0, 48).Select(i => (int)((long)i * nq / 48))).Distinct().OrderBy(r => r).ToArray();

    private static (CudaAllocator Allocator, CudaFlashPrefillKernels Flash) Open()
    {
        var allocator = new CudaAllocator();
        CudaFlashPrefillKernels flash = allocator.FlashPrefillKernels;
        Assert.True(flash != null && flash.Runs8BitPrompt(D), "the gate saw the 8-bit prompt attention; this allocator's module has none");
        return (allocator, flash);
    }

    /// <summary>
    /// Within the 8-bit arithmetic's precision budget of an F64 attention, on every shape and data (the 16-bit kernel's own error
    /// printed beside it). The budget is the probabilities': P in E4M3 keeps 3 mantissa bits, about 3.6% relative RMS rounding a
    /// probability, which a zero-mean random V passes to the output nearly undiminished (2.9-3.9% measured on Gaussian data at
    /// every shape on an RTX PRO 4500; the 16-bit kernel 0.04%), and a key's INT8 scale set by outlier channels coarsens its other
    /// channels (6.8-11% with four channels 20 times larger; 2.6-2.7% on peaked rows). Gates, from the first run on an RTX PRO
    /// 4500 (seeded data): relative RMS at most 5% (Gaussian, peaked), 15% (key outliers), the largest absolute difference within
    /// 1.2 (twice the largest measured, 0.58). The kernels themselves are the direct
    /// engine's bit for bit (the eng oracle, t2 sec. 10.2 K6); what the precision does to a model is measured end to end.
    /// </summary>
    [CudaTheory(PromptAttention8 = true)]
    [InlineData(24, 4, 2048, 2048, 0)]    // Qwen3.8-27B's first 2,048-row chunk
    [InlineData(24, 4, 1570, 3618, 0)]    // its second chunk over the first's keys
    [InlineData(24, 4, 968, 47981, 0)]    // the issue's B turn: 968 rows over 47,981 keys
    [InlineData(16, 4, 300, 4000, 33)]    // GQA 4, keys below a floor
    [InlineData(32, 4, 256, 9000, 0)]     // GQA 8
    [InlineData(24, 4, 24, 4000, 0)]      // a few rows over a long window: split over the keys
    public void MatchesAnF64Reference(int heads, int kvHeads, int nq, int len, int floor)
    {
        var (allocator, flash) = Open();
        using (allocator)
        {
            int splits = flash.Splits(nq * (heads / kvHeads), kvHeads, len);
            int[] rows = SampledRows(nq);
            var failures = new System.Collections.Generic.List<string>();
            foreach (Data data in Enum.GetValues<Data>())
            {
                using var chunk = new Chunk(allocator, heads, kvHeads, nq, len, floor, data);
                float[] eight = chunk.Run8(allocator, flash, splits), sixteen = chunk.Run16(allocator, flash, splits);
                var r8 = Compare(eight, chunk.Reference, rows, heads);
                var r16 = Compare(sixteen, chunk.Reference, rows, heads);
                _output.WriteLine($"{heads}/{kvHeads} heads, {nq} rows over {len} keys (floor {floor}), {splits} splits, {data}: 8-bit rel RMS " +
                    $"{r8.Rel:E3} max |d| {r8.MaxAbs:E3}; 16-bit rel RMS {r16.Rel:E3} max |d| {r16.MaxAbs:E3} ({r8.Rel / r16.Rel:F1}x)");
                double gate = data == Data.KeyOutliers ? 0.15 : 0.05;
                if (!(r8.Rel <= gate)) failures.Add($"{data}: relative RMS {r8.Rel:E3} against the F64 attention (gate {gate})");
                if (!(r8.MaxAbs <= 1.2)) failures.Add($"{data}: largest absolute difference {r8.MaxAbs:E3} (gate 1.2)");
                if (!Array.TrueForAll(eight, float.IsFinite)) failures.Add($"{data}: a non-finite output");
            }
            Assert.True(failures.Count == 0, string.Join("; ", failures));
        }
    }

    /// <summary>Two launches over the same operands give the same bits (no atomics; the splits' combine in a fixed order).</summary>
    [CudaTheory(PromptAttention8 = true)]
    [InlineData(24, 4, 1570, 3618, 0)]
    [InlineData(24, 4, 24, 4000, 0)]
    public void IsDeterministic(int heads, int kvHeads, int nq, int len, int floor)
    {
        var (allocator, flash) = Open();
        using (allocator)
        using (var chunk = new Chunk(allocator, heads, kvHeads, nq, len, floor, Data.Gaussian))
        {
            int splits = flash.Splits(nq * (heads / kvHeads), kvHeads, len);
            float[] a = chunk.Run8(allocator, flash, splits), b = chunk.Run8(allocator, flash, splits);
            Assert.True(a.AsSpan().SequenceEqual(b), $"{nq} rows over {len} keys, {splits} splits: two launches differ");
        }
    }

    /// <summary>A launch split over the keys (1, 2 and 8 splits, the combine merging their partials) agrees with the unsplit one
    /// within 1% relative RMS: only where the online softmax rescales, and the last tile's sums, round apart.</summary>
    [CudaTheory(PromptAttention8 = true)]
    [InlineData(32, 4, 256, 9000, 0)]
    [InlineData(24, 4, 24, 4000, 0)]
    [InlineData(16, 4, 300, 4000, 33)]
    public void SplitsAgreeWithTheUnsplitLaunch(int heads, int kvHeads, int nq, int len, int floor)
    {
        var (allocator, flash) = Open();
        using (allocator)
        using (var chunk = new Chunk(allocator, heads, kvHeads, nq, len, floor, Data.Peaked))
        {
            float[] one = chunk.Run8(allocator, flash, 1);
            int[] rows = Enumerable.Range(0, nq).ToArray();
            foreach (int splits in new[] { 2, 8 })
            {
                float[] split = chunk.Run8(allocator, flash, splits);
                var (rel, max) = Compare(split, t => Row(one, t, heads), rows, heads);
                _output.WriteLine($"{nq} rows over {len} keys: {splits} splits against 1, rel RMS {rel:E3}, max |d| {max:E3}");
                Assert.True(rel <= 0.01, $"{splits} splits: relative RMS {rel:E3} against the unsplit launch");
            }
        }
    }

    /// <summary>
    /// A row's result does not depend on where its chunk ends, except through the V scale of the chunk's last, partial 32-key
    /// tile: 3,618 keys forwarded as 2,048 + 1,570 rows and as 2,000 + 1,618 give every row the same bits but the 16 whose causal
    /// end lies in the partial tile [1984, 2000) of the 2,000-row chunk (within 3% relative RMS there).
    /// </summary>
    [CudaFact(PromptAttention8 = true)]
    public void RowsOutsideTheLastPartialTileAreChunkInvariant()
    {
        var (allocator, flash) = Open();
        using (allocator)
        using (var whole = new Chunk(allocator, 24, 4, 3618, 3618, 0, Data.Gaussian))
        {
            float[] Forward(int split)
            {
                var result = new float[3618 * 24 * D];
                using var first = new Chunk(allocator, whole, 0, split);
                using var second = new Chunk(allocator, whole, split, 3618 - split);
                Array.Copy(first.Run8(allocator, flash, 1), 0, result, 0, (long)split * 24 * D);
                Array.Copy(second.Run8(allocator, flash, 1), 0, result, (long)split * 24 * D, (long)(3618 - split) * 24 * D);
                return result;
            }
            float[] a = Forward(2048), b = Forward(2000);
            var apart = Enumerable.Range(0, 3618)
                .Where(t => !a.AsSpan(t * 24 * D, 24 * D).SequenceEqual(b.AsSpan(t * 24 * D, 24 * D))).ToArray();
            _output.WriteLine($"rows apart between the 2,048 and 2,000-row chunkings: {apart.Length} ([{string.Join(",", apart.Take(40))}])");
            Assert.All(apart, t => Assert.InRange(t, 1984, 1999));
            if (apart.Length > 0)
            {
                var (rel, _) = Compare(b, t => Row(a, t, 24), apart, 24);
                _output.WriteLine($"those rows' relative RMS: {rel:E3}");
                Assert.True(rel <= 0.03, $"the partial tile's rows: relative RMS {rel:E3}");
            }
        }
    }

    /// <summary>
    /// The band holds exactly its definition: per key, codes round-to-nearest-even of x / s clamped to [-127, 127] with
    /// s = amax / 127 (0 past len); per 32-key tile, the smallest e with amax / 2^e &lt;= 448 (stored e + 127) and every value
    /// x / 2^e in E4M3 (round to nearest even, saturating) in the tile's key order - computed here on the host from the F16 cache.
    /// </summary>
    [CudaTheory(PromptAttention8 = true)]
    [InlineData(24, 4, 1570, 3618, 0, Data.KeyOutliers)]   // a partial last tile (2 keys)
    [InlineData(16, 4, 300, 4000, 33, Data.Gaussian)]      // a floor: the band starts at its tile
    [InlineData(24, 4, 300, 1100, 0, Data.Peaked)]
    public void BandMatchesItsDefinition(int heads, int kvHeads, int nq, int len, int floor, Data data)
    {
        var (allocator, flash) = Open();
        using (allocator)
        using (var chunk = new Chunk(allocator, heads, kvHeads, nq, len, floor, data))
        {
            byte[] band = null;
            chunk.Run8(allocator, flash, 1, b => band = b);
            int bandKeys = CudaFlashPrefillKernels.Attn8BandKeys(len), tiles = bandKeys / 32, cap = chunk.Cap;
            var layout = CudaFlashPrefillKernels.Attn8BandLayout(bandKeys, kvHeads);
            int differ = 0;
            void Expect(long offset, byte value, string what)
            {
                if (band[offset] != value && differ++ < 8) _output.WriteLine($"{what}: {band[offset]:X2}, expected {value:X2}");
            }
            for (int h = 0; h < kvHeads; h++)
                for (int tile = floor / 32; tile < (len + 31) / 32; tile++)
                {
                    for (int key = tile * 32; key < tile * 32 + 32; key++)
                    {
                        float amax = 0;
                        for (int x = 0; x < D; x++) amax = key < len ? MathF.Max(amax, MathF.Abs(chunk.K[((long)h * cap + key) * D + x])) : 0;
                        float s = amax > 0 ? amax / 127f : 1f, inv = 1f / s;
                        long row = (long)h * bandKeys + key;
                        float scale = BitConverter.ToSingle(band, (int)(layout.KScales + row * 4));
                        if (BitConverter.SingleToInt32Bits(scale) != BitConverter.SingleToInt32Bits(key < len ? s : 0f) && differ++ < 8)
                            _output.WriteLine($"head {h} key {key} scale {scale:R}, expected {(key < len ? s : 0f):R}");
                        for (int x = 0; x < D; x++)
                        {
                            float y = key < len ? chunk.K[((long)h * cap + key) * D + x] * inv : 0f;
                            int c = Math.Clamp((int)Math.Round((double)y, MidpointRounding.ToEven), -127, 127);
                            Expect(layout.K8 + row * D + x, (byte)(sbyte)c, $"head {h} key {key} K code {x}");
                        }
                    }
                    float vmax = 0;
                    for (int key = tile * 32; key < Math.Min(len, tile * 32 + 32); key++)
                        for (int x = 0; x < D; x++) vmax = MathF.Max(vmax, MathF.Abs(chunk.V[((long)h * cap + key) * D + x]));
                    int e = 0;
                    if (vmax > 0)
                    {
                        float r = vmax / 448f;
                        e = MathF.ILogB(r) + 1;                       // frexpf's exponent
                        if (MathF.ScaleB(1f, e - 1) >= r) e--;
                    }
                    e = Math.Clamp(e, -127, 127);
                    Expect(layout.VScales + (long)h * tiles + tile, (byte)(e + 127), $"head {h} tile {tile} V scale");
                    float vinv = MathF.ScaleB(1f, -e);
                    for (int x = 0; x < D; x++)
                        for (int j = 0; j < 32; j++)
                        {
                            int key = tile * 32 + Perm(j);
                            float y = key < len ? chunk.V[((long)h * cap + key) * D + x] * vinv : 0f;
                            Expect(layout.VT + ((long)h * tiles + tile) * D * 32 + x * 32 + j, E4m3(y), $"head {h} tile {tile} V^T [{x}][{j}]");
                        }
                }
            _output.WriteLine($"{heads}/{kvHeads} heads over {len} keys (floor {floor}), {data}: {differ} band bytes apart from the definition");
            Assert.Equal(0, differ);
        }
    }

    /// <summary>The key of column j of a V^T tile (ts_fp_a8_perm).</summary>
    private static int Perm(int j)
    {
        int q = (j & 15) >> 2, i = j & 3;
        return (j & 16) + (i < 2 ? 2 * q + i : 8 + 2 * q + i - 2);
    }

    /// <summary>F32 to E4M3 (e4m3fn: bias 7, largest finite 448), round to nearest even, saturating; the sign kept on zeros.</summary>
    internal static byte E4m3(float x)
    {
        byte sign = (byte)(BitConverter.SingleToInt32Bits(x) < 0 ? 0x80 : 0);
        double a = Math.Abs((double)x);
        if (a >= 448) return (byte)(sign | 0x7E);
        if (a < 1.0 / 64)
        {
            // Subnormals (and the smallest normal, which code 8 is): steps of 2^-9.
            int m = (int)Math.Round(a * 512, MidpointRounding.ToEven);
            return (byte)(sign | m);
        }
        int exp = Math.ILogB(a);
        double mant = Math.ScaleB(a, -exp) - 1;                          // [0, 1)
        int code = (int)Math.Round(mant * 8, MidpointRounding.ToEven);   // 8 carries into the exponent
        int value = ((exp + 7) << 3) + code;
        return (byte)(sign | Math.Min(value, 0x7E));
    }
}
