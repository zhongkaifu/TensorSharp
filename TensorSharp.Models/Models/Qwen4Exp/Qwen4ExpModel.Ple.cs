// Copyright (c) Zhongkai Fu. All rights reserved.
// https://github.com/zhongkaifu/TensorSharp
//
// This file is part of TensorSharp.
//
// TensorSharp is licensed under the BSD-3-Clause license found in the LICENSE file in the root directory of this source tree.
//
// TensorSharp is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the BSD-3-Clause License for more details.
using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Threading.Tasks;
using TensorSharp.Core;

namespace TensorSharp.Models
{
    public partial class Qwen4ExpModel
    {
        // Tokens preceding the current batch, for the n-gram window. A fresh sequence
        // (or one whose positions are not contiguous with the last batch) reads EOS,
        // which is what the reference's zero-padded start gives.
        // Not readonly: the per-sequence holder swap repoints it (PerSeqCache).
        private List<int> _pleHistory = new();
        private int _pleNextPos;

        private void ResetPleHistory()
        {
            _pleHistory.Clear();
            _pleNextPos = 0;
        }

        /// <summary>
        /// The n-gram hash. For each token and each n in 2..ngram:
        ///   <c>mixed = ctx[0]*m[0] ^ ctx[1]*m[1] ^ ... ^ ctx[n-1]*m[n-1]</c>
        ///   <c>row[h] = mixed % vocab[h] + offset[h]</c>
        /// Predecessors reset at an EOS: an EOS anywhere in the window hides everything
        /// at or before it. The token's OWN id does not cut its context - the reference
        /// takes the last EOS strictly before the position - so a boundary only hides
        /// tokens from the positions that follow it.
        ///
        /// Multipliers reach ~2^45, so this runs in ulong on the host; there is no
        /// 64-bit integer multiply or xor to do it with on the device.
        /// </summary>
        private int[] ComputePleRows(int[] tokens, int startPos)
        {
            int[] rows = ComputePleRows(tokens, startPos, _pleHistory, _pleNextPos);
            CommitPleHistory(_pleHistory, tokens, startPos, _pleNextPos);
            _pleNextPos = startPos + tokens.Length;
            return rows;
        }

        // The arena must peek all requests before native admission; a decline
        // leaves each request's n-gram history unchanged for its solo retry.
        private int[] ComputePleRows(int[] tokens, int startPos, List<int> history, int nextPos)
        {
            int n = tokens.Length;
            int nGram = _pleNgram;
            int eos = _pleEosTokenId;
            var rows = new int[(long)_pleHeads * n];

            // Snapshot the incoming history first: reading and updating in one pass
            // would let an early token in this batch pick up a later one as context.
            var hist = nextPos == startPos ? new List<int>(history) : new List<int>();
            while (hist.Count < nGram - 1) hist.Insert(0, eos);

            var ctx = new long[nGram];
            for (int i = 0; i < n; i++)
            {
                int pos = startPos + i;
                ctx[0] = tokens[i];
                bool cut = false;
                for (int s = 1; s < nGram; s++)
                {
                    long tok;
                    if (cut)
                    {
                        tok = eos;
                    }
                    else
                    {
                        int j = i - s;
                        if (j >= 0)
                        {
                            tok = tokens[j];
                        }
                        else
                        {
                            int back = s - i;                     // positions before this batch
                            int idx = hist.Count - back;
                            tok = (back > 0 && idx >= 0 && idx < hist.Count && pos - s >= 0)
                                ? hist[idx] : eos;
                        }
                    }
                    ctx[s] = tok;
                    if (tok == eos) cut = true;
                }

                for (int g = 2; g <= nGram; g++)
                {
                    ulong mixed = (ulong)ctx[0] * _pleMultipliers[0];
                    for (int j = 1; j < g; j++)
                        mixed ^= (ulong)ctx[j] * _pleMultipliers[j];

                    int baseHead = (g - 2) * _pleHeadsPerNgram;
                    for (int hh = 0; hh < _pleHeadsPerNgram; hh++)
                    {
                        int h = baseHead + hh;
                        rows[(long)i * _pleHeads + h] =
                            (int)(mixed % _pleHeadVocabSizes[h] + _pleHeadOffsets[h]);
                    }
                }
            }

            return rows;
        }

        private void CommitPleHistory(List<int> history, int[] tokens, int startPos, int nextPos)
        {
            if (nextPos != startPos) history.Clear();
            history.AddRange(tokens);
            if (history.Count > _pleNgram - 1)
                history.RemoveRange(0, history.Count - (_pleNgram - 1));
        }

        /// <summary>
        /// The PLE block, run on the layers named by <c>ple.layers</c> (one, in the
        /// shipped file). It gathers <c>ple_n_heads</c> rows of the ~320 M row n-gram
        /// table per token, scores them against the residual, and adds both the gated
        /// value and a dilated depthwise convolution of it back into every stream.
        ///
        /// Updates <paramref name="res"/> in place.
        /// </summary>
        // The per-token stages parallelize above this batch size; below it the
        // Parallel.For overhead costs more than it saves (decode is T=1).
        private const int PleParallelThreshold = 16;

        private unsafe void PleLayer(Tensor res, int[] tokens, int seqLen, int startPos, int il)
        {
            int n = Config.HiddenSize;
            long tStage = Stopwatch.GetTimestamp();
            int[] rows = ComputePleRows(tokens, startPos);
            PhaseLog(seqLen, "ple.rows", tStage);

            // Gather: heads laid out slowest, so a token's rows concatenate into one
            // hidden-wide vector.
            tStage = Stopwatch.GetTimestamp();
            var emb = new Tensor(_allocator, DType.Float32, seqLen, n);
            GatherPleRows(emb, rows, seqLen);
            PhaseLog(seqLen, "ple.gather", tStage);

            tStage = Stopwatch.GetTimestamp();
            Tensor key = LinearForward(emb, $"blk.{il}.ple_key.weight");     // [T, hcDim]
            Tensor value = LinearForward(emb, $"blk.{il}.ple_value.weight"); // [T, n_embd]
            emb.Dispose();
            PhaseLog(seqLen, "ple.proj", tStage);

            tStage = Stopwatch.GetTimestamp();
            GroupedNormInPlace(key, $"blk.{il}.ple_norm_key.weight", seqLen);
            var query = new Tensor(_allocator, DType.Float32, seqLen, _hcDim);
            Ops.Copy(query, res);
            GroupedNormInPlace(query, $"blk.{il}.ple_norm_query.weight", seqLen);
            PhaseLog(seqLen, "ple.norms", tStage);
            tStage = Stopwatch.GetTimestamp();

            // Per-stream dot product, then a signed square root before the sigmoid.
            // Token rows are independent, so a prefill spreads them across cores.
            var gated = new Tensor(_allocator, DType.Float32, seqLen, _hcDim);
            {
                long kA = (long)GetFloatPtr(key);
                long qA = (long)GetFloatPtr(query);
                long vA = (long)GetFloatPtr(value);
                long gA = (long)GetFloatPtr(gated);
                float invSqrt = 1.0f / MathF.Sqrt(n);
                int hc = _hc, hcDim = _hcDim;
                void GateToken(int t)
                {
                    float* kr = (float*)kA + (long)t * hcDim;
                    float* qr = (float*)qA + (long)t * hcDim;
                    float* vr = (float*)vA + (long)t * n;
                    float* gr = (float*)gA + (long)t * hcDim;
                    for (int c = 0; c < hc; c++)
                    {
                        float* kc = kr + (long)c * n;
                        float* qc = qr + (long)c * n;
                        double dot = 0;
                        for (int i = 0; i < n; i++) dot += (double)kc[i] * qc[i];
                        float s = (float)dot * invSqrt;
                        float mag = MathF.Sqrt(Math.Clamp(MathF.Abs(s), 1e-6f, float.PositiveInfinity));
                        float signed = MathF.Sign(s) * mag;
                        float g = 1.0f / (1.0f + MathF.Exp(-signed));
                        float* gc = gr + (long)c * n;
                        for (int i = 0; i < n; i++) gc[i] = vr[i] * g;
                    }
                }
                if (seqLen >= PleParallelThreshold) Parallel.For(0, seqLen, GateToken);
                else for (int t = 0; t < seqLen; t++) GateToken(t);
                InvalidateTensorDeviceCache(gated);
            }
            key.Dispose(); query.Dispose(); value.Dispose();
            PhaseLog(seqLen, "ple.gate", tStage);
            tStage = Stopwatch.GetTimestamp();

            // Dilated causal depthwise conv over the gated value, as a sum of shifted
            // per-channel-scaled copies. History from earlier batches is prepended so a
            // chunked prefill sees what a single-shot one would.
            var normed = new Tensor(_allocator, DType.Float32, seqLen, _hcDim);
            Ops.Copy(normed, gated);
            GroupedNormInPlace(normed, $"blk.{il}.ple_norm_conv.weight", seqLen);

            var conv = new Tensor(_allocator, DType.Float32, seqLen, _hcDim);
            {
                int kern = _pleConvKernel;
                int dil = _pleNgram;
                int hist = (kern - 1) * dil;
                int total = hist + seqLen;

                // One contiguous [hist + T, hcDim] buffer: history rows first, this
                // batch after, so a chunked prefill sees what a single-shot one would.
                var padded = new float[(long)total * _hcDim];
                Array.Copy(_pleConvState, padded, (long)hist * _hcDim);
                float* np = GetFloatPtr(normed);
                fixed (float* pad = padded)
                {
                    for (int t = 0; t < seqLen; t++)
                    {
                        Buffer.MemoryCopy(np + (long)t * _hcDim,
                            pad + (long)(hist + t) * _hcDim, _hcDim * 4L, _hcDim * 4L);
                    }

                    long cA = (long)GetFloatPtr(conv);
                    long wA = (long)GetFloatPtr(_weights[$"blk.{il}.ple_conv1d.weight"]);
                    long pA = (long)pad;
                    int hcDim = _hcDim;
                    // `padded` is fully written above and only read here, so the
                    // output rows are independent.
                    void ConvToken(int t)
                    {
                        float* o = (float*)cA + (long)t * hcDim;
                        for (int c = 0; c < hcDim; c++) o[c] = 0f;
                        for (int kk = 0; kk < kern; kk++)
                        {
                            // tap kk reads (kern-1-kk) dilated positions back
                            int r = hist + t - (kern - 1 - kk) * dil;
                            if (r < 0) continue;
                            float* x = (float*)pA + (long)r * hcDim;
                            float* wp = (float*)wA;
                            // ple_conv1d is [kernel, channels]: column kk is one
                            // weight per channel
                            for (int c = 0; c < hcDim; c++)
                                o[c] += x[c] * wp[(long)c * kern + kk];
                        }
                        for (int c = 0; c < hcDim; c++)
                            o[c] = o[c] / (1.0f + MathF.Exp(-o[c]));
                    }
                    if (seqLen >= PleParallelThreshold) Parallel.For(0, seqLen, ConvToken);
                    else for (int t = 0; t < seqLen; t++) ConvToken(t);
                }
                // Keep the last `hist` rows for the next batch.
                Array.Copy(padded, (long)(total - hist) * _hcDim, _pleConvState, 0L,
                    (long)hist * _hcDim);
                InvalidateTensorDeviceCache(conv);
            }
            normed.Dispose();
            PhaseLog(seqLen, "ple.conv", tStage);
            tStage = Stopwatch.GetTimestamp();

            // res += gated + conv
            {
                long rA = (long)GetFloatPtr(res);
                long gA = (long)GetFloatPtr(gated);
                long cA = (long)GetFloatPtr(conv);
                int hcDim = _hcDim;
                void AddToken(int t)
                {
                    float* rp = (float*)rA + (long)t * hcDim;
                    float* gp = (float*)gA + (long)t * hcDim;
                    float* cp = (float*)cA + (long)t * hcDim;
                    for (int i = 0; i < hcDim; i++) rp[i] += gp[i] + cp[i];
                }
                if (seqLen >= PleParallelThreshold) Parallel.For(0, seqLen, AddToken);
                else for (int t = 0; t < seqLen; t++) AddToken(t);
                InvalidateTensorDeviceCache(res);
            }
            gated.Dispose();
            conv.Dispose();
            PhaseLog(seqLen, "ple.add", tStage);
        }

        /// <summary>
        /// Gather this batch's n-gram rows out of the PLE table.
        ///
        /// <paramref name="dest"/> is [T, hidden], which is the same memory as
        /// [T * ple_heads, ple_head_dim] - the heads concatenate into one hidden-wide
        /// row per token, which is what the reference's flatten over the head axis
        /// gives. The table is ~320 M rows and is read 16 rows per token, so it is
        /// dequantized row by row from host memory rather than made device-resident.
        /// </summary>
        private unsafe void GatherPleRows(Tensor dest, int[] rows, int seqLen)
        {
            const string name = "per_layer_token_embd.weight";
            using Tensor flat = dest.View((long)seqLen * _pleHeads, _pleHeadDim);

            if (_quantWeights.TryGetValue(name, out var qw))
            {
                if (!qw.HasHostData)
                {
                    throw new InvalidOperationException(
                        "qwen4exp needs a host copy of the PLE n-gram table: it is gathered "
                        + $"{_pleHeads} scattered rows per token, which no device get_rows path serves.");
                }
                // Row dequants are independent native calls scattered over a ~24 GB
                // table; a prefill reads tens of thousands of them, so spread the
                // page-touching and the dequant across cores.
                long qRowBytes = NativeDequant.RowSize(qw.GgmlType, qw.Ne0);
                int dim = (int)qw.Ne0;
                long baseA = (long)qw.Data;
                long dstA = (long)GetFloatPtr(flat);
                var ggmlType = qw.GgmlType;
                void DequantRow(int i)
                {
                    NativeDequant.DequantizeToFloat32Native(
                        ggmlType,
                        (IntPtr)((byte*)baseA + (long)rows[i] * qRowBytes),
                        (IntPtr)((float*)dstA + (long)i * dim),
                        dim);
                }
                if (rows.Length >= PleParallelThreshold * _pleHeads)
                    Parallel.For(0, rows.Length, DequantRow);
                else
                    for (int i = 0; i < rows.Length; i++) DequantRow(i);
                InvalidateTensorDeviceCache(dest);
                return;
            }

            Tensor table = _weights[name];
            float* src = GetFloatPtr(table);
            float* dst = GetFloatPtr(flat);
            long rowBytes = _pleHeadDim * sizeof(float);
            for (int i = 0; i < rows.Length; i++)
            {
                Buffer.MemoryCopy(src + (long)rows[i] * _pleHeadDim,
                    dst + (long)i * _pleHeadDim, rowBytes, rowBytes);
            }
            InvalidateTensorDeviceCache(dest);
        }

        // The n-gram table is read from the GGUF mapping on demand and never uploaded,
        // so a row not in the page cache is a page fault that goes to the SSD.
        private bool _pleTableOnDisk;

        /// <summary>
        /// The table is ~29 GB of rows read at random, ple_n_heads of them a token -
        /// llama.cpp's lazily read tensor. Advise the kernel accordingly, so a fault
        /// reads the row's page and not a read-ahead cluster around it.
        /// </summary>
        private void PreparePleTableAccess()
        {
            if (_pleHeads <= 0
                || !_quantWeights.TryGetValue("per_layer_token_embd.weight", out var qw)
                || !qw.IsFileBacked)
                return;
            _pleTableOnDisk = true;
            bool advised = qw.AdviseRandomAccess();
            Console.WriteLine($"  PLE n-gram table: {qw.RawBytes / 1e9:F1} GB read on demand from the GGUF mapping, " +
                $"{_pleHeads} rows a token" + (advised ? " (random-access advice)." : "."));
        }

        /// <summary>
        /// Gather the n-gram rows into a raw [T * hidden] float buffer - the span
        /// uploads it as a graph input. A row read from disk costs a page fault, so a
        /// table on disk is fetched in parallel even for one token: its rows are
        /// independent faults the SSD can serve concurrently.
        /// </summary>
        private unsafe void GatherPleRowsRaw(float* dst, int[] rows, int seqLen)
        {
            const string name = "per_layer_token_embd.weight";
            if (_quantWeights.TryGetValue(name, out var qw))
            {
                if (!qw.HasHostData)
                    throw new InvalidOperationException("qwen4exp needs a host copy of the PLE n-gram table.");
                long qRowBytes = NativeDequant.RowSize(qw.GgmlType, qw.Ne0);
                int dim = (int)qw.Ne0;
                long baseA = (long)qw.Data;
                long dstA = (long)dst;
                var ggmlType = qw.GgmlType;
                void DequantRow(int i)
                {
                    NativeDequant.DequantizeToFloat32Native(
                        ggmlType,
                        (IntPtr)((byte*)baseA + (long)rows[i] * qRowBytes),
                        (IntPtr)((float*)dstA + (long)i * dim),
                        dim);
                }
                if (rows.Length >= PleParallelThreshold * _pleHeads || (_pleTableOnDisk && rows.Length > 1))
                    Parallel.For(0, rows.Length, DequantRow);
                else
                    for (int i = 0; i < rows.Length; i++) DequantRow(i);
                return;
            }

            Tensor table = _weights[name];
            float* src = GetFloatPtr(table);
            long rowBytes = _pleHeadDim * sizeof(float);
            for (int i = 0; i < rows.Length; i++)
            {
                Buffer.MemoryCopy(src + (long)rows[i] * _pleHeadDim,
                    dst + (long)i * _pleHeadDim, rowBytes, rowBytes);
            }
        }

        /// <summary>
        /// RMS norm over each residual stream separately, then an affine weight that
        /// spans the whole hc*n_embd row - the same shape the hyper-connection mixer
        /// uses. In place.
        /// </summary>
        private unsafe void GroupedNormInPlace(Tensor x, string weightName, int seqLen)
        {
            int n = Config.HiddenSize;
            long pA = (long)GetFloatPtr(x);
            long wA = (long)GetFloatPtr(_weights[weightName]);
            float eps = Config.Eps;
            int hc = _hc, hcDim = _hcDim;
            void NormToken(int t)
            {
                float* row = (float*)pA + (long)t * hcDim;
                float* w = (float*)wA;
                for (int c = 0; c < hc; c++)
                {
                    float* xc = row + (long)c * n;
                    double ss = 0;
                    for (int i = 0; i < n; i++) ss += (double)xc[i] * xc[i];
                    float inv = (float)(1.0 / Math.Sqrt(ss / n + eps));
                    int b = c * n;
                    for (int i = 0; i < n; i++) xc[i] = xc[i] * inv * w[b + i];
                }
            }
            if (seqLen >= PleParallelThreshold) Parallel.For(0, seqLen, NormToken);
            else for (int t = 0; t < seqLen; t++) NormToken(t);
            InvalidateTensorDeviceCache(x);
        }
    }
}
