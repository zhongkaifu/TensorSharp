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
using System.Buffers;
using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;
using System.Numerics;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;
using System.Threading;
using System.Threading.Tasks;
using TensorSharp;
using TensorSharp.Cpu;
using TensorSharp.Models.Architecture;
using TensorSharp.Cuda;
using TensorSharp.GGML;
using TensorSharp.MLX;

namespace TensorSharp.Models
{

    // Tensor-parallel weight sharding and the collectives the sharded layers issue.
    // Split out of ModelBase.cs: it is one self-contained concern, it only runs when a
    // TP group is live, and it was a third of the file.
    public abstract partial class ModelBase
    {
        #region Tensor Parallelism

        /// <summary>
        /// Shard linear weights across GPUs for tensor parallelism. Call after
        /// <see cref="LoadWeights"/> and any weight fusion (e.g. FuseQKVWeights).
        ///
        /// Column-parallel weights (QKV, gate_up) are split along the output
        /// dimension (ne1 / Sizes[0]): each GPU gets outDim/tp consecutive rows.
        /// Row-parallel weights (output, down) are split along the input
        /// dimension (ne0 / Sizes[1]): each GPU gets inDim/tp columns.
        ///
        /// Replicated weights (norms, embeddings) are not sharded — every GPU
        /// keeps the full copy via the existing <see cref="_weights"/> dictionary.
        /// </summary>
        protected void ShardWeightsForTensorParallelism(
            IEnumerable<string> columnParallelPatterns,
            IEnumerable<string> rowParallelPatterns)
        {
            if (!IsTensorParallel) return;

            int tp = TpDegree;
            int globalTp = GlobalTpDegree;
            int rankOffset = TpRankOffset;
            var colPatterns = new List<string>(columnParallelPatterns);
            var rowPatterns = new List<string>(rowParallelPatterns);

            if (tp > 1 || globalTp > 1)
                Console.WriteLine($"  ShardWeightsForTP: tp={tp} globalTp={globalTp} rankOffset={rankOffset} colPatterns=[{string.Join(",", colPatterns)}] rowPatterns=[{string.Join(",", rowPatterns)}]");

            // Shard quantized weights.
            var quantToRemove = new List<string>();
            var colParallelOwners = new List<QuantizedWeight>();
            foreach (var kv in _quantWeights)
            {
                string name = kv.Key;
                bool isCol = colPatterns.Any(p => name.Contains(p));
                bool isRow = !isCol && rowPatterns.Any(p => name.Contains(p));
                if (!isCol && !isRow) continue;

                var qw = kv.Value;
                var shards = new QuantizedWeight[tp];

                if (isCol)
                {
                    // Split along ne1 (output dim): consecutive rows.
                    long rowsPerShard = qw.Ne1 / globalTp;
                    long rowBytes = NativeDequant.RowSize(qw.GgmlType, qw.Ne0);
                    long bytesPerShard = rowsPerShard * rowBytes;

                    for (int lr = 0; lr < tp; lr++)
                    {
                        int globalRank = rankOffset + lr;
                        IntPtr shardPtr = new IntPtr((long)qw.Data + globalRank * bytesPerShard);
                        shards[lr] = QuantizedWeight.CreateExternalView(
                            shardPtr, bytesPerShard, qw.GgmlType, qw.Ne0, rowsPerShard, qw);
                    }

                    // Keep the original owner alive: column-parallel shards
                    // are external views into its buffer. Disposing it here
                    // would leave the shards with dangling pointers.
                    colParallelOwners.Add(qw);
                }
                else
                {
                    // Split along ne0 (input dim): extract block-aligned columns per row.
                    var type = (GgmlTensorType)qw.GgmlType;
                    long blockSize = GgufFile.GetBlockSize(type);
                    long typeSize = GgufFile.GetTypeSize(type);
                    long blocksPerRow = qw.Ne0 / blockSize;
                    long blocksPerShard = blocksPerRow / globalTp;
                    if (blocksPerRow % globalTp != 0)
                        throw new NotSupportedException(
                            $"Row-parallel shard of '{name}' would drop a tail: {qw.Ne0} columns is " +
                            $"{blocksPerRow} blocks of {blockSize} for {globalTp} ranks. A quantized " +
                            $"row can only be split on block boundaries, so ne0 must be a multiple of " +
                            $"{blockSize * globalTp}.");
                    long ne0PerShard = blocksPerShard * blockSize;
                    long srcRowBytes = NativeDequant.RowSize(qw.GgmlType, qw.Ne0);
                    long dstRowBytes = (ne0PerShard / blockSize) * typeSize;
                    long totalBytesPerShard = qw.Ne1 * dstRowBytes;
                    long blockBytesPerShard = blocksPerShard * typeSize;

                    if (quantToRemove.Count < 2)
                        Console.WriteLine($"    Row-parallel quant debug: {name} origNe0={qw.Ne0} origNe1={qw.Ne1} globalTp={globalTp} blockSize={blockSize} blocksPerRow={blocksPerRow} blocksPerShard={blocksPerShard} ne0PerShard={ne0PerShard} totalBytesPerShard={totalBytesPerShard}");


                    // Extract every rank's slice concurrently: each rank writes
                    // its own buffer, and when the source is the memory-mapped
                    // GGUF the parallel ranks fault in disjoint file regions at
                    // once — on slow or network-backed storage the page-fault
                    // reads, not the memcpy, are what this loop actually costs.
                    Parallel.For(0, tp, lr =>
                    {
                        int globalRank = rankOffset + lr;
                        IntPtr shardPtr = QuantizedWeight.AllocateBuffer(totalBytesPerShard);
                        unsafe
                        {
                            byte* src = (byte*)qw.Data.ToPointer();
                            byte* dst = (byte*)shardPtr.ToPointer();
                            long srcBlockOffset = globalRank * blocksPerShard * typeSize;
                            for (long row = 0; row < qw.Ne1; row++)
                            {
                                Buffer.MemoryCopy(
                                    src + row * srcRowBytes + srcBlockOffset,
                                    dst + row * dstRowBytes,
                                    dstRowBytes,
                                    blockBytesPerShard);
                            }
                        }
                        shards[lr] = new QuantizedWeight(shardPtr, totalBytesPerShard,
                            qw.GgmlType, ne0PerShard, qw.Ne1);
                        // A per-tensor scale is shard-invariant: it does not depend on
                        // the output row, and it distributes over the row-parallel
                        // AllReduce, so every shard carries the parent's value.
                        shards[lr].Scale = qw.Scale;
                    });
                }

                _tpQuantWeights[name] = shards;
                RecordTpWeightScale(name, qw);
                quantToRemove.Add(name);
            }

            // Remove original unsharded quantized weights that have been sharded.
            // Column-parallel owners are kept alive because their shards are
            // external views into the original buffer; only row-parallel
            // originals (whose shards own independent copies) are disposed.
            foreach (var name in quantToRemove)
            {
                var qw = _quantWeights[name];
                if (!colParallelOwners.Contains(qw))
                    qw.Dispose();
                _quantWeights.Remove(name);
            }

            // Shard F32 weights.
            var f32ToRemove = new List<string>();
            foreach (var kv in _weights)
            {
                string name = kv.Key;
                bool isCol = colPatterns.Any(p => name.Contains(p));
                bool isRow = !isCol && rowPatterns.Any(p => name.Contains(p));
                if (!isCol && !isRow) continue;

                var w = kv.Value;
                var shards = new Tensor[tp];

                if (isCol)
                {
                    // Split along dim 0 (output dim).
                    long shardSize = w.Sizes[0] / globalTp;
                    for (int lr = 0; lr < tp; lr++)
                    {
                        int globalRank = rankOffset + lr;
                        var view = w.Narrow(0, globalRank * shardSize, shardSize);
                        shards[lr] = Ops.NewContiguous(view);
                        view.Dispose();
                    }
                }
                else
                {
                    // Split along dim 1 (input dim).
                    long shardSize = w.Sizes[1] / globalTp;
                    for (int lr = 0; lr < tp; lr++)
                    {
                        int globalRank = rankOffset + lr;
                        var view = w.Narrow(1, globalRank * shardSize, shardSize);
                        shards[lr] = Ops.NewContiguous(view);
                        view.Dispose();
                    }
                }

                _tpWeights[name] = shards;
                f32ToRemove.Add(name);
            }

            foreach (var name in f32ToRemove)
            {
                _weights[name].Dispose();
                _weights.Remove(name);
            }

            Console.WriteLine($"  TP sharded: {quantToRemove.Count} quantized + {f32ToRemove.Count} F32 weights across {tp} GPUs.");
        }

        /// <summary>
        /// Output dimension (row count) of a possibly-quantized weight, or 0 if
        /// the weight is not present in either weight dictionary.
        /// </summary>
        protected int GetFusedOutputDim(string weightName)
        {
            if (_quantWeights.TryGetValue(weightName, out var qw))
                return (int)qw.Ne1;
            if (_weights.TryGetValue(weightName, out var w))
                return (int)w.Sizes[0];
            return 0;
        }

        /// <summary>
        /// Shard a column-parallel weight whose output dimension is the
        /// concatenation of several logical segments — e.g. a fused QKV
        /// projection [Q | K | V] or a fused FFN [gate | up].
        ///
        /// The generic <see cref="ShardWeightsForTensorParallelism"/> column
        /// split assigns each rank one CONTIGUOUS block of output rows. For a
        /// concatenated weight that mixes whole segments across ranks (rank 0
        /// gets all of Q/gate, rank 1 gets all of K+V/up), which is wrong: the
        /// forward pass re-splits each rank's shard expecting it to contain that
        /// rank's slice of EVERY segment, i.e. [seg0_r | seg1_r | …].
        ///
        /// This method instead gathers, for each rank, its 1/tp slice of every
        /// segment in order, producing the per-rank layout the forward pass
        /// expects. Every segment dim must be divisible by the global TP degree
        /// (enforced by the per-model ValidateTpConstraints for head counts and
        /// intermediate sizes). No-op if the weight is not present.
        /// </summary>
        protected void ShardConcatenatedColumnParallel(string weightName, params int[] segmentDims)
        {
            if (!IsTensorParallel) return;
            int globalTp = GlobalTpDegree;

            // Output-row indices for a global rank: its 1/tp slice of each segment.
            ShardOutputRows(weightName, globalRank =>
            {
                var idx = new List<int>();
                int baseOff = 0;
                foreach (int segDim in segmentDims)
                {
                    int perRank = segDim / globalTp;
                    int start = baseOff + globalRank * perRank;
                    for (int i = 0; i < perRank; i++)
                        idx.Add(start + i);
                    baseOff += segDim;
                }
                return idx.ToArray();
            });
        }

        /// <summary>
        /// Column-parallel shard by an arbitrary set of output rows per rank, gathered in the
        /// order <paramref name="rowsOfGlobalRank"/> gives them (any head layout: segmented,
        /// block-cyclic, ...). Quantized rows are independent blocks, so they move unchanged;
        /// F32 rows are copied. No-op when the weight is absent or nothing is sharded.
        /// </summary>
        protected void ShardOutputRows(string weightName, Func<int, int[]> rowsOfGlobalRank)
        {
            if (!IsTensorParallel) return;

            int tp = TpDegree;
            int rankOffset = TpRankOffset;

            if (_quantWeights.TryGetValue(weightName, out var qw))
            {
                long rowBytes = NativeDequant.RowSize(qw.GgmlType, qw.Ne0);
                var shards = new QuantizedWeight[tp];
                for (int r = 0; r < tp; r++)
                {
                    int[] rows = rowsOfGlobalRank(rankOffset + r);
                    long totalBytes = rows.Length * rowBytes;
                    IntPtr shardPtr = QuantizedWeight.AllocateBuffer(totalBytes);
                    unsafe
                    {
                        byte* src = (byte*)qw.Data.ToPointer();
                        byte* dst = (byte*)shardPtr.ToPointer();
                        for (int row = 0; row < rows.Length; row++)
                            Buffer.MemoryCopy(
                                src + (long)rows[row] * rowBytes,
                                dst + (long)row * rowBytes,
                                rowBytes, rowBytes);
                    }
                    shards[r] = new QuantizedWeight(shardPtr, totalBytes,
                        qw.GgmlType, qw.Ne0, rows.Length);
                    // A per-tensor scale is shard-invariant: it does not depend on
                    // the output row, and it distributes over the row-parallel
                    // AllReduce, so every shard carries the parent's value.
                    shards[r].Scale = qw.Scale;
                }

                _tpQuantWeights[weightName] = shards;
                RecordTpWeightScale(weightName, qw);
                _quantWeights.Remove(weightName);
                qw.Dispose();
            }
            else if (_weights.TryGetValue(weightName, out var w))
            {
                int inDim = (int)w.Sizes[1];
                var shards = new Tensor[tp];
                for (int r = 0; r < tp; r++)
                {
                    int[] rows = rowsOfGlobalRank(rankOffset + r);
                    var shard = new Tensor(_tpGroup.GetAllocator(r), DType.Float32, rows.Length, inDim);
                    unsafe
                    {
                        float* srcPtr = GetFloatPtr(w);
                        float* dstPtr = GetFloatPtr(shard);
                        for (int row = 0; row < rows.Length; row++)
                            Buffer.MemoryCopy(
                                srcPtr + (long)rows[row] * inDim,
                                dstPtr + (long)row * inDim,
                                (long)inDim * 4, (long)inDim * 4);
                    }
                    shards[r] = shard;
                }

                _tpWeights[weightName] = shards;
                RecordTpWeightScale(weightName, null);
                _weights.Remove(weightName);
                w.Dispose();
            }
        }

        /// <summary>
        /// Row-parallel shard whose input columns are gathered in groups: rank r keeps the
        /// <paramref name="groupWidth"/>-wide column groups <paramref name="groupsOfGlobalRank"/>
        /// lists, in that order (e.g. a recurrent block's output projection, whose input is the
        /// rank's block-cyclic value heads). A quantized weight moves whole source blocks when
        /// the group width is a multiple of the block; otherwise each row is dequantized,
        /// gathered and re-encoded as Q8_0 (or kept F32 when 32 does not divide the group).
        /// </summary>
        protected void ShardInputColumnGroups(string weightName, Func<int, int[]> groupsOfGlobalRank, int groupWidth)
        {
            if (!IsTensorParallel) return;

            int tp = TpDegree;
            int rankOffset = TpRankOffset;

            if (_quantWeights.TryGetValue(weightName, out var qw))
            {
                var type = (GgmlTensorType)qw.GgmlType;
                long blockSize = GgufFile.GetBlockSize(type);
                long srcRowBytes = NativeDequant.RowSize(qw.GgmlType, qw.Ne0);
                var shards = new QuantizedWeight[tp];
                if (blockSize > 0 && groupWidth % blockSize == 0)
                {
                    long groupBytes = groupWidth / blockSize * GgufFile.GetTypeSize(type);
                    for (int r = 0; r < tp; r++)
                    {
                        int[] groups = groupsOfGlobalRank(rankOffset + r);
                        long ne0 = (long)groups.Length * groupWidth;
                        long dstRowBytes = groups.Length * groupBytes;
                        long total = qw.Ne1 * dstRowBytes;
                        IntPtr shardPtr = QuantizedWeight.AllocateBuffer(total);
                        unsafe
                        {
                            byte* src = (byte*)qw.Data.ToPointer();
                            byte* dst = (byte*)shardPtr.ToPointer();
                            for (long row = 0; row < qw.Ne1; row++)
                                for (int g = 0; g < groups.Length; g++)
                                    Buffer.MemoryCopy(
                                        src + row * srcRowBytes + groups[g] * groupBytes,
                                        dst + row * dstRowBytes + g * groupBytes,
                                        groupBytes, groupBytes);
                        }
                        shards[r] = new QuantizedWeight(shardPtr, total, qw.GgmlType, ne0, qw.Ne1) { Scale = qw.Scale };
                    }
                    _tpQuantWeights[weightName] = shards;
                }
                else if (groupWidth % 32 == 0)
                {
                    const int q8Type = (int)GgmlTensorType.Q8_0;
                    Console.WriteLine($"  {weightName}: {type} block ({blockSize} elements) does not divide the " +
                        $"{groupWidth}-column groups; re-encoding its tensor-parallel shards as Q8_0.");
                    for (int r = 0; r < tp; r++)
                    {
                        int[] groups = groupsOfGlobalRank(rankOffset + r);
                        long ne0 = (long)groups.Length * groupWidth;
                        long dstRowBytes = NativeDequant.RowSize(q8Type, ne0);
                        long total = qw.Ne1 * dstRowBytes;
                        IntPtr shardPtr = QuantizedWeight.AllocateBuffer(total);
                        unsafe
                        {
                            byte* src = (byte*)qw.Data.ToPointer();
                            byte* dst = (byte*)shardPtr.ToPointer();
                            var srcRow = new float[qw.Ne0];
                            var dstRow = new float[ne0];
                            fixed (float* sf = srcRow, df = dstRow)
                            {
                                for (long row = 0; row < qw.Ne1; row++)
                                {
                                    NativeDequant.DequantizeToFloat32Native(qw.GgmlType, (IntPtr)(src + row * srcRowBytes), (IntPtr)sf, qw.Ne0);
                                    for (int g = 0; g < groups.Length; g++)
                                        Buffer.MemoryCopy(sf + (long)groups[g] * groupWidth, df + (long)g * groupWidth,
                                            groupWidth * sizeof(float), groupWidth * sizeof(float));
                                    ManagedQuantizedOps.QuantizeRowFromFloat32(q8Type, df, (IntPtr)(dst + row * dstRowBytes), ne0);
                                }
                            }
                        }
                        shards[r] = new QuantizedWeight(shardPtr, total, q8Type, ne0, qw.Ne1) { Scale = qw.Scale };
                    }
                    _tpQuantWeights[weightName] = shards;
                }
                else
                {
                    var dense = new Tensor[tp];
                    for (int r = 0; r < tp; r++)
                    {
                        int[] groups = groupsOfGlobalRank(rankOffset + r);
                        long ne0 = (long)groups.Length * groupWidth;
                        var shard = new Tensor(_tpGroup.GetAllocator(r), DType.Float32, qw.Ne1, ne0);
                        unsafe
                        {
                            byte* src = (byte*)qw.Data.ToPointer();
                            float* dstBase = GetFloatPtr(shard);
                            var srcRow = new float[qw.Ne0];
                            fixed (float* sf = srcRow)
                            {
                                for (long row = 0; row < qw.Ne1; row++)
                                {
                                    NativeDequant.DequantizeToFloat32Native(qw.GgmlType, (IntPtr)(src + row * srcRowBytes), (IntPtr)sf, qw.Ne0);
                                    for (int g = 0; g < groups.Length; g++)
                                        Buffer.MemoryCopy(sf + (long)groups[g] * groupWidth, dstBase + row * ne0 + (long)g * groupWidth,
                                            groupWidth * sizeof(float), groupWidth * sizeof(float));
                                }
                            }
                        }
                        dense[r] = shard;
                    }
                    _tpWeights[weightName] = dense;
                }
                RecordTpWeightScale(weightName, qw);
                _quantWeights.Remove(weightName);
                qw.Dispose();
            }
            else if (_weights.TryGetValue(weightName, out var w))
            {
                int outDim = (int)w.Sizes[0];
                int inDim = (int)w.Sizes[1];
                var shards = new Tensor[tp];
                for (int r = 0; r < tp; r++)
                {
                    int[] groups = groupsOfGlobalRank(rankOffset + r);
                    int ne0 = groups.Length * groupWidth;
                    var shard = new Tensor(_tpGroup.GetAllocator(r), DType.Float32, outDim, ne0);
                    unsafe
                    {
                        float* srcPtr = GetFloatPtr(w);
                        float* dstPtr = GetFloatPtr(shard);
                        for (int row = 0; row < outDim; row++)
                            for (int g = 0; g < groups.Length; g++)
                                Buffer.MemoryCopy(srcPtr + (long)row * inDim + (long)groups[g] * groupWidth,
                                    dstPtr + (long)row * ne0 + (long)g * groupWidth,
                                    groupWidth * sizeof(float), groupWidth * sizeof(float));
                    }
                    shards[r] = shard;
                }
                _tpWeights[weightName] = shards;
                RecordTpWeightScale(weightName, null);
                _weights.Remove(weightName);
                w.Dispose();
            }
        }

        /// <summary>
        /// Shard a fused [gate | up] FFN projection as two equal column-parallel
        /// segments, deriving the half size from the weight itself. No-op if the
        /// weight is not present (e.g. a MoE layer with no dense gate_up).
        /// </summary>
        protected void ShardFusedGateUpColumnParallel(string weightName)
        {
            int fullDim = GetFusedOutputDim(weightName);
            if (fullDim <= 0) return;
            int half = fullDim / 2;
            ShardConcatenatedColumnParallel(weightName, half, half);
        }

        /// <summary>
        /// Column-parallel shard from SEPARATE source weights that were never
        /// fused (e.g. Q/K/V with mismatched quant types that prevented
        /// <see cref="ShardConcatenatedColumnParallel"/>). Each source is
        /// dequantized (if quantized) and sliced per-rank on the fly, so no
        /// full-model F32 intermediate is ever materialised. The per-rank
        /// results are stored under <paramref name="fusedName"/> - as Q8_0 in
        /// <see cref="_tpQuantWeights"/> when the input dim allows it, otherwise
        /// as F32 in <see cref="_tpWeights"/>; source weights are removed and
        /// disposed.
        /// </summary>
        protected void ShardSeparateColumnParallel(string fusedName, string[] sourceNames, int[] segmentDims)
        {
            if (!IsTensorParallel) return;
            if (sourceNames.Length != segmentDims.Length)
                throw new ArgumentException("sourceNames and segmentDims must have the same length.");

            int tp = TpDegree;
            int globalTp = GlobalTpDegree;
            int rankOffset = TpRankOffset;

            // Resolve sources and common input dim.
            int inDim = -1;
            var quants = new QuantizedWeight[sourceNames.Length];
            var tensors = new Tensor[sourceNames.Length];
            for (int s = 0; s < sourceNames.Length; s++)
            {
                if (_quantWeights.TryGetValue(sourceNames[s], out quants[s]))
                {
                    int d = (int)quants[s].Ne0;
                    if (inDim < 0) inDim = d; else if (inDim != d) return;
                }
                else if (_weights.TryGetValue(sourceNames[s], out tensors[s]))
                {
                    int d = (int)tensors[s].Sizes[1];
                    if (inDim < 0) inDim = d; else if (inDim != d) return;
                }
                else
                {
                    return; // source missing — nothing to shard
                }
            }

            // This path exists because the sources had MIXED quant types, so they
            // could not be packed in place. Materializing the shard in F32 as well
            // costs ~8x the source bytes AND drops the layer off the cached
            // quantized matmul onto the generic Ops.Addmm path, which re-uploads
            // the whole weight per call - on a mixed-quant Qwen3.8-27B that alone
            // was several GB per rank and erased the throughput tensor parallelism
            // was supposed to buy. Re-encode each gathered row as Q8_0 instead: the
            // gather is row-wise, so a row's values are untouched, and an 8-bit
            // round-trip of an already <=6-bit source is lossless in practice. Q8_0
            // blocks are 32 elements along the INPUT dim, so this needs inDim % 32.
            const int q8Type = (int)GgmlTensorType.Q8_0;
            bool requantize = (inDim % 32) == 0;

            var shards = requantize ? null : new Tensor[tp];
            var quantShards = requantize ? new QuantizedWeight[tp] : null;
            long q8RowBytes = requantize ? NativeDequant.RowSize(q8Type, inDim) : 0;

            for (int r = 0; r < tp; r++)
            {
                int globalRank = rankOffset + r;
                int totalRows = 0;
                foreach (int segDim in segmentDims)
                    totalRows += segDim / globalTp;

                Tensor shard = requantize
                    ? null
                    : new Tensor(_tpGroup.GetAllocator(r), DType.Float32, totalRows, inDim);
                IntPtr quantPtr = requantize
                    ? QuantizedWeight.AllocateBuffer((long)totalRows * q8RowBytes)
                    : IntPtr.Zero;
                // One reusable row buffer when re-quantizing: the shard is written
                // row at a time, so no full-size F32 intermediate is ever allocated.
                float[] rowScratch = requantize ? new float[inDim] : null;

                unsafe
                {
                    float* dstBase = requantize ? null : GetFloatPtr(shard);
                    byte* quantBase = requantize ? (byte*)quantPtr.ToPointer() : null;
                    long dstRow = 0;

                    fixed (float* scratch = rowScratch)
                    {
                        for (int s = 0; s < sourceNames.Length; s++)
                        {
                            int perRank = segmentDims[s] / globalTp;
                            int srcStart = globalRank * perRank;

                            for (int row = 0; row < perRank; row++)
                            {
                                float* dstRowPtr = requantize
                                    ? scratch
                                    : dstBase + (dstRow + row) * inDim;

                                if (quants[s] != null)
                                {
                                    var qw = quants[s];
                                    long rowBytes = NativeDequant.RowSize(qw.GgmlType, qw.Ne0);
                                    byte* srcBase = (byte*)qw.Data.ToPointer();
                                    NativeDequant.DequantizeToFloat32Native(
                                        qw.GgmlType,
                                        (IntPtr)(srcBase + (long)(srcStart + row) * rowBytes),
                                        (IntPtr)dstRowPtr,
                                        inDim);
                                    // The sources keep their own per-tensor sidecar scales,
                                    // but the pack has one name and so one recorded scale.
                                    // Bake each source's into its rows here. Skipped at 1 so
                                    // unscaled (UD / MXFP4 / Q*) packs stay bit-identical.
                                    if (qw.Scale != 1.0f)
                                    {
                                        float sourceScale = qw.Scale;
                                        for (int i = 0; i < inDim; i++)
                                            dstRowPtr[i] *= sourceScale;
                                    }
                                }
                                else
                                {
                                    float* srcPtr = GetFloatPtr(tensors[s]);
                                    long bytes = (long)inDim * sizeof(float);
                                    Buffer.MemoryCopy(
                                        srcPtr + (long)(srcStart + row) * inDim,
                                        dstRowPtr, bytes, bytes);
                                }

                                if (requantize)
                                    ManagedQuantizedOps.QuantizeRowFromFloat32(
                                        q8Type, dstRowPtr,
                                        (IntPtr)(quantBase + (dstRow + row) * q8RowBytes),
                                        inDim);
                            }
                            dstRow += perRank;
                        }
                    }
                }

                if (requantize)
                    quantShards[r] = new QuantizedWeight(quantPtr, (long)totalRows * q8RowBytes,
                        q8Type, inDim, totalRows);
                else
                    shards[r] = shard;
            }

            // This pack is built from several sources and every source's scale is
            // already baked into its rows above (Q8_0 and F32 alike), so record
            // nothing here: a recorded scale would apply a second time.
            if (requantize)
                _tpQuantWeights[fusedName] = quantShards;
            else
                _tpWeights[fusedName] = shards;

            for (int s = 0; s < sourceNames.Length; s++)
            {
                if (quants[s] != null)
                {
                    _quantWeights.Remove(sourceNames[s]);
                    quants[s].Dispose();
                }
                else
                {
                    _weights.Remove(sourceNames[s]);
                    tensors[s].Dispose();
                }
            }
        }

        /// <summary>
        /// Column-parallel shard of a 1-D bias whose length is the concatenation
        /// of several logical segments (fused QKV bias [Q|K|V], fused gate_up
        /// bias [gate|up], …). The bias must be regrouped exactly like the
        /// weight it accompanies (<see cref="ShardConcatenatedColumnParallel"/>),
        /// otherwise each rank would add the wrong bias slice to its outputs.
        /// No-op if the bias is not present.
        /// </summary>
        protected void ShardConcatenatedBiasColumnParallel(string biasName, params int[] segmentDims)
        {
            if (!IsTensorParallel) return;
            if (!_weights.TryGetValue(biasName, out var bias)) return;

            int tp = TpDegree;
            int globalTp = GlobalTpDegree;
            int rankOffset = TpRankOffset;

            int[] BuildRows(int globalRank)
            {
                var idx = new List<int>();
                int baseOff = 0;
                foreach (int segDim in segmentDims)
                {
                    int perRank = segDim / globalTp;
                    int start = baseOff + globalRank * perRank;
                    for (int i = 0; i < perRank; i++)
                        idx.Add(start + i);
                    baseOff += segDim;
                }
                return idx.ToArray();
            }

            var shards = new Tensor[tp];
            for (int r = 0; r < tp; r++)
            {
                int[] rows = BuildRows(rankOffset + r);
                var shard = new Tensor(_tpGroup.GetAllocator(r), DType.Float32, rows.Length);
                unsafe
                {
                    float* srcPtr = GetFloatPtr(bias);
                    float* dstPtr = GetFloatPtr(shard);
                    for (int i = 0; i < rows.Length; i++)
                        dstPtr[i] = srcPtr[rows[i]];
                }
                shards[r] = shard;
            }

            _tpWeights[biasName] = shards;
            _weights.Remove(biasName);
            bias.Dispose();
        }

        /// <summary>
        /// Column-parallel shard of a logical fused bias whose SOURCE tensors
        /// were never fused (e.g. Q/K/V biases next to mixed-quant-type Q/K/V
        /// weights that declined load-time fusion). Gathers each rank's slice
        /// of every source bias, in order, into one per-rank tensor registered
        /// under <paramref name="fusedBiasName"/> — the bias counterpart of
        /// <see cref="ShardSeparateColumnParallel"/>, producing exactly the
        /// per-rank [seg0_r | seg1_r | …] layout the forward's fused lookup
        /// expects. No-op if any source bias is absent.
        /// </summary>
        protected void ShardSeparateBiasColumnParallel(string fusedBiasName, string[] sourceNames, int[] segmentDims)
        {
            if (!IsTensorParallel) return;
            if (sourceNames.Length != segmentDims.Length)
                throw new ArgumentException("sourceNames and segmentDims must have the same length.");

            var sources = new Tensor[sourceNames.Length];
            for (int s = 0; s < sourceNames.Length; s++)
            {
                if (!_weights.TryGetValue(sourceNames[s], out sources[s]))
                    return; // biases are optional — mirror the fused-bias no-op
            }

            int tp = TpDegree;
            int globalTp = GlobalTpDegree;
            int rankOffset = TpRankOffset;

            int totalLen = 0;
            foreach (int segDim in segmentDims)
                totalLen += segDim / globalTp;

            var shards = new Tensor[tp];
            for (int r = 0; r < tp; r++)
            {
                int globalRank = rankOffset + r;
                var shard = new Tensor(_tpGroup.GetAllocator(r), DType.Float32, totalLen);
                unsafe
                {
                    float* dst = GetFloatPtr(shard);
                    int dstOff = 0;
                    for (int s = 0; s < sourceNames.Length; s++)
                    {
                        int perRank = segmentDims[s] / globalTp;
                        float* src = GetFloatPtr(sources[s]);
                        int start = globalRank * perRank;
                        for (int i = 0; i < perRank; i++)
                            dst[dstOff + i] = src[start + i];
                        dstOff += perRank;
                    }
                }
                shards[r] = shard;
            }

            _tpWeights[fusedBiasName] = shards;
            for (int s = 0; s < sourceNames.Length; s++)
            {
                _weights.Remove(sourceNames[s]);
                sources[s].Dispose();
            }
        }

        /// <summary>
        /// Column-parallel shard of a fused [gate | up] bias as two equal
        /// segments, deriving the half length from the bias itself.
        /// </summary>
        protected void ShardFusedGateUpBiasColumnParallel(string biasName)
        {
            if (!_weights.TryGetValue(biasName, out var bias)) return;
            int total = (int)bias.ElementCount();
            int half = total / 2;
            ShardConcatenatedBiasColumnParallel(biasName, half, half);
        }

        internal static readonly bool TpScaleDebug =
            Environment.GetEnvironmentVariable("TS_TP_SCALE_DEBUG") == "1";
        private readonly HashSet<string> _tpScaleSeen = new(StringComparer.Ordinal);

        /// <summary>Remember a sharded weight's per-tensor scale under the name
        /// the TP linears resolve it by. Tolerates a null source (an F32 shard
        /// cut from a tensor that never had one).</summary>
        protected void RecordTpWeightScale(string weightName, QuantizedWeight source)
        {
            if (weightName == null)
                return;
            float s = source?.Scale ?? 1.0f;
            if (s != 1.0f)
            {
                _tpWeightScales[weightName] = s;
                if (TpScaleDebug)
                    Console.Error.WriteLine($"[tp-scale] recorded {weightName} = {s}");
            }
        }

        /// <summary>Whether any of these sharded weights carries a per-tensor
        /// scale. The fused per-block TP kernels have no hook to apply one, so
        /// they must decline rather than silently drop it.</summary>
        protected bool TpAnyWeightScaled(params string[] weightNames)
        {
            if (_tpWeightScales.Count == 0 || weightNames == null)
                return false;
            foreach (string n in weightNames)
                if (n != null && _tpWeightScales.ContainsKey(n))
                    return true;
            return false;
        }

        /// <summary>Multiply every rank's result by the sharded weight's
        /// per-tensor scale. Shard-invariant on both splits: column-parallel
        /// divides output rows and the scalar does not depend on the row, and
        /// row-parallel divides the contraction, where scaling each partial
        /// before the all-reduce equals scaling the sum.</summary>
        protected void TpApplyNamedWeightScale(Tensor[] results, string weightName)
        {
            if (TpScaleDebug)
            {
                bool found = _tpWeightScales.TryGetValue(weightName, out float dbg);
                if (_tpScaleSeen.Add(weightName))
                    Console.Error.WriteLine($"[tp-scale] {weightName}: " +
                        (found ? $"applying {dbg}" : "no recorded scale"));
            }
            if (results == null || !_tpWeightScales.TryGetValue(weightName, out float s))
                return;
            for (int r = 0; r < results.Length; r++)
                if (results[r] != null)
                    Ops.Mul(results[r], results[r], s);
        }

        /// <summary>
        /// Column-parallel linear forward: each GPU computes its output slice
        /// using its weight shard. Input is replicated across all GPUs.
        /// Returns one output tensor per GPU, each with outDim/tp columns.
        /// </summary>
        protected Tensor[] TpColumnParallelLinear(Tensor input, string weightName)
        {
            int tp = TpDegree;
            var results = new Tensor[tp];
            long t0 = Stopwatch.GetTimestamp();

            if (_tpQuantWeights.TryGetValue(weightName, out var qShards))
            {
                int seqLen = (int)input.Sizes[0];
                _tpGroup.RunPerRank(r =>
                {
                    var qw = qShards[r];
                    int outDim = (int)qw.Ne1;
                    var alloc = _tpGroup.GetAllocator(r);
                    results[r] = new Tensor(alloc, DType.Float32, seqLen, outDim);
                    var localInput = ReplicateTensorToRank(input, r);
                    AddmmQuantManaged(results[r], localInput, qw);
                    // ReplicateTensorToRank allocates a fresh cross-GPU copy for
                    // every rank > 0; without this it leaked one [seqLen, inDim]
                    // tensor per rank per call, on every layer of every token.
                    if (!ReferenceEquals(localInput, input)) localInput.Dispose();
                });
            }
            else if (_tpWeights.TryGetValue(weightName, out var wShards))
            {
                int seqLen = (int)input.Sizes[0];
                _tpGroup.RunPerRank(r =>
                {
                    var w = wShards[r];
                    int outDim = (int)w.Sizes[0];
                    var alloc = _tpGroup.GetAllocator(r);
                    using var wT = w.Transpose();
                    results[r] = new Tensor(alloc, DType.Float32, seqLen, outDim);
                    var localInput = ReplicateTensorToRank(input, r);
                    Ops.Addmm(results[r], 0, results[r], 1.0f, localInput, wT);
                    if (!ReferenceEquals(localInput, input)) localInput.Dispose();
                });
            }
            else
            {
                throw new KeyNotFoundException($"TP column-parallel weight '{weightName}' not found in sharded weights.");
            }

            // Every branch above lands here: the fused multi-rank linear, the
            // generic per-rank quantized loop, and the F32 shards.
            TpApplyNamedWeightScale(results, weightName);
            _linearTicks += Stopwatch.GetTimestamp() - t0;
            return results;
        }

        /// <summary>
        /// Row-parallel linear forward: each GPU computes a partial result using
        /// its weight shard, then AllReduce sums the partials. Returns the
        /// reduced result (replicated on all GPUs); the caller typically uses
        /// the rank-0 tensor and disposes the rest.
        /// </summary>
        protected Tensor TpRowParallelLinear(Tensor[] inputs, string weightName)
        {
            int tp = TpDegree;
            var partials = new Tensor[tp];
            long t0 = Stopwatch.GetTimestamp();

            if (_tpQuantWeights.TryGetValue(weightName, out var qShards))
            {
                int seqLen = (int)inputs[0].Sizes[0];
                _tpGroup.RunPerRank(r =>
                {
                    var qw = qShards[r];
                    int rows = (int)inputs[r].Sizes[0];
                    int outDim = (int)qw.Ne1;
                    var alloc = _tpGroup.GetAllocator(r);
                    partials[r] = new Tensor(alloc, DType.Float32, rows, outDim);
                    AddmmQuantManaged(partials[r], inputs[r], qw);
                });
            }
            else if (_tpWeights.TryGetValue(weightName, out var wShards))
            {
                _tpGroup.RunPerRank(r =>
                {
                    var w = wShards[r];
                    int seqLen = (int)inputs[r].Sizes[0];
                    int outDim = (int)w.Sizes[0];
                    var alloc = _tpGroup.GetAllocator(r);
                    using var wT = w.Transpose();
                    partials[r] = new Tensor(alloc, DType.Float32, seqLen, outDim);
                    Ops.Addmm(partials[r], 0, partials[r], 1.0f, inputs[r], wT);
                });
            }
            else
            {
                throw new KeyNotFoundException($"TP row-parallel weight '{weightName}' not found in sharded weights.");
            }

            _linearTicks += Stopwatch.GetTimestamp() - t0;

            // Same as TpRowParallelLinearAllRanks: a per-tensor scale distributes over
            // the sum, so each partial is scaled before the AllReduce. A no-op unless
            // the family opted into SupportsTensorParallelWeightScales and this weight
            // carries a sidecar scale; without it an NVFP4 attn_output / ffn_down
            // under --tp came out 1/scale2 too large.
            TpApplyNamedWeightScale(partials, weightName);

            _tpGroup.AllReduce(partials);

            // Dispose non-zero-rank tensors; caller uses rank-0 result.
            for (int r = 1; r < tp; r++)
                partials[r].Dispose();

            return partials[0];
        }

        /// <summary>
        /// Row-parallel linear that keeps every rank's copy of the reduced result.
        /// AllReduce already leaves the sum on all GPUs, so callers that need the
        /// value replicated (a residual add on each rank, say) should use this
        /// instead of <see cref="TpRowParallelLinear"/> followed by
        /// <see cref="BroadcastTensorToAllRanks"/> — that pair throws away the
        /// reduced copies and then re-sends rank 0's, costing an extra full
        /// cross-GPU broadcast per call (and on a box without working P2P, an
        /// extra host round trip).
        /// </summary>
        protected Tensor[] TpRowParallelLinearAllRanks(Tensor[] inputs, string weightName)
        {
            int tp = TpDegree;
            var partials = new Tensor[tp];
            long t0 = Stopwatch.GetTimestamp();

            if (_tpQuantWeights.TryGetValue(weightName, out var qShards))
            {
                int seqLen = (int)inputs[0].Sizes[0];
                _tpGroup.RunPerRank(r =>
                {
                    var qw = qShards[r];
                    int rows = (int)inputs[r].Sizes[0];
                    var alloc = _tpGroup.GetAllocator(r);
                    partials[r] = new Tensor(alloc, DType.Float32, rows, (int)qw.Ne1);
                    AddmmQuantManaged(partials[r], inputs[r], qw);
                });
            }
            else if (_tpWeights.TryGetValue(weightName, out var wShards))
            {
                _tpGroup.RunPerRank(r =>
                {
                    var w = wShards[r];
                    int seqLen = (int)inputs[r].Sizes[0];
                    var alloc = _tpGroup.GetAllocator(r);
                    using var wT = w.Transpose();
                    partials[r] = new Tensor(alloc, DType.Float32, seqLen, (int)w.Sizes[0]);
                    Ops.Addmm(partials[r], 0, partials[r], 1.0f, inputs[r], wT);
                });
            }
            else
            {
                throw new KeyNotFoundException($"TP row-parallel weight '{weightName}' not found in sharded weights.");
            }

            _linearTicks += Stopwatch.GetTimestamp() - t0;

            TpApplyNamedWeightScale(partials, weightName);

            _tpGroup.AllReduce(partials);
            return partials;
        }

        /// <summary>
        /// Ensure a tensor is available on the given rank's GPU. If the tensor
        /// is already on that GPU (rank 0 typically), returns it as-is.
        /// Otherwise copies it to the target GPU.
        /// </summary>
        protected Tensor ReplicateTensorToRank(Tensor tensor, int rank)
        {
            if (rank == 0) return tensor;

            // GGML tensors live in host memory that every rank's backend can read
            // directly, so "replicating" is just handing over the same buffer.
            // Copying would add a full activation-sized memcpy per rank per
            // linear for no benefit.
            if (IsGgmlBackend) return tensor;

            var alloc = _tpGroup.GetAllocator(rank);
            var copy = new Tensor(alloc, tensor.ElementType, tensor.Sizes);
            Ops.Copy(copy, tensor);
            return copy;
        }

        /// <summary>
        /// Broadcast a rank-0 tensor to all other GPUs. Returns an array where
        /// element 0 is the original tensor and elements 1..tp-1 are copies.
        /// </summary>
        protected Tensor[] BroadcastTensorToAllRanks(Tensor tensor)
        {
            int tp = TpDegree;
            var result = new Tensor[tp];

            // Note: do NOT collapse the GGML case to one shared buffer, even
            // though every rank's backend can read the same host memory. Callers
            // accumulate into these per rank (TpResidualAdd is
            // `hidden[r] += residual[r]` under RunPerRank), so aliasing them
            // would apply the same residual tp times — concurrently, on one
            // buffer. The copy below is what keeps that correct.

            // Clone for rank 0 too — the caller may dispose the original
            // tensor after broadcasting, and sharing the storage would leave
            // rank 0 with a dangling reference.
            var alloc0 = _tpGroup.GetAllocator(0);
            result[0] = new Tensor(alloc0, tensor.ElementType, tensor.Sizes);
            Ops.Copy(result[0], tensor);
            for (int r = 1; r < tp; r++)
                result[r] = ReplicateTensorToRank(tensor, r);
            return result;
        }

        // Per-rank copies of REPLICATED weights (norm alphas, gate vectors) that a
        // rank-r kernel reads directly. The source lives on rank 0, so a rank-r
        // kernel reading it would cross GPUs — fault without peer access, wrong
        // data with it. Cached because these are re-read on every layer of every
        // token: re-copying per call cost one cross-GPU transfer per norm per
        // rank per token, which on a host-staged (no working P2P) box dominated
        // the forward pass.
        private Dictionary<(Tensor, int), Tensor> _tpWeightReplicaCache;

        /// <summary>
        /// A replicated weight resident on rank <paramref name="rank"/>'s GPU.
        /// Rank 0 aliases the original; other ranks get a lazily-cached copy.
        /// </summary>
        protected Tensor TpReplicatedWeight(Tensor weight, int rank)
        {
            if (rank == 0 || weight == null)
                return weight;

            _tpWeightReplicaCache ??= new Dictionary<(Tensor, int), Tensor>();
            var key = (weight, rank);
            if (!_tpWeightReplicaCache.TryGetValue(key, out var copy))
            {
                copy = ReplicateTensorToRank(weight, rank);
                _tpWeightReplicaCache[key] = copy;
            }
            return copy;
        }

        protected void DisposeTpWeightReplicaCache()
        {
            if (_tpWeightReplicaCache == null)
                return;
            // Every entry is a rank >= 1 copy (rank 0 aliases the original and is
            // never stored), so all of them are ours to free.
            foreach (var copy in _tpWeightReplicaCache.Values)
                copy?.Dispose();
            _tpWeightReplicaCache.Clear();
            _tpWeightReplicaCache = null;
        }

        /// <summary>
        /// TP-aware RMSNorm: runs independently on each GPU (replicated weights).
        /// </summary>
        protected Tensor[] TpRMSNorm(Tensor[] inputs, string weightName)
        {
            int tp = TpDegree;
            var results = new Tensor[tp];
            var alpha = _weights[weightName];

            // Materialize every rank's replica up front: TpReplicatedWeight
            // populates a shared dictionary, which is not safe to fill from the
            // concurrent rank workers below.
            var alphaLocal = new Tensor[tp];
            for (int r = 0; r < tp; r++)
                alphaLocal[r] = TpReplicatedWeight(alpha, r);

            _tpGroup.RunPerRank(r =>
            {
                int rows = (int)inputs[r].Sizes[0];
                int dim = (int)(inputs[r].ElementCount() / rows);
                Tensor input2d = inputs[r].Sizes.Length != 2 ? inputs[r].View(rows, dim) : null;
                Tensor src = input2d ?? inputs[r];

                results[r] = Ops.RMSNorm(null, src, alphaLocal[r], null, Config.Eps);

                input2d?.Dispose();
            });

            return results;
        }

        /// <summary>
        /// TP-aware residual add: hidden[r] += residual[r] on each GPU.
        /// </summary>
        protected void TpResidualAdd(Tensor[] hidden, Tensor[] residual)
        {
            _tpGroup.RunPerRank(r => Ops.Add(hidden[r], hidden[r], residual[r]));
        }

        /// <summary>
        /// GGML counterpart of <see cref="PrepareCudaQuantizedWeightsForInferenceTP"/>:
        /// upload rank r's shard of every sharded weight to rank r's GPU, and the
        /// unsharded weights to rank 0.
        ///
        /// This is what actually splits the model across the GPUs. The shards are
        /// keyed in the native cache by host pointer *per rank*, so rank r's
        /// matmul finds its slice already resident on its own device and no GPU
        /// holds more than its share.
        ///
        /// Host copies are kept: unlike direct CUDA, the GGML path reads
        /// activations and any non-resident weight straight out of host memory,
        /// and the mmap'd GGUF pages cost no extra RAM.
        /// </summary>
        private void PrepareGgmlQuantizedWeightsForInferenceTP()
        {
            if (_cudaQuantWeightsPrepared || _tpGroup == null)
                return;

            EnsureQuantBackendAvailable();

            int tp = TpDegree;
            long[] bytesPerRank = new long[tp];
            int[] countPerRank = new int[tp];
            int tooLarge = 0;

            // Upload every rank concurrently: each rank's worker pushes only its
            // own shards (rank 0 also takes the replicated weights), so the
            // host reads — page faults against the memory-mapped GGUF included —
            // and the PCIe transfers to different GPUs overlap instead of
            // running one after another. RunPerRank pins each worker to its
            // rank, and the native per-rank buffer caches are mutex-protected,
            // so the concurrency here matches what inference already does.
            _tpGroup.RunPerRank(r =>
            {
                foreach (var kv in _tpQuantWeights)
                {
                    var qw = kv.Value[r];
                    // Same residency policy as the single-GPU preload: a routed
                    // expert belonging to a --n-cpu-moe layer is multiplied on the
                    // host, and uploading its shard would spend exactly the VRAM
                    // the flag exists to save.
                    if (!qw.HasHostData || !ShouldPreloadCudaQuantWeightToDevice(kv.Key))
                        continue;

                    // A zero-sized (or otherwise degenerate) shard is a producer
                    // bug in the model's TP sharding code. Say WHICH weight it is:
                    // the native preload can only report "invalid cache key, host
                    // data, and size", which is undiagnosable across the ~2400
                    // tensors of a real model.
                    if (qw.RawBytes <= 0 || qw.Ne0 <= 0 || qw.Ne1 <= 0)
                        throw new InvalidOperationException(
                            $"TP shard for weight '{kv.Key}' (rank {r}) is degenerate: " +
                            $"type={(GgmlTensorType)qw.GgmlType}, ne0={qw.Ne0}, ne1={qw.Ne1}, rawBytes={qw.RawBytes}. " +
                            "This is a bug in the model's TP weight sharding.");

                    IntPtr cacheKey = qw.EnsureDeviceCacheKey();
                    if (!GgmlBasicOps.PreloadQuantizedWeight(cacheKey, qw.Data, qw.GgmlType, qw.Ne0, qw.Ne1, qw.RawBytes))
                    {
                        qw.MarkDevicePreloadTooLarge();
                        Interlocked.Increment(ref tooLarge);
                        continue;
                    }
                    bytesPerRank[r] += qw.RawBytes;
                    countPerRank[r]++;
                }

                // Replicated weights (embeddings, the LM head, anything the
                // model did not shard) stay on rank 0: they are read by the
                // pre/post-transformer stages, which run there. Duplicating
                // them on every GPU would undo a chunk of the memory saving TP
                // just bought.
                if (r != 0)
                    return;
                foreach (var kv in _quantWeights)
                {
                    QuantizedWeight qw = kv.Value;
                    if (!qw.HasHostData || !ShouldPreloadCudaQuantWeightToDevice(kv.Key))
                        continue;
                    // A source weight that was folded into a TP shard has no reader
                    // left under tensor parallelism, but it is still sitting in
                    // _quantWeights. Uploading it puts a full unsharded copy of that
                    // tensor on rank 0 on top of the shards - which is exactly the
                    // "TP still loads the whole model on each GPU" complaint.
                    if (IsSupersededByTpShard(kv.Key))
                        continue;
                    if (string.Equals(kv.Key, "token_embd.weight", StringComparison.Ordinal)
                        && !CanUseGgmlQuantizedGetRows(qw.GgmlType)
                        && (_quantWeights.ContainsKey("output.weight") || _weights.ContainsKey("output.weight")))
                        continue;

                    IntPtr cacheKey = qw.EnsureDeviceCacheKey();
                    if (!GgmlBasicOps.PreloadQuantizedWeight(cacheKey, qw.Data, qw.GgmlType, qw.Ne0, qw.Ne1, qw.RawBytes))
                    {
                        qw.MarkDevicePreloadTooLarge();
                        Interlocked.Increment(ref tooLarge);
                        continue;
                    }
                    bytesPerRank[0] += qw.RawBytes;
                    countPerRank[0]++;
                }
            });

            // Architecture-specific per-rank weights that are not plain
            // QuantizedWeight shards (Gemma 4's stacked expert slices). A second
            // fan-out (rather than a tail call in the first) keeps the hook free
            // to assume every generic shard is already resident.
            _tpGroup.RunPerRank(r => PreloadGgmlTpAuxiliaryWeightsForRank(r, bytesPerRank, countPerRank));

            for (int r = 0; r < tp; r++)
            {
                Console.WriteLine($"  TP rank {r}: {countPerRank[r]} weight(s), {bytesPerRank[r] / 1024 / 1024} MB resident on GPU {_ggmlContext.DeviceIds[r]}");
            }
            if (tooLarge > 0)
                Console.WriteLine($"  {tooLarge} weight(s) exceeded the device single-buffer limit and stream from host memory.");

            // The banner above counts only quantized shards. Two other categories
            // decide whether tensor parallelism actually shrank per-GPU memory, and
            // neither was ever reported -- so "each GPU still loads the whole model"
            // could not be confirmed or refuted from a normal load log:
            //   * _tpWeights: shards that had to be DEQUANTIZED to F32 (mixed-quant
            //     Q/K/V, the packed GDN in-projection). These are ~8x the quantized
            //     bytes AND are served by the uncached generic matmul.
            //   * _quantWeights: whatever was never sharded at all, which stays
            //     replicated on rank 0 on top of its shards.
            long tpF32Bytes = 0;
            int tpF32Count = 0;
            foreach (var kv in _tpWeights)
            {
                var shards = kv.Value;
                if (shards == null) continue;
                tpF32Count++;
                foreach (var t in shards)
                    if (t != null) tpF32Bytes += t.ElementCount() * t.ElementType.Size();
            }
            if (tpF32Count > 0)
            {
                Console.WriteLine($"  TP F32 shards: {tpF32Count} weight(s), {tpF32Bytes / 1024 / 1024} MB across all ranks " +
                    "(dequantized because the source could not be split in its quant type; " +
                    "these also bypass the cached quantized matmul).");
            }
            long replicatedBytes = 0, supersededBytes = 0;
            int replicatedCount = 0, supersededCount = 0;
            foreach (var kv in _quantWeights)
            {
                if (kv.Value == null) continue;
                // Host-only lookup tables and stacked expert views are not replicated device weights.
                if (!ShouldPreloadCudaQuantWeightToDevice(kv.Key)) continue;
                if (IsSupersededByTpShard(kv.Key))
                {
                    supersededCount++;
                    supersededBytes += kv.Value.RawBytes;
                }
                else
                {
                    replicatedCount++;
                    replicatedBytes += kv.Value.RawBytes;
                }
            }
            if (replicatedCount > 0)
            {
                Console.WriteLine($"  TP unsharded on rank 0: {replicatedCount} quantized weight(s), " +
                    $"{replicatedBytes / 1024 / 1024} MB.");
            }

            // A large unsharded trunk can limit scaling. This is a weight-size
            // diagnostic, not a measured speed comparison. Sum the actual
            // shards because uneven partitions give ranks different sizes.
            long shardedBytes = AdditionalTpShardedBytes;
            foreach (var kv in _tpQuantWeights)
                if (kv.Value != null)
                    foreach (var shard in kv.Value)
                        if (shard != null) shardedBytes += shard.RawBytes;
            if (replicatedBytes > 0 && shardedBytes < replicatedBytes)
            {
                Console.Error.WriteLine(
                    $"WARNING: tensor parallelism sharded only {shardedBytes / 1024 / 1024} MB across " +
                    $"{TpDegree} ranks while {replicatedBytes / 1024 / 1024} MB remains unsharded on rank 0. " +
                    "Unsharded work and collective overhead may limit scaling; benchmark the selected " +
                    "configuration before assuming a speedup.");
            }
            if (supersededCount > 0)
            {
                Console.WriteLine($"  TP not resident: {supersededCount} fusion-source weight(s), " +
                    $"{supersededBytes / 1024 / 1024} MB (superseded by a shard; host mapping kept).");
            }

            GgmlBasicOps.SetActiveRank(0);
            _cudaQuantWeightsPrepared = true;
        }

        /// <summary>
        /// Hook for per-rank weights that do not live in
        /// <see cref="_tpQuantWeights"/> — Gemma 4's stacked expert slices, for
        /// instance. Called once per rank from a <c>RunPerRank</c> fan-out, so
        /// every rank uploads concurrently: the calling thread is already pinned
        /// to <paramref name="rank"/>'s GPU (do NOT call
        /// <c>GgmlBasicOps.SetActiveRank</c>), the implementation must touch only
        /// that rank's shards, and it must add what it uploaded to that rank's
        /// slot in the running totals so the load report stays accurate.
        /// </summary>
        protected virtual void PreloadGgmlTpAuxiliaryWeightsForRank(int rank, long[] bytesPerRank, int[] countPerRank) { }

        /// <summary>
        /// True when <paramref name="weightName"/> is still present in
        /// <see cref="_quantWeights"/> only because a fusion kept its sources alive,
        /// and tensor parallelism reads the fused/sharded tensor instead. Such a
        /// weight must not be uploaded to rank 0: it would duplicate, unsharded, a
        /// tensor the TP path already holds in shards. The host mapping stays, so a
        /// rare non-TP reader can still stream it.
        /// </summary>
        protected virtual bool IsSupersededByTpShard(string weightName) => false;

        /// <summary>
        /// Bytes that tensor parallelism really did split but that do NOT live in
        /// <see cref="_tpQuantWeights"/> - the expert-parallel stacked slices, for
        /// instance, which are per-rank views rather than QuantizedWeight shards.
        /// Counted alongside <see cref="_tpQuantWeights"/> when judging whether the
        /// split was worthwhile, so an architecture that shards the bulk of its
        /// weight through that route is not accused of having sharded almost nothing.
        /// </summary>
        protected virtual long AdditionalTpShardedBytes => 0;

        /// <summary>
        /// True when the MoE layers under TP fall back to the per-token,
        /// per-expert dispatch loop, which is slow enough that startup needs the
        /// lightweight warmup. Architectures with a batched per-rank MoE path
        /// override this to false.
        /// </summary>
        protected virtual bool MoEUnderTpIsSlow => IsTensorParallel && HasMoEExpertWeights;

        /// <summary>
        /// Preload TP-sharded quantized weights onto their respective GPUs.
        /// Each shard is uploaded to the GPU that will use it, then the host
        /// copy is released.
        /// </summary>
        protected void PrepareCudaQuantizedWeightsForInferenceTP()
        {
            if (_backend == BackendType.GgmlCuda || _backend == BackendType.GgmlVulkan)
            {
                PrepareGgmlQuantizedWeightsForInferenceTP();
                return;
            }

            if (_backend != BackendType.Cuda || _tpQuantWeights.Count == 0)
                return;

            // When CUDA kernels are unavailable (PTX load failed), device-side
            // quantized matmul will fail and every op falls back to the CPU
            // dequant path.  In that case we MUST keep the host data alive
            // regardless of ShouldRetainCudaHostQuantWeight.
            bool kernelsAvailable = _allocator is CudaAllocator primaryAlloc
                && CudaQuantizedOps.AreKernelsAvailable(primaryAlloc);

            long preloadedBytes = 0;
            int preloadedCount = 0;

            long tpTotalBytes = 0;
            int tpTotalCount = 0;
            int tpDebugPrinted = 0;
            foreach (var kv in _tpQuantWeights)
            {
                foreach (var s in kv.Value)
                {
                    tpTotalBytes += s.RawBytes;
                    tpTotalCount++;
                }
                if (tpDebugPrinted < 3)
                {
                    var firstShard = kv.Value[0];
                    Console.WriteLine($"    TP debug: {kv.Key} shards={kv.Value.Length} shardRawBytes={firstShard.RawBytes} ne0={firstShard.Ne0} ne1={firstShard.Ne1} type={firstShard.GgmlType}");
                    tpDebugPrinted++;
                }
            }
            Console.WriteLine($"  TP sharded weights to preload: {tpTotalCount} shards, {tpTotalBytes / 1024 / 1024} MB total");

            int skippedExpertShards = 0;
            var oomShardsPerRank = new int[TpDegree];
            var oomBytesPerRank = new long[TpDegree];
            foreach (var kv in _tpQuantWeights)
            {
                var shards = kv.Value;
                bool retain = !kernelsAvailable || ShouldRetainCudaHostQuantWeight(kv.Key);
                // MoE expert shards are numerous (experts × layers × ranks) but they
                // are also the overwhelming majority of the model's bytes — on
                // Qwen3.5-35B-A3B they are ~15.6 GB of a 17 GB checkpoint. Keeping
                // them on the host (as this path used to) meant TP loaded almost
                // nothing into VRAM and every routed expert ran the CPU dequant
                // fallback in AddmmQuantManaged, which is what made TP prefill
                // ~100x slower than single-GPU. Each rank only holds a 1/tp slice,
                // so the sharded experts fit comfortably; the OOM catch below still
                // degrades gracefully to the host path if a GPU really is too small.
                for (int r = 0; r < shards.Length; r++)
                {
                    var qw = shards[r];
                    if (!qw.HasHostData || !CudaQuantizedOps.SupportsQuantizedType(qw.GgmlType)
                        // A routed expert on a --n-cpu-moe layer stays in system
                        // RAM; uploading its shard would spend exactly the VRAM
                        // the flag exists to save.
                        || !ShouldPreloadCudaQuantWeightToDevice(kv.Key))
                    {
                        if (kv.Key.Contains("_exps."))
                            skippedExpertShards++;
                        continue;
                    }

                    var alloc = (CudaAllocator)_tpGroup.GetAllocator(r);
                    IntPtr cacheKey = qw.EnsureDeviceCacheKey();
                    try
                    {
                        CudaQuantizedOps.PreloadQuantizedWeight(
                            alloc, cacheKey, qw.Data, qw.GgmlType, qw.Ne0, qw.Ne1, qw.RawBytes);
                    }
                    catch (Exception ex) when (ex.Message.Contains("out of memory"))
                    {
                        // With experts now preloaded this can fire thousands of
                        // times on an undersized GPU, so report the first per rank
                        // and tally the rest.
                        if (oomShardsPerRank[r]++ == 0)
                        {
                            Console.WriteLine($"  Skipping TP device preload for {kv.Key} rank={r} ({qw.RawBytes / 1024 / 1024} MB) — GPU {r} out of memory, keeping host copy for CPU fallback.");
                        }
                        oomBytesPerRank[r] += qw.RawBytes;
                        continue;
                    }
                    preloadedBytes += qw.RawBytes;
                    preloadedCount++;
                    if (!retain)
                        qw.ReleaseHostData();
                }
            }

            // Also preload any remaining non-sharded quantized weights (e.g. embedding).
            if (_allocator is CudaAllocator cudaAllocator)
            {
                long remainingTotalBytes = 0;
                int remainingCount = 0;
                foreach (var kv in _quantWeights)
                {
                    remainingTotalBytes += kv.Value.RawBytes;
                    remainingCount++;
                }
                if (remainingCount > 0)
                    Console.WriteLine($"  TP non-sharded quantized weights remaining: {remainingCount} tensors, {remainingTotalBytes / 1024 / 1024} MB");

                foreach (var kv in _quantWeights)
                {
                    var qw = kv.Value;
                    if (!qw.HasHostData || !CudaQuantizedOps.SupportsQuantizedType(qw.GgmlType))
                        continue;

                    IntPtr cacheKey = qw.EnsureDeviceCacheKey();
                    try
                    {
                        CudaQuantizedOps.PreloadQuantizedWeight(
                            cudaAllocator, cacheKey, qw.Data, qw.GgmlType, qw.Ne0, qw.Ne1, qw.RawBytes);
                    }
                    catch (Exception ex) when (ex.Message.Contains("out of memory"))
                    {
                        // GPU is full — keep the host copy so the CPU fallback
                        // path (EmbeddingManagedQuantized / AddmmQuantManaged)
                        // can serve this weight from host memory.
                        Console.WriteLine($"  Skipping device preload for {kv.Key} ({qw.RawBytes / 1024 / 1024} MB) — GPU out of memory, keeping host copy for CPU fallback.");
                        continue;
                    }
                    preloadedBytes += qw.RawBytes;
                    preloadedCount++;
                    if (kernelsAvailable && !ShouldRetainCudaHostQuantWeight(kv.Key))
                        qw.ReleaseHostData();
                }
            }

            _cudaQuantWeightsPrepared = true;

            // Only dispose the GGUF file when no external host views remain
            // (column-parallel shards may still reference mmap'd data) and
            // no TP shards were kept on host (expert CPU fallback).
            bool anyHostViewsRemain = false;
            foreach (var kv in _tpQuantWeights)
            {
                foreach (var qw in kv.Value)
                {
                    if (qw.HasHostData)
                    {
                        anyHostViewsRemain = true;
                        break;
                    }
                }
                if (anyHostViewsRemain) break;
            }
            if (!anyHostViewsRemain)
            {
                foreach (QuantizedWeight qw in _quantWeights.Values)
                {
                    if (qw.HasHostData)
                    {
                        anyHostViewsRemain = true;
                        break;
                    }
                }
            }
            if (!anyHostViewsRemain)
                _gguf?.Dispose();

            if (preloadedCount > 0)
                Console.WriteLine($"  TP CUDA resident quantized weights: {preloadedBytes / 1024 / 1024} MB across {preloadedCount} shards (host copies released)");

            for (int r = 0; r < oomShardsPerRank.Length; r++)
            {
                if (oomShardsPerRank[r] > 0)
                {
                    Console.WriteLine($"  TP GPU {r}: {oomShardsPerRank[r]} shards ({oomBytesPerRank[r] / 1024 / 1024} MB) did not fit in VRAM and stay on the host (CPU dequant fallback — expect much slower decode).");
                }
            }
            if (skippedExpertShards > 0)
            {
                Console.WriteLine($"  TP: {skippedExpertShards} expert shards have no CUDA quantized kernel for their type and stay on the host.");
            }

            for (int r = 0; r < TpDegree; r++)
            {
                if (_tpGroup.GetAllocator(r) is CudaAllocator rankAlloc)
                    rankAlloc.LogVram($"after TP quant weight preload (rank {r})");
            }
        }

        #endregion

    }
}
