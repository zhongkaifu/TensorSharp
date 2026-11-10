// Copyright (c) Zhongkai Fu. All rights reserved.
// https://github.com/zhongkaifu/TensorSharp
//
// This file is part of TensorSharp.
//
// TensorSharp is licensed under the BSD-3-Clause license found in the LICENSE file in the root directory of this source tree.
using System;
using System.Collections.Generic;

using TensorSharp.GGML;
using TensorSharp.Runtime.Paged;
using TensorSharp.Runtime.Scheduling;

namespace TensorSharp.Models
{
    /// <summary>
    /// Continuous batching for every shared-runtime family, on every placement: one forward
    /// packs every scheduled token of every sequence, scatters K/V into per-(layer, rank) paged
    /// buffers by slot mapping and runs per-sequence causal attention over the rank's heads -
    /// the native paged-attention kernel on GGML backends, the managed kernel elsewhere. Tensor
    /// parallelism and whole-layer stages need nothing extra here: the same layer loop that
    /// serves one sequence serves the batch.
    /// </summary>
    public abstract partial class StandardDecoderModel : IBatchedPagedModel
    {
        private float[][][] _pagedK;   // [layer][rank]
        private float[][][] _pagedV;
        private int _pagedBlocks;
        private int _pagedBlockSize;
        private readonly object _nativePagedAttentionLock = new();

        /// <summary>Number of paged batched forwards served (tests check the engine used them).</summary>
        internal int BatchedForwardCount { get; private set; }

        /// <summary>Per-step metadata of a paged batch.</summary>
        internal sealed class PagedBatch
        {
            public required int NumSequences { get; init; }
            public required int[] QueryStart { get; init; }
            public required int[] SequenceLengths { get; init; }
            public required int[] Positions { get; init; }
            public required int[] SlotMapping { get; init; }
            public required int[][] BlockTables { get; init; }
            public int[] FlatBlockTables { get; init; }
            public int[] BlockTableOffsets { get; init; }
        }

        /// <summary>
        /// The paged buffers are F32 and nothing migrates a block-quantized linear cache into
        /// them, so a q8_0/q4_0 KV cache keeps the snapshot path (byte-exact for those dtypes).
        /// A multi-node pipeline uses the serialized protocol, which carries no per-sequence
        /// metadata.
        /// </summary>
        public bool BatchedForwardAvailable => !_kvCacheDtype.IsBlockQuantized() && LayerSplitGroup == null;

        public IReadOnlyList<float[]> ForwardBatch(BatchedForwardContext ctx)
        {
            // A step with a text-LoRA adapter selected must apply it to every planned weight.
            bool loraAudit = OpenLoraForwardAudit();
            try
            {
                var logits = ForwardBatchCore(ctx);
                if (loraAudit) VerifyLoraForwardAudit(logits);
                return logits;
            }
            finally
            {
                if (loraAudit) CloseLoraForwardAudit();
            }
        }

        private IReadOnlyList<float[]> ForwardBatchCore(BatchedForwardContext ctx)
        {
            ArgumentNullException.ThrowIfNull(ctx);
            if (!BatchedForwardAvailable)
                throw new NotSupportedException("This placement serves concurrent requests through the per-sequence path.");
            int numSeqs = ctx.Sequences.Count;
            if (numSeqs == 0) return Array.Empty<float[]>();

            int blockSize = ctx.Sequences[0].BlockTable.BlockSize;
            int maxBlock = 0;
            foreach (int[] table in ctx.BlockTables)
                foreach (int block in table)
                    maxBlock = Math.Max(maxBlock, block);
            EnsurePagedBuffers(maxBlock + 1, blockSize);

            int numTokens = 0;
            for (int s = 0; s < numSeqs; s++) numTokens += ctx.NumScheduledTokens[s];
            int[] positions = ctx.Positions.ToArray();
            int[] queryStart = ctx.QueryStartLoc.ToArray();
            int[] seqLens = new int[numSeqs];
            var lastRows = new int[numSeqs];
            for (int s = 0; s < numSeqs; s++)
            {
                seqLens[s] = ctx.Sequences[s].NumComputedTokens + ctx.NumScheduledTokens[s];
                lastRows[s] = queryStart[s] + ctx.NumScheduledTokens[s] - 1;
            }

            // Decode steps forward the sampled-but-not-yet-committed token, which is absent
            // from the sequence's token list, so the executor passes the flat tokens.
            int[] flatTokens;
            if (ctx.OverrideFlatTokens != null)
            {
                if (ctx.OverrideFlatTokens.Length != numTokens)
                    throw new ArgumentException($"OverrideFlatTokens length {ctx.OverrideFlatTokens.Length} != numTokens {numTokens}.");
                flatTokens = ctx.OverrideFlatTokens;
            }
            else
            {
                flatTokens = new int[numTokens];
                int cursor = 0;
                for (int s = 0; s < numSeqs; s++)
                {
                    var seq = ctx.Sequences[s];
                    for (int i = 0; i < ctx.NumScheduledTokens[s]; i++)
                        flatTokens[cursor++] = seq.TokenAt(seq.NumComputedTokens + i);
                }
            }

            (int[] flatTables, int[] offsets) = IsGgmlBackend ? FlattenBlockTables(ctx.BlockTables) : (null, null);
            var batch = new PagedBatch
            {
                NumSequences = numSeqs,
                QueryStart = queryStart,
                SequenceLengths = seqLens,
                Positions = positions,
                SlotMapping = ctx.SlotMapping.ToArray(),
                BlockTables = ctx.BlockTables,
                FlatBlockTables = flatTables,
                BlockTableOffsets = offsets,
            };

            _forwardSw.Start();
            try
            {
                BatchedForwardCount++;
                EnterStage(0);
                Tensor hidden = Embedding(flatTokens);
                // Multimodal requests are prepared serially upstream, so a batch carrying
                // media holds one sequence; its computed count is the chunk's absolute base.
                InjectMedia(hidden, ctx.Sequences[0].NumComputedTokens);
                hidden = RunLayers(hidden, TokenPositions.Explicit(positions), batch);
                float[] all = ProjectRows(hidden, lastRows);
                hidden.Dispose();

                int vocab = Config.VocabSize;
                var perSeq = new float[numSeqs][];
                for (int s = 0; s < numSeqs; s++)
                {
                    perSeq[s] = new float[vocab];
                    Array.Copy(all, (long)s * vocab, perSeq[s], 0, vocab);
                }
                return perSeq;
            }
            finally
            {
                EnterStage(0);
                _forwardSw.Stop();
            }
        }

        private Tensor PagedAttention(RankScope scope, int layer, Tensor q, Tensor k, Tensor v, PagedBatch batch)
        {
            int heads = scope.LocalHeads, kvHeads = scope.LocalKvHeads;
            int tokens = batch.Positions.Length;
            int qDim = heads * _keyDim, kDim = kvHeads * _keyDim;
            float scale = 1.0f / MathF.Sqrt(_keyDim);
            float[] pagedK = _pagedK[layer][scope.Rank], pagedV = _pagedV[layer][scope.Rank];

            float[] kFlat = k.GetElementsAsFloat(tokens * kDim);
            float[] vFlat = v.GetElementsAsFloat(tokens * kDim);
            k.Dispose();
            v.Dispose();
            PagedKvBatchOps.ScatterKv(kFlat, vFlat, pagedK, pagedV, batch.SlotMapping, tokens, kvHeads, _keyDim, _pagedBlockSize);

            float[] qFlat = q.GetElementsAsFloat(tokens * qDim);
            q.Dispose();
            var attnFlat = new float[tokens * qDim];
            if (batch.FlatBlockTables != null)
            {
                // One native call at a time: the ranks of a stage may run concurrently, and the
                // kernel's host-side staging is not specified as re-entrant.
                lock (_nativePagedAttentionLock)
                {
                    GgmlBasicOps.PagedAttentionForward(qFlat, pagedK, pagedV, attnFlat,
                        batch.QueryStart, batch.SequenceLengths, batch.Positions,
                        batch.FlatBlockTables, batch.BlockTableOffsets,
                        batch.NumSequences, tokens, heads, kvHeads, _keyDim, _pagedBlockSize, scale);
                }
            }
            else
            {
                ManagedPagedAttention.Forward(qFlat, pagedK, pagedV, attnFlat,
                    tokens, heads, kvHeads, _keyDim, _pagedBlockSize,
                    batch.QueryStart, batch.SequenceLengths, batch.Positions, batch.BlockTables, batch.NumSequences,
                    scale, causal: true);
            }

            var result = new Tensor(scope.Allocator, DType.Float32, tokens, qDim);
            result.SetElementsAsFloat(attnFlat);
            HostWritten(result);
            return result;
        }

        private static (int[] flat, int[] offsets) FlattenBlockTables(int[][] tables)
        {
            int total = 0;
            var offsets = new int[tables.Length];
            for (int s = 0; s < tables.Length; s++)
            {
                offsets[s] = total;
                total += tables[s].Length;
            }
            var flat = new int[total];
            for (int s = 0; s < tables.Length; s++)
                Array.Copy(tables[s], 0, flat, offsets[s], tables[s].Length);
            return (flat, offsets);
        }

        /// <summary>Grow the paged buffers, preserving the K/V already written: a later step can
        /// add sequences whose decode still reads an earlier sequence's prefill rows.</summary>
        private void EnsurePagedBuffers(int numBlocks, int blockSize)
        {
            if (_pagedK != null && _pagedBlocks >= numBlocks && _pagedBlockSize == blockSize)
                return;
            int target = Math.Max(numBlocks, _pagedBlocks * 2);
            float[][][] oldK = _pagedK, oldV = _pagedV;
            int oldBlocks = _pagedBlocks, oldBlockSize = _pagedBlockSize;
            int ranks = _mesh.TensorParallel;

            _pagedK = new float[Config.NumLayers][][];
            _pagedV = new float[Config.NumLayers][][];
            for (int l = 0; l < Config.NumLayers; l++)
            {
                if (!OwnsLayer(l)) continue;
                _pagedK[l] = new float[ranks][];
                _pagedV[l] = new float[ranks][];
                for (int r = 0; r < ranks; r++)
                {
                    _pagedK[l][r] = PagedKvBatchOps.AllocateLayerBuffer(target, blockSize, _localKvHeads, _keyDim);
                    _pagedV[l][r] = PagedKvBatchOps.AllocateLayerBuffer(target, blockSize, _localKvHeads, _valueDim);
                    if (oldK?[l]?[r] != null && oldBlockSize == blockSize)
                    {
                        long copy = Math.Min((long)oldBlocks * blockSize * _localKvHeads * _keyDim, oldK[l][r].LongLength);
                        Array.Copy(oldK[l][r], _pagedK[l][r], copy);
                        Array.Copy(oldV[l][r], _pagedV[l][r], copy);
                    }
                }
            }
            _pagedBlocks = target;
            _pagedBlockSize = blockSize;
        }

        private void DisposePagedBuffers()
        {
            _pagedK = _pagedV = null;
            _pagedBlocks = 0;
        }
    }
}
