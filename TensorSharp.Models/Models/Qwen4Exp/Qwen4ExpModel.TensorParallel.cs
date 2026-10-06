// Copyright (c) Zhongkai Fu. All rights reserved.
// Licensed under the BSD-3-Clause license in the repository root.
using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Threading.Tasks;
using TensorSharp.GGML;

namespace TensorSharp.Models
{
    public partial class Qwen4ExpModel
    {
        // Each rank keeps all expert IDs. Gate/up shard intermediate channels;
        // down shards output rows after gathering the full activated input.
        // Attention, QSA, GDN and PLE are replicated, with independent device state.
        private readonly Dictionary<string, StackedExpertWeights[]> _q4eTpExperts = new();
        private Qwen4ExpFfnArgs[][] _q4eTpFfnArgs;
        private IntPtr[] _q4eTpPlans;
        private long _q4eTpExpertBytes;

        protected override bool ShouldPrefaultWeight(GgufTensorInfo info)
            => !string.Equals(info.Name, "per_layer_token_embd.weight", StringComparison.Ordinal);

        protected override long AdditionalTpShardedBytes => _q4eTpExpertBytes;
        protected override bool MoEUnderTpIsSlow => false;

        private void ValidateQwen4ExpTensorParallelMetadata()
        {
            if (!IsTensorParallel) return;
            if (!IsGgmlBackend || GlobalTpDegree != TpDegree || LayerSplitDegree > 1
                || !GgmlBasicOps.TensorParallelFusedAvailable(TpDegree))
                throw new NotSupportedException("Qwen4Exp tensor parallelism needs a local GGML device group with fused collectives.");
            if (_expertFf <= 0 || _sharedFf <= 0 || _expertFf % TpDegree != 0 || _sharedFf % TpDegree != 0
                || Config.HiddenSize % TpDegree != 0)
                throw new NotSupportedException($"Qwen4Exp FFN/output widths {_expertFf}/{_sharedFf}/{Config.HiddenSize} must be divisible by TP degree {TpDegree}.");
            for (int l = 0; l < Config.NumLayers; ++l)
                foreach (string suffix in new[] { "gate_exps", "up_exps", "down_exps", "gate_shexp", "up_shexp", "down_shexp" })
                {
                    string name = $"blk.{l}.ffn_{suffix}.weight";
                    if (!_gguf.Tensors.TryGetValue(name, out var info) || info.Shape.Length < 2)
                        throw new NotSupportedException($"Qwen4Exp TP needs separate gate/up tensors: missing {name}.");
                    long block = GgufFile.GetBlockSize(info.Type);
                    if (block <= 0 || info.Shape[0] % (ulong)block != 0 || info.Shape[1] % (ulong)TpDegree != 0)
                        throw new NotSupportedException($"Qwen4Exp TP: {name} cannot be split into {TpDegree} whole-block shards (block {block}).");
                    try
                    {
                        GgmlBasicOps.RequireQwen4ExpTensorParallelWeight((int)info.Type,
                            checked((int)info.Shape[0]), checked((int)info.Shape[1]),
                            info.Shape.Length > 2 ? checked((int)info.Shape[2]) : 1,
                            TpDegree, !suffix.StartsWith("down", StringComparison.Ordinal));
                    }
                    catch (NotSupportedException e)
                    { throw new NotSupportedException($"Qwen4Exp TP cannot preserve {name} ({info.Type}): {e.Message}", e); }
                }
        }

        private void PrepareQwen4ExpTensorParallel()
        {
            if (!IsTensorParallel) return;
            if (!IsGgmlBackend || GlobalTpDegree != TpDegree || LayerSplitDegree > 1
                || !GgmlBasicOps.TensorParallelFusedAvailable(TpDegree))
                throw new NotSupportedException("Qwen4Exp tensor parallelism needs a local GGML device group with fused collectives.");
            if (_fusedGateUpExperts || _expertFf % TpDegree != 0 || _sharedFf % TpDegree != 0
                || Config.HiddenSize % TpDegree != 0)
                throw new NotSupportedException("Qwen4Exp TP requires separate gate/up expert tensors and FFN widths divisible by the TP degree.");

            // All projections slice complete output rows, preserving each dot's
            // original input width and its complete quantization blocks.
            for (int l = 0; l < Config.NumLayers; ++l)
            {
                foreach (string suffix in new[] { "gate", "up", "down" })
                {
                    string name = $"blk.{l}.ffn_{suffix}_exps.weight";
                    if (!_stackedExpertWeights.TryGetValue(name, out var weight))
                        throw new NotSupportedException($"Qwen4Exp TP needs stacked expert tensor {name}.");
                    ValidateTensorParallelExpertShape(weight, TpDegree, false);
                }
            }

            var timer = Stopwatch.StartNew();
            var work = new List<(string Name, int Rank)>();
            for (int l = 0; l < Config.NumLayers; ++l)
                foreach (string suffix in new[] { "gate", "up", "down" })
                {
                    string name = $"blk.{l}.ffn_{suffix}_exps.weight";
                    _q4eTpExperts.Add(name, new StackedExpertWeights[TpDegree]);
                    for (int r = 0; r < TpDegree; ++r) work.Add((name, r));
                }
            try
            {
                Parallel.ForEach(work, item =>
                    _q4eTpExperts[item.Name][item.Rank] = SliceTensorParallelExpert(
                        _stackedExpertWeights[item.Name], item.Rank, TpDegree, false,
                        item.Name.Contains(".ffn_down_") ? 1 : Qwen4ExpMmqRowAlignment(
                            _stackedExpertWeights[item.Name].GgmlType, _expertFf)));
                foreach (var shards in _q4eTpExperts.Values)
                    foreach (var shard in shards) _q4eTpExpertBytes += shard.TotalRawBytes;
                ShardQwen4ExpSharedMmqWeights();
                ShardWeightsForTensorParallelism(
                    new[] { ".ffn_gate_shexp.weight", ".ffn_up_shexp.weight", ".ffn_down_shexp.weight" },
                    Array.Empty<string>());
            }
            catch { DisposeQwen4ExpTensorParallel(); throw; }
            Console.WriteLine($"  Qwen4Exp TP: expert channels {_expertFf}->{_expertFf / TpDegree}, shared channels {_sharedFf}->{_sharedFf / TpDegree}; "
                + $"down output rows {Config.HiddenSize}->{Config.HiddenSize / TpDegree}; "
                + $"{_q4eTpExpertBytes / 1024 / 1024} MiB tensor-sliced in {timer.ElapsedMilliseconds} ms across {TpDegree} ranks.");
        }

        // MMQ's stream-K reduction depends on the full output-row grid. Keep
        // complete 128-row tiles around a rank's logical slice, then crop their
        // outputs in the native graph. Neighboring ranks may share edge rows.
        internal static int Qwen4ExpMmqRowAlignment(int type, long rows)
            => rows % 128 == 0 && (GgmlTensorType)type is
                GgmlTensorType.Q8_0 or GgmlTensorType.Q2_K or GgmlTensorType.Q3_K or
                GgmlTensorType.Q4_K or GgmlTensorType.Q5_K or GgmlTensorType.Q6_K or
                GgmlTensorType.IQ2_XXS or GgmlTensorType.IQ2_XS or GgmlTensorType.IQ3_XXS or
                GgmlTensorType.IQ1_S or GgmlTensorType.IQ4_NL or GgmlTensorType.IQ3_S or
                GgmlTensorType.IQ2_S or GgmlTensorType.IQ4_XS ? 128 : 1;

        internal static (long First, long Count) Qwen4ExpOutputRowRange(
            long rows, int rank, int degree, int alignment = 1)
        {
            if (degree < 1 || rows <= 0 || rows % degree != 0 || alignment < 1 || rows % alignment != 0)
                throw new ArgumentException("Invalid Qwen4Exp output-row partition.");
            if ((uint)rank >= (uint)degree) throw new ArgumentOutOfRangeException(nameof(rank));
            long first = rows / degree * rank, end = first + rows / degree;
            long storedFirst = first / alignment * alignment;
            long storedEnd = (end + alignment - 1) / alignment * alignment;
            return (storedFirst, storedEnd - storedFirst);
        }

        private void ShardQwen4ExpSharedMmqWeights()
        {
            for (int layer = 0; layer < Config.NumLayers; ++layer)
                foreach (string projection in new[] { "gate", "up" })
                {
                    string name = $"blk.{layer}.ffn_{projection}_shexp.weight";
                    if (!_quantWeights.TryGetValue(name, out var source)) continue;
                    int alignment = Qwen4ExpMmqRowAlignment(source.GgmlType, source.Ne1);
                    if (alignment == 1) continue;
                    long rowBytes = NativeDequant.RowSize(source.GgmlType, source.Ne0);
                    var shards = new QuantizedWeight[TpDegree];
                    for (int rank = 0; rank < TpDegree; ++rank)
                    {
                        var range = Qwen4ExpOutputRowRange(source.Ne1, rank, TpDegree, alignment);
                        shards[rank] = QuantizedWeight.CreateExternalView(
                            new IntPtr(checked(source.Data.ToInt64() + range.First * rowBytes)),
                            range.Count * rowBytes, source.GgmlType, source.Ne0, range.Count, source);
                        shards[rank].Scale = source.Scale;
                    }
                    _tpQuantWeights.Add(name, shards);
                    RecordTpWeightScale(name, source);
                    _quantWeights.Remove(name); // Each external view retains its source owner.
                }
        }

        internal static void ValidateTensorParallelExpertShape(StackedExpertWeights source, int degree, bool rowParallel)
        {
            if (degree < 1 || source.NumExperts < 1 || source.PerExpertNe0 <= 0 || source.PerExpertNe1 <= 0)
                throw new ArgumentException("Invalid Qwen4Exp expert shard geometry.");
            long block = GgufFile.GetBlockSize((GgmlTensorType)source.GgmlType);
            long dimension = rowParallel ? source.PerExpertNe0 : source.PerExpertNe1;
            long alignment = rowParallel ? block * degree : degree;
            if (block <= 0 || source.PerExpertNe0 % block != 0 || dimension % alignment != 0)
                throw new NotSupportedException($"Qwen4Exp expert dimension {dimension} cannot be divided across {degree} ranks with block size {block}.");
        }

        internal static unsafe StackedExpertWeights SliceTensorParallelExpert(
            StackedExpertWeights source, int rank, int degree, bool rowParallel, int outputRowAlignment = 1)
        {
            ValidateTensorParallelExpertShape(source, degree, rowParallel);
            if ((uint)rank >= (uint)degree) throw new ArgumentOutOfRangeException(nameof(rank));
            long sourceRow = NativeDequant.RowSize(source.GgmlType, source.PerExpertNe0);
            long localNe0 = rowParallel ? source.PerExpertNe0 / degree : source.PerExpertNe0;
            var rows = rowParallel ? (First: 0L, Count: source.PerExpertNe1)
                : Qwen4ExpOutputRowRange(source.PerExpertNe1, rank, degree, outputRowAlignment);
            long localNe1 = rows.Count;
            long localRow = NativeDequant.RowSize(source.GgmlType, localNe0);
            long localExpert = checked(localRow * localNe1);
            long bytes = checked(localExpert * source.NumExperts);
            IntPtr buffer = QuantizedWeight.AllocateBuffer(bytes);
            try
            {
                byte* src = (byte*)source.Data;
                byte* dst = (byte*)buffer;
                for (int e = 0; e < source.NumExperts; ++e)
                {
                    byte* srcExpert = src + e * source.PerExpertRawBytes;
                    byte* dstExpert = dst + e * localExpert;
                    if (rowParallel)
                        for (long row = 0; row < localNe1; ++row)
                            Buffer.MemoryCopy(srcExpert + row * sourceRow + rank * localRow,
                                dstExpert + row * localRow, localRow, localRow);
                    else
                        Buffer.MemoryCopy(srcExpert + rows.First * sourceRow, dstExpert, localExpert, localExpert);
                }
                return new StackedExpertWeights(buffer, source.GgmlType, localNe0, localNe1,
                    source.NumExperts, bytes, false, null, buffer);
            }
            catch { QuantizedWeight.FreeBuffer(buffer); throw; }
        }

        private bool TryGetQwen4ExpExpert(string name, int rank, out StackedExpertWeights weight)
        {
            if (_q4eTpExperts.TryGetValue(name, out var shards)) { weight = shards[rank]; return true; }
            return _stackedExpertWeights.TryGetValue(name, out weight);
        }

        private unsafe bool TryResolveQwen4ExpTpQuant(string name, int rank, out IntPtr ptr, out int type, out long bytes)
        {
            if (_tpQuantWeights.TryGetValue(name, out var qw))
            { ptr = qw[rank].CacheKey; type = qw[rank].GgmlType; bytes = qw[rank].RawBytes; return true; }
            if (_tpWeights.TryGetValue(name, out var w))
            { ptr = (IntPtr)GetFloatPtr(w[rank]); type = 0; bytes = w[rank].ElementCount() * sizeof(float); return true; }
            return TryResolveQuant(name, out ptr, out type, out bytes);
        }

        protected override unsafe void PreloadGgmlTpAuxiliaryWeightsForRank(int rank, long[] bytesPerRank, int[] countPerRank)
        {
            foreach (var pair in _q4eTpExperts)
            {
                var w = pair.Value[rank];
                if (!GgmlBasicOps.PreloadQuantizedWeight(w.Data, w.Data, w.GgmlType,
                        w.PerExpertNe0, w.PerExpertNe1 * w.NumExperts, w.TotalRawBytes))
                    throw new NotSupportedException($"Qwen4Exp TP expert shard exceeds the device allocation limit: {pair.Key} (rank {rank}).");
                bytesPerRank[rank] += w.TotalRawBytes;
                ++countPerRank[rank];
            }
            // Non-FFN trunk weights are replicated. Eager upload keeps first-token
            // latency bounded and prevents graph allocation from competing with it.
            foreach (var pair in _quantWeights)
            {
                if (rank == 0 || !pair.Key.StartsWith("blk.", StringComparison.Ordinal)
                    || !ShouldPreloadCudaQuantWeightToDevice(pair.Key)) continue;
                var w = pair.Value;
                if (GgmlBasicOps.PreloadQuantizedWeight(w.EnsureDeviceCacheKey(), w.Data,
                    w.GgmlType, w.Ne0, w.Ne1, w.RawBytes))
                { bytesPerRank[rank] += w.RawBytes; ++countPerRank[rank]; }
            }
        }

        private unsafe bool ExecuteQwen4ExpSpan(
            IntPtr ffn, IntPtr gdn, IntPtr attn, IntPtr kinds,
            int layerBegin, int layerEnd,
            IntPtr resData, IntPtr maskData,
            int nEmbd, int hc, int hcLowRank, int nTokens,
            int headKDim, int headVDim, int nKHeads, int nVHeads, int dConv,
            int headDim, int nHead, int nHeadKv, int kvCapacity, int nKv, int position,
            int nRot, float ropeBase, float ropeFreqScale, float attnScale,
            int nExpert, int nExpertUsed, int nFf, int nFfSh,
            float eps, int cacheSlot,
            IntPtr head = default, IntPtr logitsOut = default,
            IntPtr ple = default, int pleLayer = -1, IntPtr pleEmb = default,
            IntPtr mropePos = default, IntPtr mropeSections = default,
            int ropePosition = -1,
            // GPU this span's layers live on (layer split). -1 / 0 = the current
            // rank, which is the only rank on a single-GPU run.
            int device = 0, IntPtr hiddenOut = default, int logitsRows = 1,
            IntPtr qsa = default, IntPtr qsaPositions = default, int qsaPositionCount = 0)
        {
            if (IsTensorParallel)
            {
                if (_q4eTpFfnArgs == null)
                {
                    _q4eTpFfnArgs = new Qwen4ExpFfnArgs[TpDegree][];
                    _q4eTpPlans = new IntPtr[TpDegree];
                    for (int r = 0; r < TpDegree; ++r)
                    {
                        var args = GC.AllocateArray<Qwen4ExpFfnArgs>(Config.NumLayers, pinned: true);
                        for (int l = 0; l < Config.NumLayers; ++l)
                            if (!TryFillFfnArgs(l, ref args[l], r))
                                throw new NotSupportedException("Qwen4Exp TP descriptors could not be resolved.");
                        _q4eTpFfnArgs[r] = args;
                    }
                }
                for (int r = 0; r < TpDegree; ++r)
                    fixed (Qwen4ExpFfnArgs* rankFfn = _q4eTpFfnArgs[r])
                        _q4eTpPlans[r] = GgmlBasicOps.Qwen4ExpTokenSpanTp((IntPtr)rankFfn, gdn, attn, kinds, layerBegin, layerEnd,
                            resData, maskData, nEmbd, hc, hcLowRank, nTokens,
                            headKDim, headVDim, nKHeads, nVHeads, dConv,
                            headDim, nHead, nHeadKv, kvCapacity, nKv, position,
                            nRot, ropeBase, ropeFreqScale, attnScale,
                            nExpert, nExpertUsed, nFf / TpDegree, nFfSh / TpDegree, eps, cacheSlot,
                            r == 0 ? head : IntPtr.Zero, r == 0 ? logitsOut : IntPtr.Zero,
                            ple, pleLayer, pleEmb, mropePos, mropeSections, ropePosition,
                            r, r == 0 ? hiddenOut : IntPtr.Zero, logitsRows, qsa, qsaPositions, qsaPositionCount);
                GgmlBasicOps.TensorParallelExecutePlans(_q4eTpPlans);
                if (!string.IsNullOrEmpty(Environment.GetEnvironmentVariable("TS_Q4E_NODE_DUMP")))
                    GgmlBasicOps.Qwen4ExpTestDumpTpPlans(_q4eTpPlans, layerBegin, layerEnd,
                        nTokens, position, logitsRows, nKv);
                return true;
            }
            return GgmlBasicOps.Qwen4ExpTokenSpan(ffn, gdn, attn, kinds, layerBegin, layerEnd,
                resData, maskData, nEmbd, hc, hcLowRank, nTokens,
                headKDim, headVDim, nKHeads, nVHeads, dConv,
                headDim, nHead, nHeadKv, kvCapacity, nKv, position,
                nRot, ropeBase, ropeFreqScale, attnScale,
                nExpert, nExpertUsed, nFf, nFfSh, eps, cacheSlot,
                head, logitsOut, ple, pleLayer, pleEmb, mropePos, mropeSections, ropePosition,
                device, hiddenOut, logitsRows, qsa, qsaPositions, qsaPositionCount);
        }

        private void DisposeQwen4ExpTensorParallel()
        {
            _q4eTpFfnArgs = null;
            _q4eTpPlans = null;
            if (_q4eTpExperts == null) return;
            foreach (var shards in _q4eTpExperts.Values)
                foreach (var w in shards)
                    if (w != null && w.OwnedBuffer != IntPtr.Zero) QuantizedWeight.FreeBuffer(w.OwnedBuffer);
            _q4eTpExperts.Clear();
            _q4eTpExpertBytes = 0;
        }
    }
}
