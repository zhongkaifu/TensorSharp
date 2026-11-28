// Copyright (c) Zhongkai Fu. All rights reserved.
// Licensed under the BSD-3-Clause license in the repository root.
using System;
using System.Collections.Generic;
using TensorSharp.GGML;
using TensorSharp.Runtime;

namespace TensorSharp.Models;

public partial class MuseSparkModel
{
    // Attention uses real column/row tensor sharding. Routed experts use whole-
    // expert parallelism: their nonlinear latent normalization cannot be applied
    // independently to partial inner-dimension products.
    private int[] _tpKvFirst, _tpKvCount, _tpQueryFirst, _tpQueryCount, _tpExpertFirst, _tpExpertCount;
    private readonly List<QuantizedWeight> _sparkTpColumnOwners = new();
    private readonly Dictionary<string, Tensor[]> _sparkTpNorms = new();
    private Tensor[] _sparkTpHeadOnes;
    private StackedExpertWeights[][] _sparkTpGate, _sparkTpUp, _sparkTpDown;
    private float[][] _sparkTpPostNorm;
    private float[][][] _sparkTpPagedKeys, _sparkTpPagedValues;
    internal IReadOnlyList<long> TensorParallelPlannedWeightBytes { get; private set; }

    protected override bool SupportsTensorParallelWeightScales => true;
    protected override bool MoEUnderTpIsSlow => false;

    internal static (int First, int Count) SparkPartition(int total, int degree, int rank)
    {
        if (degree <= 0 || degree > total || (uint)rank >= (uint)degree)
            throw new ArgumentOutOfRangeException(nameof(degree), "Each rank must own at least one item.");
        int count = total / degree, remainder = total % degree;
        return (rank * count + Math.Min(rank, remainder), count + (rank < remainder ? 1 : 0));
    }

    private void PreflightSparkTensorParallel()
    {
        if (!IsTensorParallel) return;
        if (TpNodeCount != 1 || TpRankOffset != 0 || GlobalTpDegree != TpDegree)
            throw new NotSupportedException("Muse Spark tensor parallelism currently requires one local process.");
        if (TpDegree > Config.NumKVHeads || TpDegree > _numExperts)
            throw new NotSupportedException("Muse Spark requires at least one KV head group and expert per TP rank.");
        long[] fixedBytes = new long[TpDegree];
        long bytesPerExpert = 0;
        foreach (var info in _gguf.Tensors.Values)
        {
            long bytes = PlacementWeightBytes(info, _gguf.GetTensorByteCount(info), _backend);
            bool isLayer = MoeCpuOffloadConfig.TryParseLayerIndex(info.Name, out int layer);
            bool experts = isLayer && info.Name.Contains("_exps.", StringComparison.Ordinal);
            bool offloaded = experts && MoeOffload.IsLayerOnCpu(layer);
            if (offloaded && IsQuantizedLinearWeight(info)) continue;
            bool query = info.Name.EndsWith("attn_q.weight", StringComparison.Ordinal) ||
                info.Name.EndsWith("attn_gate.weight", StringComparison.Ordinal) ||
                info.Name.EndsWith("attn_output.weight", StringComparison.Ordinal);
            bool kv = info.Name.EndsWith("attn_k.weight", StringComparison.Ordinal) ||
                info.Name.EndsWith("attn_v.weight", StringComparison.Ordinal);
            bool headNorm = info.Name.EndsWith("attn_q_norm.weight", StringComparison.Ordinal) ||
                info.Name.EndsWith("attn_k_norm.weight", StringComparison.Ordinal);
            if (experts && !offloaded)
            {
                if (bytes % _numExperts != 0)
                    throw new InvalidOperationException($"Muse Spark expert bytes are not evenly divisible in {info.Name}.");
                bytesPerExpert = checked(bytesPerExpert + bytes / _numExperts);
            }
            else if (query || kv)
            {
                int total = Config.NumKVHeads;
                for (int rank = 0; rank < TpDegree; rank++)
                {
                    var (first, count) = SparkPartition(total, TpDegree, rank);
                    fixedBytes[rank] = checked(fixedBytes[rank] + bytes / total * count);
                    if (info.Name.EndsWith("attn_output.weight", StringComparison.Ordinal))
                    {
                        long block = GgufFile.GetBlockSize(info.Type);
                        int groupWidth = Config.NumHeads / Config.NumKVHeads * _headDim;
                        if ((long)first * groupWidth % block != 0 || (long)count * groupWidth % block != 0)
                            throw new NotSupportedException($"Muse Spark TP rank {rank} splits a {block}-value quantization block in {info.Name}.");
                    }
                }
            }
            else if (headNorm)
                for (int rank = 0; rank < TpDegree; rank++) fixedBytes[rank] = checked(fixedBytes[rank] + bytes);
            else fixedBytes[0] = checked(fixedBytes[0] + bytes);
        }
        bool measuredBackend = _backend is BackendType.GgmlCuda or BackendType.GgmlVulkan;
        long reserve = measuredBackend || _sparkTpPlanningFreeBytes != null
            ? ResolveScratchReserveBytes(Config.HiddenSize, _latentSize, _expertFfnLength,
                _prefillChunk, Environment.GetEnvironmentVariable(ScratchReserveVariable)) : 0;
        long[] free = _sparkTpPlanningFreeBytes;
        if (free == null)
        {
            free = new long[TpDegree];
            if (measuredBackend) free = QueryCapacityFreeBytes(TpDegree);
            else Array.Fill(free, -1L);
        }
        _sparkTpPlacement = PlanSparkExperts(_numExperts, fixedBytes, bytesPerExpert, free, reserve);
        long[] weights = new long[TpDegree];
        for (int rank = 0; rank < TpDegree; rank++) weights[rank] = _sparkTpPlacement.WeightBytes[rank];
        TensorParallelPlannedWeightBytes = _sparkTpPlacement.WeightBytes;
        if (!measuredBackend && _sparkTpPlanningFreeBytes == null) return;
        string refusal = CapacityRefusal(weights, free, reserve);
        if (refusal != null)
            throw new ModelLoadRefusedException(refusal + " Tensor-parallel weights were counted by actual GQA-group and whole-expert shards; " +
                "router, latent projections and output head are on rank 0. Increase --n-cpu-moe, use more/larger GPUs, or choose a smaller quantization. " +
                "No offload or placement setting was changed.");
        Console.WriteLine($"Muse Spark TP capacity: estimated resident weight MiB [{string.Join(',', Array.ConvertAll(weights, x => x / (1L << 20)))}], " +
            $"{reserve / (1L << 20)} MiB compute/active-KV reserve per rank. Additional vision/request-cache memory is workload-dependent.");
        if (Array.Exists(free, x => x < 0))
            Console.WriteLine("Muse Spark TP capacity: memory unavailable on one or more ranks; those ranks were not checked.");
    }

    private void InitializeSparkTensorParallel()
    {
        if (!IsTensorParallel) return;
        if (TpNodeCount != 1 || TpRankOffset != 0 || GlobalTpDegree != TpDegree)
            throw new NotSupportedException("Muse Spark tensor parallelism currently requires one local process.");
        if (TpDegree > Config.NumKVHeads || TpDegree > _numExperts)
            throw new NotSupportedException("Muse Spark requires at least one KV head group and expert per TP rank.");

        _tpKvFirst = new int[TpDegree]; _tpKvCount = new int[TpDegree];
        _tpQueryFirst = new int[TpDegree]; _tpQueryCount = new int[TpDegree];
        _tpExpertFirst = new int[TpDegree]; _tpExpertCount = new int[TpDegree];
        if (_sparkTpPlacement == null || _sparkTpPlacement.Counts.Count != TpDegree)
            throw new InvalidOperationException("Muse Spark TP materialization requires its preflight expert plan.");
        _sparkTpHeadOnes = new Tensor[TpDegree];
        int group = Config.NumHeads / Config.NumKVHeads;
        for (int r = 0; r < TpDegree; r++)
        {
            (_tpKvFirst[r], _tpKvCount[r]) = SparkPartition(Config.NumKVHeads, TpDegree, r);
            _tpQueryFirst[r] = _tpKvFirst[r] * group; _tpQueryCount[r] = _tpKvCount[r] * group;
            _tpExpertCount[r] = _sparkTpPlacement.Counts[r];
            _tpExpertFirst[r] = r == 0 ? 0 : _tpExpertFirst[r - 1] + _tpExpertCount[r - 1];
        }
        // Fail before any weight mutation if an output-projection cut would
        // bisect a quantization block. Complete GQA groups satisfy this for the
        // published 128Q/16KV/64 Muse Spark checkpoint, including uneven TP=9.
        foreach (string p in _prefixes)
            if (_quantWeights.TryGetValue(p + "attn_output.weight", out var weight))
            {
                long block = GgufFile.GetBlockSize((GgmlTensorType)weight.GgmlType);
                for (int r = 0; r < TpDegree; r++)
                    if ((_tpQueryFirst[r] * _headDim) % block != 0 || (_tpQueryCount[r] * _headDim) % block != 0)
                        throw new NotSupportedException($"Muse Spark TP rank {r} splits a {block}-value quantization block in {p}attn_output.weight.");
            }
        foreach (string p in _prefixes)
        {
            ShardSparkProjection(p + "attn_q.weight", _tpQueryFirst, _tpQueryCount, false);
            ShardSparkProjection(p + "attn_k.weight", _tpKvFirst, _tpKvCount, false);
            ShardSparkProjection(p + "attn_v.weight", _tpKvFirst, _tpKvCount, false);
            ShardSparkProjection(p + "attn_gate.weight", _tpQueryFirst, _tpQueryCount, false);
            ShardSparkProjection(p + "attn_output.weight", _tpQueryFirst, _tpQueryCount, true);
            _sparkTpNorms.Add(p + "attn_q_norm.weight", new Tensor[TpDegree]);
            _sparkTpNorms.Add(p + "attn_k_norm.weight", new Tensor[TpDegree]);
        }
        _tpGroup.RunPerRank(r =>
        {
            var allocator = _tpGroup.GetAllocator(r);
            _sparkTpHeadOnes[r] = new Tensor(allocator, DType.Float32, _headDim);
            Ops.Fill(_sparkTpHeadOnes[r], 1f);
            foreach (var (name, tensors) in _sparkTpNorms)
            {
                tensors[r] = new Tensor(allocator, DType.Float32, _headDim);
                Ops.Copy(tensors[r], _weights[name]);
            }
        });
        _sparkTpGate = new StackedExpertWeights[Config.NumLayers][];
        _sparkTpUp = new StackedExpertWeights[Config.NumLayers][];
        _sparkTpDown = new StackedExpertWeights[Config.NumLayers][];
        _sparkTpPostNorm = new float[Config.NumLayers][];
        for (int l = 0; l < Config.NumLayers; l++)
        {
            string p = _prefixes[l];
            _sparkTpPostNorm[l] = TensorToFloatArray(_weights[p + "ffn_post_exp_norm.weight"]);
            if (MoeOffload.IsLayerOnCpu(l)) continue;
            if (!_stackedExpertWeights.TryGetValue(p + "ffn_gate_exps.weight", out var gate) ||
                !_stackedExpertWeights.TryGetValue(p + "ffn_up_exps.weight", out var up) ||
                !_stackedExpertWeights.TryGetValue(p + "ffn_down_exps.weight", out var down)) continue;
            _sparkTpGate[l] = new StackedExpertWeights[TpDegree];
            _sparkTpUp[l] = new StackedExpertWeights[TpDegree];
            _sparkTpDown[l] = new StackedExpertWeights[TpDegree];
            for (int r = 0; r < TpDegree; r++)
            {
                _sparkTpGate[l][r] = SparkExpertSlice(gate, _tpExpertFirst[r], _tpExpertCount[r]);
                _sparkTpUp[l][r] = SparkExpertSlice(up, _tpExpertFirst[r], _tpExpertCount[r]);
                _sparkTpDown[l][r] = SparkExpertSlice(down, _tpExpertFirst[r], _tpExpertCount[r]);
            }
        }
        Console.WriteLine($"Muse Spark TP: {TpDegree} ranks; KV heads [{string.Join(',', _tpKvCount)}], " +
            $"experts [{string.Join(',', _tpExpertCount)}] ({_sparkTpPlacement.Policy}). Attention is tensor-sharded; experts are partitioned whole; " +
            "router, latent projections and residual stream execute on rank 0.");
    }

    private unsafe void ShardSparkProjection(string name, int[] firstHeads, int[] counts, bool inputColumns)
    {
        if (_quantWeights.TryGetValue(name, out var original))
        {
            var shards = new QuantizedWeight[TpDegree];
            try
            {
                long rowBytes = NativeDequant.RowSize(original.GgmlType, original.Ne0);
                long block = GgufFile.GetBlockSize((GgmlTensorType)original.GgmlType);
                long typeBytes = GgufFile.GetTypeSize((GgmlTensorType)original.GgmlType);
                for (int r = 0; r < TpDegree; r++)
                {
                    long first = (long)firstHeads[r] * _headDim, width = (long)counts[r] * _headDim;
                    if (!inputColumns)
                        shards[r] = QuantizedWeight.CreateExternalView(original.Data + (nint)(first * rowBytes),
                            width * rowBytes, original.GgmlType, original.Ne0, width, original);
                    else
                    {
                        long destinationRow = width / block * typeBytes;
                        long bytes = destinationRow * original.Ne1;
                        IntPtr memory = QuantizedWeight.AllocateBuffer(bytes);
                        shards[r] = new QuantizedWeight(memory, bytes, original.GgmlType, width, original.Ne1);
                        for (long row = 0; row < original.Ne1; row++)
                            Buffer.MemoryCopy((byte*)original.Data + row * rowBytes + first / block * typeBytes,
                                (byte*)memory + row * destinationRow, destinationRow, destinationRow);
                    }
                    shards[r].Scale = original.Scale;
                }
                _tpQuantWeights.Add(name, shards);
                RecordTpWeightScale(name, original);
                _quantWeights.Remove(name);
                if (inputColumns) original.Dispose(); else _sparkTpColumnOwners.Add(original);
            }
            catch
            {
                foreach (var shard in shards) shard?.Dispose();
                throw;
            }
        }
        else
        {
            var originalFloat = _weights[name];
            var shards = new Tensor[TpDegree];
            try
            {
                _tpGroup.RunPerRank(r =>
                {
                    using var view = originalFloat.Narrow(inputColumns ? 1 : 0,
                        (long)firstHeads[r] * _headDim, (long)counts[r] * _headDim);
                    shards[r] = new Tensor(_tpGroup.GetAllocator(r), DType.Float32, view.Sizes);
                    Ops.Copy(shards[r], view);
                });
                _tpWeights.Add(name, shards);
                _weights.Remove(name);
                originalFloat.Dispose();
            }
            catch
            {
                foreach (var shard in shards) shard?.Dispose();
                throw;
            }
        }
    }

    private static StackedExpertWeights SparkExpertSlice(StackedExpertWeights source, int first, int count)
        => new(source.Data + (nint)(first * source.PerExpertRawBytes), source.GgmlType,
            source.PerExpertNe0, source.PerExpertNe1, count, source.PerExpertRawBytes * count,
            isExternalView: true, ownerToken: source, ownedBuffer: IntPtr.Zero);

    protected override void PreloadGgmlTpAuxiliaryWeightsForRank(int rank, long[] bytesPerRank, int[] countPerRank)
    {
        if (_sparkTpGate == null) return;
        for (int l = 0; l < Config.NumLayers; l++)
        {
            if (_sparkTpGate[l] == null) continue;
            foreach (var stack in new[] { _sparkTpGate[l][rank], _sparkTpUp[l][rank], _sparkTpDown[l][rank] })
            {
                if (!GgmlBasicOps.PreloadQuantizedWeight(stack.Data, stack.Data, stack.GgmlType,
                    stack.PerExpertNe0, stack.PerExpertNe1 * stack.NumExperts, stack.TotalRawBytes))
                    throw new InvalidOperationException($"Muse Spark expert shard for layer {l}, rank {rank} could not be made resident.");
                bytesPerRank[rank] += stack.TotalRawBytes; countPerRank[rank]++;
            }
        }
    }

    protected override long AdditionalTpShardedBytes
    {
        get
        {
            long bytes = 0;
            if (_sparkTpGate != null)
                for (int l = 0; l < _sparkTpGate.Length; l++)
                    if (_sparkTpGate[l] != null)
                        for (int r = 0; r < TpDegree; r++)
                            bytes += _sparkTpGate[l][r].TotalRawBytes + _sparkTpUp[l][r].TotalRawBytes + _sparkTpDown[l][r].TotalRawBytes;
            return bytes;
        }
    }

    private Tensor SparkRankInput(Tensor input, int rank)
    {
        var local = new Tensor(_tpGroup.GetAllocator(rank), DType.Float32, input.Sizes);
        try { Ops.Copy(local, input); return local; }
        catch { local.Dispose(); throw; }
    }

    private Tensor SparkRankLinear(Tensor input, string name, int rank)
    {
        long width = _tpQuantWeights.TryGetValue(name, out var quant) ? quant[rank].Ne1 : _tpWeights[name][rank].Sizes[0];
        var output = new Tensor(_tpGroup.GetAllocator(rank), DType.Float32, input.Sizes[0], width);
        try
        {
            if (quant != null)
            {
                AddmmQuantManaged(output, input, quant[rank]);
                if (quant[rank].Scale != 1f) Ops.Mul(output, output, quant[rank].Scale);
            }
            else
            {
                SparkFloatProjection(output, input, _tpWeights[name][rank]);
            }
            return output;
        }
        catch { output.Dispose(); throw; }
    }

    private Tensor SparkReduce(Tensor[] partials)
    {
        try
        {
            if (_tpGroup is GgmlTensorParallelGroup ggmlGroup) ggmlGroup.AllReduceFloat32(partials);
            else _tpGroup.AllReduce(partials);
            // InvalidateHostBuffer visits every device cache itself; the rank
            // workers have already joined at this point.
            foreach (var partial in partials) InvalidateTensorDeviceCache(partial);
            Tensor result = partials[0];
            partials[0] = null;
            return result;
        }
        finally
        {
            for (int rank = 0; rank < partials.Length; rank++) { partials[rank]?.Dispose(); partials[rank] = null; }
        }
    }

    private void DisposeSparkTensorParallel()
    {
        foreach (var tensors in _sparkTpNorms.Values) foreach (var tensor in tensors) tensor?.Dispose();
        _sparkTpNorms.Clear();
        if (_sparkTpHeadOnes != null) foreach (var tensor in _sparkTpHeadOnes) tensor?.Dispose();
        _sparkTpHeadOnes = null;
        // These are non-owning whole-expert views. Base.Dispose owns the source
        // stacks and releases their mappings after inference has stopped.
        _sparkTpGate = _sparkTpUp = _sparkTpDown = null;
        _sparkTpPagedKeys = _sparkTpPagedValues = null;
        _sparkTpPostNorm = null;
        // File-backed views and their retained owners must be released while
        // the GGUF mapping is still valid. Their release can advise mapped
        // pages; Base.Dispose unmaps the GGUF before its generic TP cleanup.
        foreach (var shards in _tpQuantWeights.Values) foreach (var shard in shards) shard?.Dispose();
        _tpQuantWeights.Clear();
        foreach (var owner in _sparkTpColumnOwners) owner.Dispose();
        _sparkTpColumnOwners.Clear();
    }

    protected override void OnBeforeReleaseGgmlDeviceResidency()
    {
        // F32 grouped experts bind individual matrix planes. Exact cache keys
        // for expert > 0 are interior pointers, so releasing only the enclosing
        // tensor's base address cannot evict them. No tensor data is read here.
        foreach (var (name, tensor) in _weights)
        {
            if (tensor == null || tensor.DimensionCount != 3 || tensor.ElementType != DType.Float32 ||
                !(name.EndsWith("ffn_gate_exps.weight", StringComparison.Ordinal) ||
                  name.EndsWith("ffn_up_exps.weight", StringComparison.Ordinal) ||
                  name.EndsWith("ffn_down_exps.weight", StringComparison.Ordinal))) continue;
            for (long expert = 0; expert < tensor.Sizes[0]; expert++)
                GgmlBasicOps.InvalidateHostBuffer(tensor.Storage.PtrAtElement(
                    checked(tensor.StorageOffset + expert * tensor.Strides[0])));
        }
        // Preloaded trunk matrices also use opaque cache keys; their retained
        // Data pointer is the reload source, not necessarily the resident key.
        // Ordinary and layer-split preload use the same opaque-key contract.
        foreach (var weight in _quantWeights.Values)
            if (weight != null && weight.CacheKey != IntPtr.Zero) GgmlBasicOps.InvalidateHostBuffer(weight.CacheKey);
        if (!IsTensorParallel) return;
        foreach (var shards in _tpQuantWeights.Values)
            foreach (var shard in shards)
                if (shard != null && shard.CacheKey != IntPtr.Zero) GgmlBasicOps.InvalidateHostBuffer(shard.CacheKey);
        foreach (var shards in _tpWeights.Values) foreach (var shard in shards) InvalidateTensorDeviceCache(shard);
        foreach (var copies in _sparkTpNorms.Values) foreach (var copy in copies) InvalidateTensorDeviceCache(copy);
        if (_sparkTpHeadOnes != null) foreach (var ones in _sparkTpHeadOnes) InvalidateTensorDeviceCache(ones);
        foreach (var layers in new[] { _sparkTpGate, _sparkTpUp, _sparkTpDown })
            if (layers != null)
                foreach (var ranks in layers)
                    if (ranks != null)
                        foreach (var stack in ranks)
                            if (stack != null) GgmlBasicOps.InvalidateHostBuffer(stack.Data);
    }
}
