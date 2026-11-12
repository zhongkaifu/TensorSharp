using System;
using System.Linq;
using System.Security.Cryptography;
using System.Text;
using TensorSharp.GGML;

namespace TensorSharp.Models;

public abstract partial class ModelBase
{
    protected LayerSplitPlan LayerPlacement { get; private set; }
    private IAllocator[] _layerAllocators;

    /// <summary>Initialize a family-independent dense layer pipeline after reading Config.</summary>
    /// <summary>Whether a tensor's bytes weigh in the whole-layer balance: a weight no device holds (Flash Next's 28.8 GB
    /// PLE table, read from the mapping) does not - counted as the head's, it left the last of two A40s 10 of 48 layers.</summary>
    protected virtual bool CountsInLayerBalance(string tensorName) => true;

    protected void ConfigureLayerPlacement(string executionIdentity, bool balanceWeightBytes = false)
    {
        if (LayerSplitGroup == null && LayerSplitDegree <= 1) return;
        int node = LayerSplitGroup?.NodeId ?? 0, nodes = LayerSplitGroup?.NodeCount ?? 1;
        bool driverLast = nodes > 1 && LayerSplitPlan.DriverRunsLastStageFromEnvironment();
        if (balanceWeightBytes)
        {
            var bytes = new long[Config.NumLayers];
            long embedding = 0, head = 0;
            foreach (var info in _gguf.Tensors.Values)
            {
                if (!CountsInLayerBalance(info.Name)) continue;
                long resident = IsQuantizedLinearWeight(info) ? _gguf.GetTensorByteCount(info) : checked(info.NumElements * 4);
                if (LayerSplitPlan.TryGetLayer(info.Name, out int layer) && layer < bytes.Length)
                    bytes[layer] = checked(bytes[layer] + resident);
                else if (info.Name == "token_embd.weight") embedding = checked(embedding + resident);
                else head = checked(head + resident);
            }
            if (!_gguf.Tensors.ContainsKey("output.weight")) head = checked(head + embedding);
            LayerPlacement = LayerSplitPlan.CreateWeighted(bytes, LayerSplitDegree, node, nodes, embedding, head, driverLast);
        }
        else LayerPlacement = new LayerSplitPlan(Config.NumLayers, LayerSplitDegree, node, nodes, driverLast);
        _layerAllocators = new IAllocator[LayerSplitDegree];
        for (int rank = 0; rank < _layerAllocators.Length; rank++)
            _layerAllocators[rank] = LayerSplitGroup?.GetAllocator(rank) ?? CreateLayerDeviceAllocator(rank);

        if (LayerSplitGroup != null)
        {
            if (LayerSplitGroup is not INestedTensorParallelGroup nested)
                throw new NotSupportedException("Layer split requires startup configuration agreement.");
            // Metadata and execution policy must agree before loading weights or entering a pipeline.
            // Deployment tooling also verifies checkpoint hashes outside timed inference.
            var identity = new StringBuilder(executionIdentity);
            identity.Append('|').Append(Config.NumLayers).Append('|').Append(Config.HiddenSize)
                .Append('|').Append(Config.VocabSize).Append('|').Append(ResolveConfiguredContextLength())
                .Append('|').Append(balanceWeightBytes).Append('|').Append(driverLast);
            foreach (var entry in _gguf.Tensors.OrderBy(entry => entry.Key, StringComparer.Ordinal))
                identity.Append('|').Append(entry.Key).Append(':').Append(entry.Value.Type)
                    .Append(':').Append(string.Join(',', entry.Value.Shape));
            byte[] digest = SHA256.HashData(Encoding.UTF8.GetBytes(identity.ToString()));
            var votes = new float[digest.Length * 8];
            for (int bit = 0; bit < votes.Length; bit++) votes[bit] = (digest[bit / 8] >> (bit % 8)) & 1;
            nested.CrossNodeAllReduce(votes, votes.Length);
            for (int bit = 0; bit < votes.Length; bit++)
                if (votes[bit] != (((digest[bit / 8] >> (bit % 8)) & 1) * LayerSplitGroup.NodeCount))
                    throw new InvalidOperationException("Layer-split nodes have different model metadata or execution settings.");
        }
    }

    /// <summary>The allocator for local device <paramref name="rank"/> of a whole-layer split:
    /// the GGML context's rank, a direct-CUDA device (TENSORSHARP_LAYER_SPLIT_DEVICES picks
    /// which), or, on the managed cpu backend, a separate host "device" (functional placement
    /// for tests; never a performance mode).</summary>
    private IAllocator CreateLayerDeviceAllocator(int rank) => _backend switch
    {
        BackendType.Cpu => rank == 0 ? _allocator : new TensorSharp.Cpu.CpuAllocator(BlasEnum.DotNet),
        BackendType.Cuda => rank == 0 ? _allocator
            : new TensorSharp.Cuda.CudaAllocator(ParseDeviceIds(LayerSplitDegree, "TENSORSHARP_LAYER_SPLIT_DEVICES")[rank]),
        BackendType.Mlx => throw new NotSupportedException("The mlx backend exposes one device; whole-layer placement needs several."),
        _ => new GgmlAllocator(_ggmlContext, rank),
    };

    /// <summary>The allocator of local layer-split device <paramref name="rank"/> (rank 0 when unsplit).</summary>
    protected IAllocator LayerDeviceAllocator(int rank) => _layerAllocators == null ? _allocator : _layerAllocators[rank];

    protected IAllocator LayerAllocator(int layer) => LayerPlacement == null
        ? _allocator : _layerAllocators[LayerPlacement.LocalRank(layer)];

    protected void SelectLayerDevice(int rank)
    {
        if (_layerAllocators == null) return;
        _allocator = _layerAllocators[rank];
        if (IsGgmlBackend) GgmlBasicOps.SetActiveRank(rank);
    }

    /// <summary>Only the first stage embeds; each stage executes its own whole layers;
    /// the last stage projects. Its logits stay local when node zero runs the last
    /// stage (the default order) and are returned to node zero otherwise. Persistent layer state
    /// remains with its owner. Families with extra cross-layer state use the same
    /// ILayerSplitGroup transport with their complete boundary-state representation.</summary>
    protected float[] ForwardLayerPipeline(int[] tokens, Func<Tensor> embed,
        Func<Tensor, int, Tensor> runLayer, Func<Tensor, float[]> project)
    {
        if (LayerPlacement == null) throw new InvalidOperationException("Layer placement has not been configured.");
        ArgumentNullException.ThrowIfNull(tokens);
        if (tokens.Length == 0) throw new ArgumentException("A forward requires at least one token.", nameof(tokens));
        int elements = checked(tokens.Length * Config.HiddenSize);
        Tensor hidden = null;
        SelectLayerDevice(0);
        try
        {
            if (LayerPlacement.IsFirst)
                hidden = embed();
            else
            {
                var input = new float[elements];
                LayerSplitGroup.Receive(input, elements, LayerPlacement.PreviousNode);
                hidden = new Tensor(_allocator, DType.Float32, tokens.Length, Config.HiddenSize);
                hidden.SetElementsAsFloat(input);
            }
            int previousRank = 0;
            for (int layer = LayerPlacement.FirstLayer; layer < LayerPlacement.EndLayer; layer++)
            {
                int rank = LayerPlacement.LocalRank(layer);
                if (rank != previousRank)
                {
                    var values = TensorToFloatArray(hidden);
                    hidden.Dispose();
                    hidden = null;
                    SelectLayerDevice(rank);
                    hidden = new Tensor(_allocator, DType.Float32, tokens.Length, Config.HiddenSize);
                    hidden.SetElementsAsFloat(values);
                    previousRank = rank;
                }
                hidden = runLayer(hidden, layer);
            }
            if (LayerPlacement.IsLast)
            {
                float[] result = project(hidden);
                if (result.Length != Config.VocabSize)
                    throw new InvalidOperationException("The output stage returned an unexpected vocabulary size.");
                if (LayerPlacement.NodeId != 0) LayerSplitGroup.Send(result, result.Length, 0);
                return result;
            }

            var outgoing = TensorToFloatArray(hidden);
            if (outgoing.Length != elements) throw new InvalidOperationException("Invalid layer-stage activation shape.");
            LayerSplitGroup.Send(outgoing, elements, LayerPlacement.NextNode);
            if (LayerPlacement.NodeId != 0) return Array.Empty<float>();
            var logits = new float[Config.VocabSize];
            LayerSplitGroup.Receive(logits, logits.Length, LayerPlacement.NodeOfStage(LayerPlacement.NodeCount - 1));
            return logits;
        }
        catch (Exception failure)
        {
            // A failed stage cannot participate in the next request: downstream
            // peers may already be waiting for this activation or its logits.
            // Close only the network: local pools still own the model's weights
            // and caches, whose cleanup must precede local-group disposal.
            try
            {
                if (LayerSplitGroup is IDistributedExecutionGroup distributed) distributed.Abort(failure);
                else LayerSplitGroup?.Dispose();
            }
            catch { /* preserve the stage error */ }
            throw;
        }
        finally
        {
            hidden?.Dispose();
            SelectLayerDevice(0);
        }
    }
}
