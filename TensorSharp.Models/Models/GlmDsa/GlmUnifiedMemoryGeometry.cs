// Copyright (c) Zhongkai Fu. Licensed under the BSD-3-Clause license.
#nullable enable
using System;
using System.Globalization;
using System.Linq;
using TensorSharp.Memory.Planning;
using TensorSharp.Runtime;

namespace TensorSharp.Models;

/// <summary>Native GLM5Next payload forecasts. Whole-source mappings are pageable
/// file views, not anonymous RAM credit. Actual native buffers are quota charged;
/// managed/native metadata, page residency and vendor pools retain headroom.</summary>
internal sealed record GlmUnifiedMemoryGeometry
{
    // Explicit GLM adapter v7 uses one bounded F32 route-validation stack.
    // The independent Qwen adapter retains its existing 4096-element bound.
    internal const long SelectedRouteScratchHostBytes = 8192L * sizeof(float);
    // Conservatively include the retained callback IDs, unique-union plan,
    // remapped/route-major IDs and reserved (slot, expert) miss pairs even when
    // their actual lifetimes do not overlap. Native heap payload is a forecast,
    // not a fully instrumented allocation or a claim about vector capacity/GC.
    internal const long SelectedRouteMetadataHostBytes = 8192L * (4 * sizeof(int) + 2 * sizeof(int));
    internal string Architecture { get; init; } = "";
    internal int Layers { get; init; }
    internal int DeclaredContextTokens { get; init; }
    internal int RecurrentLayers { get; init; }
    internal int AttentionLayers => Layers - RecurrentLayers;
    internal int Hidden { get; init; }
    internal int Heads { get; init; }
    internal int Vocab { get; init; }
    internal int KvRow { get; init; }
    internal int IndexerDim { get; init; }
    internal int KdaDim { get; init; }
    internal int ConvKernel { get; init; }
    internal int HyperConnections { get; init; }
    internal int ExpertFf { get; init; }
    internal int UsedExperts { get; init; }
    internal int ExpertCount { get; init; }
    // Preserve source quantization, rather than forecasting a dequantized F32
    // weight copy. Dense-only layers have no routed triple.
    internal GlmUnifiedExpertLayout?[] ExpertLayerLayouts { get; init; } = [];
    internal long DenseDeviceBytes { get; init; }
    internal long[] ExpertLayerBytes { get; init; } = [];
    internal long ExpertSourceBytes => ExpertLayerBytes.Aggregate(0L, (a, b) => checked(a + b));
    internal long SourceBytes { get; init; }
    internal long FixedStateBytes => checked(4L * RecurrentLayers *
        (3L * Heads * KdaDim * (ConvKernel - 1) + (long)Heads * KdaDim * KdaDim));
    internal long KvBytesPerToken => checked(2L * AttentionLayers * (KvRow + 2L * IndexerDim));
    // The native slot allocates all n_ctx rows, independent of output allowance.
    internal long SlotBytes(int context) => context > 0 && context <= DeclaredContextTokens
        ? checked(FixedStateBytes + KvBytesPerToken * context + (64L << 10) * Layers)
        : throw new ArgumentOutOfRangeException(nameof(context));

    internal InferenceMemoryBytes Workspace(int context, int rows, int? outputRows = null)
    {
        if (context <= 0 || context > DeclaredContextTokens || rows <= 0 || rows > 1024)
            throw new ArgumentOutOfRangeException(nameof(rows));
        int outputs = outputRows ?? rows;
        if (outputs < 1 || outputs > rows) throw new ArgumentOutOfRangeException(nameof(outputRows));
        checked
        {
            // The full-indexer masks remain alive across the graph. Include both
            // causal and selected masks, the logits, mHC streams, routed FFN
            // intermediates and a recurrent-output generation. This is a forecast,
            // not a replacement for allocator callbacks or driver observation.
            long masks = 4L * AttentionLayers * context * rows;
            // The non-speculative serial executor selects the final hidden row
            // before its LM head. Joint decode emits one logits row per lane.
            // Trunk intermediates and masks still scale with every input row.
            long logits = 4L * outputs * Vocab;
            long activation = logits + 4L * rows * (32L * Hidden * HyperConnections + 8L * ExpertFf * UsedExperts);
            long device = (64L << 20) + masks + activation + FixedStateBytes;
            long host = (32L << 20) + SelectedRouteScratchHostBytes + SelectedRouteMetadataHostBytes + logits
                + 4L * rows * (8L * Hidden + 8L * ExpertFf * UsedExperts)
                + 4L * context * rows;
            return new(host, device);
        }
    }

    internal InferenceMemoryBytes GraphPeak(int context, int rows, bool batched, int generations = 2)
    {
        if (generations < 1 || generations > 33) throw new ArgumentOutOfRangeException(nameof(generations));
        var workspace = Workspace(context, rows, batched ? rows : 1);
        return new(checked(generations * workspace.Host), checked(generations *
            (workspace.Device + (batched ? (rows - 1L) * FixedStateBytes : 0))));
    }

    internal long ExecutionPlacementReserve(int context, int prefillRows, int requests)
    {
        if (requests < 1 || requests > 8) throw new ArgumentOutOfRangeException(nameof(requests));
        var solo = GraphPeak(context, 1, false);
        var prefill = GraphPeak(context, prefillRows, false);
        var joint = GraphPeak(context, requests, requests > 1);
        return checked(SlotBytes(context) * requests + Math.Max(solo.Device, Math.Max(prefill.Device, joint.Device)));
    }

    // Wider prefill is separately admitted from the actual prompt. This
    // placement still protects every full physical slot and full-context joint
    // decode; it neither promises nor authorizes a maximum-width, full-context graph.
    internal long AdaptiveExecutionPlacementReserve(int context, int requests)
    {
        if (requests < 1 || requests > 8) throw new ArgumentOutOfRangeException(nameof(requests));
        var solo = GraphPeak(context, 1, false);
        var joint = GraphPeak(context, requests, requests > 1);
        return checked(SlotBytes(context) * requests + Math.Max(solo.Device, joint.Device));
    }

    internal int SelectCpuExpertLayers(long deviceCeiling, long executionReserve, long visionBytes)
    {
        if (deviceCeiling < 0 || executionReserve < 0 || visionBytes < 0)
            throw new ArgumentOutOfRangeException(nameof(deviceCeiling));
        long required = checked(DenseDeviceBytes + ExpertSourceBytes + executionReserve + visionBytes);
        int cpu = 0;
        while (cpu < Layers && required > deviceCeiling)
            required = checked(required - ExpertLayerBytes[cpu++]);
        if (required > deviceCeiling)
            throw new TensorSharp.Memory.MemoryPressureException("GLM dense/vision weights and admitted execution state exceed CUDA capacity even with every routed expert on the host.");
        return cpu;
    }
    internal static long SelectExpertCacheCapacity(long deviceCeiling, long denseBytes,
        long executionReserve, long visionBytes, long operatorMaximum)
    {
        if (deviceCeiling < 0 || denseBytes < 0 || executionReserve < 0 || visionBytes < 0 || operatorMaximum < 0)
            throw new ArgumentOutOfRangeException(nameof(deviceCeiling));
        decimal slack = (decimal)deviceCeiling - denseBytes - executionReserve - visionBytes;
        return (long)Math.Max(0, Math.Min(slack, operatorMaximum));
    }

    internal static GlmUnifiedMemoryGeometry Read(GgufFile file)
    {
        string a = file.GetString("general.architecture") ?? "";
        if (!GlmDsaArchitecture.IsGlm5Next(a))
            throw new NotSupportedException("Unified GLM requires glm5next or glm5-next metadata.");
        int Int(string name, int fallback = 0)
        {
            uint value = file.GetUint32(a + "." + name, (uint)fallback);
            if (value > int.MaxValue) throw new NotSupportedException("GLM metadata exceeds supported range: " + name);
            return (int)value;
        }
        int all = Int("block_count"), next = Int("nextn_predict_layers"), layers = checked(all - next);
        int hidden = Int("embedding_length"), heads = Int("attention.head_count"), vocab = Int("vocab_size");
        int kv = Int("attention.kv_lora_rank"), dim = Int("kda.head_dim"), conv = Int("ssm.conv_kernel");
        int idx = Int("attention.indexer.key_length"), hc = Int("hyper_connection.count");
        int ff = Int("expert_feed_forward_length"), used = Int("expert_used_count"), experts = Int("expert_count");
        uint[]? layout = file.GetUint32Array(a + ".attention.head_count_kv");
        if (layers <= 0 || next < 0 || next >= all || hidden <= 0 || heads <= 0 || vocab <= 0
            || kv <= 0 || dim <= 0 || conv <= 1 || idx <= 0 || hc <= 0 || ff <= 0 || used <= 0 || used > experts
            || layout == null || layout.Length != all || layout.Take(layers).Any(n => n > 1))
            throw new NotSupportedException("Incomplete GLM5Next unified state/expert geometry.");
        if (Int("rope.dimension_count") != 0 || Int("attention.indexer.kpool") <= 0)
            throw new NotSupportedException("The unified GLM5Next adapter requires NoPE MLA and pooled indexing.");
        long dense = 4L << 20, source = 0;
        long[] expertBytes = new long[layers];
        var projections = new GlmUnifiedExpertProjection?[layers, 3];
        checked
        {
            foreach (var tensor in file.Tensors.Values)
            {
                long bytes = file.GetTensorFileRegion(tensor.Name).ByteLength;
                source += bytes;
                int layer = -1;
                if (tensor.Name.StartsWith("blk.", StringComparison.Ordinal))
                {
                    int dot = tensor.Name.IndexOf('.', 4);
                    if (dot < 0 || !int.TryParse(tensor.Name.AsSpan(4, dot - 4), NumberStyles.None,
                        CultureInfo.InvariantCulture, out layer)) throw new NotSupportedException("Invalid GLM layer tensor: " + tensor.Name);
                    if (layer >= layers) continue; // explicit trunk-only policy
                }
                if (MoeCpuOffloadConfig.IsRoutedExpertWeightName(tensor.Name))
                {
                    if (layer < 0 || tensor.Shape.Length != 3 || tensor.Shape[2] != (ulong)experts)
                        throw new NotSupportedException("Invalid GLM routed expert layout: " + tensor.Name);
                    bool down = tensor.Name.EndsWith("ffn_down_exps.weight", StringComparison.Ordinal);
                    if ((!down && !tensor.Name.EndsWith("ffn_gate_exps.weight", StringComparison.Ordinal)
                            && !tensor.Name.EndsWith("ffn_up_exps.weight", StringComparison.Ordinal))
                        || tensor.Shape[0] != (ulong)(down ? ff : hidden)
                        || tensor.Shape[1] != (ulong)(down ? hidden : ff))
                        throw new NotSupportedException("GLM routed tensor disagrees with its declared uniform expert geometry: " + tensor.Name);
                    expertBytes[layer] += bytes + (64L << 10);
                    int part = down ? 2 : tensor.Name.EndsWith("ffn_gate_exps.weight", StringComparison.Ordinal) ? 0 : 1;
                    long width = down ? ff : hidden, rows = down ? hidden : ff;
                    long block = GgufFile.GetBlockSize(tensor.Type), size = GgufFile.GetTypeSize(tensor.Type);
                    if (width % block != 0 || width / block * size * rows * experts != bytes)
                        throw new NotSupportedException("Invalid GLM expert projection byte geometry: " + tensor.Name);
                    long padding = (512 - width % 512) % 512;
                    bool quantized = block > 1 && (uint)tensor.Type <= (uint)GgmlTensorType.Q2_0;
                    bool supported = quantized && padding % block == 0;
                    long tail = supported ? padding / block * size : 0;
                    projections[layer, part] = new(tensor.Type, bytes / experts, tail, supported);
                }
                else dense += bytes + (64L << 10); // native keeps source quantization; includes CUDA tails/alignment
            }
        }
        if (!file.Tensors.TryGetValue("token_embd.weight", out var embedding)
            || embedding.Shape.Length != 2 || embedding.Shape[0] != (ulong)hidden || embedding.Shape[1] != (ulong)vocab)
            throw new NotSupportedException("GLM token embedding disagrees with its metadata.");
        var expertLayouts = new GlmUnifiedExpertLayout?[layers];
        for (int layer = 0; layer < layers; layer++)
        {
            if (expertBytes[layer] == 0) continue;
            if (projections[layer, 0] is not { } gate || projections[layer, 1] is not { } up
                || projections[layer, 2] is not { } down)
                throw new NotSupportedException("Incomplete GLM routed expert triple at layer " + layer);
            expertLayouts[layer] = new(gate, up, down);
        }
        return new() { Architecture = a, Layers = layers, DeclaredContextTokens = Int("context_length"),
            RecurrentLayers = layout.Take(layers).Count(n => n == 0), Hidden = hidden, Heads = heads, Vocab = vocab,
            KvRow = kv, IndexerDim = idx, KdaDim = dim, ConvKernel = conv, HyperConnections = hc,
            ExpertFf = ff, UsedExperts = used, ExpertCount = experts, ExpertLayerLayouts = expertLayouts,
            DenseDeviceBytes = dense, ExpertLayerBytes = expertBytes, SourceBytes = source };
    }
}

internal sealed class GlmConstructionCleanupException : Exception
{
    internal GlmDsaModel UnreleasedModel { get; }
    internal GlmConstructionCleanupException(GlmDsaModel model, Exception creation, Exception cleanup)
        : base("GLM creation and cleanup both failed; its session retains ownership for retry.", new AggregateException(creation, cleanup))
        => UnreleasedModel = model;
}

public partial class GlmDsaModel
{
    // A shared-budget session retains one complete native graph. A replacement
    // may coexist during allocation, which the session forecasts separately.
    internal const int UnifiedGraphCacheCapacity = 1;
    private int? _unifiedRetainedSlotLimit;
    private Action<GlmUnifiedGraphShape>? _unifiedGraphAdmission;
    internal static void ValidateUnifiedMemoryPolicy(ModelMemoryPolicy? policy, BackendType backend,
        int tp, int split, bool directCuda)
    {
        if (policy == null) return;
        if (backend != BackendType.GgmlCuda || tp != 1 || split != 1 || directCuda
            || Environment.GetEnvironmentVariable("TS_GLM_NATIVE") == "0")
            throw new NotSupportedException("Unified GLM requires one native GGML CUDA device.");
        if (!policy.OmitEmbeddedDraftWeights || policy.NativeMoeCpuLayers is null)
            throw new NotSupportedException("Unified GLM requires explicit trunk-only placement.");
        if (policy.NativeMoeCpuLayers < 0 || policy.PrefillChunkTokens > 1024 || policy.DeviceExecutionReserveBytes < 0)
            throw new ArgumentOutOfRangeException(nameof(policy));
    }
    internal void ConfigureUnifiedMemory(int retainedSlots)
    {
        ArgumentOutOfRangeException.ThrowIfNegative(retainedSlots);
        lock (_nativeSync)
        {
            if (_exec is not NativeGlmExecutor native || MemoryPolicy == null)
                throw new NotSupportedException("Unified GLM needs its native memory-policy executor.");
            if (!native.ConfigureGraphCache(UnifiedGraphCacheCapacity, 0)) throw new InvalidOperationException("Native GLM graph-cache policy was refused.");
            if (!native.ConfigureSharedGraphOwner(true)) throw new InvalidOperationException("Native GLM shared graph ownership was refused.");
            _unifiedRetainedSlotLimit = retainedSlots;
        }
    }
    internal void ConfigureUnifiedGraphAdmission(Action<GlmUnifiedGraphShape> admission)
    {
        ArgumentNullException.ThrowIfNull(admission);
        lock (_nativeSync) _unifiedGraphAdmission = admission;
    }
    internal int UnifiedExpertGroups
    {
        get { lock (_nativeSync) return _exec is NativeGlmExecutor native ? native.SelectedExpertGroups : 0; }
    }
    internal long PlanUnifiedExpertMinimum(int rows)
    {
        lock (_nativeSync)
            return _exec is NativeGlmExecutor native ? native.PlanSelectedExpertMinimum(rows)
                : throw new NotSupportedException("Selected experts require the native GLM executor.");
    }
    internal void ConfigureUnifiedSelectedExperts()
    {
        lock (_nativeSync)
            if (_exec is not NativeGlmExecutor native || !native.ConfigureSelectedExperts(true))
                throw new InvalidOperationException("Native GLM selected-expert scope does not match its measured cache plan.");
    }
    private void PrepareUnifiedGraph(int context, int rows, bool batched)
    {
        if (MemoryPolicy == null) return;
        if (_unifiedGraphAdmission == null) throw new InvalidOperationException("Unified GLM graph admission is not configured.");
        _unifiedGraphAdmission(new(context, rows, batched));
    }
    internal bool CanReuseUnifiedSlot(string requestId, bool includeIdlePrimary = true)
    {
        lock (_nativeSync)
            return _slotByRequest?.ContainsKey(requestId) == true
                || (includeIdlePrimary && _primarySlot >= 0
                    && (_slotByRequest == null || !_slotByRequest.ContainsValue(_primarySlot)));
    }
    private Func<string, bool>? _unifiedPrimarySlotReuse;
    internal void ConfigureUnifiedPrimarySlotReuse(Func<string, bool> canReuse)
    {
        ArgumentNullException.ThrowIfNull(canReuse);
        lock (_nativeSync) _unifiedPrimarySlotReuse = canReuse;
    }
    internal void TrimUnifiedGraphs(bool onlyPrefill = false)
    {
        lock (_nativeSync)
            if (_exec is NativeGlmExecutor native && !native.ConfigureGraphCache(UnifiedGraphCacheCapacity, onlyPrefill ? 2 : 1))
                throw new InvalidOperationException("Native GLM idle graph reclamation was refused.");
    }
    internal void TrimUnifiedRetainedSlots(bool terminalOnly = false)
    {
        lock (_nativeSync)
        {
            if (_retainedSlots == null || _exec == null) return;
            // Discretionary wider-prefill reclamation must not discard the
            // shared checkpoints which startup just prepared. Real pressure
            // may still reclaim every idle owner through the default path.
            string[] keys = terminalOnly
                ? _retainedSlots.Where(entry => !entry.Value.IsCopy).Select(entry => entry.Key).ToArray()
                : _retainedSlots.Keys.ToArray();
            for (int i = 0; i < keys.Length; ++i)
            {
                SlotRetention.ReleaseSlots(_retainedSlots, keys.AsSpan(i, 1), new ExecutorGlmSlotStore(_exec));
                // A refused physical free retains both its native ownership
                // and reusable metadata. Notify each successful removal now,
                // even if a later free throws before the trim can complete.
                if (!_retainedSlots.ContainsKey(keys[i]))
                    _prefixCacheSink?.OnPayloadInvalidated(keys[i],
                        TensorSharp.Runtime.Scheduling.PrefixCache.InvalidationReason.NativeSlotReclaimed);
            }
        }
    }
    internal bool TrimUnifiedMappedResidency()
    {
        lock (_nativeSync)
            return _exec is NativeGlmExecutor native && native.TrimMappedResidency();
    }
    public override void TrimIdleMemory()
    {
        if (MemoryPolicy != null) TrimUnifiedGraphs();
        base.TrimIdleMemory();
    }
}

internal readonly record struct GlmUnifiedGraphShape(int ContextTokens, int Rows, bool Batched);
