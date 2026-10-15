// Copyright (c) Zhongkai Fu. Licensed under the BSD-3-Clause license.
using System;
using System.Linq;
using TensorSharp.Memory.Planning;
using TensorSharp.Runtime;

namespace TensorSharp.Models;

internal sealed class Qwen4ExpConstructionCleanupException : Exception
{
    internal Qwen4ExpModel UnreleasedModel { get; }
    internal Qwen4ExpConstructionCleanupException(Qwen4ExpModel model, Exception creation, Exception cleanup)
        : base("Qwen4Exp creation and cleanup both failed; the model remains owned for cleanup retry.", new AggregateException(creation, cleanup))
        => UnreleasedModel = model;
}

/// <summary>Additional host payload forecast for one serial, file-backed CPU
/// expert layer. Native tickets validate actual buffer/graph-plan sizes before
/// allocation; this forecast is not a bound on libc, thread stacks or page cache.</summary>
internal readonly record struct Qwen4ExpHostPrefillMemory(
    long ExpertStageBytes, long GraphTensorBytes, long GraphWorkBytes, long AuxiliaryBytes)
{
    internal long TotalBytes => checked(ExpertStageBytes + GraphTensorBytes + GraphWorkBytes + AuxiliaryBytes);
}

/// <summary>One compact expert layer in the dedicated CUDA stream allocator.
/// Weights include each projection's quantized tail and conservative alignment;
/// graph bytes exclude those weights. Backend/driver pools remain observed margin.</summary>
internal readonly record struct Qwen4ExpDeviceExpertStreamMemory(
    long ExpertWeightBytes, long GraphTensorBytes, long AllocationSlackBytes)
{
    internal long TotalBytes => checked(ExpertWeightBytes + GraphTensorBytes + AllocationSlackBytes);
}

/// <summary>Single GGML-CUDA trunk payload forecast. Source mappings, GC objects,
/// backend pools and driver memory are not a physical hard-limit certificate.
/// Native allocation callbacks must still enforce actual graph/cache allocations.</summary>
internal sealed record Qwen4ExpUnifiedGeometry
{
    internal int Layers { get; init; }
    internal int DeclaredContextTokens { get; init; }
    internal int RecurrentLayers { get; init; }
    internal int AttentionLayers { get; init; }
    internal int QsaLayers { get; init; }
    internal int Hidden { get; init; }
    internal int HyperHidden { get; init; }
    internal int Heads { get; init; }
    internal int KvHeads { get; init; }
    internal int HeadDim { get; init; }
    internal int Vocab { get; init; }
    internal int ConvDim { get; init; }
    internal int ConvKernel { get; init; }
    internal int StateDim { get; init; }
    internal int ValueHeads { get; init; }
    internal int ExpertFf { get; init; }
    internal int SharedFf { get; init; }
    internal int UsedExperts { get; init; }
    internal int ExpertCount { get; init; }
    internal int IndexerHeads { get; init; }
    internal int IndexerDim { get; init; }
    internal long PleStateBytes { get; init; }
    internal long LoadingDeviceBoundBytes { get; init; }
    internal long DenseDeviceBytes => LoadingDeviceBoundBytes;
    internal long ExpertSourceBytes => ExpertLayerBytes.Aggregate(0L, (sum, value) => checked(sum + value));
    internal long ResidentHostPayloadBytes { get; init; }
    internal long LoadingHostTemporaryBytes { get; init; }
    internal long MappedSourceBytes { get; init; }
    internal long[] ExpertLayerBytes { get; init; } = [];
    // Separate parts preserve mixed quantization and each compact tensor's
    // alignment; dividing a model-wide byte sum would understate large layers.
    internal long[][] ExpertProjectionBytes { get; init; } = [];
    // CUDA adds one padded row tail to a quantized tensor, not one per expert.
    // Keep each part's type-specific tail, including zero for F16/F32 tensors.
    internal long[][] ExpertProjectionCudaTailBytes { get; init; } = [];
    internal long[] ExpertCacheMinimumBytes { get; init; } = [];
    internal long[] ExpertCacheAllocationFloorBytes { get; init; } = [];
    // Native quotas are equal per layer. Sum alone is insufficient for mixed quants.
    internal long AllLayerExpertCacheMinimumBytes => ExpertCacheMinimumBytes.Length == Layers
        && ExpertCacheMinimumBytes.All(x => x > 0) ? checked(ExpertCacheMinimumBytes.Max() * Layers) : 0;

    internal long KvBytesPerToken(int elementBytes) => checked((2L * AttentionLayers * KvHeads * HeadDim
        + (long)QsaLayers * IndexerDim) * elementBytes);

    internal Qwen4ExpHostPrefillMemory HostPrefillPeak(int prefillTokens, int cpuThreads = 0)
    {
        ArgumentOutOfRangeException.ThrowIfNegativeOrZero(prefillTokens);
        ArgumentOutOfRangeException.ThrowIfNegative(cpuThreads);
        // Cache hits can avoid this allocation, but admission cannot assume a
        // hit: disabled/refused device cache also routes one-token requests to
        // the bounded file-backed CPU path. Forecast every positive row count.
        if (ExpertCount <= 0 || UsedExperts <= 0 || UsedExperts > ExpertCount
            || Hidden <= 0 || ExpertFf <= 0 || ExpertProjectionBytes.Length != Layers || Layers <= 0)
            throw new NotSupportedException("Incomplete qwen4exp CPU expert staging geometry.");
        // Native auto selection is capped at 64 regardless of hardware. Respect
        // the managed explicit override, including values above 64. A direct
        // native override outside this configuration still has actual quota
        // enforcement, but cannot be inferred from this metadata forecast.
        int threads = cpuThreads > 0 ? cpuThreads : MoeCpuOffloadConfig.CpuThreads > 0 ? MoeCpuOffloadConfig.CpuThreads : 64;
        checked
        {
            long b = prefillTokens, h = Hidden, f = ExpertFf, k = UsedExperts;
            long unique = Math.Min(ExpertCount, b * k);
            long stage = 0;
            foreach (long[] parts in ExpertProjectionBytes)
            {
                if (parts is null || parts.Length != 3)
                    throw new NotSupportedException("CPU expert staging requires separate gate/up/down metadata.");
                long layer = 0;
                foreach (long bytes in parts)
                {
                    if (bytes <= 0 || bytes % ExpertCount != 0)
                        throw new NotSupportedException("Invalid per-expert file payload size.");
                    long compact = bytes / ExpertCount * unique;
                    layer += (compact + 63) / 64 * 64;
                }
                // A single aligned arena is reused across layers; native grows
                // it only after freeing/refunding the previous arena.
                stage = Math.Max(stage, layer + 63);
            }
            // gate/up/GLU, down/weighted, K-1 aggregation outputs, IDs and
            // weights. 64-byte allocation slack covers the per-tensor binds.
            long tensors = 4 * b * (3 * f * k + h * (3 * k - 1)) + 8 * b * k + 64 * (32 + 3 * k);
            // Upstream MUL_MAT_ID planning: converted input (<=2 bytes/element),
            // expert-row mapping, counters/atomics, and a 512 KiB tile slot per
            // thread. Work arena has its own 63-byte alignment allocation.
            long work = 2 * b * Math.Max(h, f * k) + 8 * unique * k * b + 72 * unique
                + (512L << 10) * threads + 128L * threads + 1024 + 63;
            // Pooled 4 MiB graph context, retained seam input/output/routes and
            // conservative compact-ID/source descriptors. These can coexist
            // with the staged weights and both graph allocations.
            long auxiliary = (4L << 20) + 8 * b * h + 8 * b * k + 64 * (ExpertCount + b * k + 32);
            return new(stage, tensors, work, auxiliary);
        }
    }

    internal Qwen4ExpDeviceExpertStreamMemory DeviceExpertStreamPeak(int prefillTokens)
    {
        ArgumentOutOfRangeException.ThrowIfNegativeOrZero(prefillTokens);
        if (ExpertCount <= 0 || UsedExperts <= 0 || UsedExperts > ExpertCount || Hidden <= 0 || ExpertFf <= 0
            || Layers <= 0 || ExpertProjectionBytes.Length != Layers || ExpertProjectionCudaTailBytes.Length != Layers)
            throw new NotSupportedException("Incomplete qwen4exp CUDA expert streaming geometry.");
        checked
        {
            long b = prefillTokens, h = Hidden, f = ExpertFf, k = UsedExperts;
            long unique = Math.Min(ExpertCount, b * k);
            long weights = 0;
            for (int layer = 0; layer < Layers; layer++)
            {
                long[] parts = ExpertProjectionBytes[layer], tails = ExpertProjectionCudaTailBytes[layer];
                if (parts is null || parts.Length != 3 || tails is null || tails.Length != 3)
                    throw new NotSupportedException("CUDA expert streaming requires separate gate/up/down metadata.");
                long compact = 0;
                for (int part = 0; part < 3; part++)
                {
                    if (parts[part] <= 0 || parts[part] % ExpertCount != 0 || tails[part] < 0)
                        throw new NotSupportedException("Invalid CUDA expert projection payload or padding.");
                    long bytes = parts[part] / ExpertCount * unique + tails[part];
                    // The pinned CUDA buft uses 128-byte alignment. 256 leaves
                    // conservative allocation slack without padding every expert.
                    compact += (bytes + 255) / 256 * 256;
                }
                weights = Math.Max(weights, compact);
            }
            // Separate gate/up/GLU, down/weighted, K-1 route-order sums, plus
            // input/output and pinned route IDs/weights. Count all intermediates
            // even when gallocr can reuse their lifetimes; do not include weights.
            long tensors = 4 * b * (3 * f * k + h * (3 * k + 1)) + 8 * b * k + 256 * (32 + 3 * k);
            // This file-stream gallocr is independent of the live outer graph.
            // Native scope Trim releases it at the quiescent request boundary;
            // layer growth frees the old allocation before reserving the new one.
            // Do not sum layer arenas or manufacture an old+new duplicate here.
            // Forecast all positive B: native caches its environment threshold
            // (default 128, override may be 1 or disabled) on first use. Reading
            // current process environment cannot safely prove a path is disabled.
            return new(weights, tensors, 2L << 20);
        }
    }

    internal InferenceMemoryBytes RequestPeak(int capacityTokens, int prefillTokens, int elementBytes = 2,
        int checkpointCopies = 0)
    {
        ArgumentOutOfRangeException.ThrowIfNegativeOrZero(capacityTokens);
        ArgumentOutOfRangeException.ThrowIfNegative(checkpointCopies);
        if (prefillTokens <= 0 || prefillTokens > capacityTokens) throw new ArgumentOutOfRangeException(nameof(prefillTokens));
        if (elementBytes is not (2 or 4)) throw new NotSupportedException("qwen4exp live caches require F16 or F32.");
        checked
        {
            long kv = KvBytesPerToken(elementBytes) * capacityTokens;
            long conv = (long)RecurrentLayers * (ConvKernel - 1) * ConvDim * 4;
            long state = (long)RecurrentLayers * ValueHeads * StateDim * StateDim * 4;
            long positions = QsaLayers > 0 ? 3L * capacityTokens * 4 : 0;
            // Host conv ring and tensor seed are distinct. Device has one conv
            // seed/state copy; PLE state has separate host/device ownership too.
            long hostState = kv + 2 * conv + state + PleStateBytes + positions;
            long deviceState = kv + conv + state + PleStateBytes;
            long perRow = 4L * (12L * HyperHidden + 16L * Hidden + 6L * ConvDim
                + 4L * ExpertFf * UsedExperts + 3L * UsedExperts * Hidden + 4L * SharedFf
                + (long)(Heads + IndexerHeads) * capacityTokens);
            long graph = (64L << 20) + prefillTokens * perRow + 4L * Vocab;
            var expertStream = DeviceExpertStreamPeak(prefillTokens);
            long existingRoutedActivations = 4L * prefillTokens * (4L * ExpertFf * UsedExperts + 3L * UsedExperts * Hidden);
            // The old graph estimate already included routed expert activation
            // space. Replace that component with the larger bound instead of
            // charging it twice; the independent selected weights are additional.
            graph = graph - existingRoutedActivations + Math.Max(existingRoutedActivations, expertStream.GraphTensorBytes)
                + expertStream.ExpertWeightBytes + expertStream.AllocationSlackBytes;
            // A retained decode graph can coexist with prefill and live state.
            long decodeGraph = (64L << 20) + Layers * perRow + state;
            long hostScratch = (16L << 20) + prefillTokens * (4L * (3L * Hidden + HyperHidden) + 12)
                + 2L * capacityTokens * prefillTokens + 8L * Vocab;
            // An exact-state checkpoint is a separate host-authoritative copy.
            // Reserve it in addition to the old/new holder growth allowance;
            // its tensor allocations remain charged after the request ends.
            return new((2L + checkpointCopies) * hostState + hostScratch + HostPrefillPeak(prefillTokens).TotalBytes,
                2 * deviceState + graph + decodeGraph);
        }
    }

    /// <summary>One retained joint-decode entry. Native uses N live lanes and
    /// one scratch lane for attention/GDN/PLE; QSA uses only the N live lanes.
    /// This is additional to every request's solo state and selected experts.</summary>
    internal InferenceMemoryBytes JointDecodePeak(int contextTokens, int sequences)
    {
        if (contextTokens <= 0 || sequences is < 2 or > 8) throw new ArgumentOutOfRangeException();
        checked
        {
            long cap = ((long)contextTokens + 255) / 256 * 256;
            long lanes = sequences + 1L;
            long conv = (long)RecurrentLayers * (ConvKernel - 1) * ConvDim * 4;
            long state = (long)RecurrentLayers * ValueHeads * StateDim * StateDim * 4;
            long attention = 4L * AttentionLayers * KvHeads * HeadDim * cap;
            long qsa = 2L * QsaLayers * IndexerDim * cap * sequences;
            // ratio>=1 => block count<=cap and ratio*ceil(cap/ratio)<=cap+63.
            long selectors = 4L * QsaLayers * sequences * (7 * cap + 67);
            long replicas = lanes * (attention + conv + state + PleStateBytes) + qsa + selectors;
            long perRow = 4L * (12L * HyperHidden + 16L * Hidden + 6L * ConvDim
                + 4L * ExpertFf * UsedExperts + 3L * UsedExperts * Hidden + 4L * SharedFf
                + (long)(Heads + IndexerHeads) * cap);
            var seam = DeviceExpertStreamPeak(sequences);
            long graph = (128L << 20) + (long)Layers * sequences * perRow + 4L * sequences * Vocab;
            long nodes = (long)Layers * (192 + 160L * sequences) + 32768;
            // Native graph metadata and managed pinned descriptor arrays are
            // forecast margins, not claimed as instrumented payload owners.
            long host = (64L << 20) + (nodes + 8192) * 512 + nodes * 32
                + 4L * sequences * Vocab + 12L * sequences * cap + HostPrefillPeak(sequences).TotalBytes;
            return new(host, replicas + graph + 2 * cap * sequences
                + seam.ExpertWeightBytes + seam.GraphTensorBytes + seam.AllocationSlackBytes);
        }
    }

    internal static Qwen4ExpUnifiedGeometry Read(GgufFile file)
    {
        ArgumentNullException.ThrowIfNull(file);
        const string a = Qwen4ExpModel.ArchitectureId;
        if (file.GetString("general.architecture") != a) throw new NotSupportedException("Expected qwen4exp metadata.");
        int Int(string key, int fallback = 0)
        {
            if (!file.Metadata.TryGetValue(a + "." + key, out var raw)) return fallback;
            long value = raw switch
            {
                byte v => v, sbyte v => v, ushort v => v, short v => v, uint v => v,
                int v => v, long v => v, ulong v when v <= int.MaxValue => (long)v,
                _ => throw new NotSupportedException("Unsupported scalar geometry: " + key)
            };
            if (value < 0 || value > int.MaxValue) throw new NotSupportedException("Invalid geometry: " + key);
            return (int)value;
        }
        int layers = Int("block_count"), hidden = Int("embedding_length"), heads = Int("attention.head_count");
        if (layers <= 0 || hidden <= 0 || heads <= 0) throw new NotSupportedException("Missing qwen4exp geometry.");
        int kvHeads = Int("attention.head_count_kv"), headDim = Int("attention.key_length", hidden / heads);
        int hc = Int("hyper_connection.count"), convKernel = Int("ssm.conv_kernel"), stateDim = Int("ssm.state_size");
        int keyHeads = Int("ssm.group_count"), valueHeads = Int("ssm.time_step_rank");
        int ff = Int("expert_feed_forward_length"), used = Int("expert_used_count"), experts = Int("expert_count");
        if (new[] { kvHeads, headDim, hc, convKernel, stateDim, keyHeads, valueHeads, ff, used, experts }.Any(x => x <= 0)
            || used > experts) throw new NotSupportedException("Incomplete qwen4exp state/expert geometry.");
        uint[] recurrent = file.GetUint32Array(a + ".attention.recurrent_layers");
        if (recurrent != null && recurrent.Length != layers) throw new NotSupportedException("Incomplete recurrent layout.");
        int interval = Int("full_attention_interval", 4);
        if (interval <= 0) throw new NotSupportedException("Invalid attention interval.");
        int[] compression = file.GetInt32Array(a + ".attention.compress_ratios");
        if (compression != null && compression.Length != layers) throw new NotSupportedException("Incomplete QSA layout.");
        int indexerHeads = Int("attention.indexer.head_count"), indexerDim = Int("attention.indexer.key_length");
        int recurrentCount = 0, qsaCount = 0;
        for (int i = 0; i < layers; i++)
        {
            bool rec = recurrent != null ? recurrent[i] != 0 : (i + 1) % interval != 0;
            if (rec) recurrentCount++;
            else if (compression != null && compression[i] > 0 && indexerHeads > 0 && indexerDim > 0) qsaCount++;
        }
        if (!file.Tensors.TryGetValue("token_embd.weight", out var embedding) || embedding.Shape.Length != 2
            || embedding.Shape[0] != (ulong)hidden || embedding.Shape[1] > int.MaxValue)
            throw new NotSupportedException("Invalid token embedding geometry.");
        long device = 0, host = 0, temporary = 0, mapped = 0;
        var expertBytes = new long[layers]; var minimum = new long[layers]; var floors = new long[layers];
        var projectionBytes = new long[layers][]; var projectionTails = new long[layers][];
        checked
        {
            foreach (var t in file.Tensors.Values)
            {
                long source = file.GetTensorFileRegion(t.Name).ByteLength;
                mapped += source;
                if (t.Name == "per_layer_token_embd.weight" || MoeCpuOffloadConfig.IsRoutedExpertWeightName(t.Name)) continue;
                bool raw = !Qwen4ExpModel.RequiresFloatGeometry(t.Name)
                    && ModelBase.ShouldStoreWeightQuantized(BackendType.GgmlCuda, t);
                long payload = raw ? source : t.NumElements * 4;
                device += payload + (64L << 10); // quantized allocation tails/alignment
                if (!raw) { host += payload; temporary = Math.Max(temporary, source + payload); }
            }
            for (int layer = 0; layer < layers; layer++)
            {
                projectionBytes[layer] = new long[3];
                projectionTails[layer] = new long[3];
                long selected = 0, tails = 0; bool supported = used <= 64;
                string[] parts = ["gate", "up", "down"];
                for (int i = 0; i < 3; i++)
                {
                    if (!file.Tensors.TryGetValue($"blk.{layer}.ffn_{parts[i]}_exps.weight", out var t))
                    { supported = false; continue; }
                    long bytes = file.GetTensorFileRegion(t.Name).ByteLength;
                    projectionBytes[layer][i] = bytes;
                    expertBytes[layer] += bytes;
                    long width = i == 2 ? ff : hidden, rows = i == 2 ? hidden : ff;
                    long block = GgufFile.GetBlockSize(t.Type), size = GgufFile.GetTypeSize(t.Type);
                    if (block <= 0 || size <= 0 || width % block != 0 || t.Shape.Length != 3 || t.Shape[0] != (ulong)width
                        || t.Shape[1] != (ulong)rows || t.Shape[2] != (ulong)experts
                        || width / block * size * rows * experts != bytes)
                        throw new NotSupportedException("Invalid qwen4exp expert projection geometry: " + t.Name);
                    long paddingElements = (512 - width % 512) % 512;
                    if (block > 1 && paddingElements % block != 0)
                        throw new NotSupportedException("Unsupported CUDA expert row padding: " + t.Name);
                    projectionTails[layer][i] = block > 1 ? paddingElements / block * size : 0;
                    if (block <= 1 || width % block != 0 || t.Shape.Length != 3 || t.Shape[0] != (ulong)width
                        || t.Shape[1] != (ulong)rows || t.Shape[2] != (ulong)experts
                        || width / block * size * rows * experts != bytes) { supported = false; continue; }
                    selected += bytes / experts * used;
                    tails += ((512 - width % 512) % 512) / block * size;
                }
                if (file.Tensors.TryGetValue($"blk.{layer}.ffn_gate_up_exps.weight", out var fused))
                { expertBytes[layer] += file.GetTensorFileRegion(fused.Name).ByteLength; supported = false; }
                if (supported)
                {
                    floors[layer] = (1L << 20) + selected;
                    minimum[layer] = floors[layer] + tails + 4L * (4L * ff * used + (3L * used + 1) * hidden + 2L * used) + 256L * 128;
                }
            }
            long ple = file.GetInt32Array(a + ".ple.layers")?.Length > 0
                ? (long)Math.Max(0, Int("ple.conv_kernel") - 1) * Int("ple.ngram_size") * hc * hidden * 4 : 0;
            return new() { Layers = layers, DeclaredContextTokens = Int("context_length"), RecurrentLayers = recurrentCount, AttentionLayers = layers - recurrentCount,
                QsaLayers = qsaCount, Hidden = hidden, HyperHidden = hc * hidden, Heads = heads, KvHeads = kvHeads,
                HeadDim = headDim, Vocab = (int)embedding.Shape[1], ConvDim = stateDim * (2 * keyHeads + valueHeads),
                ConvKernel = convKernel, StateDim = stateDim, ValueHeads = valueHeads, ExpertFf = ff,
                SharedFf = Int("expert_shared_feed_forward_length", ff), UsedExperts = used, ExpertCount = experts,
                IndexerHeads = indexerHeads, IndexerDim = indexerDim, PleStateBytes = ple,
                LoadingDeviceBoundBytes = device, ResidentHostPayloadBytes = host, LoadingHostTemporaryBytes = temporary,
                MappedSourceBytes = mapped, ExpertLayerBytes = expertBytes, ExpertProjectionBytes = projectionBytes,
                ExpertProjectionCudaTailBytes = projectionTails,
                ExpertCacheMinimumBytes = minimum,
                ExpertCacheAllocationFloorBytes = floors };
        }
    }
}

public partial class Qwen4ExpModel
{
    private Qwen4ExpUnifiedGeometry _unifiedGeometry;
    internal Qwen4ExpUnifiedGeometry UnifiedMemoryGeometry => _unifiedGeometry ??= Qwen4ExpUnifiedGeometry.Read(_gguf);

    internal static void ValidateUnifiedMemoryPolicy(ModelMemoryPolicy policy, BackendType backend,
        int tpDegree, int layerSplitDegree, string draftPath)
    {
        if (policy == null) return;
        if (backend != BackendType.GgmlCuda || tpDegree != 1 || layerSplitDegree != 1)
            throw new NotSupportedException("The qwen4exp memory-policy adapter currently requires one GGML CUDA device.");
        if (policy.ExpertCacheBytes < 0 || policy.DeviceExecutionReserveBytes < 0
            || policy.ExpertCachePrefillRows is < 1 or > 128)
            throw new ArgumentOutOfRangeException(nameof(policy));
        if (!policy.OmitEmbeddedDraftWeights || !string.IsNullOrEmpty(draftPath))
            throw new NotSupportedException("The qwen4exp memory-policy adapter requires explicit trunk-only execution.");
    }

    internal InferenceMemoryBytes GetUnifiedMemoryRequestPeak(int contextTokens, int prefillTokens,
        int checkpointCopies = 0, bool independentHolder = false, string? requestId = null)
    {
        if (contextTokens <= 0 || contextTokens > _maxContextLength || prefillTokens <= 0
            || prefillTokens > contextTokens || (MemoryPolicy != null && prefillTokens > MemoryPolicy.PrefillChunkTokens))
            throw new ArgumentOutOfRangeException(nameof(contextTokens));
        long capacity = GetUnifiedMemoryCapacity(contextTokens, independentHolder, requestId);
        return UnifiedMemoryGeometry.RequestPeak(checked((int)capacity), prefillTokens,
            _kvCacheDtype == KvCacheDtype.F16 ? 2 : 4, checkpointCopies);
    }

    internal int GetUnifiedMemoryCapacity(int requiredTokens, bool independentHolder = true, string? requestId = null)
    {
        if (requiredTokens <= 0 || requiredTokens > _maxContextLength)
            throw new ArgumentOutOfRangeException(nameof(requiredTokens));
        long capacity = independentHolder && _initialKvCacheCapacity > 0 ? _initialKvCacheCapacity : _kvCacheCapacity;
        if (requestId != null && _fusedHolders != null && _fusedHolders.TryGetValue(requestId, out var holder))
            capacity = string.Equals(requestId, _activeFusedKey, StringComparison.Ordinal)
                ? _kvCacheCapacity : holder.KvCapacity;
        capacity = Math.Max(1, capacity);
        while (capacity < requiredTokens) capacity = Math.Min(_maxContextLength, Math.Max(requiredTokens, capacity * 2));
        return checked((int)capacity);
    }

    internal long GetUnifiedMemoryPersistentDeviceBytes() => UnifiedMemoryGeometry.LoadingDeviceBoundBytes;
    internal long GetUnifiedMemoryExpertCacheMinimumBytes() => UnifiedMemoryGeometry.AllLayerExpertCacheMinimumBytes;
}
