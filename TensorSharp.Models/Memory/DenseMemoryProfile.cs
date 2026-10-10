// Copyright (c) Zhongkai Fu. Licensed under the BSD-3-Clause license.
#nullable enable
using System;
using System.Collections.Generic;
using System.Linq;
using TensorSharp.Memory;
using TensorSharp.Memory.Planning;
using TensorSharp.Runtime;

namespace TensorSharp.Models;

/// <summary>Conservative initial geometry for the two adapters with file-backed
/// text execution. This does not certify a native graph's exact allocation size;
/// native allocation callbacks enforce their actual bytes. Unknown architecture,
/// experts and draft layouts must supply a different qualified adapter.</summary>
internal sealed record DenseMemoryProfile(InferenceModelMemory Model, long FusionBytes,
    long LargestProjectionBytes, long Hidden, long Intermediate, long Heads, long Vocab,
    string? StreamingRefusal)
{
    // These are distinct representations. Mapped file ranges may occupy the OS
    // page cache, but are not a second anonymous allocation owned by the model.
    // RetainedMappedWeightBytes is a source-range upper bound: a fusion can
    // replace views with an owned copy or use an adjacent zero-copy view.
    internal long SourceWeightBytes { get; init; }
    internal long ResidentHostWeightBytes { get; init; }
    internal long RetainedMappedWeightBytes { get; init; }
    internal bool RetainsSlidingWindowPrefill { get; init; }
    internal long AttentionProjectionWidth { get; init; }
    internal int RetainedDecodeGraphLayers { get; init; }

    internal InferenceMemoryBytes RequestPeak(int context, int chunk, bool streaming)
    {
        ArgumentOutOfRangeException.ThrowIfNegativeOrZero(context);
        ArgumentOutOfRangeException.ThrowIfNegativeOrZero(chunk);
        long host = Model.RecurrentStatePerSequence.Host, device = Model.RecurrentStatePerSequence.Device;
        foreach (var kv in Model.KvCaches)
        {
            long tokens = kv.WindowTokens == 0 ? context : Math.Min(context, kv.WindowTokens);
            tokens = checked((tokens + kv.AllocationBlockTokens - 1) / kv.AllocationBlockTokens * kv.AllocationBlockTokens);
            long bytes = checked(tokens * kv.BytesPerToken * kv.LayerCount);
            if (kv.Tier == MemoryTier.Host) host = checked(host + bytes);
            else if (kv.Tier == MemoryTier.Accelerator) device = checked(device + bytes);
            else throw new NotSupportedException("Serial live KV must be in RAM or on device.");
        }
        var workspace = Workspace(chunk, context);
        // Old and replacement states coexist while a cache grows/restores.
        // Do not subtract arbitrary committed bytes: they can belong to another
        // request, weight, cache shape, or another engine on the shared ledger.
        return new(checked(host * 2 + workspace.Host),
            checked(device * 2 + workspace.Device + (streaming ? LargestProjectionBytes : 0)));
    }

    internal InferenceMemoryBytes Workspace(int chunk, int context)
    {
        // Include logits, both FFN branches, residual/projection scratch and a
        // materialized attention score matrix even if flash attention avoids it.
        // Existing graph/host pools also retain slabs between invocations.
        long perToken = checked(4 * checked(2 * Intermediate + 16 * Math.Max(Hidden, AttentionProjectionWidth) + Heads * context));
        long bytes = checked((64L << 20) + chunk * perToken + Vocab * 4);
        long sliding = 0, largestExtended = 0;
        if (RetainsSlidingWindowPrefill && chunk > 1)
            foreach (var kv in Model.KvCaches.Where(k => k.Tier == MemoryTier.Host && k.WindowTokens > 0 && context > k.WindowTokens))
            {
                // Gemma's whole-model graph gathers every old SWA window before
                // any cache writes, and keeps donor chunk inputs for shared KV.
                // Extended/converted attention inputs are layer-local on CUDA;
                // sum retained donors but only take the largest local working set.
                long rows = checked(kv.WindowTokens + ((long)chunk + 255) / 256 * 256);
                sliding = checked(sliding + rows * kv.BytesPerToken * kv.LayerCount);
                largestExtended = Math.Max(largestExtended, checked(rows * kv.BytesPerToken));
            }
        sliding = checked(sliding + 2 * largestExtended);
        // Qwen's capture-safe decode context keeps distinct per-layer activation
        // and recurrent-output slots (it cannot use the generic one-layer reuse
        // allocator). That graph can coexist with retained prefill workspace and
        // live KV/state. Account it on device even during the prefill phase, so a
        // later graph build does not exhaust a successfully admitted envelope.
        long decodeGraph = RetainedDecodeGraphLayers == 0 ? 0 : checked((64L << 20)
            + RetainedDecodeGraphLayers * perToken + Vocab * 4 + Model.RecurrentStatePerSequence.Device);
        return new(checked(bytes + sliding), checked(bytes + sliding + decodeGraph));
    }

    internal static DenseMemoryProfile Read(GgufFile file, AdaptiveModelMemoryOptions options,
        bool retainHostQuantizedWeights = false, bool omitEmbeddedDraftWeights = false)
    {
        string? arch = file.GetString("general.architecture");
        if (arch is not ("gemma4" or "qwen35"))
            throw new NotSupportedException($"No adaptive dense CUDA geometry adapter for '{arch}'.");
        long Int(string key, uint fallback = 0)
        {
            if (!file.Metadata.TryGetValue($"{arch}.{key}", out var value)) return fallback;
            // GgufFile's convenience getter accepts arrays by taking element 0.
            // A geometry forecast must not silently discard an unknown layout.
            long number = value switch
            {
                byte v => v, sbyte v => v, ushort v => v, short v => v,
                uint v => v, int v => v, long v => v, ulong v when v <= long.MaxValue => (long)v,
                _ => throw new NotSupportedException($"Unknown scalar geometry: {arch}.{key}.")
            };
            if (number < 0 || number > int.MaxValue)
                throw new NotSupportedException($"Unsupported geometry range: {arch}.{key}.");
            return number;
        }
        int layers = checked((int)Int("block_count"));
        int nextnLayers = checked((int)Int("nextn_predict_layers"));
        if (layers <= 0 || Int("expert_count") != 0 || Int("expert_used_count") != 0
            || (nextnLayers > 0 && (arch != "qwen35" || !omitEmbeddedDraftWeights || nextnLayers >= layers)))
            throw new NotSupportedException("Adaptive dense loading requires a nonempty dense trunk; embedded Qwen draft layers must be explicitly omitted by the loading policy.");
        layers -= nextnLayers; // Qwen block_count includes trailing NextN layers.
        var activeTensors = file.Tensors.Values.Where(t => nextnLayers == 0
            || !Qwen35Model.IsEmbeddedMtpWeight(t.Name, layers, nextnLayers)).ToArray();
        long hidden = Int("embedding_length"), heads = Int("attention.head_count");
        if (hidden <= 0 || heads <= 0)
            throw new NotSupportedException("Unknown attention geometry.");
        long intermediate = Int("feed_forward_length");
        if (intermediate <= 0) throw new NotSupportedException("Missing feed-forward geometry.");
        long vocab = file.Tensors.TryGetValue("token_embd.weight", out var embedding) && embedding.Shape.Length == 2
            ? checked((long)embedding.Shape[1]) : 0;
        if (vocab <= 0 || vocab > int.MaxValue || embedding!.Shape[0] != (ulong)hidden)
            throw new NotSupportedException("Missing or inconsistent token embedding geometry.");

        long weights = 0, small = 0, largest = 0, sourceWeights = 0, residentHost = 0, mapped = 0;
        long quantFusion = 0, floatFusionPeak = 0, conversionPeak = 0, extraDevice = 0;
        var storedBytes = new Dictionary<string, long>(StringComparer.Ordinal);
        foreach (var tensor in activeTensors)
        {
            if (tensor.Shape.Length == 0 || tensor.Shape.Any(d => d == 0 || d > int.MaxValue)
                || tensor.Name.Contains("_exps.", StringComparison.Ordinal)
                || tensor.Type is GgmlTensorType.PQ2_0 or GgmlTensorType.PTQ1_0)
                throw new NotSupportedException($"No dense memory geometry for tensor '{tensor.Name}'.");
            long sourceBytes = file.GetTensorFileRegion(tensor.Name).ByteLength;
            sourceWeights = checked(sourceWeights + sourceBytes);
            bool vector = tensor.Shape.Length == 1;
            // Match ModelBase.ShouldStoreWeightQuantized(GgmlCuda): every
            // non-F32 2-D tensor, including F16/BF16, stays in its file format.
            bool raw = IsRawMatrix(tensor);
            long bytes = raw ? sourceBytes : checked(tensor.NumElements * 4);
            storedBytes.Add(tensor.Name, bytes);
            if (!raw && tensor.Type != GgmlTensorType.F32)
                conversionPeak = Math.Max(conversionPeak, sourceBytes); // raw read buffer before decode
            if (vector) small = checked(small + bytes);
            else
            {
                weights = checked(weights + bytes);
                if (!raw) residentHost = checked(residentHost + bytes);
                else if (retainHostQuantizedWeights || tensor.Name is "token_embd.weight" or "per_layer_token_embd.weight")
                    mapped = checked(mapped + sourceBytes);
                // Lookup tables gather rows. The tied output head consumes just
                // the final prefill row; it does not need a full N-token GEMM.
                if (tensor.Name is not ("token_embd.weight" or "per_layer_token_embd.weight" or "output.weight"))
                    largest = Math.Max(largest, bytes);
            }
        }
        // Padded device rows and the additional K-as-V projection in Gemma can
        // make the device representation larger than its original file span.
        long padding = checked(activeTensors.Length * (64L << 10));
        // Ordinary fused projections replace source matrices before preload;
        // Qwen's quantized recurrent input pack retains its separate sources;
        // its F32 packing branch replaces them like ordinary projections.
        long persistent = checked(small + (128L << 20));
        var kv = new List<InferenceKvCache>();
        var recurrentLayers = new bool[layers];
        var sharedLayers = new bool[layers];
        long recurrent = 0, attentionWidth = hidden;
        if (arch == "gemma4")
        {
            bool[]? local = file.GetBoolArray($"{arch}.attention.sliding_window_pattern");
            if (file.Metadata.ContainsKey($"{arch}.attention.sliding_window_pattern") && (local == null || local.Length != layers))
                throw new NotSupportedException("Gemma sliding-window pattern must have one boolean per layer.");
            int[]? kvHeads = file.GetInt32Array($"{arch}.attention.head_count_kv");
            if (kvHeads != null && (kvHeads.Length != layers || kvHeads.Any(n => n <= 0)))
                throw new NotSupportedException("Gemma KV-head array must have one positive count per layer.");
            long scalarKvHeads = kvHeads != null ? kvHeads[0] : Int("attention.head_count_kv");
            long globalKvHeads = Int("attention.global_head_count_kv");
            bool globalHeadsExplicit = globalKvHeads != 0;
            if (globalKvHeads == 0 && kvHeads != null && local != null)
                for (int layer = 0; layer < layers; layer++)
                    if (!local[layer]) { globalKvHeads = kvHeads[layer]; break; }
            if (globalKvHeads == 0) globalKvHeads = scalarKvHeads;
            int window = checked((int)Int("attention.sliding_window", 512));
            if (window <= 0) throw new NotSupportedException("Invalid Gemma sliding-window capacity.");
            int shared = checked((int)Int("attention.shared_kv_layers"));
            if (shared >= layers && shared != 0) throw new NotSupportedException("Gemma shared KV layers have no donor region.");
            long localDimension = Int("attention.key_length_swa", 256), globalDimension = Int("attention.key_length", 512);
            // Gemma DetectHeadDimsFromWeights overrides the metadata with the
            // first actual K projection of each attention class.
            foreach (bool sliding in new[] { true, false })
                for (int layer = 0; layer < layers; layer++)
                    if ((local?[layer] ?? false) == sliding
                        && file.Tensors.TryGetValue($"blk.{layer}.attn_k.weight", out var keyTensor))
                    {
                        long count = sliding ? scalarKvHeads : globalKvHeads;
                        if (count <= 0 || keyTensor.Shape.Length != 2 || keyTensor.Shape[0] != (ulong)hidden
                            || keyTensor.Shape[1] % (ulong)count != 0)
                            throw new NotSupportedException("Inconsistent Gemma K projection geometry.");
                        if (sliding) localDimension = checked((long)keyTensor.Shape[1] / count);
                        else globalDimension = checked((long)keyTensor.Shape[1] / count);
                        break;
                    }
            for (int layer = 0; layer < layers; layer++)
            {
                bool sliding = local?[layer] ?? false;
                long count = sliding ? scalarKvHeads : globalKvHeads;
                if (kvHeads != null && (sliding || !globalHeadsExplicit) && kvHeads[layer] != count)
                    throw new NotSupportedException("Gemma supports one KV-head count per local/global attention class.");
                if (local != null && layer >= layers - shared)
                    for (int donor = layers - shared - 1; donor >= 0; donor--)
                        if (local[donor] == sliding) { sharedLayers[layer] = true; break; }
                if (sharedLayers[layer]) continue; // physical owner is its earlier donor
                long dimension = sliding ? localDimension : globalDimension;
                if (count <= 0 || dimension <= 0) throw new NotSupportedException("Unknown Gemma KV geometry.");
                // F32 bounds both F16 and quantized KV on host and device.
                AddKv(checked(8 * count * dimension), sliding ? window : 0);
            }
        }
        else
        {
            long count = Int("attention.head_count_kv", checked((uint)heads));
            long key = Int("attention.key_length"), value = Int("attention.value_length");
            if (key == 0 && value == 0 && hidden % heads != 0)
                throw new NotSupportedException("Qwen requires an explicit head dimension when embedding width is not divisible by head count.");
            // Match ModelConfig.HeadDim and Qwen35.InitCaches: both K and V
            // allocate the same head dimension, even if the two metadata lengths
            // differ. A value-only checkpoint also uses that value for both.
            long headDimension = key > 0 ? key : value > 0 ? value : hidden / heads;
            attentionWidth = checked(heads * headDimension);
            int interval = checked((int)Int("full_attention_interval", 4));
            if (interval <= 0) throw new NotSupportedException("Invalid recurrent layer interval.");
            var layerTypes = file.GetStringArray($"{arch}.layer_types");
            if (file.Metadata.ContainsKey($"{arch}.layer_types") && (layerTypes == null || layerTypes.Length != layers
                || layerTypes.Any(t => !string.Equals(t, "linear_attention", StringComparison.OrdinalIgnoreCase)
                    && !string.Equals(t, "full_attention", StringComparison.OrdinalIgnoreCase))))
                throw new NotSupportedException("Unknown Qwen layer-types layout.");
            long inner = Int("ssm.inner_size"), state = Int("ssm.state_size"), groups = Int("ssm.group_count"), conv = Int("ssm.conv_kernel");
            long valueHeads = Int("ssm.time_step_rank");
            if (inner <= 0 || state <= 0 || groups <= 0 || conv <= 0 || valueHeads <= 0 || inner % valueHeads != 0
                || valueHeads % groups != 0 || count <= 0 || headDimension <= 0)
                throw new NotSupportedException("Unknown Qwen recurrent state geometry.");
            for (int layer = 0; layer < layers; layer++)
            {
                bool rec = layerTypes?.Length == layers
                    ? string.Equals(layerTypes[layer], "linear_attention", StringComparison.OrdinalIgnoreCase)
                    : (layer + 1) % interval != 0;
                recurrentLayers[layer] = rec;
                if (rec)
                    recurrent = checked(recurrent + 4 * checked(inner * state + conv * checked(inner + 2 * groups * state)));
                else AddKv(checked(8 * count * headDimension), 0);
            }
        }

        for (int layer = 0; layer < layers; layer++)
        {
            string p = $"blk.{layer}.";
            if (arch == "qwen35" && recurrentLayers[layer])
                Fusion([p + "attn_qkv.weight", p + "attn_gate.weight", p + "ssm_beta.weight", p + "ssm_alpha.weight"], keepSources: true);
            else if (!sharedLayers[layer])
            {
                string v = p + "attn_v.weight";
                if (arch == "gemma4" && !file.Tensors.ContainsKey(v)) v = p + "attn_k.weight";
                Fusion([p + "attn_q.weight", p + "attn_k.weight", v]);
            }
            Fusion([p + "ffn_gate.weight", p + "ffn_up.weight"]);
        }
        long fusion = checked((retainHostQuantizedWeights ? 0 : quantFusion) + floatFusionPeak + conversionPeak);
        if (retainHostQuantizedWeights) residentHost = checked(residentHost + quantFusion);
        long deviceWeights = checked(weights + extraDevice + padding);

        string? refusal = null;
        try
        {
            if (arch == "gemma4") Gemma4Model.ValidateStreamingWeightMetadata(arch, BackendType.GgmlCuda,
                1, 1, 0, 0, null!, activeTensors);
            else Qwen35Model.ValidateStreamingWeightMetadata(arch, BackendType.GgmlCuda,
                1, 1, 0, nextnLayers, null!, activeTensors);
        }
        catch (NotSupportedException ex) { refusal = ex.Message; }
        return new(new()
        {
            DenseWeights = new(weights, deviceWeights), Persistent = new(persistent, persistent),
            RecurrentStatePerSequence = new(recurrent, recurrent), KvCaches = kv
        }, fusion, checked(largest * 2 + (32L << 20)), hidden, intermediate, heads, vocab, refusal)
        {
            SourceWeightBytes = sourceWeights, ResidentHostWeightBytes = residentHost,
            RetainedMappedWeightBytes = mapped,
            AttentionProjectionWidth = attentionWidth,
            RetainedDecodeGraphLayers = arch == "qwen35" ? layers : 0,
            RetainsSlidingWindowPrefill = arch == "gemma4"
        };

        void Fusion(string[] names, bool keepSources = false)
        {
            if (names.Any(n => !file.Tensors.ContainsKey(n))) return;
            var tensors = names.Select(n => file.Tensors[n]).ToArray();
            if (tensors.Any(t => t.Shape.Length != 2 || t.Shape[0] != tensors[0].Shape[0])) return;
            bool raw = IsRawMatrix(tensors[0]);
            if (tensors.Any(t => IsRawMatrix(t) != raw)) return; // mixed F32/raw stays split
            // Qwen TryFuseWeights applies keepSources only to raw quantized
            // matrices. Its F32 branch always disposes the source tensors.
            bool retainSources = raw && keepSources;
            long source = names.Distinct().Sum(n => storedBytes[n]);
            long packed = names.Sum(n => storedBytes[n]);
            if (raw && tensors.Any(t => t.Type != tensors[0].Type))
            {
                // Both supported single-CUDA families keep mixed Gate/Up (and
                // mixed attention packs) in their original formats. No fused
                // allocation or load-time requantization scratch is needed.
                return;
            }
            largest = Math.Max(largest, packed);
            extraDevice = checked(extraDevice + (retainSources ? packed : Math.Max(0, packed - source)));
            if (raw) quantFusion = checked(quantFusion + packed);
            else
            {
                residentHost = checked(residentHost + Math.Max(0, packed - source));
                floatFusionPeak = Math.Max(floatFusionPeak, packed);
            }
        }

        void AddKv(long bytes, int window)
        {
            // Gemma allocates an entire local-attention ring even when the
            // admitted context is shorter. Rounding to W preserves that floor.
            int allocationBlock = window > 0 ? window : 256;
            kv.Add(new(bytes, WindowTokens: window, AllocationBlockTokens: allocationBlock));
            kv.Add(new(bytes, WindowTokens: window, AllocationBlockTokens: allocationBlock, Tier: MemoryTier.Host));
        }
    }

    private static bool IsRawMatrix(GgufTensorInfo tensor) => tensor.Shape.Length == 2 && tensor.Type != GgmlTensorType.F32;
}
