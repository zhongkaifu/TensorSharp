// Copyright (c) Zhongkai Fu. All rights reserved.
// Licensed under the BSD-3-Clause license. See LICENSE in the repo root.

namespace TensorSharp.TestMatrix.Matrix;

/// <summary>
/// One environment variable whose on/off state the matrix exercises.
/// The <see cref="AppliesTo"/> predicate decides whether the var is
/// meaningful for a given (model, backend, feature) combination — flags
/// outside that set are not swept.
/// </summary>
public sealed record EnvVarSpec(
    string Name,
    string Category,
    IReadOnlyList<string> Values,
    string DefaultValue,
    string Notes,
    Func<ModelSpec, BackendInfo, FeatureSpec, bool> AppliesTo);

public static class EnvVarMatrix
{
    private static readonly string[] BoolValues = { "0", "1" };

    public static readonly IReadOnlyList<EnvVarSpec> All = new EnvVarSpec[]
    {
        // Continuous batching / batched forward
        new(
            Name: "TS_GPTOSS_BATCHED",
            Category: "BatchedForward",
            Values: BoolValues,
            DefaultValue: "0",
            Notes: "GPT OSS batched forward (opt-in).",
            AppliesTo: (m, b, f) =>
                string.Equals(m.Family, "gptoss", StringComparison.OrdinalIgnoreCase)
                && f.Kind is FeatureKind.Text or FeatureKind.UploadedText or FeatureKind.MultiTurn or FeatureKind.SyntheticDecode),

        new(
            Name: "TS_QWEN35_BATCHED",
            Category: "BatchedForward",
            Values: BoolValues,
            DefaultValue: "1",
            Notes: "Qwen 3.5 / 3.6 batched paged-attention forward. Default ON; set to 0 to force the per-seq KV-swap fallback.",
            AppliesTo: (m, b, f) =>
                m.Family.StartsWith("qwen35", StringComparison.OrdinalIgnoreCase)
                || string.Equals(m.Family, "qwen3next", StringComparison.OrdinalIgnoreCase)),

        new(
            Name: "TS_QWEN35_BATCHED_GDN_NATIVE",
            Category: "BatchedForward",
            Values: BoolValues,
            DefaultValue: "0",
            Notes: "Native batched GatedDeltaNet kernel for Qwen 3.5 / 3.6.",
            AppliesTo: (m, b, f) =>
                m.Family.StartsWith("qwen35", StringComparison.OrdinalIgnoreCase)
                || string.Equals(m.Family, "qwen3next", StringComparison.OrdinalIgnoreCase)),

        new(
            Name: "TS_NEMOTRON_BATCHED",
            Category: "BatchedForward",
            Values: BoolValues,
            DefaultValue: "0",
            Notes: "Nemotron-H batched forward (opt-in).",
            AppliesTo: (m, b, f) => m.Family.StartsWith("nemotron", StringComparison.OrdinalIgnoreCase)),

        new(
            Name: "TS_GEMMA4_BATCHED",
            Category: "BatchedForward",
            Values: BoolValues,
            DefaultValue: "1",
            Notes: "Gemma 4 batched forward (default ON, set 0 to opt out).",
            AppliesTo: (m, b, f) => string.Equals(m.Family, "gemma4", StringComparison.OrdinalIgnoreCase)),

        new(
            Name: "TS_NEMOTRON_MAMBA2_BATCHED_NATIVE",
            Category: "BatchedForward",
            Values: BoolValues,
            DefaultValue: "0",
            Notes: "Native Mamba2 batched step kernel for Nemotron-H.",
            AppliesTo: (m, b, f) => m.Family.StartsWith("nemotron", StringComparison.OrdinalIgnoreCase)),

        new(
            Name: "TS_BATCHED_N1_FAST_PATH",
            Category: "BatchedForward",
            Values: BoolValues,
            DefaultValue: "0",
            Notes: "Use the N=1 fast path through the batched scheduler.",
            AppliesTo: (m, b, f) => true),

        new(
            Name: "TS_SCHED_DISABLE_BATCHED",
            Category: "BatchedForward",
            Values: BoolValues,
            DefaultValue: "0",
            Notes: "Fall through to per-seq KV-swap (legacy scheduler).",
            AppliesTo: (m, b, f) => true),

        // KV cache
        new(
            Name: "KV_CACHE_DTYPE",
            Category: "KvCache",
            Values: new[] { "f32", "f16", "q8_0" },
            DefaultValue: "f32",
            Notes: "Precision of the KV cache.",
            AppliesTo: (m, b, f) => true),

        new(
            Name: "TS_KV_PAGED_QUANT_BITS",
            Category: "KvCache",
            Values: new[] { "0", "4", "8" },
            DefaultValue: "0",
            Notes: "Paged-KV TurboQuant codec bits (0 = off).",
            AppliesTo: (m, b, f) => true),

        new(
            Name: "MAX_CONTEXT",
            Category: "Context",
            Values: new[] { "4096", "8192", "16384" },
            DefaultValue: "(model default)",
            Notes: "Hard cap on context length.",
            AppliesTo: (m, b, f) => f.Id is "long_text" or "uploaded_text"),

        // Prefill / decode
        new(
            Name: "TS_PREFILL_CHUNK",
            Category: "Prefill",
            Values: new[] { "256", "512", "1024" },
            DefaultValue: "(arch default)",
            Notes: "Chunked prefill block size (GPT OSS, Qwen 3.5 / 3.6).",
            AppliesTo: (m, b, f) =>
                (string.Equals(m.Family, "gptoss", StringComparison.OrdinalIgnoreCase)
                 || m.Family.StartsWith("qwen35", StringComparison.OrdinalIgnoreCase)
                 || string.Equals(m.Family, "qwen3next", StringComparison.OrdinalIgnoreCase))
                && f.Kind is FeatureKind.UploadedText or FeatureKind.Text or FeatureKind.SyntheticPrefill
                && f.Id is "long_text" or "uploaded_text" or "pp2048"),

        new(
            Name: "GDN_DISABLE_CHUNKED_PREFILL",
            Category: "Prefill",
            Values: BoolValues,
            DefaultValue: "0",
            Notes: "Disable GDN chunked prefill (Qwen3next).",
            AppliesTo: (m, b, f) => string.Equals(m.Family, "qwen3next", StringComparison.OrdinalIgnoreCase)),

        new(
            Name: "TS_GGML_ASYNC_COMPUTE",
            Category: "Prefill",
            Values: BoolValues,
            DefaultValue: "0",
            Notes: "Async compute submission on GGML backends.",
            AppliesTo: (m, b, f) => b.Id.StartsWith("ggml", StringComparison.OrdinalIgnoreCase)),

        // Multimodal
        new(
            Name: "VIDEO_SAMPLE_FPS",
            Category: "Multimodal",
            Values: new[] { "1", "2" },
            DefaultValue: "1",
            Notes: "Frames sampled per second of video (time-based extraction).",
            AppliesTo: (m, b, f) => f.Kind == FeatureKind.Video),

        new(
            Name: "VIDEO_MAX_FRAMES",
            Category: "Multimodal",
            Values: new[] { "8", "16" },
            DefaultValue: "<unset> (no cap)",
            Notes: "Optional upper bound on sampled frames per video; unset/0 = no cap (pure time-based).",
            AppliesTo: (m, b, f) => f.Kind == FeatureKind.Video),

        new(
            Name: "TS_NEMOTRON_IMAGE_MAX_TILES",
            Category: "Multimodal",
            Values: new[] { "4", "8", "12" },
            DefaultValue: "(arch default)",
            Notes: "Max image tiles for Nemotron-H Omni.",
            AppliesTo: (m, b, f) => m.Family.StartsWith("nemotron", StringComparison.OrdinalIgnoreCase)
                                    && f.Kind == FeatureKind.Image),

        // MLX
        new(
            Name: "TS_MLX_BATCHED_MOE_DECODE",
            Category: "MLX",
            Values: BoolValues,
            DefaultValue: "0",
            Notes: "MoE decode kernel on MLX (Qwen 3.5 / 3.6).",
            AppliesTo: (m, b, f) => b.Id == "mlx" && m.Family.StartsWith("qwen35", StringComparison.OrdinalIgnoreCase)),

        new(
            Name: "TS_MLX_DEVICE_ROUTER",
            Category: "MLX",
            Values: BoolValues,
            DefaultValue: "0",
            Notes: "Run MoE router on-device on MLX.",
            AppliesTo: (m, b, f) => b.Id == "mlx" && m.Family.StartsWith("qwen35", StringComparison.OrdinalIgnoreCase)),

        new(
            Name: "TS_MLX_PIPELINED_DECODE",
            Category: "MLX",
            Values: BoolValues,
            DefaultValue: "1",
            Notes: "Pipelined greedy decode on MLX (device-side argmax + next-embedding lookup). Default ON for greedy decode; set 0 to fall back to per-token host sync.",
            AppliesTo: (m, b, f) => b.Id == "mlx" && f.Kind != FeatureKind.SyntheticPrefill),

        new(
            Name: "TS_MLX_DEVICE_KV_COPY",
            Category: "MLX",
            Values: BoolValues,
            DefaultValue: "1",
            Notes: "On-device KV scatter on MLX.",
            AppliesTo: (m, b, f) => b.Id == "mlx"),

        new(
            Name: "TS_MLX_QWEN35_GDN_PACKED_KERNELS",
            Category: "MLX",
            Values: BoolValues,
            DefaultValue: "0",
            Notes: "Packed GDN kernels on MLX (Qwen 3.5 / 3.6).",
            AppliesTo: (m, b, f) => b.Id == "mlx"
                                    && (m.Family.StartsWith("qwen35", StringComparison.OrdinalIgnoreCase)
                                        || string.Equals(m.Family, "qwen3next", StringComparison.OrdinalIgnoreCase))),
    };

    public static EnvVarSpec? FindByName(string name)
    {
        foreach (EnvVarSpec v in All)
        {
            if (string.Equals(v.Name, name, StringComparison.Ordinal))
            {
                return v;
            }
        }
        return null;
    }
}
