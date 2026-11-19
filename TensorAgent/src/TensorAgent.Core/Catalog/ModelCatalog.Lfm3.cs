// Copyright (c) Zhongkai Fu. All rights reserved.
// Licensed under the BSD-3-Clause license in the repository root.

namespace TensorAgent.Core.Catalog;

public static partial class ModelCatalog
{
    // Liquid LFM3 (general.architecture "lfm3"): 24 layers, of which 13 are short-convolution
    // layers (a fixed 6-column state, no K/V) and 11 are GQA attention with four KV heads of
    // 128 - six full-attention layers and five sliding-window layers (window 256). Read on
    // 2026-11-16 from the pinned GGUFs' metadata (attention.head_count_kv per layer,
    // sliding_window_pattern, key/value_length). The files are config/lfm3-1.7b.json's and
    // config/lfm3-omni-2b.json's: sizes and SHA-256 are the LFS objects at the pinned commits,
    // checked against the Hugging Face tree API the same day.
    //
    // KvBytesPerToken counts what grows with the context: the five sliding-window layers keep
    // rings of 768 cells (window 256 + a 512-row step, llama.cpp's iSWA sizing: 7.5 MB a
    // sequence at F16, StandardDecoderModel.SlidingRings.cs), so only the six full layers grow,
    // 6 x 4 x 128 x 2 x 2 = 12,288 bytes a token (22,528 with linear caches on all eleven).
    private const long Lfm3KvBytesPerToken = 6 * 4 * 128 * 2 * 2;
    private const string Lfm3License = "LFM Open License v1.0";

    private static CatalogModel Lfm3Text() => new()
    {
        Id = "lfm3-1.7b-q4km",
        DisplayName = "LFM3 1.7B",
        Family = CatalogFamily.Lfm3,
        Kind = CatalogArchitectureKind.Dense,
        Parameters = "1.7B",
        Quantization = "Q4_K_M",
        Files = new[]
        {
            new CatalogFile(CatalogFileRole.Weights, "LFM3-1.7B-Q4_K_M.gguf",
                "https://huggingface.co/LiquidAI/LFM3-1.7B-GGUF/resolve/cd0c74c575b5c416f3011b5b4c298077f26ca8c2/LFM3-1.7B-Q4_K_M.gguf",
                1_099_627_808, "3793e1e6e2266919d6e015287221b41100175a98eddaba92e111aed3b05c93c4"),
        },
        Modalities = CatalogModalities.Text,
        // The phone tier: a 1.1 GB mapped file and 0.4 GB of F16 K/V at the full window
        // (charged twice on Metal) sit far inside a 12 GB phone's ~8.5 GB budget.
        MinDeviceMemoryGB = 12,
        // A reasoning model: its traces need room (see qwen3.5-9b-iq4xs), and at 12 KiB a
        // token the window is cheap. Desktop window 32,768.
        ContextLength = 32768,
        KvBytesPerToken = Lfm3KvBytesPerToken,
        KvCacheDtype = "f16",
        // The model card's general-purpose preset (also the GGUF's general.sampling):
        // temperature 0.1, top_k 50, repetition penalty 1.1 (the engine's default).
        Sampling = new CatalogSampling(0.1f, 50, 1.0f, 0.0f),
        SupportsThinking = true,
        Experimental = true,
        License = Lfm3License,
        Notes = "catalog.lfm3.notes",
    };

    private static CatalogModel Lfm3Omni() => new()
    {
        Id = "lfm3-omni-2b-q4km",
        DisplayName = "LFM3-Omni 2B",
        Family = CatalogFamily.Lfm3,
        Kind = CatalogArchitectureKind.Dense,
        Parameters = "2B",
        Quantization = "Q4_K_M",
        Files = new[]
        {
            new CatalogFile(CatalogFileRole.Weights, "LFM3-Omni-2B-Q4_K_M.gguf",
                "https://huggingface.co/LiquidAI/LFM3-Omni-2B-GGUF/resolve/d508b88f08a5b4286f2d2805a6a8d69a0ff44c72/LFM3-Omni-2B-Q4_K_M.gguf",
                1_099_630_080, "9f8d30e9d4fd7a9349d586e077988229b3ca0a0e4ad730897563d1eccc9893e2"),
            // One projector carries the image/video tower, the audio encoder and the speech
            // generator. Required: taking images, video and sound in is what the entry is for.
            new CatalogFile(CatalogFileRole.Projector, "mmproj-LFM3-Omni-2B-Q8_0.gguf",
                "https://huggingface.co/LiquidAI/LFM3-Omni-2B-GGUF/resolve/d508b88f08a5b4286f2d2805a6a8d69a0ff44c72/mmproj-LFM3-Omni-2B-Q8_0.gguf",
                506_842_528, "8a25f9d1931244429de404fccacfb7fb933970a36aa5688d2f5069f29c6b27b1"),
        },
        // Spoken replies come from the projector's speech decoder (docs/models/lfm3.md#speech-output):
        // the chat page offers its Speak replies switch when the loaded model reports speechOutput.
        // The entry still lists input modalities only, because AudioOutput means "a video generator
        // whose clips have sound" to the Models page (ModelBrowser.Capabilities: it also drops the
        // "chat" keyword). A distinct speech capability would be a separate Models page change.
        Modalities = CatalogModalities.Image | CatalogModalities.Video | CatalogModalities.Audio,
        MinDeviceMemoryGB = 12,
        // A video costs up to 12,288 tokens (two frames per pair, frames at 2 fps), an
        // image up to 2,048. Desktop window 32,768.
        ContextLength = 32768,
        KvBytesPerToken = Lfm3KvBytesPerToken,
        KvCacheDtype = "f16",
        // Greedy with no repetition penalty, the publisher's generation_config for text.
        Sampling = new CatalogSampling(0.0f, 1, 1.0f, 0.0f) { RepetitionPenalty = 1.0f },
        SupportsThinking = false,
        Experimental = true,
        License = Lfm3License,
        Notes = "catalog.lfm3Omni.notes",
    };
}
