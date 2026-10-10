// Copyright (c) Zhongkai Fu. All rights reserved.
// Licensed under the BSD-3-Clause license in the repository root.

namespace TensorAgent.Core.Catalog;

public static partial class ModelCatalog
{
    // Publisher file sizes and LFS SHA-256 identities checked on 2026-10-07.
    // Resolve URLs pin immutable revisions. These metadata checks are not model/device
    // inference validation: all newly exposed entries remain explicitly experimental.
    // Memory tiers are conservative capacity estimates, without unmeasured disk-paging
    // deductions; actual backend support and generation memory depend on the model card.
    // The layer counts, KV heads and head dims behind each KvBytesPerToken were read on
    // 2026-10-08 from the metadata of the pinned GGUF files themselves (block_count,
    // attention.head_count_kv, key/value_length, sliding_window, kv_lora_rank).
    private static IEnumerable<CatalogModel> ExtendedModels()
    {
        yield return new CatalogModel
        {
            Id = "gemma-4-26b-a4b-qat-q4kxl",
            DisplayName = "Gemma 4 26B-A4B QAT",
            Family = CatalogFamily.Gemma4,
            Kind = CatalogArchitectureKind.MixtureOfExperts,
            Parameters = "catalog.extended.26b4b.parameters",
            Quantization = "UD-Q4_K_XL",
            Files = new[]
            {
                new CatalogFile(CatalogFileRole.Weights, "gemma-4-26B-A4B-it-qat-UD-Q4_K_XL.gguf",
                    "https://huggingface.co/unsloth/gemma-4-26B-A4B-it-qat-GGUF/resolve/7b92b5b28818151e8669af2e45e88d6086f490dd/gemma-4-26B-A4B-it-qat-UD-Q4_K_XL.gguf",
                    14_249_047_104, "a7c5bc715f5ff8e99a3e8901ce7d2b42b402c669bf24f7c5250747633d0f5891"),
                new CatalogFile(CatalogFileRole.Projector, "mmproj-BF16.gguf",
                    "https://huggingface.co/unsloth/gemma-4-26B-A4B-it-qat-GGUF/resolve/7b92b5b28818151e8669af2e45e88d6086f490dd/mmproj-BF16.gguf",
                    1_194_828_256, "7b06953ccdbe8cf363f47841a7afaacd2b1c2ff9a8d6b426fdec7521a6878744", Optional: true),
                new CatalogFile(CatalogFileRole.Draft, "mtp-gemma-4-26B-A4B-it.gguf",
                    "https://huggingface.co/unsloth/gemma-4-26B-A4B-it-qat-GGUF/resolve/7b92b5b28818151e8669af2e45e88d6086f490dd/mtp-gemma-4-26B-A4B-it.gguf",
                    251_939_328, "7272d97595f0d4c74bd7b623492b7dbdaafd8b7c72f329a8270ba4eca68f768a", Optional: true),
            },
            Modalities = CatalogModalities.Image | CatalogModalities.Video,
            MinDeviceMemoryGB = 32,
            ContextLength = 8192,
            // 30 layers, one in six global with two KV heads of 512; the other 25 keep
            // 1,024-row sliding rings, a fixed 200 MiB: 5 x 2 x 512 x 2 x 2 bytes.
            // Desktop window 32,768.
            KvBytesPerToken = 5 * 2 * 512 * 2 * 2,
            KvCacheDtype = "f16",
            LeanCaches = true,
            Sampling = new CatalogSampling(1.0f, 64, 0.95f, 0.0f),
            SupportsThinking = true,
            Experimental = true,
            License = GemmaLicense,
            Notes = "catalog.extended.gemmaLarge.notes",
        };

        yield return new CatalogModel
        {
            Id = "gemma-4-31b-q4-0",
            DisplayName = "Gemma 4 31B",
            Family = CatalogFamily.Gemma4,
            Kind = CatalogArchitectureKind.Dense,
            Parameters = "31B",
            Quantization = "Q4_0",
            Files = new[]
            {
                new CatalogFile(CatalogFileRole.Weights, "gemma-4-31B-it-Q4_0.gguf",
                    "https://huggingface.co/ggml-org/gemma-4-31B-it-GGUF/resolve/4fa4fdf38bee237b5c9e8a5b4e72cf39404c9dcc/gemma-4-31B-it-Q4_0.gguf",
                    17_992_313_088, "031dc1c5fa9c5a0abbf3c39c5173fb2af65f5ac2dc2a090268561d3c72dcd834"),
                new CatalogFile(CatalogFileRole.Projector, "mmproj-gemma-4-31B-it-Q8_0.gguf",
                    "https://huggingface.co/ggml-org/gemma-4-31B-it-GGUF/resolve/4fa4fdf38bee237b5c9e8a5b4e72cf39404c9dcc/mmproj-gemma-4-31B-it-Q8_0.gguf",
                    809_541_728, "8872f1dd7ba6a750a039c04b45812511f4ecf004e229420cea58c8049a970fb6", Optional: true),
                new CatalogFile(CatalogFileRole.Draft, "mtp-gemma-4-31B-it-Q8_0.gguf",
                    "https://huggingface.co/ggml-org/gemma-4-31B-it-GGUF/resolve/4fa4fdf38bee237b5c9e8a5b4e72cf39404c9dcc/mtp-gemma-4-31B-it-Q8_0.gguf",
                    514_687_104, "6b52ab20af503aee320dc09e93f886133b18d89ffc9075c7d9dcaf681e20b375", Optional: true),
            },
            Modalities = CatalogModalities.Image | CatalogModalities.Video,
            MinDeviceMemoryGB = 32,
            ContextLength = 8192,
            // 60 layers, one in six global with four KV heads of 512 (the sliding layers'
            // 1,024-row rings are a fixed 800 MiB): 10 x 4 x 512 x 2 x 2 bytes = 80 KiB.
            // Beside 18 GB of weights the 32 GB tier affords 20,480 on a desktop.
            KvBytesPerToken = 10 * 4 * 512 * 2 * 2,
            KvCacheDtype = "f16",
            LeanCaches = true,
            Sampling = new CatalogSampling(1.0f, 64, 0.95f, 0.0f),
            SupportsThinking = true,
            Experimental = true,
            License = GemmaLicense,
            Notes = "catalog.extended.gemmaLarge.notes",
        };

        yield return new CatalogModel
        {
            Id = "qwen3.5-35b-a3b-q4km",
            DisplayName = "Qwen3.5 35B-A3B",
            Family = CatalogFamily.Qwen35,
            Kind = CatalogArchitectureKind.MixtureOfExperts,
            Parameters = "catalog.extended.35b3b.parameters",
            Quantization = "Q4_K_M",
            Files = new[]
            {
                new CatalogFile(CatalogFileRole.Weights, "Qwen3.5-35B-A3B-Q4_K_M.gguf",
                    "https://huggingface.co/unsloth/Qwen3.5-35B-A3B-GGUF/resolve/bc014a17be43adabd7066b7a86075ff935c6a4e2/Qwen3.5-35B-A3B-Q4_K_M.gguf",
                    22_016_023_168, "3b46d1066bc91cc2d613e3bc22ce691dd77e6f0d33c9060690d24ce6de494375"),
                new CatalogFile(CatalogFileRole.Projector, "mmproj-F16.gguf",
                    "https://huggingface.co/unsloth/Qwen3.5-35B-A3B-GGUF/resolve/bc014a17be43adabd7066b7a86075ff935c6a4e2/mmproj-F16.gguf",
                    899_283_648, "a516ab92e8240da4734d68352bdfba84c16e830ee40010b8fac80d69c77272ff", Optional: true),
            },
            Modalities = CatalogModalities.Image,
            MinDeviceMemoryGB = 48,
            ContextLength = 8192,
            // qwen35moe: 40 layers, every fourth full attention (two KV heads of 256), the
            // rest linear attention with a fixed state: 10 x 2 x 256 x 2 x 2 bytes.
            // Desktop window 32,768.
            KvBytesPerToken = 10 * 2 * 256 * 2 * 2,
            KvCacheDtype = "q8_0",
            LeanCaches = true,
            Sampling = new CatalogSampling(0.7f, 20, 0.8f, 0.0f),
            SupportsThinking = true,
            Experimental = true,
            License = "Apache-2.0",
            Notes = "catalog.extended.qwen35Large.notes",
        };

        yield return new CatalogModel
        {
            Id = "qwen3.6-35b-a3b-q4km",
            DisplayName = "Qwen3.6 35B-A3B",
            Family = CatalogFamily.Qwen36,
            Kind = CatalogArchitectureKind.MixtureOfExperts,
            Parameters = "catalog.extended.35b3b.parameters",
            Quantization = "UD-Q4_K_M",
            Files = new[]
            {
                new CatalogFile(CatalogFileRole.Weights, "Qwen3.6-35B-A3B-UD-Q4_K_M.gguf",
                    "https://huggingface.co/unsloth/Qwen3.6-35B-A3B-MTP-GGUF/resolve/5bc3e238d916f48a861bac2f8a1990a0e9b7e98d/Qwen3.6-35B-A3B-UD-Q4_K_M.gguf",
                    22_663_387_424, "0b21525e972670ed59e1812e170b27c26355381f0656ecc4e25617ece7dac58b"),
                new CatalogFile(CatalogFileRole.Projector, "mmproj-F16.gguf",
                    "https://huggingface.co/unsloth/Qwen3.6-35B-A3B-MTP-GGUF/resolve/5bc3e238d916f48a861bac2f8a1990a0e9b7e98d/mmproj-F16.gguf",
                    899_283_584, "71f3cbc1f7cc0f30d09d41cfa924c0060827ebc33bf15ace7e86661e856f0160", Optional: true),
            },
            Modalities = CatalogModalities.Image,
            MinDeviceMemoryGB = 48,
            ContextLength = 8192,
            // qwen35moe: 40 layers and the MTP block; every fourth layer and the MTP block
            // are full attention (two KV heads of 256): 11 x 2 x 256 x 2 x 2 bytes.
            // Desktop window 32,768.
            KvBytesPerToken = 11 * 2 * 256 * 2 * 2,
            KvCacheDtype = "q8_0",
            LeanCaches = true,
            Sampling = new CatalogSampling(0.7f, 20, 0.8f, 0.0f),
            SupportsThinking = true,
            Experimental = true,
            License = "Apache-2.0",
            Notes = "catalog.extended.qwen36Large.notes",
        };

        yield return new CatalogModel
        {
            Id = "qwen3.6-27b-q4km",
            DisplayName = "Qwen3.6 27B",
            Family = CatalogFamily.Qwen36,
            Kind = CatalogArchitectureKind.Dense,
            Parameters = "27B",
            Quantization = "Q4_K_M",
            Files = new[]
            {
                new CatalogFile(CatalogFileRole.Weights, "Qwen3.6-27B-Q4_K_M.gguf",
                    "https://huggingface.co/unsloth/Qwen3.6-27B-MTP-GGUF/resolve/5cb35eb3dcbf52dbce5f87dbc64df6aaffadcace/Qwen3.6-27B-Q4_K_M.gguf",
                    17_106_773_120, "a7cbd3ecc0e3f9b333edee61ae66bc87ed713c5d49587a8355814722ed329e0f"),
                new CatalogFile(CatalogFileRole.Projector, "mmproj-BF16.gguf",
                    "https://huggingface.co/unsloth/Qwen3.6-27B-MTP-GGUF/resolve/5cb35eb3dcbf52dbce5f87dbc64df6aaffadcace/mmproj-BF16.gguf",
                    931_146_304, "05353347512982ee62317b9d8c89372bc815f4b4043580e7ef3ad411ec1a1cd3", Optional: true),
            },
            Modalities = CatalogModalities.Image,
            MinDeviceMemoryGB = 32,
            ContextLength = 8192,
            // qwen35: 64 layers and the MTP block; every fourth layer and the MTP block are
            // full attention (four KV heads of 256): 17 x 4 x 256 x 2 x 2 bytes = 68 KiB at
            // f16, half that at the entry's q8_0. At f16 the 32 GB tier affords 28,672.
            KvBytesPerToken = 17 * 4 * 256 * 2 * 2,
            KvCacheDtype = "q8_0",
            LeanCaches = true,
            Sampling = new CatalogSampling(0.7f, 20, 0.8f, 0.0f),
            SupportsThinking = true,
            Experimental = true,
            License = "Apache-2.0",
            Notes = "catalog.extended.qwen36Large.notes",
        };

        yield return new CatalogModel
        {
            Id = "gpt-oss-20b-mxfp4",
            DisplayName = "GPT OSS 20B",
            Family = CatalogFamily.GptOss,
            Kind = CatalogArchitectureKind.MixtureOfExperts,
            Parameters = "20B",
            Quantization = "MXFP4",
            Files = new[]
            {
                new CatalogFile(CatalogFileRole.Weights, "gpt-oss-20b-MXFP4.gguf",
                    "https://huggingface.co/ggml-org/gpt-oss-20b-GGUF/resolve/ef9b12f2ff56c69cf32153a02784e7a3c88bf524/gpt-oss-20b-MXFP4.gguf",
                    12_109_566_624, "27cd6c432c7672cb812a92f611cf3ba7bbc35928262bb1e1253ff4ee6ae35901"),
            },
            Modalities = CatalogModalities.Text,
            MinDeviceMemoryGB = 24,
            ContextLength = 8192,
            // 24 layers with eight KV heads of 64. Half are 128-token sliding layers, but the
            // engine sizes every layer's cache to the whole context (GptOssModel.InitKVCache),
            // so all count: 24 x 8 x 64 x 2 x 2 bytes = 48 KiB. Desktop window 32,768.
            KvBytesPerToken = 24 * 8 * 64 * 2 * 2,
            KvCacheDtype = "f16",
            LeanCaches = true,
            Sampling = new CatalogSampling(0.7f, 40, 0.95f, 0.0f),
            SupportsThinking = true,
            Experimental = true,
            License = "Apache-2.0",
            Notes = "catalog.extended.gptOss.notes",
        };

        yield return new CatalogModel
        {
            Id = "nemotron-h-8b-q4km",
            DisplayName = "Nemotron-H 8B",
            Family = CatalogFamily.Nemotron,
            Kind = CatalogArchitectureKind.Dense,
            Parameters = "8B",
            Quantization = "Q4_K_M",
            Files = new[]
            {
                new CatalogFile(CatalogFileRole.Weights, "nvidia_Nemotron-H-8B-Reasoning-128K-Q4_K_M.gguf",
                    "https://huggingface.co/bartowski/nvidia_Nemotron-H-8B-Reasoning-128K-GGUF/resolve/cf45b6ff44dcd5c2105a23abeb9e77797f1a4014/nvidia_Nemotron-H-8B-Reasoning-128K-Q4_K_M.gguf",
                    4_983_360_480, "312ebd50999058707868f49011d87225fcc92af0f5a526cd3553f821dc161956"),
            },
            Modalities = CatalogModalities.Text,
            MinDeviceMemoryGB = 24,
            ContextLength = 8192,
            // 52 layers, of which only 7, 18, 29 and 40 are attention (eight KV heads of
            // 128); the rest are Mamba2 or FFN with a fixed state: 4 x 8 x 128 x 2 x 2
            // bytes = 16 KiB. Desktop window 32,768.
            KvBytesPerToken = 4 * 8 * 128 * 2 * 2,
            KvCacheDtype = "f16",
            LeanCaches = true,
            Sampling = new CatalogSampling(0.7f, 40, 0.95f, 0.0f),
            SupportsThinking = true,
            Experimental = true,
            License = "NVIDIA Open Model License",
            Notes = "catalog.extended.nemotron.notes",
        };

        yield return new CatalogModel
        {
            Id = "nemotron-h-47b-q4km",
            DisplayName = "Nemotron-H 47B",
            Family = CatalogFamily.Nemotron,
            Kind = CatalogArchitectureKind.Dense,
            Parameters = "47B",
            Quantization = "Q4_K_M",
            Files = new[]
            {
                new CatalogFile(CatalogFileRole.Weights, "nvidia_Nemotron-H-47B-Reasoning-128K-Q4_K_M.gguf",
                    "https://huggingface.co/bartowski/nvidia_Nemotron-H-47B-Reasoning-128K-GGUF/resolve/ce89f02cd0eafe3931acec38f4d91a654ae1ca4d/nvidia_Nemotron-H-47B-Reasoning-128K-Q4_K_M.gguf",
                    28_187_710_688, "3e9217e00201528d8d0c5bea3405b095a120f7e5cd920b34e3c7a961f6e01d71"),
            },
            Modalities = CatalogModalities.Text,
            MinDeviceMemoryGB = 48,
            ContextLength = 8192,
            // 98 layers, five of them attention (eight KV heads of 128), the rest Mamba2 or
            // FFN: 5 x 8 x 128 x 2 x 2 bytes. Desktop window 32,768.
            KvBytesPerToken = 5 * 8 * 128 * 2 * 2,
            KvCacheDtype = "f16",
            LeanCaches = true,
            Sampling = new CatalogSampling(0.7f, 40, 0.95f, 0.0f),
            SupportsThinking = true,
            Experimental = true,
            License = "NVIDIA Open Model License",
            Notes = "catalog.extended.nemotron.notes",
        };

        yield return new CatalogModel
        {
            Id = "nemotron-3-nano-omni-30b-a3b-q4kxl",
            DisplayName = "Nemotron 3 Nano Omni 30B-A3B",
            Family = CatalogFamily.Nemotron,
            Kind = CatalogArchitectureKind.MixtureOfExperts,
            Parameters = "catalog.extended.30b3b.parameters",
            Quantization = "UD-Q4_K_XL",
            Files = new[]
            {
                new CatalogFile(CatalogFileRole.Weights, "NVIDIA-Nemotron-3-Nano-Omni-30B-A3B-Reasoning-UD-Q4_K_XL.gguf",
                    "https://huggingface.co/unsloth/NVIDIA-Nemotron-3-Nano-Omni-30B-A3B-Reasoning-GGUF/resolve/571758804835f56154718683f5c0e388b7d0fef9/NVIDIA-Nemotron-3-Nano-Omni-30B-A3B-Reasoning-UD-Q4_K_XL.gguf",
                    23_926_934_976, "82eb2c0a47383fc0d59d06faf0b60ad0ff0d566b8d403047525e751824e077ea"),
                new CatalogFile(CatalogFileRole.Projector, "mmproj-BF16.gguf",
                    "https://huggingface.co/unsloth/NVIDIA-Nemotron-3-Nano-Omni-30B-A3B-Reasoning-GGUF/resolve/571758804835f56154718683f5c0e388b7d0fef9/mmproj-BF16.gguf",
                    1_589_506_304, "649fd5abd406c34b90942ca8616f4930953d8061083ff9e48172839e1f75313c", Optional: true),
            },
            Modalities = CatalogModalities.Image,
            MinDeviceMemoryGB = 48,
            ContextLength = 8192,
            // 52 layers, six attention (two KV heads of 128), the rest Mamba2 or MoE:
            // 6 x 2 x 128 x 2 x 2 bytes. Desktop window 32,768.
            KvBytesPerToken = 6 * 2 * 128 * 2 * 2,
            KvCacheDtype = "f16",
            LeanCaches = true,
            Sampling = new CatalogSampling(0.7f, 40, 0.95f, 0.0f),
            SupportsThinking = true,
            Experimental = true,
            License = "NVIDIA Nemotron Open Model License",
            Notes = "catalog.extended.nemotronOmni.notes",
        };

        yield return new CatalogModel
        {
            Id = "nemotron-3.5-lightning-30b-a3b-mxfp4",
            DisplayName = "Nemotron 3.5 Lightning 30B-A3B",
            Family = CatalogFamily.Nemotron,
            Kind = CatalogArchitectureKind.MixtureOfExperts,
            Parameters = "catalog.extended.30b3b.parameters",
            Quantization = "MXFP4_MOE",
            Files = new[]
            {
                new CatalogFile(CatalogFileRole.Weights, "NVIDIA-Nemotron-3.5-Lightning-30B-A3B-MXFP4_MOE.gguf",
                    "https://huggingface.co/unsloth/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-GGUF/resolve/f2d3fe3694501008786e81e5f20360cbf715496a/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-MXFP4_MOE.gguf",
                    23_210_817_600, "c5a1433ecfb206d63be963faaa81d7c1170cb13abb14c73390968fe7c79a99f6"),
            },
            Modalities = CatalogModalities.Text,
            MinDeviceMemoryGB = 48,
            ContextLength = 8192,
            // 53 layers, seven attention (two KV heads of 128), the rest Mamba2 or MoE:
            // 7 x 2 x 128 x 2 x 2 bytes. Desktop window 32,768.
            KvBytesPerToken = 7 * 2 * 128 * 2 * 2,
            KvCacheDtype = "f16",
            LeanCaches = true,
            Sampling = new CatalogSampling(0.7f, 40, 0.95f, 0.0f),
            SupportsThinking = true,
            Experimental = true,
            License = "OpenMDW-1.1",
            Notes = "catalog.extended.nemotron.notes",
        };

        yield return new CatalogModel
        {
            Id = "mistral-small-3.1-24b-q4km",
            DisplayName = "Mistral Small 3.1 24B",
            Family = CatalogFamily.Mistral3,
            Kind = CatalogArchitectureKind.Dense,
            Parameters = "24B",
            Quantization = "Q4_K_M",
            Files = new[]
            {
                new CatalogFile(CatalogFileRole.Weights, "mistralai_Mistral-Small-3.1-24B-Instruct-2503-Q4_K_M.gguf",
                    "https://huggingface.co/bartowski/mistralai_Mistral-Small-3.1-24B-Instruct-2503-GGUF/resolve/f73dfd9e812922fb503a993e3fa5671424f486d3/mistralai_Mistral-Small-3.1-24B-Instruct-2503-Q4_K_M.gguf",
                    14_333_910_496, "c5743c1bf39db0ae8a5ade5df0374b8e9e492754a199cfdad7ef393c1590f7c0"),
                new CatalogFile(CatalogFileRole.Projector, "mmproj-mistralai_Mistral-Small-3.1-24B-Instruct-2503-f16.gguf",
                    "https://huggingface.co/bartowski/mistralai_Mistral-Small-3.1-24B-Instruct-2503-GGUF/resolve/f73dfd9e812922fb503a993e3fa5671424f486d3/mmproj-mistralai_Mistral-Small-3.1-24B-Instruct-2503-f16.gguf",
                    878_054_144, "f5add93ad360ef6ccba571bba15e8b4bd4471f3577440a8b18785f8707d987ed", Optional: true),
            },
            Modalities = CatalogModalities.Image,
            MinDeviceMemoryGB = 32,
            ContextLength = 8192,
            // 40 full-attention layers with eight KV heads of 128: 40 x 8 x 128 x 2 x 2 bytes
            // = 160 KiB, the most per token of any chat entry. Beside 14.3 GB of weights
            // the 32 GB tier affords 16,384 on a desktop.
            KvBytesPerToken = 40 * 8 * 128 * 2 * 2,
            KvCacheDtype = "f16",
            LeanCaches = true,
            Sampling = new CatalogSampling(0.7f, 40, 0.95f, 0.0f),
            SupportsThinking = false,
            Experimental = true,
            License = "Apache-2.0",
            Notes = "catalog.extended.mistral.notes",
        };

        yield return new CatalogModel
        {
            Id = "hy-mt2-1.8b-q4km",
            DisplayName = "Hy-MT2 1.8B",
            Family = CatalogFamily.HunyuanDense,
            Kind = CatalogArchitectureKind.Dense,
            Parameters = "1.8B",
            Quantization = "Q4_K_M",
            Files = new[]
            {
                new CatalogFile(CatalogFileRole.Weights, "Hy-MT2-1.8B-Q4_K_M.gguf",
                    "https://huggingface.co/tencent/Hy-MT2-1.8B-GGUF/resolve/a0c709d9fac510f2c807aa3af52872340dc37a4a/Hy-MT2-1.8B-Q4_K_M.gguf",
                    1_133_080_448, "dc5f44fcf1fa496ee7ad725982c0c8c553a4de00259b53af84c4b89fb0c06699"),
            },
            Modalities = CatalogModalities.Text,
            MinDeviceMemoryGB = 24,
            ContextLength = 8192,
            // 32 full-attention layers with four KV heads of 128: 32 x 4 x 128 x 2 x 2 bytes.
            // Desktop window 32,768.
            KvBytesPerToken = 32 * 4 * 128 * 2 * 2,
            KvCacheDtype = "f16",
            LeanCaches = true,
            Sampling = new CatalogSampling(0.7f, 40, 0.95f, 0.0f),
            SupportsThinking = false,
            Experimental = true,
            License = "Apache-2.0",
            Notes = "catalog.extended.hunyuan.notes",
        };

        yield return new CatalogModel
        {
            Id = "deepseek-v4-flash-0731-q2kxl",
            DisplayName = "DeepSeek V4 Flash 0731",
            Family = CatalogFamily.DeepSeek4,
            Kind = CatalogArchitectureKind.MixtureOfExperts,
            Parameters = "284B",
            Quantization = "UD-Q2_K_XL",
            Files = new[]
            {
                new CatalogFile(CatalogFileRole.Weights, "DeepSeek-V4-Flash-0731-UD-Q2_K_XL-00001-of-00003.gguf",
                    "https://huggingface.co/unsloth/DeepSeek-V4-Flash-0731-GGUF/resolve/fbbb5b93fb787c21338159b0af3318bb3f4d9768/UD-Q2_K_XL/DeepSeek-V4-Flash-0731-UD-Q2_K_XL-00001-of-00003.gguf",
                    5_257_664, "de65c5bb660817b95cc281bf54935f9d1bcc3b22535f2eae63b35a14c7c5724b"),
                new CatalogFile(CatalogFileRole.WeightsShard, "DeepSeek-V4-Flash-0731-UD-Q2_K_XL-00002-of-00003.gguf",
                    "https://huggingface.co/unsloth/DeepSeek-V4-Flash-0731-GGUF/resolve/fbbb5b93fb787c21338159b0af3318bb3f4d9768/UD-Q2_K_XL/DeepSeek-V4-Flash-0731-UD-Q2_K_XL-00002-of-00003.gguf",
                    49_437_013_568, "c8728a298862cd8736782e2e4056e5e3e03c415ba552a38173264494e41ef327"),
                new CatalogFile(CatalogFileRole.WeightsShard, "DeepSeek-V4-Flash-0731-UD-Q2_K_XL-00003-of-00003.gguf",
                    "https://huggingface.co/unsloth/DeepSeek-V4-Flash-0731-GGUF/resolve/fbbb5b93fb787c21338159b0af3318bb3f4d9768/UD-Q2_K_XL/DeepSeek-V4-Flash-0731-UD-Q2_K_XL-00003-of-00003.gguf",
                    47_390_237_120, "a4bae0788c025c1b2388754592c98fefa1372981748ac9875039bf6a30e6eeb4"),
                new CatalogFile(CatalogFileRole.Draft, "DSpark-drafter-Q2K-Q8-0731.gguf",
                    "https://huggingface.co/bleysg/DeepSeek-V4-Flash-DSpark-drafter-GGUF/resolve/57fae967d0c5dac1f32a444ea8884ec46459247b/DSpark-drafter-Q2K-Q8-0731.gguf",
                    6_971_241_504, "8fa269560dc76fd73e4233ad9b1938b5f65dd363381fd9b1a5c6183f7d12d686", Optional: true),
            },
            Modalities = CatalogModalities.Text,
            MinDeviceMemoryGB = 192,
            ContextLength = 8192,
            // 43 layers, each a 512-wide latent row (K and V share it) and a 128-wide indexer
            // key. Most layers keep a quarter or 1/128 of the rows; counting every row of
            // every layer is an upper bound: 43 x (512 + 128) x 2 bytes. Desktop 32,768.
            KvBytesPerToken = 43 * (512 + 128) * 2,
            KvCacheDtype = "f16",
            LeanCaches = true,
            Sampling = new CatalogSampling(0.0f, 0, 1.0f, 0.0f),
            SupportsThinking = true,
            Experimental = true,
            License = "MIT",
            Notes = "catalog.extended.deepseek4.notes",
        };

        yield return new CatalogModel
        {
            Id = "deepseek-v4.1-flash-engramq5-q2k",
            DisplayName = "DeepSeek V4.1 Flash",
            Family = CatalogFamily.DeepSeek41,
            Kind = CatalogArchitectureKind.MixtureOfExperts,
            Parameters = "catalog.extended.384experts.parameters",
            Quantization = "Q2_K / Q5_K Engram",
            Files = new[]
            {
                new CatalogFile(CatalogFileRole.Weights, "DeepSeek-V4.1-Flash-EngramQ5-Q2_K-00001-of-00010.gguf",
                    "https://huggingface.co/smalinin/DeepSeek-V4.1-Flash-GGUF/resolve/d1de55c19f95172c882906cc83c0e55932d26a63/Q2_K-Q5/DeepSeek-V4.1-Flash-EngramQ5-Q2_K-00001-of-00010.gguf",
                    31_006_920_384, "8126b49dfcfde02cb3db24b6f98d56b031f0a34eaca2509fc7b2b9d362cf0ac8"),
                new CatalogFile(CatalogFileRole.WeightsShard, "DeepSeek-V4.1-Flash-EngramQ5-Q2_K-00002-of-00010.gguf",
                    "https://huggingface.co/smalinin/DeepSeek-V4.1-Flash-GGUF/resolve/d1de55c19f95172c882906cc83c0e55932d26a63/Q2_K-Q5/DeepSeek-V4.1-Flash-EngramQ5-Q2_K-00002-of-00010.gguf",
                    31_461_213_696, "22bb293aee509a348ce32a739e006fa41f2348c6bcfafa3be76a3ee079eadf96"),
                new CatalogFile(CatalogFileRole.WeightsShard, "DeepSeek-V4.1-Flash-EngramQ5-Q2_K-00003-of-00010.gguf",
                    "https://huggingface.co/smalinin/DeepSeek-V4.1-Flash-GGUF/resolve/d1de55c19f95172c882906cc83c0e55932d26a63/Q2_K-Q5/DeepSeek-V4.1-Flash-EngramQ5-Q2_K-00003-of-00010.gguf",
                    31_001_150_976, "db894848b4f14d42c39e18faa907c737cd4850f2fcb9deff9d5ebd86f580aaf7"),
                new CatalogFile(CatalogFileRole.WeightsShard, "DeepSeek-V4.1-Flash-EngramQ5-Q2_K-00004-of-00010.gguf",
                    "https://huggingface.co/smalinin/DeepSeek-V4.1-Flash-GGUF/resolve/d1de55c19f95172c882906cc83c0e55932d26a63/Q2_K-Q5/DeepSeek-V4.1-Flash-EngramQ5-Q2_K-00004-of-00010.gguf",
                    31_001_150_976, "655a3400f2c092d6e3c11b8b18bf319b29563e59b960337115266574321953e3"),
                new CatalogFile(CatalogFileRole.WeightsShard, "DeepSeek-V4.1-Flash-EngramQ5-Q2_K-00005-of-00010.gguf",
                    "https://huggingface.co/smalinin/DeepSeek-V4.1-Flash-GGUF/resolve/d1de55c19f95172c882906cc83c0e55932d26a63/Q2_K-Q5/DeepSeek-V4.1-Flash-EngramQ5-Q2_K-00005-of-00010.gguf",
                    31_461_213_696, "4b9378c6819d1130517e8719026b34e5257f1f73bd1d50cf6f30243982100bac"),
                new CatalogFile(CatalogFileRole.WeightsShard, "DeepSeek-V4.1-Flash-EngramQ5-Q2_K-00006-of-00010.gguf",
                    "https://huggingface.co/smalinin/DeepSeek-V4.1-Flash-GGUF/resolve/d1de55c19f95172c882906cc83c0e55932d26a63/Q2_K-Q5/DeepSeek-V4.1-Flash-EngramQ5-Q2_K-00006-of-00010.gguf",
                    31_001_150_976, "04f161084d82032c65247c02e6169784a757be7db9e068b0baba6833125b6bb8"),
                new CatalogFile(CatalogFileRole.WeightsShard, "DeepSeek-V4.1-Flash-EngramQ5-Q2_K-00007-of-00010.gguf",
                    "https://huggingface.co/smalinin/DeepSeek-V4.1-Flash-GGUF/resolve/d1de55c19f95172c882906cc83c0e55932d26a63/Q2_K-Q5/DeepSeek-V4.1-Flash-EngramQ5-Q2_K-00007-of-00010.gguf",
                    9_838_264_896, "b1b1bf3cfbbc7388ce49b5ced69c42a13c7c5c5d3d609ac86902897a08d3c88a"),
                new CatalogFile(CatalogFileRole.WeightsShard, "DeepSeek-V4.1-Flash-EngramQ5-Q2_K-00008-of-00010.gguf",
                    "https://huggingface.co/smalinin/DeepSeek-V4.1-Flash-GGUF/resolve/d1de55c19f95172c882906cc83c0e55932d26a63/Q2_K-Q5/DeepSeek-V4.1-Flash-EngramQ5-Q2_K-00008-of-00010.gguf",
                    67_585_085_760, "fae7f35123ae3557034a541507bb9fc24fb62c2e16ff8441c32e8e0477743d5d"),
                new CatalogFile(CatalogFileRole.WeightsShard, "DeepSeek-V4.1-Flash-EngramQ5-Q2_K-00009-of-00010.gguf",
                    "https://huggingface.co/smalinin/DeepSeek-V4.1-Flash-GGUF/resolve/d1de55c19f95172c882906cc83c0e55932d26a63/Q2_K-Q5/DeepSeek-V4.1-Flash-EngramQ5-Q2_K-00009-of-00010.gguf",
                    67_586_936_224, "781fd69e9dd17c09676865830523a2d077a97544d6a004429aab95d2570f8534"),
                new CatalogFile(CatalogFileRole.WeightsShard, "DeepSeek-V4.1-Flash-EngramQ5-Q2_K-00010-of-00010.gguf",
                    "https://huggingface.co/smalinin/DeepSeek-V4.1-Flash-GGUF/resolve/d1de55c19f95172c882906cc83c0e55932d26a63/Q2_K-Q5/DeepSeek-V4.1-Flash-EngramQ5-Q2_K-00010-of-00010.gguf",
                    3_438_927_040, "fa5affb1f971cd6e7effad5684b73780472b486a46330cd7c5776f87c38fea97"),
            },
            Modalities = CatalogModalities.Text,
            MinDeviceMemoryGB = 512,
            ContextLength = 8192,
            // 40 layers, each a 512-wide latent row and a 128-wide indexer key; the compressed
            // layers keep half of the rows, so counting all of them is an upper bound:
            // 40 x (512 + 128) x 2 bytes. Desktop window 32,768.
            KvBytesPerToken = 40 * (512 + 128) * 2,
            KvCacheDtype = "f16",
            LeanCaches = true,
            Sampling = new CatalogSampling(0.7f, 40, 0.95f, 0.0f),
            SupportsThinking = true,
            Experimental = true,
            License = "MIT",
            Notes = "catalog.extended.deepseek41.notes",
        };

        yield return new CatalogModel
        {
            Id = "glm-5.2-iq2xxs",
            DisplayName = "GLM 5.2",
            Family = CatalogFamily.Glm5,
            Kind = CatalogArchitectureKind.MixtureOfExperts,
            Parameters = "catalog.extended.744b40b.parameters",
            Quantization = "UD-IQ2_XXS",
            Files = new[]
            {
                new CatalogFile(CatalogFileRole.Weights, "GLM-5.2-UD-IQ2_XXS-00001-of-00006.gguf",
                    "https://huggingface.co/unsloth/GLM-5.2-GGUF/resolve/abc55e72527792c6e77069c99b4cb7de16fa9f23/UD-IQ2_XXS/GLM-5.2-UD-IQ2_XXS-00001-of-00006.gguf",
                    9_423_744, "7bf96eeabbe887e58b6c44364962731ddc9dc5bf46fec8d097c1dff64bea4a18"),
                new CatalogFile(CatalogFileRole.WeightsShard, "GLM-5.2-UD-IQ2_XXS-00002-of-00006.gguf",
                    "https://huggingface.co/unsloth/GLM-5.2-GGUF/resolve/abc55e72527792c6e77069c99b4cb7de16fa9f23/UD-IQ2_XXS/GLM-5.2-UD-IQ2_XXS-00002-of-00006.gguf",
                    49_105_028_960, "d94adaa58ddd5abbcf2514192958084416b1aa36bd4d21409028a164341bac36"),
                new CatalogFile(CatalogFileRole.WeightsShard, "GLM-5.2-UD-IQ2_XXS-00003-of-00006.gguf",
                    "https://huggingface.co/unsloth/GLM-5.2-GGUF/resolve/abc55e72527792c6e77069c99b4cb7de16fa9f23/UD-IQ2_XXS/GLM-5.2-UD-IQ2_XXS-00003-of-00006.gguf",
                    49_143_176_640, "1cd0b1a3d9d939ce5a184c548f1b1c42edafaf1856cb0d7e586a2884a366256b"),
                new CatalogFile(CatalogFileRole.WeightsShard, "GLM-5.2-UD-IQ2_XXS-00004-of-00006.gguf",
                    "https://huggingface.co/unsloth/GLM-5.2-GGUF/resolve/abc55e72527792c6e77069c99b4cb7de16fa9f23/UD-IQ2_XXS/GLM-5.2-UD-IQ2_XXS-00004-of-00006.gguf",
                    49_143_176_640, "10f3965db697a46ba66494475045af183c1bcaf639984160930c91a377816d3e"),
                new CatalogFile(CatalogFileRole.WeightsShard, "GLM-5.2-UD-IQ2_XXS-00005-of-00006.gguf",
                    "https://huggingface.co/unsloth/GLM-5.2-GGUF/resolve/abc55e72527792c6e77069c99b4cb7de16fa9f23/UD-IQ2_XXS/GLM-5.2-UD-IQ2_XXS-00005-of-00006.gguf",
                    49_143_176_640, "40d7d4524ff07e0f9af494fb13130dc7090184800cc5af0a1563188b076af50d"),
                new CatalogFile(CatalogFileRole.WeightsShard, "GLM-5.2-UD-IQ2_XXS-00006-of-00006.gguf",
                    "https://huggingface.co/unsloth/GLM-5.2-GGUF/resolve/abc55e72527792c6e77069c99b4cb7de16fa9f23/UD-IQ2_XXS/GLM-5.2-UD-IQ2_XXS-00006-of-00006.gguf",
                    41_914_650_304, "eeceb9084350e64be8eebcd1f19ab14bbbb6b40132c86d77ffc65e72f425044d"),
            },
            Modalities = CatalogModalities.Text,
            MinDeviceMemoryGB = 384,
            ContextLength = 8192,
            // MLA: one 576-wide row per token (kv_lora_rank 512 + 64 rope) and the DSA
            // indexer's 128-wide key, for 78 trunk layers and the MTP block:
            // 79 x (576 + 128) x 2 bytes. Desktop window 32,768.
            KvBytesPerToken = 79 * (576 + 128) * 2,
            KvCacheDtype = "f16",
            LeanCaches = true,
            Sampling = new CatalogSampling(0.7f, 40, 0.95f, 0.0f),
            SupportsThinking = true,
            Experimental = true,
            License = "MIT",
            Notes = "catalog.extended.glm.notes",
        };

        yield return new CatalogModel
        {
            Id = "glm-5.3-q2kxl",
            DisplayName = "GLM 5.3",
            Family = CatalogFamily.Glm5,
            Kind = CatalogArchitectureKind.MixtureOfExperts,
            Parameters = "catalog.extended.744b40b.parameters",
            Quantization = "UD-Q2_K_XL",
            Files = new[]
            {
                new CatalogFile(CatalogFileRole.Weights, "GLM-5.3-UD-Q2_K_XL-00001-of-00007.gguf",
                    "https://huggingface.co/unsloth/GLM-5.3-GGUF/resolve/346b3591c7f28d1a23716f97a065ecf12ec14771/UD-Q2_K_XL/GLM-5.3-UD-Q2_K_XL-00001-of-00007.gguf",
                    9_428_677, "4f902dd0ad1458056d156cbfe226d159ef9f68786bcdbfd4fe1d04dca76fca41"),
                new CatalogFile(CatalogFileRole.WeightsShard, "GLM-5.3-UD-Q2_K_XL-00002-of-00007.gguf",
                    "https://huggingface.co/unsloth/GLM-5.3-GGUF/resolve/346b3591c7f28d1a23716f97a065ecf12ec14771/UD-Q2_K_XL/GLM-5.3-UD-Q2_K_XL-00002-of-00007.gguf",
                    49_172_427_136, "979122459ed09d9d895b85e340ddc9c93a7e82f525e20bf5c03e36da19a41b04"),
                new CatalogFile(CatalogFileRole.WeightsShard, "GLM-5.3-UD-Q2_K_XL-00003-of-00007.gguf",
                    "https://huggingface.co/unsloth/GLM-5.3-GGUF/resolve/346b3591c7f28d1a23716f97a065ecf12ec14771/UD-Q2_K_XL/GLM-5.3-UD-Q2_K_XL-00003-of-00007.gguf",
                    49_091_626_976, "f7c9df5efffa2c1eb9b35e061f24f4d8b4a7edbd141e97455673e3e093e803db"),
                new CatalogFile(CatalogFileRole.WeightsShard, "GLM-5.3-UD-Q2_K_XL-00004-of-00007.gguf",
                    "https://huggingface.co/unsloth/GLM-5.3-GGUF/resolve/346b3591c7f28d1a23716f97a065ecf12ec14771/UD-Q2_K_XL/GLM-5.3-UD-Q2_K_XL-00004-of-00007.gguf",
                    49_091_626_976, "8892cc45af116660fb0d6d568814e49b2e6bf173b10d5107a6ab0495c851dbd0"),
                new CatalogFile(CatalogFileRole.WeightsShard, "GLM-5.3-UD-Q2_K_XL-00005-of-00007.gguf",
                    "https://huggingface.co/unsloth/GLM-5.3-GGUF/resolve/346b3591c7f28d1a23716f97a065ecf12ec14771/UD-Q2_K_XL/GLM-5.3-UD-Q2_K_XL-00005-of-00007.gguf",
                    49_091_626_976, "99fff6a71cd659807196e9b624aff72771e79a0cd153f04efd6f8344c82ca489"),
                new CatalogFile(CatalogFileRole.WeightsShard, "GLM-5.3-UD-Q2_K_XL-00006-of-00007.gguf",
                    "https://huggingface.co/unsloth/GLM-5.3-GGUF/resolve/346b3591c7f28d1a23716f97a065ecf12ec14771/UD-Q2_K_XL/GLM-5.3-UD-Q2_K_XL-00006-of-00007.gguf",
                    49_994_703_552, "2e7c0cefe90a2cf54e7831331febd9c8c10beb1fa00437297f153f220aa258d3"),
                new CatalogFile(CatalogFileRole.WeightsShard, "GLM-5.3-UD-Q2_K_XL-00007-of-00007.gguf",
                    "https://huggingface.co/unsloth/GLM-5.3-GGUF/resolve/346b3591c7f28d1a23716f97a065ecf12ec14771/UD-Q2_K_XL/GLM-5.3-UD-Q2_K_XL-00007-of-00007.gguf",
                    7_426_966_496, "1c17315acbe0cd8e79faa81426412da58da960f85e5db3246e038a058f62339b"),
            },
            Modalities = CatalogModalities.Text,
            MinDeviceMemoryGB = 384,
            ContextLength = 8192,
            // MLA: one 576-wide row per token (kv_lora_rank 512 + 64 rope) and the DSA
            // indexer's 128-wide key, for 78 trunk layers and the MTP block:
            // 79 x (576 + 128) x 2 bytes. Desktop window 32,768.
            KvBytesPerToken = 79 * (576 + 128) * 2,
            KvCacheDtype = "f16",
            LeanCaches = true,
            Sampling = new CatalogSampling(0.7f, 40, 0.95f, 0.0f),
            SupportsThinking = true,
            Experimental = true,
            License = "GLM-5.3 License",
            Notes = "catalog.extended.glm.notes",
        };

        yield return new CatalogModel
        {
            Id = "glm-5.3-flash-q2kxl",
            DisplayName = "GLM 5.3 Flash",
            Family = CatalogFamily.Glm5,
            Kind = CatalogArchitectureKind.MixtureOfExperts,
            Parameters = "320B",
            Quantization = "UD-Q2_K_XL",
            Files = new[]
            {
                new CatalogFile(CatalogFileRole.Weights, "GLM-5.3-Flash-UD-Q2_K_XL-00001-of-00004.gguf",
                    "https://huggingface.co/unsloth/GLM-5.3-Flash-GGUF/resolve/a38483c8cd5df544f53d70fb281afe97369d5ab6/UD-Q2_K_XL/GLM-5.3-Flash-UD-Q2_K_XL-00001-of-00004.gguf",
                    9_429_984, "2f32eba33ebc75a715e9c85a010faac4c67542e38e8970bd961d544bfe815f9e"),
                new CatalogFile(CatalogFileRole.WeightsShard, "GLM-5.3-Flash-UD-Q2_K_XL-00002-of-00004.gguf",
                    "https://huggingface.co/unsloth/GLM-5.3-Flash-GGUF/resolve/a38483c8cd5df544f53d70fb281afe97369d5ab6/UD-Q2_K_XL/GLM-5.3-Flash-UD-Q2_K_XL-00002-of-00004.gguf",
                    49_294_975_936, "f4a9e1ab13d5d9620f5590c9a4aba4c169e6707a1554f3be3fe3112f80a66825"),
                new CatalogFile(CatalogFileRole.WeightsShard, "GLM-5.3-Flash-UD-Q2_K_XL-00003-of-00004.gguf",
                    "https://huggingface.co/unsloth/GLM-5.3-Flash-GGUF/resolve/a38483c8cd5df544f53d70fb281afe97369d5ab6/UD-Q2_K_XL/GLM-5.3-Flash-UD-Q2_K_XL-00003-of-00004.gguf",
                    49_949_266_048, "330a8ad76c787b3ab6df062dd30abea7aafe6a0e3d5860e8720dfa4abb2434a5"),
                new CatalogFile(CatalogFileRole.WeightsShard, "GLM-5.3-Flash-UD-Q2_K_XL-00004-of-00004.gguf",
                    "https://huggingface.co/unsloth/GLM-5.3-Flash-GGUF/resolve/a38483c8cd5df544f53d70fb281afe97369d5ab6/UD-Q2_K_XL/GLM-5.3-Flash-UD-Q2_K_XL-00004-of-00004.gguf",
                    9_466_399_584, "5c294f42edc5d69cf00a79ab444e61f0742e9933105fc5c85f97da54dab8946f"),
                new CatalogFile(CatalogFileRole.Projector, "mmproj-BF16.gguf",
                    "https://huggingface.co/unsloth/GLM-5.3-Flash-GGUF/resolve/a38483c8cd5df544f53d70fb281afe97369d5ab6/mmproj-BF16.gguf",
                    1_164_010_176, "82b7adb44411dbb298b1c689567804e5ed0790d80f308ed8b5b6b8ac0527d2ac", Optional: true),
            },
            Modalities = CatalogModalities.Image,
            MinDeviceMemoryGB = 192,
            ContextLength = 8192,
            // glm5-next: 12 of its 46 layers keep a 512-wide MLA row (no rope) and a
            // 128-wide indexer key, the rest a fixed KDA state: 12 x (512 + 128) x 2 bytes.
            // Desktop window 32,768.
            KvBytesPerToken = 12 * (512 + 128) * 2,
            KvCacheDtype = "f16",
            LeanCaches = true,
            Sampling = new CatalogSampling(0.7f, 40, 0.95f, 0.0f),
            SupportsThinking = true,
            Experimental = true,
            License = "MIT",
            Notes = "catalog.extended.glmFlash.notes",
        };

        yield return new CatalogModel
        {
            Id = "diffusiongemma-26b-a4b-q4km",
            DisplayName = "DiffusionGemma 26B-A4B",
            Family = CatalogFamily.DiffusionGemma,
            Kind = CatalogArchitectureKind.Diffusion,
            Parameters = "catalog.extended.26b4b.parameters",
            Quantization = "Q4_K_M",
            Files = new[]
            {
                new CatalogFile(CatalogFileRole.Weights, "diffusiongemma-26B-A4B-it-Q4_K_M.gguf",
                    "https://huggingface.co/unsloth/diffusiongemma-26B-A4B-it-GGUF/resolve/f4183a2c7a354128d02545752303c4354d165bf0/diffusiongemma-26B-A4B-it-Q4_K_M.gguf",
                    16_806_810_208, "24523b6c833c9ce9f5f34f9b333ab1517d73d6f1e76a103645353114c8028bc5"),
                new CatalogFile(CatalogFileRole.Projector, "model-00011-of-00011.safetensors",
                    "https://huggingface.co/google/diffusiongemma-26B-A4B-it/resolve/f7f5b7f5fa82ffc52addd066915886d497f5517b/model-00011-of-00011.safetensors",
                    2_838_371_118, "afec047176bb2a05f078566576aec6bdb71ad4d041275d0d0c89473fda6d6d87", Optional: true),
            },
            Modalities = CatalogModalities.Image,
            MinDeviceMemoryGB = 48,
            ContextLength = 8192,
            KvCacheDtype = "f16",
            LeanCaches = true,
            Sampling = new CatalogSampling(0.7f, 40, 0.95f, 0.0f),
            SupportsThinking = false,
            Experimental = true,
            License = GemmaLicense,
            Notes = "catalog.extended.diffusionGemma.notes",
        };

        yield return new CatalogModel
        {
            Id = "wan2.1-t2v-1.3b-q8",
            DisplayName = "Wan 2.1 T2V 1.3B",
            Family = CatalogFamily.Wan,
            Kind = CatalogArchitectureKind.Diffusion,
            Parameters = "1.3B",
            Quantization = "Q8_0",
            Files = new[]
            {
                new CatalogFile(CatalogFileRole.Weights, "Wan2.1-T2V-1.3B-Q8_0.gguf",
                    "https://huggingface.co/samuelchristlie/Wan2.1-T2V-1.3B-GGUF/resolve/5a512b15fc35d1b67a074cfe55a591be9e9ef9b5/Wan2.1-T2V-1.3B-Q8_0.gguf",
                    1_535_768_800, "30a44f695b4275a915810120360d6fd26152ec303c2226b5152ec33a93c380e4"),
                new CatalogFile(CatalogFileRole.TextEncoder, "umt5-xxl-encoder-Q8_0.gguf",
                    "https://huggingface.co/city96/umt5-xxl-encoder-gguf/resolve/b535255bee98c2b0a59ea7c0ae2dcd0c6657b3b7/umt5-xxl-encoder-Q8_0.gguf",
                    6_043_068_256, "2521d4de0bf9e1cc6549866463ceae85e4ec3239bc6063f7488810be39033bbc"),
                new CatalogFile(CatalogFileRole.Vae, "Wan2.1_VAE.safetensors",
                    "https://huggingface.co/QuantStack/Wan2.2-I2V-A14B-GGUF/resolve/6c6717459277b9cd1f72579d78a0fd62a79e57dc/VAE/Wan2.1_VAE.safetensors",
                    253_815_318, "2fc39d31359a4b0a64f55876d8ff7fa8d780956ae2cb13463b0223e15148976b"),
            },
            Modalities = CatalogModalities.VideoOutput,
            MinDeviceMemoryGB = 32,
            ContextLength = 0,
            KvCacheDtype = "f16",
            LeanCaches = true,
            Sampling = new CatalogSampling(1.0f, 0, 1.0f, 0.0f),
            SupportsThinking = false,
            Experimental = true,
            License = "Apache-2.0",
            Notes = "catalog.extended.wanText.notes",
        };

        yield return new CatalogModel
        {
            Id = "wan2.1-t2v-14b-q4km",
            DisplayName = "Wan 2.1 T2V 14B",
            Family = CatalogFamily.Wan,
            Kind = CatalogArchitectureKind.Diffusion,
            Parameters = "14B",
            Quantization = "Q4_K_M",
            Files = new[]
            {
                new CatalogFile(CatalogFileRole.Weights, "wan2.1-t2v-14b-Q4_K_M.gguf",
                    "https://huggingface.co/city96/Wan2.1-T2V-14B-gguf/resolve/511cbce9f475a6ca0269be901d23b125f44f5c0d/wan2.1-t2v-14b-Q4_K_M.gguf",
                    10_124_581_504, "c9a64d612831debbac77af93744d4b5196b840bf65e9f9f5b958d03fc3baddf2"),
                new CatalogFile(CatalogFileRole.TextEncoder, "umt5-xxl-encoder-Q8_0.gguf",
                    "https://huggingface.co/city96/umt5-xxl-encoder-gguf/resolve/b535255bee98c2b0a59ea7c0ae2dcd0c6657b3b7/umt5-xxl-encoder-Q8_0.gguf",
                    6_043_068_256, "2521d4de0bf9e1cc6549866463ceae85e4ec3239bc6063f7488810be39033bbc"),
                new CatalogFile(CatalogFileRole.Vae, "Wan2.1_VAE.safetensors",
                    "https://huggingface.co/QuantStack/Wan2.2-I2V-A14B-GGUF/resolve/6c6717459277b9cd1f72579d78a0fd62a79e57dc/VAE/Wan2.1_VAE.safetensors",
                    253_815_318, "2fc39d31359a4b0a64f55876d8ff7fa8d780956ae2cb13463b0223e15148976b"),
            },
            Modalities = CatalogModalities.VideoOutput,
            MinDeviceMemoryGB = 48,
            ContextLength = 0,
            KvCacheDtype = "f16",
            LeanCaches = true,
            Sampling = new CatalogSampling(1.0f, 0, 1.0f, 0.0f),
            SupportsThinking = false,
            Experimental = true,
            License = "Apache-2.0",
            Notes = "catalog.extended.wanText.notes",
        };

        yield return new CatalogModel
        {
            Id = "wan2.2-ti2v-5b-q8",
            DisplayName = "Wan 2.2 TI2V 5B",
            Family = CatalogFamily.Wan,
            Kind = CatalogArchitectureKind.Diffusion,
            Parameters = "5B",
            Quantization = "Q8_0",
            Files = new[]
            {
                new CatalogFile(CatalogFileRole.Weights, "Wan2.2-TI2V-5B-Q8_0.gguf",
                    "https://huggingface.co/QuantStack/Wan2.2-TI2V-5B-GGUF/resolve/57437632ddd08bdcbd1508c866aa22e126ed51d2/Wan2.2-TI2V-5B-Q8_0.gguf",
                    5_400_179_040, "57bece983817ab2f957546683bb670f13be7d99022d45674840cd999a050ea8f"),
                new CatalogFile(CatalogFileRole.TextEncoder, "umt5-xxl-encoder-Q8_0.gguf",
                    "https://huggingface.co/city96/umt5-xxl-encoder-gguf/resolve/b535255bee98c2b0a59ea7c0ae2dcd0c6657b3b7/umt5-xxl-encoder-Q8_0.gguf",
                    6_043_068_256, "2521d4de0bf9e1cc6549866463ceae85e4ec3239bc6063f7488810be39033bbc"),
                new CatalogFile(CatalogFileRole.Vae, "Wan2.2_VAE.safetensors",
                    "https://huggingface.co/QuantStack/Wan2.2-TI2V-5B-GGUF/resolve/57437632ddd08bdcbd1508c866aa22e126ed51d2/VAE/Wan2.2_VAE.safetensors",
                    1_409_400_960, "e40321bd36b9709991dae2530eb4ac303dd168276980d3e9bc4b6e2b75fed156"),
            },
            Modalities = CatalogModalities.VideoOutput | CatalogModalities.Image,
            MinDeviceMemoryGB = 32,
            ContextLength = 0,
            KvCacheDtype = "f16",
            LeanCaches = true,
            Sampling = new CatalogSampling(1.0f, 0, 1.0f, 0.0f),
            SupportsThinking = false,
            Experimental = true,
            License = "Apache-2.0",
            Notes = "catalog.extended.wanBase.notes",
        };

        yield return new CatalogModel
        {
            Id = "wan2.2-ti2v-5b-turbo-q8",
            DisplayName = "Wan 2.2 TI2V 5B Turbo",
            Family = CatalogFamily.Wan,
            Kind = CatalogArchitectureKind.Diffusion,
            Parameters = "5B",
            Quantization = "Q8_0",
            Files = new[]
            {
                new CatalogFile(CatalogFileRole.Weights, "Wan2_2-TI2V-5B-Turbo-Q8_0.gguf",
                    "https://huggingface.co/hum-ma/Wan2.2-TI2V-5B-Turbo-GGUF/resolve/9873ba175b4b204a81f4a43c9644e6c385ee4459/Wan2_2-TI2V-5B-Turbo-Q8_0.gguf",
                    5_404_990_176, "b4d7d1888d5ec2c29046cc443239faad99614cd3d356c61d8192fcac5f4c98fc"),
                new CatalogFile(CatalogFileRole.TextEncoder, "umt5-xxl-encoder-Q8_0.gguf",
                    "https://huggingface.co/city96/umt5-xxl-encoder-gguf/resolve/b535255bee98c2b0a59ea7c0ae2dcd0c6657b3b7/umt5-xxl-encoder-Q8_0.gguf",
                    6_043_068_256, "2521d4de0bf9e1cc6549866463ceae85e4ec3239bc6063f7488810be39033bbc"),
                new CatalogFile(CatalogFileRole.Vae, "Wan2.2_VAE.safetensors",
                    "https://huggingface.co/QuantStack/Wan2.2-TI2V-5B-GGUF/resolve/57437632ddd08bdcbd1508c866aa22e126ed51d2/VAE/Wan2.2_VAE.safetensors",
                    1_409_400_960, "e40321bd36b9709991dae2530eb4ac303dd168276980d3e9bc4b6e2b75fed156"),
            },
            Modalities = CatalogModalities.VideoOutput | CatalogModalities.Image,
            MinDeviceMemoryGB = 32,
            ContextLength = 0,
            KvCacheDtype = "f16",
            LeanCaches = true,
            Sampling = new CatalogSampling(1.0f, 0, 1.0f, 0.0f),
            SupportsThinking = false,
            Experimental = true,
            License = "Apache-2.0",
            Notes = "catalog.extended.wanTurbo.notes",
        };

        yield return new CatalogModel
        {
            Id = "wan2.2-t2v-a14b-q4km",
            DisplayName = "Wan 2.2 T2V A14B",
            Family = CatalogFamily.Wan,
            Kind = CatalogArchitectureKind.Diffusion,
            Parameters = "catalog.extended.14bExpert.parameters",
            Quantization = "Q4_K_M",
            Files = new[]
            {
                new CatalogFile(CatalogFileRole.Weights, "Wan2.2-T2V-A14B-HighNoise-Q4_K_M.gguf",
                    "https://huggingface.co/QuantStack/Wan2.2-T2V-A14B-GGUF/resolve/73eafba53a1a8f29254e4c77f92e74ea27d7cd6f/HighNoise/Wan2.2-T2V-A14B-HighNoise-Q4_K_M.gguf",
                    9_650_090_496, "e0c490c6e316fd91ff52034e4ca66b825717e33ff11624585c0ccfcb5d410c59"),
                new CatalogFile(CatalogFileRole.SecondaryWeights, "Wan2.2-T2V-A14B-LowNoise-Q4_K_M.gguf",
                    "https://huggingface.co/QuantStack/Wan2.2-T2V-A14B-GGUF/resolve/73eafba53a1a8f29254e4c77f92e74ea27d7cd6f/LowNoise/Wan2.2-T2V-A14B-LowNoise-Q4_K_M.gguf",
                    9_650_090_496, "091a5bae02e14aa016bc9b10a7892efda4c629346b81c5dcebbe30ea2ac8923a"),
                new CatalogFile(CatalogFileRole.TextEncoder, "umt5-xxl-encoder-Q8_0.gguf",
                    "https://huggingface.co/city96/umt5-xxl-encoder-gguf/resolve/b535255bee98c2b0a59ea7c0ae2dcd0c6657b3b7/umt5-xxl-encoder-Q8_0.gguf",
                    6_043_068_256, "2521d4de0bf9e1cc6549866463ceae85e4ec3239bc6063f7488810be39033bbc"),
                new CatalogFile(CatalogFileRole.Vae, "Wan2.1_VAE.safetensors",
                    "https://huggingface.co/QuantStack/Wan2.2-I2V-A14B-GGUF/resolve/6c6717459277b9cd1f72579d78a0fd62a79e57dc/VAE/Wan2.1_VAE.safetensors",
                    253_815_318, "2fc39d31359a4b0a64f55876d8ff7fa8d780956ae2cb13463b0223e15148976b"),
            },
            Modalities = CatalogModalities.VideoOutput,
            MinDeviceMemoryGB = 48,
            ContextLength = 0,
            KvCacheDtype = "f16",
            LeanCaches = true,
            Sampling = new CatalogSampling(1.0f, 0, 1.0f, 0.0f),
            SupportsThinking = false,
            Experimental = true,
            License = "Apache-2.0",
            Notes = "catalog.extended.wanA14Text.notes",
        };

        yield return new CatalogModel
        {
            Id = "wan2.2-i2v-a14b-q4km",
            DisplayName = "Wan 2.2 I2V A14B",
            Family = CatalogFamily.Wan,
            Kind = CatalogArchitectureKind.Diffusion,
            Parameters = "catalog.extended.14bExpert.parameters",
            Quantization = "Q4_K_M",
            Files = new[]
            {
                new CatalogFile(CatalogFileRole.Weights, "Wan2.2-I2V-A14B-HighNoise-Q4_K_M.gguf",
                    "https://huggingface.co/QuantStack/Wan2.2-I2V-A14B-GGUF/resolve/6c6717459277b9cd1f72579d78a0fd62a79e57dc/HighNoise/Wan2.2-I2V-A14B-HighNoise-Q4_K_M.gguf",
                    9_651_728_896, "836250abfaa3411694e2c9cf3a0cc18265329d5156d81aa116d5366b0f8f02e7"),
                new CatalogFile(CatalogFileRole.SecondaryWeights, "Wan2.2-I2V-A14B-LowNoise-Q4_K_M.gguf",
                    "https://huggingface.co/QuantStack/Wan2.2-I2V-A14B-GGUF/resolve/6c6717459277b9cd1f72579d78a0fd62a79e57dc/LowNoise/Wan2.2-I2V-A14B-LowNoise-Q4_K_M.gguf",
                    9_651_728_896, "e2f98d834af009d035c6b0918268f2eba0aa8a63025ce942277e2384d40b0866"),
                new CatalogFile(CatalogFileRole.TextEncoder, "umt5-xxl-encoder-Q8_0.gguf",
                    "https://huggingface.co/city96/umt5-xxl-encoder-gguf/resolve/b535255bee98c2b0a59ea7c0ae2dcd0c6657b3b7/umt5-xxl-encoder-Q8_0.gguf",
                    6_043_068_256, "2521d4de0bf9e1cc6549866463ceae85e4ec3239bc6063f7488810be39033bbc"),
                new CatalogFile(CatalogFileRole.Vae, "Wan2.1_VAE.safetensors",
                    "https://huggingface.co/QuantStack/Wan2.2-I2V-A14B-GGUF/resolve/6c6717459277b9cd1f72579d78a0fd62a79e57dc/VAE/Wan2.1_VAE.safetensors",
                    253_815_318, "2fc39d31359a4b0a64f55876d8ff7fa8d780956ae2cb13463b0223e15148976b"),
            },
            Modalities = CatalogModalities.VideoOutput | CatalogModalities.Image,
            MinDeviceMemoryGB = 48,
            ContextLength = 0,
            KvCacheDtype = "f16",
            LeanCaches = true,
            Sampling = new CatalogSampling(1.0f, 0, 1.0f, 0.0f),
            SupportsThinking = false,
            Experimental = true,
            License = "Apache-2.0",
            Notes = "catalog.extended.wanA14Image.notes",
        };

        yield return new CatalogModel
        {
            Id = "wan2.2-i2v-a14b-lightx2v-q4km",
            DisplayName = "Wan 2.2 I2V A14B LightX2V",
            Family = CatalogFamily.Wan,
            Kind = CatalogArchitectureKind.Diffusion,
            Parameters = "catalog.extended.14bExpert.parameters",
            Quantization = "Q4_K_M",
            Files = new[]
            {
                new CatalogFile(CatalogFileRole.Weights, "wan2.2_i2v_A14b_high_noise_lightx2v_4step-Q4_K_M.gguf",
                    "https://huggingface.co/jayn7/WAN2.2-I2V_A14B-DISTILL-LIGHTX2V-4STEP-GGUF/resolve/338fb8eedd8f485c9188cf1b1de541721fc81d66/high_noise/wan2.2_i2v_A14b_high_noise_lightx2v_4step-Q4_K_M.gguf",
                    9_661_569_664, "5945a053f65185b47119f53d75b52b95a7cdca7e5e2f8c5d5559d1cbf076c128"),
                new CatalogFile(CatalogFileRole.SecondaryWeights, "wan2.2_i2v_A14b_low_noise_lightx2v_4step-Q4_K_M.gguf",
                    "https://huggingface.co/jayn7/WAN2.2-I2V_A14B-DISTILL-LIGHTX2V-4STEP-GGUF/resolve/338fb8eedd8f485c9188cf1b1de541721fc81d66/low_noise/wan2.2_i2v_A14b_low_noise_lightx2v_4step-Q4_K_M.gguf",
                    9_661_569_664, "7776878df502c39a815cd6cc6a19b9ff78aa921c3ade88dfc33f62abc129535e"),
                new CatalogFile(CatalogFileRole.TextEncoder, "umt5-xxl-encoder-Q8_0.gguf",
                    "https://huggingface.co/city96/umt5-xxl-encoder-gguf/resolve/b535255bee98c2b0a59ea7c0ae2dcd0c6657b3b7/umt5-xxl-encoder-Q8_0.gguf",
                    6_043_068_256, "2521d4de0bf9e1cc6549866463ceae85e4ec3239bc6063f7488810be39033bbc"),
                new CatalogFile(CatalogFileRole.Vae, "Wan2.1_VAE.safetensors",
                    "https://huggingface.co/QuantStack/Wan2.2-I2V-A14B-GGUF/resolve/6c6717459277b9cd1f72579d78a0fd62a79e57dc/VAE/Wan2.1_VAE.safetensors",
                    253_815_318, "2fc39d31359a4b0a64f55876d8ff7fa8d780956ae2cb13463b0223e15148976b"),
            },
            Modalities = CatalogModalities.VideoOutput | CatalogModalities.Image,
            MinDeviceMemoryGB = 48,
            ContextLength = 0,
            KvCacheDtype = "f16",
            LeanCaches = true,
            Sampling = new CatalogSampling(1.0f, 0, 1.0f, 0.0f),
            SupportsThinking = false,
            Experimental = true,
            License = "Apache-2.0",
            Notes = "catalog.extended.wanA14Image.notes",
        };

    }
}
