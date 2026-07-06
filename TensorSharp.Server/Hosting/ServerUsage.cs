// Copyright (c) Zhongkai Fu. All rights reserved.
// https://github.com/zhongkaifu/TensorSharp
//
// This file is part of TensorSharp.
//
// TensorSharp is licensed under the BSD-3-Clause license found in the LICENSE file in the root directory of this source tree.
//
// TensorSharp is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the BSD-3-Clause License for more details.

using System;
using System.IO;

namespace TensorSharp.Server.Hosting
{
    /// <summary>
    /// Informational entry points that print and exit before the web host is
    /// built: the full usage page (shown for a bare <c>TensorSharp.Server</c>
    /// invocation or <c>--help</c>) and the Vulkan GPU listing
    /// (<c>--list-gpus</c>). Kept out of <see cref="ServerOptionsBuilder"/> so
    /// the option parser stays pure and testable.
    /// </summary>
    internal static class ServerUsage
    {
        public static bool IsHelpRequested(string[] args)
        {
            if (args == null)
                return false;
            foreach (string a in args)
            {
                if (string.Equals(a, "--help", StringComparison.OrdinalIgnoreCase) ||
                    string.Equals(a, "-h", StringComparison.OrdinalIgnoreCase) ||
                    string.Equals(a, "-?", StringComparison.Ordinal) ||
                    string.Equals(a, "/?", StringComparison.Ordinal))
                {
                    return true;
                }
            }
            return false;
        }

        public static bool IsListGpusRequested(string[] args)
        {
            if (args == null)
                return false;
            foreach (string a in args)
            {
                if (string.Equals(a, "--list-gpus", StringComparison.OrdinalIgnoreCase))
                    return true;
            }
            return false;
        }

        /// <summary>
        /// Print the Vulkan devices ggml-vulkan can see (index + adapter name) so the
        /// operator knows what to pass to <c>--gpu-device</c> on multi-GPU hosts.
        /// Enumerating spins up the Vulkan instance but no backend/device state.
        /// Mirrors the CLI's <c>--list-gpus</c>.
        /// </summary>
        public static void PrintVulkanGpus(TextWriter writer)
        {
            int count = TensorSharp.GGML.GgmlBasicOps.GetVulkanDeviceCount();
            if (count <= 0)
            {
                writer.WriteLine("No Vulkan devices found. Ensure the native GGML bridge is built with Vulkan support " +
                    "(TensorSharp.GGML.Native/build-windows.ps1 --vulkan) and a Vulkan driver is installed.");
                return;
            }

            writer.WriteLine($"Vulkan devices ({count}):");
            for (int i = 0; i < count; i++)
            {
                writer.WriteLine($"  {i}: {TensorSharp.GGML.GgmlBasicOps.GetVulkanDeviceDescription(i) ?? "(unknown)"}");
            }
            writer.WriteLine("Select one with: --backend ggml_vulkan --gpu-device <index>");
        }

        /// <summary>One option entry on the usage page.</summary>
        private readonly record struct OptionHelp(string Flag, string Description, string Example);

        // Grouped to match the option passes in Program.cs / ServerOptionsBuilder.
        // Keep flags in sync with ServerOptionsBuilder.ParseArgs and its
        // SuggestFlagCorrection known-flag list.
        private static readonly (string Section, OptionHelp[] Options)[] Sections =
        {
            ("Model", new[]
            {
                new OptionHelp("--model <path>",
                    "GGUF model to host at startup. Default: none — pass any other option (e.g. --backend) to start " +
                    "without a model and load one later from the web UI or /api/models/load.",
                    "--model C:\\models\\Qwen3.5-9B-Q8_0.gguf"),
                new OptionHelp("--mmproj <path|none>",
                    "Multimodal projector GGUF. A bare filename is resolved next to the model; 'none' disables it. " +
                    "Requires --model. Default: auto-detect next to the model.",
                    "--mmproj gemma-4-E4B-mmproj-F16.gguf"),
            }),
            ("Compute backend", new[]
            {
                new OptionHelp("--backend <type>",
                    "Compute backend: cpu, cuda, mlx, ggml_cpu, ggml_metal, ggml_cuda, or ggml_vulkan. " +
                    "Default: ggml_metal on macOS, ggml_cpu elsewhere (BACKEND env var overrides).",
                    "--backend ggml_vulkan"),
                new OptionHelp("--gpu-device <N>",
                    "Vulkan device index for the ggml_vulkan backend on multi-GPU hosts (e.g. an integrated Intel GPU " +
                    "next to a discrete NVIDIA one). Default: 0 (TS_GGML_VULKAN_DEVICE env var overrides).",
                    "--backend ggml_vulkan --gpu-device 1"),
                new OptionHelp("--list-gpus",
                    "List the Vulkan devices ggml-vulkan can see (index + adapter name) and exit.",
                    "--list-gpus"),
            }),
            ("Generation defaults (used when a request omits the field)", new[]
            {
                new OptionHelp("--max-tokens <N>",
                    "Maximum tokens to generate per request. Default: 20000 (MAX_TOKENS env var overrides).",
                    "--max-tokens 4096"),
                new OptionHelp("--temperature <f>",
                    "Sampling temperature; 0 = greedy. Default: 0.8 (TENSORSHARP_TEMPERATURE env var).",
                    "--temperature 0"),
                new OptionHelp("--top-k <N>",
                    "Top-K filtering; 0 disables. Default: 40 (TENSORSHARP_TOP_K env var).",
                    "--top-k 64"),
                new OptionHelp("--top-p <f>",
                    "Nucleus sampling threshold; 1.0 disables. Default: 0.9 (TENSORSHARP_TOP_P env var).",
                    "--top-p 0.95"),
                new OptionHelp("--min-p <f>",
                    "Minimum-probability filtering; 0 disables. Default: 0 (TENSORSHARP_MIN_P env var).",
                    "--min-p 0.05"),
                new OptionHelp("--repeat-penalty <f>",
                    "Repetition penalty; 1.0 = none. Default: 1.1 (TENSORSHARP_REPEAT_PENALTY env var).",
                    "--repeat-penalty 1.0"),
                new OptionHelp("--presence-penalty <f>",
                    "Presence penalty; 0 disables. Default: 0 (TENSORSHARP_PRESENCE_PENALTY env var).",
                    "--presence-penalty 0.2"),
                new OptionHelp("--frequency-penalty <f>",
                    "Frequency penalty; 0 disables. Default: 0 (TENSORSHARP_FREQUENCY_PENALTY env var).",
                    "--frequency-penalty 0.3"),
                new OptionHelp("--seed <N>",
                    "Random seed; -1 = non-deterministic. Default: -1 (TENSORSHARP_SEED env var).",
                    "--seed 42"),
                new OptionHelp("--stop <text>",
                    "Stop sequence; repeat the flag to pin several. Default: none.",
                    "--stop \"</s>\" --stop \"<|eot|>\""),
            }),
            ("KV cache", new[]
            {
                new OptionHelp("--kv-cache-dtype <t>",
                    "KV cache precision: f32, f16, q8_0, or q4_0. Quantized caches trade small numerical drift for " +
                    "memory. Default: auto — the backend/model pick (KV_CACHE_DTYPE env var overrides).",
                    "--kv-cache-dtype q8_0"),
            }),
            ("Cross-session paged KV cache", new[]
            {
                new OptionHelp("--paged-kv | --no-paged-kv",
                    "Enable/disable the cross-session paged KV cache (prefix reuse across requests). Default: off.",
                    "--paged-kv"),
                new OptionHelp("--paged-kv-block-size <N>",
                    "Tokens per KV block. Default: 256.",
                    "--paged-kv-block-size 128"),
                new OptionHelp("--paged-kv-ram-mb <N>",
                    "RAM budget for evicted KV blocks, in MB. Default: 1024.",
                    "--paged-kv-ram-mb 2048"),
                new OptionHelp("--paged-kv-ssd-dir <path>",
                    "Directory for the SSD spill tier. Default: disabled.",
                    "--paged-kv-ssd-dir D:\\ts-kv-spill"),
                new OptionHelp("--paged-kv-ssd-mb <N>",
                    "SSD budget for spilled KV blocks, in MB. Default: 16384.",
                    "--paged-kv-ssd-mb 32768"),
                new OptionHelp("--paged-kv-quant-bits <b>",
                    "Quantize spilled KV blocks: 0 (off), 4, or 8 bits. Default: 0.",
                    "--paged-kv-quant-bits 8"),
            }),
            ("Scheduling", new[]
            {
                new OptionHelp("--continuous-batching | --no-continuous-batching",
                    "Paged-attention continuous batching across concurrent requests (aliases --paged-batching / " +
                    "--no-paged-batching). Default: on.",
                    "--no-continuous-batching"),
                new OptionHelp("--prefill-chunk-size <N>",
                    "Chunked-prefill granularity under contention; smaller chunks give parallel decodes more frequent " +
                    "turns at the GPU. Default: 1024.",
                    "--prefill-chunk-size 256"),
            }),
            ("MTP speculative decoding (models that ship an MTP/NextN draft head)", new[]
            {
                new OptionHelp("--mtp-spec | --no-mtp-spec",
                    "Enable/disable MTP speculative decoding. Default: off.",
                    "--mtp-spec"),
                new OptionHelp("--mtp-draft <N>",
                    "Maximum draft tokens per step. Default: 8.",
                    "--mtp-draft 4"),
                new OptionHelp("--mtp-pmin <f>",
                    "Minimum draft-token probability in (0, 1]; drafting stops below it. Default: 0.75.",
                    "--mtp-pmin 0.6"),
                new OptionHelp("--mtp-draft-model <path>",
                    "Separate draft GGUF for models whose draft head ships as its own file (Gemma 4 assistant). " +
                    "Qwen3.6 embeds the draft head and needs no flag. Default: none.",
                    "--mtp-draft-model gemma-4-E4B-it-assistant.Q8_0.gguf"),
            }),
            ("Qwen-Image-Edit companion models (qwen_image DiT GGUFs)", new[]
            {
                new OptionHelp("--qwen-image-vae <path>",
                    "VAE GGUF. Default: same-directory scan next to the DiT model.",
                    "--qwen-image-vae qwen-image-vae.gguf"),
                new OptionHelp("--qwen-image-vl <path>",
                    "Qwen2.5-VL text-encoder GGUF. Default: same-directory scan.",
                    "--qwen-image-vl qwen-image-te-Qwen2.5-VL-7B-Q4_K_M.gguf"),
                new OptionHelp("--qwen-image-mmproj <path>",
                    "Vision projector GGUF for the text encoder. Default: same-directory scan.",
                    "--qwen-image-mmproj Qwen2.5-VL-7B-mmproj-BF16.gguf"),
                new OptionHelp("--qwen-image-lora <path>",
                    "DiT LoRA (e.g. a Lightning step-distillation checkpoint); also switches sampling defaults. " +
                    "Default: none.",
                    "--qwen-image-lora Qwen-Image-Edit-Lightning-8steps.safetensors"),
            }),
            ("Help", new[]
            {
                new OptionHelp("--help",
                    "Show this help and exit (also shown when the server is started with no arguments).",
                    "--help"),
            }),
        };

        public static void PrintUsage(TextWriter writer)
        {
            writer.WriteLine("Usage: TensorSharp.Server [options]");
            writer.WriteLine();
            writer.WriteLine("Hosts an OpenAI- and Ollama-compatible inference server (plus a built-in web chat UI)");
            writer.WriteLine("on http://0.0.0.0:5000. Run with no arguments to show this help; pass at least one");
            writer.WriteLine("option to start the server.");

            foreach (var (section, options) in Sections)
            {
                writer.WriteLine();
                writer.WriteLine(section + ":");
                foreach (var option in options)
                {
                    writer.WriteLine($"  {option.Flag}");
                    WriteWrapped(writer, option.Description, indent: "      ");
                    writer.WriteLine($"      Example: {option.Example}");
                }
            }

            writer.WriteLine();
            writer.WriteLine("Examples:");
            writer.WriteLine("  TensorSharp.Server --model C:\\models\\gemma-4-E4B-it-Q8_0.gguf --backend ggml_vulkan --gpu-device 1");
            writer.WriteLine("  TensorSharp.Server --model model.gguf --mmproj mmproj.gguf --backend ggml_cuda --max-tokens 4096");
            writer.WriteLine("  TensorSharp.Server --backend ggml_cpu    (start with no model; load one from the web UI)");
            writer.WriteLine();
            writer.WriteLine("Logging env vars: TENSORSHARP_LOG_LEVEL (Information), TENSORSHARP_LOG_DIR (./logs),");
            writer.WriteLine("TENSORSHARP_LOG_FILE=0 disables file logging.");
        }

        private const int WrapColumn = 100;

        private static void WriteWrapped(TextWriter writer, string text, string indent)
        {
            int width = WrapColumn - indent.Length;
            var line = new System.Text.StringBuilder();
            foreach (string word in text.Split(' ', StringSplitOptions.RemoveEmptyEntries))
            {
                if (line.Length > 0 && line.Length + 1 + word.Length > width)
                {
                    writer.WriteLine(indent + line);
                    line.Clear();
                }
                if (line.Length > 0)
                    line.Append(' ');
                line.Append(word);
            }
            if (line.Length > 0)
                writer.WriteLine(indent + line);
        }
    }
}
