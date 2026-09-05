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
using TensorSharp.AgentHost.Skills;
using System.Collections.Generic;
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
    public static class ServerUsage
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
                    "GGUF model to host at startup. Required for inference. Other options can start a model-less " +
                    "status process, but /api/models/load cannot select a GGUF that was not supplied at startup.",
                    "--model C:\\models\\gemma-4-E4B-it-Q8_0.gguf"),
                new OptionHelp("--mmproj <path|none>",
                    "Multimodal projector GGUF. A bare filename is resolved next to the model; 'none' disables it. " +
                    "Requires --model. Default: none — pass the matching projector explicitly.",
                    "--mmproj mmproj-gemma-4-E4B-it-Q8_0.gguf"),
            }),
            ("Network", new[]
            {
                new OptionHelp("--port <N>",
                    $"TCP port to listen on (1-65535). Default: {ServerHostingOptions.DefaultPort} (PORT env var overrides). On macOS, port 5000 is " +
                    "taken by the AirPlay Receiver in Control Center, so pick another port or turn that off.",
                    "--port 8080"),
                new OptionHelp("--host <address>",
                    "Interface to bind. Default: 0.0.0.0 — every interface, so the server is reachable from other " +
                    "machines and from outside a container. Use 127.0.0.1 to restrict it to this machine " +
                    "(HOST env var overrides).",
                    "--host 127.0.0.1 --port 8080"),
                new OptionHelp("--urls <urls>",
                    "Full listen URL(s), semicolon-separated, for cases --port/--host cannot express (HTTPS, or " +
                    "binding several endpoints at once). Overridden by --port/--host when both are given; falls back " +
                    "to the ASPNETCORE_URLS env var.",
                    "--urls \"http://0.0.0.0:8080;https://0.0.0.0:8443\""),
                new OptionHelp("--no-webui",
                    "Do not serve the bundled web UI; GET / answers the plain liveness text instead. All HTTP API " +
                    "endpoints (including /uploads) stay up. Default: UI on (TS_NO_WEBUI env var overrides).",
                    "--no-webui"),
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
            ("Tensor parallelism (multi-GPU serving)", new[]
            {
                new OptionHelp("--tp <N>",
                    "Split the model across N GPUs on this machine (tensor parallelism): each GPU holds 1/N of every " +
                    "weight and the shards cooperate on every token. Use it when a model does not fit on one GPU. " +
                    "Range: 1 to the number of local GPUs. Applies to the cuda, ggml_cuda, and ggml_vulkan backends. " +
                    "Multi-GPU is implemented PER ARCHITECTURE, not per backend, and in two forms. Architectures that shard weights run true tensor parallelism. qwen4exp (Qwen3.8-Flash-Next) shards nothing, so --tp N runs it as a LAYER SPLIT instead - each GPU holds a contiguous run of whole layers, which is the same and only multi-GPU mode llama.cpp offers for it. That is a CAPACITY feature: it lets a model, context or resident-weight set that one GPU cannot hold fit across several, and is not expected to raise tok/s. The startup line says which mode actually ran. An architecture that supports neither says so on stderr and runs on one GPU rather than silently leaving the others idle. " +
                    "Default: 1 — no splitting (TENSORSHARP_TP_DEGREE env var overrides).",
                    "--model Qwen3.5-35B-A3B-Q4_K_M.gguf --backend ggml_cuda --tp 2"),
                new OptionHelp("--tp-node-id <N>",
                    "This node's 0-based ID for multi-node (distributed) tensor parallelism over TCP. The server can " +
                    "only be node 0 — the driver that owns sampling and serves HTTP; start every other node as a " +
                    "worker with TensorSharp.Cli using the same model, backend, and --tp-peers list. Requires " +
                    "--tp-peers. Default: none — single-node (TENSORSHARP_TP_NODE_ID env var overrides).",
                    "--tp 2 --tp-node-id 0 --tp-peers 192.168.1.10:9500,192.168.1.11:9500"),
                new OptionHelp("--tp-peers <list>",
                    "Comma-separated host:port list of ALL nodes in the distributed TP cluster, ordered by node ID; " +
                    "every node passes the identical list. Requires --tp-node-id. Default: none " +
                    "(TENSORSHARP_TP_PEERS env var overrides).",
                    "--tp-peers 192.168.1.10:9500,192.168.1.11:9500"),
            }),
            ("Generation defaults (pinned values also override requests — see --sampling-precedence)", new[]
            {
                new OptionHelp("--max-tokens <N>",
                    "Maximum tokens to generate per request: fills in when the request omits a limit, and caps a " +
                    "request that asks for more. Default: 20000, uncapped (MAX_TOKENS env var overrides).",
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
                new OptionHelp("--repeat-last-n <n>",
                    "Recent-token penalty window; 0 disables, -1 uses all. Default: 64 (TENSORSHARP_REPEAT_LAST_N env var).",
                    "--repeat-last-n 128"),
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
                new OptionHelp("--sampling-precedence <config|request>",
                    "Who wins when a request also carries a sampling parameter you pinned above. 'config' " +
                    "(default) keeps your values — clients such as VS Code Copilot Chat hardcode temperature/top_p " +
                    "into every request and would otherwise silently override them; parameters you did NOT pin " +
                    "still come from the request. 'request' restores client-always-wins. Pinned stop sequences " +
                    "are merged with the request's rather than replacing them " +
                    "(TENSORSHARP_SAMPLING_PRECEDENCE env var overrides).",
                    "--sampling-precedence request"),
            }),
            ("Mixture-of-Experts CPU offload", new[]
            {
                new OptionHelp("--n-cpu-moe <N> | -ncmoe <N>",
                    "Keep the routed MoE expert weights of the first N layers in system RAM and multiply them on " +
                    "the CPU; attention, norms, the router and the shared expert stay on the accelerator. This is " +
                    "what makes a 35B-A3B MoE fit beside a long-context KV cache on a 12-16 GB card. Pass 'all' " +
                    "for every layer. Default: 0 (everything on the accelerator; TS_N_CPU_MOE env var overrides).",
                    "--n-cpu-moe 32"),
                new OptionHelp("--cpu-moe | -cmoe",
                    "Shorthand for --n-cpu-moe all: every routed expert stays in system RAM. Default: off " +
                    "(TS_CPU_MOE env var overrides).",
                    "--cpu-moe"),
                new OptionHelp("--cpu-moe-threads <N>",
                    "Worker threads for the host-side expert matmul. Default: one less than the CPU parallelism " +
                    "this process can actually use (hardware threads clamped by the affinity mask and the cgroup " +
                    "CPU quota), leaving a core for accelerator submission. Do not set this above the quota: " +
                    "ggml's pool spins at its barriers, so oversubscription collapses throughput rather than " +
                    "degrading it (TS_CPU_MOE_THREADS env var overrides).",
                    "--cpu-moe-threads 12"),
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
                    "Quantize spilled KV blocks with the TurboQuant codec: 0 (off), 2, 4, or 8 bits per element. " +
                    "2-bit uses an affine min+scale layout (~4x smaller than the f16 payload). Default: 0.",
                    "--paged-kv-quant-bits 8"),
                new OptionHelp("--paged-kv-redis-url <url>",
                    "Redis connection string for a shared KV cache tier (e.g. localhost:6379). Default: disabled.",
                    "--paged-kv-redis-url localhost:6379"),
                new OptionHelp("--paged-kv-redis-ttl <min>",
                    "TTL in minutes for Redis KV entries (0 = no TTL). Default: 1440.",
                    "--paged-kv-redis-ttl 60"),
                new OptionHelp("--redis-url <url>",
                    "Redis connection string for both the KV cache tier and the Responses API store.",
                    "--redis-url localhost:6379"),
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
            ("Speculative decoding (one name per option; the old --mtp-*/--spec-draft-model spellings error with a pointer here)", new[]
            {
                new OptionHelp("--spec | --no-spec",
                    "Enable/disable speculative decoding: a drafter proposes the next few tokens and the trunk " +
                    "verifies them in one batched forward. Every emitted token still comes from a trunk row, so " +
                    "the output is what standard decoding would have produced. Needed only for a drafter EMBEDDED " +
                    "in the checkpoint (Qwen 3.6 / GLM 5.2 NextN) - it pages extra weights into VRAM, so it stays " +
                    "an explicit choice; a drafter named on --draft-model engages by itself. Engages for solo " +
                    "(non-concurrent) sequences. Default: off.",
                    "--spec"),
                new OptionHelp("--spec-type <name>",
                    "Speculation algorithm: 'auto' (default) uses whatever drafter the checkpoint carries; " +
                    "'draft-head' and 'block' pin one explicitly; 'ngram' needs no trained weights at all and " +
                    "works on every model, drafting by suffix match over the context (strong when the answer " +
                    "quotes its input: summarizing, editing, structured output, agentic loops).",
                    "--spec --spec-type ngram"),
                new OptionHelp("--spec-draft <N>",
                    "Maximum draft tokens per step (1-64). Default: 8.",
                    "--spec-draft 4"),
                new OptionHelp("--spec-pmin <f>",
                    "Draft-confidence gate in [0, 1]; drafting stops below it, and 0 disables the gate " +
                    "entirely. What the number means is per algorithm, so each brings its own default — 0.15 " +
                    "for a per-token draft head, 0.35 for a block drafter (where the gate is the CUMULATIVE " +
                    "prefix probability, so the same number means something much stricter), 0 for n-gram.",
                    "--spec-pmin 0.6"),
                new OptionHelp("--draft-model <path>",
                    "A drafter that ships as its own GGUF, whatever its kind: DeepSeek V4's DSpark, the DFlash / " +
                    "DFlash2 drafters for Muse-Glimmer and Qwen 3.8, or Gemma 4's per-token assistant head. " +
                    "The file's own general.architecture decides how it loads (a block drafter is fused before " +
                    "the layer split, a per-token head attaches after) - never its file name, and never a " +
                    "second flag. Naming the file IS the request: speculation turns on with it, no --spec " +
                    "needed, and an explicit --no-spec vetoes it. Qwen 3.6 and GLM 5.2 embed their drafter in " +
                    "the trunk and use --spec instead. Default: none.",
                    "--draft-model Qwen3.8-27B-DFlash2-Q4_K_M.gguf"),
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
                new OptionHelp("--offload-cpu",
                    "Stream the DiT weights from RAM instead of holding them resident in VRAM " +
                    "(sd.cpp --offload-to-cpu equivalent): slower per step, but the freed VRAM lets " +
                    "native ~1 MP edits run on small cards. Default: auto (engages only when the " +
                    "target resolution does not fit beside the resident weights).",
                    "--offload-cpu"),
            }),
            ("Video-generation defaults and companion models", new[]
            {
                new OptionHelp("--video-width <px>",
                    "Default output width when a request omits 'width'. THE main quality lever: " +
                    "the Web UI sends no size of its own, so without this every clip is generated " +
                    "at the model's default. 640x384 is a good starting point for MiniMax-H3; " +
                    "--width is accepted as an alias. Rounded up to the model's grid.",
                    "--video-width 640"),
                new OptionHelp("--video-height <px>",
                    "Default output height when a request omits 'height'. Alias: --height. " +
                    "If only one of width/height is given, MiniMax-H3 takes the other from the " +
                    "conditioning image's aspect ratio.",
                    "--video-height 384"),
                new OptionHelp("--video-steps <N>",
                    "Default denoising steps when a request omits 'steps'. The quality/time " +
                    "trade-off after resolution. The server has NO default of its own - unset, " +
                    "each model uses its own: 20 for MiniMax-H3, and for Wan the checkpoint's " +
                    "trained step count when it is step-distilled, otherwise 50 (TI2V-5B), 40 " +
                    "(A14B) or 30 (Wan 2.1). MiniMax-H3 is step-distilled and CFG-free, so 4-8 " +
                    "is the fast operating point (some chromatic fringing around moving subjects " +
                    "at the low end, gone by ~20), 16-24 is visibly cleaner, and past ~30 gains " +
                    "little. Setting this pins EVERY request, so on a step-distilled Wan " +
                    "checkpoint it overrides the 4 steps the file was trained for - leave it " +
                    "unset there rather than paying 4x the work for a worse result.",
                    "--video-steps 16"),
                new OptionHelp("--video-mode <mode>",
                    "Default conditioning mode when a request omits 'videoMode': t2v (text only), " +
                    "i2v (the image is the first frame and is animated), fl2v (first AND last frame " +
                    "pinned) or ref (the images are identity/appearance references for a new scene). " +
                    "Omit it and the mode is inferred from what each request supplies, which is " +
                    "usually what you want; pin it for a deployment that only offers one.",
                    "--video-mode ref"),
                new OptionHelp("--video-frames <N>",
                    "Default output frame count when a request omits 'frames'. The count is snapped to the " +
                    "model's temporal grid (4k+1 for Wan, 17k+5 for MiniMax-H3). Model default: 33, or 49 for " +
                    "Wan2.2-TI2V. A request value overrides it.",
                    "--video-frames 121"),
                new OptionHelp("--fps <N>",
                    "Default MP4 playback rate when a request omits 'fps'. Model default: 16, or 24 for " +
                    "Wan2.2-TI2V. A request value overrides it; FPS changes playback rate, not generation work. " +
                    "Models trained at a fixed rate (MiniMax-H3, 24 fps) override any other value.",
                    "--fps 24"),
                new OptionHelp("--video-vae <path>",
                    "Video VAE (wan_2.1_vae.safetensors, or Wan2.2_VAE.safetensors for TI2V-5B; " +
                    "minimax_h3_video_vae_fp16.safetensors for MiniMax-H3). Default: same-directory scan next " +
                    "to the DiT model, VAE/ subfolders included (TS_VIDEO_VAE). The former spelling " +
                    "--wan-vae is still accepted.",
                    "--video-vae Wan2.2_VAE.safetensors"),
                new OptionHelp("--video-text-encoder <path>",
                    "Text-encoder GGUF (UMT5-XXL for Wan, Qwen3-VL-32B for MiniMax-H3). Default: " +
                    "same-directory scan (TS_VIDEO_TEXT_ENCODER). Also spelled --video-te; the former " +
                    "spelling --wan-te is still accepted.",
                    "--video-text-encoder umt5-xxl-encoder-Q8_0.gguf"),
                new OptionHelp("--video-dit2 <path>",
                    "Second diffusion expert on dual-expert models (Wan 2.2 A14B's high/low-noise partner of " +
                    "--model). Default: auto-resolved by name from the same or a sibling folder " +
                    "(TS_VIDEO_DIT2); needed only when the pair is not co-located, or to name it in a " +
                    "--config file. The former spelling --wan-dit2 is still accepted.",
                    "--video-dit2 wan2.2_i2v_A14b_low_noise-Q4_K_M.gguf"),
                new OptionHelp("--audio-vae <path>",
                    "Audio VAE for models that generate an audio track jointly with the video " +
                    "(minimax_h3_audio_vae_fp32.safetensors). Without it such a model still runs and produces " +
                    "video, just no audio (TS_VIDEO_AUDIO_VAE).",
                    "--audio-vae minimax_h3_audio_vae_fp32.safetensors"),
            }),
            ("Upload storage (the uploads/ directory next to the server binary)", new[]
            {
                new OptionHelp("--upload-max-mb <N>",
                    "Per-file cap in MB on client-originated writes: multipart /api/upload files and base64 " +
                    "attachments decoded out of chat requests. Default: 500, the request-body limit " +
                    "(TS_UPLOAD_MAX_MB env var overrides).",
                    "--upload-max-mb 25"),
                new OptionHelp("--upload-quota-mb <N>",
                    "Total budget in MB for the upload directory, counting client uploads, decoded attachments, " +
                    "and generated outputs (edited images, videos). Requests that would exceed it are rejected " +
                    "up front — before any model work runs. Default: off (TS_UPLOAD_QUOTA_MB env var overrides).",
                    "--upload-quota-mb 2048"),
                new OptionHelp("--upload-ttl-hours <N>",
                    "Delete upload-directory files older than this many hours (fractions allowed). Default: off, " +
                    "because chat sessions reference attachments by path and may reuse them later; enable it when " +
                    "the server is reachable by untrusted clients (TS_UPLOAD_TTL_HOURS env var overrides).",
                    "--upload-ttl-hours 24"),
            }),
            ("Agent skills (SKILL.md bundles; the skills/ directory next to the server binary)", new[]
            {
                new OptionHelp("--skills-dir <path>",
                    "Directory to scan for skills. A root may hold one skill (it contains SKILL.md) or many, " +
                    "nested up to three levels, so a checkout of a skills repository works as-is. Repeat the " +
                    "flag for several; earlier roots take precedence on a name clash. Default: the skills/ " +
                    "directory next to the binary, created on startup (TS_SKILLS_DIR env var overrides, " +
                    "path-separated).",
                    "--skills-dir ./skills"),
                new OptionHelp("--skill <name>",
                    "Give EVERY request this skill, instead of waiting for a client to name it in the request's " +
                    "\"skills\" array. Repeat the flag for several. A request that names its own skills replaces " +
                    "this selection rather than adding to it, so a client can always narrow it. Default: none.",
                    "--skill pdf --skill xlsx"),
                new OptionHelp("--list-skills",
                    "Print the skill registry and exit, without loading a model: what was found, where it came " +
                    "from, and every directory that looked like a skill but failed to load, with the reason - " +
                    "which is otherwise invisible, since a skill whose SKILL.md will not parse is simply absent.",
                    "--list-skills"),
                new OptionHelp("--no-skills",
                    "Disable Agent Skills entirely: no directory is scanned, the /v1/skills and /api/skills " +
                    "endpoints are not mapped, and a 'skills' field on a chat request is rejected instead of " +
                    "silently ignored. Default: enabled (TS_NO_SKILLS env var overrides).",
                    "--no-skills"),
                new OptionHelp("--skills-no-discovery",
                    "Show a request only the skills it named, instead of also advertising the rest of the " +
                    "registry so the model can pick up one the caller did not think of. A request can override " +
                    "this per call with \"skills_discovery\". Default: discovery on.",
                    "--skills-no-discovery"),
                new OptionHelp("--skills-allow-exec",
                    "Let the model RUN a skill's bundled scripts on this host. This is arbitrary code execution " +
                    "under the server's own account, chosen by a model reading uploaded Markdown - leave it off " +
                    "on any server that accepts skill uploads or is reachable by untrusted clients. Default: off " +
                    "(TS_SKILLS_ALLOW_EXEC env var overrides).",
                    "--skills-allow-exec"),
                new OptionHelp("--skills-sandbox <off|preferred|required>",
                    "How hard to insist on OS isolation for a skill's scripts. required (the default) refuses to " +
                    "run them at all on a host with no sandbox, rather than running them unconfined; preferred " +
                    "sandboxes where it can and runs them anyway where it cannot; off applies only the in-process " +
                    "limits. macOS uses sandbox-exec, Linux requires bubblewrap (bwrap) 0.12.0 or newer " +
                    "(TS_SKILLS_SANDBOX env var overrides).",
                    "--skills-sandbox required"),
                new OptionHelp("--skills-allow-network",
                    "Let a sandboxed skill script reach the network. Denied by default, which is what stops a " +
                    "script that read something it should not from sending it anywhere " +
                    "(TS_SKILLS_ALLOW_NETWORK env var overrides).",
                    "--skills-allow-network"),
                new OptionHelp("--skills-max-rounds <N>",
                    "How many times a model may fetch skill content - or run, read and fix code - in one turn before " +
                    "it must answer (1-64). Each round is a full generation, so this bounds what one " +
                    "malfunctioning request costs. Default: 8, or 24 when --code-exec is on, because " +
                    "writing a file, running it and fixing the traceback takes more steps than reading " +
                    "files. A value set here is used as given, either way " +
                    "(TS_SKILLS_MAX_ROUNDS env var overrides).",
                    "--skills-max-rounds 4"),
            }),
            ("Code execution (the shell tool: commands the MODEL writes, run in a sandbox)", new[]
            {
                new OptionHelp("--code-exec",
                    "Offer the model a 'shell' tool: it types a command, this host runs it in a sandbox, and " +
                    "the model reads back the exit code and everything it printed. This is how the model does " +
                    "all its work with files and code - write one with a heredoc, run it, search it, install " +
                    "what it needs, check its own output - which is the shape Codex and Claude Code use. Files " +
                    "it writes are kept and handed to the user as download links, and the working directory " +
                    "PERSISTS for the rest of that chat session, so one step's output is the next step's input " +
                    "and cd and exported variables survive from call to call (PATH does not, so an activated " +
                    "virtualenv does not stay activated - installed packages are already on the path). One " +
                    "tool comes with it: 'apply_patch', Codex's patch envelope, which creates, updates, deletes " +
                    "and renames SEVERAL files in one all-or-nothing call. It is there because a heredoc " +
                    "rewrites a whole file while a patch changes three lines of it, and because the HOST places " +
                    "the bytes - from anchors it either finds or refuses to guess at - rather than the model " +
                    "retyping a file it half-remembers. Separate from --skills-allow-exec on purpose: that runs " +
                    "a script an operator put on disk, this runs commands written during the request. The " +
                    "sandbox is required, not optional - on a host that cannot confine a process (no " +
                    "sandbox-exec on macOS, no safe bwrap (0.12.0+) on Linux, and Windows job objects, which bound CPU but not " +
                    "files or sockets) the tool refuses to run rather than running unconfined, and says so at " +
                    "startup. The one way past that is --code-exec-unconfined, which the server now accepts as " +
                    "well as the CLI: on Windows there is no confining sandbox to fall back to at all, so " +
                    "without it --code-exec is a flag that can never do anything. It is an explicit statement " +
                    "about the machine the server runs on - do not set it on a server others can reach. " +
                    "Commands cannot reach the network by default; --code-exec-allow-network is the separate " +
                    "explicit opt-in. Package installs do not grant that access: the HOST performs them rather " +
                    "than running the model's install command. Default: off. TS_CODE_EXEC " +
                    "also turns it on, and note the rule: ANY value except 0 counts as on, so " +
                    "TS_CODE_EXEC=false enables it.",
                    "--code-exec"),
                new OptionHelp("--code-exec-allow-install",
                    "Let the model install packages (pip / npm) into an environment the chat session keeps - so " +
                    "a later command, and a skill's own scripts, can import them too. This is the half that " +
                    "touches the network, so it is a separate decision. What makes it safe is that the " +
                    "model's install command is READ, not run: the host takes the tool and the package names " +
                    "out of it, validates the names, and performs the install itself with an argument vector " +
                    "it built - then substitutes the install out of the line, so installing never widens the " +
                    "rest of the command's separately configured network policy. Two holes closed at once, " +
                    "which is why it is done this way: an argument the model wrote can no longer choose the index, and a socket " +
                    "granted for an install can no longer be shared with whatever else was on the line. An " +
                    "option that would change where a package comes from (--index-url, -i, --find-links, " +
                    "--registry, a URL requirement) is refused by name; an installer the host cannot perform " +
                    "(gem, cargo) is refused too; -r requirements.txt is read and each line validated. Only " +
                    "prebuilt wheels are installed and install scripts never run (pip --only-binary=:all:, " +
                    "npm --ignore-scripts), so a package shipping source only fails to install rather than " +
                    "executing its setup.py as this account. Also what lets a skill script's missing " +
                    "dependency be installed automatically instead of failing. Requires --code-exec. " +
                    "Default: off (TS_CODE_EXEC_ALLOW_INSTALL env var overrides).",
                    "--code-exec --code-exec-allow-install"),
                new OptionHelp("--code-exec-allow-network",
                    "Give every model-authored command unrestricted IP network access (subject to the host OS " +
                    "and firewall), so generated code can fetch URLs, follow redirects, use DNS and call " +
                    "remote APIs. This is a separate " +
                    "operator decision from package installs and is off by default, following the same " +
                    "default-deny, explicit-opt-in principle as Codex and Claude's sandbox runtime. This " +
                    "mode grants the full host network rather than their optional domain-filtered proxy mode. " +
                    "Write/home-read confinement remains active on macOS and Linux, subject to macOS's " +
                    "shared /private/tmp read/write compatibility exception. Linux also bounds descendants with a " +
                    "PID namespace; macOS Seatbelt is inherited, but a deliberately detached child may outlive the " +
                    "request and every result reports that gap. Network access " +
                    "raises prompt-injection, data-exfiltration, host-local service and untrusted-download " +
                    "risk; credential-free host proxy settings are passed only in this mode. Custom-CA bundles " +
                    "up to 16 MiB are read once, with only validated public certificates copied into a read-only " +
                    "session snapshot; the source path and adjacent data are not exposed. " +
                    "Authenticated proxies need a credential-free host-side forwarder. " +
                    "Package allow-lists and install-domain settings constrain only TensorSharp's recognised " +
                    "host installer; unrestricted generated code can fetch or execute artifacts directly. " +
                    "On Windows, --code-exec-unconfined is still required because a job object cannot confine filesystem " +
                    "access. Default: off (TS_CODE_EXEC_ALLOW_NETWORK env var overrides).",
                    "--code-exec --code-exec-allow-network"),
                new OptionHelp("--code-exec-unconfined",
                    "Run the model's commands even where the OS cannot confine them. On Windows this is not " +
                    "an edge case but the only way to use --code-exec at all: a job object bounds a process " +
                    "tree's CPU, memory and lifetime and cannot restrict one file or one socket, and " +
                    "TensorSharp's current Windows backend does not yet implement the separate-user/WFP " +
                    "isolation needed to confine a shell. Without this flag --code-exec is inert " +
                    "there. It means model-written commands run with this process's access to the " +
                    "filesystem and the network; the in-process bounds still apply (a workspace-confined " +
                    "working directory, an argument vector rather than a command line, a scrubbed " +
                    "environment, host-performed installs, and time and output caps), and every tool result " +
                    "says which of them are NOT confinement. It is a statement about the machine this " +
                    "server runs on - do not set it on a server others can reach. Default: off.",
                    "--code-exec --code-exec-unconfined"),
                new OptionHelp("--code-exec-packages <list>",
                    "Restrict installs to these package names, comma-separated; anything else is refused and " +
                    "the model is told which names are allowed. Matching is on the bare name, so a version the " +
                    "model pins (numpy==2.1.0) still matches the entry 'numpy'; at most 16 packages are " +
                    "installed at once regardless. This is enforceable for recognised installs because the HOST " +
                    "READS the names out of the model's command rather than running that command, " +
                    "and builds the installer's argument vector itself — so the list applies whether the model " +
                    "wrote pip, python -m pip, or a requirements file. With unrestricted command networking this " +
                    "is not a security boundary, because generated code can fetch or execute artifacts directly. Only meaningful with " +
                    "--code-exec-allow-install. Default: empty, meaning any package may be installed.",
                    "--code-exec-packages numpy,pandas,reportlab"),
                new OptionHelp("--code-exec-install-index <url>",
                    "Package index installs are pointed at, instead of the tool's default. The host applies it - " +
                    "an --index-url the MODEL writes is still refused, because an argument the model wrote must " +
                    "never choose where a package comes from. This is for a host that cannot reach pypi.org: a " +
                    "corporate proxy, an internal mirror, or a network that filters TLS by SNI, where every " +
                    "install otherwise dies with a bare SSLError and there is no way out. The index's own host " +
                    "is admitted through the egress allowlist automatically; if the mirror serves downloads " +
                    "from a second hostname, name that one in --code-exec-install-domains as well. Default: " +
                    "unset (TS_CODE_EXEC_INSTALL_INDEX env var overrides).",
                    "--code-exec-install-index https://pypi.example.com/simple"),
                new OptionHelp("--code-exec-install-domains <list>",
                    "The hosts a host-performed install may reach, comma-separated; exact names or *.suffix " +
                    "wildcards. A loopback CONNECT proxy holds the list and the installer is pointed at it " +
                    "through HTTPS_PROXY. On macOS the sandbox admits exactly that one port, so every other " +
                    "destination is unreachable at the OS level; elsewhere the installer follows the proxy " +
                    "because pip and npm honour that variable - obedience rather than confinement, with what " +
                    "still holds being that the host built the argument vector, so nothing in it points " +
                    "elsewhere. Exact names by default and no wildcards, because '*.pypi.org' would also admit " +
                    "upload.pypi.org. An EMPTY value disables the pinning entirely. Default: pypi.org," +
                    "files.pythonhosted.org,registry.npmjs.org (TS_CODE_EXEC_INSTALL_DOMAINS env var " +
                    "overrides; a value set with this flag wins over it).",
                    "--code-exec-install-domains pypi.org,files.pythonhosted.org"),
                new OptionHelp("--code-exec-timeout <seconds>",
                    "How long one command may run before it is killed, in whole seconds; must be a positive " +
                    "number. Bounds what a single runaway command costs. A call may ask for less or more, up to " +
                    "10 minutes; a command that runs over is stopped and the model still gets everything it " +
                    "printed first, because a timeout that discards output makes it re-run the command blind. " +
                    "Default: 120.",
                    "--code-exec-timeout 300"),
                new OptionHelp("--code-exec-temperature <number>",
                    "Sampling temperature for a turn in which the model can run code, between 0 and 2; a " +
                    "negative number leaves sampling alone entirely. The built-in defaults are Ollama's CHAT " +
                    "defaults - temperature 0.8, top_k 40, top_p 0.9, repeat_penalty 1.1 over a 64-token " +
                    "window - inherited for API compatibility and never chosen for code. That penalty window " +
                    "is two to four lines of Python, so it penalises the indentation, the 'self.', the " +
                    "'return' and the closing delimiters that carry the code's structure against each other; " +
                    "this setting turns it off as well as setting the temperature. Only values still sitting " +
                    "at the built-in default are changed, so a temperature a client sent or an operator " +
                    "pinned is never overruled. Applies to the whole turn, the final written answer " +
                    "included. No TEMPERATURE is set by default, deliberately: the Agents SDK leaves " +
                    "temperature, top_p and both penalties unset and omits them from the request, and " +
                    "Claude Code exposes no sampling setting at all, so there is no precedent for adding " +
                    "one and it is offered as a switch instead. The repetition PENALTY is removed for " +
                    "coding turns regardless, because that REMOVES an Ollama chat default which neither " +
                    "reference has any analogue of. Default: not set.",
                    "--code-exec-temperature 0.1"),
                new OptionHelp("--code-exec-shell <path|name>",
                    "Which shell to run commands through, when the host's own choice is wrong or absent. The " +
                    "default is bash (then sh) on macOS and Linux, and PowerShell on Windows - where a bare " +
                    "'bash' on PATH is refused on purpose, because it is the WSL launcher and would run the " +
                    "command inside a Linux VM that the job object holding the launcher cannot reach. Point " +
                    "this at a real bash (Git Bash, MSYS2) to use one there. The model is told which dialect it " +
                    "is typing into, so this also changes the examples in its tool description. Default: chosen " +
                    "by platform.",
                    "--code-exec-shell /opt/homebrew/bin/bash"),
                new OptionHelp("--code-exec-max-output <bytes>",
                    "How much of one command's output is kept and shown to the model. What is dropped comes out " +
                    "of the MIDDLE, keeping the head and the tail: the end of a build or a test run is where " +
                    "the failure is, and head-only truncation reliably discards exactly the part that was " +
                    "wanted. Default: 32768.",
                    "--code-exec-max-output 65536"),
            }),
            ("Configuration file", new[]
            {
                new OptionHelp("--config <path>",
                    "Read options from a JSON file whose keys are the same long option names listed here (with or " +
                    "without the leading --). Anything also passed on the command line overrides the file; when the " +
                    "flag is repeated, later files win over earlier ones. String/number values map to '--key value', " +
                    "true maps to the bare '--key' switch, and an array maps to a repeated flag (e.g. \"stop\": [..]). " +
                    "A \"variables\" object lets values share ${name} references; a file option may instead be an " +
                    "object { \"path\": \"...\", \"urls\": [ \"...\" ] } that auto-downloads on first run. See the " +
                    "config/ folder and config/README.md for examples.",
                    "--config server.json --backend ggml_cuda"),
            }),
            ("Help", new[]
            {
                new OptionHelp("--help",
                    "Show this help and exit (also shown when the server is started with no arguments).",
                    "--help"),
            }),
        };

        /// <summary>
        /// Every flag token named on the usage page, placeholders stripped.
        ///
        /// Exists so a test can assert the page and the parser agree. They drifted
        /// twice: <c>--wan-vae</c>/<c>--wan-te</c> and later every <c>--spec*</c>
        /// spelling were documented here while ServerOptionsBuilder.ParseArgs
        /// rejected them as unknown options, so the server refused to start on a
        /// flag its own <c>--help</c> advertised.
        ///
        /// Placeholders are removed BEFORE splitting on '|', because a value
        /// placeholder can itself contain one (<c>--mmproj &lt;path|none&gt;</c>,
        /// <c>--sampling-precedence &lt;config|request&gt;</c>).
        /// </summary>
        /// <summary>
        /// Print the skill registry for <c>--list-skills</c>: what loaded, where it came
        /// from, and — the reason the command earns its keep — every directory that
        /// looked like a skill and did not load, with the reason. A skill whose
        /// <c>SKILL.md</c> will not parse is otherwise simply absent from every listing,
        /// which is the hardest kind of problem for its author to diagnose.
        /// </summary>
        public static void PrintSkills(TextWriter writer, SkillRegistry registry, bool enabled)
        {
            if (!enabled || registry == null)
            {
                writer.WriteLine("Agent skills are disabled (--no-skills / TS_NO_SKILLS).");
                return;
            }

            writer.WriteLine($"Skill roots: {string.Join(", ", registry.Roots)}");
            writer.WriteLine();

            if (registry.Skills.Count == 0)
            {
                writer.WriteLine("No skills found.");
                writer.WriteLine();
                writer.WriteLine("A skill is a directory containing SKILL.md. Drop one under a root above,");
                writer.WriteLine("point at your own with --skills-dir <path>, or POST a .zip to /api/skills.");
            }
            else
            {
                writer.WriteLine($"{registry.Skills.Count} skill(s):");
                writer.WriteLine();
                foreach (Skill skill in registry.Skills)
                {
                    string origin = skill.Origin == SkillOrigin.Installed ? "installed" : "discovered";
                    writer.WriteLine($"  {skill.Id}  [{origin}]");
                    writer.WriteLine($"      {SkillTextBudget.Truncate(skill.Description, 300)}");
                    int bundled = System.Linq.Enumerable.Count(skill.BundledFiles);
                    writer.WriteLine(
                        $"      {bundled} bundled file(s), {SkillTextBudget.FormatBytes(skill.TotalBytes)}, "
                        + $"~{skill.Manifest.ApproximateBodyTokens} tokens of instructions");
                    foreach (string warning in skill.Manifest.Warnings)
                        writer.WriteLine($"      warning: {warning}");
                    writer.WriteLine();
                }
            }

            if (registry.Errors.Count > 0)
            {
                writer.WriteLine($"{registry.Errors.Count} directory/directories could not be loaded:");
                foreach (SkillLoadError error in registry.Errors)
                    writer.WriteLine($"  {error.Path}: {error.Message}");
                writer.WriteLine();
            }

            writer.WriteLine("Select one per request with \"skills\": [\"<name>\"], or for every request with --skill <name>.");
        }

        internal static IEnumerable<string> DocumentedFlags()
        {
            foreach (var (_, options) in Sections)
            {
                foreach (var opt in options)
                {
                    string flag = opt.Flag;
                    int lt;
                    while ((lt = flag.IndexOf('<')) >= 0)
                    {
                        int gt = flag.IndexOf('>', lt);
                        if (gt < 0) { flag = flag.Substring(0, lt); break; }
                        flag = flag.Remove(lt, gt - lt + 1);
                    }
                    foreach (string part in flag.Split('|'))
                    {
                        string token = part.Trim();
                        if (token.StartsWith("--", StringComparison.Ordinal))
                            yield return token;
                    }
                }
            }
        }

        public static void PrintUsage(TextWriter writer)
        {
            writer.WriteLine("Usage: TensorSharp.Server [options]");
            writer.WriteLine();
            writer.WriteLine("Hosts an OpenAI- and Ollama-compatible inference server (plus a built-in web chat UI)");
            writer.WriteLine("on http://0.0.0.0:5000 by default (change it with --port / --host). Run with no");
            writer.WriteLine("arguments to show this help; pass at least one option to start the server.");

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
            writer.WriteLine("  TensorSharp.Server --model C:\\models\\gemma-4-E4B-it-Q8_0.gguf --backend ggml_cpu");
            writer.WriteLine("  TensorSharp.Server --model gemma-4-E4B-it-Q8_0.gguf --mmproj mmproj-gemma-4-E4B-it-Q8_0.gguf --backend ggml_cuda");
            writer.WriteLine("  TensorSharp.Server --model Qwen3.5-35B-A3B-Q4_K_M.gguf --backend ggml_cuda --tp 2    (split across 2 GPUs)");
            writer.WriteLine("  TensorSharp.Server --model Wan2.2-TI2V-5B-Q8_0.gguf --backend ggml_cuda --video-frames 121 --fps 24");
            writer.WriteLine("  TensorSharp.Server --backend ggml_cpu    (model-less status process; inference unavailable)");
            writer.WriteLine("  TensorSharp.Server --config server.json    (read options from a file)");
            writer.WriteLine("  TensorSharp.Server --config server.json --backend ggml_cuda    (file, but override the backend)");
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
