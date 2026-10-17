// Copyright (c) Zhongkai Fu. All rights reserved.
// https://github.com/zhongkaifu/TensorSharp
//
// This file is part of TensorSharp.
//
// TensorSharp is licensed under the BSD-3-Clause license found in the LICENSE file in the root directory of this source tree.
using System;
using System.IO;
using System.Linq;
using TensorSharp;
using TensorSharp.Core;
using TensorSharp.Runtime;
using TensorSharp.Models.Video;

namespace TensorSharp.Models.WanVideo
{
    /// <summary>Which Wan generation/checkpoint family a DiT GGUF belongs to.</summary>
    public enum WanVariant
    {
        /// <summary>Wan 2.1 / 2.2 single-DiT text-to-video (16-channel latent, Wan 2.1 VAE).</summary>
        T2V,
        /// <summary>Wan 2.2 A14B two-expert MoE (high-noise + low-noise DiT GGUFs switched at a
        /// timestep boundary; 16-channel T2V or 36-channel I2V input, Wan 2.1 VAE).</summary>
        A14B,
        /// <summary>Wan 2.2 TI2V-5B dense text/image-to-video (48-channel latent, Wan 2.2 VAE,
        /// 16x16x4 compression, 24 fps).</summary>
        TI2V,
    }

    /// <summary>
    /// Wan 2.1 / 2.2 video-generation model. Orchestrates networks that the DiT GGUF
    /// (<c>general.architecture = wan</c>) does <b>not</b> itself contain:
    /// <list type="bullet">
    ///   <item>the Wan DiT diffusion transformer (the loaded <c>_gguf</c>; Wan 2.2 A14B
    ///     models additionally load the second high/low-noise expert GGUF),</item>
    ///   <item>the Wan causal 3D video VAE (wan_2.1_vae.safetensors, or Wan2.2_VAE.safetensors
    ///     for the TI2V-5B model), and</item>
    ///   <item>the UMT5-XXL text encoder (umt5-xxl-encoder GGUF).</item>
    /// </list>
    /// Companions resolve next to the DiT GGUF or via the <c>TS_VIDEO_VAE</c> /
    /// <c>TS_VIDEO_TEXT_ENCODER</c> / <c>TS_VIDEO_DIT2</c> environment variables (what
    /// <c>--video-vae</c> / <c>--video-text-encoder</c> / <c>--video-dit2</c> set). Not an autoregressive
    /// text model; generation is driven through <see cref="GenerateVideo"/>.
    /// Text-to-video works on every variant; supplying <see cref="VideoGenerationParams.Image"/>
    /// generates image-to-video on the Wan 2.2 models (TI2V-5B / I2V-A14B).
    /// </summary>
    public sealed class WanVideoModel : ModelBase, IVideoGenerationModel
    {
        private readonly string _ditPath;
        private readonly string _vaePath;
        private readonly string _tePath;
        private readonly string _dit2Path;   // A14B: the second expert GGUF (or null)
        private GgufFile _gguf2;

        internal GgufFile DitGguf => _gguf;
        internal string VaePath => _vaePath;
        internal string TePath => _tePath;
        internal BackendType Backend => _backend;
        internal IAllocator DirectAllocator => _allocator;

        /// <summary>Checkpoint family (see <see cref="WanVariant"/>).</summary>
        public WanVariant Variant { get; }

        /// <summary>Latent channels the DiT denoises (16, or 48 for TI2V-5B).</summary>
        public int LatentChannels { get; }

        /// <summary>DiT input channels (36 for the A14B I2V conditioning concat).</summary>
        public int InputChannels { get; }

        /// <summary>Pixels per latent cell (8, or 16 for TI2V-5B).</summary>
        public int VaeSpatialScale { get; }

        /// <summary>True when generation needs a conditioning image (A14B I2V checkpoints).</summary>
        public bool RequiresImage => Variant == WanVariant.A14B && InputChannels == 36;

        /// <summary>True when the model accepts an optional conditioning image.</summary>
        public bool SupportsImage => Variant == WanVariant.TI2V || RequiresImage;

        /// <summary>A14B two-expert GGUF pair (high-noise expert active above the boundary).</summary>
        internal GgufFile HighNoiseGguf { get; }
        internal GgufFile LowNoiseGguf { get; }

        /// <summary>
        /// Denoising steps a step-distilled checkpoint was trained for; 0 for an
        /// ordinary one. Step distillation (Wan2.2-TI2V-5B-Turbo, Wan2.2-Lightning,
        /// lightx2v, FastWan/DMD) is the difference between a 5-second 720p video
        /// costing 100 DiT passes and costing 4, so the pipeline has to notice — run
        /// a distilled checkpoint on the 50-step CFG recipe and it is 25x slower for
        /// a worse picture, since these models are also trained to run WITHOUT
        /// classifier-free guidance.
        /// </summary>
        public int DistilledSteps { get; }

        /// <summary>
        /// Steps a distilled Wan checkpoint/LoRA advertises in its file name.
        /// "…-4steps-…" / "…8step…" wins; otherwise a distillation marker implies the
        /// 4 steps every published Wan 2.2 distillation recommends.
        /// </summary>
        internal static int ParseDistilledSteps(string fileName)
        {
            if (string.IsNullOrEmpty(fileName)) return 0;
            string n = Path.GetFileNameWithoutExtension(fileName).ToLowerInvariant();
            bool marker = n.Contains("turbo") || n.Contains("distill") || n.Contains("lightning")
                          || n.Contains("lightx2v") || n.Contains("fastwan") || n.Contains("-dmd");
            var m = System.Text.RegularExpressions.Regex.Match(n, @"(\d+)\s*_?steps?\b");
            if (m.Success && int.TryParse(m.Groups[1].Value, out int s) && s > 0 && s <= 16)
                return s;
            return marker ? 4 : 0;
        }

        public WanVideoModel(string ggufPath, BackendType backend) : base(ggufPath, backend)
        {
            if (backend is not (BackendType.GgmlCuda or BackendType.GgmlCpu or BackendType.GgmlMetal
                or BackendType.GgmlVulkan or BackendType.Cuda or BackendType.Cpu))
                throw new NotSupportedException(
                    "Wan video generation runs on the GGML backends (ggml_cuda, ggml_cpu, ggml_metal, " +
                    $"ggml_vulkan) and the direct cuda / cpu backends; backend '{backend}' is not supported. " +
                    "Pass e.g. --backend ggml_cuda.");

            _ditPath = ggufPath;
            Config = new ModelConfig
            {
                Architecture = "wan",
                VocabSize = 0,
            };

            // The 5D patch embedding [kw, kh, kd, ic, oc] tells the family apart:
            // ic 16 = T2V latent, 36 = A14B I2V (16 latent + 4 mask + 16 image latent),
            // 48 = TI2V-5B's high-compression latent.
            if (!_gguf.Tensors.TryGetValue("patch_embedding.weight", out var patch) || patch.Shape.Length < 5)
                throw new InvalidDataException("Wan DiT GGUF has no 5D patch_embedding.weight tensor.");
            InputChannels = (int)patch.Shape[3];

            string dir = Path.GetDirectoryName(Path.GetFullPath(ggufPath)) ?? ".";
            switch (InputChannels)
            {
                case 48:
                    Variant = WanVariant.TI2V;
                    LatentChannels = 48;
                    VaeSpatialScale = 16;
                    break;
                case 36:
                case 16:
                    LatentChannels = 16;
                    VaeSpatialScale = 8;
                    _dit2Path = ResolveSecondExpert(ggufPath, dir);
                    Variant = _dit2Path != null || InputChannels == 36 ? WanVariant.A14B : WanVariant.T2V;
                    break;
                default:
                    throw new NotSupportedException(
                        $"Unsupported Wan DiT input width {InputChannels} (expected 16, 36 or 48 channels).");
            }

            if (InputChannels == 36 && _dit2Path == null)
                throw new FileNotFoundException(
                    "Wan2.2-I2V-A14B needs BOTH expert GGUFs (…HighNoise… and …LowNoise…). Put them in " +
                    "the same directory (or HighNoise/ + LowNoise/ subdirectories, the QuantStack layout), " +
                    "or name the second expert with --video-dit2 (TS_VIDEO_DIT2).");

            // A14B: assign the loaded + companion GGUFs to their denoising phases by name.
            if (_dit2Path != null)
            {
                _gguf2 = new GgufFile(_dit2Path);
                bool loadedIsHigh = IsHighNoiseName(Path.GetFileName(ggufPath));
                HighNoiseGguf = loadedIsHigh ? _gguf : _gguf2;
                LowNoiseGguf = loadedIsHigh ? _gguf2 : _gguf;
            }

            bool wan22Vae = Variant == WanVariant.TI2V;
            _vaePath = ResolveCompanion("TS_VIDEO_VAE", dir,
                wan22Vae
                    ? new[] { "Wan2.2_VAE.safetensors", "wan2.2_vae.safetensors", Path.Combine("VAE", "Wan2.2_VAE.safetensors") }
                    : new[] { "wan_2.1_vae.safetensors", "wan2.1_vae.safetensors", Path.Combine("VAE", "wan_2.1_vae.safetensors") },
                n => n.Contains("vae") && n.EndsWith(".safetensors")
                     && n.Contains("2.2") == wan22Vae);
            _tePath = ResolveCompanion("TS_VIDEO_TEXT_ENCODER", dir,
                new[] { "umt5-xxl-encoder-Q8_0.gguf" },
                n => (n.Contains("umt5") || n.Contains("t5xxl") || n.Contains("t5-xxl")) && n.EndsWith(".gguf"));

            DistilledSteps = ParseDistilledSteps(Path.GetFileName(ggufPath));

            Console.WriteLine($"Wan video ({Variant}): DiT={Path.GetFileName(ggufPath)}");
            if (DistilledSteps > 0)
                Console.WriteLine($"  step-distilled checkpoint detected -> {DistilledSteps} steps, guidance off " +
                                  $"(--diffusion-steps / --cfg override)");
            if (_dit2Path != null)
                Console.WriteLine($"  expert #2    = {_dit2Path}");
            Console.WriteLine($"  VAE          = {_vaePath ?? "<missing>"}");
            Console.WriteLine($"  text-encoder = {_tePath ?? "<missing>"}");

            if (_vaePath == null || _tePath == null)
                throw new FileNotFoundException(
                    "Wan video generation needs companion models next to the DiT GGUF (or via --video-vae / --video-text-encoder): " +
                    (wan22Vae ? "Wan2.2_VAE.safetensors (Wan2.2 TI2V VAE)" : "wan_2.1_vae.safetensors (Comfy-Org/Wan_2.1_ComfyUI_repackaged)") +
                    " and a umt5-xxl encoder GGUF (city96/umt5-xxl-encoder-gguf).");
        }

        private static bool IsHighNoiseName(string name)
        {
            string n = name.ToLowerInvariant().Replace("_", "").Replace("-", "");
            return n.Contains("highnoise");
        }

        private static bool IsLowNoiseName(string name)
        {
            string n = name.ToLowerInvariant().Replace("_", "").Replace("-", "");
            return n.Contains("lownoise");
        }

        // Wan 2.2 A14B ships as two expert GGUFs. Given one, find the other: the same
        // file name with HighNoise <-> LowNoise swapped, searched in the same directory
        // and in sibling HighNoise/ / LowNoise/ directories (the QuantStack repo layout).
        private static string ResolveSecondExpert(string ggufPath, string dir)
        {
            string env = Environment.GetEnvironmentVariable("TS_VIDEO_DIT2");
            if (!string.IsNullOrWhiteSpace(env) && File.Exists(env)) return env;

            string name = Path.GetFileName(ggufPath);
            bool isHigh = IsHighNoiseName(name);
            bool isLow = IsLowNoiseName(name);
            if (!isHigh && !isLow) return null;

            var swaps = isHigh
                ? new (string, string)[] { ("HighNoise", "LowNoise"), ("highnoise", "lownoise"), ("high_noise", "low_noise"), ("high-noise", "low-noise"), ("High_Noise", "Low_Noise") }
                : new (string, string)[] { ("LowNoise", "HighNoise"), ("lownoise", "highnoise"), ("low_noise", "high_noise"), ("low-noise", "high-noise"), ("Low_Noise", "High_Noise") };

            string parent = Path.GetDirectoryName(dir);
            var searchDirs = new System.Collections.Generic.List<string> { dir };
            if (parent != null)
            {
                // QuantStack layout: HighNoise/model.gguf next to LowNoise/model.gguf.
                foreach (var sub in Directory.Exists(parent) ? Directory.GetDirectories(parent) : Array.Empty<string>())
                    if (!string.Equals(sub, dir, StringComparison.OrdinalIgnoreCase))
                        searchDirs.Add(sub);
            }

            foreach (var (from, to) in swaps)
            {
                if (!name.Contains(from, StringComparison.Ordinal)) continue;
                string sibling = name.Replace(from, to);
                foreach (var d in searchDirs)
                {
                    string p = Path.Combine(d, sibling);
                    if (File.Exists(p)) return p;
                }
            }

            // Fall back to any GGUF in the search dirs whose name carries the opposite tag.
            foreach (var d in searchDirs)
            {
                if (!Directory.Exists(d)) continue;
                foreach (var f in Directory.EnumerateFiles(d, "*.gguf"))
                {
                    string fn = Path.GetFileName(f);
                    if (string.Equals(f, ggufPath, StringComparison.OrdinalIgnoreCase)) continue;
                    if (isHigh ? IsLowNoiseName(fn) : IsHighNoiseName(fn)) return f;
                }
            }
            return null;
        }

        private static string ResolveCompanion(string envVar, string dir, string[] preferred, Func<string, bool> match)
        {
            string env = Environment.GetEnvironmentVariable(envVar);
            if (!string.IsNullOrWhiteSpace(env) && File.Exists(env)) return env;
            // Search the DiT's directory, then its parent (the A14B experts commonly sit
            // in HighNoise/ + LowNoise/ subdirectories next to the VAE and text encoder).
            string parent = Path.GetDirectoryName(dir);
            foreach (var d in new[] { dir, parent })
            {
                if (d == null) continue;
                foreach (var name in preferred)
                {
                    string p = Path.Combine(d, name);
                    if (File.Exists(p)) return p;
                }
            }
            foreach (var d in new[] { dir, parent })
            {
                if (d == null || !Directory.Exists(d)) continue;
                foreach (var f in Directory.EnumerateFiles(d, "*", SearchOption.AllDirectories).OrderBy(f => f.Length))
                {
                    if (match(Path.GetFileName(f).ToLowerInvariant())) return f;
                }
            }
            return null;
        }

        /// <summary>Generate a video from a text prompt (plus an optional conditioning image
        /// via <see cref="VideoGenerationParams.Image"/> on the Wan 2.2 models).</summary>
        public GeneratedVideo GenerateVideo(string prompt, VideoGenerationParams p = null)
        {
            var pipeline = GetPipeline();
            return pipeline.Generate(prompt, p ?? new VideoGenerationParams());
        }

        private WanVideoPipeline _pipeline;
        private WanVideoPipeline GetPipeline() => _pipeline ??= new WanVideoPipeline(this);

        // ---- IVideoGenerationModel capabilities ----
        // Wan is video-only and takes at most a first-frame image; the reference-conditioning
        // and joint-audio paths belong to other families (MiniMax-H3).
        string IVideoGenerationModel.VideoModelFamily => "wan";
        int IVideoGenerationModel.DefaultPreviewFrames => 21;
        bool IVideoGenerationModel.SupportsAudio => false;
        // A T2V-only checkpoint refuses an image, so it must not invite one (chat would offer a photo that fails).
        bool IVideoGenerationModel.SupportsImageConditioning => SupportsImage;
        bool IVideoGenerationModel.SupportsEndImageConditioning => false;
        bool IVideoGenerationModel.SupportsReferenceConditioning => false;

        // ---- IModelArchitecture autoregressive surface: not applicable ----
        protected override float[] ForwardCore(int[] tokens) =>
            throw new NotSupportedException("WanVideoModel is a video-generation model; use GenerateVideo().");

        protected override void ResetKVCacheCore() { }

        // The diffusion networks load lazily on the first GenerateVideo call.
        public override void WarmUpKernels() { }

        public override void Dispose()
        {
            _pipeline?.Dispose();
            _gguf2?.Dispose();
            _gguf2 = null;
            base.Dispose();
        }
    }
}
