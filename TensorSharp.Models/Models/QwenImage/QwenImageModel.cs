// Copyright (c) Zhongkai Fu. All rights reserved.
// https://github.com/zhongkaifu/TensorSharp
//
// This file is part of TensorSharp.
//
// TensorSharp is licensed under the BSD-3-Clause license found in the LICENSE file in the root directory of this source tree.
using System;
using System.IO;
using TensorSharp.Core;
using TensorSharp.Runtime;

namespace TensorSharp.Models.QwenImage
{
    /// <summary>
    /// Qwen-Image-Edit-2511 image-editing model. Unlike the autoregressive LLMs this
    /// orchestrates three networks that the diffusion-transformer GGUF
    /// (<c>general.architecture = qwen_image</c>) does <b>not</b> itself contain:
    /// <list type="bullet">
    ///   <item>the MMDiT diffusion transformer (the loaded <c>_gguf</c>),</item>
    ///   <item>the Qwen-Image VAE (image &lt;-&gt; 16-channel latent), and</item>
    ///   <item>the Qwen2.5-VL-7B text encoder (prompt -&gt; 3584-dim conditioning).</item>
    /// </list>
    /// The companion GGUFs are resolved next to the DiT GGUF (or via the
    /// <c>TS_QWEN_IMAGE_VAE</c> / <c>TS_QWEN_IMAGE_TE</c> / <c>TS_QWEN_IMAGE_MMPROJ</c>
    /// environment variables). This class is not an <see cref="IModelArchitecture"/>
    /// text generator; the autoregressive entry points throw, and image editing is
    /// driven through <see cref="EditImage"/>.
    /// </summary>
    public sealed partial class QwenImageModel : ModelBase
    {
        // ---- DiT architecture constants (the qwen_image GGUF carries no hyperparams) ----
        public const int DitHiddenSize = 3072;
        public const int DitNumLayers = 60;
        public const int DitNumHeads = 24;
        public const int DitHeadDim = 128;            // 24 * 128 = 3072
        public const int DitInChannels = 64;          // 16 latent channels * 2x2 patch
        public const int DitTextDim = 3584;           // Qwen2.5-VL hidden
        public const int VaeLatentChannels = 16;
        public const int VaeScaleFactor = 8;          // spatial 8x downsample
        public static readonly int[] AxesDimsRope = { 16, 56, 56 };  // sums to head_dim 128
        public const float DitEps = 1e-6f;

        private readonly string _vaePath;
        private readonly string _tePath;
        private readonly string _mmprojPath;   // optional (vision grounding, Stage 4)

        private GgufFile _vaeGguf;
        private GgufFile _teGguf;

        private readonly string _ditPath;
        /// <summary>The diffusion-transformer GGUF (alias of the base <c>_gguf</c>).</summary>
        internal GgufFile DitGguf => _gguf;
        internal string DitGgufPath => _ditPath;
        internal GgufFile VaeGguf => _vaeGguf ??= new GgufFile(_vaePath);
        internal GgufFile TeGguf => _teGguf ??= new GgufFile(_tePath);
        internal string MmprojPath => _mmprojPath;
        internal string TePath => _tePath;
        internal string VaePath => _vaePath;
        internal BackendType Backend => _backend;

        public QwenImageModel(string ggufPath, BackendType backend) : base(ggufPath, backend)
        {
            _ditPath = ggufPath;
            Config = new ModelConfig
            {
                Architecture = "qwen_image",
                HiddenSize = DitHiddenSize,
                NumLayers = DitNumLayers,
                NumHeads = DitNumHeads,
                NumKVHeads = DitNumHeads,
                Eps = DitEps,
                VocabSize = 0,
            };

            string dir = Path.GetDirectoryName(Path.GetFullPath(ggufPath)) ?? ".";
            _vaePath = ResolveCompanion("TS_QWEN_IMAGE_VAE", dir,
                "qwen_image_vae.gguf", n => n.Contains("vae") && n.EndsWith(".gguf"));
            _tePath = ResolveCompanion("TS_QWEN_IMAGE_TE", dir,
                "Qwen2.5-VL-7B-Instruct-UD-IQ2_XXS.gguf",
                n => (n.Contains("qwen2.5-vl") || n.Contains("qwen2_5_vl")) && !n.Contains("mmproj") && n.EndsWith(".gguf"));
            _mmprojPath = ResolveCompanionOptional("TS_QWEN_IMAGE_MMPROJ", dir,
                n => n.Contains("mmproj") && (n.Contains("qwen2") || n.Contains("qwen-image")) && n.EndsWith(".gguf"));

            Console.WriteLine($"Qwen-Image-Edit: DiT={Path.GetFileName(ggufPath)}");
            Console.WriteLine($"  VAE          = {_vaePath ?? "<missing>"}");
            Console.WriteLine($"  text-encoder = {_tePath ?? "<missing>"}");
            Console.WriteLine($"  mmproj       = {_mmprojPath ?? "<none> (text-only grounding)"}");

            if (_vaePath == null || _tePath == null)
                throw new FileNotFoundException(
                    "Qwen-Image-Edit needs companion VAE + Qwen2.5-VL text-encoder GGUFs. " +
                    "Place them next to the DiT GGUF or set TS_QWEN_IMAGE_VAE / TS_QWEN_IMAGE_TE.");
        }

        private static string ResolveCompanion(string envVar, string dir, string preferred, Func<string, bool> match)
        {
            string env = Environment.GetEnvironmentVariable(envVar);
            if (!string.IsNullOrWhiteSpace(env) && File.Exists(env)) return env;
            string pref = Path.Combine(dir, preferred);
            if (File.Exists(pref)) return pref;
            return ResolveCompanionOptional(null, dir, match);
        }

        private static string ResolveCompanionOptional(string envVar, string dir, Func<string, bool> match)
        {
            if (envVar != null)
            {
                string env = Environment.GetEnvironmentVariable(envVar);
                if (!string.IsNullOrWhiteSpace(env) && File.Exists(env)) return env;
            }
            if (!Directory.Exists(dir)) return null;
            foreach (var f in Directory.EnumerateFiles(dir, "*.gguf"))
            {
                if (match(Path.GetFileName(f).ToLowerInvariant())) return f;
            }
            return null;
        }

        /// <summary>
        /// Edit <paramref name="input"/> per the natural-language <paramref name="prompt"/>,
        /// returning the modified image. The heavy lifting lives in
        /// <see cref="QwenImagePipeline"/> (added incrementally across the VAE / text-encoder
        /// / DiT stages).
        /// </summary>
        public RgbImage EditImage(string prompt, RgbImage input, QwenImageParams p)
        {
            var pipeline = GetPipeline();
            return pipeline.Edit(prompt, input, p ?? new QwenImageParams());
        }

        private QwenImagePipeline _pipeline;
        private QwenImagePipeline GetPipeline() => _pipeline ??= new QwenImagePipeline(this);

        // ---- IModelArchitecture autoregressive surface: not applicable to an image model ----
        public override float[] Forward(int[] tokens) =>
            throw new NotSupportedException("QwenImageModel is an image-editing model; use EditImage().");

        public override void ResetKVCache() { /* no autoregressive KV cache */ }

        // The base warmup runs a dummy autoregressive Forward(); skip it for the image model
        // (the diffusion nets are loaded lazily on the first EditImage call).
        public override void WarmUpKernels() { }

        public override void Dispose()
        {
            _pipeline?.Dispose();
            _vaeGguf?.Dispose();
            _teGguf?.Dispose();
            base.Dispose();
        }
    }
}
