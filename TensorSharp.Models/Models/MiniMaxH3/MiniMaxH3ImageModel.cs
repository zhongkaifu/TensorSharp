// Copyright (c) Zhongkai Fu. All rights reserved.
// https://github.com/zhongkaifu/TensorSharp
//
// This file is part of TensorSharp.
//
// TensorSharp is licensed under the BSD-3-Clause license found in the LICENSE file in the root directory of this source tree.
//
// MiniMax-H3-Image: text-to-image with the H3 DiT (33B, 50 blocks) without its audio
// stream, the H3 Qwen3-VL-32B text encoder, and an image VAE whose 2-D ViT decoder
// reads the H3 latent space. Distilled to six Euler steps with guidance 1 on the
// checkpoint's own sigma grid (MiniMaxH3ImageSampling), about 16 MP natively and
// 4 MP in ComfyUI's template.
//
//   TensorSharp.Cli --config config/minimax-h3-image.json --prompt "..." --output out.png
//
// The denoiser is read from the safetensors release in its original names
// (Comfy-Org/MiniMax-H3-Image, diffusion_models/minimax_h3_image_pruned_bf16.safetensors)
// or from a GGUF with the same tensors. The text encoder is the H3 video model's
// (--video-text-encoder) and the VAE is passed as --video-vae, the H3 companion flags.
using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.IO;
using System.Linq;
using TensorSharp.Models.QwenImage;
using TensorSharp.Runtime;

namespace TensorSharp.Models.MiniMaxH3
{
    public sealed class MiniMaxH3ImageModel : ModelBase, IImageGenerationModel
    {
        public const string ArchitectureId = "minimax-h3-image";

        /// <summary>The size a request that names none gets: ComfyUI's template default
        /// (2048 x 2048, 4 MP). The model's native size is about 16 MP (4096 x 4096).</summary>
        public const int DefaultSize = 2048;
        /// <summary>Canvas sides are multiples of 32: the 16x VAE and the 2x2 patch.</summary>
        public const int SizeMultiple = 32;

        private readonly string _ditPath;
        private readonly string _tePath;
        private readonly string _vaePath;
        private readonly float[] _grid;
        private readonly object _generationLock = new();
        private MiniMaxH3TextEncoder _textEncoder;
        private MiniMaxH3DiT _dit;
        private MiniMaxH3ImageVae _vae;
        private volatile bool _disposed;

        public MiniMaxH3Config DitConfig { get; }

        /// <summary>The sigma grid the output heads were distilled on.</summary>
        public IReadOnlyList<float> PddSigmas => _grid;

        /// <summary>Phase seconds of the last image: text encode, denoise, VAE decode.</summary>
        public (double Text, double Denoise, double Decode, int Steps) LastTimings { get; private set; }

        /// <summary>True when <paramref name="path"/> is a safetensors file holding a
        /// MiniMax-H3-Image denoiser (read from its header only).</summary>
        public static bool IsImageDenoiserSafetensors(string path)
        {
            if (!MiniMaxH3WeightFile.IsSafetensors(path) || !File.Exists(path)) return false;
            try
            {
                using var file = new MiniMaxH3WeightFile(path);
                return MiniMaxH3Config.IsMiniMaxH3Image(file.Tensors);
            }
            catch (Exception e) when (e is IOException or InvalidDataException or NotSupportedException
                or System.Text.Json.JsonException)
            {
                return false;
            }
        }

        /// <summary>The architecture detector for a GGUF that holds the image denoiser.</summary>
        public static bool LooksLikeMiniMaxH3Image(GgufFile gguf) =>
            gguf != null && MiniMaxH3Config.IsMiniMaxH3Image(gguf.Tensors);

        public MiniMaxH3ImageModel(string ditPath, BackendType backend)
            : base(ditPath, backend, openGguf: false)
        {
            if (backend is not (BackendType.GgmlCuda or BackendType.GgmlMetal or BackendType.GgmlCpu))
                throw new ModelLoadRefusedException(
                    $"MiniMax-H3-Image runs on --backend ggml_cuda, ggml_metal or ggml_cpu, not {backend}.");
            _ditPath = ditPath;
            using (var probe = new MiniMaxH3WeightFile(ditPath))
            {
                DitConfig = MiniMaxH3Config.Detect(probe.Tensors);
                if (DitConfig.HasAudio)
                    throw new ModelLoadRefusedException(
                        $"{Path.GetFileName(ditPath)} is a MiniMax-H3 audio-video denoiser; load it as minimax-h3.");
                if (DitConfig.PddSteps < 2)
                    throw new ModelLoadRefusedException(
                        $"{Path.GetFileName(ditPath)} has one output head; MiniMax-H3-Image's distilled denoiser stacks one per step.");
                _grid = MiniMaxH3ImageSampling.GridFrom(probe.SafetensorsMetadata, DitConfig.PddSteps);
            }
            Config = new ModelConfig { Architecture = ArchitectureId, VocabSize = 0 };

            string dir = Path.GetDirectoryName(Path.GetFullPath(ditPath));
            _tePath = MiniMaxH3Model.ResolveCompanion("TS_VIDEO_TEXT_ENCODER", dir,
                new[] { "qwen3vl_32b_minimax_h3-Q4_K_M.gguf" },
                n => n.Contains("qwen3vl") && n.EndsWith(".gguf"));
            _vaePath = MiniMaxH3Model.ResolveCompanion("TS_VIDEO_VAE", dir,
                new[] { "minimax_h3_image_vae_fp16.safetensors" },
                n => n.Contains("h3_image_vae") && n.EndsWith(".safetensors"));

            Console.WriteLine($"MiniMax-H3-Image: DiT={Path.GetFileName(ditPath)}");
            Console.WriteLine($"  {DitConfig}");
            Console.WriteLine($"  sigma grid   = [{string.Join(", ", _grid.Select(s => s.ToString("0.#####")))}]");
            Console.WriteLine($"  text-encoder = {_tePath ?? "<missing>"}");
            Console.WriteLine($"  image VAE    = {_vaePath ?? "<missing>"}");
            if (_tePath == null || _vaePath == null)
                throw new FileNotFoundException(
                    "MiniMax-H3-Image needs companion models beside the denoiser (or via " +
                    "--video-text-encoder / --video-vae): the MiniMax-H3 Qwen3-VL-32B text encoder GGUF " +
                    "(unsloth/MiniMax-H3-GGUF, with vocab.json, merges.txt and tokenizer_config.json from " +
                    "MiniMaxAI/MiniMax-H3 processor/ beside it) and minimax_h3_image_vae_fp16.safetensors " +
                    "(Comfy-Org/MiniMax-H3-Image).");
        }

        // ---- IImageGenerationModel ----
        string IImageGenerationModel.ImageModelFamily => ArchitectureId;
        public bool SupportsImageEdit => false;

        public RgbImage EditImage(string prompt, IReadOnlyList<RgbImage> inputs, QwenImageParams p) =>
            throw new NotSupportedException("MiniMax-H3-Image editing (reference images) is not implemented yet; use text to image.");

        /// <summary>
        /// One image. Steps default to the six the checkpoint is distilled to, the size to
        /// 2048 x 2048; guidance is fixed at 1 and there is no negative prompt, so a request
        /// asking for either is refused. The seed draws the initial latent as ComfyUI does
        /// (CPU torch.randn), so a seed reproduces ComfyUI's template up to kernel precision.
        /// </summary>
        public RgbImage GenerateImage(string prompt, QwenImageParams p = null)
        {
            p ??= new QwenImageParams();
            int steps = p.Steps > 0 ? p.Steps : MiniMaxH3ImageSampling.Steps;
            if (p.CfgScale > 1f)
                throw new ArgumentException("MiniMax-H3-Image is guidance-distilled: cfg must be 1 (or left unset).");
            if (!string.IsNullOrWhiteSpace(p.NegativePrompt))
                throw new ArgumentException("MiniMax-H3-Image takes no negative prompt.");
            if (p.Mask != null)
                throw new ArgumentException("MiniMax-H3-Image takes no mask.");
            var (width, height) = ResolveSize(p);

            lock (_generationLock)
            {
                if (_disposed) throw new ObjectDisposedException(nameof(MiniMaxH3ImageModel));
                return Generate(prompt ?? string.Empty, width, height, steps, p);
            }
        }

        internal static (int Width, int Height) ResolveSize(QwenImageParams p)
        {
            int width = p.Width, height = p.Height;
            if (width <= 0 && height <= 0)
            {
                if (p.TargetArea > 0)
                {
                    int side = (int)Math.Round(Math.Sqrt(p.TargetArea) / SizeMultiple) * SizeMultiple;
                    width = height = Math.Max(SizeMultiple, side);
                }
                else width = height = DefaultSize;
            }
            else if (width <= 0 || height <= 0)
                throw new ArgumentException("MiniMax-H3-Image needs both width and height, or neither.");
            if (width % SizeMultiple != 0 || height % SizeMultiple != 0)
                throw new ArgumentException($"MiniMax-H3-Image sizes are multiples of {SizeMultiple}; got {width}x{height}.");
            return (width, height);
        }

        private RgbImage Generate(string prompt, int width, int height, int steps, QwenImageParams p)
        {
            int lh = height / MiniMaxH3ImageVae.Downsample, lw = width / MiniMaxH3ImageVae.Downsample;
            var total = Stopwatch.StartNew();

            // ---- text: raw prompt tokens (no chat template), hidden states of layer 50 ----
            var clock = Stopwatch.StartNew();
            _textEncoder ??= new MiniMaxH3TextEncoder(_tePath, null, _backend, null);
            var ids = _textEncoder.Tokenize(prompt);
            if (ids.Count == 0) ids.Add(_textEncoder.Tokenizer.BosTokenId);
            float[] textHidden = _textEncoder.Encode(ids);
            int textLength = ids.Count;
            double textSeconds = clock.Elapsed.TotalSeconds;

            // ---- noise: torch.randn([1, 24, 1, lh, lw]) on the CPU generator ----
            float[] chw = TorchCpuRandn.Normal((long)MiniMaxH3ImageVae.LatentChannels * lh * lw, unchecked((ulong)p.Seed));
            var hwc = new float[chw.Length];
            int plane = lh * lw;
            for (int c = 0; c < MiniMaxH3ImageVae.LatentChannels; c++)
                for (int i = 0; i < plane; i++)
                    hwc[(long)i * MiniMaxH3ImageVae.LatentChannels + c] = chw[(long)c * plane + i];
            float[] x = MiniMaxH3DiT.PatchifyVideo(hwc, 1, lh, lw, MiniMaxH3ImageVae.LatentChannels);
            int imageTokens = (lh / MiniMaxH3Geometry.PatchH) * (lw / MiniMaxH3Geometry.PatchW);

            // ---- denoise: Euler on ComfyUI's schedule, the PDD head of each interval ----
            clock.Restart();
            _dit ??= new MiniMaxH3DiT(_ditPath, _backend, null);
            float[] sigmas = MiniMaxH3ImageSampling.Schedule(_grid, steps);
            for (int i = 0; i < steps; i++)
            {
                float sigma = sigmas[i], next = sigmas[i + 1];
                var layout = MiniMaxH3ImageLayout.Build(textLength, lh, lw, sigma);
                var (start, stop) = MiniMaxH3ImageSampling.Interval(_grid, sigma, next);
                var stepClock = Stopwatch.StartNew();
                float[] v = _dit.ForwardImage(x, imageTokens, textHidden, textLength, layout, _grid, start, stop);
                float dt = next - sigma;
                for (long k = 0; k < x.Length; k++) x[k] += v[k] * dt;
                Console.WriteLine($"  [h3-image] step {i + 1}/{steps} sigma {sigma:0.#####} -> {next:0.#####} " +
                                  $"(heads {start}..{stop}) {stepClock.Elapsed.TotalSeconds:F2}s");
                p.OnStep?.Invoke(i + 1, steps, null);
            }
            double denoiseSeconds = clock.Elapsed.TotalSeconds;

            // ---- decode ----
            clock.Restart();
            float[] latent = MiniMaxH3DiT.UnpatchifyVideo(x, 1, lh, lw, MiniMaxH3ImageVae.LatentChannels);
            _vae ??= new MiniMaxH3ImageVae(_vaePath);
            RgbImage image = _vae.Decode(latent, lh, lw);
            double decodeSeconds = clock.Elapsed.TotalSeconds;

            LastTimings = (textSeconds, denoiseSeconds, decodeSeconds, steps);
            Console.WriteLine($"  [h3-image] {width}x{height}, {textLength} text tokens, {imageTokens} image tokens: " +
                              $"text {textSeconds:F2}s, denoise {denoiseSeconds:F2}s ({denoiseSeconds / steps:F2}s/step), " +
                              $"decode {decodeSeconds:F2}s ({_vae.LastTileCount} tiles, {_vae.LastNativeSeconds:F2}s native), " +
                              $"total {total.Elapsed.TotalSeconds:F2}s");
            return image;
        }

        // ---- not an autoregressive model ----
        protected override float[] ForwardCore(int[] tokens) =>
            throw new NotSupportedException("MiniMaxH3ImageModel is an image-generation model; use GenerateImage().");

        protected override void ResetKVCacheCore() { }

        public override void WarmUpKernels() { }

        public override void Dispose()
        {
            _disposed = true;
            lock (_generationLock)
            {
                _vae?.Dispose(); _vae = null;
                _dit?.Dispose(); _dit = null;
                _textEncoder?.Dispose(); _textEncoder = null;
                base.Dispose();
            }
        }
    }
}
