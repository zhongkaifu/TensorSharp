// Copyright (c) Zhongkai Fu. All rights reserved.
// https://github.com/zhongkaifu/TensorSharp
//
// This file is part of TensorSharp.
//
// TensorSharp is licensed under the BSD-3-Clause license found in the LICENSE file in the root directory of this source tree.
//
// MiniMax-H3-Image (text to image and editing) as a model the hosts load: the H3 DiT without its audio stream, the H3
// Qwen3-VL-32B text encoder with its vision tower and the image VAE, run by the shared diffusion runtime (DiffusionPipeline
// over MiniMaxH3ImageFamily). Distilled to six Euler steps with guidance 1 on the checkpoint's own sigma grid, about 16 MP
// natively and 4 MP by default. An edit takes up to 16 reference pictures, which the prompt names <Picture 1>,
// <Picture 2>, ... in their order.
//
//   TensorSharp.Cli --config config/minimax-h3-image.json --prompt "..." --output out.png
//   TensorSharp.Cli --config config/minimax-h3-image.json --image a.png --image b.png --prompt "..." --output edit.png
//
// The denoiser is the safetensors release in its original names (Comfy-Org/MiniMax-H3-Image) or a GGUF with the same
// tensors; the text encoder and the VAE come through the H3 companion flags --video-text-encoder and --video-vae, or sit
// beside the denoiser.
using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Threading;
using TensorSharp.Models.MiniMaxH3;
using TensorSharp.Models.QwenImage;
using TensorSharp.Models.SharedDiffusion;
using TensorSharp.Runtime;
using TensorSharp.Runtime.Scheduling.Diffusion;

namespace TensorSharp.Models.MiniMaxH3Image
{
    public sealed partial class MiniMaxH3ImageDiffusionModel : ModelBase, IImageGenerationModel
    {
        public const string ArchitectureId = MiniMaxH3ImageFamily.ArchitectureId;

        /// <summary>The size a request that names none gets: ComfyUI's template default (2048 x 2048, 4 MP).</summary>
        public const int DefaultSize = 2048;

        /// <summary>Canvas sides are multiples of 32: the 16x VAE and the 2x2 patch.</summary>
        public const int SizeMultiple = 32;

        private readonly MiniMaxH3ImageFamily _family;
        private readonly DiffusionPipeline _pipeline;
        private readonly object _generationLock = new();
        private volatile bool _disposed;

        public MiniMaxH3ImageDiffusionModel(string ditPath, BackendType backend)
            : base(ditPath, backend, openGguf: false)
        {
            if (backend is not (BackendType.GgmlCuda or BackendType.GgmlMetal or BackendType.GgmlCpu))
                throw new ModelLoadRefusedException(
                    $"MiniMax-H3-Image runs on --backend ggml_cuda, ggml_metal or ggml_cpu, not {backend}.");
            float[] grid;
            using (var probe = new MiniMaxH3WeightFile(ditPath))
            {
                DitConfig = MiniMaxH3Config.Detect(probe.Tensors);
                if (DitConfig.HasAudio)
                    throw new ModelLoadRefusedException(
                        $"{Path.GetFileName(ditPath)} is a MiniMax-H3 audio-video denoiser; load it as minimax-h3.");
                if (DitConfig.PddSteps < 2)
                    throw new ModelLoadRefusedException(
                        $"{Path.GetFileName(ditPath)} has one output head; MiniMax-H3-Image's distilled denoiser stacks one per step.");
                grid = MiniMaxH3ImageSampling.GridFrom(probe.SafetensorsMetadata, DitConfig.PddSteps);
            }
            _family = new MiniMaxH3ImageFamily(grid);
            Config = new ModelConfig { Architecture = ArchitectureId, VocabSize = 0 };

            string dir = Path.GetDirectoryName(Path.GetFullPath(ditPath));
            string tePath = MiniMaxH3Model.ResolveCompanion("TS_VIDEO_TEXT_ENCODER", dir,
                new[] { "qwen3vl_32b_minimax_h3-Q4_K_M.gguf" },
                n => n.Contains("qwen3vl") && n.EndsWith(".gguf"));
            string vaePath = MiniMaxH3Model.ResolveCompanion("TS_VIDEO_VAE", dir,
                new[] { "minimax_h3_image_vae_fp16.safetensors" },
                n => n.Contains("h3_image_vae") && n.EndsWith(".safetensors"));

            Console.WriteLine($"MiniMax-H3-Image: DiT={Path.GetFileName(ditPath)}");
            Console.WriteLine($"  {DitConfig}");
            Console.WriteLine($"  sigma grid   = [{string.Join(", ", grid.Select(s => s.ToString("0.#####")))}]");
            Console.WriteLine($"  text-encoder = {tePath ?? "<missing>"}");
            Console.WriteLine($"  image VAE    = {vaePath ?? "<missing>"}");
            if (tePath == null || vaePath == null)
                throw new FileNotFoundException(
                    "MiniMax-H3-Image needs companion models beside the denoiser (or via " +
                    "--video-text-encoder / --video-vae): the MiniMax-H3 Qwen3-VL-32B text encoder GGUF " +
                    "(unsloth/MiniMax-H3-GGUF, with vocab.json, merges.txt and tokenizer_config.json from " +
                    "MiniMaxAI/MiniMax-H3 processor/ beside it) and minimax_h3_image_vae_fp16.safetensors " +
                    "(Comfy-Org/MiniMax-H3-Image).");

            var load = new DiffusionLoad
            {
                Backend = backend,
                Files = new Dictionary<string, string>
                {
                    [DiffusionLoad.Denoiser] = ditPath,
                    [DiffusionLoad.TextEncoder] = tePath,
                    [DiffusionLoad.Vae] = vaePath,
                },
            };
            _pipeline = new DiffusionPipeline(_family, load);
            ApplyInheritedLoras();
        }

        public MiniMaxH3Config DitConfig { get; }

        /// <summary>The sigma grid the output heads were distilled on.</summary>
        public IReadOnlyList<float> PddSigmas => _family.Grid;

        /// <summary>Phase seconds of the last image: text encode, denoise, VAE decode.</summary>
        public (double Text, double Denoise, double Decode, int Steps) LastTimings { get; private set; }

        /// <summary>The last image's full result: every phase's seconds, the recipe and the residency plan.</summary>
        public DiffusionResult LastResult { get; private set; }

        /// <summary>Text tokens the last image was conditioned on.</summary>
        public int LastTextTokens { get; private set; }

        /// <summary>VAE tiles the last edit encoded (all its references).</summary>
        public int LastEncodeTiles { get; private set; }

        /// <summary>VAE tiles the last image decoded, and the seconds its decoder programs ran.</summary>
        public int LastTileCount { get; private set; }
        public double LastDecodeNativeSeconds { get; private set; }

        /// <summary>The runtime pipeline (a test seam: residency, components).</summary>
        internal DiffusionPipeline Pipeline => _pipeline;

        /// <summary>A test seam: receives every progress event of a run (phases and steps), before the step callback.</summary>
        internal Action<DiffusionProgress> ProgressObserver { get; set; }

        // ---- IImageGenerationModel ----
        string IImageGenerationModel.ImageModelFamily => ArchitectureId;
        public bool SupportsImageEdit => true;
        public bool SupportsEditMask => true;
        public long MinimumGenerationArea => MiniMaxH3ImageFamily.MinGenerationArea;

        /// <summary>
        /// One edit of up to 16 reference pictures, which the prompt names <c>&lt;Picture 1&gt;</c>, <c>&lt;Picture 2&gt;</c>, ...
        /// in the order of <paramref name="inputs"/>, as MiniMax's pipeline conditions on them (each through the vision tower
        /// beside the prompt, and through the VAE as rows of the packed sequence). The size is the request's (a multiple of
        /// 32), <c>&lt;Picture 1&gt;</c>'s with keepSourceSize, or <c>&lt;Picture 1&gt;</c>'s aspect at the target area; below
        /// 2048 x 2048 it samples larger and is resized back. A seed draws the references' posterior samples, then the
        /// target's noise, then the references' augmentation, from one CPU torch generator (diffusers' order). Guidance is
        /// 1 and there is no negative prompt.
        /// <para>A mask (<see cref="QwenImageParams.Mask"/>, <c>&lt;Picture 1&gt;</c>'s exact size, with its mode, invert,
        /// feather and crop) limits the edit to its selection, TensorSharp's own method (docs/design/diffusion-runtime.md
        /// section 4.5; upstream has none): the kept latent cells are re-noised from the picture's own latent after every
        /// step and end on it, and the result is <c>&lt;Picture 1&gt;</c>'s canvas with every unselected pixel its own
        /// bytes. <c>&lt;Picture 1&gt;</c> (or the selection's crop) is still a reference, so the model sees what it blends
        /// with; the other pictures stay references.</para>
        /// </summary>
        public RgbImage EditImage(string prompt, IReadOnlyList<RgbImage> inputs, QwenImageParams p)
        {
            if (inputs is not { Count: > 0 })
                throw new ArgumentException("A MiniMax-H3-Image edit needs at least one reference image.");
            var references = new DiffusionReference[inputs.Count];
            for (int i = 0; i < references.Length; i++)
                references[i] = new DiffusionReference(inputs[i] ?? throw new ArgumentException($"Reference image {i + 1} is missing."));
            return Run(prompt, p, references);
        }

        /// <summary>
        /// One image. Steps default to the six the checkpoint is distilled to, the size to 2048 x 2048; guidance is fixed at
        /// 1 and there is no negative prompt, so a request asking for either is refused. The seed draws the initial latent
        /// as ComfyUI does (CPU torch.randn), so a seed reproduces ComfyUI's template up to kernel precision. Progress is a
        /// tick a step (no decoded previews).
        /// </summary>
        public RgbImage GenerateImage(string prompt, QwenImageParams p = null) =>
            Run(prompt, p ?? new QwenImageParams(), Array.Empty<DiffusionReference>());

        private RgbImage Run(string prompt, QwenImageParams p, IReadOnlyList<DiffusionReference> references)
        {
            ArgumentNullException.ThrowIfNull(p);
            if (p.CfgScale > 1f)
                throw new ArgumentException("MiniMax-H3-Image is guidance-distilled: cfg must be 1 (or left unset).");
            if (!string.IsNullOrWhiteSpace(p.NegativePrompt))
                throw new ArgumentException("MiniMax-H3-Image takes no negative prompt.");
            if (p.Mask != null && references.Count == 0)
                throw new ArgumentException("A mask requires a first reference image to edit.");
            if (references.Count == 0 && p.KeepSourceSize)
                throw new ArgumentException("keepSourceSize needs a reference image.");
            Action<int, int, RgbImage> onStep = p.OnStep;
            Action<DiffusionProgress> observer = ProgressObserver;
            var request = new DiffusionRequest
            {
                Prompt = prompt ?? string.Empty,
                References = references,
                Width = p.Width,
                Height = p.Height,
                TargetArea = p.TargetArea,
                KeepSourceSize = p.KeepSourceSize,
                Mask = references.Count == 0 ? null : QwenImageEditMask.Options(p),
                Seed = p.Seed,
                Steps = p.Steps,
                Cfg = p.CfgScale,
                PreviewCount = 0,
                Progress = onStep == null && observer == null ? null : progress =>
                {
                    observer?.Invoke(progress);
                    if (progress.IsStep) onStep?.Invoke(progress.Step, progress.Total, null);
                },
            };

            lock (_generationLock)
            {
                if (_disposed) throw new ObjectDisposedException(nameof(MiniMaxH3ImageDiffusionModel));
                long tilesBefore = _family.Vae?.EncodedTiles ?? 0;
                // The plug-ins in force now (SetLoras swaps them under this lock).
                DiffusionResult result = _pipeline.Run(request.WithAdapters(_adapters), CancellationToken.None);
                LastEncodeTiles = (int)((_family.Vae?.EncodedTiles ?? 0) - tilesBefore);
                DiffusionTimings t = result.Timings;
                int steps = result.Recipe.Steps;
                LastResult = result;
                LastTimings = (t.Text, t.Denoise, t.Decode, steps);
                var (width, height) = result.Recipe.Canvas.Sampling;
                int imageTokens = (height / MiniMaxH3ImageVaeCodec.Downsample / 2) * (width / MiniMaxH3ImageVaeCodec.Downsample / 2);
                LastTextTokens = _family.Conditioner?.LastTokens ?? 0;
                LastTileCount = _family.Vae?.LastTileCount ?? 0;
                LastDecodeNativeSeconds = _family.Vae?.LastNativeSeconds ?? 0;
                if (references.Count == 0)
                {
                    Console.WriteLine($"  [h3-image] {width}x{height}, {LastTextTokens} text tokens, {imageTokens} image tokens: " +
                                      $"text {t.Text:F2}s, denoise {t.Denoise:F2}s ({t.Denoise / steps:F2}s/step), " +
                                      $"decode {t.Decode:F2}s ({LastTileCount} tiles, {LastDecodeNativeSeconds:F2}s native), " +
                                      $"total {t.Total:F2}s");
                }
                else
                {
                    var (ow, oh) = result.Recipe.Canvas.Output;
                    Console.WriteLine($"  [h3-image] edit {width}x{height}" + (ow != width || oh != height ? $" -> {ow}x{oh}" : "") +
                                      $", {references.Count} reference(s), {LastTextTokens} conditioning tokens, {imageTokens} image tokens: " +
                                      $"text {t.Text:F2}s, vision {t.Vision:F2}s, VAE encode {t.VaeEncode:F2}s " +
                                      $"({LastEncodeTiles} tiles), denoise {t.Denoise:F2}s ({t.Denoise / steps:F2}s/step), " +
                                      $"decode {t.Decode:F2}s ({LastTileCount} tiles), total {t.Total:F2}s");
                }
                return result.Image;
            }
        }

        // ---- not an autoregressive model ----
        protected override float[] ForwardCore(int[] tokens) =>
            throw new NotSupportedException("MiniMaxH3ImageDiffusionModel is an image-generation model; use GenerateImage().");

        protected override void ResetKVCacheCore() { }

        public override void WarmUpKernels() { }

        public override void Dispose()
        {
            _disposed = true;
            lock (_generationLock)
            {
                _pipeline.Dispose();
                DisposeLoras();
                base.Dispose();
            }
        }
    }
}
