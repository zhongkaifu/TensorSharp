// Copyright (c) Zhongkai Fu. All rights reserved.
// https://github.com/zhongkaifu/TensorSharp
//
// This file is part of TensorSharp.
//
// TensorSharp is licensed under the BSD-3-Clause license found in the LICENSE file in the root directory of this source tree.
//
// Qwen-Image-2.1 and Qwen-Image-2.1-Turbo (text to image and editing) as a model the hosts load, run by the shared diffusion
// runtime (DiffusionPipeline over QwenImage21Family): a 32-layer, 4096-wide single-stream diffusion transformer
// (QwenImage21Denoiser, graph programs on every backend), the dedicated 64-channel /16 RGBA VAE and Qwen3-VL-8B with its
// vision projector for edits.
//
//   TensorSharp.Cli --config config/qwen-image-2.1-turbo.json --prompt "..." --output out.png
//   TensorSharp.Cli --config config/qwen-image-2.1.json --image a.png --prompt "..." --output edit.png
//
// The companions sit beside the transformer GGUF or are named by the hosts (--qwen-image-vae, --qwen-image-vl,
// --qwen-image-mmproj, published as TS_QWEN_IMAGE_VAE / TS_QWEN_IMAGE_TE / TS_QWEN_IMAGE_MMPROJ).
using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Threading;
using TensorSharp.Core;
using TensorSharp.GGML;
using TensorSharp.Models.QwenImage;
using TensorSharp.Models.SharedDiffusion;
using TensorSharp.Runtime;
using TensorSharp.Runtime.Scheduling.Diffusion;

namespace TensorSharp.Models.QwenImage21
{
    public sealed class QwenImage21DiffusionModel : ModelBase, ILoraPluginHost, IImageGenerationModel
    {
        public const string ArchitectureId = QwenImage21Family.ArchitectureId;

        private readonly string _ditPath, _vaePath, _tePath, _mmprojPath;
        private readonly bool _variantDeclared;
        private readonly QwenImage21Family _family;
        private readonly DiffusionPipeline _pipeline;
        private readonly object _generationLock = new();
        private GgufFile _vaeGguf, _teGguf;
        private SafetensorsModel _vaeSafetensors;
        private IFloatTensorStore _vaeWeights;
        private QwenImage21Denoiser _denoiser;
        private QwenImage21VaeCodec _vae;
        private AdapterSet _adapters = AdapterSet.Empty;
        private volatile bool _disposed;

        public QwenImage21DiffusionModel(string ggufPath, BackendType backend, int tpDegree = 1, ITensorParallelGroup tpGroup = null)
            : base(ggufPath, backend, tpDegree, tpGroup)
        {
            try
            {
                _ditPath = ggufPath;
                bool version21 = _gguf.Tensors.Keys.Any(n => n == "txt_in.text_norm.weight" ||
                    n.EndsWith(".txt_in.text_norm.weight", StringComparison.Ordinal));
                if (!version21)
                    throw new ModelLoadRefusedException(
                        $"'{Path.GetFileName(ggufPath)}' is not a Qwen-Image-2.1 diffusion transformer (no txt_in.text_norm.weight tensor). " +
                        "Earlier Qwen-Image / Qwen-Image-Edit checkpoints such as Qwen-Image-Edit-2511 are no longer supported; " +
                        "use a Qwen-Image-2.1 GGUF (see docs/models/qwenimage21.md).");
                if (!IsGgmlBackend && backend != BackendType.Cpu)
                    throw new NotSupportedException("Qwen-Image-2.1 requires a GGML backend (ggml_metal, ggml_cuda, ggml_vulkan or ggml_cpu) or the pure-C# cpu backend.");
                if (backend == BackendType.Cpu && (tpDegree > 1 || tpGroup != null))
                    throw new ModelLoadRefusedException(
                        $"Qwen-Image-2.1 on the pure-C# cpu backend runs in one process; --tp {Math.Max(tpDegree, tpGroup?.Degree ?? 1)} " +
                        "shards the transformer over the GPUs of a GGML backend (ggml_cuda, ggml_vulkan). Remove --tp or choose a GGML GPU backend.");
                if (IsTensorParallel && (TpNodeCount > 1 || GlobalTpDegree != TpDegree))
                    throw new ModelLoadRefusedException(
                        "Qwen-Image-2.1 tensor parallelism shards over the GPUs of one machine; a multi-node --tp group is not supported.");
                if (IsTensorParallel && DitTensorParallelRefusal(TpDegree) is string tpRefusal)
                    throw new ModelLoadRefusedException(tpRefusal);
                Config = new ModelConfig
                {
                    Architecture = ArchitectureId, HiddenSize = 4096, NumLayers = 32, NumHeads = 32, NumKVHeads = 32, Eps = 1e-6f, VocabSize = 0,
                };

                string dir = Path.GetDirectoryName(Path.GetFullPath(ggufPath)) ?? ".";
                _vaePath = ResolveCompanion("TS_QWEN_IMAGE_VAE", dir,
                    n => n.Contains("qwen_image_2.1_vae") && n.EndsWith(".safetensors"));
                _tePath = ResolveCompanion("TS_QWEN_IMAGE_TE", dir,
                    n => (n.Contains("qwen3vl-8b") || n.Contains("qwen3-vl-8b")) && !n.Contains("mmproj") && n.EndsWith(".gguf"));
                _mmprojPath = ResolveCompanion("TS_QWEN_IMAGE_MMPROJ", dir,
                    n => n.Contains("mmproj") && (n.Contains("qwen3vl-8b") || n.Contains("qwen3-vl-8b")) && n.EndsWith(".gguf"));

                // Turbo and the base checkpoint share every tensor name and shape, and the GGUFs name at most their
                // architecture: the host declares which one this is (QwenImage21Turbo).
                string variantNote;
                try
                {
                    Variant = QwenImage21Turbo.Resolve(Environment.GetEnvironmentVariable(QwenImageVariantFlag.EnvironmentVariable),
                        ggufPath, out variantNote);
                }
                catch (ArgumentException e)
                {
                    throw new ModelLoadRefusedException(e.Message, e);
                }
                _variantDeclared = !string.IsNullOrWhiteSpace(Environment.GetEnvironmentVariable(QwenImageVariantFlag.EnvironmentVariable));

                Console.WriteLine($"Qwen-Image-2.1: DiT={Path.GetFileName(ggufPath)} (diffusion runtime)");
                Console.WriteLine($"  variant      = {variantNote}");
                Console.WriteLine($"  VAE          = {_vaePath ?? "<missing>"}");
                Console.WriteLine($"  text-encoder = {_tePath ?? "<missing>"}");
                Console.WriteLine($"  mmproj       = {_mmprojPath ?? "<none> (text-only grounding)"}");

                if (_vaePath == null || _tePath == null)
                    throw new FileNotFoundException(
                        "Qwen-Image-2.1 needs its 2.1 VAE and Qwen3-VL-8B text-encoder GGUF. " +
                        "Place them next to the DiT GGUF or set TS_QWEN_IMAGE_VAE / TS_QWEN_IMAGE_TE.");

                // Header-only checks fail before any expensive generation work.
                QwenImage21CompanionValidation.ValidateVae(VaeWeightSource);
                QwenImage21CompanionValidation.ValidateText(TeGguf);
                if (_mmprojPath != null)
                {
                    using var vision = new GgufFile(_mmprojPath);
                    QwenImage21CompanionValidation.ValidateVision(vision);
                }

                _family = new QwenImage21Family(backend, Variant)
                {
                    ConditionerFactory = load => new QwenImage21ConditionerComponent(_tePath, _mmprojPath, backend, () => ReleaseComputeBuffers()),
                    DenoiserFactory = load => _denoiser = CreateDenoiser(),
                    VaeFactory = load => _vae = new QwenImage21VaeCodec(() => VaeWeightSource, backend, FileLength(_vaePath),
                        owner => ReleaseComputeBuffers(owner)),
                };
                var diffusionLoad = new DiffusionLoad
                {
                    Backend = backend,
                    Files = new Dictionary<string, string>
                    {
                        [DiffusionLoad.Denoiser] = ggufPath,
                        [DiffusionLoad.TextEncoder] = _tePath,
                        [DiffusionLoad.Vae] = _vaePath,
                    },
                    TensorParallelGroup = DitTensorParallelGroup,
                };
                _pipeline = new DiffusionPipeline(_family, diffusionLoad);

                // LoRA plug-ins from the host (--lora / TS_LORAS), loaded and checked against the transformer now, so a bad
                // adapter fails before any request. Their configuration mistakes are refusals (one line, exit 2), not crashes.
                if (!string.IsNullOrWhiteSpace(Environment.GetEnvironmentVariable("TS_QWEN_IMAGE_LORA")))
                    throw new ModelLoadRefusedException(
                        "TS_QWEN_IMAGE_LORA belonged to the retired Qwen-Image-Edit-2511 pipeline. Pass Qwen-Image-2.1 LoRA plug-ins " +
                        $"with --lora (the hosts publish them as {LoraCliFlags.EnvironmentVariable}) and unset TS_QWEN_IMAGE_LORA.");
                try
                {
                    // TS_LORAS can also carry another family's plug-ins (a MiniMax-H3 config inherited by this process); the
                    // host names what it skips, and this model applies only the ones it accepts.
                    var specs = LoraCliFlags.FromJson(Environment.GetEnvironmentVariable(LoraCliFlags.EnvironmentVariable))
                        .Where(AcceptsLora).ToList();
                    if (specs.Count > 0) SetLoras(specs);
                }
                catch (Exception e) when (e is ArgumentException or System.Text.Json.JsonException or FormatException or
                    KeyNotFoundException or (InvalidOperationException and not ModelLoadRefusedException))
                {
                    throw new ModelLoadRefusedException("LoRA plug-in refused: " + e.Message, e);
                }
            }
            catch { Dispose(); throw; }
        }

        private QwenImage21Denoiser CreateDenoiser()
        {
            var weights = new QwenImage21Weights(_gguf, ownsFile: false, Path.GetFileName(DitPath));
            return new QwenImage21Denoiser(weights, Backend, null, null, DitTensorParallelGroup)
            {
                ReleaseAfterPhase = true,
                ReleaseCompute = () => ReleaseComputeBuffers(),
                AdapterBytes = () => Loras?.FactorBytes ?? 0,
            };
        }

        private static long FileLength(string path)
        {
            try { return new FileInfo(path).Length; }
            catch (Exception e) when (e is IOException or UnauthorizedAccessException) { return 0; }
        }

        private static string ResolveCompanion(string envVar, string dir, Func<string, bool> match)
        {
            string explicitPath = Environment.GetEnvironmentVariable(envVar);
            if (!string.IsNullOrWhiteSpace(explicitPath))
            {
                if (!File.Exists(explicitPath))
                    throw new FileNotFoundException($"Qwen-Image-2.1 companion specified by {envVar} does not exist.", explicitPath);
                return Path.GetFullPath(explicitPath);
            }
            return Directory.EnumerateFiles(dir).Where(f => match(Path.GetFileName(f).ToLowerInvariant()))
                .OrderByDescending(f => new FileInfo(f).Length).ThenBy(f => f, StringComparer.Ordinal).FirstOrDefault();
        }

        internal static bool IsSafetensorsPath(string p) =>
            p != null && (p.EndsWith(".safetensors", StringComparison.OrdinalIgnoreCase) || p.EndsWith(".index.json", StringComparison.OrdinalIgnoreCase));

        internal string DitPath => _ditPath;
        internal GgufFile TeGguf => _teGguf ??= new GgufFile(_tePath);
        internal string MmprojPath => _mmprojPath;
        internal string TePath => _tePath;
        internal BackendType Backend => _backend;
        internal bool UsesGgml => IsGgmlBackend;
        internal QwenImage21Family Family => _family;
        internal DiffusionPipeline Pipeline => _pipeline;

        /// <summary>The VAE weights: a <c>.safetensors</c> file (BF16 upcast to F32) or a converted F32 GGUF; both give the same floats.</summary>
        internal IFloatTensorStore VaeWeightSource => _vaeWeights ??= OpenVaeWeightSource();

        private IFloatTensorStore OpenVaeWeightSource()
        {
            IFloatTensorStore source = IsSafetensorsPath(_vaePath)
                ? _vaeSafetensors = SafetensorsModel.Open(_vaePath)
                : new GgufFloatTensorStore(_vaeGguf ??= new GgufFile(_vaePath));
            return new QwenImage21VaeTensorStore(source);
        }

        /// <summary>
        /// Which 2.1 checkpoint the transformer is, as declared by the host (<see cref="QwenImageVariantFlag.EnvironmentVariable"/>)
        /// or, without a declaration, assumed from the file name (<see cref="QwenImage21Turbo.Resolve"/>).
        /// </summary>
        public QwenImageVariant Variant { get; }

        // ---- tensor parallelism

        /// <summary>Why the transformer cannot shard over <paramref name="ranks"/> GPUs, or null: every GPU holds whole heads.</summary>
        internal static string DitTensorParallelRefusal(int ranks) =>
            ranks < 2 || QwenImage21Weights.ReleaseHeads % ranks == 0
                ? null
                : $"Qwen-Image-2.1 tensor parallelism needs a GPU count that divides its {QwenImage21Weights.ReleaseHeads} " +
                  $"attention heads, because every GPU holds whole heads; --tp {ranks} does not.";

        /// <summary>The group the transformer shards over, or null on one device.</summary>
        internal ITensorParallelGroup DitTensorParallelGroup => IsTensorParallel ? _tpGroup : null;

        // QwenImage21Denoiser shards its blocks on this group when it is made.
        protected override bool ShardsWeightsInComponents => true;

        // ---- LoRA

        /// <summary>The LoRA plug-ins in use (empty when none).</summary>
        public IReadOnlyList<LoraSpec> LoraSpecs { get; private set; } = Array.Empty<LoraSpec>();

        public string LoraPluginType => LoraCliFlags.QwenImagePluginType;

        /// <summary>A Qwen-Image-2.1 plug-in config, or weights with no TensorSharp plug-in config (bare, or with a PEFT or PDD
        /// config); another family's plug-in is not.</summary>
        public bool AcceptsLora(LoraSpec spec)
        {
            string declared = LoraCliFlags.DeclaredPluginType(spec);
            return declared == null || declared == LoraCliFlags.QwenImagePluginType;
        }

        internal QwenImage21LoraSet Loras { get; private set; }

        /// <summary>The checkpoint's own recipe in the LoRA loader's terms (Turbo's), or null for the base checkpoint.</summary>
        internal QwenImage21LoraRecipe CheckpointRecipe => Variant == QwenImageVariant.Turbo ? QwenImage21Turbo.Recipe : null;

        /// <summary>
        /// Replace the LoRA plug-ins applied to every later request (an empty list removes them). The adapters are validated
        /// against this transformer immediately; a failure leaves the previous set in place.
        /// </summary>
        public void SetLoras(IReadOnlyList<LoraSpec> specs)
        {
            ArgumentNullException.ThrowIfNull(specs);
            var resolved = LoraCliFlags.Resolve(specs);
            QwenImage21LoraSet loaded = null;
            SamplingRecipe recipe = null;
            if (resolved.Count > 0)
            {
                var timer = System.Diagnostics.Stopwatch.StartNew();
                string prefix = _gguf.Tensors.ContainsKey("img_in.weight") ? "" : "model.diffusion_model.";
                // A step-distillation plug-in brings its own schedule; Turbo has one already.
                string checkpoint = CheckpointRecipe == null ? null : _variantDeclared
                    ? "declared"
                    : $"assumed from its file name; declare {QwenImageVariantFlag.Flag} base if it is not";
                loaded = QwenImage21LoraSet.Load(resolved, _gguf, prefix, _backend, IsTensorParallel ? TpDegree : 1,
                    CheckpointRecipe, checkpoint);
                try { recipe = _family?.RecipeOf(loaded.Recipe); }
                catch { loaded.Dispose(); throw; }
                Console.WriteLine($"Qwen-Image-2.1 LoRA: {resolved.Count} plug-in(s), applied unmerged " +
                    $"({loaded.FactorBytes / (1024.0 * 1024.0):F0} MiB of factors, loaded in {timer.Elapsed.TotalSeconds:F1}s)");
                Console.WriteLine(loaded.Summary);
            }
            lock (_generationLock)
            {
                _denoiser?.ReleaseLora();
                Loras?.Dispose();
                Loras = loaded;
                LoraSpecs = resolved;
                _family.LoraRecipe = recipe;
                _adapters = loaded == null ? AdapterSet.Empty : new AdapterSet(resolved.Select(s => new DiffusionAdapter
                {
                    Name = Path.GetFileName(s.Path), Scale = s.Scale ?? 1f, Payload = loaded,
                }));
            }
        }

        // ---- IImageGenerationModel

        string IImageGenerationModel.ImageModelFamily => ArchitectureId;
        public bool SupportsImageEdit => true;
        public bool SupportsEditMask => true;

        /// <summary>The last image's full result: every phase's seconds, the recipe and the residency plan.</summary>
        public DiffusionResult LastResult { get; private set; }

        /// <summary>A test seam: receives every progress event of a run (phases and steps), before the step callback.</summary>
        internal Action<DiffusionProgress> ProgressObserver { get; set; }

        public RgbImage GenerateImage(string prompt, QwenImageParams p = null) =>
            Run(prompt, p ?? new QwenImageParams(), Array.Empty<RgbImage>());

        /// <summary>Edit <paramref name="input"/> per the natural-language <paramref name="prompt"/>.</summary>
        public RgbImage EditImage(string prompt, RgbImage input, QwenImageParams p)
        {
            ArgumentNullException.ThrowIfNull(input);
            return Run(prompt, p ?? new QwenImageParams(), new[] { input });
        }

        /// <summary>
        /// Multi-image edit: every entry of <paramref name="inputs"/> conditions the transformer as its own reference, tagged
        /// <c>&lt;image1&gt;</c>, <c>&lt;image2&gt;</c>, ... in order, and <paramref name="inputs"/>[0] drives the output geometry.
        /// </summary>
        public RgbImage EditImage(string prompt, IReadOnlyList<RgbImage> inputs, QwenImageParams p)
        {
            if (inputs == null || inputs.Count == 0)
                throw new ArgumentException("At least one input image is required.", nameof(inputs));
            return Run(prompt, p ?? new QwenImageParams(), inputs.ToArray());
        }

        private RgbImage Run(string prompt, QwenImageParams p, RgbImage[] inputs)
        {
            ArgumentNullException.ThrowIfNull(prompt);
            foreach (var input in inputs) ArgumentNullException.ThrowIfNull(input);
            // Parse first: a misspelled setting fails before any work, text to image included.
            bool followReferences = QwenImage21Noise.EditNoiseFollowsReferences(
                Environment.GetEnvironmentVariable(QwenImage21Noise.EditNoiseVariable));
            if (inputs.Length > 0 && _mmprojPath == null)
                throw new InvalidOperationException("Qwen-Image-2.1 editing requires the Qwen3-VL-8B vision projector; set --qwen-image-mmproj or TS_QWEN_IMAGE_MMPROJ.");
            Action<int, int, RgbImage> onStep = p.OnStep;
            Action<DiffusionProgress> observer = ProgressObserver;
            var request = new DiffusionRequest
            {
                Prompt = prompt,
                References = inputs.Select(i => new DiffusionReference(i)).ToArray(),
                Width = p.Width,
                Height = p.Height,
                TargetArea = p.TargetArea,
                KeepSourceSize = p.KeepSourceSize,
                Mask = QwenImageEditMask.Options(p),
                Seed = p.Seed,
                Steps = p.Steps,
                Cfg = p.CfgScale,
                NegativePrompt = p.NegativePrompt ?? "",
                PreviewCount = p.PreviewCount,
                Progress = onStep == null && observer == null ? null : progress =>
                {
                    observer?.Invoke(progress);
                    if (progress.IsStep) onStep?.Invoke(progress.Step, progress.Total, progress.Preview as RgbImage);
                },
            };
            lock (_generationLock)
            {
                if (_disposed) throw new ObjectDisposedException(nameof(QwenImage21DiffusionModel));
                _family.FollowReferences = followReferences;
                // Refuse a size this machine cannot hold before any encoder, transformer or VAE work.
                if (_backend == BackendType.Cpu) CheckCpuMemory(request);
                if (!p.KeepSourceSize && QwenImage21Canvas.UsesHostCpuAutomaticSize(request, _backend))
                    Console.WriteLine($"  automatic size on the cpu backend: about {QwenImage21Canvas.HostCpuAutomaticArea / (1024 * 1024)} MP. " +
                        "The native 2048x2048 area has four times the tokens and takes about 5x longer per step on a CPU; " +
                        "pass --width/--height (width/height in an API request) for another size.");
                if (inputs.Length > 0 && !followReferences)
                    Console.WriteLine($"  [qwen21] {QwenImage21Noise.EditNoiseVariable}=seed: this edit starts from the seed's text-to-image " +
                        "noise (stable-diffusion.cpp parity); an edit of a picture drawn at this seed and size may retrace it.");
                DiffusionResult result = _pipeline.Run(request.WithAdapters(_adapters), CancellationToken.None);
                LastResult = result;
                if (_family.LastStream != 0)
                    Console.WriteLine($"  [qwen21] edit noise stream {_family.LastStream:x16}");
                return result.Image;
            }
        }

        /// <summary>Refuses (ArgumentException, a 400 on the server) a size whose estimated peak does not fit in this machine's
        /// memory (<see cref="QwenImage21CpuMemory"/>), and warns when it exceeds the memory free right now.</summary>
        private void CheckCpuMemory(DiffusionRequest request)
        {
            RgbImage first = request.References.Count > 0 ? request.References[0].Image : null;
            EditMask mask = request.Mask == null ? null : EditMask.Create(request.Mask, first);
            if (mask?.IsEmpty == true) return;
            var (width, height) = _family.ResolveCanvas(request, mask).Sampling;
            long prefix = 256;
            foreach (DiffusionReference input in request.References)
            {
                var (rw, rh) = QwenImage21Canvas.ReferenceDimensions(input.Image, width, height);
                prefix += (long)(rw / 16) * (rh / 16) * 5 / 4;
            }
            int prefixTokens = (int)Math.Min(int.MaxValue, prefix);
            long transformerBytes = FileLength(DitPath);
            if (Environment.GetEnvironmentVariable(QwenImage21CpuMemory.CheckVariable) != "0" &&
                QwenImage21CpuMemory.Refusal(width, height, prefixTokens, transformerBytes, QwenImage21CpuMemory.TotalMemoryBytes()) is string refusal)
                throw new ArgumentException(refusal);
            long needed = QwenImage21CpuMemory.EstimatePeakBytes(width, height, prefixTokens, transformerBytes);
            long free = QwenImage21CpuMemory.FreeMemoryBytes();
            if (free > 0 && needed > free)
                Console.WriteLine(string.Create(System.Globalization.CultureInfo.InvariantCulture,
                    $"  [qwen21] WARNING: {width}x{height} needs about {QwenImage21CpuMemory.Gib(needed):F1} GiB at its peak, " +
                    $"but only {QwenImage21CpuMemory.Gib(free):F1} GiB is free right now; close other programs or expect paging."));
        }

        // ---- the text encoder as a multiple-choice scorer

        /// <summary>The longest question <see cref="ChooseAnswer"/> scores, in tokens of the prompt <see cref="ChooseAnswerPrompt"/> renders.</summary>
        public const int MaxQuestionTokens = QwenImageTextEncoder.MaxScoringTokens;

        /// <summary>The prompt <see cref="ChooseAnswer"/> scores for these arguments, with the options in the order given.</summary>
        public static string ChooseAnswerPrompt(string system, string user, IReadOnlyList<string> options)
        {
            QwenImageIntentScorer.Validate(system, user, options);
            return QwenImageIntentScorer.RenderChatMl(QwenImageIntentScorer.StripControlMarkers(system),
                QwenImageIntentScorer.RenderQuestion(user, options, Enumerable.Range(0, options.Count).ToArray()));
        }

        /// <summary>
        /// Ask this model's own Qwen3-VL-8B which of <paramref name="options"/> fits the question, each option scored by the
        /// probability of its letter as the reply's first token over every rotation of the options
        /// (<see cref="QwenImageIntentScorer.Choose"/>). It builds the text encoder for the call and disposes it before
        /// returning. Null when the model cannot answer (a backend that is not GGML, an encoder-only GGUF, a question over
        /// <see cref="MaxQuestionTokens"/> tokens), with the reason printed.
        /// </summary>
        public ImageIntentChoice? ChooseAnswer(string system, string user, IReadOnlyList<string> options,
            CancellationToken cancellationToken = default)
        {
            QwenImageIntentScorer.Validate(system, user, options);
            try
            {
                if (!UsesGgml)
                    throw new ImageIntentUnavailableException("next-token scoring runs on GGML backends only");
                if (QwenImageTextEncoder.ScoringHeadRefusal(TeGguf) is string refusal)
                    throw new ImageIntentUnavailableException(refusal);
                cancellationToken.ThrowIfCancellationRequested();
                lock (_generationLock)
                {
                    try
                    {
                        using var encoder = new QwenImageTextEncoder(_tePath, _backend);
                        return QwenImageIntentScorer.Choose(system, user, options,
                            text => encoder.Tokenizer.Encode(text, addSpecial: false), encoder.NextTokenLogits,
                            QwenImageTextEncoder.MaxScoringTokens, cancellationToken);
                    }
                    finally
                    {
                        ReleaseComputeBuffers();
                    }
                }
            }
            catch (ImageIntentUnavailableException e)
            {
                Console.WriteLine($"Qwen-Image-2.1: the text encoder cannot score this question: {e.Message}.");
                return null;
            }
        }

        // ---- memory

        /// <summary>Releases the GGML scratch and this model's device memory (and <paramref name="extra"/> owners'), while no
        /// other model runs a step. The pure-C# cpu backend has neither and never calls the native library.</summary>
        internal void ReleaseComputeBuffers(params IGgmlHostMemoryOwner[] extra)
        {
            if (!UsesGgml) return;
            var owners = new List<IGgmlHostMemoryOwner> { this };
            if (extra != null) owners.AddRange(extra.Where(o => o != null));
            RunWhileOtherGgmlModelsAreIdle(() =>
            {
                GgmlBasicOps.ReleaseReuseComputeBuffers();
                GgmlOwnerRelease.Release(owners.ToArray());
            });
        }

        /// <summary>The companions' mappings this model opened (the transformer's is its own model's).</summary>
        protected override void AddOwnedHostMemory(List<(IntPtr Base, long Length)> ranges, List<IntPtr> keys)
        {
            GgmlOwnerRelease.AddMappings(ranges, _vaeGguf);
            GgmlOwnerRelease.AddMappings(ranges, _teGguf);
        }

        // ---- not an autoregressive model
        protected override float[] ForwardCore(int[] tokens) =>
            throw new NotSupportedException("QwenImage21DiffusionModel is an image model; use GenerateImage() or EditImage().");

        protected override void ResetKVCacheCore() { }

        public override void WarmUpKernels() { }

        public override void Dispose()
        {
            _disposed = true;
            lock (_generationLock)
            {
                _pipeline?.Dispose();
                Loras?.Dispose();
                Loras = null;
                _vaeSafetensors?.Dispose();
                _vaeGguf?.Dispose();
                _teGguf?.Dispose();
                base.Dispose();
            }
        }
    }
}
