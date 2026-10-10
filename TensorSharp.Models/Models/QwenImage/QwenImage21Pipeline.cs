// Copyright (c) Zhongkai Fu. All rights reserved.
// Licensed under the BSD-3-Clause license in the repository root.
using System;
using System.Buffers;
using System.Buffers.Binary;
using System.Collections.Generic;
using System.Diagnostics;
using System.Globalization;
using System.Security.Cryptography;
using System.Threading;
using TensorSharp.Runtime;

namespace TensorSharp.Models.QwenImage
{
    /// <summary>Qwen-Image-2.1 flow-matching pipeline. Latents are 64-channel /16, without patch packing.</summary>
    internal sealed class QwenImage21Pipeline : IDisposable
    {
        private readonly QwenImageModel _model;
        private QwenImage21Vae _vae;
        private QwenImage21DiT _dit;
        private QwenImage21Vae Vae => _vae ??= new QwenImage21Vae(_model);
        private QwenImage21DiT Dit => _dit ??= new QwenImage21DiT(_model.DitGgufPath, _model.Backend, _model.DitTensorParallelGroup, _model.Loras);

        public QwenImage21Pipeline(QwenImageModel model) => _model = model;

        public RgbImage Run(string prompt, RgbImage[] inputs, QwenImageParams p)
        {
            ArgumentNullException.ThrowIfNull(prompt);
            ArgumentNullException.ThrowIfNull(inputs);
            ArgumentNullException.ThrowIfNull(p);
            if (p.Steps < 0 || !float.IsFinite(p.CfgScale) || p.CfgScale < 0)
                throw new ArgumentException("Steps and CFG must be finite and nonnegative (zero selects the model default).");
            foreach (var input in inputs) ArgumentNullException.ThrowIfNull(input);
            // Parse first: a misspelled setting fails before any work, text-to-image included.
            bool followReferences = QwenImage21Sampling.EditNoiseFollowsReferences(
                Environment.GetEnvironmentVariable(QwenImage21Sampling.EditNoiseVariable));
            var request = new Request(inputs, p, _model.Backend, followReferences);
            var mask = request.Mask;
            var geometry = request.Geometry;
            var (width, height) = geometry.Sampling;
            if (mask?.IsEmpty == true)
            {
                Console.WriteLine("Qwen-Image-2.1: empty edit mask; returning the original image without inference.");
                return mask.UnchangedCopy();
            }
            inputs = request.Inputs;
            if (inputs.Length > 0 && _model.MmprojPath == null)
                throw new InvalidOperationException("Qwen-Image-2.1 editing requires the Qwen3-VL-8B vision projector; set --qwen-image-mmproj or TS_QWEN_IMAGE_MMPROJ.");
            var recipe = _model.SamplingRecipe;
            int h = request.LatentHeight, w = request.LatentWidth, sequence = checked(h * w);
            // Validate the schedule before any encoder or VAE work.
            var (steps, cfg, sigmas) = ResolveSampling(p, recipe, sequence);
            if (_model.Loras is { OutputHeads.Length: > 0 } bundle && bundle.OutputHeads.Length != steps)
                throw new ArgumentException($"The LoRA bundle has one output head per trained step ({bundle.OutputHeads.Length}); it cannot run {steps} steps.");
            // Refuse a size this machine cannot hold before any encoder, transformer or VAE work.
            if (_model.Backend == BackendType.Cpu) CheckCpuMemory(width, height, inputs);
            // After the refusals, so a refused request hashes nothing; 0 for text-to-image.
            ulong noiseStream = request.NoiseStream;
            var total = Stopwatch.StartNew();
            var phase = Stopwatch.StartNew();
            void Phase(string name)
            {
                Console.WriteLine($"  [qwen21-timing] {name}: {phase.Elapsed.TotalMilliseconds:F0}ms");
                if (ReportMemory) Console.WriteLine($"  [qwen21-memory] after {name}: {MemoryLine()}");
                phase.Restart();
            }
            string stream = noiseStream != 0 ? $" (edit noise stream {noiseStream:x16})" : "";
            Console.WriteLine($"Qwen-Image-2.1{(_model.Variant == QwenImageVariant.Turbo ? " Turbo" : "")}: {width}x{height}, " +
                $"{steps} steps, CFG {cfg}, seed {p.Seed}{stream}, {inputs.Length} reference(s)");
            if (inputs.Length > 0 && !followReferences)
                Console.WriteLine($"  [qwen21] {QwenImage21Sampling.EditNoiseVariable}=seed: this edit starts from the seed's text-to-image " +
                    "noise (stable-diffusion.cpp parity); an edit of a picture drawn at this seed and size may retrace it.");
            if (mask != null)
                Console.WriteLine($"  [qwen21-mask] {p.MaskMode.ToString().ToLowerInvariant()}, canvas {mask.Source.Width}x{mask.Source.Height}, " +
                    $"region {mask.X},{mask.Y},{mask.Width},{mask.Height}, feather {p.MaskFeather}; exact protected pixels");
            if (mask == null && geometry.Output != geometry.Sampling)
                Console.WriteLine($"  [qwen21] keeping the source size {geometry.Output.Width}x{geometry.Output.Height}: " +
                    $"sampled at {width}x{height}, then resized to it");
            if (!p.KeepSourceSize && UsesHostCpuAutomaticSize(p, _model.Backend))
                Console.WriteLine($"  automatic size on the cpu backend: {width}x{height} (about " +
                    $"{HostCpuAutomaticArea / (1024 * 1024)} MP). The native 2048x2048 area has four times the tokens and takes " +
                    "about 5x longer per step on a CPU; pass --width/--height (width/height in an API request) for another size.");
            if (recipe is { HasSchedule: true })
                Console.WriteLine(recipe.LogLine(steps, sequence));

            // Drawn before any encoder or VAE work (4 MB at 2048x2048), so a test without weights
            // sees the latent sampling starts from (LatentsDrawn).
            float[] latents = request.InitialLatents();
            LatentsDrawn.Value?.Invoke(latents);
            try
            {
                var refs = new RgbImage[inputs.Length];
                var refTokens = new float[inputs.Length][];
                var refHeights = new int[inputs.Length];
                var refWidths = new int[inputs.Length];
                float[] sourceTokens = null;
                float[] latentMask = mask?.LatentWeights(w, h);
                for (int i = 0; i < inputs.Length; i++)
                {
                    // Vision and VAE must see the SAME geometry: one image slot expands
                    // to four /16 VAE tokens in the single-stream transformer.
                    var (rw, rh) = ResolveReferenceDimensions(inputs[i], width, height);
                    refs[i] = ImageIO.Resize(inputs[i], rw, rh);
                    var latent = Vae.Encode(refs[i]);
                    refHeights[i] = latent.Height;
                    refWidths[i] = latent.Width;
                    refTokens[i] = ToTokens(latent.Data, latent.Height, latent.Width);
                    if (i == 0 && mask != null && !mask.IsFull && rw == width && rh == height)
                        sourceTokens = refTokens[i];
                }
                if (mask != null && !mask.IsFull && sourceTokens == null)
                {
                    // Reference conditioning is capped at 1MP; flow reinjection needs the
                    // canvas encoded at the ACTUAL sampling size, with the same crop transform.
                    var encodedSource = Vae.Encode(ImageIO.Resize(mask.Reference, width, height));
                    sourceTokens = ToTokens(encodedSource.Data, encodedSource.Height, encodedSource.Width);
                }
                if (refs.Length > 0) Phase("VAE encode");

                float[] positive, negative = null;
                int positiveLength, negativeLength = 0;
                int[] positiveSlots, negativeSlots = null;
                // Encoder residency ends before the DiT starts. This also bounds
                // GPU memory on discrete devices; no encoder is needed during denoising.
                using (var conditioner = new QwenImage21Conditioner(_model.TePath, _model.MmprojPath, _model.Backend))
                {
                    (positive, positiveLength, positiveSlots) = conditioner.EncodePrompt(prompt, refs);
                    if (cfg > 1f)
                        (negative, negativeLength, negativeSlots) = conditioner.EncodePrompt(p.NegativePrompt ?? "", refs);
                }
                Phase("text and vision encode");
                _model.ReleaseComputeBuffers();
                // On the cpu backend the conditioner's packed vision weights and working buffers
                // are managed arrays (GBs for an edit). Nothing during the denoise allocates
                // managed memory, so without a collection they stay committed until the VAE.
                if (!_model.UsesGgml) GC.Collect(GC.MaxGeneration, GCCollectionMode.Forced, blocking: true, compacting: false);

                float[] maskNoise = sourceTokens == null ? null : (float[])latents.Clone();
                if (sourceTokens != null)
                    QwenImageEditMask.Reinject(latents, sourceTokens, maskNoise, latentMask, sigmas[0]);
                // Text and reference tokens are modulated at t=0, so their K/V are the
                // same at every step: the first step stores them per CFG branch and the
                // rest compute only the target image. Released before VAE decoding.
                QwenImage21DiT.PrefixCache positiveCache = null, negativeCache = null;
                try
                {
                    positiveCache = QwenImage21DiT.CreatePrefixCache(positive, positiveSlots, refTokens);
                    if (cfg > 1f) negativeCache = QwenImage21DiT.CreatePrefixCache(negative, negativeSlots, refTokens);
                    for (int step = 0; step < steps; step++)
                    {
                        var timer = Stopwatch.StartNew();
                        bool bf16Time = recipe?.TimestepBf16 ?? false;
                        float[] velocity = Dit.Predict(latents, h, w, positive, positiveLength, sigmas[step],
                            positiveSlots, refTokens, refHeights, refWidths, positiveCache, step, bf16Time);
                        if (cfg > 1f)
                        {
                            float[] unconditional = Dit.Predict(latents, h, w, negative, negativeLength, sigmas[step],
                                negativeSlots, refTokens, refHeights, refWidths, negativeCache, step, bf16Time);
                            for (int j = 0; j < velocity.Length; j++)
                                velocity[j] = unconditional[j] + cfg * (velocity[j] - unconditional[j]);
                        }
                        float dt = sigmas[step + 1] - sigmas[step];
                        for (int j = 0; j < latents.Length; j++)
                        {
                            if (!float.IsFinite(velocity[j]))
                                throw new InvalidOperationException($"Qwen-Image-2.1 produced a non-finite velocity at step {step + 1}.");
                            latents[j] += dt * velocity[j];
                        }
                        if (sourceTokens != null)
                            QwenImageEditMask.Reinject(latents, sourceTokens, maskNoise, latentMask, sigmas[step + 1]);
                        if (step == 0)
                        {
                            ReportPrefixCache(positiveCache, "conditional", Dit.TensorParallelRanks);
                            ReportPrefixCache(negativeCache, "negative", Dit.TensorParallelRanks);
                        }
                        string path = positiveCache == null ? "" : $" prefix={positiveCache.LastPath.ToString().ToLowerInvariant()}";
                        Console.WriteLine($"  [qwen21-step] {step + 1}/{steps}: {timer.Elapsed.TotalSeconds:F3}s sigma={sigmas[step]:F6}{path}");
                        RgbImage preview = null;
                        int interval = p.PreviewCount > 0 ? Math.Max(1, (steps + p.PreviewCount) / (p.PreviewCount + 1)) : 0;
                        if (p.OnStep != null && interval > 0 && step + 1 < steps && (step + 1) % interval == 0)
                        {
                            // The VAE expects a clean latent. Euler's current state
                            // still contains noise at sigma_next; x0 = x_next -
                            // sigma_next * velocity is the denoised flow estimate.
                            // Keep the sampling state unchanged by the preview.
                            try
                            {
                                var previewTokens = QwenImage21Sampling.PreviewLatents(latents, velocity, sigmas[step + 1]);
                                if (sourceTokens != null)
                                    QwenImageEditMask.Reinject(previewTokens, sourceTokens, maskNoise, latentMask, 0f);
                                preview = DecodePreview(previewTokens, h, w);
                                if (mask != null) preview = mask.Composite(preview);
                            }
                            catch (Exception error) when (error is not OperationCanceledException)
                            {
                                Console.WriteLine($"  [qwen21] preview decode skipped: {error.Message}");
                            }
                        }
                        p.OnStep?.Invoke(step + 1, steps, preview);
                    }
                }
                finally
                {
                    positiveCache?.Dispose();
                    negativeCache?.Dispose();
                }
                Phase("denoise");
                // Denoising is finished; release resident DiT weights (on the cpu
                // backend, the transformer's activation scratch) before allocating the
                // much larger full-resolution VAE feature maps.
                _model.ReleaseComputeBuffers();
                _dit?.ReleaseScratch();
                var output = Vae.Decode(new VaeLatent(64, h, w, ToChannels(latents, h, w)));
                output = FinishOutput(output, mask, geometry.Output);
                Phase("VAE decode");
                Console.WriteLine($"  [qwen21-timing] total: {total.Elapsed.TotalSeconds:F3}s");
                return output;
            }
            finally
            {
                _model.ReleaseComputeBuffers();
                _dit?.ReleaseScratch();
            }
        }

        /// <summary>
        /// Test seam: called on the calling flow with the latent <see cref="Run"/> samples from,
        /// after the request's refusals and before any encoder, VAE or transformer work. A test
        /// that throws from it runs <see cref="Run"/> with no weights. Null outside tests.
        /// </summary>
        internal static readonly AsyncLocal<Action<float[]>> LatentsDrawn = new();

        /// <summary>
        /// The model-free part of a request, settled before any encoder work: the selection, the
        /// geometry (<see cref="ResolveGeometry"/>), the references the encoders see and the
        /// initial noise. <see cref="Run"/> samples from exactly <see cref="InitialLatents"/>
        /// (<see cref="LatentsDrawn"/> shows it), so the noise's wiring is tested without a model.
        /// </summary>
        internal sealed class Request
        {
            private readonly long _seed;
            private readonly bool _followReferences;
            private ulong? _noiseStream;

            /// <param name="callerInputs">The reference pictures as the caller passed them.</param>
            /// <param name="followReferences">Whether an edit's noise follows its references
            /// (<see cref="QwenImage21Sampling.EditNoiseFollowsReferences"/>).</param>
            internal Request(RgbImage[] callerInputs, QwenImageParams p, BackendType? backend, bool followReferences)
            {
                ArgumentNullException.ThrowIfNull(callerInputs);
                ArgumentNullException.ThrowIfNull(p);
                CallerInputs = callerInputs;
                _seed = p.Seed;
                _followReferences = followReferences;
                RgbImage first = callerInputs.Length > 0 ? callerInputs[0] : null;
                Mask = QwenImageEditMask.Create(p, first);
                Geometry = ResolveGeometry(p, first, Mask, backend);
                Inputs = callerInputs;
                if (Mask is { IsEmpty: false })
                {
                    // Keep every additional reference in its original coordinate system.
                    // The first reference supplies the masked canvas or selected crop.
                    Inputs = (RgbImage[])callerInputs.Clone();
                    Inputs[0] = Mask.Reference;
                }
            }

            /// <summary>The pictures the caller gave, which an edit's noise follows: not the
            /// selection's canvas or crop in <see cref="Inputs"/>, nor the resized copies the
            /// encoders see (Lanczos differs between hosts).</summary>
            internal RgbImage[] CallerInputs { get; }

            /// <summary>The references the encoders see: the caller's, with the selection's
            /// canvas or crop in place of the first.</summary>
            internal RgbImage[] Inputs { get; }

            internal QwenImageEditMask Mask { get; }

            internal ((int Width, int Height) Sampling, (int Width, int Height) Output) Geometry { get; }

            internal int LatentHeight => Geometry.Sampling.Height / 16;

            internal int LatentWidth => Geometry.Sampling.Width / 16;

            /// <summary>The initial noise's Philox stream (<see cref="QwenImage21Sampling.NoiseStream"/>),
            /// hashed on first use so a request refused before sampling hashes nothing.</summary>
            internal ulong NoiseStream => _noiseStream ??= QwenImage21Sampling.NoiseStream(CallerInputs, _followReferences);

            /// <summary>The latent sampling starts from, drawn for the sampling size: an edit that
            /// keeps its source size samples at <see cref="KeptSourceSampling"/>, not at the
            /// source's own size.</summary>
            internal float[] InitialLatents() => QwenImage21Sampling.InitialLatents(_seed, NoiseStream, LatentHeight, LatentWidth);
        }

        /// <summary>
        /// The steps, CFG and steps+1 sigmas a request samples with. A LoRA plug-in's recipe (a
        /// step-distilled adapter's trained schedule) or the checkpoint's own (Turbo's,
        /// <see cref="QwenImage21Turbo"/>) supplies the defaults, otherwise 40 steps on the
        /// scheduler's shifted schedule at CFG 1; explicit steps / CFG still win, and a step
        /// count a recipe has no schedule for is refused (ArgumentException).
        /// </summary>
        internal static (int Steps, float Cfg, float[] Sigmas) ResolveSampling(QwenImageParams p, QwenImage21LoraRecipe recipe, int imageTokens)
        {
            int steps = p.Steps != 0 ? p.Steps : recipe is { DefaultSteps: > 0 } ? recipe.DefaultSteps : 40;
            // The released 2.1 checkpoint is intended for sampling without CFG.
            // An explicit value > 1 still opts into the additional negative pass.
            float cfg = p.CfgScale != 0 ? p.CfgScale : recipe?.Cfg ?? 1f;
            float[] sigmas = recipe is { HasSchedule: true } ? recipe.Sigmas(steps, imageTokens) : QwenImage21Sampling.Sigmas(steps, imageTokens);
            return (steps, cfg, sigmas);
        }

        // TS_QWEN21_MEMORY=1 prints the process memory after every phase (working set, commit,
        // and their peaks so far), for sizing a machine or checking the estimate.
        private static readonly bool ReportMemory = Environment.GetEnvironmentVariable("TS_QWEN21_MEMORY") == "1";

        private static string MemoryLine()
        {
            using var self = Process.GetCurrentProcess();
            self.Refresh();
            static string Mib(long bytes) => (bytes / (1024.0 * 1024.0)).ToString("F0", CultureInfo.InvariantCulture);
            return $"working set {Mib(self.WorkingSet64)} MiB (peak {Mib(self.PeakWorkingSet64)}), " +
                $"private {Mib(self.PrivateMemorySize64)} MiB (peak commit {Mib(self.PeakPagedMemorySize64)}), " +
                $"GC heap {Mib(GC.GetTotalMemory(false))} MiB";
        }

        /// <summary>
        /// Refuses (ArgumentException, a 400 on the server) a size whose estimated peak does not fit in
        /// this machine's memory (<see cref="QwenImage21CpuMemory"/>), and warns when it exceeds the
        /// memory free right now (other processes may give it back, so that is not a refusal).
        /// </summary>
        private void CheckCpuMemory(int width, int height, RgbImage[] inputs)
        {
            // The joint sequence's prefix: the prompt (a few hundred tokens at most) and, per
            // reference, its /16 VAE tokens plus about a quarter as many vision slots.
            long prefix = 256;
            foreach (var input in inputs)
            {
                var (rw, rh) = ResolveReferenceDimensions(input, width, height);
                prefix += (long)(rw / 16) * (rh / 16) * 5 / 4;
            }
            int prefixTokens = (int)Math.Min(int.MaxValue, prefix);
            long transformerBytes = 0;
            try { transformerBytes = new System.IO.FileInfo(_model.DitGgufPath).Length; }
            catch (Exception e) when (e is System.IO.IOException or UnauthorizedAccessException or ArgumentException or NotSupportedException) { }
            if (Environment.GetEnvironmentVariable(QwenImage21CpuMemory.CheckVariable) != "0" &&
                QwenImage21CpuMemory.Refusal(width, height, prefixTokens, transformerBytes, QwenImage21CpuMemory.TotalMemoryBytes()) is string refusal)
                throw new ArgumentException(refusal);
            long needed = QwenImage21CpuMemory.EstimatePeakBytes(width, height, prefixTokens, transformerBytes);
            long free = QwenImage21CpuMemory.FreeMemoryBytes();
            if (free > 0 && needed > free)
                Console.WriteLine(string.Create(CultureInfo.InvariantCulture,
                    $"  [qwen21] WARNING: {width}x{height} needs about {QwenImage21CpuMemory.Gib(needed):F1} GiB at its peak, " +
                    $"but only {QwenImage21CpuMemory.Gib(free):F1} GiB is free right now; close other programs or expect paging."));
        }

        private static void ReportPrefixCache(QwenImage21DiT.PrefixCache cache, string branch, int ranks)
        {
            if (cache == null) return;
            // Each tensor-parallel rank stores its own heads; the info describes rank 0.
            var info = cache.Info;
            string perGpu = ranks > 1 ? $" per GPU ({ranks} GPUs)" : "";
            if (info.State == 1)
                Console.WriteLine($"  [qwen21] prefix KV cache ({branch}): {info.Tokens} tokens, " +
                    $"{info.Bytes / (1024.0 * 1024.0):F1} MiB {TypeName(info.KeyType)}/{TypeName(info.ValueType)}{perGpu}; " +
                    "later steps compute only the target image");
            else if (info.State == 2)
                Console.WriteLine($"  [qwen21] prefix KV cache ({branch}) declined: {info.Tokens} tokens need " +
                    $"{info.Bytes / (1024.0 * 1024.0):F1} MiB{perGpu}; every step recomputes the prefix");
        }

        private static string TypeName(int ggmlType) => ggmlType switch { 0 => "F32", 1 => "F16", 8 => "Q8_0", _ => ggmlType.ToString() };

        private RgbImage DecodePreview(float[] tokens, int height, int width)
        {
            int factor = Math.Max(1, (Math.Max(height, width) + 23) / 24);
            if (factor == 1) return Vae.Decode(new VaeLatent(64, height, width, ToChannels(tokens, height, width)));
            int h = Math.Max(1, height / factor), w = Math.Max(1, width / factor);
            var pooled = new float[64 * h * w];
            for (int y = 0; y < h; y++)
                for (int x = 0; x < w; x++)
                    for (int c = 0; c < 64; c++)
                    {
                        float sum = 0;
                        int count = 0;
                        for (int yy = y * factor; yy < Math.Min(height, (y + 1) * factor); yy++)
                            for (int xx = x * factor; xx < Math.Min(width, (x + 1) * factor); xx++)
                            { sum += tokens[(yy * width + xx) * 64 + c]; count++; }
                        pooled[(c * h + y) * w + x] = sum / count;
                    }
            return Vae.Decode(new VaeLatent(64, h, w, pooled));
        }

        internal const string DefaultWidthVariable = "TS_QWEN_IMAGE_WIDTH";
        internal const string DefaultHeightVariable = "TS_QWEN_IMAGE_HEIGHT";

        // What an omitted targetArea resolves to. The Web UI / API layer resolves it before
        // the request reaches the pipeline, so this value is indistinguishable from "no area".
        private static readonly long AutomaticTargetArea = new QwenImageParams().ResolveTargetArea();

        // The default-size configuration last warned about, so each one is reported once.
        private static string _defaultSizeWarnedFor;

        /// <summary>
        /// The automatic output area on the pure-C# cpu backend: 1024x1024, the area the references
        /// are conditioned at. The native 2048x2048 area has 16384 image tokens against 4096, so a
        /// denoising step takes about 5x as long (attention grows with the square): many minutes
        /// per step on a CPU, hours per image at the default 40 steps. Explicit sizes, an explicit
        /// area and the server's --width/--height are unaffected. ggml_cpu keeps the native area,
        /// as it always has: only the backend this default was introduced with changes behaviour.
        /// </summary>
        internal const long HostCpuAutomaticArea = 1024L * 1024;

        internal static bool IsHostCpu(BackendType backend) => backend == BackendType.Cpu;

        /// <summary>Output geometry. <paramref name="backend"/> selects the automatic area: the
        /// native 2048x2048 one, or <see cref="HostCpuAutomaticArea"/> on the cpu backend (null
        /// keeps the native area).</summary>
        internal static (int Width, int Height) ResolveDimensions(QwenImageParams p, RgbImage reference, BackendType? backend = null)
        {
            int width = p.Width, height = p.Height;
            if (width != 0 || height != 0)
            {
                if (width <= 0 || height <= 0 || width % 32 != 0 || height % 32 != 0)
                    throw new ArgumentException("Qwen-Image-2.1 width and height must both be positive multiples of 32.");
                return (width, height);
            }
            // The server's default size (--width/--height) stands in only for a request that
            // named neither a size nor an area; an explicit area keeps its own geometry.
            if (!AreaRequested(p) && DefaultSize() is { } size)
                return size;
            return DimensionsForArea(reference?.Width ?? 1, reference?.Height ?? 1, AutomaticArea(p, backend));
        }

        private static bool AreaRequested(QwenImageParams p) => p.TargetArea > 0 && p.TargetArea != AutomaticTargetArea;

        // The area an automatic size spends: the request's own, or HostCpuAutomaticArea on the cpu backend.
        private static long AutomaticArea(QwenImageParams p, BackendType? backend) =>
            !AreaRequested(p) && backend is { } b && IsHostCpu(b) ? HostCpuAutomaticArea : p.ResolveTargetArea();

        /// <summary>
        /// Where a request samples and the size it returns. Unmasked, the output is the sampled
        /// image itself, or with <see cref="QwenImageParams.KeepSourceSize"/> the first reference's
        /// exact size (<see cref="KeptSourceSampling"/>). Masked, the output is the source canvas,
        /// which <see cref="QwenImageEditMask.Composite"/> restores whatever the sampling size
        /// (the crop's share of the canvas's), so keeping the source size changes nothing there.
        /// </summary>
        internal static ((int Width, int Height) Sampling, (int Width, int Height) Output) ResolveGeometry(
            QwenImageParams p, RgbImage reference, QwenImageEditMask mask, BackendType? backend = null)
        {
            var kept = KeptSourceSize(p, reference);
            if (mask != null)
            {
                var (width, height) = ResolveDimensions(p, reference, backend);
                return (mask.SamplingDimensions(width, height), (mask.Source.Width, mask.Source.Height));
            }
            if (kept is { } size)
                return (KeptSourceSampling(p, reference, backend), size);
            var sampling = ResolveDimensions(p, reference, backend);
            return (sampling, sampling);
        }

        /// <summary>The size <see cref="QwenImageParams.KeepSourceSize"/> keeps: the first reference's
        /// exact width and height, or null when the request does not ask for it. Refuses a request
        /// that also names a width/height (two answers to one question) or has no reference.</summary>
        internal static (int Width, int Height)? KeptSourceSize(QwenImageParams p, RgbImage reference)
        {
            if (!p.KeepSourceSize) return null;
            if (p.Width != 0 || p.Height != 0)
                throw new ArgumentException("Keeping the source size cannot be combined with an explicit width/height; send one or the other.");
            if (reference == null)
                throw new ArgumentException("Keeping the source size requires an input image to edit.");
            return (reference.Width, reference.Height);
        }

        /// <summary>
        /// The smallest area an edit that keeps its source size samples at, budget permitting:
        /// the area references are conditioned at (<see cref="ResolveReferenceDimensions"/>).
        /// Below it the model works from far fewer image tokens than it was trained at -- a
        /// 100 x 100 icon would have been edited from 36 -- and a selection edit of the same
        /// picture would sample at the full budget.
        /// </summary>
        internal const long KeptSourceSamplingFloor = 1024L * 1024;

        /// <summary>
        /// Where an edit that keeps its source size samples: at about the source's aspect ratio
        /// (the 32-pixel grid rounds it), within the area the request would otherwise sample at,
        /// and at about the source's own area between <see cref="KeptSourceSamplingFloor"/> and
        /// that budget. The budget is the request's targetArea, the automatic area, or the area
        /// of the server's default size (whose own aspect ratio would distort the source once
        /// resized back to it). A picture larger than the budget samples exactly where an
        /// automatic size would and is then resized up to its own size, rather than handed back
        /// smaller; one smaller than the floor samples at the floor and is resized down. In
        /// between, the source holds no detail for more tokens to restore, so it samples at its
        /// own size, and a source on the 32-pixel grid is not resized at all.
        /// </summary>
        internal static (int Width, int Height) KeptSourceSampling(QwenImageParams p, RgbImage source, BackendType? backend = null)
        {
            long budget = !AreaRequested(p) && DefaultSize() is { } size
                ? (long)size.Width * size.Height
                : AutomaticArea(p, backend);
            long own = checked((long)source.Width * source.Height);
            return DimensionsForArea(source.Width, source.Height, Math.Min(budget, Math.Max(own, KeptSourceSamplingFloor)));
        }

        /// <summary>
        /// The decoded picture as the request returns it: composited into the source canvas for a
        /// selection, otherwise resized to <paramref name="output"/> -- a no-op, the same
        /// instance, when the sampling size already is the output size, so nothing changes for a
        /// request that does not keep its source size.
        /// </summary>
        internal static RgbImage FinishOutput(RgbImage decoded, QwenImageEditMask mask, (int Width, int Height) output)
        {
            if (mask != null) return mask.Composite(decoded);
            return decoded.Width == output.Width && decoded.Height == output.Height
                ? decoded
                : ImageIO.Resize(decoded, output.Width, output.Height);
        }

        /// <summary>Whether <see cref="ResolveDimensions"/> picks <see cref="HostCpuAutomaticArea"/>.</summary>
        internal static bool UsesHostCpuAutomaticSize(QwenImageParams p, BackendType backend) =>
            IsHostCpu(backend) && p.Width == 0 && p.Height == 0 && !AreaRequested(p) && DefaultSize() == null;

        /// <summary>The operator's default output size (TS_QWEN_IMAGE_WIDTH/HEIGHT, which the
        /// server's --width/--height set), or null when there is none usable. It is a fallback
        /// for every request that names no size, so a bad value must not fail each of them:
        /// sides that are not multiples of 32 snap down (minimum 32), and a half-configured or
        /// unparsable pair is ignored. Either is reported once per configuration.</summary>
        internal static (int Width, int Height)? DefaultSize()
        {
            string rawWidth = Environment.GetEnvironmentVariable(DefaultWidthVariable)?.Trim();
            string rawHeight = Environment.GetEnvironmentVariable(DefaultHeightVariable)?.Trim();
            bool hasWidth = !string.IsNullOrEmpty(rawWidth), hasHeight = !string.IsNullOrEmpty(rawHeight);
            if (!hasWidth && !hasHeight)
                return null;

            string configuration = rawWidth + "x" + rawHeight;
            const string automatic = "requests that name no size keep the automatic size " +
                "(the native 2048x2048 area, 1024x1024 on the cpu backend, following the first reference " +
                "image's aspect ratio on an edit).";
            if (hasWidth != hasHeight)
            {
                string set = hasWidth ? DefaultWidthVariable : DefaultHeightVariable;
                string missing = hasWidth ? DefaultHeightVariable : DefaultWidthVariable;
                WarnDefaultSizeOnce(configuration,
                    $"{set} is set without {missing}; the default image size needs both (the server's --width " +
                    $"and --height). Ignoring it: {automatic}");
                return null;
            }

            if (!TryParsePixels(rawWidth, out int width) || !TryParsePixels(rawHeight, out int height))
            {
                WarnDefaultSizeOnce(configuration,
                    $"{DefaultWidthVariable}={rawWidth} / {DefaultHeightVariable}={rawHeight} is not a pair of " +
                    $"positive pixel counts. Ignoring it: {automatic}");
                return null;
            }

            int snappedWidth = Math.Max(32, width / 32 * 32), snappedHeight = Math.Max(32, height / 32 * 32);
            if (snappedWidth != width || snappedHeight != height)
            {
                WarnDefaultSizeOnce(configuration,
                    $"the default image size {width}x{height} ({DefaultWidthVariable}/{DefaultHeightVariable}) " +
                    $"is not a multiple of 32 on both sides; requests that name no size render at " +
                    $"{snappedWidth}x{snappedHeight} instead.");
            }
            return (snappedWidth, snappedHeight);
        }

        private static bool TryParsePixels(string raw, out int pixels) =>
            int.TryParse(raw, System.Globalization.NumberStyles.Integer,
                System.Globalization.CultureInfo.InvariantCulture, out pixels) && pixels > 0;

        private static void WarnDefaultSizeOnce(string configuration, string message)
        {
            if (string.Equals(System.Threading.Interlocked.Exchange(ref _defaultSizeWarnedFor, configuration),
                    configuration, StringComparison.Ordinal))
                return;
            Console.Error.WriteLine($"[qwen-image] WARNING: {message} Reported once.");
        }

        private static (int Width, int Height) DimensionsForArea(int sourceWidth, int sourceHeight, long area)
        {
            double ratio = (double)sourceWidth / sourceHeight;
            // Match std::round in sd.cpp, including exact half-grid values;
            // keep the grid multiplication checked as well as the conversion.
            return (Math.Max(32, checked((int)Math.Round(Math.Sqrt(area * ratio) / 32, MidpointRounding.AwayFromZero) * 32)),
                Math.Max(32, checked((int)Math.Round(Math.Sqrt(area / ratio) / 32, MidpointRounding.AwayFromZero) * 32)));
        }

        internal static (int Width, int Height) ResolveReferenceDimensions(RgbImage image, int outputWidth, int outputHeight)
        {
            // Diffusers conditions 2K output on approximately 1MP references.
            // Keep small previews small as well; doubling output resolution must
            // not quadruple every reference's VAE, vision and transformer work.
            long area = Math.Min(1024L * 1024, checked((long)outputWidth * outputHeight));
            return DimensionsForArea(image.Width, image.Height, area);
        }

        internal static float[] ToTokens(float[] chw, int height, int width)
        {
            int count = checked(height * width);
            if (chw.Length != checked(count * 64)) throw new ArgumentException("Expected 64-channel latent.");
            var result = new float[chw.Length];
            for (int i = 0; i < count; i++)
                for (int c = 0; c < 64; c++) result[i * 64 + c] = chw[c * count + i];
            return result;
        }

        internal static float[] ToChannels(float[] tokens, int height, int width)
        {
            int count = checked(height * width);
            if (tokens.Length != checked(count * 64)) throw new ArgumentException("Expected 64-channel latent.");
            var result = new float[tokens.Length];
            for (int i = 0; i < count; i++)
                for (int c = 0; c < 64; c++) result[c * count + i] = tokens[i * 64 + c];
            return result;
        }

        /// <summary>Release the transformer so the next request rebuilds it (a LoRA change).</summary>
        internal void ResetTransformer()
        {
            _dit?.Dispose();
            _dit = null;
        }

        public void Dispose()
        {
            _dit?.Dispose();
            _vae?.Dispose();
        }
    }

    internal static class QwenImage21Sampling
    {
        internal static float[] PreviewLatents(ReadOnlySpan<float> updatedLatents, ReadOnlySpan<float> velocity, float nextSigma)
        {
            if (updatedLatents.Length != velocity.Length)
                throw new ArgumentException("Preview latent and velocity lengths must match.");
            if (!float.IsFinite(nextSigma) || nextSigma < 0f || nextSigma > 1f)
                throw new ArgumentOutOfRangeException(nameof(nextSigma));
            var clean = new float[updatedLatents.Length];
            for (int i = 0; i < clean.Length; i++)
                clean[i] = updatedLatents[i] - nextSigma * velocity[i];
            return clean;
        }

        internal static float[] Sigmas(int steps, int imageTokens)
        {
            if (steps <= 0 || imageTokens <= 0) throw new ArgumentOutOfRangeException();
            // Official Qwen/Qwen-Image-2.1 scheduler_config.json: dynamic exponential
            // shift (256 -> 0.5, 8192 -> 0.9), followed by shift_terminal=0.02.
            // The anchors are interpolated/extrapolated, including 16384 tokens at 2K.
            // One step has no interval to stretch; retain a single Euler update to zero.
            if (steps == 1) return new[] { 1f, 0f };
            double mu = 0.5 + (imageTokens - 256) * (0.9 - 0.5) / (8192 - 256);
            double exp = Math.Exp(mu);
            double lastShifted = exp / (exp + steps - 1);
            double terminalScale = (1 - lastShifted) / (1 - 0.02);
            var result = new float[steps + 1];
            for (int i = 0; i < steps; i++)
            {
                // Equivalent to the pipeline's linspace(1, 1 / steps, steps).
                double t = (double)(steps - i) / steps;
                double shifted = exp / (exp + (1 / t - 1));
                result[i] = (float)(1 - (1 - shifted) / terminalScale);
            }
            result[steps - 1] = 0.02f;
            return result;
        }

        /// <summary>
        /// <c>references</c> (the default): an edit's noise also depends on its reference
        /// pictures (<see cref="NoiseStream"/>). <c>seed</c>: an edit starts from the seed's
        /// text-to-image noise, as stable-diffusion.cpp and diffusers do, for matched-noise A/B runs.
        /// </summary>
        internal const string EditNoiseVariable = "TS_QWEN21_EDIT_NOISE";

        /// <summary>Whether an edit's noise follows its references; anything other than
        /// <c>references</c> (or unset) and <c>seed</c> is refused.</summary>
        internal static bool EditNoiseFollowsReferences(string value) =>
            (value ?? "").Trim().ToLowerInvariant() switch
            {
                "" or "references" => true,
                "seed" => false,
                _ => throw new ArgumentException($"{EditNoiseVariable}='{value}' is not one of references, seed."),
            };

        /// <summary>
        /// The Philox stream of a request's initial noise: 0 for text-to-image (and for an edit
        /// when <paramref name="followReferences"/> is false), otherwise <see cref="ReferenceStream"/>
        /// of <paramref name="callerInputs"/>, the pictures as the caller passed them.
        /// </summary>
        internal static ulong NoiseStream(IReadOnlyList<RgbImage> callerInputs, bool followReferences)
        {
            ArgumentNullException.ThrowIfNull(callerInputs);
            return callerInputs.Count == 0 || !followReferences ? 0 : ReferenceStream(callerInputs);
        }

        /// <summary>
        /// The initial latent, token-major: <see cref="Noise(int, long, ulong)"/> drawn in CHW
        /// order and transposed, so equal seeds and streams identify equal noise.
        /// </summary>
        internal static float[] InitialLatents(long seed, ulong stream, int latentHeight, int latentWidth) =>
            QwenImage21Pipeline.ToTokens(Noise(checked(latentHeight * latentWidth * 64), seed, stream), latentHeight, latentWidth);

        /// <summary>
        /// A nonzero stream id for an edit, keyed to its reference pictures. Without it, the noise
        /// depends on the seed and size alone, so an edit of a picture at the seed and size it was
        /// drawn at starts from the very latent that became that picture, and Qwen-Image 2.1 can
        /// retrace the picture instead of following the instruction. A fox drawn at seed 0
        /// (1024x1024, CFG 1) and changed at seed 0 came back over-sharpened, still in the snow,
        /// for "change the background to a sandy beach" (at 12 and 40 steps, and with a speed
        /// LoRA); drawn and changed at seed 5 it did the same; the seed-5 fox changed at seed 0,
        /// and the seed-0 fox at seeds 1, 2, 3 and 7, got the beach. A teapot recolored at CFG 6
        /// from its own seed's noise did change, so the retrace is likely rather than certain.
        ///
        /// <para>SHA-256 over a versioned encoding: the domain string, the reference count, then
        /// per reference its width, height, an alpha flag, its RGB quantized to 8 bits as a PNG
        /// stores it (<see cref="Quantize8"/>), and the quantized alpha when any pixel is not
        /// opaque. Quantizing makes a picture kept in memory and the same picture reloaded from its
        /// PNG give one stream. Every host decodes a PNG TensorSharp wrote, or any PNG without
        /// colour information, to the same bytes; the Apple provider converts one that embeds a
        /// colour profile (a macOS or iPhone screenshot) to sRGB, as it does an opaque one whose
        /// gAMA or cHRM describes another space (eng/validation/apple-png-decode-check.py), and a
        /// JPEG or HEIC decoder may differ by a few levels. Such a picture gets another stream there.
        /// All-opaque alpha hashes as no alpha: the VAE reads both the same way.
        /// The first eight digest bytes, little-endian, are the id; 0 (text-to-image) maps to 1.</para>
        /// </summary>
        internal static ulong ReferenceStream(IReadOnlyList<RgbImage> references)
        {
            ArgumentNullException.ThrowIfNull(references);
            if (references.Count == 0) return 0;
            using var hash = IncrementalHash.CreateHash(HashAlgorithmName.SHA256);
            hash.AppendData("TensorSharp/qwen-image-2.1/edit-noise/v1"u8);
            Span<byte> header = stackalloc byte[9];
            BinaryPrimitives.WriteInt32LittleEndian(header, references.Count);
            hash.AppendData(header[..4]);
            byte[] buffer = ArrayPool<byte>.Shared.Rent(64 * 1024);
            try
            {
                foreach (var image in references)
                {
                    ArgumentNullException.ThrowIfNull(image);
                    bool translucent = false;
                    ReadOnlySpan<float> alphas = image.Alpha;
                    for (int i = 0; i < alphas.Length && !translucent; i++) translucent = Quantize8(alphas[i]) != 255;
                    BinaryPrimitives.WriteInt32LittleEndian(header, image.Width);
                    BinaryPrimitives.WriteInt32LittleEndian(header[4..], image.Height);
                    header[8] = translucent ? (byte)1 : (byte)0;
                    hash.AppendData(header);
                    AppendQuantized(hash, image.Pixels, buffer);
                    if (translucent) AppendQuantized(hash, image.Alpha, buffer);
                }
            }
            finally
            {
                ArrayPool<byte>.Shared.Return(buffer);
            }
            Span<byte> digest = stackalloc byte[32];
            hash.GetHashAndReset(digest);
            ulong stream = BinaryPrimitives.ReadUInt64LittleEndian(digest);
            return stream == 0 ? 1 : stream;
        }

        // One plain loop per chunk: the Mono hosts run vector helpers slower than this.
        private static void AppendQuantized(IncrementalHash hash, float[] values, byte[] buffer)
        {
            for (int start = 0; start < values.Length; start += buffer.Length)
            {
                ReadOnlySpan<float> source = values.AsSpan(start, Math.Min(buffer.Length, values.Length - start));
                Span<byte> bytes = buffer.AsSpan(0, source.Length);
                for (int i = 0; i < source.Length; i++) bytes[i] = Quantize8(source[i]);
                hash.AppendData(bytes);
            }
        }

        /// <summary>ImageIO's float-to-byte rule (round half up, clamped), with NaN as 0 on every runtime.</summary>
        [System.Runtime.CompilerServices.MethodImpl(System.Runtime.CompilerServices.MethodImplOptions.AggressiveInlining)]
        internal static byte Quantize8(float value)
        {
            float scaled = value * 255f + 0.5f;
            return scaled >= 0f ? scaled <= 255f ? (byte)scaled : (byte)255 : (byte)0;
        }

        // Philox4x32-10 + Box-Muller, matching stable-diffusion.cpp --rng cuda.
        // Generate in CHW order before transposing, so equal seeds identify equal noise.
        internal static float[] Noise(int count, long seed) => Noise(count, seed, 0);

        /// <summary>
        /// <see cref="Noise(int, long)"/> on another Philox stream. sd.cpp's counter is
        /// (offset, 0, i, 0); the stream fills the two words it leaves at zero, so stream 0 is
        /// its sequence bit for bit and the key keeps meaning the seed (mixing the stream into
        /// the key would make an edit's noise some other seed's text-to-image noise). For one key
        /// Philox maps distinct counters to distinct 128-bit blocks; Box-Muller reads two of the
        /// four words and rounds to float, so two streams' values coincide only by chance.
        /// Word 0 stays free for a sampler that draws more than once.
        /// </summary>
        internal static float[] Noise(int count, long seed, ulong stream)
        {
            var result = new float[count];
            const float inv32 = 2.3283064e-10f;
            const float inv32Tau = inv32 * 6.2831855f;
            uint streamLow = (uint)stream, streamHigh = (uint)(stream >> 32);
            for (int i = 0; i < count; i++)
            {
                uint a = 0, b = streamLow, c = (uint)i, d = streamHigh;
                uint k0 = (uint)seed, k1 = (uint)((ulong)seed >> 32);
                for (int round = 0; round < 10; round++)
                {
                    ulong p0 = (ulong)a * 0xD2511F53u, p1 = (ulong)c * 0xCD9E8D57u;
                    (a, b, c, d) = ((uint)(p1 >> 32) ^ b ^ k0, (uint)p1, (uint)(p0 >> 32) ^ d ^ k1, (uint)p0);
                    k0 = unchecked(k0 + 0x9E3779B9u);
                    k1 = unchecked(k1 + 0xBB67AE85u);
                }
                float u = (float)a * inv32 + inv32 / 2;
                float v = (float)b * inv32Tau + inv32Tau / 2;
                result[i] = (float)(Math.Sqrt(-2f * Math.Log(u)) * Math.Sin(v));
            }
            return result;
        }
    }
}
