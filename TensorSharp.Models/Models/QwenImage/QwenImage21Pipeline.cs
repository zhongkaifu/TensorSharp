// Copyright (c) Zhongkai Fu. All rights reserved.
// Licensed under the BSD-3-Clause license in the repository root.
using System;
using System.Diagnostics;
using TensorSharp.GGML;

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
            if (inputs.Length > 0 && _model.MmprojPath == null)
                throw new InvalidOperationException("Qwen-Image-2.1 editing requires the Qwen3-VL-8B vision projector; set --qwen-image-mmproj or TS_QWEN_IMAGE_MMPROJ.");

            var (width, height) = ResolveDimensions(p, inputs.Length > 0 ? inputs[0] : null);
            // A LoRA plug-in's recipe (a step-distilled adapter's trained schedule) supplies
            // the defaults; explicit steps / CFG still win.
            var recipe = _model.Loras?.Recipe;
            int steps = p.Steps != 0 ? p.Steps : recipe is { DefaultSteps: > 0 } ? recipe.DefaultSteps : 40;
            // The released 2.1 checkpoint is intended for sampling without CFG.
            // An explicit value > 1 still opts into the additional negative pass.
            float cfg = p.CfgScale != 0 ? p.CfgScale : recipe?.Cfg ?? 1f;
            int h = height / 16, w = width / 16, sequence = checked(h * w);
            // Validate the schedule before any encoder or VAE work.
            float[] sigmas = recipe is { HasSchedule: true } ? recipe.Sigmas(steps, sequence) : QwenImage21Sampling.Sigmas(steps, sequence);
            if (_model.Loras is { OutputHeads.Length: > 0 } bundle && bundle.OutputHeads.Length != steps)
                throw new ArgumentException($"The LoRA bundle has one output head per trained step ({bundle.OutputHeads.Length}); it cannot run {steps} steps.");
            var total = Stopwatch.StartNew();
            var phase = Stopwatch.StartNew();
            void Phase(string name)
            {
                Console.WriteLine($"  [qwen21-timing] {name}: {phase.Elapsed.TotalMilliseconds:F0}ms");
                phase.Restart();
            }
            Console.WriteLine($"Qwen-Image-2.1: {width}x{height}, {steps} steps, CFG {cfg}, seed {p.Seed}, {inputs.Length} reference(s)");
            if (recipe is { HasSchedule: true })
                Console.WriteLine($"  [lora] sampling recipe ({System.IO.Path.GetFileName(recipe.Source)}): {recipe.Describe(steps, sequence)}");

            try
            {
                var refs = new RgbImage[inputs.Length];
                var refTokens = new float[inputs.Length][];
                var refHeights = new int[inputs.Length];
                var refWidths = new int[inputs.Length];
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
                GgmlBasicOps.ReleaseReuseComputeBuffers();
                GgmlBasicOps.ClearHostBufferCache();

                float[] latents = ToTokens(QwenImage21Sampling.Noise(checked(sequence * 64), p.Seed), h, w);
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
                            try { preview = DecodePreview(QwenImage21Sampling.PreviewLatents(latents, velocity, sigmas[step + 1]), h, w); }
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
                GgmlBasicOps.ReleaseReuseComputeBuffers();
                // Denoising is finished; release resident DiT weights before
                // allocating the much larger full-resolution VAE feature maps.
                GgmlBasicOps.ClearHostBufferCache();
                var output = Vae.Decode(new VaeLatent(64, h, w, ToChannels(latents, h, w)));
                Phase("VAE decode");
                Console.WriteLine($"  [qwen21-timing] total: {total.Elapsed.TotalSeconds:F3}s");
                return output;
            }
            finally
            {
                GgmlBasicOps.ReleaseReuseComputeBuffers();
                GgmlBasicOps.ClearHostBufferCache();
            }
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

        internal static (int Width, int Height) ResolveDimensions(QwenImageParams p, RgbImage reference)
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
            bool areaRequested = p.TargetArea > 0 && p.TargetArea != AutomaticTargetArea;
            if (!areaRequested && DefaultSize() is { } size)
                return size;
            long area = p.ResolveTargetArea();
            return DimensionsForArea(reference?.Width ?? 1, reference?.Height ?? 1, area);
        }

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
                "(the native 2048x2048 area, following the first reference image's aspect ratio on an edit).";
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

        // Philox4x32-10 + Box-Muller, matching stable-diffusion.cpp --rng cuda.
        // Generate in CHW order before transposing, so equal seeds identify equal noise.
        internal static float[] Noise(int count, long seed)
        {
            var result = new float[count];
            const float inv32 = 2.3283064e-10f;
            const float inv32Tau = inv32 * 6.2831855f;
            for (int i = 0; i < count; i++)
            {
                uint a = 0, b = 0, c = (uint)i, d = 0;
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
