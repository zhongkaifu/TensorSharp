// Copyright (c) Zhongkai Fu. All rights reserved.
// https://github.com/zhongkaifu/TensorSharp
//
// This file is part of TensorSharp.
//
// TensorSharp is licensed under the BSD-3-Clause license found in the LICENSE file in the root directory of this source tree.
using System;
using System.Globalization;
using TensorSharp.Models.QwenImage;
using TensorSharp.Models.SharedDiffusion;
using TensorSharp.Runtime;

namespace TensorSharp.Models.QwenImage21
{
    /// <summary>
    /// Where a Qwen-Image-2.1 request samples and the size it returns: an explicit size (multiples of 32), the server's default
    /// size, or an area (2048 x 2048 by default, 1024 x 1024 on the pure-C# cpu backend) at the first reference's aspect; an
    /// edit that keeps its source size samples near its source's own area and returns the source's exact size; a masked edit
    /// returns its source canvas. References are conditioned at about 1 MP (or the output's area if smaller).
    /// </summary>
    public static class QwenImage21Canvas
    {
        public const string DefaultWidthVariable = "TS_QWEN_IMAGE_WIDTH";
        public const string DefaultHeightVariable = "TS_QWEN_IMAGE_HEIGHT";

        /// <summary>The area a request that names none samples at.</summary>
        public const long AutomaticTargetArea = 2048L * 2048;

        /// <summary>
        /// The automatic output area on the pure-C# cpu backend: 1024x1024, the area the references are conditioned at. The
        /// native 2048x2048 area has 16384 image tokens against 4096, so a denoising step takes about 5x as long on a CPU.
        /// Explicit sizes, an explicit area and the server's --width/--height are unaffected; ggml_cpu keeps the native area.
        /// </summary>
        public const long HostCpuAutomaticArea = 1024L * 1024;

        /// <summary>The smallest area an edit that keeps its source size samples at, budget permitting: the area references
        /// are conditioned at.</summary>
        public const long KeptSourceSamplingFloor = 1024L * 1024;

        internal static bool IsHostCpu(BackendType backend) => backend == BackendType.Cpu;

        private static bool AreaRequested(DiffusionRequest r) => r.TargetArea > 0 && r.TargetArea != AutomaticTargetArea;

        private static long ResolveTargetArea(DiffusionRequest r) => r.TargetArea > 0 ? r.TargetArea : AutomaticTargetArea;

        private static long AutomaticArea(DiffusionRequest r, BackendType? backend) =>
            !AreaRequested(r) && backend is { } b && IsHostCpu(b) ? HostCpuAutomaticArea : ResolveTargetArea(r);

        /// <summary>Output geometry. <paramref name="backend"/> selects the automatic area (null keeps the native one).</summary>
        public static (int Width, int Height) ResolveDimensions(DiffusionRequest r, RgbImage reference, BackendType? backend = null)
        {
            int width = r.Width, height = r.Height;
            if (width != 0 || height != 0)
            {
                if (width <= 0 || height <= 0 || width % 32 != 0 || height % 32 != 0)
                    throw new ArgumentException("Qwen-Image-2.1 width and height must both be positive multiples of 32.");
                return (width, height);
            }
            // The server's default size (--width/--height) stands in only for a request that named neither a size nor an
            // area; an explicit area keeps its own geometry.
            if (!AreaRequested(r) && DefaultSize() is { } size)
                return size;
            return DimensionsForArea(reference?.Width ?? 1, reference?.Height ?? 1, AutomaticArea(r, backend));
        }

        /// <summary>Where a request samples and the size it returns (<see cref="DiffusionCanvas"/>).</summary>
        public static DiffusionCanvas Resolve(DiffusionRequest r, RgbImage reference, EditMask mask, BackendType? backend = null)
        {
            var kept = KeptSourceSize(r, reference);
            if (mask != null)
            {
                var (width, height) = ResolveDimensions(r, reference, backend);
                return new DiffusionCanvas(mask.SamplingDimensions(width, height), (mask.Source.Width, mask.Source.Height));
            }
            if (kept is { } size)
                return new DiffusionCanvas(KeptSourceSampling(r, reference, backend), size);
            var sampling = ResolveDimensions(r, reference, backend);
            return new DiffusionCanvas(sampling, sampling);
        }

        /// <summary>The size keepSourceSize keeps: the first reference's exact size, or null when not asked for. Refuses a
        /// request that also names a width/height or has no reference.</summary>
        public static (int Width, int Height)? KeptSourceSize(DiffusionRequest r, RgbImage reference)
        {
            if (!r.KeepSourceSize) return null;
            if (r.Width != 0 || r.Height != 0)
                throw new ArgumentException("Keeping the source size cannot be combined with an explicit width/height; send one or the other.");
            if (reference == null)
                throw new ArgumentException("Keeping the source size requires an input image to edit.");
            return (reference.Width, reference.Height);
        }

        /// <summary>
        /// Where an edit that keeps its source size samples: at about the source's aspect ratio, within the area the request
        /// would otherwise sample at, and at about the source's own area between <see cref="KeptSourceSamplingFloor"/> and
        /// that budget.
        /// </summary>
        public static (int Width, int Height) KeptSourceSampling(DiffusionRequest r, RgbImage source, BackendType? backend = null)
        {
            long budget = !AreaRequested(r) && DefaultSize() is { } size
                ? (long)size.Width * size.Height
                : AutomaticArea(r, backend);
            long own = checked((long)source.Width * source.Height);
            return DimensionsForArea(source.Width, source.Height, Math.Min(budget, Math.Max(own, KeptSourceSamplingFloor)));
        }

        /// <summary>Whether <see cref="ResolveDimensions"/> picks <see cref="HostCpuAutomaticArea"/>.</summary>
        public static bool UsesHostCpuAutomaticSize(DiffusionRequest r, BackendType backend) =>
            IsHostCpu(backend) && r.Width == 0 && r.Height == 0 && !AreaRequested(r) && DefaultSize() == null;

        // The default-size configuration last warned about, so each one is reported once.
        private static string _defaultSizeWarnedFor;

        /// <summary>The operator's default output size (TS_QWEN_IMAGE_WIDTH/HEIGHT, which the server's --width/--height set), or
        /// null when there is none usable. Sides that are not multiples of 32 snap down (minimum 32), and a half-configured or
        /// unparsable pair is ignored; either is reported once per configuration.</summary>
        public static (int Width, int Height)? DefaultSize()
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
            int.TryParse(raw, NumberStyles.Integer, CultureInfo.InvariantCulture, out pixels) && pixels > 0;

        private static void WarnDefaultSizeOnce(string configuration, string message)
        {
            if (string.Equals(System.Threading.Interlocked.Exchange(ref _defaultSizeWarnedFor, configuration),
                    configuration, StringComparison.Ordinal))
                return;
            Console.Error.WriteLine($"[qwen-image] WARNING: {message} Reported once.");
        }

        /// <summary>The 32-pixel grid size of about <paramref name="area"/> pixels at the source's aspect (std::round as sd.cpp,
        /// half-grid values away from zero).</summary>
        public static (int Width, int Height) DimensionsForArea(int sourceWidth, int sourceHeight, long area)
        {
            double ratio = (double)sourceWidth / sourceHeight;
            return (Math.Max(32, checked((int)Math.Round(Math.Sqrt(area * ratio) / 32, MidpointRounding.AwayFromZero) * 32)),
                Math.Max(32, checked((int)Math.Round(Math.Sqrt(area / ratio) / 32, MidpointRounding.AwayFromZero) * 32)));
        }

        /// <summary>A reference's conditioning size: about 1 MP (diffusers conditions 2K output on about 1 MP references), or
        /// the output's area if smaller, at the reference's aspect.</summary>
        public static (int Width, int Height) ReferenceDimensions(RgbImage image, int outputWidth, int outputHeight)
        {
            long area = Math.Min(1024L * 1024, checked((long)outputWidth * outputHeight));
            return DimensionsForArea(image.Width, image.Height, area);
        }
    }
}
