// Copyright (c) Zhongkai Fu. All rights reserved.
// Licensed under the BSD-3-Clause license in the repository root.
using System;
using System.Collections.Generic;
using System.Linq;
using TensorSharp.Models.Video;

namespace TensorSharp.Models.Ltx25;

/// <summary>A normalized channel-major VAE image and its output pixel-frame anchor.</summary>
public sealed record Ltx25EncodedImage(float[] Latent, int FrameIndex, float Strength = 1);

/// <summary>Official combined image conditioning: replace frame zero and append
/// the other guides with exact pixel-time positions. Given image guides do not
/// carry the absolute-position marker used for generated keyframe slots.</summary>
public static class Ltx25ImageConditioning
{
    public const int MaxImages = 16;

    public static IReadOnlyList<VideoImageCondition> Resolve(VideoGenerationParams parameters, int frames)
    {
        ArgumentNullException.ThrowIfNull(parameters);
        if (frames <= 0) throw new ArgumentOutOfRangeException(nameof(frames));
        var images = new List<VideoImageCondition>();
        if (parameters.Image != null || parameters.ImageBytes is { Length: > 0 } || !string.IsNullOrWhiteSpace(parameters.ImagePath))
            images.Add(new() { Image = parameters.Image, ImageBytes = parameters.ImageBytes, ImagePath = parameters.ImagePath });
        if (parameters.EndImage != null || !string.IsNullOrWhiteSpace(parameters.EndImagePath))
            images.Add(new() { Image = parameters.EndImage, ImagePath = parameters.EndImagePath, FrameIndex = frames - 1 });
        if (parameters.ReferenceImagePaths is { Count: > 0 } references)
        {
            if (images.Count > 0 || parameters.ImageConditions is { Count: > 0 })
                throw new ArgumentException("Use ordered reference images or explicit image conditions, rather than combining both.");
            for (int i = 0; i < references.Count; ++i)
                images.Add(new() { ImagePath = references[i], FrameIndex = references.Count == 1 ? 0 :
                    checked((int)Math.Round(i * (frames - 1d) / (references.Count - 1), MidpointRounding.ToEven)) });
        }
        if (parameters.ImageConditions != null)
            foreach (var image in parameters.ImageConditions)
            {
                if (image == null) throw new ArgumentException("Image conditions cannot contain null entries.");
                images.Add(new() { Image = image.Image, ImageBytes = image.ImageBytes, ImagePath = image.ImagePath,
                    FrameIndex = image.FrameIndex == -1 ? frames - 1 : image.FrameIndex, Strength = image.Strength });
            }
        if (images.Count > MaxImages) throw new ArgumentException($"LTX accepts at most {MaxImages} image guides per request.");
        var positions = new HashSet<int>();
        foreach (var image in images)
        {
            if (image.Image == null && image.ImageBytes is not { Length: > 0 } && string.IsNullOrWhiteSpace(image.ImagePath))
                throw new ArgumentException("Each image condition requires an image.");
            if (image.FrameIndex < 0 || image.FrameIndex >= frames)
                throw new ArgumentException($"Image frame index must be in [0,{frames - 1}] or -1 for the last frame.");
            if (!float.IsFinite(image.Strength) || image.Strength < 0 || image.Strength > 1)
                throw new ArgumentException("Image conditioning strength must be finite and in [0,1].");
            if (!positions.Add(image.FrameIndex)) throw new ArgumentException("Only one image guide is allowed at each output frame.");
        }
        if (images.Count > 0 && parameters.Mode is "t2v" or "t2va")
            throw new ArgumentException("Text-only mode cannot include images; use i2v, fl2v or keyframes.");
        if (images.Count == 0 && parameters.Mode is "i2v" or "fl2v" or "keyframes")
            throw new ArgumentException("The selected image generation mode requires images.");
        // i2v animates a first frame; an end image alone would otherwise become a last-frame guide unasked.
        if (parameters.Mode == "i2v" && !positions.Contains(0))
            throw new ArgumentException("i2v animates a first-frame image; to pin only the last frame use fl2v.");
        // fl2v pins the last frame and optionally the first, as MiniMax-H3's fl2v does (sglang [-1] / [0,-1]).
        if (parameters.Mode == "fl2v" && (images.Count is < 1 or > 2 || !positions.Contains(frames - 1)
                                          || images.Count == 2 && !positions.Contains(0)))
            throw new ArgumentException("fl2v pins a last-frame image, optionally with a first-frame image; use keyframes for other frames.");
        return images;
    }

    /// <summary>Noise already includes storage for all appended guides. Audio is
    /// unchanged. Encoded image storage is [channel,height,width], not token-major.
    /// A refinement pass supplies its token-major <paramref name="rawNoise"/> and
    /// <paramref name="noiseScale"/>: like GaussianNoiser after the conditionings,
    /// the first-frame guide blends its clean latent with the caller's already
    /// noised prior-stage latent. Appended guides have a zero prior latent, so
    /// they blend with scale·noise instead. Unguided tokens are unchanged.</summary>
    public static Ltx25DenoiserInput Apply(Ltx25DenoiserInput initial, Ltx25GenerationSettings settings,
        IReadOnlyList<Ltx25EncodedImage> images, float[] rawNoise = null, float noiseScale = 1)
    {
        ArgumentNullException.ThrowIfNull(initial);
        ArgumentNullException.ThrowIfNull(settings);
        ArgumentNullException.ThrowIfNull(images);
        if (!float.IsFinite(noiseScale) || noiseScale < 0 || noiseScale > 1 || (rawNoise == null) != (noiseScale == 1) ||
            rawNoise != null && rawNoise.Length != initial.Video.Length)
            throw new ArgumentException("A refinement pass supplies its raw noise with a scale in [0,1).");
        if (images.Count == 0) return initial;
        int spatial = checked(settings.LatentHeight * settings.LatentWidth);
        int guides = images.Count(image => image.FrameIndex != 0);
        int tokens = checked(settings.VideoTokens + guides * spatial);
        if (initial.VideoTokens != tokens || initial.Video.Length != checked(tokens * 128))
            throw new ArgumentException("Initial noise must include the appended image guide tokens.");
        var video = (float[])initial.Video.Clone();
        var clean = new float[video.Length];
        var mask = Enumerable.Repeat(1f, tokens).ToArray();
        // The first canvas latent frame always encodes a single pixel frame;
        // upstream marks it even in text-to-video. Appended given guides stay 0.
        var keyframes = new float[tokens];
        Array.Fill(keyframes, 1f, 0, spatial);
        var positions = new float[checked(tokens * 6)];
        Ltx25PositionGrid basePositions = Ltx25Geometry.VideoPositions(settings.LatentFrames, settings.LatentHeight, settings.LatentWidth, settings.Fps);
        basePositions.Boundaries.CopyTo(positions, 0);
        var seen = new HashSet<int>();
        int next = settings.VideoTokens;
        foreach (var image in images)
        {
            if (image == null || image.FrameIndex < 0 || image.FrameIndex >= settings.Frames || !seen.Add(image.FrameIndex) ||
                !float.IsFinite(image.Strength) || image.Strength < 0 || image.Strength > 1 ||
                image.Latent == null || image.Latent.Length != checked(spatial * 128) || image.Latent.Any(value => !float.IsFinite(value)))
                throw new ArgumentException("Invalid encoded LTX image guide.");
            float[] packed = Ltx25Geometry.PackVideo(image.Latent, spatial, 128);
            int start = image.FrameIndex == 0 ? 0 : next;
            packed.CopyTo(clean, start * 128);
            for (int token = start; token < start + spatial; ++token)
            {
                float denoise = 1 - image.Strength;
                mask[token] = denoise;
                for (int c = 0; c < 128; ++c)
                {
                    int offset = token * 128 + c;
                    // VideoConditionByLatentIndex preserves the prior-stage latent;
                    // VideoConditionByKeyframeIndex appends zero placeholders. The
                    // official GaussianNoiser first noises those latents, then applies
                    // the clean/mask blend. Replacing the prior with clean here gives
                    // fractional-strength guides the wrong refinement starting point.
                    float noised = image.FrameIndex == 0 || rawNoise == null
                        ? video[offset] : rawNoise[offset] * noiseScale;
                    video[offset] = denoise == 0 ? clean[offset] : denoise == 1 ? noised
                        : clean[offset] + (noised - clean[offset]) * denoise;
                }
                if (image.FrameIndex != 0)
                {
                    int local = token - start, p = token * 6;
                    positions[p] = image.FrameIndex / (float)settings.Fps;
                    positions[p + 1] = (image.FrameIndex + 1) / (float)settings.Fps;
                    positions[p + 2] = local / settings.LatentWidth * 32;
                    positions[p + 3] = positions[p + 2] + 32;
                    positions[p + 4] = local % settings.LatentWidth * 32;
                    positions[p + 5] = positions[p + 4] + 32;
                }
            }
            if (image.FrameIndex != 0) next += spatial;
        }
        return new()
        {
            Video = video, VideoTokens = tokens, Audio = initial.Audio, AudioTokens = initial.AudioTokens,
            VideoContext = initial.VideoContext, VideoContextTokens = initial.VideoContextTokens,
            AudioContext = initial.AudioContext, AudioContextTokens = initial.AudioContextTokens,
            VideoPositions = new(tokens, 3, positions), AudioPositions = initial.AudioPositions,
            VideoSigma = initial.VideoSigma, AudioSigma = initial.AudioSigma, FrequencyPrecision = initial.FrequencyPrecision,
            VideoDenoiseMask = mask, CleanVideo = clean,
            KeyframeMask = keyframes,
        };
    }
}
