// Copyright (c) Zhongkai Fu. All rights reserved.
// Licensed under the BSD-3-Clause license in the repository root.
using System;
using TensorSharp.Models.SharedDiffusion;

namespace TensorSharp.Models.MiniMaxH3;

/// <summary>Philox noise and layout conversion matching stable-diffusion.cpp's
/// MiniMax-H3 packed latent with <c>--rng cuda</c>.</summary>
internal static class MiniMaxH3Noise
{
    internal static (float[] Video, float[] Audio) Initial(MiniMaxH3Shape shape, long seed)
    {
        ArgumentNullException.ThrowIfNull(shape);
        ValidateVideo(shape.LatentFrames, shape.LatentHeight, shape.LatentWidth);
        if (shape.AudioLatentFrames <= 0) throw new ArgumentOutOfRangeException(nameof(shape));
        int volume = checked(shape.LatentFrames * shape.LatentHeight * shape.LatentWidth);
        int videoLength = checked(volume * MiniMaxH3Geometry.VideoLatentChannels);
        int audioLength = checked(shape.AudioLatentFrames * MiniMaxH3Geometry.AudioChannels * MiniMaxH3Geometry.AudioLatentChannels);
        int extraChannels = checked((audioLength + volume - 1) / volume);
        // The reference draws once over the complete padded packed tensor. Keep
        // that counter geometry rather than drawing separate video/audio streams.
        float[] packed = PhiloxNoise.Draw(checked(volume * (MiniMaxH3Geometry.VideoLatentChannels + extraChannels)), seed);
        float[] thwc = ToThwc(packed, shape.LatentFrames, shape.LatentHeight, shape.LatentWidth);
        float[] video = MiniMaxH3DiT.PatchifyVideo(thwc, shape.LatentFrames, shape.LatentHeight,
            shape.LatentWidth, MiniMaxH3Geometry.VideoLatentChannels);
        var audio = new float[audioLength];
        for (int channel = 0; channel < MiniMaxH3Geometry.AudioChannels; ++channel)
            for (int time = 0; time < shape.AudioLatentFrames; ++time)
                for (int dimension = 0; dimension < MiniMaxH3Geometry.AudioLatentChannels; ++dimension)
                    audio[(channel * shape.AudioLatentFrames + time) * MiniMaxH3Geometry.AudioLatentChannels + dimension] =
                        packed[videoLength + (dimension * MiniMaxH3Geometry.AudioChannels + channel) * shape.AudioLatentFrames + time];
        return (video, audio);
    }

    internal static void AddVisual(float[] latent, long seed, int frames, int height, int width)
    {
        ArgumentNullException.ThrowIfNull(latent);
        ValidateVideo(frames, height, width);
        int length = checked(frames * height * width * MiniMaxH3Geometry.VideoLatentChannels);
        if (latent.Length != length) throw new ArgumentException("H3 visual conditioning must have THWC latent geometry.", nameof(latent));
        // Every reference is reseeded, independently of the initial packed draw.
        float[] noise = ToThwc(PhiloxNoise.Draw(length, seed), frames, height, width);
        const float timestep = MiniMaxH3Scheduler.VisualConditionTimestep;
        for (int i = 0; i < length; ++i)
            latent[i] = latent[i] * timestep + noise[i] * (1 - timestep);
    }

    private static float[] ToThwc(float[] chw, int frames, int height, int width)
    {
        int volume = checked(frames * height * width);
        var result = new float[checked(volume * MiniMaxH3Geometry.VideoLatentChannels)];
        for (int spatial = 0; spatial < volume; ++spatial)
            for (int channel = 0; channel < MiniMaxH3Geometry.VideoLatentChannels; ++channel)
                result[spatial * MiniMaxH3Geometry.VideoLatentChannels + channel] = chw[channel * volume + spatial];
        return result;
    }

    private static void ValidateVideo(int frames, int height, int width)
    {
        if (frames <= 0 || height <= 0 || width <= 0 ||
            height % MiniMaxH3Geometry.PatchH != 0 || width % MiniMaxH3Geometry.PatchW != 0)
            throw new ArgumentException("H3 noise requires positive dimensions aligned to its video patch.");
    }
}
