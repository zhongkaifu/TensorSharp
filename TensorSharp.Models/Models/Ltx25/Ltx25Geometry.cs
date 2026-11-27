// Copyright (c) Zhongkai Fu. All rights reserved.
// Licensed under the BSD-3-Clause license in the repository root.
using System;
using System.Threading;

namespace TensorSharp.Models.Ltx25;

public sealed record Ltx25PositionGrid(int Tokens, int Axes, float[] Boundaries);
public sealed record Ltx25Rope(int Tokens, int Heads, int HeadDim, float[] Cosine, float[] Sine);
public enum Ltx25FrequencyPrecision { Float32, Float64 }

/// <summary>Latent packing and causal coordinates from the original LTX
/// patchifiers. This describes denoiser inputs, not validated VAE outputs.</summary>
public static class Ltx25Geometry
{
    public const int SpatialScale = 32, TemporalScale = 8;
    public static (int Frames, int Height, int Width) VideoLatentShape(int frames, int height, int width)
    {
        if (frames < 1 || (frames - 1) % TemporalScale != 0 || height < SpatialScale || width < SpatialScale ||
            height % SpatialScale != 0 || width % SpatialScale != 0)
            throw new ArgumentException("LTX video geometry requires 8n+1 frames and positive 32-aligned spatial sizes.");
        return ((frames - 1) / TemporalScale + 1, height / SpatialScale, width / SpatialScale);
    }
    public static Ltx25PositionGrid VideoPositions(int frames, int height, int width, float fps)
    {
        if (frames <= 0 || height <= 0 || width <= 0 || !float.IsFinite(fps) || fps <= 0) throw new ArgumentOutOfRangeException(nameof(frames));
        _ = checked(frames * TemporalScale);
        _ = checked(height * SpatialScale);
        _ = checked(width * SpatialScale);
        int tokens = checked(frames * height * width);
        var positions = new float[checked(tokens * 6)];
        int token = 0;
        for (int t = 0; t < frames; ++t)
        for (int h = 0; h < height; ++h)
        for (int w = 0; w < width; ++w)
        {
            int offset = token++ * 6;
            positions[offset] = Math.Max(0, t * TemporalScale + 1 - TemporalScale) / fps;
            positions[offset + 1] = Math.Max(0, (t + 1) * TemporalScale + 1 - TemporalScale) / fps;
            positions[offset + 2] = checked(h * SpatialScale); positions[offset + 3] = checked((h + 1) * SpatialScale);
            positions[offset + 4] = checked(w * SpatialScale); positions[offset + 5] = checked((w + 1) * SpatialScale);
        }
        return new(tokens, 3, positions);
    }
    public static Ltx25PositionGrid AudioPositions(int frames)
    {
        if (frames <= 0) throw new ArgumentOutOfRangeException(nameof(frames));
        _ = checked(frames * 4);
        var positions = new float[checked(frames * 2)];
        for (int t = 0; t < frames; ++t)
        {
            positions[t * 2] = Math.Max(0, t * 4 + 1 - 4) * 160f / 16000f;
            positions[t * 2 + 1] = Math.Max(0, (t + 1) * 4 + 1 - 4) * 160f / 16000f;
        }
        return new(frames, 1, positions);
    }
    public static Ltx25PositionGrid TemporalPositions(Ltx25PositionGrid video)
    {
        Validate(video);
        if (video.Axes != 3) throw new ArgumentException("Expected three-axis video positions.");
        var values = new float[checked(video.Tokens * 2)];
        for (int i = 0; i < video.Tokens; ++i) Array.Copy(video.Boundaries, i * 6, values, i * 2, 2);
        return new(video.Tokens, 1, values);
    }
    public static Ltx25Rope BuildRope(Ltx25PositionGrid positions, int innerDim, int heads,
        Ltx25FrequencyPrecision precision = Ltx25FrequencyPrecision.Float64, bool middle = true, float theta = 10000f)
    {
        Validate(positions);
        if (heads <= 0 || innerDim <= 0 || innerDim % heads != 0 || innerDim / heads % 2 != 0 ||
            !float.IsFinite(theta) || theta <= 1 || !Enum.IsDefined(precision)) throw new ArgumentException("Invalid LTX rotary geometry.");
        int half = innerDim / 2, count = innerDim / (2 * positions.Axes), pad = half - count * positions.Axes;
        if (count < 1) throw new ArgumentException("Rotary width is smaller than its position-axis count.");
        var frequencies = new float[count];
        for (int i = 0; i < count; ++i)
        {
            // torch.linspace computes the latter half backwards from its end
            // point. At large theta a one-ULP ratio difference becomes a
            // visible phase error, so preserve that Float32 convention.
            float step = count == 1 ? 0 : 1f / (count - 1);
            float ratio = count == 1 ? 0 : i < count / 2 ? i * step : MathF.FusedMultiplyAdd(-(count - i - 1), step, 1f);
            frequencies[i] = precision == Ltx25FrequencyPrecision.Float64
                ? (float)(Math.Pow(theta, count == 1 ? 0 : (double)i / (count - 1)) * (Math.PI / 2))
                : MathF.Pow(theta, ratio) * (MathF.PI / 2);
        }
        var cos = new float[checked(positions.Tokens * half)];
        var sin = new float[cos.Length];
        int nonfinite = 0;
        // Every token's row depends only on its own coordinates, so blocks of tokens run on several cores with the same bits.
        void Rows(int begin, int end)
        {
            Span<float> coordinates = stackalloc float[3];
            for (int token = begin; token < end; ++token)
            {
                for (int axis = 0; axis < positions.Axes; ++axis)
                {
                    int coordinateOffset = (token * positions.Axes + axis) * 2;
                    float coordinate = middle ? (positions.Boundaries[coordinateOffset] + positions.Boundaries[coordinateOffset + 1]) * .5f
                        : positions.Boundaries[coordinateOffset];
                    float maximum = axis == 0 ? 20f : 2048f;
                    coordinates[axis] = coordinate / maximum * 2f - 1f;
                }
                for (int d = 0; d < pad; ++d) cos[token * half + d] = 1;
                for (int index = 0; index < count; ++index)
                for (int axis = 0; axis < positions.Axes; ++axis)
                {
                    float angle = frequencies[index] * coordinates[axis];
                    if (!float.IsFinite(angle)) { Volatile.Write(ref nonfinite, 1); return; }
                    int offset = token * half + pad + index * positions.Axes + axis;
                    (sin[offset], cos[offset]) = MathF.SinCos(angle);
                }
            }
        }
        const int TokensPerBlock = 64;
        if ((long)positions.Tokens * half <= 1 << 16) Rows(0, positions.Tokens);
        else
            System.Threading.Tasks.Parallel.For(0, (positions.Tokens + TokensPerBlock - 1) / TokensPerBlock,
                block => Rows(block * TokensPerBlock, Math.Min(positions.Tokens, (block + 1) * TokensPerBlock)));
        if (nonfinite != 0) throw new ArgumentException("LTX rotary phase exceeds Float32 range.");
        return new(positions.Tokens, heads, innerDim / heads, cos, sin);
    }
    private static void Validate(Ltx25PositionGrid positions)
    {
        if (positions == null || positions.Tokens <= 0 || positions.Axes is not (1 or 3) || positions.Boundaries == null ||
            positions.Boundaries.Length != checked(positions.Tokens * positions.Axes * 2)) throw new ArgumentException("Invalid LTX position grid.");
        for (int i = 0; i < positions.Boundaries.Length; i += 2)
            if (!float.IsFinite(positions.Boundaries[i]) || !float.IsFinite(positions.Boundaries[i + 1]) || positions.Boundaries[i + 1] < positions.Boundaries[i])
                throw new ArgumentException("LTX position boundaries must be finite and ordered.");
    }
    public static float[] PackVideo(ReadOnlySpan<float> channelMajor, int tokens, int channels)
    {
        if (tokens <= 0 || channels <= 0 || channelMajor.Length != checked(tokens * channels)) throw new ArgumentException("Invalid video latent storage.");
        var output = new float[channelMajor.Length];
        for (int token = 0; token < tokens; ++token)
        for (int channel = 0; channel < channels; ++channel) output[token * channels + channel] = channelMajor[channel * tokens + token];
        return output;
    }
    public static float[] UnpackVideo(ReadOnlySpan<float> tokenMajor, int tokens, int channels)
    {
        if (tokens <= 0 || channels <= 0 || tokenMajor.Length != checked(tokens * channels)) throw new ArgumentException("Invalid video token storage.");
        var output = new float[tokenMajor.Length];
        for (int token = 0; token < tokens; ++token)
        for (int channel = 0; channel < channels; ++channel) output[channel * tokens + token] = tokenMajor[token * channels + channel];
        return output;
    }
    public static float[] PackAudio(ReadOnlySpan<float> ctf, int time, int channels = 8, int bins = 16)
    {
        if (time <= 0 || channels <= 0 || bins <= 0 || ctf.Length != checked(time * channels * bins)) throw new ArgumentException("Invalid audio latent storage.");
        var output = new float[ctf.Length];
        for (int t = 0; t < time; ++t)
        for (int c = 0; c < channels; ++c)
        for (int f = 0; f < bins; ++f) output[(t * channels + c) * bins + f] = ctf[(c * time + t) * bins + f];
        return output;
    }
}
