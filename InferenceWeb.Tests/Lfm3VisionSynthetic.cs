// Copyright (c) Zhongkai Fu. All rights reserved.
// https://github.com/zhongkaifu/TensorSharp
//
// This file is part of TensorSharp.
//
// TensorSharp is licensed under the BSD-3-Clause license found in the LICENSE file in the root directory of this source tree.
// Shared with eng/lfm3/Lfm3VisionProbe (linked there): one C# copy of the harness's synthetic pictures.
using System.Collections.Generic;
using System.Linq;
using TensorSharp.Models;
using TensorSharp.Models.Media;

namespace InferenceWeb.Tests;

/// <summary>The synthetic RGB test picture eng/lfm3/lfm3-vision-ref.cpp draws (synth_rgb): the same integer formula, so
/// llama.cpp and these tests start from identical bytes.</summary>
internal static class Lfm3VisionSynthetic
{
    public static Lfm3RgbImage Image(int w, int h, uint seed)
    {
        var rgb = new byte[(long)w * h * 3];
        unchecked
        {
            uint cx = (uint)w / 3, cy = (uint)(2 * h) / 5;
            for (uint y = 0; y < (uint)h; y++)
                for (uint x = 0; x < (uint)w; x++)
                {
                    uint dx = x > cx ? x - cx : cx - x, dy = y > cy ? y - cy : cy - y;
                    uint ring = ((dx * dx + dy * dy) / (uint)(w / 16 + 1)) & 63u;
                    for (uint c = 0; c < 3; c++)
                    {
                        uint k = x * 73856093u ^ y * 19349663u ^ (seed * 83492791u + c * 2654435761u);
                        k ^= k >> 13;
                        k *= 0x5bd1e995u;
                        k ^= k >> 15;
                        uint b = (x * (c + 1) * 160u) / (uint)w + (y * (3 - c) * 120u) / (uint)h;
                        uint block = ((x / 24u + y / 24u + c) & 1u) * 40u;
                        rgb[((long)y * w + x) * 3 + c] = (byte)((b + ring + block + (k >> 27)) & 255u);
                    }
                }
        }
        return new Lfm3RgbImage(w, h, rgb);
    }
}

/// <summary>A fake decoder serving synthetic frames (frame i: the synthetic picture with seed 1000 + i), as the
/// harness's synthvideo case serves them to llama.cpp.</summary>
internal sealed class Lfm3SyntheticVideoDecoder : IVideoDecoder
{
    private readonly VideoInfo _info;

    public Lfm3SyntheticVideoDecoder(int width, int height, int frames, double fps, int deliverable = int.MaxValue)
    {
        _info = new VideoInfo(fps, frames, width, height);
        Deliverable = deliverable;
    }

    /// <summary>Frames the clip really has (a container may report more, as OpenCV's counts from the duration do).</summary>
    public int Deliverable { get; }

    /// <summary>The frames delivered, in order.</summary>
    public List<int> Served { get; } = new();

    /// <summary>Every request's frame list.</summary>
    public List<int[]> Requests { get; } = new();

    public VideoInfo Probe(string path) => _info;

    public void ReadFrames(string path, IReadOnlyList<int> frameIndices, FrameCallback onFrame)
    {
        Requests.Add(frameIndices.ToArray());
        foreach (int i in frameIndices)
        {
            if (i >= Deliverable || i >= _info.FrameCount) return;
            Lfm3RgbImage f = Lfm3VisionSynthetic.Image(_info.Width, _info.Height, 1000u + (uint)i);
            // hand it over as BGRA with padded rows, the way AVFoundation does, to cover the conversion too
            int stride = _info.Width * 4 + 16;
            var bgra = new byte[stride * _info.Height];
            for (int y = 0; y < _info.Height; y++)
                for (int x = 0; x < _info.Width; x++)
                {
                    int s = (y * _info.Width + x) * 3, d = y * stride + x * 4;
                    bgra[d] = f.Rgb[s + 2];
                    bgra[d + 1] = f.Rgb[s + 1];
                    bgra[d + 2] = f.Rgb[s];
                    bgra[d + 3] = 77;
                }
            Served.Add(i);
            onFrame(i, bgra, _info.Width, _info.Height, stride, PixelLayout.Bgra);
        }
    }
}
