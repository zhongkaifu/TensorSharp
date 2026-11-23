// Copyright (c) Zhongkai Fu. All rights reserved.
// https://github.com/zhongkaifu/TensorSharp
//
// This file is part of TensorSharp.
//
// TensorSharp is licensed under the BSD-3-Clause license found in the LICENSE file in the root directory of this source tree.
using System.Collections.Generic;
using System.Diagnostics;
using TensorSharp.Models.Media;
using TensorSharp.Models.Media.Desktop;

namespace InferenceWeb.Tests;

/// <summary>[Fact] that needs ffmpeg and ffprobe (TS_FFMPEG / TS_FFPROBE or PATH): skips visibly without them.</summary>
[AttributeUsage(AttributeTargets.Method)]
public sealed class FfmpegDecoderFactAttribute : FactAttribute
{
    public FfmpegDecoderFactAttribute()
        => Skip = FfmpegVideoDecoder.TryCreate() == null ? "Requires ffmpeg and ffprobe (TS_FFMPEG / TS_FFPROBE or PATH)." : null;
}

/// <summary>
/// The ffmpeg frame reader that serves video where OpenCvSharp's native library does not load: the probe
/// reports what OpenCV does (displayed size, frame rate, frame count), and the reader delivers exactly the requested frames
/// in order, a repeated index twice, and stops quietly past the end of the clip.
/// </summary>
public sealed class FfmpegVideoDecoderTests : IDisposable
{
    private readonly string _dir = Path.Combine(Path.GetTempPath(), "ts-ffmpeg-video-" + Guid.NewGuid().ToString("N"));

    public FfmpegVideoDecoderTests() => Directory.CreateDirectory(_dir);

    public void Dispose()
    {
        try { Directory.Delete(_dir, recursive: true); } catch (IOException) { }
    }

    /// <summary>Ten frames at 10 fps, 64 x 48: frames 0-4 red, 5-9 blue (intra-only, so every frame is exact).</summary>
    private string RedThenBlue(string sar = "1")
    {
        string clip = Path.Combine(_dir, $"red-blue-{sar.Replace('/', '_')}.mp4");
        var psi = new ProcessStartInfo(MediaExecutableLocator.Find("ffmpeg", "TS_FFMPEG"))
        {
            RedirectStandardError = true, RedirectStandardOutput = true, UseShellExecute = false,
        };
        foreach (string a in new[] { "-hide_banner", "-loglevel", "error",
                     "-f", "lavfi", "-i", "color=c=red:s=64x48:r=10:d=0.5", "-f", "lavfi", "-i", "color=c=blue:s=64x48:r=10:d=0.5",
                     "-filter_complex", $"[0][1]concat=n=2:v=1:a=0,setsar={sar}", "-c:v", "mpeg4", "-q:v", "2", "-g", "1", clip })
            psi.ArgumentList.Add(a);
        using var p = Process.Start(psi)!;
        string err = p.StandardError.ReadToEnd();
        p.WaitForExit();
        Assert.True(p.ExitCode == 0, err);
        return clip;
    }

    [FfmpegDecoderFact]
    public void ProbeReportsRateCountAndDisplayedSize()
    {
        var decoder = FfmpegVideoDecoder.TryCreate()!;
        VideoInfo info = decoder.Probe(RedThenBlue());
        Assert.Equal(10.0, info.Fps, 3);
        Assert.Equal(10, info.FrameCount);
        Assert.Equal(64, info.Width);
        Assert.Equal(48, info.Height);
        // Pixels twice as wide as tall are shown twice as wide, as OpenCV's FFmpeg backend reports them.
        VideoInfo wide = decoder.Probe(RedThenBlue("2/1"));
        Assert.Equal(128, wide.Width);
        Assert.Equal(48, wide.Height);
    }

    [FfmpegDecoderFact]
    public void ReadFramesDeliversTheRequestedFramesInOrder()
    {
        var decoder = FfmpegVideoDecoder.TryCreate()!;
        var seen = new List<(int Index, byte R, byte B, int Width, int Height, PixelLayout Layout)>();
        decoder.ReadFrames(RedThenBlue(), new[] { 0, 3, 3, 7, 12 }, (index, pixels, width, height, stride, layout) =>
        {
            int centre = (height / 2) * stride + (width / 2) * 3;
            seen.Add((index, pixels[centre], pixels[centre + 2], width, height, layout));
        });
        // Frame 12 is past the clip: a short read, not an error.
        Assert.Equal(new[] { 0, 3, 3, 7 }, seen.ConvertAll(s => s.Index));
        foreach (var s in seen)
        {
            Assert.Equal((64, 48, PixelLayout.Rgb), (s.Width, s.Height, s.Layout));
            bool red = s.Index < 5;
            Assert.True(red ? s.R > 200 && s.B < 60 : s.B > 200 && s.R < 60, $"frame {s.Index}: R={s.R} B={s.B}");
        }
    }

    [FfmpegDecoderFact]
    public void ReadFramesRefusesDecreasingIndices()
    {
        var decoder = FfmpegVideoDecoder.TryCreate()!;
        Assert.Throws<ArgumentException>(() => decoder.ReadFrames(RedThenBlue(), new[] { 4, 2 }, (_, _, _, _, _, _) => { }));
    }
}
