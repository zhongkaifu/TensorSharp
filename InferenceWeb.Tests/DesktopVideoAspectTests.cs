// Copyright (c) Zhongkai Fu. All rights reserved.
// https://github.com/zhongkaifu/TensorSharp
//
// This file is part of TensorSharp.
//
// TensorSharp is licensed under the BSD-3-Clause license found in the LICENSE file in the root directory of this source tree.
using System.Diagnostics;
using TensorSharp.Models;
using TensorSharp.Models.Media;
using TensorSharp.Models.Media.Desktop;

namespace InferenceWeb.Tests;

/// <summary>[Fact] that writes its clip with ffmpeg (TS_FFMPEG or PATH) and reads it with OpenCV: skips visibly without either.</summary>
[AttributeUsage(AttributeTargets.Method)]
public sealed class FfmpegVideoFactAttribute : FactAttribute
{
    public FfmpegVideoFactAttribute()
        => Skip = TestGates.VideoSkip
                  ?? (MediaExecutableLocator.Find("ffmpeg", "TS_FFMPEG") == null ? "Requires ffmpeg (TS_FFMPEG or PATH) to write the clip." : null);
}

/// <summary>
/// An anamorphic clip (stored pixels not square) is probed at the size it is shown at, on every OpenCV backend: the FFmpeg
/// backend (Linux, Windows) decodes the encoded size and had reported it, so LFM3-Omni planned Linux frames 720 wide where
/// macOS (AVFoundation) and llama.cpp's mtmd take 854 for test-3.mp4 (pixel aspect 427:360) - 549 prompt tokens apart.
/// </summary>
public sealed class DesktopVideoAspectTests : IDisposable
{
    private readonly string _dir = Path.Combine(Path.GetTempPath(), "ts-video-aspect-" + Guid.NewGuid().ToString("N"));

    public DesktopVideoAspectTests() => Directory.CreateDirectory(_dir);

    public void Dispose()
    {
        try { Directory.Delete(_dir, recursive: true); } catch (IOException) { }
    }

    [FfmpegVideoFact]
    public void AnAnamorphicClipIsProbedAtItsDisplayedSize()
    {
        // 320 x 256 stored, pixels twice as wide as tall: shown 640 x 256.
        string clip = Path.Combine(_dir, "anamorphic.mp4");
        var psi = new ProcessStartInfo(MediaExecutableLocator.Find("ffmpeg", "TS_FFMPEG"))
        {
            RedirectStandardError = true, RedirectStandardOutput = true, UseShellExecute = false,
        };
        foreach (string a in new[] { "-hide_banner", "-loglevel", "error", "-f", "lavfi", "-i", "testsrc=size=320x256:rate=10:duration=2",
                     "-vf", "setsar=2/1", "-c:v", "mpeg4", "-q:v", "4", clip })
            psi.ArgumentList.Add(a);
        using (var p = Process.Start(psi)!)
        {
            string err = p.StandardError.ReadToEnd();
            p.WaitForExit();
            Assert.True(p.ExitCode == 0, err);
        }

        var provider = new DesktopMediaProvider();
        VideoInfo info = provider.Probe(clip);
        Assert.Equal(640, info.Width);
        Assert.Equal(256, info.Height);

        // LFM3-Omni plans the frames from the displayed size, and the decoded frames are resized to the plan in one step.
        Lfm3VideoPlan plan = Lfm3Video.Plan(clip, decoder: provider);
        Assert.Equal(640, plan.SourceWidth);
        Assert.True(plan.Width > plan.Height * 2, $"planned {plan.Width} x {plan.Height} for a 640 x 256 picture");
    }
}
