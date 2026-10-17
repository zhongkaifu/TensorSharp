// Copyright (c) Zhongkai Fu. Licensed under the BSD-3-Clause license.
using TensorSharp.Models.Media;
using TensorSharp.Studio;
namespace TensorSharp.Chat.Studio;
public static class StudioHostOptions
{
    public static StudioOptions Create(long maxImportBytes = 0)
    {
        var options = new StudioOptions();
        if (maxImportBytes > 0) options.MaxImportBytes = System.Math.Min(options.MaxImportBytes, maxImportBytes);
        if (!options.EnableExternalProcesses) return options;
        options.FfmpegPath = MediaExecutableLocator.Find("ffmpeg", "TS_FFMPEG") ?? "ffmpeg";
        options.FfprobePath = MediaExecutableLocator.Find("ffprobe", "TS_FFPROBE", options.FfmpegPath) ?? "ffprobe";
        return options;
    }
}
