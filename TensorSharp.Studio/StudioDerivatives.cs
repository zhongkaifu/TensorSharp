// Copyright (c) Zhongkai Fu. Licensed under the BSD-3-Clause license.
using System.Globalization;

namespace TensorSharp.Studio;

public sealed record StudioWaveform(string AssetId, double DurationSeconds, IReadOnlyList<float> Peaks);

public sealed partial class StudioService
{
    // A single decoder at a time bounds auxiliary memory and CPU independently of UI refreshes.
    private readonly SemaphoreSlim derivativeGate = new(1, 1);
    private const string DerivativeFormats = "mov,matroska,avi,wav,mp3,flac,ogg,image2,png_pipe,jpeg_pipe,webp_pipe,gif,aiff";

    public async Task<string> GetThumbnailAsync(string projectId, string assetId, CancellationToken ct = default)
    {
        var project = await GetProjectAsync(projectId, ct).ConfigureAwait(false);
        var asset = project.Assets.SingleOrDefault(a => a.Id == assetId) ?? throw new StudioException("asset_not_found", "Asset does not belong to this project.");
        if (asset.Kind == "audio") throw new StudioException("unsupported_thumbnail", "Audio assets have a waveform instead of a thumbnail.");
        string source = GetAssetPath(projectId, assetId);
        string target = DerivativePath(projectId, assetId, ".jpg");
        if (File.Exists(target)) return target;
        if (!Options.EnableExternalProcesses) throw new StudioException("unsupported_platform", "Thumbnail generation requires a desktop media decoder on this host.");
        await derivativeGate.WaitAsync(ct).ConfigureAwait(false);
        try
        {
            if (File.Exists(target)) return target;
            var temporary = target + "." + StudioJson.NewId() + ".jpg";
            try
            {
                List<string> args = ["-v", "error", "-nostdin", "-y", "-protocol_whitelist", "file,pipe", "-format_whitelist", DerivativeFormats];
                if (asset.Kind == "video") args.AddRange(["-ss", Math.Min(.1, asset.DurationSeconds / 2).ToString("R", CultureInfo.InvariantCulture)]);
                args.AddRange(["-i", source, "-an", "-frames:v", "1", "-vf", "scale=320:180:force_original_aspect_ratio=decrease,pad=320:180:(ow-iw)/2:(oh-ih)/2,setsar=1", "-q:v", "4", temporary]);
                await StudioMedia.RunAsync(Options.FfmpegPath, args, null, ct, TimeSpan.FromSeconds(30)).ConfigureAwait(false);
                if (!File.Exists(temporary) || new FileInfo(temporary).Length is < 4 or > 1024 * 1024)
                    throw new StudioException("thumbnail_failed", "The media decoder did not produce a bounded thumbnail.");
                File.Move(temporary, target, true);
                return target;
            }
            finally { if (File.Exists(temporary)) File.Delete(temporary); }
        }
        finally { derivativeGate.Release(); }
    }

    public async Task<StudioWaveform> GetWaveformAsync(string projectId, string assetId, CancellationToken ct = default)
    {
        var project = await GetProjectAsync(projectId, ct).ConfigureAwait(false);
        var asset = project.Assets.SingleOrDefault(a => a.Id == assetId) ?? throw new StudioException("asset_not_found", "Asset does not belong to this project.");
        if (!asset.HasAudio) throw new StudioException("no_audio", "This asset has no audio stream.");
        string source = GetAssetPath(projectId, assetId);
        string target = DerivativePath(projectId, assetId, ".json");
        if (File.Exists(target)) return StudioStorage.Read<StudioWaveform>(target);
        if (!Options.EnableExternalProcesses) throw new StudioException("unsupported_platform", "Waveform generation requires a desktop media decoder on this host.");
        await derivativeGate.WaitAsync(ct).ConfigureAwait(false);
        try
        {
            if (File.Exists(target)) return StudioStorage.Read<StudioWaveform>(target);
            // FFmpeg decodes incrementally. Only 512 peak buckets are retained in .NET/the page,
            // never a full PCM soundtrack. Blocks are at most 100 ms, smaller for short clips.
            var peaks = new float[512];
            long block = 0;
            int blockSamples = (int)Math.Clamp(Math.Ceiling(asset.DurationSeconds * 48000 / peaks.Length), 64, 4800);
            await StudioMedia.RunAsync(Options.FfmpegPath,
                ["-v", "error", "-nostdin", "-protocol_whitelist", "file,pipe", "-format_whitelist", DerivativeFormats,
                 "-i", source, "-vn", "-af", $"aresample=48000,asetnsamples=n={blockSamples},astats=metadata=1:reset=1:measure_perchannel=none:measure_overall=Peak_level,ametadata=mode=print:key=lavfi.astats.Overall.Peak_level:file=-",
                 "-f", "null", "-"], null, ct, TimeSpan.FromMinutes(3), line =>
                {
                    const string prefix = "lavfi.astats.Overall.Peak_level=";
                    if (!line.StartsWith(prefix, StringComparison.Ordinal)) return;
                    int bucket = Math.Clamp((int)(block++ * blockSamples / 48000d / asset.DurationSeconds * peaks.Length), 0, peaks.Length - 1);
                    if (double.TryParse(line.AsSpan(prefix.Length), NumberStyles.Float, CultureInfo.InvariantCulture, out var decibels))
                    {
                        float value = double.IsNegativeInfinity(decibels) ? 0 : (float)Math.Clamp(Math.Pow(10, decibels / 20), 0, 1);
                        if (float.IsFinite(value)) peaks[bucket] = Math.Max(peaks[bucket], value);
                    }
                }).ConfigureAwait(false);
            if (block == 0) throw new StudioException("waveform_failed", "The decoder returned no waveform measurements.");
            var waveform = new StudioWaveform(asset.Id, asset.DurationSeconds, peaks);
            StudioStorage.Save(target, waveform);
            return waveform;
        }
        finally { derivativeGate.Release(); }
    }

    private string DerivativePath(string projectId, string assetId, string extension)
    {
        // The cache key includes managed content size and modification time to invalidate relinks.
        var source = GetAssetPath(projectId, assetId);
        var info = new FileInfo(source);
        var directory = StudioPaths.SafeChild(RootPath, "previews", StudioPaths.Id(projectId));
        Directory.CreateDirectory(directory);
        return StudioPaths.SafeChild(directory, StudioPaths.Id(assetId) + "-" + info.Length + "-" + info.LastWriteTimeUtc.Ticks + extension);
    }
}
