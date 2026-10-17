// Copyright (c) Zhongkai Fu. Licensed under the BSD-3-Clause license.
using System;
using System.Collections.Concurrent;
using System.Diagnostics;
using System.Globalization;
using System.IO;
using System.Threading;

namespace TensorSharp.Models.Media.Desktop;

public sealed partial class DesktopMediaProvider
{
    private static readonly ConcurrentDictionary<(string Path, long Modified), Lazy<bool>> H264ConditioningCapabilities = new();

    public bool SupportsH264ImageConditioning => HasH264ConditioningEncoder(FindFfmpeg());

    // Cache the bounded discovery probe: model capability/admission queries must not
    // launch a process on every UI refresh. A replaced executable gets a new key.
    internal static bool HasH264ConditioningEncoder(string executable)
    {
        if (string.IsNullOrWhiteSpace(executable)) return false;
        long modified;
        try { modified = File.GetLastWriteTimeUtc(executable).Ticks; }
        catch (Exception ex) when (ex is IOException or UnauthorizedAccessException or ArgumentException) { return false; }
        return H264ConditioningCapabilities.GetOrAdd((executable, modified),
            key => new Lazy<bool>(() => ProbeH264ConditioningEncoder(key.Path))).Value;
    }

    private static bool ProbeH264ConditioningEncoder(string executable)
    {
        using var process = new Process
        {
            StartInfo = new ProcessStartInfo(executable)
            { UseShellExecute = false, RedirectStandardOutput = true, RedirectStandardError = true, CreateNoWindow = true },
        };
        process.StartInfo.ArgumentList.Add("-hide_banner");
        process.StartInfo.ArgumentList.Add("-encoders");
        try
        {
            process.Start();
            using var deadline = new CancellationTokenSource(TimeSpan.FromSeconds(5));
            var output = process.StandardOutput.ReadToEndAsync(deadline.Token);
            var errors = process.StandardError.ReadToEndAsync(deadline.Token);
            string encoders;
            try
            {
                process.WaitForExitAsync(deadline.Token).GetAwaiter().GetResult();
                errors.GetAwaiter().GetResult();
                encoders = output.GetAwaiter().GetResult();
            }
            catch (OperationCanceledException)
            {
                try { process.Kill(entireProcessTree: true); } catch (InvalidOperationException) { }
                process.WaitForExit();
                return false;
            }
            if (process.ExitCode != 0) return false;
            foreach (string line in encoders.Split('\n'))
            {
                string[] fields = line.Split((char[])null, StringSplitOptions.RemoveEmptyEntries);
                if (fields.Length > 1 && fields[0].StartsWith('V') && fields[1] == "libx264") return true;
            }
            return false;
        }
        catch (Exception ex) when (ex is System.ComponentModel.Win32Exception or InvalidOperationException or IOException) { return false; }
    }

    public byte[] RecompressH264Image(byte[] rgb8, int width, int height, int crf,
        CancellationToken cancellationToken = default)
    {
        ArgumentNullException.ThrowIfNull(rgb8);
        if (width < 2 || height < 2 || (width & 1) != 0 || (height & 1) != 0 || rgb8.LongLength != (long)width * height * 3)
            throw new ArgumentException("H.264 image conditioning requires packed RGB8 with positive even dimensions.", nameof(rgb8));
        if (crf < 0 || crf > 51) throw new ArgumentOutOfRangeException(nameof(crf));
        cancellationToken.ThrowIfCancellationRequested();
        string ffmpeg = FindFfmpeg() ?? throw new NotSupportedException(
            "LTX image conditioning requires FFmpeg with libx264. Install FFmpeg on PATH or set TS_FFMPEG.");
        string directory = Path.Combine(Path.GetTempPath(), $"ts-image-conditioning-{Guid.NewGuid():N}");
        Directory.CreateDirectory(directory);
        try
        {
            string source = Path.Combine(directory, "source.rgb");
            string encoded = Path.Combine(directory, "frame.mp4");
            string decoded = Path.Combine(directory, "decoded.rgb");
            File.WriteAllBytes(source, rgb8);
            RunImageConditioningFfmpeg(ffmpeg, ["-y", "-nostdin", "-loglevel", "error", "-f", "rawvideo",
                "-pixel_format", "rgb24", "-video_size", $"{width}x{height}", "-framerate", "1", "-i", source,
                "-sws_flags", "bilinear", "-frames:v", "1", "-c:v", "libx264", "-preset", "veryfast",
                "-crf", crf.ToString(CultureInfo.InvariantCulture), "-pix_fmt", "yuv420p", "-thread_type", "slice", encoded], cancellationToken);
            RunImageConditioningFfmpeg(ffmpeg, ["-y", "-nostdin", "-loglevel", "error", "-i", encoded,
                "-sws_flags", "bilinear", "-frames:v", "1", "-f", "rawvideo", "-pix_fmt", "rgb24", decoded], cancellationToken);
            byte[] result = File.ReadAllBytes(decoded);
            if (result.Length != rgb8.Length) throw new InvalidDataException("FFmpeg returned an unexpected image-conditioning frame size.");
            return result;
        }
        finally
        {
            try { Directory.Delete(directory, recursive: true); } catch (IOException) { } catch (UnauthorizedAccessException) { }
        }
    }

    private static void RunImageConditioningFfmpeg(string executable, string[] arguments, CancellationToken cancellationToken)
    {
        var start = new ProcessStartInfo(executable) { UseShellExecute = false, RedirectStandardError = true, CreateNoWindow = true };
        foreach (string argument in arguments) start.ArgumentList.Add(argument);
        using var process = new Process { StartInfo = start };
        try { process.Start(); }
        catch (Exception ex) when (ex is System.ComponentModel.Win32Exception or InvalidOperationException)
        {
            throw new InvalidOperationException("LTX image conditioning could not start FFmpeg. Check PATH or TS_FFMPEG and install a build with libx264.", ex);
        }
        using var deadline = CancellationTokenSource.CreateLinkedTokenSource(cancellationToken);
        deadline.CancelAfter(TimeSpan.FromSeconds(60));
        var errors = process.StandardError.ReadToEndAsync(deadline.Token);
        string error;
        try
        {
            process.WaitForExitAsync(deadline.Token).GetAwaiter().GetResult();
            error = errors.GetAwaiter().GetResult();
        }
        catch (OperationCanceledException)
        {
            try { process.Kill(entireProcessTree: true); } catch (InvalidOperationException) { }
            process.WaitForExit();
            cancellationToken.ThrowIfCancellationRequested();
            throw new TimeoutException("FFmpeg exceeded the 60-second LTX image-conditioning deadline.");
        }
        if (process.ExitCode != 0)
            throw new InvalidOperationException($"LTX image conditioning requires FFmpeg with libx264; FFmpeg exited {process.ExitCode}: {error.Trim()}");
    }
}
