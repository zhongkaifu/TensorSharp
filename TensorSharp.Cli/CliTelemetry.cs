// Copyright (c) Zhongkai Fu. All rights reserved.
// Licensed under the BSD-3-Clause license in the repository root.
using System;
using System.Diagnostics;
using System.IO;
using TensorSharp.Models.Video;
using TensorSharp.Runtime.Telemetry;

namespace TensorSharp.Cli;

/// <summary>What the CLI's spans and metrics say about the model a run uses.</summary>
internal static class CliTelemetry
{
    /// <summary>
    /// <c>tensorsharp.backend</c> for <paramref name="model"/>, spelled the way <c>--backend</c>
    /// takes it and the server reports it; null for a model with no execution plan (a test fake).
    /// </summary>
    internal static string BackendOf(object model) =>
        model is IModelArchitecture { ExecutionPlan: { } plan }
            ? TensorSharp.Server.BackendCatalog.ToBackendValue(plan.BackendType)
            : null;
}

/// <summary>
/// The <c>video.export</c> child of a CLI clip: writing its frames to <c>--output</c> (an MP4, or
/// PNG files for a <c>.png</c> path, which never goes through <c>VideoIO.SaveMp4</c>) and the WAV
/// sidecar. It ends when it is disposed; <see cref="Saved"/> before that says the frames were
/// written. A failed save is not caught in the CLI (it propagates to <c>RunToExit</c>), so a span
/// disposed without <see cref="Saved"/> failed, and says so with <c>error.type=_OTHER</c>, the
/// semantic conventions' value for an error whose type is not known where it is recorded.
/// </summary>
internal sealed class CliVideoExport : IDisposable
{
    // The conventions' error.type for an error of unknown type.
    private const string UnknownErrorType = "_OTHER";

    private readonly Activity _span;
    private readonly string _outputPath;
    private readonly int _frames;
    private bool _saved;

    private CliVideoExport(Activity span, string outputPath, int frames)
    {
        _span = span;
        _outputPath = outputPath;
        _frames = frames;
    }

    /// <summary>Start the export of <paramref name="frames"/> frames to <paramref name="outputPath"/>, now.</summary>
    internal static CliVideoExport Start(VideoClipTelemetry clip, string outputPath, int frames) =>
        new(clip?.StartChild("video.export"), outputPath, frames);

    /// <summary>The frames were written with <paramref name="codec"/> (<c>png</c>, or what SaveMp4 used).</summary>
    internal void Saved(string codec)
    {
        _saved = true;
        if (_span is not { IsAllDataRequested: true })
            return;
        if (!string.IsNullOrEmpty(codec))
            _span.SetTag(TelemetryAttributes.VideoCodec, codec);
        if (OutputBytes() is long bytes)
            _span.SetTag(TelemetryAttributes.VideoBytes, bytes);
    }

    public void Dispose()
    {
        if (_span is null)
            return;
        if (!_saved)
        {
            _span.SetStatus(ActivityStatusCode.Error);
            if (_span.IsAllDataRequested)
            {
                _span.SetTag(TelemetryAttributes.Outcome, TelemetryAttributes.OutcomeFailed);
                _span.SetTag(TelemetryAttributes.ErrorType, UnknownErrorType);
            }
        }
        _span.Dispose();
    }

    /// <summary>
    /// Bytes written for the frames: the MP4, or every PNG under the names
    /// <c>Program.RunVideoGeneration</c> gives them (<c>out.png</c>, <c>out_00001.png</c>, ...).
    /// Null when a file cannot be read back.
    /// </summary>
    private long? OutputBytes()
    {
        try
        {
            long total = new FileInfo(_outputPath).Length;
            if (_outputPath.EndsWith(".png", StringComparison.OrdinalIgnoreCase))
            {
                string stem = Path.ChangeExtension(_outputPath, null);
                for (int i = 1; i < _frames; i++)
                    total += new FileInfo(stem + $"_{i:D5}.png").Length;
            }
            return total;
        }
        catch (Exception ex) when (ex is IOException or UnauthorizedAccessException or ArgumentException or NotSupportedException)
        {
            return null;
        }
    }
}
