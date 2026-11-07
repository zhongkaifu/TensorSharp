// Copyright (c) Zhongkai Fu. All rights reserved.
// Licensed under the BSD-3-Clause license in the repository root.
using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Threading;
using System.Threading.Tasks;
using Microsoft.Extensions.Logging;
using TensorSharp.Runtime.Telemetry;
using TensorSharp.Server.Hosting;
using TensorSharp.Server.Jev;

namespace TensorSharp.Server;

public partial class ModelService
{
    private readonly JevExecutionGate _jevExecution = new(ReadJevLimit("TS_JEV_MAX_PENDING", 32, 1, 1024));
    private UploadStoragePolicy _jevMediaStorage;
    private JevAudioTranscriber? _jevAudioTranscriber;

    /// <summary>Optional speech-to-text companion for Jev audio attachments. The callback receives
    /// an upload-confined audio path and must return the full transcript or fail; it must not
    /// silently truncate. When unset, TS_JEV_TRANSCRIPTION_URL configures an HTTP companion.</summary>
    public Func<string, CancellationToken, Task<string>>? JevAudioTranscriber { get; set; }

    /// <summary>
    /// Where Jev attachments are materialized before extraction and inference. The server
    /// host assigns the upload policy it already governs, so an operator's <c>--upload-max-mb</c>,
    /// quota and TTL cover these files exactly as they cover chat attachments.
    ///
    /// <para>Left unset (in-process callers), a process-local directory under the system temp path
    /// is used and named once in the log, rather than silently choosing a location.</para>
    /// </summary>
    public UploadStoragePolicy MediaStorage { get; set; }

    /// <summary>Evaluate Jev noul, choice and score questions using one denoise read per sample.</summary>
    public Task<object> JevAsync(JevRequest request, CancellationToken cancellationToken = default)
    {
        ArgumentNullException.ThrowIfNull(request);
        return TraceJevAsync(requestTelemetry => _jevExecution.ExecuteAwaitedAsync(async ct =>
        {
            if (_lifecycle.Model is not DiffusionGemmaModel model)
                throw new JevModelUnavailableException("Jev inference requires a loaded DiffusionGemma model.");
            if (request.Model != null && request.Model is not ("jev-latest" or "jev-preview") &&
                !string.Equals(request.Model, LoadedModelName, StringComparison.OrdinalIgnoreCase) &&
                !string.Equals(request.Model, Path.GetFileNameWithoutExtension(LoadedModelName), StringComparison.OrdinalIgnoreCase))
                throw new JevModelNotFoundException("model must name the loaded model, jev-latest, or jev-preview");
            if (request.Images.Length != 0 && model.VisionEncoder == null)
                throw new JevModelUnavailableException(
                    "Image input requires the DiffusionGemma vision tower. Start the server with the " +
                    "vision shard declared by config/jev-diffusiongemma-q4.json, or send a text-only state.");

            // Extraction and ASR do not hold the GPU lock. Admission remains held so model
            // reload/disposal cannot invalidate the model while an attachment is being prepared.
            JevPreparedAttachments? prepared = null;
            if (request.Images.Length != 0 || request.Attachments.Length != 0)
            {
                Func<string, CancellationToken, Task<string>>? transcribe = JevAudioTranscriber;
                if (transcribe == null && request.Attachments.Any(a => a.Kind == "audio"))
                {
                    _jevAudioTranscriber ??= Jev.JevAudioTranscriber.FromEnvironment();
                    if (_jevAudioTranscriber != null) transcribe = _jevAudioTranscriber.TranscribeAsync;
                }
                prepared = await JevAttachmentPreparer.PrepareAsync(request, ResolveMediaStorage(), transcribe, ct).ConfigureAwait(false);
            }
            string[] imagePaths = prepared?.ImagePaths ?? Array.Empty<string>();
            if (imagePaths.Length != 0 && model.VisionEncoder == null)
                throw new JevModelUnavailableException("These attachments require the DiffusionGemma vision tower; load the configured vision shard.");
            JevRequest inferenceRequest = prepared == null ? request : request with { State = prepared.State };

            // Native inference and Monitor ownership stay on one worker thread.
            return await Task.Run(() => RunPrepared(), ct).ConfigureAwait(false);

            object RunPrepared()
            {
                bool entered = false;
                DiffusionComputeTurns turns = model.ComputeTurns;
                var vision = new JevVisionBinder(new JevDiffusionVisionTarget(model), imagePaths);
                try
                {
                    // Structured reads cannot race the chat scheduler's shared native buffers, so
                    // they take the same GpuComputeLock. Taking it as a registered turn makes the
                    // scheduler hand it over before its next forward instead of after its block.
                    turns.Enter(ct);
                    entered = true;
                    ct.ThrowIfCancellationRequested();
                    var renderer = new KVCachePromptRenderer(new GgufPromptRenderer());
                    int[] Render(string system, string state)
                    {
                        // The images ride on the user turn, so the protocol's renderer puts one
                        // <|image> marker per image ahead of the state text; the binder then expands
                        // each marker into its soft-token span for THIS chunk prompt.
                        var history = new List<ChatMessage>
                        {
                            new() { Role = "system", Content = system },
                            new() { Role = "user", Content = state, ImagePaths = vision.ImagePathsOrNull() },
                        };
                        int[] tokens = renderer.RenderToTokens(model.Tokenizer, model.Config.ChatTemplate, history,
                            model.Config.Architecture, addGenerationPrompt: true, enableThinking: false).ToArray();
                        return vision.Expand(history, tokens);
                    }
                    int eos = model.Tokenizer.LookupToken("<turn|>");
                    if (eos < 0) throw new JevModelUnavailableException("DiffusionGemma tokenizer is missing the <turn|> token.");
                    int pad = model.Tokenizer.LookupToken("<pad>");
                    if (pad < 0) pad = model.MaskTokenId;
                    return JevInference.Run(inferenceRequest, LoadedModelName, text => model.Tokenizer.Encode(text, false).ToArray(),
                        Render, vision.Read, Math.Min(model.CanvasLength, ReadJevLimit("TS_JEV_MAX_CANVAS", 64, 8, 4096)),
                        model.MaxContextLength, eos, pad, model.Tokenizer.VocabSize, ct,
                        imagePaths.Length, prepared?.Diagnostics, prepared?.PreprocessingMs ?? 0,
                        // The same counts the response's usage reports. A read has no finish
                        // reason and no first token: it scores labels, it does not stream.
                        (inputTokens, outputTokens) => requestTelemetry.RecordUsage(
                            inputTokens, outputTokens, 0, finishReason: null, firstTokenAtUtc: null));
                }
                finally
                {
                    // Disposal frees model-global image spans, so it belongs inside the lock: a span
                    // left installed would be spliced into whatever prompt runs next.
                    vision.Dispose();
                    if (entered) turns.Exit();
                }
            }
        }, cancellationToken));
    }

    // error.type of a Jev request refused for a reason the caller can act on (a 4xx or a retry).
    private const string JevRefusedInvalidRequest = "invalid_request";
    private const string JevRefusedModelNotFound = "model_not_found";
    private const string JevRefusedModelUnavailable = "model_unavailable";
    private const string JevRefusedQueueFull = "queue_full";

    /// <summary>
    /// The <c>chat</c> span and request metrics of one Jev request, which drives DiffusionGemma
    /// itself rather than through ChatGenerationPipeline and so has no span of its own otherwise.
    /// Started before the execution gate, so time queued behind other reads counts in the
    /// request's duration; and in an async method of its own, so the span it makes current
    /// does not stay current in the caller once this returns.
    /// </summary>
    private async Task<object> TraceJevAsync(Func<GenAiRequestTelemetry, Task<object>> run)
    {
        using GenAiRequestTelemetry requestTelemetry = GenAiRequestTelemetry.Start(
            TelemetryAttributes.OperationChat, GenAiRequestTelemetry.ModelNameFromPath(LoadedModelPath), LoadedBackend);
        try
        {
            return await run(requestTelemetry).ConfigureAwait(false);
        }
        catch (Exception ex)
        {
            switch (ex)
            {
                case JevValidationException:
                    requestTelemetry.MarkRefused(JevRefusedInvalidRequest);
                    break;
                case JevModelNotFoundException:
                    requestTelemetry.MarkRefused(JevRefusedModelNotFound);
                    break;
                case JevModelUnavailableException:
                    requestTelemetry.MarkRefused(JevRefusedModelUnavailable);
                    break;
                case JevQueueFullException:
                    requestTelemetry.MarkRefused(JevRefusedQueueFull);
                    break;
                default:
                    requestTelemetry.MarkFailed(ex);   // a cancellation counts as cancelled
                    break;
            }
            throw;
        }
    }

    private UploadStoragePolicy ResolveMediaStorage()
    {
        if (MediaStorage != null) return MediaStorage;
        if (_jevMediaStorage != null) return _jevMediaStorage;
        string directory = Path.Combine(Path.GetTempPath(), "tensorsharp-jev-media");
        Directory.CreateDirectory(directory);
        _logger.LogInformation(
            "Jev attachments are stored in {Directory}; set ModelService.MediaStorage to govern them with an upload policy.",
            directory);
        return _jevMediaStorage = new UploadStoragePolicy(directory);
    }

    private static int ReadJevLimit(string variable, int fallback, int minimum, int maximum)
    {
        string? value = Environment.GetEnvironmentVariable(variable);
        if (string.IsNullOrEmpty(value)) return fallback;
        if (!int.TryParse(value, out int result) || result < minimum || result > maximum)
            throw new ArgumentException($"{variable} must be an integer from {minimum} to {maximum}.");
        return result;
    }
}
