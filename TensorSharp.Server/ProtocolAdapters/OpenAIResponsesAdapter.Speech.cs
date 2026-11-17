// Copyright (c) Zhongkai Fu. All rights reserved.
// https://github.com/zhongkaifu/TensorSharp
//
// This file is part of TensorSharp.
//
// TensorSharp is licensed under the BSD-3-Clause license found in the LICENSE file in the root directory of this source tree.
//
// Spoken /v1/responses (`modalities` with "audio", `audio` {voice, format}), on the engine as a driven request. The
// response's message carries an `output_text` part (the transcript: an interleaved reply's text) and an `output_audio`
// part {id, expires_at, format, data, transcript}; a stream sends the transcript as `response.output_audio_transcript.delta`
// and the speech as `response.output_audio.delta` (base64 PCM16, a frame's 80 ms at a time), the event names of OpenAI's
// realtime API.

using System;
using System.Collections.Generic;
using System.Text;
using System.Text.Json;
using System.Threading.Tasks;
using Microsoft.AspNetCore.Http;
using Microsoft.Extensions.Logging;
using TensorSharp.Server.Hosting;
using TensorSharp.Server.Responses;
using TensorSharp.Server.ResponseSerializers;
using TensorSharp.Server.StreamingWriters;

namespace TensorSharp.Server.ProtocolAdapters;

public sealed partial class OpenAIResponsesAdapter
{
    private async Task SpeechResponseAsync(HttpContext ctx, string requestId, string modelName, string instructions, int maxOutputTokens,
        List<ChatMessage> messages, SamplingConfig samplingConfig, SpeechReplyRequest speech, bool stream, bool store, ILogger logger)
    {
        if (!HostedModelGuard.TryEnsureHostedModelLoaded(_svc, modelName,
                _options.StartupModelPath, _options.StartupMmProjPath, _options.DefaultBackend, out string loadError))
        {
            await WriteErrorAsync(ctx, 404, loadError).ConfigureAwait(false);
            return;
        }
        string itemId = OpenAIResponsesFactory.NewMessageItemId();
        string audioId = SpeechReplyStore.NewId();
        var transcript = new StringBuilder();
        var pcm = stream ? null : new List<float>();
        int rate = 0;
        ChatStreamUpdate terminal = default;
        bool started = false;
        IAsyncEnumerator<ChatStreamUpdate> updates = _svc.ChatStreamWithSpeechAsync(messages, maxOutputTokens, ctx.RequestAborted,
            samplingConfig, enableThinking: false, speech.Options).GetAsyncEnumerator(ctx.RequestAborted);
        try
        {
            while (true)
            {
                try
                {
                    if (!await updates.MoveNextAsync().ConfigureAwait(false)) break;
                }
                catch (Exception ex) when (!started && ex is NotSupportedException or ArgumentException)
                {
                    logger.LogWarning(LogEventIds.HttpRequestRejected, "/v1/responses spoken reply rejected: {Error}", ex.Message);
                    await WriteErrorAsync(ctx, 400, ex.Message).ConfigureAwait(false);
                    return;
                }
                ChatStreamUpdate update = updates.Current;
                if (stream && !started)
                {
                    SseWriter.ApplyHeaders(ctx.Response);
                    await SseWriter.WriteNamedEventAsync(ctx.Response, "response.created", OpenAIResponsesFactory.Created(requestId, _svc.LoadedModelName),
                        ctx.RequestAborted).ConfigureAwait(false);
                    await SseWriter.WriteNamedEventAsync(ctx.Response, "response.output_item.added", OpenAIResponsesFactory.OutputItemAdded(0,
                        new { id = itemId, type = "message", status = "in_progress", role = "assistant", content = Array.Empty<object>() }),
                        ctx.RequestAborted).ConfigureAwait(false);
                    await SseWriter.WriteNamedEventAsync(ctx.Response, "response.content_part.added", new
                    {
                        type = "response.content_part.added", item_id = itemId, output_index = 0, content_index = 0,
                        part = new { type = "output_audio", id = audioId, transcript = "" },
                    }, ctx.RequestAborted).ConfigureAwait(false);
                }
                started = true;
                if (update.Done)
                {
                    terminal = update;
                    continue;
                }
                if (update.Audio is { Length: > 0 } samples)
                {
                    rate = update.AudioSampleRate;
                    if (stream)
                        await SseWriter.WriteNamedEventAsync(ctx.Response, "response.output_audio.delta", new
                        {
                            type = "response.output_audio.delta", item_id = itemId, output_index = 0, content_index = 0,
                            delta = Convert.ToBase64String(TensorSharp.Models.Lfm3.Lfm3SpeechAudio.Pcm16(samples)),
                        }, ctx.RequestAborted).ConfigureAwait(false);
                    else
                        pcm.AddRange(samples);
                }
                if (!string.IsNullOrEmpty(update.Piece))
                {
                    transcript.Append(update.Piece);
                    if (stream)
                        await SseWriter.WriteNamedEventAsync(ctx.Response, "response.output_audio_transcript.delta", new
                        {
                            type = "response.output_audio_transcript.delta", item_id = itemId, output_index = 0, content_index = 0,
                            delta = update.Piece,
                        }, ctx.RequestAborted).ConfigureAwait(false);
                }
            }
        }
        finally
        {
            await updates.DisposeAsync().ConfigureAwait(false);
        }
        if (rate == 0) rate = terminal.AudioSampleRate > 0 ? terminal.AudioSampleRate : 24000;
        SpeechReplyStore.Entry entry = OpenAIChatAdapter.SpeechReplies.Add(transcript.ToString(), terminal.SpeechFrames, audioId);
        long expiresAt = entry.ExpiresAt.ToUnixTimeSeconds();
        bool cut = terminal.FinishReason is "max_tokens";
        string status = cut ? "incomplete" : "completed";
        object audioPart = stream
            ? new { type = "output_audio", id = audioId, expires_at = expiresAt, format = "pcm16", transcript = transcript.ToString() }
            : new
            {
                type = "output_audio", id = audioId, expires_at = expiresAt, format = speech.Format,
                data = Convert.ToBase64String(speech.Format == "wav"
                    ? TensorSharp.Models.Lfm3.Lfm3SpeechAudio.Wav(pcm.ToArray(), rate)
                    : TensorSharp.Models.Lfm3.Lfm3SpeechAudio.Pcm16(pcm.ToArray())),
                transcript = transcript.ToString(),
            };
        var item = new
        {
            id = itemId, type = "message", status, role = "assistant",
            content = new object[]
            {
                audioPart,
                new { type = "output_text", text = transcript.ToString(), annotations = Array.Empty<object>() },
            },
        };
        object response = OpenAIResponsesFactory.Response(requestId, _svc.LoadedModelName, status, instructions, maxOutputTokens,
            new object[] { item }, store, samplingConfig, terminal.PromptTokens, terminal.EvalTokens, terminal.KvCacheReusedTokens,
            incompleteReason: cut ? "max_output_tokens" : null);
        if (store)
            _store.Store(new StoredResponse { Id = requestId, Json = JsonSerializer.Serialize(response, JsonOptions.IgnoreNulls) });
        logger.LogInformation(LogEventIds.ChatCompleted, "/v1/responses spoken reply: id={ResponseId} audio={AudioId} mode={Mode} frames={Frames} status={Status}",
            store ? LogContent.Of(requestId) : requestId, audioId, speech.Mode, terminal.SpeechFrames?.Count ?? 0, status);
        if (!stream)
        {
            ctx.Response.ContentType = "application/json; charset=utf-8";
            await ctx.Response.WriteAsync(JsonSerializer.Serialize(response, JsonOptions.IgnoreNulls)).ConfigureAwait(false);
            return;
        }
        await SseWriter.WriteNamedEventAsync(ctx.Response, "response.output_audio.done",
            new { type = "response.output_audio.done", item_id = itemId, output_index = 0, content_index = 0 }, ctx.RequestAborted).ConfigureAwait(false);
        await SseWriter.WriteNamedEventAsync(ctx.Response, "response.output_audio_transcript.done",
            new { type = "response.output_audio_transcript.done", item_id = itemId, output_index = 0, content_index = 0, transcript = transcript.ToString() },
            ctx.RequestAborted).ConfigureAwait(false);
        await SseWriter.WriteNamedEventAsync(ctx.Response, "response.content_part.done",
            new { type = "response.content_part.done", item_id = itemId, output_index = 0, content_index = 0, part = audioPart }, ctx.RequestAborted).ConfigureAwait(false);
        await SseWriter.WriteNamedEventAsync(ctx.Response, "response.output_item.done", OpenAIResponsesFactory.OutputItemDone(0, item), ctx.RequestAborted)
            .ConfigureAwait(false);
        await SseWriter.WriteNamedEventAsync(ctx.Response, "response.completed", OpenAIResponsesFactory.Completed(response), ctx.RequestAborted)
            .ConfigureAwait(false);
        await ctx.Response.Body.FlushAsync(ctx.RequestAborted).ConfigureAwait(false);
    }
}
