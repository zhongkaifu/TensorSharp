// Copyright (c) Zhongkai Fu. All rights reserved.
// https://github.com/zhongkaifu/TensorSharp
//
// This file is part of TensorSharp.
//
// TensorSharp is licensed under the BSD-3-Clause license found in the LICENSE file in the root directory of this source tree.
//
// Spoken chat replies (OpenAI's audio output): `modalities` ["text","audio"] (interleaved text and speech, the text being
// the speech's transcript) or ["audio"] (speech only), with `audio` {voice, format}. The reply runs on the engine as a
// driven request (TensorSharp.Models.Audio.ISpeechOutputModel), batched with every other request.

using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Text.Json;
using System.Threading.Tasks;
using Microsoft.AspNetCore.Http;
using Microsoft.Extensions.Logging;
using TensorSharp.Models.Audio;
using TensorSharp.Server.Hosting;
using TensorSharp.Server.Responses;
using TensorSharp.Server.ResponseSerializers;
using TensorSharp.Server.StreamingWriters;

namespace TensorSharp.Server.ProtocolAdapters;

/// <summary>What a request asks a spoken reply to be (parsed from <c>modalities</c> and <c>audio</c>).</summary>
internal sealed class SpeechReplyRequest
{
    public SpeechReplyMode Mode { get; init; }
    public string Voice { get; init; }
    public string Language { get; init; }
    /// <summary>"wav" or "pcm16" (16-bit little-endian mono).</summary>
    public string Format { get; init; }
    public uint? Seed { get; init; }
    public float? Temperature { get; init; }

    public SpeechReplyOptions Options => new()
    {
        Mode = Mode, Voice = Voice, Language = Language, Seed = Seed, Temperature = Temperature, Synthesis = false,
    };

    /// <summary>
    /// Read <c>modalities</c> and <c>audio</c> from an OpenAI chat or Responses body: null (and no error) for a text reply,
    /// else the spoken reply; <paramref name="error"/> says what is wrong with them. <c>audio</c>: <c>voice</c>,
    /// <c>format</c> (<c>wav</c> or <c>pcm16</c>; a stream is <c>pcm16</c>, as OpenAI's), and TensorSharp's
    /// <c>language</c>, <c>seed</c>, <c>temperature</c> (the speech's).
    /// </summary>
    public static SpeechReplyRequest Parse(JsonElement body, bool stream, out string error)
    {
        error = null;
        bool hasModalities = body.TryGetProperty("modalities", out JsonElement modalities) && modalities.ValueKind != JsonValueKind.Null;
        bool hasAudio = body.TryGetProperty("audio", out JsonElement audio) && audio.ValueKind != JsonValueKind.Null;
        if (!hasModalities)
        {
            if (hasAudio) error = "audio is set but modalities is not: ask for [\"text\", \"audio\"] or [\"audio\"].";
            return null;
        }
        if (modalities.ValueKind != JsonValueKind.Array)
        {
            error = "modalities must be an array of \"text\" and \"audio\".";
            return null;
        }
        bool text = false, speech = false;
        foreach (JsonElement m in modalities.EnumerateArray())
        {
            string value = m.ValueKind == JsonValueKind.String ? m.GetString() : null;
            if (value == "text") text = true;
            else if (value == "audio") speech = true;
            else
            {
                error = $"modalities holds {m.GetRawText()}: only \"text\" and \"audio\".";
                return null;
            }
        }
        if (!speech)
        {
            if (hasAudio) error = "audio is set but modalities does not ask for \"audio\".";
            return null;
        }
        if (hasAudio && audio.ValueKind != JsonValueKind.Object)
        {
            error = "audio must be an object: {\"voice\": ..., \"format\": \"wav\" | \"pcm16\"}.";
            return null;
        }
        string voice = null, format = null, language = null;
        uint? seed = null;
        float? temperature = null;
        if (hasAudio)
        {
            if (audio.TryGetProperty("voice", out JsonElement v) && v.ValueKind != JsonValueKind.Null)
            {
                if (v.ValueKind != JsonValueKind.String) { error = "audio.voice must be a string."; return null; }
                voice = v.GetString();
            }
            if (audio.TryGetProperty("format", out JsonElement f) && f.ValueKind != JsonValueKind.Null)
            {
                format = f.ValueKind == JsonValueKind.String ? f.GetString() : null;
                if (format is not ("wav" or "pcm16"))
                {
                    error = $"audio.format {f.GetRawText()} is not served: wav or pcm16 (16-bit little-endian mono).";
                    return null;
                }
            }
            if (audio.TryGetProperty("language", out JsonElement l) && l.ValueKind == JsonValueKind.String) language = l.GetString();
            if (audio.TryGetProperty("seed", out JsonElement s) && s.ValueKind != JsonValueKind.Null)
            {
                if (s.ValueKind != JsonValueKind.Number || !s.TryGetInt64(out long sv) || sv < 0 || sv > uint.MaxValue)
                {
                    error = "audio.seed must be an integer from 0 to 4294967295.";
                    return null;
                }
                seed = (uint)sv;
            }
            if (audio.TryGetProperty("temperature", out JsonElement t) && t.ValueKind != JsonValueKind.Null)
            {
                if (t.ValueKind != JsonValueKind.Number || !(t.GetDouble() >= 0) || !double.IsFinite(t.GetDouble()))
                {
                    error = "audio.temperature (the speech's) must be a number of 0 (argmax) or more.";
                    return null;
                }
                temperature = (float)t.GetDouble();
            }
        }
        format ??= stream ? "pcm16" : "wav";
        if (stream && format != "pcm16")
        {
            error = "a streamed spoken reply carries pcm16 audio (as OpenAI's): set audio.format to \"pcm16\".";
            return null;
        }
        return new SpeechReplyRequest
        {
            Mode = text ? SpeechReplyMode.Interleaved : SpeechReplyMode.Audio,
            Voice = voice, Format = format, Language = language, Seed = seed, Temperature = temperature,
        };
    }

    /// <summary>Why the loaded model cannot give this reply (null: it can): it does not speak, its speech decoder is not
    /// loaded, or the voice is not one of its.</summary>
    public string Refusal(ISpeechOutputModel speaker, string architecture)
    {
        if (speaker == null) return $"The loaded model ({architecture ?? "none"}) has no speech output: ask for modalities [\"text\"].";
        if (!speaker.HasSpeechOutput) return speaker.SpeechOutputUnavailableReason;
        if (Voice != null && !speaker.IsSpeechVoice(Voice))
            return $"audio.voice '{Voice}' is not one of the loaded model's voices: {string.Join(", ", speaker.SpeechVoices)} " +
                   "(case does not matter). OpenAI clients send 'alloy' by default: name one of these instead.";
        return null;
    }
}

public sealed partial class OpenAIChatAdapter
{
    /// <summary>The spoken replies sent (OpenAI's audio ids), shared by the chat and Responses adapters.</summary>
    internal static SpeechReplyStore SpeechReplies { get; } = new();

    /// <summary>
    /// A spoken reply. Non-streaming: <c>message.audio</c> = {id, expires_at, data (base64 WAV or PCM16 of the whole
    /// reply), transcript}, <c>message.content</c> null (OpenAI's shape). Streaming: <c>delta.audio</c> = {id} first, then
    /// {transcript} pieces and {data} (base64 PCM16, a frame's 80 ms at a time) in the order the reply makes them, then
    /// {expires_at}, the finish chunk and <c>[DONE]</c>.
    /// </summary>
    private async Task SpeechCompletionAsync(HttpContext ctx, string requestId, string modelName, List<ChatMessage> messages,
        int maxTokens, SamplingConfig? samplingConfig, SpeechReplyRequest speech, bool stream, TextLoraAdapter? lora, ILogger logger)
    {
        if (!HostedModelGuard.TryEnsureHostedModelLoaded(_svc, modelName,
                _options.StartupModelPath, _options.StartupMmProjPath, _options.DefaultBackend, out string loadError))
        {
            ctx.Response.StatusCode = 404;
            await ctx.Response.WriteAsJsonAsync(new { error = new { message = loadError, type = "invalid_request_error" } }).ConfigureAwait(false);
            return;
        }
        string audioId = SpeechReplyStore.NewId();
        var transcript = new StringBuilder();
        var pcm = stream ? null : new List<float>();
        int rate = 0;
        ChatStreamUpdate terminal = default;
        bool started = false;
        IAsyncEnumerator<ChatStreamUpdate> updates = _svc.ChatStreamWithSpeechAsync(messages, maxTokens, ctx.RequestAborted, samplingConfig,
            enableThinking: false, speech.Options, lora).GetAsyncEnumerator(ctx.RequestAborted);
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
                    logger.LogWarning(LogEventIds.HttpRequestRejected, "/v1/chat/completions spoken reply rejected: {Error} (id={ChatcmplId})",
                        ex.Message, requestId);
                    ctx.Response.StatusCode = 400;
                    await ctx.Response.WriteAsJsonAsync(new { error = new { message = ex.Message, type = "invalid_request_error" } }).ConfigureAwait(false);
                    return;
                }
                ChatStreamUpdate update = updates.Current;
                if (stream && !started)
                {
                    started = true;
                    SseWriter.ApplyHeaders(ctx.Response);
                    await SseWriter.WriteEventAsync(ctx.Response, AudioChunk(requestId, new { id = audioId }, role: "assistant"),
                        ctx.RequestAborted).ConfigureAwait(false);
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
                        await SseWriter.WriteEventAsync(ctx.Response,
                            AudioChunk(requestId, new { data = Convert.ToBase64String(TensorSharp.Models.Lfm3.Lfm3SpeechAudio.Pcm16(samples)) }),
                            ctx.RequestAborted).ConfigureAwait(false);
                    else
                        pcm.AddRange(samples);
                }
                if (!string.IsNullOrEmpty(update.Piece))
                {
                    transcript.Append(update.Piece);
                    if (stream)
                        await SseWriter.WriteEventAsync(ctx.Response, AudioChunk(requestId, new { transcript = update.Piece }),
                            ctx.RequestAborted).ConfigureAwait(false);
                }
            }
        }
        finally
        {
            await updates.DisposeAsync().ConfigureAwait(false);
        }
        if (rate == 0) rate = terminal.AudioSampleRate > 0 ? terminal.AudioSampleRate : 24000;
        SpeechReplyStore.Entry entry = SpeechReplies.Add(transcript.ToString(), terminal.SpeechFrames, audioId, terminal.SpokenTurn);
        long expiresAt = entry.ExpiresAt.ToUnixTimeSeconds();
        string finish = FinishReasonMapper.ToOpenAIChat(terminal.FinishReason, false);
        logger.LogInformation(LogEventIds.ChatCompleted,
            "/v1/chat/completions spoken reply: id={ChatcmplId} audio={AudioId} mode={Mode} frames={Frames} transcriptChars={Chars} finish={Finish}",
            requestId, audioId, speech.Mode, terminal.SpeechFrames?.Count ?? 0, transcript.Length, finish);
        if (stream)
        {
            await SseWriter.WriteEventAsync(ctx.Response, AudioChunk(requestId, new { expires_at = expiresAt }), ctx.RequestAborted).ConfigureAwait(false);
            await SseWriter.WriteEventAsync(ctx.Response, OpenAIResponseFactory.EndChunk(requestId, _svc.LoadedModelName, finish,
                terminal.PromptTokens, terminal.EvalTokens, terminal.KvCacheReusedTokens), ctx.RequestAborted).ConfigureAwait(false);
            await SseWriter.WriteDoneSentinelAsync(ctx.Response, ctx.RequestAborted).ConfigureAwait(false);
            await ctx.Response.Body.FlushAsync(ctx.RequestAborted).ConfigureAwait(false);
            return;
        }
        float[] all = pcm.ToArray();
        byte[] data = speech.Format == "wav" ? TensorSharp.Models.Lfm3.Lfm3SpeechAudio.Wav(all, rate) : TensorSharp.Models.Lfm3.Lfm3SpeechAudio.Pcm16(all);
        var message = new Dictionary<string, object>
        {
            ["role"] = "assistant",
            ["content"] = null,
            ["audio"] = new { id = audioId, expires_at = expiresAt, data = Convert.ToBase64String(data), transcript = transcript.ToString() },
        };
        ctx.Response.ContentType = "application/json; charset=utf-8";
        await ctx.Response.WriteAsync(JsonSerializer.Serialize(OpenAIResponseFactory.Completion(requestId, _svc.LoadedModelName, message,
            finish, terminal.PromptTokens, terminal.EvalTokens, terminal.KvCacheReusedTokens))).ConfigureAwait(false);
    }

    /// <summary>A streamed chunk whose delta carries <c>audio</c> (OpenAI's audio deltas).</summary>
    private object AudioChunk(string requestId, object audio, string role = null) => new
    {
        id = requestId,
        @object = "chat.completion.chunk",
        created = DateTimeOffset.UtcNow.ToUnixTimeSeconds(),
        model = _svc.LoadedModelName,
        choices = new[]
        {
            new
            {
                index = 0,
                delta = role != null ? (object)new { role, content = (string)null, audio } : new { audio },
                finish_reason = (string)null,
            },
        },
    };
}
