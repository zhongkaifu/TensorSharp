// Copyright (c) Zhongkai Fu. All rights reserved.
// https://github.com/zhongkaifu/TensorSharp
//
// This file is part of TensorSharp.
//
// TensorSharp is licensed under the BSD-3-Clause license found in the LICENSE file in the root directory of this source tree.
//
// OpenAI's audio routes over the loaded audio model:
//
//   POST /v1/audio/transcriptions   multipart: file, language, prompt, response_format
//                                   (json | text | srt | vtt | verbose_json), temperature,
//                                   timestamp_granularities[] (segment | word)      -> Whisper
//   POST /v1/audio/translations     the same, translated to English                  -> Whisper
//   POST /v1/audio/speech           JSON: input, voice, speed, response_format (wav | pcm)
//                                                                                    -> Kokoro
//                                   LFM3-Omni (a language model that speaks) on the engine:
//                                   input, voice (female | male), language, response_format
//                                   (wav | pcm | pcm16), stream / stream_format (audio | sse),
//                                   seed, temperature, max_frames; speed only 1
//   POST /v1/audio/music            JSON: prompt (style tags), lyrics, duration, bpm, key,
//                                   time_signature, language, seed, steps -> WAV    -> ACE-Step
//
// One request runs at a time per model; each holds a use of the model and its compute lock,
// so an unload waits for it and a request that arrives during one is refused.
using System;
using System.Globalization;
using System.IO;
using System.Linq;
using System.Threading;
using System.Threading.Tasks;
using System.Text.Json;
using Microsoft.AspNetCore.Http;
using Microsoft.Extensions.Logging;
using TensorSharp.Models;
using TensorSharp.Models.Audio;
using TensorSharp.Models.Video;

namespace TensorSharp.Server.ProtocolAdapters;

public sealed class AudioAdapter
{
    private readonly ModelService _svc;
    private readonly ILogger _logger;

    public AudioAdapter(ModelService svc, ILoggerFactory loggerFactory)
    {
        _svc = svc;
        _logger = loggerFactory.CreateLogger("TensorSharp.Server.Audio");
    }

    private static IResult Error(int status, string message, string type = "invalid_request_error") =>
        Results.Json(new { error = new { message, type } }, statusCode: status);

    public async Task<IResult> TranscriptionsAsync(HttpRequest req, bool translate)
    {
        if (_svc.Model is not ISpeechRecognitionModel)
            return Error(400, "The loaded model is not a speech recognition model (load a Whisper GGUF).");
        if (!req.HasFormContentType)
            return Error(400, "Send multipart/form-data with the audio in a 'file' field.");
        var form = await req.ReadFormAsync().ConfigureAwait(false);
        var file = form.Files.GetFile("file");
        if (file == null || file.Length == 0)
            return Error(400, "The 'file' field is required.");

        string format = form["response_format"].FirstOrDefault() ?? "json";
        if (!TranscriptFormats.Names.Contains(format))
            return Error(400, $"response_format must be one of {string.Join(", ", TranscriptFormats.Names)}.");
        var granularities = form["timestamp_granularities[]"].Concat(form["timestamp_granularities"]).ToList();
        bool words = granularities.Any(g => string.Equals(g, "word", StringComparison.OrdinalIgnoreCase));
        if (words && format != "verbose_json")
            return Error(400, "timestamp_granularities=word needs response_format=verbose_json.");
        var options = new TranscriptionOptions
        {
            Language = translate ? null : form["language"].FirstOrDefault(),
            Translate = translate,
            Prompt = form["prompt"].FirstOrDefault(),
            WordTimestamps = words,
        };
        if (float.TryParse(form["temperature"].FirstOrDefault(), NumberStyles.Float, CultureInfo.InvariantCulture, out float t))
            options.Temperature = Math.Clamp(t, 0f, 1f);

        string ext = Path.GetExtension(file.FileName);
        string temp = Path.Combine(Path.GetTempPath(), $"tensorsharp-audio-{Guid.NewGuid():N}{(string.IsNullOrEmpty(ext) ? ".wav" : ext)}");
        try
        {
            await using (var fs = File.Create(temp))
                await file.CopyToAsync(fs).ConfigureAwait(false);
            DecodedAudio audio;
            try { audio = AudioIO.Decode(temp, 16000, 1); }
            catch (Exception ex) when (ex is NotSupportedException or InvalidDataException or IOException)
            {
                return Error(400, $"Could not decode the audio: {ex.Message}");
            }
            var result = await RunAsync<ISpeechRecognitionModel, TranscriptionResult>(m => m.Transcribe(audio, options)).ConfigureAwait(false);
            if (result.Error != null) return result.Error;
            string body = TranscriptFormats.Render(result.Value, format, words);
            return Results.Text(body, TranscriptFormats.ContentType(format));
        }
        finally
        {
            try { File.Delete(temp); } catch (IOException) { } catch (UnauthorizedAccessException) { }
        }
    }

    public async Task<IResult> SpeechAsync(HttpRequest req)
    {
        JsonDocument doc;
        try { doc = await JsonDocument.ParseAsync(req.Body).ConfigureAwait(false); }
        catch (JsonException ex) { return Error(400, $"The body is not JSON: {ex.Message}"); }
        using (doc)
        {
            var root = doc.RootElement;
            if (root.ValueKind != JsonValueKind.Object) return Error(400, "The body must be a JSON object.");
            // A language model that speaks (LFM3-Omni) runs on the engine, batched with the chat requests.
            if (_svc.SpeechOutputModel is { } speaker)
                return await LlmSpeechAsync(req, root, speaker).ConfigureAwait(false);
            if (_svc.Model is not ISpeechSynthesisModel)
                return Error(400, "The loaded model is not a speech synthesis model (load a Kokoro GGUF, or LFM3-Omni with its mmproj).");
            string input = Str(root, "input");
            if (string.IsNullOrWhiteSpace(input))
                return Error(400, "input is required.");
            string format = Str(root, "response_format") ?? "wav";
            if (format is not ("wav" or "pcm"))
                return Error(400, "response_format must be wav or pcm (16-bit little-endian mono at the model's rate).");
            if (FirstPresent(root, "seed", "temperature", "max_frames") is { } sampled)
                return Error(400, $"{sampled} applies to a model that samples its speech (LFM3-Omni); the loaded model's speech is not sampled.", "invalid_request_error");
            var options = new SpeechSynthesisOptions
            {
                Voice = Str(root, "voice"),
                Language = Str(root, "language"),
                Speed = root.TryGetProperty("speed", out var sp) && sp.ValueKind == JsonValueKind.Number ? (float)sp.GetDouble() : 1f,
            };
            if (options.Speed is < 0.25f or > 4f)
                return Error(400, "speed must be between 0.25 and 4.");
            var result = await RunAsync<ISpeechSynthesisModel, DecodedAudio>(m => m.Synthesize(input, options)).ConfigureAwait(false);
            if (result.Error != null) return result.Error;
            return AudioResult(result.Value, format);
        }
    }

    private static string FirstPresent(JsonElement root, params string[] names)
    {
        foreach (string name in names)
            if (root.TryGetProperty(name, out var v) && v.ValueKind != JsonValueKind.Null) return name;
        return null;
    }

    /// <summary>
    /// <c>POST /v1/audio/speech</c> on a language model that speaks (LFM3-Omni): its TTS prompt runs on the engine as a
    /// driven request. Fields: <c>input</c>; <c>voice</c> (one of the model's, case-insensitive; anything else, OpenAI's
    /// default <c>alloy</c> included, is a 400 listing them); <c>language</c>; <c>response_format</c> <c>wav</c> (default),
    /// <c>pcm</c> or <c>pcm16</c> (both 16-bit little-endian mono); <c>speed</c> only 1 (no rate control); <c>seed</c>,
    /// <c>temperature</c> (the speech temperature), <c>max_frames</c> (80 ms each, default 512); <c>stream</c>
    /// <c>true</c> (or OpenAI's <c>stream_format</c> <c>"audio"</c>) sends the audio chunked as each frame is decoded, a WAV
    /// stream with an unknown-length header (sizes 0xFFFFFFFF, as llama.cpp and ffmpeg write); <c>stream_format</c>
    /// <c>"sse"</c> sends OpenAI's <c>speech.audio.delta</c> / <c>speech.audio.done</c> events (base64 PCM16).
    /// </summary>
    private async Task<IResult> LlmSpeechAsync(HttpRequest req, JsonElement root, ISpeechOutputModel speaker)
    {
        if (!speaker.HasSpeechOutput)
            return Error(400, speaker.SpeechOutputUnavailableReason ?? "The loaded model has no speech output.");
        string input = Str(root, "input");
        if (string.IsNullOrWhiteSpace(input))
            return Error(400, "input is required.");
        if (root.TryGetProperty("voice", out var voiceProp) && voiceProp.ValueKind is not (JsonValueKind.String or JsonValueKind.Null))
            return Error(400, $"voice must be a string naming one of the model's voices ({string.Join(", ", speaker.SpeechVoices)}).");
        string voice = Str(root, "voice");
        if (voice != null && !speaker.IsSpeechVoice(voice))
            return Error(400, $"voice '{voice}' is not one of the loaded model's voices: {string.Join(", ", speaker.SpeechVoices)} " +
                             "(case does not matter). OpenAI clients send 'alloy' by default: name one of these instead.");
        string format = Str(root, "response_format") ?? "wav";
        if (format == "pcm16") format = "pcm";
        if (format is not ("wav" or "pcm"))
            return Error(400, $"response_format '{format}' is not served: wav, pcm or pcm16 (16-bit little-endian mono at {speaker.SpeechSampleRate} Hz).");
        if (root.TryGetProperty("speed", out var sp) && sp.ValueKind != JsonValueKind.Null
            && (sp.ValueKind != JsonValueKind.Number || sp.GetDouble() != 1.0))
            return Error(400, "speed must be 1: the loaded model has no speaking-rate control.");
        string streamFormat = Str(root, "stream_format");
        if (streamFormat is not (null or "audio" or "sse"))
            return Error(400, "stream_format must be \"audio\" or \"sse\".");
        bool stream = streamFormat != null || (root.TryGetProperty("stream", out var st) && st.ValueKind == JsonValueKind.True);
        if (root.TryGetProperty("stream", out var stv) && stv.ValueKind is not (JsonValueKind.True or JsonValueKind.False or JsonValueKind.Null))
            return Error(400, "stream must be a boolean.");
        bool sse = streamFormat == "sse";
        if (sse && format != "pcm")
            return Error(400, "stream_format \"sse\" carries PCM16 (response_format pcm); a WAV stream is stream_format \"audio\".");
        var options = new SpeechSynthesisOptions { Voice = voice, Language = Str(root, "language") };
        if (root.TryGetProperty("seed", out var seedProp) && seedProp.ValueKind != JsonValueKind.Null)
        {
            if (seedProp.ValueKind != JsonValueKind.Number || !seedProp.TryGetInt64(out long seed) || seed < 0 || seed > uint.MaxValue)
                return Error(400, "seed must be an integer from 0 to 4294967295.");
            options.Seed = (uint)seed;
        }
        if (root.TryGetProperty("temperature", out var tProp) && tProp.ValueKind != JsonValueKind.Null)
        {
            if (tProp.ValueKind != JsonValueKind.Number || tProp.GetDouble() < 0 || !double.IsFinite(tProp.GetDouble()))
                return Error(400, "temperature (the speech temperature) must be a number of 0 (argmax) or more.");
            options.Temperature = (float)tProp.GetDouble();
        }
        if (root.TryGetProperty("max_frames", out var mfProp) && mfProp.ValueKind != JsonValueKind.Null)
        {
            if (mfProp.ValueKind != JsonValueKind.Number || !mfProp.TryGetInt32(out int frames) || frames < 1)
                return Error(400, "max_frames must be a positive integer (80 ms of speech each).");
            options.MaxFrames = frames;
        }

        // The first update settles the request: a refusal (an unloaded model, an engine that cannot run it) is still a
        // status code, not a broken stream.
        var updates = _svc.SpeechSynthesisStreamAsync(input, options, req.HttpContext.RequestAborted).GetAsyncEnumerator(req.HttpContext.RequestAborted);
        bool any;
        try
        {
            any = await updates.MoveNextAsync().ConfigureAwait(false);
        }
        catch (Exception ex) when (ex is ArgumentException or NotSupportedException)
        {
            await updates.DisposeAsync().ConfigureAwait(false);
            _logger.LogWarning(ex, "Speech request rejected");
            return Error(400, ex.Message);
        }
        catch (Exception ex) when (ex is InvalidOperationException && !req.HttpContext.RequestAborted.IsCancellationRequested)
        {
            await updates.DisposeAsync().ConfigureAwait(false);
            _logger.LogError(ex, "Speech request failed");
            return Error(ex is ModelUnloadedException ? 503 : 500, ex.Message, "server_error");
        }
        int rate = speaker.SpeechSampleRate;
        if (!stream)
        {
            var pcm = new System.Collections.Generic.List<float>();
            try
            {
                while (any)
                {
                    if (updates.Current.Audio is { } chunk) pcm.AddRange(chunk);
                    any = await updates.MoveNextAsync().ConfigureAwait(false);
                }
            }
            catch (InvalidOperationException ex)
            {
                _logger.LogError(ex, "Speech request failed");
                return Error(500, ex.Message, "server_error");
            }
            finally
            {
                await updates.DisposeAsync().ConfigureAwait(false);
            }
            float[] samples = pcm.ToArray();
            return format == "wav"
                ? Results.File(TensorSharp.Models.Lfm3.Lfm3SpeechAudio.Wav(samples, rate), "audio/wav", "speech.wav")
                : Results.File(TensorSharp.Models.Lfm3.Lfm3SpeechAudio.Pcm16(samples), "audio/pcm");
        }
        return new SpeechStreamResult(updates, any, format, sse, rate, _logger);
    }

    /// <summary>A speech reply streamed as its frames are decoded: chunked audio (a WAV stream's header first) or OpenAI's
    /// SSE events.</summary>
    private sealed class SpeechStreamResult : IResult
    {
        private readonly System.Collections.Generic.IAsyncEnumerator<ChatStreamUpdate> _updates;
        private readonly bool _first, _sse;
        private readonly string _format;
        private readonly int _rate;
        private readonly ILogger _logger;

        public SpeechStreamResult(System.Collections.Generic.IAsyncEnumerator<ChatStreamUpdate> updates, bool first, string format, bool sse,
            int rate, ILogger logger)
        {
            _updates = updates;
            _first = first;
            _format = format;
            _sse = sse;
            _rate = rate;
            _logger = logger;
        }

        public async Task ExecuteAsync(HttpContext context)
        {
            var response = context.Response;
            var ct = context.RequestAborted;
            response.StatusCode = 200;
            response.ContentType = _sse ? "text/event-stream" : _format == "wav" ? "audio/wav" : "audio/pcm";
            response.Headers.CacheControl = "no-cache";
            if (_sse) response.Headers["X-Accel-Buffering"] = "no";
            int promptTokens = 0, frames = 0;
            try
            {
                if (!_sse && _format == "wav")
                    await response.Body.WriteAsync(TensorSharp.Models.Lfm3.Lfm3SpeechAudio.StreamingWavHeader(_rate), ct).ConfigureAwait(false);
                await response.Body.FlushAsync(ct).ConfigureAwait(false);
                bool any = _first;
                while (any)
                {
                    ChatStreamUpdate update = _updates.Current;
                    if (update.Audio is { Length: > 0 } chunk)
                    {
                        byte[] pcm = TensorSharp.Models.Lfm3.Lfm3SpeechAudio.Pcm16(chunk);
                        if (_sse)
                            await WriteEventAsync(response, new { type = "speech.audio.delta", audio = Convert.ToBase64String(pcm) }, ct).ConfigureAwait(false);
                        else
                            await response.Body.WriteAsync(pcm, ct).ConfigureAwait(false);
                        await response.Body.FlushAsync(ct).ConfigureAwait(false);
                    }
                    if (update.Done)
                    {
                        promptTokens = update.PromptTokens;
                        frames = update.EvalTokens;
                    }
                    any = await _updates.MoveNextAsync().ConfigureAwait(false);
                }
                if (_sse)
                    await WriteEventAsync(response, new
                    {
                        type = "speech.audio.done",
                        usage = new { input_tokens = promptTokens, output_tokens = frames, total_tokens = promptTokens + frames },
                    }, ct).ConfigureAwait(false);
            }
            catch (OperationCanceledException) when (ct.IsCancellationRequested)
            {
                // The client went away: the pipeline's iterator stops the engine request when it is disposed.
            }
            catch (Exception ex)
            {
                // The status is sent: the stream just ends (an SSE client is told why).
                _logger.LogError(ex, "Speech stream failed");
                if (_sse && !ct.IsCancellationRequested)
                    await WriteEventAsync(response, new { type = "error", error = new { message = ex.Message, type = "server_error" } }, CancellationToken.None)
                        .ConfigureAwait(false);
            }
            finally
            {
                await _updates.DisposeAsync().ConfigureAwait(false);
            }
        }

        private static async Task WriteEventAsync(HttpResponse response, object payload, CancellationToken ct)
        {
            byte[] line = System.Text.Encoding.UTF8.GetBytes("data: " + JsonSerializer.Serialize(payload) + "\n\n");
            await response.Body.WriteAsync(line, ct).ConfigureAwait(false);
            await response.Body.FlushAsync(ct).ConfigureAwait(false);
        }
    }

    public async Task<IResult> MusicAsync(HttpRequest req)
    {
        if (_svc.Model is not IMusicGenerationModel)
            return Error(400, "The loaded model is not a music generation model (load an ACE-Step GGUF).");
        using var doc = await JsonDocument.ParseAsync(req.Body).ConfigureAwait(false);
        var root = doc.RootElement;
        var request = new MusicGenerationRequest
        {
            Tags = Str(root, "prompt") ?? Str(root, "tags") ?? string.Empty,
            Lyrics = Str(root, "lyrics") ?? "[Instrumental]",
            KeyScale = Str(root, "key"),
            TimeSignature = Str(root, "time_signature"),
            Language = Str(root, "language") ?? "en",
        };
        if (root.TryGetProperty("duration", out var d) && d.ValueKind == JsonValueKind.Number) request.DurationSeconds = d.GetDouble();
        if (root.TryGetProperty("bpm", out var b) && b.ValueKind == JsonValueKind.Number) request.Bpm = b.GetInt32();
        if (root.TryGetProperty("seed", out var s) && s.ValueKind == JsonValueKind.Number) request.Seed = s.GetInt64();
        if (root.TryGetProperty("steps", out var st) && st.ValueKind == JsonValueKind.Number) request.Steps = st.GetInt32();
        if (root.TryGetProperty("audio_codes", out var ac) && ac.ValueKind is JsonValueKind.True or JsonValueKind.False)
            request.GenerateAudioCodes = ac.GetBoolean();
        if (request.DurationSeconds is <= 0 or > 600)
            return Error(400, "duration must be between 0 and 600 seconds.");
        string format = Str(root, "response_format") ?? "wav";
        if (format is not ("wav" or "pcm"))
            return Error(400, "response_format must be wav or pcm.");
        var result = await RunAsync<IMusicGenerationModel, DecodedAudio>(m => m.GenerateMusic(request)).ConfigureAwait(false);
        if (result.Error != null) return result.Error;
        return AudioResult(result.Value, format);
    }

    private static string Str(JsonElement root, string name) =>
        root.TryGetProperty(name, out var v) && v.ValueKind == JsonValueKind.String ? v.GetString() : null;

    private static IResult AudioResult(DecodedAudio audio, string format)
    {
        if (format == "wav")
            return Results.File(WavWriter.Encode(audio.Channels, audio.SampleRate), "audio/wav", "audio.wav");
        int n = audio.SampleCount, ch = audio.ChannelCount;
        var pcm = new byte[(long)n * ch * 2];
        for (int i = 0; i < n; i++)
            for (int c = 0; c < ch; c++)
            {
                short v = (short)Math.Clamp(Math.Round(audio.Channels[c][i] * 32767.0), short.MinValue, short.MaxValue);
                pcm[((long)i * ch + c) * 2] = (byte)v;
                pcm[((long)i * ch + c) * 2 + 1] = (byte)(v >> 8);
            }
        return Results.File(pcm, "audio/pcm");
    }

    private readonly record struct Outcome<T>(T Value, IResult Error);

    /// <summary>Run <paramref name="work"/> on the loaded model off the request thread, holding a use of
    /// the model and its compute lock; the model's refusals become 400s and its failures 500s.</summary>
    private async Task<Outcome<TResult>> RunAsync<TModel, TResult>(Func<TModel, TResult> work) where TModel : class
    {
        if (_svc.Model is not TModel typed || _svc.Model is not { } owner)
            return new(default, Error(400, "The loaded model changed before the request started."));
        if (!owner.TryEnterUse())
            return new(default, Error(503, "The model is being unloaded; retry after loading it.", "server_error"));
        try
        {
            return await Task.Run(() =>
            {
                lock (owner.GpuComputeLock)
                {
                    if (owner.IsRetiring)
                        return new Outcome<TResult>(default, Error(503, "The model is being unloaded.", "server_error"));
                    try
                    {
                        return new Outcome<TResult>(work(typed), null);
                    }
                    catch (Exception ex) when (ex is ArgumentException or NotSupportedException)
                    {
                        _logger.LogWarning(ex, "Audio request rejected");
                        return new Outcome<TResult>(default, Error(400, ex.Message));
                    }
                    catch (InvalidOperationException ex)
                    {
                        _logger.LogError(ex, "Audio request failed");
                        return new Outcome<TResult>(default, Error(500, ex.Message, "server_error"));
                    }
                }
            }).ConfigureAwait(false);
        }
        finally
        {
            owner.ExitUse();
        }
    }
}
