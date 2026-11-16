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
//   POST /v1/audio/music            JSON: prompt (style tags), lyrics, duration, bpm, key,
//                                   time_signature, language, seed, steps -> WAV    -> ACE-Step
//
// One request runs at a time per model; each holds a use of the model and its compute lock,
// so an unload waits for it and a request that arrives during one is refused.
using System;
using System.Globalization;
using System.IO;
using System.Linq;
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
        if (_svc.Model is not ISpeechSynthesisModel)
            return Error(400, "The loaded model is not a speech synthesis model (load a Kokoro GGUF).");
        using var doc = await JsonDocument.ParseAsync(req.Body).ConfigureAwait(false);
        var root = doc.RootElement;
        string input = Str(root, "input");
        if (string.IsNullOrWhiteSpace(input))
            return Error(400, "input is required.");
        string format = Str(root, "response_format") ?? "wav";
        if (format is not ("wav" or "pcm"))
            return Error(400, "response_format must be wav or pcm (16-bit little-endian mono at the model's rate).");
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
