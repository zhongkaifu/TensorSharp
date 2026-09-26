// Copyright (c) Zhongkai Fu. All rights reserved.
// Licensed under the BSD-3-Clause license in the repository root.
using System;
using System.Collections.Generic;
using System.IO;
using System.Net.Http;
using System.Net.Http.Headers;
using System.Security.Cryptography;
using System.Text.Json;
using System.Threading;
using System.Threading.Tasks;

namespace TensorSharp.Server.Jev;

/// <summary>
/// Speech-to-text companion for checkpoints without an audio tower. Only an operator-provided
/// endpoint receives audio; request fields cannot choose a URL. The persistent HTTP client and
/// bounded content cache avoid reconnecting and transcribing repeated attachments.
/// </summary>
internal sealed class JevAudioTranscriber
{
    private const int MaxResponseBytes = 1024 * 1024;
    private const int MaxTextCharacters = 32768;
    private const int CacheCapacity = 16;
    private static readonly HttpClient SharedClient = new(new SocketsHttpHandler
    {
        AllowAutoRedirect = false,
        PooledConnectionLifetime = TimeSpan.FromMinutes(5),
    }) { Timeout = Timeout.InfiniteTimeSpan };
    private readonly HttpClient _client;
    private readonly Uri _endpoint;
    private readonly string? _model;
    private readonly string? _apiKey;
    private readonly TimeSpan _timeout;
    private readonly object _cacheLock = new();
    private readonly Dictionary<string, LinkedListNode<(string Key, string Text)>> _cache = new();
    private readonly LinkedList<(string Key, string Text)> _lru = new();

    internal JevAudioTranscriber(Uri endpoint, string? model = null, string? apiKey = null,
        TimeSpan? timeout = null, HttpClient? client = null)
    {
        if (!endpoint.IsAbsoluteUri || endpoint.Scheme is not ("http" or "https") ||
            !string.IsNullOrEmpty(endpoint.UserInfo) || !string.IsNullOrEmpty(endpoint.Fragment))
            throw new ArgumentException("The Jev transcription endpoint must be an absolute HTTP(S) URL without userinfo or a fragment.");
        _endpoint = endpoint;
        _model = model;
        _apiKey = apiKey;
        _timeout = timeout ?? TimeSpan.FromSeconds(120);
        _client = client ?? SharedClient;
    }

    internal static JevAudioTranscriber? FromEnvironment(Func<string, string?>? readEnvironment = null)
    {
        readEnvironment ??= Environment.GetEnvironmentVariable;
        try
        {
            string? endpoint = readEnvironment("TS_JEV_TRANSCRIPTION_URL");
            if (string.IsNullOrWhiteSpace(endpoint)) return null;
            string? rawTimeout = readEnvironment("TS_JEV_TRANSCRIPTION_TIMEOUT_SECONDS");
            int seconds = 120;
            if (rawTimeout != null && (!int.TryParse(rawTimeout, out seconds) || seconds is < 1 or > 600))
                throw new ArgumentException("TS_JEV_TRANSCRIPTION_TIMEOUT_SECONDS must be an integer from 1 to 600.");
            if (!Uri.TryCreate(endpoint, UriKind.Absolute, out var uri))
                throw new ArgumentException("TS_JEV_TRANSCRIPTION_URL must be an absolute HTTP(S) URL.");
            return new(uri, readEnvironment("TS_JEV_TRANSCRIPTION_MODEL"),
                readEnvironment("TS_JEV_TRANSCRIPTION_API_KEY"), TimeSpan.FromSeconds(seconds));
        }
        catch (ArgumentException error)
        {
            // Configuration is resolved on first audio use. Keep operator mistakes in the
            // same explicit 503 contract as a missing/unavailable companion.
            throw new JevModelUnavailableException("Jev transcription configuration is invalid: " + error.Message);
        }
    }

    internal async Task<string> TranscribeAsync(string path, CancellationToken cancellationToken)
    {
        cancellationToken.ThrowIfCancellationRequested();
        await using var stream = new FileStream(path, FileMode.Open, FileAccess.Read, FileShare.Read,
            81920, FileOptions.Asynchronous | FileOptions.SequentialScan);
        string key = Convert.ToHexString(await SHA256.HashDataAsync(stream, cancellationToken).ConfigureAwait(false))
            + Path.GetExtension(path).ToLowerInvariant();
        lock (_cacheLock)
        {
            if (_cache.TryGetValue(key, out var node))
            {
                _lru.Remove(node);
                _lru.AddLast(node);
                return node.Value.Text;
            }
        }
        stream.Position = 0;
        using var deadline = CancellationTokenSource.CreateLinkedTokenSource(cancellationToken);
        deadline.CancelAfter(_timeout);
        using var multipart = new MultipartFormDataContent();
        var file = new StreamContent(stream);
        file.Headers.ContentType = new MediaTypeHeaderValue("application/octet-stream");
        multipart.Add(file, "file", "audio" + Path.GetExtension(path));
        multipart.Add(new StringContent("json"), "response_format");
        if (!string.IsNullOrWhiteSpace(_model)) multipart.Add(new StringContent(_model), "model");
        using var request = new HttpRequestMessage(HttpMethod.Post, _endpoint) { Content = multipart };
        if (!string.IsNullOrWhiteSpace(_apiKey)) request.Headers.Authorization = new AuthenticationHeaderValue("Bearer", _apiKey);
        try
        {
            using var response = await _client.SendAsync(request, HttpCompletionOption.ResponseHeadersRead, deadline.Token).ConfigureAwait(false);
            if (!response.IsSuccessStatusCode)
                throw new JevModelUnavailableException($"The Jev speech transcription service returned HTTP {(int)response.StatusCode}.");
            if (response.Content.Headers.ContentLength > MaxResponseBytes)
                throw new JevModelUnavailableException("The Jev speech transcription response exceeds the 1 MiB limit.");
            await using var body = await response.Content.ReadAsStreamAsync(deadline.Token).ConfigureAwait(false);
            using var buffer = new MemoryStream();
            byte[] chunk = new byte[8192];
            int count;
            while ((count = await body.ReadAsync(chunk, deadline.Token).ConfigureAwait(false)) != 0)
            {
                if (buffer.Length + count > MaxResponseBytes)
                    throw new JevModelUnavailableException("The Jev speech transcription response exceeds the 1 MiB limit.");
                buffer.Write(chunk, 0, count);
            }
            using var json = JsonDocument.Parse(buffer.GetBuffer().AsMemory(0, checked((int)buffer.Length)));
            if (json.RootElement.ValueKind != JsonValueKind.Object ||
                !json.RootElement.TryGetProperty("text", out var text) || text.ValueKind != JsonValueKind.String)
                throw new JevModelUnavailableException("The Jev speech transcription service must return a JSON object with a string 'text' field.");
            string transcript = text.GetString()!.Trim();
            if (transcript.Length == 0)
                throw new JevValidationException("audio: the transcription service detected no speech; no decision was inferred from an empty transcript.");
            if (transcript.Length > MaxTextCharacters)
                throw new JevValidationException($"audio: transcript exceeds {MaxTextCharacters} characters; split the recording.");
            lock (_cacheLock)
            {
                if (_cache.Remove(key, out var existing)) _lru.Remove(existing);
                _cache.Add(key, _lru.AddLast((key, transcript)));
                if (_cache.Count > CacheCapacity)
                {
                    _cache.Remove(_lru.First!.Value.Key);
                    _lru.RemoveFirst();
                }
            }
            return transcript;
        }
        catch (OperationCanceledException) when (!cancellationToken.IsCancellationRequested)
        {
            throw new JevModelUnavailableException("The Jev speech transcription service timed out.");
        }
        catch (HttpRequestException)
        {
            throw new JevModelUnavailableException("The Jev speech transcription service is unavailable.");
        }
        catch (IOException)
        {
            throw new JevModelUnavailableException("The Jev speech transcription response could not be read completely.");
        }
        catch (JsonException)
        {
            throw new JevModelUnavailableException("The Jev speech transcription service returned invalid JSON.");
        }
    }
}
