// Copyright (c) Zhongkai Fu. All rights reserved.
// Licensed under the BSD-3-Clause license in the repository root.
using System.Net;
using System.Text;
using TensorSharp.Server.Jev;

namespace InferenceWeb.Tests;

public sealed class JevAudioTranscriberTests : IDisposable
{
    private readonly string _path = Path.Combine(Path.GetTempPath(), "jev-audio-" + Guid.NewGuid().ToString("N") + ".wav");
    public JevAudioTranscriberTests() => File.WriteAllBytes(_path, "RIFF test audio payload"u8.ToArray());
    public void Dispose() => File.Delete(_path);

    [Theory]
    [InlineData("file:///tmp/audio", null)]
    [InlineData("not a url", null)]
    [InlineData("http://localhost/inference", "0")]
    [InlineData("http://localhost/inference", "601")]
    public void InvalidOperatorConfigurationReturnsUnavailable(string endpoint, string? timeout)
    {
        var error = Assert.Throws<JevModelUnavailableException>(() => JevAudioTranscriber.FromEnvironment(
            key => key == "TS_JEV_TRANSCRIPTION_URL" ? endpoint : key == "TS_JEV_TRANSCRIPTION_TIMEOUT_SECONDS" ? timeout : null));
        Assert.Contains("configuration is invalid", error.Message);
    }

    [Fact]
    public async Task SendsMultipartAndCachesByContentInsteadOfFilename()
    {
        int calls = 0;
        using var client = Client(async (request, ct) =>
        {
            calls++;
            Assert.Equal("http://localhost:8178/inference", request.RequestUri!.AbsoluteUri);
            Assert.Equal("Bearer secret", request.Headers.Authorization!.ToString());
            var parts = Assert.IsType<MultipartFormDataContent>(request.Content).ToArray();
            Assert.Equal(3, parts.Length);
            Assert.Contains("file", parts[0].Headers.ContentDisposition!.Name);
            Assert.Equal(await File.ReadAllBytesAsync(_path, ct), await parts[0].ReadAsByteArrayAsync(ct));
            Assert.Equal("json", await parts[1].ReadAsStringAsync(ct));
            Assert.Equal("local-speech", await parts[2].ReadAsStringAsync(ct));
            return Json("{\"text\":\"  Cancel the subscription.  \"}");
        });
        var transcriber = new JevAudioTranscriber(new("http://localhost:8178/inference"), "local-speech", "secret", client: client);
        Assert.Equal("Cancel the subscription.", await transcriber.TranscribeAsync(_path, default));
        Assert.Equal("Cancel the subscription.", await transcriber.TranscribeAsync(_path, default));
        Assert.Equal(1, calls);
        await File.WriteAllTextAsync(_path, "changed bytes");
        await transcriber.TranscribeAsync(_path, default);
        Assert.Equal(2, calls);
        using var cancelled = new CancellationTokenSource();
        cancelled.Cancel();
        await Assert.ThrowsAnyAsync<OperationCanceledException>(() => transcriber.TranscribeAsync(_path, cancelled.Token));
    }

    [Theory]
    [InlineData("{broken")]
    [InlineData("[]")]
    [InlineData("{\"text\":5}")]
    public async Task RejectsMalformedServiceResponsesAndDoesNotCacheFailures(string body)
    {
        int calls = 0;
        using var client = Client((_, _) => Task.FromResult(Json(++calls == 1 ? body : "{\"text\":\"recovered\"}")));
        var transcriber = new JevAudioTranscriber(new("http://localhost/inference"), client: client);
        await Assert.ThrowsAsync<JevModelUnavailableException>(() => transcriber.TranscribeAsync(_path, default));
        Assert.Equal("recovered", await transcriber.TranscribeAsync(_path, default));
        Assert.Equal(2, calls);
    }

    [Theory]
    [InlineData("")]
    [InlineData("  ")]
    public async Task RefusesEmptySpeechInsteadOfMakingAnUnconditionedDecision(string transcript)
    {
        using var client = Client((_, _) => Task.FromResult(Json(System.Text.Json.JsonSerializer.Serialize(new { text = transcript }))));
        var transcriber = new JevAudioTranscriber(new("http://localhost/inference"), client: client);
        await Assert.ThrowsAsync<JevValidationException>(() => transcriber.TranscribeAsync(_path, default));
    }

    [Fact]
    public async Task BoundsResponseAndTranscriptWithoutTruncation()
    {
        using var oversized = Client((_, _) => Task.FromResult(Json(new string('x', 1024 * 1024 + 1))));
        await Assert.ThrowsAsync<JevModelUnavailableException>(() => new JevAudioTranscriber(new("http://localhost/inference"), client: oversized).TranscribeAsync(_path, default));
        using var longText = Client((_, _) => Task.FromResult(Json("{\"text\":\"" + new string('x', 32769) + "\"}")));
        await Assert.ThrowsAsync<JevValidationException>(() => new JevAudioTranscriber(new("http://localhost/inference"), client: longText).TranscribeAsync(_path, default));
    }

    [Fact]
    public async Task TimeoutIsUnavailableButCallerCancellationIsPreserved()
    {
        using var client = Client(async (_, ct) => { await Task.Delay(Timeout.Infinite, ct); return Json("{}"); });
        var transcriber = new JevAudioTranscriber(new("http://localhost/inference"), timeout: TimeSpan.FromMilliseconds(25), client: client);
        await Assert.ThrowsAsync<JevModelUnavailableException>(() => transcriber.TranscribeAsync(_path, default));
        using var cancellation = new CancellationTokenSource(TimeSpan.FromMilliseconds(25));
        transcriber = new JevAudioTranscriber(new("http://localhost/inference"), client: client);
        await Assert.ThrowsAnyAsync<OperationCanceledException>(() => transcriber.TranscribeAsync(_path, cancellation.Token));
    }

    [Fact]
    public async Task DoesNotExposeRemoteErrorsOrCredentials()
    {
        using var client = Client((_, _) => Task.FromResult(new HttpResponseMessage(HttpStatusCode.Unauthorized) { Content = new StringContent("secret details") }));
        var transcriber = new JevAudioTranscriber(new("http://localhost/inference"), client: client);
        var error = await Assert.ThrowsAsync<JevModelUnavailableException>(() => transcriber.TranscribeAsync(_path, default));
        Assert.Contains("401", error.Message);
        Assert.DoesNotContain("secret", error.Message);
    }

    [Fact]
    public async Task AsyncAdmissionProtectsModelLifetimeWhileAwaitingPreprocessing()
    {
        var gate = new JevExecutionGate(1);
        var started = new TaskCompletionSource(TaskCreationOptions.RunContinuationsAsynchronously);
        var finish = new TaskCompletionSource(TaskCreationOptions.RunContinuationsAsynchronously);
        var running = gate.ExecuteAwaitedAsync(async ct => { started.SetResult(); await finish.Task; return 7; }, default);
        await started.Task;
        await Assert.ThrowsAsync<JevQueueFullException>(() => gate.ExecuteAsync(_ => 1, default));
        var changing = Task.Run(() => { using var lease = gate.BeginChange(); });
        Assert.False(changing.IsCompleted);
        finish.SetResult();
        Assert.Equal(7, await running);
        await changing.WaitAsync(TimeSpan.FromSeconds(5));
    }

    private static HttpResponseMessage Json(string body) => new(HttpStatusCode.OK) { Content = new StringContent(body, Encoding.UTF8, "application/json") };
    private static HttpClient Client(Func<HttpRequestMessage, CancellationToken, Task<HttpResponseMessage>> send) => new(new Handler(send));
    private sealed class Handler(Func<HttpRequestMessage, CancellationToken, Task<HttpResponseMessage>> send) : HttpMessageHandler
    {
        protected override Task<HttpResponseMessage> SendAsync(HttpRequestMessage request, CancellationToken cancellationToken) => send(request, cancellationToken);
    }
}
