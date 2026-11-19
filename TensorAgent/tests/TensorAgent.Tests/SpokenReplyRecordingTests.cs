// Copyright (c) Zhongkai Fu. All rights reserved.
// https://github.com/zhongkaifu/TensorSharp
//
// This file is part of TensorSharp.
//
// TensorSharp is licensed under the BSD-3-Clause license found in the LICENSE file in the root directory of this source tree.
//
// TensorSharp is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the BSD-3-Clause License for more details.

using System.Text.Json;
using TensorAgent.Core.Hosting;
using TensorAgent.Core.Sessions;

namespace TensorAgent.Tests;

/// <summary>
/// A spoken reply on the host side of TensorAgent: the turn that wrote it records its audio id, WAV and length from its
/// done frame, so a reopened chat replays it, the next turn splices it, and the conversation keeps its WAV among the
/// uploads it references; Save resolves only that kind of upload.
/// </summary>
public sealed class SpokenReplyRecordingTests : IDisposable
{
    private readonly string _root = Path.Combine(Path.GetTempPath(), "tensoragent-spoken-" + Guid.NewGuid().ToString("N"));

    public void Dispose()
    {
        try { Directory.Delete(_root, recursive: true); } catch { }
    }

    private static async Task WaitFor(Func<bool> condition)
    {
        for (int i = 0; i < 400 && !condition(); i++)
            await Task.Delay(25);
        Assert.True(condition(), "the condition never came true");
    }

    private async Task<StoredMessage> RecordAsync(Func<IAsyncEnumerable<object>> frames)
    {
        var store = new ConversationStore(_root);
        Conversation conversation = store.Create();
        var recorder = new ConversationRecorder(store);
        recorder.Bind("session-spoken", conversation.Id);
        using var turns = new ChatTurnManager(recorder);
        string id = turns.Start(conversation.Id, _ => frames());
        await WaitFor(() => !turns.StatusOfId(id)!.IsRunning);
        return Assert.Single(new ConversationStore(_root).Load(conversation.Id)!.Messages);
    }

    [Fact]
    public async Task TheTurnRecordsTheSpokenReplysIdWavAndLength()
    {
        StoredMessage answer = await RecordAsync(Frames);

        Assert.Equal("Hello, Anna!", answer.Content);
        Assert.Equal("audio_abc", answer.AudioId);
        Assert.Equal("/uploads/speech-1.wav", answer.AudioUrl);
        Assert.Equal(0.24, answer.AudioSeconds);
        Assert.Null(answer.VideoUrl);
        Assert.Contains("speech-1.wav", answer.ReferencedUploads);
        // Saved under the page's own names, which the page sends back.
        string json = JsonSerializer.Serialize(answer);
        Assert.Contains("\"audioId\":\"audio_abc\"", json, StringComparison.Ordinal);
        Assert.Contains("\"audioSeconds\":0.24", json, StringComparison.Ordinal);

        static async IAsyncEnumerable<object> Frames()
        {
            await Task.Yield();
            yield return new { token = "Hello, " };
            yield return new { audio = "AAAAQADA/38=", sampleRate = 24000 };
            yield return new { token = "Anna!" };
            yield return new
            {
                done = true, sessionId = "session-spoken", tokenCount = 9, elapsed = 1.5, tokPerSec = 6.0,
                audioId = "audio_abc", audioUrl = "/uploads/speech-1.wav", audioSeconds = 0.24,
            };
        }
    }

    [Fact]
    public async Task ASpokenReplyWithNoSpeechKeepsOnlyItsId()
    {
        StoredMessage answer = await RecordAsync(Frames);

        Assert.Equal("audio_text", answer.AudioId);
        Assert.Null(answer.AudioUrl);
        Assert.Null(answer.AudioSeconds);

        static async IAsyncEnumerable<object> Frames()
        {
            await Task.Yield();
            yield return new { token = "Hi." };
            yield return new { done = true, sessionId = "session-spoken", tokenCount = 2, elapsed = 0.5, tokPerSec = 4.0, audioId = "audio_text" };
        }
    }

    [Fact]
    public void SaveResolvesASpokenReplysWavAndNothingElse()
    {
        string uploads = Path.Combine(_root, "uploads");
        Directory.CreateDirectory(uploads);
        File.WriteAllBytes(Path.Combine(uploads, "speech-1.wav"), new byte[44]);
        File.WriteAllBytes(Path.Combine(uploads, "generated.png"), MediaFixtures.RedCircleOnWhitePng(16));
        const string page = "http://127.0.0.1:43210/?token=launch";

        Assert.True(UploadDownload.TryResolve(uploads, "/uploads/speech-1.wav", page, "audio", out string? wav));
        Assert.Equal(Path.Combine(uploads, "speech-1.wav"), wav);
        Assert.False(UploadDownload.TryResolve(uploads, "/uploads/generated.png", page, "audio", out _));
        Assert.False(ImageDownload.TryResolve(uploads, "/uploads/speech-1.wav", page, out _));
        Assert.True(ImageDownload.TryResolve(uploads, "/uploads/generated.png", page, out _));
        Assert.False(UploadDownload.TryResolve(uploads, "/uploads/..%2Fspeech-1.wav", page, "audio", out _));
    }
}
