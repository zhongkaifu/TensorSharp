// Copyright (c) Zhongkai Fu. All rights reserved.
// https://github.com/zhongkaifu/TensorSharp
//
// This file is part of TensorSharp.
//
// TensorSharp is licensed under the BSD-3-Clause license found in the LICENSE file in the root directory of this source tree.
//
// TensorSharp is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the BSD-3-Clause License for more details.

using System.Runtime.CompilerServices;
using System.Text.Json;
using TensorAgent.Core.Sessions;

namespace TensorAgent.Tests;

/// <summary>
/// A spoken turn's speech goes through the app the way its text does: the turn manager and the route hand every
/// <c>{audio, sampleRate}</c> frame to the page as it is made, between the text frames it was made among, not when the
/// turn ends.
/// </summary>
public sealed partial class ChatTurnTests
{
    private static readonly TimeSpan RelayPatience = TimeSpan.FromSeconds(30);

    [Fact]
    public async Task ASpokenTurnsAudioReachesThePageAsItIsMadeNotWhenTheTurnEnds()
    {
        string root = Path.Combine(Path.GetTempPath(), "turns-" + Guid.NewGuid().ToString("N")[..8]);
        Directory.CreateDirectory(root);
        try
        {
            using var harness = new RouteHarness(root);
            object[] script =
            {
                new { token = "Hello. " },
                new { audio = "AAAAQADA/38=", sampleRate = 24000 },
                new { token = "How are " },
                new { audio = "AIAAIA==", sampleRate = 24000 },
                new { token = "you?" },
                new { audio = "AAA=", sampleRate = 24000 },
                new { done = true, sessionId = harness.SessionId, tokenCount = 3, audioId = "audio_abc" },
            };
            // Each frame is made only once the page has received the one before it, so a relay that held speech back
            // (to the end of the turn, or to a batch of frames) stalls the turn here instead of passing.
            var received = new SemaphoreSlim(0);
            harness.Frames = ct => Lockstep(script, received, ct);

            using HttpResponseMessage response = await harness.SendChatAsync("Say hello.");
            await using Stream body = await response.Content.ReadAsStreamAsync();
            using var reader = new StreamReader(body);
            var seen = new List<JsonElement>();
            while (seen.Count < script.Length)
            {
                string? line = await reader.ReadLineAsync().WaitAsync(RelayPatience);
                if (line is null)
                    break;
                if (!line.StartsWith("data: ", StringComparison.Ordinal))
                    continue;
                using JsonDocument frame = JsonDocument.Parse(line["data: ".Length..]);
                seen.Add(frame.RootElement.Clone());
                received.Release();
            }

            Assert.All(seen, f => Assert.False(f.TryGetProperty("error", out JsonElement e), e.ToString()));
            Assert.Equal(
                new[] { "token", "audio", "token", "audio", "token", "audio", "done" },
                seen.Select(f => f.EnumerateObject().First().Name));
            Assert.Equal(
                new[] { "AAAAQADA/38=", "AIAAIA==", "AAA=" },
                seen.Where(f => f.TryGetProperty("audio", out _)).Select(f => f.GetProperty("audio").GetString()));
            Assert.All(seen.Where(f => f.TryGetProperty("audio", out _)), f => Assert.Equal(24000, f.GetProperty("sampleRate").GetInt32()));

            // The transcript is the text; the speech is not written into it.
            await WaitFor(() => !harness.Turns.StatusOfKey(harness.ConversationId)!.IsRunning);
            StoredMessage answer = Assert.Single(
                new ConversationStore(harness.ConversationRoot).Load(harness.ConversationId)!.Messages, m => m.Role == "assistant");
            Assert.Equal("Hello. How are you?", answer.Content);
        }
        finally
        {
            try { Directory.Delete(root, true); } catch (IOException) { }
        }
    }

    /// <summary><paramref name="script"/>, one frame at a time, each after the reader has taken the one before it.</summary>
    private static async IAsyncEnumerable<object> Lockstep(
        object[] script, SemaphoreSlim received, [EnumeratorCancellation] CancellationToken ct)
    {
        for (int i = 0; i < script.Length; i++)
        {
            if (i > 0 && !await received.WaitAsync(RelayPatience, ct))
                throw new InvalidOperationException($"the page never received frame {i - 1}, so frame {i} was never made");
            yield return script[i];
        }
    }
}
