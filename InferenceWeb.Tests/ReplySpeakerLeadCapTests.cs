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
using TensorSharp.Models.Audio;
using TensorSharp.Server;

namespace InferenceWeb.Tests;

/// <summary>
/// A spoken reply's speech shares the engine with its text, so while the text is still being written ReplySpeaker starts
/// the next piece only once the speech made so far is less than its cap ahead of the listener (the web chat's default,
/// 3 s); once the text is done every piece is spoken at once. The speaker's clock is a seam here, so nothing depends on
/// how fast the test runs.
/// </summary>
public sealed class ReplySpeakerLeadCapTests
{
    private static readonly TimeSpan Patience = TimeSpan.FromSeconds(30);

    /// <summary>A TTS whose every piece is 5 s of speech, made at once.</summary>
    private static async IAsyncEnumerable<ChatStreamUpdate> FiveSeconds(string piece, int index, List<int> calls,
        [EnumeratorCancellation] CancellationToken ct)
    {
        lock (calls) calls.Add(index);
        await Task.Yield();
        yield return ChatStreamUpdate.AudioChunk(new float[5 * 24000], 24000);
        yield return new ChatStreamUpdate("", true, 3, 1, 0, 0, 0, 0, "eos") { AudioSampleRate = 24000 };
    }

    private static async Task Until(Func<bool> condition, string what)
    {
        var deadline = DateTime.UtcNow + Patience;
        while (!condition())
        {
            if (DateTime.UtcNow > deadline) Assert.Fail("never: " + what);
            await Task.Delay(10);
        }
    }

    private static int Count(List<int> calls) { lock (calls) return calls.Count; }

    [Fact]
    public async Task TheNextPieceWaitsWhileTheSpeechIsFarAheadAndTheTextIsStillComing()
    {
        double now = 0;
        var calls = new List<int>();
        using var speaker = new ReplySpeaker((piece, index, ct) => FiveSeconds(piece, index, calls, ct),
            new SpeechTextSegmenter(firstPieceMinChars: 0), CancellationToken.None, ReplySpeaker.DefaultMaxLeadSeconds, () => now);

        speaker.Say("The first sentence. The second sentence. The third");
        await Until(() => Count(calls) == 1 && speaker.Samples == 5 * 24000, "the first piece spoken");
        // 5 s of speech made, none of it heard yet: 5 s ahead, past the 3 s cap, and the text is still being written.
        Assert.Equal(5.0, speaker.LeadSeconds, 3);
        await Task.Delay(300);
        Assert.Equal(1, Count(calls));

        // 2.5 s later the listener is 2.5 s from the end of it: the next piece goes.
        now = 2500;
        await Until(() => Count(calls) == 2, "the second piece started once the lead was under the cap");
        Assert.True(speaker.HeldMs >= 2500, $"held {speaker.HeldMs} ms");

        // 7.5 s ahead again, but the text is done: the rest is spoken at once.
        speaker.Complete();
        await Until(() => Count(calls) == 3, "the last piece spoken once the text was done");
        await speaker.Completion.WaitAsync(Patience);
        Assert.Equal(new[] { "The first sentence.", "The second sentence.", "The third" }, speaker.Spoken);
    }

    [Fact]
    public async Task WithoutACapEveryPieceIsSpokenAsSoonAsItIsCut()
    {
        var calls = new List<int>();
        using var speaker = new ReplySpeaker((piece, index, ct) => FiveSeconds(piece, index, calls, ct),
            new SpeechTextSegmenter(firstPieceMinChars: 0), CancellationToken.None, null, () => 0);

        speaker.Say("The first sentence. The second sentence. The third");
        await Until(() => Count(calls) == 2, "both cut pieces spoken, however far ahead");
        Assert.Equal(0, speaker.HeldMs);
        speaker.Complete();
        await speaker.Completion.WaitAsync(Patience);
        Assert.Equal(3, Count(calls));
    }

    [Fact]
    public void TheWebChatCapsItsSpeechAtThreeSeconds()
    {
        Assert.Equal(3.0, ReplySpeaker.DefaultMaxLeadSeconds);
        Assert.Throws<ArgumentOutOfRangeException>(() => new ReplySpeaker((_, _, _) => Empty(), null, CancellationToken.None, 0));
    }

    private static async IAsyncEnumerable<ChatStreamUpdate> Empty()
    {
        await Task.Yield();
        yield break;
    }
}
