// Copyright (c) Zhongkai Fu. All rights reserved.
// https://github.com/zhongkaifu/TensorSharp
//
// This file is part of TensorSharp.
//
// TensorSharp is licensed under the BSD-3-Clause license found in the LICENSE file in the root directory of this source tree.

using System.Runtime.CompilerServices;
using TensorSharp.Models;
using TensorSharp.Models.Audio;
using TensorSharp.Models.Lfm3;
using TensorSharp.Server;

namespace InferenceWeb.Tests;

/// <summary>
/// The language of a spoken reply and how it is spoken: LFM3-Omni's own speech modes speak English only, so a reply in
/// another language (the request's, else the one the user writes in) is generated as text and spoken sentence by sentence
/// by the TTS prompt. Checked without a model: each piece's language, the route, the sentence segmenter and the stream
/// that speaks a text reply as it is generated.
/// </summary>
public sealed class SpokenReplyLanguageTests
{
    // ------------------------------------------------------------------------------------------------ languages

    [Theory]
    // No language asked: the text's script's, else none (the default, English).
    [InlineData("北京是中国的首都。", null, "zh")]
    [InlineData("The capital of France is Paris.", null, null)]
    // The language asked, where the text's script suits it.
    [InlineData("北京是中国的首都。", "zh", "zh")]
    [InlineData("北京是中国的首都。", "zh-CN", "zh-CN")]
    [InlineData("C'est la capitale.", "fr", "fr")]
    [InlineData("東京", "ja", "ja")]
    [InlineData("2024年10月1日。", "zh", "zh")]
    [InlineData("123。", "zh", "zh")]
    // English letters in a reply whose language has a script of its own; another script in a Latin-script reply.
    [InlineData("OK, let's go.", "zh", "en")]
    [InlineData("北京是中国的首都。", "fr", "zh")]
    [InlineData("北京是中国的首都。", "en", "zh")]
    public void APieceIsSpokenInTheLanguageItIsWrittenIn(string text, string preferred, string expected)
        => Assert.Equal(expected, Lfm3SpeechPrompt.SpeechLanguageOf(text, preferred));

    [Theory]
    [InlineData(null, true)]
    [InlineData("", true)]
    [InlineData("en", true)]
    [InlineData("EN-GB", true)]
    [InlineData("en-us", true)]
    [InlineData("zh", false)]
    [InlineData("zh-cn", false)]
    [InlineData("ja", false)]
    [InlineData("fr", false)]
    public void OnlyEnglishIsSpokenInTheReplyItself(string language, bool direct)
        => Assert.Equal(direct, Lfm3SpeechPrompt.SpeaksRepliesIn(language));

    // ------------------------------------------------------------------------------------------------ segmenter

    private static List<string> Segment(string text, int chunk = 0, int firstPieceMinChars = SpeechTextSegmenter.DefaultFirstPieceMinChars)
    {
        var segmenter = new SpeechTextSegmenter(firstPieceMinChars: firstPieceMinChars);
        var pieces = new List<string>();
        if (chunk <= 0)
        {
            pieces.AddRange(segmenter.Append(text));
        }
        else
        {
            // Streamed a few characters at a time, as tokens arrive.
            for (int i = 0; i < text.Length; i += chunk)
                pieces.AddRange(segmenter.Append(text.Substring(i, Math.Min(chunk, text.Length - i))));
        }
        pieces.AddRange(segmenter.Flush());
        return pieces;
    }

    [Theory]
    [InlineData(0)]
    [InlineData(1)]
    [InlineData(3)]
    public void SentencesAreThePiecesWhicheverWayTheTextArrives(int chunk)
    {
        Assert.Equal(new[] { "北京是中国的首都。", "它有很多名胜古迹！", "你想去哪里？" },
            Segment("北京是中国的首都。它有很多名胜古迹！你想去哪里？", chunk));
        Assert.Equal(new[] { "The capital is Paris.", "Pi is 3.14 and the file is a.txt!", "Done?" },
            Segment("The capital is Paris. Pi is 3.14 and the file is a.txt! Done?", chunk));
        // Closing quotes stay with their sentence.
        Assert.Equal(new[] { "他说：“你好。”", "然后走了。" }, Segment("他说：“你好。”然后走了。", chunk, firstPieceMinChars: 0));
    }

    [Fact]
    public void MarkdownIsSpokenAsItsText()
    {
        string reply = "以下是三种健康的早餐：\n\n1. **燕麦粥**：富含纤维 🥣\n- *希腊酸奶*配水果\n## 小结\n> 详见[指南](https://example.com)。\n---\n| 早餐 | 热量 |\n|---|---|\n";
        Assert.Equal(new[] { "以下是三种健康的早餐：", "1. 燕麦粥：富含纤维", "希腊酸奶配水果", "小结", "详见指南。", "早餐 热量" },
            Segment(reply, 2));
    }

    [Fact]
    public void FencedCodeIsNotRead()
    {
        Assert.Equal(new[] { "运行这段代码：", "它会打印你好。" },
            Segment("运行这段代码：\n```python\nprint('你好')\n```\n它会打印你好。", 2));
    }

    [Fact]
    public void ALongRunIsCutAtACommaAndTheFirstPieceStartsEarly()
    {
        string clause = "这是一个很长的句子";
        string text = string.Join("，", Enumerable.Repeat(clause, 30)) + "。";
        List<string> pieces = Segment(text, 4);
        Assert.Equal("这是一个很长的句子，", pieces[0]);   // the first piece ends at the first comma after 8 characters
        Assert.All(pieces, p => Assert.True(p.Length <= SpeechTextSegmenter.DefaultMaxChars, p));
        Assert.Equal(text, string.Concat(pieces));
        // Without the early first piece, the first cut is the length limit's.
        Assert.True(Segment(text, 4, firstPieceMinChars: 0)[0].Length > clause.Length * 5);
    }

    [Fact]
    public void NothingSpeakableIsNoPiece()
    {
        Assert.Empty(Segment("🎉🎉\n\n**  **\n---\n"));
        Assert.Equal(new[] { "好的。" }, Segment("好的。😀"));
    }

    // ------------------------------------------------------------------------------------------------ the reply's language

    [Fact]
    public void TheReplysLanguageIsTheRequestsElseTheUsersLastText()
    {
        var speaker = new LanguageSpeaker();
        var history = new List<ChatMessage>
        {
            new() { Role = "system", Content = "You are helpful." },
            new() { Role = "user", Content = "今天天气怎么样？" },
            new() { Role = "assistant", Content = "今天晴天。" },
            new() { Role = "user", Content = "", AudioPaths = new List<string> { "question.wav" } },
        };
        Assert.Equal("zh", ChatGenerationPipeline.SpokenReplyLanguage(speaker, new SpeechReplyOptions(), history));
        Assert.Equal("fr", ChatGenerationPipeline.SpokenReplyLanguage(speaker, new SpeechReplyOptions { Language = " fr " }, history));
        history.Add(new ChatMessage { Role = "user", Content = "And in English?" });
        Assert.Null(ChatGenerationPipeline.SpokenReplyLanguage(speaker, new SpeechReplyOptions(), history));
        Assert.Null(ChatGenerationPipeline.LastUserText(new List<ChatMessage> { new() { Role = "system", Content = "x" } }));
    }

    [Fact]
    public void AChineseConversationIsSpokenFromItsTextAndAnEnglishOneDirectly()
    {
        var speaker = new LanguageSpeaker();
        List<ChatMessage> Ask(string text) => new() { new ChatMessage { Role = "user", Content = text } };
        var web = new SpeechReplyOptions { Mode = SpeechReplyMode.Interleaved, Voice = "female" };   // the Web UI's speech: true

        Assert.Equal(SpeechReplyPath.Synthesized, ChatGenerationPipeline.SpokenReplyRoute(speaker, web, Ask("今天下雨了，我应该带什么？"), out string zh));
        Assert.Equal("zh", zh);
        Assert.Equal(SpeechReplyPath.Direct, ChatGenerationPipeline.SpokenReplyRoute(speaker, web, Ask("What should I bring?"), out string en));
        Assert.Null(en);
        // modalities ["audio"] too; a named language wins over the message's script.
        var audioZh = new SpeechReplyOptions { Mode = SpeechReplyMode.Audio, Language = "zh" };
        Assert.Equal(SpeechReplyPath.Synthesized, ChatGenerationPipeline.SpokenReplyRoute(speaker, audioZh, Ask("Hello"), out _));
        var interleavedEn = new SpeechReplyOptions { Mode = SpeechReplyMode.Interleaved, Language = "en" };
        Assert.Equal(SpeechReplyPath.Direct, ChatGenerationPipeline.SpokenReplyRoute(speaker, interleavedEn, Ask("你好"), out _));
        // TTS, text replies and a model without its speech decoder are never rerouted.
        Assert.Equal(SpeechReplyPath.Direct, ChatGenerationPipeline.SpokenReplyRoute(speaker,
            new SpeechReplyOptions { Mode = SpeechReplyMode.Text, Synthesis = true }, Ask("你好"), out _));
        Assert.Equal(SpeechReplyPath.Direct, ChatGenerationPipeline.SpokenReplyRoute(new LanguageSpeaker { Loaded = false }, web,
            Ask("你好"), out _));
        Assert.Equal(SpeechReplyPath.Direct, ChatGenerationPipeline.SpokenReplyRoute(null, web, Ask("你好"), out _));
    }

    private sealed class LanguageSpeaker : ISpeechOutputModel
    {
        public bool Loaded { get; init; } = true;
        public bool HasSpeechOutput => Loaded;
        public string SpeechOutputUnavailableReason => null;
        public IReadOnlyList<string> SpeechVoices => Lfm3SpeechPrompt.Voices;
        public int SpeechSampleRate => 24000;
        public bool IsSpeechVoice(string voice) => Lfm3SpeechPrompt.IsVoice(voice);
        public List<int> SpeechSynthesisPrompt(string text, string language, string voice) => new();
        public string SpeechReplyInstruction(SpeechReplyOptions options) => "";
        public int SpeechOpeningToken(SpeechReplyMode mode) => -1;
        public SpeechReplyDriver CreateSpeechDriver(IReadOnlyList<int> promptTokens, SpeechReplyOptions options) => throw new NotSupportedException();
        public string SpeechLanguageOf(string text, string preferred) => Lfm3SpeechPrompt.SpeechLanguageOf(text, preferred);
        public SpeechReplyPath SpeechReplyPathFor(string language)
            => Lfm3SpeechPrompt.SpeaksRepliesIn(language) ? SpeechReplyPath.Direct : SpeechReplyPath.Synthesized;
    }

    [Fact]
    public void AModelThatSaysNothingSpeaksEveryLanguageDirectly()
    {
        ISpeechOutputModel plain = new PlainSpeaker();
        Assert.Equal(SpeechReplyPath.Direct, plain.SpeechReplyPathFor("zh"));
        Assert.Equal("fr", plain.SpeechLanguageOf("北京", "fr"));
    }

    private sealed class PlainSpeaker : ISpeechOutputModel
    {
        public bool HasSpeechOutput => true;
        public string SpeechOutputUnavailableReason => null;
        public IReadOnlyList<string> SpeechVoices => new[] { "a" };
        public int SpeechSampleRate => 24000;
        public bool IsSpeechVoice(string voice) => voice == "a";
        public List<int> SpeechSynthesisPrompt(string text, string language, string voice) => new();
        public string SpeechReplyInstruction(SpeechReplyOptions options) => "";
        public int SpeechOpeningToken(SpeechReplyMode mode) => -1;
        public SpeechReplyDriver CreateSpeechDriver(IReadOnlyList<int> promptTokens, SpeechReplyOptions options) => throw new NotSupportedException();
    }

    // ------------------------------------------------------------------------------------------------ the spoken stream

    private static readonly string[] ReplyPieces = { "北京是", "中国的首都。", "它有故宫", "和长城。" };

    private static async IAsyncEnumerable<ChatStreamUpdate> TextReply(string[] pieces, [EnumeratorCancellation] CancellationToken ct,
        Exception failFirst = null, TaskCompletionSource gate = null)
    {
        if (failFirst != null) throw failFirst;
        for (int i = 0; i < pieces.Length; i++)
        {
            if (i == 3 && gate != null) await gate.Task.WaitAsync(ct);
            await Task.Yield();
            ct.ThrowIfCancellationRequested();
            yield return ChatStreamUpdate.Text(pieces[i]);
        }
        yield return new ChatStreamUpdate("", true, 21, 9, 4, 1, 2, 3, "eos") { RawOutputTokens = new[] { 5, 6, 7 } };
    }

    private sealed class Tts
    {
        public readonly List<(string Text, int Index)> Calls = new();
        public Exception FailOn;
        public TaskCompletionSource Hold;
        public bool SawCancellation;

        public async IAsyncEnumerable<ChatStreamUpdate> Speak(string text, int index, [EnumeratorCancellation] CancellationToken ct)
        {
            lock (Calls) Calls.Add((text, index));
            if (FailOn != null) throw FailOn;
            for (int f = 0; f < 2; f++)
            {
                if (Hold != null)
                {
                    try { await Hold.Task.WaitAsync(ct); }
                    catch (OperationCanceledException) { SawCancellation = true; throw; }
                }
                await Task.Yield();
                yield return ChatStreamUpdate.AudioChunk(new float[] { index, f }, 24000);
            }
            yield return new ChatStreamUpdate("", true, 30, 2, 0, 0, 0, 0, "eos")
            {
                SpeechFrames = new[] { new[] { index, 0 }, new[] { index, 1 } }, AudioSampleRate = 24000,
            };
        }
    }

    private static async Task<List<ChatStreamUpdate>> Drain(IAsyncEnumerable<ChatStreamUpdate> stream)
    {
        var all = new List<ChatStreamUpdate>();
        await foreach (ChatStreamUpdate u in stream) all.Add(u);
        return all;
    }

    [Fact]
    public async Task ATextReplyIsSpokenSentenceBySentenceAsItIsGenerated()
    {
        var tts = new Tts();
        var stats = new ChatGenerationPipeline.SynthesizedReplyStats();
        // The last text piece waits until the first sentence has been spoken: speech starts before the text is done (the
        // sentence is cut once the next piece shows that no closing quote follows its end).
        var gate = new TaskCompletionSource(TaskCreationOptions.RunContinuationsAsynchronously);
        var spokenFirst = new List<ChatStreamUpdate>();
        List<ChatStreamUpdate> all = new();
        await foreach (ChatStreamUpdate u in ChatGenerationPipeline.SpeakTextReplyAsync(ct => TextReply(ReplyPieces, ct, gate: gate), tts.Speak,
                           new SpeechTextSegmenter(firstPieceMinChars: 0), stats, CancellationToken.None))
        {
            all.Add(u);
            if (u.Audio != null && all.Count(x => x.Audio != null) == 2) gate.TrySetResult();
        }

        Assert.Equal(new[] { ("北京是中国的首都。", 0), ("它有故宫和长城。", 1) }, tts.Calls.ToArray());
        Assert.Equal(string.Concat(ReplyPieces), string.Concat(all.Where(u => !u.Done).Select(u => u.Piece)));
        // The first sentence's audio came before the rest of the text.
        int firstAudio = all.FindIndex(u => u.Audio != null);
        int lastText = all.FindIndex(u => u.Piece == ReplyPieces[3]);
        Assert.True(firstAudio < lastText, $"audio at {firstAudio}, last text piece at {lastText}");
        // Each piece's audio in order.
        Assert.Equal(new[] { 0f, 0f, 1f, 1f }, all.Where(u => u.Audio != null).Select(u => u.Audio[0]).ToArray());

        ChatStreamUpdate terminal = all[^1];
        Assert.True(terminal.Done);
        Assert.Equal((21, 9, 4, "eos"), (terminal.PromptTokens, terminal.EvalTokens, terminal.KvCacheReusedTokens, terminal.FinishReason));
        Assert.Equal(new[] { 5, 6, 7 }, terminal.RawOutputTokens);
        Assert.Equal(24000, terminal.AudioSampleRate);
        Assert.Equal(new[] { new[] { 0, 0 }, new[] { 0, 1 }, new[] { 1, 0 }, new[] { 1, 1 } }, terminal.SpeechFrames);
        Assert.Null(terminal.SpokenTurn);
        Assert.Equal(2, stats.Pieces);
        Assert.Equal(8, stats.Samples);
        Assert.True(stats.FirstAudioMs >= 0);
    }

    [Fact]
    public async Task ARefusalOfTheTextReplyIsWhatTheStreamThrows()
    {
        var tts = new Tts();
        var ex = await Assert.ThrowsAsync<NotSupportedException>(() => Drain(ChatGenerationPipeline.SpeakTextReplyAsync(
            ct => TextReply(ReplyPieces, ct, failFirst: new NotSupportedException("no speech decoder")), tts.Speak,
            new SpeechTextSegmenter(), null, CancellationToken.None)));
        Assert.Equal("no speech decoder", ex.Message);
        Assert.Empty(tts.Calls);
    }

    [Fact]
    public async Task ASpeechFailureEndsTheStreamWithIt()
    {
        var tts = new Tts { FailOn = new InvalidOperationException("engine error") };
        var ex = await Assert.ThrowsAsync<InvalidOperationException>(() => Drain(ChatGenerationPipeline.SpeakTextReplyAsync(
            ct => TextReply(ReplyPieces, ct), tts.Speak, new SpeechTextSegmenter(), null, CancellationToken.None)));
        Assert.Equal("engine error", ex.Message);
    }

    [Fact]
    public async Task AConsumerThatStopsReadingStopsTheSpeech()
    {
        var tts = new Tts { Hold = new TaskCompletionSource(TaskCreationOptions.RunContinuationsAsynchronously) };
        await using (IAsyncEnumerator<ChatStreamUpdate> e = ChatGenerationPipeline.SpeakTextReplyAsync(ct => TextReply(ReplyPieces, ct), tts.Speak,
                         new SpeechTextSegmenter(firstPieceMinChars: 0), null, CancellationToken.None).GetAsyncEnumerator())
        {
            Assert.True(await e.MoveNextAsync());
            Assert.Equal(ReplyPieces[0], e.Current.Piece);
            // Wait until the first sentence's speech has started (and is held), then walk away.
            for (int i = 0; i < 200 && tts.Calls.Count == 0; i++) await Task.Delay(10);
            Assert.Single(tts.Calls);
        }
        Assert.True(tts.SawCancellation);
    }

    [Fact]
    public async Task ACancelledRequestThrowsItsCancellation()
    {
        using var cts = new CancellationTokenSource();
        var tts = new Tts { Hold = new TaskCompletionSource(TaskCreationOptions.RunContinuationsAsynchronously) };
        var task = Drain(ChatGenerationPipeline.SpeakTextReplyAsync(ct => TextReply(ReplyPieces, ct), tts.Speak,
            new SpeechTextSegmenter(firstPieceMinChars: 0), null, cts.Token));
        for (int i = 0; i < 200 && tts.Calls.Count == 0; i++) await Task.Delay(10);
        cts.Cancel();
        await Assert.ThrowsAnyAsync<OperationCanceledException>(() => task);
    }
}
