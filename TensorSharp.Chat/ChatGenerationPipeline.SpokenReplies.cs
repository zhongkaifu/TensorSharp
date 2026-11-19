// Copyright (c) Zhongkai Fu. All rights reserved.
// https://github.com/zhongkaifu/TensorSharp
//
// This file is part of TensorSharp.
//
// TensorSharp is licensed under the BSD-3-Clause license found in the LICENSE file in the root directory of this source tree.
//
// Spoken replies in a language the model's own speech modes do not speak (LFM3-Omni's interleaved and audio modes speak
// English only; its TTS prompt speaks 19 languages): the reply is generated as an ordinary text reply, and each sentence
// is spoken by the model's TTS prompt as soon as it is complete, on the engine beside the text that is still being
// generated. The text is the speech's transcript, as an interleaved reply's is.

using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Runtime.CompilerServices;
using System.Threading;
using System.Threading.Channels;
using System.Threading.Tasks;
using Microsoft.Extensions.Logging;
using TensorSharp.Models.Audio;
using TensorSharp.Models.Lora;

namespace TensorSharp.Server
{
    internal sealed partial class ChatGenerationPipeline
    {
        /// <summary>
        /// A conversation's spoken reply (<c>/api/chat</c> with <c>speech</c>, chat completions and Responses asking for
        /// audio). Its language is the request's, else the one the user's last message is written in
        /// (<see cref="ISpeechOutputModel.SpeechLanguageOf"/>); where the model speaks that language in the reply itself
        /// (<see cref="SpeechReplyPath.Direct"/>) it is one driven request (text slots and speech frames), else the reply is
        /// generated as text and spoken sentence by sentence by the TTS prompt (<see cref="SpeechReplyPath.Synthesized"/>).
        /// </summary>
        internal IAsyncEnumerable<ChatStreamUpdate> SpokenReplyStreamAsync(ChatSession session, List<ChatMessage> history, int maxTokens,
            CancellationToken cancellationToken, SamplingConfig samplingConfig, bool enableThinking, SpeechReplyOptions speech,
            TextLoraAdapter lora)
        {
            ArgumentNullException.ThrowIfNull(speech);
            var speaker = _lifecycle.Model as ISpeechOutputModel;
            if (SpokenReplyRoute(speaker, speech, history, out string language) == SpeechReplyPath.Synthesized)
                return SynthesizedSpokenReplyAsync(session, history, maxTokens, cancellationToken, samplingConfig, enableThinking,
                    speaker, speech, language, lora);
            return ChatStreamWithMetricsAsync(session, history, maxTokens, cancellationToken, samplingConfig, null, enableThinking,
                new ChatTurnContext { Speech = speech, Adapter = lora });
        }

        /// <summary>How a conversation's spoken reply is made, and in which language (null: the model's default): a model
        /// without speech output (refused later, as before), TTS (<see cref="SpeechReplyOptions.Synthesis"/>) and text replies
        /// speak directly.</summary>
        internal static SpeechReplyPath SpokenReplyRoute(ISpeechOutputModel speaker, SpeechReplyOptions speech,
            IReadOnlyList<ChatMessage> history, out string language)
        {
            language = null;
            if (speaker is not { HasSpeechOutput: true } || speech.Synthesis || speech.Mode == SpeechReplyMode.Text)
                return SpeechReplyPath.Direct;
            language = SpokenReplyLanguage(speaker, speech, history);
            return speaker.SpeechReplyPathFor(language);
        }

        /// <summary>The language a spoken reply speaks: the request's, else what the latest user message with text is written
        /// in (null: the model's default).</summary>
        internal static string SpokenReplyLanguage(ISpeechOutputModel speaker, SpeechReplyOptions speech, IReadOnlyList<ChatMessage> history)
        {
            if (!string.IsNullOrWhiteSpace(speech.Language)) return speech.Language.Trim();
            return speaker.SpeechLanguageOf(LastUserText(history), null);
        }

        /// <summary>The text of the latest user message that has any (a spoken question carries none), or null.</summary>
        internal static string LastUserText(IReadOnlyList<ChatMessage> history)
        {
            if (history == null) return null;
            for (int i = history.Count - 1; i >= 0; i--)
                if (history[i]?.Role == "user" && !string.IsNullOrWhiteSpace(history[i].Content))
                    return history[i].Content;
            return null;
        }

        private async IAsyncEnumerable<ChatStreamUpdate> SynthesizedSpokenReplyAsync(ChatSession session, List<ChatMessage> history,
            int maxTokens, [EnumeratorCancellation] CancellationToken cancellationToken, SamplingConfig samplingConfig, bool enableThinking,
            ISpeechOutputModel speaker, SpeechReplyOptions speech, string language, TextLoraAdapter lora)
        {
            // What a driven reply would refuse, refused before anything is generated (the protocols answer it with a 400).
            if (speech.Voice != null && !speaker.IsSpeechVoice(speech.Voice))
                throw new ArgumentException($"voice '{speech.Voice}' is not one of this model's voices ({string.Join(", ", speaker.SpeechVoices)}; " +
                    "case does not matter).");
            if (speech.Temperature is float t && (!(t >= 0f) || float.IsInfinity(t)))
                throw new ArgumentOutOfRangeException(nameof(speech), $"speech temperature {t}: it is 0 (argmax) or more.");
            var clock = Stopwatch.StartNew();
            var stats = new SynthesizedReplyStats();
            await foreach (ChatStreamUpdate update in SpeakTextReplyAsync(
                ct => ChatStreamWithMetricsAsync(session, history, maxTokens, ct, samplingConfig, null, enableThinking,
                    new ChatTurnContext { Adapter = lora }),
                (piece, index, ct) => SpeechSynthesisStreamAsync(piece, new SpeechSynthesisOptions
                {
                    Voice = speech.Voice,
                    Language = speaker.SpeechLanguageOf(piece, language),
                    Seed = speech.Seed is uint seed ? unchecked(seed + (uint)index) : null,
                    Temperature = speech.Temperature,
                }, ct),
                new SpeechTextSegmenter(), stats, cancellationToken).ConfigureAwait(false))
            {
                if (update.Done)
                {
                    _logger.LogInformation(LogEventIds.ChatCompleted,
                        "speech.reply synthesized language={Language} pieces={Pieces} frames={Frames} audioSeconds={Seconds:F2} " +
                        "firstAudioMs={FirstAudioMs:F0} totalMs={TotalMs:F0} finish={Finish}",
                        language, stats.Pieces, stats.Frames.Count, stats.Samples / (double)Math.Max(1, stats.SampleRate),
                        stats.FirstAudioMs, clock.Elapsed.TotalMilliseconds, update.FinishReason);
                }
                yield return update;
            }
        }

        /// <summary>What a synthesized spoken reply spoke (filled as it runs).</summary>
        internal sealed class SynthesizedReplyStats
        {
            public int Pieces;
            public long Samples;
            public int SampleRate;
            public double FirstAudioMs = -1;
            public readonly List<int[]> Frames = new();
        }

        /// <summary>
        /// A text reply spoken as it is generated: <paramref name="textReply"/>'s updates pass through as they come; its text
        /// goes through <paramref name="segmenter"/>, and each piece it completes is spoken by <paramref name="speak"/> (one
        /// piece at a time, in order, while the text goes on), whose audio updates are passed through between the text ones.
        /// The terminal update is the text reply's, with the speech's frames (every piece's, in order) and sample rate and
        /// the whole request's time; it carries no <see cref="ChatStreamUpdate.SpokenTurn"/>: the reply is a text turn, which a
        /// later request renders (and reuses) as its text. An error of either side ends the stream with it (the text's
        /// first); a consumer that stops reading stops both.
        /// </summary>
        internal static async IAsyncEnumerable<ChatStreamUpdate> SpeakTextReplyAsync(
            Func<CancellationToken, IAsyncEnumerable<ChatStreamUpdate>> textReply,
            Func<string, int, CancellationToken, IAsyncEnumerable<ChatStreamUpdate>> speak,
            SpeechTextSegmenter segmenter, SynthesizedReplyStats stats,
            [EnumeratorCancellation] CancellationToken cancellationToken)
        {
            stats ??= new SynthesizedReplyStats();
            var clock = Stopwatch.StartNew();
            using var linked = CancellationTokenSource.CreateLinkedTokenSource(cancellationToken);
            CancellationToken token = linked.Token;
            var output = Channel.CreateUnbounded<ChatStreamUpdate>(new UnboundedChannelOptions { SingleReader = true });
            var pieces = Channel.CreateUnbounded<string>(new UnboundedChannelOptions { SingleReader = true, SingleWriter = true });
            ChatStreamUpdate terminal = default;
            bool sawTerminal = false;

            Task text = Task.Run(async () =>
            {
                try
                {
                    await foreach (ChatStreamUpdate update in textReply(token).WithCancellation(token).ConfigureAwait(false))
                    {
                        if (update.Done)
                        {
                            terminal = update;
                            sawTerminal = true;
                            continue;
                        }
                        output.Writer.TryWrite(update);
                        if (!string.IsNullOrEmpty(update.Piece))
                            foreach (string piece in segmenter.Append(update.Piece))
                                pieces.Writer.TryWrite(piece);
                    }
                    foreach (string piece in segmenter.Flush())
                        pieces.Writer.TryWrite(piece);
                }
                finally
                {
                    pieces.Writer.TryComplete();
                }
            }, CancellationToken.None);

            Task speech = Task.Run(async () =>
            {
                int index = 0;
                await foreach (string piece in pieces.Reader.ReadAllAsync(token).ConfigureAwait(false))
                {
                    await foreach (ChatStreamUpdate update in speak(piece, index, token).WithCancellation(token).ConfigureAwait(false))
                    {
                        if (update.Done)
                        {
                            if (update.SpeechFrames != null) stats.Frames.AddRange(update.SpeechFrames);
                            if (update.AudioSampleRate > 0) stats.SampleRate = update.AudioSampleRate;
                            continue;
                        }
                        if (update.Audio is { Length: > 0 } samples)
                        {
                            if (stats.FirstAudioMs < 0) stats.FirstAudioMs = clock.Elapsed.TotalMilliseconds;
                            stats.Samples += samples.Length;
                            if (update.AudioSampleRate > 0) stats.SampleRate = update.AudioSampleRate;
                        }
                        output.Writer.TryWrite(update);
                    }
                    stats.Pieces = ++index;
                }
            }, CancellationToken.None);

            // Either side's failure stops the other; the stream ends when both have.
            _ = text.ContinueWith(t => { if (t.IsFaulted || t.IsCanceled) linked.Cancel(); }, TaskScheduler.Default);
            _ = speech.ContinueWith(t => { if (t.IsFaulted || t.IsCanceled) linked.Cancel(); }, TaskScheduler.Default);
            _ = Task.WhenAll(text, speech).ContinueWith(_ => output.Writer.TryComplete(), TaskScheduler.Default);

            bool finished = false;
            try
            {
                await foreach (ChatStreamUpdate update in output.Reader.ReadAllAsync(CancellationToken.None).ConfigureAwait(false))
                    yield return update;
                finished = true;
            }
            finally
            {
                if (!finished)
                {
                    // The consumer stopped reading: stop both requests and let them unwind.
                    linked.Cancel();
                    try { await Task.WhenAll(text, speech).ConfigureAwait(false); } catch { /* the consumer is gone */ }
                }
            }

            // The error that stopped the reply: the speech's when it failed by itself (the text was then cancelled for it),
            // else the text's (a refusal, a cancellation), else the speech's cancellation.
            if (speech.IsFaulted && speech.Exception?.InnerException is not OperationCanceledException)
                await speech.ConfigureAwait(false);
            await text.ConfigureAwait(false);
            await speech.ConfigureAwait(false);
            if (!sawTerminal)
                throw new InvalidOperationException("the text reply ended without its terminal update");
            yield return terminal with
            {
                TotalNs = (long)(clock.Elapsed.TotalSeconds * 1_000_000_000L),
                SpeechFrames = stats.Frames,
                AudioSampleRate = stats.SampleRate,
                SpokenTurn = null,
            };
        }
    }
}
