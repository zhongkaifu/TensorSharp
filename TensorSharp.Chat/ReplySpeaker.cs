// Copyright (c) Zhongkai Fu. All rights reserved.
// https://github.com/zhongkaifu/TensorSharp
//
// This file is part of TensorSharp.
//
// TensorSharp is licensed under the BSD-3-Clause license found in the LICENSE file in the root directory of this source tree.
//
// The speech layer of a spoken turn that is a text turn: the text the user reads is spoken, piece by piece, by the model's
// TTS prompt while the turn goes on (its rounds, its tool calls, its reasoning, none of which is spoken).

using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Threading;
using System.Threading.Channels;
using System.Threading.Tasks;
using TensorSharp.Models.Audio;

namespace TensorSharp.Server
{
    /// <summary>
    /// Speaks a reply's visible text while it is written. The caller feeds the answer text the user reads as it streams
    /// (<see cref="Say"/>; never reasoning, tool-call markup or tool results), takes back text that turned out not to be
    /// answer (<see cref="Retract"/>), and marks the end of a round's text (<see cref="Pause"/>: a tool call follows, so the
    /// text before it is spoken now rather than when the next round's first sentence completes). A
    /// <see cref="SpeechTextSegmenter"/> cuts the text into pieces and <c>speak</c> (the model's TTS prompt on the engine)
    /// speaks each, one at a time and in order, on a task of its own: the text never waits for the speech. The audio comes
    /// out on <see cref="Audio"/> as it is made; <see cref="Completion"/> ends when every piece has been spoken
    /// (<see cref="Complete"/>), faults with the speech's error, or is cancelled (<see cref="Cancel"/>, the turn's token).
    /// One feeder at a time (the turn's consumer); the speech side is this class's own.
    /// </summary>
    internal sealed class ReplySpeaker : IDisposable
    {
        private readonly Func<string, int, CancellationToken, IAsyncEnumerable<ChatStreamUpdate>> _speak;
        private readonly SpeechTextSegmenter _segmenter;
        private readonly Channel<(string Text, double QueuedMs)> _pieces = Channel.CreateUnbounded<(string, double)>(
            new UnboundedChannelOptions { SingleReader = true, SingleWriter = true });
        private readonly Channel<ChatStreamUpdate> _audio = Channel.CreateUnbounded<ChatStreamUpdate>(
            new UnboundedChannelOptions { SingleReader = false, SingleWriter = true });
        private readonly CancellationTokenSource _cancel;
        private readonly Stopwatch _clock = Stopwatch.StartNew();
        private readonly Func<double> _now;
        private readonly object _gate = new();
        private readonly List<string> _spoken = new();
        private bool _closed;
        private Task<bool> _audioWait;

        /// <param name="speak">Speaks one piece (its text, its index in the reply): audio updates, then a terminal update
        /// with the piece's frames.</param>
        /// <param name="maxLeadSeconds">While the text is still being written, start the next piece only once the speech
        /// made so far is less than this far ahead of its playback (estimated from the first audio on, as a listener hears
        /// it); null speaks every piece as soon as it is cut. Speech and text share the engine, so speech made far ahead of
        /// the listener only slows the text down.</param>
        public ReplySpeaker(Func<string, int, CancellationToken, IAsyncEnumerable<ChatStreamUpdate>> speak,
            SpeechTextSegmenter segmenter, CancellationToken cancellationToken, double? maxLeadSeconds = null)
            : this(speak, segmenter, cancellationToken, maxLeadSeconds, null)
        {
        }

        /// <summary>The cap a web chat's spoken turn uses (WebUiChatService): enough lead to ride out a sentence boundary on
        /// a phone, little enough that the text keeps most of the engine.</summary>
        public const double DefaultMaxLeadSeconds = 3.0;

        /// <param name="clockMs">Test seam: the speaker's clock in milliseconds (a Stopwatch by default).</param>
        internal ReplySpeaker(Func<string, int, CancellationToken, IAsyncEnumerable<ChatStreamUpdate>> speak,
            SpeechTextSegmenter segmenter, CancellationToken cancellationToken, double? maxLeadSeconds, Func<double> clockMs)
        {
            _now = clockMs ?? (() => _clock.Elapsed.TotalMilliseconds);
            _speak = speak ?? throw new ArgumentNullException(nameof(speak));
            _segmenter = segmenter ?? new SpeechTextSegmenter();
            _cancel = CancellationTokenSource.CreateLinkedTokenSource(cancellationToken);
            if (maxLeadSeconds is double cap && !(cap > 0)) throw new ArgumentOutOfRangeException(nameof(maxLeadSeconds), "more than 0");
            MaxLeadSeconds = maxLeadSeconds;
            Completion = Task.Run(SpeakAsync, CancellationToken.None);
        }

        /// <summary>See the constructor: how far ahead of its playback the speech may run while the text is written.</summary>
        public double? MaxLeadSeconds { get; }

        /// <summary>How far the speech made so far is ahead of its playback: seconds of audio made, less the time since the
        /// first audio (when a listener started hearing it). 0 before the first audio.</summary>
        public double LeadSeconds
        {
            get
            {
                if (FirstAudioMs < 0 || SampleRate <= 0) return 0;
                return Samples / (double)SampleRate - (_now() - FirstAudioMs) / 1000.0;
            }
        }

        /// <summary>Time spent holding pieces back for <see cref="MaxLeadSeconds"/>, in milliseconds.</summary>
        public double HeldMs { get; private set; }

        private volatile bool _textDone;

        /// <summary>Every piece spoken (after <see cref="Complete"/>), the speech's error, or its cancellation.</summary>
        public Task Completion { get; }

        /// <summary>The speech's audio updates (<see cref="ChatStreamUpdate.Audio"/>), in the order they were made;
        /// completed when the speech ends, however it ends.</summary>
        public ChannelReader<ChatStreamUpdate> Audio => _audio.Reader;

        /// <summary>The pieces spoken so far, in order (what a listener heard, as text).</summary>
        public IReadOnlyList<string> Spoken { get { lock (_spoken) return _spoken.ToArray(); } }

        /// <summary>Pieces spoken.</summary>
        public int Pieces { get; private set; }

        /// <summary>Samples of speech made.</summary>
        public long Samples { get; private set; }

        /// <summary>The speech's sample rate (0 before its first audio).</summary>
        public int SampleRate { get; private set; }

        /// <summary>Milliseconds from this speaker's start to its first audio, or -1.</summary>
        public double FirstAudioMs { get; private set; } = -1;

        /// <summary>Every spoken piece's frames, in order.</summary>
        public List<int[]> Frames { get; } = new();

        /// <summary>When each piece was cut from the text, given to the engine, first heard from and finished, in
        /// milliseconds from this speaker's start, and how much speech it made: what tells a reply whose speech fell
        /// behind its playback (a piece that started late, or ran slower than realtime) from one that did not.</summary>
        public IReadOnlyList<PieceTiming> Timings { get { lock (_timings) return _timings.ToArray(); } }

        private readonly List<PieceTiming> _timings = new();

        /// <summary>One spoken piece's timeline (<see cref="Timings"/>); -1 where it did not get that far. The gap between a
        /// piece's last audio and the next piece's start (when the next was already queued) is time the speech made nothing.</summary>
        public readonly record struct PieceTiming(int Index, int Chars, double QueuedMs, double StartMs, double FirstAudioMs,
            double LastAudioMs, double EndMs, double AudioSeconds)
        {
            public override string ToString() => FormattableString.Invariant(
                $"#{Index} {Chars}ch queued={QueuedMs:F0} start={StartMs:F0} first={FirstAudioMs:F0} last={LastAudioMs:F0} end={EndMs:F0} audio={AudioSeconds:F2}s");
        }

        /// <summary>Visible answer text, as it streams.</summary>
        public void Say(string text)
        {
            if (string.IsNullOrEmpty(text)) return;
            lock (_gate)
            {
                if (_closed) return;
                foreach (string piece in _segmenter.Append(text)) _pieces.Writer.TryWrite((piece, _now()));
            }
        }

        /// <summary>Answer text taken back (it was reasoning): dropped where it is still waiting to be spoken.</summary>
        public void Retract(string text)
        {
            if (string.IsNullOrEmpty(text)) return;
            lock (_gate)
            {
                if (!_closed) _segmenter.Retract(text);
            }
        }

        /// <summary>The end of a round's text (a tool call follows): what is waiting is spoken now.</summary>
        public void Pause()
        {
            lock (_gate)
            {
                if (_closed) return;
                foreach (string piece in _segmenter.Flush()) _pieces.Writer.TryWrite((piece, _now()));
            }
        }

        /// <summary>No more text: the rest is spoken, then <see cref="Completion"/> ends.</summary>
        public void Complete()
        {
            lock (_gate)
            {
                if (_closed) return;
                _closed = true;
                _textDone = true;
                foreach (string piece in _segmenter.Flush()) _pieces.Writer.TryWrite((piece, _now()));
                _pieces.Writer.TryComplete();
            }
        }

        /// <summary>Stop speaking (a stopped turn, a failed one, a consumer gone): what is waiting is dropped and the
        /// piece being spoken is cancelled on the engine.</summary>
        public void Cancel()
        {
            lock (_gate)
            {
                _closed = true;
                _pieces.Writer.TryComplete();
            }
            try { _cancel.Cancel(); } catch (ObjectDisposedException) { /* already disposed */ }
        }

        /// <summary>A task that completes when audio can be read or the speech has ended (one at a time, so a caller that
        /// waits for it beside something else never piles up waits).</summary>
        public Task WhenAudioOrEnd()
        {
            if (_audioWait == null || _audioWait.IsCompleted)
                _audioWait = _audio.Reader.WaitToReadAsync().AsTask();
            return _audioWait;
        }

        private async Task SpeakAsync()
        {
            try
            {
                CancellationToken token = _cancel.Token;
                int index = 0;
                await foreach ((string piece, double queuedMs) in _pieces.Reader.ReadAllAsync(token).ConfigureAwait(false))
                {
                    // Far enough ahead of the listener while the text is still being written: let the text have the engine
                    // until the lead is down to the cap (the rest of a reply whose text is done is spoken at once).
                    if (MaxLeadSeconds is double cap)
                    {
                        double held = _now();
                        while (!_textDone && LeadSeconds > cap)
                            await Task.Delay(TimeSpan.FromMilliseconds(Math.Clamp((LeadSeconds - cap) * 1000, 20, 250)), token).ConfigureAwait(false);
                        HeldMs += _now() - held;
                    }
                    double startMs = _now(), firstMs = -1, lastMs = -1;
                    long samplesBefore = Samples;
                    await foreach (ChatStreamUpdate update in _speak(piece, index, token).WithCancellation(token).ConfigureAwait(false))
                    {
                        if (update.Done)
                        {
                            if (update.SpeechFrames != null) Frames.AddRange(update.SpeechFrames);
                            if (update.AudioSampleRate > 0) SampleRate = update.AudioSampleRate;
                            continue;
                        }
                        if (update.Audio is not { Length: > 0 } samples) continue;
                        lastMs = _now();
                        if (firstMs < 0) firstMs = lastMs;
                        if (FirstAudioMs < 0) FirstAudioMs = firstMs;
                        Samples += samples.Length;
                        if (update.AudioSampleRate > 0) SampleRate = update.AudioSampleRate;
                        _audio.Writer.TryWrite(update);
                    }
                    lock (_timings)
                        _timings.Add(new PieceTiming(index, piece.Length, queuedMs, startMs, firstMs, lastMs, _now(),
                            (Samples - samplesBefore) / (double)Math.Max(1, SampleRate)));
                    lock (_spoken) _spoken.Add(piece);
                    Pieces = ++index;
                }
            }
            finally
            {
                _audio.Writer.TryComplete();
            }
        }

        public void Dispose()
        {
            Cancel();
            // The speech task holds the token source until it has unwound; it is left to the finalizer-free GC rather than
            // disposed under it.
        }
    }
}
