// Copyright (c) Zhongkai Fu. All rights reserved.
// https://github.com/zhongkaifu/TensorSharp
//
// This file is part of TensorSharp.
//
// TensorSharp is licensed under the BSD-3-Clause license found in the LICENSE file in the root directory of this source tree.
using System;
using System.Collections.Generic;
using System.Threading;

using TensorSharp.Models.Audio;
using TensorSharp.Runtime.Scheduling;

namespace TensorSharp.Models.Lfm3
{
    /// <summary>
    /// One LFM3-Omni reply's slots (llama.cpp mtmd-helper-gen.cpp lfm3a_gen_audio_pipeline::step_gen, HF generate): the
    /// <see cref="Lfm3SpeechTurn"/> schedule decides whether a slot is a text token (drawn from the logits with the request's
    /// sampler) or a speech frame (3 depthformer passes over the previous position's final hidden row); a frame's codes go to
    /// the reply's codec stream, whose PCM is queued, and its feedback row (the sum of its 12 codebook embeddings) is the
    /// next input row. The end-of-speech frame of a TTS reply ends it; in a conversation it flushes the codec and the reply
    /// goes on in text, its row decoded. Every slot the turn says to decode is decoded; the reply ends where the turn stops it
    /// (&lt;|im_end|&gt;, or &lt;|text_end|&gt; outside an interleaved block).
    /// </summary>
    internal sealed class Lfm3SpeechDriver : SpeechReplyDriver
    {
        private readonly Lfm3SpeechOutput _speech;
        private readonly Lfm3SpeechTokens _tokens;
        private readonly Lfm3SpeechTurn _turn;
        private readonly Lfm3FrameSampler _sampler;
        private readonly Func<Lfm3CodecStream> _rentCodec;
        private readonly Action<Lfm3CodecStream> _returnCodec;
        private readonly object _computeLock;
        private readonly object _gate = new();
        private readonly List<float> _pcm = new(2048);
        private Lfm3CodecStream _codec;
        private bool _ended, _completed, _disposed;

        internal Lfm3SpeechDriver(Lfm3SpeechOutput speech, Lfm3SpeechTokens tokens, int lastPromptToken, bool continuing,
            uint seed, Lfm3SpeechSampling sampling, Func<Lfm3CodecStream> rentCodec, Action<Lfm3CodecStream> returnCodec, object computeLock)
        {
            _speech = speech ?? throw new ArgumentNullException(nameof(speech));
            _tokens = tokens ?? throw new ArgumentNullException(nameof(tokens));
            _turn = new Lfm3SpeechTurn(tokens, lastPromptToken, continuing);
            _sampler = speech.CreateSampler(seed, sampling);
            _computeLock = computeLock ?? new object();
            _rentCodec = rentCodec ?? (() => speech.CreateCodecStream());
            // A stream nobody pools frees its device state: between the engine's steps (DisposeBetweenSteps).
            _returnCodec = returnCodec ?? (c => DisposeBetweenSteps(c, _computeLock));
            Seed = seed;
        }

        public uint Seed { get; }

        /// <summary>The schedule's current mode.</summary>
        public Lfm3SpeechMode Mode => _turn.Mode;

        public override int SampleRate => _speech.Config.SampleRate;

        public override int FrameToken => _tokens.AudioOut;

        /// <summary>The id the prompt expander gives a past spoken turn's run of rows (<see cref="Lfm3SpeechOutput.FramesContentId"/>).</summary>
        protected override string SpanContentId(IReadOnlyList<int[]> frames, int first, int count)
            => Lfm3SpeechOutput.FramesContentId(frames, first, count);

        public override bool IsControlToken(int token)
            => token == _tokens.AudioStart || token == _tokens.MixedStart || token == _tokens.TextEnd || token == _tokens.ImEnd
               || token == _tokens.AudioOut;

        public override bool NextNeedsLogits
        {
            get
            {
                lock (_gate) return !_ended && !_disposed && _turn.NextIsText;
            }
        }

        public override bool NextNeedsHidden
        {
            get
            {
                lock (_gate) return !_ended && !_disposed && !_turn.NextIsText;
            }
        }

        public override DriverSlot Next(float[] logits, float[] hidden, Func<int> drawToken)
        {
            lock (_gate)
            {
                if (_disposed) return DriverSlot.Stop("cancelled");
                if (_ended) return DriverSlot.Stop("eos");
                if (_turn.NextIsText)
                {
                    if (logits == null) throw new InvalidOperationException("LFM3 speech: a text slot without the previous position's logits.");
                    int token = drawToken();
                    TextSlots++;
                    Lfm3SlotAction a = _turn.OnText(token);
                    if (a.Stop)
                    {
                        EndLocked();
                        return DriverSlot.Stop("eos");
                    }
                    return DriverSlot.Text(token);
                }
                if (hidden == null) throw new InvalidOperationException("LFM3 speech: a speech slot without the previous position's hidden row.");
                Lfm3SpeechFrame frame = _sampler.Sample(hidden);
                Frames++;
                Lfm3SlotAction action = _turn.OnFrame(frame.IsEndOfAudio);
                if (action.Stop)
                {
                    EndLocked();
                    return DriverSlot.Stop("eos");
                }
                if (action.FlushCodec) FlushLocked();
                if (action.DecodeWaveform)
                {
                    _codec ??= _rentCodec();
                    _pcm.Clear();
                    _codec.Decode(frame.Codes, _pcm);
                    QueueAudio(_pcm);
                    RecordFrame(frame.Codes);
                }
                RecordFedFrame(frame.Codes);
                return DriverSlot.Embedding(_tokens.AudioOut, _speech.FeedbackRow(frame.Codes));
            }
        }

        /// <summary>The codec's held-back tail of the current speech segment is queued and the stream returns to its start.</summary>
        private void FlushLocked()
        {
            if (_codec == null) return;
            _pcm.Clear();
            _codec.Flush(_pcm);
            QueueAudio(_pcm);
        }

        private void EndLocked()
        {
            _ended = true;
            FlushLocked();
        }

        public override void Complete()
        {
            lock (_gate)
            {
                if (_completed || _disposed) return;
                _completed = true;
                if (!_ended) EndLocked();
            }
        }

        public override void Dispose()
        {
            Lfm3CodecStream codec;
            lock (_gate)
            {
                if (_disposed) return;
                _disposed = true;
                codec = _codec;
                _codec = null;
            }
            if (codec == null) return;
            // Reset is host state only (the stream's position, its ISTFT tail): the next request that rents the stream does so
            // inside a step and decodes from its cold start. It used to be done under the GPU lock, and that wait was what
            // stopped a long spoken reply: the engine worker takes the lock for every step a few microseconds after
            // releasing it, and a Monitor hands a released lock to whoever asks next, so this thread sat through the text
            // reply's steps (2.5 s on an M5 Pro, until the text ended on a phone) - and the reply's next sentence, which
            // is spoken only once this one's request has ended, waited with it.
            codec.Reset();
            _returnCodec(codec);
        }

        /// <summary>
        /// Free <paramref name="codec"/>'s device state between the engine's steps: registered as a waiter
        /// (<see cref="GpuLockWaiters"/>), so the worker lets this thread in at its next step boundary instead of taking the
        /// lock again first.
        /// </summary>
        internal static void DisposeBetweenSteps(Lfm3CodecStream codec, object computeLock)
        {
            GpuLockWaiters.EnterBetweenSteps(computeLock);
            try { codec.Dispose(); }
            finally { Monitor.Exit(computeLock); }
        }
    }
}
