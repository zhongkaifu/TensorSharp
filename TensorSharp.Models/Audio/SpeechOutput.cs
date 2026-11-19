// Copyright (c) Zhongkai Fu. All rights reserved.
// https://github.com/zhongkaifu/TensorSharp
//
// This file is part of TensorSharp.
//
// TensorSharp is licensed under the BSD-3-Clause license found in the LICENSE file in the root directory of this source tree.
//
// LLM speech output: a language model whose replies can be speech, made by an audio decoder beside the LM (LFM3-Omni's
// depthformer and codec). A reply runs as one engine request whose slots a SpeechReplyDriver chooses
// (TensorSharp.Runtime.Scheduling.ISequenceStepDriver): text tokens, and speech frames made from the LM's hidden rows and
// decoded back into it as embedding rows. The driver queues each frame's PCM; the request's token stream carries the
// driver's frame placeholder once the frame's audio is queued, so a consumer reading the stream in order takes the audio
// exactly where it belongs between the text pieces.
using System;
using System.Collections.Generic;

using TensorSharp.Runtime.Scheduling;

namespace TensorSharp.Models.Audio
{
    /// <summary>What a reply of a model that speaks emits.</summary>
    public enum SpeechReplyMode
    {
        /// <summary>Text only (the model's speech switches are banned).</summary>
        Text,

        /// <summary>Speech only: the reply opens in speech (until the end of the speech, then text, if any, to its end).</summary>
        Audio,

        /// <summary>Text and speech interleaved: the text is the speech's transcript.</summary>
        Interleaved,
    }

    /// <summary>How a conversation's spoken reply in a language is made (<see cref="ISpeechOutputModel.SpeechReplyPathFor"/>).</summary>
    public enum SpeechReplyPath
    {
        /// <summary>The model speaks in the reply itself: its speech modes (<see cref="SpeechReplyMode.Audio"/>,
        /// <see cref="SpeechReplyMode.Interleaved"/>) speak the language.</summary>
        Direct,

        /// <summary>The reply is generated as text, and each sentence is spoken by the model's TTS prompt as it completes:
        /// the model's speech modes do not speak the language, its TTS does (LFM3-Omni's speak English only).</summary>
        Synthesized,
    }

    /// <summary>How one reply speaks.</summary>
    public sealed class SpeechReplyOptions
    {
        /// <summary>The forced mode (<see cref="ISpeechOutputModel.SpeechOpeningToken"/> opens it); <see cref="SpeechReplyMode.Text"/>
        /// lets the prompt decide (a TTS prompt opens speech itself).</summary>
        public SpeechReplyMode Mode { get; init; } = SpeechReplyMode.Interleaved;

        /// <summary>A voice of <see cref="ISpeechOutputModel.SpeechVoices"/>, or null for the default.</summary>
        public string Voice { get; init; }

        /// <summary>The speech's language code, or null.</summary>
        public string Language { get; init; }

        /// <summary>The speech sampler's seed, or null for a random one.</summary>
        public uint? Seed { get; init; }

        /// <summary>The speech temperature, or null for the model's default.</summary>
        public float? Temperature { get; init; }

        /// <summary>Speech synthesis (the prompt is the model's TTS template): the reply ends at the end of its speech. False:
        /// a conversation's reply, which goes on in text after a speech segment until the model ends its turn.</summary>
        public bool Synthesis { get; init; }
    }

    /// <summary>
    /// A language model whose replies can be speech: an audio decoder beside the LM turns its hidden rows into frames of
    /// speech, fed back as embedding rows (LFM3-Omni with its mmproj). Hosts submit a reply as an engine request whose
    /// <see cref="SequenceState.StepDriver"/> is <see cref="CreateSpeechDriver"/>'s driver.
    /// </summary>
    public interface ISpeechOutputModel
    {
        /// <summary>The speech decoder is loaded (else <see cref="SpeechOutputUnavailableReason"/> says what to load).</summary>
        bool HasSpeechOutput { get; }

        /// <summary>Why <see cref="HasSpeechOutput"/> is false, or null.</summary>
        string SpeechOutputUnavailableReason { get; }

        /// <summary>The voices (the default first).</summary>
        IReadOnlyList<string> SpeechVoices { get; }

        /// <summary>The PCM rate of the speech.</summary>
        int SpeechSampleRate { get; }

        /// <summary>Whether <paramref name="voice"/> names one of <see cref="SpeechVoices"/> (as the model matches them).</summary>
        bool IsSpeechVoice(string voice);

        /// <summary>The prompt that speaks <paramref name="text"/> (the model's TTS template), as tokens.</summary>
        List<int> SpeechSynthesisPrompt(string text, string language, string voice);

        /// <summary>The system-prompt text that asks a conversation's reply to speak as <paramref name="options"/> say ("" for
        /// none), which a host appends to the request's system message.</summary>
        string SpeechReplyInstruction(SpeechReplyOptions options);

        /// <summary>The token appended to a conversation's generation prompt to force <paramref name="mode"/>, or -1 (none).</summary>
        int SpeechOpeningToken(SpeechReplyMode mode);

        /// <summary>
        /// The language (a code) to speak <paramref name="text"/> in: <paramref name="preferred"/> where the text's script
        /// suits it, else the language the text's script names (Chinese characters: <c>zh</c>, kana: <c>ja</c>, ...), else
        /// <paramref name="preferred"/>; null when neither says (the model's default). A host asks it for a reply whose request
        /// names no language (of the user's message) and for each piece of a synthesized reply.
        /// </summary>
        string SpeechLanguageOf(string text, string preferred) => preferred;

        /// <summary>How a conversation's reply in <paramref name="language"/> (null: the model's default) speaks.</summary>
        SpeechReplyPath SpeechReplyPathFor(string language) => SpeechReplyPath.Direct;

        /// <summary>A driver for one reply whose prompt is <paramref name="promptTokens"/> (its last token picks the starting
        /// mode). The caller disposes it after the request.</summary>
        SpeechReplyDriver CreateSpeechDriver(IReadOnlyList<int> promptTokens, SpeechReplyOptions options);
    }

    /// <summary>A model that loads its speech decoder from its projector (the mmproj); the host's projector load
    /// (<c>ModelMultimodalInjector.LoadProjectors</c>) calls it beside the vision and audio towers.</summary>
    public interface ISpeechOutputLoader
    {
        void LoadSpeechOutput(string mmprojPath);
    }

    /// <summary>
    /// The step driver of one spoken reply (<see cref="ISequenceStepDriver"/>): it queues each frame's PCM, which a consumer
    /// takes (<see cref="TakeAudio"/>) when the request's token stream carries <see cref="FrameToken"/>, and the codec's tail
    /// when the reply ends (<see cref="Complete"/>).
    /// </summary>
    public abstract class SpeechReplyDriver : ISequenceStepDriver, IOutputMediaDriver
    {
        private readonly object _audioGate = new();
        private List<float> _audio = new();
        private readonly List<int[]> _frames = new();

        /// <summary>The rate of the queued PCM.</summary>
        public abstract int SampleRate { get; }

        /// <summary>The placeholder token the reply's output records for a speech frame (its token stream carries it once the
        /// frame's audio is queued).</summary>
        public abstract int FrameToken { get; }

        /// <summary>Whether <paramref name="token"/> is a control token of the schedule (not transcript text).</summary>
        public abstract bool IsControlToken(int token);

        public abstract bool NextNeedsLogits { get; }

        public abstract bool NextNeedsHidden { get; }

        public abstract DriverSlot Next(float[] logits, float[] hidden, Func<int> drawToken);

        /// <summary>The reply has ended (its request finished, stopped or failed): the speech held back is queued. Idempotent;
        /// call it before the last <see cref="TakeAudio"/>.</summary>
        public abstract void Complete();

        /// <summary>Speech frames sampled (the end-of-speech frame included).</summary>
        public int Frames { get; protected set; }

        /// <summary>Text slots drawn (control tokens included).</summary>
        public int TextSlots { get; protected set; }

        /// <summary>When the first audio was queued (UTC), or null.</summary>
        public DateTime? FirstAudioAt { get; private set; }

        /// <summary>Samples queued so far.</summary>
        public long SamplesQueued { get; private set; }

        /// <summary>The codes of every speech frame decoded (one array a frame, in order): what re-renders the reply.</summary>
        public IReadOnlyList<int[]> FrameCodes
        {
            get
            {
                lock (_audioGate) return _frames.ToArray();
            }
        }

        /// <summary>The PCM queued since the last call (empty when none).</summary>
        public float[] TakeAudio()
        {
            lock (_audioGate)
            {
                if (_audio.Count == 0) return Array.Empty<float>();
                float[] samples = _audio.ToArray();
                _audio = new List<float>();
                return samples;
            }
        }

        /// <summary>Queue decoded PCM (the driver, on the engine thread).</summary>
        protected void QueueAudio(List<float> samples)
        {
            if (samples == null || samples.Count == 0) return;
            lock (_audioGate)
            {
                FirstAudioAt ??= DateTime.UtcNow;
                SamplesQueued += samples.Count;
                _audio.AddRange(samples);
            }
        }

        /// <summary>Record a decoded frame's codes.</summary>
        protected void RecordFrame(int[] codes)
        {
            lock (_audioGate) _frames.Add((int[])codes.Clone());
        }

        private readonly List<int[]> _fed = new();

        /// <summary>The codes of every frame whose row the reply decoded into its sequence (one array a frame, in order; an
        /// end-of-speech frame that handed back to text included), one per <see cref="FrameToken"/> of its output: what a
        /// later turn re-renders the reply's speech from (<c>ChatMessage.SpeechFrames</c>).</summary>
        public IReadOnlyList<int[]> FedFrameCodes
        {
            get
            {
                lock (_audioGate) return _fed.ToArray();
            }
        }

        /// <summary>Record the codes of a frame whose row is decoded into the sequence.</summary>
        protected void RecordFedFrame(int[] codes)
        {
            lock (_audioGate) _fed.Add((int[])codes.Clone());
        }

        /// <summary>The content id the model's prompt expander gives the rows of fed frames [<paramref name="first"/>,
        /// + <paramref name="count"/>) (one run of <see cref="FrameToken"/>s) when a later turn carries this reply back, or
        /// null when it has none: the reply's rows are then not keyed, and its end state is not retained.</summary>
        protected virtual string SpanContentId(IReadOnlyList<int[]> frames, int first, int count) => null;

        /// <summary>The runs of <see cref="FrameToken"/> in the reply's output, each named by its fed frames' content id
        /// (<see cref="SpanContentId"/>), so its end state is keyed as the next turn's prompt keys the reply
        /// (<see cref="IOutputMediaDriver"/>). A run at the end of an output that is not complete may still grow: it is left
        /// out, and the output keyed up to its start.</summary>
        public IReadOnlyList<PromptMediaSpan> OutputMediaSpans(IReadOnlyList<int> output, int outputStart, bool complete,
            out int keyedOutput)
        {
            ArgumentNullException.ThrowIfNull(output);
            keyedOutput = 0;
            IReadOnlyList<int[]> frames = FedFrameCodes;
            int placeholder = FrameToken;
            var spans = new List<PromptMediaSpan>();
            int frame = 0;
            for (int i = 0; i < output.Count;)
            {
                if (output[i] != placeholder)
                {
                    keyedOutput = ++i;
                    continue;
                }
                int start = i;
                while (i < output.Count && output[i] == placeholder) i++;
                if (i == output.Count && !complete) break;
                int count = i - start;
                // One fed frame a placeholder, in order; a run without its frames is not keyed.
                if (frame + count > frames.Count) break;
                string id = SpanContentId(frames, frame, count);
                if (id == null) return null;
                spans.Add(new PromptMediaSpan(outputStart + start, outputStart + i, id));
                frame += count;
                keyedOutput = i;
            }
            return spans;
        }

        public abstract void Dispose();
    }
}
