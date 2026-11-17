// Copyright (c) Zhongkai Fu. All rights reserved.
// https://github.com/zhongkaifu/TensorSharp
//
// This file is part of TensorSharp.
//
// TensorSharp is licensed under the BSD-3-Clause license found in the LICENSE file in the root directory of this source tree.
using System;
using System.Collections.Generic;

namespace TensorSharp.Models.Lfm3
{
    /// <summary>What an LFM3-Omni response emits: text tokens, speech frames, or blocks of both.</summary>
    public enum Lfm3SpeechMode
    {
        /// <summary>Text tokens (the model may open speech with &lt;|audio_start|&gt; or &lt;|mixed_start|&gt;).</summary>
        Text,

        /// <summary>Speech frames until the end-of-audio frame (opened by &lt;|audio_start|&gt;).</summary>
        Audio,

        /// <summary>Interleaved blocks: 6 text tokens, then 12 frames, ... until &lt;|text_end|&gt;, then frames until the
        /// end-of-audio frame (opened by &lt;|mixed_start|&gt;).</summary>
        Interleaved,
    }

    /// <summary>The control tokens of LFM3-Omni's speech schedule (ids from the LM's vocabulary).</summary>
    public sealed class Lfm3SpeechTokens
    {
        public int AudioStart { get; init; } = -1;
        public int MixedStart { get; init; } = -1;
        public int TextEnd { get; init; } = -1;
        public int ImEnd { get; init; } = -1;

        /// <summary>Media placeholders HF never samples as text (&lt;|audio|&gt;, &lt;|audio_out|&gt;, &lt;|image|&gt;,
        /// &lt;|video|&gt;); -1 where the vocabulary has none.</summary>
        public int[] Placeholders { get; init; } = Array.Empty<int>();

        /// <summary>The tokens by their text, from <paramref name="lookup"/> (the special token's id, or -1). Throws when a
        /// control token is missing (llama.cpp refuses such a vocabulary too).</summary>
        public static Lfm3SpeechTokens FromVocabulary(Func<string, int> lookup)
        {
            ArgumentNullException.ThrowIfNull(lookup);
            int Required(string text)
            {
                int id = lookup(text);
                if (id < 0) throw new InvalidOperationException($"the vocabulary has no {text} (LFM3 speech control token)");
                return id;
            }
            var placeholders = new List<int>();
            foreach (string p in new[] { "<|audio|>", "<|audio_out|>", "<|image|>", "<|video|>" })
            {
                int id = lookup(p);
                if (id >= 0) placeholders.Add(id);
            }
            return new Lfm3SpeechTokens
            {
                AudioStart = Required("<|audio_start|>"),
                MixedStart = Required("<|mixed_start|>"),
                TextEnd = Required("<|text_end|>"),
                ImEnd = Required("<|im_end|>"),
                Placeholders = placeholders.ToArray(),
            };
        }
    }

    /// <summary>What the caller does after a slot (<see cref="Lfm3SpeechTurn.OnText"/>, <see cref="Lfm3SpeechTurn.OnFrame"/>).</summary>
    public readonly struct Lfm3SlotAction
    {
        public Lfm3SlotAction(bool stop, bool decodeInput, bool decodeWaveform, bool flushCodec)
        {
            Stop = stop;
            DecodeInput = decodeInput;
            DecodeWaveform = decodeWaveform;
            FlushCodec = flushCodec;
        }

        /// <summary>The response ends; nothing of this slot is decoded.</summary>
        public bool Stop { get; }

        /// <summary>Decode the slot's input into the sequence: the sampled token, or the frame's feedback row.</summary>
        public bool DecodeInput { get; }

        /// <summary>The frame goes to the codec stream (a non-end frame).</summary>
        public bool DecodeWaveform { get; }

        /// <summary>Flush the codec stream (its tail closes the speech segment) and reset it: an end-of-audio frame of a
        /// continued response, which then goes on in text.</summary>
        public bool FlushCodec { get; }
    }

    /// <summary>
    /// The slot schedule of one LFM3-Omni response, exactly as llama.cpp runs it (mtmd-helper-gen.cpp
    /// lfm3a_gen_audio_pipeline::step_gen; HF generate's mode handling):
    /// <list type="bullet">
    /// <item>The prompt's last token picks the start: &lt;|audio_start|&gt; speech, &lt;|mixed_start|&gt; interleaved, anything
    /// else text.</item>
    /// <item>Text slot: &lt;|im_end|&gt; stops; outside an interleaved block &lt;|text_end|&gt; stops too, &lt;|audio_start|&gt;
    /// switches to speech and &lt;|mixed_start|&gt; opens an interleaved block (6 text slots first); inside one,
    /// &lt;|text_end|&gt; ends its text (frames follow until the end-of-audio frame). A switching token is decoded like any
    /// other.</item>
    /// <item>Audio slot: a frame. The end-of-audio frame stops a plain TTS response (not decoded); a continued response
    /// (a conversation) flushes the codec, decodes the frame's row and goes on in text.</item>
    /// <item>Interleaving: every slot (not the opening token) counts the block down; at 0 the block flips (text 6, audio 12)
    /// until &lt;|text_end|&gt;.</item>
    /// </list>
    /// Text-only responses of a conversation get <see cref="TextBias"/>: the two switching tokens banned.
    /// </summary>
    public sealed class Lfm3SpeechTurn
    {
        private readonly Lfm3SpeechTokens _tokens;
        private int _slotsLeft;
        private bool _speaking, _textDone;

        /// <param name="tokens">The control tokens.</param>
        /// <param name="lastPromptToken">The prompt's last token (it picks the starting mode).</param>
        /// <param name="continuing">The response continues a conversation (llama.cpp n_past &gt; 0): the end-of-audio frame
        /// hands back to text instead of ending it.</param>
        public Lfm3SpeechTurn(Lfm3SpeechTokens tokens, int lastPromptToken, bool continuing)
        {
            _tokens = tokens ?? throw new ArgumentNullException(nameof(tokens));
            Continuing = continuing;
            Mode = lastPromptToken == tokens.AudioStart ? Lfm3SpeechMode.Audio
                : lastPromptToken == tokens.MixedStart ? Lfm3SpeechMode.Interleaved
                : Lfm3SpeechMode.Text;
            _slotsLeft = Lfm3SpeechConfig.InterleavedTextSlots;
            if (!continuing && Mode != Lfm3SpeechMode.Audio)
                throw new ArgumentException("a TTS response's prompt ends with <|audio_start|> (the model's \"tts\" chat template)", nameof(lastPromptToken));
        }

        public bool Continuing { get; }

        public Lfm3SpeechMode Mode { get; private set; }

        /// <summary>The next slot samples a text token from the LM's logits (else a frame from its hidden row; the LM head can
        /// be skipped for that step).</summary>
        public bool NextIsText => Mode == Lfm3SpeechMode.Text || (Mode == Lfm3SpeechMode.Interleaved && !_speaking);

        /// <summary>Inside an interleaved block: the current block is speech.</summary>
        public bool Speaking => _speaking;

        /// <summary>The token to append to a continued response's prompt for a requested mode (null: none, the model chooses
        /// or the response is text-only): &lt;|audio_start|&gt; for speech, &lt;|mixed_start|&gt; for interleaved.</summary>
        public static int? OpeningToken(Lfm3SpeechTokens tokens, Lfm3SpeechMode? requested)
            => requested switch
            {
                Lfm3SpeechMode.Audio => tokens.AudioStart,
                Lfm3SpeechMode.Interleaved => tokens.MixedStart,
                _ => null,
            };

        /// <summary>The logit bias of a text-only continued response: the switching tokens banned (llama.cpp's TEXT
        /// out_mode).</summary>
        public static IReadOnlyDictionary<int, float> TextBias(Lfm3SpeechTokens tokens) => new Dictionary<int, float>
        {
            [tokens.AudioStart] = float.NegativeInfinity,
            [tokens.MixedStart] = float.NegativeInfinity,
        };

        /// <summary>A text slot sampled <paramref name="token"/>.</summary>
        public Lfm3SlotAction OnText(int token)
        {
            if (!NextIsText) throw new InvalidOperationException("the slot is a speech frame");
            bool interleaving = Mode == Lfm3SpeechMode.Interleaved;
            if (token == _tokens.ImEnd || (!interleaving && token == _tokens.TextEnd)) return new Lfm3SlotAction(true, false, false, false);
            if (!interleaving)
            {
                if (token == _tokens.AudioStart)
                {
                    Mode = Lfm3SpeechMode.Audio;
                }
                else if (token == _tokens.MixedStart)
                {
                    Mode = Lfm3SpeechMode.Interleaved;
                    _slotsLeft = Lfm3SpeechConfig.InterleavedTextSlots;
                    _speaking = false;
                    _textDone = false;
                }
            }
            else if (token == _tokens.TextEnd)
            {
                _textDone = true;
                _speaking = true;
            }
            CountDown(interleaving);
            return new Lfm3SlotAction(false, true, false, false);
        }

        /// <summary>An audio slot sampled a frame (<paramref name="endOfAudio"/>: the end-of-audio frame).</summary>
        public Lfm3SlotAction OnFrame(bool endOfAudio)
        {
            if (NextIsText) throw new InvalidOperationException("the slot is a text token");
            bool interleaving = Mode == Lfm3SpeechMode.Interleaved;
            Lfm3SlotAction action;
            if (endOfAudio)
            {
                if (!Continuing) return new Lfm3SlotAction(true, false, false, false);
                Mode = Lfm3SpeechMode.Text;
                action = new Lfm3SlotAction(false, true, false, true);
            }
            else
            {
                action = new Lfm3SlotAction(false, true, true, false);
            }
            CountDown(interleaving);
            return action;
        }

        private void CountDown(bool interleaving)
        {
            if (interleaving && !_textDone && --_slotsLeft == 0)
            {
                _speaking = !_speaking;
                _slotsLeft = _speaking ? Lfm3SpeechConfig.InterleavedAudioSlots : Lfm3SpeechConfig.InterleavedTextSlots;
            }
        }
    }
}
