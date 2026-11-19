// Copyright (c) Zhongkai Fu. All rights reserved.
// https://github.com/zhongkaifu/TensorSharp
//
// This file is part of TensorSharp.
//
// TensorSharp is licensed under the BSD-3-Clause license found in the LICENSE file in the root directory of this source tree.
using System;
using System.Collections.Generic;
using System.IO;
using System.Security.Cryptography;
using System.Threading;

using TensorSharp.Models.Audio;
using TensorSharp.Models.Lfm3;
using TensorSharp.Models.Video;
using TensorSharp.Runtime;
using TensorSharp.Runtime.Scheduling;

namespace TensorSharp.Models
{
    /// <summary>
    /// LFM3-Omni's speech output on the shared runtime: the depthformer and LiquidCodec decoder of the mmproj
    /// (<see cref="Lfm3SpeechOutput"/>), and the replies that use them.
    /// <list type="bullet">
    /// <item>A spoken reply is one engine request driven by an <see cref="Lfm3SpeechDriver"/>: its text slots are tokens, its
    /// speech slots frames made from the previous position's final hidden row and decoded back as embedding rows
    /// (<see cref="IUnifiedBatchModel.TryForwardUnifiedRows"/>), so spoken and text replies batch in the same steps.</item>
    /// <item><see cref="ISpeechSynthesisModel"/> (TTS: the GGUF's "tts" template, ending at the end of the speech) runs on the
    /// engine where a host has one, and alone on a slot of its own (<see cref="StandardDecoderModel.RunDrivenRequest"/>)
    /// for a caller without one.</item>
    /// <item>Every reply of an LFM3-Omni checkpoint (its LM GGUF carries <c>tokenizer.chat_template.tts</c>) has the media
    /// placeholders suppressed in its text slots, and a text-only reply also the two speech switches
    /// (<see cref="IReplyTokenPolicy"/>), as HF's generate does.</item>
    /// </list>
    /// </summary>
    public partial class Lfm3RuntimeModel : ISpeechOutputModel, ISpeechSynthesisModel, IReplyTokenPolicy, ISpeechOutputLoader
    {
        /// <summary>llama.cpp's default frame limit of a TTS reply (40.96 s).</summary>
        public const int DefaultSpeechFrames = 512;

        /// <summary>Codec streams kept for the next replies (each holds its compiled programs and device state).</summary>
        private const int PooledCodecStreams = 4;

        private Lfm3SpeechOutput _speech;
        private string _speechSource;
        private string _speechUnavailable;
        private readonly Stack<Lfm3CodecStream> _codecPool = new();
        private Lfm3SpeechTokens _speechTokens;
        private volatile bool _speechTokensResolved;
        private readonly object _speechTokensGate = new();
        private int[] _textReplySuppressed, _speechReplySuppressed;

        /// <summary>The speech output (null until an mmproj with one is loaded).</summary>
        public Lfm3SpeechOutput SpeechOutput => _speech;

        /// <summary>The LM GGUF is a checkpoint that speaks (it carries the "tts" chat template: LFM3-Omni, LFM3-Audio).</summary>
        public bool HasSpeechVocabulary => SpeechTokens != null;

        /// <summary>The speech control tokens, or null where the checkpoint has no "tts" template or lacks one of them.</summary>
        internal Lfm3SpeechTokens SpeechTokens
        {
            get
            {
                // Resolved once, under a lock and published after the tokens: requests are submitted from several threads
                // (InferenceEngine.SubmitRequest asks for the reply-token policy), and a caller that saw the flag before
                // the tokens were set had read none - its text reply could draw a media placeholder, and its sampler
                // suppressed nothing where its batch's others did.
                if (_speechTokensResolved) return _speechTokens;
                lock (_speechTokensGate)
                {
                    if (_speechTokensResolved) return _speechTokens;
                    Lfm3SpeechTokens tokens = null;
                    if (!string.IsNullOrEmpty(_gguf.GetString("tokenizer.chat_template.tts")) && Tokenizer != null)
                    {
                        try
                        {
                            tokens = Lfm3SpeechTokens.FromVocabulary(text => Tokenizer.LookupToken(text));
                        }
                        catch (InvalidOperationException ex)
                        {
                            Console.Error.WriteLine($"  {Config.Architecture}: the vocabulary has a \"tts\" template but {ex.Message}: speech output is off.");
                        }
                    }
                    _speechTokens = tokens;
                    _speechTokensResolved = true;
                }
                return _speechTokens;
            }
        }

        // ---------------------------------------------------------------- loading

        /// <summary>Load the speech output of an mmproj (the depthformer and codec, a.gen.*). An mmproj without one (a vision- or
        /// audio-input-only projector) leaves speech output off and says why when asked; one whose speech decoder does not fit
        /// this LM is refused.</summary>
        public void LoadSpeechOutput(string mmprojPath)
        {
            if (string.IsNullOrWhiteSpace(mmprojPath) || !File.Exists(mmprojPath)) return;
            using var gguf = new GgufFile(mmprojPath);
            if (!Lfm3SpeechConfig.IsPresent(gguf))
            {
                _speechUnavailable = $"the projector {Path.GetFileName(mmprojPath)} has no speech decoder (no a.gen.code.* / a.gen.wav.* tensors)";
                return;
            }
            if (SpeechTokens == null)
                throw new InvalidDataException($"{Config.Architecture}: {Path.GetFileName(mmprojPath)} carries a speech decoder, but this LM's " +
                    "vocabulary has no speech control tokens (<|audio_start|>, <|mixed_start|>, <|text_end|>, <|audio_out|>) or no \"tts\" " +
                    "template: load the LFM3-Omni LM GGUF with it.");
            if (SpeechTokens.AudioOut < 0)
                throw new InvalidDataException($"{Config.Architecture}: the vocabulary has no <|audio_out|> token, which speech frames are recorded as.");
            var config = Lfm3SpeechConfig.FromGguf(gguf);
            if (config.HiddenSize != Config.HiddenSize)
                throw new InvalidDataException($"{Path.GetFileName(mmprojPath)}: its speech decoder reads {config.HiddenSize}-wide hidden rows, this LM's are {Config.HiddenSize}.");
            DisposeSpeechOutput();
            var clock = System.Diagnostics.Stopwatch.StartNew();
            _speech = Lfm3SpeechOutput.Load(gguf, _allocator, _backend);
            _speechSource = Path.GetFileName(mmprojPath);
            _speechUnavailable = null;
            Console.WriteLine($"  Speech output: depthformer ({config.DepthLayers} layers) + LiquidCodec ({config.CodecLayers} layers) from {_speechSource}, " +
                $"{(_speech.Weights.QuantizedBytes + _speech.Weights.DenseBytes) / (1024.0 * 1024):F0} MB of weights + {_speech.HostBytes / (1024.0 * 1024):F0} MB " +
                $"of host rows, voices {string.Join("/", Lfm3SpeechPrompt.Voices)}, {config.SampleRate} Hz, in {clock.Elapsed.TotalSeconds:F1} s");
        }

        /// <summary>Tests: a speech output over synthetic networks (its hidden width this LM's) and control tokens of a
        /// synthetic vocabulary.</summary>
        internal void AttachSpeechOutputForTests(Lfm3SpeechOutput speech, Lfm3SpeechTokens tokens)
        {
            DisposeSpeechOutput();
            _speech = speech;
            _speechSource = "tests";
            lock (_speechTokensGate)
            {
                _speechTokens = tokens;
                _speechTokensResolved = true;
                _textReplySuppressed = _speechReplySuppressed = null;
            }
        }

        private void DisposeSpeechOutput()
        {
            lock (GpuComputeLock)
            {
                while (_codecPool.Count > 0) _codecPool.Pop().Dispose();
                _speech?.Dispose();
                _speech = null;
            }
        }

        private Lfm3CodecStream RentCodec()
        {
            lock (_codecPool)
                if (_codecPool.Count > 0) return _codecPool.Pop();
            return _speech.CreateCodecStream();
        }

        private void ReturnCodec(Lfm3CodecStream codec)
        {
            lock (_codecPool)
            {
                if (_speech != null && _codecPool.Count < PooledCodecStreams)
                {
                    _codecPool.Push(codec);
                    return;
                }
            }
            codec.Dispose();
        }

        // ---------------------------------------------------------------- ISpeechOutputModel

        public bool HasSpeechOutput => _speech != null;

        public string SpeechOutputUnavailableReason => _speech != null ? null
            : SpeechTokens == null
                ? $"{Config.Architecture}: this checkpoint does not speak (its GGUF has no \"tts\" template); speech output needs LFM3-Omni with its mmproj"
                : $"{Config.Architecture}: no speech decoder is loaded" + (_speechUnavailable != null ? $" ({_speechUnavailable})" : "") +
                  "; load LFM3-Omni's mmproj (mmproj-LFM3-Omni-2B-*.gguf, whose a.gen.* tensors hold the depthformer and codec) with --mmproj";

        public IReadOnlyList<string> SpeechVoices => Lfm3SpeechPrompt.Voices;

        public int SpeechSampleRate => _speech?.Config.SampleRate ?? 24000;

        public bool IsSpeechVoice(string voice) => Lfm3SpeechPrompt.IsVoice(voice);

        public List<int> SpeechSynthesisPrompt(string text, string language, string voice)
        {
            ArgumentNullException.ThrowIfNull(text);
            // The template's bos_token (<|startoftext|>) as the id, then the rest with special tokens parsed and nothing added
            // (llama.cpp tokenizes common_chat_get_tts_prompt's text so).
            var ids = new List<int>();
            if (Tokenizer.BosTokenId >= 0) ids.Add(Tokenizer.BosTokenId);
            ids.AddRange(Tokenizer.Encode(Lfm3SpeechPrompt.Tts(Lfm3SpeechPrompt.SpokenForm(text, language), language, voice, bosToken: "",
                think: true), addSpecial: false));
            return ids;
        }

        /// <summary>
        /// What a conversation's spoken reply adds to the system prompt: llama.cpp's realtime line naming the voice and
        /// language ("Speak in English using a female voice.") when either is given, and for an interleaved reply the HF card's
        /// "Respond with interleaved text and audio.".
        /// </summary>
        public string SpeechReplyInstruction(SpeechReplyOptions options)
        {
            ArgumentNullException.ThrowIfNull(options);
            string line = Lfm3SpeechPrompt.SpeechSystemLine(options.Language, options.Voice);
            if (options.Mode == SpeechReplyMode.Interleaved)
                line = string.IsNullOrEmpty(line) ? InterleavedInstruction : line + " " + InterleavedInstruction;
            return line;
        }

        /// <summary>The HF card's system prompt for interleaved replies.</summary>
        public const string InterleavedInstruction = "Respond with interleaved text and audio.";

        public int SpeechOpeningToken(SpeechReplyMode mode)
        {
            Lfm3SpeechTokens tokens = SpeechTokens;
            if (tokens == null) return -1;
            return mode switch
            {
                SpeechReplyMode.Audio => tokens.AudioStart,
                SpeechReplyMode.Interleaved => tokens.MixedStart,
                _ => -1,
            };
        }

        public SpeechReplyDriver CreateSpeechDriver(IReadOnlyList<int> promptTokens, SpeechReplyOptions options)
        {
            ArgumentNullException.ThrowIfNull(promptTokens);
            ArgumentNullException.ThrowIfNull(options);
            if (_speech == null) throw new InvalidOperationException(SpeechOutputUnavailableReason);
            if (promptTokens.Count == 0) throw new ArgumentException("an empty prompt", nameof(promptTokens));
            if (options.Voice != null && !IsSpeechVoice(options.Voice))
                throw new ArgumentException(UnknownVoiceMessage(options.Voice));
            Lfm3SpeechSampling sampling = Lfm3SpeechSampling.Defaults(_speech.Config);
            if (options.Temperature is float t)
            {
                if (!(t >= 0f) || float.IsInfinity(t)) throw new ArgumentOutOfRangeException(nameof(options), $"speech temperature {t}: it is 0 (argmax) or more.");
                sampling.Temperature = t;
            }
            uint seed = options.Seed ?? RandomSeed();
            return new Lfm3SpeechDriver(_speech, SpeechTokens, promptTokens[promptTokens.Count - 1], continuing: !options.Synthesis, seed, sampling,
                RentCodec, ReturnCodec, GpuComputeLock);
        }

        /// <summary>A random 32-bit seed (llama.cpp seeds from random_device when none is given).</summary>
        private static uint RandomSeed()
        {
            Span<byte> b = stackalloc byte[4];
            RandomNumberGenerator.Fill(b);
            return BitConverter.ToUInt32(b);
        }

        /// <summary>The refusal of an unknown voice, listing the voices (an OpenAI client sends "alloy" by default).</summary>
        public static string UnknownVoiceMessage(string voice)
            => $"voice '{voice}' is not one of this model's voices ({string.Join(", ", Lfm3SpeechPrompt.Voices)}; case does not matter). " +
               "OpenAI clients default to 'alloy': name one of these instead.";

        // ---------------------------------------------------------------- IReplyTokenPolicy

        public IReadOnlyList<int> SuppressedReplyTokens(bool speechReply)
        {
            Lfm3SpeechTokens tokens = SpeechTokens;
            if (tokens == null) return Array.Empty<int>();
            lock (_speechTokensGate)
            {
                // One array of each kind for the model: a scheduler step of requests that suppress the same tokens takes
                // the device's greedy pick over one mask.
                if (_textReplySuppressed == null)
                {
                    _speechReplySuppressed = (int[])tokens.Placeholders.Clone();
                    var text = new List<int>(tokens.Placeholders) { tokens.AudioStart, tokens.MixedStart };
                    _textReplySuppressed = text.ToArray();
                }
                return speechReply ? _speechReplySuppressed : _textReplySuppressed;
            }
        }

        // ---------------------------------------------------------------- ISpeechSynthesisModel

        public string SpeechModelFamily => Config.Architecture;

        IReadOnlyList<string> ISpeechSynthesisModel.Voices => Lfm3SpeechPrompt.Voices;

        public int OutputSampleRate => SpeechSampleRate;

        public bool SpeechSynthesisAvailable => _speech != null;

        public string SpeechSynthesisUnavailableReason => SpeechOutputUnavailableReason;

        public bool SupportsSpeed => false;

        public bool SamplesSpeech => true;

        public DecodedAudio Synthesize(string text, SpeechSynthesisOptions options = null) => SynthesizeStream(text, options, null);

        /// <summary>
        /// Speak <paramref name="text"/> with the "tts" template (llama-tts): the reply opens in speech and ends at its
        /// end-of-speech frame (or <see cref="SpeechSynthesisOptions.MaxFrames"/>, default 512). Runs on a slot of its own, each
        /// step under the model's compute lock (a host with an engine submits the same driven request to it instead, where it
        /// batches). <paramref name="onSamples"/> gets each frame's PCM as it is decoded.
        /// </summary>
        public DecodedAudio SynthesizeStream(string text, SpeechSynthesisOptions options, Action<float[]> onSamples,
            CancellationToken cancellationToken = default)
        {
            if (string.IsNullOrWhiteSpace(text)) throw new ArgumentException("speech synthesis needs text", nameof(text));
            options ??= new SpeechSynthesisOptions();
            if (_speech == null) throw new InvalidOperationException(SpeechOutputUnavailableReason);
            if (!string.IsNullOrEmpty(options.Phonemes))
                throw new NotSupportedException($"{Config.Architecture} speaks text, not phonemes.");
            if (options.Speed != 1f)
                throw new NotSupportedException($"{Config.Architecture} has no speaking-rate control: speed must be 1 (got {options.Speed}).");
            int maxFrames = options.MaxFrames ?? DefaultSpeechFrames;
            if (maxFrames < 1) throw new ArgumentOutOfRangeException(nameof(options), $"MaxFrames {maxFrames}: at least 1.");
            List<int> prompt = SpeechSynthesisPrompt(text, options.Language, options.Voice);
            var pcm = new List<float>();
            using SpeechReplyDriver driver = CreateSpeechDriver(prompt, new SpeechReplyOptions
            {
                Mode = SpeechReplyMode.Text,
                Voice = options.Voice,
                Language = options.Language,
                Seed = options.Seed,
                Temperature = options.Temperature,
                Synthesis = true,
            });
            void Take()
            {
                float[] samples = driver.TakeAudio();
                if (samples.Length == 0) return;
                pcm.AddRange(samples);
                onSamples?.Invoke(samples);
            }
            // A TTS reply draws no text (it opens in speech and stops at the end-of-speech frame).
            var sampler = new TokenSampler(SamplingConfig.Greedy) { SuppressedTokens = SuppressedReplyTokens(speechReply: true) };
            try
            {
                RunDrivenRequest(prompt, driver, maxFrames, sampler, Take, cancellationToken, out _);
            }
            finally
            {
                driver.Complete();
            }
            Take();
            return new DecodedAudio { Channels = new[] { pcm.ToArray() }, SampleRate = SpeechSampleRate };
        }
    }
}
