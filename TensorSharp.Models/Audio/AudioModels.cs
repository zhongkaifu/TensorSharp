// Copyright (c) Zhongkai Fu. All rights reserved.
// https://github.com/zhongkaifu/TensorSharp
//
// This file is part of TensorSharp.
//
// TensorSharp is licensed under the BSD-3-Clause license found in the LICENSE file in the root directory of this source tree.
//
// The audio model families, as the CLI and the server drive them: speech recognition (Whisper),
// speech synthesis (Kokoro) and music generation (ACE-Step). Audio travels as planar float PCM
// (DecodedAudio: one array per channel, at its own sample rate) in both directions; each model
// resamples what it is given and says what it returns.
using System;
using System.Collections.Generic;
using TensorSharp.Models.Video;

namespace TensorSharp.Models.Audio
{
    /// <summary>A speech-to-text model: audio in, a timed transcript out.</summary>
    public interface ISpeechRecognitionModel
    {
        /// <summary>The architecture id, for logs and refusals ("whisper").</summary>
        string SpeechModelFamily { get; }

        /// <summary>Language codes the model accepts for <see cref="TranscriptionOptions.Language"/>.</summary>
        IReadOnlyList<string> Languages { get; }

        /// <summary>Transcribe (or translate to English) a clip of any length, rate and channel count.</summary>
        TranscriptionResult Transcribe(DecodedAudio audio, TranscriptionOptions options = null);
    }

    public sealed class TranscriptionOptions
    {
        /// <summary>ISO 639-1 code ("en", "zh"), or null to detect it from the first 30 seconds.</summary>
        public string Language { get; set; }

        /// <summary>Translate to English instead of transcribing in the spoken language.</summary>
        public bool Translate { get; set; }

        /// <summary>Text that precedes the audio (spelling of names, style); conditions the first window.</summary>
        public string Prompt { get; set; }

        /// <summary>Sampling temperature of the first attempt; 0 decodes greedily.</summary>
        public float Temperature { get; set; }

        /// <summary>Retry a window at higher temperatures (+0.2 up to 1.0) when its text repeats itself
        /// (compression ratio above 2.4) or its average log probability is below -1, as openai/whisper does.</summary>
        public bool TemperatureFallback { get; set; } = true;

        /// <summary>Condition each window on the text of the previous ones.</summary>
        public bool ConditionOnPreviousText { get; set; } = true;

        /// <summary>Also time each word (cross-attention alignment), which costs a second decoder pass per window.</summary>
        public bool WordTimestamps { get; set; }

        /// <summary>Seed of the fallback sampler.</summary>
        public int Seed { get; set; }

        /// <summary>Called with each segment as soon as its window is decoded.</summary>
        public Action<TranscriptionSegment> OnSegment { get; set; }
    }

    public sealed class TranscriptionWord
    {
        public string Word { get; init; }
        public double Start { get; init; }
        public double End { get; init; }
        public double Probability { get; init; }
    }

    public sealed class TranscriptionSegment
    {
        public int Id { get; init; }
        public double Start { get; init; }
        public double End { get; init; }
        public string Text { get; init; }
        public int[] Tokens { get; init; }
        public double Temperature { get; init; }
        public double AvgLogprob { get; init; }
        public double CompressionRatio { get; init; }
        public double NoSpeechProb { get; init; }
        public List<TranscriptionWord> Words { get; init; }
    }

    public sealed class TranscriptionResult
    {
        public string Text { get; init; }
        public string Language { get; init; }
        public double Duration { get; init; }
        public string Task { get; init; }
        public List<TranscriptionSegment> Segments { get; init; } = new();

        /// <summary>Every word of every segment, in order (empty unless word timestamps were asked for).</summary>
        public IEnumerable<TranscriptionWord> Words
        {
            get
            {
                foreach (var s in Segments)
                    if (s.Words != null)
                        foreach (var w in s.Words) yield return w;
            }
        }
    }

    /// <summary>A text-to-speech model: text and a voice in, mono speech out.</summary>
    public interface ISpeechSynthesisModel
    {
        string SpeechModelFamily { get; }

        /// <summary>The voices bundled with the model, by name ("zf_xiaoxiao", "af_heart").</summary>
        IReadOnlyList<string> Voices { get; }

        /// <summary>The sample rate of what <see cref="Synthesize"/> returns.</summary>
        int OutputSampleRate { get; }

        DecodedAudio Synthesize(string text, SpeechSynthesisOptions options = null);
    }

    public sealed class SpeechSynthesisOptions
    {
        /// <summary>A name from <see cref="ISpeechSynthesisModel.Voices"/>, or null for the model's default.</summary>
        public string Voice { get; set; }

        /// <summary>Speaking rate; 1 is the voice's own.</summary>
        public float Speed { get; set; } = 1f;

        /// <summary>Language of the text, or null to take the voice's.</summary>
        public string Language { get; set; }

        /// <summary>Phonemes to speak instead of the text (the model's phoneme alphabet), for callers
        /// that run their own grapheme-to-phoneme step.</summary>
        public string Phonemes { get; set; }
    }

    /// <summary>A text-to-music model: a style description and lyrics in, a song out.</summary>
    public interface IMusicGenerationModel
    {
        string MusicModelFamily { get; }

        int OutputSampleRate { get; }

        DecodedAudio GenerateMusic(MusicGenerationRequest request);
    }

    public sealed class MusicGenerationRequest
    {
        /// <summary>Style tags or a caption ("upbeat synth-pop, female vocals, 120 bpm").</summary>
        public string Tags { get; set; } = string.Empty;

        /// <summary>Lyrics with [verse] / [chorus] section tags, or "[Instrumental]".</summary>
        public string Lyrics { get; set; } = "[Instrumental]";

        public double DurationSeconds { get; set; } = 30;

        /// <summary>Beats per minute, or 0 to leave it to the model.</summary>
        public int Bpm { get; set; }

        /// <summary>"C major", "A minor", or null to leave it to the model.</summary>
        public string KeyScale { get; set; }

        /// <summary>Beats per bar ("4"), or null to leave it to the model.</summary>
        public string TimeSignature { get; set; }

        /// <summary>Language of the lyrics ("en", "zh").</summary>
        public string Language { get; set; } = "en";

        public long Seed { get; set; }

        /// <summary>Diffusion steps, or 0 for the checkpoint's default.</summary>
        public int Steps { get; set; }

        /// <summary>Plan the song with the language model first (its audio codes guide the diffusion).</summary>
        public bool GenerateAudioCodes { get; set; } = true;

        /// <summary>Called after each diffusion step with (step, total).</summary>
        public Action<int, int> OnStep { get; set; }
    }
}
