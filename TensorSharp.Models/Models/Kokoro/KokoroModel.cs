// Copyright (c) Zhongkai Fu. All rights reserved.
// https://github.com/zhongkaifu/TensorSharp
//
// This file is part of TensorSharp.
//
// TensorSharp is licensed under the BSD-3-Clause license found in the LICENSE file in the root directory of this source tree.
//
// Kokoro-82M v1.0 text to speech (Apache-2.0), from a GGUF written by
// eng/audio-models/kokoro_to_gguf.py, on the managed CPU path:
//
//   TensorSharp.Cli --model kokoro-82m-v1.0.gguf --prompt "今天天气很好。" --voice zm_yunjian --output speech.wav
//
// The server exposes it as OpenAI's POST /v1/audio/speech. Text goes through the language's
// front end (Mandarin as Kokoro's own pipeline phonemizes it); callers with their own
// grapheme-to-phoneme step can pass phonemes instead.
using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.IO;
using System.Linq;
using System.Text;
using System.Text.Json;
using System.Text.RegularExpressions;
using TensorSharp.Models.Architecture;
using TensorSharp.Models.Audio;
using TensorSharp.Models.Direct;
using TensorSharp.Models.Video;
using TensorSharp.Runtime;

namespace TensorSharp.Models.Kokoro
{
    public sealed class KokoroModel : ModelBase, ISpeechSynthesisModel
    {
        public const string ArchitectureId = "kokoro";
        /// <summary>The model's context: 512 positions, two of them the boundary pads.</summary>
        public const int MaxPhonemes = 510;

        private readonly GgufFile _file;
        private readonly KokoroNetwork _net;
        private readonly Dictionary<string, int> _vocab = new(StringComparer.Ordinal);
        private readonly string[] _voices;
        private readonly Dictionary<string, float[]> _voicePacks = new(StringComparer.Ordinal);
        private readonly KokoroChineseG2P _zh;
        private readonly object _lock = new();
        private bool _disposed;

        /// <summary>Seconds of the last synthesis: front end, network.</summary>
        public (double FrontEnd, double Network, double AudioSeconds) LastTimings { get; private set; }

        /// <summary>The phonemes of the last synthesis, chunk by chunk.</summary>
        public IReadOnlyList<string> LastPhonemes { get; private set; } = Array.Empty<string>();

        /// <summary>The predicted frame durations of the last synthesis (one list per chunk).</summary>
        public IReadOnlyList<int[]> LastDurations { get; private set; } = Array.Empty<int[]>();

        public KokoroModel(string ggufPath, BackendType backend)
            : base(ggufPath, backend, openGguf: false)
        {
            if (backend != BackendType.Cpu)
                throw new ModelLoadRefusedException(
                    $"Kokoro runs on --backend cpu (its 82M-parameter network is managed C#), not {BackendNames.OptionName(backend)}.");
            _file = new GgufFile(ggufPath);
            try
            {
                _file.ThrowIfTruncated();
                string cfgJson = _file.GetString("kokoro.config")
                    ?? throw new InvalidDataException("not a TensorSharp Kokoro GGUF: kokoro.config is missing " +
                                                      "(convert the checkpoint with eng/audio-models/kokoro_to_gguf.py).");
                using var cfg = JsonDocument.Parse(cfgJson);
                string[] symbols = _file.GetStringArray("kokoro.vocab.symbols");
                int[] ids = _file.GetInt32Array("kokoro.vocab.ids");
                for (int i = 0; i < symbols.Length; i++) _vocab[symbols[i]] = ids[i];
                _voices = _file.GetStringArray("kokoro.voices") ?? Array.Empty<string>();
                if (_voices.Length == 0) throw new InvalidDataException("the Kokoro GGUF carries no voices.");
                var clock = Stopwatch.StartNew();
                _net = new KokoroNetwork(_file, cfg.RootElement.Clone());
                _net.Deterministic = Environment.GetEnvironmentVariable("TS_KOKORO_DETERMINISTIC") == "1";
                if (KokoroChineseG2P.IsAvailable(_file)) _zh = new KokoroChineseG2P(_file);
                Config = new ModelConfig { Architecture = ArchitectureId, VocabSize = _vocab.Count };
                Console.WriteLine($"Kokoro-82M: {_voices.Length} voices ({string.Join(", ", _voices)}), " +
                                  $"Mandarin front end {(_zh != null ? "on" : "absent")} ({clock.Elapsed.TotalSeconds:F1}s to load)");
            }
            catch
            {
                _file.Dispose();
                throw;
            }
        }

        // ---- ISpeechSynthesisModel ----
        public string SpeechModelFamily => ArchitectureId;
        public IReadOnlyList<string> Voices => _voices;
        public int OutputSampleRate => KokoroNetwork.SampleRate;

        public DecodedAudio Synthesize(string text, SpeechSynthesisOptions options = null)
        {
            options ??= new SpeechSynthesisOptions();
            string voice = options.Voice ?? (_voices.Contains("zm_yunjian") ? "zm_yunjian" : _voices[0]);
            if (!_voices.Contains(voice))
                throw new ArgumentException($"Kokoro has no voice '{voice}'; this file has {string.Join(", ", _voices)}.");
            if (options.Speed is <= 0f or > 4f)
                throw new ArgumentException("speed must be above 0 and at most 4.");
            string language = (options.Language ?? LanguageOfVoice(voice)).ToLowerInvariant();

            lock (_lock)
            {
                ObjectDisposedException.ThrowIf(_disposed, this);
                var clock = Stopwatch.StartNew();
                List<string> chunks = !string.IsNullOrWhiteSpace(options.Phonemes)
                    ? SplitPhonemes(options.Phonemes)
                    : Phonemize(text ?? string.Empty, language);
                double frontEnd = clock.Elapsed.TotalSeconds;
                clock.Restart();
                float[] pack = VoicePack(voice);
                var rng = new Random(0);   // the harmonic source's random phase and noise: fixed, so a request is reproducible
                var audio = new List<float>();
                var durations = new List<int[]>();
                foreach (string ps in chunks)
                {
                    Console.WriteLine($"  [kokoro] phonemes: {ps}");
                    int[] ids = ps.Select(c => _vocab.TryGetValue(c.ToString(), out int id) ? id : -1).Where(id => id >= 0).ToArray();
                    if (ids.Length == 0) continue;
                    var padded = new int[ids.Length + 2];
                    Array.Copy(ids, 0, padded, 1, ids.Length);
                    int row = Math.Min(ps.Length, MaxPhonemes) - 1;
                    float[] refS = pack.AsSpan(row * 256, 256).ToArray();
                    var (wave, dur) = _net.Forward(padded, refS, options.Speed, rng);
                    audio.AddRange(wave);
                    durations.Add(dur);
                }
                double network = clock.Elapsed.TotalSeconds;
                LastPhonemes = chunks;
                LastDurations = durations;
                double seconds = audio.Count / (double)KokoroNetwork.SampleRate;
                LastTimings = (frontEnd, network, seconds);
                Console.WriteLine($"  [kokoro] {voice}: {chunks.Count} chunk(s), {chunks.Sum(c => c.Length)} phonemes -> {seconds:F2}s of speech: " +
                                  $"front end {frontEnd:F2}s, network {network:F2}s ({seconds / Math.Max(1e-9, network):F1}x real time)");
                return new DecodedAudio { Channels = new[] { audio.ToArray() }, SampleRate = KokoroNetwork.SampleRate };
            }
        }

        private static string LanguageOfVoice(string voice) => voice.Length > 0 && voice[0] == 'z' ? "zh" : "en";

        private float[] VoicePack(string voice)
        {
            if (_voicePacks.TryGetValue(voice, out float[] pack)) return pack;
            pack = DirectOps.DequantTensor(_file, "voice." + voice);
            _voicePacks[voice] = pack;
            return pack;
        }

        private static readonly Regex Paragraphs = new(@"\n+");
        private static readonly Regex Sentences = new(@"([.!?]+)");

        /// <summary>KPipeline.__call__ for non-English text: paragraphs, then sentence-packed chunks of at most
        /// 400 characters, each phonemized and cut at 510 phonemes.</summary>
        private List<string> Phonemize(string text, string language)
        {
            if (language is not ("zh" or "z" or "cmn" or "zh-cn"))
                throw new NotSupportedException(
                    $"Kokoro's '{language}' text front end is not in this build; Mandarin text is, and any language can be " +
                    "spoken from phonemes (--phonemes / SpeechSynthesisOptions.Phonemes).");
            if (_zh == null)
                throw new NotSupportedException("this Kokoro GGUF was converted without the Mandarin tables (--no-zh); pass phonemes instead.");
            var result = new List<string>();
            foreach (string paragraph in Paragraphs.Split(text.Trim()))
            {
                if (string.IsNullOrWhiteSpace(paragraph)) continue;
                const int chunkSize = 400;
                var chunks = new List<string>();
                string[] parts = Sentences.Split(paragraph);
                var current = new StringBuilder();
                for (int i = 0; i < parts.Length; i += 2)
                {
                    string sentence = parts[i] + (i + 1 < parts.Length ? parts[i + 1] : "");
                    if (current.Length + sentence.Length <= chunkSize) current.Append(sentence);
                    else
                    {
                        if (current.Length > 0) chunks.Add(current.ToString().Trim());
                        current.Clear().Append(sentence);
                    }
                }
                if (current.Length > 0) chunks.Add(current.ToString().Trim());
                foreach (string chunk in chunks)
                {
                    if (string.IsNullOrWhiteSpace(chunk)) continue;
                    string ps = _zh.Phonemize(chunk);
                    if (ps.Length == 0) continue;
                    result.Add(ps.Length > MaxPhonemes ? ps[..MaxPhonemes] : ps);
                }
            }
            return result;
        }

        private static List<string> SplitPhonemes(string phonemes)
        {
            var result = new List<string>();
            foreach (string line in Paragraphs.Split(phonemes.Trim()))
            {
                string ps = line.Trim();
                if (ps.Length == 0) continue;
                if (ps.Length > MaxPhonemes)
                    throw new ArgumentException($"a phoneme line holds at most {MaxPhonemes} symbols; split it at a newline.");
                result.Add(ps);
            }
            return result;
        }

        /// <summary>Phonemes for text without running the network (the front end alone).</summary>
        public IReadOnlyList<string> PhonemizeOnly(string text, string language = "zh")
        {
            lock (_lock) return Phonemize(text, language);
        }

        // ---- not a chat model ----
        protected override float[] ForwardCore(int[] tokens) =>
            throw new NotSupportedException("Kokoro is a speech synthesis model; use Synthesize().");

        protected override void ResetKVCacheCore() { }

        public override void WarmUpKernels() { }

        public override void Dispose()
        {
            lock (_lock)
            {
                if (_disposed) return;
                _disposed = true;
                _file?.Dispose();
                base.Dispose();
            }
        }
    }
}
