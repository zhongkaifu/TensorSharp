// Copyright (c) Zhongkai Fu. All rights reserved.
// https://github.com/zhongkaifu/TensorSharp
//
// This file is part of TensorSharp.
//
// TensorSharp is licensed under the BSD-3-Clause license found in the LICENSE file in the root directory of this source tree.
//
// OpenAI Whisper speech recognition (large-v3, large-v3-turbo and the smaller multilingual and
// English-only checkpoints), from a GGUF written by eng/audio-models/whisper_to_gguf.py:
//
//   TensorSharp.Cli --model whisper-large-v3-turbo-f16.gguf --audio speech.mp3 [--language zh]
//       [--word-timestamps] [--output transcript.srt|.vtt|.json|.txt] --backend cuda
//
// The server exposes it as OpenAI's POST /v1/audio/transcriptions and /v1/audio/translations.
using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.IO;
using System.Linq;
using TensorSharp.Models.Architecture;
using TensorSharp.Models.Audio;
using TensorSharp.Models.Direct;
using TensorSharp.Models.Video;
using TensorSharp.Runtime;

namespace TensorSharp.Models.Whisper
{
    public sealed class WhisperModel : ModelBase, ISpeechRecognitionModel
    {
        public const string ArchitectureId = "whisper";

        private readonly GgufFile _file;
        private readonly WhisperTokenizer _tokenizer;
        private readonly WhisperFeatures _features;
        private readonly DirectContext _ctx;
        private readonly WhisperNetwork _net;
        private readonly WhisperTranscriber _transcriber;
        private readonly object _lock = new();
        private bool _disposed;

        public WhisperConfig WhisperConfig { get; }

        /// <summary>Seconds of the last transcription: features, language detection, decoding.</summary>
        public (double Features, double Language, double Decode, double AudioSeconds) LastTimings { get; private set; }

        public WhisperModel(string ggufPath, BackendType backend)
            : base(ggufPath, backend, openGguf: false)
        {
            if (backend is not (BackendType.Cpu or BackendType.Cuda))
                throw new ModelLoadRefusedException(
                    $"Whisper runs on --backend cpu or cuda (the direct tensor path), not {BackendNames.OptionName(backend)}.");
            _file = new GgufFile(ggufPath);
            try
            {
                _file.ThrowIfTruncated();
                string[] vocab = _file.GetStringArray("tokenizer.ggml.tokens")
                    ?? throw new InvalidDataException("the Whisper GGUF has no tokenizer.ggml.tokens.");
                int[] types = _file.GetInt32Array("tokenizer.ggml.token_type");
                string[] merges = _file.GetStringArray("tokenizer.ggml.merges") ?? Array.Empty<string>();
                WhisperConfig = WhisperConfig.Read(_file, vocab);
                _tokenizer = new WhisperTokenizer(vocab, types, merges, WhisperConfig);
                float[] filters = DirectOps.DequantTensor(_file, "whisper.mel_filters");
                _features = new WhisperFeatures(filters, WhisperConfig.NumMelBins, WhisperConfig.NFft, WhisperConfig.HopLength);
                Config = new ModelConfig { Architecture = ArchitectureId, VocabSize = vocab.Length };

                var clock = Stopwatch.StartNew();
                _ctx = new DirectContext(_allocator);
                _net = new WhisperNetwork(_ctx, _file, WhisperConfig);
                _transcriber = new WhisperTranscriber(WhisperConfig, _tokenizer, _net);
                Console.WriteLine($"{WhisperConfig} ({clock.Elapsed.TotalSeconds:F1}s to bind weights)");
            }
            catch
            {
                _net?.Dispose();
                _ctx?.Dispose();
                _file.Dispose();
                throw;
            }
        }

        // ---- ISpeechRecognitionModel ----
        public string SpeechModelFamily => ArchitectureId;

        public IReadOnlyList<string> Languages => WhisperConfig.LanguageTokens.Keys.OrderBy(k => k, StringComparer.Ordinal).ToList();

        public TranscriptionResult Transcribe(DecodedAudio audio, TranscriptionOptions options = null)
        {
            ArgumentNullException.ThrowIfNull(audio);
            options ??= new TranscriptionOptions();
            string language = options.Language?.Trim().ToLowerInvariant();
            if (string.IsNullOrEmpty(language)) language = null;
            if (language != null && WhisperConfig.IsMultilingual && !WhisperConfig.LanguageTokens.ContainsKey(language))
                throw new ArgumentException($"Whisper does not know language '{options.Language}'. " +
                                            $"Use one of: {string.Join(", ", Languages)}.");
            if (options.Translate && !WhisperConfig.IsMultilingual)
                throw new ArgumentException("this English-only Whisper checkpoint cannot translate.");

            lock (_lock)
            {
                ObjectDisposedException.ThrowIf(_disposed, this);
                var total = Stopwatch.StartNew();
                var conformed = AudioIO.Conform(audio, WhisperConfig.SampleRate, 1);
                float[] samples = conformed.Channels[0];
                double duration = samples.Length / (double)WhisperConfig.SampleRate;

                // openai/whisper pads the clip with a window of silence before the STFT, so the
                // last window is complete and the normalisation sees the same range.
                var padded = new float[samples.Length + WhisperConfig.ChunkSamples];
                Array.Copy(samples, padded, samples.Length);
                var clock = Stopwatch.StartNew();
                float[] mel = _features.Compute(padded, out int frames);
                double featureSeconds = clock.Elapsed.TotalSeconds;

                clock.Restart();
                if (!WhisperConfig.IsMultilingual) language = "en";
                else if (language == null)
                {
                    var window = new float[(long)WhisperConfig.NumMelBins * WhisperConfig.ChunkFrames];
                    for (int m = 0; m < WhisperConfig.NumMelBins; m++)
                        Array.Copy(mel, (long)m * frames, window, (long)m * WhisperConfig.ChunkFrames, WhisperConfig.ChunkFrames);
                    _net.Encode(window, keepAlignmentKeys: false);
                    var (code, p) = _transcriber.DetectLanguage();
                    language = code;
                    Console.WriteLine($"  [whisper] detected language: {code} (p={p:0.00})");
                }
                double languageSeconds = clock.Elapsed.TotalSeconds;

                clock.Restart();
                var result = _transcriber.Run(mel, frames, options, language, duration);
                double decodeSeconds = clock.Elapsed.TotalSeconds;
                LastTimings = (featureSeconds, languageSeconds, decodeSeconds, duration);
                Console.WriteLine($"  [whisper] {duration:F1}s of audio, {result.Segments.Count} segments: features {featureSeconds:F2}s, " +
                                  $"language {languageSeconds:F2}s, decode {decodeSeconds:F2}s, total {total.Elapsed.TotalSeconds:F2}s " +
                                  $"({duration / Math.Max(1e-9, total.Elapsed.TotalSeconds):F1}x real time)");
                return result;
            }
        }

        // ---- not a chat model ----
        protected override float[] ForwardCore(int[] tokens) =>
            throw new NotSupportedException("Whisper is a speech recognition model; use Transcribe().");

        protected override void ResetKVCacheCore() { }

        public override void WarmUpKernels() { }

        public override void Dispose()
        {
            lock (_lock)
            {
                if (_disposed) return;
                _disposed = true;
                _net?.Dispose();
                _ctx?.Dispose();
                _file?.Dispose();
                base.Dispose();
            }
        }
    }
}
