// Copyright (c) Zhongkai Fu. All rights reserved.
// https://github.com/zhongkaifu/TensorSharp
//
// This file is part of TensorSharp.
//
// TensorSharp is licensed under the BSD-3-Clause license found in the LICENSE file in the root directory of this source tree.
//
// Long-form transcription as openai/whisper's transcribe() runs it: 30-second windows of the
// clip's log-mel features, each decoded with timestamps (greedy, falling back to sampling at
// higher temperatures when the text loops or is improbable), split into segments at the
// timestamp pairs, the next window starting where the last complete segment ended, and every
// window conditioned on the text before it. The logit filters are openai/whisper's
// SuppressBlank, SuppressTokens and ApplyTimestampRules.
using System;
using System.Collections.Generic;
using System.IO;
using System.IO.Compression;
using System.Linq;
using System.Text;
using TensorSharp.Models.Audio;

namespace TensorSharp.Models.Whisper
{
    internal sealed class WhisperTranscriber
    {
        private const double CompressionRatioThreshold = 2.4;
        private const double LogprobThreshold = -1.0;
        private const double NoSpeechThreshold = 0.6;
        private static readonly float[] FallbackTemperatures = { 0.2f, 0.4f, 0.6f, 0.8f, 1.0f };

        private readonly WhisperConfig _cfg;
        private readonly WhisperTokenizer _tok;
        private readonly WhisperNetwork _net;
        private readonly int[] _suppress;
        private readonly int _sampleLength;

        public WhisperTranscriber(WhisperConfig cfg, WhisperTokenizer tok, WhisperNetwork net)
        {
            _cfg = cfg;
            _tok = tok;
            _net = net;
            var suppress = new HashSet<int>(cfg.SuppressTokens.Where(t => t >= 0));
            foreach (int t in new[] { cfg.Translate, cfg.Transcribe, cfg.StartOfTranscript, cfg.StartOfPrev, cfg.StartOfLm, cfg.NoSpeech })
                if (t >= 0) suppress.Add(t);
            _suppress = suppress.ToArray();
            _sampleLength = cfg.MaxTargetPositions / 2;
        }

        private sealed class DecodeResult
        {
            public List<int> Tokens = new();   // sampled tokens, without the end of text
            public double SumLogprob;
            public double AvgLogprob;
            public double NoSpeechProb;
            public double CompressionRatio;
            public float Temperature;
            public string Text = "";
        }

        public int[] SotSequence(string language, bool translate)
        {
            var seq = new List<int> { _cfg.StartOfTranscript };
            if (_cfg.IsMultilingual)
            {
                seq.Add(_cfg.LanguageTokens.TryGetValue(language ?? "en", out int lang) ? lang : _cfg.LanguageTokens["en"]);
                seq.Add(translate ? _cfg.Translate : _cfg.Transcribe);
            }
            return seq.ToArray();
        }

        /// <summary>The most probable language of an encoded window (decoder run on &lt;|startoftranscript|&gt; alone).</summary>
        public (string Code, double Probability) DetectLanguage()
        {
            var output = _net.Decode(new[] { _cfg.StartOfTranscript }, 0, new[] { 0 });
            float[] logits = output.Logits[0];
            double max = double.NegativeInfinity;
            foreach (var id in _cfg.LanguageTokens.Values) max = Math.Max(max, logits[id]);
            double sum = 0;
            foreach (var id in _cfg.LanguageTokens.Values) sum += Math.Exp(logits[id] - max);
            string best = "en";
            double bestP = -1;
            foreach (var (code, id) in _cfg.LanguageTokens)
            {
                double p = Math.Exp(logits[id] - max) / sum;
                if (p > bestP) { bestP = p; best = code; }
            }
            return (best, bestP);
        }

        public TranscriptionResult Run(float[] mel, int totalFrames, TranscriptionOptions o, string language, double duration)
        {
            int chunk = _cfg.ChunkFrames;
            int contentFrames = totalFrames - chunk; // the features carry 30 s of trailing silence
            var rng = new Random(o.Seed);
            int[] sot = SotSequence(language, o.Translate);
            var allTokens = new List<int>();
            var segments = new List<TranscriptionSegment>();
            int promptResetSince = 0;
            if (!string.IsNullOrWhiteSpace(o.Prompt))
                allTokens.AddRange(_tok.Encode(" " + o.Prompt.Trim()));
            int initialPromptLength = allTokens.Count;
            double lastSpeechTimestamp = 0;

            int seek = 0;
            while (seek < contentFrames)
            {
                double timeOffset = seek * (double)_cfg.HopLength / _cfg.SampleRate;
                int segmentSize = Math.Min(chunk, contentFrames - seek);
                double segmentDuration = segmentSize * (double)_cfg.HopLength / _cfg.SampleRate;
                float[] window = Slice(mel, totalFrames, seek, chunk);
                _net.Encode(window, keepAlignmentKeys: o.WordTimestamps);

                var prompt = allTokens.Skip(promptResetSince).ToList();
                DecodeResult result = DecodeWithFallback(sot, prompt, o, rng);

                if (result.NoSpeechProb > NoSpeechThreshold && result.AvgLogprob < LogprobThreshold)
                {
                    seek += segmentSize; // silence
                    continue;
                }

                var tokens = result.Tokens;
                var current = new List<(double Start, double End, List<int> Tokens)>();
                bool IsTs(int t) => t >= _cfg.TimestampBegin;
                bool singleTimestampEnding = tokens.Count >= 2 && !IsTs(tokens[^2]) && IsTs(tokens[^1]);
                var consecutive = new List<int>();
                for (int i = 1; i < tokens.Count; i++)
                    if (IsTs(tokens[i - 1]) && IsTs(tokens[i])) consecutive.Add(i);

                if (consecutive.Count > 0)
                {
                    var slices = new List<int>(consecutive);
                    if (singleTimestampEnding) slices.Add(tokens.Count);
                    int last = 0;
                    foreach (int cur in slices)
                    {
                        var sliced = tokens.GetRange(last, cur - last);
                        double start = timeOffset + (sliced[0] - _cfg.TimestampBegin) * _cfg.TimePrecision;
                        double end = timeOffset + (sliced[^1] - _cfg.TimestampBegin) * _cfg.TimePrecision;
                        current.Add((start, end, sliced));
                        last = cur;
                    }
                    if (singleTimestampEnding) seek += segmentSize;
                    else
                    {
                        int lastPos = tokens[last - 1] - _cfg.TimestampBegin;
                        seek += lastPos * 2; // two mel frames per timestamp step
                    }
                }
                else
                {
                    double d = segmentDuration;
                    var stamps = tokens.Where(IsTs).ToList();
                    if (stamps.Count > 0 && stamps[^1] != _cfg.TimestampBegin)
                        d = (stamps[^1] - _cfg.TimestampBegin) * _cfg.TimePrecision;
                    current.Add((timeOffset, timeOffset + d, new List<int>(tokens)));
                    seek += segmentSize;
                }

                var built = new List<TranscriptionSegment>();
                foreach (var (start, end, segTokens) in current)
                {
                    string text = _tok.Decode(segTokens);
                    if (start == end || string.IsNullOrWhiteSpace(text)) continue;
                    built.Add(new TranscriptionSegment
                    {
                        Start = Math.Round(start, 3), End = Math.Round(end, 3), Text = text, Tokens = segTokens.ToArray(),
                        Temperature = result.Temperature, AvgLogprob = result.AvgLogprob,
                        CompressionRatio = result.CompressionRatio, NoSpeechProb = result.NoSpeechProb,
                    });
                }

                if (o.WordTimestamps && built.Count > 0)
                {
                    var timed = WhisperTiming.AddWordTimestamps(built, _net, _cfg, _tok, sot, segmentSize, timeOffset,
                        language, ref lastSpeechTimestamp);
                    built = timed;
                    if (!singleTimestampEnding)
                    {
                        double? lastWordEnd = built.SelectMany(s => s.Words ?? new List<TranscriptionWord>())
                            .Select(w => (double?)w.End).LastOrDefault();
                        if (lastWordEnd is double end && end > timeOffset)
                            seek = (int)Math.Round(end * _cfg.SampleRate / _cfg.HopLength);
                    }
                }

                foreach (var s in built)
                {
                    var segment = new TranscriptionSegment
                    {
                        Id = segments.Count, Start = s.Start, End = s.End, Text = s.Text, Tokens = s.Tokens,
                        Temperature = s.Temperature, AvgLogprob = s.AvgLogprob, CompressionRatio = s.CompressionRatio,
                        NoSpeechProb = s.NoSpeechProb, Words = s.Words,
                    };
                    segments.Add(segment);
                    allTokens.AddRange(segment.Tokens);
                    o.OnSegment?.Invoke(segment);
                }
                if (!o.ConditionOnPreviousText || result.Temperature > 0.5f)
                    promptResetSince = allTokens.Count;
            }

            return new TranscriptionResult
            {
                Text = _tok.Decode(allTokens.Skip(initialPromptLength)),
                Language = language,
                Duration = duration,
                Task = o.Translate ? "translate" : "transcribe",
                Segments = segments,
            };
        }

        private static float[] Slice(float[] mel, int totalFrames, int start, int frames)
        {
            int nMels = mel.Length / totalFrames;
            var w = new float[(long)nMels * frames];
            int available = Math.Max(0, Math.Min(frames, totalFrames - start));
            for (int m = 0; m < nMels; m++)
                Array.Copy(mel, (long)m * totalFrames + start, w, (long)m * frames, available);
            return w;
        }

        private DecodeResult DecodeWithFallback(int[] sot, List<int> prompt, TranscriptionOptions o, Random rng)
        {
            var temperatures = new List<float> { o.Temperature };
            if (o.TemperatureFallback)
                temperatures.AddRange(FallbackTemperatures.Where(t => t > o.Temperature));
            DecodeResult result = null;
            foreach (float t in temperatures)
            {
                result = DecodeWindow(sot, prompt, t, rng);
                bool fallback = result.CompressionRatio > CompressionRatioThreshold || result.AvgLogprob < LogprobThreshold;
                if (result.NoSpeechProb > NoSpeechThreshold) fallback = false;
                if (!fallback) break;
            }
            return result;
        }

        private DecodeResult DecodeWindow(int[] sot, List<int> prompt, float temperature, Random rng)
        {
            var initial = new List<int>();
            if (prompt.Count > 0)
            {
                initial.Add(_cfg.StartOfPrev);
                int keep = _cfg.MaxTargetPositions / 2 - 1;
                initial.AddRange(prompt.Skip(Math.Max(0, prompt.Count - keep)));
            }
            int sotIndex = initial.Count;
            initial.AddRange(sot);
            int sampleBegin = initial.Count;

            var first = _net.Decode(initial, 0, new[] { sotIndex, initial.Count - 1 });
            var result = new DecodeResult { Temperature = temperature };
            {
                float[] sotLogits = first.Logits[0];
                result.NoSpeechProb = _cfg.NoSpeech >= 0 ? Softmax(sotLogits)[_cfg.NoSpeech] : 0;
            }

            var seq = new List<int>(initial);
            float[] logits = first.Logits[1];
            for (int step = 0; step < _sampleLength; step++)
            {
                ApplyFilters(logits, seq, sampleBegin);
                double[] logprobs = LogSoftmax(logits);
                int next = temperature <= 0f ? ArgMax(logits) : Sample(logits, temperature, rng);
                result.SumLogprob += logprobs[next];
                if (next == _cfg.EndOfText) break;
                seq.Add(next);
                result.Tokens.Add(next);
                if (seq.Count >= _cfg.MaxTargetPositions) break;
                logits = _net.Decode(new[] { next }, seq.Count - 1, new[] { 0 }).Logits[0];
            }
            result.AvgLogprob = result.SumLogprob / (result.Tokens.Count + 1);
            result.Text = _tok.Decode(result.Tokens);
            result.CompressionRatio = CompressionRatio(result.Text);
            return result;
        }

        private void ApplyFilters(float[] logits, List<int> seq, int sampleBegin)
        {
            int ts = _cfg.TimestampBegin, eot = _cfg.EndOfText;
            if (seq.Count == sampleBegin)
                foreach (int t in _cfg.BeginSuppressTokens) if (t >= 0 && t < logits.Length) logits[t] = float.NegativeInfinity;
            foreach (int t in _suppress) if (t < logits.Length) logits[t] = float.NegativeInfinity;

            // ApplyTimestampRules
            logits[_cfg.NoTimestamps] = float.NegativeInfinity;
            int sampled = seq.Count - sampleBegin;
            bool lastWasTs = sampled >= 1 && seq[^1] >= ts;
            bool penultimateWasTs = sampled < 2 || seq[^2] >= ts;
            if (lastWasTs)
            {
                if (penultimateWasTs) Fill(logits, ts, logits.Length);
                else Fill(logits, 0, eot);
            }
            int lastTs = -1;
            for (int i = seq.Count - 1; i >= sampleBegin; i--)
                if (seq[i] >= ts) { lastTs = seq[i]; break; }
            if (lastTs >= 0)
            {
                int bound = lastWasTs && !penultimateWasTs ? lastTs : lastTs + 1;
                Fill(logits, ts, bound);
            }
            if (sampled == 0)
            {
                Fill(logits, 0, ts);
                int lastAllowed = ts + _cfg.MaxInitialTimestampIndex;
                Fill(logits, lastAllowed + 1, logits.Length);
            }
            double[] lp = LogSoftmax(logits);
            double tsMass = LogSumExp(lp, ts, lp.Length);
            double maxText = double.NegativeInfinity;
            for (int i = 0; i < ts; i++) if (lp[i] > maxText) maxText = lp[i];
            if (tsMass > maxText) Fill(logits, 0, ts);
        }

        private static void Fill(float[] a, int from, int to)
        {
            for (int i = Math.Max(0, from); i < Math.Min(a.Length, to); i++) a[i] = float.NegativeInfinity;
        }

        internal static double[] LogSoftmax(float[] logits)
        {
            double max = double.NegativeInfinity;
            foreach (float v in logits) if (v > max) max = v;
            double sum = 0;
            foreach (float v in logits) sum += Math.Exp(v - max);
            double lse = max + Math.Log(sum);
            var r = new double[logits.Length];
            for (int i = 0; i < logits.Length; i++) r[i] = logits[i] - lse;
            return r;
        }

        internal static double[] Softmax(float[] logits)
        {
            double[] lp = LogSoftmax(logits);
            for (int i = 0; i < lp.Length; i++) lp[i] = Math.Exp(lp[i]);
            return lp;
        }

        private static double LogSumExp(double[] a, int from, int to)
        {
            double max = double.NegativeInfinity;
            for (int i = from; i < to; i++) if (a[i] > max) max = a[i];
            if (double.IsNegativeInfinity(max)) return max;
            double s = 0;
            for (int i = from; i < to; i++) s += Math.Exp(a[i] - max);
            return max + Math.Log(s);
        }

        private static int ArgMax(float[] a)
        {
            int best = 0;
            for (int i = 1; i < a.Length; i++) if (a[i] > a[best]) best = i;
            return best;
        }

        private static int Sample(float[] logits, float temperature, Random rng)
        {
            double max = double.NegativeInfinity;
            foreach (float v in logits) if (v > max) max = v;
            var p = new double[logits.Length];
            double sum = 0;
            for (int i = 0; i < logits.Length; i++)
            {
                p[i] = float.IsNegativeInfinity(logits[i]) ? 0 : Math.Exp((logits[i] - max) / temperature);
                sum += p[i];
            }
            double u = rng.NextDouble() * sum, acc = 0;
            for (int i = 0; i < p.Length; i++)
            {
                acc += p[i];
                if (acc >= u && p[i] > 0) return i;
            }
            return ArgMax(logits);
        }

        /// <summary>len(text) / len(zlib.compress(text)), on UTF-8 bytes (openai/whisper's repetition check).</summary>
        internal static double CompressionRatio(string text)
        {
            byte[] bytes = Encoding.UTF8.GetBytes(text);
            if (bytes.Length == 0) return 0;
            using var ms = new MemoryStream();
            using (var z = new ZLibStream(ms, CompressionLevel.Optimal, leaveOpen: true))
                z.Write(bytes, 0, bytes.Length);
            return bytes.Length / (double)ms.Length;
        }
    }
}
