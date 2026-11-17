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
    /// <summary>A frame of speech codes: 12 codes (all <see cref="Lfm3SpeechConfig.EndOfAudioCode"/> at the end of the
    /// speech).</summary>
    public readonly struct Lfm3SpeechFrame
    {
        public Lfm3SpeechFrame(int[] codes, bool endOfAudio)
        {
            Codes = codes;
            IsEndOfAudio = endOfAudio;
        }

        public int[] Codes { get; }

        /// <summary>A codebook sampled the end-of-audio code: the speech segment ends (the frame has no waveform).</summary>
        public bool IsEndOfAudio { get; }
    }

    /// <summary>The speech-code sampling of a request (the mmproj's defaults unless set).</summary>
    public sealed class Lfm3SpeechSampling
    {
        /// <summary>Temperature of the codes (llama.cpp: the only override a caller has; 0 = argmax).</summary>
        public float Temperature { get; set; }

        /// <summary>Keep codes scoring at least the k-th best (0: all; LFM3-Omni has none).</summary>
        public int TopK { get; set; }

        /// <summary>CTRL repetition penalty of the codes a codebook took in the last <see cref="RepetitionWindow"/> frames (1:
        /// off).</summary>
        public float RepetitionPenalty { get; set; }

        public int RepetitionWindow { get; set; }

        public static Lfm3SpeechSampling Defaults(Lfm3SpeechConfig config) => new()
        {
            Temperature = config.Temperature,
            TopK = config.TopK,
            RepetitionPenalty = config.RepetitionPenalty,
            RepetitionWindow = config.RepetitionWindow,
        };
    }

    /// <summary>
    /// The MaskGIT frame sampler of one speech request (llama.cpp mtmd-helper-gen.cpp lfm3a_gen_audio_pipeline::sample_frame;
    /// HF Lfm3OmniForConditionalGeneration.sample_audio_frame): a frame's codebooks are revealed in a random order, 4 a pass
    /// over 3 depthformer passes; each revealed codebook's logits (from the pass that reveals it) get the repetition penalty
    /// over the codes it took in the request's last frames, then are sampled at the temperature. Any codebook sampling the
    /// end-of-audio code makes the frame end-of-audio. The random stream is <see cref="Lfm3SpeechRandom"/>: a seed gives
    /// llama.cpp's (libc++) orders and draws, so given the same hidden rows the same codes come out wherever the logits agree
    /// to within the sampling margins. The history (codes of non-end frames, the last <c>RepetitionWindow</c> kept) belongs to
    /// the request: a new request takes a new sampler.
    /// </summary>
    public sealed class Lfm3FrameSampler
    {
        private const int Cb = Lfm3SpeechConfig.Codebooks, V = Lfm3SpeechConfig.CodebookSize;
        private readonly Lfm3Depthformer _depthformer;
        private readonly Lfm3SpeechSampling _sampling;
        private readonly Lfm3SpeechRandom _rng;
        private readonly LinkedList<int[]> _history = new();
        private readonly float[] _logits = new float[Cb * V];
        private readonly double[] _prob = new double[V];
        private readonly bool[] _seen = new bool[V];
        private readonly int[] _order = new int[Cb];

        internal Lfm3FrameSampler(Lfm3Depthformer depthformer, Lfm3SpeechSampling sampling, uint seed)
        {
            _depthformer = depthformer;
            _sampling = sampling;
            _rng = new Lfm3SpeechRandom(seed);
            Seed = seed;
        }

        public uint Seed { get; }

        public Lfm3SpeechSampling Sampling => _sampling;

        /// <summary>Frames sampled (end-of-audio ones included).</summary>
        public int Frames { get; private set; }

        /// <summary>The codes of the last non-end frames (oldest first), at most the repetition window.</summary>
        public IReadOnlyCollection<int[]> History => _history;

        /// <summary>The order the last frame revealed its codebooks in.</summary>
        public ReadOnlySpan<int> LastOrder => _order;

        /// <summary>Optional: every pass's logits of the codebooks it reveals, before the penalty (pass, codebook, logits), for
        /// validation.</summary>
        public Action<int, int, ReadOnlySpan<float>> OnPassLogits { get; set; }

        /// <summary>Sample the frame for the LM hidden row <paramref name="hidden"/> (the post-final-norm row of the position
        /// before it).</summary>
        public Lfm3SpeechFrame Sample(ReadOnlySpan<float> hidden)
        {
            for (int i = 0; i < Cb; i++) _order[i] = i;
            _rng.Shuffle(_order);
            var codes = new int[Cb];
            Array.Fill(codes, -1);
            ReadOnlySpan<int> schedule = Lfm3SpeechConfig.MaskGitSchedule;
            lock (_depthformer.Sync)
            {
                int k = 0;
                for (int pass = 0; pass < schedule.Length; pass++)
                {
                    ReadOnlySpan<int> pick = _order.AsSpan(k, schedule[pass]);
                    _depthformer.Pass(pass == 0 ? hidden : ReadOnlySpan<float>.Empty, codes, pick, _logits);
                    for (int j = 0; j < pick.Length; j++)
                    {
                        Span<float> l = _logits.AsSpan(j * V, V);
                        OnPassLogits?.Invoke(pass, pick[j], l);
                        ApplyRepetitionPenalty(l, pick[j]);
                        codes[pick[j]] = Draw(l);
                    }
                    k += pick.Length;
                }
            }
            Frames++;
            if (Array.IndexOf(codes, Lfm3SpeechConfig.EndOfAudioCode) >= 0)
            {
                Array.Fill(codes, Lfm3SpeechConfig.EndOfAudioCode);
                return new Lfm3SpeechFrame(codes, true);
            }
            _history.AddLast((int[])codes.Clone());
            while (_history.Count > Math.Max(0, _sampling.RepetitionWindow)) _history.RemoveFirst();
            return new Lfm3SpeechFrame(codes, false);
        }

        /// <summary>CTRL penalty: each code codebook <paramref name="codebook"/> took in the window, once: positive logits
        /// divided by the penalty, others multiplied.</summary>
        private void ApplyRepetitionPenalty(Span<float> l, int codebook)
        {
            float penalty = _sampling.RepetitionPenalty;
            if (penalty <= 1f || _sampling.RepetitionWindow <= 0 || _history.Count == 0) return;
            Array.Clear(_seen);
            foreach (int[] frame in _history)
            {
                int c = frame[codebook];
                if ((uint)c < V && !_seen[c])
                {
                    _seen[c] = true;
                    l[c] = l[c] > 0f ? l[c] / penalty : l[c] * penalty;
                }
            }
        }

        /// <summary>llama.cpp's sample(): argmax at temperature 0; else top-k (ties kept), exp((l - max) / temp) in double,
        /// one uniform draw scaled by the sum, the first code whose cumulative mass reaches it.</summary>
        private int Draw(ReadOnlySpan<float> l)
        {
            float temp = _sampling.Temperature;
            if (temp <= 0f)
            {
                int best = 0;
                for (int v = 1; v < l.Length; v++) if (l[v] > l[best]) best = v;
                return best;
            }
            float lmin = float.NegativeInfinity;
            int topK = _sampling.TopK;
            if (topK > 0 && topK < l.Length)
            {
                float[] sorted = l.ToArray();
                Array.Sort(sorted);
                lmin = sorted[sorted.Length - topK];
            }
            float lmax = float.NegativeInfinity;
            foreach (float v in l) lmax = Math.Max(lmax, v);
            double sum = 0;
            for (int v = 0; v < l.Length; v++)
            {
                _prob[v] = l[v] < lmin ? 0.0 : Math.Exp((double)(l[v] - lmax) / temp);
                sum += _prob[v];
            }
            double r = _rng.NextDouble() * sum;
            for (int v = 0; v < l.Length; v++)
            {
                r -= _prob[v];
                if (r <= 0.0) return v;
            }
            return l.Length - 1;
        }
    }
}
