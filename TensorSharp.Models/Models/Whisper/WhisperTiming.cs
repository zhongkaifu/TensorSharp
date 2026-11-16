// Copyright (c) Zhongkai Fu. All rights reserved.
// https://github.com/zhongkaifu/TensorSharp
//
// This file is part of TensorSharp.
//
// TensorSharp is licensed under the BSD-3-Clause license found in the LICENSE file in the root directory of this source tree.
//
// Word timestamps as openai/whisper's timing.py computes them: decode the window's text again
// (teacher-forced, without timestamps), take the cross-attention scores of the checkpoint's
// alignment heads, softmax them over the audio frames, normalise each frame column, median-filter
// along time, and align tokens to frames with dynamic time warping. Each word starts at the frame
// where its first token's path enters, and the heuristics that clip over-long words at pauses and
// sentence ends follow the reference.
using System;
using System.Collections.Generic;
using System.Globalization;
using System.Linq;
using TensorSharp.Models.Audio;

namespace TensorSharp.Models.Whisper
{
    internal static class WhisperTiming
    {
        private const string PrependPunctuations = "\"'\u201c\u00bf([{-";
        private const string AppendPunctuations = "\"'.\u3002,\uff0c!\uff01?\uff1f:\uff1a\u201d)]}\u3001";
        private const double TokensPerSecond = 50.0;
        private static readonly HashSet<string> UnicodeSplitLanguages = new() { "zh", "ja", "th", "lo", "my", "yue" };

        private sealed class WordTiming
        {
            public string Word;
            public List<int> Tokens;
            public double Start, End, Probability;
        }

        public static List<TranscriptionSegment> AddWordTimestamps(List<TranscriptionSegment> segments, WhisperNetwork net,
            WhisperConfig cfg, WhisperTokenizer tok, int[] sot, int numFrames, double timeOffset, string language,
            ref double lastSpeechTimestamp)
        {
            var textTokensPerSegment = segments.Select(s => s.Tokens.Where(t => t < cfg.EndOfText).ToList()).ToList();
            var textTokens = textTokensPerSegment.SelectMany(t => t).ToList();
            var alignment = FindAlignment(net, cfg, tok, sot, textTokens, numFrames, language);

            var durations = alignment.Select(w => w.End - w.Start).Where(d => d != 0).OrderBy(d => d).ToList();
            double median = durations.Count == 0 ? 0 : durations.Count % 2 == 1
                ? durations[durations.Count / 2]
                : (durations[durations.Count / 2 - 1] + durations[durations.Count / 2]) / 2;
            median = Math.Min(0.7, median);
            double maxDuration = median * 2;

            if (durations.Count > 0)
            {
                const string sentenceEnd = ".\u3002!\uff01?\uff1f";
                for (int i = 1; i < alignment.Count; i++)
                {
                    if (alignment[i].End - alignment[i].Start > maxDuration)
                    {
                        if (sentenceEnd.Contains(alignment[i].Word, StringComparison.Ordinal))
                            alignment[i].End = alignment[i].Start + maxDuration;
                        else if (sentenceEnd.Contains(alignment[i - 1].Word, StringComparison.Ordinal))
                            alignment[i].Start = alignment[i].End - maxDuration;
                    }
                }
            }
            MergePunctuations(alignment);

            var output = new List<TranscriptionSegment>();
            int wordIndex = 0;
            for (int s = 0; s < segments.Count; s++)
            {
                var segment = segments[s];
                var text = textTokensPerSegment[s];
                int saved = 0;
                var words = new List<(string Word, double Start, double End, double Probability)>();
                while (wordIndex < alignment.Count && saved < text.Count)
                {
                    var timing = alignment[wordIndex];
                    if (!string.IsNullOrEmpty(timing.Word))
                        words.Add((timing.Word, Math.Round(timeOffset + timing.Start, 2), Math.Round(timeOffset + timing.End, 2), timing.Probability));
                    saved += timing.Tokens.Count;
                    wordIndex++;
                }

                double segStart = segment.Start, segEnd = segment.End;
                if (words.Count > 0)
                {
                    var w0 = words[0];
                    if (w0.End - lastSpeechTimestamp > median * 4 &&
                        (w0.End - w0.Start > maxDuration || (words.Count > 1 && words[1].End - w0.Start > maxDuration * 2)))
                    {
                        if (words.Count > 1 && words[1].End - words[1].Start > maxDuration)
                        {
                            double boundary = Math.Max(words[1].End / 2, words[1].End - maxDuration);
                            w0.End = boundary;
                            words[1] = (words[1].Word, boundary, words[1].End, words[1].Probability);
                        }
                        w0.Start = Math.Max(0, w0.End - maxDuration);
                        words[0] = w0;
                    }
                    w0 = words[0];
                    if (segStart < w0.End && segStart - 0.5 > w0.Start)
                        words[0] = (w0.Word, Math.Max(0, Math.Min(w0.End - median, segStart)), w0.End, w0.Probability);
                    else
                        segStart = w0.Start;

                    var wl = words[^1];
                    if (segEnd > wl.Start && segEnd + 0.5 < wl.End)
                        words[^1] = (wl.Word, wl.Start, Math.Max(wl.Start + median, segEnd), wl.Probability);
                    else
                        segEnd = wl.End;
                    lastSpeechTimestamp = segEnd;
                }

                output.Add(new TranscriptionSegment
                {
                    Id = segment.Id, Start = segStart, End = segEnd, Text = segment.Text, Tokens = segment.Tokens,
                    Temperature = segment.Temperature, AvgLogprob = segment.AvgLogprob,
                    CompressionRatio = segment.CompressionRatio, NoSpeechProb = segment.NoSpeechProb,
                    Words = words.Select(w => new TranscriptionWord
                    {
                        Word = w.Word, Start = w.Start, End = w.End, Probability = Math.Round(w.Probability, 4),
                    }).ToList(),
                });
            }
            return output;
        }

        private static List<WordTiming> FindAlignment(WhisperNetwork net, WhisperConfig cfg, WhisperTokenizer tok,
            int[] sot, List<int> textTokens, int numFrames, string language)
        {
            if (textTokens.Count == 0 || cfg.AlignmentHeads.Length == 0) return new List<WordTiming>();
            var tokens = new List<int>(sot) { cfg.NoTimestamps };
            tokens.AddRange(textTokens);
            tokens.Add(cfg.EndOfText);

            // Probabilities of the text tokens (rows sot.Length .. sot.Length + n - 1 predict them).
            var rows = Enumerable.Range(sot.Length, textTokens.Count).ToArray();
            var output = net.Decode(tokens, 0, rows, captureCross: true);
            var tokenProbs = new double[textTokens.Count];
            for (int i = 0; i < textTokens.Count; i++)
            {
                float[] l = output.Logits[i];
                double max = double.NegativeInfinity;
                for (int v = 0; v < cfg.EndOfText; v++) if (l[v] > max) max = l[v];
                double sum = 0;
                for (int v = 0; v < cfg.EndOfText; v++) sum += Math.Exp(l[v] - max);
                tokenProbs[i] = Math.Exp(l[textTokens[i]] - max) / sum;
            }

            int nTok = tokens.Count;
            int frames = Math.Min(numFrames / 2, cfg.MaxSourcePositions);
            int heads = cfg.AlignmentHeads.Length;
            var weights = new double[heads][];
            for (int h = 0; h < heads; h++)
            {
                var (layer, head) = cfg.AlignmentHeads[h];
                float[] scores = net.AlignmentScores(output, layer, head, nTok, out int allFrames);
                var w = new double[nTok * frames];
                for (int r = 0; r < nTok; r++)
                {
                    double max = double.NegativeInfinity;
                    for (int f = 0; f < frames; f++) max = Math.Max(max, scores[(long)r * allFrames + f]);
                    double sum = 0;
                    for (int f = 0; f < frames; f++) sum += Math.Exp(scores[(long)r * allFrames + f] - max);
                    for (int f = 0; f < frames; f++) w[r * frames + f] = Math.Exp(scores[(long)r * allFrames + f] - max) / sum;
                }
                // Standardise each frame over the token axis (torch.std_mean(dim=-2, unbiased=False)).
                for (int f = 0; f < frames; f++)
                {
                    double mean = 0;
                    for (int r = 0; r < nTok; r++) mean += w[r * frames + f];
                    mean /= nTok;
                    double var = 0;
                    for (int r = 0; r < nTok; r++) { double dv = w[r * frames + f] - mean; var += dv * dv; }
                    double std = Math.Sqrt(var / nTok);
                    for (int r = 0; r < nTok; r++) w[r * frames + f] = (w[r * frames + f] - mean) / std;
                }
                weights[h] = MedianFilter(w, nTok, frames, cfg.MedianFilterWidth);
            }

            // Mean over heads, rows sot.Length .. nTok-2 (the no-timestamps row through the last text token).
            int first = sot.Length, n = nTok - 1 - first;
            var cost = new double[n * frames];
            for (int r = 0; r < n; r++)
                for (int f = 0; f < frames; f++)
                {
                    double s = 0;
                    for (int h = 0; h < heads; h++) s += weights[h][(first + r) * frames + f];
                    cost[r * frames + f] = -s / heads;
                }
            var (textIdx, timeIdx) = Dtw(cost, n, frames);

            var (words, wordTokens) = SplitToWordTokens(tok, cfg, textTokens.Append(cfg.EndOfText).ToList(), language);
            if (wordTokens.Count <= 1) return new List<WordTiming>();
            var boundaries = new List<int> { 0 };
            int acc = 0;
            for (int i = 0; i < wordTokens.Count - 1; i++) { acc += wordTokens[i].Count; boundaries.Add(acc); }

            var jumpTimes = new List<double>();
            for (int i = 0; i < textIdx.Length; i++)
                if (i == 0 || textIdx[i] != textIdx[i - 1]) jumpTimes.Add(timeIdx[i] / TokensPerSecond);

            var result = new List<WordTiming>();
            for (int i = 0; i + 1 < boundaries.Count; i++)
            {
                int a = boundaries[i], b = boundaries[i + 1];
                double p = 0;
                int count = 0;
                for (int k = a; k < b && k < tokenProbs.Length; k++) { p += tokenProbs[k]; count++; }
                result.Add(new WordTiming
                {
                    Word = words[i], Tokens = wordTokens[i],
                    Start = a < jumpTimes.Count ? jumpTimes[a] : jumpTimes[^1],
                    End = b < jumpTimes.Count ? jumpTimes[b] : jumpTimes[^1],
                    Probability = count > 0 ? p / count : 0,
                });
            }
            return result;
        }

        /// <summary>A median filter of odd width along the frame axis, reflect-padded as torch does.</summary>
        private static double[] MedianFilter(double[] x, int rows, int cols, int width)
        {
            if (width <= 1 || cols <= width / 2) return x;
            int pad = width / 2;
            var r = new double[x.Length];
            var window = new double[width];
            for (int row = 0; row < rows; row++)
                for (int c = 0; c < cols; c++)
                {
                    for (int k = -pad; k <= pad; k++)
                    {
                        int j = c + k;
                        if (j < 0) j = -j;
                        if (j >= cols) j = 2 * (cols - 1) - j;
                        window[k + pad] = x[row * cols + j];
                    }
                    Array.Sort(window);
                    r[row * cols + c] = window[pad];
                }
            return r;
        }

        private static (int[] TextIdx, int[] TimeIdx) Dtw(double[] x, int n, int m)
        {
            var cost = new double[(n + 1) * (m + 1)];
            var trace = new sbyte[(n + 1) * (m + 1)];
            Array.Fill(cost, double.PositiveInfinity);
            Array.Fill(trace, (sbyte)-1);
            cost[0] = 0;
            for (int j = 1; j <= m; j++)
                for (int i = 1; i <= n; i++)
                {
                    double c0 = cost[(i - 1) * (m + 1) + j - 1];
                    double c1 = cost[(i - 1) * (m + 1) + j];
                    double c2 = cost[i * (m + 1) + j - 1];
                    double c;
                    sbyte t;
                    if (c0 < c1 && c0 < c2) { c = c0; t = 0; }
                    else if (c1 < c0 && c1 < c2) { c = c1; t = 1; }
                    else { c = c2; t = 2; }
                    cost[i * (m + 1) + j] = x[(i - 1) * m + j - 1] + c;
                    trace[i * (m + 1) + j] = t;
                }
            for (int j = 0; j <= m; j++) trace[j] = 2;
            for (int i = 0; i <= n; i++) trace[i * (m + 1)] = 1;
            var path = new List<(int, int)>();
            int ii = n, jj = m;
            while (ii > 0 || jj > 0)
            {
                path.Add((ii - 1, jj - 1));
                switch (trace[ii * (m + 1) + jj])
                {
                    case 0: ii--; jj--; break;
                    case 1: ii--; break;
                    case 2: jj--; break;
                    default: throw new InvalidOperationException("DTW backtrace reached an unset cell.");
                }
            }
            path.Reverse();
            return (path.Select(p => p.Item1).ToArray(), path.Select(p => p.Item2).ToArray());
        }

        private static (List<string> Words, List<List<int>> Tokens) SplitToWordTokens(WhisperTokenizer tok, WhisperConfig cfg,
            List<int> tokens, string language)
        {
            var (subwords, subTokens) = SplitOnUnicode(tok, tokens);
            if (language != null && UnicodeSplitLanguages.Contains(language)) return (subwords, subTokens);
            var words = new List<string>();
            var wordTokens = new List<List<int>>();
            for (int i = 0; i < subwords.Count; i++)
            {
                string sub = subwords[i];
                bool special = subTokens[i][0] >= cfg.EndOfText;
                bool withSpace = sub.StartsWith(' ');
                string stripped = sub.Trim();
                bool punctuation = stripped.Length > 0 && IsPythonPunctuation(stripped);
                if (special || withSpace || punctuation || words.Count == 0)
                {
                    words.Add(sub);
                    wordTokens.Add(new List<int>(subTokens[i]));
                }
                else
                {
                    words[^1] += sub;
                    wordTokens[^1].AddRange(subTokens[i]);
                }
            }
            return (words, wordTokens);
        }

        /// <summary>"s.strip() in string.punctuation": a run that is one contiguous slice of ASCII punctuation.</summary>
        private static bool IsPythonPunctuation(string s)
        {
            const string punct = "!\"#$%&'()*+,-./:;<=>?@[\\]^_`{|}~";
            return punct.Contains(s, StringComparison.Ordinal);
        }

        private static (List<string>, List<List<int>>) SplitOnUnicode(WhisperTokenizer tok, List<int> tokens)
        {
            string full = tok.DecodeWithTimestamps(tokens);
            const char replacement = '\ufffd';
            var words = new List<string>();
            var wordTokens = new List<List<int>>();
            var current = new List<int>();
            int offset = 0;
            foreach (int t in tokens)
            {
                current.Add(t);
                string decoded = tok.DecodeWithTimestamps(current);
                int bad = decoded.IndexOf(replacement);
                if (bad < 0 || (offset + bad < full.Length && full[offset + bad] == replacement))
                {
                    words.Add(decoded);
                    wordTokens.Add(current);
                    current = new List<int>();
                    offset += decoded.Length;
                }
            }
            return (words, wordTokens);
        }

        private static void MergePunctuations(List<WordTiming> alignment)
        {
            int i = alignment.Count - 2, j = alignment.Count - 1;
            while (i >= 0)
            {
                var previous = alignment[i];
                var following = alignment[j];
                if (previous.Word.StartsWith(' ') &&
                    PrependPunctuations.Contains(previous.Word.Trim(), StringComparison.Ordinal))
                {
                    following.Word = previous.Word + following.Word;
                    following.Tokens = previous.Tokens.Concat(following.Tokens).ToList();
                    previous.Word = "";
                    previous.Tokens = new List<int>();
                }
                else j = i;
                i--;
            }
            i = 0;
            j = 1;
            while (j < alignment.Count)
            {
                var previous = alignment[i];
                var following = alignment[j];
                if (!previous.Word.EndsWith(' ') &&
                    AppendPunctuations.Contains(following.Word, StringComparison.Ordinal))
                {
                    previous.Word += following.Word;
                    previous.Tokens = previous.Tokens.Concat(following.Tokens).ToList();
                    following.Word = "";
                    following.Tokens = new List<int>();
                }
                else i = j;
                j++;
            }
        }
    }
}
