// Copyright (c) Zhongkai Fu. All rights reserved.
// https://github.com/zhongkaifu/TensorSharp
//
// This file is part of TensorSharp.
//
// TensorSharp is licensed under the BSD-3-Clause license found in the LICENSE file in the root directory of this source tree.
//
// Mandarin text to Kokoro phonemes, as Kokoro-82M v1.0's pipeline produces them
// (misaki.zh.ZHG2P without the v1.1 frontend):
//
//   cn2an.transform(text, "an2cn")       Arabic numerals read out (dates digit by digit, %, fractions, ranges)
//   ZHG2P.map_punctuation                Chinese punctuation to the ASCII marks the model knows
//   for each run of CJK characters:
//     jieba.lcut                         words: dictionary DAG with the max-probability route, the
//                                        HMM (Viterbi over B/M/E/S) for runs of unknown characters
//     pypinyin.lazy_pinyin (TONE3)       per word: forward maximum matching on the phrase dictionary,
//                                        then each character's most common reading
//     misaki pinyin_to_ipa + retone      a table, one entry per syllable, made by misaki itself
//   words joined by spaces, everything outside CJK runs passed through, U+032F removed.
//
// Every table comes from the GGUF (eng/audio-models/kokoro_to_gguf.py writes them from the
// reference libraries), so this file holds only the algorithms.
using System;
using System.Collections.Generic;
using System.Globalization;
using System.Linq;
using System.Text;
using System.Text.RegularExpressions;
using TensorSharp.Runtime;

namespace TensorSharp.Models.Kokoro
{
    internal sealed class KokoroChineseG2P
    {
        private const double MinFloat = -3.14e100;

        private readonly Dictionary<string, int> _freq;   // words and every prefix (0 for prefixes only)
        private readonly double _logTotal;
        private readonly double[] _start = new double[4];
        private readonly double[,] _trans = new double[4, 4];
        private readonly Dictionary<char, double>[] _emit = new Dictionary<char, double>[4];
        private readonly Dictionary<string, string> _charPinyin;
        private readonly Dictionary<string, string[]> _phrasePinyin;
        private readonly HashSet<string> _phrasePrefixes;
        private readonly Dictionary<string, string> _ipa;

        public static bool IsAvailable(GgufFile gguf) => gguf.GetStringArray("kokoro.zh.jieba.words") != null;

        public KokoroChineseG2P(GgufFile gguf)
        {
            string[] words = gguf.GetStringArray("kokoro.zh.jieba.words");
            int[] freqs = gguf.GetInt32Array("kokoro.zh.jieba.freqs");
            _freq = new Dictionary<string, int>(words.Length * 2);
            long total = 0;
            for (int i = 0; i < words.Length; i++)
            {
                string w = words[i];
                _freq[w] = freqs[i];
                total += freqs[i];
                for (int k = 1; k < w.Length; k++)
                {
                    string frag = w.Substring(0, k);
                    if (!_freq.ContainsKey(frag)) _freq[frag] = 0;
                }
            }
            _logTotal = Math.Log(total);

            float[] start = gguf.GetFloatArray("kokoro.zh.jieba.hmm.start");
            float[] trans = gguf.GetFloatArray("kokoro.zh.jieba.hmm.trans");
            for (int s = 0; s < 4; s++)
            {
                _start[s] = Finite(start[s]);
                for (int t = 0; t < 4; t++) _trans[s, t] = Finite(trans[s * 4 + t]);
                string state = "BMES"[s].ToString();
                string[] chars = gguf.GetStringArray($"kokoro.zh.jieba.hmm.emit.{state}.chars");
                float[] logp = gguf.GetFloatArray($"kokoro.zh.jieba.hmm.emit.{state}.logp");
                _emit[s] = new Dictionary<char, double>(chars.Length);
                for (int i = 0; i < chars.Length; i++)
                    if (chars[i].Length == 1) _emit[s][chars[i][0]] = Finite(logp[i]);
            }

            string[] pc = gguf.GetStringArray("kokoro.zh.pinyin.chars");
            string[] pr = gguf.GetStringArray("kokoro.zh.pinyin.readings");
            _charPinyin = new Dictionary<string, string>(pc.Length);
            for (int i = 0; i < pc.Length; i++) _charPinyin[pc[i]] = pr[i];
            string[] phrases = gguf.GetStringArray("kokoro.zh.pinyin.phrases");
            string[] readings = gguf.GetStringArray("kokoro.zh.pinyin.phrase_readings");
            _phrasePinyin = new Dictionary<string, string[]>(phrases.Length);
            _phrasePrefixes = new HashSet<string>();
            for (int i = 0; i < phrases.Length; i++)
            {
                _phrasePinyin[phrases[i]] = readings[i].Split(' ');
                for (int k = 1; k <= phrases[i].Length; k++) _phrasePrefixes.Add(phrases[i].Substring(0, k));
            }
            string[] syl = gguf.GetStringArray("kokoro.zh.ipa.syllables");
            string[] ipa = gguf.GetStringArray("kokoro.zh.ipa.ipa");
            _ipa = new Dictionary<string, string>(syl.Length);
            for (int i = 0; i < syl.Length; i++) _ipa[syl[i]] = ipa[i];
        }

        private static double Finite(float v) => float.IsNegativeInfinity(v) ? MinFloat : v;

        private static readonly Regex ZhRuns = new("[一-鿿]+|[^一-鿿]+", RegexOptions.Compiled);

        /// <summary>ZHG2P.__call__ for Kokoro-82M v1.0: text to phonemes.</summary>
        public string Phonemize(string text)
        {
            if (string.IsNullOrWhiteSpace(text)) return string.Empty;
            text = ChineseNumerals.Transform(text);
            text = MapPunctuation(text);
            if (text.Length == 0) return string.Empty;
            bool isZh = text[0] >= '一' && text[0] <= '鿿';
            var result = new StringBuilder();
            foreach (Match m in ZhRuns.Matches(text))
            {
                string segment = m.Value;
                if (isZh)
                    segment = string.Join(" ", Cut(segment).Select(WordToIpa));
                result.Append(segment);
                isZh = !isZh;
            }
            return result.ToString().Replace("̯", "");
        }

        /// <summary>ZHG2P.map_punctuation.</summary>
        public static string MapPunctuation(string text)
        {
            text = text.Replace("、", ", ").Replace("，", ", ");
            text = text.Replace("。", ". ").Replace("．", ". ");
            text = text.Replace("！", "! ");
            text = text.Replace("：", ": ");
            text = text.Replace("；", "; ");
            text = text.Replace("？", "? ");
            text = text.Replace("«", " “").Replace("»", "” ");
            text = text.Replace("《", " “").Replace("》", "” ");
            text = text.Replace("「", " “").Replace("」", "” ");
            text = text.Replace("【", " “").Replace("】", "” ");
            text = text.Replace("（", " (").Replace("）", ") ");
            return text.Trim();
        }

        private string WordToIpa(string word)
        {
            var sb = new StringBuilder();
            foreach (string py in LazyPinyin(word))
                if (_ipa.TryGetValue(py, out string ipa)) sb.Append(ipa);
            return sb.ToString();
        }

        // ---------------------------------------------------------------- pypinyin
        /// <summary>lazy_pinyin(word, Style.TONE3, neutral_tone_with_five=True) for a run of CJK characters.</summary>
        internal List<string> LazyPinyin(string word)
        {
            var result = new List<string>();
            foreach (string piece in MaxMatch(word))
            {
                if (_phrasePinyin.TryGetValue(piece, out string[] readings))
                    result.AddRange(readings);
                else
                    foreach (char ch in piece)
                        result.Add(_charPinyin.TryGetValue(ch.ToString(), out string py) ? py : ch.ToString());
            }
            return result;
        }

        /// <summary>pypinyin.seg.mmseg.Seg.cut (no_non_phrases=False): forward maximum matching.</summary>
        private IEnumerable<string> MaxMatch(string text)
        {
            string remain = text;
            while (remain.Length > 0)
            {
                string matched = "";
                string lastValid = "";
                int lastValidIndex = 0;
                bool broke = false;
                for (int index = 0; index < remain.Length; index++)
                {
                    string word = remain.Substring(0, index + 1);
                    if (_phrasePrefixes.Contains(word))
                    {
                        matched = word;
                        lastValid = word;
                        lastValidIndex = index + 1;
                    }
                    else
                    {
                        if (lastValid.Length > 0)
                        {
                            yield return lastValid;
                            remain = remain.Substring(lastValidIndex);
                        }
                        else if (matched.Length > 0)
                        {
                            yield return matched;
                            remain = remain.Substring(matched.Length);
                        }
                        else
                        {
                            yield return remain.Substring(0, 1);
                            remain = remain.Substring(1);
                        }
                        broke = true;
                        break;
                    }
                }
                if (!broke)
                {
                    if (lastValid.Length > 0)
                    {
                        yield return lastValid;
                        remain = remain.Substring(lastValidIndex);
                    }
                    else
                    {
                        yield return remain;
                        break;
                    }
                }
            }
        }

        // ---------------------------------------------------------------- jieba
        private static bool IsJiebaHan(char c) =>
            (c >= '一' && c <= '鿕') || (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') ||
            (c >= '0' && c <= '9') || c is '+' or '#' or '&' or '.' or '_' or '%' or '-';

        /// <summary>jieba.lcut(sentence, cut_all=False, HMM=True).</summary>
        internal List<string> Cut(string sentence)
        {
            var words = new List<string>();
            int i = 0;
            while (i < sentence.Length)
            {
                int j = i;
                bool han = IsJiebaHan(sentence[i]);
                while (j < sentence.Length && IsJiebaHan(sentence[j]) == han) j++;
                string block = sentence.Substring(i, j - i);
                if (han) words.AddRange(CutDag(block));
                else
                {
                    // re_skip "(\r\n|\s)" splits; the rest is yielded one character at a time.
                    for (int k = 0; k < block.Length; k++)
                    {
                        if (block[k] == '\r' && k + 1 < block.Length && block[k + 1] == '\n') { words.Add("\r\n"); k++; }
                        else words.Add(block[k].ToString());
                    }
                }
                i = j;
            }
            return words;
        }

        private IEnumerable<string> CutDag(string s)
        {
            int n = s.Length;
            var dag = new List<int>[n];
            for (int k = 0; k < n; k++)
            {
                var list = new List<int>();
                int i = k;
                string frag = s.Substring(k, 1);
                while (i < n && _freq.TryGetValue(frag, out int f))
                {
                    if (f > 0) list.Add(i);
                    i++;
                    if (i < n) frag = s.Substring(k, i - k + 1);
                }
                if (list.Count == 0) list.Add(k);
                dag[k] = list;
            }
            var route = new (double Logp, int End)[n + 1];
            route[n] = (0, 0);
            for (int idx = n - 1; idx >= 0; idx--)
            {
                (double, int) best = (double.NegativeInfinity, -1);
                foreach (int x in dag[idx])
                {
                    int f = _freq.TryGetValue(s.Substring(idx, x - idx + 1), out int v) ? v : 0;
                    double score = Math.Log(f == 0 ? 1 : f) - _logTotal + route[x + 1].Logp;
                    if (score > best.Item1 || (score == best.Item1 && x > best.Item2)) best = (score, x);
                }
                route[idx] = best;
            }

            var output = new List<string>();
            var buf = new StringBuilder();
            void FlushBuf()
            {
                if (buf.Length == 0) return;
                string b = buf.ToString();
                if (b.Length == 1) output.Add(b);
                else if (!_freq.TryGetValue(b, out int f) || f == 0) output.AddRange(HmmCut(b));
                else foreach (char c in b) output.Add(c.ToString());
                buf.Clear();
            }
            int pos = 0;
            while (pos < n)
            {
                int y = route[pos].End + 1;
                string word = s.Substring(pos, y - pos);
                if (y - pos == 1) buf.Append(word);
                else
                {
                    FlushBuf();
                    output.Add(word);
                }
                pos = y;
            }
            FlushBuf();
            return output;
        }

        /// <summary>jieba.finalseg.cut on a buffer of characters the dictionary does not join.</summary>
        private IEnumerable<string> HmmCut(string sentence)
        {
            var output = new List<string>();
            int i = 0;
            while (i < sentence.Length)
            {
                int j = i;
                bool han = sentence[i] >= '一' && sentence[i] <= '鿕';
                while (j < sentence.Length && (sentence[j] >= '一' && sentence[j] <= '鿕') == han) j++;
                string block = sentence.Substring(i, j - i);
                if (han) output.AddRange(Viterbi(block));
                else output.AddRange(Regex.Split(block, @"([a-zA-Z0-9]+(?:\.\d+)?%?)").Where(x => x.Length > 0));
                i = j;
            }
            return output;
        }

        private static readonly int[][] PrevStatus = { new[] { 3, 2 }, new[] { 1, 0 }, new[] { 0, 1 }, new[] { 2, 3 } }; // B: E,S  M: M,B  E: B,M  S: S,E
        private const int B = 0, M = 1, E = 2, S = 3;

        private IEnumerable<string> Viterbi(string obs)
        {
            int n = obs.Length;
            var v = new double[n, 4];
            var back = new int[n, 4];
            for (int y = 0; y < 4; y++)
                v[0, y] = _start[y] + Emit(y, obs[0]);
            for (int t = 1; t < n; t++)
                for (int y = 0; y < 4; y++)
                {
                    double em = Emit(y, obs[t]);
                    double best = double.NegativeInfinity;
                    int bestState = -1;
                    foreach (int y0 in PrevStatus[y])
                    {
                        double p = v[t - 1, y0] + _trans[y0, y] + em;
                        // Python's max over (prob, state) tuples: ties go to the larger state letter.
                        if (bestState < 0 || p > best || (p == best && "BMES"[y0] > "BMES"[bestState]))
                        {
                            best = p;
                            bestState = y0;
                        }
                    }
                    v[t, y] = best;
                    back[t, y] = bestState;
                }
            int last = v[n - 1, E] > v[n - 1, S] || (v[n - 1, E] == v[n - 1, S] && 'E' > 'S') ? E : S;
            var states = new int[n];
            states[n - 1] = last;
            for (int t = n - 1; t > 0; t--) states[t - 1] = back[t, states[t]];

            var words = new List<string>();
            int begin = 0, next = 0;
            for (int i = 0; i < n; i++)
            {
                if (states[i] == B) begin = i;
                else if (states[i] == E) { words.Add(obs.Substring(begin, i + 1 - begin)); next = i + 1; }
                else if (states[i] == S) { words.Add(obs[i].ToString()); next = i + 1; }
            }
            if (next < n) words.Add(obs.Substring(next));
            return words;
        }

        private double Emit(int state, char c) => _emit[state].TryGetValue(c, out double p) ? p : MinFloat;
    }

    /// <summary>cn2an.transform(text, "an2cn"): Arabic numerals in running text read out in Chinese.</summary>
    internal static class ChineseNumerals
    {
        private const string MeasureWords =
            "斤|克|千克|公斤|吨|米|厘米|毫米|公里|升|毫升|元|角|分|个|只|条|张|块|瓶|杯|份|本|" +
            "辆|台|匹|头|位|亩|小时|分钟|秒|天|半";
        private static readonly string[] Digits = { "零", "一", "二", "三", "四", "五", "六", "七", "八", "九" };
        private static readonly string[] Units = { "", "十", "百", "千", "万", "十", "百", "千", "亿", "十", "百", "千", "万", "十", "百", "千" };

        private static readonly Regex Range = new($@"\d+(\.\d+)?-\d+(\.\d+)?(?=({MeasureWords}))");
        private static readonly Regex Date = new(@"(\d{2,4}年)?(\d{1,2}月)?(\d{1,2}日)?");
        private static readonly Regex Fraction = new(@"\d+/\d+");
        private static readonly Regex Percent = new(@"-?(\d+\.)?\d+%");
        private static readonly Regex Celsius = new(@"\d+℃");
        private static readonly Regex Number = new(@"-?(\d+\.)?\d+");
        private static readonly Regex DigitsRun = new(@"\d+");
        private static readonly Regex YearDigits = new(@"\d+(?=年)");
        private static readonly Regex WanQian = new("([万亿])零([一二三四五六七八九壹贰叁肆伍陆柒捌玖][千仟])");

        public static string Transform(string input)
        {
            input = Range.Replace(input, m => Safe(m.Value, v =>
            {
                string[] parts = v.Split('-');
                return $"{Low(parts[0])}到{Low(parts[1])}";
            }));
            input = Date.Replace(input, m => m.Value.Length == 0 ? "" : Safe(m.Value, v =>
            {
                v = YearDigits.Replace(v, d => Direct(d.Value));
                return DigitsRun.Replace(v, d => Low(d.Value));
            }));
            input = Fraction.Replace(input, m => Safe(m.Value, v =>
            {
                string[] parts = DigitsRun.Replace(v, d => Low(d.Value)).Split('/');
                return $"{parts[1]}分之{parts[0]}";
            }));
            input = Percent.Replace(input, m => Safe(m.Value, v => "百分之" + Low(v[..^1])));
            input = Celsius.Replace(input, m => Safe(m.Value, v => Low(v[..^1]) + "摄氏度"));
            return Number.Replace(input, m => Safe(m.Value, Low));
        }

        /// <summary>An exception in a conversion leaves the text as it was (cn2an warns and returns its input).</summary>
        private static string Safe(string value, Func<string, string> convert)
        {
            try { return convert(value); }
            catch (FormatException) { return value; }
            catch (OverflowException) { return value; }
        }

        private static string Normalize(string s)
        {
            var sb = new StringBuilder(s.Length);
            foreach (char c in s)
                sb.Append(char.IsDigit(c) ? (char)('0' + (int)char.GetNumericValue(c)) : c);
            return sb.ToString();
        }

        internal static string Direct(string s)
        {
            s = Normalize(s);
            var sb = new StringBuilder();
            foreach (char c in s) sb.Append(c == '.' ? "点" : Digits[c - '0']);
            return sb.ToString();
        }

        /// <summary>An2Cn.an2cn(s, "low").</summary>
        internal static string Low(string s)
        {
            s = Normalize(s);
            string sign = "";
            if (s.StartsWith('-')) { sign = "负"; s = s[1..]; }
            string[] split = s.Split('.');
            if (split.Length > 2 || s.Length == 0) throw new FormatException(s);
            string output = Integer(split[0]);
            if (split.Length == 2)
            {
                string dec = split[1].Length > 16 ? split[1][..16] : split[1];
                if (dec.Length > 0)
                {
                    var sb = new StringBuilder("点");
                    foreach (char c in dec) sb.Append(Digits[c - '0']);
                    output += sb.ToString();
                }
            }
            return sign + output;
        }

        private static string Integer(string data)
        {
            if (data.Length == 0) throw new FormatException("empty integer part");
            data = data.TrimStart('0');
            if (data.Length == 0) data = "0";
            int len = data.Length;
            if (len > Units.Length) throw new OverflowException($"cn2an reads at most {Units.Length} digits");
            var sb = new StringBuilder();
            for (int i = 0; i < len; i++)
            {
                int d = data[i] - '0';
                if (d != 0)
                    sb.Append(Digits[d]).Append(Units[len - i - 1]);
                else
                {
                    if ((len - i - 1) % 4 == 0)
                        sb.Append(Digits[d]).Append(Units[len - i - 1]);
                    if (i > 0 && sb.Length > 0 && sb[^1] != '零')
                        sb.Append(Digits[d]);
                }
            }
            string output = sb.ToString().Replace("零零", "零").Replace("零万", "万").Replace("零亿", "亿").Replace("亿万", "亿")
                .Trim('零');
            output = WanQian.Replace(output, "$1$2");
            if (output.StartsWith("一十", StringComparison.Ordinal)) output = output[1..];
            if (output.Length == 0) output = "零";
            return output;
        }
    }
}
