// Copyright (c) Zhongkai Fu. All rights reserved.
// https://github.com/zhongkaifu/TensorSharp
//
// This file is part of TensorSharp.
//
// TensorSharp is licensed under the BSD-3-Clause license found in the LICENSE file in the root directory of this source tree.
using System;
using System.Text;
using System.Text.RegularExpressions;

namespace TensorSharp.Models.Audio
{
    /// <summary>
    /// A TTS front end for English: numerals and the arithmetic signs between them written out as words, so a speech model
    /// that reads words well and digits poorly says them (LFM3-Omni's TTS says "366 days" as "state-sex days", "42,925" as
    /// "worthy", "37 × 43 = 1591" as noise). Integers with or without thousands separators (42,925 -> forty-two thousand
    /// nine hundred twenty-five; 13 digits or more digit by digit), decimals (3.14 -> three point one four), ordinals (29th
    /// -> twenty-ninth), a leading minus (-5 -> minus five), percentages (50% -> fifty percent), degrees (18 °C -> eighteen
    /// degrees Celsius) and × ÷ = + − between numbers (times, divided by, equals, plus, minus). A number inside a word (x86,
    /// mp3, v2) is left as it is; text without digits is returned as it is.
    /// </summary>
    public static class EnglishNumberReading
    {
        private static readonly string[] Ones =
        {
            "zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten", "eleven", "twelve",
            "thirteen", "fourteen", "fifteen", "sixteen", "seventeen", "eighteen", "nineteen",
        };

        private static readonly string[] Tens = { "", "", "twenty", "thirty", "forty", "fifty", "sixty", "seventy", "eighty", "ninety" };

        private static readonly (long Value, string Name)[] Scales =
        {
            (1_000_000_000_000, "trillion"), (1_000_000_000, "billion"), (1_000_000, "million"), (1_000, "thousand"),
        };

        // A sign between two numbers (spaces allowed): 2 × 4, 3+8, 10 ÷ 2, 1 = 1, 5 − 3.
        private static readonly Regex Sign = new(@"(?<=\d|\d\))\s*([×xX÷=+−])\s*(?=\(?-?\d)", RegexOptions.Compiled);
        private static readonly Regex Degrees = new(@"(?<=\d)\s*°\s*([CF])?\b", RegexOptions.Compiled);
        private static readonly Regex Number = new(
            @"(?<![\w.,])(-)?(\d{1,3}(?:,\d{3})+|\d+)(?:\.(\d+))?(st|nd|rd|th)?(%)?(?![\w])", RegexOptions.Compiled);

        /// <summary>The text with its numbers and the signs between them as English words.</summary>
        public static string Read(string text)
        {
            if (string.IsNullOrEmpty(text) || !HasDigit(text)) return text ?? string.Empty;
            string t = Sign.Replace(text, m => m.Groups[1].Value switch
            {
                "×" or "x" or "X" => " times ",
                "÷" => " divided by ",
                "=" => " equals ",
                "+" => " plus ",
                _ => " minus ",
            });
            t = Degrees.Replace(t, m => m.Groups[1].Value switch
            {
                "C" => " degrees Celsius",
                "F" => " degrees Fahrenheit",
                _ => " degrees",
            });
            t = Number.Replace(t, m =>
            {
                string digits = m.Groups[2].Value.Replace(",", "");
                var sb = new StringBuilder();
                if (m.Groups[1].Success) sb.Append("minus ");
                string whole = digits.Length >= 13 || !long.TryParse(digits, out long value) ? DigitByDigit(digits) : Cardinal(value);
                if (m.Groups[4].Success && !m.Groups[3].Success) whole = Ordinal(whole);
                sb.Append(whole);
                if (m.Groups[3].Success) sb.Append(" point ").Append(DigitByDigit(m.Groups[3].Value));
                if (m.Groups[5].Success) sb.Append(" percent");
                return sb.ToString();
            });
            return Regex.Replace(t, @"[ \t]{2,}", " ");
        }

        private static bool HasDigit(string text)
        {
            foreach (char c in text)
                if (c >= '0' && c <= '9') return true;
            return false;
        }

        /// <summary>An integer below 10^13 in words (1,591 -> one thousand five hundred ninety-one).</summary>
        public static string Cardinal(long value)
        {
            if (value < 0) return "minus " + Cardinal(-value);
            if (value < 1000) return BelowThousand((int)value);
            var parts = new StringBuilder();
            long rest = value;
            foreach (var (scale, name) in Scales)
            {
                if (rest < scale) continue;
                if (parts.Length > 0) parts.Append(' ');
                parts.Append(BelowThousand((int)(rest / scale))).Append(' ').Append(name);
                rest %= scale;
            }
            if (rest > 0) parts.Append(' ').Append(BelowThousand((int)rest));
            return parts.ToString();
        }

        private static string BelowThousand(int n)
        {
            if (n < 20) return Ones[n];
            if (n < 100) return Tens[n / 10] + (n % 10 == 0 ? "" : "-" + Ones[n % 10]);
            return Ones[n / 100] + " hundred" + (n % 100 == 0 ? "" : " " + BelowThousand(n % 100));
        }

        private static string DigitByDigit(string digits)
        {
            var sb = new StringBuilder();
            foreach (char c in digits)
            {
                if (c < '0' || c > '9') continue;
                if (sb.Length > 0) sb.Append(' ');
                sb.Append(Ones[c - '0']);
            }
            return sb.ToString();
        }

        /// <summary>The ordinal of a cardinal's words (twenty-nine -> twenty-ninth, one hundred -> one hundredth).</summary>
        private static string Ordinal(string cardinal)
        {
            int cut = Math.Max(cardinal.LastIndexOf(' '), cardinal.LastIndexOf('-')) + 1;
            string head = cardinal.Substring(0, cut), last = cardinal.Substring(cut);
            string ordinal = last switch
            {
                "one" => "first", "two" => "second", "three" => "third", "five" => "fifth", "eight" => "eighth",
                "nine" => "ninth", "twelve" => "twelfth",
                _ when last.EndsWith("y", StringComparison.Ordinal) => last.Substring(0, last.Length - 1) + "ieth",
                _ => last + "th",
            };
            return head + ordinal;
        }
    }
}
