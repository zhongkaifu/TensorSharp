// Copyright (c) Zhongkai Fu. All rights reserved.
// https://github.com/zhongkaifu/TensorSharp
//
// This file is part of TensorSharp.
//
// TensorSharp is licensed under the BSD-3-Clause license found in the LICENSE file in the root directory of this source tree.
using System.Text;
using System.Text.RegularExpressions;

namespace TensorSharp.Models.Audio
{
    /// <summary>
    /// A TTS front end for Chinese: the Arabic numerals of a text written out as Mandarin reads them, so a speech model that
    /// reads Chinese characters well and digits poorly (LFM3-Omni reads "2024年10月1日" and "199元" as noise) says them.
    /// Dates (2024-10-01, 2024年10月1日: the year digit by digit), clock times (3:30 -> 三点三十分), percentages
    /// (百分之五十), currency signs (¥ $ € £), decimals (三点一四), thousands separators, ranges (1-5 -> 一到五), "2" before a
    /// measure word (两个; 第2个 stays 第二个), plain integers (一万零五; codes and runs of 10 digits or more digit by digit) and the
    /// arithmetic signs (25 × 3 = 75 -> 二十五乘三等于七十五; 1+1 -> 一加一). Text without digits or signs is returned as it is.
    /// </summary>
    public static class ChineseNumberReading
    {
        private static readonly string[] Digits = { "零", "一", "二", "三", "四", "五", "六", "七", "八", "九" };

        private const string Measures = "个只位本件条张次天杯种份双台辆块人家点岁周名样句步页层门所座节套颗";

        private static readonly Regex Date = new(@"(?<!\d)(\d{4})\s*[-/.年]\s*(\d{1,2})\s*[-/.月]\s*(\d{1,2})(?!\d)\s*日?", RegexOptions.Compiled);
        private static readonly Regex Year = new(@"(?<!\d)(\d{4})(?=\s*年)", RegexOptions.Compiled);
        private static readonly Regex Time = new(@"(?<![\d.])(\d{1,2}):(\d{2})(?::(\d{2}))?(?![\d.])", RegexOptions.Compiled);
        private static readonly Regex Percent = new(@"(\d+(?:\.\d+)?)\s*[%％]", RegexOptions.Compiled);
        private static readonly Regex Currency = new(@"([¥￥$€£])\s*(\d+(?:\.\d+)?)", RegexOptions.Compiled);
        private static readonly Regex Grouped = new(@"(?<![\d.])\d{1,3}(?:,\d{3})+(?![\d,])", RegexOptions.Compiled);
        private static readonly Regex Range = new(@"(?<=\d)\s*[-~～—–]\s*(?=\d)", RegexOptions.Compiled);
        private static readonly Regex Decimal = new(@"(?<![\d.])(\d+)\.(\d+)(?![\d.])", RegexOptions.Compiled);
        private static readonly Regex Integer = new(@"\d+", RegexOptions.Compiled);

        /// <summary><paramref name="text"/> with its numerals read out in Chinese (full-width digits included).</summary>
        public static string Read(string text)
        {
            if (string.IsNullOrEmpty(text)) return text ?? string.Empty;
            var sb = new StringBuilder(text.Length);
            bool digits = false;
            foreach (char c in text)
            {
                char d = c is >= '０' and <= '９' ? (char)('0' + (c - '０')) : c == '．' ? '.' : c;
                if (d is >= '0' and <= '9') digits = true;
                sb.Append(d);
            }
            string t = sb.ToString();
            if (!digits) return t.IndexOfAny(Signs) < 0 ? text : ReadSigns(t);
            t = Date.Replace(t, m => DigitByDigit(m.Groups[1].Value) + "年" + Number(int.Parse(m.Groups[2].Value).ToString()) + "月" +
                Number(int.Parse(m.Groups[3].Value).ToString()) + "日");
            t = Year.Replace(t, m => DigitByDigit(m.Groups[1].Value));
            t = Time.Replace(t, m =>
            {
                int hour = int.Parse(m.Groups[1].Value), minute = int.Parse(m.Groups[2].Value);
                string s = Number(hour.ToString()) + "点" + (minute == 0 ? "" : (minute < 10 ? "零" : "") + Number(minute.ToString()) + "分");
                if (m.Groups[3].Success && int.Parse(m.Groups[3].Value) is int second and > 0) s += Number(second.ToString()) + "秒";
                return s;
            });
            t = Percent.Replace(t, m => "百分之" + Value(m.Groups[1].Value));
            t = Currency.Replace(t, m => Value(m.Groups[2].Value) + m.Groups[1].Value switch
            {
                "$" => "美元",
                "€" => "欧元",
                "£" => "英镑",
                _ => "元",
            });
            t = Grouped.Replace(t, m => m.Value.Replace(",", ""));
            t = Range.Replace(t, "到");
            t = Plus.Replace(t, "加");
            t = Decimal.Replace(t, m => Number(m.Groups[1].Value) + "点" + DigitByDigit(m.Groups[2].Value));
            t = Integer.Replace(t, m =>
            {
                int after = m.Index + m.Length;
                bool ordinal = m.Index > 0 && t[m.Index - 1] == '第';
                if (m.Value == "2" && !ordinal && after < t.Length && Measures.IndexOf(t[after]) >= 0) return "两";
                return Number(m.Value);
            });
            return ReadSigns(t);
        }

        private static readonly char[] Signs = { '=', '×', '÷', '≈', '≠', '≤', '≥' };
        private static readonly Regex Plus = new(@"(?<=\d)\s*\+\s*(?=\d)", RegexOptions.Compiled);

        /// <summary>Arithmetic signs as Mandarin reads them (× 乘, ÷ 除以, = 等于, ≈ 约等于, != and ≠ 不等于, ≤ ≥).</summary>
        private static string ReadSigns(string t)
        {
            if (t.IndexOfAny(Signs) < 0) return t;
            return t.Replace("!=", "不等于").Replace("==", "等于").Replace("≠", "不等于").Replace("≈", "约等于").Replace("≤", "小于等于")
                .Replace("≥", "大于等于").Replace("×", "乘").Replace("÷", "除以").Replace("=", "等于");
        }

        /// <summary>A decimal or an integer as Mandarin reads it.</summary>
        private static string Value(string s)
        {
            int dot = s.IndexOf('.');
            return dot < 0 ? Number(s) : Number(s.Substring(0, dot)) + "点" + DigitByDigit(s.Substring(dot + 1));
        }

        private static string DigitByDigit(string s)
        {
            var sb = new StringBuilder(s.Length);
            foreach (char c in s) sb.Append(Digits[c - '0']);
            return sb.ToString();
        }

        /// <summary>An integer as Mandarin reads it: 10 -> 十, 110 -> 一百一十, 1005 -> 一千零五, 100010000 -> 一亿零一万; a code
        /// (a leading zero) or a run of 10 digits or more (a phone number, an account) digit by digit.</summary>
        public static string Number(string s)
        {
            if (s.Length >= 10 || (s.Length > 1 && s[0] == '0')) return DigitByDigit(s);
            long n = long.Parse(s);
            if (n == 0) return Digits[0];
            var sb = new StringBuilder();
            bool started = false, zero = false;
            void Group(long g, string unit)
            {
                if (g == 0)
                {
                    if (started) zero = true;
                    return;
                }
                if (started && (zero || g < 1000)) sb.Append(Digits[0]);
                sb.Append(Four((int)g)).Append(unit);
                started = true;
                zero = false;
            }
            Group(n / 100_000_000, "亿");
            Group(n / 10_000 % 10_000, "万");
            Group(n % 10_000, "");
            string r = sb.ToString();
            return r.StartsWith("一十", System.StringComparison.Ordinal) ? r.Substring(1) : r;
        }

        private static readonly string[] Units = { "千", "百", "十", "" };

        private static string Four(int g)
        {
            var sb = new StringBuilder();
            bool any = false, zero = false;
            int[] d = { g / 1000, g / 100 % 10, g / 10 % 10, g % 10 };
            for (int i = 0; i < 4; i++)
            {
                if (d[i] == 0)
                {
                    if (any) zero = true;
                    continue;
                }
                if (zero) sb.Append(Digits[0]);
                sb.Append(Digits[d[i]]).Append(Units[i]);
                any = true;
                zero = false;
            }
            return sb.ToString();
        }
    }
}
