// Copyright (c) Zhongkai Fu. All rights reserved.
// https://github.com/zhongkaifu/TensorSharp
//
// This file is part of TensorSharp.
//
// TensorSharp is licensed under the BSD-3-Clause license found in the LICENSE file in the root directory of this source tree.

using TensorSharp.Models.Audio;
using TensorSharp.Models.Lfm3;

namespace InferenceWeb.Tests;

/// <summary>
/// LFM3-Omni's TTS prompt without a named language speaks the language the text is written in (by its script; Latin
/// letters keep the template's default, American English), and Chinese text reaches it with its numerals read out as
/// Mandarin says them (the model says digits in Chinese text as noise).
/// </summary>
public sealed class Lfm3TtsLanguageTests
{
    [Theory]
    [InlineData("请用一句话介绍一下北京。", "zh")]
    [InlineData("今天下雨了，我应该带什么？", "zh")]
    [InlineData("我想学习Python编程，应该从哪里开始？", "zh")]
    [InlineData("什么是machine learning？请用简单的话解释。", "zh")]
    [InlineData("臺灣的首都是哪裡？", "zh")]
    [InlineData("日本語のテキストです。", "ja")]
    [InlineData("東京は日本の首都です。", "ja")]
    [InlineData("안녕하세요, 반갑습니다.", "ko")]
    [InlineData("مرحبا بالعالم", "ar")]
    [InlineData("שלום עולם", "he")]
    [InlineData("Καλημέρα κόσμε", "el")]
    [InlineData("What is the capital of France?", null)]
    [InlineData("Quelle est la capitale de la France ?", null)]
    [InlineData("What does 你好 mean?", null)]
    [InlineData("2024-10-01 12:30", null)]
    [InlineData("", null)]
    [InlineData(null, null)]
    public void TheScriptNamesTheLanguage(string text, string language)
        => Assert.Equal(language, Lfm3SpeechPrompt.LanguageOfText(text));

    [Fact]
    public void TheTtsPromptSpeaksTheTextsLanguageWhenNoneIsNamed()
    {
        Assert.Contains("Perform TTS in Chinese using a female voice.", Lfm3SpeechPrompt.Tts("北京是中国的首都。"));
        Assert.Contains("Perform TTS in Japanese using a male voice.", Lfm3SpeechPrompt.Tts("東京は日本の首都です。", voice: "male"));
        Assert.Contains("Perform TTS in Korean using a female voice.", Lfm3SpeechPrompt.Tts("안녕하세요"));
        Assert.Contains("Perform TTS in American English using a female voice.", Lfm3SpeechPrompt.Tts("Hello there."));
        // A named language wins, a regional code by its language.
        Assert.Contains("Perform TTS in American English using a female voice.", Lfm3SpeechPrompt.Tts("北京", "en"));
        Assert.Contains("Perform TTS in Chinese using a female voice.", Lfm3SpeechPrompt.Tts("北京", "zh-CN"));
        Assert.Contains("Perform TTS in Klingon using a female voice.", Lfm3SpeechPrompt.Tts("北京", "Klingon"));
    }

    [Fact]
    public void ChineseTextIsGivenToTheTtsPromptWithItsNumeralsReadOut()
    {
        Assert.Equal("二零二四年十月一日是国庆节，我们放假七天。", Lfm3SpeechPrompt.SpokenForm("2024年10月1日是国庆节，我们放假七天。", null));
        Assert.Equal("原价一百九十九元", Lfm3SpeechPrompt.SpokenForm("原价199元", "zh"));
        // Not Chinese: as it is (English digits are the English TTS's to read).
        Assert.Equal("A leap year has 366 days.", Lfm3SpeechPrompt.SpokenForm("A leap year has 366 days.", null));
        Assert.Equal("原价199元", Lfm3SpeechPrompt.SpokenForm("原价199元", "en"));
    }

    [Theory]
    [InlineData("一年有365天", "一年有三百六十五天")]
    [InlineData("2024年10月1日", "二零二四年十月一日")]
    [InlineData("2024-10-01", "二零二四年十月一日")]
    [InlineData("只要159.2元", "只要一百五十九点二元")]
    [InlineData("下午3:30开会", "下午三点三十分开会")]
    [InlineData("08:05", "八点零五分")]
    [InlineData("12:00", "十二点")]
    [InlineData("增长了50%", "增长了百分之五十")]
    [InlineData("¥199", "一百九十九元")]
    [InlineData("$5", "五美元")]
    [InlineData("约21,000公里", "约二万一千公里")]
    [InlineData("1-5天", "一到五天")]
    [InlineData("2个人", "两个人")]
    [InlineData("第2名", "第二名")]
    [InlineData("2月", "二月")]
    [InlineData("10", "十")]
    [InlineData("110", "一百一十")]
    [InlineData("1005", "一千零五")]
    [InlineData("15000", "一万五千")]
    [InlineData("100010000", "一亿零一万")]
    [InlineData("0.5", "零点五")]
    [InlineData("电话13800138000", "电话一三八零零一三八零零零")]
    [InlineData("编号007", "编号零零七")]
    [InlineData("１２３", "一百二十三")]
    [InlineData("没有数字。", "没有数字。")]
    [InlineData("25 元 × 3 = 75 元", "二十五 元 乘 三 等于 七十五 元")]
    [InlineData("1+1=2", "一加一等于二")]
    [InlineData("a != b，x ≈ 3.5", "a 不等于 b，x 约等于 三点五")]
    [InlineData("C++语言", "C++语言")]
    public void ChineseNumeralsAreReadAsMandarinSaysThem(string text, string read)
        => Assert.Equal(read, ChineseNumberReading.Read(text));
}
