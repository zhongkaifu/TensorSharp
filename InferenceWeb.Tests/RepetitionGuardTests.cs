// Copyright (c) Zhongkai Fu. All rights reserved.
// https://github.com/zhongkaifu/TensorSharp
//
// This file is part of TensorSharp.
//
// TensorSharp is licensed under the BSD-3-Clause license found in the LICENSE file in the root directory of this source tree.
//
// TensorSharp is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the BSD-3-Clause License for more details.

using System;
using System.Collections.Generic;
using System.Globalization;
using System.Linq;
using System.Text;
using System.Text.RegularExpressions;
using TensorSharp.Runtime.Scheduling;
using Xunit;

namespace InferenceWeb.Tests;

/// <summary>
/// The engine's loop detector, on token streams shaped like the failure it was
/// written for and like the writing it must let through.
///
/// <para>
/// The failure: a 4-bit 9B model on a phone, writing XML inside a Python string,
/// fell into <c>","+","+","+"</c> and produced 230 copies of it over five and a half
/// minutes, ending only when the user tapped Stop. No sampling setting rules that
/// out for every model and every text, so the engine has to notice.
/// </para>
/// </summary>
public sealed class RepetitionGuardTests
{
    private static List<int> Repeat(IReadOnlyList<int> unit, int times, int prefixLength = 0)
    {
        var tokens = new List<int>();
        for (int i = 0; i < prefixLength; i++)
            tokens.Add(1000 + i);            // distinct, never periodic
        for (int r = 0; r < times; r++)
            tokens.AddRange(unit);
        return tokens;
    }

    [Fact]
    public void TheObservedLoop_ThreeTokenUnit_IsCaughtOnceItSpansTheMinimum()
    {
        // `","+"` -> three tokens. 42 copies is 126 tokens: not yet. 43 is 129: caught.
        var unit = new[] { 7, 8, 9 };
        List<int> notYet = Repeat(unit, 42, prefixLength: 50);
        Assert.False(RepetitionGuard.IsLooping(notYet, notYet.Count, out _, out _));

        List<int> caught = Repeat(unit, 43, prefixLength: 50);
        Assert.True(RepetitionGuard.IsLooping(caught, caught.Count, out int period, out int repeats));
        Assert.Equal(3, period);
        Assert.Equal(43, repeats);
    }

    [Fact]
    public void ASingleTokenRunNeeds128Copies()
    {
        List<int> short127 = Repeat(new[] { 5 }, 127, prefixLength: 10);
        Assert.False(RepetitionGuard.IsLooping(short127, short127.Count, out _, out _));

        List<int> run = Repeat(new[] { 5 }, 128, prefixLength: 10);
        Assert.True(RepetitionGuard.IsLooping(run, run.Count, out int period, out int repeats));
        Assert.Equal(1, period);
        Assert.Equal(128, repeats);
    }

    /// <summary>
    /// The corner the thresholds exist for: FOUR byte-identical copies of a long block
    /// is ordinary structured output — four empty paragraphs of OOXML, four identical
    /// table rows — and must survive. Eight in a row, with no variation at all, is not
    /// something a model writes on purpose.
    /// </summary>
    [Fact]
    public void ALongUnitNeedsEightCopies_AndTheLongestUnitCountsToo()
    {
        int[] paragraph = Enumerable.Range(200, 40).ToArray();
        foreach (int copies in new[] { 3, 4, 7 })
        {
            List<int> tolerated = Repeat(paragraph, copies, prefixLength: 5);
            Assert.False(RepetitionGuard.IsLooping(tolerated, tolerated.Count, out _, out _));
        }

        List<int> eight = Repeat(paragraph, 8, prefixLength: 5);
        Assert.True(RepetitionGuard.IsLooping(eight, eight.Count, out int period, out int repeats));
        Assert.Equal(40, period);
        Assert.Equal(8, repeats);

        int[] longest = Enumerable.Range(300, RepetitionGuard.MaxPeriod).ToArray();
        List<int> atLimit = Repeat(longest, RepetitionGuard.MinRepeats);
        Assert.True(RepetitionGuard.IsLooping(atLimit, atLimit.Count, out period, out _));
        Assert.Equal(RepetitionGuard.MaxPeriod, period);

        // A unit longer than the cap is not judged by this rule: twelve copies (780
        // tokens) are not a loop. Only the long-block rule can stop it, and that rule
        // needs 4,096 tokens of copy (see the block tests below).
        int[] tooLong = Enumerable.Range(300, RepetitionGuard.MaxPeriod + 1).ToArray();
        List<int> pastLimit = Repeat(tooLong, 12);
        Assert.False(RepetitionGuard.IsLooping(pastLimit, pastLimit.Count, out _, out _));
    }

    [Fact]
    public void WritingThatMerelyRepeatsShortStretchesIsLeftAlone()
    {
        // A wide markdown table's separator row: "| ---" ten times is 20 tokens.
        List<int> separator = Repeat(new[] { 30, 31 }, 10, prefixLength: 200);
        Assert.False(RepetitionGuard.IsLooping(separator, separator.Count, out _, out _));

        // A loop body that recurs with a varying line between copies is not periodic.
        var tokens = new List<int>();
        for (int i = 0; i < 60; i++)
        {
            tokens.AddRange(new[] { 40, 41, 42 });
            tokens.Add(500 + i);
        }
        Assert.False(RepetitionGuard.IsLooping(tokens, tokens.Count, out _, out _));

        // Fewer tokens than the minimum span can never be a loop, whatever they are.
        List<int> few = Repeat(new[] { 1 }, 100);
        Assert.False(RepetitionGuard.IsLooping(few, few.Count, out _, out _));
        Assert.False(RepetitionGuard.IsLooping(null, 0, out _, out _));
    }

    [Fact]
    public void OnlyTheCountedPrefixIsJudged_SoAnUnpublishedSpeculativeTailCannotTrigger()
    {
        // 127 copies then two more that a speculative step appended but the engine
        // has not yet accepted: judged at 127 the answer is no, at 129 it is yes.
        List<int> tokens = Repeat(new[] { 3 }, 129);
        Assert.False(RepetitionGuard.IsLooping(tokens, 127, out _, out _));
        Assert.True(RepetitionGuard.IsLooping(tokens, 129, out _, out int repeats));
        Assert.Equal(129, repeats);
    }

    [Fact]
    public void RepeatsAreCountedBackToWhereTheLoopBegan()
    {
        List<int> tokens = Repeat(new[] { 7, 8, 9 }, 230, prefixLength: 30);
        Assert.True(RepetitionGuard.IsLooping(tokens, tokens.Count, out int period, out int repeats));
        Assert.Equal(3, period);
        Assert.Equal(230, repeats);
    }

    [Fact]
    public void TheDescriptionQuotesTheUnitAndTheCount()
    {
        List<int> tokens = Repeat(new[] { 7, 8, 9 }, 43);
        Assert.True(RepetitionGuard.IsLooping(tokens, tokens.Count, out int period, out int repeats));

        string described = RepetitionGuard.Describe(tokens, tokens.Count, period, repeats,
            ids => string.Concat(ids.Select(id => id == 7 ? "\"," : id == 8 ? "\"" : "+")));
        Assert.Equal("the output repeated `\",\"+` 43 times in a row", described);

        // No tokenizer, or one that throws: still a sentence, never an exception.
        Assert.Equal("the output repeated a 3-token sequence 43 times in a row",
            RepetitionGuard.Describe(tokens, tokens.Count, period, repeats, null));
        Assert.Equal("the output repeated a 3-token sequence 43 times in a row",
            RepetitionGuard.Describe(tokens, tokens.Count, period, repeats, _ => throw new InvalidOperationException()));

        // Newlines are shown, not printed, and a long unit is cut.
        string multiline = RepetitionGuard.Describe(tokens, tokens.Count, period, repeats, _ => "a\nb");
        Assert.Contains("`a\\nb`", multiline);
        string longUnit = RepetitionGuard.Describe(tokens, tokens.Count, period, repeats, _ => new string('x', 100));
        Assert.Contains(new string('x', 48) + "…", longUnit);
    }

    [Fact]
    public void TheTextFormFindsTheSameLoopInCharacters()
    {
        string tail = "office_theme = '''<a:latin type=\"nn\" bon=\"Aa+" + string.Concat(Enumerable.Repeat("\",\"+", 230));
        Assert.True(RepetitionGuard.TryFindTextLoop(tail, out string unit, out int repeats));
        Assert.Equal("\",\"+", unit);
        Assert.Equal(230, repeats);

        // Seventy-eight characters, so the periodicity scan judges it rather than the
        // length guard turning it away before the scan is reached.
        Assert.False(RepetitionGuard.TryFindTextLoop(
            "a perfectly ordinary sentence, long enough to be scanned, with no loop in it at all.", out _, out _));
        Assert.False(RepetitionGuard.TryFindTextLoop(string.Empty, out _, out _));
        Assert.False(RepetitionGuard.TryFindTextLoop(null, out _, out _));
        // Sixty-three characters of one letter is under the minimum span; sixty-four is not.
        Assert.False(RepetitionGuard.TryFindTextLoop(new string('+', 63), out _, out _));
        Assert.True(RepetitionGuard.TryFindTextLoop(new string('+', 64), out unit, out repeats));
        Assert.Equal("+", unit);
        Assert.Equal(64, repeats);

        // A long unit is found too: a 64-token unit (the short rule's cap) can be
        // several hundred characters, and the description would otherwise fall back
        // to the vague sentence.
        string longUnit = new string('a', 200) + "|";
        Assert.True(RepetitionGuard.TryFindTextLoop(string.Concat(Enumerable.Repeat(longUnit, 4)), out unit, out repeats));
        Assert.Equal(longUnit, unit);
        Assert.Equal(4, repeats);
    }

    // ---- The long-block rule ------------------------------------------------------------------------

    /// <summary>
    /// The failure the second rule was written for. On TensorSharp.Server, LFM3-Omni 2B wrote an
    /// <c>apply_patch</c> body that repeated one 16-line markdown block word for word, 520 times,
    /// for 17 minutes. This is the same text, cut into ids by a stand-in tokenizer. The short
    /// rule never fires on it. The block rule stops it as soon as the exactly periodic tail fills
    /// <see cref="RepetitionGuard.MinBlockSpan"/>, at the next check.
    /// </summary>
    [Fact]
    public void TheObservedBlockLoop_InAToolCallBody_IsStoppedOnceItFillsTheBlockSpan()
    {
        var tokenizer = new TextTokens();
        int period = tokenizer.Encode(ObservedBlockLoop.Block).Count;
        Assert.InRange(period, RepetitionGuard.MaxPeriod + 1, 500);
        List<int> tokens = tokenizer.Encode(ObservedBlockLoop.Stream(copies: 30));
        int start = PeriodicFrom(tokens, period);

        int expected = RoundUp(start + RepetitionGuard.MinBlockSpan, RepetitionGuard.BlockCheckInterval);
        Assert.Equal(expected, FirstStop(tokens, out int found, out int repeats));
        Assert.Equal(period, found);
        Assert.Equal((expected - start) / period, repeats);
        Assert.InRange(repeats, 10, 20);

        string described = RepetitionGuard.Describe(tokens, expected, found, repeats, tokenizer.Decode);
        Assert.StartsWith("the output repeated `", described, StringComparison.Ordinal);
        Assert.EndsWith($"…` {repeats} times in a row", described, StringComparison.Ordinal);
    }

    /// <summary>
    /// Units of 100, 300 and 1,000+ tokens, up to the cap: each is stopped at the first check
    /// after its periodic tail spans <c>max(MinBlockSpan, MinBlockRepeats × period)</c>, and the
    /// guard reports the unit's own length, not a multiple of it.
    /// </summary>
    [Theory]
    [InlineData(65)]
    [InlineData(100)]
    [InlineData(300)]
    [InlineData(1000)]
    [InlineData(1024)]
    [InlineData(1500)]
    [InlineData(RepetitionGuard.MaxBlockPeriod)]
    public void BlockLoopsOfHundredsToThousandsOfTokens_AreStoppedWhereTheRuleSays(int period)
    {
        const int prefix = 37;
        int span = Math.Max(RepetitionGuard.MinBlockSpan, RepetitionGuard.MinBlockRepeats * period);
        List<int> tokens = Repeat(Enumerable.Range(10_000, period).ToArray(), span / period + 3, prefixLength: prefix);

        int expected = RoundUp(prefix + span, RepetitionGuard.BlockCheckInterval);
        Assert.Equal(expected, FirstStop(tokens, out int found, out int repeats));
        Assert.Equal(period, found);
        Assert.Equal((expected - prefix) / period, repeats);
    }

    [Fact]
    public void ABlockLongerThanTheCap_IsNeverALoop()
    {
        List<int> tokens = Repeat(Enumerable.Range(10_000, RepetitionGuard.MaxBlockPeriod + 1).ToArray(), 6);
        Assert.Equal(-1, FirstStop(tokens, out _, out _));
    }

    /// <summary>
    /// The thresholds. Exact copies of a long block are left alone until they fill 4,096 tokens
    /// AND number at least four. A few verbatim copies of a section, or a dozen of a paragraph,
    /// can be ordinary output.
    /// </summary>
    [Fact]
    public void ABlockUnderTheSpanOrWithTooFewCopies_IsLeftAlone()
    {
        // Thirteen copies of a 300-token block, 3,900 tokens, after 500 tokens of other text.
        Assert.Equal(-1, FirstStop(Repeat(Enumerable.Range(10_000, 300).ToArray(), 13, prefixLength: 500), out _, out _));
        // Three copies of a 1,500-token block: 4,500 tokens, past the span, but only three copies.
        Assert.Equal(-1, FirstStop(Repeat(Enumerable.Range(10_000, 1500).ToArray(), 3, prefixLength: 100), out _, out _));
        // Four copies of a 1,024-token block less one token: one short of the span.
        List<int> almost = Repeat(Enumerable.Range(10_000, 1024).ToArray(), 4, prefixLength: 64);
        almost.RemoveAt(almost.Count - 1);
        Assert.Equal(-1, FirstStop(almost, out _, out _));
    }

    /// <summary>
    /// "Exact", not "similar": the same 300-token block 60 times over (18,000 tokens), each copy
    /// numbered the way a model numbers slides, is never a loop. A loop that varies that
    /// little is the price of never stopping a list.
    /// </summary>
    [Fact]
    public void ABlockRepeatedWithOneTokenChangedPerCopy_IsNotALoop()
    {
        var tokens = new List<int>();
        for (int copy = 0; copy < 60; copy++)
        {
            tokens.Add(50_000 + copy);
            tokens.AddRange(Enumerable.Range(10_000, 299));
        }
        Assert.Equal(-1, FirstStop(tokens, out _, out _));
    }

    /// <summary>
    /// Long output that legitimately repeats its structure, judged after EVERY token as the
    /// engine does. Each is long enough for the block rule to run on it many times. None may stop.
    /// </summary>
    [Theory]
    [InlineData("csv-500-similar-rows")]
    [InlineData("csv-rows-identical-but-the-id")]
    [InlineData("code-repeated-getters")]
    [InlineData("markdown-table")]
    [InlineData("list-of-similar-items")]
    [InlineData("document-sections-sharing-headings")]
    [InlineData("base64")]
    [InlineData("json-array-of-similar-objects")]
    public void LongStructuredOutput_IsNeverStopped(string kind)
    {
        string text = LegitimateOutput.Of(kind);
        List<int> tokens = new TextTokens().Encode(text);
        Assert.True(tokens.Count >= 2 * RepetitionGuard.MinBlockSpan, $"{kind}: only {tokens.Count} tokens");
        Assert.Equal(-1, FirstStop(tokens, out int period, out int repeats));
        Assert.Equal(0, period);
        Assert.Equal(0, repeats);
    }

    /// <summary>
    /// The short rule is unchanged: for every period it covers, the stop comes exactly where it
    /// always did, at <c>max(MinSpan, MinRepeats × period)</c> tokens of loop, with that period.
    /// </summary>
    [Fact]
    public void TheShortRule_StillStopsExactlyWhereItDid_ForEveryPeriodUpTo64()
    {
        const int prefix = 11;
        for (int p = 1; p <= RepetitionGuard.MaxPeriod; p++)
        {
            int span = Math.Max(RepetitionGuard.MinSpan, RepetitionGuard.MinRepeats * p);
            List<int> tokens = Repeat(Enumerable.Range(20_000, p).ToArray(), span / p + 2, prefixLength: prefix);
            Assert.Equal(prefix + span, FirstStop(tokens, out int found, out int repeats));
            Assert.Equal(p, found);
            Assert.Equal(span / p, repeats);
        }
    }

    /// <summary>
    /// The answer depends only on the counted tokens, so the engine's per-token call and the
    /// speculative paths' calls (an unpublished tail, a token added and removed again) agree.
    /// </summary>
    [Fact]
    public void TheBlockRule_JudgesOnlyTheCountedPrefix()
    {
        List<int> tokens = Repeat(Enumerable.Range(10_000, 300).ToArray(), 40, prefixLength: 37);
        int stop = RoundUp(37 + RepetitionGuard.MinBlockSpan, RepetitionGuard.BlockCheckInterval);
        Assert.False(RepetitionGuard.IsLooping(tokens, stop - RepetitionGuard.BlockCheckInterval, out _, out _));
        Assert.True(RepetitionGuard.IsLooping(tokens, stop, out int period, out _));
        Assert.Equal(300, period);
        // The same prefix through a list that is not a List<int> or an array.
        Assert.True(RepetitionGuard.IsLooping(tokens.AsReadOnly(), stop, out period, out _));
        Assert.Equal(300, period);
        Assert.True(RepetitionGuard.IsLooping(tokens.ToArray(), stop, out period, out _));
        Assert.Equal(300, period);
    }

    /// <summary>
    /// The cost, counted rather than timed. At TensorAgent's default reply limit (262,144 tokens)
    /// the block rule reads a fixed window at each check, so its work per check is the same at
    /// the end of the reply as at the start, and its average per token is small. That holds both
    /// on structured text, where the newest token (a comma, a newline) recurs often, and on
    /// random ids.
    /// </summary>
    [Fact]
    public void TheBlockCheck_DoesAFixedAmountOfWork_HoweverLongTheOutputIs()
    {
        const int length = 262_144;
        // The most one check can do: the scan, two probes per candidate, the capped verifications.
        long bound = RepetitionGuard.MaxBlockPeriod + 16L * RepetitionGuard.MaxBlockPeriod
            + (long)RepetitionGuard.MaxBlockVerifications * RepetitionGuard.MinBlockRepeats * RepetitionGuard.MaxBlockPeriod;
        var random = new Random(7);
        int[] randomIds = Enumerable.Range(0, length).Select(_ => random.Next(65_536)).ToArray();
        int[] csv = new TextTokens().Encode(LegitimateOutput.SimilarCsv(14_000)).Take(length).ToArray();
        Assert.Equal(length, csv.Length);

        foreach (int[] stream in new[] { csv, randomIds })
        {
            long total = 0, first = 0, last = 0;
            for (int c = RepetitionGuard.MinBlockSpan; c <= stream.Length; c += RepetitionGuard.BlockCheckInterval)
            {
                long work = 0;
                int window = Math.Min(c, RepetitionGuard.MinBlockRepeats * RepetitionGuard.MaxBlockPeriod);
                Assert.False(RepetitionGuard.FindBlockLoop(stream.AsSpan(c - window, window), out _, out _, ref work));
                Assert.InRange(work, 1, bound);
                total += work;
                if (c < 32_768) first += work;
                if (c > length - 28_672) last += work;
            }
            // Under 100 compared tokens per output token on average...
            Assert.True(total < 100L * length, $"{total} for {length} tokens");
            // ...and no growth: the last 28K tokens cost about what the first did, once the window is full.
            Assert.True(last < 2 * first, $"first {first}, last {last}");
        }
    }

    private static int RoundUp(int value, int multiple) => (value + multiple - 1) / multiple * multiple;

    /// <summary>Asks the guard after every token, as the engine does: the count at which it first says yes, or -1.</summary>
    private static int FirstStop(List<int> tokens, out int period, out int repeats)
    {
        for (int count = 1; count <= tokens.Count; count++)
        {
            if (RepetitionGuard.IsLooping(tokens, count, out period, out repeats))
                return count;
        }
        period = 0;
        repeats = 0;
        return -1;
    }

    /// <summary>Where the stream becomes <paramref name="period"/>-periodic all the way to its end.</summary>
    private static int PeriodicFrom(List<int> tokens, int period)
    {
        int start = tokens.Count - period;
        while (start > 0 && tokens[start - 1] == tokens[start - 1 + period])
            start--;
        return start;
    }
}

/// <summary>
/// The runaway round from the trace: TensorSharp.Server, LFM3-Omni 2B Q4_K_M on ggml_metal, the
/// PowerPoint control prompt after a refused <c>spawn_agent</c>. Its <c>apply_patch</c> body
/// wrote a first copy of this block (with "Slide Title"), then the block itself 520 more times.
/// </summary>
internal static class ObservedBlockLoop
{
    public const string Header =
        "<tool_call>\n<function=apply_patch>\n<parameter=patch>\n*** Begin Patch\n"
        + "*** Add File: water-cycle-slide-design.md\n+# Water Cycle PowerPoint Design Specs\n+\n+";

    /// <summary>The repeated unit: 747 characters, 16 lines.</summary>
    public const string Block =
        "## Slide 1: Title Slide\n"
        + "+**Title**: Water Cycle Overview\n"
        + "+**Subtitle**: The cycle of water moving through Earth\n"
        + "+**Presenter**: [Your Name]\n"
        + "+**Date**: [YYYY-MM-DD]\n"
        + "+**Logo**: [Insert company logo here]\n"
        + "+\n"
        + "+## Visual Layout & Style\n"
        + "+**Layout Pattern**: Minimalist Corporate (title slide)\n"
        + "+**Visual Style**: Clean, professional, corporate\n"
        + "+**Colors**: Primary: #1E3A5F (Dark Blue); Secondary: #3498DB (Bright Blue); Accent: #E74C3C (Red); "
        + "Background: #F5F7FA (Light Gray); Text: #2C3E50 (Dark Gray)\n"
        + "+**Typography**: Headings - Montserrat Bold; Body - Open Sans Regular\n"
        + "+**Icons**: Simple geometric shapes (circle, square, triangle) for concepts\n"
        + "+**Image**: Full-width stock photo of water cycle (e.g., from Unsplash)\n"
        + "+**Layout Specs**:\n"
        + "+```markdown\n"
        + "+";

    /// <summary>The header and the first copy, which said "Slide Title" where the rest say "Title".</summary>
    public static string LeadIn => Header + Block.Replace("+**Title**:", "+**Slide Title**:", StringComparison.Ordinal);

    public static string Stream(int copies) => LeadIn + string.Concat(Enumerable.Repeat(Block, copies));
}

/// <summary>
/// A stand-in for a subword tokenizer: whitespace runs and punctuation marks are one id each,
/// and a word is cut into pieces of at most four characters. The same piece always gets the
/// same id. Like a real tokenizer it is deterministic and local, which is all that periodicity
/// depends on.
/// </summary>
internal sealed class TextTokens
{
    private static readonly Regex Piece = new(@"(?<w>\w+)|\s+|[^\w\s]", RegexOptions.CultureInvariant);
    private readonly Dictionary<string, int> _ids = new(StringComparer.Ordinal);
    private readonly List<string> _pieces = new();

    public List<int> Encode(string text)
    {
        var ids = new List<int>(text.Length / 3);
        foreach (Match match in Piece.Matches(text))
        {
            string value = match.Value;
            if (!match.Groups["w"].Success)
            {
                ids.Add(Id(value));
                continue;
            }
            for (int at = 0; at < value.Length; at += 4)
                ids.Add(Id(value.Substring(at, Math.Min(4, value.Length - at))));
        }
        return ids;
    }

    public string Decode(List<int> ids) => string.Concat(ids.Select(id => _pieces[id]));

    private int Id(string piece)
    {
        if (!_ids.TryGetValue(piece, out int id))
        {
            id = _pieces.Count;
            _ids.Add(piece, id);
            _pieces.Add(piece);
        }
        return id;
    }
}

/// <summary>
/// Long output that repeats its structure on purpose: what the long-block rule must never stop.
/// Every generator is deterministic.
/// </summary>
internal static class LegitimateOutput
{
    private static readonly string[] Regions = { "north", "south", "east", "west", "central" };
    private static readonly string[] Products = { "widget", "gadget", "gizmo", "sprocket" };
    private static readonly string[] Words =
    {
        "water", "vapour", "cloud", "river", "ocean", "rain", "snow", "soil", "plant", "heat", "cool",
        "flow", "basin", "ground", "storage", "runoff", "the", "a", "of", "and", "into", "from", "moves",
        "rises", "falls", "collects", "returns", "slowly", "quickly", "during", "after", "season",
    };

    public static string Of(string kind) => kind switch
    {
        "csv-500-similar-rows" => SimilarCsv(500),
        "csv-rows-identical-but-the-id" => IdenticalButTheId(500),
        "code-repeated-getters" => Getters(150),
        "markdown-table" => Table(400),
        "list-of-similar-items" => Items(500),
        "document-sections-sharing-headings" => Sections(40),
        "base64" => Base64(30_000),
        "json-array-of-similar-objects" => JsonObjects(300),
        _ => throw new ArgumentOutOfRangeException(nameof(kind), kind, null),
    };

    public static string SimilarCsv(int rows)
    {
        var sb = new StringBuilder("order_id,customer,region,product,quantity,unit_price,status,created_at\n");
        for (int i = 1; i <= rows; i++)
        {
            sb.Append(CultureInfo.InvariantCulture,
                $"{10_000 + i},customer_{i % 37:000},{Regions[i % 5]},{Products[i % 4]},{1 + i % 9},")
              .Append(CultureInfo.InvariantCulture, $"{i * 7919 % 10_000 / 100.0:0.00},shipped,2026-{1 + i % 12:00}-{1 + i % 28:00}\n");
        }
        return sb.ToString();
    }

    private static string IdenticalButTheId(int rows)
    {
        var sb = new StringBuilder("id,status,q1,q2,q3,q4,q5,q6,q7,q8,owner,reviewer,notes\n");
        for (int i = 1; i <= rows; i++)
            sb.Append(CultureInfo.InvariantCulture, $"{i},pending,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,none,none,awaiting first review\n");
        return sb.ToString();
    }

    private static string Getters(int count)
    {
        var sb = new StringBuilder("namespace Contoso.Settings;\n\npublic sealed class Options\n{\n");
        for (int i = 0; i < count; i++)
            sb.Append(CultureInfo.InvariantCulture, $"    private int _option{i};\n");
        sb.Append('\n');
        for (int i = 0; i < count; i++)
        {
            sb.Append(CultureInfo.InvariantCulture, $"    /// <summary>Gets or sets option {i}. The default is zero.</summary>\n")
              .Append(CultureInfo.InvariantCulture, $"    public int Option{i}\n    {{\n        get {{ return _option{i}; }}\n")
              .Append(CultureInfo.InvariantCulture, $"        set {{ _option{i} = value; OnChanged(nameof(Option{i})); }}\n    }}\n\n");
        }
        sb.Append("}\n");
        return sb.ToString();
    }

    private static string Table(int rows)
    {
        var sb = new StringBuilder("| # | Station | Rainfall (mm) | Runoff (mm) | Status |\n| --- | --- | --- | --- | --- |\n");
        for (int i = 1; i <= rows; i++)
            sb.Append(CultureInfo.InvariantCulture, $"| {i} | Station {i % 23} | {i * 37 % 400} | {i * 11 % 90} | {(i % 3 == 0 ? "review" : "ok")} |\n");
        return sb.ToString();
    }

    private static string Items(int count)
    {
        var sb = new StringBuilder("Maintenance checklist:\n\n");
        for (int i = 1; i <= count; i++)
            sb.Append(CultureInfo.InvariantCulture,
                $"- Item {i}: Check the pressure valve on unit {i % 12} and record the reading in the maintenance log before the shift ends.\n");
        return sb.ToString();
    }

    private static string Sections(int count)
    {
        // Every section has the same three headings and ends with the same boilerplate note.
        const string note = "Note: figures in this section are estimates from the regional survey and may be revised "
            + "when the final measurements are published. Contact the data team for the underlying tables.\n\n";
        var random = new Random(11);
        var sb = new StringBuilder("# Regional Water Report\n\n");
        for (int s = 1; s <= count; s++)
        {
            sb.Append(CultureInfo.InvariantCulture, $"## Region {s}\n\n### Overview\n\n").Append(Sentences(random, 3))
              .Append("\n### Details\n\n").Append(Sentences(random, 4))
              .Append("\n### Summary\n\n").Append(Sentences(random, 1)).Append('\n').Append(note);
        }
        return sb.ToString();
    }

    private static string Sentences(Random random, int count)
    {
        var sb = new StringBuilder();
        for (int i = 0; i < count; i++)
        {
            int words = 9 + random.Next(8);
            for (int w = 0; w < words; w++)
                sb.Append(w == 0 ? "" : " ").Append(Words[random.Next(Words.Length)]);
            sb.Append(". ");
        }
        return sb.Append('\n').ToString();
    }

    private static string Base64(int bytes)
    {
        var data = new byte[bytes];
        new Random(5).NextBytes(data);
        return Convert.ToBase64String(data, Base64FormattingOptions.InsertLineBreaks);
    }

    private static string JsonObjects(int count)
    {
        var sb = new StringBuilder("[\n");
        for (int i = 1; i <= count; i++)
        {
            sb.Append(CultureInfo.InvariantCulture,
                $"  {{\"id\": {i}, \"name\": \"user{i}\", \"active\": true, \"roles\": [\"reader\"], \"quota\": 100, \"region\": \"{Regions[i % 5]}\"}}")
              .Append(i == count ? "\n" : ",\n");
        }
        return sb.Append("]\n").ToString();
    }
}
