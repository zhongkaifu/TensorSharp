using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using TensorSharp;
using TensorSharp.Runtime.Grammar;

namespace InferenceWeb.Tests;

public class DeepSeek41GrammarTriggerTests
{
    private const string Calls = "<｜DSML｜ calls>";
    private const string End = "</think>";
    private static byte[] Utf8(string text) => Encoding.UTF8.GetBytes(text);

    public static IEnumerable<object[]> MarkerSplits() =>
        Enumerable.Range(1, Utf8(Calls).Length - 1).Select(i => new object[] { i });

    [Theory]
    [MemberData(nameof(MarkerSplits))]
    public void Utf8MarkerCanSplitAtEveryByteAndConsumesSameTokenSuffix(int split)
    {
        byte[] marker = Utf8(Calls);
        var tokens = new BytesTokenizer(marker[..split], marker[split..].Concat(Utf8("{\"ok\":true}")).ToArray());
        var constraint = Json(tokens, Calls);
        constraint.Accept(1);
        Assert.False(constraint.IsActive);
        AssertAllowed(constraint, 2, true);
        constraint.Accept(2);
        Assert.True(constraint.IsActive && constraint.IsComplete && !constraint.IsDead);
    }

    [Fact]
    public void DormantMaskRejectsInvalidSuffixBeforeTokenIsSelectedIncludingAliasesAndControls()
    {
        var tokens = new BytesTokenizer(Utf8(End + "{"), Utf8(End + "invalid"),
            Utf8(End + "invalid"), Utf8(End + "["), Utf8("unrestricted reasoning"), Utf8(End));
        tokens.Special = new[] { 0, 4, 6 };
        var constraint = Json(tokens, End);
        AssertAllowed(constraint, 1, true);
        AssertAllowed(constraint, 2, false);
        AssertAllowed(constraint, 3, false);
        AssertAllowed(constraint, 4, false);
        AssertAllowed(constraint, 5, true);
        AssertAllowed(constraint, 6, true);
        constraint.Accept(1);
        Assert.True(constraint.IsActive);
        Assert.False(constraint.IsComplete || constraint.IsDead);
        constraint.AcceptBytes(Utf8("\"ok\":true}"));
        Assert.True(constraint.IsComplete);
    }

    [Fact]
    public void FragmentedMarkerCompletingTokenIsMaskedUsingCurrentPrefix()
    {
        var tokens = new BytesTokenizer(Utf8("</th"), Utf8("ink>{}"), Utf8("ink>x"), Utf8("other"));
        var constraint = Json(tokens, End);
        AssertAllowed(constraint, 3, true);
        constraint.Accept(1);
        AssertAllowed(constraint, 2, true);
        AssertAllowed(constraint, 3, false);
        AssertAllowed(constraint, 4, true);
        constraint.Accept(2);
        Assert.True(constraint.IsComplete && constraint.IsActive);
    }

    [Fact]
    public void OrderedGatesIgnoreQuotedToolMarkersUntilReasoningEnds()
    {
        var tokens = new BytesTokenizer(Utf8(Calls + " malformed quoted example"), Utf8(End),
            Utf8("plain final text"), Utf8(Calls + "{}"), Utf8(Calls + "not-json"));
        tokens.Special = new[] { 0, 2 };
        var constraint = Json(tokens, End, Calls);
        AssertAllowed(constraint, 1, true);
        AssertAllowed(constraint, 5, true);
        constraint.Accept(1);
        Assert.False(constraint.IsActive);
        constraint.Accept(2); // the same trained token is used by forced-budget closure
        Assert.False(constraint.IsActive);
        AssertAllowed(constraint, 3, true);
        AssertAllowed(constraint, 4, true);
        AssertAllowed(constraint, 5, false);
        constraint.Accept(4);
        Assert.True(constraint.IsActive && constraint.IsComplete);
    }

    [Fact]
    public void OrderedGatesAndGrammarSuffixMayAllOccurInOneToken()
    {
        var tokens = new BytesTokenizer(Utf8(Calls + " quote " + End + " prelude " + Calls + "{}"),
            Utf8(End + Calls + "wrong"));
        var constraint = Json(tokens, End, Calls);
        AssertAllowed(constraint, 1, true);
        AssertAllowed(constraint, 2, false);
        constraint.Accept(1);
        Assert.True(constraint.IsActive && constraint.IsComplete);
    }

    [Fact]
    public void OrderedGateStateIsForkedAndDoesNotShareProgressAcrossRequests()
    {
        var tokens = new BytesTokenizer(Utf8(End), Utf8(Calls + "{}"), Utf8(Calls + "wrong"));
        var first = Json(tokens, End, Calls);
        var before = first.Fork();
        first.Accept(1);
        var after = first.Fork();
        AssertAllowed(before, 3, true);
        AssertAllowed(after, 3, false);
        after.Accept(2);
        Assert.True(after.IsActive && after.IsComplete);
        Assert.False(first.IsActive || before.IsActive);
        before.Accept(2);
        Assert.False(before.IsActive);
    }

    [Fact]
    public void GrammarSuffixCanEndMidUtf8AndNextMaskAllowsOnlyValidContinuation()
    {
        var tokens = new BytesTokenizer(Utf8(Calls + "{\"currency\":\"").Concat(new byte[] { 0xE2 }).ToArray(),
            new byte[] { 0x82 }, new byte[] { 0xAC, (byte)'"', (byte)'}' }, Utf8("a"), new byte[] { 0xFF });
        var constraint = Json(tokens, Calls);
        AssertAllowed(constraint, 1, true);
        constraint.Accept(1);
        Assert.True(constraint.IsActive);
        Assert.False(constraint.IsComplete);
        AssertAllowed(constraint, 2, true);
        AssertAllowed(constraint, 4, false);
        AssertAllowed(constraint, 5, false);
        constraint.Accept(2);
        AssertAllowed(constraint, 3, true);
        constraint.Accept(3);
        Assert.True(constraint.IsComplete && !constraint.IsDead);
    }

    [Fact]
    public void ActiveGrammarAllowsEquivalentTokenAliasesAndByteFallbackCharacters()
    {
        var tokens = new BytesTokenizer(Utf8("\""), new byte[] { 0xE2 }, new byte[] { 0x82 },
            new byte[] { 0xAC }, Utf8("\""), Utf8("wrong"));
        var constraint = new GrammarConstraint(Grammar.Parse("root ::= \"\\\"€\\\"\""), tokens);
        AssertAllowed(constraint, 1, true);
        AssertAllowed(constraint, 5, true);
        foreach (int id in new[] { 1, 2, 3, 4, 5 })
        {
            AssertAllowed(constraint, id, true);
            constraint.Accept(id);
        }
        Assert.True(constraint.IsComplete && !constraint.IsDead);
    }

    [Fact]
    public void RepeatedAndOverlappingMarkerPrefixesDoNotLoseAValidMatch()
    {
        var tokens = new BytesTokenizer(Utf8("aba"), Utf8("ababab{}"), Utf8("babwrong"));
        var constraint = Json(tokens, "ababab");
        constraint.Accept(1);
        AssertAllowed(constraint, 3, false);
        constraint.Accept(2);
        Assert.True(constraint.IsComplete && !constraint.IsDead);
    }

    [Fact]
    public void PreludeAndForkReuseCachedMaskWithoutRepeatedTokenDecoding()
    {
        var tokens = new BytesTokenizer(Utf8("reasoning "), Utf8(Calls + "{}"), Utf8(Calls + "wrong"));
        var constraint = Json(tokens, Calls);
        ulong[] first = constraint.CurrentMask();
        constraint.Accept(1);
        int decoded = tokens.DecodeCalls;
        Assert.Same(first, constraint.CurrentMask());
        Assert.Same(first, constraint.Fork().CurrentMask());
        Assert.Equal(decoded, tokens.DecodeCalls);
    }

    [Fact]
    public void DifferentialMasksMatchIndependentWholeByteSuffixParsingAtEveryPrefix()
    {
        byte[] marker = Utf8(Calls);
        var candidates = new List<byte[]> { Utf8("plain"), Utf8("{}"), new byte[] { 0xFF } };
        for (int i = 0; i < marker.Length; i++)
            foreach (string suffix in new[] { "", "{}", "{", "invalid", "[]", "{\"a\":\"€\"}" })
                candidates.Add(marker[i..].Concat(Utf8(suffix)).ToArray());
        var tokens = new BytesTokenizer(candidates.ToArray());
        for (int prefix = 0; prefix < marker.Length; prefix++)
        {
            var constraint = Json(tokens, Calls);
            constraint.AcceptBytes(marker.AsSpan(0, prefix));
            for (int id = 1; id < tokens.VocabSize; id++)
            {
                byte[] whole = marker[..prefix].Concat(tokens.Bytes[id]).ToArray();
                int location = whole.AsSpan().IndexOf(marker);
                bool expected = true;
                if (location >= 0)
                {
                    var reference = new GrammarConstraint(Grammar.JsonObject(), tokens);
                    reference.AcceptBytes(whole.AsSpan(location + marker.Length));
                    expected = !reference.IsDead;
                }
                Assert.Equal(expected, Allowed(constraint, id));
            }
        }
    }

    [Fact]
    public void OrderedMarkersRejectInvalidConfiguration()
    {
        var constraint = Json(new BytesTokenizer(Utf8("x")), End);
        Assert.Throws<ArgumentNullException>(() => constraint.ActivateAfterTriggers(null!));
        Assert.Throws<ArgumentException>(() => constraint.ActivateAfterTriggers());
        Assert.Throws<ArgumentException>(() => constraint.ActivateAfterTriggers(End, ""));
    }

    // Campaign re-run 2026-09-16: Nemotron-H 8B Reasoning spells "</think>\n\n" as
    // "</think" + ">\n\n". The JSON root cannot start with whitespace, so the merged
    // token completing the marker was masked, the model wrote "</think}" instead, the
    // marker never matched and 10 of 36 --thinking json cases ended with content null
    // (json_schema: HTTP 422). The response_format path skips whitespace after the marker.
    [Fact]
    public void LeadingWhitespaceAfterMarker_IsPreludeWhenOptedIn()
    {
        var tokens = new BytesTokenizer(Utf8("</think"), Utf8(">\n\n"), Utf8("}"), Utf8("\n"),
            Utf8("{\"ok\":true}"), Utf8(" \t{}"), Utf8(">\n\nnot json"), Utf8("x"), Utf8(">\n{}"));
        var constraint = new GrammarConstraint(Grammar.JsonObject(), tokens);
        constraint.ActivateAfter(End, skipLeadingWhitespace: true);
        constraint.Accept(1);
        AssertAllowed(constraint, 2, true);
        AssertAllowed(constraint, 7, false);
        AssertAllowed(constraint, 9, true);
        var fork = constraint.Fork();
        constraint.Accept(2);
        Assert.True(constraint.IsActive);
        Assert.False(constraint.IsComplete);
        AssertAllowed(constraint, 3, false);
        AssertAllowed(constraint, 4, true);
        AssertAllowed(constraint, 6, true);
        AssertAllowed(constraint, 8, false);
        AssertAllowed(constraint, 5, true);
        constraint.Accept(4);
        constraint.Accept(5);
        Assert.True(constraint.IsComplete && !constraint.IsDead);
        // A fork taken before the marker keeps its own position, and one token may
        // complete the marker, skip whitespace and write the whole value.
        AssertAllowed(fork, 2, true);
        fork.Accept(9);
        Assert.True(fork.IsActive && fork.IsComplete);
    }

    [Fact]
    public void LeadingWhitespaceAfterMarker_StaysMaskedByDefault()
    {
        var tokens = new BytesTokenizer(Utf8("</think"), Utf8(">\n\n"), Utf8("}"), Utf8(">{}"));
        var constraint = Json(tokens, End);
        constraint.Accept(1);
        AssertAllowed(constraint, 2, false);
        AssertAllowed(constraint, 3, true);
        AssertAllowed(constraint, 4, true);
    }

    private static GrammarConstraint Json(BytesTokenizer tokens, params string[] markers)
    {
        var constraint = new GrammarConstraint(Grammar.JsonObject(), tokens);
        constraint.ActivateAfterTriggers(markers);
        return constraint;
    }

    private static bool Allowed(GrammarConstraint constraint, int token)
    {
        ulong[] mask = constraint.CurrentMask();
        return (mask[token >> 6] & (1UL << (token & 63))) != 0;
    }

    private static void AssertAllowed(GrammarConstraint constraint, int token, bool expected)
    {
        Assert.Equal(expected, Allowed(constraint, token));
        var logits = Enumerable.Repeat(1f, constraint.CurrentMask().Length * 64).ToArray();
        constraint.ApplyMask(logits, constraint.IsComplete);
        Assert.Equal(expected, !float.IsNegativeInfinity(logits[token]));
    }

    private sealed class BytesTokenizer : ITokenizer, ISpecialTokenVocabulary
    {
        public readonly byte[][] Bytes;
        public int DecodeCalls;
        public int[] Special = { 0 };
        public BytesTokenizer(params byte[][] pieces)
        {
            Bytes = new[] { Utf8("<eos>") }.Concat(pieces).ToArray();
            Vocab = Bytes.Select(Encoding.UTF8.GetString).ToArray();
        }
        public string[] Vocab { get; }
        public int BosTokenId => -1;
        public int[] EosTokenIds => new[] { 0 };
        public int VocabSize => Bytes.Length;
        public IReadOnlyCollection<int> SpecialTokenIds => Special;
        public List<int> Encode(string text, bool addSpecial = true) => new() { LookupToken(text) };
        public string Decode(List<int> ids) => Encoding.UTF8.GetString(ids.SelectMany(id => Bytes[id]).ToArray());
        public void AppendTokenBytes(int tokenId, List<byte> bytes) { DecodeCalls++; bytes.AddRange(Bytes[tokenId]); }
        public bool IsEos(int tokenId) => tokenId == 0;
        public int LookupToken(string token) => Array.IndexOf(Vocab, token);
    }
}
