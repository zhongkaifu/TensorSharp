// Copyright (c) Zhongkai Fu. All rights reserved.
// https://github.com/zhongkaifu/TensorSharp
//
// This file is part of TensorSharp.
//
// TensorSharp is licensed under the BSD-3-Clause license found in the LICENSE file in the root directory of this source tree.
//
// Qwen 3.5 M-RoPE positions against a reference implementation.
//
// Fixtures/Qwen35MRope/reference_positions.json is written by
// eng/validation/qwen35_mrope_reference/generate_reference_positions.py from SGLang's own
// get_rope_index(model_type="qwen3_5") and its decode rule (delta - 1 + seq_len), for
// synthetic prompts with one image, two images, an image-first tall grid, a wide grid, a
// two-pair video and plain text. Each scenario also carries a follow-up turn (prompt +
// six generated tokens + new text) laid out from scratch, whose tail is where a cache
// built by DECODING must have put those tokens for reuse to be exact.
//
// The tests drive the real code: the injector's layout, and the model's own position
// state machine (BeginRopePositions / RopePosition, reached by reflection on an
// uninitialized Qwen35RuntimeModel - no weights are involved in choosing a position).
using System.Reflection;
using System.Runtime.CompilerServices;
using System.Text.Json;
using TensorSharp.Models;

namespace InferenceWeb.Tests;

public class Qwen35MRopeReferencePositionTests
{
    private sealed record Span(int PadTokenId, int MergedHeight, int MergedWidth, int TokenCount);

    private sealed record Scenario(
        string Name, int[] PromptTokens, Span[] Spans, int[] Positions, int Delta, int[] DecodePositions,
        int[] FollowUpTokens, int[] FollowUpPositions, int FollowUpDelta);

    private sealed record Reference(int ImagePadTokenId, int VideoPadTokenId, string SglangRevision, Scenario[] Scenarios);

    private static readonly Lazy<Reference> Fixture = new(Load);

    private static Reference Load()
    {
        using Stream stream = typeof(Qwen35MRopeReferencePositionTests).Assembly.GetManifestResourceStream(
            "InferenceWeb.Tests.Fixtures.Qwen35MRope.reference_positions.json")!;
        Assert.NotNull(stream);
        return JsonSerializer.Deserialize<Reference>(stream, new JsonSerializerOptions { PropertyNameCaseInsensitive = true })!;
    }

    public static IEnumerable<object[]> ScenarioNames()
        => Load().Scenarios.Select(s => new object[] { s.Name });

    private static Scenario Get(string name) => Fixture.Value.Scenarios.Single(s => s.Name == name);

    private static HashSet<int> PadIds() => new() { Fixture.Value.ImagePadTokenId, Fixture.Value.VideoPadTokenId };

    /// <summary>The prompt as the chat template renders it: one placeholder per span.</summary>
    private static List<int> Collapse(int[] expanded, HashSet<int> pads)
    {
        var tokens = new List<int>(expanded.Length);
        for (int i = 0; i < expanded.Length; i++)
        {
            if (pads.Contains(expanded[i]) && i > 0 && expanded[i - 1] == expanded[i])
                continue;
            tokens.Add(expanded[i]);
        }
        return tokens;
    }

    private static (List<int> Tokens, int[] Positions) Layout(int[] expandedTokens, Span[] spans)
    {
        var pads = PadIds();
        var (tokens, positions, _) = ModelMultimodalInjector.LayoutQwenVLPrompt(
            Collapse(expandedTokens, pads),
            spans.Select(s => new ModelMultimodalInjector.QwenVLVisionSpan(s.PadTokenId, s.TokenCount, s.MergedHeight, s.MergedWidth)).ToList(),
            pads);
        return (tokens, positions);
    }

    [Theory]
    [MemberData(nameof(ScenarioNames))]
    public void PromptLayout_MatchesSGLangGetRopeIndex(string name)
    {
        Scenario s = Get(name);
        var (tokens, positions) = Layout(s.PromptTokens, s.Spans);
        Assert.Equal(s.PromptTokens, tokens);
        Assert.Equal(s.Positions, positions);
    }

    [Theory]
    [MemberData(nameof(ScenarioNames))]
    public void RopeDelta_MatchesSGLangMropePositionDelta(string name)
    {
        Scenario s = Get(name);
        var (_, positions) = Layout(s.PromptTokens, s.Spans);
        int rows = positions.Length / 3;
        Assert.Equal(s.Delta, Qwen35RopePositions.DeltaAfterRows(positions, rows, rows));

        // The injector answers the same for a request, which is what the paged batched
        // path asks for every sequence past its table.
        var injector = new ModelMultimodalInjector(NewModel());
        InjectorSetPositions(injector, "req", positions);
        Assert.True(InjectorTryGetDelta(injector, "req", out int delta));
        Assert.Equal(s.Delta, delta);
        Assert.False(InjectorTryGetDelta(injector, "text-only", out _));
    }

    /// <summary>
    /// The model's own state machine, fed the prompt the way the engine feeds it (a
    /// position-table slice per prefill chunk, then single decode tokens with none),
    /// must rotate every prompt row at the reference position and every decoded token
    /// at SGLang's decode position - for any chunking, including one-token chunks.
    /// </summary>
    [Theory]
    [MemberData(nameof(ScenarioNames))]
    public void ModelPositions_PrefillChunksAndDecode_MatchSGLang(string name)
    {
        Scenario s = Get(name);
        var (_, table) = Layout(s.PromptTokens, s.Spans);
        int promptLen = table.Length / 3;
        foreach (int chunk in new[] { promptLen, 7, 3, 1 })
        {
            var model = NewModel();
            var produced = new List<int>();
            for (int start = 0; start < promptLen; start += chunk)
            {
                int n = Math.Min(chunk, promptLen - start);
                Stage(model, table.AsSpan(3 * start, 3 * n).ToArray());
                produced.AddRange(Begin(model, start, n, out _));
            }
            Assert.Equal(s.Positions, produced);
            Assert.Equal(s.Delta, model.ActiveRopeDelta);

            for (int k = 0; k < s.DecodePositions.Length; k++)
            {
                int[] decode = Begin(model, promptLen + k, 1, out bool tabled);
                Assert.False(tabled);
                Assert.Equal(new[] { s.DecodePositions[k], s.DecodePositions[k], s.DecodePositions[k] }, decode);
            }
        }
    }

    /// <summary>
    /// The property reuse depends on: a follow-up turn laid out from scratch puts the
    /// generated tokens exactly where decode put them, and the model continuing that
    /// cache with the new turn's table produces the same positions as the table.
    /// </summary>
    [Theory]
    [MemberData(nameof(ScenarioNames))]
    public void FollowUpTurn_ReprefillEqualsDecodeThenContinuation(string name)
    {
        Scenario s = Get(name);
        var (tokens, followTable) = Layout(s.FollowUpTokens, s.Spans);
        Assert.Equal(s.FollowUpTokens, tokens);
        Assert.Equal(s.FollowUpPositions, followTable);
        Assert.Equal(s.Delta, s.FollowUpDelta);

        int promptLen = s.PromptTokens.Length;
        for (int k = 0; k < s.DecodePositions.Length; k++)
        {
            int row = 3 * (promptLen + k);
            Assert.Equal(new[] { s.DecodePositions[k], s.DecodePositions[k], s.DecodePositions[k] },
                followTable.AsSpan(row, 3).ToArray());
        }

        // A cache that prefilled the prompt and decoded all but the last generated token
        // (the engine never forwards the final sample) continues with the rest of the new
        // prompt: that slice is text, so the model drops the table and rotates it at
        // KV index + delta - which must be the table's own positions.
        var (_, promptTable) = Layout(s.PromptTokens, s.Spans);
        var model = NewModel();
        Stage(model, promptTable);
        Begin(model, 0, promptLen, out _);
        int cached = promptLen + s.DecodePositions.Length - 1;
        for (int i = promptLen; i < cached; i++)
            Begin(model, i, 1, out _);

        int suffix = s.FollowUpTokens.Length - cached;
        int[] slice = followTable.AsSpan(3 * cached, 3 * suffix).ToArray();
        Stage(model, slice);
        int[] continued = Begin(model, cached, suffix, out bool tabled);
        Assert.False(tabled);   // text at KV index + delta: the scalar form
        Assert.Equal(slice, continued);
        Assert.Equal(s.FollowUpDelta, model.ActiveRopeDelta);
    }

    /// <summary>Qwen3.8 Flash Next uses the same Qwen-VL layout, with the signed
    /// offset stored as a gap in its retained holder. Check its actual gap update
    /// and MTP scalar-position selection against the independent reference before
    /// allowing the radix cache to resume through images or video.</summary>
    [Theory]
    [MemberData(nameof(ScenarioNames))]
    public void Qwen4ExpPositions_DecodeAndRetainedContinuation_MatchSGLang(string name)
    {
        Scenario s = Get(name);
        int promptLen = s.PromptTokens.Length;
        foreach (int chunk in new[] { promptLen, 7, 3, 1 })
        {
            var model = (Qwen4ExpModel)RuntimeHelpers.GetUninitializedObject(typeof(Qwen4ExpModel));
            try
            {
                for (int start = 0; start < promptLen; start += chunk)
                {
                    int n = Math.Min(chunk, promptLen - start);
                    model.SetMRoPEPositions(s.Positions.AsSpan(3 * start, 3 * n).ToArray());
                    Qwen4ExpMtpMathTests.Invoke(model, "UpdateMropeGap", [start, n]);
                }
                model.SetMRoPEPositions(null);
                int gap = (int)typeof(Qwen4ExpModel).GetField("_mropeCacheGap", BindingFlags.Instance | BindingFlags.NonPublic)!
                    .GetValue(model)!;
                Assert.Equal(-s.Delta, gap);
                for (int k = 0; k < s.DecodePositions.Length; k++)
                    Assert.Equal(s.DecodePositions[k], Qwen4ExpModel.ResolveMtpPositions([], promptLen + k, 1, gap).RopePosition);

                object holder = Qwen4ExpMtpMathTests.Invoke(model, "SnapshotActiveCache", [])!;
                Assert.Equal(gap, (int)holder.GetType().GetField("MropeCacheGap")!.GetValue(holder)!);
                int cached = promptLen + s.DecodePositions.Length - 1;
                int suffix = s.FollowUpTokens.Length - cached;
                var continuation = Qwen4ExpModel.ResolveMtpPositions([], cached, suffix, gap);
                Assert.Null(continuation.MultiAxis);
                for (int i = 0; i < suffix; i++)
                    Assert.Equal(s.FollowUpPositions[3 * (cached + i)], continuation.RopePosition + i);
                model.SetMRoPEPositions(s.FollowUpPositions.AsSpan(3 * cached, 3 * suffix).ToArray());
                Qwen4ExpMtpMathTests.Invoke(model, "UpdateMropeGap", [cached, suffix]);
                Assert.Equal(-s.FollowUpDelta, (int)typeof(Qwen4ExpModel)
                    .GetField("_mropeCacheGap", BindingFlags.Instance | BindingFlags.NonPublic)!.GetValue(model)!);
            }
            finally { GC.SuppressFinalize(model); }
        }
    }

    [Fact]
    public void NewHistory_ResetsTheDelta_AndAHolderSnapshotCarriesIt()
    {
        Scenario s = Get("two_images");
        var (_, table) = Layout(s.PromptTokens, s.Spans);
        var model = NewModel();
        Stage(model, table);
        Begin(model, 0, table.Length / 3, out _);
        Assert.Equal(s.Delta, model.ActiveRopeDelta);

        // Swapping a sequence out captures its delta with its rows (the slot carries it).
        object slot = typeof(StandardDecoderModel).GetMethod("SnapshotActive", Hidden)!.Invoke(model, null)!;
        Assert.Equal(s.Delta, (int)slot.GetType().GetField("RopeDelta")!.GetValue(slot)!);

        // A decode continues the delta; a text forward from index 0 starts a new history.
        Begin(model, table.Length / 3, 1, out _);
        Assert.Equal(s.Delta, model.ActiveRopeDelta);
        Begin(model, 0, 4, out _);
        Assert.Equal(0, model.ActiveRopeDelta);
    }

    [Fact]
    public void Fixture_RecordsItsReferenceSource()
    {
        Assert.Matches("^[0-9a-f]{40}$", Fixture.Value.SglangRevision);
        Assert.Contains(Fixture.Value.Scenarios, s => s.Spans.Length == 2 && s.Spans.All(x => x.PadTokenId == Fixture.Value.ImagePadTokenId));
        Assert.Contains(Fixture.Value.Scenarios, s => s.Spans.Length == 1);
    }

    // ---- reflection plumbing ----

    private const BindingFlags Hidden = BindingFlags.Instance | BindingFlags.NonPublic | BindingFlags.Public;

    /// <summary>The shared runtime's position state, without weights: Qwen-VL's interleaved M-RoPE sections let it
    /// take a table.</summary>
    private static Qwen35RuntimeModel NewModel()
    {
        var model = (Qwen35RuntimeModel)RuntimeHelpers.GetUninitializedObject(typeof(Qwen35RuntimeModel));
        typeof(Qwen35RuntimeModel).GetField("_ropeSections", Hidden)!.SetValue(model, new[] { 11, 11, 10, 0 });
        return model;
    }

    /// <summary>The table the prompt expander stages for the next forward (null: none).</summary>
    private static void Stage(Qwen35RuntimeModel model, int[] table) => ((TensorSharp.Models.Architecture.IMRoPEPositionSink)model).SetMRoPEPositions(table);

    /// <summary>The (T, H, W) of each row of a forward of <paramref name="rows"/> rows at KV index
    /// <paramref name="start"/>, as the runtime rotates them; <paramref name="tabled"/>: from a table, not the
    /// scalar KV index + delta.</summary>
    private static int[] Begin(Qwen35RuntimeModel model, int start, int rows, out bool tabled)
    {
        object p = typeof(StandardDecoderModel).GetMethod("BeginRotaryPositions", Hidden)!.Invoke(model, new object[] { start, rows })!;
        var table = (int[])p.GetType().GetProperty("Table")!.GetValue(p);
        int delta = (int)p.GetType().GetProperty("RopeDelta")!.GetValue(p)!;
        tabled = table != null;
        var result = new int[3 * rows];
        for (int i = 0; i < rows; i++)
            for (int a = 0; a < 3; a++)
                result[3 * i + a] = table != null ? table[3 * i + a] : start + i + delta;
        return result;
    }

    private static void InjectorSetPositions(ModelMultimodalInjector injector, string requestId, int[] table)
        => injector.SetMRoPEPositions(requestId, table);

    private static bool InjectorTryGetDelta(ModelMultimodalInjector injector, string requestId, out int delta)
        => injector.TryGetMRoPEPositionDelta(requestId, out delta);
}
