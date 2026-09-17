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
// uninitialized Qwen35Model - no weights are involved in choosing a position).
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
                int[] slice = table.AsSpan(3 * start, 3 * n).ToArray();
                model.SetMRoPEPositions(slice);
                Begin(model, start, n);
                int[] pending = Pending(model);
                for (int i = 0; i < n; i++)
                {
                    if (pending != null)
                        produced.AddRange(new[] { pending[3 * i], pending[3 * i + 1], pending[3 * i + 2] });
                    else
                    {
                        int p = RopePosition(model, start + i);
                        produced.AddRange(new[] { p, p, p });
                    }
                }
                model.SetMRoPEPositions(null);   // ForwardCore clears it after the forward
            }
            Assert.Equal(s.Positions, produced);
            Assert.Equal(s.Delta, model.ActiveRopePositionDelta);

            for (int k = 0; k < s.DecodePositions.Length; k++)
            {
                Begin(model, promptLen + k, 1);
                Assert.Null(Pending(model));
                Assert.Equal(s.DecodePositions[k], RopePosition(model, promptLen + k));
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
        model.SetMRoPEPositions(promptTable);
        Begin(model, 0, promptLen);
        model.SetMRoPEPositions(null);
        int cached = promptLen + s.DecodePositions.Length - 1;
        for (int i = promptLen; i < cached; i++)
            Begin(model, i, 1);

        int suffix = s.FollowUpTokens.Length - cached;
        int[] slice = followTable.AsSpan(3 * cached, 3 * suffix).ToArray();
        model.SetMRoPEPositions(slice);
        Begin(model, cached, suffix);
        Assert.Null(Pending(model));
        for (int i = 0; i < suffix; i++)
            Assert.Equal(slice[3 * i], RopePosition(model, cached + i));
        Assert.Equal(s.FollowUpDelta, model.ActiveRopePositionDelta);
    }

    [Fact]
    public void NewHistory_ResetsTheDelta_AndAHolderSnapshotCarriesIt()
    {
        Scenario s = Get("two_images");
        var (_, table) = Layout(s.PromptTokens, s.Spans);
        var model = NewModel();
        model.SetMRoPEPositions(table);
        Begin(model, 0, table.Length / 3);
        model.SetMRoPEPositions(null);
        Assert.Equal(s.Delta, model.ActiveRopePositionDelta);

        // Swapping a sequence out captures its delta with its rows.
        object holder = typeof(Qwen35Model)
            .GetMethod("SnapshotActiveCache", BindingFlags.Instance | BindingFlags.NonPublic)!
            .Invoke(model, null)!;
        Assert.Equal(s.Delta, (int)holder.GetType().GetField("RopeDelta")!.GetValue(holder)!);

        // A decode continues the delta; a text forward from index 0 starts a new history.
        Begin(model, table.Length / 3, 1);
        Assert.Equal(s.Delta, model.ActiveRopePositionDelta);
        Begin(model, 0, 4);
        Assert.Equal(0, model.ActiveRopePositionDelta);
    }

    [Fact]
    public void Fixture_RecordsItsReferenceSource()
    {
        Assert.Matches("^[0-9a-f]{40}$", Fixture.Value.SglangRevision);
        Assert.Contains(Fixture.Value.Scenarios, s => s.Spans.Length == 2 && s.Spans.All(x => x.PadTokenId == Fixture.Value.ImagePadTokenId));
        Assert.Contains(Fixture.Value.Scenarios, s => s.Spans.Length == 1);
    }

    // ---- reflection plumbing ----

    private static Qwen35Model NewModel() => (Qwen35Model)RuntimeHelpers.GetUninitializedObject(typeof(Qwen35Model));

    private static void Begin(Qwen35Model model, int startPos, int seqLen)
        => typeof(Qwen35Model).GetMethod("BeginRopePositions", BindingFlags.Instance | BindingFlags.NonPublic)!
            .Invoke(model, new object[] { startPos, seqLen });

    private static int RopePosition(Qwen35Model model, int kvIndex)
        => (int)typeof(Qwen35Model).GetMethod("RopePosition", BindingFlags.Instance | BindingFlags.NonPublic)!
            .Invoke(model, new object[] { kvIndex })!;

    private static int[] Pending(Qwen35Model model)
        => (int[])typeof(Qwen35Model).GetField("_pendingMRoPEPositions", BindingFlags.Instance | BindingFlags.NonPublic)!
            .GetValue(model);

    private static void InjectorSetPositions(ModelMultimodalInjector injector, string requestId, int[] table)
        => injector.SetMRoPEPositions(requestId, table);

    private static bool InjectorTryGetDelta(ModelMultimodalInjector injector, string requestId, out int delta)
        => injector.TryGetMRoPEPositionDelta(requestId, out delta);
}
