// Copyright (c) Zhongkai Fu. All rights reserved.
// Licensed under the BSD-3-Clause license in the repository root.
using InferenceWeb.Tests.PrefixCache.Fakes;
using Microsoft.Extensions.Logging.Abstractions;
using TensorSharp.Cli;
using TensorSharp.Runtime.Scheduling;
using TensorSharp.Runtime.Scheduling.PrefixCache;

namespace InferenceWeb.Tests.PrefixCache;

[CollectionDefinition("Retained end-state budget environment", DisableParallelization = true)]
public sealed class RetainedEndStateBudgetEnvironmentCollection { }

/// <summary>
/// Every conversation a server runs in parallel keeps its own state for its next turn. The count of
/// retained conversation states was a fixed 4 (<c>TS_RETAINED_FUSED_CACHE_MAX</c>), so with eight
/// parallel conversations the fifth to finish evicted the first's: in every family that continues from a
/// retained state (GptOss, Gemma 4, Qwen 3.5/3.6/3.8, DeepSeek V4.1, GLM) the four oldest conversations
/// reused nothing on every later turn, 0 of 62 and 0 of 84 tokens against 52 and 74 for the other four.
/// The count now follows the scheduler's concurrency. Each family's oracle is given the memory for every
/// conversation, so only a count could limit reuse here, and every answer is compared with a cold run, so
/// a reuse that is not exact fails as well.
/// </summary>
[Collection("Retained end-state budget environment")]
public sealed class ParallelConversationReuseTests : IDisposable
{
    private const int Turns = 3, MaxNew = 12;
    private const string BudgetVariable = "TS_RETAINED_FUSED_CACHE_MAX";
    private readonly string? _restoreBudget = Environment.GetEnvironmentVariable(BudgetVariable);

    // What these tests pin is the budget when nothing sets it.
    public ParallelConversationReuseTests() => Environment.SetEnvironmentVariable(BudgetVariable, null);

    public void Dispose() => Environment.SetEnvironmentVariable(BudgetVariable, _restoreBudget);

    public static IEnumerable<object[]> Families() => AllFamilies.Select(f => new object[] { f.Name });

    // The DESIGN table, and Nemotron-H on its batched route.
    private static readonly (string Name, Func<OracleModel> Create)[] AllFamilies =
        OracleFakes.All.Append(("R3", () => OracleFakes.R3())).ToArray();

    [Theory]
    [MemberData(nameof(Families))]
    public async Task EightParallelConversations_EachReuseAsMuchAsTwoDo_AndAnswerLikeAColdRun(string family)
    {
        int[][] two = await RunAsync(family, 2);
        int[][] eight = await RunAsync(family, 8);
        for (int turn = 1; turn < Turns; turn++)
        {
            Assert.True(two[turn][0] > 0, $"{family}: two parallel conversations reused nothing on turn {turn}");
            for (int i = 0; i < eight[turn].Length; i++)
                Assert.True(eight[turn][i] == two[turn][0],
                    $"{family} turn {turn}: conversation {i} of 8 reused {eight[turn][i]} tokens, each of 2 reused {two[turn][0]}");
        }
    }

    /// <summary>
    /// On a Mac a retained conversation's memory is the machine's own (one memory for the GPU and
    /// everything else), and a spare figure measured after a payload was allocated never falls below what
    /// is cached, so nothing evicted: a Qwen3.8-27B MLX server kept sixteen conversations sized for the
    /// whole context, passed 34 GB on a 48 GB Mac and ran 3x slower in swap. The model reports how far the
    /// machine is short (<see cref="IPrefixCacheModel.QueryHostMemoryDeficitBytes"/>) and the prefix cache
    /// evicts that much before admitting a request and after keeping one. The retained states go; a public
    /// prefix may stay, and so does the live primary cache of a family that keeps one resident (it is
    /// allocated either way: evicting it frees nothing), and every answer is still the cold run's.
    /// </summary>
    [Theory]
    [MemberData(nameof(Families))]
    public async Task UnderAMemoryDeficit_TheRetainedConversationsAreEvicted_AndAnswerLikeAColdRun(string family)
    {
        const int conversations = 4;
        var plentyBytes = new long[Turns];
        var shortBytes = new long[Turns];
        int[][] plenty = await RunAsync(family, conversations, afterTurn: (engine, turn) => plentyBytes[turn] = CachedBytes(engine));
        int[][] shortOfMemory = await RunAsync(family, conversations,
            (model, turn) => model.HostMemoryDeficitBytes = turn > 0 ? long.MaxValue / 4 : 0,
            (engine, turn) => shortBytes[turn] = CachedBytes(engine));
        // What a family cannot give back stays (pages in the pool's own blocks, which P2 reuses through; the slot
        // N keeps), and a page family's host slabs go at the next admission; nothing is ever kept or reused more.
        for (int turn = 1; turn < Turns; turn++)
        {
            Assert.True(shortOfMemory[turn].Sum() <= plenty[turn].Sum(),
                $"{family} turn {turn}: reused more short of memory ([{string.Join(", ", shortOfMemory[turn])}]) than with it ([{string.Join(", ", plenty[turn])}])");
            Assert.True(shortBytes[turn] <= plentyBytes[turn],
                $"{family} turn {turn}: cached {shortBytes[turn]} bytes short of memory, {plentyBytes[turn]} with it");
        }
    }

    /// <summary>The case the deficit is for: Qwen 3.5 holders (R). With memory the server keeps every conversation
    /// it is not running; short of it, the finished conversations are given back as they are kept, and what a turn
    /// captured on the way (a prompt-tail checkpoint, which does not pass the trim) goes at the next admission.</summary>
    [Fact]
    public async Task UnderAMemoryDeficit_Qwen35HoldersAreGivenBack()
    {
        const int conversations = 4;
        var plentyBytes = new long[Turns];
        var shortBytes = new long[Turns];
        int[][] plenty = await RunAsync("R", conversations, afterTurn: (engine, turn) => plentyBytes[turn] = CachedBytes(engine));
        int[][] shortOfMemory = await RunAsync("R", conversations,
            (model, turn) => model.HostMemoryDeficitBytes = turn > 0 ? long.MaxValue / 4 : 0,
            (engine, turn) => shortBytes[turn] = CachedBytes(engine));
        for (int turn = 1; turn < Turns; turn++)
        {
            Assert.True(plentyBytes[turn] > 0 && plenty[turn].All(r => r > 0), $"turn {turn}: with memory the conversations were not kept");
            Assert.True(shortBytes[turn] < plentyBytes[turn] / 2,
                $"turn {turn}: {shortBytes[turn]} bytes kept short of memory, {plentyBytes[turn]} with it");
            Assert.True(shortOfMemory[turn].Sum() < plenty[turn].Sum(),
                $"turn {turn}: reused [{string.Join(", ", shortOfMemory[turn])}] short of memory, [{string.Join(", ", plenty[turn])}] with it");
        }
    }

    private static long CachedBytes(InferenceEngine engine)
    {
        ResourceVector c = engine.RadixCache!.Tree.Cached;
        return c.HostKv + c.StateSnapshot + c.DeviceKv + c.NativeSlot;
    }

    [Fact]
    public void TheBudget_FollowsTheConcurrency_UnlessSetExplicitly()
    {
        Assert.Null(ExecutionOptions.FromEnvironment().RetainedFusedCacheBudget);
        Assert.Equal(16, ExecutionOptions.FromEnvironment().RetainedFusedCacheBudgetFor(16));
        Assert.Equal(ExecutionOptions.MinDefaultRetainedFusedCacheBudget, ExecutionOptions.FromEnvironment().RetainedFusedCacheBudgetFor(1));

        // TensorAgent pins 1 on the phone, and 0 turns retention off: an explicit value is used as given.
        foreach (int pinned in new[] { 0, 1, 64 })
        {
            Environment.SetEnvironmentVariable(BudgetVariable, pinned.ToString());
            Assert.Equal(pinned, ExecutionOptions.FromEnvironment().RetainedFusedCacheBudgetFor(16));
            Assert.Contains($"{BudgetVariable}={pinned}", ExecutionOptions.FromEnvironment().DescribeOverrides(), StringComparison.Ordinal);
        }
        Environment.SetEnvironmentVariable(BudgetVariable, "lots");
        Assert.Equal(16, ExecutionOptions.FromEnvironment().RetainedFusedCacheBudgetFor(16));
    }

    /// <summary>Runs <paramref name="conversations"/> conversations in parallel for <see cref="Turns"/> turns,
    /// each turn admitted in one step, and returns what every turn of every conversation reused.</summary>
    private static async Task<int[][]> RunAsync(string family, int conversations, Action<OracleModel, int>? beforeTurn = null,
        Action<InferenceEngine, int>? afterTurn = null)
    {
        using var model = Create(family, conversations);
        var config = new SchedulerConfig
        {
            BlockSize = 16, NumBlocks = 1024, MaxNumRunningSequences = conversations, MaxNumBatchedTokens = 512,
            SoloPrefillChunkSize = 512, EnablePrefixCaching = true,
            StopRepetition = false,
        };
        using var engine = new InferenceEngine(model, config, NullLogger.Instance);
        var prompts = Enumerable.Range(0, conversations).Select(FirstPrompt).ToArray();
        var reused = new int[Turns][];
        using var timeout = new CancellationTokenSource(TimeSpan.FromSeconds(60));
        for (int turn = 0; turn < Turns; turn++)
        {
            beforeTurn?.Invoke(model, turn);
            var requests = prompts.Select((p, i) => new SequenceState($"c{i}t{turn}", p.ToList(), MaxNew, 16,
                SamplingConfig.Greedy, cacheScope: $"conversation{i}")).ToArray();
            var gate = new ComputeGate();
            gate.Close();
            engine.ComputeGate = gate;
            var handles = requests.Select(r => engine.SubmitRequest(r)).ToArray();
            gate.Open();
            InferenceCompletion[] done = await Task.WhenAll(handles.Select(h => h.Completion)).WaitAsync(timeout.Token);
            reused[turn] = done.Select(d => d.PrefixCacheReusedTokens).ToArray();
            afterTurn?.Invoke(engine, turn);
            for (int i = 0; i < conversations; i++)
            {
                Assert.Equal(Cold(family, prompts[i], timeout.Token), requests[i].OutputTokens);
                prompts[i] = prompts[i].Concat(requests[i].OutputTokens).Concat(NextUser(i, turn)).ToList();
            }
        }
        return reused;
    }

    private static OracleModel Create(string family, int conversations)
    {
        OracleModel model = AllFamilies.Single(f => f.Name == family).Create();
        if (model.Traits.NativeSlotLimit == 0) return model;
        // Slots for every conversation, running and retained, and the primary.
        var traits = model.Traits with { NativeSlotLimit = 2 * conversations + 1 };
        model.Dispose();
        return new OracleModel(traits);
    }

    /// <summary>The answer a fresh model with no cache gives for <paramref name="prompt"/>.</summary>
    private static List<int> Cold(string family, List<int> prompt, CancellationToken token)
    {
        using var model = AllFamilies.Single(f => f.Name == family).Create();
        var config = new SchedulerConfig { BlockSize = 16, NumBlocks = 64, MaxNumRunningSequences = 1, StopRepetition = false };
        using var session = new CliInferenceSession(model, config, NullLogger.Instance);
        return session.Generate(prompt, MaxNew, SamplingConfig.Greedy, cancellationToken: token,
            enablePrefixCache: false).Tokens;
    }

    // A shared 20-token system prompt, then a user message of its own.
    private static List<int> FirstPrompt(int conversation)
        => Enumerable.Range(10, 20).Concat(Enumerable.Range(0, 20).Select(k => 40 + (conversation * 13 + k * 7) % 190)).ToList();

    private static IEnumerable<int> NextUser(int conversation, int turn)
        => Enumerable.Range(0, 10).Select(k => 40 + (conversation * 17 + turn * 5 + k * 3) % 190);
}
