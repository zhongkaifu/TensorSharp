// Copyright (c) Zhongkai Fu. All rights reserved.
// https://github.com/zhongkaifu/TensorSharp
//
// This file is part of TensorSharp.
//
// TensorSharp is licensed under the BSD-3-Clause license found in the LICENSE file in the root directory of this source tree.
//
// A per-block-capture (recurrent) family used to copy its whole running state into
// EVERY page: on Nemotron-H 8B that is 24 Mamba2 layers, ~99 MiB a page against ~4 MiB
// of K/V, so a 7.2k-token warm-up held ~2.8 GiB of managed slabs. Yet only a block
// captured exactly at its own end (KvBlock.IsRestorablePrefixEnd) can be resumed from,
// and the state copied into any other block was never read. Pages now carry the state
// only at a restore point and K/V rows everywhere else, the pool allocates each slab at
// its real length, and the radix page budget charges what the slabs really hold. Decode
// makes a restore point of every block it fills (each ends at the forward that filled it),
// so inside a reply only the newest keeps its state.
//
// The fake folds every (position, token) into a running state its logits come from, so
// any resume from the wrong state changes the greedy stream; its state section is 64x
// the K/V of a block, roughly Nemotron-H's ratio.
using Microsoft.Extensions.Logging.Abstractions;
using TensorSharp.Runtime.Paged;
using TensorSharp.Runtime.Scheduling;
using TensorSharp.Runtime.Scheduling.PrefixCache;
using InferenceWeb.Tests.PrefixCache;

namespace InferenceWeb.Tests;

[Collection(EngineEnvironmentCollection.Name)]
public sealed class RecurrentPageSlabTests
{
    private const int BlockSize = 8;
    private const int VocabSize = 97;
    private const int KvBytesPerToken = sizeof(int);
    private const int StateBytes = 64 * BlockSize * KvBytesPerToken;
    private const long FullBlockBytes = BlockSize * KvBytesPerToken + StateBytes;
    private const long KvOnlyBlockBytes = BlockSize * KvBytesPerToken;
    private const string Conversation = "conv";

    // ------------------------------------------------------------------ (a) slab contents

    [Fact]
    public async Task NonRestorableBlocks_HoldOnlyKv_RestorePointsHoldTheState()
    {
        // One 24-token forward (the prompt end is split at its last block boundary): blocks
        // 0 and 1 fill in the middle of it, block 2 at its end.
        int[] prompt = Prompt(27, seed: 5);
        var model = new RecurrentStateModel();
        using var engine = new InferenceEngine(model, Config(prefixCaching: true, prefillChunk: 64), NullLogger.Instance);
        await RunAsync(engine, "a", prompt, maxNew: 1);

        Assert.Equal(new[] { 24, 3 }, model.Forwards.Take(2).Select(f => f.Count));
        Assert.Equal(new[] { (0, false), (8, false), (16, true) }, model.Extracts);

        // The three pages the radix cache keeps: a full slab exactly where the block is a
        // restore point, the K/V rows alone everywhere else.
        var pages = TreePages(engine);
        Assert.Equal(3, pages.Count);
        Assert.Equal(2, pages.Count(p => !p.Restorable && p.SlabBytes == KvOnlyBlockBytes));
        Assert.Equal(1, pages.Count(p => p.Restorable && p.SlabBytes == FullBlockBytes));
    }

    [Fact]
    public async Task ModelWithoutAKvOnlyForm_StillWritesEveryBlockInFull()
    {
        int[] prompt = Prompt(27, seed: 5);
        var model = new RecurrentStateModel(kvOnlyForm: false);
        using var engine = new InferenceEngine(model, Config(prefixCaching: true, prefillChunk: 64), NullLogger.Instance);
        await RunAsync(engine, "a", prompt, maxNew: 1);

        Assert.Equal(new[] { (0, true), (8, true), (16, true) }, model.Extracts);
        var pages = TreePages(engine);
        Assert.Equal(3, pages.Count);
        Assert.All(pages, p => Assert.Equal(FullBlockBytes, p.SlabBytes));
        Assert.Equal(3 * FullBlockBytes, engine.RadixCache!.Tree.Cached.HostKv);
    }

    /// <summary>The blocks the radix cache holds as pages, with their flag and slab length.</summary>
    private static List<(bool Restorable, long SlabBytes)> TreePages(InferenceEngine engine)
    {
        PrefixTree tree = engine.RadixCache!.Tree;
        var pages = new List<(bool, long)>();
        for (int id = 0; id < engine.Pool.NumBlocks; id++)
        {
            KvBlock block = engine.Pool.GetBlock(id);
            if (tree.TryGetBlockOwner(block, out _))
                pages.Add((block.IsRestorablePrefixEnd, engine.Pool.Storage.SlabLength(id)));
        }
        return pages;
    }

    // ------------------------------------------------------------------ (b) exact restores

    [Fact]
    public async Task RestoreFromARestorablePage_EqualsAColdRun()
    {
        int[] shared = Prompt(24, seed: 11);
        int[] promptA = shared.Concat(new[] { 7, 8, 9 }).ToArray();
        int[] promptB = shared.Concat(new[] { 20, 21, 22, 23 }).ToArray();
        int[] coldB = await RunColdAsync(promptB, maxNew: 6);

        var model = new RecurrentStateModel();
        using var engine = new InferenceEngine(model, Config(prefixCaching: true, prefillChunk: 64), NullLogger.Instance);
        await RunAsync(engine, "a", promptA, maxNew: 3);
        int injectsBefore = model.KvOnlyInjects + model.FullInjects;

        SequenceState b = await RunAsync(engine, "b", promptB, maxNew: 6);

        // B resumed at the restore point closing block 2, rebuilt from two K/V-only pages and
        // the page that carries the state, and decoded exactly what a cold run decodes.
        Assert.Equal(24, b.PrefixCacheReusedTokens);
        Assert.Equal(3, model.KvOnlyInjects + model.FullInjects - injectsBefore);
        Assert.Equal(2, model.KvOnlyInjects);
        Assert.Equal(1, model.FullInjects);
        Assert.Equal(coldB, b.OutputTokens.ToArray());
    }

    [Fact]
    public async Task PagesWithoutState_AreNeverAResumePoint()
    {
        // C shares only blocks 0 and 1 of A's prompt: both K/V-only, so nothing is resumable
        // and C prefills from zero - with the cold output.
        int[] shared = Prompt(24, seed: 13);
        int[] promptA = shared.Concat(new[] { 7, 8, 9 }).ToArray();
        int[] promptC = shared.Take(2 * BlockSize).Concat(new[] { 40, 41, 42, 43, 44 }).ToArray();
        int[] coldC = await RunColdAsync(promptC, maxNew: 5);

        var model = new RecurrentStateModel();
        using var engine = new InferenceEngine(model, Config(prefixCaching: true, prefillChunk: 64), NullLogger.Instance);
        await RunAsync(engine, "a", promptA, maxNew: 3);
        SequenceState c = await RunAsync(engine, "c", promptC, maxNew: 5);

        Assert.Equal(0, c.PrefixCacheReusedTokens);
        Assert.Equal(coldC, c.OutputTokens.ToArray());
    }

    [Theory]
    [InlineData(true, true)]
    [InlineData(true, false)]
    [InlineData(false, true)]
    public async Task OwnershipSwapsEveryToken_ReinjectMixedSlabs_AndMatchColdRuns(bool kvOnlyForm, bool prefixCaching)
    {
        // Two requests on the single-cache path with a one-token decode quantum: ownership
        // rotates every step, so every swap-in rebuilds the sequence from its blocks - K/V-only
        // blocks, state-carrying ones, and the swapped-out partial tail.
        int[] promptA = Prompt(19, seed: 21);
        int[] promptB = Prompt(21, seed: 22);
        const int maxNew = 14;
        int[] coldA = await RunColdAsync(promptA, maxNew);
        int[] coldB = await RunColdAsync(promptB, maxNew);

        var model = new RecurrentStateModel(kvOnlyForm);
        using var engine = new InferenceEngine(model,
            Config(prefixCaching, prefillChunk: 64, decodeQuantum: 1), NullLogger.Instance);
        var a = new SequenceState("a", promptA.ToList(), maxNew, BlockSize, SamplingConfig.Greedy);
        var b = new SequenceState("b", promptB.ToList(), maxNew, BlockSize, SamplingConfig.Greedy);
        var ha = engine.SubmitRequest(a);
        var hb = engine.SubmitRequest(b);
        await Task.WhenAll(ha.Completion, hb.Completion).WaitAsync(TimeSpan.FromSeconds(30));

        Assert.True(model.FullInjects > 0, "ownership never rotated, so nothing was re-injected");
        if (kvOnlyForm) Assert.True(model.KvOnlyInjects > 0, "no K/V-only block was re-injected");
        Assert.Equal(coldA, a.OutputTokens.ToArray());
        Assert.Equal(coldB, b.OutputTokens.ToArray());
    }

    [Fact]
    public async Task ACompleteInjectEndingOnAKvOnlyBlock_ResumesFromTheLastRestorePoint()
    {
        // A model that does not ask the tree for state at page ends lets B adopt A's first
        // two pages, both K/V-only: every injected block is whole, yet the last one holds no
        // state, so the running state after the inject is the reset one. The executor must
        // not resume there; it backs off to the last restore point (none here) and forwards
        // the prefix again, so B decodes what a cold run decodes.
        int[] shared = Prompt(24, seed: 17);
        int[] promptA = shared.Concat(new[] { 7, 8, 9 }).ToArray();
        int[] promptB = shared.Take(2 * BlockSize).Concat(new[] { 40, 41, 42, 43, 44 }).ToArray();
        int[] coldB = await RunColdAsync(promptB, maxNew: 5);

        var model = new RecurrentStateModel(pagesNeedStateAtEnd: false);
        using var engine = new InferenceEngine(model, Config(prefixCaching: true, prefillChunk: 64), NullLogger.Instance);
        await RunAsync(engine, "a", promptA, maxNew: 3);
        int forwardsBefore = model.Forwards.Count;
        SequenceState b = await RunAsync(engine, "b", promptB, maxNew: 5);

        // Both pages went in, the 16 tokens were forwarded again from zero, and the reuse
        // admission claimed was revoked.
        Assert.Equal(2, model.KvOnlyInjects);
        Assert.Equal(0, model.FullInjects);
        Assert.Contains((0, 2 * BlockSize), model.Forwards.Skip(forwardsBefore));
        Assert.Equal(0, b.PrefixCacheReusedTokens);
        Assert.Equal(coldB, b.OutputTokens.ToArray());
    }

    // ------------------------------------------------------------------ (b2) decode restore points

    [Fact]
    public async Task DecodeFilledBlocks_OnlyTheNewestRestorePointKeepsItsState()
    {
        // 27 prompt tokens forward as 24 + 3, then 22 decode steps fill blocks 3, 4 and 5 one
        // token at a time, each at the forward that filled it - so each is a restore point
        // when captured. Only the newest keeps the running state: the one before is rewritten
        // as K/V rows when the next arrives, and the cache takes the newest once A stops.
        int[] prompt = Prompt(27, seed: 5);
        var model = new RecurrentStateModel();
        using var engine = new InferenceEngine(model, Config(prefixCaching: true, prefillChunk: 64), NullLogger.Instance);
        await RunAsync(engine, "a", prompt, maxNew: 22);

        Assert.Equal(new[]
        {
            (0, false), (8, false), (16, true),   // the prompt: its restore point closes block 2
            (24, true),                           // block 3 fills at token 32
            (32, true), (24, false),              // block 4 replaces it
            (40, true), (32, false),              // block 5 replaces block 4
        }, model.Extracts);

        var pages = TreePages(engine);
        Assert.Equal(6, pages.Count);
        Assert.Equal(2, pages.Count(p => p.Restorable && p.SlabBytes == FullBlockBytes));
        Assert.Equal(4, pages.Count(p => !p.Restorable && p.SlabBytes == KvOnlyBlockBytes));
        PrefixTree tree = engine.RadixCache!.Tree;
        Assert.Equal(2 * FullBlockBytes + 4 * KvOnlyBlockBytes, tree.Cached.HostKv);
        Tk.Valid(tree);
    }

    [Fact]
    public async Task AFollowUpOfTheReply_ResumesFromTheNewestDecodeRestorePoint()
    {
        // A's reply fills blocks 3-5; another chat then takes the model, so A's follow-up
        // (its prompt, its reply, a new turn) can only come back through the pages. It
        // resumes at the end of block 5 - the restore point that kept its state - and
        // decodes what a cold run of the follow-up decodes.
        int[] promptA = Prompt(27, seed: 5);
        var model = new RecurrentStateModel();
        using var engine = new InferenceEngine(model, Config(prefixCaching: true, prefillChunk: 64), NullLogger.Instance);
        SequenceState a = await RunAsync(engine, "a", promptA, maxNew: 22);
        await RunAsync(engine, "x", Prompt(19, seed: 41), maxNew: 2);

        int[] followUp = promptA.Concat(a.OutputTokens).Concat(new[] { 50, 51, 52 }).ToArray();
        int[] cold = await RunColdAsync(followUp, maxNew: 6);
        SequenceState f = await RunAsync(engine, "f", followUp, maxNew: 6);

        Assert.Equal(6 * BlockSize, f.PrefixCacheReusedTokens);
        Assert.Equal(cold, f.OutputTokens.ToArray());
    }

    // ------------------------------------------------------------------ (c) the page budget

    [Fact]
    public async Task PageBudget_ChargesTheRealSlabBytes()
    {
        int[] prompt = Prompt(43, seed: 31);
        var model = new RecurrentStateModel();
        using var engine = new InferenceEngine(model, Config(prefixCaching: true, prefillChunk: 64), NullLogger.Instance);
        await RunAsync(engine, "a", prompt, maxNew: 2);

        PrefixTree tree = engine.RadixCache!.Tree;
        BlockPool pool = engine.Pool;
        long pages = 0, slabBytes = 0, kvOnlyPages = 0;
        for (int id = 0; id < pool.NumBlocks; id++)
        {
            KvBlock block = pool.GetBlock(id);
            if (!tree.TryGetBlockOwner(block, out _)) continue;
            pages++;
            slabBytes += pool.Storage.SlabLength(id);
            if (pool.Storage.SlabLength(id) == KvOnlyBlockBytes) kvOnlyPages++;
        }

        // 43 prompt tokens forward as 40 + 3: blocks 0-3 K/V-only, block 4 the restore point.
        Assert.Equal(5, pages);
        Assert.Equal(4, kvOnlyPages);
        Assert.Equal(pages, tree.Cached.PoolPages);
        Assert.Equal(slabBytes, tree.Cached.HostKv);
        Assert.Equal(4 * KvOnlyBlockBytes + FullBlockBytes, tree.Cached.HostKv);
        Assert.True(tree.Cached.HostKv < pages * FullBlockBytes);
        Tk.Valid(tree);
    }

    [Fact]
    public void Tree_ChargesEachPagesReportedSlab_AndReleasesExactlyThat()
    {
        const long fullPage = 1000, kvPage = 40;
        var host = new FakePageHost(Tk.B);
        PrefixTree t = Tk.Tree(Tk.Caps(needStateAtEnd: true), host: host, pageHostBytes: fullPage);
        int s = Tk.Scope(t);
        RadixNode n = t.Insert(Tk.Key(t, Tk.Seq(1, 100)), 4 * Tk.B, s, 0, NodeFlags.None, null);
        PageRef[] pages = Tk.Pages(host, 4, PageStore.A1HostSlab);
        for (int i = 0; i < 3; i++)
        {
            host.SlabBytes[pages[i].Block] = kvPage;
            pages[i] = pages[i] with { StateAtEnd = false };
        }
        host.SlabBytes[pages[3].Block] = fullPage;
        Assert.Equal(4, t.AttachPages(n, pages));
        Assert.Equal(new ResourceVector { PoolPages = 4, HostKv = 3 * kvPage + fullPage }, t.Cached);
        Tk.Valid(t);

        // A split moves the leading pages, each with its own charge.
        RadixNode branch = t.Insert(Tk.Key(t, Tk.Cat(Tk.Seq(1, 2 * Tk.B), Tk.Seq(500, 10))), 2 * Tk.B + 10, s, 0, NodeFlags.None, null);
        t.CollectIfEmpty(branch);
        Assert.Equal(2 * Tk.B, n.Parent!.Depth);
        Assert.Equal(new ResourceVector { PoolPages = 2, HostKv = 2 * kvPage }, n.Parent.Bytes);
        Assert.Equal(new ResourceVector { PoolPages = 2, HostKv = kvPage + fullPage }, n.Bytes);
        Assert.Equal(3 * kvPage + fullPage, t.Cached.HostKv);
        Tk.Valid(t);

        // Under a host cap that the real bytes fit and full-size charges would not, nothing goes.
        PrefixTree capped = Tk.Tree(Tk.Caps(needStateAtEnd: true), host: host, pageHostBytes: fullPage,
            optionCap: new ResourceVector { HostKv = 3 * kvPage + fullPage + 1 });
        int cs = Tk.Scope(capped);
        RadixNode cn = capped.Insert(Tk.Key(capped, Tk.Seq(1, 100)), 4 * Tk.B, cs, 0, NodeFlags.None, null);
        PageRef[] cpages = Tk.Pages(host, 4, PageStore.A1HostSlab);
        for (int i = 0; i < 4; i++) host.SlabBytes[cpages[i].Block] = i < 3 ? kvPage : fullPage;
        Assert.Equal(4, capped.AttachPages(cn, cpages));
        Assert.True(capped.EnforceCaps(EvictionTier.PublicTop));
        Assert.Equal(4, capped.Cached.PoolPages);

        // Evicting returns exactly what was charged.
        Assert.True(t.Evict(ResourceClass.HostKv, long.MaxValue / 2, ReleaseReason.Evicted, EvictionTier.PublicTop) || t.Cached.HostKv == 0);
        Assert.Equal(0, t.Cached.HostKv);
        Assert.Equal(0, t.Cached.PoolPages);
        Tk.Valid(t);
    }

    [Fact]
    public void Checker_FlagsAPageWhoseSlabChangedUnderItsCharge()
    {
        var host = new FakePageHost(Tk.B);
        PrefixTree t = Tk.Tree(Tk.Caps(needStateAtEnd: true), host: host, pageHostBytes: 1000);
        RadixNode n = t.Insert(Tk.Key(t, Tk.Seq(1, 100)), 2 * Tk.B, Tk.Scope(t), 0, NodeFlags.None, null);
        PageRef[] pages = Tk.Pages(host, 2, PageStore.A1HostSlab);
        host.SlabBytes[pages[0].Block] = 40;
        host.SlabBytes[pages[1].Block] = 1000;
        Assert.Equal(2, t.AttachPages(n, pages));
        Tk.Valid(t);

        // A cached page rewritten in place would leave the budget charging the old length.
        host.SlabBytes[pages[0].Block] = 1000;
        Assert.Contains(PrefixTreeInvariantChecker.CheckOnce(t, default), v => v.Id == "I12");
    }

    [Fact]
    public void HostWithoutSlabLengths_ChargesTheFullPage()
    {
        var host = new FakePageHost(Tk.B);
        PrefixTree t = Tk.Tree(Tk.Caps(), host: host, pageHostBytes: 64);
        RadixNode n = t.Insert(Tk.Key(t, Tk.Seq(1, 100)), 3 * Tk.B, Tk.Scope(t), 0, NodeFlags.None, null);
        t.AttachPages(n, Tk.Pages(host, 3, PageStore.Both));
        Assert.Equal(new ResourceVector { PoolPages = 3, HostKv = 3 * 64 }, t.Cached);
        Tk.Valid(t);
    }

    [Fact]
    public void Storage_AllocatesExactSlabLengths_AndAccountsThem()
    {
        var storage = new PagedKvStorage(4, 100);
        Assert.Equal(40, storage.GetSpan(0, 40).Length);
        Assert.Equal(100, storage.GetSpan(1).Length);
        Assert.Equal(40, storage.GetReadOnlySpan(0).Length);
        Assert.Equal(140, storage.AllocatedBytes);
        Assert.Equal(400, storage.ReservedBytes);

        // Rewriting a block in the other form replaces its slab.
        Assert.Equal(100, storage.GetSpan(0, 100).Length);
        Assert.Equal(100, storage.GetSpan(1, 100).Length);
        Assert.Equal(200, storage.AllocatedBytes);
        Assert.Equal(100, storage.GetSpan(0).Length);
        Assert.Equal(20, storage.GetSpan(0, 20).Length);
        Assert.Equal(100, storage.GetSpan(0).Length);   // the full-size accessor restores a full slab
        storage.ReleaseSlab(0);
        storage.ReleaseSlab(0);
        Assert.Equal(0, storage.SlabLength(0));
        Assert.Equal(100, storage.AllocatedBytes);
        Assert.Throws<ArgumentOutOfRangeException>(() => storage.GetSpan(2, 101));
    }

    // ------------------------------------------------------------------ helpers

    private static int[] Prompt(int length, int seed)
        => Enumerable.Range(0, length).Select(i => 3 + (i * 19 + seed * 7) % 80).ToArray();

    private static SchedulerConfig Config(bool prefixCaching, int prefillChunk, int decodeQuantum = BlockSize) => new()
    {
        MaxNumBatchedTokens = 64,
        MaxNumRunningSequences = 4,
        MaxPrefillChunkSize = prefillChunk,
        NumBlocks = 64,
        BlockSize = BlockSize,
        EnablePrefixCaching = prefixCaching,
        DecodeQuantumTokens = decodeQuantum,
    };

    private static async Task<int[]> RunColdAsync(int[] prompt, int maxNew)
    {
        var model = new RecurrentStateModel();
        using var engine = new InferenceEngine(model, Config(prefixCaching: false, prefillChunk: 64), NullLogger.Instance);
        return (await RunAsync(engine, "cold", prompt, maxNew)).OutputTokens.ToArray();
    }

    private static async Task<SequenceState> RunAsync(InferenceEngine engine, string id, int[] prompt, int maxNew)
    {
        var seq = new SequenceState(id, prompt.ToList(), maxNew, BlockSize, SamplingConfig.Greedy, cacheScope: Conversation);
        var completion = await engine.SubmitRequest(seq).Completion.WaitAsync(TimeSpan.FromSeconds(20));
        Assert.Equal(SequenceStatus.FinishedLengthCapped, completion.Status);
        return seq;
    }

    /// <summary>
    /// A recurrent fake whose block snapshot is its K/V rows (one int per token) plus, in the
    /// full form, its running state: a 64-bit fold of every (position, token) padded to
    /// <see cref="StateBytes"/> with a pattern derived from it. The K/V-only form is the rows
    /// alone; injecting it leaves the running state as it was.
    /// </summary>
    private sealed class RecurrentStateModel(bool kvOnlyForm = true, bool pagesNeedStateAtEnd = true)
        : IModelArchitecture, IPageOnlyPrefixCacheModel
    {
        private const ulong Seed = 1469598103934665603UL;
        private readonly object _gate = new();
        private readonly List<int> _rows = new();
        private ulong _state = Seed;

        public List<(int Start, bool Full)> Extracts { get; } = new();
        public int FullInjects { get; private set; }
        public int KvOnlyInjects { get; private set; }
        public List<(int Start, int Count)> Forwards { get; } = new();

        public PrefixCacheCapabilities GetPrefixCacheCapabilities() => new()
        {
            Class = FamilyClass.R,
            NamespaceFingerprint = KVStateFingerprint,
            EndState = EndStateSupport.None,
            PrimaryResident = true,
            Truncation = TruncationKind.None,
            RewindCapTokens = 16,
            Pages = PageSupport.A1HostSlab,
            PagesNeedStateAtEnd = pagesNeedStateAtEnd,
        };

        public long QuerySpareBytes(ResourceClass cls) => -1;

        public ModelConfig Config { get; } = new() { VocabSize = VocabSize };
        public ITokenizer Tokenizer { get; } = new NumberTokenizer();
        public IMultimodalInjector MultimodalInjector => null;
        public IBackendExecutionPlan ExecutionPlan => null;
        public bool SupportsKVCacheTruncation => false;
        public bool SupportsKVStateSnapshot => true;
        public bool RequiresPerBlockCapture => true;
        public string KVStateFingerprint => "recurrent-state-" + (kvOnlyForm ? "kv-only" : "full");

        public float[] Forward(int[] tokens)
        {
            lock (_gate)
            {
                Forwards.Add((_rows.Count, tokens.Length));
                foreach (int t in tokens)
                {
                    _state = Fold(_state, _rows.Count, t);
                    _rows.Add(t);
                }
                var logits = new float[VocabSize];
                logits[(int)(_state % (VocabSize - 1)) + 1] = 10f;
                return logits;
            }
        }

        public void ResetKVCache() { lock (_gate) { _rows.Clear(); _state = Seed; } }

        public void TruncateKVCache(int tokenCount) => throw new NotSupportedException();

        public void Dispose() { }

        public long ComputeKVBlockByteSize(int tokenCount) => (long)tokenCount * KvBytesPerToken + StateBytes;

        public long ComputeKVBlockByteSizeWithoutRecurrentState(int tokenCount)
            => kvOnlyForm ? (long)tokenCount * KvBytesPerToken : ComputeKVBlockByteSize(tokenCount);

        public bool TryExtractKVBlock(int startToken, int tokenCount, Span<byte> destination)
        {
            lock (_gate)
            {
                if (startToken < 0 || startToken + tokenCount > _rows.Count) return false;
                bool full = destination.Length == ComputeKVBlockByteSize(tokenCount);
                if (!full && destination.Length != ComputeKVBlockByteSizeWithoutRecurrentState(tokenCount)) return false;
                for (int i = 0; i < tokenCount; i++)
                    BitConverter.TryWriteBytes(destination.Slice(i * KvBytesPerToken, KvBytesPerToken), _rows[startToken + i]);
                if (full)
                    WriteState(destination.Slice(tokenCount * KvBytesPerToken, StateBytes), _state);
                Extracts.Add((startToken, full));
                return true;
            }
        }

        public bool TryInjectKVBlock(int destToken, int tokenCount, ReadOnlySpan<byte> source)
        {
            lock (_gate)
            {
                if (destToken != _rows.Count) return false;
                bool full = source.Length == ComputeKVBlockByteSize(tokenCount);
                if (!full && source.Length != ComputeKVBlockByteSizeWithoutRecurrentState(tokenCount)) return false;
                ulong state = _state;
                if (full && !TryReadState(source.Slice(tokenCount * KvBytesPerToken, StateBytes), out state))
                    return false;
                for (int i = 0; i < tokenCount; i++)
                    _rows.Add(BitConverter.ToInt32(source.Slice(i * KvBytesPerToken, KvBytesPerToken)));
                _state = state;
                if (full) FullInjects++; else KvOnlyInjects++;
                return true;
            }
        }

        private static void WriteState(Span<byte> section, ulong state)
        {
            BitConverter.TryWriteBytes(section[..sizeof(ulong)], state);
            for (int i = sizeof(ulong); i < section.Length; i++)
                section[i] = (byte)(state >> (i % 8 * 8));
        }

        private static bool TryReadState(ReadOnlySpan<byte> section, out ulong state)
        {
            state = BitConverter.ToUInt64(section[..sizeof(ulong)]);
            for (int i = sizeof(ulong); i < section.Length; i++)
                if (section[i] != (byte)(state >> (i % 8 * 8))) return false;
            return true;
        }

        private static ulong Fold(ulong h, int position, int token)
        {
            h = (h ^ (ulong)position) * 1099511628211UL;
            return (h ^ (ulong)token) * 1099511628211UL;
        }
    }

    private sealed class NumberTokenizer : ITokenizer
    {
        public string[] Vocab { get; } = Enumerable.Range(0, RecurrentPageSlabTests.VocabSize).Select(i => i.ToString()).ToArray();
        public int BosTokenId => -1;
        public int[] EosTokenIds => Array.Empty<int>();
        public int VocabSize => Vocab.Length;
        public List<int> Encode(string text, bool addSpecial = true) => new();
        public string Decode(List<int> ids) => string.Join(",", ids);
        public void AppendTokenBytes(int tokenId, List<byte> buffer) { }
        public bool IsEos(int tokenId) => false;
        public int LookupToken(string tokenStr) => -1;
    }
}
