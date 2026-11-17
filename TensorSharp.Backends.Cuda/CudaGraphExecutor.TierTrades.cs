// Copyright (c) Zhongkai Fu. All rights reserved.
// https://github.com/zhongkaifu/TensorSharp
//
// This file is part of TensorSharp.
//
// TensorSharp is licensed under the BSD-3-Clause license found in the LICENSE file in the root directory of this source tree.
using System;
using System.Collections.Generic;
using TensorSharp.Cuda.Interop;
using TensorSharp.Graph;

namespace TensorSharp.Cuda
{
    /// <summary>
    /// Adaptive tier trades overlapped with the decode (the removed direct Qwen3.8 engine's asynchronous adaptive residency).
    /// Copied on the step's stream before the next step, a round's trades made the decode wait for
    /// them: up to 64 trades of Qwen3.8 Flash Next UD-Q2_K_XL (about 200 MB) every 12 steps, and the runtime decoded 60.4 /
    /// 56.0 tok/s on an A40 (short prompt / after a 4K one) against the engine's 61.9 / 60.1, whose trades overlap its
    /// decode. Here a round's copies run on a stream of their own in three phases a step boundary apart, so the tables a step
    /// reads never name bytes in flight:
    ///   1. each incoming (host) expert copied into a staging slot from its place in the stack's mapping, and each outgoing
    ///      (device) expert's slot into another; the steps meanwhile read both where they were (both only read);
    ///   2. both experts' entries pointed at their staging copies (the incoming one now computed on the device), then the
    ///      incoming expert copied into the outgoing one's slot, which no entry names any more;
    ///   3. the incoming expert's entries pointed at its slot, the outgoing one's at its bytes in the host's memory.
    /// The staging slots are the prompt arenas' (CudaGraphExecutor.PromptPrefetch's two halves, then the arena of the
    /// shorter chunks), idle while decode steps run: a step that stages experts through them or reads the tier by its
    /// resident copies (a prompt chunk, a step wider than the doorbell) first finishes the round, its stream waiting on the
    /// device for copies not yet landed. Where no arena is allocated a round blocks, as before, at most eight trades.
    /// </summary>
    public sealed partial class CudaGraphExecutor
    {
        /// <summary>Steps (decode steps, verify windows, batched steps: recorded launches and runs) between adaptive rounds,
        /// the Qwen3.8 engines' cadence (AdaptEvery). Counted, never timed, as the phases of a round are: which experts the
        /// device holds at a given step - and so how that step's routes round - follows the steps alone (a quarter-second
        /// timer gave each run of one prompt on Qwen3.8 Flash Next UD-Q2_K_XL its own tokens).</summary>
        internal const int AdaptEvery = 4;

        /// <summary>
        /// Trades a round makes at most: 256 when they overlap the decode (and no more than the staging slots hold,
        /// <see cref="OverlappedTradeCapacity"/>), 8 when they block it, and 64 in the round after a prompt - the Qwen3.8
        /// engines' budgets (AdaptSwapBudget, PromptAdaptSwaps). At 64 a round the answer to a long prompt caught up with
        /// what it routes to over more windows: on an RTX 5090 the engine's first eight 4-row verify windows after a long
        /// prompt took 12.7-13.4 ms against 10.9 at 256. The decode after a 4K-token prompt routes much as the prompt did:
        /// there its first verify windows left 170-290 experts to the CPU against 25-60 once the tiers had caught up.
        /// </summary>
        internal const int OverlappedTradesPerRound = 256, BlockingTradesPerRound = 8, PromptTradesPerRound = 64;

        /// <summary>
        /// Bytes a round's first copies are given a step for: its second phase comes that many steps after it began (one at
        /// least), and a step that comes before the copies landed waits for them on the device. 256 MB is about 11 ms of an
        /// A40's PCIe 4 link from page-locked memory (some 24 GB/s), two thirds of a Qwen3.8 Flash Next UD-Q2_K_XL decode
        /// step at 60 tok/s: a 64-trade round (about 200 MB) takes one step, a full round of 256 four.
        /// </summary>
        private const long TradeBytesPerStep = 256L << 20;

        /// <summary>
        /// Expert ranges the worker page-locks in a step's time before a round's copies begin (at 1 ms a range, the slow end of
        /// the 0.1-1 ms cuMemHostRegister took on an A40, 16 are a Qwen3.8 Flash Next UD-Q2_K_XL decode step): a round whose
        /// outgoing experts are not locked yet begins that many steps after it was proposed (one at least), at the first
        /// decode-size step from then, and waits for the worker if it is late.
        /// </summary>
        private const int LockRangesPerStep = 16;

        /// <summary>A round whose outgoing experts the worker is locking: its proposals, the step its copies may begin at, and
        /// the worker's job.</summary>
        private sealed class LockingRound
        {
            public IReadOnlyList<(int Layer, int Out, int In)> Swaps;
            public Dictionary<int, HostedTables> ByLayer;
            public long DueStep;
            public System.Threading.Tasks.Task Work;
        }

        private LockingRound _lockingRound;

        /// <summary>The worker's last job (its jobs run one after another; disposal waits for it).</summary>
        private System.Threading.Tasks.Task _lockWork;

        private long _tradeLockedBytes;

        /// <summary>Rounds that waited for the worker to lock outgoing experts, the bytes it locked (the locked total's growth:
        /// nothing is unlocked while the model serves), and trades dropped because their outgoing expert could not be locked
        /// and mapped (the pin limit reached; tests, telemetry).</summary>
        internal long LockingRounds { get; private set; }

        /// <inheritdoc cref="LockingRounds"/>
        internal long TradeLockedBytes => System.Threading.Interlocked.Read(ref _tradeLockedBytes);

        /// <inheritdoc cref="LockingRounds"/>
        internal long UnlockableTrades { get; private set; }

        /// <summary>How adaptive trades run on this executor (tests): overlapped where the staging allows (otherwise
        /// blocking), blocking, or not at all.</summary>
        internal enum TierTrades { Overlapped, Blocking, Off }

        /// <inheritdoc cref="TierTrades"/>
        internal TierTrades TierTradesForTests { get; set; } = TierTrades.Overlapped;

        /// <summary>Trades overlapped rounds and blocking rounds committed (tests).</summary>
        internal long OverlappedTrades { get; private set; }

        /// <inheritdoc cref="OverlappedTrades"/>
        internal long BlockingTrades { get; private set; }

        /// <summary>Steps that started while a round's copies were in flight, and of them the recorded steps' launches
        /// (tests check that the trades ran beside the steps).</summary>
        internal long StepsBesideTrades { get; private set; }

        /// <inheritdoc cref="StepsBesideTrades"/>
        internal long RecordedStepsBesideTrades { get; private set; }

        private sealed class PendingTrade
        {
            public HostedTables Tables;
            public int Out, In;
            /// <summary>The outgoing expert's device slot per matrix (gate, up, down): the incoming expert's from phase 3.</summary>
            public readonly IntPtr[] Slot = new IntPtr[3];
            /// <summary>The staging bases of the incoming expert's copy and of the outgoing expert's.</summary>
            public IntPtr InCopy, OutCopy;
            /// <summary>Each matrix's offset in a staging copy, and its bytes.</summary>
            public readonly long[] Offset = new long[3], Bytes = new long[3];
            /// <summary>The incoming expert's place in the layer's packed host copies, per matrix, which the outgoing expert takes
            /// (zero: the outgoing expert is read from the model file's mapping).</summary>
            public readonly IntPtr[] Place = new IntPtr[3];
        }

        private readonly List<PendingTrade> _trades = new();

        // The trades' copy stream; the step stream's point the copies start after; each phase's copies landed.
        private IntPtr _tradeStream, _tradeGate, _tradeStaged, _tradePlaced;

        /// <summary>The phase of the round in flight (0: none; 1: copying into staging; 2: copying into the slots), and the
        /// step at which it advances.</summary>
        private int _tradePhase;
        private long _tradeDueStep;

        /// <summary>Steps counted so far, and since the last round began.</summary>
        private long _stepOrdinal;
        private int _stepsSinceRound;

        /// <summary>A harvest made a round due; a step wider than the doorbell ran since the last decode-size step (the round
        /// after a prompt is due). Either begins at the next decode-size step (<see cref="BeginDueRound"/>).</summary>
        private bool _roundDue, _afterPrompt;

        /// <summary>
        /// A step is about to run (a recorded step's launch, or a run's start): counted (a recording once, by its launch),
        /// the prompt chunks' routes handed to the expert hosts, a round in flight advanced when its next phase is due, and
        /// every <see cref="AdaptEvery"/> steps the device's route counts harvested for the next round. Nothing else happens
        /// inside a recording.
        /// </summary>
        private void StepBoundary(bool launch)
        {
            var cuda = (CudaAllocator)Allocator;
            bool capturing = CudaGraphCapture.IsCapturing(cuda);
            if (launch || !capturing)
            {
                _stepOrdinal++;
                if (_stepsSinceRound < int.MaxValue) _stepsSinceRound++;
            }
            if (capturing || _hostedTables.Count == 0) return;
            HarvestPromptCounts();
            AdvanceTrades(force: false);
            if (_stepsSinceRound >= AdaptEvery && _tradePhase == 0 && _lockingRound == null && !_roundDue)
                _roundDue = HarvestRouteCounts();
        }

        /// <summary>
        /// At a decode-size step (a recorded launch, or a doorbell call outside a recording): a round whose outgoing experts
        /// the worker locked, once its step has come; else the round after a prompt, else the round the last harvest made due,
        /// when none is in flight - begun here, its copies beside this step and the next ones. A round begins only here: begun
        /// before a prompt chunk, it would hold that chunk's first layer until its copies landed. True when an overlapped round
        /// began.
        /// </summary>
        private bool BeginDueRound()
        {
            if (_tradePhase != 0 || CudaGraphCapture.IsCapturing((CudaAllocator)Allocator)) return false;
            if (_lockingRound != null)
                return _stepOrdinal >= _lockingRound.DueStep && BeginLockedRound();
            if (!_afterPrompt && !_roundDue) return false;
            bool afterPrompt = _afterPrompt;
            _afterPrompt = _roundDue = false;
            _stepsSinceRound = 0;
            return AdaptTiers(HostedLayers(), afterPrompt);
        }

        /// <summary>
        /// An adaptive round (<see cref="IGraphExpertHost.ProposeSwaps"/>) over this executor's layers. A trade's outgoing
        /// expert must be the device's to read at its host place from the step it leaves the tier (hybrid misses, a prompt
        /// chunk's device staging). With the host experts packed in pinned host memory (<see cref="HostExpertCopies"/>, the
        /// direct engines' layout) it takes the incoming expert's place, its bytes copied there by the round itself: nothing is
        /// registered or unlocked while the model serves. Where they are read from the model file's mapping instead (the host
        /// could not pin the copies), a worker locks the outgoing experts' ranges and the round begins at a counted step after
        /// (<see cref="LockRangesPerStep"/>); nothing is unlocked - replacing a lock takes its pages' other experts off the
        /// device for the while: after every 4-step round on an A40 that held Qwen3.8 Flash Next UD-Q2_K_XL's decode at 52.8 /
        /// 48.1 tok/s against 60-61 / 56-57 - and an outgoing expert past the pin limit drops its trade. True when an
        /// overlapped round began.
        /// </summary>
        private bool AdaptTiers(HostedTables[] layers, bool afterPrompt)
        {
            if (layers.Length == 0 || TierTradesForTests == TierTrades.Off) return false;
            int capacity = OverlappedTradeCapacity(layers);
            int budget = capacity > 0 ? Math.Min(afterPrompt ? PromptTradesPerRound : OverlappedTradesPerRound, capacity)
                : afterPrompt ? PromptTradesPerRound : BlockingTradesPerRound;
            IReadOnlyList<(int Layer, int Out, int In)> swaps = layers[0].Host.ProposeSwaps(budget);
            if (swaps.Count == 0) return false;
            var byLayer = new Dictionary<int, HostedTables>();
            foreach (HostedTables t in layers) byLayer[t.Layer] = t;
            var ranges = new List<(IntPtr Ptr, long Bytes)>();
            foreach (var (layer, outOf, into) in swaps)
            {
                if (!byLayer.TryGetValue(layer, out HostedTables t) || t.Sets == null || TakesPlace(t, into)) continue;
                for (int m = 0; m < 3; m++)
                {
                    long bytes = ExpertBytesOf(t.Sets[m], outOf);
                    IntPtr host = bytes > 0 ? HostBytes(t.Sets[m], outOf) : IntPtr.Zero;
                    if (host != IntPtr.Zero && CudaHostPins.DevicePointer(host, bytes) == IntPtr.Zero) ranges.Add((host, bytes));
                }
            }
            if (ranges.Count == 0) return BeginTrades(swaps, byLayer);
            // The worker locks them (its jobs one after another), off this thread: 0.1-1 ms a range on an A40, and a round
            // moves up to 256 experts x 3.
            var cuda = (CudaAllocator)Allocator;
            _tradedPins ??= new CudaHostPins();
            CudaHostPins pins = _tradedPins;
            System.Threading.Tasks.Task previous = _lockWork ?? System.Threading.Tasks.Task.CompletedTask;
            _lockWork = previous.ContinueWith(_ =>
            {
                cuda.Context.MakeCurrent();
                foreach ((IntPtr ptr, long bytes) in ranges)
                {
                    pins.LockOutside(ptr, bytes, out long registered);
                    System.Threading.Interlocked.Add(ref _tradeLockedBytes, registered);
                }
            }, System.Threading.Tasks.TaskScheduler.Default);
            _lockingRound = new LockingRound
            {
                Swaps = swaps, ByLayer = byLayer, Work = _lockWork,
                DueStep = _stepOrdinal + Math.Max(1, (ranges.Count + LockRangesPerStep - 1) / LockRangesPerStep),
            };
            LockingRounds++;
            return false;
        }

        /// <summary>Tests: host experts (outside the tier) the device cannot read where they lie (Mask 0), across the layers.</summary>
        internal int UnmappedHostExpertsForTests()
        {
            int unmapped = 0;
            foreach (HostedTables t in HostedLayers())
                foreach (byte m in t.Mask)
                    if (m == 0) unmapped++;
            return unmapped;
        }

        /// <summary>Tests: host bytes this executor page-locked (a trade's outgoing experts, a prompt's staging sources; none
        /// where the host experts are packed).</summary>
        internal long TradePinnedBytesForTests => _tradedPins?.OwnBytes ?? 0;

        /// <summary>Tests: a round waiting for its locks begun now, and the round in flight landed.</summary>
        internal void FinishTradesForTests()
        {
            if (_lockingRound != null && _tradePhase == 0) BeginLockedRound();
            AdvanceTrades(force: true);
        }

        /// <summary>The locking round's copies begun (the worker waited for where it is late).</summary>
        private bool BeginLockedRound()
        {
            LockingRound round = _lockingRound;
            _lockingRound = null;
            try { round.Work.Wait(); }
            catch (AggregateException e)
            {
                Console.Error.WriteLine($"[cuda] adaptive tiers: page-locking the outgoing experts failed ({e.InnerException?.Message}); "
                    + "their trades are dropped.");
            }
            return BeginTrades(round.Swaps, round.ByLayer);
        }

        /// <summary>The trades whose outgoing expert is locked and mapped begun, overlapped where the prompt arenas can stage
        /// them, else blocking; the others dropped. True when an overlapped round began.</summary>
        private bool BeginTrades(IReadOnlyList<(int Layer, int Out, int In)> swaps, Dictionary<int, HostedTables> byLayer)
        {
            var kept = new List<(int Layer, int Out, int In)>(swaps.Count);
            foreach ((int layer, int outOf, int into) in swaps)
            {
                if (byLayer.TryGetValue(layer, out HostedTables t) && t.Sets != null && (TakesPlace(t, into) || HostMapped(t.Sets, outOf)))
                    kept.Add((layer, outOf, into));
                else UnlockableTrades++;
            }
            if (kept.Count == 0) return false;
            if (OverlappedTradeCapacity(HostedLayers()) > 0 && BeginOverlappedTrades(kept, byLayer)) return true;
            TradeBlocking(kept, byLayer);
            return false;
        }

        /// <summary>Whether a trade bringing <paramref name="into"/> into the tier gives its outgoing partner its place in the
        /// layer's packed host copies.</summary>
        private static bool TakesPlace(HostedTables t, int into) => t.Copies != null && t.Copies.PlaceOf(into) >= 0;

        /// <summary>An expert's bytes in a stack (0 where it has no member weight).</summary>
        private static long ExpertBytesOf(IGraphExpertWeights set, int e)
        {
            try { return set.Expert(e).Bytes; }
            catch (NotSupportedException) { return 0; }
        }

        /// <summary>Whether every matrix of expert <paramref name="e"/> is page-locked and mapped where it lies in the host's
        /// memory, so <see cref="MapHostExpert"/> maps it.</summary>
        private static bool HostMapped(IGraphExpertWeights[] sets, int e)
        {
            for (int m = 0; m < 3; m++)
            {
                long bytes = ExpertBytesOf(sets[m], e);
                IntPtr host = bytes > 0 ? HostBytes(sets[m], e) : IntPtr.Zero;
                if (bytes % 16 != 0 || host.ToInt64() % 16 != 0 || CudaHostPins.DevicePointer(host, bytes) == IntPtr.Zero) return false;
            }
            return true;
        }

        private HostedTables[] HostedLayers()
        {
            lock (_hostedTables) return new List<HostedTables>(_hostedTables.Values).ToArray();
        }

        private static long Align256(long bytes) => (bytes + 255) / 256 * 256;

        /// <summary>An expert's staging slot: its three matrices, each from a 256-byte boundary (the kernels' aligned words).</summary>
        private static long TradeStride(long[] expertBytes) => Align256(expertBytes[0]) + Align256(expertBytes[1]) + Align256(expertBytes[2]);

        /// <summary>The device memory a round's staging copies borrow: the long prompt chunks' arena (both halves), then the
        /// shorter chunks' arena. Both are sized at the first expert step and read only by prompt steps.</summary>
        private List<(IntPtr Base, long Bytes)> TradeStaging()
        {
            var regions = new List<(IntPtr, long)>(2);
            if (_prefetchArena != IntPtr.Zero && _halfBytes > 0) regions.Add((_prefetchArena, 2 * _halfBytes));
            if (_stagingArena != IntPtr.Zero && _stagingBytes > 0) regions.Add((_stagingArena, _stagingBytes));
            return regions;
        }

        /// <summary>Trades one overlapped round can stage: two slots each, of the layers' largest expert (0: the round blocks).
        /// Qwen3.8 Flash Next UD-Q2_K_XL's experts are about 3.1 MB, so 1.6 GB of arenas carry a full round of 256.</summary>
        private int OverlappedTradeCapacity(HostedTables[] layers)
        {
            if (TierTradesForTests != TierTrades.Overlapped) return 0;
            long stride = 0;
            foreach (HostedTables t in layers)
                if (t.ExpertBytes != null) stride = Math.Max(stride, TradeStride(t.ExpertBytes));
            if (stride == 0) return 0;
            long slots = 0;
            foreach ((IntPtr Base, long Bytes) region in TradeStaging()) slots += region.Bytes / stride;
            return (int)Math.Min(int.MaxValue, slots / 2);
        }

        private bool EnsureTradeStream()
        {
            if (_tradeStream != IntPtr.Zero) return true;
            if (CudaDriverApi.cuStreamCreate(out IntPtr stream, 0x1 /* NON_BLOCKING */) != 0) return false;
            var events = new IntPtr[3];
            for (int i = 0; i < events.Length; i++)
            {
                if (CudaDriverApi.cuEventCreate(out events[i], 0x2 /* DISABLE_TIMING */) == 0) continue;
                for (int j = 0; j < i; j++) CudaDriverApi.cuEventDestroy(events[j]);
                CudaDriverApi.cuStreamDestroy(stream);
                return false;
            }
            _tradeStream = stream;
            (_tradeGate, _tradeStaged, _tradePlaced) = (events[0], events[1], events[2]);
            return true;
        }

        private static void SetWord(int[] words, int e, long p)
        {
            words[2 * e] = unchecked((int)p);
            words[2 * e + 1] = unchecked((int)(p >> 32));
        }

        /// <summary>
        /// Phase 1 of an overlapped round: each trade's resident copies handed from the outgoing expert's keys to the incoming
        /// one's (the bytes follow in phase 2; nothing reads the tier by its keys while a round is in flight, see
        /// <see cref="FlushTrades"/>), and both experts copied into staging slots on the trades' stream, after everything the
        /// step's stream queued so far. False, with nothing begun, where no trade qualifies or there is no stream.
        /// </summary>
        private bool BeginOverlappedTrades(IReadOnlyList<(int Layer, int Out, int In)> swaps, Dictionary<int, HostedTables> byLayer)
        {
            var cuda = (CudaAllocator)Allocator;
            cuda.Context.MakeCurrent();
            if (!EnsureTradeStream()) return false;
            List<(IntPtr Base, long Bytes)> regions = TradeStaging();
            int region = 0;
            long used = 0;
            IntPtr Slot(long stride)
            {
                while (region < regions.Count && used + stride > regions[region].Bytes)
                {
                    region++;
                    used = 0;
                }
                if (region == regions.Count) return IntPtr.Zero;
                IntPtr p = regions[region].Base + (nint)used;
                used += stride;
                return p;
            }
            CudaDriverApi.cuEventRecord(_tradeGate, cuda.Stream.Handle).ThrowOnError();
            CudaDriverApi.cuStreamWaitEvent(_tradeStream, _tradeGate, 0).ThrowOnError();
            // And after the long chunks' copies into the halves (one staged ahead for a layer that never took it may still run).
            for (int h = 0; h < 2; h++)
                if (_halfReady[h] != IntPtr.Zero) CudaDriverApi.cuStreamWaitEvent(_tradeStream, _halfReady[h], 0).ThrowOnError();
            var fromKeys = new IntPtr[3];
            var toKeys = new IntPtr[3];
            var types = new int[3];
            var bytes = new long[3];
            var ptrs = new IntPtr[3];
            var taken = new HashSet<(HostedTables, int)>();
            long copied = 0;
            foreach ((int layer, int outOf, int into) in swaps)
            {
                if (!byLayer.TryGetValue(layer, out HostedTables t) || t.Sets == null || t.ExpertBytes == null || t.Mask[outOf] != 1
                    || t.Mask[into] == 1 || taken.Contains((t, outOf)) || taken.Contains((t, into)))
                    continue;
                bool ok = true;
                for (int m = 0; m < 3 && ok; m++)
                {
                    IGraphExpertWeights set = t.Sets[m];
                    IGraphWeight a = set.Expert(outOf), b = set.Expert(into);
                    fromKeys[m] = a.CacheKey;
                    toKeys[m] = b.CacheKey;
                    types[m] = set.Format;
                    bytes[m] = b.Bytes;
                    ok = a.Bytes == b.Bytes && b.Bytes == t.ExpertBytes[m] && set.Data != IntPtr.Zero;
                }
                // The incoming expert locked and mapped (Mask 2): a copy from pageable pages would hold this thread until the
                // trades' stream reached it, behind every step queued so far, and locking it here would put the driver's
                // registration (0.1-1 ms a range on an A40) between two steps. An expert the host could not lock stays out.
                if (!ok || t.Mask[into] != 2) continue;
                (int atRegion, long atUsed) = (region, used);
                long stride = TradeStride(t.ExpertBytes);
                IntPtr inCopy = Slot(stride), outCopy = Slot(stride);
                if (inCopy == IntPtr.Zero || outCopy == IntPtr.Zero) break;   // the staging is full
                if (!CudaQuantizedOps.TryMoveResident(cuda, fromKeys, toKeys, types, bytes, ptrs))
                {
                    (region, used) = (atRegion, atUsed);
                    continue;
                }
                var trade = new PendingTrade { Tables = t, Out = outOf, In = into, InCopy = inCopy, OutCopy = outCopy };
                bool place = TakesPlace(t, into);
                long offset = 0;
                for (int m = 0; m < 3; m++)
                {
                    trade.Slot[m] = ptrs[m];
                    trade.Offset[m] = offset;
                    trade.Bytes[m] = bytes[m];
                    trade.Place[m] = place ? t.Copies.Bytes(m, into) : IntPtr.Zero;
                    CudaHostPins.CopyToDevice(inCopy + (nint)offset, HostBytes(t, m, into), bytes[m], _tradeStream).ThrowOnError();
                    CudaDriverApi.cuMemcpyDtoDAsync(outCopy + (nint)offset, ptrs[m], new UIntPtr((ulong)bytes[m]), _tradeStream).ThrowOnError();
                    offset += Align256(bytes[m]);
                    copied += bytes[m];
                }
                taken.Add((t, outOf));
                taken.Add((t, into));
                _trades.Add(trade);
            }
            if (_trades.Count == 0) return false;
            CudaDriverApi.cuEventRecord(_tradeStaged, _tradeStream).ThrowOnError();
            // The long chunks' arena holds no layer staged ahead any more.
            _halfLayer[0] = _halfLayer[1] = -1;
            _tradePhase = 1;
            _tradeDueStep = _stepOrdinal + Math.Max(1, (copied + TradeBytesPerStep - 1) / TradeBytesPerStep);
            return true;
        }

        /// <summary>At a step boundary: the round in flight's next phase when its step has come (one phase a boundary); with
        /// <paramref name="force"/> every phase left, now.</summary>
        private void AdvanceTrades(bool force)
        {
            while (_tradePhase != 0 && (force || _stepOrdinal >= _tradeDueStep))
            {
                if (_tradePhase == 1) TradesToStaging();
                else TradesToSlots();
                if (!force) return;
            }
        }

        /// <summary>
        /// The round in flight finished before a step stages experts through the arenas it borrows or reads the tier by its
        /// resident copies (a prompt chunk, a step wider than the doorbell, the host-routed fallback): its phases run now and
        /// the step's stream waits on the device for copies not yet landed (the host never waits). A recording cannot hold
        /// that: it is abandoned and the step runs as usual.
        /// </summary>
        private void FlushTrades()
        {
            // A round still waiting for its outgoing experts' locks is dropped (its copies not begun; the locks the worker makes
            // stay, as every lock does while the model serves): the counts keep its trades for the next round.
            _lockingRound = null;
            if (_tradePhase == 0) return;
            if (CudaGraphCapture.IsCapturing((CudaAllocator)Allocator))
                throw new CudaGraphCaptureAbortedException("an adaptive trade round is in flight; a recorded step cannot finish it");
            AdvanceTrades(force: true);
        }

        /// <summary>Phase 2: once the staging copies landed (the step's stream waits for them), both experts' entries on their
        /// staging copies; then, after every step queued so far (which may read the slot through the old entries, or the
        /// incoming expert's packed place through its host routes), the incoming expert into the outgoing one's slot and the
        /// outgoing one into the incoming one's packed place.</summary>
        private void TradesToStaging()
        {
            var cuda = (CudaAllocator)Allocator;
            cuda.Context.MakeCurrent();
            CudaDriverApi.cuStreamWaitEvent(cuda.Stream.Handle, _tradeStaged, 0).ThrowOnError();
            var changed = new HashSet<HostedTables>();
            foreach (PendingTrade s in _trades)
            {
                HostedTables t = s.Tables;
                for (int m = 0; m < 3; m++)
                {
                    SetWord(t.Words[m], s.In, (s.InCopy + (nint)s.Offset[m]).ToInt64());
                    SetWord(t.Words[m], s.Out, (s.OutCopy + (nint)s.Offset[m]).ToInt64());
                }
                t.Mask[s.In] = 1;   // the device computes it from here
                changed.Add(t);
            }
            foreach (HostedTables t in changed)
            {
                TierMoved(t);
                UploadTables(t);
            }
            CudaDriverApi.cuEventRecord(_tradeGate, cuda.Stream.Handle).ThrowOnError();
            CudaDriverApi.cuStreamWaitEvent(_tradeStream, _tradeGate, 0).ThrowOnError();
            foreach (PendingTrade s in _trades)
                for (int m = 0; m < 3; m++)
                {
                    CudaDriverApi.cuMemcpyDtoDAsync(s.Slot[m], s.InCopy + (nint)s.Offset[m], new UIntPtr((ulong)s.Bytes[m]), _tradeStream).ThrowOnError();
                    // The outgoing expert into the incoming one's packed place: no table names it any more, and every step queued
                    // before - whose host routes may have read it - is behind the gate.
                    if (s.Place[m] != IntPtr.Zero)
                        CudaDriverApi.cuMemcpyDtoHAsync(s.Place[m], s.OutCopy + (nint)s.Offset[m], new UIntPtr((ulong)s.Bytes[m]), _tradeStream).ThrowOnError();
                }
            CudaDriverApi.cuEventRecord(_tradePlaced, _tradeStream).ThrowOnError();
            _tradePhase = 2;
            _tradeDueStep = _stepOrdinal + 1;
        }

        /// <summary>Phase 3: once the slot copies landed, the incoming experts on their slots and the outgoing ones at their
        /// bytes in the host's memory (<see cref="MapHostExpert"/>: the incoming expert's packed place, which the outgoing one
        /// now holds, or its place in the mapping); the tiers follow.</summary>
        private void TradesToSlots()
        {
            var cuda = (CudaAllocator)Allocator;
            cuda.Context.MakeCurrent();
            CudaDriverApi.cuStreamWaitEvent(cuda.Stream.Handle, _tradePlaced, 0).ThrowOnError();
            var changed = new HashSet<HostedTables>();
            foreach (PendingTrade s in _trades)
            {
                HostedTables t = s.Tables;
                for (int m = 0; m < 3; m++) SetWord(t.Words[m], s.In, s.Slot[m].ToInt64());
                if (s.Place[0] != IntPtr.Zero) t.Copies.Move(s.In, s.Out);
                MapHostExpert(t.Sets, t.Copies, s.Out, t.Words, t.Mask);
                changed.Add(t);
            }
            foreach (HostedTables t in changed)
            {
                TierMoved(t);
                UploadTables(t);
            }
            foreach (PendingTrade s in _trades)
            {
                s.Tables.Host.CommitSwap(s.Tables.Layer, s.Out, s.In);
                OverlappedTrades++;
            }
            // A prompt's copies into the long chunks' arena wait for its halves to be free: after the round's last reads too.
            for (int h = 0; h < 2; h++)
                if (_halfFree[h] != IntPtr.Zero) CudaDriverApi.cuEventRecord(_halfFree[h], cuda.Stream.Handle).ThrowOnError();
            _trades.Clear();
            _tradePhase = 0;
        }

        /// <summary>
        /// A blocking round: each trade's incoming expert copied into the outgoing one's device slot from its packed place (or
        /// the stack's page-locked mapping), the outgoing one into that packed place through a device copy of it, the resident
        /// copies re-keyed and the layer's tables and flags rewritten, all on the step's stream before the next step (a
        /// recorded step reads the tables when it runs; every step queued before has read the place). The outgoing expert's
        /// entries point where it lies in the host's memory when the device can read it there (<see cref="MapHostExpert"/>).
        /// </summary>
        private void TradeBlocking(IReadOnlyList<(int Layer, int Out, int In)> swaps, Dictionary<int, HostedTables> byLayer)
        {
            var cuda = (CudaAllocator)Allocator;
            var dirty = new HashSet<HostedTables>();
            var fromKeys = new IntPtr[3];
            var toKeys = new IntPtr[3];
            var types = new int[3];
            var bytes = new long[3];
            var ptrs = new IntPtr[3];
            cuda.Context.MakeCurrent();
            foreach ((int layer, int outOf, int into) in swaps)
            {
                // (The incoming expert locked and mapped, as an overlapped round's.)
                if (!byLayer.TryGetValue(layer, out HostedTables t) || t.Sets == null || t.Mask[outOf] != 1 || t.Mask[into] != 2) continue;
                bool ok = true;
                for (int m = 0; m < 3 && ok; m++)
                {
                    IGraphExpertWeights set = t.Sets[m];
                    IGraphWeight a = set.Expert(outOf), b = set.Expert(into);
                    fromKeys[m] = a.CacheKey;
                    toKeys[m] = b.CacheKey;
                    types[m] = set.Format;
                    bytes[m] = b.Bytes;
                    ok = a.Bytes == b.Bytes && set.Data != IntPtr.Zero;
                }
                if (!ok || !CudaQuantizedOps.TryMoveResident(cuda, fromKeys, toKeys, types, bytes, ptrs)) continue;
                bool place = TakesPlace(t, into);
                using Tensor held = place ? new Tensor(Allocator, DType.UInt8, Math.Max(bytes[0], Math.Max(bytes[1], bytes[2]))) : null;
                for (int m = 0; m < 3; m++)
                {
                    IntPtr packed = place ? t.Copies.Bytes(m, into) : IntPtr.Zero;
                    // (The scratch is reused matrix by matrix: the step's stream orders its copies.)
                    if (place)
                        CudaDriverApi.cuMemcpyDtoDAsync(Ptr(held), ptrs[m], new UIntPtr((ulong)bytes[m]), cuda.Stream.Handle).ThrowOnError();
                    CudaHostPins.CopyToDevice(ptrs[m], HostBytes(t, m, into), bytes[m], cuda.Stream.Handle).ThrowOnError();
                    if (place)
                        CudaDriverApi.cuMemcpyDtoHAsync(packed, Ptr(held), new UIntPtr((ulong)bytes[m]), cuda.Stream.Handle).ThrowOnError();
                    SetWord(t.Words[m], into, ptrs[m].ToInt64());
                }
                t.Mask[into] = 1;
                if (place) t.Copies.Move(into, outOf);
                MapHostExpert(t.Sets, t.Copies, outOf, t.Words, t.Mask);
                TierMoved(t);
                t.Host.CommitSwap(layer, outOf, into);
                BlockingTrades++;
                dirty.Add(t);
            }
            foreach (HostedTables t in dirty)
                UploadTables(t);
        }

        /// <summary>The trades' stream and events released (any round finished first, <see cref="FlushTrades"/>), after the
        /// worker's last locks (they register pages of the mappings, with the executor's page locks).</summary>
        private void DisposeTrades()
        {
            _lockingRound = null;
            if (_lockWork != null)
            {
                try { _lockWork.Wait(); }
                catch (AggregateException) { }
                _lockWork = null;
            }
            if (_tradeStream == IntPtr.Zero) return;
            ((CudaAllocator)Allocator).Context.MakeCurrent();
            CudaDriverApi.cuStreamSynchronize(_tradeStream);
            foreach (IntPtr e in new[] { _tradeGate, _tradeStaged, _tradePlaced })
                if (e != IntPtr.Zero) CudaDriverApi.cuEventDestroy(e);
            CudaDriverApi.cuStreamDestroy(_tradeStream);
            _tradeStream = _tradeGate = _tradeStaged = _tradePlaced = IntPtr.Zero;
            _trades.Clear();
            _tradePhase = 0;
        }
    }
}
