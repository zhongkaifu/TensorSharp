// Copyright (c) Zhongkai Fu. All rights reserved.
// https://github.com/zhongkaifu/TensorSharp
//
// This file is part of TensorSharp.
//
// TensorSharp is licensed under the BSD-3-Clause license found in the LICENSE file in the root directory of this source tree.
using System;
using System.Collections.Generic;
using System.Globalization;
using System.Linq;
using TensorSharp.Runtime.Scheduling.Diffusion;

namespace TensorSharp.Models.SharedDiffusion
{
    /// <summary>The components a diffusion pipeline keeps resident or releases.</summary>
    public enum DiffusionComponent { Conditioner, Vae, Denoiser, AudioVae }

    /// <summary>What one phase needs on the device: its component's weights and its working memory.</summary>
    public readonly record struct PhaseDemand(DiffusionPhase Phase, DiffusionComponent Component, long WorkingBytes)
    {
        /// <summary>The fewest of its component's weight bytes the phase must hold on the device at once when it streams its
        /// weights through the device a group at a time (<see cref="IResidentComponent.StreamedWeightBytes"/>); 0 when the
        /// phase holds every weight, as a component that cannot stream does.</summary>
        public long StreamedWeightBytes { get; init; }
    }

    /// <summary>A phase that holds at most <see cref="WeightBudget"/> of its component's weight bytes on the device at once,
    /// streaming the rest through it a group at a time (<see cref="IResidentComponent.StreamWithin"/>). The component holds
    /// none of them after the phase.</summary>
    public readonly record struct ResidencyStream(DiffusionPhase Phase, DiffusionComponent Component, long WeightBudget);

    /// <summary>What a pipeline drops before a phase: a component's weights and workspace (<see cref="Weights"/>), or
    /// only its workspace (activations, compute buffers).</summary>
    public readonly record struct ResidencyRelease(DiffusionPhase Before, DiffusionComponent Component, bool Weights);

    /// <summary>
    /// A request's residency decision: the releases to make before each phase, or none (<see cref="KeepsAll"/>, the
    /// behaviour of a device that holds every component). A phase whose component and working memory do not fit the budget
    /// even with every other component released streams its weights where its component can (<see cref="Streams"/>).
    /// <see cref="Fits"/> is false when some phase alone needs more than the budget even so; the plan then keeps as little
    /// resident as it can and the run goes on (the device's own limits decide).
    /// </summary>
    public sealed class ResidencyPlan
    {
        internal ResidencyPlan(long? budget, long peak, bool keepsAll, bool fits, IReadOnlyList<ResidencyRelease> releases,
            long? capacity = null, PhaseDemand limiting = default, long limitingWeights = 0,
            IReadOnlyList<ResidencyStream> streams = null)
        {
            Budget = budget;
            Peak = peak;
            KeepsAll = keepsAll;
            Fits = fits;
            Releases = releases;
            Capacity = capacity;
            Limiting = limiting;
            LimitingWeights = limitingWeights;
            Streams = streams ?? Array.Empty<ResidencyStream>();
        }

        /// <summary>Device bytes the components may use (free bytes less headroom, plus what they hold now), or null when
        /// the backend reports none (the CPU backends): everything stays resident. On Metal the budget is the recommended
        /// working set, a hard ceiling on unified memory.</summary>
        public long? Budget { get; }

        /// <summary>The largest device use the plan reaches.</summary>
        public long Peak { get; }

        public bool KeepsAll { get; }
        public bool Fits { get; }
        public IReadOnlyList<ResidencyRelease> Releases { get; }

        /// <summary>The phases that stream their component's weights through the device (none when every phase fits).</summary>
        public IReadOnlyList<ResidencyStream> Streams { get; }

        /// <summary>The weight bytes <paramref name="phase"/> may hold at once when it streams, or null when it holds them
        /// all.</summary>
        public long? StreamBudget(DiffusionPhase phase)
        {
            foreach (ResidencyStream s in Streams)
                if (s.Phase == phase) return s.WeightBudget;
            return null;
        }

        /// <summary>The most the device can give the components: its free bytes with no headroom, plus what they hold now;
        /// null where the backend reports no device memory (nothing is refused there).</summary>
        public long? Capacity { get; }

        /// <summary>The phase that needs the most with every other component released: its own component's weights (the
        /// fewest it streams with, where it can stream) and its working memory (<see cref="MinimumPeak"/>). No plan can go
        /// below it.</summary>
        public PhaseDemand Limiting { get; }

        /// <summary>The weights of <see cref="Limiting"/>'s component the phase must hold at once: all of them, or the fewest
        /// it streams with (<see cref="PhaseDemand.StreamedWeightBytes"/>).</summary>
        public long LimitingWeights { get; }

        /// <summary>The smallest peak any plan can reach: <see cref="Limiting"/>'s weights and working memory.</summary>
        public long MinimumPeak => LimitingWeights + Limiting.WorkingBytes;

        /// <summary>
        /// True when the request cannot run on this device at all: even with every other component released, its most
        /// demanding phase needs more than <see cref="Capacity"/> by more than <see cref="ResidencyPlanner.RefusalAllowanceBytes"/>.
        /// The run would fail inside the phase (on ggml-cuda, an allocation that fails inside a kernel ends the process), so
        /// the pipeline refuses it before any phase runs. Short of that the plan's own headroom is not a reason to refuse: a
        /// phase may use part of the headroom (bf16 at 4096x4096 on the 46 GB A40 ran at 42.6 GiB with the plan over its
        /// budget).
        /// </summary>
        public bool Refused => Capacity is long capacity && MinimumPeak > capacity + ResidencyPlanner.RefusalAllowanceBytes;

        /// <summary>The releases to make before <paramref name="phase"/>, in order.</summary>
        public IEnumerable<ResidencyRelease> Before(DiffusionPhase phase) => Releases.Where(r => r.Before == phase);

        /// <summary>One line for the log and telemetry.</summary>
        public string Describe()
        {
            string budget = Budget is long b ? Gb(b) : "unknown";
            if (KeepsAll) return $"keep every component resident (peak {Gb(Peak)}, budget {budget})";
            string releases = Releases.Count == 0 && Streams.Count == 0 ? "no releases" : string.Join("; ", Releases.Select(r =>
                $"before {r.Before}: release the {r.Component.ToString().ToLowerInvariant()}{(r.Weights ? "" : "'s workspace")}")
                .Concat(Streams.Select(st => $"{st.Phase}: stream the {st.Component.ToString().ToLowerInvariant()}'s weights " +
                                             $"{Gb(st.WeightBudget)} at a time")));
            string capacity = Refused ? $"; refused: {Limiting.Phase} needs {Gb(MinimumPeak)}, the device has {Gb(Capacity.Value)}" : "";
            return $"{releases} (peak {Gb(Peak)}, budget {budget}{(Fits ? "" : "; a phase alone exceeds the budget")}{capacity})";
        }

        private static string Gb(long bytes) => (bytes / 1e9).ToString("0.0", CultureInfo.InvariantCulture) + " GB";
    }

    /// <summary>
    /// Decides, per device, which components stay resident between phases and between requests
    /// (docs/design/diffusion-runtime.md section 3.4). When everything fits, everything stays (no regression on a device
    /// that holds it all). Otherwise, before each phase it makes room for the phase's component and working memory by
    /// releasing what the phase does not use: only workspaces when that makes the room (only compute buffers are lost),
    /// otherwise weights (with their workspace) of whatever is not needed again in this request, then of what is needed
    /// latest, preferring one release that makes the room on its own. Released weights stay mapped on the host, so a re-upload is a device copy. A phase
    /// whose own weights and working memory are past the budget with nothing else held streams its weights through the device
    /// where its component can (<see cref="PhaseDemand.StreamedWeightBytes"/>): it holds what the room leaves of them at once,
    /// and none of them after the phase (a 24 GB Mac's 17.2 GB working set holds neither MiniMax-H3-Image's 15.5 GB text
    /// encoder beside an edit's working memory, nor its 12.7 GB Q4_K_M denoiser beside a two-picture edit's step). The planner
    /// logs each distinct decision once.
    /// </summary>
    public sealed class ResidencyPlanner
    {
        /// <summary>
        /// How far a phase's estimated need may exceed the device's free memory before the request is refused. The
        /// working-memory estimates are calibrated to cover the measured peaks (MiniMaxH3ImageResidencyTests: 1.0-1.3x),
        /// and on the RTX A4500 every MiniMax-H3-Image request whose estimate stayed within the free memory ran (up to 0.22 GB
        /// under it) while every one past it failed inside the phase (from 0.29 GB over): no allowance is needed beyond the
        /// estimates' own (MiniMaxH3ImageResidencyTests).
        /// </summary>
        public const long RefusalAllowanceBytes = 0;

        private readonly Func<long?> _spareBytes;
        private readonly Func<long?> _freeBytes;
        private readonly Action<string> _log;
        private readonly HashSet<string> _logged = new(StringComparer.Ordinal);
        private long _capacitySeen;

        /// <param name="spareBytes">Free device bytes less the headroom reserve, or null when the backend has no device
        /// budget (GpuMemoryBudget.TryGetSpareBytes). Read once a request.</param>
        /// <param name="log">Where each new decision is written once (null: the console).</param>
        /// <param name="freeBytes">Free device bytes with no headroom (the most a phase can have), or null when unknown:
        /// a request whose most demanding phase alone needs more is refused (<see cref="ResidencyPlan.Refused"/>). Null
        /// (the default) refuses nothing.</param>
        public ResidencyPlanner(Func<long?> spareBytes, Action<string> log = null, Func<long?> freeBytes = null)
        {
            _spareBytes = spareBytes ?? throw new ArgumentNullException(nameof(spareBytes));
            _freeBytes = freeBytes;
            _log = log ?? Console.WriteLine;
        }

        /// <summary>The device bytes a request may use when the components hold <paramref name="heldBytes"/> now (the spare
        /// bytes plus those), or null when the backend reports no budget.</summary>
        public long? Budget(long heldBytes) => _spareBytes() is long spare ? spare + heldBytes : null;

        /// <summary>
        /// The plan for a request running <paramref name="phases"/> in order, with <paramref name="weights"/> the device
        /// bytes of each component's weights and <paramref name="resident"/> what the device holds now (weights, and
        /// the workspaces of <paramref name="residentWorkspaces"/> with their bytes). The budget is the spare bytes plus
        /// what the components hold now.
        /// </summary>
        public ResidencyPlan Plan(IReadOnlyList<PhaseDemand> phases, IReadOnlyDictionary<DiffusionComponent, long> weights,
            IReadOnlyCollection<DiffusionComponent> resident, IReadOnlyDictionary<DiffusionComponent, long> residentWorkspaces,
            string geometryClass)
        {
            long? spare = _spareBytes();
            long held = resident.Sum(c => weights.GetValueOrDefault(c)) + residentWorkspaces.Values.Sum();
            long? budget = spare is long s ? s + held : null;
            long? free = _freeBytes?.Invoke();
            long? capacity = free is long f ? f + held : null;
            // The backend keeps the memory pools a finished phase grew (ggml-cuda's do not shrink), so after a large request
            // the device reads less free than this pipeline can still use: the largest capacity this planner has seen bounds
            // the refusal. (A later allocation by another process is then not refused for: the run decides, as without it.)
            if (capacity is long c)
                lock (_logged)
                {
                    _capacitySeen = Math.Max(_capacitySeen, c);
                    capacity = _capacitySeen;
                }
            ResidencyPlan plan = Decide(phases, weights, resident, residentWorkspaces, budget, capacity);
            // Once per geometry class and decision (not per budget: free memory moves between requests).
            string decision = (plan.KeepsAll ? "all" : string.Join(";", plan.Releases) + "|" +
                                  string.Join(";", plan.Streams.Select(st => $"{st.Phase}:{st.Component}"))) + (plan.Refused ? ";refused" : "");
            lock (_logged)
                if (_logged.Add(geometryClass + "\n" + decision))
                    _log($"[diffusion-residency] {geometryClass}: {plan.Describe()}");
            return plan;
        }

        /// <summary>The decision itself, for a known <paramref name="budget"/> (null: keep everything) and the device's
        /// <paramref name="capacity"/> (free bytes with no headroom plus what the components hold; null: refuse nothing).</summary>
        public static ResidencyPlan Decide(IReadOnlyList<PhaseDemand> phases, IReadOnlyDictionary<DiffusionComponent, long> weights,
            IReadOnlyCollection<DiffusionComponent> resident, IReadOnlyDictionary<DiffusionComponent, long> residentWorkspaces,
            long? budget, long? capacity = null)
        {
            ArgumentNullException.ThrowIfNull(phases);
            ArgumentNullException.ThrowIfNull(weights);
            resident ??= Array.Empty<DiffusionComponent>();
            residentWorkspaces ??= new Dictionary<DiffusionComponent, long>();
            long W(DiffusionComponent c) => weights.GetValueOrDefault(c);
            // The weights a phase must hold at once: all of its component's, or the fewest it streams with.
            bool Streams(PhaseDemand p) => p.StreamedWeightBytes > 0 && p.StreamedWeightBytes < W(p.Component);
            long Least(PhaseDemand p) => Streams(p) ? p.StreamedWeightBytes : W(p.Component);
            // The phase no plan can make smaller: the least of its own component's weights it can hold and its working memory,
            // nothing else resident.
            PhaseDemand limiting = phases.Count == 0 ? default : phases.MaxBy(p => Least(p) + p.WorkingBytes);
            long limitingWeights = phases.Count == 0 ? 0 : Least(limiting);

            // Everything this request uses, plus what is resident anyway, with the largest working set beside it.
            var all = new HashSet<DiffusionComponent>(resident);
            foreach (PhaseDemand p in phases) all.Add(p.Component);
            long working = phases.Count == 0 ? 0 : phases.Max(p => p.WorkingBytes);
            var workspaces = new Dictionary<DiffusionComponent, long>(residentWorkspaces);
            foreach (PhaseDemand p in phases)
                workspaces[p.Component] = Math.Max(workspaces.GetValueOrDefault(p.Component), p.WorkingBytes);
            long everything = all.Sum(W) + workspaces.Values.Sum();
            if (budget is not long limit || everything <= limit)
                return new ResidencyPlan(budget, Math.Max(everything, all.Sum(W) + working), keepsAll: true, fits: true,
                    Array.Empty<ResidencyRelease>(), capacity, limiting, limitingWeights);

            var held = new HashSet<DiffusionComponent>(resident);
            var heldWork = new Dictionary<DiffusionComponent, long>(residentWorkspaces);
            var releases = new List<ResidencyRelease>();
            var streams = new List<ResidencyStream>();
            bool fits = true;
            long peak = 0;
            for (int i = 0; i < phases.Count; i++)
            {
                PhaseDemand phase = phases[i];
                long OthersWork() => heldWork.Where(kv => kv.Key != phase.Component).Sum(kv => kv.Value);
                long Need() => held.Union(new[] { phase.Component }).Sum(W) + OthersWork() + phase.WorkingBytes;
                int NextUse(DiffusionComponent c)
                {
                    for (int j = i + 1; j < phases.Count; j++) if (phases[j].Component == c) return j;
                    return int.MaxValue;
                }
                // Weights only when dropping every other workspace would not make the room: prefer one release that
                // makes it alone, among what is used latest (or never) again, and take its workspace with it.
                bool exhausted = false;
                while (Need() - OthersWork() > limit)
                {
                    DiffusionComponent[] candidates = held.Where(c => c != phase.Component).ToArray();
                    if (candidates.Length == 0) { exhausted = true; break; }
                    long excess = Need() - limit;
                    DiffusionComponent victim = candidates
                        .OrderByDescending(c => W(c) + heldWork.GetValueOrDefault(c) >= excess)
                        .ThenByDescending(NextUse)
                        .ThenByDescending(W)
                        .First();
                    held.Remove(victim);
                    heldWork.Remove(victim);
                    releases.Add(new ResidencyRelease(phase.Phase, victim, Weights: true));
                }
                // Then workspaces, the largest first: only their compute buffers are lost.
                foreach (DiffusionComponent c in heldWork.Keys.Where(c => c != phase.Component)
                             .OrderByDescending(c => heldWork[c]).ToArray())
                {
                    if (Need() <= limit) break;
                    heldWork.Remove(c);
                    releases.Add(new ResidencyRelease(phase.Phase, c, Weights: false));
                }
                // Its own weights and working memory still past the budget with nothing else held: a component that streams
                // holds what the room leaves of its weights at once (never fewer than it streams with), and holds none of
                // them, nor its workspace, after the phase. One that cannot runs as it is: the device decides.
                if (exhausted && Need() > limit && Streams(phase))
                {
                    long rest = Need() - W(phase.Component);
                    long room = Math.Max(phase.StreamedWeightBytes, limit - rest);
                    streams.Add(new ResidencyStream(phase.Phase, phase.Component, room));
                    if (rest + room > limit) fits = false;
                    peak = Math.Max(peak, rest + room);
                    held.Remove(phase.Component);
                    heldWork.Remove(phase.Component);
                    continue;
                }
                if (exhausted) fits = false;
                peak = Math.Max(peak, Need());
                held.Add(phase.Component);
                heldWork[phase.Component] = Math.Max(heldWork.GetValueOrDefault(phase.Component), phase.WorkingBytes);
            }
            return new ResidencyPlan(budget, peak, keepsAll: false, fits, releases, capacity, limiting, limitingWeights, streams);
        }
    }
}
