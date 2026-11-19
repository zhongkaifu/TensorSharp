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
    public enum DiffusionComponent { Conditioner, Vae, Denoiser }

    /// <summary>What one phase needs on the device: its component's weights and its working memory.</summary>
    public readonly record struct PhaseDemand(DiffusionPhase Phase, DiffusionComponent Component, long WorkingBytes);

    /// <summary>What a pipeline drops before a phase: a component's weights and workspace (<see cref="Weights"/>), or
    /// only its workspace (activations, compute buffers).</summary>
    public readonly record struct ResidencyRelease(DiffusionPhase Before, DiffusionComponent Component, bool Weights);

    /// <summary>
    /// A request's residency decision: the releases to make before each phase, or none (<see cref="KeepsAll"/>, the
    /// behaviour of a device that holds every component). <see cref="Fits"/> is false when some phase alone needs more
    /// than the budget; the plan then keeps as little resident as it can and the run goes on (the device's own limits
    /// decide).
    /// </summary>
    public sealed class ResidencyPlan
    {
        internal ResidencyPlan(long? budget, long peak, bool keepsAll, bool fits, IReadOnlyList<ResidencyRelease> releases)
        {
            Budget = budget;
            Peak = peak;
            KeepsAll = keepsAll;
            Fits = fits;
            Releases = releases;
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

        /// <summary>The releases to make before <paramref name="phase"/>, in order.</summary>
        public IEnumerable<ResidencyRelease> Before(DiffusionPhase phase) => Releases.Where(r => r.Before == phase);

        /// <summary>One line for the log and telemetry.</summary>
        public string Describe()
        {
            string budget = Budget is long b ? Gb(b) : "unknown";
            if (KeepsAll) return $"keep every component resident (peak {Gb(Peak)}, budget {budget})";
            string releases = Releases.Count == 0 ? "no releases" : string.Join("; ", Releases.Select(r =>
                $"before {r.Before}: release the {r.Component.ToString().ToLowerInvariant()}{(r.Weights ? "" : "'s workspace")}"));
            return $"{releases} (peak {Gb(Peak)}, budget {budget}{(Fits ? "" : "; a phase alone exceeds the budget")})";
        }

        private static string Gb(long bytes) => (bytes / 1e9).ToString("0.0", CultureInfo.InvariantCulture) + " GB";
    }

    /// <summary>
    /// Decides, per device, which components stay resident between phases and between requests
    /// (docs/design/diffusion-runtime.md section 3.4). When everything fits, everything stays (no regression on a device
    /// that holds it all). Otherwise, before each phase it makes room for the phase's component and working memory by
    /// releasing what the phase does not use: only workspaces when that makes the room (only compute buffers are lost),
    /// otherwise weights (with their workspace) of whatever is not needed again in this request, then of what is needed
    /// latest, preferring one release that makes the room on its own. Released weights stay mapped on the host, so a re-upload is a device copy. The planner logs each
    /// distinct decision once.
    /// </summary>
    public sealed class ResidencyPlanner
    {
        private readonly Func<long?> _spareBytes;
        private readonly Action<string> _log;
        private readonly HashSet<string> _logged = new(StringComparer.Ordinal);

        /// <param name="spareBytes">Free device bytes less the headroom reserve, or null when the backend has no device
        /// budget (GpuMemoryBudget.TryGetSpareBytes). Read once a request.</param>
        /// <param name="log">Where each new decision is written once (null: the console).</param>
        public ResidencyPlanner(Func<long?> spareBytes, Action<string> log = null)
        {
            _spareBytes = spareBytes ?? throw new ArgumentNullException(nameof(spareBytes));
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
            ResidencyPlan plan = Decide(phases, weights, resident, residentWorkspaces, budget);
            // Once per geometry class and decision (not per budget: free memory moves between requests).
            string decision = plan.KeepsAll ? "all" : string.Join(";", plan.Releases);
            lock (_logged)
                if (_logged.Add(geometryClass + "\n" + decision))
                    _log($"[diffusion-residency] {geometryClass}: {plan.Describe()}");
            return plan;
        }

        /// <summary>The decision itself, for a known <paramref name="budget"/> (null: keep everything).</summary>
        public static ResidencyPlan Decide(IReadOnlyList<PhaseDemand> phases, IReadOnlyDictionary<DiffusionComponent, long> weights,
            IReadOnlyCollection<DiffusionComponent> resident, IReadOnlyDictionary<DiffusionComponent, long> residentWorkspaces,
            long? budget)
        {
            ArgumentNullException.ThrowIfNull(phases);
            ArgumentNullException.ThrowIfNull(weights);
            resident ??= Array.Empty<DiffusionComponent>();
            residentWorkspaces ??= new Dictionary<DiffusionComponent, long>();
            long W(DiffusionComponent c) => weights.GetValueOrDefault(c);

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
                    Array.Empty<ResidencyRelease>());

            var held = new HashSet<DiffusionComponent>(resident);
            var heldWork = new Dictionary<DiffusionComponent, long>(residentWorkspaces);
            var releases = new List<ResidencyRelease>();
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
                while (Need() - OthersWork() > limit)
                {
                    DiffusionComponent[] candidates = held.Where(c => c != phase.Component).ToArray();
                    if (candidates.Length == 0) { fits = false; break; }
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
                peak = Math.Max(peak, Need());
                held.Add(phase.Component);
                heldWork[phase.Component] = Math.Max(heldWork.GetValueOrDefault(phase.Component), phase.WorkingBytes);
            }
            return new ResidencyPlan(budget, peak, keepsAll: false, fits, releases);
        }
    }
}
