// Copyright (c) Zhongkai Fu. All rights reserved.
// https://github.com/zhongkaifu/TensorSharp
//
// This file is part of TensorSharp.
//
// TensorSharp is licensed under the BSD-3-Clause license found in the LICENSE file in the root directory of this source tree.
using System;
using System.Collections.Generic;
using System.Linq;

using TensorSharp.Runtime;

namespace TensorSharp.Models.Architecture
{
    /// <summary>The five serving mechanisms every model family must account for.</summary>
    public enum ServingMechanism
    {
        /// <summary>Concurrent requests share batched forwards (paged batches or native
        /// per-sequence slots), rather than taking turns through a KV ownership swap.</summary>
        ContinuousBatching,

        /// <summary><c>--tp N</c>: every layer is split across N devices that reduce.</summary>
        TensorParallel,

        /// <summary><c>--layer-split N</c>: contiguous whole layers per device (pipeline stages).</summary>
        LayerSplit,

        /// <summary><c>--tp T --layer-split P</c> together: P stages, each a T-device TP group.</summary>
        HybridParallel,

        /// <summary><c>--n-cpu-moe N</c> / <c>--cpu-moe</c>: routed experts stay in host memory and run there.</summary>
        ExpertOffload,
    }

    /// <summary>Who implements a family's serving mechanisms.</summary>
    public enum ServingImplementation
    {
        /// <summary>The family supplies a layer recipe; the shared decoder runtime
        /// implements every mechanism (see docs/design/unified-execution.md). Every new
        /// family must be built this way.</summary>
        SharedRuntime,

        /// <summary>The family implements its mechanisms itself (per-family forwards or a
        /// native whole-model executor). Allowed only for the families that predate the
        /// shared runtime; the conformance tests hold that list closed.</summary>
        Bespoke,
    }

    public enum SupportKind
    {
        /// <summary>Implemented on <see cref="MechanismSupport.Backends"/>.</summary>
        Supported,

        /// <summary>The mechanism has no meaning for this family (a dense model has no
        /// experts to offload; a diffusion pipeline has no decode batch).</summary>
        NotApplicable,
    }

    /// <summary>
    /// How one family supports one mechanism. Immutable; built with <see cref="On"/>,
    /// <see cref="NotApplicable"/> or <see cref="None"/>.
    ///
    /// A backend the family runs on but that is not in <see cref="Backends"/> is a GAP,
    /// and <see cref="GapReason"/> is what an operator is told when they ask for it there.
    /// Gaps are tracked by the conformance tests' ratchet: they can be closed, never added.
    /// </summary>
    public sealed class MechanismSupport
    {
        private static readonly BackendType[] NoBackends = Array.Empty<BackendType>();

        private MechanismSupport(SupportKind kind, IReadOnlyList<BackendType> backends, string reason,
            string gapReason, Func<int, BackendType, string> precondition, BackendType? loopbackBackend,
            Func<bool> loopbackEnabled, IReadOnlyList<BackendType> batchedBackends = null)
        {
            Kind = kind;
            Backends = backends;
            Reason = reason;
            GapReason = gapReason;
            Precondition = precondition;
            LoopbackBackend = loopbackBackend;
            LoopbackEnabled = loopbackEnabled;
            BatchedBackends = batchedBackends ?? NoBackends;
        }

        public SupportKind Kind { get; }

        /// <summary>Backends where the mechanism is implemented (empty for NotApplicable).</summary>
        public IReadOnlyList<BackendType> Backends { get; }

        /// <summary>Why the mechanism does not apply (NotApplicable only).</summary>
        public string Reason { get; }

        /// <summary>Why the family's other backends lack it. Required whenever the family runs
        /// on a backend outside <see cref="Backends"/>.</summary>
        public string GapReason { get; }

        /// <summary>Optional shape/degree check run after the backend check: returns a refusal
        /// message, or null when the request can be honoured (e.g. "2 KV heads cap --tp at 2").
        /// Receives the number of devices this process drives for the mechanism (the local
        /// tensor-parallel degree, or the stage count) and the backend.</summary>
        public Func<int, BackendType, string> Precondition { get; }

        /// <summary>A single-device backend on which the family can still run its multi-rank
        /// code over virtual ranks, for fixture tests only (GLM's ggml_cpu oversubscription).
        /// Never evidence of multi-device execution.</summary>
        public BackendType? LoopbackBackend { get; }

        /// <summary>Whether the loopback mode is switched on for this process.</summary>
        public Func<bool> LoopbackEnabled { get; }

        /// <summary>
        /// For a placement mechanism (TP, layer split, hybrid): the backends on which concurrent
        /// requests are STILL batched while it is active. A backend in <see cref="Backends"/> but
        /// not here serves concurrency through the serial per-sequence ownership swap under this
        /// mechanism, which the conformance ratchet tracks as a gap of its own.
        /// </summary>
        public IReadOnlyList<BackendType> BatchedBackends { get; }

        /// <summary>Supported on exactly these backends.</summary>
        public static MechanismSupport On(params BackendType[] backends)
        {
            if (backends == null || backends.Length == 0)
                throw new ArgumentException("A supported mechanism must name at least one backend; use None for a gap everywhere.", nameof(backends));
            return new MechanismSupport(SupportKind.Supported, backends.Distinct().ToArray(), null, null, null, null, null);
        }

        /// <summary>Not implemented on any backend; <paramref name="gapReason"/> is the refusal.</summary>
        public static MechanismSupport None(string gapReason)
        {
            if (string.IsNullOrWhiteSpace(gapReason))
                throw new ArgumentException("A gap must say why.", nameof(gapReason));
            return new MechanismSupport(SupportKind.Supported, NoBackends, null, gapReason, null, null, null);
        }

        /// <summary>The mechanism has no meaning for this family.</summary>
        public static MechanismSupport NotApplicable(string reason)
        {
            if (string.IsNullOrWhiteSpace(reason))
                throw new ArgumentException("NotApplicable must say why.", nameof(reason));
            return new MechanismSupport(SupportKind.NotApplicable, NoBackends, reason, null, null, null, null);
        }

        /// <summary>The refusal for the family's other backends.</summary>
        public MechanismSupport ElsewhereGap(string gapReason)
        {
            if (string.IsNullOrWhiteSpace(gapReason))
                throw new ArgumentException("A gap must say why.", nameof(gapReason));
            return new MechanismSupport(Kind, Backends, Reason, gapReason, Precondition, LoopbackBackend, LoopbackEnabled, BatchedBackends);
        }

        /// <summary>Adds a degree/shape check (see <see cref="Precondition"/>).</summary>
        public MechanismSupport When(Func<int, BackendType, string> precondition)
            => new(Kind, Backends, Reason, GapReason, precondition ?? throw new ArgumentNullException(nameof(precondition)),
                LoopbackBackend, LoopbackEnabled, BatchedBackends);

        /// <summary>Adds the fixture-only loopback mode (see <see cref="LoopbackBackend"/>).</summary>
        public MechanismSupport WithLoopback(BackendType backend, Func<bool> enabled)
            => new(Kind, Backends, Reason, GapReason, Precondition, backend, enabled ?? throw new ArgumentNullException(nameof(enabled)),
                BatchedBackends);

        /// <summary>Concurrent requests stay batched under this mechanism on these backends
        /// (see <see cref="BatchedBackends"/>).</summary>
        public MechanismSupport KeepsBatchingOn(params BackendType[] backends)
            => new(Kind, Backends, Reason, GapReason, Precondition, LoopbackBackend, LoopbackEnabled,
                (backends ?? NoBackends).Distinct().ToArray());

        /// <summary>Concurrent requests stay batched wherever the mechanism is supported.</summary>
        public MechanismSupport KeepsBatching() => KeepsBatchingOn(Backends.ToArray());

        public bool IsSupportedOn(BackendType backend)
            => Kind == SupportKind.Supported && (Backends.Contains(backend) || IsLoopback(backend));

        /// <summary>True when <paramref name="backend"/> is this mechanism's enabled loopback backend.</summary>
        public bool IsLoopback(BackendType backend)
            => LoopbackBackend == backend && LoopbackEnabled?.Invoke() == true;

        /// <summary>Backends among <paramref name="universe"/> that lack the mechanism.</summary>
        public IEnumerable<BackendType> GapsWithin(IEnumerable<BackendType> universe)
            => Kind == SupportKind.NotApplicable ? Enumerable.Empty<BackendType>() : universe.Where(b => !Backends.Contains(b));

        public override string ToString() => Kind == SupportKind.NotApplicable
            ? "n/a (" + Reason + ")"
            : Backends.Count == 0 ? "gap (" + GapReason + ")" : string.Join(", ", Backends.Select(BackendNames.OptionName));
    }

    /// <summary>
    /// The mandatory declaration of how a model family serves requests: one entry per
    /// <see cref="ServingMechanism"/>. It is the only thing the loader consults to accept or
    /// refuse <c>--tp</c>, <c>--layer-split</c>, their combination and expert offload, so a
    /// declared gap is also the operator-facing refusal. See docs/design/unified-execution.md.
    /// </summary>
    public sealed class ServingContract
    {
        /// <summary>Every backend TensorSharp ships.</summary>
        public static readonly IReadOnlyList<BackendType> AllBackends = Enum.GetValues<BackendType>();

        /// <summary>Backends that can expose more than one device to a process. TP, layer split
        /// and their hybrid are only meaningful there; on the others a request for them is
        /// refused by the topology planner for every family alike, not counted as a family gap.</summary>
        public static readonly IReadOnlyList<BackendType> MultiDeviceBackends =
            new[] { BackendType.Cuda, BackendType.GgmlCuda, BackendType.GgmlVulkan };

        /// <summary>Backends whose weights already live in host memory and run on the CPU: an
        /// expert-offload request is trivially honoured there, for every family.</summary>
        public static readonly IReadOnlyList<BackendType> HostBackends =
            new[] { BackendType.Cpu, BackendType.GgmlCpu };

        public required ServingImplementation Implementation { get; init; }

        /// <summary>Backends the family loads on at all. Defaults to every backend.</summary>
        public IReadOnlyList<BackendType> Backends { get; init; } = AllBackends;

        /// <summary>Whether the family decodes autoregressively (text LMs). Media and diffusion
        /// families set false; batching of chat requests is then NotApplicable.</summary>
        public bool Autoregressive { get; init; } = true;

        /// <summary>Whether the family has routed experts. Dense families set false and declare
        /// expert offload NotApplicable.</summary>
        public required bool HasRoutedExperts { get; init; }

        public required MechanismSupport ContinuousBatching { get; init; }
        public required MechanismSupport TensorParallel { get; init; }
        public required MechanismSupport LayerSplit { get; init; }
        public required MechanismSupport HybridParallel { get; init; }
        public required MechanismSupport ExpertOffload { get; init; }

        public MechanismSupport this[ServingMechanism mechanism] => mechanism switch
        {
            ServingMechanism.ContinuousBatching => ContinuousBatching,
            ServingMechanism.TensorParallel => TensorParallel,
            ServingMechanism.LayerSplit => LayerSplit,
            ServingMechanism.HybridParallel => HybridParallel,
            ServingMechanism.ExpertOffload => ExpertOffload,
            _ => throw new ArgumentOutOfRangeException(nameof(mechanism)),
        };

        /// <summary>The backends on which a missing <paramref name="mechanism"/> counts as a gap:
        /// the family's backends, narrowed to the multi-device ones for TP / layer split / hybrid.</summary>
        public IEnumerable<BackendType> GapUniverse(ServingMechanism mechanism)
            => mechanism switch
            {
                ServingMechanism.TensorParallel or ServingMechanism.LayerSplit or ServingMechanism.HybridParallel
                    => Backends.Intersect(MultiDeviceBackends),
                ServingMechanism.ExpertOffload => Backends.Except(HostBackends),
                _ => Backends,
            };

        /// <summary>Whether <paramref name="mechanism"/> is a placement mode under which batching
        /// of concurrent requests has to survive.</summary>
        public static bool IsPlacement(ServingMechanism mechanism)
            => mechanism is ServingMechanism.TensorParallel or ServingMechanism.LayerSplit or ServingMechanism.HybridParallel;

        /// <summary>
        /// Every gap this family declares: a backend it runs on that lacks a mechanism
        /// (<c>BatchingOnly = false</c>), or a backend where a placement mechanism works but
        /// concurrent requests stop being batched under it (<c>BatchingOnly = true</c>).
        /// </summary>
        public IEnumerable<ServingGap> Gaps()
        {
            foreach (ServingMechanism mechanism in Enum.GetValues<ServingMechanism>())
            {
                MechanismSupport support = this[mechanism];
                foreach (BackendType backend in support.GapsWithin(GapUniverse(mechanism)))
                    yield return new ServingGap(mechanism, backend, BatchingOnly: false);
                if (!IsPlacement(mechanism) || support.Kind != SupportKind.Supported)
                    continue;
                foreach (BackendType backend in support.Backends)
                {
                    if (ContinuousBatching.IsSupportedOn(backend) && !support.BatchedBackends.Contains(backend))
                        yield return new ServingGap(mechanism, backend, BatchingOnly: true);
                }
            }
        }

        /// <summary>
        /// The contract of a family built on the shared decoder runtime: the runtime
        /// implements every mechanism on every backend the family loads on (TP and layer
        /// split wherever the backend provides the devices). Not overridable downwards.
        /// </summary>
        public static ServingContract SharedRuntime(bool hasRoutedExperts, IReadOnlyList<BackendType> backends = null)
        {
            backends ??= AllBackends;
            var all = backends.ToArray();
            var multi = all.Intersect(MultiDeviceBackends).ToArray();
            MechanismSupport multiDevice = multi.Length > 0
                ? MechanismSupport.On(multi)
                : MechanismSupport.NotApplicable("the family runs only on single-device backends");
            return new ServingContract
            {
                Implementation = ServingImplementation.SharedRuntime,
                Backends = all,
                HasRoutedExperts = hasRoutedExperts,
                ContinuousBatching = MechanismSupport.On(all),
                TensorParallel = multi.Length > 0 ? multiDevice.KeepsBatching() : multiDevice,
                LayerSplit = multi.Length > 0 ? multiDevice.KeepsBatching() : multiDevice,
                HybridParallel = multi.Length > 0 ? multiDevice.KeepsBatching() : multiDevice,
                ExpertOffload = hasRoutedExperts
                    ? MechanismSupport.On(all)
                    : MechanismSupport.NotApplicable("dense: no routed experts"),
            };
        }

        /// <summary>Structural checks run at registration. Throws with the family id.</summary>
        internal void Validate(string id)
        {
            if (Backends == null || Backends.Count == 0)
                throw new InvalidOperationException($"Architecture '{id}' declares no backends in its serving contract.");
            foreach (ServingMechanism mechanism in Enum.GetValues<ServingMechanism>())
            {
                MechanismSupport support = this[mechanism]
                    ?? throw new InvalidOperationException($"Architecture '{id}' does not declare {mechanism} in its serving contract.");
                if (support.Kind == SupportKind.Supported)
                {
                    BackendType[] foreign = support.Backends.Except(Backends).ToArray();
                    if (foreign.Length > 0)
                        throw new InvalidOperationException(
                            $"Architecture '{id}' claims {mechanism} on {string.Join(", ", foreign.Select(BackendNames.OptionName))}, " +
                            "where the family does not load at all.");
                    if (support.GapsWithin(GapUniverse(mechanism)).Any() && string.IsNullOrWhiteSpace(support.GapReason))
                        throw new InvalidOperationException(
                            $"Architecture '{id}' lacks {mechanism} on some of its backends but does not say why " +
                            "(MechanismSupport.ElsewhereGap); that reason is the operator's refusal message.");
                }
            }

            if (Autoregressive && ContinuousBatching.Kind == SupportKind.NotApplicable)
                throw new InvalidOperationException(
                    $"Architecture '{id}' decodes autoregressively, so continuous batching cannot be NotApplicable; declare where it is supported and the gap elsewhere.");
            if (!HasRoutedExperts && ExpertOffload.Kind != SupportKind.NotApplicable)
                throw new InvalidOperationException($"Architecture '{id}' has no routed experts but declares expert offload.");
            if (HasRoutedExperts && ExpertOffload.Kind == SupportKind.NotApplicable)
                throw new InvalidOperationException(
                    $"Architecture '{id}' has routed experts, so expert offload cannot be NotApplicable; declare where it is supported and the gap elsewhere.");
            if (HybridParallel.Kind == SupportKind.Supported)
            {
                foreach (BackendType backend in HybridParallel.Backends)
                {
                    if (!TensorParallel.IsSupportedOn(backend) || !LayerSplit.IsSupportedOn(backend))
                        throw new InvalidOperationException(
                            $"Architecture '{id}' claims hybrid TP x layer split on {BackendNames.OptionName(backend)} without supporting both halves there.");
                }
            }

            if (Implementation == ServingImplementation.SharedRuntime)
            {
                ServingContract expected = SharedRuntime(HasRoutedExperts, Backends);
                foreach (ServingMechanism mechanism in Enum.GetValues<ServingMechanism>())
                {
                    if (this[mechanism].Kind != expected[mechanism].Kind ||
                        !this[mechanism].Backends.OrderBy(b => b).SequenceEqual(expected[mechanism].Backends.OrderBy(b => b)) ||
                        !this[mechanism].BatchedBackends.OrderBy(b => b).SequenceEqual(expected[mechanism].BatchedBackends.OrderBy(b => b)))
                        throw new InvalidOperationException(
                            $"Architecture '{id}' is built on the shared runtime, which provides {mechanism} on " +
                            $"{expected[mechanism]}; a shared-runtime family cannot declare less ({this[mechanism]}). " +
                            "Use ServingContract.SharedRuntime(...).");
                }
            }
        }

        /// <summary>One line per mechanism, for startup logs and documentation.</summary>
        public string Describe()
        {
            var lines = new List<string> { "implementation: " + Implementation };
            foreach (ServingMechanism mechanism in Enum.GetValues<ServingMechanism>())
                lines.Add($"{mechanism}: {this[mechanism]}");
            return string.Join("\n", lines);
        }
    }

    /// <summary>One declared gap; <see cref="ToString"/> is the stable key the conformance ratchet lists.</summary>
    public readonly record struct ServingGap(ServingMechanism Mechanism, BackendType Backend, bool BatchingOnly)
    {
        public override string ToString()
            => $"{Mechanism}{(BatchingOnly ? "+batching" : "")}@{BackendNames.OptionName(Backend)}";
    }

    /// <summary>The operator-facing spellings of backends (what <c>--backend</c> accepts).</summary>
    public static class BackendNames
    {
        public static string OptionName(BackendType backend) => backend switch
        {
            BackendType.Cpu => "cpu",
            BackendType.Cuda => "cuda",
            BackendType.Mlx => "mlx",
            BackendType.GgmlCpu => "ggml_cpu",
            BackendType.GgmlMetal => "ggml_metal",
            BackendType.GgmlCuda => "ggml_cuda",
            BackendType.GgmlVulkan => "ggml_vulkan",
            _ => backend.ToString(),
        };

        public static string Describe(IEnumerable<BackendType> backends)
            => string.Join(", ", backends.Select(b => "--backend " + OptionName(b)));
    }
}
