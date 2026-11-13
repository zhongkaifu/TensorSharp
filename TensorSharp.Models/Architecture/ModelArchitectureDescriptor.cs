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
    /// <summary>
    /// One architecture plug-in: everything the shared loader needs to know about a
    /// model family, declared by the family itself.
    ///
    /// This is the seam that keeps <see cref="ModelBase.Create"/> free of per-model
    /// knowledge. Adding an architecture means adding its <c>Models/&lt;Name&gt;/</c>
    /// directory, a static <c>Descriptor</c> on its model class, and one line in
    /// <see cref="BuiltInArchitectures"/> - no switch, name list, or capability table
    /// anywhere else has to be edited.
    ///
    /// Descriptors are consulted at LOAD time only. Nothing here is on a per-token
    /// path, so the indirection costs nothing at inference time; runtime routing keeps
    /// using the capability interfaces (<c>IBatchedPagedModel</c>,
    /// <c>ISpeculativeTarget</c>, ...) that <c>ExecutionCapabilities</c> already reads.
    /// </summary>
    public sealed class ModelArchitectureDescriptor
    {
        /// <summary>Canonical architecture id, used in logs and diagnostics.</summary>
        public required string Id { get; init; }

        /// <summary>Every <c>general.architecture</c> string that selects this plug-in,
        /// matched case-insensitively. Must contain <see cref="Id"/>.</summary>
        public required IReadOnlyList<string> Aliases { get; init; }

        /// <summary>Builds the model. Called once, after the multi-GPU gate has resolved
        /// <see cref="ModelCreateContext.TpDegree"/> / <see cref="ModelCreateContext.LayerSplitDegree"/>.</summary>
        public required Func<ModelCreateContext, ModelBase> Factory { get; init; }

        /// <summary>Human-readable family name for banners; defaults to <see cref="Id"/>.</summary>
        public string DisplayName { get; init; }

        /// <summary>
        /// Recognises a GGUF that declares NO architecture metadata at all. Only a
        /// handful of published files are like this (MiniMax-H3); without a detector,
        /// resolution fails because the loader does not guess an architecture. Null
        /// for the normal case.
        /// </summary>
        public Func<GgufFile, bool> DetectFromTensors { get; init; }

        /// <summary>
        /// Recognises a GGUF of this family that a converter labelled with a GENERIC
        /// architecture instead of the family's own id - llama.cpp's older
        /// <c>convert_hf_to_gguf.py</c> wrote Mistral Small 3.1 as
        /// <c>general.architecture = llama</c>, for example. Receives the declared
        /// architecture and the probe; must return true only when the file's metadata and
        /// tensor layout are exactly what this family's model implements.
        ///
        /// A generic label is deliberately NOT an alias: "llama" also names Llama 3 (with
        /// rope_freqs scaling), Mixtral (experts) and others that a Mistral graph would run
        /// into fluent garbage. Consulted only when no alias matched. Null for the normal case.
        /// </summary>
        public Func<string, GgufFile, bool> RecognizeRelabelledFile { get; init; }

        /// <summary>What <see cref="RecognizeRelabelledFile"/> accepts, in one phrase, for the
        /// refusal message when it accepts nothing. Required when the recogniser is set.</summary>
        public string RelabelledFileDescription { get; init; }

        /// <summary>
        /// The mandatory declaration of how this family serves requests: continuous batching,
        /// tensor parallelism, layer split, their hybrid and expert offload, per backend. The
        /// loader accepts or refuses <c>--tp</c> / <c>--layer-split</c> from this alone, and the
        /// conformance tests prove it and hold its gaps to a shrinking list. See
        /// docs/design/unified-execution.md.
        /// </summary>
        public required ServingContract Serving { get; init; }

        /// <summary>The executor loads and runs only its node's whole layers and
        /// exchanges every required stage boundary tensor through ILayerSplitGroup.</summary>
        public bool SupportsDistributedLayerSplit { get; init; }

        public IReadOnlyList<BackendType> DistributedLayerSplitBackends { get; init; } =
            new[] { BackendType.GgmlCuda, BackendType.GgmlVulkan };

        /// <summary>Whether the tensor-parallel executor consumes cross-node
        /// collectives. Only applies to TensorParallel architectures. Native
        /// executors must connect their rank reductions to the supplied group;
        /// owning several local ranks alone does not provide distributed support.</summary>
        public bool SupportsDistributedTensorParallel { get; init; } = true;

        /// <summary>Optional backend restriction for cross-node collectives.
        /// Checked before the host creates GPU ranks or connects to peers;
        /// null permits the shared distributed GPU backends.</summary>
        public IReadOnlyList<BackendType> DistributedTensorParallelBackends { get; init; }

        /// <summary>Optional startup description for an executor that owns device
        /// placement and reductions itself. Receives the requested GPU count; may
        /// describe or validate native environment overrides. The shared loader still
        /// creates no tensor-parallel group. Null uses the standard layer-split message.</summary>
        public Func<int, string> DescribeMultiGpuPlacement { get; init; }

        /// <summary>
        /// Process-wide native tuning this architecture needs applied BEFORE its
        /// weights load (ggml environment switches and the like). Runs once per model
        /// load, with the probe GGUF available for shape-dependent decisions. Null for
        /// architectures that need none - which is nearly all of them.
        /// </summary>
        public Action<ModelCreateContext> ApplyNativeTunables { get; init; }

        /// <summary>
        /// File-name patterns, tried in order, for finding this family's mmproj
        /// companion beside the model GGUF when the operator did not pass one
        /// explicitly. Plain names are probed directly; patterns containing '*' are
        /// matched against the directory listing (first match wins, ordered by name so
        /// the choice is stable). Empty for text-only families.
        ///
        /// This lives here so the CLI and server can auto-load a projector without a
        /// chain of architecture-name comparisons; the decision to look at all comes
        /// from the model implementing <see cref="IVisionCapableModel"/> /
        /// <see cref="IAudioCapableModel"/>, not from its name.
        /// </summary>
        public IReadOnlyList<string> ProjectorFileHints { get; init; } = Array.Empty<string>();

        /// <summary>
        /// The <c>"type"</c> of the LoRA plug-in configs this family's model applies (one of
        /// <see cref="LoraCliFlags.PluginTypes"/>), or null when it takes none. The hosts read
        /// it from the probe GGUF to refuse a <c>--lora</c> the model could not apply before any
        /// plug-in file is resolved or downloaded.
        /// </summary>
        public string LoraPluginType { get; init; }

        internal string Name => DisplayName ?? Id;

        internal void Validate()
        {
            if (string.IsNullOrWhiteSpace(Id))
                throw new InvalidOperationException("Architecture descriptor has no Id.");
            if (Aliases == null || Aliases.Count == 0)
                throw new InvalidOperationException($"Architecture '{Id}' declares no aliases.");
            if (!Aliases.Contains(Id, StringComparer.OrdinalIgnoreCase))
                throw new InvalidOperationException($"Architecture '{Id}' must list its own id among its aliases.");
            if (Factory == null)
                throw new InvalidOperationException($"Architecture '{Id}' has no factory.");
            if (RecognizeRelabelledFile != null && string.IsNullOrWhiteSpace(RelabelledFileDescription))
            {
                throw new InvalidOperationException(
                    $"Architecture '{Id}' recognises relabelled files but does not say which; the refusal " +
                    "message for a file it rejects would not tell the operator what is accepted.");
            }
            if (Serving == null)
                throw new InvalidOperationException($"Architecture '{Id}' declares no serving contract.");
            Serving.Validate(Id);
            if (SupportsDistributedLayerSplit && !DistributedLayerSplitBackends.All(Serving.LayerSplit.IsSupportedOn))
                throw new InvalidOperationException(
                    $"Architecture '{Id}' runs distributed layer split on a backend where it does not declare local layer split.");
            if (LoraPluginType != null && !LoraCliFlags.PluginTypes.Contains(LoraPluginType, StringComparer.Ordinal))
            {
                throw new InvalidOperationException(
                    $"Architecture '{Id}' takes LoRA plug-ins of type '{LoraPluginType}', which is not one of " +
                    $"{string.Join(", ", LoraCliFlags.PluginTypes)}.");
            }
        }
    }
}
