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
    /// <summary>How an architecture uses more than one GPU.</summary>
    public enum MultiGpuMode
    {
        /// <summary>Weights are sharded across ranks and the layers issue collectives
        /// (<c>--tp N</c> in its literal sense).</summary>
        TensorParallel,

        /// <summary>The shared loader passes the device count to the architecture's
        /// own executor without creating a tensor-parallel group. The default is
        /// whole-layer placement. Executors with an additional native sharding mode
        /// describe it through <see cref="ModelArchitectureDescriptor.DescribeMultiGpuPlacement"/>.</summary>
        LayerSplit,

        /// <summary>The architecture cannot use a second GPU through the shared
        /// machinery at all. <see cref="ModelArchitectureDescriptor.MultiGpuLimitation"/>
        /// says why, and is printed instead of silently idling the extra devices.
        /// (An architecture that drives several GPUs through its OWN executor - DeepSeek
        /// V4 sized by TS_DSV4_NGPU - is not this: it is TensorParallel as far as the
        /// shared gate is concerned, because the gate must not interfere.)</summary>
        SingleDevice,
    }

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

        /// <summary>How this architecture uses several GPUs. See <see cref="MultiGpuMode"/>.</summary>
        public MultiGpuMode MultiGpu { get; init; } = MultiGpuMode.TensorParallel;

        /// <summary>
        /// Why the shared tensor-parallel path is unavailable. REQUIRED for
        /// <see cref="MultiGpuMode.LayerSplit"/> and <see cref="MultiGpuMode.SingleDevice"/>,
        /// because that message is the whole value of the gate: it is what tells an
        /// operator why the extra GPU is idle, or why it holds layers rather than shards.
        /// </summary>
        public string MultiGpuLimitation { get; init; }

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

        /// <summary>Backends on which <see cref="MultiGpuMode.LayerSplit"/> can actually
        /// find several devices to split across.</summary>
        internal static bool BackendHasSeveralDevices(BackendType backend)
            => backend is BackendType.GgmlCuda or BackendType.GgmlVulkan;

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
            if (MultiGpu != MultiGpuMode.TensorParallel && string.IsNullOrWhiteSpace(MultiGpuLimitation))
            {
                throw new InvalidOperationException(
                    $"Architecture '{Id}' declares MultiGpu={MultiGpu} but no MultiGpuLimitation. That message is " +
                    "what tells an operator why the extra GPUs are idle or holding layers; an empty one is a bug.");
            }
        }
    }
}
