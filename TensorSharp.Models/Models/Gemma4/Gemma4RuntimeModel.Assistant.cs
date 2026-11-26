// Copyright (c) Zhongkai Fu. All rights reserved.
// https://github.com/zhongkaifu/TensorSharp
//
// This file is part of TensorSharp.
//
// TensorSharp is licensed under the BSD-3-Clause license found in the LICENSE file in the root directory of this source tree.
using System;
using System.IO;

using TensorSharp.GGML;
using TensorSharp.Graph;

namespace TensorSharp.Models
{
    /// <summary>
    /// The gemma4-assistant MTP draft head on the shared runtime (llama.cpp's gemma4-assistant; the legacy model's
    /// LoadMtpDraftWeights / DraftStep): a separate GGUF of a few Gemma blocks of its own width that keep no K/V. A step embeds
    /// the token with the trunk's table times sqrt(n_embd), concatenates the trunk's final-normed row (or the previous step's
    /// chained row), projects it (<c>pre_projection</c>), runs each block's queries against the trunk's caches at the trunk's
    /// length (the sliding-window blocks read the last local layer's cache, the global block the last global layer's, both
    /// through the KV-sharing map), and ends in <c>output_norm</c>, the head's own <c>token_embd</c> (no softcap) and
    /// <c>post_projection</c> back to the trunk's width for the next step. Every step queries the same position, so nothing is
    /// caught up and a draft resumes at any position.
    /// </summary>
    internal sealed partial class Gemma4RuntimeModel : IAttachableDraftHead
    {
        /// <summary>A draft head is attached (the gemma4-assistant head, or a DFlash drafter).</summary>
        public bool HasDraftHead => DraftHeadKind != TensorSharp.Runtime.Speculative.DraftHeadKind.None;

        /// <summary>Load a gemma4-assistant GGUF (either converter's spelling) and attach it as the per-token drafter.</summary>
        public void LoadMtpDraftWeights(string draftGgufPath)
        {
            if (string.IsNullOrEmpty(draftGgufPath) || !File.Exists(draftGgufPath))
                throw new FileNotFoundException("Gemma 4 MTP draft GGUF not found.", draftGgufPath);
            if (Mesh.TensorParallel > 1)
                throw new NotSupportedException($"{Config.Architecture}: the gemma4-assistant draft head reads the trunk's whole caches, " +
                    "which tensor parallelism shards by KV head; run without --tp to draft with it (n-gram speculation still runs).");
            using var draft = new GgufFile(draftGgufPath);
            string arch = draft.GetString("general.architecture", "");
            if (arch != "gemma4-assistant" && arch != "gemma4_assistant")
                throw new InvalidOperationException($"Expected a 'gemma4-assistant' GGUF for the Gemma 4 MTP draft head but got '{arch}'.");

            int layers = (int)draft.GetUint32($"{arch}.block_count");
            int width = (int)draft.GetUint32($"{arch}.embedding_length");
            int backbone = (int)draft.GetUint32($"{arch}.embedding_length_out", (uint)Config.HiddenSize);
            int heads = (int)draft.GetUint32($"{arch}.attention.head_count");
            int globalHd = (int)draft.GetUint32($"{arch}.attention.key_length", (uint)_globalHeadDim);
            int localHd = (int)draft.GetUint32($"{arch}.attention.key_length_swa", (uint)_localHeadDim);
            bool[] pattern = draft.GetBoolArray($"{arch}.attention.sliding_window_pattern");
            if (backbone != Config.HiddenSize)
                throw new InvalidOperationException($"MTP draft backbone dim {backbone} != target hidden size {Config.HiddenSize}.");
            if (globalHd != _globalHeadDim || localHd != _localHeadDim)
                throw new InvalidOperationException($"MTP draft head dims (local {localHd}, global {globalHd}) do not match the target " +
                    $"(local {_localHeadDim}, global {_globalHeadDim}).");
            if (pattern == null || pattern.Length != layers)
                throw new InvalidOperationException("MTP draft sliding-window pattern is missing or the wrong length.");

            // llama.cpp's share(): the sliding-window blocks read the trunk's last local layer, the global block its last global
            // layer, each through the KV-sharing map to the layer that holds the cache.
            int n = Config.NumLayers;
            if (!_isLocal[n - 2] || _isLocal[n - 1])
                throw new InvalidOperationException($"MTP donor layers don't have the expected types (layer {n - 2} should be " +
                    $"sliding-window, layer {n - 1} global).");
            int localDonor = _kvSource[n - 2], globalDonor = _kvSource[n - 1];
            if (StageOfLayerForDraft(localDonor) != StageOfLayerForDraft(globalDonor))
                throw new NotSupportedException($"{Config.Architecture}: the draft head's donor layers {localDonor} and {globalDonor} sit on " +
                    "different stages; load without --layer-split to draft with it.");
            foreach (int donor in new[] { localDonor, globalDonor })
                if (heads <= 0 || heads % _kvHeads[donor] != 0)
                    throw new InvalidOperationException($"MTP draft head count {heads} does not group onto donor layer {donor}'s {_kvHeads[donor]} KV heads.");

            LoadAssistantTensors(draft, AllocatorForWeight(LayerName(globalDonor, "attn_norm.weight")));
            var scales = new float[layers];
            for (int l = 0; l < layers; l++)
                scales[l] = _weights.TryGetValue($"mtp.blk.{l}.layer_output_scale.weight", out Tensor s) ? s.GetElementAsFloat(0) : 1f;

            bool complete = HasWeight("mtp.nextn.pre_projection.weight") && HasWeight("mtp.nextn.post_projection.weight")
                && HasWeight("mtp.token_embd.weight") && HasWeight("mtp.output_norm.weight");
            for (int l = 0; l < layers && complete; l++)
                foreach (string suffix in new[] { "attn_q.weight", "attn_output.weight", "ffn_gate.weight", "ffn_up.weight", "ffn_down.weight",
                    "attn_norm.weight", "attn_q_norm.weight", "post_attention_norm.weight", "ffn_norm.weight", "post_ffw_norm.weight" })
                    complete &= HasWeight($"mtp.blk.{l}.{suffix}");
            if (!complete)
            {
                Console.WriteLine("  Gemma 4 MTP draft GGUF loaded but incomplete; MTP drafting disabled.");
                return;
            }
            FamilyTokenDrafter?.Dispose();
            FamilyTokenDrafter = new AssistantDrafter(this, layers, heads, pattern, scales, localDonor, globalDonor);
            Console.WriteLine($"  Gemma 4 MTP draft head ready ({layers} layers, hidden {width}, draftHeads={heads}, " +
                $"donors local={localDonor}/global={globalDonor}; the shared runtime's draft program).");
        }

        private int StageOfLayerForDraft(int layer) => Mesh.StageOf(layer);

        /// <summary>The assistant's tensors under <c>mtp.</c> (both converters' names: a <c>mtp.</c> prefix or none, and the
        /// recurrent projections with or without <c>nextn.</c>); its <c>rope_freqs</c> is the trunk's, and the clustered head
        /// (<c>centroids</c> / <c>token_ordering</c>) is not used: the drafts score the whole <c>token_embd</c> head.</summary>
        private unsafe void LoadAssistantTensors(GgufFile draft, IAllocator allocator)
        {
            foreach (GgufTensorInfo info in draft.Tensors.Values)
            {
                string core = info.Name.StartsWith("mtp.", StringComparison.Ordinal) ? info.Name[4..] : info.Name;
                if (core is "rope_freqs.weight" or "centroids.weight" or "token_ordering.weight") continue;
                if (core == "pre_projection.weight") core = "nextn.pre_projection.weight";
                else if (core == "post_projection.weight") core = "nextn.post_projection.weight";
                string name = "mtp." + core;
                long byteCount = draft.GetTensorByteCount(info);
                if (IsQuantizedLinearWeight(info) && info.Type != GgmlTensorType.F16 && info.Type != GgmlTensorType.BF16)
                {
                    if (IsGgmlBackend) EnsureQuantBackendAvailable();
                    IntPtr ptr = QuantizedWeight.AllocateBuffer(byteCount);
                    draft.ReadTensorDataToNative(info, ptr, byteCount);
                    _quantWeights[name] = new QuantizedWeight(ptr, byteCount, (int)info.Type, (long)info.Shape[0], (long)info.Shape[1]);
                    continue;
                }
                long[] shape = new long[info.Shape.Length];
                for (int i = 0; i < shape.Length; i++) shape[i] = (long)info.Shape[shape.Length - 1 - i];
                var tensor = new Tensor(allocator, DType.Float32, shape);
                IntPtr dest = TensorComputePrimitives.GetStoragePointer(tensor);
                if (info.Type == GgmlTensorType.F32)
                    draft.ReadTensorDataToFloat32Native(info, dest, info.NumElements);
                else
                {
                    IntPtr temp = QuantizedWeight.AllocateBuffer(byteCount);
                    try
                    {
                        draft.ReadTensorDataToNative(info, temp, byteCount);
                        ManagedQuantizedOps.DequantizeToFloat32Native((int)info.Type, temp, dest, info.NumElements);
                    }
                    finally { QuantizedWeight.FreeBuffer(temp); }
                }
                if (_weights.Remove(name, out Tensor old)) old.Dispose();
                _weights[name] = tensor;
            }
        }

        /// <summary>The assistant's steps: one graph a step over the active cache, replayed by shape.</summary>
        private sealed class AssistantDrafter : FamilyTokenDrafter
        {
            private readonly Gemma4RuntimeModel _m;
            private readonly int _layers, _heads, _localDonor, _globalDonor;
            private readonly bool[] _local;
            private readonly float[] _scales;

            public AssistantDrafter(Gemma4RuntimeModel m, int layers, int heads, bool[] local, float[] scales, int localDonor, int globalDonor)
            {
                _m = m;
                _layers = layers;
                _heads = heads;
                _local = local;
                _scales = scales;
                _localDonor = localDonor;
                _globalDonor = globalDonor;
            }

            public override int FeatureWidth => _m.Config.HiddenSize;

            public override void Step(int token, float[] hPrev, float[] logitsOut, float[] hOut)
            {
                int hidden = _m.Config.HiddenSize, vocab = _m.Config.VocabSize;
                if (hPrev == null || hPrev.Length < hidden) throw new ArgumentException($"A draft step reads a row of {hidden} floats.", nameof(hPrev));
                // The trunk's length: its caches hold [0, length), and every step queries position length.
                int length = _m.ActiveCommittedLength;
                if (length < 1) throw new InvalidOperationException("The gemma4-assistant draft head drafts after at least one trunk row.");
                Tensor embedded = _m.Embedding(new[] { token });
                Tensor carried = _m.CreateFloatTensor(hPrev.AsSpan(0, hidden).ToArray(), 1, hidden);
                CachedGraph graph = null;
                bool owned = false;
                try
                {
                    // The step's row stands at the cache's last position (length - 1), so the mask and windows reach [0, length)
                    // as a decode row's do; its rotary position is length.
                    graph = _m.RunActiveDraftProgram(1, length - 1, _globalDonor, step => Emit(step, embedded, carried), new[] { embedded, carried },
                        out StepOutputs outputs, out owned, "gemma4-assistant");
                    if (graph == null)
                        throw new NotSupportedException(_m.SpeculationRefusal ?? $"{_m.Config.Architecture}: the draft head cannot run as a graph.");
                    if (logitsOut != null) Array.Copy(graph.Outputs[outputs.Logits], logitsOut, Math.Min(logitsOut.Length, vocab));
                    if (hOut != null) Array.Copy(graph.Outputs[outputs.Final], hOut, Math.Min(hOut.Length, hidden));
                }
                finally
                {
                    if (owned) graph?.Dispose();
                    embedded.Dispose();
                    carried.Dispose();
                }
            }

            private StepOutputs Emit(GraphStep step, Tensor embedded, Tensor carried)
            {
                int hidden = _m.Config.HiddenSize;
                var outputs = new StepOutputs { Logits = -1, Final = -1, Hidden = -1 };
                step.SetPositions(new[] { step.Starts[0] + 1 + step.Slots[0].RopeDelta });
                int x = step.Op(GraphOp.Scale, stackalloc[] { step.HiddenInput(embedded, hidden, 1) }, default, stackalloc[] { MathF.Sqrt(hidden) });
                int joined = step.Op(GraphOp.Concat, stackalloc[] { x, step.HiddenInput(carried, hidden, 1) }, stackalloc long[] { 0 });
                int cur = step.MatMul("mtp.nextn.pre_projection.weight", joined);                                   // [width, 1]
                for (int l = 0; l < _layers; l++) cur = EmitLayer(step, l, cur);
                int normed = step.RmsNorm(cur, "mtp.output_norm.weight");
                outputs.Logits = step.Output(step.MatMul("mtp.token_embd.weight", normed), _m.Config.VocabSize);
                outputs.Final = step.Output(step.MatMul("mtp.nextn.post_projection.weight", normed), hidden);
                return outputs;
            }

            /// <summary>One block: its queries (per-head RMSNorm, the trunk's rotary of the block's kind) against the donor's
            /// cache at scale 1, the sandwich-normed output and GeGLU feed-forward, the block's output scale.</summary>
            private int EmitLayer(GraphStep step, int l, int input)
            {
                string p = $"mtp.blk.{l}.";
                bool local = _local[l];
                int d = local ? _m._localHeadDim : _m._globalHeadDim;
                int normed = step.RmsNorm(input, p + "attn_norm.weight");
                int q = step.Reshape(step.MatMul(p + "attn_q.weight", normed), d, _heads, 1);
                q = step.Rope(step.RmsNorm(q, p + "attn_q_norm.weight"), local ? _m._localRope : _m._globalRope);
                int attn = _m.EmitCacheReadAttention(step, local ? _localDonor : _globalDonor, q, _heads, 1f);
                int h = step.Op(GraphOp.Add, step.RmsNorm(step.MatMul(p + "attn_output.weight", attn), p + "post_attention_norm.weight"), input);
                int ffnIn = step.RmsNorm(h, p + "ffn_norm.weight");
                int act = step.Op(GraphOp.GegluSplit, step.MatMul(p + "ffn_gate.weight", ffnIn), step.MatMul(p + "ffn_up.weight", ffnIn));
                h = step.Op(GraphOp.Add, h, step.RmsNorm(step.MatMul(p + "ffn_down.weight", act), p + "post_ffw_norm.weight"));
                return _scales[l] != 1f ? step.Op(GraphOp.Scale, stackalloc[] { h }, default, stackalloc[] { _scales[l] }) : h;
            }
        }
    }
}
