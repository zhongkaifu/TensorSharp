// Copyright (c) Zhongkai Fu. All rights reserved.
// https://github.com/zhongkaifu/TensorSharp
//
// This file is part of TensorSharp.
//
// TensorSharp is licensed under the BSD-3-Clause license found in the LICENSE file in the root directory of this source tree.
using System;
using TensorSharp.Cuda;

namespace TensorSharp.Models
{
    /// <summary>
    /// Qwen3.5 / Qwen3.8 dense on the direct-CUDA engine (<see cref="Q35CudaEngine"/>): reads the
    /// GGUF's hyper-parameters or the validated NInfer recipe and hands the engine bounded weight
    /// sources its loader streams into VRAM (NVFP4 byte planes repacked without requantization).
    /// </summary>
    public sealed partial class Qwen35CudaExecutor : IDisposable
    {
        private const string Tag = "q35-cuda";
        private const string A = "qwen35";
        private GgufShardSet _gguf;
        private Q35CudaEngine _engine;

        public Q35CudaEngine Engine => _engine;
        public int VocabSize => _engine.VocabSize;
        public int ContextSize => _engine.ContextSize;

        // Sequence slots by id (null: free); the engine's active slot is _slots[ActiveSlot].
        private readonly System.Collections.Generic.List<Q35CudaEngine.Slot> _slots = new();
        /// <summary>Device memory a slot allocation must leave free (scratch growth, the drafter's
        /// per-step buffers, the CUDA context).</summary>
        private const long SlotReserveBytes = 1L << 30;

        public int ActiveSlot { get; private set; }
        public int NPast => _engine.Active.NPast;

        /// <summary>The attached DFlash drafter's settings, or null.</summary>
        public DFlashConfig Draft { get; private set; }

        public Qwen35CudaExecutor(string ggufPath, int maxContext, int nUbatch, string draftGgufPath = null)
        {
            var sw = System.Diagnostics.Stopwatch.StartNew();
            if (NinferQwen38ModelSource.IsNinferArtifact(ggufPath))
            {
                NinferQwen38ModelSource.ValidateExecutionRoute(TensorSharp.Runtime.BackendType.Cuda, 1, null, 1, draftGgufPath);
                using var source = NinferQwen38ModelSource.Open(ggufPath);
                InitializeNinfer(source, maxContext, nUbatch);
                Console.Error.WriteLine($"[{Tag}] NInfer target ready in {sw.Elapsed.TotalSeconds:F1}s");
                return;
            }
            _gguf = new GgufShardSet(ggufPath, Tag);
            try
            {
                string arch = _gguf.First.GetString("general.architecture") ?? string.Empty;
                if (arch != A)
                    throw new NotSupportedException($"[{Tag}] the direct-CUDA qwen35 engine runs {A}, not '{arch}'.");
                _gguf.BeginStreaming();
                _engine = CreateEngineWithCompanionHead(maxContext, nUbatch);
                _gguf.EndLoad();
                if (!string.IsNullOrWhiteSpace(draftGgufPath))
                    AttachDrafter(draftGgufPath);
                _slots.Add(_engine.Active);
                ActiveSlot = 0;
            }
            catch
            {
                Dispose();
                throw;
            }
            Console.Error.WriteLine($"[{Tag}] model ready in {sw.Elapsed.TotalSeconds:F1}s");
        }

        public void Forward(int[] tokens, float[] logitsOut, int[] ropePos = null) => _engine.Forward(tokens, logitsOut, ropePos);

        // ---- sequence slots ----

        /// <summary>A new empty slot, or -1 when the device has no room for one.</summary>
        public int SlotAlloc()
        {
            // No attention blocks yet: a prefill adds them as it goes, a checkpoint shares its source's.
            if (_engine.FreeDeviceBytes() < _engine.SlotBytes(0) + SlotReserveBytes)
                return -1;
            Q35CudaEngine.Slot s;
            try
            {
                s = _engine.CreateSlot(0);
            }
            catch (Exception ex)
            {
                Console.Error.WriteLine($"[{Tag}] slot allocation failed: {ex.Message}");
                return -1;
            }
            int id = _slots.IndexOf(null);
            if (id < 0)
            {
                id = _slots.Count;
                _slots.Add(s);
            }
            else
                _slots[id] = s;
            return id;
        }

        public bool SetActiveSlot(int id)
        {
            if (id < 0 || id >= _slots.Count || _slots[id] == null)
                return false;
            _engine.Active = _slots[id];
            ActiveSlot = id;
            return true;
        }

        /// <summary>Frees a slot other than the active one.</summary>
        public bool SlotFree(int id)
        {
            if (id < 0 || id >= _slots.Count || _slots[id] == null || id == ActiveSlot)
                return false;
            _engine.FreeSlot(_slots[id]);
            _slots[id] = null;
            return true;
        }

        public bool SlotHead(int id, out int head)
        {
            head = id >= 0 && id < _slots.Count && _slots[id] != null ? _slots[id].NPast : -1;
            return head >= 0;
        }

        public Q35CudaEngine.Slot Slot(int id) => id >= 0 && id < _slots.Count ? _slots[id] : null;

        /// <summary>The recurrent state only moves forward: a slot rewinds to its own head or to 0.</summary>
        public bool Rewind(int nPast) => nPast == NPast || (nPast == 0 && ResetChecked());

        public bool ResetChecked()
        {
            _engine.Reset();
            return true;
        }

        /// <summary>One decode step of several slots (positions must be their heads).</summary>
        public bool ForwardBatchedDecode(int[] slotIds, int[] tokens, int[] positions, float[] logits)
        {
            var slots = new Q35CudaEngine.Slot[slotIds.Length];
            for (int i = 0; i < slotIds.Length; i++)
            {
                slots[i] = Slot(slotIds[i]);
                if (slots[i] == null || slots[i].NPast != positions[i])
                    return false;
            }
            _engine.DecodeBatch(slots, tokens, logits, null);
            return true;
        }

        private void AttachDrafter(string path)
        {
            using var g = new GgufShardSet(path, Tag);
            var cfg = DFlashConfig.FromGguf(g.First);
            if (!cfg.HasConv || !cfg.HasSelector || cfg.MarkovRank > 0 || cfg.HasAttentionSinks || cfg.FinalLogitSoftcap != 0f)
                throw new NotSupportedException($"[{Tag}] the direct-CUDA engine drafts with DFlash2 (convolution and selector) drafters");
            g.BeginStreaming();
            var d = new Q35CudaEngine.DraftDesc
            {
                NEmbd = cfg.HiddenSize,
                NFf = cfg.IntermediateSize,
                NHead = cfg.NumHeads,
                NKvHead = cfg.NumKVHeads,
                HeadDim = cfg.HeadDim,
                BlockSize = cfg.BlockSize,
                MaskToken = cfg.MaskTokenId,
                SlidingWindow = cfg.SlidingWindow,
                ConvTaps = cfg.ConvKernelSize,
                ConvGroup = cfg.ConvGroupSize,
                SelRank = cfg.SelectorRank,
                SelTopK = cfg.SelectorTopK,
                RopeBase = cfg.RopeBase,
                Eps = cfg.Eps,
                LogitScale = cfg.LogitScale,
                TargetLayers = cfg.TargetLayerIds,
                Fc = g.Weight("fc.weight"),
                SelHidden = g.Weight("selector_hidden.weight"),
                SelPred = g.Weight("selector_predecessor.weight"),
                SelSucc = g.Weight("selector_successor.weight"),
                EncNorm = g.Floats("enc.output_norm.weight"),
                OutNorm = g.Floats("output_norm.weight"),
                Layers = new Q35CudaEngine.DraftLayerDesc[cfg.NumLayers],
            };
            for (int l = 0; l < cfg.NumLayers; l++)
            {
                string p = $"blk.{l}.";
                d.Layers[l] = new Q35CudaEngine.DraftLayerDesc
                {
                    Q = g.Weight(p + "attn_q.weight"), K = g.Weight(p + "attn_k.weight"), V = g.Weight(p + "attn_v.weight"),
                    O = g.Weight(p + "attn_output.weight"), Gate = g.Weight(p + "ffn_gate.weight"), Up = g.Weight(p + "ffn_up.weight"),
                    Down = g.Weight(p + "ffn_down.weight"),
                    AttnConvProj = g.Weight(p + "attn_conv_proj.weight"), FfnConvProj = g.Weight(p + "ffn_conv_proj.weight"),
                    AttnNorm = g.Floats(p + "attn_norm.weight"), FfnNorm = g.Floats(p + "ffn_norm.weight"),
                    QNorm = g.Floats(p + "attn_q_norm.weight"), KNorm = g.Floats(p + "attn_k_norm.weight"),
                    AttnConvBase = g.Floats(p + "attn_conv_base"), FfnConvBase = g.Floats(p + "ffn_conv_base"),
                };
            }
            _engine.AttachDrafter(d);
            g.EndLoad();
            Draft = cfg;
        }
        public void Reset() => _engine.Reset();

        private Q35CudaEngine.ModelDesc BuildModelDesc(int nCtx, int nUbatch)
        {
            var f = _gguf.First;
            int blocks = (int)f.GetUint32($"{A}.block_count");
            int nextn = (int)f.GetUint32($"{A}.nextn_predict_layers");
            int interval = (int)f.GetUint32($"{A}.full_attention_interval", 4);
            int[] sections = f.GetInt32Array($"{A}.rope.dimension_sections") ?? new[] { 11, 11, 10, 0 };
            int ropeDims = (int)f.GetUint32($"{A}.rope.dimension_count");
            if (ropeDims != Q35CudaEngine.RopeDims)
                throw new NotSupportedException($"[{Tag}] rotary width {ropeDims} (the kernels take {Q35CudaEngine.RopeDims})");
            if (sections.Length > 3 && sections[3] != 0)
                throw new NotSupportedException($"[{Tag}] a fourth M-RoPE section ({sections[3]} pairs)");
            int ssmState = (int)f.GetUint32($"{A}.ssm.state_size");
            if (ssmState != Q35CudaEngine.GdnHeadDim)
                throw new NotSupportedException($"[{Tag}] Gated DeltaNet state size {ssmState} (the kernels take {Q35CudaEngine.GdnHeadDim})");
            var tokEmbd = _gguf.Weight("token_embd.weight");
            var m = new Q35CudaEngine.ModelDesc
            {
                NLayer = blocks - nextn,
                NEmbd = (int)f.GetUint32($"{A}.embedding_length"),
                NFf = (int)f.GetUint32($"{A}.feed_forward_length"),
                NVocab = tokEmbd.Ne1,
                NHead = (int)f.GetUint32($"{A}.attention.head_count"),
                NKvHead = (int)f.GetUint32($"{A}.attention.head_count_kv"),
                HeadDim = (int)f.GetUint32($"{A}.attention.key_length"),
                GdnKHeads = (int)f.GetUint32($"{A}.ssm.group_count"),
                GdnVHeads = (int)f.GetUint32($"{A}.ssm.time_step_rank"),
                DConv = (int)f.GetUint32($"{A}.ssm.conv_kernel"),
                SectT = sections[0],
                SectH = sections[1],
                SectW = sections[2],
                RopeBase = f.GetFloat32($"{A}.rope.freq_base", 10000f),
                Eps = f.GetFloat32($"{A}.attention.layer_norm_rms_epsilon", 1e-6f),
                NCtx = nCtx,
                NUbatch = nUbatch,
                TokEmbd = tokEmbd,
                Output = _gguf.Weight("output.weight"),
                OutputNorm = _gguf.Floats("output_norm.weight"),
            };
            m.Layers = new Q35CudaEngine.LayerDesc[m.NLayer];
            for (int il = 0; il < m.NLayer; il++)
            {
                string p = $"blk.{il}.";
                bool recurrent = (il + 1) % interval != 0;
                CudaWeightDesc W(string name)
                {
                    if (_gguf.Has(p + name + ".scale"))
                        throw new NotSupportedException($"[{Tag}] {p}{name} carries a per-tensor scale; the engine takes unscaled NVFP4");
                    return _gguf.Weight(p + name + ".weight");
                }
                var d = new Q35CudaEngine.LayerDesc
                {
                    Recurrent = recurrent,
                    FfnGate = W("ffn_gate"),
                    FfnUp = W("ffn_up"),
                    FfnDown = W("ffn_down"),
                    AttnNorm = _gguf.Floats(p + "attn_norm.weight"),
                    PostNorm = _gguf.Floats(p + "post_attention_norm.weight"),
                };
                if (recurrent)
                {
                    d.InProj = new[] { W("attn_qkv"), W("attn_gate"), W("ssm_beta"), W("ssm_alpha") };
                    d.OutProj = W("ssm_out");
                    d.ConvW = _gguf.Floats(p + "ssm_conv1d.weight");
                    d.DtBias = _gguf.Floats(p + "ssm_dt.bias");
                    d.SsmA = _gguf.Floats(p + "ssm_a");
                    d.SsmNorm = _gguf.Floats(p + "ssm_norm.weight");
                }
                else
                {
                    d.InProj = new[] { W("attn_q"), W("attn_k"), W("attn_v") };
                    d.OutProj = W("attn_output");
                    d.QNorm = _gguf.Floats(p + "attn_q_norm.weight");
                    d.KNorm = _gguf.Floats(p + "attn_k_norm.weight");
                }
                m.Layers[il] = d;
            }
            return m;
        }

        public void Dispose()
        {
            if (_engine != null)
                foreach (var s in _slots)
                    if (s != null && !ReferenceEquals(s, _engine.Active))
                        _engine.FreeSlot(s);
            _slots.Clear();
            _engine?.Dispose();
            _engine = null;
            _gguf?.Dispose();
            _gguf = null;
        }
    }
}
