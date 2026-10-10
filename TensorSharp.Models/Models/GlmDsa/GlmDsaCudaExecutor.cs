// Copyright (c) Zhongkai Fu. All rights reserved.
// https://github.com/zhongkaifu/TensorSharp
//
// This file is part of TensorSharp.
//
// TensorSharp is licensed under the BSD-3-Clause license found in the LICENSE file in the root directory of this source tree.
using System;
using TensorSharp.Cuda;
using TensorSharp.Runtime;

namespace TensorSharp.Models
{
    /// <summary>
    /// GLM-5.3-Flash on the direct-CUDA engine (<see cref="GlmCudaEngine"/>): reads the GGUF's
    /// hyper-parameters and hands the engine its weights as shard offsets, which its loader
    /// streams into VRAM.
    /// </summary>
    internal sealed class GlmDsaCudaExecutor : IGlmExecutor
    {
        private const string Tag = "glm-cuda";
        private GgufShardSet _gguf;
        private GlmCudaEngine _engine;

        /// <summary>The engine behind this executor, for tests that probe its stages.</summary>
        internal GlmCudaEngine Engine => _engine;

        public string Kind => "cuda";
        public int NPast => _engine.NPast;
        public int VocabSize => _engine.VocabSize;
        public int ContextSize => _engine.ContextSize;
        public int UBatch => _engine.UBatch;
        public int ActiveSlot => _engine.ActiveSlot;

        public GlmDsaCudaExecutor(string ggufPath, int maxContext, int nUbatch, int nGpu)
        {
            var sw = System.Diagnostics.Stopwatch.StartNew();
            _gguf = new GgufShardSet(ggufPath, Tag);
            try
            {
                string arch = _gguf.First.GetString("general.architecture") ?? string.Empty;
                if (!GlmDsaArchitecture.IsGlm5Next(arch))
                    throw new NotSupportedException(
                        $"[{Tag}] the direct-CUDA GLM engine runs GLM-5.3-Flash (glm5next), not '{arch}'; use --backend ggml_cuda.");
                _gguf.BeginStreaming();
                var desc = BuildModelDesc(arch, maxContext, nUbatch);
                _engine = new GlmCudaEngine(desc, nGpu);
                _gguf.EndLoad();
            }
            catch
            {
                Dispose();
                throw;
            }
            Console.Error.WriteLine($"[{Tag}] model ready in {sw.Elapsed.TotalSeconds:F1}s");
        }

        public bool Forward(int[] tokens, float[] logitsOut)
        {
            _engine.Forward(tokens, logitsOut);
            return true;
        }

        public void Reset() => _engine.Reset();
        public bool ResetChecked() => _engine.ResetChecked();
        public bool Rewind(int nPast) => _engine.Rewind(nPast);

        // ---- sequence slots (the native executor's TSGgml_GlmSlot* contract) ----
        public int SlotAlloc() => _engine.SlotAlloc();
        public bool SetActiveSlot(int slot) => _engine.SetActiveSlot(slot);
        public bool SlotFree(int slot) => _engine.SlotFree(slot);
        public bool SlotHead(int slot, out int head) => _engine.SlotHead(slot, out head);
        // The engine's scratch is sized at load, so no step ever needs a retained slot's memory:
        // reclaim happens only when a new slot does not fit, which the model handles.
        public bool SetReclaimableSlots(int[] slots) => true;
        public int TakeReclaimedSlots(int[] buffer) => 0;
        public bool ForwardBatchedDecode(int[] slots, int[] tokens, int[] positions, float[] logits)
            => _engine.ForwardBatchedDecode(slots, tokens, positions, logits);

        public bool QueueVisionRows(float[] rows, int nRows, int index) => _engine.QueueVisionRows(rows, nRows, index);
        public void ClearVisionRows() => _engine.ClearVisionRows();

        // ---- speculative decoding: verify windows and the KDA snapshot serve the n-gram drafter;
        // the checkpoint's NextN block is not built (nor on the native executor), so no draft head ----
        public bool HasDraftHead => false;
        /// <summary>A verify window for the n-gram drafter (the checkpoint's NextN block is not built).</summary>
        public bool SpecForward(int[] tokens, float[] hOut, float[] logitsOut, bool allLogitsRows)
        {
            if (tokens.Length > GlmCudaEngine.MaxSpecRows)
                return false;
            _engine.SpecForward(tokens, hOut, logitsOut, allLogitsRows);
            return true;
        }
        public bool DraftStep(int token, float[] hPrev, int pos, float[] logitsOut, float[] hOut) => false;
        public bool DraftCatchUp(int[] tokens, float[] hRows, int startPos) => false;
        public bool KdaStateCapture() => _engine.KdaStateCapture();
        public int KdaStateRestore() => _engine.KdaStateRestore();

        private GlmCudaEngine.ModelDesc BuildModelDesc(string a, int nCtx, int nUbatch)
        {
            GgufFile g = _gguf.First;
            int nAll = (int)g.GetUint32($"{a}.block_count");
            int nNextn = (int)g.GetUint32($"{a}.nextn_predict_layers", 0);
            int nLayer = nAll - nNextn;
            if ((int)g.GetUint32($"{a}.rope.dimension_count", 0) != 0)
                throw new NotSupportedException($"[{Tag}] glm5next with rotary MLA dimensions is not supported");
            if ((int)g.GetUint32($"{a}.hyper_connection.count", 0) != 4)
                throw new NotSupportedException($"[{Tag}] the hyper-connection kernels take 4 streams");
            uint gating = g.GetUint32($"{a}.expert_gating_func", 2);
            if (gating != 0 && gating != 2)
                throw new NotSupportedException($"[{Tag}] expert gating function {gating} (sigmoid expected)");
            uint[] kvHeads = g.GetUint32Array($"{a}.attention.head_count_kv")
                ?? throw new NotSupportedException($"[{Tag}] glm5next needs attention.head_count_kv to tell KDA layers from MLA ones");
            float[] clamp = g.GetFloatArray($"{a}.swiglu_clamp_exp");
            int nDenseLead = (int)g.GetUint32($"{a}.leading_dense_block_count", 0);
            int ctxTrain = (int)g.GetUint32($"{a}.context_length", 0);

            var m = new GlmCudaEngine.ModelDesc
            {
                NLayer = nLayer,
                NEmbd = (int)g.GetUint32($"{a}.embedding_length"),
                NHead = (int)g.GetUint32($"{a}.attention.head_count"),
                NVocab = (int)g.GetUint32($"{a}.vocab_size", 0),
                KdaHeadDim = (int)g.GetUint32($"{a}.kda.head_dim"),
                DConv = (int)g.GetUint32($"{a}.ssm.conv_kernel"),
                KdaGateLowerBound = g.GetFloat32($"{a}.kda.gate_lower_bound", -5.0f),
                QLoraRank = (int)g.GetUint32($"{a}.attention.q_lora_rank"),
                KvLoraRank = (int)g.GetUint32($"{a}.attention.kv_lora_rank"),
                HeadDimK = (int)g.GetUint32($"{a}.attention.key_length_mla"),
                HeadDimV = (int)g.GetUint32($"{a}.attention.value_length_mla"),
                IdxNHead = (int)g.GetUint32($"{a}.attention.indexer.head_count"),
                IdxHeadDim = (int)g.GetUint32($"{a}.attention.indexer.key_length"),
                IdxTopK = (int)g.GetUint32($"{a}.attention.indexer.top_k"),
                IdxKpool = (int)g.GetUint32($"{a}.attention.indexer.kpool"),
                NExpert = (int)g.GetUint32($"{a}.expert_count"),
                NExpertUsed = (int)g.GetUint32($"{a}.expert_used_count"),
                NFfExp = (int)g.GetUint32($"{a}.expert_feed_forward_length"),
                NFf = (int)g.GetUint32($"{a}.feed_forward_length"),
                NFfShexp = (int)g.GetUint32($"{a}.expert_shared_feed_forward_length",
                    g.GetUint32($"{a}.expert_feed_forward_length")),
                ExpertWeightsScale = g.GetFloat32($"{a}.expert_weights_scale", 1.0f),
                ExpertWeightsNorm = g.GetBool($"{a}.expert_weights_norm", false),
                SwigluClamp = clamp != null && clamp.Length > 0 ? clamp[0] : 0f,
                HcSinkhornIters = (int)g.GetUint32($"{a}.hyper_connection.sinkhorn_iterations", 20),
                HcEps = g.GetFloat32($"{a}.hyper_connection.epsilon", 1e-6f),
                RmsEps = g.GetFloat32($"{a}.attention.layer_norm_rms_epsilon", 1e-5f),
                NormEps = g.GetFloat32($"{a}.attention.layer_norm_epsilon", 1e-6f),
                NCtx = ctxTrain > 0 ? Math.Min(nCtx, ctxTrain) : nCtx,
                NUbatch = nUbatch,
                TokEmbd = _gguf.Weight("token_embd.weight"),
                OutputNorm = _gguf.Floats("output_norm.weight"),
                Layers = new GlmCudaEngine.LayerDesc[nLayer],
            };
            // An untied head, or the embedding read back.
            m.Output = _gguf.Has("output.weight") ? _gguf.Weight("output.weight") : m.TokEmbd;
            if (m.NVocab == 0)
                m.NVocab = m.Output.Ne1;
            if (m.SwigluClamp <= 0)
                throw new NotSupportedException($"[{Tag}] glm5next without swiglu_clamp_exp");

            for (int il = 0; il < nLayer; il++)
            {
                string p = $"blk.{il}.";
                bool recurrent = il < kvHeads.Length && kvHeads[il] == 0;
                var L = new GlmCudaEngine.LayerDesc
                {
                    Recurrent = recurrent,
                    Moe = il >= nDenseLead,
                    AttnNorm = _gguf.Floats(p + "attn_norm.weight"),
                    FfnNorm = _gguf.Floats(p + "ffn_norm.weight"),
                    HcAttnFn = _gguf.Weight(p + "hc_attn_fn.weight"),
                    HcAttnScale = _gguf.Floats(p + "hc_attn_scale.weight"),
                    HcAttnBase = _gguf.Floats(p + "hc_attn_base.weight"),
                    HcFfnFn = _gguf.Weight(p + "hc_ffn_fn.weight"),
                    HcFfnScale = _gguf.Floats(p + "hc_ffn_scale.weight"),
                    HcFfnBase = _gguf.Floats(p + "hc_ffn_base.weight"),
                };
                if (recurrent)
                {
                    L.KdaQ = _gguf.Weight(p + "attn_q.weight");
                    L.KdaK = _gguf.Weight(p + "attn_k.weight");
                    L.KdaV = _gguf.Weight(p + "attn_v.weight");
                    L.KdaFA = _gguf.Weight(p + "ssm_f_a.weight");
                    L.KdaFB = _gguf.Weight(p + "ssm_f_b.weight");
                    L.KdaGA = _gguf.Weight(p + "ssm_g_a.weight");
                    L.KdaGB = _gguf.Weight(p + "ssm_g_b.weight");
                    L.KdaBeta = _gguf.Weight(p + "ssm_beta.weight");
                    L.KdaOut = _gguf.Weight(p + "attn_output.weight");
                    L.ConvQ = _gguf.Floats(p + "ssm_conv1d_q.weight");
                    L.ConvK = _gguf.Floats(p + "ssm_conv1d_k.weight");
                    L.ConvV = _gguf.Floats(p + "ssm_conv1d_v.weight");
                    L.DtBias = _gguf.Floats(p + "ssm_dt.bias");
                    L.SsmA = _gguf.Floats(p + "ssm_a");
                    L.SsmNorm = _gguf.Floats(p + "ssm_norm.weight");
                }
                else
                {
                    L.WqA = _gguf.Weight(p + "attn_q_a.weight");
                    L.QANorm = _gguf.Floats(p + "attn_q_a_norm.weight");
                    L.WqB = _gguf.Weight(p + "attn_q_b.weight");
                    L.WkvA = _gguf.Weight(p + "attn_kv_a_mqa.weight");
                    L.KvANorm = _gguf.Floats(p + "attn_kv_a_norm.weight");
                    L.WkB = _gguf.Weight(p + "attn_k_b.weight");
                    L.WvB = _gguf.Weight(p + "attn_v_b.weight");
                    L.Wo = _gguf.Weight(p + "attn_output.weight");
                    L.IdxK = _gguf.Weight(p + "indexer.attn_k.weight");
                    L.IdxGate = _gguf.Weight(p + "indexer_compressor_gate.weight");
                    L.IdxKNormW = _gguf.Floats(p + "indexer.k_norm.weight");
                    L.IdxKNormB = _gguf.Floats(p + "indexer.k_norm.bias");
                    L.IdxQB = _gguf.Weight(p + "indexer.attn_q_b.weight");
                    L.IdxProj = _gguf.Floats(p + "indexer.proj.weight");
                    L.IdxApe = _gguf.Floats(p + "indexer_compressor_ape.weight");
                }
                if (L.Moe)
                {
                    L.GateInp = _gguf.Floats(p + "ffn_gate_inp.weight");
                    L.ExpProbsBias = _gguf.Floats(p + "exp_probs_b.bias", required: false);
                    L.GateExps = _gguf.Weight(p + "ffn_gate_exps.weight");
                    L.UpExps = _gguf.Weight(p + "ffn_up_exps.weight");
                    L.DownExps = _gguf.Weight(p + "ffn_down_exps.weight");
                    L.GateShexp = _gguf.Weight(p + "ffn_gate_shexp.weight");
                    L.UpShexp = _gguf.Weight(p + "ffn_up_shexp.weight");
                    L.DownShexp = _gguf.Weight(p + "ffn_down_shexp.weight");
                }
                else
                {
                    L.FfnGate = _gguf.Weight(p + "ffn_gate.weight");
                    L.FfnUp = _gguf.Weight(p + "ffn_up.weight");
                    L.FfnDown = _gguf.Weight(p + "ffn_down.weight");
                }
                m.Layers[il] = L;
            }
            return m;
        }

        public void Dispose()
        {
            _engine?.Dispose();
            _engine = null;
            _gguf?.Dispose();
            _gguf = null;
        }
    }
}
