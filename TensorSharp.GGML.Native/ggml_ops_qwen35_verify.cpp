// Copyright (c) Zhongkai Fu. All rights reserved.
// https://github.com/zhongkaifu/TensorSharp
//
// This file is part of TensorSharp.
//
// TensorSharp is licensed under the BSD-3-Clause license found in the LICENSE file in the root directory of this source tree.
//
// TensorSharp is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the BSD-3-Clause License for more details.
#include "ggml_ops_internal.h"
#include "ggml_ops_transformer_common.h"
#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdio>

using namespace tsg;

// ============================================================================
// TSGgml_Qwen35ModelVerify  --  the N-token sibling of TSGgml_Qwen35ModelDecode.
//
// MTP speculative decoding verifies a window of (1 + draft) tokens in one trunk
// pass. The per-op SpecForward fallback runs that pass op-by-op (~1 s/step on
// WDDM ggml_cuda); this kernel runs the WHOLE hybrid transformer over N tokens as
// ONE ggml graph: prefill-style causal flash attention (read cache prefix [0,
// start_pos) + the N fresh K/V, causal F16 mask, append the N rows to the cache),
// GDN recurrence over the N tokens via ggml_gated_delta_net(K=N) (the op also
// emits a state snapshot per prefix length so partial-acceptance rollback can pick
// the committed state without a re-forward), batched MoE/dense FFN, and a folded
// final-norm that outputs BOTH the per-row logits [vocab, N] AND the post-norm
// hidden [hidden, N] the MTP draft head consumes.
//
// GDN state is passed per-layer via host in/out pointers (conv_state_in/out in
// ggml [time, channel] layout, delta_state_in/out as [S, S, H]); the kernel reads
// the pre-window state and writes the post-window state. Non-persistent (rebuilt
// per call) -- correctness first; CUDA-graph capture is a follow-up that needs a
// fixed-N persistent cache (cf. g_q35dc). Returns 0 on anything it cannot handle
// so the C# caller falls back to the per-op SpecForward.
// ============================================================================
namespace
{
    // Persistent verify-graph cache (multi-entry, keyed by N + window stride). With
    // TS_Q35_VERIFY_PERSIST (default ON) the per-(N,window) graph is built ONCE and
    // reused, so subsequent verify steps of the same shape only upload the per-call
    // inputs (hidden / pos / kv_index / mask / GDN state) + recompute -> the
    // ~150-580 ms C++ build is amortized AND ggml-cuda can CUDA-graph-capture the
    // replay. Multi-entry because spec alternates N=draft+1 (speculative) and
    // N=accepted+1 (rollback re-forward); a single entry would evict + rebuild every
    // call.
    //
    // Each entry's activations live in a PRIVATE gallocr (lifetime-packed) that is
    // planned exactly ONCE and never re-reserved, so node addresses stay stable for
    // graph reuse / CUDA capture — same stability as the previous every-tensor-its-
    // own-slot alloc_ctx_tensors layout, at a fraction of the VRAM. Own-slot layouts
    // measured 0.5 GB (N=1) to 1.8 GB (N=20) PER ENTRY on Qwen3.6-27B: the dominant
    // cost was 48 GDN layers x ~3.2 MB delta-state in/out slots duplicated into
    // every entry, which the 16-slot cache turned into multi-GB VRAM growth during
    // ordinary MTP decoding. The GDN state slices now live in ONE shared device
    // buffer (g_q35v_state, below) bound into every entry's graph, and gallocr
    // packing recycles the per-layer intermediates, so an N<=9 verify entry costs
    // ~10-30 MB.
    struct Q35VerifyCache
    {
        bool valid = false;
        int n = 0, window = 0, num_layers = 0, out_vocab = 0, n_logits = 0;
        bool has_normed = false;
        const void* sig = nullptr;
        ggml_context* ctx = nullptr;
        ggml_backend_buffer_t buffer = nullptr;
        ggml_cgraph* graph = nullptr;
        ggml_tensor* hidden_t = nullptr;
        ggml_tensor* pos_t = nullptr;
        ggml_tensor* kv_index = nullptr;
        ggml_tensor* mask_t = nullptr;
        ggml_tensor* logits_out = nullptr;
        ggml_tensor* normed_out = nullptr;
        /// DFlash residual taps: one [H, N] block per requested layer, holding the
        /// residual ENTERING it. Empty unless the caller asked for them.
        std::vector<ggml_tensor*> capture_out;
        int capture_count = 0;
        std::vector<ggml_tensor*> conv_in, delta_in, conv_out, delta_out;
        /// Per-token recurrent-state snapshots, one entry per RECURRENT layer.
        /// conv_snaps is [convDim * conv_dim, K] and delta_snaps [D, K], slot s
        /// holding the state as it stood s tokens before the end of the batch.
        /// They exist so a partially-rejected verify does not have to restore a
        /// pre-verify snapshot and re-forward the accepted prefix: the state it
        /// would have recomputed is already sitting in slot (N-1-accepted).
        std::vector<ggml_tensor*> conv_snaps, delta_snaps;
        /// Per (recurrent layer, slot) views of the two above, shaped exactly like
        /// the live conv_state_in / delta_state_in so one slot can be committed with
        /// a device-to-device tensor copy. Index [i * n_snapshots + slot].
        std::vector<ggml_tensor*> conv_snap_slots, delta_snap_slots;
        int n_snapshots = 0;
        /// The post-window state was NOT downloaded; the caller commits it (or one
        /// snapshot slot) into the live slices on the device instead. True for every
        /// step of a speculative session, single-row plain steps included - which is
        /// the point: the 151 MB state then never crosses PCIe at all.
        bool deferred_state = false;
        std::size_t buffer_bytes = 0;
        std::uint64_t lru = 0;
        void reset()
        {
            if (buffer != nullptr) { ggml_backend_buffer_free(buffer); buffer = nullptr; }
            if (ctx != nullptr) { ggml_free(ctx); ctx = nullptr; }
            graph = nullptr; valid = false;
            hidden_t = pos_t = kv_index = mask_t = logits_out = normed_out = nullptr;
            conv_in.clear(); delta_in.clear(); conv_out.clear(); delta_out.clear();
            conv_snaps.clear(); delta_snaps.clear();
            conv_snap_slots.clear(); delta_snap_slots.clear(); n_snapshots = 0;
            deferred_state = false;
            capture_out.clear(); capture_count = 0;
            n = window = num_layers = out_vocab = n_logits = 0; has_normed = false; sig = nullptr;
            buffer_bytes = 0;
        }
    };
    Q35VerifyCache g_q35vc[16];
    std::uint64_t g_q35vc_clock = 0;

    // The entry whose snapshots are live, i.e. the one the most recent verify
    // computed or replayed. TSGgml_Qwen35FetchStateSnapshot reads from it, and it
    // is cleared whenever a call runs without snapshots so a stale fetch cannot
    // silently return the wrong sequence's state.
    Q35VerifyCache* g_q35vc_last_snap = nullptr;
    std::size_t g_q35vc_snap_conv_bytes = 0;
    std::size_t g_q35vc_snap_delta_bytes = 0;

    // Total-VRAM budget for the resident persist verify graphs. Each entry is a
    // whole-model graph in its own alloc_ctx buffer (own slots, needed for CUDA-graph
    // capture), sized ~0.2 GB (N=1) to ~0.8 GB (N=maxDraft+1). Without a cap, a
    // session that exercises many draft lengths (plain N=1, speculative N=draft+1,
    // and every rollback N=accepted+1) could resident-cache all ~9 shapes and add
    // several GB on top of the weights + KV cache — re-overcommitting a 16 GB card.
    // When a new entry would push the cached total past this budget, the least-
    // recently-used entries are evicted (their buffers freed) first. Default 1.5 GB
    // keeps the two hottest shapes (N=1 and the current draft N) plus a rollback or
    // two resident; TS_Q35_VERIFY_CACHE_BUDGET_MB overrides (0 = unbounded/legacy).
    std::int64_t q35_verify_cache_budget_bytes()
    {
        static const std::int64_t budget = []{
            const char* e = std::getenv("TS_Q35_VERIFY_CACHE_BUDGET_MB");
            std::int64_t mb = 1536;
            if (e != nullptr && *e != '\0') { char* end = nullptr; long v = std::strtol(e, &end, 10); if (end != e && v >= 0) mb = v; }
            return mb * 1024 * 1024;
        }();
        return budget;
    }

    std::int64_t q35_verify_cache_resident_bytes()
    {
        std::int64_t total = 0;
        for (const auto& c : g_q35vc)
            if (c.valid) total += static_cast<std::int64_t>(c.buffer_bytes);
        return total;
    }

    // Free LRU entries until the cached total (including `incoming` about-to-be-added
    // bytes) fits the budget. `keep` is a just-populated slot that must not be evicted.
    void q35_verify_cache_evict_to_budget(std::size_t incoming, const Q35VerifyCache* keep)
    {
        const std::int64_t budget = q35_verify_cache_budget_bytes();
        if (budget <= 0) return;
        while (q35_verify_cache_resident_bytes() + static_cast<std::int64_t>(incoming) > budget)
        {
            Q35VerifyCache* victim = nullptr;
            for (auto& c : g_q35vc)
            {
                if (!c.valid || &c == keep) continue;
                if (victim == nullptr || c.lru < victim->lru) victim = &c;
            }
            if (victim == nullptr) break; // nothing else evictable
            victim->reset();
        }
    }

    // Shared device buffer holding the verify's per-GDN-layer state slices (delta
    // in/out + conv in/out windows), mirroring llama.cpp's dedicated recurrent-state
    // (RS) buffer. Host-mode verify uploads the current state into the *_in slices
    // before each compute and downloads the post-window state from the *_out slices
    // after, so sharing them across every cached graph (and the non-persist prefill
    // path) is safe — and it keeps the ~300 MB of state out of each entry's
    // activation buffer AND out of the reuse gallocr. Deliberately NOT part of the
    // host-buffer cache: the C# side invalidates the decode's host-keyed state
    // bindings around every verify for cache coherence, which would free a shared
    // cacheable buffer out from under the cached (possibly CUDA-captured) graphs that
    // pin it. Freed only on backend swap; contents are re-uploaded every call, so
    // staleness is impossible.
    // Per-rank: under tensor parallelism every rank builds its own prefill
    // graph on its own backend, and a single shared buffer keyed to one
    // backend would be freed out from under rank 0 the moment rank 1 builds
    // (the backend-swap check below). Non-TP runs always use slot 0.
    ggml_backend_buffer_t g_q35v_state_bufs[TSG_MAX_DEVICES] = {};
    std::size_t g_q35v_state_buf_sizes[TSG_MAX_DEVICES] = {};
    ggml_backend_t g_q35v_state_backends[TSG_MAX_DEVICES] = {};
#define g_q35v_state_buf      (g_q35v_state_bufs[::tsg::g_active_rank])
#define g_q35v_state_buf_size (g_q35v_state_buf_sizes[::tsg::g_active_rank])
#define g_q35v_state_backend  (g_q35v_state_backends[::tsg::g_active_rank])

    // Resources a tensor-parallel verify graph borrows between "build" and
    // "execute" (the same shape as gemma4's G4VerifyTpPending): the TP prefill
    // allocates from the context pool + the per-rank reuse gallocr, so nothing
    // naturally outlives the call — and the caller only runs the graph after
    // every rank has built one. Each rank parks its context here; the slot is
    // recycled when that rank builds its next graph.
    struct Q35VerifyTpPending
    {
        PooledContextHandle context;
        std::vector<BufferHandle> ephemeral;
        tsg::TpRankPlan plan;

        void reset()
        {
            plan.clear();
            ephemeral.clear();
            context = PooledContextHandle();
        }
    };
    Q35VerifyTpPending g_q35v_tp[TSG_MAX_DEVICES];

    // Ensure the shared GDN state buffer covers `needed` bytes. Any resize would
    // move slices pinned by cached graphs, so the caller must reset the verify
    // cache before growing (only happens on a model-shape change).
    bool ensure_q35v_state_buf(std::size_t needed)
    {
        if (g_q35v_state_backend != g_backend)
        {
            // Backend swapped (model reload): the old backend already freed its
            // buffers on teardown, so drop the stale handle rather than freeing
            // through it.
            g_q35v_state_buf = nullptr;
            g_q35v_state_buf_size = 0;
            g_q35v_state_backend = g_backend;
        }
        if (g_q35v_state_buf != nullptr && g_q35v_state_buf_size >= needed)
            return true;
        for (auto& c : g_q35vc) c.reset();
        if (g_q35v_state_buf != nullptr)
        {
            ggml_backend_buffer_free(g_q35v_state_buf);
            g_q35v_state_buf = nullptr;
            g_q35v_state_buf_size = 0;
        }
        ggml_backend_buffer_type_t buft = ggml_backend_get_default_buffer_type(g_backend);
        if (buft == nullptr)
            return false;
        g_q35v_state_buf = ggml_backend_buft_alloc_buffer(buft, needed);
        if (g_q35v_state_buf == nullptr)
            return false;
        ggml_backend_buffer_set_usage(g_q35v_state_buf, GGML_BACKEND_BUFFER_USAGE_COMPUTE);
        g_q35v_state_buf_size = needed;
        if (vram_log_enabled())
            vram_log("q35-verify-state-buf", static_cast<std::int64_t>(needed));
        return true;
    }

    void fill_verify_causal_mask(std::vector<ggml_fp16_t>& mask, int window, int n, int start_pos, int total_len)
    {
        mask.resize(static_cast<std::size_t>(window) * n);
        const ggml_fp16_t neg_inf = ggml_fp32_to_fp16(-std::numeric_limits<float>::infinity());
        const ggml_fp16_t zero_val = ggml_fp32_to_fp16(0.0f);
        for (int qi = 0; qi < n; qi++)
        {
            int threshold = start_pos + qi;
            ggml_fp16_t* row = &mask[static_cast<std::size_t>(qi) * window];
            for (int ki = 0; ki < window; ki++)
                row[ki] = (ki <= threshold && ki < total_len) ? zero_val : neg_inf;
        }
    }

    int qwen35_model_verify_impl(
        const TSGgmlQwen35LayerDesc* layers, int num_layers,
        void* hidden_data, int hidden_size, int start_pos, int num_tokens,
        int num_heads, int num_kv_heads, int head_dim, int cache_size,
        int rope_n_dims, int rope_mode, int kv_cache_type,
        int conv_kernel, int head_k_dim, int head_v_dim, int num_k_heads, int num_v_heads,
        float eps, float rope_base, float rope_freq_scale,
        int num_experts, int num_experts_used, int expert_ff, int shared_ff,
        int norm_topk, float expert_weights_scale,
        void* logits_data, int vocab_size,
        const void* lm_head_data, int lm_head_type, std::int64_t lm_head_ne0, std::int64_t lm_head_ne1, std::int64_t lm_head_bytes,
        const void* final_norm_data, void* normed_out, int n_logit_rows,
        const std::int32_t* mrope_pos, const std::int32_t* mrope_sections,
        int tp_degree, void** tp_plan_out,
        float* capture_data, const int* capture_layers, int capture_count,
        int state_snapshots, int* state_snapshots_used, int device_state_current,
        int defer_state_download)
    {
        if (state_snapshots_used != nullptr)
            *state_snapshots_used = 1;
        if (!ensure_backend())
            return 0;
        if (layers == nullptr || num_layers <= 0 || hidden_data == nullptr || num_tokens < 1)
        {
            set_last_error("Qwen3.5 model verify: invalid arguments.");
            return 0;
        }
        if (layers[0].struct_bytes != static_cast<std::int32_t>(sizeof(TSGgmlQwen35LayerDesc)))
        {
            set_last_error("Qwen3.5 model verify: descriptor size mismatch.");
            return 0;
        }
        // gated_delta_net requires S_k == S_v (state is [S_v, S_v, H]).
        if (head_k_dim != head_v_dim)
        {
            set_last_error("Qwen3.5 model verify: head_k_dim != head_v_dim unsupported.");
            return 0;
        }

        // Tensor parallelism — the prefill sibling of qwen35_model_decode_impl's
        // tp_mode: the caller drives one rank at a time (SetActiveRank); this
        // builds the active rank's whole-model N-token graph over its shards,
        // uploads the inputs, and hands back a segmented plan (via tp_plan_out)
        // instead of running it. Per-layer dims (attention heads, GDN heads,
        // stacked experts, shared-expert width) arrive already sharded; the MoE
        // router stays global and the ep_lut/ep_mask pair below confines each
        // rank's mul_mat_id to the whole experts it owns.
        // Plan mode is requested by PASSING tp_plan_out, not by the degree: a
        // distributed run drives one local rank per node and still needs the
        // plan, because its reduction happens across nodes rather than across
        // local ranks. Callers that want the graph run inline pass nullptr.
        const bool tp_mode = tp_degree >= 1 && tp_plan_out != nullptr;
        if (tp_mode)
        {
            *tp_plan_out = nullptr;
            // Recycle this rank's previous parked graph — the caller finished
            // executing it before asking for another.
            g_q35v_tp[g_active_rank].reset();
        }
        const int tp_rank = tp_mode ? g_active_rank : 0;
        // Cluster degree, not this process's - see ggml_ops_qwen35_decode.cpp.
        const int tp_group_degree = tsg::tp_global_degree(tp_degree);
        const int stacked_experts = tp_mode && num_experts > 0 ? num_experts / tp_group_degree : num_experts;
        if (tp_mode && num_experts > 0 &&
            (num_experts % tp_degree != 0 || stacked_experts < num_experts_used))
        {
            set_last_error("Qwen3.5 model verify: expert count is not shardable across the tensor-parallel ranks.");
            return 0;
        }

        const int N = num_tokens;
        const int H = hidden_size;
        const int totalSeqLen = start_pos + N;
        const int qDim = num_heads * head_dim;
        const int qFullDim = qDim * 2;            // Q + gate interleaved per head
        const int kDim = num_kv_heads * head_dim;
        const float attn_scale = 1.0f / std::sqrt(static_cast<float>(head_dim));
        const int convDim = conv_kernel - 1;
        const int key_dim = head_k_dim * num_k_heads;
        const int value_dim = head_v_dim * num_v_heads;
        const int conv_dim = 2 * key_dim + value_dim;
        const ggml_type kvType = static_cast<ggml_type>(kv_cache_type);
        if (convDim <= 0 || totalSeqLen > cache_size)
        {
            set_last_error("Qwen3.5 model verify: bad conv dim or sequence exceeds cache.");
            return 0;
        }
        const bool fold = logits_data != nullptr && lm_head_data != nullptr &&
                          final_norm_data != nullptr && vocab_size > 0;
        if (!fold)
        {
            set_last_error("Qwen3.5 model verify: folded lm_head required.");
            return 0;
        }

        // Prefill only needs the LAST token's logits (to sample the first decode
        // token); MTP verify needs all N rows. n_logit_rows in [1, N) computes the
        // lm_head only over the last n_logit_rows columns of the post-norm hidden,
        // so a 2048-token prefill writes vocab*1 floats (not vocab*2048 ~ 2 GB) and
        // skips the lm_head matmul over the first N-1 tokens. <=0 or >=N => all N.
        const int n_logits = (n_logit_rows > 0 && n_logit_rows < N) ? n_logit_rows : N;

        // Persistent per-(N,window) graph cache (build amortization + CUDA-graph
        // capture). DEFAULT ON: the earlier reuse access-violation (0xC0000005) was the
        // 3D N-row set_rows (heads in ne2) faulting on cgraph reuse; replacing it with a
        // 2D set_rows PER HEAD (llama.cpp's proven KV-write shape) made reuse stable
        // (validated 252 reuses, no crash). Reuse: setup ~8 ms + compute ~12-20 ms vs
        // ~61 ms non-persist build. TS_Q35_VERIFY_PERSIST=0 forces the rebuild path.
        constexpr const char* kQ35VerifyKernel = "Qwen3.5 model verify";
        // MoE CPU offload segments this graph, and the persist replay below
        // re-runs the cached graph as ONE submission — which would skip every
        // seam. The persist path only serves the small MTP-verify shapes, so
        // decline it rather than teach it a second execution mode.
        bool any_cpu_moe = false;
        for (int l = 0; l < num_layers; l++)
            if (layers[l].is_moe != 0 && layers[l].cpu_moe != 0) { any_cpu_moe = true; break; }
        // The DFlash drafter's encoder reads the residual entering a handful of
        // trunk layers. Tapping them here keeps speculation on the fused trunk
        // instead of forcing the op-by-op loop just to observe them.
        const int cap_count = (capture_data != nullptr && capture_layers != nullptr && capture_count > 0)
            ? capture_count : 0;
        // Per-token recurrent-state snapshots for a rollback-free partial accept.
        // Only on the persist path, because the fetch reads the graph's own output
        // tensors after the fact and only a persisted graph still owns them; only in
        // host state mode, because resident mode updates the state in place; and only
        // when the caller asked for at least two (one is what the plain path already
        // keeps). TS_Q35_VERIFY_SNAPSHOTS=0 forces the old snapshot/re-forward path.
        static const bool fv_snapshots_cfg = []{ const char* e = std::getenv("TS_Q35_VERIFY_SNAPSHOTS"); return e == nullptr || e[0] != '0'; }();
        static const bool fv_persist_cfg = []{ const char* e = std::getenv("TS_Q35_VERIFY_PERSIST"); return e == nullptr || e[0] != '0'; }();
        // Multimodal MRoPE: per-axis positions (T/H/W/E axis-concatenated, [4N] I32)
        // route the attention RoPE through ggml_rope_multi (interleaved MRoPE, the
        // Qwen3-VL LLM rope). Text prompts keep the plain sequential NeoX rope.
        const bool use_mrope = mrope_pos != nullptr && mrope_sections != nullptr;

        // PREFILL (n_logits < N) processes a long prompt one-shot, so it never reuses
        // the cached graph; force the NON-PERSIST path (pooled ctx + gallocr lifetime-
        // packing) which reuses activation buffers across the graph. The persist path
        // gives every intermediate its own slot (no reuse) — for a 40-layer × N-token
        // graph on the VRAM-tight 35B that thrashes WDDM paging (N=512) or OOMs (N>=1024).
        // MTP verify (n_logits == N) keeps the persist+capture fast-replay reuse.
        // MRoPE calls are prefill-only; keep them off the persist cache too.
        // The single-layer MTP DRAFT (num_layers == 1) reuses this kernel, but its
        // tiny folded-lm_head graph HANGS on CUDA-graph capture REPLAY (deadlocks the
        // stream on the 3rd invocation — the first true replay). Force the draft onto
        // the non-persist path (fresh graph + gallocr lifetime-packing, no capture):
        // still ONE fused graph per draft (vs the op-by-op block), just rebuilt each
        // call — for a 1-layer graph the build is cheap. The trunk verify
        // (num_layers > 1) keeps its persist+capture fast replay.
        // TP always takes the non-persist path: the plan executes after this
        // call returns, so the context is parked in g_q35v_tp instead, and a
        // prefill-sized graph never repeats its exact shape anyway.
        // num_layers == 1 is the MTP draft block. It was pinned to the non-persist
        // path because its captured graph used to deadlock the stream on the third
        // replay; TS_Q35_MTP_DRAFT_PERSIST=1 re-tests that on the current ggml,
        // because rebuilding a graph per draft call costs ~3.7 ms of the 6.2 ms a
        // draft step takes.
        static const bool mtp_draft_persist = []{
            const char* e = std::getenv("TS_Q35_MTP_DRAFT_PERSIST");
            return e != nullptr && e[0] == '1';
        }();
        const bool fv_persist = fv_persist_cfg && (n_logits >= N) && !use_mrope
            && (num_layers > 1 || mtp_draft_persist) && !tp_mode && !any_cpu_moe;

        const std::size_t convStateBytes = static_cast<std::size_t>(convDim) * conv_dim * sizeof(float);
        const std::size_t deltaStateBytes = static_cast<std::size_t>(head_k_dim) * head_v_dim * num_v_heads * sizeof(float);
        constexpr int kVerifyKvStride = 256;
        // Persist mode pads the attention window to a fixed stride so one cached graph
        // serves every start_pos in that stride (the mask masks the unused tail).
        // Pad the attended window up to the flash-attention KV stride on BOTH
        // paths. The mask already -infs ki >= totalSeqLen so the padded columns
        // contribute nothing; what the padding buys is a graph shape constant
        // across prefill chunks and the stride ggml-cuda's GQA-optimised flash
        // kernel requires. (Previously only the persist path padded.)
        const int window = std::min(cache_size,
            ((totalSeqLen + kVerifyKvStride - 1) / kVerifyKvStride) * kVerifyKvStride);
        const void* sig = layers[0].attn_norm_w;

        // Device-resident GDN state: when the C# caller points each recurrent layer's
        // conv_state_in and conv_state_out at the SAME buffer (the decode's device-
        // resident _fdConvScratch slot + _deltaStateTensor), the verify reads/writes the
        // GDN state IN-PLACE on the device (cacheable COMPUTE binding) instead of
        // uploading + downloading it every call (~60 MB delta + 3 MB conv). The state
        // persists across verify/plain steps exactly like the captured decode's; the C#
        // snapshots it (drain) only before a draft-verify for rollback.
        // Device-resident in-place GDN update requires the OWN-SLOT persist path
        // (like TSGgml_Qwen35ModelDecode, which uses the identical in-place cpy). On
        // the NON-persist (prefill / rollback re-forward) path activations are packed
        // by a lifetime gallocr, and the in-place cpy into a bound external state
        // buffer aliases/faults (ggml_cuda_cpy invalid-argument). Prefill is rare and
        // cheap, so it keeps host mode; the frequent MTP verify + plain steps
        // (fv_persist == true) get resident. The C# caller mirrors this gate (it only
        // points conv_in == conv_out for all-row / single-token calls), so both sides
        // agree on which calls are resident.
        bool resident_state = false;
        if (fv_persist)
        {
            for (int l = 0; l < num_layers; l++)
            {
                if (layers[l].is_recurrent != 0)
                {
                    resident_state = (layers[l].conv_state_in == layers[l].conv_state_out) && layers[l].conv_state_in != nullptr;
                    break;
                }
            }
        }

        // Deferring the state download needs a graph whose output tensors outlive
        // the call, which is exactly the persist path; resident mode has nothing to
        // defer (it updates the state in place).
        const bool defer_state = fv_snapshots_cfg && defer_state_download != 0 && fv_persist && !resident_state;
        const int n_snap = (defer_state && state_snapshots > 1 && state_snapshots <= N)
            ? state_snapshots : 1;
        // What the caller has to do next, and getting it wrong silently decodes from
        // a stale recurrent state:
        //    0 -> deferred with no snapshots; commit slot -1 (the post-window state)
        //    1 -> downloaded, as it always used to be; nothing to do
        //   >1 -> deferred with N snapshots; commit slot (N-1-accepted)
        if (state_snapshots_used != nullptr)
            *state_snapshots_used = defer_state ? (n_snap > 1 ? n_snap : 0) : 1;

        // ===== Persist reuse fast-path: upload the per-call inputs + replay =====
        if (fv_persist)
        {
            for (auto& c : g_q35vc)
            {
                if (!c.valid || c.n != N || c.window != window || c.sig != sig ||
                    c.num_layers != num_layers || c.out_vocab != vocab_size ||
                    c.n_logits != n_logits ||
                    c.has_normed != (normed_out != nullptr) ||
                    c.capture_count != cap_count ||
                    c.n_snapshots != n_snap ||
                    c.deferred_state != defer_state)
                    continue;
                // llama.cpp pattern (llama-context.cpp): before re-setting the inputs of
                // a REUSED graph we must fully synchronize, else we overwrite input
                // tensors the previous (async) graph_compute is still reading -> the
                // pipeline accumulates across reuses and faults. host_read_barrier()
                // only syncs conditionally, so force a full backend sync here.
                tsg::sync_backend(g_backend);
                ggml_backend_tensor_set(c.hidden_t, hidden_data, 0, static_cast<std::size_t>(H) * N * sizeof(float));
                std::vector<std::int32_t> pv(N);
                std::vector<std::int64_t> kv(N);
                for (int i = 0; i < N; i++) { pv[i] = start_pos + i; kv[i] = start_pos + i; }
                ggml_backend_tensor_set(c.pos_t, pv.data(), 0, static_cast<std::size_t>(N) * sizeof(std::int32_t));
                ggml_backend_tensor_set(c.kv_index, kv.data(), 0, static_cast<std::size_t>(N) * sizeof(std::int64_t));
                std::vector<ggml_fp16_t> mk;
                fill_verify_causal_mask(mk, window, N, start_pos, totalSeqLen);
                ggml_backend_tensor_set(c.mask_t, mk.data(), 0, mk.size() * sizeof(ggml_fp16_t));
                // Host mode uploads the per-call GDN state; resident keeps it device-
                // resident (cacheable, in-place), so no upload/download here.
                if (!resident_state && device_state_current == 0)
                {
                    int gi = 0;
                    for (int l = 0; l < num_layers; l++)
                    {
                        if (layers[l].is_recurrent == 0) continue;
                        ggml_backend_tensor_set(c.conv_in[gi], layers[l].conv_state_in, 0, convStateBytes);
                        ggml_backend_tensor_set(c.delta_in[gi], layers[l].delta_state_in, 0, deltaStateBytes);
                        gi++;
                    }
                }
                // device_state_current: the live state slices already hold what the
                // caller wants (a previous verify's snapshot was committed into them
                // on the device), so the ~300 MB round trip this upload and the
                // matching download used to cost per step is simply not paid.
                // Profiled (a plain compute unless TS_GGML_NODE_PROFILE is set): this
                // is the hot path - every warm speculative verify replays here - and
                // it was the one graph the node profiler could not see.
                if (tsg::graph_compute_profiled(g_backend, c.graph, "qwen35 verify replay") != GGML_STATUS_SUCCESS) { c.reset(); break; }
                if (!resident_state && !c.deferred_state)
                {
                    int gi = 0;
                    for (int l = 0; l < num_layers; l++)
                    {
                        if (layers[l].is_recurrent == 0) continue;
                        finalize_compute_with_download(c.conv_out[gi], layers[l].conv_state_out, convStateBytes);
                        finalize_compute_with_download(c.delta_out[gi], layers[l].delta_state_out, deltaStateBytes);
                        gi++;
                    }
                }
                // Deferred: the state stays on the device until the caller commits it.
                g_q35vc_last_snap = c.deferred_state ? &c : nullptr;
                g_q35vc_snap_conv_bytes = convStateBytes;
                g_q35vc_snap_delta_bytes = deltaStateBytes;
                if (normed_out != nullptr && c.normed_out != nullptr)
                    finalize_compute_with_download(c.normed_out, normed_out, static_cast<std::size_t>(H) * N * sizeof(float));
                for (int ci = 0; ci < c.capture_count; ci++)
                {
                    if (c.capture_out[ci] == nullptr) continue;
                    finalize_compute_with_download(c.capture_out[ci],
                        capture_data + static_cast<std::size_t>(ci) * H * N,
                        static_cast<std::size_t>(H) * N * sizeof(float));
                }
                finalize_compute_with_download(c.logits_out, logits_data, static_cast<std::size_t>(vocab_size) * n_logits * sizeof(float));
                host_read_barrier();
                c.lru = ++g_q35vc_clock;
                clear_last_error();
                return 1;
            }
        }

        // ===== Build a fresh graph. Persist: raw ctx + alloc_ctx_tensors (each tensor
        // its OWN slot = stable addresses, required for reuse/capture). Non-persist:
        // pooled ctx + gallocr lifetime-packing. =====
        ggml_context* ctx = nullptr;
        PooledContextHandle context;
        if (fv_persist)
        {
            ggml_init_params ip = { 32 * 1024 * 1024, nullptr, /*no_alloc=*/true };
            ctx = ggml_init(ip);
            if (ctx == nullptr) { set_last_error("Qwen3.5 model verify: failed to init persist ctx."); return 0; }
        }
        else
        {
            if (!context.init(32 * 1024 * 1024)) { set_last_error("Qwen3.5 model verify: failed to acquire ggml context."); return 0; }
            ctx = context.value;
        }

        ggml_tensor* hidden_t = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, H, N);
        // MRoPE positions are axis-concatenated [T0..Tn-1, H.., W.., E..] = 4N ints
        // (ggml_rope_multi asserts pos->ne[0] == 4 * a->ne[2]).
        ggml_tensor* pos_tensor = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, use_mrope ? 4 * N : N);
        ggml_tensor* lm_head_t = ggml_new_tensor_2d(ctx, static_cast<ggml_type>(lm_head_type), lm_head_ne0, lm_head_ne1);
        ggml_tensor* final_norm_t = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, H);

        ggml_tensor* kv_index = ggml_new_tensor_1d(ctx, GGML_TYPE_I64, N);
        ggml_tensor* attn_mask = ggml_new_tensor_4d(ctx, GGML_TYPE_F16, window, N, 1, 1);
        const bool uses_dynamic_kv_index =
            !(g_backend_type == BACKEND_TYPE_METAL && !fv_persist);
        ggml_set_input(hidden_t);
        ggml_set_input(pos_tensor);
        if (uses_dynamic_kv_index)
            ggml_set_input(kv_index);
        ggml_set_input(attn_mask);
        std::vector<std::int64_t> kv_index_data(N);
        for (int i = 0; i < N; i++) kv_index_data[i] = start_pos + i;
        std::vector<ggml_fp16_t> attn_mask_data;
        fill_verify_causal_mask(attn_mask_data, window, N, start_pos, totalSeqLen);

        struct LayerTensors {
            ggml_tensor* attn_norm_w;
            ggml_tensor* qkv_w; ggml_tensor* k_w; ggml_tensor* v_w;
            ggml_tensor* q_norm_w; ggml_tensor* k_norm_w; ggml_tensor* o_w;
            ggml_tensor* k_cache_base; ggml_tensor* v_cache_base;
            ggml_tensor* k_cpy; ggml_tensor* v_cpy;
            std::vector<ggml_fp16_t> mask_data;
            ggml_tensor* mask_t;
            // gdn
            ggml_tensor* gdn_qkv_w; ggml_tensor* gdn_gate_w;
            ggml_tensor* ssm_beta_w; ggml_tensor* ssm_alpha_w;
            ggml_tensor* conv1d_w; ggml_tensor* ssm_dt_w; ggml_tensor* ssm_a_w;
            ggml_tensor* ssm_norm_w; ggml_tensor* ssm_out_w;
            ggml_tensor* conv_state_in; ggml_tensor* delta_state_in;
            ggml_tensor* conv_state_out; ggml_tensor* delta_state_out;
            // ffn
            ggml_tensor* post_attn_norm_w; ggml_tensor* gu_w; ggml_tensor* down_w;
            ggml_tensor* ffn_gate_w; ggml_tensor* ffn_up_w;
            ggml_tensor* conv_snaps = nullptr;   // [convDim * conv_dim, n_snap]
            ggml_tensor* delta_snaps = nullptr;  // [D, n_snap] (a view of the GDN output)
            std::vector<ggml_tensor*> conv_snap_slots;   // per slot, shaped like conv_state_in
            std::vector<ggml_tensor*> delta_snap_slots;  // per slot, shaped like delta_state_in
            ggml_tensor* gate_inp_w; ggml_tensor* gate_exps; ggml_tensor* up_exps; ggml_tensor* down_exps;
            ggml_tensor* shexp_gate_w; ggml_tensor* shexp_up_w; ggml_tensor* shexp_down_w; ggml_tensor* shexp_gate_inp_w;
            ggml_tensor* psc[TSQ35_SC_COUNT];
        };
        std::vector<LayerTensors> lt(num_layers);

        // Host-mode GDN state slices live in the shared device state buffer (see
        // g_q35v_state_buf): per recurrent layer a delta in + delta out slice and a
        // conv in + conv out slice. Bound at tensor-creation time so neither the
        // per-entry gallocr nor the reuse gallocr ever carries the ~300 MB of state.
        // The state is NOT updated in place — the graph reads *_state_in (uploaded
        // each call) and writes *_state_out (downloaded each call), matching the
        // original separate-in/out host semantics. (In-place on an input that is
        // ALSO re-uploaded each call breaks ggml-cuda's CUDA-graph capture on the
        // persist replay path — the resident path can do in-place only because it
        // never re-uploads.)
        std::size_t state_align = 256;
        std::size_t state_stride = 0;
        std::uint8_t* state_base = nullptr;
        std::size_t delta_slice_bytes = 0;
        std::size_t conv_slice_bytes = 0;
        if (!resident_state)
        {
            int gdn_count = 0;
            for (int l = 0; l < num_layers; l++)
                if (layers[l].is_recurrent != 0) gdn_count++;
            if (gdn_count > 0)
            {
                ggml_backend_buffer_type_t buft = ggml_backend_get_default_buffer_type(g_backend);
                if (buft != nullptr)
                    state_align = std::max<std::size_t>(state_align, ggml_backend_buft_get_alignment(buft));
                auto align_up = [&](std::size_t v) { return (v + state_align - 1) / state_align * state_align; };
                delta_slice_bytes = align_up(deltaStateBytes);
                conv_slice_bytes = align_up(convStateBytes);
                state_stride = 2 * delta_slice_bytes + 2 * conv_slice_bytes;
                if (!ensure_q35v_state_buf(static_cast<std::size_t>(gdn_count) * state_stride))
                {
                    set_last_error("Qwen3.5 model verify: failed to allocate the shared GDN state buffer.");
                    if (fv_persist) ggml_free(ctx);
                    return 0;
                }
                state_base = static_cast<std::uint8_t*>(ggml_backend_buffer_get_base(g_q35v_state_buf));
            }
        }

        int state_slot = 0;
        for (int l = 0; l < num_layers; l++)
        {
            const TSGgmlQwen35LayerDesc& d = layers[l];
            LayerTensors& t = lt[l];
            t.attn_norm_w = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, H);
            t.post_attn_norm_w = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, H);
            if (d.is_recurrent == 0)
            {
                t.qkv_w = ggml_new_tensor_2d(ctx, static_cast<ggml_type>(d.qkv_type), d.qkv_ne0, d.qkv_ne1);
                if (d.separate_qkv != 0)
                {
                    t.k_w = ggml_new_tensor_2d(ctx, static_cast<ggml_type>(d.k_type), d.k_ne0, d.k_ne1);
                    t.v_w = ggml_new_tensor_2d(ctx, static_cast<ggml_type>(d.v_type), d.v_ne0, d.v_ne1);
                }
                t.q_norm_w = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, head_dim);
                t.k_norm_w = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, head_dim);
                t.o_w = ggml_new_tensor_2d(ctx, static_cast<ggml_type>(d.o_type), d.o_ne0, d.o_ne1);
                t.k_cache_base = ggml_new_tensor_3d(ctx, kvType, head_dim, cache_size, num_kv_heads);
                t.v_cache_base = ggml_new_tensor_3d(ctx, kvType, head_dim, cache_size, num_kv_heads);
            }
            else
            {
                t.gdn_qkv_w = ggml_new_tensor_2d(ctx, static_cast<ggml_type>(d.gdn_qkv_type), d.gdn_qkv_ne0, d.gdn_qkv_ne1);
                // Packed in-projection (the TP shards): one [hidden, Q|K|V|Z|beta|alpha]
                // weight instead of four separate ones; gdn_gate_w == null marks it,
                // and the z/beta/alpha weights are neither created nor bound.
                if (d.gdn_gate_w != nullptr)
                {
                    t.gdn_gate_w = ggml_new_tensor_2d(ctx, static_cast<ggml_type>(d.gdn_gate_type), d.gdn_gate_ne0, d.gdn_gate_ne1);
                    t.ssm_beta_w = ggml_new_tensor_2d(ctx, static_cast<ggml_type>(d.ssm_beta_type), d.ssm_beta_ne0, d.ssm_beta_ne1);
                    t.ssm_alpha_w = ggml_new_tensor_2d(ctx, static_cast<ggml_type>(d.ssm_alpha_type), d.ssm_alpha_ne0, d.ssm_alpha_ne1);
                }
                t.conv1d_w = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, conv_kernel, conv_dim);
                t.ssm_dt_w = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, num_v_heads);
                t.ssm_a_w = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, num_v_heads);
                t.ssm_norm_w = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, head_v_dim);
                t.ssm_out_w = ggml_new_tensor_2d(ctx, static_cast<ggml_type>(d.ssm_out_type), d.ssm_out_ne0, d.ssm_out_ne1);
                t.conv_state_in = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, convDim, conv_dim);
                t.delta_state_in = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, head_k_dim, head_v_dim, num_v_heads);
                if (resident_state)
                {
                    // Resident: conv/delta state lives in a device-resident cacheable
                    // COMPUTE buffer (bound below); the post-window state is written
                    // IN-PLACE to conv_state_in / delta_state_in (no separate out tensor,
                    // no per-call upload/download). Saves the ~60 MB delta out alloc too.
                    t.conv_state_out = nullptr;
                    t.delta_state_out = nullptr;
                }
                else
                {
                    // Host mode: bind the four state tensors (delta in/out + conv
                    // in/out) into per-slot slices of the shared state buffer. The
                    // graph reads *_state_in (uploaded each call) and writes
                    // *_state_out (downloaded each call) — no in-place, so the
                    // persist replay's CUDA-graph capture stays valid.
                    std::uint8_t* slice = state_base + static_cast<std::size_t>(state_slot) * state_stride;
                    t.delta_state_out = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, head_k_dim, head_v_dim, num_v_heads);
                    t.conv_state_out = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, convDim, conv_dim);
                    if (ggml_backend_tensor_alloc(g_q35v_state_buf, t.delta_state_in, slice) != GGML_STATUS_SUCCESS ||
                        ggml_backend_tensor_alloc(g_q35v_state_buf, t.delta_state_out, slice + delta_slice_bytes) != GGML_STATUS_SUCCESS ||
                        ggml_backend_tensor_alloc(g_q35v_state_buf, t.conv_state_in, slice + 2 * delta_slice_bytes) != GGML_STATUS_SUCCESS ||
                        ggml_backend_tensor_alloc(g_q35v_state_buf, t.conv_state_out, slice + 2 * delta_slice_bytes + conv_slice_bytes) != GGML_STATUS_SUCCESS)
                    {
                        set_last_error("Qwen3.5 model verify: failed to bind GDN state slices.");
                        if (fv_persist) ggml_free(ctx);
                        return 0;
                    }
                    // Preserved per-call inputs (uploaded each call). The flag is
                    // metadata only here — the tensors already have a buffer, so
                    // gallocr skips them — but it keeps parity with the original
                    // set_input semantics the CUDA-graph replay was validated under.
                    ggml_set_input(t.conv_state_in);
                    ggml_set_input(t.delta_state_in);
                    state_slot++;
                }
            }
            if (d.is_moe == 0)
            {
                if (d.gu_w != nullptr)
                {
                    t.gu_w = ggml_new_tensor_2d(ctx, static_cast<ggml_type>(d.gu_type), d.gu_ne0, d.gu_ne1);
                }
                else
                {
                    t.ffn_gate_w = ggml_new_tensor_2d(ctx, static_cast<ggml_type>(d.ffn_gate_type), d.ffn_gate_ne0, d.ffn_gate_ne1);
                    t.ffn_up_w   = ggml_new_tensor_2d(ctx, static_cast<ggml_type>(d.ffn_up_type),   d.ffn_up_ne0,   d.ffn_up_ne1);
                }
                t.down_w = ggml_new_tensor_2d(ctx, static_cast<ggml_type>(d.down_type), d.down_ne0, d.down_ne1);
            }
            else
            {
                t.gate_inp_w = ggml_new_tensor_2d(ctx, static_cast<ggml_type>(d.gate_inp_type), d.gate_inp_ne0, d.gate_inp_ne1);
                // MoE CPU offload: leave the routed-expert tensors null so the
                // bind pass never asks for a device copy of them. That omission
                // IS the VRAM saving - the host reads the same GGUF mmap bytes.
                if (d.cpu_moe == 0)
                {
                    // Under TP the stacked expert tensors hold only this rank's
                    // whole-expert slice; the router dims stay global.
                    t.gate_exps = ggml_new_tensor_3d(ctx, static_cast<ggml_type>(d.gate_exps_type), hidden_size, expert_ff, stacked_experts);
                    t.up_exps = ggml_new_tensor_3d(ctx, static_cast<ggml_type>(d.up_exps_type), hidden_size, expert_ff, stacked_experts);
                    t.down_exps = ggml_new_tensor_3d(ctx, static_cast<ggml_type>(d.down_exps_type), expert_ff, hidden_size, stacked_experts);
                }
                t.shexp_gate_w = ggml_new_tensor_2d(ctx, static_cast<ggml_type>(d.shexp_gate_type), d.shexp_gate_ne0, d.shexp_gate_ne1);
                t.shexp_up_w = ggml_new_tensor_2d(ctx, static_cast<ggml_type>(d.shexp_up_type), d.shexp_up_ne0, d.shexp_up_ne1);
                t.shexp_down_w = ggml_new_tensor_2d(ctx, static_cast<ggml_type>(d.shexp_down_type), d.shexp_down_ne0, d.shexp_down_ne1);
                t.shexp_gate_inp_w = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, hidden_size);
            }
        }

        // Expert-parallel routing constants (TP MoE only) — the verify sibling of
        // qwen35_model_decode_impl's ep_lut/ep_mask. Whole experts partition
        // across ranks, but the top-k runs on the GLOBAL router probabilities
        // (identical on every rank), so the selected ids must be confined to
        // this rank's slice before feeding mul_mat_id:
        //   ep_lut  I32 [1, num_experts]: global id -> local id (foreign -> 0)
        //   ep_mask F32 [1, num_experts]: 1 for owned experts, else 0
        // The lookups run over the flattened [num_used*N] selection so a plain
        // 2D get_rows serves every token (ggml_get_rows has no ne2 broadcast).
        ggml_tensor* ep_lut = nullptr;
        ggml_tensor* ep_mask = nullptr;
        std::vector<std::int32_t> ep_lut_data;
        std::vector<float> ep_mask_data;
        if (tp_mode && num_experts > 0)
        {
            // Only layers whose experts are RESIDENT need the LUT. With
            // --cpu-moe every MoE layer takes the global-routing branch,
            // so the LUT would end up in no graph node at all -- and the
            // upload below would then fault on its missing buffer.
            bool any_moe = false;
            for (int l = 0; l < num_layers; l++)
                if (layers[l].is_moe != 0 && layers[l].cpu_moe == 0) { any_moe = true; break; }
            if (any_moe)
            {
                ep_lut = ggml_new_tensor_2d(ctx, GGML_TYPE_I32, 1, num_experts);
                ep_mask = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 1, num_experts);
                ggml_set_input(ep_lut);
                ggml_set_input(ep_mask);
                const int ep_first = tsg::tp_global_rank() * stacked_experts;
                const int ep_last = ep_first + stacked_experts;
                ep_lut_data.resize(static_cast<std::size_t>(num_experts));
                ep_mask_data.resize(static_cast<std::size_t>(num_experts));
                for (int e = 0; e < num_experts; e++)
                {
                    const bool own = e >= ep_first && e < ep_last;
                    ep_lut_data[static_cast<std::size_t>(e)] = own ? e - ep_first : 0;
                    ep_mask_data[static_cast<std::size_t>(e)] = own ? 1.0f : 0.0f;
                }
            }
        }

        // Tensor-parallel cut points: the row-parallel projections (attention
        // o-proj, GDN ssm_out, dense FFN down) and the MoE routed+shared sum,
        // whose per-rank outputs the collective reduces between segments.
        std::vector<ggml_tensor*> tp_partial;
        std::vector<ggml_tensor*> tp_boundary;
        if (tp_mode)
        {
            tp_partial.reserve(static_cast<std::size_t>(num_layers) * 2);
            tp_boundary.reserve(static_cast<std::size_t>(num_layers) * 2);
        }

        // MoE CPU offload: filled per offloaded layer while the graph is built
        // (its boundary tensors are expanded in place, inside the layer loop),
        // then turned into segment cut points once the node order is fixed.
        std::vector<tsg::HostMoeSegment> host_moe;

        // --- build the chained graph over N tokens ---
        // The graph is created BEFORE the layer loop and each layer's KV writes +
        // GDN state writes are expanded AS THE LAYER IS BUILT (llama.cpp's build
        // order). Expanding them after the whole loop (the previous layout) placed
        // every state-write cpy at the END of the node order, so the gallocr had
        // to keep each GDN layer's gated_delta_net output AND conv concat input
        // alive across the entire remainder of the graph — for a 2048-token, 48-
        // recurrent-layer prefill that inflated the reuse gallocr to 7.9 GB.
        // Expanded per layer, the allocator recycles them layer-by-layer.
        // Per-head set_rows adds ~2*num_kv_heads nodes per attention layer, so size
        // the graph generously to avoid GGML_ASSERT(n_nodes < size).
        const std::size_t graph_size = static_cast<std::size_t>(num_layers) * (260 + 2 * num_kv_heads) + 1024;
        ggml_cgraph* graph = ggml_new_graph_custom(ctx, graph_size, false);
        ggml_tensor* hidden = hidden_t;
        // llama.cpp feeds strided head/token views straight to the norm, CPY and
        // gated-delta-net kernels rather than materializing them, and ggml-cuda
        // consumes the strides just as ggml-metal does. Doing the same here removes
        // ~170 CONT nodes and their copies from a 64-layer graph. Metal was already
        // on this path; CUDA/Vulkan joined it after the outputs were checked
        // byte-identical. TS_Q35_VERIFY_STRIDED_VIEWS=0 restores the materializing
        // layout if a backend ever disagrees.
        static const bool strided_views_cfg = []{
            const char* e = std::getenv("TS_Q35_VERIFY_STRIDED_VIEWS");
            return e == nullptr || e[0] != '0';
        }();
        const bool metal_strided_views = strided_views_cfg
            && (g_backend_type == BACKEND_TYPE_METAL || g_backend_type == BACKEND_TYPE_CUDA);
        std::vector<ggml_tensor*> capture_out(cap_count, nullptr);
        for (int l = 0; l < num_layers; l++)
        {
            const TSGgmlQwen35LayerDesc& d = layers[l];
            LayerTensors& t = lt[l];

            // The residual ENTERING layer l, copied out before the block runs -
            // exactly res->t_layer_inp[l] in llama.cpp's DFlash capture.
            for (int ci = 0; ci < cap_count; ci++)
            {
                if (capture_layers[ci] != l) continue;
                ggml_tensor* dst = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, H, N);
                capture_out[ci] = ggml_cpy(ctx, hidden, dst);
                ggml_set_output(capture_out[ci]);
            }

            ggml_tensor* normed = ggml_mul(ctx, ggml_rms_norm(ctx, hidden, eps), t.attn_norm_w); // [H, N]
            ggml_tensor* block_out;

            if (d.is_recurrent == 0)
            {
                // ===== Full attention (prefill-style causal flash attention) =====
                ggml_tensor* qg_part; ggml_tensor* k_raw; ggml_tensor* v_raw;
                if (d.separate_qkv != 0)
                {
                    qg_part = q35_scaled(ctx, ggml_mul_mat(ctx, t.qkv_w, normed), q35_psc(ctx, t, d, TSQ35_SC_QKV));  // [qFullDim, N]
                    k_raw = q35_scaled(ctx, ggml_mul_mat(ctx, t.k_w, normed), q35_psc(ctx, t, d, TSQ35_SC_K));      // [kDim, N]
                    v_raw = q35_scaled(ctx, ggml_mul_mat(ctx, t.v_w, normed), q35_psc(ctx, t, d, TSQ35_SC_V));
                }
                else
                {
                    ggml_tensor* qkv_out = q35_scaled(ctx, ggml_mul_mat(ctx, t.qkv_w, normed), q35_psc(ctx, t, d, TSQ35_SC_QKV)); // [qFullDim+2kDim, N]
                    qg_part = ggml_view_2d(ctx, qkv_out, qFullDim, N, qkv_out->nb[1], 0);
                    k_raw = ggml_view_2d(ctx, qkv_out, kDim, N, qkv_out->nb[1], static_cast<std::size_t>(qFullDim) * sizeof(float));
                    v_raw = ggml_view_2d(ctx, qkv_out, kDim, N, qkv_out->nb[1], static_cast<std::size_t>(qFullDim + kDim) * sizeof(float));
                }

                // Separate Q/K/V projections are already contiguous matmul outputs.
                // A fused-QKV row slice remains strided and still needs materializing.
                ggml_tensor* qg_dense =
                    metal_strided_views && d.separate_qkv != 0 ? qg_part : ggml_cont(ctx, qg_part);
                ggml_tensor* k_dense =
                    metal_strided_views && d.separate_qkv != 0 ? k_raw : ggml_cont(ctx, k_raw);
                ggml_tensor* v_dense =
                    metal_strided_views && d.separate_qkv != 0 ? v_raw : ggml_cont(ctx, v_raw);
                ggml_tensor* qg_4d = ggml_reshape_4d(ctx, qg_dense, head_dim, 2, num_heads, N);
                ggml_tensor* q_view = ggml_view_3d(ctx, qg_4d, head_dim, num_heads, N, qg_4d->nb[2], qg_4d->nb[3], 0);
                ggml_tensor* gate_view = ggml_view_3d(ctx, qg_4d, head_dim, num_heads, N, qg_4d->nb[2], qg_4d->nb[3], qg_4d->nb[1]);
                ggml_tensor* gate_cont = ggml_cont(ctx, gate_view); // [head_dim, num_heads, N]
                ggml_tensor* k_3d_raw = ggml_reshape_3d(ctx, k_dense, head_dim, num_kv_heads, N);
                ggml_tensor* v_3d_raw = ggml_reshape_3d(ctx, v_dense, head_dim, num_kv_heads, N);

                ggml_tensor* q_normed;
                ggml_tensor* k_normed;
                if (metal_strided_views)
                {
                    // Q is interleaved with its gate, but every head row is dense.
                    // Metal consumes the explicit strides, as llama.cpp's Qwen35
                    // graph does, and the norm outputs themselves are contiguous.
                    q_normed = ggml_mul(ctx, ggml_rms_norm(ctx, q_view, eps), t.q_norm_w);
                    k_normed = ggml_mul(ctx, ggml_rms_norm(ctx, k_3d_raw, eps), t.k_norm_w);
                }
                else
                {
                    ggml_tensor* q_cont = ggml_cont(ctx, q_view); // [head_dim, num_heads, N]
                    ggml_tensor* q_norm_in = ggml_reshape_2d(ctx, q_cont, head_dim, num_heads * N);
                    ggml_tensor* k_norm_in = ggml_reshape_2d(ctx, k_3d_raw, head_dim, num_kv_heads * N);
                    q_normed = ggml_mul(ctx, ggml_rms_norm(ctx, q_norm_in, eps), t.q_norm_w);
                    k_normed = ggml_mul(ctx, ggml_rms_norm(ctx, k_norm_in, eps), t.k_norm_w);
                }

                ggml_tensor* q_4d = ggml_reshape_4d(ctx, q_normed, head_dim, num_heads, N, 1);
                ggml_tensor* k_4d = ggml_reshape_4d(ctx, k_normed, head_dim, num_kv_heads, N, 1);
                ggml_tensor* q_roped;
                ggml_tensor* k_roped;
                if (use_mrope)
                {
                    // Interleaved MRoPE (Qwen3-VL LLM): per-pair modality assignment
                    // comes from the GGUF's mrope sections; positions per token are
                    // the (T,H,W,E) axes uploaded into pos_tensor [4N].
                    int sections_local[4] = { mrope_sections[0], mrope_sections[1], mrope_sections[2], mrope_sections[3] };
                    q_roped = ggml_rope_multi(ctx, q_4d, pos_tensor, nullptr, rope_n_dims, sections_local, GGML_ROPE_TYPE_IMROPE, 0, rope_base, rope_freq_scale, 0, 1, 0, 0);
                    k_roped = ggml_rope_multi(ctx, k_4d, pos_tensor, nullptr, rope_n_dims, sections_local, GGML_ROPE_TYPE_IMROPE, 0, rope_base, rope_freq_scale, 0, 1, 0, 0);
                }
                else
                {
                    q_roped = ggml_rope_ext(ctx, q_4d, pos_tensor, nullptr, rope_n_dims, rope_mode, 0, rope_base, rope_freq_scale, 0, 1, 0, 0);
                    k_roped = ggml_rope_ext(ctx, k_4d, pos_tensor, nullptr, rope_n_dims, rope_mode, 0, rope_base, rope_freq_scale, 0, 1, 0, 0);
                }

                ggml_tensor* q_attn = ggml_permute(ctx, q_roped, 0, 2, 1, 3); // [head_dim, N, num_heads]
                ggml_tensor* k_fresh = ggml_reshape_3d(ctx, ggml_cont(ctx, ggml_permute(ctx, k_roped, 0, 2, 1, 3)), head_dim, N, num_kv_heads);
                ggml_tensor* v_3d_pre = ggml_reshape_4d(ctx, v_3d_raw, head_dim, num_kv_heads, N, 1);
                ggml_tensor* v_fresh = ggml_reshape_3d(ctx, ggml_cont(ctx, ggml_permute(ctx, v_3d_pre, 0, 2, 1, 3)), head_dim, N, num_kv_heads);

                if (g_backend_type == BACKEND_TYPE_METAL && !fv_persist)
                {
                    // A prefill graph is one-shot and already keyed by start_pos,
                    // so bake the contiguous destination offset into a cache view.
                    // This is the same cpy-based linear KV write used by llama.cpp
                    // and avoids Metal's fragile multi-dimensional set_rows path.
                    const std::size_t kv_offset =
                        static_cast<std::size_t>(start_pos) * t.k_cache_base->nb[1];
                    ggml_tensor* k_dst = ggml_view_3d(ctx, t.k_cache_base,
                        head_dim, N, num_kv_heads,
                        t.k_cache_base->nb[1], t.k_cache_base->nb[2], kv_offset);
                    ggml_tensor* v_dst = ggml_view_3d(ctx, t.v_cache_base,
                        head_dim, N, num_kv_heads,
                        t.v_cache_base->nb[1], t.v_cache_base->nb[2], kv_offset);
                    ggml_build_forward_expand(graph, ggml_cpy(ctx, k_fresh, k_dst));
                    ggml_build_forward_expand(graph, ggml_cpy(ctx, v_fresh, v_dst));
                }
                else
                {
                    // Dynamic KV write: a 2D ggml_set_rows PER HEAD — dst
                    // [head_dim, cache_size], src [head_dim, N], idx [N].
                    // The position input keeps persistent CUDA/Vulkan graph
                    // topology constant within an attention-window stride.
                    for (int h = 0; h < num_kv_heads; h++)
                    {
                        ggml_tensor* k_dst_h = ggml_view_2d(ctx, t.k_cache_base, head_dim, cache_size,
                            t.k_cache_base->nb[1], static_cast<std::size_t>(h) * t.k_cache_base->nb[2]);
                        ggml_tensor* v_dst_h = ggml_view_2d(ctx, t.v_cache_base, head_dim, cache_size,
                            t.v_cache_base->nb[1], static_cast<std::size_t>(h) * t.v_cache_base->nb[2]);
                        ggml_tensor* k_src_h = ggml_view_2d(ctx, k_fresh, head_dim, N,
                            k_fresh->nb[1], static_cast<std::size_t>(h) * k_fresh->nb[2]);
                        ggml_tensor* v_src_h = ggml_view_2d(ctx, v_fresh, head_dim, N,
                            v_fresh->nb[1], static_cast<std::size_t>(h) * v_fresh->nb[2]);
                        ggml_build_forward_expand(graph, ggml_set_rows(ctx, k_dst_h, k_src_h, kv_index));
                        ggml_build_forward_expand(graph, ggml_set_rows(ctx, v_dst_h, v_src_h, kv_index));
                    }
                }

                // Attend over the fixed window [0, window) (now holds the N fresh rows);
                // the shared causal mask zeroes valid keys and -inf's the rest.
                ggml_tensor* k_full = view_kv_cache_window(ctx, t.k_cache_base, head_dim, cache_size, num_kv_heads, 0, window, kv_cache_type, N);
                ggml_tensor* v_full = view_kv_cache_window(ctx, t.v_cache_base, head_dim, cache_size, num_kv_heads, 0, window, kv_cache_type, N);
                if (k_full == nullptr || v_full == nullptr)
                {
                    set_last_error("Qwen3.5 model verify: failed to build KV cache views.");
                    return 0;
                }

                ggml_tensor* attn_flat;
                ggml_tensor* fa = ggml_flash_attn_ext(ctx, q_attn, k_full, v_full, attn_mask, attn_scale, 0.0f, 0.0f);
                ggml_flash_attn_ext_set_prec(fa, GGML_PREC_F32);
                if (backend_supports_op(fa))
                {
                    attn_flat = ggml_reshape_2d(ctx, fa, qDim, N);
                }
                else
                {
                    ggml_tensor* q_attn_cont = ggml_cont(ctx, q_attn);
                    ggml_tensor* scores = ggml_mul_mat(ctx, k_full, q_attn_cont);
                    ggml_mul_mat_set_prec(scores, GGML_PREC_F32);
                    ggml_tensor* probs = ggml_soft_max_ext(ctx, scores, attn_mask, attn_scale, 0.0f);
                    ggml_tensor* v_perm = ggml_cont(ctx, ggml_permute(ctx, v_full, 1, 0, 2, 3));
                    ggml_tensor* attn_out = ggml_mul_mat(ctx, v_perm, probs);
                    ggml_tensor* attn_perm = ggml_cont(ctx, ggml_permute(ctx, attn_out, 0, 2, 1, 3));
                    attn_flat = ggml_reshape_2d(ctx, attn_perm, qDim, N);
                }

                ggml_tensor* gate_flat = ggml_reshape_2d(ctx, gate_cont, qDim, N);
                ggml_tensor* attn_gated = ggml_mul(ctx, attn_flat, ggml_sigmoid(ctx, gate_flat));
                block_out = q35_scaled(ctx, ggml_mul_mat(ctx, t.o_w, attn_gated), q35_psc(ctx, t, d, TSQ35_SC_O)); // [H, N]
                if (tp_mode) { tp_partial.push_back(block_out); tp_boundary.push_back(block_out); }
            }
            else
            {
                // ===== Gated Delta Net over N tokens (one ggml_gated_delta_net, K=N) =====
                ggml_tensor* qkv_mixed;
                ggml_tensor* z_all;
                ggml_tensor* beta_all;
                ggml_tensor* alpha_all;
                if (t.gdn_gate_w == nullptr)
                {
                    // Packed in-projection: one matmul, row-sliced per token into
                    // [Q|K|V | Z | beta | alpha]. The slices are strided views, so
                    // materialize each — the unary/reshape consumers below need
                    // contiguous inputs on CUDA/Vulkan.
                    ggml_tensor* packed = ggml_mul_mat(ctx, t.gdn_qkv_w, normed); // [packed_dim, N]
                    qkv_mixed = ggml_cont(ctx, ggml_view_2d(ctx, packed, conv_dim, N, packed->nb[1], 0));
                    z_all = ggml_cont(ctx, ggml_view_2d(ctx, packed, value_dim, N, packed->nb[1],
                        static_cast<std::size_t>(conv_dim) * sizeof(float)));
                    beta_all = ggml_sigmoid(ctx, ggml_cont(ctx, ggml_view_2d(ctx, packed, num_v_heads, N, packed->nb[1],
                        static_cast<std::size_t>(conv_dim + value_dim) * sizeof(float))));
                    alpha_all = ggml_cont(ctx, ggml_view_2d(ctx, packed, num_v_heads, N, packed->nb[1],
                        static_cast<std::size_t>(conv_dim + value_dim + num_v_heads) * sizeof(float)));
                }
                else
                {
                    qkv_mixed = q35_scaled(ctx, ggml_mul_mat(ctx, t.gdn_qkv_w, normed), q35_psc(ctx, t, d, TSQ35_SC_GDN_QKV));  // [conv_dim, N]
                    z_all = q35_scaled(ctx, ggml_mul_mat(ctx, t.gdn_gate_w, normed), q35_psc(ctx, t, d, TSQ35_SC_GDN_GATE));     // [value_dim, N]
                    beta_all = ggml_sigmoid(ctx, q35_scaled(ctx, ggml_mul_mat(ctx, t.ssm_beta_w, normed), q35_psc(ctx, t, d, TSQ35_SC_BETA))); // [num_v_heads, N]
                    alpha_all = q35_scaled(ctx, ggml_mul_mat(ctx, t.ssm_alpha_w, normed), q35_psc(ctx, t, d, TSQ35_SC_ALPHA)); // [num_v_heads, N]
                }
                ggml_tensor* g_all = ggml_softplus(ctx, ggml_add(ctx, alpha_all, t.ssm_dt_w));
                g_all = ggml_mul(ctx, g_all, t.ssm_a_w); // [num_v_heads, N]

                // conv over the N new timesteps prepended with the conv ring state.
                // Concat straight from the non-contiguous transpose view — ggml_concat
                // materializes a contiguous result for ssm_conv, so the separate
                // transpose-cont copy (cpy_scalar_transpose) is redundant (llama.cpp's
                // build_conv_state concats the transposed view the same way).
                ggml_tensor* conv_input = ggml_concat(ctx, t.conv_state_in, ggml_transpose(ctx, qkv_mixed), 0); // [convDim+N, conv_dim]
                ggml_tensor* conv_out = ggml_silu(ctx, ggml_ssm_conv(ctx, conv_input, t.conv1d_w)); // [conv_dim, N]
                // new conv state = the last convDim timesteps (rows [N, N+convDim)).
                ggml_tensor* new_conv_view = ggml_view_2d(ctx, conv_input, convDim, conv_dim,
                    conv_input->nb[1], static_cast<std::size_t>(N) * conv_input->nb[0]);
                ggml_tensor* new_conv =
                    metal_strided_views ? new_conv_view : ggml_cont(ctx, new_conv_view);
                // Resident: write the post-window conv state IN-PLACE to conv_state_in
                // (the device-resident buffer); host mode: to the shared-state-buffer
                // out slice.
                t.conv_state_out = ggml_cpy(ctx, new_conv, resident_state ? t.conv_state_in : t.conv_state_out);

                if (n_snap > 1)
                {
                    // The conv state after row r is simply rows [r+1, r+1+convDim) of
                    // conv_input - the same window new_conv takes at r = N-1 - so every
                    // snapshot is already in a tensor the graph built. Slot s is s
                    // tokens back from the end, matching the GDN op's slot order.
                    t.conv_snaps = ggml_new_tensor_2d(ctx, GGML_TYPE_F32,
                        static_cast<std::int64_t>(convDim) * conv_dim, n_snap);
                    ggml_set_output(t.conv_snaps);
                    for (int ssl = 0; ssl < n_snap; ssl++)
                    {
                        ggml_tensor* src = ggml_view_2d(ctx, conv_input, convDim, conv_dim,
                            conv_input->nb[1], static_cast<std::size_t>(N - ssl) * conv_input->nb[0]);
                        ggml_tensor* dst = ggml_view_2d(ctx, t.conv_snaps, convDim, conv_dim,
                            static_cast<std::size_t>(convDim) * sizeof(float),
                            static_cast<std::size_t>(ssl) * convDim * conv_dim * sizeof(float));
                        ggml_build_forward_expand(graph, ggml_cpy(ctx, src, dst));
                        t.conv_snap_slots.push_back(dst);
                    }
                }

                // l2-norm over head_k_dim. q/k keep num_k_heads heads: the fused
                // gated_delta_net kernel broadcasts each v-head h to k-head (h % num_k_heads)
                // internally (kernel iq1 = h_idx % neqk1, neqk1 = q->ne[1]), so pre-tiling
                // q/k up to num_v_heads via concat+cont (~2% of prefill, 4 concats/layer)
                // is redundant — llama.cpp's fused GDN path passes the un-tiled q/k too.
                ggml_tensor* q4;
                ggml_tensor* k4;
                ggml_tensor* v4;
                if (metal_strided_views)
                {
                    // conv_out is [Q|K|V, token]. Preserve that token stride in
                    // 4D views instead of copying all three slices. Metal's GDN
                    // kernel advances tokens with nb[2], matching llama.cpp.
                    const std::size_t token_stride = conv_out->nb[1];
                    const std::size_t sequence_stride = token_stride * static_cast<std::size_t>(N);
                    ggml_tensor* q_view = ggml_view_4d(ctx, conv_out,
                        head_k_dim, num_k_heads, N, 1,
                        ggml_row_size(conv_out->type, head_k_dim),
                        token_stride, sequence_stride, 0);
                    ggml_tensor* k_view = ggml_view_4d(ctx, conv_out,
                        head_k_dim, num_k_heads, N, 1,
                        ggml_row_size(conv_out->type, head_k_dim),
                        token_stride, sequence_stride,
                        ggml_row_size(conv_out->type, key_dim));
                    v4 = ggml_view_4d(ctx, conv_out,
                        head_v_dim, num_v_heads, N, 1,
                        ggml_row_size(conv_out->type, head_v_dim),
                        token_stride, sequence_stride,
                        ggml_row_size(conv_out->type, 2 * key_dim));
                    q4 = ggml_l2_norm(ctx, q_view, eps);
                    k4 = ggml_l2_norm(ctx, k_view, eps);
                }
                else
                {
                    ggml_tensor* q_part = ggml_cont(ctx, ggml_view_2d(ctx, conv_out,
                        key_dim, N, conv_out->nb[1], 0));
                    ggml_tensor* k_part = ggml_cont(ctx, ggml_view_2d(ctx, conv_out,
                        key_dim, N, conv_out->nb[1], static_cast<std::size_t>(key_dim) * sizeof(float)));
                    ggml_tensor* v_part = ggml_cont(ctx, ggml_view_2d(ctx, conv_out,
                        value_dim, N, conv_out->nb[1], static_cast<std::size_t>(2 * key_dim) * sizeof(float)));
                    ggml_tensor* q_hn = ggml_l2_norm(ctx,
                        ggml_reshape_2d(ctx, q_part, head_k_dim, num_k_heads * N), eps);
                    ggml_tensor* k_hn = ggml_l2_norm(ctx,
                        ggml_reshape_2d(ctx, k_part, head_k_dim, num_k_heads * N), eps);
                    q4 = ggml_reshape_4d(ctx, q_hn, head_k_dim, num_k_heads, N, 1);
                    k4 = ggml_reshape_4d(ctx, k_hn, head_k_dim, num_k_heads, N, 1);
                    v4 = ggml_reshape_4d(ctx, v_part, head_v_dim, num_v_heads, N, 1);
                }
                ggml_tensor* g4 = ggml_reshape_4d(ctx,
                    metal_strided_views ? g_all : ggml_cont(ctx, g_all),
                    1, num_v_heads, N, 1);
                ggml_tensor* beta4 = ggml_reshape_4d(ctx,
                    metal_strided_views ? beta_all : ggml_cont(ctx, beta_all),
                    1, num_v_heads, N, 1);
                ggml_tensor* state4 = ggml_reshape_4d(ctx, t.delta_state_in, head_k_dim, head_v_dim, num_v_heads, 1);

                // K=1: the op recurs over all N tokens internally and emits the per-
                // token outputs (rows [0,N)) + ONLY the FINAL state snapshot (we roll
                // back via host snapshot/re-forward, not the per-prefix snapshots, so
                // requesting K=N would waste ~19 MB/layer of VRAM on unused states).
                // n_snap slots, most-recent first: slot 0 is the post-window state
                // (all the plain path needs) and slot s the state s tokens earlier,
                // which is what a partial accept rolls back to.
                ggml_tensor* gdn = ggml_gated_delta_net(ctx, q4, k4, v4, g4, beta4, state4, n_snap);
                // Per-token outputs occupy the first N rows ([S_v*H] each).
                ggml_tensor* gdn_out = ggml_view_2d(ctx, gdn, value_dim, N, ggml_row_size(gdn->type, value_dim), 0);
                // Final state snapshot (slot 0, most-recent) at offset N * (S_v*H).
                ggml_tensor* new_state = ggml_view_4d(ctx, gdn, head_k_dim, head_v_dim, num_v_heads, 1,
                    ggml_row_size(gdn->type, head_k_dim),
                    ggml_row_size(gdn->type, head_k_dim * head_v_dim),
                    ggml_row_size(gdn->type, head_k_dim * head_v_dim * num_v_heads),
                    ggml_row_size(gdn->type, value_dim) * static_cast<std::size_t>(N));
                // Resident: write the post-window delta state IN-PLACE to delta_state_in
                // (state4 aliases it). Host mode: write to the separate delta_state_out
                // slice (downloaded after compute) — NOT in-place, so the persist
                // replay's captured CUDA graph stays valid across re-uploads.
                t.delta_state_out = ggml_cpy(ctx, new_state, resident_state ? state4 : t.delta_state_out);
                if (n_snap > 1)
                {
                    const std::int64_t d_elems =
                        static_cast<std::int64_t>(head_k_dim) * head_v_dim * num_v_heads;
                    t.delta_snaps = ggml_view_2d(ctx, gdn, d_elems, n_snap,
                        ggml_row_size(gdn->type, d_elems),
                        ggml_row_size(gdn->type, value_dim) * static_cast<std::size_t>(N));
                    ggml_set_output(t.delta_snaps);
                    for (int ssl = 0; ssl < n_snap; ssl++)
                    {
                        // Shaped like delta_state_in so committing a slot is one
                        // device-to-device tensor copy rather than a host round trip.
                        t.delta_snap_slots.push_back(ggml_view_3d(ctx, gdn,
                            head_k_dim, head_v_dim, num_v_heads,
                            ggml_row_size(gdn->type, head_k_dim),
                            ggml_row_size(gdn->type, head_k_dim * head_v_dim),
                            ggml_row_size(gdn->type, value_dim) * static_cast<std::size_t>(N)
                                + ggml_row_size(gdn->type, d_elems) * static_cast<std::size_t>(ssl)));
                    }
                }

                // gated RMSNorm with z, per token: rms_norm(out) * ssm_norm * silu(z).
                ggml_tensor* out_2d = ggml_reshape_2d(ctx,
                    metal_strided_views ? gdn_out : ggml_cont(ctx, gdn_out),
                    head_v_dim, num_v_heads * N);
                ggml_tensor* out_n = ggml_mul(ctx, ggml_rms_norm(ctx, out_2d, eps), t.ssm_norm_w);
                ggml_tensor* out_n_3d = ggml_reshape_3d(ctx, out_n, head_v_dim, num_v_heads, N);
                ggml_tensor* z_3d = ggml_reshape_3d(ctx, z_all, head_v_dim, num_v_heads, N);
                ggml_tensor* gated = ggml_mul(ctx, out_n_3d, ggml_silu(ctx, z_3d));
                ggml_tensor* gated_flat = ggml_reshape_2d(ctx, gated, value_dim, N);
                block_out = q35_scaled(ctx, ggml_mul_mat(ctx, t.ssm_out_w, gated_flat), q35_psc(ctx, t, d, TSQ35_SC_SSM_OUT)); // [H, N]
                if (tp_mode) { tp_partial.push_back(block_out); tp_boundary.push_back(block_out); }

                ggml_set_output(t.conv_state_out);
                ggml_set_output(t.delta_state_out);
                ggml_build_forward_expand(graph, t.conv_state_out);
                ggml_build_forward_expand(graph, t.delta_state_out);
            }

            ggml_tensor* residual1 = ggml_add(ctx, hidden, block_out); // [H, N]

            // ===== FFN =====
            ggml_tensor* ffn_normed = ggml_mul(ctx, ggml_rms_norm(ctx, residual1, eps), t.post_attn_norm_w); // [H, N]
            ggml_tensor* ffn_out;
            if (d.is_moe == 0)
            {
                ggml_tensor* act;
                if (t.gu_w != nullptr)
                {
                    act = ggml_swiglu(ctx, q35_scaled(ctx, ggml_mul_mat(ctx, t.gu_w, ffn_normed), q35_psc(ctx, t, d, TSQ35_SC_GU)));
                }
                else
                {
                    // Unfused mixed-quant gate/up: two matmuls, same arithmetic.
                    ggml_tensor* g = q35_scaled(ctx, ggml_mul_mat(ctx, t.ffn_gate_w, ffn_normed), q35_psc(ctx, t, d, TSQ35_SC_FFN_GATE));
                    ggml_tensor* u = q35_scaled(ctx, ggml_mul_mat(ctx, t.ffn_up_w, ffn_normed), q35_psc(ctx, t, d, TSQ35_SC_FFN_UP));
                    act = ggml_mul(ctx, ggml_silu(ctx, g), u);
                }
                ffn_out = q35_scaled(ctx, ggml_mul_mat(ctx, t.down_w, act), q35_psc(ctx, t, d, TSQ35_SC_DOWN)); // [H, N]
                if (tp_mode) { tp_partial.push_back(ffn_out); tp_boundary.push_back(ffn_out); }
            }
            else
            {
                ggml_tensor* router_logits = ggml_mul_mat(ctx, t.gate_inp_w, ffn_normed); // [num_experts, N]
                ggml_tensor* sel_ids;
                ggml_tensor* w_final;
                // An offloaded layer runs once on the host over the unsharded
                // expert stack, so it takes the global-routing branch even under
                // TP: global ids, unmasked weights (see HostMoeSegment).
                if (ep_lut == nullptr || d.cpu_moe != 0)
                {
                    // ggml-cuda's fusable gating shape (see build_topk_moe_routing).
                    // Expanded here, as a unit: this kernel builds and expands layer
                    // by layer, so emitting it now keeps the chain contiguous instead
                    // of letting the expert matmuls pull in its halves separately.
                    tsg::MoeTopKRouting r = tsg::build_topk_moe_routing(
                        ctx, router_logits, num_experts, num_experts_used, N, norm_topk != 0);
                    ggml_tensor* w_2d = r.weights_2d;
                    w_final = r.weights_3d;
                    if (expert_weights_scale != 1.0f)
                    {
                        w_2d = ggml_scale(ctx, w_2d, expert_weights_scale);
                        w_final = ggml_reshape_3d(ctx, w_2d, 1, num_experts_used, N);
                    }
                    sel_ids = r.ids;
                    ggml_build_forward_expand(graph, w_final);
                }
                else
                {
                    // Expert-parallel TP. ggml's batched mul_mat_id paths (the
                    // CUDA mm_ids_helper and the generic fallback) require the
                    // ids WITHIN one token to be DISTINCT — their (expert, token)
                    // compaction collapses duplicates and the row bookkeeping
                    // shifts, silently scrambling expert outputs. Mapping every
                    // foreign route to a filler id therefore cannot work at N > 1.
                    // Instead each rank runs its own top-k over MASK-ZEROED probs,
                    // which selects num_used DISTINCT experts it owns; any expert
                    // in the global top-k that this rank owns has a prob >= every
                    // non-selected prob, so it is guaranteed to be in that set.
                    // The gating weight keeps only global top-k members — an
                    // expert is one iff its prob >= the k-th largest global prob
                    // (w_min, read through the DESC argsort, since CUDA's TOP_K
                    // output is unsorted) — and zero for the rest, so the ranks'
                    // partial sums add up to exactly the single-GPU MoE.
                    // get_rows only ever sees whole (buffer-aligned) tensors as its
                    // id input: ggml-vulkan asserts zero buffer-offset remainders
                    // for GET_ROWS, so an element-offset view of the argsort output
                    // cannot be gathered directly. Gather ALL sorted probs once and
                    // slice the k-th column as a view — the elementwise consumers
                    // below carry offsets in their push constants.
                    // Its own softmax/reshape: this branch masks and re-sorts,
                    // so it can never match the fused gating shape above.
                    ggml_tensor* probs = ggml_soft_max(ctx, router_logits);
                    ggml_tensor* probs_r = ggml_reshape_3d(ctx, probs, 1, num_experts, N);
                    ggml_tensor* sorted_g = ggml_argsort(ctx, probs, GGML_SORT_ORDER_DESC); // I32 [num_experts, N]
                    ggml_tensor* sorted_p = ggml_reshape_2d(ctx,
                        ggml_get_rows(ctx, probs_r, sorted_g), num_experts, N);            // probs, descending per token
                    ggml_tensor* w_min = ggml_view_2d(ctx, sorted_p, 1, N,
                        sorted_p->nb[1], static_cast<std::size_t>(num_experts_used - 1) * sizeof(float));

                    // Zero foreign experts, then float every owned entry above
                    // zero: a softmax prob can underflow to exactly 0.0f, and a
                    // zero-for-zero tie in top_k could pick a FOREIGN index —
                    // whose LUT filler id would reintroduce the duplicate-ids
                    // hazard this scheme exists to avoid.
                    ggml_tensor* mask_col = ggml_reshape_2d(ctx, ep_mask, num_experts, 1);
                    ggml_tensor* probs_m = ggml_add(ctx, ggml_mul(ctx, probs, mask_col),
                        ggml_scale(ctx, mask_col, 1e-30f));
                    ggml_tensor* sel_r = ggml_top_k(ctx, probs_m, num_experts_used);       // owned, distinct, [num_used, N]
                    ggml_tensor* p_r = ggml_reshape_2d(ctx,
                        ggml_get_rows(ctx, probs_r, sel_r), num_experts_used, N);          // unmasked probs of the picks

                    // keep = [p >= w_min] = 1 - step(w_min - p); w = p * keep.
                    ggml_tensor* wmp = ggml_add(ctx, ggml_scale(ctx, p_r, -1.0f), w_min);
                    ggml_tensor* st = ggml_step(ctx, wmp);
                    ggml_tensor* w_r = ggml_sub(ctx, p_r, ggml_mul(ctx, p_r, st));
                    if (norm_topk != 0)
                    {
                        // Normalizer over the GLOBAL top-k (identical on every rank):
                        // the first num_used rows of the sorted probs.
                        ggml_tensor* w_g = ggml_cont(ctx, ggml_view_2d(ctx, sorted_p,
                            num_experts_used, N, sorted_p->nb[1], 0));
                        ggml_tensor* z_sum = ggml_sum_rows(ctx, w_g);
                        w_r = ggml_div(ctx, w_r, z_sum);
                    }
                    if (expert_weights_scale != 1.0f)
                        w_r = ggml_scale(ctx, w_r, expert_weights_scale);
                    w_final = ggml_reshape_3d(ctx, w_r, 1, num_experts_used, N);

                    // Local ids for this rank's stacked expert slice.
                    ggml_tensor* sel_flat = ggml_reshape_1d(ctx, sel_r, num_experts_used * N);
                    ggml_tensor* local_ids = ggml_get_rows(ctx, ep_lut, sel_flat);         // I32 [1, num_used*N]
                    sel_ids = ggml_reshape_2d(ctx, local_ids, num_experts_used, N);
                }

                ggml_tensor* moe_out_2d;
                if (d.cpu_moe != 0)
                {
                    // ---- MoE CPU offload seam (see tsg::HostMoeSegment) ----
                    // Prefill hands the host all N tokens at once, so the
                    // offloaded side is a real GEMM over the chunk rather than N
                    // matvecs. Attention, the GDN blocks, the router, the shared
                    // expert and the LM head all stay in this one fused graph.
                    tsg::HostMoeSegment hm;
                    hm.layer = l;
                    hm.moe_in = ggml_cont(ctx, ffn_normed);
                    // sel_ids is a VIEW of the argsort on the fusable path (see
                    // build_topk_moe_routing): cont BEFORE reshape, or ggml_reshape
                    // asserts on the non-contiguous source.
                    hm.sel_ids = ggml_reshape_1d(ctx, ggml_cont(ctx, sel_ids), static_cast<std::int64_t>(num_experts_used) * N);
                    hm.weights = ggml_cont(ctx, ggml_reshape_1d(ctx, w_final, static_cast<std::int64_t>(num_experts_used) * N));
                    ggml_set_output(hm.moe_in);
                    ggml_set_output(hm.sel_ids);
                    ggml_set_output(hm.weights);

                    // Written by the host between segments; flagged BOTH input
                    // and output so ggml-alloc pre-allocates it and never
                    // recycles the block behind our back.
                    ggml_tensor* moe_host_out = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, H, N);
                    ggml_set_input(moe_host_out);
                    ggml_set_output(moe_host_out);

                    hm.moe_out = moe_host_out;
                    hm.gate_data = d.gate_exps;   hm.gate_type = d.gate_exps_type;
                    hm.gate_ne0 = H;              hm.gate_ne1 = expert_ff;   hm.gate_bytes = d.gate_exps_bytes;
                    hm.up_data = d.up_exps;       hm.up_type = d.up_exps_type;
                    hm.up_ne0 = H;                hm.up_ne1 = expert_ff;     hm.up_bytes = d.up_exps_bytes;
                    hm.down_data = d.down_exps;   hm.down_type = d.down_exps_type;
                    hm.down_ne0 = expert_ff;      hm.down_ne1 = H;           hm.down_bytes = d.down_exps_bytes;
                    hm.activation = 0;            // silu(gate) * up
                    // The offloaded stack is never sharded (the descriptor points
                    // at the whole GGUF tensor even under TP), so the host sees
                    // the global expert count and the global ids above.
                    hm.num_experts = num_experts;
                    hm.n_used = num_experts_used;
                    hm.n_ff = expert_ff;
                    hm.seq_len = N;
                    hm.hidden = H;
                    // ffn_out below sums this with the Megatron-split shared
                    // expert, and that sum IS the layer's AllReduce point.
                    hm.tp_reduced = tp_mode ? 1 : 0;

                    // Expand the seam's boundary tensors HERE, right after this
                    // layer's KV/state writes: the cut has to land where the
                    // accelerator actually reaches the router, and a trailing
                    // expand would drag the whole model's forward chain ahead of
                    // layer 0's cut instead.
                    ggml_build_forward_expand(graph, hm.moe_in);
                    ggml_build_forward_expand(graph, hm.sel_ids);
                    ggml_build_forward_expand(graph, hm.weights);
                    host_moe.push_back(hm);
                    moe_out_2d = moe_host_out;
                }
                else
                {
                    ggml_tensor* moe_in_3d = ggml_reshape_3d(ctx, ffn_normed, H, 1, N);
                    ggml_tensor* g_exp = ggml_mul_mat_id(ctx, t.gate_exps, moe_in_3d, sel_ids); // [expert_ff, num_used, N]
                    ggml_tensor* u_exp = ggml_mul_mat_id(ctx, t.up_exps, moe_in_3d, sel_ids);
                    ggml_tensor* act = ggml_mul(ctx, ggml_silu(ctx, g_exp), u_exp);
                    ggml_tensor* moe_down = ggml_mul_mat_id(ctx, t.down_exps, act, sel_ids);    // [H, num_used, N]
                    ggml_tensor* weighted = ggml_mul(ctx, moe_down, w_final);
                    ggml_tensor* moe_out = ggml_cont(ctx, ggml_view_3d(ctx, weighted, H, 1, N, weighted->nb[1], weighted->nb[2], 0));
                    for (int u = 1; u < num_experts_used; ++u)
                    {
                        ggml_tensor* vu = ggml_view_3d(ctx, weighted, H, 1, N, weighted->nb[1], weighted->nb[2], static_cast<std::size_t>(u) * weighted->nb[1]);
                        moe_out = ggml_add(ctx, moe_out, vu);
                    }
                    moe_out_2d = ggml_reshape_2d(ctx, moe_out, H, N);
                }

                ggml_tensor* sh_g = ggml_mul_mat(ctx, t.shexp_gate_w, ffn_normed); // [shared_ff, N]
                ggml_tensor* sh_u = ggml_mul_mat(ctx, t.shexp_up_w, ffn_normed);
                ggml_tensor* sh_act = ggml_mul(ctx, ggml_silu(ctx, sh_g), sh_u);
                ggml_tensor* sh_down = ggml_mul_mat(ctx, t.shexp_down_w, sh_act); // [H, N]
                ggml_tensor* sh_gate = ggml_sigmoid(ctx, ggml_mul_mat(ctx, ggml_reshape_2d(ctx, t.shexp_gate_inp_w, H, 1), ffn_normed)); // [1, N]
                ggml_tensor* sh_out = ggml_mul(ctx, sh_down, sh_gate);
                // Both the routed sum (this rank's experts only) and the
                // Megatron-split shared expert are partials, so their sum is
                // the layer's single reduction point.
                ffn_out = ggml_add(ctx, moe_out_2d, sh_out);
                if (tp_mode) { tp_partial.push_back(ffn_out); tp_boundary.push_back(ffn_out); }
            }

            hidden = ggml_add(ctx, residual1, ffn_out); // [H, N]
        }

        // Final norm over all N rows -> the MTP head's input AND the LM head.
        ggml_tensor* fn = ggml_mul(ctx, ggml_rms_norm(ctx, hidden, eps), final_norm_t); // [H, N]
        ggml_tensor* normed_out_t = nullptr;
        ggml_tensor* normed_cpy = nullptr;
        if (normed_out != nullptr)
        {
            normed_out_t = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, H, N);
            normed_cpy = ggml_cpy(ctx, fn, normed_out_t);
            ggml_set_output(normed_cpy);
        }
        // Prefill (n_logits < N) folds the lm_head over only the LAST n_logits
        // columns of the post-norm hidden — the trailing token(s) we sample from.
        ggml_tensor* fn_head_in = fn;                                  // [H, N]
        if (n_logits < N)
        {
            ggml_tensor* fn_last = ggml_view_2d(ctx, fn, H, n_logits, fn->nb[1],
                static_cast<std::size_t>(N - n_logits) * fn->nb[1]);
            fn_head_in = ggml_cont(ctx, fn_last);                      // [H, n_logits]
        }
        ggml_tensor* logits = ggml_mul_mat(ctx, lm_head_t, fn_head_in); // [vocab, n_logits]
        ggml_tensor* logits_out_t = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, vocab_size, n_logits);
        ggml_tensor* logits_cpy = ggml_cpy(ctx, logits, logits_out_t);
        ggml_set_output(logits_cpy);
        // A capture whose layer never ran would leave an unwritten output; refuse
        // rather than hand the drafter uninitialised residuals.
        for (int ci = 0; ci < cap_count; ci++)
        {
            if (capture_out[ci] == nullptr)
            {
                set_last_error("Qwen3.5 model verify: a DFlash capture layer is outside the trunk.");
                if (fv_persist) ggml_free(ctx);
                return 0;
            }
        }

        if (normed_cpy != nullptr)
            ggml_build_forward_expand(graph, normed_cpy);
        for (int ci = 0; ci < cap_count; ci++)
            ggml_build_forward_expand(graph, capture_out[ci]);
        ggml_build_forward_expand(graph, logits_cpy);

        // Turn the recorded seams into node cut points (see
        // host_moe_build_segment_ends). It fails when the builder and the
        // expander disagree, which would silently feed the host stale values.
        std::vector<int> host_moe_seg_end;
        if (!host_moe_build_segment_ends(graph, host_moe, host_moe_seg_end, kQ35VerifyKernel))
        {
            if (fv_persist) ggml_free(ctx);
            return 0;
        }

        // --- bind tensors ---
        ggml_backend_dev_t dev = ggml_backend_get_device(g_backend);
        struct HostBinding { ggml_tensor* tensor; void* data; std::size_t bytes; };
        std::vector<HostBinding> upload_list;
        std::vector<BufferHandle> ephemeral_bufs;
        auto bind_or_mark = [&](ggml_tensor* tgt, void* data, std::size_t bytes, bool cacheable,
                                enum ggml_backend_buffer_usage usage = GGML_BACKEND_BUFFER_USAGE_WEIGHTS) {
            if (tgt == nullptr || data == nullptr) return;
            if (cacheable && bytes >= 4096)
            {
                ggml_backend_buffer_t buf = nullptr; void* addr = nullptr; bool needs_upload = false;
                if (try_bind_cached_tensor(g_backend, dev, tgt, data, bytes, needs_upload, usage))
                { if (needs_upload) upload_list.push_back({tgt, data, bytes}); return; }
            }
            if (bytes >= 4096)
            {
                ggml_backend_buffer_t buf = nullptr;
                if (try_get_host_ptr_buffer(g_backend, dev, data, bytes, cacheable, buf))
                { if (!cacheable) ephemeral_bufs.emplace_back(buf);
                  if (ggml_backend_tensor_alloc(buf, tgt, data) == GGML_STATUS_SUCCESS) return; }
            }
            upload_list.push_back({tgt, data, bytes});
        };

        for (int l = 0; l < num_layers; l++)
        {
            const TSGgmlQwen35LayerDesc& d = layers[l];
            LayerTensors& t = lt[l];
            bind_or_mark(t.attn_norm_w, d.attn_norm_w, static_cast<std::size_t>(H) * sizeof(float), true);
            for (int s = 0; s < TSQ35_SC_COUNT; s++)
                if (t.psc[s] != nullptr)
                    bind_or_mark(t.psc[s], static_cast<float*>(d.proj_scales) + s, sizeof(float), true);
            bind_or_mark(t.post_attn_norm_w, d.post_attn_norm_w, static_cast<std::size_t>(H) * sizeof(float), true);
            if (d.is_moe == 0)
            {
                if (t.gu_w != nullptr)
                {
                    bind_or_mark(t.gu_w, d.gu_w, static_cast<std::size_t>(d.gu_bytes), true);
                }
                else
                {
                    bind_or_mark(t.ffn_gate_w, d.ffn_gate_w, static_cast<std::size_t>(d.ffn_gate_bytes), true);
                    bind_or_mark(t.ffn_up_w,   d.ffn_up_w,   static_cast<std::size_t>(d.ffn_up_bytes),   true);
                }
                bind_or_mark(t.down_w, d.down_w, static_cast<std::size_t>(d.down_bytes), true);
            }
            else
            {
                bind_or_mark(t.gate_inp_w, d.gate_inp_w, static_cast<std::size_t>(d.gate_inp_bytes), true);
                if (t.gate_exps != nullptr)
                {
                    bind_or_mark(t.gate_exps, d.gate_exps, static_cast<std::size_t>(d.gate_exps_bytes), true);
                    bind_or_mark(t.up_exps, d.up_exps, static_cast<std::size_t>(d.up_exps_bytes), true);
                    bind_or_mark(t.down_exps, d.down_exps, static_cast<std::size_t>(d.down_exps_bytes), true);
                }
                bind_or_mark(t.shexp_gate_w, d.shexp_gate_w, static_cast<std::size_t>(d.shexp_gate_bytes), true);
                bind_or_mark(t.shexp_up_w, d.shexp_up_w, static_cast<std::size_t>(d.shexp_up_bytes), true);
                bind_or_mark(t.shexp_down_w, d.shexp_down_w, static_cast<std::size_t>(d.shexp_down_bytes), true);
                bind_or_mark(t.shexp_gate_inp_w, d.shexp_gate_inp_w, static_cast<std::size_t>(H) * sizeof(float), true);
            }
            if (d.is_recurrent == 0)
            {
                bind_or_mark(t.qkv_w, d.qkv_w, static_cast<std::size_t>(d.qkv_bytes), true);
                if (d.separate_qkv != 0)
                {
                    bind_or_mark(t.k_w, d.k_w, static_cast<std::size_t>(d.k_bytes), true);
                    bind_or_mark(t.v_w, d.v_w, static_cast<std::size_t>(d.v_bytes), true);
                }
                bind_or_mark(t.o_w, d.o_w, static_cast<std::size_t>(d.o_bytes), true);
                bind_or_mark(t.q_norm_w, d.q_norm_w, static_cast<std::size_t>(head_dim) * sizeof(float), true);
                bind_or_mark(t.k_norm_w, d.k_norm_w, static_cast<std::size_t>(head_dim) * sizeof(float), true);
                bind_or_mark(t.k_cache_base, d.k_cache, kv_cache_bytes(num_kv_heads, cache_size, head_dim, kv_cache_type), true, GGML_BACKEND_BUFFER_USAGE_COMPUTE);
                bind_or_mark(t.v_cache_base, d.v_cache, kv_cache_bytes(num_kv_heads, cache_size, head_dim, kv_cache_type), true, GGML_BACKEND_BUFFER_USAGE_COMPUTE);
                if (!t.mask_data.empty())
                    bind_or_mark(t.mask_t, t.mask_data.data(), t.mask_data.size() * sizeof(ggml_fp16_t), false);
            }
            else
            {
                bind_or_mark(t.gdn_qkv_w, d.gdn_qkv_w, static_cast<std::size_t>(d.gdn_qkv_bytes), true);
                bind_or_mark(t.gdn_gate_w, d.gdn_gate_w, static_cast<std::size_t>(d.gdn_gate_bytes), true);
                bind_or_mark(t.ssm_beta_w, d.ssm_beta_w, static_cast<std::size_t>(d.ssm_beta_bytes), true);
                bind_or_mark(t.ssm_alpha_w, d.ssm_alpha_w, static_cast<std::size_t>(d.ssm_alpha_bytes), true);
                bind_or_mark(t.conv1d_w, d.conv1d_w, static_cast<std::size_t>(conv_kernel) * conv_dim * sizeof(float), true);
                bind_or_mark(t.ssm_dt_w, d.ssm_dt_w, static_cast<std::size_t>(num_v_heads) * sizeof(float), true);
                bind_or_mark(t.ssm_a_w, d.ssm_a_w, static_cast<std::size_t>(num_v_heads) * sizeof(float), true);
                bind_or_mark(t.ssm_norm_w, d.ssm_norm_w, static_cast<std::size_t>(head_v_dim) * sizeof(float), true);
                bind_or_mark(t.ssm_out_w, d.ssm_out_w, static_cast<std::size_t>(d.ssm_out_bytes), true);
                if (resident_state)
                {
                    // Device-resident GDN state: bind cacheable COMPUTE (keyed by the
                    // decode's _fdConvScratch / _deltaStateTensor host ptrs) so the
                    // buffer persists across calls and is updated in-place. The cacheable
                    // path uploads only when the host key is invalidated (the C# seed);
                    // subsequent calls cache-hit (no upload).
                    bind_or_mark(t.conv_state_in, d.conv_state_in, convStateBytes, true, GGML_BACKEND_BUFFER_USAGE_COMPUTE);
                    bind_or_mark(t.delta_state_in, d.delta_state_in, deltaStateBytes, true, GGML_BACKEND_BUFFER_USAGE_COMPUTE);
                }
                // else: conv/delta state tensors are pre-bound into the shared
                // g_q35v_state_buf slices at creation time and uploaded each call below.
            }
        }
        bind_or_mark(lm_head_t, const_cast<void*>(lm_head_data), static_cast<std::size_t>(lm_head_bytes), true);
        bind_or_mark(final_norm_t, const_cast<void*>(final_norm_data), static_cast<std::size_t>(H) * sizeof(float), true);

        optimize_graph_for_metal(graph);

        // Persist: give every still-unbound tensor (intermediates, inputs, outputs,
        // small weights) its OWN stable slot via alloc_ctx_tensors. Own slots (no
        // lifetime packing / buffer aliasing) are REQUIRED for ggml-cuda's CUDA-graph
        // capture to replay correctly — a private lifetime-packed gallocr produced a
        // ~20x smaller buffer but the captured graph's replay read stale aliased
        // slots and the trunk output degraded to garbage after the first verify.
        // The bulk of the old per-entry VRAM was the GDN state (delta in/out + conv
        // in/out, ~300 MB for 48 recurrent layers), which now lives in the ONE shared
        // g_q35v_state_buf (pre-bound above), so each entry's own-slot buffer is only
        // its activations + I/O — small for the N<=9 verify shapes MTP decode uses.
        ggml_backend_buffer_t persist_buf = nullptr;
        if (fv_persist)
        {
            persist_buf = ggml_backend_alloc_ctx_tensors(ctx, g_backend);
            if (persist_buf == nullptr)
            {
                set_last_error("Qwen3.5 model verify: failed to allocate persist buffer.");
                ggml_free(ctx);
                return 0;
            }
            if (vram_log_enabled())
            {
                char tag[96];
                std::snprintf(tag, sizeof(tag), "q35-verify-persist(N=%d,w=%d)", N, window);
                vram_log(tag, static_cast<std::int64_t>(ggml_backend_buffer_get_size(persist_buf)));
            }
        }
        else if (!alloc_graph_reuse_gallocr(graph))
        {
            BufferHandle buffer(ggml_backend_alloc_ctx_tensors(ctx, g_backend));
            if (buffer.value == nullptr)
            {
                set_last_error("Qwen3.5 model verify: failed to allocate backend buffer.");
                return 0;
            }
            if (vram_log_enabled())
            {
                char tag[96];
                std::snprintf(tag, sizeof(tag), "q35-verify-ctx(N=%d,w=%d)", N, window);
                vram_log(tag, static_cast<std::int64_t>(ggml_backend_buffer_get_size(buffer.value)));
            }
        }

        // Every marked tensor must have come out of the allocation above with a
        // buffer. One that did not is a build bug (a tensor created in the context
        // but never wired into the graph, which the graph-driven allocator has no
        // reason to place) and ggml_backend_tensor_set would abort the PROCESS on
        // it. Decline instead: the caller falls back to the per-op path, which is
        // slow but correct, and the reason is reportable.
        for (std::size_t ui = 0; ui < upload_list.size(); ui++)
        {
            auto& u = upload_list[ui];
            if (u.tensor->buffer != nullptr) continue;
            char msg[320];
            std::snprintf(msg, sizeof(msg),
                "Qwen3.5 model verify: upload #%zu (tensor '%s' type=%d ne=[%lld,%lld,%lld,%lld], "
                "%zu bytes) was never allocated - it is in the context but not in the graph.",
                ui, u.tensor->name, static_cast<int>(u.tensor->type),
                (long long)u.tensor->ne[0], (long long)u.tensor->ne[1],
                (long long)u.tensor->ne[2], (long long)u.tensor->ne[3], u.bytes);
            set_last_error(msg);
            if (persist_buf != nullptr) ggml_backend_buffer_free(persist_buf);
            if (fv_persist) ggml_free(ctx);
            return 0;
        }
        host_read_barrier();
        for (auto& u : upload_list)
            ggml_backend_tensor_set(u.tensor, resolve_upload_source(u.data), 0, u.bytes);
        ggml_backend_tensor_set(hidden_t, hidden_data, 0, static_cast<std::size_t>(H) * N * sizeof(float));
        if (use_mrope)
        {
            ggml_backend_tensor_set(pos_tensor, mrope_pos, 0, static_cast<std::size_t>(4) * N * sizeof(std::int32_t));
        }
        else
        {
            std::vector<std::int32_t> pos_vals(N);
            for (int i = 0; i < N; i++) pos_vals[i] = start_pos + i;
            ggml_backend_tensor_set(pos_tensor, pos_vals.data(), 0, static_cast<std::size_t>(N) * sizeof(std::int32_t));
        }
        if (uses_dynamic_kv_index)
            ggml_backend_tensor_set(kv_index, kv_index_data.data(), 0, static_cast<std::size_t>(N) * sizeof(std::int64_t));
        ggml_backend_tensor_set(attn_mask, attn_mask_data.data(), 0, attn_mask_data.size() * sizeof(ggml_fp16_t));
        if (!resident_state && device_state_current == 0)
        {
            // Host mode: upload the per-call GDN state. Resident mode skips this — the
            // state is device-resident (cacheable), seeded only on invalidation; so
            // does a caller whose last snapshot commit already left the live slices
            // correct on the device.
            for (int l = 0; l < num_layers; l++)
            {
                if (layers[l].is_recurrent == 0) continue;
                ggml_backend_tensor_set(lt[l].conv_state_in, layers[l].conv_state_in, 0, convStateBytes);
                ggml_backend_tensor_set(lt[l].delta_state_in, layers[l].delta_state_in, 0, deltaStateBytes);
            }
        }
        if (ep_lut != nullptr)
        {
            ggml_backend_tensor_set(ep_lut, ep_lut_data.data(), 0, ep_lut_data.size() * sizeof(std::int32_t));
            ggml_backend_tensor_set(ep_mask, ep_mask_data.data(), 0, ep_mask_data.size() * sizeof(float));
        }

        // Tensor-parallel mode: every input is staged; hand the caller a
        // segmented plan instead of computing. The graph, its pooled context
        // and the gallocr buffers stay alive parked in this rank's pending
        // slot until the next TP build recycles them; the driver downloads the
        // logits slice and the post-window GDN states after the final sync.
        if (tp_mode)
        {
            auto& pending = g_q35v_tp[g_active_rank];
            pending.plan.clear();
            pending.plan.graph = graph;
            pending.plan.ar_tensor = tp_partial;
            // Offloaded layers pause this graph too; tp_plan_segments merges
            // their cuts into the same schedule as the AllReduce ones.
            pending.plan.host_moe = host_moe;
            if (!tp_plan_segments(pending.plan, tp_boundary))
            {
                pending.reset();
                return 0;
            }
            pending.plan.out_tensor = logits_out_t;
            pending.plan.out_host = logits_data;
            pending.plan.out_bytes = static_cast<std::size_t>(vocab_size) * n_logits * sizeof(float);
            if (normed_out != nullptr && normed_out_t != nullptr)
                pending.plan.extra_out.push_back({ normed_cpy, normed_out,
                    static_cast<std::size_t>(H) * N * sizeof(float) });
            for (int ci = 0; ci < cap_count; ci++)
                pending.plan.extra_out.push_back({ capture_out[ci],
                    capture_data + static_cast<std::size_t>(ci) * H * N,
                    static_cast<std::size_t>(H) * N * sizeof(float) });
            for (int l = 0; l < num_layers; l++)
            {
                const TSGgmlQwen35LayerDesc& d = layers[l];
                if (d.is_recurrent == 0) continue;
                if (d.conv_state_out != nullptr)
                    pending.plan.extra_out.push_back({ lt[l].conv_state_out, d.conv_state_out, convStateBytes });
                if (d.delta_state_out != nullptr)
                    pending.plan.extra_out.push_back({ lt[l].delta_state_out, d.delta_state_out, deltaStateBytes });
            }
            pending.context = std::move(context);
            pending.ephemeral = std::move(ephemeral_bufs);
            *tp_plan_out = &pending.plan;
            clear_last_error();
            return 1;
        }

        if (vram_log_enabled())
        {
            char tag[96];
            std::snprintf(tag, sizeof(tag), "q35-verify-compute-begin(N=%d)", N);
            vram_log(tag, 0);
        }
        // On Metal, submit without waiting so the CPU can enqueue the 48 conv-state
        // and 48 delta-state downloads while the long prefill graph is running.
        // The final host_read_barrier below remains the single synchronization
        // point. Other backends retain their established synchronous path.
        // MoE CPU offload runs the graph in segments, pausing at each offloaded
        // layer for the host expert matmul. Everything else still goes out as
        // one graph submission.
        ggml_status status = GGML_STATUS_SUCCESS;
        if (!host_moe.empty())
        {
            if (!host_moe_execute_segments(graph, host_moe, host_moe_seg_end, kQ35VerifyKernel))
                status = GGML_STATUS_FAILED;
        }
        else
        {
            // On Metal, submit without waiting so the CPU can enqueue the 48
            // conv-state and 48 delta-state downloads while the long prefill
            // graph is running. The final host_read_barrier below remains the
            // single synchronization point.
            status =
                g_backend_type == BACKEND_TYPE_METAL &&
                g_async_compute_enabled.load(std::memory_order_acquire)
                    ? ggml_backend_graph_compute_async(g_backend, graph)
                    : tsg::graph_compute_profiled(g_backend, graph, "qwen35 model verify");
        }
        if (status != GGML_STATUS_SUCCESS)
        {
            if (persist_buf != nullptr) ggml_backend_buffer_free(persist_buf);
            if (fv_persist) ggml_free(ctx);
            if (host_moe.empty())
                set_last_error("Qwen3.5 model verify: graph execution failed.");
            return 0;
        }
        if (vram_log_enabled())
        {
            tsg::sync_backend(g_backend);
            char tag[96];
            std::snprintf(tag, sizeof(tag), "q35-verify-compute-end(N=%d)", N);
            vram_log(tag, 0);
        }

        // Download the post-window GDN state (per recurrent layer) + outputs. Resident
        // mode skips the state download (it stays device-resident, updated in-place);
        // so does snapshot mode, where the caller fetches exactly one slot once it
        // knows how much of the draft the sampler accepted.
        if (!resident_state && !defer_state)
        {
            for (int l = 0; l < num_layers; l++)
            {
                const TSGgmlQwen35LayerDesc& d = layers[l];
                if (d.is_recurrent == 0) continue;
                if (d.conv_state_out != nullptr)
                    finalize_compute_with_download(lt[l].conv_state_out, d.conv_state_out, convStateBytes);
                if (d.delta_state_out != nullptr)
                    finalize_compute_with_download(lt[l].delta_state_out, d.delta_state_out, deltaStateBytes);
            }
        }
        if (normed_out != nullptr && normed_out_t != nullptr)
            finalize_compute_with_download(normed_out_t, normed_out, static_cast<std::size_t>(H) * N * sizeof(float));
        for (int ci = 0; ci < cap_count; ci++)
            finalize_compute_with_download(capture_out[ci],
                capture_data + static_cast<std::size_t>(ci) * H * N,
                static_cast<std::size_t>(H) * N * sizeof(float));
        finalize_compute_with_download(logits_out_t, logits_data, static_cast<std::size_t>(vocab_size) * n_logits * sizeof(float));
        host_read_barrier();


        // Persist: keep ctx/graph/buffer alive + record tensor handles so later steps
        // of the same (N, window) shape just upload inputs + replay (capturable).
        if (fv_persist)
        {
            Q35VerifyCache* slot = nullptr;
            for (auto& c : g_q35vc) { if (!c.valid) { slot = &c; break; } }
            if (slot == nullptr)
            {
                slot = &g_q35vc[0];
                for (auto& c : g_q35vc) if (c.lru < slot->lru) slot = &c;
                slot->reset();
            }
            slot->valid = true;
            slot->n = N; slot->window = window; slot->sig = sig;
            slot->num_layers = num_layers; slot->out_vocab = vocab_size;
            slot->n_logits = n_logits;
            slot->has_normed = (normed_out != nullptr);
            slot->capture_count = cap_count;
            slot->capture_out = capture_out;
            slot->n_snapshots = n_snap;
            slot->deferred_state = defer_state;
            slot->conv_snaps.clear(); slot->delta_snaps.clear();
            slot->conv_snap_slots.clear(); slot->delta_snap_slots.clear();
            if (n_snap > 1)
            {
                for (int l = 0; l < num_layers; l++)
                {
                    if (layers[l].is_recurrent == 0) continue;
                    slot->conv_snaps.push_back(lt[l].conv_snaps);
                    slot->delta_snaps.push_back(lt[l].delta_snaps);
                    for (int ssl = 0; ssl < n_snap; ssl++)
                    {
                        slot->conv_snap_slots.push_back(lt[l].conv_snap_slots[ssl]);
                        slot->delta_snap_slots.push_back(lt[l].delta_snap_slots[ssl]);
                    }
                }
            }
            slot->ctx = ctx; slot->buffer = persist_buf; slot->graph = graph;
            slot->buffer_bytes = persist_buf != nullptr ? ggml_backend_buffer_get_size(persist_buf) : 0;
            slot->hidden_t = hidden_t; slot->pos_t = pos_tensor;
            slot->kv_index = kv_index; slot->mask_t = attn_mask;
            slot->logits_out = logits_out_t; slot->normed_out = normed_out_t;
            slot->conv_in.clear(); slot->delta_in.clear(); slot->conv_out.clear(); slot->delta_out.clear();
            for (int l = 0; l < num_layers; l++)
            {
                if (layers[l].is_recurrent == 0) continue;
                slot->conv_in.push_back(lt[l].conv_state_in);
                slot->delta_in.push_back(lt[l].delta_state_in);
                slot->conv_out.push_back(lt[l].conv_state_out);
                slot->delta_out.push_back(lt[l].delta_state_out);
            }
            slot->lru = ++g_q35vc_clock;
            g_q35vc_last_snap = defer_state ? slot : nullptr;
            g_q35vc_snap_conv_bytes = convStateBytes;
            g_q35vc_snap_delta_bytes = deltaStateBytes;
            // Bound the resident persist-graph total: evict LRU entries (never the one
            // just built) so the cache never re-overcommits VRAM across many N shapes.
            q35_verify_cache_evict_to_budget(0, slot);
        }
        else if (defer_state)
        {
            // A deferring call that did not persist cannot be committed from.
            g_q35vc_last_snap = nullptr;
        }
        clear_last_error();
        return 1;
    }

    void reset_qwen35_verify_cache()
    {
        g_q35vc_last_snap = nullptr;
        for (auto& c : g_q35vc) c.reset();
    }
}

TSG_EXPORT int TSGgml_Qwen35ModelVerify(
    const TSGgmlQwen35LayerDesc* layers, int num_layers,
    void* hidden_data, int hidden_size, int start_pos, int num_tokens,
    int num_heads, int num_kv_heads, int head_dim, int cache_size,
    int rope_n_dims, int rope_mode, int kv_cache_type,
    int conv_kernel, int head_k_dim, int head_v_dim, int num_k_heads, int num_v_heads,
    float eps, float rope_base, float rope_freq_scale,
    int num_experts, int num_experts_used, int expert_ff, int shared_ff,
    int norm_topk, float expert_weights_scale,
    void* logits_data, int vocab_size,
    const void* lm_head_data, int lm_head_type, std::int64_t lm_head_ne0, std::int64_t lm_head_ne1, std::int64_t lm_head_bytes,
    const void* final_norm_data, void* normed_out, int n_logit_rows,
    const std::int32_t* mrope_pos, const std::int32_t* mrope_sections,
    int tp_degree, void** tp_plan_out,
    float* capture_data, const int* capture_layers, int capture_count,
    int state_snapshots, int* state_snapshots_used, int device_state_current,
    int defer_state_download)
{
    try
    {
        // Arena coherence: prefill/verify writes go to the resident copies;
        // flush + retire any arena slots holding these caches/state first.
        if (layers != nullptr)
        {
            for (int l = 0; l < num_layers; l++)
            {
                tsg_q35arena::on_external_touch(layers[l].k_cache);
                tsg_q35arena::on_external_touch(layers[l].conv_state_in);
                tsg_q35arena::on_external_touch(layers[l].delta_state_in);
            }
        }
        int r = qwen35_model_verify_impl(
            layers, num_layers, hidden_data, hidden_size, start_pos, num_tokens,
            num_heads, num_kv_heads, head_dim, cache_size,
            rope_n_dims, rope_mode, kv_cache_type,
            conv_kernel, head_k_dim, head_v_dim, num_k_heads, num_v_heads,
            eps, rope_base, rope_freq_scale,
            num_experts, num_experts_used, expert_ff, shared_ff,
            norm_topk, expert_weights_scale,
            logits_data, vocab_size,
            lm_head_data, lm_head_type, lm_head_ne0, lm_head_ne1, lm_head_bytes,
            final_norm_data, normed_out, n_logit_rows, mrope_pos, mrope_sections,
            tp_degree, tp_plan_out, capture_data, capture_layers, capture_count,
            state_snapshots, state_snapshots_used, device_state_current,
            defer_state_download);
        return r;
    }
    catch (const std::exception& ex) { set_last_error(ex.what()); return 0; }
    catch (...) { set_last_error("Unknown error in Qwen3.5 model verify."); return 0; }
}

// Fetch ONE per-token recurrent-state snapshot from the verify that just ran.
//
// This is the whole point of the snapshots: a partially-rejected draft used to
// restore a pre-verify copy of the recurrent state and re-forward the accepted
// prefix through the entire trunk - a second whole-model forward, plus the state
// crossing PCIe twice - because the state after row m simply did not exist
// anywhere. It does now: the gated-delta-net op emits it, and the conv state after
// row m is a window of a tensor the graph already built. `slot` counts BACK from
// the end of the batch, so slot 0 is the post-window state and slot (N-1-accepted)
// is the one a partial accept wants.
//
// Returns 0 when there is nothing to fetch (no snapshotting verify has run, a
// non-persist call intervened, or the slot is out of range), and the caller keeps
// the old restore-and-re-forward path.
TSG_EXPORT int TSGgml_Qwen35FetchStateSnapshot(
    int slot, void** conv_out_arr, void** delta_out_arr, int num_recurrent_layers)
{
    try
    {
        Q35VerifyCache* c = g_q35vc_last_snap;
        if (c == nullptr || !c->valid || c->n_snapshots <= 1)
            return 0;
        if (slot < 0 || slot >= c->n_snapshots)
            return 0;
        if (conv_out_arr == nullptr || delta_out_arr == nullptr)
            return 0;
        if (static_cast<int>(c->conv_snaps.size()) != num_recurrent_layers
            || static_cast<int>(c->delta_snaps.size()) != num_recurrent_layers)
        {
            set_last_error("Qwen3.5 state snapshot: recurrent layer count mismatch.");
            return 0;
        }

        host_read_barrier();
        for (int i = 0; i < num_recurrent_layers; i++)
        {
            if (c->conv_snaps[i] == nullptr || c->delta_snaps[i] == nullptr)
                return 0;
            if (conv_out_arr[i] != nullptr)
            {
                ggml_backend_tensor_get(c->conv_snaps[i], conv_out_arr[i],
                    static_cast<std::size_t>(slot) * g_q35vc_snap_conv_bytes,
                    g_q35vc_snap_conv_bytes);
            }
            if (delta_out_arr[i] != nullptr)
            {
                ggml_backend_tensor_get(c->delta_snaps[i], delta_out_arr[i],
                    static_cast<std::size_t>(slot) * g_q35vc_snap_delta_bytes,
                    g_q35vc_snap_delta_bytes);
            }
        }
        host_read_barrier();
        clear_last_error();
        return 1;
    }
    catch (const std::exception& ex) { set_last_error(ex.what()); return 0; }
    catch (...) { set_last_error("Unknown error in Qwen3.5 state snapshot fetch."); return 0; }
}

namespace
{
    /// ggml_backend_tensor_copy's precondition, without reaching into ggml-impl.h.
    bool q35v_same_layout(const ggml_tensor* a, const ggml_tensor* b)
    {
        if (a->type != b->type)
            return false;
        for (int i = 0; i < GGML_MAX_DIMS; i++)
        {
            if (a->ne[i] != b->ne[i] || a->nb[i] != b->nb[i])
                return false;
        }
        return true;
    }
}

// Commit ONE recurrent-state snapshot into the LIVE state, entirely on the device.
//
// The live conv_state_in / delta_state_in slices live in one shared buffer that
// EVERY cached verify graph binds, so writing them here is visible to the next
// verify whatever shape it runs at - which is what lets that verify skip its
// ~300 MB state upload, and this step skip the matching download. That round trip
// was the single largest per-step cost of speculative decoding on this trunk.
//
// `slot` counts back from the end of the verified batch: 0 is the post-window
// state, (N-1-accepted) the state the accepted prefix ends in.
TSG_EXPORT int TSGgml_Qwen35CommitStateSnapshot(int slot, int num_recurrent_layers)
{
    try
    {
        Q35VerifyCache* c = g_q35vc_last_snap;
        if (c == nullptr || !c->valid || !c->deferred_state)
            return 0;
        // slot -1 is the post-window state in the *_state_out slices, which is what a
        // single-row step (and a fully-accepted verify with no snapshots) commits.
        const bool from_out = slot < 0;
        if (!from_out && (c->n_snapshots <= 1 || slot >= c->n_snapshots))
            return 0;
        if (static_cast<int>(c->conv_in.size()) != num_recurrent_layers
            || static_cast<int>(c->delta_in.size()) != num_recurrent_layers
            || static_cast<int>(c->conv_out.size()) != num_recurrent_layers
            || static_cast<int>(c->delta_out.size()) != num_recurrent_layers
            || (!from_out
                && (static_cast<int>(c->conv_snap_slots.size()) != num_recurrent_layers * c->n_snapshots
                    || static_cast<int>(c->delta_snap_slots.size()) != num_recurrent_layers * c->n_snapshots)))
        {
            set_last_error("Qwen3.5 state commit: recurrent layer count mismatch.");
            return 0;
        }

        for (int i = 0; i < num_recurrent_layers; i++)
        {
            const int idx = i * c->n_snapshots + (from_out ? 0 : slot);
            ggml_tensor* csrc = from_out ? c->conv_out[i] : c->conv_snap_slots[idx];
            ggml_tensor* dsrc = from_out ? c->delta_out[i] : c->delta_snap_slots[idx];
            if (csrc == nullptr || dsrc == nullptr || c->conv_in[i] == nullptr || c->delta_in[i] == nullptr)
                return 0;
            // ggml_backend_tensor_copy is void and ASSERTS same-layout; the slot
            // views were built with exactly the live slices' shapes, so a mismatch
            // is a build-side bug - but check rather than abort the process.
            if (!q35v_same_layout(csrc, c->conv_in[i]) || !q35v_same_layout(dsrc, c->delta_in[i]))
            {
                set_last_error("Qwen3.5 state commit: snapshot/live layout mismatch.");
                return 0;
            }
            ggml_backend_tensor_copy(csrc, c->conv_in[i]);
            ggml_backend_tensor_copy(dsrc, c->delta_in[i]);
        }
        clear_last_error();
        return 1;
    }
    catch (const std::exception& ex) { set_last_error(ex.what()); return 0; }
    catch (...) { set_last_error("Unknown error in Qwen3.5 state commit."); return 0; }
}

// Read the LIVE recurrent state back to the host. The device copy is authoritative
// while a speculative session keeps committing snapshots into it; anything that has
// to run the op-by-op recurrent path (a prefill chunk, an unsupported shape, a
// backend without the fused verify) needs the host mirror to catch up first.
TSG_EXPORT int TSGgml_Qwen35DrainDeviceState(
    void** conv_out_arr, void** delta_out_arr, int num_recurrent_layers)
{
    try
    {
        Q35VerifyCache* c = g_q35vc_last_snap;
        if (c == nullptr || !c->valid)
            return 0;
        if (conv_out_arr == nullptr || delta_out_arr == nullptr)
            return 0;
        if (static_cast<int>(c->conv_in.size()) != num_recurrent_layers
            || static_cast<int>(c->delta_in.size()) != num_recurrent_layers)
        {
            set_last_error("Qwen3.5 state drain: recurrent layer count mismatch.");
            return 0;
        }

        host_read_barrier();
        for (int i = 0; i < num_recurrent_layers; i++)
        {
            if (c->conv_in[i] == nullptr || c->delta_in[i] == nullptr)
                return 0;
            if (conv_out_arr[i] != nullptr)
                ggml_backend_tensor_get(c->conv_in[i], conv_out_arr[i], 0, g_q35vc_snap_conv_bytes);
            if (delta_out_arr[i] != nullptr)
                ggml_backend_tensor_get(c->delta_in[i], delta_out_arr[i], 0, g_q35vc_snap_delta_bytes);
        }
        host_read_barrier();
        clear_last_error();
        return 1;
    }
    catch (const std::exception& ex) { set_last_error(ex.what()); return 0; }
    catch (...) { set_last_error("Unknown error in Qwen3.5 state drain."); return 0; }
}

// Release every rank's parked tensor-parallel prefill graph. The C# side calls
// this when the model is torn down (the parked contexts hold pooled memory and
// the plans reference per-rank gallocr buffers).
TSG_EXPORT void TSGgml_Qwen35ReleaseVerifyTpGraphs()
{
    // This is the model-teardown hook for both TP and single-device Qwen35.
    // Drop graphs before their shared recurrent-state buffer so Metal residency
    // sets do not retain a live allocation past backend destruction.
    reset_qwen35_verify_cache();
    for (int r = 0; r < TSG_MAX_DEVICES; ++r)
    {
        tsg::ScopedRank rank(r);
        if (g_q35v_tp[r].plan.graph != nullptr || g_q35v_tp[r].context.value != nullptr)
            g_q35v_tp[r].reset();
        if (g_q35v_state_bufs[r] != nullptr)
        {
            ggml_backend_buffer_free(g_q35v_state_bufs[r]);
            g_q35v_state_bufs[r] = nullptr;
            g_q35v_state_buf_sizes[r] = 0;
            g_q35v_state_backends[r] = nullptr;
        }
    }
}

// Drop the persistent verify-graph cache. Called from C# whenever the attention KV
// device buffer or GDN-state buffers may have moved (KV cache grow / reset), since
// the cached graphs pin those addresses.
TSG_EXPORT void TSGgml_Qwen35ResetVerifyCache()
{
    reset_qwen35_verify_cache();
}
