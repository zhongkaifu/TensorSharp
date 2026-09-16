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
#include "ggml-impl.h"
#include "ggml_ops_qwen4exp_qsa.h"
#include "ggml_ops_matmul_precision.h"
#include "ggml_ops_dsv4_fused.h"
#ifdef TSG_GGML_USE_CUDA
#include "ggml-cuda.h"
#endif
#include <cstdlib>
#include <cstdio>

#include <cmath>
#include <cstring>
#include <cstdint>
#include <type_traits>
#include <vector>
#include <unordered_map>

using namespace tsg;
#include "ggml_ops_qwen4exp_qsa.inc"

// ============================================================================
// Qwen3.8-Flash-Next (qwen4exp) fused layers.
//
// Three per-layer half-kernels (FFN, GDN, attention), and above them ONE graph
// for a whole run of layers - which in practice means one graph per token, cut
// only where the PLE layer has to read the residual on the host.
//
// Why. A qwen4exp decode step is dispatch bound, not arithmetic bound: the
// op-by-op path issued roughly 850 GGML submissions per token; the per-layer
// fused kernels cut that to 96, but 96 is still 96 graph launches, 96 stream
// drains and 96 host round trips of the residual where llama.cpp has exactly
// one of each. The token-span kernel is that same shape: one graph, the
// residual crossing the PCIe bus twice per token instead of 192 times, and one
// CUDA graph for ggml-cuda to capture instead of 96 alternating ones.
//
// The per-layer entry points remain as the fallback - QSA over budget, foreign
// shapes, debugging - and share the node builders with the span, so there is a
// single source of truth for the graph each half builds.
// ============================================================================
namespace
{
    constexpr const char* kQwen4ExpFfnKernel = "qwen4exp fused FFN block";
    constexpr const char* kQwen4ExpGdnKernel = "qwen4exp fused GDN block";
    constexpr const char* kQwen4ExpAttnKernel = "qwen4exp fused attention block";
    constexpr const char* kQwen4ExpSpanKernel = "qwen4exp fused token span";
    constexpr int kQwen4ExpMaxSlots = 128;
    constexpr int kQwen4ExpSpanSlots = 128;
    // A 48-layer span builds ~150 nodes a layer; 16384 leaves headroom.
    constexpr int kQwen4ExpSpanGraphSize = 16384;

    // Q4eCachedBind (a cache-resolved weight binding a REPLAY can re-resolve)
    // lives in ggml_ops_internal.h, shared with the arena kernel.

    // One built graph per layer (or per span), kept across tokens.
    //
    // Building the graph is most of a decode step's cost here: the arithmetic for
    // one token is trivial next to a ggml context, a gallocr plan and a fresh
    // topology 48 times a token. Holding the graph makes a decode step an upload of
    // the residual, a replay and a download - and keeps the topology byte-identical
    // token to token, which is what lets ggml-cuda's graph capture engage.
    struct Qwen4ExpFfnCache
    {
        bool valid = false;
        ggml_context* ctx = nullptr;
        ggml_cgraph* graph = nullptr;
        ggml_gallocr_t alloc = nullptr;
        ggml_tensor* res_in = nullptr;
        ggml_tensor* res_out = nullptr;
        int n_tokens = 0;
        int hc_dim = 0;
        const void* sig = nullptr;      // the descriptor this graph was built from
        // Span graphs are keyed on all three descriptor arrays and the layer range.
        const void* sig2 = nullptr;
        const void* sig3 = nullptr;
        const void* sig4 = nullptr;      // head descriptor, or null
        const void* sig5 = nullptr;      // PLE descriptor, or null
        const void* qsa_sig = nullptr;
        std::vector<Q4eQsaInputs> qsa_inputs;
        ggml_backend_t precise_backend = nullptr;
        ggml_tensor* logits = nullptr;
        int logits_rows = 1;
        bool export_hidden = false;
        ggml_tensor* ple_emb_in = nullptr;
        int layer_begin = -1;
        int layer_end = -1;
        int kv_capacity = -1;
        // Span only: the first layer contributes only its FFN half (its attention
        // half already ran through the per-layer kernel).
        int first_ffn_only = 0;
        // Whether the pos tensor carries the 4 IMRoPE sections (image prompts).
        int use_mrope = 0;
        // Recurrent state, read through *_in and written through *_out; the caller
        // copies out -> in after every compute. The span writes the state in place
        // instead and only uses the buffer.
        ggml_tensor* conv_in = nullptr;
        ggml_tensor* conv_out = nullptr;
        ggml_tensor* ssm_in = nullptr;
        ggml_tensor* ssm_out = nullptr;
        ggml_backend_buffer_t state_buf = nullptr;
        bool state_ready = false;
        // Whether res_in is bound to the shared device buffer. A graph built one way
        // cannot be replayed the other: a non-resident graph owns a private res_in, so
        // replaying it in resident mode chains nothing and the layers stop composing.
        int res_resident = -1;
        // Span only: state write-backs issued host-side after each compute when
        // TS_Q4E_SPAN_STATE=host - (src, dst) pairs, src a graph output, dst the
        // persistent state tensor.
        std::vector<std::pair<ggml_tensor*, ggml_tensor*>> span_copies;
        std::vector<Q4eCachedBind> rebinds;
        unsigned rebind_tick = 0;
        // Span attention inputs, one set per in-span attention layer. Private per
        // layer, exactly as the per-layer kernels have them.
        std::vector<ggml_tensor*> span_masks;
        std::vector<ggml_tensor*> span_pos;
        std::vector<ggml_tensor*> span_kvidx;
        std::vector<ggml_tensor*> gdn_probe;
        // Attention only: the mask is an input and the graph shape follows n_kv.
        ggml_tensor* mask = nullptr;
        ggml_tensor* pos = nullptr;
        ggml_tensor* kv_idx = nullptr;
        int n_kv = -1;

        // Drop the graph but KEEP the recurrent state: a shape change does this.
        void reset_graph()
        {
            if (precise_backend) { ggml_backend_synchronize(precise_backend); ggml_backend_free(precise_backend); precise_backend = nullptr; }
            qsa_sig = nullptr; qsa_inputs.clear();
            if (alloc) { ggml_gallocr_free(alloc); alloc = nullptr; }
            if (ctx) { ggml_free(ctx); ctx = nullptr; }
            graph = nullptr; res_in = nullptr; res_out = nullptr;
            conv_in = conv_out = ssm_in = ssm_out = nullptr;
            valid = false; n_tokens = 0; hc_dim = 0; sig = nullptr; res_resident = -1;
            sig2 = nullptr; sig3 = nullptr; sig4 = nullptr; sig5 = nullptr;
            logits = nullptr; logits_rows = 1; export_hidden = false; ple_emb_in = nullptr;
            layer_begin = -1; layer_end = -1; kv_capacity = -1; first_ffn_only = 0;
            use_mrope = 0;
            mask = nullptr; pos = nullptr; kv_idx = nullptr; n_kv = -1;
            span_copies.clear(); rebinds.clear(); gdn_probe.clear();
            span_masks.clear(); span_pos.clear(); span_kvidx.clear();
        }

        // Drop everything including the state: a KV reset does this.
        void reset()
        {
            reset_graph();
            if (state_buf) { ggml_backend_buffer_free(state_buf); state_buf = nullptr; }
            state_ready = false;
        }
    };

    // The PLE conv history: one persistent device buffer (one PLE layer in the
    // shipped checkpoint). ready=false re-seeds from the host on the next build.
    // ---- per-sequence recurrent-state store -------------------------------
    // GDN conv+ssm state (one entry per layer per sequence) and the PLE conv
    // history live in device buffers KEYED BY THE HOST SEED POINTER the C# side
    // passes in the descriptor (each sequence holder owns its own pinned seed
    // arrays, so the key is per-sequence per-layer for free). The buffer base
    // is baked into every persisted graph that binds it; the graphs are keyed
    // on the (per-holder) descriptor addresses, so a graph only ever binds its
    // own holder's state entry. `ready` gates the one-time seed upload: a
    // rebuild binds the existing buffer WITHOUT re-seeding (the device copy is
    // authoritative; the host seed is stale after the first forward).
    // (Q4eSeqStateEntry itself lives in ggml_ops_internal.h - the arena kernel
    // joins from and flushes into these entries device-to-device.)
    // ---- per-device executor state ---------------------------------------
    //
    // A layer split puts a contiguous run of layers on each GPU, so the same
    // process drives several devices within one token. Every field below either
    // IS device memory or bakes a device pointer into a persisted graph, so a
    // single shared copy would hand device 1 device 0's buffers.
    //
    // Indexed by tsg::g_active_rank, which ScopedRank sets around each span.
    struct Q4eDeviceState
    {
        Qwen4ExpFfnCache ffn[kQwen4ExpMaxSlots];
        Qwen4ExpFfnCache gdn[kQwen4ExpMaxSlots];
        Qwen4ExpFfnCache attn[kQwen4ExpMaxSlots];
        Qwen4ExpFfnCache span[kQwen4ExpSpanSlots];

        // GDN conv+ssm state and the PLE conv history, keyed by the HOST SEED
        // POINTER from the descriptors. Per device as well as per key: with a
        // layer split, layer L's recurrent state must live on layer L's GPU, and
        // the same holder's seed pointers are used for layers on both.
        std::unordered_map<const void*, Q4eSeqStateEntry> seq_state;

        // The 4-wide residual held on the device across a span.
        ggml_backend_buffer_t res_buf = nullptr;
        std::size_t res_capacity = 0;
        ggml_context* res_ctx = nullptr;
        ggml_tensor* res = nullptr;
    };

    Q4eDeviceState g_q4e_devs[tsg::TSG_MAX_DEVICES];
    inline Q4eDeviceState& q4e_dev() { return g_q4e_devs[tsg::g_active_rank]; }

#define g_q4e_ffn            (q4e_dev().ffn)
#define g_q4e_gdn            (q4e_dev().gdn)
#define g_q4e_attn           (q4e_dev().attn)
#define g_q4e_span           (q4e_dev().span)
#define g_q4e_seq_state      (q4e_dev().seq_state)
#define g_q4e_res_buf        (q4e_dev().res_buf)
#define g_q4e_res_capacity   (q4e_dev().res_capacity)
#define g_q4e_res_ctx        (q4e_dev().res_ctx)
#define g_q4e_res            (q4e_dev().res)
}

// External linkage (declared in ggml_ops_internal.h): shared with the arena
// kernel. Both operate on the ACTIVE rank's map - callers select the device
// first.
Q4eSeqStateEntry* q4e_seq_state(const void* key, std::size_t bytes)
    {
        if (key == nullptr) return nullptr;
        Q4eSeqStateEntry& e = g_q4e_seq_state[key];
        if (e.buf != nullptr && e.bytes < bytes)
        {
            ggml_backend_buffer_free(e.buf);
            e.buf = nullptr;
            e.ready = false;
        }
        if (e.buf == nullptr)
        {
            e.buf = ggml_backend_buft_alloc_buffer(
                    ggml_backend_get_default_buffer_type(g_backend), bytes);
            e.bytes = bytes;
            e.ready = false;
            if (e.buf == nullptr) return nullptr;
        }
        return &e;
    }

// Lookup WITHOUT creating: the arena join declines on a missing entry instead
// of fabricating recurrent state for a sequence that never ran through the
// span.
Q4eSeqStateEntry* q4e_seq_state_find(const void* key)
{
    if (key == nullptr) return nullptr;
    auto it = g_q4e_seq_state.find(key);
    return it == g_q4e_seq_state.end() ? nullptr : &it->second;
}

// ggml-cuda's flash attention takes F16 K/V and one of a fixed set of head sizes;
// for head_dim 256 the only other condition is V->ne[0] == K->ne[0], which holds
// here. TS_Q4E_FLASH_ATTN=0 falls back to the soft_max path.
bool q4e_flash_attn_ok(int kv_type, int head_dim)
{
    static const bool enabled = []{
        const char* e = std::getenv("TS_Q4E_FLASH_ATTN");
        return !(e != nullptr && e[0] == '0');
    }();
    if (!enabled || kv_type != GGML_TYPE_F16) return false;
    switch (head_dim)
    {
        case 64: case 80: case 96: case 112: case 128: case 256: return true;
        default: return false;
    }
}

namespace
{

    // ggml-cuda picks its GQA-optimised flash-attention kernel only when
    // K->ne[1] % FATTN_KQ_STRIDE == 0, so the window is padded to that and the pad
    // masked off. This is what llama.cpp's get_n_kv rounds to, for the same reason.
    // The padding is only worth it under flash attention: the soft_max path pays for
    // every padded column instead of skipping the block.
    constexpr int kQwen4ExpKvStride = 256;

    // The span writes the GDN state in place inside the graph - cpy(tail ->
    // conv_state) expanded after every node that reads the state, so node order
    // sequences the write behind the read. This is the DEFAULT: no host-issued
    // copies and no extra synchronize per span. The historical "in-place writes
    // do not take effect" failures - including an earlier note in this file
    // declaring them measurably wrong - were the gallocr leaf-free bug corrupting
    // the small gate weights, not the write-back; with uploaded leafs
    // OUTPUT-flagged the in-place dataflow verifies clean at every length.
    // TS_Q4E_SPAN_STATE=host restores the copied-out dataflow for comparison.
    // TS_Q4E_SPAN_REBUILD=1 disables the span replay path entirely - every call
    // rebuilds the graph. Diagnosis only: separates a wrong-graph bug from a
    // wrong-replay one.
    bool q4e_span_force_rebuild()
    {
        static const bool v = []{
            const char* e = std::getenv("TS_Q4E_SPAN_REBUILD");
            return e != nullptr && e[0] == '1';
        }();
        return v;
    }

    // TS_Q4E_SPAN_FA_MAX=N: only the first N attention layers in a span use flash
    // attention; the rest run the soft_max path over the SAME padded window.
    // Diagnosis only - N=0 separates "the pad poisons the output" from "the flash
    // attention node does".
    int q4e_span_fa_max()
    {
        static const int v = []{
            const char* e = std::getenv("TS_Q4E_SPAN_FA_MAX");
            return (e != nullptr && *e != 0) ? std::atoi(e) : 1 << 30;
        }();
        return v;
    }

    // TS_Q4E_SPAN_TRACE=1 prints the residual L2 norm after every layer of every
    // span call. Diagnosis only: diffing a good run against a bad one names the
    // first layer whose output moves.
    bool q4e_span_trace()
    {
        static const bool v = []{
            const char* e = std::getenv("TS_Q4E_SPAN_TRACE");
            return e != nullptr && e[0] == '1';
        }();
        return v;
    }

    bool q4e_span_state_in_graph()
    {
        static const bool v = []{
            const char* e = std::getenv("TS_Q4E_SPAN_STATE");
            return !(e != nullptr && e[0] == 'h');
        }();
        return v;
    }

    // Re-resolve a persisted graph's cache-bound weights before a replay. If the
    // resident cache moved (or re-created) any device copy, the graph's captured
    // pointers are stale: report it so the caller rebuilds. Content refreshes
    // (needs_upload with an unmoved pointer) are handled in place.
    bool q4e_refresh_bindings(Qwen4ExpFfnCache* slot, ggml_backend_dev_t dev)
    {
        // The full walk is a few hundred hash lookups; a moved device copy has
        // never been observed (the guard exists as insurance), so sample it. A
        // rebuild always re-binds everything regardless.
        if (++slot->rebind_tick % 32 != 1)
            return true;
        for (const Q4eCachedBind& cb : slot->rebinds)
        {
            void* before = cb.tensor->data;
            bool needs_upload = false;
            if (!try_bind_cached_tensor(g_backend, dev, cb.tensor, cb.data, cb.bytes,
                                        needs_upload, cb.usage))
                return false;
            if (cb.tensor->data != before)
            {
                fprintf(stderr, "[q4e] cached weight moved (%p -> %p, %zu bytes); rebuilding the graph%c",
                        before, cb.tensor->data, cb.bytes, 10);
                return false;
            }
            if (needs_upload)
            {
                // Same redirect as Q4eBinder::flush: for a quantized weight cb.data
                // is a CacheKey (a GCHandle value), not memory.
                ggml_backend_tensor_set(cb.tensor, resolve_upload_source(cb.data), 0, cb.bytes);
            }
        }
        return true;
    }

    // Print the L2 of the probed GDN nodes. Diagnosis only.
    void q4e_trace_probe(Qwen4ExpFfnCache* slot, const char* tag, int position)
    {
        if (slot->gdn_probe.empty()) return;
        static const char* names[] = { "mixed", "qkv", "convout", "gdnout", "proj" };
        fprintf(stderr, "[q4e-gdn0] %s pos=%d:", tag, position);
        std::vector<float> buf;
        for (std::size_t i = 0; i < slot->gdn_probe.size() && i < 5; ++i)
        {
            ggml_tensor* t = slot->gdn_probe[i];
            buf.resize((std::size_t)ggml_nelements(t));
            ggml_backend_tensor_get(t, buf.data(), 0, ggml_nbytes(t));
            double n2 = 0.0;
            for (float f : buf) n2 += (double)f * f;
            fprintf(stderr, " %s=%.9e", names[i], std::sqrt(n2));
        }
        fprintf(stderr, "%c", 10);
    }

    // Print the L2 of every state tensor a span carries. Diagnosis only.
    void q4e_trace_state(Qwen4ExpFfnCache* slot, const char* tag, int position)
    {
        if (!q4e_span_trace() || slot->span_copies.empty()) return;
        fprintf(stderr, "[q4e-state] %s pos=%d:", tag, position);
        std::vector<float> buf;
        for (std::size_t i = 0; i < slot->span_copies.size(); ++i)
        {
            ggml_tensor* st = slot->span_copies[i].second;
            buf.resize((std::size_t)ggml_nelements(st));
            ggml_backend_tensor_get(st, buf.data(), 0, ggml_nbytes(st));
            double n2 = 0.0;
            for (float f : buf) n2 += (double)f * f;
            fprintf(stderr, " %.9e", std::sqrt(n2));
        }
        fprintf(stderr, "%c", 10);
    }

    int q4e_pad_kv(int n_kv, int kv_capacity, bool use_flash)
    {
        const int stride = use_flash ? kQwen4ExpKvStride : 1;
        int n_kv_pad = ((n_kv + stride - 1) / stride) * stride;
        if (n_kv_pad > kv_capacity) n_kv_pad = kv_capacity;
        return n_kv_pad;
    }

    // Fill the two index inputs: RoPE positions and the KV rows this step writes.
    // Values change per token, shapes do not - which is what lets the graph persist.
    // A multimodal graph's pos tensor holds 4 sections (T|H|W|zero, IMRoPE order);
    // mrope3 is the per-token (t,h,w) table for image prompts, null for text where
    // every component is the scalar position.
    // position indexes the KV cache rows; rope_position is the rotary position of
    // the first token, which falls BEHIND the cache index once an image has been
    // compacted into the position stream (IMRoPE gives an HxW image max(H,W)
    // positions, not HxW). llama.cpp's mtmd advances n_past the same way.
    void q4e_set_attn_indices(ggml_tensor* pos, ggml_tensor* kv_idx, int T, int position,
                              const int32_t* mrope3 = nullptr, int rope_position = -1)
    {
        if (rope_position < 0) rope_position = position;
        std::vector<int64_t> k((std::size_t)T);
        for (int i = 0; i < T; ++i) k[i] = position + i;
        ggml_backend_tensor_set(kv_idx, k.data(), 0, (std::size_t)T * sizeof(int64_t));
        if (pos == nullptr) return;
        const int comps = (int)(pos->ne[0] / T);
        std::vector<int32_t> p((std::size_t)comps * T);
        if (comps == 1)
        {
            for (int i = 0; i < T; ++i) p[i] = rope_position + i;
        }
        else
        {
            for (int i = 0; i < T; ++i)
            {
                const int32_t t = mrope3 ? mrope3[3 * i + 0] : rope_position + i;
                const int32_t h = mrope3 ? mrope3[3 * i + 1] : rope_position + i;
                const int32_t w = mrope3 ? mrope3[3 * i + 2] : rope_position + i;
                p[i] = t; p[T + i] = h; p[2 * T + i] = w; p[3 * T + i] = 0;
            }
        }
        ggml_backend_tensor_set(pos, p.data(), 0, p.size() * sizeof(int32_t));
    }

    // TS_Q4E_LOG=1 reports how often each kernel rebuilt its graph rather than
    // replaying it.
    struct Q4eStat { long builds = 0; long replays = 0; };
    Q4eStat g_q4e_stat[4];   // 0 = FFN, 1 = GDN, 2 = ATTN, 3 = SPAN
    bool q4e_log_enabled()
    {
        static const bool v = []{
            const char* e = std::getenv("TS_Q4E_LOG");
            return e != nullptr && e[0] == '1';
        }();
        return v;
    }
    void q4e_note(int k, bool build)
    {
        if (!q4e_log_enabled()) return;
        if (build) g_q4e_stat[k].builds++; else g_q4e_stat[k].replays++;
        long total = 0;
        for (const Q4eStat& st : g_q4e_stat) total += st.builds + st.replays;
        if (total % 2000 == 0 || (k == 3 && (g_q4e_stat[3].builds + g_q4e_stat[3].replays) % 200 == 0))
            fprintf(stderr, "[q4e] ffn b=%ld r=%ld | gdn b=%ld r=%ld | attn b=%ld r=%ld | span b=%ld r=%ld\n",
                    g_q4e_stat[0].builds, g_q4e_stat[0].replays,
                    g_q4e_stat[1].builds, g_q4e_stat[1].replays,
                    g_q4e_stat[2].builds, g_q4e_stat[2].replays,
                    g_q4e_stat[3].builds, g_q4e_stat[3].replays);
    }

    // Stamp every persisted graph with a stable non-zero id.
    //
    // ggml_new_graph leaves uid at 0, and ggml-cuda treats 0 as "unknown", so on every
    // replay it re-walks the nodes comparing a copy of each tensor struct and its
    // sources to decide whether the captured CUDA graph is still valid. With a stable
    // id it recognises the graph and skips that walk.
    // TS_Q4E_GRAPH_UID=0 leaves uid at 0 so ggml-cuda re-checks node properties on
    // every replay instead of trusting the id.
    bool q4e_graph_uid_enabled()
    {
        static const bool v = []{
            const char* e = std::getenv("TS_Q4E_GRAPH_UID");
            return !(e != nullptr && e[0] == '0');
        }();
        return v;
    }

    // TS_Q4E_PHASE=1 prints wall times for the span build phases at T>1.
    bool q4e_phase_log()
    {
        static const bool v = []{
            const char* e = std::getenv("TS_Q4E_PHASE");
            return e != nullptr && e[0] == '1';
        }();
        return v;
    }
    double q4e_now_ms()
    {
        return (double)ggml_time_us() / 1000.0;
    }

    uint64_t q4e_next_graph_uid()
    {
        static uint64_t next = 1;
        return next++;
    }

    // The 4-wide residual, held on the DEVICE for the whole forward.
    //
    // Used by the per-layer fallback's residency experiment; the token span does not
    // need it - inside one graph the residual never exists on the host at all.
}

// Clamp a caller-supplied device index to an initialized rank. -1 (or an
// out-of-range value from a host that predates the layer split) means "the
// current rank", which is 0 on every single-GPU run. External linkage: the
// arena kernel selects its per-device pool with the same rule as the span.
int q4e_resolve_device(int device)
{
    if (device < 0) return tsg::g_active_rank;
    const int ndev = tsg::g_device_count.load(std::memory_order_acquire);
    if (device >= ndev || device >= tsg::TSG_MAX_DEVICES) return tsg::g_active_rank;
    return device;
}

namespace
{
    // Ensure the shared residual tensor exists and is at least `bytes` big.
    // Per device: see Q4eDeviceState.
    bool q4e_res_ensure(std::size_t bytes)
    {
        if (g_q4e_res != nullptr && g_q4e_res_capacity >= bytes)
            return true;
        // Every persisted graph binds its res_in to this buffer's base, so a realloc
        // here strands every one of them. Loud on purpose.
        if (g_q4e_res != nullptr)
            fprintf(stderr, "[q4e] residual buffer REALLOC %zu -> %zu bytes\n",
                    g_q4e_res_capacity, bytes);
        if (g_q4e_res_ctx) { ggml_free(g_q4e_res_ctx); g_q4e_res_ctx = nullptr; }
        if (g_q4e_res_buf) { ggml_backend_buffer_free(g_q4e_res_buf); g_q4e_res_buf = nullptr; }
        g_q4e_res = nullptr;

        ggml_init_params ip{};
        ip.mem_size = ggml_tensor_overhead() * 4;
        ip.mem_buffer = nullptr;
        ip.no_alloc = true;
        g_q4e_res_ctx = ggml_init(ip);
        if (g_q4e_res_ctx == nullptr) return false;

        g_q4e_res = ggml_new_tensor_1d(g_q4e_res_ctx, GGML_TYPE_F32, (int64_t)(bytes / sizeof(float)));
        g_q4e_res_buf = ggml_backend_buft_alloc_buffer(
                ggml_backend_get_default_buffer_type(g_backend), bytes);
        if (g_q4e_res_buf == nullptr) return false;
        if (ggml_backend_tensor_alloc(g_q4e_res_buf, g_q4e_res,
                ggml_backend_buffer_get_base(g_q4e_res_buf)) != GGML_STATUS_SUCCESS)
            return false;
        g_q4e_res_capacity = bytes;
        return true;
    }

}

// One place for the weight-binding policy every block builder shares (the
// struct lives in ggml_ops_internal.h so the arena kernel binds through the
// same policy). Collecting the uploads rather than binding immediately is what
// lets several layers build into one graph: everything is bound before a
// single allocation pass runs over the lot.
void Q4eBinder::add(ggml_tensor* tgt, void* data, std::size_t bytes,
                    ggml_backend_buffer_usage usage)
{
            if (tgt == nullptr || data == nullptr) return;
            // Mutable KV must survive graph rebuilds even in small models. Its
            // host seed may be stale after the first forward, so a graph-local
            // fallback would silently replace live cache rows with that seed.
            if (bytes >= 4096 || usage == GGML_BACKEND_BUFFER_USAGE_ANY)
            {
                bool needs_upload = false;
                if (try_bind_cached_tensor(g_backend, dev, tgt, data, bytes, needs_upload, usage))
                {
                    cached.push_back({tgt, data, bytes, usage});
                    if (needs_upload) upload_list.push_back({tgt, data, bytes});
                    return;
                }
                ggml_backend_buffer_t buf = nullptr;
                if (try_get_host_ptr_buffer(g_backend, dev, data, bytes, true, buf))
                {
                    if (ggml_backend_tensor_alloc(buf, tgt, data) == GGML_STATUS_SUCCESS)
                        return;
                }
                if (usage == GGML_BACKEND_BUFFER_USAGE_ANY)
                    throw std::runtime_error("qwen4exp: could not persist mutable cache buffer");
                // A weight this size normally cache-binds; falling through here means
                // the resident cache could not take it (VRAM pressure). It will live
                // in the graph's own allocation instead - flagged below - and that is
                // worth being able to see.
                fprintf(stderr, "[q4e] weight (%zu bytes) fell out of the resident cache into the graph allocation%c",
                        bytes, 10);
            }
            // The tensor becomes a gallocr-owned leaf, uploaded once at build time.
            // It MUST carry the OUTPUT flag: ggml_gallocr_free_node only exempts
            // outputs ("graph outputs are never freed") - the INPUT flag controls
            // early allocation but the free path ignores it - so an unprotected
            // leaf is freed after its last consumer and its memory reused by later
            // intermediates. The FIRST compute reads the weight correctly and then
            // overwrites it in place; every REPLAY of the persisted graph reads
            // whatever activations landed there, an error that scales with their
            // magnitude. A build works exactly once - precisely the difference
            // between a rebuilt-per-token graph that stays correct and a persisted
            // one that decays. The INPUT flag stays for the early allocation.
            ggml_set_input(tgt);
            ggml_set_output(tgt);
            upload_list.push_back({tgt, data, bytes});
}

void Q4eBinder::flush()
{
            // resolve_upload_source, not the raw pointer. For a QUANTIZED weight the
            // "host pointer" C# passes is a CacheKey - a GCHandle value, not memory -
            // and the redirect turns it back into the real bytes. Every other fused
            // executor in this repo does this (dflash, qwen35, gemma4, gptoss,
            // muse_glimmer); qwen4exp got away without it only because rank 0 always
            // had every weight preloaded, so needs_upload was never true here.
            // A layer split makes a misplaced weight reachable, and without this the
            // symptom is a cudaMemcpy from a handle value. It also counts the
            // redirect (reported by TS_GGML_LOG_VRAM=1), so a misplacement shows up
            // as a number instead of a crash.
            for (const Q4eHostBinding& hb : upload_list)
                ggml_backend_tensor_set(hb.tensor, resolve_upload_source(hb.data), 0, hb.bytes);
            upload_list.clear();
}

// The per-layer descriptor structs (TSGgmlQwen4ExpFfnArgs / GdnArgs / AttnArgs)
// live in ggml_ops_internal.h, shared with the arena kernel; the C# side
// mirrors their layout.

// ============================================================================
// Node builders. Each appends one half-layer to ctx and returns the new
// residual; the entry points and the token span share them, so the graph a
// half builds has a single source of truth.
// ============================================================================

// FFN half: hyper-connection mixer -> routed experts + gated shared expert ->
// hyper-connection scatter. No side effects; expands nothing.
ggml_tensor* q4e_nodes_ffn(
    ggml_context* ctx, Q4eBinder& bnd,
    const TSGgmlQwen4ExpFfnArgs* a, ggml_tensor* res_in,
    int n_embd, int hc, int hc_low_rank, int T,
    int n_expert, int n_expert_used, int n_ff, int n_ff_sh, float eps)
{
    const int hc_dim = hc * n_embd;
    ggml_tensor* w_norm = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, hc_dim);
    ggml_tensor* w_down = ggml_new_tensor_2d(ctx, (ggml_type)a->hc_down_type, hc_dim, hc_low_rank);
    ggml_tensor* w_up = ggml_new_tensor_2d(ctx, (ggml_type)a->hc_up_type, hc_low_rank, hc_dim);
    ggml_tensor* w_inject = ggml_new_tensor_2d(ctx, (ggml_type)a->hc_inject_type, hc_dim, hc);
    ggml_tensor* w_router = ggml_new_tensor_2d(ctx, (ggml_type)a->router_type, n_embd, n_expert);
    ggml_tensor* w_gate_e = ggml_new_tensor_3d(ctx, (ggml_type)a->gate_exps_type, n_embd, n_ff, n_expert);
    ggml_tensor* w_up_e = ggml_new_tensor_3d(ctx, (ggml_type)a->up_exps_type, n_embd, n_ff, n_expert);
    ggml_tensor* w_down_e = ggml_new_tensor_3d(ctx, (ggml_type)a->down_exps_type, n_ff, n_embd, n_expert);
    ggml_tensor* w_sh_gi = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, n_embd, 1);
    ggml_tensor* w_sh_g = ggml_new_tensor_2d(ctx, (ggml_type)a->sh_gate_type, n_embd, n_ff_sh);
    ggml_tensor* w_sh_u = ggml_new_tensor_2d(ctx, (ggml_type)a->sh_up_type, n_embd, n_ff_sh);
    ggml_tensor* w_sh_d = ggml_new_tensor_2d(ctx, (ggml_type)a->sh_down_type, n_ff_sh, n_embd);

    // ---- hyper-connection mixer ------------------------------------------
    // Grouped RMS norm: normalise over ONE residual stream, then scale the
    // whole hc-wide row by the gamma vector.
    ggml_tensor* res3 = ggml_reshape_3d(ctx, res_in, n_embd, hc, T);
    ggml_tensor* xn = ggml_rms_norm(ctx, res3, eps);
    xn = ggml_reshape_2d(ctx, xn, hc_dim, T);
    xn = ggml_mul(ctx, xn, w_norm);

    ggml_tensor* lo = ggml_mul_mat(ctx, w_down, xn);
    lo = ggml_silu(ctx, ggml_scale(ctx, lo, 1.0f / (float)hc));
    ggml_tensor* gate = ggml_sigmoid(ctx, ggml_mul_mat(ctx, w_up, lo));

    ggml_tensor* gated = ggml_mul(ctx, xn, gate);
    gated = ggml_reshape_3d(ctx, gated, n_embd, hc, T);

    // Collapse the streams by their mean.
    ggml_tensor* mixed = ggml_cont(ctx, ggml_view_2d(ctx, gated, n_embd, T,
            ggml_row_size(gated->type, n_embd) * hc, 0));
    for (int c = 1; c < hc; ++c)
    {
        ggml_tensor* s = ggml_view_2d(ctx, gated, n_embd, T,
                ggml_row_size(gated->type, n_embd) * hc,
                ggml_row_size(gated->type, n_embd) * c);
        mixed = ggml_add(ctx, mixed, s);
    }
    mixed = ggml_scale(ctx, mixed, 1.0f / (float)hc);

    ggml_tensor* inject = ggml_mul_mat(ctx, w_inject, xn);   // [hc, T]

    // ---- routed experts ---------------------------------------------------
    // Softmax over every expert, top-k, then renormalise the selected
    // weights - llama.cpp's build_moe_ffn with norm_w.
    ggml_tensor* logits = ggml_mul_mat(ctx, w_router, mixed);      // [n_expert, T]
    ggml_tensor* probs = ggml_soft_max(ctx, logits);
    // ggml_argsort_top_k, not ggml_top_k: this is the exact node shape llama.cpp's
    // build_moe_ffn emits, and ggml-cuda's topk_moe fusion matches on the node
    // sequence rather than on intent.
    ggml_tensor* sel = ggml_argsort_top_k(ctx, probs, n_expert_used);  // [n_used, T] i32

    ggml_tensor* w_sel = ggml_get_rows(ctx,
            ggml_reshape_3d(ctx, probs, 1, n_expert, T), sel);     // [1, n_used, T]
    w_sel = ggml_reshape_2d(ctx, w_sel, n_expert_used, T);
    ggml_tensor* w_sum = ggml_sum_rows(ctx, w_sel);                // [1, T]
    w_sel = ggml_div(ctx, w_sel, w_sum);
    w_sel = ggml_reshape_3d(ctx, w_sel, 1, n_expert_used, T);

    ggml_tensor* moe_in = ggml_reshape_3d(ctx, mixed, n_embd, 1, T);
    ggml_tensor* e_up = ggml_mul_mat_id(ctx, w_up_e, moe_in, sel);      // [n_ff, n_used, T]
    ggml_tensor* e_gate = ggml_mul_mat_id(ctx, w_gate_e, moe_in, sel);
    ggml_tensor* par = ggml_mul(ctx, ggml_silu(ctx, e_gate), e_up);
    ggml_tensor* experts = ggml_mul_mat_id(ctx, w_down_e, par, sel);    // [n_embd, n_used, T]
    experts = ggml_mul(ctx, experts, w_sel);

    ggml_tensor* moe_out = ggml_view_2d(ctx, experts, n_embd, T,
            experts->nb[2], 0);
    for (int k = 1; k < n_expert_used; ++k)
    {
        ggml_tensor* s = ggml_view_2d(ctx, experts, n_embd, T,
                experts->nb[2], (std::size_t)k * experts->nb[1]);
        moe_out = ggml_add(ctx, moe_out, s);
    }

    // ---- shared expert, behind its own sigmoid scalar ---------------------
    ggml_tensor* sg = ggml_mul_mat(ctx, w_sh_g, mixed);
    ggml_tensor* su = ggml_mul_mat(ctx, w_sh_u, mixed);
    ggml_tensor* sh = ggml_mul_mat(ctx, w_sh_d, ggml_mul(ctx, ggml_silu(ctx, sg), su));
    ggml_tensor* s_gate = ggml_sigmoid(ctx, ggml_mul_mat(ctx, w_sh_gi, mixed)); // [1, T]
    ggml_tensor* ffn_out = ggml_add(ctx, moe_out, ggml_mul(ctx, sh, s_gate));

    // ---- hyper-connection scatter ----------------------------------------
    // 2*sigmoid centres the weights on 1, so an untrained injection matrix
    // reproduces a plain residual add.
    ggml_tensor* wsc = ggml_scale(ctx, ggml_sigmoid(ctx,
            ggml_scale(ctx, inject, 1.0f / (float)hc)), 2.0f);
    wsc = ggml_reshape_3d(ctx, wsc, 1, hc, T);

    ggml_tensor* b = ggml_reshape_3d(ctx, ffn_out, n_embd, 1, T);
    b = ggml_repeat_4d(ctx, b, n_embd, hc, T, 1);

    ggml_tensor* res_out = ggml_add(ctx, res3, ggml_mul(ctx, b, wsc));
    res_out = ggml_reshape_2d(ctx, res_out, hc_dim, T);

    bnd.add(w_norm, a->hc_norm, (std::size_t)hc_dim * sizeof(float));
    bnd.add(w_down, a->hc_down, (std::size_t)a->hc_down_bytes);
    bnd.add(w_up, a->hc_up, (std::size_t)a->hc_up_bytes);
    bnd.add(w_inject, a->hc_inject, (std::size_t)a->hc_inject_bytes);
    bnd.add(w_router, a->router, (std::size_t)a->router_bytes);
    bnd.add(w_gate_e, a->gate_exps, (std::size_t)a->gate_exps_bytes);
    bnd.add(w_up_e, a->up_exps, (std::size_t)a->up_exps_bytes);
    bnd.add(w_down_e, a->down_exps, (std::size_t)a->down_exps_bytes);
    bnd.add(w_sh_gi, a->sh_gate_inp, (std::size_t)n_embd * sizeof(float));
    bnd.add(w_sh_g, a->sh_gate, (std::size_t)a->sh_gate_bytes);
    bnd.add(w_sh_u, a->sh_up, (std::size_t)a->sh_up_bytes);
    bnd.add(w_sh_d, a->sh_down, (std::size_t)a->sh_down_bytes);

    return res_out;
}

// GDN half: hyper-connection mixer -> projections -> causal conv ->
// gated delta net -> sigmoid-gated norm -> out proj -> scatter.
//
// The caller owns conv_state / ssm_state (created in ctx, allocated into the
// layer's persistent state buffer) and decides how the write-back reaches
// them: the per-layer entry copies out -> in after the compute exactly as it
// always has; the span expands cpy(tail -> conv_state) into the graph AFTER
// the nodes that read the state, so node order sequences the write behind the
// read. The write-back sources come out through `wb`.
ggml_tensor* q4e_nodes_gdn(
    ggml_context* ctx, Q4eBinder& bnd,
    const TSGgmlQwen4ExpGdnArgs* a, ggml_tensor* res_in,
    ggml_tensor* conv_state, ggml_tensor* ssm_state,
    int n_embd, int hc, int hc_low_rank, int T,
    int head_k_dim, int head_v_dim, int n_k_heads, int n_v_heads, int d_conv,
    float eps, Q4eGdnWriteback* wb,
    std::vector<ggml_tensor*>* probe)
{
    const int hc_dim = hc * n_embd;
    const int key_dim = head_k_dim * n_k_heads;
    const int value_dim = head_v_dim * n_v_heads;
    const int conv_dim = key_dim * 2 + value_dim;
    const int hist = d_conv - 1;

    ggml_tensor* w_norm    = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, hc_dim);
    ggml_tensor* w_down    = ggml_new_tensor_2d(ctx, (ggml_type)a->hc_down_type, hc_dim, hc_low_rank);
    ggml_tensor* w_up      = ggml_new_tensor_2d(ctx, (ggml_type)a->hc_up_type, hc_low_rank, hc_dim);
    ggml_tensor* w_inject  = ggml_new_tensor_2d(ctx, (ggml_type)a->hc_inject_type, hc_dim, hc);
    ggml_tensor* w_qkv     = ggml_new_tensor_2d(ctx, (ggml_type)a->qkv_type, n_embd, conv_dim);
    ggml_tensor* w_gate    = ggml_new_tensor_2d(ctx, (ggml_type)a->gate_type, n_embd, value_dim);
    ggml_tensor* w_beta    = ggml_new_tensor_2d(ctx, (ggml_type)a->beta_type, n_embd, n_v_heads);
    ggml_tensor* w_alpha   = ggml_new_tensor_2d(ctx, (ggml_type)a->alpha_type, n_embd, n_v_heads);
    ggml_tensor* w_conv    = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, d_conv, conv_dim);
    ggml_tensor* w_dt      = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, n_v_heads);
    ggml_tensor* w_a       = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, n_v_heads);
    ggml_tensor* w_ssmnorm = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, head_v_dim);
    ggml_tensor* w_out     = ggml_new_tensor_2d(ctx, (ggml_type)a->out_proj_type, value_dim, n_embd);

    // ---- hyper-connection mixer ----
    ggml_tensor* res3 = ggml_reshape_3d(ctx, res_in, n_embd, hc, T);
    ggml_tensor* xn = ggml_rms_norm(ctx, res3, eps);
    xn = ggml_reshape_2d(ctx, xn, hc_dim, T);
    xn = ggml_mul(ctx, xn, w_norm);

    ggml_tensor* lo = ggml_silu(ctx, ggml_scale(ctx, ggml_mul_mat(ctx, w_down, xn), 1.0f / (float)hc));
    ggml_tensor* gt = ggml_sigmoid(ctx, ggml_mul_mat(ctx, w_up, lo));
    ggml_tensor* gated = ggml_reshape_3d(ctx, ggml_mul(ctx, xn, gt), n_embd, hc, T);

    ggml_tensor* mixed = ggml_cont(ctx, ggml_view_2d(ctx, gated, n_embd, T,
            ggml_row_size(gated->type, n_embd) * hc, 0));
    for (int c = 1; c < hc; ++c)
    {
        mixed = ggml_add(ctx, mixed, ggml_view_2d(ctx, gated, n_embd, T,
                ggml_row_size(gated->type, n_embd) * hc,
                ggml_row_size(gated->type, n_embd) * c));
    }
    mixed = ggml_scale(ctx, mixed, 1.0f / (float)hc);
    ggml_tensor* inject = ggml_mul_mat(ctx, w_inject, xn);

    // ---- projections ----
    ggml_tensor* qkv = ggml_mul_mat(ctx, w_qkv, mixed);        // [conv_dim, T]
    ggml_tensor* z = ggml_mul_mat(ctx, w_gate, mixed);         // [value_dim, T]
    ggml_tensor* beta_raw = ggml_mul_mat(ctx, w_beta, mixed);  // [n_v_heads, T]
    ggml_tensor* alpha_raw = ggml_mul_mat(ctx, w_alpha, mixed);

    // ---- causal depthwise conv over the ring history ----
    // conv_state is [hist, conv_dim]; qkv transposed is [T, conv_dim].
    ggml_tensor* qkv_t = ggml_reshape_3d(ctx, ggml_cont(ctx, ggml_transpose(ctx, qkv)),
            T, conv_dim, 1);
    ggml_tensor* conv_in = ggml_concat(ctx, conv_state, qkv_t, 0); // [hist + T, conv_dim, 1]
    ggml_tensor* conv_out = ggml_silu(ctx, ggml_ssm_conv(ctx, conv_in, w_conv)); // [conv_dim, T, 1]

    // keep the last `hist` columns for the next token
    ggml_tensor* tail = ggml_cont(ctx, ggml_view_3d(ctx, conv_in, hist, conv_dim, 1,
            conv_in->nb[1], conv_in->nb[2], ggml_row_size(conv_in->type, T)));

    // ---- delta net ----
    ggml_tensor* q = ggml_view_3d(ctx, conv_out, head_k_dim, n_k_heads, T,
            ggml_row_size(conv_out->type, head_k_dim), conv_out->nb[1], 0);
    ggml_tensor* k = ggml_view_3d(ctx, conv_out, head_k_dim, n_k_heads, T,
            ggml_row_size(conv_out->type, head_k_dim), conv_out->nb[1],
            ggml_row_size(conv_out->type, key_dim));
    ggml_tensor* v = ggml_view_3d(ctx, conv_out, head_v_dim, n_v_heads, T,
            ggml_row_size(conv_out->type, head_v_dim), conv_out->nb[1],
            ggml_row_size(conv_out->type, 2 * key_dim));

    q = ggml_l2_norm(ctx, ggml_cont(ctx, q), eps);
    k = ggml_l2_norm(ctx, ggml_cont(ctx, k), eps);

    // Repeat q/k up to the value-head count. ggml_repeat TILES (head h reads
    // h % n_k_heads), which is the convention Qwen 3.5's kernel and llama.cpp's
    // non-fused path both use; leaving it to the op's own broadcast produced
    // fluent-looking noise.
    if (n_k_heads != n_v_heads)
    {
        q = ggml_repeat_4d(ctx, q, head_k_dim, n_v_heads, T, 1);
        k = ggml_repeat_4d(ctx, k, head_k_dim, n_v_heads, T, 1);
    }
    q = ggml_reshape_4d(ctx, ggml_cont(ctx, q), head_k_dim, n_v_heads, T, 1);
    k = ggml_reshape_4d(ctx, ggml_cont(ctx, k), head_k_dim, n_v_heads, T, 1);
    v = ggml_reshape_4d(ctx, ggml_cont(ctx, v), head_v_dim, n_v_heads, T, 1);

    // The op scales q internally (llama.cpp passes it unscaled), and a uniform
    // scale here would be absorbed by the RMS norm below in any case.

    ggml_tensor* b4 = ggml_reshape_4d(ctx, ggml_sigmoid(ctx, beta_raw), 1, n_v_heads, T, 1);
    ggml_tensor* g4 = ggml_reshape_4d(ctx,
            ggml_mul(ctx, ggml_softplus(ctx, ggml_add(ctx, alpha_raw, w_dt)), w_a),
            1, n_v_heads, T, 1);
    ggml_tensor* s4 = ggml_reshape_4d(ctx, ssm_state, head_v_dim, head_v_dim, n_v_heads, 1);

    ggml_tensor* gdn_out = ggml_gated_delta_net(ctx, q, k, v, g4, b4, s4, 1);

    const int64_t attn_elems = (int64_t)head_v_dim * n_v_heads * T;
    ggml_tensor* core = ggml_view_3d(ctx, gdn_out, head_v_dim, n_v_heads, T,
            ggml_row_size(gdn_out->type, head_v_dim),
            ggml_row_size(gdn_out->type, head_v_dim * n_v_heads), 0);
    ggml_tensor* new_state = ggml_view_3d(ctx, gdn_out, head_v_dim, head_v_dim, n_v_heads,
            ggml_row_size(gdn_out->type, head_v_dim),
            ggml_row_size(gdn_out->type, head_v_dim * head_v_dim),
            ggml_row_size(gdn_out->type, attn_elems));

    // qwen4exp closes with a SIGMOID gate, where Qwen 3.5 uses SiLU.
    ggml_tensor* normed = ggml_mul(ctx, ggml_rms_norm(ctx, ggml_cont(ctx, core), eps), w_ssmnorm);
    ggml_tensor* zg = ggml_sigmoid(ctx, ggml_reshape_3d(ctx, z, head_v_dim, n_v_heads, T));
    ggml_tensor* out2 = ggml_reshape_2d(ctx, ggml_mul(ctx, normed, zg), value_dim, T);
    ggml_tensor* proj = ggml_mul_mat(ctx, w_out, out2);        // [n_embd, T]

    // ---- hyper-connection scatter ----
    ggml_tensor* wsc = ggml_reshape_3d(ctx, ggml_scale(ctx,
            ggml_sigmoid(ctx, ggml_scale(ctx, inject, 1.0f / (float)hc)), 2.0f), 1, hc, T);
    ggml_tensor* bexp = ggml_repeat_4d(ctx, ggml_reshape_3d(ctx, proj, n_embd, 1, T),
            n_embd, hc, T, 1);
    ggml_tensor* res_out = ggml_reshape_2d(ctx,
            ggml_add(ctx, res3, ggml_mul(ctx, bexp, wsc)), hc_dim, T);

    bnd.add(w_norm, a->hc_norm, (std::size_t)hc_dim * sizeof(float));
    bnd.add(w_down, a->hc_down, (std::size_t)a->hc_down_bytes);
    bnd.add(w_up, a->hc_up, (std::size_t)a->hc_up_bytes);
    bnd.add(w_inject, a->hc_inject, (std::size_t)a->hc_inject_bytes);
    bnd.add(w_qkv, a->qkv, (std::size_t)a->qkv_bytes);
    bnd.add(w_gate, a->gate, (std::size_t)a->gate_bytes);
    bnd.add(w_beta, a->beta, (std::size_t)a->beta_bytes);
    bnd.add(w_alpha, a->alpha, (std::size_t)a->alpha_bytes);
    bnd.add(w_conv, a->conv1d, (std::size_t)d_conv * conv_dim * sizeof(float));
    bnd.add(w_dt, a->ssm_dt, (std::size_t)n_v_heads * sizeof(float));
    bnd.add(w_a, a->ssm_a, (std::size_t)n_v_heads * sizeof(float));
    bnd.add(w_ssmnorm, a->ssm_norm, (std::size_t)head_v_dim * sizeof(float));
    bnd.add(w_out, a->out_proj, (std::size_t)a->out_proj_bytes);

    if (probe != nullptr)
    {
        ggml_tensor* nodes[] = { mixed, qkv, conv_out, gdn_out, proj };
        for (ggml_tensor* t : nodes) { ggml_set_output(t); probe->push_back(t); }
    }

    wb->tail = tail;
    wb->new_state = new_state;
    return res_out;
}

// Attention half: hyper-connection mixer -> joint query|gate -> Q/K norm ->
// partial rotary -> KV append -> (flash) attention -> sigmoid gate -> out
// proj -> scatter.
//
// The KV write is expanded into `graph` HERE, before the caller expands the
// returned residual: nothing in the residual's tree depends on the write -
// k_full is a plain view of the cache and ggml does not treat view aliasing as
// an edge - so node order is the only thing sequencing the write against the
// read, and this token has to be able to attend to itself.
ggml_tensor* q4e_nodes_attn(
    ggml_context* ctx, ggml_cgraph* graph, Q4eBinder& bnd,
    const TSGgmlQwen4ExpAttnArgs* a, ggml_tensor* res_in,
    ggml_tensor* mask, ggml_tensor* pos, ggml_tensor* kv_idx,
    int n_embd, int hc, int hc_low_rank, int T,
    int head_dim, int n_head, int n_head_kv, int kv_capacity, int n_kv_pad,
    int n_rot, float rope_base, float rope_freq_scale, float attn_scale,
    float eps, bool use_flash,
    std::vector<ggml_tensor*>* kv_out,
    std::vector<ggml_tensor*>* probe,
    const int32_t* mrope_sections,
    Q4eAttnArenaIO* arena, ggml_tensor* owned_k, ggml_tensor* owned_v, const Q4eQsaGraph* qsa)
{
    const int hc_dim = hc * n_embd;
    const int q_dim = head_dim * n_head;

    ggml_tensor* w_norm   = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, hc_dim);
    ggml_tensor* w_down   = ggml_new_tensor_2d(ctx, (ggml_type)a->hc_down_type, hc_dim, hc_low_rank);
    ggml_tensor* w_up     = ggml_new_tensor_2d(ctx, (ggml_type)a->hc_up_type, hc_low_rank, hc_dim);
    ggml_tensor* w_inject = ggml_new_tensor_2d(ctx, (ggml_type)a->hc_inject_type, hc_dim, hc);
    ggml_tensor* wq       = ggml_new_tensor_2d(ctx, (ggml_type)a->wq_type, n_embd, q_dim * 2);
    ggml_tensor* wk       = ggml_new_tensor_2d(ctx, (ggml_type)a->wk_type, n_embd, head_dim * n_head_kv);
    ggml_tensor* wv       = ggml_new_tensor_2d(ctx, (ggml_type)a->wv_type, n_embd, head_dim * n_head_kv);
    ggml_tensor* wo       = ggml_new_tensor_2d(ctx, (ggml_type)a->wo_type, q_dim, n_embd);
    ggml_tensor* q_norm_w = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, head_dim);
    ggml_tensor* k_norm_w = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, head_dim);
    // In arena mode the KV lives in the caller's slot-stable arena tensors;
    // the holder-resident cache pair is neither created nor bound.
    ggml_tensor* k_cache  = nullptr;
    ggml_tensor* v_cache  = nullptr;
    if (arena == nullptr)
    {
        k_cache = owned_k != nullptr ? owned_k
            : ggml_new_tensor_3d(ctx, (ggml_type)a->kv_type, head_dim, kv_capacity, n_head_kv);
        v_cache = owned_v != nullptr ? owned_v
            : ggml_new_tensor_3d(ctx, (ggml_type)a->kv_type, head_dim, kv_capacity, n_head_kv);
    }

    // ---- hyper-connection mixer ----
    ggml_tensor* res3 = ggml_reshape_3d(ctx, res_in, n_embd, hc, T);
    ggml_tensor* xn = ggml_mul(ctx,
            ggml_reshape_2d(ctx, ggml_rms_norm(ctx, res3, eps), hc_dim, T), w_norm);
    ggml_tensor* lo = ggml_silu(ctx, ggml_scale(ctx, ggml_mul_mat(ctx, w_down, xn), 1.0f / (float)hc));
    ggml_tensor* gt = ggml_sigmoid(ctx, ggml_mul_mat(ctx, w_up, lo));
    ggml_tensor* gated = ggml_reshape_3d(ctx, ggml_mul(ctx, xn, gt), n_embd, hc, T);
    ggml_tensor* mixed = ggml_cont(ctx, ggml_view_2d(ctx, gated, n_embd, T,
            ggml_row_size(gated->type, n_embd) * hc, 0));
    for (int c = 1; c < hc; ++c)
        mixed = ggml_add(ctx, mixed, ggml_view_2d(ctx, gated, n_embd, T,
                ggml_row_size(gated->type, n_embd) * hc, ggml_row_size(gated->type, n_embd) * c));
    mixed = ggml_scale(ctx, mixed, 1.0f / (float)hc);
    ggml_tensor* inject = ggml_mul_mat(ctx, w_inject, xn);

    if (qsa != nullptr)
        mask = q4e_nodes_qsa_mask(ctx, graph, bnd, *qsa, mixed, mask, kv_idx,
            n_embd, T, n_kv_pad, n_rot, rope_base, rope_freq_scale, eps);

    // ---- q | gate, interleaved per head ----
    ggml_tensor* qg = ggml_mul_mat(ctx, wq, mixed);                 // [q_dim*2, T]
    const std::size_t esz = ggml_element_size(qg);
    ggml_tensor* q = ggml_view_3d(ctx, qg, head_dim, n_head, T,
            esz * head_dim * 2, esz * head_dim * 2 * n_head, 0);
    ggml_tensor* gate = ggml_cont(ctx, ggml_view_3d(ctx, qg, head_dim, n_head, T,
            esz * head_dim * 2, esz * head_dim * 2 * n_head, esz * head_dim));

    q = ggml_mul(ctx, ggml_rms_norm(ctx, ggml_cont(ctx, q), eps), q_norm_w);
    ggml_tensor* k = ggml_reshape_3d(ctx, ggml_mul_mat(ctx, wk, mixed), head_dim, n_head_kv, T);
    k = ggml_mul(ctx, ggml_rms_norm(ctx, k, eps), k_norm_w);
    ggml_tensor* v = ggml_reshape_3d(ctx, ggml_mul_mat(ctx, wv, mixed), head_dim, n_head_kv, T);

    // Partial rotary over the first n_rot dims. IMRoPE reduces to NEOX when every
    // position component is equal, which it is for text - so text graphs keep the
    // exact NEOX path they have always had (byte-stable), and only a graph whose
    // pos tensor carries the 4 IMRoPE sections takes the multi-axis rotation, the
    // same op llama.cpp's qwen4exp runs.
    if (mrope_sections != nullptr)
    {
        int sect[4] = { mrope_sections[0], mrope_sections[1], mrope_sections[2], mrope_sections[3] };
        q = ggml_rope_multi(ctx, q, pos, nullptr, n_rot, sect, GGML_ROPE_TYPE_IMROPE,
                            0, rope_base, rope_freq_scale, 0.0f, 1.0f, 0.0f, 0.0f);
        k = ggml_rope_multi(ctx, k, pos, nullptr, n_rot, sect, GGML_ROPE_TYPE_IMROPE,
                            0, rope_base, rope_freq_scale, 0.0f, 1.0f, 0.0f, 0.0f);
    }
    else
    {
        q = ggml_rope_ext(ctx, q, pos, nullptr, n_rot, 2, 0, rope_base, rope_freq_scale,
                          0.0f, 1.0f, 0.0f, 0.0f);
        k = ggml_rope_ext(ctx, k, pos, nullptr, n_rot, 2, 0, rope_base, rope_freq_scale,
                          0.0f, 1.0f, 0.0f, 0.0f);
    }

    ggml_tensor* attn = nullptr;
    if (arena != nullptr)
    {
        // ---- slot-stable arena KV (T == n_slots, one token per slot) ----
        // ONE absolute-row scatter per K/V for the whole batch; the attention
        // reads views of the set_rows RESULT so the write->read edge is a real
        // src edge (the solo path below relies on node order instead). The
        // caller's mask is [cap, 1, 1, n_slots] over the FULL rounded cap, so
        // the solo path's n_kv_pad rebuild cadence does not apply here.
        const std::size_t kv_row_bytes = ggml_row_size((ggml_type)a->kv_type, head_dim);
        ggml_tensor* k_rows = ggml_reshape_2d(ctx, k, head_dim, (int64_t)n_head_kv * T);
        ggml_tensor* v_rows = ggml_reshape_2d(ctx, v, head_dim, (int64_t)n_head_kv * T);
        arena->k_set = ggml_set_rows(ctx, arena->k_arena, k_rows, arena->kv_idx_abs);
        arena->v_set = ggml_set_rows(ctx, arena->v_arena, v_rows, arena->kv_idx_abs);
        ggml_tensor* k_view = ggml_view_4d(ctx, arena->k_set, head_dim, arena->cap, n_head_kv, T,
                kv_row_bytes,
                (std::size_t)arena->cap * kv_row_bytes,
                (std::size_t)arena->rows_per_slot * kv_row_bytes, 0);
        ggml_tensor* v_view = ggml_view_4d(ctx, arena->v_set, head_dim, arena->cap, n_head_kv, T,
                kv_row_bytes,
                (std::size_t)arena->cap * kv_row_bytes,
                (std::size_t)arena->rows_per_slot * kv_row_bytes, 0);
        ggml_tensor* q_4d = ggml_reshape_4d(ctx, q, head_dim, 1, n_head, T);
        arena->fa = ggml_flash_attn_ext(ctx, q_4d, k_view, v_view, mask,
                                        attn_scale, 0.0f, 0.0f);
        ggml_flash_attn_ext_set_prec(arena->fa, GGML_PREC_F32);
        attn = ggml_reshape_3d(ctx, arena->fa, head_dim, n_head, T);
    }
    else
    {
    // ---- append to the cache ----
    ggml_tensor* k_write = ggml_cont(ctx, ggml_permute(ctx, k, 0, 2, 1, 3)); // [hd, T, kvH]
    ggml_tensor* v_write = ggml_cont(ctx, ggml_permute(ctx, v, 0, 2, 1, 3));
    ggml_tensor* k_full = ggml_view_3d(ctx, k_cache, head_dim, n_kv_pad, n_head_kv,
            k_cache->nb[1], k_cache->nb[2], 0);
    ggml_tensor* v_full = ggml_view_3d(ctx, v_cache, head_dim, n_kv_pad, n_head_kv,
            v_cache->nb[1], v_cache->nb[2], 0);

    // The KV write goes into the graph FIRST - see the function comment.
    ggml_build_forward_expand(graph, ggml_set_rows(ctx, k_cache, k_write, kv_idx));
    ggml_build_forward_expand(graph, ggml_set_rows(ctx, v_cache, v_write, kv_idx));

    // ---- attention ----
    ggml_tensor* q_attn = ggml_cont(ctx, ggml_permute(ctx, q, 0, 2, 1, 3));  // [hd, T, nH]
    if (use_flash)
    {
        // One fused kernel in place of mul_mat -> soft_max -> cont(permute(V)) ->
        // mul_mat -> cont(permute). It never materialises the [n_kv, T, n_head]
        // scores and never copies the whole V window. This is also what llama.cpp
        // runs here. Result lands as [hd, nH, T] - already the layout the gate and
        // the output projection want.
        attn = ggml_flash_attn_ext(ctx, q_attn, k_full, v_full, mask,
                                   attn_scale, 0.0f, 0.0f);
        ggml_flash_attn_ext_set_prec(attn, GGML_PREC_F32);
    }
    else
    {
        ggml_tensor* scores = ggml_mul_mat(ctx, k_full, q_attn);             // [n_kv, T, nH]
        ggml_mul_mat_set_prec(scores, GGML_PREC_F32);
        ggml_tensor* probs = ggml_soft_max_ext(ctx, scores, mask, attn_scale, 0.0f);
        ggml_tensor* v_perm = ggml_cont(ctx, ggml_permute(ctx, v_full, 1, 0, 2, 3));
        attn = ggml_mul_mat(ctx, v_perm, probs);                             // [hd, T, nH]
        attn = ggml_cont(ctx, ggml_permute(ctx, attn, 0, 2, 1, 3));          // [hd, nH, T]
        if (probe != nullptr)
        {
            ggml_set_output(scores); ggml_set_output(probs); ggml_set_output(attn);
            probe->push_back(scores); probe->push_back(probs); probe->push_back(attn);
        }
    }
    }

    // qwen4exp gates the attention output before the output projection.
    attn = ggml_mul(ctx, attn, ggml_sigmoid(ctx, gate));
    ggml_tensor* proj = ggml_mul_mat(ctx, wo, ggml_reshape_2d(ctx, attn, q_dim, T));

    // ---- hyper-connection scatter ----
    ggml_tensor* wsc = ggml_reshape_3d(ctx, ggml_scale(ctx,
            ggml_sigmoid(ctx, ggml_scale(ctx, inject, 1.0f / (float)hc)), 2.0f), 1, hc, T);
    ggml_tensor* bexp = ggml_repeat_4d(ctx, ggml_reshape_3d(ctx, proj, n_embd, 1, T),
            n_embd, hc, T, 1);
    ggml_tensor* res_out = ggml_reshape_2d(ctx,
            ggml_add(ctx, res3, ggml_mul(ctx, bexp, wsc)), hc_dim, T);

    bnd.add(w_norm, a->hc_norm, (std::size_t)hc_dim * sizeof(float));
    bnd.add(w_down, a->hc_down, (std::size_t)a->hc_down_bytes);
    bnd.add(w_up, a->hc_up, (std::size_t)a->hc_up_bytes);
    bnd.add(w_inject, a->hc_inject, (std::size_t)a->hc_inject_bytes);
    bnd.add(wq, a->wq, (std::size_t)a->wq_bytes);
    bnd.add(wk, a->wk, (std::size_t)a->wk_bytes);
    bnd.add(wv, a->wv, (std::size_t)a->wv_bytes);
    bnd.add(wo, a->wo, (std::size_t)a->wo_bytes);
    bnd.add(q_norm_w, a->q_norm, (std::size_t)head_dim * sizeof(float));
    bnd.add(k_norm_w, a->k_norm, (std::size_t)head_dim * sizeof(float));
    // The caches are read AND written, so they need a device buffer that outlives
    // the graph rather than a weights binding. (Arena mode owns no cache pair.)
    if (arena == nullptr)
    {
        if (owned_k == nullptr)
            bnd.add(k_cache, a->k_cache, (std::size_t)a->kv_bytes, GGML_BACKEND_BUFFER_USAGE_ANY);
        if (owned_v == nullptr)
            bnd.add(v_cache, a->v_cache, (std::size_t)a->kv_bytes, GGML_BACKEND_BUFFER_USAGE_ANY);
        if (kv_out != nullptr) { kv_out->push_back(k_cache); kv_out->push_back(v_cache); }
    }

    return res_out;
}

// PLE half: key/query grouped norms -> per-stream gate -> value projection ->
// dilated causal depthwise conv over the persistent history -> residual add.
// The caller owns `conv_hist` ([hc_dim, hist] for the span, a
// [hc_dim, hist, n_streams] arena view for the arena) and its seeding; this
// builder only reads it and produces the tail write-back. With `writes` null
// the write-back cpy is expanded into `graph` right after the residual (the
// span's proven ordering); otherwise it is appended for the caller's
// OUTPUT-flag + expand pass.
ggml_tensor* q4e_nodes_ple(
    ggml_context* ctx, ggml_cgraph* graph, Q4eBinder& bnd,
    const TSGgmlQwen4ExpPleArgs* a, ggml_tensor* res_in, ggml_tensor* ple_emb_in,
    ggml_tensor* conv_hist,
    int n_embd, int hc, int T, int n_streams, float eps,
    std::vector<ggml_tensor*>* writes)
{
    const int hc_dim2 = hc * n_embd;
    const int kern = a->kern;
    const int dil = a->dil;
    const int hist = (kern - 1) * dil;
    const int TT = T * n_streams;   // total residual columns in this call

    ggml_tensor* w_key   = ggml_new_tensor_2d(ctx, (ggml_type)a->key_type, n_embd, hc_dim2);
    ggml_tensor* w_value = ggml_new_tensor_2d(ctx, (ggml_type)a->value_type, n_embd, n_embd);
    ggml_tensor* w_nk    = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, hc_dim2);
    ggml_tensor* w_nq    = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, hc_dim2);
    ggml_tensor* w_nc    = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, hc_dim2);
    ggml_tensor* w_ct    = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, hc_dim2, kern);

    // key/query grouped norms: normalise each stream, scale the full row.
    auto gnorm = [&](ggml_tensor* x, ggml_tensor* w) {
        ggml_tensor* x3 = ggml_reshape_3d(ctx, x, n_embd, hc, TT);
        ggml_tensor* nx = ggml_reshape_2d(ctx, ggml_rms_norm(ctx, x3, eps), hc_dim2, TT);
        return ggml_mul(ctx, nx, w);
    };

    ggml_tensor* keyn = gnorm(ggml_mul_mat(ctx, w_key, ple_emb_in), w_nk);   // [hc_dim, TT]
    ggml_tensor* qryn = gnorm(res_in, w_nq);

    // Per-stream dot, scaled, signed-sqrt, sigmoid: the PLE gate.
    ggml_tensor* prod = ggml_reshape_3d(ctx, ggml_mul(ctx, keyn, qryn), n_embd, hc, TT);
    ggml_tensor* sdot = ggml_scale(ctx, ggml_sum_rows(ctx, prod),
            1.0f / std::sqrt((float)n_embd));                                 // [1, hc, TT]
    ggml_tensor* sg   = ggml_sgn(ctx, sdot);
    ggml_tensor* mag  = ggml_sqrt(ctx, ggml_clamp(ctx, ggml_mul(ctx, sdot, sg),
            1e-6f, 3.0e38f));
    ggml_tensor* gate = ggml_sigmoid(ctx, ggml_mul(ctx, sg, mag));            // [1, hc, TT]

    ggml_tensor* val  = ggml_mul_mat(ctx, w_value, ple_emb_in);               // [n_embd, TT]
    ggml_tensor* v3   = ggml_repeat_4d(ctx,
            ggml_reshape_3d(ctx, val, n_embd, 1, TT), n_embd, hc, TT, 1);
    ggml_tensor* gated = ggml_reshape_2d(ctx, ggml_mul(ctx, v3, gate), hc_dim2, TT);

    // Dilated causal depthwise conv over the conv-normed gate output.
    ggml_tensor* normc = gnorm(gated, w_nc);
    ggml_tensor* conv = nullptr;
    ggml_tensor* tail = nullptr;
    if (n_streams == 1)
    {
        ggml_tensor* padded = (conv_hist != nullptr)
                ? ggml_concat(ctx, conv_hist, normc, 1)                       // [hc_dim, hist+T]
                : normc;
        ggml_tensor* acc = nullptr;
        for (int kk = 0; kk < kern; ++kk)
        {
            // tap kk reads (kern-1-kk) dilated positions back: with hist rows
            // of history in front, that is a plain offset of kk*dil rows.
            ggml_tensor* slice = ggml_view_2d(ctx, padded, hc_dim2, TT,
                    padded->nb[1], (std::size_t)(kk * dil) * padded->nb[1]);
            ggml_tensor* wk = ggml_view_2d(ctx, w_ct, hc_dim2, 1,
                    w_ct->nb[1], (std::size_t)kk * w_ct->nb[1]);
            ggml_tensor* term = ggml_mul(ctx, slice, wk);
            acc = (acc == nullptr) ? term : ggml_add(ctx, acc, term);
        }
        conv = ggml_silu(ctx, acc);                                           // [hc_dim, TT]
        if (conv_hist != nullptr)
            tail = ggml_view_2d(ctx, padded, hc_dim2, hist,
                    padded->nb[1], (std::size_t)TT * padded->nb[1]);
    }
    else
    {
        // Arena: T == 1 per stream, per-stream history planes. Same taps with
        // one extra (stream) axis; the tap weight broadcasts across it.
        ggml_tensor* normc3 = ggml_reshape_3d(ctx, normc, hc_dim2, T, n_streams);
        ggml_tensor* padded = ggml_concat(ctx, conv_hist, normc3, 1);         // [hc_dim, hist+T, S]
        ggml_tensor* acc = nullptr;
        for (int kk = 0; kk < kern; ++kk)
        {
            ggml_tensor* slice = ggml_view_3d(ctx, padded, hc_dim2, T, n_streams,
                    padded->nb[1], padded->nb[2], (std::size_t)(kk * dil) * padded->nb[1]);
            ggml_tensor* wk = ggml_view_2d(ctx, w_ct, hc_dim2, 1,
                    w_ct->nb[1], (std::size_t)kk * w_ct->nb[1]);
            ggml_tensor* term = ggml_mul(ctx, slice, wk);
            acc = (acc == nullptr) ? term : ggml_add(ctx, acc, term);
        }
        conv = ggml_reshape_2d(ctx, ggml_silu(ctx, acc), hc_dim2, TT);
        tail = ggml_view_3d(ctx, padded, hc_dim2, hist, n_streams,
                padded->nb[1], padded->nb[2], (std::size_t)T * padded->nb[1]);
    }

    // res += gated + conv
    ggml_tensor* res_out = ggml_add(ctx, ggml_add(ctx, res_in, gated), conv);
    ggml_build_forward_expand(graph, res_out);
    if (tail != nullptr)
    {
        // Keep the last `hist` rows for the next step. Expanded here, the
        // residual expand above has already ordered every read of the state
        // first; collected via `writes`, the caller's expand pass (in build
        // order) provides the same ordering.
        ggml_tensor* wr = ggml_cpy(ctx, tail, conv_hist);
        if (writes == nullptr)
            ggml_build_forward_expand(graph, wr);
        else
            writes->push_back(wr);
    }

    bnd.add(w_key, a->key_w, (std::size_t)a->key_bytes);
    bnd.add(w_value, a->value_w, (std::size_t)a->value_bytes);
    bnd.add(w_nk, a->norm_key, (std::size_t)hc_dim2 * sizeof(float));
    bnd.add(w_nq, a->norm_query, (std::size_t)hc_dim2 * sizeof(float));
    bnd.add(w_nc, a->norm_conv, (std::size_t)hc_dim2 * sizeof(float));
    bnd.add(w_ct, a->conv1d_t, (std::size_t)hc_dim2 * kern * sizeof(float));
    return res_out;
}

// Head: the final hyper-connection mixer (which IS the output norm) + LM head
// over `res_last` [hc_dim, T], T-generic so the arena can run it with
// T = n_slots (every slot's token is "last"). Returns logits [vocab, T]; the
// caller flags and expands it.
ggml_tensor* q4e_nodes_head(
    ggml_context* ctx, Q4eBinder& bnd,
    const TSGgmlQwen4ExpHeadArgs* a, ggml_tensor* res_last,
    int n_embd, int hc, int hc_low_rank, int T, float eps)
{
    const int hc_dim = hc * n_embd;
    ggml_tensor* w_fnorm = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, hc_dim);
    ggml_tensor* w_fdown = ggml_new_tensor_2d(ctx, (ggml_type)a->hc_down_type, hc_dim, hc_low_rank);
    ggml_tensor* w_fup   = ggml_new_tensor_2d(ctx, (ggml_type)a->hc_up_type, hc_low_rank, hc_dim);
    ggml_tensor* w_head  = ggml_new_tensor_2d(ctx, (ggml_type)a->head_type, n_embd, a->vocab);

    ggml_tensor* res3f = ggml_reshape_3d(ctx, res_last, n_embd, hc, T);
    ggml_tensor* xnf = ggml_mul(ctx,
            ggml_reshape_2d(ctx, ggml_rms_norm(ctx, res3f, eps), hc_dim, T), w_fnorm);
    ggml_tensor* lof = ggml_silu(ctx, ggml_scale(ctx,
            ggml_mul_mat(ctx, w_fdown, xnf), 1.0f / (float)hc));
    ggml_tensor* gtf = ggml_sigmoid(ctx, ggml_mul_mat(ctx, w_fup, lof));
    ggml_tensor* gatedf = ggml_reshape_3d(ctx, ggml_mul(ctx, xnf, gtf), n_embd, hc, T);
    ggml_tensor* mixedf = ggml_cont(ctx, ggml_view_2d(ctx, gatedf, n_embd, T,
            ggml_row_size(gatedf->type, n_embd) * hc, 0));
    for (int c = 1; c < hc; ++c)
        mixedf = ggml_add(ctx, mixedf, ggml_view_2d(ctx, gatedf, n_embd, T,
                ggml_row_size(gatedf->type, n_embd) * hc,
                ggml_row_size(gatedf->type, n_embd) * c));
    mixedf = ggml_scale(ctx, mixedf, 1.0f / (float)hc);

    ggml_tensor* logits = ggml_mul_mat(ctx, w_head, mixedf);                  // [vocab, T]

    bnd.add(w_fnorm, a->hc_norm, (std::size_t)hc_dim * sizeof(float));
    bnd.add(w_fdown, a->hc_down, (std::size_t)a->hc_down_bytes);
    bnd.add(w_fup, a->hc_up, (std::size_t)a->hc_up_bytes);
    bnd.add(w_head, a->head, (std::size_t)a->head_bytes);
    return logits;
}

// ============================================================================
// Per-layer entry points (the fallback path).
// ============================================================================

// res_data is the 4-wide residual, [hc * n_embd, n_tokens] row-major on the
// host, read and written in place.
TSG_EXPORT int TSGgml_Qwen4ExpFfnBlock(
    const TSGgmlQwen4ExpFfnArgs* a,
    void* res_data,
    int n_embd, int hc, int hc_low_rank, int n_tokens,
    int n_expert, int n_expert_used, int n_ff, int n_ff_sh,
    float eps, int cache_slot, int res_resident)
{
    try
    {
        if (a == nullptr || res_data == nullptr)
        {
            set_last_error("qwen4exp FFN block: null args.");
            return 0;
        }
        if (!ensure_backend())
            return 0;

        const int hc_dim = hc * n_embd;
        const int T = n_tokens;
        const std::size_t res_bytes = (std::size_t)hc_dim * T * sizeof(float);

        // Replay: same layer, same shape, same weights.
        Qwen4ExpFfnCache* slot = (cache_slot >= 0 && cache_slot < kQwen4ExpMaxSlots)
            ? &g_q4e_ffn[cache_slot] : nullptr;
        if (slot != nullptr && slot->valid && slot->n_tokens == T && slot->hc_dim == hc_dim
            && slot->sig == (const void*)a && slot->res_resident == res_resident
            && q4e_refresh_bindings(slot, ggml_backend_get_device(g_backend)))
        {
            if (!res_resident) ggml_backend_tensor_set(slot->res_in, res_data, 0, res_bytes);
            q4e_note(0, false);
            if (graph_compute_profiled(g_backend, slot->graph, kQwen4ExpFfnKernel) != GGML_STATUS_SUCCESS)
            {
                slot->reset();
                set_last_error("qwen4exp FFN block: replay failed.");
                return 0;
            }
            if (res_resident)
            {
                // Chain on the device: the next layer reads what this one wrote.
                ggml_backend_tensor_copy(slot->res_out, slot->res_in);
            }
            else
            {
                ggml_backend_tensor_get(slot->res_out, res_data, 0, res_bytes);
            }
            return 1;
        }
        if (slot != nullptr) slot->reset();

        ggml_init_params ip{};
        ip.mem_size = ggml_tensor_overhead() * 512 + ggml_graph_overhead();
        ip.mem_buffer = nullptr;
        ip.no_alloc = true;
        ggml_context* ctx = ggml_init(ip);
        if (ctx == nullptr)
        {
            set_last_error("qwen4exp FFN block: ggml_init failed.");
            return 0;
        }

        ggml_tensor* res_in = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, hc_dim, T);
        ggml_set_input(res_in);
        if (res_resident)
        {
            if (!q4e_res_ensure(res_bytes) ||
                ggml_backend_tensor_alloc(g_q4e_res_buf, res_in,
                        ggml_backend_buffer_get_base(g_q4e_res_buf)) != GGML_STATUS_SUCCESS)
            {
                ggml_free(ctx);
                set_last_error("qwen4exp FFN block: failed to bind the residual buffer.");
                return 0;
            }
        }

        Q4eBinder binder{ggml_backend_get_device(g_backend)};
        ggml_tensor* res_out = q4e_nodes_ffn(ctx, binder, a, res_in,
                n_embd, hc, hc_low_rank, T, n_expert, n_expert_used, n_ff, n_ff_sh, eps);
        if (res_out == nullptr)
        {
            ggml_free(ctx);
            set_last_error("qwen4exp FFN block: failed to build the graph.");
            return 0;
        }
        ggml_set_output(res_out);

        ggml_cgraph* graph = ggml_new_graph(ctx);
        if (q4e_graph_uid_enabled()) graph->uid = q4e_next_graph_uid();
        ggml_build_forward_expand(graph, res_out);

        ggml_gallocr_t alloc = ggml_gallocr_new(ggml_backend_get_default_buffer_type(g_backend));
        if (alloc == nullptr || !ggml_gallocr_alloc_graph(alloc, graph))
        {
            if (alloc) ggml_gallocr_free(alloc);
            ggml_free(ctx);
            set_last_error("qwen4exp FFN block: failed to allocate graph tensors.");
            return 0;
        }

        binder.flush();
        if (!res_resident) ggml_backend_tensor_set(res_in, res_data, 0, res_bytes);

        q4e_note(0, true);
        if (graph_compute_profiled(g_backend, graph, kQwen4ExpFfnKernel) != GGML_STATUS_SUCCESS)
        {
            ggml_gallocr_free(alloc);
            ggml_free(ctx);
            set_last_error("qwen4exp FFN block: graph compute failed.");
            return 0;
        }

        if (res_resident)
        {
            ggml_backend_tensor_copy(res_out, res_in);
        }
        else
        {
            ggml_backend_tensor_get(res_out, res_data, 0, res_bytes);
        }

        if (slot != nullptr)
        {
            // Hand the graph to the cache. The weight bindings and the gallocr plan
            // stay valid as long as the descriptor and the shape do, which the
            // replay check above enforces.
            slot->ctx = ctx;
            slot->graph = graph;
            slot->alloc = alloc;
            slot->res_in = res_in;
            slot->res_out = res_out;
            slot->n_tokens = T;
            slot->hc_dim = hc_dim;
            slot->sig = (const void*)a;
            slot->res_resident = res_resident;
            slot->rebinds = std::move(binder.cached);
            slot->valid = true;
            return 1;
        }

        ggml_gallocr_free(alloc);
        ggml_free(ctx);
        return 1;
    }
    catch (const std::exception& e)
    {
        set_last_error(std::string("qwen4exp FFN block: ") + e.what());
        return 0;
    }
    catch (...)
    {
        set_last_error("qwen4exp FFN block: unknown error.");
        return 0;
    }
}

TSG_EXPORT int TSGgml_Qwen4ExpGdnBlock(
    const TSGgmlQwen4ExpGdnArgs* a,
    void* res_data,
    int n_embd, int hc, int hc_low_rank, int n_tokens,
    int head_k_dim, int head_v_dim, int n_k_heads, int n_v_heads, int d_conv,
    float eps, int cache_slot, int res_resident)
{
    try
    {
        if (a == nullptr || res_data == nullptr) { set_last_error("qwen4exp GDN block: null args."); return 0; }
        if (!ensure_backend()) return 0;
        tsg_q4earena::on_external_touch(a->conv_state);

        const int hc_dim = hc * n_embd;
        const int T = n_tokens;
        const int key_dim = head_k_dim * n_k_heads;
        const int value_dim = head_v_dim * n_v_heads;
        const int conv_dim = key_dim * 2 + value_dim;
        const int hist = d_conv - 1;
        const std::size_t res_bytes = (std::size_t)hc_dim * T * sizeof(float);

        Qwen4ExpFfnCache* slot = (cache_slot >= 0 && cache_slot < kQwen4ExpMaxSlots)
            ? &g_q4e_gdn[cache_slot] : nullptr;
        if (slot != nullptr && slot->valid && slot->n_tokens == T && slot->hc_dim == hc_dim
            && slot->sig == (const void*)a
            && q4e_refresh_bindings(slot, ggml_backend_get_device(g_backend)))
        {
            ggml_backend_tensor_set(slot->res_in, res_data, 0, res_bytes);
            q4e_note(1, false);
            if (graph_compute_profiled(g_backend, slot->graph, kQwen4ExpGdnKernel) != GGML_STATUS_SUCCESS)
            { slot->reset(); set_last_error("qwen4exp GDN block: replay failed."); return 0; }
            tsg::sync_backend(g_backend);
            if (slot->conv_out != nullptr)
            {
                ggml_backend_tensor_copy(slot->conv_out, slot->conv_in);
                ggml_backend_tensor_copy(slot->ssm_out, slot->ssm_in);
            }
            if (res_resident) ggml_backend_tensor_copy(slot->res_out, slot->res_in);
            else ggml_backend_tensor_get(slot->res_out, res_data, 0, res_bytes);
            return 1;
        }
        // Rebuild the graph; the state buffer below is untouched by that.
        if (slot != nullptr) slot->reset_graph();

        ggml_init_params ip{};
        ip.mem_size = ggml_tensor_overhead() * 512 + ggml_graph_overhead();
        ip.mem_buffer = nullptr;
        ip.no_alloc = true;
        ggml_context* ctx = ggml_init(ip);
        if (ctx == nullptr) { set_last_error("qwen4exp GDN block: ggml_init failed."); return 0; }

        ggml_tensor* res_in = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, hc_dim, T);
        ggml_set_input(res_in);
        if (res_resident)
        {
            if (!q4e_res_ensure(res_bytes) ||
                ggml_backend_tensor_alloc(g_q4e_res_buf, res_in,
                        ggml_backend_buffer_get_base(g_q4e_res_buf)) != GGML_STATUS_SUCCESS)
            {
                ggml_free(ctx);
                set_last_error("qwen4exp GDN block: failed to bind the residual buffer.");
                return 0;
            }
        }

        // The state is read through *_in and written through a SEPARATE *_out, copied
        // back device-to-device after the graph runs - the per-layer path's proven
        // dataflow, kept as is. (The token span writes the state in place instead,
        // with node order sequencing the write behind the read.)
        ggml_tensor* conv_state = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, hist, conv_dim, 1);
        ggml_tensor* ssm_state  = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, head_v_dim, head_v_dim, n_v_heads);
        ggml_tensor* conv_state_out = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, hist, conv_dim, 1);
        ggml_tensor* ssm_state_out  = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, head_v_dim, head_v_dim, n_v_heads);
        ggml_set_input(conv_state);      ggml_set_input(ssm_state);
        ggml_set_output(conv_state_out); ggml_set_output(ssm_state_out);

        Q4eBinder binder{ggml_backend_get_device(g_backend)};
        Q4eGdnWriteback wb{};
        ggml_tensor* res_out = q4e_nodes_gdn(ctx, binder, a, res_in,
                conv_state, ssm_state,
                n_embd, hc, hc_low_rank, T,
                head_k_dim, head_v_dim, n_k_heads, n_v_heads, d_conv, eps, &wb);
        ggml_set_output(res_out);

        ggml_cgraph* graph = ggml_new_graph(ctx);
        if (q4e_graph_uid_enabled()) graph->uid = q4e_next_graph_uid();
        ggml_build_forward_expand(graph, res_out);
        // state write-back rides the same graph
        ggml_build_forward_expand(graph, ggml_cpy(ctx, wb.tail, conv_state_out));
        ggml_build_forward_expand(graph, ggml_cpy(ctx, wb.new_state, ssm_state_out));

        // Give the two *_in state tensors their OWN device buffer so gallocr never
        // owns them and a graph rebuild cannot disturb them. Carrying the state across
        // a rebuild by copying was the alternative and it did not survive contact:
        // one fused layer was enough to derail the model.
        const std::size_t conv_bytes = (std::size_t)hist * conv_dim * sizeof(float);
        const std::size_t ssm_bytes = (std::size_t)head_v_dim * head_v_dim * n_v_heads * sizeof(float);
        const std::size_t ssm_off = (conv_bytes + 255) & ~(std::size_t)255;
        bool zero_state = true;
        Q4eSeqStateEntry* st = nullptr;
        if (slot != nullptr)
        {
            st = q4e_seq_state(a->conv_state, ssm_off + ssm_bytes);
            if (st == nullptr)
            {
                ggml_free(ctx);
                set_last_error("qwen4exp GDN block: failed to allocate the state buffer.");
                return 0;
            }
            zero_state = !st->ready;
            std::uint8_t* base = (std::uint8_t*)ggml_backend_buffer_get_base(st->buf);
            if (ggml_backend_tensor_alloc(st->buf, conv_state, base) != GGML_STATUS_SUCCESS ||
                ggml_backend_tensor_alloc(st->buf, ssm_state, base + ssm_off) != GGML_STATUS_SUCCESS)
            {
                ggml_free(ctx);
                set_last_error("qwen4exp GDN block: failed to bind the state buffer.");
                return 0;
            }
        }

        // Seed the state only when the buffer is new; a rebuild keeps what is there.
        if (zero_state)
        {
            binder.upload_list.push_back({conv_state, a->conv_state, conv_bytes});
            binder.upload_list.push_back({ssm_state, a->ssm_state, ssm_bytes});
        }

        ggml_gallocr_t alloc = ggml_gallocr_new(ggml_backend_get_default_buffer_type(g_backend));
        if (alloc == nullptr || !ggml_gallocr_alloc_graph(alloc, graph))
        {
            if (alloc) ggml_gallocr_free(alloc);
            ggml_free(ctx);
            set_last_error("qwen4exp GDN block: failed to allocate graph tensors.");
            return 0;
        }

        binder.flush();
        if (st != nullptr) st->ready = true;
        if (!res_resident) ggml_backend_tensor_set(res_in, res_data, 0, res_bytes);

        q4e_note(1, true);
        if (graph_compute_profiled(g_backend, graph, kQwen4ExpGdnKernel) != GGML_STATUS_SUCCESS)
        {
            ggml_gallocr_free(alloc); ggml_free(ctx);
            set_last_error("qwen4exp GDN block: graph compute failed.");
            return 0;
        }

        tsg::sync_backend(g_backend);
        ggml_backend_tensor_copy(conv_state_out, conv_state);
        ggml_backend_tensor_copy(ssm_state_out, ssm_state);
        if (res_resident) ggml_backend_tensor_copy(res_out, res_in);
        else ggml_backend_tensor_get(res_out, res_data, 0, res_bytes);

        if (slot != nullptr)
        {
            slot->ctx = ctx; slot->graph = graph; slot->alloc = alloc;
            slot->res_in = res_in; slot->res_out = res_out;
            slot->conv_in = conv_state; slot->conv_out = conv_state_out;
            slot->ssm_in = ssm_state; slot->ssm_out = ssm_state_out;
            slot->n_tokens = T; slot->hc_dim = hc_dim; slot->sig = (const void*)a;
            slot->res_resident = res_resident;
            slot->rebinds = std::move(binder.cached);
            slot->valid = true;
            return 1;
        }
        ggml_gallocr_free(alloc); ggml_free(ctx);
        return 1;
    }
    catch (const std::exception& e)
    { set_last_error(std::string("qwen4exp GDN block: ") + e.what()); return 0; }
    catch (...)
    { set_last_error("qwen4exp GDN block: unknown error."); return 0; }
}

// mask_data is [n_kv_pad, T] F16, built host-side: 0 where token t may attend to
// cell j, -inf otherwise (the C# side pads with the same predicate and stride).
TSG_EXPORT int TSGgml_Qwen4ExpAttnBlock(
    const TSGgmlQwen4ExpAttnArgs* a,
    void* res_data,
    const void* mask_data,
    int n_embd, int hc, int hc_low_rank, int n_tokens,
    int head_dim, int n_head, int n_head_kv, int kv_capacity, int n_kv, int position,
    int n_rot, float rope_base, float rope_freq_scale, float attn_scale,
    float eps, int cache_slot, int res_resident)
{
    try
    {
        if (a == nullptr || res_data == nullptr) { set_last_error("qwen4exp attn block: null args."); return 0; }
        if (!ensure_backend()) return 0;
        tsg_q4earena::on_external_touch(a->k_cache);
        tsg_q4earena::on_external_touch(a->v_cache);

        const int hc_dim = hc * n_embd;
        const int T = n_tokens;
        const std::size_t res_bytes = (std::size_t)hc_dim * T * sizeof(float);
        const bool use_flash = q4e_flash_attn_ok(a->kv_type, head_dim);
        const int n_kv_pad = q4e_pad_kv(n_kv, kv_capacity, use_flash);
        const std::size_t mask_bytes = (std::size_t)n_kv_pad * T * sizeof(uint16_t);

        Qwen4ExpFfnCache* slot = (cache_slot >= 0 && cache_slot < kQwen4ExpMaxSlots)
            ? &g_q4e_attn[cache_slot] : nullptr;

        // Keyed on the PADDED width, so the topology only moves once every stride
        // tokens instead of every token. Position and the KV write row reach the graph
        // as inputs, so their values change without the shape moving.
        if (slot != nullptr && slot->valid && slot->n_tokens == T && slot->hc_dim == hc_dim
            && slot->sig == (const void*)a && slot->res_resident == res_resident
            && slot->n_kv == n_kv_pad
            && q4e_refresh_bindings(slot, ggml_backend_get_device(g_backend)))
        {
            if (!res_resident) ggml_backend_tensor_set(slot->res_in, res_data, 0, res_bytes);
            ggml_backend_tensor_set(slot->mask, mask_data, 0, mask_bytes);
            q4e_set_attn_indices(slot->pos, slot->kv_idx, T, position);
            q4e_note(2, false);
            if (graph_compute_profiled(g_backend, slot->graph, kQwen4ExpAttnKernel) != GGML_STATUS_SUCCESS)
            { slot->reset_graph(); set_last_error("qwen4exp attn block: replay failed."); return 0; }
            if (res_resident) ggml_backend_tensor_copy(slot->res_out, slot->res_in);
            else ggml_backend_tensor_get(slot->res_out, res_data, 0, res_bytes);
            return 1;
        }
        if (slot != nullptr) slot->reset_graph();

        ggml_init_params ip{};
        ip.mem_size = ggml_tensor_overhead() * 512 + ggml_graph_overhead();
        ip.mem_buffer = nullptr;
        ip.no_alloc = true;
        ggml_context* ctx = ggml_init(ip);
        if (ctx == nullptr) { set_last_error("qwen4exp attn block: ggml_init failed."); return 0; }

        ggml_tensor* res_in = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, hc_dim, T);
        ggml_set_input(res_in);
        if (res_resident)
        {
            if (!q4e_res_ensure(res_bytes) ||
                ggml_backend_tensor_alloc(g_q4e_res_buf, res_in,
                        ggml_backend_buffer_get_base(g_q4e_res_buf)) != GGML_STATUS_SUCCESS)
            {
                ggml_free(ctx);
                set_last_error("qwen4exp attn block: failed to bind the residual buffer.");
                return 0;
            }
        }

        ggml_tensor* mask = ggml_new_tensor_2d(ctx, GGML_TYPE_F16, n_kv_pad, T);
        ggml_set_input(mask);
        ggml_tensor* pos = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, T);
        ggml_set_input(pos);
        // The rows this step writes, as an INPUT rather than a baked view offset.
        ggml_tensor* kv_idx = ggml_new_tensor_1d(ctx, GGML_TYPE_I64, T);
        ggml_set_input(kv_idx);

        // The builder expands the KV write into the graph before we expand the
        // residual below, so the graph exists first.
        ggml_cgraph* graph = ggml_new_graph(ctx);
        if (q4e_graph_uid_enabled()) graph->uid = q4e_next_graph_uid();

        Q4eBinder binder{ggml_backend_get_device(g_backend)};
        std::vector<ggml_tensor*> kv_tensors;
        ggml_tensor* res_out = q4e_nodes_attn(ctx, graph, binder, a, res_in,
                mask, pos, kv_idx,
                n_embd, hc, hc_low_rank, T,
                head_dim, n_head, n_head_kv, kv_capacity, n_kv_pad,
                n_rot, rope_base, rope_freq_scale, attn_scale, eps, use_flash,
                &kv_tensors);
        ggml_set_output(res_out);
        ggml_build_forward_expand(graph, res_out);

        ggml_gallocr_t alloc = ggml_gallocr_new(ggml_backend_get_default_buffer_type(g_backend));
        if (alloc == nullptr || !ggml_gallocr_alloc_graph(alloc, graph))
        {
            if (alloc) ggml_gallocr_free(alloc);
            ggml_free(ctx);
            set_last_error("qwen4exp attn block: failed to allocate graph tensors.");
            return 0;
        }

        binder.flush();
        // Same pad-reads-uninitialised-memory hazard as the span; see the comment
        // there. Zero the device K/V at the start of the sequence.
        if (position == 0)
            for (ggml_tensor* t : kv_tensors)
                ggml_backend_tensor_memset(t, 0, 0, ggml_nbytes(t));
        if (!res_resident) ggml_backend_tensor_set(res_in, res_data, 0, res_bytes);
        ggml_backend_tensor_set(mask, mask_data, 0, mask_bytes);
        q4e_set_attn_indices(pos, kv_idx, T, position);

        q4e_note(2, true);
        if (graph_compute_profiled(g_backend, graph, kQwen4ExpAttnKernel) != GGML_STATUS_SUCCESS)
        {
            ggml_gallocr_free(alloc); ggml_free(ctx);
            set_last_error("qwen4exp attn block: graph compute failed.");
            return 0;
        }

        if (res_resident) ggml_backend_tensor_copy(res_out, res_in);
        else ggml_backend_tensor_get(res_out, res_data, 0, res_bytes);

        if (slot != nullptr)
        {
            slot->ctx = ctx; slot->graph = graph; slot->alloc = alloc;
            slot->res_in = res_in; slot->res_out = res_out; slot->mask = mask;
            slot->pos = pos; slot->kv_idx = kv_idx;
            slot->n_tokens = T; slot->hc_dim = hc_dim; slot->sig = (const void*)a;
            slot->res_resident = res_resident; slot->n_kv = n_kv_pad;
            slot->rebinds = std::move(binder.cached);
            slot->valid = true;
            return 1;
        }
        ggml_gallocr_free(alloc); ggml_free(ctx);
        return 1;
    }
    catch (const std::exception& e)
    { set_last_error(std::string("qwen4exp attn block: ") + e.what()); return 0; }
    catch (...)
    { set_last_error("qwen4exp attn block: unknown error."); return 0; }
}

// ============================================================================
// The token span: layers [layer_begin, layer_end) - both halves each - as ONE
// graph. With the PLE layer the only host interruption, a decode token is two
// of these calls instead of 96 per-layer ones: one residual upload, one graph
// launch, one download per span, and a single stable topology for ggml-cuda's
// CUDA graph capture instead of 96 alternating ones.
//
// ffn/gdn/attn are the BASE pointers of the full per-layer descriptor arrays
// (pinned on the C# side), indexed here by absolute layer id. kinds[il] != 0
// marks a recurrent (GDN) layer.
//
// GDN state lives in the per-layer slots' state buffers (g_q4e_gdn[il]), so
// the span and the per-layer fallback read and write the SAME state and a
// fallback mid-sequence stays coherent. The span writes the state in place -
// cpy(tail -> conv_state) expanded after the nodes that read conv_state, so
// node order sequences the write behind the read.
// ============================================================================
// TSGgmlQwen4ExpPleArgs / TSGgmlQwen4ExpHeadArgs live in ggml_ops_internal.h
// (shared with the arena kernel).

static int q4e_token_span_impl(
    const TSGgmlQwen4ExpFfnArgs* ffn,
    const TSGgmlQwen4ExpGdnArgs* gdn,
    const TSGgmlQwen4ExpAttnArgs* attn,
    const unsigned char* kinds,
    int layer_begin, int layer_end,
    void* res_data,
    const void* mask_data,
    int n_embd, int hc, int hc_low_rank, int n_tokens,
    int head_k_dim, int head_v_dim, int n_k_heads, int n_v_heads, int d_conv,
    int head_dim, int n_head, int n_head_kv, int kv_capacity, int n_kv, int position,
    int n_rot, float rope_base, float rope_freq_scale, float attn_scale,
    int n_expert, int n_expert_used, int n_ff, int n_ff_sh,
    float eps, int cache_slot, int first_ffn_only,
    const TSGgmlQwen4ExpHeadArgs* head, void* logits_out,
    const TSGgmlQwen4ExpPleArgs* ple, int ple_layer, const void* ple_emb,
    const int* mrope_pos, const int* mrope_sections, int rope_position,
    int device, void* hidden_out, int logits_rows,
    const TSGgmlQwen4ExpQsaArgs* qsa, const int32_t* qsa_positions, int qsa_position_count)
{
    try
    {
        if (ffn == nullptr || gdn == nullptr || attn == nullptr || kinds == nullptr
            || res_data == nullptr || layer_begin < 0 || layer_end <= layer_begin
            || layer_end > kQwen4ExpMaxSlots)
        {
            set_last_error("qwen4exp token span: bad args.");
            return 0;
        }
        if (n_tokens <= 0 || n_embd <= 0 || hc <= 0
            || (logits_rows != 1 && logits_rows != n_tokens)
            || (hidden_out != nullptr && head == nullptr))
        {
            set_last_error("qwen4exp token span: invalid speculative output shape.");
            return 0;
        }
        // LAYER SPLIT: run this span's layers on their own GPU. Everything the
        // span touches - the persisted graph slot, the resident weight copies,
        // the KV device copies, the GDN/PLE state buffers, the residual buffer -
        // is selected by the active rank, so the scope is the whole mechanism.
        tsg::ScopedRank q4e_rank(q4e_resolve_device(device));
        if (!ensure_backend()) return 0;
        // Retire any arena slots holding this holder's caches or recurrent
        // state: the span reads the resident KV copies and the seq-state
        // buffers, whose newest truth lives in the arena until flushed. Any
        // registered pointer of a slot retires the whole slot; no-ops while
        // nothing is registered, so solo-only serving is untouched.
        for (int il = layer_begin; il < layer_end; ++il)
        {
            if (kinds[il] == 0)
                tsg_q4earena::on_external_touch(attn[il].k_cache);
            else
                tsg_q4earena::on_external_touch(gdn[il].conv_state);
        }
        if (ple != nullptr)
            tsg_q4earena::on_external_touch(ple->conv_state);
        if ((head != nullptr) != (logits_out != nullptr))
        {
            set_last_error("qwen4exp token span: head and logits_out come together.");
            return 0;
        }
        const bool has_ple = ple != nullptr && ple_layer >= layer_begin && ple_layer < layer_end;
        const bool use_mrope = mrope_pos != nullptr && mrope_sections != nullptr;
        if (has_ple && ple_emb == nullptr)
        {
            set_last_error("qwen4exp token span: the PLE block needs the gathered rows.");
            return 0;
        }

        const int hc_dim = hc * n_embd;
        const int T = n_tokens;
        const std::size_t res_bytes = (std::size_t)hc_dim * T * sizeof(float);

        bool has_attn = false;
        for (int il = layer_begin; il < layer_end; ++il)
        {
            if (il == layer_begin && first_ffn_only != 0) continue;
            if (kinds[il] == 0) { has_attn = true; break; }
        }
        if (has_attn && mask_data == nullptr)
        {
            set_last_error("qwen4exp token span: an attention layer needs the mask.");
            return 0;
        }

        // One padded window for every attention layer in the span; they share one
        // mask, one position tensor and one write-row tensor.
        bool use_flash = false;
        int n_kv_pad = 0;
        if (has_attn)
        {
            int first_attn = (first_ffn_only != 0) ? layer_begin + 1 : layer_begin;
            while (kinds[first_attn] != 0) ++first_attn;
            use_flash = q4e_flash_attn_ok(attn[first_attn].kv_type, head_dim);
            n_kv_pad = q4e_pad_kv(n_kv, kv_capacity, use_flash);
        }
        const std::size_t mask_bytes = (std::size_t)n_kv_pad * T * sizeof(uint16_t);
        std::map<int, Q4eQsaPlan> qsa_plans;
        if (qsa != nullptr)
        {
            if (!qsa_positions || n_kv <= 0 || T > n_kv || qsa_position_count != n_kv || position < 0 || position > n_kv - T)
                throw std::invalid_argument("qwen4exp QSA: incomplete position history");
            for (int il = layer_begin; il < layer_end; ++il)
            {
                const auto& a = qsa[il];
                if (!a.ratio) continue;
                if (kinds[il] != 0 || first_ffn_only || a.ratio < 1 || a.ratio > 64
                    || a.top_k <= 0 || a.top_k > INT32_MAX - a.ratio || a.head_dim < n_rot || a.head_dim > 4096
                    || a.heads <= 0 || a.heads > 1024 || n_rot <= 0
                    || !std::isfinite(eps) || eps <= 0 || !std::isfinite(rope_base) || rope_base <= 0
                    || !std::isfinite(rope_freq_scale) || rope_freq_scale <= 0
                    || !a.k_proj || !a.q_proj || !a.k_norm || !a.q_norm || !a.cache
                    || (a.cache_type != GGML_TYPE_F32 && a.cache_type != GGML_TYPE_F16))
                    throw std::invalid_argument("qwen4exp QSA: invalid descriptor");
                auto matrix_valid = [&](int type, long long bytes, int64_t rows) {
                    if (type < 0 || type >= GGML_TYPE_COUNT || bytes <= 0 || rows <= 0) return false;
                    const auto* tr = ggml_get_type_traits((ggml_type)type);
                    if (!tr->type_size || !tr->blck_size || n_embd % tr->blck_size
                        || (type != GGML_TYPE_F32 && type != GGML_TYPE_F16 && type != GGML_TYPE_BF16
                            && !(tr->is_quantized && tr->to_float))) return false;
                    return (uint64_t)bytes == (uint64_t)(n_embd / tr->blck_size) * tr->type_size * rows;
                };
                if (!matrix_valid(a.k_type, a.k_bytes, a.head_dim)
                    || !matrix_valid(a.q_type, a.q_bytes, (int64_t)a.head_dim * a.heads)
                    || a.cache_bytes != (int64_t)ggml_row_size((ggml_type)a.cache_type, a.head_dim) * kv_capacity)
                    throw std::invalid_argument("qwen4exp QSA: invalid storage size/type");
                int64_t section_sum = 0;
                for (int section : a.rope_sections) { if (section < 0) throw std::invalid_argument("qwen4exp QSA: negative rotary section"); section_sum += section; }
                if (section_sum <= 0 || section_sum > n_rot || n_rot % 2)
                    throw std::invalid_argument("qwen4exp QSA: invalid rotary sections");
                if (!qsa_plans.count(a.ratio))
                    qsa_plans.emplace(a.ratio, q4e_qsa_plan(qsa_positions, n_kv, n_kv_pad, position, T, a.ratio));
            }
        }
        auto upload_qsa = [&](const std::vector<Q4eQsaInputs>& inputs) {
            for (const auto& in : inputs)
            {
                const auto& plan = qsa_plans.at(in.ratio);
                auto put = [](ggml_tensor* t, const auto& values) {
                    // Selection inputs are unused below the sparse width; no buffer then.
                    if (t && t->buffer) ggml_backend_tensor_set(t, values.data(), 0, values.size() * sizeof(values[0]));
                };
                put(in.cell_blocks, plan.cell_blocks); put(in.block_cells, plan.block_cells);
                put(in.block_positions, plan.block_positions); put(in.query_positions, plan.query_positions);
                put(in.bias, plan.bias);
            }
        };


        Qwen4ExpFfnCache* slot = (cache_slot >= 0 && cache_slot < kQwen4ExpSpanSlots)
            ? &g_q4e_span[cache_slot] : nullptr;
        if (slot == nullptr)
        {
            set_last_error("qwen4exp token span: bad cache slot.");
            return 0;
        }

        // Replay: same span, same shape, same descriptors, same padded window - and
        // every cache-bound weight still where the graph believes it is.
        if (!q4e_span_force_rebuild()
            && slot->valid
            && q4e_refresh_bindings(slot, ggml_backend_get_device(g_backend))
            && slot->n_tokens == T && slot->hc_dim == hc_dim
            && slot->sig == (const void*)ffn && slot->sig2 == (const void*)gdn
            && slot->sig3 == (const void*)attn
            && slot->qsa_sig == (const void*)qsa
            && slot->layer_begin == layer_begin && slot->layer_end == layer_end
            && slot->kv_capacity == kv_capacity && slot->n_kv == n_kv_pad
            && slot->first_ffn_only == first_ffn_only
            && slot->sig4 == (const void*)head
            && slot->logits_rows == logits_rows
            && slot->export_hidden == (hidden_out != nullptr)
            && slot->sig5 == (const void*)(has_ple ? ple : nullptr)
            && slot->use_mrope == (use_mrope ? 1 : 0))
        {
            ggml_backend_tensor_set(slot->res_in, res_data, 0, res_bytes);
            if (slot->ple_emb_in != nullptr)
                ggml_backend_tensor_set(slot->ple_emb_in, ple_emb, 0,
                        (std::size_t)n_embd * T * sizeof(float));
            for (ggml_tensor* m : slot->span_masks)
                ggml_backend_tensor_set(m, mask_data, 0, mask_bytes);
            for (std::size_t i = 0; i < slot->span_pos.size(); ++i)
                q4e_set_attn_indices(slot->span_pos[i], slot->span_kvidx[i], T, position,
                        use_mrope ? (const int32_t*)mrope_pos : nullptr, rope_position);
            upload_qsa(slot->qsa_inputs);
            q4e_note(3, false);
            if (q4e_span_trace() && !slot->span_copies.empty() && T == 1)
            {
                // What the graph will actually read as its recurrent state, sampled
                // immediately before the compute.
                std::vector<float> pb;
                fprintf(stderr, "[q4e-prestate] slot%d pos=%d:", cache_slot, position);
                for (std::size_t i = 0; i < slot->span_copies.size(); ++i)
                {
                    ggml_tensor* st = slot->span_copies[i].second;
                    pb.resize((std::size_t)ggml_nelements(st));
                    ggml_backend_tensor_get(st, pb.data(), 0, ggml_nbytes(st));
                    double n2 = 0.0;
                    for (float f : pb) n2 += (double)f * f;
                    fprintf(stderr, " %.9e", std::sqrt(n2));
                }
                fprintf(stderr, "%c", 10);
            }
            if (graph_compute_profiled(slot->precise_backend ? slot->precise_backend : g_backend, slot->graph, kQwen4ExpSpanKernel) != GGML_STATUS_SUCCESS)
            {
                slot->reset_graph();
                set_last_error("qwen4exp token span: replay failed.");
                return 0;
            }
            if (slot->logits != nullptr)
            {
                if (!slot->span_copies.empty())
                    tsg::sync_backend(g_backend);
                for (const auto& c : slot->span_copies)
                    ggml_backend_tensor_copy(c.first, c.second);
                ggml_backend_tensor_get(slot->logits, logits_out, 0,
                        (std::size_t)head->vocab * logits_rows * sizeof(float));
                if (hidden_out != nullptr)
                    ggml_backend_tensor_get(slot->res_out, hidden_out, 0, res_bytes);
                q4e_trace_state(slot, "replay", position);
                return 1;
            }
            // The synchronize is LOAD-BEARING: ggml_backend_tensor_copy issues a
            // legacy-stream memcpy, and ggml-cuda's compute stream is non-blocking,
            // so without the drain the copy can read conv_out/ssm_out while the
            // graph is still writing them. That race is timing-dependent - short
            // contexts kept the GPU caught up and hid it; long contexts queue
            // deeper and the copy wins, which corrupted the recurrent state from
            // the first replay after a long prefill.
            if (!slot->span_copies.empty())
                tsg::sync_backend(g_backend);
            for (const auto& c : slot->span_copies)
                ggml_backend_tensor_copy(c.first, c.second);
            ggml_backend_tensor_get(slot->res_out, res_data, 0, res_bytes);
            q4e_trace_probe(slot, "replay", position);
            q4e_trace_state(slot, "replay", position);
            return 1;
        }
        slot->reset_graph();

        const double t0 = q4e_phase_log() ? q4e_now_ms() : 0.0;
        const int n_layers = layer_end - layer_begin;
        ggml_init_params ip{};
        ip.mem_size = ggml_tensor_overhead() * ((std::size_t)n_layers * 256 + 1024)
                    + ggml_graph_overhead_custom(kQwen4ExpSpanGraphSize, false);
        ip.mem_buffer = nullptr;
        ip.no_alloc = true;
        ggml_context* ctx = ggml_init(ip);
        if (ctx == nullptr) { set_last_error("qwen4exp token span: ggml_init failed."); return 0; }
        // Keep local ownership until the complete graph is published below.
        // Binder/vector/state setup can throw after ggml_init or allocation.
        std::unique_ptr<ggml_context, decltype(&ggml_free)> owned_ctx(ctx, ggml_free);

        ggml_tensor* res_in = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, hc_dim, T);
        ggml_set_input(res_in);

        // ONE mask, position and write-row tensor shared by every attention layer
        // in the span - same values for all, so three uploads a replay instead of
        // three dozen. (Private per-layer copies were briefly used to bisect the
        // leaf-free bug; the shared tensors were never at fault.)
        ggml_tensor* mask = nullptr;
        ggml_tensor* pos = nullptr;
        ggml_tensor* kv_idx = nullptr;
        if (has_attn)
        {
            mask = ggml_new_tensor_2d(ctx, GGML_TYPE_F16, n_kv_pad, T);
            ggml_set_input(mask);
            pos = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, use_mrope ? 4 * T : T);
            ggml_set_input(pos);
            kv_idx = ggml_new_tensor_1d(ctx, GGML_TYPE_I64, T);
            ggml_set_input(kv_idx);
            slot->span_masks.push_back(mask);
            slot->span_pos.push_back(pos);
            slot->span_kvidx.push_back(kv_idx);
        }

        ggml_tensor* ple_emb_in = nullptr;
        if (has_ple)
        {
            ple_emb_in = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, n_embd, T);
            ggml_set_input(ple_emb_in);
        }

        ggml_cgraph* graph = ggml_new_graph_custom(ctx, kQwen4ExpSpanGraphSize, false);
        if (q4e_graph_uid_enabled()) graph->uid = q4e_next_graph_uid();

        Q4eBinder binder{ggml_backend_get_device(g_backend)};
        slot->qsa_inputs.reserve(qsa_plans.size());
        for (const auto& item : qsa_plans)
        {
            Q4eQsaInputs in;
            in.ratio = item.first;
            const int blocks = (n_kv_pad + in.ratio - 1) / in.ratio;
            in.cell_blocks = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, n_kv_pad);
            in.block_cells = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, (int64_t)in.ratio * blocks);
            in.block_positions = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, (int64_t)4 * blocks);
            in.query_positions = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, (int64_t)4 * T);
            in.bias = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, blocks, T);
            for (auto* t : {in.cell_blocks, in.block_cells, in.block_positions, in.query_positions, in.bias}) ggml_set_input(t);
            slot->qsa_inputs.push_back(in);
        }
#ifdef TSG_GGML_USE_CUDA
        if (!qsa_plans.empty() && ggml_backend_is_cuda(g_backend))
        {
            slot->precise_backend = tsg_dsv4_fused_backend_init(g_backend);
            if (!slot->precise_backend) throw std::runtime_error("qwen4exp QSA: precise backend creation failed");
        }
#endif


        const std::size_t conv_dim = (std::size_t)(head_k_dim * n_k_heads) * 2
                                   + (std::size_t)(head_v_dim * n_v_heads);
        const std::size_t conv_bytes = (std::size_t)(d_conv - 1) * conv_dim * sizeof(float);
        const std::size_t ssm_bytes = (std::size_t)head_v_dim * head_v_dim * n_v_heads * sizeof(float);
        const std::size_t ssm_off = (conv_bytes + 255) & ~(std::size_t)255;

        std::vector<Qwen4ExpFfnCache*> seeded;              // unused since the seq-state map; kept for shape
        std::vector<Q4eSeqStateEntry*> seeded_states;
        std::vector<ggml_tensor*> kv_tensors;
        std::vector<ggml_tensor*> trace_res;
        std::vector<ggml_tensor*> probe_nodes;
        int attn_seen = 0;
        ggml_tensor* res = res_in;
        bool failed = false;
        const char* fail_what = nullptr;

        for (int il = layer_begin; il < layer_end && !failed; ++il)
        {
            if (has_ple && il == ple_layer)
            {
                // ---- the PLE block, ahead of this layer's halves ----
                // Only the conv-history STATE handling lives here; the node
                // math is q4e_nodes_ple, shared with the arena kernel.
                const int hist = (ple->kern - 1) * ple->dil;
                ggml_tensor* conv_state = nullptr;
                if (hist > 0)
                {
                    // conv history: persistent device state, like the GDN state.
                    const std::size_t st_bytes = (std::size_t)hist * hc_dim * sizeof(float);
                    Q4eSeqStateEntry* ple_st = q4e_seq_state(ple->conv_state, st_bytes);
                    if (ple_st == nullptr)
                    { failed = true; fail_what = "ple state alloc"; break; }
                    conv_state = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, hc_dim, hist);
                    ggml_set_input(conv_state);
                    if (ggml_backend_tensor_alloc(ple_st->buf, conv_state,
                            ggml_backend_buffer_get_base(ple_st->buf)) != GGML_STATUS_SUCCESS)
                    { failed = true; fail_what = "ple state bind"; break; }
                    if (!ple_st->ready)
                    {
                        binder.upload_list.push_back({conv_state, ple->conv_state, st_bytes});
                        seeded_states.push_back(ple_st);
                    }
                }
                res = q4e_nodes_ple(ctx, graph, binder, ple, res, ple_emb_in,
                        conv_state, n_embd, hc, T, 1, eps);
            }

            if (il == layer_begin && first_ffn_only != 0)
            {
                // The attention half of this layer already ran per-layer; the span
                // picks up from its FFN half below.
            }
            else if (kinds[il] != 0)
            {
                // ---- recurrent half ----
                Q4eSeqStateEntry* gst = q4e_seq_state(gdn[il].conv_state, ssm_off + ssm_bytes);
                if (gst == nullptr)
                { failed = true; fail_what = "state buffer alloc"; break; }
                ggml_tensor* conv_state = ggml_new_tensor_3d(ctx, GGML_TYPE_F32,
                        d_conv - 1, (int64_t)conv_dim, 1);
                ggml_tensor* ssm_state = ggml_new_tensor_3d(ctx, GGML_TYPE_F32,
                        head_v_dim, head_v_dim, n_v_heads);
                ggml_set_input(conv_state);
                ggml_set_input(ssm_state);
                std::uint8_t* base = (std::uint8_t*)ggml_backend_buffer_get_base(gst->buf);
                if (ggml_backend_tensor_alloc(gst->buf, conv_state, base) != GGML_STATUS_SUCCESS ||
                    ggml_backend_tensor_alloc(gst->buf, ssm_state, base + ssm_off) != GGML_STATUS_SUCCESS)
                { failed = true; fail_what = "state buffer bind"; break; }
                if (!gst->ready)
                {
                    binder.upload_list.push_back({conv_state, gdn[il].conv_state, conv_bytes});
                    binder.upload_list.push_back({ssm_state, gdn[il].ssm_state, ssm_bytes});
                    seeded_states.push_back(gst);
                }

                Q4eGdnWriteback wb{};
                res = q4e_nodes_gdn(ctx, binder, &gdn[il], res,
                        conv_state, ssm_state,
                        n_embd, hc, hc_low_rank, T,
                        head_k_dim, head_v_dim, n_k_heads, n_v_heads, d_conv, eps, &wb,
                        (q4e_span_trace() && il == layer_begin && cache_slot == 0 && T == 1)
                            ? &slot->gdn_probe : nullptr);
                // The residual first: its tree holds every node that READS the state
                // (the concat, the delta-net input). Then the write-back, so node
                // order puts the write strictly after the read.
                ggml_build_forward_expand(graph, res);
                if (q4e_span_state_in_graph())
                {
                    ggml_build_forward_expand(graph, ggml_cpy(ctx, wb.tail, conv_state));
                    ggml_build_forward_expand(graph, ggml_cpy(ctx, wb.new_state, ssm_state));
                }
                else
                {
                    ggml_tensor* conv_out = ggml_new_tensor_3d(ctx, GGML_TYPE_F32,
                            d_conv - 1, (int64_t)conv_dim, 1);
                    ggml_tensor* ssm_out = ggml_new_tensor_3d(ctx, GGML_TYPE_F32,
                            head_v_dim, head_v_dim, n_v_heads);
                    ggml_set_output(conv_out);
                    ggml_set_output(ssm_out);
                    ggml_build_forward_expand(graph, ggml_cpy(ctx, wb.tail, conv_out));
                    ggml_build_forward_expand(graph, ggml_cpy(ctx, wb.new_state, ssm_out));
                    slot->span_copies.push_back({conv_out, conv_state});
                    slot->span_copies.push_back({ssm_out, ssm_state});
                }
            }
            else
            {
                // ---- attention half (expands its own KV write first) ----
                bool fa_here = use_flash && (attn_seen < q4e_span_fa_max());
                ++attn_seen;
                Q4eQsaGraph qsa_graph;
                if (qsa != nullptr && qsa[il].ratio > 0)
                {
                    const auto& a = qsa[il];
                    qsa_graph.args = &a;
                    for (const auto& in : slot->qsa_inputs) if (in.ratio == a.ratio) qsa_graph.inputs = &in;
                    auto* state = q4e_seq_state(a.cache, (size_t)a.cache_bytes);
                    if (!state) throw std::runtime_error("qwen4exp QSA: cache allocation failed");
                    qsa_graph.cache = ggml_new_tensor_2d(ctx, (ggml_type)a.cache_type, a.head_dim, kv_capacity);
                    ggml_set_input(qsa_graph.cache);
                    if (ggml_backend_tensor_alloc(state->buf, qsa_graph.cache, ggml_backend_buffer_get_base(state->buf)) != GGML_STATUS_SUCCESS)
                        throw std::runtime_error("qwen4exp QSA: cache binding failed");
                    if (!state->ready)
                    {
                        binder.upload_list.push_back({qsa_graph.cache, a.cache, (size_t)a.cache_bytes});
                        seeded_states.push_back(state);
                    }
                }
                res = q4e_nodes_attn(ctx, graph, binder, &attn[il], res,
                        mask, pos, kv_idx,
                        n_embd, hc, hc_low_rank, T,
                        head_dim, n_head, n_head_kv, kv_capacity, n_kv_pad,
                        n_rot, rope_base, rope_freq_scale, attn_scale, eps, fa_here,
                        &kv_tensors,
                        (q4e_span_trace() && attn_seen == 1 && T == 1) ? &probe_nodes : nullptr,
                        use_mrope ? (const int32_t*)mrope_sections : nullptr, nullptr, nullptr, nullptr,
                        qsa_graph.args ? &qsa_graph : nullptr);
                ggml_build_forward_expand(graph, res);
            }
            if (q4e_span_trace()) { ggml_set_output(res); trace_res.push_back(res); }

            // ---- FFN half ----
            res = q4e_nodes_ffn(ctx, binder, &ffn[il], res,
                    n_embd, hc, hc_low_rank, T,
                    n_expert, n_expert_used, n_ff, n_ff_sh, eps);
            ggml_build_forward_expand(graph, res);
            if (q4e_span_trace()) { ggml_set_output(res); trace_res.push_back(res); }
        }

        if (failed)
        {
            set_last_error(std::string("qwen4exp token span: ") + (fail_what ? fail_what : "build") + " failed.");
            return 0;
        }

        const double t_nodes = q4e_phase_log() ? q4e_now_ms() : 0.0;
        ggml_tensor* res_out = res;
        ggml_tensor* logits = nullptr;
        if (head != nullptr)
        {
            // Verification needs all rows. Ordinary decode/prefill retains its
            // original last-row head and does not keep the wide residual alive.
            ggml_tensor* head_input = res_out;
            if (logits_rows == 1)
                head_input = ggml_cont(ctx, ggml_view_2d(ctx, res_out, hc_dim, 1,
                        res_out->nb[1], (std::size_t)(T - 1) * res_out->nb[1]));
            logits = q4e_nodes_head(ctx, binder, head, head_input,
                    n_embd, hc, hc_low_rank, logits_rows, eps);
            if (hidden_out != nullptr)
                ggml_set_output(res_out);
            ggml_set_output(logits);
            ggml_build_forward_expand(graph, logits);
        }
        else
        {
            ggml_set_output(res_out);
        }

        const double t_pregal = q4e_phase_log() ? q4e_now_ms() : 0.0;
        ggml_gallocr_t alloc = ggml_gallocr_new(ggml_backend_get_default_buffer_type(g_backend));
        std::unique_ptr<std::remove_pointer_t<ggml_gallocr_t>, decltype(&ggml_gallocr_free)>
            owned_alloc(alloc, ggml_gallocr_free);
        if (alloc == nullptr || !ggml_gallocr_alloc_graph(alloc, graph))
        {
            set_last_error("qwen4exp token span: failed to allocate graph tensors.");
            return 0;
        }
        const double t_gal = q4e_phase_log() ? q4e_now_ms() : 0.0;

        binder.flush();
        for (Q4eSeqStateEntry* e : seeded_states) e->ready = true;
        if (ple_emb_in != nullptr)
            ggml_backend_tensor_set(ple_emb_in, ple_emb, 0,
                    (std::size_t)n_embd * T * sizeof(float));

        // A fresh sequence starts with a KV cache whose device copy holds whatever
        // was already in that memory - the host buffer it uploads from is a pooled
        // allocation with no zero guarantee. The window this graph reads is padded
        // past the rows any token has written, and a masked-off column contributes
        // nothing only if it is FINITE: an Inf in a never-written K row makes its
        // score Inf, Inf plus the -inf mask is NaN, and one NaN takes the whole
        // softmax row (flash attention alike). Zero the device copies at the start
        // of the sequence so the pad always reads zeros.
        if (position == 0)
            for (ggml_tensor* t : kv_tensors)
                ggml_backend_tensor_memset(t, 0, 0, ggml_nbytes(t));

        ggml_backend_tensor_set(res_in, res_data, 0, res_bytes);
        for (ggml_tensor* m : slot->span_masks)
            ggml_backend_tensor_set(m, mask_data, 0, mask_bytes);
        for (std::size_t i = 0; i < slot->span_pos.size(); ++i)
            q4e_set_attn_indices(slot->span_pos[i], slot->span_kvidx[i], T, position,
                    use_mrope ? (const int32_t*)mrope_pos : nullptr, rope_position);

        upload_qsa(slot->qsa_inputs);
        const double t_up = q4e_phase_log() ? q4e_now_ms() : 0.0;
        q4e_note(3, true);
        if (graph_compute_profiled(slot->precise_backend ? slot->precise_backend : g_backend, graph, kQwen4ExpSpanKernel) != GGML_STATUS_SUCCESS)
        {
            set_last_error("qwen4exp token span: graph compute failed.");
            return 0;
        }
        if (q4e_phase_log() && T > 1)
        {
            tsg::sync_backend(g_backend);
            const double t_cmp = q4e_now_ms();
            fprintf(stderr, "[q4e-phase] span %d..%d T=%d: nodes=%.1fms binder=%.1fms gallocr=%.1fms uploads=%.1fms compute=%.1fms%c",
                    layer_begin, layer_end, T,
                    t_nodes - t0, t_pregal - t_nodes, t_gal - t_pregal, t_up - t_gal, t_cmp - t_up, 10);
        }

        // See the replay path: the drain before the copies is load-bearing.
        if (!slot->span_copies.empty())
            tsg::sync_backend(g_backend);
        for (const auto& c : slot->span_copies)
            ggml_backend_tensor_copy(c.first, c.second);
        if (logits != nullptr)
        {
            ggml_backend_tensor_get(logits, logits_out, 0,
                    (std::size_t)head->vocab * logits_rows * sizeof(float));
            if (hidden_out != nullptr)
                ggml_backend_tensor_get(res_out, hidden_out, 0, res_bytes);
        }
        else
            ggml_backend_tensor_get(res_out, res_data, 0, res_bytes);
        q4e_trace_probe(slot, "build", position);
        q4e_trace_state(slot, "build", position);

        if (!probe_nodes.empty())
        {
            static const char* names[] = { "scores", "probs", "attn" };
            for (std::size_t i = 0; i < probe_nodes.size() && i < 3; ++i)
            {
                ggml_tensor* t = probe_nodes[i];
                std::vector<float> pb((std::size_t)ggml_nelements(t));
                ggml_backend_tensor_get(t, pb.data(), 0, ggml_nbytes(t));
                double n2 = 0.0; float amax = 0.0f; std::size_t iamax = 0; bool bad = false;
                for (std::size_t j = 0; j < pb.size(); ++j)
                {
                    float f = pb[j];
                    n2 += (double)f * f;
                    if (std::fabs(f) > amax) { amax = std::fabs(f); iamax = j; }
                    if (!std::isfinite(f)) bad = true;
                }
                fprintf(stderr, "[q4e-probe] %s ne=[%d,%d,%d] l2=%.9e amax=%.6e@%zu head=%.6e %.6e %.6e%s%c",
                        names[i], (int)t->ne[0], (int)t->ne[1], (int)t->ne[2],
                        std::sqrt(n2), amax, iamax,
                        pb.size() > 0 ? pb[0] : 0.0f, pb.size() > 1 ? pb[1] : 0.0f,
                        pb.size() > 52 ? pb[52] : 0.0f, bad ? " NAN" : "", 10);
            }
        }
        if (q4e_span_trace() && has_attn && T == 1)
        {
            // Read back the K/V pad rows this graph could see. A masked-off pad
            // column only contributes nothing if its K row is exactly zero.
            std::vector<uint16_t> kv_buf;
            for (std::size_t t = 0; t < kv_tensors.size(); ++t)
            {
                ggml_tensor* kc = kv_tensors[t];
                const std::size_t total = ggml_nbytes(kc);
                kv_buf.resize(total / 2);
                ggml_backend_tensor_get(kc, kv_buf.data(), 0, total);
                // [head_dim, capacity, kvH]; pad rows are n_kv..n_kv_pad of dim 1.
                int max_bits = 0; long nonzero = 0;
                for (int64_t h = 0; h < kc->ne[2]; ++h)
                    for (int64_t r = n_kv; r < n_kv_pad; ++r)
                    {
                        const uint16_t* row = kv_buf.data()
                            + (h * kc->ne[1] + r) * kc->ne[0];
                        for (int64_t c = 0; c < kc->ne[0]; ++c)
                        {
                            int m = row[c] & 0x7FFF;   // f16 magnitude bits
                            if (m != 0) { ++nonzero; if (m > max_bits) max_bits = m; }
                        }
                    }
                if (nonzero > 0)
                    fprintf(stderr, "[q4e-kvpad] tensor %zu: %ld nonzero pad values, max f16 bits 0x%04x (rows %d..%d)%c",
                            t, nonzero, (unsigned)max_bits, n_kv, n_kv_pad, 10);
            }
            fprintf(stderr, "[q4e-kvpad] scan done (%zu tensors)%c", kv_tensors.size(), 10);
        }
        if (q4e_span_trace())
        {
            std::vector<float> buf((std::size_t)hc_dim * T);
            fprintf(stderr, "[q4e-trace] span %d..%d T=%d pos=%d:", layer_begin, layer_end, T, position);
            for (std::size_t i = 0; i < trace_res.size(); ++i)
            {
                ggml_backend_tensor_get(trace_res[i], buf.data(), 0, buf.size() * sizeof(float));
                double n2 = 0.0; bool bad = false;
                for (float f : buf) { n2 += (double)f * f; if (!std::isfinite(f)) bad = true; }
                fprintf(stderr, " %zu:%.4e%s", i, std::sqrt(n2), bad ? "!NAN" : "");
            }
            fprintf(stderr, "\n");
        }

        slot->ctx = owned_ctx.release(); slot->graph = graph; slot->alloc = owned_alloc.release();
        slot->res_in = res_in; slot->res_out = res_out;
        slot->n_tokens = T; slot->hc_dim = hc_dim;
        slot->sig = (const void*)ffn; slot->sig2 = (const void*)gdn; slot->sig3 = (const void*)attn;
        slot->qsa_sig = (const void*)qsa;
        slot->sig4 = (const void*)head; slot->logits = logits;
        slot->logits_rows = logits_rows; slot->export_hidden = hidden_out != nullptr;
        slot->sig5 = (const void*)(has_ple ? ple : nullptr);
        slot->ple_emb_in = ple_emb_in;
        slot->layer_begin = layer_begin; slot->layer_end = layer_end;
        slot->kv_capacity = kv_capacity; slot->n_kv = n_kv_pad;
        slot->first_ffn_only = first_ffn_only;
        slot->use_mrope = use_mrope ? 1 : 0;
        slot->rebinds = std::move(binder.cached);
        slot->res_resident = 0;
        slot->valid = true;
        return 1;
    }
    catch (const std::exception& e)
    { set_last_error(std::string("qwen4exp token span: ") + e.what()); return 0; }
    catch (...)
    { set_last_error("qwen4exp token span: unknown error."); return 0; }
}

// The original export keeps its ABI and last-row behavior. Speculation uses
// the versioned export so old applications never pass an uninitialized tail.
#define Q4E_SPAN_PARAMETERS \
    const TSGgmlQwen4ExpFfnArgs* ffn, const TSGgmlQwen4ExpGdnArgs* gdn, \
    const TSGgmlQwen4ExpAttnArgs* attn, const unsigned char* kinds, \
    int layer_begin, int layer_end, void* res_data, const void* mask_data, \
    int n_embd, int hc, int hc_low_rank, int n_tokens, \
    int head_k_dim, int head_v_dim, int n_k_heads, int n_v_heads, int d_conv, \
    int head_dim, int n_head, int n_head_kv, int kv_capacity, int n_kv, int position, \
    int n_rot, float rope_base, float rope_freq_scale, float attn_scale, \
    int n_expert, int n_expert_used, int n_ff, int n_ff_sh, \
    float eps, int cache_slot, int first_ffn_only, \
    const TSGgmlQwen4ExpHeadArgs* head, void* logits_out, \
    const TSGgmlQwen4ExpPleArgs* ple, int ple_layer, const void* ple_emb, \
    const int* mrope_pos, const int* mrope_sections, int rope_position, int device
#define Q4E_SPAN_ARGUMENTS \
    ffn, gdn, attn, kinds, layer_begin, layer_end, res_data, mask_data, \
    n_embd, hc, hc_low_rank, n_tokens, head_k_dim, head_v_dim, n_k_heads, n_v_heads, d_conv, \
    head_dim, n_head, n_head_kv, kv_capacity, n_kv, position, \
    n_rot, rope_base, rope_freq_scale, attn_scale, n_expert, n_expert_used, n_ff, n_ff_sh, \
    eps, cache_slot, first_ffn_only, head, logits_out, ple, ple_layer, ple_emb, \
    mrope_pos, mrope_sections, rope_position, device

TSG_EXPORT int TSGgml_Qwen4ExpTokenSpan(Q4E_SPAN_PARAMETERS)
{
    return q4e_token_span_impl(Q4E_SPAN_ARGUMENTS, nullptr, 1, nullptr, nullptr, 0);
}

TSG_EXPORT int TSGgml_Qwen4ExpTokenSpanEx(
    Q4E_SPAN_PARAMETERS, void* hidden_out, int logits_rows)
{
    return q4e_token_span_impl(Q4E_SPAN_ARGUMENTS, hidden_out, logits_rows, nullptr, nullptr, 0);
}
TSG_EXPORT int TSGgml_Qwen4ExpTokenSpanQsa(
    Q4E_SPAN_PARAMETERS, void* hidden_out, int logits_rows,
    const TSGgmlQwen4ExpQsaArgs* qsa, const int32_t* positions, int position_count)
{
    return q4e_token_span_impl(Q4E_SPAN_ARGUMENTS, hidden_out, logits_rows, qsa, positions, position_count);
}
#undef Q4E_SPAN_ARGUMENTS
#undef Q4E_SPAN_PARAMETERS

// Copy the authoritative indexer cache before managed growth. The source key
// stays owned until ReleaseSeqState drops every graph which can reference it.
TSG_EXPORT int TSGgml_Qwen4ExpCopyQsaCache(const void* key, void* destination, long long bytes, int device)
{
    try
    {
        if (!key || !destination || bytes <= 0) throw std::invalid_argument("qwen4exp QSA: invalid cache copy");
        tsg::ScopedRank rank(q4e_resolve_device(device));
        auto* state = q4e_seq_state_find(key);
        if (!state || !state->ready || state->bytes != (size_t)bytes)
            throw std::invalid_argument("qwen4exp QSA: missing or mismatched cache copy source");
        ggml_init_params params{ggml_tensor_overhead() * 4, nullptr, true};
        std::unique_ptr<ggml_context, decltype(&ggml_free)> ctx(ggml_init(params), ggml_free);
        if (!ctx) throw std::bad_alloc();
        auto* tensor = ggml_new_tensor_1d(ctx.get(), GGML_TYPE_I8, bytes);
        if (ggml_backend_tensor_alloc(state->buf, tensor, ggml_backend_buffer_get_base(state->buf)) != GGML_STATUS_SUCCESS)
            throw std::runtime_error("qwen4exp QSA: cache copy binding failed");
        tsg::sync_backend(g_backend);
        ggml_backend_tensor_get(tensor, destination, 0, (size_t)bytes);
        return 1;
    }
    catch (const std::exception& e) { set_last_error(e.what()); return 0; }
    catch (...) { set_last_error("qwen4exp QSA: cache copy failed"); return 0; }
}

// Copy the residual to / from the device-resident buffer. The op-by-op attention
// half still works on the host, so it brackets itself with these.
TSG_EXPORT int TSGgml_Qwen4ExpResUpload(const void* data, long long bytes)
{
    try
    {
        if (!ensure_backend() || data == nullptr || bytes <= 0) return 0;
        if (!q4e_res_ensure((std::size_t)bytes)) return 0;
        ggml_backend_tensor_set(g_q4e_res, data, 0, (std::size_t)bytes);
        return 1;
    }
    catch (...) { set_last_error("qwen4exp residual upload failed."); return 0; }
}

TSG_EXPORT int TSGgml_Qwen4ExpResDownload(void* data, long long bytes)
{
    try
    {
        if (!ensure_backend() || data == nullptr || bytes <= 0) return 0;
        if (g_q4e_res == nullptr || g_q4e_res_capacity < (std::size_t)bytes) return 0;
        tsg::sync_backend(g_backend);
        ggml_backend_tensor_get(g_q4e_res, data, 0, (std::size_t)bytes);
        return 1;
    }
    catch (...) { set_last_error("qwen4exp residual download failed."); return 0; }
}

TSG_EXPORT void TSGgml_Qwen4ExpResetFfnCache()
{
    // Sweep every initialized device. Under a layer split the token's layers are
    // spread across GPUs, each with its own graph slots and residual buffer;
    // resetting only the active rank would leave live graphs elsewhere pointing
    // at state the caller believes it has dropped.
    const int ndev = tsg::g_device_count.load(std::memory_order_acquire);
    for (int d = 0; d < ndev && d < tsg::TSG_MAX_DEVICES; ++d)
    {
        tsg::ScopedRank rank(d);
        for (int i = 0; i < kQwen4ExpMaxSlots; ++i)
        {
            g_q4e_ffn[i].reset();
            g_q4e_gdn[i].reset();
            g_q4e_attn[i].reset();
        }
        for (int i = 0; i < kQwen4ExpSpanSlots; ++i)
            g_q4e_span[i].reset();
        if (g_q4e_res_ctx) { ggml_free(g_q4e_res_ctx); g_q4e_res_ctx = nullptr; }
        if (g_q4e_res_buf) { ggml_backend_buffer_free(g_q4e_res_buf); g_q4e_res_buf = nullptr; }
        g_q4e_res = nullptr; g_q4e_res_capacity = 0;
    }
}

// Drop (graphs only - state kept) every cached per-layer/span graph built from
// the given holder descriptor arrays, on every device. The arena flush calls
// this after invalidating a holder's resident KV copies: captured solo graphs
// bake those buffers, and q4e_refresh_bindings only SAMPLES every 32nd replay,
// which is no protection against a freed pointer. Per-layer slots key on
// &array[layer], span slots on the array base, so both a stride-range and an
// exact-base match are needed.
void q4e_drop_holder_graphs(const void* attn_base, const void* gdn_base, const void* ple_base)
{
    auto in_array = [](const void* sig, const void* base, std::size_t stride) {
        if (sig == nullptr || base == nullptr) return false;
        const std::uintptr_t sv = (std::uintptr_t)sig;
        const std::uintptr_t bv = (std::uintptr_t)base;
        if (sv < bv) return false;
        const std::uintptr_t d = sv - bv;
        return d < stride * (std::uintptr_t)kQwen4ExpMaxSlots && d % stride == 0;
    };
    const int ndev = tsg::g_device_count.load(std::memory_order_acquire);
    for (int dv = 0; dv < ndev && dv < tsg::TSG_MAX_DEVICES; ++dv)
    {
        tsg::ScopedRank rank(dv);
        for (int i = 0; i < kQwen4ExpMaxSlots; ++i)
        {
            if (g_q4e_attn[i].valid && in_array(g_q4e_attn[i].sig, attn_base, sizeof(TSGgmlQwen4ExpAttnArgs)))
                g_q4e_attn[i].reset_graph();
            if (g_q4e_gdn[i].valid && in_array(g_q4e_gdn[i].sig, gdn_base, sizeof(TSGgmlQwen4ExpGdnArgs)))
                g_q4e_gdn[i].reset_graph();
        }
        for (int i = 0; i < kQwen4ExpSpanSlots; ++i)
        {
            Qwen4ExpFfnCache& sp = g_q4e_span[i];
            if (!sp.valid) continue;
            if ((attn_base != nullptr && sp.sig3 == attn_base) ||
                (gdn_base != nullptr && sp.sig2 == gdn_base) ||
                (ple_base != nullptr && sp.sig5 == ple_base))
                sp.reset_graph();
        }
    }
}

/// Mark one sequence-state entry (keyed by its host seed pointer) as needing a
/// re-seed on the next graph build. The buffer and its baked addresses stay
/// valid; only the one-time upload re-arms. Used by the managed reset, whose
/// host copies are freshly zeroed.
TSG_EXPORT void TSGgml_Qwen4ExpInvalidateSeqState(const void* key)
{
    const int ndev = tsg::g_device_count.load(std::memory_order_acquire);
    for (int d = 0; d < ndev && d < tsg::TSG_MAX_DEVICES; ++d)
    {
        tsg::ScopedRank rank(d);
        auto it = g_q4e_seq_state.find(key);
        if (it != g_q4e_seq_state.end()) it->second.ready = false;
    }
}

/// Free EVERY sequence-state entry and drop every cached graph. Called on
/// model dispose so a later model load in the same process can never collide
/// with stale entries keyed on recycled host addresses.
TSG_EXPORT void TSGgml_Qwen4ExpReleaseAllSeqState()
{
    // The freed buffers may still be flush targets of registered arena slots;
    // drop the registrations WITHOUT flushing (the managed dispose ordering
    // flushes first when it wants the bytes).
    tsg_q4earena::on_drop_all();
    const int ndev = tsg::g_device_count.load(std::memory_order_acquire);
    for (int d = 0; d < ndev && d < tsg::TSG_MAX_DEVICES; ++d)
    {
        tsg::ScopedRank rank(d);
        for (auto& kv : g_q4e_seq_state)
            if (kv.second.buf) ggml_backend_buffer_free(kv.second.buf);
        g_q4e_seq_state.clear();
    }
    TSGgml_Qwen4ExpResetFfnCache();
}

/// Free the sequence-state entries for a released sequence holder and drop
/// every cached graph (graphs bake state-buffer addresses; surviving holders
/// rebuild on their next token and re-bind their still-alive entries without
/// re-seeding).
TSG_EXPORT void TSGgml_Qwen4ExpReleaseSeqState(const void* const* keys, int n)
{
    // See ReleaseAllSeqState: retire (without flushing) any arena slot still
    // registered on a key whose device state is about to be freed.
    for (int i = 0; i < n; ++i)
        tsg_q4earena::on_drop(keys[i]);
    bool freed = false;
    const int ndev = tsg::g_device_count.load(std::memory_order_acquire);
    for (int d = 0; d < ndev && d < tsg::TSG_MAX_DEVICES; ++d)
    {
        tsg::ScopedRank rank(d);
        for (int i = 0; i < n; ++i)
        {
            auto it = g_q4e_seq_state.find(keys[i]);
            if (it == g_q4e_seq_state.end()) continue;
            if (it->second.buf) ggml_backend_buffer_free(it->second.buf);
            g_q4e_seq_state.erase(it);
            freed = true;
        }
    }
    if (freed)
        TSGgml_Qwen4ExpResetFfnCache();
}

namespace
{
    struct Q4eStateSnapshotEntry
    {
        const void* key = nullptr;
        int device = 0;
        ggml_backend_buffer_t source = nullptr;
        ggml_backend_buffer_t saved = nullptr;
        ggml_context* ctx = nullptr;
        ggml_tensor* live_tensor = nullptr;
        ggml_tensor* saved_tensor = nullptr;
        std::size_t bytes = 0;
        bool ready = false;

        ~Q4eStateSnapshotEntry()
        {
            if (saved) ggml_backend_buffer_free(saved);
            if (ctx) ggml_free(ctx);
        }

        void prepare(std::size_t required)
        {
            if (bytes == required && saved != nullptr) return;
            if (saved) { ggml_backend_buffer_free(saved); saved = nullptr; }
            if (ctx) { ggml_free(ctx); ctx = nullptr; }
            bytes = 0;
            if (required == 0 || required % sizeof(float) != 0)
                throw std::runtime_error("qwen4exp state snapshot: invalid state byte count");
            ggml_init_params ip{};
            ip.mem_size = 2 * ggml_tensor_overhead();
            ip.no_alloc = true;
            ctx = ggml_init(ip);
            if (ctx == nullptr) throw std::bad_alloc();
            live_tensor = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, required / sizeof(float));
            saved_tensor = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, required / sizeof(float));
            auto buft = ggml_backend_get_default_buffer_type(g_backend);
            saved = ggml_backend_buft_alloc_buffer(buft,
                    ggml_backend_buft_get_alloc_size(buft, saved_tensor));
            if (saved == nullptr) throw std::bad_alloc();
            if (ggml_backend_tensor_alloc(saved, saved_tensor,
                    ggml_backend_buffer_get_base(saved)) != GGML_STATUS_SUCCESS)
                throw std::runtime_error("qwen4exp state snapshot: saved buffer binding failed");
            bytes = required;
        }

        void bind_live(ggml_backend_buffer_t buffer)
        {
            // Metadata only; this tensor does not own the source allocation.
            // A snapshot never outlives its managed holder, and restore checks
            // the exact allocation identity before binding or writing it.
            live_tensor->buffer = nullptr;
            live_tensor->data = nullptr;
            live_tensor->extra = nullptr;
            if (ggml_backend_tensor_alloc(buffer, live_tensor,
                    ggml_backend_buffer_get_base(buffer)) != GGML_STATUS_SUCCESS)
                throw std::runtime_error("qwen4exp state snapshot: live buffer binding failed");
        }
    };

    struct Q4eStateSnapshot
    {
        std::vector<std::unique_ptr<Q4eStateSnapshotEntry>> entries;
        const void* attn_base = nullptr;
        const void* gdn_base = nullptr;
        const void* ple_base = nullptr;
        bool captured = false;
    };
}

TSG_EXPORT void* TSGgml_Qwen4ExpStateSnapshotCreate(
    const void* const* keys, const int* devices, int count,
    const void* attn_base, const void* gdn_base, const void* ple_base)
{
    try
    {
        set_last_error("");
        if (count < 0 || count > kQwen4ExpMaxSlots + 1
            || (count != 0 && (keys == nullptr || devices == nullptr)))
            throw std::invalid_argument("qwen4exp state snapshot: invalid keys");
        auto snapshot = std::make_unique<Q4eStateSnapshot>();
        snapshot->entries.reserve(count);
        snapshot->attn_base = attn_base;
        snapshot->gdn_base = gdn_base;
        snapshot->ple_base = ple_base;
        const int device_count = tsg::g_device_count.load(std::memory_order_acquire);
        for (int i = 0; i < count; ++i)
        {
            if (keys[i] == nullptr || devices[i] < 0 || devices[i] >= device_count)
                throw std::invalid_argument("qwen4exp state snapshot: invalid key/device");
            auto entry = std::make_unique<Q4eStateSnapshotEntry>();
            entry->key = keys[i];
            entry->device = devices[i];
            snapshot->entries.push_back(std::move(entry));
        }
        return snapshot.release();
    }
    catch (const std::exception& e) { set_last_error(e.what()); return nullptr; }
    catch (...) { set_last_error("qwen4exp state snapshot: creation failed"); return nullptr; }
}

TSG_EXPORT int TSGgml_Qwen4ExpSpecApiVersion()
{
    return 2; // v1 snapshot/MTP; v2 adds QSA span and authoritative indexer export.
}

TSG_EXPORT int TSGgml_Qwen4ExpStateSnapshotCapture(void* handle)
{
    auto* snapshot = static_cast<Q4eStateSnapshot*>(handle);
    if (snapshot == nullptr) { set_last_error("qwen4exp state snapshot: null handle"); return 0; }
    snapshot->captured = false;
    try
    {
        set_last_error("");
        for (auto& entry : snapshot->entries)
        {
            tsg::ScopedRank rank(entry->device);
            if (!ensure_backend()) return 0;
            // Batched decode may own the latest state until its slot is flushed.
            tsg_q4earena::on_external_touch(entry->key);
            auto* live = q4e_seq_state_find(entry->key);
            entry->source = live != nullptr ? live->buf : nullptr;
            entry->ready = live != nullptr && live->ready;
            if (!entry->ready) continue; // the unchanged host seed remains authoritative
            entry->prepare(live->bytes);
            entry->bind_live(live->buf);
            tsg::sync_backend(g_backend);
            ggml_backend_tensor_copy(entry->live_tensor, entry->saved_tensor);
        }
        snapshot->captured = true;
        return 1;
    }
    catch (const std::exception& e) { set_last_error(e.what()); return 0; }
    catch (...) { set_last_error("qwen4exp state snapshot: capture failed"); return 0; }
}

TSG_EXPORT int TSGgml_Qwen4ExpStateSnapshotRestore(void* handle)
{
    auto* snapshot = static_cast<Q4eStateSnapshot*>(handle);
    if (snapshot == nullptr || !snapshot->captured)
    { set_last_error("qwen4exp state snapshot: no completed capture"); return 0; }
    try
    {
        set_last_error("");
        // Validate every allocation before changing any state. A freed/replaced
        // holder is never silently recreated from a stale snapshot.
        for (auto& entry : snapshot->entries)
        {
            tsg::ScopedRank rank(entry->device);
            auto* live = q4e_seq_state_find(entry->key);
            if (entry->ready && (live == nullptr || live->buf != entry->source
                    || live->bytes != entry->bytes))
                throw std::runtime_error("qwen4exp state snapshot: holder allocation changed");
        }
        bool reseed = false;
        for (auto& entry : snapshot->entries)
        {
            tsg::ScopedRank rank(entry->device);
            tsg_q4earena::on_external_touch(entry->key);
            auto* live = q4e_seq_state_find(entry->key);
            if (!entry->ready)
            {
                if (live != nullptr) { live->ready = false; reseed = true; }
                continue;
            }
            entry->bind_live(live->buf);
            tsg::sync_backend(g_backend);
            ggml_backend_tensor_copy(entry->saved_tensor, entry->live_tensor);
            live->ready = true;
        }
        if (reseed)
            q4e_drop_holder_graphs(snapshot->attn_base, snapshot->gdn_base, snapshot->ple_base);
        return 1;
    }
    catch (const std::exception& e) { set_last_error(e.what()); return 0; }
    catch (...) { set_last_error("qwen4exp state snapshot: restore failed"); return 0; }
}

TSG_EXPORT void TSGgml_Qwen4ExpStateSnapshotFree(void* handle)
{
    try { delete static_cast<Q4eStateSnapshot*>(handle); }
    catch (...) { set_last_error("qwen4exp state snapshot: release failed"); }
}
