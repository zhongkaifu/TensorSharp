// Copyright (c) Zhongkai Fu. All rights reserved.
// https://github.com/zhongkaifu/TensorSharp
//
// This file is part of TensorSharp.
//
// TensorSharp is licensed under the BSD-3-Clause license found in the LICENSE file in the root directory of this source tree.
//
// MiniMax-M3 native executor (ggml_ops_minimax_m3_exec.cpp) self-consistency.
//
// A small fixture GGUF with MiniMax-M3's structure is written here: the
// executor reads only minimax-m3.* keys and tensors, so no tokenizer and no
// download are needed. Geometry: 4 layers (1 dense lead), 8 query / 2 KV heads
// of 128, partial NEOX RoPE over 64 dims, an indexer on every layer with
// 16-row blocks and top-4 (so attention is sparse from position 64 on), NVFP4
// experts with per-expert output scales, Q8_0 attention and shared expert,
// BF16 embeddings and head.
//
// One sequence's logits must not depend on how it is fed: every check below
// compares a way of evaluating the same tokens with a one-shot prefill of all
// rows (positions 0..N-1, dense and sparse ranges both):
//   chunked prefill, token-by-token decode, an 8-row verification, rewind +
//   re-forward, slot copy + continue, a 3-slot batched decode, a mixed
//   segments call, the sparse path forced below the dense span
//   (TS_MINIMAX_M3_MSA=ggml vs dense, exact there), and per mode a second
//   placement: 2-GPU layer split vs 1 GPU (cuda-multi), routed experts on the
//   host vs on the GPU (cuda-cpumoe), tensor parallelism over 2 GPUs (ggml's
//   meta backend, attention mirrored, the FFN split) vs 1 GPU (cuda-tp). A
//   rewind and a slot copy must also reproduce the very same calls made
//   without them bit for bit.
// Failure paths, with faults injected through the test hooks: a slot copy
// whose destination must grow while memory is short frees other retained
// slots but never its own source; a loader thread that cannot start, or an
// upload worker that throws, is a load refusal (the process survives);
// TS_MINIMAX_M3_MSA=kernel is refused on a GPU the MSA kernels cannot run on
// (a CUDA GPU older than compute capability 8.0, any Metal GPU: metal mode),
// where auto keeps upstream ggml's sparse ops.
// Neither backend computes a row identically at every batch shape, so the
// bound is three times the rounding the dense path itself shows between two
// feedings (measured first); the fixture keeps rounding away from every
// discrete choice (block and expert selection), so a real defect lands far
// above it. The exception is an 8-row verification against the one-token
// decodes it stands for: an x86-64 CPU computes those rows identically (1e-3
// bound there). Parity with llama.cpp is a separate check
// (eng/validation/minimax-m3-parity.cpp).
//
// Usage: GgmlOpsMiniMaxM3ExecTest cpu|cuda|cuda-multi|cuda-cpumoe|cuda-tp|metal [FIXTURE_DIR]
// Exit 77 when the backend (or the second GPU) is unavailable.
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#include "ggml.h"
#include "gguf.h"

extern "C" {
int TSGgml_IsBackendAvailable(int);
int TSGgml_GetGpuDeviceCount(int);
const char * TSGgml_GetLastError();
void * TSGgml_MiniMaxM3LoadModelTp(const char *, int, int, int, int, int, const char *, int, const char *, int);
void TSGgml_MiniMaxM3SetOption(const char *, const char *);
void TSGgml_MiniMaxM3Free(void *);
int TSGgml_MiniMaxM3VocabSize(void *);
int TSGgml_MiniMaxM3NPast(void *);
int TSGgml_MiniMaxM3Forward(void *, const int32_t *, int, float *, int);
int TSGgml_MiniMaxM3ForwardBatchedDecode(void *, int, const int32_t *, const int32_t *, const int32_t *, float *);
int TSGgml_MiniMaxM3ForwardSegments(void *, int, const int32_t *, const int32_t *, const int32_t *, const int32_t *, float *);
int TSGgml_MiniMaxM3ResetChecked(void *);
int TSGgml_MiniMaxM3Rewind(void *, int);
int TSGgml_MiniMaxM3SlotAlloc(void *);
int TSGgml_MiniMaxM3SetActiveSlot(void *, int);
int TSGgml_MiniMaxM3SlotFree(void *, int);
int TSGgml_MiniMaxM3SlotCopy(void *, int, int);
int TSGgml_MiniMaxM3SlotNPast(void *, int);
int TSGgml_MiniMaxM3GetStats(void *, int64_t *, int);
int TSGgml_MiniMaxM3SetReclaimableSlots(void *, const int32_t *, int);
int TSGgml_MiniMaxM3TakeReclaimedSlots(void *, int32_t *, int);
void TSGgml_MiniMaxM3TestInjectFault(int, int);
void TSGgml_MiniMaxM3TestMsaOldGpus(int);
}

namespace
{
constexpr int N_EMBD = 256, N_LAYER = 4, N_LEAD = 1, N_HEAD = 8, N_HEAD_KV = 2, HEAD_DIM = 128, N_ROT = 64;
constexpr int N_FF = 512, N_EXPERT = 8, N_USED = 2, N_FF_EXP = 128, N_VOCAB = 512;
constexpr int IDX_DIM = 128, IDX_TOPK = 4, IDX_BLK = 16;
constexpr int DENSE_SPAN = IDX_TOPK * IDX_BLK;   // 64

struct rng
{
    uint64_t s;
    explicit rng(uint64_t seed) : s(seed * 0x9E3779B97F4A7C15ull + 1) {}
    uint32_t next() { s ^= s << 13; s ^= s >> 7; s ^= s << 17; return (uint32_t) (s >> 32); }
    float uniform() { return (next() + 0.5f) / 4294967296.0f; }
    float normal()
    {
        const float u1 = uniform(), u2 = uniform();
        return sqrtf(-2.0f * logf(u1)) * cosf(6.2831853f * u2);
    }
};

/// One fixture tensor: F32 values quantized to `type` on the way into the GGUF.
static void add_tensor(gguf_context * gg, ggml_context * ctx, const char * name, ggml_type type,
                       std::vector<int64_t> ne, const std::vector<float> & v)
{
    ggml_tensor * t = nullptr;
    if (ne.size() == 1) t = ggml_new_tensor_1d(ctx, type, ne[0]);
    else if (ne.size() == 2) t = ggml_new_tensor_2d(ctx, type, ne[0], ne[1]);
    else t = ggml_new_tensor_3d(ctx, type, ne[0], ne[1], ne[2]);
    ggml_set_name(t, name);
    const int64_t n = ggml_nelements(t);
    if ((int64_t) v.size() != n) { fprintf(stderr, "fixture: %s size mismatch\n", name); exit(1); }
    if (type == GGML_TYPE_F32) memcpy(t->data, v.data(), (size_t) n * sizeof(float));
    else if (type == GGML_TYPE_BF16) ggml_fp32_to_bf16_row(v.data(), (ggml_bf16_t *) t->data, n);
    else ggml_quantize_chunk(type, v.data(), t->data, 0, n / ne[0], ne[0], nullptr);
    gguf_add_tensor(gg, t);
}

static std::vector<float> gauss(rng & r, size_t n, float sd, float mean = 0.0f)
{
    std::vector<float> v(n);
    for (auto & x : v) x = mean + sd * r.normal();
    return v;
}

static bool write_fixture(const std::string & path)
{
    gguf_context * gg = gguf_init_empty();
    gguf_set_val_str(gg, "general.architecture", "minimax-m3");
    auto u32 = [&](const char * k, uint32_t v) { gguf_set_val_u32(gg, (std::string("minimax-m3.") + k).c_str(), v); };
    auto f32 = [&](const char * k, float v) { gguf_set_val_f32(gg, (std::string("minimax-m3.") + k).c_str(), v); };
    u32("block_count", N_LAYER);
    u32("context_length", 8192);
    u32("embedding_length", N_EMBD);
    u32("feed_forward_length", N_FF);
    u32("attention.head_count", N_HEAD);
    u32("attention.head_count_kv", N_HEAD_KV);
    u32("attention.key_length", HEAD_DIM);
    u32("attention.value_length", HEAD_DIM);
    f32("rope.freq_base", 1.0e7f);
    u32("rope.dimension_count", N_ROT);
    f32("attention.layer_norm_rms_epsilon", 1e-6f);
    u32("expert_count", N_EXPERT);
    u32("expert_used_count", N_USED);
    u32("expert_gating_func", 2);
    u32("expert_feed_forward_length", N_FF_EXP);
    u32("expert_shared_count", 1);
    f32("expert_weights_scale", 2.0f);
    gguf_set_val_bool(gg, "minimax-m3.expert_weights_norm", true);
    u32("attention.indexer.head_count", N_HEAD_KV);
    u32("attention.indexer.key_length", IDX_DIM);
    u32("attention.indexer.top_k", IDX_TOPK);
    u32("attention.indexer.block_size", IDX_BLK);
    u32("attention.indexer.local_blocks", 1);
    u32("leading_dense_block_count", N_LEAD);

    ggml_init_params p = { (size_t) 64 << 20, nullptr, false };
    ggml_context * ctx = ggml_init(p);
    rng r(7);
    const float s_embd = 1.0f / sqrtf((float) N_EMBD);
    // Rounding must not decide anything discrete, or two ways of computing the
    // same rows diverge by far more than rounding (a flipped block or expert):
    //  * embedding dim 0 is a large positive constant that every layer's
    //    residual keeps, and the indexer projections read only that dim, so
    //    after their RMS norms the index queries and keys are fixed vectors
    //    and block scores depend on positions alone (through RoPE);
    //  * the router bias ranks the experts with gaps wider than the sigmoid's
    //    range (a different ranking per layer), so the routed experts are
    //    fixed per layer while their weights still vary per token.
    std::vector<float> embd = gauss(r, (size_t) N_EMBD * N_VOCAB, 1.0f);
    for (int t = 0; t < N_VOCAB; t++) embd[(size_t) t * N_EMBD] = 8.0f;
    add_tensor(gg, ctx, "token_embd.weight", GGML_TYPE_BF16, { N_EMBD, N_VOCAB }, embd);
    auto from_dim0 = [&](int64_t n_out) -> std::vector<float>
    {
        std::vector<float> w((size_t) N_EMBD * n_out, 0.0f);
        for (int64_t j = 0; j < n_out; j++) w[(size_t) (j * N_EMBD)] = r.normal();
        return w;
    };
    add_tensor(gg, ctx, "output_norm.weight", GGML_TYPE_F32, { N_EMBD }, gauss(r, N_EMBD, 0.05f, 1.0f));
    add_tensor(gg, ctx, "output.weight", GGML_TYPE_BF16, { N_EMBD, N_VOCAB }, gauss(r, (size_t) N_EMBD * N_VOCAB, 2.0f * s_embd));
    char nm[128];
    auto name = [&](int il, const char * s) { snprintf(nm, sizeof(nm), "blk.%d.%s", il, s); return nm; };
    for (int il = 0; il < N_LAYER; il++)
    {
        add_tensor(gg, ctx, name(il, "attn_norm.weight"), GGML_TYPE_F32, { N_EMBD }, gauss(r, N_EMBD, 0.05f, 1.0f));
        add_tensor(gg, ctx, name(il, "ffn_norm.weight"), GGML_TYPE_F32, { N_EMBD }, gauss(r, N_EMBD, 0.05f, 1.0f));
        add_tensor(gg, ctx, name(il, "attn_q.weight"), GGML_TYPE_Q8_0, { N_EMBD, N_HEAD * HEAD_DIM },
                   gauss(r, (size_t) N_EMBD * N_HEAD * HEAD_DIM, s_embd));
        add_tensor(gg, ctx, name(il, "attn_k.weight"), GGML_TYPE_Q8_0, { N_EMBD, N_HEAD_KV * HEAD_DIM },
                   gauss(r, (size_t) N_EMBD * N_HEAD_KV * HEAD_DIM, s_embd));
        add_tensor(gg, ctx, name(il, "attn_v.weight"), GGML_TYPE_Q8_0, { N_EMBD, N_HEAD_KV * HEAD_DIM },
                   gauss(r, (size_t) N_EMBD * N_HEAD_KV * HEAD_DIM, s_embd));
        add_tensor(gg, ctx, name(il, "attn_output.weight"), GGML_TYPE_Q8_0, { N_HEAD * HEAD_DIM, N_EMBD },
                   gauss(r, (size_t) N_HEAD * HEAD_DIM * N_EMBD, 0.5f / sqrtf((float) (N_HEAD * HEAD_DIM))));
        add_tensor(gg, ctx, name(il, "attn_q_norm.weight"), GGML_TYPE_F32, { HEAD_DIM }, gauss(r, HEAD_DIM, 0.05f, 1.0f));
        add_tensor(gg, ctx, name(il, "attn_k_norm.weight"), GGML_TYPE_F32, { HEAD_DIM }, gauss(r, HEAD_DIM, 0.05f, 1.0f));
        add_tensor(gg, ctx, name(il, "indexer.q_proj.weight"), GGML_TYPE_F32, { N_EMBD, N_HEAD_KV * IDX_DIM },
                   from_dim0(N_HEAD_KV * IDX_DIM));
        add_tensor(gg, ctx, name(il, "indexer.k_proj.weight"), GGML_TYPE_F32, { N_EMBD, IDX_DIM }, from_dim0(IDX_DIM));
        add_tensor(gg, ctx, name(il, "indexer.q_norm.weight"), GGML_TYPE_F32, { IDX_DIM }, gauss(r, IDX_DIM, 0.05f, 1.0f));
        add_tensor(gg, ctx, name(il, "indexer.k_norm.weight"), GGML_TYPE_F32, { IDX_DIM }, gauss(r, IDX_DIM, 0.05f, 1.0f));
        if (il < N_LEAD)
        {
            add_tensor(gg, ctx, name(il, "ffn_gate.weight"), GGML_TYPE_Q8_0, { N_EMBD, N_FF }, gauss(r, (size_t) N_EMBD * N_FF, s_embd));
            add_tensor(gg, ctx, name(il, "ffn_up.weight"), GGML_TYPE_Q8_0, { N_EMBD, N_FF }, gauss(r, (size_t) N_EMBD * N_FF, s_embd));
            add_tensor(gg, ctx, name(il, "ffn_down.weight"), GGML_TYPE_Q8_0, { N_FF, N_EMBD },
                       gauss(r, (size_t) N_FF * N_EMBD, 0.5f / sqrtf((float) N_FF)));
            continue;
        }
        add_tensor(gg, ctx, name(il, "ffn_gate_inp.weight"), GGML_TYPE_F32, { N_EMBD, N_EXPERT },
                   gauss(r, (size_t) N_EMBD * N_EXPERT, 4.0f * s_embd));
        std::vector<float> bias(N_EXPERT);
        for (int e = 0; e < N_EXPERT; e++) bias[(size_t) e] = 16.0f + 2.0f * (float) ((e * 3 + il * 5) % N_EXPERT);
        add_tensor(gg, ctx, name(il, "exp_probs_b.bias"), GGML_TYPE_F32, { N_EXPERT }, bias);
        // NVFP4 stacks hold W * G per expert; the .scale vector multiplies 1/G back
        // into each expert's output, as the converter's 1/weight_global_scale does.
        std::vector<float> sc_gate(N_EXPERT), sc_up(N_EXPERT), sc_down(N_EXPERT);
        auto stack = [&](const char * s, int64_t k, int64_t n, std::vector<float> & sc, float sd)
        {
            std::vector<float> w = gauss(r, (size_t) k * n * N_EXPERT, sd);
            for (int e = 0; e < N_EXPERT; e++)
            {
                const float g = 32.0f + 8.0f * (float) e;
                sc[(size_t) e] = 1.0f / g;
                for (int64_t i = 0; i < k * n; i++) w[(size_t) (e * k * n + i)] *= g;
            }
            add_tensor(gg, ctx, name(il, s), GGML_TYPE_NVFP4, { k, n, N_EXPERT }, w);
        };
        stack("ffn_gate_exps.weight", N_EMBD, N_FF_EXP, sc_gate, s_embd);
        stack("ffn_up_exps.weight", N_EMBD, N_FF_EXP, sc_up, s_embd);
        stack("ffn_down_exps.weight", N_FF_EXP, N_EMBD, sc_down, 0.5f / sqrtf((float) N_FF_EXP));
        add_tensor(gg, ctx, name(il, "ffn_gate_exps.scale"), GGML_TYPE_F32, { N_EXPERT }, sc_gate);
        add_tensor(gg, ctx, name(il, "ffn_up_exps.scale"), GGML_TYPE_F32, { N_EXPERT }, sc_up);
        add_tensor(gg, ctx, name(il, "ffn_down_exps.scale"), GGML_TYPE_F32, { N_EXPERT }, sc_down);
        add_tensor(gg, ctx, name(il, "ffn_gate_shexp.weight"), GGML_TYPE_Q8_0, { N_EMBD, N_FF_EXP },
                   gauss(r, (size_t) N_EMBD * N_FF_EXP, s_embd));
        add_tensor(gg, ctx, name(il, "ffn_up_shexp.weight"), GGML_TYPE_Q8_0, { N_EMBD, N_FF_EXP },
                   gauss(r, (size_t) N_EMBD * N_FF_EXP, s_embd));
        add_tensor(gg, ctx, name(il, "ffn_down_shexp.weight"), GGML_TYPE_Q8_0, { N_FF_EXP, N_EMBD },
                   gauss(r, (size_t) N_FF_EXP * N_EMBD, 0.5f / sqrtf((float) N_FF_EXP)));
    }
    const bool ok = gguf_write_to_file(gg, path.c_str(), false);
    gguf_free(gg);
    ggml_free(ctx);
    return ok;
}

struct config
{
    const char * backend;
    int n_gpu;
    int n_cpu_moe;
    const char * split;
};

static int g_failures = 0;
/// Bound for a verification row against the one-token decode of the same
/// position. An x86-64 CPU computes a row the same way at 1 and at 8 rows per
/// graph (vec-dot matmuls, the untiled flash attention with the visible rows
/// in ascending order), so there only rounding-free paths pass. ARM's Q8_0
/// kernel takes column pairs (1e-6 apart from one column on an Apple M-series
/// CPU) and CUDA switches kernels with the row count: main tol there.
static float g_verify_tol = 0.0f;

/// Largest |a - b| over rows [r0, r1) of two [rows, vocab] logit sets, and
/// how many of those rows agree on the argmax.
static void compare(const char * what, const std::vector<float> & a, const std::vector<float> & b, int64_t r0, int64_t r1,
                    float tol, int64_t a_off = 0, int64_t b_off = 0)
{
    double mx = 0.0;
    int64_t agree = 0;
    for (int64_t rr = r0; rr < r1; rr++)
    {
        const float * x = a.data() + (size_t) (rr - a_off) * N_VOCAB;
        const float * y = b.data() + (size_t) (rr - b_off) * N_VOCAB;
        int ax = 0, ay = 0;
        for (int i = 0; i < N_VOCAB; i++)
        {
            mx = std::max(mx, (double) fabsf(x[i] - y[i]));
            if (x[i] > x[ax]) ax = i;
            if (y[i] > y[ay]) ay = i;
        }
        agree += ax == ay ? 1 : 0;
    }
    const bool ok = mx <= tol && std::isfinite(mx);
    if (!ok) g_failures++;
    printf("%-58s rows %4lld-%-4lld max|d| %.3g (tol %.3g) top1 %lld/%lld %s\n", what, (long long) r0, (long long) r1 - 1,
           mx, (double) tol, (long long) agree, (long long) (r1 - r0), ok ? "ok" : "FAIL");
}

static void * load(const std::string & gguf, const config & c, int n_ctx = 2048, int tp = 0)
{
    void * h = TSGgml_MiniMaxM3LoadModelTp(gguf.c_str(), c.n_gpu, n_ctx, 64, 0, c.n_cpu_moe, c.backend, 1, c.split, tp);
    if (!h) fprintf(stderr, "load failed: %s\n", TSGgml_GetLastError());
    return h;
}

/// All-rows logits of tokens fed in pieces of `piece` (0: one call).
static bool feed(void * h, const std::vector<int32_t> & toks, int64_t from, int64_t to, int64_t piece, std::vector<float> & out)
{
    out.assign((size_t) (to - from) * N_VOCAB, 0.0f);
    if (piece <= 0) piece = to - from;
    for (int64_t off = from; off < to; off += piece)
    {
        const int64_t n = std::min(piece, to - off);
        if (!TSGgml_MiniMaxM3Forward(h, toks.data() + off, (int) n, out.data() + (size_t) (off - from) * N_VOCAB, 1)) return false;
    }
    return true;
}

static std::vector<int32_t> tokens(uint64_t seed, int64_t n)
{
    rng r(seed);
    std::vector<int32_t> t((size_t) n);
    for (auto & x : t) x = (int32_t) (r.next() % N_VOCAB);
    return t;
}

/// The rounding the plain dense path shows between a one-shot prefill and both
/// 37-token calls and token-by-token decode of the same tokens (max |d|).
static float dense_noise(const std::string & gguf, const config & c, int tp)
{
    TSGgml_MiniMaxM3SetOption("TS_MINIMAX_M3_MSA", "dense");
    void * hd = load(gguf, c, 2048, tp);
    TSGgml_MiniMaxM3SetOption("TS_MINIMAX_M3_MSA", nullptr);
    if (!hd) return -1.0f;
    const auto toks = tokens(11, 200);
    std::vector<float> a, b, d((size_t) 80 * N_VOCAB);
    bool ok = feed(hd, toks, 0, 200, 0, a) && TSGgml_MiniMaxM3ResetChecked(hd) && feed(hd, toks, 0, 200, 37, b) &&
              TSGgml_MiniMaxM3Rewind(hd, 120);
    for (int p = 120; ok && p < 200; p++)
        ok = TSGgml_MiniMaxM3Forward(hd, &toks[(size_t) p], 1, d.data() + (size_t) (p - 120) * N_VOCAB, 0) != 0;
    TSGgml_MiniMaxM3Free(hd);
    if (!ok) return -1.0f;
    float noise = 0.0f;
    for (size_t i = 0; i < a.size(); i++) noise = std::max(noise, fabsf(a[i] - b[i]));
    for (size_t i = 0; i < d.size(); i++) noise = std::max(noise, fabsf(a[(size_t) 120 * N_VOCAB + i] - d[i]));
    return noise;
}

/// Every single-model check on one loaded model. `ref` receives the one-shot
/// logits of the main sequence for cross-placement comparisons.
static bool run_model(void * h, float tol, std::vector<float> & ref)
{
    const int64_t N = 200;   // dense rows 0..63, sparse 64..199
    const auto toks = tokens(11, N);
    if (!TSGgml_MiniMaxM3ResetChecked(h) || !feed(h, toks, 0, N, 0, ref)) { fprintf(stderr, "one-shot forward failed\n"); return false; }

    std::vector<float> got;
    TSGgml_MiniMaxM3ResetChecked(h);
    if (!feed(h, toks, 0, N, 37, got)) return false;
    compare("chunked prefill (37-token calls) == one-shot", ref, got, 0, N, tol);

    // decode continuation: prefill 120 rows, then one token per call
    TSGgml_MiniMaxM3ResetChecked(h);
    if (!feed(h, toks, 0, 120, 0, got)) return false;
    std::vector<float> dec((size_t) (N - 120) * N_VOCAB);
    for (int64_t p = 120; p < N; p++)
        if (!TSGgml_MiniMaxM3Forward(h, &toks[(size_t) p], 1, dec.data() + (size_t) (p - 120) * N_VOCAB, 0)) return false;
    compare("decode (1 token per call) == one-shot", ref, dec, 120, N, tol, 0, 120);

    // a prompt whose earlier ubatches leave only caches behind (last row only;
    // on a layer split this is the pipelined prefill), then decode on them
    {
        TSGgml_MiniMaxM3ResetChecked(h);
        std::vector<float> last((size_t) N_VOCAB), dec2((size_t) (N - 180) * N_VOCAB);
        if (!TSGgml_MiniMaxM3Forward(h, toks.data(), 180, last.data(), 0)) return false;
        compare("180-token prompt, last row only == one-shot", ref, last, 179, 180, tol, 0, 179);
        for (int64_t p = 180; p < N; p++)
            if (!TSGgml_MiniMaxM3Forward(h, &toks[(size_t) p], 1, dec2.data() + (size_t) (p - 180) * N_VOCAB, 0)) return false;
        compare("decode after that prompt == one-shot", ref, dec2, 180, N, tol, 0, 180);
    }

    // verification: one 8-row all-rows call == the 8 one-token decodes it
    // stands for, over the same cached prefix (speculative decoding must not
    // change the output), in the dense range and past the dense span
    for (const int64_t at : { (int64_t) 40, (int64_t) 150 })
    {
        const int64_t nv = 8;
        std::vector<float> dec8((size_t) nv * N_VOCAB), ver8((size_t) nv * N_VOCAB);
        TSGgml_MiniMaxM3ResetChecked(h);
        if (!feed(h, toks, 0, at, 0, got)) return false;
        for (int64_t i = 0; i < nv; i++)
            if (!TSGgml_MiniMaxM3Forward(h, &toks[(size_t) (at + i)], 1, dec8.data() + (size_t) i * N_VOCAB, 0)) return false;
        if (!TSGgml_MiniMaxM3Rewind(h, (int) at)) { fprintf(stderr, "rewind failed\n"); return false; }
        if (!TSGgml_MiniMaxM3Forward(h, &toks[(size_t) at], (int) nv, ver8.data(), 1)) return false;
        char what[80];
        snprintf(what, sizeof(what), "8-row verification at %lld == 8 one-token decodes", (long long) at);
        compare(what, dec8, ver8, 0, nv, g_verify_tol);
    }

    // rewind: the head moves back, re-feeding the same tokens reproduces the
    // rows -- exactly those of the same calls without the detour (the
    // verification rows 150..157 left behind are overwritten or masked)
    std::vector<float> fresh;
    if (!TSGgml_MiniMaxM3Rewind(h, 150) || TSGgml_MiniMaxM3NPast(h) != 150) { fprintf(stderr, "rewind failed\n"); return false; }
    if (!feed(h, toks, 150, N, 0, got)) return false;
    compare("rewind to 150 + re-forward == one-shot", ref, got, 150, N, tol, 0, 150);
    TSGgml_MiniMaxM3ResetChecked(h);
    if (!feed(h, toks, 0, 150, 0, fresh) || !feed(h, toks, 150, N, 0, fresh)) return false;
    compare("rewind to 150 + re-forward == the same calls, no rewind", fresh, got, 0, N - 150, 0.0f);

    // slot copy: a copy at 100 continues exactly like the original
    const int s1 = TSGgml_MiniMaxM3SlotAlloc(h);
    if (s1 < 0) { fprintf(stderr, "slot alloc failed\n"); return false; }
    TSGgml_MiniMaxM3ResetChecked(h);
    if (!feed(h, toks, 0, 100, 0, got)) return false;
    // the source continues with other tokens afterwards: the copy must not see them
    const int s0 = 0;
    if (!TSGgml_MiniMaxM3SlotCopy(h, s0, s1)) { fprintf(stderr, "slot copy failed\n"); return false; }
    const auto other = tokens(99, 60);
    if (!feed(h, other, 0, 60, 0, got)) return false;
    TSGgml_MiniMaxM3SetActiveSlot(h, s1);
    if (TSGgml_MiniMaxM3NPast(h) != 100) { fprintf(stderr, "copied slot head %d != 100\n", TSGgml_MiniMaxM3NPast(h)); return false; }
    if (!feed(h, toks, 100, N, 0, got)) return false;
    compare("slot copy at 100 + continue == one-shot", ref, got, 100, N, tol, 0, 100);
    TSGgml_MiniMaxM3SetActiveSlot(h, s0);
    TSGgml_MiniMaxM3SlotFree(h, s1);
    // ...and exactly like the source slot continuing with the same calls
    TSGgml_MiniMaxM3ResetChecked(h);
    if (!feed(h, toks, 0, 100, 0, fresh) || !feed(h, toks, 100, N, 0, fresh)) return false;
    compare("slot copy at 100 + continue == the source continuing", fresh, got, 0, N - 100, 0.0f);

    // 3-slot batched decode == each slot decoded alone (one dense, two sparse slots)
    const int64_t plen[3] = { 30, 90, 150 };
    const int64_t steps = 12;
    std::vector<std::vector<int32_t>> seq;
    for (int i = 0; i < 3; i++) seq.push_back(tokens(200 + (uint64_t) i, plen[i] + steps));
    std::vector<float> alone((size_t) 3 * steps * N_VOCAB), batched((size_t) 3 * steps * N_VOCAB);
    for (int i = 0; i < 3; i++)
    {
        TSGgml_MiniMaxM3ResetChecked(h);
        if (!feed(h, seq[(size_t) i], 0, plen[i], 0, got)) return false;
        for (int64_t s = 0; s < steps; s++)
            if (!TSGgml_MiniMaxM3Forward(h, &seq[(size_t) i][(size_t) (plen[i] + s)], 1,
                                         alone.data() + ((size_t) i * steps + (size_t) s) * N_VOCAB, 0)) return false;
    }
    int slots[3] = { s0, TSGgml_MiniMaxM3SlotAlloc(h), TSGgml_MiniMaxM3SlotAlloc(h) };
    if (slots[1] < 0 || slots[2] < 0) { fprintf(stderr, "slot alloc failed\n"); return false; }
    for (int i = 0; i < 3; i++)
    {
        TSGgml_MiniMaxM3SetActiveSlot(h, slots[i]);
        TSGgml_MiniMaxM3ResetChecked(h);
        if (!feed(h, seq[(size_t) i], 0, plen[i], 0, got)) return false;
    }
    TSGgml_MiniMaxM3SetActiveSlot(h, s0);
    std::vector<float> step_out((size_t) 3 * N_VOCAB);
    for (int64_t s = 0; s < steps; s++)
    {
        int32_t tk[3], pos[3];
        for (int i = 0; i < 3; i++) { tk[i] = seq[(size_t) i][(size_t) (plen[i] + s)]; pos[i] = (int32_t) (plen[i] + s); }
        if (!TSGgml_MiniMaxM3ForwardBatchedDecode(h, 3, slots, tk, pos, step_out.data())) { fprintf(stderr, "batched decode failed\n"); return false; }
        for (int i = 0; i < 3; i++)
            memcpy(batched.data() + ((size_t) i * steps + (size_t) s) * N_VOCAB, step_out.data() + (size_t) i * N_VOCAB,
                   N_VOCAB * sizeof(float));
    }
    compare("3-slot batched decode == each slot alone", alone, batched, 0, 3 * steps, tol);

    // segments: one slot prefills 9 tokens while two others decode, in one graph
    for (int i = 0; i < 3; i++)
    {
        TSGgml_MiniMaxM3SetActiveSlot(h, slots[i]);
        TSGgml_MiniMaxM3ResetChecked(h);
        if (!feed(h, seq[(size_t) i], 0, plen[i], 0, got)) return false;
    }
    TSGgml_MiniMaxM3SetActiveSlot(h, s0);
    {
        const int32_t lens[3] = { 1, 9, 1 };
        std::vector<int32_t> tk;
        int32_t pos[3];
        for (int i = 0; i < 3; i++)
        {
            for (int j = 0; j < lens[i]; j++) tk.push_back(seq[(size_t) i][(size_t) (plen[i] + j)]);
            pos[i] = (int32_t) plen[i];
        }
        if (!TSGgml_MiniMaxM3ForwardSegments(h, 3, slots, lens, tk.data(), pos, step_out.data())) { fprintf(stderr, "segments failed\n"); return false; }
        // reference: each slot alone, last token of its segment
        std::vector<float> seg_ref((size_t) 3 * N_VOCAB), row((size_t) 9 * N_VOCAB);
        for (int i = 0; i < 3; i++)
        {
            TSGgml_MiniMaxM3SetActiveSlot(h, slots[i]);
            if (!TSGgml_MiniMaxM3Rewind(h, (int) plen[i])) { fprintf(stderr, "rewind failed\n"); return false; }
            if (!TSGgml_MiniMaxM3Forward(h, &seq[(size_t) i][(size_t) plen[i]], lens[i], row.data(), 0)) return false;
            memcpy(seg_ref.data() + (size_t) i * N_VOCAB, row.data(), N_VOCAB * sizeof(float));
        }
        TSGgml_MiniMaxM3SetActiveSlot(h, s0);
        compare("segments (decode + 9-token prefill + decode) == alone", seg_ref, step_out, 0, 3, tol);
    }
    TSGgml_MiniMaxM3SlotFree(h, slots[1]);
    TSGgml_MiniMaxM3SlotFree(h, slots[2]);

    int64_t st[16] = {};
    const int n_st = TSGgml_MiniMaxM3GetStats(h, st, 16);
    printf("stats: graphs built %lld reused %lld, sparse segments %lld, dense segments %lld, sparse layers %lld (%d values)\n",
           (long long) st[0], (long long) st[1], (long long) st[13], (long long) st[14], (long long) st[15], n_st);
    // auto (st[12] 0) runs dense attention below the dense span and sparse past
    // it; dense (1) never runs sparse; ggml (2) and kernel (3) always do (the
    // kernel computes the dense span's exact shortcut itself).
    const bool want_sparse = n_st >= 16 && st[12] != 1, want_dense = n_st >= 16 && (st[12] == 0 || st[12] == 1);
    if (n_st < 16 || (st[13] > 0) != want_sparse || (st[14] > 0) != want_dense || st[15] != N_LAYER)
    {
        printf("expected %s segments on %d sparse layers (attention mode %lld): FAIL\n",
               want_sparse && want_dense ? "sparse and dense" : want_sparse ? "only sparse" : "only dense", N_LAYER,
               (long long) st[12]);
        g_failures++;
    }
    return true;
}
/// A slot copy whose destination has to grow while memory is short (every
/// admission refused through a huge TS_MINIMAX_M3_SLOT_HEADROOM_MB) frees
/// retained slots for the growth, but never the copy's own source. On a layer
/// split the growth then proceeds unchecked and the copy decodes like its
/// source; under --tp (no unchecked growth) the copy is refused and the source
/// survives either way.
static void check_slot_copy_keeps_source(const std::string & gguf, const config & c, int tp, float tol)
{
    // New slots start at 256 rows while this check runs (the option is read per allocation).
    TSGgml_MiniMaxM3SetOption("TS_MINIMAX_M3_SLOT_INIT_ROWS", "256");
    void * h = load(gguf, c, 2048, tp);
    if (!h) { TSGgml_MiniMaxM3SetOption("TS_MINIMAX_M3_SLOT_INIT_ROWS", nullptr); g_failures++; return; }
    const auto toks = tokens(51, 301);
    std::vector<float> got;
    const int other = TSGgml_MiniMaxM3SlotAlloc(h), src = TSGgml_MiniMaxM3SlotAlloc(h);
    bool ok = other >= 0 && src >= 0 && TSGgml_MiniMaxM3SetActiveSlot(h, other) && feed(h, toks, 0, 40, 0, got) &&
              TSGgml_MiniMaxM3SetActiveSlot(h, src) && feed(h, toks, 0, 300, 0, got) && TSGgml_MiniMaxM3SetActiveSlot(h, 0);
    const int dst = ok ? TSGgml_MiniMaxM3SlotAlloc(h) : -1;   // 256 rows: the copy of 300 must grow it
    const int32_t retained[2] = { other, src };
    ok = ok && dst >= 0 && TSGgml_MiniMaxM3SetReclaimableSlots(h, retained, 2);
    TSGgml_MiniMaxM3SetOption("TS_MINIMAX_M3_SLOT_HEADROOM_MB", "100000000");
    const bool copied = ok && TSGgml_MiniMaxM3SlotCopy(h, src, dst) != 0;
    TSGgml_MiniMaxM3SetOption("TS_MINIMAX_M3_SLOT_HEADROOM_MB", nullptr);
    int32_t reclaimed[8] = {};
    const int n_reclaimed = ok ? TSGgml_MiniMaxM3TakeReclaimedSlots(h, reclaimed, 8) : -1;
    const bool kept = ok && TSGgml_MiniMaxM3SlotNPast(h, src) == 300;
    const bool other_freed = n_reclaimed == 1 && reclaimed[0] == other && TSGgml_MiniMaxM3SlotNPast(h, other) == -1;
    const bool want_copy = tp <= 1;
    bool same = !want_copy;
    if (want_copy && copied)
    {
        // the copy and its source decode the same next token identically
        std::vector<float> a((size_t) N_VOCAB), b((size_t) N_VOCAB);
        same = TSGgml_MiniMaxM3SlotNPast(h, dst) == 300 && TSGgml_MiniMaxM3SetActiveSlot(h, src) &&
               TSGgml_MiniMaxM3Forward(h, &toks[300], 1, a.data(), 0) && TSGgml_MiniMaxM3SetActiveSlot(h, dst) &&
               TSGgml_MiniMaxM3Forward(h, &toks[300], 1, b.data(), 0);
        float d = 0.0f;
        for (int i = 0; same && i < N_VOCAB; i++) d = std::max(d, fabsf(a[(size_t) i] - b[(size_t) i]));
        same = same && d <= tol;
    }
    const bool pass = ok && kept && other_freed && copied == want_copy && same;
    printf("%-58s %s\n", tp > 1 ? "tp: copy under memory pressure keeps its source (refused)"
                                 : "copy under memory pressure frees others, keeps its source", pass ? "ok" : "FAIL");
    if (!pass)
    {
        printf("  setup %d, copied %d, source head %d, reclaimed %d slot(s) (first %d, other %d), copy matches %d\n", ok,
               copied, TSGgml_MiniMaxM3SlotNPast(h, src), n_reclaimed, n_reclaimed > 0 ? reclaimed[0] : -1, other, same);
        g_failures++;
    }
    TSGgml_MiniMaxM3Free(h);
    TSGgml_MiniMaxM3SetOption("TS_MINIMAX_M3_SLOT_INIT_ROWS", nullptr);
}

/// Faults injected into the load's thread teams are load refusals with a
/// reason, never a terminated process; a progress thread that cannot start
/// costs only the progress lines.
static void check_load_faults(const std::string & gguf, const config & c, int tp)
{
    struct fault { const char * what; int thread_start, upload_job; const char * says; const char * alternate; };
    const fault cases[] = {
        { "a loader thread that cannot start is a load refusal", 1, -1, "cannot start loader thread", nullptr },
        // std::bad_alloc::what() is implementation-defined; MSVC says "bad allocation".
        { "an upload worker that throws is a load refusal", -1, 0, "bad_alloc", "bad allocation" },
        { "a progress thread that cannot start costs only its lines", 0, -1, nullptr, nullptr },
    };
    for (const fault & f : cases)
    {
        TSGgml_MiniMaxM3TestInjectFault(f.thread_start, f.upload_job);
        void * h = load(gguf, c, 2048, tp);
        TSGgml_MiniMaxM3TestInjectFault(-1, -1);
        const char * err = TSGgml_GetLastError();
        const bool ok = f.says ? !h && err && (strstr(err, f.says) ||
                                             (f.alternate && strstr(err, f.alternate))) : h != nullptr;
        if (h) TSGgml_MiniMaxM3Free(h);
        printf("%-58s %s\n", f.what, ok ? "ok" : "FAIL");
        if (!ok) { printf("  last error: %s\n", err ? err : "(none)"); g_failures++; }
    }
}

/// TS_MINIMAX_M3_MSA=kernel on a GPU that cannot run the kernels (`old`: the
/// test hook makes every CUDA GPU report compute capability 7.5; Metal never
/// can) is refused; auto loads there with no layer on the kernels.
static void check_kernel_gate(const std::string & gguf, const config & c, bool old_cuda)
{
    if (old_cuda) TSGgml_MiniMaxM3TestMsaOldGpus(1);
    TSGgml_MiniMaxM3SetOption("TS_MINIMAX_M3_MSA", "kernel");
    void * h = load(gguf, c);
    TSGgml_MiniMaxM3SetOption("TS_MINIMAX_M3_MSA", nullptr);
    const char * err = TSGgml_GetLastError();
    const bool refused = !h && err && strstr(err, old_cuda ? "compute capability" : "has none");
    if (h) TSGgml_MiniMaxM3Free(h);
    printf("%-58s %s\n", old_cuda ? "MSA=kernel on a CUDA GPU older than 8.0 is refused" : "MSA=kernel on a Metal GPU is refused",
           refused ? "ok" : "FAIL");
    if (!refused) { printf("  last error: %s\n", err ? err : "(none)"); g_failures++; }
    h = load(gguf, c);
    if (old_cuda) TSGgml_MiniMaxM3TestMsaOldGpus(0);
    int64_t st[17] = {};
    const int n = h ? TSGgml_MiniMaxM3GetStats(h, st, 17) : 0;
    std::vector<float> got;
    const auto toks = tokens(11, 150);
    const bool runs = h && n >= 17 && st[16] == 0 && feed(h, toks, 0, 150, 0, got);
    bool finite = runs;
    for (float x : got) finite = finite && std::isfinite(x);
    printf("%-58s %s\n", "...where auto runs every layer on upstream ggml's ops", finite ? "ok" : "FAIL");
    if (!finite) { printf("  loaded %d, %d stats, kernel layers %lld\n", h != nullptr, n, (long long) st[16]); g_failures++; }
    if (h) TSGgml_MiniMaxM3Free(h);
}
} // namespace

int main(int argc, char ** argv)
{
    const std::string mode = argc > 1 ? argv[1] : "cpu";
    const bool metal = mode == "metal";
    const bool cuda = mode != "cpu" && !metal;
    if (!TSGgml_IsBackendAvailable(cuda ? 3 : metal ? 1 : 2)) { printf("SKIP: backend unavailable\n"); return 77; }
    if ((mode == "cuda-multi" || mode == "cuda-tp") && TSGgml_GetGpuDeviceCount(3) < 2) { printf("SKIP: needs two GPUs\n"); return 77; }
    std::string dir = argc > 2 ? argv[2] : "";
    if (dir.empty())
    {
        const char * tmp = getenv("TMPDIR");
        dir = tmp && *tmp ? tmp : "/tmp";
    }
    const std::string gguf = dir + "/minimax-m3-exec-test.gguf";
    if (!write_fixture(gguf)) { fprintf(stderr, "cannot write %s\n", gguf.c_str()); return 1; }

    if (metal)
    {
        // The MSA kernels exist for CUDA only: kernel mode is refused on a Metal GPU.
        check_kernel_gate(gguf, config{ "Metal", 1, 0, nullptr }, false);
        remove(gguf.c_str());
        printf("%s: %d failure(s)\n", mode.c_str(), g_failures);
        return g_failures == 0 ? 0 : 1;
    }
    const config base = cuda ? config{ "CUDA", 1, 0, nullptr } : config{ "CPU", 0, 0, nullptr };
    std::vector<float> ref, ref2;

    // Neither backend is batch-shape invariant (the CPU's tiled GEMM and flash
    // attention, CUDA's MMVQ/MMQ and vector/MMA kernels), so the bound is set
    // by the rounding the plain dense path shows between a one-shot prefill
    // and both 37-token calls and token-by-token decode of the same tokens: a
    // dropped or extra block, a wrong row or a stale cache shows up far above it.
    float tol = 0.0f;
    {
        TSGgml_MiniMaxM3SetOption("TS_MINIMAX_M3_MSA", "dense");
        void * hd = load(gguf, base);
        if (!hd) return 1;
        const auto toks = tokens(11, 200);
        std::vector<float> a, b, c((size_t) 80 * N_VOCAB);
        if (!feed(hd, toks, 0, 200, 0, a) || !TSGgml_MiniMaxM3ResetChecked(hd) || !feed(hd, toks, 0, 200, 37, b)) return 1;
        if (!TSGgml_MiniMaxM3Rewind(hd, 120)) return 1;
        for (int p = 120; p < 200; p++)
            if (!TSGgml_MiniMaxM3Forward(hd, &toks[(size_t) p], 1, c.data() + (size_t) (p - 120) * N_VOCAB, 0)) return 1;
        TSGgml_MiniMaxM3Free(hd);
        TSGgml_MiniMaxM3SetOption("TS_MINIMAX_M3_MSA", nullptr);
        float noise = 0.0f;
        for (size_t i = 0; i < a.size(); i++) noise = std::max(noise, fabsf(a[i] - b[i]));
        for (size_t i = 0; i < c.size(); i++) noise = std::max(noise, fabsf(a[(size_t) 120 * N_VOCAB + i] - c[i]));
        tol = std::max(1e-3f, 3.0f * noise);
        printf("rounding of the dense path (one-shot vs 37-token calls and decode): max|d| %.3g -> bound %.3g\n", noise, tol);
    }

#if defined(__x86_64__) || defined(_M_X64)
    g_verify_tol = cuda ? tol : 1e-3f;
#else
    g_verify_tol = tol;
#endif
    void * h = load(gguf, base);
    if (!h) return 1;
    if (!run_model(h, tol, ref)) { TSGgml_MiniMaxM3Free(h); return 1; }
    TSGgml_MiniMaxM3Free(h);

    // The sparse path below the dense span selects every visible block: it is
    // dense attention there (the "auto" shortcut's justification).
    {
        const auto toks = tokens(11, DENSE_SPAN);
        std::vector<float> dense_rows, sparse_rows;
        TSGgml_MiniMaxM3SetOption("TS_MINIMAX_M3_MSA", "dense");
        h = load(gguf, base);
        if (!h || !feed(h, toks, 0, DENSE_SPAN, 0, dense_rows)) return 1;
        TSGgml_MiniMaxM3Free(h);
        TSGgml_MiniMaxM3SetOption("TS_MINIMAX_M3_MSA", "ggml");
        h = load(gguf, base);
        if (!h || !feed(h, toks, 0, DENSE_SPAN, 0, sparse_rows)) return 1;
        std::vector<float> dec((size_t) 16 * N_VOCAB), sparse_dec;
        // ...and the forced-sparse decode path agrees there too
        if (!TSGgml_MiniMaxM3Rewind(h, DENSE_SPAN - 16)) return 1;
        for (int p = DENSE_SPAN - 16; p < DENSE_SPAN; p++)
            if (!TSGgml_MiniMaxM3Forward(h, &toks[(size_t) p], 1, dec.data() + (size_t) (p - (DENSE_SPAN - 16)) * N_VOCAB, 0)) return 1;
        TSGgml_MiniMaxM3Free(h);
        TSGgml_MiniMaxM3SetOption("TS_MINIMAX_M3_MSA", nullptr);
        compare("forced sparse prefill == dense below the dense span", dense_rows, sparse_rows, 0, DENSE_SPAN, tol);
        compare("forced sparse decode == dense below the dense span", dense_rows, dec, DENSE_SPAN - 16, DENSE_SPAN, tol, 0,
                DENSE_SPAN - 16);
    }

    if (mode == "cuda-multi" || mode == "cuda-cpumoe")
    {
        const config other = mode == "cuda-multi" ? config{ "CUDA", 2, 0, "2,2" } : config{ "CUDA", 1, 2, nullptr };
        h = load(gguf, other);
        if (!h) return 1;
        int64_t st[16] = {};
        TSGgml_MiniMaxM3GetStats(h, st, 16);
        printf("%s: %lld GPU(s), %lld layer(s) with host experts\n", mode.c_str(), (long long) st[4], (long long) st[3]);
        if (mode == "cuda-multi" && st[4] != 2) { printf("expected 2 GPUs: FAIL\n"); g_failures++; }
        if (mode == "cuda-cpumoe" && st[3] != 1) { printf("expected 1 MoE layer with host experts (layers 0-1, layer 0 dense): FAIL\n"); g_failures++; }
        const int64_t pipelined_before = st[6];
        if (!run_model(h, tol, ref2)) { TSGgml_MiniMaxM3Free(h); return 1; }
        TSGgml_MiniMaxM3GetStats(h, st, 16);
        TSGgml_MiniMaxM3Free(h);
        if (mode == "cuda-multi")
        {
            // the 180-token prompt of run_model pipelines over the two GPUs
            printf("pipelined prefill calls during the checks: %lld (micro-batches %lld) %s\n",
                   (long long) (st[6] - pipelined_before), (long long) st[7], st[6] > pipelined_before ? "ok" : "FAIL");
            if (st[6] <= pipelined_before) g_failures++;
        }
        compare(mode == "cuda-multi" ? "2-GPU layer split == 1 GPU" : "host experts (n_cpu_moe 2) == all on GPU", ref, ref2, 0,
                (int64_t) (ref.size() / N_VOCAB), tol);
    }

    if (mode == "cuda-tp")
    {
        // Tensor parallelism over 2 GPUs: the FFN's partial sums meet in an
        // all-reduce (whose NCCL path reduces large tensors in BF16), so the
        // bound is the TP model's own dense-path rounding, measured the same way.
        const float noise_tp = dense_noise(gguf, base, 2);
        if (noise_tp < 0.0f) { fprintf(stderr, "tensor-parallel dense load failed: %s\n", TSGgml_GetLastError()); return 1; }
        const float tol_tp = std::max(tol, 3.0f * noise_tp);
        printf("rounding of the tensor-parallel dense path: max|d| %.3g -> bound %.3g\n", noise_tp, tol_tp);
        h = load(gguf, base, 2048, 2);
        if (!h) return 1;
        int64_t st[16] = {};
        TSGgml_MiniMaxM3GetStats(h, st, 16);
        printf("cuda-tp: tensor-parallel degree %lld, %lld layer(s) with host experts, pipelined %lld\n", (long long) st[11],
               (long long) st[3], (long long) st[9]);
        if (st[11] != 2 || st[3] != 0 || st[9] != 0) { printf("expected tp 2, no host experts, no pipeline: FAIL\n"); g_failures++; }
        g_verify_tol = tol_tp;
        if (!run_model(h, tol_tp, ref2)) { TSGgml_MiniMaxM3Free(h); return 1; }
        TSGgml_MiniMaxM3Free(h);
        compare("tensor parallel over 2 GPUs == 1 GPU", ref, ref2, 0, (int64_t) (ref.size() / N_VOCAB), tol_tp);
        // ...and the sparse path forced below the dense span agrees there under TP too
        const auto toks = tokens(11, DENSE_SPAN);
        std::vector<float> dense_rows, sparse_rows;
        TSGgml_MiniMaxM3SetOption("TS_MINIMAX_M3_MSA", "ggml");
        h = load(gguf, base, 2048, 2);
        TSGgml_MiniMaxM3SetOption("TS_MINIMAX_M3_MSA", nullptr);
        if (!h || !feed(h, toks, 0, DENSE_SPAN, 0, sparse_rows)) return 1;
        std::vector<float> dec((size_t) 16 * N_VOCAB);
        if (!TSGgml_MiniMaxM3Rewind(h, DENSE_SPAN - 16)) return 1;
        for (int p = DENSE_SPAN - 16; p < DENSE_SPAN; p++)
            if (!TSGgml_MiniMaxM3Forward(h, &toks[(size_t) p], 1, dec.data() + (size_t) (p - (DENSE_SPAN - 16)) * N_VOCAB, 0)) return 1;
        TSGgml_MiniMaxM3Free(h);
        compare("tp: forced sparse prefill == 1-GPU dense below the dense span", ref, sparse_rows, 0, DENSE_SPAN, tol_tp);
        compare("tp: forced sparse decode == 1-GPU dense below the dense span", ref, dec, DENSE_SPAN - 16, DENSE_SPAN, tol_tp, 0,
                DENSE_SPAN - 16);
    }

    // Failure paths (see the header).
    check_load_faults(gguf, base, mode == "cuda-tp" ? 2 : 0);
    if (cuda) check_slot_copy_keeps_source(gguf, base, mode == "cuda-tp" ? 2 : 0, tol);
    if (mode == "cuda") check_kernel_gate(gguf, base, true);

    // What tensor parallelism cannot run is refused with a reason, before any GPU is touched.
    {
        struct refusal { const char * what; const char * backend; const char * split; int cpu_moe; const char * msa; int tp; const char * says; };
        const int n_cuda = cuda ? TSGgml_GetGpuDeviceCount(3) : 0;
        std::vector<refusal> cases = {
            { "--tp 2 on the CPU backend is refused", "CPU", nullptr, 0, nullptr, 2, "CUDA GPUs only" },
            { "--tp 2 with a layer split is refused", "CUDA", "2,2", 0, nullptr, 2, "cannot be combined" },
            { "--tp 2 with --n-cpu-moe is refused", "CUDA", nullptr, 2, nullptr, 2, "--n-cpu-moe" },
            { "--tp 2 with TS_MINIMAX_M3_MSA=kernel is refused", "CUDA", nullptr, 0, "kernel", 2, "meta backend" },
        };
        if (cuda) cases.push_back({ "--tp past the visible GPUs is refused", "CUDA", nullptr, 0, nullptr, n_cuda + 1, "are visible" });
        for (const refusal & r : cases)
        {
            if (r.msa) TSGgml_MiniMaxM3SetOption("TS_MINIMAX_M3_MSA", r.msa);
            void * t = TSGgml_MiniMaxM3LoadModelTp(gguf.c_str(), 2, 2048, 256, 0, r.cpu_moe, r.backend, 1, r.split, r.tp);
            if (r.msa) TSGgml_MiniMaxM3SetOption("TS_MINIMAX_M3_MSA", nullptr);
            const char * err = TSGgml_GetLastError();
            const bool ok = !t && err && strstr(err, r.says);
            if (t) TSGgml_MiniMaxM3Free(t);
            printf("%-58s %s\n", r.what, ok ? "ok" : "FAIL");
            if (!ok) { printf("  last error: %s\n", err ? err : "(none)"); g_failures++; }
        }
    }
    remove(gguf.c_str());
    printf("%s: %d failure(s)\n", mode.c_str(), g_failures);
    return g_failures == 0 ? 0 : 1;
}
