// Copyright (c) Zhongkai Fu. All rights reserved.
// Licensed under the BSD-3-Clause license in the repository root.
//
// The DFlash draft with its block written into the ring at the block's own slots (the default) against the draft that
// concatenates the ring with the block in every layer (TS_DFLASH_RING_BLOCK=0): the same drafts, confidences and
// selector lattices over a wrapping ring, step after step (replayed graphs included), and the ring rows a later window
// can read left exactly as the injects wrote them. On CUDA the batched draft (two lanes, each on its own ring) is checked
// against the single-lane drafts too. On Metal the fused convolution (TS_DFLASH_METAL_FUSE: row-offset views) is
// checked bit for bit against the per-tap chains.
//
// The selector keeps every token (top-k = the vocabulary) and its lattice is compared keyed by token ids: the drafts
// differ by float rounding, which can reorder near-tied candidates (on CUDA's attention, ~6e-5) and so permute the
// lattice without changing a single score.
#include "ggml.h"
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

extern "C" {
int TSGgml_IsBackendAvailable(int);
const char* TSGgml_GetLastError();
int TSGgml_SyncHostBuffer(void*, size_t);
void TSGgml_DFlashResetCaches();
void TSGgml_ClearHostBufferCache();
void TSGgml_Shutdown();
int TSGgml_DFlashInjectScaled(
    const float* feat_rows, int feature_size, int n_rows, const std::int64_t* ring_rows_idx, const int* positions,
    int num_layers, int hidden_size, int head_dim, int num_kv_heads, int ring_rows,
    float eps, float rope_base, float rope_freq_scale,
    const void* fc_data, int fc_type, std::int64_t fc_ne0, std::int64_t fc_ne1, std::int64_t fc_bytes,
    const void* enc_norm_data,
    void** k_arr, int* k_type_arr, std::int64_t* k_ne0_arr, std::int64_t* k_ne1_arr, std::int64_t* k_bytes_arr,
    void** v_arr, int* v_type_arr, std::int64_t* v_ne0_arr, std::int64_t* v_ne1_arr, std::int64_t* v_bytes_arr,
    void** k_norm_arr, void** ring_k_arr, void** ring_v_arr, int ring_dtype,
    const float* scales, int scale_count);
#define TS_DF_DRAFT_TAIL                                                                                                  \
    void** attn_norm_arr,                                                                                                 \
    void** q_arr, int* q_type_arr, std::int64_t* q_ne0_arr, std::int64_t* q_ne1_arr, std::int64_t* q_bytes_arr,           \
    void** k_arr, int* k_type_arr, std::int64_t* k_ne0_arr, std::int64_t* k_ne1_arr, std::int64_t* k_bytes_arr,           \
    void** v_arr, int* v_type_arr, std::int64_t* v_ne0_arr, std::int64_t* v_ne1_arr, std::int64_t* v_bytes_arr,           \
    void** q_norm_arr, void** k_norm_arr,                                                                                 \
    void** o_arr, int* o_type_arr, std::int64_t* o_ne0_arr, std::int64_t* o_ne1_arr, std::int64_t* o_bytes_arr,           \
    void** ffn_norm_arr,                                                                                                  \
    void** gate_arr, int* gate_type_arr, std::int64_t* gate_ne0_arr, std::int64_t* gate_ne1_arr, std::int64_t* gate_bytes_arr, \
    void** up_arr, int* up_type_arr, std::int64_t* up_ne0_arr, std::int64_t* up_ne1_arr, std::int64_t* up_bytes_arr,      \
    void** down_arr, int* down_type_arr, std::int64_t* down_ne0_arr, std::int64_t* down_ne1_arr, std::int64_t* down_bytes_arr, \
    void** ring_k_arr, void** ring_v_arr, int ring_dtype,                                                                 \
    const void* out_norm_data,                                                                                            \
    const void* tok_embd_data, int tok_embd_type, std::int64_t tok_embd_ne0, std::int64_t tok_embd_ne1, std::int64_t tok_embd_bytes, \
    const void* lm_head_data, int lm_head_type, std::int64_t lm_head_ne0, std::int64_t lm_head_ne1, std::int64_t lm_head_bytes, \
    int vocab_size, int* ids_out, float* conf_out,                                                                        \
    int conv_taps, int conv_group_size, int conv_num_groups,                                                              \
    void** attn_conv_base_arr, void** attn_conv_proj_arr, int* attn_conv_proj_type_arr,                                  \
    std::int64_t* attn_conv_proj_ne0_arr, std::int64_t* attn_conv_proj_ne1_arr, std::int64_t* attn_conv_proj_bytes_arr,  \
    void** ffn_conv_base_arr, void** ffn_conv_proj_arr, int* ffn_conv_proj_type_arr,                                      \
    std::int64_t* ffn_conv_proj_ne0_arr, std::int64_t* ffn_conv_proj_ne1_arr, std::int64_t* ffn_conv_proj_bytes_arr,      \
    int sel_rank, int sel_top_k, float sel_logit_scale, float sel_logit_softcap,                                          \
    const void* sel_hidden_data, int sel_hidden_type, std::int64_t sel_hidden_ne0, std::int64_t sel_hidden_ne1, std::int64_t sel_hidden_bytes, \
    const void* sel_pred_data, int sel_pred_type, std::int64_t sel_pred_ne0, std::int64_t sel_pred_ne1, std::int64_t sel_pred_bytes, \
    const void* sel_succ_data, int sel_succ_type, std::int64_t sel_succ_ne0, std::int64_t sel_succ_ne1, std::int64_t sel_succ_bytes, \
    float* sel_scores_out, int* sel_cand_out,                                                                             \
    const float* scales, int scale_count
int TSGgml_DFlashDraftBlockScaled(
    const int* block_ids, int block_len, const int* positions,
    int num_layers, int hidden_size, int head_dim, int num_heads, int num_kv_heads, int ring_rows,
    float eps, float rope_base, float rope_freq_scale, float kq_scale,
    const int* ring_slot_pos, int sliding_window, TS_DF_DRAFT_TAIL);
int TSGgml_DFlashDraftBlockBatched(
    int n_lanes, const int* block_ids, int block_len, const int* positions,
    int num_layers, int hidden_size, int head_dim, int num_heads, int num_kv_heads, int ring_rows,
    float eps, float rope_base, float rope_freq_scale, float kq_scale,
    const int* ring_slot_pos, int sliding_window, TS_DF_DRAFT_TAIL);
}

namespace {
// Two query heads over one KV head; the ring has a block and an anchor of headroom past the window (DFlashConfig.RingRows).
// The hidden size is Flash Next's on CPU and Metal: rows past 1024 floats once caught a ggml-metal copy that stopped at
// 1024. CUDA runs at 64: its F32 matmuls are TF32 (CUBLAS_TF32_TENSOR_OP_MATH), and at 2560 wide the two key orders
// moved DFlash1's plain argmax across near-ties (values within the tolerance, picks not).
int H = 2560;
constexpr int N = 2, NH = 2, NKV = 1, HD = 32, FF = 128, FEAT = 128, VOCAB = 64;
constexpr int RING = 160, SW = 128, B = 8, RANK = 16, TOPK = VOCAB, TAPS = 2, GROUP = 16;
constexpr int PROMPT = 150, STEPS = 14;

std::uint32_t random_state;
std::vector<float> data(int n, float magnitude = .1f) {
    std::vector<float> result(n);
    for (float& v : result) {
        random_state = random_state * 1664525u + 1013904223u;
        v = float(int(random_state >> 16) - 32768) * (magnitude / 32768.f);
    }
    return result;
}
struct Mat {
    std::vector<std::vector<float>> values;
    std::vector<void*> p;
    std::vector<int> type;
    std::vector<std::int64_t> ne0, ne1, bytes;
    Mat(int in, int out, int count) {
        for (int l = 0; l < count; ++l) {
            values.push_back(data(in * out));
            p.push_back(values.back().data()); type.push_back(GGML_TYPE_F32);
            ne0.push_back(in); ne1.push_back(out); bytes.push_back(std::int64_t(in) * out * sizeof(float));
        }
    }
};
#define ARGS(m) m.p.data(), m.type.data(), m.ne0.data(), m.ne1.data(), m.bytes.data()
#define ONE(m) m.p[0], m.type[0], m.ne0[0], m.ne1[0], m.bytes[0]

struct Weights {
    Mat fc{FEAT, H, 1}, emb{H, VOCAB, 1}, head{H, VOCAB, 1}, sh{H, RANK, 1}, sp{RANK, VOCAB, 1}, ss{RANK, VOCAB, 1};
    Mat q{H, NH * HD, N}, k{H, NKV * HD, N}, v{H, NKV * HD, N}, o{NH * HD, H, N};
    Mat gate{H, FF, N}, up{H, FF, N}, down{FF, H, N};
    Mat ac{H, 2 * TAPS * (H / GROUP), N}, fconv{H, 2 * TAPS * (H / GROUP), N};
    std::vector<float> norm = std::vector<float>(H, 1.f), hnorm = std::vector<float>(HD, 1.f), conv_base;
    std::vector<void*> norms, hnorms, bases;
    Weights() {
        conv_base = data(2 * TAPS * H, .5f);
        for (int side = 0; side < 2; ++side)
            for (int c = 0; c < H; ++c) conv_base[side * TAPS * H + c] += 1.f;   // tap 0 near identity
        norms.assign(N, norm.data()); hnorms.assign(N, hnorm.data()); bases.assign(N, conv_base.data());
    }
};

// One sequence on its own ring: the injected frontier, the slot map, and the ring rows.
struct Lane {
    std::vector<std::vector<float>> rk, rv;
    std::vector<void*> ringk, ringv;
    int filled = 0;
    std::uint32_t seed;
    explicit Lane(std::uint32_t s) : rk(N, std::vector<float>(std::size_t(RING) * HD * NKV)), rv(rk), ringk(N), ringv(N), seed(s) {
        for (int l = 0; l < N; ++l) { ringk[l] = rk[l].data(); ringv[l] = rv[l].data(); }
    }
    std::vector<int> slot_map() const {
        std::vector<int> map(RING, -1);
        for (int p = std::max(0, filled - RING); p < filled; ++p) map[p % RING] = p;
        return map;
    }
};

bool check(int status, const char* what) {
    if (status != 1) std::fprintf(stderr, "FAIL %s: %s\n", what, TSGgml_GetLastError());
    return status == 1;
}

bool inject(Weights& w, Lane& lane, int start, int n) {
    // Features are a function of the position, so both arms inject the same rows.
    std::vector<float> feats;
    std::vector<std::int64_t> idx(n);
    std::vector<int> pos(n);
    for (int i = 0; i < n; ++i) {
        random_state = lane.seed * 7919u + std::uint32_t(start + i) * 104729u;
        std::vector<float> f = data(FEAT, 1.f);
        feats.insert(feats.end(), f.begin(), f.end());
        idx[i] = (start + i) % RING;
        pos[i] = start + i;
    }
    if (!check(TSGgml_DFlashInjectScaled(feats.data(), FEAT, n, idx.data(), pos.data(), N, H, HD, NKV, RING, 1e-6f, 10000.f, 1.f,
            ONE(w.fc), w.norm.data(), ARGS(w.k), ARGS(w.v), w.hnorms.data(), lane.ringk.data(), lane.ringv.data(),
            GGML_TYPE_F32, nullptr, 0), "inject"))
        return false;
    lane.filled = start + n;
    return true;
}

struct Step { std::vector<int> ids; std::vector<float> floats; std::vector<float> ring; };

std::vector<int> block_tokens(const Lane& lane, int step) {
    std::vector<int> t(B);
    for (int i = 0; i < B; ++i) t[i] = int((lane.seed * 31u + std::uint32_t(step) * 7u + std::uint32_t(i) * 13u) % VOCAB);
    return t;
}

// The lattice keyed by token ids: each position's candidates sorted by id, the anchor scores and every
// [predecessor][candidate] matrix permuted to match.
void canonical(std::vector<int>& cand, std::vector<float>& scores) {
    constexpr int G = B - 1;
    std::vector<std::vector<int>> order(G, std::vector<int>(TOPK));
    for (int e = 0; e < G; ++e) {
        for (int i = 0; i < TOPK; ++i) order[e][i] = i;
        std::sort(order[e].begin(), order[e].end(), [&](int a, int b) { return cand[e * TOPK + a] < cand[e * TOPK + b]; });
    }
    std::vector<int> c2(cand.size());
    std::vector<float> s2(scores.size());
    for (int e = 0; e < G; ++e)
        for (int i = 0; i < TOPK; ++i) c2[e * TOPK + i] = cand[e * TOPK + order[e][i]];
    for (int i = 0; i < TOPK; ++i) s2[i] = scores[order[0][i]];
    for (int e = 1; e < G; ++e)
        for (int p = 0; p < TOPK; ++p)
            for (int c = 0; c < TOPK; ++c)
                s2[TOPK + (e - 1) * TOPK * TOPK + p * TOPK + c] =
                    scores[TOPK + (e - 1) * TOPK * TOPK + order[e - 1][p] * TOPK + order[e][c]];
    cand.swap(c2);
    scores.swap(s2);
}

int draft_one(Weights& w, Lane& lane, const std::vector<int>& tokens, bool selector, std::vector<int>& ids, std::vector<float>& vals) {
    std::vector<int> positions(B), map = lane.slot_map();
    for (int i = 0; i < B; ++i) positions[i] = lane.filled + i;
    std::vector<int> id(B), cand(TOPK * (B - 1));
    std::vector<float> conf(B), scores(TOPK + TOPK * TOPK * (B - 2));
    int status = TSGgml_DFlashDraftBlockScaled(tokens.data(), B, positions.data(), N, H, HD, NH, NKV, RING,
        1e-6f, 10000.f, 1.f, 1.f / std::sqrt(float(HD)), map.data(), SW, w.norms.data(),
        ARGS(w.q), ARGS(w.k), ARGS(w.v), w.hnorms.data(), w.hnorms.data(), ARGS(w.o), w.norms.data(),
        ARGS(w.gate), ARGS(w.up), ARGS(w.down), lane.ringk.data(), lane.ringv.data(), GGML_TYPE_F32, w.norm.data(),
        ONE(w.emb), ONE(w.head), VOCAB, id.data(), conf.data(),
        selector ? TAPS : 0, GROUP, H / GROUP, w.bases.data(), ARGS(w.ac), w.bases.data(), ARGS(w.fconv),
        selector ? RANK : 0, TOPK, .8f, 3.f, ONE(w.sh), ONE(w.sp), ONE(w.ss), scores.data(), cand.data(), nullptr, 0);
    ids = selector ? cand : id;
    vals = selector ? scores : conf;
    if (selector && status == 1) canonical(ids, vals);
    return status;
}

// The ring rows a window at or after the frontier can read: positions [filled - SW, filled).
std::vector<float> live_rows(Lane& lane) {
    std::vector<float> rows;
    for (int l = 0; l < N; ++l) {
        TSGgml_SyncHostBuffer(lane.rk[l].data(), lane.rk[l].size() * sizeof(float));
        TSGgml_SyncHostBuffer(lane.rv[l].data(), lane.rv[l].size() * sizeof(float));
        for (int p = std::max(0, lane.filled - SW); p < lane.filled; ++p)
            for (int h = 0; h < NKV; ++h) {
                const float* k = lane.rk[l].data() + (std::size_t(h) * RING + p % RING) * HD;
                const float* v = lane.rv[l].data() + (std::size_t(h) * RING + p % RING) * HD;
                rows.insert(rows.end(), k, k + HD);
                rows.insert(rows.end(), v, v + HD);
            }
    }
    return rows;
}

int accepted(int step) { return (step * 3) % B + 1; }

// The whole sequence (prompt, then STEPS of draft + accept) under one TS_DFLASH_RING_BLOCK setting.
bool run(Weights& w, bool ring_block, bool selector, std::uint32_t seed, std::vector<Step>& out, bool fuse = true) {
    setenv("TS_DFLASH_RING_BLOCK", ring_block ? "1" : "0", 1);
    setenv("TS_DFLASH_METAL_FUSE", fuse ? "1" : "0", 1);
    Lane lane(seed);
    bool ok = true;
    for (int p = 0; ok && p < PROMPT; p += B) ok = inject(w, lane, p, std::min(B, PROMPT - p));
    for (int step = 0; ok && step < STEPS; ++step) {
        Step s;
        ok = check(draft_one(w, lane, block_tokens(lane, step), selector, s.ids, s.floats), "draft");
        ok = ok && inject(w, lane, lane.filled, accepted(step));
        if (ok) s.ring = live_rows(lane);
        out.push_back(std::move(s));
    }
    TSGgml_DFlashResetCaches();
    TSGgml_ClearHostBufferCache();
    return ok;
}

float max_rel(const std::vector<float>& a, const std::vector<float>& b) {
    if (a.size() != b.size()) return INFINITY;
    float r = 0;
    for (std::size_t i = 0; i < a.size(); ++i) {
        if (!std::isfinite(a[i]) || !std::isfinite(b[i])) return INFINITY;
        r = std::max(r, std::fabs(a[i] - b[i]) / std::max(1.f, std::fabs(a[i])));
    }
    return r;
}

// CUDA: two lanes drafted at once (each on its own ring, at its own frontier) against their single-lane drafts.
// The relative difference the two key orders may show. ggml-cuda runs F32 matmuls (every projection of this test's
// F32 weights, and the attention's F32 fallback at these shapes) on TF32 tensor cores (CUBLAS_TF32_TENSOR_OP_MATH,
// a 10-bit mantissa), where two key orders differ by ~4e-3 after two layers and the selector; CPU and Metal stay
// within ~4e-5.
float tolerance(int backend) { return backend == 3 ? 2e-2f : 1e-3f; }

bool batched(Weights& w, bool selector, int backend) {
    setenv("TS_DFLASH_RING_BLOCK", "1", 1);
    Lane a(11), b(29);
    bool ok = true;
    for (int p = 0; ok && p < PROMPT; p += B) ok = inject(w, a, p, std::min(B, PROMPT - p));
    for (int p = 0; ok && p < PROMPT + 9; p += B) ok = inject(w, b, p, std::min(B, PROMPT + 9 - p));
    float worst = 0;
    bool same = true;
    for (int step = 0; ok && step < 6; ++step) {
        std::vector<int> ta = block_tokens(a, step), tb = block_tokens(b, step), ids_a, ids_b;
        std::vector<float> va, vb;
        ok = check(draft_one(w, a, ta, selector, ids_a, va), "draft lane a") && check(draft_one(w, b, tb, selector, ids_b, vb), "draft lane b");
        std::vector<int> tokens = ta, positions(2 * B), maps = a.slot_map(), mb = b.slot_map();
        tokens.insert(tokens.end(), tb.begin(), tb.end());
        maps.insert(maps.end(), mb.begin(), mb.end());
        for (int i = 0; i < B; ++i) { positions[i] = a.filled + i; positions[B + i] = b.filled + i; }
        std::vector<void*> rk, rv;
        for (Lane* l : {&a, &b}) { rk.insert(rk.end(), l->ringk.begin(), l->ringk.end()); rv.insert(rv.end(), l->ringv.begin(), l->ringv.end()); }
        std::vector<int> id(2 * B), cand(2 * TOPK * (B - 1));
        std::vector<float> conf(2 * B), scores(2 * (TOPK + TOPK * TOPK * (B - 2)));
        ok = ok && check(TSGgml_DFlashDraftBlockBatched(2, tokens.data(), B, positions.data(), N, H, HD, NH, NKV, RING,
            1e-6f, 10000.f, 1.f, 1.f / std::sqrt(float(HD)), maps.data(), SW, w.norms.data(),
            ARGS(w.q), ARGS(w.k), ARGS(w.v), w.hnorms.data(), w.hnorms.data(), ARGS(w.o), w.norms.data(),
            ARGS(w.gate), ARGS(w.up), ARGS(w.down), rk.data(), rv.data(), GGML_TYPE_F32, w.norm.data(),
            ONE(w.emb), ONE(w.head), VOCAB, id.data(), conf.data(),
            selector ? TAPS : 0, GROUP, H / GROUP, w.bases.data(), ARGS(w.ac), w.bases.data(), ARGS(w.fconv),
            selector ? RANK : 0, TOPK, .8f, 3.f, ONE(w.sh), ONE(w.sp), ONE(w.ss), scores.data(), cand.data(), nullptr, 0),
            "batched draft");
        if (!ok) break;
        std::vector<int> ids = selector ? cand : id, single = ids_a;
        single.insert(single.end(), ids_b.begin(), ids_b.end());
        std::vector<float> vals = selector ? scores : conf, sv = va;
        if (selector) {
            // each lane's block of the batched outputs, keyed by token ids as the single drafts are
            const std::size_t nc = std::size_t(TOPK) * (B - 1), ns = TOPK + std::size_t(TOPK) * TOPK * (B - 2);
            std::vector<int> c0(ids.begin(), ids.begin() + nc), c1(ids.begin() + nc, ids.end());
            std::vector<float> s0(vals.begin(), vals.begin() + ns), s1(vals.begin() + ns, vals.end());
            canonical(c0, s0);
            canonical(c1, s1);
            ids = c0; ids.insert(ids.end(), c1.begin(), c1.end());
            vals = s0; vals.insert(vals.end(), s1.begin(), s1.end());
        }
        sv.insert(sv.end(), vb.begin(), vb.end());
        same = same && ids == single;
        worst = std::max(worst, max_rel(vals, sv));
        ok = inject(w, a, a.filled, accepted(step)) && inject(w, b, b.filled, accepted(step + 1));
    }
    TSGgml_DFlashResetCaches();
    TSGgml_ClearHostBufferCache();
    std::printf("DFlash%d batched (2 lanes) vs single: max relative difference %.3g, same picks %d\n", selector ? 2 : 1, worst, int(same));
    return ok && same && worst < tolerance(backend);
}
}

int main(int argc, char** argv) {
    if (argc != 2) return 2;
    const int backend = !std::strcmp(argv[1], "cuda") ? 3 : !std::strcmp(argv[1], "metal") ? 1 : 2;
    if (backend == 3) H = 64;
    if (!TSGgml_IsBackendAvailable(backend)) return 77;
    random_state = 4242;
    Weights w;
    for (bool selector : {false, true}) {
        std::vector<Step> concat, slots;
        if (!run(w, false, selector, 3, concat) || !run(w, true, selector, 3, slots)) return 1;
        float worst = 0;
        bool same_picks = true, same_rows = true;
        for (int s = 0; s < STEPS; ++s) {
            worst = std::max(worst, max_rel(concat[s].floats, slots[s].floats));
            same_picks = same_picks && concat[s].ids == slots[s].ids;
            same_rows = same_rows && concat[s].ring == slots[s].ring;   // bit for bit: the draft never touches them
        }
        std::printf("DFlash%d: ring slots vs concatenation over %d steps: max relative difference %.3g, same picks %d, "
                    "live ring rows identical %d\n", selector ? 2 : 1, STEPS, worst, int(same_picks), int(same_rows));
        if (!(worst < tolerance(backend)) || !same_picks || !same_rows) return 1;
        if (backend == 3 && !batched(w, selector, backend)) return 1;
        if (backend == 1) {
            std::vector<Step> chains;
            if (!run(w, true, selector, 3, chains, false)) return 1;
            bool exact = true;
            for (int s = 0; s < STEPS; ++s)
                exact = exact && chains[s].ids == slots[s].ids && chains[s].floats == slots[s].floats && chains[s].ring == slots[s].ring;
            float fdiff = 0;
            for (int s = 0; s < STEPS; ++s) fdiff = std::max(fdiff, max_rel(chains[s].floats, slots[s].floats));
            std::printf("DFlash%d: fused convolution vs per-tap chains: bit-identical %d (max relative difference %.3g)\n",
                        selector ? 2 : 1, int(exact), fdiff);
            if (!exact) return 1;
        }
    }
    unsetenv("TS_DFLASH_METAL_FUSE");
    unsetenv("TS_DFLASH_RING_BLOCK");
    TSGgml_Shutdown();
    std::puts("PASS DFlash ring-slot block: drafts, confidences, selector lattices and live ring rows match the concatenation");
    return 0;
}
