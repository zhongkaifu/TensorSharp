// Copyright (c) Zhongkai Fu. All rights reserved.
// Licensed under the BSD-3-Clause license in the repository root.
// Actual DFlash graphs with sidecars vs exactly folded F32 reference weights.
#include "ggml.h"
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <new>
#include <vector>

// Ring rows aligned to 16 KB (a Metal page), as the allocator's tensors are: the inject writes a ring in place only
// where the backend can wrap its host pointer, std::vector's 16-byte alignment is below what the CPU buffer type
// needs, and the inject declines such a ring (glibc's large allocations are 16 bytes past a page, so this failed on
// Linux CPU and not on macOS).
template <class T> struct PageAligned {
    using value_type = T;
    PageAligned() = default;
    template <class U> PageAligned(const PageAligned<U>&) {}
    T* allocate(std::size_t n) { return static_cast<T*>(::operator new(n * sizeof(T), std::align_val_t(16384))); }
    void deallocate(T* p, std::size_t) { ::operator delete(p, std::align_val_t(16384)); }
    template <class U> bool operator==(const PageAligned<U>&) const { return true; }
    template <class U> bool operator!=(const PageAligned<U>&) const { return false; }
};
using RingRows = std::vector<float, PageAligned<float>>;

extern "C" {
int TSGgml_IsBackendAvailable(int);
const char* TSGgml_GetLastError();
int TSGgml_SyncHostBuffer(void*, size_t);
void TSGgml_DFlashResetCaches();
void TSGgml_ClearHostBufferCache();
void TSGgml_Shutdown();
int TSGgml_DFlashInjectScaled(
    const float* feat_rows, int feature_size, int n_rows,
    const std::int64_t* ring_rows_idx,          // ring slot per row (position % ring_rows)
    const int* positions,                        // absolute position per row (for RoPE)
    int num_layers, int hidden_size, int head_dim, int num_kv_heads, int ring_rows,
    float eps, float rope_base, float rope_freq_scale,
    const void* fc_data, int fc_type, std::int64_t fc_ne0, std::int64_t fc_ne1, std::int64_t fc_bytes,
    const void* enc_norm_data,
    void** k_arr, int* k_type_arr, std::int64_t* k_ne0_arr, std::int64_t* k_ne1_arr, std::int64_t* k_bytes_arr,
    void** v_arr, int* v_type_arr, std::int64_t* v_ne0_arr, std::int64_t* v_ne1_arr, std::int64_t* v_bytes_arr,
    void** k_norm_arr,
    void** ring_k_arr, void** ring_v_arr,
    int ring_dtype,
    const float* scales, int scale_count);
int TSGgml_DFlashDraftBlockScaled(
    const int* block_ids, int block_len, const int* positions,
    int num_layers, int hidden_size, int head_dim, int num_heads, int num_kv_heads, int ring_rows,
    float eps, float rope_base, float rope_freq_scale, float kq_scale,
    // ring slot -> absolute position, or -1 when the slot is not live
    const int* ring_slot_pos, int sliding_window,
    void** attn_norm_arr,
    void** q_arr, int* q_type_arr, std::int64_t* q_ne0_arr, std::int64_t* q_ne1_arr, std::int64_t* q_bytes_arr,
    void** k_arr, int* k_type_arr, std::int64_t* k_ne0_arr, std::int64_t* k_ne1_arr, std::int64_t* k_bytes_arr,
    void** v_arr, int* v_type_arr, std::int64_t* v_ne0_arr, std::int64_t* v_ne1_arr, std::int64_t* v_bytes_arr,
    void** q_norm_arr, void** k_norm_arr,
    void** o_arr, int* o_type_arr, std::int64_t* o_ne0_arr, std::int64_t* o_ne1_arr, std::int64_t* o_bytes_arr,
    void** ffn_norm_arr,
    void** gate_arr, int* gate_type_arr, std::int64_t* gate_ne0_arr, std::int64_t* gate_ne1_arr, std::int64_t* gate_bytes_arr,
    void** up_arr, int* up_type_arr, std::int64_t* up_ne0_arr, std::int64_t* up_ne1_arr, std::int64_t* up_bytes_arr,
    void** down_arr, int* down_type_arr, std::int64_t* down_ne0_arr, std::int64_t* down_ne1_arr, std::int64_t* down_bytes_arr,
    void** ring_k_arr, void** ring_v_arr, int ring_dtype,
    const void* out_norm_data,
    const void* tok_embd_data, int tok_embd_type, std::int64_t tok_embd_ne0, std::int64_t tok_embd_ne1, std::int64_t tok_embd_bytes,
    const void* lm_head_data, int lm_head_type, std::int64_t lm_head_ne0, std::int64_t lm_head_ne1, std::int64_t lm_head_bytes,
    int vocab_size, int* ids_out, float* conf_out,
    // ---- DFlash2 grouped dynamic convolution (conv_taps == 0 disables) ----
    int conv_taps, int conv_group_size, int conv_num_groups,
    void** attn_conv_base_arr,
    void** attn_conv_proj_arr, int* attn_conv_proj_type_arr,
    std::int64_t* attn_conv_proj_ne0_arr, std::int64_t* attn_conv_proj_ne1_arr, std::int64_t* attn_conv_proj_bytes_arr,
    void** ffn_conv_base_arr,
    void** ffn_conv_proj_arr, int* ffn_conv_proj_type_arr,
    std::int64_t* ffn_conv_proj_ne0_arr, std::int64_t* ffn_conv_proj_ne1_arr, std::int64_t* ffn_conv_proj_bytes_arr,
    // ---- DFlash2 candidate selector (sel_rank == 0 disables) ----
    int sel_rank, int sel_top_k, float sel_logit_scale, float sel_logit_softcap,
    const void* sel_hidden_data, int sel_hidden_type, std::int64_t sel_hidden_ne0, std::int64_t sel_hidden_ne1, std::int64_t sel_hidden_bytes,
    const void* sel_pred_data, int sel_pred_type, std::int64_t sel_pred_ne0, std::int64_t sel_pred_ne1, std::int64_t sel_pred_bytes,
    const void* sel_succ_data, int sel_succ_type, std::int64_t sel_succ_ne0, std::int64_t sel_succ_ne1, std::int64_t sel_succ_bytes,
    float* sel_scores_out, int* sel_cand_out,
    const float* scales, int scale_count);
}

namespace {
constexpr int N = 2, H = 64, HD = 64, FF = 128, FEAT = 128, VOCAB = 64;
constexpr int RING = 256, B = 8, RANK = 16, TOPK = 4, TAPS = 2, GROUP = 16;
constexpr int SCALE_COUNT = 6 + N * 9;

uint32_t random_state;
std::vector<float> data(int n, float magnitude = .1f) {
    std::vector<float> result(n);
    for (float& v : result) {
        random_state = random_state * 1664525u + 1013904223u;
        v = float(int(random_state >> 16) - 32768) * (magnitude / 32768.f);
    }
    return result;
}
float sidecar(int index) { return index % 3 == 0 ? .5f : index % 3 == 1 ? 2.f : .25f; }
struct Mat {
    std::vector<std::vector<float>> values;
    std::vector<void*> p;
    std::vector<int> type;
    std::vector<int64_t> ne0, ne1, bytes;
    Mat(int in, int out, int count, int slot, bool folded) {
        for (int l = 0; l < count; ++l) {
            values.push_back(data(in * out));
            if (folded) for (float& v : values.back()) v *= sidecar(count == N ? 6 + 9 * l + slot : slot);
            p.push_back(values.back().data()); type.push_back(GGML_TYPE_F32);
            ne0.push_back(in); ne1.push_back(out); bytes.push_back(int64_t(in) * out * sizeof(float));
        }
    }
};
#define ARGS(m) m.p.data(), m.type.data(), m.ne0.data(), m.ne1.data(), m.bytes.data()
#define ONE(m) m.p[0], m.type[0], m.ne0[0], m.ne1[0], m.bytes[0]

struct Result { std::vector<float> floats; std::vector<int> ids; };

bool run(int mode, bool selector, Result& result) {
    random_state = 6789;
    bool folded = mode == 2;
    Mat fc(FEAT,H,1,0,folded), emb(H,VOCAB,1,1,folded), head(H,VOCAB,1,2,folded);
    Mat sh(H,RANK,1,3,folded), sp(RANK,VOCAB,1,4,folded), ss(RANK,VOCAB,1,5,folded);
    Mat q(H,H,N,0,folded), k(H,H,N,1,folded), v(H,H,N,2,folded), o(H,H,N,3,folded);
    Mat gate(H,FF,N,4,folded), up(H,FF,N,5,folded), down(FF,H,N,6,folded);
    Mat ac(H,2*TAPS*(H/GROUP),N,7,folded), fconv(H,2*TAPS*(H/GROUP),N,8,folded);
    std::vector<float> norm(H,1.f), conv_base(2*TAPS*H);
    for (int side = 0; side < 2; ++side)
        std::fill(conv_base.begin() + side*TAPS*H, conv_base.begin() + side*TAPS*H + H, 1.f);
    std::vector<void*> norms(N,norm.data()), bases(N,conv_base.data());
    std::vector<RingRows> rk(N,RingRows(RING*HD)), rv = rk;
    std::vector<void*> ringk(N), ringv(N);
    for (int l = 0; l < N; ++l) { ringk[l] = rk[l].data(); ringv[l] = rv[l].data(); }
    std::vector<float> features = data(FEAT*B,1.f), scales(SCALE_COUNT);
    for (int s = 0; s < SCALE_COUNT; ++s) scales[s] = sidecar(s);
    std::vector<int64_t> indices(B);
    std::vector<int> positions(B), tokens(B), ringpos(RING,-1), ids(B), candidates(TOPK*(B-1));
    std::vector<float> conf(B), scores(TOPK+TOPK*TOPK*(B-2));
    for (int i = 0; i < B; ++i) { indices[i] = i; positions[i] = i; ringpos[i] = i; }
    auto inject = [&](const float* sc) {
        return TSGgml_DFlashInjectScaled(features.data(),FEAT,B,indices.data(),positions.data(),
            N,H,HD,1,RING,1e-6f,10000.f,1.f,ONE(fc),norm.data(),ARGS(k),ARGS(v),norms.data(),
            ringk.data(),ringv.data(),GGML_TYPE_F32,sc,sc ? SCALE_COUNT : 0);
    };
    auto draft = [&](const float* sc) {
        return TSGgml_DFlashDraftBlockScaled(tokens.data(),B,positions.data(),N,H,HD,1,1,RING,
            1e-6f,10000.f,1.f,1.f/std::sqrt(float(HD)),ringpos.data(),RING,norms.data(),
            ARGS(q),ARGS(k),ARGS(v),norms.data(),norms.data(),ARGS(o),norms.data(),
            ARGS(gate),ARGS(up),ARGS(down),ringk.data(),ringv.data(),GGML_TYPE_F32,norm.data(),
            ONE(emb),ONE(head),VOCAB,ids.data(),conf.data(),
            selector ? TAPS : 0,GROUP,H/GROUP,bases.data(),ARGS(ac),bases.data(),ARGS(fconv),
            selector ? RANK : 0,TOPK,.8f,3.f,ONE(sh),ONE(sp),ONE(ss),scores.data(),candidates.data(),
            sc,sc ? SCALE_COUNT : 0);
    };
    auto check = [](int status) {
        if (status != 1) std::fprintf(stderr,"FAIL DFlash: %s\n",TSGgml_GetLastError());
        return status == 1;
    };
    // Seed an unscaled graph with the SAME weight/ring pointers before the
    // scaled call. Its cached computation must not satisfy the scaled request.
    bool ok = mode != 1 || check(inject(nullptr));
    const float* sc = mode == 1 ? scales.data() : nullptr;
    ok = ok && check(inject(sc));
    for (int l = 0; ok && l < N; ++l) {
        ok = check(TSGgml_SyncHostBuffer(rk[l].data(),rk[l].size()*sizeof(float)))
            && check(TSGgml_SyncHostBuffer(rv[l].data(),rv[l].size()*sizeof(float)));
        result.floats.insert(result.floats.end(),rk[l].begin(),rk[l].end());
        result.floats.insert(result.floats.end(),rv[l].begin(),rv[l].end());
    }
    for (int step = 0; ok && step < 3; ++step) {
        for (int i = 0; i < B; ++i) { positions[i] = B + i + step; tokens[i] = (i*7+step*3)%VOCAB; }
        if (step == 0 && mode == 1) ok = check(draft(nullptr));
        ok = ok && check(draft(sc));
        const auto& values = selector ? scores : conf;
        const auto& selected = selector ? candidates : ids;
        result.floats.insert(result.floats.end(),values.begin(),values.end());
        result.ids.insert(result.ids.end(),selected.begin(),selected.end());
    }
    TSGgml_DFlashResetCaches();
    TSGgml_ClearHostBufferCache();
    return ok;
}

float max_diff(const std::vector<float>& a, const std::vector<float>& b) {
    if (a.size() != b.size()) return INFINITY;
    float result = 0;
    for (size_t i = 0; i < a.size(); ++i) {
        if (!std::isfinite(a[i]) || !std::isfinite(b[i])) return INFINITY;
        result = std::max(result,std::fabs(a[i]-b[i]));
    }
    return result;
}
}

int main(int argc, char** argv) {
    if (argc != 2) return 2;
    int backend = !std::strcmp(argv[1],"cuda") ? 3 : !std::strcmp(argv[1],"metal") ? 1 : 2;
    if (!TSGgml_IsBackendAvailable(backend)) return 77;
    for (bool selector : {false,true}) {
        Result unscaled, scaled, folded;
        if (!run(0,selector,unscaled) || !run(1,selector,scaled) || !run(2,selector,folded)) return 1;
        float error = max_diff(scaled.floats,folded.floats);
        float omitted = max_diff(unscaled.floats,folded.floats);
        std::printf("DFlash%d: sidecar error %.8g, omitted scale error %.8g, ids match %d\n",
            selector ? 2 : 1,error,omitted,int(scaled.ids==folded.ids));
        if (!(error < .002f && omitted > .05f) || scaled.ids != folded.ids) return 1;
    }
    TSGgml_Shutdown();
    std::puts("PASS DFlash1/2: injection, borrowed embedding/head, all layer projections, "
        "dynamic convolution, selector, ring bytes, confidence, replay and scale cache identity");
    return 0;
}
