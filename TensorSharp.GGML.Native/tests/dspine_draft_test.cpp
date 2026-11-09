// Copyright (c) Zhongkai Fu. All rights reserved.
// Licensed under the BSD-3-Clause license in the repository root.
//
// The DSpine draft graph (ggml_ops_dspine.cpp) against eng/dspine-reference.py, the NumPy reference of the same
// math, on small synthetic weights:
//
//   * the refinement tables TG / TSV (TSGgml_DSpineTablesCreate) against the reference's;
//   * every stage of one draft (each layer's pre-write rows, bridge codes and post-write rows, the final rows and the
//     base logits) within float tolerance, and per-row checksums of the stages for every draft;
//   * the same top-K candidate sets, the same lattice scores and the same greedy chain (walked like the host does),
//     except where the reference itself says the answer is within float noise (a candidate set whose K-th and
//     (K+1)-th logits are closer than the tolerance; a chain step whose best two scores are) - those are counted;
//   * with F32 and BF16 tables, and with the bridge weights bound as BF16 (production) and as F32 (same values);
//   * a narrower block is the first rows of a wider one (attention is causal inside the block);
//   * a replayed (cached) graph returns exactly what the first run returned.
//
// The weights are not stored: both sides draw them from the same 32-bit LCG in the same order (see make_weights in
// the reference); tests/data/dspine_draft_reference.bin holds only expected values. Configs "small" (all stages)
// and "wide" (hidden 1280 and 1056-wide codes, past the 1024-float rows a ggml-metal copy once stopped at).
//
//     GgmlOpsDSpineDraftTest <cpu|metal|cuda> <fixture>
#include "ggml.h"
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <string>
#include <vector>

extern "C" {
struct TsgDsWeight { const void* data; std::int64_t ne0, ne1, bytes; std::int32_t type, reserved; };
struct TsgDsParams {
    std::int32_t num_layers, hidden, num_heads, num_kv_heads, head_dim, vocab;
    std::int32_t code_size, rank, top_k, ring_rows, ring_dtype, sliding_window;
    float eps, bridge_eps, rope_base, rope_freq_scale, kq_scale;
    std::int32_t reserved[3];
};
struct TsgDsProbe { float* raw; float* codes; float* out; float* final_rows; float* base; };
int TSGgml_IsBackendAvailable(int);
const char* TSGgml_GetLastError();
void TSGgml_ClearHostBufferCache();
void TSGgml_Shutdown();
void TSGgml_DFlashResetCaches();
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
void* TSGgml_DSpineTablesCreate(const TsgDsParams*, const TsgDsWeight*, int, int);
void TSGgml_DSpineTablesFree(void*);
int TSGgml_DSpineTablesRead(void*, int, int, int, float*);
void TSGgml_DSpineResetCaches();
int TSGgml_DSpineDraftBlock(const TsgDsParams*, const TsgDsWeight*, int, void*, void**, void**,
    const int*, int, const int*, const int*, float*, int*, const TsgDsProbe*);
}

namespace {

// ---- the reference's LCG and weight order ----
struct Lcg {
    std::uint32_t state;
    explicit Lcg(std::uint32_t s) : state(s) {}
    std::vector<float> take(std::size_t n, float mag) {
        std::vector<float> v(n);
        const float scale = mag / 32768.f;
        for (auto& x : v) {
            state = state * 1664525u + 1013904223u;
            x = float(int(state >> 16) - 32768) * scale;
        }
        return v;
    }
};
float mag_in(int fan_in) { return 1.75f / std::sqrt(float(fan_in)); }
std::uint16_t to_bf16(float f) {
    std::uint32_t u;
    std::memcpy(&u, &f, 4);
    u += 0x7FFFu + ((u >> 16) & 1u);
    return std::uint16_t(u >> 16);
}
float from_bf16(std::uint16_t h) {
    std::uint32_t u = std::uint32_t(h) << 16;
    float f;
    std::memcpy(&f, &u, 4);
    return f;
}
std::vector<float> plus_one(std::vector<float> v) { for (auto& x : v) x = 1.f + x; return v; }
std::vector<float> bf16_values(std::vector<float> v) { for (auto& x : v) x = from_bf16(to_bf16(x)); return v; }

struct Config { int H, NH, NKV, HD, FF, L, NT, C, R, V, K, B, SW, SEED, RING; };

// One stored matrix [rows][cols] (= ggml [cols, rows]); as F32 and, for the bridge, as BF16 bits.
struct Mat {
    std::vector<float> f;
    std::vector<std::uint16_t> h;
    std::int64_t cols = 0, rows = 0;
    Mat() = default;
    Mat(std::vector<float> v, std::int64_t c, std::int64_t r, bool bf16) : f(std::move(v)), cols(c), rows(r) {
        if (bf16) { h.resize(f.size()); for (std::size_t i = 0; i < f.size(); ++i) h[i] = to_bf16(f[i]); }
    }
    TsgDsWeight weight(bool as_bf16) const {
        if (as_bf16 && !h.empty()) return { h.data(), cols, rows, std::int64_t(h.size() * 2), GGML_TYPE_BF16, 0 };
        return { f.data(), cols, rows, std::int64_t(f.size() * 4), GGML_TYPE_F32, 0 };
    }
};

struct Model {
    Config c;
    Mat tok, head, fc, enc_norm, out_norm, codebook, bridge, gate_bias;
    struct Layer { Mat attn_norm, q, k, v, q_norm, k_norm, o, ffn_norm, gate, up, down, read, write; };
    std::vector<Layer> layers;

    explicit Model(const Config& cfg) : c(cfg) {
        const int H = c.H, C = c.C, R = c.R, V = c.V, FEAT = c.NT * c.H;
        Lcg g(std::uint32_t(c.SEED));
        tok = Mat(g.take(std::size_t(V) * H, 1.f), H, V, false);
        head = Mat(g.take(std::size_t(V) * H, 3.f * mag_in(H)), H, V, false);
        fc = Mat(g.take(std::size_t(H) * FEAT, mag_in(FEAT)), FEAT, H, false);
        enc_norm = Mat(plus_one(g.take(H, .25f)), H, 1, false);
        out_norm = Mat(plus_one(g.take(H, .25f)), H, 1, false);
        codebook = Mat(bf16_values(g.take(std::size_t(V) * C, mag_in(C))), C, V, true);
        bridge = Mat(bf16_values(g.take(std::size_t(3) * R * C, mag_in(C))), C, 3 * R, true);
        gate_bias = Mat(g.take(R, .5f), R, 1, false);
        const int qd = c.NH * c.HD, kvd = c.NKV * c.HD;
        for (int l = 0; l < c.L; ++l) {
            Layer y;
            y.attn_norm = Mat(plus_one(g.take(H, .25f)), H, 1, false);
            y.q = Mat(g.take(std::size_t(qd) * H, mag_in(H)), H, qd, false);
            y.k = Mat(g.take(std::size_t(kvd) * H, mag_in(H)), H, kvd, false);
            y.v = Mat(g.take(std::size_t(kvd) * H, mag_in(H)), H, kvd, false);
            y.q_norm = Mat(plus_one(g.take(c.HD, .25f)), c.HD, 1, false);
            y.k_norm = Mat(plus_one(g.take(c.HD, .25f)), c.HD, 1, false);
            y.o = Mat(g.take(std::size_t(H) * qd, mag_in(qd)), qd, H, false);
            y.ffn_norm = Mat(plus_one(g.take(H, .25f)), H, 1, false);
            y.gate = Mat(g.take(std::size_t(c.FF) * H, mag_in(H)), H, c.FF, false);
            y.up = Mat(g.take(std::size_t(c.FF) * H, mag_in(H)), H, c.FF, false);
            y.down = Mat(g.take(std::size_t(H) * c.FF, mag_in(c.FF)), c.FF, H, false);
            y.read = Mat(bf16_values(g.take(std::size_t(C) * H, mag_in(H))), H, C, true);
            y.write = Mat(bf16_values(g.take(std::size_t(H) * R, .5f * mag_in(R))), R, H, true);
            layers.push_back(std::move(y));
        }
    }

    std::vector<TsgDsWeight> table(bool bridge_bf16) const {
        std::vector<TsgDsWeight> w(8 + 16 * c.L, TsgDsWeight{});
        w[0] = tok.weight(false); w[1] = head.weight(false); w[2] = out_norm.weight(false);
        w[3] = codebook.weight(bridge_bf16); w[4] = bridge.weight(bridge_bf16); w[5] = gate_bias.weight(false);
        for (int l = 0; l < c.L; ++l) {
            const Layer& y = layers[l];
            TsgDsWeight* s = &w[8 + 16 * l];
            s[0] = y.attn_norm.weight(false); s[1] = y.q.weight(false); s[2] = y.k.weight(false); s[3] = y.v.weight(false);
            s[4] = y.q_norm.weight(false); s[5] = y.k_norm.weight(false); s[6] = y.o.weight(false);
            s[7] = y.ffn_norm.weight(false); s[8] = y.gate.weight(false); s[9] = y.up.weight(false); s[10] = y.down.weight(false);
            s[11] = y.read.weight(bridge_bf16); s[12] = y.write.weight(bridge_bf16);
        }
        return w;
    }

    TsgDsParams params() const {
        TsgDsParams p{};
        p.num_layers = c.L; p.hidden = c.H; p.num_heads = c.NH; p.num_kv_heads = c.NKV; p.head_dim = c.HD; p.vocab = c.V;
        p.code_size = c.C; p.rank = c.R; p.top_k = c.K; p.ring_rows = c.RING; p.ring_dtype = GGML_TYPE_F32;
        p.sliding_window = c.SW; p.eps = 1e-6f; p.bridge_eps = 1e-6f; p.rope_base = 10000.f; p.rope_freq_scale = 1.f;
        p.kq_scale = 1.f / std::sqrt(float(c.HD));
        return p;
    }
};

// One sequence's ring, filled by the DFlash inject (the same pass A+B DSpine uses).
struct Ring {
    std::vector<std::vector<float>> k, v;
    std::vector<void*> kp, vp;
    int filled = 0;
    explicit Ring(const Config& c) : k(c.L, std::vector<float>(std::size_t(c.RING) * c.NKV * c.HD)), v(k), kp(c.L), vp(c.L) {
        for (int l = 0; l < c.L; ++l) { kp[l] = k[l].data(); vp[l] = v[l].data(); }
    }
    std::vector<int> slot_map(int rows) const {
        std::vector<int> map(rows, -1);
        for (int p = std::max(0, filled - rows); p < filled; ++p) map[p % rows] = p;
        return map;
    }
};

bool check(int status, const char* what) {
    if (status != 1) std::fprintf(stderr, "FAIL %s: %s\n", what, TSGgml_GetLastError());
    return status == 1;
}

bool inject(const Model& m, Ring& ring, int start, int n) {
    const Config& c = m.c;
    const int FEAT = c.NT * c.H;
    std::vector<float> feats;
    std::vector<std::int64_t> idx(n);
    std::vector<int> pos(n);
    for (int i = 0; i < n; ++i) {
        Lcg g(std::uint32_t(977 * (start + i) + 12345));
        std::vector<float> f = g.take(FEAT, 1.f);
        feats.insert(feats.end(), f.begin(), f.end());
        idx[i] = (start + i) % c.RING;
        pos[i] = start + i;
    }
    std::vector<void*> kw(c.L), vw(c.L), kn(c.L);
    std::vector<int> kt(c.L, GGML_TYPE_F32), vt(c.L, GGML_TYPE_F32);
    std::vector<std::int64_t> k0(c.L, c.H), k1(c.L, c.NKV * c.HD), kb(c.L, std::int64_t(c.H) * c.NKV * c.HD * 4);
    for (int l = 0; l < c.L; ++l) {
        kw[l] = const_cast<float*>(m.layers[l].k.f.data());
        vw[l] = const_cast<float*>(m.layers[l].v.f.data());
        kn[l] = const_cast<float*>(m.layers[l].k_norm.f.data());
    }
    if (!check(TSGgml_DFlashInjectScaled(feats.data(), FEAT, n, idx.data(), pos.data(), c.L, c.H, c.HD, c.NKV, c.RING,
            1e-6f, 10000.f, 1.f, m.fc.f.data(), GGML_TYPE_F32, FEAT, c.H, std::int64_t(m.fc.f.size()) * 4,
            m.enc_norm.f.data(), kw.data(), kt.data(), k0.data(), k1.data(), kb.data(),
            vw.data(), vt.data(), k0.data(), k1.data(), kb.data(), kn.data(), ring.kp.data(), ring.vp.data(),
            GGML_TYPE_F32, nullptr, 0), "inject"))
        return false;
    ring.filled = start + n;
    return true;
}

// ---- the fixture ----
struct Arr { std::vector<std::uint32_t> dims; std::vector<float> f; std::vector<int> i; };
std::map<std::string, Arr> g_fx;

bool load_fixture(const char* path) {
    FILE* fp = std::fopen(path, "rb");
    if (fp == nullptr) { std::fprintf(stderr, "FAIL cannot open %s\n", path); return false; }
    char magic[4];
    std::uint32_t version = 0, count = 0;
    bool ok = std::fread(magic, 1, 4, fp) == 4 && std::fread(&version, 4, 1, fp) == 1 && std::fread(&count, 4, 1, fp) == 1
        && std::memcmp(magic, "DSPF", 4) == 0 && version == 1;
    for (std::uint32_t e = 0; ok && e < count; ++e) {
        std::uint16_t len = 0;
        std::uint8_t dt = 0, nd = 0;
        ok = std::fread(&len, 2, 1, fp) == 1;
        std::string name(len, '\0');
        ok = ok && std::fread(&name[0], 1, len, fp) == len && std::fread(&dt, 1, 1, fp) == 1 && std::fread(&nd, 1, 1, fp) == 1;
        Arr a;
        a.dims.resize(nd);
        ok = ok && (nd == 0 || std::fread(a.dims.data(), 4, nd, fp) == nd);
        std::size_t n = 1;
        for (auto d : a.dims) n *= d;
        if (dt == 0) { a.f.resize(n); ok = ok && std::fread(a.f.data(), 4, n, fp) == n; }
        else { a.i.resize(n); ok = ok && std::fread(a.i.data(), 4, n, fp) == n; }
        g_fx[name] = std::move(a);
    }
    std::fclose(fp);
    if (!ok) std::fprintf(stderr, "FAIL malformed fixture %s\n", path);
    return ok;
}

const Arr& fx(const std::string& name) {
    auto it = g_fx.find(name);
    if (it == g_fx.end()) { std::fprintf(stderr, "FAIL fixture has no %s\n", name.c_str()); std::exit(1); }
    return it->second;
}

// max |a - ref| / max |ref|
double nerr(const float* a, const float* ref, std::size_t n) {
    double num = 0, den = 1e-30;
    for (std::size_t i = 0; i < n; ++i) {
        if (!std::isfinite(a[i])) return INFINITY;
        num = std::max(num, double(std::fabs(a[i] - ref[i])));
        den = std::max(den, double(std::fabs(ref[i])));
    }
    return num / den;
}

struct Tol { double stage, lattice, sum, gap, margin; };

struct Report {
    bool ok = true;
    double worst_stage = 0, worst_lattice = 0, worst_sum = 0, worst_conf = 0;
    int cand_rows = 0, cand_ambiguous = 0, chain_checked = 0, chain_stopped = 0;
    void fail(const char* what) { std::fprintf(stderr, "FAIL %s\n", what); ok = false; }
};

// Native lattice -> candidates sorted by id per row, scores permuted to match (the reference's layout).
void canonical(int K, int g, std::vector<int>& cand, std::vector<float>& scores) {
    std::vector<std::vector<int>> order(g, std::vector<int>(K));
    for (int e = 0; e < g; ++e) {
        for (int i = 0; i < K; ++i) order[e][i] = i;
        std::sort(order[e].begin(), order[e].end(), [&](int a, int b) { return cand[e * K + a] < cand[e * K + b]; });
    }
    std::vector<int> c2(cand.size());
    std::vector<float> s2(scores.size());
    for (int e = 0; e < g; ++e)
        for (int i = 0; i < K; ++i) c2[e * K + i] = cand[e * K + order[e][i]];
    for (int i = 0; i < K; ++i) s2[i] = scores[order[0][i]];
    for (int e = 1; e < g; ++e)
        for (int p = 0; p < K; ++p)
            for (int c = 0; c < K; ++c)
                s2[K + std::size_t(e - 1) * K * K + p * K + c] = scores[K + std::size_t(e - 1) * K * K + order[e - 1][p] * K + order[e][c]];
    cand.swap(c2);
    scores.swap(s2);
}

struct Draft { std::vector<int> cand; std::vector<float> scores, base; };

bool draft(const Model& m, const std::vector<TsgDsWeight>& w, void* tables, Ring& ring, int anchor, int b,
           Draft& d, TsgDsProbe* probe = nullptr) {
    const Config& c = m.c;
    const int g = b - 1;
    std::vector<int> ids(b, c.V - 1), pos(b);
    ids[0] = anchor;
    for (int i = 0; i < b; ++i) pos[i] = ring.filled + i;
    std::vector<int> map = ring.slot_map(c.RING);
    d.cand.assign(std::size_t(c.K) * g, -1);
    d.scores.assign(c.K + std::size_t(c.K) * c.K * (g - 1), NAN);
    TsgDsParams p = m.params();
    return check(TSGgml_DSpineDraftBlock(&p, w.data(), int(w.size()), tables, ring.kp.data(), ring.vp.data(),
        ids.data(), b, pos.data(), map.data(), d.scores.data(), d.cand.data(), probe), "draft");
}

// The reference's chain over a canonical lattice: first max (the lowest id) per step.
std::vector<int> walk(int K, int g, const std::vector<int>& cand, const std::vector<float>& scores, std::vector<double>* conf) {
    std::vector<int> out(g);
    int j = 0;
    for (int e = 0; e < g; ++e) {
        const float* row = e == 0 ? scores.data() : scores.data() + K + std::size_t(e - 1) * K * K + std::size_t(j) * K;
        int best = 0;
        for (int i = 1; i < K; ++i) if (row[i] > row[best]) best = i;
        if (conf != nullptr) {
            double sum = 0;
            for (int i = 0; i < K; ++i) sum += std::exp(double(row[i]) - row[best]);
            conf->push_back(1.0 / sum);
        }
        j = best;
        out[e] = cand[e * K + j];
    }
    return out;
}

// Candidates, lattice and chain of one draft against the fixture (canonical lattice in d).
void compare_tail(const Config& c, int g, const std::string& pre, const std::string& step, Draft& d, const Tol& t,
                  bool have_scores, Report& r) {
    const int K = c.K;
    const Arr& rc = fx(step + ".cand");
    const Arr& gap = fx(step + ".kgap");
    std::vector<bool> same(g);
    for (int e = 0; e < g; ++e) {
        same[e] = std::equal(d.cand.begin() + e * K, d.cand.begin() + (e + 1) * K, rc.i.begin() + e * K);
        ++r.cand_rows;
        if (!same[e]) {
            if (gap.f[e] < t.gap) ++r.cand_ambiguous;
            else { std::fprintf(stderr, "  %s row %d: candidate set differs (k-gap %.3g)\n", step.c_str(), e, gap.f[e]); r.fail("candidates"); }
        }
    }
    if (have_scores) {
        const Arr& rs = fx(pre + ".scores");     // [g][K(p)][K(c)]
        double den = 1e-30, num = 0;
        for (float v : rs.f) den = std::max(den, double(std::fabs(v)));
        for (int e = 0; e < g; ++e) {
            if (!same[e] || (e > 0 && !same[e - 1])) continue;
            for (int p = 0; p < (e == 0 ? 1 : K); ++p)
                for (int cc = 0; cc < K; ++cc) {
                    const float a = e == 0 ? d.scores[cc] : d.scores[K + std::size_t(e - 1) * K * K + p * K + cc];
                    const float b = rs.f[(std::size_t(e) * K + p) * K + cc];
                    num = std::max(num, double(std::fabs(a - b)));
                }
        }
        r.worst_lattice = std::max(r.worst_lattice, num / den);
        if (!(num / den < t.lattice)) r.fail((pre + " lattice scores").c_str());
    }
    std::vector<double> conf;
    std::vector<int> chain = walk(K, g, d.cand, d.scores, &conf);
    const Arr& ref = fx(pre + ".chain");
    const Arr& margin = fx(pre + ".margin");
    const Arr& rconf = fx(pre + ".conf");
    for (int e = 0; e < g; ++e) {
        if (!same[e] || margin.f[e] < t.margin) { ++r.chain_stopped; break; }   // past here the walk may fork legally
        ++r.chain_checked;
        if (chain[e] != ref.i[e]) {
            std::fprintf(stderr, "  %s chain step %d: %d vs reference %d (margin %.3g)\n", pre.c_str(), e, chain[e], ref.i[e], margin.f[e]);
            r.fail("chain");
            break;
        }
        r.worst_conf = std::max(r.worst_conf, std::fabs(conf[e] - rconf.f[e]));
    }
}

bool run_config(const std::string& name, int backend, bool bridge_bf16, const Tol& t, Report& r) {
    const Arr& cf = fx(name + ".config");
    Config c{cf.i[0], cf.i[1], cf.i[2], cf.i[3], cf.i[4], cf.i[5], cf.i[6], cf.i[7], cf.i[8], cf.i[9], cf.i[10], cf.i[11],
             cf.i[12], cf.i[13], cf.i[14]};
    Model m(c);
    const std::vector<TsgDsWeight> w = m.table(bridge_bf16);
    const TsgDsParams p = m.params();
    void* tab[2] = { TSGgml_DSpineTablesCreate(&p, w.data(), int(w.size()), GGML_TYPE_F32),
                     TSGgml_DSpineTablesCreate(&p, w.data(), int(w.size()), GGML_TYPE_BF16) };
    if (tab[0] == nullptr || tab[1] == nullptr) { std::fprintf(stderr, "FAIL tables: %s\n", TSGgml_GetLastError()); return false; }

    // the tables
    if (name == "small") {
        const Arr& rows = fx(name + ".table_rows");
        for (int which = 0; which < 2; ++which) {
            const Arr& ref = fx(name + (which == 0 ? ".tg" : ".tsv"));
            const int cols = which == 0 ? c.R : 2 * c.R;
            for (int ti = 0; ti < 2; ++ti) {
                std::vector<float> got(rows.i.size() * cols), one(cols);
                for (std::size_t i = 0; i < rows.i.size(); ++i) {
                    if (!check(TSGgml_DSpineTablesRead(tab[ti], which, rows.i[i], 1, one.data()), "tables read")) return false;
                    std::copy(one.begin(), one.end(), got.begin() + i * cols);
                }
                const double e = nerr(got.data(), ref.f.data(), got.size());
                const double tol = ti == 0 ? t.stage : 1e-2;      // a BF16 table is the F32 one rounded
                std::printf("  %s %s %s table: normalized max error %.3g\n", name.c_str(), which == 0 ? "TG" : "TSV",
                            ti == 0 ? "F32" : "BF16", e);
                if (!(e < tol)) r.fail("tables");
            }
        }
    }

    Ring ring(c);
    const int steps[3][4] = { {0, 40, 7, 16}, {40, 5, 11, 16}, {45, 75, 3, 16} };
    Draft last16;
    for (int si = 0; si < 3; ++si) {
        if (!inject(m, ring, steps[si][0], steps[si][1])) return false;
        const int anchor = steps[si][2], b = steps[si][3], g = b - 1;
        const std::string step = name + ".step" + std::to_string(si);
        for (int ti = 0; ti < 2; ++ti) {
            const std::string pre = step + (ti == 0 ? ".f32" : ".bf16");
            Draft d;
            // F32 tables: every stage through a probe (fresh graph) ...
            std::vector<float> raw(std::size_t(c.L) * b * c.H), codes(std::size_t(c.L) * g * c.C), out(raw.size()),
                fin(std::size_t(b) * c.H), base(std::size_t(g) * c.K);
            TsgDsProbe probe{ raw.data(), codes.data(), out.data(), fin.data(), base.data() };
            if (!draft(m, w, tab[ti], ring, anchor, b, d, ti == 0 ? &probe : nullptr)) return false;
            if (ti == 0) {
                const bool full = name == "small" && si == 0;
                // per-row checksums of every stage, all steps
                auto sums = [&](const std::string& key, const float* a, int rows, int cols, int row0) {
                    const Arr& rs = fx(step + "." + key + ".rowsum");
                    const Arr& rq = fx(step + "." + key + ".rowsq");
                    for (int i = 0; i < rows; ++i) {
                        double s = 0, q = 0;
                        for (int j = 0; j < cols; ++j) { s += a[std::size_t(i) * cols + j]; q += double(a[std::size_t(i) * cols + j]) * a[std::size_t(i) * cols + j]; }
                        const double es = std::fabs(s - rs.f[row0 + i]) / std::sqrt(double(rq.f[row0 + i]) * cols + 1e-30);
                        const double eq = std::fabs(q - rq.f[row0 + i]) / (rq.f[row0 + i] + 1e-30);
                        r.worst_sum = std::max(r.worst_sum, std::max(es, eq));
                        if (!(es < t.sum && eq < t.sum)) { std::fprintf(stderr, "  %s %s row %d: sum %.3g sq %.3g\n", step.c_str(), key.c_str(), row0 + i, es, eq); r.fail("stage checksums"); return; }
                    }
                };
                for (int l = 0; l < c.L; ++l) {
                    const float* a_raw = raw.data() + std::size_t(l) * b * c.H;
                    const float* a_codes = codes.data() + std::size_t(l) * g * c.C;
                    const float* a_out = out.data() + std::size_t(l) * b * c.H;
                    sums("raw" + std::to_string(l), a_raw, b, c.H, 0);
                    sums("codes" + std::to_string(l), a_codes, g, c.C, 1);     // the native codes are rows 1..b-1
                    sums("out" + std::to_string(l), a_out, b, c.H, 0);
                    if (full) {
                        const double er = nerr(a_raw, fx(step + ".raw" + std::to_string(l)).f.data(), std::size_t(b) * c.H);
                        const double ec = nerr(a_codes, fx(step + ".codes" + std::to_string(l)).f.data() + c.C, std::size_t(g) * c.C);
                        const double eo = nerr(a_out, fx(step + ".out" + std::to_string(l)).f.data(), std::size_t(b) * c.H);
                        std::printf("  %s layer %d: raw %.3g codes %.3g out %.3g\n", step.c_str(), l, er, ec, eo);
                        r.worst_stage = std::max({ r.worst_stage, er, ec, eo });
                        if (!(er < t.stage && ec < t.stage && eo < t.stage)) r.fail("layer stages");
                    }
                }
                sums("final", fin.data(), b, c.H, 0);
                if (full) {
                    const double ef = nerr(fin.data(), fx(step + ".final").f.data(), fin.size());
                    r.worst_stage = std::max(r.worst_stage, ef);
                    if (!(ef < t.stage)) r.fail("final rows");
                }
                // base logits, in the native candidate order -> by id
                const Arr& rb = fx(step + ".base");
                const Arr& rc = fx(step + ".cand");
                double num = 0, den = 1e-30;
                for (float v : rb.f) den = std::max(den, double(std::fabs(v)));
                for (int e = 0; e < g; ++e)
                    for (int i = 0; i < c.K; ++i) {
                        const int id = d.cand[e * c.K + i];
                        const int* row = rc.i.data() + e * c.K;
                        const int* at = std::find(row, row + c.K, id);
                        if (at != row + c.K) num = std::max(num, double(std::fabs(base[std::size_t(e) * c.K + i] - rb.f[std::size_t(e) * c.K + (at - row)])));
                    }
                r.worst_stage = std::max(r.worst_stage, num / den);
                if (!(num / den < t.stage)) r.fail("base logits");
            }
            canonical(c.K, g, d.cand, d.scores);
            compare_tail(c, g, pre, step, d, t, ti == 0 || (name == "small" && si == 0), r);

            if (si == 2 && ti == 1) {
                // A replay of the cached graph returns exactly what the first run returned.
                Draft again;
                if (!draft(m, w, tab[ti], ring, anchor, b, again)) return false;
                canonical(c.K, g, again.cand, again.scores);
                if (again.cand != d.cand || again.scores != d.scores) r.fail("cached replay differs from its first run");
                last16 = d;
                // Causal block: 6 rows are the first rows of the 16-row block.
                Draft six;
                if (!draft(m, w, tab[ti], ring, anchor, 6, six)) return false;
                canonical(c.K, 5, six.cand, six.scores);
                const bool cand_same = std::equal(six.cand.begin(), six.cand.end(), last16.cand.begin());
                const double e = nerr(six.scores.data(), last16.scores.data(), six.scores.size());
                std::printf("  %s block 6 vs the first rows of block 16: candidates same %d, lattice %.3g\n",
                            name.c_str(), int(cand_same), e);
                if (!cand_same || !(e < t.lattice)) r.fail("causal block prefix");
            }
        }
    }
    TSGgml_DSpineResetCaches();
    TSGgml_DSpineTablesFree(tab[0]);
    TSGgml_DSpineTablesFree(tab[1]);
    TSGgml_DFlashResetCaches();
    TSGgml_ClearHostBufferCache();
    return true;
}
}

int main(int argc, char** argv) {
    if (argc != 3) { std::fprintf(stderr, "usage: %s <cpu|metal|cuda> <fixture>\n", argv[0]); return 2; }
    const int backend = !std::strcmp(argv[1], "cuda") ? 3 : !std::strcmp(argv[1], "metal") ? 1 : 2;
    if (!TSGgml_IsBackendAvailable(backend)) return 77;
    if (!load_fixture(argv[2])) return 1;

    // Tolerances. CPU and CUDA keep F32 in the F32 matmuls; ggml-metal's batched matmuls round activations to
    // half, and a BF16 bridge weight makes CPU round the activation it multiplies to BF16. The reference holds
    // exact (float64) values, so these bound the backend's rounding, not a difference in the math.
    // gap / margin: a candidate set or chain step is only compared where the reference's own separation is larger.
    bool all = true;
    for (bool bridge_bf16 : {true, false}) {
        Tol t;
        if (backend == 1)      t = { 3e-3, 2e-3, 3e-3, 1e-2, 1e-2 };
        else if (bridge_bf16)  t = { 8e-3, 4e-3, 8e-3, 2e-2, 2e-2 };
        else                   t = { 1e-4, 1e-4, 1e-4, 1e-3, 1e-3 };
        for (const char* name : {"small", "wide"}) {
            Report r;
            if (!run_config(name, backend, bridge_bf16, t, r)) return 1;
            std::printf("%s %s bridge %s: stages %.3g, checksums %.3g, lattice %.3g, confidences %.3g; candidate rows %d "
                        "(%d within the reference's k-gap noise), chain steps checked %d (%d walks stopped at a near tie)\n",
                        r.ok ? "ok  " : "FAIL", name, bridge_bf16 ? "BF16" : "F32", r.worst_stage, r.worst_sum,
                        r.worst_lattice, r.worst_conf, r.cand_rows, r.cand_ambiguous, r.chain_checked, r.chain_stopped);
            all = all && r.ok && r.chain_checked > 0;
        }
    }
    TSGgml_Shutdown();
    if (!all) { std::puts("FAIL DSpine draft vs the NumPy reference"); return 1; }
    std::puts("PASS DSpine draft: tables, every stage, candidates, lattices and greedy chains match the NumPy reference");
    return 0;
}
