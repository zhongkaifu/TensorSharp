// Copyright (c) Zhongkai Fu. All rights reserved.
// Licensed under the BSD-3-Clause license in the repository root.
//
// Row-invariance probe for the kernels a Qwen 3.8 Flash Next (qwen4exp) target
// span runs, at the shapes and weight types of the published UD-Q2_K_XL
// checkpoint. For every op and every width T in 2..8 it computes the same T
// token rows three ways on one device:
//
//   batched  - one node over all T rows (what a speculative verify builds today)
//   single   - T separate one-row graphs (what T plain decode steps build)
//   variant  - the candidate row-invariant construction for that op
//
// and reports whether batched and variant rows are bit-identical to the
// single-row graphs, the largest absolute difference, and the mean wall time
// of the batched and variant graphs (synchronized, replayed so ggml-cuda can
// capture them). It asserts nothing; it is the measurement behind the verify
// row policy in ggml_ops_qwen4exp.cpp.
//
//   GgmlOpsQwen4ExpRowKernelProbe [--repeats N] [--experts E] [--kv N]
#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"
#ifdef TSG_GGML_USE_CUDA
#include "ggml-cuda.h"
#endif

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <random>
#include <string>
#include <vector>

namespace {

void require(bool ok, const char* what)
{
    if (!ok) { std::fprintf(stderr, "probe failure: %s\n", what); std::exit(1); }
}

int g_repeats = 30;
int g_experts = 32;
int g_kv = 4096;

// Host data for one weight tensor, already in its storage type.
struct Weight
{
    ggml_type type;
    std::vector<int64_t> ne;
    std::vector<uint8_t> bytes;
};

Weight make_weight(ggml_type type, std::vector<int64_t> ne, uint32_t seed)
{
    Weight w{type, ne, {}};
    const int64_t n_per_row = ne[0];
    int64_t rows = 1;
    for (size_t d = 1; d < ne.size(); ++d) rows *= ne[d];
    std::mt19937 random(seed);
    std::normal_distribution<float> normal(0.0f, 0.02f);
    std::vector<float> values((size_t)(n_per_row * rows));
    for (float& v : values) v = normal(random);
    if (type == GGML_TYPE_F32)
    {
        w.bytes.resize(values.size() * 4);
        std::memcpy(w.bytes.data(), values.data(), w.bytes.size());
        return w;
    }
    if (type == GGML_TYPE_F16 || type == GGML_TYPE_BF16)
    {
        w.bytes.resize(values.size() * 2);
        for (size_t i = 0; i < values.size(); ++i)
        {
            if (type == GGML_TYPE_F16) { ggml_fp16_t h = ggml_fp32_to_fp16(values[i]); std::memcpy(&w.bytes[2 * i], &h, 2); }
            else { ggml_bf16_t h = ggml_fp32_to_bf16(values[i]); std::memcpy(&w.bytes[2 * i], &h, 2); }
        }
        return w;
    }
    ggml_quantize_init(type);
    std::vector<float> imatrix((size_t)n_per_row, 1.0f);
    const size_t row_size = ggml_row_size(type, n_per_row);
    w.bytes.resize(row_size * (size_t)rows);
    // One matrix (expert) at a time: the imatrix applies per n_per_row.
    const int64_t per_matrix = ne.size() > 1 ? ne[1] : 1;
    const int64_t matrices = rows / per_matrix;
    for (int64_t m = 0; m < matrices; ++m)
        ggml_quantize_chunk(type, values.data() + m * per_matrix * n_per_row, w.bytes.data() + m * per_matrix * row_size,
                            0, per_matrix, n_per_row, ggml_quantize_requires_imatrix(type) ? imatrix.data() : nullptr);
    return w;
}

// One graph over one context; inputs are named host vectors uploaded before
// each compute, weights uploaded once.
struct Graph
{
    ggml_context* ctx = nullptr;
    ggml_cgraph* graph = nullptr;
    ggml_backend_buffer_t buffer = nullptr;
    std::vector<std::pair<ggml_tensor*, const Weight*>> weights;
    std::vector<std::pair<ggml_tensor*, std::vector<uint8_t>>> inputs;
    std::vector<ggml_tensor*> outputs;

    Graph()
    {
        ggml_init_params params{ggml_tensor_overhead() * 4096 + ggml_graph_overhead_custom(8192, false), nullptr, true};
        ctx = ggml_init(params);
        graph = ggml_new_graph_custom(ctx, 8192, false);
    }
    ~Graph()
    {
        if (buffer) ggml_backend_buffer_free(buffer);
        ggml_free(ctx);
    }
    ggml_tensor* weight(const Weight& w)
    {
        ggml_tensor* t = ggml_new_tensor(ctx, w.type, (int)w.ne.size(), w.ne.data());
        weights.push_back({t, &w});
        return t;
    }
    ggml_tensor* input(ggml_type type, std::vector<int64_t> ne, const std::vector<uint8_t>& data)
    {
        ggml_tensor* t = ggml_new_tensor(ctx, type, (int)ne.size(), ne.data());
        ggml_set_input(t);
        inputs.push_back({t, data});
        return t;
    }
    void output(ggml_tensor* t)
    {
        ggml_set_output(t);
        ggml_build_forward_expand(graph, t);
        outputs.push_back(t);
    }
    void allocate(ggml_backend_t backend)
    {
        // Weights and inputs live in a context buffer; intermediates in gallocr.
        buffer = ggml_backend_alloc_ctx_tensors(ctx, backend);
        require(buffer != nullptr, "weight allocation");
        for (auto& w : weights) ggml_backend_tensor_set(w.first, w.second->bytes.data(), 0, ggml_nbytes(w.first));
        for (auto& in : inputs) ggml_backend_tensor_set(in.first, in.second.data(), 0, ggml_nbytes(in.first));
    }
};

std::vector<uint8_t> floats(const std::vector<float>& v)
{
    std::vector<uint8_t> b(v.size() * 4);
    std::memcpy(b.data(), v.data(), b.size());
    return b;
}

std::vector<float> read(ggml_tensor* t)
{
    std::vector<float> v((size_t)ggml_nelements(t));
    ggml_backend_tensor_get(t, v.data(), 0, v.size() * 4);
    return v;
}

double time_graph(ggml_backend_t backend, ggml_cgraph* graph)
{
    for (int i = 0; i < 3; ++i) require(ggml_backend_graph_compute(backend, graph) == GGML_STATUS_SUCCESS, "warmup");
    ggml_backend_synchronize(backend);
    const auto start = std::chrono::steady_clock::now();
    for (int i = 0; i < g_repeats; ++i) require(ggml_backend_graph_compute(backend, graph) == GGML_STATUS_SUCCESS, "compute");
    ggml_backend_synchronize(backend);
    return std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now() - start).count() / g_repeats;
}

struct Compare { bool exact = true; double max_abs = 0; };

// rows[r] is a flat per-row vector; batched holds T rows laid out along the
// token axis with `per_row` elements each.
Compare compare_rows(const std::vector<float>& batched, const std::vector<std::vector<float>>& rows, size_t per_row)
{
    Compare c;
    for (size_t r = 0; r < rows.size(); ++r)
        for (size_t i = 0; i < per_row; ++i)
        {
            const float a = batched[r * per_row + i], b = rows[r][i];
            if (std::memcmp(&a, &b, 4) != 0) c.exact = false;
            c.max_abs = std::max(c.max_abs, (double)std::fabs((double)a - b));
        }
    return c;
}

void report(const char* op, const std::string& shape, int T, const Compare& batched, const Compare& variant,
            const char* variant_name, double us_batched, double us_variant)
{
    std::printf("%-10s %-34s T=%d batched:%s max_abs=%-10.3g %s:%s max_abs=%-10.3g us_batched=%.1f us_%s=%.1f\n",
                op, shape.c_str(), T, batched.exact ? "exact " : "DIFFER", batched.max_abs,
                variant_name, variant.exact ? "exact " : "DIFFER", variant.max_abs, us_batched, variant_name, us_variant);
}

std::vector<float> random_vector(size_t n, uint32_t seed, float scale = 1.0f)
{
    std::mt19937 random(seed);
    std::normal_distribution<float> normal(0.0f, scale);
    std::vector<float> v(n);
    for (float& x : v) x = normal(random);
    return v;
}

// ---- dense MUL_MAT -----------------------------------------------------------
void probe_dense(ggml_backend_t backend, ggml_type type, int64_t in, int64_t out, int T)
{
    Weight w = make_weight(type, {in, out}, 11);
    std::vector<float> x = random_vector((size_t)(in * T), 101);
    auto run_rows = [&]() {
        std::vector<std::vector<float>> rows;
        for (int r = 0; r < T; ++r)
        {
            Graph g;
            std::vector<float> xr(x.begin() + r * in, x.begin() + (r + 1) * in);
            auto* xi = g.input(GGML_TYPE_F32, {in, 1}, floats(xr));
            g.output(ggml_mul_mat(g.ctx, g.weight(w), ggml_scale(g.ctx, xi, 1.0f)));
            g.allocate(backend);
            require(ggml_backend_graph_compute(backend, g.graph) == GGML_STATUS_SUCCESS, "dense single");
            rows.push_back(read(g.outputs[0]));
        }
        return rows;
    };
    auto rows = run_rows();
    Graph b;
    {
        auto* xi = b.input(GGML_TYPE_F32, {in, T}, floats(x));
        b.output(ggml_mul_mat(b.ctx, b.weight(w), ggml_scale(b.ctx, xi, 1.0f)));
        b.allocate(backend);
    }
    const double us_b = time_graph(backend, b.graph);
    const Compare cb = compare_rows(read(b.outputs[0]), rows, (size_t)out);
    // Variant: tokens moved onto the broadcast (channel) axis, one launch.
    Graph v;
    {
        auto* xi = v.input(GGML_TYPE_F32, {in, T}, floats(x));
        auto* xs = ggml_reshape_3d(v.ctx, ggml_scale(v.ctx, xi, 1.0f), in, 1, T);
        v.output(ggml_mul_mat(v.ctx, v.weight(w), xs));
        v.allocate(backend);
    }
    const double us_v = time_graph(backend, v.graph);
    const Compare cv = compare_rows(read(v.outputs[0]), rows, (size_t)out);
    report("MUL_MAT", std::string(ggml_type_name(type)) + " " + std::to_string(in) + "->" + std::to_string(out),
           T, cb, cv, "channels", us_b, us_v);
}

// ---- expert MUL_MAT_ID -------------------------------------------------------
void probe_experts(ggml_backend_t backend, ggml_type type, int64_t in, int64_t out, int used, bool per_slot_input, int T)
{
    const int E = g_experts;
    Weight w = make_weight(type, {in, out, E}, 23);
    const int slots = per_slot_input ? used : 1;
    std::vector<float> x = random_vector((size_t)(in * slots * T), 202);
    std::vector<int32_t> ids((size_t)(used * T));
    std::mt19937 random(303);
    for (int t = 0; t < T; ++t)
    {
        std::vector<int32_t> all(E);
        for (int e = 0; e < E; ++e) all[e] = e;
        std::shuffle(all.begin(), all.end(), random);
        for (int u = 0; u < used; ++u) ids[(size_t)(t * used + u)] = all[u];
    }
    auto ids_bytes = [&](int first, int count) {
        std::vector<uint8_t> b((size_t)(used * count) * 4);
        std::memcpy(b.data(), ids.data() + first * used, b.size());
        return b;
    };
    std::vector<std::vector<float>> rows;
    for (int r = 0; r < T; ++r)
    {
        Graph g;
        std::vector<float> xr(x.begin() + r * in * slots, x.begin() + (r + 1) * in * slots);
        auto* xi = g.input(GGML_TYPE_F32, {in, slots, 1}, floats(xr));
        auto* ii = g.input(GGML_TYPE_I32, {used, 1}, ids_bytes(r, 1));
        g.output(ggml_mul_mat_id(g.ctx, g.weight(w), ggml_scale(g.ctx, xi, 1.0f), ii));
        g.allocate(backend);
        require(ggml_backend_graph_compute(backend, g.graph) == GGML_STATUS_SUCCESS, "experts single");
        rows.push_back(read(g.outputs[0]));
    }
    Graph b;
    {
        auto* xi = b.input(GGML_TYPE_F32, {in, slots, T}, floats(x));
        auto* ii = b.input(GGML_TYPE_I32, {used, T}, ids_bytes(0, T));
        b.output(ggml_mul_mat_id(b.ctx, b.weight(w), ggml_scale(b.ctx, xi, 1.0f), ii));
        b.allocate(backend);
    }
    const double us_b = time_graph(backend, b.graph);
    const Compare cb = compare_rows(read(b.outputs[0]), rows, (size_t)(out * used));
    // Variant: one MUL_MAT_ID per token row inside one graph, concatenated.
    Graph v;
    {
        auto* xi = v.input(GGML_TYPE_F32, {in, slots, T}, floats(x));
        auto* ii = v.input(GGML_TYPE_I32, {used, T}, ids_bytes(0, T));
        auto* xs = ggml_scale(v.ctx, xi, 1.0f);
        auto* wt = v.weight(w);
        ggml_tensor* acc = nullptr;
        for (int r = 0; r < T; ++r)
        {
            auto* xr = ggml_cont(v.ctx, ggml_view_3d(v.ctx, xs, in, slots, 1, xs->nb[1], xs->nb[2], (size_t)r * xs->nb[2]));
            auto* ir = ggml_view_2d(v.ctx, ii, used, 1, ii->nb[1], (size_t)r * ii->nb[1]);
            auto* y = ggml_mul_mat_id(v.ctx, wt, xr, ir);
            acc = acc ? ggml_concat(v.ctx, acc, y, 2) : y;
        }
        v.output(acc);
        v.allocate(backend);
    }
    const double us_v = time_graph(backend, v.graph);
    const Compare cv = compare_rows(read(v.outputs[0]), rows, (size_t)(out * used));
    report("MUL_MAT_ID", std::string(ggml_type_name(type)) + " " + std::to_string(in) + "->" + std::to_string(out)
           + " E" + std::to_string(E) + " k" + std::to_string(used) + (per_slot_input ? " slots" : ""),
           T, cb, cv, "rows", us_b, us_v);
}

// ---- flash attention ---------------------------------------------------------
void probe_flash(ggml_backend_t backend, int T, int head_dim, int n_head, int n_head_kv, int kv_pad, int live_before)
{
    const std::vector<float> kf = random_vector((size_t)head_dim * kv_pad * n_head_kv, 404, 0.5f);
    const std::vector<float> vf = random_vector((size_t)head_dim * kv_pad * n_head_kv, 405, 0.5f);
    auto f16 = [](const std::vector<float>& v) {
        std::vector<uint8_t> b(v.size() * 2);
        for (size_t i = 0; i < v.size(); ++i) { ggml_fp16_t h = ggml_fp32_to_fp16(v[i]); std::memcpy(&b[2 * i], &h, 2); }
        return b;
    };
    const std::vector<float> q = random_vector((size_t)head_dim * T * n_head, 406, 0.5f);
    auto mask_rows = [&](int first, int count, int width) {
        std::vector<uint8_t> b((size_t)width * count * 2);
        for (int t = 0; t < count; ++t)
            for (int k = 0; k < width; ++k)
            {
                ggml_fp16_t h = k <= live_before + first + t ? ggml_fp32_to_fp16(0.0f) : ggml_fp32_to_fp16(-INFINITY);
                std::memcpy(&b[2 * ((size_t)t * width + k)], &h, 2);
            }
        return b;
    };
    const float scale = 1.0f / std::sqrt((float)head_dim);
    auto build = [&](Graph& g, ggml_tensor* qt, int first, int count, int width) {
        auto* kt = g.input(GGML_TYPE_F16, {head_dim, kv_pad, n_head_kv}, f16(kf));
        auto* vt = g.input(GGML_TYPE_F16, {head_dim, kv_pad, n_head_kv}, f16(vf));
        auto* kv = ggml_view_3d(g.ctx, kt, head_dim, width, n_head_kv, kt->nb[1], kt->nb[2], 0);
        auto* vv = ggml_view_3d(g.ctx, vt, head_dim, width, n_head_kv, vt->nb[1], vt->nb[2], 0);
        auto* mt = g.input(GGML_TYPE_F16, {width, count}, mask_rows(first, count, width));
        auto* fa = ggml_flash_attn_ext(g.ctx, qt, kv, vv, mt, scale, 0.0f, 0.0f);
        ggml_flash_attn_ext_set_prec(fa, GGML_PREC_F32);
        return fa;
    };
    std::vector<std::vector<float>> rows;
    for (int r = 0; r < T; ++r)
    {
        Graph g;
        std::vector<float> qr((size_t)head_dim * n_head);
        for (int h = 0; h < n_head; ++h)
            for (int d = 0; d < head_dim; ++d) qr[(size_t)h * head_dim + d] = q[((size_t)h * T + r) * head_dim + d];
        auto* qt = ggml_scale(g.ctx, g.input(GGML_TYPE_F32, {head_dim, 1, n_head}, floats(qr)), 1.0f);
        g.output(build(g, qt, r, 1, kv_pad));
        g.allocate(backend);
        require(ggml_backend_graph_compute(backend, g.graph) == GGML_STATUS_SUCCESS, "flash single");
        rows.push_back(read(g.outputs[0]));
    }
    Graph b;
    {
        auto* qt = ggml_scale(b.ctx, b.input(GGML_TYPE_F32, {head_dim, T, n_head}, floats(q)), 1.0f);
        b.output(build(b, qt, 0, T, kv_pad));
        b.allocate(backend);
    }
    const double us_b = time_graph(backend, b.graph);
    const Compare cb = compare_rows(read(b.outputs[0]), rows, (size_t)head_dim * n_head);
    Graph v;
    {
        auto* qall = ggml_scale(v.ctx, v.input(GGML_TYPE_F32, {head_dim, T, n_head}, floats(q)), 1.0f);
        auto* kt = v.input(GGML_TYPE_F16, {head_dim, kv_pad, n_head_kv}, f16(kf));
        auto* vt = v.input(GGML_TYPE_F16, {head_dim, kv_pad, n_head_kv}, f16(vf));
        auto* mt = v.input(GGML_TYPE_F16, {kv_pad, T}, mask_rows(0, T, kv_pad));
        ggml_tensor* acc = nullptr;
        for (int r = 0; r < T; ++r)
        {
            auto* qr = ggml_cont(v.ctx, ggml_view_3d(v.ctx, qall, head_dim, 1, n_head, qall->nb[1], qall->nb[2], (size_t)r * qall->nb[1]));
            auto* mr = ggml_cont(v.ctx, ggml_view_2d(v.ctx, mt, kv_pad, 1, mt->nb[1], (size_t)r * mt->nb[1]));
            auto* fa = ggml_flash_attn_ext(v.ctx, qr, kt, vt, mr, scale, 0.0f, 0.0f);
            ggml_flash_attn_ext_set_prec(fa, GGML_PREC_F32);
            acc = acc ? ggml_concat(v.ctx, acc, fa, 2) : fa;
        }
        v.output(acc);
        v.allocate(backend);
    }
    const double us_v = time_graph(backend, v.graph);
    const Compare cv = compare_rows(read(v.outputs[0]), rows, (size_t)head_dim * n_head);
    report("FLASH_ATTN", "hd" + std::to_string(head_dim) + " h" + std::to_string(n_head) + "/" + std::to_string(n_head_kv)
           + " kv" + std::to_string(kv_pad), T, cb, cv, "rows", us_b, us_v);
}

// ---- gated delta net ---------------------------------------------------------
void probe_gdn(ggml_backend_t backend, int T, int S, int H)
{
    const auto q = random_vector((size_t)S * H * T, 501, 0.1f), k = random_vector((size_t)S * H * T, 502, 0.1f),
               vv = random_vector((size_t)S * H * T, 503, 0.5f), g = random_vector((size_t)H * T, 504, 0.2f),
               beta = random_vector((size_t)H * T, 505, 0.5f), s0 = random_vector((size_t)S * S * H, 506, 0.05f);
    auto slice = [](const std::vector<float>& v, size_t per, int r) {
        return std::vector<float>(v.begin() + per * r, v.begin() + per * (r + 1));
    };
    auto build = [&](Graph& gr, int count, const std::vector<float>& qv, const std::vector<float>& kv, const std::vector<float>& vv2,
                     const std::vector<float>& gv, const std::vector<float>& bv, const std::vector<float>& state) {
        auto* qt = gr.input(GGML_TYPE_F32, {S, H, count, 1}, floats(qv));
        auto* kt = gr.input(GGML_TYPE_F32, {S, H, count, 1}, floats(kv));
        auto* vt = gr.input(GGML_TYPE_F32, {S, H, count, 1}, floats(vv2));
        auto* gt = gr.input(GGML_TYPE_F32, {1, H, count, 1}, floats(gv));
        auto* bt = gr.input(GGML_TYPE_F32, {1, H, count, 1}, floats(bv));
        auto* st = gr.input(GGML_TYPE_F32, {S, S, H, 1}, floats(state));
        return ggml_gated_delta_net(gr.ctx, ggml_scale(gr.ctx, qt, 1.0f), kt, vt, ggml_scale(gr.ctx, ggml_sigmoid(gr.ctx, gt), -1.0f), ggml_sigmoid(gr.ctx, bt), st, 1);
    };
    std::vector<std::vector<float>> rows;
    std::vector<float> state = s0;
    const size_t core = (size_t)S * H;
    for (int r = 0; r < T; ++r)
    {
        Graph gr;
        gr.output(build(gr, 1, slice(q, core, r), slice(k, core, r), slice(vv, core, r), slice(g, H, r), slice(beta, H, r), state));
        gr.allocate(backend);
        require(ggml_backend_graph_compute(backend, gr.graph) == GGML_STATUS_SUCCESS, "gdn single");
        auto out = read(gr.outputs[0]);
        rows.push_back(std::vector<float>(out.begin(), out.begin() + core));
        state.assign(out.begin() + core, out.end());
    }
    Graph b;
    b.output(build(b, T, q, k, vv, g, beta, s0));
    b.allocate(backend);
    const double us_b = time_graph(backend, b.graph);
    auto out = read(b.outputs[0]);
    Compare cb = compare_rows(out, rows, core);
    const std::vector<float> final_state(out.begin() + core * T, out.end());
    Compare cs = compare_rows(final_state, {state}, state.size());
    report("GDN", "S" + std::to_string(S) + " H" + std::to_string(H), T, cb, cs, "state", us_b, 0.0);
}

} // namespace

int main(int argc, char** argv)
{
    std::setvbuf(stdout, nullptr, _IONBF, 0);
    for (int i = 1; i < argc; ++i)
    {
        if (!std::strcmp(argv[i], "--repeats") && i + 1 < argc) g_repeats = std::atoi(argv[++i]);
        else if (!std::strcmp(argv[i], "--experts") && i + 1 < argc) g_experts = std::atoi(argv[++i]);
        else if (!std::strcmp(argv[i], "--kv") && i + 1 < argc) g_kv = std::atoi(argv[++i]);
    }
#ifdef TSG_GGML_USE_CUDA
    if (ggml_backend_cuda_get_device_count() == 0) return 77;
    ggml_backend_t backend = ggml_backend_cuda_init(0);
#else
    ggml_backend_t backend = ggml_backend_cpu_init();
#endif
    require(backend != nullptr, "backend");
    std::printf("backend %s repeats=%d experts=%d kv=%d\n", ggml_backend_name(backend), g_repeats, g_experts, g_kv);
    const int widths[] = {2, 3, 4, 5, 6, 7, 8};
    for (int T : widths)
    {
        // Dense projections of the UD-Q2_K_XL checkpoint.
        probe_dense(backend, GGML_TYPE_Q5_K, 2560, 10240, T);  // attn_qkv
        probe_dense(backend, GGML_TYPE_Q6_K, 2560, 512, T);    // attn_k / attn_v
        probe_dense(backend, GGML_TYPE_Q8_0, 10240, 320, T);   // hc_*_down
        probe_dense(backend, GGML_TYPE_Q4_K, 2560, 32768, T);  // output (vocabulary slice)
        probe_dense(backend, GGML_TYPE_F32, 2560, 512, T);     // ffn_gate_inp (router)
        probe_dense(backend, GGML_TYPE_F32, 10240, 4, T);      // hc_*_inject
        probe_dense(backend, GGML_TYPE_BF16, 2560, 512, T);    // indexer.q_proj
        probe_dense(backend, GGML_TYPE_F16, 2560, 640, T);     // F16 checkpoints
        // Routed experts.
        probe_experts(backend, GGML_TYPE_IQ2_XS, 2560, 640, 10, false, T);  // ffn_up/gate_exps
        probe_experts(backend, GGML_TYPE_IQ4_NL, 640, 2560, 10, true, T);   // ffn_down_exps
        probe_experts(backend, GGML_TYPE_F16, 2560, 640, 10, false, T);     // F16 experts (fixture/BF16 releases)
        // Attention: 24 query heads over 2 KV heads, head 256, F16 KV.
        probe_flash(backend, T, 256, 24, 2, g_kv, g_kv / 2);
        probe_gdn(backend, T, 128, 48);
    }
    ggml_backend_free(backend);
    return 0;
}
