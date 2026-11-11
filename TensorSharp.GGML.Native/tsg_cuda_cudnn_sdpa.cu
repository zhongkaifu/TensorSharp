// Copyright (c) Zhongkai Fu. All rights reserved.
// Licensed under the BSD-3-Clause license in the repository root.
//
// Long unmasked attention on cuDNN's fused SDPA (flash attention) forward.
//
// MiniMax-H3's DiT attends over one packed sequence with no mask: 20,511 tokens,
// 56 heads x 128 for a 480x864, 158-frame clip. ggml-cuda runs that on its MMA F16
// flash-attention kernel at 16.6 TFLOP/s where cuDNN's SDPA ran the same shape
// at 80-83 TFLOP/s on the same GPU, so it is offered here as an execution of
// the graph's own GGML_OP_FLASH_ATTN_EXT node (see graph_compute_h3_offload): the
// graph, its allocation and every other node stay ggml's, and ggml is unchanged.
//
// The operation is cuDNN's single backend SDPA descriptor
// (CUDNN_BACKEND_OPERATION_SDPA_FWD_DESCRIPTOR, cuDNN 9.13+), so no cudnn-frontend
// is needed. Inputs are F16 with F32 accumulation inside cuDNN. ggml's node has an
// F32 Q and an F32 output; Q is narrowed to F16 into scratch (as ggml's kernel
// itself narrows Q in registers) and the F16 result widened into the node's
// buffer. Unlike ggml's MMA kernel, whose softmax numerator accumulates in FP16
// (see h3_attend), cuDNN accumulates P*V in FP32.
//
// cuDNN is resolved at run time exactly like tsg_cuda_cudnn_conv.cu (same search
// order and CUDA-major check), so GgmlOps still loads without it; a missing
// library, an older cuDNN or a shape no engine takes declines and the node runs on
// ggml. TS_H3_CUDNN_ATTENTION=0 turns the route off.
#if defined(TSG_GGML_USE_CUDA) && defined(TSG_HAVE_CUDNN)

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cudnn.h>

#include "tsg_cuda_stage_profile.h"

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <mutex>
#include <string>
#include <tuple>
#include <vector>

#if defined(_WIN32)
#  define WIN32_LEAN_AND_MEAN
#  include <windows.h>
#else
#  include <dlfcn.h>
#endif

namespace {

#define TSG_SDPA_ENTRIES(X)                \
    X(cudnnCreate)                         \
    X(cudnnDestroy)                        \
    X(cudnnSetStream)                      \
    X(cudnnGetVersion)                     \
    X(cudnnGetErrorString)                 \
    X(cudnnGetLastErrorString)             \
    X(cudnnBackendCreateDescriptor)        \
    X(cudnnBackendDestroyDescriptor)       \
    X(cudnnBackendSetAttribute)            \
    X(cudnnBackendGetAttribute)            \
    X(cudnnBackendFinalize)                \
    X(cudnnBackendExecute)

struct SdpaApi
{
#define TSG_SDPA_DECL(name) decltype(&::name) fn_##name = nullptr;
    TSG_SDPA_ENTRIES(TSG_SDPA_DECL)
#undef TSG_SDPA_DECL
    bool loaded = false;
};

#if defined(_WIN32)
using LibHandle = HMODULE;

void prepend_to_path(const std::string& dir)
{
    if (dir.empty()) return;
    DWORD n = GetEnvironmentVariableA("PATH", nullptr, 0);
    std::string cur;
    if (n > 0) { cur.resize(n); GetEnvironmentVariableA("PATH", &cur[0], n); cur.resize(n - 1); }
    if (cur.find(dir) != std::string::npos) return;
    SetEnvironmentVariableA("PATH", (dir + ";" + cur).c_str());
}

LibHandle lib_open(const std::string& path)
{
    const bool hasDir = path.find('\\') != std::string::npos || path.find('/') != std::string::npos;
    std::wstring w(path.begin(), path.end());
    return LoadLibraryExW(w.c_str(), nullptr, hasDir ? LOAD_WITH_ALTERED_SEARCH_PATH : 0);
}
void* lib_sym(LibHandle h, const char* n) { return (void*) GetProcAddress(h, n); }

// The dispatcher aborts the process when a sibling is missing (see the conv
// module), so look before loading.
bool siblings_present(const std::string& dir)
{
    static const char* const required[] = {
        "cudnn_graph64_9.dll", "cudnn_ops64_9.dll", "cudnn_heuristic64_9.dll",
        "cudnn_engines_precompiled64_9.dll",
    };
    for (const char* r : required)
    {
        const std::string p = dir + "\\" + r;
        if (GetFileAttributesA(p.c_str()) == INVALID_FILE_ATTRIBUTES) return false;
    }
    return true;
}
const char* const kCudnnNames[] = { "cudnn64_9.dll" };
const char kSep = '\\';
// As the convolution module, plus bin\arm64 where eng/fetch-cudnn puts the ARM64 redistributable.
const char* const kLibSubdirs[] = { "bin\\x64", "bin\\arm64", "bin", "" };
#else
using LibHandle = void*;
void prepend_to_path(const std::string&) {}
LibHandle lib_open(const std::string& path) { return dlopen(path.c_str(), RTLD_LAZY | RTLD_GLOBAL); }
void* lib_sym(LibHandle h, const char* n) { return dlsym(h, n); }
bool siblings_present(const std::string&) { return true; }
const char* const kCudnnNames[] = { "libcudnn.so.9" };
const char kSep = '/';
const char* const kLibSubdirs[] = { "lib", "lib64", "" };
#endif

LibHandle open_cudnn()
{
    std::vector<std::string> roots;
    if (const char* e = std::getenv("TS_CUDNN_DIR"); e != nullptr && e[0] != '\0')
        roots.emplace_back(e);
#if defined(TSG_CUDNN_RUNTIME_DIR)
    roots.emplace_back(TSG_CUDNN_RUNTIME_DIR);
#endif
    for (const std::string& root : roots)
        for (const char* sub : kLibSubdirs)
        {
            const std::string dir = sub[0] == '\0' ? root : root + kSep + sub;
            if (!siblings_present(dir)) continue;
            prepend_to_path(dir);
            for (const char* n : kCudnnNames)
                if (LibHandle h = lib_open(dir + kSep + n); h != nullptr) return h;
        }
    for (const char* n : kCudnnNames)
        if (LibHandle h = lib_open(n); h != nullptr) return h;
    return nullptr;
}

SdpaApi& api()
{
    static SdpaApi a = []() {
        SdpaApi r;
        LibHandle h = open_cudnn();
        if (h == nullptr) return r;
        using GetCudartVersionFn = size_t (CUDNNWINAPI *)(void);
        auto getCudartVersion = (GetCudartVersionFn) lib_sym(h, "cudnnGetCudartVersion");
        if (getCudartVersion == nullptr || getCudartVersion() / 1000 != CUDART_VERSION / 1000) return r;
        bool ok = true;
#define TSG_SDPA_LOAD(name)                                     \
        r.fn_##name = (decltype(&::name)) lib_sym(h, #name);    \
        if (r.fn_##name == nullptr) ok = false;
        TSG_SDPA_ENTRIES(TSG_SDPA_LOAD)
#undef TSG_SDPA_LOAD
        // The single SDPA backend operation arrived in 9.13.
        r.loaded = ok && r.fn_cudnnGetVersion() >= 91300;
        return r;
    }();
    return a;
}

#define cudnnCreate                   (api().fn_cudnnCreate)
#define cudnnDestroy                  (api().fn_cudnnDestroy)
#define cudnnSetStream                (api().fn_cudnnSetStream)
#define cudnnGetErrorString           (api().fn_cudnnGetErrorString)
#define cudnnGetLastErrorString       (api().fn_cudnnGetLastErrorString)
#define cudnnBackendCreateDescriptor  (api().fn_cudnnBackendCreateDescriptor)
#define cudnnBackendDestroyDescriptor (api().fn_cudnnBackendDestroyDescriptor)
#define cudnnBackendSetAttribute      (api().fn_cudnnBackendSetAttribute)
#define cudnnBackendGetAttribute      (api().fn_cudnnBackendGetAttribute)
#define cudnnBackendFinalize          (api().fn_cudnnBackendFinalize)
#define cudnnBackendExecute           (api().fn_cudnnBackendExecute)

// Owns one backend descriptor.
struct Desc
{
    cudnnBackendDescriptor_t d = nullptr;
    Desc() = default;
    explicit Desc(cudnnBackendDescriptorType_t t) { if (cudnnBackendCreateDescriptor(t, &d) != CUDNN_STATUS_SUCCESS) d = nullptr; }
    Desc(const Desc&) = delete;
    Desc& operator=(const Desc&) = delete;
    Desc(Desc&& o) noexcept : d(o.d) { o.d = nullptr; }
    Desc& operator=(Desc&& o) noexcept { if (this != &o) { reset(); d = o.d; o.d = nullptr; } return *this; }
    ~Desc() { reset(); }
    void reset() { if (d != nullptr && api().loaded) cudnnBackendDestroyDescriptor(d); d = nullptr; }
};

bool verbose()
{
    static const bool v = []{ const char* e = std::getenv("TS_H3_CUDNN_ATTENTION_VERBOSE"); return e != nullptr && e[0] == '1'; }();
    return v;
}

bool check(cudnnStatus_t st, const char* what)
{
    if (st == CUDNN_STATUS_SUCCESS) return true;
    if (verbose())
    {
        char msg[1024] = {};
        cudnnGetLastErrorString(msg, sizeof(msg));
        std::fprintf(stderr, "[cudnn-sdpa] %s: %s %s\n", what, cudnnGetErrorString(st), msg);
    }
    return false;
}

enum : int64_t { kUidQ = 1, kUidK = 2, kUidV = 3, kUidO = 4, kUidScale = 5 };

bool make_tensor(Desc& out, int64_t uid, cudnnDataType_t type, const int64_t dims[4], const int64_t strides[4],
                 bool byValue)
{
    Desc t(CUDNN_BACKEND_TENSOR_DESCRIPTOR);
    if (t.d == nullptr) return false;
    const int64_t align = byValue ? 4 : 16;
    if (!check(cudnnBackendSetAttribute(t.d, CUDNN_ATTR_TENSOR_DATA_TYPE, CUDNN_TYPE_DATA_TYPE, 1, &type), "tensor type")) return false;
    if (!check(cudnnBackendSetAttribute(t.d, CUDNN_ATTR_TENSOR_DIMENSIONS, CUDNN_TYPE_INT64, 4, dims), "tensor dims")) return false;
    if (!check(cudnnBackendSetAttribute(t.d, CUDNN_ATTR_TENSOR_STRIDES, CUDNN_TYPE_INT64, 4, strides), "tensor strides")) return false;
    if (!check(cudnnBackendSetAttribute(t.d, CUDNN_ATTR_TENSOR_UNIQUE_ID, CUDNN_TYPE_INT64, 1, &uid), "tensor uid")) return false;
    if (!check(cudnnBackendSetAttribute(t.d, CUDNN_ATTR_TENSOR_BYTE_ALIGNMENT, CUDNN_TYPE_INT64, 1, &align), "tensor align")) return false;
    if (byValue)
    {
        const bool yes = true;
        if (!check(cudnnBackendSetAttribute(t.d, CUDNN_ATTR_TENSOR_IS_BY_VALUE, CUDNN_TYPE_BOOLEAN, 1, &yes), "tensor by value")) return false;
    }
    if (!check(cudnnBackendFinalize(t.d), "tensor finalize")) return false;
    out = std::move(t);
    return true;
}

// One finalized execution plan per (heads, sequence lengths, head size, layouts).
struct Plan
{
    Desc q, k, v, o, scale, op, graph, plan;
    int64_t workspace = 0;
    bool ok = false;
};

using PlanKey = std::tuple<int64_t, int64_t, int64_t, int64_t, int64_t, int64_t, int64_t, int64_t, int64_t, int64_t, int64_t, int64_t>;

struct SdpaState
{
    std::mutex mu;
    cudnnHandle_t handle = nullptr;
    bool probed = false, usable = false;
    std::map<PlanKey, Plan> plans;
    void* workspace = nullptr;
    std::size_t workspaceBytes = 0;
    void* scratch = nullptr;          // narrowed Q, then the F16 result
    std::size_t scratchBytes = 0;
};

SdpaState& state()
{
    static SdpaState s;
    return s;
}

bool ensure_handle(SdpaState& s)
{
    if (s.probed) return s.usable;
    s.probed = true;
    if (const char* e = std::getenv("TS_H3_CUDNN_ATTENTION"); e != nullptr && e[0] == '0') return s.usable = false;
    if (!api().loaded) return s.usable = false;
    if (cudnnCreate(&s.handle) != CUDNN_STATUS_SUCCESS) { s.handle = nullptr; return s.usable = false; }
    // cuDNN runs on the legacy default stream; the caller orders it against ggml's
    // non-blocking stream on the host (as the convolution offload does).
    cudnnSetStream(s.handle, nullptr);
    return s.usable = true;
}

void* ensure_buffer(void*& p, std::size_t& have, std::size_t need)
{
    if (need == 0) return p;
    if (p != nullptr && have >= need) return p;
    if (p != nullptr) cudaFree(p);
    p = nullptr;
    have = 0;
    if (cudaMalloc(&p, need) != cudaSuccess) { p = nullptr; return nullptr; }
    have = need;
    return p;
}

bool build_plan(SdpaState& s, Plan& p, const int64_t qd[4], const int64_t qs[4], const int64_t kd[4], const int64_t ks[4],
                const int64_t vd[4], const int64_t vs[4], const int64_t od[4], const int64_t os[4])
{
    const int64_t one[4] = { 1, 1, 1, 1 };
    if (!make_tensor(p.q, kUidQ, CUDNN_DATA_HALF, qd, qs, false)) return false;
    if (!make_tensor(p.k, kUidK, CUDNN_DATA_HALF, kd, ks, false)) return false;
    if (!make_tensor(p.v, kUidV, CUDNN_DATA_HALF, vd, vs, false)) return false;
    if (!make_tensor(p.o, kUidO, CUDNN_DATA_HALF, od, os, false)) return false;
    if (!make_tensor(p.scale, kUidScale, CUDNN_DATA_FLOAT, one, one, true)) return false;

    p.op = Desc(CUDNN_BACKEND_OPERATION_SDPA_FWD_DESCRIPTOR);
    if (p.op.d == nullptr) return false;
    if (!check(cudnnBackendSetAttribute(p.op.d, CUDNN_ATTR_OPERATION_SDPA_FWD_QDESC, CUDNN_TYPE_BACKEND_DESCRIPTOR, 1, &p.q.d), "sdpa q")) return false;
    if (!check(cudnnBackendSetAttribute(p.op.d, CUDNN_ATTR_OPERATION_SDPA_FWD_KDESC, CUDNN_TYPE_BACKEND_DESCRIPTOR, 1, &p.k.d), "sdpa k")) return false;
    if (!check(cudnnBackendSetAttribute(p.op.d, CUDNN_ATTR_OPERATION_SDPA_FWD_VDESC, CUDNN_TYPE_BACKEND_DESCRIPTOR, 1, &p.v.d), "sdpa v")) return false;
    if (!check(cudnnBackendSetAttribute(p.op.d, CUDNN_ATTR_OPERATION_SDPA_FWD_ODESC, CUDNN_TYPE_BACKEND_DESCRIPTOR, 1, &p.o.d), "sdpa o")) return false;
    if (!check(cudnnBackendSetAttribute(p.op.d, CUDNN_ATTR_OPERATION_SDPA_FWD_SCALEDESC, CUDNN_TYPE_BACKEND_DESCRIPTOR, 1, &p.scale.d), "sdpa scale")) return false;
    if (!check(cudnnBackendFinalize(p.op.d), "sdpa finalize")) return false;

    p.graph = Desc(CUDNN_BACKEND_OPERATIONGRAPH_DESCRIPTOR);
    if (p.graph.d == nullptr) return false;
    if (!check(cudnnBackendSetAttribute(p.graph.d, CUDNN_ATTR_OPERATIONGRAPH_OPS, CUDNN_TYPE_BACKEND_DESCRIPTOR, 1, &p.op.d), "graph ops")) return false;
    if (!check(cudnnBackendSetAttribute(p.graph.d, CUDNN_ATTR_OPERATIONGRAPH_HANDLE, CUDNN_TYPE_HANDLE, 1, &s.handle), "graph handle")) return false;
    if (!check(cudnnBackendFinalize(p.graph.d), "graph finalize")) return false;

    for (cudnnBackendHeurMode_t mode : { CUDNN_HEUR_MODE_A, CUDNN_HEUR_MODE_FALLBACK })
    {
        Desc heur(CUDNN_BACKEND_ENGINEHEUR_DESCRIPTOR);
        if (heur.d == nullptr) return false;
        if (!check(cudnnBackendSetAttribute(heur.d, CUDNN_ATTR_ENGINEHEUR_OPERATION_GRAPH, CUDNN_TYPE_BACKEND_DESCRIPTOR, 1, &p.graph.d), "heur graph")) continue;
        if (!check(cudnnBackendSetAttribute(heur.d, CUDNN_ATTR_ENGINEHEUR_MODE, CUDNN_TYPE_HEUR_MODE, 1, &mode), "heur mode")) continue;
        if (!check(cudnnBackendFinalize(heur.d), "heur finalize")) continue;
        int64_t count = 0;
        if (!check(cudnnBackendGetAttribute(heur.d, CUDNN_ATTR_ENGINEHEUR_RESULTS, CUDNN_TYPE_BACKEND_DESCRIPTOR, 0, &count, nullptr), "heur count")) continue;
        if (count <= 0) continue;
        std::vector<Desc> cfgs;
        std::vector<cudnnBackendDescriptor_t> raw;
        for (int64_t i = 0; i < count; i++)
        {
            cfgs.emplace_back(CUDNN_BACKEND_ENGINECFG_DESCRIPTOR);
            raw.push_back(cfgs.back().d);
        }
        int64_t got = 0;
        if (!check(cudnnBackendGetAttribute(heur.d, CUDNN_ATTR_ENGINEHEUR_RESULTS, CUDNN_TYPE_BACKEND_DESCRIPTOR, count, &got, raw.data()), "heur results")) continue;
        for (int64_t i = 0; i < got; i++)
        {
            Desc plan(CUDNN_BACKEND_EXECUTION_PLAN_DESCRIPTOR);
            if (plan.d == nullptr) continue;
            if (!check(cudnnBackendSetAttribute(plan.d, CUDNN_ATTR_EXECUTION_PLAN_HANDLE, CUDNN_TYPE_HANDLE, 1, &s.handle), "plan handle")) continue;
            if (!check(cudnnBackendSetAttribute(plan.d, CUDNN_ATTR_EXECUTION_PLAN_ENGINE_CONFIG, CUDNN_TYPE_BACKEND_DESCRIPTOR, 1, &raw[i]), "plan config")) continue;
            if (!check(cudnnBackendFinalize(plan.d), "plan finalize")) continue;
            int64_t ws = 0, n = 0;
            if (!check(cudnnBackendGetAttribute(plan.d, CUDNN_ATTR_EXECUTION_PLAN_WORKSPACE_SIZE, CUDNN_TYPE_INT64, 1, &n, &ws), "plan workspace")) continue;
            p.plan = std::move(plan);
            p.workspace = ws;
            if (verbose())
                std::fprintf(stderr, "[cudnn-sdpa] plan for H=%lld Sq=%lld Skv=%lld D=%lld: config %lld of %lld (heuristic %d), workspace %lld bytes\n",
                    (long long) qd[1], (long long) qd[2], (long long) kd[2], (long long) qd[3],
                    (long long) i, (long long) got, (int) mode, (long long) ws);
            return true;
        }
    }
    return false;
}

__global__ void narrow_f32_f16(const float* __restrict__ src, __half* __restrict__ dst, long long n)
{
    for (long long i = (long long) blockIdx.x * blockDim.x + threadIdx.x; i < n; i += (long long) gridDim.x * blockDim.x)
        dst[i] = __float2half_rn(src[i]);
}

// narrow_f32_f16 for a strided Q, element (d, s, h) at d + s*stride_s + h*stride_h
// floats, into the contiguous head-major F16 cuDNN reads. One block per (head, token).
__global__ void narrow_strided_f32_f16(const float* __restrict__ src, long long stride_s, long long stride_h,
                                       __half* __restrict__ dst, long long seq, long long rows, int head_dim)
{
    const long long r = blockIdx.x;
    if (r >= rows) return;
    const long long h = r / seq, s = r - h * seq;
    const float* __restrict__ in = src + s * stride_s + h * stride_h;
    __half* __restrict__ out = dst + r * head_dim;
    for (int d = threadIdx.x; d < head_dim; d += blockDim.x) out[d] = __float2half_rn(in[d]);
}

__global__ void widen_f16_f32(const __half* __restrict__ src, float* __restrict__ dst, long long n)
{
    for (long long i = (long long) blockIdx.x * blockDim.x + threadIdx.x; i < n; i += (long long) gridDim.x * blockDim.x)
        dst[i] = __half2float(src[i]);
}

int grid_for(long long n)
{
    const long long blocks = (n + 255) / 256;
    return (int) (blocks < 65535 ? blocks : 65535);
}

// The plan for one layout, built on first use (null when no engine takes it).
Plan* find_plan(SdpaState& s, int64_t H, int64_t Sq, int64_t Skv, int64_t D,
                int64_t k_stride_s, int64_t k_stride_h, int64_t v_stride_s, int64_t v_stride_h)
{
    const int64_t qd[4] = { 1, H, Sq, D }, qs[4] = { H * Sq * D, Sq * D, D, 1 };
    const int64_t kd[4] = { 1, H, Skv, D }, ksd[4] = { H * k_stride_h, k_stride_h, k_stride_s, 1 };
    const int64_t vd[4] = { 1, H, Skv, D }, vsd[4] = { H * v_stride_h, v_stride_h, v_stride_s, 1 };
    // The result lands in ggml's [D, H, Sq] order: heads adjacent within a token.
    const int64_t od[4] = { 1, H, Sq, D }, os[4] = { Sq * H * D, D, H * D, 1 };
    const PlanKey key { H, Sq, Skv, D, k_stride_s, k_stride_h, v_stride_s, v_stride_h, 0, 0, 0, 0 };
    auto it = s.plans.find(key);
    if (it == s.plans.end())
    {
        Plan p;
        p.ok = build_plan(s, p, qd, qs, kd, ksd, vd, vsd, od, os);
        if (!p.ok && verbose())
            std::fprintf(stderr, "[cudnn-sdpa] no engine for H=%lld Sq=%lld Skv=%lld D=%lld; ggml keeps this attention\n",
                (long long) H, (long long) Sq, (long long) Skv, (long long) D);
        it = s.plans.emplace(key, std::move(p)).first;
    }
    return it->second.ok ? &it->second : nullptr;
}

bool execute_plan(SdpaState& s, Plan& p, const void* q16, const void* k, const void* v, void* o16, float scale)
{
    void* ws = nullptr;
    if (p.workspace > 0 && (ws = ensure_buffer(s.workspace, s.workspaceBytes, (std::size_t) p.workspace)) == nullptr) return false;
    float scaleValue = scale;
    int64_t uids[5] = { kUidQ, kUidK, kUidV, kUidO, kUidScale };
    void* ptrs[5] = { const_cast<void*>(q16), const_cast<void*>(k), const_cast<void*>(v), o16, &scaleValue };
    Desc pack(CUDNN_BACKEND_VARIANT_PACK_DESCRIPTOR);
    if (pack.d == nullptr) return false;
    if (!check(cudnnBackendSetAttribute(pack.d, CUDNN_ATTR_VARIANT_PACK_UNIQUE_IDS, CUDNN_TYPE_INT64, 5, uids), "pack uids")) return false;
    if (!check(cudnnBackendSetAttribute(pack.d, CUDNN_ATTR_VARIANT_PACK_DATA_POINTERS, CUDNN_TYPE_VOID_PTR, 5, ptrs), "pack pointers")) return false;
    if (!check(cudnnBackendSetAttribute(pack.d, CUDNN_ATTR_VARIANT_PACK_WORKSPACE, CUDNN_TYPE_VOID_PTR, 1, &ws), "pack workspace")) return false;
    if (!check(cudnnBackendFinalize(pack.d), "pack finalize")) return false;
    return check(cudnnBackendExecute(s.handle, p.plan.d, pack.d), "execute");
}

} // namespace

// The fused stages around the attention (tsg_cuda_h3_fused.cu).
extern "C" bool tsg_h3_attn_prep(const float* q, int64_t q_sh, int64_t q_ss, const float* k, int64_t k_sh, int64_t k_ss,
                                 const float* v, int64_t v_sh, int64_t v_ss, const float* qb, const float* kb,
                                 const float* vb, int64_t bias_hs, const float* qn, const float* kn,
                                 const float* cosT, const float* ssinT, int rot, float eps, float vInv,
                                 int heads, int seq, int head_dim, void* q16, void* k16, void* v16);
extern "C" bool tsg_h3_widen_scaled(const void* src16, float* dst, int64_t n, float scale);
#if defined(TSG_HAVE_FLASH_ATTN)
// The same product on FlashAttention-2 (tsg_cuda_flash_attn.cu).
extern "C" bool tsg_flash_attn_h3(const void* q, const void* k, const void* v, void* o, int64_t heads, int64_t seq,
                                  int64_t head_dim, float scale);
extern "C" void tsg_flash_attn_release(void);
#endif

extern "C" {

bool tsg_cudnn_sdpa_available(void)
{
    SdpaState& s = state();
    std::lock_guard<std::mutex> lock(s.mu);
    return ensure_handle(s);
}

// Builds (and caches) the plan for an H3 attention of this shape ahead of graph
// construction, so a graph stands a fused node in for the attention only when
// cuDNN has an engine for it. True when it does.
bool tsg_cudnn_sdpa_prepare(int64_t heads, int64_t seq, int64_t head_dim)
{
    SdpaState& s = state();
    std::lock_guard<std::mutex> lock(s.mu);
    if (!ensure_handle(s)) return false;
    return find_plan(s, heads, seq, seq, head_dim, head_dim, seq * head_dim, head_dim, seq * head_dim) != nullptr;
}

// O = softmax(scale * Q K^T) V over one batch, without a mask.
//   q: F32, contiguous, element (d, s, h) at d + s*D + h*D*Sq   (ggml [D, Sq, H])
//   k, v: F16, element (d, s, h) at d*1 + s*kv_s + h*kv_h      (strides in elements)
//   out: F32, contiguous, element (d, h, s) at d + h*D + s*D*H  (ggml [D, H, Sq])
// Returns false (having touched nothing the caller relies on) when cuDNN or an
// engine for the shape is unavailable; the caller then runs the node on ggml.
// Synchronizes the default stream before returning.
bool tsg_cudnn_sdpa_f32_strided(const float* q, int64_t q_stride_s, int64_t q_stride_h, const void* k, const void* v,
                                float* out, int64_t heads, int64_t seq_q, int64_t seq_kv, int64_t head_dim,
                                int64_t k_stride_s, int64_t k_stride_h, int64_t v_stride_s, int64_t v_stride_h,
                                float scale);

bool tsg_cudnn_sdpa_f32(const float* q, const void* k, const void* v, float* out,
                        int64_t heads, int64_t seq_q, int64_t seq_kv, int64_t head_dim,
                        int64_t k_stride_s, int64_t k_stride_h, int64_t v_stride_s, int64_t v_stride_h,
                        float scale)
{
    return tsg_cudnn_sdpa_f32_strided(q, head_dim, seq_q * head_dim, k, v, out, heads, seq_q, seq_kv, head_dim,
                                      k_stride_s, k_stride_h, v_stride_s, v_stride_h, scale);
}

// tsg_cudnn_sdpa_f32 for a Q that need not be contiguous: element (d, s, h) at
// d + s*q_stride_s + h*q_stride_h floats (a permuted view of [D, H, Sq], say).
bool tsg_cudnn_sdpa_f32_strided(const float* q, int64_t q_stride_s, int64_t q_stride_h, const void* k, const void* v,
                                float* out, int64_t heads, int64_t seq_q, int64_t seq_kv, int64_t head_dim,
                                int64_t k_stride_s, int64_t k_stride_h, int64_t v_stride_s, int64_t v_stride_h,
                                float scale)
{
    SdpaState& s = state();
    std::lock_guard<std::mutex> lock(s.mu);
    if (!ensure_handle(s)) return false;
    Plan* p = find_plan(s, heads, seq_q, seq_kv, head_dim, k_stride_s, k_stride_h, v_stride_s, v_stride_h);
    if (p == nullptr) return false;

    const long long nq = (long long) heads * seq_q * head_dim;
    const std::size_t pad = ((std::size_t) nq * sizeof(__half) + 255) / 256 * 256;
    // Q's narrow copy and the F16 result are both live during the call.
    if (ensure_buffer(s.scratch, s.scratchBytes, 2 * pad) == nullptr) return false;
    __half* q16 = static_cast<__half*>(s.scratch);
    __half* o16 = reinterpret_cast<__half*>(static_cast<char*>(s.scratch) + pad);

    if (q_stride_s == head_dim && q_stride_h == seq_q * head_dim)
        narrow_f32_f16<<<grid_for(nq), 256>>>(q, q16, nq);
    else
    {
        if (heads * seq_q > 0x7fffffffLL || head_dim > 1024) return false;
        narrow_strided_f32_f16<<<(unsigned int) (heads * seq_q), (unsigned int) head_dim>>>(
            q, q_stride_s, q_stride_h, q16, seq_q, heads * seq_q, (int) head_dim);
    }
    if (cudaGetLastError() != cudaSuccess) return false;
    if (!execute_plan(s, *p, q16, k, v, o16, scale)) return false;
    widen_f16_f32<<<grid_for(nq), 256>>>(o16, out, nq);
    if (cudaGetLastError() != cudaSuccess) return false;
    return cudaStreamSynchronize(nullptr) == cudaSuccess;
}

// MiniMax-H3's whole DiT attention after the qkv projection: the qkv bias (qb/kb/vb,
// each nullable; head h's slice at h * bias_hs, 0 meaning D), q/k RMSNorm, RoPE and
// the F16 narrowing in one pass (tsg_h3_attn_prep), cuDNN SDPA, and the result
// widened to F32 [D*H, S] times the power-of-two V scale. q/k/v are F32 [D, H, S]
// views with head and token strides in floats. Synchronizes.
//
// A build with FlashAttention (TSG_HAVE_FLASH_ATTN) runs the product on its kernel where
// it has one for the shape, and on cuDNN's engine otherwise. The plan is still required,
// so the graph's choice of this node does not depend on which of the two runs.
bool tsg_cudnn_h3_attention(const float* q, int64_t q_sh, int64_t q_ss, const float* k, int64_t k_sh, int64_t k_ss,
                            const float* v, int64_t v_sh, int64_t v_ss, const float* qb, const float* kb,
                            const float* vb, int64_t bias_hs, const float* qn, const float* kn,
                            const float* cosT, const float* ssinT, int rot, float eps, float vScale, float scale,
                            int heads, int seq, int head_dim, float* out)
{
    SdpaState& s = state();
    std::lock_guard<std::mutex> lock(s.mu);
    if (!ensure_handle(s)) return false;
    const int64_t D = head_dim, S = seq;
    Plan* p = find_plan(s, heads, S, S, D, D, S * D, D, S * D);
    if (p == nullptr) return false;
    const long long n = (long long) heads * S * D;
    const std::size_t pad = ((std::size_t) n * sizeof(__half) + 255) / 256 * 256;
    if (ensure_buffer(s.scratch, s.scratchBytes, 4 * pad) == nullptr) return false;
    char* base = static_cast<char*>(s.scratch);
    void* q16 = base;
    void* k16 = base + pad;
    void* v16 = base + 2 * pad;
    void* o16 = base + 3 * pad;
    tsg_stage::Clock clock;
    if (!tsg_h3_attn_prep(q, q_sh, q_ss, k, k_sh, k_ss, v, v_sh, v_ss, qb, kb, vb, bias_hs, qn, kn, cosT, ssinT, rot, eps,
                          1.0f / vScale, heads, seq, head_dim, q16, k16, v16))
        return false;
    clock.lap("attention prep");
    bool flash = false;
#if defined(TSG_HAVE_FLASH_ATTN)
    flash = tsg_flash_attn_h3(q16, k16, v16, o16, heads, S, D, scale);
#endif
    if (!flash && !execute_plan(s, *p, q16, k16, v16, o16, scale)) return false;
    clock.lap(flash ? "attention flash" : "attention sdpa");
    if (!tsg_h3_widen_scaled(o16, out, n, vScale)) return false;
    const bool ok = cudaStreamSynchronize(nullptr) == cudaSuccess;
    clock.lap("attention widen");
    return ok;
}

// Frees the plans, scratch and handle; the next call re-creates them. Called with
// the convolution release at the pipeline's stage boundaries.
void tsg_cudnn_sdpa_release(void)
{
    SdpaState& s = state();
    std::lock_guard<std::mutex> lock(s.mu);
    s.plans.clear();
    if (s.scratch != nullptr) { cudaFree(s.scratch); s.scratch = nullptr; s.scratchBytes = 0; }
    if (s.workspace != nullptr) { cudaFree(s.workspace); s.workspace = nullptr; s.workspaceBytes = 0; }
    if (s.handle != nullptr && api().loaded) cudnnDestroy(s.handle);
    s.handle = nullptr;
    s.probed = false;
    s.usable = false;
#if defined(TSG_HAVE_FLASH_ATTN)
    tsg_flash_attn_release();
#endif
}

} // extern "C"

#endif // TSG_GGML_USE_CUDA && TSG_HAVE_CUDNN
