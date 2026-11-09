// Copyright (c) Zhongkai Fu. All rights reserved.
// Licensed under the BSD-3-Clause license in the repository root.
//
// See tsm_moe.h. Uses the unchanged ggml-metal device/buffer helpers at the build revision (its MTLDevice, its command
// queue, a tensor's MTLBuffer); the kernels are Strata's grouped expert kernels (tsm/kernels, MIT) and the plan, the
// dispatch geometry and the glue are TensorSharp's.
#include "tsm_moe.h"
#include "../ggml_ops_internal.h"
#include "ggml-backend-impl.h"
#include "ggml-metal.h"
#include "ggml-metal/ggml-metal-device.h"
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstring>
#include <thread>
#include <map>
#include <mutex>
#include <stdexcept>
#include <string>
#include <vector>

#include "tsm_moe_source.inc"
#include "tsm_mpp_source.inc"
#include "tsm_moe_mm_source.inc"

namespace
{
    // The plan layout of moe.metal (P_* there): one int buffer.
    constexpr int P_NG = 0, P_IGU = 4, P_IDN = 8, P_MAXE = 128, P_GEXP = 16, P_GSTART = P_GEXP + P_MAXE,
                  P_ETOK = P_GSTART + P_MAXE + 4, P_EW = P_ETOK + P_MAXE, P_TOKENT = P_EW + P_MAXE,
                  P_INTS = P_TOKENT + P_MAXE;
    constexpr int kMaxRows = 8;   // a decode step or a verify window

    // moe.metal's MoeArgs, byte for byte.
    struct MoeArgs
    {
        std::uint64_t gate_off, up_off, down_off, gu_stride, d_stride;
        std::uint32_t gu_row, d_row, n_embd, n_ff, n_tok, topk, n_expert, gx_gu, gx_dn, pad;
    };
    static_assert(sizeof(MoeArgs) == 80, "MoeArgs must match moe.metal");

    // ggml types (= quant.metal's T_*) the instantiated moe_gu / moe_down take.
    bool gate_up_type(int t)
    {
        switch (t) { case 42: case 16: case 17: case 18: case 21: case 22: case 23: case 12: case 13: case 14: case 8: return true; }
        return false;
    }
    bool down_type(int t) { switch (t) { case 42: case 7: case 8: case 6: case 20: return true; } return false; }
    // xprep's plane for a gate/up format (units.metal XKind): 0 raw, 1 Q2_0, 2 Q4_K/Q5_K, 3 Q6_K/Q3_K.
    int x_kind(int t) { return t == 42 ? 1 : (t == 12 || t == 13) ? 2 : (t == 14 || t == 11) ? 3 : 0; }
    std::size_t row_bytes(int type, int n) { return ggml_row_size(static_cast<ggml_type>(type), n); }

    struct State
    {
        std::mutex mu;
        id<MTLDevice> device = nil;
        id<MTLLibrary> library = nil;
        std::map<std::string, id<MTLComputePipelineState>> pipelines;
        id<MTLBuffer> xp = nil, h = nil, ye = nil, plan = nil, ent = nil;
        id<MTLCommandBuffer> last = nil;   // the last committed (for the probe's wait)
        id<MTLSharedEvent> event = nil;    // the seams' handshakes (seams_*)
        std::uint64_t event_value = 0;     // the last value a pass used
        std::size_t xp_bytes = 0, h_bytes = 0, ye_bytes = 0;
        std::vector<int> plan_host;
        // gdn_prompt_encode's scratch: normalized q / k, conv'd v, gate+beta, k.q, the scan's output, the staged tail
        id<MTLBuffer> gqn = nil, gkn = nil, gvc = nil, ggb = nil, gkq = nil, go = nil, ghist = nil;
        std::size_t gqn_bytes = 0, gkn_bytes = 0, gvc_bytes = 0, ggb_bytes = 0, gkq_bytes = 0, go_bytes = 0, ghist_bytes = 0;
        // qsa_select_encode's scratch: the per-token block scores; qsa_attn_encode's: the queries in half
        id<MTLBuffer> qs = nil, qh = nil;
        std::size_t qs_bytes = 0, qh_bytes = 0;
        // The tensor-API library (tsm_mpp_source.inc, Metal language 4.0), compiled on first use: 0 untried, 1 ready,
        // -1 not available on this device or SDK (the simdgroup kernels run instead).
        id<MTLLibrary> library_mpp = nil;
        int mpp_state = 0;
        std::map<std::string, id<MTLComputePipelineState>> pipelines_mpp;
        // The prompt expert GEMM library (tsm_moe_mm_source.inc: quant.metal + Strata's moe_mm, language 4.0) and
        // moe_mm_encode's plan and scratch.
        id<MTLLibrary> library_mm = nil;
        int mm_state = 0;
        std::map<std::string, id<MTLComputePipelineState>> pipelines_mm;
        // hc_read_encode's scratch: xn in fp16, the down GEMM's output, silu(down) in fp16, the fp16 weights
        id<MTLBuffer> hr_xnh = nil, hr_v = nil, hr_lo = nil, hr_wd = nil, hr_wu = nil;
        std::size_t hr_xnh_bytes = 0, hr_v_bytes = 0, hr_lo_bytes = 0, hr_wd_bytes = 0, hr_wu_bytes = 0;
        id<MTLBuffer> mm_bcnt = nil, mm_boff = nil, mm_eoff = nil, mm_tiles = nil, mm_meta = nil, mm_stok = nil,
            mm_sw = nil, mm_pos = nil, mm_xs = nil, mm_hs = nil, mm_ye = nil;
        std::size_t mm_bcnt_bytes = 0, mm_boff_bytes = 0, mm_eoff_bytes = 0, mm_tiles_bytes = 0, mm_meta_bytes = 0,
            mm_stok_bytes = 0, mm_sw_bytes = 0, mm_pos_bytes = 0, mm_xs_bytes = 0, mm_hs_bytes = 0, mm_ye_bytes = 0;
    };
    State& state() { static State s; return s; }

    ggml_metal_device_t metal_device(ggml_backend_t backend)
    {
        if (backend == nullptr || !ggml_backend_is_metal(backend)) return nullptr;
        return static_cast<ggml_metal_device_t>(ggml_backend_get_device(backend)->context);
    }

    id<MTLComputePipelineState> pipeline(State& s, const std::string& name)
    {
        auto it = s.pipelines.find(name);
        if (it != s.pipelines.end()) return it->second;
        NSError* error = nil;
        MTLFunctionConstantValues* values = [MTLFunctionConstantValues new];
        bool ctab = true;   // QUANT_CTAB (23): the i-quant grids from constant memory, as Strata runs them
        [values setConstantValue:&ctab type:MTLDataTypeBool atIndex:23];
        id<MTLFunction> f = [s.library newFunctionWithName:[NSString stringWithUTF8String:name.c_str()]
                                            constantValues:values error:&error];
        if (f == nil)
            throw std::runtime_error("tsm: no kernel " + name + (error ? std::string(": ") + error.localizedDescription.UTF8String : ""));
        id<MTLComputePipelineState> p = [s.device newComputePipelineStateWithFunction:f error:&error];
        if (p == nil)
            throw std::runtime_error("tsm: pipeline " + name + ": " + (error ? error.localizedDescription.UTF8String : "?"));
        s.pipelines[name] = p;
        return p;
    }

    void ensure_library(State& s, id<MTLDevice> device)
    {
        if (s.library != nil && s.device == device) return;
        s.pipelines.clear();
        s.xp = s.h = s.ye = s.plan = s.ent = nil;
        s.event = nil;
        s.event_value = 0;
        s.xp_bytes = s.h_bytes = s.ye_bytes = 0;
        s.gqn = s.gkn = s.gvc = s.ggb = s.gkq = s.go = s.ghist = nil;
        s.gqn_bytes = s.gkn_bytes = s.gvc_bytes = s.ggb_bytes = s.gkq_bytes = s.go_bytes = s.ghist_bytes = 0;
        s.qs = s.qh = nil;
        s.qs_bytes = s.qh_bytes = 0;
        s.library_mpp = nil;
        s.mpp_state = 0;
        s.pipelines_mpp.clear();
        s.library_mm = nil;
        s.mm_state = 0;
        s.pipelines_mm.clear();
        s.mm_bcnt = s.mm_boff = s.mm_eoff = s.mm_tiles = s.mm_meta = s.mm_stok = s.mm_sw = s.mm_pos = nil;
        s.mm_xs = s.mm_hs = s.mm_ye = nil;
        s.mm_bcnt_bytes = s.mm_boff_bytes = s.mm_eoff_bytes = s.mm_tiles_bytes = s.mm_meta_bytes = 0;
        s.mm_stok_bytes = s.mm_sw_bytes = s.mm_pos_bytes = s.mm_xs_bytes = s.mm_hs_bytes = s.mm_ye_bytes = 0;
        MTLCompileOptions* options = [MTLCompileOptions new];
        // Safe math, as Strata compiles them: no reassociation, so a row's sums do not depend on the window around it.
        if (@available(macOS 15.0, iOS 18.0, *)) options.mathMode = MTLMathModeSafe;
        else options.fastMathEnabled = NO;
        NSError* error = nil;
        id<MTLLibrary> library = [device newLibraryWithSource:[NSString stringWithUTF8String:kTsmMoeSource]
                                                      options:options error:&error];
        if (library == nil)
            throw std::runtime_error(std::string("tsm: the expert kernels do not compile: ")
                + (error ? error.localizedDescription.UTF8String : "?"));
        s.device = device;
        s.library = library;
        s.plan = [device newBufferWithLength:P_INTS * sizeof(int) options:MTLResourceStorageModeShared];
        s.ent = [device newBufferWithLength:kMaxRows * 32 * sizeof(int) options:MTLResourceStorageModeShared];
        s.plan_host.assign(P_INTS, 0);
    }

    // Whether the tensor-API kernels run here: the Metal 4 family, an M5 / M6 / A19 / A20 (ggml's rule: the tensor API
    // pays from those on; TS_METAL_TENSOR_ENABLE=1 tries it on any Metal 4 device), and the library compiles.
    // TS_METAL_QSA_ATTN_MM=0 keeps the simdgroup kernels.
    bool mpp_available(State& s, id<MTLDevice> device)
    {
        if (const char* e = std::getenv("TS_METAL_QSA_ATTN_MM"); e && e[0] == '0') return false;
        if (s.mpp_state != 0) return s.mpp_state > 0;
        s.mpp_state = -1;
        const char* why = nullptr;
        NSString* name = device.name;
        const bool recent = [name containsString:@"M5"] || [name containsString:@"M6"] || [name containsString:@"A19"]
            || [name containsString:@"A20"];
        const char* force = std::getenv("TS_METAL_TENSOR_ENABLE");
        if (![device supportsFamily:(MTLGPUFamily)5002]) why = "not a Metal 4 device";
        else if (!recent && !(force && force[0] == '1')) why = "a pre-M5 device (TS_METAL_TENSOR_ENABLE=1 tries it)";
        else
        {
            MTLCompileOptions* options = [MTLCompileOptions new];
            options.languageVersion = (MTLLanguageVersion)(4 << 16);   // MTLLanguageVersion4_0, absent in older SDKs
            if (@available(macOS 15.0, iOS 18.0, *)) options.mathMode = MTLMathModeSafe;
            NSError* error = nil;
            id<MTLLibrary> library = [device newLibraryWithSource:[NSString stringWithUTF8String:kTsmMppSource]
                                                          options:options error:&error];
            if (library == nil) why = "the tensor-API kernels do not compile";
            else { s.library_mpp = library; s.mpp_state = 1; }
            if (library == nil && error != nil)
                std::fprintf(stderr, "[tsm] tensor API: %s\n", error.localizedDescription.UTF8String);
        }
        if (why != nullptr) std::fprintf(stderr, "[tsm] the tensor-API kernels do not run here: %s\n", why);
        return s.mpp_state > 0;
    }

    id<MTLComputePipelineState> pipeline_mpp(State& s, const std::string& name)
    {
        auto it = s.pipelines_mpp.find(name);
        if (it != s.pipelines_mpp.end()) return it->second;
        NSError* error = nil;
        id<MTLFunction> f = [s.library_mpp newFunctionWithName:[NSString stringWithUTF8String:name.c_str()]];
        if (f == nil) throw std::runtime_error("tsm: no tensor-API kernel " + name);
        id<MTLComputePipelineState> p = [s.device newComputePipelineStateWithFunction:f error:&error];
        if (p == nil)
            throw std::runtime_error("tsm: pipeline " + name + ": " + (error ? error.localizedDescription.UTF8String : "?"));
        s.pipelines_mpp[name] = p;
        return p;
    }

    // The prompt expert GEMM library, compiled on first use where the tensor API runs (mpp_available decides that).
    bool mm_library(State& s, id<MTLDevice> device)
    {
        if (s.mm_state != 0) return s.mm_state > 0;
        s.mm_state = -1;
        if (!mpp_available(s, device)) return false;
        MTLCompileOptions* options = [MTLCompileOptions new];
        options.languageVersion = (MTLLanguageVersion)(4 << 16);
        if (@available(macOS 15.0, iOS 18.0, *)) options.mathMode = MTLMathModeSafe;
        NSError* error = nil;
        id<MTLLibrary> library = [device newLibraryWithSource:[NSString stringWithUTF8String:kTsmMoeMmSource]
                                                      options:options error:&error];
        if (library == nil)
        {
            std::fprintf(stderr, "[tsm] the prompt expert GEMM does not compile: %s\n",
                error ? error.localizedDescription.UTF8String : "?");
            return false;
        }
        s.library_mm = library;
        s.mm_state = 1;
        return true;
    }

    id<MTLComputePipelineState> pipeline_mm(State& s, const std::string& name)
    {
        auto it = s.pipelines_mm.find(name);
        if (it != s.pipelines_mm.end()) return it->second;
        NSError* error = nil;
        MTLFunctionConstantValues* values = [MTLFunctionConstantValues new];
        bool ctab = true;   // QUANT_CTAB (23), as the expert kernels above
        [values setConstantValue:&ctab type:MTLDataTypeBool atIndex:23];
        id<MTLFunction> f = [s.library_mm newFunctionWithName:[NSString stringWithUTF8String:name.c_str()]
                                               constantValues:values error:&error];
        if (f == nil)
            throw std::runtime_error("tsm: no prompt expert kernel " + name + (error ? std::string(": ") + error.localizedDescription.UTF8String : ""));
        id<MTLComputePipelineState> p = [s.device newComputePipelineStateWithFunction:f error:&error];
        if (p == nil)
            throw std::runtime_error("tsm: pipeline " + name + ": " + (error ? error.localizedDescription.UTF8String : "?"));
        s.pipelines_mm[name] = p;
        return p;
    }

    id<MTLBuffer> grow(State& s, id<MTLBuffer> b, std::size_t& have, std::size_t need)
    {
        if (b != nil && have >= need) return b;
        have = need;
        return [s.device newBufferWithLength:need options:MTLResourceStorageModeShared];
    }

    // The MTLBuffer and offset under host-visible address `data` of a ggml-metal buffer.
    ggml_metal_buffer_id buffer_at(ggml_backend_buffer_t buffer, void* data, std::size_t bytes)
    {
        if (buffer == nullptr) throw std::runtime_error("tsm: an unbound tensor");
        ggml_tensor view{};
        view.type = GGML_TYPE_I8;
        view.ne[0] = static_cast<std::int64_t>(bytes);
        view.ne[1] = view.ne[2] = view.ne[3] = 1;
        view.nb[0] = 1; view.nb[1] = view.nb[2] = view.nb[3] = bytes;
        view.buffer = buffer;
        view.data = data;
        ggml_metal_buffer_id bid = ggml_metal_buffer_get_id(static_cast<ggml_metal_buffer_t>(buffer->context), &view);
        if (bid.metal == nullptr) throw std::runtime_error("tsm: a tensor outside every Metal buffer");
        return bid;
    }

    ggml_metal_buffer_id tensor_id(const ggml_tensor* t)
    {
        ggml_backend_buffer_t buffer = t->view_src ? t->view_src->buffer : t->buffer;
        return buffer_at(buffer, t->data, ggml_nbytes(t));
    }

    void bind(id<MTLComputeCommandEncoder> enc, const ggml_metal_buffer_id& bid, std::size_t extra, int index)
    {
        [enc setBuffer:(__bridge id<MTLBuffer>)bid.metal offset:bid.offs + extra atIndex:index];
    }

    // The tiles: Strata's for these formats (gate/up 32 lanes a row, 1 row, 4 simdgroups; down Q2_0 8 lanes, 1 row;
    // the others 4 lanes, 2 rows; 4 simdgroups).
    struct Geometry
    {
        int gt = 0, dt = 0, gl = 32, gr = 1, gs = 4, dl = 4, dr = 2, ds = 4, gu_rows = 0, dn_rows = 0;
        bool dq2 = false;
        std::string gu, dn;
    };

    Geometry geometry(const tsg::MoeHotTier& tier, int hidden, int n_ff)
    {
        Geometry g;
        g.gt = tier.type[0];
        g.dt = tier.type[2];
        g.dq2 = g.dt == 42;
        g.dl = g.dq2 ? 8 : 4;
        g.dr = g.dq2 ? 1 : 2;
        g.gu_rows = g.gs * (32 / g.gl) * g.gr;
        g.dn_rows = g.ds * (32 / g.dl) * g.dr;
        if (n_ff % g.gu_rows != 0 || hidden % g.dn_rows != 0)
            throw std::runtime_error("tsm: expert rows do not fill the kernel tiles");
        g.gu = "moe_gu_" + std::to_string(g.gt) + "_" + std::to_string(g.gl) + "_" + std::to_string(g.gr) + "_" + std::to_string(g.gs);
        g.dn = "moe_down_" + std::to_string(g.dq2 ? 142 : g.dt) + "_" + std::to_string(g.dl) + "_" + std::to_string(g.dr) + "_" + std::to_string(g.ds);
        return g;
    }

    // The plan (moe.metal's layout) into s.plan and the route -> entry map into s.ent: by slot ascending, each slot's
    // entries in route order; the grids of moe_gu / moe_down at P_IGU / P_IDN. Returns the number of slot groups.
    int write_plan(State& s, const Geometry& g, int hidden, int n_ff, int T, int n_used, const std::int32_t* slots,
                   const float* weights)
    {
        const int ne = T * n_used;
        std::vector<int>& plan = s.plan_host;
        std::fill(plan.begin(), plan.end(), 0);
        int* ent = static_cast<int*>([s.ent contents]);
        for (int i = 0; i < ne; ++i) ent[i] = -1;
        static thread_local std::vector<int> order;
        order.clear();
        for (int i = 0; i < ne; ++i) if (slots[i] >= 0) order.push_back(i);
        std::stable_sort(order.begin(), order.end(), [&](int a, int b) { return slots[a] < slots[b]; });
        int ng = 0;
        for (std::size_t j = 0; j < order.size(); ++j)
        {
            const int i = order[j];
            if (j == 0 || slots[order[j - 1]] != slots[i])
            {
                plan[P_GEXP + ng] = slots[i];
                plan[P_GSTART + ng] = static_cast<int>(j);
                ++ng;
            }
            plan[P_ETOK + j] = i / n_used;
            float w = weights[i];
            std::memcpy(&plan[P_EW + j], &w, sizeof(float));
            plan[P_TOKENT + i] = static_cast<int>(j);
            ent[i] = static_cast<int>(j);
        }
        plan[P_NG] = ng;
        plan[P_GSTART + ng] = static_cast<int>(order.size());
        plan[P_IGU] = n_ff / g.gu_rows; plan[P_IGU + 1] = ng; plan[P_IGU + 2] = 1;
        plan[P_IDN] = hidden / g.dn_rows; plan[P_IDN + 1] = ng; plan[P_IDN + 2] = 1;
        std::memcpy([s.plan contents], plan.data(), P_INTS * sizeof(int));
        return ng;
    }

    MoeArgs moe_args(const tsg::MoeHotTier& tier, const Geometry& g, int hidden, int n_ff, int T, int n_used)
    {
        MoeArgs a{};
        a.gate_off = tier.offset[0];
        a.up_off = tier.offset[1];
        a.down_off = tier.offset[2];
        a.gu_stride = tier.stride[0];
        a.d_stride = tier.stride[2];
        a.gu_row = static_cast<std::uint32_t>(row_bytes(g.gt, hidden));
        a.d_row = static_cast<std::uint32_t>(row_bytes(g.dt, n_ff));
        a.n_embd = static_cast<std::uint32_t>(hidden);
        a.n_ff = static_cast<std::uint32_t>(n_ff);
        a.n_tok = static_cast<std::uint32_t>(T);
        a.topk = static_cast<std::uint32_t>(n_used);
        a.n_expert = static_cast<std::uint32_t>(tier.count);
        a.gx_gu = static_cast<std::uint32_t>(n_ff / g.gu_rows);
        a.gx_dn = static_cast<std::uint32_t>(hidden / g.dn_rows);
        return a;
    }

    // Scratch: the prepared x (4 planes + S16), h and ye per entry.
    void ensure_scratch(State& s, int hidden, int n_ff, int T)
    {
        const std::size_t plane = static_cast<std::size_t>(T) * hidden;
        s.xp = grow(s, s.xp, s.xp_bytes, (4 * plane + plane / 16) * sizeof(float));
        s.h = grow(s, s.h, s.h_bytes, static_cast<std::size_t>(P_MAXE) * n_ff * sizeof(float));
        s.ye = grow(s, s.ye, s.ye_bytes, static_cast<std::size_t>(P_MAXE) * hidden * sizeof(float));
    }

    // xprep: one thread per 16 values of a row.
    void encode_xprep(State& s, id<MTLComputeCommandEncoder> enc, const ggml_metal_buffer_id& x_id, int hidden, int T)
    {
        id<MTLComputePipelineState> p = pipeline(s, "xprep");
        const std::uint32_t n = static_cast<std::uint32_t>(hidden), nt = static_cast<std::uint32_t>(T);
        [enc setComputePipelineState:p];
        bind(enc, x_id, 0, 0);
        [enc setBuffer:s.xp offset:0 atIndex:1];
        [enc setBytes:&n length:sizeof(n) atIndex:2];
        [enc setBytes:&nt length:sizeof(nt) atIndex:3];
        const NSUInteger threads = static_cast<NSUInteger>(hidden / 16) * T;
        [enc dispatchThreadgroups:MTLSizeMake((threads + 63) / 64, 1, 1) threadsPerThreadgroup:MTLSizeMake(64, 1, 1)];
    }

    // moe_gu and moe_down over the plan's slot groups: `ng` of them, or (ng < 0) as many as the plan says when the
    // kernels run (the grids read from P_IGU / P_IDN).
    void encode_experts(State& s, id<MTLComputeCommandEncoder> enc, const Geometry& g, const MoeArgs& a,
                        const ggml_metal_buffer_id& w_id, int hidden, int T, int ng)
    {
        const std::size_t plane = static_cast<std::size_t>(T) * hidden;
        id<MTLComputePipelineState> pgu = pipeline(s, g.gu), pdn = pipeline(s, g.dn);
        [enc setComputePipelineState:pgu];
        bind(enc, w_id, 0, 0);
        [enc setBuffer:s.xp offset:x_kind(g.gt) * plane * sizeof(float) atIndex:1];
        [enc setBuffer:s.h offset:0 atIndex:2];
        [enc setBuffer:s.plan offset:0 atIndex:3];
        [enc setBytes:&a length:sizeof(a) atIndex:4];
        [enc setBuffer:s.xp offset:4 * plane * sizeof(float) atIndex:5];
        [enc setBuffer:s.plan offset:0 atIndex:6];   // smap: unused without the arena (constant 20)
        for (int i = 7; i <= 9; ++i) bind(enc, w_id, 0, i);
        if (ng < 0)
            [enc dispatchThreadgroupsWithIndirectBuffer:s.plan indirectBufferOffset:P_IGU * sizeof(int)
                                  threadsPerThreadgroup:MTLSizeMake(32 * g.gs, 1, 1)];
        else
            [enc dispatchThreadgroups:MTLSizeMake(a.gx_gu, ng, 1) threadsPerThreadgroup:MTLSizeMake(32 * g.gs, 1, 1)];

        [enc setComputePipelineState:pdn];
        bind(enc, w_id, 0, 0);
        [enc setBuffer:s.h offset:0 atIndex:1];
        [enc setBuffer:s.ye offset:0 atIndex:2];
        [enc setBuffer:s.plan offset:0 atIndex:3];
        [enc setBytes:&a length:sizeof(a) atIndex:4];
        [enc setBuffer:s.h offset:0 atIndex:5];      // hs16: the raw-input formats read none
        [enc setBuffer:s.plan offset:0 atIndex:6];
        for (int i = 7; i <= 9; ++i) bind(enc, w_id, 0, i);
        if (ng < 0)
            [enc dispatchThreadgroupsWithIndirectBuffer:s.plan indirectBufferOffset:P_IDN * sizeof(int)
                                  threadsPerThreadgroup:MTLSizeMake(32 * g.ds, 1, 1)];
        else
            [enc dispatchThreadgroups:MTLSizeMake(a.gx_dn, ng, 1) threadsPerThreadgroup:MTLSizeMake(32 * g.ds, 1, 1)];
    }

    void encode_combine(State& s, id<MTLComputeCommandEncoder> enc, const ggml_metal_buffer_id& out_id, int hidden,
                        int T, int n_used)
    {
        id<MTLComputePipelineState> p = pipeline(s, "tsm_moe_combine_held");
        const std::uint32_t args[4] = {static_cast<std::uint32_t>(hidden), static_cast<std::uint32_t>(T),
                                       static_cast<std::uint32_t>(n_used), 0};
        [enc setComputePipelineState:p];
        [enc setBuffer:s.ye offset:0 atIndex:0];
        [enc setBuffer:s.ent offset:0 atIndex:1];
        bind(enc, out_id, 0, 2);
        [enc setBytes:args length:sizeof(args) atIndex:3];
        const NSUInteger threads = static_cast<NSUInteger>(hidden) * T;
        [enc dispatchThreadgroups:MTLSizeMake((threads + 255) / 256, 1, 1) threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
    }

    ggml_metal_buffer_id tier_id(const tsg::MoeHotTier& tier)
    {
        return buffer_at(tier.buffer, ggml_backend_buffer_get_base(tier.buffer), ggml_backend_buffer_get_size(tier.buffer));
    }

    void add_timing(id<MTLCommandBuffer> cb, int rows, int groups)
    {
        static const bool timing = [] { const char* e = std::getenv("TS_METAL_MOE_TIMING"); return e && e[0] == '1'; }();
        if (!timing) return;
        [cb addCompletedHandler:^(id<MTLCommandBuffer> done) {
            static std::mutex m;
            static double sum = 0; static int n = 0;
            std::lock_guard<std::mutex> lk(m);
            sum += (done.GPUEndTime - done.GPUStartTime) * 1e3;
            if (++n % 96 == 0)
            {
                std::fprintf(stderr, "[tsm] grouped experts: %.3f ms GPU a layer (last: %d rows, %d slots)\n", sum / 96, rows, groups);
                sum = 0;
            }
        }];
    }

    void report_once(int gt, int dt)
    {
        static bool reported = false;
        if (reported) return;
        reported = true;
        std::fprintf(stderr, "[tsm] tiered MoE seams compute their held routes with the grouped expert kernels "
            "(gate/up %d, down %d; TS_METAL_MOE=0 keeps MUL_MAT_ID)\n", gt, dt);
    }
}

namespace tsm
{
    // Read per graph build (a test process toggles it).
    bool moe_enabled()
    {
        const char* e = std::getenv("TS_METAL_MOE");
        return !(e && e[0] == '0');
    }

    bool moe_supported(ggml_backend* backend, const tsg::MoeHotTier& tier, int hidden, int n_ff, int T, int n_used)
    {
        if (!moe_enabled() || metal_device(backend) == nullptr || tier.buffer == nullptr) return false;
        if (T < 1 || T > kMaxRows || n_used < 1 || n_used > 32 || T * n_used > P_MAXE) return false;
        const int gt = tier.type[0], dt = tier.type[2];
        if (tier.type[1] != gt || !gate_up_type(gt) || !down_type(dt)) return false;
        // Every unit of a row whole, and the tiles below covering the rows exactly.
        if (hidden % 256 != 0 || n_ff % 64 != 0) return false;
        if (n_ff % 4 != 0 || hidden % 32 != 0) return false;
        // moe.metal addresses up at gate's per-expert stride.
        if (tier.stride[0] != tier.stride[1] || tier.stride[0] != row_bytes(gt, hidden) * n_ff
            || tier.stride[2] != row_bytes(dt, n_ff) * hidden) return false;
        return true;
    }

    bool moe_encode(ggml_backend* backend, const tsg::MoeHotTier& tier, const ggml_tensor* moe_in, ggml_tensor* out,
                    int hidden, int n_ff, int T, int n_used, const std::int32_t* slots, const float* weights)
    {
        try
        {
            ggml_metal_device_t dev = metal_device(backend);
            if (dev == nullptr) throw std::runtime_error("tsm: not a Metal backend");
            State& s = state();
            std::lock_guard<std::mutex> lock(s.mu);
            @autoreleasepool
            {
                id<MTLDevice> device = (__bridge id<MTLDevice>)ggml_metal_device_get_obj(dev);
                id<MTLCommandQueue> queue = (__bridge id<MTLCommandQueue>)ggml_metal_device_get_queue(dev);
                ensure_library(s, device);
                const Geometry g = geometry(tier, hidden, n_ff);
                const int ng = write_plan(s, g, hidden, n_ff, T, n_used, slots, weights);
                const MoeArgs a = moe_args(tier, g, hidden, n_ff, T, n_used);
                ensure_scratch(s, hidden, n_ff, T);
                const ggml_metal_buffer_id x_id = tensor_id(moe_in), out_id = tensor_id(out), w_id = tier_id(tier);

                id<MTLCommandBuffer> cb = [queue commandBuffer];
                id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
                encode_xprep(s, enc, x_id, hidden, T);
                if (ng > 0) encode_experts(s, enc, g, a, w_id, hidden, T, ng);
                encode_combine(s, enc, out_id, hidden, T, n_used);
                [enc endEncoding];
                add_timing(cb, T, ng);
                [cb commit];
                s.last = cb;
                report_once(g.gt, g.dt);
            }
            return true;
        }
        catch (const std::exception& e)
        {
            tsg::set_last_error(e.what());
            return false;
        }
    }

    // ---- seams on a shared event (tsm_moe.h)
    namespace
    {
        struct SeamPass
        {
            bool active = false;
            int n_seams = 0;
            int current = -1;         // the seam whose host turn is running (seams_wait set it)
            std::uint64_t base = 0;   // seam k: A = base + 2k + 1 (the GPU's), B = base + 2k + 2 (the host's)
            id<MTLSharedEvent> event = nil;
            std::vector<id<MTLCommandBuffer>> cbs;
            std::vector<Geometry> geo;
            // the current seam's plan, staged by the route split and written when the host releases the seam
            std::vector<std::int32_t> slots;
            std::vector<float> weights;
            int hidden = 0, n_ff = 0, T = 0, n_used = 0;
            bool staged = false;
        };
        thread_local SeamPass t_pass;
        std::atomic<long long> g_seam_passes{0};   // passes that ran to the end (TSGgml_TsmSeamPasses)

        // TS_METAL_SEAM_PROFILE=1: on the GPU's clock, per pass, the time between one seam's command buffer and the
        // next (the ggml segment between them, its command buffer's start included) and the seams' own (their kernels
        // and the waits for the host's B and C), summed over 32 passes of one width.
        bool seam_profile()
        {
            static const bool on = [] { const char* e = std::getenv("TS_METAL_SEAM_PROFILE"); return e && e[0] == '1'; }();
            return on;
        }
        // One accumulator per width (a speculative session changes width nearly every pass, so a shared one that
        // restarted on every change never printed).
        struct SeamWidthAcc { double segments = 0, seams = 0; int n = 0; };
        struct SeamProfile
        {
            std::mutex m;
            std::map<std::uint64_t, std::vector<std::pair<double, double>>> passes;   // by pass base: (start, end) a seam
            std::map<int, SeamWidthAcc> widths;
        };
        SeamProfile& seam_prof() { static SeamProfile p; return p; }
        void seam_profile_note(std::uint64_t pass, int seam, int n_seams, int rows, double start, double end)
        {
            SeamProfile& p = seam_prof();
            std::lock_guard<std::mutex> lk(p.m);
            auto& v = p.passes[pass];
            if (v.empty()) v.assign(n_seams, {0.0, 0.0});
            v[seam] = {start, end};
            for (const auto& x : v) if (x.second == 0.0) return;
            SeamWidthAcc& a = p.widths[rows];
            for (int k = 0; k < n_seams; ++k)
            {
                a.seams += v[k].second - v[k].first;
                if (k > 0) a.segments += v[k].first - v[k - 1].second;
            }
            p.passes.erase(pass);
            if (++a.n == 32)
            {
                std::fprintf(stderr, "[tsm] seamed pass on the GPU (%d rows, %d seams): segments 1..%d %.2f ms, seams %.2f ms "
                    "(their kernels and the wait for the host)\n", rows, n_seams, n_seams - 1, a.segments * 1e3 / 32,
                    a.seams * 1e3 / 32);
                a = SeamWidthAcc{};
            }
        }
    }

    // ---- the keep-alive (Strata's, metal/src/mtl_rt.mm start_keepalive): while seamed passes run, short spin kernels
    // on a second queue keep the GPU awake between a seam's signal and the host's release. On an M5 Pro a handshake
    // [signal | wait | release] costs ~190 us of GPU time on a quiet GPU and ~83 us beside the spin (p90 880 -> 475 us);
    // a pass has 48. Bounded kernels, re-committed by a thread, two in flight: a GPU spin cannot be stopped by a flag
    // the CPU writes (the GPU does not see it). TS_METAL_KEEPALIVE=0 turns it off.
    namespace
    {
        std::atomic<std::int64_t> g_ka_last_ns{0};
        std::atomic<bool> g_ka_started{false};
        std::int64_t ka_now_ns()
        {
            return std::chrono::duration_cast<std::chrono::nanoseconds>(
                std::chrono::steady_clock::now().time_since_epoch()).count();
        }
        bool keepalive_enabled()
        {
            const char* e = std::getenv("TS_METAL_KEEPALIVE");
            return !(e && e[0] == '0');
        }
        void keepalive_touch(id<MTLDevice> device)
        {
            g_ka_last_ns.store(ka_now_ns(), std::memory_order_relaxed);
            if (g_ka_started.exchange(true)) return;
            std::thread([device]() {
                @autoreleasepool
                {
                    static const char* kSpin = R"MSL(
#include <metal_stdlib>
using namespace metal;
kernel void tsm_keepalive(device atomic_uint* c [[buffer(0)]], constant uint& n [[buffer(1)]], uint t [[thread_position_in_grid]]) {
    uint x = t;
    for (uint i = 0; i < n; ++i) x = x * 1664525u + 1013904223u;
    if (x == 0xdeadbeefu) atomic_fetch_add_explicit(c, 1u, memory_order_relaxed);
}
)MSL";
                    NSError* err = nil;
                    id<MTLLibrary> lib = [device newLibraryWithSource:[NSString stringWithUTF8String:kSpin] options:nil error:&err];
                    id<MTLComputePipelineState> ps = lib ? [device newComputePipelineStateWithFunction:
                        [lib newFunctionWithName:@"tsm_keepalive"] error:&err] : nil;
                    if (ps == nil) return;
                    id<MTLCommandQueue> q = [device newCommandQueue];
                    id<MTLBuffer> sink = [device newBufferWithLength:64 options:MTLResourceStorageModeShared];
                    std::uint32_t iters = 20000;   // retuned toward ~150 us a kernel from the measured GPU time
                    id<MTLCommandBuffer> prev = nil;
                    for (;;)
                    {
                        if (ka_now_ns() - g_ka_last_ns.load(std::memory_order_relaxed) > 20'000'000LL)
                        {
                            if (prev != nil) { [prev waitUntilCompleted]; prev = nil; }
                            std::this_thread::sleep_for(std::chrono::milliseconds(1));
                            continue;
                        }
                        @autoreleasepool
                        {
                            id<MTLCommandBuffer> cb = [q commandBuffer];
                            id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
                            [enc setComputePipelineState:ps];
                            [enc setBuffer:sink offset:0 atIndex:0];
                            [enc setBytes:&iters length:sizeof(iters) atIndex:1];
                            [enc dispatchThreadgroups:MTLSizeMake(1, 1, 1) threadsPerThreadgroup:MTLSizeMake(32, 1, 1)];
                            [enc endEncoding];
                            [cb commit];
                            if (prev != nil)
                            {
                                [prev waitUntilCompleted];
                                const double us = (prev.GPUEndTime - prev.GPUStartTime) * 1e6;
                                if (us > 1.0)
                                    iters = static_cast<std::uint32_t>(std::clamp(iters * (150.0 / us), 1000.0, 2'000'000.0));
                            }
                            prev = cb;
                        }
                    }
                }
            }).detach();
        }
    }

    bool seams_enabled(ggml_backend* backend)
    {
        // Read per pass (a test process compares both).
        const char* e = std::getenv("TS_METAL_SEAM_EVENTS");
        return !(e && e[0] == '0') && moe_enabled() && metal_device(backend) != nullptr;
    }

    bool seams_host_visible(const ggml_tensor* t)
    {
        if (t == nullptr) return false;
        ggml_backend_buffer_t buffer = t->view_src ? t->view_src->buffer : t->buffer;
        // ggml-metal's buffers (shared, private, mapped) are its device's; their context is a ggml_metal_buffer_t.
        if (buffer == nullptr || buffer->buft == nullptr || g_backend == nullptr
            || !ggml_backend_is_metal(g_backend)
            || ggml_backend_buft_get_device(buffer->buft) != ggml_backend_get_device(g_backend)) return false;
        return ggml_metal_buffer_is_shared(static_cast<ggml_metal_buffer_t>(buffer->context));
    }

    bool seams_begin(ggml_backend* backend, int n_seams)
    {
        try
        {
            ggml_metal_device_t dev = metal_device(backend);
            if (dev == nullptr) throw std::runtime_error("tsm: not a Metal backend");
            State& s = state();
            std::lock_guard<std::mutex> lock(s.mu);
            id<MTLDevice> device = (__bridge id<MTLDevice>)ggml_metal_device_get_obj(dev);
            ensure_library(s, device);
            if (s.event == nil)
            {
                s.event = [device newSharedEvent];
                if (s.event == nil) throw std::runtime_error("tsm: no shared event");
                s.event_value = s.event.signaledValue;
            }
            if (keepalive_enabled()) keepalive_touch(device);
            SeamPass& p = t_pass;
            p.active = true;
            p.n_seams = n_seams;
            p.base = s.event_value + 4;
            s.event_value = p.base + 2 * static_cast<std::uint64_t>(n_seams) + 4;
            p.event = s.event;
            p.current = -1;
            p.cbs.assign(n_seams, nil);
            p.geo.assign(n_seams, Geometry{});
            static bool reported = false;
            if (!reported)
            {
                reported = true;
                std::fprintf(stderr, "[tsm] tiered MoE seams hand over on a shared event: segments are enqueued ahead "
                    "(TS_METAL_SEAM_EVENTS=0 keeps a submit and a wait per seam)\n");
            }
            return true;
        }
        catch (const std::exception& e)
        {
            tsg::set_last_error(e.what());
            return false;
        }
    }

    bool seams_active() { return t_pass.active; }

    bool seams_enqueue(int seam, const tsg::MoeHotTier& tier, const ggml_tensor* moe_in, const ggml_tensor* sel_ids,
                       const ggml_tensor* weights, ggml_tensor* out, int hidden, int n_ff, int T, int n_used)
    {
        SeamPass& p = t_pass;
        try
        {
            if (!p.active || seam < 0 || seam >= p.n_seams) throw std::runtime_error("tsm: a seam outside the pass");
            ggml_metal_device_t dev = metal_device(tier.backend != nullptr ? tier.backend : g_backend);
            if (dev == nullptr) throw std::runtime_error("tsm: not a Metal backend");
            State& s = state();
            std::lock_guard<std::mutex> lock(s.mu);
            @autoreleasepool
            {
                id<MTLCommandQueue> queue = (__bridge id<MTLCommandQueue>)ggml_metal_device_get_queue(dev);
                const Geometry g = geometry(tier, hidden, n_ff);
                p.geo[seam] = g;
                const MoeArgs a = moe_args(tier, g, hidden, n_ff, T, n_used);
                ensure_scratch(s, hidden, n_ff, T);
                const ggml_metal_buffer_id x_id = tensor_id(moe_in), out_id = tensor_id(out), w_id = tier_id(tier);
                const ggml_metal_buffer_id ids_id = tensor_id(sel_ids), wts_id = tensor_id(weights);
                const std::uint64_t A = p.base + 2 * static_cast<std::uint64_t>(seam) + 1;

                id<MTLCommandBuffer> cb = [queue commandBuffer];
                // xprep reads moe_in, and declares the routes the host reads: hazard tracking runs it after the segment
                // that wrote them, so A means "the segment is done" (as in one command buffer).
                id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
                encode_xprep(s, enc, x_id, hidden, T);
                [enc useResource:(__bridge id<MTLBuffer>)ids_id.metal usage:MTLResourceUsageRead];
                [enc useResource:(__bridge id<MTLBuffer>)wts_id.metal usage:MTLResourceUsageRead];
                [enc endEncoding];
                [cb encodeSignalEvent:p.event value:A];
                // One wait: the host releases the seam with its plan written and its own routes already in moe_out.
                // (A second wait for those, so they could overlap the kernels, cost more than it hid: each wait is a
                // 0.1-0.15 ms round trip on an M5 Pro, and at one row the host has no routes of its own.)
                [cb encodeWaitForEvent:p.event value:A + 1];
                enc = [cb computeCommandEncoder];
                encode_experts(s, enc, g, a, w_id, hidden, T, -1);
                // the combine writes `out`, which the next segment reads: the segment runs after it
                encode_combine(s, enc, out_id, hidden, T, n_used);
                [enc endEncoding];
                add_timing(cb, T, -1);
                if (seam_profile())
                {
                    const std::uint64_t pass = p.base;
                    const int n_seams = p.n_seams, rows = T;
                    [cb addCompletedHandler:^(id<MTLCommandBuffer> done) {
                        seam_profile_note(pass, seam, n_seams, rows, done.GPUStartTime, done.GPUEndTime);
                    }];
                }
                [cb commit];
                p.cbs[seam] = cb;
                s.last = cb;
                report_once(g.gt, g.dt);
            }
            return true;
        }
        catch (const std::exception& e)
        {
            tsg::set_last_error(e.what());
            return false;
        }
    }

    bool seams_wait(int seam)
    {
        SeamPass& p = t_pass;
        if (!p.active || seam < 0 || seam >= p.n_seams || p.cbs[seam] == nil)
        {
            tsg::set_last_error("tsm: waiting on a seam that was not enqueued");
            return false;
        }
        const std::uint64_t want = p.base + 2 * static_cast<std::uint64_t>(seam) + 1;
        id<MTLCommandBuffer> cb = p.cbs[seam];
        // Spin (a segment is ~0.5 ms; a sleep would cost more than it saves), yielding now and then, and watch the
        // command buffer: one that failed never signals.
        const auto t0 = std::chrono::steady_clock::now();
        for (std::uint64_t n = 0; p.event.signaledValue < want; ++n)
        {
            if ((n & 255) == 255)
            {
                const MTLCommandBufferStatus st = cb.status;
                if (st == MTLCommandBufferStatusError)
                {
                    tsg::set_last_error(std::string("tsm: a seam's command buffer failed: ")
                        + (cb.error ? cb.error.localizedDescription.UTF8String : "?"));
                    return false;
                }
                if (std::chrono::steady_clock::now() - t0 > std::chrono::seconds(60))
                {
                    tsg::set_last_error("tsm: a seam waited 60 s for the accelerator");
                    return false;
                }
                std::this_thread::yield();
            }
        }
        p.current = seam;
        g_ka_last_ns.store(ka_now_ns(), std::memory_order_relaxed);
        return true;
    }

    void seams_write_plan(int hidden, int n_ff, int T, int n_used, const std::int32_t* slots, const float* weights)
    {
        SeamPass& p = t_pass;
        const std::size_t routes = static_cast<std::size_t>(T) * n_used;
        p.slots.assign(slots, slots + routes);
        p.weights.assign(weights, weights + routes);
        p.hidden = hidden; p.n_ff = n_ff; p.T = T; p.n_used = n_used;
        p.staged = true;
    }

    void seams_release_out()
    {
        SeamPass& p = t_pass;
        const int seam = p.current;
        if (p.staged)
        {
            State& s = state();
            std::lock_guard<std::mutex> lock(s.mu);
            // Nothing reads the plan now: the previous seam's kernels ran before this seam's segment.
            write_plan(s, p.geo[seam], p.hidden, p.n_ff, p.T, p.n_used, p.slots.data(), p.weights.data());
        }
        else
        {
            // No split ran (it failed): an empty plan, nothing held.
            State& s = state();
            std::lock_guard<std::mutex> lock(s.mu);
            std::vector<std::int32_t> none(static_cast<std::size_t>(std::max(1, p.T * p.n_used)), -1);
            std::vector<float> zero(none.size(), 0.0f);
            write_plan(s, p.geo[seam], p.hidden, p.n_ff, std::max(1, p.T), std::max(1, p.n_used), none.data(), zero.data());
        }
        p.staged = false;
        p.event.signaledValue = p.base + 2 * static_cast<std::uint64_t>(seam) + 2;
    }

    long long seams_pass_count() { return g_seam_passes.load(std::memory_order_relaxed); }

    void seams_end(bool failed)
    {
        SeamPass& p = t_pass;
        if (!p.active) return;
        if (failed && p.event != nil)
            p.event.signaledValue = p.base + 2 * static_cast<std::uint64_t>(p.n_seams) + 2;   // every wait released
        else
            g_seam_passes.fetch_add(1, std::memory_order_relaxed);
        p.active = false;
        p.staged = false;
        p.cbs.clear();
        p.event = nil;
    }

    // ---- prompt GDN (tsm_moe.h GdnPromptIO)
    namespace
    {
        struct TsmGdnArgs { std::uint32_t n_tok, h_k, h_v, channels; float eps; std::uint32_t pad; };
        static_assert(sizeof(TsmGdnArgs) == 24, "TsmGdnArgs must match tsm_gdn_prompt.metal");
        std::atomic<long long> g_gdn_prompt_count{0};
    }

    bool gdn_prompt_supported(ggml_backend* backend, int head_k_dim, int head_v_dim, int h_k, int h_v, int d_conv)
    {
        const char* e = std::getenv("TS_METAL_GDN_PROMPT");
        if (e && e[0] == '0') return false;
        return metal_device(backend) != nullptr && head_k_dim == 128 && head_v_dim == 128 && d_conv == 4
            && h_k > 0 && h_v > 0 && h_v % 2 == 0 && h_v % h_k == 0;
    }

    bool gdn_prompt_encode(ggml_backend* backend, const GdnPromptIO& io)
    {
        try
        {
            ggml_metal_device_t dev = metal_device(backend);
            if (dev == nullptr) throw std::runtime_error("tsm gdn: not a Metal backend");
            if (io.T <= 0 || io.channels != 2 * io.h_k * 128 + io.h_v * 128)
                throw std::runtime_error("tsm gdn: shapes do not match the kernels");
            State& s = state();
            std::lock_guard<std::mutex> lock(s.mu);
            @autoreleasepool
            {
                id<MTLDevice> device = (__bridge id<MTLDevice>)ggml_metal_device_get_obj(dev);
                id<MTLCommandQueue> queue = (__bridge id<MTLCommandQueue>)ggml_metal_device_get_queue(dev);
                ensure_library(s, device);
                const std::size_t T = static_cast<std::size_t>(io.T);
                s.gqn = grow(s, s.gqn, s.gqn_bytes, T * io.h_k * 128 * sizeof(float));
                s.gkn = grow(s, s.gkn, s.gkn_bytes, T * io.h_k * 128 * sizeof(float));
                s.gvc = grow(s, s.gvc, s.gvc_bytes, T * io.h_v * 128 * sizeof(float));
                s.ggb = grow(s, s.ggb, s.ggb_bytes, T * io.h_v * 2 * sizeof(float));
                s.gkq = grow(s, s.gkq, s.gkq_bytes, T * io.h_k * sizeof(float));
                s.go = grow(s, s.go, s.go_bytes, T * io.h_v * 128 * sizeof(float));
                s.ghist = grow(s, s.ghist, s.ghist_bytes, static_cast<std::size_t>(io.channels) * 3 * sizeof(float));
                const TsmGdnArgs args{static_cast<std::uint32_t>(io.T), static_cast<std::uint32_t>(io.h_k),
                    static_cast<std::uint32_t>(io.h_v), static_cast<std::uint32_t>(io.channels), io.eps, 0};
                const ggml_metal_buffer_id qkv = tensor_id(io.qkv), z = tensor_id(io.z), al = tensor_id(io.alpha),
                    be = tensor_id(io.beta), cw = tensor_id(io.conv_w), dt = tensor_id(io.dt), sa = tensor_id(io.a),
                    nw = tensor_id(io.norm_w), hist = tensor_id(io.conv_state), st = tensor_id(io.ssm_state),
                    y = tensor_id(io.y);

                id<MTLCommandBuffer> cb = [queue commandBuffer];
                id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];   // serial: each pass sees the last
                [enc setComputePipelineState:pipeline(s, "tsm_gdn_pre")];
                bind(enc, qkv, 0, 0); bind(enc, hist, 0, 1); bind(enc, cw, 0, 2); bind(enc, al, 0, 3); bind(enc, be, 0, 4);
                bind(enc, dt, 0, 5); bind(enc, sa, 0, 6);
                [enc setBuffer:s.gqn offset:0 atIndex:7]; [enc setBuffer:s.gkn offset:0 atIndex:8];
                [enc setBuffer:s.gvc offset:0 atIndex:9]; [enc setBuffer:s.ggb offset:0 atIndex:10];
                [enc setBuffer:s.gkq offset:0 atIndex:11]; [enc setBuffer:s.ghist offset:0 atIndex:12];
                [enc setBytes:&args length:sizeof(args) atIndex:13];
                [enc dispatchThreadgroups:MTLSizeMake(io.h_k + io.h_v / 2, T, 1) threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];

                [enc setComputePipelineState:pipeline(s, "tsm_gdn_hist_store")];
                [enc setBuffer:s.ghist offset:0 atIndex:0]; bind(enc, hist, 0, 1);
                [enc setBytes:&args length:sizeof(args) atIndex:2];
                const NSUInteger hist_n = static_cast<NSUInteger>(io.channels) * 3;
                [enc dispatchThreadgroups:MTLSizeMake((hist_n + 255) / 256, 1, 1) threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];

                [enc setComputePipelineState:pipeline(s, "tsm_gdn_scan")];
                bind(enc, st, 0, 0);
                [enc setBuffer:s.gqn offset:0 atIndex:1]; [enc setBuffer:s.gkn offset:0 atIndex:2];
                [enc setBuffer:s.gvc offset:0 atIndex:3]; [enc setBuffer:s.ggb offset:0 atIndex:4];
                [enc setBuffer:s.gkq offset:0 atIndex:5]; [enc setBuffer:s.go offset:0 atIndex:6];
                [enc setBytes:&args length:sizeof(args) atIndex:7];
                [enc dispatchThreadgroups:MTLSizeMake(static_cast<NSUInteger>(io.h_v) * 4, 1, 1)
                    threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];

                [enc setComputePipelineState:pipeline(s, "tsm_gdn_post")];
                [enc setBuffer:s.go offset:0 atIndex:0]; bind(enc, z, 0, 1); bind(enc, nw, 0, 2); bind(enc, y, 0, 3);
                [enc setBytes:&args length:sizeof(args) atIndex:4];
                [enc dispatchThreadgroups:MTLSizeMake(io.h_v, T, 1) threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];
                [enc endEncoding];
                [cb commit];
                s.last = cb;
            }
            g_gdn_prompt_count.fetch_add(1, std::memory_order_relaxed);
            static std::atomic<bool> reported{false};
            if (!reported.exchange(true))
                std::fprintf(stderr, "[tsm] prompt GDN layers run on TensorSharp's kernels (Strata's gdn_prompt: pre, scan, "
                    "post; TS_METAL_GDN_PROMPT=0 keeps ggml's ops)\n");
            return true;
        }
        catch (const std::exception& e)
        {
            tsg::set_last_error(e.what());
            return false;
        }
    }

    long long gdn_prompt_count() { return g_gdn_prompt_count.load(std::memory_order_relaxed); }

    // ---- prompt hyper-connection passes (tsm_moe.h hc_*)
    namespace
    {
        struct TsmHcArgs { std::uint32_t n_tok, n_embd, hc, pad; };
        static_assert(sizeof(TsmHcArgs) == 16, "TsmHcArgs must match tsm_hc_prompt.metal");
        std::atomic<long long> g_hc_prompt_count{0};

        void require_f32(const ggml_tensor* t, std::int64_t elems, const char* what)
        {
            if (t == nullptr || t->type != GGML_TYPE_F32 || !ggml_is_contiguous(t) || ggml_nelements(t) != elems)
                throw std::runtime_error(std::string("tsm hc: ") + what + " is not a contiguous F32 tensor of the shape");
        }

        // One dispatch of `name` over (ceil(n / 256), T) threadgroups of 256, on ggml-metal's queue.
        bool hc_encode(ggml_backend* backend, const char* name, std::initializer_list<const ggml_tensor*> tensors,
                       const TsmHcArgs& args)
        {
            try
            {
                ggml_metal_device_t dev = metal_device(backend);
                if (dev == nullptr) throw std::runtime_error("tsm hc: not a Metal backend");
                std::vector<ggml_metal_buffer_id> ids;
                for (const ggml_tensor* t : tensors) ids.push_back(tensor_id(t));
                State& s = state();
                std::lock_guard<std::mutex> lock(s.mu);
                @autoreleasepool
                {
                    id<MTLDevice> device = (__bridge id<MTLDevice>)ggml_metal_device_get_obj(dev);
                    id<MTLCommandQueue> queue = (__bridge id<MTLCommandQueue>)ggml_metal_device_get_queue(dev);
                    ensure_library(s, device);
                    id<MTLCommandBuffer> cb = [queue commandBuffer];
                    id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
                    [enc setComputePipelineState:pipeline(s, name)];
                    int index = 0;
                    for (const ggml_metal_buffer_id& bid : ids) bind(enc, bid, 0, index++);
                    [enc setBytes:&args length:sizeof(args) atIndex:index];
                    [enc dispatchThreadgroups:MTLSizeMake((args.n_embd + 255) / 256, args.n_tok, 1)
                        threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
                    [enc endEncoding];
                    [cb commit];
                    s.last = cb;
                }
                g_hc_prompt_count.fetch_add(1, std::memory_order_relaxed);
                static std::atomic<bool> reported{false};
                if (!reported.exchange(true))
                    std::fprintf(stderr, "[tsm] prompt hyper-connection passes run on TensorSharp's kernels (mix, scatter; "
                        "TS_METAL_HC_PROMPT=0 keeps ggml's ops)\n");
                return true;
            }
            catch (const std::exception& e)
            {
                tsg::set_last_error(e.what());
                return false;
            }
        }
    }

    bool hc_prompt_supported(ggml_backend* backend)
    {
        const char* e = std::getenv("TS_METAL_HC_PROMPT");
        if (e && e[0] == '0') return false;
        return metal_device(backend) != nullptr;
    }

    bool hc_mix_encode(ggml_backend* backend, const ggml_tensor* xn, const ggml_tensor* G, ggml_tensor* mixed,
                       int n_embd, int hc, int T)
    {
        try
        {
            const std::int64_t n = n_embd, wide = static_cast<std::int64_t>(hc) * n_embd * T;
            require_f32(xn, wide, "xn"); require_f32(G, wide, "G"); require_f32(mixed, n * T, "mixed");
        }
        catch (const std::exception& e) { tsg::set_last_error(e.what()); return false; }
        const TsmHcArgs args{static_cast<std::uint32_t>(T), static_cast<std::uint32_t>(n_embd), static_cast<std::uint32_t>(hc), 0};
        return hc_encode(backend, "tsm_hc_mix", {xn, G, mixed}, args);
    }

    bool hc_scatter_encode(ggml_backend* backend, const ggml_tensor* res, const ggml_tensor* b, const ggml_tensor* inject,
                           ggml_tensor* out, int n_embd, int hc, int T)
    {
        try
        {
            const std::int64_t n = n_embd, wide = static_cast<std::int64_t>(hc) * n_embd * T;
            require_f32(res, wide, "res"); require_f32(b, n * T, "b");
            require_f32(inject, static_cast<std::int64_t>(hc) * T, "inject"); require_f32(out, wide, "out");
        }
        catch (const std::exception& e) { tsg::set_last_error(e.what()); return false; }
        const TsmHcArgs args{static_cast<std::uint32_t>(T), static_cast<std::uint32_t>(n_embd), static_cast<std::uint32_t>(hc), 0};
        return hc_encode(backend, "tsm_hc_scatter", {res, b, inject, out}, args);
    }

    long long hc_prompt_count() { return g_hc_prompt_count.load(std::memory_order_relaxed); }

    // ---- prompt QSA selection (tsm_moe.h qsa_select_*)
    namespace
    {
        struct TsmQsaArgs { std::uint32_t n_tok, n_cells, n_blocks, heads, width, write_mask, write_idx, pad; };
        struct TsmQsaAttnArgs
        {
            std::uint32_t n_tok, n_head, n_head_kv, n_cells, width, k_row, k_head, v_row, v_head;
            float scale;
            std::uint32_t pad0, pad1;
        };
        static_assert(sizeof(TsmQsaAttnArgs) == 48, "TsmQsaAttnArgs must match tsm_qsa_prompt.metal");
        std::atomic<long long> g_qsa_attn_count{0};
        static_assert(sizeof(TsmQsaArgs) == 32, "TsmQsaArgs must match tsm_qsa_prompt.metal");
        std::atomic<long long> g_qsa_select_count{0};

        void require_type(const ggml_tensor* t, ggml_type type, std::int64_t elems, const char* what)
        {
            if (t == nullptr || t->type != type || !ggml_is_contiguous(t) || ggml_nelements(t) != elems)
                throw std::runtime_error(std::string("tsm qsa: ") + what + " is not a contiguous tensor of the type and shape");
        }
    }

    bool qsa_select_supported(ggml_backend* backend)
    {
        const char* e = std::getenv("TS_METAL_QSA_SELECT");
        if (e && e[0] == '0') return false;
        return metal_device(backend) != nullptr;
    }

    bool qsa_select_encode(ggml_backend* backend, const ggml_tensor* scores, const ggml_tensor* bias,
                           const ggml_tensor* cell_blocks, const ggml_tensor* mask_in, ggml_tensor* mask_out,
                           ggml_tensor* idx, ggml_tensor* cnt, int T, int n_cells, int n_blocks, int heads, int width)
    {
        try
        {
            ggml_metal_device_t dev = metal_device(backend);
            if (dev == nullptr) throw std::runtime_error("tsm qsa: not a Metal backend");
            if (T <= 0 || n_cells <= 0 || n_blocks <= 0 || heads <= 0 || width <= 0)
                throw std::runtime_error("tsm qsa: empty selection");
            const std::int64_t t = T, cells = n_cells, blocks = n_blocks;
            require_type(scores, GGML_TYPE_F32, blocks * heads * t, "scores");
            require_type(bias, GGML_TYPE_F32, blocks * t, "bias");
            require_type(cell_blocks, GGML_TYPE_I32, cells, "cell_blocks");
            require_type(mask_in, GGML_TYPE_F16, cells * t, "mask_in");
            if (mask_out == nullptr && idx == nullptr) throw std::runtime_error("tsm qsa: nothing to write");
            if (mask_out) require_type(mask_out, GGML_TYPE_F16, cells * t, "mask_out");
            if (idx)
            {
                require_type(idx, GGML_TYPE_I32, static_cast<std::int64_t>(width) * t, "idx");
                require_type(cnt, GGML_TYPE_I32, t, "cnt");
            }
            const ggml_metal_buffer_id sc = tensor_id(scores), bi = tensor_id(bias), cb = tensor_id(cell_blocks),
                mi = tensor_id(mask_in);
            // Unwritten outputs are bound to an input (nothing reads or writes them).
            const ggml_metal_buffer_id mo = mask_out ? tensor_id(mask_out) : mi;
            const ggml_metal_buffer_id ix = idx ? tensor_id(idx) : cb, cn = idx ? tensor_id(cnt) : cb;
            const TsmQsaArgs args{static_cast<std::uint32_t>(T), static_cast<std::uint32_t>(n_cells),
                static_cast<std::uint32_t>(n_blocks), static_cast<std::uint32_t>(heads), static_cast<std::uint32_t>(width),
                mask_out ? 1u : 0u, idx ? 1u : 0u, 0};
            State& s = state();
            std::lock_guard<std::mutex> lock(s.mu);
            @autoreleasepool
            {
                id<MTLDevice> device = (__bridge id<MTLDevice>)ggml_metal_device_get_obj(dev);
                id<MTLCommandQueue> queue = (__bridge id<MTLCommandQueue>)ggml_metal_device_get_queue(dev);
                ensure_library(s, device);
                s.qs = grow(s, s.qs, s.qs_bytes, static_cast<std::size_t>(blocks * t) * sizeof(float));
                id<MTLCommandBuffer> cmd = [queue commandBuffer];
                id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];   // serial: select reads the scores
                [enc setComputePipelineState:pipeline(s, "tsm_qsa_block_scores")];
                bind(enc, sc, 0, 0); bind(enc, bi, 0, 1);
                [enc setBuffer:s.qs offset:0 atIndex:2];
                [enc setBytes:&args length:sizeof(args) atIndex:3];
                [enc dispatchThreadgroups:MTLSizeMake((n_blocks + 255) / 256, T, 1) threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
                [enc setComputePipelineState:pipeline(s, "tsm_qsa_select")];
                [enc setBuffer:s.qs offset:0 atIndex:0];
                bind(enc, cb, 0, 1); bind(enc, mi, 0, 2); bind(enc, mo, 0, 3);
                [enc setBytes:&args length:sizeof(args) atIndex:4];
                bind(enc, ix, 0, 5); bind(enc, cn, 0, 6);
                [enc dispatchThreadgroups:MTLSizeMake(T, 1, 1) threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
                [enc endEncoding];
                [cmd commit];
                s.last = cmd;
            }
            g_qsa_select_count.fetch_add(1, std::memory_order_relaxed);
            static std::atomic<bool> reported{false};
            if (!reported.exchange(true))
                std::fprintf(stderr, "[tsm] prompt QSA selection runs on TensorSharp's kernels (block scores, radix select; "
                    "TS_METAL_QSA_SELECT=0 keeps ggml's ops)\n");
            return true;
        }
        catch (const std::exception& e)
        {
            tsg::set_last_error(e.what());
            return false;
        }
    }

    long long qsa_select_count() { return g_qsa_select_count.load(std::memory_order_relaxed); }

    // Whether the attention over the chosen cells runs on the tensor-API kernel here (compiles it on first use).
    bool qsa_attn_tensor_api(ggml_backend* backend)
    {
        ggml_metal_device_t dev = metal_device(backend);
        if (dev == nullptr) return false;
        try
        {
            State& s = state();
            std::lock_guard<std::mutex> lock(s.mu);
            @autoreleasepool
            {
                id<MTLDevice> device = (__bridge id<MTLDevice>)ggml_metal_device_get_obj(dev);
                ensure_library(s, device);
                return mpp_available(s, device);
            }
        }
        catch (const std::exception&) { return false; }
    }

    bool qsa_attn_supported(ggml_backend* backend, int head_dim, int n_head, int n_head_kv, int kv_type, int n_cells)
    {
        const char* e = std::getenv("TS_METAL_QSA_ATTN");
        if (e && e[0] == '0') return false;
        const char* m = std::getenv("TS_METAL_QSA_ATTN_MIN_CELLS");   // read per graph build (a test lowers it)
        // M5 Pro, Flash Next, a 4096-token chunk's 12 QSA layers against ggml's flash attention: the simdgroup kernel
        // +0.40 s at a 4K window, +0.20 s at 8K, -0.23 s at 12K, -0.62 s at 16K; the tensor-API kernel (1.9x faster)
        // +0.18 s at 4K, -0.20 s at 8K, -0.57 s at 12K, -0.97 s at 16K.
        int min_cells = 12288;
        if (qsa_attn_tensor_api(backend)) min_cells = 8192;
        if (m && std::atoi(m) > 0) min_cells = std::atoi(m);
        return metal_device(backend) != nullptr && head_dim == 256 && n_head_kv > 0 && n_head % n_head_kv == 0
            && n_head / n_head_kv <= 16 && kv_type == GGML_TYPE_F16 && n_cells >= min_cells;
    }

    bool qsa_attn_encode(ggml_backend* backend, const ggml_tensor* q, const ggml_tensor* k, const ggml_tensor* v,
                         const ggml_tensor* idx, const ggml_tensor* cnt, const ggml_tensor* mask_in, ggml_tensor* out,
                         int T, int n_head, int n_head_kv, int n_cells, int width, float scale)
    {
        try
        {
            ggml_metal_device_t dev = metal_device(backend);
            if (dev == nullptr) throw std::runtime_error("tsm qsa attn: not a Metal backend");
            const std::int64_t t = T, D = 256;
            require_type(q, GGML_TYPE_F32, D * t * n_head, "q");
            require_type(out, GGML_TYPE_F32, D * n_head * t, "out");
            require_type(idx, GGML_TYPE_I32, static_cast<std::int64_t>(width) * t, "idx");
            require_type(cnt, GGML_TYPE_I32, t, "cnt");
            require_type(mask_in, GGML_TYPE_F16, static_cast<std::int64_t>(n_cells) * t, "mask_in");
            for (const ggml_tensor* c : {k, v})
                if (c == nullptr || c->type != GGML_TYPE_F16 || c->ne[0] != D || c->nb[0] != 2 || c->ne[2] != n_head_kv
                    || c->ne[1] < n_cells || c->nb[1] % 8 != 0 || c->nb[2] % 8 != 0)
                    throw std::runtime_error("tsm qsa attn: the cache is not F16 [256, cells, kv heads] with aligned rows");
            const ggml_metal_buffer_id qi = tensor_id(q), ki = tensor_id(k), vi = tensor_id(v), xi = tensor_id(idx),
                ci = tensor_id(cnt), mi = tensor_id(mask_in), oi = tensor_id(out);
            const TsmQsaAttnArgs args{static_cast<std::uint32_t>(T), static_cast<std::uint32_t>(n_head),
                static_cast<std::uint32_t>(n_head_kv), static_cast<std::uint32_t>(n_cells), static_cast<std::uint32_t>(width),
                static_cast<std::uint32_t>(k->nb[1] / 2), static_cast<std::uint32_t>(k->nb[2] / 2),
                static_cast<std::uint32_t>(v->nb[1] / 2), static_cast<std::uint32_t>(v->nb[2] / 2), scale, 0, 0};
            State& s = state();
            std::lock_guard<std::mutex> lock(s.mu);
            @autoreleasepool
            {
                id<MTLDevice> device = (__bridge id<MTLDevice>)ggml_metal_device_get_obj(dev);
                id<MTLCommandQueue> queue = (__bridge id<MTLCommandQueue>)ggml_metal_device_get_queue(dev);
                ensure_library(s, device);
                s.qh = grow(s, s.qh, s.qh_bytes, static_cast<std::size_t>(n_head_kv) * 16 * t * D * sizeof(std::uint16_t));
                id<MTLCommandBuffer> cmd = [queue commandBuffer];
                id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];   // serial: the attention reads qh
                [enc setComputePipelineState:pipeline(s, "tsm_qsa_q_half")];
                bind(enc, qi, 0, 0);
                [enc setBuffer:s.qh offset:0 atIndex:1];
                [enc setBytes:&args length:sizeof(args) atIndex:2];
                [enc dispatchThreadgroups:MTLSizeMake(16, T, n_head_kv) threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
                [enc setComputePipelineState:mpp_available(s, device) ? pipeline_mpp(s, "tsm_qsa_attn_mm")
                                                                       : pipeline(s, "tsm_qsa_attn")];
                [enc setBuffer:s.qh offset:0 atIndex:0]; bind(enc, ki, 0, 1); bind(enc, vi, 0, 2); bind(enc, xi, 0, 3);
                bind(enc, ci, 0, 4); bind(enc, mi, 0, 5); bind(enc, oi, 0, 6);
                [enc setBytes:&args length:sizeof(args) atIndex:7];
                [enc dispatchThreadgroups:MTLSizeMake(T, n_head_kv, 1) threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];
                [enc endEncoding];
                [cmd commit];
                s.last = cmd;
            }
            g_qsa_attn_count.fetch_add(1, std::memory_order_relaxed);
            static std::atomic<bool> reported{false};
            if (!reported.exchange(true))
                std::fprintf(stderr, "[tsm] prompt QSA attention reads only the chosen cells (TS_METAL_QSA_ATTN=0 keeps "
                    "ggml's flash attention)\n");
            return true;
        }
        catch (const std::exception& e)
        {
            tsg::set_last_error(e.what());
            return false;
        }
    }

    long long qsa_attn_count() { return g_qsa_attn_count.load(std::memory_order_relaxed); }

    // ---- the prompt expert GEMM (Strata's moe_mm: tsm/kernels/tsm_moe_mm.metal) ---------------------------------
    namespace
    {
        std::atomic<long long> g_moe_mm_count{0};
        constexpr int kMmTm = 16;   // the plan's tile height (rows of one expert)
        struct MmPlanArgs { std::uint32_t n_ent, topk, n_expert, tm, n_blocks; };
        struct MmArgs
        {
            std::uint64_t gate_off, up_off, down_off, gu_stride, d_stride;
            std::uint32_t gu_row, d_row, n_embd, n_ff;
        };
        static_assert(sizeof(MmArgs) == 56, "MmArgs must match tsm_moe_mm.metal");
    }

    bool moe_mm_enabled()
    {
        const char* e = std::getenv("TS_METAL_MOE_MM");
        return !(e != nullptr && e[0] == '0');
    }

    bool moe_mm_supported(ggml_backend* backend, int gate_type, int up_type, int down_type, int hidden, int n_ff,
                          int n_expert, int n_used)
    {
        if (!moe_mm_enabled()) return false;
        ggml_metal_device_t dev = metal_device(backend);
        if (dev == nullptr) return false;
        // The instantiated tiles: Q2_0 everywhere; gate/up 128 n_ff rows a threadgroup, down 128 n_embd rows; K in
        // chunks of 32; the plan's scan runs one thread an expert in one threadgroup.
        if (gate_type != 42 || up_type != 42 || down_type != 42) return false;
        if (hidden <= 0 || n_ff <= 0 || hidden % 128 != 0 || n_ff % 128 != 0) return false;
        if (n_expert <= 0 || n_expert > 1024 || n_used <= 0) return false;
        try
        {
            State& s = state();
            std::lock_guard<std::mutex> lock(s.mu);
            id<MTLDevice> device = (__bridge id<MTLDevice>)ggml_metal_device_get_obj(dev);
            ensure_library(s, device);
            return mm_library(s, device);
        }
        catch (const std::exception&)
        {
            return false;
        }
    }

    bool moe_mm_encode(ggml_backend* backend, const ggml_tensor* x, const ggml_tensor* ids, const ggml_tensor* wts,
                       const ggml_tensor* gate, const ggml_tensor* up, const ggml_tensor* down, ggml_tensor* out,
                       int T, int hidden, int n_ff, int n_expert, int n_used, bool wait)
    {
        try
        {
            ggml_metal_device_t dev = metal_device(backend);
            if (dev == nullptr) throw std::runtime_error("tsm moe mm: not a Metal backend");
            const std::int64_t t = T;
            require_type(x, GGML_TYPE_F32, hidden * t, "x");
            require_type(out, GGML_TYPE_F32, hidden * t, "out");
            require_type(ids, GGML_TYPE_I32, n_used * t, "ids");
            require_type(wts, GGML_TYPE_F32, n_used * t, "weights");
            for (const ggml_tensor* w : {gate, up, down})
                if (w == nullptr || w->type != GGML_TYPE_Q2_0 || w->ne[2] != n_expert || !ggml_is_contiguous(w))
                    throw std::runtime_error("tsm moe mm: the experts are not contiguous Q2_0 stacks");
            if (gate->ne[0] != hidden || gate->ne[1] != n_ff || up->ne[0] != hidden || up->ne[1] != n_ff
                || down->ne[0] != n_ff || down->ne[1] != hidden)
                throw std::runtime_error("tsm moe mm: expert shapes");
            const ggml_metal_buffer_id gi = tensor_id(gate), ui = tensor_id(up), di = tensor_id(down);
            // Strata's kernels address gate, up and down from one buffer: base + offset + expert * stride.
            if (gi.metal != ui.metal || gi.metal != di.metal)
                throw std::runtime_error("tsm moe mm: the expert stacks are in different buffers");
            const std::size_t base = std::min({gi.offs, ui.offs, di.offs});
            MmArgs ma{};
            ma.gate_off = gi.offs - base; ma.up_off = ui.offs - base; ma.down_off = di.offs - base;
            ma.gu_row = static_cast<std::uint32_t>(row_bytes(GGML_TYPE_Q2_0, hidden));
            ma.d_row = static_cast<std::uint32_t>(row_bytes(GGML_TYPE_Q2_0, n_ff));
            ma.gu_stride = static_cast<std::uint64_t>(ma.gu_row) * n_ff;
            ma.d_stride = static_cast<std::uint64_t>(ma.d_row) * hidden;
            ma.n_embd = static_cast<std::uint32_t>(hidden);
            ma.n_ff = static_cast<std::uint32_t>(n_ff);
            if (gate->nb[2] != ma.gu_stride || up->nb[2] != ma.gu_stride || down->nb[2] != ma.d_stride)
                throw std::runtime_error("tsm moe mm: expert strides");
            const std::uint32_t n_ent = static_cast<std::uint32_t>(n_used) * static_cast<std::uint32_t>(T);
            const std::uint32_t n_blocks = (n_ent + 1023) / 1024;
            const std::size_t max_rows = static_cast<std::size_t>(n_ent) + static_cast<std::size_t>(n_expert) * (kMmTm - 1);
            const std::size_t max_tiles = (max_rows + kMmTm - 1) / kMmTm;
            const MmPlanArgs pa{n_ent, static_cast<std::uint32_t>(n_used), static_cast<std::uint32_t>(n_expert),
                static_cast<std::uint32_t>(kMmTm), n_blocks};
            const ggml_metal_buffer_id xi = tensor_id(x), ii = tensor_id(ids), wi = tensor_id(wts), oi = tensor_id(out);
            State& s = state();
            std::lock_guard<std::mutex> lock(s.mu);
            @autoreleasepool
            {
                id<MTLDevice> device = (__bridge id<MTLDevice>)ggml_metal_device_get_obj(dev);
                id<MTLCommandQueue> queue = (__bridge id<MTLCommandQueue>)ggml_metal_device_get_queue(dev);
                ensure_library(s, device);
                if (!mm_library(s, device)) throw std::runtime_error("tsm moe mm: the tensor API does not run here");
                const std::size_t blocks_bytes = static_cast<std::size_t>(n_blocks) * n_expert * sizeof(std::uint32_t);
                s.mm_bcnt = grow(s, s.mm_bcnt, s.mm_bcnt_bytes, blocks_bytes);
                s.mm_boff = grow(s, s.mm_boff, s.mm_boff_bytes, blocks_bytes);
                s.mm_eoff = grow(s, s.mm_eoff, s.mm_eoff_bytes, static_cast<std::size_t>(n_expert) * sizeof(std::uint32_t));
                s.mm_tiles = grow(s, s.mm_tiles, s.mm_tiles_bytes, max_tiles * 2 * sizeof(std::int32_t));
                s.mm_stok = grow(s, s.mm_stok, s.mm_stok_bytes, max_rows * sizeof(std::int32_t));
                s.mm_sw = grow(s, s.mm_sw, s.mm_sw_bytes, max_rows * sizeof(float));
                s.mm_pos = grow(s, s.mm_pos, s.mm_pos_bytes, static_cast<std::size_t>(n_ent) * sizeof(std::int32_t));
                s.mm_xs = grow(s, s.mm_xs, s.mm_xs_bytes, max_rows * hidden * sizeof(std::uint16_t));
                s.mm_hs = grow(s, s.mm_hs, s.mm_hs_bytes, max_rows * n_ff * sizeof(std::uint16_t));
                s.mm_ye = grow(s, s.mm_ye, s.mm_ye_bytes, max_rows * hidden * sizeof(float));
                if (s.mm_meta == nil)
                    s.mm_meta = grow(s, s.mm_meta, s.mm_meta_bytes, 16 * sizeof(std::uint32_t));
                // The GEMMs' grid widths, which the scan leaves to the host (it writes their heights and the gather's).
                auto* meta = static_cast<std::uint32_t*>([s.mm_meta contents]);
                meta[2] = static_cast<std::uint32_t>(n_ff / (32 * 4));
                meta[5] = static_cast<std::uint32_t>(hidden / (64 * 2));
                id<MTLBuffer> W = (__bridge id<MTLBuffer>)gi.metal;
                id<MTLCommandBuffer> cmd = [queue commandBuffer];
                id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];   // serial: each pass reads the last
                const std::uint32_t nclr = static_cast<std::uint32_t>(max_rows);
                [enc setComputePipelineState:pipeline_mm(s, "mm_clear")];
                [enc setBuffer:s.mm_stok offset:0 atIndex:0];
                [enc setBytes:&nclr length:sizeof(nclr) atIndex:1];
                [enc dispatchThreadgroups:MTLSizeMake((max_rows + 255) / 256, 1, 1) threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
                [enc setComputePipelineState:pipeline_mm(s, "mm_hist")];
                bind(enc, ii, 0, 0);
                [enc setBuffer:s.mm_bcnt offset:0 atIndex:1];
                [enc setBytes:&pa length:sizeof(pa) atIndex:2];
                [enc dispatchThreadgroups:MTLSizeMake(n_blocks, 1, 1) threadsPerThreadgroup:MTLSizeMake(1024, 1, 1)];
                [enc setComputePipelineState:pipeline_mm(s, "mm_scan")];
                [enc setBuffer:s.mm_bcnt offset:0 atIndex:0];
                [enc setBuffer:s.mm_boff offset:0 atIndex:1];
                [enc setBuffer:s.mm_eoff offset:0 atIndex:2];
                [enc setBuffer:s.mm_tiles offset:0 atIndex:3];
                [enc setBuffer:s.mm_meta offset:0 atIndex:4];
                [enc setBytes:&pa length:sizeof(pa) atIndex:5];
                // whole simdgroups: each publishes its totals from lane 31 (threads past the experts count 0 rows)
                [enc dispatchThreadgroups:MTLSizeMake(1, 1, 1)
                    threadsPerThreadgroup:MTLSizeMake(static_cast<std::size_t>((n_expert + 31) / 32 * 32), 1, 1)];
                [enc setComputePipelineState:pipeline_mm(s, "mm_place")];
                bind(enc, ii, 0, 0);
                bind(enc, wi, 0, 1);
                [enc setBuffer:s.mm_boff offset:0 atIndex:2];
                [enc setBuffer:s.mm_eoff offset:0 atIndex:3];
                [enc setBuffer:s.mm_stok offset:0 atIndex:4];
                [enc setBuffer:s.mm_sw offset:0 atIndex:5];
                [enc setBuffer:s.mm_pos offset:0 atIndex:6];
                [enc setBytes:&pa length:sizeof(pa) atIndex:7];
                [enc dispatchThreadgroups:MTLSizeMake(n_blocks, 1, 1) threadsPerThreadgroup:MTLSizeMake(1024, 1, 1)];
                const std::uint32_t ne = static_cast<std::uint32_t>(hidden);
                [enc setComputePipelineState:pipeline_mm(s, "mm_gather")];
                bind(enc, xi, 0, 0);
                [enc setBuffer:s.mm_stok offset:0 atIndex:1];
                [enc setBuffer:s.mm_xs offset:0 atIndex:2];
                [enc setBytes:&ne length:sizeof(ne) atIndex:3];
                [enc dispatchThreadgroupsWithIndirectBuffer:s.mm_meta indirectBufferOffset:8 * sizeof(std::uint32_t)
                                      threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
                // gate/up then down; buffers 5..8 (the SSD arena's slot map and chunks) are unused without its constant
                [enc setComputePipelineState:pipeline_mm(s, "mm_gu3_42_16_32_32_4")];
                [enc setBuffer:W offset:base atIndex:0];
                [enc setBuffer:s.mm_xs offset:0 atIndex:1];
                [enc setBuffer:s.mm_hs offset:0 atIndex:2];
                [enc setBuffer:s.mm_tiles offset:0 atIndex:3];
                [enc setBytes:&ma length:sizeof(ma) atIndex:4];
                for (int b = 5; b <= 8; ++b) [enc setBuffer:W offset:base atIndex:b];
                [enc dispatchThreadgroupsWithIndirectBuffer:s.mm_meta indirectBufferOffset:2 * sizeof(std::uint32_t)
                                      threadsPerThreadgroup:MTLSizeMake(32 * 4, 1, 1)];
                [enc setComputePipelineState:pipeline_mm(s, "mm_down3_42_16_64_32_2")];
                [enc setBuffer:W offset:base atIndex:0];
                [enc setBuffer:s.mm_hs offset:0 atIndex:1];
                [enc setBuffer:s.mm_ye offset:0 atIndex:2];
                [enc setBuffer:s.mm_tiles offset:0 atIndex:3];
                [enc setBytes:&ma length:sizeof(ma) atIndex:4];
                for (int b = 5; b <= 8; ++b) [enc setBuffer:W offset:base atIndex:b];
                [enc dispatchThreadgroupsWithIndirectBuffer:s.mm_meta indirectBufferOffset:5 * sizeof(std::uint32_t)
                                      threadsPerThreadgroup:MTLSizeMake(32 * 2, 1, 1)];
                const std::uint32_t ca[4] = {static_cast<std::uint32_t>(T), static_cast<std::uint32_t>(hidden),
                    static_cast<std::uint32_t>(n_used), 0};
                [enc setComputePipelineState:pipeline_mm(s, "mm_combine")];
                [enc setBuffer:s.mm_ye offset:0 atIndex:0];
                [enc setBuffer:s.mm_pos offset:0 atIndex:1];
                [enc setBuffer:s.mm_sw offset:0 atIndex:2];
                bind(enc, oi, 0, 3);
                [enc setBytes:ca length:sizeof(ca) atIndex:4];
                [enc dispatchThreadgroups:MTLSizeMake((static_cast<std::size_t>(T) * hidden + 255) / 256, 1, 1)
                    threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
                [enc endEncoding];
                [cmd commit];
                s.last = cmd;
                if (wait)
                {
                    [cmd waitUntilCompleted];
                    if (cmd.status == MTLCommandBufferStatusError)
                        throw std::runtime_error(std::string("tsm moe mm: ") +
                            (cmd.error ? cmd.error.localizedDescription.UTF8String : "command buffer failed"));
                }
            }
            g_moe_mm_count.fetch_add(1, std::memory_order_relaxed);
            static std::atomic<bool> reported{false};
            if (!reported.exchange(true))
                std::fprintf(stderr, "[tsm] prompt experts run on Strata's grouped GEMM (moe_mm on the tensor API; "
                    "TS_METAL_MOE_MM=0 keeps MUL_MAT_ID)\n");
            return true;
        }
        catch (const std::exception& e)
        {
            tsg::set_last_error(e.what());
            return false;
        }
    }

    long long moe_mm_count() { return g_moe_mm_count.load(std::memory_order_relaxed); }

    void wait_last()
    {
        State& s = state();
        id<MTLCommandBuffer> last = nil;
        {
            std::lock_guard<std::mutex> lock(s.mu);
            last = s.last;
        }
        if (last != nil) [last waitUntilCompleted];
    }

    void* copy_async(ggml_backend* backend, const ggml_tensor* const* srcs, const ggml_tensor* const* dsts, int n)
    {
        try
        {
            ggml_metal_device_t dev = metal_device(backend);
            if (dev == nullptr) throw std::runtime_error("tsm copy: not a Metal backend");
            if (n <= 0 || srcs == nullptr || dsts == nullptr) throw std::runtime_error("tsm copy: nothing to copy");
            std::vector<ggml_metal_buffer_id> from, to;
            std::vector<std::size_t> bytes;
            auto metal_buffer = [](const ggml_tensor* t) {
                ggml_backend_buffer_t b = t->view_src ? t->view_src->buffer : t->buffer;
                const char* name = b ? ggml_backend_buft_name(ggml_backend_buffer_get_type(b)) : nullptr;
                return name != nullptr && std::strncmp(name, "MTL", 3) == 0;
            };
            for (int i = 0; i < n; ++i)
            {
                if (!metal_buffer(srcs[i]) || !metal_buffer(dsts[i])) throw std::runtime_error("tsm copy: not a Metal buffer");
                const std::size_t b = ggml_nbytes(srcs[i]);
                if (b != ggml_nbytes(dsts[i])) throw std::runtime_error("tsm copy: sizes differ");
                from.push_back(tensor_id(srcs[i]));
                to.push_back(tensor_id(dsts[i]));
                bytes.push_back(b);
            }
            @autoreleasepool
            {
                id<MTLCommandQueue> queue = (__bridge id<MTLCommandQueue>)ggml_metal_device_get_queue(dev);
                id<MTLCommandBuffer> cb = [queue commandBuffer];
                id<MTLBlitCommandEncoder> enc = [cb blitCommandEncoder];
                for (int i = 0; i < n; ++i)
                    [enc copyFromBuffer:(__bridge id<MTLBuffer>)from[i].metal sourceOffset:from[i].offs
                               toBuffer:(__bridge id<MTLBuffer>)to[i].metal destinationOffset:to[i].offs size:bytes[i]];
                [enc endEncoding];
                [cb commit];
                return (__bridge_retained void*)cb;
            }
        }
        catch (const std::exception& e)
        {
            tsg::set_last_error(e.what());
            return nullptr;
        }
    }

    void wait_handle(void* handle)
    {
        if (handle == nullptr) return;
        id<MTLCommandBuffer> cb = (__bridge id<MTLCommandBuffer>)handle;
        [cb waitUntilCompleted];
    }

    void release_handle(void* handle)
    {
        if (handle == nullptr) return;
        id<MTLCommandBuffer> cb = (__bridge_transfer id<MTLCommandBuffer>)handle;
        (void)cb;
    }

    // ---- the fused hyper-connection read (tsm_moe.h hc_read_*)
    namespace
    {
        struct TsmHrArgs { std::uint32_t n_tok, n_embd, hc, lr; float eps; std::uint32_t pad0, pad1, pad2; };
        static_assert(sizeof(TsmHrArgs) == 32, "TsmHrArgs must match tsm_hc_read.metal");
        struct TsmDqArgs { std::uint32_t n_in, n_out, row_bytes, perm, dst_row0, pad0, pad1, pad2; };
        static_assert(sizeof(TsmDqArgs) == 32, "TsmDqArgs must match tsm_hc_read.metal");
        struct TsmGemmArgs { std::int32_t M, N, K, pad; };
        std::atomic<long long> g_hc_read_count{0};

        const char* dq_kernel(int type)
        {
            switch (type)
            {
                case GGML_TYPE_F32: return "tsm_dq_f32";
                case GGML_TYPE_F16: return "tsm_dq_f16";
                case GGML_TYPE_BF16: return "tsm_dq_bf16";
                case GGML_TYPE_Q8_0: return "tsm_dq_q8_0";
                default: return nullptr;
            }
        }

        bool hc_read_enabled()
        {
            const char* e = std::getenv("TS_METAL_HC_READ");   // read per span build (tests switch it)
            return !(e && e[0] == '0');
        }

        // One weight's rows into the fp16 matrix at dst_row0 (perm: see tsm_hc_read.metal).
        void encode_dq(State& s, id<MTLComputeCommandEncoder> enc, const ggml_tensor* w, id<MTLBuffer> dst,
                       std::uint32_t dst_row0, std::uint32_t perm)
        {
            const TsmDqArgs a{static_cast<std::uint32_t>(w->ne[0]), static_cast<std::uint32_t>(w->ne[1]),
                static_cast<std::uint32_t>(w->nb[1]), perm, dst_row0, 0, 0, 0};
            [enc setComputePipelineState:pipeline_mm(s, dq_kernel(w->type))];
            bind(enc, tensor_id(w), 0, 0);
            [enc setBuffer:dst offset:0 atIndex:1];
            [enc setBytes:&a length:sizeof(a) atIndex:2];
            const NSUInteger runs = static_cast<NSUInteger>(w->ne[0] / 32);
            [enc dispatchThreads:MTLSizeMake(runs, static_cast<NSUInteger>(w->ne[1]), 1)
                threadsPerThreadgroup:MTLSizeMake(std::min<NSUInteger>(runs, 64), 1, 1)];
        }
    }

    bool hc_read_supported(ggml_backend* backend, int down_type, int inject_type, int up_type, int n_embd, int hc, int lr)
    {
        if (!hc_read_enabled()) return false;
        ggml_metal_device_t dev = metal_device(backend);
        if (dev == nullptr) return false;
        if (dq_kernel(down_type) == nullptr || dq_kernel(inject_type) == nullptr || dq_kernel(up_type) == nullptr) return false;
        // Rows dequantize in runs of 32; the up GEMM's 64-column tiles hold 64 / hc whole outputs; the norm's
        // threadgroup reduces with 8 simdgroups.
        if (n_embd <= 0 || hc <= 0 || lr <= 0 || (static_cast<std::int64_t>(n_embd) * hc) % 64 != 0 || 64 % hc != 0
            || lr % 32 != 0)
            return false;
        try
        {
            State& s = state();
            std::lock_guard<std::mutex> lock(s.mu);
            id<MTLDevice> device = (__bridge id<MTLDevice>)ggml_metal_device_get_obj(dev);
            ensure_library(s, device);
            return mm_library(s, device);
        }
        catch (const std::exception&)
        {
            return false;
        }
    }

    bool hc_read_encode(ggml_backend* backend, const HcReadIO& io)
    {
        try
        {
            ggml_metal_device_t dev = metal_device(backend);
            if (dev == nullptr) throw std::runtime_error("tsm hc read: not a Metal backend");
            const std::int64_t n = io.n_embd, hc = io.hc, lr = io.lr, T = io.T, wide = n * hc;
            auto need = [](const ggml_tensor* t, const char* what, bool ok) {
                if (t == nullptr || !ok) throw std::runtime_error(std::string("tsm hc read: ") + what + " does not match the kernels");
            };
            need(io.res, "res", io.res && io.res->type == GGML_TYPE_F32 && ggml_is_contiguous(io.res) && ggml_nelements(io.res) == wide * T);
            need(io.gamma, "gamma", io.gamma && io.gamma->type == GGML_TYPE_F32 && ggml_nelements(io.gamma) == wide);
            need(io.w_down, "w_down", io.w_down && io.w_down->ne[0] == wide && io.w_down->ne[1] == lr);
            need(io.w_inject, "w_inject", io.w_inject && io.w_inject->ne[0] == wide && io.w_inject->ne[1] == hc);
            need(io.w_up, "w_up", io.w_up && io.w_up->ne[0] == lr && io.w_up->ne[1] == wide);
            need(io.mixed, "mixed", io.mixed && io.mixed->type == GGML_TYPE_F32 && ggml_nelements(io.mixed) == n * T);
            need(io.inject, "inject", io.inject && io.inject->type == GGML_TYPE_F32 && ggml_nelements(io.inject) == hc * T);
            const ggml_metal_buffer_id res = tensor_id(io.res), gamma = tensor_id(io.gamma), mixed = tensor_id(io.mixed),
                inject = tensor_id(io.inject);
            State& s = state();
            std::lock_guard<std::mutex> lock(s.mu);
            @autoreleasepool
            {
                id<MTLDevice> device = (__bridge id<MTLDevice>)ggml_metal_device_get_obj(dev);
                id<MTLCommandQueue> queue = (__bridge id<MTLCommandQueue>)ggml_metal_device_get_queue(dev);
                ensure_library(s, device);
                if (!mm_library(s, device)) throw std::runtime_error("tsm hc read: the tensor-API library is not available");
                const std::size_t t = static_cast<std::size_t>(T), w = static_cast<std::size_t>(wide);
                const std::size_t nr = static_cast<std::size_t>(lr + hc);
                s.hr_xnh = grow(s, s.hr_xnh, s.hr_xnh_bytes, t * w * 2);
                s.hr_v = grow(s, s.hr_v, s.hr_v_bytes, t * nr * 4);
                s.hr_lo = grow(s, s.hr_lo, s.hr_lo_bytes, t * static_cast<std::size_t>(lr) * 2);
                s.hr_wd = grow(s, s.hr_wd, s.hr_wd_bytes, nr * w * 2);
                s.hr_wu = grow(s, s.hr_wu, s.hr_wu_bytes, w * static_cast<std::size_t>(lr) * 2);
                const TsmHrArgs ha{static_cast<std::uint32_t>(T), static_cast<std::uint32_t>(n), static_cast<std::uint32_t>(hc),
                    static_cast<std::uint32_t>(lr), io.eps, 0, 0, 0};

                id<MTLCommandBuffer> cb = [queue commandBuffer];
                id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];   // serial: each pass sees the last
                // the weights in fp16: [down rows | injection rows] and the up rows with the streams interleaved
                encode_dq(s, enc, io.w_down, s.hr_wd, 0, 0);
                encode_dq(s, enc, io.w_inject, s.hr_wd, static_cast<std::uint32_t>(lr), 0);
                encode_dq(s, enc, io.w_up, s.hr_wu, 0, static_cast<std::uint32_t>(hc));

                [enc setComputePipelineState:pipeline_mm(s, "tsm_hr_norm")];
                bind(enc, res, 0, 0); bind(enc, gamma, 0, 1);
                [enc setBuffer:s.hr_xnh offset:0 atIndex:2];
                [enc setBytes:&ha length:sizeof(ha) atIndex:3];
                [enc dispatchThreadgroups:MTLSizeMake(static_cast<NSUInteger>(hc), t, 1) threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];

                const TsmGemmArgs gd{static_cast<std::int32_t>(T), static_cast<std::int32_t>(nr), static_cast<std::int32_t>(wide), 0};
                [enc setComputePipelineState:pipeline_mm(s, "tsm_gemm_f16_64_64_4")];
                [enc setBuffer:s.hr_xnh offset:0 atIndex:0]; [enc setBuffer:s.hr_wd offset:0 atIndex:1];
                [enc setBuffer:s.hr_v offset:0 atIndex:2];
                [enc setBytes:&gd length:sizeof(gd) atIndex:3];
                [enc dispatchThreadgroups:MTLSizeMake((nr + 63) / 64, (t + 63) / 64, 1) threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];

                [enc setComputePipelineState:pipeline_mm(s, "tsm_hr_epi_down")];
                [enc setBuffer:s.hr_v offset:0 atIndex:0]; [enc setBuffer:s.hr_lo offset:0 atIndex:1];
                bind(enc, inject, 0, 2);
                [enc setBytes:&ha length:sizeof(ha) atIndex:3];
                [enc dispatchThreads:MTLSizeMake(t * nr, 1, 1) threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];

                const TsmGemmArgs gu{static_cast<std::int32_t>(T), static_cast<std::int32_t>(wide), static_cast<std::int32_t>(lr), 0};
                [enc setComputePipelineState:pipeline_mm(s, "tsm_gemm_hc_up_64_64_4")];
                [enc setBuffer:s.hr_lo offset:0 atIndex:0]; [enc setBuffer:s.hr_wu offset:0 atIndex:1];
                [enc setBuffer:s.hr_xnh offset:0 atIndex:2]; bind(enc, mixed, 0, 3);
                [enc setBytes:&gu length:sizeof(gu) atIndex:4];
                [enc setBytes:&ha length:sizeof(ha) atIndex:5];
                [enc dispatchThreadgroups:MTLSizeMake(w / 64, (t + 63) / 64, 1) threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];
                [enc endEncoding];
                [cb commit];
                s.last = cb;
            }
            g_hc_read_count.fetch_add(1, std::memory_order_relaxed);
            g_hc_prompt_count.fetch_add(1, std::memory_order_relaxed);   // a hyper-connection pass on these kernels
            static std::atomic<bool> reported{false};
            if (!reported.exchange(true))
                std::fprintf(stderr, "[tsm] prompt hyper-connection reads run fused on the matrix units (norm, fp16 GEMMs, "
                    "the mix in the up projection's epilogue; TS_METAL_HC_READ=0 keeps ggml's ops)\n");
            return true;
        }
        catch (const std::exception& e)
        {
            tsg::set_last_error(e.what());
            return false;
        }
    }

    long long hc_read_count() { return g_hc_read_count.load(std::memory_order_relaxed); }

    void moe_mm_release()
    {
        State& s = state();
        std::lock_guard<std::mutex> lock(s.mu);
        s.hr_xnh = s.hr_v = s.hr_lo = s.hr_wd = s.hr_wu = nil;
        s.hr_xnh_bytes = s.hr_v_bytes = s.hr_lo_bytes = s.hr_wd_bytes = s.hr_wu_bytes = 0;
        if (s.mm_xs == nil && s.mm_ye == nil) return;
        // The last pass's command buffer holds its own references; dropping ours frees the scratch once it completes.
        s.mm_bcnt = s.mm_boff = s.mm_eoff = s.mm_tiles = s.mm_stok = s.mm_sw = s.mm_pos = nil;
        s.mm_xs = s.mm_hs = s.mm_ye = nil;
        s.mm_bcnt_bytes = s.mm_boff_bytes = s.mm_eoff_bytes = s.mm_tiles_bytes = 0;
        s.mm_stok_bytes = s.mm_sw_bytes = s.mm_pos_bytes = s.mm_xs_bytes = s.mm_hs_bytes = s.mm_ye_bytes = 0;
    }
}

// ---- test probe: one layer's held routes through moe_encode, on expert bytes the caller gives (stacked [expert]
// gate rows, up rows, down rows in GGUF blocks), against a reference the caller computes. Returns 1, or 0 with the
// last error set (the formats or shapes the kernels do not take, or no Metal backend).
// ---- test probe: one layer's routed experts through moe_mm_encode (Strata's prompt expert GEMM) on stacked Q2_0
// expert bytes the caller gives (gate [expert][n_ff rows of hidden], up likewise, down [expert][hidden rows of n_ff]),
// T rows of x and n_used (id, weight) routes a row, into out [T][hidden]. Returns 1, or 0 with the last error set.
TSG_EXPORT int TSGgml_TsmMoeMmProbe(int hidden, int n_ff, int n_expert, const void* gate, const void* up, const void* down,
    int T, int n_used, const std::int32_t* ids, const float* weights, const float* x, float* out)
{
    using namespace tsg;
    if (!ensure_backend()) return 0;
    ggml_backend_t backend = g_backend;
    ggml_backend_buffer_t wbuf = nullptr, abuf = nullptr;
    ggml_context* ctx = nullptr;
    try
    {
        if (!tsm::moe_mm_supported(backend, 42, 42, 42, hidden, n_ff, n_expert, n_used))
            throw std::runtime_error("tsm moe mm probe: the kernels do not run here or take these shapes");
        ggml_init_params ip{};
        ip.mem_size = 16 * ggml_tensor_overhead();
        ip.no_alloc = true;
        ctx = ggml_init(ip);
        ggml_tensor* g = ggml_new_tensor_3d(ctx, GGML_TYPE_Q2_0, hidden, n_ff, n_expert);
        ggml_tensor* u = ggml_new_tensor_3d(ctx, GGML_TYPE_Q2_0, hidden, n_ff, n_expert);
        ggml_tensor* d = ggml_new_tensor_3d(ctx, GGML_TYPE_Q2_0, n_ff, hidden, n_expert);
        wbuf = ggml_backend_alloc_ctx_tensors(ctx, backend);   // the three stacks in one buffer, as the kernels address them
        if (wbuf == nullptr) throw std::runtime_error("tsm moe mm probe: weight buffer");
        ggml_backend_tensor_set(g, gate, 0, ggml_nbytes(g));
        ggml_backend_tensor_set(u, up, 0, ggml_nbytes(u));
        ggml_backend_tensor_set(d, down, 0, ggml_nbytes(d));
        ggml_context* actx = nullptr;
        ggml_init_params ap{};
        ap.mem_size = 8 * ggml_tensor_overhead();
        ap.no_alloc = true;
        actx = ggml_init(ap);
        ggml_tensor* xin = ggml_new_tensor_2d(actx, GGML_TYPE_F32, hidden, T);
        ggml_tensor* res = ggml_new_tensor_2d(actx, GGML_TYPE_F32, hidden, T);
        ggml_tensor* it = ggml_new_tensor_2d(actx, GGML_TYPE_I32, n_used, T);
        ggml_tensor* wt = ggml_new_tensor_3d(actx, GGML_TYPE_F32, 1, n_used, T);
        abuf = ggml_backend_alloc_ctx_tensors(actx, backend);
        if (abuf == nullptr) { ggml_free(actx); throw std::runtime_error("tsm moe mm probe: activation buffer"); }
        ggml_backend_tensor_set(xin, x, 0, ggml_nbytes(xin));
        ggml_backend_tensor_set(it, ids, 0, ggml_nbytes(it));
        ggml_backend_tensor_set(wt, weights, 0, ggml_nbytes(wt));
        const bool ok = tsm::moe_mm_encode(backend, xin, it, wt, g, u, d, res, T, hidden, n_ff, n_expert, n_used, true);
        if (ok) ggml_backend_tensor_get(res, out, 0, ggml_nbytes(res));
        ggml_backend_buffer_free(abuf);
        ggml_free(actx);
        ggml_backend_buffer_free(wbuf);
        ggml_free(ctx);
        return ok ? 1 : 0;
    }
    catch (const std::exception& e)
    {
        if (abuf) ggml_backend_buffer_free(abuf);
        if (wbuf) ggml_backend_buffer_free(wbuf);
        if (ctx) ggml_free(ctx);
        tsg::set_last_error(e.what());
        return 0;
    }
}

// Test hook: the prompt expert layers that ran on moe_mm_encode in this process.
TSG_EXPORT long long TSGgml_TsmMoeMmCount() { return tsm::moe_mm_count(); }

// ---- test probe: one hyper-connection read through hc_read_encode on weights the caller gives in ggml's layout
// (w_down [n_embd * hc] x lr rows, w_inject [n_embd * hc] x hc rows, w_up [lr] x (n_embd * hc) rows, each of its own
// type: F32, F16, BF16 or Q8_0 bytes), T rows of the residual and gamma, into mixed [T][n_embd] and inject [T][hc].
// Returns 1, or 0 with the last error set.
TSG_EXPORT int TSGgml_TsmHcReadProbe(int n_embd, int hc, int lr, int down_type, const void* w_down, int inject_type,
    const void* w_inject, int up_type, const void* w_up, const float* gamma, int T, const float* res, float eps,
    float* mixed, float* inject)
{
    using namespace tsg;
    if (!ensure_backend()) return 0;
    ggml_backend_t backend = g_backend;
    ggml_backend_buffer_t buf = nullptr;
    ggml_context* ctx = nullptr;
    try
    {
        if (!tsm::hc_read_supported(backend, down_type, inject_type, up_type, n_embd, hc, lr))
            throw std::runtime_error("tsm hc read probe: the kernels do not run here or take these types and shapes");
        const int wide = n_embd * hc;
        ggml_init_params ip{};
        ip.mem_size = 16 * ggml_tensor_overhead();
        ip.no_alloc = true;
        ctx = ggml_init(ip);
        ggml_tensor* wd = ggml_new_tensor_2d(ctx, static_cast<ggml_type>(down_type), wide, lr);
        ggml_tensor* wi = ggml_new_tensor_2d(ctx, static_cast<ggml_type>(inject_type), wide, hc);
        ggml_tensor* wu = ggml_new_tensor_2d(ctx, static_cast<ggml_type>(up_type), lr, wide);
        ggml_tensor* g = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, wide);
        ggml_tensor* r = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, wide, T);
        ggml_tensor* m = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, n_embd, T);
        ggml_tensor* inj = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, hc, T);
        buf = ggml_backend_alloc_ctx_tensors(ctx, backend);
        if (buf == nullptr) throw std::runtime_error("tsm hc read probe: buffer");
        ggml_backend_tensor_set(wd, w_down, 0, ggml_nbytes(wd));
        ggml_backend_tensor_set(wi, w_inject, 0, ggml_nbytes(wi));
        ggml_backend_tensor_set(wu, w_up, 0, ggml_nbytes(wu));
        ggml_backend_tensor_set(g, gamma, 0, ggml_nbytes(g));
        ggml_backend_tensor_set(r, res, 0, ggml_nbytes(r));
        tsm::HcReadIO io;
        io.res = r; io.gamma = g; io.w_down = wd; io.w_inject = wi; io.w_up = wu; io.mixed = m; io.inject = inj;
        io.T = T; io.n_embd = n_embd; io.hc = hc; io.lr = lr; io.eps = eps;
        const bool ok = tsm::hc_read_encode(backend, io);
        if (ok)
        {
            tsm::wait_last();
            ggml_backend_tensor_get(m, mixed, 0, ggml_nbytes(m));
            ggml_backend_tensor_get(inj, inject, 0, ggml_nbytes(inj));
        }
        tsm::moe_mm_release();
        ggml_backend_buffer_free(buf);
        ggml_free(ctx);
        return ok ? 1 : 0;
    }
    catch (const std::exception& e)
    {
        if (buf) ggml_backend_buffer_free(buf);
        if (ctx) ggml_free(ctx);
        tsg::set_last_error(e.what());
        return 0;
    }
}

// Test hook: the hyper-connection reads that ran on hc_read_encode in this process.
TSG_EXPORT long long TSGgml_TsmHcReadCount() { return tsm::hc_read_count(); }

TSG_EXPORT int TSGgml_TsmMoeProbe(int gate_type, int down_type, int hidden, int n_ff, int n_expert,
    const void* gate, const void* up, const void* down, int T, int n_used, const std::int32_t* slots,
    const float* weights, const float* x, float* out)
{
    using namespace tsg;
    if (!ensure_backend()) return 0;
    ggml_backend_t backend = g_backend;
    ggml_backend_buffer_t wbuf = nullptr, abuf = nullptr;
    ggml_context* ctx = nullptr;
    try
    {
        const std::size_t gs = row_bytes(gate_type, hidden) * n_ff, ds = row_bytes(down_type, n_ff) * hidden;
        const std::size_t total = (2 * gs + ds) * static_cast<std::size_t>(n_expert);
        wbuf = ggml_backend_buft_alloc_buffer(ggml_backend_get_default_buffer_type(backend), total);
        if (wbuf == nullptr) throw std::runtime_error("tsm probe: weight buffer");
        MoeHotTier tier;
        tier.backend = backend; tier.buffer = wbuf; tier.cpu_visible = true;
        tier.type[0] = tier.type[1] = gate_type; tier.type[2] = down_type;
        tier.stride[0] = tier.stride[1] = gs; tier.stride[2] = ds;
        tier.offset[0] = 0; tier.offset[1] = gs * n_expert; tier.offset[2] = 2 * gs * n_expert;
        tier.num_experts = tier.count = tier.fixed = n_expert;
        const void* parts[3] = {gate, up, down};
        auto* base = static_cast<std::uint8_t*>(ggml_backend_buffer_get_base(wbuf));
        for (int p = 0; p < 3; ++p)
        {
            ggml_tensor view{};
            view.type = GGML_TYPE_I8;
            view.ne[0] = static_cast<std::int64_t>(tier.stride[p]) * n_expert;
            view.ne[1] = view.ne[2] = view.ne[3] = 1;
            view.nb[0] = 1; view.nb[1] = view.nb[2] = view.nb[3] = static_cast<std::size_t>(view.ne[0]);
            view.buffer = wbuf;
            view.data = base + tier.offset[p];
            ggml_backend_tensor_set(&view, parts[p], 0, static_cast<std::size_t>(view.ne[0]));
        }
        ggml_init_params ip{};
        ip.mem_size = 4 * ggml_tensor_overhead();
        ip.no_alloc = true;
        ctx = ggml_init(ip);
        ggml_tensor* in = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, hidden, T);
        ggml_tensor* res = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, hidden, T);
        abuf = ggml_backend_alloc_ctx_tensors(ctx, backend);
        if (abuf == nullptr) throw std::runtime_error("tsm probe: activation buffer");
        ggml_backend_tensor_set(in, x, 0, ggml_nbytes(in));
        if (!tsm::moe_supported(backend, tier, hidden, n_ff, T, n_used))
            throw std::runtime_error("tsm probe: the kernels do not take these formats or shapes");
        if (!tsm::moe_encode(backend, tier, in, res, hidden, n_ff, T, n_used, slots, weights)) throw std::runtime_error(g_last_error);
        {
            State& st = state();
            std::lock_guard<std::mutex> lock(st.mu);
            if (st.last != nil) [st.last waitUntilCompleted];
        }
        ggml_backend_tensor_get(res, out, 0, ggml_nbytes(res));
        ggml_backend_buffer_free(abuf);
        ggml_backend_buffer_free(wbuf);
        ggml_free(ctx);
        return 1;
    }
    catch (const std::exception& e)
    {
        if (abuf) ggml_backend_buffer_free(abuf);
        if (wbuf) ggml_backend_buffer_free(wbuf);
        if (ctx) ggml_free(ctx);
        set_last_error(e.what());
        return 0;
    }
}

// ---- test hook: seamed passes (tsm::seams_*) that ran to the end in this process.
TSG_EXPORT long long TSGgml_TsmSeamPasses()
{
    return tsm::seams_pass_count();
}

// ---- test hook: prompt GDN layers that ran on the tsm kernels in this process.
TSG_EXPORT long long TSGgml_TsmGdnPromptCount()
{
    return tsm::gdn_prompt_count();
}

// ---- test hook: prompt hyper-connection passes that ran on the tsm kernels in this process.
TSG_EXPORT long long TSGgml_TsmHcPromptCount()
{
    return tsm::hc_prompt_count();
}

// ---- test hook: prompt QSA selections that ran on the tsm kernels in this process.
TSG_EXPORT long long TSGgml_TsmQsaSelectCount()
{
    return tsm::qsa_select_count();
}

// ---- test probe: one QSA selection through qsa_select_encode on inputs the caller gives (scores F32 [B, H, T], bias
// F32 [B, T], cell_blocks I32 [cells], mask F16 bits [cells, T]); the selected mask (F16 bits) comes back. Returns 1, or
// 0 with the last error set.
TSG_EXPORT int TSGgml_TsmQsaSelectProbe(int T, int n_cells, int n_blocks, int heads, int width, const float* scores,
    const float* bias, const std::int32_t* cell_blocks, const std::uint16_t* mask_in, std::uint16_t* mask_out)
{
    using namespace tsg;
    if (!ensure_backend()) return 0;
    ggml_backend_t backend = g_backend;
    ggml_backend_buffer_t buf = nullptr;
    ggml_context* ctx = nullptr;
    try
    {
        ggml_init_params ip{};
        ip.mem_size = 8 * ggml_tensor_overhead();
        ip.no_alloc = true;
        ctx = ggml_init(ip);
        ggml_tensor* sc = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, n_blocks, heads, T);
        ggml_tensor* bi = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, n_blocks, T);
        ggml_tensor* cb = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, n_cells);
        ggml_tensor* mi = ggml_new_tensor_2d(ctx, GGML_TYPE_F16, n_cells, T);
        ggml_tensor* mo = ggml_new_tensor_2d(ctx, GGML_TYPE_F16, n_cells, T);
        buf = ggml_backend_alloc_ctx_tensors(ctx, backend);
        if (buf == nullptr) throw std::runtime_error("tsm qsa probe: buffer");
        ggml_backend_tensor_set(sc, scores, 0, ggml_nbytes(sc));
        ggml_backend_tensor_set(bi, bias, 0, ggml_nbytes(bi));
        ggml_backend_tensor_set(cb, cell_blocks, 0, ggml_nbytes(cb));
        ggml_backend_tensor_set(mi, mask_in, 0, ggml_nbytes(mi));
        if (!tsm::qsa_select_encode(backend, sc, bi, cb, mi, mo, nullptr, nullptr, T, n_cells, n_blocks, heads, width))
            throw std::runtime_error(g_last_error);
        {
            State& st = state();
            std::lock_guard<std::mutex> lock(st.mu);
            if (st.last != nil) [st.last waitUntilCompleted];
        }
        ggml_backend_tensor_get(mo, mask_out, 0, ggml_nbytes(mo));
        ggml_backend_buffer_free(buf);
        ggml_free(ctx);
        return 1;
    }
    catch (const std::exception& e)
    {
        if (buf) ggml_backend_buffer_free(buf);
        if (ctx) ggml_free(ctx);
        set_last_error(e.what());
        return 0;
    }
}

// ---- test hook: prompt QSA attention passes that ran on the tsm kernels in this process.
TSG_EXPORT long long TSGgml_TsmQsaAttnCount()
{
    return tsm::qsa_attn_count();
}

// ---- test probe: one QSA attention over the chosen cells (qsa_attn_encode) on inputs the caller gives - q F32
// [256, T, H], k and v F16 bits [256, cells, kv heads], idx I32 [width, T], cnt I32 [T], mask F16 bits [cells, T] -
// into out F32 [256, H, T]. With iterations > 0 it then times that many back-to-back passes (*ms_per_pass). Returns 1,
// or 0 with the last error set.
TSG_EXPORT int TSGgml_TsmQsaAttnProbe(int T, int n_head, int n_head_kv, int n_cells, int width, float scale,
    const float* q, const std::uint16_t* k, const std::uint16_t* v, const std::int32_t* idx, const std::int32_t* cnt,
    const std::uint16_t* mask, float* out, int iterations, double* ms_per_pass)
{
    using namespace tsg;
    if (!ensure_backend()) return 0;
    ggml_backend_t backend = g_backend;
    ggml_backend_buffer_t buf = nullptr;
    ggml_context* ctx = nullptr;
    try
    {
        ggml_init_params ip{};
        ip.mem_size = 10 * ggml_tensor_overhead();
        ip.no_alloc = true;
        ctx = ggml_init(ip);
        ggml_tensor* tq = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, 256, T, n_head);
        ggml_tensor* tk = ggml_new_tensor_3d(ctx, GGML_TYPE_F16, 256, n_cells, n_head_kv);
        ggml_tensor* tv = ggml_new_tensor_3d(ctx, GGML_TYPE_F16, 256, n_cells, n_head_kv);
        ggml_tensor* ti = ggml_new_tensor_2d(ctx, GGML_TYPE_I32, width, T);
        ggml_tensor* tc = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, T);
        ggml_tensor* tm = ggml_new_tensor_2d(ctx, GGML_TYPE_F16, n_cells, T);
        ggml_tensor* to = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, 256, n_head, T);
        buf = ggml_backend_alloc_ctx_tensors(ctx, backend);
        if (buf == nullptr) throw std::runtime_error("tsm qsa attn probe: buffer");
        ggml_backend_tensor_set(tq, q, 0, ggml_nbytes(tq));
        ggml_backend_tensor_set(tk, k, 0, ggml_nbytes(tk));
        ggml_backend_tensor_set(tv, v, 0, ggml_nbytes(tv));
        ggml_backend_tensor_set(ti, idx, 0, ggml_nbytes(ti));
        ggml_backend_tensor_set(tc, cnt, 0, ggml_nbytes(tc));
        ggml_backend_tensor_set(tm, mask, 0, ggml_nbytes(tm));
        auto wait_last = [] {
            State& st = state();
            std::lock_guard<std::mutex> lock(st.mu);
            if (st.last != nil) [st.last waitUntilCompleted];
        };
        if (!tsm::qsa_attn_encode(backend, tq, tk, tv, ti, tc, tm, to, T, n_head, n_head_kv, n_cells, width, scale))
            throw std::runtime_error(g_last_error);
        wait_last();
        ggml_backend_tensor_get(to, out, 0, ggml_nbytes(to));
        if (iterations > 0 && ms_per_pass != nullptr)
        {
            const auto t0 = std::chrono::steady_clock::now();
            for (int i = 0; i < iterations; ++i)
                if (!tsm::qsa_attn_encode(backend, tq, tk, tv, ti, tc, tm, to, T, n_head, n_head_kv, n_cells, width, scale))
                    throw std::runtime_error(g_last_error);
            wait_last();
            *ms_per_pass = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count() / iterations;
        }
        ggml_backend_buffer_free(buf);
        ggml_free(ctx);
        return 1;
    }
    catch (const std::exception& e)
    {
        if (buf) ggml_backend_buffer_free(buf);
        if (ctx) ggml_free(ctx);
        set_last_error(e.what());
        return 0;
    }
}
