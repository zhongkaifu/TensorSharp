// Copyright (c) Zhongkai Fu. All rights reserved.
// https://github.com/zhongkaifu/TensorSharp
//
// This file is part of TensorSharp.
//
// TensorSharp is licensed under the BSD-3-Clause license found in the LICENSE file in the root directory of this source tree.
//
// Inkling 2's routed experts' products for prompt chunks on FP4 tensor cores (W4A4, NVFP4 weights as ggml's block_nvfp4
// rows): the pairs (row, slot) grouped by expert into tiles of 64 (a tile list built on the device), the activations
// quantized per row to NVFP4 (a row scale amax / (6 * 448), a UE4M3 scale per 16 values, the nibbles interleaved as the
// weights' blocks), and each block one tile's 64 pairs x 128 of the expert's output rows over K in 64-value steps through a
// cp.async pipeline, mma.sync m16n8k64 kind::mxf4nvf4 with the block scales in hardware. No includes of its own.
#pragma once

    // A stage: two 64-value blocks (72 bytes) of each of the tile's 64 pair rows (A) and 128 weight rows (B), a shared row of
    // 20 words (18 used: the padding spreads the fragments' loads over the banks).
#ifndef TSG_F4_PAIRS
#define TSG_F4_PAIRS 128
#endif
    // A tile's pairs (64 or 128) x 128 output rows, a warp 32 x 32: (pairs / 32) x 4 warps.
    constexpr int kF4Pairs = TSG_F4_PAIRS, kF4Rows = 128, kF4Stages = 3, kF4KB = 2, kF4RowWords = 20;
    // The GEMM's grid for n output rows over at most max_tiles tiles: the n-blocks of a tile side by side (its activations
    // shared; an expert's tiles one after another read its weight block from L2 or DRAM again: tile-major or expert-major
    // orders were 2-7% faster for gate/up and slower or no faster for down, NOTES.md).
    inline dim3 f4_grid(int n, unsigned int max_tiles) { return dim3((unsigned int)(n / kF4Rows), max_tiles); }
    constexpr int kF4WarpsM = kF4Pairs / 32, kF4Threads = kF4WarpsM * (kF4Rows / 32) * 32;
    constexpr int kF4A = kF4Pairs * kF4RowWords, kF4B = kF4Rows * kF4RowWords;
    constexpr int kF4Stage = kF4A + kF4B;   // uint32 words a stage
    constexpr size_t kF4Smem = sizeof(uint32_t) * kF4Stage * kF4Stages;

    struct Fp4Args
    {
        const char* w; long long w_row, w_expert;   // block_nvfp4 rows [e][n][k / 64]
        const char* aq; long long a_row;            // activations as block_nvfp4 rows [acts][k / 64], a_row bytes apart
        const float* arow;                          // [acts] row scales
        const int* tiles;                           // [tile][3]: expert, first in list, pairs (<= 64)
        const int* ntiles;
        const int* list;                            // the pairs, grouped by expert (t used + j)
        int used, acts, k, n;
        float* y;                                   // [pairs][n]
    };

    __device__ __forceinline__ void f4_cp16(void* smem, const void* gmem, int bytes)
    {
        const unsigned s = (unsigned)__cvta_generic_to_shared(smem);
        asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;\n" :: "r"(s), "l"(gmem), "r"(bytes));
    }

    __device__ __forceinline__ void f4_cp4(void* smem, const void* gmem, int bytes)
    {
        const unsigned s = (unsigned)__cvta_generic_to_shared(smem);
        asm volatile("cp.async.ca.shared.global [%0], [%1], 4, %2;\n" :: "r"(s), "l"(gmem), "r"(bytes));
    }

    __device__ __forceinline__ void f4_cp8(void* smem, const void* gmem, int bytes)
    {
        const unsigned s = (unsigned)__cvta_generic_to_shared(smem);
        asm volatile("cp.async.ca.shared.global [%0], [%1], 8, %2;\n" :: "r"(s), "l"(gmem), "r"(bytes));
    }

    __device__ __forceinline__ void f4_commit() { asm volatile("cp.async.commit_group;\n" ::); }

    template <int N>
    __device__ __forceinline__ void f4_wait() { asm volatile("cp.async.wait_group %0;\n" :: "n"(N)); }

    __device__ __forceinline__ void f4_ldsm_x4(uint32_t (&r)[4], const void* smem)
    {
        const unsigned s = (unsigned)__cvta_generic_to_shared(smem);
        asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                     : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3]) : "r"(s));
    }

    __device__ __forceinline__ void f4_ldsm_x2(uint32_t& r0, uint32_t& r1, const void* smem)
    {
        const unsigned s = (unsigned)__cvta_generic_to_shared(smem);
        asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0, %1}, [%2];\n" : "=r"(r0), "=r"(r1) : "r"(s));
    }

    // The block-scaled FP4 MMA exists on the architecture-specific ("a") targets only.
#if defined(__CUDA_ARCH_FEAT_SM120_ALL) || defined(__CUDA_ARCH_FEAT_SM121_ALL)
#define TSG_F4_MMA 1
#else
#define TSG_F4_MMA 0
#endif

    __global__ void inkling_f4_probe(int* out) { *out = TSG_F4_MMA; }

    __device__ __forceinline__ void f4_mma(float (&d)[4], const uint32_t (&a)[4], uint32_t b0, uint32_t b1, uint32_t sa, uint32_t sb)
    {
#if TSG_F4_MMA
        asm volatile(
            "mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X.m16n8k64.row.col.f32.e2m1.e2m1.f32.ue4m3 "
            "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3}, %10, {0, 0}, %11, {0, 0};"
            : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
            : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1), "r"(sa), "r"(sb));
#else
        __trap();
#endif
    }

    // The tiles: every distinct expert's pairs (moe groups) in runs of kF4Pairs. One thread (at most 128 experts).
    __global__ void inkling_f4_tiles(const int* __restrict__ groups, int cap, int* __restrict__ tiles, int* __restrict__ ntiles)
    {
        if (threadIdx.x != 0 || blockIdx.x != 0) return;
        const int n = groups[0];
        const int* distinct = groups + 1;
        const int* start = groups + 1 + cap;
        int t = 0;
        for (int u = 0; u < n; u++)
        {
            const int s0 = start[u], s1 = start[u + 1];
            for (int p = s0; p < s1; p += kF4Pairs)
            {
                tiles[3 * t] = distinct[u];
                tiles[3 * t + 1] = p;
                tiles[3 * t + 2] = min(kF4Pairs, s1 - p);
                t++;
            }
        }
        *ntiles = t;
    }

    // UE4M3 from a positive scale (round to nearest, saturating at 448).
    __device__ __forceinline__ uint8_t f4_ue4m3(float v)
    {
        const __nv_fp8_e4m3 f(v);
        return (uint8_t)(f.__x & 0x7F);
    }

    __device__ __forceinline__ float f4_ue4m3_value(uint8_t code)
    {
        __nv_fp8_e4m3 f;
        f.__x = code;
        return (float)f;
    }

    // Squared error of 16 values (already over the row scale) quantized at a sub-scale.
    __device__ __forceinline__ float f4_error(const float (&v)[16], float scale)
    {
        if (scale <= 0.0f) return 3.0e38f;
        const float inv = 1.0f / scale;
        const float table[16] = { 0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 3.0f, 4.0f, 6.0f, -0.0f, -0.5f, -1.0f, -1.5f, -2.0f, -3.0f, -4.0f, -6.0f };
        float err = 0.0f;
#pragma unroll
        for (int i = 0; i < 16; i += 4)
        {
            const __nv_fp4x4_e2m1 q(make_float4(v[i] * inv, v[i + 1] * inv, v[i + 2] * inv, v[i + 3] * inv));
            const unsigned bits = (unsigned)q.__x;
#pragma unroll
            for (int c = 0; c < 4; c++)
            {
                const float d = v[i + c] - table[(bits >> (4 * c)) & 0xF] * scale;
                err += d * d;
            }
        }
        return err;
    }

    // One activation row a block: its row scale, then each 16 values' UE4M3 scale (the best of the nearest code and its
    // neighbours) and nibbles (bytes j of the sub-block's two words: values j, j + 8 and 4 + j, 12 + j).
    // One warp an activation row (kF4QuantRows rows a block): the row's amax, then its 16-value sub-blocks a lane each.
    constexpr int kF4QuantRows = 8;

    __global__ void __launch_bounds__(32 * kF4QuantRows) inkling_f4_quantize(const float* __restrict__ x, long long x_row, long long x_tok,
        int acts, int k, char* __restrict__ aq, long long a_row, float* __restrict__ arow, long long rows_total)
    {
        const long long act = (long long)blockIdx.x * kF4QuantRows + (threadIdx.x >> 5);
        if (act >= rows_total) return;
        const int lane = threadIdx.x & 31;
        const long long t = act / acts, j = act - t * acts;
        const float* xr = x + t * x_tok + j * x_row;
        float amax = 0.0f;
        for (int i = lane * 4; i < k; i += 128)
        {
            const float4 v = *reinterpret_cast<const float4*>(xr + i);
            amax = fmaxf(amax, fmaxf(fmaxf(fabsf(v.x), fabsf(v.y)), fmaxf(fabsf(v.z), fabsf(v.w))));
        }
#pragma unroll
        for (int o = 16; o > 0; o >>= 1) amax = fmaxf(amax, __shfl_xor_sync(0xffffffffu, amax, o));
        const float row_scale = amax / (6.0f * 448.0f);
        if (lane == 0) arow[act] = row_scale;
        const float inv_row = row_scale > 0.0f ? 1.0f / row_scale : 0.0f;
        char* dst_row = aq + act * a_row;
        for (int sb = lane; sb < k / 16; sb += 32)
        {
            float v[16];
            float amax_sub = 0.0f;
#pragma unroll
            for (int i = 0; i < 16; i++) { v[i] = xr[sb * 16 + i] * inv_row; amax_sub = fmaxf(amax_sub, fabsf(v[i])); }
            int code = f4_ue4m3(amax_sub / 6.0f);
            float best = f4_error(v, f4_ue4m3_value((uint8_t)code));
#pragma unroll
            for (int dlt = -1; dlt <= 1; dlt += 2)
            {
                const int c = code + dlt;
                if (c < 1 || c > 0x7E) continue;
                const float e = f4_error(v, f4_ue4m3_value((uint8_t)c));
                if (e < best) { best = e; code = c; }
            }
            const float scale = f4_ue4m3_value((uint8_t)code);
            const float inv = scale > 0.0f ? 1.0f / scale : 0.0f;
            const __nv_fp4x4_e2m1 q0l(make_float4(v[0] * inv, v[8] * inv, v[1] * inv, v[9] * inv));
            const __nv_fp4x4_e2m1 q0h(make_float4(v[2] * inv, v[10] * inv, v[3] * inv, v[11] * inv));
            const __nv_fp4x4_e2m1 q1l(make_float4(v[4] * inv, v[12] * inv, v[5] * inv, v[13] * inv));
            const __nv_fp4x4_e2m1 q1h(make_float4(v[6] * inv, v[14] * inv, v[7] * inv, v[15] * inv));
            const uint32_t w0 = (uint32_t)q0l.__x | ((uint32_t)q0h.__x << 16);
            const uint32_t w1 = (uint32_t)q1l.__x | ((uint32_t)q1h.__x << 16);
            // block kb: d[4] then qs[32] (the sub-block's two words at 4 + 8 sub).
            const int kb = sb >> 2, sub = sb & 3;
            char* blk = dst_row + kb * 36;
            reinterpret_cast<uint32_t*>(blk + 4 + sub * 8)[0] = w0;
            reinterpret_cast<uint32_t*>(blk + 4 + sub * 8)[1] = w1;
            reinterpret_cast<uint8_t*>(blk)[sub] = (uint8_t)code;
        }
    }

    // A block: tile blockIdx.y's pairs (A, 64 rows) x the expert's output rows [128 blockIdx.x, + 128) (B), K in stages of two
    // 64-value blocks. Warp (wm, wn): pairs [32 wm, + 32) (2 m tiles) x rows [32 wn, + 32) (4 n tiles). A fragment's word i of
    // a block's qs is the row's word 1 + i of the block (after its 4 scale bytes), the row's scales word 0.
    __global__ void __launch_bounds__(kF4Threads) inkling_f4_gemm(Fp4Args a)
    {
        extern __shared__ __align__(16) uint32_t f4_sm[];
        __shared__ int s_act[kF4Pairs], s_pair[kF4Pairs];
        const int tile = blockIdx.y, nblock = blockIdx.x;
        if (tile >= *a.ntiles) return;
        const int e = a.tiles[3 * tile], p0 = a.tiles[3 * tile + 1], np = a.tiles[3 * tile + 2];
        const int n0 = nblock * kF4Rows;
        const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31, g = lane >> 2, tig = lane & 3;
        const int wm = warp % kF4WarpsM, wn = warp / kF4WarpsM;
        if (tid < kF4Pairs)
        {
            const bool in = tid < np;
            const int pair = in ? a.list[p0 + tid] : 0;
            const int t = pair / a.used, j = pair - t * a.used;
            s_pair[tid] = in ? pair : -1;
            s_act[tid] = in ? t * a.acts + (a.acts == 1 ? 0 : j) : -1;
        }
        __syncthreads();
        const int stages = a.k / (64 * kF4KB);
        const char* wbase = a.w + (long long)e * a.w_expert + (long long)n0 * a.w_row;
        auto load = [&](int st, int buf)
        {
            uint32_t* s = f4_sm + buf * kF4Stage;
            const long long off = (long long)st * kF4KB * 36;
            for (int i = tid; i < kF4Pairs * 9; i += kF4Threads)
            {
                const int r = i / 9, c = i - r * 9, act = s_act[r];
                const char* src = act >= 0 ? a.aq + act * a.a_row + off + c * 8 : a.aq;
                f4_cp8(s + r * kF4RowWords + c * 2, src, act >= 0 ? 8 : 0);
            }
            for (int i = tid; i < kF4Rows * 9; i += kF4Threads)
            {
                const int r = i / 9, c = i - r * 9;
                f4_cp8(s + kF4A + r * kF4RowWords + c * 2, wbase + (long long)r * a.w_row + off + c * 8, 8);
            }
        };
        float acc[2][4][4];
#pragma unroll
        for (int mt = 0; mt < 2; mt++)
#pragma unroll
            for (int nt = 0; nt < 4; nt++)
#pragma unroll
                for (int c = 0; c < 4; c++) acc[mt][nt][c] = 0.0f;
#pragma unroll
        for (int st = 0; st < kF4Stages - 1; st++)
        {
            if (st < stages) load(st, st);
            f4_commit();
        }
        for (int st = 0; st < stages; st++)
        {
            f4_wait<kF4Stages - 2>();
            __syncthreads();
            {
                const int next = st + kF4Stages - 1;
                if (next < stages) load(next, next % kF4Stages);
                f4_commit();
            }
            const uint32_t* s = f4_sm + (st % kF4Stages) * kF4Stage;
#pragma unroll
            for (int kb = 0; kb < kF4KB; kb++)
            {
                const int w0 = kb * 9;   // the block's scale word; its qs words w0 + 1 .. w0 + 8
                uint32_t A[2][4], sa[2];
#pragma unroll
                for (int mt = 0; mt < 2; mt++)
                {
                    const uint32_t* r0 = s + (wm * 32 + mt * 16 + g) * kF4RowWords + w0 + 1;
                    const uint32_t* r8 = r0 + 8 * kF4RowWords;
                    A[mt][0] = r0[tig];
                    A[mt][1] = r8[tig];
                    A[mt][2] = r0[tig + 4];
                    A[mt][3] = r8[tig + 4];
                    sa[mt] = s[(wm * 32 + mt * 16 + (lane >> 2) + (lane & 1) * 8) * kF4RowWords + w0];
                }
#pragma unroll
                for (int nt = 0; nt < 4; nt++)
                {
                    const uint32_t* rb = s + kF4A + (wn * 32 + nt * 8 + g) * kF4RowWords + w0;
                    const uint32_t b0 = rb[1 + tig], b1 = rb[5 + tig], sb = rb[0];
#pragma unroll
                    for (int mt = 0; mt < 2; mt++) f4_mma(acc[mt][nt], A[mt], b0, b1, sa[mt], sb);
                }
            }
        }
        // y[pair][row] = acc x the pair's row scale.
#pragma unroll
        for (int mt = 0; mt < 2; mt++)
#pragma unroll
            for (int h = 0; h < 2; h++)
            {
                const int pr = wm * 32 + mt * 16 + g + h * 8;
                if (pr >= np) continue;
                const float rs = a.arow[s_act[pr]];
                float* yr = a.y + (long long)s_pair[pr] * a.n + n0;
#pragma unroll
                for (int nt = 0; nt < 4; nt++)
                {
                    const int col = wn * 32 + nt * 8 + tig * 2;
                    *reinterpret_cast<float2*>(yr + col) = make_float2(acc[mt][nt][h * 2] * rs, acc[mt][nt][h * 2 + 1] * rs);
                }
            }
    }
