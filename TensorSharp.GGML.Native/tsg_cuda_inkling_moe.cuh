// Copyright (c) Zhongkai Fu. All rights reserved.
// https://github.com/zhongkaifu/TensorSharp
//
// This file is part of TensorSharp.
//
// TensorSharp is licensed under the BSD-3-Clause license found in the LICENSE file in the root directory of this source tree.
//
// Inkling 2's routed experts' products for a few rows (TSG_INKLING_EXPERTS), included by tsg_cuda_inkling_fused.cu inside
// its anonymous namespace (after its blocks() helper); no includes of its own.
#pragma once

    // ---- the routed experts' products for a few rows (decode, verify windows). The pairs (row t, slot j) are grouped by
    // expert, so a block (an expert the rows picked, a tile of its output rows) reads the expert's rows once for all of its
    // pairs; ggml's small-batch kernel gives every pair its own warp, which reads the expert again. The activations go to
    // int8 in blocks of 32 (as ggml's q8_1, with an F32 scale) and the products run on dp4a.
    constexpr int kMoeThreads = 256;
    // The v3 product's block: its warps' rows share the block's staged activations (512 or 128 threads were slower).
    constexpr int kMoe3Threads = 256;
#ifndef TSG_MOE_R8
#define TSG_MOE_R8 8
#endif
    // A warp's rows by the pair cap (8, 16, 32): kept small enough that the accumulators stay in registers.
    template <int P> __host__ __device__ constexpr int moe_rows_per_warp() { return P <= 8 ? TSG_MOE_R8 : (P <= 16 ? 4 : 2); }

    struct MoeArgs
    {
        const char* w; long long w_row, w_expert;     // experts [k, m, e]: a row's and an expert's bytes
        const float* x; long long x_nb1, x_nb2;       // [k, acts, rows]
        const int32_t* ids; long long ids_nb1;        // [used, rows]
        float* dst;                                   // [m, used, rows]
        int k, m, experts, used, rows, acts;          // acts: x's ne1 (1: shared by the slots, else used)
        int* groups;                                  // [0] distinct experts, [1, 1 + cap) their ids, then cap + 1 starts, then the pairs
        int8_t* q; float* qd;                         // the activations [acts rows][k], their scales [acts rows][k / 32]
        int cap;                                      // most distinct experts: min(experts, used rows)
        int chunk0;                                   // the single pass: an expert's pairs from this one (0, P, 2 P, ...)
    };

    __global__ void inkling_moe_group(MoeArgs a)
    {
        extern __shared__ int sh[];
        int* count = sh;                    // [experts]
        int* slot = sh + a.experts;         // [experts]: an expert's place among the distinct ones
        const int pairs = a.used * a.rows;
        for (int e = threadIdx.x; e < a.experts; e += blockDim.x) count[e] = 0;
        __syncthreads();
        for (int p = threadIdx.x; p < pairs; p += blockDim.x)
        {
            const int t = p / a.used, j = p - t * a.used;
            const int e = a.ids[(long long)t * a.ids_nb1 / 4 + j];
            if (e >= 0 && e < a.experts) atomicAdd(&count[e], 1);
        }
        __syncthreads();
        int* distinct = a.groups + 1;
        int* start = a.groups + 1 + a.cap;
        int* list = start + a.cap + 1;
        if (threadIdx.x == 0)
        {
            int n = 0, at = 0;
            for (int e = 0; e < a.experts; e++)
            {
                if (count[e] == 0) continue;
                distinct[n] = e;
                start[n] = at;
                slot[e] = n;
                at += count[e];
                count[e] = start[n];        // now the next free place in its list
                n++;
            }
            start[n] = at;
            a.groups[0] = n;
        }
        __syncthreads();
        for (int p = threadIdx.x; p < pairs; p += blockDim.x)
        {
            const int t = p / a.used, j = p - t * a.used;
            const int e = a.ids[(long long)t * a.ids_nb1 / 4 + j];
            if (e >= 0 && e < a.experts) list[atomicAdd(&count[e], 1)] = p;
        }
    }

    // One warp a block of 32 activations: d = amax / 127, q = round(x / d).
    __global__ void inkling_moe_quantize(MoeArgs a)
    {
        const int lane = threadIdx.x & 31;
        const long long blk = (long long)blockIdx.x * (blockDim.x / 32) + (threadIdx.x >> 5);
        const int per = a.k / 32;
        if (blk >= (long long)a.acts * a.rows * per) return;
        const int act = (int)(blk / per), b = (int)(blk - (long long)act * per);
        const int t = act / a.acts, j = act - t * a.acts;
        const float v = a.x[((long long)t * a.x_nb2 + (long long)j * a.x_nb1) / 4 + b * 32 + lane];
        float amax = fabsf(v);
#pragma unroll
        for (int o = 16; o > 0; o >>= 1) amax = fmaxf(amax, __shfl_xor_sync(0xffffffffu, amax, o));
        const float d = amax / 127.0f;
        a.q[(long long)act * a.k + b * 32 + lane] = (int8_t)(d > 0.0f ? __float2int_rn(v / d) : 0);
        if (lane == 0) a.qd[(long long)act * per + b] = d;
    }

    __device__ __forceinline__ float moe_ue4m3(uint8_t x)
    {
        // The UE4M3 scale halved (the E2M1 table is doubled), 0x7F (NaN) as 0, as ggml's.
        if (x == 0 || x == 0x7F) return 0.0f;
        const int exp = (x >> 3) & 0xF, man = x & 0x7;
        const float v = exp == 0 ? ldexpf((float)man, -9) : ldexpf(1.0f + man / 8.0f, exp - 7);
        return 0.5f * v;
    }

    __device__ __forceinline__ int2 moe_table16(uint32_t q4, const uint32_t* t32)
    {
        // Eight 4-bit indices -> the low nibbles' four int8 values (x) and the high nibbles' (y), by byte permutes (ggml's
        // get_int_from_table_16): the low 3 bits pick from each half of the table, the 4th between the halves.
        uint32_t tmp[2];
        const uint32_t select = 0x32103210u | ((q4 & 0x88888888u) >> 1);
#pragma unroll
        for (uint32_t i = 0; i < 2; ++i)
        {
            const uint32_t shift = 16 * i;
            const uint32_t low = __byte_perm(t32[0], t32[1], q4 >> shift);
            const uint32_t high = __byte_perm(t32[2], t32[3], q4 >> shift);
            tmp[i] = __byte_perm(low, high, select >> shift);
        }
        return make_int2((int)__byte_perm(tmp[0], tmp[1], 0x6420), (int)__byte_perm(tmp[0], tmp[1], 0x7531));
    }

    // The E2M1 values doubled and IQ4_NL's, four to a word.
    __constant__ uint32_t kMoeFp4[4] = { 0x03020100u, 0x0C080604u, 0xFDFEFF00u, 0xF4F8FAFCu };
    __constant__ uint32_t kMoeIq4nl[4] = { 0xBFAD9881u, 0xF6EADDCFu, 0x26190D01u, 0x71594535u };

    // A unit is 8 of a row's values: where in the row its weights and scale are, the elements its low 4 start at (its
    // high 4 at + gap). The same for every row.
    template <int TYPE>
    __device__ __forceinline__ void moe_unit_place(int u, int& wo, int& so, int& at, int& gap)
    {
        if constexpr (TYPE == GGML_TYPE_NVFP4)
        {
            // block (36 bytes): d[4] UE4M3 then qs[32]; qs[s * 8 + j] = element s * 16 + j (low), + 8 (high).
            const int b = u >> 3, sb = (u & 7) >> 1, h = u & 1;
            wo = b * 36 + 4 + sb * 8 + h * 4; so = b * 36 + sb; at = b * 64 + sb * 16 + h * 4; gap = 8;
        }
        else if constexpr (TYPE == GGML_TYPE_IQ4_NL)
        {
            // block (18 bytes): half d, qs[16]; qs[j] = element j (low), j + 16 (high).
            const int b = u >> 2, j0 = (u & 3) * 4;
            wo = b * 18 + 2 + j0; so = b * 18; at = b * 32 + j0; gap = 16;
        }
        else
        {
            // Q8_0 block (34 bytes): half d, qs[32].
            const int b = u >> 2, j0 = (u & 3) * 8;
            wo = b * 34 + 2 + j0; so = b * 34; at = b * 32 + j0; gap = 4;
        }
    }

    // A unit's int8 weights (low 4, high 4) and scale from its row.
    template <int TYPE>
    __device__ __forceinline__ void moe_unit_load(const char* row, int wo, int so, int2& w, float& dw, const uint32_t* fp4, const uint32_t* iq4)
    {
        if constexpr (TYPE == GGML_TYPE_NVFP4)
        {
            w = moe_table16(__ldg(reinterpret_cast<const uint32_t*>(row + wo)), fp4);
            dw = moe_ue4m3((uint8_t)__ldg(reinterpret_cast<const unsigned char*>(row + so)));
        }
        else if constexpr (TYPE == GGML_TYPE_IQ4_NL)
        {
            const uint16_t* q = reinterpret_cast<const uint16_t*>(row + wo);
            w = moe_table16((uint32_t)__ldg(q) | ((uint32_t)__ldg(q + 1) << 16), iq4);
            dw = __half2float(__ldg(reinterpret_cast<const half*>(row + so)));
        }
        else
        {
            const uint16_t* q = reinterpret_cast<const uint16_t*>(row + wo);
            w = make_int2((int)((uint32_t)__ldg(q) | ((uint32_t)__ldg(q + 1) << 16)), (int)((uint32_t)__ldg(q + 2) | ((uint32_t)__ldg(q + 3) << 16)));
            dw = __half2float(__ldg(reinterpret_cast<const half*>(row + so)));
        }
    }

    // A block: one of the distinct experts (blockIdx.y), kMoeRows of its output rows (blockIdx.x). A warp's rows, a lane
    // every 32nd unit of them: the unit's activations (every pair's) loaded once, then each row's weights against them.
    template <int TYPE, int P>
    __global__ void __launch_bounds__(kMoeThreads) inkling_moe_product(MoeArgs a)
    {
        const int u_index = blockIdx.y;
        if (u_index >= a.groups[0]) return;
        const int* start = a.groups + 1 + a.cap;
        const int* list = start + a.cap + 1;
        const int e = a.groups[1 + u_index];
        const int first = start[u_index], np = min(start[u_index + 1] - first, P);
        const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
        constexpr int R = moe_rows_per_warp<P>();
        const int row0 = blockIdx.x * (kMoeThreads / 32 * R) + warp * R;
        if (row0 >= a.m) return;
        const int rows_here = min(R, a.m - row0);
        int pair[P], act[P];
        const int per = a.k / 32;
#pragma unroll
        for (int p = 0; p < P; p++)
        {
            pair[p] = p < np ? list[first + p] : 0;
            const int t = pair[p] / a.used, j = pair[p] - t * a.used;
            act[p] = t * a.acts + (a.acts == 1 ? 0 : j);
        }
        const uint32_t fp4[4] = { kMoeFp4[0], kMoeFp4[1], kMoeFp4[2], kMoeFp4[3] };
        const uint32_t iq4[4] = { kMoeIq4nl[0], kMoeIq4nl[1], kMoeIq4nl[2], kMoeIq4nl[3] };
        float acc[R][P];
#pragma unroll
        for (int r = 0; r < R; r++)
#pragma unroll
            for (int p = 0; p < P; p++) acc[r][p] = 0.0f;
        const int units = a.k / 8;
        const char* base = a.w + (long long)e * a.w_expert + (long long)row0 * a.w_row;
        for (int u = lane; u < units; u += 32)
        {
            int wo, so, at, gap;
            moe_unit_place<TYPE>(u, wo, so, at, gap);
            int xlo[P], xhi[P];
            float dx[P];
#pragma unroll
            for (int p = 0; p < P; p++)
            {
                if (p < np)
                {
                    const int8_t* xq = a.q + (long long)act[p] * a.k;
                    xlo[p] = *reinterpret_cast<const int*>(xq + at);
                    xhi[p] = *reinterpret_cast<const int*>(xq + at + gap);
                    dx[p] = a.qd[(long long)act[p] * per + (at >> 5)];
                }
            }
#pragma unroll
            for (int r = 0; r < R; r++)
            {
                if (r >= rows_here) break;
                int2 w; float dw;
                moe_unit_load<TYPE>(base + (long long)r * a.w_row, wo, so, w, dw, fp4, iq4);
#pragma unroll
                for (int p = 0; p < P; p++)
                {
                    if (p >= np) break;
                    const int sumi = __dp4a(w.y, xhi[p], __dp4a(w.x, xlo[p], 0));
                    acc[r][p] = fmaf((float)sumi, dw * dx[p], acc[r][p]);
                }
            }
        }
#pragma unroll
        for (int r = 0; r < R; r++)
        {
            if (r >= rows_here) break;
#pragma unroll
            for (int p = 0; p < P; p++)
            {
                if (p >= np) break;
                float v = acc[r][p];
#pragma unroll
                for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
                if (lane == 0) a.dst[(long long)pair[p] * a.m + row0 + r] = v;
            }
        }
    }

    // A unit's raw weight words and scale bits (loaded ahead of their use).
    template <int TYPE>
    __device__ __forceinline__ void moe_unit_raw(const char* row, int wo, int so, uint32_t& q0, uint32_t& q1, uint32_t& sc)
    {
        if constexpr (TYPE == GGML_TYPE_NVFP4)
        {
            q0 = __ldg(reinterpret_cast<const uint32_t*>(row + wo));
            q1 = 0;
            sc = __ldg(reinterpret_cast<const unsigned char*>(row + so));
        }
        else if constexpr (TYPE == GGML_TYPE_IQ4_NL)
        {
            const uint16_t* q = reinterpret_cast<const uint16_t*>(row + wo);
            q0 = (uint32_t)__ldg(q) | ((uint32_t)__ldg(q + 1) << 16);
            q1 = 0;
            sc = __ldg(reinterpret_cast<const unsigned short*>(row + so));
        }
        else
        {
            const uint16_t* q = reinterpret_cast<const uint16_t*>(row + wo);
            q0 = (uint32_t)__ldg(q) | ((uint32_t)__ldg(q + 1) << 16);
            q1 = (uint32_t)__ldg(q + 2) | ((uint32_t)__ldg(q + 3) << 16);
            sc = __ldg(reinterpret_cast<const unsigned short*>(row + so));
        }
    }

    // The raw words as int8 weights and the scale.
    template <int TYPE>
    __device__ __forceinline__ void moe_unit_decode(uint32_t q0, uint32_t q1, uint32_t sc, int2& w, float& dw, const uint32_t* fp4, const uint32_t* iq4)
    {
        if constexpr (TYPE == GGML_TYPE_NVFP4) { w = moe_table16(q0, fp4); dw = moe_ue4m3((uint8_t)sc); }
        else if constexpr (TYPE == GGML_TYPE_IQ4_NL) { w = moe_table16(q0, iq4); dw = __half2float(__ushort_as_half((unsigned short)sc)); }
        else { w = make_int2((int)q0, (int)q1); dw = __half2float(__ushort_as_half((unsigned short)sc)); }
    }

    // A block: one of the distinct experts (blockIdx.y), 8 R of its output rows (blockIdx.x), a warp R of them, a lane every
    // 32nd unit (U at most). Every weight a lane takes is loaded first (all of the warp's loads in flight at once); the pairs'
    // int8 activations go to shared memory meanwhile; then each unit's activations are read once for the warp's R rows.
    // CHUNKED (rows past P): an expert's pairs in chunks of P; else at most P pairs, one pass (no loop: the loop costs ~170
    // registers more and halved the 8-row kernel's speed).
    template <int TYPE, int P, int R, int U, bool CHUNKED>
    __global__ void __launch_bounds__(kMoe3Threads) inkling_moe_product3(MoeArgs a)
    {
        extern __shared__ __align__(16) unsigned char moe_sm[];
        const int u_index = blockIdx.y;
        if (u_index >= a.groups[0]) return;
        const int* start = a.groups + 1 + a.cap;
        const int* list = start + a.cap + 1;
        const int e = a.groups[1 + u_index];
        const int first = start[u_index], total = start[u_index + 1] - first;
        const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
        const int row0 = blockIdx.x * (kMoe3Threads / 32 * R) + warp * R;
        const int units = a.k / 8, per = a.k / 32;
        const char* base = a.w + (long long)e * a.w_expert;

        // 2.-4. The pairs in chunks of P: each chunk's activations and scales into shared memory, its products, its outputs.
        int8_t* sq = reinterpret_cast<int8_t*>(moe_sm);                  // [P][k]
        float* sd = reinterpret_cast<float*>(moe_sm + (size_t)P * a.k);  // [P][k / 32]
        int* sp = reinterpret_cast<int*>(sd + (size_t)P * per);           // [P]
        const int chunks = a.k / 16;
        const uint32_t fp4[4] = { kMoeFp4[0], kMoeFp4[1], kMoeFp4[2], kMoeFp4[3] };
        const uint32_t iq4[4] = { kMoeIq4nl[0], kMoeIq4nl[1], kMoeIq4nl[2], kMoeIq4nl[3] };
#pragma unroll 1
        if (!CHUNKED && a.chunk0 >= total) return;   // a later pass: this expert has no pairs past it
#pragma unroll 1
        for (int c0 = CHUNKED ? 0 : a.chunk0; CHUNKED ? c0 < total : c0 == a.chunk0; c0 += P)
        {
            const int np = min(P, total - c0);
            if (c0 > 0) __syncthreads();   // the previous chunk's reads of shared memory are done
            // The warp's weights, every chunk (a later one reads them from the cache): nothing lives across chunks.
            uint32_t q0[U][R], q1[U][R], sc[U][R];
#pragma unroll
            for (int i = 0; i < U; i++)
            {
                const int u = lane + 32 * i;
                int wo = 0, so = 0, at, gap;
                moe_unit_place<TYPE>(u < units ? u : 0, wo, so, at, gap);
#pragma unroll
                for (int r = 0; r < R; r++)
                {
                    q0[i][r] = 0; q1[i][r] = 0; sc[i][r] = 0;
                    if (u < units && row0 + r < a.m) moe_unit_raw<TYPE>(base + (long long)(row0 + r) * a.w_row, wo, so, q0[i][r], q1[i][r], sc[i][r]);
                }
            }

            for (int idx = threadIdx.x; idx < np * chunks; idx += blockDim.x)
            {
                const int pi = idx / chunks, c = idx - pi * chunks;
                const int pair = list[first + c0 + pi], t = pair / a.used, j = pair - t * a.used;
                const long long act = (long long)t * a.acts + (a.acts == 1 ? 0 : j);
                reinterpret_cast<int4*>(sq + (size_t)pi * a.k)[c] = __ldg(reinterpret_cast<const int4*>(a.q + act * a.k) + c);
            }
            for (int idx = threadIdx.x; idx < np * per; idx += blockDim.x)
            {
                const int pi = idx / per, c = idx - pi * per;
                const int pair = list[first + c0 + pi], t = pair / a.used, j = pair - t * a.used;
                const long long act = (long long)t * a.acts + (a.acts == 1 ? 0 : j);
                sd[(size_t)pi * per + c] = a.qd[act * per + c];
            }
            if (threadIdx.x < np) sp[threadIdx.x] = list[first + c0 + threadIdx.x];
            __syncthreads();
            if (row0 >= a.m) continue;
            float acc[R][P];
#pragma unroll
            for (int r = 0; r < R; r++)
#pragma unroll
                for (int pp = 0; pp < P; pp++) acc[r][pp] = 0.0f;
#pragma unroll
            for (int i = 0; i < U; i++)
            {
                const int u = lane + 32 * i;
                if (u < units)
                {
                    int wo, so, at, gap;
                    moe_unit_place<TYPE>(u, wo, so, at, gap);
                    int2 w[R];
                    float dw[R];
#pragma unroll
                    for (int r = 0; r < R; r++) moe_unit_decode<TYPE>(q0[i][r], q1[i][r], sc[i][r], w[r], dw[r], fp4, iq4);
#pragma unroll
                    for (int pp = 0; pp < P; pp++)
                    {
                        if (pp < np)
                        {
                            const int xlo = *reinterpret_cast<const int*>(sq + (size_t)pp * a.k + at);
                            const int xhi = *reinterpret_cast<const int*>(sq + (size_t)pp * a.k + at + gap);
                            const float dx = sd[(size_t)pp * per + (at >> 5)];
#pragma unroll
                            for (int r = 0; r < R; r++)
                            {
                                const int sumi = __dp4a(w[r].y, xhi, __dp4a(w[r].x, xlo, 0));
                                acc[r][pp] = fmaf((float)sumi, dw[r] * dx, acc[r][pp]);
                            }
                        }
                    }
                }
            }
#pragma unroll
            for (int r = 0; r < R; r++)
            {
#pragma unroll
                for (int pp = 0; pp < P; pp++)
                {
                    if (pp < np && row0 + r < a.m)
                    {
                        float v = acc[r][pp];
#pragma unroll
                        for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
                        if (lane == 0) a.dst[(long long)sp[pp] * a.m + row0 + r] = v;
                    }
                }
            }
        }
    }

#ifndef TSG_MOE_R3
#define TSG_MOE_R3 2
#endif

    template <int TYPE, int P, int U, bool CHUNKED>
    void launch_moe_product3(const MoeArgs& a, cudaStream_t stream)
    {
        constexpr int R = TSG_MOE_R3;
        const size_t smem = (size_t)P * a.k + (size_t)P * (a.k / 32) * 4 + P * 4;
        static bool attr = false;
        if (!attr && smem > 48 * 1024)
        {
            cudaFuncSetAttribute(inkling_moe_product3<TYPE, P, R, U, CHUNKED>, cudaFuncAttributeMaxDynamicSharedMemorySize, 96 * 1024);
            attr = true;
        }
        inkling_moe_product3<TYPE, P, R, U, CHUNKED><<<dim3(blocks(a.m, kMoe3Threads / 32 * R), (unsigned int)a.cap), kMoe3Threads, smem, stream>>>(a);
    }

    template <int TYPE, int P>
    bool launch_moe_product3_k(const MoeArgs& a, cudaStream_t stream)
    {
        // Units a lane takes: k / 8 / 32 rounded up (9: k 2304, 5: k 1152).
        const int u = (a.k / 8 + 31) / 32;
        if ((size_t)P * a.k + (size_t)P * (a.k / 32) * 4 + P * 4 > 96 * 1024) return false;
        switch (u)
        {
            case 9: if (a.rows > P) launch_moe_product3<TYPE, P, 9, true>(a, stream); else launch_moe_product3<TYPE, P, 9, false>(a, stream); return true;
            case 5: if (a.rows > P) launch_moe_product3<TYPE, P, 5, true>(a, stream); else launch_moe_product3<TYPE, P, 5, false>(a, stream); return true;
            default: return false;
        }
    }

    template <int TYPE>
    void launch_moe_product(const MoeArgs& a, cudaStream_t stream)
    {
        static const bool v3 = []() { const char* e = std::getenv("TS_INKLING_EXPERTS_V3"); return e == nullptr || e[0] != '0'; }();
        if (v3)
        {
            // Wider than 8 rows: the 8-pair single pass once for each 8 pairs an expert can have (a pass past an expert's
            // pairs exits at once; few experts get more than 8 of a verify's pairs), not the chunked kernel (its loop keeps
            // twice the registers and reads at ~60% of the single pass). TS_INKLING_EXPERTS_PASSES=0: the chunked kernel.
            static const bool passes = []() { const char* e = std::getenv("TS_INKLING_EXPERTS_PASSES"); return e == nullptr || e[0] != '0'; }();
            const int units = (a.k / 8 + 31) / 32;
            if (passes && a.rows > 8 && (units == 9 || units == 5))
            {
                // A row's slots are distinct experts: an expert has at most a pair a row.
                MoeArgs b = a;
                b.rows = 8;   // the single-pass variant
                for (b.chunk0 = 0; b.chunk0 < a.rows; b.chunk0 += 8) launch_moe_product3_k<TYPE, 8>(b, stream);
                return;
            }
            if (launch_moe_product3_k<TYPE, 8>(a, stream)) return;
        }
        // Rows a block: 8 warps of 8 rows (a pair cap of 8), 4 (16) or 2 (32).
        const unsigned int cap = (unsigned int)a.cap, warps = kMoeThreads / 32;
        if (a.rows <= 8) inkling_moe_product<TYPE, 8><<<dim3(blocks(a.m, warps * moe_rows_per_warp<8>()), cap), kMoeThreads, 0, stream>>>(a);
        else if (a.rows <= 16) inkling_moe_product<TYPE, 16><<<dim3(blocks(a.m, warps * moe_rows_per_warp<16>()), cap), kMoeThreads, 0, stream>>>(a);
        else inkling_moe_product<TYPE, 32><<<dim3(blocks(a.m, warps * moe_rows_per_warp<32>()), cap), kMoeThreads, 0, stream>>>(a);
    }
