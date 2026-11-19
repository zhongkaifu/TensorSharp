// Copyright (c) Zhongkai Fu. All rights reserved.
// https://github.com/zhongkaifu/TensorSharp
//
// This file is part of TensorSharp.
//
// TensorSharp is licensed under the BSD-3-Clause license found in the LICENSE file in the root directory of this source tree.
//
// ---------------------------------------------------------------------------
// Tensor-core flash prefill attention: GQA groups folded into the query tile, F16 K/V tiles in
// shared memory, online softmax, causal masking with a floor, split-KV for long contexts
// (ts_flash_prefill_f16_d<hd>, CudaGraphExecutor's attention for prompt rows).
//
// The committed compute_75 PTX in native/ptx is what every build without nvcc loads, so it is a
// tracked artifact that must be refreshed with the source (DEVELOPMENT.md). The probe
// is how the host tells a stale image from a current one: CudaFlashPrefillKernels
// compares TS_FLASH_PREFILL_VERSION, as the loaded image reports it, with the version its
// C# side was written against, and refuses a module that does not match - a
// stale committed PTX then keeps every route on its fallback instead of launching
// kernels that are not there (or that take other arguments).
//
// Bump TS_FLASH_PREFILL_VERSION, and CudaFlashPrefillKernels.CurrentVersion with it, whenever
// a kernel is added or its signature changes.
//
// sm_80-only code (cp.async, m16n8k16, bf16 mma) must sit under
// #if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800 and be gated on the host with
// CudaKernels.ArchGuardedKernelsUsable; prefer sm_75-clean code (m16n8k8 + ldmatrix)
// so the committed compute_75 PTX carries working kernels.
//
// 8-bit prompt attention (ts_flash_prompt_band_f16_d256 + ts_flash_prompt_i8e4_d256): a prompt chunk's rows the runtime
// marks as one (FlashAttnExt ip[2], GraphAttentionRows.PromptChunk), over 256-wide heads, attended with Q K^T on the INT8
// tensor cores (exact int32) and P V on the block-scaled E4M3 tensor cores, which only an sm_120a / sm_121a image carries
// (ts_flash_prefill_attn8_caps reports it; every other image traps in these kernels and the host never launches them).
// The keys are converted once per layer and chunk into a band (INT8 K with one F32 scale per key, E4M3 V^T tiles with one
// power-of-two scale per 32 keys), which the attention streams instead of the F16 cache. The direct engine's kernels
// (038fb303, ts_q35_attn8_prefill_f32) on the graph op's operands: the same arithmetic per row (about 2% relative RMS
// against the 16-bit kernel a layer). At one key split, a row's result depends on where its chunk ends only through the V
// scale of the chunk's last, partial 32-key tile: the at most 31 rows whose causal end lies in that tile (the other tiles
// are whole, K and Q are scaled per key and per row). The key splits add a second dependence: the executor picks their
// count from the chunk's own row count (CudaFlashPrefillKernels.Splits), and each split's key range is a share of the keys
// of its block's last row (kAll), so two chunkings that give a row another split count or another block round it apart
// outside that tile too: each split rounds its probabilities to E4M3 against its own running max, so such rows differ by
// about the 8-bit arithmetic's own error (a 24-row chunk over 3,618 keys runs 16 splits on 82 SMs, the same rows inside a
// 1,570-row chunk one: 3.0% relative RMS apart on Gaussian data). The 16-bit prompt kernel splits the same way, with F16
// probabilities. Off by default (CudaGraphExecutor.PromptAttention8Enabled).
// ---------------------------------------------------------------------------

#define TS_FLASH_PREFILL_VERSION 12

/// Writes the module version to out[0] and the architecture this image was compiled for
/// (__CUDA_ARCH__ of the code the driver selected: the SASS of a fatbin, or the PTX target
/// a JIT compiled from) to out[1]. One thread; launch <<<1, 1>>>.
extern "C" __global__ void ts_flash_prefill_caps(int* out)
{
    if (threadIdx.x != 0 || blockIdx.x != 0)
        return;
    out[0] = TS_FLASH_PREFILL_VERSION;
#if defined(__CUDA_ARCH__)
    out[1] = __CUDA_ARCH__;
#else
    out[1] = 0;
#endif
}

// ---------------------------------------------------------------------------
// Prompt attention over a head-first F16 cache (CudaGraphExecutor's attention for prompt rows): the direct-CUDA
// Qwen3.5 engine's prefill attention (ts_q35_attn_pipe) on the graph op's operands.
//   q     [heads][nq][HD] F32 (the query heads of a KV head are its G = heads / nKv consecutive heads)
//   k, v  [nKv][cap][HD] F16 (the cache; keys 0..len-1 are the window)
//   out   [nq][heads][HD] F32
// Query t sees keys [floor, len - nq + t] (causal, the rows the last nq of the window). A block of 8 warps takes 128
// rows of one KV head (row r: query t = r / G of head kvh * G + r % G) and streams 32-key K and V tiles through a
// two-stage cp.async pipeline. The mma accumulate in F16 over short runs only: each 16-value step of QK^T (Q
// pre-scaled to log2 units) is rounded once and summed in F32, and PV over one 32-key tile (probabilities at most 1)
// is rounded twice before it joins the F32 accumulators. With dyn set (a recorded decode step) the window's length is
// dyn[0], read on the device. splits > 1 (a grid too small to fill the device: a few
// query rows over a long window): block x = (row tile, key split), each split takes a 32-key-aligned share of its
// rows' keys and writes unnormalized partials (acc, ln-domain max, sum) that ts_flash_prefill_combine_f32 merges. A recorded
// decode step's launch may hold more splits than it runs: those its window of 256-key buckets gives its length
// (ts_fp_live_splits), so a window padded to the slot's capacity splits each length as the bucket window around it does.
// With cells set (one decode row, nq = 1: a sparse attention's selection) the keys are the listed cache rows
// cells[0 .. cellCount[0]) instead of 0..len-1, each with its mask value cellMask[cell] added (len and floor unused);
// without cellMask (a selection whose kept cells all hold 0: QSA's) nothing is added, the same sums.
// sm_80+ (cp.async, ldmatrix, mma.f16): the compute_75 image traps here; the host launches it only where
// CudaPrefillModule.ArchGuardedKernelsUsable holds.
//
// The _q8 kernels take a q8_0 cache (ggml's block_q8_0 rows: per 32 values an F16 scale d then 32 int8 codes, 34 bytes, so
// HD / 32 * 34 bytes a row, [nKv][cap] rows) and run the F16 kernels' arithmetic on its values: each 32-key tile's raw rows
// stream through a ring of TS_FP_Q8_STAGES (three) cp.async stages, two tiles in flight ahead of the one being dequantized
// (the F16 kernels' pipeline has two stages), and every thread of the block - the warps a decode row or a verify
// window leaves idle among them - dequantizes them into one F16 tile pair, value = RN_F16(code * d) (__hmul2: code * d is exact
// in F32), before the warps compute exactly as the F16 kernel does from its tile. A q8_0 cache then answers as the F16 kernel
// over the F16 cache of its values, bit for bit, with half the bytes read.
// ---------------------------------------------------------------------------
#include <cuda_fp16.h>
#include <stdint.h>

#define TS_FP_KT 32      // keys per tile
#define TS_FP_ROWS 128   // query rows per block (8 warps x 16)

__device__ __forceinline__ uint32_t ts_fp_smem_u32(const void* p)
{
    return static_cast<uint32_t>(__cvta_generic_to_shared(p));
}

__device__ __forceinline__ void ts_fp_cp16(void* dst, const void* src, bool valid)
{
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;\n"
                 :: "r"(ts_fp_smem_u32(dst)), "l"(src), "r"(valid ? 16 : 0));
#else
    __trap();
#endif
}

// 8 bytes (a q8_0 row of 128 values is 136 bytes: its rows are 8-byte aligned only).
__device__ __forceinline__ void ts_fp_cp8(void* dst, const void* src, bool valid)
{
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
    asm volatile("cp.async.ca.shared.global [%0], [%1], 8, %2;\n"
                 :: "r"(ts_fp_smem_u32(dst)), "l"(src), "r"(valid ? 8 : 0));
#else
    __trap();
#endif
}

__device__ __forceinline__ void ts_fp_cp_commit()
{
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
    asm volatile("cp.async.commit_group;\n" ::);
#else
    __trap();
#endif
}

__device__ __forceinline__ void ts_fp_cp_wait0()
{
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
    asm volatile("cp.async.wait_group 0;\n" ::);
#else
    __trap();
#endif
}

// Until at most N of this thread's newest cp.async groups are pending.
template <int N>
__device__ __forceinline__ void ts_fp_cp_wait()
{
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
    asm volatile("cp.async.wait_group %0;\n" :: "n"(N));
#else
    __trap();
#endif
}

__device__ __forceinline__ void ts_fp_ldsm_x4(uint32_t (&r)[4], const void* p)
{
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n"
                 : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3]) : "r"(ts_fp_smem_u32(p)));
#else
    r[0] = r[1] = r[2] = r[3] = 0;
    __trap();
#endif
}

__device__ __forceinline__ void ts_fp_ldsm_x4_t(uint32_t (&r)[4], const void* p)
{
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0,%1,%2,%3}, [%4];\n"
                 : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3]) : "r"(ts_fp_smem_u32(p)));
#else
    r[0] = r[1] = r[2] = r[3] = 0;
    __trap();
#endif
}

// m16n8k16 with F16 accumulation.
__device__ __forceinline__ void ts_fp_mma_f16h(uint32_t (&c)[2], const uint32_t (&a)[4], uint32_t b0, uint32_t b1)
{
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f16.f16.f16.f16 {%0,%1}, {%2,%3,%4,%5}, {%6,%7}, {%0,%1};\n"
                 : "+r"(c[0]), "+r"(c[1]) : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
#else
    __trap();
#endif
}

__device__ __forceinline__ uint32_t ts_fp_pack_h2(float lo, float hi)
{
    const half2 v = __floats2half2_rn(lo, hi);
    return *reinterpret_cast<const uint32_t*>(&v);
}

// Dynamic shared memory of ts_flash_prefill_f16_d<HD>: 2 stages x (K, V) x TS_FP_KT rows of HD + 8 halves
// (CudaFlashPrefillKernels.SharedBytes computes the same).
template <int HD>
struct TsFpAttnSmem
{
    static constexpr int ROW = HD + 8;   // F16 tile row stride (bank-spread)
    half k[2][TS_FP_KT * ROW];
    half v[2][TS_FP_KT * ROW];
    __device__ __forceinline__ const half* kt(int st) const { return k[st]; }
    __device__ __forceinline__ const half* vt(int st) const { return v[st]; }
};

// Bytes of a q8_0 cache row of HD values (ggml's block_q8_0: F16 d, then 32 int8 codes).
template <int HD>
struct TsFpQ8Row
{
    static constexpr int BYTES = HD / 32 * 34;
    static constexpr int CHUNK = BYTES % 16 == 0 ? 16 : 8;   // the cp.async size its rows' alignment allows
    static constexpr int CHUNKS = BYTES / CHUNK;
};

// Raw q8_0 tile stages in flight: a q8_0 tile is half an F16 tile's bytes, so two of them ahead of the one being dequantized
// keep as many bytes coming as the F16 kernel's one (with two stages the q8_0 kernels waited on memory latency, not bandwidth:
// an 8-row Qwen3.8-27B window 4% faster than F16, not 40%, on an RTX PRO 4500).
#define TS_FP_Q8_STAGES 3

// Dynamic shared memory of ts_flash_*_q8_d<HD>: TS_FP_Q8_STAGES stages x (K, V) x TS_FP_KT raw q8_0 rows, and one F16 tile pair
// their values are dequantized into (CudaFlashPrefillKernels.SharedBytes computes the same).
template <int HD>
struct TsFpAttnSmemQ8
{
    static constexpr int ROW = HD + 8;
    static constexpr int RAW = TsFpQ8Row<HD>::BYTES;
    uint8_t raw[TS_FP_Q8_STAGES][2][TS_FP_KT * RAW];   // [stage][K, V]
    half k[TS_FP_KT * ROW];
    half v[TS_FP_KT * ROW];
    __device__ __forceinline__ const half* kt(int) const { return k; }
    __device__ __forceinline__ const half* vt(int) const { return v; }
};

template <int HD, bool Q8> struct TsFpSmemOf { typedef TsFpAttnSmem<HD> T; typedef half Cache; };
template <int HD> struct TsFpSmemOf<HD, true> { typedef TsFpAttnSmemQ8<HD> T; typedef uint8_t Cache; };

// The cache row `row` of the head at `head` (q8_0 rows of TsFpQ8Row<HD>::BYTES bytes): the one place a tile loader forms a row's
// address (a block table would change it alone).
template <int HD>
__device__ __forceinline__ const uint8_t* ts_fp_q8_row(const uint8_t* head, int row)
{
    return head + (long long)row * TsFpQ8Row<HD>::BYTES;
}

// One q8_0 block (34 bytes at b: F16 d, then 32 int8 codes, 2-byte aligned) as 32 F16 values RN_F16(code * d) at dst (16-byte
// aligned): each pair of codes biased into the mantissa of 1024 (0x6400 | (code + 128): exact), less 1152, times d.
__device__ __forceinline__ void ts_fp_q8_block_to_h(const uint8_t* b, half* dst)
{
    const uint16_t* w = reinterpret_cast<const uint16_t*>(b);
    const half2 d2 = __half2half2(__ushort_as_half(w[0]));
    const half2 bias = __half2half2(__ushort_as_half((unsigned short)0x6480));   // 1152
    uint32_t o[16];
#pragma unroll
    for (int j = 0; j < 16; ++j)
    {
        const uint32_t h = __byte_perm((uint32_t)w[1 + j] ^ 0x8080u, 0x64u, 0x4140);
        half2 x = __hsub2(*reinterpret_cast<const half2*>(&h), bias);
        x = __hmul2(x, d2);
        o[j] = *reinterpret_cast<const uint32_t*>(&x);
    }
    uint4* out = reinterpret_cast<uint4*>(dst);
#pragma unroll
    for (int j = 0; j < 4; ++j)
        out[j] = make_uint4(o[4 * j], o[4 * j + 1], o[4 * j + 2], o[4 * j + 3]);
}

// CudaFlashPrefillKernels.Splits on the device: key splits for `rows` query rows of each of nKv KV heads over `keys` keys,
// enough blocks for about two waves of the device's sms multiprocessors, each split at least four key tiles.
__device__ __forceinline__ int ts_fp_splits(int rows, int nKv, int keys, int sms)
{
    const int blocks = (rows + TS_FP_ROWS - 1) / TS_FP_ROWS * nKv;
    int splits = 1;
    while (blocks * splits * 2 <= 2 * sms && keys / (splits * 2) >= 4 * TS_FP_KT && splits < 64)
        splits *= 2;
    return splits;
}

// The splits a row of `len` keys runs under the bucket rule (bucket > 0): the host's rule over the window of `bucket`-key
// steps its length falls in - min(windowKeys, the length rounded up to the bucket) - never more than the launch holds; the
// launch's own count without the rule.
__device__ __forceinline__ int ts_fp_rule_splits(int launched, int len, int windowKeys, int bucket, int rows, int nKv, int sms)
{
    if (bucket <= 0)
        return launched;
    const int keys = min(windowKeys, (len + bucket - 1) / bucket * bucket);
    return min(launched, ts_fp_splits(rows, nKv, keys, sms));
}

// The splits a recorded step's launch runs (bucket > 0, dyn set): the bucket rule over its length dyn[0] plus lenBias, so a
// window as wide as the slot's capacity splits each length as the bucket window around it would; the launch's own count
// otherwise (and never more: it covers the widest window).
__device__ __forceinline__ int ts_fp_live_splits(int launched, const int* __restrict__ dyn, int lenBias, int windowKeys, int bucket,
    int rows, int nKv, int sms)
{
    if (dyn == nullptr)
        return launched;
    return ts_fp_rule_splits(launched, dyn[0] + lenBias, windowKeys, bucket, rows, nKv, sms);
}

// WIN: the nq rows are a verify window's (q [nq][nHead][HD]), each run as the one-row launch at its position would run it:
// a row runs the splits its own length's bucket window gives (ts_fp_rule_splits over windowKeys and bucket: the decode row's
// at its position, `live` - the window's last row's - at most) and splits its keys by its own length (the one-row launch's
// share), the block walks the union of its rows' key tiles (each KT-aligned), and keys outside a row's own range are masked
// for it - they add exact zeros and leave its maximum, so a row's sums are the one-row launch's bit for bit, while every key
// tile is read once for all the window's rows.
// PROMPT: a prompt chunk's rows (more than a verify window's), scheduled for the causal triangle: the row tiles run heaviest
// first (block x takes tile tiles - 1 - x / splits: the last rows see the most keys, and a light tile now ends the grid where
// the heaviest had started last), and a warp skips the key tiles past every one of its rows' ends (a whole tile its rows
// mask: the zeros it added left every sum as it was) - each row's arithmetic is the plain launch's.
// Q8: kc and vc are q8_0 caches (TsFpQ8Row<HD>::BYTES a row): each tile's raw rows are dequantized into the F16 tile by the
// whole block between two barriers, then every warp computes from it as the F16 kernel computes from its own (keys a tile does
// not load are zero rows in both).
template <int HD, bool WIN = false, bool PROMPT = false, bool Q8 = false>
__device__ __forceinline__ void ts_fp_attn(
    const float* __restrict__ q, const typename TsFpSmemOf<HD, Q8>::Cache* __restrict__ kc,
    const typename TsFpSmemOf<HD, Q8>::Cache* __restrict__ vc, int cap, int nq, int len,
    int floor, int nHead, int nKv, float scale, int splits, float* __restrict__ partial, float* __restrict__ out,
    const int* __restrict__ dyn, const int* __restrict__ cells, const int* __restrict__ cellCount, const float* __restrict__ cellMask,
    int live, int windowKeys = 0, int bucket = 0, int sms = 0)
{
    // A recorded decode step reads the window's length from the step's parameter block (CudaDecodeDynParams).
    if (dyn != nullptr)
        len = dyn[0];
    if (cells != nullptr)
    {
        len = cellCount[0];
        floor = 0;
    }
    typedef typename TsFpSmemOf<HD, Q8>::T Smem;
    constexpr int ROW = Smem::ROW, KS = HD / 16, NS = HD / 8;
    extern __shared__ __align__(16) uint8_t ts_fp_attn_smem[];
    Smem& sm = *reinterpret_cast<Smem*>(ts_fp_attn_smem);
    const int kvh = blockIdx.y;
    const int G = nHead / nKv;
    const int R = nq * G;
    // The grid holds `splits` splits a row tile (the partials that many apart); `live` of them run (ts_fp_live_splits), each the
    // share a launch of `live` splits gives it, so a recorded launch over the slot's whole window runs what the bucket window's
    // launch runs. The others leave before the first barrier, the whole block.
    const int split = (int)(blockIdx.x % splits);
    const int tileOfBlock = (int)(blockIdx.x / splits);
    const int rowBase = (PROMPT ? (R + TS_FP_ROWS - 1) / TS_FP_ROWS - 1 - tileOfBlock : tileOfBlock) * TS_FP_ROWS;
    if (split >= live)
        return;
    const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5, g = lane >> 2, qd = lane & 3;
    const int past = len - nq;   // keys before the first query row
    // A split's key range over kAll keys run in `runs` splits (the one-row launch's, kAll its row's length).
    auto range = [&](int kAll, int runs, int& lo, int& hi) {
        const int share = ((kAll + runs - 1) / runs + TS_FP_KT - 1) / TS_FP_KT * TS_FP_KT;
        lo = max(min(kAll, split * share), min(kAll, floor / TS_FP_KT * TS_FP_KT));
        hi = min(kAll, split * share + share);
    };
    // The splits a row of kAll keys runs: a window row its own length's (the decode row's at its position), else the launch's.
    auto runsOf = [&](int kAll) { return WIN ? ts_fp_rule_splits(live, kAll, windowKeys, bucket, G, nKv, sms) : live; };
    int kLo, kHi;
    if (WIN)
    {
        // The union of the block's rows' ranges that hold keys (each row's splits its own, so a later row's range need not
        // reach below an earlier one's): each starts on a tile boundary, so every row's tiles are its one-row launch's.
        kLo = len;
        kHi = 0;
        const int tLast = (min(rowBase + TS_FP_ROWS, R) - 1) / G;
        for (int t = rowBase / G; t <= tLast; ++t)
        {
            int lo, hi;
            range(past + t + 1, runsOf(past + t + 1), lo, hi);
            if (hi > lo)
            {
                kLo = min(kLo, lo);
                kHi = max(kHi, hi);
            }
        }
        if (kHi <= kLo)
            kLo = kHi = 0;
    }
    else
        range(past + (min(rowBase + TS_FP_ROWS, R) - 1) / G + 1, live, kLo, kHi);

    const int r0 = rowBase + warp * 16 + g, r1 = r0 + 8;
    const bool live0 = r0 < R, live1 = r1 < R;
    const int end0 = live0 ? past + r0 / G + 1 : 0, end1 = live1 ? past + r1 / G + 1 : 0;
    // A window row's own splits and key range (the launch's and the whole block's elsewhere).
    const int runs0 = live0 ? runsOf(end0) : live, runs1 = live1 ? runsOf(end1) : live;
    int lo0 = kLo, hi0 = kHi, lo1 = kLo, hi1 = kHi;
    if (WIN)
    {
        if (live0) range(end0, runs0, lo0, hi0);
        if (live1) range(end1, runs1, lo1, hi1);
    }
    const bool warpLive = rowBase + warp * 16 < R;
    // A prompt chunk's warp: the end of its last-seeing row (tiles from there are masked for every row of it).
    int warpEnd = max(end0, end1);
    if (PROMPT)
    {
#pragma unroll
        for (int o = 16; o > 0; o >>= 1) warpEnd = max(warpEnd, __shfl_xor_sync(0xffffffffu, warpEnd, o));
    }
    typedef typename TsFpSmemOf<HD, Q8>::Cache Cache;
    const Cache* kHead = kc + (long long)kvh * cap * (Q8 ? TsFpQ8Row<HD>::BYTES : HD);
    const Cache* vHead = vc + (long long)kvh * cap * (Q8 ? TsFpQ8Row<HD>::BYTES : HD);

    auto load = [&](int st, int first) {
        if constexpr (Q8)
        {
            // The tile's raw q8_0 rows, in the cp.async chunks their alignment allows.
            constexpr int CH = TsFpQ8Row<HD>::CHUNK, CHUNKS = TsFpQ8Row<HD>::CHUNKS, RAW = TsFpQ8Row<HD>::BYTES;
            for (int i = tid; i < 2 * TS_FP_KT * CHUNKS; i += blockDim.x)
            {
                const int isV = i >= TS_FP_KT * CHUNKS, c = isV ? i - TS_FP_KT * CHUNKS : i;
                const int key = c / CHUNKS, part = c % CHUNKS;
                const bool ok = first + key < kHi;
                const int row = !ok ? 0 : cells != nullptr ? cells[first + key] : first + key;
                const uint8_t* src = ts_fp_q8_row<HD>(isV ? vHead : kHead, row) + part * CH;
                uint8_t* dst = sm.raw[st][isV] + key * RAW + part * CH;
                if constexpr (CH == 16) ts_fp_cp16(dst, src, ok);
                else ts_fp_cp8(dst, src, ok);
            }
        }
        else
        {
            constexpr int CHUNKS = HD / 8;   // 16-byte chunks of a key row
            for (int i = tid; i < 2 * TS_FP_KT * CHUNKS; i += blockDim.x)
            {
                const int isV = i >= TS_FP_KT * CHUNKS, c = isV ? i - TS_FP_KT * CHUNKS : i;
                const int key = c / CHUNKS, part = c % CHUNKS;
                const bool ok = first + key < kHi;
                const int row = !ok ? 0 : cells != nullptr ? cells[first + key] : first + key;
                const half* src = (isV ? vHead : kHead) + (long long)row * HD + part * 8;
                ts_fp_cp16((isV ? sm.v[st] : sm.k[st]) + key * ROW + part * 8, src, ok);
            }
        }
    };
    if constexpr (Q8)
    {
#pragma unroll
        for (int s = 0; s < TS_FP_Q8_STAGES - 1; ++s)
        {
            if (kLo + s * TS_FP_KT < kHi)
                load(s, kLo + s * TS_FP_KT);
            ts_fp_cp_commit();
        }
    }
    else
    {
        if (kHi > kLo)
            load(0, kLo);
        ts_fp_cp_commit();
    }

    const float log2Scale = scale * 1.4426950408889634f;
    uint32_t qa[KS][4];
    {
        const float* q0 = !live0 ? nullptr : WIN ? q + ((long long)(r0 / G) * nHead + kvh * G + r0 % G) * HD
            : q + ((long long)(kvh * G + r0 % G) * nq + r0 / G) * HD;
        const float* q1 = !live1 ? nullptr : WIN ? q + ((long long)(r1 / G) * nHead + kvh * G + r1 % G) * HD
            : q + ((long long)(kvh * G + r1 % G) * nq + r1 / G) * HD;
#pragma unroll
        for (int kk = 0; kk < KS; ++kk)
        {
            const int c = kk * 16 + qd * 2;
            qa[kk][0] = q0 ? ts_fp_pack_h2(q0[c] * log2Scale, q0[c + 1] * log2Scale) : 0u;
            qa[kk][1] = q1 ? ts_fp_pack_h2(q1[c] * log2Scale, q1[c + 1] * log2Scale) : 0u;
            qa[kk][2] = q0 ? ts_fp_pack_h2(q0[c + 8] * log2Scale, q0[c + 9] * log2Scale) : 0u;
            qa[kk][3] = q1 ? ts_fp_pack_h2(q1[c + 8] * log2Scale, q1[c + 9] * log2Scale) : 0u;
        }
    }

    float acc[NS][4];
#pragma unroll
    for (int n = 0; n < NS; ++n)
        acc[n][0] = acc[n][1] = acc[n][2] = acc[n][3] = 0.0f;
    float m0 = -INFINITY, m1 = -INFINITY, l0 = 0.0f, l1 = 0.0f;

    for (int first = kLo, t = 0, q8Stage = 0; first < kHi; first += TS_FP_KT, ++t)
    {
        const int st = Q8 ? q8Stage : t & 1;
        if constexpr (Q8)
            ts_fp_cp_wait<TS_FP_Q8_STAGES - 2>();
        else
            ts_fp_cp_wait0();
        __syncthreads();   // this tile landed; every warp is done with the previous one
        if constexpr (Q8)
        {
            // The stage the previous tile was dequantized from takes the tile TS_FP_Q8_STAGES - 1 ahead.
            const int ahead = first + (TS_FP_Q8_STAGES - 1) * TS_FP_KT;
            if (ahead < kHi)
                load(q8Stage == 0 ? TS_FP_Q8_STAGES - 1 : q8Stage - 1, ahead);
            q8Stage = q8Stage + 1 == TS_FP_Q8_STAGES ? 0 : q8Stage + 1;
        }
        else if (first + TS_FP_KT < kHi)
            load(st ^ 1, first + TS_FP_KT);
        ts_fp_cp_commit();
        if constexpr (Q8)
        {
            // Every thread dequantizes (key, block) pairs of this stage's raw rows into the one F16 tile pair (the warps a
            // decode row or window leaves idle among them); the barrier above saw every warp done with the tile's last values.
            constexpr int NB = HD / 32, RAW = TsFpQ8Row<HD>::BYTES;
            for (int i = tid; i < 2 * TS_FP_KT * NB; i += blockDim.x)
            {
                const int isV = i >= TS_FP_KT * NB, c = isV ? i - TS_FP_KT * NB : i;
                const int key = c / NB, blk = c % NB;
                ts_fp_q8_block_to_h(sm.raw[st][isV] + key * RAW + blk * 34, (isV ? sm.v : sm.k) + key * ROW + blk * 32);
            }
            __syncthreads();
        }
        if (!warpLive || (PROMPT && first >= warpEnd))
            continue;

        float s[4][4];
        {
            // B fragments of 2 n-tiles x 1 k-step per ldmatrix.x4 (rows = keys).
            const half* kb = sm.kt(st) + ((lane & 7) + ((lane >> 4) << 3)) * ROW + ((lane >> 3) & 1) * 8;
#pragma unroll
            for (int np = 0; np < 2; ++np)
            {
#pragma unroll
                for (int e = 0; e < 4; ++e)
                    s[2 * np][e] = s[2 * np + 1][e] = 0.0f;
#pragma unroll
                for (int kk = 0; kk < KS; ++kk)
                {
                    uint32_t h0[2] = { 0u, 0u }, h1[2] = { 0u, 0u };
                    uint32_t b[4];
                    ts_fp_ldsm_x4(b, kb + np * 16 * ROW + kk * 16);
                    ts_fp_mma_f16h(h0, qa[kk], b[0], b[1]);
                    ts_fp_mma_f16h(h1, qa[kk], b[2], b[3]);
                    const float2 a0 = __half22float2(*reinterpret_cast<const half2*>(&h0[0]));
                    const float2 a1 = __half22float2(*reinterpret_cast<const half2*>(&h0[1]));
                    const float2 c0 = __half22float2(*reinterpret_cast<const half2*>(&h1[0]));
                    const float2 c1 = __half22float2(*reinterpret_cast<const half2*>(&h1[1]));
                    s[2 * np][0] += a0.x; s[2 * np][1] += a0.y; s[2 * np][2] += a1.x; s[2 * np][3] += a1.y;
                    s[2 * np + 1][0] += c0.x; s[2 * np + 1][1] += c0.y; s[2 * np + 1][2] += c1.x; s[2 * np + 1][3] += c1.y;
                }
            }
        }

        float t0 = m0, t1 = m1;
#pragma unroll
        for (int j = 0; j < 4; ++j)
        {
            const int key = first + j * 8 + qd * 2;
            float mk0 = 0.0f, mk1 = 0.0f;   // a listed cell's mask value (log2 units)
            if (cells != nullptr && cellMask != nullptr)
            {
                mk0 = key < kHi ? cellMask[cells[key]] * 1.4426950408889634f : 0.0f;
                mk1 = key + 1 < kHi ? cellMask[cells[key + 1]] * 1.4426950408889634f : 0.0f;
            }
            s[j][0] = key < end0 && key >= floor && key >= lo0 && key < hi0 ? s[j][0] + mk0 : -INFINITY;
            s[j][1] = key + 1 < end0 && key + 1 >= floor && key + 1 >= lo0 && key + 1 < hi0 ? s[j][1] + mk1 : -INFINITY;
            s[j][2] = key < end1 && key >= floor && key >= lo1 && key < hi1 ? s[j][2] + mk0 : -INFINITY;
            s[j][3] = key + 1 < end1 && key + 1 >= floor && key + 1 >= lo1 && key + 1 < hi1 ? s[j][3] + mk1 : -INFINITY;
            t0 = fmaxf(t0, fmaxf(s[j][0], s[j][1]));
            t1 = fmaxf(t1, fmaxf(s[j][2], s[j][3]));
        }
#pragma unroll
        for (int off = 1; off < 4; off <<= 1)
        {
            t0 = fmaxf(t0, __shfl_xor_sync(0xffffffff, t0, off));
            t1 = fmaxf(t1, __shfl_xor_sync(0xffffffff, t1, off));
        }
        const float c0 = t0 == -INFINITY ? 1.0f : exp2f(m0 - t0);
        const float c1 = t1 == -INFINITY ? 1.0f : exp2f(m1 - t1);
        m0 = t0;
        m1 = t1;
        l0 *= c0;
        l1 *= c1;
        if (!__all_sync(0xffffffff, c0 == 1.0f && c1 == 1.0f))
        {
#pragma unroll
            for (int n = 0; n < NS; ++n)
            {
                acc[n][0] *= c0; acc[n][1] *= c0;
                acc[n][2] *= c1; acc[n][3] *= c1;
            }
        }
#pragma unroll
        for (int j = 0; j < 4; ++j)
        {
            s[j][0] = t0 == -INFINITY ? 0.0f : exp2f(s[j][0] - t0);
            s[j][1] = t0 == -INFINITY ? 0.0f : exp2f(s[j][1] - t0);
            s[j][2] = t1 == -INFINITY ? 0.0f : exp2f(s[j][2] - t1);
            s[j][3] = t1 == -INFINITY ? 0.0f : exp2f(s[j][3] - t1);
            l0 += s[j][0] + s[j][1];
            l1 += s[j][2] + s[j][3];
        }
        uint32_t pa[2][4];
#pragma unroll
        for (int kk = 0; kk < 2; ++kk)
        {
            pa[kk][0] = ts_fp_pack_h2(s[2 * kk][0], s[2 * kk][1]);
            pa[kk][1] = ts_fp_pack_h2(s[2 * kk][2], s[2 * kk][3]);
            pa[kk][2] = ts_fp_pack_h2(s[2 * kk + 1][0], s[2 * kk + 1][1]);
            pa[kk][3] = ts_fp_pack_h2(s[2 * kk + 1][2], s[2 * kk + 1][3]);
        }
        const half* vrow = sm.vt(st) + ((lane & 7) + (((lane >> 3) & 1) << 3)) * ROW + ((lane >> 4) << 3);
#pragma unroll
        for (int dn = 0; dn < HD / 16; ++dn)
        {
            uint32_t h0[2] = { 0u, 0u }, h1[2] = { 0u, 0u };
#pragma unroll
            for (int kk = 0; kk < 2; ++kk)
            {
                uint32_t b[4];
                ts_fp_ldsm_x4_t(b, vrow + kk * 16 * ROW + dn * 16);
                ts_fp_mma_f16h(h0, pa[kk], b[0], b[1]);
                ts_fp_mma_f16h(h1, pa[kk], b[2], b[3]);
            }
            const float2 a0 = __half22float2(*reinterpret_cast<const half2*>(&h0[0]));
            const float2 a1 = __half22float2(*reinterpret_cast<const half2*>(&h0[1]));
            const float2 b0 = __half22float2(*reinterpret_cast<const half2*>(&h1[0]));
            const float2 b1 = __half22float2(*reinterpret_cast<const half2*>(&h1[1]));
            acc[2 * dn][0] += a0.x; acc[2 * dn][1] += a0.y; acc[2 * dn][2] += a1.x; acc[2 * dn][3] += a1.y;
            acc[2 * dn + 1][0] += b0.x; acc[2 * dn + 1][1] += b0.y; acc[2 * dn + 1][2] += b1.x; acc[2 * dn + 1][3] += b1.y;
        }
    }
    ts_fp_cp_wait0();
    if (!warpLive)
        return;
#pragma unroll
    for (int off = 1; off < 4; off <<= 1)
    {
        l0 += __shfl_xor_sync(0xffffffff, l0, off);
        l1 += __shfl_xor_sync(0xffffffff, l1, off);
    }
#pragma unroll
    for (int half_ = 0; half_ < 2; ++half_)
    {
        const int r = half_ ? r1 : r0;
        if (r >= R)
            continue;
        const int t = r / G, h = kvh * G + r % G;
        const long long base = (long long)t * nHead + h;
        const float l = half_ ? l1 : l0;
        // A window row that runs fewer splits than the block's: this split is not one of them.
        const int runs = half_ ? runs1 : runs0;
        if (split >= runs)
            continue;
        if (runs > 1)
        {
            float* p = partial + (base * splits + split) * (HD + 2);
#pragma unroll
            for (int n = 0; n < NS; ++n)
            {
                const int x = n * 8 + qd * 2;
                p[x] = acc[n][half_ * 2];
                p[x + 1] = acc[n][half_ * 2 + 1];
            }
            if (qd == 0)
            {
                const float m = half_ ? m1 : m0;
                p[HD] = m == -INFINITY ? -INFINITY : m * 0.6931471805599453f;   // log2 -> ln domain
                p[HD + 1] = l;
            }
            continue;
        }
        const float inv = l > 0.0f ? 1.0f / l : 0.0f;
        float* o = out + base * HD;
#pragma unroll
        for (int n = 0; n < NS; ++n)
        {
            const int x = n * 8 + qd * 2;
            o[x] = acc[n][half_ * 2] * inv;
            o[x + 1] = acc[n][half_ * 2 + 1] * inv;
        }
    }
}

// T: the cache's element (half, or uint8_t for the q8_0 rows of the _q8 kernels, Q8 true).
#define TS_FP_ATTN(NAME, HD, T, Q8) \
    extern "C" __global__ void __launch_bounds__(256) NAME( \
        const float* __restrict__ q, const T* __restrict__ kc, const T* __restrict__ vc, int cap, int nq, int len, \
        int floor, int nHead, int nKv, float scale, int splits, float* __restrict__ partial, float* __restrict__ out, \
        const int* __restrict__ dyn, const int* __restrict__ cells, const int* __restrict__ cellCount, const float* __restrict__ cellMask, \
        int lenBias, int windowKeys, int bucket, int sms) \
    { \
        ts_fp_attn<HD, false, false, Q8>(q, kc, vc, cap, nq, len, floor, nHead, nKv, scale, splits, partial, out, dyn, cells, cellCount, \
            cellMask, ts_fp_live_splits(splits, dyn, lenBias, windowKeys, bucket, nq * (nHead / nKv), nKv, sms)); \
    }

TS_FP_ATTN(ts_flash_prefill_f16_d128, 128, half, false)
TS_FP_ATTN(ts_flash_prefill_f16_d256, 256, half, false)

// A prompt chunk's rows (CudaGraphExecutor's attention past a verify window's rows): ts_flash_prefill_f16_d<hd>'s operands and
// launch, scheduled for the causal triangle (ts_fp_attn's PROMPT: each row's arithmetic theirs, bit for bit). Qwen3.8-27B on an
// RTX PRO 4000: a 16K prompt 4.08 s against 4.23 s with the plain launch, the same perplexity to the last digit.
#define TS_FP_PROMPT(NAME, HD, T, Q8) \
    extern "C" __global__ void __launch_bounds__(256) NAME( \
        const float* __restrict__ q, const T* __restrict__ kc, const T* __restrict__ vc, int cap, int nq, int len, \
        int floor, int nHead, int nKv, float scale, int splits, float* __restrict__ partial, float* __restrict__ out, \
        const int* __restrict__ dyn, const int* __restrict__ cells, const int* __restrict__ cellCount, const float* __restrict__ cellMask, \
        int lenBias, int windowKeys, int bucket, int sms) \
    { \
        ts_fp_attn<HD, false, true, Q8>(q, kc, vc, cap, nq, len, floor, nHead, nKv, scale, splits, partial, out, dyn, cells, \
            cellCount, cellMask, ts_fp_live_splits(splits, dyn, lenBias, windowKeys, bucket, nq * (nHead / nKv), nKv, sms)); \
    }

TS_FP_PROMPT(ts_flash_prompt_f16_d128, 128, half, false)
TS_FP_PROMPT(ts_flash_prompt_f16_d256, 256, half, false)

// A verify window's rows, each run as the decode row at its position (CudaGraphExecutor's window rows): grid z is the
// window row, which attends the window's keys less the rows - 1 - z after it (the window's length read from dyn[0] on
// the device when the step is recorded). One launch for what were `rows` one-row launches, each row through the same
// code as ts_flash_prefill_f16_d<hd> with nq = 1, so a row's sums are the decode row's bit for bit. Every row splits its
// keys as its own length's bucket says (ts_fp_rule_splits over the query heads of one KV head, as the decode row at its
// position); a selection's rows run the launch's splits (bucket 0).
//   q, out  [rows][nHead][HD] F32;  partial  rows x nHead x splits x (HD + 2) floats (combined over rows * nHead).
// With cells (a sparse attention's selection): row z's keys are cells[z * cellStride ..], cellCount[z] of them, each with
// its value in the mask row cellMask[z * cellStride ..] added (len and floor unused; none without cellMask).
// Launch: grid (row tiles * splits, nKv, rows), block 256, the dynamic shared memory of ts_flash_prefill_f16_d<hd>.
#define TS_FP_ATTN_ROWS(NAME, HD, T, Q8) \
    extern "C" __global__ void __launch_bounds__(256) NAME( \
        const float* __restrict__ q, const T* __restrict__ kc, const T* __restrict__ vc, int cap, int rows, int len, \
        int floor, int nHead, int nKv, float scale, int splits, float* __restrict__ partial, float* __restrict__ out, \
        const int* __restrict__ dyn, const int* __restrict__ cells, const int* __restrict__ cellCount, \
        const float* __restrict__ cellMask, int cellStride, int windowKeys, int bucket, int sms) \
    { \
        const int z = blockIdx.z; \
        const long long at = (long long)z * nHead * HD; \
        const bool listed = cells != nullptr; \
        const int rowLen = (dyn != nullptr ? dyn[0] : len) - (rows - 1 - z); \
        ts_fp_attn<HD, false, false, Q8>(q + at, kc, vc, cap, 1, rowLen, floor, nHead, nKv, scale, \
            splits, partial != nullptr ? partial + (long long)z * nHead * splits * (HD + 2) : nullptr, out + at, nullptr, \
            listed ? cells + (long long)z * cellStride : nullptr, listed ? cellCount + z : nullptr, \
            listed && cellMask != nullptr ? cellMask + (long long)z * cellStride : nullptr, \
            listed ? ts_fp_live_splits(splits, dyn, 0, windowKeys, bucket, nHead / nKv, nKv, sms) \
                : ts_fp_rule_splits(splits, rowLen, windowKeys, bucket, nHead / nKv, nKv, sms)); \
    }

TS_FP_ATTN_ROWS(ts_flash_decode_rows_f16_d128, 128, half, false)
TS_FP_ATTN_ROWS(ts_flash_decode_rows_f16_d256, 256, half, false)

// ts_flash_decode_rows_f16_d<hd> with the window's rows in the row tiles of one launch (ts_fp_attn's WIN): every key tile is
// read once for all the rows, where the row grid read the window's keys once a row (8 times for a Qwen3.8-27B 8-row window:
// 140 us a layer after a 4K prompt against the direct engine's 44). Each row's sums are its one-row launch's, bit for bit,
// its splits its own length's (`splits` holds the last row's, the most). No selection (cells). Launch: grid
// (ceil(rows * nHead / nKv / 128) * splits, nKv), block 256, the same shared memory.
#define TS_FP_ATTN_WINDOW(NAME, HD, T, Q8) \
    extern "C" __global__ void __launch_bounds__(256) NAME( \
        const float* __restrict__ q, const T* __restrict__ kc, const T* __restrict__ vc, int cap, int rows, int len, \
        int floor, int nHead, int nKv, float scale, int splits, float* __restrict__ partial, float* __restrict__ out, \
        const int* __restrict__ dyn, int windowKeys, int bucket, int sms) \
    { \
        ts_fp_attn<HD, true, false, Q8>(q, kc, vc, cap, rows, len, floor, nHead, nKv, scale, splits, partial, out, dyn, nullptr, nullptr, \
            nullptr, ts_fp_live_splits(splits, dyn, 0, windowKeys, bucket, nHead / nKv, nKv, sms), windowKeys, bucket, sms); \
    }

TS_FP_ATTN_WINDOW(ts_flash_decode_window_f16_d128, 128, half, false)
TS_FP_ATTN_WINDOW(ts_flash_decode_window_f16_d256, 256, half, false)

// ---------------------------------------------------------------------------
// 8-bit prompt attention (see the header). The band of one sequence's keys, per KV head h of nKv, for a band of bandKeys
// keys (a multiple of 32), indexed by absolute key as the cache is:
//   k8 [nKv][bandKeys][256] int8, one row a key;  ks [nKv][bandKeys] F32 (amax / 127; 0 for a key at or past len)
//   vt [nKv][bandKeys / 32][256][32] E4M3 (V^T, a tile's 32 keys in ts_fp_a8_perm order)
//   vs [nKv][bandKeys / 32] UE8M0 (the smallest e with the tile's amax / 2^e <= 448, stored e + 127)
// ---------------------------------------------------------------------------
#if defined(__CUDA_ARCH__) && (defined(__CUDA_ARCH_FEAT_SM120_ALL) || defined(__CUDA_ARCH_FEAT_SM121_ALL))
#define TS_FP_A8 1
#else
#define TS_FP_A8 0
#endif

#define TS_FP_A8_HD 256
#define TS_FP_A8_KSTRIDE (TS_FP_A8_HD + 16)   // smem K row: 272 bytes, conflict-free B fragments
#define TS_FP_A8_VSTRIDE (TS_FP_KT + 16)      // smem V^T row: 48 bytes
#define TS_FP_A8_BAND_VROW (TS_FP_A8_HD + 8)  // the band kernel's smem V row, in halves

/// out[0] = 1 when this image carries the 8-bit prompt attention's block-scaled E4M3 MMA (an sm_120a / sm_121a image),
/// else 0. A launch, not a symbol lookup: a fatbin holds both kinds of image. One thread.
extern "C" __global__ void ts_flash_prefill_attn8_caps(int* out)
{
    out[0] = TS_FP_A8;
}

// Key order in a V^T tile: A-fragment column j = 4 q + i (quad lane q, byte i) of a 32-key m16n8k32 step holds key
// perm(j) = 16 (j >> 4) + {2q, 2q + 1, 8 + 2q, 9 + 2q}[i], exactly the keys a thread's score accumulators of n-tiles (0, 1)
// and (2, 3) hold, so P never leaves registers.
__device__ __forceinline__ int ts_fp_a8_perm(int j)
{
    const int q = (j & 15) >> 2, i = j & 3;
    return (j & 16) + (i < 2 ? 2 * q + i : 8 + 2 * q + i - 2);
}

__device__ __forceinline__ float ts_fp_ex2(float x)
{
    float y;
    asm("ex2.approx.ftz.f32 %0, %1;" : "=f"(y) : "f"(x));
    return y;
}

__device__ __forceinline__ void ts_fp_mma_s8(int (&d)[4], const uint32_t (&a)[4], uint32_t b0, uint32_t b1)
{
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
    asm volatile("mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
                 : "+r"(d[0]), "+r"(d[1]), "+r"(d[2]), "+r"(d[3])
                 : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
#else
    __trap();
#endif
}

// D += A * (B * 2^(sfb - 127)), E4M3 operands, A unscaled. sfb: the same UE8M0 byte in every lane.
__device__ __forceinline__ void ts_fp_mma_e4m3_sb(float (&d)[4], const uint32_t (&a)[4], uint32_t b0, uint32_t b1, uint32_t sfb)
{
#if TS_FP_A8
    asm volatile("mma.sync.aligned.kind::mxf8f6f4.block_scale.scale_vec::1X.m16n8k32.row.col.f32.e4m3.e4m3.f32.ue8m0 "
                 "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3}, {%10}, {%11,%12}, {%13}, {%11,%12};"
                 : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
                 : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1), "r"(0x7f7f7f7fu),
                   "h"((unsigned short)0), "h"((unsigned short)0), "r"(sfb));
#else
    __trap();
#endif
}

// Two F32 -> E4M3 (round to nearest even, saturating), lo in the low byte.
__device__ __forceinline__ uint32_t ts_fp_e4m3x2(float lo, float hi)
{
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 890
    unsigned short r;
    asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(r) : "f"(hi), "f"(lo));
    return r;
#else
    __trap();
    return 0;
#endif
}

// Sum of the four E4M3 bytes of a word as F32 (the rounded probabilities the value product consumes).
__device__ __forceinline__ float ts_fp_e4m3x4_sum(uint32_t w)
{
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 890
    uint32_t lo, hi;
    asm("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(lo) : "h"((unsigned short)(w & 0xffffu)));
    asm("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(hi) : "h"((unsigned short)(w >> 16)));
    const float2 a = __half22float2(*reinterpret_cast<const half2*>(&lo));
    const float2 b = __half22float2(*reinterpret_cast<const half2*>(&hi));
    return (a.x + a.y) + (b.x + b.y);
#else
    __trap();
    return 0.0f;
#endif
}

// One 16-byte part (8 halves) of a key row of the head-first F16 cache [nKv][cap][256]: the band's only reader of the
// cache, so another cache format or layout is another source here.
__device__ __forceinline__ uint4 ts_fp_a8_row16(const half* __restrict__ cache, long long row, int part)
{
    return *reinterpret_cast<const uint4*>(cache + row * TS_FP_A8_HD + part * 8);
}

// The band of the keys [floor & ~31, len) of one sequence's head-first F16 caches kc, vc [nKv][cap][256] (layout above):
// the direct engine's band arithmetic (ts_q35_a8_band_tile) - K per key: s = amax / 127 (1 for a zero row), codes
// round-to-nearest-even of x / s clamped to [-127, 127]; V per 32-key tile: x / 2^e in E4M3 (round to nearest even,
// saturating) - read straight from the F16 rows (no F32 staging). Keys of a partial last tile at or past len are zeros
// (scale 0). Launch: grid ((len + 31) / 32 - floor / 32, nKv), block 256, no dynamic shared memory.
extern "C" __global__ void __launch_bounds__(256) ts_flash_prompt_band_f16_d256(
    const half* __restrict__ kc, const half* __restrict__ vc, int cap, int len, int floor, int bandKeys,
    int8_t* __restrict__ k8, float* __restrict__ ks, uint8_t* __restrict__ vt, uint8_t* __restrict__ vs)
{
#if TS_FP_A8
    __shared__ __align__(16) half sv[TS_FP_KT * TS_FP_A8_BAND_VROW];
    __shared__ float red[8];
    __shared__ float vinv;
    const int kvh = blockIdx.y, tile = floor / TS_FP_KT + (int)blockIdx.x, first = tile * TS_FP_KT;
    const int valid = min(TS_FP_KT, len - first);
    const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
    const long long cacheRow = (long long)kvh * cap + first, bandRow = (long long)kvh * bandKeys + first;

    // V: the tile's rows into shared memory (a warp a key, 16 bytes a lane), their amax on the way.
    float vmax = 0.0f;
#pragma unroll
    for (int key = warp; key < TS_FP_KT; key += 8)
    {
        const uint4 w = key < valid ? ts_fp_a8_row16(vc, cacheRow + key, lane) : make_uint4(0u, 0u, 0u, 0u);
        *reinterpret_cast<uint4*>(sv + key * TS_FP_A8_BAND_VROW + lane * 8) = w;
        const half2* h = reinterpret_cast<const half2*>(&w);
#pragma unroll
        for (int e = 0; e < 4; ++e)
        {
            const float2 f = __half22float2(h[e]);
            vmax = fmaxf(vmax, fmaxf(fabsf(f.x), fabsf(f.y)));
        }
    }
    // K: warp w takes keys w, w + 8, w + 16, w + 24; a lane owns 8 dims.
#pragma unroll
    for (int key = warp; key < TS_FP_KT; key += 8)
    {
        const uint4 w = key < valid ? ts_fp_a8_row16(kc, cacheRow + key, lane) : make_uint4(0u, 0u, 0u, 0u);
        const half2* h = reinterpret_cast<const half2*>(&w);
        float x[8], amax = 0.0f;
#pragma unroll
        for (int e = 0; e < 4; ++e)
        {
            const float2 f = __half22float2(h[e]);
            x[2 * e] = f.x;
            x[2 * e + 1] = f.y;
            amax = fmaxf(amax, fmaxf(fabsf(f.x), fabsf(f.y)));
        }
#pragma unroll
        for (int o = 16; o > 0; o >>= 1) amax = fmaxf(amax, __shfl_xor_sync(0xffffffffu, amax, o));
        const float s = amax > 0.0f ? amax / 127.0f : 1.0f, inv = 1.0f / s;
        uint32_t w0 = 0, w1 = 0;
#pragma unroll
        for (int e = 0; e < 8; ++e)
        {
            const int c = max(-127, min(127, __float2int_rn(x[e] * inv)));
            if (e < 4) w0 |= (uint32_t)(c & 0xff) << (8 * e); else w1 |= (uint32_t)(c & 0xff) << (8 * (e - 4));
        }
        *reinterpret_cast<uint2*>(k8 + (bandRow + key) * TS_FP_A8_HD + lane * 8) = make_uint2(w0, w1);
        if (lane == 0) ks[bandRow + key] = key < valid ? s : 0.0f;
    }
    // V's tile scale: one power of two.
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) vmax = fmaxf(vmax, __shfl_xor_sync(0xffffffffu, vmax, o));
    if (lane == 0) red[warp] = vmax;
    __syncthreads();
    if (tid == 0)
    {
        float m = 0.0f;
        for (int w = 0; w < 8; ++w) m = fmaxf(m, red[w]);
        // Smallest e with m / 2^e <= 448 (m == 0: e = 0).
        int e = 0;
        if (m > 0.0f) { frexpf(m / 448.0f, &e); if (ldexpf(1.0f, e - 1) >= m / 448.0f) --e; }
        e = max(-127, min(127, e));
        vinv = ldexpf(1.0f, -e);
        vs[(long long)kvh * (bandKeys / TS_FP_KT) + tile] = (uint8_t)(e + 127);
    }
    __syncthreads();
    const float inv = vinv;
    // V^T: word (d, q) of the tile's [256][32] bytes holds E4M3(V[perm(4q + i)][d] / 2^e), i = 0..3; a warp writes 128
    // contiguous bytes.
    uint32_t* vtTile = reinterpret_cast<uint32_t*>(vt + bandRow * TS_FP_A8_HD);
    for (int i = tid; i < TS_FP_A8_HD * TS_FP_KT / 4; i += 256)
    {
        const int d = i >> 3, j = (i & 7) * 4;
        const float a = __half2float(sv[ts_fp_a8_perm(j) * TS_FP_A8_BAND_VROW + d]) * inv;
        const float b = __half2float(sv[ts_fp_a8_perm(j + 1) * TS_FP_A8_BAND_VROW + d]) * inv;
        const float c = __half2float(sv[ts_fp_a8_perm(j + 2) * TS_FP_A8_BAND_VROW + d]) * inv;
        const float e = __half2float(sv[ts_fp_a8_perm(j + 3) * TS_FP_A8_BAND_VROW + d]) * inv;
        vtTile[i] = ts_fp_e4m3x2(a, b) | ts_fp_e4m3x2(c, e) << 16;
    }
#else
    __trap();
#endif
}

// Dynamic shared memory of ts_flash_prompt_i8e4_d256: two stages of a 32-key K8 tile, its V^T tile and its K scales
// (CudaFlashPrefillKernels.Attn8SharedBytes computes the same: 42,240 bytes).
struct TsFpA8Smem
{
    int8_t k[2][TS_FP_KT * TS_FP_A8_KSTRIDE];
    uint8_t vt[2][TS_FP_A8_HD * TS_FP_A8_VSTRIDE];
    float ks[2][TS_FP_KT];
};

// 8-bit causal attention of one sequence's prompt rows over their band (ts_flash_prompt_band_f16_d256): q [heads][nq][256]
// F32 (the query heads of KV head h are its G = heads / nKv consecutive heads), out [nq][heads][256] F32. Query t sees keys
// [floor, len - nq + t]. GQA-packed rows as the 16-bit prompt kernel's (row r of KV head h: query r / G of head h G + r % G),
// 128 rows a block (8 warps x 16), the row tiles heaviest first; 32-key band tiles in two cp.async stages. Q quantized per
// row in registers (amax / 127); S = Q K^T in int32 (exact), scaled by the row's and the key's scales and scale * log2(e);
// the online softmax in base 2 with P = 2^(s - m + 8) in E4M3 (at most 256, small probabilities stay normal), the row sum
// over the rounded P the value product uses; P V on the block-scaled E4M3 MMA, V's tile scale as the B scale. A warp skips
// the key tiles past all its rows' ends. splits > 1: block x = (row tile, key split), each split a 32-key-aligned share of
// its rows' keys as the 16-bit kernel splits them, writing (acc, max * ln 2, sum) partials that ts_flash_prefill_combine_f32
// merges (acc and sum carry the same 2^8, which cancels). Launch: grid (ceil(nq * G / 128) * splits, nKv), block 256,
// dynamic shared memory sizeof(TsFpA8Smem). The direct engine's ts_q35_attn8_prefill_f32 (038fb303) on these operands:
// with one split and no floor a row's output is that kernel's before its gate, bit for bit.
extern "C" __global__ void __launch_bounds__(256, 1) ts_flash_prompt_i8e4_d256(
    const float* __restrict__ q, const int8_t* __restrict__ k8, const float* __restrict__ ks,
    const uint8_t* __restrict__ vt, const uint8_t* __restrict__ vs, int bandKeys, int nq, int len, int floor,
    int nHead, int nKv, float scale, int splits, float* __restrict__ partial, float* __restrict__ out)
{
#if TS_FP_A8
    extern __shared__ __align__(16) uint8_t ts_fp_a8_smem[];
    TsFpA8Smem& sm = *reinterpret_cast<TsFpA8Smem*>(ts_fp_a8_smem);
    const int kvh = blockIdx.y, G = nHead / nKv, R = nq * G;
    const int split = (int)(blockIdx.x % splits), tileOfBlock = (int)(blockIdx.x / splits);
    const int rowBase = ((R + TS_FP_ROWS - 1) / TS_FP_ROWS - 1 - tileOfBlock) * TS_FP_ROWS;
    const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5, g = lane >> 2, qd = lane & 3;
    const int past = len - nq;   // keys before the first query row
    // The block's keys (its last row's), this split's share of them (the 16-bit kernel's ranges), from the floor's tile.
    const int kAll = past + (min(rowBase + TS_FP_ROWS, R) - 1) / G + 1;
    const int share = ((kAll + splits - 1) / splits + TS_FP_KT - 1) / TS_FP_KT * TS_FP_KT;
    const int kLo = max(min(kAll, split * share), min(kAll, floor / TS_FP_KT * TS_FP_KT));
    const int kHi = min(kAll, split * share + share);
    const long long head = (long long)kvh * bandKeys;
    const int8_t* kh = k8 + head * TS_FP_A8_HD;
    const float* ksh = ks + head;
    const uint8_t* vth = vt + head * TS_FP_A8_HD;
    const uint8_t* vsh = vs + (long long)kvh * (bandKeys / TS_FP_KT);

    const int r0 = rowBase + warp * 16 + g, r1 = r0 + 8;
    const bool live0 = r0 < R, live1 = r1 < R;
    const int end0 = live0 ? past + r0 / G + 1 : 0, end1 = live1 ? past + r1 / G + 1 : 0;
    // A row's keys are [floor, end): one unsigned comparison of key - floor against the span.
    const unsigned span0 = (unsigned)max(end0 - floor, 0), span1 = (unsigned)max(end1 - floor, 0);
    const bool warpLive = rowBase + warp * 16 < R;
    int warpEnd = max(end0, end1);   // MMAs need the whole warp: skip tiles past every row's end
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) warpEnd = max(warpEnd, __shfl_xor_sync(0xffffffffu, warpEnd, o));

    auto load = [&](int st, int first) {
        const int8_t* kt = kh + (long long)first * TS_FP_A8_HD;
        const uint8_t* vtt = vth + (long long)first * TS_FP_A8_HD;
        for (int c = tid; c < TS_FP_KT * TS_FP_A8_HD / 16; c += blockDim.x)
        {
            const int key = c / (TS_FP_A8_HD / 16), part = c % (TS_FP_A8_HD / 16);
            ts_fp_cp16(sm.k[st] + key * TS_FP_A8_KSTRIDE + part * 16, kt + c * 16, true);
            const int d = c / (TS_FP_KT / 16), half_ = c % (TS_FP_KT / 16);
            ts_fp_cp16(sm.vt[st] + d * TS_FP_A8_VSTRIDE + half_ * 16, vtt + c * 16, true);
        }
        if (tid < TS_FP_KT / 4)
            ts_fp_cp16(sm.ks[st] + tid * 4, ksh + first + tid * 4, true);
    };
    if (kHi > kLo)
        load(0, kLo);
    ts_fp_cp_commit();

    // Q rows in registers as INT8 A fragments with one F32 scale per row (amax over the quad).
    uint32_t qa[8][4];
    float qs0 = 0.0f, qs1 = 0.0f;
    {
        const float* q0 = live0 ? q + ((long long)(kvh * G + r0 % G) * nq + r0 / G) * TS_FP_A8_HD : nullptr;
        const float* q1 = live1 ? q + ((long long)(kvh * G + r1 % G) * nq + r1 / G) * TS_FP_A8_HD : nullptr;
        float a0 = 0.0f, a1 = 0.0f;
#pragma unroll
        for (int k = 0; k < 8; ++k)
#pragma unroll
            for (int h = 0; h < 2; ++h)
            {
                const int c = k * 32 + h * 16 + 4 * qd;
                if (q0) { const float4 v = *reinterpret_cast<const float4*>(q0 + c); a0 = fmaxf(a0, fmaxf(fmaxf(fabsf(v.x), fabsf(v.y)), fmaxf(fabsf(v.z), fabsf(v.w)))); }
                if (q1) { const float4 v = *reinterpret_cast<const float4*>(q1 + c); a1 = fmaxf(a1, fmaxf(fmaxf(fabsf(v.x), fabsf(v.y)), fmaxf(fabsf(v.z), fabsf(v.w)))); }
            }
        a0 = fmaxf(a0, __shfl_xor_sync(0xffffffffu, a0, 1)); a0 = fmaxf(a0, __shfl_xor_sync(0xffffffffu, a0, 2));
        a1 = fmaxf(a1, __shfl_xor_sync(0xffffffffu, a1, 1)); a1 = fmaxf(a1, __shfl_xor_sync(0xffffffffu, a1, 2));
        qs0 = a0 > 0.0f ? a0 / 127.0f : 1.0f;
        qs1 = a1 > 0.0f ? a1 / 127.0f : 1.0f;
        const float i0 = 1.0f / qs0, i1 = 1.0f / qs1;
        auto pack = [](const float* p, float inv) {
            if (!p) return 0u;
            const float4 v = *reinterpret_cast<const float4*>(p);
            const int c0 = max(-127, min(127, __float2int_rn(v.x * inv))), c1 = max(-127, min(127, __float2int_rn(v.y * inv)));
            const int c2 = max(-127, min(127, __float2int_rn(v.z * inv))), c3 = max(-127, min(127, __float2int_rn(v.w * inv)));
            return (uint32_t)(c0 & 0xff) | (uint32_t)(c1 & 0xff) << 8 | (uint32_t)(c2 & 0xff) << 16 | (uint32_t)(c3 & 0xff) << 24;
        };
#pragma unroll
        for (int k = 0; k < 8; ++k)
        {
            const int c = k * 32 + 4 * qd;
            qa[k][0] = pack(q0 ? q0 + c : nullptr, i0);
            qa[k][1] = pack(q1 ? q1 + c : nullptr, i1);
            qa[k][2] = pack(q0 ? q0 + c + 16 : nullptr, i0);
            qa[k][3] = pack(q1 ? q1 + c + 16 : nullptr, i1);
        }
    }
    const float log2Scale = scale * 1.4426950408889634f;
    const float sc0 = qs0 * log2Scale, sc1 = qs1 * log2Scale;

    float acc[32][4];
#pragma unroll
    for (int n = 0; n < 32; ++n)
        acc[n][0] = acc[n][1] = acc[n][2] = acc[n][3] = 0.0f;
    float m0 = -INFINITY, m1 = -INFINITY, l0 = 0.0f, l1 = 0.0f;

    // The tiles [kLo, kHi): both 32-key aligned except kHi = kAll, past which every row's end lies, so no key outside the
    // split's share reaches a row.
    for (int first = kLo, t = 0; first < kHi; first += TS_FP_KT, ++t)
    {
        const int st = t & 1;
        ts_fp_cp_wait0();
        __syncthreads();   // this tile landed; every warp is done with the previous one
        if (first + TS_FP_KT < kHi)
            load(st ^ 1, first + TS_FP_KT);
        ts_fp_cp_commit();
        if (!warpLive || first >= warpEnd)
            continue;

        // S = Q K^T (int32, exact) for 4 n-tiles of 8 keys.
        int si[4][4];
#pragma unroll
        for (int nt = 0; nt < 4; ++nt) si[nt][0] = si[nt][1] = si[nt][2] = si[nt][3] = 0;
        const int8_t* kb = sm.k[st] + g * TS_FP_A8_KSTRIDE + 4 * qd;
#pragma unroll
        for (int k = 0; k < 8; ++k)
#pragma unroll
            for (int nt = 0; nt < 4; ++nt)
            {
                const int8_t* p = kb + nt * 8 * TS_FP_A8_KSTRIDE + k * 32;
                ts_fp_mma_s8(si[nt], qa[k], *reinterpret_cast<const uint32_t*>(p), *reinterpret_cast<const uint32_t*>(p + 16));
            }

        // Base-2 scores, the causal mask and the floor, tile maxima.
        float s[4][4], tm0 = -INFINITY, tm1 = -INFINITY;
#pragma unroll
        for (int nt = 0; nt < 4; ++nt)
#pragma unroll
            for (int e = 0; e < 2; ++e)
            {
                const int key = first + nt * 8 + 2 * qd + e;
                const unsigned rel = (unsigned)(key - floor);
                const float kscale = sm.ks[st][nt * 8 + 2 * qd + e];
                s[nt][e] = rel < span0 ? (float)si[nt][e] * sc0 * kscale : -INFINITY;
                s[nt][2 + e] = rel < span1 ? (float)si[nt][2 + e] * sc1 * kscale : -INFINITY;
                tm0 = fmaxf(tm0, s[nt][e]);
                tm1 = fmaxf(tm1, s[nt][2 + e]);
            }
        tm0 = fmaxf(tm0, __shfl_xor_sync(0xffffffffu, tm0, 1)); tm0 = fmaxf(tm0, __shfl_xor_sync(0xffffffffu, tm0, 2));
        tm1 = fmaxf(tm1, __shfl_xor_sync(0xffffffffu, tm1, 1)); tm1 = fmaxf(tm1, __shfl_xor_sync(0xffffffffu, tm1, 2));
        const float n0 = fmaxf(m0, tm0), n1 = fmaxf(m1, tm1);
        const float al0 = n0 == -INFINITY ? 1.0f : ts_fp_ex2(m0 - n0), al1 = n1 == -INFINITY ? 1.0f : ts_fp_ex2(m1 - n1);
        m0 = n0; m1 = n1;
        const float b0 = n0 == -INFINITY ? 0.0f : 8.0f - n0, b1 = n1 == -INFINITY ? 0.0f : 8.0f - n1;

        // P = 2^(s - m + 8) in E4M3, straight into the A fragment (the key order above).
        auto p2 = [](float x, float bias) { return x == -INFINITY ? 0.0f : ts_fp_ex2(x + bias); };
        uint32_t pa[4];
        pa[0] = ts_fp_e4m3x2(p2(s[0][0], b0), p2(s[0][1], b0)) | ts_fp_e4m3x2(p2(s[1][0], b0), p2(s[1][1], b0)) << 16;
        pa[1] = ts_fp_e4m3x2(p2(s[0][2], b1), p2(s[0][3], b1)) | ts_fp_e4m3x2(p2(s[1][2], b1), p2(s[1][3], b1)) << 16;
        pa[2] = ts_fp_e4m3x2(p2(s[2][0], b0), p2(s[2][1], b0)) | ts_fp_e4m3x2(p2(s[3][0], b0), p2(s[3][1], b0)) << 16;
        pa[3] = ts_fp_e4m3x2(p2(s[2][2], b1), p2(s[2][3], b1)) | ts_fp_e4m3x2(p2(s[3][2], b1), p2(s[3][3], b1)) << 16;
        float ps0 = ts_fp_e4m3x4_sum(pa[0]) + ts_fp_e4m3x4_sum(pa[2]);
        float ps1 = ts_fp_e4m3x4_sum(pa[1]) + ts_fp_e4m3x4_sum(pa[3]);
        ps0 += __shfl_xor_sync(0xffffffffu, ps0, 1); ps0 += __shfl_xor_sync(0xffffffffu, ps0, 2);
        ps1 += __shfl_xor_sync(0xffffffffu, ps1, 1); ps1 += __shfl_xor_sync(0xffffffffu, ps1, 2);
        l0 = fmaf(l0, al0, ps0);
        l1 = fmaf(l1, al1, ps1);
        if (__any_sync(0xffffffffu, al0 != 1.0f || al1 != 1.0f))
        {
#pragma unroll
            for (int n = 0; n < 32; ++n)
            {
                acc[n][0] *= al0; acc[n][1] *= al0;
                acc[n][2] *= al1; acc[n][3] *= al1;
            }
        }

        // O += P V^T: 32 n-tiles of 8 dims, one k32 step (the tile), V's tile scale as the B scale.
        const uint32_t sfb = vsh[first / TS_FP_KT] * 0x01010101u;
        const uint8_t* vb = sm.vt[st] + g * TS_FP_A8_VSTRIDE + 4 * qd;
#pragma unroll
        for (int n = 0; n < 32; ++n)
        {
            const uint8_t* p = vb + n * 8 * TS_FP_A8_VSTRIDE;
            ts_fp_mma_e4m3_sb(acc[n], pa, *reinterpret_cast<const uint32_t*>(p), *reinterpret_cast<const uint32_t*>(p + 16), sfb);
        }
    }
    ts_fp_cp_wait0();

    if (splits == 1)
    {
        // O / l (the 2^8 of P cancels), rows [token][head][256].
        const float inv0 = l0 > 0.0f ? 1.0f / l0 : 0.0f, inv1 = l1 > 0.0f ? 1.0f / l1 : 0.0f;
        float* o0 = live0 ? out + ((long long)(r0 / G) * nHead + kvh * G + r0 % G) * TS_FP_A8_HD : nullptr;
        float* o1 = live1 ? out + ((long long)(r1 / G) * nHead + kvh * G + r1 % G) * TS_FP_A8_HD : nullptr;
#pragma unroll
        for (int n = 0; n < 32; ++n)
        {
            const int d = n * 8 + 2 * qd;
            if (o0)
            {
                o0[d] = acc[n][0] * inv0;
                o0[d + 1] = acc[n][1] * inv0;
            }
            if (o1)
            {
                o1[d] = acc[n][2] * inv1;
                o1[d + 1] = acc[n][3] * inv1;
            }
        }
        return;
    }
    // A split's unnormalized partial of each live row (an empty share's too: max -inf, sum 0).
#pragma unroll
    for (int half_ = 0; half_ < 2; ++half_)
    {
        const int r = half_ ? r1 : r0;
        if (r >= R)
            continue;
        const long long base = (long long)(r / G) * nHead + kvh * G + r % G;
        float* p = partial + (base * splits + split) * (TS_FP_A8_HD + 2);
#pragma unroll
        for (int n = 0; n < 32; ++n)
        {
            const int x = n * 8 + qd * 2;
            p[x] = acc[n][half_ * 2];
            p[x + 1] = acc[n][half_ * 2 + 1];
        }
        if (qd == 0)
        {
            const float m = half_ ? m1 : m0;
            p[TS_FP_A8_HD] = m == -INFINITY ? -INFINITY : m * 0.6931471805599453f;   // log2 -> ln domain
            p[TS_FP_A8_HD + 1] = half_ ? l1 : l0;
        }
    }
#else
    __trap();
#endif
}

// The split partials of one (query, head): out = sum_s e^(m_s - m) acc_s / sum_s e^(m_s - m) l_s, over the splits the
// attention launch ran (ts_fp_live_splits with its arguments: `rows` its rule's rows); with one, it wrote out itself.
// Launch: grid nq * heads, block HD.
extern "C" __global__ void ts_flash_prefill_combine_f32(const float* __restrict__ partial, int splits, int hd, float* __restrict__ out,
    const int* __restrict__ dyn, int lenBias, int windowKeys, int bucket, int rows, int nKv, int sms)
{
    const int live = ts_fp_live_splits(splits, dyn, lenBias, windowKeys, bucket, rows, nKv, sms);
    if (live <= 1)
        return;
    const long long base = blockIdx.x;
    const float* p = partial + base * splits * (hd + 2);
    float mx = -INFINITY;
    for (int s = 0; s < live; ++s)
        mx = fmaxf(mx, p[s * (hd + 2) + hd]);
    float num = 0.0f, den = 0.0f;
    const int x = threadIdx.x;
    for (int s = 0; s < live; ++s)
    {
        const float* ps = p + s * (hd + 2);
        const float w = mx == -INFINITY ? 0.0f : expf(ps[hd] - mx);
        num = fmaf(ps[x], w, num);
        den = fmaf(ps[hd + 1], w, den);
    }
    out[base * hd + x] = den > 0.0f ? num / den : 0.0f;
}

// ts_flash_prefill_combine_f32 over a verify window's rows (ts_flash_decode_window / ts_flash_decode_rows without a
// selection): output (t, head) combines the splits its row ran - its own length's under the bucket rule (row t of `rows`
// attends the window's length, dyn[0] when recorded, less the rows - 1 - t keys after it), as the decode row at its position
// combines them. Launch: grid rows * nHead, block HD.
extern "C" __global__ void ts_flash_window_combine_f32(const float* __restrict__ partial, int splits, int hd, float* __restrict__ out,
    const int* __restrict__ dyn, int len, int rows, int nHead, int windowKeys, int bucket, int ruleRows, int nKv, int sms)
{
    const long long base = blockIdx.x;
    const int t = (int)(base / nHead);
    const int live = ts_fp_rule_splits(splits, (dyn != nullptr ? dyn[0] : len) - (rows - 1 - t), windowKeys, bucket, ruleRows, nKv, sms);
    if (live <= 1)
        return;
    const float* p = partial + base * splits * (hd + 2);
    float mx = -INFINITY;
    for (int s = 0; s < live; ++s)
        mx = fmaxf(mx, p[s * (hd + 2) + hd]);
    float num = 0.0f, den = 0.0f;
    const int x = threadIdx.x;
    for (int s = 0; s < live; ++s)
    {
        const float* ps = p + s * (hd + 2);
        const float w = mx == -INFINITY ? 0.0f : expf(ps[hd] - mx);
        num = fmaf(ps[x], w, num);
        den = fmaf(ps[hd + 1], w, den);
    }
    out[base * hd + x] = den > 0.0f ? num / den : 0.0f;
}

// The kernels over a q8_0 cache (ts_fp_attn's Q8): each the F16 kernel of its name with the q8_0 tile loader and dequant
// stage, the same operands and launch (the dynamic shared memory of TsFpAttnSmemQ8), the combine kernels shared. Last in the
// module, so the kernels above keep their committed PTX.
TS_FP_ATTN(ts_flash_prefill_q8_d128, 128, uint8_t, true)
TS_FP_ATTN(ts_flash_prefill_q8_d256, 256, uint8_t, true)
TS_FP_PROMPT(ts_flash_prompt_q8_d128, 128, uint8_t, true)
TS_FP_PROMPT(ts_flash_prompt_q8_d256, 256, uint8_t, true)
TS_FP_ATTN_ROWS(ts_flash_decode_rows_q8_d128, 128, uint8_t, true)
TS_FP_ATTN_ROWS(ts_flash_decode_rows_q8_d256, 256, uint8_t, true)
TS_FP_ATTN_WINDOW(ts_flash_decode_window_q8_d128, 128, uint8_t, true)
TS_FP_ATTN_WINDOW(ts_flash_decode_window_q8_d256, 256, uint8_t, true)
