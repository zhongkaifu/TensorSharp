// tsm_hc_read.metal - a prompt chunk's hyper-connection read on the matrix units (TensorSharp's tsm library).
//
// The read of a half-layer: xn = rms_norm(each stream) * gamma, the low-rank down projection and the injection
// weights from xn in one GEMM, silu(down / hc), the up projection, and the streams mixed by the up projection's
// sigmoid gates. ggml runs this as a norm, three matmuls and element-wise passes over F32 copies of xn and of the
// up projection's output; here xn is written once in fp16 (the matrix units' operand), the up projection's output is
// never written (the GEMM's epilogue mixes the streams), and the weights are dequantized to fp16 once a chunk.
//
// After Strata (https://github.com/Niko1221/Strata, metal/kernels/prompt.metal gr_norm_h / gr_epi_down / gemm_hc_up
// and metal/kernels/dense_mm.metal gemm_f16, commit 146df3c4, MIT License, Copyright (c) 2026 Niko1221 and the Strata
// contributors; tsm/LICENSE-STRATA). Changes: the norm writes fp16 only and leaves the residual unchanged (the
// scatter stays its own pass), it multiplies by the inverse RMS before gamma (ggml's order), the dequantization
// kernels are TensorSharp's for the ggml types the hyper-connection weights come in (F32, F16, BF16, Q8_0), and the
// up GEMM writes the mix in fp32 only.
//
// Needs MetalPerformancePrimitives (the tsm_moe_mm library's prelude) and ggml-common.h's block_q8_0.

struct TsmHrArgs {
    uint n_tok;
    uint n_embd;   // 2560
    uint hc;       // 4
    uint lr;       // 320: the down projection's rank
    float eps;
    uint pad0, pad1, pad2;
};

// xnh[t][c][:] = half(r[t][c][:] * rsqrt(mean(r[t][c][:]^2) + eps) * gamma[c][:]); one threadgroup per (c, t), 256
// threads.
kernel void tsm_hr_norm(device const float* R [[buffer(0)]], device const float* gamma [[buffer(1)]],
                        device half* xnh [[buffer(2)]], constant TsmHrArgs& a [[buffer(3)]],
                        uint3 tg [[threadgroup_position_in_grid]], ushort tid [[thread_index_in_threadgroup]],
                        ushort sg [[simdgroup_index_in_threadgroup]], ushort lane [[thread_index_in_simdgroup]]) {
    threadgroup float part[8];
    const uint c = tg.x, t = tg.y, n = a.n_embd;
    device const float* r = R + (ulong(t) * a.hc + c) * n;
    float ss = 0.0f;
    for (uint d = tid; d < n; d += 256) {
        const float v = r[d];
        ss = fma(v, v, ss);
    }
    ss = simd_sum(ss);
    if (lane == 0) part[sg] = ss;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float tot = 0.0f;
    for (int i = 0; i < 8; ++i) tot += part[i];
    const float rs = 1.0f / sqrt(tot / float(n) + a.eps);
    device half* o = xnh + (ulong(t) * a.hc + c) * n;
    device const float* w = gamma + c * n;
    for (uint d = tid; d < n; d += 256) o[d] = half((r[d] * rs) * w[d]);
}

// V [T][lr + hc] (the down rows, then the injection rows): lo_h[t][r] = half(silu(V / hc)) for r < lr, inject[t][c] =
// V[t][lr + c]. grid T * (lr + hc) threads.
kernel void tsm_hr_epi_down(device const float* V [[buffer(0)]], device half* lo_h [[buffer(1)]],
                            device float* inject [[buffer(2)]], constant TsmHrArgs& a [[buffer(3)]],
                            uint gid [[thread_position_in_grid]]) {
    const uint nr = a.lr + a.hc, t = gid / nr, r = gid % nr;
    if (t >= a.n_tok) return;
    const float v = V[gid];
    if (r < a.lr) {
        const float z = v * (1.0f / float(a.hc));
        lo_h[ulong(t) * a.lr + r] = half(z / (1.0f + exp(-z)));
    } else {
        inject[ulong(t) * a.hc + r - a.lr] = v;
    }
}

// ---- dequantization to fp16 rows: Wh[dst_row0 + ro(r)][k] for the n_out rows of n_in values. perm > 0 interleaves
// the rows of an [p][n_out / p] matrix: row r = c * (n_out / p) + d lands at d * p + c (the up projection, so a GEMM tile
// holds every stream of its outputs). grid (n_in / 32, n_out) threads; one thread a 32-value run.
struct TsmDqArgs {
    uint n_in, n_out, row_bytes, perm;
    uint dst_row0, pad0, pad1, pad2;
};

static inline device half* tsm_dq_dst(device half* Wh, constant TsmDqArgs& a, uint r, uint b) {
    const uint q = a.perm ? a.n_out / a.perm : 1;
    const uint ro = a.dst_row0 + (a.perm ? (r % q) * a.perm + r / q : r);
    return Wh + ulong(ro) * a.n_in + b * 32;
}

kernel void tsm_dq_q8_0(device const uchar* W [[buffer(0)]], device half* Wh [[buffer(1)]],
                        constant TsmDqArgs& a [[buffer(2)]], uint2 gid [[thread_position_in_grid]]) {
    const uint b = gid.x, r = gid.y;
    if (b * 32 >= a.n_in || r >= a.n_out) return;
    device const block_q8_0* blk = (device const block_q8_0*)(W + ulong(r) * a.row_bytes) + b;
    const float d = float(blk->d);
    device half* o = tsm_dq_dst(Wh, a, r, b);
    for (int j = 0; j < 32; ++j) o[j] = half(float(blk->qs[j]) * d);
}

kernel void tsm_dq_bf16(device const uchar* W [[buffer(0)]], device half* Wh [[buffer(1)]],
                        constant TsmDqArgs& a [[buffer(2)]], uint2 gid [[thread_position_in_grid]]) {
    const uint b = gid.x, r = gid.y;
    if (b * 32 >= a.n_in || r >= a.n_out) return;
    device const ushort* s = (device const ushort*)(W + ulong(r) * a.row_bytes) + b * 32;
    device half* o = tsm_dq_dst(Wh, a, r, b);
    for (int j = 0; j < 32; ++j) o[j] = half(as_type<float>(uint(s[j]) << 16));
}

kernel void tsm_dq_f16(device const uchar* W [[buffer(0)]], device half* Wh [[buffer(1)]],
                       constant TsmDqArgs& a [[buffer(2)]], uint2 gid [[thread_position_in_grid]]) {
    const uint b = gid.x, r = gid.y;
    if (b * 32 >= a.n_in || r >= a.n_out) return;
    device const half* s = (device const half*)(W + ulong(r) * a.row_bytes) + b * 32;
    device half* o = tsm_dq_dst(Wh, a, r, b);
    for (int j = 0; j < 32; ++j) o[j] = s[j];
}

kernel void tsm_dq_f32(device const uchar* W [[buffer(0)]], device half* Wh [[buffer(1)]],
                       constant TsmDqArgs& a [[buffer(2)]], uint2 gid [[thread_position_in_grid]]) {
    const uint b = gid.x, r = gid.y;
    if (b * 32 >= a.n_in || r >= a.n_out) return;
    device const float* s = (device const float*)(W + ulong(r) * a.row_bytes) + b * 32;
    device half* o = tsm_dq_dst(Wh, a, r, b);
    for (int j = 0; j < 32; ++j) o[j] = half(s[j]);
}

// ---- GEMMs on MetalPerformancePrimitives matmul2d (Strata's gemm_f16 / gemm_hc_up)
struct TsmGemmArgs { int M, N, K; int pad; };   // Y is M rows of N floats; A [M][K], B [N][K], fp16

// Y[m][n] = A[m] . B[n] (fp32), TM x TN tiles, K dynamic; rows past M and columns past N are not written.
// grid (ceil(N / TN), ceil(M / TM)), 32 * SG threads.
template <int TM, int TN, int SG>
kernel void tsm_gemm_f16(device half* A [[buffer(0)]], device half* B [[buffer(1)]], device float* Y [[buffer(2)]],
                         constant TsmGemmArgs& a [[buffer(3)]], uint3 tg [[threadgroup_position_in_grid]]) {
    tensor<device half, dextents<int32_t, 2>, tensor_inline> tA(A, dextents<int32_t, 2>(a.K, a.M));
    tensor<device half, dextents<int32_t, 2>, tensor_inline> tB(B, dextents<int32_t, 2>(a.K, a.N));
    tensor<device float, dextents<int32_t, 2>, tensor_inline> tY(Y, dextents<int32_t, 2>(a.N, a.M));
    constexpr auto desc = matmul2d_descriptor(TM, TN, dynamic_length_v<int>, false, true, false);
    matmul2d<desc, execution_simdgroups<SG>> op;
    auto mA = tA.slice(0, int(tg.y) * TM);
    auto mB = tB.slice(0, int(tg.x) * TN);
    auto mY = tY.slice(int(tg.x) * TN, int(tg.y) * TM);
    op.run(mA, mB, mY);
}
template [[host_name("tsm_gemm_f16_64_64_4")]] kernel decltype(tsm_gemm_f16<64, 64, 4>) tsm_gemm_f16<64, 64, 4>;

// The up projection G = lo . Wu^T with Wu's rows interleaved (dequantized with perm = hc: row d * hc + c), so a
// TN-column tile holds TN / hc outputs d with all their streams; the epilogue writes mixed[t][d] = mean over c of
// xnh[t][c][d] * sigmoid(G[t][d * hc + c]). G is never written. grid (hc * n / TN, ceil(T / TM)), 32 * SG threads.
template <int TM, int TN, int SG>
kernel void tsm_gemm_hc_up(device half* A [[buffer(0)]], device half* B [[buffer(1)]], device const half* xnh [[buffer(2)]],
                           device float* mixed [[buffer(3)]], constant TsmGemmArgs& g [[buffer(4)]],
                           constant TsmHrArgs& e [[buffer(5)]], uint3 tg [[threadgroup_position_in_grid]],
                           ushort tid [[thread_index_in_threadgroup]]) {
    threadgroup float ot[TM * TN];
    tensor<device half, dextents<int32_t, 2>, tensor_inline> tA(A, dextents<int32_t, 2>(g.K, g.M));
    tensor<device half, dextents<int32_t, 2>, tensor_inline> tB(B, dextents<int32_t, 2>(g.K, g.N));
    tensor<threadgroup float, dextents<int32_t, 2>, tensor_inline> tO(ot, dextents<int32_t, 2>(TN, TM));
    constexpr auto desc = matmul2d_descriptor(TM, TN, dynamic_length_v<int>, false, true, false);
    matmul2d<desc, execution_simdgroups<SG>> op;
    auto mA = tA.slice(0, int(tg.y) * TM);
    auto mB = tB.slice(0, int(tg.x) * TN);
    op.run(mA, mB, tO);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const int hc = int(e.hc), n = int(e.n_embd), DPT = TN / hc;
    const int t0 = int(tg.y) * TM, d0 = int(tg.x) * DPT;
    for (int idx = tid; idx < TM * DPT; idx += SG * 32) {
        const int m = idx / DPT, dd = idx - m * DPT, t = t0 + m;
        if (t >= int(e.n_tok)) continue;
        float s = 0.0f;
        for (int c = 0; c < hc; ++c)
            s = fma(float(xnh[ulong(t) * hc * n + c * n + d0 + dd]), 1.0f / (1.0f + exp(-ot[m * TN + dd * hc + c])), s);
        mixed[ulong(t) * n + d0 + dd] = s / float(hc);
    }
}
template [[host_name("tsm_gemm_hc_up_64_64_4")]] kernel decltype(tsm_gemm_hc_up<64, 64, 4>) tsm_gemm_hc_up<64, 64, 4>;
