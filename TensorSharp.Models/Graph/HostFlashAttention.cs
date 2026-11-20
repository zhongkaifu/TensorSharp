// Copyright (c) Zhongkai Fu. All rights reserved.
// https://github.com/zhongkaifu/TensorSharp
//
// This file is part of TensorSharp.
//
// TensorSharp is licensed under the BSD-3-Clause license found in the LICENSE file in the root directory of this source tree.
using System;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;
using System.Runtime.Intrinsics;
using System.Runtime.Intrinsics.Arm;

namespace TensorSharp.Models.Graph
{
    /// <summary>
    /// Attention of several query rows (a prompt chunk, a verify window) over an F16 or F32 K/V cache on ARM64, tiled with an
    /// online softmax: each task takes one KV head and a block of query rows for every query head that reads it, walks the
    /// visible keys in blocks (the block's keys widened to F32 once, K transposed so a score tile is FMLA-by-element work),
    /// and keeps a running max, sum and weighted V per query vector. Nothing the size of rows x keys is materialized, the
    /// K/V are read in place (not copied per query head), and a causal mask's invisible key blocks are skipped. Scores below
    /// the running max by more than <see cref="ModelBase.DecodeSoftmaxFlushBelow"/> count as zero (as the decode kernels'
    /// flush). Masks: none, causal aligned to the last key (each with a floor of leading keys every row masks), or any
    /// additive [rows, keys] mask.
    /// </summary>
    internal static unsafe class HostFlashAttention
    {
        internal enum MaskKind { None, Causal, Additive }

        /// <summary>The kernel can run here (ARM64 AdvSimd).</summary>
        internal static bool Supported => AdvSimd.Arm64.IsSupported;

        private const int QueryRows = 16;
        private const int KeyBlock = 64;

        [ThreadStatic] private static float[] _scratch;

        private static float* Scratch(int floats)
        {
            float[] buf = _scratch;
            if (buf == null || buf.Length < floats + 16)
            {
                buf = GC.AllocateUninitializedArray<float>(floats + 16, pinned: true);
                _scratch = buf;
            }
            float* p = (float*)Unsafe.AsPointer(ref MemoryMarshal.GetArrayDataReference(buf));
            return (float*)(((nint)p + 63) & ~(nint)63);
        }

        /// <summary>
        /// output[r, h, :] = softmax(q[h, r] . K[h / group]^T * scale + mask[r]) . V[h / group], for <paramref name="nq"/> rows.
        /// q: [heads, nq, d] (head stride <paramref name="qHeadStride"/>, rows d apart); k, v: [kvHeads, nkv, d] (head stride
        /// <paramref name="kvHeadStride"/> elements, rows d apart), F16 when <paramref name="half"/>; output [nq, heads, d].
        /// Causal: row r sees keys [floor, valid - nq + r + 1) (clamped to [floor, valid)); None: [floor, valid);
        /// Additive: keys [0, valid) plus mask[r * maskRowStride + key] (F32, or F16 when <paramref name="maskHalf"/>).
        /// </summary>
        internal static void Run(float* q, long qHeadStride, void* k, void* v, long kvHeadStride, bool half,
            int heads, int kvHeads, int nq, int d, float scale, MaskKind kind, int floor, int valid,
            void* mask, long maskRowStride, bool maskHalf, float* output)
        {
            int group = heads / kvHeads;
            int qBlocks = (nq + QueryRows - 1) / QueryRows;
            long qa = (long)q, ka = (long)k, va = (long)v, ma = (long)mask, oa = (long)output;
            TensorSharp.Cpu.CpuParallel.For(kvHeads * qBlocks, task =>
            {
                // The last query blocks first: under a causal mask a later block sees more keys, and the pool hands tasks out
                // in index order, so the tasks claimed last are the shortest. A block's result does not depend on when, or on
                // which thread, it runs.
                int qb = qBlocks - 1 - task / kvHeads, kvHead = task % kvHeads;
                int r0 = qb * QueryRows, r1 = Math.Min(nq, r0 + QueryRows);
                RunBlock((float*)qa, qHeadStride, (byte*)ka, (byte*)va, kvHeadStride, half, heads, group, kvHead, nq, d, scale,
                    kind, floor, valid, (byte*)ma, maskRowStride, maskHalf, (float*)oa, r0, r1);
            });
        }

        [MethodImpl(MethodImplOptions.AggressiveOptimization)]
        private static void RunBlock(float* q, long qHeadStride, byte* k, byte* v, long kvHeadStride, bool half,
            int heads, int group, int kvHead, int nq, int d, float scale, MaskKind kind, int floor, int valid,
            byte* mask, long maskRowStride, bool maskHalf, float* output, int r0, int r1)
        {
            int rows = r1 - r0;
            int vecs = group * rows;                  // query vectors: (g, r) at g * rows + (r - r0)
            int vecsPad = (vecs + 3) & ~3;
            int dPad = (d + 15) & ~15;
            // scratch: Q [vecsPad][dPad] | O [vecsPad][dPad] | m, l [vecsPad] | S [vecsPad][KeyBlock] | Kt [dPad][KeyBlock] | V [KeyBlock][dPad]
            int qOff = 0, oOff = qOff + vecsPad * dPad, mOff = oOff + vecsPad * dPad, lOff = mOff + vecsPad;
            int sOff = lOff + vecsPad, ktOff = sOff + vecsPad * KeyBlock, vOff = ktOff + dPad * KeyBlock;
            float* sc = Scratch(vOff + KeyBlock * dPad);
            float* Q = sc + qOff, O = sc + oOff, M = sc + mOff, L = sc + lOff, S = sc + sOff, Kt = sc + ktOff, V = sc + vOff;

            new Span<float>(Q, vecsPad * dPad).Clear();
            new Span<float>(O, vecsPad * dPad).Clear();
            for (int i = 0; i < vecsPad; i++) { M[i] = float.NegativeInfinity; L[i] = 0f; }
            for (int g = 0; g < group; g++)
                for (int r = r0; r < r1; r++)
                    Buffer.MemoryCopy(q + (kvHead * group + g) * qHeadStride + (long)r * d, Q + (long)(g * rows + r - r0) * dPad,
                        d * sizeof(float), d * sizeof(float));

            // Keys any row of the block may see, in blocks on absolute multiples of KeyBlock: a row's online softmax then
            // rescales at the same keys whatever mask kind or floor brought it here (a row's result never depends on how
            // its step was batched).
            int kFirst = kind == MaskKind.Additive ? 0 : floor / KeyBlock * KeyBlock;
            int kEnd = kind == MaskKind.Causal ? Math.Clamp(valid - nq + r1, floor, valid) : valid;
            long elem = half ? 2 : 4;
            byte* kHead = k + kvHead * kvHeadStride * elem, vHead = v + kvHead * kvHeadStride * elem;

            for (int kb = kFirst; kb < kEnd; kb += KeyBlock)
            {
                int nk = Math.Min(KeyBlock, kEnd - kb);
                LoadKeys(kHead, vHead, half, kb, nk, d, dPad, Kt, V);

                // Scores of every query vector over the block's keys (masked keys -inf).
                for (int i = 0; i < vecsPad; i += 4)
                    Scores4(Q + (long)i * dPad, dPad, Kt, d, S + (long)i * KeyBlock, scale);
                for (int i = 0; i < vecs; i++)
                {
                    int r = r0 + i % rows;
                    float* s = S + (long)i * KeyBlock;
                    for (int j = nk; j < KeyBlock; j++) s[j] = float.NegativeInfinity;
                    if (kind != MaskKind.Additive)
                        for (int j = 0; j < nk && kb + j < floor; j++) s[j] = float.NegativeInfinity;
                    switch (kind)
                    {
                        case MaskKind.Causal:
                        {
                            int end = Math.Clamp(valid - nq + r + 1, floor, valid);
                            for (int j = Math.Max(0, end - kb); j < nk; j++) s[j] = float.NegativeInfinity;
                            break;
                        }
                        case MaskKind.Additive:
                        {
                            byte* mr = mask + ((long)r * maskRowStride + kb) * (maskHalf ? 2 : 4);
                            if (maskHalf)
                                for (int j = 0; j < nk; j++) s[j] += (float)BitConverter.UInt16BitsToHalf(((ushort*)mr)[j]);
                            else
                                for (int j = 0; j < nk; j++) s[j] += ((float*)mr)[j];
                            break;
                        }
                    }
                }

                // Online softmax: P = exp(S - m_new) in place, the row's rescale factor in S's padding-free slot.
                for (int i = 0; i < vecs; i++)
                {
                    float* s = S + (long)i * KeyBlock;
                    float bm = MaxRow(s);
                    if (float.IsNegativeInfinity(bm))
                    {
                        new Span<float>(s, KeyBlock).Clear();   // no visible key here: P = 0, O and m unchanged
                        continue;
                    }
                    float mOld = M[i], mNew = MathF.Max(mOld, bm);
                    float alpha = float.IsNegativeInfinity(mOld) ? 0f : ScaleOf(mOld - mNew);
                    L[i] = L[i] * alpha + ExpRow(s, mNew);
                    M[i] = mNew;
                    if (alpha != 1f)
                        ScaleRow(O + (long)i * dPad, alpha, dPad);
                }
                // O += P V
                for (int i = 0; i < vecsPad; i += 4)
                    AccumulatePV(S + (long)i * KeyBlock, O + (long)i * dPad, dPad, V, nk);
            }

            for (int i = 0; i < vecs; i++)
            {
                int g = i / rows, r = r0 + i % rows;
                float inv = 1f / L[i];
                float* o = output + ((long)r * heads + kvHead * group + g) * d;
                float* src = O + (long)i * dPad;
                for (int j = 0; j < d; j++) o[j] = src[j] * inv;
            }
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private static float ScaleOf(float x) => x < ModelBase.DecodeSoftmaxFlushBelow ? 0f : MathF.Exp(x);

        [MethodImpl(MethodImplOptions.AggressiveOptimization)]
        private static float MaxRow(float* s)
        {
            Vector128<float> m = Vector128.Create(float.NegativeInfinity);
            for (int j = 0; j < KeyBlock; j += 4) m = AdvSimd.Max(m, Vector128.Load(s + j));
            return AdvSimd.Arm64.MaxAcross(m).ToScalar();
        }

        /// <summary>s[j] = exp(s[j] - max) (terms below the flush threshold exactly zero); returns their sum.</summary>
        [MethodImpl(MethodImplOptions.AggressiveOptimization)]
        private static float ExpRow(float* s, float max)
        {
            Vector128<float> vm = Vector128.Create(max), fl = Vector128.Create(ModelBase.DecodeSoftmaxFlushBelow);
            Vector128<float> acc = Vector128<float>.Zero;
            for (int j = 0; j < KeyBlock; j += 4)
            {
                Vector128<float> x = Vector128.Load(s + j) - vm;
                Vector128<float> below = Vector128.LessThan(x, fl);
                Vector128<float> e = Vector128.AndNot(Vector128.Exp(Vector128.ConditionalSelect(below, fl, x)), below);
                Vector128.Store(e, s + j);
                acc += e;
            }
            return Vector128.Sum(acc);
        }

        [MethodImpl(MethodImplOptions.AggressiveOptimization)]
        private static void ScaleRow(float* o, float a, int n)
        {
            Vector128<float> va = Vector128.Create(a);
            for (int j = 0; j < n; j += 4) Vector128.Store(Vector128.Load(o + j) * va, o + j);
        }

        /// <summary>The block's keys transposed into Kt [d][KeyBlock] and its values into V [KeyBlock][dPad], widened to F32;
        /// keys past <paramref name="nk"/> zero.</summary>
        [MethodImpl(MethodImplOptions.AggressiveOptimization)]
        private static void LoadKeys(byte* kHead, byte* vHead, bool half, int kb, int nk, int d, int dPad, float* Kt, float* V)
        {
            for (int j = 0; j < KeyBlock; j++)
            {
                float* vr = V + (long)j * dPad;
                if (j >= nk)
                {
                    new Span<float>(vr, dPad).Clear();
                    for (int c = 0; c < d; c++) Kt[(long)c * KeyBlock + j] = 0f;
                    continue;
                }
                long row = (long)(kb + j) * d;
                if (half)
                {
                    ushort* kr = (ushort*)kHead + row, vs = (ushort*)vHead + row;
                    int c = 0;
                    for (; c + 4 <= d; c += 4)
                    {
                        Vector128<float> kv4 = ManagedQuantizedOps.HalfBitsToSingle(Vector128.WidenLower(Vector128.CreateScalar(*(ulong*)(kr + c)).AsUInt16()));
                        Kt[(long)c * KeyBlock + j] = kv4.ToScalar();
                        Kt[(long)(c + 1) * KeyBlock + j] = kv4.GetElement(1);
                        Kt[(long)(c + 2) * KeyBlock + j] = kv4.GetElement(2);
                        Kt[(long)(c + 3) * KeyBlock + j] = kv4.GetElement(3);
                        Vector128.Store(ManagedQuantizedOps.HalfBitsToSingle(Vector128.WidenLower(Vector128.CreateScalar(*(ulong*)(vs + c)).AsUInt16())), vr + c);
                    }
                    for (; c < d; c++)
                    {
                        Kt[(long)c * KeyBlock + j] = (float)BitConverter.UInt16BitsToHalf(kr[c]);
                        vr[c] = (float)BitConverter.UInt16BitsToHalf(vs[c]);
                    }
                }
                else
                {
                    float* kr = (float*)kHead + row, vs = (float*)vHead + row;
                    for (int c = 0; c < d; c++) Kt[(long)c * KeyBlock + j] = kr[c];
                    Buffer.MemoryCopy(vs, vr, d * sizeof(float), d * sizeof(float));
                }
                for (int c = d; c < dPad; c++) vr[c] = 0f;
            }
        }

        /// <summary>Scores of four query vectors (rows dPad apart) over KeyBlock keys: s[i][j] = q_i . k_j * scale.</summary>
        [MethodImpl(MethodImplOptions.AggressiveOptimization)]
        private static void Scores4(float* q, int dPad, float* Kt, int d, float* s, float scale)
        {
            float* q0 = q, q1 = q + dPad, q2 = q + 2 * dPad, q3 = q + 3 * dPad;
            for (int j = 0; j < KeyBlock; j += 8)
            {
                Vector128<float> a00 = default, a01 = default, a10 = default, a11 = default;
                Vector128<float> a20 = default, a21 = default, a30 = default, a31 = default;
                float* kt = Kt + j;
                int c = 0;
                for (; c + 4 <= d; c += 4)
                {
                    Vector128<float> x0 = Vector128.Load(q0 + c), x1 = Vector128.Load(q1 + c);
                    Vector128<float> x2 = Vector128.Load(q2 + c), x3 = Vector128.Load(q3 + c);
                    float* kr;
                    Vector128<float> k0, k1;
                    kr = kt + (long)(c + 0) * KeyBlock;
                    k0 = Vector128.Load(kr); k1 = Vector128.Load(kr + 4);
                    a00 = AdvSimd.Arm64.FusedMultiplyAddBySelectedScalar(a00, k0, x0, 0); a01 = AdvSimd.Arm64.FusedMultiplyAddBySelectedScalar(a01, k1, x0, 0);
                    a10 = AdvSimd.Arm64.FusedMultiplyAddBySelectedScalar(a10, k0, x1, 0); a11 = AdvSimd.Arm64.FusedMultiplyAddBySelectedScalar(a11, k1, x1, 0);
                    a20 = AdvSimd.Arm64.FusedMultiplyAddBySelectedScalar(a20, k0, x2, 0); a21 = AdvSimd.Arm64.FusedMultiplyAddBySelectedScalar(a21, k1, x2, 0);
                    a30 = AdvSimd.Arm64.FusedMultiplyAddBySelectedScalar(a30, k0, x3, 0); a31 = AdvSimd.Arm64.FusedMultiplyAddBySelectedScalar(a31, k1, x3, 0);
                    kr = kt + (long)(c + 1) * KeyBlock;
                    k0 = Vector128.Load(kr); k1 = Vector128.Load(kr + 4);
                    a00 = AdvSimd.Arm64.FusedMultiplyAddBySelectedScalar(a00, k0, x0, 1); a01 = AdvSimd.Arm64.FusedMultiplyAddBySelectedScalar(a01, k1, x0, 1);
                    a10 = AdvSimd.Arm64.FusedMultiplyAddBySelectedScalar(a10, k0, x1, 1); a11 = AdvSimd.Arm64.FusedMultiplyAddBySelectedScalar(a11, k1, x1, 1);
                    a20 = AdvSimd.Arm64.FusedMultiplyAddBySelectedScalar(a20, k0, x2, 1); a21 = AdvSimd.Arm64.FusedMultiplyAddBySelectedScalar(a21, k1, x2, 1);
                    a30 = AdvSimd.Arm64.FusedMultiplyAddBySelectedScalar(a30, k0, x3, 1); a31 = AdvSimd.Arm64.FusedMultiplyAddBySelectedScalar(a31, k1, x3, 1);
                    kr = kt + (long)(c + 2) * KeyBlock;
                    k0 = Vector128.Load(kr); k1 = Vector128.Load(kr + 4);
                    a00 = AdvSimd.Arm64.FusedMultiplyAddBySelectedScalar(a00, k0, x0, 2); a01 = AdvSimd.Arm64.FusedMultiplyAddBySelectedScalar(a01, k1, x0, 2);
                    a10 = AdvSimd.Arm64.FusedMultiplyAddBySelectedScalar(a10, k0, x1, 2); a11 = AdvSimd.Arm64.FusedMultiplyAddBySelectedScalar(a11, k1, x1, 2);
                    a20 = AdvSimd.Arm64.FusedMultiplyAddBySelectedScalar(a20, k0, x2, 2); a21 = AdvSimd.Arm64.FusedMultiplyAddBySelectedScalar(a21, k1, x2, 2);
                    a30 = AdvSimd.Arm64.FusedMultiplyAddBySelectedScalar(a30, k0, x3, 2); a31 = AdvSimd.Arm64.FusedMultiplyAddBySelectedScalar(a31, k1, x3, 2);
                    kr = kt + (long)(c + 3) * KeyBlock;
                    k0 = Vector128.Load(kr); k1 = Vector128.Load(kr + 4);
                    a00 = AdvSimd.Arm64.FusedMultiplyAddBySelectedScalar(a00, k0, x0, 3); a01 = AdvSimd.Arm64.FusedMultiplyAddBySelectedScalar(a01, k1, x0, 3);
                    a10 = AdvSimd.Arm64.FusedMultiplyAddBySelectedScalar(a10, k0, x1, 3); a11 = AdvSimd.Arm64.FusedMultiplyAddBySelectedScalar(a11, k1, x1, 3);
                    a20 = AdvSimd.Arm64.FusedMultiplyAddBySelectedScalar(a20, k0, x2, 3); a21 = AdvSimd.Arm64.FusedMultiplyAddBySelectedScalar(a21, k1, x2, 3);
                    a30 = AdvSimd.Arm64.FusedMultiplyAddBySelectedScalar(a30, k0, x3, 3); a31 = AdvSimd.Arm64.FusedMultiplyAddBySelectedScalar(a31, k1, x3, 3);
                }
                for (; c < d; c++)
                {
                    float* kr = kt + (long)c * KeyBlock;
                    Vector128<float> k0 = Vector128.Load(kr), k1 = Vector128.Load(kr + 4);
                    a00 += k0 * q0[c]; a01 += k1 * q0[c];
                    a10 += k0 * q1[c]; a11 += k1 * q1[c];
                    a20 += k0 * q2[c]; a21 += k1 * q2[c];
                    a30 += k0 * q3[c]; a31 += k1 * q3[c];
                }
                Vector128<float> vs = Vector128.Create(scale);
                Vector128.Store(a00 * vs, s + j); Vector128.Store(a01 * vs, s + j + 4);
                Vector128.Store(a10 * vs, s + KeyBlock + j); Vector128.Store(a11 * vs, s + KeyBlock + j + 4);
                Vector128.Store(a20 * vs, s + 2 * KeyBlock + j); Vector128.Store(a21 * vs, s + 2 * KeyBlock + j + 4);
                Vector128.Store(a30 * vs, s + 3 * KeyBlock + j); Vector128.Store(a31 * vs, s + 3 * KeyBlock + j + 4);
            }
        }

        /// <summary>Four query vectors' O rows (dPad apart) += P (rows KeyBlock apart) . V over the block's keys.</summary>
        [MethodImpl(MethodImplOptions.AggressiveOptimization)]
        private static void AccumulatePV(float* p, float* o, int dPad, float* V, int nk)
        {
            float* p0 = p, p1 = p + KeyBlock, p2 = p + 2 * KeyBlock, p3 = p + 3 * KeyBlock;
            int nk4 = (nk + 3) & ~3;   // keys past nk have P = 0 and V = 0
            for (int c = 0; c < dPad; c += 8)
            {
                float* o0 = o + c, o1 = o + dPad + c, o2 = o + 2 * dPad + c, o3 = o + 3 * dPad + c;
                Vector128<float> a00 = Vector128.Load(o0), a01 = Vector128.Load(o0 + 4);
                Vector128<float> a10 = Vector128.Load(o1), a11 = Vector128.Load(o1 + 4);
                Vector128<float> a20 = Vector128.Load(o2), a21 = Vector128.Load(o2 + 4);
                Vector128<float> a30 = Vector128.Load(o3), a31 = Vector128.Load(o3 + 4);
                for (int j = 0; j < nk4; j += 4)
                {
                    Vector128<float> w0 = Vector128.Load(p0 + j), w1 = Vector128.Load(p1 + j);
                    Vector128<float> w2 = Vector128.Load(p2 + j), w3 = Vector128.Load(p3 + j);
                    float* vr;
                    Vector128<float> v0, v1;
                    vr = V + (long)(j + 0) * dPad + c;
                    v0 = Vector128.Load(vr); v1 = Vector128.Load(vr + 4);
                    a00 = AdvSimd.Arm64.FusedMultiplyAddBySelectedScalar(a00, v0, w0, 0); a01 = AdvSimd.Arm64.FusedMultiplyAddBySelectedScalar(a01, v1, w0, 0);
                    a10 = AdvSimd.Arm64.FusedMultiplyAddBySelectedScalar(a10, v0, w1, 0); a11 = AdvSimd.Arm64.FusedMultiplyAddBySelectedScalar(a11, v1, w1, 0);
                    a20 = AdvSimd.Arm64.FusedMultiplyAddBySelectedScalar(a20, v0, w2, 0); a21 = AdvSimd.Arm64.FusedMultiplyAddBySelectedScalar(a21, v1, w2, 0);
                    a30 = AdvSimd.Arm64.FusedMultiplyAddBySelectedScalar(a30, v0, w3, 0); a31 = AdvSimd.Arm64.FusedMultiplyAddBySelectedScalar(a31, v1, w3, 0);
                    vr = V + (long)(j + 1) * dPad + c;
                    v0 = Vector128.Load(vr); v1 = Vector128.Load(vr + 4);
                    a00 = AdvSimd.Arm64.FusedMultiplyAddBySelectedScalar(a00, v0, w0, 1); a01 = AdvSimd.Arm64.FusedMultiplyAddBySelectedScalar(a01, v1, w0, 1);
                    a10 = AdvSimd.Arm64.FusedMultiplyAddBySelectedScalar(a10, v0, w1, 1); a11 = AdvSimd.Arm64.FusedMultiplyAddBySelectedScalar(a11, v1, w1, 1);
                    a20 = AdvSimd.Arm64.FusedMultiplyAddBySelectedScalar(a20, v0, w2, 1); a21 = AdvSimd.Arm64.FusedMultiplyAddBySelectedScalar(a21, v1, w2, 1);
                    a30 = AdvSimd.Arm64.FusedMultiplyAddBySelectedScalar(a30, v0, w3, 1); a31 = AdvSimd.Arm64.FusedMultiplyAddBySelectedScalar(a31, v1, w3, 1);
                    vr = V + (long)(j + 2) * dPad + c;
                    v0 = Vector128.Load(vr); v1 = Vector128.Load(vr + 4);
                    a00 = AdvSimd.Arm64.FusedMultiplyAddBySelectedScalar(a00, v0, w0, 2); a01 = AdvSimd.Arm64.FusedMultiplyAddBySelectedScalar(a01, v1, w0, 2);
                    a10 = AdvSimd.Arm64.FusedMultiplyAddBySelectedScalar(a10, v0, w1, 2); a11 = AdvSimd.Arm64.FusedMultiplyAddBySelectedScalar(a11, v1, w1, 2);
                    a20 = AdvSimd.Arm64.FusedMultiplyAddBySelectedScalar(a20, v0, w2, 2); a21 = AdvSimd.Arm64.FusedMultiplyAddBySelectedScalar(a21, v1, w2, 2);
                    a30 = AdvSimd.Arm64.FusedMultiplyAddBySelectedScalar(a30, v0, w3, 2); a31 = AdvSimd.Arm64.FusedMultiplyAddBySelectedScalar(a31, v1, w3, 2);
                    vr = V + (long)(j + 3) * dPad + c;
                    v0 = Vector128.Load(vr); v1 = Vector128.Load(vr + 4);
                    a00 = AdvSimd.Arm64.FusedMultiplyAddBySelectedScalar(a00, v0, w0, 3); a01 = AdvSimd.Arm64.FusedMultiplyAddBySelectedScalar(a01, v1, w0, 3);
                    a10 = AdvSimd.Arm64.FusedMultiplyAddBySelectedScalar(a10, v0, w1, 3); a11 = AdvSimd.Arm64.FusedMultiplyAddBySelectedScalar(a11, v1, w1, 3);
                    a20 = AdvSimd.Arm64.FusedMultiplyAddBySelectedScalar(a20, v0, w2, 3); a21 = AdvSimd.Arm64.FusedMultiplyAddBySelectedScalar(a21, v1, w2, 3);
                    a30 = AdvSimd.Arm64.FusedMultiplyAddBySelectedScalar(a30, v0, w3, 3); a31 = AdvSimd.Arm64.FusedMultiplyAddBySelectedScalar(a31, v1, w3, 3);
                }
                Vector128.Store(a00, o0); Vector128.Store(a01, o0 + 4);
                Vector128.Store(a10, o1); Vector128.Store(a11, o1 + 4);
                Vector128.Store(a20, o2); Vector128.Store(a21, o2 + 4);
                Vector128.Store(a30, o3); Vector128.Store(a31, o3 + 4);
            }
        }
    }
}
