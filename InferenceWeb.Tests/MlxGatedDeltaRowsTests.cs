// Copyright (c) Zhongkai Fu. All rights reserved.
// Licensed under the BSD-3-Clause license in the repository root.
using System;
using System.Runtime.InteropServices;
using TensorSharp;
using TensorSharp.MLX;
using Xunit;

namespace InferenceWeb.Tests;

/// <summary>
/// MlxNative.GatedDeltaRows - a speculative verify window's GatedDeltaNet over T rows that also returns the state after
/// every row (GatedDeltaNetCache.KeepRowStates, one dispatch a layer) - against the same rows run one GatedDelta step at
/// a time, each from the previous row's state: the outputs and every row's state are the same bits.
/// </summary>
public class MlxGatedDeltaRowsTests
{
    [MlxTheory]
    [InlineData(8, 2, 4, 128, 128)]
    [InlineData(5, 16, 48, 128, 128)]
    public void GatedDeltaRows_EqualsOneStepARow(int rows, int hk, int hv, int dk, int dv)
    {
        var rng = new Random(rows * 31 + hv);
        float[] F(int n, float scale) { var a = new float[n]; for (int i = 0; i < n; i++) a[i] = (float)(rng.NextDouble() * 2 - 1) * scale; return a; }
        float[] q = F(rows * hk * dk, 0.1f), k = F(rows * hk * dk, 0.1f), v = F(rows * hv * dv, 1f);
        float[] g = new float[rows * hv], beta = new float[rows * hv];
        for (int i = 0; i < g.Length; i++) { g[i] = 0.9f + 0.1f * (float)rng.NextDouble(); beta[i] = (float)rng.NextDouble(); }
        float[] s0 = F(hv * dv * dk, 0.05f);

        MlxNative.MlxArray Arr(float[] data, params int[] shape)
        {
            var h = GCHandle.Alloc(data, GCHandleType.Pinned);
            try { return MlxNative.NewArrayFromHost(h.AddrOfPinnedObject(), shape, DType.Float32); }
            finally { h.Free(); }
        }
        float[] Read(MlxNative.MlxArray a, int n)
        {
            MlxNative.Eval(a);
            var r = new float[n];
            var h = GCHandle.Alloc(r, GCHandleType.Pinned);
            try { MlxNative.CopyArrayToHost(a, DType.Float32, h.AddrOfPinnedObject(), (long)n * 4); }
            finally { h.Free(); }
            return r;
        }

        var qa = Arr(q, 1, rows, hk, dk); var ka = Arr(k, 1, rows, hk, dk); var va = Arr(v, 1, rows, hv, dv);
        var ga = Arr(g, 1, rows, hv); var ba = Arr(beta, 1, rows, hv); var sa = Arr(s0, 1, hv, dv, dk);
        MlxNative.GatedDeltaRows(qa, ka, va, ga, ba, sa, 1, rows, hk, hv, dk, dv, out var y, out var last, out var states);
        float[] yFused = Read(y, rows * hv * dv), statesFused = Read(states, rows * hv * dv * dk), lastFused = Read(last, hv * dv * dk);

        int stateLen = hv * dv * dk;
        var state = Arr(s0, 1, hv, dv, dk);
        for (int r = 0; r < rows; r++)
        {
            var qr = Arr(q.AsSpan(r * hk * dk, hk * dk).ToArray(), 1, 1, hk, dk);
            var kr = Arr(k.AsSpan(r * hk * dk, hk * dk).ToArray(), 1, 1, hk, dk);
            var vr = Arr(v.AsSpan(r * hv * dv, hv * dv).ToArray(), 1, 1, hv, dv);
            var gr = Arr(g.AsSpan(r * hv, hv).ToArray(), 1, 1, hv);
            var br = Arr(beta.AsSpan(r * hv, hv).ToArray(), 1, 1, hv);
            MlxNative.GatedDelta(qr, kr, vr, gr, br, state, 1, 1, hk, hv, dk, dv, out var yr, out var next);
            float[] yStep = Read(yr, hv * dv), sStep = Read(next, stateLen);
            for (int i = 0; i < hv * dv; i++)
                Assert.True(Math.Abs(yStep[i] - yFused[r * hv * dv + i]) <= 1e-5f * (1 + Math.Abs(yStep[i])), $"row {r} output {i}: {yFused[r * hv * dv + i]} vs {yStep[i]}");
            for (int i = 0; i < stateLen; i++)
                Assert.True(Math.Abs(sStep[i] - statesFused[(long)r * stateLen + i]) <= 1e-5f * (1 + Math.Abs(sStep[i])), $"row {r} state {i}");
            if (r == rows - 1)
                for (int i = 0; i < stateLen; i++) Assert.Equal(statesFused[(long)r * stateLen + i], lastFused[i]);
            MlxNative.FreeArray(state);
            state = next;
            foreach (var a in new[] { qr, kr, vr, gr, br, yr }) MlxNative.FreeArray(a);
        }
        foreach (var a in new[] { qa, ka, va, ga, ba, sa, y, last, states, state }) MlxNative.FreeArray(a);
    }
}
