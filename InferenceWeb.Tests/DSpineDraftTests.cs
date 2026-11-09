// Copyright (c) Zhongkai Fu. All rights reserved.
// Licensed under the BSD-3-Clause license in the repository root.
using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Runtime.InteropServices;
using TensorSharp.GGML;
using TensorSharp.Models;
using TensorSharp.Runtime;
using Xunit;
using Xunit.Abstractions;

namespace InferenceWeb.Tests;

/// <summary>
/// The DSpine drafter (Tencent DSpine on the DFlash layout, eng/dspine-to-gguf.py):
/// <list type="bullet">
/// <item>its GGUF keys parse, and the variants no TensorSharp graph runs as trained are refused;</item>
/// <item>the host walk is the reference's greedy chain, ties to the lower token id;</item>
/// <item>the native draft graph (ggml_ops_dspine.cpp) on the pinned GGML backend against eng/dspine-reference.py's
/// expected values (TensorSharp.GGML.Native/tests/data/dspine_draft_reference.bin): the refinement tables, the
/// top-K candidate sets, the lattice scores and the greedy chain of three drafts, with F32 and BF16 tables. The weights
/// are drawn from the reference's LCG, so the fixture holds only expected values. The native ctest
/// (dspine_draft_test.cpp) also checks every intermediate stage and a 1280-wide config.</item>
/// </list>
/// </summary>
public sealed class DSpineDraftTests
{
    private readonly ITestOutputHelper _output;
    public DSpineDraftTests(ITestOutputHelper output) => _output = output;

    // ---------------------------------------------------------------------------------------------------------------
    // the GGUF keys
    // ---------------------------------------------------------------------------------------------------------------

    private static List<SyntheticGguf.Kv> DraftKv(bool causal = true, uint version = 1, uint topK = 16,
        bool dspine = true, bool selector = false)
    {
        var kv = new List<SyntheticGguf.Kv>
        {
            new SyntheticGguf.Str { Key = "general.architecture", V = "dflash" },
            new SyntheticGguf.U32 { Key = "dflash.block_count", V = 5 },
            new SyntheticGguf.U32 { Key = "dflash.embedding_length", V = 64 },
            new SyntheticGguf.U32 { Key = "dflash.feed_forward_length", V = 128 },
            new SyntheticGguf.U32 { Key = "dflash.attention.head_count", V = 4 },
            new SyntheticGguf.U32 { Key = "dflash.attention.head_count_kv", V = 2 },
            new SyntheticGguf.U32 { Key = "dflash.attention.key_length", V = 16 },
            new SyntheticGguf.U32 { Key = "dflash.attention.value_length", V = 16 },
            new SyntheticGguf.F32 { Key = "dflash.attention.layer_norm_rms_epsilon", V = 1e-6f },
            new SyntheticGguf.F32 { Key = "dflash.rope.freq_base", V = 1e7f },
            new SyntheticGguf.U32 { Key = "dflash.block_size", V = 16 },
            new SyntheticGguf.I32Arr { Key = "dflash.target_layers", V = new[] { 2, 17, 32, 47, 62 } },
            new SyntheticGguf.U32 { Key = "dflash.attention.sliding_window", V = 8192 },
            new SyntheticGguf.U32 { Key = "tokenizer.ggml.mask_token_id", V = 248070 },
            new SyntheticGguf.Bool { Key = "dflash.attention.causal", V = causal },
        };
        if (dspine)
        {
            kv.Add(new SyntheticGguf.U32 { Key = "dflash.dspine.version", V = version });
            kv.Add(new SyntheticGguf.U32 { Key = "dflash.dspine.code_size", V = 1024 });
            kv.Add(new SyntheticGguf.U32 { Key = "dflash.dspine.rank", V = 512 });
            kv.Add(new SyntheticGguf.F32 { Key = "dflash.dspine.eps", V = 1e-6f });
            kv.Add(new SyntheticGguf.U32 { Key = "dflash.dspine.head_topk", V = topK });
        }
        if (selector)
        {
            kv.Add(new SyntheticGguf.U32 { Key = "dflash.selector_rank", V = 256 });
            kv.Add(new SyntheticGguf.U32 { Key = "dflash.selector_top_k", V = 16 });
        }
        return kv;
    }

    private static DFlashConfig Parse(List<SyntheticGguf.Kv> kv)
    {
        string path = Path.Combine(Path.GetTempPath(), "dspine-cfg-" + Guid.NewGuid().ToString("N") + ".gguf");
        try
        {
            SyntheticGguf.Write(path, kv, new List<SyntheticGguf.Tensor>());
            using var gguf = new GgufFile(path);
            return DFlashConfig.FromGguf(gguf);
        }
        finally
        {
            File.Delete(path);
        }
    }

    [Fact]
    public void DSpineKeys_Parse()
    {
        DFlashConfig cfg = Parse(DraftKv());
        Assert.True(cfg.IsDSpine);
        Assert.True(cfg.CausalBlock);
        Assert.False(cfg.IsDFlash2);
        Assert.Equal(1024, cfg.DSpineCodeSize);
        Assert.Equal(512, cfg.DSpineRank);
        Assert.Equal(16, cfg.DSpineTopK);
        Assert.Equal(1e-6f, cfg.DSpineEps);
        // The anchor row is not a draft: 15 drafts from a 16-row block.
        Assert.Equal(15, cfg.MaxDraftTokens);
        // window + block + anchor, rounded up to 32: the block always fits past the window.
        Assert.Equal(8224, cfg.RingRows);
        Assert.StartsWith("dspine(", cfg.ToString());
        Assert.Contains("bridge=c1024/r512, refine k16", cfg.ToString());
    }

    [Fact]
    public void DFlashFiles_StayNonCausalAndNotDSpine()
    {
        DFlashConfig cfg = Parse(DraftKv(causal: false, dspine: false, selector: true));
        Assert.False(cfg.IsDSpine);
        Assert.False(cfg.CausalBlock);
        Assert.True(cfg.HasSelector);
    }

    [Fact]
    public void UnrunnableVariants_AreRefused()
    {
        // A version this TensorSharp does not read.
        var ex = Assert.Throws<NotSupportedException>(() => Parse(DraftKv(version: 2)));
        Assert.Contains("dflash.dspine.version 2", ex.Message);
        // A bridge without the causal block, and a causal block without a bridge.
        Assert.Throws<NotSupportedException>(() => Parse(DraftKv(causal: false)));
        ex = Assert.Throws<NotSupportedException>(() => Parse(DraftKv(dspine: false)));
        Assert.Contains("block-causal", ex.Message);
        // A bridge next to a DFlash2 selector.
        Assert.Throws<NotSupportedException>(() => Parse(DraftKv(selector: true)));
        // A refinement width that is not a power of two.
        Assert.Throws<InvalidOperationException>(() => Parse(DraftKv(topK: 12)));
    }

    // ---------------------------------------------------------------------------------------------------------------
    // the walk
    // ---------------------------------------------------------------------------------------------------------------

    [Fact]
    public void Walk_IsTheGreedyChain_WithTiesToTheLowerId()
    {
        const int k = 3, gamma = 3;
        // Candidates out of id order (as the native top-k returns them).
        int[] cand = { 30, 10, 20, /* row 1 */ 5, 7, 6, /* row 2 */ 9, 8, 1 };
        float[] scores = new float[k + k * k * (gamma - 1)];
        // Row 0: lanes 0 and 1 tie; id 10 < 30, so lane 1.
        scores[0] = 2f; scores[1] = 2f; scores[2] = 1f;
        // Row 1 given predecessor lane 1: lane 2 (id 6) best.
        float[] r1 = { 0f, 0.5f, 1.5f };
        Array.Copy(r1, 0, scores, k + 1 * k, k);
        // Row 2 given predecessor lane 2: lanes 0 and 2 tie; id 1 < 9, so lane 2.
        float[] r2 = { 3f, 1f, 3f };
        Array.Copy(r2, 0, scores, k + k * k + 2 * k, k);

        var drafts = new int[gamma];
        var conf = new float[gamma];
        Assert.Equal(gamma, ModelBase.DSpineWalkLattice(gamma, k, scores, cand, drafts, conf));
        Assert.Equal(new[] { 10, 6, 1 }, drafts);
        // The confidence is the softmax of the chosen row at the choice.
        float e = MathF.Exp(1.5f - 1.5f) / (MathF.Exp(0f - 1.5f) + MathF.Exp(0.5f - 1.5f) + 1f);
        Assert.Equal(e, conf[1], 5);

        // Without ties it is DFlash2's walk.
        var rng = new Random(5);
        float[] s2 = Enumerable.Range(0, scores.Length).Select(_ => (float)rng.NextDouble()).ToArray();
        var a = new int[gamma];
        var b = new int[gamma];
        ModelBase.DSpineWalkLattice(gamma, k, s2, cand, a, null);
        ModelBase.DFlashWalkLattice(gamma, k, s2, cand, b, null);
        Assert.Equal(b, a);
    }

    // ---------------------------------------------------------------------------------------------------------------
    // the native draft against the NumPy reference
    // ---------------------------------------------------------------------------------------------------------------

    [GgmlTheory(BackendType.GgmlCpu)]
    [InlineData(BackendType.GgmlCpu)]
    public void NativeDraft_MatchesTheReference_Cpu(BackendType backend) => NativeDraftMatchesReference(backend);

    [GgmlTheory(BackendType.GgmlMetal)]
    [InlineData(BackendType.GgmlMetal)]
    public void NativeDraft_MatchesTheReference_Metal(BackendType backend) => NativeDraftMatchesReference(backend);

    private sealed class Fixture
    {
        public readonly Dictionary<string, (uint[] Dims, float[] F, int[] I)> Arrays = new(StringComparer.Ordinal);

        public Fixture(string path)
        {
            using var r = new BinaryReader(File.OpenRead(path));
            Assert.Equal("DSPF", new string(r.ReadChars(4)));
            Assert.Equal(1u, r.ReadUInt32());
            uint count = r.ReadUInt32();
            for (uint e = 0; e < count; e++)
            {
                string name = System.Text.Encoding.UTF8.GetString(r.ReadBytes(r.ReadUInt16()));
                byte dt = r.ReadByte(), nd = r.ReadByte();
                var dims = new uint[nd];
                long n = 1;
                for (int i = 0; i < nd; i++) { dims[i] = r.ReadUInt32(); n *= dims[i]; }
                if (dt == 0)
                    Arrays[name] = (dims, Enumerable.Range(0, (int)n).Select(_ => r.ReadSingle()).ToArray(), null);
                else
                    Arrays[name] = (dims, null, Enumerable.Range(0, (int)n).Select(_ => r.ReadInt32()).ToArray());
            }
        }

        public float[] F(string name) => Arrays[name].F;
        public int[] I(string name) => Arrays[name].I;
    }

    /// <summary>The reference's LCG: state = state * 1664525 + 1013904223, value = (int(state >> 16) - 32768) * mag / 32768.</summary>
    private sealed class Lcg
    {
        private uint _state;
        public Lcg(uint seed) => _state = seed;
        public float[] Take(int n, float mag)
        {
            var v = new float[n];
            float scale = mag / 32768f;
            for (int i = 0; i < n; i++)
            {
                _state = unchecked(_state * 1664525u + 1013904223u);
                v[i] = (float)((int)(_state >> 16) - 32768) * scale;
            }
            return v;
        }
    }

    private static float MagIn(int fanIn) => 1.75f / MathF.Sqrt(fanIn);

    private static ushort ToBf16(float f)
    {
        uint u = BitConverter.SingleToUInt32Bits(f);
        u += 0x7FFFu + ((u >> 16) & 1u);
        return (ushort)(u >> 16);
    }

    private static float[] Bf16Values(float[] v)
        => v.Select(x => BitConverter.UInt32BitsToSingle((uint)ToBf16(x) << 16)).ToArray();

    private static float[] PlusOne(float[] v) => v.Select(x => 1f + x).ToArray();

    /// <summary>
    /// Unmanaged copies of the weights and rings (the native side keys device copies by these pointers), page-aligned
    /// as the allocator's tensors are: the inject writes a ring in place only where the backend can wrap its host
    /// pointer, and AllocHGlobal's 16-byte alignment is below what the CPU buffer type needs.
    /// </summary>
    private sealed class Native : IDisposable
    {
        private readonly List<IntPtr> _blocks = new();
        private IntPtr Alloc(int bytes)
        {
            IntPtr raw = Marshal.AllocHGlobal(bytes + 16384);
            _blocks.Add(raw);
            return (IntPtr)(((long)raw + 16383) & ~16383L);
        }
        public IntPtr F32(float[] v)
        {
            IntPtr p = Alloc(v.Length * sizeof(float));
            Marshal.Copy(v, 0, p, v.Length);
            return p;
        }
        public IntPtr Bf16(float[] v)
        {
            short[] bits = v.Select(x => unchecked((short)ToBf16(x))).ToArray();
            IntPtr p = Alloc(bits.Length * 2);
            Marshal.Copy(bits, 0, p, bits.Length);
            return p;
        }
        public void Dispose()
        {
            foreach (IntPtr p in _blocks) Marshal.FreeHGlobal(p);
            _blocks.Clear();
        }
    }

    private void NativeDraftMatchesReference(BackendType backend)
    {
        var fx = new Fixture(Path.Combine(LoraHostFixtures.RepoRoot(), "TensorSharp.GGML.Native", "tests", "data",
            "dspine_draft_reference.bin"));
        int[] c = fx.I("small.config");
        int H = c[0], NH = c[1], NKV = c[2], HD = c[3], FF = c[4], L = c[5], NT = c[6], C = c[7], R = c[8], V = c[9],
            K = c[10], SEED = c[13], SW = c[12], RING = c[14];
        int FEAT = NT * H, qd = NH * HD, kvd = NKV * HD;
        const int F32 = 0, BF16 = 30;

        using var mem = new Native();
        var g = new Lcg((uint)SEED);
        GgmlDSpineWeight Mat(float[] v, long cols, long rows, bool bf16) => bf16
            ? new GgmlDSpineWeight(mem.Bf16(v), BF16, cols, rows, v.LongLength * 2)
            : new GgmlDSpineWeight(mem.F32(v), F32, cols, rows, v.LongLength * 4);
        var w = new GgmlDSpineWeight[GgmlBasicOps.DSpineGlobals + L * GgmlBasicOps.DSpinePerLayer];
        w[GgmlBasicOps.DSpineTokEmbd] = Mat(g.Take(V * H, 1f), H, V, false);
        w[GgmlBasicOps.DSpineLmHead] = Mat(g.Take(V * H, 3f * MagIn(H)), H, V, false);
        float[] fc = g.Take(H * FEAT, MagIn(FEAT));
        float[] encNorm = PlusOne(g.Take(H, .25f));
        w[GgmlBasicOps.DSpineOutNorm] = Mat(PlusOne(g.Take(H, .25f)), H, 1, false);
        w[GgmlBasicOps.DSpineCodebook] = Mat(Bf16Values(g.Take(V * C, MagIn(C))), C, V, true);
        w[GgmlBasicOps.DSpineBridge] = Mat(Bf16Values(g.Take(3 * R * C, MagIn(C))), C, 3 * R, true);
        w[GgmlBasicOps.DSpineGateBias] = Mat(g.Take(R, .5f), R, 1, false);
        var kArr = new IntPtr[L];
        var vArr = new IntPtr[L];
        var kNorm = new IntPtr[L];
        for (int l = 0; l < L; l++)
        {
            int s = GgmlBasicOps.DSpineGlobals + l * GgmlBasicOps.DSpinePerLayer;
            w[s + GgmlBasicOps.DSpineAttnNorm] = Mat(PlusOne(g.Take(H, .25f)), H, 1, false);
            w[s + GgmlBasicOps.DSpineQ] = Mat(g.Take(qd * H, MagIn(H)), H, qd, false);
            w[s + GgmlBasicOps.DSpineK] = Mat(g.Take(kvd * H, MagIn(H)), H, kvd, false);
            w[s + GgmlBasicOps.DSpineV] = Mat(g.Take(kvd * H, MagIn(H)), H, kvd, false);
            w[s + GgmlBasicOps.DSpineQNorm] = Mat(PlusOne(g.Take(HD, .25f)), HD, 1, false);
            w[s + GgmlBasicOps.DSpineKNorm] = Mat(PlusOne(g.Take(HD, .25f)), HD, 1, false);
            w[s + GgmlBasicOps.DSpineO] = Mat(g.Take(H * qd, MagIn(qd)), qd, H, false);
            w[s + GgmlBasicOps.DSpineFfnNorm] = Mat(PlusOne(g.Take(H, .25f)), H, 1, false);
            w[s + GgmlBasicOps.DSpineGate] = Mat(g.Take(FF * H, MagIn(H)), H, FF, false);
            w[s + GgmlBasicOps.DSpineUp] = Mat(g.Take(FF * H, MagIn(H)), H, FF, false);
            w[s + GgmlBasicOps.DSpineDown] = Mat(g.Take(H * FF, MagIn(FF)), FF, H, false);
            w[s + GgmlBasicOps.DSpineRead] = Mat(Bf16Values(g.Take(C * H, MagIn(H))), H, C, true);
            w[s + GgmlBasicOps.DSpineWrite] = Mat(Bf16Values(g.Take(H * R, .5f * MagIn(R))), R, H, true);
            kArr[l] = w[s + GgmlBasicOps.DSpineK].Data;
            vArr[l] = w[s + GgmlBasicOps.DSpineV].Data;
            kNorm[l] = w[s + GgmlBasicOps.DSpineKNorm].Data;
        }
        IntPtr fcPtr = mem.F32(fc), encPtr = mem.F32(encNorm);
        var ringK = new IntPtr[L];
        var ringV = new IntPtr[L];
        float[] zeros = new float[RING * kvd];
        for (int l = 0; l < L; l++) { ringK[l] = mem.F32(zeros); ringV[l] = mem.F32(zeros); }

        var p = new GgmlDSpineParams
        {
            NumLayers = L, Hidden = H, NumHeads = NH, NumKvHeads = NKV, HeadDim = HD, Vocab = V, CodeSize = C, Rank = R,
            TopK = K, RingRows = RING, RingDtype = F32, SlidingWindow = SW, Eps = 1e-6f, BridgeEps = 1e-6f,
            RopeBase = 10000f, RopeFreqScale = 1f, KqScale = 1f / MathF.Sqrt(HD),
        };
        IntPtr tF32 = GgmlBasicOps.DSpineTablesCreate(p, w, F32);
        IntPtr tBf16 = GgmlBasicOps.DSpineTablesCreate(p, w, BF16);
        Assert.True(tF32 != IntPtr.Zero && tBf16 != IntPtr.Zero, GgmlBasicOps.LastNativeError());
        bool metal = backend == BackendType.GgmlMetal;
        double latticeTol = metal ? 2e-3 : 4e-3, marginTol = metal ? 1e-2 : 2e-2, gapTol = marginTol;
        try
        {
            // The tables (F32 storage) at the fixture's rows.
            int[] rows = fx.I("small.table_rows");
            var one = new float[2 * R];
            double tableErr = 0, tableMax = 1e-30;
            for (int i = 0; i < rows.Length; i++)
            {
                Assert.True(GgmlBasicOps.DSpineTablesRead(tF32, 1, rows[i], 1, one), GgmlBasicOps.LastNativeError());
                for (int j = 0; j < 2 * R; j++)
                {
                    float rf = fx.F("small.tsv")[i * 2 * R + j];
                    tableErr = Math.Max(tableErr, Math.Abs(one[j] - rf));
                    tableMax = Math.Max(tableMax, Math.Abs(rf));
                }
            }
            _output.WriteLine($"TSV (F32) normalized max error {tableErr / tableMax:G3}");
            Assert.True(tableErr / tableMax < 1e-3);

            int filled = 0;
            int[][] steps = { new[] { 0, 40, 7 }, new[] { 40, 5, 11 }, new[] { 45, 75, 3 } };
            int checkedSteps = 0;
            for (int si = 0; si < steps.Length; si++)
            {
                // Inject the step's context rows (each position's features from its own LCG stream).
                int start = steps[si][0], n = steps[si][1], anchor = steps[si][2], b = 16, gamma = b - 1;
                float[] feats = new float[n * FEAT];
                var idx = new long[n];
                var pos = new int[n];
                for (int i = 0; i < n; i++)
                {
                    Array.Copy(new Lcg((uint)(977 * (start + i) + 12345)).Take(FEAT, 1f), 0, feats, i * FEAT, FEAT);
                    idx[i] = (start + i) % RING;
                    pos[i] = start + i;
                }
                Assert.True(GgmlBasicOps.DFlashInject(feats, FEAT, n, idx, pos, L, H, HD, NKV, RING, 1e-6f, 10000f, 1f,
                    fcPtr, F32, FEAT, H, (long)fc.Length * 4, encPtr,
                    kArr, Enumerable.Repeat(F32, L).ToArray(), Enumerable.Repeat((long)H, L).ToArray(),
                    Enumerable.Repeat((long)kvd, L).ToArray(), Enumerable.Repeat((long)H * kvd * 4, L).ToArray(),
                    vArr, Enumerable.Repeat(F32, L).ToArray(), Enumerable.Repeat((long)H, L).ToArray(),
                    Enumerable.Repeat((long)kvd, L).ToArray(), Enumerable.Repeat((long)H * kvd * 4, L).ToArray(),
                    kNorm, ringK, ringV, F32), GgmlBasicOps.LastNativeError());
                filled = start + n;

                var map = Enumerable.Repeat(-1, RING).ToArray();
                for (int q = Math.Max(0, filled - RING); q < filled; q++) map[q % RING] = q;
                var ids = Enumerable.Repeat(V - 1, b).ToArray();
                ids[0] = anchor;
                var positions = Enumerable.Range(filled, b).ToArray();
                foreach (var (tables, tname) in new[] { (tF32, "f32"), (tBf16, "bf16") })
                {
                    var scores = new float[K + K * K * (gamma - 1)];
                    var cand = new int[K * gamma];
                    Assert.True(GgmlBasicOps.DSpineDraftBlock(p, w, tables, ringK, ringV, ids, b, positions, map, scores, cand),
                        GgmlBasicOps.LastNativeError());
                    checkedSteps += Compare(fx, $"small.step{si}", tname, K, gamma, cand, scores, latticeTol, marginTol, gapTol);
                }
            }
            _output.WriteLine($"{backend}: {checkedSteps} chain steps matched the reference");
            Assert.True(checkedSteps >= 40, $"only {checkedSteps} chain steps were checked");
        }
        finally
        {
            GgmlBasicOps.DSpineResetCaches();
            GgmlBasicOps.DSpineTablesFree(tF32);
            GgmlBasicOps.DSpineTablesFree(tBf16);
            GgmlBasicOps.DFlashResetCaches();
            GgmlBasicOps.ClearHostBufferCache();
        }
    }

    /// <summary>Candidates (by id), lattice scores and the greedy chain of one draft against the fixture; returns the
    /// chain steps compared (a walk stops where the reference's own best-two margin is within float noise).</summary>
    private int Compare(Fixture fx, string step, string tname, int k, int gamma, int[] cand, float[] scores,
        double latticeTol, double marginTol, double gapTol)
    {
        // Canonical order: each row's candidates sorted by id, the scores permuted to match.
        var order = new int[gamma][];
        for (int e = 0; e < gamma; e++)
            order[e] = Enumerable.Range(0, k).OrderBy(i => cand[e * k + i]).ToArray();
        int[] refCand = fx.I(step + ".cand");
        float[] kgap = fx.F(step + ".kgap");
        var same = new bool[gamma];
        for (int e = 0; e < gamma; e++)
        {
            same[e] = Enumerable.Range(0, k).All(i => cand[e * k + order[e][i]] == refCand[e * k + i]);
            Assert.True(same[e] || kgap[e] < gapTol, $"{step} row {e}: candidate set differs (k-gap {kgap[e]})");
        }
        float Score(int e, int p, int c) => e == 0
            ? scores[order[0][c]]
            : scores[k + (e - 1) * k * k + order[e - 1][p] * k + order[e][c]];

        string pre = $"{step}.{tname}";
        if (fx.Arrays.ContainsKey(pre + ".scores"))
        {
            float[] rs = fx.F(pre + ".scores");
            double err = 0, max = rs.Max(Math.Abs) + 1e-30;
            for (int e = 0; e < gamma; e++)
            {
                if (!same[e] || (e > 0 && !same[e - 1])) continue;
                for (int p = 0; p < (e == 0 ? 1 : k); p++)
                    for (int c = 0; c < k; c++)
                        err = Math.Max(err, Math.Abs(Score(e, p, c) - rs[(e * k + p) * k + c]));
            }
            Assert.True(err / max < latticeTol, $"{pre}: lattice normalized error {err / max:G3}");
        }

        int[] chain = fx.I(pre + ".chain");
        float[] margin = fx.F(pre + ".margin");
        var drafts = new int[gamma];
        var sortedCand = new int[k * gamma];
        var sortedScores = new float[scores.Length];
        for (int e = 0; e < gamma; e++)
            for (int i = 0; i < k; i++)
                sortedCand[e * k + i] = cand[e * k + order[e][i]];
        for (int c = 0; c < k; c++) sortedScores[c] = Score(0, 0, c);
        for (int e = 1; e < gamma; e++)
            for (int p = 0; p < k; p++)
                for (int c = 0; c < k; c++)
                    sortedScores[k + (e - 1) * k * k + p * k + c] = Score(e, p, c);
        ModelBase.DSpineWalkLattice(gamma, k, sortedScores, sortedCand, drafts, null);
        int checkedSteps = 0;
        for (int e = 0; e < gamma; e++)
        {
            if (!same[e] || margin[e] < marginTol) break;
            Assert.True(drafts[e] == chain[e], $"{pre} chain step {e}: {drafts[e]} vs reference {chain[e]}");
            checkedSteps++;
        }
        return checkedSteps;
    }
}
