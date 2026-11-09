// Copyright (c) Zhongkai Fu. All rights reserved.
// https://github.com/zhongkaifu/TensorSharp
//
// This file is part of TensorSharp.
//
// TensorSharp is licensed under the BSD-3-Clause license found in the LICENSE file in the root directory of this source tree.
//
// TensorSharp is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the BSD-3-Clause License for more details.
//
// ---------------------------------------------------------------------------
// DSpine (Tencent, https://github.com/Tencent/DSpine, Apache-2.0): a DFlash block
// drafter with block-causal attention and a code bridge after every layer, whose
// block is read off an exact-sender refinement of the last bridge write. The
// context side (encoder + ring inject) is DFlash's, unchanged; the draft itself is
// one native graph (TensorSharp.GGML.Native/ggml_ops_dspine.cpp, math in
// eng/dspine-reference.py) that returns DFlash2's lattice layout, so the block is
// walked like DFlash2's - greedily, with ties to the lower token id as the
// reference breaks them.
//
// There is no per-op path: a DSpine drafter attaches only on the ggml backends
// that run that graph (ggml_metal, ggml_cuda, ggml_cpu), and if the graph fails at
// run time the drafter stops drafting (verification then decodes plainly) with
// one warning, never a silent switch to the DFlash per-op draft, which would
// ignore the bridge.
// ---------------------------------------------------------------------------
using System;
using System.Collections.Generic;
using System.Linq;
using TensorSharp.GGML;
using TensorSharp.Runtime;

namespace TensorSharp.Models
{
    public abstract partial class ModelBase
    {
        /// <summary>DSpine's raw tensors: the BF16 bridge (codebook, packed gate weights,
        /// per-layer read/write projections) and any BF16/F16 backbone linear, kept as the
        /// file's bytes for the native graph instead of being dequantized to F32 (the
        /// codebook alone would be 1 GB of F32). Null unless a DSpine drafter is attached.</summary>
        private Dictionary<string, QuantizedWeight> _dspineRaw;

        private sealed class DSpineState
        {
            public GgmlDSpineWeight[] Weights;
            public GgmlDSpineParams Params;
            public IntPtr Tables;
            public float[] Scores;
            public int[] Cand;
            public int[] Ids, Positions;
            public IntPtr[] RingK, RingV;
            public bool FailureLogged;
        }

        private DSpineState _dspineState;

        /// <summary>The ggml type the refinement tables are stored in: BF16, as the reference stores them.</summary>
        private const int DSpineTableType = 30;   // GGML_TYPE_BF16

        private const string DSpineCodebookName = DFlashConfig.WeightPrefix + "dspine.codebook.weight";
        private const string DSpineBridgeName = DFlashConfig.WeightPrefix + "dspine.bridge.weight";
        private const string DSpineGateBiasName = DFlashConfig.WeightPrefix + "dspine.gate.bias";

        private static string DSpineReadName(int il) => $"{DFlashConfig.WeightPrefix}blk.{il}.dspine_read.weight";
        private static string DSpineWriteName(int il) => $"{DFlashConfig.WeightPrefix}blk.{il}.dspine_write.weight";

        /// <summary>Null when this model can run a DSpine draft, else why not.</summary>
        private string DSpineBackendRefusal()
        {
            if (IsTensorParallel)
                return "a DSpine drafter is not supported under tensor parallelism";
            if (_backend != BackendType.GgmlMetal && _backend != BackendType.GgmlCuda && _backend != BackendType.GgmlCpu)
            {
                return $"a DSpine drafter runs as one ggml graph per draft (ggml_metal, ggml_cuda or ggml_cpu); the {_backend} "
                    + "backend has no DSpine path";
            }
            return null;
        }

        /// <summary>True for a tensor of a DSpine file that stays in its file encoding.</summary>
        private static bool KeepDSpineRaw(GgufTensorInfo info)
            => info.Shape.Length == 2
               && (info.Type == GgmlTensorType.BF16 || info.Type == GgmlTensorType.F16
                   || info.Name.StartsWith("dspine.", StringComparison.Ordinal) || info.Name.Contains(".dspine_"));

        private unsafe void LoadDSpineRawTensor(GgufFile draft, GgufTensorInfo info, string name)
        {
            long bytes = draft.GetTensorByteCount(info);
            IntPtr ptr = QuantizedWeight.AllocateBuffer(bytes);
            draft.ReadTensorDataToNative(info, ptr, bytes);
            _dspineRaw ??= new Dictionary<string, QuantizedWeight>(StringComparer.Ordinal);
            _dspineRaw[name] = new QuantizedWeight(ptr, bytes, (int)info.Type, (long)info.Shape[0], (long)info.Shape[1]);
        }

        private bool TryDSpineRaw(string name, out QuantizedWeight qw)
        {
            qw = null;
            return _dspineRaw != null && _dspineRaw.TryGetValue(name, out qw) && qw != null && qw.CacheKey != IntPtr.Zero;
        }

        /// <summary>The DSpine-only tensors VerifyDFlashTensors requires, or null.</summary>
        private string MissingDSpineTensor()
        {
            foreach (string name in new[] { DSpineCodebookName, DSpineBridgeName })
                if (!TryDSpineRaw(name, out _)) return name;
            if (!_weights.ContainsKey(DSpineGateBiasName)) return DSpineGateBiasName;
            for (int il = 0; il < _dflash.NumLayers; il++)
            {
                if (!TryDSpineRaw(DSpineReadName(il), out _)) return DSpineReadName(il);
                if (!TryDSpineRaw(DSpineWriteName(il), out _)) return DSpineWriteName(il);
            }
            return null;
        }

        private bool TryDSpineLinear(string name, out GgmlDSpineWeight w)
        {
            if (TryDFlashQuant(name, out var qw))
            {
                // The DSpine graph binds weights as stored; an NVFP4 scale2 sidecar (which the DFlash graphs
                // multiply in) would be silently dropped.
                if (qw.Scale != 1f)
                {
                    throw new NotSupportedException(
                        $"DSpine drafter: {name} carries a sidecar scale ({qw.Scale}); the DSpine graph binds unscaled weights.");
                }
                w = new GgmlDSpineWeight(qw.CacheKey, qw.GgmlType, qw.Ne0, qw.Ne1, qw.RawBytes);
                return true;
            }
            w = default;
            return false;
        }

        private unsafe bool TryDSpineVector(string name, out GgmlDSpineWeight w)
        {
            if (_weights.TryGetValue(name, out var t))
            {
                long n = t.ElementCount();
                w = new GgmlDSpineWeight((IntPtr)GetFloatPtr(t), 0 /* F32 */, n, 1, n * sizeof(float));
                return true;
            }
            w = default;
            return false;
        }

        /// <summary>
        /// Builds the native weight table and the refinement tables (TG/TSV, on the device)
        /// once the drafter's tensors are merged. Throws with the reason when the graph
        /// cannot run; the codebook's host copy is released once the tables exist.
        /// </summary>
        private void AttachDSpine()
        {
            var cfg = _dflash;
            int n = cfg.NumLayers;
            var w = new GgmlDSpineWeight[GgmlBasicOps.DSpineGlobals + n * GgmlBasicOps.DSpinePerLayer];
            void Need(bool ok, string what)
            {
                if (!ok) throw new InvalidOperationException($"DSpine drafter: {what} is missing or not bindable.");
            }

            Need(TryDSpineLinear("token_embd.weight", out w[GgmlBasicOps.DSpineTokEmbd]), "the target's token_embd");
            Need(TryDSpineLinear(DFlashTargetOutputWeightName, out w[GgmlBasicOps.DSpineLmHead]), "the target's LM head");
            Need(TryDSpineVector(DFlashConfig.WeightPrefix + "output_norm.weight", out w[GgmlBasicOps.DSpineOutNorm]), "output_norm");
            Need(TryDSpineLinear(DSpineCodebookName, out w[GgmlBasicOps.DSpineCodebook]), DSpineCodebookName);
            Need(TryDSpineLinear(DSpineBridgeName, out w[GgmlBasicOps.DSpineBridge]), DSpineBridgeName);
            Need(TryDSpineVector(DSpineGateBiasName, out w[GgmlBasicOps.DSpineGateBias]), DSpineGateBiasName);
            for (int l = 0; l < n; l++)
            {
                string[] wn = _dflashLayerNames[l];
                int s = GgmlBasicOps.DSpineGlobals + l * GgmlBasicOps.DSpinePerLayer;
                Need(TryDSpineVector(wn[DfAttnNorm], out w[s + GgmlBasicOps.DSpineAttnNorm]), wn[DfAttnNorm]);
                Need(TryDSpineLinear(wn[DfAttnQ], out w[s + GgmlBasicOps.DSpineQ]), wn[DfAttnQ]);
                Need(TryDSpineLinear(wn[DfAttnK], out w[s + GgmlBasicOps.DSpineK]), wn[DfAttnK]);
                Need(TryDSpineLinear(wn[DfAttnV], out w[s + GgmlBasicOps.DSpineV]), wn[DfAttnV]);
                Need(TryDSpineVector(wn[DfAttnQNorm], out w[s + GgmlBasicOps.DSpineQNorm]), wn[DfAttnQNorm]);
                Need(TryDSpineVector(wn[DfAttnKNorm], out w[s + GgmlBasicOps.DSpineKNorm]), wn[DfAttnKNorm]);
                Need(TryDSpineLinear(wn[DfAttnOutput], out w[s + GgmlBasicOps.DSpineO]), wn[DfAttnOutput]);
                Need(TryDSpineVector(wn[DfFfnNorm], out w[s + GgmlBasicOps.DSpineFfnNorm]), wn[DfFfnNorm]);
                Need(TryDSpineLinear(wn[DfFfnGate], out w[s + GgmlBasicOps.DSpineGate]), wn[DfFfnGate]);
                Need(TryDSpineLinear(wn[DfFfnUp], out w[s + GgmlBasicOps.DSpineUp]), wn[DfFfnUp]);
                Need(TryDSpineLinear(wn[DfFfnDown], out w[s + GgmlBasicOps.DSpineDown]), wn[DfFfnDown]);
                Need(TryDSpineLinear(DSpineReadName(l), out w[s + GgmlBasicOps.DSpineRead]), DSpineReadName(l));
                Need(TryDSpineLinear(DSpineWriteName(l), out w[s + GgmlBasicOps.DSpineWrite]), DSpineWriteName(l));
            }

            var p = new GgmlDSpineParams
            {
                NumLayers = n, Hidden = cfg.HiddenSize, NumHeads = cfg.NumHeads, NumKvHeads = cfg.NumKVHeads,
                HeadDim = cfg.HeadDim, Vocab = Config.VocabSize, CodeSize = cfg.DSpineCodeSize, Rank = cfg.DSpineRank,
                TopK = cfg.DSpineTopK, RingRows = cfg.RingRows, RingDtype = GgmlTypeF32, SlidingWindow = cfg.SlidingWindow,
                Eps = cfg.Eps, BridgeEps = cfg.DSpineEps, RopeBase = cfg.RopeBase, RopeFreqScale = 1f,
                KqScale = 1f / MathF.Sqrt(cfg.HeadDim),
            };

            long t0 = System.Diagnostics.Stopwatch.GetTimestamp();
            IntPtr tables = GgmlBasicOps.DSpineTablesCreate(p, w, DSpineTableType);
            if (tables == IntPtr.Zero)
                throw new InvalidOperationException("DSpine drafter: building the refinement tables failed: " + GgmlBasicOps.LastNativeError());
            double ms = System.Diagnostics.Stopwatch.GetElapsedTime(t0).TotalMilliseconds;

            // The draft graph never reads the codebook (the anchor row's halves are a TSV row).
            if (_dspineRaw.Remove(DSpineCodebookName, out var codebook))
            {
                long freed = codebook.RawBytes;
                codebook.Dispose();
                w[GgmlBasicOps.DSpineCodebook] = default;
                Console.WriteLine($"  DSpine refinement tables built in {ms:F0} ms "
                    + $"({2L * Config.VocabSize * 3 * cfg.DSpineRank / (1024 * 1024)} MB BF16); codebook host copy released ({freed / (1024 * 1024)} MB).");
            }

            int gammaMax = cfg.BlockSize - 1;
            int k = cfg.DSpineTopK;
            _dspineState = new DSpineState
            {
                Weights = w,
                Params = p,
                Tables = tables,
                Scores = new float[k + (long)k * k * Math.Max(0, gammaMax - 1)],
                Cand = new int[k * gammaMax],
                Ids = new int[cfg.BlockSize],
                Positions = new int[cfg.BlockSize],
                RingK = new IntPtr[n],
                RingV = new IntPtr[n],
            };
        }

        /// <summary>
        /// One DSpine draft of a b-row block at <paramref name="position"/>. Returns the
        /// drafts written (b - 1), or 0 - with one warning per model - when the native graph
        /// fails, so the step verifies nothing and decodes plainly.
        /// </summary>
        private int TryFusedDSpineDraftBlock(int anchorToken, int position, int b, int[] draftOut, float[] confOut)
        {
            var s = _dspineState;
            var cfg = _dflash;
            if (s == null || b < 2)
                return 0;
            int gamma = b - 1;
            for (int i = 0; i < b; i++)
            {
                s.Ids[i] = i == 0 ? anchorToken : cfg.MaskTokenId;
                s.Positions[i] = position + i;
            }
            for (int l = 0; l < cfg.NumLayers; l++)
            {
                s.RingK[l] = TensorComputePrimitives.GetStoragePointer(_dflashRingK[l]);
                s.RingV[l] = TensorComputePrimitives.GetStoragePointer(_dflashRingV[l]);
            }
            var p = s.Params;
            p.RingRows = _dflashRingRows;
            if (!GgmlBasicOps.DSpineDraftBlock(p, s.Weights, s.Tables, s.RingK, s.RingV, s.Ids, b, s.Positions,
                    BuildRingSlotPositions(), s.Scores, s.Cand))
            {
                if (!s.FailureLogged)
                {
                    s.FailureLogged = true;
                    Console.Error.WriteLine("[dspine] the draft graph failed; this model stops drafting and every step "
                        + "decodes plainly (the output is unchanged): " + GgmlBasicOps.LastNativeError());
                }
                return 0;
            }
            int produced = DSpineWalkLattice(gamma, cfg.DSpineTopK, s.Scores, s.Cand, draftOut, confOut);
            DumpDSpineDraft(b, gamma, draftOut, confOut);
            return produced;
        }

        // ---- TS_DSPINE_DUMP=<dir>: the first drafts' inputs and outputs, for eng/dspine-reference.py real ----

        private static readonly string s_dspineDumpDir = Environment.GetEnvironmentVariable("TS_DSPINE_DUMP");
        private const int DSpineDumpDrafts = 4;
        private Dictionary<int, float[]> _dspineDumpRows;
        private int _dspineDumpCount;

        /// <summary>Keeps every injected feature row by position while TS_DSPINE_DUMP still has drafts to write.</summary>
        private void RecordDSpineDumpRows(float[] hRows, int rowOffset, int n, int startPos)
        {
            if (s_dspineDumpDir == null || _dflash == null || !_dflash.IsDSpine || _dspineDumpCount >= DSpineDumpDrafts || hRows == null)
                return;
            _dspineDumpRows ??= new Dictionary<int, float[]>();
            int feat = _dflash.FeatureSize;
            for (int i = 0; i < n; i++)
            {
                if (startPos + i < 0)
                    continue;
                var row = new float[feat];
                Array.Copy(hRows, (long)(rowOffset + i) * feat, row, 0, feat);
                _dspineDumpRows[startPos + i] = row;
            }
        }

        /// <summary>Writes draft k (DSPF format, as tests/data/dspine_draft_reference.bin): the context rows the ring holds
        /// (positions before the anchor, inside the window), the block, and the lattice and chain drafted from them.</summary>
        private void DumpDSpineDraft(int b, int gamma, int[] drafts, float[] conf)
        {
            if (s_dspineDumpDir == null || _dspineDumpRows == null || _dspineDumpCount >= DSpineDumpDrafts)
                return;
            var s = _dspineState;
            int p0 = s.Positions[0];
            var pos = new List<int>();
            foreach (int p in _dspineDumpRows.Keys)
                if (p < p0 && p >= Math.Max(0, p0 - _dflash.SlidingWindow) && p >= _dflashRingValidFrom)
                    pos.Add(p);
            pos.Sort();
            int feat = _dflash.FeatureSize, k = _dflash.DSpineTopK;
            System.IO.Directory.CreateDirectory(s_dspineDumpDir);
            string path = System.IO.Path.Combine(s_dspineDumpDir, $"draft{_dspineDumpCount++}.bin");
            using var w = new System.IO.BinaryWriter(System.IO.File.Create(path));
            var entries = new List<(string Name, bool Ints, int[] Dims, Array Data)>
            {
                ("ctx.pos", true, new[] { pos.Count }, pos.ToArray()),
                ("ctx.feat", false, new[] { pos.Count, feat }, pos.SelectMany(p => _dspineDumpRows[p]).ToArray()),
                ("ids", true, new[] { b }, s.Ids.Take(b).ToArray()),
                ("p0", true, new[] { 1 }, new[] { p0 }),
                ("cand", true, new[] { gamma, k }, s.Cand.Take(gamma * k).ToArray()),
                ("scores", false, new[] { k + k * k * Math.Max(0, gamma - 1) }, s.Scores.Take(k + k * k * Math.Max(0, gamma - 1)).ToArray()),
                ("chain", true, new[] { gamma }, drafts.Take(gamma).ToArray()),
            };
            w.Write(System.Text.Encoding.ASCII.GetBytes("DSPF"));
            w.Write(1u);
            w.Write((uint)entries.Count);
            foreach (var (name, ints, dims, data) in entries)
            {
                byte[] nb = System.Text.Encoding.UTF8.GetBytes(name);
                w.Write((ushort)nb.Length);
                w.Write(nb);
                w.Write((byte)(ints ? 1 : 0));
                w.Write((byte)dims.Length);
                foreach (int d in dims) w.Write((uint)d);
                if (ints) foreach (int v in (int[])data) w.Write(v);
                else foreach (float v in (float[])data) w.Write(v);
            }
            Console.Error.WriteLine($"[dspine] dumped draft at position {p0} ({pos.Count} context rows) to {path}");
        }

        /// <summary>
        /// <see cref="DFlashWalkLattice"/> with the reference's tie rule: among equal scores
        /// the lower token id wins (the reference keeps its candidates sorted by id and takes
        /// the first maximum; the native top-k returns them in no particular order).
        /// </summary>
        internal static int DSpineWalkLattice(int gamma, int k, float[] scores, int[] cand, int[] draftOut, float[] confOut)
        {
            float[] row = new float[k];
            int chosen = 0;
            for (int e = 0; e < gamma; e++)
            {
                long baseIdx = e == 0 ? 0 : (long)k + ((long)(e - 1) * k + chosen) * k;
                Array.Copy(scores, baseIdx, row, 0, k);
                int best = 0;
                for (int c = 1; c < k; c++)
                {
                    if (row[c] > row[best] || (row[c] == row[best] && cand[e * k + c] < cand[e * k + best]))
                        best = c;
                }
                chosen = best;
                draftOut[e] = cand[e * k + chosen];
                if (confOut != null && e < confOut.Length)
                    confOut[e] = DFlashSoftmaxAt(row, chosen);
            }
            return gamma;
        }

        private void DisposeDSpine()
        {
            if (_dspineState != null)
            {
                GgmlBasicOps.DSpineResetCaches();
                GgmlBasicOps.DSpineTablesFree(_dspineState.Tables);
                _dspineState = null;
            }
            if (_dspineRaw != null)
            {
                foreach (var qw in _dspineRaw.Values)
                    qw?.Dispose();
                _dspineRaw = null;
            }
        }
    }
}
