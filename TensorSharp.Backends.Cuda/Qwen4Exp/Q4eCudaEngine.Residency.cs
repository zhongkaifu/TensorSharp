// Copyright (c) Zhongkai Fu. All rights reserved.
// https://github.com/zhongkaifu/TensorSharp
//
// This file is part of TensorSharp.
//
// TensorSharp is licensed under the BSD-3-Clause license found in the LICENSE file in the root directory of this source tree.
//
// Expert residency: Qwen3.8-Flash-Next on one GPU whose memory cannot hold every routed expert
// (UD-Q2_K_XL: 42.9 GiB of experts beside a 32 GB card).
//
// The hottest experts, ranked by the routing profile across all layers, get VRAM slots; every other
// expert is copied once into pinned host memory the GPU can address (only those: the host copy of a
// model whose experts are 42.9 GiB is then about 24 GiB beside a 32 GB card). Each stacked expert
// weight carries a device table of one address per expert, and the expert kernels read the expert's
// rows wherever its entry points: a slot at VRAM speed, or a staging copy of its host rows.
//
// Host experts (Strata's doorbell, CudaHostExperts.cs; on unless TS_Q4E_HOST_EXPERTS=0): a decode step
// or verify window of up to 8 rows leaves the missed experts to the CPU, which computes them from the
// pinned copies while the GPU computes the rest; their table entries rest at a zeroed expert in VRAM.
// Longer ubatches (prompts) still stage. The host sums with ggml's CPU kernels, so a step with misses
// differs from the all-resident engine in the last bits, as an offloaded ggml layer does.
// No step ever returns to the host to find or compute a missed expert, so a decode step stays one
// captured graph (Strata runs a token the same way). The kernels' arithmetic does not depend on
// where the bytes live, so the output is the all-resident engine's, bit for bit.
//
// A model that fits keeps the old layout (no tables). A layer split keeps it too: residency is
// planned for one device.
using System;
using System.Collections.Generic;
using System.Threading.Tasks;
using TensorSharp.Cuda.Interop;

namespace TensorSharp.Cuda
{
    public sealed unsafe partial class Q4eCudaEngine
    {
        private sealed class ExpertResidency
        {
            /// <summary>Per layer: whether each expert has a VRAM slot.</summary>
            public bool[][] Resident;
            public long SlotBytes, HostBytes;
            public int SlotExperts, TotalExperts;
            public double CoveredRoutes = double.NaN;
            /// <summary>Pinned host copies of the experts without a slot [layer][gate, up, down], packed in
            /// expert order (zero when a layer has none), freed with the engine.</summary>
            public IntPtr[][] Host;
            /// <summary>Per layer: each expert's index in its packed host copy, or -1 when it has a slot.</summary>
            public int[][] HostIndex;
            /// <summary>Decode steps leave the missed experts to the host (see the file comment).</summary>
            public bool HostCompute;
            /// <summary>Adaptive residency (on unless TS_MOE_TIER_ADAPT=0, host compute only): host copies of each layer's tables,
            /// so a swap can rewrite them. [layer][matrix][expert].</summary>
            public ulong[][][] Tab, Src;
            public IntPtr[][] TabDev, RestDev, SrcDev;
            public byte[][] IsHostFlags;
            public IntPtr[] IsHostDev;
        }

        private ExpertResidency _residency;
        private CudaExpertStage _expertStage;
        private IntPtr _zeroExpert;
        private CudaHostExpertMailbox _hostBox;
        private CudaHostExpertServer _hostServer;

        /// <summary>Rows up to which the host can compute missed experts (the host kernel's bound, TSGgml_HostExpertRows).</summary>
        internal const int HostExpertMaxRows = 16;

        /// <summary>The default of TS_Q4E_HOST_EXPERT_ROWS: decode steps, verify windows and batched steps of up to 8 rows.</summary>
        internal const int HostExpertDefaultRows = 8;

        /// <summary>Experts with a VRAM slot under residency, or -1 when every expert is resident (tests).</summary>
        internal int ResidentExperts => _residency?.SlotExperts ?? -1;

        /// <summary>Whether decode steps leave missed experts to the host (expert residency with host experts).</summary>
        public bool HostExpertsActive => _hostServer != null;

        /// <summary>Experts the host has computed for decode steps so far, or -1 without host experts (tests).</summary>
        internal long HostExpertMisses => _hostServer?.Missed ?? -1;

        private static long ExpertStackBytes(in CudaWeightDesc w) => CudaWeightArena.Align(w.TotalBytes);

        private static long LayerExpertBytes(LayerDesc l)
            => ExpertStackBytes(l.GateExps) + ExpertStackBytes(l.UpExps) + ExpertStackBytes(l.DownExps);

        /// <summary>One expert's bytes in a layer (its gate, up and down rows).</summary>
        private static long ExpertBytes(LayerDesc l, int nExpert)
            => (l.GateExps.TotalBytes + l.UpExps.TotalBytes + l.DownExps.TotalBytes) / Math.Max(1, nExpert);

        /// <summary>
        /// Decides residency before placement, or returns null for the old all-resident layout: more
        /// than one device, or every expert fits beside everything else. <paramref name="nonExpertBytes"/>
        /// is what the layers need besides their routed experts (weights, F32 tensors, caches).
        /// </summary>
        /// <summary>What device 0 has for weights and caches after its scratch, a gigabyte of headroom, the embedding,
        /// the output head and the prompt groups.</summary>
        private long ExpertPlanBudget()
        {
            var m = _m;
            _devs[0].MakeCurrent();
            var (free, _) = _devs[0].Alloc.GetMemoryInfo();
            return (long)free - DeviceScratchBytes() - (1L << 30) - m.TokEmbd.TotalBytes
                - m.Output.TotalBytes - (long)m.NVocab * 8 - PromptGroupBytes();
        }

        /// <summary>Whether every routed expert fits in VRAM beside <paramref name="nonExpertBytes"/> (no expert residency
        /// planned; a layer split never plans one, a forced TS_Q4E_EXPERT_VRAM_GB always does).</summary>
        private bool ExpertsFitWhole(int useDevs, long nonExpertBytes)
        {
            var m = _m;
            if (useDevs != 1)
                return true;
            if (m.ExpertVramBytes >= 0)
                return false;
            long expertTotal = 0;
            for (int il = 0; il < m.NLayer; il++)
                expertTotal += LayerExpertBytes(m.Layers[il]);
            return nonExpertBytes + expertTotal <= ExpertPlanBudget();
        }

        private ExpertResidency PlanResidency(int useDevs, long nonExpertBytes)
        {
            var m = _m;
            long expertTotal = 0;
            for (int il = 0; il < m.NLayer; il++)
                expertTotal += LayerExpertBytes(m.Layers[il]);
            bool forced = m.ExpertVramBytes >= 0;
            if (useDevs != 1)
            {
                if (forced)
                    Console.Error.WriteLine("[q4e-cuda] expert residency is planned for one GPU; the layer split keeps every expert in VRAM.");
                return null;
            }

            long budget = ExpertPlanBudget();
            if (!forced && nonExpertBytes + expertTotal <= budget)
                return null;
            // The tables and the arena's per-slot alignment, besides the slots themselves, and the staging
            // the missed experts of one call are copied into (StagingExperts of the largest expert).
            long tableBytes = (long)m.NLayer * 9 * CudaWeightArena.Align((long)m.NExpert * 8)
                + (long)m.NLayer * CudaWeightArena.Align(m.NExpert) + StagingBytes() + CudaWeightArena.Align(StagingStride())
                + VerifySaveBytes();   // a speculative verify's saves, allocated on its first window
            long room = budget - nonExpertBytes - tableBytes - (256L << 20);
            if (room < 0 && !forced)
                throw new InvalidOperationException(
                    $"[q4e-cuda] the model's weights and caches besides its experts need {nonExpertBytes / 1073741824.0:F1} GiB " +
                    $"but the GPU has {budget / 1073741824.0:F1} GiB free; lower the context.");
            long slotBudget = forced ? Math.Min(m.ExpertVramBytes, Math.Max(0, room)) : Math.Max(0, room);
            if (!_devs[0].DK.HasExpertTables)
                throw new NotSupportedException(
                    "[q4e-cuda] the routed experts do not fit in VRAM, and the loaded tensorsharp_dsv4_kernels module predates " +
                    "expert residency (a committed PTX baseline not yet regenerated). Rebuild TensorSharp.Backends.Cuda with a " +
                    "CUDA 13 nvcc on PATH, or run this model with --backend ggml_cuda.");

            // The hottest (layer, expert) pairs by routed count per byte, across all layers. Without a
            // profile every expert of a layer counts alike and each layer gets its share.
            var r = new ExpertResidency { Resident = new bool[m.NLayer][], TotalExperts = m.NLayer * m.NExpert };
            var pairs = new List<(double Score, int Layer, int Expert, long Bytes)>(m.NLayer * m.NExpert);
            double routes = 0;
            for (int il = 0; il < m.NLayer; il++)
            {
                r.Resident[il] = new bool[m.NExpert];
                long bytes = ExpertBytes(m.Layers[il], m.NExpert);
                ulong[] counts = m.ExpertRouteCounts != null && il < m.ExpertRouteCounts.Length ? m.ExpertRouteCounts[il] : null;
                for (int e = 0; e < m.NExpert; e++)
                {
                    double c = counts != null && e < counts.Length ? counts[e] : 1.0 + (m.NExpert - e) * 1e-9;
                    routes += counts != null && e < counts.Length ? counts[e] : 0;
                    pairs.Add((c / Math.Max(1, bytes), il, e, bytes));
                }
            }
            pairs.Sort((a, b) => b.Score.CompareTo(a.Score));
            double covered = 0;
            foreach (var p in pairs)
            {
                if (r.SlotBytes + p.Bytes > slotBudget)
                    continue;
                r.Resident[p.Layer][p.Expert] = true;
                r.SlotBytes += p.Bytes;
                r.SlotExperts++;
                ulong[] counts = m.ExpertRouteCounts?[p.Layer];
                if (counts != null && p.Expert < counts.Length)
                    covered += counts[p.Expert];
            }
            if (routes > 0)
                r.CoveredRoutes = covered / routes;
            r.HostCompute = m.HostExperts != null && Environment.GetEnvironmentVariable("TS_Q4E_HOST_EXPERTS") != "0";
            if (r.HostCompute && !_devs[0].DK.HasHostExperts)
            {
                Console.Error.WriteLine("[q4e-cuda] host experts are off: the loaded kernel module predates their doorbell kernels; missed experts are staged over the bus.");
                r.HostCompute = false;
            }
            if (r.HostCompute && AdaptiveResidencyEnabled)
            {
                r.Tab = new ulong[m.NLayer][][];
                r.Src = new ulong[m.NLayer][][];
                r.TabDev = new IntPtr[m.NLayer][];
                r.RestDev = new IntPtr[m.NLayer][];
                r.SrcDev = new IntPtr[m.NLayer][];
                r.IsHostFlags = new byte[m.NLayer][];
                r.IsHostDev = new IntPtr[m.NLayer];
            }
            if (r.HostCompute)
                foreach (var l in m.Layers)
                    if (((l.GateExps.TotalBytes / m.NExpert) | (l.UpExps.TotalBytes / m.NExpert) | (l.DownExps.TotalBytes / m.NExpert)) % 16 != 0)
                    {
                        // A prompt's missed experts must stage when the host takes a decode step's.
                        Console.Error.WriteLine("[q4e-cuda] host experts are off: an expert's size is not a whole number of 16-byte units, so prompts could not stage it.");
                        r.HostCompute = false;
                        break;
                    }
            for (int il = 0; il < m.NLayer; il++)
            {
                long bytes = ExpertBytes(m.Layers[il], m.NExpert);
                foreach (bool x in r.Resident[il])
                    if (!x)
                        r.HostBytes += bytes;
            }
            return r;
        }

        /// <summary>The arena bytes residency adds: the slots, the tables and per-slot alignment.</summary>
        private long ResidencyArenaBytes()
        {
            var r = _residency;
            long b = 0;
            for (int il = 0; il < _m.NLayer; il++)
            {
                var l = _m.Layers[il];
                int slots = 0;
                foreach (bool x in r.Resident[il]) if (x) slots++;
                long ge = l.GateExps.TotalBytes / _m.NExpert, ue = l.UpExps.TotalBytes / _m.NExpert, de = l.DownExps.TotalBytes / _m.NExpert;
                b += slots * (CudaWeightArena.Align(ge) + CudaWeightArena.Align(ue) + CudaWeightArena.Align(de));
                // The expert tables, the tables staging copies from and restores to, and the host flags.
                b += 9 * CudaWeightArena.Align((long)_m.NExpert * 8) + CudaWeightArena.Align(_m.NExpert);
            }
            // StageOf, StageList and the zero expert's pad, then StageCount and StageFrom (an aligned int each).
            // The GPU's share of the missed experts (TS_Q4E_HYBRID_MISSES): its own StageOf, StageList and two ints.
            if (_hybridMissesEnv > 0)
                b += CudaWeightArena.Align((long)_m.NExpert * 4) + CudaWeightArena.Align((long)StagingExperts() * 4) + 512;
            return b + StagingBytes() + CudaWeightArena.Align(StagingStride()) + 3 * CudaWeightArena.Align((long)_m.NExpert * 4) + 512;
        }

        /// <summary>Experts one call can stage: every slot's expert at most, and never more than a layer has.</summary>
        private int StagingExperts() => Math.Max(1, Math.Min(_m.NUbatch * _m.NExpertUsed, _m.NExpert));

        /// <summary>One staging slot: an expert's gate, up and down rows, each 256-byte aligned, the largest layer's.</summary>
        private long StagingStride()
        {
            long stride = 0;
            foreach (var l in _m.Layers)
            {
                long ge = l.GateExps.TotalBytes / _m.NExpert, ue = l.UpExps.TotalBytes / _m.NExpert, de = l.DownExps.TotalBytes / _m.NExpert;
                stride = Math.Max(stride, CudaWeightArena.Align(ge) + CudaWeightArena.Align(ue) + CudaWeightArena.Align(de));
            }
            return stride;
        }

        private long StagingBytes() => CudaWeightArena.Align(StagingExperts() * StagingStride());

        /// <summary>The device's staging area and plan buffers, in the weight arena.</summary>
        private void CreateExpertStage(Dev dev)
        {
            dev.MakeCurrent();
            int n = _m.NExpert;
            var minusOne = new int[n];
            Array.Fill(minusOne, -1);
            var stage = new CudaExpertStage { Capacity = StagingExperts(), Stride = StagingStride() };
            fixed (int* p = minusOne)
                stage.StageOf = dev.Weights.PlaceBytes((IntPtr)p, (long)n * 4);
            int zero = 0;
            stage.StageCount = dev.Weights.PlaceBytes((IntPtr)(&zero), 4);
            stage.StageFrom = dev.Weights.PlaceBytes((IntPtr)(&zero), 4);
            stage.StageList = dev.Weights.Reserve((long)stage.Capacity * 4);
            stage.Staging = dev.Weights.Reserve(StagingBytes());
            _expertStage = stage;
            if (_residency.HostCompute)
            {
                // The zero expert: all-zero blocks decode to zero weights, so a missed expert's GPU product is 0.
                _zeroExpert = dev.Weights.Reserve(StagingStride());
                CudaDriverApi.cuMemsetD8(_zeroExpert, 0, new UIntPtr((ulong)StagingStride())).ThrowOnError();
                int g = Math.Min(_hybridMissesEnv, stage.Capacity / 2);
                if (g > 0 && dev.DK.HasHybridMisses)
                {
                    // The GPU's share of a decode step's misses: the staging's last slots. An overlapped adaptive round
                    // borrows the slots before them (AsyncSwapCapacity); a prompt, which stages through all of it, runs
                    // between steps.
                    var hybrid = new CudaExpertStage { Capacity = g, Stride = stage.Stride };
                    fixed (int* p = minusOne)
                        hybrid.StageOf = dev.Weights.PlaceBytes((IntPtr)p, (long)n * 4);
                    hybrid.StageCount = dev.Weights.PlaceBytes((IntPtr)(&zero), 4);
                    hybrid.StageFrom = dev.Weights.PlaceBytes((IntPtr)(&zero), 4);
                    hybrid.StageList = dev.Weights.Reserve((long)g * 4);
                    hybrid.Staging = stage.Staging + (nint)((long)(stage.Capacity - g) * stage.Stride);
                    _hybridStage = hybrid;
                    Console.Error.WriteLine($"[q4e-cuda] missed experts split: the GPU stages up to {g} of a decode step's "
                        + $"missed experts over the bus ({(_hybridOdd ? "odd-numbered ones" : "the first in slot order")}) while the "
                        + "host computes the rest (TS_Q4E_HYBRID_MISSES, TS_Q4E_HYBRID_SPLIT).");
                }
                else if (g > 0)
                    Console.Error.WriteLine("[q4e-cuda] missed experts split off: the loaded tensorsharp_dsv4_kernels module predates "
                        + "ts_moe_doorbell_publish_hybrid; the host computes every missed expert.");
            }
        }

        // TS_Q4E_HYBRID_MISSES: distinct missed experts of a decode step or verify window the GPU stages over the bus and
        // computes itself while the host computes the rest (0 = the host takes them all). Default 4: on the RTX 5090
        // pod a 4-row window's verify 11.4-13.0 -> 10.7-12.1 ms with the first misses in slot order.
        private readonly int _hybridMissesEnv = Math.Max(0, EnvInt("TS_Q4E_HYBRID_MISSES", 4));

        // TS_Q4E_HYBRID_SPLIT=odd: only odd-numbered experts are the GPU's, so an expert goes to the same side whatever else
        // a call selects (below capacity) and a verify row computes what its own decode step does; default "first": the
        // first missed experts in slot order (more of them; which side a row's expert lands on depends on the window).
        private readonly bool _hybridOdd = Environment.GetEnvironmentVariable("TS_Q4E_HYBRID_SPLIT") == "odd";

        /// <summary>The staging the GPU's share of the missed experts goes through, or null (see TS_Q4E_HYBRID_MISSES).</summary>
        private CudaExpertStage _hybridStage;

        /// <summary>
        /// Copies the experts without a slot into pinned host memory the GPU can address, packed per
        /// stack in expert order, in parallel (the reads are the slow part on network storage). Runs
        /// once, before the layers are uploaded.
        /// </summary>
        private void LoadHostExperts()
        {
            var m = _m;
            var r = _residency;
            r.Host = new IntPtr[m.NLayer][];
            r.HostIndex = new int[m.NLayer][];
            var jobs = new List<(IntPtr Dst, CudaWeightDesc Src, long Offset, long Bytes)>();
            _devs[0].MakeCurrent();
            for (int il = 0; il < m.NLayer; il++)
            {
                var l = m.Layers[il];
                var index = r.HostIndex[il] = new int[m.NExpert];
                int n = 0;
                for (int e = 0; e < m.NExpert; e++)
                    index[e] = r.Resident[il][e] ? -1 : n++;
                r.Host[il] = new IntPtr[3];
                if (n == 0)
                    continue;
                var stacks = new[] { l.GateExps, l.UpExps, l.DownExps };
                for (int k = 0; k < 3; k++)
                {
                    long expertBytes = stacks[k].TotalBytes / m.NExpert;
                    CudaDriverApi.cuMemHostAlloc(out IntPtr host, new UIntPtr((ulong)(n * expertBytes)),
                        CudaDriverApi.CU_MEMHOSTALLOC_PORTABLE | CudaDriverApi.CU_MEMHOSTALLOC_DEVICEMAP).ThrowOnError();
                    r.Host[il][k] = host;
                    AddRunReads(jobs, stacks[k], expertBytes, e => index[e] >= 0, e => host + (nint)(index[e] * expertBytes));
                }
            }
            RunReads(jobs);
        }

        /// <summary>One read per run of consecutive selected experts (at most 64 MiB), into consecutive destination bytes.</summary>
        private void AddRunReads(List<(IntPtr Dst, CudaWeightDesc Src, long Offset, long Bytes)> jobs, in CudaWeightDesc src,
            long expertBytes, Func<int, bool> selected, Func<int, IntPtr> dst)
        {
            int maxRun = (int)Math.Max(1, (64L << 20) / Math.Max(1, expertBytes));
            for (int e = 0; e < _m.NExpert;)
            {
                if (!selected(e)) { e++; continue; }
                int end = e + 1;
                while (end < _m.NExpert && end - e < maxRun && selected(end))
                    end++;
                jobs.Add((dst(e), src, e * expertBytes, (end - e) * expertBytes));
                e = end;
            }
        }

        private static void RunReads(List<(IntPtr Dst, CudaWeightDesc Src, long Offset, long Bytes)> jobs)
        {
            var opts = new ParallelOptions { MaxDegreeOfParallelism = Math.Max(1, EnvInt("TS_Q4E_EXPERT_READERS", 16)) };
            Parallel.ForEach(jobs, opts, j =>
            {
                if (j.Src.HostPtr != IntPtr.Zero)
                    Buffer.MemoryCopy((void*)(j.Src.HostPtr + (nint)j.Offset), (void*)j.Dst, j.Bytes, j.Bytes);
                else
                    j.Src.Source.Read(j.Src.SourceOffset + j.Offset, j.Dst, j.Bytes);
            });
        }

        /// <summary>
        /// A layer's three stacked expert weights under residency: slots for its resident experts in
        /// the device arena (read from the model file now), and per-expert tables pointing each expert at
        /// its slot or its host copy.
        /// </summary>
        private void PlaceResidentExperts(Dev dev, int il, DevLayer dst)
        {
            var l = _m.Layers[il];
            var r = _residency;
            int[] index = r.HostIndex[il];
            var stacks = new[] { l.GateExps, l.UpExps, l.DownExps };
            var placed = new DeviceWeight[3];
            var srcTabs = new IntPtr[3];
            var restTabs = new IntPtr[3];
            var table = new ulong[_m.NExpert];
            var hostTable = new ulong[_m.NExpert];
            bool hostCompute = r.HostCompute;
            int resident = 0;
            for (int e = 0; e < _m.NExpert; e++)
                if (index[e] < 0)
                    resident++;
            for (int k = 0; k < 3; k++)
            {
                var w = stacks[k];
                long expertBytes = w.TotalBytes / _m.NExpert;
                IntPtr hostDev = IntPtr.Zero;
                if (r.Host[il][k] != IntPtr.Zero)
                    CudaDriverApi.cuMemHostGetDevicePointer(out hostDev, r.Host[il][k], 0).ThrowOnError();

                // The resident experts' bytes: straight from the mapping, or read now (packed) for this stack.
                IntPtr staged = IntPtr.Zero;
                var at = new int[_m.NExpert];
                if (w.HostPtr == IntPtr.Zero && resident > 0)
                {
                    staged = (IntPtr)System.Runtime.InteropServices.NativeMemory.Alloc((nuint)(resident * expertBytes));
                    int n = 0;
                    for (int e = 0; e < _m.NExpert; e++)
                        at[e] = index[e] < 0 ? n++ : -1;
                    var jobs = new List<(IntPtr Dst, CudaWeightDesc Src, long Offset, long Bytes)>();
                    AddRunReads(jobs, w, expertBytes, e => index[e] < 0, e => staged + (nint)(at[e] * expertBytes));
                    RunReads(jobs);
                }
                try
                {
                    for (int e = 0; e < _m.NExpert; e++)
                    {
                        if (index[e] < 0)
                        {
                            IntPtr src = staged != IntPtr.Zero ? staged + (nint)(at[e] * expertBytes) : w.HostPtr + (nint)(e * expertBytes);
                            table[e] = (ulong)dev.Weights.PlaceBytes(src, expertBytes);
                            hostTable[e] = 0;
                        }
                        else
                        {
                            hostTable[e] = (ulong)(hostDev + (nint)(index[e] * expertBytes));
                            table[e] = hostCompute ? (ulong)_zeroExpert : hostTable[e];
                        }
                    }
                }
                finally
                {
                    if (staged != IntPtr.Zero)
                        System.Runtime.InteropServices.NativeMemory.Free((void*)staged);
                }
                IntPtr tableDev;
                fixed (ulong* t = table)
                    tableDev = dev.Weights.PlaceBytes((IntPtr)t, (long)_m.NExpert * 8);
                if (r.Tab != null)
                {
                    r.Tab[il] ??= new ulong[3][];
                    r.Src[il] ??= new ulong[3][];
                    r.TabDev[il] ??= new IntPtr[3];
                    r.Tab[il][k] = (ulong[])table.Clone();
                    r.Src[il][k] = (ulong[])hostTable.Clone();
                    r.TabDev[il][k] = tableDev;
                }
                fixed (ulong* t = hostTable)
                    srcTabs[k] = dev.Weights.PlaceBytes((IntPtr)t, (long)_m.NExpert * 8);
                // Staging restores the resting entries: the host copy, or the zero expert under host compute.
                if (hostCompute)
                    fixed (ulong* t = table)
                        restTabs[k] = dev.Weights.PlaceBytes((IntPtr)t, (long)_m.NExpert * 8);
                else
                    restTabs[k] = srcTabs[k];
                if (r.Tab != null)
                {
                    r.RestDev[il] ??= new IntPtr[3];
                    r.SrcDev[il] ??= new IntPtr[3];
                    r.RestDev[il][k] = restTabs[k];
                    r.SrcDev[il][k] = srcTabs[k];
                }
                placed[k] = new DeviceWeight
                {
                    Ptr = (IntPtr)(long)table[0], Type = w.GgmlType, Ne0 = w.Ne0, Ne1 = w.Ne1, RowBytes = w.RowBytes, Table = tableDev,
                };
            }
            dst.GateExps = placed[0];
            dst.UpExps = placed[1];
            dst.DownExps = placed[2];

            // Staging copies 16-byte units: a layer whose expert sizes are not whole units reads its missed
            // experts over the bus in place instead.
            long bg = l.GateExps.TotalBytes / _m.NExpert, bu = l.UpExps.TotalBytes / _m.NExpert, bd = l.DownExps.TotalBytes / _m.NExpert;
            bool stages = _expertStage != null && ((bg | bu | bd) & 15) == 0;
            if (!stages)
                Console.Error.WriteLine($"[q4e-cuda] layer {il}: experts of {bg}/{bu}/{bd} bytes are not staged; misses read over the bus.");
            var isHost = new byte[_m.NExpert];
            for (int e = 0; e < _m.NExpert; e++)
                isHost[e] = index[e] < 0 ? (byte)0 : (byte)1;
            IntPtr isHostDev;
            fixed (byte* p = isHost)
                isHostDev = dev.Weights.PlaceBytes((IntPtr)p, _m.NExpert);
            if (r.Tab != null)
            {
                r.IsHostFlags[il] = isHost;
                r.IsHostDev[il] = isHostDev;
            }
            dst.Stage = new CudaExpertStageLayer
            {
                IsHost = isHostDev,
                TabGate = placed[0].Table, TabUp = placed[1].Table, TabDown = placed[2].Table,
                SrcTabGate = srcTabs[0], SrcTabUp = srcTabs[1], SrcTabDown = srcTabs[2],
                RestTabGate = restTabs[0], RestTabUp = restTabs[1], RestTabDown = restTabs[2],
                Staged = stages,
                BytesGate = bg, BytesUp = bu, BytesDown = bd,
                OffUp = CudaWeightArena.Align(bg), OffDown = CudaWeightArena.Align(bg) + CudaWeightArena.Align(bu),
            };
        }

        /// <summary>Starts the host experts' mailbox and server once every layer is placed.</summary>
        private void StartHostExperts()
        {
            var r = _residency;
            if (r == null || !r.HostCompute)
                return;
            var m = _m;
            var layers = new CudaHostExpertServer.Layer[m.NLayer];
            for (int il = 0; il < m.NLayer; il++)
            {
                var l = m.Layers[il];
                int packed = 0;
                foreach (int x in r.HostIndex[il]) if (x >= 0) packed++;
                if (packed == 0)
                    continue;
                layers[il] = new CudaHostExpertServer.Layer
                {
                    Gate = r.Host[il][0], Up = r.Host[il][1], Down = r.Host[il][2],
                    GateType = l.GateExps.GgmlType, UpType = l.UpExps.GgmlType, DownType = l.DownExps.GgmlType,
                    Packed = packed, HostIndex = r.HostIndex[il],
                };
            }
            _devs[0].MakeCurrent();
            _hostBox = new CudaHostExpertMailbox(HostExpertMaxRows, m.NEmbd, m.NExpertUsed);
            _hostServer = new CudaHostExpertServer(_hostBox, layers, m.NEmbd, m.NFfExp, m.NExpertUsed, m.HostExperts);
            foreach (var L in _layers)
                if (L.Stage.IsValid)
                    _hostLayers++;
            if (r.Tab != null)
            {
                _adaptCounts = new float[m.NLayer][];
                for (int il = 0; il < m.NLayer; il++)
                    _adaptCounts[il] = new float[m.NExpert];
                _adaptRaw = new uint[(long)m.NLayer * m.NExpert];
                CudaDriverApi.cuMemHostAlloc(out _adaptTemp, new UIntPtr((ulong)StagingStride()), CudaDriverApi.CU_MEMHOSTALLOC_PORTABLE).ThrowOnError();
                Console.Error.WriteLine($"[q4e-cuda] adaptive residency: every {AdaptEvery} steps or verify windows up to {AdaptSwapsPerCall} "
                    + "of the most-selected host experts trade places with their layers' least-selected VRAM experts "
                    + "(TS_MOE_TIER_ADAPT=0 turns it off; text then depends on the history).");
            }
            // A layer whose selected experts all have slots rings no doorbell (Strata's device-planned groups): the GPU
            // decides it and skips the wait, where it used to wait for the host's echo at every layer. TS_Q4E_DOORBELL_SKIP=0
            // rings every layer. Adaptation then counts the selections on the device.
            if (_devs[0].DK.HasDoorbellSkip && Environment.GetEnvironmentVariable("TS_Q4E_DOORBELL_SKIP") != "0")
            {
                _doorbellMissT = AllocT(_devs[0], DType.Int32, 1);
                _doorbellMiss = Ptr(_doorbellMissT);
                if (_adaptCounts != null)
                {
                    _adaptCountsT = AllocT(_devs[0], DType.Int32, (long)m.NLayer * m.NExpert);
                    _adaptCountsDev = Ptr(_adaptCountsT);
                    Zero(_devs[0], _adaptCountsT, 4);
                    if (PromptAdaptSwaps > 0 && _devs[0].QK.HasCountSelected)
                    {
                        _promptCountsT = AllocT(_devs[0], DType.Int32, (long)m.NLayer * m.NExpert);
                        _promptCountsDev = Ptr(_promptCountsT);
                        Zero(_devs[0], _promptCountsT, 4);
                        Console.Error.WriteLine($"[q4e-cuda] adaptive residency learns from prompts: after a prompt of more than "
                            + $"{_hostExpertRows} rows up to {PromptAdaptSwaps} of its most-selected host experts trade places "
                            + "(TS_MOE_TIER_ADAPT_PROMPT_SWAPS, 0 = off).");
                    }
                }
            }
            if (_adaptCounts != null && _adaptCountsDev == IntPtr.Zero)
                _hostServer.Counts = _adaptCounts;   // an older module: the host counts what every layer's doorbell names
            ConfigureTailSkip();
        }

        /// <summary>The route-tail skip's rank in force (0: off).</summary>
        private int _tailRank;

        /// <summary>Host slots the route-tail skip left out so far, or -1 without host experts (tests).</summary>
        internal long TailSkipped => _hostServer?.TailSkipped ?? -1;

        /// <summary>
        /// Strata's route-tail skip (STRATA_ROUTE_TAIL_SKIP=7, its default): in a decode step or verify window a host expert
        /// that every row routes at rank TS_MOE_ROUTE_TAIL_SKIP or lower (0-based in the row's descending top-k: the 8th to
        /// 10th of Flash Next's 10) is neither computed nor waited for; its routes add nothing and the weights are not
        /// renormalized, so answers become approximate. Off when every expert is resident, when at least
        /// TS_MOE_TAIL_SKIP_MAX_CACHED percent of the experts have VRAM slots, and for prompts (they stage every route).
        /// </summary>
        private void ConfigureTailSkip()
        {
            var r = _residency;
            var m = _m;
            int rank = Math.Max(0, EnvInt("TS_MOE_ROUTE_TAIL_SKIP", 7));
            int maxCached = Math.Clamp(EnvInt("TS_MOE_TAIL_SKIP_MAX_CACHED", 90), 0, 100);
            _tailRank = 0;
            if (_hostServer == null || r == null || rank <= 0 || rank >= m.NExpertUsed)
                return;
            double cached = r.TotalExperts > 0 ? (double)r.SlotExperts / r.TotalExperts : 1.0;
            if (cached * 100.0 >= maxCached)
            {
                Console.Error.WriteLine($"[q4e-cuda] route-tail skip off: {cached * 100:F1}% of the experts have VRAM slots "
                    + $"(TS_MOE_TAIL_SKIP_MAX_CACHED={maxCached}).");
                return;
            }
            if (_doorbellMiss == IntPtr.Zero || !_devs[0].DK.HasDoorbellTail)
            {
                Console.Error.WriteLine("[q4e-cuda] route-tail skip off: the loaded tensorsharp_dsv4_kernels module predates "
                    + "ts_moe_doorbell_publish_tail (or the doorbell skip is off); every missed expert is computed. Rebuild "
                    + "TensorSharp.Backends.Cuda with a CUDA 13 nvcc on PATH.");
                return;
            }
            _tailRank = rank;
            _hostServer.TailRank = rank;
            Console.Error.WriteLine($"[q4e-cuda] route-tail skip: in decode steps and verify windows a host expert every row routes "
                + $"at rank {rank} or lower (0-based, of {m.NExpertUsed}) is left out, unrenormalized (Strata's "
                + "STRATA_ROUTE_TAIL_SKIP; answers become approximate). TS_MOE_ROUTE_TAIL_SKIP=0 computes every route.");
        }

        /// <summary>A device int the doorbell kernels share per layer: whether the layer rang the host (zero: every layer
        /// rings, on a module without the skip kernels).</summary>
        private IntPtr _doorbellMiss;
        private Tensor _doorbellMissT;
        private int _hostLayers;
        private long _missedSeen;

        /// <summary>A forward of <paramref name="nt"/> rows ran the doorbell layers: their routes the host did not take were
        /// the device's (the server sees only the layers that rang it).</summary>
        private void NoteHostRoutes(int nt)
        {
            if (_hostServer == null || nt > _hostExpertRows)
                return;
            long missed = _hostServer.Missed;
            CudaMoeOffloadCounters.AddDeviceRoutes((long)nt * _m.NExpertUsed * _hostLayers - (missed - _missedSeen));
            _missedSeen = missed;
        }

        private IntPtr _adaptTemp;
        // A prompt's selections per (layer, expert), counted on the device while it runs (Moe).
        private Tensor _promptCountsT;
        private IntPtr _promptCountsDev;
        private int _adaptSteps;
        private long _adaptSwaps;
        private float[][] _adaptCounts;
        private uint[] _adaptRaw;
        private Tensor _adaptCountsT;
        private IntPtr _adaptCountsDev;

        /// <summary>Adaptive residency is on unless TS_MOE_TIER_ADAPT=0 (read per engine; a test process toggles it).</summary>
        private static bool AdaptiveResidencyEnabled => Environment.GetEnvironmentVariable("TS_MOE_TIER_ADAPT") != "0";

        /// <summary>Forwards (decode steps, verify windows, batched steps) between adaptation rounds (TS_MOE_TIER_ADAPT_EVERY).</summary>
        private readonly int AdaptEvery = Math.Max(1, EnvInt("TS_MOE_TIER_ADAPT_EVERY", 4));
        /// <summary>Swaps a round makes at most, the largest gains first across the layers (TS_MOE_TIER_ADAPT_SWAPS): 8 when
        /// they block, 64 when they overlap the decode (Q4eCudaEngine.AsyncSwaps.cs).</summary>
        private int AdaptSwapsPerCall => Math.Max(1, EnvInt("TS_MOE_TIER_ADAPT_SWAPS", AsyncSwapsUsable ? 64 : 8));

        /// <summary>Swaps a round after a prompt makes at most (TS_MOE_TIER_ADAPT_PROMPT_SWAPS; 0 = prompts teach nothing).
        /// The decode after a long prompt routes much as the prompt did: on the RTX 5090 the first verify windows after a
        /// 4K-token prompt left 170-290 experts to the CPU against 25-60 once adaptation had caught up.</summary>
        private static readonly int PromptAdaptSwaps = Math.Max(0, EnvInt("TS_MOE_TIER_ADAPT_PROMPT_SWAPS", 64));

        /// <summary>A prompt's selections weigh as this many rows' worth of decode selections in the counts.</summary>
        private const float PromptAdaptRows = 128f;

        /// <summary>Swaps performed by adaptive residency so far (tests).</summary>
        internal long AdaptiveSwaps => _adaptSwaps;

        /// <summary>
        /// Adaptive residency, between forwards (decode steps, verify windows, batched steps) every
        /// <see cref="AdaptEvery"/>: the selections since the last round (counted on the device by the doorbell
        /// kernel, or by the host server on an older module) join each layer's decayed counts; then, largest gains first
        /// across the layers and at most <see cref="AdaptSwapsPerCall"/> times, a layer's most-selected expert without a
        /// slot trades places with its least-selected expert with one, while it was selected more than twice as often:
        /// the slot's bytes go to the host expert's packed place and the host expert's bytes into the slot, so memory
        /// stays as planned. Tables, flags and host indices follow; the captured graphs read them at run time.
        /// </summary>
        private void AdaptResidency()
        {
            var r = _residency;
            if (r?.Tab == null || _adaptCounts == null)
                return;
            // An overlapped round in flight takes its next step once its copies have landed.
            AdvanceAsyncSwaps(force: false);
            if (++_adaptSteps % AdaptEvery != 0)
                return;
            AdaptRound(AdaptSwapsPerCall);
        }

        /// <summary>
        /// After a prompt of <paramref name="rows"/> rows: its selections (counted on the device as it ran) join the counts,
        /// scaled to <see cref="PromptAdaptRows"/> rows' worth, and a round of up to <see cref="PromptAdaptSwaps"/> swaps
        /// brings the prompt's most-selected host experts into VRAM before its answer decodes.
        /// </summary>
        private void AdaptAfterPrompt(int rows)
        {
            var r = _residency;
            if (r?.Tab == null || _adaptCounts == null || _promptCountsDev == IntPtr.Zero || rows <= _hostExpertRows)
                return;
            var m = _m;
            var dev = _devs[0];
            dev.MakeCurrent();
            CudaDriverApi.cuStreamSynchronize(dev.Stream).ThrowOnError();
            fixed (uint* raw = _adaptRaw)
                CudaDriverApi.cuMemcpyDtoH((IntPtr)raw, _promptCountsDev, new UIntPtr((ulong)_adaptRaw.Length * 4)).ThrowOnError();
            Zero(dev, _promptCountsT, 4);
            float scale = Math.Min(1f, PromptAdaptRows / rows);
            for (int il = 0; il < m.NLayer; il++)
            {
                float[] c = _adaptCounts[il];
                for (int e = 0; e < m.NExpert; e++)
                    c[e] += scale * _adaptRaw[(long)il * m.NExpert + e];
            }
            AdaptRound(PromptAdaptSwaps, decodeCounts: false);
        }

        /// <param name="decodeCounts">Fold in the decode selections the device counted since the last round.</param>
        private void AdaptRound(int swapBudget, bool decodeCounts = true)
        {
            var r = _residency;
            var m = _m;
            var dev = _devs[0];
            dev.MakeCurrent();
            // One overlapped round at a time: the counts keep gathering until it is done.
            if (_swapStep != 0)
                return;
            CudaDriverApi.cuStreamSynchronize(dev.Stream).ThrowOnError();
            float[][] counts = _adaptCounts;
            if (decodeCounts && _adaptCountsDev != IntPtr.Zero)
            {
                fixed (uint* raw = _adaptRaw)
                    CudaDriverApi.cuMemcpyDtoH((IntPtr)raw, _adaptCountsDev, new UIntPtr((ulong)_adaptRaw.Length * 4)).ThrowOnError();
                Zero(dev, _adaptCountsT, 4);
                for (int il = 0; il < m.NLayer; il++)
                {
                    float[] c = counts[il];
                    for (int e = 0; e < m.NExpert; e++)
                        c[e] += _adaptRaw[(long)il * m.NExpert + e];
                }
            }
            if (AsyncSwapsUsable)
            {
                BeginAsyncSwaps(SelectTrades(swapBudget));
                DecayAdaptCounts();
                return;
            }
            // Each layer's best trade, then the largest gains first.
            var trades = new List<(float Gain, int Layer)>();
            for (int il = 0; il < m.NLayer; il++)
            {
                if (r.HostIndex[il] == null)
                    continue;
                var (hot, cold) = BestTrade(il);
                if (hot >= 0)
                    trades.Add((counts[il][hot] - counts[il][cold], il));
            }
            trades.Sort((a, b) => b.Gain.CompareTo(a.Gain));
            int budget = swapBudget;
            var changed = new HashSet<int>();
            foreach (var (_, il) in trades)
            {
                // A layer may trade again while its next best still qualifies and the round has budget.
                while (budget > 0)
                {
                    var (hot, cold) = BestTrade(il);
                    if (hot < 0)
                        break;
                    SwapExperts(il, hot, cold);
                    changed.Add(il);
                    budget--;
                }
                if (budget == 0)
                    break;
            }
            foreach (int il in changed)
                UploadLayerTables(il);
            DecayAdaptCounts();
        }

        private void DecayAdaptCounts()
        {
            for (int il = 0; il < _m.NLayer; il++)
            {
                float[] c = _adaptCounts[il];
                for (int e = 0; e < _m.NExpert; e++)
                    c[e] *= 0.75f;
            }
        }

        /// <summary>Layer <paramref name="il"/>'s most-selected expert without a slot and least-selected expert with one,
        /// when the first was selected more than twice as often (and at least twice); (-1, -1) otherwise.</summary>
        private (int Hot, int Cold) BestTrade(int il)
        {
            float[] c = _adaptCounts[il];
            int[] index = _residency.HostIndex[il];
            int hot = -1, cold = -1;
            for (int e = 0; e < _m.NExpert; e++)
            {
                if (index[e] >= 0) { if (hot < 0 || c[e] > c[hot]) hot = e; }
                else if (cold < 0 || c[e] < c[cold]) cold = e;
            }
            if (hot < 0 || cold < 0 || c[hot] < 2f || c[hot] <= 2f * c[cold] + 1f)
                return (-1, -1);
            return (hot, cold);
        }

        /// <summary>Host expert <paramref name="hot"/> takes VRAM expert <paramref name="cold"/>'s slots; cold takes hot's packed host place.</summary>
        private void SwapExperts(int il, int hot, int cold)
        {
            var r = _residency;
            var l = _m.Layers[il];
            int i = r.HostIndex[il][hot];
            long[] bytes = { l.GateExps.TotalBytes / _m.NExpert, l.UpExps.TotalBytes / _m.NExpert, l.DownExps.TotalBytes / _m.NExpert };
            for (int k = 0; k < 3; k++)
            {
                long eb = bytes[k];
                IntPtr host = r.Host[il][k] + (nint)(i * eb);
                IntPtr slot = (IntPtr)(long)r.Tab[il][k][cold];
                Buffer.MemoryCopy((void*)host, (void*)_adaptTemp, eb, eb);
                CudaDriverApi.cuMemcpyDtoH(host, slot, new UIntPtr((ulong)eb)).ThrowOnError();
                CudaDriverApi.cuMemcpyHtoD(slot, _adaptTemp, new UIntPtr((ulong)eb)).ThrowOnError();
                r.Src[il][k][cold] = r.Src[il][k][hot];
                r.Src[il][k][hot] = 0;
                r.Tab[il][k][hot] = (ulong)(long)slot;
                r.Tab[il][k][cold] = (ulong)(long)_zeroExpert;
            }
            r.HostIndex[il][cold] = i;
            r.HostIndex[il][hot] = -1;
            r.IsHostFlags[il][hot] = 0;
            r.IsHostFlags[il][cold] = 1;
            _adaptSwaps++;
            CudaMoeOffloadCounters.AddSwap();
        }

        private void UploadLayerTables(int il)
        {
            var r = _residency;
            long n = (long)_m.NExpert * 8;
            for (int k = 0; k < 3; k++)
            {
                fixed (ulong* t = r.Tab[il][k])
                {
                    CudaDriverApi.cuMemcpyHtoD(r.TabDev[il][k], (IntPtr)t, new UIntPtr((ulong)n)).ThrowOnError();
                    CudaDriverApi.cuMemcpyHtoD(r.RestDev[il][k], (IntPtr)t, new UIntPtr((ulong)n)).ThrowOnError();
                }
                fixed (ulong* t = r.Src[il][k])
                    CudaDriverApi.cuMemcpyHtoD(r.SrcDev[il][k], (IntPtr)t, new UIntPtr((ulong)n)).ThrowOnError();
            }
            fixed (byte* p = r.IsHostFlags[il])
                CudaDriverApi.cuMemcpyHtoD(r.IsHostDev[il], (IntPtr)p, new UIntPtr((ulong)_m.NExpert)).ThrowOnError();
        }

        /// <summary>Whether a call of <paramref name="nt"/> rows on layer <paramref name="L"/> leaves its missed experts to the host.</summary>
        private bool HostExpertsFor(DevLayer L, int nt)
            => _hostServer != null && nt <= _hostExpertRows && L.Stage.IsValid && !_groupedPrompt;

        /// <summary>Rows up to which a step leaves its missed experts to the CPU (TS_Q4E_HOST_EXPERT_ROWS, 1 to
        /// <see cref="HostExpertMaxRows"/>, default <see cref="HostExpertDefaultRows"/>); wider steps stage them over the bus.
        /// A verify window's rows miss more experts between them than one decode row does, and the GPU waits for the CPU at
        /// every layer. A batched speculative verify of four sequences is 16 rows.</summary>
        private readonly int _hostExpertRows = Math.Clamp(EnvInt("TS_Q4E_HOST_EXPERT_ROWS", HostExpertDefaultRows), 1, HostExpertMaxRows);

        /// <summary>A forward is about to run: keep the host experts' server spinning; and after it, report its failure.</summary>
        private void ArmHostExperts() => _hostServer?.Arm();

        private void CheckHostExperts(int nt = 0)
        {
            if (nt > 0)
                NoteHostRoutes(nt);
            Exception ex = _hostServer?.Error;
            if (ex != null)
                throw new InvalidOperationException("[q4e-cuda] the host experts failed: " + ex.Message, ex);
        }

        private void StopHostExperts()
        {
            _hostServer?.Dispose();
            _hostServer = null;
            _hostBox?.Dispose();
            _hostBox = null;
            if (_adaptTemp != IntPtr.Zero)
                CudaDriverApi.cuMemFreeHost(_adaptTemp);
            _adaptTemp = IntPtr.Zero;
        }

        private void ReportResidency()
        {
            var r = _residency;
            if (r == null)
                return;
            string covered = double.IsNaN(r.CoveredRoutes) ? "no routing profile" : $"{r.CoveredRoutes * 100:F1}% of profiled routes";
            Console.Error.WriteLine(
                $"[q4e-cuda] expert residency: {r.SlotExperts} of {r.TotalExperts} experts in VRAM " +
                $"({r.SlotBytes / 1073741824.0:F1} GiB, {covered}); the other experts in pinned host memory " +
                $"({r.HostBytes / 1073741824.0:F1} GiB), " +
                (r.HostCompute ? $"computed by the CPU in decode steps of up to {_hostExpertRows} rows and staged over the bus in prompts."
                               : "staged over the bus when routed."));
            if (_groupXs != null)
                Console.Error.WriteLine(
                    $"[q4e-cuda] prompts read layer by layer in groups of up to {PromptGroupRows} rows ({PromptGroupBytes() / 1048576.0:F0} MiB "
                    + "of streams; TS_Q4E_PROMPT_GROUP, 0 = ubatch order): a missed expert crosses the bus once per layer and group.");
        }

        private void FreeHostExperts()
        {
            if (_residency?.Host == null)
                return;
            foreach (var layer in _residency.Host)
                if (layer != null)
                    foreach (IntPtr p in layer)
                        if (p != IntPtr.Zero)
                            CudaDriverApi.cuMemFreeHost(p);
            _residency.Host = null;
        }
    }
}
