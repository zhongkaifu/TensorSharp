// Copyright (c) Zhongkai Fu. All rights reserved.
// https://github.com/zhongkaifu/TensorSharp
//
// This file is part of TensorSharp.
//
// TensorSharp is licensed under the BSD-3-Clause license found in the LICENSE file in the root directory of this source
// tree.
//
// A prompt group's next layer staged ahead (the Mac reads its prompt experts the same way, ggml_ops_moe_hot_tier.cpp's
// stage prefetch). Read layer by layer, a group of 4,096 rows selects most of a layer's experts without a VRAM slot,
// and staging them on the prompt's stream left the GPU idle while they crossed the bus: 0.26 s of a 4K prompt on the
// RTX 5090. The staging is split in two halves with their own plans; while one layer's ubatches run on the half that
// holds its experts, the copy engines bring every host expert of the next layer into the other half (one 2D copy per
// matrix: the host keeps a layer's experts packed in host-index order, which becomes the staging slot order, so the
// plan is written from the host), and the next layer waits for those copies instead of staging per ubatch. A half is
// reused two layers later, after the layer that used it has unstaged. The copy kernel the per-ubatch staging uses
// was tried here first: its blocks wait on the bus while holding the SMs the layer needs (a 4K prompt 1.82 s instead
// of 1.65). The bytes a kernel reads are the same, so the prompt answers bit for bit as before. Layers whose host
// experts fit a half; TS_Q4E_PROMPT_PREFETCH=0 stages per ubatch.
using System;
using TensorSharp.Cuda.Interop;

namespace TensorSharp.Cuda
{
    public sealed unsafe partial class Q4eCudaEngine
    {
        private readonly bool _promptPrefetchEnv = Environment.GetEnvironmentVariable("TS_Q4E_PROMPT_PREFETCH") != "0";
        private CudaExpertStage[] _halfStages;
        private IntPtr _prefetchStream;
        private readonly IntPtr[] _prefetchDone = new IntPtr[2], _halfFree = new IntPtr[2];
        // The staging a grouped prompt's routed experts use (a half under prefetch), else _expertStage.
        private CudaExpertStage _promptStage;
        private bool _prefetchChecked, _prefetchUsable;
        // Per layer: its host experts fit half the staging (adaptive swaps keep a layer's count).
        private bool[] _halfFits;

        // Per half: the plan written on the host (stageOf [NExpert], stageList [half], count, then the three tables
        // [NExpert] u64), in pinned memory the copy reads.
        private readonly IntPtr[] _planPinned = new IntPtr[2];
        private long _planBytes;

        /// <summary>Prompt groups whose layers were staged ahead so far (a test proves the path engaged).</summary>
        internal long PromptPrefetchGroups { get; private set; }

        /// <summary>Whether a grouped prompt stages its next layer ahead: residency on one GPU, and some pair of consecutive
        /// layers whose host experts each fit half the staging (the others stage per ubatch over the whole of it).</summary>
        private bool PromptPrefetchUsable()
        {
            if (_prefetchChecked)
                return _prefetchUsable;
            _prefetchChecked = true;
            var r = _residency;
            // Mapped host experts are not packed in host-index order (and stage through host copies).
            if (!_promptPrefetchEnv || r == null || r.Mapped || _expertStage == null || _devs.Length != 1)
                return false;
            int half = _expertStage.Capacity / 2, most = 0, fit = 0, pairs = 0;
            _halfFits = new bool[_m.NLayer];
            for (int il = 0; il < _m.NLayer; il++)
            {
                int n = 0;
                foreach (int x in r.HostIndex[il]) if (x >= 0) n++;
                most = Math.Max(most, n);
                _halfFits[il] = n <= half;
                if (_halfFits[il]) fit++;
                if (il > 0 && _halfFits[il] && _halfFits[il - 1]) pairs++;
            }
            if (pairs == 0)
            {
                Console.Error.WriteLine($"[q4e-cuda] prompt layers are staged per ubatch: no two consecutive layers keep at most {half} "
                    + $"experts on the host (half the staging; up to {most} a layer).");
                return false;
            }
            var dev = _devs[0];
            dev.MakeCurrent();
            _halfStages = new CudaExpertStage[2];
            var minusOne = new int[_m.NExpert];
            Array.Fill(minusOne, -1);
            for (int h = 0; h < 2; h++)
            {
                var s = new CudaExpertStage
                {
                    Capacity = half, Stride = _expertStage.Stride,
                    Staging = _expertStage.Staging + (nint)((long)h * half * _expertStage.Stride),
                };
                var stageOf = AllocT(dev, DType.Int32, _m.NExpert);
                fixed (int* p = minusOne)
                    CudaDriverApi.cuMemcpyHtoD(Ptr(stageOf), (IntPtr)p, new UIntPtr((ulong)_m.NExpert * 4)).ThrowOnError();
                var ints = AllocT(dev, DType.Int32, 2);
                Zero(dev, ints, 4);
                s.StageOf = Ptr(stageOf);
                s.StageCount = Ptr(ints);
                s.StageFrom = (IntPtr)((long)Ptr(ints) + 4);
                s.StageList = Ptr(AllocT(dev, DType.Int32, half));
                _halfStages[h] = s;
                CudaDriverApi.cuEventCreate(out _prefetchDone[h], 0x02).ThrowOnError();
                CudaDriverApi.cuEventCreate(out _halfFree[h], 0x02).ThrowOnError();
            }
            _planBytes = (long)_m.NExpert * 4 + (long)half * 4 + 8 + 3L * _m.NExpert * 8;
            for (int h = 0; h < 2; h++)
                CudaDriverApi.cuMemHostAlloc(out _planPinned[h], new UIntPtr((ulong)_planBytes), 0).ThrowOnError();
            CudaDriverApi.cuStreamCreate(out _prefetchStream, 0x1 /*NON_BLOCKING*/).ThrowOnError();
            Console.Error.WriteLine($"[q4e-cuda] prompt groups stage each next layer's host experts ahead ({fit} of {_m.NLayer} layers "
                + $"fit a half of {half} experts, up to {most} a layer; TS_Q4E_PROMPT_PREFETCH=0 stages per ubatch).");
            _prefetchUsable = true;
            return true;
        }

        /// <summary>Every host expert of layer <paramref name="il"/> staged into half <paramref name="h"/> by the copy
        /// engines on <paramref name="stream"/>, after the half's previous layer unstaged when <paramref name="waitFree"/>:
        /// host index i goes to staging slot i, and the plan the unstage kernel reads (and the layer's tables pointing at
        /// the copies) is written from the host.</summary>
        private void PrefetchLayer(int il, int h, IntPtr stream, bool waitFree)
        {
            var r = _residency;
            var L = _layers[il];
            var s = _halfStages[h];
            if (waitFree)
                CudaDriverApi.cuStreamWaitEvent(stream, _halfFree[h], 0).ThrowOnError();
            if (L.Stage.IsValid && L.Stage.Staged)
            {
                // The pinned plan of this half was last read by the copy that staged the layer two before: done once its
                // event fired (the host may have run ahead of the GPU).
                CudaDriverApi.cuEventSynchronize(_prefetchDone[h]).ThrowOnError();
                int[] index = r.HostIndex[il];
                int n = 0;
                int* stageOf = (int*)_planPinned[h];
                int* stageList = stageOf + _m.NExpert;
                int* count = stageList + s.Capacity;
                ulong* tabs = (ulong*)((byte*)_planPinned[h] + (long)_m.NExpert * 4 + (long)s.Capacity * 4 + 8);
                for (int e = 0; e < _m.NExpert; e++)
                {
                    int i = index[e];
                    stageOf[e] = i >= 0 ? i : -1;
                    if (i >= 0)
                    {
                        stageList[i] = e;
                        n = Math.Max(n, i + 1);
                    }
                    for (int k = 0; k < 3; k++)
                        tabs[(long)k * _m.NExpert + e] = i >= 0
                            ? (ulong)(long)(s.Staging + (nint)((long)i * s.Stride + MatrixOffset(L.Stage, k)))
                            : RestingEntry(il, k, e);
                }
                count[0] = n;
                count[1] = 0;
                CudaDriverApi.cuMemcpyHtoDAsync(s.StageOf, (IntPtr)stageOf, new UIntPtr((ulong)_m.NExpert * 4), stream).ThrowOnError();
                CudaDriverApi.cuMemcpyHtoDAsync(s.StageList, (IntPtr)stageList, new UIntPtr((ulong)n * 4), stream).ThrowOnError();
                CudaDriverApi.cuMemcpyHtoDAsync(s.StageCount, (IntPtr)count, new UIntPtr(4), stream).ThrowOnError();
                IntPtr[] tabDev = { L.Stage.TabGate, L.Stage.TabUp, L.Stage.TabDown };
                for (int k = 0; k < 3; k++)
                    CudaDriverApi.cuMemcpyHtoDAsync(tabDev[k], (IntPtr)(tabs + (long)k * _m.NExpert), new UIntPtr((ulong)_m.NExpert * 8),
                        stream).ThrowOnError();
                long[] bytes = ExpertMatrixBytes(_m.Layers[il], _m.NExpert);
                for (int k = 0; k < 3 && n > 0; k++)
                {
                    var copy = new CudaDriverApi.CudaMemcpy2D
                    {
                        SrcMemoryType = CudaDriverApi.CU_MEMORYTYPE_HOST, SrcHost = r.Host[il][k], SrcPitch = new UIntPtr((ulong)bytes[k]),
                        DstMemoryType = CudaDriverApi.CU_MEMORYTYPE_DEVICE, DstDevice = s.Staging + (nint)MatrixOffset(L.Stage, k),
                        DstPitch = new UIntPtr((ulong)s.Stride),
                        WidthInBytes = new UIntPtr((ulong)bytes[k]), Height = new UIntPtr((ulong)n),
                    };
                    CudaDriverApi.cuMemcpy2DAsync(copy, stream).ThrowOnError();
                }
            }
            CudaDriverApi.cuEventRecord(_prefetchDone[h], stream).ThrowOnError();
        }

        // Without adaptation the layers' tables at rest never change: read once from the device. (Not the RestTab arrays:
        // without host compute those hold only the host copies, zero for experts with a slot.)
        private ulong[][][] _restingTables;

        /// <summary>Layer <paramref name="il"/>'s table entry at rest of expert <paramref name="e"/> for matrix
        /// <paramref name="k"/>: adaptation's host mirror, else the layer's live table read once while at rest (the layer
        /// not running, its last staging undone).</summary>
        private ulong RestingEntry(int il, int k, int e)
        {
            var r = _residency;
            if (r.Tab != null)
                return r.Tab[il][k][e];
            _restingTables ??= new ulong[_m.NLayer][][];
            if (_restingTables[il] == null)
            {
                var st = _layers[il].Stage;
                // Whatever staging the stream still has queued for this layer is undone first.
                CudaDriverApi.cuStreamSynchronize(_devs[0].Stream).ThrowOnError();
                IntPtr[] dev = { st.TabGate, st.TabUp, st.TabDown };
                var t = new ulong[3][];
                for (int j = 0; j < 3; j++)
                {
                    t[j] = new ulong[_m.NExpert];
                    fixed (ulong* p = t[j])
                        CudaDriverApi.cuMemcpyDtoH((IntPtr)p, dev[j], new UIntPtr((ulong)_m.NExpert * 8)).ThrowOnError();
                }
                _restingTables[il] = t;
            }
            return _restingTables[il][k][e];
        }

        private void DestroyPromptPrefetch()
        {
            if (_prefetchStream == IntPtr.Zero)
                return;
            try
            {
                _devs[0].MakeCurrent();
                CudaDriverApi.cuStreamSynchronize(_prefetchStream);
                for (int h = 0; h < 2; h++)
                {
                    CudaDriverApi.cuEventDestroy(_prefetchDone[h]);
                    CudaDriverApi.cuEventDestroy(_halfFree[h]);
                    if (_planPinned[h] != IntPtr.Zero)
                        CudaDriverApi.cuMemFreeHost(_planPinned[h]);
                }
                CudaDriverApi.cuStreamDestroy(_prefetchStream);
            }
            catch
            {
            }
            _prefetchStream = IntPtr.Zero;
        }
    }
}
