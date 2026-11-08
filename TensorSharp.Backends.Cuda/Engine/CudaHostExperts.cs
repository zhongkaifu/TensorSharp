// Copyright (c) Zhongkai Fu. All rights reserved.
// https://github.com/zhongkaifu/TensorSharp
//
// This file is part of TensorSharp.
//
// TensorSharp is licensed under the BSD-3-Clause license found in the LICENSE file in the root directory of this source tree.
//
// Host experts for expert residency (Strata's doorbell). The experts without a VRAM slot live in pinned host
// memory, and a decode step computes them on the CPU while the GPU computes the rest, inside the one captured
// step: per layer the GPU publishes the FFN input and the selection into a mailbox in host-mapped memory and
// advances a counter; a host thread that spins while steps run sees it, computes the missed experts, writes
// their sum and sets a flag to the counter's value; a GPU kernel waits for the flag and adds the sum. No expert
// crosses the bus and the step never returns to the host. The GPU computes a missed expert as a zero expert
// (its table entry points at zeroed VRAM), so it contributes exactly nothing there.
using System;
using System.Runtime.CompilerServices;
using System.Threading;
using TensorSharp.Cuda.Interop;

namespace TensorSharp.Cuda
{
    /// <summary>
    /// Routed experts of <paramref name="rows"/> independent rows on the host: out[r] = sum over slots k of row r
    /// with skip[k] == 0 of weights[k] * expert_{ids[k]}(x[r]), from stacks of <paramref name="numExperts"/> experts.
    /// False for a shape or type it does not cover.
    /// </summary>
    public unsafe delegate bool CudaHostExpertRows(IntPtr gate, IntPtr up, IntPtr down, int gateType, int upType, int downType,
        int numExperts, int hidden, int nFf, int nUsed, int rows, float* x, int* ids, float* weights, byte* skip, float* output);

    /// <summary>
    /// Process totals of the direct CUDA engines' expert residency, for telemetry (tensorsharp.moe.*): decode-step
    /// routes whose expert had a VRAM slot, and adaptive residency swaps. Monotonic across engines. The host-computed
    /// routes and their time are counted by the host expert kernel itself (TSGgml_MoeOffloadCounters).
    /// </summary>
    public static class CudaMoeOffloadCounters
    {
        private static long s_deviceRoutes, s_swaps;

        public static long DeviceRoutes => Interlocked.Read(ref s_deviceRoutes);
        public static long Swaps => Interlocked.Read(ref s_swaps);

        internal static void AddDeviceRoutes(long n) { if (n > 0) Interlocked.Add(ref s_deviceRoutes, n); }
        internal static void AddSwap() => Interlocked.Increment(ref s_swaps);
    }

    /// <summary>The doorbell's shared memory: pinned, device-mapped, one block.</summary>
    internal sealed unsafe class CudaHostExpertMailbox : IDisposable
    {
        public readonly int MaxRows, Hidden, NUsed;
        public IntPtr XDev, IdsDev, WDev, HeaderDev, SeqDev, FlagDev, CountDev, OutDev;
        public float* X, W, Out;
        public int* Ids, Header, Count;
        public uint* Seq, Flag;
        private IntPtr _host;

        public CudaHostExpertMailbox(int maxRows, int hidden, int nUsed)
        {
            MaxRows = maxRows;
            Hidden = hidden;
            NUsed = nUsed;
            // Each field on its own 128-byte line: the GPU writes the counter, the host the flag.
            static long Line(long b) => (b + 127) & ~127L;
            long oX = 0, oOut = oX + Line((long)maxRows * hidden * 4), oIds = oOut + Line((long)maxRows * hidden * 4);
            long oW = oIds + Line((long)maxRows * nUsed * 4), oHdr = oW + Line((long)maxRows * nUsed * 4);
            long oSeq = oHdr + 128, oFlag = oSeq + 128, oCount = oFlag + 128, total = oCount + 128;
            CudaDriverApi.cuMemHostAlloc(out _host, new UIntPtr((ulong)total),
                CudaDriverApi.CU_MEMHOSTALLOC_PORTABLE | CudaDriverApi.CU_MEMHOSTALLOC_DEVICEMAP).ThrowOnError();
            new Span<byte>((void*)_host, (int)total).Clear();
            CudaDriverApi.cuMemHostGetDevicePointer(out IntPtr dev, _host, 0).ThrowOnError();
            byte* h = (byte*)_host;
            X = (float*)(h + oX); Out = (float*)(h + oOut); Ids = (int*)(h + oIds); W = (float*)(h + oW);
            Header = (int*)(h + oHdr); Seq = (uint*)(h + oSeq); Flag = (uint*)(h + oFlag); Count = (int*)(h + oCount);
            XDev = dev + (nint)oX; OutDev = dev + (nint)oOut; IdsDev = dev + (nint)oIds; WDev = dev + (nint)oW;
            HeaderDev = dev + (nint)oHdr; SeqDev = dev + (nint)oSeq; FlagDev = dev + (nint)oFlag; CountDev = dev + (nint)oCount;
        }

        public void Dispose()
        {
            if (_host != IntPtr.Zero)
                CudaDriverApi.cuMemFreeHost(_host);
            _host = IntPtr.Zero;
        }
    }

    /// <summary>The host thread that answers the doorbell.</summary>
    internal sealed unsafe class CudaHostExpertServer : IDisposable
    {
        /// <summary>One layer's experts without a slot: packed stacks and each expert's index in them (-1: a slot).</summary>
        public sealed class Layer
        {
            public IntPtr Gate, Up, Down;
            public int GateType, UpType, DownType, Packed;
            public int[] HostIndex;
        }

        private readonly CudaHostExpertMailbox _box;
        private readonly Layer[] _layers;
        private readonly int _hidden, _nFf, _nUsed;
        private readonly CudaHostExpertRows _compute;
        private readonly Thread _thread;
        private readonly AutoResetEvent _wake = new AutoResetEvent(false);
        private readonly long _hotTicks;
        private long _hotUntil;
        private volatile bool _stop;
        private Exception _error;
        private long _served, _missed, _rows;

        /// <summary>Per layer, how often each expert was selected in decode steps (adaptive residency), or null.
        /// Written by the server while steps run, read and decayed by the engine between them.</summary>
        public float[][] Counts;

        public CudaHostExpertServer(CudaHostExpertMailbox box, Layer[] layers, int hidden, int nFf, int nUsed, CudaHostExpertRows compute)
        {
            _box = box;
            _layers = layers;
            _hidden = hidden;
            _nFf = nFf;
            _nUsed = nUsed;
            _compute = compute;
            // The server spins this long after the engine last armed it or it last answered (steps arrive
            // back to back while a reply decodes), then sleeps until armed again.
            _hotTicks = System.Diagnostics.Stopwatch.Frequency / 20;
            _thread = new Thread(Loop) { IsBackground = true, Name = "q4e-host-experts", Priority = ThreadPriority.Highest };
            _thread.Start();
        }

        /// <summary>The first failure of a host computation; its layer's host sum was left out.</summary>
        public Exception Error => Volatile.Read(ref _error);

        public long Served => Interlocked.Read(ref _served);
        public long Missed => Interlocked.Read(ref _missed);

        /// <summary>Before work that rings the doorbell: the server spins until it has gone quiet again.</summary>
        public void Arm()
        {
            Volatile.Write(ref _hotUntil, System.Diagnostics.Stopwatch.GetTimestamp() + _hotTicks);
            _wake.Set();
        }

        private void Loop()
        {
            uint served = Volatile.Read(ref Unsafe.AsRef<uint>(_box.Seq));
            while (!_stop)
            {
                uint seq = Volatile.Read(ref Unsafe.AsRef<uint>(_box.Seq));
                if (seq != served)
                {
                    Serve(seq);
                    served = seq;
                    Volatile.Write(ref _hotUntil, System.Diagnostics.Stopwatch.GetTimestamp() + _hotTicks);
                    continue;
                }
                if (System.Diagnostics.Stopwatch.GetTimestamp() < Volatile.Read(ref _hotUntil))
                {
                    Thread.SpinWait(4);
                    continue;
                }
                _wake.WaitOne(50);
            }
        }

        private void Serve(uint seq)
        {
            var box = _box;
            int count = 0;
            try
            {
                int layer = box.Header[0], slots = box.Header[1], miss = box.Header[2];
                int hostSlots = 0;
                float[][] counts = Counts;
                if (counts != null && (uint)layer < (uint)counts.Length && (uint)slots <= (uint)(box.MaxRows * _nUsed))
                {
                    float[] c = counts[layer];
                    for (int k = 0; k < slots; k++)
                        if ((uint)box.Ids[k] < (uint)c.Length)
                            c[box.Ids[k]] += 1f;
                }
                if (miss != 0)
                {
                    if ((uint)layer >= (uint)_layers.Length || _layers[layer] == null || slots <= 0 || slots % _nUsed != 0
                        || slots > box.MaxRows * _nUsed)
                        throw new InvalidOperationException($"[q4e-cuda] host experts: a doorbell for layer {layer} with {slots} slots");
                    var L = _layers[layer];
                    int* ids = stackalloc int[slots];
                    byte* skip = stackalloc byte[slots];
                    int misses = 0;
                    for (int k = 0; k < slots; k++)
                    {
                        int e = box.Ids[k];
                        int idx = (uint)e < (uint)L.HostIndex.Length ? L.HostIndex[e] : -1;
                        bool host = idx >= 0;
                        ids[k] = host ? idx : 0;
                        skip[k] = host ? (byte)0 : (byte)1;
                        if (host) misses++;
                    }
                    if (misses > 0)
                    {
                        int rows = slots / _nUsed;
                        if (!_compute(L.Gate, L.Up, L.Down, L.GateType, L.UpType, L.DownType, L.Packed, _hidden, _nFf, _nUsed, rows,
                                box.X, ids, box.W, skip, box.Out))
                            throw new NotSupportedException(
                                $"[q4e-cuda] host experts: the host kernel does not cover layer {layer}'s experts " +
                                $"({CudaMoe.GgmlTypeName(L.GateType)}/{CudaMoe.GgmlTypeName(L.UpType)}/{CudaMoe.GgmlTypeName(L.DownType)}).");
                        count = rows;
                        hostSlots = misses;
                        Interlocked.Add(ref _missed, misses);
                        Interlocked.Add(ref _rows, rows);
                    }
                }
            }
            catch (Exception ex)
            {
                Interlocked.CompareExchange(ref _error, ex, null);
                count = 0;
            }
            Volatile.Write(ref *box.Count, count);
            Interlocked.MemoryBarrier();
            Volatile.Write(ref Unsafe.AsRef<uint>(box.Flag), seq);
            Interlocked.Increment(ref _served);
        }

        public void Dispose()
        {
            _stop = true;
            _wake.Set();
            _thread.Join();
            _wake.Dispose();
        }
    }
}
