// Copyright (c) Zhongkai Fu. All rights reserved.
// https://github.com/zhongkaifu/TensorSharp
//
// This file is part of TensorSharp.
//
// TensorSharp is licensed under the BSD-3-Clause license found in the LICENSE file in the root directory of this source tree.
using System;
using System.Collections.Concurrent;
using System.Runtime.CompilerServices;
using System.Threading;
using TensorSharp.Cuda.Interop;
using TensorSharp.Graph;

namespace TensorSharp.Cuda
{
    /// <summary>
    /// The host side of host experts in a recorded step (<see cref="GraphOp.MoeHostExperts"/> on the CUDA executor): the
    /// device publishes a layer's input and routes into a host-mapped mailbox and rings (ts_graph_moe_route_split); a
    /// thread that spins while steps run computes the routes the device tier does not hold with the model's
    /// <see cref="IGraphExpertHost"/> and raises the flag the device waits on (ts_graph_moe_wait_add). The step never
    /// returns to the host, so a decode step with host experts stays one CUDA graph (Strata's doorbell, as the Qwen3.8
    /// engines ran it).
    /// </summary>
    internal sealed unsafe class CudaExpertDoorbell : IDisposable
    {
        /// <summary>Rows one doorbell carries: a decode step, or a verify window of up to seven drafts (as the Qwen3.8
        /// engines served them).</summary>
        public const int MaxRows = 8;

        private readonly CudaHostExpertMailbox _box;
        private readonly ConcurrentDictionary<int, IGraphExpertHost> _hosts = new();
        private readonly Thread _thread;
        private readonly AutoResetEvent _wake = new(false);
        private readonly long _hotTicks;
        private long _hotUntil;
        private volatile bool _stop;
        private Exception _error;

        public CudaExpertDoorbell(CudaAllocator allocator, int hidden, int routesPerRow)
        {
            allocator.Context.MakeCurrent();
            Hidden = hidden;
            RoutesPerRow = routesPerRow;
            _box = new CudaHostExpertMailbox(MaxRows, hidden, routesPerRow);
            CudaDriverApi.cuMemAlloc(out IntPtr miss, new UIntPtr(4)).ThrowOnError();
            CudaDriverApi.cuMemsetD8Async(miss, 0, new UIntPtr(4), IntPtr.Zero).ThrowOnError();
            MissDev = miss;
            // Spins this long after it was armed or last answered (a reply's steps arrive back to back), then sleeps
            // until armed again.
            _hotTicks = System.Diagnostics.Stopwatch.Frequency / 20;
            _thread = new Thread(Loop) { IsBackground = true, Name = "cuda-expert-doorbell", Priority = ThreadPriority.Highest };
            _thread.Start();
        }

        public int Hidden { get; }
        public int RoutesPerRow { get; }
        public CudaHostExpertMailbox Box => _box;

        /// <summary>Device int: whether the last split rang (the wait kernel skips otherwise).</summary>
        public IntPtr MissDev { get; }

        /// <summary>The first failure of a host computation (its layer's host sum was left out).</summary>
        public Exception Error => Volatile.Read(ref _error);

        public long Served;

        /// <summary>The expert host serving <paramref name="layer"/>'s doorbells.</summary>
        public void Register(int layer, IGraphExpertHost host) => _hosts[layer] = host;

        /// <summary>Before work that may ring: the thread spins until it has gone quiet again.</summary>
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
                _wake.WaitOne(20);
            }
        }


        private void Serve(uint seq)
        {
            int count = 0;
            try
            {
                int layer = _box.Header[0], slots = _box.Header[1];
                if (!_hosts.TryGetValue(layer, out IGraphExpertHost host) || slots <= 0 || slots % RoutesPerRow != 0
                    || slots > MaxRows * RoutesPerRow)
                    throw new InvalidOperationException($"CUDA host experts: a doorbell for layer {layer} with {slots} routes");
                int rows = slots / RoutesPerRow;
                // The device's routes (and those the tail skip leaves out) arrive as -(e + 2) and are skipped here: the device
                // table alone splits them. The device counts every route for the routing profile.
                int* decoded = stackalloc int[slots];
                byte* skip = stackalloc byte[slots];
                for (int k = 0; k < slots; k++)
                {
                    int id = _box.Ids[k];
                    decoded[k] = id <= -2 ? -(id + 2) : id;
                    skip[k] = id < 0 ? (byte)1 : (byte)0;
                }
                var ids = new ReadOnlySpan<int>(decoded, slots);
                var output = new Span<float>(_box.Out, rows * Hidden);
                if (!host.Compute(layer, new ReadOnlySpan<float>(_box.X, rows * Hidden), ids, new ReadOnlySpan<float>(_box.W, slots),
                        new ReadOnlySpan<byte>(skip, slots), rows, output))
                    throw new NotSupportedException($"CUDA host experts: the host's expert kernels do not cover layer {layer}'s experts");
                count = rows;
            }
            catch (Exception ex)
            {
                Interlocked.CompareExchange(ref _error, ex, null);
                count = 0;
            }
            Volatile.Write(ref *_box.Count, count);
            Interlocked.MemoryBarrier();
            Volatile.Write(ref Unsafe.AsRef<uint>(_box.Flag), seq);
            Interlocked.Increment(ref Served);
        }

        public void Dispose()
        {
            if (_stop) return;
            _stop = true;
            _wake.Set();
            _thread.Join();
            _wake.Dispose();
            CudaDriverApi.cuMemFree(MissDev);
            _box.Dispose();
        }
    }
}
