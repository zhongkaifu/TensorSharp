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
        /// <summary>Rows one doorbell carries: a decode step, a verify window, or a short chunk of up to 16 rows (a
        /// continuation's few new tokens), as the Qwen3.8 engines' host mailbox took them (HostExpertMaxRows), and as many as
        /// the model's expert host computes in one call. Past 8 rows a 13-row continuation of Qwen3.8 Flash Next UD-Q2_K_XL
        /// on an A40 fell to the executor's own host path, every layer's routes read back and computed on the host while
        /// the device waited: 0.94-1.13 s to the first token where the engine took 0.07 s.</summary>
        public const int MaxRows = 16;

        private readonly CudaHostExpertMailbox _box;
        private readonly ConcurrentDictionary<int, (IGraphExpertHost Host, CudaOaiExperts Oai)> _hosts = new();
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

        /// <summary>The expert host serving <paramref name="layer"/>'s doorbells; with <paramref name="oai"/>, GPT-OSS's experts
        /// (IGraphExpertHost.ComputeOai at its alpha and limit) without the down bias, which the device adds for every route.</summary>
        public void Register(int layer, IGraphExpertHost host, CudaOaiExperts oai = null) => _hosts[layer] = (host, oai);

        /// <summary>Before work that may ring: the thread spins until it has gone quiet again.</summary>
        public void Arm()
        {
            Volatile.Write(ref _hotUntil, System.Diagnostics.Stopwatch.GetTimestamp() + _hotTicks);
            _wake.Set();
        }

        private void Loop()
        {
            while (!_stop)
            {
                // Every ring the flag has not answered, from whenever this thread first runs. It had taken the counter it
                // found then as served: a ring the device made before the new thread was scheduled (the first decode step
                // rings right after the doorbell is made) was never answered, and the step's wait kernel spun for good.
                uint seq = Volatile.Read(ref Unsafe.AsRef<uint>(_box.Seq));
                if (seq != Volatile.Read(ref Unsafe.AsRef<uint>(_box.Flag)))
                {
                    Serve(seq);
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
                if (!_hosts.TryGetValue(layer, out (IGraphExpertHost Host, CudaOaiExperts Oai) entry) || slots <= 0 || slots % RoutesPerRow != 0
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
                var x = new ReadOnlySpan<float>(_box.X, rows * Hidden);
                var w = new ReadOnlySpan<float>(_box.W, slots);
                var skipped = new ReadOnlySpan<byte>(skip, slots);
                if (entry.Oai is { } oai
                        ? !entry.Host.ComputeOai(layer, x, ids, w, skipped, rows, output, oai.Alpha, oai.Limit, downBias: false)
                        : !entry.Host.Compute(layer, x, ids, w, skipped, rows, output))
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
