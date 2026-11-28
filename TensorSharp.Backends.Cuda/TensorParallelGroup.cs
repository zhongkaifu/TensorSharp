using System;
using System.Linq;

namespace TensorSharp.Cuda
{
    /// <summary>
    /// Coordinates tensor-parallel execution across multiple CUDA GPUs within a
    /// single process. Owns one <see cref="CudaAllocator"/> per GPU and a
    /// <see cref="CudaP2PCommunicator"/> for collective operations.
    ///
    /// Usage: create with the desired TP degree (number of GPUs), then use
    /// <see cref="AllReduce"/> at row-parallel boundaries and
    /// <see cref="GetAllocator"/> to place per-rank tensors.
    /// </summary>
    public sealed class TensorParallelGroup : ITensorParallelGroup
    {
        private readonly CudaAllocator[] _allocators;
        private readonly CudaP2PCommunicator _communicator;
        private bool _disposed;

        // Diagnostic/fallback: when set, reduce across the local GPUs by staging
        // through host memory (device→host, sum, host→device) instead of the
        // device-to-device P2P path. This mirrors the multi-node AllReduce, which
        // is known-good, and is enabled with TENSORSHARP_TP_HOST_ALLREDUCE=1 to
        // isolate P2P-specific correctness issues.
        private static readonly bool _forceHostAllReduce =
            string.Equals(Environment.GetEnvironmentVariable("TENSORSHARP_TP_HOST_ALLREDUCE"), "1", StringComparison.Ordinal);

        public TensorParallelGroup(int degree)
        {
            if (degree < 1)
                throw new ArgumentOutOfRangeException(nameof(degree), "TP degree must be >= 1.");

            int deviceCount = CudaDevice.GetDeviceCount();
            if (degree > deviceCount)
                throw new InvalidOperationException(
                    $"Requested TP degree {degree} but only {deviceCount} CUDA device(s) available.");

            Degree = degree;
            _allocators = new CudaAllocator[degree];

            try
            {
                for (int i = 0; i < degree; i++)
                    _allocators[i] = new CudaAllocator(i);

                if (degree > 1)
                    _communicator = new CudaP2PCommunicator(_allocators);
            }
            catch
            {
                for (int i = 0; i < degree; i++)
                    _allocators[i]?.Dispose();
                throw;
            }

            Console.WriteLine($"Tensor parallelism: {degree} GPUs " +
                $"({string.Join(", ", System.Linq.Enumerable.Range(0, degree).Select(i => CudaDevice.GetDevice(i).Name))})");
        }

        /// <summary>The group's peer collective (tests; null with one GPU).</summary>
        internal CudaP2PCommunicator Communicator => _communicator;

        /// <summary>Number of GPUs in this parallel group.</summary>
        public int Degree { get; }

        /// <summary>True when TP is active (degree > 1).</summary>
        public bool IsActive => Degree > 1;

        /// <summary>Single-node: global degree equals local degree.</summary>
        public int GlobalDegree => Degree;

        /// <summary>Single-node: no rank offset.</summary>
        public int GlobalRankOffset => 0;

        /// <summary>Single-node: one node.</summary>
        public int NodeCount => 1;

        public IAllocator GetAllocator(int rank) => GetCudaAllocator(rank);

        /// <summary>Concretely-typed accessor for CUDA-only call sites.</summary>
        public CudaAllocator GetCudaAllocator(int rank)
        {
            if ((uint)rank >= (uint)Degree)
                throw new ArgumentOutOfRangeException(nameof(rank));
            return _allocators[rank];
        }

        /// <summary>
        /// In-place AllReduce (element-wise sum) across all GPUs.
        /// tensors[i] must be a F32 tensor on GPU i, all the same shape.
        /// After this call every tensor holds the global sum.
        /// </summary>
        public void AllReduce(Tensor[] tensors)
        {
            if (!IsActive) return;
            if (_forceHostAllReduce)
                HostAllReduce(tensors);
            else
                _communicator.AllReduce(tensors);
        }

        /// <summary>
        /// Host-staged AllReduce across the local GPUs: read each rank's tensor
        /// to host, sum, write the result back to every rank and force the device
        /// upload. Correctness fallback for the device-to-device P2P path; slower
        /// but matches the multi-node reduce exactly. Gated by
        /// TENSORSHARP_TP_HOST_ALLREDUCE.
        /// </summary>
        private void HostAllReduce(Tensor[] tensors)
        {
            if (tensors == null || tensors.Length != Degree)
                throw new ArgumentException($"Expected {Degree} tensors, got {tensors?.Length ?? 0}.");

            // Make sure every rank's partial is finished before reading it back.
            for (int r = 0; r < Degree; r++)
                _allocators[r].Synchronize();

            int n = (int)tensors[0].Storage.ElementCount;
            float[] acc = tensors[0].GetElementsAsFloat(n);
            for (int r = 1; r < Degree; r++)
            {
                float[] part = tensors[r].GetElementsAsFloat(n);
                for (int i = 0; i < n; i++)
                    acc[i] += part[i];
            }

            for (int r = 0; r < Degree; r++)
            {
                tensors[r].SetElementsAsFloat(acc);
                tensors[r].EnsureDeviceCurrent();
            }

            for (int r = 0; r < Degree; r++)
                _allocators[r].Synchronize();
        }

        /// <summary>Synchronize all GPU streams.</summary>
        public void Synchronize()
        {
            for (int i = 0; i < Degree; i++)
                _allocators[i].Synchronize();
        }

        /// <summary>Single-node: all GPUs share the process, no barrier needed.</summary>
        public void Barrier() { }

        // Single-node group: there are no worker nodes, so the driver/worker
        // control channel is never used. (NodeCount == 1.)
        public void BroadcastControl(int op, int[] payload) =>
            throw new NotSupportedException("Control broadcast is only meaningful for multi-node distributed groups.");

        public (int op, int[] payload) ReceiveControl() =>
            throw new NotSupportedException("Control receive is only meaningful for multi-node distributed groups.");

        public void Dispose()
        {
            if (_disposed) return;
            _disposed = true;

            _communicator?.Dispose();
            for (int i = 0; i < Degree; i++)
                _allocators[i]?.Dispose();
        }
    }
}
