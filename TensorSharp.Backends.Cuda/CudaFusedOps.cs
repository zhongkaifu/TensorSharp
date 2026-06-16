namespace TensorSharp.Cuda
{
    public static class CudaFusedOps
    {
        // Go/no-go PoC for the CUDA-graph rearchitecture: measure how much of a
        // launch-heavy op sequence is per-op CPU/WDDM launch overhead (which a captured
        // graph replays in ONE launch) vs GPU compute. `issueOneLaunch` must issue
        // exactly one capturable launch on `onStream`'s stream (no host sync / alloc).
        // Returns wall ms for `iters` per-op launches vs one graph replay of the same.
        public static (double peropMs, double graphMs, bool captured) MeasureGraphReplay(
            Tensor onStream, int iters, System.Action issueOneLaunch)
        {
            var storage = onStream.Storage as CudaStorage;
            if (storage == null) return (0, 0, false);
            System.IntPtr stream = storage.AllocatorImpl.Stream.Handle;
            storage.AllocatorImpl.Context.MakeCurrent();

            for (int i = 0; i < iters; i++) issueOneLaunch();   // warm up
            Interop.CudaDriverApi.cuStreamSynchronize(stream);

            var sw = System.Diagnostics.Stopwatch.StartNew();
            for (int i = 0; i < iters; i++) issueOneLaunch();
            Interop.CudaDriverApi.cuStreamSynchronize(stream);
            sw.Stop();
            double peropMs = sw.Elapsed.TotalMilliseconds;

            if (Interop.CudaDriverApi.cuStreamBeginCapture(stream,
                    Interop.CudaDriverApi.CU_STREAM_CAPTURE_MODE_THREAD_LOCAL) != 0)
                return (peropMs, 0, false);
            for (int i = 0; i < iters; i++) issueOneLaunch();
            if (Interop.CudaDriverApi.cuStreamEndCapture(stream, out System.IntPtr graph) != 0)
                return (peropMs, 0, false);
            if (Interop.CudaDriverApi.cuGraphInstantiateWithFlags(out System.IntPtr exec, graph, 0) != 0)
            {
                Interop.CudaDriverApi.cuGraphDestroy(graph);
                return (peropMs, 0, false);
            }

            Interop.CudaDriverApi.cuGraphLaunch(exec, stream);   // warm up replay
            Interop.CudaDriverApi.cuStreamSynchronize(stream);

            var sw2 = System.Diagnostics.Stopwatch.StartNew();
            Interop.CudaDriverApi.cuGraphLaunch(exec, stream);
            Interop.CudaDriverApi.cuStreamSynchronize(stream);
            sw2.Stop();
            double graphMs = sw2.Elapsed.TotalMilliseconds;

            Interop.CudaDriverApi.cuGraphExecDestroy(exec);
            Interop.CudaDriverApi.cuGraphDestroy(graph);
            return (peropMs, graphMs, true);
        }

        public static bool TryGqaPrefillAttention(
            Tensor result,
            Tensor query,
            Tensor key,
            Tensor value,
            int numQHeads,
            int numKVHeads,
            int headDim,
            int seqLen,
            int kvLen,
            int maskStart,
            int windowSize,
            float scale,
            int kvStride = -1)
        {
            return CudaKernelOps.TryGqaPrefillAttention(
                result,
                query,
                key,
                value,
                numQHeads,
                numKVHeads,
                headDim,
                seqLen,
                kvLen,
                maskStart,
                windowSize,
                scale,
                kvStride);
        }

        public static bool TryAttentionSoftmaxWithSinks(
            Tensor scores,
            Tensor sinks,
            int numHeads,
            int seqLen,
            int kvLen,
            int maskStart,
            int windowSize,
            float scale)
        {
            return CudaKernelOps.TryAttentionSoftmaxWithSinks(
                scores,
                sinks,
                numHeads,
                seqLen,
                kvLen,
                maskStart,
                windowSize,
                scale);
        }

        public static bool TrySliceColumns(Tensor result, Tensor src, int colOffset, int width)
        {
            return CudaKernelOps.TrySliceColumns(result, src, colOffset, width);
        }

        public static bool TryGqaDecodeAttention(
            Tensor result,
            Tensor query,
            Tensor keyCache,
            Tensor valueCache,
            int numQHeads,
            int numKVHeads,
            int headDim,
            int attendStart,
            int attendLen,
            int cacheSize,
            bool circular,
            float scale)
        {
            return CudaKernelOps.TryGqaDecodeAttention(
                result,
                query,
                keyCache,
                valueCache,
                numQHeads,
                numKVHeads,
                headDim,
                attendStart,
                attendLen,
                cacheSize,
                circular,
                scale);
        }

        public static bool TryGqaDecodeAttentionWithSinks(
            Tensor result,
            Tensor query,
            Tensor keyCache,
            Tensor valueCache,
            Tensor sinks,
            int numQHeads,
            int numKVHeads,
            int headDim,
            int attendStart,
            int attendLen,
            int cacheSize,
            bool circular,
            float scale)
        {
            return CudaKernelOps.TryGqaDecodeAttentionWithSinks(
                result,
                query,
                keyCache,
                valueCache,
                sinks,
                numQHeads,
                numKVHeads,
                headDim,
                attendStart,
                attendLen,
                cacheSize,
                circular,
                scale);
        }

        public static bool TryGqaPrefillAttentionWithSinks(
            Tensor result,
            Tensor query,
            Tensor keyCache,
            Tensor valueCache,
            Tensor sinks,
            int numQHeads,
            int numKVHeads,
            int headDim,
            int seqLen,
            int kvLen,
            int cacheSize,
            int maskStart,
            int windowSize,
            float scale)
        {
            return CudaKernelOps.TryGqaPrefillAttentionWithSinks(
                result,
                query,
                keyCache,
                valueCache,
                sinks,
                numQHeads,
                numKVHeads,
                headDim,
                seqLen,
                kvLen,
                cacheSize,
                maskStart,
                windowSize,
                scale);
        }

        public static bool TryAddBiasRows(Tensor tensor, Tensor bias)
        {
            return CudaKernelOps.TryAddBiasRows(tensor, bias);
        }

        public static bool TryFlatToHeadFirst(Tensor result, Tensor src, int numHeads, int seqLen, int headDim)
        {
            return CudaKernelOps.TryFlatToHeadFirst(result, src, numHeads, seqLen, headDim);
        }

        public static bool TrySplitQkvToHeadFirst(Tensor result, Tensor qkv, int colOffset, int numHeads, int seqLen, int headDim)
        {
            return CudaKernelOps.TrySplitQkvToHeadFirst(result, qkv, colOffset, numHeads, seqLen, headDim);
        }

        public static bool TryCopyHeadFirstToCache(Tensor cache, Tensor src, int startPos, int seqLen, int cacheSize, bool circular)
        {
            return CudaKernelOps.TryCopyHeadFirstToCache(cache, src, startPos, seqLen, cacheSize, circular);
        }

        public static bool TryGatherCircularHeadFirst(Tensor result, Tensor cache, int startPos, int seqLen, int cacheSize)
        {
            return CudaKernelOps.TryGatherCircularHeadFirst(result, cache, startPos, seqLen, cacheSize);
        }

        public static bool TryConcatHeadFirst(Tensor result, Tensor a, Tensor b)
        {
            return CudaKernelOps.TryConcatHeadFirst(result, a, b);
        }

        public static bool TryNeoXRoPEHeadFirst(Tensor data, Tensor cosTable, Tensor sinTable, int numHeads, int seqLen, int headDim, int ropeHalf)
        {
            return CudaKernelOps.TryNeoXRoPEHeadFirst(data, cosTable, sinTable, numHeads, seqLen, headDim, ropeHalf);
        }

        public static bool TryNeoXRoPEFlatInPlace(Tensor data, Tensor cosTable, Tensor sinTable, int numHeads, int seqLen, int headDim, int ropeHalf)
        {
            return CudaKernelOps.TryNeoXRoPEFlatInPlace(data, cosTable, sinTable, numHeads, seqLen, headDim, ropeHalf);
        }

        // residual += rms_norm(input, alpha) (Gemma post-norm), fused into one kernel.
        public static bool TryRmsNormResidualAdd(Tensor residual, Tensor input, Tensor alpha, float eps)
        {
            return CudaKernelOps.TryRmsNormResidualAdd(residual, input, alpha, eps);
        }

        public static bool TryGELUMulSplit(Tensor result, Tensor gateUp, int halfDim)
        {
            return CudaKernelOps.TryGELUMulSplit(result, gateUp, halfDim);
        }

        public static bool TrySwiGluOaiSplit(Tensor result, Tensor gateUp, int halfDim, float alpha, float limit)
        {
            return CudaKernelOps.TrySwiGluOaiSplit(result, gateUp, halfDim, alpha, limit);
        }

        public static bool TryQwen35GatedDeltaNetPacked(
            Tensor result,
            Tensor packed,
            Tensor convState,
            Tensor ssmState,
            Tensor convWeight,
            Tensor dtBias,
            Tensor aLog,
            Tensor ssmNorm,
            int seqLen,
            int packedDim,
            int qkvDim,
            int qkDim,
            int vDim,
            int numKHeads,
            int numVHeads,
            int headKDim,
            int headVDim,
            int convKernel,
            int convWriteIdx,
            float eps)
        {
            return CudaKernelOps.TryQwen35GatedDeltaNetPacked(
                result,
                packed,
                convState,
                ssmState,
                convWeight,
                dtBias,
                aLog,
                ssmNorm,
                seqLen,
                packedDim,
                qkvDim,
                qkDim,
                vDim,
                numKHeads,
                numVHeads,
                headKDim,
                headVDim,
                convKernel,
                convWriteIdx,
                eps);
        }

        public static bool TryQwen35GatedDeltaNetPackInputs(
            Tensor packed,
            Tensor qkv,
            Tensor z,
            Tensor beta,
            Tensor alpha,
            int seqLen,
            int qkvDim,
            int zDim,
            int numVHeads,
            int packedDim)
        {
            return CudaKernelOps.TryQwen35GatedDeltaNetPackInputs(
                packed,
                qkv,
                z,
                beta,
                alpha,
                seqLen,
                qkvDim,
                zDim,
                numVHeads,
                packedDim);
        }
    }
}
