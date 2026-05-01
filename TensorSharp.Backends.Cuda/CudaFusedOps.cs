namespace TensorSharp.Cuda
{
    public static class CudaFusedOps
    {
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
            float scale)
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

        public static bool TryGELUMulSplit(Tensor result, Tensor gateUp, int halfDim)
        {
            return CudaKernelOps.TryGELUMulSplit(result, gateUp, halfDim);
        }
    }
}
