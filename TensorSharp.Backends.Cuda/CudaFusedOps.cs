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
    }
}
