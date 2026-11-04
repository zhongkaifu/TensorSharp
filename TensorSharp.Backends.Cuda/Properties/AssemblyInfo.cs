using System.Runtime.CompilerServices;

[assembly: InternalsVisibleTo("InferenceWeb.Tests")]
[assembly: InternalsVisibleTo("CudaQuantizedMatmulBench")]
[assembly: InternalsVisibleTo("Ltx25CudaDequantProbe")]
[assembly: InternalsVisibleTo("Ltx25AttentionProbe")]

[assembly: InternalsVisibleTo("NinferCudaKernelProbe")]
[assembly: InternalsVisibleTo("NinferA16KernelProbe")]
[assembly: InternalsVisibleTo("NinferA16TimingProbe")]
[assembly: InternalsVisibleTo("NinferGdnChunkProbe")]
[assembly: InternalsVisibleTo("NativeHostEmbeddingProbe")]
[assembly: InternalsVisibleTo("NinferRepackProbe")]
[assembly: InternalsVisibleTo("NinferTargetVerifyProbe")]
[assembly: InternalsVisibleTo("NinferDFlashHostProbe")]
