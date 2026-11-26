using TensorSharp;
using TensorSharp.Cpu;
using TensorSharp.Cuda;
using TensorSharp.MLX;
using TensorSharp.Models.MiniMaxH3;
using TensorSharp.Runtime;

namespace InferenceWeb.Tests;

/// <summary>Select one actual backend per model-oracle process. The default
/// retains the process's pinned GGML backend; direct paths require an allocator.</summary>
internal sealed class MiniMaxH3OracleBackend : IDisposable
{
    internal BackendType Backend { get; }
    private readonly IAllocator _allocator;
    internal MiniMaxH3OracleBackend()
    {
        string selected=Environment.GetEnvironmentVariable("TS_TEST_H3_BACKEND")?.Trim().ToLowerInvariant();
        Backend=selected switch
        {
            "cuda" => BackendType.Cuda, "mlx" => BackendType.Mlx, "cpu" => BackendType.Cpu,
            "ggml_cuda" => BackendType.GgmlCuda, "ggml_metal" => BackendType.GgmlMetal, "ggml_cpu" => BackendType.GgmlCpu,
            null or "" => TestGates.PinnedGgmlBackend,
            _ => throw new ArgumentException($"Unsupported H3 oracle backend '{selected}'."),
        };
        if (Backend is BackendType.GgmlCuda or BackendType.GgmlMetal or BackendType.GgmlCpu)
        {
            string mismatch=TestGates.GgmlPinSkip(Backend);
            if (mismatch != null) throw new InvalidOperationException(mismatch);
        }
        _allocator=Backend switch
        {
            BackendType.Cuda => new CudaAllocator(),
            BackendType.Mlx => new MlxAllocator(),
            BackendType.Cpu => new CpuAllocator(BlasEnum.DotNet),
            _ => null,
        };
    }
    internal MiniMaxH3TextEncoder Text(string path,string tokenizer) => new(path,tokenizer,Backend,_allocator);
    internal MiniMaxH3Denoiser Dit(string path) => new(path,Backend,_allocator);
    internal MiniMaxH3VideoVae Video(string path) => new(path,Backend,_allocator);
    internal MiniMaxH3AudioVae Audio(string path) => new(path,Backend,_allocator != null,_allocator);
    public void Dispose() => (_allocator as IDisposable)?.Dispose();
}
