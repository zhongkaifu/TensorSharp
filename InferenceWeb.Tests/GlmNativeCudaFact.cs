using Xunit.Sdk;

namespace InferenceWeb.Tests;

[TraitDiscoverer("InferenceWeb.Tests.RequiresTraitDiscoverer", "InferenceWeb.Tests")]
public sealed class GlmNativeCudaFactAttribute : FactAttribute, ITraitAttribute
{
    public string RequiresValue => "Cuda";

    public GlmNativeCudaFactAttribute(int minimumGpuCount = 1)
    {
        Skip = Environment.GetEnvironmentVariable("TS_TEST_GLM_CUDA") != "1"
            ? "Requires TS_TEST_GLM_CUDA=1 for the native GLM CUDA fixture."
            : TestGates.CudaSkip;
        if (Skip == null && minimumGpuCount > 1)
        {
            // GLM owns its CUDA backends independently of the process-global
            // GGML backend (which the test initializer normally pins to CPU).
            int available = TensorSharp.Cuda.CudaDevice.GetDeviceCount();
            if (available < minimumGpuCount)
                Skip = $"Requires {minimumGpuCount} visible CUDA devices; found {available}.";
        }
    }
}

[TraitDiscoverer("InferenceWeb.Tests.RequiresTraitDiscoverer", "InferenceWeb.Tests")]
public sealed class GlmSnapshotBoundaryTheoryAttribute : TheoryAttribute, ITraitAttribute
{
    public string RequiresValue => "NativeTestHooks";

    public GlmSnapshotBoundaryTheoryAttribute()
    {
        if (Environment.GetEnvironmentVariable("TS_TEST_GLM_SNAPSHOT_BOUNDARY") != "1")
            Skip = "Requires TS_TEST_GLM_SNAPSHOT_BOUNDARY=1 and a native library built with test hooks.";
    }
}

[TraitDiscoverer("InferenceWeb.Tests.RequiresTraitDiscoverer", "InferenceWeb.Tests")]
public sealed class GlmSnapshotBoundaryCudaTheoryAttribute : TheoryAttribute, ITraitAttribute
{
    public string RequiresValue => "Cuda";

    public GlmSnapshotBoundaryCudaTheoryAttribute()
    {
        Skip = Environment.GetEnvironmentVariable("TS_TEST_GLM_SNAPSHOT_BOUNDARY") != "1"
            ? "Requires TS_TEST_GLM_SNAPSHOT_BOUNDARY=1 and a native library built with test hooks."
            : Environment.GetEnvironmentVariable("TS_TEST_GLM_CUDA") != "1"
                ? "Requires TS_TEST_GLM_CUDA=1 for the native GLM CUDA fixture."
                : TestGates.CudaSkip;
    }
}
