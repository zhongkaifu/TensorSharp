using Xunit.Sdk;

namespace InferenceWeb.Tests;

[TraitDiscoverer("InferenceWeb.Tests.RequiresTraitDiscoverer", "InferenceWeb.Tests")]
public sealed class GlmNativeCudaFactAttribute : FactAttribute, ITraitAttribute
{
    public string RequiresValue => "Cuda";

    public GlmNativeCudaFactAttribute()
    {
        Skip = Environment.GetEnvironmentVariable("TS_TEST_GLM_CUDA") != "1"
            ? "Requires TS_TEST_GLM_CUDA=1 for the native GLM CUDA fixture."
            : TestGates.CudaSkip;
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
