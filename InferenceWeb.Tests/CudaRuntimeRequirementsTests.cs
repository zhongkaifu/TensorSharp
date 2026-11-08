// Copyright (c) Zhongkai Fu. All rights reserved.
// Licensed under the BSD-3-Clause license in the repository root.
//
// TensorSharp runs only on CUDA 13: a CUDA 13 driver (R580+), a Turing or newer GPU
// (compute capability 7.5+) and the CUDA 13 runtime libraries. These checks drive the
// decisions with injected version numbers and paths, so they need no GPU, no driver
// and no particular toolkit layout on the machine running them.
using System.Reflection;
using System.Runtime.InteropServices;
using System.Text;
using TensorSharp.Cuda;
using TensorSharp.Cuda.Interop;

namespace InferenceWeb.Tests;

public sealed class CudaRuntimeRequirementsTests
{
    [Theory]
    [InlineData(13000, 7, 5)]
    [InlineData(13020, 8, 6)]
    [InlineData(13050, 12, 1)]
    [InlineData(14000, 10, 0)]
    public void Cuda13DriverWithATuringOrNewerGpuIsSupported(int driverVersion, int ccMajor, int ccMinor)
    {
        Assert.Null(CudaRequirements.DescribeUnsupported(driverVersion, ccMajor, ccMinor));
    }

    [Theory]
    [InlineData(12080, "12.8")]
    [InlineData(12090, "12.9")]
    [InlineData(11080, "11.8")]
    [InlineData(12000, "12.0")]
    public void APreCuda13DriverIsRefusedAndNamed(int driverVersion, string shown)
    {
        string reason = CudaRequirements.DescribeUnsupported(driverVersion, 8, 6);
        Assert.NotNull(reason);
        Assert.Contains("needs a CUDA 13 driver (R580 or newer)", reason);
        Assert.Contains("this driver is CUDA " + shown, reason);
        Assert.Contains("CUDA 12 is not supported", reason);
        Assert.Equal(reason, CudaRequirements.DescribeUnsupportedDriver(driverVersion));
    }

    [Theory]
    [InlineData(5, 2)]
    [InlineData(6, 1)]
    [InlineData(7, 0)]
    [InlineData(7, 2)]
    public void APreTuringGpuIsRefusedAndNamed(int ccMajor, int ccMinor)
    {
        string reason = CudaRequirements.DescribeUnsupported(13050, ccMajor, ccMinor, "Tesla P40");
        Assert.NotNull(reason);
        Assert.Contains("needs compute capability 7.5+ (Turing or newer)", reason);
        Assert.Contains($"Tesla P40 is {ccMajor}.{ccMinor}", reason);
        Assert.Contains("CUDA 13 dropped Maxwell, Pascal and Volta", reason);
        Assert.Contains("this GPU is", CudaRequirements.DescribeUnsupportedDevice(ccMajor, ccMinor));
    }

    [Fact]
    public void AnOldDriverIsReportedBeforeAnOldGpu()
    {
        Assert.Contains("CUDA 13 driver", CudaRequirements.DescribeUnsupported(12080, 6, 1));
    }

    [Theory]
    [InlineData(13050, "13.5")]
    [InlineData(13000, "13.0")]
    [InlineData(12080, "12.8")]
    public void DriverVersionsFormatAsMajorDotMinor(int driverVersion, string expected)
    {
        Assert.Equal(expected, CudaRequirements.FormatDriverVersion(driverVersion));
    }

    [Fact]
    public void ATooNewPtxIsaNamesTheToolkitAndTheDriver()
    {
        // CUDA error 222: a 13.4-built PTX on a 13.2 driver (the A40 VMs' 595.91).
        string reason = CudaRequirements.DescribeModuleLoadFailure(222, "13.4", "9.4", 75, 13020, 8, 6);
        Assert.Contains("PTX ISA 9.4 (built by CUDA 13.4) needs a CUDA 13.4+ driver", reason);
        Assert.Contains("this driver is CUDA 13.2", reason);
    }

    [Fact]
    public void APtxTargetAboveTheGpuIsNamed()
    {
        string reason = CudaRequirements.DescribeModuleLoadFailure(218, "13.4", "9.4", 90, 13050, 8, 6);
        Assert.Contains("no code for sm_86", reason);
        Assert.Contains("the PTX targets sm_90", reason);
    }

    [Theory]
    [InlineData(209)]
    [InlineData(218)]
    public void APreTuringGpuIsNamedWhenNothingLoads(int error)
    {
        string reason = CudaRequirements.DescribeModuleLoadFailure(error, "13.0", "9.0", 75, 13000, 6, 1);
        Assert.Contains("no code for sm_61", reason);
        Assert.Contains("older than CUDA 13 supports", reason);
    }

    [Fact]
    public void APtxThatDoesNotCompileForItsOwnTargetIsStillExplained()
    {
        // The pre-gate q35 module: compute_75 PTX whose cp.async needs sm_80 fails JIT with 218.
        string reason = CudaRequirements.DescribeModuleLoadFailure(218, "13.4", "9.4", 75, 13050, 8, 6);
        Assert.Contains("no code for sm_86", reason);
        Assert.Contains("could not compile this PTX (sm_75)", reason);
    }

    [Theory]
    [InlineData(221)]
    [InlineData(223)]
    public void ADisabledJitAsksForSass(int error)
    {
        string reason = CudaRequirements.DescribeModuleLoadFailure(error, "13.0", "9.0", 75, 13050, 8, 9);
        Assert.Contains("cannot JIT-compile PTX", reason);
        Assert.Contains("SASS for sm_89", reason);
    }

    [Theory]
    [InlineData(2)]     // out of memory
    [InlineData(200)]   // invalid image
    public void OtherLoadErrorsAreNotGivenAToolchainDiagnosis(int error)
    {
        Assert.Null(CudaRequirements.DescribeModuleLoadFailure(error, "13.0", "9.0", 75, 13050, 8, 6));
    }

    [Theory]
    [InlineData("//\r\n// Cuda compilation tools, release 13.0, V13.0.88\r\n// Based on NVVM 7.0.1\r\n//\r\n\r\n.version 9.0\r\n.target sm_75\r\n", "13.0", "9.0")]
    [InlineData("//\n// Cuda compilation tools, release 13.4, V13.4.92\n//\n\n.version 9.4\n.target sm_86\n", "13.4", "9.4")]
    [InlineData(".version 8.7\n.target sm_61\n", null, "8.7")]
    [InlineData("// .version 9.0 in a comment\n", null, null)]
    public void PtxHeadersYieldTheirToolkitAndIsa(string ptx, string? toolkit, string? isa)
    {
        var header = CudaRequirements.ParsePtxHeader(Encoding.ASCII.GetBytes(ptx));
        Assert.Equal(toolkit, header.Toolkit);
        Assert.Equal(isa, header.Isa);
    }
}

public sealed class CudaLibraryResolverTests
{
    [Fact]
    public void OnlyCuBlas13IsEverLoaded()
    {
        Assert.Equal(new[] { "cublas64_13.dll" }, CudaLibraryResolver.GetCublasCandidates(windows: true));
        Assert.Equal(new[] { "libcublas.so.13" }, CudaLibraryResolver.GetCublasCandidates(windows: false));
    }

    [Fact]
    public void OnlyCuda13BinX64DirectoriesAreAddedNewestFirst()
    {
        string root = Path.Combine(Path.GetTempPath(), "nv", "CUDA");
        string Toolkit(string name) => Path.Combine(root, name);
        string[] installed = { Toolkit("v12.6"), Toolkit("v13.4"), Toolkit("v11.8"), Toolkit("v13.10"), Toolkit("v13.0"), Toolkit("vNext"), Toolkit("extras") };
        string cudaPath = Toolkit("v13.4");

        string[] actual = CudaLibraryResolver.EnumerateCudaBinDirectories(cudaPath, null, installed, "x64").ToArray();

        string[] expected =
        {
            Path.Combine(cudaPath, "bin", "x64"),
            Path.Combine(Toolkit("v13.10"), "bin", "x64"),   // parsed version, not ordinal order
            Path.Combine(Toolkit("v13.4"), "bin", "x64"),
            Path.Combine(Toolkit("v13.0"), "bin", "x64"),
        };
        Assert.Equal(expected, actual);
        Assert.DoesNotContain(actual, path => path.EndsWith(Path.DirectorySeparatorChar + "bin", StringComparison.Ordinal));
    }

    [Fact]
    public void OnWindowsOnArmTheArm64RuntimeFolderIsProbed()
    {
        // NVIDIA ships CUDA 13 for Windows on Arm as redistributables whose DLLs sit in bin\arm64
        // (eng/install-cuda-redist.py lays them out the same way); CUDA_PATH alone must find them.
        string cudaPath = Path.Combine(Path.GetTempPath(), "cuda-13.4.3-arm64");

        Assert.Equal(new[] { Path.Combine(cudaPath, "bin", "arm64") },
            CudaLibraryResolver.EnumerateCudaBinDirectories(cudaPath, null, Array.Empty<string>(), "arm64").ToArray());
    }

    [Fact]
    public void TheProbedRuntimeFolderFollowsTheProcessArchitecture()
    {
        string expected = RuntimeInformation.ProcessArchitecture == Architecture.Arm64 ? "arm64" : "x64";
        Assert.Equal(expected, CudaLibraryResolver.WindowsCudaBinArchitecture);

        string cudaPath = Path.Combine(Path.GetTempPath(), "cuda");
        Assert.Equal(new[] { Path.Combine(cudaPath, "bin", expected) },
            CudaLibraryResolver.EnumerateCudaBinDirectories(cudaPath, null, Array.Empty<string>()).ToArray());
    }

    private static readonly Type GgmlNative =
        typeof(TensorSharp.GGML.GgmlBasicOps).Assembly.GetType("TensorSharp.GGML.GgmlNative", throwOnError: true)!;

    private static string[] GgmlDependencyDirectories(string? cudaPath, string? cudaHome, string[] installed, string? binArchitecture)
    {
        Type[] signature = binArchitecture == null
            ? new[] { typeof(string), typeof(string), typeof(IEnumerable<string>) }
            : new[] { typeof(string), typeof(string), typeof(IEnumerable<string>), typeof(string) };
        MethodInfo enumerate = GgmlNative.GetMethod("EnumerateWindowsNativeDependencyDirectories",
            BindingFlags.NonPublic | BindingFlags.Static, signature)!;
        Assert.NotNull(enumerate);
        object?[] args = binArchitecture == null
            ? new object?[] { cudaPath, cudaHome, installed }
            : new object?[] { cudaPath, cudaHome, installed, binArchitecture };
        return ((IEnumerable<string>)enumerate.Invoke(null, args)!).ToArray();
    }

    [Fact]
    public void TheGgmlBridgeSearchesTheSameCuda13Directories()
    {
        // GgmlOps (cublas64_13.dll) and cuDNN (cublasLt64_13.dll, nvrtc64_130_0.dll) resolve the
        // runtime through TensorSharp.Backends.GGML's own copy of this enumeration.
        string root = Path.Combine(Path.GetTempPath(), "nv", "CUDA");
        string[] installed = { Path.Combine(root, "v12.6"), Path.Combine(root, "v13.4"), Path.Combine(root, "v13.10") };

        foreach (string arch in new[] { "x64", "arm64" })
        {
            Assert.Equal(new[]
            {
                Path.Combine(root, "v13.4", "bin", arch),
                Path.Combine(root, "v13.10", "bin", arch),
                Path.Combine(root, "v13.4", "bin", arch),
            }, GgmlDependencyDirectories(null, Path.Combine(root, "v13.4"), installed, arch));
        }

        Assert.Equal(CudaLibraryResolver.WindowsCudaBinArchitecture, GgmlNative
            .GetProperty("WindowsCudaBinArchitecture", BindingFlags.NonPublic | BindingFlags.Static)!.GetValue(null));
        Assert.Equal(new[] { Path.Combine(root, "v13.4", "bin", CudaLibraryResolver.WindowsCudaBinArchitecture) },
            GgmlDependencyDirectories(Path.Combine(root, "v13.4"), null, Array.Empty<string>(), null));
    }

    private static string DescribeBridgeLoadFailure(string? path, Exception? error, bool windows, bool linux)
    {
        MethodInfo describe = GgmlNative.GetMethod("DescribeBridgeLoadFailure", BindingFlags.NonPublic | BindingFlags.Static)!;
        Assert.NotNull(describe);
        return (string)describe.Invoke(null, new object?[] { path, error, windows, linux })!;
    }

    [Fact]
    public void AnUnbuiltBridgeStillSaysToBuildIt()
    {
        Assert.Equal("Failed to load the native GGML bridge. Build `TensorSharp.GGML.Native` first.",
            DescribeBridgeLoadFailure(null, null, windows: true, linux: false));
    }

    [Fact]
    public void ABuiltBridgeThatDoesNotLoadNamesTheFileTheReasonAndTheCudaRuntime()
    {
        // GgmlOps.dll present but cublas64_13.dll not on the search path: Windows reports only
        // "The specified module could not be found", which used to read as "build it first".
        string path = Path.Combine(Path.GetTempPath(), "bin", "GgmlOps.dll");
        var error = new DllNotFoundException("The specified module could not be found. (0x8007007E)");

        string windows = DescribeBridgeLoadFailure(path, error, windows: true, linux: false);
        Assert.Contains(path, windows);
        Assert.Contains("exists but could not be loaded", windows);
        Assert.Contains("0x8007007E", windows);
        Assert.Contains("cublas64_13.dll", windows);
        Assert.Contains("CUDA_PATH", windows);
        Assert.Contains("bin\\arm64", windows);
        Assert.DoesNotContain("Build `TensorSharp.GGML.Native` first", windows);

        string linux = DescribeBridgeLoadFailure("/opt/ts/libGgmlOps.so",
            new DllNotFoundException("libcublas.so.13: cannot open shared object file"), windows: false, linux: true);
        Assert.Contains("libcublas.so.13: cannot open shared object file", linux);
        Assert.Contains("LD_LIBRARY_PATH", linux);
    }

    [Fact]
    public void ABridgeForAnotherArchitectureSaysSo()
    {
        string message = DescribeBridgeLoadFailure(Path.Combine(Path.GetTempPath(), "GgmlOps.dll"),
            new BadImageFormatException("An attempt was made to load a program with an incorrect format."), windows: true, linux: false);
        Assert.Contains("different architecture", message);
        Assert.Contains(RuntimeInformation.ProcessArchitecture.ToString(), message);
    }

    [Fact]
    public void TheResolverKeepsTheSignatureTheGb10ReleaseProbeReflectsOn()
    {
        // eng/tests/gb10-release-probe/StartupHook.cs invokes Resolve("cublas", assembly, null).
        MethodInfo? resolve = typeof(CudaLibraryResolver).GetMethod("Resolve", BindingFlags.NonPublic | BindingFlags.Static);
        Assert.NotNull(resolve);
        Assert.Equal(new[] { typeof(string), typeof(Assembly), typeof(DllImportSearchPath?) },
            resolve!.GetParameters().Select(p => p.ParameterType).ToArray());
        Assert.Equal(typeof(IntPtr), resolve.ReturnType);
    }
}
