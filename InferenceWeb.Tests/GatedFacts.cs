using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.IO;
using System.Linq;
using System.Runtime.InteropServices;
using Xunit;
using Xunit.Abstractions;
using Xunit.Sdk;

namespace InferenceWeb.Tests
{
    /// <summary>
    /// Environment probes shared by the gated fact attributes below. Probe
    /// results are cached: attributes run at discovery time for every test.
    /// </summary>
    internal static class TestGates
    {
        // The probes are banned from test code (BannedSymbols.txt) so gating
        // can't silently regress to in-test checks; this is the one caller.
#pragma warning disable RS0030
        private static readonly Lazy<bool> CudaAvailable = new(() =>
        {
            if (Environment.GetEnvironmentVariable("TS_TEACHER_TOKEN_EXPORT") == "1") return false;
            try { return TensorSharp.Cuda.CudaBackend.IsAvailable(); }
            catch { return false; }
        });

        private static readonly Lazy<bool> MlxAvailable = new(() =>
        {
            if (Environment.GetEnvironmentVariable("TS_TEACHER_TOKEN_EXPORT") == "1") return false;
            try { return TensorSharp.MLX.MlxBackend.IsAvailable(); }
            catch { return false; }
        });
#pragma warning restore RS0030

        public static string CudaSkip =>
            CudaAvailable.Value ? null : "Requires a CUDA device.";

        public static string MlxSkip =>
            MlxAvailable.Value ? null : "Requires the MLX native backend.";

        // Video tests synthesise their own clip rather than carry a binary fixture, so
        // they need an OpenCV build that can ENCODE. The slim runtimes ship videoio
        // without every encoder; probe once by actually writing a tiny clip.
        private static readonly Lazy<bool> VideoWritable = new(() =>
        {
            if (Environment.GetEnvironmentVariable("TS_TEACHER_TOKEN_EXPORT") == "1") return false;
            string dir = Path.Combine(Path.GetTempPath(), "ts-video-gate-" + Guid.NewGuid().ToString("N"));
            try { return VideoFixture.TryWrite(Path.Combine(dir, "probe.mp4"), frames: 4) != null; }
            catch { return false; }
            finally { try { Directory.Delete(dir, recursive: true); } catch { /* best effort */ } }
        });

        public static string VideoSkip =>
            VideoWritable.Value ? null : "Requires an OpenCV build that can encode video.";

        /// <summary>
        /// The GGML backend <see cref="GgmlBackendTestInitializer"/> pins for this
        /// process, from <c>TS_TEST_GGML_BACKEND</c> (default cpu). One place, so the
        /// initializer, the gates below and tests that follow the pin agree.
        /// </summary>
        public static TensorSharp.GGML.GgmlBackendType PinnedGgmlBackendType =>
            (Environment.GetEnvironmentVariable("TS_TEST_GGML_BACKEND") ?? "cpu").Trim().ToLowerInvariant() switch
            {
                "metal" => TensorSharp.GGML.GgmlBackendType.Metal,
                "cuda" => TensorSharp.GGML.GgmlBackendType.Cuda,
                "vulkan" => TensorSharp.GGML.GgmlBackendType.Vulkan,
                _ => TensorSharp.GGML.GgmlBackendType.Cpu,
            };

        /// <summary>The pinned GGML backend as the model-level <see cref="BackendType"/>.</summary>
        public static BackendType PinnedGgmlBackend => PinnedGgmlBackendType switch
        {
            TensorSharp.GGML.GgmlBackendType.Metal => BackendType.GgmlMetal,
            TensorSharp.GGML.GgmlBackendType.Cuda => BackendType.GgmlCuda,
            TensorSharp.GGML.GgmlBackendType.Vulkan => BackendType.GgmlVulkan,
            _ => BackendType.GgmlCpu,
        };

        /// <summary>
        /// Skip reason for a test that constructs <paramref name="required"/> in a
        /// process pinned to another GGML backend, or null to run. The native bridge
        /// allows one GGML backend per process, so such a test can only ever fail
        /// with "A different GGML backend was already initialized"; it belongs to
        /// the lane that pins its backend. Non-GGML backends never conflict.
        /// </summary>
        public static string GgmlPinSkip(BackendType required)
        {
            if (required is not (BackendType.GgmlCpu or BackendType.GgmlMetal or BackendType.GgmlCuda or BackendType.GgmlVulkan))
                return null;
            BackendType pinned = PinnedGgmlBackend;
            if (required == pinned)
                return null;
            string name = required switch
            {
                BackendType.GgmlMetal => "metal",
                BackendType.GgmlCuda => "cuda",
                BackendType.GgmlVulkan => "vulkan",
                _ => "cpu",
            };
            return $"Requires TS_TEST_GGML_BACKEND={name}: this test constructs {required}, and this process pins {pinned} (one GGML backend per process).";
        }

        /// <summary>
        /// Skip reason for weight-gated tests, or null to run. The env var may
        /// name a file or a directory; with <paramref name="ggufContains"/> the
        /// directory must hold a matching GGUF (see <see cref="FindGguf"/>).
        /// </summary>
        public static string ModelSkip(string envVar, string ggufContains = null)
        {
            string value = Environment.GetEnvironmentVariable(envVar);
            if (string.IsNullOrEmpty(value))
                return $"Requires model weights ({envVar} not set).";
            if (File.Exists(value))
                return null;
            if (!Directory.Exists(value))
                return $"Requires model weights ({envVar} points to a missing path).";
            if (ggufContains != null && FindGguf(value, ggufContains) == null)
                return $"Requires model weights (no '*{ggufContains}*' GGUF under {envVar}).";
            return null;
        }

        /// <summary>
        /// File name of the GGML native library on this OS, for tests that pin the
        /// mapped module's identity (Windows loads GgmlOps.dll; Linux and macOS load
        /// the lib-prefixed .so/.dylib).
        /// </summary>
        public static string NativeGgmlOpsFileName =>
            OperatingSystem.IsWindows() ? "GgmlOps.dll"
            : OperatingSystem.IsMacOS() ? "libGgmlOps.dylib"
            : "libGgmlOps.so";

        public static bool IsNativeGgmlOps(string path)
            => string.Equals(Path.GetFileName(path), NativeGgmlOpsFileName, StringComparison.OrdinalIgnoreCase);

        /// <summary>
        /// Identify the actual loaded library after native execution. macOS
        /// Process.Modules omits dlopen-loaded libraries, so query dyld's mapped
        /// image table there. Never substitute an unobserved on-disk candidate.
        /// </summary>
        public static string MappedNativeGgmlOpsPath()
        {
            IEnumerable<string> paths;
            if (OperatingSystem.IsMacOS())
                paths = MacMappedImagePaths();
            else
            {
                using var process = Process.GetCurrentProcess();
                paths = process.Modules.Cast<ProcessModule>().Select(module => module.FileName).ToArray();
            }
            string[] matches = paths.Where(IsNativeGgmlOps).Distinct(StringComparer.Ordinal).ToArray();
            if (matches.Length != 1)
                throw new InvalidOperationException($"Expected one mapped {NativeGgmlOpsFileName}; observed {matches.Length}: {string.Join(", ", matches)}");
            return matches[0];
        }

        private static IEnumerable<string> MacMappedImagePaths()
        {
            uint count = DyldImageCount();
            for (uint index = 0; index < count; ++index)
            {
                string path = Marshal.PtrToStringUTF8(DyldGetImageName(index));
                if (!string.IsNullOrEmpty(path)) yield return path;
            }
        }

        [DllImport("/usr/lib/libSystem.B.dylib", EntryPoint = "_dyld_image_count")]
        private static extern uint DyldImageCount();

        [DllImport("/usr/lib/libSystem.B.dylib", EntryPoint = "_dyld_get_image_name")]
        private static extern IntPtr DyldGetImageName(uint index);

        /// <summary>
        /// An explicit model file, or the first GGUF in <paramref name="dir"/> whose name contains
        /// <paramref name="contains"/> (case-insensitive; '|' separates
        /// accepted alternatives, e.g. "gpt-oss|gpt_oss"), skipping companion
        /// files (mmproj / assistant drafts). Shared by the attributes and the
        /// per-class loaders so the skip decision and the load pick the same file.
        /// </summary>
        public static string FindGguf(string dir, string contains)
            => MatchingGgufs(dir, contains).FirstOrDefault();

        /// <summary>
        /// Same match rule as <see cref="FindGguf"/>, smallest file first. For a
        /// test that only needs SOME model of a family, the smallest quant loads
        /// fastest and leaves the most room beside it — which matters on a Mac
        /// where a 27B Q8_0 is 27 GB against a 40 GB Metal working set. The
        /// predicate is deliberately shared: the gate and the loader may disagree
        /// about WHICH match to take, never about what counts as a match.
        /// </summary>
        public static string FindSmallestGguf(string dir, string contains)
            => MatchingGgufs(dir, contains)
                .OrderBy(p => new FileInfo(p).Length)
                .FirstOrDefault();

        private static IEnumerable<string> MatchingGgufs(string dir, string contains)
        {
            // ModelSkip accepts an explicit file without applying the directory
            // name filter. Keep the loader aligned with that discovery contract.
            if (File.Exists(dir)) return new[] { dir };
            string[] alternatives = contains.ToLowerInvariant().Split('|');
            return Directory.GetFiles(dir, "*.gguf").Where(p =>
            {
                string n = Path.GetFileName(p).ToLowerInvariant();
                return alternatives.Any(n.Contains)
                    && !n.Contains("mmproj") && !n.Contains("assistant");
            });
        }
    }

    /// <summary>
    /// Emits the Requires trait(s) declared by a gated fact attribute, so
    /// --filter "Requires!=..." lanes work without a separate [Trait] line.
    /// </summary>
    public sealed class RequiresTraitDiscoverer : ITraitDiscoverer
    {
        public IEnumerable<KeyValuePair<string, string>> GetTraits(IAttributeInfo traitAttribute)
        {
            foreach (string value in traitAttribute.GetNamedArgument<string>("RequiresValue").Split(','))
                yield return new KeyValuePair<string, string>("Requires", value);
        }
    }

    /// <summary>
    /// [Fact] that needs a CUDA device: skips visibly when none is present and
    /// carries Requires=Cuda. Passing a model env var adds the Requires=Models
    /// gate on the same test.
    /// </summary>
    [TraitDiscoverer("InferenceWeb.Tests.RequiresTraitDiscoverer", "InferenceWeb.Tests")]
    [AttributeUsage(AttributeTargets.Method)]
    public sealed class CudaFactAttribute : FactAttribute, ITraitAttribute
    {
        public string RequiresValue { get; }

        public CudaFactAttribute(string modelEnvVar = null, string ggufContains = null)
        {
            RequiresValue = modelEnvVar == null ? "Cuda" : "Cuda,Models";
            Skip = TestGates.CudaSkip
                ?? (modelEnvVar == null ? null : TestGates.ModelSkip(modelEnvVar, ggufContains));
        }

        /// <summary>The GGML backend the test constructs; skips unless the process pins it (<see cref="TestGates.GgmlPinSkip"/>).</summary>
        public BackendType GgmlBackend
        {
            get => _ggmlBackend;
            set { _ggmlBackend = value; Skip ??= TestGates.GgmlPinSkip(value); }
        }
        private BackendType _ggmlBackend;
    }

    /// <summary>[Theory] variant of <see cref="CudaFactAttribute"/>.</summary>
    [TraitDiscoverer("InferenceWeb.Tests.RequiresTraitDiscoverer", "InferenceWeb.Tests")]
    [AttributeUsage(AttributeTargets.Method)]
    public sealed class CudaTheoryAttribute : TheoryAttribute, ITraitAttribute
    {
        public string RequiresValue { get; }

        public CudaTheoryAttribute(string modelEnvVar = null, string ggufContains = null)
        {
            RequiresValue = modelEnvVar == null ? "Cuda" : "Cuda,Models";
            Skip = TestGates.CudaSkip
                ?? (modelEnvVar == null ? null : TestGates.ModelSkip(modelEnvVar, ggufContains));
        }
    }

    /// <summary>
    /// [Fact] that needs the MLX native backend: skips visibly when it is
    /// absent and carries Requires=Mlx.
    /// </summary>
    [TraitDiscoverer("InferenceWeb.Tests.RequiresTraitDiscoverer", "InferenceWeb.Tests")]
    [AttributeUsage(AttributeTargets.Method)]
    public sealed class MlxFactAttribute : FactAttribute, ITraitAttribute
    {
        public string RequiresValue => "Mlx";

        public MlxFactAttribute() => Skip = TestGates.MlxSkip;
    }

    /// <summary>[Theory] variant of <see cref="MlxFactAttribute"/>.</summary>
    [TraitDiscoverer("InferenceWeb.Tests.RequiresTraitDiscoverer", "InferenceWeb.Tests")]
    [AttributeUsage(AttributeTargets.Method)]
    public sealed class MlxTheoryAttribute : TheoryAttribute, ITraitAttribute
    {
        public string RequiresValue => "Mlx";

        public MlxTheoryAttribute() => Skip = TestGates.MlxSkip;
    }

    /// <summary>
    /// [Fact] that needs to synthesise a video clip: skips visibly when this
    /// machine's OpenCV cannot encode one, and carries Requires=Video.
    /// </summary>
    [TraitDiscoverer("InferenceWeb.Tests.RequiresTraitDiscoverer", "InferenceWeb.Tests")]
    [AttributeUsage(AttributeTargets.Method)]
    public sealed class VideoFactAttribute : FactAttribute, ITraitAttribute
    {
        public string RequiresValue => "Video";

        public VideoFactAttribute() => Skip = TestGates.VideoSkip;
    }

    /// <summary>[Theory] variant of <see cref="VideoFactAttribute"/>.</summary>
    [TraitDiscoverer("InferenceWeb.Tests.RequiresTraitDiscoverer", "InferenceWeb.Tests")]
    [AttributeUsage(AttributeTargets.Method)]
    public sealed class VideoTheoryAttribute : TheoryAttribute, ITraitAttribute
    {
        public string RequiresValue => "Video";

        public VideoTheoryAttribute() => Skip = TestGates.VideoSkip;
    }

    /// <summary>
    /// [Fact] that needs real GGUF weights: skips visibly when the env var is
    /// unset or the weights are absent, and carries Requires=Models.
    /// </summary>
    [TraitDiscoverer("InferenceWeb.Tests.RequiresTraitDiscoverer", "InferenceWeb.Tests")]
    [AttributeUsage(AttributeTargets.Method)]
    public sealed class ModelFactAttribute : FactAttribute, ITraitAttribute
    {
        public string RequiresValue => "Models";

        public ModelFactAttribute(string envVar, string ggufContains = null)
            => Skip = TestGates.ModelSkip(envVar, ggufContains);

        /// <summary>The GGML backend the test constructs; skips unless the process pins it (<see cref="TestGates.GgmlPinSkip"/>).</summary>
        public BackendType GgmlBackend
        {
            get => _ggmlBackend;
            set { _ggmlBackend = value; Skip ??= TestGates.GgmlPinSkip(value); }
        }
        private BackendType _ggmlBackend;
    }

    /// <summary>[Theory] variant of <see cref="ModelFactAttribute"/>.</summary>
    [TraitDiscoverer("InferenceWeb.Tests.RequiresTraitDiscoverer", "InferenceWeb.Tests")]
    [AttributeUsage(AttributeTargets.Method)]
    public sealed class ModelTheoryAttribute : TheoryAttribute, ITraitAttribute
    {
        public string RequiresValue => "Models";

        public ModelTheoryAttribute(string envVar, string ggufContains = null)
            => Skip = TestGates.ModelSkip(envVar, ggufContains);

        /// <summary>The GGML backend the test constructs; skips unless the process pins it (<see cref="TestGates.GgmlPinSkip"/>).</summary>
        public BackendType GgmlBackend
        {
            get => _ggmlBackend;
            set { _ggmlBackend = value; Skip ??= TestGates.GgmlPinSkip(value); }
        }
        private BackendType _ggmlBackend;
    }
}
