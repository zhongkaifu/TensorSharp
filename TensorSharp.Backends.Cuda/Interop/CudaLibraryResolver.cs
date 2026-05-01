using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Reflection;
using System.Runtime.InteropServices;
using System.Threading;

namespace TensorSharp.Cuda.Interop
{
    internal static class CudaLibraryResolver
    {
        private static int registered;

        public static void Register()
        {
            if (Interlocked.Exchange(ref registered, 1) != 0)
                return;

            NativeLibrary.SetDllImportResolver(typeof(CudaLibraryResolver).Assembly, Resolve);
            EnsureWindowsCudaPath();
        }

        private static IntPtr Resolve(string libraryName, Assembly assembly, DllImportSearchPath? searchPath)
        {
            if (libraryName == "cuda")
            {
                string driverName = RuntimeInformation.IsOSPlatform(OSPlatform.Windows) ? "nvcuda.dll" : "libcuda.so.1";
                if (NativeLibrary.TryLoad(driverName, out IntPtr cudaHandle))
                    return cudaHandle;
            }

            if (libraryName == "cublas")
            {
                foreach (string candidate in GetCublasCandidates())
                {
                    if (NativeLibrary.TryLoad(candidate, out IntPtr cublasHandle))
                        return cublasHandle;
                }
            }

            return IntPtr.Zero;
        }

        private static IEnumerable<string> GetCublasCandidates()
        {
            if (RuntimeInformation.IsOSPlatform(OSPlatform.Windows))
            {
                yield return "cublas64_13.dll";
                yield return "cublas64_12.dll";
                yield return "cublas64_11.dll";
                yield break;
            }

            // Prefer versioned runtime libraries; unversioned symlinks can point
            // at stubs or a different toolkit than the native GGML bridge uses.
            yield return "libcublas.so.12";
            yield return "libcublas.so.13";
            yield return "libcublas.so.11";
            yield return "libcublas.so";
        }

        private static void EnsureWindowsCudaPath()
        {
            if (!RuntimeInformation.IsOSPlatform(OSPlatform.Windows))
                return;

            string currentPath = Environment.GetEnvironmentVariable("PATH") ?? string.Empty;
            var existing = new HashSet<string>(
                currentPath.Split(Path.PathSeparator, StringSplitOptions.RemoveEmptyEntries),
                StringComparer.OrdinalIgnoreCase);

            string[] additions = EnumerateCudaBinDirectories()
                .Where(path => Directory.Exists(path) && !existing.Contains(path))
                .Distinct(StringComparer.OrdinalIgnoreCase)
                .ToArray();

            if (additions.Length == 0)
                return;

            Environment.SetEnvironmentVariable("PATH", string.Join(Path.PathSeparator, additions.Concat(new[] { currentPath })));
        }

        private static IEnumerable<string> EnumerateCudaBinDirectories()
        {
            foreach (string variableName in new[] { "CUDA_PATH", "CUDA_HOME" })
            {
                string root = Environment.GetEnvironmentVariable(variableName);
                if (!string.IsNullOrWhiteSpace(root))
                    yield return Path.Combine(root, "bin");
            }

            string programFiles = Environment.GetFolderPath(Environment.SpecialFolder.ProgramFiles);
            string cudaRoot = Path.Combine(programFiles, "NVIDIA GPU Computing Toolkit", "CUDA");
            if (!Directory.Exists(cudaRoot))
                yield break;

            foreach (string versionDir in Directory.EnumerateDirectories(cudaRoot, "v*").OrderByDescending(path => path))
                yield return Path.Combine(versionDir, "bin");
        }
    }
}
