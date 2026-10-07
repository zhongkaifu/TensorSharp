using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Reflection;
using System.Runtime.InteropServices;

namespace TensorSharp.Cuda.Interop
{
    internal static class CudaLibraryResolver
    {
        // Concurrent first callers must wait for installation to complete before
        // entering P/Invoke; publishing a flag before SetDllImportResolver races.
        private static readonly Lazy<bool> registration = new Lazy<bool>(() =>
        {
            NativeLibrary.SetDllImportResolver(typeof(CudaLibraryResolver).Assembly, Resolve);
            EnsureWindowsCudaPath();
            return true;
        });

        public static void Register() => _ = registration.Value;

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
                    // Include the assembly directory when loading bundled libraries on Linux.
                    if (NativeLibrary.TryLoad(candidate, assembly, searchPath, out IntPtr cublasHandle))
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
