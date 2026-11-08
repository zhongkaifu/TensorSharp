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

        private static IEnumerable<string> GetCublasCandidates() =>
            GetCublasCandidates(RuntimeInformation.IsOSPlatform(OSPlatform.Windows));

        /// <summary>
        /// cuBLAS 13 only. TensorSharp is built against CUDA 13, so an older cuBLAS (or an
        /// unversioned libcublas.so, which can point at one or at a stub) must never be picked
        /// up: a missing CUDA 13 runtime fails as DllNotFoundException instead of mixing majors.
        /// </summary>
        internal static IEnumerable<string> GetCublasCandidates(bool windows)
        {
            yield return windows ? "cublas64_13.dll" : "libcublas.so.13";
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
            string programFiles = Environment.GetFolderPath(Environment.SpecialFolder.ProgramFiles);
            string cudaRoot = Path.Combine(programFiles, "NVIDIA GPU Computing Toolkit", "CUDA");
            return EnumerateCudaBinDirectories(
                Environment.GetEnvironmentVariable("CUDA_PATH"),
                Environment.GetEnvironmentVariable("CUDA_HOME"),
                Directory.Exists(cudaRoot) ? Directory.EnumerateDirectories(cudaRoot, "v*") : Array.Empty<string>());
        }

        /// <summary>
        /// The Windows directories that may hold the CUDA 13 runtime DLLs this process can load,
        /// newest first. CUDA 13 installs them in bin\x64 (its bin holds only tools), and NVIDIA's
        /// Windows on Arm redistributables in bin\arm64, so the one folder for this process's
        /// architecture is probed, and of the installed toolkit folders only v13 and newer.
        /// </summary>
        internal static IEnumerable<string> EnumerateCudaBinDirectories(
            string cudaPath, string cudaHome, IEnumerable<string> installedToolkitDirectories)
            => EnumerateCudaBinDirectories(cudaPath, cudaHome, installedToolkitDirectories, WindowsCudaBinArchitecture);

        internal static IEnumerable<string> EnumerateCudaBinDirectories(
            string cudaPath, string cudaHome, IEnumerable<string> installedToolkitDirectories, string binArchitecture)
        {
            foreach (string root in new[] { cudaPath, cudaHome })
            {
                if (!string.IsNullOrWhiteSpace(root))
                    yield return Path.Combine(root, "bin", binArchitecture);
            }

            foreach (string versionDir in OrderCuda13ToolkitDirectories(installedToolkitDirectories))
                yield return Path.Combine(versionDir, "bin", binArchitecture);
        }

        /// <summary>
        /// The bin subfolder whose DLLs this process can load: arm64 for an ARM64 process, x64
        /// otherwise (including an x64 process emulated on Windows on Arm).
        /// </summary>
        internal static string WindowsCudaBinArchitecture =>
            RuntimeInformation.ProcessArchitecture == Architecture.Arm64 ? "arm64" : "x64";

        /// <summary>
        /// The "vMAJOR.MINOR" toolkit folders of CUDA 13 or newer, newest first by parsed
        /// version (v13.10 sorts above v13.4, which an ordinal sort would get wrong).
        /// </summary>
        internal static IEnumerable<string> OrderCuda13ToolkitDirectories(IEnumerable<string> directories)
        {
            return directories
                .Select(path => (Dir: path, Version: ParseToolkitVersion(Path.GetFileName(path.TrimEnd('\\', '/')))))
                .Where(entry => entry.Version != null && entry.Version.Major >= 13)
                .OrderByDescending(entry => entry.Version)
                .Select(entry => entry.Dir);
        }

        private static Version ParseToolkitVersion(string folderName)
        {
            if (string.IsNullOrEmpty(folderName) || folderName.Length < 2 || char.ToLowerInvariant(folderName[0]) != 'v')
                return null;
            return Version.TryParse(folderName.Substring(1), out Version version) ? version : null;
        }
    }
}
