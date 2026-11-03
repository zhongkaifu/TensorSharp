using System;
using System.Collections.Generic;
using System.IO;
using TensorSharp.Cuda.Interop;

namespace TensorSharp.Cuda
{
    internal sealed partial class CudaModule : IDisposable
    {
        private readonly Dictionary<string, IntPtr> functions = new Dictionary<string, IntPtr>(StringComparer.Ordinal);
        private IntPtr module;

        // The .target of the PTX this module was loaded from or beside: only the fallback for
        // CompiledArchOf, since a fatbin's SASS is usually built for higher targets.
        private readonly int ptxTargetArch;

        private CudaModule(IntPtr module, int ptxTargetArch)
        {
            this.module = module;
            this.ptxTargetArch = ptxTargetArch;
        }

        /// <summary>
        /// The SM architecture the code the driver loaded for <paramref name="kernel"/> was
        /// compiled for, as an integer (75, 86, 120): the <c>__CUDA_ARCH__</c> its body saw,
        /// over 10. 0 when unknown. Kernels whose bodies are guarded by
        /// <c>__CUDA_ARCH__ &gt;= N</c> compile to a trap or an empty body below N, and a symbol
        /// lookup cannot tell: the committed portable baseline is compute_75 (the oldest target
        /// CUDA 13 compiles), so on an sm_80+ GPU its tensor-core kernels resolve fine and then
        /// trap or write nothing. Callers gate such kernels on this.
        /// It is read from the loaded code (CU_FUNC_ATTRIBUTE_PTX_VERSION), not from the PTX
        /// header: a fatbin is built beside a PTX of a lower target (the release archives pair
        /// compute_75 PTX with SASS for sm_75 to sm_120), and the driver runs the SASS for this
        /// GPU (86 on an sm_86 device) or, when there is none, JIT-compiles the fatbin's PTX
        /// (75). PTX loaded directly reports its own target. One module holds one loaded image,
        /// so any of its kernels answers for all of them.
        /// </summary>
        public int CompiledArchOf(string kernel)
        {
            int result = CudaDriverApi.cuFuncGetAttribute(
                out int arch, CudaDriverApi.CU_FUNC_ATTRIBUTE_PTX_VERSION, GetFunction(kernel));
            // The PTX target is a lower bound for anything loaded from or beside it.
            return result == 0 && arch > 0 ? arch : ptxTargetArch;
        }

        /// <summary>
        /// The <c>sm_NN</c> number of a PTX image's <c>.target</c> directive (the header
        /// nvcc writes before any code), or 0 when there is none.
        /// </summary>
        internal static int ParsePtxTargetArch(byte[] ptx)
        {
            if (ptx == null)
                return 0;
            // The directive sits in the first few lines; never scan a whole module.
            int length = Math.Min(ptx.Length, 64 * 1024);
            string head = System.Text.Encoding.ASCII.GetString(ptx, 0, length);
            const string directive = ".target sm_";
            int at = 0;
            while ((at = head.IndexOf(directive, at, StringComparison.Ordinal)) >= 0)
            {
                bool lineStart = at == 0 || head[at - 1] == '\n' || head[at - 1] == '\r';
                int digits = at + directive.Length;
                int end = digits;
                while (end < head.Length && char.IsAsciiDigit(head[end]))
                    end++;
                if (lineStart && end > digits && int.TryParse(head.AsSpan(digits, end - digits), out int arch))
                    return arch;
                at = digits;
            }
            return 0;
        }

        /// <summary>
        /// Load a kernel module. For a <c>.ptx</c> path, the <c>.fatbin</c> the same build
        /// produced beside it is preferred: its SASS loads on any driver of the toolkit's
        /// major version, while the driver can only JIT PTX from a toolkit no newer than
        /// itself (CUDA 13.4 PTX fails on a 13.2 driver). The fatbin is used only when its
        /// <c>.fatbin.sha256</c> sidecar names this PTX, so one left behind by an earlier
        /// build never shadows newer kernels; a fatbin the driver rejects (no SASS for this
        /// GPU) falls back to the PTX. When the PTX fails too, the exception says why: the PTX
        /// ISA against the driver's CUDA version, or the PTX target against the GPU.
        /// </summary>
        public static CudaModule LoadFromFile(string path) => LoadFromFile(path, identify: false);

        /// <summary>
        /// <see cref="LoadFromFile(string)"/>; with <paramref name="identify"/> the module also records the SHA-256 of
        /// the image the driver actually loaded (the fatbin, or the PTX it fell back to) in
        /// <see cref="ImageSha256"/>, which a distributed run compares across nodes.
        /// </summary>
        public static CudaModule LoadFromFile(string path, bool identify)
        {
            if (path == null)
                throw new ArgumentNullException(nameof(path));

            byte[] bytes = File.ReadAllBytes(path);
            byte[] fatbin = TryReadMatchingFatbin(path, bytes);
            CudaException fatbinFailure = null;
            if (fatbin != null)
            {
                try
                {
                    CudaModule loaded = LoadImage(fatbin, ParsePtxTargetArch(bytes));
                    if (identify)
                        loaded.ImageSha256 = Sha256(fatbin);
                    return loaded;
                }
                catch (CudaException ex)
                {
                    // No SASS for this GPU and PTX the driver cannot JIT; try the PTX file.
                    fatbinFailure = ex;
                }
            }
            CudaModule module = LoadFromBytes(bytes, fatbinFailure);
            if (identify)
                module.ImageSha256 = Sha256(bytes);
            return module;
        }

        /// <summary>The SHA-256 (hex) of the image this module was loaded from, when the loader was asked to
        /// identify it; null otherwise.</summary>
        public string ImageSha256 { get; private set; }

        private static string Sha256(byte[] image)
            => Convert.ToHexString(System.Security.Cryptography.SHA256.HashData(image));

        internal static byte[] TryReadMatchingFatbin(string ptxPath, byte[] ptxBytes)
        {
            if (!ptxPath.EndsWith(".ptx", StringComparison.OrdinalIgnoreCase))
                return null;
            string fatbinPath = Path.ChangeExtension(ptxPath, ".fatbin");
            string sidecarPath = fatbinPath + ".sha256";
            if (!File.Exists(fatbinPath) || !File.Exists(sidecarPath))
                return null;
            try
            {
                string recorded = File.ReadAllText(sidecarPath).Trim();
                string actual = Convert.ToHexString(System.Security.Cryptography.SHA256.HashData(ptxBytes));
                return string.Equals(recorded, actual, StringComparison.OrdinalIgnoreCase)
                    ? File.ReadAllBytes(fatbinPath)
                    : null;
            }
            catch (IOException)
            {
                return null;
            }
        }

        private static unsafe CudaModule LoadImage(byte[] image, int ptxTargetArch)
        {
            fixed (byte* data = image)
            {
                CudaDriverApi.cuModuleLoadData(out IntPtr module, (IntPtr)data).ThrowOnError();
                return new CudaModule(module, ptxTargetArch);
            }
        }

        public static CudaModule LoadFromBytes(byte[] ptxBytes) => LoadFromBytes(ptxBytes, null);

        /// <summary><see cref="LoadFromBytes(byte[])"/>, recording the image's SHA-256 when <paramref name="identify"/> is set.</summary>
        public static CudaModule LoadFromBytes(byte[] ptxBytes, bool identify)
        {
            CudaModule module = LoadFromBytes(ptxBytes);
            if (identify)
                module.ImageSha256 = Sha256(ptxBytes);
            return module;
        }

        private static unsafe CudaModule LoadFromBytes(byte[] ptxBytes, CudaException fatbinFailure)
        {
            if (ptxBytes == null)
                throw new ArgumentNullException(nameof(ptxBytes));

            byte[] terminated = ptxBytes;
            if (terminated.Length == 0 || terminated[terminated.Length - 1] != 0)
            {
                terminated = new byte[ptxBytes.Length + 1];
                Buffer.BlockCopy(ptxBytes, 0, terminated, 0, ptxBytes.Length);
            }

            fixed (byte* ptx = terminated)
            {
                int result = CudaDriverApi.cuModuleLoadData(out IntPtr module, (IntPtr)ptx);
                if (result != 0)
                    throw DiagnoseLoadFailure(result, ptxBytes, fatbinFailure);
                return new CudaModule(module, ParsePtxTargetArch(ptxBytes));
            }
        }

        /// <summary>
        /// The CudaException for a PTX image the driver rejected, its message extended with
        /// <see cref="CudaRequirements.DescribeModuleLoadFailure"/> (read against the current
        /// context's device) and with the fatbin error that came first, when there was one.
        /// </summary>
        private static CudaException DiagnoseLoadFailure(int result, byte[] ptxBytes, CudaException fatbinFailure)
        {
            string message = CudaErrorHelper.GetErrorString(result);
            string diagnosis = null;
            try
            {
                if (CudaDriverApi.cuDriverGetVersion(out int driverVersion) != 0)
                    driverVersion = 0;
                int ccMajor = 0, ccMinor = 0;
                if (CudaDriverApi.cuCtxGetDevice(out int device) == 0)
                {
                    CudaDriverApi.cuDeviceGetAttribute(out ccMajor, CudaDriverApi.CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR, device);
                    CudaDriverApi.cuDeviceGetAttribute(out ccMinor, CudaDriverApi.CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR, device);
                }
                var (toolkit, isa) = CudaRequirements.ParsePtxHeader(ptxBytes);
                diagnosis = CudaRequirements.DescribeModuleLoadFailure(
                    result, toolkit, isa, ParsePtxTargetArch(ptxBytes), driverVersion, ccMajor, ccMinor);
            }
            catch
            {
                // The diagnosis is best effort; the CUDA error itself is still reported.
            }

            if (diagnosis != null)
                message = message.TrimEnd('.') + ". " + diagnosis;
            if (fatbinFailure != null)
                message += " (The fatbin built beside this PTX was rejected first: " + fatbinFailure.Message.TrimEnd('.') + ".)";
            return new CudaException(result, message);
        }

        public IntPtr GetFunction(string name)
        {
            if (!functions.TryGetValue(name, out IntPtr function))
            {
                CudaDriverApi.cuModuleGetFunction(out function, module, name).ThrowOnError();
                functions.Add(name, function);
            }

            return function;
        }

        public void Dispose()
        {
            IntPtr current = module;
            if (current != IntPtr.Zero)
            {
                module = IntPtr.Zero;
                functions.Clear();
                CudaDriverApi.cuModuleUnload(current);
            }
        }
    }
}
