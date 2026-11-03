// Copyright (c) Zhongkai Fu. All rights reserved.
// Licensed under the BSD-3-Clause license in the repository root.
using System;
using System.Collections.Generic;
using System.Globalization;
using System.Runtime.InteropServices;
using TensorSharp.Cuda.Interop;

namespace TensorSharp.Cuda
{
    /// <summary>Unconnected experimental W8G32 vocabulary module. This does not select
    /// a model policy, load/requantize weights, or replace any existing Q35 dispatch.
    /// Requires a current CUDA 13+ context and the separately compiled original-source
    /// module. Input payload validity/model quality are separate validation gates.</summary>
    public sealed unsafe class CudaW8G32HeadModule : IDisposable
    {
        public const int Vocabulary = 248320, Width = 5120, MaximumRows = 33;
        public const long CodesBytes = 1271398400, ScalesOffset = CodesBytes,
            ScalesBytes = 79462400, WeightBytes = 1350860800;
        internal const string CastSymbol = "ts_w8g32_cast_f32_bf16";
        private readonly CudaModule _module;
        private readonly IntPtr _context, _cast;
        private readonly IntPtr[] _kernels = new IntPtr[MaximumRows + 1];
        private bool _disposed;
        public string ImageSha256 => _module.ImageSha256;
        public IReadOnlyList<KernelInfo> Kernels { get; }
        public string LaunchPolicy => "original_programmatic_stream_serialization_attribute6";
        public readonly record struct KernelInfo(int Rows, string Symbol, int BlockThreads,
            int Registers, int LocalBytes, int StaticSharedBytes, int ActiveCtas);
        internal readonly record struct Dispatch(int Rows, int TileTokens, int KWarps,
            int MinimumBlocks, int Threads, uint GridX);

        [StructLayout(LayoutKind.Explicit, Size = 16)]
        internal struct OutputArgument
        {
            [FieldOffset(0)] public IntPtr Data;
            [FieldOffset(8)] public int LeadingDimension;
            [FieldOffset(12)] public int ReservedPadding;
        }

        private CudaW8G32HeadModule(string path)
        {
            if (IntPtr.Size != 8) throw new PlatformNotSupportedException("W8 requires a 64-bit ABI.");
            ValidateManagedAbi();
            CudaDriverApi.cuDriverGetVersion(out int version).ThrowOnError();
            if (version < 13000) throw new PlatformNotSupportedException("W8 native module requires CUDA 13 or newer.");
            CudaDriverApi.cuCtxGetCurrent(out _context).ThrowOnError();
            if (_context == IntPtr.Zero) throw new InvalidOperationException("Make the owning CUDA context current before loading W8.");
            _module = CudaModule.LoadFromFile(path, identify: true);
            try
            {
                var symbols = ValidateSymbols(_module.EnumerateFunctions());
                var info = new List<KernelInfo>(MaximumRows + 1);
                _cast = symbols[0].Function;
                info.Add(PrepareFunction(0, symbols[0].Name, _cast, 256, LoadFunction, InspectFunction));
                for (int n = 1; n <= MaximumRows; n++)
                {
                    _kernels[n] = symbols[n].Function;
                    info.Add(PrepareFunction(n, symbols[n].Name, _kernels[n], Plan(n).Threads, LoadFunction, InspectFunction));
                }
                Kernels = info.AsReadOnly();
            }
            catch { _module.Dispose(); throw; }
        }

        /// <summary>Loads explicit caller-supplied PTX/cubin/fatbin in the current context.
        /// Missing enumeration/parameter/PDL APIs fail; there is no silent fallback.</summary>
        public static CudaW8G32HeadModule LoadFromFile(string path)
        {
            ArgumentException.ThrowIfNullOrEmpty(path);
            return new CudaW8G32HeadModule(path);
        }

        internal static Dispatch Plan(int n)
        {
            if (n < 1 || n > MaximumRows) throw new ArgumentOutOfRangeException(nameof(n), "Only exact T=1..33 is implemented; no alternate math fallback.");
            int tile = n <= 8 ? 8 : n <= 16 ? 16 : n <= 24 ? 24 : n <= 32 ? 32 : 40;
            int warps = n <= 32 ? 8 : 4;
            return new Dispatch(n, tile, warps, 2, warps * 32, Vocabulary / 16);
        }

        // Full Itanium spelling observed in the successful original e600 NVCC ELF,
        // including schedule/cache/epilogue policy. Unknown compiler spelling is a
        // refusal, never a substring-based guess at another template specialization.
        internal static string ExpectedSymbol(int n)
        {
            var p = Plan(n);
            return "_ZN6ninfer3ops6detail21w8_small_t_mma_kernelINS1_16W8LinearGeometryILi248320ELi5120EEELi"
                + n.ToString(CultureInfo.InvariantCulture) + "ENS1_19W8SmallTMmaScheduleILi"
                + p.KWarps.ToString(CultureInfo.InvariantCulture) + "ELi" + p.TileTokens.ToString(CultureInfo.InvariantCulture)
                + "ELi2ELNS1_22W8SmallTMmaScaleAccessE" + (n > 4 ? "1" : "0")
                + "ELNS0_5CacheE0ELS7_1ELNS1_26W8SmallTMmaActivationStageE0EEENS1_18W8ContiguousOutputENS1_24W8SmallTMmaStoreEpilogueENS1_23W8SmallTMmaIdentityRowsELb0EEEvPK13__nv_bfloat16PKhSH_T2_T3_T4_";
        }
        internal static bool TryParseSymbol(string name, out int n)
        {
            n = 0;
            const string prefix = "_ZN6ninfer3ops6detail21w8_small_t_mma_kernelINS1_16W8LinearGeometryILi248320ELi5120EEELi";
            if (name == null || !name.StartsWith(prefix, StringComparison.Ordinal)) return false;
            int end = name.IndexOf('E', prefix.Length);
            if (end < 0 || !int.TryParse(name.AsSpan(prefix.Length, end - prefix.Length), NumberStyles.None, CultureInfo.InvariantCulture, out int parsed)
                || parsed < 1 || parsed > MaximumRows || !string.Equals(name, ExpectedSymbol(parsed), StringComparison.Ordinal)) return false;
            n = parsed; return true;
        }
        internal static CudaFunctionSymbol[] ValidateSymbols(CudaFunctionSymbol[] symbols)
        {
            if (symbols == null || symbols.Length != MaximumRows + 1) throw new InvalidOperationException("W8 requires exactly the cast plus all 33 original kernels.");
            var result = new CudaFunctionSymbol[MaximumRows + 1];
            var handles = new HashSet<IntPtr>();
            foreach (var symbol in symbols)
            {
                int n;
                if (symbol.Name == CastSymbol) n = 0;
                else if (!TryParseSymbol(symbol.Name, out n)) throw new InvalidOperationException("Unrecognized W8 module function: " + symbol.Name);
                if (symbol.Function == IntPtr.Zero || !handles.Add(symbol.Function) || result[n].Function != IntPtr.Zero)
                    throw new InvalidOperationException("Duplicate/null W8 function binding.");
                result[n] = symbol;
            }
            for (int n = 0; n <= MaximumRows; n++)
                if (result[n].Function == IntPtr.Zero) throw new InvalidOperationException("W8 specialization missing: " + n);
            return result;
        }

        internal static void ValidateManagedAbi()
        {
            if (IntPtr.Size != 8 || Marshal.SizeOf<OutputArgument>() != 16
                || Marshal.OffsetOf<OutputArgument>(nameof(OutputArgument.LeadingDimension)).ToInt32() != 8
                || Marshal.SizeOf<CudaDriverApi.CudaLaunchAttribute>() != 72
                || Marshal.OffsetOf<CudaDriverApi.CudaLaunchAttribute>(nameof(CudaDriverApi.CudaLaunchAttribute.ProgrammaticStreamSerializationAllowed)).ToInt32() != 8
                || Marshal.SizeOf<CudaDriverApi.CudaLaunchConfig>() != 56
                || Marshal.OffsetOf<CudaDriverApi.CudaLaunchConfig>(nameof(CudaDriverApi.CudaLaunchConfig.Stream)).ToInt32() != 32
                || Marshal.OffsetOf<CudaDriverApi.CudaLaunchConfig>(nameof(CudaDriverApi.CudaLaunchConfig.Attributes)).ToInt32() != 40
                || Marshal.OffsetOf<CudaDriverApi.CudaLaunchConfig>(nameof(CudaDriverApi.CudaLaunchConfig.NumAttributes)).ToInt32() != 48)
                throw new PlatformNotSupportedException("Managed W8/PDL ABI does not match cuda.h.");
        }
        internal static void ValidateParameterLayout(ulong[] offset, ulong[] size, ulong[] expectedOffset, ulong[] expectedSize)
        {
            if (offset.Length != expectedOffset.Length || size.Length != expectedSize.Length)
                throw new InvalidOperationException("W8 kernel parameter count changed.");
            for (int i = 0; i < offset.Length; i++)
                if (offset[i] != expectedOffset[i] || size[i] != expectedSize[i])
                    throw new InvalidOperationException("W8 device parameter ABI changed at index " + i);
        }
        // Enumerated handles may be unloaded. CUDA requires explicit materialization
        // before querying MAX_THREADS_PER_BLOCK and occupancy. Keep this local to
        // the selected W8 functions; generic enumeration stays metadata-only.
        internal static KernelInfo PrepareFunction(int rows, string name, IntPtr function, int threads,
            Action<IntPtr> load, Func<int, string, IntPtr, int, KernelInfo> inspect)
        {
            if (function == IntPtr.Zero) throw new ArgumentException("A selected W8 function is required.", nameof(function));
            load(function); // Failure propagates before any parameter/resource inspection.
            return inspect(rows, name, function, threads);
        }
        private static void LoadFunction(IntPtr function) => CudaDriverApi.cuFuncLoad(function).ThrowOnError();
        private static KernelInfo InspectFunction(int rows, string name, IntPtr function, int threads)
        {
            if (rows == 0) CheckActualParameters(function, new ulong[] { 0, 8, 16 }, new ulong[] { 8, 8, 8 });
            else CheckActualParameters(function, new ulong[] { 0, 8, 16, 24, 40, 41 }, new ulong[] { 8, 8, 8, 16, 1, 1 });
            return Resources(rows, name, function, threads);
        }
        private static void CheckActualParameters(IntPtr kernel, ulong[] expectedOffset, ulong[] expectedSize)
        {
            var offset = new ulong[expectedOffset.Length]; var size = new ulong[expectedSize.Length];
            for (int i = 0; i < offset.Length; i++)
            {
                CudaDriverApi.cuFuncGetParamInfo(kernel, new UIntPtr((uint)i), out UIntPtr o, out UIntPtr s).ThrowOnError();
                offset[i] = o.ToUInt64(); size[i] = s.ToUInt64();
            }
            ValidateParameterLayout(offset, size, expectedOffset, expectedSize);
            int extra = CudaDriverApi.cuFuncGetParamInfo(kernel, new UIntPtr((uint)offset.Length), out _, out _);
            if (extra != 1) throw new InvalidOperationException("W8 extra parameter probe did not return CUDA_ERROR_INVALID_VALUE; actual result " + extra);
        }
        private static KernelInfo Resources(int rows, string name, IntPtr function, int threads)
        {
            CudaDriverApi.cuFuncGetAttribute(out int maxThreads, 0, function).ThrowOnError();
            CudaDriverApi.cuFuncGetAttribute(out int shared, 1, function).ThrowOnError();
            CudaDriverApi.cuFuncGetAttribute(out int local, 3, function).ThrowOnError();
            CudaDriverApi.cuFuncGetAttribute(out int registers, 4, function).ThrowOnError();
            CudaDriverApi.cuOccupancyMaxActiveBlocksPerMultiprocessor(out int active, function, threads, UIntPtr.Zero).ThrowOnError();
            if (maxThreads < threads || active < 1 || registers < 1) throw new InvalidOperationException("W8 function is not launchable with its original block.");
            return new KernelInfo(rows, name, threads, registers, local, shared, active);
        }

        internal static ulong CheckBuffer(IntPtr pointer, long available, long required, int alignment)
        {
            ulong start = unchecked((ulong)pointer.ToInt64());
            if (pointer.ToInt64() <= 0 || available < required || required <= 0 || start % (uint)alignment != 0 || start > (ulong)long.MaxValue - (ulong)required)
                throw new ArgumentException("W8 pointer, declared capacity, alignment or arithmetic overflow is invalid.");
            return start;
        }
        internal static void Disjoint(ulong a, long bytesA, ulong b, long bytesB)
        {
            if (a < b + (ulong)bytesB && b < a + (ulong)bytesA) throw new ArgumentException("W8 simultaneously live buffers overlap.");
        }
        private void CheckContext()
        {
            ObjectDisposedException.ThrowIf(_disposed, this);
            CudaDriverApi.cuCtxGetCurrent(out IntPtr actual).ThrowOnError();
            if (actual != _context) throw new InvalidOperationException("The W8 module's owning CUDA context is not current.");
        }
        /// <summary>Original BF16-input kernel. Caller guarantees finite normal/zero
        /// BF16 activations and finite F16 scales. Counts are caller-declared bounds,
        /// not independent driver allocation ownership proofs. Buffers/module must
        /// remain alive until stream completion; synchronize before disposal.</summary>
        public void LaunchBf16(IntPtr weights, long weightsBytes, IntPtr inputBf16, long inputBytes,
            IntPtr outputBf16, long outputBytes, int rows, IntPtr stream)
        {
            var plan = Plan(rows); long xBytes = (long)rows * Width * 2, yBytes = (long)rows * Vocabulary * 2;
            ulong w = CheckBuffer(weights, weightsBytes, WeightBytes, 16), x = CheckBuffer(inputBf16, inputBytes, xBytes, 16), y = CheckBuffer(outputBf16, outputBytes, yBytes, 2);
            Disjoint(w, WeightBytes, x, xBytes); Disjoint(w, WeightBytes, y, yBytes); Disjoint(x, xBytes, y, yBytes);
            CheckContext();
            IntPtr scales = checked(weights + (nint)ScalesOffset);
            var output = new OutputArgument { Data = outputBf16, LeadingDimension = Vocabulary, ReservedPadding = 0 };
            byte epilogue = 0, rowPolicy = 0;
            void** args = stackalloc void*[6] { &inputBf16, &weights, &scales, &output, &epilogue, &rowPolicy };
            var attr = new CudaDriverApi.CudaLaunchAttribute { Id = 6, ProgrammaticStreamSerializationAllowed = 1 };
            var config = new CudaDriverApi.CudaLaunchConfig { GridDimX = plan.GridX, GridDimY = 1, GridDimZ = 1,
                BlockDimX = (uint)plan.Threads, BlockDimY = 1, BlockDimZ = 1, SharedMemBytes = 0,
                Stream = stream, Attributes = (IntPtr)(&attr), NumAttributes = 1 };
            CudaDriverApi.cuLaunchKernelEx(in config, _kernels[rows], (IntPtr)args, IntPtr.Zero).ThrowOnError();
        }
        /// <summary>Complete F32→BF16 cast + original W8 matmul on the same stream.
        /// No cached conversion, synchronization, argmax or output cast is hidden.</summary>
        public void LaunchF32(IntPtr weights, long weightsBytes, IntPtr inputF32, long inputBytes,
            IntPtr workspaceBf16, long workspaceBytes, IntPtr outputBf16, long outputBytes, int rows, IntPtr stream)
        {
            Plan(rows); long count = (long)rows * Width, xBytes = count * 4, bBytes = count * 2, yBytes = (long)rows * Vocabulary * 2;
            ulong w = CheckBuffer(weights, weightsBytes, WeightBytes, 16), x = CheckBuffer(inputF32, inputBytes, xBytes, 4), b = CheckBuffer(workspaceBf16, workspaceBytes, bBytes, 16), y = CheckBuffer(outputBf16, outputBytes, yBytes, 2);
            Disjoint(w, WeightBytes, x, xBytes); Disjoint(x, xBytes, b, bBytes); Disjoint(w, WeightBytes, b, bBytes);
            Disjoint(x, xBytes, y, yBytes); Disjoint(b, bBytes, y, yBytes); Disjoint(w, WeightBytes, y, yBytes);
            CheckContext(); ulong elements = (ulong)count;
            void** args = stackalloc void*[3] { &inputF32, &workspaceBf16, &elements };
            CudaDriverApi.cuLaunchKernel(_cast, (uint)((count + 255) / 256), 1, 1, 256, 1, 1, 0, stream, (IntPtr)args, IntPtr.Zero).ThrowOnError();
            LaunchBf16(weights, weightsBytes, workspaceBf16, workspaceBytes, outputBf16, outputBytes, rows, stream);
        }
        public void Dispose()
        {
            if (_disposed) return;
            CheckContext(); _module.Dispose(); _disposed = true;
        }
    }
}
