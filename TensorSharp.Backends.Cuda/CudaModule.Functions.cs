using System;
using System.Runtime.InteropServices;
using TensorSharp.Cuda.Interop;

namespace TensorSharp.Cuda
{
    internal readonly record struct CudaFunctionSymbol(string Name, IntPtr Function);

    internal sealed partial class CudaModule
    {
        /// <summary>Copies exact driver-reported names/handles while the module is
        /// alive. Handles become invalid at Dispose. No inferred demangling.</summary>
        internal unsafe CudaFunctionSymbol[] EnumerateFunctions()
        {
            if (module == IntPtr.Zero) throw new ObjectDisposedException(nameof(CudaModule));
            CudaDriverApi.cuModuleGetFunctionCount(out uint count, module).ThrowOnError();
            if (count == 0 || count > 16384) throw new InvalidOperationException("CUDA module function count is outside the bounded enumeration contract.");
            var handles = new IntPtr[checked((int)count)];
            fixed (IntPtr* data = handles)
                CudaDriverApi.cuModuleEnumerateFunctions(data, count, module).ThrowOnError();
            var symbols = new CudaFunctionSymbol[handles.Length];
            var names = new System.Collections.Generic.HashSet<string>(StringComparer.Ordinal);
            for (int i = 0; i < handles.Length; i++)
            {
                if (handles[i] == IntPtr.Zero) throw new InvalidOperationException("CUDA enumeration returned a null function.");
                CudaDriverApi.cuFuncGetName(out IntPtr pointer, handles[i]).ThrowOnError();
                string name = pointer == IntPtr.Zero ? null : Marshal.PtrToStringUTF8(pointer);
                if (string.IsNullOrEmpty(name) || name.Length > 16384 || !names.Add(name))
                    throw new InvalidOperationException("CUDA enumeration returned an absent, duplicate or unbounded name.");
                symbols[i] = new CudaFunctionSymbol(name, handles[i]);
            }
            Array.Sort(symbols, (a, b) => StringComparer.Ordinal.Compare(a.Name, b.Name));
            return symbols;
        }
    }
}
