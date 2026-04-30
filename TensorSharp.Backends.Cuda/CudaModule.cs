using System;
using System.Collections.Generic;
using System.IO;
using TensorSharp.Cuda.Interop;

namespace TensorSharp.Cuda
{
    internal sealed class CudaModule : IDisposable
    {
        private readonly Dictionary<string, IntPtr> functions = new Dictionary<string, IntPtr>(StringComparer.Ordinal);
        private IntPtr module;

        private CudaModule(IntPtr module)
        {
            this.module = module;
        }

        public static CudaModule LoadFromFile(string path)
        {
            if (path == null)
                throw new ArgumentNullException(nameof(path));

            byte[] bytes = File.ReadAllBytes(path);
            return LoadFromBytes(bytes);
        }

        public static unsafe CudaModule LoadFromBytes(byte[] ptxBytes)
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
                CudaDriverApi.cuModuleLoadData(out IntPtr module, (IntPtr)ptx).ThrowOnError();
                return new CudaModule(module);
            }
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
