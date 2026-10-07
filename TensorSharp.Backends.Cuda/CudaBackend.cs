using System.Reflection;

namespace TensorSharp.Cuda
{
    public static class CudaBackend
    {
        private static int registered;

        public static void Register()
        {
            // Whole-model engines call the driver before creating an allocator/context.
            // Install the platform DLL mappings before any of those first CUDA calls.
            Interop.CudaLibraryResolver.Register();
            if (System.Threading.Interlocked.Exchange(ref registered, 1) != 0)
                return;

            OpRegistry.RegisterAssembly(Assembly.GetExecutingAssembly());
        }

        public static bool IsAvailable() => CudaDevice.IsAvailable();
    }
}
