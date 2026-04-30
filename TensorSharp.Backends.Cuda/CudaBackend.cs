using System.Reflection;

namespace TensorSharp.Cuda
{
    public static class CudaBackend
    {
        private static int registered;

        public static void Register()
        {
            if (System.Threading.Interlocked.Exchange(ref registered, 1) != 0)
                return;

            OpRegistry.RegisterAssembly(Assembly.GetExecutingAssembly());
        }

        public static bool IsAvailable() => CudaDevice.IsAvailable();
    }
}
