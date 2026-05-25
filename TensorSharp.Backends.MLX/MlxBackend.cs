using System;
using System.Reflection;
using System.Threading;

namespace TensorSharp.MLX
{
    public static class MlxBackend
    {
        private static int registered;

        public static bool IsAvailable()
        {
            return MlxNative.IsAvailable();
        }

        public static void Register()
        {
            if (Interlocked.Exchange(ref registered, 1) != 0)
                return;

            OpRegistry.RegisterAssembly(Assembly.GetExecutingAssembly());
            MlxFallbackOps.Register();
        }

        public static void EnsureAvailable(int deviceId = 0)
        {
            if (!IsAvailable())
            {
                throw new PlatformNotSupportedException(
                    "MLX backend is not available. Install or copy libmlxc into the app directory, " +
                    "set TENSORSHARP_MLX_LIBRARY to the libmlxc path, or set TENSORSHARP_MLX_LIBRARY_DIR to a directory containing libmlxc.");
            }

            MlxNative.EnsureGpuDevice(deviceId);
        }

        public static MlxMemorySnapshot GetMemorySnapshot()
        {
            return MlxNative.GetMemorySnapshot();
        }

        public static void ClearCache()
        {
            MlxNative.ClearCache();
        }
    }
}
