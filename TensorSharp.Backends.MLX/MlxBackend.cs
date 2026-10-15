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

        /// <summary>Read process-wide MLX allocator counters without evaluating
        /// lazy arrays. Throws if the observation is unavailable; use Try for
        /// best-effort diagnostics. These counters are not total physical RAM.</summary>
        public static MlxMemorySnapshot GetMemorySnapshot()
        {
            return MlxNative.GetMemorySnapshot();
        }

        /// <summary>False means unknown, not zero. The out snapshot is usable only
        /// on success. Does not evaluate arrays, synchronize Metal, or certify
        /// ownership; pending lazy work and untracked host/driver memory are absent.</summary>
        public static bool TryGetMemorySnapshot(out MlxMemorySnapshot snapshot, out string error)
        {
            return MlxNative.TryGetMemorySnapshot(out snapshot, out error);
        }

        public static void ClearCache()
        {
            MlxNative.ClearCache();
        }
    }
}
