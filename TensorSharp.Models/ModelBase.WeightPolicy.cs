// Copyright (c) Zhongkai Fu. All rights reserved.
// Licensed under the BSD-3-Clause license in the repository root.

using System;
using System.Collections.Generic;

namespace TensorSharp.Models
{
    public abstract partial class ModelBase
    {
        // File staging avoids making every miss fault its source into this
        // process's working set. Only enable it automatically for an explicitly
        // budgeted CUDA expert cache whose mapped expert space exceeds available
        // physical RAM. Smaller resident workloads retain the ordinary mmap path.
        private bool ShouldUseExpertFileReads()
        {
            if (_backend != BackendType.GgmlCuda || !OperatingSystem.IsWindows() || !CanUseFileMappedQuantizedWeights) return false;
            string mode = Environment.GetEnvironmentVariable("TS_HOST_MOE_FILE_READ");
            if (mode != null) return mode == "1";
            if (!long.TryParse(Environment.GetEnvironmentVariable("TS_HOST_MOE_EXPERT_CACHE_MB"),
                System.Globalization.NumberStyles.None, System.Globalization.CultureInfo.InvariantCulture,
                out long cacheMb) || cacheMb <= 0 || cacheMb > long.MaxValue / (1024 * 1024)) return false;
            long mappedExpertBytes = 0;
            foreach (var info in _gguf.Tensors.Values)
                if (ShouldLoadWeight(info) && info.Shape.Length == 3 && info.Name.Contains("_exps.") && IsQuantizedLinearWeight(info))
                    mappedExpertBytes = checked(mappedExpertBytes + _gguf.GetTensorByteCount(info));
            try
            {
                long available = InferenceHardwareMemory.CaptureHost().Available;
                bool enabled = available > 0 && mappedExpertBytes > available;
                if (enabled) Console.WriteLine("[HOSTMOE-FILE] Mapped expert bytes exceed available host RAM; enabling bounded read/upload staging.");
                return enabled;
            }
            catch (System.IO.IOException) { return false; }
        }

        private void RegisterExpertFileSource(string name, IntPtr pointer, long bytes)
        {
            var region = _gguf.GetTensorFileRegion(name);
            if (region.ByteLength != bytes || !TensorSharp.GGML.GgmlBasicOps.RegisterHostFileSource(
                pointer, bytes, region.Path, region.Offset))
                throw new InvalidOperationException($"Cannot register expert source {name}.");
        }

        private void RefreshExpertFileSourcesAfterResidencyRelease()
        {
            if (!ShouldUseExpertFileReads()) return;
            foreach (var item in _stackedExpertWeights)
                if (item.Value.IsExternalView && _gguf.Tensors.TryGetValue(item.Key, out var info)
                    && _gguf.TryGetTensorDataPointer(info, out IntPtr mapped) && mapped == item.Value.Data)
                    RegisterExpertFileSource(item.Key, mapped, item.Value.TotalRawBytes);
        }

        // Stored matrix bytes approximate bandwidth cost without making a small
        // tensor of one format outweigh the rest of a mixed-precision checkpoint.
        // Each architecture identifies the matrices its active graph actually uses.
        protected static (long matchingBytes, long totalBytes) MeasureMatmulWeightBytes(
            IReadOnlyDictionary<string, QuantizedWeight> quantWeights,
            IReadOnlyDictionary<string, Tensor> weights, int ggmlType,
            Func<string, bool> isActiveMatrix)
        {
            long matchingBytes = 0, totalBytes = 0;
            foreach (var entry in quantWeights)
            {
                if (entry.Value.Ne1 <= 1 || !isActiveMatrix(entry.Key))
                    continue;
                totalBytes += entry.Value.RawBytes;
                if (entry.Value.GgmlType == ggmlType)
                    matchingBytes += entry.Value.RawBytes;
            }
            foreach (var entry in weights)
            {
                Tensor tensor = entry.Value;
                if (tensor.ElementType != DType.Float32 || tensor.DimensionCount != 2 ||
                    tensor.Sizes[0] <= 1 || !isActiveMatrix(entry.Key) || quantWeights.ContainsKey(entry.Key))
                    continue;
                // Count F32 matrices stored separately, excluding a cached F32
                // mirror when the quantized matrix is the active representation.
                totalBytes += tensor.ElementCount() * sizeof(float);
            }
            return (matchingBytes, totalBytes);
        }
    }
}
