// Copyright (c) Zhongkai Fu. Licensed under the BSD-3-Clause license.
#nullable enable
using System;
using System.Linq;
using TensorSharp.Memory;
using TensorSharp.Runtime;

namespace TensorSharp.Models;

internal readonly record struct GlmUnifiedExpertProjection(GgmlTensorType Type,
    long BytesPerExpert, long TensorTailBytes, bool SupportsSelectedTier);

internal sealed record GlmUnifiedExpertLayout(GlmUnifiedExpertProjection Gate,
    GlmUnifiedExpertProjection Up, GlmUnifiedExpertProjection Down)
{
    internal (GgmlTensorType, GgmlTensorType, GgmlTensorType) Types => (Gate.Type, Up.Type, Down.Type);
    internal long BytesPerExpert => checked(Gate.BytesPerExpert + Up.BytesPerExpert + Down.BytesPerExpert);
    internal long TensorTailBytes => checked(Gate.TensorTailBytes + Up.TensorTailBytes + Down.TensorTailBytes);
    internal bool SupportsSelectedTier => Gate.SupportsSelectedTier && Up.SupportsSelectedTier && Down.SupportsSelectedTier;
}

/// <summary>A pre-load CUDA payload forecast, never an allocator measurement or
/// a physical hard-limit certificate. The loaded model must still pass the
/// native exact minimum check before enabling the selected-expert adapter.</summary>
internal sealed record GlmSelectedExpertMemoryPlan(int PrefillRows, int CpuExpertLayers,
    long ExecutionReserveBytes, long CacheCapacityBytes, long CacheMinimumForecastBytes)
{
    // Matches the native fixed routing-validation scratch. Larger top-k models
    // need a smaller row cap; do not resize or omit this independently owned
    // scratch simply to permit wider chunks.
    private const int MaximumRouteElements = 8192;
    private const int LegacyMaximumRouteElements = 4096;

    internal static GlmSelectedExpertMemoryPlan SelectForSession(GlmUnifiedMemoryGeometry geometry,
        UnifiedMemoryModelOptions options, int context, long deviceCeiling, long visionBytes)
    {
        ArgumentNullException.ThrowIfNull(options);
        if (!options.EnableAdaptiveGlmPrefill) return Select(geometry, options, context, deviceCeiling, visionBytes);
        ArgumentNullException.ThrowIfNull(geometry);
        if (!options.EnableGlmSelectedExperts || options.MaximumRunningRequests is < 1 or > 3
            || options.PrefillChunkTokens is < 1 or > 512 || options.MaximumExpertCacheBytes < 0
            || context <= 0 || context > geometry.DeclaredContextTokens || deviceCeiling < 0 || visionBytes < 0)
            throw new ArgumentOutOfRangeException(nameof(options));
        int requestedRows = options.ResolveGlmPrefillChunkTokens();
        long reserve = geometry.AdaptiveExecutionPlacementReserve(context, options.MaximumRunningRequests);
        int cpu = geometry.SelectCpuExpertLayers(deviceCeiling, reserve, visionBytes);
        int maximum = Math.Min(context, requestedRows);
        if (cpu == 0) return new(maximum, 0, reserve, 0, 0);
        if (geometry.UsedExperts is < 1 or > 64) throw new NotSupportedException("Incomplete GLM selected-expert route geometry.");
        maximum = Math.Min(maximum, MaximumRouteElements / geometry.UsedExperts);
        long capacity = GlmUnifiedMemoryGeometry.SelectExpertCacheCapacity(deviceCeiling,
            geometry.DenseDeviceBytes, reserve, visionBytes, options.MaximumExpertCacheBytes);
        for (int rows = maximum; rows >= 1; rows--)
        {
            // Native installs this complete immutable phase floor once. Actual
            // requests may use smaller widths without replacing that plan.
            long minimum = ForecastMinimum(geometry, options.MaximumRunningRequests, rows);
            if (capacity < minimum) continue;
            long totalReserve = checked(reserve + capacity);
            cpu = geometry.SelectCpuExpertLayers(deviceCeiling, totalReserve, visionBytes);
            return new(rows, cpu, totalReserve, capacity, minimum);
        }
        throw new MemoryPressureException("GLM adaptive prefill cannot preserve complete slots, joint decode and its immutable selected-expert phase floor.");
    }

    internal static long ForecastMinimum(GlmUnifiedMemoryGeometry geometry, int decodeRows, int prefillRows,
        int maximumRouteElements = MaximumRouteElements)
    {
        ArgumentNullException.ThrowIfNull(geometry);
        if (decodeRows is < 1 or > 3 || prefillRows is < 1 or > 1024
            || maximumRouteElements is < 1 or > MaximumRouteElements)
            throw new ArgumentOutOfRangeException(nameof(prefillRows));
        if (geometry.ExpertCount <= 0 || geometry.UsedExperts is < 1 or > 64
            || geometry.UsedExperts > geometry.ExpertCount || geometry.Hidden <= 0 || geometry.ExpertFf <= 0
            || geometry.ExpertLayerLayouts.Length != geometry.Layers)
            throw new NotSupportedException("Incomplete GLM selected-expert metadata geometry.");
        if (prefillRows > maximumRouteElements / geometry.UsedExperts)
            throw new ArgumentOutOfRangeException(nameof(prefillRows), "GLM prefill exceeds its fixed selected-route scratch.");
        var layouts = geometry.ExpertLayerLayouts.OfType<GlmUnifiedExpertLayout>().ToArray();
        if (layouts.Length == 0 || layouts.Any(l => !l.SupportsSelectedTier
            || l.Gate.BytesPerExpert <= 0 || l.Up.BytesPerExpert <= 0 || l.Down.BytesPerExpert <= 0
            || l.Gate.TensorTailBytes < 0 || l.Up.TensorTailBytes < 0 || l.Down.TensorTailBytes < 0))
            throw new NotSupportedException("GLM selected-expert planning requires complete quantized gate/up/down layouts.");
        checked
        {
            // Adapter v7 serializes helper phases under one mutex. Successful
            // return may leave a device-output copy queued; the GLM DSA caller
            // fences before downstream work. Owner destructors independently
            // synchronize before free/refund, including inter-phase reclaim.
            // A wide union replaces inactive payload only after its physical
            // release. Forecast one wide
            // group plus the other groups' narrow floors, not all-wide at once.
            // This is still not physical credit; native measures every layout
            // and shared callbacks independently admit each real owner.
            var groups = layouts.GroupBy(l => l.Types).ToArray();
            long rows = prefillRows > 8 ? Math.Max(decodeRows, prefillRows) : decodeRows;
            long union = Math.Min(geometry.ExpertCount, rows * geometry.UsedExperts);
            long narrowUnion = Math.Min(geometry.ExpertCount, (long)decodeRows * geometry.UsedExperts);
            // Native GLM retains every compact prefill width through eight,
            // even when the admitted independent decode width is only one.
            // A short prefill may execute smaller microgroups to fit its live
            // expert table, so this does not increase the required weight union
            // above. It does increase the simultaneously owned graph scratch.
            long narrowRows = Math.Max(decodeRows, Math.Min(prefillRows, 8));
            long narrowRetainedRows = narrowRows * (narrowRows + 1) / 2;
            // All narrow variants 1..N and one wide variant share the quantized
            // weight leaves. Sum every allocating F32/I32 tensor even when the
            // allocator aliases or reuses it: input, ids/routes, gate/up, both
            // optional clamps, SiLU/product, down/CONT/weighted, K-1 ADDs, output.
            long elementsPerRow = geometry.Hidden + 2L * geometry.UsedExperts
                + 6L * geometry.ExpertFf * geometry.UsedExperts
                + 4L * geometry.Hidden * geometry.UsedExperts;
            long narrowScratch = 4L * narrowRetainedRows * elementsPerRow;
            long wideScratch = narrowScratch + 4L * (prefillRows > 8 ? prefillRows : 0) * elementsPerRow;
            // Unchanged ggml CUDA uses 128-byte alignment, quantized 512-row
            // tail padding (retained separately above), and this native graph
            // has at most 256 tensor descriptors per variant. Include a full
            // alignment unit for each descriptor, the three shared weights,
            // and the native per-group 1 MiB workspace allowance. No flash
            // attention tensor or F32 weight materialization occurs here.
            long narrowAlignment = 128L * (3 + 256 * narrowRows);
            long wideAlignment = narrowAlignment + (prefillRows > 8 ? 128L * 256 : 0);
            long totalNarrow = 0, largestExpansion = 0;
            foreach (var group in groups)
            {
                long expertBytes = group.Max(l => l.BytesPerExpert);
                long tail = group.Max(l => l.TensorTailBytes);
                long narrow = narrowUnion * expertBytes + tail + narrowScratch + narrowAlignment + (1L << 20);
                long wide = union * expertBytes + tail + wideScratch + wideAlignment + (1L << 20);
                totalNarrow += narrow;
                largestExpansion = Math.Max(largestExpansion, wide - narrow);
            }
            return totalNarrow + largestExpansion;
        }
    }

    internal static GlmSelectedExpertMemoryPlan Select(GlmUnifiedMemoryGeometry geometry,
        UnifiedMemoryModelOptions options, int context, long deviceCeiling, long visionBytes)
    {
        ArgumentNullException.ThrowIfNull(geometry);
        ArgumentNullException.ThrowIfNull(options);
        if (context <= 0 || context > geometry.DeclaredContextTokens || options.PrefillChunkTokens is < 1 or > 512
            || options.MaximumRunningRequests is < 1 or > 8 || deviceCeiling < 0 || visionBytes < 0
            || options.MaximumExpertCacheBytes < 0)
            throw new ArgumentOutOfRangeException(nameof(options));
        if (options.EnableGlmSelectedExperts && options.MaximumRunningRequests > 3)
            throw new NotSupportedException("GLM selected-expert planning supports at most three decode rows.");
        // Larger admitted chunks amortize reading each layer's routed expert
        // union. Keep the operator cap and decrease it only when the complete
        // slots, graph generations and expert phase floor cannot coexist.
        int cap = Math.Min(context, options.PrefillChunkTokens);
        for (int rows = cap; rows >= 1; rows--)
        {
            // Mandatory live slots include the reusable primary. Idle terminal
            // prefix slots are optional owners, reclaimed by actual admission
            // when needed; do not reserve imaginary future cache allocations.
            long reserve = geometry.ExecutionPlacementReserve(context, rows, options.MaximumRunningRequests);
            int cpu;
            try { cpu = geometry.SelectCpuExpertLayers(deviceCeiling, reserve, visionBytes); }
            // Slot, dense and vision ownership stay fixed while smaller chunks
            // reduce scratch. This applies equally to mapped CPU experts: their
            // placement can fit a smaller chunk without any selected CUDA tier.
            catch (MemoryPressureException) when (rows > 1) { continue; }
            if (!options.EnableGlmSelectedExperts || cpu == 0)
                return new(rows, cpu, reserve, 0, 0);
            if (geometry.UsedExperts <= 0 || geometry.UsedExperts > 64)
                throw new NotSupportedException("Incomplete GLM selected-expert route geometry.");
            if (rows > LegacyMaximumRouteElements / geometry.UsedExperts) continue;
            long capacity = GlmUnifiedMemoryGeometry.SelectExpertCacheCapacity(deviceCeiling,
                geometry.DenseDeviceBytes, reserve, visionBytes, options.MaximumExpertCacheBytes);
            long minimum = ForecastMinimum(geometry, options.MaximumRunningRequests, rows, LegacyMaximumRouteElements);
            if (capacity < minimum) continue;
            long totalReserve = checked(reserve + capacity);
            cpu = geometry.SelectCpuExpertLayers(deviceCeiling, totalReserve, visionBytes);
            return new(rows, cpu, totalReserve, capacity, minimum);
        }
        throw new MemoryPressureException("GLM selected-expert CUDA capacity cannot protect the complete narrow decode union and graph forecast after live slots, dense/vision weights and operator ceilings.");
    }
}
