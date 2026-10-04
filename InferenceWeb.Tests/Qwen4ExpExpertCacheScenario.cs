// Copyright (c) Zhongkai Fu. All rights reserved.
// Licensed under the BSD-3-Clause license in the repository root.
using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.IO;
using System.Linq;
using System.Runtime.InteropServices;
using TensorSharp.Models;
using TensorSharp.Runtime;

namespace InferenceWeb.Tests;

// Shared by the regression suite and the standalone validation probe. These are
// teacher-forced synthetic engine checks, not trained-model quality evidence.
internal static class Qwen4ExpExpertCacheScenario
{
    internal sealed record Stats(long ReservedBytes, long BudgetBytes, long Hits, long Misses, long Calls);

    [UnmanagedFunctionPointer(CallingConvention.Cdecl)]
    private delegate int ReadStats(out long reserved, out long budget, out long hits, out long misses, out long calls);

    internal static string MappedNativePath()
    {
        string[] names = { "GgmlOps.dll", "libGgmlOps.so", "libGgmlOps.dylib" };
        // Process.Modules omits dlopen-loaded images on macOS. Observe dyld's
        // actual image table there; a guessed on-disk path cannot prove which
        // native binary performed a validation run.
        IEnumerable<string> paths;
        if (OperatingSystem.IsMacOS())
            paths = MacMappedImagePaths();
        else
        {
            using var process = Process.GetCurrentProcess();
            paths = process.Modules.Cast<ProcessModule>().Select(module => module.FileName).ToArray();
        }
        string[] matches = paths
            .Where(path => names.Contains(Path.GetFileName(path), StringComparer.OrdinalIgnoreCase))
            .Select(Path.GetFullPath).Distinct(StringComparer.Ordinal).ToArray();
        if (matches.Length != 1)
            throw new InvalidOperationException($"Expected one mapped GgmlOps library; observed {matches.Length}: {string.Join(", ", matches)}");
        return matches[0];
    }

    private static IEnumerable<string> MacMappedImagePaths()
    {
        uint count = DyldImageCount();
        for (uint index = 0; index < count; ++index)
        {
            string path = Marshal.PtrToStringUTF8(DyldGetImageName(index));
            if (!string.IsNullOrEmpty(path)) yield return path;
        }
    }

    [DllImport("/usr/lib/libSystem.B.dylib", EntryPoint = "_dyld_image_count")]
    private static extern uint DyldImageCount();

    [DllImport("/usr/lib/libSystem.B.dylib", EntryPoint = "_dyld_get_image_name")]
    private static extern IntPtr DyldGetImageName(uint index);

    internal static Stats CacheStats()
    {
        IntPtr module = NativeLibrary.Load(MappedNativePath());
        try
        {
            var read = Marshal.GetDelegateForFunctionPointer<ReadStats>(
                NativeLibrary.GetExport(module, "TSGgml_HostMoeExpertCacheStats"));
            if (read(out long reserved, out long budget, out long hits, out long misses, out long calls) != 1)
                throw new InvalidOperationException("Native expert-cache statistics are unavailable.");
            return new Stats(reserved, budget, hits, misses, calls);
        }
        finally { NativeLibrary.Free(module); }
    }

    internal static float[][] Run(ModelBase model, int firstToken, int salt, int steps = 48)
    {
        model.ResetKVCache();
        var rows = new float[steps + 1][];
        // A one-token prefill enters exactly the same selected-expert decode
        // contract as each subsequent step; long prefill has a separate path.
        rows[0] = (float[])model.ForwardRefill(new[] { firstToken }).Clone();
        for (int i = 0; i < steps; i++)
            rows[i + 1] = (float[])model.Forward(new[] { (i * 53 + salt) % 251 }).Clone();
        if (rows.Any(row => row.Length != Qwen4ExpSyntheticModelBuilder.Vocab || row.Any(v => !float.IsFinite(v))))
            throw new InvalidOperationException("The synthetic engine returned incomplete or nonfinite logits.");
        return rows;
    }

    internal static Dictionary<string, float[][]> Exercise(string modelA, string modelB, BackendType backend, bool host,
        Action<Stats, Stats>? observeActiveCache = null)
    {
        MoeCpuOffloadConfig.Reset();
        if (host) MoeCpuOffloadConfig.SetAllLayers();
        var rows = new Dictionary<string, float[][]>();
        using (ModelBase model = ModelBase.Create(modelA, backend))
        {
            var before = observeActiveCache == null ? null : CacheStats();
            rows["a1"] = Run(model, 11, 17);
            rows["b"] = Run(model, 113, 83);
            rows["a2"] = Run(model, 11, 17);
            foreach (int width in new[] { 2, 4, 8 })
                rows[$"verify{width}"] = Verify((Qwen4ExpModel)model, width);
            observeActiveCache?.Invoke(before!, CacheStats());
        }
        // A different quantization with identical shapes makes allocator address
        // reuse dangerous: stale weight slots must never survive model disposal.
        using (ModelBase model = ModelBase.Create(modelB, backend))
        {
            var before = observeActiveCache == null ? null : CacheStats();
            rows["other-model"] = Run(model, 47, 139);
            observeActiveCache?.Invoke(before!, CacheStats());
        }
        using (ModelBase model = ModelBase.Create(modelA, backend))
        {
            var before = observeActiveCache == null ? null : CacheStats();
            rows["reloaded-a"] = Run(model, 11, 17);
            observeActiveCache?.Invoke(before!, CacheStats());
        }
        MoeCpuOffloadConfig.Reset();
        return rows;
    }

    internal static float[][] Verify(Qwen4ExpModel model, int width, Action<string, Stats, Stats>? observe = null)
    {
        int[] prefix = Enumerable.Range(0, width).Select(i => (i * 37 + 11) % 251).ToArray();
        int[] draft = Enumerable.Range(0, width).Select(i => (i * 53 + 29) % 251).ToArray();
        model.ResetKVCache();
        var before = observe == null ? null : CacheStats();
        float[] prefill = (float[])model.ForwardRefill(prefix).Clone();
        observe?.Invoke("short-prefill", before!, CacheStats());
        model.SpecSnapshotRecurrentState();
        var hidden = new float[width * model.SpecFeatureSize];
        var allLogits = new float[width * model.Config.VocabSize];
        before = observe == null ? null : CacheStats();
        model.SpecForward(draft, hidden, allLogits, allLogitsRows: true);
        observe?.Invoke("target-verification", before!, CacheStats());
        if (hidden.Any(value => !float.IsFinite(value)) || allLogits.Any(value => !float.IsFinite(value)))
            throw new InvalidOperationException("Speculative target verification returned nonfinite hidden/logit rows.");
        var rows = new List<float[]> { prefill };
        for (int row = 0; row < width; row++)
            rows.Add(allLogits[(row * model.Config.VocabSize)..((row + 1) * model.Config.VocabSize)]);
        model.SpecRestoreRecurrentState();
        model.SpecRewindCache(prefix.Length);
        rows.Add((float[])model.Forward(new[] { draft[0] }).Clone());
        float[] continued = (float[])model.Forward(new[] { 79 }).Clone();
        rows.Add(continued);
        model.ResetKVCache();
        model.ForwardRefill(prefix);
        model.Forward(new[] { draft[0] });
        float[] cold = (float[])model.Forward(new[] { 79 }).Clone();
        if (Difference(new[] { continued }, new[] { cold }).RelativeL2 != 0)
            throw new InvalidOperationException("Target verification rollback changed accepted-prefix continuation.");
        return rows.ToArray();
    }

    internal static (double RelativeL2, double MaxAbsolute, int ArgmaxMismatches) Difference(float[][] actual, float[][] expected)
    {
        if (actual.Length != expected.Length || actual.Length == 0)
            throw new InvalidOperationException("Different or empty logit row counts.");
        double worst = 0, maxAbsolute = 0;
        int mismatches = 0;
        for (int r = 0; r < actual.Length; r++)
        {
            if (actual[r].Length != expected[r].Length || actual[r].Length == 0)
                throw new InvalidOperationException("Different or empty vocabulary dimensions.");
            double square = 0, norm = 0;
            int left = 0, right = 0;
            for (int i = 0; i < actual[r].Length; i++)
            {
                double a = actual[r][i], b = expected[r][i];
                if (!double.IsFinite(a) || !double.IsFinite(b))
                    throw new InvalidOperationException("Nonfinite comparison logits.");
                square += (a - b) * (a - b);
                norm += b * b;
                maxAbsolute = Math.Max(maxAbsolute, Math.Abs(a - b));
                if (a > actual[r][left]) left = i;
                if (b > expected[r][right]) right = i;
            }
            worst = Math.Max(worst, Math.Sqrt(square / Math.Max(norm, 1e-30)));
            if (left != right) mismatches++;
        }
        return (worst, maxAbsolute, mismatches);
    }
}
