// Copyright (c) Zhongkai Fu. Licensed under the BSD-3-Clause license.
using System.Diagnostics;
using System.Security.Cryptography;
using System.Text.Json;
using TensorSharp.Runtime;

if (args.Length < 2) throw new ArgumentException("Use output.json model.gguf [another.gguf ...]");
var results = new List<object>();
foreach (string path in args.Skip(1))
{
    var runs = new List<object>();
    string? reference = null;
    for (int i = 0; i < 8; i++)
    {
        // ABBA order; first pair is labelled warmup, retained in the evidence.
        bool geometry = i % 4 is 1 or 2;
        long before = GC.GetAllocatedBytesForCurrentThread();
        var watch = Stopwatch.StartNew();
        using var file = geometry ? GgufFile.OpenForMemoryPlanning(path) : new GgufFile(path);
        watch.Stop();
        long allocated = GC.GetAllocatedBytesForCurrentThread() - before;
        var data = new
        {
            file.DataOffset,
            Metadata = file.Metadata.Where(kv => !kv.Key.StartsWith("tokenizer.", StringComparison.Ordinal)).OrderBy(kv => kv.Key),
            Tensors = file.Tensors.Values.OrderBy(t => t.Name).Select(t => new { t.Name, t.Type, t.Shape, Region = file.GetTensorFileRegion(t.Name) })
        };
        string fingerprint = Convert.ToHexString(SHA256.HashData(JsonSerializer.SerializeToUtf8Bytes(data)));
        reference ??= fingerprint;
        if (fingerprint != reference) throw new InvalidDataException("Geometry or physical tensor ownership changed.");
        runs.Add(new { Arm = geometry ? "geometry" : "full", Warmup = i < 2,
            Milliseconds = watch.Elapsed.TotalMilliseconds, AllocatedBytes = allocated, GeometrySha256 = fingerprint,
            TokenizerEntries = file.Metadata.Keys.Count(k => k.StartsWith("tokenizer.", StringComparison.Ordinal)),
            file.DataOffset, TensorCount = file.Tensors.Count });
    }
    results.Add(new { Path = Path.GetFullPath(path), Runs = runs });
}
Directory.CreateDirectory(Path.GetDirectoryName(Path.GetFullPath(args[0]))!);
File.WriteAllText(args[0], JsonSerializer.Serialize(new
{
    StartedAssemblySha256 = Convert.ToHexString(SHA256.HashData(File.ReadAllBytes(typeof(GgufFile).Assembly.Location))),
    Results = results,
    Limitations = "Metadata parsing only; file cache is uncontrolled, allocations are managed bytes allocated on the calling thread, not physical RSS or retained tokenizer size. No inference or cold-storage performance claim. First pair is warmup; later pairs alternate ABBA."
}, new JsonSerializerOptions { WriteIndented = true }));
