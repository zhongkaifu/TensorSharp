// Copyright (c) Zhongkai Fu. Licensed under the BSD-3-Clause license.
using System.Diagnostics;
using System.Reflection;
using System.Runtime.InteropServices;
using System.Security.Cryptography;
using System.Text.Json;
using TensorSharp;
using TensorSharp.GGML;
using TensorSharp.Models;
using TensorSharp.Runtime;

var options = new Dictionary<string, string>();
for (int i = 0; i < args.Length; i += 2)
{
    if (i + 1 == args.Length) throw new ArgumentException("Use --name value pairs.");
    options.Add(args[i], args[i + 1]);
}
if (options.GetValueOrDefault("--self-test") == "true") { OracleMath.SelfTest(); return 0; }
string path = Path.GetFullPath(options["--model"]), output = Path.GetFullPath(options["--output"]);
if (Directory.Exists(output)) throw new IOException("Use a fresh output directory.");
Directory.CreateDirectory(output);
string backend = options.GetValueOrDefault("--backend", "cpu");
if (backend is not ("cpu" or "cuda")) throw new ArgumentException("--backend cpu|cuda");
string[] names = options.GetValueOrDefault("--weights", "blk.0.ffn_gate.weight,blk.0.ffn_up.weight").Split(',');
int[] counts = options.GetValueOrDefault("--tokens", "1,8,9,19,38").Split(',').Select(int.Parse).ToArray();
int samples = int.Parse(options.GetValueOrDefault("--rows", "128"));
long maximumBytes = long.Parse(options.GetValueOrDefault("--max-weight-bytes", "134217728"));
if (counts.Length == 0 || counts.Any(n => n is < 1 or > 64) || counts.Distinct().Count() != counts.Length
    || names.Any(string.IsNullOrWhiteSpace) || names.Distinct().Count() != names.Length
    || samples is < 1 or > 4096 || maximumBytes <= 0)
    throw new ArgumentException("Counts 1..64, sampled rows 1..4096 and positive weight limit required.");
var decode = typeof(ModelBase).Assembly.GetType("TensorSharp.Models.ManagedQuantizedOps", true)!
    .GetMethod("DequantizeToFloat32", BindingFlags.Public | BindingFlags.Static, null,
        [typeof(int), typeof(byte[]), typeof(int), typeof(float[]), typeof(int), typeof(long)], null)!
    .CreateDelegate<Action<int, byte[], int, float[], int, long>>();
string Hash(string p) { using var s = File.OpenRead(p); return Convert.ToHexString(SHA256.HashData(s)); }
string BytesHash(ReadOnlySpan<byte> bytes) => Convert.ToHexString(SHA256.HashData(bytes));
var observations = new List<object>();
// Retain source roots if native cleanup cannot establish that borrowed pointers
// are no longer live. Process exit is safer than unpinning after failed cleanup.
var retainedPins = new List<GCHandle>();
string? error = null;
bool shutdown = false;
GgmlContext? context = null;
try
{
    using var file = new GgufFile(path);
    context = new GgmlContext([0], backend == "cuda" ? GgmlBackendType.Cuda : GgmlBackendType.Cpu);
    var allocator = new GgmlAllocator(context, 0);
    GgmlBasicOps.SetDeviceCopyBudget(0);
    foreach (string name in names)
    {
        var info = file.Tensors[name];
        if (info.Shape.Length != 2 || info.Type is not (GgmlTensorType.IQ2_S or GgmlTensorType.IQ3_XXS))
            throw new NotSupportedException($"{name}: only original dense IQ2_S/IQ3_XXS are qualified for this diagnostic.");
        int k = checked((int)info.Shape[0]), m = checked((int)info.Shape[1]), type = (int)info.Type;
        long size = file.GetTensorByteCount(info);
        if (k <= 0 || m <= 0 || size <= 0 || k % 256 != 0 || size % m != 0 || size > maximumBytes || size > int.MaxValue)
            throw new ArgumentException("Original tensor geometry or payload exceeds the explicit diagnostic limit.");
        int rowBytes = checked((int)(size / m));
        byte[] raw = GC.AllocateUninitializedArray<byte>((int)size);
        var pin = GCHandle.Alloc(raw, GCHandleType.Pinned);
        try
        {
            file.ReadTensorDataToNative(info, pin.AddrOfPinnedObject(), size);
            int[] rows = Enumerable.Range(0, Math.Min(samples, m))
                .Select(i => Math.Min(samples, m) == 1 ? 0 : (int)((long)i * (m - 1) / (Math.Min(samples, m) - 1))).Distinct().ToArray();
            float[][] weights = new float[rows.Length][];
            for (int r = 0; r < rows.Length; r++)
            {
                weights[r] = new float[k];
                var native = new float[k];
                decode(type, raw, checked(rows[r] * rowBytes), weights[r], 0, k);
                GgmlGgufTensorDequant.DequantizeToFloat32(type, raw, checked(rows[r] * rowBytes), native, 0, k);
                if (weights[r].Any(x => !float.IsFinite(x)) || !MemoryMarshal.AsBytes(weights[r].AsSpan()).SequenceEqual(MemoryMarshal.AsBytes(native.AsSpan())))
                    throw new InvalidDataException($"Independent/native weight decoding differs: {name}, row {rows[r]}.");
            }
            float[] inputs = new float[checked(k * counts.Max())];
            for (int t = 0; t < counts.Max(); t++) for (int j = 0; j < k; j++)
                inputs[t * k + j] = MathF.Sin((j + 1) * .173f + t * .37f) * (.2f + j % 13 * .21f)
                    + MathF.Cos((j + 1) * .071f - t * .19f) * .41f;
            var weightCases = new List<object>();
            foreach (int n in counts)
            {
                float[] active = inputs.AsSpan(0, checked(k * n)).ToArray();
                using var x = new Tensor(allocator, DType.Float32, n, k);
                Marshal.Copy(active, 0, x.Storage.PtrAtElement(0), active.Length);
                using var y = new Tensor(allocator, DType.Float32, n, m);
                GgmlBasicOps.AddmmQuant(y, x, pin.AddrOfPinnedObject(), type, k, m, size);
                float[] actual = y.GetElementsAsFloat(checked(n * m));
                if (actual.Any(x => !float.IsFinite(x))) throw new ArithmeticException("Nonfinite full projection.");
                var sampled = new double[checked(n * rows.Length)];
                for (int t = 0; t < n; t++) for (int r = 0; r < rows.Length; r++) sampled[t * rows.Length + r] = actual[t * m + rows[r]];
                var metrics = new Dictionary<string, OracleMath.Metrics>
                {
                    ["original-f32"] = OracleMath.Compare(OracleMath.Project(weights, active.Select(v => (double)v).ToArray(), k, n), sampled)
                };
                foreach (string kind in new[] { "cpu-q8-k", "cuda-mmvq-q8-1", "cuda-mmq-d4" })
                    metrics[kind] = OracleMath.Compare(OracleMath.Project(weights, OracleMath.Activation(active, k, kind), k, n), sampled);
                byte[] resultBytes = MemoryMarshal.AsBytes(actual.AsSpan()).ToArray();
                File.WriteAllBytes(Path.Combine(output, $"{name}.n{n}.f32"), resultBytes);
                weightCases.Add(new { Tokens = n, FullElements = actual.Length, OutputSha256 = BytesHash(resultBytes),
                    InputSha256 = BytesHash(MemoryMarshal.AsBytes(active.AsSpan())), Metrics = metrics });
                Console.WriteLine($"{name} {backend} N={n} original-F32 relL2={metrics["original-f32"].RelativeL2:G9}; quantized models: "
                    + string.Join(", ", metrics.Skip(1).Select(p => $"{p.Key}={p.Value.RelativeL2:G9}")));
            }
            observations.Add(new { Name = name, Type = info.Type.ToString(), K = k, M = m, Bytes = size,
                FileOffset = checked(file.DataOffset + (long)info.Offset), RawSha256 = BytesHash(raw),
                SampledRows = rows, SampledDequantizationBitwiseEqual = true, Cases = weightCases });
        }
        finally
        {
            try { GgmlBasicOps.ClearHostBufferCache(); pin.Free(); }
            catch { retainedPins.Add(pin); throw; }
        }
    }
}
catch (Exception ex) { error = ex.ToString(); Console.Error.WriteLine(error); }
finally
{
    try { GgmlBasicOps.ClearHostBufferCache(); context?.ReleasePooledMemory(); GgmlBasicOps.Shutdown(); shutdown = true; }
    catch (Exception ex) { error = (error + "\nCleanup: " + ex).Trim(); }
}
var nativeIdentity = Process.GetCurrentProcess().Modules.Cast<ProcessModule>()
    .Where(m => m.ModuleName.Contains("GgmlOps", StringComparison.OrdinalIgnoreCase))
    .Select(m => new { m.FileName, Sha256 = Hash(m.FileName) }).ToArray();
File.WriteAllText(Path.Combine(output, "report.json"), JsonSerializer.Serialize(new
{
    Completed = error == null && shutdown, Error = error, NativeShutdown = shutdown, Backend = backend,
    Model = path, ModelSha256 = Hash(path), ModelBytes = new FileInfo(path).Length,
    ModelsSha256 = Hash(typeof(ModelBase).Assembly.Location), ProbeSha256 = Hash(Assembly.GetExecutingAssembly().Location),
    Native = nativeIdentity, Observations = observations, RetainedSourcePins = retainedPins.Count,
    Input = "Deterministic synthetic F32, common prefixes across all N; not captured model activations.",
    QuantizerQualification = "Independently evaluated scalar models of pinned ggml quantizers; no device buffer capture or proof of actual CUDA dispatch. Compare all hypotheses without selecting the closest one as ground truth.",
    Scope = "Original complete K/M/dtype in AddmmQuant, full-output finite check, sampled-row independent weight decode and FP64 dots. Completed means diagnosis completed, not a numerical or language-quality pass. Host tensor, output and backend allocations are outside any model memory budget."
}, new JsonSerializerOptions { WriteIndented = true }));
GC.KeepAlive(retainedPins);
return error == null ? 0 : 1;
