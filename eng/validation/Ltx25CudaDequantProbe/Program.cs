using System.Diagnostics;
using System.Reflection;
using System.Runtime.InteropServices;
using System.Security.Cryptography;
using System.Text;
using System.Text.Json;
using TensorSharp;
using TensorSharp.Cuda;
using TensorSharp.Cuda.Interop;

var options = new Dictionary<string, string>(StringComparer.Ordinal);
var allowed = new HashSet<string>(["output", "main-ptx", "inner", "columns", "types", "warmup", "repeats", "inspect-module"]);
for (int i = 0; i < args.Length; i += 2)
    if (i + 1 >= args.Length || !args[i].StartsWith("--") || !allowed.Contains(args[i][2..]) ||
        !options.TryAdd(args[i][2..], args[i + 1])) throw new ArgumentException("Expected unique known --option value pairs.");
string Required(string name) => options.GetValueOrDefault(name) ?? throw new ArgumentException("Missing --" + name);
int Number(string name, int fallback) => int.Parse(options.GetValueOrDefault(name, fallback.ToString()), System.Globalization.CultureInfo.InvariantCulture);
string output = Path.GetFullPath(Required("output"));
if (File.Exists(output)) throw new IOException("Refusing to overwrite a report.");
if (options.TryGetValue("inspect-module", out string? inspect))
{
    if (inspect != "1") throw new ArgumentException("--inspect-module accepts1 only.");
    string? resolved = CudaKernels.LocatePtxPath("tensorsharp_ltx25.ptx");
    var diagnostic = new
    {
        completed = true, mode = "module-path inspection only; no CUDA allocator or driver initialized",
        created_utc = DateTimeOffset.UtcNow, module_path = resolved,
        module_sha256 = resolved == null ? null : HashFile(resolved),
        application_base_directory = AppContext.BaseDirectory,
        current_directory = Environment.CurrentDirectory,
        cuda_assembly = typeof(CudaKernels).Assembly.Location,
        cuda_assembly_sha256 = HashFile(typeof(CudaKernels).Assembly.Location),
        tool_assembly = Assembly.GetExecutingAssembly().Location,
        tool_assembly_sha256 = HashFile(Assembly.GetExecutingAssembly().Location),
    };
    Directory.CreateDirectory(Path.GetDirectoryName(output)!);
    string diagnosticJson = JsonSerializer.Serialize(diagnostic, new JsonSerializerOptions { WriteIndented = true });
    File.WriteAllText(output, diagnosticJson + "\n"); Console.WriteLine(diagnosticJson); return 0;
}
string mainPtx = Path.GetFullPath(Required("main-ptx"));
int inner = Number("inner", 4096), columns = Number("columns", 16384);
int warmup = Number("warmup", 2), repeats = Number("repeats", 10);
int[] types = options.GetValueOrDefault("types", "12,13,14").Split(',').Select(int.Parse).ToArray();
if (inner <= 0 || inner % 256 != 0 || columns <= 0 || warmup < 0 || repeats < 1 || types.Length == 0 ||
    types.Distinct().Count() != types.Length || types.Any(type => type is not (12 or 13 or 14)))
    throw new ArgumentException("Require aligned positive dimensions, nonnegative warmup, positive repeats and distinct K-quant types12/13/14.");
int elements = checked(inner * columns), halfBytes = checked(elements * sizeof(ushort));
Directory.CreateDirectory(Path.GetDirectoryName(output)!);
var cases = new List<object>();
var report = new Dictionary<string, object?>
{
    ["completed"] = false, ["created_utc"] = DateTimeOffset.UtcNow,
    ["inner"] = inner, ["columns"] = columns, ["warmup"] = warmup, ["repeats"] = repeats,
    ["scope"] = "Synthetic resident Q4_K/Q5_K/Q6_K weight-to-F16 conversion only. Compares unchanged primary PTX generic export with the optional owned blockwise converter; exact half-byte parity is required. CUDA-event durations alternate route order and include queued conversion only; synchronized wall times include dispatch, owned validation/cache lookup and waiting. Excludes host upload, allocation, output readback, GEMM, model execution and end-to-end generation.",
    ["managed_binaries"] = new[] { Assembly.GetExecutingAssembly(), typeof(CudaAllocator).Assembly, typeof(Tensor).Assembly }
        .Select(assembly => new { name = assembly.GetName().Name, path = assembly.Location, sha256 = HashFile(assembly.Location) }).ToArray(),
    ["cases"] = cases,
};
var json = new JsonSerializerOptions { WriteIndented = true };
void Save() => File.WriteAllText(output, JsonSerializer.Serialize(report, json) + "\n");
string Hash(byte[] bytes) => Convert.ToHexStringLower(SHA256.HashData(bytes));
string HashFile(string path) { using var stream = File.OpenRead(path); return Convert.ToHexStringLower(SHA256.HashData(stream)); }
try
{
    report["primary_ptx"] = new { path = mainPtx, sha256 = HashFile(mainPtx), export = "ts_dequant_weight_f16" };
    using var allocator = new CudaAllocator();
    allocator.Context.MakeCurrent();
    var deviceName = new byte[256];
    CudaDriverApi.cuDeviceGet(out int device, 0).ThrowOnError();
    CudaDriverApi.cuDeviceGetName(deviceName, deviceName.Length, device).ThrowOnError();
    report["device"] = Encoding.UTF8.GetString(deviceName).TrimEnd('\0');
    using var primary = CudaModule.LoadFromFile(mainPtx);
    IntPtr genericKernel = primary.GetFunction("ts_dequant_weight_f16");
    using var owned = new CudaLtx25ProjectionSession(allocator, requireNative: true, enableF16: true);
    if (!owned.F16DequantAvailable) throw new InvalidOperationException("Owned F16 converter is unavailable.");
    report["owned_ptx"] = new { path = owned.ModulePath, sha256 = HashFile(owned.ModulePath), export = "ts_ltx25_dequant_k_f16" };
    CudaDriverApi.cuEventCreate(out IntPtr startEvent, 0).ThrowOnError();
    IntPtr endEvent = IntPtr.Zero;
    try
    {
        CudaDriverApi.cuEventCreate(out endEvent, 0).ThrowOnError();
        foreach (int type in types)
        {
            byte[] packed = Weights(type, inner, columns);
            IntPtr host = Marshal.AllocHGlobal(packed.Length);
            try
            {
                Marshal.Copy(packed, 0, host, packed.Length);
                CudaQuantizedOps.PreloadQuantizedWeight(allocator, host, host, type, inner, columns, packed.Length);
                if (!CudaQuantizedOps.TryGetResidentDevicePtr(allocator, host, out IntPtr weights, out int storedType) || storedType != type)
                    throw new InvalidOperationException("Resident weights differ from requested raw format.");
                using var genericOutput = new Tensor(allocator, DType.Float16, columns, inner);
                using var ownedOutput = new Tensor(allocator, DType.Float16, columns, inner);
                void Run(bool optimized)
                {
                    if (optimized)
                    {
                        if (!owned.TryDequantizeF16(ownedOutput, host, type, inner, columns, packed.Length))
                            throw new InvalidOperationException("Owned converter refused a supported shape.");
                    }
                    else LaunchGeneric(genericKernel, weights, genericOutput, type, inner, elements, allocator);
                }
                for (int i = 0; i < warmup; ++i) { Run(false); Run(true); }
                allocator.Synchronize();
                Run(false); Run(true); allocator.Synchronize();
                byte[] Bytes(Tensor tensor)
                {
                    var bytes = new byte[halfBytes];
                    Marshal.Copy(tensor.Storage.PtrAtElement(0), bytes, 0, bytes.Length);
                    for (int i = 0; i < bytes.Length; i += 2)
                        if ((bytes[i + 1] & 0x7c) == 0x7c) throw new InvalidOperationException("Nonfinite half output.");
                    return bytes;
                }
                byte[] expected = Bytes(genericOutput), actual = Bytes(ownedOutput);
                if (!expected.AsSpan().SequenceEqual(actual)) throw new InvalidOperationException("Owned converter is not bitwise equal to generic primary PTX.");
                var genericGpu = new List<double>(); var ownedGpu = new List<double>();
                var genericWall = new List<double>(); var ownedWall = new List<double>();
                void Time(bool optimized)
                {
                    var wall = Stopwatch.StartNew();
                    CudaDriverApi.cuEventRecord(startEvent, allocator.Stream.Handle).ThrowOnError();
                    Run(optimized);
                    CudaDriverApi.cuEventRecord(endEvent, allocator.Stream.Handle).ThrowOnError();
                    CudaDriverApi.cuEventSynchronize(endEvent).ThrowOnError();
                    CudaDriverApi.cuEventElapsedTime(out float milliseconds, startEvent, endEvent).ThrowOnError();
                    wall.Stop();
                    (optimized ? ownedGpu : genericGpu).Add(milliseconds);
                    (optimized ? ownedWall : genericWall).Add(wall.Elapsed.TotalMilliseconds);
                }
                for (int round = 0; round < repeats; ++round)
                {
                    Time(round % 2 != 0); Time(round % 2 == 0);
                }
                // Recheck every byte after measured launches as well.
                if (!Bytes(genericOutput).AsSpan().SequenceEqual(Bytes(ownedOutput)))
                    throw new InvalidOperationException("Conversion outputs diverged after timed repeats.");
                double oldMedian = Median(genericGpu), newMedian = Median(ownedGpu);
                cases.Add(new { type, raw_weight_bytes = packed.Length, weight_sha256 = Hash(packed), output_half_bytes = halfBytes,
                    output_sha256 = Hash(expected), exact_half_parity = true, generic_gpu_ms = genericGpu, owned_gpu_ms = ownedGpu,
                    generic_gpu_median_ms = oldMedian, owned_gpu_median_ms = newMedian, gpu_speedup = oldMedian / newMedian,
                    generic_wall_ms = genericWall, owned_wall_ms = ownedWall,
                    generic_wall_median_ms = Median(genericWall), owned_wall_median_ms = Median(ownedWall) });
                Save();
            }
            finally { allocator.Synchronize(); CudaQuantizedOps.ReleaseQuantizedWeight(allocator, host); Marshal.FreeHGlobal(host); }
        }
        report["owned_converter_executions"] = owned.ExecutedF16Dequantizations;
        if (owned.ExecutedF16Dequantizations != (long)types.Length * (warmup + repeats + 1))
            throw new InvalidOperationException("Unexpected owned converter execution count.");
        report["completed"] = true;
        Save();
        Console.WriteLine(JsonSerializer.Serialize(report, json));
        return 0;
    }
    finally { allocator.Synchronize(); CudaDriverApi.cuEventDestroy(startEvent); if (endEvent != IntPtr.Zero) CudaDriverApi.cuEventDestroy(endEvent); }
}
catch (Exception error)
{
    report["error_type"] = error.GetType().FullName; report["error"] = error.Message;
    Save(); Console.Error.WriteLine(error); return 1;
}

static double Median(List<double> values)
{
    double[] sorted = values.Order().ToArray();
    return (sorted[(sorted.Length - 1) / 2] + sorted[sorted.Length / 2]) / 2;
}

static unsafe void LaunchGeneric(IntPtr kernel, IntPtr weights, Tensor output, int type, int inner, long total, CudaAllocator allocator)
{
    var storage = (CudaStorage)output.Storage;
    IntPtr target = storage.DevicePtrAtElement(output.StorageOffset);
    void** arguments = stackalloc void*[5] { &weights, &target, &type, &inner, &total };
    CudaDriverApi.cuLaunchKernel(kernel, checked((uint)((total + 255) / 256)), 1, 1,
        256, 1, 1, 0, allocator.Stream.Handle, (IntPtr)arguments, IntPtr.Zero).ThrowOnError();
    storage.MarkDeviceModified();
}

static byte[] Weights(int type, int inner, int columns)
{
    int bytesPerBlock = type == 12 ? 144 : type == 13 ? 176 : 210;
    int blocks = checked(inner / 256 * columns);
    var bytes = new byte[checked(blocks * bytesPerBlock)];
    new Random(20261013 + type).NextBytes(bytes);
    for (int block = 0; block < blocks; ++block)
    {
        int offset = block * bytesPerBlock;
        ushort scale = BitConverter.HalfToUInt16Bits((System.Half)(.0009765625f * (1 + block % 7)));
        int position = offset + (type == 14 ? 208 : 0);
        bytes[position] = (byte)scale; bytes[position + 1] = (byte)(scale >> 8);
        if (type != 14)
        {
            ushort minimum = BitConverter.HalfToUInt16Bits((System.Half).00048828125f);
            bytes[offset + 2] = (byte)minimum; bytes[offset + 3] = (byte)(minimum >> 8);
        }
    }
    return bytes;
}
