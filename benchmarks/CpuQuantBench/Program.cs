// Microbenchmark for the pure-C# managed quantized matmul (ManagedQuantizedOps).
// Measures effective weight-read bandwidth (GB/s) per quant type for the
// decode-shaped matmul (rowCount=1) and a small prefill batch (rowCount=4),
// and checks every result against the dequantize-then-dot reference.
//
//   dotnet run -c Release --project benchmarks/CpuQuantBench            # all quants
//   dotnet run -c Release --project benchmarks/CpuQuantBench q4k        # one quant
//   TENSORSHARP_CPU_NO_SIMD_KQUANT=1 dotnet run -c Release ... q4k      # scalar A/B
//
// Decode tok/s for a model is ~ (bytes read per token) / (GB/s here), so the
// GB/s number is the lever this backend optimises.
using System.Diagnostics;
using TensorSharp.Models;
using TensorSharp.Runtime;

static byte[] BuildRandom(Random rng, GgmlTensorType type, int outDim, int inDim)
{
    int blockBytes = (int)GgufFile.GetTypeSize(type);
    int blockSize = (int)GgufFile.GetBlockSize(type);
    int blocksPerRow = inDim / blockSize;
    byte[] raw = new byte[(long)outDim * blocksPerRow * blockBytes];
    int o = 0;
    for (int r = 0; r < outDim; r++)
    {
        for (int b = 0; b < blocksPerRow; b++)
        {
            int sb = o;
            switch (type)
            {
                case GgmlTensorType.Q4_0:
                case GgmlTensorType.Q8_0:
                    WriteHalf(raw, sb, 0.02f + 0.03f * (float)rng.NextDouble());
                    for (int i = 0; i < blockBytes - 2; i++) raw[sb + 2 + i] = (byte)rng.Next(0, 256);
                    break;
                case GgmlTensorType.Q4_K:
                case GgmlTensorType.Q5_K:
                    WriteHalf(raw, sb, 0.02f + 0.03f * (float)rng.NextDouble());
                    WriteHalf(raw, sb + 2, 0.01f + 0.02f * (float)rng.NextDouble());
                    for (int i = 0; i < blockBytes - 4; i++) raw[sb + 4 + i] = (byte)rng.Next(0, 256);
                    break;
                case GgmlTensorType.Q6_K:
                    for (int i = 0; i < blockBytes - 2; i++) raw[sb + i] = (byte)rng.Next(0, 256);
                    WriteHalf(raw, sb + blockBytes - 2, 0.01f + 0.02f * (float)rng.NextDouble());
                    break;
                default:
                    throw new NotSupportedException(type.ToString());
            }
            o += blockBytes;
        }
    }
    return raw;

    static void WriteHalf(byte[] buf, int off, float val)
    {
        ushort bits = BitConverter.HalfToUInt16Bits((Half)val);
        buf[off] = (byte)bits; buf[off + 1] = (byte)(bits >> 8);
    }
}

static float Reference(byte[] weights, GgmlTensorType type, int col, int inDim, float[] input, int inOff, float[] wrow)
{
    long rowBytes = NativeDequant.RowSize((int)type, inDim);
    NativeDequant.DequantizeToFloat32((int)type, weights, (int)(col * rowBytes), wrow, 0, inDim);
    float s = 0f;
    for (int i = 0; i < inDim; i++) s += wrow[i] * input[inOff + i];
    return s;
}

static (double gbps, double msPerCall, float maxRelErr) Bench(
    GgmlTensorType type, int inDim, int outDim, int rows, int iters)
{
    var rng = new Random(12345 + (int)type);
    byte[] weights = BuildRandom(rng, type, outDim, inDim);
    float[] input = new float[rows * inDim];
    for (int i = 0; i < input.Length; i++) input[i] = 0.08f * MathF.Sin(i * 0.011f);
    float[] output = new float[rows * outDim];

    long weightBytes = (long)NativeDequant.RowSize((int)type, inDim) * outDim;

    // correctness on a few columns
    float[] wrow = new float[inDim];
    float maxRel = 0f, refMag = 1e-6f;
    bool ok = ManagedQuantizedOps.TryAddmmQuantizedToFloat32(
        (int)type, weights, 0, inDim, outDim, input, 0, inDim, rows, output, 0, outDim);
    if (!ok) throw new Exception($"{type}: TryAddmm returned false");
    int[] checkCols = { 0, outDim / 3, outDim / 2, outDim - 1 };
    foreach (int c in checkCols)
    {
        float exp = Reference(weights, type, c, inDim, input, 0, wrow);
        refMag = MathF.Max(refMag, MathF.Abs(exp));
        maxRel = MathF.Max(maxRel, MathF.Abs(exp - output[c]));
    }
    maxRel /= refMag;

    // warmup
    for (int i = 0; i < 3; i++)
        ManagedQuantizedOps.TryAddmmQuantizedToFloat32(
            (int)type, weights, 0, inDim, outDim, input, 0, inDim, rows, output, 0, outDim);

    var sw = Stopwatch.StartNew();
    for (int i = 0; i < iters; i++)
        ManagedQuantizedOps.TryAddmmQuantizedToFloat32(
            (int)type, weights, 0, inDim, outDim, input, 0, inDim, rows, output, 0, outDim);
    sw.Stop();

    double seconds = sw.Elapsed.TotalSeconds;
    double gbps = (double)weightBytes * iters / seconds / (1024.0 * 1024 * 1024);
    double msPerCall = sw.Elapsed.TotalMilliseconds / iters;
    return (gbps, msPerCall, maxRel);
}

var quants = new (string name, GgmlTensorType type)[]
{
    ("q4_0", GgmlTensorType.Q4_0),
    ("q8_0", GgmlTensorType.Q8_0),
    ("q4_k", GgmlTensorType.Q4_K),
    ("q5_k", GgmlTensorType.Q5_K),
    ("q6_k", GgmlTensorType.Q6_K),
};

string filter = args.Length > 0 ? args[0].ToLowerInvariant() : null;
int inDim = 4096, outDim = 4096, iters = 200;

bool noK = Environment.GetEnvironmentVariable("TENSORSHARP_CPU_NO_SIMD_KQUANT") == "1";
bool noQ40 = Environment.GetEnvironmentVariable("TENSORSHARP_CPU_NO_SIMD_Q40") == "1";
Console.WriteLine($"cores={Environment.ProcessorCount}  matmul={inDim}x{outDim}  " +
    $"NO_SIMD_KQUANT={noK} NO_SIMD_Q40={noQ40}");
Console.WriteLine($"{"quant",-6} {"rows",4}  {"GB/s",8}  {"ms/call",9}  {"relErr",9}");
foreach (var (name, type) in quants)
{
    if (filter != null && name != filter) continue;
    foreach (int rows in new[] { 1, 4 })
    {
        var (gbps, ms, err) = Bench(type, inDim, outDim, rows, iters);
        Console.WriteLine($"{name,-6} {rows,4}  {gbps,8:F1}  {ms,9:F3}  {err,9:E2}");
    }
}
