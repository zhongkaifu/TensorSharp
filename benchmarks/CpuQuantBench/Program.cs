// Microbenchmark for the pure-C# managed quantized matmul (ManagedQuantizedOps).
//
//   dotnet run -c Release --project benchmarks/CpuQuantBench                 # GEMM table, all types
//   dotnet run -c Release --project benchmarks/CpuQuantBench gemm q4_k       # one type
//   dotnet run -c Release --project benchmarks/CpuQuantBench gemm all 256    # shapes with M <= 256
//   dotnet run -c Release --project benchmarks/CpuQuantBench gemm decode     # DRAM-bound M = 1 matvecs [type]
//   dotnet run -c Release --project benchmarks/CpuQuantBench batch           # MoE: many jobs in one call
//   dotnet run -c Release --project benchmarks/CpuQuantBench perrow [quant]  # per-row path GB/s table
//   dotnet run -c Release --project benchmarks/CpuQuantBench dg <gguf> cpu|ggml_cpu [reps] [width] [multi|step]
//   dotnet run -c Release --project benchmarks/CpuQuantBench shapes q4_k,q8_0 1,512 lfm3 [reps]  # RESULT lines (ab.py)
//
// "gemm" runs every (M, K, N) shape through the per-row path (PerRow) and
// the multi-row GEMM (AVX-512 and AVX2 kernels) in ONE process, checks the GEMM
// results against the per-row ones, and reports GOPS = 2*M*N*K / s (plus weight
// GB/s at M = 1, where the matmul is bandwidth-bound). Dequant-only types
// (Q3_K, IQ*, BF16/F16) compare DequantMatMulColumns with the float panel.
// With TS_CPU_DISABLE_AVX512=1 the PerRow column is the per-row path's AVX2
// form as well (the GEMM columns always force their kernel set).
//
// "gemm decode" rotates each shape over enough weight copies (>= 160 MB) that
// no call finds its weights in the 24 MB L3, as a decoded token does. Its
// results decided that single rows take the GEMM for every type (see
// QGemmMinRows in ManagedQuantGemm.cs).
//
// "perrow" is the effective weight-read bandwidth of the
// per-row path per quant type for rowCount 1 and 4, next to what the default
// (Auto) routing does with the same call. Decode tok/s for a model is
// ~ (bytes read per token) / (GB/s here).
//
// "shapes TYPES ROWS SHAPES [REPS]" times the default routing (what a model runs) at the given
// row counts and weight shapes ("lfm3" = LFM3-1.7B's products, or KxN,KxN,...) and prints
// eng/lfm3/bench/ab.py's "RESULT name median min max" lines: weight GB/s at one row (weights
// rotated over >= 384 MB of copies so every call reads DRAM, as a decoded token does), GOPS
// (2*M*N*K/s) above one row. An A/B builds each arm into its own directory and alternates them.
//
// "dg" profiles DiffusionGemma end to end (structured read or one 256-canvas
// diffusion step) with the model's per-stage forward timing; run it with
// TS_CPU_QGEMM_VERIFY=1 to recompute every GEMM of the forward through the
// per-row path and print the worst difference.
using System.Diagnostics;
using TensorSharp.Models;
using TensorSharp.Runtime;
using QGemmIsa = TensorSharp.Models.ManagedQuantizedOps.QGemmIsa;

string mode = args.Length > 0 ? args[0].ToLowerInvariant() : "gemm";
if (mode == "dg")
    return RunDg(args.Skip(1).ToArray());

NativeDequant.PreferManaged = true;   // dequant stays in managed code, as on the cpu backend
switch (mode)
{
    case "perrow":
        RunPerRowTable(args.Length > 1 ? args[1].ToLowerInvariant() : null);
        return 0;
    case "batch":
        RunBatchTable();
        return 0;
    case "shapes":
        return RunShapes(args.Skip(1).ToArray());
    case "gemm":
    {
        string typeFilter = args.Length > 1 ? args[1].ToLowerInvariant() : "all";
        bool decode = typeFilter == "decode";
        int maxM = args.Length > 2 && !decode ? int.Parse(args[2]) : int.MaxValue;
        bool skipPerRowBig = Environment.GetEnvironmentVariable("QBENCH_SKIP_PERROW_BIG") == "1";
        RunGemmTable(typeFilter, maxM, skipPerRowBig, decode && args.Length > 2 ? args[2].ToLowerInvariant() : null);
        return 0;
    }
    default:
        Console.Error.WriteLine($"unknown mode '{args[0]}'. Modes: gemm [type|all|decode] [maxM] | batch | " +
                                "perrow [quant] | dg <gguf> cpu|ggml_cpu [reps] [width] [multi|step]");
        return 2;
}

static void RunGemmTable(string typeFilter, int maxM, bool skipPerRowBig, string decodeType)
{
    var dg = new (int m, int k, int n)[] { (1, 2816, 1408), (2, 2816, 1408), (4, 2816, 1408), (70, 2816, 2112), (70, 2816, 4096) };
    var dgDown = new (int m, int k, int n)[] { (1, 704, 2816), (2, 704, 2816), (8, 704, 2816), (70, 2112, 2816) };
    var dit = new (int m, int k, int n)[]
    {
        (256, 4096, 4096), (256, 4096, 24576), (256, 12288, 4096), (1024, 4096, 4096), (4096, 4096, 4096),
    };
    // Dequant-only types: the float panel against DequantMatMulColumns at the
    // shapes where rebuilding the panel per row block used to lose (K = 12288).
    var dequantOnly = new (int m, int k, int n)[] { (1, 12288, 4096), (16, 12288, 4096), (64, 12288, 4096), (256, 12288, 4096), (256, 4096, 4096) };
    var plan = new List<(GgmlTensorType type, (int m, int k, int n)[] shapes)>
    {
        (GgmlTensorType.Q4_K, dg.Concat(dit).ToArray()),
        (GgmlTensorType.Q6_K, dg.Concat(dit).ToArray()),
        (GgmlTensorType.Q5_K, new[] { (1, 2816, 1408), (4, 2816, 1408), (70, 2816, 2112), (256, 4096, 4096) }),
        (GgmlTensorType.Q8_0, dg.Concat(dgDown).Concat(dit).ToArray()),
        (GgmlTensorType.Q5_0, dg.Concat(dgDown).Concat(dit).ToArray()),
        (GgmlTensorType.Q4_0, new[] { (1, 4096, 4096), (8, 704, 2816), (256, 4096, 4096) }),
        (GgmlTensorType.BF16, new[] { (1, 4096, 4096), (256, 4096, 4096), (256, 64, 4096) }),
        (GgmlTensorType.F16, new[] { (256, 4096, 4096) }),
        (GgmlTensorType.Q3_K, dequantOnly),
        (GgmlTensorType.IQ4_XS, dequantOnly),
        (GgmlTensorType.IQ2_XXS, dequantOnly),
    };
    // "decode": single-row shapes whose weights do not fit the L3, rotated
    // over several copies so every call streams them from DRAM. The third
    // argument then filters the type (`gemm decode q8_0`).
    long rotateBytes = 0;
    if (typeFilter == "decode")
    {
        var big = new[] { (1, 4096, 14336), (1, 14336, 4096) };
        plan = new List<(GgmlTensorType type, (int m, int k, int n)[] shapes)>
        {
            (GgmlTensorType.Q4_K, big), (GgmlTensorType.Q6_K, big), (GgmlTensorType.Q5_K, big),
            (GgmlTensorType.Q8_0, big), (GgmlTensorType.Q5_0, big), (GgmlTensorType.Q4_0, big),
            (GgmlTensorType.BF16, big),
        };
        typeFilter = decodeType ?? "all";
        maxM = int.MaxValue;
        rotateBytes = 160L << 20;
    }

    Console.WriteLine($"cores={Environment.ProcessorCount} avx512={ManagedQuantizedOps.QGemmAvx512Supported} " +
                      $"avx2={ManagedQuantizedOps.QGemmAvx2Supported} TS_CPU_DISABLE_AVX512=" +
                      $"{Environment.GetEnvironmentVariable("TS_CPU_DISABLE_AVX512") ?? "-"}" +
                      (rotateBytes > 0 ? $" rotating >= {rotateBytes >> 20} MB of weights per shape" : ""));
    Console.WriteLine($"{"type",-7} {"M",5} {"K",6} {"N",6} | {"perRow ms",10} {"GOPS",7} | {"avx512 ms",10} {"GOPS",7} {"x",6} | " +
                      $"{"avx2 ms",10} {"GOPS",7} {"x",6} | {"relErr512",9} {"relErr2",9} | GB/s perRow/512/avx2");
    foreach (var (type, shapes) in plan)
    {
        if (typeFilter != "all" && !type.ToString().Equals(typeFilter.Replace("_", ""), StringComparison.OrdinalIgnoreCase)
            && !type.ToString().Equals(typeFilter, StringComparison.OrdinalIgnoreCase))
            continue;
        foreach (var (m, k, n) in shapes)
        {
            if (m > maxM) continue;
            RunShape(type, m, k, n, skipPerRowBig && (long)m * n * k > 20_000_000_000L, rotateBytes);
        }
    }
}

static unsafe void RunShape(GgmlTensorType type, int m, int k, int n, bool skipPerRow, long rotateBytes)
{
    var rng = new Random(1234 + (int)type * 17 + k + n);
    long weightBytes = NativeDequant.RowSize((int)type, k) * n;
    int copies = rotateBytes > 0 ? (int)Math.Max(2, (rotateBytes + weightBytes - 1) / weightBytes) : 1;
    var weights = new byte[copies][];
    for (int c = 0; c < copies; c++)
        weights[c] = BuildRandom(rng, type, n, k);
    float[] input = new float[(long)m * k];
    for (long i = 0; i < input.Length; i++)
        input[i] = 0.08f * MathF.Sin(i * 0.011f) + 0.02f * (float)(rng.NextDouble() - 0.5);
    float[] outPerRow = new float[(long)m * n];
    float[] out512 = new float[(long)m * n];
    float[] out2 = new float[(long)m * n];
    double flops = 2.0 * m * n * k;

    var handles = weights.Select(w => System.Runtime.InteropServices.GCHandle.Alloc(w, System.Runtime.InteropServices.GCHandleType.Pinned)).ToArray();
    try
    {
        fixed (float* x = input)
        fixed (float* oL = outPerRow)
        fixed (float* o5 = out512)
        fixed (float* o2 = out2)
        {
            float* xp = x;
            int next = 0;
            // Every call takes the next copy; the error check below uses copy 0
            // (the last call of each timing loop may have used another copy).
            void Run(float* o, QGemmIsa isa, int copy = -1)
            {
                int c = copy >= 0 ? copy : next++ % copies;
                ManagedQuantizedOps.AddmmQuantizedToFloat32((int)type, handles[c].AddrOfPinnedObject(), k, n, xp, k, m, o, n, null, isa);
            }

            float* oLp = oL, o5p = o5, o2p = o2;
            double tL = skipPerRow ? double.NaN : Time(() => Run(oLp, QGemmIsa.PerRow), flops);
            double t5 = Time(() => Run(o5p, QGemmIsa.Avx512), flops);
            double t2 = Time(() => Run(o2p, QGemmIsa.Avx2), flops);
            if (!skipPerRow) Run(oLp, QGemmIsa.PerRow, 0);
            Run(o5p, QGemmIsa.Avx512, 0);
            Run(o2p, QGemmIsa.Avx2, 0);
            if (skipPerRow)
                outPerRow.AsSpan().Clear();
            float e5 = skipPerRow ? float.NaN : RelErr(outPerRow, out512);
            float e2 = RelErr(skipPerRow ? out512 : outPerRow, out2);
            string gbs = m == 1
                ? $"{weightBytes / tL / 1e9,6:F1} /{weightBytes / t5 / 1e9,6:F1} /{weightBytes / t2 / 1e9,6:F1}"
                : "";
            Console.WriteLine($"{Name(type),-7} {m,5} {k,6} {n,6} | {tL * 1e3,10:F3} {flops / tL / 1e9,7:F1} | " +
                              $"{t5 * 1e3,10:F3} {flops / t5 / 1e9,7:F1} {tL / t5,6:F2} | " +
                              $"{t2 * 1e3,10:F3} {flops / t2 / 1e9,7:F1} {tL / t2,6:F2} | {e5,9:E2} {e2,9:E2} | {gbs}");
        }
    }
    finally
    {
        foreach (var h in handles) h.Free();
    }
}

static string Name(GgmlTensorType t) => t.ToString().ToLowerInvariant();

// "batch": TryAddmmQuantizedBatch the way an MoE layer uses it.
//   prefill: 128 expert jobs in one call, each with 1..16 routed rows
//     (DiffusionGemma's shapes: gate_up 2816 -> 1408 Q4_K, down 704 -> 2816 Q5_0 / Q8_0);
//   decode: 8 jobs of ONE row (top-8 routing of a single token) - gate_up jobs share
//     one input, down jobs each have their own.
// Per-row batch vs the default routing (Auto) and the forced kernel sets.
// "quant ms" is the serial cost of quantizing the batch's rows (what the parallel
// quantization dispatch of a prefill-sized batch saves).
static unsafe void RunBatchTable()
{
    Console.WriteLine($"{"case",-8} {"type",-5} {"K",6} {"N",6} {"jobs",5} {"rows",6} | {"perRow ms",10} | {"auto ms",9} {"x",6} | " +
                      $"{"avx512 ms",10} {"x",6} | {"avx2 ms",9} {"x",6} | {"relErrAuto",10} | {"quant ms",8}");
    foreach (var (type, k, n) in new[] { (GgmlTensorType.Q4_K, 2816, 1408), (GgmlTensorType.Q5_0, 704, 2816), (GgmlTensorType.Q8_0, 704, 2816) })
        RunBatch("prefill", type, k, n, 128, rng => 1 + rng.Next(0, 16), sharedInput: false);
    foreach (var (type, k, n) in new[]
             {
                 (GgmlTensorType.Q4_K, 2816, 1408), (GgmlTensorType.Q4_0, 2816, 1408), (GgmlTensorType.Q8_0, 2816, 1408),
                 (GgmlTensorType.Q5_0, 704, 2816), (GgmlTensorType.Q8_0, 704, 2816), (GgmlTensorType.Q4_0, 704, 2816),
             })
        RunBatch("decode", type, k, n, 8, _ => 1, sharedInput: k == 2816);
}

static unsafe void RunBatch(string label, GgmlTensorType type, int k, int n, int jobs, Func<Random, int> rowsOf, bool sharedInput)
{
    var rng = new Random(7 + (int)type + k);
    int[] rows = new int[jobs];
    int totalRows = 0;
    for (int j = 0; j < jobs; j++) { rows[j] = rowsOf(rng); totalRows += rows[j]; }
    var weights = new byte[jobs][];
    var inputs = new float[jobs][];
    var outs = new float[4][][];
    for (int j = 0; j < jobs; j++)
    {
        weights[j] = BuildRandom(rng, type, n, k);
        if (sharedInput && j > 0)
        {
            inputs[j] = inputs[0];
            continue;
        }
        inputs[j] = new float[rows[j] * k];
        for (int i = 0; i < inputs[j].Length; i++) inputs[j][i] = (float)(rng.NextDouble() - 0.5) * 0.3f;
    }
    for (int v = 0; v < 4; v++)
    {
        outs[v] = new float[jobs][];
        for (int j = 0; j < jobs; j++) outs[v][j] = new float[rows[j] * n];
    }
    var handles = new List<System.Runtime.InteropServices.GCHandle>();
    IntPtr Pin(object o)
    {
        var h = System.Runtime.InteropServices.GCHandle.Alloc(o, System.Runtime.InteropServices.GCHandleType.Pinned);
        handles.Add(h);
        return h.AddrOfPinnedObject();
    }
    ManagedQuantizedOps.QuantMatMulJob[] Jobs(float[][] outputs)
    {
        var list = new ManagedQuantizedOps.QuantMatMulJob[jobs];
        for (int j = 0; j < jobs; j++)
            list[j] = new ManagedQuantizedOps.QuantMatMulJob(Pin(weights[j]), Pin(inputs[j]), Pin(outputs[j]), n, rows[j], n);
        return list;
    }
    var jl = Jobs(outs[0]); var ja = Jobs(outs[1]); var j5 = Jobs(outs[2]); var j2 = Jobs(outs[3]);
    double flops = 2.0 * totalRows * n * k;
    double tL = Time(() => ManagedQuantizedOps.TryAddmmQuantizedBatch((int)type, k, k, jl, null, QGemmIsa.PerRow), flops);
    double tA = Time(() => ManagedQuantizedOps.TryAddmmQuantizedBatch((int)type, k, k, ja, null, QGemmIsa.Auto), flops);
    double t5 = Time(() => ManagedQuantizedOps.TryAddmmQuantizedBatch((int)type, k, k, j5, null, QGemmIsa.Avx512), flops);
    double t2 = Time(() => ManagedQuantizedOps.TryAddmmQuantizedBatch((int)type, k, k, j2, null, QGemmIsa.Avx2), flops);
    float err = RelErr(outs[0].SelectMany(a => a).ToArray(), outs[1].SelectMany(a => a).ToArray());

    // Serial quantization of every routed row into the GEMM layout.
    int actBytes = ManagedQuantizedOps.QGemmActivationRowBytes(type, k);
    byte[] act = new byte[actBytes + 64];
    double tQ = Time(() =>
    {
        fixed (byte* a = act)
            for (int j = 0; j < jobs; j++)
            {
                if (sharedInput && j > 0) break;
                fixed (float* x = inputs[j])
                    for (int r = 0; r < rows[j]; r++)
                        ManagedQuantizedOps.QGemmQuantizeActivationRow(type, x + (long)r * k, a, k);
            }
    }, 0);
    Console.WriteLine($"{label,-8} {Name(type),-5} {k,6} {n,6} {jobs,5} {totalRows,6} | {tL * 1e3,10:F3} | {tA * 1e3,9:F3} {tL / tA,6:F2} | " +
                      $"{t5 * 1e3,10:F3} {tL / t5,6:F2} | {t2 * 1e3,9:F3} {tL / t2,6:F2} | {err,10:E2} | {tQ * 1e3,8:F3}");
    foreach (var h in handles) h.Free();
}

// Median-of-reps wall time: at least 3 reps and ~0.6 s of runtime, after
// ~0.3 s of warmup (enough calls for tiered JIT to reach its optimized code).
static double Time(Action run, double flops)
{
    var warm = Stopwatch.StartNew();
    do run(); while (warm.Elapsed.TotalSeconds < 0.3);
    var times = new List<double>();
    var total = Stopwatch.StartNew();
    int minReps = flops > 5e10 ? 2 : 3;
    while (times.Count < minReps || (total.Elapsed.TotalSeconds < 0.6 && times.Count < 200))
    {
        var sw = Stopwatch.StartNew();
        run();
        times.Add(sw.Elapsed.TotalSeconds);
    }
    times.Sort();
    return times[times.Count / 2];
}

// max |a - b| / max |a| over the whole output.
static float RelErr(float[] reference, float[] actual)
{
    float maxRef = 1e-20f, maxDiff = 0f;
    for (long i = 0; i < reference.Length; i++)
    {
        maxRef = MathF.Max(maxRef, MathF.Abs(reference[i]));
        maxDiff = MathF.Max(maxDiff, MathF.Abs(reference[i] - actual[i]));
    }
    return maxDiff / maxRef;
}

static byte[] BuildRandom(Random rng, GgmlTensorType type, int outDim, int inDim)
{
    int blockBytes = (int)GgufFile.GetTypeSize(type);
    int blockSize = (int)GgufFile.GetBlockSize(type);
    int blocksPerRow = inDim / blockSize;
    byte[] raw = new byte[(long)outDim * blocksPerRow * blockBytes];
    if (type is GgmlTensorType.BF16 or GgmlTensorType.F16)
    {
        for (long i = 0; i < raw.Length; i += 2)
        {
            float v = 0.05f * (float)(rng.NextDouble() - 0.5);
            ushort bits = type == GgmlTensorType.F16
                ? BitConverter.HalfToUInt16Bits((Half)v)
                : (ushort)(BitConverter.SingleToUInt32Bits(v) >> 16);
            raw[i] = (byte)bits; raw[i + 1] = (byte)(bits >> 8);
        }
        return raw;
    }
    rng.NextBytes(raw);
    for (long sb = 0; sb < raw.Length; sb += blockBytes)
    {
        switch (type)
        {
            case GgmlTensorType.Q4_0:
            case GgmlTensorType.Q8_0:
            case GgmlTensorType.Q5_0:
            case GgmlTensorType.IQ4_XS:
            case GgmlTensorType.IQ2_XXS:
                WriteHalf(raw, sb, 0.02f + 0.03f * (float)rng.NextDouble());
                break;
            case GgmlTensorType.Q4_K:
            case GgmlTensorType.Q5_K:
                WriteHalf(raw, sb, 0.02f + 0.03f * (float)rng.NextDouble());
                WriteHalf(raw, sb + 2, 0.01f + 0.02f * (float)rng.NextDouble());
                break;
            case GgmlTensorType.Q6_K:
            case GgmlTensorType.Q3_K:
                WriteHalf(raw, sb + blockBytes - 2, 0.01f + 0.02f * (float)rng.NextDouble());
                break;
            case GgmlTensorType.NVFP4:
                break;   // UE4M3 sub-scales: every byte decodes
            default:
                throw new NotSupportedException(type.ToString());
        }
    }
    return raw;

    static void WriteHalf(byte[] buf, long off, float val)
    {
        ushort bits = BitConverter.HalfToUInt16Bits((Half)val);
        buf[off] = (byte)bits; buf[off + 1] = (byte)(bits >> 8);
    }
}

static (string name, GgmlTensorType type)[] PerRowQuants() => new[]
{
    ("q4_0", GgmlTensorType.Q4_0),
    ("q8_0", GgmlTensorType.Q8_0),
    ("q5_0", GgmlTensorType.Q5_0),
    ("q4_k", GgmlTensorType.Q4_K),
    ("q5_k", GgmlTensorType.Q5_K),
    ("q6_k", GgmlTensorType.Q6_K),
};

// "q4k" and "q4_k" both name Q4_K (the old usage line said q4k).
static string NormalizeQuant(string name)
{
    if (name == null) return null;
    name = name.ToLowerInvariant();
    return name.Length == 3 && name[0] == 'q' ? $"{name.Substring(0, 2)}_{name[2]}" : name;
}

static void RunPerRowTable(string filter)
{
    filter = NormalizeQuant(filter);
    int inDim = 4096, outDim = 4096, iters = 200;
    Console.WriteLine($"cores={Environment.ProcessorCount}  matmul={inDim}x{outDim}  (per-row path, then the default routing)");
    Console.WriteLine($"{"quant",-6} {"rows",4}  {"GB/s",8}  {"ms/call",9}  {"relErr",9} | {"auto GB/s",9}  {"ms/call",9}  {"relErr",9}");
    foreach (var (name, type) in PerRowQuants())
    {
        if (filter != null && name != filter) continue;
        foreach (int rows in new[] { 1, 4 })
        {
            var (gbps, ms, err) = BenchPerRow(type, inDim, outDim, rows, iters, QGemmIsa.PerRow);
            var (gbpsA, msA, errA) = BenchPerRow(type, inDim, outDim, rows, iters, QGemmIsa.Auto);
            Console.WriteLine($"{name,-6} {rows,4}  {gbps,8:F1}  {ms,9:F3}  {err,9:E2} | {gbpsA,9:F1}  {msA,9:F3}  {errA,9:E2}");
        }
    }
}

static unsafe (double gbps, double msPerCall, float maxRelErr) BenchPerRow(
    GgmlTensorType type, int inDim, int outDim, int rows, int iters, QGemmIsa isa)
{
    var rng = new Random(12345 + (int)type);
    byte[] weights = BuildRandom(rng, type, outDim, inDim);
    float[] input = new float[rows * inDim];
    for (int i = 0; i < input.Length; i++) input[i] = 0.08f * MathF.Sin(i * 0.011f);
    float[] output = new float[rows * outDim];

    long weightBytes = (long)NativeDequant.RowSize((int)type, inDim) * outDim;
    fixed (byte* w = weights)
    fixed (float* x = input)
    fixed (float* o = output)
    {
        byte* wp = w;
        float* xp = x, op = o;
        bool Call() => ManagedQuantizedOps.TryAddmmQuantizedToFloat32(
            (int)type, (IntPtr)wp, inDim, outDim, xp, inDim, rows, op, outDim, null, isa);

        // correctness on a few columns against dequantize-then-dot
        float[] wrow = new float[inDim];
        float maxRel = 0f, refMag = 1e-6f;
        if (!Call()) throw new Exception($"{type}: TryAddmm returned false");
        int[] checkCols = { 0, outDim / 3, outDim / 2, outDim - 1 };
        long rowBytes = NativeDequant.RowSize((int)type, inDim);
        foreach (int c in checkCols)
        {
            NativeDequant.DequantizeToFloat32((int)type, weights, (int)(c * rowBytes), wrow, 0, inDim);
            float exp = 0f;
            for (int i = 0; i < inDim; i++) exp += wrow[i] * input[i];
            refMag = MathF.Max(refMag, MathF.Abs(exp));
            maxRel = MathF.Max(maxRel, MathF.Abs(exp - output[c]));
        }
        maxRel /= refMag;

        for (int i = 0; i < 3; i++) Call();
        var sw = Stopwatch.StartNew();
        for (int i = 0; i < iters; i++) Call();
        sw.Stop();

        double seconds = sw.Elapsed.TotalSeconds;
        double gbps = (double)weightBytes * iters / seconds / (1024.0 * 1024 * 1024);
        return (gbps, sw.Elapsed.TotalMilliseconds / iters, maxRel);
    }
}

// "dg": DiffusionGemma on a real GGUF. Structured read (Jev mode: prompt + a
// width-token canvas with one label slot, `multi` = 8 prompts for cross-backend
// agreement) or `step` (prompt + width all-mask canvas incl. the full-vocabulary
// LM head, one diffusion step of a generation), then the model's per-stage timing.
static int RunDg(string[] a)
{
    if (a.Length < 1)
    {
        Console.Error.WriteLine("usage: dg <gguf> cpu|ggml_cpu [reps] [width] [multi|step]");
        return 2;
    }
    string modelPath = a[0];
    string backendName = a.Length > 1 ? a[1] : "cpu";
    int reps = a.Length > 2 ? int.Parse(a[2]) : 2;
    int width = a.Length > 3 ? int.Parse(a[3]) : 16;
    bool multi = a.Length > 4 && a[4] == "multi";
    bool step = a.Length > 4 && a[4] == "step";
    BackendType backend = backendName switch
    {
        "cpu" => BackendType.Cpu,
        "ggml_cpu" => BackendType.GgmlCpu,
        _ => throw new ArgumentException(backendName),
    };
    using var model = (DiffusionGemmaModel)ModelBase.Create(Path.GetFullPath(modelPath), backend);
    var renderer = new GgufPromptRenderer();
    int[] labelIds = new[] { "A", "B", "C" }.Select(label => model.Tokenizer.Encode(label, false).Single()).ToArray();

    if (step)
    {
        string text0 = "Write a short poem about a small orange cat.";
        string r0 = renderer.Render(model.Config.ChatTemplate, [new ChatMessage { Role = "user", Content = text0 }],
            addGenerationPrompt: true, architecture: model.Config.Architecture);
        int[] p0 = model.Tokenizer.Encode(r0, addSpecial: true).ToArray();
        int[] toks = new int[p0.Length + width];
        p0.CopyTo(toks, 0);
        for (int i = p0.Length; i < toks.Length; i++) toks[i] = model.MaskTokenId;
        Console.WriteLine($"step mode: prompt={p0.Length} canvas={width}");
        for (int r = 0; r < reps; r++)
        {
            var sw = Stopwatch.StartNew();
            float[] logits = model.ForwardCanvas(toks, p0.Length);
            double sum = 0;
            for (int i = 0; i < logits.Length; i += 9973) sum += logits[i];
            Console.WriteLine($"step {r}: {sw.Elapsed.TotalMilliseconds:F1} ms  logits={logits.Length} probe={sum:F4}");
        }
        model.PrintForwardTiming();
        return 0;
    }

    string[] texts = multi
        ? new[]
        {
            "I love this product and would buy it again.",
            "The package arrived broken and support never answered.",
            "It works. Nothing special, nothing bad.",
            "Absolutely terrible, the worst purchase I have made this year.",
            "Great value for the price, my kids use it every day.",
            "The color is slightly off from the photos but it is fine.",
            "Stopped working after two days, asking for a refund.",
            "Five stars, fast shipping and excellent quality.",
        }
        : new[] { "I love this product and would buy it again." };

    foreach (string review in texts)
    {
        string text = $"Classify: {review}\nReturn a: A for positive, B for negative, C for neutral. Return b: A for a purchase recommendation, B otherwise.";
        string rendered = renderer.Render(model.Config.ChatTemplate, [new ChatMessage { Role = "user", Content = text }],
            addGenerationPrompt: true, architecture: model.Config.Architecture);
        int[] prompt = model.Tokenizer.Encode(rendered, addSpecial: true).ToArray();
        var template = model.Tokenizer.Encode("<|channel>thought\n<channel|>", false).ToList();
        template.AddRange(model.Tokenizer.Encode("a: ", false));
        int pos = template.Count;
        template.Add(labelIds[0]);
        template.AddRange(model.Tokenizer.Encode("\n<turn|>", false));
        var canvas = new int[width];
        template.CopyTo(canvas);
        Console.WriteLine($"prompt={prompt.Length} canvas={width}");
        for (int r = 0; r < reps; r++)
        {
            var sw = Stopwatch.StartNew();
            var res = model.ReadStructured(prompt, canvas, new[] { pos }, new[] { labelIds });
            Console.WriteLine($"rep {r}: {sw.Elapsed.TotalMilliseconds:F1} ms  p=[{string.Join(",", res[0].Select(x => x.ToString("F6")))}]");
        }
    }
    model.PrintForwardTiming();
    return 0;
}

static int RunShapes(string[] a)
{
    if (a.Length < 3)
    {
        Console.Error.WriteLine("usage: shapes TYPES ROWS SHAPES [REPS]   e.g. shapes q4_k,q6_k 1,4,512 lfm3 5");
        return 2;
    }
    var types = a[0].Split(',').Select(t => ParseShapeType(t)).ToArray();
    var rows = a[1].Split(',').Select(int.Parse).ToArray();
    var shapes = a[2] == "lfm3"
        ? new (int k, int n)[] { (2048, 7680), (7680, 2048), (2048, 6144), (2048, 2048), (2048, 512) }
        : a[2].Split(',').Select(s => { var p = s.Split('x'); return (int.Parse(p[0]), int.Parse(p[1])); }).ToArray();
    int reps = a.Length > 3 ? int.Parse(a[3]) : 5;
    Console.Error.WriteLine($"INFO shapes TS_CPU_THREADS={Environment.GetEnvironmentVariable("TS_CPU_THREADS") ?? "default"} cores={Environment.ProcessorCount}");
    foreach (var (name, type) in types)
        foreach (var (k, n) in shapes)
            foreach (int m in rows)
            {
                double[] rates = BenchShape(type, m, k, n, reps);
                Array.Sort(rates);
                string unit = m == 1 ? "gbs" : "gops";
                Console.WriteLine($"RESULT {name}-m{m}-{k}x{n}-{unit} {rates[rates.Length / 2]:F2} {rates[0]:F2} {rates[^1]:F2}");
            }
    return 0;
}

static (string name, GgmlTensorType type) ParseShapeType(string t)
{
    t = NormalizeQuant(t);
    return t switch
    {
        "q4_0" => (t, GgmlTensorType.Q4_0),
        "q8_0" => (t, GgmlTensorType.Q8_0),
        "q4_k" => (t, GgmlTensorType.Q4_K),
        "q5_k" => (t, GgmlTensorType.Q5_K),
        "q6_k" => (t, GgmlTensorType.Q6_K),
        "nvfp4" => (t, GgmlTensorType.NVFP4),
        "bf16" => (t, GgmlTensorType.BF16),
        "f16" => (t, GgmlTensorType.F16),
        _ => throw new ArgumentException($"unknown type {t}"),
    };
}

static unsafe double[] BenchShape(GgmlTensorType type, int m, int k, int n, int reps)
{
    var rng = new Random(4242 + (int)type);
    long rowBytes = NativeDequant.RowSize((int)type, k);
    long bytes = rowBytes * n;
    int copies = m == 1 ? (int)Math.Max(1, (384L << 20) / bytes) : 1;
    byte[] one = BuildRandom(rng, type, n, k);
    var buffers = new byte[copies][];
    for (int c = 0; c < copies; c++) buffers[c] = (byte[])one.Clone();
    var handles = buffers.Select(b => System.Runtime.InteropServices.GCHandle.Alloc(b, System.Runtime.InteropServices.GCHandleType.Pinned)).ToArray();
    try
    {
        float[] input = new float[(long)m * k];
        for (int i = 0; i < input.Length; i++) input[i] = (float)(rng.NextDouble() - 0.5);
        float[] output = new float[(long)m * n];
        int next = 0;
        fixed (float* x = input)
        fixed (float* o = output)
        {
            float* xp = x, op = o;
            void Call()
            {
                IntPtr w = handles[next].AddrOfPinnedObject();
                next = (next + 1) % copies;
                ManagedQuantizedOps.AddmmQuantizedToFloat32((int)type, w, k, n, xp, k, m, op, n);
            }
            Call(); Call();
            var rates = new double[reps];
            for (int r = 0; r < reps; r++)
            {
                int calls = 0;
                var sw = Stopwatch.StartNew();
                do { Call(); calls++; } while (calls < 3 || sw.Elapsed.TotalSeconds < 0.25);
                double s = sw.Elapsed.TotalSeconds / calls;
                rates[r] = m == 1 ? bytes / s / 1e9 : 2.0 * m * n * k / s / 1e9;
            }
            return rates;
        }
    }
    finally
    {
        foreach (var h in handles) h.Free();
    }
}
