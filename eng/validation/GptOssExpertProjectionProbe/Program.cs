// Copyright (c) Zhongkai Fu. Licensed under the BSD-3-Clause license.
using System.Diagnostics;
using System.Reflection;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;
using System.Security.Cryptography;
using System.Text.Json;
using TensorSharp;
using TensorSharp.GGML;
using TensorSharp.Models;
using TensorSharp.Runtime;

return Probe.Run(args);

static unsafe class Probe
{
    private const BindingFlags Private = BindingFlags.NonPublic | BindingFlags.Instance;
    private unsafe delegate void Activate(float* gate, float* up, int width);
    private static readonly Activate Activation = typeof(GptOssModel)
        .GetMethod("ApplySwiGluOaiInPlace", BindingFlags.NonPublic | BindingFlags.Static)!.CreateDelegate<Activate>();
    private static Fixture? _unreleased;

    public static int Run(string[] args)
    {
        var options = new Dictionary<string, string>();
        for (int i = 0; i < args.Length; i += 2)
        {
            if (i + 1 == args.Length || args[i] is not ("--model" or "--output" or "--layer" or "--expert" or "--tokens" or "--backend" or "--device-cache-bytes"))
                throw new ArgumentException("--model GGUF --output FRESH_DIRECTORY [--layer 0 --expert 0 --tokens 1,8,9,64,96,100,112,128,129,256 --backend cuda|cpu]");
            options.Add(args[i], args[i + 1]);
        }
        string path = Path.GetFullPath(options["--model"]), directory = Path.GetFullPath(options["--output"]);
        string backend = options.GetValueOrDefault("--backend", "cuda");
        int layer = int.Parse(options.GetValueOrDefault("--layer", "0"));
        int expert = int.Parse(options.GetValueOrDefault("--expert", "0"));
        long cacheBytes = long.Parse(options.GetValueOrDefault("--device-cache-bytes", "1"));
        int[] counts = options.GetValueOrDefault("--tokens", "1,8,9,64,96,100,112,128,129,256").Split(',').Select(int.Parse).ToArray();
        if (backend is not ("cpu" or "cuda") || layer < 0 || expert < 0 || counts.Length == 0 || cacheBytes < 1
            || counts.Any(n => n is < 1 or > 1025) || counts.Distinct().Count() != counts.Length)
            throw new ArgumentException("Invalid backend/layer/expert/token bounds.");
        if (Directory.Exists(directory)) throw new IOException("Use a fresh output directory.");
        Directory.CreateDirectory(directory);
        var cases = new List<object>();
        Fixture? fixture = null;
        object? weights = null, native = null;
        string? failure = null;
        bool cleanup = false, exact = true, temporaryExact = true;
        try
        {
            using var gguf = GgufFile.OpenForMemoryPlanning(path);
            string prefix = $"blk.{layer}.ffn_";
            if (gguf.Tensors.Keys.Any(name => name.StartsWith(prefix, StringComparison.Ordinal) && name.Contains("_exps.scale", StringComparison.Ordinal)))
                throw new NotSupportedException("Weight scale sidecars require a separate explicit oracle.");
            Weight gate = ReadWeight(gguf, prefix + "gate_exps.weight", expert);
            Weight up = ReadWeight(gguf, prefix + "up_exps.weight", expert);
            Weight down = ReadWeight(gguf, prefix + "down_exps.weight", expert);
            if (gate.Type != up.Type || gate.K != up.K || gate.M != up.M || down.K != gate.M || down.M != gate.K)
                throw new InvalidDataException("Expected losslessly concatenable gate/up and matching down geometry.");
            float[] gb = ReadBias(gguf, prefix + "gate_exps.bias", expert, gate.M);
            float[] ub = ReadBias(gguf, prefix + "up_exps.bias", expert, up.M);
            float[] db = ReadBias(gguf, prefix + "down_exps.bias", expert, down.M);
            fixture = Fixture.Create(backend, gate, up, down, gb, ub, db, cacheBytes);
            int alternateExpert = (expert + 1) % checked((int)gguf.Tensors[prefix + "gate_exps.weight"].Shape[2]);
            Weight alternateGate = ReadWeight(gguf, prefix + "gate_exps.weight", alternateExpert);
            Weight alternateUp = ReadWeight(gguf, prefix + "up_exps.weight", alternateExpert);
            Weight alternateDown = ReadWeight(gguf, prefix + "down_exps.weight", alternateExpert);
            fixture.AddExpert("alternate.", alternateGate, alternateUp, alternateDown,
                ReadBias(gguf, prefix + "gate_exps.bias", alternateExpert, alternateGate.M),
                ReadBias(gguf, prefix + "up_exps.bias", alternateExpert, alternateUp.M),
                ReadBias(gguf, prefix + "down_exps.bias", alternateExpert, alternateDown.M));
            native = NativeIdentity();
            weights = new[] { gate, up, down, alternateGate, alternateUp, alternateDown }.Select(w => new { w.Name, w.Type, w.K, w.M,
                raw_bytes = w.Raw.Length, raw_sha256 = Hash(w.Raw), decode = "Independent local C# scalar decoder, all elements bitwise checked against native ggml to_float." }).ToArray();
            foreach (int n in counts)
            {
                float[] x = Inputs(n, gate.K);
                using var input = fixture.Tensor(x, n, gate.K);
                float[] g = fixture.Project(input, "gate"), u = fixture.Project(input, "up");
                float[] gu = fixture.Project(input, "fused");
                float[] fg = Slice(gu, n, gate.M * 2, 0, gate.M), fu = Slice(gu, n, gate.M * 2, gate.M, gate.M);
                float[] bg = fixture.Biased(input, "gate"), bu = fixture.Biased(input, "up");
                float[] bgu = fixture.Biased(input, "fused");
                float[] bfg = Slice(bgu, n, gate.M * 2, 0, gate.M), bfu = Slice(bgu, n, gate.M * 2, gate.M, gate.M);
                float[] activated = ActivateRows(bg, bu, n, gate.M), fusedActivated = ActivateRows(bfg, bfu, n, gate.M);
                using var activatedTensor = fixture.Tensor(activated, n, gate.M);
                using var fusedActivatedTensor = fixture.Tensor(fusedActivated, n, gate.M);
                float[] staged = fixture.Biased(activatedTensor, "down"), fusedStaged = fixture.Biased(fusedActivatedTensor, "down");
                float[] splitResult = fixture.Expert(input, n, split: true), fusedResult = fixture.Expert(input, n, split: false);
                float[] temporaryResult = fixture.TemporaryExpert(input, n);
                float[] alternateFused = fixture.Expert(input, n, split: false, prefix: "alternate.");
                float[] alternateTemporary = fixture.TemporaryExpert(input, n, "alternate.");
                float[] repeatedTemporary = fixture.TemporaryExpert(input, n);
                var stages = new Dictionary<string, object>
                {
                    ["raw_gate"] = Compare(fg, g), ["raw_up"] = Compare(fu, u),
                    ["biased_gate"] = Compare(bfg, bg), ["biased_up"] = Compare(bfu, bu),
                    ["activation"] = Compare(fusedActivated, activated), ["staged_down"] = Compare(fusedStaged, staged),
                    ["production_fused_vs_staged"] = Compare(fusedStaged, fusedResult),
                    ["production_split_vs_staged"] = Compare(staged, splitResult),
                    ["production_fused_vs_split"] = Compare(fusedResult, splitResult),
                    ["production_fused_vs_temporary"] = Compare(fusedResult, temporaryResult),
                    ["alternate_fused_vs_temporary"] = Compare(alternateFused, alternateTemporary),
                    ["first_expert_after_alternating_temporary"] = Compare(fusedResult, repeatedTemporary)
                };
                int[] tokens = n == 1 ? [0] : [0, n - 1];
                int[] rows = Samples(gate.M, 32), outputs = Samples(down.M, 32);
                double[] gateOracle = DotSample(gate, x, tokens, rows), upOracle = DotSample(up, x, tokens, rows);
                double[] fullOracle = ExpertOracle(gate, up, down, gb, ub, db, x, tokens, outputs);
                object oracle = new
                {
                    tokens, projection_rows = rows, output_rows = outputs,
                    raw_gate_fused = Compare(gateOracle, Select(fg, gate.M, tokens, rows)),
                    raw_gate_split = Compare(gateOracle, Select(g, gate.M, tokens, rows)),
                    raw_up_fused = Compare(upOracle, Select(fu, gate.M, tokens, rows)),
                    raw_up_split = Compare(upOracle, Select(u, gate.M, tokens, rows)),
                    full_fused = Compare(fullOracle, Select(fusedResult, down.M, tokens, outputs)),
                    full_split = Compare(fullOracle, Select(splitResult, down.M, tokens, outputs))
                };
                bool equal = Bytes(fusedResult).SequenceEqual(Bytes(splitResult));
                exact &= equal;
                bool temporaryEqual = Bytes(fusedResult).SequenceEqual(Bytes(temporaryResult))
                    && Bytes(alternateFused).SequenceEqual(Bytes(alternateTemporary))
                    && Bytes(fusedResult).SequenceEqual(Bytes(repeatedTemporary));
                temporaryExact &= temporaryEqual;
                File.WriteAllBytes(Path.Combine(directory, $"n{n}-fused.f32"), Bytes(fusedResult).ToArray());
                File.WriteAllBytes(Path.Combine(directory, $"n{n}-split.f32"), Bytes(splitResult).ToArray());
                File.WriteAllBytes(Path.Combine(directory, $"n{n}-temporary.f32"), Bytes(temporaryResult).ToArray());
                File.WriteAllBytes(Path.Combine(directory, $"n{n}-oracle.f64"), MemoryMarshal.AsBytes(fullOracle.AsSpan()).ToArray());
                var row = new { tokens = n, fused_rows = gate.M * 2, split_rows = gate.M, stages, oracle, exact = equal,
                    temporary_exact = temporaryEqual, alternate_expert = alternateExpert };
                cases.Add(row);
                Console.WriteLine(JsonSerializer.Serialize(row));
            }
        }
        catch (Exception error) { failure = error.ToString(); Console.Error.WriteLine(error); }
        finally
        {
            try { fixture?.Dispose(); cleanup = _unreleased == null; }
            catch (Exception error) { _unreleased = fixture; failure = (failure ?? "") + "\nCleanup retained: " + error; }
        }
        File.WriteAllText(Path.Combine(directory, "report.json"), JsonSerializer.Serialize(new
        {
            completed = failure == null && cleanup, all_fused_split_exact = exact && cases.Count == counts.Length,
            all_fused_temporary_exact = temporaryExact && cases.Count == counts.Length,
            failure, cleanup, backend, device_cache_bytes = cacheBytes, model = path, layer, expert, weights, native, cases,
            input = "Deterministic synthetic F32 inputs; not captured model activations. Original expert K/M retained; one primary and the next wrapping alternate expert loaded.",
            scope = "No relaxed numerical pass threshold. Full raw/biased/activation/down arrays compare fused and split; sampled FP64 oracle uses independently decoded weights, exact source biases, scalar double dots and exact double exp with float-valued alpha 1.702f. Production uses its existing SIMD exp approximation and CUDA activation quantization. This isolates one expert, not whole-model quality or dispatch performance.",
            upstream = "Read-only decoder reference: ggml-quants.c dequantize_row_mxfp4, ggml-impl.h ggml_e8m0_to_fp32_half, ggml-common.h kvalues_fp4; pin ffa4e8b80930029a35991f94e7c8a93cd67730ab."
        }, new JsonSerializerOptions { WriteIndented = true }));
        return failure == null && cleanup ? 0 : 1;
    }

    private static Weight ReadWeight(GgufFile file, string name, int expert)
    {
        var t = file.Tensors[name];
        if (t.Shape.Length != 3 || t.Type is not (GgmlTensorType.Q8_0 or GgmlTensorType.MXFP4))
            throw new NotSupportedException($"{name}: expected Q8_0/MXFP4 [K,M,E], actual {t.Type}.");
        int k = checked((int)t.Shape[0]), m = checked((int)t.Shape[1]), e = checked((int)t.Shape[2]);
        if (k <= 0 || k % 32 != 0 || m <= 0 || expert >= e) throw new InvalidDataException("Invalid expert geometry.");
        int bytes = checked(k / 32 * ((int)t.Type == 39 ? 17 : 34) * m);
        if (bytes > 64 << 20) throw new NotSupportedException("Diagnostic bounds each selected expert tensor at 64 MiB.");
        var region = file.GetTensorFileRegion(name);
        if (region.ByteLength != checked((long)bytes * e)) throw new InvalidDataException("Unexpected tensor byte count.");
        byte[] raw = Read(region.Path, checked(region.Offset + (long)expert * bytes), bytes);
        return new(name, (int)t.Type, k, m, raw, Decode((int)t.Type, raw, checked(k * m)));
    }

    private static float[] ReadBias(GgufFile file, string name, int expert, int width)
    {
        if (!file.Tensors.TryGetValue(name, out var t)) return new float[width];
        if (t.Type != GgmlTensorType.F32 || t.Shape.Length != 2 || t.Shape[0] != (ulong)width || expert >= checked((int)t.Shape[1]))
            throw new InvalidDataException("Expected F32 [width,experts] bias: " + name);
        var region = file.GetTensorFileRegion(name);
        if (region.ByteLength != checked((long)width * checked((long)t.Shape[1]) * 4)) throw new InvalidDataException("Unexpected bias byte count.");
        var result = MemoryMarshal.Cast<byte, float>(Read(region.Path, checked(region.Offset + (long)expert * width * 4), checked(width * 4)).AsSpan()).ToArray();
        if (result.Any(value => !float.IsFinite(value))) throw new InvalidDataException("Nonfinite source bias.");
        return result;
    }

    private static byte[] Read(string path, long offset, int bytes)
    {
        byte[] data = new byte[bytes];
        using var handle = File.OpenHandle(path, FileMode.Open, FileAccess.Read, FileShare.Read);
        for (int done = 0; done < bytes;)
        {
            int count = RandomAccess.Read(handle, data.AsSpan(done), checked(offset + done));
            if (count == 0) throw new EndOfStreamException(path);
            done += count;
        }
        return data;
    }

    private static float[] Decode(int type, byte[] raw, int elements)
    {
        int[] fp4 = [0, 1, 2, 3, 4, 6, 8, 12, 0, -1, -2, -3, -4, -6, -8, -12];
        float[] decoded = new float[elements];
        for (int block = 0; block < elements / 32; block++)
        {
            int offset = block * (type == 39 ? 17 : 34);
            if (type == 39)
            {
                int exponent = raw[offset];
                if (exponent == 255) throw new InvalidDataException("MXFP4 E8M0 NaN scale.");
                float scale = BitConverter.Int32BitsToSingle(exponent < 2 ? 0x00200000 << exponent : (exponent - 1) << 23);
                for (int j = 0; j < 16; j++)
                {
                    byte packed = raw[offset + 1 + j];
                    decoded[block * 32 + j] = scale * fp4[packed & 15];
                    decoded[block * 32 + 16 + j] = scale * fp4[packed >> 4];
                }
            }
            else
            {
                float scale = (float)BitConverter.UInt16BitsToHalf(BitConverter.ToUInt16(raw, offset));
                for (int j = 0; j < 32; j++) decoded[block * 32 + j] = scale * unchecked((sbyte)raw[offset + 2 + j]);
            }
        }
        if (decoded.Any(v => !float.IsFinite(v))) throw new InvalidDataException("Nonfinite decoded weight.");
        return decoded;
    }

    private static float[] Inputs(int n, int k) => Enumerable.Range(0, checked(n * k))
        .Select(i => (float)(0.43 * Math.Sin(i * 0.031) + 0.27 * Math.Cos(i * 0.013 + i / k))).ToArray();
    private static int[] Samples(int length, int count) => length == 1 ? [0] : Enumerable.Range(0, Math.Min(length, count))
        .Select(i => (int)((long)i * (length - 1) / (Math.Min(length, count) - 1))).ToArray();
    private static float[] Slice(float[] source, int n, int stride, int offset, int width)
    { float[] result = new float[n * width]; for (int t = 0; t < n; t++) Array.Copy(source, t * stride + offset, result, t * width, width); return result; }
    private static float[] Select(float[] source, int width, int[] tokens, int[] rows) =>
        tokens.SelectMany(t => rows.Select(r => source[t * width + r])).ToArray();
    private static float[] ActivateRows(float[] gate, float[] up, int n, int width)
    {
        var result = (float[])gate.Clone();
        fixed (float* g = result, u = up) for (int t = 0; t < n; t++) Activation(g + t * width, u + t * width, width);
        return result;
    }
    private static double Dot(Weight w, int row, float[] x, int offset)
    { double sum = 0; for (int k = 0; k < w.K; k++) sum += (double)w.Decoded[row * w.K + k] * x[offset + k]; return sum; }
    private static double[] DotSample(Weight w, float[] x, int[] tokens, int[] rows) =>
        tokens.SelectMany(t => rows.Select(r => Dot(w, r, x, t * w.K))).ToArray();
    private static double[] ExpertOracle(Weight gate, Weight up, Weight down, float[] gb, float[] ub, float[] db,
        float[] x, int[] tokens, int[] rows)
    {
        double[] result = new double[tokens.Length * rows.Length];
        for (int i = 0; i < tokens.Length; i++)
        {
            double[] act = new double[gate.M];
            for (int row = 0; row < gate.M; row++)
            {
                double g = Math.Min(7, Dot(gate, row, x, tokens[i] * gate.K) + gb[row]);
                double u = Math.Clamp(Dot(up, row, x, tokens[i] * up.K) + ub[row], -7, 7);
                act[row] = g / (1 + Math.Exp(-(double)1.702f * g)) * (u + 1);
            }
            for (int j = 0; j < rows.Length; j++)
            {
                double sum = db[rows[j]];
                for (int k = 0; k < down.K; k++) sum += down.Decoded[rows[j] * down.K + k] * act[k];
                result[i * rows.Length + j] = sum;
            }
        }
        return result;
    }
    private static ReadOnlySpan<byte> Bytes(float[] x) => MemoryMarshal.AsBytes(x.AsSpan());
    private static string Hash(byte[] x) => Convert.ToHexString(SHA256.HashData(x)).ToLowerInvariant();
    private static object Compare(float[] a, float[] b) => new { bit_exact = Bytes(a).SequenceEqual(Bytes(b)), metrics = Compare(a.Select(x => (double)x).ToArray(), b) };
    private static object Compare(double[] a, float[] b)
    {
        if (a.Length != b.Length || a.Length == 0 || a.Any(x => !double.IsFinite(x)) || b.Any(x => !float.IsFinite(x)))
            throw new InvalidDataException("Nonfinite or inconsistent comparison.");
        double error = 0, norm = 0, actual = 0, dot = 0, max = 0;
        for (int i = 0; i < a.Length; i++) { double d = a[i] - b[i]; error += d * d; norm += a[i] * a[i]; actual += (double)b[i] * b[i]; dot += a[i] * b[i]; max = Math.Max(max, Math.Abs(d)); }
        return new { elements = a.Length, max_abs = max, rms = Math.Sqrt(error / a.Length), rel_l2 = norm == 0 ? (double?)null : Math.Sqrt(error / norm),
            cosine = norm == 0 || actual == 0 ? (double?)null : dot / Math.Sqrt(norm * actual) };
    }
    private static object NativeIdentity()
    {
        using var process = Process.GetCurrentProcess();
        return process.Modules.Cast<ProcessModule>().Where(m => m.ModuleName.Contains("GgmlOps", StringComparison.OrdinalIgnoreCase))
            .Select(m => new { path = m.FileName, sha256 = Hash(File.ReadAllBytes(m.FileName)) }).ToArray();
    }
    private sealed record Weight(string Name, int Type, int K, int M, byte[] Raw, float[] Decoded);

    private sealed class Fixture : IDisposable
    {
        private readonly Dictionary<string, QuantizedWeight> _quant = new();
        private readonly Dictionary<string, Tensor> _bias = new();
        private readonly GptOssModel _model = (GptOssModel)RuntimeHelpers.GetUninitializedObject(typeof(GptOssModel));
        private GgmlContext? _context;
        private GgmlAllocator _allocator = null!;
        private Fixture() { GC.SuppressFinalize(_model); }
        internal static Fixture Create(string backend, Weight gate, Weight up, Weight down, float[] gb, float[] ub, float[] db, long cacheBytes)
        {
            var f = new Fixture();
            try
            {
                f._context = new GgmlContext([0], backend == "cuda" ? GgmlBackendType.Cuda : GgmlBackendType.Cpu);
                f._allocator = new GgmlAllocator(f._context, 0);
                GgmlBasicOps.SetDeviceCopyBudget(cacheBytes);
                foreach (var w in new[] { gate, up, down })
                {
                    float[] native = new float[w.Decoded.Length];
                    GgmlGgufTensorDequant.DequantizeToFloat32(w.Type, w.Raw, 0, native, 0, native.Length);
                    if (!Bytes(native).SequenceEqual(Bytes(w.Decoded))) throw new InvalidDataException("Independent/native weight decode differs: " + w.Name);
                }
                foreach (var (name, weight) in new[] { ("gate", gate), ("up", up), ("down", down) })
                    f._quant.Add(name, new QuantizedWeight(weight.Raw, weight.Type, weight.K, weight.M));
                f._quant.Add("fused", new QuantizedWeight(gate.Raw.Concat(up.Raw).ToArray(), gate.Type, gate.K, checked(gate.M * 2)));
                f._bias.Add("gate.bias", f.Tensor(gb, 1, gb.Length)); f._bias.Add("up.bias", f.Tensor(ub, 1, ub.Length));
                f._bias.Add("down.bias", f.Tensor(db, 1, db.Length)); f._bias.Add("fused.bias", f.Tensor(gb.Concat(ub).ToArray(), 1, gb.Length + ub.Length));
                void Set(string name, object value) => typeof(ModelBase).GetField(name, Private)!.SetValue(f._model, value);
                var type = backend == "cuda" ? BackendType.GgmlCuda : BackendType.GgmlCpu;
                Set("_backend", type); Set("_allocator", f._allocator); Set("_quantWeights", f._quant); Set("_weights", f._bias);
                Set("<ExecutionPlan>k__BackingField", Activator.CreateInstance(
                    typeof(ModelBase).Assembly.GetType("TensorSharp.Models.BackendExecutionPlan", true)!, [type])!);
                return f;
            }
            catch { try { f.Dispose(); } catch { _unreleased = f; } throw; }
        }
        internal void AddExpert(string prefix, Weight gate, Weight up, Weight down, float[] gb, float[] ub, float[] db)
        {
            foreach (var (name, weight) in new[] { ("gate", gate), ("up", up), ("down", down) })
            {
                float[] native = new float[weight.Decoded.Length];
                GgmlGgufTensorDequant.DequantizeToFloat32(weight.Type, weight.Raw, 0, native, 0, native.Length);
                if (!Bytes(native).SequenceEqual(Bytes(weight.Decoded))) throw new InvalidDataException("Independent/native alternate weight decode differs: " + weight.Name);
                _quant.Add(prefix + name, new QuantizedWeight(weight.Raw, weight.Type, weight.K, weight.M));
            }
            _quant.Add(prefix + "fused", new QuantizedWeight(gate.Raw.Concat(up.Raw).ToArray(), gate.Type, gate.K, checked(gate.M * 2)));
            _bias.Add(prefix + "gate.bias", Tensor(gb, 1, gb.Length)); _bias.Add(prefix + "up.bias", Tensor(ub, 1, ub.Length));
            _bias.Add(prefix + "down.bias", Tensor(db, 1, db.Length));
            _bias.Add(prefix + "fused.bias", Tensor(gb.Concat(ub).ToArray(), 1, gb.Length + ub.Length));
        }
        internal Tensor Tensor(float[] data, int rows, int columns)
        { var t = new Tensor(_allocator, DType.Float32, rows, columns); try { t.SetElementsAsFloat(data); return t; } catch { t.Dispose(); throw; } }
        internal float[] Project(Tensor input, string name)
        {
            var w = _quant[name]; using var output = new Tensor(_allocator, DType.Float32, input.Sizes[0], w.Ne1);
            GgmlBasicOps.AddmmQuant(output, input, w.CacheKey, w.GgmlType, w.Ne0, w.Ne1, w.RawBytes);
            return output.GetElementsAsFloat(checked((int)output.ElementCount()));
        }
        internal float[] Biased(Tensor input, string name)
        {
            using var t = (Tensor)typeof(GptOssModel).GetMethod("LinearForwardWithBias", Private)!.Invoke(_model, [input, name, name + ".bias"])!;
            return t.GetElementsAsFloat(checked((int)t.ElementCount()));
        }
        internal float[] Expert(Tensor input, int n, bool split, string prefix = "")
        {
            using var t = (Tensor)typeof(GptOssModel).GetMethod(split ? "ExpertFFNSplit" : "ExpertFFN", Private)!.Invoke(_model,
                split ? [input, prefix + "gate", prefix + "gate.bias", prefix + "up", prefix + "up.bias", prefix + "down", prefix + "down.bias", n]
                    : [input, prefix + "fused", prefix + "fused.bias", prefix + "down", prefix + "down.bias", n])!;
            return t.GetElementsAsFloat(checked((int)t.ElementCount()));
        }
        internal float[] TemporaryExpert(Tensor input, int n, string prefix = "")
        {
            using var t = (Tensor)typeof(GptOssModel).GetMethod("ExpertFFNTemporary", Private)!.Invoke(_model,
                [input, prefix + "gate", prefix + "gate.bias", prefix + "up", prefix + "up.bias", prefix + "down", prefix + "down.bias", n])!;
            if (typeof(GptOssModel).GetField("_temporaryExpertGateUpOwner", Private)!.GetValue(_model) != null)
                throw new InvalidOperationException("Temporary expert owner survived successful projection.");
            return t.GetElementsAsFloat(checked((int)t.ElementCount()));
        }
        public void Dispose()
        {
            if (_context == null) return;
            typeof(GptOssModel).GetMethod("ReleaseTemporaryExpertGateUp", Private)!.Invoke(_model, null);
            GgmlBasicOps.HostReadBarrier(); GgmlBasicOps.ClearHostBufferCache(); GgmlBasicOps.ReleaseReuseComputeBuffers();
            foreach (var q in _quant.Values) q.Dispose(); _quant.Clear();
            foreach (var b in _bias.Values) b.Dispose(); _bias.Clear();
            _context.ReleasePoolForDisposal(); GgmlBasicOps.Shutdown(); _context = null;
        }
    }
}
